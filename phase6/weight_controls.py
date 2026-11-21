"""Weight-space controls for evaluation-only counterfactuals, and realized displacement.

* random_direction_state  norm-matched random-direction replacement: every trained matrix change
  dW = W_r - W_0 (per expert slice for stacked banks) is replaced by U' diag(s) V'^T with the SAME
  singular values s and Haar-random singular frames; vectors keep their norm; untouched tensors and
  frozen rows (dW = 0) stay exactly as they are. sign=-1 gives the antithetic partner theta_0 - r of the
  same draw (antithetic pairs (delta_rand, -delta_rand)).
* reflect_state           theta_0 - delta with delta = adapted - base (F_sym = (F(theta_0 + delta)
  + F(theta_0 - delta)) / 2); tensors and rows with delta = 0 (frozen) stay bitwise identical.
* wd_only_state           the decoupled-weight-decay-only trajectory of a run: the authors' optimizer
  and scheduler, built exactly as the run built them (same experiment, env, subset spec, frozen
  routers, optimizer arm), stepped max_steps times with ZERO gradients. Muon's direction is then 0
  (NS(0) = 0) and AdamW's update is 0, so only the decay acts -- through the same code path, including
  subset_adapt's masked decay of row-frozen banks.
* revert_state            C3 revert attribution: adapted weights with named parameter groups reverted
  to the base model.
* displacement            realized ||theta - theta_0||^2 per unit kind and per expert unit
  (the 'displacement form' of every time statement).
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from typing import Any, Iterable

import torch
from torch import Tensor

# ============================================================================= random direction
def _haar_frame(m: int, r: int, gen: torch.Generator) -> Tensor:
    a = torch.randn(m, r, generator=gen, dtype=torch.float64)
    q, rr = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(rr))
    sign[sign == 0] = 1.0
    return q * sign          # Haar-distributed orthonormal columns


def sv_matched_random(delta: Tensor, gen: torch.Generator) -> Tensor:
    """A random 2-D matrix with exactly the singular values of `delta` (float64 SVD on CPU)."""

    d = delta.detach().to("cpu", torch.float64)
    s = torch.linalg.svdvals(d)
    r = s.numel()
    u = _haar_frame(d.shape[0], r, gen)
    v = _haar_frame(d.shape[1], r, gen)
    return ((u * s) @ v.T).to(delta.dtype).to(delta.device)


def _gen_for(name: str, seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed((int(seed) * 1_000_003 + zlib.crc32(name.encode())) % (2 ** 62))
    return g


def random_direction_state(base_state: dict, adapted_state: dict, seed: int = 0,
                           report: dict | None = None, sign: int = 1) -> dict:
    """theta_0 + sign x (singular-value-matched random replacement of every dW); sign=-1 = antithetic partner."""

    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")

    out = dict(adapted_state)
    for name, a in adapted_state.items():
        b = base_state.get(name)
        if b is None or not torch.is_floating_point(a) or a.shape != b.shape:
            continue
        d = a.float() - b.float()
        if not bool((d != 0).any()):
            continue
        gen = _gen_for(name, seed)
        if d.ndim == 1:
            r = torch.randn(d.shape, generator=gen, dtype=torch.float64)
            nd = (r * (d.double().norm() / r.norm())).to(d.dtype)
        elif d.ndim == 2:
            nd = sv_matched_random(d, gen)
        else:
            flat = d.reshape(d.shape[0], -1, d.shape[-1]) if d.ndim > 3 else d
            nd = torch.zeros_like(flat)
            for i in range(flat.shape[0]):
                if bool((flat[i] != 0).any()):
                    nd[i] = sv_matched_random(flat[i], gen)
            nd = nd.reshape(d.shape)
        out[name] = (b.float() + sign * nd.to(b.device)).to(a.dtype)
        if report is not None:
            report[name] = {"fro_dW": float(d.norm()), "fro_random": float(nd.norm())}
    return out


def reflect_state(base_state: dict, adapted_state: dict, report: dict | None = None) -> dict:
    """theta_0 - delta = 2 theta_0 - theta_adapted for every floating tensor that changed; unchanged tensors (and,
    within changed tensors, entries with delta = 0) are returned bitwise identical (b - 0 = b exactly)."""

    out = dict(adapted_state)
    for name, a in adapted_state.items():
        b = base_state.get(name)
        if b is None or not torch.is_floating_point(a) or a.shape != b.shape:
            continue
        if torch.equal(a, b.to(a.device, a.dtype)):
            out[name] = b.to(a.device, a.dtype).clone()
            continue
        bf = b.to(a.device).float()
        d = a.float() - bf
        r = bf - d
        same = d == 0
        out[name] = torch.where(same, b.to(a.device, a.dtype), r.to(a.dtype))
        if report is not None:
            report[name] = {"fro_delta": float(d.norm()), "changed_frac": float((~same).float().mean())}
    return out


# ============================================================================= revert attribution
GROUP_DOC = {
    "embed_head": "input embedding + LM head",
    "pub": "Nestor public (shared) experts, all blocks",
    "priv": "Nestor private experts of the routed blocks (dense tail excluded)",
    "attn": "attention projections of the routed blocks (Nestor columns) / all attention (MoE, dense)",
    "shared": "norm gains + depth-read queries/gains + path gates + the dense tail's attention and private experts",
    "moe": "MoE / dense bank experts",
    "router": "router score/noise networks",
    "<group>@<scope>": "the group restricted to blocks/layers: @b3 or @3 (block 3), @l2 (layer 2 of every block), "
                       "@b1l2 (layer 2 of block 1), @0/1/2 (several, '/' = or); C3 per-layer reverts (prereg v1.3 "
                       "sec. 4.8): revert:moe@b<b>l<l> (MoE-100M: 4 blocks x 3 layers = 12 single-layer reverts), "
                       "revert:priv@b<b> (NESTOR-100M: 6 single-block reverts of the column-private group, blocks 0-5)",
}


_SCOPE_PART = re.compile(r"^(?:b?(\d+))?(?:l(\d+))?$")


def _scope(spec: str) -> list[tuple[tuple[str, int], ...]]:
    """Revert scope (prereg v1.3 sec. 4.8 per-layer / per-block reverts) as a disjunction of conjunctions.

    Parts separated by '/' are ORed; a part is 'b<i>' or '<i>' (block i), 'l<j>' (layer j of every block) or
    'b<i>l<j>' (layer j of block i, i.e. one routed stratum 'moe:b<i>:l<j>').

    Raises:
        ValueError: on an empty or malformed part (a typo must not silently revert nothing or everything).
    """
    out = []
    for part in spec.split("/"):
        m = _SCOPE_PART.match(part)
        if not part or m is None or (m.group(1) is None and m.group(2) is None):
            raise ValueError(f"bad revert scope {part!r} in {spec!r} (b<i>, <i>, l<j> or b<i>l<j>; '/' = or)")
        conj = []
        if m.group(1) is not None:
            conj.append(("block", int(m.group(1))))
        if m.group(2) is not None:
            conj.append(("layer", int(m.group(2))))
        out.append(tuple(conj))
    return out


def _group_pred(group: str, tail: int | None):
    if "@" in group:                      # '<group>@<scope>': the group restricted to transformer blocks / layers
        g, spec = group.split("@", 1)
        base = _group_pred(g, tail)
        sc = _scope(spec)
        return lambda u: base(u) and any(all(getattr(u, f) == v for f, v in conj) for conj in sc)
    if group.startswith("kind:"):
        k = group[5:]
        return lambda u: u.kind == k
    if group == "embed_head":
        return lambda u: u.kind in ("embed", "head")
    if group == "pub":
        return lambda u: u.kind == "pub"
    if group == "priv":
        return lambda u: u.kind == "priv" and u.block != tail
    if group == "attn":
        return lambda u: u.kind == "attn" and u.block != tail
    if group == "shared":
        return lambda u: u.kind in ("norm", "depth_read", "path_gate") or (u.block == tail and u.kind in ("attn", "priv"))
    if group == "moe":
        return lambda u: u.kind == "moe"
    if group == "router":
        return lambda u: u.kind == "router"
    raise ValueError(f"unknown revert group {group!r} (known: {sorted(GROUP_DOC)} or kind:<unit kind>)")


def revert_state(model, adapted_state: dict, base_state: dict, groups: Iterable[str]) -> dict:
    """Adapted state with every tensor / bank row of the named groups set back to the base state."""

    import subset_adapt as SA

    entries = SA.build_catalog(model)
    tail = SA.model_geometry(model)["dense_tail_block"]
    preds = [_group_pred(g, tail) for g in groups]
    out = dict(adapted_state)
    for e in entries:
        flags = [any(p(u) for p in preds) for u in e.units]
        if not any(flags):
            continue
        if all(flags):
            out[e.name] = base_state[e.name].clone()
        else:
            t = adapted_state[e.name].clone()
            rows = torch.tensor(flags)
            t[rows] = base_state[e.name][rows].to(t.dtype)
            out[e.name] = t
    return out


def group_partition(model, groups: Iterable[str]) -> dict[str, int]:
    """Elements per group and how many elements fall in 0 or > 1 groups (a partition check)."""

    import subset_adapt as SA

    entries = SA.build_catalog(model)
    tail = SA.model_geometry(model)["dense_tail_block"]
    preds = {g: _group_pred(g, tail) for g in groups}
    cnt = Counter()
    for e in entries:
        per = e.param[0].numel() if e.sliced else e.param.numel()
        for u in e.units:
            hits = [g for g, p in preds.items() if p(u)]
            for g in hits:
                cnt[g] += per
            cnt["_none" if not hits else ("_multi" if len(hits) > 1 else "_one")] += per
    return dict(cnt)


# ============================================================================= WD-only control
@contextlib.contextmanager
def temp_env(env: dict):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


WD_STEP_KEYS = ("DCLM_MAX_STEPS", "DCLM_WARMUP_STEPS")


def _interim_stop(run_env: dict) -> int:
    raw = run_env.get("NESTOR_INTERIM_EVAL")
    if not raw:
        return 0
    try:
        return int((json.loads(raw) if isinstance(raw, str) else raw).get("stop_after") or 0)
    except (ValueError, AttributeError):
        raise ValueError("wd_only_schedule: NESTOR_INTERIM_EVAL in run_env is not readable JSON")


def _frozen_queue_job(run_env: dict, manifest_path: str | None, queue_path: str | None) -> tuple[dict | None, str | None]:
    """The run's job in the frozen queue: FREEZE.json above the run directory names the queue; job id = DCLM_RUN_SUFFIX."""
    rid = run_env.get("DCLM_RUN_SUFFIX")
    if not rid:
        return None, None
    qp = queue_path
    if qp is None and manifest_path:
        d = os.path.dirname(os.path.abspath(manifest_path))
        for _ in range(8):
            fz = os.path.join(d, "FREEZE.json")
            if os.path.isfile(fz):
                ref = (json.load(open(fz, encoding="utf-8-sig")).get("queue") or {}).get("path")
                qp = os.path.join(d, ref) if ref else None
                break
            nd = os.path.dirname(d)
            if nd == d:
                break
            d = nd
    if not qp or not os.path.isfile(qp):
        return None, None
    job = next((j for j in json.load(open(qp, encoding="utf-8-sig")).get("jobs") or [] if j.get("id") == rid), None)
    if job is not None:
        js = (job.get("env") or {}).get("DCLM_SEED")
        if js is not None and run_env.get("DCLM_SEED") is not None and str(js) != str(run_env["DCLM_SEED"]):
            raise ValueError(f"wd_only_schedule: queue job {rid} has DCLM_SEED {js}, the run used {run_env['DCLM_SEED']}")
    return job, qp


TRAINING_QUEUE_GLOBS = ("queue_*.json", os.path.join("*", "queue_*.json"))


def _training_queue_job(run_env: dict, manifest_path: str | None) -> tuple[dict | None, str | None, list[str]]:
    """Amendment A-2: the job of the queue that trained a run which is not a job of the frozen queue (the C3 revert
    evaluations of pre-freeze G1 runs). A queue matches when its run_queue state (the queue's "state" file) records this
    run's folder as the root of the job DCLM_RUN_SUFFIX with status done, and the job's DCLM_SEED equals the run's.
    Candidates: queue_*.json in the project root (the folder of FREEZE.json above the run) and one folder below. Returns
    (job, matching queue paths relative to the root, rejected candidates); several matching queues must agree."""
    import glob

    rid = run_env.get("DCLM_RUN_SUFFIX")
    if not rid or not manifest_path:
        return None, None, []
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    root, d = None, run_dir
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "FREEZE.json")):
            root = d
            break
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    if root is None:
        return None, None, []
    matches, rejected = [], []
    for pat in TRAINING_QUEUE_GLOBS:
        for qp in sorted(glob.glob(os.path.join(root, pat))):
            try:
                qd = json.load(open(qp, encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if not isinstance(qd, dict) or not isinstance(qd.get("jobs"), list) or not qd.get("state"):
                continue
            job = next((j for j in qd["jobs"] if isinstance(j, dict) and j.get("id") == rid), None)
            if job is None:
                continue
            rel = os.path.relpath(qp, root).replace("\\", "/")
            sp = os.path.join(root, qd["state"])
            rec = (json.load(open(sp, encoding="utf-8-sig")).get(rid) if os.path.isfile(sp) else None) or {}
            why = []
            if os.path.normcase(os.path.abspath(rec.get("root") or "")) != os.path.normcase(run_dir):
                why.append("its state does not record this run's folder as the job root")
            if rec.get("status") != "done":
                why.append(f"state status {rec.get('status')!r}")
            js, rs = (job.get("env") or {}).get("DCLM_SEED"), run_env.get("DCLM_SEED")
            if js is None or rs is None or str(js) != str(rs):
                why.append(f"DCLM_SEED {js!r} vs the run's {rs!r}")
            if why:
                rejected.append(f"{rel}: " + "; ".join(why))
            else:
                matches.append((job, rel))
    if not matches:
        return None, None, rejected
    vals = {(m[0].get("steps"), m[0].get("warmup")) for m in matches}
    if len(vals) != 1:
        raise ValueError(f"wd_only_schedule: the training queues of {rid} disagree on steps / warm-up: "
                         f"{[(m[1], m[0].get('steps'), m[0].get('warmup')) for m in matches]}")
    return matches[0][0], ", ".join(m[1] for m in matches), rejected


def wd_only_schedule(run_env: dict, manifest_path: str | None = None, *, max_steps: int | None = None,
                     warmup: int | None = None, queue_path: str | None = None, builder_job=None) -> dict:
    """Optimizer steps, warm-up and executed steps of a run, from the sources its training used (post-freeze
    amendment A-1). run_queue passes the queue job's 'steps' and 'warmup' to run_exp.py as --steps/--warmup, and
    run_exp.py sets DCLM_MAX_STEPS / DCLM_WARMUP_STEPS before it builds the job, so run_env.json does not hold them.
    Sources, which must all agree when present: explicit arguments; DCLM_MAX_STEPS / DCLM_WARMUP_STEPS in run_env;
    the job of the frozen queue (FREEZE.json above the run -> queue; id = DCLM_RUN_SUFFIX); a test builder's job; for a run
    that is not a job of the frozen queue (and no queue_path is given), the job of the queue that trained it (A-2,
    _training_queue_job: its run_queue state records the run's folder as the job root, status done, same DCLM_SEED).
    Executed steps = the P-16 stop_after of NESTOR_INTERIM_EVAL when set, else max_steps. Raises ValueError when no
    source gives both values: the registry defaults of the pretraining recipe are never used."""
    found: dict[str, tuple] = {}
    if max_steps is not None or warmup is not None:
        found["arguments"] = (max_steps, warmup)
    if any(k in run_env for k in WD_STEP_KEYS):
        found["run_env"] = tuple(int(run_env[k]) if k in run_env else None for k in WD_STEP_KEYS)
    job, qp = _frozen_queue_job(run_env, manifest_path, queue_path)
    if job is not None:
        found[f"frozen queue job ({os.path.basename(qp)})"] = (int(job["steps"]) if "steps" in job else None,
                                                              int(job["warmup"]) if "warmup" in job else None)
    rejected: list[str] = []
    if job is None and queue_path is None:                     # A-2: a run the frozen queue did not train (C3 on G1 runs)
        tj, tq, rejected = _training_queue_job(run_env, manifest_path)
        if tj is not None:
            found[f"training queue job ({tq})"] = (int(tj["steps"]) if "steps" in tj else None, int(tj["warmup"]) if "warmup" in tj else None)
    if builder_job is not None:
        sc = getattr(builder_job, "scheduler", None)
        found["builder job"] = (int(builder_job.max_steps), int(getattr(sc, "warmup_steps", 0) or 0) if sc is not None else None)
    if not found:
        raise ValueError("wd_only_state: the run's step count and warm-up are unknown (run_env.json has no DCLM_MAX_STEPS/"
                         "DCLM_WARMUP_STEPS, no frozen queue job matches DCLM_RUN_SUFFIX="
                         f"{run_env.get('DCLM_RUN_SUFFIX')!r} and no training queue records this run's folder"
                         + (f"; rejected: {rejected}" if rejected else "") + "); refusing to fall back to the pretraining defaults")
    vals = set(found.values())
    if len(vals) != 1:
        raise ValueError(f"wd_only_state: the sources of the step count disagree: {found}")
    ms, wu = vals.pop()
    if ms is None or wu is None or ms <= 0 or wu < 0:
        raise ValueError(f"wd_only_state: incomplete step count / warm-up {found}")
    stop = _interim_stop(run_env)
    executed = stop or ms
    if executed > ms:
        raise ValueError(f"wd_only_state: stop_after {stop} exceeds the schedule's {ms} steps")
    return {"max_steps": ms, "warmup": wu, "executed": executed, "stop_after": stop or None, "sources": sorted(found)}


def _wd_device(base_state: dict, man: dict) -> tuple[str, str]:
    """Device of the decay-only trajectory: the eval device (CUDA) when its free memory holds an fp32 model,
    the momentum and the orthogonalisation temporaries of the trainable tensors with a 1.5 GiB margin, else the CPU
    (the authors' Newton-Schulz runs in bf16, which is slow on a CPU). NESTOR_WD_DEVICE=cpu|cuda overrides."""
    forced = os.environ.get("NESTOR_WD_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced, "NESTOR_WD_DEVICE"
    if not torch.cuda.is_available():
        return "cpu", "no CUDA device"
    n_all = sum(int(v.numel()) for v in base_state.values() if torch.is_tensor(v))
    n_tr = int(man.get("elements") or n_all) if isinstance(man.get("elements"), (int, float)) else n_all
    need = 4 * n_all + 4 * 4 * n_tr
    free, _total = torch.cuda.mem_get_info()
    if free - need > 1.5 * 2 ** 30:
        return "cuda", f"free {free / 2 ** 30:.1f} GiB, need {need / 2 ** 30:.1f} GiB"
    return "cpu", f"free {free / 2 ** 30:.1f} GiB < need {need / 2 ** 30:.1f} GiB + 1.5 GiB"


def _train_log(manifest_path: str | None) -> list[dict] | None:
    if not manifest_path:
        return None
    tl = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "trainlog", "train_steps.jsonl")
    if not os.path.isfile(tl):
        return None
    return [json.loads(x) for x in open(tl, encoding="utf-8").read().splitlines() if x.strip()]


def wd_only_state(experiment: str, base_state: dict, manifest: dict | str | None, run_env: dict,
                  *, model_builder=None, steps: int | None = None, max_steps: int | None = None,
                  warmup: int | None = None, queue_path: str | None = None, device: str | None = None,
                  report: dict | None = None) -> dict:
    """Decay-only trajectory of a run (zero gradients through the authors' optimizer + scheduler).

    run_env: the env overrides the run used (DCLM_LR_MUON, DCLM_LR_ADAMW, DCLM_WEIGHT_DECAY, NESTOR_FREEZE_ROUTERS,
    NESTOR_SUBSET_OPTIMIZER, NESTOR_SGD_*, NESTOR_INTERIM_EVAL); the queue runner writes it as run_env.json next to
    each run. The schedule length and warm-up come from wd_only_schedule: the job is rebuilt with
    DCLM_MAX_STEPS / DCLM_WARMUP_STEPS set, its max_steps is checked, and the decay-only loop runs the run's executed
    steps (optimizer.step then scheduler.step, as the trainer). Checks against the run's own records: the subset
    fingerprint of the manifest (the training env, schedule included) and, when the run's train log exists, its last
    step and the group learning rates it logged after every step. `model_builder` (tests) returns (model, job)
    instead of the registry build; `steps` overrides the executed steps (tests). `device`: where the trajectory runs
    (default: _wd_device -- the eval device when its free memory allows, else the CPU); the state is returned on the
    CPU (host copy), as for every weight-space variant.
    """
    import time as _time

    import subset_adapt as SA
    from nestor.train.frozen_routers import freeze_routers
    from nestor.train.optim import build_optimizer, build_scheduler

    t_start = _time.time()
    mpath = manifest if isinstance(manifest, str) else None
    if isinstance(manifest, str):
        if os.path.exists(manifest):
            man = json.load(open(manifest, encoding="utf-8"))
        elif run_env.get("NESTOR_SUBSET_SPEC"):
            raise FileNotFoundError(f"wd_only_state: subset run without its manifest {manifest}")
        else:
            man = {}                                   # FULL run: every parameter trainable, no subset
    else:
        man = manifest or {}
    env = {k: v for k, v in run_env.items() if k != "NESTOR_SUBSET_SPEC"}
    if str(env.get("NESTOR_FREEZE_ROUTERS", os.environ.get("NESTOR_FREEZE_ROUTERS", ""))).strip() != "1":
        raise ValueError("wd_only_state reproduces frozen-router runs only (NESTOR_FREEZE_ROUTERS=1): with live "
                         "routers the trainer's detached router-bias updates are data-dependent")
    sch_info = wd_only_schedule(run_env, mpath, max_steps=max_steps, warmup=warmup, queue_path=queue_path)
    env.update({"DCLM_MAX_STEPS": str(sch_info["max_steps"]), "DCLM_WARMUP_STEPS": str(sch_info["warmup"])})
    executed = int(steps) if steps is not None else sch_info["executed"]
    if device is None:
        device, sch_info["device_reason"] = _wd_device(base_state, man)
    with temp_env(env):
        if man.get("fingerprint") and man.get("spec") is not None:
            fp_env = {k: (man.get("env") or {}).get(k, os.environ.get(k)) for k in SA.FINGERPRINT_ENV}
            with temp_env({k: v for k, v in fp_env.items() if v is not None}):
                saved = {k: os.environ.pop(k) for k in SA.FINGERPRINT_ENV if fp_env.get(k) is None and k in os.environ}
                try:
                    fp = SA.spec_fingerprint(man["spec"])
                finally:
                    os.environ.update(saved)
            if fp != man["fingerprint"]:
                raise ValueError(f"wd_only_state: max_steps {sch_info['max_steps']} / warm-up {sch_info['warmup']} with the "
                                 "run's env do not reproduce the manifest's subset fingerprint (not the run's schedule)")
            sch_info["fingerprint"] = "reproduced"
        if model_builder is None:
            import nestor.experiments.pretrain  # noqa: F401
            from nestor.experiments.registry import get_experiment
            from nestor.models import Model

            job = get_experiment(experiment)().jobs[0]
            model = Model.from_config(job.model)
        else:                                          # tests: built inside the run's env, checked like the registry job
            model, job = model_builder()
        if int(job.max_steps) != sch_info["max_steps"]:
            raise ValueError(f"wd_only_state: the rebuilt job has max_steps {job.max_steps}, the run {sch_info['max_steps']}")
        missing, _ = model.load_state_dict(base_state, strict=False)
        names = {n for n, _ in model.named_parameters()}
        if any(m in names for m in missing):
            raise RuntimeError(f"wd_only_state: base state misses parameters, e.g. {[m for m in missing if m in names][:3]}")
        if device:
            model = model.to(device)
        SA.reset()
        SA.register_optimizer_hooks()
        if man.get("spec"):
            SA.apply_subset(model, man["spec"])
        if os.environ.get("NESTOR_FREEZE_ROUTERS", "").strip() == "1":
            freeze_routers(model)
        if os.environ.get("NESTOR_SUBSET_OPTIMIZER", "muon").strip().lower() == "sgd":
            import adapt_extras

            adapt_extras.patch_optimizer_builder()
        opt = build_optimizer(job.optimizer, model)
        sch = build_scheduler(job.scheduler, opt, job.max_steps)
        params = [p for p in model.parameters() if p.requires_grad]
        lrs = []
        for _ in range(executed):
            for p in params:
                p.grad = torch.zeros_like(p)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()
            lrs.append(sorted({round(float(g["lr"]), 12) for g in opt.param_groups}))   # as the train log records them
        state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        SA.reset()
    log = _train_log(mpath) if steps is None else None
    if log is not None:
        last = int(log[-1].get("step", 0)) if log else 0
        if last != executed:
            raise ValueError(f"wd_only_state: the run's train log ends at step {last}, the decay-only trajectory has {executed}")
        bad = [r["step"] for r, mine in zip(log, lrs) if "lrs" in r and (len(r["lrs"]) != len(mine) or any(
            abs(a - b) > 1e-9 * max(abs(a), abs(b), 1e-30) for a, b in zip(r["lrs"], mine)))]
        if bad:
            raise ValueError(f"wd_only_state: the decay-only learning rates differ from the run's train log at steps {bad[:5]}")
        sch_info["train_log_lrs"] = f"equal at all {len(lrs)} steps"
    sch_info.update(executed=executed, device=str(device or "cpu"), seconds=round(_time.time() - t_start, 2))
    print(f"[wd_only] {experiment}: max_steps {sch_info['max_steps']}, warm-up {sch_info['warmup']}, executed {executed} "
          f"(sources: {', '.join(sch_info['sources'])}); fingerprint {sch_info.get('fingerprint', 'not checked')}; "
          f"train-log LRs {sch_info.get('train_log_lrs', 'not checked')}; {sch_info['device']}, {sch_info['seconds']} s", flush=True)
    if report is not None:
        report.update(sch_info)
    return state


# ============================================================================= displacement
def displacement(model, ref_state: dict, *, detail_kinds: Iterable[str] = ("moe", "pub", "priv", "attn")) -> dict:
    """Realized squared displacement ||theta - theta_ref||_F^2 by unit kind and per unit."""

    import subset_adapt as SA

    detail = set(detail_kinds)
    by_kind: Counter = Counter()
    by_unit: dict[str, float] = defaultdict(float)
    for e in SA.build_catalog(model):
        if e.name not in ref_state:
            continue
        p = e.param.detach()
        d = p.float() - ref_state[e.name].to(p.device).float()
        if e.sliced:
            sq = d.pow(2).reshape(d.shape[0], -1).sum(1).double().cpu().tolist()
            for u, v in zip(e.units, sq):
                by_kind[u.kind] += v
                if u.kind in detail and v:
                    by_unit[u.key] += v
        else:
            v = float(d.double().pow(2).sum())
            u = e.units[0]
            by_kind[u.kind] += v
            if u.kind in detail and v:
                by_unit[u.key] += v
    return {"total_sq": float(sum(by_kind.values())), "by_kind_sq": dict(by_kind), "by_unit_sq": dict(by_unit)}
