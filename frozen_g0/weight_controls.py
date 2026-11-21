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
}


def _group_pred(group: str, tail: int | None):
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


def wd_only_state(experiment: str, base_state: dict, manifest: dict | str | None, run_env: dict,
                  *, model_builder=None, steps: int | None = None) -> dict:
    """Decay-only trajectory of a run (zero gradients through the authors' optimizer + scheduler).

    run_env: the env overrides the run used (DCLM_LR_MUON, DCLM_LR_ADAMW, DCLM_WEIGHT_DECAY,
    DCLM_MAX_STEPS, DCLM_WARMUP_STEPS, NESTOR_FREEZE_ROUTERS, NESTOR_SUBSET_OPTIMIZER, NESTOR_SGD_*);
    the queue runner writes it as run_env.json next to each run. `model_builder` (tests) returns
    (model, job) instead of the registry build.
    """

    import subset_adapt as SA
    from nestor.train.frozen_routers import freeze_routers
    from nestor.train.optim import build_optimizer, build_scheduler

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
    with temp_env(env):
        if model_builder is None:
            import nestor.experiments.pretrain  # noqa: F401
            from nestor.experiments.registry import get_experiment
            from nestor.models import Model

            job = get_experiment(experiment)().jobs[0]
            model = Model.from_config(job.model)
        else:
            model, job = model_builder()
        missing, _ = model.load_state_dict(base_state, strict=False)
        names = {n for n, _ in model.named_parameters()}
        if any(m in names for m in missing):
            raise RuntimeError(f"wd_only_state: base state misses parameters, e.g. {[m for m in missing if m in names][:3]}")
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
        for _ in range(job.max_steps if steps is None else steps):
            for p in params:
                p.grad = torch.zeros_like(p)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        SA.reset()
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
