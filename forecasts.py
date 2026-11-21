"""Prospective per-run forecasts, preregistration v1.3 Section 5 (forecasts_v13.csv / forecasts_v13_meta.json, format
forecasts_v13/2 of forecast_io.py), and the C5 French forecasts of v1.2 sec. 12 (own file, unchanged). Torch-free.

Stage C (prereg v1.3 sec. 5.6: after the G2 record, before the freeze):
  python forecasts.py units  --confirm-plan confirm/plan_confirm.json --fit-plans g1/plan_g1.json,g2/plan_g2.json \
                             --g0-config g0/g0_gate_config.json --out-dir results/forecast
      -> results/forecast/forecast_units.json (+ .sha256): target runs (sec. 5.1) with block / base / target / steps / old
         corpora; probe units (sec. 5.3: one per (base, subset, target, optimizer), count check against the registered
         108 MoE + 11 NESTOR units at n_d = 5); fit runs and their observations (sec. 5.4); KSD drive runs;
         results/forecast/queue_forecast_inputs.json: the 8-step E-ALL drive runs (train jobs, 3 interim evaluations =
         the first 64 windows of DCLM, code and math after step 8) + one theta0 probe job per base (forecast_probe.py).
  python run_queue.py results/forecast/queue_forecast_inputs.json                                    # GPU
  python forecasts.py make   --units results/forecast/forecast_units.json --probes results/forecast/probes \
                             --states results/queue_state_g1.json,results/queue_state_g2.json,results/queue_state_forecast.json \
                             --out-dir freeze
      -> freeze/forecasts_v13.csv + freeze/forecasts_v13_meta.json (forecast_io.write_forecasts) and forecast_io.verify;
         prints both SHA-256 (they enter freeze_record.json). --b3-addendum writes forecasts_v13_b3.* (sec. 5.6).
  python forecasts.py verify freeze/forecasts_v13.csv freeze/forecasts_v13_meta.json
After all runs (sec. 5.7, descriptive): python forecasts.py score --csv ... --meta ... --units ... --states ...
C5 French (v1.2 sec. 12): python forecasts.py c5-french --plan ... --drive ... --states ... --fit-plans ... --out ...

Forecasters (sec. 5.2): F_hat(r, c) = a x(unit(r), c) with x = RO_c(S) (RO), Q_E,c (QE), RO_c Q_E,c (M1), SVC_c (SVC),
|S|/N_trainable (TF), RND_c (RND), F_c(8-step E-ALL drive on T) RO_c(S) (KSD); M2 = a RO Q_E + b gamma_S,c. Scalars per
architecture (MoE bases share one set; NESTOR) on the G1/G2 runs only: log a = mean(log F - log x) over observations with
F > 0 and x > 0; M2 by OLS on F without intercept. A forecaster that cannot be computed for a run is NA with the reason in
meta.na_rules (make refuses missing probe or drive inputs unless --allow-missing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import forecast_io as FIO  # noqa: E402

C5_PROTOCOL = "v1.2 sec. 12 (C5 French); scalar rule a_c = exp(mean log(F / x)) over the non-French runs"
OLD_CORPORA = ("dclm", "code", "math")                 # C5 (v1.2) corpus order
CORPORA = tuple(FIO.EVAL_CORPORA)                     # v1.3 forecast corpora ('code' = the registered code stream)
FORMAT_UNITS = "forecast_units_v13/1"
ARCH_OF_BASE = {"moe_b1": "moe", "moe_b2": "moe", "moe_b3": "moe", "nestor": "nestor"}
FAMILY = {"moe": "moe", "nestor": "priv"}
ROLE_BLOCK = {"c1": "C1", "c2_muon": "C2", "c2_sgd": "C2", "c1_b3": "B3", "dose": "DR", "m12b": "M12B",
              "c1_nestor": "NB", "full": "C3", "c1de": "DE"}       # DE: C1-DE German draws (prereg v1.3 sec. 4.12; forecasts_v13/3)
EXCLUDED_ROLES = {
    "c5_pair": "French C5 run: own protocol (v1.2 sec. 12; v1.3 sec. 5.1)",
    "c5_full": "French C5 run: own protocol (v1.2 sec. 12; v1.3 sec. 5.1)",
    "drive": "French C5 k-step drive: own protocol (v1.2 sec. 12; v1.3 sec. 5.1)",
    "dclm_cont": "C3 run that is not a FULL run (DCLM-CONT; v1.3 sec. 5.1)",
    "freeze": "C3 run that is not a FULL run (training-time freeze; v1.3 sec. 5.1)",
    "d4": "D4 embedding-row run: the forecasters are defined for routed-expert subsets, RO is undefined (v1.3 sec. 5.1)",
}
EXCLUDED_EVALS = "all revert evaluations (C3 per-layer and group reverts) are evaluations, not runs (v1.3 sec. 5.1)"
FIT_ROLES = ("full", "eall", "aa", "aa_nestor", "dclm_cont", "sgd")
FIT_EXCLUDED = {"negctrl": "G1 negative control (unmasked decay, 8 steps, no evaluation): not a fit run of v1.3 sec. 5.4"}
REGISTERED_C3_FULL = sorted([("moe_b1", "books", 101), ("moe_b1", "books", 102), ("moe_b1", "code", 101),
                             ("moe_b1", "code", 102), ("moe_b2", "books", 101), ("moe_b2", "math", 101),
                             ("moe_b2", "code", 101), ("moe_b2", "german", 101), ("nestor", "books", 101),
                             ("nestor", "books", 102), ("nestor", "math", 101), ("nestor", "math", 102),
                             ("nestor", "code", 101), ("nestor", "code", 102)])            # decision P-7
REGISTERED_DRIVES = [("moe_b1", "books", "muon"), ("moe_b1", "german", "muon"), ("moe_b1", "german", "sgd"),
                     ("moe_b1", "math", "muon"), ("moe_b1", "dclm", "muon"), ("moe_b1", "code", "muon"),
                     ("moe_b2", "books", "muon"), ("moe_b2", "german", "muon"), ("moe_b2", "german", "sgd"),
                     ("moe_b2", "math", "muon"), ("moe_b2", "code", "muon"), ("moe_b3", "books", "muon"),
                     ("nestor", "books", "muon"), ("nestor", "german", "muon"), ("nestor", "math", "muon"),
                     ("nestor", "code", "muon")]                                          # sec. 5.3 (12 MoE + 4 NESTOR)
REGISTERED_UNITS = {"n_d": 5, "moe": 108, "nestor": 11,
                    "moe_target": {"C1": 40, "C2": 32, "B3": 6, "DR": 8, "M12B": 3, "C3": 6}, "moe_fit": 13,
                    "nestor_target": {"NB": 6, "C3": 3}, "nestor_fit": 2}               # sec. 5.3 (version 3 counts)
# C1-DE (prereg v1.3 sec. 4.12, version 5/6): 6 draws x 2 runs on MoE b1/b2 -> 12 MoE units OUTSIDE the registered 108;
# their probe cost (12 x the unit formula, ~0.17 GPU-h) is costed inside the C1-DE block of the confirm plan
C1DE_UNITS = {"moe": {"DE": 12}}
BLOCK_ORDER = {"C1": 0, "C2": 1, "B3": 2, "DR": 3, "M12B": 4, "C3": 5, "NB": 6, "DE": 7, "FIT": 8}
WITHIN_BLOCKS = ("C1", "B3", "DR", "NB", "DE")     # sec. 5.7: R2_within on the crossover contrasts (C1-DE added in version 5)
HAAR_SEED_BASE = 9000              # sec. 5.3: norm-matched Haar direction, seed 9000 + unit index
PROBE_WINDOWS = 8                  # sec. 5.3: 8 windows per old corpus
GRAD_WINDOWS = 8                   # sec. 5.3: u_S from 8 target windows
HEAD_WINDOWS = 64                  # budget v1.3: one interim set = the first 64 windows (INT_TOK)
DRIVE_STEPS, DRIVE_WARMUP, DRIVE_SEED = 8, 1, 101
EALL_SPEC = {"moe": {"mode": "train", "select": [{"kind": "moe"}]},
             "nestor": {"mode": "train", "select": [{"kind": ["priv", "pub"]}]}}       # E-ALL = every expert (G0 E_ALL)
# subset_adapt (torch) unit grammar, copied for the torch-free RO: KINDS / MACROS / filter keys
_KINDS = ("attn", "priv", "pub", "moe", "router", "path_gate", "norm", "depth_read", "embed", "head")
_MACROS = {"column": ("attn", "priv")}
_FILTER_KEYS = {"kind", "block", "layer", "column", "expert", "role", "dense_tail"}
# cost model of the registered budget (budget_v12 / budget_v13 constants)
_TP = dict(eval_moe=27_000.0, eval_nestor=8_500.0, train_moe=34_000.0, train_nestor=10_200.0)
_COMPILE_S = dict(moe=0.0, nestor=8.5 * 60)
_TOK_STEP, _INT_TOK = 262_144, 64 * 4_096
_PROBE_S = (986.6 + 986.4) / 2 / (20 * 2)      # budget_v13 PROBE_S: s per direction (its +/- eps pair) and corpus at 24
                                              # windows (G0-7: 10 books draws x sides A/B = 20 directions, 2 corpora)
_PROBE_DIRECTIONS = 2                         # u_S and h, each an antithetic pair (budget_v13 FORECAST_UNITS)
REGISTERED_INPUT_H = {"drives": 1.20, "probes": 2.08, "total": 3.28}   # prereg v1.3 sec. 5.3 / budget_v13 (n_d = 5)


def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=float)


def _sha_obj(o: Any) -> str:
    return hashlib.sha256(_canon(o).encode("utf-8")).hexdigest()


def _sha_file(p: str | None) -> str | None:
    if not p or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def set_name(corpus: str, code_stream: str) -> str:
    """Evaluation-set name of forecast corpus `corpus` ('code' is the registered code stream of the G0 record)."""
    return code_stream if corpus == "code" else corpus


# ============================================================================= subset -> routing-table keys (torch-free)
def _parse_key(k: str) -> dict:
    parts = k.split(":")
    d: dict[str, Any] = {"kind": parts[0]}
    tags = {"b": "block", "l": "layer", "c": "column", "e": "expert"}
    for p in parts[1:]:
        if p[:1] in tags and p[1:].isdigit():
            d[tags[p[0]]] = int(p[1:])
        else:
            d["role"] = p
    return d


def _want(v: Any) -> set | None:
    if v is None or v == "*":
        return None
    return set(v) if isinstance(v, (list, tuple, set)) else {v}


def _key_matches(d: dict, f: dict) -> bool:
    """subset_adapt.unit_matches on a parsed routing-table key (routed expert units only)."""
    bad = set(f) - _FILTER_KEYS
    if bad:
        raise ValueError(f"unknown filter keys {sorted(bad)}")
    kinds = _want(f.get("kind", "*"))
    if kinds is not None:
        exp = set()
        for k in kinds:
            exp.update(_MACROS.get(k, (k,)))
        if exp - set(_KINDS):
            raise ValueError(f"unknown unit kind(s) {sorted(exp - set(_KINDS))}")
        if d["kind"] not in exp:
            return False
    for key in ("block", "layer", "column", "expert", "role"):
        w = _want(f.get(key, "*"))
        if w is None:
            continue
        if key == "block" and any(isinstance(x, int) and x < 0 for x in w):
            raise ValueError("negative block indices need the model geometry (not supported torch-free)")
        if d.get(key) not in w:
            return False
    if f.get("dense_tail"):
        return False                       # routed strata exclude the dense tail (crossover.reg_strata)
    return True


def spec_keys(spec: dict | None, keys_all: list[str]) -> list[str]:
    """Routing-table unit keys trained under `spec` (None = FULL: every routed expert)."""
    if spec is None:
        return list(keys_all)
    flt = spec.get("select", [])
    hit = [k for k in keys_all if any(_key_matches(_parse_key(k), f) for f in flt)]
    if spec.get("mode", "train") == "freeze":
        hs = set(hit)
        return [k for k in keys_all if k not in hs]
    return hit


def ro_of(t, keys: list[str], set_nm: str) -> float | None:
    """RO_c(S): share of the theta0 routing statistic of corpus c (evaluation half, t.est) on the cells of S."""
    est = t.est.get(set_nm)
    if est is None:
        return None
    ks = set(keys)
    m = np.array([[k in ks for k in row] for row in t.keys], dtype=bool)
    tot = float(np.asarray(est, dtype=np.float64).sum())
    return float(np.asarray(est, dtype=np.float64)[m].sum() / tot) if tot > 0 else None


# ============================================================================= enumeration (sec. 5.1, 5.3, 5.4)
def _load(p: str) -> dict:
    doc = json.load(open(p, encoding="utf-8-sig"))
    doc["_path"] = p
    return doc


def _file_vars(q: dict) -> dict:
    """run_queue 'file_vars' ({"sgd_lr0": "g2/sgd_grid.json#grid.0"}) resolved exactly as run_queue.vars_now does."""
    out = {}
    for name, ref in (q.get("file_vars") or {}).items():
        path, _, ptr = ref.partition("#")
        if not os.path.exists(path):
            continue
        val = json.load(open(path, encoding="utf-8-sig"))
        for k in (ptr.split(".") if ptr else []):
            val = val[int(k)] if isinstance(val, list) else val[k]
        out[name] = val if isinstance(val, str) else json.dumps(val)
    return out


def _resolve_job(job: dict, fv: dict) -> dict:
    """The job with its env placeholders of file_vars filled in (other placeholders, e.g. {ckpt_*}, stay)."""
    env = dict(job.get("env") or {})
    for k, v in env.items():
        if isinstance(v, str) and "{" in v:
            for name, val in fv.items():
                v = v.replace("{" + name + "}", str(val))
            env[k] = v
    return dict(job, env=env)


def _queue(plan: dict) -> tuple[dict, dict, str]:
    qp = plan["queue"]
    q = json.load(open(qp, encoding="utf-8-sig"))
    fv = _file_vars(q)
    jobs = {j["id"]: _resolve_job(j, fv) for j in q["jobs"]}
    left = sorted({f"{j['id']}:{k}" for j in jobs.values() for k, v in (j.get("env") or {}).items()
                   if k in ("NESTOR_SGD_LR", "DCLM_LR_MUON", "DCLM_LR_ADAMW") and "{" in str(v)})
    if left:
        raise SystemExit(f"{qp}: learning-rate placeholders unresolved ({left[:4]}); their file_vars sources must exist "
                         f"at stage C (after the G2 record)")
    q = dict(q, jobs=list(jobs.values()))
    return jobs, q, qp


def optimizer_of(job: dict) -> dict:
    env = job.get("env") or {}
    if env.get("NESTOR_SUBSET_OPTIMIZER", "").lower() == "sgd":
        return {"kind": "sgd", "lr": float(env["NESTOR_SGD_LR"]), "momentum": float(env.get("NESTOR_SGD_MOMENTUM", 0.95)),
                "nesterov": str(env.get("NESTOR_SGD_NESTEROV", "1")) == "1"}
    return {"kind": "muon", "lr": float(env.get("DCLM_LR_MUON", 0.0025)), "lr_adamw": float(env.get("DCLM_LR_ADAMW", 7.5e-5)),
            "momentum": 0.95, "nesterov": True}


def enumerate_targets(cplan: dict, jobs: dict) -> tuple[list[dict], list[dict], list[str]]:
    targets, excluded, notes = [], [], []
    for r in cplan.get("runs", []):
        role = r.get("role")
        if role in EXCLUDED_ROLES:
            excluded.append({"run_id": r["id"], "role": role, "reason": EXCLUDED_ROLES[role]})
            continue
        if role not in ROLE_BLOCK:
            raise SystemExit(f"run {r['id']}: role {role!r} has no forecast rule in prereg v1.3 sec. 5.1")
        blk, tgt, base = ROLE_BLOCK[role], r.get("target"), r["base"]
        if tgt not in FIO.TARGETS or base not in FIO.BASES:
            raise SystemExit(f"run {r['id']}: target {tgt!r} / base {base!r} outside the forecast format")
        steps = [int(r["steps"])]
        if blk == "C2":
            if r.get("s_star") is None:
                notes.append(f"{r['id']}: s* not fixed by G2-3 -> forecast at the final step only")
            else:
                steps = sorted({int(r["s_star"]), int(r["steps"])})
        if r["id"] not in jobs:
            raise SystemExit(f"run {r['id']} is not in the confirm queue")
        targets.append({"run_id": r["id"], "block": blk, "role": role, "base": base, "arch": ARCH_OF_BASE[base],
                        "target": tgt, "steps": steps, "old_corpora": [c for c in CORPORA if c != tgt],
                        "seed": r.get("seed"), "d": r.get("d"), "side": r.get("side"),
                        "plan_ro": (r.get("subset") or {}).get("ro"), "_job": jobs[r["id"]]})
    return targets, excluded, notes


def enumerate_fit(fplan: dict, jobs: dict, code_stream: str) -> tuple[list[dict], list[dict]]:
    out, excl = [], []
    for r in fplan.get("runs", []):
        role = r.get("role")
        if role in FIT_EXCLUDED:
            excl.append({"run_id": r["id"], "role": role, "reason": FIT_EXCLUDED[role]})
            continue
        if role not in FIT_ROLES:
            raise SystemExit(f"fit run {r['id']}: role {role!r} is not a G1/G2 fit role of v1.3 sec. 5.4")
        tgt = r.get("target")
        sets, fsets = set(r.get("eval_sets") or []), set(r.get("full_sets") or [])
        obs = []
        for c in CORPORA:
            if c == tgt:
                continue
            sn = set_name(c, code_stream)
            if sn in sets:
                obs.append({"step": int(r["steps"]), "corpus": c, "set": sn, "kind": "final"})
            if sn in fsets:
                obs += [{"step": int(s), "corpus": c, "set": sn, "kind": "interim_full"}
                        for s in (r.get("full_steps") or []) if int(s) > 0]
        if r["id"] not in jobs:
            raise SystemExit(f"fit run {r['id']} is not in its queue {fplan.get('queue')}")
        out.append({"run_id": r["id"], "plan": fplan.get("kind"), "plan_path": fplan["_path"], "role": role,
                    "base": r["base"], "arch": ARCH_OF_BASE[r["base"]], "target": tgt, "steps": int(r["steps"]),
                    "eval_dir": r.get("eval_dir", "eval_main"), "observations": obs, "_job": jobs[r["id"]]})
    return out, excl


def build_units(targets: list[dict], fits: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for kind, recs in (("target", targets), ("fit", fits)):
        for rec in recs:
            job = rec["_job"]
            spec, opt = job.get("spec"), optimizer_of(job)
            arch = rec["arch"]
            sub = "FULL" if spec is None else ("EALL" if spec == EALL_SPEC[arch] else "SPEC")
            key = (arch, rec["base"], rec["target"], opt["kind"], f"{opt['lr']:.9g}", _sha_obj(spec) if spec else "FULL")
            data = (job.get("env") or {}).get("DCLM_DATA_PATH")
            u = groups.setdefault(key, {"arch": arch, "base": rec["base"], "target": rec["target"], "optimizer": opt,
                                        "subset": sub, "spec": spec, "subset_sha256": key[-1], "data": data,
                                        "experiment": job.get("experiment"), "target_runs": [], "fit_runs": [],
                                        "blocks": []})
            if u["data"] != data:
                raise SystemExit(f"unit {key}: runs train on different data ({u['data']} vs {data})")
            u[f"{kind}_runs"].append(rec["run_id"])
            blk = rec.get("block", "FIT") if kind == "target" else "FIT"
            if blk not in u["blocks"]:
                u["blocks"].append(blk)
            rec["_unit_key"] = key
    order = lambda u: (0 if u["arch"] == "moe" else 1, u["base"], min(BLOCK_ORDER[b] for b in u["blocks"]), u["target"],
                       u["optimizer"]["kind"], u["optimizer"]["lr"], (u["target_runs"] or u["fit_runs"])[0])
    units = sorted(groups.values(), key=order)
    for i, u in enumerate(units):
        u["index"] = i
        u["haar_seed"] = HAAR_SEED_BASE + i
        u["drive"] = f"ksd_{u['base']}_{u['target']}_{u['optimizer']['kind']}"
        u["old_corpora"] = [c for c in CORPORA if c != u["target"]]
    idx = {k: u["index"] for k, u in groups.items()}
    for recs in (targets, fits):
        for rec in recs:
            rec["unit"] = idx[rec.pop("_unit_key")]
    return units


def unit_counts(units: list[dict], targets: list[dict]) -> dict:
    """Realised unit counts by architecture and block against the registered version-3 counts (sec. 5.3)."""
    by: dict = {"moe": {"target": Counter(), "fit": 0}, "nestor": {"target": Counter(), "fit": 0}}
    for u in units:
        if u["target_runs"]:
            blk = [b for b in u["blocks"] if b != "FIT"]
            by[u["arch"]]["target"][blk[0]] += 1
        else:
            by[u["arch"]]["fit"] += 1
    nd = None
    c1d = [t["d"] for t in targets if t["block"] == "C1" and t.get("d") is not None]
    if c1d:
        nd = max(c1d) // 2
    out: dict[str, Any] = {"n_d": nd, "registered": REGISTERED_UNITS, "deviations": []}
    for arch in ("moe", "nestor"):
        tgt = dict(sorted(by[arch]["target"].items()))
        tot = sum(tgt.values()) + by[arch]["fit"]
        out[arch] = {"target_by_block": tgt, "target": sum(tgt.values()), "fit": by[arch]["fit"], "total": tot}
        reg_t = REGISTERED_UNITS[f"{arch}_target"]
        for b, n in reg_t.items():
            got = tgt.get(b, 0)
            if got != n:
                why = {"C1": f"C1 draws d = 1..2 n_d with n_d = {nd} (registered count at n_d = 5)",
                       "C2": "C2 draw count of the G2 record (registered: 8 draws x 2 arms x 2 optimizers)",
                       "B3": "third base not in the stage-C plan (late base: sec. 5.6 addendum)",
                       "NB": "NESTOR-books not run (signal and feasibility, sec. 4.4)"}.get(b, "design of the stage-C plan")
                out["deviations"].append({"arch": arch, "block": b, "registered": n, "realised": got, "reason": why})
        for b, n in C1DE_UNITS.get(arch, {}).items():       # outside the registered 108 + 11 (sec. 4.12)
            got = tgt.get(b, 0)
            runs = sum(1 for t in targets if t["block"] == b and t["arch"] == arch)
            if got != n:
                why = ("C1-DE: a side coincides between accepted draws (one unit, the same theta0 probe, serves both runs)"
                       if 0 < got < runs else "C1-DE: not run, cut by the P-5 order, or draws not ok (sec. 4.12)")
                out["deviations"].append({"arch": arch, "block": b, "registered": n, "realised": got, "runs": runs,
                                          "reason": why})
        if by[arch]["fit"] != REGISTERED_UNITS[f"{arch}_fit"]:
            out["deviations"].append({"arch": arch, "block": "fit", "registered": REGISTERED_UNITS[f"{arch}_fit"],
                                      "realised": by[arch]["fit"],
                                      "reason": "G1/G2 run set differs from the registered fit set (e.g. a G1-2 widen "
                                                "re-run adds A/A subsets)"})
    out["status"] = "registered counts" if not out["deviations"] else "deviations listed"
    return out


def structural_checks(units: list[dict], targets: list[dict]) -> list[str]:
    """Hard rules: subset blocks own one unit per run; the C3 FULL runs are the 14 of decision P-7. C1-DE (DE) is not under
    the one-unit rule: two accepted draws may share a side (the duplicate rule of sec. 4.12 is on the pair), and such runs
    share their unit (identical base, subset, target and optimizer -> identical theta0 probe); unit_counts records it."""
    errs = []
    for u in units:
        sub_blocks = [b for b in u["blocks"] if b in ("C1", "C2", "B3", "DR", "M12B", "NB")]
        if sub_blocks and len(u["target_runs"]) != 1:
            errs.append(f"unit {u['index']} ({sub_blocks}) is shared by runs {u['target_runs']}: identical subsets")
    c3 = sorted((t["base"], t["target"], int(t["seed"])) for t in targets if t["block"] == "C3")
    if c3 != REGISTERED_C3_FULL:
        errs.append(f"C3 FULL runs {c3} != the 14 registered FULL runs of decision P-7")
    return errs


# ============================================================================= stage C step 1: units + input queue
def _tables(cfg_paths: list[str]) -> tuple[dict, list[dict]]:
    import crossover as CX
    import g0_gate as GG

    tabs, files = {}, []
    for cp in cfg_paths:
        cfg = GG.load_config(cp)
        files.append({"path": cp, "sha256": _sha_file(cp), "role": "G0 gate config (routing-table dirs)"})
        for b, bb in cfg["bases"].items():
            if b in tabs:
                continue
            t = CX.reg_load_dir(bb["dir"], bb["family"], layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
            tabs[b] = t
            for c, s in sorted(t.sources.items()):
                for role in ("selection", "estimation"):
                    f = (s.get(role) or {}).get("file")
                    if f:
                        files.append({"path": f, "sha256": _sha_file(f), "role": f"theta0 routing table {b}/{c}/{role}"})
    return tabs, files


def _template(jobs_all: list[dict], base: str, opt_kind: str) -> dict | None:
    """A queue train job of `base` with optimizer kind `opt_kind` (env / experiment template of a drive run; micro-batch
    and checkpointing come from the confirm plan's start configuration)."""
    pref = ("c3_full_", "g1_moe_b1_full_", "g1_nestor_full_", "c1_", "nb_", "dose_", "b3_") if opt_kind == "muon" \
        else ("c2_",)
    cands = [j for j in jobs_all if j.get("kind") == "train" and optimizer_of(j)["kind"] == opt_kind and
             f"{{ckpt_{base}}}" == (j.get("env") or {}).get("DCLM_INIT_CHECKPOINT")]
    for p in pref:
        for j in cands:
            if j["id"].startswith(p):
                return j
    return cands[0] if cands else None


def drive_job(did: str, base: str, arch: str, target: str, opt_kind: str, data: str, tmpl: dict, sets: dict,
              code_stream: str, sgd_lr: float | None, train_cfg: dict | None = None) -> dict:
    """One KSD drive run. `train_cfg` = the confirm plan's confirm_config["train"]: the architecture's start
    configuration (micro, ckpt; run_queue.py falls back from it) and measured step time; without it the template's
    micro/ckpt and the operator training rate (older plans)."""
    st = (train_cfg or {}).get(arch) or {}
    env = {k: v for k, v in (tmpl.get("env") or {}).items() if k != "NESTOR_INTERIM_FULL"}
    env.update({"DCLM_DATA_PATH": data, "DCLM_RUN_SUFFIX": did, "DCLM_SEED": str(DRIVE_SEED)})
    if opt_kind == "sgd":
        if sgd_lr is None:
            raise SystemExit(f"{did}: no G2-2 SGD learning rate (no C2 SGD job of {base}; pass --sgd-lr)")
        env["NESTOR_SGD_LR"] = repr(float(sgd_lr))
    iset = []
    for c in CORPORA:
        sn = set_name(c, code_stream)
        d = dict(sets[sn])
        iset.append(dict({"head": HEAD_WINDOWS, "name": sn}, **{k: v for k, v in d.items() if k in
                                                                ("path", "select", "tokens", "span", "span_align")}))
    if st.get("s_per_step"):
        tr = DRIVE_STEPS * float(st["s_per_step"]) + _COMPILE_S[arch]
    else:
        tr = DRIVE_STEPS * _TOK_STEP / _TP[f"train_{arch}"] + _COMPILE_S[arch]
    ev = 2 * len(CORPORA) * _INT_TOK / _TP[f"eval_{arch}"]
    return {"id": did, "kind": "train", "experiment": tmpl.get("experiment"),
            "micro": int(st["micro"]) if "micro" in st else tmpl.get("micro", 2),
            "ckpt": bool(st["ckpt"]) if "ckpt" in st else bool(tmpl.get("ckpt", False)), "seq": int(tmpl.get("seq", 4096)),
            "tokens_per_step": int(tmpl.get("tokens_per_step", _TOK_STEP)), "steps": DRIVE_STEPS, "warmup": DRIVE_WARMUP,
            "spec": EALL_SPEC[arch], "out_root": f"outputs/forecast/{did}", "env": env, "evals": [],
            "interim": {"block": int(tmpl.get("seq", 4096)), "full_sets": [], "full_steps": [], "routing": False,
                        "sets": iset, "steps": [0, DRIVE_STEPS]},
            "est_h": round((tr + ev) / 3600.0, 4),
            "purpose": f"KSD drive (prereg v1.3 sec. 5.3): 8-step E-ALL on {target} ({opt_kind}); F_c = step-8 minus "
                       f"step-0 CE on the first {HEAD_WINDOWS} windows of DCLM, code and math"}


def probe_unit_record(u: dict) -> dict:
    """The fields of a unit that the theta0 probe depends on (its sha256 binds a probe file to its unit)."""
    return {k: u[k] for k in ("index", "arch", "base", "target", "optimizer", "spec", "subset", "data", "haar_seed",
                              "old_corpora")}


def cmd_units(confirm_plan: str, fit_plans: list[str], g0_configs: list[str], out_dir: str, sgd_lr: float | None = None,
              allow_c3_deviation: bool = False) -> dict:
    cplan = _load(confirm_plan)
    cjobs, cq, cqp = _queue(cplan)
    code_stream = cplan.get("code_stream") or "code"
    targets, excluded, notes = enumerate_targets(cplan, cjobs)
    fits, fit_excl, fplans, jobs_all = [], [], [], list(cq["jobs"])
    for fp in fit_plans:
        p = _load(fp)
        fj, fq, fqp = _queue(p)
        jobs_all += fq["jobs"]
        f, e = enumerate_fit(p, fj, code_stream)
        fits += f
        fit_excl += e
        fplans.append({"path": fp, "sha256": _sha_file(fp), "kind": p.get("kind"), "queue": fqp,
                       "queue_sha256": _sha_file(fqp), "state": p.get("state"), "base_eval_files": p.get("base_eval_files")})
    units = build_units(targets, fits)
    errs = structural_checks(units, targets)
    if errs and not (allow_c3_deviation and all(e.startswith("C3 FULL") for e in errs)):
        raise SystemExit("forecast units: " + "; ".join(errs))
    notes += [f"ACCEPTED DEVIATION: {e}" for e in errs]
    tabs, tab_files = _tables(g0_configs)
    for u in units:
        t = tabs.get(u["base"])
        if t is None:
            raise SystemExit(f"no theta0 routing tables for base {u['base']} (add its G0 config to --g0-config)")
        keys_all = [k for row in t.keys for k in row]
        ks = spec_keys(u["spec"], keys_all)
        u["n_keys"], u["n_keys_family"] = len(ks), len(keys_all)
        u["ro"] = {c: ro_of(t, ks, set_name(c, code_stream)) for c in u["old_corpora"]}
    ro_dev = []
    by_idx = {u["index"]: u for u in units}
    for t_ in targets:
        pr = t_.pop("plan_ro") or {}
        for c, v in pr.items():
            mine = by_idx[t_["unit"]]["ro"].get(c)
            if c in by_idx[t_["unit"]]["ro"] and mine is not None and v is not None:
                ro_dev.append(abs(mine - float(v)))
    # KSD drives: registered list (sec. 5.3) and the drives the units need
    need = sorted({(u["base"], u["target"], u["optimizer"]["kind"]) for u in units})
    bases = {u["base"] for u in units}
    outside = [d for d in need if d not in REGISTERED_DRIVES]
    if outside:
        raise SystemExit(f"units need KSD drives outside the registered list: {outside}")
    sgd_by_base = {u["base"]: u["optimizer"]["lr"] for u in units if u["optimizer"]["kind"] == "sgd" and "C2" in u["blocks"]}
    drives = []
    for (b, tgt, ok) in REGISTERED_DRIVES:
        if b not in bases:
            notes.append(f"registered drive {b}/{tgt}/{ok} not built: base {b} not in the stage-C plans")
            continue
        did = f"ksd_{b}_{tgt}_{ok}"
        users = [u["index"] for u in units if u["drive"] == did]
        data = next((u["data"] for u in units if u["base"] == b and u["target"] == tgt), None) or \
            next((u["data"] for u in units if u["target"] == tgt), None)
        if data is None:
            raise SystemExit(f"{did}: no training data path for target {tgt}")
        tmpl = _template(jobs_all, b, ok)
        if tmpl is None:
            raise SystemExit(f"{did}: no train job of {b} with optimizer {ok} to take the recipe from")
        sets = (cplan.get("sets") or {}).get(b) or next(iter((cplan.get("sets") or {}).values()))
        job = drive_job(did, b, ARCH_OF_BASE[b], tgt, ok, data, tmpl, sets, code_stream,
                        sgd_lr if sgd_lr is not None else sgd_by_base.get(b) if ok == "sgd" else None,
                        train_cfg=(cplan.get("confirm_config") or {}).get("train"))
        drives.append({"id": did, "base": b, "arch": ARCH_OF_BASE[b], "target": tgt, "optimizer": ok, "data": data,
                       "units": users, "registered": True, "job": job,
                       "note": None if users else "registered; no unit of the stage-C plans uses it"})
    # MF-3: no drive shares a data seed with a draw unit of the confirm plan (drives are not draws; they share
    # the registered drive seed 101 among themselves and with the FULL seed 101, each on its own target)
    draw_seeds = {int(s) for s in ((cplan.get("data_seeds") or {}).get("draw_units") or {})}
    clash = [d["id"] for d in drives if int((d["job"].get("env") or {}).get("DCLM_SEED", -1)) in draw_seeds]
    if clash:
        raise SystemExit(f"forecast drives share a data seed with a draw of the confirm plan (MF-3): {clash}")
    os.makedirs(out_dir, exist_ok=True)
    units_path = os.path.join(out_dir, "forecast_units.json")
    probes_dir = os.path.join(out_dir, "probes")
    for u in units:                 # hashed after the canonical round trip (floats at 9 significant digits)
        u["probe_sha256"] = _sha_obj(json.loads(FIO.canonical_json_bytes(probe_unit_record(u)).decode("utf-8")))
    base_info = {}
    for b in sorted(bases):
        tmpl = _template(jobs_all, b, "muon")
        base_info[b] = {"arch": ARCH_OF_BASE[b], "experiment": (tmpl or {}).get("experiment"),
                        "checkpoint_var": f"ckpt_{b}", "checkpoint": (cq.get("vars") or {}).get(f"ckpt_{b}"),
                        "sets": {c: ((cplan.get("sets") or {}).get(b) or next(iter(cplan["sets"].values())))[set_name(c, code_stream)]
                                 for c in CORPORA},
                        "family": FAMILY[ARCH_OF_BASE[b]]}
    block = int(next((j.get("seq", 4096) for j in cq["jobs"] if j.get("kind") == "train"), 4096))
    doc = {"format": FORMAT_UNITS, "prereg": "preregistration v1.3 (version 3) sec. 5.1-5.4", "block": block,
           "confirm_plan": {"path": confirm_plan, "sha256": _sha_file(confirm_plan), "queue": cqp, "queue_sha256": _sha_file(cqp)},
           "fit_plans": fplans, "code_stream": code_stream, "bases": base_info,
           "targets": [{k: v for k, v in t.items() if not k.startswith("_")} for t in targets],
           "fit_runs": [{k: v for k, v in f.items() if not k.startswith("_")} for f in fits],
           "excluded": excluded + fit_excl + [{"run_id": None, "role": "revert evaluations", "reason": EXCLUDED_EVALS}],
           "units": units, "counts": unit_counts(units, targets), "drives": drives,
           "ro_check": {"max_abs_diff_vs_plan": max(ro_dev) if ro_dev else None, "n": len(ro_dev)},
           "tables": tab_files, "notes": notes,
           "probe": {"windows_per_corpus": PROBE_WINDOWS, "grad_windows": GRAD_WINDOWS, "haar_seed_base": HAAR_SEED_BASE,
                     "dir": probes_dir, "routing": "pinned (theta0 selections and gates)"},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ub = FIO.canonical_json_bytes(doc)
    with open(units_path, "wb") as fh:
        fh.write(ub)
    with open(units_path + ".sha256", "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{FIO.sha256_bytes(ub)}  {os.path.basename(units_path)}\n")
    # the stage-C input queue: drive runs, then one probe job per base
    est_probe = {}
    for b in sorted(bases):
        slow = _TP["eval_moe"] / _TP[f"eval_{ARCH_OF_BASE[b]}"]
        est_probe[b] = round(sum((_PROBE_DIRECTIONS * len(u["old_corpora"]) * PROBE_WINDOWS / 24.0 * _PROBE_S * slow + 3.0)
                                 / 3600.0 for u in units if u["base"] == b), 4)
    jobs = [d["job"] for d in drives] + [
        {"id": f"forecast_probe_{b}", "kind": "cmd", "est_h": est_probe[b],
         "argv": ["{python}", "forecast_probe.py", "run", "--units", units_path, "--base", b, "--out", probes_dir],
         "env": {"NESTOR_FREEZE_ROUTERS": "1"}} for b in sorted(bases)]
    queue = {"_comment": "stage-C forecast inputs (prereg v1.3 sec. 5.3): KSD drive runs + theta0 probes; "
                         "built by forecasts.py units", "env": cq.get("env", {}), "vars": cq.get("vars", {}),
             "logs": "logs/forecast", "state": "results/queue_state_forecast.json", "jobs": jobs}
    qpath = os.path.join(out_dir, "queue_forecast_inputs.json")
    with open(qpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(queue, fh, indent=1)
    doc["_files"] = {"units": units_path, "queue": qpath, "units_sha256": FIO.sha256_bytes(ub)}
    de_h = round(sum((_PROBE_DIRECTIONS * len(u["old_corpora"]) * PROBE_WINDOWS / 24.0 * _PROBE_S + 3.0) / 3600.0
                     for u in units if "DE" in u["blocks"]), 4)
    doc["_est_h"] = {"drives": round(sum(d["job"]["est_h"] for d in drives), 4), "probes": round(sum(est_probe.values()), 4),
                     "probes_c1de_in_plan_block": de_h, "registered": REGISTERED_INPUT_H,
                     "note": "probes: registered formula (2 antithetic directions x old corpora x 8/24 x the G0-7 time "
                             "per direction and corpus + 3 s gradient; C3 code/math units have 2 old corpora); drives "
                             "evaluate the three heads at steps 0 and 8 (registered: 3 evaluations, +3 per drive)"}
    return doc


def load_units(path: str) -> dict:
    b = open(path, "rb").read()
    side = path + ".sha256"
    if os.path.exists(side):
        want = open(side, encoding="ascii").read().split()[0]
        if want != FIO.sha256_bytes(b):
            raise SystemExit(f"{path}: content differs from its .sha256 sidecar (units changed after stage C step 1)")
    doc = json.loads(b.decode("utf-8"))
    doc["_sha256"] = FIO.sha256_bytes(b)
    return doc


# ============================================================================= stage C step 3: fit + forecasts
FORECASTER_DEF = {
    "RO": "a RO_c(S): theta0 routing statistic (evaluation half) of corpus c on S",
    "QE": "a Q_E,c(S,T): (KL+ + KL-)/(2 eps^2) along u_S on 8 windows of c, routing pinned to theta0",
    "M1": "a RO_c(S) Q_E,c(S,T)",
    "M2": "a RO_c(S) Q_E,c(S,T) + b gamma_S,c; gamma_S,c = (dCE+ - dCE-)/(2 eps) along u_S (pinned)",
    "SVC": "a sum over the expert matrices m of S of tr(dW_m Sigma_c,m dW_m^T), dW_m = u_S on m, Sigma from theta0 inputs",
    "TF": "a |S| / N_trainable (trainable parameter elements over those of the FULL recipe)",
    "RND": "a (L(theta0 + eps h) + L(theta0 - eps h) - 2 L(theta0)) / (2 eps^2), h Haar with ||h_m|| = ||u_S,m|| "
           "(seed 9000 + unit index; pinned)",
    "KSD": "a F_c(8 steps of E-ALL on T) RO_c(S)",
}


def _paired_subset(base: dict, run: dict) -> float:
    """Mean CE difference over the run file's windows, paired by grid index with the (larger) base file."""
    ib = {int(i): k for k, i in enumerate(base["index"])}
    rows = []
    for k, i in enumerate(run["index"]):
        j = ib.get(int(i))
        if j is None:
            raise ValueError(f"window {int(i)} of {run.get('path')} not in {base.get('path')}")
        if int(base["hash"][j]) != int(run["hash"][k]) or int(base["n_tok"][j]) != int(run["n_tok"][k]):
            raise ValueError(f"window {int(i)}: token content differs ({run.get('path')})")
        rows.append(float(run["ce_sum"][k] / run["n_tok"][k] - base["ce_sum"][j] / base["n_tok"][j]))
    return float(np.mean(rows))


def _drive_F(U: dict, states: list[str]) -> tuple[dict, list[dict]]:
    import gate_common as GC

    plan = {"runs": [{"id": d["id"], "eval_dir": "eval_main"} for d in U["drives"]]}
    idx = GC.RunIndex(plan, states)
    out, files = {}, []
    for d in U["drives"]:
        out[d["id"]] = {}
        if idx.status(d["id"]) != "done":
            continue
        for c in CORPORA:
            sn = set_name(c, U["code_stream"])
            p0, p8 = idx.interim_path(d["id"], sn, 0), idx.interim_path(d["id"], sn, DRIVE_STEPS)
            if not (p0 and p8 and os.path.exists(p0) and os.path.exists(p8)):
                continue
            out[d["id"]][c] = _paired_subset(GC.load_eval(p0), GC.load_eval(p8))
            files += [{"path": p, "sha256": _sha_file(p), "role": f"KSD drive {d['id']} {sn} head"} for p in (p0, p8)]
    return out, files


def _fit_F(U: dict, states: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    import gate_common as GC

    plans = {fp["path"]: _load(fp["path"]) for fp in U["fit_plans"]}
    idxs = {p: GC.RunIndex(doc, states) for p, doc in plans.items()}
    obs, files, missing = [], [], []
    for fr in U["fit_runs"]:
        plan, idx = plans[fr["plan_path"]], idxs[fr["plan_path"]]
        if idx.status(fr["run_id"]) != "done":
            missing.append(f"{fr['run_id']}: status {idx.status(fr['run_id'])}")
            continue
        bfiles = (plan.get("base_eval_files") or {}).get(fr["base"]) or {}
        for ob in fr["observations"]:
            if ob["kind"] == "final":
                pr, pb = idx.eval_path(fr["run_id"], ob["set"]), bfiles.get(ob["set"])
            else:
                root = idx.root(fr["run_id"])
                pr = os.path.join(root, "interim", f"step{ob['step']:03d}__{ob['set']}__full.npz")
                p0 = os.path.join(root, "interim", f"step000__{ob['set']}__full.npz")
                pb = p0 if os.path.exists(p0) else bfiles.get(ob["set"])
            if not (pr and pb and os.path.exists(pr) and os.path.exists(pb)):
                missing.append(f"{fr['run_id']} step {ob['step']} {ob['set']}: evaluation file missing")
                continue
            f = GC.forgetting(GC.load_eval(pb), GC.load_eval(pr))
            obs.append({"run_id": fr["run_id"], "unit": fr["unit"], "arch": fr["arch"], "step": ob["step"],
                        "corpus": ob["corpus"], "kind": ob["kind"], "F": f["F"], "SE": f["SE"], "W": f["W"]})
            files += [{"path": pr, "sha256": _sha_file(pr), "role": f"fit run {fr['run_id']} {ob['set']} step {ob['step']}"},
                      {"path": pb, "sha256": _sha_file(pb), "role": f"theta0 {fr['base']} {ob['set']}"}]
    return obs, files, missing


def _inputs(u: dict, c: str, probe: dict | None, dF: dict) -> dict[str, tuple[Any, str | None]]:
    """Forecaster inputs of unit u on corpus c: {code: (x, reason_if_undefined)}; M2: ((RO Q_E, gamma), reason)."""
    ro = u["ro"].get(c)
    pc = ((probe or {}).get("by_corpus") or {}).get(c) or {}
    miss_p = "theta0 probe of the unit missing" if probe is None else None
    el = (probe or {}).get("elements") or {}

    def val(v, why):
        return (float(v), None) if v is not None and math.isfinite(float(v)) else (None, why)

    qe = val(pc.get("QE"), miss_p or f"Q_E undefined on {c}")
    ga = val(pc.get("gamma"), miss_p or f"gamma undefined on {c}")
    out = {"RO": val(ro, f"RO undefined: corpus {c} missing from the theta0 tables"), "QE": qe}
    out["M1"] = (out["RO"][0] * qe[0], None) if out["RO"][0] is not None and qe[0] is not None else (None, out["RO"][1] or qe[1])
    out["M2"] = ((out["M1"][0], ga[0]), None) if out["M1"][0] is not None and ga[0] is not None else (None, out["M1"][1] or ga[1])
    out["SVC"] = val(pc.get("SVC"), miss_p or f"SVC undefined on {c}")
    tf = (el.get("trainable") / el.get("full_trainable")) if el.get("trainable") is not None and el.get("full_trainable") else None
    out["TF"] = val(tf, miss_p or "trainable-element counts missing")
    out["RND"] = val(pc.get("RND"), miss_p or f"RND undefined on {c}")
    f8 = dF.get(u["drive"], {}).get(c)
    out["KSD"] = (f8 * out["RO"][0], None) if f8 is not None and out["RO"][0] is not None else \
        (None, f"KSD drive {u['drive']} has no F on {c}" if f8 is None else out["RO"][1])
    return out


def fit_scalars(obs: list[dict], xs: dict) -> dict:
    """Per architecture x forecaster: one-scalar log-LS (sec. 5.4) and M2 by OLS on F without intercept."""
    sc: dict[str, dict] = {}
    for arch in ("moe", "nestor"):
        for fc in FIO.FORECASTERS:
            rows = [(o, xs[(o["unit"], o["corpus"])][fc]) for o in obs if o["arch"] == arch]
            ex = Counter()
            if fc != "M2":
                lr = []
                for o, (x, why) in rows:
                    if x is None:
                        ex["x undefined"] += 1
                    elif o["F"] <= 0:
                        ex["F <= 0"] += 1
                    elif x <= 0:
                        ex["x <= 0"] += 1
                    else:
                        lr.append(math.log(o["F"]) - math.log(x))
                ent = {"n": len(lr), "excluded": dict(ex)}
                if lr:
                    la = float(np.mean(lr))
                    ent["a"] = math.exp(la)
                    ent["resid_sd_logF"] = float(np.std(np.array(lr) - la, ddof=1)) if len(lr) > 1 else None
                sc[f"{arch}|{fc}"] = ent
                continue
            X, Y = [], []
            for o, (x, why) in rows:
                if x is None:
                    ex["x undefined"] += 1
                else:
                    X.append(x)
                    Y.append(o["F"])
            ent = {"n": len(X), "excluded": dict(ex)}
            if len(X) >= 2:
                A = np.array(X, dtype=np.float64)
                y = np.array(Y, dtype=np.float64)
                s11, s12, s22 = float((A[:, 0] ** 2).sum()), float((A[:, 0] * A[:, 1]).sum()), float((A[:, 1] ** 2).sum())
                t1, t2 = float((A[:, 0] * y).sum()), float((A[:, 1] * y).sum())
                det = s11 * s22 - s12 * s12                    # 2x2 normal equations (no LAPACK: torch-clash safe)
                if abs(det) > 1e-300 * max(1.0, s11 * s22) and det != 0.0:
                    a, b = (t1 * s22 - t2 * s12) / det, (s11 * t2 - s12 * t1) / det
                    ent.update({"a": a, "b": b})
                    yh = A[:, 0] * a + A[:, 1] * b
                    ok = (y > 0) & (yh > 0)
                    r = np.log(y[ok]) - np.log(yh[ok])
                    ent["resid_sd_logF"] = float(np.std(r, ddof=1)) if ok.sum() > 1 else None
                    ent["n_logF"] = int(ok.sum())
                else:
                    ent["collinear"] = True
            sc[f"{arch}|{fc}"] = ent
    return sc


def cmd_make(units_path: str, probes_dir: str | None, states: list[str], out_dir: str, *, allow_missing: bool = False,
             allow_mock: bool = False, b3_addendum: bool = False, overwrite: bool = False,
             spec_path: str = "g1g2_spec.json") -> dict:
    U = load_units(units_path)
    probes_dir = probes_dir or U["probe"]["dir"]
    probes, pfiles, mock = {}, [], False
    for u in U["units"]:
        p = os.path.join(probes_dir, f"probe_u{u['index']:03d}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if d.get("unit_sha256") != u["probe_sha256"]:
            raise SystemExit(f"{p}: probe of a different unit definition (unit_sha256 mismatch)")
        mock = mock or bool(d.get("mock"))
        probes[u["index"]] = d
        pfiles.append({"path": p, "sha256": _sha_file(p), "role": f"theta0 probe unit {u['index']}"})
    if mock and not allow_mock:
        raise SystemExit("MOCK probe inputs: refusing to write registered forecasts (pass --allow-mock in tests only)")
    dF, dfiles = _drive_F(U, states)
    obs, ffiles, fmiss = _fit_F(U, states)
    missing = [f"unit {u['index']}: probe missing" for u in U["units"] if u["index"] not in probes]
    missing += [f"drive {d}: no F on {sorted(set(CORPORA) - set(v))}" for d, v in dF.items()
                if any(d == u["drive"] for u in U["units"]) and set(v) != set(CORPORA)]
    missing += fmiss
    if missing and not allow_missing:
        raise SystemExit(f"forecast inputs missing ({len(missing)}): " + "; ".join(missing[:8]) +
                         (" ..." if len(missing) > 8 else "") + " (--allow-missing writes NA with the reason)")
    units = {u["index"]: u for u in U["units"]}
    xs = {}
    for u in U["units"]:
        for c in CORPORA:
            if c in u["old_corpora"]:
                xs[(u["index"], c)] = _inputs(u, c, probes.get(u["index"]), dF)
    sc = fit_scalars(obs, xs)
    rows, na, na_rows = [], defaultdict(set), Counter()
    tsel = [t for t in U["targets"] if t["block"] == "B3"] if b3_addendum else list(U["targets"])
    for t in tsel:
        u = units[t["unit"]]
        for s in t["steps"]:
            for c in t["old_corpora"]:
                for fc in FIO.FORECASTERS:
                    x, why = xs[(u["index"], c)][fc]
                    ent = sc[f"{u['arch']}|{fc}"]
                    v = None
                    if x is None:
                        na[fc].add(why)
                        na_rows[fc] += 1
                    elif "a" not in ent:
                        na_rows[fc] += 1
                        na[fc].add(f"scalar of {u['arch']}|{fc} not identified (no fit observation with F > 0 and x > 0)"
                                   if fc != "M2" else f"M2 scalars of {u['arch']} not identified")
                    else:
                        v = ent["a"] * x[0] + ent["b"] * x[1] if fc == "M2" else ent["a"] * x
                    rows.append({"run_id": t["run_id"], "block": t["block"], "base": t["base"], "target": t["target"],
                                 "step": s, "eval_corpus": c, "forecaster": fc, "quantity": "F", "value": v})
    sha_code = {n: _sha_file(os.path.join(HERE, n)) for n in ("forecast_io.py", "forecasts.py", "forecast_probe.py")}
    inputs = ([{"path": units_path, "sha256": U["_sha256"], "role": "forecast units (stage C step 1)"}] +
              [dict(f, role=f["role"]) for f in U["tables"]] + pfiles + dfiles + ffiles +
              [{"path": U["confirm_plan"]["path"], "sha256": _sha_file(U["confirm_plan"]["path"]), "role": "confirm plan"}] +
              [{"path": fp["path"], "sha256": _sha_file(fp["path"]), "role": f"{fp['kind']} plan"} for fp in U["fit_plans"]] +
              [{"path": s, "sha256": _sha_file(s), "role": "queue state"} for s in states if os.path.exists(s)])
    meta = {"prereg": "preregistration v1.3 (version 3) sec. 5", "addendum": "third base (sec. 5.6)" if b3_addendum else None,
            "targets": {t["run_id"]: {"steps": t["steps"]} for t in tsel},
            "fit": {"runs": sorted({o["run_id"] for o in obs}), "observations": obs, "scalars": sc,
                    "rule": "log a = mean(log F - log x) over F > 0, x > 0; M2: OLS on F without intercept",
                    "missing": fmiss},
            "inputs": inputs, "code": sha_code, "na_rules": {k: "; ".join(sorted(v)) for k, v in sorted(na.items())},
            "excluded": U["excluded"], "forecasters": FORECASTER_DEF,
            "units": {"file": units_path, "sha256": U["_sha256"], "n": len(U["units"]), "counts": U["counts"],
                      "run_unit": {t["run_id"]: t["unit"] for t in tsel}},
            "spec_sha256": _sha_file(spec_path), "mock": mock, "missing_inputs": missing,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(out_dir, exist_ok=True)
    stem = "forecasts_v13_b3" if b3_addendum else "forecasts_v13"
    cp, mp = os.path.join(out_dir, f"{stem}.csv"), os.path.join(out_dir, f"{stem}_meta.json")
    new_csv = FIO.canonical_csv_bytes(rows)
    if os.path.exists(cp) and open(cp, "rb").read() != new_csv and not overwrite:
        raise SystemExit(f"{cp} exists with different forecasts: a hashed forecast is final (pass --overwrite before the freeze)")
    h = FIO.write_forecasts(rows, meta, cp, mp)
    ok, errs, h2 = FIO.verify(cp, mp)
    if not ok:
        raise SystemExit(f"written files do not verify: {errs[:5]}")
    return {"csv": cp, "meta": mp, "csv_sha256": h2["csv_sha256"], "meta_sha256": h2["meta_sha256"], "n_rows": len(rows),
            "n_targets": len(tsel), "na": dict(na_rows), "scalars": sc, "missing": missing,
            "mock": mock}


# ============================================================================= mock inputs (tests / mock chain only)
def cmd_mock_inputs(units_path: str, states_out: str, seed: int = 0) -> dict:
    """Synthetic probe files (flagged mock) and drive interim heads for the mock chain; make refuses them without
    --allow-mock."""
    import gate_common as GC

    U = load_units(units_path)
    rng = np.random.default_rng(seed)
    pdir = U["probe"]["dir"]
    os.makedirs(pdir, exist_ok=True)
    for u in U["units"]:
        bc = {}
        for c in u["old_corpora"]:
            ro = u["ro"].get(c) or 0.0
            q = 0.05 * (1.0 + rng.random()) * (ro + 0.05) * (3.0 if u["subset"] != "SPEC" else 1.0)
            bc[c] = {"QE": q, "QE_se": 0.1 * q, "gamma": float(rng.normal(0.0, 0.01)), "gamma_se": 0.001,
                     "RND": 0.3 * q, "RND_se": 0.03 * q, "SVC": 2.0 * (ro + 0.01)}
        full = 100_000_000
        el = {"trainable": full if u["subset"] == "FULL" else int(full * u["n_keys"] / max(u["n_keys_family"], 1) * 0.6),
              "full_trainable": full}
        doc = {"unit": u["index"], "unit_sha256": u["probe_sha256"], "mock": True, "by_corpus": bc, "elements": el}
        with open(os.path.join(pdir, f"probe_u{u['index']:03d}.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1)
    state = json.load(open(states_out, encoding="utf-8")) if os.path.exists(states_out) else {}
    plan_bases = {}
    for fp in U["fit_plans"]:
        for b, files in (fp.get("base_eval_files") or {}).items():
            plan_bases.setdefault(b, files)
    for d in U["drives"]:
        root = os.path.join("outputs", "forecast_mock", d["id"])
        os.makedirs(os.path.join(root, "interim"), exist_ok=True)
        for c in CORPORA:
            sn = set_name(c, U["code_stream"])
            src = (plan_bases.get(d["base"]) or plan_bases.get("moe_b1") or {}).get(sn)
            if not src or not os.path.exists(src):
                continue
            z = GC.load_eval(src)
            k = slice(0, HEAD_WINDOWS)
            F = 0.02 * (1.0 + rng.random())
            for step, add in ((0, 0.0), (DRIVE_STEPS, F)):
                np.savez(os.path.join(root, "interim", f"step{step:03d}__{sn}.npz"), index=z["index"][k], hash=z["hash"][k],
                         ce_sum=z["ce_sum"][k] + add * z["n_tok"][k], n_tok=z["n_tok"][k],
                         meta_json=json.dumps({"mock": True, "variant": "ckpt", "step": step}))
        state[d["id"]] = {"status": "done", "root": root, "mock": True}
    with open(states_out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(state, fh, indent=1)
    return {"probes": len(U["units"]), "drives": len(U["drives"]), "state": states_out}


# ============================================================================= scoring (sec. 5.7; after all runs)
def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    r[o] = np.arange(len(a), dtype=np.float64)
    for v in np.unique(a):                                # average ranks for ties
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    dx, dy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    return float((dx * dy).sum() / den) if den > 0 else None


def cmd_score(csv_path: str, meta_path: str, units_path: str, states: list[str]) -> dict:
    """Descriptive scores of the hashed forecasts (sec. 5.7): R2_oos on log F against the G1/G2 mean of log F per
    (architecture, corpus), Spearman rho, median |log(F_hat / F)|, calibration slope of log F on log F_hat, pooled and
    per block; R2_within on the crossover contrasts (C1, B3, DR, NB, DE = C1-DE; sec. 5.7 of version 6), pooled and per
    block, with I_d = [log F_code - log F_math](A) - [log F_code - log F_math](B) for forecasts and outcomes alike."""
    import gate_common as GC

    ok, errs, h = FIO.verify(csv_path, meta_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    U = load_units(units_path)
    rows, _ = FIO.parse_csv_bytes(open(csv_path, "rb").read())
    cplan = _load(U["confirm_plan"]["path"])
    idx = GC.RunIndex(cplan, states)
    bfiles = cplan.get("base_eval_files") or {}
    meas = {}
    tinfo = {t["run_id"]: t for t in U["targets"]}
    runs_by = {r["id"]: r for r in cplan.get("runs", [])}
    not_scorable = {}                     # P-16: forecast steps after a run's stop step have no state to score against
    for rid, t in tinfo.items():
        if idx.status(rid) != "done":
            continue
        stop = int((runs_by.get(rid) or {}).get("stop_after") or 0)
        fin = stop or max(t["steps"])     # the step of the run's final checkpoint (its eval_main files)
        for s in t["steps"]:
            if stop and s > stop:
                not_scorable[(rid, s)] = (f"run stopped after step {stop} of its {int((runs_by.get(rid) or {}).get('steps', 96))}"
                                          f"-step schedule (P-16): no step-{s} state")
                continue
            for c in t["old_corpora"]:
                sn = set_name(c, U["code_stream"])
                if s == fin:
                    pr, pb = idx.eval_path(rid, sn), (bfiles.get(t["base"]) or {}).get(sn)
                else:
                    root = idx.root(rid)
                    pr = os.path.join(root, "interim", f"step{s:03d}__{sn}__full.npz")
                    p0 = os.path.join(root, "interim", f"step000__{sn}__full.npz")
                    pb = p0 if os.path.exists(p0) else (bfiles.get(t["base"]) or {}).get(sn)
                if pr and pb and os.path.exists(pr) and os.path.exists(pb):
                    meas[(rid, s, c)] = GC.forgetting(GC.load_eval(pb), GC.load_eval(pr))["F"]
    bench = defaultdict(list)
    for o in meta["fit"]["observations"]:
        if o["F"] > 0:
            bench[(o["arch"], o["corpus"])].append(math.log(o["F"]))
    out = {"prospective": ok, "verify_errors": errs, "csv_sha256": h["csv_sha256"], "meta_sha256": h["meta_sha256"],
           "by_forecaster": {}}
    ns_rows = [r for r in rows if (r["run_id"], int(r["step"])) in not_scorable]
    out["not_scorable"] = {"n_rows": len(ns_rows),
                           "by_block": {b: sum(1 for r in ns_rows if r["block"] == b) for b in sorted({r["block"] for r in ns_rows})},
                           "targets": [{"run_id": k[0], "step": k[1], "status": "not_scorable", "reason": v}
                                       for k, v in sorted(not_scorable.items())],
                           "rule": "rows at a step after the run's stop step are not scored (P-16; plan diff)"}
    for fc in FIO.FORECASTERS:
        res = {}
        for blk in ["pooled"] + sorted({r["block"] for r in rows}):
            pts = []
            for r in rows:
                if r["forecaster"] != fc or r["value"] == "NA" or (blk != "pooled" and r["block"] != blk):
                    continue
                if (r["run_id"], int(r["step"])) in not_scorable:      # listed in out['not_scorable'], never scored
                    continue
                F = meas.get((r["run_id"], int(r["step"]), r["eval_corpus"]))
                fh = float(r["value"])
                arch = ARCH_OF_BASE[r["base"]]
                if F is not None and F > 0 and fh > 0 and bench.get((arch, r["eval_corpus"])):
                    pts.append((math.log(fh), math.log(F), float(np.mean(bench[(arch, r["eval_corpus"])]))))
            if len(pts) < 3:
                res[blk] = {"n": len(pts)}
                continue
            lf, lF, lb = (np.array(v) for v in zip(*pts))
            sse, sst = float(((lF - lf) ** 2).sum()), float(((lF - lb) ** 2).sum())
            dx = lf - lf.mean()
            slope = float((dx * (lF - lF.mean())).sum() / (dx * dx).sum()) if float((dx * dx).sum()) > 0 else None
            res[blk] = {"n": len(pts), "R2_oos": 1.0 - sse / sst if sst > 0 else None, "spearman": _corr(_rank(lf), _rank(lF)),
                        "median_abs_log_ratio": float(np.median(np.abs(lf - lF))), "calibration_slope": slope}
        out["by_forecaster"][fc] = res
    # R2_within on the crossover contrasts
    fcv = {(r["run_id"], r["eval_corpus"], r["forecaster"]): (None if r["value"] == "NA" else float(r["value"])) for r in rows}
    pairs = defaultdict(dict)
    for rid, t in tinfo.items():
        if t["block"] in WITHIN_BLOCKS and t.get("side") in ("A", "B"):
            pairs[(t["block"], t["base"], t.get("d"))][t["side"]] = t
    def r2w(I, Ih):
        if len(I) < 3:
            return {"n": len(I)}
        Ia, Iha = np.array(I), np.array(Ih)
        sst = float(((Ia - Ia.mean()) ** 2).sum())
        return {"n": len(I), "R2_within": 1.0 - float(((Ia - Iha) ** 2).sum()) / sst if sst > 0 else None}

    rw = {}
    for fc in FIO.FORECASTERS:
        I, Ih, blk_of = [], [], []
        for key, ab in pairs.items():
            if set(ab) != {"A", "B"}:
                continue
            s = {sd: ab[sd]["steps"][-1] for sd in ab}
            vals = {}
            for sd in ("A", "B"):
                for c in ("code", "math"):
                    vals[(sd, c)] = (meas.get((ab[sd]["run_id"], s[sd], c)), fcv.get((ab[sd]["run_id"], c, fc)))
            if all(v[0] and v[0] > 0 and v[1] and v[1] > 0 for v in vals.values()):
                lg = lambda sd, c, k: math.log(vals[(sd, c)][k])
                I.append((lg("A", "code", 0) - lg("A", "math", 0)) - (lg("B", "code", 0) - lg("B", "math", 0)))
                Ih.append((lg("A", "code", 1) - lg("A", "math", 1)) - (lg("B", "code", 1) - lg("B", "math", 1)))
                blk_of.append(key[0])
        rw[fc] = r2w(I, Ih)
        rw[fc]["by_block"] = {b: r2w([x for x, k in zip(I, blk_of) if k == b], [x for x, k in zip(Ih, blk_of) if k == b])
                              for b in sorted(set(blk_of))}
    out["R2_within"] = rw
    out["note"] = "descriptive (sec. 5.7); all eight forecasters in one specification table, no selection before scoring"
    return out


# ============================================================================= C5 French (v1.2 sec. 12; unchanged)
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=float)


def content_sha(doc: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in doc.items() if k not in ("sha256", "created")}).encode()).hexdigest()


# ----------------------------------------------------------------------------- features
def _units(run: dict) -> int | None:
    sp = run.get("subset") or {}
    return sp.get("n_units")


def feat_ro(run: dict, corpus: str, ctx: dict) -> float | None:
    if run.get("role") in ("full", "dclm_cont", "freeze"):
        return 1.0                                   # every expert trains (FULL): RO = 1 by definition
    ro = (run.get("subset") or {}).get("ro") or {}
    return ro.get(corpus)


def feat_trainable_fraction(run: dict, corpus: str, ctx: dict) -> float | None:
    if run.get("role") in ("full", "dclm_cont", "freeze"):
        return 1.0
    n, tot = _units(run), (run.get("subset") or {}).get("n_units_family")
    return (n / tot) if n and tot else None


C5_FORECASTERS = {"ro": feat_ro, "trainable_fraction": feat_trainable_fraction}


# ----------------------------------------------------------------------------- measured F of the fit runs
def measured_F(plan: dict, states: list[str]) -> list[dict]:
    """F per (run, corpus) for every finished train run of a G1/G2 plan (theta_0 files from plan['base_eval_files'])."""
    import gate_common as GC

    idx = GC.RunIndex(plan, states)
    out = []
    for r in plan.get("runs", []):
        if idx.status(r["id"]) != "done":
            continue
        base_files = (plan.get("base_eval_files") or {}).get(r["base"]) or {}
        for corpus in OLD_CORPORA:
            name = plan.get("g0", {}).get("code_stream", "code") if corpus == "code" else corpus
            pb, pr = base_files.get(name), idx.eval_path(r["id"], name)
            if not (pb and pr and os.path.exists(pb) and os.path.exists(pr)):
                continue
            f = GC.forgetting(GC.load_eval(pb), GC.load_eval(pr))
            out.append({"run": r["id"], "plan": plan.get("kind"), "role": r["role"], "arch": r["arch"], "base": r["base"],
                        "corpus": corpus, "F": f["F"], "SE": f["SE"], "subset": r.get("subset_meta")})
    return out


def c5_fit(fit_rows: list[dict], name: str) -> dict:
    """Placeholder fit rule: a[arch|corpus] = exp(mean(log F - log x)) over fit runs with F > 0 and x > 0."""
    feat = C5_FORECASTERS[name]
    acc: dict = {}
    for r in fit_rows:
        run = {"role": r["role"], "subset": r.get("subset") or {}}
        x = feat(run, r["corpus"], {})
        if x is None or x <= 0 or r["F"] is None or r["F"] <= 0:
            continue
        acc.setdefault(f"{r['arch']}|{r['corpus']}", []).append(math.log(r["F"]) - math.log(x))
    return {k: {"a": math.exp(float(np.mean(v))), "n": len(v)} for k, v in sorted(acc.items())}



def verify_json(path: str) -> bool:
    """C5 French file (v1.2 sec. 12): the content sha256 still matches (canonical JSON without 'sha256'/'created')."""
    doc = json.load(open(path, encoding="utf-8"))
    return content_sha(doc) == doc.get("sha256")


# ----------------------------------------------------------------------------- C5 French protocol (prereg sec. 12)
def c5_french(plan_path: str, drive: str, states: list[str], out: str, fit_plans: list[str] | None = None,
              log_path: str = os.path.join("logs", "c5_forecast_hashes.log")) -> dict:
    """Forecasts for the 5 French runs x 3 corpora BEFORE any French run (prereg sec. 12 items 2-3): scalars fitted on
    every non-French run available now (confirm runs already finished + the given G1/G2 plans); forecasters 'ro',
    'trainable_fraction' and 'k_step' (8: F_c of the 8-step E-ALL drive on French x RO_c(S)); the file's sha256 is
    appended with a timestamp to the project log (logs/c5_forecast_hashes.log)."""
    plan = json.load(open(plan_path, encoding="utf-8"))
    rows = [r for r in measured_F(plan, states) if r.get("run") != drive and "french" not in r["run"]]
    for fp in fit_plans or []:
        rows += measured_F(json.load(open(fp, encoding="utf-8")), states)
    drive_F = {r["corpus"]: r["F"] for r in measured_F(plan, states) if r["run"] == drive}
    if len(drive_F) < len(OLD_CORPORA):
        raise SystemExit(f"k-step drive {drive}: F missing on {sorted(set(OLD_CORPORA) - set(drive_F))}")
    scalars = {name: c5_fit(rows, name) for name in C5_FORECASTERS}
    table = []
    french = [r for r in plan["runs"] if r.get("block") == "C5" and r["role"] in ("c5_full", "c5_pair")]
    for r in french:
        for corpus in OLD_CORPORA:
            fc = {}
            for name, feat in C5_FORECASTERS.items():
                x = feat(r, corpus, {})
                a = (scalars[name].get(f"{r['arch']}|{corpus}") or {}).get("a")
                fc[name] = (a * x) if (a is not None and x is not None) else None
            ro_c = feat_ro(r, corpus, {})
            fc["k_step"] = (drive_F[corpus] * ro_c) if ro_c is not None else None
            table.append({"run": r["id"], "corpus": corpus, "forecasts": fc})
    doc = {"kind": "C5 French forecasts (prereg v1.2 sec. 12)", "protocol": C5_PROTOCOL + "; k_step = F_drive_c x RO_c(S)",
           "drive": {"run": drive, "F": drive_F}, "fit_runs": rows, "scalars": scalars, "forecasts": table,
           "french_runs": [r["id"] for r in french], "plan_content_sha256": plan.get("sha256"),
           "code_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()}
    doc["sha256"] = content_sha(doc)
    doc["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if os.path.exists(out):
        old = json.load(open(out, encoding="utf-8"))
        if old.get("sha256") != doc["sha256"]:
            raise SystemExit(f"{out} exists with different forecasts (sha256 {old.get('sha256')}): a hashed forecast is final")
        return old
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical(doc) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"{doc['created']} {doc['sha256']} {os.path.abspath(out)}\n")
    return doc


def c5_hash_gate(state: dict, prm: dict) -> dict:
    """run_queue gate (kind 'python'): the French forecasts exist, verify, and their hash is in the project log."""
    f, lg = prm["file"], prm.get("log", os.path.join("logs", "c5_forecast_hashes.log"))
    if not os.path.exists(f):
        return {"pass": False, "reason": f"{f} missing"}
    doc = json.load(open(f, encoding="utf-8"))
    posted = os.path.exists(lg) and doc.get("sha256", "-") in open(lg, encoding="utf-8").read()
    ok = verify_json(f) and posted
    return {"pass": bool(ok), "sha256": doc.get("sha256"), "verifies": verify_json(f), "posted": posted}




# ============================================================================= CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="prospective per-run forecasts (prereg v1.3 sec. 5; C5 French v1.2 sec. 12)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("units", help="stage C step 1: target runs, probe units, fit runs, drive runs + input queue")
    u.add_argument("--confirm-plan", required=True)
    u.add_argument("--fit-plans", required=True, help="comma list: g1/plan_g1.json,g2/plan_g2.json[,g1r/plan_g1r.json]")
    u.add_argument("--g0-config", required=True, help="comma list of G0 gate configs (routing tables; third base config)")
    u.add_argument("--out-dir", default=os.path.join("results", "forecast"))
    u.add_argument("--sgd-lr", type=float, default=None, help="G2-2 SGD rate for an SGD drive of a base without C2 SGD runs")
    u.add_argument("--allow-c3-deviation", action="store_true")
    m = sub.add_parser("make", help="stage C step 3: fit the scalars, write and verify the forecast files")
    m.add_argument("--units", required=True)
    m.add_argument("--probes", default=None)
    m.add_argument("--states", default="")
    m.add_argument("--out-dir", default="freeze")
    m.add_argument("--allow-missing", action="store_true")
    m.add_argument("--allow-mock", action="store_true")
    m.add_argument("--b3-addendum", action="store_true")
    m.add_argument("--overwrite", action="store_true")
    m.add_argument("--spec", default="g1g2_spec.json")
    v = sub.add_parser("verify")
    v.add_argument("csv")
    v.add_argument("meta")
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True)
    s.add_argument("--meta", required=True)
    s.add_argument("--units", required=True)
    s.add_argument("--states", default="")
    s.add_argument("--out", default=None)
    mk = sub.add_parser("mock-inputs", help="TESTS ONLY: synthetic probe + drive inputs (flagged mock)")
    mk.add_argument("--units", required=True)
    mk.add_argument("--state", default=os.path.join("results", "queue_state_forecast.json"))
    mk.add_argument("--seed", type=int, default=0)
    c5 = sub.add_parser("c5-french")
    c5.add_argument("--plan", required=True)
    c5.add_argument("--drive", required=True)
    c5.add_argument("--states", default="")
    c5.add_argument("--fit-plans", default="")
    c5.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    lst = lambda s_: [p for p in (s_ or "").split(",") if p]
    if a.cmd == "units":
        doc = cmd_units(a.confirm_plan, lst(a.fit_plans), lst(a.g0_config), a.out_dir, a.sgd_lr, a.allow_c3_deviation)
        c = doc["counts"]
        print(f"[forecasts] {len(doc['targets'])} target runs, {len(doc['fit_runs'])} fit runs, {len(doc['units'])} units "
              f"(MoE {c['moe']['total']} = {c['moe']['target']} target + {c['moe']['fit']} fit; NESTOR {c['nestor']['total']} = "
              f"{c['nestor']['target']} + {c['nestor']['fit']}; registered 108 + 11 at n_d = 5, C1-DE "
              f"{c['moe']['target_by_block'].get('DE', 0)} of 12 outside; n_d = {c['n_d']}), "
              f"{len(doc['drives'])} drive runs; est {doc['_est_h']} GPU-h")
        for d in c["deviations"]:
            print(f"[forecasts] count deviation: {d}")
        print(f"[forecasts] units {doc['_files']['units']} sha256 {doc['_files']['units_sha256']}; queue {doc['_files']['queue']}")
        return 0
    if a.cmd == "make":
        r = cmd_make(a.units, a.probes, lst(a.states), a.out_dir, allow_missing=a.allow_missing, allow_mock=a.allow_mock,
                     b3_addendum=a.b3_addendum, overwrite=a.overwrite, spec_path=a.spec)
        print(f"[forecasts] {r['n_rows']} rows for {r['n_targets']} target runs; NA rows {r['na']}; "
              f"{'MOCK INPUTS; ' if r['mock'] else ''}{len(r['missing'])} missing inputs")
        print(f"{r['csv_sha256']}  {r['csv']}\n{r['meta_sha256']}  {r['meta']}")
        return 0
    if a.cmd == "verify":
        ok, errs, h = FIO.verify(a.csv, a.meta)
        for e in errs:
            print("ERROR", e)
        print(f"{h['csv_sha256']}  {a.csv}\n{h['meta_sha256']}  {a.meta}\nRESULT: {'VALID' if ok else 'INVALID'}")
        return 0 if ok else 1
    if a.cmd == "score":
        res = cmd_score(a.csv, a.meta, a.units, lst(a.states))
        txt = json.dumps(res, indent=1, default=float)
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(txt + "\n")
        print(txt[:4000])
        return 0
    if a.cmd == "mock-inputs":
        print(f"[forecasts] MOCK inputs: {cmd_mock_inputs(a.units, a.state, a.seed)}")
        return 0
    doc = c5_french(a.plan, a.drive, lst(a.states), a.out, lst(a.fit_plans))
    print(f"[forecasts] C5 French: {len(doc['forecasts'])} forecasts, sha256 {doc['sha256']} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
