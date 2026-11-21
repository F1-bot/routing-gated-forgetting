"""Dose-response draws (preregistration v1.3 sec. 4.2; decision P-9) -- torch-free (numpy), CPU, before the freeze.

Reduced tilt through a LARGER POOL FRACTION f_low of the registered generator (crossover.py registered-draw, design c1,
target books, registered tolerance log 1.05, the C1 code stream), with nothing else changed:
  1. f_low: for f in {2/3, 3/4, 5/6} the G0-1 computation (g0_gate.gen_run: 1,000 accepted generator draws, the G0-1 seed)
     on MoE base 1 (G0 re-run statistics) gives the median evaluation-half P; f_low is the f whose median is closest to 0.5,
     the smaller f on a tie. The same f_low is used for base 2.
  2. Draws per MoE base: generator seeds 5100 + d, d = 1, 2, ... (crossover.reg_draw(design 'c1', d' = 100 + d) seeds
     REG_SEED_BASE['c1'] + 100 + d = 5100 + d); the first two draws whose SELECTION-half P_d lies in [0.35, 0.65] are
     accepted and the estimation-half P_d is the regressor, so acceptance does not select on the regressor (R2; prereg v1.3
     version 7 sec. 4.2); if fewer than two occur among the first 50 seeds, that base's dose level is not run
     and this is recorded.
Every candidate draw examined is recorded (prereg sec. 9.2), with its P_d and the per-layer overlap of A and B: for f > 1/2
the two pools overlap and the registered generator does not require disjoint subsets (observation, flagged for the
design; at the A/A limit f -> 1 the subsets are independent draws). The analysis model (P-14: E[I | P] = b1 P + b3 P^3,
no intercept) is fitted after the runs, not here.
Data seeds of the dose runs are not registered in v1.3; the C1 convention (data seed 7000 + d for generator 5000 + d) is
carried over as 7100 + d on MoE base 1 and,, 7150 + d on MoE base 2 (PROPOSED, recorded in the manifest):
both bases scan d = 1, 2, ... and may accept the same d, and the dose fit pools the draws of both bases as independent
units, so the seeds are base-specific as for C1 (MF-3); 7101-7150 and 7151-7200 cannot overlap (d <= 50).

  python dose_draws.py --gate-config g0/g0_gate_config.json --g0-record g0/gate_record.json \
      --out results/dose/dose_record.json --manifest confirm/specs/dose_draws.json [--n 1000]
Exit codes: 0 f_low chosen and >= 1 base has 2 accepted draws; 4 no base has a dose level; 2 inputs missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import crossover as CX  # noqa: E402
import g0_gate as GG  # noqa: E402

F_GRID = (2.0 / 3.0, 3.0 / 4.0, 5.0 / 6.0)
TARGET_P = 0.5
ACCEPT = (0.35, 0.65)
N_ACCEPT = 2
MAX_SEEDS = 50
DOSE_SEED_OFFSET = 100            # reg_draw d' = 100 + d  ->  seed 5000 + 100 + d = 5100 + d
DATA_SEED_BASE = 7100             # PROPOSED (C1 analogue: data 7000 + d for generator 5000 + d)
DATA_SEED_BASE_OFFSET = {"moe_b1": 0, "moe_b2": MAX_SEEDS}   # base-specific (MF-3 logic); d <= MAX_SEEDS
BASES = ("moe_b1", "moe_b2")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_f_low(rows: dict) -> float | None:
    """rows {f: median or None}: the f whose median P is closest to TARGET_P; ties -> smaller f (prereg sec. 4.2)."""
    ok = [(abs(m - TARGET_P), f) for f, m in rows.items() if m is not None and math.isfinite(m)]
    return min(ok)[1] if ok else None


def overlap_per_layer(A: list, B: list) -> list:
    return [len(set(a) & set(b)) for a, b in zip(A, B)]


def load_tables(cfg: dict, base: str):
    b = cfg["bases"][base]
    t = CX.reg_load_dir(b["dir"], b["family"], layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
    return t, CX.reg_k(t)


def f_grid(t, k: int, code: str, n_draws: int, log=print) -> dict:
    match = CX.reg_matched("c1", target="books")
    out = {}
    for f in F_GRID:
        v = GG.gen_run(t, "c1", f=f, target="books", code=code, match=match, tol=CX.REG_TOL, n_draws=n_draws, k=k)
        out[f"{f:.6f}"] = {kk: v.get(kk) for kk in ("f", "median", "median_selection", "n_pairs", "acceptance", "g0_2_pass",
                                                    "complete", "swap_convergence", "predicted")}
        if log:
            log(f"[dose] moe_b1 f={f:.4f}: median P {v.get('median')} (pairs {v.get('n_pairs')}, acceptance {v.get('acceptance')})")
    return out


def draws_for_base(t, k: int, code: str, f_low: float, log=print, base: str = "") -> dict:
    cands, accepted = [], []
    for d in range(1, MAX_SEEDS + 1):
        e = CX.reg_draw(t, "c1", DOSE_SEED_OFFSET + d, f=f_low, target="books", code=code, k=k, tol=CX.REG_TOL)
        e["reg_draw_index"], e["d"] = e["d"], d                       # registered index d of seed 5100 + d
        P = (e.get("predicted") or {}).get("P")                       # estimation half: the regressor (P-14)
        P_sel = None
        if e.get("status") == "ok":                                   # R2: acceptance on the selection half
            P_sel = float(CX.reg_predicted(t.sel, np.asarray(e["A"])[None], np.asarray(e["B"])[None], "c1",
                                           target="books", code=code)[0])
            e["selection_P"] = P_sel
        ok = e.get("status") == "ok" and P_sel is not None and ACCEPT[0] <= P_sel <= ACCEPT[1]
        row = {"d": d, "seed": e["seed"], "status": e.get("status"), "P_d": P, "P_selection": P_sel,
               "accepted": bool(ok and len(accepted) < N_ACCEPT)}
        if e.get("status") == "ok":
            row["overlap_per_layer"] = overlap_per_layer(e["A"], e["B"])
        cands.append(row)
        if row["accepted"]:
            e["dose_index"], e["data_seed"] = d, DATA_SEED_BASE + DATA_SEED_BASE_OFFSET.get(base, 0) + d
            e["data_seed_status"] = "proposed (not registered in v1.3); base-specific"
            e["base"] = base
            e["overlap_per_layer"] = row["overlap_per_layer"]
            accepted.append(e)
        if len(accepted) == N_ACCEPT:
            break
    run = len(accepted) == N_ACCEPT
    if log:
        log(f"[dose] f_low {f_low:.4f}: {len(accepted)} accepted of {len(cands)} examined -> {'run' if run else 'NOT run'}")
    return {"run": run, "examined": cands, "accepted": accepted,
            "reason": None if run else f"fewer than {N_ACCEPT} draws with P_d in {list(ACCEPT)} among the first {MAX_SEEDS} seeds"}


def build(gate_config: str, g0_record: str, n_draws: int = GG.G0_DRAWS, log=print) -> dict:
    cfg = GG.load_config(gate_config)
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    code = ((rec.get("decisions") or {}).get("code_stream") or {}).get("chosen") or "code"
    tabs = {b: load_tables(cfg, b) for b in BASES if b in cfg["bases"]}
    if "moe_b1" not in tabs:
        raise SystemExit("MoE base 1 is not in the G0 gate config")
    t1, k1 = tabs["moe_b1"]
    grid = f_grid(t1, k1, code, n_draws, log)
    f_low = choose_f_low({float(v["f"]): (v["median"] if v["g0_2_pass"] else None) for v in grid.values()})
    out = {"purpose": "prereg v1.3 sec. 4.2 (P-9): dose-response draws via a larger pool fraction f_low",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_stream": code, "target": "books", "tol": CX.REG_TOL,
           "f_grid": grid, "f_low": f_low, "target_P": TARGET_P, "accept": list(ACCEPT), "max_seeds": MAX_SEEDS,
           "seed_rule": "generator seed 5100 + d (crossover.reg_draw design c1, d' = 100 + d)",
           "data_seed_rule": "7100 + d (MoE base 1), 7150 + d (MoE base 2) (PROPOSED; not registered in v1.3; base-specific)",
           "accept_rule": "first two draws with SELECTION-half P_d in [0.35, 0.65]; the estimation-half P_d is the regressor (R2)",
           "analysis_model": "P-14: E[I | P] = b1 P + b3 P^3, no intercept (fitted after the runs; not part of this tool)",
           "bases": {}, "inputs": {"gate_config": {"path": os.path.abspath(gate_config), "sha256": _sha(gate_config)},
                                   "g0_record": {"path": os.path.abspath(g0_record), "sha256": _sha(g0_record)},
                                   "crossover.py": _sha(os.path.join(HERE, "crossover.py")),
                                   "g0_gate.py": _sha(os.path.join(HERE, "g0_gate.py"))}}
    if f_low is None:
        out["decision"] = "no f in the grid gives matched pairs with a median P: the dose-response block is not run"
        return out
    for b, (t, k) in tabs.items():
        out["bases"][b] = {"k": k, "N": t.N, "L": t.L, **draws_for_base(t, k, code, f_low, log, base=b)}
    n_run = sum(1 for v in out["bases"].values() if v["run"])
    out["decision"] = f"f_low = {f_low:.6f}; dose level run for {n_run} of {len(out['bases'])} MoE bases"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dose-response draws (prereg v1.3 sec. 4.2, P-9)")
    ap.add_argument("--gate-config", required=True)
    ap.add_argument("--g0-record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None, help="hashed manifest of the accepted draws (generator manifests, hash list 8)")
    ap.add_argument("--n", type=int, default=GG.G0_DRAWS)
    a = ap.parse_args(argv)
    for p in (a.gate_config, a.g0_record):
        if not os.path.isfile(p):
            print(f"[dose] missing {p}")
            return 2
    r = build(a.gate_config, a.g0_record, a.n)
    acc = [e for v in r["bases"].values() for e in v.get("accepted", [])]
    for v in r["bases"].values():
        v["accepted"] = [{kk: e.get(kk) for kk in ("base", "d", "reg_draw_index", "seed", "dose_index", "data_seed", "data_seed_status", "predicted", "selection_P",
                                                    "overlap_per_layer", "f", "k", "q", "status")} for e in v.get("accepted", [])]
    if a.manifest and acc:
        r["manifest"] = {"path": os.path.abspath(a.manifest),
                         "sha256": CX.write_manifest(a.manifest, acc, extra={"prereg": "v1.3 sec. 4.2 (P-9)", "f_low": r["f_low"],
                                                                             "code_stream": r["code_stream"]})}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(r, indent=1, sort_keys=True, default=float) + "\n")
    print(f"[dose] {r['decision']} -> {a.out}")
    return 0 if any(v["run"] for v in r["bases"].values()) else 4


if __name__ == "__main__":
    sys.exit(main())
