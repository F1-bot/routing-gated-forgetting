"""G0-1 feasibility of the NESTOR books cell (exploratory block). Torch-free (numpy).

The registered G0-1 procedure of g0_gate.py is applied to the cell nestor|books: g0_gate.ladder over the registered rung
of g0_gate.rungs_for('c1') only (tolerance log 1.05, prereg v1.3 sec. 4.4; pools f in {1/2, 1/3}; 1,000 draws
of the distribution run, seed 5000 + 0), on the re-run routing files of the NESTOR base, and the cell is feasible iff
it is feasible at a rung and its median predicted contrast is >= log 1.3 on the chosen code stream (g0_gate.arch_status).
The frozen gate config is only READ (the cell is evaluated by calling the gate's functions; c1_targets is not edited).
If the cell is feasible, the 3 exploratory draws d = 1..3 (crossover.reg_draw, design c1, generator seeds 5000 + d,
the cell's f and rung tolerance) are written to a hashed manifest.

  python nestor_books_feasibility.py --gate-config g0/g0_gate_config.json --g0-record g0/gate_record.json \
         --out results/nestor_books_feasibility.json [--draws-out confirm/specs/nestor_books_draws.json] [--n 1000]
Exit codes: 0 feasible; 4 infeasible; 2 inputs missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import crossover as CX  # noqa: E402
import g0_gate as GG  # noqa: E402

N_EXPLORATORY_DRAWS = 3


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slim(cell: dict) -> dict:
    """The ladder result without the per-rung arrays (kept: per-f medians, acceptance and swap convergence)."""
    out = {k: v for k, v in cell.items() if k != "tried"}
    out["tried"] = [{"rung": t["rung"], "name": t["name"], "tol": t["tol"], "matched": t["matched"],
                     "per_f": {kf: {kk: v.get(kk) for kk in ("median", "median_selection", "n_pairs", "acceptance",
                                                              "swap_convergence", "g0_2_pass", "complete")}
                               for kf, v in t["per_f"].items()}} for t in cell.get("tried", [])]
    return out


def assess(gate_config: str, g0_record: str, *, base: str = "nestor", target: str = "books", n_draws: int = GG.G0_DRAWS,
           log=print) -> dict:
    cfg = GG.load_config(gate_config)                       # read only; resolves the base dirs against the config dir
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    code = ((rec.get("decisions") or {}).get("code_stream") or {}).get("chosen") or "code"
    b = cfg["bases"][base]
    t = CX.reg_load_dir(b["dir"], b["family"], layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
    k = CX.reg_k(t)
    if b.get("k") is not None and int(b["k"]) != k:
        raise SystemExit(f"{base}: config k={b['k']} but registered k = q N = {k}")
    inputs = {}
    for c, s in t.sources.items():
        for role in ("selection", "estimation"):
            inputs[s[role]["file"]] = _sha(s[role]["file"])
    cells = {}
    for cs in cfg["code_streams"]:
        need = {target, "dclm", cs, "math"}
        if not need <= set(t.sel):
            cells[cs] = {"feasible": False, "missing": sorted(need - set(t.sel)), "median": None}
            continue
        # prereg v1.3 sec. 4.4: the G0-1 rule at the REGISTERED tolerance log 1.05 only (no relaxed rung)
        rungs = [r for r in GG.rungs_for("c1", target=target, code=cs) if r["tol"] == CX.REG_TOL]
        cells[cs] = GG.ladder(t, "c1", rungs, target=target, code=cs, n_draws=n_draws, k=k)
        if log:
            log(f"[nestor-books] {base}|{target}|{cs}: feasible {cells[cs]['feasible']} median P {cells[cs].get('median')} "
                f"f {cells[cs].get('f')} rung {cells[cs].get('rung')}")
    c = cells.get(code) or {}
    feasible = bool(c.get("feasible") and c.get("median") is not None and c["median"] >= GG.LOG_G0)
    out = {"purpose": "G0-1 feasibility of nestor|books (exploratory; registered G0-1 procedure)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": base, "target": target, "code_stream": code,
           "threshold": GG.LOG_G0, "n_draws": n_draws, "k": k, "L": t.L, "N": t.N, "strata": t.strata,
           "excluded_strata": t.excluded, "feasible": feasible,
           "cells": {cs: _slim(v) for cs, v in cells.items()},
           "decision": ("feasible: 3 draws x 2 runs, exploratory, first-order-free statistics"
                        if feasible else "infeasible: no NESTOR books block"),
           "inputs": {"gate_config": {"path": os.path.abspath(gate_config), "sha256": _sha(gate_config)},
                      "g0_record": {"path": os.path.abspath(g0_record), "sha256": _sha(g0_record)},
                      "g0_gate.py": _sha(os.path.join(HERE, "g0_gate.py")), "crossover.py": _sha(os.path.join(HERE, "crossover.py")),
                      "routing_files": inputs}}
    if feasible:
        draws = [CX.reg_draw(t, "c1", d, f=float(c["f"]), target=target, code=code, k=k, tol=float(c["tol"]))
                 for d in range(1, N_EXPLORATORY_DRAWS + 1)]
        for e in draws:
            e["data_seed"] = 7000 + int(e["d"])           # registered C1 data seeds, cell nestor|books
        out["draws"] = draws
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NESTOR books G0-1 feasibility")
    ap.add_argument("--gate-config", required=True)
    ap.add_argument("--g0-record", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws-out", default=None)
    ap.add_argument("--n", type=int, default=GG.G0_DRAWS)
    a = ap.parse_args(argv)
    for p in (a.gate_config, a.g0_record):
        if not os.path.isfile(p):
            print(f"[nestor-books] missing {p}")
            return 2
    r = assess(a.gate_config, a.g0_record, n_draws=a.n)
    draws = r.pop("draws", None)
    if draws is not None and a.draws_out:
        r["draws_manifest"] = {"path": os.path.abspath(a.draws_out),
                               "sha256": CX.write_manifest(a.draws_out, draws, extra={
                                   "prereg": "the design (NESTOR books, exploratory)", "code_stream": r["code_stream"],
                                   "cell": f"{r['base']}|{r['target']}|{r['code_stream']}"})}
    elif draws is not None:
        r["draws"] = draws
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(r, indent=1, sort_keys=True, default=float) + "\n")
    c = r["cells"].get(r["code_stream"], {})
    print(f"[nestor-books] {r['base']}|{r['target']}|{r['code_stream']}: {'FEASIBLE' if r['feasible'] else 'infeasible'} "
          f"(median P {c.get('median')}, threshold {r['threshold']:.4f}) -> {a.out}")
    return 0 if r["feasible"] else 4


if __name__ == "__main__":
    sys.exit(main())
