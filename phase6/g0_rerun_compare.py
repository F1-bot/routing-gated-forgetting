"""Decision-level comparison of the registered G0 gate record with the G0 re-run. Stdlib only.

Every G0 DECISION of the re-run must equal the registered record; numbers may differ only through the documented
nondeterminism of the bf16 probe gradient, so they are reported (absolute differences) but never decide.

  python g0_rerun_compare.py --registered g0/gate_record_registered.json --rerun g0/gate_record.json \
         [--out logs/g0_rerun_compare.json]
Exit codes: 0 all decisions equal; 1 a decision differs (the pipeline stops); 2 an input is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _m(x):
    return sorted(x) if isinstance(x, list) else x


def decisions(rec: dict) -> dict:
    """Flat {name: value} of every decision the downstream pipeline depends on."""
    g = rec.get("gates", {})
    d: dict = {"code_stream": ((rec.get("decisions") or {}).get("code_stream") or {}).get("chosen")}
    g1 = g.get("G0-1") or {}
    for key, c in sorted((g1.get("cells") or {}).items()):
        for f in ("feasible", "rung", "f"):
            d[f"G0-1|{key}|{f}"] = c.get(f)
        d[f"G0-1|{key}|matched"] = _m(c.get("matched"))
        if "missing" in c:
            d[f"G0-1|{key}|missing"] = _m(c.get("missing"))
    for f in ("moe_pass", "nestor_pass", "moe_targets", "nestor_targets", "code_stream"):
        d[f"G0-1|{f}"] = g1.get(f)
    d["G0-1|actions"] = sorted(g1.get("actions") or [])
    g2 = g.get("G0-2") or {}
    for r in g2.get("rows") or []:
        k = f"G0-2|{r.get('design')}|{r.get('cell')}"
        d[k + "|feasible"], d[k + "|rung"] = r.get("feasible"), r.get("rung")
        for fk, pf in sorted((r.get("per_f") or {}).items()):
            d[f"{k}|f={fk}|g0_2_pass"] = pf.get("g0_2_pass")
    d["G0-2|actions"] = sorted(g2.get("actions") or []) if isinstance(g2.get("actions"), list) else g2.get("actions")
    g3 = g.get("G0-3") or {}
    for f in ("base", "target", "pass", "median_dQ_ge_log2", "action", "status"):
        d[f"G0-3|{f}"] = g3.get(f)
    for f in ("feasible", "rung", "f"):
        d[f"G0-3|result|{f}"] = (g3.get("result") or {}).get(f)
    d["G0-3|result|matched"] = _m((g3.get("result") or {}).get("matched"))
    g4 = g.get("G0-4") or {}
    for f in ("base", "pass", "action", "status"):
        d[f"G0-4|{f}"] = g4.get(f)
    gen = g4.get("generator")
    if isinstance(gen, dict):
        for f in ("feasible", "rung", "f"):
            d[f"G0-4|generator|{f}"] = gen.get(f)
    g5 = g.get("G0-5") or {}
    for f in ("pass", "status", "action"):
        d[f"G0-5|{f}"] = g5.get(f)
    for b, v in sorted((g5.get("per_base") or {}).items()):
        for f in ("pass", "mean_ce_pass", "action"):
            d[f"G0-5|{b}|{f}"] = v.get(f)
    for b, v in sorted((g.get("G0-6") or {}).items()):
        if isinstance(v, dict):
            for f in ("status", "pass", "action"):
                d[f"G0-6|{b}|{f}"] = v.get(f)
    for b, v in sorted((g.get("G0-7") or {}).items()):
        if isinstance(v, dict):
            d[f"G0-7|{b}|beta0"], d[f"G0-7|{b}|margin_limited"] = v.get("beta0"), v.get("margin_limited")
    return d


def numbers(rec: dict) -> dict:
    g = rec.get("gates", {})
    n: dict = {}
    for key, c in ((g.get("G0-1") or {}).get("cells") or {}).items():
        n[f"G0-1|{key}|median"] = c.get("median")
    n["G0-3|median"] = ((g.get("G0-3") or {}).get("result") or {}).get("median")
    n["G0-4|ratio"] = (g.get("G0-4") or {}).get("ratio")
    for b, v in ((g.get("G0-5") or {}).get("per_base") or {}).items():
        n[f"G0-5|{b}|abs_dce"] = v.get("abs_dce")
    for b, v in (g.get("G0-6") or {}).items():
        if isinstance(v, dict):
            n[f"G0-6|{b}|reliability"] = v.get("reliability")
    for b, v in (g.get("G0-7") or {}).items():
        if isinstance(v, dict):
            for f in ("D", "SE", "P_bar"):
                n[f"G0-7|{b}|{f}"] = v.get(f)
    return {k: v for k, v in n.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def compare(reg: dict, new: dict) -> dict:
    a, b = decisions(reg), decisions(new)
    diffs = [{"decision": k, "registered": a.get(k), "rerun": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    na, nb = numbers(reg), numbers(new)
    num = {k: {"registered": na[k], "rerun": nb[k], "abs_diff": abs(na[k] - nb[k])} for k in sorted(set(na) & set(nb))}
    return {"decisions_equal": not diffs, "n_decisions": len(a), "differences": diffs, "numbers": num,
            "numbers_only_registered": sorted(set(na) - set(nb)), "numbers_only_rerun": sorted(set(nb) - set(na))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G0 re-run vs registered gate record: decisions must be equal")
    ap.add_argument("--registered", required=True)
    ap.add_argument("--rerun", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    for p in (a.registered, a.rerun):
        if not os.path.isfile(p):
            print(f"[g0-compare] missing input {p}")
            return 2
    reg, new = (json.load(open(p, encoding="utf-8-sig")) for p in (a.registered, a.rerun))
    r = compare(reg, new)
    r.update(created=time.strftime("%Y-%m-%dT%H:%M:%S"), registered={"path": os.path.abspath(a.registered), "sha256": _sha(a.registered)},
             rerun={"path": os.path.abspath(a.rerun), "sha256": _sha(a.rerun)},
             rule="any differing decision stops the pipeline; numbers may differ (bf16 probe gradient)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(r, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[g0-compare] {r['n_decisions']} decisions compared: {'ALL EQUAL' if r['decisions_equal'] else str(len(r['differences'])) + ' DIFFER'}")
    for d in r["differences"][:30]:
        print(f"[g0-compare]   {d['decision']}: registered {d['registered']!r} vs re-run {d['rerun']!r}")
    big = sorted(r["numbers"].items(), key=lambda kv: -kv[1]["abs_diff"])[:5]
    for k, v in big:
        print(f"[g0-compare]   number {k}: {v['registered']:.6g} vs {v['rerun']:.6g} (|diff| {v['abs_diff']:.3g})")
    return 0 if r["decisions_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
