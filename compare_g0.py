"""Compare a re-run G0 gate record with the registered one: every decision must match; numbers are reported."""
import json, sys, math
orig = json.load(open(sys.argv[1], encoding="utf-8")); new = json.load(open(sys.argv[2], encoding="utf-8"))
dec, num = [], []
def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ("inputs", "script", "created", "seconds", "config", "operationalisation"): continue
            walk(a.get(k), b.get(k), path + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b: dec.append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))):
            num.append((path, a, b, abs(a - b) / max(abs(a), abs(b), 1e-12)))
    elif a != b:
        dec.append((path, str(a)[:80], str(b)[:80]))
walk(orig.get("gates", {}), new.get("gates", {}), "gates"); walk(orig.get("decisions", {}), new.get("decisions", {}), "decisions")
num.sort(key=lambda r: -r[3])
out = {"decision_mismatches": dec, "n_numeric_diffs": len(num), "largest_numeric_diffs": num[:25],
       "verdict": "MATCH" if not dec else "MISMATCH"}
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), indent=1, default=str)
print(out["verdict"], len(dec), "decision mismatches;", len(num), "numeric diffs")
sys.exit(0 if not dec else 3)
