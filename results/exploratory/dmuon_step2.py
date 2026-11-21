"""Exploratory (post-freeze): D_Muon after the first non-zero update (step 2) of the 16 registered C2 Muon runs,
re-run with identical seeds, data, subsets, learning rate and 96-step schedule (results/exploratory/queue_c2step2.json).
D_o(s) = log(||delta_H(s)|| / ||delta_L(s)||) from interim/displacement.jsonl (total_sq), as in the registered analysis
(analyze_confirm._disp). Step 1 is recorded too: the warm-up gives the first optimizer step a learning rate of 0."""
import json, math, os, glob
ROOT = r"."
def disp(path, step):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if int(r.get("step", -1)) == int(step):
                v = float(r.get("total_sq", float("nan")))
                return math.sqrt(v) if v > 0 else 0.0
    return None
def root(kind, run):
    hits = glob.glob(os.path.join(ROOT, "outputs", kind, run, "*", "interim", "displacement.jsonl"))
    return hits[0] if len(hits) == 1 else None
rows = []
for base, ds in ((1, (1, 2, 3, 4)), (2, (5, 6, 7, 8))):
    for d in ds:
        rid = {s: f"c2_moe_b{base}_d{d}_muon_{s}" for s in ("H", "L")}
        rec = {"base": f"moe_b{base}", "d": d}
        for s in ("H", "L"):
            p2 = root(os.path.join("exploratory", "c2_step2"), rid[s])
            rec[f"step1_norm_{s}"] = disp(p2, 1) if p2 else None
            rec[f"step2_norm_{s}"] = disp(p2, 2) if p2 else None
            p8 = root("confirm", rid[s])
            rec[f"step8_norm_{s}"] = disp(p8, 8) if p8 else None
        h2, l2 = rec["step2_norm_H"], rec["step2_norm_L"]
        rec["D_muon_step2"] = math.log(h2 / l2) if h2 and l2 else None
        h8, l8 = rec["step8_norm_H"], rec["step8_norm_L"]
        rec["D_muon_step8_registered_runs"] = math.log(h8 / l8) if h8 and l8 else None
        rows.append(rec)
vals = [r["D_muon_step2"] for r in rows if r["D_muon_step2"] is not None]
out = {"label": "post-freeze exploratory; job list fixed and hashed before running (queue_c2step2.json)",
       "per_draw": rows, "n": len(vals),
       "D_muon_step2_min": min(vals) if vals else None, "D_muon_step2_max": max(vals) if vals else None,
       "D_muon_step2_mean": sum(vals) / len(vals) if vals else None,
       "D_muon_step2_max_abs": max(abs(v) for v in vals) if vals else None,
       "step1_all_zero": all((r["step1_norm_H"] == 0.0 and r["step1_norm_L"] == 0.0) for r in rows if r["step1_norm_H"] is not None)}
if __name__ == "__main__":
    p = os.path.join(ROOT, "results", "exploratory", "dmuon_step2.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "per_draw"}))
