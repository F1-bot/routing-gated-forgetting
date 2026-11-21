"""Stage B (domain adaptation) analysis in the paper's Table 6 / Figure 3 conventions.
gain = target CE before - after; forgetting = DCLM CE after - before; equal-weight averages over the four domains;
relative columns are % of the MoE value (paper Fig. 3). Inputs: results/continual.jsonl (B| keys)."""
import json, sys, os
PAPER_T6 = {  # 500M NESTOR / 500M MoE / 1B dense, no replay, 32k context, 0.5B tokens per domain (paper Table 6)
    "pg19": {"nestor": (0.184, 0.217), "moe": (0.182, 0.229), "dense": (0.181, 0.252)},
    "owm": {"nestor": (0.332, 0.106), "moe": (0.338, 0.110), "dense": (0.338, 0.134)},
    "algstack": {"nestor": (0.705, 0.294), "moe": (0.712, 0.296), "dense": (0.688, 0.385)},
    "fw2de": {"nestor": (1.706, 0.398), "moe": (1.704, 0.410), "dense": (1.733, 0.494)},
    "avg": {"nestor": (0.732, 0.254), "moe": (0.734, 0.261), "dense": (0.735, 0.316)}}
CORPORA = ["pg19", "owm", "algstack", "fw2de"]
NAMES = {"pg19": "Books", "owm": "Math", "algstack": "Code", "fw2de": "German"}
MODELS = {"dense_100m": "dense", "moe_100m": "moe", "nestor_100m": "nestor"}

def load(path="results/continual.jsonl"):
    R = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            R[r["key"]] = r
    return R

def table(R):
    rows = []
    for exp, short in MODELS.items():
        for c in CORPORA:
            bt, bd = R.get(f"B|{exp}|before|{c}"), R.get(f"B|{exp}|before|dclm")
            at, ad = R.get(f"B|{exp}|after|{c}|target"), R.get(f"B|{exp}|after|{c}|dclm")
            if not (bt and bd and at and ad):
                rows.append({"model": short, "corpus": c, "complete": False}); continue
            rows.append({"model": short, "corpus": c, "complete": True, "target_before": bt["ce"], "target_after": at["ce"],
                         "dclm_before": bd["ce"], "dclm_after": ad["ce"], "gain": bt["ce"] - at["ce"], "forgetting": ad["ce"] - bd["ce"],
                         "tokens_eval": bt.get("tokens")})
    return rows

def summarize(rows):
    out = {}
    for short in MODELS.values():
        rr = [r for r in rows if r["model"] == short and r["complete"]]
        if len(rr) == len(CORPORA):
            out[short] = {"gain": sum(r["gain"] for r in rr) / len(rr), "forgetting": sum(r["forgetting"] for r in rr) / len(rr)}
    rel = {}
    if "moe" in out:
        for short, v in out.items():
            rel[short] = {"gain_pct_vs_moe": 100 * (v["gain"] / out["moe"]["gain"] - 1), "forgetting_pct_vs_moe": 100 * (v["forgetting"] / out["moe"]["forgetting"] - 1)}
        per = {}
        for c in CORPORA:
            m = [r for r in rows if r["model"] == "moe" and r["corpus"] == c and r["complete"]]
            if not m: continue
            for short in MODELS.values():
                x = [r for r in rows if r["model"] == short and r["corpus"] == c and r["complete"]]
                if x:
                    per.setdefault(c, {})[short] = {"gain_pct_vs_moe": 100 * (x[0]["gain"] / m[0]["gain"] - 1),
                                                     "forgetting_pct_vs_moe": 100 * (x[0]["forgetting"] / m[0]["forgetting"] - 1)}
        rel["per_corpus"] = per
    paper_rel = {s: {"gain_pct_vs_moe": 100 * (PAPER_T6["avg"][s][0] / PAPER_T6["avg"]["moe"][0] - 1),
                     "forgetting_pct_vs_moe": 100 * (PAPER_T6["avg"][s][1] / PAPER_T6["avg"]["moe"][1] - 1)} for s in ("nestor", "moe", "dense")}
    return {"ours_avg": out, "ours_rel": rel, "paper_avg": PAPER_T6["avg"], "paper_rel": paper_rel}

if __name__ == "__main__":
    R = load(sys.argv[1] if len(sys.argv) > 1 else "results/continual.jsonl")
    rows = table(R); s = summarize(rows)
    os.makedirs("results", exist_ok=True)
    json.dump({"rows": rows, "summary": s}, open("results/stageB_summary.json", "w", encoding="utf-8"), indent=1)
    for r in rows:
        if r["complete"]:
            print(f"{r['model']:7s} {NAMES[r['corpus']]:7s} gain {r['gain']:.4f} forgetting {r['forgetting']:.4f}")
        else:
            print(f"{r['model']:7s} {NAMES[r['corpus']]:7s} (incomplete)")
    print(json.dumps(s, indent=1))
