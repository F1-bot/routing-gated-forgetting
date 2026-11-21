"""make_exploratory_values.py -- writes data/exploratory_values.json for paper v1.6.5 (exploratory analyses after the freeze).

Block randdir_20: the random-direction control extended after the freeze to all 20 C1 draws (results/exploratory/rand_extension.json,
written by results/exploratory/rand_extension.py with the registered functions), with the job list, the queue state and the registered
analysis output (for the per-draw excess I_sym - I_rand). Block dmuon_step2: D_Muon after the first non-zero update (step 2;
the warm-up gives step 1 a learning rate of 0) from the exploratory re-run of the 16 C2 Muon runs (results/exploratory/dmuon_step2.json); null until that file exists. Nothing here enters a registered decision. Run: python scripts/make_exploratory_values.py
[--root .]."""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, math, os, re, time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iso(t): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))

def randdir_block(root):
    ex = root / "results" / "exploratory"
    rx_p, q_p, qs_p, py_p = ex / "rand_extension.json", ex / "queue_randx.json", ex / "queue_state_randx.json", ex / "rand_extension.py"
    an_p = root / "results" / "analysis_core" / "analysis_confirm.json"
    rx, q, qs, an = (json.loads(p.read_text(encoding="utf-8")) for p in (rx_p, q_p, qs_p, an_p))
    jobs = q["jobs"]; a0 = jobs[0]["args"]
    def opt(flag): return [a0[i + 1] for i, a in enumerate(a0) if a == flag]
    sets = {s.split("=")[0]: int(s.split(":")[-1]) for s in opt("--set")}
    spans = {s.split("=")[0]: [float(x) for x in s.split("=")[1].split(",")] for s in opt("--span")}
    seeds = sorted(int(m) for j in jobs for m in re.findall(r"randdir:(\d+)", " ".join(j["args"])))
    sym = {}
    for x in an["C1"]["statistics"]["sym"]["draws"]:
        m = re.search(r"c1_(moe_b\d)_books_d(\d+)_A", x["A"]); sym[(m.group(1), int(m.group(2)))] = x["I_log"]
    exc = {(d["base"], d["d"]): sym[(d["base"], d["d"])] - d["I_rand"] for d in rx["per_draw"]}
    kmin = min(exc, key=exc.get)
    st = list(qs.values()); br = rx["beta_rand"]
    def ci(b): return {k: b[k] for k in ("n", "mean", "lo", "hi", "level")}
    six_ok = [d["I_rand"] for d in rx["per_draw"][:6]] == rx["check_registered_six"]["I"] == [x["I_log"] for x in an["C1"]["statistics"]["rand"]["draws"]]
    return {
        "label": rx["label"], "statistic": "log crossover I (the registered statistic), for random, symmetrized and raw updates",
        "n_draws": rx["n_draws_with_rand"], "n_registered_control_draws": rx["n_registered_control_draws"],
        "n_new_draws": rx["n_draws_with_rand"] - rx["n_registered_control_draws"], "n_evaluations": len(jobs),
        "seed_rule": "9700 + 100(base-1) + 2d + (0 for subset A, 1 for subset B)", "seeds_min": seeds[0], "seeds_max": seeds[-1],
        "tokens_per_corpus": sets, "span": spans, "window_tokens": int(opt("--block")[0]), "select": opt("--select")[0],
        "gpu_hours": round(sum(v["hours"] for v in st), 4), "all_done_rc0": all(v["status"] == "done" and v["rc"] == 0 for v in st),
        "first_start_utc": iso(min(v["start"] for v in st)), "last_end_utc": iso(max(v["end"] for v in st)), "job_list_mtime_utc": iso(q_p.stat().st_mtime),
        "registered_six_reproduced": six_ok,
        "mean_I": rx["mean_I"], "share_rand_over_sym": rx["share_rand_over_sym"], "share_rand_over_raw": rx["share_rand_over_raw"],
        "I_sym_minus_I_rand": ci(rx["I_sym_minus_I_rand"]), "I_raw_minus_I_rand": ci(rx["I_raw_minus_I_rand"]),
        "per_draw_excess_sym": {"n_positive": sum(v > 0 for v in exc.values()), "min": exc[kmin], "min_draw": f"{kmin[0]} d{kmin[1]}"},
        "beta_rand": {"overall": {k: br["overall"][k] for k in ("beta_hat", "lo", "hi", "D", "method")},
                      "per_base": {b: {k: v[k] for k in ("beta_hat", "lo", "hi", "D")} for b, v in br["per_base"].items()}},
        "per_base": {b: {"n": v["n"], "mean_I_rand": v["mean_I_rand"], "mean_I_sym": v["mean_I_sym"], "share_rand_over_sym": v["share_rand_over_sym"],
                         "I_sym_minus_I_rand": ci(v["I_sym_minus_I_rand"])} for b, v in rx["per_base"].items()},
        "sources": {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in (rx_p, py_p, q_p, qs_p, an_p)},
    }

def dmuon_block(root):
    ex = root / "results" / "exploratory"; res_p = ex / "dmuon_step2.json"
    if not res_p.exists():
        return None
    q2_p, qs2_p, py2_p, q1_p, qs1_p = ex / "queue_c2step2.json", ex / "queue_state_c2step2.json", ex / "dmuon_step2.py", ex / "queue_c2step1.json", ex / "queue_state_c2step1.json"
    an = json.loads((root / "results" / "analysis_core" / "analysis_confirm.json").read_text(encoding="utf-8"))
    res, qs2, qs1 = (json.loads(p.read_text(encoding="utf-8")) for p in (res_p, qs2_p, qs1_p))
    mc = an["C2"]["manipulation_check"]["per_draw"]; kept = {(r["base"], r["d"]) for r in mc}; reg8 = {(r["base"], r["d"]): r["D_muon_step8"] for r in mc}
    rows = res["per_draw"]; v5 = [r["D_muon_step2"] for r in rows if (r["base"], r["d"]) in kept and r["D_muon_step2"] is not None]
    st2, st1 = list(qs2.values()), list(qs1.values())
    a1 = [v.get("audit") for v in st1 if v.get("audit")]; au = json.loads(Path(a1[0]).read_text(encoding="utf-8")) if a1 and Path(a1[0]).exists() else {}
    return {"label": res.get("label"), "statistic": "D_Muon(s) = log(||delta_H(s)|| / ||delta_L(s)||) of the displacement, as in the registered analysis",
            "n_draws": res["n"], "n_runs": len(st2), "min": res["D_muon_step2_min"], "max": res["D_muon_step2_max"], "mean": res["D_muon_step2_mean"], "max_abs": res["D_muon_step2_max_abs"],
            "five_retained_draws": {"n": len(v5), "min": min(v5) if v5 else None, "max": max(v5) if v5 else None},
            "step1_all_zero": res["step1_all_zero"],
            "max_norm_ratio_pct": 100.0 * (math.exp(res["D_muon_step2_max_abs"]) - 1.0),
            "step8_registered_runs": {"n": len(rows), "min": min(r["D_muon_step8_registered_runs"] for r in rows), "max": max(r["D_muon_step8_registered_runs"] for r in rows)},
            "D_sgd_step8_registered": {"n": an["C2"]["decay_profile"]["8"]["n"], "min": min(an["C2"]["decay_profile"]["8"]["D_sgd"]), "max": max(an["C2"]["decay_profile"]["8"]["D_sgd"])},
            "step8_reproduces_registered": all(abs(r["D_muon_step8_registered_runs"] - reg8[(r["base"], r["d"])]) < 1e-12 for r in rows if (r["base"], r["d"]) in kept),
            "per_draw": [{"base": r["base"], "d": r["d"], "D_muon_step2": r["D_muon_step2"], "retained_in_C2": (r["base"], r["d"]) in kept} for r in rows],
            "gpu_hours": round(sum(v.get("hours", 0) for v in st2), 4), "all_done_rc0": all(v.get("status") == "done" and v.get("rc", 0) == 0 for v in st2),
            "first_start_utc": iso(min(v["start"] for v in st2)), "last_end_utc": iso(max(v["end"] for v in st2)), "job_list_mtime_utc": iso(q2_p.stat().st_mtime),
            "step1_attempt": {"runs": len(st1), "gpu_hours": round(sum(v.get("hours", 0) for v in st1), 4), "status": [v.get("status") for v in st1], "error": [v.get("error") for v in st1],
                              "trainable_unchanged_entries": len(au.get("trainable_unchanged", [])), "job_list_sha256": sha(q1_p)},
            "sources": {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in (res_p, py2_p, q2_p, qs2_p, q1_p, qs1_p)}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=r"."); a = ap.parse_args()
    root = Path(a.root); out = Path(__file__).resolve().parents[1] / "data" / "exploratory_values.json"
    doc = {"format": "exploratory_values/1", "note": "post-freeze exploratory analyses; no value enters a registered test, label or decision",
           "randdir_20": randdir_block(root), "dmuon_step2": dmuon_block(root)}
    if doc["dmuon_step2"]:
        doc["gpu_hours_total"] = round(doc["randdir_20"]["gpu_hours"] + doc["dmuon_step2"]["gpu_hours"] + doc["dmuon_step2"]["step1_attempt"]["gpu_hours"], 4)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out.name}: randdir_20 n={doc['randdir_20']['n_draws']}, six reproduced={doc['randdir_20']['registered_six_reproduced']}, dmuon_step2={'present' if doc['dmuon_step2'] else 'null'}")

if __name__ == "__main__":
    main()
