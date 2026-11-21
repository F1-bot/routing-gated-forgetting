"""make_supplementary_values.py -- numbers added in paper v1.6.3 (external review), read from their primary sources.
Writes data/supplementary_values.json. Sources: results/analysis_core/analysis_confirm.json (registered analysis output),
confirm/queue_confirm.json (frozen queue: subset specifications), logs/queue_g1/*_m2.log (PyTorch memory of G1 runs),
data/edge_pre_freeze.json (full adaptations measured before the freeze), data/fig_arch_facts.json (architecture);
v1.6.4 adds edge_tradeoff from g1/gate_record_G1.json, the G1 subset manifests (outputs/g1) and the G1 step logs.
Usage: python scripts/make_supplementary_values.py [--root .]"""
import argparse, hashlib, json, re, statistics
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(x): return sum(x) / len(x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=r"."); a = ap.parse_args()
    R = Path(a.root); V1 = R / "paper" / "v1"; AN = R / "results" / "analysis_core" / "analysis_confirm.json"
    A = json.loads(AN.read_text(encoding="utf-8")); C1, C2, E = A["C1"], A["C2"], A["blocks"]["EDGE"]
    out = {"format": "supplementary_values/1", "written_by": "scripts/make_supplementary_values.py",
           "sources": {"analysis": {"path": "results/analysis_core/analysis_confirm.json", "sha256": sha(AN)},
                       "queue": {"path": "confirm/queue_confirm.json", "sha256": sha(R / "confirm" / "queue_confirm.json")},
                       "g1_logs": "logs/queue_g1/g1_moe_b1_*_m2.log (PyTorch step records)",
                       "edge_pre_freeze": "paper/v1/data/edge_pre_freeze.json", "arch": "paper/v1/data/fig_arch_facts.json",
                       "e1_rule": "paper/v1/sections/04_design.tex: if more than 25% of a block's runs diverge, its confirmatory test is inconclusive"}}
    # C1: random share on the six draws of base 1 that carry the controls
    def base1_6(stat):
        sel = []
        for d in C1["statistics"][stat]["draws"]:
            m = re.match(r"c1_(moe_b\d)_books_d(\d+)_A", d["A"])
            if m and m.group(1) == "moe_b1" and int(m.group(2)) <= 6 and d.get("I_log") is not None: sel.append((int(m.group(2)), d["I_log"]))
        return sorted(sel)
    rs = {s: base1_6(s) for s in ("rand", "sym", "raw")}
    assert [d for d, _ in rs["rand"]] == [d for d, _ in rs["sym"]] == [d for d, _ in rs["raw"]] == [1, 2, 3, 4, 5, 6]
    mI = {s: mean([v for _, v in rs[s]]) for s in rs}; sec = C1["secondary"]; ism = sec["I_sym_minus_I_rand"]
    out["c1_random_share"] = {"base": "moe_b1", "draws": [1, 2, 3, 4, 5, 6], "n": 6, "mean_I_log": mI, "rand_over_sym": mI["rand"] / mI["sym"], "rand_over_raw": mI["rand"] / mI["raw"],
        "rand_over_sym_pct": 100 * mI["rand"] / mI["sym"], "sym_minus_rand_of_means": mI["sym"] - mI["rand"],
        "I_sym_minus_I_rand": {"mean": ism["mean"], "lo": ism["lo"], "hi": ism["hi"], "n": ism["n"]}, "field": "C1.statistics.{rand,sym,raw}.draws[*].I_log, base moe_b1, d <= 6"}
    # C1 magnitude and the band statement C1-b
    raw = C1["estimators"]["raw"]["overall"]
    out["c1_magnitude"] = {"beta_raw": {"hat": raw["beta_hat"], "lo": raw["lo"], "hi": raw["hi"]}, "exact_assumption_value": 1, "registered_margin": 0.25,
        "C1b_raw": sec["C1b_pooled_ratio_raw"], "C1b_sym": sec["C1b_pooled_ratio_sym"],
        "C1b_refuted": bool(sec["C1b_pooled_ratio_raw"]["C1b_refuted"] or sec["C1b_pooled_ratio_sym"]["C1b_refuted"])}
    # decay rule: normalized difference over six draws divided by a log ratio over 20 draws
    wd = C1["statistics"]["wd"]["draws"]; raw20 = [d["I_log"] for d in C1["statistics"]["raw"]["draws"] if d.get("I_log") is not None]
    nd6 = [d["I_nd"] for d in wd if d.get("I_nd") is not None]; fwd = [abs(v["F"]) for d in wd for v in d["F"].values() if v.get("F") is not None]
    out["decay_rule"] = {"mean_I_WD_nd": mean(nd6), "n_WD": len(nd6), "mean_I_log": mean(raw20), "n_I": len(raw20), "ratio": mean(nd6) / mean(raw20),
        "registered_ratio": sec["wd_share"]["mean_I_WD_over_mean_I"], "max_abs_F_WD_nats": max(fwd), "n_F_WD": len(fwd)}
    cs = sec["second_code_stream"]; out["code_stream2"] = {"beta_hat": cs["beta"]["overall"]["beta_hat"], "lo": cs["beta"]["overall"]["lo"], "hi": cs["beta"]["overall"]["hi"], "mean_J": cs["J_test"]["mean"]}
    # C2 losses: runs (where the registered threshold applies) and draws
    out["c2_losses"] = {"runs": C2["E1"]["runs"], "runs_diverged": C2["E1"]["diverged"], "runs_pct": 100 * C2["E1"]["diverged"] / C2["E1"]["runs"],
        "draws_registered": C2["draws_registered"], "draws_dropped": len(C2["draws_dropped"]), "draws_pct": 100 * len(C2["draws_dropped"]) / C2["draws_registered"], "threshold_pct": 25}
    # memory, one metric: PyTorch peak allocated / reserved, max over logged steps
    LG = R / "logs" / "queue_g1"
    def mem(names):
        al, rv = [], []
        for n in names:
            t = (LG / n).read_text(encoding="utf-8", errors="replace")
            al.append(max(float(x) for x in re.findall(r"train/mem_peak_alloc_gib: ([\d.]+)", t))); rv.append(max(float(x) for x in re.findall(r"train/mem_peak_reserved_gib: ([\d.]+)", t)))
        return {"runs": names, "alloc_gib_max": max(al), "reserved_gib_max": max(rv)}
    full = mem([f"g1_moe_b1_full_{c}_s10{s}_m2.log" for c in ("german", "math") for s in (1, 2)])
    eall = mem([f"g1_moe_b1_eall_german_s10{s}_m2.log" for s in (1, 2)]); aa = mem([f"g1_moe_b1_aa{i}_R{j}_m2.log" for i in (1, 2) for j in (1, 2)])
    core = {"runs": len(E["runs"]), "alloc_gib_max": max(r["vram_peak_alloc_gib"] for r in E["runs"]), "reserved_gib_max": max(r["vram_peak_reserved_gib"] for r in E["runs"])}
    sv = full["alloc_gib_max"] - core["alloc_gib_max"]
    out["memory_pytorch"] = {"metric": "max over logged steps of train/mem_peak_alloc_gib (reserved in brackets); descriptive, read after the freeze from existing logs, outside the registered EDGE block",
        "full_moe_b1": full, "all_expert_moe_b1": eall, "subset_g1_aa": aa, "core_subset_and_c2": core, "saving_gib": sv, "saving_pct": 100 * sv / full["alloc_gib_max"]}
    sub = [r for r in E["runs"] if r["kind"] == "subset"]; bm = [r["board_MiB_max"] for r in sub]
    EP = json.loads((V1 / "data" / "edge_pre_freeze.json").read_text(encoding="utf-8"))["adaptation_96_steps"]["moe_full_runs_no_ckpt"]
    out["board_memory"] = {"metric": "board memory from the external logger (includes CUDA context, allocator cache and other processes)", "core_subset": {"n": len(bm), "median_mib": statistics.median(bm), "min_mib": min(bm), "max_mib": max(bm)},
        "prefreeze_full_max_mib": EP["peak_mem_mib_max"], "prefreeze_full_max_gib": EP["peak_mem_mib_max"] / 1024}
    # update size and expert layout
    ue = {r["updated_elements"] for r in sub}; et = {r["elements_total"] for r in E["runs"]}; uu = {r["updated_units"] for r in sub}; assert len(ue) == 1 and len(et) == 1 and len(uu) == 1
    Q = json.loads((R / "confirm" / "queue_confirm.json").read_text(encoding="utf-8"))["jobs"]
    specs = [[e for e in j["spec"]["select"] if e.get("kind") == "moe" and "expert" in e] for j in Q if str(j.get("id", "")).startswith(("c1_", "c2_")) and isinstance(j.get("spec"), dict) and j["spec"].get("select")]
    kinds = sorted({str(e.get("kind")) for j in Q if str(j.get("id", "")).startswith(("c1_", "c2_")) and isinstance(j.get("spec"), dict) for e in j["spec"].get("select") or []})
    specs = [s for s in specs if s]; per = {len(s) for s in specs}; k = {len(e["expert"]) for s in specs for e in s}; mx = max(x for s in specs for e in s for x in e["expert"])
    arch = json.loads((V1 / "data" / "fig_arch_facts.json").read_text(encoding="utf-8"))
    out["update_size"] = {"updated_elements": ue.pop(), "elements_total": et.pop(), "updated_units": uu.pop(), "runs": len(E["runs"]), "bitwise_audit_ok": sum(1 for r in E["runs"] if r.get("bitwise_audit_ok")),
        "subset_specs": len(specs), "select_kinds": kinds, "moe_layers_per_spec": sorted(per), "experts_per_layer_selected": sorted(k), "max_expert_id": mx, "arch_facts": arch}
    out["update_size"]["share_pct"] = 100 * out["update_size"]["updated_elements"] / out["update_size"]["elements_total"]
    fa = arch["facts"]; L_, X_ = fa["moe.layers"]["value"], fa["moe.experts_per_layer"]["value"]; assert per == {L_} and k == {6} and mx == X_ - 1
    out["update_size"].update({"moe_layers": L_, "experts_per_layer": X_, "experts_total": L_ * X_, "updated_experts_per_layer": 6, "updated_experts": 6 * L_, "frozen_share_pct": 100 - out["update_size"]["share_pct"]})
    # time and energy against full adaptation (medians)
    sw = statistics.median(r["wall_s"] for r in sub) / 3600; se = statistics.median(r["energy_Wh"] for r in sub)
    out["time_energy"] = {"subset_wall_h_median": sw, "full_wall_h_median": EP["wall_h"][1], "time_ratio": sw / EP["wall_h"][1],
                          "subset_energy_wh_median": se, "full_energy_wh_median": EP["energy_wh"][1], "energy_ratio": se / EP["energy_wh"][1], "subset_n": len(sub), "full_n": EP["n"]}
    # v1.6.4: trade-off of full, all-expert and expert-subset adaptation in the G1 gate runs before the freeze (descriptive)
    GR = R / "g1" / "gate_record_G1.json"; FL = json.loads(GR.read_text(encoding="utf-8"))["F_levels_released"]; OUT = R / "outputs" / "g1"
    def man(run): return json.loads((OUT / run / "m2" / "subset_manifest.json").read_text(encoding="utf-8"))["elements"]
    me = man("g1_moe_b1_eall_german_s101"); tot = me["total"]; rtr = me["by_kind"]["router"]["frozen"]
    for run in ("g1_moe_b1_full_german_s101", "g1_moe_b1_full_german_s102"):
        t = (LG / f"{run}_m2.log").read_text(encoding="utf-8", errors="replace")
        assert re.search(r"frozen routers: 28 router parameters \(0\.63M of 373\.4M elements\) frozen; 161/189 parameters trainable", t)
    rows = {}
    for key, runs, upd in (("full", ["g1_moe_b1_full_german_s101", "g1_moe_b1_full_german_s102"], tot - rtr),
                           ("all_experts", ["g1_moe_b1_eall_german_s101", "g1_moe_b1_eall_german_s102"], None),
                           ("subset_q14", ["g1_moe_b1_aa1_R1", "g1_moe_b1_aa1_R2"], None)):
        if upd is None:
            ms = {man(r)["trainable"] for r in runs}; assert len(ms) == 1 and {man(r)["total"] for r in runs} == {tot}; upd = ms.pop()
        gain = [-FL[r]["F"]["german"]["F"] for r in runs]; fd = [FL[r]["F"]["dclm"]["F"] for r in runs]; m_ = mem([f"{r}_m2.log" for r in runs])
        assert all(FL[r]["F"][c]["W"] == 500 for r in runs for c in ("german", "dclm"))
        rows[key] = {"runs": runs, "german_gain": gain, "dclm_F": fd, "F_per_gain": [f / g for f, g in zip(fd, gain)], "german_gain_mean": mean(gain), "dclm_F_mean": mean(fd),
                     "F_per_gain_mean": mean([f / g for f, g in zip(fd, gain)]), "updated_elements": upd, "updated_pct": 100 * upd / tot, "alloc_gib_max": m_["alloc_gib_max"], "reserved_gib_max": m_["reserved_gib_max"]}
    out["edge_tradeoff"] = {"description": "G1 gate runs before the freeze: MoE base 1, German target, 96 steps of 262,144 tokens with the registered adaptation recipe; F on the 500 even evaluation windows per corpus; two runs per row; descriptive, not a registered test",
        "sources": {"gate_record": {"path": "g1/gate_record_G1.json", "sha256": sha(GR), "field": "F_levels_released[run].F.{german,dclm}.F (german gain = -F)"},
                    "manifests": "outputs/g1/<run>/m2/subset_manifest.json: elements.trainable; full adaptation = elements.total - elements.by_kind.router.frozen (28 router tensors frozen, 161/189 tensors trainable in the log)",
                    "memory": "logs/queue_g1/<run>_m2.log, max over logged steps of train/mem_peak_alloc_gib"},
        "elements_total": tot, "router_elements_frozen": rtr, "rows": rows,
        "subset_over_full": {"german_gain": rows["subset_q14"]["german_gain_mean"] / rows["full"]["german_gain_mean"], "dclm_F": rows["subset_q14"]["dclm_F_mean"] / rows["full"]["dclm_F_mean"]}}
    p = V1 / "data" / "supplementary_values.json"; p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8"); print("wrote", p)

if __name__ == "__main__":
    main()
