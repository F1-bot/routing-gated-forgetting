"""G2 gate record and the gate-determined freeze parameters per g1g2_spec.json v1.3.0 (prereg v1.3 sec. 6) -- torch-free.

  python g2_gate.py --plan g2/plan_g2.json --g1-record g1/gate_record_G1.json --lookup signflip_lookup_v1.csv \
      --out g2/gate_record_G2.json [--c2-draws g0/c2_draws.json]

Order (spec 'order.G2'; m depends on C2): G2-2 -> G2-3 -> G2-4 -> G2-1 -> G2-5 -> freeze block. Each gate's variables are
computed from the run outputs and decided by gate_rules_v13 (the specification's rules):
  G2-2  c = theta0_checks sgd.c (lr_first_step_matched of R1, pair a = 1); runs = the 3 SGD runs at c/3, c, 3c (E1 flag,
        training loss at steps 1 and 24, target gain at step 24 on the 64-window German interim head paired in-process
        with step 0, clip fraction); gain24_muon = the same gain of the S1 R1 run of pair a = 1 (decision O-9).
        Failure: decision O-5 -- the smallest STABLE learning rate runs the 4-draw exploratory C2 block, else C2 is not run.
  G2-3  Fs/SEs = F_DCLM of the selected SGD run at steps 8, 24, 48 on the FULL 500-window set (step<NNN>__dclm__full.npz),
        paired with the in-process theta_0 full evaluation (step000) when present, else with the G1 theta_0 evaluation.
  G2-4  FB/SEB = F of the books A/A pairs a = 2 (from S1, or its G2 repeat when G1-2 changed q or steps) and a = 3 (G2).
  G2-1  I_AA of pairs a = 1, 2, 3 (final versions); beta0 and P-bar from the G0 record, or from the G0-7R recomputation
        record when G1's G0-7R says recompute; m = int(C1_confirmatory) + int(C2_confirmatory).
Exit codes: 0 = COMPLETE (freeze parameters in the record), 5 = HOLD (missing inputs, E1/E3 problems, pending gates).
The record releases what the specification records (G2-1 'I_AA per pair (with the F values used)').
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_common as GC  # noqa: E402
import gate_rules_v13 as GR  # noqa: E402


def _load(p):
    return GC.load_eval(p) if p and os.path.exists(p) else None


def _full_path(idx: GC.RunIndex, rid: str, set_name: str, step: int) -> str | None:
    root = idx.root(rid)
    return os.path.join(root, "interim", f"step{int(step):03d}__{set_name}__full.npz") if root else None


def _spec_check(plan: dict) -> dict:
    """Identity (path, version, revision, sha256) of the registered g1g2_spec.json read at gate time and the result of
    reproducing its test vectors with gate_rules_v13 (freeze precondition; the gate records it, the hook enforces it)."""
    gc = plan.get("gate_config") or {}
    sp = gc.get("spec") or os.path.join(HERE, "g1g2_spec.json")
    lk = gc.get("lookup") or os.path.join(HERE, "signflip_lookup_v1.csv")
    if not (os.path.exists(sp) and os.path.exists(lk)):
        return {"path": sp, "status": "missing", "rules_version": GR.SPEC_VERSION}
    r = GR.reproduce_all(sp, lk)
    return {**r["spec"], "rules_version": GR.SPEC_VERSION, "vectors_reproduced": r["n"] - len(r["failed"]),
            "vectors": r["n"], "reproduced": r["reproduced"], "lookup_sha256": r["lookup_sha256"]}


def _pending(gate: str, why: str) -> dict:
    return {"gate": gate, "status": "pending", "reason": why, "rule": None, "sets": {}, "values": {}}


def _decide(fn, *a, **kw) -> dict:
    try:
        r = fn(*a, **kw)
        r["status"] = "computed" if r.get("rule") else "reported"
        return r
    except ValueError as e:
        return {"status": "not_evaluable", "error": str(e), "rule": None, "sets": {}, "values": {}}


def _pair_F(idx: GC.RunIndex, plan: dict, pair: dict, code: str, problems: list, inputs: list):
    """F and SE keyed 'R1|CODE', 'R1|math', 'R2|CODE', 'R2|math' for one A/A pair (None if incomplete)."""
    bf = plan["base_eval_files"]["moe_b1"]
    F, SE = {}, {}
    for sub in ("R1", "R2"):
        rid = pair["runs"][sub]
        if idx.status(rid) != "done":
            problems.append(f"{rid}: run status {idx.status(rid)}")
            return None
        for c, key in ((code, "CODE"), ("math", "math")):
            pb, pr = bf.get(c), idx.eval_path(rid, c)
            if not (pb and pr and os.path.exists(pb) and os.path.exists(pr)):
                problems.append(f"{rid}: missing evaluation {c}")
                return None
            inputs += [pb, pr]
            f = GC.forgetting(GC.load_eval(pb), GC.load_eval(pr))
            F[f"{sub}|{key}"], SE[f"{sub}|{key}"] = f["F"], f["SE"]
    return F, SE


def _gain24(idx: GC.RunIndex, rid: str, target: str, inputs: list) -> float | None:
    b0, b24 = _load(idx.interim_path(rid, target, 0)), _load(idx.interim_path(rid, target, 24))
    if not (b0 and b24):
        return None
    inputs += [idx.interim_path(rid, target, 0), idx.interim_path(rid, target, 24)]
    return GC.gain(b0, b24)["G"]


def _sgd_runs(plan: dict, idx: GC.RunIndex, lrs: list, problems: list, inputs: list) -> list | None:
    runs = sorted((r for r in plan["runs"] if r["role"] == "sgd"), key=lambda r: r["lr_index"])
    out = []
    for run, lr in zip(runs, lrs):
        rid = run["id"]
        if idx.status(rid) != "done":
            problems.append(f"{rid}: run status {idx.status(rid)}")
            return None
        tl = idx.train_log(rid)
        recs = GC.read_train_steps(tl) if tl and os.path.exists(tl) else []
        if not recs:
            problems.append(f"{rid}: train log missing")
            return None
        inputs.append(tl)
        e1 = GC.e1_check(recs, int(run.get("warmup", 10)))
        l1, l24 = GC.loss_at(recs, 1), GC.loss_at(recs, 24)
        g = _gain24(idx, rid, "german", inputs)
        if e1["status"] != "checked" or l1 is None or l24 is None or g is None:
            problems.append(f"{rid}: E1 {e1['status']}, loss(1) {l1}, loss(24) {l24}, gain24 {g}")
            return None
        clip = GC.clip_fraction(recs)
        out.append({"run": rid, "lr": float(lr), "e1_divergent": bool(e1["diverged"]), "loss_step1": float(l1),
                    "loss_step24": float(l24), "gain24": float(g), "clip_fraction": float(clip["fraction"]),
                    "e1": e1, "max_norm_values": clip.get("max_norm_values")})
    return out if len(out) == 3 else None


def gate(plan: dict, g1rec: dict, lookup: GR.Lookup, *, plan_path: str = "", g1_path: str = "", c2_draws: dict | None = None,
         lookup_path: str = "") -> dict:
    d1 = g1rec["decision"]
    if d1.get("outcome") != "PROCEED":
        raise SystemExit(f"G1 outcome {d1.get('outcome')!r}: G2 is not evaluated (spec order; G2 only after G1 PROCEED)")
    g1plan = json.load(open(g1rec["inputs"]["plan"]["path"], encoding="utf-8"))
    rp = (g1rec["inputs"].get("repeat_plan") or {}).get("path")
    g1rplan = json.load(open(rp, encoding="utf-8")) if (d1.get("g1_1_lengthened") and rp) else None
    s1plan = g1rplan or g1plan                                   # final S1 versions (192-step repeat after G1-1, O-2)
    idx = GC.RunIndex(plan, [plan.get("state", "")])
    idx1 = GC.RunIndex(s1plan, [s1plan.get("state", "")])
    g0 = plan["g0"]
    code = g0["code_stream"]
    problems: list[str] = []
    inputs: list[str] = [p for p in (plan_path, g1_path, lookup_path, plan.get("gate_config_file")) if p]
    gates: dict[str, dict] = {}
    # ------------------------------------------------------------------ G2-2 SGD feasibility
    t0p = plan["sgd"]["theta0_checks"]
    t0 = json.load(open(t0p, encoding="utf-8")) if os.path.exists(t0p) else None
    if t0:
        inputs.append(t0p)
    s1pairs = {int(p["pair"]): p for p in s1plan.get("aa_pairs", [])}
    muon_rid = s1pairs[1]["runs"]["R1"] if 1 in s1pairs else None
    g_mu = _gain24(idx1, muon_rid, "german", inputs) if muon_rid else None
    runs = _sgd_runs(plan, idx, list(t0["sgd"]["grid"]) if t0 else [], problems, inputs) if t0 else None
    if t0 is None or runs is None or g_mu is None:
        gates["G2-2"] = _pending("G2-2", f"theta0 checks {bool(t0)}, SGD runs complete {runs is not None}, "
                                         f"Muon reference gain {g_mu} ({muon_rid})")
    else:
        gates["G2-2"] = _decide(GR.g2_2, float(t0["sgd"]["c"]), runs, g_mu)
        gates["G2-2"]["record"] = {"runs": runs, "muon_reference_run": muon_rid, "grid_rule": t0["sgd"].get("rule")}
    g22 = gates["G2-2"].get("sets", {}).get("G22_pass")
    o5_lr = None
    if gates["G2-2"].get("rule") == "fail":
        o5_lr = GR.o5_exploratory_lr(runs)
        gates["G2-2"]["O-5"] = {"smallest_stable_lr": o5_lr,
                                "consequence": "4-draw exploratory C2 block at this lr" if o5_lr is not None else
                                "no stable lr: C2 is not run and its budget is released"}
    # ------------------------------------------------------------------ G2-3 s*
    c2_feas = bool(g0.get("c2_matching_feasible"))
    if gates["G2-2"]["status"] != "computed":
        gates["G2-3"] = _pending("G2-3", "requires G2-2")
    elif not g22:
        gates["G2-3"] = _decide(GR.g2_3, {}, {}, False, bool(g0.get("g0_3_pass")), c2_feas)
    else:
        sel = next(x["run"] for x in runs if x["lr"] == gates["G2-2"]["sets"]["sgd_lr"])
        b0p = _full_path(idx, sel, "dclm", 0)
        ref = "in-process theta_0 (step000 full)" if b0p and os.path.exists(b0p) else "G1 theta_0 evaluation"
        if not (b0p and os.path.exists(b0p)):
            b0p = g1plan["base_eval_files"]["moe_b1"]["dclm"]
        b0 = _load(b0p)
        Fs, SEs, rows = {}, {}, {}
        for s in GR.S_STAR_CANDIDATES:
            bs = _load(_full_path(idx, sel, "dclm", s))
            if b0 is None or bs is None:
                problems.append(f"{sel}: missing full DCLM evaluation at step {s} (step<NNN>__dclm__full.npz) or theta_0")
                continue
            inputs += [b0p, _full_path(idx, sel, "dclm", s)]
            f = GC.forgetting(b0, bs)
            Fs[s], SEs[s], rows[s] = f["F"], f["SE"], {"F": f["F"], "SE": f["SE"], "W": f["W"]}
        if len(Fs) != 3:
            gates["G2-3"] = _pending("G2-3", "full DCLM evaluations of the selected SGD run incomplete")
        else:
            gates["G2-3"] = _decide(GR.g2_3, Fs, SEs, True, bool(g0.get("g0_3_pass")), c2_feas)
            gates["G2-3"]["record"] = {"run": sel, "theta0_reference": ref, "per_step": rows}
    s23 = gates["G2-3"].get("sets", {})
    if gates["G2-2"].get("rule") == "fail":
        C2_conf, C2_draws, s_star = False, (GR.C2_DRAWS_EXPLORATORY if o5_lr is not None else 0), None
        sgd_lr, c2_status = o5_lr, ("exploratory (G2-2 failed; O-5: smallest stable lr)" if o5_lr is not None else
                                    "not run (G2-2 failed and no stable lr; O-5)")
    elif gates["G2-3"]["status"] == "computed":
        C2_conf, C2_draws, s_star = bool(s23.get("C2_confirmatory")), s23.get("C2_draws"), s23.get("s_star")
        sgd_lr = gates["G2-2"]["sets"]["sgd_lr"]
        c2_status = "confirmatory" if C2_conf else f"exploratory ({gates['G2-3']['rule']})"
    else:
        C2_conf = C2_draws = s_star = sgd_lr = c2_status = None
    # ------------------------------------------------------------------ G2-4 C1 configuration check (books)
    pivot = bool(d1.get("pivot"))
    gpairs = {int(p["pair"]): p for p in plan.get("aa_pairs", [])}
    p2 = gpairs.get(2) if d1.get("repeat_pair2_in_g2") else s1pairs.get(2)
    p2_idx = idx if d1.get("repeat_pair2_in_g2") else idx1
    p2_plan = plan if d1.get("repeat_pair2_in_g2") else s1plan
    fb2 = fb3 = None
    if not pivot:
        fb2 = _pair_F(p2_idx, p2_plan, p2, code, problems, inputs) if p2 else None
        fb3 = _pair_F(idx, plan, gpairs[3], code, problems, inputs) if 3 in gpairs else None
    if pivot:
        gates["G2-4"] = _decide(GR.g2_4, {}, {}, True)
    elif fb2 is None or fb3 is None:
        gates["G2-4"] = _pending("G2-4", f"books A/A pairs incomplete (a = 2: {fb2 is not None}, a = 3: {fb3 is not None})")
    else:
        FB = {f"2|{k}": v for k, v in fb2[0].items()} | {f"3|{k}": v for k, v in fb3[0].items()}
        SEB = {f"2|{k}": v for k, v in fb2[1].items()} | {f"3|{k}": v for k, v in fb3[1].items()}
        gates["G2-4"] = _decide(GR.g2_4, FB, SEB, False)
        gates["G2-4"]["record"] = {"F": FB, "SE": SEB, "pair2_source": "G2 repeat in the G1-2 configuration"
                                   if d1.get("repeat_pair2_in_g2") else "S1"}
    s24 = gates["G2-4"].get("sets", {})
    C1_conf = (not pivot) and not s24.get("pivot", False) if gates["G2-4"]["status"] == "computed" else (False if pivot else None)
    # ------------------------------------------------------------------ G0-7R values (beta0, P-bar)
    g07r_rule = d1.get("g0_7r")
    beta0 = [g0["beta0"].get("moe_b1"), g0["beta0"].get("moe_b2")]
    margin_limited = dict(g0.get("margin_limited") or {})
    Pbar, Pbar_src = g0.get("P_bar_c1_moe_b1"), g0.get("P_bar_source")
    g07r_ok = True
    if g07r_rule == "recompute":
        rp7 = plan.get("g07r_record")
        if rp7 and os.path.exists(rp7):
            r7 = json.load(open(rp7, encoding="utf-8"))
            inputs.append(rp7)
            beta0 = [r7["beta0"]["moe_b1"], r7["beta0"]["moe_b2"]]
            margin_limited = r7.get("margin_limited", margin_limited)
            Pbar, Pbar_src = r7.get("P_bar", Pbar), f"G0-7R recomputation ({rp7})"
        else:
            g07r_ok = False
            problems.append(f"G0-7R recompute required by G1 but its record is missing ({rp7})")
    # ------------------------------------------------------------------ G2-1 draws per C1 cell
    if C1_conf is False:
        gates["G2-1"] = _decide(GR.g2_1, [None, None, None], [0.25, 0.25], 1.0, False, bool(C2_conf), lookup)
    elif C1_conf is None or C2_conf is None or not g07r_ok or Pbar is None or None in beta0:
        gates["G2-1"] = _pending("G2-1", f"requires G2-4 ({C1_conf}), G2-3 ({C2_conf}), G0-7R values ({g07r_ok}), "
                                         f"P-bar ({Pbar}), beta0 ({beta0})")
    else:
        pairs_used, I, ok = {}, [], True
        p1 = s1pairs.get(1)
        for a, (pr, ix, pl) in {1: (p1, idx1, s1plan), 2: (p2, p2_idx, p2_plan), 3: (gpairs.get(3), idx, plan)}.items():
            fs = _pair_F(ix, pl, pr, code, problems, inputs) if pr else None
            if fs is None:
                ok = False
                break
            pairs_used[a] = {"F": fs[0], "target": pr["target"], "runs": pr["runs"], "I_AA": GR.i_aa(fs[0])}
            I.append(pairs_used[a]["I_AA"])
        if not ok:
            gates["G2-1"] = _pending("G2-1", "A/A pairs a = 1-3 incomplete")
        else:
            gates["G2-1"] = _decide(GR.g2_1, I, beta0, float(Pbar), True, bool(C2_conf), lookup)
            gates["G2-1"]["record"] = {"pairs": pairs_used, "beta0": beta0, "Pbar": Pbar, "Pbar_source": Pbar_src,
                                       "lookup_sha256": lookup.sha256}
    # ------------------------------------------------------------------ G2-5 per-expert gradient SNR (reported)
    snr_p = (plan.get("snr") or {}).get("out")
    if snr_p and os.path.exists(snr_p):
        inputs.append(snr_p)
        snr = json.load(open(snr_p, encoding="utf-8"))
        g25 = {"gate": "G2-5", "status": "reported", "rule": None, "sets": {},
               "values": {"summary": snr.get("summary"), "windows": snr.get("windows"),
                          "train_batch_windows": snr.get("train_batch_windows")}}
        if c2_draws:
            units = set()
            for e in c2_draws.get("entries", []):
                for s in ("A", "B"):
                    units.update((e.get("subsets") or {}).get(s, []))
            per = snr.get("per_unit", {})
            v = sorted(per[u]["snr_train"] for u in units if u in per and math.isfinite(per[u]["snr_train"]))
            g25["values"]["c2_pair_experts"] = {"units": len(units), "finite": len(v), "median_snr_train": v[len(v) // 2] if v else None,
                                                "share_snr_train_below_1": (sum(1 for x in v if x < 1) / len(v)) if v else None}
        gates["G2-5"] = g25
    else:
        gates["G2-5"] = _pending("G2-5", f"grad_snr output missing ({snr_p})")
    # ------------------------------------------------------------------ hold reasons
    hold = []
    for r in plan["runs"]:
        if r["role"] == "aa" and idx.status(r["id"]) == "done":
            tl = idx.train_log(r["id"])
            if tl and os.path.exists(tl):
                e1 = GC.e1_check(GC.read_train_steps(tl), r.get("warmup", 10))
                if e1.get("diverged"):
                    hold.append(f"E1 divergence in {r['id']}: {e1['reasons']}")
            if r.get("subset_spec") and idx.audit_ok(r["id"]) is not True:
                hold.append(f"{r['id']}: E3 bitwise audit not ok")
    open_ = [g for g in ("G2-2", "G2-3", "G2-4", "G2-1") if gates[g]["status"] in ("pending", "not_evaluable")]
    if open_:
        hold.append(f"gates pending or not evaluable: {open_}")
    if not g07r_ok:
        hold.append("G0-7R recomputation record missing")
    outcome = "HOLD" if hold else "COMPLETE"
    s21 = gates["G2-1"].get("sets", {})
    n_d = s21.get("n_d")
    m = (int(bool(C1_conf)) + int(bool(C2_conf))) if (C1_conf is not None and C2_conf is not None) else None
    freeze = {
        "holm_family": [t for t, ok in (("C1", C1_conf), ("C2", C2_conf)) if ok] if m is not None else None,
        "m": m, "C1_confirmatory": C1_conf,
        "C1_targets_per_base": {"moe_b1": ["books"], "moe_b2": ["books"]},
        "n_d": n_d, "underpowered_flag": s21.get("underpowered_flag"),
        "c1_draws_per_base": 2 * int(n_d) if n_d else None,
        "c1_draw_indices": list(range(1, 2 * int(n_d) + 1)) if n_d else None,      # P-13: d = 1..2 n_d of the books sequence
        "q": d1.get("q"), "steps": d1.get("steps"), "f": g0.get("f"),
        "tau": "pending: crossover.reg_choose_tau at the freeze (q = 1/2, weighted generator)" if d1.get("q") == GR.Q_WIDE else None,
        "code_stream": code, "beta0_1": beta0[0], "beta0_2": beta0[1], "margin_limited": margin_limited,
        "C2_confirmatory": C2_conf, "C2_draws": C2_draws, "s_star": s_star, "sgd_lr": sgd_lr, "c2_status": c2_status,
        "C2_matching_set": g0.get("c2_matched"),
        "NESTORB_signal": d1.get("NESTORB_signal"), "NESTORB_label": d1.get("NESTORB_label"), "C3N_label": d1.get("C3N_label"),
        "P_bar": Pbar, "P_bar_source": Pbar_src, "g0_7r": g07r_rule,
        "pivot": pivot or bool(s24.get("pivot", False)),
    }
    return {"stage": "G2", "spec": _spec_check(plan), "prereg": "v1.3 sec. 6",
            "outcome": outcome, "hold_reasons": hold, "gates": gates, "freeze": freeze, "problems": problems,
            "inputs": {"plan": {"path": os.path.abspath(plan_path) if plan_path else None, "sha256": plan.get("sha256")},
                       "g1_record": {"path": os.path.abspath(g1_path) if g1_path else None, "sha256": g1rec.get("sha256")},
                       "files": GC.inputs_manifest(inputs)},
            "appendix_b": {
                "G2-2 c; grid; per-lr stability, gain ratio, clip fraction; selected SGD learning rate":
                    f"{gates['G2-2'].get('values', {}).get('c')}; {gates['G2-2'].get('values', {}).get('grid')}; "
                    f"stable {gates['G2-2'].get('values', {}).get('stable')}; ratio {gates['G2-2'].get('values', {}).get('gain_ratio')}; "
                    f"selected {sgd_lr}",
                "G2-3 F_DCLM and SE of the selected SGD run at steps 8, 24, 48; s*; C2 status and draws":
                    f"s* = {s_star}; C2 {c2_status}; draws {C2_draws}",
                "G2-4 books A/A F and SE (code stream, math); C1 status": f"{gates['G2-4'].get('rule')}; C1 confirmatory {C1_conf}",
                "G2-1 I_AA per pair; sigma_I, UCL80, delta, m, alpha, power(12), power(20); n_d; flag":
                    "; ".join(f"{k} {gates['G2-1'].get('values', {}).get(k)}" for k in ("sigma_I", "UCL80", "delta", "m", "alpha", "p12", "p20"))
                    + f"; n_d {n_d}; flag {s21.get('underpowered_flag')}",
                "G2-5 per-expert gradient SNR (median, quartiles; share with snr_train < 1)": f"{gates['G2-5'].get('status')}"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G2 gate record (spec 1.3.0; torch-free)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--g1-record", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--c2-draws", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    plan = json.load(open(a.plan, encoding="utf-8"))
    g1 = json.load(open(a.g1_record, encoding="utf-8"))
    if not GC.verify_record(a.g1_record):
        print("G1 record hash does not verify", file=sys.stderr)
        return 5
    c2d = json.load(open(a.c2_draws, encoding="utf-8")) if a.c2_draws else None
    rec = gate(plan, g1, GR.Lookup(a.lookup), plan_path=a.plan, g1_path=a.g1_record, c2_draws=c2d, lookup_path=a.lookup)
    h = GC.finalize_record(rec, a.out)
    f = rec["freeze"]
    print(f"G2 outcome {rec['outcome']} | m={f['m']} holm={f['holm_family']} n_d={f['n_d']} s*={f['s_star']} sgd_lr={f['sgd_lr']} | "
          f"C2 {f['c2_status']} | C1 confirmatory {f['C1_confirmatory']} | record {a.out} sha256 {h}")
    for gid, g in rec["gates"].items():
        print(f"  {gid:6s} {g.get('status'):14s} {g.get('rule')}")
    for r in rec["hold_reasons"]:
        print("  hold:", r)
    for p in rec["problems"][:20]:
        print("  problem:", p)
    return 0 if rec["outcome"] == "COMPLETE" else 5


if __name__ == "__main__":
    sys.exit(main())
