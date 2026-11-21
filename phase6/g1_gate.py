"""G1 gate record per g1g2_spec.json v1.3.0 (preregistration v1.3 sec. 6) -- torch-free.

  python g1_gate.py --plan g1/plan_g1.json [--repeat-plan g1r/plan_g1r.json] --out g1/gate_record_G1.json

Reads the plan written by g1g2_plan.py, the run_queue state named in it, the per-window evaluation files, each run's
train_steps.jsonl and the gate inputs named in the plan's gate config (g1/g1_gate_config.json). Every gate's variables are
computed from these files and decided by gate_rules_v13 (the specification's rules; the 42 test vectors of the
specification are reproduced by test_gates.py). Order and requirements as in spec 'order.G1' and each gate's 'requires':
  G1-0 -> G1-4 -> G1-1 (-> 192-step repeat, O-2) -> G1-2, G1-3 -> G0-9L, G1-5, G1-6, G1-NS5 -> G0-7R.
A gate whose requirement did not pass is recorded as not_evaluated (no decision on runs that a failed gate invalidates).

Exit codes: 0 PROCEED (G2 may run; C1 may be PIVOT), 3 REPEAT_192 (G1-1 lengthen at attempt 1: run the g1r queue, then
re-gate with --repeat-plan), 4 STOP (G1-1 stop at attempt 2: the study stops), 5 HOLD (G1-0 or G1-4 blocked G1, E1
divergence, missing or malformed inputs: a human decision is needed before anything else runs).

Information firewall (prereg sec. 4.10): the record releases F levels of FULL, E-ALL, A/A and DCLM-CONT runs and the A/A
F_sym tables; no I_AA and no relation to P (I_AA is computed in G2-1 only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_common as GC  # noqa: E402
import gate_rules_v13 as GR  # noqa: E402

EXIT = {"PROCEED": 0, "REPEAT_192": 3, "STOP": 4, "HOLD": 5}
SUBSET_ROLES = ("aa", "aa_nestor", "eall")


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


# ============================================================================= per-run inputs
def _f_table(idx: GC.RunIndex, run: dict, base_files: dict, problems: list, inputs: list) -> dict:
    """{set: F record} for the run's main evaluation sets, paired by window hash with the base's theta_0 evaluation."""
    out = {}
    bf = base_files.get(run["base"], {})
    for s in run.get("eval_sets", []):
        p, b = idx.eval_path(run["id"], s), bf.get(s)
        if not p or not os.path.exists(p):
            problems.append(f"{run['id']}: missing evaluation {s} ({p})")
            continue
        if not b or not os.path.exists(b):
            problems.append(f"{run['id']}: missing base evaluation {s} ({b})")
            continue
        try:
            out[s] = GC.forgetting(GC.load_eval(b), GC.load_eval(p))
            inputs += [p, b]
        except ValueError as e:
            problems.append(f"{run['id']}: {s}: {e}")
    for s in run.get("reflect_sets", []):                       # completeness only (values not released)
        root = idx.root(run["id"])
        p = os.path.join(root, "eval_reflect", f"reflect__{s}.npz") if root else None
        if not p or not os.path.exists(p):
            problems.append(f"{run['id']}: missing reflected evaluation {s}")
        else:
            inputs.append(p)
    return out


def _run_checks(idx: GC.RunIndex, run: dict, problems: list, e1_div: list, inputs: list) -> dict:
    st = idx.status(run["id"])
    rec = {"status": st}
    if run["role"] == "negctrl":
        return rec
    if st != "done":
        problems.append(f"{run['id']}: run status {st}")
        return rec
    tl = idx.train_log(run["id"])
    if tl and os.path.exists(tl):
        e1 = GC.e1_check(GC.read_train_steps(tl), int(run.get("warmup", 10)))
        rec["e1"] = e1
        inputs.append(tl)
        if e1["diverged"]:
            e1_div.append(run["id"])
        elif e1["status"] != "checked":
            problems.append(f"{run['id']}: E1 not checkable ({e1['reasons']})")
    else:
        problems.append(f"{run['id']}: train log missing ({tl})")
    if run.get("subset_spec"):
        rec["audit_ok"] = idx.audit_ok(run["id"])
    return rec


def _fsym_runs(plan: dict, idx: GC.RunIndex, base_files: dict, runs_out: dict, problems: list, inputs: list):
    """Symmetrized forgetting of the A/A runs (theory verification): live, selection-pinned and pinned F_sym per
    evaluation set, the gate-pathway / flip decomposition, and the one-sided flag F_sym,pin + 2 SE < 0 (reported)."""
    out, flags = {}, []
    for run in plan["runs"]:
        if not run.get("pin_sets") or runs_out.get(run["id"], {}).get("status") != "done":
            continue
        t = GC.fsym_table(base_files.get(run["base"], {}), idx.root(run["id"]), run["pin_sets"])
        out[run["id"]] = t
        for s in run["pin_sets"]:
            if not t.get(s, {}).get("complete"):
                problems.append(f"{run['id']}: incomplete live/selection-pinned/pinned evaluations for {s} (F_sym)")
            elif t[s]["pin"].get("flag_out_of_domain"):
                flags.append({"run": run["id"], "set": s, "F_sym_pin": t[s]["pin"]["F_sym"], "SE": t[s]["pin"]["SE"]})
        root = idx.root(run["id"])
        d = os.path.join(root, "eval_pin") if root else None
        if d and os.path.isdir(d):
            inputs += [os.path.join(d, f) for f in os.listdir(d)]
    return out, flags


def _collect(plan: dict, idx: GC.RunIndex, problems: list, e1_div: list, inputs: list) -> dict:
    base_files = plan.get("base_eval_files", {})
    out = {}
    for run in plan["runs"]:
        chk = _run_checks(idx, run, problems, e1_div, inputs)
        F = _f_table(idx, run, base_files, problems, inputs) if chk.get("status") == "done" and run["role"] != "negctrl" else {}
        out[run["id"]] = {"role": run["role"], "base": run["base"], "target": run.get("target"), "seed": run["seed"],
                          "steps": run["steps"], "pair": run.get("pair"), "subset": run.get("subset"), **chk, "F": F}
    return out


# ============================================================================= gate inputs (variables)
def pipeline_counts(gcfg: dict, inputs: list) -> dict | None:
    """G1-0 variables from the GPU pipeline audit: junit XML (pytest --junitxml) if present, else the pytest summary line."""
    pt = gcfg.get("pipeline_tests") or {}
    xml, log = pt.get("junit"), pt.get("log")
    if xml and os.path.exists(xml):
        root = ET.parse(xml).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        inputs.append(xml)
        return {"tests_passed": tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"],
                "tests_failed": tot["failures"], "tests_errors": tot["errors"], "skipped": tot["skipped"], "source": xml,
                "sha256": GC.file_sha256(xml)}
    if log and os.path.exists(log):
        txt = open(log, encoding="utf-8", errors="replace").read()
        summ = [ln for ln in txt.splitlines() if re.search(r"\b(passed|failed|error)", ln) and " in " in ln]
        if summ:
            ln = summ[-1]
            get = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", ln)) else 0   # noqa: E731
            inputs.append(log)
            return {"tests_passed": get("passed"), "tests_failed": get("failed"), "tests_errors": get("error"),
                    "skipped": get("skipped"), "source": log, "summary_line": ln.strip(), "sha256": GC.file_sha256(log)}
    return None


def negctrl_audit(idx: GC.RunIndex, run: dict | None) -> str | None:
    """'fail' (the audit caught the unmasked decay, as required), 'ok' (the audit passed it: broken), None (pending)."""
    if run is None:
        return None
    st = idx.state.get(run["id"], {})
    if st.get("audit") in ("ok", "fail"):
        return st["audit"]
    if st.get("status") == "done" and st.get("audit_ok") is False:
        return "fail"
    if st.get("status") == "failed" and ("PASSED" in str(st.get("error", "")) or st.get("audit_ok") is True):
        return "ok"
    return None


def _pair_F(table: dict, pair: dict, code: str) -> tuple[dict, dict] | None:
    F, SE = {}, {}
    for sub in ("R1", "R2"):
        f = table.get(pair["runs"][sub], {}).get("F", {})
        for c, key in ((code, "CODE"), ("math", "math")):
            if c not in f:
                return None
            F[f"{sub}|{key}"], SE[f"{sub}|{key}"] = f[c]["F"], f[c]["SE"]
    return F, SE


def _decide(fn, *a, **kw) -> dict:
    try:
        r = fn(*a, **kw)
        r["status"] = "computed" if r.get("rule") else "reported"
        return r
    except ValueError as e:
        return {"status": "not_evaluable", "error": str(e), "rule": None, "sets": {}, "values": {}}


def _blocked(gate: str, why: str) -> dict:
    return {"gate": gate, "status": "not_evaluated", "reason": why, "rule": None, "sets": {}, "values": {}}


def _pending(gate: str, why: str) -> dict:
    return {"gate": gate, "status": "pending", "reason": why, "rule": None, "sets": {}, "values": {}}


# ============================================================================= gate
def gate(plan: dict, repeat_plan: dict | None = None, plan_path: str = "", repeat_path: str = "") -> dict:
    idx = GC.RunIndex(plan, [plan.get("state", "")])
    ridx = GC.RunIndex(repeat_plan, [repeat_plan.get("state", "")]) if repeat_plan else None
    g0, gcfg = plan["g0"], plan.get("gate_config") or {}
    code = g0["code_stream"]
    problems: list[str] = []
    e1_div: list[str] = []
    inputs: list[str] = [p for p in (plan_path, repeat_path, plan.get("gate_config_file")) if p]
    runs_out = _collect(plan, idx, problems, e1_div, inputs)
    rep_out = _collect(repeat_plan, ridx, problems, e1_div, inputs) if ridx else {}
    fsym_out, fsym_flags = _fsym_runs(plan, idx, plan.get("base_eval_files", {}), runs_out, problems, inputs)
    if ridx:
        f2, fl2 = _fsym_runs(repeat_plan, ridx, repeat_plan.get("base_eval_files", plan.get("base_eval_files", {})), rep_out,
                             problems, inputs)
        fsym_out.update(f2)
        fsym_flags += fl2
    gates: dict[str, dict] = {}
    # ------------------------------------------------------------------ G1-0 GPU pipeline audit
    pc = pipeline_counts(gcfg, inputs)
    if pc is None:
        gates["G1-0"] = _pending("G1-0", f"no pipeline-test result ({gcfg.get('pipeline_tests')})")
    else:
        gates["G1-0"] = _decide(GR.g1_0, pc["tests_passed"], pc["tests_failed"], pc["tests_errors"])
        gates["G1-0"]["record"] = {k: pc.get(k) for k in ("source", "sha256", "skipped", "summary_line")}
        cs = gcfg.get("code_record")
        if cs and os.path.exists(cs):
            gates["G1-0"]["record"]["code_under_test_sha256"] = GC.file_sha256(cs)
            inputs.append(cs)
    # ------------------------------------------------------------------ G1-4 bitwise audit and negative control
    neg = next((r for r in plan["runs"] if r["role"] == "negctrl"), None)
    na = negctrl_audit(idx, neg)
    subset_runs = {rid: v for rid, v in {**runs_out, **rep_out}.items() if v["role"] in SUBSET_ROLES}
    pend = [rid for rid, v in subset_runs.items() if v.get("status") != "done" or v.get("audit_ok") is None]
    if gates["G1-0"].get("rule") != "pass":
        gates["G1-4"] = _blocked("G1-4", "requires G1-0 pass")
    elif na is None or pend:
        gates["G1-4"] = _pending("G1-4", f"negative control audit {na}; subset runs without an audit result: {pend}")
    else:
        gates["G1-4"] = _decide(GR.g1_4, na, {rid: ("ok" if v["audit_ok"] else "fail") for rid, v in subset_runs.items()})
    # ------------------------------------------------------------------ G1-1 FULL signal (attempt 1, then 2 after the repeat)
    def full_values(table: dict) -> dict:
        return {rid: v["F"]["dclm"] for rid, v in table.items()
                if v["role"] == "full" and v["base"] == "moe_b1" and "dclm" in v["F"]}

    attempt, final_table = 1, runs_out
    if gates["G1-4"].get("rule") != "pass":
        gates["G1-1"] = _blocked("G1-1", "requires G1-4 pass")
    else:
        fv = full_values(runs_out)
        if len(fv) != 4:
            gates["G1-1"] = _pending("G1-1", f"{len(fv)} of 4 MoE base-1 FULL DCLM evaluations")
        else:
            gates["G1-1"] = _decide(GR.g1_1, [x["F"] for x in fv.values()], 1)
            gates["G1-1"]["record"] = {"attempt_1": {rid: {k: x[k] for k in ("F", "SE", "W")} for rid, x in fv.items()}}
            if gates["G1-1"].get("rule") == "lengthen" and ridx:
                fv2 = full_values(rep_out)
                if len(fv2) != 4:
                    gates["G1-1"] = dict(_pending("G1-1", f"repeat: {len(fv2)} of 4 192-step FULL DCLM evaluations"),
                                         first=gates["G1-1"])
                else:
                    first = gates["G1-1"]
                    gates["G1-1"] = _decide(GR.g1_1, [x["F"] for x in fv2.values()], 2)
                    gates["G1-1"]["record"] = {**first["record"], "attempt_2": {rid: {k: x[k] for k in ("F", "SE", "W")}
                                                                                for rid, x in fv2.items()}}
                    gates["G1-1"]["first"] = {k: first[k] for k in ("rule", "values")}
                    attempt, final_table = 2, rep_out
    g11 = gates["G1-1"].get("rule")
    lengthened = attempt == 2
    g11_ok = g11 == "pass"
    # ------------------------------------------------------------------ G1-2 subset signal (books A/A pair a = 2, S1)
    final_plan = repeat_plan if lengthened else plan
    pairs = {int(p["pair"]): p for p in final_plan.get("aa_pairs", [])}
    if not g11_ok:
        gates["G1-2"] = _blocked("G1-2", "requires G1-1 pass" + (" (192-step repeat pending: O-2)" if g11 == "lengthen" else ""))
    elif 2 not in pairs:
        gates["G1-2"] = _pending("G1-2", "books A/A pair a = 2 not in the plan")
    else:
        fs = _pair_F(final_table, pairs[2], code)
        gates["G1-2"] = _pending("G1-2", "pair a = 2 evaluations incomplete") if fs is None else _decide(GR.g1_2, *fs)
        if fs is not None:
            gates["G1-2"]["record"] = {"pair": {k: pairs[2][k] for k in ("pair", "target", "generator_seed", "data_seed", "k", "q")},
                                       "F": fs[0], "SE": fs[1], "steps": pairs[2].get("steps")}
    # ------------------------------------------------------------------ G1-3 NESTOR books A/A subset
    nr = next((v for v in final_table.values() if v["role"] == "aa_nestor"), None)
    if not (plan.get("nestor") or {}).get("run"):
        gates["G1-3"] = {"gate": "G1-3", "status": "not_applicable", "reason": "NESTOR runs omitted (G0-5)", "rule": None,
                         "sets": {"NESTORB_signal": False}, "values": {}}
    elif not g11_ok:
        gates["G1-3"] = _blocked("G1-3", "requires G1-1 pass")
    elif nr is None or not all(c in nr["F"] for c in (code, "math")):
        gates["G1-3"] = _pending("G1-3", "NESTOR books A/A subset evaluations incomplete")
    else:
        FN = {"CODE": nr["F"][code]["F"], "math": nr["F"]["math"]["F"]}
        SEN = {"CODE": nr["F"][code]["SE"], "math": nr["F"]["math"]["SE"]}
        gates["G1-3"] = _decide(GR.g1_3, FN, SEN)
        gates["G1-3"]["record"] = {"F": FN, "SE": SEN}
    # ------------------------------------------------------------------ G0-9L NESTOR routing-health label (O-8)
    rng = (gcfg.get("appC6_ranges") or {}).get("ranges") or {}
    rh_path = gcfg.get("routing_health")
    if not rng:
        gates["G0-9L"] = _decide(GR.g0_9l, 0, 0, False)
        gates["G0-9L"]["reason"] = "App. C.6 ranges not transcribed in the G1 gate config (MISSING INPUT, decision O-8)"
    elif not (rh_path and os.path.exists(rh_path)):
        gates["G0-9L"] = _pending("G0-9L", f"routing-health diagnostics missing ({rh_path})")
    else:
        rh = json.load(open(rh_path, encoding="utf-8"))
        inputs.append(rh_path)
        cnt = GR.routing_health_counts(rh.get("routers", {}), rng)
        gates["G0-9L"] = _decide(GR.g0_9l, cnt["routers_assessed"], cnt["routers_outside"], True)
        gates["G0-9L"]["record"] = {"per_router": cnt["per_router"], "ranges_source": (gcfg.get("appC6_ranges") or {}).get("source")}
    # ------------------------------------------------------------------ G1-5 platform duplicate (reported; O-3)
    nf = {v["seed"]: v for v in runs_out.values() if v["role"] == "full" and v["base"] == "nestor" and "dclm" in v["F"]}
    dup_seed = int(plan["config"]["seeds"]["h100_duplicate_seed"])
    if len(nf) == 2 and dup_seed in nf:
        other = next(s for s in nf if s != dup_seed)
        F_h = None
        hf = plan.get("h100_duplicate_file")
        if hf and os.path.exists(hf):
            inputs.append(hf)
            if hf.endswith(".npz"):
                bf = plan.get("base_eval_files", {}).get("nestor", {}).get("dclm")
                F_h = GC.forgetting(GC.load_eval(bf), GC.load_eval(hf))["F"]
            else:
                h = json.load(open(hf, encoding="utf-8"))
                F_h = float(h.get("F_dclm", h.get("F")))
        gates["G1-5"] = _decide(GR.g1_5, F_h, nf[dup_seed]["F"]["dclm"]["F"], nf[other]["F"]["dclm"]["F"])
        gates["G1-5"]["record"] = {"duplicate_seed": dup_seed, "other_seed": other, "file": hf}
    else:
        gates["G1-5"] = _pending("G1-5", "the two local NESTOR FULL German runs are incomplete")
    # ------------------------------------------------------------------ G1-6 DCLM-CONT (reported)
    dc = next((v for v in runs_out.values() if v["role"] == "dclm_cont" and "dclm" in v["F"]), None)
    if dc:
        fulls = {rid: v["F"]["dclm"]["F"] for rid, v in {**runs_out, **rep_out}.items()
                 if v["role"] == "full" and v["base"] == "moe_b1" and "dclm" in v["F"]}
        gates["G1-6"] = _decide(GR.g1_6, fulls, dc["F"]["dclm"]["F"])
        gates["G1-6"]["record"] = {"held_out_tokens": bool(next((r for r in plan["runs"] if r["role"] == "dclm_cont"), {})
                                                           .get("held_out_tokens"))}
    else:
        gates["G1-6"] = _pending("G1-6", "DCLM-CONT run incomplete")
    # ------------------------------------------------------------------ G1-NS5 (reported)
    t0p = plan.get("theta0_checks")
    if t0p and os.path.exists(t0p):
        t0 = json.load(open(t0p, encoding="utf-8"))
        inputs.append(t0p)
        gates["G1-NS5"] = {"gate": "G1-NS5", "status": "reported", "rule": None, "sets": {},
                           "values": {"rel_diff": (t0.get("ns5_scale_check") or {}).get("per_scale"),
                                      "d_muon_step1": t0.get("d_muon_unit_test")}}
    else:
        gates["G1-NS5"] = _pending("G1-NS5", f"theta0 checks missing ({t0p})")
    # ------------------------------------------------------------------ G0-7R (procedure; q from G1-2)
    if gates["G1-2"].get("status") == "computed":
        gates["G0-7R"] = _decide(GR.g0_7r, gates["G1-2"]["rule"] == "widen", False, bool(gcfg.get("f_or_stream_changed", False)))
    else:
        gates["G0-7R"] = _blocked("G0-7R", "requires G1-2")
    # ------------------------------------------------------------------ decision
    reasons, outcome = [], None
    for gid in ("G1-0", "G1-4"):
        if gates[gid].get("sets", {}).get("g1_blocked"):
            reasons.append(f"{gid} {gates[gid]['rule']}: {gates[gid]['action']}")
    if e1_div:
        reasons.append(f"E1 divergence in {sorted(set(e1_div))}")
    decision_gates = ["G1-0", "G1-4", "G1-1"] + (["G1-2", "G1-3"] if g11_ok else [])
    open_ = [g for g in decision_gates if gates[g]["status"] in ("pending", "not_evaluable")]
    if g11 == "stop":
        outcome = "STOP"
        reasons.append("G1-1 stop: mean F_DCLM < 0.010 nats after the 192-step repeat; the study stops")
    elif reasons or open_:
        outcome = "HOLD"
        if open_:
            reasons.append(f"gates pending or not evaluable: {open_}")
        if problems:
            reasons.append(f"{len(problems)} input problems")
    elif g11 == "lengthen":
        outcome = "REPEAT_192"
        reasons.append("G1-1 lengthen: all adaptation runs go to 192 steps; repeat the MoE FULL runs, both S1 A/A pairs and the "
                       "NESTOR A/A subset at 192 steps (g1r queue, O-2), then re-gate with --repeat-plan")
    else:
        outcome = "PROCEED"
    s12 = gates["G1-2"].get("sets", {})
    pivot = bool(s12.get("pivot")) or bool(g0.get("pivot"))
    steps = None
    if outcome == "PROCEED":
        steps = GR.STEPS_LONG if (lengthened or s12.get("steps") == GR.STEPS_LONG) else GR.STEPS_BASE
    q = None if pivot else s12.get("q")
    s09 = gates["G0-9L"].get("sets", {})
    decision = {
        "outcome": outcome, "reasons": reasons, "attempt": attempt, "g1_1_lengthened": lengthened,
        "g1_2_rule": gates["G1-2"].get("rule"), "pivot": pivot, "C1_confirmatory": (not pivot) if outcome == "PROCEED" else None,
        "c1": "PIVOT" if pivot else ("FEASIBLE" if outcome == "PROCEED" else None),
        "q": q, "steps": steps, "generator": None if q is None else ("weighted" if q == GR.Q_WIDE else "registered"),
        "repeat_pair2_in_g2": gates["G1-2"].get("rule") in ("lengthen", "widen"),
        "NESTORB_signal": gates["G1-3"].get("sets", {}).get("NESTORB_signal"),
        "NESTORB_label": s09.get("NESTORB_label", "exploratory"), "C3N_label": s09.get("C3N_label", "estimation"),
        "g0_7r": gates["G0-7R"].get("rule"),
    }
    released = {rid: {"role": v["role"], "F": {c: {"F": x["F"], "SE": x["SE"], "W": x["W"]} for c, x in v["F"].items()}}
                for rid, v in {**runs_out, **rep_out}.items() if v["role"] in ("full", "eall", "aa", "aa_nestor", "dclm_cont")}
    ab = {
        "G1-0 GPU pipeline audit: passed / failed / errors": str(gates["G1-0"].get("values") or gates["G1-0"].get("reason")),
        "G1-4 audit negative control result; subset-run audits": f"{gates['G1-4'].get('rule') or gates['G1-4'].get('status')}",
        "G1-1 mean F_DCLM over the 4 MoE FULL runs; attempt; steps": f"Fbar {gates['G1-1'].get('values', {}).get('Fbar')}; "
                                                                     f"attempt {attempt}; steps {steps}",
        "G1-2 S1 books A/A pair (a = 2) F and SE (code stream, math), r; chosen q and steps":
            f"r = {gates['G1-2'].get('values', {}).get('r')}; rule {gates['G1-2'].get('rule')}; q = {q}; steps = {steps}",
        "G1-3 NESTOR books A/A subset F and SE (code stream, math); NESTOR-books signal": f"{decision['NESTORB_signal']}",
        "G0-9L NESTOR routers outside App. C.6 ranges (share); NESTOR-books and C3-NESTOR labels":
            f"{gates['G0-9L'].get('rule')}: {gates['G0-9L'].get('values', {}).get('share_outside')}; "
            f"{decision['NESTORB_label']}; {decision['C3N_label']}",
        "G1-5 |F_H100 - F_local| / |F(seed 101) - F(seed 102)| (NESTOR FULL German, DCLM)": f"{gates['G1-5'].get('values', {}).get('ratio')}",
        "G1-6 F_DCLM-CONT; F_rel of the MoE FULL runs": f"{gates['G1-6'].get('values', {}).get('F_cont')}",
        "G1-NS5 NS-5 relative differences (c = 0.25, 0.5, 2); D_Muon at step 1": f"{gates['G1-NS5'].get('status')}",
        "G0-7R recomputation: reason; beta0_1, beta0_2 and P-bar after recomputation": f"{gates['G0-7R'].get('rule')}",
    }
    return {"stage": "G1", "spec": _spec_check(plan), "prereg": "v1.3 sec. 6",
            "decision": decision, "gates": gates,
            "runs": {rid: {k: v for k, v in r.items() if k != "F"} for rid, r in {**runs_out, **rep_out}.items()},
            "F_levels_released": released, "problems": problems, "F_sym": fsym_out, "F_sym_out_of_domain_flags": fsym_flags,
            "inputs": {"plan": {"path": os.path.abspath(plan_path) if plan_path else None, "sha256": plan.get("sha256")},
                       "repeat_plan": {"path": os.path.abspath(repeat_path), "sha256": repeat_plan.get("sha256")} if repeat_plan else None,
                       "files": GC.inputs_manifest(inputs)},
            "firewall": "prereg sec. 4.10: F levels of FULL, E-ALL, A/A, DCLM-CONT runs and A/A F_sym tables; no I_AA, no P",
            "appendix_b": ab}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1 gate record (spec 1.3.0; torch-free)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--repeat-plan", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    plan = json.load(open(a.plan, encoding="utf-8"))
    rep = json.load(open(a.repeat_plan, encoding="utf-8")) if a.repeat_plan else None
    rec = gate(plan, rep, a.plan, a.repeat_plan or "")
    h = GC.finalize_record(rec, a.out)
    d = rec["decision"]
    print(f"G1 outcome {d['outcome']} | C1 {d['c1']} q={d['q']} steps={d['steps']} | G1-2 {d['g1_2_rule']} | "
          f"NESTOR-books signal {d['NESTORB_signal']} | G0-7R {d['g0_7r']} | record {a.out} sha256 {h}")
    for gid, g in rec["gates"].items():
        print(f"  {gid:7s} {g.get('status'):14s} {g.get('rule')}")
    for r in d["reasons"]:
        print("  reason:", r)
    for p in rec["problems"][:20]:
        print("  problem:", p)
    return EXIT[d["outcome"]]


if __name__ == "__main__":
    sys.exit(main())
