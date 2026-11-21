"""fill_core.py -- mechanical final fill of the core results (paper v1.5.13; P-16 as revised before the freeze and the frozen pre-registration
version 11, with E1 as P-17 (a) final: the core is C1 on both MoE bases, C2 (SGD runs to step 24, Muon runs to step 8, statistic at step 8) and the
theta_0 base evaluations; the third base and the assumption audit are status-only deferred entries of the core output and
are not read here; the fill stops unless mode = core and core.core_blocks = [C1:moe_b1, C1:moe_b2, C2]).

Reads the output of the core analysis (analyze_confirm.py, format analysis_confirm/1, run once on the frozen code after
the CORE_DONE marker) and writes, without any hand edit of a .tex file:
  fills/core_fills.tex       one \setfill{ID}{text} per core result slot; macros.tex loads it if it exists, and every
                             \resultTBD{ID} (and \costrow, \filledor) then prints the value instead of the pending marker
  fills/fill_manifest.json   every filled ID with its text and the JSON fields it came from, the IDs left pending and why
  tables/core_draws.tex      per-draw table of the core (Appendix G)
  tables/core_variants.tex   further registered C1 variants and the C2 step-8 equivalence (Appendix G)
  data/c1_results.json, data/c2_results.json   inputs of scripts/fig_crossover.py, which is then rerun
                             (figures/fig_crossover.pdf/png: C1 per base with the G0-7 probes, and the C2 panel)

Usage (from the paper root, CPU only):
  python scripts/fill_core.py                      # paths from CONFIG
  python scripts/fill_core.py --analysis PATH --root PAPER_ROOT [--no-figure]
Decisions are copied from the analysis output (decisions.*), never recomputed. Deferred blocks are never written.
Author slots (CONFIG["author_ids"]) are filled by scripts/fill_author.py with AI-assisted drafts for the author's review, chosen
from the registered decision labels (variants in scripts/author_slot_variants.json; every draft is listed under "author_slots" in
fills/fill_manifest.json). Plan/freeze values (CONFIG["plan_freeze_ids"]) are not filled here.
Only the JSON field paths in CONFIG need to change if the core output layout differs from analysis_confirm/1.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIG = {
    "analysis_json": r"results\analysis_core\analysis_confirm.json",
    "expected_format": "analysis_confirm/1",
    "edge_pre_json": "data/edge_pre_freeze.json",          # descriptive full-adaptation reference (Table tab:edgepre)
    "out": {"fills": "fills/core_fills.tex", "manifest": "fills/fill_manifest.json", "table": "tables/core_draws.tex", "variants": "tables/core_variants.tex", "e1": "tables/core_e1.tex",
            "c1_fig_data": "data/c1_results.json", "c2_fig_data": "data/c2_results.json"},
    "bases": {"moe_b1": "base~1", "moe_b2": "base~2"},
    "streams": {"code": "AlgebraicStack", "pycode": "Python code", "math": "mathematics", "dclm": "DCLM", "books": "books"},
    "text_streams": {"registered": "code", "second": "pycode"},   # as the Results text states them
    "c2_eq_rand_step": 8,                                  # pre-registration version 11 (as in 8): step-8 checkpoint
    "c2_stop_step": 8,                                     # C2.stopped_after_step = smallest C2 stop step (Muon 8; SGD runs stop at 24)
    "expected_mode": "core",                               # fill only from analyze_confirm.py run --mode core
    "core_blocks": ["C1:moe_b1", "C1:moe_b2", "C2"],       # core.core_blocks core output (P-16 revised)
    "deferred_status_only": ["blocks.B3", "blocks.audit"], # status-only deferred entries in the core output (full analysis only)
    "e1_record_fields": ["rule", "clauses_evaluated", "clause_ii", "deleted_clause", "stop_step"],   # E1 record, P-17 (a) final
    "e1_removed_fields": ["stop_step_clause", "final_loss_clause"],                                 # gone
    "fields": {                                            # dotted paths into analysis_confirm.json ([i] for lists)
        "format": "format",
        "mode": "mode",                                    # "core" | "full"
        "core.blocks": "core.core_blocks",                 # [C1:moe_b1, C1:moe_b2, C2]
        "core.runs": "core.core_runs",
        "E1.runs": "E1",                                   # {run id: E1 record} of diverged runs: rule, clauses_evaluated, clause_ii, deleted_clause, stop_step
        "C2.stopped_after_step": "C2.stopped_after_step",  # 8
        "C2.sgd24": "C2.sgd_step24_gain",                  # step24/step8 .pooled and .per_base.{moe_b1, moe_b2} {mean, min, max, n}; windows; role
        "warnings": "warnings",
        "C1.decision": "decisions.C1",                     # label, interpretation, out_of_domain_cells, three_statistic_table
        "C1.D": "C1.D",
        "C1.dropped": "C1.draws_dropped",
        "C1.test": "C1.test",                              # p, mean, D, vectors
        "C1.per_base": "C1.mean_J_per_base",               # {base: {mean_J, n}}
        "C1.estimators": "C1.estimators",                  # raw, pin, sym, sym_pin, rand -> overall {beta_hat, lo, hi}
        "C1.per_draw": "C1.per_draw",                      # [{base, d, I, P, J}]
        "C1.sym_draws": "C1.statistics.sym.draws",         # [{base, d, I_log, P}]
        "C1.statistic": "C1.statistic",
        "C1.sym_minus_rand": "C1.secondary.I_sym_minus_I_rand",          # {mean, lo, hi}
        "C1.wd_share": "C1.secondary.wd_share.mean_I_WD_over_mean_I",
        "C1.pinned_share": "C1.secondary.pinned_share.mean_I_pin_over_mean_I",
        "holm": "holm",                                    # family, m, adjusted
        "C2.decision": "decisions.C2",
        "C2.confirmatory": "C2.confirmatory",
        "C2.s_star": "C2.s_star",
        "C2.D": "C2.D",
        "C2.test": "C2.test",
        "C2.per_base": "C2.mean_J2_per_base",              # {base: mean J2}
        "C2.Delta_e": "C2.Delta_e",                        # {beta_hat, lo, hi}
        "C2.manip": "C2.manipulation_check",               # mean_D_SGD_over_DQ_mean, D_Muon_step1, failed
        "C2.per_draw": "C2.per_draw",                      # [{base, d, I2, J2, Delta_Q}]
        "C4": "blocks.C4",                                 # overall {mean_beta_C4, t95, bootstrap}, per_base
        "budget": "blocks.budget",                         # per_block {block: {actual_gpu_h, planned_gpu_h, jobs_with_hours}}, total, cuts
        "C1.E1": "C1.E1",                                  # {runs, diverged, share, block_inconclusive, no_train_log}
        "C2.E1": "C2.E1",                                  # {runs, diverged, block_inconclusive}
        "freeze.verification": "freeze_verification.status",   # "ok" when the core analysis verified FREEZE.json
        "EDGE.excluded": "blocks.EDGE.excluded",           # [{run, reason}] runs without a usable power log
        "EDGE.runs": "blocks.EDGE.runs",                   # [{arch, kind, base, third_base, wall_s, energy_Wh, ...}]
        "C1.code_stream": "C1.code_stream",                # registered code stream (text: AlgebraicStack = "code")
        "C1.second_stream": "C1.secondary.second_code_stream",   # stream, J_test {mean, p}, beta.overall
        "C1.margin_limited": "C1.margin_limited",          # {base: bool}, fixed at gate G0-7 (author slots)
        "matched_gain": "blocks.matched_gain.C1",          # mean_I_G {mean, lo, hi, n}, per_base
        "bpb": "blocks.bpb",                               # C1_check.max_abs_I_bpb_minus_I, C1_check.n, status
        "C1b.raw": "C1.secondary.C1b_pooled_ratio_raw",    # ratio, ci95, C1b_refuted
        "C1b.sym": "C1.secondary.C1b_pooled_ratio_sym",
        "C1d": "C1.secondary.C1d_DCLM_control",            # TOST_90pct_t_interval, TOST_equivalent_within_log1.1, failed_control (systematic sign)
        "C1e": "C1.secondary.C1e_within_base_R2",          # {base: {frequency|mean_gate|second_moment: {R2, ci95, note}}}
        "M12a": "C1.secondary.M12a_alignment_Lambda",      # [{Lambda, corpus, run}] (optional)
        "C2.eq": "C2.C2_eq",                               # t90 {mean, lo, hi}, margin, equivalent
        "C2.eq_rand": "blocks.C2_eq_rand",                 # step, mean_alignment {mean, lo, hi, n}, status; v10: step 8
    },
    "edge_rows": {                                         # COST row -> filter on blocks.EDGE.runs
        "COST.moe.subset": {"arch": "MoE", "kind": "subset", "base": ["moe_b1", "moe_b2"]},   # a list means membership
        "COST.moe.sgd": {"arch": "MoE", "kind": "C2 SGD"},
        "COST.moe.muon": {"arch": "MoE", "kind": "C2 Muon"},
    },
    "edge_full_reference": "adaptation_96_steps.moe_full_runs_no_ckpt",   # in edge_pre_json: wall_h, energy_wh [min, median, max]
    "core_budget_blocks": {"C1:moe_b1": "C1 on base~1", "C1:moe_b2": "C1 on base~2", "C2": "C2", "base_eval": "the evaluations at $\\theta_0$"},   # COMPUTE.used
    "hours_outside_json": "data/hours_outside_state.json",   # GPU-hours outside the queue state (D-1 addendum 1 item 4; incident I-1; GPU check of A-2)
    "core_hos_records": {"D-1": "deviation D-1 and amendment A-1", "I-1": "incident I-1", "A-2": "amendment A-2"},   # records of the core items among the hours outside the queue state; D-1 includes the GPU check of amendment A-1 (COMPUTE.used, rule 11)
    "edge_overlap_json": "data/edge_overlap_I2.json",       # core runs slowed by the CPU work of the A-2 precheck (incident I-2)
    "edge_rows_without_overlap": {"COST.moe.subset.noI2": "COST.moe.subset"},   # EDGE row recomputed without the runs of edge_overlap_json
    "core_eval_blocks": ["base_eval"],   # core budget blocks without adaptation runs; no job count or plan value is hard-coded (2026-11-15)
    # P-17 (a) final: a training log that ends before its stop step is flagged incomplete (a warning), not divergent; the flag reaches the
    # core output only as an entry of its top-level "warnings" list
    "incomplete_log_re": r"train log of (?P<run>\S+) ends at step (?P<end>\d+), not at the stop step (?P<stop>\d+): flagged incomplete",
    "core_run_prefixes": ("c1_moe_b1", "c1_moe_b2", "c2_moe_b1", "c2_moe_b2"),   # EDGE.excluded entries of the core
    "word_limits": {"ABS.C1": 40, "ABS.C2": 32, "ABS.cost": 27},
    "author_ids": ["DISC.interpretation", "DISC.code", "CONCL.code", "EDGE.C1", "EDGE.C2"],
    "plan_freeze_ids": ["REG.hash", "REG.date", "REG.freeze", "REG.deviations", "FREEZE.amend", "POWER.C1", "BUDGET.total",
                        "BUDGET.table", "BUDGET.forecast", "ENT.NESTORbooks"],   # written in the text since v1.5.4; COMPUTE.used is a core fill
}
BETA_REF = 0.25


# ----------------------------------------------------------------------------- helpers
def get(doc, path, default=None):
    obj = doc
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            idx = int(part[1:-1])
            if not isinstance(obj, list) or idx >= len(obj):
                return default
            obj = obj[idx]
        else:
            if not isinstance(obj, dict) or part not in obj:
                return default
            obj = obj[part]
    return obj


def F(key, doc):
    return get(doc, CONFIG["fields"][key])


def isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def num(x, nd=3):
    if x in ("inf", "-inf") or (isinstance(x, float) and math.isinf(x)):
        return "$-\\infty$" if str(x).startswith("-") else "$\\infty$"
    if not isnum(x):
        return "n/a"
    s = f"{float(x):.{nd}f}"
    return s.replace("-", "$-$")


def interval(lo, hi, nd=3):
    return f"[{num(lo, nd)}, {num(hi, nd)}]"


def est(e, nd=3):
    if not isinstance(e, dict):
        return "n/a"
    return f"{num(e.get('beta_hat'), nd)} {interval(e.get('lo'), e.get('hi'), nd)}"


def pfmt(p):
    if not isnum(p):
        return "n/a"
    p = float(p)
    if p < 0.001:
        e = math.floor(math.log10(p)); m = p / 10 ** e
        return f"{m:.1f}\\times10^{{{e}}}"
    return f"{p:.3g}"


def tex_text(s):
    """An analysis string (reason, label, cell name) as TeX text: base ids by their names, statistics in math, other
    TeX-active characters escaped (a raw '%' would comment out the rest of a fill, a raw '_' breaks the build)."""
    s, math_ = str(s), []

    def m_(tex):
        math_.append(tex)
        return f"\x00{len(math_) - 1}\x01"
    for i, b in enumerate(CONFIG["bases"]):
        s = s.replace(b, f"\x02{i}\x03")
    s = re.sub(r"beta-hat_(\w+)", lambda m: m_(r"\hat\beta_{\mathrm{" + m.group(1) + "}}"), s)
    s = s.replace("beta-hat", m_(r"\hat\beta")).replace("Delta-e", m_(r"\Delta e")).replace("D_SGD / Delta_Q(mean)", m_(r"D_{\mathrm{SGD}}/\Delta_Q"))
    s = s.replace("s*", m_("s^*"))
    s = re.sub(r"\s*(>=|<=|>|<)\s*", lambda m: " " + m_({">=": r"\ge", "<=": r"\le"}.get(m.group(1), m.group(1))) + " ", s).replace("( \x00", "(\x00")
    esc = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "$": r"\$"}
    s = re.sub(r"[\\%&#_{}~^$]", lambda m: esc[m.group(0)], s)
    s = re.sub("\x00(\\d+)\x01", lambda m: "$" + math_[int(m.group(1))] + "$", s)
    return re.sub("\x02(\\d+)\x03", lambda m: list(CONFIG["bases"].values())[int(m.group(1))], s)


def words(tex):
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", tex)
    t = re.sub(r"\\[a-zA-Z]+|[{}$~\\]", " ", t)
    return [w for w in t.split() if re.search(r"[A-Za-z0-9]", w)]


def med(xs):
    xs = sorted(xs); n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- fills
class Filler:
    def __init__(self, doc, edge_pre):
        self.doc, self.ep = doc, edge_pre
        self.fills, self.pending, self.warn = {}, {}, []

    def put(self, fid, text, fields):
        lim = CONFIG["word_limits"].get(fid)
        if lim is not None and len(words(text)) > lim:
            raise SystemExit(f"[fill_core] {fid}: {len(words(text))} words > limit {lim}: {text}")
        self.fills[fid] = {"text": text, "fields": fields}

    def skip(self, fid, why):
        self.pending[fid] = why

    # -- layout checks (pre-registration version 11): nothing is filled from these entries
    def layout_checks(self):
        d = self.doc
        for path in CONFIG["deferred_status_only"]:
            blk = get(d, path)
            if blk is None:
                self.warn.append(f"{path} missing in the core output")
            elif not isinstance(blk, dict) or set(blk) - {"status"}:
                self.warn.append(f"{path} carries more than a deferred status in the core output; nothing is filled from it")
        stop = F("C2.stopped_after_step", d)
        if stop != CONFIG["c2_stop_step"]:
            self.warn.append(f"C2.stopped_after_step = {stop!r}; the registered C2 contrast is at step {CONFIG['c2_stop_step']}")
        self.e1_records = []
        for rid, rec in sorted((F("E1.runs", d) or {}).items()):
            if not isinstance(rec, dict):
                continue
            miss = [k for k in CONFIG["e1_record_fields"] if k not in rec]
            old = [k for k in CONFIG["e1_removed_fields"] if k in rec]
            if miss or old:
                self.warn.append(f"E1 record of {rid}: missing {miss}" + (f"; carries removed fields {old} (layout)" if old else ""))
            self.e1_records.append({"run": rid, **{k: rec.get(k) for k in CONFIG["e1_record_fields"]}})
        # incomplete training logs: only in the top-level "warnings" list; paths inside the output are workspace or root paths and are not validated
        self.incomplete, self.other_warnings = [], []
        for w in (F("warnings", d) or []):
            m = re.search(CONFIG["incomplete_log_re"], str(w))
            if m:
                self.incomplete.append({"run": m.group("run"), "log_ends_at_step": int(m.group("end")), "stop_step": int(m.group("stop")), "warning": w})
            elif "incomplete" in str(w):
                self.incomplete.append({"run": None, "log_ends_at_step": None, "stop_step": None, "warning": w})
            else:
                self.other_warnings.append(w)
        div = {r["run"] for r in self.e1_records}
        for x in self.incomplete:
            if x["run"] in div:
                self.warn.append(f"{x['run']} is flagged incomplete and also listed as diverged in E1")
        if self.incomplete:
            self.warn.append("training logs flagged incomplete by the analysis (P-17 (a) final; kept, not divergent): "
                             + "; ".join(f"{x['run']} ends at step {x['log_ends_at_step']} of {x['stop_step']}" if x["run"] else str(x["warning"]) for x in self.incomplete)
                             + "; stated in E1.outcome and tables/core_e1.tex")
        self.e1_blocks = {b: F(f"{b}.E1", d) for b in ("C1", "C2")}
        for b, e in self.e1_blocks.items():
            if not isinstance(e, dict):
                self.warn.append(f"{b}.E1 missing in the core output")
            elif e.get("diverged") or e.get("block_inconclusive"):
                self.warn.append(f"{b}.E1: {e.get('diverged')} of {e.get('runs')} runs diverged; block_inconclusive {e.get('block_inconclusive')}")


    # -- C1
    def c1(self):
        d = self.doc; dec = F("C1.decision", d) or {}; lab = dec.get("label")
        if not lab:
            for fid in ("C1.D", "C1.T", "C1.meanJ", "C1.betas", "C1.controls", "C1.label", "C1.result", "ABS.C1", "INTRO.C1", "CONCL.C1"):
                self.skip(fid, "no decisions.C1 in the analysis output")
            return
        D = F("C1.D", d); pb = F("C1.per_base", d) or {}; t = F("C1.test", d) or {}; ests = F("C1.estimators", d) or {}
        adj = (F("holm", d) or {}).get("adjusted", {}).get("C1")
        drop = F("C1.dropped", d) or []
        nb = "; ".join(f"{pb[b]['n']} in {CONFIG['bases'][b]}" for b in CONFIG["bases"] if b in pb)
        self.put("C1.D", f"{D} ({nb})" + (f"; {len(drop)} dropped" if drop else ""), ["C1.D", "C1.mean_J_per_base[*].n", "C1.draws_dropped"])
        self.put("C1.T", f"{num(t.get('mean'))}; $p={pfmt(t.get('p'))}$; $p_{{\\mathrm{{Holm}}}}={pfmt(adj)}$", ["C1.test.mean", "C1.test.p", "holm.adjusted.C1"])
        self.put("C1.meanJ", " / ".join(num(pb[b]["mean_J"]) for b in CONFIG["bases"] if b in pb), ["C1.mean_J_per_base[*].mean_J"])
        self.put("C1.betas", "; ".join(est((ests.get(k) or {}).get("overall")) for k in ("raw", "pin", "sym", "rand")),
                 [f"C1.estimators.{k}.overall" for k in ("raw", "pin", "sym", "rand")])
        smr = F("C1.sym_minus_rand", d) or {}
        self.put("C1.controls", f"{num(smr.get('mean'))} {interval(smr.get('lo'), smr.get('hi'))}; {num(F('C1.wd_share', d))}; {num(F('C1.pinned_share', d))}",
                 ["C1.secondary.I_sym_minus_I_rand", "C1.secondary.wd_share.mean_I_WD_over_mean_I", "C1.secondary.pinned_share.mean_I_pin_over_mean_I"])
        ood = dec.get("out_of_domain_cells") or []
        self.put("C1.label", f"{tex_text(lab)}; out of domain: {', '.join(tex_text(c) for c in ood) if ood else 'none'}", ["decisions.C1.label", "decisions.C1.out_of_domain_cells"])
        raw = (ests.get("raw") or {}).get("overall") or {}
        b = f"$\\hat\\beta={num(raw.get('beta_hat'))}$ {interval(raw.get('lo'), raw.get('hi'))}"
        mj = f"mean $J={num(t.get('mean'))}$, Holm-adjusted $p={pfmt(adj)}$"
        interp = dec.get("interpretation") or []
        t3 = (dec.get("three_statistic_table") or {}).get("label", "")
        if lab.startswith("confirmed in the pooled"):
            res = f"In the pooled test of {D} draws the crossover was significant ({mj}; {b}), but mean $J$ was not positive in both bases, so the result is not replicated across bases."
            ab = "Forgetting tracked routing mass in the pooled test only."
        elif lab.startswith("confirmed"):
            head = (f"Across {D} draws in two bases, the forgetting of code and mathematics after books adaptation tracked their routing mass "
                    f"through the adapted experts ({mj}; {b})")
            if any("routing-gated sensitivity" in x for x in interp):
                res = head + "; norm-matched random perturbations of the same experts showed the same crossover, so we describe the effect as routing-gated sensitivity to fixed-norm perturbations."
            else:
                res = head + ", and the crossover of learned updates exceeded that of norm-matched random directions."
            if "margin-limited" in lab:
                res += " The result holds in a margin-limited base."
            ab = f"Forgetting tracked routing mass (elasticity {num(raw.get('beta_hat'), 2)} {interval(raw.get('lo'), raw.get('hi'), 2)})."
        elif lab.startswith("refuted"):
            if "learned-update law refuted" in t3:
                res = "Forgetting of learned updates did not track routing mass, although norm-matched random perturbations did; routing localization holds, and learned updates do not follow it."
            elif "does not localize" in t3:
                res = "In these models routing mass does not localize forgetting: the upper 95\\% bounds of $\\hat\\beta$, $\\hat\\beta_{\\mathrm{pin}}$, $\\hat\\beta_{\\mathrm{sym}}$ and $\\hat\\beta_{\\mathrm{rand}}$ lie below 0.25."
            else:
                res = f"Forgetting did not track routing mass: the upper 95\\% bounds of $\\hat\\beta$, $\\hat\\beta_{{\\mathrm{{pin}}}}$ and $\\hat\\beta_{{\\mathrm{{sym}}}}$ lie below 0.25 ({b})."
            if "reversed" in lab:
                res += " The crossover had the opposite sign."
            ab = "Forgetting did not track routing mass (elasticity below 0.25)."
        else:
            if t3.startswith("first-order"):
                res = f"The raw crossover was distorted by first-order terms; by the registered rule the outcome is not a refutation ({b})."
            elif isnum(t.get("mean")) and t["mean"] > 0 and isnum(adj) and adj > 0.05:
                res = f"The crossover was positive but not significant ({mj}; {b}), so C1 is inconclusive; the test was flagged underpowered before the freeze."
            else:
                res = f"C1 is inconclusive ({'; '.join(tex_text(r) for r in (dec.get('reasons') or ['reason in the analysis output']))}; {b})."
            ab = f"C1 was inconclusive (flagged underpowered; elasticity {num(raw.get('beta_hat'), 2)} {interval(raw.get('lo'), raw.get('hi'), 2)})."
        if any("decay-driven" in x for x in interp):
            res += " The weight-decay-only control reached at least half of the learned crossover, so the effect is substantially decay-driven."
        if ood:
            res += f" Out-of-domain cells: {', '.join(tex_text(c) for c in ood)}."
        flds = ["decisions.C1", "C1.test", "holm.adjusted.C1", "C1.estimators.raw.overall"]
        self.put("C1.result", res, flds)
        self.put("ABS.C1", ab, flds)
        self.put("INTRO.C1", f"C1 was {lab} ({mj}).", flds)
        self.put("CONCL.C1", f"C1 was {lab}: {res[0].lower() + res[1:]}" if lab.startswith(("confirmed", "refuted")) else res, flds)

    # -- C2
    def c2(self):
        d = self.doc; dec = F("C2.decision", d) or {}; lab = dec.get("label")
        if not lab or lab == "not run":
            for fid in ("C2.status", "C2.manip", "C2.T", "C2.de", "C2.label", "C2.result", "ABS.C2", "INTRO.C2", "CONCL.C2", "C2.sgd24"):
                self.skip(fid, "C2 not in the analysis output")
            return
        conf = bool(F("C2.confirmatory", d)); s = F("C2.s_star", d); D = F("C2.D", d); t = F("C2.test", d) or {}
        adj = (F("holm", d) or {}).get("adjusted", {}).get("C2"); de = F("C2.Delta_e", d) or {}; mc = F("C2.manip", d) or {}
        pb = F("C2.per_base", d) or {}
        self.put("C2.status", f"{'confirmatory' if conf else 'exploratory'}; $s^*={s}$; $D={D}$", ["C2.confirmatory", "C2.s_star", "C2.D"])
        dm = [x for x in (mc.get("D_Muon_step1") or []) if isnum(x)]   # step-1 distances can be missing (null)
        self.put("C2.manip", f"{num(mc.get('mean_D_SGD_over_DQ_mean'), 2)}" + ("" if not mc.get("failed") else " (failed)") +
                 (f"; median {num(med(dm), 4)}" if dm else ""), ["C2.manipulation_check.mean_D_SGD_over_DQ_mean", "C2.manipulation_check.D_Muon_step1"])
        self.put("C2.T", f"{num(t.get('mean'))}; $p={pfmt(t.get('p'))}$; $p_{{\\mathrm{{Holm}}}}={pfmt(adj) if conf else 'n/a'}$", ["C2.test.mean", "C2.test.p", "holm.adjusted.C2"])
        self.put("C2.de", " / ".join(num(pb[b]) for b in CONFIG["bases"] if b in pb) + f"; {est(de)}", ["C2.mean_J2_per_base", "C2.Delta_e"])
        self.put("C2.label", lab, ["decisions.C2.label"])
        g = F("C2.sgd24", d) or {}                         # SGD German-head gain CE(step 0) - CE(step s): evidence that the SGD arm learns, no decision role
        s24, s8 = g.get("step24") or {}, g.get("step8") or {}
        pool, pb = s24.get("pooled") or {}, s24.get("per_base") or {}
        if isinstance(g, dict) and isnum(pool.get("mean")):
            if g.get("decision_role"):
                self.warn.append("C2.sgd_step24_gain carries a decision role; the pre-registration (version 11, sec. 3.2) makes it descriptive")
            win = g.get("windows") or []
            head = f"{win[0]}-window" if len(win) == 1 and isnum(win[0]) else "interim"
            per = ", ".join(f"{CONFIG['bases'][b]} {num(pb[b]['mean'])}" for b in CONFIG["bases"] if isinstance(pb.get(b), dict) and isnum(pb[b].get("mean")))
            p8 = (s8.get("pooled") or {}).get("mean")
            n, npl, exc = pool.get("n"), g.get("n_planned"), g.get("excluded") or []
            txt = (f"On the {head} German head the gain from step 0 to step 24 averaged {num(pool['mean'])} nats over {n}"
                   + (f" of the {npl}" if isnum(npl) and npl != n else "") + f" SGD runs (range {num(pool.get('min'))} to {num(pool.get('max'))}"
                   + (f"; {per}" if per else "") + ")" + (f", against {num(p8)} nats at step 8" if isnum(p8) else "") + "."
                   + (f" {len(exc)} SGD runs were not usable." if exc else ""))
            self.put("C2.sgd24", txt, [CONFIG["fields"]["C2.sgd24"] + ".step24", CONFIG["fields"]["C2.sgd24"] + ".step8"])
        else:
            self.skip("C2.sgd24", f"field {CONFIG['fields']['C2.sgd24']}.step24.pooled.mean not in the core output")
        pdraws = F("C2.per_draw", d) or []
        kpos = sum(1 for r in pdraws if isnum(r.get("J2")) and r["J2"] > 0)
        mj = f"mean $J2={num(t.get('mean'))}$, Holm-adjusted $p={pfmt(adj)}$"
        if lab == "confirmed":
            res = (f"Under small-step SGD, DCLM forgetting depended more on the target's routing mass than under per-matrix Muon ({mj}; "
                   f"$\\Delta e={num(de.get('beta_hat'))}$ {interval(de.get('lo'), de.get('hi'))}) at step {s}, the earliest step that resolved forgetting.")
            ab = "SGD made forgetting depend more on target routing than Muon."
        elif lab == "refuted":
            res = (f"At step {s} the dependence of forgetting on target routing mass did not differ between small-step SGD and per-matrix Muon, "
                   f"contrary to the derived prediction ($\\Delta e={num(de.get('beta_hat'))}$ {interval(de.get('lo'), de.get('hi'))}).")
            ab = f"Optimizers did not differ at step {s} ($\\Delta e$ below 0.25)."
        elif lab == "uninformative":
            res = (f"The SGD arm did not realize the predicted displacement contrast (mean $D_{{\\mathrm{{SGD}}}}/\\Delta_Q={num(mc.get('mean_D_SGD_over_DQ_mean'), 2)}$ "
                   f"at step 8), so C2 is uninformative by the registered rule.")
            ab = "C2 was uninformative: SGD missed its displacement contrast."
        elif lab == "inconclusive":
            res = f"$J2$ was positive in {kpos} of {D} draws but not significant ({mj}), so C2 is inconclusive."
            ab = f"C2 was inconclusive at step {s}."
        else:
            res = (f"C2 is {tex_text(lab)} ({'; '.join(tex_text(r) for r in (dec.get('reasons') or []))}); estimates only: mean $J2={num(t.get('mean'))}$, "
                   f"$\\Delta e={num(de.get('beta_hat'))}$ {interval(de.get('lo'), de.get('hi'))}.")
            ab = "C2 was exploratory; estimates only."
        flds = ["decisions.C2", "C2.test", "holm.adjusted.C2", "C2.Delta_e", "C2.per_draw[*].J2"]
        self.put("C2.result", res, flds)
        self.put("ABS.C2", ab, flds)
        self.put("INTRO.C2", f"C2 was {lab} at step {s} ({mj}).", flds)
        self.put("CONCL.C2", res, flds)

    # -- C4
    def c4(self):
        c4 = F("C4", self.doc) or {}; ov = c4.get("overall") or {}
        if not isnum(ov.get("mean_beta_C4")):
            self.skip("C4.result", "no C4 block"); return
        t95 = ov.get("t95") or {}; bs = (ov.get("bootstrap") or {}).get("ci95") or [None, None]
        pb = c4.get("per_base") or {}
        per = "; ".join(f"{CONFIG['bases'][b]} {num((pb[b] or {}).get('mean', (pb[b] or {}).get('mean_beta_C4')))}" for b in CONFIG["bases"] if b in pb)
        self.put("C4.result", f"Over the {ov.get('n')} C1 draws the mean per-window slope $\\beta_{{\\mathrm{{C4}}}}$ was {num(ov['mean_beta_C4'])} "
                 f"(95\\% $t$-interval {interval(t95.get('lo'), t95.get('hi'))}; two-way bootstrap {interval(bs[0], bs[1])})" + (f", {per}." if per else "."),
                 ["blocks.C4.overall", "blocks.C4.per_base"])

    # -- EDGE (core runs) and cost sentences
    def edge(self):
        runs = F("EDGE.runs", self.doc) or []
        rows = {}
        for fid, flt in CONFIG["edge_rows"].items():
            rr = [r for r in runs if all((r.get(k) in v) if isinstance(v, list) else (r.get(k) == v) for k, v in flt.items())]
            if not rr:
                self.skip(fid, "no covered run of this type"); continue
            w = [r["wall_s"] / 3600 for r in rr if isnum(r.get("wall_s"))]; e = [r["energy_Wh"] for r in rr if isnum(r.get("energy_Wh"))]
            m = [max(r.get("vram_peak_reserved_gib") or 0, r.get("vram_peak_alloc_gib") or 0) for r in rr]
            tp = [r["tokens_per_s_steady"] for r in rr if isnum(r.get("tokens_per_s_steady"))]
            def cell(x, nd):
                return "n/a" if not x else (f"{med(x):.{nd}f} [{min(x):.{nd}f}, {max(x):.{nd}f}]" if len(x) >= 3 else ", ".join(f"{v:.{nd}f}" for v in x))
            def mcell(x, fmt):                                # median, or the tables' dash when the analysis gives no value
                return format(med(x), fmt) if x else "--"
            txt = f"{cell(w, 3)} & {cell(e, 1)} & {mcell(m, '.1f')} & {mcell(tp, ',.0f')}"
            rows[fid] = (w, e)
            self.put(fid, txt, [f"blocks.EDGE.runs[{json.dumps(flt)}].{k}" for k in ("wall_s", "energy_Wh", "vram_peak_reserved_gib", "tokens_per_s_steady")])
            for did, base in CONFIG["edge_rows_without_overlap"].items():
                if base != fid:
                    continue
                ovl = getattr(self, "overlap", set())
                rw = [r for r in rr if r.get("run") not in ovl]
                if not ovl or len(rw) == len(rr):
                    self.skip(did, "no covered run of this row overlapped incident I-2"); continue
                if not rw:
                    self.skip(did, "every covered run of this row overlapped incident I-2"); continue
                w2 = [r["wall_s"] / 3600 for r in rw if isnum(r.get("wall_s"))]; e2 = [r["energy_Wh"] for r in rw if isnum(r.get("energy_Wh"))]
                m2 = [max(r.get("vram_peak_reserved_gib") or 0, r.get("vram_peak_alloc_gib") or 0) for r in rw]
                tp2 = [r["tokens_per_s_steady"] for r in rw if isnum(r.get("tokens_per_s_steady"))]
                rows[did] = (w2, e2)
                self.put(did, f"{cell(w2, 3)} & {cell(e2, 1)} & {mcell(m2, '.1f')} & {mcell(tp2, ',.0f')}",
                         [f"blocks.EDGE.runs[{json.dumps(flt)}] without {CONFIG['edge_overlap_json']}:runs.{k}" for k in ("wall_s", "energy_Wh", "vram_peak_reserved_gib", "tokens_per_s_steady")])
        ref = get(self.ep, CONFIG["edge_full_reference"]) if self.ep else None
        sub = rows.get("COST.moe.subset")
        if sub and ref and sub[0] and sub[1]:
            ws, es = med(sub[0]), med(sub[1]); wf, ef = ref["wall_h"][1], ref["energy_wh"][1]
            noi2 = ""
            if rows.get("COST.moe.subset.noI2") and rows["COST.moe.subset.noI2"][0] and rows["COST.moe.subset.noI2"][1]:
                w2, e2 = rows["COST.moe.subset.noI2"]
                noi2 = f" ({med(w2):.3f}\\,h and {med(e2):.1f}\\,Wh over {len(w2)} runs without the {len(sub[0]) - len(w2)} that incident~I-2 slowed)"
            flds = ["blocks.EDGE.runs[subset, bases 1-2]", f"{CONFIG['edge_pre_json']}:{CONFIG['edge_full_reference']}"] + ([f"{CONFIG['edge_overlap_json']}:runs"] if noi2 else [])
            self.put("EDGE.cost", (f"The median expert-subset run of MoE bases 1--2 took {ws:.3f}\\,h and {es:.1f}\\,Wh over {len(sub[0])} runs{noi2}, "
                                   f"{ws / wf:.2f} times the wall time and {es / ef:.2f} times the energy of the median MoE full adaptation measured before the freeze "
                                   f"(Table~\\ref{{tab:edgepre}}, descriptive)."), flds)
            self.put("ABS.cost", f"Subset updates took {ws:.2f}\\,h, {es:.0f}\\,Wh (full: {wf:.2f}\\,h, {ef:.0f}\\,Wh).", flds)
            self.put("CONCL.edge", (f"On the local GPU, an expert-subset adaptation took {ws:.3f}\\,h and {es:.1f}\\,Wh at the median, against {wf:.3f}\\,h and "
                                    f"{ef:.1f}\\,Wh for a full adaptation measured before the freeze."), flds)
        else:
            for fid in ("EDGE.cost", "ABS.cost", "CONCL.edge"):
                self.skip(fid, "no subset runs in EDGE or no pre-freeze reference")

    # -- COMPUTE.used at the core fill (2026-11-15): measured GPU-hours of the core jobs (blocks.budget) and their board energy (blocks.EDGE);
    #    job counts and planned hours come from the output only
    def compute_used(self):
        d = self.doc; pb = (F("budget", d) or {}).get("per_block") or {}; lbl = CONFIG["core_budget_blocks"]
        rows = [(b, pb[b]) for b in lbl if isinstance(pb.get(b), dict) and isnum(pb[b].get("actual_gpu_h"))]
        if not rows:
            self.skip("COMPUTE.used", "no measured GPU-hours of the core blocks in blocks.budget.per_block"); return
        missing = [b for b in lbl if b not in dict(rows)]
        if missing:
            self.warn.append(f"blocks.budget.per_block lacks the core blocks {missing}")
        if F("freeze.verification", d) != "ok":
            self.warn.append(f"freeze verification of the core output is {F('freeze.verification', d)!r}: its planned hours are not verified against the frozen plan")
        act = sum(r["actual_gpu_h"] for _, r in rows); plan = sum(r["planned_gpu_h"] for _, r in rows if isnum(r.get("planned_gpu_h")))
        jobs = sum(int(r.get("jobs_with_hours") or 0) for _, r in rows)
        nrun = F("core.runs", d); adapt = sum(int(r.get("jobs_with_hours") or 0) for b, r in rows if b not in CONFIG["core_eval_blocks"])
        if isnum(nrun) and adapt != nrun:
            self.warn.append(f"blocks.budget counts {adapt} C1 and C2 jobs with hours; core.core_runs is {nrun}")
        lab = [f"{r['actual_gpu_h']:.2f} for {lbl[b]}" for b, r in rows]
        parts = lab[0] if len(lab) == 1 else ", ".join(lab[:-1]) + " and " + lab[-1]
        runs = [r for r in (F("EDGE.runs", d) or []) if r.get("base") in CONFIG["bases"] and not r.get("third_base")]
        e = [r["energy_Wh"] for r in runs if isnum(r.get("energy_Wh"))]
        exc = [x for x in (F("EDGE.excluded", d) or []) if str(x.get("run", "")).startswith(CONFIG["core_run_prefixes"])]
        if isnum(nrun) and len(e) + len(exc) != nrun:
            self.warn.append(f"EDGE covers {len(e)} core runs with energy and {len(exc)} excluded; core.core_runs is {nrun}")
        txt = f"The {jobs} jobs of the core took {act:.2f} GPU-hours in the queue state against the {plan:.2f} planned ({parts})"
        hos = getattr(self, "hos", None); hh = hos.get("hours") if isinstance(hos, dict) else None
        if isinstance(hh, dict) and hh:   # hours outside the queue state of the core (records D-1, I-1, A-2); Appendix A states each item, the last row of Table tab:budget the total (rule 11)
            tot_all = sum(v for v in hh.values() if isnum(v)); rec = hos.get("records") or {}
            if isnum(hos.get("total")) and abs(hos["total"] - tot_all) > 5e-5:
                self.warn.append(f"{CONFIG['hours_outside_json']}: total {hos['total']} differs from the sum of its hours {tot_all:.4f}")
            cr = CONFIG["core_hos_records"]; core = [k for k in hh if rec.get(k) in cr]
            pend = [k for k in core if hh[k] is None]
            if pend:
                self.warn.append(f"{CONFIG['hours_outside_json']}: {pend} not yet recorded; COMPUTE.used and the totals omit them")
            tot = sum(hh[k] for k in core if isnum(hh[k])); names = [cr[r] for r in cr if any(rec.get(k) == r for k in core)]
            if names:
                txt += (f", and {tot:.4f} GPU-hours outside the queue state for {names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]} "
                        f"(included in the last row of Table~\\ref{{tab:budget}}) bring the core and its recovery to {act + tot:.2f} GPU-hours")
            else:
                self.warn.append(f"{CONFIG['hours_outside_json']}: no item with a core record {list(cr)}: COMPUTE.used omits the GPU-hours outside the queue state")
        else:
            self.warn.append(f"{CONFIG['hours_outside_json']} missing: COMPUTE.used omits the GPU-hours outside the queue state")
        txt += "."
        if e:
            txt += f" The {len(e)} core adaptation runs with a usable power log drew {sum(e):,.1f}\\,Wh of board energy"
            if exc:
                txt += f" ({len(exc)} further run{'s' if len(exc) != 1 else ''} excluded from the energy total)"
            txt += "."
        txt += " The remaining blocks ran after the core."
        self.put("COMPUTE.used", txt, ["blocks.budget.per_block[core blocks].actual_gpu_h", "blocks.budget.per_block[core blocks].planned_gpu_h",
                                       "blocks.budget.per_block[core blocks].jobs_with_hours", "blocks.EDGE.runs[bases 1-2].energy_Wh", "blocks.EDGE.excluded",
                                       f"{CONFIG['hours_outside_json']}:hours"])

    # -- hours outside the queue state (v1.5.9): Table 13, Appendix C and Methods print \filledor defaults taken from
    #    data/hours_outside_state.json; at the fill the same file gives the total, the item list and the GPU check of A-2
    def hours_outside(self):
        hos = getattr(self, "hos", None); hh = hos.get("hours") if isinstance(hos, dict) else None
        if not (isinstance(hh, dict) and hh):
            for fid in ("HOS.total", "HOS.A2"):
                self.skip(fid, f"{CONFIG['hours_outside_json']} missing")
            return
        lab, rec = hos.get("labels") or {}, hos.get("records") or {}
        items = [(k, v) for k, v in hh.items() if isnum(v)]
        tot = sum(v for _, v in items)
        names = [(lab.get(k, k)[4:] if lab.get(k, k).startswith("the ") else lab.get(k, k)) + f" ({v:.4f})" for k, v in items]
        src = [f"{CONFIG['hours_outside_json']}:hours", f"{CONFIG['hours_outside_json']}:labels"]
        self.put("HOS.total", f"{tot:.4f}", src)
        # HOS.items: scripts/fill_full.py --integrate removes its slot (rule 11: Appendix A states the items, the last row of
        # Table tab:budget only their total), so the item list is not written
        a2 = [v for k, v in items if rec.get(k) == "A-2"]
        if a2:
            self.put("HOS.A2", f", and the GPU check of amendment A-2 used {a2[0]:.4f} GPU-hours", src)
        else:
            self.skip("HOS.A2", f"GPU check of amendment A-2 not yet recorded in {CONFIG['hours_outside_json']}")

    # -- E1 outcome (2026-11-15): one Results sentence (E1.outcome, \filledor, empty until the fill) and the per-run record of
    #    Appendix G (tables/core_e1.tex); counts from C1.E1 and C2.E1, flagged logs from the parsed top-level warnings
    def e1_outcome(self):
        self.e1_table = None
        c1, c2 = self.e1_blocks.get("C1"), self.e1_blocks.get("C2")
        if not all(isinstance(e, dict) and isnum(e.get("runs")) for e in (c1, c2)):
            self.skip("E1.outcome", "C1.E1 or C2.E1 missing in the core output"); return
        nn = lambda x: "none" if not x else str(int(x))
        d1, d2 = int(c1.get("diverged") or 0), int(c2.get("diverged") or 0); ref = "Appendix~\\ref{app:coredraws}"
        txt = f"Of the runs screened under the divergence rule E1, {nn(d1)} of the {int(c1['runs'])} in C1 and {nn(d2)} of the {int(c2['runs'])} in C2 diverged"
        inc = [b for b, e in (("C1", c1), ("C2", c2)) if e.get("block_inconclusive")]
        if inc:
            txt += (", so by the registered rule for more than 25\\% divergent runs the " + " and the ".join(inc)
                    + (" tests are" if len(inc) > 1 else " test is") + " inconclusive")
        ntl = [(b, r) for b, e in (("C1", c1), ("C2", c2)) for r in (e.get("no_train_log") or [])]
        flagged = [x for x in self.incomplete if x.get("run")]; unparsed = [x for x in self.incomplete if not x.get("run")]
        tail = f" ({ref})." if (d1 or d2 or flagged or unparsed or ntl) else "."
        if not (flagged or unparsed or ntl):
            txt += ", and no training log was flagged incomplete" + tail
        else:
            parts = []
            if len(flagged) == 1:
                x = flagged[0]
                parts.append(f"the training log of {texid(x['run'])} ends at step {x['log_ends_at_step']}, before its stop step {x['stop_step']}, "
                             "so the analysis flagged the run as incomplete, and the run stays in its test")
            elif flagged:
                items = [f"{texid(x['run'])} (last logged step {x['log_ends_at_step']} of {x['stop_step']})" for x in flagged]
                parts.append("the training logs of " + ", ".join(items[:-1]) + " and " + items[-1]
                             + " end before their stop steps, so the analysis flagged these runs as incomplete, and each stays in its test")
            if unparsed:
                parts.append(f"{len(unparsed)} further warning{'s' if len(unparsed) != 1 else ''} of the analysis flag{'' if len(unparsed) != 1 else 's'} a training log as incomplete")
            if ntl:
                parts.append(f"{len(ntl)} run{'s' if len(ntl) != 1 else ''} had no training log to screen")
            txt += "; " + "; ".join(parts) + tail
        self.put("E1.outcome", txt, ["C1.E1.runs", "C1.E1.diverged", "C2.E1.runs", "C2.E1.diverged", "warnings (flagged incomplete)", "E1 (diverged runs)"])
        rows, ii, e1runs = [], [], F("E1.runs", self.doc) or {}
        for r in self.e1_records:
            blk, base = run_block(r["run"])
            det = None
            for why in ((e1runs.get(r["run"]) or {}).get("reasons") or []):    # v1.6.1 (final pass S1): measured values, not the rule text
                mm = re.search(r"EMA\(0\.1\) loss ([\d.]+) at step (\d+) > step-1 loss ([\d.]+)", str(why))
                if mm:
                    det = (f"clause (ii): EMA{' of the training loss' if not ii else ''} {float(mm.group(1)):.3f} at step {mm.group(2)}, "
                           f"step-1 loss {float(mm.group(3)):.3f}")
                    ii.append(r); break
            if det is None:
                det = "; ".join(f"{k.replace('_', ' ')} {texval(r.get(k))}" for k in ("rule", "clauses_evaluated", "stop_step") if r.get(k) is not None)
            rows.append(f"{run_name(r['run'])} & {blk}, base~{base} & diverged & {det or '--'}\\\\")
        for x in flagged:
            blk, base = run_block(x["run"])
            rows.append(f"{run_name(x['run'])} & {blk}, base~{base} & incomplete log; stays in its test & last logged step {x['log_ends_at_step']} of {x['stop_step']}\\\\")
        for x in unparsed:
            rows.append(f"-- & -- & incomplete log (warning) & {texval(x['warning'])}\\\\")
        for b, run in ntl:
            blk, base = run_block(str(run))
            rows.append(f"{run_name(str(run))} & {b}, base~{base} & no training log & --\\\\")
        hdr = "% generated by scripts/fill_core.py from the core analysis output -- do not edit\n"
        intro = f"\\paragraph{{Divergence rule E1, per run.}}\nThe core analysis screened the {int(c1['runs'])} C1 runs and the {int(c2['runs'])} C2 runs under E1. "
        if not rows:
            self.e1_table = hdr + intro + "No run diverged, every run had a training log, and no training log was flagged incomplete.\n"
            return
        self.e1_table = (hdr + intro + "Table~\\ref{tab:coree1} lists every core run that E1 marks as divergent, that has no training log, or whose training log "
                         "the analysis flagged incomplete; a run with an incomplete log is not divergent and stays in its test.\n"
                         "\\begin{table}[ht]\n\\centering\n\\caption{" + self.e1_caption(ii) + "}\n"
                         "\\label{tab:coree1}\n\\small\n\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xl>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}X@{}}\n"
                         "\\toprule\nRun & Block, base & Status & Training log\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")

    def e1_caption(self, ii):
        """caption of the E1 table; v1.6.1 (final pass S1) adds the clause-(ii) rule when every listed divergence is a clause-(ii) case"""
        cap = "Core runs marked under the divergence rule E1, as written by the registered analysis."
        if not ii or len(ii) != len(self.e1_records):
            return cap
        cl = next((c for c in (ii[0].get("clauses_evaluated") or []) if str(c).startswith("(ii)")), "")
        mw, mt = re.search(r"steps (\d+)-\d+", cl), re.search(r"more than ([\d.]+) nats", cl)
        stops = {r.get("stop_step") for r in ii}
        if not (mw and mt and len(stops) == 1):
            return cap
        n, stop = len(ii), stops.pop()
        words_n = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        tail = f"all {words_n.get(n, str(n))} runs stop at step {stop}." if n > 1 else f"the run stops at step {stop}."
        return (cap + f" Clause (ii) of rule E1 marks a run whose EMA(0.1) of the training loss exceeds its step-1 value by more than {mt.group(1)} nats "
                f"after the {int(mw.group(1)) - 1}-step warm-up; " + tail)

    # -- secondary analyses of the core in the main text (v1.5.1)
    def secondary(self):
        d = self.doc
        mg = F("matched_gain", d) or {}; mi = mg.get("mean_I_G") or {}
        if isnum(mi.get("mean")):
            pb = mg.get("per_base") or {}
            per = "; ".join(f"{CONFIG['bases'][b]} {num((pb[b] or {}).get('mean'))}" for b in CONFIG["bases"] if b in pb)
            self.put("C1.matchedgain", f"At matched gain the mean crossover $I_G$ was {num(mi['mean'])} (95\\% $t$-interval {interval(mi.get('lo'), mi.get('hi'))} over {mi.get('n')} draws" + (f"; {per})." if per else ")."),
                     ["blocks.matched_gain.C1.mean_I_G", "blocks.matched_gain.C1.per_base"])
        else:
            self.skip("C1.matchedgain", "no blocks.matched_gain.C1.mean_I_G")
        sc = F("C1.second_stream", d) or {}; reg = F("C1.code_stream", d)
        if reg != CONFIG["text_streams"]["registered"] or sc.get("stream") != CONFIG["text_streams"]["second"]:
            self.warn.append(f"code streams: analysis registered {reg!r}, second {sc.get('stream')!r}; the Results text says {CONFIG['text_streams']}")
        jt = sc.get("J_test") or {}; be = (sc.get("beta") or {}).get("overall") or {}
        if isnum(jt.get("mean")):
            self.put("C1.code2", f"Mean $J$ was {num(jt['mean'])} (one-sided sign-flip $p={pfmt(jt.get('p'))}$, unadjusted) and $\\hat\\beta={num(be.get('beta_hat'))}$ {interval(be.get('lo'), be.get('hi'))}.",
                     ["C1.secondary.second_code_stream.J_test", "C1.secondary.second_code_stream.beta.overall"])
        else:
            self.skip("C1.code2", "no second-code-stream test in the analysis output")
        bp = F("bpb", d) or {}; ck = bp.get("C1_check") or {}; x = ck.get("max_abs_I_bpb_minus_I")
        if isnum(x):
            xs = "0" if x == 0 else pfmt(x)
            self.put("BPB.check", f"In bits per byte the C1 crossovers equal those in nats, because each statistic compares one corpus on identical windows; over the {ck.get('n')} draws the largest difference was ${xs}$.",
                     ["blocks.bpb.C1_check.max_abs_I_bpb_minus_I", "blocks.bpb.C1_check.n"])
        else:
            self.skip("BPB.check", f"bits per byte: {bp.get('status', 'no C1 check')}")

    # -- further registered C1 variants and the C2 step-8 equivalence (Appendix G, TAB.corevariants)
    def variants(self):
        d = self.doc; rows = []
        for k, lab in (("C1b.raw", "raw"), ("C1b.sym", "symmetrized")):
            v = F(k, d) or {}
            if isnum(v.get("ratio")):
                ci = v.get("ci95") or [None, None]
                rows.append(("C1-b", f"pooled ratio $\\sum_d I_d/\\sum_d P_d$, {lab}, 95\\% bootstrap interval",
                             f"{num(v['ratio'])} {interval(ci[0], ci[1])}; refuted: {'yes' if v.get('C1b_refuted') else 'no'}"))
        cd = F("C1d", d) or {}; ti = cd.get("TOST_90pct_t_interval") or {}
        if isnum(ti.get("mean")):
            rows.append(("C1-d", "DCLM control, mean of $\\log F_{\\mathrm{sym,DCLM}}(B_d)/F_{\\mathrm{sym,DCLM}}(A_d)$ minus the log ratio of the DCLM routing overlaps of $B_d$ and $A_d$, 90\\% $t$-interval",
                         f"{num(ti['mean'])} {interval(ti.get('lo'), ti.get('hi'))}; equivalent within $\\log1.1$: {'yes' if cd.get('TOST_equivalent_within_log1.1') else 'no'}"))
        ce = F("C1e", d) or {}
        for st, lab in (("frequency", "frequency"), ("mean_gate", "mean gate"), ("second_moment", "second moment")):
            parts = []
            for b, name in CONFIG["bases"].items():
                e = (ce.get(b) or {}).get(st) or {}
                if isnum(e.get("R2")):
                    ci = e.get("ci95") or [None, None]; parts.append(f"{name} {num(e['R2'])} {interval(ci[0], ci[1])}")
                elif e:
                    parts.append(f"{name} undefined")
            if parts:
                rows.append(("C1-e", f"within-base $R^2$ of $I_d$ on the {lab} contrast, 95\\% bootstrap interval", "; ".join(parts)))
        ma = F("M12a", d)
        if isinstance(ma, list) and ma:
            by = {}
            for r in ma:
                if isnum(r.get("Lambda")): by.setdefault(r.get("corpus"), []).append(float(r["Lambda"]))
            if by:
                rows.append(("M12-a", "alignment $\\Lambda$, median over runs", ", ".join(f"{CONFIG['streams'].get(c, c)} {num(med(v))}" for c, v in by.items()) + f" ({len(ma)} run-corpus pairs)"))
        eq = F("C2.eq", d) or {}; t9 = eq.get("t90") or {}
        if isnum(t9.get("mean")):
            rows.append(("C2", f"Muon elasticity $e_d$ at step 8, 90\\% $t$-interval; equivalence margin {num(eq.get('margin'), 3)}",
                         f"{num(t9['mean'])} {interval(t9.get('lo'), t9.get('hi'))}; equivalent: {'yes' if eq.get('equivalent') else 'no'}"))
        er = F("C2.eq_rand", d) or {}; al = er.get("mean_alignment") or {}
        if isnum(al.get("mean")):
            stp = er.get("step")
            if stp != CONFIG["c2_eq_rand_step"]:
                self.warn.append(f"C2_eq_rand evaluated at step {stp}; pre-registration version 11 says step {CONFIG['c2_eq_rand_step']}")
            rows.append(("C2", f"$b_{{\\mathrm{{Muon}}}}-b^{{\\mathrm{{rand}}}}_{{\\mathrm{{Muon}}}}$ at step {stp}, 95\\% $t$-interval", f"{num(al['mean'])} {interval(al.get('lo'), al.get('hi'))} over {al.get('n')} draws"))
        elif er:
            rows.append(("C2", "$b_{\\mathrm{Muon}}-b^{\\mathrm{rand}}_{\\mathrm{Muon}}$", "not computable from the core files"))
        if not rows:
            self.skip("TAB.corevariants", "no variant in the analysis output"); return None
        lines = ["% generated by scripts/fill_core.py from the core analysis output -- do not edit",
                 "\\begin{table}[ht]", "\\centering",
                 "\\caption{Further registered C1 variants and the step-8 equivalence analysis of C2, as written by the core analysis; secondary, unadjusted and outside every decision. C1-b counts as refuted only if the 95\\% bootstrap interval of the pooled ratio lies entirely below 0.5 or entirely above 2 (pre-registration, version 1.2, Section 9, item 2, which version 1.3 does not amend).}",
                 "\\label{tab:corevariants}", "\\small",
                 "\\begin{tabularx}{\\linewidth}{@{}l>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}X@{}}", "\\toprule",
                 "Variant & Statistic & Value\\\\", "\\midrule"]
        lines += [f"{a} & {b} & {c}\\\\" for a, b, c in rows]
        lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
        self.put("TAB.corevariants", f"{len(rows)} rows in tables/core_variants.tex", ["C1.secondary.C1b_*", "C1.secondary.C1d_DCLM_control", "C1.secondary.C1e_within_base_R2", "C1.secondary.M12a_alignment_Lambda", "C2.C2_eq", "blocks.C2_eq_rand"])
        return "\n".join(lines)

    # -- figure caption (FIG.crossover) and per-draw table (TAB.coredraws)
    def figure_and_table(self, root):
        d = self.doc
        pdr = F("C1.per_draw", d) or []; sym = {(r["base"], r["d"]): r.get("I_log") for r in (F("C1.sym_draws", d) or [])}
        c1fig = {}
        for r in pdr:
            row = {"d": r["d"], "P": r["P"], "I": r["I"]}
            if isnum(sym.get((r["base"], r["d"]))):
                row["I_sym"] = sym[(r["base"], r["d"])]
            c1fig.setdefault(r["base"], []).append(row)
        c2fig = {}
        for r in F("C2.per_draw", d) or []:
            c2fig.setdefault(r["base"], []).append({"d": r["d"], "DQ": r["Delta_Q"], "I2": r["I2"], "J2": r["J2"]})
        lines = ["% generated by scripts/fill_core.py from the core analysis output -- do not edit",
                 "\\begin{table}[ht]", "\\centering",
                 "\\caption{Per-draw results of the core, as written by the registered analysis. C1: crossover $I_d$, predicted contrast $P_d$ "
                 "and $J_d=I_d-\\beta_{0,b}P_d$; C2: interaction $I2_d$ at step 8, target contrast $\\Delta_{Q,d}$ and $J2_d=I2_d-0.25\\,\\Delta_{Q,d}$.}",
                 "\\label{tab:coredraws}", "\\small", "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
                 "Block & Base & $d$ & $I_d$ / $I2_d$ & $P_d$ / $\\Delta_{Q,d}$ & $J_d$ / $J2_d$\\\\", "\\midrule"]
        hit = set()
        for rid in getattr(self, "overlap", set()):
            mm = re.match(r"c1_(moe_b\d)_books_d(\d+)_[AB]$", str(rid))
            if mm:
                hit.add((mm.group(1), int(mm.group(2))))
        mark = lambda r: "$^{\\dagger}$" if (r["base"], int(r["d"])) in hit else ""
        names = dict(CONFIG["bases"])
        for r in pdr:
            lines.append(f"C1 & {names.get(r['base'], r['base'])} & {r['d']}{mark(r)} & {num(r['I'])} & {num(r['P'])} & {num(r['J'])}\\\\")
        if c2fig:
            lines.append("\\midrule")
            for r in F("C2.per_draw", d) or []:
                lines.append(f"C2 & {names.get(r['base'], r['base'])} & {r['d']} & {num(r['I2'])} & {num(r['Delta_Q'])} & {num(r['J2'])}\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        if any((r["base"], int(r["d"])) in hit for r in pdr):
            lines.append("\\par\\smallskip{\\footnotesize $^{\\dagger}$\\,A run of this draw overlapped the CPU work of the A-2 precheck (incident I-2, "
                         "Appendix~\\ref{app:prereg}); its wall time and energy are affected, the values in this table are not.\\par}")
        lines += ["\\end{table}", ""]
        cap = ("C1 crossover against the predicted contrast for MoE bases 1 and~2 (a, b) and the C2 interaction against the target contrast at step 8 (c). "
               "Crosses: G0-7 probe crossovers at the pretrained weights (pinned, norm-matched random directions); filled and open circles: adapted draws, "
               "raw and symmetrized; in (c), diamonds and squares: C2 draws of bases 1 and~2. Solid line: the prediction $I=P$ ($\\beta=1$); dotted lines: "
               "the margins $I=0.25\\,P$ and $I2=0.25\\,\\Delta_{Q,d}$.")
        if pdr:
            self.put("FIG.crossover", cap, ["C1.per_draw", "C1.statistics.sym.draws", "C2.per_draw"])
        else:
            self.skip("FIG.crossover", "no C1 draws")
        return c1fig, c2fig, "\n".join(lines) if pdr else None


def texid(run):
    """a run id in typewriter type, TeX-escaped"""
    return "\\texttt{" + str(run).replace("\\", "").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#") + "}"


def texval(v):
    """a field value of an E1 record or a warning as TeX-safe text"""
    if isinstance(v, dict):
        v = ", ".join(f"{k} {texval(x)}" for k, x in v.items())
    elif isinstance(v, (list, tuple)):
        v = ", ".join(texval(x) for x in v)
    s = str(v)
    for a, b in (("\\", ""), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#"), ("$", "\\$"), ("{", "("), ("}", ")")):
        s = s.replace(a, b)
    return s


def run_name(run):
    """a C2 core run named by block, base, draw, optimizer and subset (v1.6.8), such as c2_moe_b1_d3_sgd_H ->
    C2, base~1, draw~3, SGD, high-usage subset; any other id stays in typewriter type"""
    m = re.match(r"c2_moe_b(\d)_d(\d+)_(sgd|muon)_([HL])$", str(run))
    if not m:
        return texid(run)
    return (f"C2, base~{m.group(1)}, draw~{int(m.group(2))}, {'SGD' if m.group(3) == 'sgd' else 'Muon'}, "
            f"{'high' if m.group(4) == 'H' else 'low'}-usage subset")


def run_block(run):
    """block and base number from a core run id such as c2_moe_b1_d1_sgd_H"""
    m = re.match(r"c(\d)_moe_b(\d)", str(run))
    return (f"C{m.group(1)}", m.group(2)) if m else ("--", "--")


def tex_escape_check(text, fid=""):
    if text.count("{") != text.count("}"):
        raise SystemExit(f"[fill_core] unbalanced braces: {text}")
    if "#" in text:
        raise SystemExit(f"[fill_core] '#' in fill text: {text}")
    if re.search(r"(?<!\\)%", text):
        raise SystemExit(f"[fill_core] unescaped '%' in fill text: {text}")
    if not fid.startswith("TAB.") and re.search(r"(?<!\\)_", re.sub(r"\$[^$]*\$|\\url\{[^}]*\}", "", text)):
        raise SystemExit(f"[fill_core] '_' outside math in fill text: {text}")


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def apply_final_texts(f, root, sha):
    """v1.6.1: final texts of the final pass (scripts/final_texts_core.json), applied after the fill through Filler.put
    (word limits apply) when the file names the SHA-256 of this analysis output; otherwise nothing is replaced and a warning is kept"""
    p = root / "scripts" / "final_texts_core.json"
    if not p.exists():
        return None
    ft = json.load(open(p, encoding="utf-8"))
    if ft.get("analysis_sha256") != sha:
        f.warn.append(f"scripts/final_texts_core.json is for analysis sha256 {str(ft.get('analysis_sha256'))[:12]}, not {sha[:12]}: not applied")
        return {"file": "scripts/final_texts_core.json", "applied": [], "reason": "analysis sha256 differs"}
    done = []
    for fid, e in ft["texts"].items():
        if fid not in f.fills:
            raise SystemExit(f"[fill_core] final text for {fid}, which this fill did not produce")
        f.put(fid, e["text"], e.get("fields") or f.fills[fid]["fields"]); done.append(fid)
    return {"file": "scripts/final_texts_core.json", "sha256": sha256(p), "source": ft.get("source"), "applied": done}


def main(argv=None):
    ap = argparse.ArgumentParser(description="fill the core result slots of the paper from the core analysis output")
    ap.add_argument("--analysis", default=CONFIG["analysis_json"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root); src = Path(a.analysis)
    doc = json.load(open(src, encoding="utf-8"))
    if F("format", doc) != CONFIG["expected_format"]:
        raise SystemExit(f"[fill_core] format {F('format', doc)!r} != {CONFIG['expected_format']!r}: update CONFIG['fields'] first")
    sha = sha256(src); side = src.with_name(src.name + ".sha256")
    if side.exists() and side.read_text(encoding="ascii").split()[0] != sha:
        raise SystemExit("[fill_core] analysis_confirm.json does not match its .sha256 file")
    mode, cb = F("mode", doc), F("core.blocks", doc)
    if mode != CONFIG["expected_mode"]:
        raise SystemExit(f"[fill_core] mode {mode!r} != {CONFIG['expected_mode']!r}: the paper is filled from the core analysis only; nothing written")
    if list(cb or []) != CONFIG["core_blocks"]:
        raise SystemExit(f"[fill_core] core.core_blocks {cb!r} != {CONFIG['core_blocks']!r}: not the registered core (P-16 revised); nothing written")
    ep_path = root / CONFIG["edge_pre_json"]
    ep = json.load(open(ep_path, encoding="utf-8")) if ep_path.exists() else None
    f = Filler(doc, ep)
    hp = root / CONFIG["hours_outside_json"]
    f.hos = json.load(open(hp, encoding="utf-8")) if hp.exists() else None
    op = root / CONFIG["edge_overlap_json"]
    f.overlap = set(json.load(open(op, encoding="utf-8")).get("runs") or []) if op.exists() else set()
    f.layout_checks(); f.c1(); f.c2(); f.c4(); f.edge(); f.compute_used(); f.hours_outside(); f.e1_outcome(); f.secondary()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fill_author                                     # author slots: AI-assisted drafts for the author's review
    f.author = fill_author.fill_author(f, sys.modules[__name__])
    vtab = f.variants()
    c1fig, c2fig, table = f.figure_and_table(root)
    final = apply_final_texts(f, root, sha)
    for fid, v in f.fills.items():
        tex_escape_check(v["text"], fid)   # TAB.* texts are bookkeeping, typeset only if the generated table is missing
    out = CONFIG["out"]
    hdr = (f"% generated by scripts/fill_core.py -- do not edit\n% source {src} sha256 {sha}\n% format {F('format', doc)}; "
           f"analysis created {doc.get('created_utc')}; fills written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
           f"% author slots {', '.join(sorted(f.author))}: {fill_author.STATUS} (scripts/fill_author.py)\n")
    write_atomic(root / out["fills"], hdr + "".join(f"\\setfill{{{k}}}{{{v['text']}}}\n" for k, v in sorted(f.fills.items())))
    if table:
        write_atomic(root / out["table"], table)
    if vtab:
        write_atomic(root / out["variants"], vtab)
    if f.e1_table:
        write_atomic(root / out["e1"], f.e1_table)
    write_atomic(root / out["c1_fig_data"], json.dumps(c1fig, indent=1) + "\n")
    if c2fig:
        write_atomic(root / out["c2_fig_data"], json.dumps(c2fig, indent=1) + "\n")
    man = {"source": str(src), "sha256": sha, "format": F("format", doc), "analysis_created_utc": doc.get("created_utc"),
           "analysis_warnings": F("warnings", doc) or [], "fill_warnings": f.warn, "fills": f.fills, "pending": f.pending,
           "core": {"blocks": cb, "runs": F("core.runs", doc)}, "E1_diverged_runs": f.e1_records, "E1_incomplete_logs": f.incomplete, "E1_blocks": f.e1_blocks, "E1_table": out["e1"] if f.e1_table else None,
           "author_ids": CONFIG["author_ids"], "author_slots": {"status": fill_author.STATUS, "slots": f.author},
           "plan_freeze_ids": CONFIG["plan_freeze_ids"], "final_texts": final}
    write_atomic(root / out["manifest"], json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    if not a.no_figure:
        r = subprocess.run([sys.executable, str(root / "scripts" / "fig_crossover.py")], cwd=str(root), input="", capture_output=True, text=True,
                           env=dict(os.environ, CORE_DATA_DIR=str(root / "data")))
        if r.returncode:
            raise SystemExit("[fill_core] fig_crossover.py failed: " + r.stderr[-800:])
    print(f"[fill_core] {len(f.fills)} filled, {len(f.pending)} pending: {sorted(f.pending)}; warnings in analysis: {len(man['analysis_warnings'])} "
          f"(training logs flagged incomplete: {len(f.incomplete)}); fill warnings: {f.warn}")
    print(f"[fill_core] final texts: {len(final['applied']) if final else 0} applied from scripts/final_texts_core.json")
    print(f"[fill_core] author slots: {fill_author.STATUS}: " + "; ".join(f"{k}={v['variant']}" for k, v in sorted(f.author.items())))
    for w in man["analysis_warnings"]:
        print(f"[fill_core] analysis warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
