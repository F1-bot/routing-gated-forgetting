"""fill_full.py -- mechanical fill of the registered blocks outside the core (paper v1.6.7 onward).

Reads the FULL-mode output of the confirmatory analysis (analyze_confirm.py run --mode full, format analysis_confirm/1,
results/analysis/analysis_confirm.json, written once after the registered queue) and the confirm queue state (the three
overrun guards), and writes, without any hand edit of a .tex file:
  fills/full_fills.tex        one \\setfill{ID}{text} per slot of the remaining blocks (macros.tex loads it if it exists)
  fills/full_manifest.json    every filled ID with its text, variant, registered rule, JSON fields and number map
                              (printed token -> field of data/full_results.json); the IDs left pending and why
  data/full_results.json      trimmed derived copy: only the values printed by the fills (source of the ledger entries)
  tables/block_status.tex     Table tab:blockstatus with the status of every registered block
  tables/full_*.tex           per-draw values, dose candidates, M12-b, selpin shares, C3 decomposition, C3 single-layer
                              reverts, scores of the forecasters, planned and measured GPU-hours, assumption audit (generated tables)
  scripts/ledger_manual.json  with --ledger: one 'token@fills/full_fills.tex' entry per number printed by the fills
Labels are copied from the analysis output, never recomputed. Each registered label is re-derived from the printed numbers
by its registered rule as a consistency check only; if the two disagree, nothing is written. Every template follows the
house-style rules of the de-logging check of v1.6.6 (rule 6 as overridden): findings in the past tense with
estimate, interval, number of draws and registered label; runs named by base, draw, subset, target and seed; no job
identifiers, queue positions, dates or times; exclusions as counts under the registered rule; 'exploratory' in every
sentence with an exploratory value; no status language; each number stated once (tables are not repeated in the text).
The results of the assumption audit (FULL.audit and Table tab:fullaudit) come from blocks.audit, read strictly and cross-checked
with the A-3 re-run record base by base. The cost block over all covered runs (FULL.EDGE.all, Appendix F, from v1.6.11) comes from blocks.EDGE (by_type, excluded, runs, pre_update_check) and stays pending if the analysis output no longer fits its approved wording. Five post-freeze items are filled from their
records: deviation D-2 with amendment A-3 (assumption audit) from the A-3 re-run record, deviation D-3 with amendment A-4 (bitwise
audit of the two C3 freeze runs) from the A-4 acceptance record and the queue state, incident I-4 (dose-response), incidents
I-5 and I-6 (two interrupted attempts of one C1-DE run) and incident I-7 (one interrupted run of the assumption audit); their hours come from data/hours_outside_state.json, which must hold the
items of CONFIG hos_new with their records and labels, and the core fill (scripts/fill_core.py) must
have been run on that file first, so that its HOS slots carry the same total; COMPUTE.used names the core items without their
hours and points to the last row of Table tab:budget (final_build_patch_compute_used.py, rule 11), or the fill stops.

One-time structural step (--integrate, after the operator has approved the text changes in templates_remaining_blocks.md):
macros.tex loads the fills, build_ledger.py and check_paper.py see fills/full_fills.tex, write_ledger_md.py gets the slot
descriptions (scripts/placeholders_full.json), ledger_sources.json registers the generated tables, the appendix with the
slots is added (sections/H_full.tex), the sentences that say 'reported separately' point to it, and fill_core.py stops
writing HOS.items, whose slot the integration removes from the last row of Table tab:budget. Every patch is
idempotent and refuses, before writing anything, if one of its anchor texts is not found exactly once.

Usage (from the paper root, CPU only):
  python scripts/fill_full.py --integrate
  python scripts/fill_full.py [--analysis PATH] [--state PATH] [--a2-acceptance PATH] [--a3-result PATH] [--a4-acceptance PATH] [--ledger]
  python scripts/fill_full.py --write-templates templates_remaining_blocks.md
then rebuild, scripts/build_ledger.py, scripts/write_ledger_md.py, scripts/check_paper.py.
The A-2 acceptance record is {"accepted": {"<evaluation job of the G1 run>": true | false, ...}} for the three evaluations
listed in CONFIG["a2_jobs"]; a missing or malformed record stops the fill. The A-3 re-run record is exactly {"utc", "run_queue_exit",
"dry_run": false, "jobs": {job: {"status", "rc", "hours", "outputs": {the three registered files: true|false}, "shim_line_in_log"}}}
for the two audit jobs of CONFIG["a3_jobs"]; the A-4 acceptance record is exactly {"accepted": {job: true|false}, "checks", "rule",
"utc"} for the two runs of CONFIG["a4_jobs"]. A defect of either record, a missing item of CONFIG["hos_new"], a total that differs
from the sum of the items, or core fills with another total stop the fill before anything is written. blocks.audit must hold, for
each base of CONFIG["a3_jobs"], either the registered result of the frozen audit analysis (B5, B6 for subsets A and B, A0 for DCLM,
code and math, 2,000 window resamples) or the analysis's 'not available' status, as the A-3 record says; otherwise the fill stops.
A degenerate result that the frozen code allows and the registered design does not describe (no label, fewer than 24 groups,
fewer valid resamples) leaves FULL.audit and its table without a fill, with the reason in the manifest.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fill_core as FC  # noqa: E402  helpers shared with the core fill: get, isnum, num, pfmt, words, tex_escape_check, write_atomic, sha256

CONFIG = {
    "analysis_json": r"results\analysis\analysis_confirm.json",
    "state_json": r"results\queue_state_confirm.json",
    "expected_format": "analysis_confirm/1",
    "expected_mode": "full",
    "out": {"fills": "fills/full_fills.tex", "manifest": "fills/full_manifest.json", "data": "data/full_results.json",
            "status": "tables/block_status.tex", "draws": "tables/full_draws.tex", "dose": "tables/full_dose.tex",
            "m12b": "tables/full_m12b.tex", "selpin": "tables/full_selpin.tex", "c3": "tables/full_c3.tex",
            "c3layer": "tables/full_c3layer.tex", "forecasts": "tables/full_forecasts.tex", "budget": "tables/full_budget.tex",
            "audit": "tables/full_audit.tex", "placeholders": "scripts/placeholders_full.json"},
    "bases": {"moe_b1": "base~1", "moe_b2": "base~2", "moe_b3": "base~3", "nestor": "\\nestor{}"},
    "base_long": {"moe_b1": "MoE base~1", "moe_b2": "MoE base~2", "moe_b3": "MoE base~3", "nestor": "\\nestor{}"},
    # registered cut order of the frozen plan (prereg v1.3 v11 sec. 7) and the run_queue gate of each guard (confirm_plan.py R10)
    "guards": {"M12b": "overrun_guard_M12b", "dose": "overrun_guard_dose", "C1-DE": "overrun_guard_C1-DE"},
    "guard_names": {"M12b": "M12-b", "dose": "the dose-response", "C1-DE": "C1-DE"},
    # amendment A-2: the three weight-decay-only evaluations of G1 runs whose values need its acceptance rule; key = evaluation job
    # of the confirm queue (machine side only), value = (base, the run as the text names it: base, target, seed)
    "a2_jobs": {"c3_revert_g1_nestor_full_german_s101": ("nestor", "german", "the \\nestor{} run on German with seed 101"),
                "c3_revert_g1_moe_b1_full_math_s101": ("moe_b1", "math", "the MoE base~1 run on mathematics with seed 101"),
                "c3_revert_g1_moe_b1_full_german_s101": ("moe_b1", "german", "the MoE base~1 run on German with seed 101")},
    # weight-decay-only evaluations of the C3 full-adaptation runs exist for MoE base 1 and NESTOR only (frozen plan: confirm runs
    # c3_full_moe_b1_books_s101, c3_full_nestor_books_s101, c3_full_nestor_math_s101 plus the three A-2 evaluations); base 2 has none
    "wd_bases": [("moe_b1", "MoE base~1"), ("nestor", "\\nestor{}")],
    # amendment A-2 acceptance record, written by confirm/post_queue/a2_acceptance.py: exactly {"accepted": {"<evaluation job>": true|false}, "checks": {...}}
    "a2_acceptance_json": r"results\a2_acceptance.json",
    # deviation D-2, amendment A-3: re-run record of the assumption audit, written by confirm/rerun/watch_and_rerun.ps1; key = audit job
    # of the confirm queue (machine side only), value = (base, the base as the text names it)
    "a3_result_json": r"confirm\rerun\rerun_result.json",
    "a3_jobs": {"audit_moe_b1": ("moe_b1", "MoE base~1"), "audit_moe_b2": ("moe_b2", "MoE base~2")},
    "a3_outputs": ["audit_realized_groups.json", "audit_realized_windows.npz", "audit_realized.json"],
    # results of the assumption audit (prereg v1.3 sec. 4.3; frozen assumption_audit.analyze_realized as read by analyze_confirm.block_audit):
    # subsets of C1 draw 1 (run-id suffix) with their names in the text, the old corpora in registered order (code stream of the G0
    # record), groups per base, window resamples and the label cuts of the frozen code (B5_CUTS, CV_HOMOG, A0_R, A0_SHARE)
    "audit": {"subsets": {"A": "$A_1$", "B": "$B_1$"}, "corpora": ["dclm", "code", "math"], "groups": 24, "n_boot": 2000,
              "boot_unit": "windows within corpus", "b5_cuts": [(0.1, "approximately separable"), (0.3, "moderately non-separable"),
              (float("inf"), "non-separable")], "cv_homog": 0.25, "a0_r": 0.2, "a0_share": 0.25},
    # deviation D-3, amendment A-4: acceptance record of the two C3 freeze runs, written by confirm/rerun_a4/a4_acceptance.py; key =
    # training job of the confirm queue (machine side only), value = (base, the run as the text names it; target German, seed 103)
    "a4_acceptance_json": r"results\a4_acceptance.json",
    "a4_jobs": {"c3_freeze_moe_b1_german_s103": ("moe_b1", "the MoE base~1 run"), "c3_freeze_nestor_german_s103": ("nestor", "the \\nestor{} run")},
    # hours outside the planned runs: the core fill's source, updated from the deviation record before the full fill; the post-freeze
    # items that must be present, with their record, and the phrases of the Appendix C sentence (every item in exactly one phrase)
    "hos_json": "data/hours_outside_state.json",
    "expl_json": "data/exploratory_values.json",
    "c5_forecasts_json": r"confirm\c5_french_forecasts.json",   # hashed C5 file: read only to test whether each forecast is null (deviation D-4)
    "c5_not_evaluable": "not evaluable as registered",
    "audit_wording": {"A0 violated": "A0 does not hold approximately"},   # analysis strings that are not registered labels -> printed wording (20 November)              # exploratory runs outside the registered plan: gpu_hours_total (Appendix app:exploratory)
    "hos_new": {"failed_start_audit": "D-2", "first_attempts_c3_freeze": "D-3", "interrupted_attempt_dose": "I-4",
                "interrupted_attempt_c1de_i5": "I-5", "interrupted_attempt_c1de_i6": "I-6",
                "interrupted_attempt_audit_i7": "I-7"},
    "hos_phrases": [(("failed_attempt_d2_A",), "the stopped first attempt of one C1 run (deviation D-1)"),
                    (("gpu_check_A1",), "the GPU check of amendment A-1"),
                    (("killed_attempt_d10_B",), "the interrupted attempt of one C1 run (incident I-1)"),
                    (("gpu_check_A2",), "the GPU check of amendment A-2"),
                    (("failed_start_audit",), "the two stopped starts of the assumption audit (deviation D-2)"),
                    (("first_attempts_c3_freeze",), "the first attempts of the two C3 freeze runs (deviation D-3)"),
                    (("interrupted_attempt_dose",), "the interrupted attempt of one dose-response run (incident I-4)"),
                    (("interrupted_attempt_c1de_i5", "interrupted_attempt_c1de_i6"), "the two interrupted attempts of one C1-DE run (incidents I-5 and I-6)"),
                    (("interrupted_attempt_audit_i7",), "the interrupted run of the assumption audit under amendment A-3 (incident I-7)")],
    # the two registered figures (prereg v1.3 sec. 4.8), produced by the figure scripts from the full-mode output; --integrate stops if a file is missing
    "figures": {"c3layers": {"file": "figures/fig_c3_layers.pdf", "label": "fig:c3layers", "slot": "FIG.c3layers.caption"},
                "c4windows": {"file": "figures/fig_c4_windows.pdf", "label": "fig:c4windows", "slot": "FIG.c4windows.caption"}},
    "forecasters": [("RO", "RO only"), ("QE", "$Q_E$ only"), ("M1", "M1"), ("M2", "M2"), ("SVC", "second moments"),
                    ("TF", "trainable fraction"), ("RND", "random-direction probe"), ("KSD", "$k$-step drive")],
    "forecast_blocks": [("C1", "C1"), ("B3", "base~3"), ("DE", "C1-DE"), ("DR", "dose"), ("M12B", "M12-b"), ("C2", "C2"), ("C3", "C3")],
    "c5_names": {"ro": "RO only", "k_step": "the $k$-step drive", "trainable_fraction": "the trainable fraction"},
    "budget_names": {"C1:moe_b1": "C1, base~1", "C1:moe_b2": "C1, base~2", "C2": "C2", "base_eval": "Evaluations at $\\theta_0$",
                     "selpin": "C1 selection-pinned evaluations", "C3": "C3", "DCLM-CONT": "C3, DCLM continuation runs",
                     "audit": "Assumption audit", "C5": "C5", "b3": "Third base", "dose": "Dose-response", "M12b": "M12-b",
                     "C1-DE": "C1-DE"},
    # descriptive names of revert groups and variants (house-style rule 2/4: no internal identifiers in the text)
    "groups": {"experts": "routed experts", "rest": "attention, embeddings, head and norms", "column_private": "column-private group",
               "shared": "shared group", "moe": "routed experts", "attn+priv": "column-private group"},
    "variants": {"revert:moe": "routed experts", "revert:kind:attn+embed_head+kind:norm": "attention, embeddings, head and norms",
                 "revert:embed_head": "embeddings and head", "revert:attn+priv": "column-private group", "revert:attn": "column attention",
                 "revert:priv": "private experts", "revert:pub": "public experts", "revert:shared+pub+embed_head": "shared group"},
    "word_limits": {"FULL.summary": 60},
    "appendix_file": "sections/H_full.tex",
    "appendix_position": "before_exploratory",    # decision: 'before_exploratory' (new H, exploratory becomes I) or 'after_exploratory'
}
CORPUS = {"code": "code", "math": "mathematics", "dclm": "DCLM", "german": "German", "books": "books", "pycode": "Python code"}
FIELDS = "results/analysis/analysis_confirm.json"

# numbers printed by the fixed template text (not from the analysis): token -> meaning; every other printed number
# must come from a recorded field, or the fill stops
CONSTANTS = {
    "0": "comparison value 0 of the registered rules; subscript 0 of beta_{0,3} (notation)",
    "1": "base 1 (notation); label digit of deviation D-1 and incident I-1; draw 1 of the C1 books runs whose displacements the assumption audit probes (prereg v1.3 sec. 4.3)",
    "2": "base 2; R^2 (notation); label digit of deviation D-2 and amendment A-2; draw 2 of the C1-DE run of incidents I-5 and I-6",
    "3": "base 3; the three draws of the registered third-base and M12-b designs (prereg v1.3 secs. 4.1, 4.5); label digit of deviation D-3 and amendment A-3; draw 3 of the dose-response run of incident I-4",
    "4": "label digit of amendment A-4 and incident I-4",
    "7": "label digit of incident I-7 (FULL.hos names incidents I-5, I-6 and I-7)",
    "103": "seed 103 of the two C3 freeze runs (runs named by base, target and seed)",
    "8": "Muon stop step of the C2 runs (prereg v1.3 v11 sec. 3.4); in FULL.EDGE.all, the step count of the short subset runs of the cost block (blocks.EDGE.runs, steps)",
    "24": "SGD stop step of the C2 runs (prereg v1.3 v11 sec. 3.4); the 24 module groups per base and the 24 windows per corpus of the assumption audit (prereg v1.3 sec. 4.3)",
    "50": "first 50 candidate seeds of the dose-response acceptance rule (prereg v1.3 sec. 4.2)",
    "95": "95% intervals (registered level)",
    "96": "96 steps of the registered adaptation schedule",
    "101": "seed 101 of the full-adaptation runs of gate G1 (runs named by base, target and seed)",
    "600": "solver time limit of the C1-DE generator in seconds (prereg v1.3 sec. 4.12)",
    "0.05": "registered test level (prereg v1.3 secs. 4.1, 4.2)",
    "0.25": "registered C1 margin beta_0 used by C1-DE (prereg v1.3 sec. 4.12)",
    "2.048": "2.048M DCLM tokens per single-layer revert evaluation (prereg v1.3 sec. 4.8)",
    "12": "the 12 single-layer reverts of MoE base 1 (prereg v1.3 sec. 4.8); the 12 layers of the assumption audit, one group per subset and layer (prereg v1.3 sec. 4.3)",
    "6": "the 6 single-block reverts of the NESTOR column-private group, numbered 1 to 6 as in Figure 2 (prereg v1.3 sec. 4.8: blocks 0-5); label digit of incident I-6",
    "5": "block 5, the last of the pre-registration's 0-based NESTOR blocks 0-5 (prereg v1.3 sec. 4.8); label digit of incident I-5",
    "10.1": "section 10.1 of pre-registration v1.2 (C3 groups, comparison rule), cited",
    "1.2": "pre-registration version 1.2, cited",
    "13": "section 13 of pre-registration v1.2 (missing values), cited",
}


class Pending(Exception):
    pass


# shared parts of the deviation templates (house-style rules 5 and 6 as overridden: reason in one clause, no sequence of events)
D2_HEAD = r"The pre-registration runs the assumption audit at $\theta_0$ on MoE bases~1 and~2 directly after the core. Its frozen code calls a function that the frozen data-selection module does not define, so the audit ran after the other registered blocks through a wrapper that restores that function verbatim (amendment A-3). The wrapper runs the frozen audit code with the registered arguments and changes no frozen file; the inputs of the audit were complete when the core ended, and no other block uses its outputs, so the later order changes no value. "
D2_COST = r" The two starts that stopped at the missing function used [[hfail]] GPU-hours and the two audit runs under the amendment [[hrun]] GPU-hours (Appendix~\ref{app:power})."
D3_HEAD = r"The registered bitwise audit of a run requires every trainable parameter to change. In the two freeze runs of block C3, on German with seed 103 for MoE base~1 and for \nestor{}, the pseudo-query of the first depth-attention read of the attention residual keeps its initial value, because that read has a single source and so receives an exactly zero gradient; both runs were therefore repeated with the same seeds under an audit that exempts this one named parameter (amendment A-4). Every other condition of the audit is unchanged, so no frozen tensor, buffer or optimizer state may change, and the acceptance rule of the amendment accepts a run only if exactly the named parameter is unchanged and its subset manifest and environment equal those of the first attempt. "
D3_TAIL = r" No other run of the study has a freeze specification, and no confirmatory statistic uses these two runs. The first attempts used [[hfirst]] GPU-hours and the repeated runs [[hrep]] GPU-hours (Appendix~\ref{app:power})."


# parts of the assumption-audit template (prereg v1.3 sec. 4.3): registered quantities and labels only; the values stand in Table tab:fullaudit
AUD_SCOPE = r" along the realized displacements of the C1 books runs of draw~1, for 24 groups $g$[[per]], the routed experts of subset $A_1$ or $B_1$ in one of the 12 layers, on DCLM, code and mathematics with 24 windows each"
AUD_QUANT = r" Table~\ref{tab:fullaudit} gives the registered quantities with their labels, where $\gamma_c(g)$ and $Q_c(g)$ are the first- and second-order terms of group $g$ on corpus $c$, $a_g$ is the group effect of the decomposition $\log Q_c(g)=a_g+b_c+e_{gc}$, and $r=|\gamma_c(g)|/Q_c(g)$.[[labels]][[corr]]"
AUDIT_CAPTION = (r"Assumption audit at $\theta_0$ (secondary; no decision depends on the labels). (a)~B5: share of the interaction $e_{gc}$ in the "
                 r"decomposition of $\log Q_c(g)$ after subtracting the expected noise, with its 95\% window-bootstrap interval (2,000 resamples), and "
                 r"the cells with $Q_c(g)\le 0$, which B5 and A0 exclude. (b)~B6, per subset: coefficient of variation (CV) of $e^{a_g}$ over the 12 "
                 r"groups, Spearman's $\rho$ between $a_g$ and the group's share of $P$ with its bootstrap interval, and the Jensen gap "
                 r"$\log\mathrm{mean}(e^{a_g})-\mathrm{mean}(a_g)$. (c)~A0, per corpus: median of $r$ and share of cells with $r\ge 0.2$. Registered "
                 r"labels: B5 ``approximately separable'' below 0.1, ``moderately non-separable'' from 0.1 to 0.3 and ``non-separable'' above 0.3; B6 "
                 r"``homogeneous'' if the CV is below 0.25, otherwise ``heterogeneous, uncorrelated'' if the interval of $\rho$ contains 0 and "
                 r"``heterogeneous, correlated'' if not; ``A0 approximately holds'' if the share is at most 0.25; the pre-registration names no label for a larger share, for which the table states that A0 does not hold approximately.")
AUDIT_HEADS = [r"(a) B5 & Interaction share & 95\% interval & Cells with $Q\le 0$ & Label",
               r"(b) B6 & Subset & CV & $\rho$ & 95\% interval of $\rho$ & Jensen gap & Label",
               r"(c) A0 & Corpus & Median $r$ & Share $r\ge 0.2$ & Label"]
AUDIT_SPECS = ["@{}lrll>{\\raggedright\\arraybackslash}X@{}", "@{}llrrlr>{\\raggedright\\arraybackslash}X@{}", "@{}llrr>{\\raggedright\\arraybackslash}X@{}"]
# exact key sets of the frozen audit analysis (analyze_realized) and its registered labels
AUDIT_KEYS = {"B5": {"interaction_share", "ci95", "label", "SS", "noise_var_logQ", "cells_Q_nonpositive", "cells", "bootstrap"},
              "B6": {"cv", "spearman", "jensen_gap", "cv_ci95", "spearman_ci95", "label"},
              "A0": {"median_r", "share_r_ge_0.2", "n", "label"}}
AUDIT_LABELS = {"B5": ["approximately separable", "moderately non-separable", "non-separable"],
                "B6": ["homogeneous", "heterogeneous, uncorrelated", "heterogeneous, correlated"],
                "A0": ["A0 approximately holds", "A0 violated"]}


# ----------------------------------------------------------------------------- templates (registered sentences)
# [[name]] marks a value slot. Slots inside $...$ get math-mode numbers. 'where' and 'length' plan the placement; 'rule'
# cites the registered reporting rule.
T = {
    "FULL.summary": {
        "block": "All remaining blocks (main text)", "where": "Section 5.5, replaces the last clause of its last sentence", "length": "about 4.5 lines (at most 60 words; 57 on the mock with the clause that refers to Table tab:fullaudit, which adds about half a line); net about +2 lines (estimate from the text, not built)",
        "rule": "registered labels only, no interpretation; the summary names no label of the assumption audit, whose labels are given per base, assumption and subset or corpus, and refers to Table tab:fullaudit in one clause, added only when that table has rows and the sentence stays within its word limit (the runs of the audit in Appendix A under D-2, its results in Appendix H)",
        "variants": {"labels": r"Appendix~\ref{app:full} reports the other registered blocks, labelled [[items]]; C3, the selection-only pinned replay and the forecaster scores are descriptive[[aud]]."},
        "parts": {"aud": r", and Table~\ref{tab:fullaudit} gives the labels of the assumption audit"}},
    "FULL.B3.result": {
        "block": "Third base (B3)", "where": "new appendix, paragraph 'Third base'", "length": "6 lines",
        "rule": "prereg v1.3 sec. 4.1: 'Consistent with C1' if J_d > 0 for d = 1, 2, 3; 'Inconsistent' if the upper 95% t-bound of mean J is below 0; otherwise 'inconclusive'; mean J with t-interval (df 2); exact one-sided sign-flip p (minimum 1/8, no test at 0.05); beta-hat_3 through the origin with its test-inversion interval, unbounded at D = 3 (R3); beta-hat_3,pin and beta-hat_3,sym; beta-hat_3 - beta-hat (C1, bases pooled) with a Welch interval; threat 9.1 item 5",
        "variants": {
            "consistent with C1": r"On the third base, $J_d=I_d-\beta_{0,3}P_d$ with the margin $\beta_{0,3}=[[b0]]$ was positive in each of the [[n]] draws, so the registered label is ``consistent with C1''.",
            "inconsistent": r"On the third base, with the margin $\beta_{0,3}=[[b0]]$, the upper 95\% $t$-bound of mean $J$ over the [[n]] draws was below 0, so the registered label is ``inconsistent''.",
            "inconclusive": r"On the third base, with the margin $\beta_{0,3}=[[b0]]$, not every $J_d$ of the [[n]] draws was positive and the upper 95\% $t$-bound of mean $J$ was not below 0, so the registered label is ``inconclusive''.",
            "no usable draws": r"The third base has no usable draws under rule E1, so it has no label."},
        "tail": r" Mean $J$ was [[mJ]] (95\% $t$-interval [[ciJ]], [[df]] degrees of freedom), and the exact one-sided sign-flip $p$ was $[[p]]$, whose minimum with [[n]] draws is $[[minp]]$, so no test at the 0.05 level is possible. Regression through the origin gave $\hat\beta_3=[[b3]]$, whose 95\% test-inversion interval is [[b3ci]] at $D=[[n]]$, with $\hat\beta_{3,\mathrm{pin}}=[[bpin]]$ and $\hat\beta_{3,\mathrm{sym}}=[[bsym]]$; the difference from C1 with both bases pooled was $\hat\beta_3-\hat\beta=[[db]]$ [[cidb]] (Welch interval). Base~3 differs from bases 1 and~2 in training stage, seed and training platform together, and this comparison does not separate them.[[e1]] Table~\ref{tab:fulldraws} gives the values per draw."},
    "FULL.dose.result": {
        "block": "Dose-response", "where": "new appendix, paragraph 'Dose-response'", "length": "4 lines",
        "rule": "prereg v1.3 sec. 4.2: 'linear dose-response' iff the 95% CI of b1 excludes 0 and the b3 test does not reject; 'non-linear dose-response' if the b1 CI excludes 0 and the b3 test rejects; 'no dose-response' if the b1 CI contains 0 (raw I primary; HC3 t-intervals; two-sided HC3 test of b3 = 0 at 0.05)",
        "variants": {
            "linear dose-response": r"The dose-response model $E[I\mid P]=b_1P+b_3P^3$ without intercept, fitted by least squares to [[n]] draws at [[nlev]] levels, gave $b_1=[[b1]]$ (95\% HC3 interval [[cib1]], which excludes 0) and $b_3=[[b3]]$ [[cib3]], and the two-sided HC3 test of $b_3=0$ did not reject at the 0.05 level ($p=[[pb3]]$), so the registered label is ``linear dose-response''.",
            "non-linear dose-response": r"The dose-response model $E[I\mid P]=b_1P+b_3P^3$ without intercept, fitted by least squares to [[n]] draws at [[nlev]] levels, gave $b_1=[[b1]]$ (95\% HC3 interval [[cib1]], which excludes 0) and $b_3=[[b3]]$ [[cib3]], and the two-sided HC3 test of $b_3=0$ rejected at the 0.05 level ($p=[[pb3]]$), so the registered label is ``non-linear dose-response''.",
            "no dose-response": r"The dose-response model $E[I\mid P]=b_1P+b_3P^3$ without intercept, fitted by least squares to [[n]] draws at [[nlev]] levels, gave $b_1=[[b1]]$ with a 95\% HC3 interval [[cib1]] that contains 0, and $b_3=[[b3]]$ [[cib3]], so the registered label is ``no dose-response''.",
            "cut": r"The dose-response block was cut under the registered overrun rule (Appendix~\ref{app:power}), so it has no fit and no label."}},
    "FULL.dose.details": {
        "block": "Dose-response", "where": "new appendix, after FULL.dose.result", "length": "5 lines",
        "rule": "prereg v1.3 secs. 4.2 and 9.2: f_low, every candidate draw examined and the accepted draws; stratified bootstrap alongside; I_sym alongside; kappa descriptive with bootstrap CI; sensitivity fits (i) base-specific b1, (ii) without the A/A level; pipeline-asymmetry check in units of sigma_I, never confirmatory",
        "variants": {"fit": r"[[rec]] The stratified bootstrap, which resamples draws within level and base, gave [[bb1]] for $b_1$ and [[bb3]] for $b_3$, and with the symmetrized crossover the label was ``[[lsym]]''.[[lev]][[kap]][[sens]][[aa]]"},
        "parts": {"rec": r"With the registered pool fraction $f_{\mathrm{low}}=[[flow]]$, [[a1]] of [[e1]] candidate draws were accepted on base~1 and [[a2]] of [[e2]] on base~2 (Table~\ref{tab:fulldose}).",
                  "notrun": r" On [[bn]], fewer than two of the first 50 candidate seeds were accepted, so its dose level is not run under the registered rule.",
                  "lev": r" With three levels, the test of $b_3$ is a test of proportionality of the dose and C1 level means and does not distinguish quadratic from cubic curvature.",
                  "kap": r" The ratio of the slopes between adjacent level means, pooled over bases, was $\kappa=[[k]]$ (bootstrap 95\% interval [[kci]]).",
                  "sens": r" With a base-specific $b_1$ the fit gave $b_{1}=[[s1]]$ on base~1 and [[s2]] on base~2, and without the A/A level $b_1=[[s0]]$.",
                  "aa": r" The mean $I$ of the books A/A pairs, the intercept that the model excludes and 0 exactly under label-swap symmetry, was [[mI]], or [[mIs]] in units of $\hat\sigma_I$ ([[pv]]); it checks the pipeline and is not part of the dose-response model."}},
    "FULL.M12b.result": {
        "block": "M12-b", "where": "new appendix, paragraph 'M12-b'", "length": "3 lines",
        "rule": "prereg v1.3 sec. 4.5: 'Target-dependent' if all three Delta_d share a sign and the t-interval (df 2) excludes 0; otherwise 'no evidence of target dependence'; scope sentence; Delta_d per draw reported (sec. 9.2)",
        "variants": {
            "target-dependent": r"For M12-b, the three differences $\Delta_d=IT_d^{\mathrm{learned}}-IT_d^{\mathrm{rand}}$ shared one sign and their 95\% $t$-interval excluded 0 (mean $\Delta=[[mD]]$ [[ciD]], [[df]] degrees of freedom), so the registered label is ``target-dependent''.",
            "no evidence of target dependence": r"For M12-b, [[cond]] (mean $\Delta=[[mD]]$ [[ciD]], [[df]] degrees of freedom), so the registered label is ``no evidence of target dependence''.",
            "cut": r"M12-b was cut under the registered overrun rule (Appendix~\ref{app:power}), so it has no label."},
        "tail": r" The statistic removes per-module norm effects, not differences in the quality of the update direction; Table~\ref{tab:fullm12b} gives $\Delta_d$ per draw."},
    "FULL.C1DE.result": {
        "block": "C1-DE (exploratory)", "where": "new appendix, paragraph 'C1-DE'", "length": "7 lines",
        "rule": "prereg v1.3 sec. 4.12 (MF-6): descriptive label, not a test: 'sign-consistent with C1' if all six J_d > 0, 'sign-inconsistent' if all six J_d < 0, otherwise 'mixed'; p and intervals nominal under independence; module overlap reported; margin 0.25 not calibrated; never a replication of C1 on a second target; manifests, draws not generated, time-limit flags (sec. 9.2, R6); house-style rule 8: 'exploratory' in every sentence with a value",
        "variants": {
            "sign-consistent with C1": r"In C1-DE, an exploratory block, $J_d$ was positive in all [[n]] usable draws, so its descriptive label is ``sign-consistent with C1''; the label is not a test.",
            "sign-inconsistent": r"In C1-DE, an exploratory block, $J_d$ was negative in all [[n]] usable draws, so its descriptive label is ``sign-inconsistent''; the label is not a test.",
            "mixed": r"In C1-DE, an exploratory block, $J_d$ was positive in [[npos]] and negative in [[nneg]] of the [[n]] usable draws, so its descriptive label is ``mixed''; the label is not a test.",
            "cut": r"C1-DE, an exploratory block, was cut under the registered overrun rule (Appendix~\ref{app:power}), so it has no label."},
        "tail": r" Over the [[n]] draws of this exploratory block, mean $J$ was [[mJ]] (95\% $t$-interval [[ciJ]], [[df]] degrees of freedom; [[mJ1]] on base~1 and [[mJ2]] on base~2), the exact one-sided sign-flip $p$ was $[[p]]$, and regression through the origin gave $\hat\beta_{\mathrm{DE}}=[[bDE]]$ with the test-inversion interval [[ciDE]], $\hat\beta_{\mathrm{DE,pin}}=[[bpin]]$ and $\hat\beta_{\mathrm{DE,sym}}=[[bsym]]$, while the difference from C1 on books was $\hat\beta_{\mathrm{DE}}-\hat\beta=[[db]]$ [[cidb]] (Welch interval). These exploratory $p$-values and intervals are nominal under independence: two draws of a base share [[ov0]] to [[ov1]] of their modules, and the margin of 0.25 is not calibrated for these draws. The label is not a replication of C1 on a second target.[[tl]][[ng]] Table~\ref{tab:fulldraws} gives the values per draw."},
    "FULL.selpin.result": {
        "block": "C1 selection-pinned evaluations (selpin)", "where": "new appendix, paragraph 'Selection-only pinned replay'", "length": "4 lines",
        "rule": "prereg v1.3 sec. 4.6: descriptive; F - F^selpin router flips, F^selpin - F^pin gate-value pathway, F^pin frozen-routing pathway, each a share of F with t-intervals over draws, with I, I^selpin, I^pin; endpoint decomposition, the order of pathways matters; house-style rule 11: the shares are stated once, in the table",
        "variants": {"descriptive": r"The selection-only pinned replay splits the forgetting $F$ of every C1, third-base and C1-DE run into router flips, $F-F^{\mathrm{selpin}}$, the gate-value pathway, $F^{\mathrm{selpin}}-F^{\mathrm{pin}}$, and the frozen-routing expert pathway, $F^{\mathrm{pin}}$; Table~\ref{tab:fullselpin} gives their shares of $F$ as means over draws with 95\% $t$-intervals. It is an endpoint decomposition, and the order of the pathways matters."}},
    "FULL.C3.result": {
        "block": "C3 revert decomposition", "where": "new appendix, paragraph 'C3'", "length": "6 lines",
        "rule": "prereg v1.2 sec. 10.1 with v1.3 sec. 4.8: shares phi_g/F averaged over seeds within a domain, then over domains; t-interval from the pooled within-domain variance, MoE base 2 as points; interaction iota; 'The endpoint decomposition is not a path decomposition'; freeze cross-check; random partial-revert control not run (decision (e)); MoE-NESTOR comparison only at matched base loss, otherwise none; NESTOR exploratory if the routing-health label applies; house-style rule 11: the values are stated once, in the table",
        "variants": {"descriptive": r"For the endpoint revert decomposition of C3, Table~\ref{tab:fullc3} gives the shares $\varphi_g/F$ of the DCLM forgetting of the full-adaptation runs that each group carries, their interaction $\iota/F$, the single-group revert deltas and the freeze cross-check, as means over domains; MoE base~2 has one seed per domain, so its values are points. The endpoint decomposition is not a path decomposition. The random partial-revert control is not run under the frozen protocol, because no tool reverts a random parameter subset matched to a group in parameter count and displacement norm (Appendix~\ref{app:prereg}), so the shares carry no size calibration; no comparison between MoE and \nestor{} is made, because no MoE checkpoint at matched base loss is registered. [[nl]]"},
        "parts": {"nl_unchanged": r"C3 for \nestor{} keeps its base-specific secondary label, because the routing-health label of \nestor{} could not be evaluated (Appendix~\ref{app:prereg}).",
                  "nl_pass": r"C3 for \nestor{} keeps its base-specific secondary label.",
                  "nl_exploratory": r"Because the routing-health label applies, C3 for \nestor{} is reported as exploratory."}},
    "FULL.C3.wd": {
        "block": "C3 weight-decay-only evaluations", "where": "new appendix, after FULL.C3.result", "length": "2 lines",
        "rule": "descriptive weight-decay-only shares; amendment A-2 (Appendix A): a value of the three evaluations of G1 runs is reported only if its acceptance rule is met, otherwise withheld and recorded as a deviation; the fill stops if the record (CONFIG a2_acceptance_json) is missing or lacks a key",
        "variants": {"values": r"The weight-decay-only evaluations, which exist for the full-adaptation runs of MoE base~1 and \nestor{}, gave descriptive values of $F_{\mathrm{WD}}/F$ of [[list]].[[held]]"},
        "parts": {"held": r" The value of [[which]] is withheld, because the acceptance rule of amendment A-2 is not met for it (Appendix~\ref{app:prereg})."}},
    "FULL.C3.layer": {
        "block": "C3 single-layer reverts", "where": "new appendix, after FULL.C3.wd", "length": "3 lines",
        "rule": "prereg v1.3 sec. 4.8: Delta_l and Delta_l/F per layer, sum over layers against the all-group revert (additivity gap), domain means with the domain range; one seed per domain, no CI; descriptive; the registered figure is fig:c3layers",
        "variants": {"descriptive": r"For the single-layer reverts, with one full-adaptation run per domain and therefore no interval, the sum of the per-layer deltas $\Delta_\ell$ differed from the revert of the whole group by [[g1]] on MoE base~1 (range over domains [[r1]]) and by [[gn]] for the column-private blocks of \nestor{} (range [[rn]]); Table~\ref{tab:fullc3layer} gives $\Delta_\ell/F$ per layer and block, and Figure~\ref{fig:c3layers} shows them."}},
    "FULL.Frel.result": {
        "block": "C3 additions: relative forgetting (DCLM continuation runs)", "where": "new appendix, after FULL.C3.layer", "length": "3 lines",
        "rule": "prereg v1.2 l. 327 and sec. 9 item 13 (secondary): F_rel = F - F_DCLM-CONT; C1 version reported; C2 version needs 96-step runs and is not computable (P-16)",
        "variants": {"C1": r"Measured against the DCLM continuation runs, $F_{\mathrm{rel}}=F-F_{\mathrm{cont}}$, the C1 statistic over the [[n]] draws gave mean $J=[[mJ]]$ (95\% $t$-interval [[ciJ]]) and a one-sided sign-flip $p=[[p]]$ (secondary, unadjusted). The C2 version needs C2 runs of 96 steps and is not computed, because the C2 runs stop after step 24 under SGD and after step 8 under Muon."}},
    "FULL.C5.result": {
        "block": "C5 French forecasts", "where": "new appendix, paragraph 'C5 and the forecasters'", "length": "4 lines",
        "rule": "prereg v1.2 secs. 10.3 and 12: descriptive, no hypothesis test; any forecast changed after hashing is reported as non-prospective; French metrics with scalars frozen from the non-French runs; deviation record v19, D-4 section 7 (c): when no forecaster has a pooled score and every forecast in the hashed file is null, block C5 is not evaluable as registered and the variant not_evaluable replaces the label variants",
        "variants": {"prospective": r"The French forecasts meet the registered hashing rule, since their hash precedes the first French run and the file matches it, so they are ``prospective''. [[met]] [[r2w]]",
                     "non-prospective": r"The French forecasts do not meet the registered hashing rule, because [[reasons]], so they are ``non-prospective''. [[met]] [[r2w]]",
                     "not_evaluable": r"No forecast was formed for the full-adaptation French run of MoE base~1, because the forecasting code did not recognise the run's role as a full-adaptation run, so block C5 is not evaluable as registered; a forecast formed after hashing would be non-prospective under the registered rule, so none is reported (deviation D-4, Appendix~\ref{app:prereg})."},
        "parts": {"met": r"On the French run, the pooled $R^2_{\mathrm{oos}}$ against the training-fold benchmark was [[list]], over [[n]] corpus values each.",
                  "r2w_none": r"$R^2_{\mathrm{within}}$ is not defined, because there are fewer than three French crossover pairs."}},
    "FULL.forecasts.result": {
        "block": "Scores of the forecasters", "where": "new appendix, after FULL.C5.result", "length": "2 lines",
        "rule": "prereg v1.3 sec. 5.7: all eight forecasters in one specification table, pooled and per block, no selection of a best forecaster; rows after a run's stop step are not scored (P-16)",
        "variants": {"scores": r"Table~\ref{tab:fullforecasts} gives the scores of all eight registered forecasters, pooled and per block, in one specification table without selecting a best forecaster; rows of C2 runs at steps after their stop step are not scored."}},
    "FULL.compute": {
        "block": "Budget: planned against measured GPU-hours; overrun guards; measured total", "where": "Appendix C, after the COMPUTE.used fill", "length": "4 lines",
        "rule": "prereg v1.3 secs. 7 (R10) and 9.2: planned against actual GPU-h per block, every cut, restoration and envelope extension, and the projections of the three overrun guards against the envelope; compute numbers appear in Appendix C only, the hours per block once, in the table; operator (20 November): one sentence gives the measured total at one decimal, so that it agrees with the printed components at any rounding, the hours spent at the freeze (analysis parameters) and outside the queue state (data/hours_outside_state.json) added to the hours of the queue state, without and with the exploratory hours (data/exploratory_values.json), against the envelope",
        "variants": {"totals": r"Table~\ref{tab:fullbudget} gives the planned and measured GPU-hours of every block of the registered queue and their totals; [[gd]].[[ext]][[tot]]"},
        "parts": {"none_cut": r"the three overrun guards, before M12-b, the dose-response and C1-DE, projected at most [[pmax]] GPU-hours against the envelope of [[env]] GPU-hours",
                  "cut": r"the overrun guard before [[blk]] projected [[proj]] GPU-hours against the envelope of [[env]] GPU-hours",
                  "tot": r" With the GPU-hours spent at the freeze and outside the queue state (Table~\ref{tab:budget}), the measured compute came to [[t1]] GPU-hours, or [[t2]] with the exploratory hours (Appendix~\ref{app:exploratory}), below the envelope of [[envi]]."}},
    "FULL.EDGE.all": {
        "block": "Cost block (descriptive): all covered runs", "where": "Appendix F, paragraph 'All covered runs.' after the paragraph 'Cost measurements.'", "length": "11 lines",
        "rule": "prereg v1.3 sec. 4.11 (descriptive; no hypothesis; outside the Holm family): medians with [min, max] of wall time and GPU energy per run type over G1, G2, the forecast drive runs and every post-freeze run, the excluded runs named with the reason, and the pre-update check per architecture as the window of a probe batch divided by its units, against the median subset adaptation run; final check of v1.6.10 (M1): wording as approved, every number from blocks.EDGE (by_type, excluded, runs, pre_update_check), counts below 10 in words; the paragraph stays pending if the analysis output no longer fits the wording (other stages or run types, another number of excluded runs, another NESTOR reference, C2 runs other than those of Table tab:cost)",
        "variants": {"all": r"Over the gate runs of G1 and G2, the forecast drive runs and the runs after the freeze, the cost block covers [[nrun]] runs and excludes one, the [[exname]], in whose window two consecutive power samples were [[gap]]\,s apart. With the MoE bases pooled, the [[ns]] MoE subset runs took a median of [[msh]]\,h [[mshci]] and [[mse]]\,Wh [[mseci]], where the lower ends belong to the [[n8]] runs of [[s8]] steps; the [[nf]] MoE full adaptations took [[mfh]]\,h [[mfhci]] and [[mfe]]\,Wh [[mfeci]], the [[nnf]] \nestor{} full adaptations [[nfh]]\,h [[nfhci]] and [[nfe]]\,Wh [[nfeci]], and the [[nns]] \nestor{} subset runs, [[nns8]] of them of [[s8]] steps, [[nsh]]\,h [[nshci]] and [[nse]]\,Wh [[nseci]]. The MoE freeze run of C3 took [[fzh]]\,h and [[fze]]\,Wh, and the C2 runs are those of Table~\ref{tab:cost}. Measured as the window of a probe batch divided by its number of units, one probe unit of the pre-update check took [[pms]]\,s and [[pme]]\,Wh on MoE base~1, which is [[pmrw]] of the wall time and [[pmre]] of the energy of the median MoE subset adaptation run, and [[pns]]\,s and [[pne]]\,Wh on \nestor{}, [[pnrw]] and [[pnre]] of those of the one \nestor{} subset adaptation run that serves as reference."}},
    "FULL.guards": {
        "block": "Overrun guards (outcome)", "where": "new appendix, end of its opening paragraph", "length": "1 line",
        "rule": "prereg v1.3 sec. 7 (R10): before each cuttable block (M12-b, dose-response, C1-DE) measured spent plus remaining planned hours are compared with the envelope, and above it the registered cut order applies; the sentence names the outcome only and points to Appendix C, which holds the hours",
        "variants": {"none_cut": r"None of the three overrun guards, before M12-b, the dose-response and C1-DE, cut a block (Appendix~\ref{app:power}).",
                     "cut": r"The overrun guard before [[blk]] cut [[cut]] under the registered cut order, so [[pron]] not run (Appendix~\ref{app:power})."}},
    "FULL.A2": {
        "block": "Amendment A-2: acceptance of the three values", "where": "Appendix A, replaces the last sentence of the paragraph on amendment A-2", "length": "2 lines",
        "rule": "A-2 acceptance rule (Appendix A): a value is reported only if the log of its evaluation records the 96 executed steps from the training queue and learning rates equal to the training log at all 96 steps; otherwise withheld and recorded as a deviation; the fill stops if the record (CONFIG a2_acceptance_json) is missing or lacks a key",
        "variants": {"met": r"The acceptance rule is met for all three evaluations, so their values are reported with block C3 (Appendix~\ref{app:full}).",
                     "partial": r"The acceptance rule is not met for the evaluation of [[which]], so [[pron]] withheld and recorded as a deviation; the other values are reported with block C3 (Appendix~\ref{app:full}).",
                     "none": r"The acceptance rule is not met for any of the three evaluations, so their values are withheld and recorded as a deviation."}},
    "FULL.dev.D2": {
        "block": "Deviation D-2, amendment A-3: assumption audit", "where": "Appendix A, paragraph 'Deviation D-2 and amendment A-3.' after R-1", "length": "10 lines in Appendix A on the mock build with A-3 done; about 1 line more when one base has no outputs, half a line more when neither has",
        "rule": "house-style rules 5 and 6 (a deviation in four parts: what was registered, what was done instead with its reason in one clause, the effect and the compute cost, with no sequence of events); outcome per base from the A-3 record (a completed run with the three registered outputs and the wrapper line in its log), hours of the stopped starts from data/hours_outside_state.json and of the audit runs under A-3 from the record; the fill stops if the record is missing or malformed",
        "variants": {"done": D2_HEAD + r"On both bases the audit wrote its registered outputs (Appendix~\ref{app:full}), and it enters no confirmatory statistic." + D2_COST,
                     "partial": D2_HEAD + r"On [[ok]] the audit wrote its registered outputs (Appendix~\ref{app:full}); on [[bad]] it wrote none, and that base is reported as missing. The audit enters no confirmatory statistic." + D2_COST,
                     "none": D2_HEAD + r"On neither base did the audit write its registered outputs, so both bases are reported as missing; the audit enters no confirmatory statistic." + D2_COST}},
    "FULL.dev.D4": {
        "block": "Deviation D-4: C5 French forecasts not formed", "where": "Appendix A, paragraph 'Deviation D-4.' after D-3", "length": "about 10 lines in Appendix A",
        "rule": "deviation record v19, D-4 section 7 (a), unchanged in substance (word references as \\ref): house-style rule 5 in four parts, what was registered, what was done instead with its reason, the effect and the compute cost; printed only when FULL.C5.result is not evaluable as registered",
        "variants": {"recorded": r"The pre-registration forms, before the full-adaptation French run of MoE base~1 with seed 101, the forecasts of every registered forecaster for that run's forgetting on DCLM, code and mathematics, with scalars fitted on all non-French runs available at that time; it hashes the forecast file, posts the hash before the run starts and then compares the forecasts with the measured forgetting, as a descriptive block without a test. No forecast was formed, because the forecasting code did not recognise the run's role as a full-adaptation run, and the same code fitted the scalars on the full-adaptation runs of block C3 alone, because it read the subset description of the other runs under a key that the plan does not use. The forecast file was hashed and its hash posted before the French run as registered, but the file holds no forecast, so block C5 is not evaluable as registered; no forecast was formed after hashing and none is reported, since the registration reports any forecast changed after hashing as non-prospective, and no confirmatory statistic depends on C5. The French run and its eight-step drive run were run and evaluated as planned, so the deviation added no GPU-hours to the planned runs of the block (Appendix~\ref{app:power})."}},
    "FULL.dev.D3": {
        "block": "Deviation D-3, amendment A-4: bitwise audit of the two C3 freeze runs", "where": "Appendix A, paragraph 'Deviation D-3 and amendment A-4.' after D-2", "length": "12 lines in Appendix A on the mock build with A-4 met; about 1 line more when one run is withheld, the same when both are",
        "rule": "house-style rules 5 and 6 (a deviation in four parts, with no sequence of events); A-4 acceptance rule: a repeated run is accepted only if no frozen tensor, buffer or optimizer state changed, exactly the named parameter is unchanged and the run is the same job as its first attempt, otherwise its values are withheld and recorded as a deviation; acceptance per run from the A-4 acceptance record, hours of the first attempts from data/hours_outside_state.json and of the repeated runs from the queue state; the fill stops if the record is missing or malformed",
        "variants": {"met": D3_HEAD + r"Both runs met the acceptance rule, so their values are reported with block C3 (Appendix~\ref{app:full})." + D3_TAIL,
                     "partial": D3_HEAD + r"[[Ok]] met the acceptance rule, so its values are reported with block C3 (Appendix~\ref{app:full}); the values of [[bad]] are withheld under that rule and recorded as a deviation." + D3_TAIL,
                     "none": D3_HEAD + r"Neither run met the acceptance rule, so their values are withheld and recorded as a deviation." + D3_TAIL}},
    "FULL.dev.I4": {
        "block": "Incident I-4: interrupted dose-response run", "where": "Appendix A, paragraph 'Incident I-4.' after D-3", "length": "7 lines in Appendix A on the mock build",
        "rule": "as incident I-1: the registered recovery rule repeats an interrupted run from its start with the same seeds and settings and keeps the unfinished output; only the repetition's values are registered; hours from data/hours_outside_state.json",
        "variants": {"repeated": r"A restart of the software that launches the runs ended one dose-response run during training (MoE base~1, books, draw 3, subset $A$). The run was repeated from its start with the same seeds and settings under the registered recovery rule, which keeps the unfinished output and deletes nothing; the environment files of the two attempts are identical. The interrupted attempt wrote no checkpoint, interim value or evaluation, so no value comes from it, and its job log was not kept. It used [[hi4]] GPU-hours outside the queue state (Appendix~\ref{app:power})."}},
    "FULL.dev.I5I6": {
        "block": "Incidents I-5 and I-6: two interrupted attempts of one C1-DE run", "where": "Appendix A, paragraph 'Incidents I-5 and I-6.' after I-4", "length": "about 11 lines in Appendix A (estimate from the text, not built)",
        "rule": "as incidents I-1 and I-4: the registered recovery rule repeats an interrupted run from its start with the same seeds and settings and keeps the unfinished output; only the completed attempt's values are registered; one plain clause for the reading of training-loss lines outside the registered reading rule (closing addendum of I-5 and I-6); hours from data/hours_outside_state.json",
        "variants": {"repeated": r"Two attempts of one C1-DE run (MoE base~1, German, draw 2, subset $B$) ended during their evaluations, after training had finished: the first through a restart of the software that launches the runs, the second through a restart of the computer, whose cause is not established. Under the registered recovery rule, which keeps the unfinished output and deletes nothing, the run was repeated from its start with the same seeds and settings each time, and its third attempt completed; the environment and subset-manifest files of the three attempts are byte-identical. No value comes from the interrupted attempts, and their job logs were kept. To check the repetition, training-loss lines of this run were read before the queue ended, which the registered reading rule does not provide for; no evaluation value was read, and no statistic was computed from them. The first interrupted attempt used [[hi5]] GPU-hours and the second [[hi6]] GPU-hours, both outside the queue state (Appendix~\ref{app:power})."}},
    "FULL.dev.I7": {
        "block": "Incident I-7: interrupted run of the assumption audit under amendment A-3", "where": "Appendix A, paragraph 'Incident I-7.' after I-5 and I-6", "length": "about 6 lines in Appendix A (estimate from the text, not built)",
        "rule": "as incidents I-1 and I-4: the registered recovery rule repeats an interrupted run from its start with the same call and settings; only the completed run's outputs are registered; the start by hand is stated in one plain clause; hours from data/hours_outside_state.json",
        "variants": {"repeated": r"A restart of the computer, whose cause is not established, ended the first run of the assumption audit on MoE base~1 under amendment A-3 before it had written any output. The run was repeated from its start with the same call, environment and checks under the registered recovery rule; because the automatic procedure of the amendment could not resume an interrupted job, the repetition was started by hand. No value comes from the interrupted run, and its job log was kept. It used [[hi7]] GPU-hours outside the queue state (Appendix~\ref{app:power})."}},
    "FULL.hos": {
        "block": "Hours outside the planned runs (Appendix C sentence)", "where": "Appendix C, replaces the sentence on the hours outside the queue state (the old sentence stays as the default of the slot)", "length": "about 12 lines in Appendix C, 5 more than the old sentence it replaces (estimate; 10 lines on the mock build before incidents I-5 to I-7)",
        "rule": "MF-A1-2 hand rule of addendum 1 (hours outside the queue state are added to every guard projection); each number stated once: Appendix A states each item with its hours (cost part of rule 5) and the last row of Table tab:budget only their total (core slot HOS.total, filled by scripts/fill_core.py from data/hours_outside_state.json), so the sentence names the items without numbers",
        "variants": {"list": r"After the freeze, [[items]] spent GPU-hours outside the queue state, which Appendix~\ref{app:prereg} states item by item and the last row of Table~\ref{tab:budget} in total. The overrun guards add such hours to their projections, and where a corrected projection exceeds the envelope at a guard that passed, the registered cut order is applied by hand and logged; the guards counted the stopped starts of deviation D-2 and the first attempts of deviation D-3 as spent, because these left the queue state only when their repetitions, which ran after the last guard, replaced them. The hours of incidents I-5, I-6 and I-7 arose after the last guard."}},
    "FIG.c3layers.caption": {
        "block": "C3 single-layer figure (caption)", "where": "new appendix, caption of Figure fig:c3layers (figures/fig_c3_layers.pdf)", "length": "9 lines",
        "rule": "prereg v1.3 secs. 4.8 and 9.2 (figure reported regardless, descriptive, no interval with one seed per domain); v1.2 sec. 10.1 (no comparison between architectures unless at matched base loss; NESTOR exploratory if the routing-health label applies); v1.2 sec. 13 (missing runs counted); caption draft",
        "variants": {"figure": "C3 single-layer reverts (descriptive). For one full-adaptation run per domain and architecture (seed 101; domains [[DOMAINS]]; [[N_RUNS_B1]] runs of MoE base~1 and [[N_RUNS_NESTOR]] of \\nestor{}), the routed experts of one layer of MoE base~1, or the column-private group (column attention and private experts) of one \\nestor{} block, were set back to $\\theta_0$, and the run was evaluated on DCLM (2.048M tokens). (a,~c)~Share of the run's DCLM forgetting $F$ removed by the revert of unit~$\\ell$, $\\Delta_\\ell/F$ with $\\Delta_\\ell=F-F(\\text{revert }\\ell)$: mean over domains (marker) and range over domains (bar); units are numbered from 1: MoE layers 1 to 12 in block-then-layer order, \\nestor{} blocks 1 to 6 as in Figure~\\ref{fig:arch} (blocks 0--5 of the pre-registration). (b,~d)~Per domain, the sum of the single-unit deltas divided by the delta of reverting the whole group at once, $\\Sigma_\\ell\\Delta_\\ell/\\Delta_{\\mathrm{all}}$; the dotted line marks equality. With one run per domain no interval is shown. The two architectures are drawn on separate axes and are not compared (comparisons only at matched base loss; pre-registration v1.2, Section~10.1).[[OPTIONAL_C3]]"},
        "parts": {"exploratory": r" C3 for \nestor{} is exploratory (routing-health label; pre-registration v1.2, Section~10.1).",
                  "fewer": r" Open marker: mean over fewer domains, where a single-unit evaluation is missing ([[units]]).",
                  "dropped": r" [[k]] planned runs are not shown under the missing-value rule of pre-registration v1.2, Section~13."}},
    "FIG.c4windows.caption": {
        "block": "C4 per-window figure (caption)", "where": "main text, Section 5.5, caption of Figure fig:c4windows (figures/fig_c4_windows.pdf)", "length": "8 lines",
        "rule": "prereg v1.3 sec. 4.8 (main-text figure of the per-window results of v1.2 sec. 10.2, no new computation) and sec. 9.2; v1.2 sec. 10.2 (estimand, t-interval and two-way bootstrap over draws and windows, within-draw correlation); v1.2 sec. 13 (dropped draws counted); caption draft",
        "variants": {"figure": 'Per-window difference-in-differences within DCLM (C4; secondary, descriptive). For each of the [[N_DRAWS]] C1 draws ([[N_B1]] of MoE base~1, [[N_B2]] of MoE base~2) and each of the [[W]] DCLM evaluation windows~$w$, $\\Delta F_w=F_w(B_d)-F_w(A_d)$ and $\\Delta R_w=\\log\\RO_w(B_d)-\\log\\RO_w(A_d)$ at $\\theta_0$; $b_d$ is the least-squares slope of $\\Delta F_w$ on $\\Delta R_w$ over the windows and $\\beta_{\\mathrm{C4},d}=b_d/\\bar F_d$, with $\\bar F_d$ the mean DCLM forgetting of the runs on $A_d$ and $B_d$. (a)~$\\beta_{\\mathrm{C4},d}$ per draw (open markers, in draw order) and its mean over draws per base and over all draws (filled markers), with the 95\\% t-interval over draws (thick bar) and the 95\\% two-way bootstrap interval over draws and windows ($B=[[B]]$; thin bar). (b)~The within-draw correlation of $\\Delta F_w$ and $\\Delta R_w$ per draw and its mean over draws. MoE base~2 is drawn in a lighter blue with dashed intervals. [[DROPPED_C4]]'},
        "parts": {"none": "No draw was dropped.", "dropped": r"[[k]] draws were dropped under the registered missing-value rule (Appendix~\ref{app:prereg}).",
                  "noboot": r" For [[name]] the two-way bootstrap interval is not defined."}},
    "FULL.audit.C1": {
        "block": "Assumption-audit labels reported with C1 (main text)", "where": "Section 5.3, end of the C1 results paragraph (after C1.result)", "length": "about 4 lines",
        "rule": "prereg v1.3 sec. 4.3: the labels are reported with C1 and no decision depends on them; one sentence with the labels per base (B5; B6 for subsets A_1 and B_1; A0 per corpus) from blocks.audit, a pointer to Table tab:fullaudit and the statement that the audit enters no confirmatory statistic; labels verbatim as registered, the analysis string 'A0 violated' printed as 'A0 does not hold approximately'",
        "variants": {"labels": r"The registered assumption audit at $\theta_0$, which enters no confirmatory statistic, labelled [[b5]], and [[b6]]; [[a0]] (Table~\ref{tab:fullaudit}).",
                     "one": r"The registered assumption audit at $\theta_0$, which enters no confirmatory statistic, wrote outputs only on [[ok]]; there it labelled [[b5]], and [[b6]]; [[a0]] (Table~\ref{tab:fullaudit}).",
                     "none": r"The registered assumption audit at $\theta_0$, which enters no confirmatory statistic, wrote no outputs on either base (Appendix~\ref{app:prereg})."}},
    "FULL.audit": {
        "block": "Assumption audit at theta_0 (B5, B6, A0)", "where": "new appendix, paragraph 'Assumption audit', with the generated Table tab:fullaudit", "length": "about 10 lines in the new appendix on the mock build with both bases and the same labels, about 11 with one base missing, about 12 when the labels differ by base and about 2 with no outputs on either base; Table tab:fullaudit about 27 lines (about 22 with one base, none without outputs); estimate from the text, not built",
        "rule": "prereg v1.3 sec. 4.3 (secondary; the labels are reported with C1, no decision depends on them, and the audit enters no confirmatory statistic); labels verbatim as registered, except that the analysis string 'A0 violated', which the pre-registration does not name, is printed as 'A0 does not hold approximately' (the complement of the registered label 'A0 approximately holds'): registered quantities and labels only, no interpretation beyond the registered one. B5: interaction share of the two-way decomposition log Q_c(g) = a_g + b_c + e_gc per base after subtracting the expected noise contribution, with its 95% window-bootstrap interval (2,000 resamples); cells with Q <= 0 counted, reported and excluded; labels below 0.1 'approximately separable', 0.1 to 0.3 'moderately non-separable', above 0.3 'non-separable'. B6, per subset: CV of exp(a_g) over its 12 groups and Spearman rho between a_g and the group's share of P with its bootstrap interval; 'homogeneous' if CV < 0.25, 'heterogeneous, uncorrelated' if CV >= 0.25 and the interval of rho contains 0, otherwise 'heterogeneous, correlated', which the pre-registration names as the case that biases P; Jensen gap reported. A0: median of r = |gamma_c(g)|/Q_c(g) and share of cells with r >= 0.2 per base and corpus; 'A0 approximately holds' if that share is at most 0.25. Each value stated once, in Table tab:fullaudit; the bootstrap interval of the CV that the frozen code also writes is not a registered quantity and is not printed. The bases follow the A-3 record, and a base without outputs is reported as missing; the fill stops on a structure other than the registered one, on a disagreement with the A-3 record and on a label that differs from the registered rule applied to its numbers",
        "variants": {"both": r"The assumption audit (Section~\ref{sec:secondary}) evaluated pinned probes at $\theta_0$ on MoE bases~1 and~2" + AUD_SCOPE + r"; it enters no confirmatory statistic, and no decision depends on its labels." + AUD_QUANT,
                     "one": r"The assumption audit (Section~\ref{sec:secondary}) evaluated pinned probes at $\theta_0$ on [[ok]]" + AUD_SCOPE + r"; on [[bad]] it wrote no outputs, and that base is reported as missing (Appendix~\ref{app:prereg}). The audit enters no confirmatory statistic, and no decision depends on its labels." + AUD_QUANT,
                     "none": r"The assumption audit (Section~\ref{sec:secondary}) wrote no outputs on either base, so both bases are reported as missing (Appendix~\ref{app:prereg}) and it has no registered quantities or labels; it enters no confirmatory statistic."},
        "parts": {"per": r" per base",
                  "labels_both": r" On both bases the registered labels were [[set]].",
                  "labels_base": r" On [[name]] the registered labels were [[set]].",
                  "labels_base2": r" On [[name]] they were [[set]].",
                  "set_join": r"[[b5]]; [[b6]]; and [[a0]]",
                  "set_B5": r"``[[l5]]'' for B5",
                  "set_B6_same": r"``[[l6]]'' for B6 in both subsets",
                  "set_B6_diff": r"``[[l6a]]'' for B6 in subset $A_1$ and ``[[l6b]]'' in $B_1$",
                  "set_A0_same": r"``[[l0]]'' on all three corpora",
                  "set_A0_group": r"``[[l0]]'' on [[corp]]",
                  "set_A0_same_c": r"A0 does not hold approximately on any of the three corpora",
                  "set_A0_group_c": r"A0 does not hold approximately on [[corp]]",
                  "corr": r" The pre-registration identifies ``heterogeneous, correlated'' as the case that biases $P$."}},
}
TABLE_IDS = {"TAB.fulldraws": "draws", "TAB.fulldose": "dose", "TAB.fullm12b": "m12b", "TAB.fullselpin": "selpin",
             "TAB.fullc3": "c3", "TAB.fullc3layer": "c3layer", "TAB.fullforecasts": "forecasts", "TAB.fullbudget": "budget", "TAB.fullaudit": "audit"}
# Table 8 cells of the post-freeze items: generated in tables/block_status.tex (not fills) and listed in the template catalogue
TAB8 = {"C5.not_evaluable": r"not evaluable as registered, because no forecast was formed for the French run (deviation D-4, Appendix~\ref{app:prereg})",
        "audit.done": r"run after the other blocks under amendment A-3 (Appendix~\ref{app:prereg}); results in Appendix~\ref{app:full}",
        "audit.partial": r"run after the other blocks under amendment A-3 (Appendix~\ref{app:prereg}); results for [[ok]] in Appendix~\ref{app:full}, none for [[bad]]",
        "audit.none": r"run after the other blocks under amendment A-3, without results on either base (Appendix~\ref{app:prereg})",
        "C3.met": r"; the two freeze runs ran under amendment A-4 (Appendix~\ref{app:prereg})",
        "C3.partial": r"; the two freeze runs ran under amendment A-4, and the values of [[bad]] are withheld under its acceptance rule (Appendix~\ref{app:prereg})",
        "C3.none": r"; the two freeze runs ran under amendment A-4, and their values are withheld under its acceptance rule (Appendix~\ref{app:prereg})",
        "dose.I4": r"; one run was repeated from its start under the registered recovery rule (incident I-4, Appendix~\ref{app:prereg})",
        "C1DE.I5I6": r"; one run was repeated twice from its start under the registered recovery rule (incidents I-5 and I-6, Appendix~\ref{app:prereg})",
        "audit.I7": r"; its first run on MoE base~1 under amendment A-3 was repeated from its start under the registered recovery rule (incident I-7, Appendix~\ref{app:prereg})"}
# reasons of the C5 hash check (analyze_confirm_ext.block_c5_french) in plain words (house-style rules 2 and 9)
C5_REASONS = [(r"no longer matches its sha256", "a forecast changed after hashing"),
              (r"hash is not in", "the hash was not posted"),
              (r"posted after the first French run", "the hash was posted after the first French run started"),
              (r"forecast file lists", "the forecast file and the plan list different French runs")]


def subst(tpl, vals):
    def rep(m):
        k = m.group(1)
        if k not in vals:
            raise KeyError(f"template slot [[{k}]] has no value")
        return vals[k]
    return re.sub(r"\[\[(\w+)\]\]", rep, tpl)


def math_slots(tpl):
    """names of the [[slots]] that stand inside $...$ in a template"""
    out, inside = set(), False
    for part in re.split(r"(\$)", tpl):
        if part == "$":
            inside = not inside
        elif inside:
            out |= set(re.findall(r"\[\[(\w+)\]\]", part))
    return out


def texstr(s):
    """an analysis string as TeX text: base ids by their names, TeX-active characters escaped"""
    s = str(s)
    for b in CONFIG["bases"]:
        s = s.replace(b, "\x02" + b + "\x03")
    esc = {"\\": r"\textbackslash{}", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "$": r"\$"}
    s = re.sub(r"[\\%&#_{}~^$]", lambda m: esc[m.group(0)], s)
    return re.sub("\x02(\\w+)\x03", lambda m: CONFIG["bases"].get(m.group(1), m.group(1)), s)


def fmt_int(v):
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def mnum(v, nd=3):
    """a number for math mode (minus sign as '-')"""
    if isinstance(v, float) and math.isinf(v):
        return "-\\infty" if v < 0 else "\\infty"
    return f"{float(v):.{nd}f}"


def words_join(items, conj="and"):
    items = [x for x in items if x]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


# path segments: a key, a list index [i], or a quoted key ["k"] for a key that contains a dot (A0 share_r_ge_0.2 of the audit)
PATH_RE = re.compile(r'\["[^"]*"\]|[^.\[\]]+|\[\d+\]')


def _is_index(part):
    return part.startswith("[") and not part.startswith('["')


def getp(doc, path):
    """FC.get with quoted keys: the value at path, or None"""
    obj = doc
    for part in PATH_RE.findall(path):
        if _is_index(part):
            idx = int(part[1:-1])
            if not isinstance(obj, list) or idx >= len(obj):
                return None
            obj = obj[idx]
        else:
            key = part[2:-2] if part.startswith('["') else part
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
    return obj


def put_path(root, path, value):
    parts = PATH_RE.findall(path)
    obj = root
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        nxt = parts[k + 1] if not last else None
        if _is_index(part):
            idx = int(part[1:-1])
            while len(obj) <= idx:
                obj.append(None)
            if last:
                obj[idx] = value
            else:
                if obj[idx] is None:
                    obj[idx] = [] if _is_index(nxt) else {}
                obj = obj[idx]
        else:
            key = part[2:-2] if part.startswith('["') else part
            if last:
                obj[key] = value
            else:
                if key not in obj or obj[key] is None:
                    obj[key] = [] if _is_index(nxt) else {}
                obj = obj[key]


# ----------------------------------------------------------------------------- the fill
class Filler:
    def __init__(self, doc, state, a2, BL):
        self.doc, self.state, self.a2, self.BL = doc, state or {}, a2, BL
        self.fills, self.pending, self.warn = {}, {}, []
        self.data, self.derived, self.nums, self.fid = {}, {}, {}, None
        self.tables, self.status, self.labels, self.cut, self.guard_info = {}, {}, {}, {}, {}

    # -- values: every printed number is recorded with its field (ledger)
    def toks(self, s):
        return [m.group(0) for m in self.BL.TOKEN.finditer(self.BL.visible(s))]

    def _rec(self, s, field, v):
        for t in self.toks(s if not s.startswith("$") else s):
            self.nums.setdefault(self.fid, []).append({"token": t, "field": field, "value": v})

    def val(self, path):
        v = FC.get(self.doc, path)
        if isinstance(v, str) and v in ("inf", "-inf"):
            v = float(v)
        if not (FC.isnum(v) or (isinstance(v, float) and math.isinf(v))):
            raise Pending(f"{path} = {v!r}")
        self.data[path] = v
        return v

    def n(self, path, nd=3, m=False):
        v = self.val(path)
        s = mnum(v, nd) if m else FC.num(v, nd)
        self._rec(s, path, v)
        return s

    def i(self, path):
        v = self.val(path)
        s = fmt_int(v)
        self._rec(s, path, v)
        return s

    def p(self, path):
        v = self.val(path)
        s = FC.pfmt(v)
        self._rec(s, path, v)
        return s

    def ci(self, lo, hi, nd=3):
        return f"[{self.n(lo, nd)}, {self.n(hi, nd)}]"

    def dv(self, key, value, how, nd=3, kind="num", m=False):
        path = f"derived.{key}"
        self.data[path] = value
        self.derived[path] = how
        s = fmt_int(value) if kind == "int" else (mnum(value, nd) if m else FC.num(value, nd))
        self._rec(s, path, value)
        return s

    def blk(self, name):
        b = (self.doc.get("blocks") or {}).get(name)
        if not isinstance(b, dict):
            raise Pending(f"blocks.{name} missing")
        if str(b.get("status", "")).startswith(("failed", "not computable", "not computed")):
            raise Pending(f"blocks.{name}: {b['status']}")
        return b

    def put(self, fid, text, fields, variant=None, label=None):
        lim = CONFIG["word_limits"].get(fid)
        if lim is not None and len(FC.words(text)) > lim:
            raise SystemExit(f"[fill_full] {fid}: {len(FC.words(text))} words > limit {lim}: {text}")
        self.fills[fid] = {"text": text, "fields": fields, "variant": variant, "label": label, "rule": T.get(fid, {}).get("rule"),
                           "numbers": self.nums.get(fid, [])}

    def skip(self, fid, why):
        self.pending[fid] = why
        self.nums.pop(fid, None)
        self.fills.pop(fid, None)

    def run(self, fid, fn):
        self.fid = fid
        self.nums[fid] = []
        try:
            fn()
        except Pending as e:
            self.skip(fid, f"{e}")

    # -- guards (confirm queue state)
    def guards(self):
        for blk, gid in CONFIG["guards"].items():
            s = self.state.get(gid)
            if not isinstance(s, dict):
                self.guard_info[blk] = {"gate": gid, "status": "missing", "detail": None, "failed": False}
                continue
            det = s.get("detail") if isinstance(s.get("detail"), dict) else None
            failed = s.get("status") in ("fail", "failed") or (det is not None and det.get("pass") is False)
            self.guard_info[blk] = {"gate": gid, "status": s.get("status"), "detail": det, "failed": failed}
            if failed:
                for b in (det or {}).get("cut_set") or [blk]:
                    self.cut[b] = gid

    def check(self, block, printed, expected):
        if printed != expected:
            raise SystemExit(f"[fill_full] {block}: the analysis label {printed!r} differs from the registered rule applied to its numbers ({expected!r}); nothing written")

    def fill(self, fid, variant, vals, fields, label=None, tail=True):
        tpl = T[fid]["variants"][variant] + (T[fid].get("tail", "") if tail else "")
        self.put(fid, subst(tpl, vals), fields, variant, label)

    # ------------------------------------------------------------------ blocks
    def b3(self):
        fid = "FULL.B3.result"; base = "blocks.B3"; b = self.blk("B3"); lab = b.get("label")
        if lab == "no usable draws":
            self.fill(fid, "no usable draws", {}, [f"{base}.label", f"{base}.E1"], lab, tail=False); self.labels["B3"] = None; return
        if lab not in ("consistent with C1", "inconsistent", "inconclusive"):
            raise Pending(f"{base}.label = {lab!r}")
        J = [r.get("J") for r in b.get("per_draw") or []]
        hi = (b.get("mean_J") or {}).get("hi")
        self.check("third base", lab, "consistent with C1" if (len(J) >= 3 and all(FC.isnum(j) and j > 0 for j in J)) else ("inconsistent" if FC.isnum(hi) and hi < 0 else "inconclusive"))
        bounded = ((b.get("estimators") or {}).get("raw") or {}).get("overall", {}).get("bounded")
        e1 = ""
        if b.get("draws_dropped"):
            e1 = (f" Under rule E1, {self.dv('B3.dropped', len(b['draws_dropped']), 'len(blocks.B3.draws_dropped)', kind='int')} of the "
                  f"{self.i(f'{base}.draws_planned')} draws were dropped.")
        v = {"b0": self.n(f"{base}.beta0_3", 2, m=True), "n": self.i(f"{base}.mean_J.n"), "mJ": self.n(f"{base}.mean_J.mean"),
             "ciJ": self.ci(f"{base}.mean_J.lo", f"{base}.mean_J.hi"), "df": self.i(f"{base}.mean_J.df"), "p": self.p(f"{base}.signflip.p"),
             "minp": self.p(f"{base}.signflip.min_p"), "b3": self.n(f"{base}.estimators.raw.overall.beta_hat", m=True),
             "b3ci": self.ci(f"{base}.estimators.raw.overall.lo", f"{base}.estimators.raw.overall.hi") if bounded else "unbounded",
             "bpin": self.n(f"{base}.estimators.pin.overall.beta_hat", m=True), "bsym": self.n(f"{base}.estimators.sym.overall.beta_hat", m=True),
             "db": self.n(f"{base}.beta_minus_beta_C1.diff", m=True), "cidb": self.ci(f"{base}.beta_minus_beta_C1.lo", f"{base}.beta_minus_beta_C1.hi"), "e1": e1}
        self.fill(fid, lab, v, [f"{base}.{k}" for k in ("label", "mean_J", "signflip", "estimators", "beta_minus_beta_C1", "beta0_3", "draws_dropped", "per_draw")], lab)
        self.labels["B3"] = lab

    def dose(self):
        fid = "FULL.dose.result"; base = "blocks.dose"
        if "dose" in self.cut:
            self.fill(fid, "cut", {}, [f"queue state {CONFIG['guards']['dose']}"], "cut"); self.labels["dose"] = "cut"; return
        b = self.blk("dose"); lab = b.get("label"); raw = (b.get("fits") or {}).get("raw") or {}
        if lab not in ("linear dose-response", "non-linear dose-response", "no dose-response"):
            raise Pending(f"{base}.label = {lab!r}")
        c = (raw.get("primary") or {}).get("coef") or {}
        b1, b3 = c.get("b1") or {}, c.get("b3") or {}
        excl = FC.isnum(b1.get("lo")) and FC.isnum(b1.get("hi")) and not (b1["lo"] <= 0.0 <= b1["hi"])
        rej = FC.isnum(b3.get("p_two_sided")) and b3["p_two_sided"] <= 0.05
        self.check("dose-response", lab, ("non-linear dose-response" if rej else "linear dose-response") if excl else "no dose-response")
        v = {"n": self.i(f"{base}.fits.raw.n"),
             "nlev": self.dv("dose.levels", len({u.get('level') for u in raw.get("units") or []}), "number of distinct unit levels in blocks.dose.fits.raw.units", kind="int"),
             "b1": self.n(f"{base}.fits.raw.primary.coef.b1.b", m=True), "cib1": self.ci(f"{base}.fits.raw.primary.coef.b1.lo", f"{base}.fits.raw.primary.coef.b1.hi"),
             "b3": self.n(f"{base}.fits.raw.primary.coef.b3.b", m=True), "cib3": self.ci(f"{base}.fits.raw.primary.coef.b3.lo", f"{base}.fits.raw.primary.coef.b3.hi"),
             "pb3": self.p(f"{base}.fits.raw.primary.coef.b3.p_two_sided")}
        self.fill(fid, lab, v, [f"{base}.label", f"{base}.fits.raw.primary.coef", f"{base}.fits.raw.n", f"{base}.fits.raw.units"], lab)
        self.labels["dose"] = lab

    def dose_details(self):
        fid = "FULL.dose.details"; base = "blocks.dose"
        if self.labels.get("dose") == "cut":
            self.put(fid, "", [f"queue state {CONFIG['guards']['dose']}"], "cut"); return
        if self.labels.get("dose") is None:
            raise Pending("no dose-response fit")
        b = self.blk("dose"); P = T[fid]["parts"]; bases = (b.get("record") or {}).get("bases") or {}
        raw = ((b.get("fits") or {}).get("raw") or {})
        cnt = {}
        for bb in ("moe_b1", "moe_b2"):
            x = bases.get(bb) or {}
            cnt[bb] = (self.dv(f"dose.accepted_{bb}", len(x.get("accepted") or []), f"len(blocks.dose.record.bases.{bb}.accepted)", kind="int"),
                       self.dv(f"dose.examined_{bb}", len(x.get("examined") or []), f"len(blocks.dose.record.bases.{bb}.examined)", kind="int"))
        rec = subst(P["rec"], {"flow": self.n(f"{base}.record.f_low", 3, m=True), "a1": cnt["moe_b1"][0], "e1": cnt["moe_b1"][1], "a2": cnt["moe_b2"][0], "e2": cnt["moe_b2"][1]})
        for bb in ("moe_b1", "moe_b2"):
            if (bases.get(bb) or {}).get("reason"):
                rec += subst(P["notrun"], {"bn": CONFIG["bases"][bb]})
        nlev = len({u.get("level") for u in raw.get("units") or []})
        lev = P["lev"] if nlev == 3 else ""
        if nlev != 3:
            self.warn.append(f"dose: the fit has {nlev} levels, not the registered three; the sentence on the b3 test is not printed")
        bt = raw.get("bootstrap") or {}
        kap = ""
        if FC.isnum(bt.get("kappa")):
            kap = subst(P["kap"], {"k": self.n(f"{base}.fits.raw.bootstrap.kappa", m=True), "kci": self.ci(f"{base}.fits.raw.bootstrap.kappa_ci95[0]", f"{base}.fits.raw.bootstrap.kappa_ci95[1]")})
        else:
            self.warn.append("dose: kappa is None in the analysis output; its sentence is not printed")
        try:
            sens = subst(P["sens"], {"s1": self.n(f"{base}.fits.raw.sensitivity_base_specific_b1.coef.b1_moe_b1.b", m=True),
                                     "s2": self.n(f"{base}.fits.raw.sensitivity_base_specific_b1.coef.b1_moe_b2.b"),
                                     "s0": self.n(f"{base}.fits.raw.sensitivity_without_AA.coef.b1.b", m=True)})
        except Pending as e:
            sens = ""; self.warn.append(f"dose: sensitivity fits not printed ({e})")
        pa = b.get("pipeline_asymmetry_check") or {}
        aa = ""
        if pa.get("pairs") and FC.isnum(pa.get("mean_I")) and FC.isnum(pa.get("mean_I_in_sigma_units")):
            def pair_no(v):   # A/A pair id as the analysis writes it: n or "aa<n>"
                m_ = re.fullmatch(r"(?:aa)?(\d+)", str(v))
                if not m_:
                    raise Pending(f"{base}.pipeline_asymmetry_check: pair id {v!r} is neither n nor aa<n>")
                return int(m_.group(1))
            pr_ = {pair_no(x.get("pair")): k for k, x in enumerate(pa["pairs"])}
            g2 = {int(k): v for k, v in (pa.get("G2_record_I_AA") or {}).items()}
            same = all(FC.isnum(g2.get(a_)) and a_ in pr_ and FC.isnum(pa["pairs"][pr_[a_]].get("I")) and round(pa["pairs"][pr_[a_]]["I"], 3) == round(g2[a_], 3) for a_ in (2, 3))
            if same:
                pv = "pairs 2 and 3 of gate G2-1, whose values Appendix~\\ref{app:power} gives"
                for a_ in (2, 3):
                    self.data[f"{base}.pipeline_asymmetry_check.pairs[{pr_[a_]}].I"] = pa["pairs"][pr_[a_]]["I"]
            else:
                pv = f"{self.n(f'{base}.pipeline_asymmetry_check.pairs[{pr_.get(2, 0)}].I')} and {self.n(f'{base}.pipeline_asymmetry_check.pairs[{pr_.get(3, 1)}].I')} for pairs 2 and 3"
            aa = subst(P["aa"], {"mI": self.n(f"{base}.pipeline_asymmetry_check.mean_I"), "mIs": self.n(f"{base}.pipeline_asymmetry_check.mean_I_in_sigma_units", 2), "pv": pv})
        else:
            self.warn.append("dose: the pipeline-asymmetry check has no A/A pairs in the analysis output; its sentence is not printed")
        v = {"rec": rec, "bb1": self.ci(f"{base}.fits.raw.bootstrap.b1_ci95[0]", f"{base}.fits.raw.bootstrap.b1_ci95[1]"),
             "bb3": self.ci(f"{base}.fits.raw.bootstrap.b3_ci95[0]", f"{base}.fits.raw.bootstrap.b3_ci95[1]"),
             "lsym": texstr(((b.get("fits") or {}).get("sym") or {}).get("label")), "lev": lev, "kap": kap, "sens": sens, "aa": aa}
        self.fill(fid, "fit", v, [f"{base}.record", f"{base}.fits.raw.bootstrap", f"{base}.fits.sym.label", f"{base}.fits.raw.sensitivity_base_specific_b1",
                                  f"{base}.fits.raw.sensitivity_without_AA", f"{base}.pipeline_asymmetry_check"])

    def m12b(self):
        fid = "FULL.M12b.result"; base = "blocks.M12b"
        if "M12b" in self.cut:
            self.fill(fid, "cut", {}, [f"queue state {CONFIG['guards']['M12b']}"], "cut", tail=False); self.labels["M12b"] = "cut"; return
        b = self.blk("M12b"); lab = b.get("label")
        if lab not in ("target-dependent", "no evidence of target dependence"):
            raise Pending(f"{base}.label = {lab!r}")
        D = [r.get("Delta") for r in b.get("per_draw") or [] if FC.isnum(r.get("Delta"))]
        ti = b.get("mean_Delta") or {}
        same = bool(D) and (all(x > 0 for x in D) or all(x < 0 for x in D))
        excl = FC.isnum(ti.get("lo")) and FC.isnum(ti.get("hi")) and not (ti["lo"] <= 0 <= ti["hi"])
        self.check("M12-b", lab, "target-dependent" if (len(D) == 3 and same and excl) else "no evidence of target dependence")
        v = {"mD": self.n(f"{base}.mean_Delta.mean", m=True), "ciD": self.ci(f"{base}.mean_Delta.lo", f"{base}.mean_Delta.hi"), "df": self.i(f"{base}.mean_Delta.df")}
        if lab == "no evidence of target dependence":
            dd = r"the three differences $\Delta_d=IT_d^{\mathrm{learned}}-IT_d^{\mathrm{rand}}$"
            if len(D) < 3:
                v["cond"] = f"{self.dv('M12b.n_delta', len(D), 'number of finite Delta_d in blocks.M12b.per_draw', kind='int')} of the three draws gave a difference $\\Delta_d$"
            elif not same:
                v["cond"] = dd + " did not share one sign"
            else:
                v["cond"] = dd + r" shared one sign, but their 95\% $t$-interval contained 0"
        self.fill(fid, lab, v, [f"{base}.label", f"{base}.mean_Delta", f"{base}.per_draw", f"{base}.sign_agreement"], lab)
        self.labels["M12b"] = lab

    def c1de(self):
        fid = "FULL.C1DE.result"; base = "blocks.C1_DE"
        if "C1-DE" in self.cut:
            self.fill(fid, "cut", {}, [f"queue state {CONFIG['guards']['C1-DE']}"], "cut", tail=False); self.labels["C1_DE"] = "cut"; return
        b = self.blk("C1_DE"); lab = b.get("label")
        if lab not in ("sign-consistent with C1", "sign-inconsistent", "mixed"):
            raise Pending(f"{base}.label = {lab!r}")
        J = [r.get("J") for r in b.get("per_draw") or [] if FC.isnum(r.get("J"))]
        self.check("C1-DE", lab, "sign-consistent with C1" if J and all(j > 0 for j in J) else ("sign-inconsistent" if J and all(j < 0 for j in J) else "mixed"))
        ov = [x[s] for mf in (b.get("manifests") or {}).values() for x in mf.get("pairwise_module_overlap") or [] for s in ("A", "B") if FC.isnum(x.get(s))]
        if not ov:
            raise Pending("no pairwise module overlaps in blocks.C1_DE.manifests")
        bounded = ((b.get("estimators") or {}).get("raw") or {}).get("overall", {}).get("bounded")

        def drawname(base_id, d):
            return f"draw {d} of {CONFIG['bases'].get(base_id, texstr(base_id))}"
        tl = [drawname(*x.split(":d")) for x in b.get("time_limit_draws") or []]
        ng = [drawname(k.split(":")[-1], d) for k, mf in (b.get("manifests") or {}).items() for d in mf.get("not_generated") or []]
        v = {"n": self.i(f"{base}.mean_J.n"), "mJ": self.n(f"{base}.mean_J.mean"), "ciJ": self.ci(f"{base}.mean_J.lo", f"{base}.mean_J.hi"), "df": self.i(f"{base}.mean_J.df"),
             "mJ1": self.n(f"{base}.mean_J_per_base.moe_b1.mean"), "mJ2": self.n(f"{base}.mean_J_per_base.moe_b2.mean"), "p": self.p(f"{base}.signflip.p"),
             "bDE": self.n(f"{base}.estimators.raw.overall.beta_hat", m=True),
             "ciDE": self.ci(f"{base}.estimators.raw.overall.lo", f"{base}.estimators.raw.overall.hi") if bounded else "unbounded",
             "bpin": self.n(f"{base}.estimators.pin.overall.beta_hat", m=True), "bsym": self.n(f"{base}.estimators.sym.overall.beta_hat", m=True),
             "db": self.n(f"{base}.beta_minus_beta_C1.diff", m=True), "cidb": self.ci(f"{base}.beta_minus_beta_C1.lo", f"{base}.beta_minus_beta_C1.hi"),
             "ov0": self.dv("C1_DE.module_overlap_min", min(ov), "min over blocks.C1_DE.manifests[*].pairwise_module_overlap[*].{A,B}", 2),
             "ov1": self.dv("C1_DE.module_overlap_max", max(ov), "max over blocks.C1_DE.manifests[*].pairwise_module_overlap[*].{A,B}", 2),
             "tl": (f" The solves of {words_join(tl)} reached the solver time limit of 600 seconds, so {'this draw is' if len(tl) == 1 else 'these draws are'} flagged as not reproducible on another machine." if tl else ""),
             "ng": (f" The registered generator gave no pair for {words_join(ng)}, so {'this draw is' if len(ng) == 1 else 'these draws are'} not run." if ng else "")}
        if lab == "mixed":
            v["npos"] = self.dv("C1_DE.n_positive", sum(1 for j in J if j > 0), "count of J_d > 0 in blocks.C1_DE.per_draw", kind="int")
            v["nneg"] = self.dv("C1_DE.n_negative", sum(1 for j in J if j < 0), "count of J_d < 0 in blocks.C1_DE.per_draw", kind="int")
        self.fill(fid, lab, v, [f"{base}.{k}" for k in ("label", "mean_J", "mean_J_per_base", "signflip", "estimators", "beta_minus_beta_C1", "manifests", "time_limit_draws", "per_draw")], lab)
        self.labels["C1_DE"] = lab

    def selpin(self):
        fid = "FULL.selpin.result"; b = self.blk("selpin")
        if not (b.get("C1") or {}).get("shares_over_draws"):
            raise Pending("blocks.selpin.C1.shares_over_draws missing")
        self.fill(fid, "descriptive", {}, ["blocks.selpin"])

    def c3(self):
        fid = "FULL.C3.result"; base = "blocks.C3"; b = self.blk("C3"); P = T[fid]["parts"]
        if not b.get("per_base"):
            raise Pending(f"{base}.per_base missing")
        nl = b.get("NESTOR_label") or {}
        lab, out = nl.get("label"), (nl.get("G0_9L") or {}).get("outcome")
        nlt = P["nl_exploratory"] if (lab or "").startswith("exploratory") else (P["nl_unchanged"] if out in (None, "not_evaluable", "not evaluated") else P["nl_pass"])
        if (b.get("nestor_moe_comparison") or {}).get("made"):
            self.warn.append("C3: the analysis made a MoE-NESTOR comparison; the template states that none is made: revise the template")
        if not str((b.get("random_partial_revert") or {}).get("status", "")).startswith("not computable"):
            self.warn.append("C3: a random partial-revert control is present in the analysis; the template states that it is not run: revise the template")
        self.fill(fid, "descriptive", {"nl": nlt}, [f"{base}.per_base", f"{base}.freeze_cross_check", f"{base}.NESTOR_label", f"{base}.random_partial_revert", f"{base}.nestor_moe_comparison"], label=lab)

    def c3wd(self):
        fid = "FULL.C3.wd"; base = "blocks.C3.per_base"; self.blk("C3")
        acc = self.a2["accepted"]
        bad = sorted(j for j in CONFIG["a2_jobs"] if acc.get(j) is not True)
        items = []
        for bb, name in CONFIG["wd_bases"]:
            w = ((((self.doc.get("blocks") or {}).get("C3") or {}).get("per_base") or {}).get(bb) or {}).get("WD_share", {}).get("F_WD_over_F") or {}
            held = {CONFIG["a2_jobs"][j][1] for j in bad if CONFIG["a2_jobs"][j][0] == bb}
            per = {d: v for d, v in (w.get("per_domain") or {}).items() if FC.isnum(v)}
            if not per:
                raise Pending(f"{base}.{bb}.WD_share.F_WD_over_F has no values")
            if not held:
                items.append(f"{self.n(f'{base}.{bb}.WD_share.F_WD_over_F.mean')} on {name} (mean over its {self.i(f'{base}.{bb}.WD_share.F_WD_over_F.n_domains')} domains)")
                continue
            keep = sorted(d for d in per if d not in held)
            if len(keep) == 1:
                items.append(f"{self.n(f'{base}.{bb}.WD_share.F_WD_over_F.per_domain.{keep[0]}')} on {name} ({CORPUS[keep[0]]})")
            elif keep:
                mv = sum(per[d] for d in keep) / len(keep)
                items.append(f"{self.dv(f'C3.WD_{bb}_kept_mean', mv, f'mean of {base}.{bb}.WD_share.F_WD_over_F.per_domain over ' + ', '.join(keep))} on {name} (mean over {words_join([CORPUS[d] for d in keep])})")
        held_txt = ""
        if bad:
            held_txt = subst(T[fid]["parts"]["held"], {"which": words_join([CONFIG["a2_jobs"][j][2] for j in bad])})
            if len(bad) > 1:
                held_txt = held_txt.replace("The value of", "The values of").replace("is withheld", "are withheld").replace("is not met for it", "is not met for them")
        self.fill(fid, "values", {"list": words_join(items), "held": held_txt}, [f"{base}.moe_b1.WD_share", f"{base}.nestor.WD_share", "A-2 acceptance record"])

    def c3layer(self):
        fid = "FULL.C3.layer"; base = "blocks.C3_per_layer.per_base"; self.blk("C3_per_layer")
        v = {"g1": self.n(f"{base}.moe_b1.additivity_gap.mean", 4), "r1": self.ci(f"{base}.moe_b1.additivity_gap.range[0]", f"{base}.moe_b1.additivity_gap.range[1]", 4),
             "gn": self.n(f"{base}.nestor.additivity_gap.mean", 4), "rn": self.ci(f"{base}.nestor.additivity_gap.range[0]", f"{base}.nestor.additivity_gap.range[1]", 4)}
        self.fill(fid, "descriptive", v, [f"{base}.*.additivity_gap", f"{base}.*.per_unit"])

    def frel(self):
        fid = "FULL.Frel.result"; base = "blocks.F_rel.C1"; self.blk("F_rel")
        v = {"n": self.i(f"{base}.n"), "mJ": self.n(f"{base}.mean_J.mean", m=True), "ciJ": self.ci(f"{base}.mean_J.lo", f"{base}.mean_J.hi"), "p": self.p(f"{base}.signflip.p")}
        self.fill(fid, "C1", v, [base, "blocks.F_rel.C2.status"])

    def c5_all_null(self, hp):
        """True if the hashed C5 file is the one the analysis verified and every forecast in it is null; no value of the file is read or printed"""
        p = CONFIG["c5_forecasts_json"]
        try:
            fdoc = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            raise Pending(f"{p} unreadable ({e})")
        if fdoc.get("sha256") != hp.get("sha256"):
            raise Pending(f"{p}: sha256 field differs from blocks.C5_french.hash_protocol.sha256")
        vals = [v for row in (fdoc.get("forecasts") or []) for v in ((row or {}).get("forecasts") or {}).values()]
        return bool(vals) and all(v is None for v in vals)

    def c5(self):
        fid = "FULL.C5.result"; base = "blocks.C5_french"; b = self.blk("C5_french"); P = T[fid]["parts"]; lab = b.get("label")
        if lab not in ("prospective", "non-prospective"):
            raise Pending(f"{base}.label = {lab!r}")
        hp = b.get("hash_protocol") or {}
        self.check("C5", lab, "prospective" if hp.get("prospective") else "non-prospective")
        scored = [nm for nm in (b.get("forecasters_in_file") or []) if FC.isnum((((b.get("metrics") or {}).get(nm) or {}).get("pooled") or {}).get("n")) and b["metrics"][nm]["pooled"]["n"] > 0]
        if not scored:                                    # deviation D-4: no forecast formed, block C5 not evaluable as registered
            if b.get("runs_not_usable"):
                raise Pending(f"no pooled score and runs not usable {b['runs_not_usable']}: no approved text")
            if not self.c5_all_null(hp):
                raise Pending("no pooled score although the hashed file holds forecasts: no approved text")
            self.fill(fid, "not_evaluable", {}, [f"{base}.label", f"{base}.hash_protocol", f"{base}.metrics", f"{base}.runs_not_usable", f"{CONFIG['c5_forecasts_json']}: forecasts (null test only)"], CONFIG["c5_not_evaluable"])
            self.labels["C5"] = CONFIG["c5_not_evaluable"]
            return
        names = [nm for nm in (b.get("forecasters_in_file") or []) if FC.isnum((((b.get("metrics") or {}).get(nm) or {}).get("pooled") or {}).get("R2_oos"))]
        if not names:
            raise Pending(f"no pooled R2_oos in {base}.metrics")
        ns = {b["metrics"][nm]["pooled"].get("n") for nm in names}
        items = [f"{self.n(f'{base}.metrics.{nm}.pooled.R2_oos')} for {CONFIG['c5_names'].get(nm, texstr(nm))}" for nm in names]
        if len(ns) == 1:
            met = subst(P["met"], {"list": words_join(items), "n": self.i(f"{base}.metrics.{names[0]}.pooled.n")})
        else:
            met = subst(P["met"].replace(", over [[n]] corpus values each", ""), {"list": words_join(items)})
        r2 = {nm: ((b.get("R2_within") or {}).get(nm) or {}).get("R2_within") for nm in names}
        notes = [str(((b.get("R2_within") or {}).get(nm) or {}).get("note", "")) for nm in names]
        if all(x is None for x in r2.values()) and not all(nt.startswith("fewer than 3 French crossover pairs") for nt in notes):
            raise Pending(f"{base}.R2_within is None without the registered reason: {notes}")
        r2w = P["r2w_none"] if all(x is None for x in r2.values()) else "$R^2_{\\mathrm{within}}$ was " + words_join(
            [f"{self.n(f'{base}.R2_within.{nm}.R2_within')} for {CONFIG['c5_names'].get(nm, texstr(nm))}" for nm, x in r2.items() if x is not None]) + "."
        reasons = []
        for r in hp.get("reasons") or []:
            hit = next((plain for pat, plain in C5_REASONS if re.search(pat, str(r))), None)
            if hit is None:
                raise Pending(f"C5 reason without a plain-word mapping: {r!r}")
            reasons.append(hit)
        self.fill(fid, lab, {"met": met, "r2w": r2w, "reasons": words_join(reasons)}, [f"{base}.label", f"{base}.hash_protocol", f"{base}.metrics", f"{base}.R2_within"], lab)
        self.labels["C5"] = lab

    def forecasts(self):
        fid = "FULL.forecasts.result"; b = self.blk("forecasts")
        if not (b.get("scores") or {}).get("by_forecaster"):
            raise Pending("blocks.forecasts.scores missing")
        self.fill(fid, "scores", {}, ["blocks.forecasts.scores"])

    # -- assumption audit (prereg v1.3 sec. 4.3): blocks.audit, read strictly and cross-checked with the A-3 re-run record
    def aval(self, path):
        v = getp(self.doc, path)
        if not FC.isnum(v):
            raise Pending(f"{path} = {v!r}")
        self.data[path] = v
        return v

    def an(self, path, nd=3):
        v = self.aval(path)
        s = FC.num(v, nd)
        self._rec(s, path, v)
        return s

    def ai(self, path):
        v = self.aval(path)
        s = fmt_int(v)
        self._rec(s, path, v)
        return s

    def audit_bases(self):
        """the bases with and without audit results, in the order of CONFIG a3_jobs; a structure other than the registered one,
        or results that disagree with the A-3 re-run record, stop the fill"""
        A = CONFIG["audit"]

        def stop(msg):
            raise SystemExit(f"[fill_full] blocks.audit: {msg}; nothing written")

        def isint(x):
            return isinstance(x, int) and not isinstance(x, bool) and x >= 0

        def pair(x):
            return x is None or (isinstance(x, list) and len(x) == 2)

        au = (self.doc.get("blocks") or {}).get("audit")
        if not isinstance(au, dict):
            stop("missing from the full analysis output")
        bases = {b: job for job, (b, _) in CONFIG["a3_jobs"].items()}
        extra = sorted(k for k in au if k != "prereg" and k not in bases)
        if extra:
            stop(f"keys outside the bases of the A-3 record: {extra}")
        have, lack = [], []
        for b, job in bases.items():
            v, p = au.get(b), f"blocks.audit.{b}"
            if isinstance(v, dict) and set(v) == {"status"} and str(v["status"]).startswith("not available"):
                res = False
            elif isinstance(v, dict) and set(v) == {"B5", "B6", "A0", "eps", "corpora", "bootstrap"}:
                res = True
            else:
                stop(f"{p} is neither the registered result nor a 'not available' status: {sorted(v) if isinstance(v, dict) else v!r}")
            if res != self.a3_ok(job):
                stop(f"{p} {'has results' if res else 'has no results'}, while the A-3 re-run record {'lacks' if res else 'has'} the registered outputs of that base")
            if res:
                bs, b5, b6, a0 = v["bootstrap"], v["B5"], v["B6"], v["A0"]
                if v["corpora"] != A["corpora"]:
                    stop(f"{p}.corpora = {v['corpora']!r}, registered {A['corpora']!r}")
                if not (isinstance(bs, dict) and set(bs) == {"n", "seed", "unit"} and bs["n"] == A["n_boot"] and bs["unit"] == A["boot_unit"]):
                    stop(f"{p}.bootstrap = {bs!r}; registered: {A['n_boot']} resamples of {A['boot_unit']}")
                if not (isinstance(b5, dict) and set(b5) == AUDIT_KEYS["B5"] and pair(b5["ci95"]) and isint(b5["cells"])
                        and isint(b5["cells_Q_nonpositive"]) and isint(b5["bootstrap"]) and b5["label"] in AUDIT_LABELS["B5"] + [None]):
                    stop(f"{p}.B5 does not have the registered structure and labels")
                if not (isinstance(b6, dict) and set(b6) == set(A["subsets"])):
                    stop(f"{p}.B6 subsets {sorted(b6) if isinstance(b6, dict) else b6!r}, registered {sorted(A['subsets'])}")
                for sub, x in b6.items():
                    if not (isinstance(x, dict) and set(x) == AUDIT_KEYS["B6"] and pair(x["cv_ci95"]) and pair(x["spearman_ci95"])
                            and x["label"] in AUDIT_LABELS["B6"] + [None]):
                        stop(f"{p}.B6.{sub} does not have the registered structure and labels")
                if not (isinstance(a0, dict) and set(a0) == set(A["corpora"])):
                    stop(f"{p}.A0 corpora {sorted(a0) if isinstance(a0, dict) else a0!r}, registered {A['corpora']}")
                for c, x in a0.items():
                    if not (isinstance(x, dict) and set(x) == AUDIT_KEYS["A0"] and isint(x["n"]) and x["n"] <= A["groups"]
                            and x["label"] in AUDIT_LABELS["A0"] + [None]):
                        stop(f"{p}.A0.{c} does not have the registered structure and labels")
                if b5["cells"] == A["groups"] * len(A["corpora"]) and sum(A["groups"] - a0[c]["n"] for c in A["corpora"]) != b5["cells_Q_nonpositive"]:
                    stop(f"{p}: the cells with Q > 0 per corpus (A0 n) do not add up with B5.cells_Q_nonpositive")
            (have if res else lack).append(b)
        return have, lack

    def audit_set(self, lab):
        """the registered labels of one base as a phrase: B5; B6 per subset; A0 per corpus"""
        A, P = CONFIG["audit"], T["FULL.audit"]["parts"]
        l5, l6, l0 = lab
        s5 = subst(P["set_B5"], {"l5": texstr(l5)})
        if len(set(l6)) == 1:
            s6 = subst(P["set_B6_same"], {"l6": texstr(l6[0])})
        else:
            s6 = subst(P["set_B6_diff"], {"l6a": texstr(l6[0]), "l6b": texstr(l6[1])})
        if len(set(l0)) == 1:
            s0 = P["set_A0_same_c"] if l0[0] in CONFIG["audit_wording"] else subst(P["set_A0_same"], {"l0": texstr(l0[0])})
        else:
            grp = {}
            for c, l in zip(A["corpora"], l0):
                grp.setdefault(l, []).append(CORPUS[c])
            s0 = ", ".join(subst(P["set_A0_group_c" if l in CONFIG["audit_wording"] else "set_A0_group"], {"l0": texstr(l), "corp": words_join(cs)}) for l, cs in grp.items())
        return subst(P["set_join"], {"b5": s5, "b6": s6, "a0": s0})

    def audit(self):
        fid = "FULL.audit"; A = CONFIG["audit"]; names = {b: n for _, (b, n) in CONFIG["a3_jobs"].items()}
        self.audit_have = []
        have, lack = self.audit_bases()
        fields = ["blocks.audit.<base>.B5: interaction_share, ci95, cells, cells_Q_nonpositive, bootstrap, label",
                  "blocks.audit.<base>.B6.<subset>: cv, spearman, spearman_ci95, jensen_gap, label",
                  'blocks.audit.<base>.A0.<corpus>: median_r, ["share_r_ge_0.2"], n, label', "blocks.audit.<base>.bootstrap, corpora",
                  "A-3 re-run record: jobs.*.status, outputs, shim_line_in_log"]
        if not have:
            self.fill(fid, "none", {}, fields, "no outputs on either base"); return
        labs = {}
        for b in have:
            p = f"blocks.audit.{b}"; v = self.doc["blocks"]["audit"][b]; b5 = v["B5"]
            if b5["cells"] != A["groups"] * len(A["corpora"]):
                raise Pending(f"{p}.B5.cells = {b5['cells']}; the registered design has {A['groups']} groups on {len(A['corpora'])} corpora")
            if b5["bootstrap"] != v["bootstrap"]["n"]:
                raise Pending(f"{p}.B5.bootstrap: {b5['bootstrap']} of {v['bootstrap']['n']} resamples gave an interaction share")
            sh5 = self.aval(f"{p}.B5.interaction_share"); self.aval(f"{p}.B5.ci95[0]"); self.aval(f"{p}.B5.ci95[1]")
            if b5["label"] is None:
                raise Pending(f"{p}.B5.label = None")
            self.check(f"assumption audit, {b}, B5", b5["label"], next(lab for cut, lab in A["b5_cuts"] if sh5 < cut))
            for sub in A["subsets"]:
                q = f"{p}.B6.{sub}"; x = v["B6"][sub]
                cv = self.aval(q + ".cv"); self.aval(q + ".spearman"); self.aval(q + ".jensen_gap")
                lo, hi = self.aval(q + ".spearman_ci95[0]"), self.aval(q + ".spearman_ci95[1]")
                if x["label"] is None:
                    raise Pending(f"{q}.label = None")
                self.check(f"assumption audit, {b}, B6 subset {sub}", x["label"],
                           "homogeneous" if cv < A["cv_homog"] else ("heterogeneous, uncorrelated" if lo <= 0.0 <= hi else "heterogeneous, correlated"))
            for c in A["corpora"]:
                q = f"{p}.A0.{c}"; x = v["A0"][c]
                sh0 = self.aval(q + '["share_r_ge_0.2"]'); self.aval(q + ".median_r")
                if x["label"] is None:
                    raise Pending(f"{q}.label = None")
                self.check(f"assumption audit, {b}, A0 {c}", x["label"], "A0 approximately holds" if sh0 <= A["a0_share"] else "A0 violated")
            labs[b] = (b5["label"], tuple(v["B6"][sub]["label"] for sub in A["subsets"]), tuple(v["A0"][c]["label"] for c in A["corpora"]))
        P = T[fid]["parts"]
        if len(have) > 1 and len(set(labs.values())) == 1:
            lt = subst(P["labels_both"], {"set": self.audit_set(labs[have[0]])})
        else:
            lt = "".join(subst(P["labels_base" if k == 0 else "labels_base2"], {"name": names[b], "set": self.audit_set(labs[b])})
                         for k, b in enumerate(have))
        vals = {"labels": lt, "corr": P["corr"] if any("heterogeneous, correlated" in labs[b][1] for b in have) else ""}
        if lack:
            vals.update({"per": "", "ok": words_join([names[b] for b in have]), "bad": words_join([names[b] for b in lack])})
            self.fill(fid, "one", vals, fields, "one base")
        else:
            vals["per"] = P["per"]
            self.fill(fid, "both", vals, fields, "both bases")
        self.audit_have, self.audit_labels, self.audit_missing = have, labs, lack

    def audit_c1(self):
        """FULL.audit.C1: the audit labels of FULL.audit in one main-text sentence, grouped by label (prereg v1.3 sec. 4.3: the labels are reported with C1)"""
        fid = "FULL.audit.C1"; A = CONFIG["audit"]; names = {b: n for _, (b, n) in CONFIG["a3_jobs"].items()}
        have, labs = getattr(self, "audit_have", None), getattr(self, "audit_labels", None)
        if have is None or labs is None:
            raise Pending("FULL.audit has no labels (it must be filled first)")
        fields = ["labels of FULL.audit (blocks.audit.<base>.B5/B6/A0 .label, checked against the registered rules)"]
        if not have:
            self.fill(fid, "none", {}, fields, "no outputs on either base"); return
        q = lambda l: "``" + texstr(l) + "''"
        b5 = [labs[b][0] for b in have]
        s5 = "B5 " + (q(b5[0]) + (" on both bases" if len(have) > 1 else f" on {names[have[0]]}") if len(set(b5)) == 1 else words_join([f"{q(l)} on {names[b]}" for b, l in zip(have, b5)]).replace(" on ", " on ", 1))
        cells6 = {}
        for b in have:
            for k, sub in enumerate(A["subsets"]):
                cells6.setdefault(labs[b][1][k], []).append((b, A["subsets"][sub]))
        def where6(cs):
            out = []
            for b in have:
                ss = [s for bb, s in cs if bb == b]
                if ss:
                    out.append(f"both subsets of {names[b]}" if len(ss) == len(A["subsets"]) else f"subset {words_join(ss)} of {names[b]}")
            return " and in ".join(out)
        s6 = "B6 " + " and ".join(f"{q(l)} in {where6(cs)}" for l, cs in cells6.items())
        cells0 = {}
        for b in have:
            for k, c in enumerate(A["corpora"]):
                cells0.setdefault(labs[b][2][k], []).append((b, CORPUS[c]))
        def where0(cs):
            if len(cs) == len(have) * len(A["corpora"]):
                return "any of the three corpora of either base" if len(have) > 1 else f"any of the three corpora of {names[have[0]]}"
            return " and ".join(f"{words_join([c for bb, c in cs if bb == b])} of {names[b]}" for b in have if any(bb == b for bb, _ in cs))
        s0 = "; ".join((f"A0 does not hold approximately on {where0(cs)}" if l in CONFIG["audit_wording"] else f"{q(l)} on {where0(cs).replace('any of', 'all')}") for l, cs in cells0.items())
        lack = getattr(self, "audit_missing", [])
        self.fill(fid, "one" if lack else "labels", {"b5": s5, "b6": s6, "a0": s0, "ok": words_join([names[b] for b in have])}, fields, "one base" if lack else "both bases")

    def audit_rows(self):
        """rows of Table tab:fullaudit; every value is recorded with its field under TAB.fullaudit (number map in the manifest)"""
        A = CONFIG["audit"]; names = {b: n for _, (b, n) in CONFIG["a3_jobs"].items()}; au = self.doc["blocks"]["audit"]
        r5, r6, r0 = [], [], []
        for b in self.audit_have:
            p = f"blocks.audit.{b}"
            share, lo, hi = self.an(p + ".B5.interaction_share"), self.an(p + ".B5.ci95[0]"), self.an(p + ".B5.ci95[1]")
            nq, nc = self.ai(p + ".B5.cells_Q_nonpositive"), self.ai(p + ".B5.cells")
            r5.append(f"{names[b]} & {share} & [{lo}, {hi}] & {nq} of {nc} & {texstr(au[b]['B5']['label'])}")
            for k, sub in enumerate(A["subsets"]):
                q = f"{p}.B6.{sub}"
                cv, rho, rlo, rhi, jg = (self.an(q + x) for x in (".cv", ".spearman", ".spearman_ci95[0]", ".spearman_ci95[1]", ".jensen_gap"))
                r6.append(f"{names[b] if k == 0 else ''} & {A['subsets'][sub]} & {cv} & {rho} & [{rlo}, {rhi}] & {jg} & {texstr(au[b]['B6'][sub]['label'])}")
            for k, c in enumerate(A["corpora"]):
                q = f"{p}.A0.{c}"
                med, sh = self.an(q + ".median_r"), self.an(q + '["share_r_ge_0.2"]')
                r0.append(f"{names[b] if k == 0 else ''} & {CORPUS[c]} & {med} & {sh} & {texstr(CONFIG['audit_wording'].get(au[b]['A0'][c]['label'], au[b]['A0'][c]['label']))}")
        return r5, r6, r0

    def guard_outcome(self, need_detail=True):
        g = self.guard_info
        miss = [x["gate"] for x in g.values() if x["status"] == "missing"]
        if miss:
            raise Pending(f"guard records missing in the queue state: {miss}")
        failed = [k for k, x in g.items() if x.get("failed")]
        if failed:
            if not g[failed[0]].get("detail"):
                raise Pending(f"guard {g[failed[0]]['gate']} failed without detail in the queue state")
            return "cut", failed[0]
        ok = [k for k, x in g.items() if (x.get("detail") or {}).get("pass") is True or (not need_detail and x["status"] in ("pass", "passed"))]
        if len(ok) != len(g) or (need_detail and any(not x.get("detail") for x in g.values())):
            raise Pending("guard records without detail in the queue state: " + ", ".join(f"{x['gate']} ({x['status']})" for x in g.values() if not x.get("detail")))
        return "none_cut", None

    def compute(self):
        fid = "FULL.compute"; base = "blocks.budget"; b = self.blk("budget"); P = T[fid]["parts"]
        if not b.get("per_block"):
            raise Pending(f"{base}.per_block missing")
        kind, k = self.guard_outcome(need_detail=True); g = self.guard_info
        for kk, x in g.items():
            for fld in ("projected_gpu_h", "envelope_gpu_h", "spent_gpu_h", "remaining_planned_gpu_h", "after_cuts_gpu_h"):
                if FC.isnum((x.get("detail") or {}).get(fld)):
                    self.data[f"guards.{kk}.{fld}"] = x["detail"][fld]
        if kind == "none_cut":
            gd = subst(P["none_cut"], {"pmax": self.dv("guards.projected_max", max(float(x["detail"]["projected_gpu_h"]) for x in g.values()), "max over the three guards of detail.projected_gpu_h", 2),
                                       "env": self.dv("guards.envelope", max(float(x["detail"]["envelope_gpu_h"]) for x in g.values()), "detail.envelope_gpu_h of the guards", 1)})
        else:
            d = g[k]["detail"]
            gd = subst(P["cut"], {"blk": CONFIG["guard_names"][k], "proj": self.dv(f"guards.{k}.projected", float(d["projected_gpu_h"]), f"detail.projected_gpu_h of the guard before {k}", 2),
                                  "env": self.dv(f"guards.{k}.envelope", float(d["envelope_gpu_h"]), f"detail.envelope_gpu_h of the guard before {k}", 1)})
        ext = ""
        x = (b.get("envelope_extension") or {}).get("gpu_h")
        if FC.isnum(x) and x > 0:
            ext = f" The envelope was extended by {self.n(f'{base}.envelope_extension.gpu_h', 2)} GPU-hours under the registered envelope decision."
        # measured total (20 November): hours spent at the freeze + hours of the queue state + hours outside it, without and with the exploratory hours
        if ext:
            raise Pending("the envelope was extended; the measured-total sentence has no approved text for an extended envelope")
        spent = ((self.doc.get("parameters") or {}).get("planned budget total") or {}).get("spent_gpu_h")
        st = self.state if isinstance(self.state, dict) else {}
        notdone = sorted(j for j, e in st.items() if (e or {}).get("status") not in ("done", "pass"))
        qh = [(e or {}).get("hours") for e in st.values()]
        hos_t = (getattr(self, "hos", None) or {}).get("total"); ex = (getattr(self, "expl", None) or {}).get("gpu_hours_total")
        miss = [w for w, ok in (("parameters.planned budget total.spent_gpu_h", FC.isnum(spent)), ("queue state with every entry done or pass", bool(st) and not notdone),
                                ("hours of every queue entry", bool(qh) and all(FC.isnum(h) for h in qh)), (f"{CONFIG['hos_json']} total", FC.isnum(hos_t)),
                                (f"{CONFIG['expl_json']} gpu_hours_total", FC.isnum(ex))) if not ok]
        if miss:
            raise Pending(f"measured total: missing or incomplete {miss}" + (f" (not done: {notdone[:5]})" if notdone else ""))
        qsum = round(sum(float(h) for h in qh), 4); self.data["compute.queue_state_hours"] = qsum; self.data["compute.spent_at_freeze"] = float(spent)
        t1 = round(float(spent) + qsum + float(hos_t), 4); t2 = round(t1 + float(ex), 4)
        env_ = max(float(x["detail"]["envelope_gpu_h"]) for x in g.values())
        if not t2 < env_:
            raise Pending(f"measured total {t2} GPU-hours with the exploratory hours is not below the envelope {env_}; no approved text")
        pr_q = round(float((b.get("total") or {}).get("actual_gpu_h") or qsum), 2)       # the queue total as Table tab:fullbudget prints it
        if round(float(spent) + pr_q + float(hos_t), 1) != round(t1, 1) or round(float(spent) + pr_q + float(hos_t) + float(ex), 1) != round(t2, 1):
            self.warn.append(f"FULL.compute: the one-decimal totals {round(t1, 1)} and {round(t2, 1)} differ from the sums of the printed components")
        tot = subst(P["tot"], {"t1": self.dv("compute.measured_total", t1, f"parameters.planned budget total.spent_gpu_h + sum of hours over the queue state + {CONFIG['hos_json']} total (record value; printed at one decimal)", 1),
                               "t2": self.dv("compute.measured_total_with_exploratory", t2, f"compute.measured_total + {CONFIG['expl_json']} gpu_hours_total (record value; printed at one decimal)", 1),
                               "envi": self.dv("compute.envelope", env_, "max over the three guards of detail.envelope_gpu_h", 0)})
        self.fill(fid, "totals", {"gd": gd, "ext": ext, "tot": tot}, [f"{base}.per_block", f"{base}.total", f"{base}.envelope_extension", "queue state: overrun_guard_* detail",
                                                                  "parameters.planned budget total.spent_gpu_h", "queue state: hours", f"{CONFIG['hos_json']}: total", f"{CONFIG['expl_json']}: gpu_hours_total"])

    def edge_all(self):
        """Appendix F, paragraph 'All covered runs.' (v1.6.11; final check of v1.6.10, M1): the registered cost block over
        every covered run, from blocks.EDGE (by_type, excluded, runs, pre_update_check); hours are wall_s / 3600"""
        fid = "FULL.EDGE.all"; base = "blocks.EDGE"; b = self.blk("EDGE")
        bt, runs, exc, pu = b.get("by_type") or {}, b.get("runs") or [], b.get("excluded") or [], b.get("pre_update_check") or {}
        if not (bt and runs and pu.get("per_base")):
            raise Pending(f"{base}: by_type, runs or pre_update_check missing")
        W = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        types = {"MoE|subset": "msub", "MoE|FULL": "mfull", "NESTOR|FULL": "nfull", "NESTOR|subset": "nsub",
                 "MoE|FULL minus a frozen group (C3 freeze run)": "mfreeze", "MoE|C2 SGD": "c2sgd", "MoE|C2 Muon": "c2muon"}
        name = lambda k: k.replace("|", " ")        # the claims ledger is a markdown table: no vertical bar in a field or its description
        if set(bt) != set(types):
            raise Pending(f"{base}.by_type has the run types {sorted(name(k) for k in bt)}; the approved wording covers {sorted(name(k) for k in types)}")
        if sum(int(bt[k].get("n") or 0) for k in types) != len(runs):
            raise Pending(f"{base}: the counts of by_type do not add up to the {len(runs)} runs")
        stages = sorted({str(r.get("stage")) for r in runs})
        if stages != ["confirm", "forecast", "g1", "g2"]:
            raise Pending(f"{base}.runs: stages {stages}; the approved wording covers G1, G2, the forecast drive runs and the runs after the freeze")
        if len(exc) != 1:
            raise Pending(f"{base}.excluded: {len(exc)} runs; the approved wording names one excluded run")
        m = re.fullmatch(r"c3_freeze_nestor_(books|code|german|math)_s\d+", str(exc[0].get("run")))
        g = re.search(r"power samples ([0-9.]+) s apart", str(exc[0].get("reason")))
        if not (m and g) or any(r.get("run") == exc[0].get("run") for r in runs):
            raise Pending(f"{base}.excluded[0] = {exc[0]!r}: no approved description")
        dom = {"books": "books", "code": "code", "german": "German", "math": "mathematics"}[m.group(1)]
        v = {"nrun": self.dv("EDGE.all.runs_n", len(runs), f"number of entries of {base}.runs (the covered runs)", kind="int"),
             "exname": f"\\nestor{{}} freeze run of C3 on {dom}",
             "gap": self.dv("EDGE.all.excluded_gap_s", float(g.group(1)), f"gap in seconds between consecutive power samples stated in {base}.excluded[0].reason", 1)}

        def cnt(key, n, how):                       # counts below 10 in words (no ledger token), from 10 on in digits with a field
            return W[n] if 0 < n < 10 else self.dv(key, n, how, kind="int")

        def span(k, what):                          # median and [min, max] of one run type: hours from wall_s, or Wh
            d = bt[k].get("wall_s" if what == "h" else "energy_Wh") or {}
            if not all(FC.isnum(d.get(x)) for x in ("median", "min", "max")):
                raise Pending(f"{base}.by_type, run type {name(k)}: no median, min and max")
            unit, nd, f_ = ("wall_s / 3600", 3, lambda x: float(x) / 3600) if what == "h" else ("energy_Wh", 1, float)
            s = [self.dv(f"EDGE.all.{types[k]}.{'wall_h' if what == 'h' else 'energy_Wh'}_{x}", f_(d[x]), f"{x} of {unit} of the run type {name(k)} in {base}.by_type", nd)
                 for x in ("median", "min", "max")]
            return s[0], f"[{s[1]}, {s[2]}]"
        for k, hk, ek in (("MoE|subset", "msh", "mse"), ("MoE|FULL", "mfh", "mfe"), ("NESTOR|FULL", "nfh", "nfe"), ("NESTOR|subset", "nsh", "nse")):
            v[hk], v[hk + "ci"] = span(k, "h")
            v[ek], v[ek + "ci"] = span(k, "Wh")
        for k, key in (("MoE|subset", "ns"), ("MoE|FULL", "nf"), ("NESTOR|FULL", "nnf"), ("NESTOR|subset", "nns")):
            v[key] = cnt(f"EDGE.all.{types[k]}.n", int(bt[k]["n"]), f"n of the run type {name(k)} in {base}.by_type")
        # the short runs: one step count besides the 96 steps of the registered adaptation; the lower ends of the MoE subset
        # ranges belong to them only if the slowest and costliest short run stays below the fastest and cheapest 96-step run
        msub = [r for r in runs if r.get("arch") == "MoE" and r.get("kind") == "subset"]
        nsub = [r for r in runs if r.get("arch") == "NESTOR" and r.get("kind") == "subset"]
        steps = sorted({r.get("steps") for r in msub + nsub})
        if len(msub) != int(bt["MoE|subset"]["n"]) or len(nsub) != int(bt["NESTOR|subset"]["n"]) or len(steps) != 2 or steps[1] != 96:
            raise Pending(f"{base}.runs: subset runs with steps {steps}; the approved wording has one shorter step count besides 96")
        sh = [r for r in msub if r["steps"] == steps[0]]
        lg = [r for r in msub if r["steps"] == 96]
        if not (sh and lg and max(r["wall_s"] for r in sh) < min(r["wall_s"] for r in lg) and max(r["energy_Wh"] for r in sh) < min(r["energy_Wh"] for r in lg)):
            raise Pending(f"{base}.runs: the lower ends of the MoE subset ranges do not belong to the runs of {steps[0]} steps only")
        for key, x, how in (("short_max_wall_h", max(r["wall_s"] for r in sh) / 3600, "largest wall_s / 3600 of the short MoE subset runs (check, not printed)"),
                            ("long_min_wall_h", min(r["wall_s"] for r in lg) / 3600, "smallest wall_s / 3600 of the 96-step MoE subset runs (check, not printed)")):
            self.data[f"derived.EDGE.all.{key}"] = round(x, 4)
            self.derived[f"derived.EDGE.all.{key}"] = how
        v["n8"] = cnt("EDGE.all.msub.short_n", len(sh), f"number of MoE subset runs in {base}.runs with the smaller step count")
        v["s8"] = fmt_int(steps[0])                 # printed step count: ledger constant '8' (no recorded field, so its registered meaning stays)
        v["nns8"] = cnt("EDGE.all.nsub.short_n", sum(1 for r in nsub if r["steps"] == steps[0]), f"number of NESTOR subset runs in {base}.runs with the smaller step count")
        kz = "MoE|FULL minus a frozen group (C3 freeze run)"
        fw, fe = (bt[kz].get("wall_s") or {}).get("values") or [], (bt[kz].get("energy_Wh") or {}).get("values") or []
        if int(bt[kz].get("n") or 0) != 1 or len(fw) != 1 or len(fe) != 1:
            raise Pending(f"{base}.by_type: the approved wording has one MoE freeze run of C3")
        v["fzh"] = self.dv("EDGE.all.mfreeze.wall_h", float(fw[0]) / 3600, f"wall_s / 3600 of the run type {name(kz)} in {base}.by_type", 3)
        v["fze"] = self.dv("EDGE.all.mfreeze.energy_Wh", float(fe[0]), f"energy_Wh of the run type {name(kz)} in {base}.by_type", 1)
        # 'the C2 runs are those of Table tab:cost': the core fill builds the C2 rows of that table from the core analysis with its CONFIG edge_rows
        core = Path(FC.CONFIG["analysis_json"])
        if not core.exists():
            raise Pending(f"core analysis {core} missing: the C2 runs cannot be compared with Table tab:cost")
        cr = ((json.load(open(core, encoding="utf-8")).get("blocks") or {}).get("EDGE") or {}).get("runs") or []

        def rows(rr, flt):
            return sorted((r.get("run"), r.get("wall_s"), round(float(r.get("energy_Wh") or 0), 6)) for r in rr
                          if all((r.get(k) in x) if isinstance(x, list) else r.get(k) == x for k, x in flt.items()))
        for key, k in (("COST.moe.sgd", "MoE|C2 SGD"), ("COST.moe.muon", "MoE|C2 Muon")):
            rf, rc = rows(runs, FC.CONFIG["edge_rows"][key]), rows(cr, FC.CONFIG["edge_rows"][key])
            if not rc or rf != rc or len(rf) != int(bt[k]["n"]):
                raise Pending(f"the runs of {key} (Table tab:cost, core analysis) differ from the run type {name(k)} of the full analysis")
        pp = f"{base}.pre_update_check.per_base"
        pb = pu.get("per_base") or {}
        rm = re.fullmatch(r"median over (\d+) MoE subset adaptation runs \(training window\)", str((pb.get("moe_b1") or {}).get("reference")))
        rn = re.fullmatch(r"median over (\d+) NESTOR subset adaptation runs \(training window\)", str((pb.get("nestor") or {}).get("reference")))
        if not str(pu.get("method", "")).startswith("batch window / units") or not rm or not rn or int(rn.group(1)) != 1:
            raise Pending(f"{base}.pre_update_check: method or references without approved wording")
        v.update({"pms": self.n(f"{pp}.moe_b1.wall_s_per_unit", 0), "pme": self.n(f"{pp}.moe_b1.energy_Wh_per_unit", 2),
                  "pmrw": self.n(f"{pp}.moe_b1.ratio_wall_vs_median_run", 3), "pmre": self.n(f"{pp}.moe_b1.ratio_energy_vs_median_run", 3),
                  "pns": self.n(f"{pp}.nestor.wall_s_per_unit", 0), "pne": self.n(f"{pp}.nestor.energy_Wh_per_unit", 2),
                  "pnrw": self.n(f"{pp}.nestor.ratio_wall_vs_median_run", 3), "pnre": self.n(f"{pp}.nestor.ratio_energy_vs_median_run", 3)})
        self.fill(fid, "all", v, [f"{base}.by_type", f"{base}.excluded", f"{base}.runs", f"{base}.pre_update_check", f"core analysis {base}.runs (C2 rows of Table tab:cost)"])

    def guards_text(self):
        fid = "FULL.guards"; kind, k = self.guard_outcome(need_detail=False)
        if kind == "none_cut":
            self.fill(fid, "none_cut", {}, ["queue state: overrun_guard_* status"]); return
        cut = (self.guard_info[k]["detail"] or {}).get("cut_set") or [k]
        self.fill(fid, "cut", {"blk": CONFIG["guard_names"][k], "cut": words_join([CONFIG["guard_names"].get(c, texstr(c)) for c in cut]),
                               "pron": "its runs were" if len(cut) == 1 else "their runs were"}, ["queue state: overrun_guard_* detail"])

    def a2_text(self):
        fid = "FULL.A2"
        acc = self.a2["accepted"]
        bad = [j for j in CONFIG["a2_jobs"] if acc.get(j) is not True]
        if not bad:
            self.fill(fid, "met", {}, ["A-2 acceptance record"]); return
        if len(bad) == len(CONFIG["a2_jobs"]):
            self.fill(fid, "none", {}, ["A-2 acceptance record"]); return
        self.fill(fid, "partial", {"which": words_join([CONFIG["a2_jobs"][j][2] for j in sorted(bad)]), "pron": "its value is" if len(bad) == 1 else "their values are"}, ["A-2 acceptance record"])

    # ------------------------------------------------------------------ post-freeze items D-2 (A-3), D-3 (A-4) and I-4 (Appendix A, Table 8)
    def hos_val(self, key):
        """an item of the hours outside the planned runs (data/hours_outside_state.json), four decimals"""
        v = (self.hos.get("hours") or {}).get(key)
        if not FC.isnum(v):
            raise Pending(f"{CONFIG['hos_json']}: hours.{key} missing")
        return self.dv(f"hours_outside.{key}", float(v), f"{CONFIG['hos_json']} hours.{key}", 4)

    def a3_ok(self, job):
        e = self.a3["jobs"][job]
        return e.get("status") == "done" and all(e["outputs"].get(x) is True for x in CONFIG["a3_outputs"]) and e.get("shim_line_in_log") is True

    def d2_text(self):
        fid = "FULL.dev.D2"; jobs = self.a3["jobs"]
        ok = [j for j in CONFIG["a3_jobs"] if self.a3_ok(j)]; bad = [j for j in CONFIG["a3_jobs"] if j not in ok]
        miss = [j for j in CONFIG["a3_jobs"] if not FC.isnum(jobs[j].get("hours"))]
        if miss:
            raise Pending("A-3 re-run record: no hours for " + ", ".join(CONFIG["a3_jobs"][j][1] for j in miss))
        vals = {"hfail": self.hos_val("failed_start_audit"),
                "hrun": self.dv("A3.run_hours", sum(float(jobs[j]["hours"]) for j in CONFIG["a3_jobs"]), "sum of jobs.*.hours of the A-3 re-run record (CONFIG a3_result_json)", 4)}
        fields = ["A-3 re-run record: jobs.*.status, outputs, shim_line_in_log, hours", f"{CONFIG['hos_json']}: hours.failed_start_audit"]
        if not bad:
            self.fill(fid, "done", vals, fields); return
        if not ok:
            self.fill(fid, "none", vals, fields); return
        vals.update({"ok": words_join([CONFIG["a3_jobs"][j][1] for j in ok]), "bad": words_join([CONFIG["a3_jobs"][j][1] for j in bad])})
        self.fill(fid, "partial", vals, fields)

    def d3_text(self):
        fid = "FULL.dev.D3"; acc = self.a4["accepted"]; hrs = []
        for j in CONFIG["a4_jobs"]:
            e = self.state.get(j) if isinstance(self.state, dict) else None
            if not (isinstance(e, dict) and e.get("status") in ("done", "failed") and FC.isnum(e.get("hours"))):
                raise Pending(f"queue state: no final entry with hours for the C3 freeze run of {CONFIG['a4_jobs'][j][0]}")
            hrs.append(float(e["hours"]))
        vals = {"hfirst": self.hos_val("first_attempts_c3_freeze"),
                "hrep": self.dv("A4.repeat_hours", sum(hrs), "sum of the queue-state hours of the two C3 freeze runs repeated under A-4", 4)}
        fields = ["A-4 acceptance record: accepted", "queue state: hours of the two C3 freeze runs", f"{CONFIG['hos_json']}: hours.first_attempts_c3_freeze"]
        good = [j for j in CONFIG["a4_jobs"] if acc[j] is True]; bad = [j for j in CONFIG["a4_jobs"] if acc[j] is not True]
        if not bad:
            self.fill(fid, "met", vals, fields); return
        if not good:
            self.fill(fid, "none", vals, fields); return
        okn = words_join([CONFIG["a4_jobs"][j][1] for j in good])
        vals.update({"Ok": okn[0].upper() + okn[1:], "bad": words_join([CONFIG["a4_jobs"][j][1] for j in bad])})
        self.fill(fid, "partial", vals, fields)

    def i4_text(self):
        self.fill("FULL.dev.I4", "repeated", {"hi4": self.hos_val("interrupted_attempt_dose")}, [f"{CONFIG['hos_json']}: hours.interrupted_attempt_dose"])

    def i5i6_text(self):
        self.fill("FULL.dev.I5I6", "repeated", {"hi5": self.hos_val("interrupted_attempt_c1de_i5"), "hi6": self.hos_val("interrupted_attempt_c1de_i6")},
                  [f"{CONFIG['hos_json']}: hours.interrupted_attempt_c1de_i5", f"{CONFIG['hos_json']}: hours.interrupted_attempt_c1de_i6"])

    def i7_text(self):
        self.fill("FULL.dev.I7", "repeated", {"hi7": self.hos_val("interrupted_attempt_audit_i7")}, [f"{CONFIG['hos_json']}: hours.interrupted_attempt_audit_i7"])

    def hos_text(self):
        hh = self.hos["hours"]
        items = words_join([ph for keys, ph in CONFIG["hos_phrases"] if any(k in hh for k in keys)])
        self.fill("FULL.hos", "list", {"items": items}, [f"{CONFIG['hos_json']}: hours (items, CONFIG hos_phrases)"])

    def tab8_audit(self):
        ok = [j for j in CONFIG["a3_jobs"] if self.a3_ok(j)]; bad = [j for j in CONFIG["a3_jobs"] if j not in ok]
        if not bad:
            return TAB8["audit.done"]
        if not ok:
            return TAB8["audit.none"]
        return subst(TAB8["audit.partial"], {"ok": words_join([CONFIG["a3_jobs"][j][1] for j in ok]), "bad": words_join([CONFIG["a3_jobs"][j][1] for j in bad])})

    def tab8_c3(self):
        acc = self.a4["accepted"]; bad = [j for j in CONFIG["a4_jobs"] if acc[j] is not True]
        if not bad:
            return TAB8["C3.met"]
        if len(bad) == len(CONFIG["a4_jobs"]):
            return TAB8["C3.none"]
        return subst(TAB8["C3.partial"], {"bad": words_join([CONFIG["a4_jobs"][j][1] for j in bad])})

    def fig_c3_caption(self):
        fid = "FIG.c3layers.caption"; base = "blocks.C3_per_layer.per_base"; b = self.blk("C3_per_layer"); P = T[fid]["parts"]
        pb = b.get("per_base") or {}
        if not all(k in pb for k in ("moe_b1", "nestor")):
            raise Pending(f"{base}: moe_b1 or nestor missing")
        n_units = {k: len(pb[k].get("per_unit") or {}) for k in ("moe_b1", "nestor")}
        if n_units != {"moe_b1": 12, "nestor": 6}:
            raise Pending(f"{base}: unit counts {n_units}, registered 12 and 6")
        dom = {k: sorted(pb[k].get("domains") or []) for k in ("moe_b1", "nestor")}
        wds = {k: words_join([CORPUS.get(d, texstr(d)) for d in v]) for k, v in dom.items()}
        domains = wds["moe_b1"] if dom["moe_b1"] == dom["nestor"] else f"{wds['moe_b1']} for MoE base~1 and {wds['nestor']} for \\nestor{{}}"
        opt = ""
        if str(((((self.doc.get("blocks") or {}).get("C3") or {}).get("NESTOR_label")) or {}).get("label") or "").startswith("exploratory"):
            opt += P["exploratory"]
        fewer = []
        for k, nm in (("moe_b1", "MoE base~1"), ("nestor", "\\nestor{}")):
            for j, (u, y) in enumerate((pb[k].get("per_unit") or {}).items()):
                if FC.isnum(y.get("n_domains")) and y["n_domains"] < len(dom[k]):
                    fewer.append(f"layer {j + 1} of {nm}" if k == "moe_b1" else f"block {int(str(u).lstrip('b')) + 1} of {nm}")
        if fewer:
            opt += subst(P["fewer"], {"units": words_join(fewer)})
        missing_runs = max(8 - len(dom["moe_b1"]) - len(dom["nestor"]), len(b.get("dropped") or []))
        if missing_runs > 0:
            t_ = subst(P["dropped"], {"k": self.dv("FIG.c3.runs_not_shown", missing_runs, "8 planned runs (4 domains, 2 bases) minus the domains of blocks.C3_per_layer.per_base.*.domains, or len(dropped)", kind="int")})
            opt += t_.replace(" planned runs are not shown", " planned run is not shown") if missing_runs == 1 else t_
        v = {"DOMAINS": domains, "N_RUNS_B1": self.dv("FIG.c3.n_runs_moe_b1", len(dom["moe_b1"]), "len(blocks.C3_per_layer.per_base.moe_b1.domains)", kind="int"),
             "N_RUNS_NESTOR": self.dv("FIG.c3.n_runs_nestor", len(dom["nestor"]), "len(blocks.C3_per_layer.per_base.nestor.domains)", kind="int"), "OPTIONAL_C3": opt}
        self.fill(fid, "figure", v, [f"{base}.*.domains", f"{base}.*.per_unit", "blocks.C3_per_layer.dropped", "blocks.C3.NESTOR_label"])

    def fig_c4_caption(self):
        fid = "FIG.c4windows.caption"; base = "blocks.C4"; b = self.blk("C4"); P = T[fid]["parts"]
        pdr = b.get("per_draw") or []
        ws_ = {r.get("W") for r in pdr}
        if not pdr or len(ws_) != 1:
            raise Pending(f"{base}.per_draw: window counts {sorted(map(str, ws_))}")
        k_ = len(b.get("dropped") or []) + sum(1 for r in pdr if r.get("beta_C4") is None)
        dropped = P["none"] if k_ == 0 else subst(P["dropped"], {"k": self.dv("FIG.c4.dropped", k_, "len(blocks.C4.dropped) + draws without beta_C4", kind="int")})
        if k_ == 1:
            dropped = dropped.replace(" draws were dropped", " draw was dropped")
        for key, nm, lab_ in (("per_base", "moe_b1", "MoE base~1"), ("per_base", "moe_b2", "MoE base~2"), (None, "overall", "all draws")):
            summ = ((b.get(key) or {}).get(nm) if key else b.get(nm)) or {}
            if (summ.get("bootstrap") or {}).get("status"):
                dropped += subst(P["noboot"], {"name": lab_})
        bpath = next((f"{base}.{k}.bootstrap.B" for k in ("overall", "per_base.moe_b1", "per_base.moe_b2") if FC.isnum(FC.get(self.doc, f"{base}.{k}.bootstrap.B"))), None)
        if bpath is None:
            raise Pending(f"{base}: no bootstrap B in any summary")
        v = {"N_DRAWS": self.i(f"{base}.overall.n"), "N_B1": self.i(f"{base}.per_base.moe_b1.n"), "N_B2": self.i(f"{base}.per_base.moe_b2.n"),
             "W": self.i(f"{base}.per_draw[0].W"), "B": self.i(bpath), "DROPPED_C4": dropped}
        self.fill(fid, "figure", v, [f"{base}.overall", f"{base}.per_base", f"{base}.per_draw", f"{base}.dropped"])

    def d4_text(self):
        fid = "FULL.dev.D4"
        if self.labels.get("C5") != CONFIG["c5_not_evaluable"]:
            raise Pending(f"deviation D-4 applies only when block C5 is not evaluable as registered (C5 label {self.labels.get('C5')!r})")
        self.fill(fid, "recorded", {}, ["deviation record v19, D-4 section 7 (a)", "label of FULL.C5.result"])

    def summary(self):
        fid = "FULL.summary"; L = self.labels
        missing = [k for k in ("B3", "dose", "M12b", "C1_DE", "C5") if k not in L]
        if missing:
            raise Pending(f"labels missing for {missing}")
        names = {"B3": "the third base", "dose": "the dose-response", "M12b": "M12-b", "C1_DE": "the exploratory C1-DE", "C5": "the C5 forecasts"}
        lab = [f"``{texstr(L[k])}'' for {names[k]}" for k in ("B3", "dose", "M12b", "C1_DE", "C5") if L[k] and L[k] != "cut" and not (k == "C5" and L[k] == CONFIG["c5_not_evaluable"])]
        cut = [names[k] for k in ("M12b", "dose", "C1_DE") if L[k] == "cut"]
        items = words_join(lab)
        if cut:
            items += "; " + words_join(cut) + (" was" if len(cut) == 1 else " were") + " cut under the registered overrun rule"
        if not L["B3"]:
            items += "; the third base has no usable draws under rule E1"
        if L["C5"] == CONFIG["c5_not_evaluable"]:                          # deviation record v19, D-4 section 7 (d)
            items += "; the C5 forecasts are not evaluable as registered, since no forecast was formed (deviation D-4)"
        aud, lim = "", CONFIG["word_limits"].get(fid)
        nw = lambda it: len(FC.words(subst(T[fid]["variants"]["labels"], dict(items=it, aud=""))))
        if lim is not None and nw(items) > lim and "for the exploratory C1-DE" in items:   # operator, 20 November: the article goes, the C5 clause stays
            items = items.replace("for the exploratory C1-DE", "for exploratory C1-DE"); self.warn.append("FULL.summary: 'the' before 'exploratory C1-DE' left out to keep the limit of %d words" % lim)
        if lim is not None and nw(items) > lim:
            raise Pending("FULL.summary: %d words above the limit of %d without the audit clause" % (nw(items), lim))
        if (self.fills.get("FULL.audit") or {}).get("variant") in ("both", "one"):   # Table tab:fullaudit has rows
            n_ = len(FC.words(subst(T[fid]["variants"]["labels"], {"items": items, "aud": T[fid]["parts"]["aud"]})))
            if lim is None or n_ <= lim:
                aud = T[fid]["parts"]["aud"]
            else:
                self.warn.append(f"FULL.summary: the reference to Table tab:fullaudit would make {n_} words, above the limit of {lim}; left out")
        self.fill(fid, "labels", {"items": items, "aud": aud}, ["labels of FULL.B3.result, FULL.dose.result, FULL.M12b.result, FULL.C1DE.result, FULL.C5.result",
                                                              "variant of FULL.audit (reference to Table tab:fullaudit within the word limit)"])

    # ------------------------------------------------------------------ generated tables (each number appears once in the paper)
    def tab_wrap(self, label, caption, spec, head, rows, size="small", x=False):
        env, beg = ("tabularx", f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}") if x else ("tabular", f"\\begin{{tabular}}{{{spec}}}")
        L = ["% generated by scripts/fill_full.py from the full analysis output -- do not edit", "\\begin{table}[ht]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\{size}", beg, "\\toprule", head + "\\\\", "\\midrule"]
        L += [r if r.startswith("\\midrule") else r + "\\\\" for r in rows]
        L += ["\\bottomrule", f"\\end{{{env}}}", "\\end{table}", ""]
        return "\n".join(L)

    def tables_all(self):
        d, num = self.doc.get("blocks") or {}, FC.num

        def ivl(x, nd=3):
            if not isinstance(x, dict) or not FC.isnum(x.get("mean")):
                return "--"
            s = num(x["mean"], nd)
            if FC.isnum(x.get("lo")) and FC.isnum(x.get("hi")):
                s += f" [{num(x['lo'], nd)}, {num(x['hi'], nd)}]"
            elif isinstance(x.get("range"), list) and len(x["range"]) == 2 and all(FC.isnum(v) for v in x["range"]):
                s += f" ({num(x['range'][0], nd)} to {num(x['range'][1], nd)})"
            return s
        rows = []
        for key, name in (("B3", "Third base"), ("C1_DE", "C1-DE (exploratory)")):
            if self.labels.get(key if key != "C1_DE" else "C1_DE") in (None, "cut") and key in self.labels:
                continue
            for r in (d.get(key) or {}).get("per_draw") or []:
                rows.append(f"{name} & {CONFIG['bases'].get(r.get('base'), texstr(r.get('base')))} & {r.get('d')} & {num(r.get('I'))} & {num(r.get('P'))} & {num(r.get('J'))}")
        if rows:
            self.tables["draws"] = self.tab_wrap("tab:fulldraws", "Values per draw of the third base and of the exploratory block C1-DE, as written by the full analysis; $J_d=I_d-\\beta_0P_d$ with the registered margins.", "@{}llrrrr@{}",
                                                 "Block & Base & $d$ & $I_d$ & $P_d$ & $J_d$", rows)
        rec = ((d.get("dose") or {}).get("record") or {}).get("bases") or {}
        units = {(u.get("base"), u.get("d")): u for u in ((((d.get("dose") or {}).get("fits") or {}).get("raw") or {}).get("units") or []) if u.get("level") == "dose"}
        rows = []
        if self.labels.get("dose") not in (None, "cut"):
            for bb, x in rec.items():
                for e in x.get("examined") or []:
                    u = units.get((bb, e.get("d"))) if e.get("accepted") else None
                    rows.append(f"{CONFIG['bases'].get(bb, texstr(bb))} & {e.get('d')} & {e.get('seed')} & {num(e.get('P_selection'))} & {num(e.get('P_d'))} & {'yes' if e.get('accepted') else 'no'} & {num(u.get('I')) if u else '--'}")
        if rows:
            self.tables["dose"] = self.tab_wrap("tab:fulldose", "Candidate draws of the dose-response block: selection-half and estimation-half $P$, acceptance (selection-half $P$ in [0.35, 0.65]) and the crossover $I_d$ of the accepted draws.", "@{}lrrrrlr@{}",
                                                "Base & $d$ & Seed & $P$, selection & $P_d$, estimation & Accepted & $I_d$", rows)
        rows = [f"{r.get('d')} & {num(r.get('IT_learned'))} & {num(r.get('IT_rand'))} & {num(r.get('Delta'))} & {'yes' if r.get('common_haar_draws') else 'no'}"
                for r in ((d.get("M12b") or {}).get("per_draw") or [])] if self.labels.get("M12b") not in (None, "cut") else []
        if rows:
            self.tables["m12b"] = self.tab_wrap("tab:fullm12b", "M12-b per draw on MoE base~1: the target interaction of the learned updates and of the random directions, and their difference $\\Delta_d$; the random directions use the same Haar draws as the books runs.", "@{}rrrrl@{}",
                                                "$d$ & $IT_d^{\\mathrm{learned}}$ & $IT_d^{\\mathrm{rand}}$ & $\\Delta_d$ & Common draws", rows)
        sp = d.get("selpin") or {}
        rows = []
        for key, name in (("C1", "C1"), ("B3", "Third base"), ("C1_DE", "C1-DE (exploratory)")):
            sh = (sp.get(key) or {}).get("shares_over_draws") or {}
            for c in ("code", "math"):
                x = sh.get(c) or {}
                if x:
                    rows.append(f"{name} & {CORPUS[c]} & {ivl(x.get('flips'))} & {ivl(x.get('gate_values'))} & {ivl(x.get('frozen_routing'))}")
        if rows:
            self.tables["selpin"] = self.tab_wrap("tab:fullselpin", "Selection-only pinned replay: shares of $F$ from router flips ($F-F^{\\mathrm{selpin}}$), the gate-value pathway ($F^{\\mathrm{selpin}}-F^{\\mathrm{pin}}$) and the frozen-routing pathway ($F^{\\mathrm{pin}}$), means over draws with 95\\% $t$-intervals.", "@{}llccc@{}",
                                                  "Block & Corpus & Router flips & Gate values & Frozen routing", rows, size="footnotesize")
        c3 = d.get("C3") or {}
        rows = []
        for bb in ("moe_b1", "moe_b2", "nestor"):
            x = (c3.get("per_base") or {}).get(bb) or {}
            if not x:
                continue
            sh = x.get("shares") or {}
            g, g2 = CONFIG["groups"].get(x.get("g"), texstr(x.get("g"))), CONFIG["groups"].get(x.get("g2"), texstr(x.get("g2")))
            rows += [f"{CONFIG['base_long'][bb]} & $\\varphi_g/F$, {g} & {ivl(sh.get('share_g'))}", f" & $\\varphi_{{g'}}/F$, {g2} & {ivl(sh.get('share_g2'))}",
                     f" & $\\iota/F$ & {ivl(sh.get('iota_over_F'))}"]
            for vv, y in (x.get("single_revert_Delta_over_F") or {}).items():
                rows.append(f" & $\\Delta/F$, revert of the {CONFIG['variants'].get(vv, texstr(vv))} & {ivl(y)}")
        for arch, name in (("moe", "MoE base~1"), ("nestor", "\\nestor{}")):
            y = (c3.get("freeze_cross_check") or {}).get(arch) or {}
            if FC.isnum(y.get("freeze_attribution")):
                rows.append(f"Freeze check, {name} & freeze minus revert attribution & {num(y.get('freeze_minus_revert_attribution'), 4)} (freeze {num(y.get('freeze_attribution'), 4)}, revert {num(y.get('revert_attribution'), 4)})")
        if rows:
            self.tables["c3"] = self.tab_wrap("tab:fullc3", "C3 endpoint revert decomposition of the DCLM forgetting of the full-adaptation runs: shares, interaction and single-group revert deltas as means over domains, with 95\\% $t$-intervals where seeds allow and the range over domains otherwise, and the freeze cross-check.", "@{}>{\\raggedright\\arraybackslash}p{2.4cm}>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{4.4cm}@{}",
                                              "Base & Quantity & Value", rows, x=True)
        cl = (d.get("C3_per_layer") or {}).get("per_base") or {}
        rows = []
        for bb, x in cl.items():
            for k, (unit, y) in enumerate((x.get("per_unit") or {}).items()):
                uname = f"layer {k + 1}" if bb != "nestor" else f"block {int(str(unit).lstrip('b')) + 1}"   # blocks numbered from 1, as in Figure 2 (pre-registration: 0-5)
                rows.append(f"{CONFIG['base_long'].get(bb, texstr(bb))} & {uname} & {num(y.get('Delta_mean'), 4)} & {ivl({'mean': y.get('Delta_over_F_mean'), 'range': y.get('Delta_over_F_range')})}")
        if rows:
            self.tables["c3layer"] = self.tab_wrap("tab:fullc3layer", "C3 single-layer reverts: means over domains of $\\Delta_\\ell=F-F(\\text{revert }\\ell)$ and of $\\Delta_\\ell/F$ with the range over domains (one full-adaptation run per domain, no interval); units are numbered as in Figure~\\ref{fig:c3layers}.", "@{}llrl@{}",
                                                   "Base & Unit & $\\Delta_\\ell$ & $\\Delta_\\ell/F$", rows)
        sc = ((d.get("forecasts") or {}).get("scores")) or {}
        bf, rw = sc.get("by_forecaster") or {}, sc.get("R2_within") or {}
        rows = []
        for key, name in CONFIG["forecasters"]:
            x = (bf.get(key) or {}).get("pooled") or {}
            if x:
                rows.append(f"{name} & {num(x.get('R2_oos'))} & {num(x.get('spearman'))} & {num(x.get('median_abs_log_ratio'))} & {num(x.get('calibration_slope'))} & {x.get('n')} & {num((rw.get(key) or {}).get('R2_within'))}")
        if rows:
            blk_rows = [f"{name} & " + " & ".join(num(((bf.get(key) or {}).get(b) or {}).get("R2_oos")) for b, _ in CONFIG["forecast_blocks"]) for key, name in CONFIG["forecasters"] if bf.get(key)]
            self.tables["forecasts"] = (self.tab_wrap("tab:fullforecasts", "Scores of all eight registered forecasters, pooled over the blocks: $R^2_{\\mathrm{oos}}$ against the benchmark of the training-fold mean, Spearman's $\\rho$, median $|\\log(\\hat F/F)|$, calibration slope, number of scored values and $R^2_{\\mathrm{within}}$ on the crossover contrasts; descriptive, no forecaster is selected.", "@{}lrrrrrr@{}",
                                                      "Forecaster & $R^2_{\\mathrm{oos}}$ & $\\rho$ & Median & Slope & $n$ & $R^2_{\\mathrm{within}}$", rows)
                                        + self.tab_wrap("tab:fullforecastblocks", "$R^2_{\\mathrm{oos}}$ of every registered forecaster per block (descriptive); rows of C2 runs after their stop step are not scored.", "@{}l" + "r" * len(CONFIG["forecast_blocks"]) + "@{}",
                                                        "Forecaster & " + " & ".join(n_ for _, n_ in CONFIG["forecast_blocks"]), blk_rows))
        bd = d.get("budget") or {}
        pb = bd.get("per_block") or {}
        order = [k for k in CONFIG["budget_names"] if k in pb] + [k for k in pb if k not in CONFIG["budget_names"]]
        rows = [f"{CONFIG['budget_names'].get(k, texstr(k))} & {num(pb[k].get('planned_gpu_h'), 2)} & {num(pb[k].get('actual_gpu_h'), 2)} & {pb[k].get('jobs_with_hours')}" for k in order]
        if rows:
            t = bd.get("total") or {}
            rows += ["\\midrule", f"Total & {num(t.get('planned_gpu_h'), 2)} & {num(t.get('actual_gpu_h'), 2)} & "]
            self.tables["budget"] = self.tab_wrap("tab:fullbudget", "Planned GPU-hours of the frozen plan and measured GPU-hours per block of the registered queue, with the number of runs and evaluations that recorded hours.", "@{}lrrr@{}",
                                                  "Block & Planned & Measured & Jobs", rows)
        if getattr(self, "audit_have", None):
            self.fid = "TAB.fullaudit"; self.nums[self.fid] = []
            self.tables["audit"] = audit_table(*self.audit_rows())
            self.fid = None
        self.tables["status"] = self.block_status()

    def block_status(self):
        d = self.doc.get("blocks") or {}
        app = "Appendix~\\ref{app:full}"

        def lab(key, kind="label"):
            if self.labels.get(key) == "cut":
                return "cut under the registered overrun rule (Appendix~\\ref{app:power})"
            if self.labels.get(key):
                return f"{kind} ``{texstr(self.labels[key])}'' ({app})"
            if key == "B3" and "B3" in self.labels:
                return f"no usable draws under rule E1 ({app})"
            return "\\resultTBD{FULL.status." + key + "}"
        sp = d.get("selpin") or {}
        sel = [n for k, n in (("C1", "C1"), ("B3", "third-base"), ("C1_DE", "C1-DE")) if sp.get(k)]
        rows = [("C1 selection-pinned evaluations (all C1 runs)", "main (part of C1)", f"descriptive, for the {words_join(sel)} runs ({app})" if sel else "\\resultTBD{FULL.status.selpin}"),
                ("C3 revert decomposition, with its DCLM-continuation runs and single-layer reverts", "main", (f"descriptive ({app}); random partial-revert control not run" if (d.get("C3") or {}).get("per_base") else "\\resultTBD{FULL.status.C3}") + self.tab8_c3()),
                ("Assumption audit at $\\theta_0$", "main", self.tab8_audit() + TAB8["audit.I7"]),
                ("C5: French run, drive run and scoring of the forecasters", "estimation", (TAB8["C5.not_evaluable"] if self.labels.get("C5") == CONFIG["c5_not_evaluable"] else (f"forecasts ``{texstr(self.labels['C5'])}'' ({app})" if self.labels.get("C5") else "\\resultTBD{FULL.status.C5}"))),
                ("Third base: C1 on books and its evaluation at $\\theta_0$", "secondary", lab("B3")),
                ("Dose-response", "secondary / exploratory", lab("dose") + (TAB8["dose.I4"] if self.labels.get("dose") != "cut" else "")),
                ("M12-b: learned-versus-random interaction at fixed subsets", "secondary / exploratory", lab("M12b")),
                ("C1-DE", "exploratory", lab("C1_DE", "descriptive label") + (TAB8["C1DE.I5I6"] if self.labels.get("C1_DE") != "cut" else "")),
                ("SEC: DCLM-tilted draws and embedding-row runs", "secondary / exploratory", "no runs in the frozen plan (gate G0-4)"),
                ("\\nestor{}-books", "dropped at gate G1-3", "not run (gate outcome)")]
        for r in rows:
            self.status[r[0]] = r[2]
        L = ["% generated by scripts/fill_full.py from the full analysis output and the confirm queue state -- do not edit",
             "\\begin{table}[t]", "\\centering",
             "\\caption{Registered blocks outside the core and their status. The tier is that of the frozen plan (Section~7 of the frozen pre-registration), which orders the blocks for budget cuts; the core is reported in the main text (Section~\\ref{sec:prereg}) and the other blocks in Appendix~\\ref{app:full}.}",
             "\\label{tab:blockstatus}", "\\small", "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}Xl>{\\raggedright\\arraybackslash}p{5.4cm}@{}}", "\\toprule",
             "Block & Tier & Status\\\\", "\\midrule"] + [f"{a} & {b} & {c}\\\\" for a, b, c in rows] + ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]
        return "\n".join(L)


def audit_table(r5, r6, r0):
    """Table tab:fullaudit: three panels (B5, B6, A0) in one float"""
    L = ["% generated by scripts/fill_full.py from the full analysis output -- do not edit", "\\begin{table}[ht]", "\\centering",
         f"\\caption{{{AUDIT_CAPTION}}}", "\\label{tab:fullaudit}", "\\footnotesize"]
    for k, (spec, head, rows) in enumerate(zip(AUDIT_SPECS, AUDIT_HEADS, (r5, r6, r0))):
        if k:
            L.append("\\par\\medskip")
        L += [f"\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}", "\\toprule", head + "\\\\", "\\midrule"] + [r + "\\\\" for r in rows] + ["\\bottomrule", "\\end{tabularx}"]
    L += ["\\end{table}", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- ledger
def ledger_entries(fills_text, f, BL):
    """one 'token@fills/full_fills.tex' entry per number of the fills file, as build_ledger.py tokenizes it"""
    rel = CONFIG["out"]["fills"]
    first = {}
    for fid, v in f.fills.items():
        for r in v.get("numbers") or []:
            first.setdefault(BL.norm(r["token"]), (fid, r))
    out, unmapped = {}, []
    for t in sorted({m.group(0) for m in BL.TOKEN.finditer(BL.visible(fills_text))}):
        k = BL.norm(t)
        if k in first:
            fid, r = first[k]
            field = r["field"]
            how = f"derived: {f.derived[field]}" if field in f.derived else f"{FIELDS} {field}"
            out[f"{t}@{rel}"] = {"source": CONFIG["out"]["data"], "field": field, "meaning": f"scripts/fill_full.py, {fid}: {how}"}
        elif t in CONSTANTS or k in CONSTANTS:
            out[f"{t}@{rel}"] = {"source": "scripts/fill_full.py template text (registered constant or notation)", "field": "text (constant)",
                                 "meaning": CONSTANTS.get(t) or CONSTANTS.get(k)}
        else:
            unmapped.append(t)
    return out, unmapped


# ----------------------------------------------------------------------------- one-time integration (anchored, idempotent)
APPENDIX_TEX = r"""% written once by scripts/fill_full.py --integrate; every value below is a fill of scripts/fill_full.py
\section{Registered blocks outside the core}\label{app:full}
This appendix reports the registered blocks outside the core, analysed with the same frozen code as the core (Section~\ref{sec:prereg}); none of them enters the Holm family, and no decision depends on them. Table~\ref{tab:blockstatus} gives the status of every block. \resultTBD{FULL.guards}

\paragraph{Third base.}
\resultTBD{FULL.B3.result}

\paragraph{Dose-response.}
\resultTBD{FULL.dose.result}
\resultTBD{FULL.dose.details}

\paragraph{M12-b.}
\resultTBD{FULL.M12b.result}

\paragraph{C1-DE.}
\resultTBD{FULL.C1DE.result}

\paragraph{Selection-only pinned replay.}
\resultTBD{FULL.selpin.result}

\paragraph{C3.}
\resultTBD{FULL.C3.result}
\resultTBD{FULL.C3.wd}
\resultTBD{FULL.C3.layer}
\resultTBD{FULL.Frel.result}
%%FIG_C3%%

\paragraph{C5 and the forecasters.}
\resultTBD{FULL.C5.result}
\resultTBD{FULL.forecasts.result}

\paragraph{Assumption audit.}
\resultTBD{FULL.audit}

\IfFileExists{tables/full_draws.tex}{\input{tables/full_draws}}{\resultTBD{TAB.fulldraws}}
\IfFileExists{tables/full_dose.tex}{\input{tables/full_dose}}{\resultTBD{TAB.fulldose}}
\IfFileExists{tables/full_m12b.tex}{\input{tables/full_m12b}}{\resultTBD{TAB.fullm12b}}
\IfFileExists{tables/full_selpin.tex}{\input{tables/full_selpin}}{\resultTBD{TAB.fullselpin}}
\IfFileExists{tables/full_c3.tex}{\input{tables/full_c3}}{\resultTBD{TAB.fullc3}}
\IfFileExists{tables/full_c3layer.tex}{\input{tables/full_c3layer}}{\resultTBD{TAB.fullc3layer}}
\IfFileExists{tables/full_forecasts.tex}{\input{tables/full_forecasts}}{\resultTBD{TAB.fullforecasts}}
\IfFileExists{tables/full_audit.tex}{\input{tables/full_audit}}{\resultTBD{TAB.fullaudit}}
"""
# an appendix written by an earlier version of this script gets the input of the audit table after the forecasts table
AUDIT_TAB_ANCHOR = r"\IfFileExists{tables/full_forecasts.tex}{\input{tables/full_forecasts}}{\resultTBD{TAB.fullforecasts}}"
AUDIT_TAB_LINE = r"\IfFileExists{tables/full_audit.tex}{\input{tables/full_audit}}{\resultTBD{TAB.fullaudit}}"

# text changes of the integration: (file, old text or alternatives, new text, note)
TEXT_CHANGES = [
    ('sections/06_results.tex', '\\resultTBD{C1.result}\n% C1.result: registered outcome labels', '\\resultTBD{C1.result} \\resultTBD{FULL.audit.C1}\n% C1.result: registered outcome labels', 'Section 5.3, C1 results paragraph: the assumption-audit labels reported with C1 (prereg v1.3 sec. 4.3; v1.6.10)'),
    ("sections/01_introduction.tex", "a secondary replication on a third base, pretrained on three times as many tokens, is reported separately.",
     "a secondary replication on a third base, pretrained on three times as many tokens, is reported in Appendix~\\ref{app:full}.", "pointer"),
    ("sections/03_account.tex", "which is reported separately (Section~\\ref{sec:secondary})", "which is reported in Appendix~\\ref{app:full}", "pointer"),
    ("sections/04_design.tex", "The relative-forgetting version of C1 needs the DCLM-continuation run of base~2, which was not complete at submission, and is reported separately.",
     "The relative-forgetting version of C1 uses the DCLM-continuation runs and is reported in Appendix~\\ref{app:full}.", "replaces a status sentence (de-logging check, p. 13)"),
    ("sections/04_design.tex", "keep their registered designs and analyses and are reported separately; Table~\\ref{tab:blockstatus} gives their status at submission.",
     "keep their registered designs and analyses and are reported in Appendix~\\ref{app:full}; Table~\\ref{tab:blockstatus} gives their status.", "replaces a status sentence (de-logging check, p. 13)"),
    ("sections/04_design.tex", "are analysed afterwards on the same code and reported separately.", "are analysed with the same code and reported in Appendix~\\ref{app:full}.", "pointer"),
    ("sections/06_results.tex", "C1-DE, an exploratory block that uses such pairs, is reported separately (Section~\\ref{sec:secondary}).",
     "C1-DE, an exploratory block that uses such pairs, is reported in Appendix~\\ref{app:full}.", "pointer"),
    ("sections/06_results.tex", "the step-8 equivalence analysis of the Muon arm of C2; the relative-forgetting version of C1, the third base, the assumption audit and the further blocks of Section~\\ref{sec:secondary} are reported separately.",
     "the step-8 equivalence analysis of the Muon arm of C2.\n\\resultTBD{FULL.summary}", "main-text summary slot (labels only)"),
    ("sections/07_discussion.tex", "the replication of C1 on base~3, pretrained on 1.5B tokens, is reported separately together with the assumption audit, so this paper tests C1 at one training stage,",
     "the replication of C1 on base~3, pretrained on 1.5B tokens, is a secondary block with three draws (Appendix~\\ref{app:full}), so the confirmatory test of C1 covers one training stage,", "Discussion, factual update (operator to approve)"),
    ("sections/07_discussion.tex", "they were still in progress when this paper was submitted and will be reported in full, whatever their outcome, under the registered protocol.",
     "they are reported in Appendix~\\ref{app:full} with their registered labels.", "replaces a status sentence (de-logging check, p. 26)"),
    ("sections/A_prereg.tex", "this paper reports the core analysis; the other registered blocks are reported separately, and Table~\\ref{tab:blockstatus} gives their status at submission",
     "this paper reports the core analysis in the main text; the other registered blocks are reported in Appendix~\\ref{app:full}, and Table~\\ref{tab:blockstatus} gives their status", "pointer"),
    ("sections/A_prereg.tex", ("The three values are not part of this paper and are reported with block C3 only if the acceptance rule is met.",
                               "The acceptance check of A-2, the last step of its installation procedure (the final input check and the acceptance lines of the three evaluations), applies when the three C3 evaluations run at queue positions 137, 142 and 147, after submission; their values are not part of this paper, and they are reported with block C3 only if the check passes."),
     "\\resultTBD{FULL.A2}", "A-2 acceptance slot (v1.6.7 wording of the A-2 paragraph; the v1.6.6 sentence is accepted as a fallback)"),
    ("sections/C_power_budget.tex", "\\filledor{COMPUTE.used}{The measured GPU-hours of the post-freeze runs, block by block against the plan, are reported with the full analysis.}\n",
     "\\filledor{COMPUTE.used}{The measured GPU-hours of the post-freeze runs, block by block against the plan, are reported with the full analysis.}\n\\resultTBD{FULL.compute}\n\\IfFileExists{tables/full_budget.tex}{\\input{tables/full_budget}}{\\resultTBD{TAB.fullbudget}}\n", "Appendix C budget slots"),
    ('sections/04_design.tex', 'After the freeze, two amendments corrected the evaluation code of the weight-decay-only control, which no confirmatory statistic uses, two interrupted runs were repeated from their start with the same seeds,', 'After the freeze, two amendments corrected the evaluation code of the weight-decay-only control, which no confirmatory statistic uses, two further amendments restored a function that the assumption audit calls and exempted one parameter with an exactly zero gradient from the bitwise audit of the two C3 freeze runs, five interrupted runs, one of them twice, were repeated from their start with the same seeds and settings,', 'Section 4.6, deviation sentence: one clause for amendments A-3 and A-4 (descriptive names, no numbers); the interrupted runs become five, one of them repeated twice (D-1, I-1, I-4, I-5 with I-6, I-7)'),
    ('sections/04_design.tex', 'and the registered step-1 value of the Muon manipulation check was not recorded; none of these changed a confirmatory statistic', 'the registered step-1 value of the Muon manipulation check was not recorded, and no French forecast was formed, so block C5 is not evaluable as registered; none of these changed a confirmatory statistic', 'Section 4.6, deviation sentence: clause for deviation D-4 (v1.6.10)'),
    ('sections/A_prereg.tex', ['After the freeze the study has one deviation in a control evaluation with two code amendments (D-1, A-1 and A-2), three execution incidents (I-1 to I-3) and one reporting deviation (R-1).', 'After the freeze the study has three deviations with four code amendments, one in a control evaluation (D-1, A-1 and A-2), one in the assumption audit (D-2, A-3) and one in the bitwise audit of the C3 freeze runs (D-3, A-4), seven execution incidents (I-1 to I-7) and one reporting deviation (R-1).'], 'After the freeze the study has four deviations, three of them with four code amendments, one in a control evaluation (D-1, A-1 and A-2), one in the assumption audit (D-2, A-3) and one in the bitwise audit of the C3 freeze runs (D-3, A-4), and one without an amendment in the C5 forecasts (D-4); seven execution incidents (I-1 to I-7); and one reporting deviation (R-1).', 'Appendix A, opening count of the post-freeze items (v1.6.10: deviation D-4)'),
    ('sections/A_prereg.tex', 'and used 0.0124 GPU-hours, which brings the hours outside the planned runs to 1.3646 GPU-hours (Appendix~\\ref{app:power}).', 'and used 0.0124 GPU-hours, which count towards the hours outside the planned runs (Appendix~\\ref{app:power}).', 'Appendix A, paragraph A-2: the running total of the hours outside the planned runs is stated once, in Appendix C (rule 11)'),
    ('sections/A_prereg.tex', ['\\paragraph{Theory statements.}', '\\paragraph{Deviation D-2 and amendment A-3.}\n\\resultTBD{FULL.dev.D2}\n\n\\paragraph{Deviation D-3 and amendment A-4.}\n\\resultTBD{FULL.dev.D3}\n\n\\paragraph{Incident I-4.}\n\\resultTBD{FULL.dev.I4}\n\n\\paragraph{Incidents I-5 and I-6.}\n\\resultTBD{FULL.dev.I5I6}\n\n\\paragraph{Incident I-7.}\n\\resultTBD{FULL.dev.I7}\n\n\\paragraph{Theory statements.}'], '\\paragraph{Deviation D-2 and amendment A-3.}\n\\resultTBD{FULL.dev.D2}\n\n\\paragraph{Deviation D-3 and amendment A-4.}\n\\resultTBD{FULL.dev.D3}\n\n\\paragraph{Deviation D-4.}\n\\resultTBD{FULL.dev.D4}\n\n\\paragraph{Incident I-4.}\n\\resultTBD{FULL.dev.I4}\n\n\\paragraph{Incidents I-5 and I-6.}\n\\resultTBD{FULL.dev.I5I6}\n\n\\paragraph{Incident I-7.}\n\\resultTBD{FULL.dev.I7}\n\n\\paragraph{Theory statements.}', 'Appendix A, paragraphs D-2 with A-3, D-3 with A-4, D-4, I-4, I-5 with I-6, and I-7 after R-1 (v1.6.10: D-4 after D-3)'),
    ('sections/C_power_budget.tex', 'After the freeze, the stopped first attempt of one C1 run (deviation D-1) used 1.1047 GPU-hours and the GPU check of its correction 0.0104 GPU-hours, and the interrupted attempt of one C1 run (incident I-1) used 0.2371 GPU-hours (Appendix~\\ref{app:prereg})\\filledor{HOS.A2}{, and the GPU check of amendment A-2 used 0.0124 GPU-hours}; these \\filledor{HOS.total}{1.3646} GPU-hours lie outside the queue state, so the overrun guards add them to their projections, and where a corrected projection exceeds the envelope at a guard that passed, the registered cut order is applied by hand and logged.', '\\filledor{FULL.hos}{After the freeze, the stopped first attempt of one C1 run (deviation D-1) used 1.1047 GPU-hours and the GPU check of amendment A-1 used 0.0104 GPU-hours, and the interrupted attempt of one C1 run (incident I-1) used 0.2371 GPU-hours (Appendix~\\ref{app:prereg})\\filledor{HOS.A2}{, and the GPU check of amendment A-2 used 0.0124 GPU-hours}; these \\filledor{HOS.total}{--} GPU-hours lie outside the queue state, so the overrun guards add them to their projections, and where a corrected projection exceeds the envelope at a guard that passed, the registered cut order is applied by hand and logged.}', 'Appendix C, sentence on the hours outside the queue state (slot FULL.hos; the old sentence stays as its default, with no number for the old total)'),
    ('tables/budget_final.tex', 'The last row gives the GPU-hours spent after the freeze outside the queue state (deviation D-1, incident I-1 and amendment A-2, Appendix~\\ref{app:prereg}), which the overrun guards add to their projections.', 'The last row gives the total of the GPU-hours spent after the freeze outside the queue state; Appendix~\\ref{app:prereg} states each item.', 'Appendix C, caption of Table tab:budget: the last row gives the total, Appendix A the items (rule 11)'),
    ('tables/budget_final.tex', 'Spent after the freeze outside the queue state: \\filledor{HOS.items}{failed first attempt of a C1 job (1.1047), GPU check of amendment A-1 (0.0104), attempt of a C1 job ended by a restart of the software that launches the jobs (0.2371) and GPU check of amendment A-2 (0.0124)}', 'Spent after the freeze outside the queue state, items in Appendix~\\ref{app:prereg}', 'Appendix C, last row of Table tab:budget: the total only, the items stand in Appendix A (rule 11)'),
    ('sections/A_prereg.tex', 'and the check 0.0104 GPU-hours, together', 'and the GPU check of amendment A-1 used 0.0104 GPU-hours, together', 'Appendix A, paragraph D-1: the GPU check of amendment A-1 named as wherever it is itemized'),
    ('tables/budget_final.tex', 'G0, its re-run, the G0 of the third base, G1 and G2', 'G0, its repetition, the G0 of the third base, G1 and G2', 'Appendix C, first row of Table tab:budget: "its repetition" for "its re-run" (house-style scan)'),
    ('sections/C_power_budget.tex', '19.1 on G0, its re-run, the G0 of the third base, G1 and G2', '19.1 on G0, its repetition, the G0 of the third base, G1 and G2', 'Appendix C, sentence on the GPU-hours spent at the freeze: "its repetition", as in the first row of Table tab:budget'),
    ('tables/budget_final.tex', '\\filledor{HOS.total}{1.3646}', '\\filledor{HOS.total}{--}', 'Appendix C, last row of Table tab:budget: the default of the core slot HOS.total carries no number, so only the final total, filled by scripts/fill_core.py from data/hours_outside_state.json, is stated'),
    ('sections/F_compute.tex', 'it changes no run and reads existing logs on the CPU, and in this paper it reports the core runs, while the costs of the runs that follow the core are reported with them.',
     'it changes no run and reads existing logs on the CPU. Table~\\ref{tab:cost} reports it for the core runs, and the paragraph on all covered runs below reports it for every run type.', 'Appendix F, paragraph on the cost measurements: what the cost block reports (v1.6.11, final check M1)'),
    ('sections/F_compute.tex', 'a run enters only if consecutive power samples inside its window are at most 5\\,s apart, with excluded runs listed.',
     'a run enters only if consecutive power samples inside its window are at most 5\\,s apart; the one excluded run is named in the paragraph on all covered runs.', 'Appendix F, paragraph on the cost measurements: the excluded run (v1.6.11, M1)'),
    ('sections/F_compute.tex', '\n\n\\paragraph{Peak memory of full, all-expert and subset adaptation.}', '\n\n\\paragraph{All covered runs.}\n\\resultTBD{FULL.EDGE.all}\n\n\\paragraph{Peak memory of full, all-expert and subset adaptation.}',
     'Appendix F: new paragraph on all covered runs (slot FULL.EDGE.all) before the paragraph on peak memory (v1.6.11, M1)'),
]

SCRIPT_PATCHES = [
    ("sections/macros.tex", "\\InputIfFileExists{fills/core_fills.tex}{}{}",
     "\\InputIfFileExists{fills/core_fills.tex}{}{}\n\\InputIfFileExists{fills/full_fills.tex}{}{}% remaining blocks: scripts/fill_full.py"),
    ("scripts/build_ledger.py", '    walk(ROOT / "main.tex")\n    return order',
     '    walk(ROOT / "main.tex")\n    ff = ROOT / "fills" / "full_fills.tex"  # fills of the remaining blocks (scripts/fill_full.py): scanned for the ledger\n    if ff.exists() and ff not in seen:\n        seen.add(ff); order.append(ff)\n    return order'),
    ("scripts/check_paper.py", '        if unknown or bad: hard.append("fills")\n',
     '        if unknown or bad: hard.append("fills")\n'
     '    ffp, fmp = ROOT / "fills" / "full_fills.tex", ROOT / "fills" / "full_manifest.json"   # remaining blocks (scripts/fill_full.py)\n'
     '    if ffp.exists():\n'
     '        ftxt = ffp.read_text(encoding="utf-8"); ids = re.findall(r"\\\\setfill\\{([^}]*)\\}", ftxt)\n'
     '        man = json.loads(fmp.read_text(encoding="utf-8")) if fmp.exists() else {}\n'
     '        sh = re.search(r"sha256 ([0-9a-f]{64})", ftxt)\n'
     '        slots = set(tbd) | {m.group(1) for s in vis.values() for m in re.finditer(r"\\\\filledor\\{([^}]*)\\}", s)}\n'
     '        unknown = [i for i in ids if i not in slots]\n'
     '        bad = (not sh) or sh.group(1) != man.get("sha256") or sorted(ids) != sorted(man.get("fills", {}))\n'
     '        print(f"[8] full fills: {len(ids)} IDs from sha256 {sh.group(1)[:12] if sh else None}; not a slot in the text: {unknown}; manifest mismatch: {bad}")\n'
     '        if unknown or bad: hard.append("full fills")\n'),
    ("scripts/write_ledger_md.py", "def main():",
     "_PF = ROOT / \"scripts\" / \"placeholders_full.json\"   # slots of the remaining blocks (scripts/fill_full.py)\n"
     "if _PF.exists():\n    PLACEHOLDERS.update({k: (\"full\", v) for k, v in json.loads(_PF.read_text(encoding=\"utf-8\")).items()})\n\n\ndef main():"),
    ("scripts/write_ledger_md.py", '{"core": 0, "author": 1, "plan/freeze": 2}', '{"core": 0, "full": 1, "author": 2, "plan/freeze": 3}'),
    # the slot HOS.items leaves the text (TEXT_CHANGES, last row of Table tab:budget), so the core fill stops writing it
    ("scripts/fill_core.py", '        self.put("HOS.items", names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1], src)\n',
     '        # HOS.items: scripts/fill_full.py --integrate removes its slot (rule 11: Appendix A states the items, the last row of\n        # Table tab:budget only their total), so the item list is not written\n'),
    ("scripts/fill_core.py", '            for fid in ("HOS.total", "HOS.items", "HOS.A2"):\n', '            for fid in ("HOS.total", "HOS.A2"):\n'),
]


def placeholder_descriptions():
    out = {fid: f"scripts/fill_full.py from `{FIELDS}`: {t['block']}; {t['rule']}" for fid, t in T.items()}
    for tid, key in TABLE_IDS.items():
        out[tid] = f"scripts/fill_full.py from `{FIELDS}`: generated table {CONFIG['out'][key]} (bookkeeping slot, printed only if the table is missing)"
    for key in ("B3", "dose", "M12b", "C1_DE", "C5", "selpin", "C3"):
        out[f"FULL.status.{key}"] = f"scripts/fill_full.py: Table 8 status cell of {key}, printed only if the block has no result in the full analysis"
    out["FULL.status.C5"] += "; when the hashed forecast file holds no forecast, the cell gives the C5 status of deviation D-4 (not evaluable as registered)"
    return out


C4_OLD = "The per-window difference-in-differences C4 within DCLM uses the C1 runs. \\resultTBD{C4.result}"
C4_NEW = "The per-window difference-in-differences C4 within DCLM uses the C1 runs (Figure~\\ref{fig:c4windows}). \\resultTBD{C4.result}\n"


def fig_env(key, root=None):
    """figure environment of a registered figure; its caption is the fill slot CONFIG['figures'][key]['slot']"""
    f = CONFIG["figures"][key]
    return ("\\begin{figure}[t]%\n\\centering\n\\includegraphics[width=\\linewidth]{" + f["file"] + "}\n\\caption{\\resultTBD{" + f["slot"] + "}}\n\\label{"
            + f["label"] + "}\n\\end{figure}\n")


def figures_missing(root):
    return [v["file"] for v in CONFIG["figures"].values() if not (Path(root) / v["file"]).exists()]


def integrate(root):
    miss = figures_missing(root)
    if miss:
        raise SystemExit(f"[fill_full] --integrate: registered figure file(s) missing: {miss} (C3 single-layer reverts and C4 per-window results, "
                         "prereg v1.3 sec. 4.8, produced by the figure scripts from the full-mode output); nothing written")
    staged, log = {}, []

    def current(rel):
        return staged[rel] if rel in staged else (root / rel).read_text(encoding="utf-8")

    def apply(rel, olds, new):
        s = current(rel)
        if new in s:
            log.append(f"{rel}: already applied: {new[:60]!r}"); staged[rel] = s; return
        olds = [olds] if isinstance(olds, str) else list(olds)
        hit = sorted((o for o in olds if s.count(o) == 1), key=len, reverse=True)   # the most specific anchor first (v1.6.10)
        if not hit:
            raise SystemExit(f"[fill_full] --integrate: anchor not found exactly once in {rel}: {olds[0][:90]!r}; nothing written")
        staged[rel] = s.replace(hit[0], new, 1)
        log.append(f"{rel}: {new[:60]!r}")
    for rel, old, new in SCRIPT_PATCHES:
        apply(rel, old, new)
    for rel, old, new, _ in TEXT_CHANGES:
        apply(rel, old, new)
    apply("sections/06_results.tex", C4_OLD, C4_NEW + fig_env("c4windows", root))
    if (root / CONFIG["appendix_file"]).exists():         # appendix of an earlier version: add the audit table (anchored, idempotent)
        apply(CONFIG["appendix_file"], AUDIT_TAB_ANCHOR, AUDIT_TAB_ANCHOR + "\n" + AUDIT_TAB_LINE)
    ms = current("main.tex"); inc = "\\input{" + CONFIG["appendix_file"][:-4] + "}"
    if inc not in ms:
        anchor = "\\input{sections/H_exploratory}"
        if ms.count(anchor) != 1:
            raise SystemExit("[fill_full] --integrate: \\input{sections/H_exploratory} not found exactly once in main.tex; nothing written")
        ms = ms.replace(anchor, inc + "\n" + anchor, 1) if CONFIG["appendix_position"] == "before_exploratory" else ms.replace(anchor, anchor + "\n" + inc, 1)
        staged["main.tex"] = ms; log.append(f"main.tex: {inc} ({CONFIG['appendix_position']})")
    for rel, s in staged.items():                        # every anchor was found: write
        FC.write_atomic(root / rel, s)
    if not (root / CONFIG["appendix_file"]).exists():
        FC.write_atomic(root / CONFIG["appendix_file"], APPENDIX_TEX.replace("%%FIG_C3%%\n", fig_env("c3layers", root))); log.append(f"{CONFIG['appendix_file']}: created")
    FC.write_atomic(root / CONFIG["out"]["placeholders"], json.dumps(placeholder_descriptions(), indent=1, ensure_ascii=False) + "\n")
    lsp = root / "scripts" / "ledger_sources.json"; ls = json.loads(lsp.read_text(encoding="utf-8"))
    gen = ls.setdefault("generated_tables", {})
    for key in ("status", "draws", "dose", "m12b", "selpin", "c3", "c3layer", "forecasts", "budget", "audit"):
        rel = CONFIG["out"][key]
        gen[rel] = f"{rel}, generated by scripts/fill_full.py from {FIELDS} and results/queue_state_confirm.json (written after the registered queue)"
    FC.write_atomic(lsp, json.dumps(ls, indent=1, ensure_ascii=False) + "\n")
    log.append("scripts/ledger_sources.json: generated tables registered; scripts/placeholders_full.json written")
    return log


# ----------------------------------------------------------------------------- template catalogue (templates_remaining_blocks.md)
def write_templates(path):
    L = ["# Reporting templates for the remaining registered blocks", "",
         "Scratch file, not part of the paper. Written by `scripts/fill_full.py --write-templates` from the template table that the fill itself uses, so the two cannot differ. `[[name]]` marks a value that the fill takes from the full analysis output (`results/analysis/analysis_confirm.json`, mode full); the exact JSON field of every printed number is recorded per fill in `fills/full_manifest.json` and becomes a `token@fills/full_fills.tex` ledger entry. Each variant is the sentence for one registered outcome. The fill selects it by the label that the analysis wrote and stops if that label disagrees with the registered rule applied to the printed numbers. No variant interprets a result or adds a claim beyond the registered rule.", "",
         "The templates follow the eleven house-style rules of the de-logging check of v1.6.6, with rule 6 as overridden (a deviation is described without a sequence of events, with its reason stated plainly in one clause, and a defect is not presented as a property of the method). In practice: findings in the past tense with estimate, interval, number of draws and registered label; runs named by base, draw, subset, target and seed, never by job identifier or queue position; no dates, times or GPU-hours outside Appendix C; exclusions as counts under rule E1; the word exploratory in every sentence that carries a value of C1-DE; no status language; every number stated once, so the descriptive blocks (selection-only pinned replay, C3, forecasters, budget) point to their generated tables instead of repeating them.", "",
         "## Placement and length", "",
         "| Slot | Block | Where | Expected length |", "|---|---|---|---|"]
    for fid, t in T.items():
        L.append(f"| `{fid}` | {t['block']} | {t['where']} | {t['length']} |")
    L += ["", "Main text: `FULL.summary` (Section 5.5) adds about 2 lines in place of a clause, half a line of it for the clause that refers to Table `tab:fullaudit` (57 of 60 words on the mock), and the shorter Discussion and status sentences remove about 1.5 lines. The registered C4 per-window figure (prereg v1.3 sec. 4.8 makes it a main-text figure) follows the C4 result in Section 5.5; at a height of about 50 mm it adds about 14 lines. Everything else goes into the new appendix (`sections/H_full.tex`, label `app:full`, before the exploratory appendix, which becomes Appendix I; every reference to an appendix of this paper is a `\\ref`, so no letter is typed by hand) and into Appendix C (`FULL.compute` with the projections of the overrun guards, Table `tab:fullbudget`). `FULL.guards` closes the opening paragraph of the new appendix and names the outcome only. Table 8 (`tables/block_status.tex`) is regenerated by the fill. The new appendix holds about 1.7 pages of text, ten generated tables (the tenth is Table `tab:fullaudit` of the assumption audit) and the C3 single-layer figure; the abstract and the conclusions do not change. The assumption audit adds about 37 lines (about 0.6 page) to the new appendix, so the mock build grows to about 63 pages (estimate from the text, not built).", "",
          "Inputs that stop the run if they are missing: `--integrate` stops before writing anything if `figures/fig_c3_layers.pdf` or `figures/fig_c4_windows.pdf` does not exist (both produced by the figure scripts from the full-mode output; their captions are the fill slots `FIG.c3layers.caption` and `FIG.c4windows.caption`, whose templates are the caption drafts). The fill stops if the A-2 acceptance record `results\\a2_acceptance.json` is missing, is not exactly the two objects `accepted` and `checks` (with the optional strings `rule` and `utc` that `confirm/post_queue/a2_acceptance.py` writes), or lacks a true/false value for one of the three evaluations; it also stops if the analysis output does not match its `.sha256` file or is not in mode full. For the assumption audit it stops if `blocks.audit` does not hold, for each base of the A-3 re-run record, either the registered result of the frozen audit analysis (exact keys; subsets A and B; corpora DCLM, code and math; 2,000 window resamples; registered labels only) or the analysis's 'not available' status, if a base has results while the A-3 record lacks its registered outputs or the reverse, and if a label differs from the registered rule applied to its numbers. A degenerate audit result that the frozen code allows (no label, fewer than 24 groups, fewer valid resamples) leaves `FULL.audit` and its table without a fill, with the reason in the manifest.", "",
          "Post-freeze items of the final version: `FULL.dev.D2`, `FULL.dev.D3`, `FULL.dev.D4`, `FULL.dev.I4`, `FULL.dev.I5I6` and `FULL.dev.I7` are new paragraphs of Appendix A after R-1 (deviation D-2 with amendment A-3, deviation D-3 with amendment A-4, deviation D-4, incident I-4, incidents I-5 and I-6, incident I-7), each in the four parts of house-style rule 5 with its reason in one clause. `--integrate` adds their headings and slots, updates the opening count of Appendix A, states the total of the hours outside the planned runs once (Appendix C, no longer in the A-2 paragraph), adds one clause without numbers to the deviation sentence of Section 4.6 and makes its interrupted runs five, one of them repeated twice, with the same seeds and settings, and makes the Appendix C sentence on these hours the slot `FULL.hos`, which names the items without numbers; Appendix A states each item with its hours and the last row of Table `tab:budget` only their total. Table 8 gets the cells listed under 'Table 8 cells'. Expected added length: measured on the mock build of v1.6.8 with A-3 done and A-4 met. Main text: the Section 4.6 sentence grows from 5 to 7 lines; on the mock the next pages absorb the 2 lines before p. 21, and Author contributions stay at p. 27, line 20. Appendix A: about 37 lines (opening count +1, Table 8 cells +7 table lines with C3 +2, audit +2 and dose-response +3, A-2 paragraph 0, D-2 10, D-3 12, I-4 7); its last page (p. 40) then ends about 3 lines above the foot, so the partial variants (+1 line each) still fit, and Appendix B stays at p. 41. Appendix C: 5 lines on that build (FULL.hos +3, Table 14 caption -1, its last row +3), which moved Table 17 from p. 50 to p. 52 and the start of Appendix H from p. 54 to p. 55. Not yet built, estimated from the text: the last row of Table 14 without its items is 3 lines shorter than on that build (1 line shorter than in v1.6.8), COMPUTE.used after the final-build patch about 1.5 lines shorter (132 to 103 words), FULL.hos unchanged, and the D-1 cost sentence in Appendix A about a third of a line longer; Appendix C is then within about 1 line of its v1.6.8 length, and Appendix A keeps about 3 lines to spare on its last page. Total: 61 pages, unchanged. Incidents I-5 to I-7, estimated from the text (not built): FULL.dev.I5I6 about 11 lines and FULL.dev.I7 about 6 lines in Appendix A, and the Table 8 notes for C1-DE and the assumption audit about 3 and 4 table lines, together more than the about 3 lines left on the last page of Appendix A, so Appendix A runs onto a new page and Appendix B and every later page move by one (about 62 pages on the mock); FULL.hos about 12 lines in Appendix C; the Section 4.6 sentence about a third of a line longer than in v4, at most 1 line of main text; the opening count of Appendix A keeps its length.", "",
          "Fill-time order for the hours outside the planned runs: add the six items to `data/hours_outside_state.json` from the deviation record (keys `failed_start_audit` with record D-2, `first_attempts_c3_freeze` with record D-3, `interrupted_attempt_dose` with record I-4, `interrupted_attempt_c1de_i5` with record I-5 and `interrupted_attempt_c1de_i6` with record I-6 and `interrupted_attempt_audit_i7` with record I-7, each with a label in the style of the existing ones, `the stopped starts of the assumption audit`, `the first attempts of the two C3 freeze runs`, `the attempt of a dose-response job ended by a restart of the software that launches the jobs`, `the attempt of a C1-DE job ended by a restart of the software that launches the jobs` and `the attempt of a C1-DE job ended by a restart of the computer` and `the attempt of an assumption-audit job ended by a restart of the computer`, and the new total), apply `final_build_patch_compute_used.py` (rule 11: `COMPUTE.used` then states the core's own 1.3646 GPU-hours for deviation D-1 and amendment A-1, incident I-1 and amendment A-2 and points to the last row of Table `tab:budget`, which gives only the total, and the manual ledger entry of 1.3646 names these items), run `scripts/fill_core.py`, whose slot `HOS.total` (the last row of Table `tab:budget`) then carries the final total, and then `scripts/fill_full.py`, which stops if an item of these six lacks its record or label, the total differs from the sum of the items, the core fills carry another total or `COMPUTE.used` still lists the items. The A-3 re-run record (`confirm/rerun/rerun_result.json`) and the A-4 acceptance record (`results/a4_acceptance.json`) are read as strictly as the A-2 record; a missing or malformed record stops the fill.", "",
          "## Templates", ""]
    for fid, t in T.items():
        L += [f"### `{fid}`: {t['block']}", "", f"- Where: {t['where']}; expected length: {t['length']}.", f"- Registered rule: {t['rule']}.", ""]
        if not t["variants"]:
            L += ["Placeholder only; no template text is written before the audit runs under amendment A-3 have ended.", ""]
            continue
        for var, txt in t["variants"].items():
            L += [f"Variant `{var}`:", "", "```latex", txt + (t.get("tail", "") if var not in ("cut", "no usable draws") else ""), "```", ""]
        for k, txt in (t.get("parts") or {}).items():
            L += [f"Part `{k}`:", "", "```latex", txt, "```", ""]
    L += ["## Table 8 cells of the post-freeze items", "", "Generated in `tables/block_status.tex`, not fills: the status of the assumption audit from the A-3 re-run record, a note on the C3 row from the A-4 acceptance record and a note on the dose-response row (incident I-4).", ""]
    for k, txt in TAB8.items():
        L += [f"Cell `{k}`:", "", "```latex", txt, "```", ""]
    L += ["## Table `tab:fullaudit` of the assumption audit (generated)", "", "Written by the fill into `tables/full_audit.tex` from `blocks.audit`: "
          "panel (a) one row per base, panel (b) one row per base and subset, panel (c) one row per base and corpus, every value with three "
          "decimals and the count of cells with $Q\\le 0$ as an integer; each printed value is recorded with its JSON field under "
          "`table_numbers` of `TAB.fullaudit` in `fills/full_manifest.json`. A base without outputs has no rows, and with no outputs on "
          "either base the table is empty. Caption:", "", "```latex", AUDIT_CAPTION, "```", "", "Column heads:", ""]
    L += [f"- `{h}`" for h in AUDIT_HEADS] + [""]
    L += ["## Text changes applied by `--integrate`", "", "The old texts are the v1.6.7 wording; each change is applied only if its old text occurs exactly once.", ""]
    for rel, old, new, note in TEXT_CHANGES:
        L += [f"`{rel}` ({note}). Old:", "", "```latex", old if isinstance(old, str) else old[0], "```", "", "New:", "", "```latex", new, "```", ""]
    L += [f"`sections/06_results.tex` (registered C4 per-window figure, main text). Old:", "", "```latex", C4_OLD, "```", "", "New (the caption is a fill slot):", "", "```latex",
          C4_NEW + fig_env("c4windows", Path("__no_caption_dir__")), "```", "", "`sections/H_full.tex` carries the C3 single-layer figure after the C3 paragraph:", "", "```latex",
          fig_env("c3layers", Path("__no_caption_dir__")), "```", ""]
    L += ["Script and macro changes of `--integrate`: " + "; ".join(sorted({f'`{rel}`' for rel, _, _ in SCRIPT_PATCHES})) + "; `scripts/ledger_sources.json` registers the generated tables; `scripts/placeholders_full.json` holds the slot descriptions for `claims_ledger.md`; `main.tex` inputs `sections/H_full`.", "",
          "## Registered constants printed by the template text", "", "| Token | Meaning |", "|---|---|"] + [f"| {k} | {v} |" for k, v in CONSTANTS.items()] + [""]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def load_a2(path):
    """the A-2 acceptance record, exactly {"accepted": {job: bool for the three evaluations}, "checks": {...}}, optionally with the string
    fields "rule" and "utc" that confirm/post_queue/a2_acceptance.py writes; any defect stops the fill"""
    if not path or not os.path.exists(path):
        raise SystemExit(f"[fill_full] A-2 acceptance record {path} is missing; nothing written")
    try:
        rec = json.load(open(path, encoding="utf-8"))
    except ValueError as e:
        raise SystemExit(f"[fill_full] A-2 acceptance record {path} is not valid JSON ({e}); nothing written")
    if (not isinstance(rec, dict) or set(rec) not in ({"accepted", "checks"}, {"accepted", "checks", "rule", "utc"}) or not isinstance(rec.get("accepted"), dict)
            or not isinstance(rec.get("checks"), dict) or not all(isinstance(rec[k], str) for k in ("rule", "utc") if k in rec)):
        raise SystemExit(f"[fill_full] A-2 acceptance record {path}: expected exactly the objects 'accepted' and 'checks' (and the strings 'rule' and 'utc'), found "
                         f"{sorted(rec) if isinstance(rec, dict) else type(rec).__name__}; nothing written")
    want, acc = set(CONFIG["a2_jobs"]), rec["accepted"]
    missing, extra = sorted(want - set(acc)), sorted(set(acc) - want)
    nonbool = sorted(k for k in acc if not isinstance(acc[k], bool))
    if missing or extra or nonbool:
        raise SystemExit(f"[fill_full] A-2 acceptance record {path}: keys missing {missing}, unknown {extra}, not true/false {nonbool}; nothing written")
    return rec


def _load_json(path, what):
    if not path or not os.path.exists(path):
        raise SystemExit(f"[fill_full] {what} {path} is missing; nothing written")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except ValueError as e:
        raise SystemExit(f"[fill_full] {what} {path} is not valid JSON ({e}); nothing written")


def load_a3(path):
    """the A-3 re-run record of the assumption audit (confirm/rerun/watch_and_rerun.ps1), exactly {utc, run_queue_exit, dry_run: false,
    jobs: {job: {status, rc, hours, outputs: {the three registered files: bool}, shim_line_in_log: bool}}}; any defect stops the fill"""
    what = "A-3 re-run record"; rec = _load_json(path, what)
    if not isinstance(rec, dict) or set(rec) != {"utc", "run_queue_exit", "dry_run", "jobs"} or not isinstance(rec.get("jobs"), dict):
        raise SystemExit(f"[fill_full] {what} {path}: expected exactly utc, run_queue_exit, dry_run and jobs, found "
                         f"{sorted(rec) if isinstance(rec, dict) else type(rec).__name__}; nothing written")
    if rec["dry_run"] is not False:
        raise SystemExit(f"[fill_full] {what} {path}: dry_run is {rec['dry_run']!r}, so no audit ran; nothing written")
    want, jobs = set(CONFIG["a3_jobs"]), rec["jobs"]
    missing, extra = sorted(want - set(jobs)), sorted(set(jobs) - want)
    keys, outs = {"status", "rc", "hours", "outputs", "shim_line_in_log"}, set(CONFIG["a3_outputs"])
    bad = sorted(j for j in want & set(jobs) if not (isinstance(jobs[j], dict) and set(jobs[j]) == keys and isinstance(jobs[j]["outputs"], dict)
                                                       and set(jobs[j]["outputs"]) == outs and all(isinstance(v, bool) for v in jobs[j]["outputs"].values())
                                                       and isinstance(jobs[j]["shim_line_in_log"], bool)))
    if missing or extra or bad:
        raise SystemExit(f"[fill_full] {what} {path}: jobs missing {missing}, unknown {extra}, malformed {bad}; nothing written")
    return rec


def load_a4(path):
    """the A-4 acceptance record (confirm/rerun_a4/a4_acceptance.py), exactly {accepted: {job: bool for the two C3 freeze runs}, checks,
    rule, utc}; any defect stops the fill"""
    what = "A-4 acceptance record"; rec = _load_json(path, what)
    if not isinstance(rec, dict) or set(rec) != {"accepted", "checks", "rule", "utc"} or not isinstance(rec.get("accepted"), dict) or not isinstance(rec.get("checks"), dict):
        raise SystemExit(f"[fill_full] {what} {path}: expected exactly accepted, checks, rule and utc, found "
                         f"{sorted(rec) if isinstance(rec, dict) else type(rec).__name__}; nothing written")
    want, acc = set(CONFIG["a4_jobs"]), rec["accepted"]
    missing, extra = sorted(want - set(acc)), sorted(set(acc) - want)
    nonbool = sorted(k for k in acc if not isinstance(acc[k], bool))
    if missing or extra or nonbool:
        raise SystemExit(f"[fill_full] {what} {path}: runs missing {missing}, unknown {extra}, not true/false {nonbool}; nothing written")
    return rec


def load_expl(root):
    """the exploratory values of the paper tree (gpu_hours_total for the measured total of FULL.compute); a missing or invalid file leaves that slot without a fill"""
    p = Path(root) / CONFIG["expl_json"]
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def load_hos(root):
    """the hours outside the planned runs with the post-freeze items of CONFIG hos_new, one phrase per item and a total equal to the sum
    of the items; any defect stops the fill"""
    what = CONFIG["hos_json"]; hos = _load_json(str(Path(root) / what), what)
    hh = hos.get("hours") if isinstance(hos, dict) else None
    if not isinstance(hh, dict) or not hh or not all(FC.isnum(v) for v in hh.values()):
        raise SystemExit(f"[fill_full] {what}: 'hours' must map each item to a number; nothing written")
    rec = hos.get("records") or {}
    missing = [k for k in CONFIG["hos_new"] if k not in hh]
    wrong = [k for k, r in CONFIG["hos_new"].items() if k in hh and rec.get(k) != r]
    lab = hos.get("labels") or {}
    nolabel = [k for k in CONFIG["hos_new"] if k in hh and not (isinstance(lab.get(k), str) and lab[k].strip())]
    if missing or wrong or nolabel:
        raise SystemExit(f"[fill_full] {what}: post-freeze items missing {missing}, record differs from CONFIG hos_new {wrong}, without a label {nolabel}; add the items of "
                         "D-2, D-3 and I-4 to I-7 with their records and labels from the deviation record and run scripts/fill_core.py before the full fill; nothing written")
    groups = [keys for keys, _ in CONFIG["hos_phrases"]]
    uncovered = [k for k in hh if sum(k in g for g in groups) != 1]
    partial = [list(g) for g in groups if 0 < sum(k in hh for k in g) < len(g)]
    if uncovered or partial:
        raise SystemExit(f"[fill_full] {what}: items without exactly one phrase in CONFIG hos_phrases {uncovered}, incomplete phrases {partial}; nothing written")
    tot = sum(float(v) for v in hh.values())
    if not FC.isnum(hos.get("total")) or abs(float(hos["total"]) - tot) > 5e-5:
        raise SystemExit(f"[fill_full] {what}: total {hos.get('total')!r} differs from the sum of its items {tot:.4f}; nothing written")
    return hos


def check_core_hos(root, hos):
    """the core fill must carry the same total of the hours outside the planned runs (scripts/fill_core.py run on the updated file),
    and its COMPUTE.used must name the core items without their hours (final-build patch of COMPUTE.used, rule 11)"""
    cf = Path(root) / "fills" / "core_fills.tex"
    m = re.search(r"\\setfill\{HOS\.total\}\{([^}]*)\}", cf.read_text(encoding="utf-8")) if cf.exists() else None
    want = f"{float(hos['total']):.4f}"
    if not m or m.group(1) != want:
        raise SystemExit(f"[fill_full] fills/core_fills.tex has HOS.total {m.group(1) if m else None!r} and {CONFIG['hos_json']} totals {want}: "
                         "run scripts/fill_core.py on the updated hours first; nothing written")
    mc = re.search(r"\\setfill\{COMPUTE\.used\}\{(.*)\}\s*$", cf.read_text(encoding="utf-8"), re.M)
    listed = [k for k, v in hos["hours"].items() if mc and FC.isnum(v) and f"{float(v):.4f} for " in mc.group(1)]
    if listed:   # rule 11: the items stand once, in the last row of Table tab:budget
        raise SystemExit(f"[fill_full] fills/core_fills.tex: COMPUTE.used still lists the items {listed} of {CONFIG['hos_json']}; apply "
                         "final_build_patch_compute_used.py and run scripts/fill_core.py first; nothing written")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="fill the slots of the remaining registered blocks from the full analysis output")
    ap.add_argument("--analysis", default=CONFIG["analysis_json"])
    ap.add_argument("--state", default=CONFIG["state_json"])
    ap.add_argument("--a2-acceptance", default=CONFIG["a2_acceptance_json"], help="A-2 acceptance record {'accepted': {evaluation job: true|false}, 'checks': {...}}")
    ap.add_argument("--a3-result", default=CONFIG["a3_result_json"], help="A-3 re-run record of the assumption audit")
    ap.add_argument("--a4-acceptance", default=CONFIG["a4_acceptance_json"], help="A-4 acceptance record of the two C3 freeze runs")
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--ledger", action="store_true", help="write the token@fills/full_fills.tex entries into scripts/ledger_manual.json")
    ap.add_argument("--integrate", action="store_true", help="one-time structural step (anchored, idempotent)")
    ap.add_argument("--write-templates", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.write_templates:
        write_templates(a.write_templates)
        print(f"[fill_full] templates -> {a.write_templates}")
        return 0
    if a.integrate:
        for line in integrate(root):
            print("[fill_full] integrate:", line)
        return 0
    import build_ledger as BL
    src = Path(a.analysis)
    doc = json.load(open(src, encoding="utf-8"))
    if doc.get("format") != CONFIG["expected_format"]:
        raise SystemExit(f"[fill_full] format {doc.get('format')!r} != {CONFIG['expected_format']!r}; nothing written")
    if doc.get("mode") != CONFIG["expected_mode"]:
        raise SystemExit(f"[fill_full] mode {doc.get('mode')!r} != 'full': the remaining blocks are filled from the full analysis only; nothing written")
    sha = FC.sha256(src)
    side = src.with_name(src.name + ".sha256")
    if side.exists() and side.read_text(encoding="ascii").split()[0] != sha:
        raise SystemExit("[fill_full] analysis_confirm.json does not match its .sha256 file; nothing written")
    state = json.load(open(a.state, encoding="utf-8")) if a.state and os.path.exists(a.state) else {}
    a2 = load_a2(a.a2_acceptance)
    a3, a4, hos = load_a3(a.a3_result), load_a4(a.a4_acceptance), load_hos(root)
    check_core_hos(root, hos)
    if (root / CONFIG["appendix_file"]).exists() and figures_missing(root):
        raise SystemExit(f"[fill_full] registered figure file(s) missing: {figures_missing(root)}; nothing written")
    f = Filler(doc, state, a2, BL)
    f.a3, f.a4, f.hos = a3, a4, hos
    f.expl = load_expl(root)
    f.guards()
    for fid, fn in (("FULL.B3.result", f.b3), ("FULL.dose.result", f.dose), ("FULL.dose.details", f.dose_details), ("FULL.M12b.result", f.m12b),
                    ("FULL.C1DE.result", f.c1de), ("FULL.selpin.result", f.selpin), ("FULL.C3.result", f.c3), ("FULL.C3.wd", f.c3wd),
                    ("FULL.C3.layer", f.c3layer), ("FULL.Frel.result", f.frel), ("FULL.C5.result", f.c5), ("FULL.forecasts.result", f.forecasts), ("FULL.audit", f.audit), ("FULL.audit.C1", f.audit_c1),
                    ("FULL.compute", f.compute), ("FULL.EDGE.all", f.edge_all), ("FULL.guards", f.guards_text), ("FULL.A2", f.a2_text), ("FULL.dev.D2", f.d2_text), ("FULL.dev.D3", f.d3_text), ("FULL.dev.D4", f.d4_text), ("FULL.dev.I4", f.i4_text), ("FULL.dev.I5I6", f.i5i6_text), ("FULL.dev.I7", f.i7_text), ("FULL.hos", f.hos_text), ("FULL.summary", f.summary),
                    ("FIG.c3layers.caption", f.fig_c3_caption), ("FIG.c4windows.caption", f.fig_c4_caption)):
        f.run(fid, fn)
    f.tables_all()
    TAB_BLOCK = {"dose": "dose", "m12b": "M12b"}           # tables of blocks that the overrun rule can cut
    for tid, key in TABLE_IDS.items():
        if key in f.tables:
            f.fills[tid] = {"text": f"generated table {CONFIG['out'][key]}", "fields": ["see the table"], "variant": "table", "numbers": []}
            if f.nums.get(tid):   # number map of a generated table (printed token -> field), kept apart from the ledger numbers of the fills
                f.fills[tid]["table_numbers"] = f.nums[tid]
        elif key == "audit" and (f.fills.get("FULL.audit") or {}).get("variant") == "none":
            f.fills[tid] = {"text": "", "fields": ["A-3 re-run record: no audit outputs on either base"], "variant": "none", "numbers": []}
            f.tables[key] = "% generated by scripts/fill_full.py: no table, the assumption audit has no outputs on either base\n"
        elif key == "audit" and "FULL.audit" in f.pending:
            f.pending[tid] = f"no table without the paragraph FULL.audit ({f.pending['FULL.audit']})"
            f.tables[key] = f"% generated by scripts/fill_full.py: no table without the paragraph FULL.audit\n\\resultTBD{{{tid}}}\n"
        elif f.labels.get(TAB_BLOCK.get(key)) == "cut":
            f.fills[tid] = {"text": "", "fields": ["block cut under the registered overrun rule"], "variant": "cut", "numbers": []}
            f.tables[key] = f"% generated by scripts/fill_full.py: no table, the block was cut under the registered overrun rule\n"
        else:
            f.pending[tid] = "no rows in the analysis output"
            f.tables[key] = f"% generated by scripts/fill_full.py: no rows in the analysis output\n\\resultTBD{{{tid}}}\n"
    for fid, v in f.fills.items():
        FC.tex_escape_check(v["text"], fid)
    out = CONFIG["out"]
    hdr = (f"% generated by scripts/fill_full.py -- do not edit\n% source {src} sha256 {sha}\n% format {doc.get('format')}; mode {doc.get('mode')}; "
           f"analysis created {doc.get('created_utc')}; fills written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    ftxt = hdr + "".join(f"\\setfill{{{k}}}{{{v['text']}}}\n" for k, v in sorted(f.fills.items()))
    entries, unmapped = ledger_entries(ftxt, f, BL)
    if unmapped:
        raise SystemExit(f"[fill_full] numbers in the fills without a recorded field or registered constant: {unmapped}; nothing written")
    data = {"source": str(src), "sha256": sha, "format": doc.get("format"), "mode": doc.get("mode"), "created_utc": doc.get("created_utc")}
    for path, v in sorted(f.data.items()):
        put_path(data, path, v)
    FC.write_atomic(root / out["fills"], ftxt)
    FC.write_atomic(root / out["data"], json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for key in ("status", "draws", "dose", "m12b", "selpin", "c3", "c3layer", "forecasts", "budget", "audit"):
        if key in f.tables:
            FC.write_atomic(root / out[key], f.tables[key])
    man = {"source": str(src), "sha256": sha, "format": doc.get("format"), "mode": doc.get("mode"), "analysis_created_utc": doc.get("created_utc"),
           "analysis_warnings": doc.get("warnings") or [], "not_computable": doc.get("not_computable_from_registered_files") or [],
           "fill_warnings": f.warn, "fills": f.fills, "pending": f.pending, "labels": f.labels, "guards": f.guard_info, "cut": f.cut,
           "a2_acceptance": a2, "a3_rerun": a3, "a4_acceptance": a4,
           "audit": {"bases_with_results": getattr(f, "audit_have", []), "bases_missing": getattr(f, "audit_missing", []), "labels": getattr(f, "audit_labels", {})}, "hours_outside": hos, "derived": f.derived, "table8": f.status, "ledger_entries": len(entries), "tables": {k: out[k] for k in f.tables}}
    FC.write_atomic(root / out["manifest"], json.dumps(man, indent=1, ensure_ascii=False, default=str) + "\n")
    if a.ledger:
        lp = root / "scripts" / "ledger_manual.json"
        lm = json.loads(lp.read_text(encoding="utf-8"))
        old = [k for k in lm if k.endswith("@" + out["fills"])]
        for k in old:
            del lm[k]
        lm.update(entries)
        FC.write_atomic(lp, json.dumps(lm, indent=1, ensure_ascii=False) + "\n")
        print(f"[fill_full] ledger: {len(old)} old and {len(entries)} new entries for {out['fills']}")
    print(f"[fill_full] {len(f.fills)} filled, {len(f.pending)} pending: {sorted(f.pending)}; labels {f.labels}; cut {sorted(f.cut)}; warnings {f.warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
