"""Writes claims_ledger.md from data/claims_ledger.json, scripts/ledger_sources.json and scripts/ledger_manual.json.
Run after scripts/build_ledger.py. Exit code 1 if a \\resultTBD ID in the included TeX has no 'filled from' description."""
import json, re, subprocess, sys, collections, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_ledger as BL  # noqa: E402

DATA_LOCATIONS = {
 "reproduction_stageA.json": "stored version 5d08b209",
 "gate_record_G0.json": "stored version d271c71e",
 "gate_record_b3.json": "g0\\gate_record_b3.json (sha256 fb9ae367...)",
 "entanglement_results.json": "stored version 42dbd3a7",
 "gap_trajectory.json": "stored version 4e616dad",
 "provenance_b15s44.json": "stored version 5d9643c2",
 "ref_verification.json": "CrossRef / arXiv queries, 2026-11-13",
 "g0_rerun_compare.json": "logs\\g0_rerun_compare.json (created 2026-11-13T16:44:45; registered record sha256 9259ddd6..., re-run bdf3d79c...)",
 "hardware_local.json": "nvidia-smi query on the local host, 2026-11-14T00:21:38"
}
GENERATED = "## Generated tables\n\n| Table | Generated from | Fields |\n|---|---|---|\n| `tables/stageA.tex` | `data/reproduction_stageA.json`, `data/provenance_b15s44.json` | `local_rtx4080super[*].ce`, `h100_modal[*].ce`, `paper_table3_5B_tokens_heldout`; differences computed from the `ce` fields; base 3 `eval_h100_fp32.val09/val10.ce` |\n| `tables/g0_gates.tex` | `data/gate_record_G0.json`, `data/gate_record_b3.json` | `gates.G0-1.cells[*].tried[r].per_f[\"0.500000\"]` (n_pairs, predicted.median), `gates.G0-3.result.median`, `gates.G0-4.per_base[*].ratio`, `gates.G0-5.per_base[*].abs_dce`, `gates.G0-6[*].reliability`, `gates.G0-7[*]` (D, SE, n_draws, beta0) |\n| `tables/entanglement_full.tex` | `data/entanglement_results.json` | `cells[\"<base>|c1|<target>|<stream>|f=0.500\"]` (point.xi_LP, xi_LP_ci95, point.rho2_piD, point.rho2_piT_D, point.corr_TD_share), f = 1/3 point.xi_LP, outcome from `agreement.rows` |\n| `tables/gap_summary.tex` | `data/gap_trajectory.json` | `gaps.<gap>.last_half` (mean, mean_ci95_block_bootstrap, slope_nats_per_Btok x 0.25, slope_ci95_pairs_block_bootstrap x 0.25), `train_vs_heldout_agreement.<gap>.heldout_val09/val10` |\n\nHand-written tables carry their sources in the rows of this ledger: `tables/gates_full.tex` (pre-registration v1.2/v1.3 and specification 1.3.0 as summarized in pre-registration v1.3 sec. 6), the power, budget and sign-flip tables in `sections/C_power_budget.tex` and `sections/B_gates.tex` (pre-registration v1.3 sec. 3.3 and 7; v1.2 sec. 14.9, recomputed by exact enumeration), and the cost table in `sections/06b_edge.tex` (placeholders only; EDGE block). The former theory-status and licence tables are prose in v1.1 (Appendices A and F); their sources (theory notes v3/v4, verification v3, change log v1.2; dataset licences table, version 19592d7b) are unchanged, and the two table files stay in `tables/` unused.\n| `tables/stageB.tex` | `data/stageB_window_ci.json` | `per_model[*].forgetting[corpus][0]`, `per_model[*].mean_forgetting[0]`, `per_model[*].mean_gain[0]`, `differences_vs_moe[*].forgetting[corpus]`, `differences_vs_moe[*].mean_forgetting`, `.mean_forgetting_pct`, `.mean_gain`, `paper_table6_5B`; paper differences computed from its rounded values |\n| `tables/stageC.tex` | `data/stageC_results.json` | `rows[*].ce_chunked_eval`, `rows[*].tokens`; MoE minus dense computed from `ce_chunked_eval` || `tables/edgepre.tex` | `data/edge_pre_freeze.json` | `adaptation_96_steps.*`, `oom_attempts_G1[*]`, `stageC[*]` (train segments), `gates_total.*` |\n| `tables/core_draws.tex` (after the analysis lock) | `results/analysis_core/analysis_confirm.json`, by `scripts/fill_core.py` | `C1.per_draw`, `C2.per_draw` |\n| `fills/core_fills.tex` (after the analysis lock) | `results/analysis_core/analysis_confirm.json`, by `scripts/fill_core.py` | listed per ID in `fills/fill_manifest.json` |\n"
PLACEHOLDERS = {  # ID -> (class, source); classes: core (scripts/fill_core.py), author, plan/freeze
    "ABS.C1": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C1.label, C1.estimators.raw.overall; registered sentence, at most 10 words"),
    "ABS.C2": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C2.label, C2.s_star; registered sentence, at most 10 words"),
    "ABS.cost": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs (subset, bases 1-2) and data/edge_pre_freeze.json adaptation_96_steps.moe_full_runs_no_ckpt; at most 12 words; measured values only"),
    "INTRO.C1": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C1.label, C1.test.mean, holm.adjusted.C1"),
    "INTRO.C2": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C2.label, C2.s_star, C2.test.mean, holm.adjusted.C2"),
    "C1.D": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.D, C1.mean_J_per_base[*].n, C1.draws_dropped"),
    "C1.T": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.test.mean, C1.test.p, holm.adjusted.C1"),
    "C1.meanJ": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.mean_J_per_base[*].mean_J"),
    "C1.betas": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.estimators.{raw,pin,sym,rand}.overall (beta_hat, lo, hi)"),
    "C1.controls": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.secondary.I_sym_minus_I_rand, C1.secondary.wd_share.mean_I_WD_over_mean_I, C1.secondary.pinned_share.mean_I_pin_over_mean_I"),
    "C1.label": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C1.label, decisions.C1.out_of_domain_cells"),
    "C1.result": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C1 (label, interpretation, three_statistic_table); registered narratives C1-A to C1-K in the TeX comment"),
    "C2.status": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C2.confirmatory, C2.s_star, C2.D"),
    "C2.manip": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C2.manipulation_check.mean_D_SGD_over_DQ_mean, C2.manipulation_check.D_Muon_step1"),
    "C2.T": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C2.test.mean, C2.test.p, holm.adjusted.C2"),
    "C2.de": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C2.mean_J2_per_base, C2.Delta_e"),
    "C2.label": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C2.label"),
    "C2.result": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: decisions.C2; registered narratives C2-A to C2-F in the TeX comment; step 8 (P-16)"),
    "C4.result": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.C4.overall, blocks.C4.per_base; secondary analysis of the core files; operator to confirm it is in the core list"),
    "EDGE.cost": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs (subset, bases 1-2) against data/edge_pre_freeze.json adaptation_96_steps.moe_full_runs_no_ckpt (descriptive reference)"),
    "COST.moe.subset": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs with arch MoE, kind subset, base moe_b1 or moe_b2: wall_s, energy_Wh, vram_peak_reserved/alloc_gib, tokens_per_s_steady"),
    "COST.moe.sgd": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs with arch MoE, kind C2 SGD (8 steps)"),
    "COST.moe.muon": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs with arch MoE, kind C2 Muon (8 steps)"),
    "COST.moe.subset.noI2": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.EDGE.runs as COST.moe.subset without the runs listed in data/edge_overlap_I2.json (incident I-2)"),
    "CONCL.C1": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: as C1.result"),
    "CONCL.C2": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: as C2.result"),
    "CONCL.edge": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: as EDGE.cost"),
    "TAB.coredraws": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.per_draw, C2.per_draw; writes tables/core_draws.tex (Appendix G)"),
    "C1.matchedgain": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.matched_gain.C1.mean_I_G and per_base (secondary, pre-registration version 8 sec. 3.5)"),
    "C1.code2": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.secondary.second_code_stream (J_test, beta.overall); unadjusted"),
    "BPB.check": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: blocks.bpb.C1_check.max_abs_I_bpb_minus_I (identity check)"),
    "TAB.corevariants": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C1.secondary C1-b, C1-d, C1-e, M12-a; C2.C2_eq; blocks.C2_eq_rand; writes tables/core_variants.tex (Appendix G)"),
    "DISC.interpretation": ("author", "written by the authors from the core outcome, following the registered discussion outline in the TeX comment"),
    "DISC.code": ("author", "written by the authors from the C1 code stream (AlgebraicStack), Python stream alongside"),
    "DISC.release": ("author", "decision on the release policy"),
    "CONCL.code": ("author", "one sentence from DISC.code"),
    "EDGE.C1": ("author", "written by the authors: what the C1 outcome implies for choosing an update set"),
    "EDGE.C2": ("author", "written by the authors: the C2 outcome at step 8, no optimizer-memory statement"),
    "REG.hash": ("plan/freeze", "FREEZE.json sha256 (freeze record)"),
    "REG.date": ("plan/freeze", "freeze timestamp (UTC)"),
    "REG.freeze": ("plan/freeze", "freeze record location and hashes"),
    "REG.deviations": ("plan/freeze", "post-freeze deviation log"),
    "FREEZE.amend": ("plan/freeze", "amendment table for the five files of the G0 code freeze, slots E-1a to E-5e (pre-registration v1.3 version 8, sec. 8)"),
    "POWER.C1": ("plan/freeze", "simulated power of C1 at beta = 0.75 for n_d = 5, from the plan built before the freeze (pre-registration v1.3 version 8, sec. 3.3 and 7)"),
    "BUDGET.total": ("plan/freeze", "planned, spent and total GPU-hours with contingency, from the plan built before the freeze (slots B-*)"),
    "BUDGET.table": ("plan/freeze", "budget table per block, from the plan built before the freeze (28 slots)"),
    "BUDGET.forecast": ("plan/freeze", "GPU-hours of the forecast inputs (probe units and drive runs), slot B-FC"),
    "COMPUTE.used": ("plan/freeze", "GPU-hours spent per block through the core, from the budget ledger"),
    "ENT.NESTORbooks": ("plan/freeze", "NESTOR-books feasibility record (G0-1 rule on nestor|books), written before the freeze by stageC13 of the chain: results/nestor_books_feasibility.json (exists; path confirmed on 2026-11-15); kept as a plan/freeze slot until the operator releases it"),
    "C2.sgd24": ("core", "scripts/fill_core.py from `results/analysis_core/analysis_confirm.json`: C2.sgd_step24_gain (SGD step-24 target gain, prereg v1.3 version 10 sec. 3.2; no decision role; field not yet written by the core analysis)"),
}

def cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")

_PF = ROOT / "scripts" / "placeholders_full.json"   # slots of the remaining blocks (scripts/fill_full.py)
if _PF.exists():
    PLACEHOLDERS.update({k: ("full", v) for k, v in json.loads(_PF.read_text(encoding="utf-8")).items()})


def main():
    led = json.loads((ROOT / "data" / "claims_ledger.json").read_text(encoding="utf-8"))
    src = json.loads((ROOT / "scripts" / "ledger_sources.json").read_text(encoding="utf-8"))
    tbd = sorted({m.group(1) for f in BL.included_files()
                  for m in re.finditer(r"\\resultTBD\{([^}]*)\}", BL.strip_comments(f.read_text(encoding="utf-8")))})
    missing = [t for t in tbd if t not in PLACEHOLDERS]
    if missing:
        print("placeholders without a description:", missing)
        return 1
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_paper.py")], capture_output=True, text=True)
    result = re.search(r"RESULT: (.*)", r.stdout).group(1).strip()
    nchk = re.search(r"curated data entries machine-checked: (\d+)", r.stdout).group(1)
    kinds = collections.Counter(e["kind"] for e in led.values())
    nbyf = sum(len(e.get("by_file", {})) for e in led.values())
    out = ["# Claims ledger: paper v1.6.11", "",
           f"Every number printed in the TeX files that `main.tex` reads (outside comments) is listed in `data/claims_ledger.json` ({len(led)} distinct tokens), "
           f"built by `scripts/build_ledger.py` and checked by `scripts/check_paper.py` (result on {datetime.date.today().isoformat()}: {result}). "
           f"Manual entries name the exact source field, and those that point to a JSON field are re-verified at the printed rounding by the check script ({nchk} entries). "
           f"Where one number means different things in different files, a file-specific manual entry (key `token@file`, {nbyf} entries) gives the source for that file. "
           "Generated tokens appear only in tables written by `scripts/make_tables.py`; the check script regenerates the tables and requires byte-identical output. "
           "Document tokens were found as standalone numbers in a source document, in the priority order of the sources table; occurrences inside identifiers, links, dates, "
           "version strings and heading numbers are skipped. A document match shows that the number occurs in that source; for numbers that recur, such as small constants, "
           "the meaning comes from the section text.", "",
           "## Sources", "", "| Code | Source | Location |", "|---|---|---|"]
    for code, name in src["data"]:
        out.append(f"| data | `data/{name}` | {cell(DATA_LOCATIONS.get(name, 'data/' + name))} |")
    for d in src["docs"]:
        out.append(f"| {d['code']} | {cell(d['desc'])} | {cell(d['location'])} |")
    out += ["", f"Counts: {dict(kinds)}.", "", "## Curated entries (main-text claims)", "",
            "| Printed | Meaning | Source | Field or location | Also | Appears in |", "|---|---|---|---|---|---|"]
    for t, e in led.items():
        rows = []
        if e["kind"] == "manual":
            rows.append((e, [f for f in e["files"] if f not in e.get("by_file", {})]))
        for f, be in e.get("by_file", {}).items():
            rows.append((be, [f]))
        for ent, fl in rows:
            out.append(f"| {cell(t)} | {cell(ent.get('meaning', ''))} | {cell(ent.get('source', ''))} | `{cell(ent.get('field', ''))}` | {cell(ent.get('also', ''))} | "
                       + ", ".join(Path(f).name for f in fl) + " |")
    out += ["", GENERATED.rstrip(), "", "## Placeholders", "",
            "Under the core-first schedule (decision P-16) the paper keeps only core placeholders and plan/freeze values. "
            "Core IDs are written by `scripts/fill_core.py` from the core analysis output; author IDs are written by the authors from the core outcome; "
            "plan/freeze IDs come from the plan and the freeze record.", "",
            "| ID | Class | Filled from |", "|---|---|---|"]
    for t in sorted(tbd, key=lambda t: ({"core": 0, "full": 1, "author": 2, "plan/freeze": 3}[PLACEHOLDERS[t][0]], t)):
        out.append(f"| `{t}` | {PLACEHOLDERS[t][0]} | {cell(PLACEHOLDERS[t][1])} |")
    fm = ROOT / "fills" / "fill_manifest.json"
    out += ["", "## Core fills", ""]
    if fm.exists():
        man = json.loads(fm.read_text(encoding="utf-8"))
        out += [f"Written by `scripts/fill_core.py` from `{man['source']}` (sha256 `{man['sha256']}`, analysis created {man.get('analysis_created_utc')}).", "",
                "| ID | Text | Fields |", "|---|---|---|"]
        for k, v in sorted(man["fills"].items()):
            out.append(f"| `{k}` | {cell(v['text'])} | {cell(', '.join(v['fields']))} |")
        out += ["", "Left pending: " + (", ".join(f"`{k}` ({v})" for k, v in sorted(man["pending"].items())) or "none") + "."]
    else:
        out.append("No fills yet: after the analysis lock, `python scripts/fill_core.py` writes `fills/core_fills.tex` and `fills/fill_manifest.json`, and this section lists every filled value with its source field.")
    by_doc = collections.defaultdict(list)
    for t, e in led.items():
        if e["kind"] == "document":
            by_doc[e["document"]].append(t)
    out += ["", "## Document-matched tokens, by source", ""]
    for d, ts in sorted(by_doc.items(), key=lambda kv: -len(kv[1])):
        out.append(f"**{d}** ({len(ts)}): " + ", ".join(ts) + ".")
        out.append("")
    p = ROOT / "claims_ledger.md"
    tmp = p.with_suffix(".tmp")
    tmp.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    tmp.replace(p)
    print("claims_ledger.md:", len(led), "tokens,", len(tbd), "placeholders,", dict(kinds), "check:", result)
    return 0

if __name__ == "__main__":
    sys.exit(main())
