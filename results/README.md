# Results

Results produced before the confirmatory runs:

| Path | Content |
|---|---|
| `queue_state_g1.json`, `queue_state_g2.json`, `queue_state_forecast.json` | Per-job status and GPU hours of the G1, G2 and forecast-input queues |
| `forecast/` | Forecast units and the forecast-input queue with its probes |
| `dose/` | Dose-response draws |
| `c1de/` | Draws of the exploratory German variant of C1 |
| `nestor_books_draws.json`, `stage_c_preview/` | NESTOR-books draws |
| `gap_trajectory/` | NESTOR-MoE gap trajectory: inputs, series, analysis script and figure |
| `continual.jsonl`, `stageB_summary.json`, `stageB_window_ci.json` | NESTOR reproduction, stage B |

## Core confirmatory analysis

| Path | Content |
|---|---|
| `analysis_core/analysis_confirm.json` | Output of the core confirmatory analysis (`analyze_confirm.py run --mode core`): C1 and C2 with the Holm family, exclusions (E1), descriptive blocks, the parameters and inputs used, and the log of the freeze check |
| `analysis_core/analysis_confirm.md` | Summary of the same analysis |
| `confirm/CORE_DONE.json` | Marker written when every core job of the confirmatory queue had run (74 of 74 jobs) |

The core analysis covers C1 on both MoE bases and C2. The other blocks are analysed by the full analysis
(`--mode full`, below).

## Full confirmatory analysis and checks after the main queue

| Path | Content |
|---|---|
| `analysis/analysis_confirm.json` | Output of the full confirmatory analysis (`analyze_confirm.py run --mode full`, 2026-11-20): C1 and C2 with the Holm family, the secondary and descriptive blocks, exclusions (E1), the parameters and inputs used, and the log of the freeze check |
| `analysis/analysis_confirm.md` | Summary of the same analysis |
| `queue_state_confirm.json` | Per-job status and GPU hours of the confirmatory queue (final state) |
| `a1_wd_only_checks.json` | Check of the weight-decay-only (`[wd_only]`) lines of the C3 FULL jobs at queue positions 131, 133 and 140 (`confirm/post_queue/a1_wd_only_checks.py`) |
| `a2_acceptance.json` | Acceptance of the three weight-decay-only values of the jobs at queue positions 137, 142 and 147 under amendment A-2 (`confirm/post_queue/a2_acceptance.py`) |
| `a4_acceptance.json` | Acceptance of the re-runs of queue positions 154 and 155 under amendment A-4 (`confirm/rerun_a4/a4_acceptance.py`) |

Script paths and `confirm_windows/` in this section are relative to the repository root.

All three weight-decay-only values of A-2 were accepted (`c3_revert_g1_nestor_full_german_s101`,
`c3_revert_g1_moe_b1_full_math_s101`, `c3_revert_g1_moe_b1_full_german_s101`); all three C3 FULL jobs passed the check
of their `[wd_only]` lines; both A-4 re-runs were accepted (`c3_freeze_moe_b1_german_s103`,
`c3_freeze_nestor_german_s103`). The full analysis was started with `confirm/post_queue/run_full_analysis.ps1` after
these checks.

Both `analysis_confirm.json` files are cleaned copies: local paths, approval notes, role labels and references to
internal planning documents in text fields were edited; numbers and structure are unchanged. Their field `sha256`
refers to the original file; `VERIFY.md` lists the SHA-256 of the original and of the published file.

To reproduce: `python analyze_confirm.py run --root . --freeze FREEZE.json --mode core` (or `--mode full`). The
analysis checks the freeze record first and reads the study records and the evaluation outputs of the confirmatory
runs under `outputs/confirm/`; their per-window results, run summaries and displacement series are published in
`confirm_windows/` (same layout), and the records are available from the author on reasonable request (see `VERIFY.md`).
