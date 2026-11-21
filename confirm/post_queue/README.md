# confirm/post_queue: after the confirm queue (written 2026-11-18, while the queue runs)

Nothing here runs by itself and nothing here is listed in FREEZE.json or in the code set of `chain6b_util.code_set`
(checked: 170 files, none under `confirm/`). Every script is CPU only.

## Order

1. **After positions 137, 142 and 147 have ended** (any final state), the A-2 acceptance of the three WD-only values
   (rule: deviation record, entry A-2, Rules item 5; MF-A2-2; README_A2 step 9):

       & .venv\nestor\Scripts\python.exe confirm\post_queue\a2_acceptance.py

   writes `results\a2_acceptance.json` = {"accepted": {job: true|false}, "checks": {...}, "rule": "...", "utc": "..."}.
   Exit 0 written; 3 refused because a job has not ended (nothing written); 2 refused (inputs, the output exists, or
   the output path is listed in FREEZE.json). It reads only the job logs, the queue files, the queue states and the
   three G1 training logs, never an evaluation value. The registered input check of README_A2 step 9
   (`stage\a2\check_a2_inputs.ps1`) is separate and still to be run.
   **After positions 131, 133 and 140 have ended**, the check of D-1 addendum 1, item 6 (the [wd_only] lines of the three
   C3 FULL jobs; separate from the A-2 file, which the paper fill reads):

       & .venv\nestor\Scripts\python.exe confirm\post_queue\a1_wd_only_checks.py

   writes `results\a1_wd_only_checks.json` = {"pass": {job: true|false}, "checks": {...}, "rule": "...", "utc": "..."}:
   per job the line of `logs\queue_confirm\<job>_eval_wd.log`, max_steps / warm-up / executed against the frozen queue
   job (96/10/96), the sources (must include `frozen queue job (queue_confirm.json)`), "train-log LRs equal at all 96
   steps", the training log of the job root (steps 1..96 with learning rates), log path and sha256; fingerprint and
   device are recorded (a fingerprint other than "not checked" and the device cpu are notes). Same safety and exit codes
   as `a2_acceptance.py`; that file is unchanged (this script imports its helpers).
2. **The A-3 watcher** (`confirm\rerun\watch_and_rerun.ps1`, running) re-runs the two audit jobs after the main queue and
   writes `confirm\rerun\rerun_result.json`.
3. **The A-4 watcher** (`confirm\rerun_a4\watch_and_rerun_a4.ps1`; `confirm\rerun_a4\README.md`) waits for the A-3 watcher,
   re-runs positions 154 and 155 (C3 freeze-mode runs) and writes `confirm\rerun_a4\rerun_a4_result.json` and, through
   `confirm\rerun_a4\a4_acceptance.py`, `results\a4_acceptance.json`.
4. **The full analysis**, when 1, 2 and 3 are done and nothing runs on the GPU:

       powershell -NoProfile -ExecutionPolicy Bypass -File confirm\post_queue\run_full_analysis.ps1 -CheckOnly
       powershell -NoProfile -ExecutionPolicy Bypass -File confirm\post_queue\run_full_analysis.ps1

   Command: `<Root>\.venv\nestor\Scripts\python.exe confirm\post_queue\analysis_notorch.py --root <Root> --log
   <Root>\logs\analysis_full.log -- run --mode full --root <Root>`, i.e. the frozen `analyze_confirm.py run --mode full`
   with torch imports blocked, Idle priority, OMP/MKL/OPENBLAS/NUMEXPR/RAYON threads 1, PATH prefixed by
   `<conda>\envs\nestor\Library\bin`, `TIKTOKEN_CACHE_DIR=<Root>\data\tiktoken`,
   PYTHONPATH, PYTHONUTF8 and TEMP as in `chain6b.ps1` lines 58-74. Output (analyze_confirm.py line 2500):
   `results\analysis\analysis_confirm.json`, `.json.sha256`, `analysis_confirm.md`; log `logs\analysis_full.log`
   (UTF-8, appended). analyze_confirm.py verifies FREEZE.json itself (exit 23 if that fails).

   Preconditions (every failure is listed; exit = the smallest code; nothing is run):
   41 `results\chain6b\confirm.done` present and `results\chain6b\LOCK` reads `released ... (exit 0)`;
   42 no `results\STOP` or `PAUSE`, no open job in `logs\queue_confirm\queue.log`, no queue/audit/analysis python
   (when the process list is readable), GPU memory <= `-GpuIdleMiB` (default 4096 MiB);
   43 `confirm\rerun\rerun_result.json` present and not from a dry run (both audit states are reported, from it and
   from `results\queue_state_confirm.json`; an audit that is not 'done' is a warning);
   44 `results\a2_acceptance.json` present with exactly the keys accepted, checks, rule, utc;
   45 `results\analysis\analysis_confirm.json` absent (never overwritten);
   46 root, python, launcher or analyze_confirm.py missing;
   47 `confirm\rerun_a4\rerun_a4_result.json` present and not from a dry run, and `results\a4_acceptance.json` present
   with exactly the keys accepted, checks, rule, utc.
   Warnings only, never a refusal: `results\a1_wd_only_checks.json` missing, unreadable or not passed; a position 154 or
   155 re-run that `results\a4_acceptance.json` does not accept ("WARNING: <job> is not accepted under A-4: its values are
   withheld (not reported; D-3)", with the reasons).

## The learning-rate difference of the A-2 check

`weight_controls.wd_only_state` (lines 512-521) compares the decay-only learning rates with the training log in
memory and raises unless |a - b| <= 1e-9 max(|a|, |b|) at every step; the `[wd_only]` line records only the outcome.
`checks.<job>.lr_max_abs_diff` is therefore `{"value": null, "bound": 1e-9 max_lr / (1 - 1e-9)}` with max_lr from the
training log: 7.07e-12 for the NESTOR run, 2.89e-12 for the two MoE runs. (The A-2 GPU check of position 137 measured
0.0 itself: `checks\c3_wd_precheck\gpu_g1_137_...json`, `c_lrs.max_abs_diff`.) Because the message
"equal at all N steps" uses N = executed steps, the script also requires the training log to hold steps 1..96, each
with its learning rates.

## Files (sha256)

| File | sha256 |
|---|---|
| `a2_acceptance.py` | `dd429deb351abfcd202045b6580a9cfe8df47a50fc2af4c6a94626cd0fd37e39` |
| `analysis_notorch.py` | `91f852ec7cc4feaef898e6ec2a72fc88514b6e383bdc940c544bb21ae19bffe0` |
| `a1_wd_only_checks.py` | `39624d29f2df7c6515b0c0049b2f04ca47543e7c5006488968b267df8e051283` |
| `run_full_analysis.ps1` | `214f842631d340f1a8d4ac4474fcff2a90d60a091b3a68c37c0dc9ba1da0cac4` (v3, 2026-11-19: precondition 47 and the A-4 warning; v2 with the A-1 warning `36f20b2a…`; first version `a219597c…`) |
| `test/run_a2_tests.py` | `c63f5e5e8c0e015a5aca07f94ae9b94163dea501ec753b6c1255616569b3b960` |
| `test/a2_tests_report.json` | `1fc6654fdd22e0d09a4c4943f5bd913ed6fa4c13356cc4f605ffb781833f1009` |
| `test/run_a1_tests.py` | `b91521b21aa73a0ca1e8e2aa407324dc8aa8345b31846e212956baac89163663` |
| `test/a1_tests_report.json` | `35de64829cd563b635d33b3f31e9d6d9ed8566482fad75bbff7d93c04d5f1eb4` |
| `test/rehearsal_full_analysis/rehearsal_full_analysis.json` | `8d11ff67ab34fafdd11dfbb8501313f4805a7fd1b19292a1998d156576ec70f1` |
| `test/rehearsal_full_analysis/rehearsal_full_analysis_pass2.json` | `da4f28c0b347413ee3cda1ac323f860f071047f2ff50a2fd6c9168e3f6f877f3` |

## Tests and rehearsal (2026-11-18, Idle, one thread)

- `test/run_a2_tests.py`: 17/17 cases (`test/a2_tests_report.json`). The positive line is rebuilt from the A-2 GPU check
  of position 137 (its stored `staged.report`, with the f-string of weight_controls.py lines 523-525); variants,
  missing logs, a failed or pending job and the refusals of the default output are tested on copies. The real
  `results\a2_acceptance.json` was not written. `test/T*/` hold the test inputs and outputs; `test/T4_mockroot/` holds
  copies of the queue files, the two queue states and FREEZE.json (inert copies; nothing reads them).
- `test/rehearsal_full_analysis/`: `run_full_analysis.ps1` on a chain-built mock root (simulated outputs from
  mock_confirm.py; analysis modules byte-identical to the frozen ones; FREEZE.json of the mock verifies): exit 0,
  analysis 38.8 s, whole script 39.7 s, torch not imported; `mock_confirm.py simulate` 5.9 s. Negative cases 41, 42
  (two), 43, 44 (two), 45: 7/7. On the real root `-CheckOnly` refused with exit 41 at 15:12 (queue running) and wrote
  nothing.
- `test/run_a1_tests.py`: 18/18 cases (`test/a1_tests_report.json`). Positive lines in the format of the 15 real
  confirm lines (`*_eval_wd.log`: 96/10/96, frozen queue job, LRs equal at all 96 steps) with fingerprint "not checked";
  mock job roots hold a copy of a real 96-step training log (b3_c1_moe_b3_books_d1_A). Single faults (training-queue
  source, executed 95, warm-up 0, LRs not checked, max_steps 192, missing log, two lines, failed job, other experiment,
  95-step training log) fail; device cpu and fingerprint "reproduced" pass with a note; refusals 3, 2, 0, 2, 2 as
  `a2_acceptance.py`. The real `results\a1_wd_only_checks.json` was not written.
- Second pass after the A-1 block (`test/rehearsal_full_analysis/rehearsal_full_analysis_pass2.json`): `-CheckOnly`
  without the A-1 file exit 0 with the warning; full run exit 0 with the A-1 line and its warning, analysis 24.6 s,
  script 25.5 s; with an unreadable A-1 file only a warning (exit 45 there came from the existing output). The analysis
  output of the two passes is identical apart from `created_utc` (and the "Created (UTC)" line of the .md).
- Precondition 47 (2026-11-19, v3): `confirm\rerun_a4\test\run_rfa_checks.py` runs the real script with `-CheckOnly` on
  minimal mock roots that meet every other precondition: 7/7 (`confirm\rerun_a4\test\rfa_checks_report.json`): all met
  exit 0; A-4 result missing, from a dry run, acceptance missing or with other keys exit 47; a withheld Nestor re-run
  exit 0 with its WARNING line; A-3 and A-4 results missing exit 43 with both listed; nothing written in any case.
  `-CheckOnly` on the mock `fa\r3` refused with exit 45 (its earlier analysis output) and listed 47 twice; on the real
  root at 10:05 it refused with exit 41 (queue running) and listed 42, 43 and 47; `logs\analysis_full.log` unchanged.
