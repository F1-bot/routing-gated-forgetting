# Does Routing Gate Forgetting? Pre-registered Tests for Resource-Constrained Mixture-of-Experts Adaptation

Code and results for the paper of the same title.

Serhii Dolhopolov ([ORCID 0000-0001-9418-0943](https://orcid.org/0000-0001-9418-0943)),
Kyiv National University of Construction and Architecture.

## Overview

The study builds on NESTOR (Chu, Pai and Ma, 2026, *Nested Sparsity for Scalable Pretraining and Continual
Learning*, arXiv:2610.31818; code: https://github.com/Ma-Lab-Berkeley/Nestor). It tests whether forgetting
during continued pretraining of routed transformers (mixture-of-experts and NESTOR models) is governed by the
overlap between the experts that the adaptation data use and the experts that the earlier data use.

The design was pre-registered and frozen on 2026-11-15 at 07:38:55 UTC, before any confirmatory run. The
confirmatory family (Holm, m = 2) has two tests:

- **C1, cross-corpus test on books adaptation.** For each of two MoE bases, pairs of books subsets are matched
  on the routing usage of the target and of DCLM but differ in their usage of the code and math experts. The
  registered contrast compares the forgetting of code and math after adaptation; it is tested with a one-sided
  exact sign-flip test over 20 draws.
- **C2, optimizer crossover test.** High- and low-usage subsets, matched on DCLM routing usage, are adapted
  with SGD and with Muon (8 draws).

Secondary and descriptive blocks cover a third MoE base, a dose-response block, an assumption audit at the base
checkpoint, per-layer reverts (C3), routing entanglement, the NESTOR-MoE gap trajectory, the cost of on-device
adaptation (EDGE) and an exploratory German variant of C1 (C1-DE). Hypotheses, statistics, decision rules and
exclusions are defined in the pre-registration (version 1.3). The pre-registration and the other study records
are not part of this repository; they are available from the author on reasonable request, and their SHA-256
values are listed in [VERIFY.md](VERIFY.md).

## Status of this release

This repository contains the pipeline code, the configurations and run manifests, the routing statistics and
evaluation data used by the gates, the results of the NESTOR reproduction (stages A to C), the GPU power logs, and
the figure and table scripts of the paper with their inputs. Every job of the confirmatory queue has run (final states
in `results/queue_state_confirm.json`). The output of the core confirmatory analysis (2026-11-17) is in
[results/analysis_core/](results/README.md) and that of the full analysis (2026-11-20) in
[results/analysis/](results/README.md); the per-window evaluation results of the confirmatory runs are in
`confirm_windows/`. The data, figures and tables of the paper are in `paper/v1/`.

After the main queue, two re-runs were made under recorded amendments after the freeze: the two assumption-audit jobs
(amendment A-3; `confirm/rerun/`, outputs in `audit/`) and the C3 freeze-mode jobs at queue positions 154 and 155, whose
first attempts failed the frozen-parameter audit (amendment A-4; `confirm/rerun_a4/`). Both A-4 re-runs passed their
acceptance check (`results/a4_acceptance.json`). The queue was resumed after pauses and interruptions; the resume logs
are in `logs/`. The amendments and the deviation record are study records (see [VERIFY.md](VERIFY.md)).

## Repository map

| Path | Content |
|---|---|
| root `*.py`, `*.ps1` | Pipeline code: pretraining and adaptation (`run_exp.py`, `run_continual.py`, `subset_adapt.py`), evaluation (`eval_windows.py`), routing statistics (`routing_stats.py`), gates (`g0_gate.py`, `g1_gate.py`, `g2_gate.py`), planning (`confirm_plan.py`), analysis (`analyze_confirm.py`), queue runner (`run_queue.py`), orchestration (`chain6b.ps1`), tests (`test_*.py`) |
| `phase5/`, `phase6/` | Code bundles installed at the root; `phase6/chain6b_util.py` contains the freeze check |
| `confirm/` | Confirmatory plan, queue, run specifications and evaluations of the base checkpoints; `confirm/rerun/`: re-run of the two assumption-audit jobs after the main queue (amendment A-3) and its result record (`confirm/rerun/rerun_result.json`); `confirm/rerun_a4/`: re-run of queue positions 154 and 155 (amendment A-4), its acceptance check and result record (`confirm/rerun_a4/rerun_a4_result.json`); `confirm/post_queue/`: acceptance checks after the main queue and the launcher of the full analysis (`run_full_analysis.ps1`, see `confirm/post_queue/README.md`) |
| `audit/` | Outputs of the assumption audit at the base checkpoints of `moe_b1` and `moe_b2`, written by the re-run of amendment A-3 |
| `confirm_windows/` | Per-window evaluation results (`.npz`), run summaries (`summary.json`) and displacement series (`interim/displacement.jsonl`) of the confirmatory runs, in the folder layout of `outputs/confirm/` that the analysis reads (final run folders only; C3 revert evaluations in `c3_reverts/` and `c3_layer_reverts/`) |
| `g0/`, `g1/`, `g2/` | Gate configurations, plans, queues and the routing statistics used by the gates |
| `frozen_g0/`, `inputs_v13/`, `theory/v4/`, `hooks/` | Code used by the G0 re-run, the routing-entanglement block and the pipeline hooks |
| `results/` | Queue states, forecast units and probes, dose-response, C1-DE and NESTOR-books draws, gap trajectory, stage-B summaries, the outputs of the core and the full confirmatory analysis (`results/analysis_core/`, `results/analysis/`) and the acceptance checks after the main queue (`results/a1_wd_only_checks.json`, `results/a2_acceptance.json`, `results/a4_acceptance.json`); `results/exploratory/`: exploratory runs after the freeze (no registered test, label or decision uses them) |
| `reproduction_stageA.json`, `stageB_windows/`, `stagec_reeval/` | NESTOR reproduction: stage A results, per-window evaluations of stage B, stage C re-evaluation |
| `data/` | Data manifest and data instructions; the tokenized data are not included |
| `rebuild/` | Download and rebuild scripts for the tokenized data |
| `modal/` | Scripts of the base runs on a cloud H100 GPU (Modal) |
| `logs/` | Environment snapshots (`logs/env/`), GPU power logs (`logs/power/`), the queue logs and job logs of the G1, G2 and forecast-input queues (`logs/queue_g1/`, `logs/queue_g2/`, `logs/forecast/`) and of the confirmatory queue (`logs/queue_confirm/`), logs of the cloud base run, resume logs after the pauses of 2026-11-18 and the interruptions of 2026-11-19 and 2026-11-20 (`logs/resume_*.log`, `logs/chain6b_resume_*_console.log`), the logs of the watchers of the two re-runs (`logs/audit_rerun.log`, `logs/a4_rerun.log`), console logs of the re-run watchers (`logs/watch_*_console.log`) and the log of the full analysis (`logs/analysis_full.log`) |
| `paper/v1/` | Figure and table scripts (`scripts/`), their inputs (`data/`), figures and tables, and the scripts that fill and check the TeX sources of the paper (the TeX sources and the PDF are not included) |
| `tools/` | Upstream setup and the freeze check with the records package |
| `patches/` | Notes on modifications of the upstream code (there are none) |

### Published copies of the code and the study records

The code files are copies of the files used in the study. Only comments, docstrings, message texts and default
paths were edited: absolute paths of the original machine were replaced by relative paths or by the placeholder
`<conda>`, and references to internal planning documents were removed. The program logic is unchanged; for every
Python file this was checked by comparing the syntax trees of the original and the published file (docstrings
excluded). Three test functions were renamed (test discovery is unchanged). Text that the code matches against the
frozen study records, and test fixtures that the tests compare, were left as they are.

The original files, the freeze record (`FREEZE.json`) and the other study records (pre-registration, gate,
decision and deviation records, code records) are not included. [VERIFY.md](VERIFY.md) lists their SHA-256 values;
copies are available from the author on reasonable request. Some steps read these records as inputs: the freeze
checks of the pipeline (`chain6b.ps1`, `chain6b_util.py`), the confirmatory analysis
(`analyze_confirm.py --freeze FREEZE.json`) and the scripts `fig_crossover.py` and `make_tables.py`; they need the
records package.

Some published scripts, logs and result files name files that are not in this repository: the study records above,
and notes, incident records, resume scripts and evidence folders written while the confirmatory queue ran. In
particular, `confirm/rerun/rerun_result.json` was written after an interruption on 2026-11-20 by a resume script that is
not published, with the code and the fields of `confirm/rerun/watch_and_rerun.ps1`; no later run overwrote it. These
files are available from the author on reasonable request.

Scripts with the `<conda>` placeholder need the path of the conda environment that provides the Python runtime
and its `Library\bin` folder: `chain6a.ps1`, `chain6b.ps1`, `hooks_after_B.ps1`, `hooks_after_G0.ps1`, `rebuild/rebuild_all.py`, `rebuild_env.ps1`, `results/gap_trajectory/gap_analysis.py` and their copies in
`phase6/` (`chain6b.ps1` also accepts the path as `-CondaLibBin`). Scripts whose default root was the original
project folder now default to the current directory or to the repository root; run them from the repository
root or pass `-Root` or `--root`.

## Environment

- Windows x64, Python 3.12.16. All runs used one environment; `requirements.txt` pins every package of it
  (exported from the environment; `logs/env/pip_freeze.txt` and `logs/env/conda_list_explicit.txt` are snapshots
  of it, with local paths replaced).
- PyTorch 2.14.1 with CUDA 13.0 (`--extra-index-url https://download.pytorch.org/whl/cu130`, included in
  `requirements.txt`), triton-windows 3.9.0.post30, tiktoken 0.14.0 (tokenizer cl100k_base).
- The code uses torch, triton, numpy, scipy, pandas, matplotlib, tiktoken, zstandard, pyarrow (data build),
  huggingface_hub, requests and pytest directly; the other entries of `requirements.txt` are their dependencies
  and tools of the original environment.
- Triton kernels and `torch.compile` on Windows need the Microsoft C++ build tools (Visual Studio 2022) and the
  Windows SDK; `msvc_env.ps1` sets up the compiler environment. On Linux, install `triton` instead of
  `triton-windows`.
- The upstream NESTOR code is not included. `python tools/setup_upstream.py` clones it at commit
  `c797821bb2b933523d863b8cbb019f3473197ac5` into `repo/` and creates the small `shim/` package that
  `run_exp.py` and `tok_files.py` import. No upstream file is modified (see [patches/README.md](patches/README.md)).

```
git clone https://github.com/F1-bot/routing-gated-forgetting.git
cd routing-gated-forgetting
python -m venv .venv\nestor
.venv\nestor\Scripts\activate     # Linux/macOS: python -m venv .venv/nestor && source .venv/nestor/bin/activate
pip install -r requirements.txt
python tools/setup_upstream.py
```

## Data sources and licences

The tokenized data files (about 9.5 GB) are not distributed. [data/README.md](data/README.md) lists every file
with its token and document counts and sha256, and explains how to rebuild them;
`data/data_manifest.json` records every raw input file (repository, path, size, sha256) and the tokenization
commands.

| Corpus | Source | Licence |
|---|---|---|
| DCLM-baseline 1.0 | `mlfoundations/dclm-baseline-1.0` | CC BY 4.0 |
| PG-19 (books) | `emozilla/pg19` (copy of `deepmind/pg19`) | Apache-2.0 |
| OpenWebMath | `EleutherAI/proof-pile-2`, `open-web-math` | ODC-By 1.0; Common Crawl terms of use |
| AlgebraicStack | `EleutherAI/proof-pile-2`, `algebraic-stack` | No single licence; the licences of the source repositories apply |
| FineWeb-2 (German `deu_Latn`, French `fra_Latn`) | `HuggingFaceFW/fineweb-2` | ODC-By 1.0; Common Crawl terms of use |
| Python code evaluation stream | `codeparrot/codeparrot-clean-valid` | The licences of the source repositories apply |

## Reproduction

1. Set up the environment and the upstream code as above.
2. Rebuild the data ([data/README.md](data/README.md)) and compare the files with the SHA-256 values in
   [VERIFY.md](VERIFY.md).
3. Base checkpoints. The seven base checkpoints (`dense_100m`, `moe_100m`, `moe_b1`, `moe_b2`, `moe_b3`,
   `nestor`, `nestor_100m`) are not distributed because of their size; `ckpt_pins.json` pins their sha256.
   They were trained with `run_exp.py`, locally and with the scripts in `modal/`.
4. Gates. The G1 and G2 runs are listed in `g1/queue_g1.json` and `g2/queue_g2.json`
   (`python run_queue.py <queue.json>`); `g0_gate.py`, `g1_gate.py` and `g2_gate.py` compute the gate decisions
   from the configurations in `g0/`, `g1/` and `g2/`.
5. Confirmatory runs: `python run_queue.py confirm/queue_confirm.json`. The orchestration script
   (`chain6b.ps1 -Root <path> -CondaLibBin <path>`) first checks the freeze record (records package) and then runs
   the same queue.
6. Analysis: `python analyze_confirm.py run --root . --freeze FREEZE.json --mode core` for the core analysis
   and `--mode full` for the complete analysis; the analysis checks the freeze record first and needs the records
   package. It reads the evaluation outputs of the confirmatory runs under `outputs/confirm/`; their per-window
   results, run summaries and displacement series are published in `confirm_windows/` in the same layout. The full analysis of this
   release was started with `confirm/post_queue/run_full_analysis.ps1` after the acceptance checks
   (`confirm/post_queue/a1_wd_only_checks.py`, `confirm/post_queue/a2_acceptance.py`,
   `confirm/rerun_a4/a4_acceptance.py`); `confirm/post_queue/README.md` gives the order, the preconditions and the
   command.
7. Figures and tables: the scripts in `paper/v1/scripts/` (`fig_*.py`, `make_tables.py`) read `paper/v1/data/`
   and `results/`; `fig_crossover.py` and `make_tables.py` also read study records. `fig_c3_single_layer.py` and
   `fig_c4_per_window.py` verify the original output of the full analysis from the records package (its embedded
   content hash and its `.sha256` file) and stop on the published cleaned copy; their published PDF and PNG files are
   byte-identical to the renders from the original output. [VERIFY.md](VERIFY.md) (check 3) lists what each script
   needs.

## Hardware

All gate, adaptation and evaluation runs used one NVIDIA GeForce RTX 4080 SUPER (16 GB) under Windows. Three
base pretraining runs used one NVIDIA H100 GPU in the cloud (Modal). GPU power logs and their description are
in `logs/power/` (`README_power_log.txt`).

## Verification

[VERIFY.md](VERIFY.md) describes the checks that this repository supports (code tests, data hashes, tables and
figures from `results/`), lists the SHA-256 values of the study records and of the original files, and explains
the freeze check with the records package (`python tools/verify_release.py`). Without the records package the script
stops with exit code 2, which means that nothing was checked; VERIFY.md lists what can be checked without it.

## Licence

- Code: MIT ([LICENSE](LICENSE)).
- Results, data files, figures and tables: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).
- The upstream NESTOR code is not part of this repository and keeps its own licence. The datasets keep their
  own licences (table above).

## Citation

Please cite the paper; [CITATION.cff](CITATION.cff) describes this repository.

```bibtex
@misc{dolhopolov2026routing,
  author       = {Dolhopolov, Serhii},
  title        = {Does Routing Gate Forgetting? Pre-registered Tests for Resource-Constrained Mixture-of-Experts Adaptation},
  year         = {2026},
  howpublished = {\url{https://github.com/F1-bot/routing-gated-forgetting}}
}
```
