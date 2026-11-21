# Confirmatory analysis (analysis_confirm/1)

Prereg: preregistration v1.3 version 8 (v1.2 secs. 5, 8, 9, 13 where kept)  
FREEZE.json verification: ok  
Created (UTC): 2026-11-17T13:32:42Z  

## C1 (books; confirmatory)

**Decision: confirmed**

D = 20 draws, statistic log; mean J = 1.429; exact one-sided sign-flip p = 9.537e-07 (1 of 1048576); Holm-adjusted p = 1.907e-06

| base | mean J | n | margin-limited |
|---|---|---|---|
| moe_b1 | 1.356 | 10 | False |
| moe_b2 | 1.503 | 10 | False |

| estimator | overall | moe_b1 | moe_b2 |
|---|---|---|---|
| beta-hat_raw | 1.654 [1.585, 1.726] | 1.604 [1.515, 1.69] | 1.7 [1.589, 1.832] |
| beta-hat_pin | 1.776 [1.697, 1.859] | 1.749 [1.645, 1.826] | 1.801 [1.652, 1.966] |
| beta-hat_sym | 1.337 [1.311, 1.365] | 1.337 [1.286, 1.383] | 1.338 [1.305, 1.378] |
| beta-hat_sym_pin | 1.398 [1.37, 1.428] | 1.417 [1.375, 1.46] | 1.38 [1.334, 1.428] |
| beta-hat_rand | 1.083 [0.9397, 1.211] | 1.083 [0.9397, 1.211] | NA |

Out-of-domain cells: moe_b1, moe_b2; interpretation: substantially decay-driven (mean I_WD >= 0.5 x mean I)

## C2

**confirmed**

s* = 8, D = 5, mean J2 = 4.286, p = 0.03125; Delta-e = 2.071 [-inf, inf]; manipulation check mean D_SGD/Delta_Q(mean) = 1.118


## EDGE (4.11)

Covered runs: 72; excluded: 0

- MoE|C2 Muon: n = 16; energy Wh {'median': 11.666668949924276, 'min': 11.458243653507257, 'max': 11.922839377251153}; wall s {'median': 246.5, 'min': 239.0, 'max': 266.0}
- MoE|C2 SGD: n = 16; energy Wh {'median': 20.260104287873297, 'min': 19.93445468397671, 'max': 21.376663969602177}; wall s {'median': 389.0, 'min': 374.0, 'max': 433.0}
- MoE|subset: n = 40; energy Wh {'median': 54.490497236585234, 'min': 53.139119299449604, 'max': 92.30724542475551}; wall s {'median': 924.0, 'min': 885.0, 'max': 2106.0}

Limits: Hardware: a desktop consumer GPU (RTX 4080 SUPER, 16 GB) stands in for edge hardware; absolute numbers do not transfer to edge devices; only relative costs within this study are interpretable. Board power only: CPU, DRAM, storage and power-supply losses are not measured; the board draw includes idle power and any other process on the GPU. Scheduling: the pipeline runs at BelowNormal CPU priority, which can lengthen wall time; no power or clock limit is set. Sampling: 1 Hz; the driver's power reading is not calibrated against an external meter.


## C4 per-window difference-in-differences (v1.2 sec. 10.2)

Overall: n = 20, mean beta_C4 = 0.9922, t95 0.9922 [0.9575, 1.027], two-way bootstrap [0.9157, 1.073]; mean within-draw corr 0.7469
- moe_b1: n = 10, mean 1.048, t95 1.048 [1.023, 1.073], two-way bootstrap [0.9677, 1.136]
- moe_b2: n = 10, mean 0.9365, t95 0.9365 [0.8942, 0.9788], two-way bootstrap [0.8575, 1.018]

## C2-eq alignment component and per-expert SNR (v1.2 sec. 8.2 item 7)

b_Muon - b_Muon^rand: 0.2361 [-0.1951, 0.6673]
Per-expert gradient SNR (G2-5): {"moe": {"median_snr_train": 2.798075489578145, "q25": 1.7919369104514111, "q75": 5.741643474761577, "share_snr_train_below_1": 0.1597222222222222, "units": 288}}

## Matched gain (v1.2 sec. 11)

- C1 mean I_G: 1.937 [1.794, 2.08]
- C2 mean I2_G: -
- FULL across architectures: 0 pairs (descriptive)
- Pareto points: 160

## Bits per byte (v1.2 sec. 9 item 13)

With F_bpb(c) = sum_w (CE_sum_run - CE_sum_0) / (ln 2 * sum_w n_bytes) (eval_windows diff_bits_per_byte) and equal-token windows, F_bpb(c) = kappa_c * F(c) with one factor per corpus and window set; every C1 / C2 statistic compares one corpus on identical windows, so its log and normalized-difference forms are unchanged in bits per byte. Only F levels change.

Status: computed; C1 check max |I_bpb - I| = 8.882e-16

## Analysed elsewhere

- routing_entanglement (v1.3 sec. 4.9): phase6/run_entanglement.py (stored version c6e2e322-d995-4f6b-9a1e-8e290dcc2842); predictions E1-E7 hashed before the run: results/entanglement/predictions_E1_E7.md, logs/entanglement_predictions.log; results entanglement_results.json (stored version 42dbd3a7-61c7-4adb-9929-7579d86673e9); note entanglement_note.md (version 15b283b4-11af-47ea-9b6b-901abe65c97d)
- gap_trajectory (v1.3 sec. 4.10): results/gap_trajectory/gap_analysis.py; results results/gap_trajectory/gap_trajectory.json (version 4e616dad-663c-4d1c-b058-acd2557b85d3); note results/gap_trajectory/gap_note.md (version 7f4a9295-b8d8-4f85-8229-7322803b8ddc); figure results/gap_trajectory/gap_trajectory.png (version 2a325ea8-ffe6-42a1-b147-778eb0933b06)

## Not computable from registered files

- C2 decay of the optimizer contrast at step 24: not computable (C2 runs stopped after step 8, P-16)
- C2 decay of the optimizer contrast at step 48: not computable (C2 runs stopped after step 8, P-16)
- C2 decay of the optimizer contrast at step 96: not computable (C2 runs stopped after step 8, P-16)

## Core analysis (P-16; prereg v1.3 version 8 sec. 3.5)

Core marker results/confirm/CORE_DONE.json (sha256 0893ab74346ed309138673c2c59972ddd0a0c092e852baf6b057d6cc5c8b7b04); core blocks C1:moe_b1, C1:moe_b2, C2; 72 core runs. The confirmatory decisions are final; C4, C1 matched gain and bits per byte are secondary and descriptive.

Deferred to the full analysis after the queue: B3, audit, selpin, C3, C3_per_layer, C5_french, DCLM-CONT, SEC, dose, M12b, C1_DE, NESTOR_books, forecasts, F_rel

## Warnings

- none
