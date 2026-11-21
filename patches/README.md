# Modifications of the upstream code

None. The study uses the upstream NESTOR repository (https://github.com/Ma-Lab-Berkeley/Nestor) at commit
`c797821bb2b933523d863b8cbb019f3473197ac5` without changes. A comparison of the working copy used for all runs
with that commit found no content differences (line endings only), so this folder contains no patch files.

Everything the study adds runs outside the upstream tree:

- The scripts at the repository root (for example `run_exp.py`, `run_continual.py`, `subset_adapt.py`) import
  the upstream package from `repo/`.
- `compat.py` (identical copy in `modal/`) installs two run-time shims for torch 2.14.1 on Windows:
  `varlen_attn(..., is_causal=True)` is translated to `window_size=(-1, 0)`, and the head dimension of
  scaled-dot-product attention is zero-padded to a multiple of 8 so that the fused kernels are used (the scores
  are unchanged, the original scale is passed explicitly and the padded output columns are removed).
- `conftest.py` (identical copy in `modal/`) applies the first shim when the upstream tests run.
- `shim/nestor/data/token_documents.py` and `modal/token_documents.py` are verbatim copies of the upstream file
  `nestor/data/token_documents.py`; they are not included here and are created by `tools/setup_upstream.py`.

Setup: `python tools/setup_upstream.py`.
