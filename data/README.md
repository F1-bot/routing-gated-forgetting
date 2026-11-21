# Data

The study uses 13 tokenized data files. They are not distributed: together they take 9.4 GiB, and the licences of the source corpora apply (see the README). Each file is a
flat array of uint32 token ids of the cl100k_base tokenizer (tiktoken); documents are separated by the
end-of-text token 100257, which also ends each file.

| File | Source | Tokens | Documents | MiB | sha256 | Built by |
|---|---|---:|---:|---:|---|---|
| `data/dclm/dclm_cont_heldout.bin` | DCLM-baseline 1.0, global shard 3 | 29,998,792 | 24,528 | 114.4 | `7000c98f95bba8b4…` | `tok_files.py` |
| `data/dclm/dclm_val.bin` | DCLM-baseline 1.0, global shard 9 | 19,999,909 | 17,005 | 76.3 | `0644eac43505ad8f…` | `tok_files.py` |
| `data/domain/algstack_test.bin` | AlgebraicStack, test | 42,160,647 | 17,997 | 160.8 | `952a5831acf24269…` | `tok_domain.py` |
| `data/domain/algstack_train.bin` | AlgebraicStack, train (16 shards) | 1,676,450,112 | 458,783 | 6,395.1 | `24a089e7b56a7571…` | `tok_domain.py` |
| `data/domain/fr_test.bin` | FineWeb-2 fra_Latn, test | 39,038,319 | 44,245 | 148.9 | `b989c7ed77d53eb6…` | `prep_all_streams.py` |
| `data/domain/fr_train.bin` | FineWeb-2 fra_Latn, train | 150,285,222 | 147,456 | 573.3 | `0a84d8b65c17372d…` | `prep_all_streams.py` |
| `data/domain/fw2de_test.bin` | FineWeb-2 deu_Latn, test | 60,571,959 | 65,176 | 231.1 | `27e080a102cd6e8a…` | `tok_domain.py` |
| `data/domain/fw2de_train.bin` | FineWeb-2 deu_Latn, train | 150,124,418 | 157,184 | 572.7 | `c0ce16fba68ae066…` | `tok_domain.py` |
| `data/domain/owm_test.bin` | OpenWebMath, test | 63,933,458 | 31,577 | 243.9 | `77cc6d9fd78c434b…` | `tok_domain.py` |
| `data/domain/owm_train.bin` | OpenWebMath, train | 150,430,566 | 74,752 | 573.8 | `3a6bb73e1a3d43f9…` | `tok_domain.py` |
| `data/domain/pg19_test.bin` | PG-19, test | 9,872,926 | 100 | 37.7 | `3ad95cf1304d1a24…` | `tok_domain.py` |
| `data/domain/pg19_train.bin` | PG-19, train | 118,347,447 | 1,244 | 451.5 | `24ec92cb6dea867c…` | `tok_domain.py` |
| `data/streams/pycode_eval.bin` | codeparrot-clean-valid (file 54), Python | 2,204,757 | 985 | 8.4 | `08926d4af5b2d387…` | `prep_all_streams.py` |

The full sha256 values are in `VERIFY.md` and in `data_manifest.json`. The manifest also records every raw
input file (75 files, 18.3 GB: repository, path, size, sha256), the tokenization commands and the checks of the
rebuild of 2026-11-13, in which the rebuilt files reproduced the original token and document counts.

## Rebuilding

Run from the repository root, in the environment of `requirements.txt` (the scripts expect the interpreter at
`.venv/nestor/Scripts/python.exe`):

```
python rebuild/dl_raw.py --plan rebuild/raw_plan.json
python rebuild/rebuild_all.py --no-env-gate
python rebuild/build_manifest.py --out data/data_manifest.rebuilt.json
```

1. `dl_raw.py` downloads the raw inputs listed in `rebuild/raw_plan.json` from the Hugging Face Hub and checks
   their sha256. The Python code file of the code stream is downloaded by `prep_all_streams.py`.
2. `rebuild_all.py` re-runs the recorded tokenization commands. It builds every file of the manifest, including
   files that the freeze does not use; `--only NAME[,NAME]` restricts it to named jobs (job names as in the
   `commands` section of `data_manifest.json`).
3. `build_manifest.py` re-scans the rebuilt files. Write its output to a new file: the default output path is
   `data/data_manifest.json`, the manifest of this repository.
4. `rebuild/verify_bin.py BIN --tokens N --docs M --sha256 HEX` checks a single file against the counts above
   and its SHA-256 value in `VERIFY.md`.
