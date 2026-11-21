"""Build every G0/C1/C2/C5 data stream (CPU only), sequentially, one log per step (no pipes).

Steps follow protocol_engineering_v2 section 10 plus the design (general Python stream).
Already-finished outputs are skipped, so the driver is resumable.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv", "nestor", "Scripts", "python.exe")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUTF8": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
       "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_XET": "1", "TIKTOKEN_CACHE_DIR": os.environ.get("TIKTOKEN_CACHE_DIR", "")}
if not ENV["TIKTOKEN_CACHE_DIR"]:
    ENV.pop("TIKTOKEN_CACHE_DIR")
LOGDIR = os.path.join(HERE, "logs", "streams"); os.makedirs(LOGDIR, exist_ok=True)


def step(name, args, done_marker):
    if done_marker and os.path.exists(os.path.join(HERE, done_marker)):
        print(f"[prep] skip {name} (exists: {done_marker})", flush=True); return
    log = os.path.join(LOGDIR, f"{name}.log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call([PY, *args], cwd=HERE, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[prep] {name}: exit {rc} in {time.time()-t0:.0f}s (log {log})", flush=True)
    if rc != 0:
        raise SystemExit(f"step {name} failed")


S = "prep_streams.py"
step("code", [S, "sample", "--in", "data/domain/algstack_test.bin", "--out-dir", "data/streams",
              "--part", "code_eval=2200000", "--part", "code_route=4400000", "--seed", "0"], "data/streams/code_route.bin")
step("math", [S, "sample", "--in", "data/domain/owm_test.bin", "--out-dir", "data/streams",
              "--part", "math_eval=2200000", "--part", "math_route=4400000", "--seed", "0"], "data/streams/math_route.bin")
step("german_eval", [S, "sample", "--in", "data/domain/fw2de_test.bin", "--out-dir", "data/streams",
                     "--part", "german_eval=2200000", "--seed", "0"], "data/streams/german_eval.bin")
step("german_route", [S, "sample", "--in", "data/domain/fw2de_train.bin", "--out-dir", "data/streams",
                      "--part", "german_route=2200000", "--seed", "1"], "data/streams/german_route.bin")
step("books_route", [S, "sample", "--in", "data/domain/pg19_train.bin", "--out-dir", "data/streams",
                     "--part", "books_route=2200000", "--seed", "1"], "data/streams/books_route.bin")
# general Python stream
if not os.path.exists(os.path.join(HERE, "data/raw/codeparrot/file-000000000054.json.gz")):
    step("pycode_dl", ["-c", "from huggingface_hub import hf_hub_download as d; "
                       "print(d('codeparrot/codeparrot-clean-valid', 'file-000000000054.json.gz', repo_type='dataset', local_dir='data/raw/codeparrot'))"], None)
step("pycode_tok", ["pycode_tok.py", "--src", "data/raw/codeparrot/file-000000000054.json.gz", "--out", "data/domain/pycode_test.bin",
                    "--max-tokens", "12000000"], "data/domain/pycode_test.bin")
step("pycode", [S, "sample", "--in", "data/domain/pycode_test.bin", "--out-dir", "data/streams",
                "--part", "pycode_eval=2200000", "--part", "pycode_route=4400000", "--seed", "0"], "data/streams/pycode_route.bin")
# FineWeb-2 French (C5 prospective domain): train shard 0 capped at 150M tokens + full test file
step("fr", [S, "fineweb2", "--lang", "fra_Latn", "--name", "fr", "--raw-dir", "data/raw/fineweb-2", "--out-dir", "data/domain",
            "--train-shards", "1", "--train-max-tokens", "150000000"], "data/domain/fr_train.bin")
step("fr_eval", [S, "sample", "--in", "data/domain/fr_test.bin", "--out-dir", "data/streams",
                 "--part", "fr_eval=2200000", "--seed", "0"], "data/streams/fr_eval.bin")
step("fr_route", [S, "sample", "--in", "data/domain/fr_train.bin", "--out-dir", "data/streams",
                  "--part", "fr_route=2200000", "--seed", "1"], "data/streams/fr_route.bin")
print("[prep] ALL DONE", flush=True)
