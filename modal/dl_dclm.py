"""Download the exact DCLM-baseline shards used by the local runs (public HF dataset, no token).

train: global-shard_01_of_10/local-shard_0_of_10/shard_000000{00..15}_processed.jsonl.zst
val:   global-shard_09_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/data/raw"
REPO = "mlfoundations/dclm-baseline-1.0"
files = [f"global-shard_01_of_10/local-shard_0_of_10/shard_{i:08d}_processed.jsonl.zst" for i in range(16)]
files.append("global-shard_09_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst")


def get(rel):
    t0 = time.time()
    err = None
    for attempt in range(4):
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=ROOT)
            return rel, os.path.getsize(p), round(time.time() - t0, 1)
        except Exception as e:  # retry transient errors
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{rel}: {err}")


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for rel, size, dt in ex.map(get, files):
        print(f"[dl] {rel} {size} bytes {dt}s", flush=True)
print(f"[dl] done {len(files)} files in {time.time() - t0:.0f}s", flush=True)
