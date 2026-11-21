"""Tokenize a raw text corpus (parquet / jsonl.zst) into one eot-separated uint32 .bin.

Same reader and encoder as the repo's scripts/preprocess_text_bin.py
(nestor.data.token_documents.iter_document_texts + encode_documents), single process
(the sandbox forbids multiprocessing pipes); tiktoken's batch encoder still uses threads.
  python tok_domain.py --out OUT.bin [--max-tokens N] file1 [file2 ...]
"""
import argparse, glob, os, time
import numpy as np
from pathlib import Path
from nestor.data.token_documents import encode_documents, init_tokenizer_worker, iter_document_texts, write_bin_meta

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=int, default=None)
ap.add_argument("inputs", nargs="+")
a = ap.parse_args()
files = sorted(f for pat in a.inputs for f in glob.glob(pat))
init_tokenizer_worker("cl100k_base", 1, None)
t0, total, docs = time.time(), 0, 0
with open(a.out + ".tmp", "wb") as out:
    for f in files:
        batch = []
        for text in iter_document_texts(Path(f)):
            batch.append(text)
            if len(batch) >= 512:
                arr, kept = encode_documents(batch); arr.tofile(out); total += arr.size; docs += kept; batch = []
                if a.max_tokens and total >= a.max_tokens:
                    break
        if batch and not (a.max_tokens and total >= a.max_tokens):
            arr, kept = encode_documents(batch); arr.tofile(out); total += arr.size; docs += kept
        if a.max_tokens and total >= a.max_tokens:
            break
os.replace(a.out + ".tmp", a.out)
write_bin_meta(a.out, num_tokens=total, num_docs=docs, tokenizer_name="cl100k_base",
               extra={"inputs": [os.path.basename(f) for f in files], "max_tokens": a.max_tokens,
                      "elapsed_sec": round(time.time() - t0, 1)})
print(f"[tok_domain] {a.out}: {total:,} tokens, {docs:,} docs, {time.time()-t0:.0f}s", flush=True)
