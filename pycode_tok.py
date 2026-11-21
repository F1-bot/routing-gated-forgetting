"""Tokenize the codeparrot-clean-valid Python file (json.gz lines with a 'content' field) with the authors'
cl100k encoder and eot convention (nestor.data.token_documents), capped at --max-tokens.

  python pycode_tok.py --src data/raw/codeparrot/file-000000000054.json.gz --out data/domain/pycode_test.bin --max-tokens 12000000
"""
import argparse, gzip, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))
import nestor.data.token_documents as TD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=12_000_000)
a = ap.parse_args()
TD.init_tokenizer_worker("cl100k_base", 1, None)
t0, total, docs, seen = time.time(), 0, 0, 0
with gzip.open(a.src, "rt", encoding="utf-8") as fin, open(a.out + ".tmp", "wb") as fout:
    batch = []
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("content")
        except json.JSONDecodeError:
            continue
        if not text:
            continue
        seen += 1
        batch.append(text)
        if len(batch) >= 256:
            arr, kept = TD.encode_documents(batch); arr.tofile(fout); total += int(arr.size); docs += kept; batch = []
            if total >= a.max_tokens:
                break
    if batch and total < a.max_tokens:
        arr, kept = TD.encode_documents(batch); arr.tofile(fout); total += int(arr.size); docs += kept
os.replace(a.out + ".tmp", a.out)
TD.write_bin_meta(a.out, num_tokens=total, num_docs=docs, tokenizer_name="cl100k_base",
                  extra={"source": "codeparrot/codeparrot-clean-valid " + os.path.basename(a.src), "field": "content",
                         "max_tokens": a.max_tokens, "docs_read": seen, "elapsed_sec": round(time.time() - t0, 1)})
print(f"[pycode_tok] {a.out}: {total:,} tokens, {docs:,} docs ({seen:,} read) in {time.time()-t0:.0f}s", flush=True)
