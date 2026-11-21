"""Scan a uint32 token .bin (end-of-text separated, cl100k): bytes, tokens, docs, sha256, sanity checks.

  python verify_bin.py BIN [--tokens N] [--docs M] [--sha256 HEX] [--out result.json]

Tokens = bytes/4. Docs = number of end-of-text ids (100257), counted independently of the .meta.json sidecar,
which is cross-checked when present. Additional sanity values: last token is the eot, empty documents (two eots in
a row), maximum token id (< 100277 for cl100k_base).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

EOT = 100257
NVOCAB = 100277


def scan(path, chunk_tokens=1 << 24):
    size = os.path.getsize(path)
    if size % 4:
        raise ValueError(f"{path}: size {size} is not a multiple of 4 bytes")
    h = hashlib.sha256()
    n_tok = n_eot = empty = 0
    max_id = 0
    prev_last_eot = None
    first = last = None
    t0 = time.time()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk_tokens * 4)
            if not b:
                break
            h.update(b)
            a = np.frombuffer(b, dtype=np.uint32)
            e = a == EOT
            n_eot += int(e.sum())
            empty += int((e[1:] & e[:-1]).sum())
            if prev_last_eot and e[0]:
                empty += 1
            prev_last_eot = bool(e[-1])
            max_id = max(max_id, int(a.max()))
            if first is None:
                first = int(a[0])
            last = int(a[-1])
            n_tok += int(a.size)
    return {"path": path, "bytes": size, "tokens": n_tok, "docs": n_eot, "sha256": h.hexdigest(),
            "ends_with_eot": last == EOT, "empty_docs": empty, "max_token_id": max_id,
            "first_token_is_eot": first == EOT, "scan_seconds": round(time.time() - t0, 1)}


def check(path, tokens=None, docs=None, sha256=None):
    r = scan(path)
    meta_path = path + ".meta.json"
    if os.path.isfile(meta_path):
        try:
            m = json.load(open(meta_path, encoding="utf-8"))
            r["meta_tokens"], r["meta_docs"] = m.get("num_tokens"), m.get("num_docs")
        except Exception as e:  # pragma: no cover
            r["meta_error"] = str(e)
    exp = {"tokens": tokens, "docs": docs, "sha256": sha256}
    r["expected"] = exp
    problems = []
    if tokens is not None and r["tokens"] != tokens:
        problems.append(f"tokens {r['tokens']:,} != expected {tokens:,}")
    if docs is not None and r["docs"] != docs:
        problems.append(f"docs {r['docs']:,} != expected {docs:,}")
    if sha256 is not None and r["sha256"] != sha256:
        problems.append(f"sha256 {r['sha256'][:16]}.. != expected {sha256[:16]}..")
    if r.get("meta_tokens") is not None and (r["meta_tokens"] != r["tokens"] or r["meta_docs"] != r["docs"]):
        problems.append(f"sidecar says {r['meta_tokens']}/{r['meta_docs']} but scan finds {r['tokens']}/{r['docs']}")
    if not r["ends_with_eot"]:
        problems.append("last token is not the eot")
    if r["max_token_id"] >= NVOCAB:
        problems.append(f"token id {r['max_token_id']} outside cl100k vocabulary")
    r["problems"] = problems
    r["match"] = not problems if (tokens is not None or docs is not None or sha256 is not None) else None
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--docs", type=int)
    ap.add_argument("--sha256")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = check(a.bin, a.tokens, a.docs, a.sha256)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: res[k] for k in ("path", "bytes", "tokens", "docs", "sha256", "match", "problems")}, indent=1))
    sys.exit(0 if res["match"] in (True, None) else 2)
