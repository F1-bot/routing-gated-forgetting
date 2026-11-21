"""De-duplication check of a query token bin against reference bins (condition 4: codeparrot vs
AlgebraicStack test and the DCLM evaluation shards). Exact-document content hashes and within-document n-gram overlap.

  python dedup_check.py --query data/domain/pycode_test.bin --ref algstack=data/domain/algstack_test.bin \
      --ref dclm09=data/dclm/dclm_val.bin --ref dclm10=data/dclm/dclm_val10.bin [--n 13] --out g0/dedup_pycode.json

Reported per reference:
  doc_exact_rate      share of query documents whose exact token content (blake2b of the token bytes) occurs in the reference
  ngram_token_rate    share of query n-gram positions (within documents) whose n-gram occurs in the reference
  doc_any_ngram_rate  share of query documents with at least one shared n-gram
  doc_half_ngram_rate share of query documents with >= 50 % of their n-grams shared (near-duplicates)
  doc_ids             query document indices with an exact match (first 200) -> filter list
n-grams are hashed with a 64-bit polynomial rolling hash (collision probability negligible at these sizes) and never
cross an end-of-text token. Token bins are read like nestor.data.token_documents.open_token_bin (sidecar dtype/eot).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

DEFAULT_EOT = 100257
_B = np.uint64(0x100000001B3)          # FNV-style odd multiplier (wrap-around arithmetic mod 2^64)


def open_bin(path: str) -> tuple[np.ndarray, int]:
    meta = {}
    for mp in (path + ".meta.json", os.path.splitext(path)[0] + ".meta.json"):
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding="utf-8-sig"))
            break
    dtype = np.uint16 if meta.get("token_dtype") == "uint16" else np.uint32
    return np.memmap(path, dtype=dtype, mode="r"), int(meta.get("eot_token", DEFAULT_EOT))


def doc_bounds(toks: np.ndarray, eot: int) -> np.ndarray:
    """[n_docs, 2] (start, end) with end exclusive of the eot token; a trailing document without eot is kept."""
    e = np.flatnonzero(np.asarray(toks) == eot)
    starts = np.concatenate([[0], e + 1])
    ends = np.concatenate([e, [len(toks)]])
    keep = ends > starts
    return np.stack([starts[keep], ends[keep]], 1)


def doc_hashes(toks: np.ndarray, bounds: np.ndarray) -> list[bytes]:
    a = np.asarray(toks)
    return [hashlib.blake2b(np.ascontiguousarray(a[s:e], dtype=np.uint32).tobytes(), digest_size=16).digest() for s, e in bounds]


def ngram_hashes(toks: np.ndarray, eot: int, n: int, chunk: int = 1 << 24) -> tuple[np.ndarray, np.ndarray]:
    """(hashes, start positions) of every n-gram that contains no eot token."""
    a = np.asarray(toks).astype(np.uint64)
    L = len(a) - n + 1
    if L <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    hs, ps = [], []
    is_eot = (np.asarray(toks) == eot).astype(np.int64)
    ce = np.concatenate([[0], np.cumsum(is_eot)])
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        h = np.zeros(e - s, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for j in range(n):
                h = h * _B + (a[s + j: e + j] + np.uint64(1))
        pos = np.arange(s, e)
        ok = (ce[pos + n] - ce[pos]) == 0
        hs.append(h[ok])
        ps.append(pos[ok])
    return np.concatenate(hs), np.concatenate(ps)


def compare(query: str, refs: dict[str, str], n: int = 13, log=print) -> dict:
    t0 = time.time()
    qt, qe = open_bin(query)
    qb = doc_bounds(qt, qe)
    qdh = doc_hashes(qt, qb)
    qh, qp = ngram_hashes(qt, qe, n)
    doc_of = np.searchsorted(qb[:, 0], qp, side="right") - 1
    n_per_doc = np.bincount(doc_of, minlength=len(qb))
    out = {"query": os.path.abspath(query), "query_tokens": int(len(qt)), "query_docs": int(len(qb)), "n": n,
           "query_ngrams": int(len(qh)), "refs": {}}
    for name, path in refs.items():
        rt, re_ = open_bin(path)
        rb = doc_bounds(rt, re_)
        rset = set(doc_hashes(rt, rb))
        exact = np.array([h in rset for h in qdh], dtype=bool)
        rh = np.unique(ngram_hashes(rt, re_, n)[0])
        hit = np.isin(qh, rh, assume_unique=False)
        hit_per_doc = np.bincount(doc_of[hit], minlength=len(qb))
        frac = np.divide(hit_per_doc, n_per_doc, out=np.zeros(len(qb)), where=n_per_doc > 0)
        out["refs"][name] = {"path": os.path.abspath(path), "ref_tokens": int(len(rt)), "ref_docs": int(len(rb)),
                             "doc_exact_rate": float(exact.mean()) if len(exact) else 0.0, "doc_exact_count": int(exact.sum()),
                             "ngram_token_rate": float(hit.mean()) if len(hit) else 0.0,
                             "doc_any_ngram_rate": float((hit_per_doc > 0).mean()) if len(qb) else 0.0,
                             "doc_half_ngram_rate": float((frac >= 0.5).mean()) if len(qb) else 0.0,
                             "doc_ids": np.flatnonzero(exact)[:200].tolist()}
        if log:
            r = out["refs"][name]
            log(f"[dedup] {name}: exact docs {r['doc_exact_count']} ({r['doc_exact_rate']:.4%}), {n}-gram tokens "
                f"{r['ngram_token_rate']:.4%}, docs >=50% {r['doc_half_ngram_rate']:.4%} ({time.time() - t0:.0f}s)")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exact-document and n-gram de-duplication rates")
    ap.add_argument("--query", required=True)
    ap.add_argument("--ref", action="append", required=True, help="name=path.bin (repeatable)")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = compare(a.query, dict(r.split("=", 1) for r in a.ref), a.n)
    txt = json.dumps(res, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(txt)
    print(json.dumps({k: {kk: v[kk] for kk in ("doc_exact_rate", "ngram_token_rate", "doc_half_ngram_rate")}
                      for k, v in res["refs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
