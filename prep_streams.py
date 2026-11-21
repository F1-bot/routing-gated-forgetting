"""Data preparation for the phase-3 evaluation streams (the operator runs it; single process, authors' encoder).

1. Disjoint document-level streams from an existing token bin (code = AlgebraicStack test, math =
   OpenWebMath test, and any other test bin). Documents (eot-terminated spans, the authors'
   document_spans) are shuffled with a seed and dealt in that order to the named parts until each part
   reaches its token budget; every part is written in the shuffled order, so the FIRST windows of a
   part are a random sample of the source (a test bin's file order can be grouped by source/language).
   A document is never split and never lands in two parts:
     python prep_streams.py sample --in data/domain/algstack_test.bin --out-dir data/streams \
            --part code_eval=2200000 --part code_route=4400000 --seed 0
     python prep_streams.py sample --in data/domain/owm_test.bin --out-dir data/streams \
            --part math_eval=2200000 --part math_route=4400000 --seed 0
   (eval = forgetting evaluation windows; route = routing statistics for subset selection, split into
   cross-fitting halves downstream -- selection and evaluation never share a document.)

2. The prospective domain, FineWeb-2 French (fra_Latn; same layout as the authors' deu_Latn recipe in
   scripts/slurm_prep_domain_data.sh): download the first N training parquets + the test parquet(s)
   from HuggingFaceFW/fineweb-2, tokenize with cl100k (token_documents.encode_documents), train capped
   like the other domains (150M tokens), test from the dataset's own test split (disjoint by release):
     python prep_streams.py fineweb2 --lang fra_Latn --name fr --raw-dir data/raw/fineweb-2 \
            --out-dir data/domain --train-shards 1 --train-max-tokens 150000000
   then e.g.  python prep_streams.py sample --in data/domain/fr_test.bin --out-dir data/streams \
            --part fr_eval=2200000 --part fr_route=2200000 --seed 0

3. Generic tokenization (tok_domain.py semantics):  python prep_streams.py tokenize --out X.bin file1 ...
Every output gets the authors' <bin>.meta.json sidecar; each `sample` call also writes
<out-dir>/<first part>.manifest.json (source fingerprint, seed, per-part sha256, tokens, documents).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "repo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_meta(path: str, *, num_tokens: int, num_docs: int, eot: int, extra: dict | None = None) -> None:
    """The authors' <bin>.meta.json format (token_documents.write_bin_meta) without needing tiktoken."""

    meta = {"token_dtype": "uint32", "num_tokens": int(num_tokens), "num_docs": int(num_docs),
            "tokenizer": "cl100k_base", "eot_token": int(eot)}
    meta.update(extra or {})
    Path(f"{path}.meta.json").write_text(json.dumps(meta, indent=2))


def sample_parts(src: str, parts: list[tuple[str, int]], out_dir: str, seed: int = 0,
                 min_doc_tokens: int = 1) -> dict:
    """Seeded document-level split of `src` into disjoint parts with token budgets."""

    from nestor.data.token_documents import document_spans, open_token_bin

    toks, eot = open_token_bin(Path(src))
    spans = document_spans(toks, eot)
    lens = spans[:, 1] - spans[:, 0]
    keep = np.flatnonzero(lens >= min_doc_tokens)
    order = keep[np.random.default_rng(seed).permutation(len(keep))]
    os.makedirs(out_dir, exist_ok=True)
    pos, report = 0, {}
    dtype = np.uint32                                  # outputs are always uint32 (the sidecar says so)
    used = np.zeros(len(spans), dtype=bool)
    for name, budget in parts:
        out = os.path.join(out_dir, f"{name}.bin")
        n_tok, n_doc = 0, 0
        with open(out + ".tmp", "wb") as fh:
            while n_tok < budget and pos < len(order):
                d = int(order[pos]); pos += 1
                s, e = int(spans[d, 0]), int(spans[d, 1])
                arr = np.empty(e - s + 1, dtype=dtype)
                arr[:-1] = toks[s:e]
                arr[-1] = eot
                arr.tofile(fh)
                n_tok += arr.size; n_doc += 1
                if used[d]:
                    raise AssertionError("document assigned twice")
                used[d] = True
        os.replace(out + ".tmp", out)
        write_meta(out, num_tokens=n_tok, num_docs=n_doc, eot=eot,
                       extra={"source": os.path.abspath(src), "seed": seed, "budget": budget,
                              "method": "seeded document shuffle, whole documents, disjoint parts"})
        report[name] = {"path": os.path.abspath(out), "tokens": n_tok, "docs": n_doc, "sha256": sha256_file(out),
                        "short_of_budget": bool(n_tok < budget)}
        print(f"[prep_streams] {name}: {n_tok:,} tokens, {n_doc:,} docs", flush=True)
    man = {"source": os.path.abspath(src), "source_tokens": int(len(toks)), "source_docs": int(len(spans)),
           "seed": seed, "min_doc_tokens": min_doc_tokens, "parts": report, "docs_unused": int(len(keep) - pos),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(out_dir, f"{parts[0][0]}.manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    return man


def tokenize(files: list[str], out: str, max_tokens: int | None = None, *, fake_encoder: bool = False) -> dict:
    """tok_domain.py: authors' reader + encoder, one eot-separated uint32 bin."""

    import nestor.data.token_documents as TD

    if fake_encoder:                       # tests only: a byte-level stand-in for cl100k
        class _Fake:
            eot_token = 100257

            def encode_ordinary_batch(self, texts):
                return [list(t.encode("utf-8")) for t in texts]
        TD._ENC, TD._MIN_DOC_TOKENS, TD._MAX_DOC_TOKENS = _Fake(), 0, None
    else:
        TD.init_tokenizer_worker("cl100k_base", 1, None)
    t0, total, docs = time.time(), 0, 0
    with open(out + ".tmp", "wb") as fh:
        for f in files:
            batch = []
            for text in TD.iter_document_texts(Path(f)):
                batch.append(text)
                if len(batch) >= 512:
                    arr, kept = TD.encode_documents(batch); arr.tofile(fh); total += arr.size; docs += kept; batch = []
                    if max_tokens and total >= max_tokens:
                        break
            if batch and not (max_tokens and total >= max_tokens):
                arr, kept = TD.encode_documents(batch); arr.tofile(fh); total += arr.size; docs += kept
            if max_tokens and total >= max_tokens:
                break
    os.replace(out + ".tmp", out)
    write_meta(out, num_tokens=total, num_docs=docs, eot=TD._ENC.eot_token,
                      extra={"inputs": [os.path.basename(f) for f in files], "max_tokens": max_tokens,
                             "elapsed_sec": round(time.time() - t0, 1), "encoder": "fake" if fake_encoder else "cl100k_base"})
    print(f"[prep_streams] {out}: {total:,} tokens, {docs:,} docs, {time.time() - t0:.0f}s", flush=True)
    return {"path": os.path.abspath(out), "tokens": total, "docs": docs}


def fineweb2(lang: str, name: str, raw_dir: str, out_dir: str, train_shards: int = 1,
             train_max_tokens: int | None = 150_000_000, test_max_tokens: int | None = None,
             download: bool = True) -> dict:
    """FineWeb-2 <lang>: download (huggingface_hub) + tokenize train/test like the authors' fw2de recipe."""

    repo = "HuggingFaceFW/fineweb-2"
    train_rel = [f"data/{lang}/train/000_{i:05d}.parquet" for i in range(train_shards)]
    if download:
        from huggingface_hub import HfApi, hf_hub_download

        test_rel = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(f"data/{lang}/test/")]
        if not test_rel:
            raise SystemExit(f"no test files under data/{lang}/test/ in {repo}")
        for rel in train_rel + test_rel:
            hf_hub_download(repo, rel, repo_type="dataset", local_dir=raw_dir)
    test_files = sorted(glob.glob(os.path.join(raw_dir, "data", lang, "test", "*.parquet")))
    train_files = [os.path.join(raw_dir, r) for r in train_rel]
    os.makedirs(out_dir, exist_ok=True)
    rep = {"lang": lang, "repo": repo,
           "test": tokenize(test_files, os.path.join(out_dir, f"{name}_test.bin"), test_max_tokens),
           "train": tokenize(train_files, os.path.join(out_dir, f"{name}_train.bin"), train_max_tokens)}
    json.dump(rep, open(os.path.join(out_dir, f"{name}_prep.json"), "w", encoding="utf-8"), indent=1)
    return rep


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="phase-3 stream preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--in", dest="src", required=True); s.add_argument("--out-dir", required=True)
    s.add_argument("--part", action="append", required=True, help="name=tokens"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-doc-tokens", type=int, default=1)
    t = sub.add_parser("tokenize"); t.add_argument("--out", required=True); t.add_argument("--max-tokens", type=int, default=None)
    t.add_argument("inputs", nargs="+"); t.add_argument("--fake-encoder", action="store_true", help=argparse.SUPPRESS)
    f = sub.add_parser("fineweb2"); f.add_argument("--lang", default="fra_Latn"); f.add_argument("--name", default="fr")
    f.add_argument("--raw-dir", default="data/raw/fineweb-2"); f.add_argument("--out-dir", default="data/domain")
    f.add_argument("--train-shards", type=int, default=1); f.add_argument("--train-max-tokens", type=int, default=150_000_000)
    f.add_argument("--test-max-tokens", type=int, default=None); f.add_argument("--no-download", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        parts = [(p.split("=")[0], int(p.split("=")[1])) for p in a.part]
        sample_parts(a.src, parts, a.out_dir, a.seed, a.min_doc_tokens)
    elif a.cmd == "tokenize":
        files = sorted(x for pat in a.inputs for x in glob.glob(pat))
        tokenize(files, a.out, a.max_tokens, fake_encoder=a.fake_encoder)
    else:
        fineweb2(a.lang, a.name, a.raw_dir, a.out_dir, a.train_shards, a.train_max_tokens, a.test_max_tokens,
                 download=not a.no_download)
    return 0


if __name__ == "__main__":
    sys.exit(main())
