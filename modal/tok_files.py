"""Tokenize DCLM-style jsonl.zst shards without multiprocessing pipes (sandbox-safe).

Uses the authors' encoder (nestor.data.token_documents.encode_documents via the
shim package) so the token stream is identical to scripts/preprocess_dclm.py:
documents in sorted file order, each followed by the cl100k end-of-text id.

Modes
-----
worker:  python tok_files.py worker --out-dir D --min-doc-tokens N f1 f2 ...
         -> for every file f: D/<stem>.bin (uint32) and D/<stem>.lens.npy (doc lengths, eot excluded)
driver:  python tok_files.py driver --root R --global-shards 1 --out OUT.bin --procs 4 [--min-doc-tokens N]
         [--target-tokens T] [--file-glob PATTERN] [--first-file I --last-file J]
         -> spawns workers (subprocess, logs to files), then concatenates per-file bins in sorted order.
"""
from __future__ import annotations

import argparse, glob, io, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def list_files(root: str, global_shards: list[int] | None) -> list[str]:
    files = []
    for gdir in sorted(d for d in os.listdir(root) if d.startswith("global-shard_")):
        gnum = int(gdir.split("_")[1])
        if global_shards is not None and gnum not in global_shards:
            continue
        gpath = os.path.join(root, gdir)
        for ldir in sorted(os.listdir(gpath)):
            lpath = os.path.join(gpath, ldir)
            if os.path.isdir(lpath):
                files += [os.path.join(lpath, f) for f in sorted(os.listdir(lpath)) if f.endswith(".jsonl.zst")]
    return files


def stem(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return "__".join(parts[-3:]).replace(".jsonl.zst", "")


def worker(args) -> None:
    import zstandard as zstd
    from nestor.data.token_documents import encode_documents, init_tokenizer_worker
    init_tokenizer_worker("cl100k_base", args.min_doc_tokens, None)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.files:
        out_bin = os.path.join(args.out_dir, stem(path) + ".bin")
        if os.path.exists(out_bin) and os.path.exists(out_bin[:-4] + ".lens.npy"):
            continue
        t0 = time.time(); ntok = 0; lens_all = []
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as fh, open(out_bin + ".part", "wb") as out:
            text_stream = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
            batch = []
            def flush(batch):
                nonlocal ntok
                arr, kept = encode_documents(batch)
                arr.tofile(out)
                ntok += int(arr.shape[0])
                # recover per-doc lengths from eot positions (eot excluded)
                eot_pos = np.flatnonzero(arr == 100257)
                prev = np.concatenate([[-1], eot_pos[:-1]])
                lens_all.append((eot_pos - prev - 1).astype(np.int64))
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = doc.get("text")
                if not text:
                    continue
                batch.append(text)
                if len(batch) >= 512:
                    flush(batch); batch = []
            if batch:
                flush(batch)
        os.replace(out_bin + ".part", out_bin)
        np.save(out_bin[:-4] + ".lens.npy", np.concatenate(lens_all) if lens_all else np.zeros(0, np.int64))
        print(f"[worker] {stem(path)}: {ntok:,} tokens in {time.time()-t0:.0f}s", flush=True)


def driver(args) -> None:
    gs = [int(x) for x in args.global_shards.split(",")] if args.global_shards else None
    files = list_files(args.root, gs)
    if args.last_file is not None:
        files = files[args.first_file: args.last_file + 1]
    else:
        files = files[args.first_file:]
    print(f"[driver] {len(files)} files -> {args.out}", flush=True)
    part_dir = args.out + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    chunks = [files[i::args.procs] for i in range(args.procs)]
    procs = []
    env = dict(os.environ)
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        log = open(os.path.join(HERE, "logs", f"tok_{os.path.basename(args.out)}_{i}.log"), "w")
        cmd = [sys.executable, os.path.abspath(__file__), "worker", "--out-dir", part_dir,
               "--min-doc-tokens", str(args.min_doc_tokens), *ch]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    t0 = time.time()
    for p, log in procs:
        p.wait(); log.close()
        if p.returncode != 0:
            raise SystemExit(f"worker failed with code {p.returncode}; see logs")
    # concatenate in sorted file order (same order as scripts/preprocess_dclm.py)
    total, docs = 0, 0
    with open(args.out + ".tmp", "wb") as out:
        for f in files:
            b = os.path.join(part_dir, stem(f) + ".bin")
            arr = np.fromfile(b, dtype=np.uint32)
            if args.target_tokens is not None and total + arr.shape[0] > args.target_tokens:
                # cut at the last eot at or before the target so documents stay whole
                need = args.target_tokens - total
                eots = np.flatnonzero(arr[:need] == 100257)
                arr = arr[: (eots[-1] + 1) if eots.size else 0]
            arr.tofile(out)
            total += int(arr.shape[0]); docs += int((arr == 100257).sum())
            if args.target_tokens is not None and total >= args.target_tokens - 10_000_000:
                break
    os.replace(args.out + ".tmp", args.out)
    from nestor.data.token_documents import write_bin_meta
    write_bin_meta(args.out, num_tokens=total, num_docs=docs, tokenizer_name="cl100k_base",
                   extra={"dclm_root": args.root, "global_shards": args.global_shards or "all",
                          "min_doc_tokens": args.min_doc_tokens, "files": [os.path.basename(f) for f in files],
                          "elapsed_sec": round(time.time() - t0, 1), "builder": "tok_files.py (per-file, same encoder)"})
    print(f"[driver] DONE {total:,} tokens, {docs:,} docs -> {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker"); w.add_argument("--out-dir", required=True)
    w.add_argument("--min-doc-tokens", type=int, default=1); w.add_argument("files", nargs="+")
    d = sub.add_parser("driver"); d.add_argument("--root", required=True); d.add_argument("--out", required=True)
    d.add_argument("--global-shards", default=None); d.add_argument("--procs", type=int, default=4)
    d.add_argument("--min-doc-tokens", type=int, default=1); d.add_argument("--target-tokens", type=int, default=None)
    d.add_argument("--first-file", type=int, default=0); d.add_argument("--last-file", type=int, default=None)
    a = ap.parse_args()
    worker(a) if a.mode == "worker" else driver(a)
