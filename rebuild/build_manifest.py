"""Build data\\data_manifest.json: every rebuilt .bin re-scanned (bytes, tokens, docs, sha256) and
compared with the reference counts of the original logs (+ known sha256 where one exists), plus cross-checks.

  python build_manifest.py [--out data\\data_manifest.json]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rebuild"))
import rebuild_all as RA  # noqa: E402  (EXP table)
import verify_bin as VB  # noqa: E402

PART0 = ("data/dclm/dclm_train16.bin.parts/global-shard_01_of_10__local-shard_0_of_10__shard_00000000_processed.bin",
         "d208526e3b0c39e5d351aac26f3e4eab6a8680fa8bc97b86a61ea1a4bda4285d", 292746848)
WORKER = re.compile(r"\[worker\] (\S+): ([\d,]+) tokens")


def prefix_sha256(path, nbytes, chunk=1 << 26):
    h, left = hashlib.sha256(), nbytes
    with open(path, "rb") as fh:
        while left > 0:
            b = fh.read(min(chunk, left))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def VB_sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def worker_counts(dirpath, out_base):
    d = {}
    for fp in sorted(glob.glob(os.path.join(dirpath, f"tok_{out_base}_*.log"))):
        for m in WORKER.finditer(open(fp, encoding="utf-8", errors="replace").read()):
            d[m.group(1)] = int(m.group(2).replace(",", ""))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "data_manifest.json"))
    a = ap.parse_args()
    files, summary = [], {"matched": 0, "mismatched": 0, "missing": 0, "no_reference": 0}
    for rel, exp in RA.EXP.items():
        path = os.path.join(ROOT, rel.replace("/", os.sep))
        ent = {"path": rel, "expected": ({"tokens": exp[0], "docs": exp[1], "sha256": exp[2]} if exp else None)}
        if not os.path.isfile(path):
            q = path + ".MISMATCH"
            ent.update({"bytes": None, "tokens": None, "docs": None, "sha256": None, "match": False,
                        "problems": ["file missing" + (" (quarantined as .MISMATCH)" if os.path.isfile(q) else "")]})
            if os.path.isfile(q):
                r = VB.check(q, *(exp or (None, None, None)))
                ent.update({"bytes": r["bytes"], "tokens": r["tokens"], "docs": r["docs"], "sha256": r["sha256"],
                            "quarantined_as": rel + ".MISMATCH"})
            summary["missing"] += 1
            files.append(ent)
            continue
        r = VB.check(path, *(exp or (None, None, None)))
        ent.update({"bytes": r["bytes"], "tokens": r["tokens"], "docs": r["docs"], "sha256": r["sha256"],
                    "match": r["match"], "problems": r["problems"], "empty_docs": r["empty_docs"],
                    "ends_with_eot": r["ends_with_eot"], "max_token_id": r["max_token_id"]})
        if exp is None:
            ent["note"] = "NEW in no reference counts exist; counts recorded only"
            summary["no_reference"] += 1
        elif r["match"]:
            summary["matched"] += 1
        else:
            summary["mismatched"] += 1
        files.append(ent)
        print(f"{rel:55s} {ent['tokens']:>14,} {ent['docs']:>10,} match={ent['match']}", flush=True)

    extra = []
    # 1. the 12-file train bin must be the exact prefix of the 16-file bin
    t16, t12 = os.path.join(ROOT, "data/dclm/dclm_train16.bin"), os.path.join(ROOT, "data/dclm/dclm_train.bin")
    if os.path.isfile(t16) and os.path.isfile(t12):
        n12 = os.path.getsize(t12)
        extra.append({"check": "dclm_train.bin == first %d bytes of dclm_train16.bin" % n12,
                      "ok": prefix_sha256(t16, n12) == [f for f in files if f["path"].endswith("dclm_train.bin")][0]["sha256"]})
    # 2. original per-file token counts (surviving worker logs) vs the rebuilt per-file counts
    orig_dir = os.path.join(ROOT, "logs_orig_20261113")
    for rel in RA.EXP:
        if not rel.startswith(("data/dclm/", "data/longctx/")):
            continue
        base = os.path.basename(rel)
        old, new = worker_counts(orig_dir, base), worker_counts(os.path.join(ROOT, "logs"), base)
        if old or new:
            ent = {"check": f"per-file token counts of {base} vs original worker logs", "files_original": len(old),
                   "files_rebuilt": len(new), "ok": (old == new) if old else None,
                   "differences": {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}}
            if not old:
                ent["note"] = "no original worker logs: this bin is new in the design (nothing to compare)"
                ent["differences"] = {}
            extra.append(ent)
    # 3. the per-file part with a known sha256 (local_token_hashes.json)
    p0 = os.path.join(ROOT, PART0[0].replace("/", os.sep))
    if os.path.isfile(p0):
        r = VB.scan(p0)
        extra.append({"check": "train16 part 0 (global-shard_01 file 0) sha256 vs local_token_hashes.json", "bytes": r["bytes"],
                      "sha256": r["sha256"], "ok": r["sha256"] == PART0[1] and r["bytes"] == PART0[2]})
    # 4. tokenizer provenance + raw inputs
    cache = os.path.join(ROOT, "data", "tiktoken", "9b5ad71b2ce5302211f9c61530b329a4922fc6a4")
    if os.path.isfile(cache):
        extra.append({"check": "cl100k_base BPE file (tiktoken cache)", "sha256": hashlib.sha256(open(cache, "rb").read()).hexdigest()})
    # 5. raw inputs: every planned HF file re-hashed now and compared with the LFS oid from the HF tree API
    raw = []
    plan_path = os.path.join(ROOT, "rebuild", "raw_plan.json")
    if os.path.isfile(plan_path):
        for it in json.load(open(plan_path, encoding="utf-8")):
            d = it["dest"]
            ok = os.path.isfile(d) and os.path.getsize(d) == it["size"]
            digest = VB_sha(d) if ok else None
            raw.append({"repo": it["repo"], "path": it["path"], "local": os.path.relpath(d, ROOT).replace("\\", "/"),
                        "bytes": it["size"], "sha256": digest, "lfs_sha256": it["sha256"], "match": bool(digest) and digest == it["sha256"]})
        extra.append({"check": "raw HF files re-hashed vs the LFS oid published by the HF tree API", "n_files": len(raw),
                      "n_ok": sum(1 for r in raw if r["match"]), "bytes": sum(r["bytes"] for r in raw),
                      "ok": all(r["match"] for r in raw)})
    # 6. software versions of the interpreter that produced the bins
    env_info = None
    try:
        import subprocess
        code = ("import sys, tiktoken, zstandard, pyarrow, numpy, huggingface_hub; print(sys.version.split()[0], tiktoken.__version__, "
                "zstandard.__version__, pyarrow.__version__, numpy.__version__, huggingface_hub.__version__)")
        o = subprocess.run([RA.VENV_PY, "-c", code], capture_output=True, env=RA.make_env(RA.Job("v", [], 1, 1, [])), timeout=300).stdout.decode().split()
        env_info = dict(zip(["python", "tiktoken", "zstandard", "pyarrow", "numpy", "huggingface_hub"], o))
    except Exception as e:  # pragma: no cover
        env_info = {"error": str(e)}
    commands = {j.name: "python " + " ".join(j.argv) for j in RA.build_jobs()}
    policy = {"cwd": ROOT, "max_compute_threads": RA.BUDGET, "priority": "BelowNormal",
              "tiktoken_pool_threads": "1 per worker process for --procs 4 jobs, 2 for single-process jobs (TIKTOKEN_NUM_THREADS via rebuild/threadcap)",
              "OMP/MKL/RAYON_NUM_THREADS": "1 for --procs 4 jobs, 2 otherwise", "gpu": "not used (CUDA_VISIBLE_DEVICES=-1)",
              "PYTHONPATH": "shim;rebuild/threadcap;sitecustom", "TIKTOKEN_CACHE_DIR": "data/tiktoken", "HF_HUB_DISABLE_XET": "1"}
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": ROOT, "tokenizer": "cl100k_base (tiktoken)",
           "interpreter": RA.VENV_PY, "software": env_info, "resource_policy": policy, "commands": commands,
           "summary": summary, "files": files, "extra_checks": extra, "raw_inputs": raw}
    json.dump(man, open(a.out, "w", encoding="utf-8"), indent=1)
    print("summary", summary)
    for e in extra:
        print("extra:", e["check"], "->", e.get("ok", e.get("sha256", "")))
    return 0 if summary["mismatched"] == 0 and summary["missing"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
