"""Thread-budgeted, resumable scheduler for the tokenized-dataset rebuild.

Runs the ORIGINAL commands (cwd = ., interpreter = .venv\\nestor, BelowNormal priority, GPU hidden),
at most BUDGET compute threads at once (each job's weight = worker processes x tiktoken pool threads), verifies every
output against the logged token/document counts (and the known sha256 where one exists), quarantines mismatches
(<bin>.MISMATCH) and never lets a dependent step run on a mismatching input.

  python rebuild_all.py [--dry-run] [--no-env-gate] [--only NAME[,NAME]]

Control files: rebuild\\STOP (no new jobs are started), state in rebuild\\state\\<job>.json, log in logs\\rebuild.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rebuild"))
import verify_bin as VB  # noqa: E402

VENV_PY = os.path.join(ROOT, ".venv", "nestor", "Scripts", "python.exe")
CONDA_LIBBIN = r"<conda>\envs\nestor\Library\bin"
SHIM = os.path.join(ROOT, "shim")
THREADCAP = os.path.join(ROOT, "rebuild", "threadcap")
SITECUSTOM = os.path.join(ROOT, "sitecustom")
TMPDIR = os.path.join(ROOT, ".tmp")
STATE_DIR = os.path.join(ROOT, "rebuild", "state")
VERIFY_DIR = os.path.join(ROOT, "rebuild", "verify")
LOGDIR = os.path.join(ROOT, "logs", "rebuild")
LOG = os.path.join(LOGDIR, "scheduler.log")
ENV_LOG = os.path.join(ROOT, "logs", "env_check.log")
STOP = os.path.join(ROOT, "rebuild", "STOP")
BUDGET = 6
MAX_DOMAIN_PARALLEL = 2
BELOW_NORMAL, NO_WINDOW = 0x00004000, 0x08000000

# expected (tokens, docs, sha256-or-None) from the original logs / modal hash lists
EXP = {
    "data/dclm/dclm_train16.bin": (1738922107, 1425616, "455754d1d0098da83fb3bfc33d96e292520751a1e431d7a037e6f1bf0be9cd16"),
    "data/dclm/dclm_train.bin": (1304222207, 1065522, None),
    "data/dclm/dclm_val.bin": (19999909, 17005, "0644eac43505ad8f3ba78c8dd8580862e7b69dcbf24080dd3e8082692186ed04"),
    "data/dclm/dclm_val10.bin": (19996569, 16702, "9bdfce3d6555fb083e7c01d92bdc23ca48454cdce296606179a76de98c0bd14e"),
    "data/longctx/dclm_longdoc_ge8192_train.bin": (228500837, 15103, None),
    "data/longctx/dclm_longdoc_ge16384_train.bin": (99976687, 3512, None),
    "data/longctx/dclm_longdoc_ge8192_val.bin": (20209183, 1326, None),
    "data/longctx/dclm_longdoc_ge16384_val.bin": (9299515, 352, None),
    "data/domain/algstack_test.bin": (42160647, 17997, None),
    "data/domain/algstack_train.bin": (1676450112, 458783, None),
    "data/domain/fw2de_test.bin": (60571959, 65176, None),
    "data/domain/fw2de_train.bin": (150124418, 157184, None),
    "data/domain/owm_test.bin": (63933458, 31577, None),
    "data/domain/owm_train.bin": (150430566, 74752, None),
    "data/domain/pg19_test.bin": (9872926, 100, None),
    "data/domain/pg19_train.bin": (118347447, 1244, None),
    "data/domain/fr_test.bin": (39038319, 44245, None),
    "data/domain/fr_train.bin": (150285222, 147456, None),
    "data/domain/pycode_test.bin": (12495229, 5376, None),
    "data/streams/code_eval.bin": (2251640, 936, None),
    "data/streams/code_route.bin": (4400027, 1897, None),
    "data/streams/math_eval.bin": (2201168, 1076, None),
    "data/streams/math_route.bin": (4411086, 2024, None),
    "data/streams/german_eval.bin": (2200177, 2423, None),
    "data/streams/german_route.bin": (2201255, 2155, None),
    "data/streams/books_route.bin": (2228733, 22, None),
    "data/streams/pycode_eval.bin": (2204757, 985, None),
    "data/streams/pycode_route.bin": (4400367, 1826, None),
    "data/streams/fr_eval.bin": (2200635, 2337, None),
    "data/streams/fr_route.bin": (2208905, 2344, None),
    "data/dclm/dclm_cont_heldout.bin": None,  # NEW in no reference counts exist
}

_lock = threading.Lock()


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with _lock:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


class Job:
    def __init__(self, name, argv, weight, tk, outputs, inputs=(), after=(), family="", torch=False, canary=False):
        self.name, self.argv, self.weight, self.tk = name, argv, weight, tk
        self.outputs, self.inputs, self.after = list(outputs), list(inputs), list(after)
        self.family, self.torch, self.canary = family, torch, canary
        self.status = "PENDING"
        self.result = {}
        self.proc = None


def tf(out, *extra, procs, shards, w, tk, **kw):
    name = os.path.basename(out)[:-4]
    return Job(name, ["tok_files.py", "driver", "--root", r"data\dclm\raw", "--global-shards", str(shards), "--out", out,
                      "--procs", str(procs), *extra], w, tk, [out.replace("\\", "/")], family="dclm", **kw)


def dom(name, out, pattern, *extra, **kw):
    return Job(name, ["tok_domain.py", "--out", out, *extra, pattern], 2, 2, [out], family="domain", **kw)


def build_jobs():
    R = "data/domain/raw"
    sh = lambda g, n: (f"data/dclm/raw/global-shard_{g:02d}_of_10/local-shard_0_of_10/*.jsonl.zst", n)
    J = []
    J.append(tf(r"data\dclm\dclm_val.bin", "--target-tokens", "20000000", procs=1, shards=9, w=2, tk=2,
                inputs=[sh(9, 1)], canary=True))
    J.append(dom("pg19_test", "data/domain/pg19_test.bin", f"{R}/emozilla__pg19/data/test-*.parquet",
                 inputs=[(f"{R}/emozilla__pg19/data/test-*.parquet", 1)], canary=True))
    J.append(tf(r"data\dclm\dclm_val10.bin", "--target-tokens", "20000000", procs=1, shards=10, w=2, tk=2, inputs=[sh(10, 1)]))
    J.append(tf(r"data\dclm\dclm_train16.bin", procs=4, shards=1, w=4, tk=1, inputs=[sh(1, 16)]))
    J.append(dom("owm_test", "data/domain/owm_test.bin", f"{R}/EleutherAI__proof-pile-2/open-web-math/test/*.jsonl.zst",
                 inputs=[(f"{R}/EleutherAI__proof-pile-2/open-web-math/test/*.jsonl.zst", 1)]))
    J.append(dom("fw2de_test", "data/domain/fw2de_test.bin", f"{R}/HuggingFaceFW__fineweb-2/data/deu_Latn/test/*.parquet",
                 inputs=[(f"{R}/HuggingFaceFW__fineweb-2/data/deu_Latn/test/*.parquet", 1)]))
    J.append(dom("algstack_test", "data/domain/algstack_test.bin", f"{R}/EleutherAI__proof-pile-2/algebraic-stack/test/*.jsonl.zst",
                 inputs=[(f"{R}/EleutherAI__proof-pile-2/algebraic-stack/test/*.jsonl.zst", 19)]))
    J.append(dom("pg19_train", "data/domain/pg19_train.bin", f"{R}/emozilla__pg19/data/train-*.parquet", "--max-tokens", "150000000",
                 inputs=[(f"{R}/emozilla__pg19/data/train-*.parquet", 1)]))
    J.append(dom("owm_train", "data/domain/owm_train.bin", f"{R}/EleutherAI__proof-pile-2/open-web-math/train/*.jsonl.zst",
                 "--max-tokens", "150000000", inputs=[(f"{R}/EleutherAI__proof-pile-2/open-web-math/train/*.jsonl.zst", 1)]))
    J.append(dom("algstack_train", "data/domain/algstack_train.bin", f"{R}/EleutherAI__proof-pile-2/algebraic-stack/train/*.jsonl.zst",
                 inputs=[(f"{R}/EleutherAI__proof-pile-2/algebraic-stack/train/*.jsonl.zst", 16)]))
    J.append(tf(r"data\dclm\dclm_train.bin", "--last-file", "11", procs=4, shards=1, w=4, tk=1, inputs=[sh(1, 16)]))
    J.append(dom("fw2de_train", "data/domain/fw2de_train.bin", f"{R}/HuggingFaceFW__fineweb-2/data/deu_Latn/train/*.parquet",
                 "--max-tokens", "150000000", inputs=[(f"{R}/HuggingFaceFW__fineweb-2/data/deu_Latn/train/*.parquet", 1)]))
    J.append(tf(r"data\longctx\dclm_longdoc_ge8192_train.bin", "--min-doc-tokens", "8192", procs=4, shards=2, w=4, tk=1, inputs=[sh(2, 12)]))
    J.append(tf(r"data\longctx\dclm_longdoc_ge16384_train.bin", "--min-doc-tokens", "16384", procs=4, shards=2, w=4, tk=1, inputs=[sh(2, 12)]))
    J.append(tf(r"data\longctx\dclm_longdoc_ge8192_val.bin", "--min-doc-tokens", "8192", procs=1, shards=10, w=2, tk=2, inputs=[sh(10, 1)]))
    J.append(tf(r"data\longctx\dclm_longdoc_ge16384_val.bin", "--min-doc-tokens", "16384", procs=1, shards=10, w=2, tk=2, inputs=[sh(10, 1)]))
    J.append(tf(r"data\dclm\dclm_cont_heldout.bin", "--target-tokens", "30000000", procs=1, shards=3, w=2, tk=2, inputs=[sh(3, 1)]))
    J.append(Job("pycode_tok", ["pycode_tok.py", "--src", "data/raw/codeparrot/file-000000000054.json.gz",
                                "--out", "data/domain/pycode_test.bin", "--max-tokens", "12000000"], 2, 2,
                 ["data/domain/pycode_test.bin"], inputs=[("data/raw/codeparrot/file-000000000054.json.gz", 1)],
                 family="domain", torch=True))
    streams_out = [k for k in EXP if k.startswith("data/streams/")] + ["data/domain/fr_test.bin", "data/domain/fr_train.bin"]
    J.append(Job("streams", ["prep_all_streams.py"], 2, 2, streams_out,
                 inputs=[("data/raw/fineweb-2/data/fra_Latn/train/000_00000.parquet", 1),
                         ("data/raw/fineweb-2/data/fra_Latn/test/*.parquet", 1)],
                 after=["algstack_test", "owm_test", "fw2de_test", "fw2de_train", "pg19_train", "pycode_tok"],
                 family="domain", torch=True))
    return J


def make_env(job):
    e = dict(os.environ)
    e.pop("PYTHONSAFEPATH", None)
    if job.argv and job.argv[0] == "tok_files.py":  # 4 worker processes x 1 pool thread, or 1 worker x 2 pool threads
        n = "1" if job.argv[job.argv.index("--procs") + 1] == "4" else "2"
    else:
        n = str(job.tk)
    e.update({
        "PYTHONPATH": ";".join([SHIM, THREADCAP, SITECUSTOM]),
        "TIKTOKEN_CACHE_DIR": os.path.join(ROOT, "data", "tiktoken"),
        "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",  # '' does not hide the GPU on Windows (torch.cuda.is_available() was True); -1 does
        "PYTHONUTF8": "1", "TEMP": TMPDIR, "TMP": TMPDIR,
        "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "RAYON_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n, "TIKTOKEN_NUM_THREADS": str(job.tk),
    })
    if not job.torch:  # torch-free process: conda numpy needs its Library\bin first on PATH; torch processes must not get it
        e["PATH"] = CONDA_LIBBIN + ";" + e.get("PATH", "")
    return e


def env_gate_open():
    try:
        b = open(ENV_LOG, "rb").read()
    except OSError:
        return False
    s = b.decode("utf-16", "replace") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8", "replace")
    return "ENV_OK" in s


def preflight(torch):
    probe = Job("preflight", [], 1, 1, [], torch=torch)
    code = ("import sys, tiktoken, tiktoken.core as c, zstandard, pyarrow, numpy, huggingface_hub; "
            "import nestor.data.token_documents as T; "
            "print('PF', sys.version.split()[0], tiktoken.__version__, zstandard.__version__, pyarrow.__version__, "
            "numpy.__version__, huggingface_hub.__version__, getattr(c.Encoding.encode_ordinary_batch, '_threadcap', None))")
    if torch:  # import only: no CUDA call of any kind (the data jobs must not touch the GPU)
        code += "; import os, torch; print('PFT', torch.__version__, os.environ.get('CUDA_VISIBLE_DEVICES'))"
    r = subprocess.run([VENV_PY, "-c", code], capture_output=True, env=make_env(probe), cwd=ROOT, timeout=600,
                       creationflags=BELOW_NORMAL | NO_WINDOW)
    out = r.stdout.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8", "replace")
    pf = [ln for ln in out.splitlines() if ln.startswith("PF ")]
    ok = r.returncode == 0 and bool(pf) and pf[0].split()[-1] == "1"
    if torch:
        pft = [ln for ln in out.splitlines() if ln.startswith("PFT ")]
        ok = ok and bool(pft) and pft[0].split()[-1] == "-1"
    return ok, (out + err)[-800:]


def inputs_ready(job):
    for pat, n in job.inputs:
        if len(glob.glob(os.path.join(ROOT, pat.replace("/", os.sep)))) != n:
            return False
    return True


def run_job(job):
    t0 = time.time()
    st = {"started": time.strftime("%H:%M:%S"), "argv": job.argv, "weight": job.weight, "tiktoken_threads": job.tk}
    logp = os.path.join(LOGDIR, f"{job.name}.log")
    with open(logp, "w", encoding="utf-8") as lf:
        # no stdin=DEVNULL: opening the NUL device is denied (WinError 5) in this sandbox
        job.proc = subprocess.Popen([VENV_PY, *job.argv], cwd=ROOT, env=make_env(job), stdout=lf, stderr=subprocess.STDOUT,
                                    creationflags=BELOW_NORMAL | NO_WINDOW)
        log(f"START {job.name} pid={job.proc.pid} weight={job.weight} :: {' '.join(job.argv)}")
        rc = job.proc.wait()
    st["rc"] = rc
    st["run_seconds"] = round(time.time() - t0, 1)
    if rc != 0:
        job.status = "FAILED"
        st["error"] = f"exit code {rc}; see {logp}"
    else:
        outs, bad = [], []
        for out in job.outputs:
            path = os.path.join(ROOT, out.replace("/", os.sep))
            if not os.path.isfile(path):
                bad.append(f"{out}: missing")
                continue
            exp = EXP.get(out)
            res = VB.check(path, *(exp if exp else (None, None, None)))
            json.dump(res, open(os.path.join(VERIFY_DIR, os.path.basename(out) + ".json"), "w", encoding="utf-8"), indent=1)
            outs.append({k: res[k] for k in ("path", "bytes", "tokens", "docs", "sha256", "match", "problems", "empty_docs")})
            if res["problems"]:
                bad.append(f"{out}: " + "; ".join(res["problems"]))
                if res["match"] is False or any("sidecar" in p or "eot" in p or "vocabulary" in p for p in res["problems"]):
                    for suffix in ("", ".meta.json"):
                        if os.path.exists(path + suffix):
                            os.replace(path + suffix, path + suffix + ".MISMATCH")
        st["outputs"] = outs
        job.status = "MISMATCH" if bad else "OK"
        if bad:
            st["mismatch"] = bad
    st["ended"] = time.strftime("%H:%M:%S")
    job.result = st
    json.dump({"status": job.status, **st}, open(os.path.join(STATE_DIR, job.name + ".json"), "w", encoding="utf-8"), indent=1)
    log(f"END {job.name}: {job.status} rc={rc} in {st['run_seconds']}s" + (f" :: {st.get('mismatch') or st.get('error')}" if job.status != "OK" else ""))


def resume_state(jobs):
    for j in jobs:
        fp = os.path.join(STATE_DIR, j.name + ".json")
        if not os.path.isfile(fp):
            continue
        s = json.load(open(fp, encoding="utf-8"))
        if s.get("status") == "OK" and s.get("outputs") and all(
                os.path.isfile(o["path"]) and os.path.getsize(o["path"]) == o["bytes"] for o in s["outputs"]):
            j.status, j.result = "OK", s
            log(f"RESUME {j.name}: already OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-env-gate", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for d in (STATE_DIR, VERIFY_DIR, LOGDIR, TMPDIR):
        os.makedirs(d, exist_ok=True)
    try:  # BelowNormal for the scheduler too; children inherit and are also created BelowNormal
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)
    except Exception as e:  # pragma: no cover
        log(f"priority not set: {e}")
    jobs = build_jobs()
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j.name in keep]
    byname = {j.name: j for j in jobs}
    if a.dry_run:
        for j in jobs:
            print(f"{j.name:24s} w={j.weight} tk={j.tk} torch={j.torch} canary={j.canary} fam={j.family} after={j.after}\n   {' '.join(j.argv)}")
        return 0
    resume_state(jobs)
    log(f"scheduler start: {len(jobs)} jobs, budget {BUDGET} compute threads")
    gate_logged = False
    preflight_done = {False: False, True: False}
    running = {}
    last_wait_log = 0.0
    while True:
        pend = [j for j in jobs if j.status == "PENDING"]
        for j in list(running):
            if not running[j].is_alive():
                running.pop(j)
        if not pend and not running:
            break
        stop = os.path.exists(STOP)
        used = sum(j.weight for j in running)
        if pend and not stop and (a.no_env_gate or env_gate_open()):
            if not gate_logged:
                log("env gate open (ENV_OK found in logs/env_check.log)")
                gate_logged = True
            for j in pend:
                # propagate failures
                if any(byname[x].status in ("FAILED", "MISMATCH", "BLOCKED") for x in j.after if x in byname):
                    j.status = "BLOCKED"
                    log(f"BLOCKED {j.name}: upstream {[x for x in j.after if byname[x].status in ('FAILED', 'MISMATCH', 'BLOCKED')]}")
                    continue
                if j.family and not j.canary:
                    cs = [c for c in jobs if c.canary and c.family == j.family]
                    if any(c.status in ("FAILED", "MISMATCH") for c in cs):
                        j.status = "BLOCKED"
                        log(f"BLOCKED {j.name}: canary of family {j.family} failed")
                        continue
                    if any(c.status != "OK" for c in cs):
                        continue
                if any(x in byname and byname[x].status != "OK" for x in j.after):
                    continue
                if not inputs_ready(j):
                    continue
                if used + j.weight > BUDGET:
                    continue
                if j.family == "domain" and sum(1 for r in running if r.family == "domain") >= MAX_DOMAIN_PARALLEL:
                    continue  # tok_domain "one process at a time or two"
                if not preflight_done[j.torch]:
                    ok, info = preflight(j.torch)
                    log(f"preflight(torch={j.torch}) ok={ok}: {info.strip()[-300:]}")
                    if not ok:
                        log("preflight failed: not starting any job; will retry in 120 s")
                        time.sleep(120)
                        break
                    preflight_done[j.torch] = True
                j.status = "RUNNING"
                t = threading.Thread(target=run_job, args=(j,), daemon=False)
                running[j] = t
                used += j.weight
                t.start()
        if time.time() - last_wait_log > 600:
            last_wait_log = time.time()
            waiting = {j.name: ("env gate" if not (a.no_env_gate or env_gate_open()) else
                                "inputs" if not inputs_ready(j) else
                                "deps/canary/budget") for j in pend}
            log(f"waiting: running={[j.name for j in running]} used={used}/{BUDGET} pending={waiting}")
        time.sleep(5)
        if stop and not running:
            log("STOP file present and nothing running: exit")
            break
    summary = {j.name: j.status for j in jobs}
    log("scheduler end: " + json.dumps(summary))
    json.dump(summary, open(os.path.join(STATE_DIR, "_summary.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
