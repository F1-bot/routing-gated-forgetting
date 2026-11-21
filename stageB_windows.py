"""Per-window re-evaluation of reproduction stage B (paired CIs for gain/forgetting), run on the GPU after stage B.
Windows: 500 evenly spaced 4096-token windows (2,048,000 tokens) on DCLM val09 and on the EVALUATION half of each
target test bin (prereg sec. 4.3 layout: span 0.5-1; PG-19 split at a book boundary). Same windows for every model,
so model-vs-model differences of forgetting are paired by window. Resumable: an existing output dir is skipped."""
import glob, json, os, subprocess, sys, time
PY = sys.executable
CORPORA = {"pg19": "0.5,1,doc", "owm": "0.5,1", "algstack": "0.5,1", "fw2de": "0.5,1"}
MODELS = ["dense_100m", "moe_100m", "nestor_100m"]
OUT = "stageB_windows"

def base_ckpt(exp):
    """Pinned base checkpoint (NESTOR_CKPT_PINS = ckpt_pins.json, chain6b) or, without pins, the newest by mtime."""
    pins = os.environ.get("NESTOR_CKPT_PINS", "").strip()
    if pins:
        doc = json.load(open(pins, encoding="utf-8-sig"))
        e = (doc.get("pins") or doc).get(exp)
        p = e.get("path") if isinstance(e, dict) else e
        if not p:
            return None
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            raise SystemExit(f"pinned checkpoint for {exp} missing: {p}")
        return p
    c = sorted(glob.glob(f"outputs/{exp}_b05/*/checkpoints/model_final.pt"), key=os.path.getmtime)
    return os.path.abspath(c[-1]) if c else None

def adapted(R, exp, corpus):
    r = R.get(f"B|{exp}|after|{corpus}|target")
    return r.get("ckpt") if r else None

def run(exp, ckpt, tag, sets):
    out = os.path.join(OUT, exp, tag)
    if os.path.exists(os.path.join(out, "DONE")):
        print(f"[stageB_windows] skip {exp} {tag}", flush=True); return 0
    cmd = [PY, "eval_windows.py", "run", exp, ckpt, "--select", "even", "--out", out,
           "--set", "dclm09=data/dclm/dclm_val.bin:2048000"]
    for c in sets:
        cmd += ["--set", f"{c}=data/domain/{c}_test.bin:2048000", "--span", f"{c}={CORPORA[c]}"]
    t0 = time.time()
    p = subprocess.run(cmd)
    if p.returncode == 0:
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "DONE"), "w").write(json.dumps({"ckpt": ckpt, "sets": sets, "seconds": time.time() - t0}))
    print(f"[stageB_windows] {exp} {tag} rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    return p.returncode

def main():
    R = {}
    for l in open("results/continual.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); R[r["key"]] = r
    bad = 0
    for exp in MODELS:
        b = base_ckpt(exp)
        if b:
            bad += run(exp, b, "base", list(CORPORA)) != 0
        for c in CORPORA:
            a = adapted(R, exp, c)
            if a and os.path.exists(a):
                bad += run(exp, a, f"adapt_{c}", [c]) != 0
            else:
                print(f"[stageB_windows] missing adapted checkpoint {exp} {c}", flush=True)
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
