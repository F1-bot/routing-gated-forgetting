"""Continual-learning phase (scaled down), driven by the repo's own recipes.

Stage A  pretraining evals: held-out CE at 4k of the three 0.5B-token checkpoints
          (shard-09 set as in the paper, 10M tokens, fp32; shard-10 set as a robustness check).
Stage B  domain adaptation at 4k (paper: 32k after a 32k extension): 4 corpora x {dense, moe, nestor},
          96 steps x 262,144 tokens (~25M tokens; paper 0.5B), Muon 0.0025 / AdamW 7.5e-5, 10% warmup,
          cosine; routers frozen for moe/nestor (NESTOR_FREEZE_ROUTERS=1), dense fully fine-tuned.
          Gain on the target test stream, forgetting on DCLM held-out (both before and after, fp32).
Stage C  context extension 4k -> 8k and 16k: repo ExtensionRun recipe (rope base, LRs, warmup, r(S) from
          nestor.continual.extension.extension_flop_ratio) with 48 equal-token steps x 1,048,576 tokens
          (~50M tokens; paper 477 steps). Nestor also at equal FLOPs on new documents: ceil(48 / r) steps,
          LRs x r. Held-out: first S tokens of 200 documents >= S tokens (shard 10), fp32, run's rope base.
Every result is appended to results/continual.jsonl; finished items are skipped on restart.
"""
from __future__ import annotations

import glob, json, math, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "repo"))
PY = os.path.join(HERE, ".venv", "nestor", "Scripts", "python.exe")
RES = os.path.join(HERE, "results", "continual.jsonl")
os.makedirs("results", exist_ok=True)
MODELS = ["dense_100m", "moe_100m", "nestor_100m"]
PRE_MICRO = {"dense_100m": 4, "moe_100m": 2, "nestor_100m": 4}      # sequences of 4096 per micro-step
PRE_CKPT = {"dense_100m": False, "moe_100m": False, "nestor_100m": True}
CORPORA = ["pg19", "owm", "algstack", "fw2de"]
EVAL_TOKENS_4K = 2_000_000


def log(msg: str) -> None:
    line = f"[continual] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open("logs/continual_driver.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def done_keys() -> set[str]:
    if not os.path.exists(RES):
        return set()
    return {json.loads(l)["key"] for l in open(RES, encoding="utf-8") if l.strip()}


def record(key: str, **kw) -> None:
    with open(RES, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}) + "\n")


def pretrained(exp: str) -> str | None:
    c = sorted(glob.glob(f"outputs/{exp}_b05/*/checkpoints/model_final.pt"), key=os.path.getmtime)
    return os.path.abspath(c[-1]) if c else None


def evaluate(exp: str, ckpt: str, val_bin: str, *, block: int = 4096, rope: int | None = None,
             first_per_doc: bool = False, docs: int = 200, max_tokens: int = EVAL_TOKENS_4K) -> dict | None:
    env = dict(os.environ, DCLM_BLOCK_SIZE=str(block))
    if rope:
        env["DCLM_ROPE_BASE"] = str(rope)
    cmd = [PY, "eval_ckpt.py", exp, ckpt, val_bin, "--block", str(block), "--max-tokens", str(max_tokens)]
    if first_per_doc:
        cmd += ["--first-per-doc", "--docs", str(docs)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    log(f"EVAL FAILED {exp} {val_bin}: {p.stdout[-400:]} {p.stderr[-800:]}")
    return None


def train(name: str, exp: str, env_over: dict, steps: int, warmup: int, seq: int) -> str | None:
    """Run one continued-training job with OOM fallbacks; return its model_final.pt."""
    tokens_per_step = env_over.pop("_TOKENS_PER_STEP")
    base_micro_tokens = PRE_MICRO[exp] * 4096
    attempts = []
    micro = max(1, base_micro_tokens // seq)
    attempts.append((micro, PRE_CKPT[exp]))
    if not PRE_CKPT[exp]:
        attempts.append((micro, True))
    if micro > 1:
        attempts.append((micro // 2, True))
    for micro, ckpt in attempts:
        accum = tokens_per_step // (micro * seq)
        root = os.path.join(HERE, "outputs", "continual", f"{name}_m{micro}{'c' if ckpt else ''}")
        env = dict(os.environ, **env_over, NESTOR_OUTPUT_ROOT=root, MEM_FRAC="0.85", DCLM_LOG_FREQ="5",
                   DCLM_CKPT_FREQ="100000", NESTOR_ACTIVATION_CHECKPOINT="1" if ckpt else "0",
                   NESTOR_ACTIVATION_CHECKPOINT_SPLIT="1" if ckpt else "0",
                   NESTOR_ACTIVATION_CHECKPOINT_LEAN="1" if ckpt else "0")
        logf = f"logs/cont_{name}_m{micro}{'c' if ckpt else ''}.log"
        log(f"train {name}: {exp} steps={steps} warmup={warmup} seq={seq} micro={micro} accum={accum} ckpt={ckpt}")
        t0 = time.time()
        with open(logf, "w", encoding="utf-8") as lf:
            p = subprocess.run([PY, "run_exp.py", exp, "--steps", str(steps), "--warmup", str(warmup),
                                "--micro", str(micro), "--accum", str(accum)], stdout=lf, stderr=subprocess.STDOUT, env=env)
        txt = open(logf, encoding="utf-8", errors="replace").read()
        final = sorted(glob.glob(os.path.join(root, "**", "model_final.pt"), recursive=True))
        if p.returncode == 0 and final:
            log(f"train {name} done in {(time.time()-t0)/60:.1f} min")
            return final[-1]
        if "OutOfMemoryError" in txt:
            log(f"train {name} OOM at micro={micro} ckpt={ckpt}; falling back")
            shutil.rmtree(root, ignore_errors=True)
            continue
        log(f"train {name} FAILED rc={p.returncode}; see {logf}")
        return None
    return None


def stage_a() -> None:
    keys = done_keys()
    for exp in MODELS:
        ck = pretrained(exp)
        if ck is None:
            log(f"stage A: no checkpoint for {exp}")
            continue
        for tag, vb, ntok in [("val09", "data/dclm/dclm_val.bin", 10_000_000), ("val10", "data/dclm/dclm_val10.bin", 5_000_000)]:
            key = f"A|{exp}|{tag}"
            if key in keys or not os.path.exists(vb):
                continue
            r = evaluate(exp, ck, vb, max_tokens=ntok)
            if r:
                record(key, stage="pretrain_eval", model=exp, set=tag, ce=r["ce"], tokens=r["tokens"], ckpt=ck)
                log(f"{key}: ce={r['ce']:.4f} ({r['tokens']:,} tokens)")


def stage_b() -> None:
    for exp in MODELS:
        ck = pretrained(exp)
        if ck is None:
            continue
        sets = {c: f"data/domain/{c}_test.bin" for c in CORPORA} | {"dclm": "data/dclm/dclm_val.bin"}
        for tag, vb in sets.items():
            key = f"B|{exp}|before|{tag}"
            if key in done_keys():
                continue
            r = evaluate(exp, ck, vb)
            if r:
                record(key, stage="adapt_before", model=exp, set=tag, ce=r["ce"], tokens=r["tokens"])
                log(f"{key}: ce={r['ce']:.4f}")
    for corpus in CORPORA:
        for exp in MODELS:
            ck = pretrained(exp)
            key_t, key_d = f"B|{exp}|after|{corpus}|target", f"B|{exp}|after|{corpus}|dclm"
            if ck is None or (key_t in done_keys() and key_d in done_keys()):
                continue
            env = {"DCLM_BLOCK_SIZE": "4096", "DCLM_DATA_PATH": os.path.abspath(f"data/domain/{corpus}_train.bin"),
                   "DCLM_INIT_CHECKPOINT": ck, "DCLM_LR_MUON": "0.0025", "DCLM_LR_ADAMW": "7.5e-05",
                   "DCLM_RUN_SUFFIX": f"adapt_{corpus}", "_TOKENS_PER_STEP": 262_144}
            if exp != "dense_100m":
                env["NESTOR_FREEZE_ROUTERS"] = "1"
            fin = train(f"{exp}_adapt_{corpus}", exp, env, steps=96, warmup=10, seq=4096)
            if not fin:
                continue
            for key, tag, vb in [(key_t, corpus, f"data/domain/{corpus}_test.bin"), (key_d, "dclm", "data/dclm/dclm_val.bin")]:
                r = evaluate(exp, fin, vb)
                if r:
                    record(key, stage="adapt_after", model=exp, corpus=corpus, set=tag, ce=r["ce"], tokens=r["tokens"], ckpt=fin)
                    log(f"{key}: ce={r['ce']:.4f}")


def stage_c() -> None:
    from nestor.continual.extension import LR_ADAMW, LR_MUON, ROPE_BASE, extension_flop_ratio
    base_steps = 48
    for S in (8192, 16384):
        val = os.path.abspath(f"data/longctx/dclm_longdoc_ge{S}_val.bin")
        corpus = os.path.abspath(f"data/longctx/dclm_longdoc_ge{S}_train.bin")
        for exp, setting in [("dense_100m", "equal_tokens"), ("moe_100m", "equal_tokens"),
                             ("nestor_100m", "equal_tokens"), ("nestor_100m", "equal_flops_new")]:
            key = f"C|{exp}|{S}|{setting}"
            ck = pretrained(exp)
            if ck is None or key in done_keys():
                continue
            r_s = extension_flop_ratio(exp, S) if setting != "equal_tokens" else 1.0
            steps = math.ceil(base_steps / r_s)
            env = {"DCLM_BLOCK_SIZE": str(S), "DCLM_ROPE_BASE": str(ROPE_BASE[S]), "DCLM_DATA_PATH": corpus,
                   "DCLM_INIT_CHECKPOINT": ck, "DCLM_LR_MUON": f"{LR_MUON * r_s:.6g}", "DCLM_LR_ADAMW": f"{LR_ADAMW * r_s:.6g}",
                   "DCLM_RUN_SUFFIX": f"ext{S}_{setting}", "NESTOR_GATHERED_CAP_BUCKET": str(S // 32),
                   "_TOKENS_PER_STEP": 1_048_576}
            fin = train(f"{exp}_ext{S}_{setting}", exp, env, steps=steps, warmup=int(steps * 0.1), seq=S)
            if not fin:
                continue
            r = evaluate(exp, fin, val, block=S, rope=ROPE_BASE[S], first_per_doc=True, docs=200)
            if r:
                record(key, stage="extension", model=exp, S=S, setting=setting, r_S=r_s, steps=steps,
                       ce=r["ce"], tokens=r["tokens"], windows=r["windows"], ckpt=fin)
                log(f"{key}: ce={r['ce']:.4f} (steps={steps}, r={r_s:.4f})")


if __name__ == "__main__":
    stages = sys.argv[1:] or ["A", "B", "C"]
    for s in stages:
        log(f"=== stage {s} ===")
        {"A": stage_a, "B": stage_b, "C": stage_c}[s]()
    log("ALL DONE")
