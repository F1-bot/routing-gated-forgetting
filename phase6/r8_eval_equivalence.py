"""R8 (review v3, recommendation 8): equivalence of the amended eval_windows.py with the frozen G0 copy
on one theta_0 evaluation, at the prereg's 1e-6 tolerance.

Runs `eval_windows.py run` twice, each in its own process, on the same model, windows and device:
  frozen  <root>/frozen_g0/eval_windows.py  PYTHONPATH = <root>/frozen_g0;<root>;<root>/repo, so the frozen copies in
          frozen_g0 (eval_windows.py and the weight_controls.py it imports) shadow the installed root files; the
          authors' repo and the data resolve from the root.
  current <root>/eval_windows.py            the installed (amended) copy.
Model: MoE base 1 (experiment moe_100m, theta_0 = ckpt_pins.json "moe_b1", size and sha256 checked).
Windows: the registered DCLM evaluation set, data/dclm/dclm_val.bin, 2,048,000 tokens = 500 windows of 4096,
select even (g2/plan_g2.json sets.moe_b1.dclm). Precision fp32 (eval_windows default).
Variants compared: ckpt, pinned, reflect (theta_0 - delta). The current run also requests selpin, reflect+pinned and
reflect+selpin, so its pinned variant takes the joint path used by the post-freeze evaluations
(confirm_plan.PIN_VARIANTS); --current-single adds a third run of the current code with the frozen variant list.

Criterion (tolerance 1e-6 nats/token, the stage-A spot-check tolerance of prereg v1.3 sec. 8 item 1): for every
compared variant the window grid is identical (index, starts, 64-bit content hash, n_tok), every per-window mean CE
differs by at most 1e-6, and the mean CE over all windows differs by at most 1e-6. Bitwise-equal windows are counted.

At theta_0 (checkpoint = base = theta_0, the review's "theta_0 evaluation") delta = 0: pinned replays theta_0's own
routing and reflect equals theta_0, so the three variants coincide by construction and the test compares the code
paths (window grid, CE accounting, pin record and replay, state build and load). --adapted <checkpoint> evaluates an
adapted MoE base-1 checkpoint against theta_0 instead (delta != 0): pinned and reflect then differ from ckpt and the
delta-dependent arithmetic is compared as well.

Writes only under --out: frozen/, current/ (and current_single) with the eval_windows outputs and eval.log, plus
r8_result.json. Never deletes; refuses a non-empty run directory. Exit 0 = PASS, 1 = FAIL, 2 = setup error.

  python r8_eval_equivalence.py --root . --out <dir> [--device cuda] [--adapted <ckpt.pt>]
         [--tokens 2048000] [--current-single] [--strip-child-path <conda Library\bin>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

FROZEN_SHA = {"eval_windows.py": "4aa2f2520dcd6f12a3fe2fb483dd9fdc9c34440d31f6db704a00c3a6b4efdfcc",
              "weight_controls.py": "03bfcab6c85c5de6c620a0675a93d7870a8070aa27c1f85150c6592d61227625"}
CURRENT_SHA = {"eval_windows.py": "0f909d1e23c2e1bd3ec3765778ddf927ad2e3ccc036a6c1eadce8e822f2ecb41",
               "weight_controls.py": "473ff8c58542f7f398431181cf11afb14b0b8bf0f261591a88120f3fdc6b2835"}
EXPERIMENT, BASE_KEY = "moe_100m", "moe_b1"
DCLM_REL, DCLM_TOKENS, SELECT, BLOCK = os.path.join("data", "dclm", "dclm_val.bin"), 2_048_000, "even", 4096
TOL = 1e-6
FROZEN_VARIANTS = "ckpt,pinned,reflect"
CURRENT_VARIANTS = "ckpt,pinned,reflect,selpin,reflect+pinned,reflect+selpin"
COMPARED = ("ckpt", "pinned", "reflect")
GRID_KEYS = ("index", "starts", "hash", "n_tok")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a: dict, b: dict, tol: float = TOL) -> dict:
    """Per-window comparison of two eval_windows results (b - a)."""
    import numpy as np

    grid = {k: bool(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))) for k in GRID_KEYS}
    out = {"windows": [int(len(a["ce_sum"])), int(len(b["ce_sum"]))], "grid_equal": grid}
    if not all(grid.values()):
        out.update(ok=False, reason="window grid differs")
        return out
    na, nb = np.asarray(a["n_tok"], dtype=np.float64), np.asarray(b["n_tok"], dtype=np.float64)
    ca, cb = np.asarray(a["ce_sum"], dtype=np.float64), np.asarray(b["ce_sum"], dtype=np.float64)
    wa, wb = ca / na, cb / nb
    d = wb - wa
    ma, mb = float(ca.sum() / na.sum()), float(cb.sum() / nb.sum())
    i = int(np.argmax(np.abs(d)))
    out.update(max_abs_window_diff=float(abs(d[i])), worst_window=int(np.asarray(a["index"])[i]),
               max_rel_window_diff=float(np.max(np.abs(d) / np.abs(wa))), mean_a=ma, mean_b=mb,
               mean_diff=mb - ma, bitwise_equal_windows=int(np.sum(ca == cb)), tol=tol)
    out["ok"] = bool(out["max_abs_window_diff"] <= tol and abs(out["mean_diff"]) <= tol)
    if not out["ok"]:
        out["reason"] = "difference above tolerance"
    return out


def load(path: str) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.pop("meta_json"))) if "meta_json" in d else {}
    return d


def child_env(pythonpath: list[str], strip: list[str], root: str) -> dict:
    env = dict(os.environ)
    old = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath + [p for p in old if p not in pythonpath])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(root, "data", "tiktoken"))
    if strip:
        norm = {os.path.normcase(os.path.normpath(s)) for s in strip}
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                      if p and os.path.normcase(os.path.normpath(p)) not in norm)
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default=None, help="eval_windows --device (default: its own choice, cuda if available)")
    ap.add_argument("--adapted", default=None, help="adapted MoE base-1 checkpoint (evaluated against theta_0)")
    ap.add_argument("--tokens", type=int, default=DCLM_TOKENS, help="DCLM tokens (smoke: 8192 = 2 windows)")
    ap.add_argument("--current-single", action="store_true", help="also run the current code with the frozen variants")
    ap.add_argument("--strip-child-path", action="append", default=[],
                    help="PATH entry removed for the eval processes (the conda Library\\bin prefix of torch-free steps)")
    ap.add_argument("--timeout-h", type=float, default=6.0)
    a = ap.parse_args(argv)

    root, out = os.path.abspath(a.root), os.path.abspath(a.out)
    frozen_dir = os.path.join(root, "frozen_g0")
    if os.path.normcase(out) == os.path.normcase(root) or os.path.normcase(out).startswith(os.path.normcase(frozen_dir)):
        print("R8 setup error: --out must not be the root or inside frozen_g0", flush=True)
        return 2
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "root": root, "out": out, "tolerance": TOL,
           "criterion": "identical window grid; |CE_current - CE_frozen| <= 1e-6 nats/token per window and for the mean",
           "tolerance_source": "prereg v1.3 sec. 8 item 1 (stage-A spot check 4.628702 +- 1e-6)", "hashes": {}}

    # ---- code identity: the frozen copies are the G0 freeze, the root copies are the MF-4 'current' row
    freeze = json.load(open(os.path.join(root, "g0_code_freeze.json"), encoding="utf-8")).get("files", {})
    bad = []
    for f, want in FROZEN_SHA.items():
        got = sha256(os.path.join(frozen_dir, f))
        rec = freeze.get(f)
        res["hashes"][f"frozen_g0/{f}"] = {"sha256": got, "expected": want, "g0_code_freeze": rec}
        if got != want or (rec is not None and rec != got):
            bad.append(f"frozen_g0/{f}")
    for f, want in CURRENT_SHA.items():
        got = sha256(os.path.join(root, f))
        res["hashes"][f] = {"sha256": got, "expected": want}
        if got != want:
            bad.append(f)
    # ---- checkpoints
    pins = json.load(open(os.path.join(root, "ckpt_pins.json"), encoding="utf-8"))["pins"][BASE_KEY]
    theta0 = os.path.join(root, pins["path"])
    if os.path.getsize(theta0) != int(pins["size"]) or sha256(theta0) != pins["sha256"]:
        bad.append(f"theta_0 {theta0} (ckpt_pins.json {BASE_KEY})")
    res["theta0"] = {"path": theta0, "size": int(pins["size"]), "sha256": pins["sha256"]}
    ckpt = theta0
    if a.adapted:
        ckpt = a.adapted if os.path.isabs(a.adapted) else os.path.join(root, a.adapted)
        if not os.path.isfile(ckpt):
            bad.append(f"adapted checkpoint {ckpt}")
        else:
            res["adapted"] = {"path": ckpt, "size": os.path.getsize(ckpt), "sha256": sha256(ckpt)}
    data = os.path.join(root, DCLM_REL)
    if not os.path.isfile(data):
        bad.append(data)
    if bad:
        res.update(status="setup_error", problems=bad)
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, "r8_result.json"), "w", encoding="utf-8"), indent=1)
        print("R8 setup error: " + "; ".join(bad), flush=True)
        return 2

    set_arg = f"dclm={data}:{int(a.tokens)}"
    runs = [("frozen", os.path.join(frozen_dir, "eval_windows.py"), FROZEN_VARIANTS,
             [frozen_dir, root, os.path.join(root, "repo")]),
            ("current", os.path.join(root, "eval_windows.py"), CURRENT_VARIANTS, [root, os.path.join(root, "repo")])]
    if a.current_single:
        runs.append(("current_single", os.path.join(root, "eval_windows.py"), FROZEN_VARIANTS,
                     [root, os.path.join(root, "repo")]))
    res["runs"] = {}
    for tag, script, variants, pp in runs:
        d = os.path.join(out, tag)
        if os.path.isdir(d) and os.listdir(d):
            print(f"R8 setup error: {d} exists and is not empty (choose a new --out)", flush=True)
            return 2
        os.makedirs(d, exist_ok=True)
        cmd = [a.python, script, "run", EXPERIMENT, ckpt, "--set", set_arg, "--select", SELECT, "--block", str(BLOCK),
               "--variants", variants, "--base", theta0, "--out", d] + (["--device", a.device] if a.device else [])
        env = child_env(pp, a.strip_child_path, root)
        t0 = time.time()
        print(f"[R8] {tag}: {' '.join(cmd)}", flush=True)
        with open(os.path.join(d, "eval.log"), "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=a.timeout_h * 3600).returncode
        res["runs"][tag] = {"cmd": cmd, "pythonpath": env["PYTHONPATH"], "rc": rc, "seconds": round(time.time() - t0, 1)}
        print(f"[R8] {tag}: rc {rc} in {res['runs'][tag]['seconds']} s", flush=True)
        if rc != 0:
            res.update(status="setup_error", problems=[f"{tag} eval_windows exited {rc} (see {d}\\eval.log)"])
            json.dump(res, open(os.path.join(out, "r8_result.json"), "w", encoding="utf-8"), indent=1)
            return 2

    # ---- comparison
    res["compare"], ok = {}, True
    fz = {v: load(os.path.join(out, "frozen", f"{v}__dclm.npz")) for v in COMPARED}
    for tag in [t for t, *_ in runs if t != "frozen"]:
        for v in COMPARED:
            c = compare(fz[v], load(os.path.join(out, tag, f"{v}__dclm.npz")))
            res["compare"][f"{tag}:{v}"] = c
            ok = ok and c["ok"]
            print(f"[R8] frozen vs {tag} {v:8s}: grid {all(c['grid_equal'].values())}, max |dCE| "
                  f"{c.get('max_abs_window_diff', float('nan')):.3e}, mean dCE {c.get('mean_diff', float('nan')):.3e}, "
                  f"bitwise-equal windows {c.get('bitwise_equal_windows', 0)}/{c['windows'][0]} -> "
                  f"{'ok' if c['ok'] else 'FAIL'}", flush=True)
    res["meta"] = {v: fz[v]["meta"] for v in COMPARED}
    # ---- information only: theta_0 identities (delta = 0) and the G1 theta_0 evaluation of the same windows
    info = {}
    if not a.adapted:
        for tag in [t for t, *_ in runs]:
            ck = load(os.path.join(out, tag, "ckpt__dclm.npz"))
            for v in ("pinned", "reflect"):
                c = compare(ck, load(os.path.join(out, tag, f"{v}__dclm.npz")))
                info[f"{tag}: {v} vs ckpt at theta_0"] = {k: c.get(k) for k in ("ok", "max_abs_window_diff",
                                                                                "bitwise_equal_windows")}
        g1 = os.path.join(root, "g1", "base", BASE_KEY, "ckpt__dclm.npz")
        if os.path.isfile(g1) and int(a.tokens) == DCLM_TOKENS:
            c = compare(load(g1), load(os.path.join(out, "current", "ckpt__dclm.npz")))
            info["G1 theta_0 evaluation (g1/base/moe_b1/ckpt__dclm.npz) vs current ckpt"] = {
                k: c.get(k) for k in ("ok", "max_abs_window_diff", "mean_diff", "bitwise_equal_windows")}
    res["information"] = info
    res["status"] = "PASS" if ok else "FAIL"
    json.dump(res, open(os.path.join(out, "r8_result.json"), "w", encoding="utf-8"), indent=1)
    print(f"R8 {res['status']} (tolerance {TOL:g} nats/token; {os.path.join(out, 'r8_result.json')})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
