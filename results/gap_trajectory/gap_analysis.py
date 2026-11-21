#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
gap_analysis.py - NESTOR vs MoE vs dense training-loss gap trajectory (CPU only, torch-free).

Inputs  : results\gap_trajectory\inputs\  (copies of the four pretraining logs, the
          stage-A held-out JSON, the two local config.json files and the launcher scripts; sha256 in MANIFEST.json)
Outputs : gap_trajectory.json, training_loss_parsed.csv, gap_series.csv  (same folder as this script's parent)
Rules   : <= 6 CPU worker threads in total (this process: 1 python thread, BLAS/OMP capped at 4), BelowNormal
          priority, TEMP/TMP on the persistent project drive, no GPU, no network.
"""
import os, sys
THREADS = "4"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = THREADS
_LIBBIN = r"<conda>\envs\nestor\Library\bin"   # torch-free process only (numpy BLAS abort 0xC06D007F otherwise)
if os.path.isdir(_LIBBIN):
    os.environ["PATH"] = _LIBBIN + os.pathsep + os.environ.get("PATH", "")
_TMP = r".tmp"
os.makedirs(_TMP, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = _TMP

def _below_normal():
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass.restype = wintypes.BOOL
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000))
    except Exception:
        return False
PRIORITY_BELOW_NORMAL = _below_normal()

import json, re, math, hashlib, time, datetime, platform
import numpy as np

ROOT = r"."
OUT = os.path.join(ROOT, "results", "gap_trajectory")
INP = os.path.join(OUT, "inputs")
REPO = os.path.join(ROOT, "repo")

# ------------------------------------------------------------------ constants
TOKENS_PER_STEP = 262_144
T_MAX, WARMUP, ETA_MIN = 1907, 153, 1e-5            # this reproduction (10 % of the paper's budget)
PAPER_T_MAX, PAPER_WARMUP = 19_080, 1_526            # paper / registered 100m cell (5.0B tokens)
LR_ROUND_HALF = 5e-5                                 # logged lr has 4 decimals
LAST_HALF_MIN_STEP = T_MAX / 2.0                     # logged steps strictly above 953.5 -> 960..1900 (95 points)
END_WINDOW_POINTS = 10                               # steps 1810..1900
BLOCK_LEN = 5                                        # primary moving-block length (logged points = 50 steps)
BLOCK_SENS = (1, 3, 5, 8, 12)
B_MAIN, B_BAND, B_TS = 10_000, 2_000, 2_000
SEED = 20261113
PAPER = {"dense": 3.5779, "moe": 3.3859, "nestor": 3.3562}   # held-out CE at 5B tokens (repo pretrain.py docstring == paper Table 3)

RUNS = {
    "dense":  dict(file="logs/pretrain_dense_100m_b05.log",          label="Dense",          seed=42, platform="RTX 4080 SUPER (local)", experiment="dense_100m"),
    "moe42":  dict(file="logs/pretrain_moe_100m_b05.log",            label="MoE (seed 42)",  seed=42, platform="RTX 4080 SUPER (local)", experiment="moe_100m"),
    "nestor": dict(file="logs/pretrain_nestor_100m_b05_h100.log",    label="NESTOR",         seed=42, platform="H100 (Modal)",           experiment="nestor_100m"),
    "moe43":  dict(file="logs/pretrain_moe_100m_b05s43_h100.log",    label="MoE (seed 43)",  seed=43, platform="H100 (Modal)",           experiment="moe_100m"),
}
RUN_ORDER = ("dense", "moe42", "moe43", "nestor")

# ------------------------------------------------------------------ helpers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

def read_text_auto(p):
    b = open(p, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16"), "utf-16"
    if b[:3] == b"\xef\xbb\xbf":
        return b.decode("utf-8-sig"), "utf-8-sig"
    if b[1:200].count(b"\x00") > 50:
        return b.decode("utf-16-le"), "utf-16-le"
    return b.decode("utf-8", errors="replace"), "utf-8"

STEP_RE = re.compile(r"-\s*INFO\s*-\s*Step (\d+) \|(.*)$")
def parse_log(path):
    text, enc = read_text_auto(path)
    rows = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        d = {"step": int(m.group(1))}
        for part in m.group(2).split("|"):
            if ":" in part:
                k, v = part.split(":", 1)
                try:
                    d[k.strip().replace("train/", "")] = float(v)
                except ValueError:
                    pass
        rows.append(d)
    info = {"encoding": enc}
    m = re.search(r"\[pretrain\].*?\| (\d+) steps x ([\d,]+) tok/step", text)
    if m:
        info["header_steps"], info["header_tokens_per_step"] = int(m.group(1)), int(m.group(2).replace(",", ""))
    m = re.search(r"Model parameters:\s*([\d,]+)", text)
    if m:
        info["model_parameters"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Training batches:\s*(\d+)", text)
    if m:
        info["dataloader_batches"] = int(m.group(1))
    m = re.search(r"Git SHA:\s*([0-9a-f]{40})", text)
    if m:
        info["git_sha"] = m.group(1)
    info["training_complete"] = "Training complete!" in text
    m0 = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - .*Starting training", text, flags=re.M)
    m1 = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - .*Training complete!", text, flags=re.M)
    if m0 and m1:
        fmt = "%Y-%m-%d %H:%M:%S"
        info["wall_clock_h"] = (datetime.datetime.strptime(m1.group(1), fmt) - datetime.datetime.strptime(m0.group(1), fmt)).total_seconds() / 3600.0
    return rows, info

def lr_multiplier(N, base):
    """LR / peak after N scheduler steps (the LR used by update index N), torch SequentialLR[warmup, cosine(T_MAX)]."""
    N = np.asarray(N, dtype=float)
    t = np.clip(N - WARMUP, 0.0, None)
    cosv = ETA_MIN / base + (1 - ETA_MIN / base) * 0.5 * (1 + np.cos(np.pi * t / T_MAX))
    return np.where(N < WARMUP, N / WARMUP, cosv)

def lr_multiplier_generic(N, T, W, base, eta_min=ETA_MIN):
    N = np.asarray(N, dtype=float)
    t = np.clip(N - W, 0.0, None)
    cosv = eta_min / base + (1 - eta_min / base) * 0.5 * (1 + np.cos(np.pi * t / T))
    return np.where(N < W, N / W, cosv)

def loess_matrix(x, k, deg=1):
    """Smoother matrix of LOESS with a nearest-neighbour window: the k logged points nearest to x0 receive tricube
    weights (h = distance of the k-th nearest point), so the window widens one-sidedly at the ends of the run."""
    n = len(x)
    L = np.zeros((n, n))
    e0 = np.zeros(deg + 1); e0[0] = 1.0
    for i, x0 in enumerate(x):
        d = np.abs(x - x0)
        h = np.sort(d)[min(int(k), n) - 1] * 1.0001 + 1e-9
        dd = d / h
        w = np.where(dd < 1, (1 - dd ** 3) ** 3, 0.0)
        idx = np.where(w > 0)[0]
        X = np.vander(x[idx] - x0, deg + 1, increasing=True)
        W = w[idx]
        A = X.T @ (X * W[:, None])
        L[i, idx] = np.linalg.solve(A, e0) @ (X.T * W[None, :])
    return L

def pick_span(cv, tol=0.01):
    """Largest k (smoothest fit) whose LOO-CV error is within tol of the minimum (smoothing-favoured one-SE-like rule)."""
    m = min(cv.values())
    return max(k for k, v in cv.items() if v <= m * (1 + tol))

def loo_cv(L, y):
    return float(np.mean(((y - L @ y) / (1 - np.diag(L))) ** 2))

def eff_block_len(n, ell):
    """Block length actually used: never more than a quarter of the window (matters only for windows of < 20 points)."""
    return max(1, min(int(ell), max(1, n // 4)))

def mbb_idx(n, ell, B, rng):
    """Circular moving-block bootstrap indices (Politis & Romano 1992): every point has equal weight, so the
    bootstrap mean is unbiased (the overlapping non-circular scheme under-weights the window ends)."""
    ell = eff_block_len(n, ell)
    k = math.ceil(n / ell)
    starts = rng.integers(0, n, size=(B, k))
    return ((starts[:, :, None] + np.arange(ell)[None, None, :]) % n).reshape(B, k * ell)[:, :n]

def slope_rows(X, Y):
    xm = X.mean(1, keepdims=True); ym = Y.mean(1, keepdims=True)
    return ((X - xm) * (Y - ym)).sum(1) / ((X - xm) ** 2).sum(1)

def ols(x, y):
    b = np.polyfit(x, y, 1)
    r = y - (b[1] + b[0] * x)
    n = len(x)
    s2 = float(r @ r / (n - 2))
    se = math.sqrt(s2 / float(((x - x.mean()) ** 2).sum()))
    return float(b[0]), float(b[1]), r, se

def pct(v, lo=2.5, hi=97.5):
    a = np.percentile(v, [lo, hi])
    return [float(a[0]), float(a[1])]

def pairs_slope_boot(x, y, ell, B, rng):
    idx = mbb_idx(len(x), ell, B, rng)
    return slope_rows(x[idx], y[idx])

def resid_slope_boot(x, y, ell, B, rng):
    b, a, e, _ = ols(x, y)
    fit = a + b * x
    idx = mbb_idx(len(x), ell, B, rng)
    Y = fit[None, :] + e[idx]
    return slope_rows(np.broadcast_to(x, Y.shape), Y)

def theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median(((y[j] - y[i])[ok]) / dx[ok]))

def theil_sen_boot(x, y, ell, B, rng, chunk=250):
    n = len(x)
    i, j = np.triu_indices(n, 1)
    out = []
    for c0 in range(0, B, chunk):
        idx = mbb_idx(n, ell, min(chunk, B - c0), rng)
        xs, ys = x[idx], y[idx]
        dx = xs[:, j] - xs[:, i]
        dy = ys[:, j] - ys[:, i]
        s = np.where(dx != 0, dy / np.where(dx != 0, dx, 1.0), np.nan)
        out.append(np.nanmedian(s, axis=1))
    return np.concatenate(out)

def mean_boot_ci(y, ell, B, rng):
    idx = mbb_idx(len(y), ell, B, rng)
    return pct(y[idx].mean(1))

def acf(r, lags=(1, 2, 3)):
    r = r - r.mean()
    return [float((r[:-l] * r[l:]).sum() / (r * r).sum()) for l in lags]

def smooth_with_band(L, y, ell, B, rng, level=95.0):
    yhat = L @ y
    e = y - yhat
    df = float(np.trace(L))
    e = e * math.sqrt(len(y) / max(len(y) - df, 1.0))
    idx = mbb_idx(len(y), ell, B, rng)
    ys = yhat[None, :] + e[idx]
    sm = ys @ L.T
    a = (100 - level) / 2
    lo, hi = np.percentile(sm, [a, 100 - a], axis=0)
    return yhat, lo, hi, df

def f(v):
    return float(v)

# ------------------------------------------------------------------ self tests
def selftest():
    rng = np.random.default_rng(0)
    x = np.arange(10, 200, 10.0)
    y = np.sin(x / 40) + 0.1 * rng.normal(size=x.size)
    for deg in (1, 2):
        k = 9
        L = loess_matrix(x, k, deg)
        for i in (0, 7, len(x) - 1):
            d = np.abs(x - x[i])
            h = np.sort(d)[k - 1] * 1.0001 + 1e-9
            dd = d / h
            w = np.where(dd < 1, (1 - dd ** 3) ** 3, 0.0)
            sw = w > 0
            X = np.vander(x[sw] - x[i], deg + 1, increasing=True)
            beta = np.linalg.solve(X.T @ (X * w[sw][:, None]), X.T @ (w[sw] * y[sw]))
            assert abs(beta[0] - (L @ y)[i]) < 1e-9, (deg, i)
        assert np.allclose(L.sum(1), 1.0)
        lin = 2.0 + 0.3 * x
        assert np.allclose(L @ lin, lin, atol=1e-8)          # local-linear/quadratic reproduce a line exactly
    assert float(lr_multiplier(0, 0.02)) == 0.0 and abs(float(lr_multiplier(WARMUP, 0.02)) - 1.0) < 1e-12
    xx = np.arange(20.0); yy = 3 * xx + 1
    assert abs(float(slope_rows(xx[None, :], yy[None, :])[0]) - 3) < 1e-12
    assert abs(theil_sen(xx, yy) - 3) < 1e-12
    rng = np.random.default_rng(1)
    idx = mbb_idx(95, 5, 1000, rng)
    assert idx.shape == (1000, 95) and idx.min() >= 0 and idx.max() < 95
    # circular bootstrap: every point has equal expected weight
    w = np.bincount(mbb_idx(95, 5, 20000, np.random.default_rng(2)).ravel(), minlength=95) / (20000 * 95)
    assert abs(w.max() - 1 / 95) < 0.0015 and abs(w.min() - 1 / 95) < 0.0015
    return True

# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    selftest()
    manifest = json.load(open(os.path.join(INP, "MANIFEST.json"), encoding="utf-8"))
    manifest_check = {}
    for rel, meta in manifest.items():
        p = os.path.join(INP, rel.replace("/", os.sep))
        manifest_check[rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p), "original_path": meta["original_path"]}
        assert manifest_check[rel]["sha256"] == meta["sha256"], "input changed since staging: " + rel

    # ---- 1. parse ----
    parsed, rinfo = {}, {}
    for r in RUN_ORDER:
        rows, info = parse_log(os.path.join(INP, RUNS[r]["file"].replace("/", os.sep)))
        parsed[r], rinfo[r] = rows, info
    steps = np.array([d["step"] for d in parsed["dense"]])
    for r in RUN_ORDER:
        s = np.array([d["step"] for d in parsed[r]])
        assert np.array_equal(s, steps), f"step grid differs for {r}"
    assert len(steps) == 190 and steps[0] == 10 and steps[-1] == 1900 and set(np.diff(steps)) == {10}
    loss = {r: np.array([d["loss"] for d in parsed[r]]) for r in RUN_ORDER}
    ce_equal = {r: bool(np.array_equal(loss[r], np.array([d["cross_entropy"] for d in parsed[r]]))) for r in RUN_ORDER}
    lr_max = {r: np.array([d["lr_max"] for d in parsed[r]]) for r in RUN_ORDER}
    lr_g0 = {r: np.array([d["lr"] for d in parsed[r]]) for r in RUN_ORDER}
    assert all(ce_equal.values()) and all(np.isfinite(loss[r]).all() for r in RUN_ORDER)
    tok = steps * float(TOKENS_PER_STEP)
    tokB = tok / 1e9
    for r in RUN_ORDER:
        assert rinfo[r].get("header_steps") == T_MAX and rinfo[r].get("header_tokens_per_step") == TOKENS_PER_STEP, r
        assert rinfo[r]["training_complete"], r

    # ---- 2. learning-rate schedule check against the logged lr values ----
    sched_check = {}
    for r in RUN_ORDER:
        out = {}
        for key, arr in (("lr_max", lr_max[r]), ("lr_group0", lr_g0[r])):
            peak = 0.01
            for _ in range(30):
                m = lr_multiplier(steps, peak)
                peak = float(np.sum(m * arr) / np.sum(m * m))
            resid = arr - peak * lr_multiplier(steps, peak)
            # alternative indexing (off by one scheduler step) as a falsification check
            alt = peak * lr_multiplier(steps + 1, peak)
            out[key] = {"fitted_peak": peak, "max_abs_residual": float(np.abs(resid).max()),
                        "n_outside_rounding": int((np.abs(resid) > LR_ROUND_HALF * 1.05).sum()),
                        "max_abs_residual_if_index_off_by_one": float(np.abs(arr - alt).max())}
        sched_check[r] = out

    # ---- 3. smoothing parameters (leave-one-out CV on the logged grid) ----
    x = steps.astype(float)
    def gap_series():
        g = {
            "N_minus_M42": loss["nestor"] - loss["moe42"],
            "N_minus_M43": loss["nestor"] - loss["moe43"],
            "N_minus_Mavg": loss["nestor"] - 0.5 * (loss["moe42"] + loss["moe43"]),
            "N_minus_D": loss["nestor"] - loss["dense"],
            "M43_minus_M42": loss["moe43"] - loss["moe42"],
            "D_minus_M42": loss["dense"] - loss["moe42"],
        }
        return g
    gaps = gap_series()
    gap_desc = {
        "N_minus_M42": "NESTOR (seed 42, H100) - MoE (seed 42, local RTX 4080 SUPER)",
        "N_minus_M43": "NESTOR (seed 42, H100) - MoE (seed 43, H100)",
        "N_minus_Mavg": "NESTOR - mean of the two MoE seeds (pooled comparator; extra)",
        "N_minus_D": "NESTOR (seed 42, H100) - dense (seed 42, local RTX 4080 SUPER)",
        "M43_minus_M42": "MoE seed 43 (H100) - MoE seed 42 (local): seed(+platform) noise reference",
        "D_minus_M42": "dense - MoE (seed 42): known-ordering validation row (extra)",
    }
    main_gap_keys = ("N_minus_M42", "N_minus_M43", "N_minus_D", "M43_minus_M42")
    k_grid_gap = [11, 15, 21, 25, 31, 41, 51, 61]
    k_grid_curve = [15, 21, 25, 31, 41, 51, 61]
    cv_gap = {k_: float(np.mean([loo_cv(loess_matrix(x, k_, 1), gaps[kk]) for kk in main_gap_keys])) for k_ in k_grid_gap}
    cv_curve = {k_: float(np.mean([loo_cv(loess_matrix(x, k_, 2), loss[r]) for r in RUN_ORDER])) for k_ in k_grid_curve}
    K_GAP, K_CURVE = pick_span(cv_gap), pick_span(cv_curve)
    K_GAP_CVMIN, K_CURVE_CVMIN = min(cv_gap, key=cv_gap.get), min(cv_curve, key=cv_curve.get)
    L_gap = loess_matrix(x, K_GAP, 1)
    L_curve = loess_matrix(x, K_CURVE, 2)

    # ---- 4. held-out references ----
    sa = json.load(open(os.path.join(INP, "reproduction_stageA.json"), encoding="utf-8"))
    loc, h1 = sa["local_rtx4080super"], sa["h100_modal"]
    ho = {}
    for v in ("val09", "val10"):
        ho[v] = {"dense": loc[f"dense_100m|{v}"]["ce"], "moe42": loc[f"moe_100m|{v}"]["ce"],
                 "moe43": h1[f"moe_100m_s43|{v}"]["ce"], "nestor": loc[f"nestor_100m|{v}"]["ce"],
                 "nestor_h100_eval": h1[f"nestor_100m|{v}"]["ce"]}
    def ho_gap(key, v):
        h = ho[v]
        return {"N_minus_M42": h["nestor"] - h["moe42"], "N_minus_M43": h["nestor"] - h["moe43"],
                "N_minus_Mavg": h["nestor"] - 0.5 * (h["moe42"] + h["moe43"]), "N_minus_D": h["nestor"] - h["dense"],
                "M43_minus_M42": h["moe43"] - h["moe42"], "D_minus_M42": h["dense"] - h["moe42"]}[key]

    # ---- 5. per-gap analysis ----
    lh = steps > LAST_HALF_MIN_STEP
    xl = tokB[lh]
    SPAN = 0.25   # nominal length of the last half in B tokens
    gap_out = {}
    for gi, (key, g) in enumerate(gaps.items()):
        rng = np.random.default_rng([SEED, gi, 1])
        yl = g[lh]
        b, a, resid, se = ols(xl, yl)
        sgn = 1.0 if yl.mean() >= 0 else -1.0
        bp = pairs_slope_boot(xl, yl, BLOCK_LEN, B_MAIN, rng)
        br = resid_slope_boot(xl, yl, BLOCK_LEN, B_MAIN, np.random.default_rng([SEED, gi, 2]))
        ci_p, ci_r = pct(bp), pct(br)
        ts = theil_sen(xl, yl)
        bts = theil_sen_boot(xl, yl, BLOCK_LEN, B_TS, np.random.default_rng([SEED, gi, 3]))
        ci_ts = pct(bts)
        # verdict from the primary (pairs MBB) CI, oriented so that negative = |gap| shrinking
        o_lo, o_hi = sorted([sgn * ci_p[0], sgn * ci_p[1]])
        verdict = "narrows" if o_hi < 0 else ("widens" if o_lo > 0 else "flat")
        # sensitivity: block length
        sens_len = {str(l): pct(pairs_slope_boot(xl, yl, l, 4000, np.random.default_rng([SEED, gi, 10 + l]))) for l in BLOCK_SENS}
        # sensitivity: drop last logged point; window start
        m2 = lh & (steps <= 1890)
        b2 = float(np.polyfit(tokB[m2], g[m2], 1)[0])
        ci2 = pct(pairs_slope_boot(tokB[m2], g[m2], BLOCK_LEN, 4000, np.random.default_rng([SEED, gi, 4])))
        win = {}
        for frac in (0.4, 0.6):
            mm = steps >= frac * T_MAX
            win[str(frac)] = {"n": int(mm.sum()), "first_step": int(steps[mm][0]),
                              "slope": float(np.polyfit(tokB[mm], g[mm], 1)[0]),
                              "slope_ci95": pct(pairs_slope_boot(tokB[mm], g[mm], BLOCK_LEN, 4000, np.random.default_rng([SEED, gi, 5 + int(frac * 10)])))}
        # quarter difference Q4 - Q3
        q3 = (steps >= 960) & (steps <= 1430); q4 = (steps >= 1440) & (steps <= 1900)
        r_q = np.random.default_rng([SEED, gi, 6])
        i3 = mbb_idx(int(q3.sum()), BLOCK_LEN, B_MAIN, r_q); i4 = mbb_idx(int(q4.sum()), BLOCK_LEN, B_MAIN, r_q)
        dq = g[q4][i4].mean(1) - g[q3][i3].mean(1)
        # end window (last 10 logged points)
        ew = steps >= steps[-END_WINDOW_POINTS]
        sm, lo, hi, df = smooth_with_band(L_gap, g, BLOCK_LEN, B_BAND, np.random.default_rng([SEED, gi, 7]))
        res_all = g - sm
        z = res_all / (1.4826 * np.median(np.abs(res_all - np.median(res_all))))
        gap_out[key] = {
            "definition": gap_desc[key],
            "sign_convention": "first minus second; positive = first run has the higher loss",
            "last_half": {
                "steps": [int(steps[lh][0]), int(steps[lh][-1])], "n_points": int(lh.sum()),
                "tokens_B": [f(xl[0]), f(xl[-1])],
                "mean": f(yl.mean()), "mean_ci95_block_bootstrap": mean_boot_ci(yl, BLOCK_LEN, B_MAIN, np.random.default_rng([SEED, gi, 8])),
                "sd_per_point": f(yl.std(ddof=1)),
                "slope_nats_per_Btok": b, "slope_ci95_pairs_block_bootstrap": ci_p,
                "slope_ci95_residual_block_bootstrap": ci_r,
                "slope_theil_sen": ts, "slope_theil_sen_ci95_block_bootstrap": ci_ts,
                "slope_se_iid_ols": se, "residual_acf_lag1_2_3": acf(resid),
                "fraction_bootstrap_slopes_negative": float((bp < 0).mean()),
                "change_over_last_half_nats": b * SPAN,
                "change_over_last_half_ci95": [ci_p[0] * SPAN, ci_p[1] * SPAN],
                "change_over_last_half_relative_to_mean": b * SPAN / float(yl.mean()) if abs(yl.mean()) > 1e-12 else None,
                "largest_|change|_not_excluded_nats": max(abs(ci_p[0]), abs(ci_p[1])) * SPAN,
                "quarter_difference": {"Q3_steps": [960, 1430], "Q4_steps": [1440, 1900],
                                       "Q3_mean": f(g[q3].mean()), "Q4_mean": f(g[q4].mean()),
                                       "diff_Q4_minus_Q3": f(g[q4].mean() - g[q3].mean()), "diff_ci95": pct(dq)},
                "verdict": verdict,
                "verdict_rule": "primary CI = pairs moving-block bootstrap (block 5 logged points = 50 steps, B=10000) of the OLS slope of the gap on tokens; CI wholly below 0 (for a positive gap) = narrows, wholly above 0 = widens, else flat (no change detectable at 95%).",
            },
            "robustness": {
                "block_length_sensitivity_slope_ci95": sens_len,
                "drop_last_logged_point": {"slope": b2, "slope_ci95": ci2},
                "window_start_fraction_of_run": win,
            },
            "end_window": {"steps": [int(steps[ew][0]), int(steps[ew][-1])], "n_points": int(ew.sum()),
                           "mean": f(g[ew].mean()), "mean_ci95_block_bootstrap": mean_boot_ci(g[ew], 3, B_MAIN, np.random.default_rng([SEED, gi, 9]))},
            "heldout_final": {"val09": ho_gap(key, "val09"), "val10": ho_gap(key, "val10")},
            "smoother": {"method": "local-linear LOESS (tricube), k nearest logged points", "k_neighbors": K_GAP, "interior_half_width_steps": (K_GAP - 1) / 2 * 10, "df": df,
                         "band": "95% pointwise, residual moving-block bootstrap (block 5, B=2000), smoothing bias ignored"},
            "series": {"raw": [f(v) for v in g], "smooth": [f(v) for v in sm], "lo95": [f(v) for v in lo], "hi95": [f(v) for v in hi]},
            "outliers_robust_z_gt_3p5": [{"step": int(s_), "z": f(z_)} for s_, z_ in zip(steps, z) if abs(z_) > 3.5],
        }

    # ---- 6. phases of the gap (descriptive), NESTOR vs MoE and dense ----
    phases_def = [("warmup (10-150)", 10, 150), ("post-warmup, pre-transition (160-250)", 160, 250),
                  ("transition (260-340)", 260, 340), ("early stable (350-950)", 350, 950), ("last half (960-1900)", 960, 1900)]
    phases = {}
    for key in ("N_minus_M42", "N_minus_M43", "N_minus_Mavg", "N_minus_D", "M43_minus_M42"):
        g = gaps[key]; rows = []
        for pi, (nm, s0, s1) in enumerate(phases_def):
            mm = (steps >= s0) & (steps <= s1)
            rows.append({"phase": nm, "steps": [s0, s1], "tokens_B": [s0 * TOKENS_PER_STEP / 1e9, s1 * TOKENS_PER_STEP / 1e9], "n": int(mm.sum()),
                         "lr_over_peak_range": [f(lr_multiplier(s1, 0.0283)), f(lr_multiplier(s0, 0.0283))] if s0 >= WARMUP else None,
                         "mean": f(g[mm].mean()), "mean_ci95": mean_boot_ci(g[mm], min(BLOCK_LEN, int(mm.sum())), 4000, np.random.default_rng([SEED, 99, pi]))})
        phases[key] = rows

    # ---- 7. smoothed loss curves (+ per-run summaries) ----
    curves, run_out = {}, {}
    for r in RUN_ORDER:
        sm = L_curve @ loss[r]
        curves[r] = {"raw": [f(v) for v in loss[r]], "smooth": [f(v) for v in sm]}
        ew = steps >= steps[-END_WINDOW_POINTS]
        first10_lh = lh & (steps <= 1050)
        res = loss[r] - sm
        run_out[r] = {
            "label": RUNS[r]["label"], "seed": RUNS[r]["seed"], "platform": RUNS[r]["platform"], "experiment": RUNS[r]["experiment"],
            "log_file": manifest_check[RUNS[r]["file"]]["original_path"], "log_sha256": manifest_check[RUNS[r]["file"]]["sha256"],
            "log_encoding": rinfo[r]["encoding"], "model_parameters": rinfo[r].get("model_parameters"),
            "dataloader_batches_of_micro_size": rinfo[r].get("dataloader_batches"), "git_sha_in_log": rinfo[r].get("git_sha"),
            "wall_clock_h_start_to_complete": rinfo[r].get("wall_clock_h"),
            "n_logged_points": int(len(steps)), "logged_steps": [int(steps[0]), int(steps[-1])], "unlogged_final_steps": T_MAX - int(steps[-1]),
            "loss_mean_last_half": f(loss[r][lh].mean()), "loss_mean_last_10_points": f(loss[r][ew].mean()),
            "loss_mean_first_10_points_of_last_half": f(loss[r][first10_lh].mean()),
            "drop_within_last_half_nats": f(loss[r][first10_lh].mean() - loss[r][ew].mean()),
            "residual_sd_about_smooth_steps_ge_300": f(res[steps >= 300].std(ddof=1)),
            "heldout_ce": {"val09": ho["val09"][r], "val10": ho["val10"][r]},
            "peak_lr_max_fitted": sched_check[r]["lr_max"]["fitted_peak"], "peak_lr_group0_fitted": sched_check[r]["lr_group0"]["fitted_peak"],
        }
    # residual correlation between runs (is the data order shared across architectures?)
    mask = steps >= 300
    resid = {r: (loss[r] - loess_matrix(x, 13, 2) @ loss[r])[mask] for r in RUN_ORDER}
    names = list(RUN_ORDER)
    rc = {}
    for i in range(4):
        for j in range(i + 1, 4):
            rc[f"{names[i]}~{names[j]}"] = float(np.corrcoef(resid[names[i]], resid[names[j]])[0, 1])
    n_res = int(mask.sum())

    # ---- 8. schedule description ----
    def mult5(tokens_B):
        return float(lr_multiplier_generic(tokens_B * 1e9 / TOKENS_PER_STEP, PAPER_T_MAX, PAPER_WARMUP, 0.0283))
    def mult1(tokens_B):
        return float(lr_multiplier_generic(tokens_B * 1e9 / TOKENS_PER_STEP, T_MAX, WARMUP, 0.0283))
    eq_tokens = [0.02, 0.05, 0.0675, 0.09, 0.10, 0.20, 0.25, 0.30, 0.40, 0.45, 0.5]
    equal_token_table = [{"tokens_B": t_, "step": t_ * 1e9 / TOKENS_PER_STEP, "lr_over_peak_this_run_0p5B_schedule": mult1(t_),
                          "lr_over_peak_5B_schedule_same_tokens": mult5(t_)} for t_ in eq_tokens]
    progress_table = [{"progress": p, "step_0p5B": p * T_MAX, "tokens_0p5B_B": p * T_MAX * TOKENS_PER_STEP / 1e9,
                       "step_5B": p * PAPER_T_MAX, "tokens_5B_B": p * PAPER_T_MAX * TOKENS_PER_STEP / 1e9,
                       "lr_over_peak_0p5B": float(lr_multiplier_generic(p * T_MAX, T_MAX, WARMUP, 0.0283)),
                       "lr_over_peak_5B": float(lr_multiplier_generic(p * PAPER_T_MAX, PAPER_T_MAX, PAPER_WARMUP, 0.0283))}
                      for p in (0.04, 0.08, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00)]
    repo_files = {}
    for rel in (r"nestor\configs\sched.py", r"nestor\train\optim.py", r"nestor\experiments\pretrain.py", r"nestor\train\trainer.py"):
        p = os.path.join(REPO, rel)
        repo_files[rel.replace("\\", "/")] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)} if os.path.exists(p) else None
    cfgs = {}
    for r, rel in (("dense", "configs/dense_100m_b05_config.json"), ("moe42", "configs/moe_100m_b05_config.json")):
        c = json.load(open(os.path.join(INP, rel), encoding="utf-8"))["job"]
        cfgs[r] = {"scheduler": c["scheduler"], "optimizer": {k: c["optimizer"][k] for k in ("lr_muon", "lr_adamw", "weight_decay", "betas")},
                   "max_steps": c["max_steps"], "grad_accum_steps": c["grad_accum_steps"], "micro_batch": c["train_dataloader"]["batch_size"],
                   "log_frequency": c["log_frequency"], "seed": c["seed"], "max_grad_norm": c["max_grad_norm"]}
    schedule = {
        "type": "cosine annealing with linear warmup: torch SequentialLR([LambdaLR(step/warmup), CosineAnnealingLR(T_max, eta_min)], milestones=[warmup]); all param groups share the multiplier",
        "source": "repo nestor/experiments/pretrain.py (CosineAnnealingConfig(T_max=max_steps, warmup_steps=0.08*max_steps, eta_min=1e-5)); nestor/train/optim.py build_scheduler/_create_warmup_scheduler; paper Table 12 (warmup linear 8 % of steps, cosine, minimum 1e-5)",
        "this_reproduction": {"T_max": T_MAX, "warmup_steps": WARMUP, "eta_min": ETA_MIN, "warmup_fraction": WARMUP / T_MAX, "tokens_per_step": TOKENS_PER_STEP,
                              "config_json_local_runs": cfgs,
                              "h100_runs": "config.json not stored locally; launcher scripts modal/prod.sh and modal/moe2.sh pass --steps 1907 --warmup 153 (copies in inputs/launchers); logged lr identical in shape (see verification)"},
        "paper_5B_run": {"T_max": PAPER_T_MAX, "warmup_steps": PAPER_WARMUP, "eta_min": ETA_MIN, "warmup_fraction": PAPER_WARMUP / PAPER_T_MAX, "tokens_per_step": TOKENS_PER_STEP,
                         "warmup_tokens_B": PAPER_WARMUP * TOKENS_PER_STEP / 1e9},
        "formula": "m(N) = N/W for N < W; m(N) = [eta_min + (peak - eta_min) * (1 + cos(pi (N - W) / T_max)) / 2] / peak for N >= W, where N = number of optimizer steps already taken (the lr used by update N+1)",
        "cosine_phase_uses_T_max_not_T_max_minus_warmup": True,
        "cosine_progress_reached_at_end": (T_MAX - WARMUP) / T_MAX,
        "lr_over_peak_at_logged_step_1900": float(lr_multiplier(1900, 0.0283)), "lr_over_peak_after_final_step_1907": float(lr_multiplier(1907, 0.0283)),
        "reaches_eta_min": False,
        "lr_over_peak_at_start_of_last_half_step_960": float(lr_multiplier(960, 0.0283)),
        "same_function_of_progress_as_paper_5B_run": True,
        "verification_against_logged_lr": sched_check,
        "peak_lr_by_architecture": {r: {"largest_group_lr_max": sched_check[r]["lr_max"]["fitted_peak"], "smallest_muon_group_lr": sched_check[r]["lr_group0"]["fitted_peak"]} for r in RUN_ORDER},
        "equal_progress_table": progress_table, "equal_token_table": equal_token_table,
        "repo_files_sha256": repo_files,
    }

    # ---- 9. paper comparison ----
    pd_, pm_, pn_ = PAPER["dense"], PAPER["moe"], PAPER["nestor"]
    paper_gaps = {"N_minus_M": pn_ - pm_, "N_minus_D": pn_ - pd_, "M_minus_D": pm_ - pd_}
    ours09 = {k: ho_gap(k, "val09") for k in gaps}
    ours10 = {k: ho_gap(k, "val10") for k in gaps}
    swing = {
        "N_minus_M_needed_change_vs_seed42_MoE_val09": paper_gaps["N_minus_M"] - ours09["N_minus_M42"],
        "N_minus_M_needed_change_vs_seed43_MoE_val09": paper_gaps["N_minus_M"] - ours09["N_minus_M43"],
        "N_minus_D_needed_change_val09": paper_gaps["N_minus_D"] - ours09["N_minus_D"],
        "MoE_minus_dense_needed_change_val09": paper_gaps["M_minus_D"] - (-ours09["D_minus_M42"]),
    }
    paper_cmp = {
        "paper_heldout_5B": PAPER, "paper_gaps": paper_gaps,
        "ours_heldout_0p5B_val09": ours09, "ours_heldout_0p5B_val10": ours10,
        "change_needed_between_0p5B_and_5B": swing,
        "ratio_needed_change_N_minus_M_to_MoE_minus_dense": swing["N_minus_M_needed_change_vs_seed42_MoE_val09"] / swing["MoE_minus_dense_needed_change_val09"],
        "ordering_0p5B_val09": sorted(("dense", "moe42", "moe43", "nestor"), key=lambda k_: ho["val09"][k_]),
        "ordering_paper_5B": ["nestor", "moe", "dense"],
        "paper_two_seed_heldout_pairs_200M_10B_tokens_Table2": {"NESTOR_ramp_up": [3.092, 3.088], "dense_middle_schedule": [3.088, 3.087]},
        "note": "held-out sets differ (paper: its own DCLM split; here: global shard 09 file 0 first 10.0M tokens = val09, shard 10 first 5.0M tokens = val10); only differences between architectures on one evaluation set are compared.",
    }

    # ---- 10. seed-noise reference summary ----
    sref = gap_out["M43_minus_M42"]
    interior = (steps >= 100) & (steps <= 1850)
    seed_ref = {
        "definition": gap_desc["M43_minus_M42"],
        "n_seed_pairs": 1,
        "caveat": "a single seed pair (1 degree of freedom) whose difference also contains the platform difference (seed 43 on H100, seed 42 local); use as an order of magnitude only",
        "last_half_mean": sref["last_half"]["mean"], "last_half_sd_per_point": sref["last_half"]["sd_per_point"],
        "last_half_slope_nats_per_Btok": sref["last_half"]["slope_nats_per_Btok"], "last_half_slope_ci95": sref["last_half"]["slope_ci95_pairs_block_bootstrap"],
        "last_half_change_ci95": sref["last_half"]["change_over_last_half_ci95"],
        "max_abs_smoothed_steps_100_1850": float(np.abs(np.array(sref["series"]["smooth"])[interior]).max()),
        "sd_smoothed_last_half": float(np.array(sref["series"]["smooth"])[lh].std(ddof=1)),
        "heldout_val09": ho_gap("M43_minus_M42", "val09"), "heldout_val10": ho_gap("M43_minus_M42", "val10"),
        "ratio_last_half_gap_to_heldout_seed_diff_val09": {k: gap_out[k]["last_half"]["mean"] / abs(ho_gap("M43_minus_M42", "val09")) for k in ("N_minus_M42", "N_minus_M43", "N_minus_Mavg", "N_minus_D")},
    }

    # ---- 11. training-loss vs held-out agreement (end window) ----
    agree = {}
    for k in gaps:
        e = gap_out[k]["end_window"]["mean"]
        agree[k] = {"train_last_10_points": e, "train_last_half_mean": gap_out[k]["last_half"]["mean"],
                    "heldout_val09": ho_gap(k, "val09"), "heldout_val10": ho_gap(k, "val10"),
                    "train_last_10_minus_val09": e - ho_gap(k, "val09")}

    # ---- caveats, deviations, answer summary ----
    sd_run = float(np.mean([run_out[r]["residual_sd_about_smooth_steps_ge_300"] for r in RUN_ORDER]))
    rc_all = list(rc.values())
    d_nm = gap_out["N_minus_M42"]["last_half"]["mean"] - gap_out["N_minus_M43"]["last_half"]["mean"]
    caveats = [
        f"Each logged loss is the mean CE of ONE optimizer step (262,144 tokens, before the update), logged every 10th step (190 of 1,907 steps per run; steps 1901-1907 are not logged); per-point noise about the smooth curve is SD {sd_run:.3f} nats per run (a gap: {gap_out['N_minus_M42']['last_half']['sd_per_point']:.3f}).",
        f"The four runs do not share batches (residual correlations between runs {min(rc_all):+.2f}..{max(rc_all):+.2f}; same-seed pairs {np.mean([rc['dense~moe42'], rc['dense~nestor'], rc['moe42~nestor']]):+.2f} vs cross-seed pairs {np.mean([rc['dense~moe43'], rc['moe42~moe43'], rc['moe43~nestor']]):+.2f} on average), so seed 42 does not pair the data across architectures and paired differences do not cancel batch noise.",
        "The seed noise reference is ONE MoE seed pair (1 d.o.f.) and also contains the platform difference (seed 43 on H100, seed 42 on the local GPU); read it as an order of magnitude. NESTOR-MoE(seed 43) is H100 vs H100 and NESTOR-MoE(seed 42) is H100 vs local; the two differ by " + f"{d_nm:+.3f} nats (last-half means), which equals the seed-reference mean by construction.",
        "NESTOR is a single run (seed 42): the NESTOR-MoE(seed 42) and NESTOR-MoE(seed 43) curves share it and are not independent evidence of the gap; NESTOR's own seed variability is unknown (the MoE seed pair is the only seed reference available).",
        "Peak learning rates differ by architecture (largest Muon group: dense %.4f, MoE %.5f, NESTOR %.5f; Muon base 0.01 times a matrix-shape scale), so 'same recipe' is not the same effective LR; whether each architecture is near its LR optimum at a 10x shorter horizon is untested." % tuple(sched_check[r]["lr_max"]["fitted_peak"] for r in ("dense", "moe42", "nestor")),
        "Tokens and the LR multiplier are perfectly collinear within one run (cosine: %.2f of peak at the start of the last half, %.3f at step 1900), so a slope in the last half cannot separate a token effect from an annealing effect." % (lr_multiplier(960, 0.0283), lr_multiplier(1900, 0.0283)),
        "The first 0.4B tokens of the paper's 5B run are warmup (1,526 steps) whereas this run warms up in 0.04B tokens: the early phase in which the NESTOR deficit appears (steps 250-350, 0.07-0.09B tokens, LR near peak) has no counterpart at the same token count in the 5B run (LR %.2f-%.2f of peak there)." % (mult5(0.0675), mult5(0.09)),
        "Held-out comparison with the paper mixes different evaluation sets (paper: its own DCLM split; here: shard-09/shard-10 slices), different data order (authors: DistributedSampler seed 0; here: single-GPU RandomSampler) and a single seed per cell; only between-architecture differences on one evaluation set are compared.",
    ]
    deviations = [
        "No reduction in data scope: all four logs were parsed in full (190 of 190 logged points per run, 760 rows); the logs themselves contain only every 10th step and not steps 1901-1907.",
        "Method not performed, by instruction: power-law (or any other) extrapolation of the gap to the 5B-token budget. The design asked for power-law extrapolations; task item 4 forbids naive extrapolation across the LR schedule. The within-run trend is reported descriptively only.",
        "outputs/*_b05/*/ contain only config.json (local runs) and checkpoints, no logs; the local logs are logs/pretrain_{dense,moe}_100m_b05.log, the local NESTOR log is a stub (NESTOR was trained on the H100) and the H100 artifact logs were used for NESTOR and MoE seed 43. Input copies and sha256 are in inputs/.",
        "The two H100 runs have no config.json on disk: their schedule is established from the launcher scripts (--steps 1907 --warmup 153) and verified against the logged lr (max residual 5.2e-5, within 4-decimal rounding), not from a stored config.",
        "Smoothing span (leave-one-out CV with a 1 % smoothing-favoured rule; display only), block length (5 logged points), bootstrap sizes and the definition of the last half (logged steps 960-1900) are analysis choices of this report; sensitivity results are stored under gaps.*.robustness.",
    ]

    # ---- assemble ----
    result = {
        "meta": {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "script": os.path.abspath(__file__), "script_sha256": sha256_file(os.path.abspath(__file__)),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "threads_env": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "priority_below_normal_set": PRIORITY_BELOW_NORMAL, "gpu_used": False, "network_used": False,
            "rng_seed": SEED, "bootstrap": {"B_main": B_MAIN, "B_band": B_BAND, "B_theil_sen": B_TS, "block_len_primary_points": BLOCK_LEN, "block_len_primary_steps": BLOCK_LEN * 10, "type": "circular moving-block (Politis-Romano), percentile CI; block length capped at n/4 for windows < 20 points"},
            "inputs_sha256": manifest_check, "elapsed_s": None,
            "conventions": {
                "tokens": "tokens = step x 262,144 (cumulative tokens through the logged optimizer step); the logged loss at step N is the mean token CE (nats) of that single optimizer step's 262,144-token batch, computed before the update (not a window average)",
                "gap": "first minus second run, nats per token, same logged step; positive = first run has the higher loss",
                "last_half": "logged steps 960..1900 (tokens 0.2517..0.4981 B), i.e. strictly after step 953.5 = half of 1907; steps 1901-1907 are not logged",
                "slope_units": "nats per 1B tokens; 'change over the last half' = slope x 0.25B tokens",
            },
        },
        "runs": run_out,
        "data_quality": {"loss_equals_cross_entropy_all_rows": ce_equal, "n_points_per_run": 190, "step_grid": [10, 1900, 10],
                         "residual_corr_between_runs_steps_ge_300": rc, "n_residual_points": n_res, "null_95pct_abs_corr": 1.96 / math.sqrt(n_res),
                         "same_seed_pairs_mean_corr": float(np.mean([rc["dense~moe42"], rc["dense~nestor"], rc["moe42~nestor"]])),
                         "cross_seed_pairs_mean_corr": float(np.mean([rc["dense~moe43"], rc["moe42~moe43"], rc["moe43~nestor"]])),
                         "interpretation": "all six residual correlations lie within +-0.18 and the same-seed pairs (dense/MoE-42/NESTOR) are not more correlated than the cross-seed pairs; shared batches at a given step would give strongly positive correlations, so there is no evidence that seed 42 pairs the data across architectures (the sampler seed is drawn from the global RNG after model initialisation, which differs by architecture). Per-step noise is therefore treated as independent between runs."},
        "smoothing": {"gap": {"method": "local-linear LOESS, tricube, window = k nearest logged points", "k_neighbors": K_GAP, "interior_half_width_steps": (K_GAP - 1) / 2 * 10,
                              "rule": "largest k whose pooled leave-one-out CV MSE (4 main gap series) is within 1% of the minimum", "k_at_cv_minimum": K_GAP_CVMIN,
                              "cv_grid_mse": {str(k_): v for k_, v in cv_gap.items()}},
                      "curve": {"method": "local-quadratic LOESS, tricube, window = k nearest logged points", "k_neighbors": K_CURVE, "interior_half_width_steps": (K_CURVE - 1) / 2 * 10,
                                "rule": "largest k whose pooled leave-one-out CV MSE (4 loss curves) is within 1% of the minimum", "k_at_cv_minimum": K_CURVE_CVMIN,
                                "cv_grid_mse": {str(k_): v for k_, v in cv_curve.items()}},
                      "use": "smooth curves and bands are for display only; all slopes, means and CIs are computed on the raw logged points"},
        "schedule": schedule,
        "series": {"steps": [int(s_) for s_ in steps], "tokens_B": [f(v) for v in tokB], "loss": curves,
                   "lr_max_logged": {r: [f(v) for v in lr_max[r]] for r in RUN_ORDER}, "lr_group0_logged": {r: [f(v) for v in lr_g0[r]] for r in RUN_ORDER}},
        "gaps": gap_out, "phases": phases, "seed_noise_reference": seed_ref,
        "heldout_stageA": {"values": ho, "source": "reproduction_stageA.json (fp32 CE, val09 first 10,004,085 tokens of shard 09; val10 first 5,004,090 tokens of shard 10)"},
        "train_vs_heldout_agreement": agree, "paper_comparison": paper_cmp,
        "caveats": caveats, "deviations": deviations,
    }
    result["meta"]["elapsed_s"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gap_trajectory.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, allow_nan=False)
    # csv
    with open(os.path.join(OUT, "training_loss_parsed.csv"), "w", encoding="utf-8", newline="") as fh:
        fh.write("run,step,tokens,loss,lr,lr_max\n")
        for r in RUN_ORDER:
            for d in parsed[r]:
                fh.write(f"{r},{d['step']},{d['step'] * TOKENS_PER_STEP},{d['loss']:.4f},{d['lr']:.4f},{d['lr_max']:.4f}\n")
    with open(os.path.join(OUT, "gap_series.csv"), "w", encoding="utf-8", newline="") as fh:
        keys = list(gaps)
        fh.write("step,tokens_B," + ",".join(f"{k}_raw,{k}_smooth,{k}_lo95,{k}_hi95" for k in keys) + "\n")
        for i, s_ in enumerate(steps):
            cells = []
            for k in keys:
                sr = gap_out[k]["series"]
                cells += [f"{sr['raw'][i]:.6f}", f"{sr['smooth'][i]:.6f}", f"{sr['lo95'][i]:.6f}", f"{sr['hi95'][i]:.6f}"]
            fh.write(f"{s_},{tokB[i]:.6f}," + ",".join(cells) + "\n")
    print("OK elapsed %.1fs; priority_below_normal=%s; K_GAP=%s K_CURVE=%s" % (time.time() - t0, PRIORITY_BELOW_NORMAL, K_GAP, K_CURVE))

if __name__ == "__main__":
    main()
