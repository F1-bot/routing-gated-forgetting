#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
power_sim_v1.py -- simulation-based power analysis and design diagnostics for preregistration v1.0
("Routing-gated forgetting in routed transformers"; extension of arXiv 2610.31818, NESTOR), implementing the
the design (crossover design C1/C2, gates G0-G2).

PLANNING SIMULATION. Every generative assumption is listed in ASSUMPTIONS (written to assumptions.json) and in
Appendix A of the pre-registration. Where the review supplied a number it is reused:
  * per-run noise SD 0.07 in log F and module sensitivity nu log-normal with CV 0.25/0.5/1.0, 70% of its
    log-variance shared across corpora (an independent recomputation, R1/R2);
  * SGD mu_Q exponent 1.99 / 1.87 / 1.32 / 0.46 at median gain 0.066 / 0.29 / 0.84 / 1.0 and per-matrix Muon
    -0.00..0.01 (R5);
  * per-run costs MoE 0.40 h, Nestor 0.97 h (incl. compile and 2 x 2.048M-token evaluation), one extra 2.048M-token
    evaluation 3.56 min at 9.6k tok/s (R8).

Parts (python power_sim_v1.py <part> [--out DIR] [--reps R] [--seed S] [--quick]):
  lookup   power of the one-sided exact sign-flip test over D independent draws vs standardized effect
           delta = E[I]/sd(I) (normal and t5 noise); minimum attainable p per D
  c1       MoE world simulation of the C1 cross-corpus crossover: G0 feasibility (median predicted contrast),
           matching acceptance, sd(I) components, type-I error under the scientific null (incl. the review's
           fixed-pair counterfactual), power vs elasticity beta for 3 and 5 draws per cell; two-stage G2 rule
  c1n      Nestor private-expert crossover (C1-N, secondary): sign agreement and t-test power for 3/5 draws
  c2       C2 optimizer crossover (SGD vs per-matrix Muon x target-usage tilt): power vs number of draws,
           exact sign-flip vs t-test; TOST of mu_Q-invariance under Muon
  q2       weighted (Gumbel-top-k) generator at q = 1/2 (G1 fallback): predicted contrast vs tilt strength
  budget   GPU-hours per block for the pre-registered design variants
  all      everything above;  figure   power_curves_v1.png from the CSVs in --out

Hard constraints honoured: CPU only (CUDA_VISIBLE_DEVICES=''), <= 6 BLAS threads, no multiprocessing.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import argparse
import csv
import itertools
import json
import math
import time

import numpy as np

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

# ----------------------------------------------------------------------------------------------------------------
# Pre-registered constants (must match preregistration_v1.0_draft.md)
# ----------------------------------------------------------------------------------------------------------------
ALPHA_FWER = 0.05
LOG_G0 = math.log(1.3)             # G0: minimum median predicted crossover contrast |P|
LOG_C2_TILT = math.log(1.5)        # G0-C2: minimum median target-usage tilt Delta_Q
MATCH_TOL = math.log(1.05)         # matching tolerance on RO (log scale)
BETA0 = 0.25                       # C1 minimum-effect null: H0 beta <= 0.25 (statistic J = I - BETA0 * P)
BETA_PLAN = 0.75                   # C1 planning effect (25% attenuation of the theory value 1)
SESOI_BETA = BETA_PLAN - BETA0     # margin above the null used by the G2 seed rule (= 0.5 x P-bar)
DE0 = 0.25                         # C2 minimum-effect null: H0 e_SGD - e_Muon <= 0.25 (statistic J2 = I2 - DE0 * dQ)
DE_PLAN = 1.0                      # C2 planning effect: exponent difference 1 (first-order SGD branch, no A0)
SESOI_DE = DE_PLAN - DE0
EQ_MARGIN = 0.25                   # C2 secondary TOST margin on the Muon mu_Q elasticity
TARGET_POWER = 0.80
UCL_LEVEL = 0.80
N_AA = 3                           # A/A pairs in G2
ND_CHOICES = (3, 5)                # draws per C1 cell (base x target)
Q_DEFAULT = 0.25
POOLS = (1 / 2, 1 / 3)             # pool fractions of the constrained generator, in order of preference

CORPORA = ("dclm", "code", "math", "german", "books")
CI = {c: i for i, c in enumerate(CORPORA)}
# log-SD of per-expert routing shares at 'specialization 1' (paper App. C.6, 500M/32k: on code and German a third
# to a half of the experts of some layers carry < 1/2 of the uniform load -> log-SD ~0.8-1.2; math milder; books
# near DCLM). The 100M / 0.5B-token bases may be less specialized: specialization factor lambda scales all non-DCLM
# log-SDs. DCLM load CV 0.10 as in the review's R1.
CORPUS_SD = {"dclm": 0.10, "code": 0.90, "math": 0.60, "german": 0.90, "books": 0.30}
CV_SEL = 0.03                      # relative error of a module's selection-half share estimate (256 windows)
CV_EST = 0.02                      # relative error of the estimation (evaluation-window) share

ASSUMPTIONS = {
    "routing_model": "per layer, log-normal routing shares per module and corpus, log-SD = CORPUS_SD x lambda "
                     "(DCLM fixed at 0.10), Gaussian copula across corpora (corr(DCLM, other) 0.3, corr(books, DCLM) 0.6, "
                     "corr(code, math) = rho_cm, other pairs 0.2); shares serve as effective-gate second moments",
    "sensitivity_model": "nu_{m,c} = exp(sqrt(0.7) a_m + sqrt(0.3) b_{m,c}), a, b log-normal with CV cv_nu (R1)",
    "forgetting_model": "F_c(S) = K_c * A(S) * sum_{m in S} (N w_c(m))^beta nu_{m,c} * exp(noise); A(S) common to corpora",
    "noise_model": "per run and corpus: training noise N(0, sigma_run^2) in log F (0.07), independent across "
                   "runs; evaluation noise relative SE r_eval with correlation 0.5 between runs evaluated on the same windows",
    "c2_model": "F_DCLM under optimizer o: sum_{m in S} N w_P(m) nu_m h_o(m); h_SGD = (N w_T(m) g_m)^e_SGD with per-module "
                "target-gradient-scale heterogeneity g_m log-normal CV cv_g; h_Muon = (N w_T(m))^e_Muon",
    "nestor_private": "13 layers x 8 columns x 8 private experts; share = column share (log-SD 0.10 x lambda, last layer "
                      "uniform) x within-column share (log-SD CORPUS_SD x lambda)",
}

MODEL = {"moe": dict(L=12, N=24), "nestor": dict(L=13, N=64, C=8, E=8)}

# Cost model (R8 + this design)
COST = dict(moe_run=0.40, nestor_run=0.97, ev=2.048e6 / 9600 / 3600,
            int_c1=3 * 4 * 262144 / 9600 / 3600,      # interim steps 8/24/48 x {DCLM, target, code, math} x 64 windows
            int_std=3 * 2 * 262144 / 9600 / 3600)     # interim steps x {DCLM, target} x 64 windows


# ----------------------------------------------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def chi2_ppf(p, df):
    if _st is not None:
        return float(_st.chi2.ppf(p, df))
    z = math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)  # pragma: no cover
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


def _erfinv(y):  # pragma: no cover
    a = 0.147
    ln = math.log(1.0 - y * y)
    t = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def t_ppf(p, df):
    return float(_st.t.ppf(p, df))


def lognorm(rng, shape, cv):
    if cv <= 0:
        return np.ones(shape)
    s2 = math.log1p(cv * cv)
    return np.exp(math.sqrt(s2) * rng.standard_normal(shape) - 0.5 * s2)


def corr_matrix(rho_cm, rho_floor=None):
    R = np.array([[1.0, 0.3, 0.3, 0.3, 0.6],
                  [0.3, 1.0, rho_cm, 0.2, 0.2],
                  [0.3, rho_cm, 1.0, 0.2, 0.2],
                  [0.3, 0.2, 0.2, 1.0, 0.2],
                  [0.6, 0.2, 0.2, 0.2, 1.0]])
    if rho_floor is not None:                    # router-popularity sensitivity: common factor across corpora
        off = ~np.eye(5, dtype=bool)
        R[off] = np.maximum(R[off], rho_floor)
    assert np.linalg.eigvalsh(R).min() > 0, "correlation matrix not PD"
    return R


_SIGNS = {}


def sign_matrix(D, n_mc=4095, seed=12345):
    """Rows = sign vectors. Exact enumeration for D <= 14, else identity + n_mc random rows."""
    if D in _SIGNS:
        return _SIGNS[D]
    if D <= 14:
        S = np.array(list(itertools.product((-1.0, 1.0), repeat=D)))
        exact = True
    else:
        r = np.random.default_rng(seed + D)
        S = np.vstack([np.ones((1, D)), r.integers(0, 2, size=(n_mc, D)) * 2.0 - 1.0])
        exact = False
    _SIGNS[D] = (S, exact)
    return _SIGNS[D]


def signflip_p(I):
    """One-sided (mean > 0) sign-flip p-values, row-wise. I: (R, D)."""
    R, D = I.shape
    S, exact = sign_matrix(D)
    obs = I.mean(1)
    null = I @ S.T / D
    cnt = (null >= obs[:, None] - 1e-12).sum(1)
    if exact:
        return cnt / S.shape[0]
    return cnt / S.shape[0]  # identity row included -> (1 + #{null >= obs among random}) / (1 + n_mc)


def t_p(I):
    R, D = I.shape
    m = I.mean(1)
    s = I.std(1, ddof=1)
    t = m / (s / math.sqrt(D))
    return 1.0 - _st.t.cdf(t, D - 1)


# ----------------------------------------------------------------------------------------------------------------
# worlds (one pretrained base = one world)
# ----------------------------------------------------------------------------------------------------------------
def make_world(rng, arch, lam, rho_cm, cv_nu, cv_g=0.5, rho_shared=0.7, rho_floor=None):
    R = corr_matrix(rho_cm, rho_floor)
    chol = np.linalg.cholesky(R)
    sds = np.array([CORPUS_SD["dclm"]] + [CORPUS_SD[c] * lam for c in CORPORA[1:]])
    if arch == "moe":
        L, N = MODEL["moe"]["L"], MODEL["moe"]["N"]
        Z = rng.standard_normal((L, N, 5)) @ chol.T
        u = np.exp(Z * sds - 0.5 * sds ** 2)
        w = u / u.sum(1, keepdims=True)
    else:
        L, C, E = MODEL["nestor"]["L"], MODEL["nestor"]["C"], MODEL["nestor"]["E"]
        N = C * E
        col_sd = np.array([0.05] + [0.10 * lam] * 4)
        Zc = rng.standard_normal((L, C, 5)) @ chol.T
        uc = np.exp(Zc * col_sd - 0.5 * col_sd ** 2)
        uc[-1] = 1.0                                         # last block: every column selected (Kc = C)
        wc = uc / uc.sum(1, keepdims=True)
        Ze = rng.standard_normal((L, C, E, 5)) @ chol.T
        ue = np.exp(Ze * sds - 0.5 * sds ** 2)
        we = ue / ue.sum(2, keepdims=True)
        w = (wc[:, :, None, :] * we).reshape(L, N, 5)
    s2 = math.log1p(cv_nu ** 2)
    s = math.sqrt(s2)
    a = s * rng.standard_normal((L, N)) - 0.5 * s2
    b = s * rng.standard_normal((L, N, 5)) - 0.5 * s2
    nu = np.exp(math.sqrt(rho_shared) * a[..., None] + math.sqrt(1 - rho_shared) * b)
    w_sel = w * lognorm(rng, w.shape, CV_SEL)
    w_sel /= w_sel.sum(1, keepdims=True)
    w_est = w * lognorm(rng, w.shape, CV_EST)
    w_est /= w_est.sum(1, keepdims=True)
    g = lognorm(rng, (L, N), cv_g)
    return dict(L=L, N=N, w=w, w_sel=w_sel, w_est=w_est, nu=nu, g=g)


def gather(W, S):
    """W: (L, N) per-module values; S: (M, L, k) module indices -> (M, L, k) values."""
    L = W.shape[0]
    return W[np.arange(L)[None, :, None], S]


def ro(W, S):
    return gather(W, S).sum(2).mean(1)


def propose(rng, world, tilt, k, f_pool, M):
    """Pool generator: per layer, A = random k of the lowest-tilt f_pool, B = random k of the highest-tilt f_pool."""
    L, N = tilt.shape
    npool = int(round(f_pool * N))
    if npool < k:
        raise ValueError("pool smaller than subset")
    order = np.argsort(tilt, axis=1)
    lo, hi = order[:, :npool], order[:, N - npool:]
    ka = np.argsort(rng.random((M, L, npool)), axis=2)[:, :, :k]
    kb = np.argsort(rng.random((M, L, npool)), axis=2)[:, :, :k]
    A = np.take_along_axis(np.broadcast_to(lo[None], (M, L, npool)), ka, 2)
    B = np.take_along_axis(np.broadcast_to(hi[None], (M, L, npool)), kb, 2)
    return A, B


def propose_random(rng, world, k, M):
    """A/A generator: disjoint random k-subsets per layer (no designed contrast)."""
    L, N = world["L"], world["N"]
    perm = np.argsort(rng.random((M, L, N)), axis=2)
    return perm[:, :, :k], perm[:, :, k:2 * k]


def propose_weighted(rng, world, tilt, k, tau, M):
    """Gumbel-top-k generator (q = 1/2 fallback): A ~ weights exp(-tau z), B ~ exp(+tau z), independent."""
    L, N = tilt.shape
    zt = (tilt - tilt.mean(1, keepdims=True)) / (tilt.std(1, keepdims=True) + 1e-12)
    ga = -tau * zt[None] - np.log(-np.log(rng.random((M, L, N))))
    gb = +tau * zt[None] - np.log(-np.log(rng.random((M, L, N))))
    return np.argsort(-ga, axis=2)[:, :, :k], np.argsort(-gb, axis=2)[:, :, :k]


def swap_search(rng, world, tilt, k, f_pool, M, corpora, tol=MATCH_TOL, iters=800):
    """Pre-registered fallback generator: M chains, each initialised with random k-subsets of the low-tilt pool (A)
    and the high-tilt pool (B) per layer, then randomized greedy swaps (member <-> non-member of the same pool and
    layer, alternating A and B) accepted only if they reduce max_c |log RO_c(A) - log RO_c(B)| over the matched corpora;
    chains that reach the tolerance are returned. Selection uses selection-half statistics only (outcome-blind)."""
    L, N = tilt.shape
    npool = int(round(f_pool * N))
    order = np.argsort(tilt, axis=1)
    pools = (order[:, :npool], order[:, N - npool:])
    Wm = [world["w_sel"][..., CI[c]] for c in corpora]
    sets = []
    for side in range(2):
        perm = np.argsort(rng.random((M, L, npool)), axis=2)
        sets.append([perm[:, :, :k].copy(), perm[:, :, k:].copy()])

    def module_idx(side, pos):
        return np.take_along_axis(np.broadcast_to(pools[side][None], (pos.shape[0], L, npool)), pos, 2)

    ro_ = np.zeros((2, len(corpora), M))
    for side in range(2):
        idx = module_idx(side, sets[side][0])
        for ci, W in enumerate(Wm):
            ro_[side, ci] = gather(W, idx).sum(2).mean(1)

    def obj(r):
        return np.abs(np.log(r[0]) - np.log(r[1])).max(0)

    cur = obj(ro_)
    ar = np.arange(M)
    for it in range(iters):
        if (cur <= tol).all():
            break
        side = it % 2
        l = rng.integers(0, L, M)
        i = rng.integers(0, k, M)
        j = rng.integers(0, npool - k, M)
        pin = sets[side][0][ar, l, i]
        pout = sets[side][1][ar, l, j]
        m_in = pools[side][l, pin]
        m_out = pools[side][l, pout]
        new = ro_.copy()
        for ci, W in enumerate(Wm):
            new[side, ci] = ro_[side, ci] + (W[l, m_out] - W[l, m_in]) / L
        nobj = obj(new)
        acc = (nobj < cur) & (cur > tol)
        sets[side][0][ar[acc], l[acc], i[acc]] = pout[acc]
        sets[side][1][ar[acc], l[acc], j[acc]] = pin[acc]
        ro_[:, :, acc] = new[:, :, acc]
        cur = np.where(acc, nobj, cur)
    ok = cur <= tol
    return module_idx(0, sets[0][0][ok]), module_idx(1, sets[1][0][ok]), float(ok.mean())


def match_mask(world, A, B, corpora, tol=MATCH_TOL):
    ok = np.ones(A.shape[0], bool)
    for c in corpora:
        Wc = world["w_sel"][..., CI[c]]
        ok &= np.abs(np.log(ro(Wc, A)) - np.log(ro(Wc, B))) <= tol
    return ok


def pred_contrast(world, A, B):
    We = world["w_est"]
    lc = np.log(ro(We[..., CI["code"]], B)) - np.log(ro(We[..., CI["code"]], A))
    lm = np.log(ro(We[..., CI["math"]], B)) - np.log(ro(We[..., CI["math"]], A))
    return lc - lm


def tilt_cm(world):
    Ws = world["w_sel"]
    return np.log(Ws[..., CI["code"]]) - np.log(Ws[..., CI["math"]])


def draw_pairs(rng, world, kind, target, k, n_need, f_pool=None, tau=None, M=512, max_batches=40, tol=MATCH_TOL):
    """Return n_need accepted pairs (A, B) [arrays (n, L, k)], the acceptance rate and #batches; None if infeasible."""
    As, Bs, acc, nb = [], [], [], 0
    got = 0
    while got < n_need and nb < max_batches:
        nb += 1
        if kind == "c1":
            A, B = propose(rng, world, tilt_cm(world), k, f_pool, M)
            mc = (target, "dclm")
        elif kind == "c1w":
            A, B = propose_weighted(rng, world, tilt_cm(world), k, tau, M)
            mc = (target, "dclm")
        elif kind == "aa":
            A, B = propose_random(rng, world, k, M)
            mc = (target, "dclm")
        elif kind == "c2":
            # low-target-usage pool -> A (= L_Q), high -> B (= H_Q); matched on DCLM, code, math
            A, B = propose(rng, world, np.log(world["w_sel"][..., CI[target]]), k, f_pool, M)
            mc = ("dclm", "code", "math")
        else:
            raise ValueError(kind)
        ok = match_mask(world, A, B, mc, tol)
        acc.append(ok.mean())
        if ok.any():
            As.append(A[ok])
            Bs.append(B[ok])
            got += int(ok.sum())
    if got < n_need and kind == "c1":
        # pre-registered fallback: randomized swap search from fresh random initialisations
        for _ in range(4):
            A, B, _ = swap_search(rng, world, tilt_cm(world), k, f_pool, 2 * (n_need - got) + 8, (target, "dclm"), tol)
            if A.shape[0]:
                As.append(A)
                Bs.append(B)
                got += A.shape[0]
            if got >= n_need:
                break
    if got < n_need:
        return None, None, float(np.mean(acc)), nb
    A = np.concatenate(As)[:n_need]
    B = np.concatenate(Bs)[:n_need]
    return A, B, float(np.mean(acc)), nb


def g0_choose_pool(rng, world, target, k, M=512):
    """G0 rule ("best feasible contrast"): among the pre-registered pool fractions (1/2, 1/3), take the one with the
    larger median predicted contrast over 64 accepted generator draws; G0 passes if that median >= log 1.3.
    Returns (chosen pool, its median P, pass flag, median P per pool, acceptance per pool)."""
    meds, accs, pairs = {}, {}, {}
    for f in POOLS:
        A, B, acc, _ = draw_pairs(rng, world, "c1", target, k, 64, f_pool=f, M=M, max_batches=20)
        accs[f] = acc
        pairs[f] = (A, B)
        meds[f] = float(np.median(pred_contrast(world, A, B))) if A is not None else float("-inf")
    f_best = max(POOLS, key=lambda f: meds[f])
    g0_choose_pool.last_pairs = pairs[f_best]
    return f_best, meds[f_best], bool(meds[f_best] >= LOG_G0), meds, accs


def log_forget(world, S, corpus, beta):
    """log of the subset-specific part of F_c(S) for several betas: returns (len(beta), M)."""
    Wc = world["w"][..., CI[corpus]] * world["N"]
    nu = world["nu"][..., CI[corpus]]
    out = []
    for b in beta:
        out.append(np.log(gather(Wc ** b * nu, S).sum((1, 2))))
    return np.array(out)


def c1_I(rng, world, A, B, betas, sig_run, r_eval, rho_win=0.5):
    """Crossover statistic per draw for several betas (common noise)."""
    n = A.shape[0]
    lcB, lcA = log_forget(world, B, "code", betas), log_forget(world, A, "code", betas)
    lmB, lmA = log_forget(world, B, "math", betas), log_forget(world, A, "math", betas)
    sd_noise = math.sqrt(4 * sig_run ** 2 + 4 * r_eval ** 2 * (1 - rho_win))
    noise = sd_noise * rng.standard_normal(n)
    return (lcB - lcA) - (lmB - lmA) + noise[None, :]


# ----------------------------------------------------------------------------------------------------------------
# Part lookup
# ----------------------------------------------------------------------------------------------------------------
def part_lookup(rng, reps, out):
    rows = []
    for D in (3, 4, 5, 6, 8, 10, 12, 16, 20, 24):
        S, exact = sign_matrix(D)
        pmin = 1.0 / S.shape[0] if exact else 1.0 / S.shape[0]
        for dist in ("normal", "t5"):
            for delta in np.round(np.arange(0.0, 2.51, 0.1), 2):
                if dist == "normal":
                    I = delta + rng.standard_normal((reps, D))
                else:
                    I = delta + rng.standard_t(5, (reps, D)) / math.sqrt(5 / 3)
                p = signflip_p(I)
                pt = t_p(I)
                rows.append(dict(D=D, dist=dist, delta=float(delta), exact=int(exact), p_min=pmin,
                                 power_sf_025=float((p <= 0.025).mean()), power_sf_05=float((p <= 0.05).mean()),
                                 power_t_025=float((pt <= 0.025).mean()), power_t_05=float((pt <= 0.05).mean())))
        log(f"lookup D={D} done")
    write_csv(os.path.join(out, "lookup_v1.csv"), rows)
    return rows


def lookup_power(rows, D, delta, alpha=0.025, dist="normal"):
    key = "power_sf_025" if abs(alpha - 0.025) < 1e-9 else "power_sf_05"
    pts = sorted((r["delta"], r[key]) for r in rows if int(r["D"]) == D and r["dist"] == dist)
    xs = np.array([a for a, _ in pts])
    ys = np.maximum.accumulate(np.array([b for _, b in pts]))
    return float(np.interp(delta, xs, ys))


def nd_rule(rows, sesoi_I, sig_hat, alpha):
    """G2 rule: n_d = 3 if power(SESOI / UCL80, D = 12) >= 0.8, else 5; flag if power(., 20) < 0.8."""
    ucl = sig_hat * math.sqrt(N_AA / chi2_ppf(1 - UCL_LEVEL, N_AA))
    d = sesoi_I / max(ucl, 1e-12)
    if lookup_power(rows, 12, d, alpha) >= TARGET_POWER:
        return 3, False, ucl
    return 5, lookup_power(rows, 20, d, alpha) < TARGET_POWER, ucl


# ----------------------------------------------------------------------------------------------------------------
# Part c1 (MoE crossover)
# ----------------------------------------------------------------------------------------------------------------
C1_BASE = dict(lam=0.5, rho_cm=0.4, cv_nu=0.5, sig_run=0.07, r_eval=0.10)
C1_VARIANTS = [
    ("baseline", {}),
    ("lambda 1.0 (500M-like specialization)", dict(lam=1.0)),
    ("lambda 0.35 (weak specialization)", dict(lam=0.35)),
    ("lambda 0.2 (very weak specialization)", dict(lam=0.2)),
    ("rho_cm 0.0", dict(rho_cm=0.0)),
    ("rho_cm 0.7", dict(rho_cm=0.7)),
    ("cv_nu 0.25", dict(cv_nu=0.25)),
    ("cv_nu 1.0", dict(cv_nu=1.0)),
    ("cv_nu 1.0, corpus-specific share 0.5", dict(cv_nu=1.0, rho_shared=0.5)),
    ("sigma_run 0.035", dict(sig_run=0.035)),
    ("sigma_run 0.14", dict(sig_run=0.14)),
    ("r_eval 0.05", dict(r_eval=0.05)),
    ("pessimistic: lambda 0.35, rho_cm 0.7, cv_nu 1.0, sigma_run 0.14", dict(lam=0.35, rho_cm=0.7, cv_nu=1.0, sig_run=0.14)),
]
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def c1_world_run(rng, sc, lookup_rows, alpha_c1, n_max=5):
    """One simulated study: 2 MoE bases x 2 targets; G0 per base; n_max draws per cell; A/A pairs on base 1."""
    k = int(round(Q_DEFAULT * MODEL["moe"]["N"]))
    res = dict(g0_pass=True, pools=[], medP=[], acc=[], P=[], I={b: [] for b in BETAS}, fixed_I={b: [] for b in BETAS})
    aa_I = {b: [] for b in BETAS}
    for base in range(2):
        world = make_world(rng, "moe", sc["lam"], sc["rho_cm"], sc["cv_nu"], rho_shared=sc.get("rho_shared", 0.7))
        for ti, target in enumerate(("german", "books")):
            f, medf, ok, meds, accs = g0_choose_pool(rng, world, target, k)
            if not np.isfinite(medf):
                return None
            res["medP"].append(medf)
            res["g0_pass"] &= ok
            res["pools"].append(f)
            A64, B64 = g0_choose_pool.last_pairs
            l0 = [log_forget(world, S, c, (0.0,))[0] for S, c in ((B64, "code"), (A64, "code"), (B64, "math"), (A64, "math"))]
            res.setdefault("pool_effect", []).append(float(np.mean((l0[0] - l0[1]) - (l0[2] - l0[3]))))
            res.setdefault("pool_effect_P", []).append(float(np.mean(pred_contrast(world, A64, B64))))
            res["acc"].append(accs[f])
            A, B, acc, _ = draw_pairs(rng, world, "c1", target, k, n_max, f_pool=f)
            if A is None:
                return None
            res["P"].append(pred_contrast(world, A, B))
            Ib = c1_I(rng, world, A, B, BETAS, sc["sig_run"], sc["r_eval"])
            for bi, b in enumerate(BETAS):
                res["I"][b].append(Ib[bi])
            # the review's flaw for comparison: ONE fixed pair per (base, target), replicated by n_max "seeds"
            Af, Bf = A[:1].repeat(n_max, 0), B[:1].repeat(n_max, 0)
            If = c1_I(rng, world, Af, Bf, BETAS, sc["sig_run"], sc["r_eval"])
            for bi, b in enumerate(BETAS):
                res["fixed_I"][b].append(If[bi])
            if base == 0:
                nAA = 1 if target == "german" else 2
                Aa, Ba, _, _ = draw_pairs(rng, world, "aa", target, k, nAA)
                if Aa is None:
                    return None
                Ia = c1_I(rng, world, Aa, Ba, BETAS, sc["sig_run"], sc["r_eval"])
                for bi, b in enumerate(BETAS):
                    aa_I[b].append(Ia[bi])
    res["P"] = np.concatenate(res["P"])
    for b in BETAS:
        res["I"][b] = np.stack(res["I"][b])            # (4 cells, n_max)
        res["fixed_I"][b] = np.stack(res["fixed_I"][b])
        aa_I[b] = np.concatenate(aa_I[b])               # (3,)
    res["aa_I"] = aa_I
    return res


def part_c1(rng, reps, out, lookup_rows):
    rows = []
    for name, ov in C1_VARIANTS:
        sc = dict(C1_BASE, **ov)
        sims = []
        t0 = time.time()
        while len(sims) < reps:
            r = c1_world_run(rng, sc, lookup_rows, 0.025)
            if r is not None:
                sims.append(r)
        g0 = np.array([s["g0_pass"] for s in sims])
        medP = np.array([np.nanmin(s["medP"]) for s in sims])         # binding cell
        accs = np.array([np.mean(s["acc"]) for s in sims])
        Pall = np.concatenate([s["P"] for s in sims])
        row = dict(scenario=name, **sc, reps=len(sims), p_g0_pass=float(g0.mean()),
                   medP_binding_median=float(np.median(medP)), medP_binding_p10=float(np.quantile(medP, 0.1)),
                   medP_binding_p90=float(np.quantile(medP, 0.9)), P_draw_mean=float(Pall.mean()),
                   P_draw_sd=float(Pall.std()), match_accept_mean=float(accs.mean()),
                   pool_half_share=float(np.mean([np.mean(np.array(s["pools"]) == 0.5) for s in sims])))
        pe = np.array([s["pool_effect"] for s in sims])             # (R, 4 cells): base-1 cells 0-1, base-2 cells 2-3
        row["pool_effect_sd"] = float(pe[:, [0, 2]].std())
        row["pool_effect_over_P"] = float(np.median(np.abs(pe[:, [0, 2]]) / np.array([s["pool_effect_P"] for s in sims])[:, [0, 2]]))
        for b in BETAS:
            Iall = np.stack([s["I"][b] for s in sims])               # (R, 4 cells, n_max)
            Pm = np.stack([s["P"].reshape(4, -1) for s in sims])
            for nd in ND_CHOICES:
                for b0, tag0 in ((0.0, "plain"), (BETA0, "min")):
                    J = Iall[:, :, :nd] - b0 * Pm[:, :, :nd]
                    p = signflip_p(J.reshape(len(sims), -1))
                    both_pos = (J[:, 0:2].mean((1, 2)) > 0) & (J[:, 2:4].mean((1, 2)) > 0)
                    p1 = signflip_p(J[:, 0:2].reshape(len(sims), -1))
                    p2 = signflip_p(J[:, 2:4].reshape(len(sims), -1))
                    for alpha, tag in ((0.025, "025"), (0.05, "05")):
                        row[f"{tag0}_pooled_{tag}_b{b}_nd{nd}"] = float((p <= alpha).mean())
                        row[f"{tag0}_rule_{tag}_b{b}_nd{nd}"] = float(((p <= alpha) & both_pos).mean())
                        row[f"{tag0}_iut_{tag}_b{b}_nd{nd}"] = float(((p1 <= alpha) & (p2 <= alpha)).mean())
                    if tag0 == "min":
                        row[f"min_rule_025_b{b}_nd{nd}_g0pass"] = float(((p <= 0.025) & both_pos)[g0].mean()) if g0.any() else float("nan")
            row[f"sdI_b{b}"] = float((Iall - b * Pm).std())
            row[f"meanI_over_meanP_b{b}"] = float(Iall.mean() / Pm.mean())
            # elasticity estimate (regression through the origin over the 12 draws) and its spread
            I12, P12 = Iall[:, :, :3].reshape(len(sims), -1), Pm[:, :, :3].reshape(len(sims), -1)
            bhat = (I12 * P12).sum(1) / (P12 ** 2).sum(1)
            row[f"betahat_mean_b{b}"] = float(bhat.mean())
            row[f"betahat_sd_b{b}"] = float(bhat.std())
            Fi = np.stack([s["fixed_I"][b] for s in sims])[:, :, :3].reshape(len(sims), -1)
            row[f"fixedpair_plain_pooled_025_b{b}"] = float((signflip_p(Fi) <= 0.025).mean())
            # two-stage G2 rule (alpha 0.025 = Holm step 1 with m = 2) with the confirmatory rule (min-effect null)
            aa = np.stack([s["aa_I"][b] for s in sims])               # (R, 3)
            sig_hat = np.sqrt((aa ** 2).mean(1))
            pbar = np.array([s["medP"][0] for s in sims])            # base 1, German cell: chosen pool's median P
            dec, nds, flags = [], [], []
            for i in range(len(sims)):
                nd, flag, _ = nd_rule(lookup_rows, SESOI_BETA * pbar[i], sig_hat[i], 0.025)
                J = Iall[i, :, :nd] - BETA0 * Pm[i, :, :nd]
                pos = (J[0:2].mean() > 0) and (J[2:4].mean() > 0)
                dec.append(bool(signflip_p(J.reshape(1, -1))[0] <= 0.025 and pos))
                nds.append(nd)
                flags.append(flag)
            row[f"twostage_rule_025_b{b}"] = float(np.mean(dec))
            row[f"twostage_mean_nd_b{b}"] = float(np.mean(nds))
            row[f"twostage_flag_b{b}"] = float(np.mean(flags))
            row[f"aa_sighat_median_b{b}"] = float(np.median(sig_hat))
        rows.append(row)
        log(f"c1 '{name}': {len(sims)} worlds in {time.time() - t0:.1f}s; G0 pass {row['p_g0_pass']:.3f}; "
            f"medP {row['medP_binding_median']:.3f}; pool-effect sd {row['pool_effect_sd']:.3f}; "
            f"typeI plain/min (b0, nd3) {row['plain_pooled_025_b0.0_nd3']:.3f}/{row['min_rule_025_b0.0_nd3']:.3f}; "
            f"power min-rule (b=.75, nd3) {row['min_rule_025_b0.75_nd3']:.3f}")
    write_csv(os.path.join(out, "c1_power.csv"), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Part g0map (feasibility of the C1 contrast and the C2 tilt over specialization x code-math correlation)
# ----------------------------------------------------------------------------------------------------------------
def part_g0map(rng, reps, out):
    rows = []
    k = int(round(Q_DEFAULT * MODEL["moe"]["N"]))
    for lam in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for rho in (0.0, 0.4, 0.7, 0.9):
            meds, passes, accs, dqs, c2ok = [], [], [], [], []
            for _ in range(reps):
                world = make_world(rng, "moe", lam, rho, 0.5)
                f, medf, ok, _, a = g0_choose_pool(rng, world, "german", k, M=256)
                meds.append(medf)
                passes.append(ok)
                accs.append(a[f])
                if rho == 0.4:
                    tiltQ = np.log(world["w_sel"][..., CI["german"]])
                    best = -np.inf
                    for fp in POOLS:
                        A, B, _ = swap_search(rng, world, tiltQ, k, fp, 32, ("dclm", "code", "math"))
                        if A.shape[0] >= 8:
                            We = world["w_est"][..., CI["german"]]
                            best = max(best, float(np.median(np.log(ro(We, B)) - np.log(ro(We, A)))))
                    dqs.append(best)
                    c2ok.append(best >= LOG_C2_TILT)
            rows.append(dict(lam=lam, rho_cm=rho, reps=reps, medP_median=float(np.median(meds)),
                             medP_p10=float(np.quantile(meds, 0.1)), medP_p90=float(np.quantile(meds, 0.9)),
                             p_g0_pass=float(np.mean(passes)), accept_mean=float(np.mean(accs)),
                             c2_medDQ_median=float(np.median(dqs)) if dqs else float("nan"),
                             p_c2_tilt_ok=float(np.mean(c2ok)) if c2ok else float("nan")))
        log(f"g0map lam={lam} done")
    write_csv(os.path.join(out, "g0map.csv"), rows)
    return rows


def part_popularity(rng, reps, out):
    """Router-popularity sensitivity (theory note v2 toy: cross-corpus log-usage correlations 0.78-0.96).
    G0-1 (C1 contrast) and G0-3 (C2 tilt, thresholds log 1.5 and log 2) on MoE-100M, German target."""
    rows = []
    k = int(round(Q_DEFAULT * MODEL["moe"]["N"]))
    for label, floor, rho in (("planning correlations", None, 0.4), ("popularity floor 0.85", 0.85, 0.85),
                              ("popularity floor 0.95", 0.95, 0.95)):
        for lam in (0.35, 0.5, 0.75, 1.0):
            meds, passes, accs, dqs = [], [], [], []
            for _ in range(reps):
                world = make_world(rng, "moe", lam, rho, 0.5, rho_floor=floor)
                f, medf, ok, _, a = g0_choose_pool(rng, world, "german", k, M=256)
                meds.append(medf)
                passes.append(bool(ok) and bool(np.isfinite(medf)))
                accs.append(a[f])
                tiltQ = np.log(world["w_sel"][..., CI["german"]])
                best = -np.inf
                for fp in POOLS:
                    A, B, _ = swap_search(rng, world, tiltQ, k, fp, 32, ("dclm", "code", "math"))
                    if A.shape[0] >= 8:
                        We = world["w_est"][..., CI["german"]]
                        best = max(best, float(np.median(np.log(ro(We, B)) - np.log(ro(We, A)))))
                dqs.append(best)
            dqs = np.array(dqs)
            rows.append(dict(scenario=label, rho_floor=(floor if floor is not None else 0.0), rho_cm=rho, lam=lam, reps=reps,
                             medP_median=float(np.nanmedian(meds)), p_g0_pass=float(np.mean(passes)),
                             accept_mean=float(np.mean(accs)), c2_medDQ_median=float(np.median(dqs)),
                             p_dq_ge_log15=float(np.mean(dqs >= np.log(1.5))), p_dq_ge_log2=float(np.mean(dqs >= np.log(2.0)))))
            log(f"popularity '{label}' lam={lam}: medP {rows[-1]['medP_median']:.3f}, G0 {rows[-1]['p_g0_pass']:.2f}, "
                f"medDQ {rows[-1]['c2_medDQ_median']:.3f}, P(dQ>=log2) {rows[-1]['p_dq_ge_log2']:.2f}")
    write_csv(os.path.join(out, "popularity_g0.csv"), rows)
    return rows


def part_c2match(rng, reps, out):
    """C2 generator fallback: tilt German usage while matching on DCLM usage only (the C2 primary outcome is DCLM
    forgetting), against the registered 3-corpus matching (DCLM, code, math). Share of bases with >= 8 of 32
    converged swap chains, median tilt, and shares reaching log 1.5 / log 2. MoE-100M; tolerance log 1.05."""
    rows = []
    k = int(round(Q_DEFAULT * MODEL["moe"]["N"]))
    for label, floor, rho in (("planning correlations", None, 0.4), ("popularity floor 0.85", 0.85, 0.85),
                              ("popularity floor 0.95", 0.95, 0.95)):
        for lam in (0.35, 0.5, 1.0):
            res = {m: [] for m in ("dclm+code+math", "dclm only")}
            for _ in range(reps):
                world = make_world(rng, "moe", lam, rho, 0.5, rho_floor=floor)
                tiltQ = np.log(world["w_sel"][..., CI["german"]])
                We = world["w_est"][..., CI["german"]]
                for mode, match in (("dclm+code+math", ("dclm", "code", "math")), ("dclm only", ("dclm",))):
                    best = -np.inf
                    for fp in POOLS:
                        A, B, _ = swap_search(rng, world, tiltQ, k, fp, 32, match)
                        if A.shape[0] >= 8:
                            best = max(best, float(np.median(np.log(ro(We, B)) - np.log(ro(We, A)))))
                    res[mode].append(best)
            for mode, v in res.items():
                v = np.array(v)
                rows.append(dict(scenario=label, lam=lam, match=mode, reps=reps, p_feasible=float(np.mean(np.isfinite(v))),
                                 medDQ_median=float(np.median(v)), p_dq_ge_log15=float(np.mean(v >= np.log(1.5))),
                                 p_dq_ge_log2=float(np.mean(v >= np.log(2.0)))))
                log(f"c2match '{label}' lam={lam} [{mode}]: feasible {rows[-1]['p_feasible']:.2f}, medDQ {rows[-1]['medDQ_median']:.3f}, "
                    f">=log1.5 {rows[-1]['p_dq_ge_log15']:.2f}, >=log2 {rows[-1]['p_dq_ge_log2']:.2f}")
    write_csv(os.path.join(out, "c2_matching_fallback.csv"), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Part c1n (Nestor private experts; secondary)
# ----------------------------------------------------------------------------------------------------------------
def part_c1n(rng, reps, out):
    rows = []
    k = int(round(Q_DEFAULT * MODEL["nestor"]["N"]))
    for name, ov in (("baseline", {}), ("lambda 1.0", dict(lam=1.0)), ("cv_nu 1.0, sigma_run 0.14", dict(cv_nu=1.0, sig_run=0.14))):
        sc = dict(C1_BASE, **ov)
        Is, Ps, medPs, g0 = [], [], [], []
        while len(Is) < reps:
            world = make_world(rng, "nestor", sc["lam"], sc["rho_cm"], sc["cv_nu"])
            f, medf, ok, meds, _ = g0_choose_pool(rng, world, "german", k)
            if not np.isfinite(medf):
                continue
            g0.append(ok)
            medPs.append(medf)
            A, B, _, _ = draw_pairs(rng, world, "c1", "german", k, 5, f_pool=f)
            if A is None:
                continue
            Ps.append(pred_contrast(world, A, B))
            Is.append(c1_I(rng, world, A, B, BETAS, sc["sig_run"], sc["r_eval"]))
        Is = np.stack(Is)          # (R, nbeta, 5)
        Ps = np.stack(Ps)          # (R, 5)
        row = dict(scenario=name, **sc, reps=reps, p_g0_pass=float(np.mean(g0)), medP_median=float(np.median(medPs)))
        for bi, b in enumerate(BETAS):
            for nd in (3, 5):
                J = Is[:, bi, :nd] - BETA0 * Ps[:, :nd]          # registered C1-N statistic (as C1)
                row[f"allpos_b{b}_nd{nd}"] = float((J > 0).all(1).mean())
                row[f"t05_b{b}_nd{nd}"] = float((t_p(J) <= 0.05).mean())
                row[f"sf05_b{b}_nd{nd}"] = float((signflip_p(J) <= 0.05).mean())
                row[f"meanJ_pos_b{b}_nd{nd}"] = float((J.mean(1) > 0).mean())
        rows.append(row)
        log(f"c1n '{name}' done: G0 {row['p_g0_pass']:.2f}, medP {row['medP_median']:.3f}")
    write_csv(os.path.join(out, "c1n_power.csv"), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Part c2 (optimizer crossover)
# ----------------------------------------------------------------------------------------------------------------
C2_BASE = dict(lam=0.5, rho_cm=0.4, cv_nu=0.5, cv_g=0.5, sig_run=0.07, r_eval=0.15)
C2_VARIANTS = [
    ("baseline", {}),
    ("lambda 1.0", dict(lam=1.0)),
    ("lambda 0.35", dict(lam=0.35)),
    ("cv_g 0.25", dict(cv_g=0.25)),
    ("cv_g 1.0", dict(cv_g=1.0)),
    ("sigma_run 0.14", dict(sig_run=0.14)),
    ("r_eval 0.25", dict(r_eval=0.25)),
    ("pessimistic: lambda 0.35, cv_g 1.0, sigma_run 0.14, r_eval 0.25", dict(lam=0.35, cv_g=1.0, sig_run=0.14, r_eval=0.25)),
]
C2_EXP = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 0.1), (2.0, 0.0)]   # (e_SGD, e_Muon)
C2_D = (3, 4, 5, 6, 8)


def c2_draws(rng, world, sc, n, k):
    """G0-C2 rule: pool fraction with the larger median target-usage tilt Delta_Q (matched on DCLM, code, math);
    feasible if that median >= log 1.5."""
    medD = {}
    We = world["w_est"][..., CI["german"]]
    tiltQ = np.log(world["w_sel"][..., CI["german"]])
    mc = ("dclm", "code", "math")
    for f in POOLS:
        A, B, _ = swap_search(rng, world, tiltQ, k, f, 32, mc)
        medD[f] = float(np.median(np.log(ro(We, B)) - np.log(ro(We, A)))) if A.shape[0] >= 8 else float("-inf")
    f = max(POOLS, key=lambda x: medD[x])
    if not np.isfinite(medD[f]):
        return None
    ok = medD[f] >= LOG_C2_TILT
    As, Bs, got = [], [], 0
    for _ in range(6):
        A, B, _ = swap_search(rng, world, tiltQ, k, f, 2 * n, mc)
        As.append(A)
        Bs.append(B)
        got += A.shape[0]
        if got >= n:
            break
    if got < n:
        return None
    A = np.concatenate(As)[:n]
    B = np.concatenate(Bs)[:n]
    We = world["w_est"][..., CI["german"]]
    dQ = np.log(ro(We, B)) - np.log(ro(We, A))
    N = world["N"]
    wP = world["w"][..., CI["dclm"]] * N
    wT = world["w"][..., CI["german"]] * N
    nu = world["nu"][..., CI["dclm"]]
    g = world["g"]
    outI, outR = {}, {}
    sd = math.sqrt(4 * sc["sig_run"] ** 2 + 4 * sc["r_eval"] ** 2 * 0.5)
    sd_m = math.sqrt(2 * sc["sig_run"] ** 2 + 2 * sc["r_eval"] ** 2 * 0.5)
    z = rng.standard_normal(n)
    zr = rng.standard_normal(n)
    for eS, eM in C2_EXP:
        hS = wP * nu * (wT * g) ** eS
        hM = wP * nu * wT ** eM
        lSB, lSA = np.log(gather(hS, B).sum((1, 2))), np.log(gather(hS, A).sum((1, 2)))
        lMB, lMA = np.log(gather(hM, B).sum((1, 2))), np.log(gather(hM, A).sum((1, 2)))
        # marginal noise models only: rM (Muon-arm contrast, used by the TOST) and I2 (used by the sign-flip test)
        # are analysed separately, so their noise draws are independent here
        rM = (lMB - lMA) + sd_m * zr
        I2 = (lSB - lSA) - (lMB - lMA) + sd * z
        outI[(eS, eM)] = I2
        outR[(eS, eM)] = rM
    return dict(ok=ok, medD=medD[f], dQ=dQ, I=outI, R=outR)


def part_c2(rng, reps, out):
    rows = []
    k = int(round(Q_DEFAULT * MODEL["moe"]["N"]))
    nmax = max(C2_D)
    for name, ov in C2_VARIANTS:
        sc = dict(C2_BASE, **ov)
        sims = []
        t0 = time.time()
        while len(sims) < reps:
            w1 = make_world(rng, "moe", sc["lam"], sc["rho_cm"], sc["cv_nu"], cv_g=sc["cv_g"])
            w2 = make_world(rng, "moe", sc["lam"], sc["rho_cm"], sc["cv_nu"], cv_g=sc["cv_g"])
            d1 = c2_draws(rng, w1, sc, nmax, k)
            d2 = c2_draws(rng, w2, sc, nmax, k)
            if d1 is None or d2 is None:
                continue
            sims.append((d1, d2))
        row = dict(scenario=name, **sc, reps=len(sims), p_tilt_feasible=float(np.mean([a["ok"] for a, _ in sims])),
                   medDQ_median=float(np.median([a["medD"] for a, _ in sims])),
                   dQ_draw_mean=float(np.mean(np.concatenate([a["dQ"] for a, _ in sims]))))
        for eS, eM in C2_EXP:
            for D in C2_D:
                # D = 3: base 1 only (layout); otherwise ceil(D/2) from base 1 and floor(D/2) from base 2
                n1 = D if D == 3 else (D + 1) // 2
                n2 = 0 if D == 3 else D // 2
                I = np.stack([np.concatenate([a["I"][(eS, eM)][:n1], b["I"][(eS, eM)][:n2]]) for a, b in sims])
                dq_ = np.stack([np.concatenate([a["dQ"][:n1], b["dQ"][:n2]]) for a, b in sims])
                J = I - DE0 * dq_
                p = signflip_p(J)
                pt = t_p(J)
                if n2 > 0:
                    both = (J[:, :n1].mean(1) > 0) & (J[:, n1:].mean(1) > 0)
                else:
                    both = np.ones(len(sims), bool)
                tag = f"eS{eS}_eM{eM}_D{D}"
                row[f"sf025_{tag}"] = float((p <= 0.025).mean())
                row[f"sf05_{tag}"] = float((p <= 0.05).mean())
                row[f"rule025_{tag}"] = float(((p <= 0.025) & both).mean())
                row[f"rule05_{tag}"] = float(((p <= 0.05) & both).mean())
                row[f"t025_{tag}"] = float((pt <= 0.025).mean())
                row[f"plain_sf025_{tag}"] = float((signflip_p(I) <= 0.025).mean())
                if eM == 0.0 and eS in (1.0, 0.0):
                    Rm = np.stack([np.concatenate([a["R"][(eS, eM)][:n1], b["R"][(eS, eM)][:n2]]) for a, b in sims])
                    dq = np.stack([np.concatenate([a["dQ"][:n1], b["dQ"][:n2]]) for a, b in sims])
                    e_d = Rm / dq
                    m = e_d.mean(1)
                    se = e_d.std(1, ddof=1) / math.sqrt(D)
                    hw = t_ppf(0.95, D - 1) * se
                    row[f"tost_{tag}"] = float(((m - hw > -EQ_MARGIN) & (m + hw < EQ_MARGIN)).mean())
            Iall = np.concatenate([np.concatenate([a["I"][(eS, eM)], b["I"][(eS, eM)]]) for a, b in sims])
            dqa = np.concatenate([np.concatenate([a["dQ"], b["dQ"]]) for a, b in sims])
            row[f"meanI_eS{eS}_eM{eM}"] = float(Iall.mean())
            row[f"sdI_eS{eS}_eM{eM}"] = float((Iall - (eS - eM) * dqa).std())
        rows.append(row)
        log(f"c2 '{name}': {len(sims)} sims in {time.time() - t0:.1f}s; medDQ {row['medDQ_median']:.3f}; "
            f"rule025 power (1,0,D6) {row['rule025_eS1.0_eM0.0_D6']:.3f}; null (0.5,0.5,D6) {row['rule025_eS0.5_eM0.5_D6']:.3f}")
    write_csv(os.path.join(out, "c2_power.csv"), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Part q2 (weighted generator at q = 1/2)
# ----------------------------------------------------------------------------------------------------------------
def part_q2(rng, reps, out):
    rows = []
    k = int(round(0.5 * MODEL["moe"]["N"]))
    for lam in (0.35, 0.5, 1.0):
        for tau in (0.5, 1.0, 2.0, 4.0):
            meds, accs, ovl = [], [], []
            for _ in range(reps):
                world = make_world(rng, "moe", lam, 0.4, 0.5)
                A, B, acc, _ = draw_pairs(rng, world, "c1w", "german", k, 64, tau=tau, max_batches=20)
                accs.append(acc)
                if A is None:
                    continue
                meds.append(np.median(pred_contrast(world, A, B)))
                ovl.append(np.mean([[len(np.intersect1d(A[i, l], B[i, l])) / k for l in range(A.shape[1])]
                                    for i in range(min(8, A.shape[0]))]))
            rows.append(dict(lam=lam, tau=tau, q=0.5, reps=reps, medP_median=float(np.median(meds)) if meds else float("nan"),
                             p_medP_ge_g0=float(np.mean(np.array(meds) >= LOG_G0)) if meds else float("nan"),
                             accept_mean=float(np.mean(accs)), overlap_frac=float(np.mean(ovl)) if ovl else float("nan")))
        log(f"q2 lam={lam} done")
    write_csv(os.path.join(out, "q2_generator.csv"), rows)
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Part budget
# ----------------------------------------------------------------------------------------------------------------
def block_costs(nd=3, c2_draws=6, c1n_draws=3):
    M, Nn, EV, I1, IS = COST["moe_run"], COST["nestor_run"], COST["ev"], COST["int_c1"], COST["int_std"]
    n_c1 = 2 * 2 * 2 * nd
    blocks = [
        ("S0 / G0 pre-adaptation (routing statistics on selection halves and evaluation windows, generator feasibility, "
         "forecast inputs, Nestor local forward check)", "core", 1.50),
        ("S1 / G1 signal gate (S1: 9 MoE + 3 Nestor runs, interim, A/A code/math evals) + audit negative control",
         "core", 9 * M + 3 * Nn + 9 * COST["int_std"] + 2 * 2 * EV + 0.10),
        ("G2: 2 books A/A pairs (4 MoE runs + code/math evals) + 3 SGD calibration runs (interim + full DCLM/target at steps 24, 48)",
         "core", 4 * M + 4 * 2 * EV + 3 * (M + IS + 4 * EV)),
        ("S2 second MoE base (0.5B tokens, seed 43) + routing statistics", "core", 4.40),
        (f"C1 crossover MoE: {n_c1} runs (+ code/math evals, interim on 4 corpora); random-direction (12 x 3 corpora), "
         "pinned routing (12 x 2), WD-only (12 subsets x 2)", "core", n_c1 * (M + 2 * EV + I1) + 12 * 3 * EV + 12 * 2 * EV + 12 * 2 * EV),
        (f"C3 decomposition: Nestor 6 FULL + 1 DCLM-CONT, 6 reverts x 8 FULL; MoE 8 FULL + 1 DCLM-CONT (base 2), "
         "3 reverts x 12 FULL; WD-only 6", "core", 7 * Nn + 8 * 6 * EV + 9 * M + 12 * 3 * EV + 6 * EV),
        (f"C2 optimizer crossover: {c2_draws} draws x 4 runs (interim + full DCLM/target at s*)", "confirmatory-conditional",
         c2_draws * 4 * (M + IS + 2 * EV)),
        (f"C1-N crossover Nestor: {c1n_draws} draws x 2 runs (+ code/math, interim; random-direction x 3, pinned x 2)",
         "secondary", 2 * c1n_draws * (Nn + 2 * EV + I1 + 3 * EV + 2 * EV)),
        ("C5 prospective French: FULL + 2 crossover pairs (5 MoE runs + code/math evals) + 8-step k-step drive", "estimation",
         5 * M + 4 * 2 * EV + 5 * IS + 0.2),
        ("SEC: DCLM LOW/HIGH tilted draws (8 runs) + D4 embedding rows (4 runs, ~0.9x)", "secondary", 8 * M + 4 * M * 0.9),
    ]
    return blocks


def part_budget(out):
    rows = []
    for nd in ND_CHOICES:
        for c2d in (3, 6):
            for c1n in (3, 5):
                blocks = block_costs(nd, c2d, c1n)
                tot = {}
                for name, tier, h in blocks:
                    tot[tier] = tot.get(tier, 0.0) + h
                all_h = sum(tot.values())
                rows.append(dict(nd=nd, c2_draws=c2d, c1n_draws=c1n, **{f"h_{k}": v for k, v in tot.items()},
                                 total_h=all_h, total_h_contingency15=1.15 * all_h, reserve_of_120h=120 - 1.15 * all_h))
    write_csv(os.path.join(out, "budget_v1.csv"), rows)
    write_csv(os.path.join(out, "budget_blocks_v1.csv"),
              [dict(block=n, tier=t, gpu_h=h) for n, t, h in block_costs(3, 6, 3)])
    log("budget done")
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------
def make_figure(out, path="power_curves_v1.png", style=None, panel_letter=None):
    """Six panels; every plotted number is read from the CSVs written by the other parts."""
    import matplotlib.pyplot as plt

    if style is not None:
        style()
    FS = 7
    c1 = {r["scenario"]: r for r in read_csv(os.path.join(out, "c1_power.csv"))}
    c2 = {r["scenario"]: r for r in read_csv(os.path.join(out, "c2_power.csv"))}
    lk = read_csv(os.path.join(out, "lookup_v1.csv"))
    gm = read_csv(os.path.join(out, "g0map.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1))
    ax = axes.ravel()

    # (a) G0 feasibility map
    a = ax[0]
    cols = {0.0: "#c7e9c0", 0.4: "#74c476", 0.7: "#31a354", 0.9: "#006d2c"}
    for rho, col in cols.items():
        rr = sorted([r for r in gm if abs(r["rho_cm"] - rho) < 1e-9], key=lambda r: r["lam"])
        a.plot([r["lam"] for r in rr], [r["medP_median"] for r in rr], "-o", ms=3, lw=1.3, color=col, label=f"{rho:g}")
    a.axhline(LOG_G0, color="0.45", lw=0.8, ls=":")
    a.text(1.0, LOG_G0 + 0.05, "G0 threshold log 1.3", fontsize=FS, color="0.35", ha="right", va="bottom")
    a.set_xlabel("routing specialization λ")
    a.set_ylabel("median predicted contrast P")
    a.set_title("G0: contrast needs specialization", loc="left")
    a.legend(title="code-math corr.", frameon=False, fontsize=FS, title_fontsize=FS, loc="upper left")

    # (b) type-I error under the scientific null
    b = ax[1]
    keys = ["cv_nu 0.25", "baseline", "cv_nu 1.0", "cv_nu 1.0, corpus-specific share 0.5"]
    labs = ["0.25", "0.5", "1.0", "1.0, 50%\nspecific"]
    x = np.arange(len(keys))
    b.plot(x, [c1[k]["fixedpair_plain_pooled_025_b0.0"] for k in keys], "-s", ms=3.5, lw=1.2, color="#d95f02", label="one fixed pair, I > 0")
    b.plot(x, [c1[k]["plain_pooled_025_b0.0_nd3"] for k in keys], "-o", ms=3.5, lw=1.2, color="#7570b3", label="fresh draws, I > 0")
    b.plot(x, [c1[k]["min_rule_025_b0.0_nd3"] for k in keys], "-o", ms=3.5, lw=1.5, color="#1b9e77", label="fresh draws, β > 0.25")
    b.axhline(0.025, color="0.45", lw=0.8, ls=":")
    b.set_xticks(x)
    b.set_xticklabels(labs)
    b.set_xlabel("CV of module sensitivity")
    b.set_ylabel("rejection rate when β = 0")
    b.set_ylim(-0.005, 0.2)
    b.set_title("Type-I error under β = 0", loc="left")
    b.legend(frameon=False, fontsize=FS, loc="upper left")

    # (c) C1 power vs elasticity (confirmatory rule)
    c = ax[2]
    for key, col, lab in (("baseline", "#1b9e77", "baseline"),
                          ("pessimistic: lambda 0.35, rho_cm 0.7, cv_nu 1.0, sigma_run 0.14", "#7570b3", "pessimistic")):
        for nd, ls in ((3, "-"), (5, "--")):
            c.plot(BETAS, [c1[key][f"min_rule_025_b{bb}_nd{nd}"] for bb in BETAS], ls, marker="o", ms=3, lw=1.2, color=col,
                   label=(lab if nd == 3 else None))
    c.axhline(TARGET_POWER, color="0.45", lw=0.8, ls=":")
    c.axvline(BETA_PLAN, color="0.6", lw=0.8, ls="-.")
    c.set_xlabel("elasticity β")
    c.set_ylabel("P(C1 confirmed), α = 0.025")
    c.set_title("C1 power against elasticity", loc="left")
    c.set_ylim(0, 1.03)
    c.legend(frameon=False, fontsize=FS, loc="upper left", bbox_to_anchor=(0.0, 0.72))
    c.text(0.98, 0.05, "solid: 12 draws\ndashed: 20 draws", transform=c.transAxes, fontsize=FS, color="0.35", ha="right")

    # (d) exact sign-flip lookup
    d = ax[3]
    for D, col in ((3, "0.6"), (6, "#fdae6b"), (12, "#e6550d"), (20, "#a63603")):
        rr = sorted([r for r in lk if int(r["D"]) == D and r["dist"] == "normal"], key=lambda r: r["delta"])
        d.plot([r["delta"] for r in rr], [r["power_sf_025"] for r in rr], "-", lw=1.4, color=col, label=f"{D} draws")
    d.axhline(TARGET_POWER, color="0.45", lw=0.8, ls=":")
    d.set_xlabel("standardized effect E[J] / sd(J)")
    d.set_ylabel("power, exact sign-flip, α = 0.025")
    d.set_title("3 draws can never reject", loc="left")
    d.set_ylim(0, 1.03)
    d.legend(frameon=False, fontsize=FS, loc="lower right", bbox_to_anchor=(1.0, 0.05))

    # (e) C2 power vs number of draws
    e = ax[4]
    b2 = c2["baseline"]
    for (eS, eM), col, lab in (((0.5, 0.0), "#9ecae1", "difference 0.5"), ((1.0, 0.0), "#3182bd", "difference 1"),
                               ((2.0, 0.0), "#08519c", "difference 2"), ((0.5, 0.5), "#d95f02", "null, both 0.5")):
        e.plot(C2_D, [b2[f"rule025_eS{eS}_eM{eM}_D{D}"] for D in C2_D], "-o" if eM == 0 else "-s", ms=3, lw=1.3, color=col, label=lab)
    e.axhline(TARGET_POWER, color="0.45", lw=0.8, ls=":")
    e.set_xticks(C2_D)
    e.set_xlabel("C2 draws (3 = one base)")
    e.set_ylabel("P(C2 confirmed), α = 0.025")
    e.set_title("C2 needs at least 6 draws", loc="left")
    e.set_ylim(0, 1.03)
    e.legend(title="SGD − Muon exponent", frameon=False, fontsize=FS, title_fontsize=FS, loc="center left", bbox_to_anchor=(0.0, 0.55))

    # (f) two-stage G2 rule
    f = ax[5]
    sel = ["sigma_run 0.035", "baseline", "sigma_run 0.14", "pessimistic: lambda 0.35, rho_cm 0.7, cv_nu 1.0, sigma_run 0.14"]
    xl = ["σ_run\n0.035", "baseline", "σ_run\n0.14", "pessi-\nmistic"]
    x = np.arange(len(sel))
    f.bar(x - 0.2, [c1[k][f"twostage_rule_025_b{BETA_PLAN}"] for k in sel], width=0.38, color="#1b9e77", label="P(confirmed), β = 0.75")
    f.bar(x + 0.2, [(c1[k][f"twostage_mean_nd_b{BETA_PLAN}"] - 3) / 2 for k in sel], width=0.38, color="#bdbdbd", label="share choosing 5 draws")
    f.axhline(TARGET_POWER, color="0.45", lw=0.8, ls=":")
    f.set_xticks(x)
    f.set_xticklabels(xl, fontsize=6)
    f.set_ylim(0, 1.36)
    f.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    f.set_ylabel("probability")
    f.set_title("G2 rule: power and cost", loc="left")
    f.legend(frameon=False, fontsize=FS, loc="upper left", ncol=1)

    for i, axx in enumerate(ax):
        if panel_letter is not None:
            panel_letter(axx, "abcdef"[i])
    fig.tight_layout(w_pad=1.0, h_pad=1.3)
    fig.savefig(path, dpi=300)
    return fig


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=["lookup", "g0map", "c1", "c1n", "c2", "q2", "budget", "popularity", "c2match", "all", "figure"])
    ap.add_argument("--out", default="power_v1_out")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    reps = 60 if a.quick else a.reps
    ss = np.random.SeedSequence(a.seed)
    names = ["lookup", "g0map", "c1", "c1n", "c2", "q2", "popularity", "c2match"]   # appended names keep earlier streams
    rngs = {nm: np.random.default_rng(s) for nm, s in zip(names, ss.spawn(len(names)))}
    with open(os.path.join(a.out, "assumptions_v1.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ASSUMPTIONS, corpus_sd=CORPUS_SD, cv_sel=CV_SEL, cv_est=CV_EST, match_tol=MATCH_TOL, log_g0=LOG_G0,
                       log_c2_tilt=LOG_C2_TILT, sesoi_beta=SESOI_BETA, sesoi_de=SESOI_DE, eq_margin=EQ_MARGIN,
                       ucl_level=UCL_LEVEL, n_aa=N_AA, nd_choices=ND_CHOICES, q=Q_DEFAULT, pools=POOLS, cost=COST,
                       c1_base=C1_BASE, c2_base=C2_BASE, seed=a.seed, reps=reps), fh, indent=1, default=str)
    parts = [a.part] if a.part != "all" else ["lookup", "g0map", "c1", "c1n", "c2", "q2", "popularity", "c2match", "budget"]
    t0 = time.time()
    lk_path = os.path.join(a.out, "lookup_v1.csv")
    for p in parts:
        if p == "lookup":
            part_lookup(rngs["lookup"], 20000 if not a.quick else 2000, a.out)
        elif p == "g0map":
            part_g0map(rngs["g0map"], max(reps // 5, 12), a.out)
        elif p == "c1":
            if not os.path.exists(lk_path):
                part_lookup(rngs["lookup"], 20000 if not a.quick else 2000, a.out)
            part_c1(rngs["c1"], reps, a.out, read_csv(lk_path))
        elif p == "c1n":
            part_c1n(rngs["c1n"], reps, a.out)
        elif p == "c2":
            part_c2(rngs["c2"], reps, a.out)
        elif p == "q2":
            part_q2(rngs["q2"], max(reps // 5, 20), a.out)
        elif p == "popularity":
            part_popularity(rngs["popularity"], max(reps // 5, 20), a.out)
        elif p == "c2match":
            part_c2match(rngs["c2match"], max(reps // 10, 20), a.out)
        elif p == "budget":
            part_budget(a.out)
        elif p == "figure":
            make_figure(a.out)
    log(f"finished {parts} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
