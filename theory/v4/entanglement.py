"""Routing entanglement: feasibility and attainable contrast of the registered pair generator.

Theory note v4 addendum, section 1. Notation: per stratum l (routed layer) and module m, s2_c(l, m) is the
registered statistic (per-token mean squared effective gate) of corpus c; x_c = log s2_c. For the C1 design the
tilt is z = x_code - x_math and the matched corpora are M = (target, dclm).

Statistics (all computed on SELECTION tables, pooled within strata, i.e. after removing per-stratum means):
  R2        squared multiple correlation of z on {x_c : c in M}   ("entanglement")
  rho2_zD   squared correlation of z with x_dclm
  rho2_zT_D squared partial correlation of z with x_target given x_dclm;  1 - R2 = (1 - rho2_zD)(1 - rho2_zT_D)
  xi_G      Gaussian-theory load of the pool generator, (i_f / i_k) * sqrt(R2 / (1 - (1 - v_f) R2));
            exact matching is attainable iff xi_G < 1, i.e. R2 < R2_crit(f, q) = i_k^2 / (i_f^2 + (1 - v_f) i_k^2)
  Pratio_G  (1 - R2) / (1 - (1 - v_f) R2): expected matched contrast / unmatched pool contrast
  xi_LP     load of the LP relaxation in the share domain with the registered RO and pools (exact geometry):
            1 / max{lam : the RO-difference vector (1 - lam) mu is attainable}; feasible iff xi_LP <= 1
Here i_f is the selection intensity of the pool (top npool of N), v_f the within-pool variance factor and
i_k the intensity of choosing k of npool (finite-N values by Monte Carlo).

CLI on regenerated G0 directories (needs the frozen crossover.py and eval_windows.py on sys.path):
  python entanglement.py --g0-dir <P>/g0/moe_b1 --family moe --targets german,books --codes code,pycode \
      --repo <dir with crossover.py> --out ent_moe_b1.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np

LOG_TOL = math.log(1.05)


def log_usage(s2: np.ndarray) -> np.ndarray:
    """log s2 with the floor of crossover._log_s2 (1e-12 x table mean)."""
    floor = 1e-12 * max(float(np.mean(s2)), 1e-300)
    return np.log(np.maximum(s2, floor))


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def entanglement_r2(sel: dict, matched: list[str], code: str = "code", math_: str = "math",
                    tilt: np.ndarray | None = None) -> dict:
    """R2 of the tilt on the matched corpora's log usage, pooled within strata, with its decomposition."""
    z = _center(log_usage(sel[code]) - log_usage(sel[math_]) if tilt is None else tilt).ravel()
    X = np.stack([_center(log_usage(sel[c])).ravel() for c in matched], 1)
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    rss = float(((z - X @ beta) ** 2).sum())
    tss = float((z ** 2).sum())
    out = {"R2": 1.0 - rss / tss, "sd_z": float(z.std()), "beta": beta.tolist(), "matched": list(matched)}
    C = np.corrcoef(np.column_stack([z, X]).T)
    out["corr"] = C.tolist()
    if len(matched) == 2:
        r_zt, r_zd, r_td = C[0, 1], C[0, 2], C[1, 2]
        out["rho2_zD"] = float(r_zd ** 2)
        out["rho2_zT"] = float(r_zt ** 2)
        out["corr_TD"] = float(r_td)
        den = (1 - r_zd ** 2) * (1 - r_td ** 2)
        out["rho2_zT_D"] = float((r_zt - r_zd * r_td) ** 2 / den) if den > 0 else float("nan")
    return out


@lru_cache(maxsize=None)
def intensities(N: int, npool: int, k: int, nmc: int = 400_000, seed: int = 12345) -> dict:
    """Finite-N selection constants for standard normal scores (Monte Carlo, cached).

    i_f : E[mean of the top npool of N]          (pool intensity)
    v_f : E[population variance of those values]  (within-pool variance factor)
    i_k : E[mean of the top k of npool] (iid normals; within-pool compensation intensity)"""
    rng = np.random.default_rng(seed)
    out = {"i_f": 0.0, "v_f": 0.0, "i_k": 0.0}
    done, chunk = 0, 50_000
    while done < nmc:
        m = min(chunk, nmc - done)
        a = np.sort(rng.standard_normal((m, N)), axis=1)[:, N - npool:]
        out["i_f"] += float(a.mean(1).sum())
        out["v_f"] += float(a.var(1).sum())
        b = np.sort(rng.standard_normal((m, npool)), axis=1)[:, npool - k:]
        out["i_k"] += float(b.mean(1).sum())
        done += m
    return {kk: v / nmc for kk, v in out.items()}


def gauss_theory(R2: float, N: int, f: float, q: float) -> dict:
    npool, k = int(round(f * N)), int(round(q * N))
    c = intensities(N, npool, k)
    i_f, v_f, i_k = c["i_f"], c["v_f"], c["i_k"]
    R2 = min(max(R2, 0.0), 0.999999)
    xi = (i_f / i_k) * math.sqrt(R2 / (1 - (1 - v_f) * R2))
    return {"xi_G": xi, "Pratio_G": (1 - R2) / (1 - (1 - v_f) * R2),
            "R2_crit": i_k ** 2 / (i_f ** 2 + (1 - v_f) * i_k ** 2), **c, "npool": npool, "k": k}


def pools(tilt: np.ndarray, f: float):
    """(A-pool, B-pool) index arrays [L, npool] exactly as crossover.reg_propose / reg_swap_search."""
    L, N = tilt.shape
    npool = int(round(f * N))
    order = np.argsort(tilt, axis=1, kind="stable")
    return order[:, :npool], order[:, N - npool:]


def lp_load(sel: dict, tilt: np.ndarray, k: int, f: float, matched: list[str]) -> dict:
    """xi_LP: drag / capacity of the LP relaxation (share domain, registered global-sum RO)."""
    from scipy.optimize import linprog

    L, N = tilt.shape
    pa, pb = pools(tilt, f)
    npool = pa.shape[1]
    W = {c: sel[c] / sel[c].sum() for c in matched}
    nv = 2 * L * npool + 1
    A_eq, b_eq = [], []
    for l in range(L):
        row = np.zeros(nv); row[l * npool:(l + 1) * npool] = 1; A_eq.append(row); b_eq.append(k)
        row = np.zeros(nv); row[L * npool + l * npool:L * npool + (l + 1) * npool] = 1; A_eq.append(row); b_eq.append(k)
    mu = {}
    for c in matched:
        wa = np.take_along_axis(W[c], pa, 1).ravel()
        wb = np.take_along_axis(W[c], pb, 1).ravel()
        m = (k / npool) * (wb.sum() - wa.sum())
        ro_bar = (k / npool) * 0.5 * (wb.sum() + wa.sum())
        mu[c] = m / ro_bar                     # relative drag of random pool subsets (~ log RO_B/RO_A)
        row = np.concatenate([-wa / ro_bar, wb / ro_bar, [mu[c]]])
        A_eq.append(row); b_eq.append(mu[c])
    if max(abs(v) for v in mu.values()) < 1e-12:
        return {"xi_LP": 0.0, "lam": float("inf"), "mu": mu}
    cost = np.zeros(nv); cost[-1] = -1.0
    bounds = [(0, 1)] * (nv - 1) + [(0, 1e4)]
    r = linprog(cost, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if r.status != 0:
        return {"xi_LP": float("nan"), "lam": float("nan"), "mu": mu, "lp_status": int(r.status)}
    lam = float(-r.fun)
    return {"xi_LP": (1.0 / lam) if lam > 0 else float("inf"), "lam": lam, "mu": mu}


def pool_sign_r2(sel: dict, matched: list[str], f: float, k: int, code: str = "code", math_: str = "math",
                 tilt: np.ndarray | None = None) -> dict:
    """PRIMARY entanglement statistic (share domain, robust to zero-usage modules).

    Over the pool modules of every stratum, pi = -1 (A-pool) / +1 (B-pool) and y_c = registered RO weight
    s2_c / sum(s2_c) for every matched corpus, both centred within stratum and pool-set. R2_pi is the pooled OLS R2
    of pi on y. For a binary score the Gaussian load becomes xi = sqrt(R2_pi / (1 - R2_pi)) / i_k (i_f = 1, v = 0);
    here i_k is replaced by the EMPIRICAL capacity along the optimal direction u = Sigma_w^-1 dmu, giving xi_emp."""
    if tilt is None:
        tilt = log_usage(sel[code]) - log_usage(sel[math_])
    pa, pb = pools(tilt, f)
    L, npool = pa.shape
    Y = np.stack([sel[c] / sel[c].sum() for c in matched], -1)                    # [L, N, C]
    ya, yb = np.take_along_axis(Y, pa[..., None], 1), np.take_along_axis(Y, pb[..., None], 1)
    yy = np.concatenate([ya, yb], 1)                                               # [L, 2 npool, C]
    pi = np.concatenate([-np.ones((L, npool)), np.ones((L, npool))], 1)
    yc = (yy - yy.mean(1, keepdims=True)).reshape(-1, len(matched))
    pc = (pi - pi.mean(1, keepdims=True)).ravel()
    beta, *_ = np.linalg.lstsq(yc, pc, rcond=None)
    R2 = 1.0 - float(((pc - yc @ beta) ** 2).sum()) / float((pc ** 2).sum())
    out = {"R2_pi": R2}
    if len(matched) == 2:
        C = np.corrcoef(np.column_stack([pc, yc]).T)
        r_pt, r_pd, r_td = C[0, 1], C[0, 2], C[1, 2]
        den = (1 - r_pd ** 2) * (1 - r_td ** 2)
        out.update(rho2_piD=float(r_pd ** 2), rho2_piT=float(r_pt ** 2), corr_TD_share=float(r_td),
                   rho2_piT_D=float((r_pt - r_pd * r_td) ** 2 / den) if den > 0 else float("nan"))
    # empirical drag / capacity along the Mahalanobis direction (sums, registered global-sum geometry)
    dmu = (k / npool) * (yb.sum((0, 1)) - ya.sum((0, 1)))
    wa = ya - ya.mean(1, keepdims=True)
    wb = yb - yb.mean(1, keepdims=True)
    Sw = (np.einsum("lnc,lnd->cd", wa, wa) + np.einsum("lnc,lnd->cd", wb, wb)) / (2 * L * npool)
    try:
        u = np.linalg.solve(Sw, dmu)
    except np.linalg.LinAlgError:
        u = np.linalg.lstsq(Sw, dmu, rcond=None)[0]
    sa, sb = ya @ u, yb @ u                                                         # [L, npool]
    cap = (np.sort(sa, 1)[:, npool - k:].sum() - (k / npool) * sa.sum()) + ((k / npool) * sb.sum() - np.sort(sb, 1)[:, :k].sum())
    out["xi_emp"] = float(u @ dmu / cap) if cap > 0 else float("inf")
    return out


def residual_tilt(sel: dict, matched: list[str], code: str = "code", math_: str = "math") -> np.ndarray:
    """EXPLORATORY (not registered): the code-vs-math tilt with the matched corpora's log usage regressed out
    (pooled within-stratum OLS). Pools built on it carry no drag in expectation (section 1.4 of the addendum)."""
    z = _center(log_usage(sel[code]) - log_usage(sel[math_]))
    X_ = np.stack([_center(log_usage(sel[c])).ravel() for c in matched], 1)
    beta, *_ = np.linalg.lstsq(X_, z.ravel(), rcond=None)
    return (z.ravel() - X_ @ beta).reshape(z.shape)


def cell_stats(sel: dict, target: str, *, N: int, f: float, q: float, code: str = "code", math_: str = "math",
               dclm: str = "dclm") -> dict:
    """All entanglement statistics of one C1 cell (selection tables)."""
    matched = [target, dclm]
    tilt = log_usage(sel[code]) - log_usage(sel[math_])        # == crossover.reg_tilt(sel, 'c1')
    ent = entanglement_r2(sel, matched, code=code, math_=math_)
    th = gauss_theory(ent["R2"], N, f, q)
    k = int(round(q * N))
    lp = lp_load(sel, tilt, k, f, matched)
    ps = pool_sign_r2(sel, matched, f, k, code=code, math_=math_)
    return {**ent, **th, **ps, "xi_LP": lp["xi_LP"], "lp_mu": lp["mu"], "f": f, "q": q}


def design_stats(sel: dict, design: str, *, N: int, f: float, q: float, target: str = "german", code: str = "code",
                 dclm_only: bool = False) -> dict:
    """Load statistics for any registered design (tilt and matched set exactly as crossover.reg_tilt / reg_matched):
    c1 (tilt code-vs-math, match target + dclm), c2 (tilt target usage, match dclm [+ code + math]),
    dclm_tilt (tilt dclm usage, match target + code + math)."""
    if design in ("c1", "c5"):
        tilt, matched = log_usage(sel[code]) - log_usage(sel["math"]), [target, "dclm"]
    elif design == "c2":
        tilt, matched = log_usage(sel[target]), (["dclm"] if dclm_only else ["dclm", code, "math"])
    elif design == "dclm_tilt":
        tilt, matched = log_usage(sel["dclm"]), [target, code, "math"]
    else:
        raise ValueError(design)
    k = int(round(q * N))
    lp = lp_load(sel, tilt, k, f, matched)
    ps = pool_sign_r2(sel, matched, f, k, tilt=tilt)
    return {"design": design, "matched": matched, "f": f, "xi_LP": lp["xi_LP"], "R2_pi": ps["R2_pi"], "xi_emp": ps["xi_emp"],
            **{kk: ps[kk] for kk in ("rho2_piD", "rho2_piT_D", "corr_TD_share") if kk in ps}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-dir", required=True)
    ap.add_argument("--family", default="moe")
    ap.add_argument("--targets", default="german,books")
    ap.add_argument("--codes", default="code,pycode")
    ap.add_argument("--fs", default="0.5,0.333333333333")
    ap.add_argument("--repo", default=None, help="directory holding the frozen crossover.py and eval_windows.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.repo:
        sys.path.insert(0, a.repo)
    import crossover as X                                      # frozen G0 code (sha d3fcf6c2... in g0_code_freeze)

    t = X.reg_load_dir(a.g0_dir, a.family)
    rows, other = [], []
    base_name = os.path.basename(os.path.normpath(a.g0_dir))
    fs = [float(v) for v in a.fs.split(",")]
    for code in a.codes.split(","):
        for target in a.targets.split(","):
            if target not in t.sel or code not in t.sel:
                continue
            for f in fs:
                s = cell_stats(t.sel, target, N=t.N, f=f, q=X.REG_Q, code=code)
                rows.append({"cell": base_name + "|" + target + "|" + code, **s, "call": load_call(s["xi_LP"])})
    if all(c in t.sel for c in ("german", "dclm", "code", "math")):                  # C2 and DCLM-tilt rows of G0-2/G0-3
        for design, kw in (("c2", {"dclm_only": True}), ("c2", {"dclm_only": False}), ("dclm_tilt", {})):
            for f in fs:
                s = design_stats(t.sel, design, N=t.N, f=f, q=X.REG_Q, target="german", **kw)
                other.append({"cell": base_name + "|" + design + ("|dclm_only" if kw.get("dclm_only") else ""), **s,
                              "call": load_call(s["xi_LP"])})
    json.dump({"g0_dir": a.g0_dir, "family": a.family, "L": t.L, "N": t.N, "thresholds": THRESHOLDS, "rows": rows,
               "other_designs": other}, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['cell']:28s} f={r['f']:.3f} xi_LP={r['xi_LP']:.2f} xi_emp={r['xi_emp']:.2f} R2_pi={r['R2_pi']:.3f} "
              f"rho2_piD={r.get('rho2_piD', float('nan')):.3f} rho2_piT.D={r.get('rho2_piT_D', float('nan')):.3f} "
              f"corr_TD={r.get('corr_TD_share', float('nan')):.2f} | log-domain R2={r['R2']:.3f} -> {r['call']}")
    for r in other:
        print(f"{r['cell']:28s} f={r['f']:.3f} xi_LP={r['xi_LP']:.2f} R2_pi={r['R2_pi']:.3f} -> {r['call']}")


# Toy-calibrated reading of the LP load for the registered search at f = 1/2, tol = log 1.05 (addendum sec. 1.5):
THRESHOLDS = {"existence": 1.0, "g0_2_operational": 0.67, "always_pass_below": 0.4}


def load_call(xi: float) -> str:
    if not np.isfinite(xi):
        return "undefined"
    if xi > THRESHOLDS["existence"]:
        return "infeasible (no exact matching even fractionally)"
    if xi < THRESHOLDS["always_pass_below"]:
        return "feasible (toy: always passes G0-2)"
    if xi < THRESHOLDS["g0_2_operational"]:
        return "likely feasible (below operational threshold)"
    return "likely fails G0-2 (search cannot reach the feasible corner)"


if __name__ == "__main__":
    main()
