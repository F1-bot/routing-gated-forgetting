"""Gate-second-moment routing overlap (RO), cross-fitting, and constrained random subset pairs.

Module statistic (primary): u_c(m) = E_{tokens of corpus c}[g_m^2], the per-token mean of the squared
EFFECTIVE gate of module m (eval_windows.py definitions; 0 for tokens not routed to m). Secondary:
selection frequency ('freq') and mean gate ('g'). RO_c(S) = sum_{m in S} u_c(m) (unweighted; optional
parameter-count weights); RO shares divide by the family total.

Cross-fitting: the windows of every corpus are split into two seeded halves (eval_windows.split_halves).
Subsets are SELECTED with half 0 and their RO is ESTIMATED with half 1, so selection noise does not
inflate the reported contrasts (winner's curse).

Pair generator (C1 crossover, C2 tilt, DCLM tilts): per stratum (one routed layer) k modules for A and k
for B (disjoint), drawn with a tilted random proposal and repaired by random swaps until the matching
constraints hold on the selection half:
  crossover(low=code, high=math): A = low code / high math usage, B = the reverse; predicted
      I = [log RO_code(B) - log RO_code(A)] - [log RO_math(B) - log RO_math(A)]  (> 0 by design)
  tilt(corpus=german): A = high, B = low usage of that corpus; contrast = log RO(A) - log RO(B)
  match=[...]: |log RO_c(A) - log RO_c(B)| <= tol for every matched corpus c
Draws must reach min_contrast and differ from each other (Jaccard <= max_jaccard for A and for B), so
each draw is an independent replicate (the unit of replication of C1), not a re-use of one fixed set.

REGISTERED generator (prereg v1.0 sec. 4.5; reg_* functions, section "registered generator"): pool proposals,
global-sum RO, rejection (512 x <= 200) + power_sim_v1.swap_search fallback, seeds 5000/6000/6500/6800 + d:
  python crossover.py registered-draw --g0-dir g0/moe_b1 --family moe --design c1 --target german --f 0.5 --d 1,2,3 --out m.json
  python crossover.py registered-aa --g0-dir g0/moe_b1 --family moe --target german --a 1 --seed-base <S> --out aa.json
  python crossover.py kappa --g0-dir g0/moe_b1 --family moe            # G0-4, q = 1/8
The tilted generator below (draw_pairs) is kept for the phase-3 G0 probe directions only.

Effect statistics (after the runs; eval_windows result dirs with <variant>__<corpus>.npz):
  python crossover.py interaction --base <base eval dir> --pair d0=<A dir>,<B dir> ... [--low code --high math]
  python crossover.py optimizer --base <dir> --sgd d0=<A>,<B> ... --muon d0=<A>,<B> ... [--corpus dclm]
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

EPS = 1e-12


# ============================================================================= module tables
def _experts_per_column(routing: dict, layer_key: str) -> int | None:
    if not layer_key.startswith("priv:"):
        return None
    b = layer_key.split(":")[1]
    col = routing.get(f"column:{b}")
    a = routing[layer_key].shape[-1]
    if col is None:
        raise ValueError(f"{layer_key}: no column router summary to infer experts per column")
    return a // col.shape[-1]


def corpus_table(res: dict, family: str, half: np.ndarray | None = None, which: int | None = None,
                 block: int | None = None) -> dict[str, dict]:
    """{layer_key: {'keys', 'tokens', 'freq', 'g', 'g2'}} from one eval_windows result (+ half mask)."""

    import eval_windows as EW

    rt = res["routing"]
    block = block or int(res["meta"].get("set", {}).get("block", 4096))
    sel = np.ones(len(res["ce_sum"]), dtype=bool) if half is None else (np.asarray(half) == which)
    ntok = float(sel.sum()) * block
    out = {}
    for lk, arr in rt.items():
        if not lk.startswith(family + ":"):
            continue
        s = arr[sel].sum(0)                      # [3, A]
        keys = EW.module_keys(lk, arr.shape[-1], _experts_per_column(rt, lk))
        out[lk] = {"keys": keys, "tokens": ntok, "freq": s[0] / ntok, "g": s[1] / ntok, "g2": s[2] / ntok}
    if not out:
        raise ValueError(f"no routing summaries of family {family!r} in result")
    return out


class FamilyTable:
    """Per-stratum module statistics u[corpus][half][stratum] for one routed family."""

    def __init__(self, family: str, measure: str = "g2"):
        self.family, self.measure = family, measure
        self.strata: list[str] = []
        self.keys: list[list[str]] = []
        self.u: dict[str, dict[int, list[np.ndarray]]] = defaultdict(dict)
        self.weights: list[np.ndarray] | None = None

    def add_corpus(self, corpus: str, res: dict, half_seed: int = 0):
        import eval_windows as EW

        half = EW.split_halves(len(res["ce_sum"]), seed=half_seed)
        for h in (0, 1):
            tab = corpus_table(res, self.family, half, h)
            lks = sorted(tab, key=_layer_sort_key)
            if not self.strata:
                self.strata, self.keys = lks, [tab[k]["keys"] for k in lks]
            elif lks != self.strata:
                raise ValueError(f"{corpus}: strata differ from the first corpus")
            self.u[corpus][h] = [np.asarray(tab[k][self.measure], dtype=np.float64) for k in lks]
        self.u[corpus]["all"] = [0.5 * (a + b) for a, b in zip(self.u[corpus][0], self.u[corpus][1])]
        return self

    def set_param_weights(self, unit_params: dict[str, int]):
        self.weights = [np.array([unit_params[k] for k in ks], dtype=np.float64) for ks in self.keys]

    def ro(self, corpus: str, half, sel: list[np.ndarray]) -> float:
        """sum over strata of sum_{m in S} w_m u(m); `sel` = per-stratum index arrays."""

        tot = 0.0
        for s, idx in enumerate(sel):
            v = self.u[corpus][half][s][idx]
            tot += float((v * self.weights[s][idx]).sum() if self.weights is not None else v.sum())
        return tot

    def ro_share(self, corpus: str, half, sel) -> float:
        full = [np.arange(len(k)) for k in self.keys]
        return self.ro(corpus, half, sel) / max(self.ro(corpus, half, full), EPS)


def _layer_sort_key(lk: str):
    parts = lk.split(":")[1:]
    return tuple(int(p[1:]) for p in parts)


def load_tables(files: dict[str, str], family: str, measure: str = "g2", half_seed: int = 0) -> FamilyTable:
    """FamilyTable from {corpus: eval_windows npz with --routing}."""

    import eval_windows as EW

    t = FamilyTable(family, measure)
    for corpus, path in files.items():
        t.add_corpus(corpus, EW.load_result(path), half_seed)
    return t


# ============================================================================= contrasts
def _log(x: float) -> float:
    return math.log(max(x, EPS))


def contrast_value(t: FamilyTable, contrast: dict, half, A, B) -> float:
    if contrast["type"] == "crossover":
        lo, hi = contrast["low"], contrast["high"]
        return (_log(t.ro(lo, half, B)) - _log(t.ro(lo, half, A))) - (_log(t.ro(hi, half, B)) - _log(t.ro(hi, half, A)))
    if contrast["type"] == "tilt":
        c = contrast["corpus"]
        return _log(t.ro(c, half, A)) - _log(t.ro(c, half, B))
    raise ValueError(contrast)


def match_errors(t: FamilyTable, match: list[str], half, A, B) -> dict[str, float]:
    return {c: _log(t.ro(c, half, A)) - _log(t.ro(c, half, B)) for c in match}


def _scores(t: FamilyTable, contrast: dict, half) -> list[np.ndarray]:
    out = []
    for s in range(len(t.strata)):
        if contrast["type"] == "crossover":
            v = np.log(t.u[contrast["high"]][half][s] + EPS) - np.log(t.u[contrast["low"]][half][s] + EPS)
        else:
            v = np.log(t.u[contrast["corpus"]][half][s] + EPS)
        sd = v.std()
        out.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))
    return out


def _violation(t, contrast, match, tol, min_contrast, half, A, B) -> tuple[float, float, dict]:
    c = contrast_value(t, contrast, half, A, B)
    e = match_errors(t, match, half, A, B)
    v = sum(max(0.0, abs(x) - tol) for x in e.values()) + max(0.0, min_contrast - c)
    return v, c, e


def _weighted_pick(rng, cand: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    p = w / w.sum()
    return np.sort(rng.choice(cand, size=k, replace=False, p=p))


def _one_draw(t, contrast, match, k, tol, min_contrast, sel_half, rng, lam, repair_iters, maximize=False):
    z = _scores(t, contrast, sel_half)
    A, B = [], []
    for s, zs in enumerate(z):
        n = len(zs)
        if 2 * k > n:
            raise ValueError(f"stratum {t.strata[s]}: 2k = {2 * k} > {n} modules")
        a = _weighted_pick(rng, np.arange(n), np.exp(lam * zs), k)
        rest = np.setdiff1d(np.arange(n), a)
        b = _weighted_pick(rng, rest, np.exp(-lam * zs[rest]), k)
        A.append(a); B.append(b)
    v, c, e = _violation(t, contrast, match, tol, min_contrast, sel_half, A, B)
    for it in range(repair_iters):
        if v == 0.0 and not maximize:
            break
        s = int(rng.integers(len(A)))
        n = len(z[s])
        X, Y = (A, B) if rng.random() < 0.5 else (B, A)
        free = np.setdiff1d(np.arange(n), np.concatenate([A[s], B[s]]))
        old = X[s].copy()
        i = int(rng.integers(k))
        if len(free) and rng.random() < 0.8:
            X[s] = np.sort(np.concatenate([np.delete(X[s], i), [int(rng.choice(free))]]))
            oldY = None
        else:                                    # swap one module between A and B
            j = int(rng.integers(k))
            oldY = Y[s].copy()
            xi, yj = X[s][i], Y[s][j]
            X[s] = np.sort(np.concatenate([np.delete(X[s], i), [yj]]))
            Y[s] = np.sort(np.concatenate([np.delete(Y[s], j), [xi]]))
        v2, c2, e2 = _violation(t, contrast, match, tol, min_contrast, sel_half, A, B)
        better = v2 < v or (v2 == v and (c2 >= c if maximize else rng.random() < 0.5))
        if better:
            v, c, e = v2, c2, e2
        else:
            X[s] = old
            if oldY is not None:
                Y[s] = oldY
    return A, B, v, c, e


def _flat(t: FamilyTable, sel) -> list[str]:
    return [t.keys[s][i] for s, idx in enumerate(sel) for i in idx]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def spec_for(keys: list[str]) -> dict:
    """subset_adapt 'train' spec training exactly these units.

    Expert keys (moe/pub/priv ...:e<i>) become per-layer expert lists; column keys (column:b<i>:c<j>)
    become per-block column lists (a selected column trains all its units in every layer of the block).
    """

    groups: dict[tuple, list[int]] = defaultdict(list)
    for key in keys:
        parts = key.split(":")
        kind = parts[0]
        d = {p[0]: int(p[1:]) for p in parts[1:]}
        if kind == "column":
            groups[(kind, d["b"], None, None)].append(d["c"])
        else:
            groups[(kind, d.get("b"), d.get("l"), d.get("c"))].append(d["e"])
    sel = []
    order = lambda kv: (kv[0][0],) + tuple(-1 if x is None else x for x in kv[0][1:])
    for (kind, b, l, c), idx in sorted(groups.items(), key=order):
        if kind == "column":
            sel.append({"kind": "column", "block": b, "column": sorted(idx)})
            continue
        f = {"kind": kind, "block": b, "layer": l, "expert": sorted(idx)}
        if c is not None:
            f["column"] = c
        sel.append(f)
    return {"mode": "train", "select": sel}


def _describe(t, contrast, match, A, B):
    corpora = sorted(t.u)
    out = {}
    for h, name in ((0, "selection_half"), (1, "estimation_half")):
        out[name] = {"contrast": contrast_value(t, contrast, h, A, B),
                     "match_err": match_errors(t, match, h, A, B),
                     "ro_share": {c: [t.ro_share(c, h, A), t.ro_share(c, h, B)] for c in corpora}}
    return out


def draw_pairs(t: FamilyTable, *, k: int, contrast: dict, match: list[str], tol: float = 0.05,
               n_draws: int = 3, seed: int = 0, lam: float = 1.5, min_contrast: float = math.log(1.3),
               max_jaccard: float = 0.5, max_tries: int = 300, repair_iters: int = 400,
               best_search: int = 40) -> dict[str, Any]:
    """Constrained random subset pairs (A, B) + the best feasible contrast found by a search."""

    rng = np.random.default_rng(seed)
    draws, tries, feasible = [], 0, 0
    while len(draws) < n_draws and tries < max_tries:
        tries += 1
        A, B, v, c, e = _one_draw(t, contrast, match, k, tol, min_contrast, 0, rng, lam, repair_iters)
        if v > 0:
            continue
        feasible += 1
        fa, fb = _flat(t, A), _flat(t, B)
        if any(_jaccard(fa, d["A"]) > max_jaccard or _jaccard(fb, d["B"]) > max_jaccard for d in draws):
            continue
        draws.append({"A": fa, "B": fb, "spec_A": spec_for(fa), "spec_B": spec_for(fb), **_describe(t, contrast, match, A, B)})
    for i, d in enumerate(draws):
        d["jaccard_to_other_draws"] = {"A": [round(_jaccard(d["A"], o["A"]), 3) for j, o in enumerate(draws) if j != i],
                                       "B": [round(_jaccard(d["B"], o["B"]), 3) for j, o in enumerate(draws) if j != i]}
    best = None
    brng = np.random.default_rng(seed + 7919)
    for _ in range(best_search):                 # strongly tilted proposals, then contrast-maximizing swaps
        A, B, v, c, e = _one_draw(t, contrast, match, k, tol, -1e9, 0, brng, 4.0, repair_iters, maximize=True)
        if v == 0 and (best is None or c > best["selection_contrast"]):
            best = {"selection_contrast": c, "A": _flat(t, A), "B": _flat(t, B), **_describe(t, contrast, match, A, B)}
    return {"family": t.family, "measure": t.measure, "k_per_stratum": k, "strata": len(t.strata), "contrast": contrast,
            "match": match, "tol": tol, "min_contrast": min_contrast, "max_jaccard": max_jaccard, "lam": lam, "seed": seed,
            "attempts": tries, "feasible_attempts": feasible, "draws": draws, "n_draws_found": len(draws),
            "best_feasible": best,
            "best_estimation_contrast": None if best is None else best["estimation_half"]["contrast"]}


def stop_rule(best_by_arch: dict[str, float | None], threshold: float = math.log(1.3)) -> dict:
    """G0 STOP/PIVOT: stop when the best feasible predicted |I| < threshold in every architecture."""

    vals = {a: (abs(v) if v is not None else 0.0) for a, v in best_by_arch.items()}
    stop = all(v < threshold for v in vals.values())
    return {"decision": "STOP_PIVOT" if stop else "GO", "threshold": threshold, "best_abs_I": vals,
            "rule": "STOP/PIVOT if the best feasible crossover contrast predicts |I| < log 1.3 in both architectures"}


# ============================================================================= effect statistics (C1 / C2)
def forgetting(base: dict, after: dict) -> float:
    """F = paired per-window CE increase (nats/token) of `after` over `base` on the same windows."""

    import eval_windows as EW

    return EW.paired(base, after)["diff"]


def signflip_p(values: list[float]) -> float:
    """Exact one-sided sign-flip p-value of mean(values) > 0 (all 2^n flips; n <= 20)."""

    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0 or n > 20:
        return float("nan")
    obs = v.mean()
    signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1
    means = (signs * v[None, :]).mean(1)
    return float((means >= obs - 1e-15).mean())


def _res(d: str, variant: str, corpus: str) -> dict:
    import eval_windows as EW

    return EW.load_result(os.path.join(d, f"{variant}__{corpus}.npz"))


def c1_interaction(base_dir: str, pairs: dict[str, tuple[str, str]], low: str = "code", high: str = "math",
                   variant: str = "ckpt") -> dict:
    """C1: I_d = R_low - R_high with R_c = log F_c(B) - log F_c(A) per draw d (A = low-`low`/high-`high`
    usage, B = the reverse); the law predicts I > 0 (predicted |I| = the routing contrast at beta = 1)."""

    rows = {}
    for d, (da, db) in pairs.items():
        r = {}
        for c in (low, high):
            base = _res(base_dir, variant, c)
            fa, fb = forgetting(base, _res(da, variant, c)), forgetting(base, _res(db, variant, c))
            r[c] = {"F_A": fa, "F_B": fb, "R": math.log(fb) - math.log(fa) if fa > 0 and fb > 0 else float("nan")}
        rows[d] = {"I": r[low]["R"] - r[high]["R"], **r}
    vals = [x["I"] for x in rows.values() if math.isfinite(x["I"])]
    return {"statistic": "I = [log F_low(B) - log F_low(A)] - [log F_high(B) - log F_high(A)]", "low": low, "high": high,
            "draws": rows, "n": len(vals), "mean_I": float(np.mean(vals)) if vals else float("nan"),
            "p_one_sided_signflip": signflip_p(vals), "n_nonpositive_F": len(rows) - len(vals)}


def c2_interaction(base_dir: str, sgd: dict[str, tuple[str, str]], muon: dict[str, tuple[str, str]],
                   corpus: str = "dclm", variant: str = "ckpt") -> dict:
    """C2: I_opt,d = R_SGD - R_Muon with R = log F_old(A) - log F_old(B) (A = high, B = low target usage, matched
    on old-data usage); the law predicts R_SGD > 0 (mu_Q-dependence) and R_Muon ~ 0, i.e. I_opt > 0."""

    base = _res(base_dir, variant, corpus)
    rows = {}
    for d in sorted(set(sgd) & set(muon)):
        rr = {}
        for arm, pairs in (("sgd", sgd), ("muon", muon)):
            fa, fb = forgetting(base, _res(pairs[d][0], variant, corpus)), forgetting(base, _res(pairs[d][1], variant, corpus))
            rr[arm] = {"F_A": fa, "F_B": fb, "R": math.log(fa) - math.log(fb) if fa > 0 and fb > 0 else float("nan")}
        rows[d] = {"I_opt": rr["sgd"]["R"] - rr["muon"]["R"], **rr}
    vals = [x["I_opt"] for x in rows.values() if math.isfinite(x["I_opt"])]
    return {"statistic": "I_opt = R_SGD - R_Muon, R = log F(A) - log F(B)", "corpus": corpus, "draws": rows,
            "n": len(vals), "mean_I_opt": float(np.mean(vals)) if vals else float("nan"),
            "p_one_sided_signflip": signflip_p(vals)}


def _pairs(items: list[str]) -> dict[str, tuple[str, str]]:
    out = {}
    for it in items or []:
        d, _, ab = it.partition("=")
        a, _, b = ab.partition(",")
        out[d] = (a, b)
    return out


def _effect_cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="C1 / C2 effect statistics from eval_windows result directories")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("interaction"); i.add_argument("--base", required=True)
    i.add_argument("--pair", action="append", required=True, help="draw=<A eval dir>,<B eval dir>")
    i.add_argument("--low", default="code"); i.add_argument("--high", default="math")
    o = sub.add_parser("optimizer"); o.add_argument("--base", required=True); o.add_argument("--corpus", default="dclm")
    o.add_argument("--sgd", action="append", required=True); o.add_argument("--muon", action="append", required=True)
    for p_ in (i, o):
        p_.add_argument("--variant", default="ckpt"); p_.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "interaction":
        rep = c1_interaction(a.base, _pairs(a.pair), a.low, a.high, a.variant)
    else:
        rep = c2_interaction(a.base, _pairs(a.sgd), _pairs(a.muon), a.corpus, a.variant)
    txt = json.dumps(rep, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0


# ============================================================================= CLI
# ============================================================================= registered generator (prereg v1.0 sec. 4.5)
# Everything below implements the PRE-REGISTERED pool generator. The tilted generator above (draw_pairs,
# _one_draw) is kept unchanged for the phase-3 G0 probe directions; it is NOT the registered algorithm.
#   statistic   s2_c(m) = per-token mean of g_eff(m)^2 (eval_windows 'g2'), selection vs estimation windows
#   RO          RO_c(S) = sum_{m in S} s2_c(m) / sum_{m in E} s2_c(m)  (E = the stratified expert family)
#   strata      per layer, k = q N: MoE N = 24, k = 6; NESTOR private experts of ROUTED blocks N = 64, k = 16
#   tilt z      C1/C5: log s2_code - log s2_math; C2: log s2_target; DCLM-tilt: log s2_dclm   (selection stats)
#   pools       per layer: lowest f N by z -> A-pool, highest f N -> B-pool, f in {1/2, 1/3}
#   draw        per layer A takes k uniformly from its pool, B k from its pool; accept iff for every matched
#               corpus |log RO_c(A) - log RO_c(B)| <= log 1.05 (selection stats); rejection in batches of 512
#               (<= 200 batches), then power_sim_v1.swap_search from fresh random initialisations
#   seeds       5000 + d (C1), 6000 + d (C2), 6500 + d (DCLM-tilt), 6800 + d (C5); draw d = first accepted pair
#   predicted   P = [log RO_code(B) - log RO_code(A)] - [log RO_math(B) - log RO_math(A)]  (C1/C5)
#               dQ = log RO_T(H) - log RO_T(L), H = B, L = A (C2);  P_P = log RO_dclm(HIGH=B) - log RO_dclm(LOW=A)
#               all on ESTIMATION statistics (the evaluation windows at theta_0; cross-fitting)
# With every layer's s2 normalised to the same total, RO here equals power_sim_v1.ro (layer mean of shares),
# which is what the equivalence tests check (identical RNG consumption, identical subsets).
REG_TOL = math.log(1.05)                 # sec. 4.5 item 6 / G0-2
REG_TOL_RELAXED = math.log(1.10)         # sec. 6 G0-2 action
REG_POOLS = (1 / 2, 1 / 3)               # sec. 4.5 item 5
REG_Q = 1 / 4                            # sec. 4.5 item 3
REG_BATCH = 512                          # sec. 4.5 item 6
REG_MAX_BATCHES = 200
REG_SWAP_ITERS = 800                     # power_sim_v1.swap_search default / G0-2 "within 800 iterations"
REG_SWAP_ROUNDS = 4                      # power_sim_v1.draw_pairs fallback rounds
REG_SEED_BASE = {"c1": 5000, "c2": 6000, "dclm_tilt": 6500, "c5": 6800}
REG_TAUS = (1.0, 2.0, 4.0)               # sec. 4.5 item 9 (q = 1/2 fallback, G1-2 only)
REG_LOG_G0 = math.log(1.3)               # G0-1
REG_LOG_C2 = math.log(1.5)               # G0-3
REG_KAPPA_Q = 1 / 8                      # G0-4
REG_KAPPA_MIN = 1.25


class RegTables:
    """Registered statistic s2_c(m) of one stratified expert family.

    sel[c], est[c]: [L, N] arrays (strata x modules) of selection / estimation statistics; keys[l][m] are
    the unit keys (eval_windows.module_keys) used for subset_adapt specs; sources records where every table
    came from (file, windows, layout)."""

    def __init__(self, family: str, strata: list[str], keys: list[list[str]], sel: dict, est: dict,
                 sources: dict, excluded: list[str] | None = None):
        self.family, self.strata, self.keys = family, strata, keys
        self.sel, self.est, self.sources = sel, est, sources
        self.excluded = excluded or []

    @property
    def L(self) -> int:
        return len(self.strata)

    @property
    def N(self) -> int:
        return len(self.keys[0])

    def subset_keys(self, S: np.ndarray) -> list[str]:
        S = np.asarray(S)
        return [self.keys[l][int(m)] for l in range(S.shape[0]) for m in S[l]]


def reg_strata(routing: dict, family: str, block: int) -> tuple[list[str], list[str]]:
    """(kept, excluded) layer keys of the registered stratification. For NESTOR private experts only ROUTED
    blocks are kept: a block whose column router selects every column for every token (the dense tail,
    K_c = C) is excluded (sec. 4.5 item 3)."""

    lks = sorted((k for k in routing if k.startswith(family + ":")), key=_layer_sort_key)
    if not lks:
        raise ValueError(f"no routing summaries of family {family!r}")
    if family != "priv":
        return lks, []
    keep, drop = [], []
    for lk in lks:
        b = lk.split(":")[1]
        col = routing.get(f"column:{b}")
        if col is None:
            raise ValueError(f"{lk}: no column router summary (needed to detect the dense tail)")
        n_win, n_col = col.shape[0], col.shape[-1]
        per_tok = float(col[:, 0].sum()) / (n_win * block)          # column selections per token = K_c
        (drop if abs(per_tok - n_col) < 1e-6 else keep).append(lk)
    return keep, drop


def reg_s2(res: dict, strata: list[str], mask: np.ndarray | None = None) -> np.ndarray:
    """[L, N] per-token mean squared effective gate over the (masked) windows of one eval_windows result."""

    rt = res["routing"]
    block = int(res["meta"].get("set", {}).get("block", 4096))
    sel = np.ones(len(res["ce_sum"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not sel.any():
        raise ValueError("empty window mask")
    ntok = float(sel.sum()) * block
    rows = [np.asarray(rt[lk][sel, 2].sum(0), dtype=np.float64) / ntok for lk in strata]
    if len({len(r) for r in rows}) != 1:
        raise ValueError("strata have different module counts")
    return np.stack(rows)


def _res_block(res: dict) -> int:
    return int(res["meta"].get("set", {}).get("block", 4096))


def reg_load_dir(g0_dir: str, family: str, corpora: list[str] | None = None, layout: str = "auto",
                 half_seed: int = 0) -> RegTables:
    """RegTables from a g0_probe output directory.

    layout 'registered' : routing__<c>.npz holds the SELECTION windows (all used) and routing__<c>_eval.npz the
                          EVALUATION windows (prereg sec. 4.3); predicted contrasts use the latter.
    layout 'crossfit'   : (phase-3 configs) one stream per corpus; selection = half 0, estimation = half 1 of
                          eval_windows.split_halves(n, half_seed) -- a recorded DEVIATION from sec. 4.3.
    layout 'auto'       : per corpus, 'registered' if routing__<c>_eval.npz exists, else 'crossfit'."""

    import eval_windows as EW

    if corpora is None:
        corpora = sorted(f[len("routing__"):-len(".npz")] for f in os.listdir(g0_dir)
                         if f.startswith("routing__") and f.endswith(".npz") and not f.endswith("_eval.npz"))
    if not corpora:
        raise ValueError(f"{g0_dir}: no routing__<corpus>.npz files")
    sel, est, sources = {}, {}, {}
    strata = keys = excluded = None
    for c in corpora:
        f_sel = os.path.join(g0_dir, f"routing__{c}.npz")
        f_est = os.path.join(g0_dir, f"routing__{c}_eval.npz")
        mode = layout if layout != "auto" else ("registered" if os.path.exists(f_est) else "crossfit")
        r_sel = EW.load_result(f_sel)
        if strata is None:
            strata, excluded = reg_strata(r_sel["routing"], family, _res_block(r_sel))
            keys = [EW.module_keys(lk, r_sel["routing"][lk].shape[-1], _experts_per_column(r_sel["routing"], lk))
                    for lk in strata]
        else:
            s2_, ex2 = reg_strata(r_sel["routing"], family, _res_block(r_sel))
            if s2_ != strata or ex2 != excluded:
                raise ValueError(f"{c}: strata differ from the first corpus")
        n = len(r_sel["ce_sum"])
        if mode == "registered":
            r_est = EW.load_result(f_est)
            sel[c], est[c] = reg_s2(r_sel, strata), reg_s2(r_est, strata)
            sources[c] = {"layout": "registered",
                          "selection": {"file": f_sel, "windows": n, "set": r_sel["meta"].get("set"),
                                        "checkpoint": r_sel["meta"].get("checkpoint")},
                          "estimation": {"file": f_est, "windows": int(len(r_est["ce_sum"])), "set": r_est["meta"].get("set"),
                                         "checkpoint": r_est["meta"].get("checkpoint")}}
        elif mode == "crossfit":
            half = EW.split_halves(n, seed=half_seed)
            sel[c], est[c] = reg_s2(r_sel, strata, half == 0), reg_s2(r_sel, strata, half == 1)
            ck = r_sel["meta"].get("checkpoint")
            sources[c] = {"layout": "crossfit", "selection": {"file": f_sel, "windows": int((half == 0).sum()), "half": 0, "checkpoint": ck},
                          "estimation": {"file": f_sel, "windows": int((half == 1).sum()), "half": 1, "checkpoint": ck},
                          "half_seed": half_seed, "set": r_sel["meta"].get("set")}
        else:
            raise ValueError(layout)
    return RegTables(family, strata, keys, sel, est, sources, excluded)


def reg_k(t: RegTables, q: float = REG_Q) -> int:
    return int(round(q * t.N))


def _log_s2(x: np.ndarray) -> np.ndarray:
    """log s2 with a floor for modules never routed in the selection windows (1e-12 x the table mean)."""

    floor = 1e-12 * max(float(np.mean(x)), 1e-300)
    return np.log(np.maximum(x, floor))


def reg_tilt(sel: dict, design: str, *, target: str | None = None, code: str = "code") -> np.ndarray:
    """Tilt score z(m) on selection statistics (sec. 4.5 item 4)."""

    if design in ("c1", "c5"):
        return _log_s2(sel[code]) - _log_s2(sel["math"])
    if design == "c2":
        return _log_s2(sel[target])
    if design == "dclm_tilt":
        return _log_s2(sel["dclm"])
    raise ValueError(design)


def reg_matched(design: str, *, target: str | None = None, code: str = "code", dclm_only: bool = False) -> list[str]:
    """Matched corpora (sec. 4.5 item 6); C2 with dclm_only = the G0-2 fallback (matching on DCLM only)."""

    if design in ("c1", "c5"):
        return [target, "dclm"]
    if design == "c2":
        return ["dclm"] if dclm_only else ["dclm", code, "math"]
    if design == "dclm_tilt":
        return [target or "german", code, "math"]
    raise ValueError(design)


def reg_ro(s2: np.ndarray, S: np.ndarray) -> np.ndarray:
    """RO(S) = sum_{m in S} s2(m) / sum_E s2 for S of shape [M, L, k] (-> [M]) or [L, k] (-> scalar)."""

    S = np.asarray(S)
    one = S.ndim == 2
    S3 = S[None] if one else S
    L = s2.shape[0]
    v = s2[np.arange(L)[None, :, None], S3].sum((1, 2)) / s2.sum()
    return v[0] if one else v


def reg_propose(rng, tilt: np.ndarray, k: int, f: float, M: int):
    """Pool proposals (== power_sim_v1.propose, identical RNG consumption)."""

    L, N = tilt.shape
    npool = int(round(f * N))
    if npool < k:
        raise ValueError("pool smaller than subset")
    order = np.argsort(tilt, axis=1, kind="stable")
    lo, hi = order[:, :npool], order[:, N - npool:]
    ka = np.argsort(rng.random((M, L, npool)), axis=2)[:, :, :k]
    kb = np.argsort(rng.random((M, L, npool)), axis=2)[:, :, :k]
    A = np.take_along_axis(np.broadcast_to(lo[None], (M, L, npool)), ka, 2)
    B = np.take_along_axis(np.broadcast_to(hi[None], (M, L, npool)), kb, 2)
    return A, B


def reg_propose_random(rng, L: int, N: int, k: int, M: int):
    """A/A proposals: disjoint uniformly random k-subsets per layer (== power_sim_v1.propose_random)."""

    perm = np.argsort(rng.random((M, L, N)), axis=2)
    return perm[:, :, :k], perm[:, :, k:2 * k]


def reg_propose_weighted(rng, tilt: np.ndarray, k: int, tau: float, M: int):
    """q = 1/2 fallback (G1-2 only): Gumbel-top-k with weights exp(-+tau z_std) (== power_sim_v1.propose_weighted)."""

    zt = (tilt - tilt.mean(1, keepdims=True)) / (tilt.std(1, keepdims=True) + 1e-12)
    L, N = tilt.shape
    ga = -tau * zt[None] - np.log(-np.log(rng.random((M, L, N))))
    gb = +tau * zt[None] - np.log(-np.log(rng.random((M, L, N))))
    return np.argsort(-ga, axis=2)[:, :, :k], np.argsort(-gb, axis=2)[:, :, :k]


def reg_match_errors(sel: dict, A: np.ndarray, B: np.ndarray, corpora: list[str]) -> np.ndarray:
    """[len(corpora), M] log RO_c(A) - log RO_c(B) on the given (selection) statistics."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([np.log(reg_ro(sel[c], A)) - np.log(reg_ro(sel[c], B)) for c in corpora])


def reg_match_mask(sel: dict, A, B, corpora: list[str], tol: float = REG_TOL) -> np.ndarray:
    err = reg_match_errors(sel, A, B, corpora)
    with np.errstate(invalid="ignore"):
        return (np.abs(err) <= tol).all(0)


def reg_swap_search(rng, sel: dict, tilt: np.ndarray, k: int, f: float, M: int, corpora: list[str],
                    tol: float = REG_TOL, iters: int = REG_SWAP_ITERS):
    """power_sim_v1.swap_search with the registered RO (family-total normalisation): M chains initialised with
    random k-subsets of the A- and B-pools per layer, then randomized greedy swaps (member <-> non-member of the
    same pool and layer, alternating A and B) accepted only if they reduce max_c |log RO_c(A) - log RO_c(B)|.
    Returns (A [n_conv, L, k], B, converged fraction, n_conv)."""

    L, N = tilt.shape
    npool = int(round(f * N))
    order = np.argsort(tilt, axis=1, kind="stable")
    pools = (order[:, :npool], order[:, N - npool:])
    Wm = [sel[c] / sel[c].sum() for c in corpora]
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
            ro_[side, ci] = W[np.arange(L)[None, :, None], idx].sum((1, 2))

    def obj(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(np.log(r[0]) - np.log(r[1])).max(0)

    cur = obj(ro_)
    ar = np.arange(M)
    if npool > k:
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
                new[side, ci] = ro_[side, ci] + (W[l, m_out] - W[l, m_in])
            nobj = obj(new)
            acc = (nobj < cur) & (cur > tol)
            sets[side][0][ar[acc], l[acc], i[acc]] = pout[acc]
            sets[side][1][ar[acc], l[acc], j[acc]] = pin[acc]
            ro_[:, :, acc] = new[:, :, acc]
            cur = np.where(acc, nobj, cur)
    ok = cur <= tol
    return module_idx(0, sets[0][0][ok]), module_idx(1, sets[1][0][ok]), float(ok.mean()), int(ok.sum())


def reg_pairs(sel: dict, tilt: np.ndarray | None, k: int, f: float | None, n_need: int, seed: int | None,
              corpora: list[str], tol: float = REG_TOL, *, M: int = REG_BATCH, max_batches: int = REG_MAX_BATCHES,
              swap: bool = True, swap_rounds: int = REG_SWAP_ROUNDS, iters: int = REG_SWAP_ITERS,
              tau: float | None = None, aa: bool = False, rng=None) -> dict:
    """Registered search (sec. 4.5 item 6): rejection batches of M proposals (<= max_batches) until n_need pairs
    are accepted; if too few, swap_search rounds with 2 (n_need - got) + 8 fresh chains (<= swap_rounds).
    Proposals: pools (default), weighted (tau given; q = 1/2 fallback, no swap search) or A/A (aa=True: disjoint
    uniform subsets of the whole layer, no swap search). Returns the first n_need pairs in generation order
    plus the G0-2 statistics (acceptance = accepted / proposed over all rejection batches; swap convergence =
    converged / chains over all swap rounds, None if no swap search ran)."""

    rng = np.random.default_rng(seed) if rng is None else rng
    L, N = next(iter(sel.values())).shape
    As, Bs, got, nb, n_prop, n_acc = [], [], 0, 0, 0, 0
    while got < n_need and nb < max_batches:
        nb += 1
        if aa:
            A, B = reg_propose_random(rng, L, N, k, M)
        elif tau is not None:
            A, B = reg_propose_weighted(rng, tilt, k, tau, M)
        else:
            A, B = reg_propose(rng, tilt, k, f, M)
        ok = reg_match_mask(sel, A, B, corpora, tol)
        n_prop += M
        n_acc += int(ok.sum())
        if ok.any():
            As.append(A[ok])
            Bs.append(B[ok])
            got += int(ok.sum())
    n_rej = got
    chains = conv = rounds = 0
    if got < n_need and swap and not aa and tau is None:
        for _ in range(swap_rounds):
            m = 2 * (n_need - got) + 8
            A, B, _, nc = reg_swap_search(rng, sel, tilt, k, f, m, corpora, tol, iters)
            rounds += 1
            chains += m
            conv += nc
            if A.shape[0]:
                As.append(A)
                Bs.append(B)
                got += A.shape[0]
            if got >= n_need:
                break
    A = np.concatenate(As)[:n_need] if As else np.zeros((0, L, k), dtype=np.int64)
    B = np.concatenate(Bs)[:n_need] if Bs else np.zeros((0, L, k), dtype=np.int64)
    return {"A": A, "B": B, "n": int(A.shape[0]), "complete": bool(A.shape[0] >= n_need), "seed": seed,
            "acceptance": n_acc / n_prop if n_prop else float("nan"), "proposals": n_prop, "batches": nb,
            "accepted_by_rejection": n_rej, "swap_rounds": rounds, "swap_chains": chains, "swap_converged": conv,
            "swap_convergence": (conv / chains) if chains else None, "from_swap": int(max(0, got - n_rej)),
            "tol": tol, "matched": list(corpora), "f": f, "k": k, "tau": tau, "aa": aa}


def reg_predicted(est: dict, A: np.ndarray, B: np.ndarray, design: str, *, target: str | None = None,
                  code: str = "code") -> np.ndarray:
    """Predicted contrasts on ESTIMATION statistics: P (C1/C5), Delta_Q (C2, H = B, L = A), P_P (DCLM-tilt)."""

    def lr(c, X):
        with np.errstate(divide="ignore"):
            return np.log(reg_ro(est[c], X))

    if design in ("c1", "c5"):
        return (lr(code, B) - lr(code, A)) - (lr("math", B) - lr("math", A))
    if design == "c2":
        return lr(target, B) - lr(target, A)
    if design == "dclm_tilt":
        return lr("dclm", B) - lr("dclm", A)
    raise ValueError(design)


def reg_choose_tau(t: RegTables, k: int, *, target: str, code: str = "code", n: int = 1000, seed: int | None = None,
                   tol: float = REG_TOL) -> dict:
    """q = 1/2 fallback: the smallest tau in {1, 2, 4} whose median predicted contrast P >= log 1.3."""

    tilt = reg_tilt(t.sel, "c1", code=code)
    rows = {}
    for tau in REG_TAUS:
        r = reg_pairs(t.sel, tilt, k, None, n, REG_SEED_BASE["c1"] if seed is None else seed,
                      reg_matched("c1", target=target), tol, tau=tau)
        med = float(np.median(reg_predicted(t.est, r["A"], r["B"], "c1", code=code))) if r["n"] else float("-inf")
        rows[tau] = {"median_P": med, "acceptance": r["acceptance"], "n": r["n"]}
        if med >= REG_LOG_G0:
            return {"tau": tau, "rows": rows}
    return {"tau": None, "rows": rows}


def reg_kappa(t: RegTables, corpus: str = "dclm", q: float = REG_KAPPA_Q) -> dict:
    """G0-4: per stratum LOW-q / HIGH-q = the ceil(qN) modules with the lowest / highest SELECTION usage of the
    corpus; RO of both on ESTIMATION statistics. kappa_X = RO(X) / E[RO(RAND-q)] with E[RO(RAND-q)] = k / N exactly
    (uniform random k-subsets per stratum); kappa_H / kappa_L = RO(HIGH) / RO(LOW)."""

    k = int(math.ceil(q * t.N - 1e-9))
    order = np.argsort(t.sel[corpus], axis=1, kind="stable")
    low, high = order[:, :k], order[:, t.N - k:]
    ro_l, ro_h = float(reg_ro(t.est[corpus], low)), float(reg_ro(t.est[corpus], high))
    rand = k / t.N
    return {"q": q, "k": k, "N": t.N, "L": t.L, "ro_low": ro_l, "ro_high": ro_h, "ro_rand_expected": rand,
            "kappa_L": ro_l / rand, "kappa_H": ro_h / rand,
            "ratio": ro_h / ro_l if ro_l > 0 else float("inf"), "low": low.tolist(), "high": high.tolist()}


def reg_draw(t: RegTables, design: str, d: int, *, f: float, target: str | None = None, code: str = "code",
             k: int | None = None, tol: float = REG_TOL, dclm_only: bool = False, M: int = REG_BATCH,
             max_batches: int = REG_MAX_BATCHES) -> dict:
    """Registered draw d of a design: the first accepted pair of the generator seeded REG_SEED_BASE[design] + d,
    with the manifest fields of sec. 4.5 item 8 (subsets, seed, matched values, predicted contrast)."""

    k = reg_k(t) if k is None else int(k)
    seed = REG_SEED_BASE[design] + int(d)
    tilt = reg_tilt(t.sel, design, target=target, code=code)
    corpora = reg_matched(design, target=target, code=code, dclm_only=dclm_only)
    r = reg_pairs(t.sel, tilt, k, f, 1, seed, corpora, tol, M=M, max_batches=max_batches)
    entry = {"design": design, "d": int(d), "seed": seed, "f": f, "k": k, "q": k / t.N, "N": t.N, "L": t.L,
             "family": t.family, "strata": t.strata, "excluded_strata": t.excluded, "target": target, "code": code,
             "tol": tol, "matched_corpora": corpora,
             "generator": {kk: r[kk] for kk in ("acceptance", "proposals", "batches", "swap_rounds", "swap_chains",
                                                "swap_converged", "swap_convergence", "from_swap")}}
    if not r["n"]:
        entry["status"] = "infeasible"
        return entry
    A, B = r["A"][0], r["B"][0]
    err = reg_match_errors(t.sel, A[None], B[None], corpora)[:, 0]
    name = {"c1": "P", "c5": "P", "c2": "Delta_Q", "dclm_tilt": "P_P"}[design]
    pred = float(reg_predicted(t.est, A[None], B[None], design, target=target, code=code)[0])
    ka, kb = t.subset_keys(A), t.subset_keys(B)
    entry.update(status="ok", A=A.tolist(), B=B.tolist(),
                 matched_values={c: {"log_RO_A": float(np.log(reg_ro(t.sel[c], A))), "log_RO_B": float(np.log(reg_ro(t.sel[c], B))),
                                     "diff": float(e)} for c, e in zip(corpora, err)},
                 predicted={name: pred}, subsets={"A": ka, "B": kb}, spec_A=spec_for(ka), spec_B=spec_for(kb),
                 sources=t.sources)
    return entry


def reg_aa_draw(t: RegTables, a: int, *, target: str, seed_base: int, k: int | None = None, tol: float = REG_TOL,
                M: int = REG_BATCH, max_batches: int = REG_MAX_BATCHES) -> dict:
    """A/A pair (sec. 4.5 item 7): disjoint uniformly random k-subsets per layer, matched on target and DCLM.
    The prereg registers no A/A seed; seed_base is a parameter (see g0_reconciliation.md)."""

    k = reg_k(t) if k is None else int(k)
    corpora = [target, "dclm"]
    r = reg_pairs(t.sel, None, k, None, 1, seed_base + int(a), corpora, tol, M=M, max_batches=max_batches, aa=True)
    entry = {"design": "aa", "a": int(a), "seed": seed_base + int(a), "k": k, "target": target, "tol": tol,
             "matched_corpora": corpora, "generator": {"acceptance": r["acceptance"], "batches": r["batches"]}}
    if not r["n"]:
        entry["status"] = "infeasible"
        return entry
    A, B = r["A"][0], r["B"][0]
    ka, kb = t.subset_keys(A), t.subset_keys(B)
    entry.update(status="ok", A=A.tolist(), B=B.tolist(), subsets={"A": ka, "B": kb}, spec_A=spec_for(ka), spec_B=spec_for(kb))
    return entry


def write_manifest(path: str, entries: list[dict], extra: dict | None = None) -> str:
    """Manifest of draws (sec. 4.5 item 8) as canonical JSON; returns its sha256 (also written to <path>.sha256)."""

    import hashlib

    doc = {"prereg": "v1.0 sec. 4.5", "entries": entries, **(extra or {})}
    txt = json.dumps(doc, sort_keys=True, indent=1, default=float)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{h}  {os.path.basename(path)}\n")
    return h


def _registered_cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="registered subset generator (prereg v1.0 sec. 4.5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dr = sub.add_parser("registered-draw", help="draws d of one design -> hashed manifest")
    dr.add_argument("--g0-dir", required=True)
    dr.add_argument("--family", required=True, choices=["moe", "priv"])
    dr.add_argument("--design", required=True, choices=sorted(REG_SEED_BASE))
    dr.add_argument("--target", default=None)
    dr.add_argument("--code", default="code")
    dr.add_argument("--f", type=float, required=True, help="pool fraction fixed at G0 (0.5 or 0.3333333333333333)")
    dr.add_argument("--d", required=True, help="draw indices, e.g. 1,2,3 (d >= 1; d = 0 is the G0 distribution run)")
    dr.add_argument("--k", type=int, default=None)
    dr.add_argument("--tol", type=float, default=REG_TOL)
    dr.add_argument("--dclm-only", action="store_true", help="C2 G0-2 fallback (matching on DCLM only)")
    dr.add_argument("--layout", default="auto", choices=["auto", "registered", "crossfit"])
    dr.add_argument("--half-seed", type=int, default=0)
    dr.add_argument("--out", required=True)
    aa = sub.add_parser("registered-aa", help="A/A pairs -> hashed manifest")
    aa.add_argument("--g0-dir", required=True)
    aa.add_argument("--family", required=True, choices=["moe", "priv"])
    aa.add_argument("--target", required=True)
    aa.add_argument("--a", required=True, help="pair indices, e.g. 1,2,3")
    aa.add_argument("--seed-base", type=int, required=True)
    aa.add_argument("--k", type=int, default=None)
    aa.add_argument("--layout", default="auto", choices=["auto", "registered", "crossfit"])
    aa.add_argument("--half-seed", type=int, default=0)
    aa.add_argument("--out", required=True)
    kp = sub.add_parser("kappa", help="G0-4 kappa at q = 1/8")
    kp.add_argument("--g0-dir", required=True)
    kp.add_argument("--family", required=True, choices=["moe", "priv"])
    kp.add_argument("--q", type=float, default=REG_KAPPA_Q)
    kp.add_argument("--layout", default="auto", choices=["auto", "registered", "crossfit"])
    kp.add_argument("--half-seed", type=int, default=0)
    a = ap.parse_args(argv)
    t = reg_load_dir(a.g0_dir, a.family, layout=a.layout, half_seed=a.half_seed)
    if a.cmd == "kappa":
        r = reg_kappa(t, q=a.q)
        print(json.dumps({k: v for k, v in r.items() if k not in ("low", "high")}, indent=1))
        return 0
    if a.cmd == "registered-draw":
        ents = [reg_draw(t, a.design, int(d), f=a.f, target=a.target, code=a.code, k=a.k, tol=a.tol,
                         dclm_only=a.dclm_only) for d in a.d.split(",") if d]
    else:
        ents = [reg_aa_draw(t, int(i), target=a.target, seed_base=a.seed_base, k=a.k) for i in a.a.split(",") if i]
    h = write_manifest(a.out, ents)
    print(json.dumps({"manifest": a.out, "sha256": h, "entries": [{kk: e.get(kk) for kk in ("design", "d", "a", "seed", "status", "predicted")}
                                                                  for e in ents]}, default=float))
    return 0


def main(argv=None) -> int:
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("interaction", "optimizer"):
        return _effect_cli(argv)
    if argv and argv[0] in ("registered-draw", "registered-aa", "kappa"):
        return _registered_cli(argv)

    ap = argparse.ArgumentParser(description="constrained subset-pair generator (C1 / C2)")
    ap.add_argument("--routing", action="append", required=True, help="corpus=eval_windows routing npz (repeatable)")
    ap.add_argument("--family", required=True, choices=["moe", "priv", "pub", "column"])
    ap.add_argument("--k", type=int, required=True, help="modules per stratum (layer) in each subset")
    ap.add_argument("--crossover", default=None, help="low,high  e.g. code,math (A = low code / high math)")
    ap.add_argument("--tilt", default=None, help="corpus to tilt on (A high, B low)")
    ap.add_argument("--match", default="", help="comma list of corpora to match")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--min-contrast", type=float, default=math.log(1.3))
    ap.add_argument("--max-jaccard", type=float, default=0.5)
    ap.add_argument("--measure", default="g2", choices=["g2", "freq", "g"])
    ap.add_argument("--half-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    files = dict(x.split("=", 1) for x in a.routing)
    t = load_tables(files, a.family, a.measure, a.half_seed)
    if bool(a.crossover) == bool(a.tilt):
        raise SystemExit("give exactly one of --crossover low,high or --tilt corpus")
    contrast = ({"type": "crossover", "low": a.crossover.split(",")[0], "high": a.crossover.split(",")[1]}
                if a.crossover else {"type": "tilt", "corpus": a.tilt})
    rep = draw_pairs(t, k=a.k, contrast=contrast, match=[m for m in a.match.split(",") if m], tol=a.tol,
                     n_draws=a.draws, seed=a.seed, lam=a.lam, min_contrast=a.min_contrast, max_jaccard=a.max_jaccard)
    rep["inputs"] = files
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n_draws_found", "attempts", "feasible_attempts", "best_estimation_contrast")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
