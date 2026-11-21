#!/usr/bin/env python
"""analyze_confirm.py -- confirmatory, secondary and exploratory analyses after the confirm queue.

Implements preregistration v1.3 (version 6) with the v1.2 sections it keeps (secs. 5, 8, 9, 13):
  C1      books-only (v1.3 sec. 3.1; v1.2 sec. 8.1): E5 log / normalized-difference rule, J_d = I_d - beta0_b P_d,
          one-sided exact sign-flip test over all D draws (bases pooled), Holm over {C1, C2 if s* = 8}, the base rule,
          beta-hat / beta-hat_pin / beta-hat_sym / beta-hat_sym,pin / beta-hat_rand by regression through the origin
          with test-inversion 95 % CIs (overall and per base), the decision rule, the three-statistic table, the
          out-of-domain annotations and the interpretation rules; the v1.2 sec. 9 secondary items that use C1 runs.
  C2      (v1.2 sec. 8.2): I2 / J2 at s*, exact sign-flip test, Delta-e, e_SGD, e_Muon, the manipulation check, C2-eq.
  blocks  third base (4.1), dose-response (4.2, decision P-14), assumption audit (4.3), NESTOR-books (4.4), M12-b (4.5),
          selpin decomposition (4.6), EDGE (4.11), C1-DE (4.12), forecast scoring (5.7), planned vs actual GPU-h (9.2).
  (analyze_confirm_ext.py, called from run_all): C3 endpoint revert decomposition with the freeze cross-check
          (v1.2 sec. 10.1), C3 per layer (v1.3 sec. 4.8), C4 (v1.2 sec. 10.2), C5 French (v1.2 secs. 10.3, 12), C2-eq
          b_Muon - b_Muon^rand and the per-expert SNR (v1.2 sec. 8.2 item 7), matched gain (v1.2 sec. 11), F_rel and
          bits-per-byte versions (v1.2 sec. 9 item 13). Routing entanglement (v1.3 sec. 4.9) and the gap trajectory
          (sec. 4.10) are analysed by their own scripts; the output points to them.

Torch-free (numpy; scipy.stats for t quantiles). Deterministic: every sign-flip p-value and test-inversion interval
is computed by exact enumeration of the 2^D sign vectors (no Monte Carlo, so the v1.2 Monte Carlo seed 20261102 is
never used); bootstraps use the registered seeds (20261101 draw bootstraps, v1.2; 20261113 dose-response, v1.3 sec. 4.2;
the assumption audit's window bootstrap is assumption_audit.analyze_realized with its frozen seed).

Inputs (read only; every file read is listed with its SHA-256 in the output):
  FREEZE.json (parameters, the plan / queue / record / forecast hashes; verified first unless --no-verify),
  the confirm plan and queue it names, results/queue_state_confirm.json and the run roots of the confirm queue
  (eval_<name>/<variant>__<set>.npz, interim/stepNNN__<set>[__full].npz, interim/displacement.jsonl,
  trainlog/train_steps.jsonl, subset_manifest.json), the theta_0 evaluation files of the plan, the draw manifests of
  the plan (and the G0 routing tables they name), the G1 / G2 plans, states, records and A/A run roots (dose-response
  P ~ 0 level), results/dose/dose_record.json, audit/<base>/ (assumption_audit run --directions realized), the hashed
  forecast files with results/forecast/forecast_units.json, logs/power/gpu_power_*.csv and README_power_log.txt, the
  run logs logs/queue_<stage>/<run>_m<micro>[c].log (the attempt of the state root) and logs/queue_forecast/queue.log
  (EDGE).

Outputs: <out>/analysis_confirm.json (canonical JSON with an embedded sha256 and a .sha256 sidecar) and
         <out>/analysis_confirm.md (summary tables).

  python analyze_confirm.py run [--root .] [--freeze FREEZE.json] [--out results/analysis] [--no-verify]
  python analyze_confirm.py selftest
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FORMAT = "analysis_confirm/1"
SEED_BOOT = 20261101          # v1.2 analysis seed: draw bootstraps
SEED_MC = 20261102            # v1.2 Monte Carlo seed: not used (every sign-flip test is enumerated exactly)
SEED_DOSE = 20261113          # v1.3 sec. 4.2: stratified dose-response bootstrap
N_BOOT = 10000
ALPHA = 0.05
Z_RES = 3.0                   # E5: resolved if F >= 3 SE
BETA_REF = 0.25               # refutation bound (C1), Delta-e_0 (C2), fixed margin (NESTOR-books, plain secondary)
OOD_RANGE = (0.67, 1.5)       # v1.2 sec. 8.1 item 7
FO_SHARE = 0.2                # resolved first-order share threshold
LOG_TOST = math.log(1.1)      # C1-d TOST margin
C2EQ_MARGIN = 1.0 / 3.0
C2_MANIP = 0.5                # mean D_SGD / Delta_Q(mean) below this -> "uninformative"
E1_BLOCK_SHARE = 0.25         # more than 25 % divergent runs -> the block's confirmatory test is inconclusive
# P-16 (2026-11-15; prereg v1.3 version 8 sec. 3.5): staged analysis. The core analysis runs once when the core marker
# exists, reads only core files (file firewall below) and writes results/analysis_core; the full analysis runs after the
# whole queue with the same code. P-16 revised (2026-11-15 ~03:00): the core is C1 on both MoE bases and C2 (with the
# theta_0 evaluations they need); the third base and the assumption audit are analysed by the full analysis only.
CORE_BLOCKS_DEFAULT = ("C1:moe_b1", "C1:moe_b2", "C2")
CORE_MARKER_DEFAULT = "results/confirm/CORE_DONE.json"
CORE_OUT = os.path.join("results", "analysis_core")
CORE_SECONDARY = ("C4", "matched_gain", "bpb")          # precision 2026-11-15: secondary, descriptive, no decision
CORE_EXPLORATORY = ("C2_eq_rand",)   # addendum 4 (2026-11-15): b_Muon - b_Muon^rand on the step-8 checkpoint, core files
CORE_DEFERRED = ("B3", "audit", "selpin", "C3", "C3_per_layer", "C5_french", "DCLM-CONT", "SEC", "dose", "M12b", "C1_DE",
                 "NESTOR_books", "forecasts", "F_rel")
DEFERRED_STATUS = "deferred to the full analysis after the queue (P-16; prereg v1.3 version 8 sec. 3.5)"
E1_RULE = ("P-17 (a) final (2026-11-15 ~06:20; plan diff MF-D1), amending prereg v1.2 sec. 13 E1 for every "
           "confirmatory run, stopped or not: a run diverges if (i) the training loss is non-finite at any executed step, or "
           "(ii) the EMA(0.1) of the training loss exceeds its step-1 value by more than 0.5 nats at any executed step after "
           "warm-up. The clause 'final training loss >= initial training loss' (stopped-run form: loss at the stop step not "
           "below step 1) is deleted for every run. A training log that ends before its stop step is flagged incomplete, "
           "not divergent")
E1_STOP_RULE = E1_RULE                                 # name, kept for callers


class FirewallError(RuntimeError):
    """P-16 / prereg v1.3 sec. 3.5: the core analysis tried to read a file of a non-core job."""
MAX_ENUM = 24                 # exact enumeration up to 2^24 sign vectors
EDGE_GAP_S = 5.0


# ============================================================================= statistics (exact, deterministic)
def _arr(x) -> np.ndarray:
    return np.asarray([float(v) for v in x], dtype=np.float64)


def signsums(x) -> np.ndarray:
    """sum_i s_i x_i for all 2^D sign vectors s; element 0 is s = (+1, ..., +1)."""
    x = _arr(x)
    if x.size > MAX_ENUM:
        raise ValueError(f"exact enumeration limited to D <= {MAX_ENUM}")
    s = np.zeros(1)
    for v in x:
        s = np.concatenate((s + v, s - v))
    return s


def _tol(*xs) -> float:
    return 1e-12 * max(1.0, float(sum(np.abs(_arr(x)).sum() for x in xs)))


def signflip_p(x, alternative: str = "greater") -> dict | None:
    """Exact sign-flip test of mean(x) (v1.2 sec. 8.1 item 3): p = #{s : mean(s o x) >= mean(x)} / 2^D (one-sided);
    two-sided: #{s : |mean(s o x)| >= |mean(x)|} / 2^D. Ties count (float tolerance 1e-12 x sum|x|)."""
    x = _arr(x)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return None
    S = signsums(x)
    T, tol = S[0], _tol(x)
    if alternative == "greater":
        k = int((S >= T - tol).sum())
    elif alternative == "less":
        k = int((S <= T + tol).sum())
    elif alternative == "two-sided":
        k = int((np.abs(S) >= abs(T) - tol).sum())
    else:
        raise ValueError(alternative)
    n = int(S.size)
    return {"p": k / n, "count": k, "vectors": n, "D": int(x.size), "mean": float(T / x.size),
            "min_p": (1 if alternative != "two-sided" else 2) / n, "alternative": alternative}


def beta_origin(I, P) -> float | None:
    I, P = _arr(I), _arr(P)
    d = float((P * P).sum())
    return float((I * P).sum() / d) if I.size and d > 0 else None


def se_origin(I, P) -> float | None:
    """Residual SE of the through-origin slope (used only for the registered Welch intervals)."""
    I, P = _arr(I), _arr(P)
    b = beta_origin(I, P)
    if b is None or I.size < 2:
        return None
    r = I - b * P
    return float(math.sqrt(float((r * r).sum()) / ((I.size - 1) * float((P * P).sum()))))


def ci_invert(I, P, alpha: float = ALPHA, grid: int = 401) -> dict | None:
    """95 % CI of beta by inverting the two-sided exact sign-flip test of I_d - beta P_d at level alpha (v1.2 sec. 8.1
    item 5): the set {beta : p(beta) > alpha}. The bounds are found by expansion and bisection from beta-hat
    (relative precision 1e-10); a grid scan over three times the interval checks convexity (holes or islands are
    reported, and the reported bounds are then the hull). The minimum two-sided p is 2 / 2^D, so for D <= 5 no beta is
    rejected at 0.05 and the interval is unbounded (reported as such)."""
    I, P = _arr(I), _arr(P)
    D = int(I.size)
    if D == 0 or P.size != D or not (np.all(np.isfinite(I)) and np.all(np.isfinite(P))):
        return None
    bh = beta_origin(I, P)
    if bh is None:
        return None
    SI, SP = signsums(I), signsums(P)
    A, B, n = SI[0], SP[0], SI.size
    aI, aP = float(np.abs(I).sum()), float(np.abs(P).sum())

    def pval(b: float) -> float:
        tol = 1e-12 * max(1.0, aI + abs(b) * aP)
        return int((np.abs(SI - b * SP) >= abs(A - b * B) - tol).sum()) / n

    out = {"beta_hat": bh, "D": D, "alpha": alpha, "p_at_beta_hat": pval(bh), "min_two_sided_p": 2.0 / n,
           "method": "inversion of the two-sided exact sign-flip test of I_d - beta P_d"}
    if 2.0 / n > alpha:
        out.update(lo=-math.inf, hi=math.inf, bounded=False,
                   note=f"D = {D}: minimum two-sided p {2.0 / n:.4g} > {alpha}; no beta is rejected")
        return out
    if out["p_at_beta_hat"] <= alpha:
        out.update(lo=None, hi=None, bounded=False, note="beta-hat itself is rejected; see grid")
    scale = max(abs(bh), float(np.max(np.abs(I)) / max(float(np.max(np.abs(P))), 1e-300)), 1e-6)
    LIMIT = 1e6 * scale

    def edge(direction: int):
        ok, step, b = bh, 1e-3 * scale, None
        while True:
            b = bh + direction * step
            if pval(b) <= alpha:
                break
            ok = b
            step *= 2.0
            if step > LIMIT:
                return direction * math.inf
        rej = b
        for _ in range(200):
            mid = 0.5 * (ok + rej)
            if pval(mid) > alpha:
                ok = mid
            else:
                rej = mid
            if abs(rej - ok) <= 1e-10 * max(1.0, abs(ok)):
                break
        return ok

    if out["p_at_beta_hat"] > alpha:
        lo, hi = edge(-1), edge(+1)
    else:
        lo = hi = None
    # grid check (convexity) over [lo - w, hi + w]
    if lo is not None and math.isfinite(lo) and math.isfinite(hi):
        w = max(hi - lo, 1e-6 * scale)
        gs = np.linspace(lo - w, hi + w, grid)
    else:
        gs = np.linspace(bh - 10 * scale, bh + 10 * scale, grid)
    acc = np.array([pval(float(g)) > alpha for g in gs])
    holes = islands = 0
    if lo is not None and math.isfinite(lo) and math.isfinite(hi):
        inside = (gs >= lo) & (gs <= hi)
        holes = int((inside & ~acc).sum())
        islands = int((~inside & acc).sum())
        if islands:
            lo, hi = min(lo, float(gs[acc].min())), max(hi, float(gs[acc].max()))
    elif acc.any():
        lo, hi = float(gs[acc].min()), float(gs[acc].max())
    out.update(lo=lo, hi=hi, bounded=bool(lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi)),
               grid_points=int(grid), grid_rejected_inside=holes, grid_accepted_outside=islands,
               convex=bool(holes == 0 and islands == 0))
    return out


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values (v1.2 sec. 8.3)."""
    items = sorted(((k, float(p)) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m, run, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def tq(df: float, q: float) -> float:
    from scipy import stats
    return float(stats.t.ppf(q, df))


def t_interval(x, level: float = 0.95) -> dict | None:
    x = _arr(x)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return None
    m = float(x.mean())
    if n < 2:
        return {"n": n, "mean": m, "lo": None, "hi": None, "df": 0, "sd": None, "level": level}
    sd = float(x.std(ddof=1))
    h = tq(n - 1, 0.5 + level / 2) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "df": n - 1, "lo": m - h, "hi": m + h, "level": level}


def welch_diff(b1, se1, n1, b2, se2, n2, level: float = 0.95) -> dict | None:
    """Welch interval for the difference of two through-origin slopes (residual SEs, Welch-Satterthwaite df)."""
    if None in (b1, se1, b2, se2) or n1 < 2 or n2 < 2:
        return None
    v1, v2 = se1 * se1, se2 * se2
    if v1 + v2 <= 0:
        return None
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    h = tq(df, 0.5 + level / 2) * math.sqrt(v1 + v2)
    d = b1 - b2
    return {"diff": d, "lo": d - h, "hi": d + h, "df": df, "se": math.sqrt(v1 + v2), "level": level,
            "method": "Welch: through-origin residual SEs, Welch-Satterthwaite df"}


def ols_hc3(y, X, names: list[str]) -> dict | None:
    """OLS without intercept, HC3 covariance, t-intervals with df n - k (v1.3 sec. 4.2)."""
    from scipy import stats
    y, X = _arr(y), np.asarray(X, dtype=np.float64)
    n, k = X.shape
    if n <= k:
        return None
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < k:
        return None
    Xi = np.linalg.inv(XtX)
    b = Xi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, Xi, X)
    if np.any(h >= 1 - 1e-12):
        return {"coef": dict(zip(names, b.tolist())), "hc3": None, "note": "a unit has leverage 1; HC3 undefined"}
    meat = (X * ((e / (1 - h)) ** 2)[:, None]).T @ X
    V = Xi @ meat @ Xi
    se = np.sqrt(np.diag(V))
    df = n - k
    q = tq(df, 0.975)
    out = {"n": n, "df": df, "coef": {}, "residual_ss": float(e @ e)}
    for i, nm in enumerate(names):
        t = float(b[i] / se[i]) if se[i] > 0 else math.nan
        out["coef"][nm] = {"b": float(b[i]), "se_hc3": float(se[i]), "lo": float(b[i] - q * se[i]),
                           "hi": float(b[i] + q * se[i]), "t": t,
                           "p_two_sided": float(2 * stats.t.sf(abs(t), df)) if math.isfinite(t) else None}
    return out


def pct_ci(v, level: float = 0.95) -> list | None:
    v = _arr(v)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    a = (1 - level) / 2
    return [float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))]


def resolved(F, se) -> bool:
    return bool(F is not None and se is not None and math.isfinite(F) and math.isfinite(se) and F > 0
                and F >= Z_RES * se)


def _fin(x):
    """JSON-safe float (inf / nan -> strings)."""
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else "-inf" if x < 0 else "nan"
    return x


# ============================================================================= read-only access to the result files
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# variant pairs per statistic kind: (theta_0 + delta side, theta_0 - delta side or antithetic partner)
KINDS = {
    "raw": ("ckpt", None), "sym": ("ckpt", "reflect"),
    "pin": ("pinned", None), "sym_pin": ("pinned", "reflect+pinned"),
    "selpin": ("selpin", None), "sym_selpin": ("selpin", "reflect+selpin"),
    "rand": ("randdir:{h}", "randdir-:{h}"),        # antithetic mean (v1.2 sec. 5, control quantities)
    "wd": ("wdonly", None),
}


class Ctx:
    """Read-only access to the registered result files of one project root. Every file read is recorded with its
    SHA-256 (output 'inputs'); a missing planned file is recorded (output 'missing') and never guessed."""

    def __init__(self, root: str, freeze: str = "FREEZE.json", verify: bool = True, mode: str = "full"):
        import gate_common as GC
        self.GC = GC
        self.root = os.path.abspath(root)
        self.mode = mode
        self._core_roots: list[str] | None = None            # set once the plan is read (core mode only)
        self.reads: dict[str, str] = {}
        self.warnings: list[str] = []
        self.missing: list[str] = []
        self._load: dict = {}
        self._pair: dict = {}
        self.freeze_rel = freeze
        self.freeze_path = self.p(freeze)
        self.freeze = self.json(self.freeze_path)
        self.verify = self._verify_freeze() if verify else {"status": "skipped (--no-verify)"}
        par = self.freeze.get("parameters") or {}
        self.params = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in par.items()}
        pref = self.freeze.get("plan") or {}
        self.plan_path = self.p(pref.get("path") or "confirm/plan_confirm.json")
        self.plan = self.json(self.plan_path)
        self.state_path = self.p(self.plan.get("state") or "results/queue_state_confirm.json")
        self.state = self.json(self.state_path) if os.path.exists(self.state_path) else {}
        if not self.state:
            self.warnings.append(f"confirm queue state {self.state_path} missing or empty")
        self.runs = {r["id"]: r for r in self.plan.get("runs", [])}
        p16 = self.plan.get("p16") or {}
        self.core_blocks = tuple(b for b in (p16.get("core_blocks") or CORE_BLOCKS_DEFAULT) if b != "base_eval")
        self.core_ids = {rid for rid, r in self.runs.items() if r.get("block") in self.core_blocks}
        self.core_marker = p16.get("marker") or CORE_MARKER_DEFAULT
        if mode == "core":                                 # the file firewall: confirm outputs of core runs only
            self._core_roots = [os.path.normcase(os.path.join(self.root, "outputs", "confirm", rid)) + os.sep
                                for rid in sorted(self.core_ids)]
            # MF-P3 (2026-11-15): the assumption audit writes outside outputs/confirm (cmd jobs, --out audit/<base>)
            self._core_forbid = [os.path.normcase(os.path.join(self.root, "audit")) + os.sep]
            qp_ = self.p(self.plan.get("queue") or "confirm/queue_confirm.json")
            if os.path.exists(qp_):
                for j in (json.load(open(qp_, encoding="utf-8")).get("jobs") or []):
                    av = j.get("argv") or []
                    if j.get("kind") == "cmd" and "assumption_audit.py" in " ".join(av) and "--out" in av:
                        self._core_forbid.append(os.path.normcase(os.path.abspath(self.p(av[av.index("--out") + 1]))) + os.sep)
        self.code = self.params.get("code_stream") or self.plan.get("code_stream") or "code"
        self.code2 = self.plan.get("code_secondary")
        self.base_files = self.plan.get("base_eval_files") or {}
        self.e1: dict = {}
        self._tables: dict = {}
        self._manifests: dict = {}
        self._gplans: dict = {}

    # ---------------------------------------------------------------- files
    def p(self, path: str | None) -> str | None:
        if not path:
            return None
        path = str(path).replace("/", os.sep).replace("\\", os.sep)
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def rel(self, path: str) -> str:
        try:
            r = os.path.relpath(path, self.root)
            return r.replace(os.sep, "/") if not r.startswith("..") else path
        except ValueError:
            return path

    def _read(self, path: str) -> None:
        if self._core_roots is not None:
            self._firewall(path)
        if path not in self.reads:
            self.reads[path] = sha256_file(path)

    def _firewall(self, path: str) -> None:
        """Core mode: no file under outputs/confirm/ except those of core runs, and no selpin-block evaluation."""
        n = os.path.normcase(os.path.abspath(path))
        if f"{os.sep}eval_selpin{os.sep}" in n:
            raise FirewallError(f"core analysis: {self.rel(path)} is an evaluation of the selpin block (after the core)")
        if any(n.startswith(f) for f in getattr(self, "_core_forbid", [])):
            raise FirewallError(f"core analysis: {self.rel(path)} is a file of the assumption audit (after the core)")
        conf = os.path.normcase(os.path.join(self.root, "outputs", "confirm")) + os.sep
        if n.startswith(conf) and not any(n.startswith(r) for r in self._core_roots):
            raise FirewallError(f"core analysis: {self.rel(path)} is not a file of a core run")

    def json(self, path: str):
        path = self.p(path)
        self._read(path)
        return json.load(open(path, encoding="utf-8-sig"))

    def load(self, path: str) -> dict | None:
        path = self.p(path)
        if path in self._load:
            return self._load[path]
        if not path or not os.path.exists(path):
            self._load[path] = None
            return None
        self._read(path)
        self._load[path] = self.GC.load_eval(path)
        return self._load[path]

    def _verify_freeze(self) -> dict:
        import contextlib
        import io
        try:
            import chain6b_util as CU
        except Exception as e:                                  # pragma: no cover
            return {"status": "not verified", "error": f"chain6b_util not importable: {e}"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CU.freeze_verify(self.root, self.rel(self.freeze_path))
        txt = buf.getvalue().strip().splitlines()
        return {"status": "ok" if rc == 0 else "FAILED", "rc": rc, "log": txt[-30:]}

    # ---------------------------------------------------------------- runs
    def status(self, rid: str) -> str:
        return (self.state.get(rid) or {}).get("status", "pending")

    def root_of(self, rid: str) -> str | None:
        r = (self.state.get(rid) or {}).get("root")
        return self.p(r) if r else None

    def eval_file(self, rid: str, variant: str, set_name: str) -> str | None:
        """<root>/eval_<name>/<variant with ':' -> '-'>__<set>.npz for the evaluation of the plan that lists it."""
        run, root = self.runs.get(rid) or {}, self.root_of(rid)
        if not root:
            return None
        for ev in run.get("evals") or []:
            vs = ev.get("variants", "ckpt")
            vs = vs.split(",") if isinstance(vs, str) else list(vs)
            if variant in vs and set_name in (ev.get("sets") or []):
                return os.path.join(root, f"eval_{ev.get('name', 'main')}", f"{variant.replace(':', '-')}__{set_name}.npz")
        return None

    def haar(self, rid: str) -> str | None:
        for ev in (self.runs.get(rid) or {}).get("evals") or []:
            vs = ev.get("variants", "")
            for v in (vs.split(",") if isinstance(vs, str) else vs):
                if v.startswith("randdir:"):
                    return v.split(":", 1)[1]
        return None

    def base_file(self, base: str, set_name: str) -> str | None:
        return self.p((self.base_files.get(base) or {}).get(set_name))

    def e1_of(self, rid: str) -> dict:
        """E1 (v1.2 sec. 13) from the run's train_steps.jsonl (gate_common.e1_check), cached."""
        if rid in self.e1:
            return self.e1[rid]
        root = self.root_of(rid)
        tl = os.path.join(root, "trainlog", "train_steps.jsonl") if root else None
        if not tl or not os.path.exists(tl):
            r = {"status": "no_train_log", "diverged": None}
        else:
            self._read(tl)
            run = self.runs.get(rid) or {}
            warm = int(run.get("warmup", 10))
            r = self.GC.e1_check(self.GC.read_train_steps(tl), warm)
            stop, steps = int(run.get("stop_after") or 0), int(run.get("steps", 96))
            if r.get("status") == "checked":                # P-17 (a) final (MF-D1): clauses (i) and (ii) only, every run
                fs = int(r.get("final_step") or 0)
                keep = [x for x in (r.get("reasons") or []) if not str(x).startswith("final loss")]
                clauses = ["(i) non-finite training loss at any executed step (steps 1-%d)" % fs]
                if fs > warm:
                    clauses.append(f"(ii) EMA(0.1) more than 0.5 nats above the step-1 loss after warm-up (steps {warm + 1}-{fs})")
                    c2 = "evaluated"
                else:
                    c2 = f"not reached: the log ends at step {fs}, inside the {warm}-step warm-up"
                r = dict(r, reasons=keep, diverged=bool(keep), rule=E1_RULE, clauses_evaluated=clauses, clause_ii=c2,
                         deleted_clause="final training loss >= step-1 loss (P-17 (a) final)")
                if stop and stop < steps:
                    r["stop_step"] = stop
                    if fs != stop:                          # flagged, not divergent (an infrastructure fault is re-run, E2)
                        msg = (f"train log of {rid} ends at step {fs}, not at the stop step {stop}: flagged incomplete "
                               f"(P-17 (a) final)")
                        r = dict(r, status="incomplete_log", diverged=None, reasons=list(r.get("reasons") or []) + [msg])
                        self.warnings.append(msg)
        self.e1[rid] = r
        return r

    def usable(self, rid: str) -> tuple[bool, str | None]:
        if rid not in self.runs:
            return False, "not in the plan"
        st = self.status(rid)
        if st != "done":
            return False, f"status {st}"
        if self.e1_of(rid).get("diverged"):
            return False, "E1 divergence: " + "; ".join(self.e1[rid].get("reasons") or [])
        return True, None

    # ---------------------------------------------------------------- per-window pairing
    def pair(self, base_path: str | None, run_path: str | None, subset: bool = False) -> dict | None:
        """Paired per-window CE (theta_0, run) by grid index with equal token hashes and counts (gate_common rule).
        subset=True pairs the run's windows (a 64-window head) with the matching windows of a full theta_0 file."""
        if not base_path or not run_path:
            return None
        key = (base_path, run_path, subset)
        if key in self._pair:
            return self._pair[key]
        b, r = self.load(base_path), self.load(run_path)
        if b is None or r is None:
            if r is None and run_path:
                self.missing.append(self.rel(run_path))
            if b is None and base_path:
                self.missing.append(self.rel(base_path))
            self._pair[key] = None
            return None
        ib = {int(i): k for k, i in enumerate(b["index"])}
        ir = {int(i): k for k, i in enumerate(r["index"])}
        if subset:
            if not set(ir) <= set(ib):
                raise ValueError(f"{run_path}: windows not in {base_path}")
            common = sorted(ir)
        else:
            if set(ib) != set(ir):
                raise ValueError(f"window grids differ: {base_path} vs {run_path}")
            common = sorted(ib)
        kb, kr = np.array([ib[i] for i in common]), np.array([ir[i] for i in common])
        if not np.array_equal(b["hash"][kb], r["hash"][kr]) or not np.array_equal(b["n_tok"][kb], r["n_tok"][kr]):
            raise ValueError(f"paired windows differ in content: {base_path} vs {run_path}")
        out = {"index": np.array(common, dtype=np.int64),
               "cb": (b["ce_sum"][kb] / b["n_tok"][kb]).astype(np.float64),
               "cr": (r["ce_sum"][kr] / r["n_tok"][kr]).astype(np.float64)}
        self._pair[key] = out
        return out

    # ---------------------------------------------------------------- forgetting statistics
    @staticmethod
    def _stat(d: np.ndarray) -> dict:
        W = int(d.size)
        return {"F": float(d.mean()), "SE": float(d.std(ddof=1) / math.sqrt(W)) if W > 1 else math.nan, "W": W}

    def windows(self, rid: str, set_name: str, kind: str, *, base: str | None = None, files: dict | None = None):
        """Per-window values d_w of one statistic kind (F: CE_run - CE_0; sym / rand: the mean of the two sides;
        odd: half their difference under key 'odd'), or None when a planned file is missing."""
        plus, minus = KINDS[kind]
        h = self.haar(rid) if "{h}" in plus else None
        if "{h}" in plus and h is None:
            return None
        plus, minus = plus.format(h=h), (minus.format(h=h) if minus else None)
        run = self.runs.get(rid) or {}
        bpath = (files or {}).get("base") or self.base_file(base or run.get("base"), set_name)
        fp = (files or {}).get(plus) or self.eval_file(rid, plus, set_name)
        pp = self.pair(bpath, fp)
        if pp is None:
            return None
        if minus is None:
            return {"index": pp["index"], "d": pp["cr"] - pp["cb"]}
        fm = (files or {}).get(minus) or self.eval_file(rid, minus, set_name)
        pm = self.pair(bpath, fm)
        if pm is None:
            return None
        if not np.array_equal(pp["index"], pm["index"]):
            raise ValueError(f"{rid}/{set_name}: the two sides of {kind} have different windows")
        return {"index": pp["index"], "d": 0.5 * ((pp["cr"] - pp["cb"]) + (pm["cr"] - pm["cb"])),
                "odd": 0.5 * (pp["cr"] - pm["cr"])}

    def F(self, rid: str, set_name: str, kind: str = "raw", **kw) -> dict | None:
        w = self.windows(rid, set_name, kind, **kw)
        if w is None:
            return None
        out = self._stat(w["d"])
        if "odd" in w:
            o = self._stat(w["odd"])
            out["F_odd"], out["SE_odd"] = o["F"], o["SE"]
        return out

    # ---------------------------------------------------------------- crossover statistic with the E5 rule
    def ro(self, rid: str, corpus: str) -> float | None:
        ro = ((self.runs.get(rid) or {}).get("subset") or {}).get("ro") or {}
        key = "code" if corpus == self.code else corpus
        v = ro.get(key)
        return float(v) if v is not None else None

    def cross(self, draws: list[dict], kind: str, corpora: tuple, *, Fget=None, ro_get=None, P_key: str = "P") -> dict:
        """I_d for one statistic kind over a test's draws (v1.2 sec. 8.1 item 2): the log statistic if EVERY one of
        the 4D values is resolved (F >= 3 SE), otherwise the normalized difference with F-bar_c = the mean over all
        runs of the (base, target) cell and P_nd from RO in the same way. Draws with a missing value are dropped and
        counted. Both statistics are returned whenever defined."""
        c1, c2 = corpora
        Fget = Fget or (lambda rid, c: self.F(rid, c, kind))
        ro_get = ro_get or self.ro
        rows, dropped = [], []
        for dr in draws:
            v, why = {}, None
            for s in ("A", "B"):
                for c in corpora:
                    f = Fget(dr[s], c)
                    if f is None:
                        why = f"{dr[s]}: {kind} {c} not available"
                        break
                    v[(s, c)] = f
                if why:
                    break
            if why:
                dropped.append({"base": dr.get("base"), "d": dr.get("d"), "reason": why})
            else:
                rows.append((dr, v))
        allv = [x for _, v in rows for x in v.values()]
        use_log = bool(rows) and all(resolved(x["F"], x["SE"]) for x in allv)
        log_def = bool(rows) and all(x["F"] > 0 for x in allv)
        cells = defaultdict(lambda: defaultdict(list))
        rocell = defaultdict(lambda: defaultdict(list))
        for dr, v in rows:
            cell = dr.get("cell", dr.get("base"))
            for s in ("A", "B"):
                for c in corpora:
                    cells[cell][c].append(v[(s, c)]["F"])
                    r = ro_get(dr[s], c)
                    if r is not None:
                        rocell[cell][c].append(r)
        recs = []
        for dr, v in rows:
            cell = dr.get("cell", dr.get("base"))
            rec = {"base": dr.get("base"), "d": dr.get("d"), "A": dr["A"], "B": dr["B"], "P": dr.get(P_key),
                   "F": {f"{s}|{c}": {"F": v[(s, c)]["F"], "SE": v[(s, c)]["SE"]} for s in ("A", "B") for c in corpora}}
            if log_def:
                rec["I_log"] = (math.log(v[("B", c1)]["F"]) - math.log(v[("A", c1)]["F"])) - \
                               (math.log(v[("B", c2)]["F"]) - math.log(v[("A", c2)]["F"]))
            fb = {c: float(np.mean(cells[cell][c])) for c in corpora}
            if all(fb[c] != 0 for c in corpora):
                rec["I_nd"] = (v[("B", c1)]["F"] - v[("A", c1)]["F"]) / fb[c1] - (v[("B", c2)]["F"] - v[("A", c2)]["F"]) / fb[c2]
            ros = {(s, c): ro_get(dr[s], c) for s in ("A", "B") for c in corpora}
            if all(x is not None for x in ros.values()) and all(rocell[cell][c] for c in corpora):
                rb = {c: float(np.mean(rocell[cell][c])) for c in corpora}
                rec["P_nd"] = (ros[("B", c1)] - ros[("A", c1)]) / rb[c1] - (ros[("B", c2)] - ros[("A", c2)]) / rb[c2]
            recs.append(rec)
        stat = "log" if use_log else "normalized_difference"
        ik, pk = ("I_log", P_key) if use_log else ("I_nd", "P_nd")
        ok = [r for r in recs if r.get(ik) is not None and (r.get("P") if use_log else r.get("P_nd")) is not None]
        for r in recs:
            if r not in ok:
                dropped.append({"base": r["base"], "d": r["d"], "reason": f"{stat} statistic or its P not defined"})
        return {"kind": kind, "corpora": list(corpora), "statistic": stat, "both_defined": bool(log_def and all(
                    "I_nd" in r for r in recs)), "n": len(ok), "dropped": dropped, "draws": recs,
                "I": [r[ik] for r in ok], "Pv": [(r["P"] if use_log else r["P_nd"]) for r in ok],
                "bases": [r["base"] for r in ok], "ds": [r["d"] for r in ok], "As": [r["A"] for r in ok]}


    # ---------------------------------------------------------------- draws, manifests, G0 routing tables
    def draws_of(self, role: str) -> list[dict]:
        by = defaultdict(dict)
        for r in self.plan.get("runs", []):
            if r.get("role") == role and r.get("side") in ("A", "B") and r.get("d") is not None:
                by[(r["base"], r.get("target"), int(r["d"]))][r["side"]] = r
        out = []
        for (b, tgt, d), s in sorted(by.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
            if {"A", "B"} <= set(s):
                if s["A"].get("P") is not None and s["B"].get("P") is not None and \
                        abs(float(s["A"]["P"]) - float(s["B"]["P"])) > 1e-12:
                    self.warnings.append(f"{role} {b} d{d}: P differs between the two runs of the draw")
                out.append({"base": b, "target": tgt, "d": d, "A": s["A"]["id"], "B": s["B"]["id"], "P": s["A"].get("P"),
                            "manifest": s["A"].get("draw_manifest"), "haar": s["A"].get("haar_seed")})
        return out

    def screen(self, draws: list[dict]) -> tuple[list[dict], list[dict], dict]:
        """Missing-value rule (v1.2 sec. 13): a draw with a run that is not done or diverged (E1) is dropped and counted."""
        ok, dropped, n_runs, n_div = [], [], 0, 0
        for dr in draws:
            why = []
            for s in ("A", "B"):
                n_runs += 1
                u, r = self.usable(dr[s])
                if self.e1_of(dr[s]).get("diverged"):
                    n_div += 1
                if not u:
                    why.append(f"{dr[s]}: {r}")
            (dropped.append({"base": dr["base"], "d": dr["d"], "reasons": why}) if why else ok.append(dr))
        e1 = {"runs": n_runs, "diverged": n_div, "share": n_div / n_runs if n_runs else None,
              "block_inconclusive": bool(n_runs and n_div / n_runs > E1_BLOCK_SHARE),
              "no_train_log": sorted(r for dr in draws for r in (dr["A"], dr["B"])
                                     if self.e1_of(r).get("status") == "no_train_log")}
        return ok, dropped, e1

    def manifest(self, path: str) -> dict | None:
        p = self.p(path)
        if p not in self._manifests:
            self._manifests[p] = self.json(p) if p and os.path.exists(p) else None
        return self._manifests[p]

    def entry(self, draw: dict) -> dict | None:
        m = self.manifest(draw.get("manifest")) if draw.get("manifest") else None
        for e in (m or {}).get("entries") or []:
            if int(e.get("d", -1)) == int(draw["d"]):
                return e
        return None

    def tables(self, sources: dict, family: str, corpora: list[str], stat: int = 2):
        """crossover.RegTables of the G0 routing tables named by a manifest entry's 'sources' for the given corpora.
        stat 2 = the registered statistic (per-token mean squared effective gate); stat 1 = mean effective gate
        (v1.2 sec. 5 mean-gate usage); stat 0 = selection frequency. Same files, layout and halves as the registered
        tables (the loader is crossover.reg_load_dir; only the statistic index differs)."""
        import crossover as CX
        cs = [c for c in corpora if c in (sources or {})]
        if len(cs) != len(corpora):
            return None
        f_sel = sources[cs[0]]["selection"]["file"]
        g0_dir = os.path.dirname(self.p(f_sel))
        half_seed = int(sources[cs[0]].get("half_seed", 0) or 0)
        key = (g0_dir, family, tuple(corpora), stat, half_seed)
        if key in self._tables:
            return self._tables[key]
        for c in corpora:
            for f in (os.path.join(g0_dir, f"routing__{c}.npz"), os.path.join(g0_dir, f"routing__{c}_eval.npz")):
                if os.path.exists(f):
                    self._read(f)
        if stat == 2:
            t = CX.reg_load_dir(g0_dir, family, corpora=list(corpora), half_seed=half_seed)
        else:
            orig = CX.reg_s2

            def s_k(res, strata, mask=None):
                rt = res["routing"]
                sel = np.ones(len(res["ce_sum"]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
                ntok = float(sel.sum()) * int(res["meta"].get("set", {}).get("block", 4096))
                return np.stack([np.asarray(rt[lk][sel, stat].sum(0), dtype=np.float64) / ntok for lk in strata])

            CX.reg_s2 = s_k
            try:
                t = CX.reg_load_dir(g0_dir, family, corpora=list(corpora), half_seed=half_seed)
            finally:
                CX.reg_s2 = orig
        self._tables[key] = t
        return t

    def predicted(self, entry: dict, design: str, *, code: str | None = None, target: str | None = None,
                  stat: int = 2, sources: dict | None = None) -> float | None:
        """crossover.reg_predicted on the estimation half (P: design c1; Delta_Q: design c2, H = B, L = A)."""
        import crossover as CX
        corp = [code or self.code, "math"] if design == "c1" else [target]
        t = self.tables(sources or entry.get("sources"), entry.get("family", "moe"), corp, stat)
        if t is None:
            return None
        A, B = np.asarray(entry["A"]), np.asarray(entry["B"])
        with np.errstate(divide="ignore"):
            v = float(CX.reg_predicted(t.est, A, B, design, target=target, code=code or self.code))
        return v if math.isfinite(v) else None

    def ro_table(self, entry: dict, side: str, corpus: str, stat: int = 2, sources: dict | None = None) -> float | None:
        import crossover as CX
        t = self.tables(sources or entry.get("sources"), entry.get("family", "moe"), [corpus], stat)
        return None if t is None else float(CX.reg_ro(t.est[corpus], np.asarray(entry[side])))


# ============================================================================= helpers for the blocks
def _beta_set(I, P, bases, kind: str) -> dict:
    """beta-hat with its test-inversion CI, overall and per base."""
    out = {"kind": kind, "overall": ci_invert(I, P), "per_base": {}}
    for b in sorted(set(bases)):
        ix = [i for i, x in enumerate(bases) if x == b]
        out["per_base"][b] = ci_invert([I[i] for i in ix], [P[i] for i in ix])
    return out


def _ub(est: dict | None):
    if not est or est.get("hi") is None:
        return None
    return est["hi"]


def _lb(est: dict | None):
    if not est or est.get("lo") is None:
        return None
    return est["lo"]


def _boot_ratio(I, P, bases, n: int = N_BOOT, seed: int = SEED_BOOT) -> dict | None:
    """Pooled ratio sum I / sum P with a draw bootstrap stratified by base (v1.2 sec. 9 item 2; seed 20261101)."""
    I, P = _arr(I), _arr(P)
    if I.size < 2 or float(P.sum()) == 0:
        return None
    rng = np.random.default_rng(seed)
    strata = [np.array([i for i, b in enumerate(bases) if b == x]) for x in sorted(set(bases))]
    vals = np.empty(n)
    for r in range(n):
        ix = np.concatenate([s[rng.integers(0, s.size, s.size)] for s in strata])
        sp = float(P[ix].sum())
        vals[r] = float(I[ix].sum()) / sp if sp != 0 else np.nan
    return {"ratio": float(I.sum() / P.sum()), "ci95": pct_ci(vals), "n_boot": n, "seed": seed,
            "stratified_by": "base"}


def _r2_boot(I, P, n: int = N_BOOT, seed: int = SEED_BOOT) -> dict | None:
    """Within-base R^2 of I_d on P_d (linear, with intercept) and its draw-bootstrap CI (v1.2 sec. 9 item 10)."""
    I, P = _arr(I), _arr(P)
    if I.size < 3:
        return None

    def r2(i, p):
        if float(np.var(p)) == 0 or float(np.var(i)) == 0:
            return np.nan
        c = float(np.corrcoef(i, p)[0, 1])
        return c * c

    if float(np.var(P)) == 0:
        return {"R2": None, "ci95": None, "n": int(I.size), "note": "undefined: the contrast does not vary over the draws"}
    rng = np.random.default_rng(seed)
    vals = np.array([r2(I[ix], P[ix]) for ix in (rng.integers(0, I.size, I.size) for _ in range(n))])
    return {"R2": r2(I, P), "ci95": pct_ci(vals), "n": int(I.size), "n_boot": n, "seed": seed}


# ============================================================================= C1 (v1.3 sec. 3.1; v1.2 secs. 8.1, 9)
def block_c1(ctx: Ctx) -> dict:
    corp = (ctx.code, "math")
    draws, dropped, e1 = ctx.screen(ctx.draws_of("c1"))
    beta0 = {"moe_b1": ctx.params.get("beta0_1"), "moe_b2": ctx.params.get("beta0_2")}
    ml = ctx.params.get("margin_limited per MoE base") or {}
    out = {"prereg": "v1.3 sec. 3.1; v1.2 sec. 8.1", "confirmatory": bool(ctx.params.get("C1_confirmatory", True)),
           "code_stream": ctx.code, "beta0": beta0, "margin_limited": ml, "draws_planned": len(ctx.draws_of("c1")),
           "draws_dropped": dropped, "E1": e1}
    if not draws:
        out["status"] = "no usable draws"
        return out
    kinds = ("raw", "sym", "pin", "sym_pin", "selpin", "sym_selpin", "wd")
    if getattr(ctx, "mode", "full") == "core":  # P-16: selpin evaluations are jobs of the selpin block (after the core)
        kinds = tuple(k for k in kinds if "selpin" not in k)
        out["deferred_to_full_analysis"] = ["selpin", "sym_selpin"]
    X = {k: ctx.cross(draws, k, corp) for k in kinds}
    rbase = sorted({d["base"] for d in draws if ctx.haar(d["A"]) and ctx.haar(d["B"])})
    cbase = "moe_b1" if "moe_b1" in rbase else (rbase[0] if rbase else None)
    X["rand"] = ctx.cross([d for d in draws if d["base"] == cbase and ctx.haar(d["A"]) and ctx.haar(d["B"])], "rand", corp)
    out["controls_base"] = cbase
    raw = X["raw"]
    J = [i - float(beta0[b]) * p for i, p, b in zip(raw["I"], raw["Pv"], raw["bases"])]
    out["statistic"] = raw["statistic"]
    out["per_draw"] = [{"base": b, "d": d, "I": i, "P": p, "J": j} for b, d, i, p, j in
                       zip(raw["bases"], raw["ds"], raw["I"], raw["Pv"], J)]
    out["D"] = len(J)
    out["test"] = signflip_p(J, "greater")
    out["mean_J_per_base"] = {b: {"mean_J": float(np.mean([j for j, x in zip(J, raw["bases"]) if x == b])),
                                  "n": sum(1 for x in raw["bases"] if x == b)} for b in sorted(set(raw["bases"]))}
    out["estimators"] = {k: _beta_set(X[k]["I"], X[k]["Pv"], X[k]["bases"], k) for k in ("raw", "pin", "sym", "sym_pin")}
    out["estimators"]["rand"] = _beta_set(X["rand"]["I"], X["rand"]["Pv"], X["rand"]["bases"], "rand")
    out["estimators"]["rand"]["empirical_sd_I_rand"] = float(np.std(X["rand"]["I"], ddof=1)) if X["rand"]["n"] > 1 else None
    out["estimators"]["rand"]["t_interval_slope"] = None
    if X["rand"]["n"] > 1:
        b_, s_ = beta_origin(X["rand"]["I"], X["rand"]["Pv"]), se_origin(X["rand"]["I"], X["rand"]["Pv"])
        if s_ is not None:
            h = tq(X["rand"]["n"] - 1, 0.975) * s_
            out["estimators"]["rand"]["t_interval_slope"] = {"lo": b_ - h, "hi": b_ + h, "se": s_, "df": X["rand"]["n"] - 1}
    out["statistics"] = {k: {kk: X[k][kk] for kk in ("statistic", "n", "dropped", "draws", "both_defined")} for k in X}
    out["_X"] = X                      # internal (removed before writing)
    out["ood"] = _c1_ood(ctx, draws, X)
    out["secondary"] = _c1_secondary(ctx, draws, X, beta0)
    return out


def _c1_ood(ctx: Ctx, draws: list[dict], X: dict) -> dict:
    """Out-of-domain annotations per (base x target) cell (v1.2 sec. 8.1 item 7; sec. 9 items 14 and 19)."""
    corp = (ctx.code, "math")
    cells = {}
    for b in sorted({d["base"] for d in draws}):
        dd = [d for d in draws if d["base"] == b]
        # (a, b) interim ratio I_k / I_96 on the 64 interim windows
        ratios = {}
        for k in (8, 24, 48):
            vals, undefined = [], []
            for dr in dd:
                Fk, F96 = {}, {}
                for s in ("A", "B"):
                    root = ctx.root_of(dr[s])
                    for c in corp:
                        h0 = os.path.join(root, "interim", f"step000__{c}.npz") if root else None
                        hk = os.path.join(root, "interim", f"step{k:03d}__{c}.npz") if root else None
                        pk = ctx.pair(h0, hk) if h0 and os.path.exists(h0) else ctx.pair(ctx.base_file(b, c), hk, subset=True)
                        w = ctx.windows(dr[s], c, "raw")
                        if pk is None or w is None:
                            Fk[(s, c)] = None
                            continue
                        Fk[(s, c)] = float((pk["cr"] - pk["cb"]).mean())
                        sel = np.isin(w["index"], pk["index"])
                        F96[(s, c)] = float(w["d"][sel].mean())
                if any(v is None for v in Fk.values()):
                    undefined.append({"d": dr["d"], "reason": "interim file missing"})
                    continue
                if any(v <= 0 for v in Fk.values()) or any(v <= 0 for v in F96.values()):
                    undefined.append({"d": dr["d"], "reason": "F <= 0 at an interim step or at step 96 (head windows)"})
                    continue
                lI = lambda F: (math.log(F[("B", corp[0])]) - math.log(F[("A", corp[0])])) - \
                    (math.log(F[("B", corp[1])]) - math.log(F[("A", corp[1])]))  # noqa: E731
                i96 = lI(F96)
                if i96 == 0:
                    undefined.append({"d": dr["d"], "reason": "I_96 = 0"})
                    continue
                vals.append(lI(Fk) / i96)
            med = float(np.median(vals)) if vals and not undefined else None
            ratios[k] = {"median": med, "values": vals, "undefined": undefined,
                         "flag": bool(med is None or not (OOD_RANGE[0] <= med <= OOD_RANGE[1]))}
        # (c) resolved first-order share, (d) F_sym <= 0 flag
        fo = {}
        flag_sym_pin, live_flags = [], []
        for c in corp:
            res_sh, unres_sh = [], []
            for dr in dd:
                for s in ("A", "B"):
                    f = ctx.F(dr[s], c, "sym")
                    if f is None:
                        continue
                    if f["F"] != 0:
                        unres_sh.append(abs(f["F_odd"]) / abs(f["F"]))
                        res_sh.append(abs(f["F_odd"]) / abs(f["F"]) if abs(f["F_odd"]) > 2 * f["SE_odd"] else 0.0)
                    if f["F"] + 2 * f["SE"] < 0:
                        live_flags.append(f"{dr[s]}|{c}")
                    fp = ctx.F(dr[s], c, "sym_pin")
                    if fp is not None and fp["F"] + 2 * fp["SE"] < 0:
                        flag_sym_pin.append(f"{dr[s]}|{c}")
            fo[c] = {"median_resolved_share": float(np.median(res_sh)) if res_sh else None,
                     "median_unresolved_share": float(np.median(unres_sh)) if unres_sh else None, "runs": len(res_sh),
                     "flag": bool(res_sh and float(np.median(res_sh)) > FO_SHARE)}
        cells[b] = {"interim_ratio": ratios, "first_order": fo, "F_sym_pin_flag_runs": flag_sym_pin,
                    "F_sym_pin_flag": bool(flag_sym_pin), "live_F_sym_negative (curvature or router flips)": live_flags}
        cells[b]["out_of_domain"] = bool(any(r["flag"] for r in ratios.values()) or any(v["flag"] for v in fo.values())
                                         or flag_sym_pin)
    return cells


def _diff_ci(X1: dict, X2: dict) -> dict | None:
    """Draw-level t-interval of I1_d - I2_d over the draws both statistics have (matched by base and d)."""
    a = {(b, d): i for b, d, i in zip(X1["bases"], X1["ds"], X1["I"])}
    c = {(b, d): i for b, d, i in zip(X2["bases"], X2["ds"], X2["I"])}
    keys = sorted(set(a) & set(c))
    if not keys:
        return None
    r = t_interval([a[k] - c[k] for k in keys])
    r["statistics"] = [X1["statistic"], X2["statistic"]]
    if X1["statistic"] != X2["statistic"]:
        r["note"] = "the two statistics use different E5 forms"
    return r


def _ratio_median(ctx: Ctx, draws, k_num: str, k_den: str, corp) -> dict:
    out = {}
    for c in corp:
        v = []
        for dr in draws:
            for s in ("A", "B"):
                n, d = ctx.F(dr[s], c, k_num), ctx.F(dr[s], c, k_den)
                if n is not None and d is not None and d["F"] != 0:
                    v.append(n["F"] / d["F"])
        out[c] = {"median": float(np.median(v)) if v else None, "runs": len(v)}
    return out


def _c1_secondary(ctx: Ctx, draws: list[dict], X: dict, beta0: dict) -> dict:
    corp = (ctx.code, "math")
    raw, sym, rand, pin, wd = X["raw"], X["sym"], X["rand"], X["pin"], X["wd"]
    sec = {"label": "secondary (v1.2 sec. 9; unadjusted)"}
    sec["plain_null_I_gt_0"] = signflip_p(raw["I"], "greater")
    sec["fixed_margin_0.25"] = signflip_p([i - BETA_REF * p for i, p in zip(raw["I"], raw["Pv"])], "greater")
    # C1-b magnitude (item 2)
    for nm, Xk in (("sym", sym), ("raw", raw)):
        r = _boot_ratio(Xk["I"], Xk["Pv"], Xk["bases"])
        if r and r["ci95"]:
            lo, hi = r["ci95"]
            r["C1b_refuted"] = bool(hi < 0.5 or lo > 2.0)
        sec[f"C1b_pooled_ratio_{nm}"] = r
    # C1-d DCLM negative control (item 3)
    sec["C1d_DCLM_control"] = _c1d(ctx, draws)
    # random-direction contrast (item 4), M12-a/c (item 17), beta_rand (item 15)
    sec["I_sym_minus_I_rand"] = _diff_ci(sym, rand)
    sec["I_minus_I_rand"] = _diff_ci(raw, rand)
    sec["F_over_F_rand"] = _ratio_median(ctx, [d for d in draws if ctx.haar(d["A"])], "raw", "rand", corp)
    lam = []
    for dr in draws:
        for s in ("A", "B"):
            for c in corp + ("dclm",):
                f, fr = ctx.F(dr[s], c, "raw"), ctx.F(dr[s], c, "rand")
                if f and fr and f["F"] > 0 and fr["F"] > 0:
                    lam.append({"run": dr[s], "corpus": c, "Lambda": math.log(f["F"] / fr["F"])})
    sec["M12a_alignment_Lambda"] = lam
    est = ci_invert(rand["I"], rand["Pv"]) if rand["n"] else None
    sec["C1R_beta_rand"] = {"estimate": est, "localization_failed": bool(est and est.get("bounded") and (
        est["hi"] < 0.5 or est["lo"] > 2.0))}
    # pinned share (item 5), WD share (item 6)
    mI = float(np.mean(raw["I"])) if raw["n"] else None
    sec["pinned_share"] = {"mean_I_pin_over_mean_I": (float(np.mean(pin["I"])) / mI) if pin["n"] and mI else None,
                           "F_pin_over_F": _ratio_median(ctx, draws, "pin", "raw", corp),
                           "F_sym_pin_over_F_sym": _ratio_median(ctx, draws, "sym_pin", "sym", corp),
                           "note": "1 - F_pin/F is the routing-pathway share (router flips plus gate values)"}
    sec["wd_share"] = {"mean_I_WD_over_mean_I": (float(np.mean(wd["I"])) / mI) if wd["n"] and mI else None,
                       "F_WD_over_F": _ratio_median(ctx, draws, "wd", "raw", corp), "n_draws": wd["n"]}
    # second code stream (item 18)
    sec["second_code_stream"] = _c1_code2(ctx, draws, beta0)
    # C1-e (item 10): within-base R^2 of I on P for second-moment, mean-gate and frequency contrasts
    sec["C1e_within_base_R2"] = _c1e(ctx, draws, raw)
    return sec


def _c1d(ctx: Ctx, draws: list[dict]) -> dict:
    vals, raw_vals, miss = [], [], []
    for dr in draws:
        fa, fb = ctx.F(dr["A"], "dclm", "sym"), ctx.F(dr["B"], "dclm", "sym")
        ra, rb = ctx.ro(dr["A"], "dclm"), ctx.ro(dr["B"], "dclm")
        if not (fa and fb and ra and rb) or fa["F"] <= 0 or fb["F"] <= 0:
            miss.append(dr["d"])
            continue
        vals.append(math.log(fb["F"] / fa["F"]) - (math.log(rb) - math.log(ra)))
        ga, gb = ctx.F(dr["A"], "dclm", "raw"), ctx.F(dr["B"], "dclm", "raw")
        if ga and gb and ga["F"] > 0 and gb["F"] > 0:
            raw_vals.append(math.log(gb["F"] / ga["F"]))
    out = {"statistic": "log F_sym,DCLM(B)/F_sym,DCLM(A) - [log RO_DCLM(B) - log RO_DCLM(A)]", "n": len(vals),
           "dropped_draws": miss}
    if vals:
        out["mean"] = float(np.mean(vals))
        out["signflip_two_sided"] = signflip_p(vals, "two-sided")
        ti = t_interval(vals, 0.90)
        out["TOST_90pct_t_interval"] = ti
        out["TOST_equivalent_within_log1.1"] = bool(ti and ti["lo"] is not None and ti["lo"] > -LOG_TOST and ti["hi"] < LOG_TOST)
        out["failed_control (systematic sign)"] = bool(out["signflip_two_sided"]["p"] <= ALPHA)
    out["raw_unadjusted"] = {"log F_DCLM(B)/F_DCLM(A)": raw_vals, "mean": float(np.mean(raw_vals)) if raw_vals else None,
                             "signflip_two_sided": signflip_p(raw_vals, "two-sided") if raw_vals else None}
    return out


def _c1_code2(ctx: Ctx, draws: list[dict], beta0: dict) -> dict:
    c2 = ctx.code2
    if not c2:
        return {"status": "no second stream in the plan"}
    Pc2 = {}
    for dr in draws:
        e = ctx.entry(dr)
        Pc2[(dr["base"], dr["d"])] = ctx.predicted(e, "c1", code=c2) if e else None
    dd = [dict(dr, P2=Pc2[(dr["base"], dr["d"])]) for dr in draws]
    side_of = {}
    for dr in draws:
        e = ctx.entry(dr)
        if e:
            side_of[dr["A"]], side_of[dr["B"]] = (e, "A"), (e, "B")

    def ro2(rid, c):                                     # RO of the second stream is not in the plan: G0 tables
        if rid not in side_of:
            return None
        e, sd = side_of[rid]
        return ctx.ro_table(e, sd, c)

    X = ctx.cross(dd, "raw", (c2, "math"), P_key="P2", ro_get=ro2)
    out = {"stream": c2, "statistic": X["statistic"], "n": X["n"], "dropped": X["dropped"],
           "per_draw": [{"base": b, "d": d, "I": i, "P_code2": p} for b, d, i, p in zip(X["bases"], X["ds"], X["I"], X["Pv"])]}
    if X["n"]:
        J = [i - float(beta0[b]) * p for i, p, b in zip(X["I"], X["Pv"], X["bases"])]
        out["J_test"] = signflip_p(J, "greater")
        out["plain_null"] = signflip_p(X["I"], "greater")
        out["beta"] = _beta_set(X["I"], X["Pv"], X["bases"], "raw (second stream)")
    return out


def _c1e(ctx: Ctx, draws: list[dict], raw: dict) -> dict:
    out = {}
    Pm, Pf = {}, {}
    for dr in draws:
        e = ctx.entry(dr)
        Pm[(dr["base"], dr["d"])] = ctx.predicted(e, "c1", stat=1) if e else None
        Pf[(dr["base"], dr["d"])] = ctx.predicted(e, "c1", stat=0) if e else None
    for b in sorted(set(raw["bases"])):
        ix = [i for i, x in enumerate(raw["bases"]) if x == b]
        I = [raw["I"][i] for i in ix]
        keys = [(b, raw["ds"][i]) for i in ix]
        res = {"second_moment": _r2_boot(I, [raw["Pv"][i] for i in ix])}
        for nm, src in (("mean_gate", Pm), ("frequency", Pf)):
            pv = [src.get(k) for k in keys]
            res[nm] = _r2_boot(I, pv) if all(v is not None for v in pv) else None
        out[b] = res
    out["note"] = "descriptive; no refutation rule (v1.2 sec. 9 item 10)"
    return out


def decide_c1(c1: dict, p_adj: float | None) -> dict:
    """Decision of v1.2 sec. 8.1 item 6 with the three-statistic table (item 7) and interpretation rules (item 8)."""
    if c1.get("status") == "no usable draws":
        return {"label": "inconclusive", "reasons": ["no usable draws"]}
    est = c1["estimators"]
    ub = {k: _ub(est[k]["overall"]) for k in ("raw", "pin", "sym")}
    ml = [b for b, v in (c1.get("margin_limited") or {}).items() if v]
    reasons = []
    per_base_pos = all(v["mean_J"] > 0 for v in c1["mean_J_per_base"].values()) and len(c1["mean_J_per_base"]) >= 2
    if c1["E1"]["block_inconclusive"]:
        label = "inconclusive"
        reasons.append(f"E1: {c1['E1']['diverged']} of {c1['E1']['runs']} runs diverged (> 25 %)")
    elif p_adj is not None and p_adj <= ALPHA and per_base_pos:
        label = "confirmed (margin-limited)" if ml else "confirmed"
    elif p_adj is not None and p_adj <= ALPHA:
        label = "confirmed in the pooled analysis only (not replicated across bases)"
    elif all(u is not None and u < BETA_REF for u in ub.values()) and not ml:
        label = "refuted, reversed" if all(u < 0 for u in ub.values()) else "refuted"
    else:
        label = "inconclusive"
        if p_adj is not None:
            reasons.append(f"Holm-adjusted p = {p_adj:.4g} > {ALPHA}")
        nb = [k for k, u in ub.items() if u is None or u >= BETA_REF]
        if nb:
            reasons.append("upper 95 % bound not below 0.25 for " + ", ".join(f"beta-hat_{k} ({ub[k]})" for k in nb))
        if ml and all(u is not None and u < BETA_REF for u in ub.values()):
            reasons.append(f"would-be refutation in a margin-limited base ({', '.join(ml)})")
    if not c1.get("confirmatory", True):
        reasons.append("C1 is not confirmatory (pivot recorded at the freeze): the label is exploratory")
    out = {"label": label, "reasons": reasons, "upper_bounds": ub, "holm_adjusted_p": p_adj,
           "mean_J_positive_in_each_base": per_base_pos}
    # three-statistic table (when the upper bound of beta-hat is below 0.25)
    sy, ra = est["sym"]["overall"], est["rand"]["overall"]
    if ub["raw"] is not None and ub["raw"] < BETA_REF:
        sym_hi = bool(sy and sy.get("lo") is not None and ((sy["lo"] <= 1.0 <= sy["hi"]) or
                                                          (sy["beta_hat"] >= 0.5 and sy["lo"] > BETA_REF)))
        rand_hi = bool(ra and ra.get("lo") is not None and ra["beta_hat"] >= 0.5 and ra["lo"] > BETA_REF)
        sym_lo = bool(sy and sy.get("hi") is not None and sy["hi"] < BETA_REF)
        rand_lo = bool(ra and ra.get("hi") is not None and ra["hi"] < BETA_REF)
        if sym_hi and rand_hi:
            t3 = "first-order (transfer) regime: the raw crossover is outside the theory's domain; the second-order law holds; not refuted"
        elif sym_lo and rand_hi:
            t3 = "learned-update law refuted; routing localization holds"
        elif sym_lo and rand_lo:
            t3 = "refuted: routing mass does not localize forgetting (L1 or B5 fails)"
        else:
            t3 = "inconclusive (other combination of beta-hat_sym and beta-hat_rand)"
        out["three_statistic_table"] = {"label": t3, "beta_sym": sy, "beta_rand": ra}
    ood = [b for b, v in (c1.get("ood") or {}).items() if v.get("out_of_domain")]
    out["out_of_domain_cells"] = ood
    sec = c1.get("secondary") or {}
    interp = []
    if label.startswith("confirmed"):
        d = sec.get("I_sym_minus_I_rand")
        if d is None or d.get("lo") is None or d["lo"] <= 0:
            interp.append("routing-gated sensitivity to fixed-norm perturbations (not: interference specific to learned updates)")
    wd = (sec.get("wd_share") or {}).get("mean_I_WD_over_mean_I")
    if wd is not None and wd >= 0.5:
        interp.append("substantially decay-driven (mean I_WD >= 0.5 x mean I)")
    out["interpretation"] = interp
    return out



# ============================================================================= C2 (v1.2 sec. 8.2; v1.3 sec. 3.2)
def _disp(ctx: Ctx, rid: str, step: int) -> float | None:
    root = ctx.root_of(rid)
    f = os.path.join(root, "interim", "displacement.jsonl") if root else None
    if not f or not os.path.exists(f):
        return None
    ctx._read(f)
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if int(r.get("step", -1)) == int(step):
                v = float(r.get("total_sq", math.nan))
                return math.sqrt(v) if v > 0 else None
    return None


def _c2_F(ctx: Ctx, rid: str, base: str, step: int) -> dict | None:
    """F_DCLM of a C2 run at a step: the full 2.048M-token set (interim/stepNNN__dclm__full.npz at 8 / 24 / 48, the
    final evaluation at step 96), paired with the theta_0 file of the base."""
    if step >= int((ctx.runs.get(rid) or {}).get("steps", 96)):
        return ctx.F(rid, "dclm", "raw")
    root = ctx.root_of(rid)
    p = os.path.join(root, "interim", f"step{int(step):03d}__dclm__full.npz") if root else None
    pp = ctx.pair(ctx.base_file(base, "dclm"), p)
    return None if pp is None else ctx._stat(pp["cr"] - pp["cb"])


SGD_GAIN_ROLE = ("exploratory, descriptive: evidence that the SGD arm learns (reported in the prereg); no decision role - "
                 "not part of any test, interval, gate, E-rule or Holm family")


def _sgd_target_gain(ctx: Ctx, cells: dict) -> dict:
    """C2.sgd_step24_gain: the C2 target (German) gain of every usable C2 SGD run on its
    interim head, G_s = mean over the head windows w of [CE_step0(w) - CE_step_s(w)], s = 24 (reported) and s = 8
    (comparison). Step 0 = the run's in-process step-000 head (theta_0 evaluated before the first update, the same
    windows); windows are paired by grid index with equal token hashes (Ctx.pair). Positive = the target loss fell."""
    out = {"role": SGD_GAIN_ROLE, "decision_role": None,
           "definition": "G_s = mean_w [CE_step0(w) - CE_step_s(w)] on the run's interim head of the C2 target set "
                         "(interim/step000__<target>.npz vs interim/stepNNN__<target>.npz); s = 24 and, for comparison, s = 8",
           "source": "the queue's interim block of the C2 jobs: 64-window heads of 'dclm' and the target set at the interim "
                     "steps (SGD: 0, 8, 24 - the run stops after step 24, P-16 / MF-P1 (A)); step 0 is evaluated in-process",
           "runs": "C2 SGD runs that are usable (status done, no E1 divergence); the others are listed under 'excluded'"}
    per_run, excluded, missing = [], [], []
    for (b, d), rr in sorted(cells.items()):
        for side in ("H", "L"):
            r = rr.get(("sgd", side))
            if r is None:
                continue
            rid = r["id"]
            u, why = ctx.usable(rid)
            if not u:
                excluded.append({"run": rid, "reason": why})
                continue
            tgt = r.get("target", "german")
            root = ctx.root_of(rid)
            rec = {"run": rid, "base": b, "d": int(d), "side": side, "set": tgt}
            b0 = os.path.join(root, "interim", f"step000__{tgt}.npz") if root else None
            for s in (8, 24):
                p = os.path.join(root, "interim", f"step{s:03d}__{tgt}.npz") if root else None
                if not (b0 and p and os.path.exists(b0) and os.path.exists(p)):
                    rec[f"G_step{s}"] = None
                    missing.append(f"{rid}: step {s} (" + ("no step-000 head" if not (b0 and os.path.exists(b0)) else
                                                         f"no step-{s:03d} head") + ")")
                    continue
                pp = ctx.pair(b0, p)
                if pp is None:
                    rec[f"G_step{s}"] = None
                    missing.append(f"{rid}: step {s} (unreadable)")
                    continue
                rec[f"G_step{s}"] = float(np.mean(pp["cb"] - pp["cr"]))
                rec[f"windows_step{s}"] = int(len(pp["index"]))
            per_run.append(rec)

    def summ(vals: list) -> dict:
        v = [float(x) for x in vals if x is not None]
        return {"mean": float(np.mean(v)) if v else None, "min": min(v) if v else None, "max": max(v) if v else None, "n": len(v)}

    for s in (24, 8):
        k = f"G_step{s}"
        out[f"step{s}"] = {"pooled": summ([x[k] for x in per_run]),
                           "per_base": {bb: summ([x[k] for x in per_run if x["base"] == bb])
                                        for bb in sorted({x["base"] for x in per_run})}}
    out["windows"] = sorted({x[k] for x in per_run for k in ("windows_step8", "windows_step24") if k in x})
    out["n_planned"] = sum(1 for rr in cells.values() for side in ("H", "L") if ("sgd", side) in rr)
    out["per_run"] = per_run
    out["excluded"] = excluded
    out["missing_heads"] = missing
    return out


def block_c2(ctx: Ctx) -> dict:
    s_star = ctx.params.get("s_star")
    fam = ctx.params.get("holm_family") or ["C1"]
    confirm = bool(ctx.params.get("C2_confirmatory")) and "C2" in fam
    out = {"prereg": "v1.2 sec. 8.2; v1.3 sec. 3.2", "s_star": s_star, "confirmatory": confirm,
           "draws_registered": ctx.params.get("C2_draws")}
    cells = defaultdict(dict)
    for r in ctx.plan.get("runs", []):
        if r.get("role") in ("c2_sgd", "c2_muon"):
            cells[(r["base"], int(r["d"]))][(r["arm"], r["side"])] = r
    if not cells:
        out["status"] = "not run (no C2 runs in the plan)"
        return out
    step = int(s_star) if s_star in (8, 24, 48) else 96
    if s_star not in (8, 24, 48):
        out["note"] = f"s* = {s_star}: no interim step passed G2-3; analysed at step 96 and reported as exploratory (O-6)"
    draws, dropped, n_runs, n_div = [], [], 0, 0
    for (b, d), rr in sorted(cells.items()):
        need = [("sgd", "H"), ("sgd", "L"), ("muon", "H"), ("muon", "L")]
        why = [f"{a}-{s} not planned" for a, s in need if (a, s) not in rr]
        for k in need:
            if k in rr:
                n_runs += 1
                n_div += bool(ctx.e1_of(rr[k]["id"]).get("diverged"))
                u, w = ctx.usable(rr[k]["id"])
                if not u:
                    why.append(f"{rr[k]['id']}: {w}")
        dq = {float(rr[k].get("Delta_Q")) for k in need if k in rr and rr[k].get("Delta_Q") is not None}
        if len(dq) != 1:
            why.append("Delta_Q missing or not equal across the four runs")
        if why:
            dropped.append({"base": b, "d": d, "reasons": why})
            continue
        man = rr[("sgd", "H")].get("draw_manifest")
        draws.append({"base": b, "d": d, "runs": {k: rr[k]["id"] for k in need}, "DQ": dq.pop(),
                      "entry": ctx.entry({"manifest": man, "d": d}), "target": rr[("sgd", "H")].get("target", "german")})
    out["draws_dropped"] = dropped
    out["E1"] = {"runs": n_runs, "diverged": n_div, "block_inconclusive": bool(n_runs and n_div / n_runs > E1_BLOCK_SHARE)}
    out["sgd_step24_gain"] = _sgd_target_gain(ctx, cells)       # descriptive, no decision role

    def at(stp: int) -> dict:
        Fv = {}
        for dr in draws:
            for k, rid in dr["runs"].items():
                Fv[(dr["base"], dr["d"], k)] = _c2_F(ctx, rid, dr["base"], stp)
        ok = [dr for dr in draws if all(Fv[(dr["base"], dr["d"], k)] is not None for k in dr["runs"])]
        allv = [Fv[(dr["base"], dr["d"], k)] for dr in ok for k in dr["runs"]]
        use_log = bool(ok) and all(resolved(x["F"], x["SE"]) for x in allv)
        log_def = bool(ok) and all(x["F"] > 0 for x in allv)
        fbar = defaultdict(list)
        for dr in ok:
            for (a, s) in dr["runs"]:
                fbar[(dr["base"], a)].append(Fv[(dr["base"], dr["d"], (a, s))]["F"])
        recs = []
        for dr in ok:
            f = {k: Fv[(dr["base"], dr["d"], k)]["F"] for k in dr["runs"]}
            r = {"base": dr["base"], "d": dr["d"], "Delta_Q": dr["DQ"], "F": f}
            if log_def:
                r["r_sgd"] = math.log(f[("sgd", "H")]) - math.log(f[("sgd", "L")])
                r["r_muon"] = math.log(f[("muon", "H")]) - math.log(f[("muon", "L")])
                r["I2_log"] = r["r_sgd"] - r["r_muon"]
            mb = {a: float(np.mean(fbar[(dr["base"], a)])) for a in ("sgd", "muon")}
            if all(mb[a] != 0 for a in mb):
                r["I2_nd"] = (f[("sgd", "H")] - f[("sgd", "L")]) / mb["sgd"] - (f[("muon", "H")] - f[("muon", "L")]) / mb["muon"]
            recs.append(r)
        return {"step": stp, "statistic": "log" if use_log else "normalized_difference", "records": recs,
                "use_log": use_log, "n": len(recs), "missing": len(draws) - len(ok)}

    main = at(step)
    out["statistic"] = main["statistic"]
    recs = main["records"]
    if not main["use_log"]:                     # Delta_Q^nd from RO_T (estimation half) in the same way as P_nd
        rob = defaultdict(list)
        ro = {}
        for dr in draws:
            e = dr["entry"]
            if e:
                for side in ("A", "B"):
                    v = ctx.ro_table(e, side, dr["target"])
                    ro[(dr["base"], dr["d"], side)] = v
                    if v is not None:
                        rob[dr["base"]].append(v)
        for r in recs:
            a, b_ = ro.get((r["base"], r["d"], "A")), ro.get((r["base"], r["d"], "B"))
            if a is not None and b_ is not None and rob[r["base"]]:
                r["Delta_Q_nd"] = (b_ - a) / float(np.mean(rob[r["base"]]))
    ik, pk = ("I2_log", "Delta_Q") if main["use_log"] else ("I2_nd", "Delta_Q_nd")
    use = [r for r in recs if r.get(ik) is not None and r.get(pk) is not None]
    I2, DQ, bases = [r[ik] for r in use], [r[pk] for r in use], [r["base"] for r in use]
    J2 = [i - BETA_REF * q for i, q in zip(I2, DQ)]
    out["per_draw"] = [{"base": r["base"], "d": r["d"], "I2": r[ik], "Delta_Q": r[pk], "J2": j,
                        "F_DCLM": {f"{a}-{s}": v for (a, s), v in r["F"].items()}} for r, j in zip(use, J2)]
    out["D"] = len(J2)
    out["test"] = signflip_p(J2, "greater") if J2 else None
    out["mean_J2_per_base"] = {b: float(np.mean([j for j, x in zip(J2, bases) if x == b])) for b in sorted(set(bases))}
    out["Delta_e"] = ci_invert(I2, DQ) if I2 else None
    if main["use_log"]:
        out["e_SGD"] = ci_invert([r["r_sgd"] for r in use], DQ)
        out["e_Muon"] = ci_invert([r["r_muon"] for r in use], DQ)
    # manipulation check at step 8 (v1.2 sec. 8.2 item 8)
    man = []
    for dr in draws:
        e = dr["entry"]
        dqm = ctx.predicted(e, "c2", target=dr["target"], stat=1) if e else None
        rec = {"base": dr["base"], "d": dr["d"], "Delta_Q_mean": dqm}
        for arm in ("sgd", "muon"):
            for stp in (1, 8):
                h, l_ = _disp(ctx, dr["runs"][(arm, "H")], stp), _disp(ctx, dr["runs"][(arm, "L")], stp)
                rec[f"D_{arm}_step{stp}"] = (math.log(h) - math.log(l_)) if h and l_ else None
        rec["D_sgd_over_DQmean"] = (rec["D_sgd_step8"] / dqm) if rec["D_sgd_step8"] is not None and dqm else None
        man.append(rec)
    rat = [m["D_sgd_over_DQmean"] for m in man if m["D_sgd_over_DQmean"] is not None]
    out["manipulation_check"] = {"per_draw": man, "mean_D_SGD_over_DQ_mean": float(np.mean(rat)) if rat else None,
                                 "failed": bool(rat and float(np.mean(rat)) < C2_MANIP), "n": len(rat),
                                 "D_Muon_step1": [m["D_muon_step1"] for m in man]}
    # C2-eq (exploratory; v1.2 sec. 8.2 item 7) at the analysis step
    if main["use_log"] and len(use) >= 2:
        e_d = [r["r_muon"] / r["Delta_Q"] for r in use]
        ti = t_interval(e_d, 0.90)
        dm = []
        for dr in draws:
            h, l_ = _disp(ctx, dr["runs"][("muon", "H")], step), _disp(ctx, dr["runs"][("muon", "L")], step)
            dqm = next((m["Delta_Q_mean"] for m in man if (m["base"], m["d"]) == (dr["base"], dr["d"])), None)
            if h and l_ and dqm:
                dm.append((math.log(h) - math.log(l_)) / dqm)
        mdm = float(np.mean(dm)) if dm else None
        margin = C2EQ_MARGIN + (2 * mdm if mdm is not None and mdm > 0.1 else 0.0)
        out["C2_eq"] = {"e_d": e_d, "t90": ti, "margin": margin, "mean_D_Muon_over_DQ_mean": mdm,
                        "widened": bool(mdm is not None and mdm > 0.1),
                        "equivalent": bool(ti and ti["lo"] is not None and -margin < ti["lo"] and ti["hi"] < margin),
                        "label": "exploratory"}
    # decay profile (v1.2 sec. 9 item 20)
    prof = {}
    stops = {int((ctx.runs.get(rid) or {}).get("stop_after") or 0) for dr in draws for rid in dr["runs"].values()}
    s_stop = min(stops) if stops and 0 not in stops else 0     # MF-P1 (A): Muon 8 / SGD 24 -> the contrast ends at step 8
    if s_stop:                                  # P-16: the runs stopped after step s_stop; later steps were not run
        out["stopped_after_step"] = s_stop
        out["not_computable_P16"] = [f"C2 decay of the optimizer contrast at step {x}: not computable (C2 runs stopped "
                                     f"after step {s_stop}, P-16)" for x in (24, 48, 96) if x > s_stop]
    for stp in [x for x in (8, 24, 48, 96) if not s_stop or x <= s_stop]:
        a = at(stp)
        rr = [r for r in a["records"] if r.get("I2_log") is not None]
        prof[stp] = {"statistic": a["statistic"], "n": len(rr),
                     "I2": [r["I2_log"] for r in rr], "J2": [r["I2_log"] - BETA_REF * r["Delta_Q"] for r in rr],
                     "D_sgd": [None if (h := _disp(ctx, dr["runs"][("sgd", "H")], stp)) is None or
                               (l_ := _disp(ctx, dr["runs"][("sgd", "L")], stp)) is None else math.log(h / l_) for dr in draws],
                     "D_muon": [None if (h := _disp(ctx, dr["runs"][("muon", "H")], stp)) is None or
                                (l_ := _disp(ctx, dr["runs"][("muon", "L")], stp)) is None else math.log(h / l_) for dr in draws]}
    out["decay_profile"] = prof
    return out


def decide_c2(c2: dict, p_adj: float | None) -> dict:
    if c2.get("status") or not c2.get("D"):
        return {"label": "not run" if c2.get("status") else "inconclusive", "reasons": [c2.get("status") or "no draws"]}
    reasons = []
    per_base = c2["mean_J2_per_base"]
    pos = len(per_base) >= 2 and all(v > 0 for v in per_base.values())
    p_use = p_adj if c2["confirmatory"] else c2["test"]["p"]
    ub = _ub(c2.get("Delta_e"))
    if c2["E1"]["block_inconclusive"]:
        label = "inconclusive"
        reasons.append("E1: more than 25 % of the C2 runs diverged")
    elif p_use is not None and p_use <= ALPHA and pos:
        label = "confirmed"
    elif ub is not None and ub < BETA_REF:
        label = "refuted"
    else:
        label = "inconclusive"
        reasons.append(f"p = {p_use} and upper bound of Delta-e = {ub}")
    if not label.startswith("confirmed") and c2["manipulation_check"]["failed"]:
        label = "uninformative"
        reasons.append("SGD manipulation failed (mean D_SGD / Delta_Q(mean) < 0.5 at step 8)")
    if not c2["confirmatory"]:
        label = f"exploratory: {label}"
        reasons.append("C2 is not in the confirmatory family fixed at the freeze (s* or gates)")
    return {"label": label, "reasons": reasons, "holm_adjusted_p": p_adj if c2["confirmatory"] else None,
            "upper_bound_Delta_e": ub, "mean_J2_positive_in_each_base": pos}


# ============================================================================= small-draw blocks (third base, C1-DE, NESTOR-books)
def _small_block(ctx: Ctx, role: str, beta0_of, primary: str, df_label: str, c1_ref: dict | None) -> dict:
    corp = (ctx.code, "math")
    all_draws = ctx.draws_of(role)
    draws, dropped, e1 = ctx.screen(all_draws)
    out = {"draws_planned": len(all_draws), "draws_dropped": dropped, "E1": e1, "primary": primary}
    if not draws:
        out["status"] = "not run" if not all_draws else "no usable draws"
        return out
    X = {k: ctx.cross(draws, k, corp) for k in ("raw", "sym", "pin", "sym_pin", "selpin", "sym_selpin")}
    Xp = X[primary]
    J = [i - float(beta0_of(b)) * p for i, p, b in zip(Xp["I"], Xp["Pv"], Xp["bases"])]
    out["statistic"] = Xp["statistic"]
    out["per_draw"] = [{"base": b, "d": d, "I": i, "P": p, "J": j} for b, d, i, p, j in
                       zip(Xp["bases"], Xp["ds"], Xp["I"], Xp["Pv"], J)]
    out["mean_J"] = t_interval(J)
    out["mean_J_per_base"] = {b: t_interval([j for j, x in zip(J, Xp["bases"]) if x == b]) for b in sorted(set(Xp["bases"]))}
    out["signflip"] = signflip_p(J, "greater")
    out["estimators"] = {k: _beta_set(X[k]["I"], X[k]["Pv"], X[k]["bases"], k) for k in ("raw", "pin", "sym", "sym_pin")}
    out["alongside"] = {k: {"statistic": X[k]["statistic"], "I": X[k]["I"], "mean": float(np.mean(X[k]["I"])) if X[k]["n"] else None}
                        for k in ("raw", "sym", "sym_pin")}
    if c1_ref and c1_ref.get("_X"):
        r = c1_ref["_X"]["raw"]
        out["beta_minus_beta_C1"] = welch_diff(beta_origin(X["raw"]["I"], X["raw"]["Pv"]), se_origin(X["raw"]["I"], X["raw"]["Pv"]),
                                               X["raw"]["n"], beta_origin(r["I"], r["Pv"]), se_origin(r["I"], r["Pv"]), r["n"])
    out["_X"] = X
    out["df_label"] = df_label
    return out


def _label_small(blk: dict, consistent_rule: str) -> str:
    if blk.get("status"):
        return blk["status"]
    J = [r["J"] for r in blk["per_draw"]]
    ti = blk["mean_J"]
    if consistent_rule == "all_positive" and len(J) >= 3 and all(j > 0 for j in J):
        return "consistent with C1"
    if consistent_rule == "p_0.025" and blk["signflip"]["p"] <= 0.025:
        return "consistent with C1"
    if ti and ti.get("hi") is not None and ti["hi"] < 0:
        return "inconsistent"
    return "inconclusive"


C1DE_LABELS = ("sign-consistent with C1", "sign-inconsistent", "mixed")
C1DE_NOMINAL = ("descriptive label, not a test (decision (e), MF-6): the exact sign-flip p and the t-interval are nominal "
                "under independence; two draws of a base share 64-93 % of their modules, the margin 0.25 is not calibrated "
                "for these draws (no G0-7 probe), and the label is never a replication of C1 on a second target")


def _label_c1de(blk: dict) -> str:
    """C1-DE label (decision (e)): 'sign-consistent with C1' if every J_d > 0, 'sign-inconsistent' if every
    J_d < 0, otherwise 'mixed' (over the usable draws; a dropped draw is listed in draws_dropped)."""
    if blk.get("status"):
        return blk["status"]
    J = [r["J"] for r in blk.get("per_draw") or []]
    if J and all(j > 0 for j in J):
        return C1DE_LABELS[0]
    if J and all(j < 0 for j in J):
        return C1DE_LABELS[1]
    return C1DE_LABELS[2]


def block_b3(ctx: Ctx, c1: dict) -> dict:
    st = ctx.params.get("moe_b3 status and beta0_3") or {}
    b0 = st.get("beta0_3") if isinstance(st, dict) else None
    out = _small_block(ctx, "c1_b3", lambda b: b0 if b0 is not None else BETA_REF, "raw", "t df 2", c1)
    out.update(prereg="v1.3 sec. 4.1 (secondary)", freeze_status=st, beta0_3=b0)
    out["label"] = _label_small(out, "all_positive")
    lab = [r.get("label") for r in ctx.plan.get("runs", []) if r.get("role") == "c1_b3" and r.get("label")]
    if lab:
        out["run_label"] = sorted(set(lab))
    return out


def block_c1de(ctx: Ctx, c1: dict) -> dict:
    b0 = {"moe_b1": ctx.params.get("beta0_1"), "moe_b2": ctx.params.get("beta0_2")}
    out = _small_block(ctx, "c1de", lambda b: b0[b], "raw", "t df 5 pooled, df 2 per base", c1)
    out["prereg"] = "v1.3 sec. 4.12 (exploratory; outside the Holm family; p and intervals nominal, threat 9.1 item 15)"
    out["label"] = _label_c1de(out)
    out["label_rule"] = "all usable J_d > 0 / all < 0 / otherwise mixed"
    out["nominal"] = C1DE_NOMINAL
    # manifests: every draw, the ones not generated, pairwise module overlaps, matched log ratios
    mans = {}
    for key, ref in ((ctx.plan.get("blocks") or {}).get("_manifests") or {}).items():
        if not key.startswith("C1-DE:"):
            continue
        p = ctx.p(ref["path"])
        if not os.path.exists(p):
            mans[key] = {"status": "manifest missing"}
            continue
        h = sha256_file(p)
        m = ctx.json(p)
        ents = m.get("entries") or []
        rows = []
        for e in ents:
            gen = e.get("generator") or {}
            tl = bool(gen.get("time_limit_hit")) or any(a.get("time_limit_hit") or "time limit" in str(a.get("message", "")).lower()
                                                        for a in gen.get("attempts") or [])
            rows.append({"d": e.get("d"), "seed": e.get("seed"), "status": e.get("status"),
                         "attempts": len(gen.get("attempts") or []), "time_limit_hit": tl,
                         "solver": sorted({a.get("message", "")[:60] for a in gen.get("attempts") or []}),
                         "matched_log_ratios": {c: (v or {}).get("diff") for c, v in (e.get("matched_values") or {}).items()},
                         "P_d": (e.get("predicted") or {}).get("P")})
        ok = [e for e in ents if e.get("status") == "ok"]
        ov = []
        for e1_, e2_ in itertools.combinations(ok, 2):
            rec = {"d": [e1_["d"], e2_["d"]]}
            for side in ("A", "B"):
                s1 = {(l, int(m_)) for l, row in enumerate(e1_[side]) for m_ in row}
                s2 = {(l, int(m_)) for l, row in enumerate(e2_[side]) for m_ in row}
                rec[side] = len(s1 & s2) / max(len(s1), 1)
            ov.append(rec)
        planned = {int(r["d"]) for r in ctx.plan.get("runs", []) if r.get("role") == "c1de" and r.get("base") == m.get("base")}
        mans[key] = {"path": ref["path"], "sha256_recorded": ref.get("sha256"), "sha256_now": h,
                     "unchanged": h == ref.get("sha256"), "draws": rows, "not_generated": [r["d"] for r in rows if r["status"] != "ok"],
                     "planned_draws": sorted(planned), "pairwise_module_overlap": ov,
                     "time_limit_hits": [r["d"] for r in rows if r["time_limit_hit"]]}
    out["manifests"] = mans
    # R6: a draw whose solve hit the 600 s limit is not reproducible on another machine -> flagged
    out["time_limit_draws"] = sorted(f"{k.split(':', 1)[-1]}:d{d}" for k, v in mans.items() for d in v.get("time_limit_hits") or [])
    return out


def block_nestor_books(ctx: Ctx, g1rec: dict | None) -> dict:
    out = _small_block(ctx, "c1_nestor", lambda b: BETA_REF, "sym_pin", "t df 2", None)
    out["prereg"] = "v1.3 sec. 4.4 (exploratory)"
    out["label"] = _label_small(out, "all_positive")
    out["NESTORB_label"] = ctx.params.get("NESTORB_label")
    out["NESTORB_run"] = ctx.params.get("NESTORB_run")
    g9 = ((g1rec or {}).get("gates") or {}).get("G0-9L") or {}
    out["G0_9L"] = {k: g9.get(k) for k in ("outcome", "reason", "action")}
    return out


# ============================================================================= selpin decomposition (v1.3 sec. 4.6)
def block_selpin(ctx: Ctx, blocks: dict) -> dict:
    corp = (ctx.code, "math")
    out = {"prereg": "v1.3 sec. 4.6 (descriptive; endpoint decomposition, the order of pathways matters)"}
    for name, blk in blocks.items():
        X = (blk or {}).get("_X")
        if not X:
            continue
        draws = [{"base": r["base"], "d": r["d"], "A": r["A"], "B": r["B"]} for r in X["raw"]["draws"]]
        per_run, per_draw = [], defaultdict(lambda: defaultdict(list))
        for dr in draws:
            for s in ("A", "B"):
                for c in corp:
                    f, fs, fp = ctx.F(dr[s], c, "raw"), ctx.F(dr[s], c, "selpin"), ctx.F(dr[s], c, "pin")
                    if not (f and fs and fp):
                        continue
                    rec = {"run": dr[s], "corpus": c, "F": f["F"], "flips": f["F"] - fs["F"], "gate_values": fs["F"] - fp["F"],
                           "frozen_routing": fp["F"]}
                    if f["F"] != 0:
                        for k in ("flips", "gate_values", "frozen_routing"):
                            rec[f"share_{k}"] = rec[k] / f["F"]
                            per_draw[(dr["base"], dr["d"], c)][k].append(rec[f"share_{k}"])
                    per_run.append(rec)
        shares = {}
        for c in corp:
            shares[c] = {k: t_interval([float(np.mean(v[k])) for (b, d, cc), v in per_draw.items() if cc == c and v[k]])
                         for k in ("flips", "gate_values", "frozen_routing")}
        contrasts = {k: {"statistic": X[k]["statistic"], "I": X[k]["I"], "mean": t_interval(X[k]["I"])}
                     for k in ("raw", "selpin", "pin", "sym", "sym_selpin", "sym_pin") if k in X}
        out[name] = {"per_run": per_run, "shares_over_draws": shares, "crossover_contrasts": contrasts}
    return out


# ============================================================================= dose-response (v1.3 sec. 4.2, decision P-14)
def _gplan(ctx: Ctx, stage: str) -> tuple[dict, dict] | None:
    if stage in ctx._gplans:
        return ctx._gplans[stage]
    pp = ctx.p(f"{stage}/plan_{stage}.json")
    if not os.path.exists(pp):
        ctx._gplans[stage] = None
        return None
    gp = ctx.json(pp)
    sp = ctx.p(gp.get("state") or f"results/queue_state_{stage}.json")
    st = ctx.json(sp) if os.path.exists(sp) else {}
    ctx._gplans[stage] = (gp, st)
    return ctx._gplans[stage]


def _aa_units(ctx: Ctx, base: str = "moe_b1", pairs=(2, 3)) -> tuple[list[dict], list[str]]:
    """The books A/A pairs a = 2, 3 of MoE base 1 (G1 / G2 runs), R1 = manifest subset A, R2 = B, with their computed P
    (crossover.reg_predicted on the estimation half of the C1 tables of the base)."""
    units, notes = [], []
    c1d = [d for d in ctx.draws_of("c1") if d["base"] == base]
    src_entry = ctx.entry(c1d[0]) if c1d else None
    for stage in ("g1", "g2"):
        g = _gplan(ctx, stage)
        if not g:
            continue
        gp, st = g
        rr = defaultdict(dict)
        for r in gp.get("runs", []):
            if r.get("role") == "aa" and r.get("base") == base and r.get("target") == "books" and r.get("pair") in pairs:
                rr[r["pair"]][r.get("subset")] = r
        for a, sides in sorted(rr.items()):
            if {"R1", "R2"} - set(sides):
                notes.append(f"A/A pair {a}: R1/R2 not both planned")
                continue
            man = ctx.json(sides["R1"]["draw_file"]) if os.path.exists(ctx.p(sides["R1"]["draw_file"])) else None
            P = None
            if man and src_entry:
                e = dict(man, sources=src_entry.get("sources"), family=src_entry.get("family", "moe"))
                P = ctx.predicted(e, "c1")
            u = {"base": base, "d": f"aa{a}", "pair": a, "stage": stage, "P": P, "entry": man, "src": src_entry,
                 "files": {}, "status": {}}
            for sub, side in (("R1", "A"), ("R2", "B")):
                rid = sides[sub]["id"]
                s = st.get(rid) or {}
                root = ctx.p(s.get("root")) if s.get("root") else None
                u[side] = rid
                u["status"][side] = s.get("status", "pending")
                u["files"][rid] = {"root": root, "base": {c: ctx.p(p) for c, p in ((gp.get("base_eval_files") or {}).get(base) or {}).items()}}
            if P is None:
                notes.append(f"A/A pair {a}: computed P not available (manifest or tables missing)")
            units.append(u)
    return units, notes


def _aa_F(ctx: Ctx, u: dict, rid: str, c: str, kind: str) -> dict | None:
    f = u["files"][rid]
    if not f["root"]:
        return None
    files = {"base": f["base"].get(c), "ckpt": os.path.join(f["root"], "eval_main", f"ckpt__{c}.npz"),
             "reflect": os.path.join(f["root"], "eval_reflect", f"reflect__{c}.npz")}
    return ctx.F(rid, c, kind, files=files)


def block_dose(ctx: Ctx, c1: dict) -> dict:
    corp = (ctx.code, "math")
    out = {"prereg": "v1.3 sec. 4.2 (secondary; decision P-14: E[I|P] = b1 P + b3 P^3, no intercept)"}
    rec_p = ctx.p("results/dose/dose_record.json")
    if os.path.exists(rec_p):
        rec = ctx.json(rec_p)
        out["record"] = {"f_low": rec.get("f_low"), "f_grid": rec.get("f_grid"), "decision": rec.get("decision"),
                         "bases": {b: {"accepted": v.get("accepted"), "examined": v.get("examined"), "reason": v.get("reason")}
                                   for b, v in (rec.get("bases") or {}).items()}}
    dose, dropped, e1 = ctx.screen(ctx.draws_of("dose"))
    aa, notes = _aa_units(ctx)
    aa_ok = [u for u in aa if all(v == "done" for v in u["status"].values()) and u["P"] is not None]
    out["aa_notes"] = notes
    out["draws_dropped"] = dropped
    c1d, _, _ = ctx.screen(ctx.draws_of("c1"))
    units = [dict(u, level="AA", cell=("AA", u["base"])) for u in aa_ok] + \
            [dict(d, level="dose", cell=("dose", d["base"])) for d in dose] + \
            [dict(d, level="C1", cell=("C1", d["base"])) for d in c1d]
    aa_ids = {u["A"]: u for u in aa_ok}
    aa_ids.update({u["B"]: u for u in aa_ok})

    def Fget_k(kind):
        def f(rid, c):
            return _aa_F(ctx, aa_ids[rid], rid, c, kind) if rid in aa_ids else ctx.F(rid, c, kind)
        return f

    def ro_get(rid, c):
        if rid in aa_ids:
            u = aa_ids[rid]
            if not (u["entry"] and u["src"]):
                return None
            e = dict(u["entry"], sources=u["src"].get("sources"), family=u["src"].get("family", "moe"))
            return ctx.ro_table(e, "A" if u["A"] == rid else "B", c)
        return ctx.ro(rid, c)

    fits = {}
    for kind in ("raw", "sym"):
        X = ctx.cross(units, kind, corp, Fget=Fget_k(kind), ro_get=ro_get)
        levA = {u["A"]: u["level"] for u in units}           # run ids are unique; (base, d) is not across levels
        I, P = _arr(X["I"]), _arr(X["Pv"])
        L = [levA[a] for a in X["As"]]
        B = list(X["bases"])
        res = {"statistic": X["statistic"], "n": X["n"], "units": [{"level": l_, "base": b, "d": d, "I": i, "P": p}
                                                                  for l_, b, d, i, p in zip(L, B, X["ds"], X["I"], X["Pv"])],
               "dropped": X["dropped"]}
        if X["n"] >= 3:
            res["primary"] = ols_hc3(I, np.column_stack([P, P ** 3]), ["b1", "b3"])
            res["bootstrap"] = _dose_boot(I, P, L, B)
            pr = res["primary"]
            if pr and pr.get("coef") and pr["coef"]["b1"].get("lo") is not None:
                b1, b3 = pr["coef"]["b1"], pr["coef"]["b3"]
                excl = not (b1["lo"] <= 0.0 <= b1["hi"])
                rej = b3["p_two_sided"] is not None and b3["p_two_sided"] <= ALPHA
                res["label"] = ("non-linear dose-response" if rej else "linear dose-response") if excl else "no dose-response"
            ind = lambda b: np.array([1.0 if x == b else 0.0 for x in B])  # noqa: E731
            res["sensitivity_base_specific_b1"] = ols_hc3(I, np.column_stack([P * ind("moe_b1"), P * ind("moe_b2"), P ** 3]),
                                                          ["b1_moe_b1", "b1_moe_b2", "b3"])
            keep = np.array([l_ != "AA" for l_ in L])
            res["sensitivity_without_AA"] = ols_hc3(I[keep], np.column_stack([P[keep], P[keep] ** 3]), ["b1", "b3"])
        fits[kind] = res
    out["fits"] = fits
    out["label"] = fits["raw"].get("label")
    # pipeline-asymmetry check
    g2 = None
    gp = _gplan(ctx, "g2")
    rp = ctx.p("g2/gate_record_G2.json")
    if os.path.exists(rp):
        g2 = ctx.json(rp)
    sig = (((g2 or {}).get("gates") or {}).get("G2-1") or {}).get("values", {}).get("sigma_I")
    aa_I = [u for u in fits["raw"]["units"] if u["level"] == "AA"]
    rec_I = {}
    for a, v in ((((g2 or {}).get("gates") or {}).get("G2-1") or {}).get("record") or {}).get("pairs", {}).items():
        rec_I[int(a)] = v.get("I_AA")
    mI = float(np.mean([u["I"] for u in aa_I])) if aa_I else None
    out["pipeline_asymmetry_check"] = {"pairs": [{"pair": u["d"], "I": u["I"], "P": u["P"]} for u in aa_I],
                                       "mean_I": mI, "sigma_I_G2_1": sig,
                                       "mean_I_in_sigma_units": (mI / sig) if (mI is not None and sig) else None,
                                       "G2_record_I_AA": {k: rec_I.get(k) for k in (2, 3)},
                                       "note": "checks the pipeline; not part of the dose-response model; never confirmatory"}
    return out


def _dose_boot(I, P, L, B, n: int = N_BOOT, seed: int = SEED_DOSE) -> dict:
    """Stratified bootstrap (draws resampled within level x base; 10,000 resamples; seed 20261113): b1, b3 and
    kappa = beta_low / beta_high, the slopes between adjacent level means pooled over bases."""
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i, (l_, b) in enumerate(zip(L, B)):
        strata[(l_, b)].append(i)
    S = [np.array(v) for _, v in sorted(strata.items())]
    Larr = np.array(L)
    b1s, b3s, ks = np.empty(n), np.empty(n), np.empty(n)

    def kappa(ix):
        m = {}
        for lev in ("AA", "dose", "C1"):
            sel = ix[Larr[ix] == lev]
            if sel.size == 0:
                return np.nan
            m[lev] = (float(I[sel].mean()), float(P[sel].mean()))
        dl, dh = m["dose"][1] - m["AA"][1], m["C1"][1] - m["dose"][1]
        if dl == 0 or dh == 0:
            return np.nan
        bl, bh = (m["dose"][0] - m["AA"][0]) / dl, (m["C1"][0] - m["dose"][0]) / dh
        return bl / bh if bh != 0 else np.nan

    for r in range(n):
        ix = np.concatenate([s[rng.integers(0, s.size, s.size)] for s in S])
        X = np.column_stack([P[ix], P[ix] ** 3])
        try:
            b = np.linalg.lstsq(X, I[ix], rcond=None)[0]
            b1s[r], b3s[r] = b
        except np.linalg.LinAlgError:
            b1s[r] = b3s[r] = np.nan
        ks[r] = kappa(ix)
    k0 = kappa(np.arange(len(L)))
    return {"b1_ci95": pct_ci(b1s), "b3_ci95": pct_ci(b3s), "kappa": None if not np.isfinite(k0) else float(k0),
            "kappa_ci95": pct_ci(ks), "n_boot": n, "seed": seed, "strata": {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(strata.items())}}


# ============================================================================= assumption audit (v1.3 sec. 4.3)
def block_audit(ctx: Ctx) -> dict:
    out = {"prereg": "v1.3 sec. 4.3 (secondary; labels reported with C1; no decision depends on them)"}
    jobs = []
    qp = ctx.p(ctx.plan.get("queue") or "confirm/queue_confirm.json")
    if os.path.exists(qp):
        q = ctx.json(qp)
        jobs = [j for j in q.get("jobs", []) if j.get("kind") == "cmd" and "assumption_audit.py" in " ".join(j.get("argv", []))]
    if not jobs:
        out["status"] = "no audit job in the confirm queue"
        return out
    import assumption_audit as AU
    for j in jobs:
        a = j["argv"]
        base = a[a.index("--base") + 1] if "--base" in a else j["id"]
        od = ctx.p(a[a.index("--out") + 1]) if "--out" in a else None
        need = [os.path.join(od, f) for f in (AU.REAL_GROUPS, AU.REAL_WINDOWS)] if od else []
        if not od or not all(os.path.exists(f) for f in need) or ctx.status(j["id"]) != "done":
            out[base] = {"status": f"not available (job {j['id']}: {ctx.status(j['id'])})"}
            continue
        for f in need:
            ctx._read(f)
        rep = AU.analyze_realized(od)
        out[base] = {"B5": rep["B5"], "B6": rep["B6"], "A0": rep["A0"], "eps": rep["eps"], "corpora": rep["corpora"],
                     "bootstrap": rep["bootstrap"]}
    return out


# ============================================================================= M12-b (v1.3 sec. 4.5)
def block_m12b(ctx: Ctx) -> dict:
    out = {"prereg": "v1.3 sec. 4.5 (descriptive)",
           "statistic": "IT_d = [log F_DCLM - log F_code](German) - [log F_DCLM - log F_code](books); Delta_d = IT_learned - IT_rand"}
    runs = [r for r in ctx.plan.get("runs", []) if r.get("role") == "m12b"]
    if not runs:
        out["status"] = "not run"
        return out
    rows, deltas = [], []
    for r in sorted(runs, key=lambda x: int(x["d"])):
        g, b = r["id"], r.get("c1_partner")
        rec = {"d": r["d"], "german_run": g, "books_run": b, "haar": {"german": ctx.haar(g), "books": ctx.haar(b)}}
        rec["common_haar_draws"] = rec["haar"]["german"] is not None and rec["haar"]["german"] == rec["haar"]["books"]
        why = [f"{x}: {w}" for x in (g, b) for u, w in [ctx.usable(x)] if not u]
        if why:
            rec["dropped"] = why
            rows.append(rec)
            continue
        it = {}
        for kind in ("raw", "rand"):
            f = {(run, c): ctx.F(run, c, kind) for run in (g, b) for c in ("dclm", ctx.code)}
            if any(v is None or v["F"] <= 0 for v in f.values()):
                it[kind] = None
                continue
            lg = lambda run: math.log(f[(run, "dclm")]["F"]) - math.log(f[(run, ctx.code)]["F"])  # noqa: E731
            it[kind] = lg(g) - lg(b)
        rec["IT_learned"], rec["IT_rand"] = it.get("raw"), it.get("rand")
        if None not in (rec["IT_learned"], rec["IT_rand"]):
            rec["Delta"] = rec["IT_learned"] - rec["IT_rand"]
            deltas.append(rec["Delta"])
        rows.append(rec)
    out["per_draw"] = rows
    out["mean_Delta"] = t_interval(deltas) if deltas else None
    out["signflip_two_sided"] = signflip_p(deltas, "two-sided") if deltas else None
    same = bool(deltas) and (all(x > 0 for x in deltas) or all(x < 0 for x in deltas))
    ti = out["mean_Delta"]
    excl = bool(ti and ti.get("lo") is not None and not (ti["lo"] <= 0 <= ti["hi"]))
    out["sign_agreement"] = same
    out["label"] = "target-dependent" if (len(deltas) == 3 and same and excl) else "no evidence of target dependence"
    return out


# ============================================================================= forecast scoring (v1.3 sec. 5.7) and budget (sec. 9.2)
def block_forecasts(ctx: Ctx) -> dict:
    fc = ctx.freeze.get("forecasts") or {}
    csv_p, meta_p = ctx.p((fc.get("csv") or {}).get("path")), ctx.p((fc.get("meta") or {}).get("path"))
    up = ctx.p("results/forecast/forecast_units.json")
    out = {"prereg": "v1.3 sec. 5.7 (descriptive; all eight forecasters, no selection)"}
    if not (csv_p and meta_p and os.path.exists(csv_p) and os.path.exists(meta_p) and os.path.exists(up)):
        out["status"] = "forecast files or units missing"
        return out
    for f in (csv_p, meta_p, up):
        ctx._read(f)
    try:
        import forecasts as FC
        out["scores"] = FC.cmd_score(csv_p, meta_p, up, [ctx.state_path])
    except Exception as e:                                   # recorded, never hidden
        out["status"] = f"scoring failed: {type(e).__name__}: {e}"
    return out


def block_budget(ctx: Ctx) -> dict:
    planned = defaultdict(float)
    actual = defaultdict(float)
    n_done = defaultdict(int)
    for r in ctx.plan.get("estimates", []):
        blk = r.get("block", "?")
        planned[blk] += float(r.get("total_h") or 0.0)
        s = ctx.state.get(r.get("id")) or {}
        if s.get("hours") is not None:
            actual[blk] += float(s["hours"])
            n_done[blk] += 1
    return {"prereg": "v1.3 sec. 9.2 (planned against actual GPU-h per block)",
            "per_block": {b: {"planned_gpu_h": round(planned[b], 4), "actual_gpu_h": round(actual[b], 4),
                              "jobs_with_hours": n_done[b]} for b in sorted(planned)},
            "total": {"planned_gpu_h": round(sum(planned.values()), 4), "actual_gpu_h": round(sum(actual.values()), 4)},
            "cuts": {"applied": ctx.plan.get("cut_order_applied"), "restored": ctx.plan.get("cut_order_restored")},
            "envelope_extension": ctx.params.get("envelope extension (decision P-6; 0 if none)")}



# ============================================================================= EDGE (v1.3 sec. 4.11; descriptive)
_EPOCH = datetime(1970, 1, 1)
LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) - (\S+) - [A-Z]+ - (.*)$")
STEP_RE = re.compile(r"\bStep (\d+) \|")
HDR_RE = re.compile(r"\[pretrain\].*?([\d,]+) steps x ([\d,]+) tok/step")
QLOG_RE = re.compile(r"^\[queue\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (start|end) (\S+?)(?::| |$)")
EDGE_LIMITS = [
    "Hardware: a desktop consumer GPU (RTX 4080 SUPER, 16 GB) stands in for edge hardware; absolute numbers do not "
    "transfer to edge devices; only relative costs within this study are interpretable.",
    "Board power only: CPU, DRAM, storage and power-supply losses are not measured; the board draw includes idle power "
    "and any other process on the GPU.",
    "Scheduling: the pipeline runs at BelowNormal CPU priority, which can lengthen wall time; no power or clock limit is set.",
    "Sampling: 1 Hz; the driver's power reading is not calibrated against an external meter."]


def _sec(ts: str, fmt: str) -> float:
    return (datetime.strptime(ts, fmt) - _EPOCH).total_seconds()     # local wall clock as a naive time scale


def _clock(sec: float) -> str:
    return (_EPOCH + timedelta(seconds=float(sec))).strftime("%Y-%m-%d %H:%M:%S")


def read_power(paths: list[str]) -> dict:
    """Rows 'YYYY/MM/DD HH:MM:SS.mmm, <W> W, ..., <MiB> MiB, ...' without header; fields identified by their units."""
    t, w, m, layout, bad = [], [], [], None, 0
    for p in paths:
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 2:
                bad += bool(line.strip())          # a truncated row (the logger is still writing) is counted
                continue
            try:
                ts = _sec(parts[0], "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                bad += 1
                continue
            iw = next((i for i, x in enumerate(parts) if x.endswith(" W")), None)
            im = next((i for i, x in enumerate(parts) if x.endswith(" MiB")), None)
            try:
                wv = float(parts[iw][:-2]) if iw is not None else None
                mv = float(parts[im][:-4]) if im is not None else math.nan
            except ValueError:
                wv = None
            if wv is None:
                bad += 1
                continue
            t.append(ts)
            w.append(wv)
            m.append(mv)
            if layout is None:
                layout = {"fields": len(parts), "timestamp": 0, "power_W": iw, "memory_MiB": im,
                          "example": line.strip()[:120]}
    o = np.argsort(np.array(t), kind="stable")
    return {"t": np.array(t)[o], "w": np.array(w)[o], "m": np.array(m)[o], "layout": layout, "rows": len(t),
            "unparsed_rows": bad, "files": [os.path.basename(p) for p in paths]}


def run_log_path(logs: str, rid: str, root: str | None) -> str | None:
    """The training log of the attempt that wrote the state's root: run_queue.py names both after the attempt,
    <out_root>/m<micro>[c] and <logs>/<run>_m<micro>[c].log (c = activation checkpointing), so after an OOM fallback
    (NESTOR m4c -> m2c) the successful attempt is read; without a root the last-named log."""
    m = re.search(r"(m\d+c?)$", root or "")
    if m:
        return os.path.join(logs, f"{rid}_{m.group(1)}.log")
    cand = sorted(glob.glob(os.path.join(logs, f"{rid}_m*.log")))
    return cand[-1] if cand else None


def parse_run_log(path: str) -> dict:
    ts, steps, hdr = [], [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        mm = LOG_RE.match(line.rstrip("\n"))
        if mm:
            t = _sec(mm.group(1), "%Y-%m-%d %H:%M:%S")
            ts.append(t)
            sm = STEP_RE.search(mm.group(3))
            if sm:
                rec = {"step": int(sm.group(1)), "t": t}
                for k in ("train/sec_per_step", "train/mem_peak_alloc_gib", "train/mem_peak_reserved_gib"):
                    km = re.search(re.escape(k) + r":\s*([-\d.eE+]+)", mm.group(3))
                    if km:
                        rec[k] = float(km.group(1))
                steps.append(rec)
        elif hdr is None:
            hm = HDR_RE.search(line)
            if hm:
                hdr = {"steps": int(hm.group(1).replace(",", "")), "tok_per_step": int(hm.group(2).replace(",", ""))}
    return {"t_start": ts[0] if ts else None, "t_end": ts[-1] if ts else None, "monotone": bool(np.all(np.diff(ts) >= 0)) if ts else None,
            "steps": steps, "header": hdr, "lines": len(ts)}


def window_energy(pw: dict, t0: float, t1: float) -> dict:
    """Trapezoidal integral of board power over [t0, t1] with linear interpolation at both edges; gap check."""
    t, w, m = pw["t"], pw["w"], pw["m"]
    if t.size == 0 or t0 < t[0] or t1 > t[-1]:
        return {"ok": False, "reason": "window not inside the power log"}
    i0, i1 = int(np.searchsorted(t, t0, "right")) - 1, int(np.searchsorted(t, t1, "left"))
    tt, ww = t[i0:i1 + 1], w[i0:i1 + 1]
    gaps = np.diff(tt)
    if gaps.size and (np.any(gaps < 0) or np.any(gaps > 3000)):
        return {"ok": False, "reason": "clock change in the window"}
    if gaps.size and float(gaps.max()) > EDGE_GAP_S:
        return {"ok": False, "reason": f"power samples {float(gaps.max()):.1f} s apart (> {EDGE_GAP_S:.0f} s)"}
    w0 = float(np.interp(t0, tt, ww))
    w1 = float(np.interp(t1, tt, ww))
    inner = (tt > t0) & (tt < t1)
    xs = np.concatenate([[t0], tt[inner], [t1]])
    ys = np.concatenate([[w0], ww[inner], [w1]])
    E = float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)))            # J
    mm = m[i0:i1 + 1][(tt >= t0) & (tt <= t1)]
    return {"ok": True, "energy_J": E, "energy_Wh": E / 3600.0, "energy_kJ": E / 1000.0, "samples": int(inner.sum()),
            "mean_W": E / (t1 - t0) if t1 > t0 else None, "peak_W": float(ww[inner].max()) if inner.any() else max(w0, w1),
            "board_MiB_max": float(np.nanmax(mm)) if mm.size and np.any(np.isfinite(mm)) else None}


def _stages(ctx: Ctx) -> list[dict]:
    st = []
    for stage in ("g1", "g2"):
        g = _gplan(ctx, stage)
        if g:
            gp, state = g
            q = ctx.p(gp.get("queue") or f"{stage}/queue_{stage}.json")
            logs = (ctx.json(q).get("logs") if q and os.path.exists(q) else None) or f"logs/queue_{stage}"
            st.append({"stage": stage, "runs": {r["id"]: r for r in gp.get("runs", [])}, "state": state, "logs": ctx.p(logs)})
    fq = ctx.p("results/forecast/queue_forecast_inputs.json")
    if os.path.exists(fq):
        q = ctx.json(fq)
        sp = ctx.p(q.get("state") or "results/queue_state_forecast.json")
        runs = {}
        for j in q.get("jobs", []):
            if j.get("kind") == "train":
                b = re.match(r"ksd_(moe_b\d|nestor|dense\w*)_", j["id"])
                base = b.group(1) if b else None
                runs[j["id"]] = {"id": j["id"], "base": base, "arch": "nestor" if base == "nestor" else "moe",
                                 "role": "drive (E-ALL)", "subset_spec": True, "steps": j.get("steps")}
        st.append({"stage": "forecast", "runs": runs, "state": ctx.json(sp) if os.path.exists(sp) else {},
                   "logs": ctx.p(q.get("logs") or "logs/forecast"), "queue": q})
    qc = ctx.p(ctx.plan.get("queue") or "confirm/queue_confirm.json")
    logs = (ctx.json(qc).get("logs") if os.path.exists(qc) else None) or "logs/queue_confirm"
    st.append({"stage": "confirm", "runs": {r["id"]: r for r in ctx.plan.get("runs", [])}, "state": ctx.state,
               "logs": ctx.p(logs)})
    return st


def _run_type(r: dict) -> tuple[str, str]:
    arch = {"moe": "MoE", "nestor": "NESTOR", "dense": "dense"}.get(str(r.get("arch", "moe")).lower(), str(r.get("arch")))
    role = str(r.get("role", ""))
    if role in ("c2_sgd", "c2_muon"):
        kind = "C2 " + ("SGD" if role.endswith("sgd") else "Muon")
    elif role == "freeze":
        kind = "FULL minus a frozen group (C3 freeze run)"
    elif r.get("subset_spec") or role in ("aa", "c1", "dose", "m12b", "c1_nestor", "c1de", "c1_b3", "c5_pair") or "drive" in role:
        kind = "subset"
    else:
        kind = "FULL"
    return arch, kind


def _subset_size(root: str | None, arch: str) -> dict | None:
    p = os.path.join(root, "subset_manifest.json") if root else None
    if not p or not os.path.exists(p):
        return None
    sm = json.load(open(p, encoding="utf-8"))
    units = sm.get("units_trainable") or []
    out = {"updated_elements": (sm.get("elements") or {}).get("trainable"), "updated_units": sm.get("units_selected", len(units)),
           "elements_total": (sm.get("elements") or {}).get("total")}
    if arch == "NESTOR":
        out["private_experts"] = sum(1 for u in units if str(u).startswith("priv"))
        out["public_experts"] = sum(1 for u in units if str(u).startswith("pub"))
    return out


def block_edge(ctx: Ctx) -> dict:
    out = {"prereg": "v1.3 sec. 4.11 (descriptive; no hypothesis; outside the Holm family)", "limits": EDGE_LIMITS}
    pdir = ctx.p("logs/power")
    files = sorted(glob.glob(os.path.join(pdir, "gpu_power_*.csv")))
    readme = os.path.join(pdir, "README_power_log.txt")
    if not files:
        out["status"] = "no power log"
        return out
    for f in files:
        ctx._read(f)
    pw = read_power(files)
    out["power_log"] = {"files": pw["files"], "rows": pw["rows"], "unparsed_rows": pw["unparsed_rows"], "layout": pw["layout"],
                        "start": _clock(pw["t"][0]) if pw["rows"] else None,
                        "end": _clock(pw["t"][-1]) if pw["rows"] else None}
    if os.path.exists(readme):
        ctx._read(readme)
        lines = open(readme, encoding="utf-8", errors="replace").read().splitlines()
        out["power_log"]["readme"] = {"sha256": ctx.reads[readme],
                                      "column_lines": [x.strip() for x in lines if re.search(r"power\.draw|MiB|columns?", x, re.I)][:6]}
    else:
        out["power_log"]["readme"] = None
        ctx.warnings.append("EDGE: README_power_log.txt missing")
    runs, excluded = [], []
    for st in _stages(ctx):
        for rid, r in st["runs"].items():
            s = st["state"].get(rid) or {}
            if s.get("status") != "done" or r.get("kind") == "eval":
                continue
            if getattr(ctx, "mode", "full") == "core" and rid not in ctx.core_ids:     # P-16: EDGE of the core runs only
                continue
            arch, kind = _run_type(r)
            root = ctx.p(s.get("root")) if s.get("root") else None
            log = run_log_path(st["logs"], rid, root)
            if not log or not os.path.exists(log):
                excluded.append({"stage": st["stage"], "run": rid, "reason": "run log not found"})
                continue
            ctx._read(log)
            L = parse_run_log(log)
            if L["t_start"] is None:
                excluded.append({"stage": st["stage"], "run": rid, "reason": "no timestamped logger line"})
                continue
            if not L["monotone"]:
                excluded.append({"stage": st["stage"], "run": rid, "reason": "clock change in the run log"})
                continue
            if L["t_start"] < pw["t"][0]:
                excluded.append({"stage": st["stage"], "run": rid, "reason": "window starts before the power logger"})
                continue
            e = window_energy(pw, L["t_start"], L["t_end"])
            if not e["ok"]:
                excluded.append({"stage": st["stage"], "run": rid, "reason": e["reason"]})
                continue
            hdr = L["header"] or {}
            stp = L["steps"]
            rec = {"stage": st["stage"], "run": rid, "base": r.get("base"), "arch": arch, "kind": kind,
                   "third_base": r.get("base") == "moe_b3", "wall_s": L["t_end"] - L["t_start"],
                   **{k: e[k] for k in ("energy_Wh", "energy_kJ", "samples", "mean_W", "peak_W", "board_MiB_max")},
                   "vram_peak_reserved_gib": max((x.get("train/mem_peak_reserved_gib", -1) for x in stp), default=None),
                   "vram_peak_alloc_gib": max((x.get("train/mem_peak_alloc_gib", -1) for x in stp), default=None),
                   "steps": hdr.get("steps"), "tok_per_step": hdr.get("tok_per_step")}
            so = (ctx.runs.get(rid) or {}).get("stop_after")
            if so and hdr.get("steps"):                  # P-16: the optimizer steps actually run (stopped C2 runs)
                rec.update(steps=min(int(hdr["steps"]), int(so)), steps_registered=hdr["steps"])
            if len(stp) >= 2 and hdr.get("tok_per_step") and stp[-1]["t"] > stp[0]["t"]:
                rec["tokens_per_s_steady"] = (stp[-1]["step"] - stp[0]["step"]) * hdr["tok_per_step"] / (stp[-1]["t"] - stp[0]["t"])
            if rec["steps"] and hdr.get("tok_per_step") and rec["wall_s"] > 0:
                rec["tokens_per_s_end_to_end"] = rec["steps"] * hdr["tok_per_step"] / rec["wall_s"]
            if kind != "FULL":
                ss = _subset_size(root, arch)
                if ss:
                    rec.update(ss)
                rec["bitwise_audit_ok"] = s.get("audit_ok")
            runs.append(rec)
    out["runs"] = runs
    out["excluded"] = excluded
    # medians and ranges per run type (cells with fewer than three runs show their values)
    meas = ("wall_s", "energy_Wh", "energy_kJ", "mean_W", "peak_W", "vram_peak_reserved_gib", "vram_peak_alloc_gib",
            "board_MiB_max", "tokens_per_s_steady", "tokens_per_s_end_to_end", "updated_elements", "updated_units")
    types = defaultdict(list)
    for rr in runs:
        types[f"{rr['arch']}|{rr['kind']}"].append(rr)
    summ = {}
    for k, rr in sorted(types.items()):
        cell = {"n": len(rr)}
        for mname in meas:
            v = [x[mname] for x in rr if x.get(mname) is not None]
            if not v:
                continue
            cell[mname] = {"values": v} if len(v) < 3 else {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
        summ[k] = cell
    out["by_type"] = summ
    out["pre_update_check"] = (_edge_probes(ctx, pw, runs) if getattr(ctx, "mode", "full") != "core" else
                               {"status": "deferred to the full analysis (P-16: the core reports the EDGE of the core runs)"})
    return out


def _edge_probes(ctx: Ctx, pw: dict, runs: list[dict]) -> dict:
    """theta_0 probe units (sec. 5.3): the probe script logs per-unit seconds without timestamps, so the window is the
    probe batch (the queue's timestamped start / end lines of forecast_probe_<base>) divided by its number of units
    (sec. 4.11 rule for per-batch lines); the per-unit seconds are reported alongside."""
    fq = ctx.p("results/forecast/queue_forecast_inputs.json")
    up = ctx.p("results/forecast/forecast_units.json")
    if not (os.path.exists(fq) and os.path.exists(up)):
        return {"status": "forecast queue or units missing"}
    q = ctx.json(fq)
    U = ctx.json(up)
    qlog = os.path.join(ctx.p(q.get("logs") or "logs/forecast"), "queue.log")
    if not os.path.exists(qlog):
        return {"status": f"{ctx.rel(qlog)} missing"}
    ctx._read(qlog)
    se = defaultdict(dict)
    for line in open(qlog, encoding="utf-8", errors="replace"):
        mm = QLOG_RE.match(line.strip())
        if mm and mm.group(3).startswith("forecast_probe_"):
            se[mm.group(3)][mm.group(2)] = _sec(mm.group(1), "%Y-%m-%d %H:%M:%S")
    out = {"method": "batch window / units of the batch (per-unit log lines carry no timestamp)", "per_base": {}}
    pdir = (U.get("probe") or {}).get("dir")
    for jid, t in sorted(se.items()):
        base = jid[len("forecast_probe_"):]
        n = sum(1 for u in U.get("units", []) if u.get("base") == base)
        rec = {"job": jid, "units": n}
        if "start" not in t or "end" not in t or n == 0:
            rec["status"] = "start / end line or units missing"
            out["per_base"][base] = rec
            continue
        e = window_energy(pw, t["start"], t["end"])
        rec["batch_wall_s"] = t["end"] - t["start"]
        rec["wall_s_per_unit"] = rec["batch_wall_s"] / n
        if e["ok"]:
            rec["energy_Wh_per_unit"] = e["energy_Wh"] / n
        else:
            rec["energy_excluded"] = e["reason"]
        secs = []
        for u in U.get("units", []):
            f = os.path.join(ctx.p(pdir), f"probe_u{int(u['index']):03d}.json") if pdir else None
            if u.get("base") == base and f and os.path.exists(f):
                v = json.load(open(f, encoding="utf-8")).get("seconds")
                if v is not None:
                    secs.append(float(v))
        rec["script_seconds_per_unit_median"] = float(np.median(secs)) if secs else None
        arch = "NESTOR" if base == "nestor" else "MoE"
        ref = [r for r in runs if r["arch"] == arch and r["kind"] == "subset" and r["stage"] != "forecast"]
        if ref:
            mw = float(np.median([r["wall_s"] for r in ref]))
            rec["ratio_wall_vs_median_run"] = rec["wall_s_per_unit"] / mw
            me = [r["energy_Wh"] for r in ref]
            if "energy_Wh_per_unit" in rec:
                rec["ratio_energy_vs_median_run"] = rec["energy_Wh_per_unit"] / float(np.median(me))
            rec["reference"] = f"median over {len(ref)} {arch} subset adaptation runs (training window)"
        out["per_base"][base] = rec
    return out


# ============================================================================= report
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return _fin(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False)


def _f(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.{nd}g}"
    return str(x)


def _ci(e):
    if not e:
        return "NA"
    return f"{_f(e.get('beta_hat'))} [{_f(e.get('lo'))}, {_f(e.get('hi'))}]"


def markdown(doc: dict) -> str:
    L = [f"# Confirmatory analysis ({doc['format']})", "",
         f"Prereg: {doc['prereg']}  ", f"FREEZE.json verification: {doc['freeze_verification'].get('status')}  ",
         f"Created (UTC): {doc['created_utc']}  ", ""]
    c1, d1 = doc["C1"], doc["decisions"]["C1"]
    L += ["## C1 (books; confirmatory)", "", f"**Decision: {d1['label']}**" + (f" -- {'; '.join(d1['reasons'])}" if d1["reasons"] else ""), ""]
    if c1.get("test"):
        L += [f"D = {c1['D']} draws, statistic {c1['statistic']}; mean J = {_f(c1['test']['mean'])}; exact one-sided "
              f"sign-flip p = {_f(c1['test']['p'])} ({c1['test']['count']} of {c1['test']['vectors']}); Holm-adjusted p = "
              f"{_f(doc['holm']['adjusted'].get('C1'))}", ""]
        L += ["| base | mean J | n | margin-limited |", "|---|---|---|---|"]
        for b, v in c1["mean_J_per_base"].items():
            L.append(f"| {b} | {_f(v['mean_J'])} | {v['n']} | {(c1.get('margin_limited') or {}).get(b)} |")
        L += ["", "| estimator | overall | " + " | ".join(sorted(c1["mean_J_per_base"])) + " |",
              "|---|---|" + "---|" * len(c1["mean_J_per_base"])]
        for k in ("raw", "pin", "sym", "sym_pin", "rand"):
            e = c1["estimators"][k]
            L.append(f"| beta-hat_{k} | {_ci(e['overall'])} | " + " | ".join(_ci(e["per_base"].get(b)) for b in sorted(c1["mean_J_per_base"])) + " |")
        if d1.get("three_statistic_table"):
            L += ["", f"Three-statistic table: {d1['three_statistic_table']['label']}"]
        L += ["", f"Out-of-domain cells: {', '.join(d1['out_of_domain_cells']) or 'none'}; interpretation: "
              f"{'; '.join(d1['interpretation']) or 'none'}", ""]
    c2, d2 = doc["C2"], doc["decisions"]["C2"]
    L += ["## C2", "", f"**{d2['label']}**" + (f" -- {'; '.join(d2['reasons'])}" if d2.get("reasons") else ""), ""]
    if c2.get("test"):
        L += [f"s* = {c2['s_star']}, D = {c2['D']}, mean J2 = {_f(c2['test']['mean'])}, p = {_f(c2['test']['p'])}; "
              f"Delta-e = {_ci(c2.get('Delta_e'))}; manipulation check mean D_SGD/Delta_Q(mean) = "
              f"{_f(c2['manipulation_check']['mean_D_SGD_over_DQ_mean'])}", ""]
    for name, key in (("Third base (4.1)", "B3"), ("C1-DE (4.12)", "C1_DE"), ("NESTOR-books (4.4)", "NESTOR_books")):
        b = doc["blocks"].get(key)
        if b is None or b.get("status") == DEFERRED_STATUS:      # P-16: not computed by the core analysis
            continue
        L += [f"## {name}", "", f"Label: **{b.get('label')}**", ""]
        if b.get("per_draw"):
            L += [f"mean J = {_f(b['mean_J']['mean'])} [{_f(b['mean_J'].get('lo'))}, {_f(b['mean_J'].get('hi'))}] (t); "
                  f"sign-flip p = {_f(b['signflip']['p'])}; beta-hat = {_ci(b['estimators']['raw']['overall'])}; "
                  f"beta-hat_pin = {_ci(b['estimators']['pin']['overall'])}; beta-hat_sym = {_ci(b['estimators']['sym']['overall'])}", ""]
    dz = doc["blocks"].get("dose") or {}
    if "dose" in doc["blocks"] and dz.get("status") != DEFERRED_STATUS:
        L += ["## Dose-response (4.2)", "", f"Label (raw I): **{dz.get('label')}**", ""]
    pr = ((dz.get("fits") or {}).get("raw") or {}).get("primary")
    if pr and pr.get("coef"):
        L += [f"b1 = {_f(pr['coef']['b1']['b'])} [{_f(pr['coef']['b1']['lo'])}, {_f(pr['coef']['b1']['hi'])}] (HC3); "
              f"b3 = {_f(pr['coef']['b3']['b'])}, p = {_f(pr['coef']['b3']['p_two_sided'])}; kappa = "
              f"{_f((dz['fits']['raw'].get('bootstrap') or {}).get('kappa'))}", ""]
    if "M12b" in doc["blocks"] and doc["blocks"]["M12b"].get("status") != DEFERRED_STATUS:
        m = doc["blocks"]["M12b"]
        L += ["## M12-b (4.5)", "", f"Label: **{m.get('label')}**; mean Delta = {_f((m.get('mean_Delta') or {}).get('mean'))}", ""]
    au = doc["blocks"]["audit"]
    if au.get("status") != DEFERRED_STATUS:                  # P-16 revised: the audit is analysed in full mode only
        L += ["## Assumption audit (4.3)", ""]
    for b, v in (au.items() if au.get("status") != DEFERRED_STATUS else ()):
        if isinstance(v, dict) and "B5" in v:
            b6 = ", ".join(s + ": " + str(x.get("label")) for s, x in v["B6"].items())
            a0 = ", ".join(c + ": " + str(x.get("label")) for c, x in v["A0"].items())
            L.append(f"- {b}: B5 {v['B5'].get('label')} ({_f(v['B5'].get('interaction_share'))}); B6 {b6}; A0 {a0}")
        elif isinstance(v, dict):
            L.append(f"- {b}: {v.get('status')}")
    ed = doc["blocks"]["EDGE"]
    L += ["", "## EDGE (4.11)", "", f"Covered runs: {len(ed.get('runs') or [])}; excluded: {len(ed.get('excluded') or [])}", ""]
    for k, v in (ed.get("by_type") or {}).items():
        L.append(f"- {k}: n = {v['n']}; energy Wh {v.get('energy_Wh')}; wall s {v.get('wall_s')}")
    L += ["", "Limits: " + " ".join(ed.get("limits") or []), ""]
    import analyze_confirm_ext as X
    L += X.markdown(doc)
    if doc.get("mode") == "core":                  # P-16 staged analysis
        co = doc.get("core") or {}
        L += ["## Core analysis (P-16; prereg v1.3 version 8 sec. 3.5)", "",
              f"Core marker {co.get('marker', {}).get('path')} (sha256 {co.get('marker', {}).get('sha256')}); core blocks "
              f"{', '.join(co.get('core_blocks') or [])}; {co.get('core_runs')} core runs. The confirmatory decisions are "
              f"final; C4, C1 matched gain and bits per byte are secondary and descriptive.", "",
              "Deferred to the full analysis after the queue: " + ", ".join(co.get("deferred_to_full_analysis") or []), ""]
    L += ["## Warnings", ""]
    L += [f"- {w}" for w in doc["warnings"]] or ["- none"]
    return "\n".join(L) + "\n"


def run_all(ctx: Ctx) -> dict:
    c1 = block_c1(ctx)
    c2 = block_c2(ctx)
    fam = list(ctx.params.get("holm_family") or ["C1"])
    tests = {"C1": c1.get("test"), "C2": c2.get("test")}
    # the family is fixed at the freeze: a member without a test statistic enters Holm with p = 1 (m never shrinks)
    pv = {k: (float(tests[k]["p"]) if tests.get(k) else 1.0) for k in fam}
    adj = holm(pv)
    holm_doc = {"family": fam, "m": len(fam), "p": pv, "adjusted": adj,
                "entered_with_p_1": [k for k in fam if not tests.get(k)],      # MF-1 / decision (b)
                "first_step_alpha": 0.05 / max(len(fam), 1), "FWER": 0.05,
                "m_freeze": ctx.params.get("m")}
    if ctx.params.get("m") is not None and int(ctx.params["m"]) != len(fam):
        ctx.warnings.append(f"holm_family {fam} and m = {ctx.params['m']} differ in FREEZE.json")
    import analyze_confirm_ext as X                         # blocks (hash-list entry 2.17)
    core = None
    if getattr(ctx, "mode", "full") == "core":
        # P-16 (prereg v1.3 v8 sec. 3.5): the confirmatory analysis of the core and its three secondary analyses; the
        # secondary blocks are called directly so that a firewall breach is raised, never recorded as a failed block
        blocks = {"budget": block_budget(ctx), "EDGE": block_edge(ctx)}     # P-16 revised: B3 and audit in full mode only
        for name, fn in X.BLOCKS:
            if name in CORE_SECONDARY + CORE_EXPLORATORY:
                try:
                    blocks[name] = fn(ctx, sys.modules[__name__])
                except FirewallError:
                    raise
                except Exception as e:                       # recorded, never hidden
                    blocks[name] = {"status": f"failed: {type(e).__name__}: {e}"}
                    ctx.warnings.append(f"block {name} failed: {type(e).__name__}: {e}")
        mk = ctx.p(ctx.core_marker)
        mdoc = ctx.json(mk)
        core = {"rule": "P-16 staged analysis (prereg v1.3 version 8 sec. 3.5): core files only; decisions final",
                "marker": {"path": ctx.rel(mk), "sha256": ctx.reads.get(mk), "created_utc": mdoc.get("created_utc"),
                           "n_jobs": mdoc.get("n_jobs"), "n_done": mdoc.get("n_done"), "not_done": mdoc.get("not_done")},
                "core_blocks": list(ctx.core_blocks), "core_runs": len(ctx.core_ids), "secondary": list(CORE_SECONDARY),
                "exploratory": list(CORE_EXPLORATORY), "deferred_to_full_analysis": list(CORE_DEFERRED)}
        for name in CORE_DEFERRED:                       # the analysis_confirm/1 layout keeps every block name
            if name not in ("SEC", "DCLM-CONT"):
                blocks.setdefault(name, {"status": DEFERRED_STATUS})
    else:
        g1rec = None
        g1p = ctx.p("g1/gate_record_G1.json")
        if os.path.exists(g1p):
            g1rec = ctx.json(g1p)
        blocks = {"B3": block_b3(ctx, c1), "C1_DE": block_c1de(ctx, c1), "NESTOR_books": block_nestor_books(ctx, g1rec)}
        blocks["selpin"] = block_selpin(ctx, {"C1": c1, "B3": blocks["B3"], "C1_DE": blocks["C1_DE"], "NESTOR_books": blocks["NESTOR_books"]})
        blocks["dose"] = block_dose(ctx, c1)
        blocks["audit"] = block_audit(ctx)
        blocks["M12b"] = block_m12b(ctx)
        blocks["forecasts"] = block_forecasts(ctx)
        blocks["budget"] = block_budget(ctx)
        blocks["EDGE"] = block_edge(ctx)
        blocks.update(X.run_ext(ctx, sys.modules[__name__]))
    decisions = {"C1": decide_c1(c1, adj.get("C1")), "C2": decide_c2(c2, adj.get("C2"))}
    doc = {"format": FORMAT, "mode": getattr(ctx, "mode", "full"), "core": core,
           "prereg": "preregistration v1.3 version 8 (v1.2 secs. 5, 8, 9, 13 where kept)",
           "freeze": {"path": ctx.rel(ctx.freeze_path), "sha256_record": ctx.freeze.get("sha256"),
                      "prereg_ref": ctx.freeze.get("prereg")},
           "freeze_verification": ctx.verify, "parameters": ctx.params, "seeds": {"draw_bootstrap": SEED_BOOT,
           "dose_bootstrap": SEED_DOSE, "monte_carlo": f"{SEED_MC} (unused: exact enumeration)", "n_boot": N_BOOT},
           "C1": c1, "C2": c2, "holm": holm_doc, "decisions": decisions, "blocks": blocks,
           "analysed_elsewhere": X.POINTERS,
           "not_computable_from_registered_files": X.not_computable(blocks) + list(c2.get("not_computable_P16") or []),
           "format_ext": X.FORMAT_EXT,
           "E1": {rid: v for rid, v in sorted(ctx.e1.items()) if v.get("diverged")},
           "missing_files": sorted(set(ctx.missing)), "warnings": ctx.warnings,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    doc["inputs"] = {ctx.rel(p): h for p, h in sorted(ctx.reads.items())}
    return _clean(doc)


def write_outputs(doc: dict, out_dir: str, stem: str = "analysis_confirm") -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    body = {k: v for k, v in doc.items() if k not in ("created_utc", "sha256")}
    doc["sha256"] = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()
    jp = os.path.join(out_dir, f"{stem}.json")
    with open(jp + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical(doc))
    os.replace(jp + ".tmp", jp)
    with open(jp + ".sha256", "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{sha256_file(jp)}  {stem}.json\n")
    mp = os.path.join(out_dir, f"{stem}.md")
    with open(mp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(doc))
    return jp, mp


# ============================================================================= self-test (known answers)
def selftest() -> int:
    fails = []
    rng = np.random.default_rng(7)
    for D in (1, 3, 6, 10):
        for _ in range(10):
            x = np.round(rng.normal(0.1, 0.3, D), 2)
            T = x.mean()
            cnt = sum(1 for s in itertools.product((1, -1), repeat=D) if float(np.mean(np.array(s) * x)) >= T - 1e-12)
            if signflip_p(x)["count"] != cnt:
                fails.append(f"sign-flip count D={D}")
    if signflip_p([0.5, 0.4, 0.3, 0.2, 0.1, 0.6])["p"] != 1 / 64 or signflip_p([0.5, 0.4, 0.3, 0.2, -0.1, 0.6])["p"] != 2 / 64:
        fails.append("sign-flip known answers")
    I = [0.31, 0.12, 0.45, 0.22, 0.05, 0.38, 0.27, 0.19]
    P = [0.6, 0.4, 0.8, 0.5, 0.3, 0.7, 0.55, 0.45]
    ci = ci_invert(I, P)
    if not (abs(ci["lo"] - 0.325) < 1e-6 and abs(ci["hi"] - 0.5365853658) < 1e-6 and ci["convex"]):
        fails.append(f"test inversion {ci['lo']} {ci['hi']}")
    if ci_invert(I[:3], P[:3])["bounded"]:
        fails.append("D = 3 must be unbounded")
    if holm({"C1": 0.03, "C2": 0.01}) != {"C2": 0.02, "C1": 0.03}:
        fails.append("Holm")
    y = np.array([0.1, 0.2, 0.35, 0.5, 0.42, 0.61, 0.66, 0.9])
    p = np.array([0.0, 0.1, 0.3, 0.45, 0.4, 0.6, 0.62, 0.95])
    X = np.column_stack([p, p ** 3])
    r = ols_hc3(y, X, ["b1", "b3"])
    Xi = np.linalg.inv(X.T @ X)
    b = Xi @ X.T @ y
    e = y - X @ b
    h = np.array([X[i] @ Xi @ X[i] for i in range(len(y))])
    V = Xi @ sum(np.outer(X[i], X[i]) * (e[i] / (1 - h[i])) ** 2 for i in range(len(y))) @ Xi
    if abs(r["coef"]["b1"]["se_hc3"] - math.sqrt(V[0, 0])) > 1e-12 or abs(r["coef"]["b3"]["b"] - b[1]) > 1e-12:
        fails.append("HC3")
    print(json.dumps({"selftest": "FAIL" if fails else "ok", "fails": fails}))
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="confirmatory and secondary analyses after the confirm queue (prereg v1.3 v6)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", default=".")
    r.add_argument("--freeze", default="FREEZE.json")
    r.add_argument("--out", default=None, help="default results/analysis (full) or results/analysis_core (core)")
    r.add_argument("--mode", choices=["full", "core"], default="full",
                   help="P-16 (prereg v1.3 v8 sec. 3.5): core = once, when the core marker exists, on core files only; "
                        "full = after the whole queue")
    r.add_argument("--no-verify", action="store_true", help="skip chain6b_util.freeze_verify (tests only)")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    root = os.path.abspath(a.root)
    os.chdir(root)
    ctx = Ctx(root, a.freeze, verify=not a.no_verify, mode=a.mode)
    if ctx.verify.get("status") == "FAILED":
        print("[analyze_confirm] FREEZE.json verification FAILED: " + "; ".join(ctx.verify.get("log", [])[-5:]))
        return 23
    stem = "analysis_confirm"                    # addendum 4: the core output keeps the analysis_confirm/1 layout and name
    out_dir = ctx.p(a.out or (CORE_OUT if a.mode == "core" else os.path.join("results", "analysis")))
    if a.mode == "core":
        mk = ctx.p(ctx.core_marker)
        if not os.path.exists(mk):
            print(f"[analyze_confirm] core analysis refused: {ctx.rel(mk)} missing (P-16: the core analysis runs once, "
                  f"when the last core job has finished)")
            return 30
        if os.path.exists(os.path.join(out_dir, f"{stem}.json")):
            print(f"[analyze_confirm] core analysis refused: {ctx.rel(os.path.join(out_dir, stem + '.json'))} exists "
                  f"(the core analysis runs once; its output is never overwritten)")
            return 31
        # MF-P2 (2026-11-15): never on an incomplete core -- every run of the core blocks must be 'done'
        # (a core job that failed for an infrastructure reason is re-run with the same seeds first, E2)
        nd = sorted((rid, ctx.status(rid)) for rid in ctx.core_ids if ctx.status(rid) != "done")
        if nd:
            print(f"[analyze_confirm] core analysis refused: {len(nd)} core run(s) not done: "
                  + ", ".join(f"{r} ({s})" for r, s in nd[:10]) + (" ..." if len(nd) > 10 else "")
                  + " (re-run them with the same seeds first, E2); nothing written")
            return 33
    try:
        doc = run_all(ctx)
    except FirewallError as e:
        print(f"[analyze_confirm] FIREWALL: {e}; nothing written")
        return 32
    jp, mp = write_outputs(doc, out_dir, stem)
    d = doc["decisions"]
    print(f"[analyze_confirm] C1: {d['C1']['label']} | C2: {d['C2']['label']} | -> {ctx.rel(jp)}, {ctx.rel(mp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
