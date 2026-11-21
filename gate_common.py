"""Torch-free helpers shared by g1_gate.py and g2_gate.py (prereg v1.1 secs. 5, 6, 8.1, 13; memos v4-v6).

Nothing here imports torch: the gate scripts run in a torch-free process (on the host with the conda
Library\\bin PATH prefix, host fact 1 of the design).

  forgetting(base, run)      F = mean_w [CE_run(w) - CE_0(w)] over the paired evaluation windows (sec. 5),
                             SE = sd_w / sqrt(W); windows are paired by grid index and their token hashes must agree
  criterion(F, SE)           sub-signal rule of G1-2/G1-3/G2-3/G2-4: F >= 0.005 nats AND F >= 10 SE, with
                             r = min(F / 0.005, F / (10 SE)) (G1-2 fallback ladder)
  e1_check(steps)            sec. 13 E1 divergence rule from train_steps.jsonl (adapt_extras v4 logs the loss)
  aa_statistic(...)          I_AA of an A/A pair with the sec. 8.1 / E5 log-vs-normalized-difference rule
  lookup_power(...)          signflip_lookup_v1.csv power at (D, delta, alpha), linear in delta
  finalize_record(...)       canonical JSON + sha256 (embedded and <file>.sha256)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from typing import Any, Iterable

import numpy as np

THR_SUB = 0.005          # nats, sub-signal threshold (G1-2, G1-3, G2-3, G2-4)
Z_SUB = 10.0             # ... and >= 10 SE
THR_FULL = 0.010         # nats, G1-1 mean F_DCLM over the 4 MoE FULL runs
Z_RESOLVED = 3.0         # E5: F >= 3 SE is "resolved" (log statistic) for a whole test
UCL80_FACTOR = 1.7276    # sqrt(3 / chi2_{0.20, 3}) (G2-1)
EMA_ALPHA = 0.1          # E1: EMA(0.1) of the training loss
E1_RISE = 0.5            # E1: nats above the step-1 loss after warm-up
POWER_TARGET = 0.80


# ============================================================================= evaluation results
def load_eval(path: str) -> dict:
    """An eval_windows result (.npz): per-window CE sums, token counts, grid indices, token hashes, meta."""

    z = np.load(path, allow_pickle=False)
    need = ("index", "hash", "ce_sum", "n_tok")
    miss = [k for k in need if k not in z.files]
    if miss:
        raise ValueError(f"{path}: missing arrays {miss}")
    out = {k: np.asarray(z[k]) for k in need}
    out["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    out["path"] = os.path.abspath(path)
    return out


def _paired(base: dict, run: dict) -> tuple[np.ndarray, np.ndarray]:
    ib = {int(i): k for k, i in enumerate(base["index"])}
    ir = {int(i): k for k, i in enumerate(run["index"])}
    if set(ib) != set(ir):
        raise ValueError(f"window grids differ: {len(set(ib) ^ set(ir))} unpaired windows "
                         f"({base.get('path')} vs {run.get('path')})")
    common = sorted(ib)
    kb, kr = np.array([ib[i] for i in common]), np.array([ir[i] for i in common])
    if not np.array_equal(base["hash"][kb], run["hash"][kr]):
        bad = int((base["hash"][kb] != run["hash"][kr]).sum())
        raise ValueError(f"{bad} paired windows have different token content")
    if not np.array_equal(base["n_tok"][kb], run["n_tok"][kr]):
        raise ValueError("paired windows have different token counts")
    return base["ce_sum"][kb] / base["n_tok"][kb], run["ce_sum"][kr] / run["n_tok"][kr]


def forgetting(base: dict, run: dict) -> dict:
    """Registered F_{r,c} = (1/W) sum_w [CE_r(w) - CE_0(w)], SE = sd_w / sqrt(W) (prereg sec. 5)."""

    cb, cr = _paired(base, run)
    d = (cr - cb).astype(np.float64)
    W = int(d.size)
    F = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(W)) if W > 1 else float("nan")
    return {"F": F, "SE": se, "W": W, "z": F / se if se and se > 0 else float("nan"),
            "ce_base": float(cb.mean()), "ce_run": float(cr.mean())}


def gain(base: dict, run: dict) -> dict:
    """Target gain G = mean_w [CE_0(w) - CE_r(w)] (sec. 5) with its window SE."""

    f = forgetting(base, run)
    return {"G": -f["F"], "SE": f["SE"], "W": f["W"], "ce_base": f["ce_base"], "ce_run": f["ce_run"]}


def criterion(F: float, se: float, thr: float = THR_SUB, z: float = Z_SUB) -> dict:
    """F >= thr AND F >= z SE; r = min(F/thr, F/(z SE)) (the G1-2 ratio; r >= 1 iff pass)."""

    if not (math.isfinite(F) and math.isfinite(se)):
        return {"pass": False, "r": float("nan"), "thr": thr, "z": z}
    r = min(F / thr, F / (z * se)) if se > 0 else F / thr
    return {"pass": bool(F >= thr and F >= z * se), "r": float(r), "thr": thr, "z": z}


def resolved(F: float, se: float) -> bool:
    return bool(math.isfinite(F) and math.isfinite(se) and F >= Z_RESOLVED * se and F > 0)


# ============================================================================= E1 divergence (sec. 13)
def read_train_steps(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: int(r["step"]))
    return recs


def _loss_of(r: dict) -> float | None:
    for k in ("ce", "loss"):          # unweighted cross-entropy first (dose arms weight the total)
        v = r.get(k)
        if v is not None:
            return float(v)
    return None


def e1_check(recs: list[dict], warmup: int) -> dict:
    """E1: NaN/Inf loss; EMA(0.1) training loss > 0.5 nats above its step-1 value at any step after warm-up;
    final training loss >= initial (step 1). Uses the per-step loss logged by adapt_extras v4 ("ce", else "loss")."""

    losses = [(int(r["step"]), _loss_of(r)) for r in recs]
    if not losses or any(v is None for _, v in losses):
        return {"status": "no_loss_logged", "diverged": None, "reasons": ["train_steps.jsonl has no loss field"]}
    steps = [s for s, _ in losses]
    vals = np.array([v for _, v in losses], dtype=np.float64)
    reasons = []
    if not np.all(np.isfinite(vals)):
        reasons.append(f"non-finite loss at step {steps[int(np.flatnonzero(~np.isfinite(vals))[0])]}")
    l1 = float(vals[0])
    ema, ema_max, ema_max_step = None, -math.inf, None
    for s, v in zip(steps, vals):
        if not math.isfinite(v):
            continue
        ema = v if ema is None else EMA_ALPHA * v + (1 - EMA_ALPHA) * ema
        if s > warmup and ema > ema_max:
            ema_max, ema_max_step = ema, s
    if ema_max_step is not None and ema_max > l1 + E1_RISE:
        reasons.append(f"EMA(0.1) loss {ema_max:.4f} at step {ema_max_step} > step-1 loss {l1:.4f} + {E1_RISE}")
    final = float(vals[-1])
    if math.isfinite(final) and final >= l1:
        reasons.append(f"final loss {final:.4f} (step {steps[-1]}) >= step-1 loss {l1:.4f}")
    return {"status": "checked", "diverged": bool(reasons), "reasons": reasons, "step1_loss": l1,
            "final_loss": final, "final_step": steps[-1], "ema_max_after_warmup": None if ema_max_step is None else ema_max,
            "steps_logged": len(steps), "loss_field": "ce" if recs[0].get("ce") is not None else "loss"}


def loss_at(recs: list[dict], step: int) -> float | None:
    for r in recs:
        if int(r["step"]) == int(step):
            return _loss_of(r)
    return None


def clip_fraction(recs: list[dict]) -> dict:
    n = len(recs)
    clipped = sum(1 for r in recs if r.get("clipped"))
    clip_on = sorted({r.get("max_norm") for r in recs}, key=lambda x: (x is None, x))
    return {"steps": n, "clipped_steps": clipped, "fraction": clipped / n if n else float("nan"), "max_norm_values": clip_on}


# ============================================================================= A/A statistic (sec. 8.1, E5)
def aa_statistic(pairs: list[dict], cell_of: dict | None = None) -> dict:
    """I_AA for each A/A pair from F levels {pair: {"A": {"code": (F, SE), "math": (F, SE)}, "B": {...}}}.

    Log statistic when EVERY raw F of the test is resolved (F >= 3 SE; E5 decides for the whole test),
    otherwise the normalized difference with F-bar = the mean over the A/A runs of the same cell (target).
    Returns per-pair values ONLY to the caller (the gate releases their RMS, never the values; sec. 4.10)."""

    allF = [pairs_i[s][c] for pairs_i in pairs for s in ("A", "B") for c in ("code", "math")]
    use_log = all(resolved(F, se) for F, se in allF)
    cell_of = cell_of or {}
    vals = []
    if use_log:
        for p in pairs:
            I = (math.log(p["B"]["code"][0]) - math.log(p["A"]["code"][0])) - \
                (math.log(p["B"]["math"][0]) - math.log(p["A"]["math"][0]))
            vals.append(I)
    else:
        cells: dict[str, dict[str, list]] = {}
        for i, p in enumerate(pairs):
            cell = cell_of.get(i, p.get("cell", "all"))
            for s in ("A", "B"):
                for c in ("code", "math"):
                    cells.setdefault(cell, {}).setdefault(c, []).append(p[s][c][0])
        for i, p in enumerate(pairs):
            cell = cell_of.get(i, p.get("cell", "all"))
            fb_code = float(np.mean(cells[cell]["code"]))
            fb_math = float(np.mean(cells[cell]["math"]))
            I = (p["B"]["code"][0] - p["A"]["code"][0]) / fb_code - (p["B"]["math"][0] - p["A"]["math"][0]) / fb_math
            vals.append(I)
    v = np.array(vals, dtype=np.float64)
    return {"statistic": "log" if use_log else "normalized_difference", "n_pairs": int(v.size),
            "rms": float(math.sqrt(float(np.mean(v * v)))) if v.size else float("nan"), "_values": vals}


# ============================================================================= sign-flip power lookup
def load_lookup(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lookup_power(rows: list[dict], D: int, delta: float, alpha: float, dist: str = "normal") -> dict:
    """Power of the one-sided exact sign-flip test from the lookup table (signflip_lookup_v1.csv),
    linear interpolation in delta between grid points (clamped to the grid). alpha 0.05 -> power_sf_05,
    0.025 -> power_sf_025 (m = 1 or 2 in alpha = 0.05/m)."""

    col = {0.05: "power_sf_05", 0.025: "power_sf_025"}.get(round(alpha, 6))
    if col is None:
        raise ValueError(f"no sign-flip column for alpha={alpha}")
    pts = sorted((float(r["delta"]), float(r[col])) for r in rows if int(r["D"]) == int(D) and r["dist"] == dist)
    if not pts:
        raise ValueError(f"lookup table has no rows for D={D}, dist={dist}")
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    clamped = bool(delta < xs[0] or delta > xs[-1])
    val = float(np.interp(min(max(delta, xs[0]), xs[-1]), xs, ys))
    lo = int(np.searchsorted(xs, delta, side="right") - 1)
    floor = float(ys[max(0, min(lo, len(ys) - 1))]) if delta >= xs[0] else float(ys[0])
    return {"D": int(D), "delta": float(delta), "alpha": alpha, "column": col, "dist": dist,
            "power": val, "power_grid_floor": floor, "clamped": clamped}


# ============================================================================= records
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=_jsonable, allow_nan=True)


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def finalize_record(rec: dict, out: str) -> str:
    """Write the gate record as canonical JSON with an embedded sha256 of its content (the record without the
    'sha256' field) and a <out>.sha256 sidecar of the file bytes. Returns the content hash."""

    rec = {k: v for k, v in rec.items() if k != "sha256"}
    rec.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec.setdefault("host", {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()})
    content = hashlib.sha256(canonical(rec).encode("utf-8")).hexdigest()
    rec["sha256"] = content
    txt = canonical(rec)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    os.replace(tmp, out)
    with open(out + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{file_sha256(out)}  {os.path.basename(out)}\n")
    return content


def verify_record(path: str) -> bool:
    rec = json.load(open(path, encoding="utf-8"))
    h = rec.pop("sha256", None)
    return h is not None and hashlib.sha256(canonical(rec).encode("utf-8")).hexdigest() == h


def inputs_manifest(paths: Iterable[str]) -> dict:
    return {os.path.abspath(p): file_sha256(p) for p in sorted(set(paths)) if p and os.path.exists(p)}


# ============================================================================= run bookkeeping
class RunIndex:
    """Run roots from a run_queue state file (results/queue_state_<stage>.json) plus the plan's run table."""

    def __init__(self, plan: dict, state_paths: list[str]):
        self.plan = plan
        self.state: dict = {}
        for sp in state_paths:
            if sp and os.path.exists(sp):
                self.state.update(json.load(open(sp, encoding="utf-8")))
        self.runs = {r["id"]: r for r in plan.get("runs", [])}

    def status(self, rid: str) -> str:
        return self.state.get(rid, {}).get("status", "pending")

    def root(self, rid: str) -> str | None:
        return self.state.get(rid, {}).get("root")

    def eval_path(self, rid: str, set_name: str, variant: str = "ckpt") -> str | None:
        root = self.root(rid)
        if not root:
            return None
        return os.path.join(root, self.runs.get(rid, {}).get("eval_dir", "eval_main"), f"{variant}__{set_name}.npz")

    def interim_path(self, rid: str, set_name: str, step: int) -> str | None:
        root = self.root(rid)
        return os.path.join(root, "interim", f"step{int(step):03d}__{set_name}.npz") if root else None

    def train_log(self, rid: str) -> str | None:
        root = self.root(rid)
        return os.path.join(root, "trainlog", "train_steps.jsonl") if root else None

    def audit_ok(self, rid: str) -> bool | None:
        return self.state.get(rid, {}).get("audit_ok")


# ============================================================================= symmetrized forgetting (theory verification r3)
FSYM_MODES = {"live": ("ckpt", "reflect"), "selpin": ("selpin", "reflect+selpin"), "pin": ("pinned", "reflect+pinned")}


def fsym(base: dict, plus: dict, minus: dict) -> dict:
    """F_sym = (1/W) sum_w 1/2 [(CE_+(w) - CE_0(w)) + (CE_-(w) - CE_0(w))], + = theta_0 + delta, - = theta_0 - delta;
    SE = sd_w / sqrt(W). Returns the per-window values under '_w' (callers must not release them)."""

    cb, cp = _paired(base, plus)
    cb2, cm = _paired(base, minus)
    if not np.array_equal(cb, cb2):
        raise ValueError("theta_0 windows differ between the two sides")
    d = 0.5 * ((cp - cb) + (cm - cb)).astype(np.float64)
    W = int(d.size)
    return {"F_sym": float(d.mean()), "SE": float(d.std(ddof=1) / math.sqrt(W)) if W > 1 else float("nan"), "W": W, "_w": d}


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"value": float(d.mean()), "SE": float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan")}


def fsym_decomposition(base: dict, results: dict) -> dict:
    """F_sym for the live, selection-pinned and pinned replays (results = {variant: eval result} with the variants of
    FSYM_MODES), the decomposition F_sym,live = F_sym,pin + G + Phi with G = F_sym,selpin - F_sym,pin (smooth
    gate-value pathway) and Phi = F_sym,live - F_sym,selpin (selection flips), paired window SEs, and the one-sided
    out-of-domain flag F_sym,pin + 2 SE < 0 (on the live statistic only as 'curvature or flips')."""

    out: dict = {}
    w: dict = {}
    for mode, (p, m) in FSYM_MODES.items():
        if p in results and m in results:
            r = fsym(base, results[p], results[m])
            w[mode] = r.pop("_w")
            out[mode] = r
    if "pin" in out:
        out["pin"]["flag_out_of_domain"] = bool(out["pin"]["F_sym"] + 2 * out["pin"]["SE"] < 0)
    if "live" in out:
        out["live"]["flag_curvature_or_flips"] = bool(out["live"]["F_sym"] + 2 * out["live"]["SE"] < 0)
    if "selpin" in w and "pin" in w:
        out["G_gate_pathway"] = _diff(w["selpin"], w["pin"])
    if "live" in w and "selpin" in w:
        out["Phi_flips"] = _diff(w["live"], w["selpin"])
    out["complete"] = all(k in out for k in ("live", "selpin", "pin"))
    return out


def fsym_table(base_files: dict, root: str | None, sets: list[str], dirs: dict | None = None) -> dict:
    """{set: fsym_decomposition} for one run from its evaluation directories (eval_main: ckpt, eval_reflect: reflect,
    eval_pin: pinned / selpin / reflect+pinned / reflect+selpin). Missing files leave a mode out."""

    dirs = dirs or {"ckpt": "eval_main", "reflect": "eval_reflect", "pinned": "eval_pin", "selpin": "eval_pin",
                    "reflect+pinned": "eval_pin", "reflect+selpin": "eval_pin"}
    out = {}
    if not root:
        return out
    for s in sets:
        b = base_files.get(s)
        if not b or not os.path.exists(b):
            continue
        res = {}
        for v, d in dirs.items():
            p = os.path.join(root, d, f"{v}__{s}.npz")
            if os.path.exists(p):
                res[v] = load_eval(p)
        if res:
            out[s] = fsym_decomposition(load_eval(b), res)
    return out
