"""G0 gate record exactly as registered (prereg v1.0 sec. 6, Appendix B) from g0_probe output directories.

No model, no torch, deterministic (fixed generator seeds). Hash this file before reading G0 results.

    python g0_gate.py --config g0_gate_config.json [--out g0/gate_record.json] [--draws 1000]

Config (JSON; relative paths resolve against the config file's directory):
  {"bases": {"moe_b1": {"dir": "g0/moe_b1", "arch": "moe", "base": 1},
             "moe_b2": {"dir": "g0/moe_b2", "arch": "moe", "base": 2},
             "nestor": {"dir": "g0/nestor", "arch": "nestor", "base": 1}},
   "decision_base": {"moe": "moe_b1", "nestor": "nestor"},     # G0-1 decisions (Appendix B rows); others reported
   "c1_targets": {"moe": ["german", "books"], "nestor": ["german"]},
   "code_streams": ["code", "pycode"],                         # the design (missing streams are skipped)
   "c2": {"base": "moe_b1", "target": "german"},
   "dclm_tilt": {"base": "moe_b1", "match_target": "german"},
   "g0_5": {"local": "<per-window npz>", "h100": "<per-window npz>"}   # or {"provisional": {"local_ce": x, "h100_ce": y}}
   "layout": "auto", "half_seed": 0}

Registered rules implemented (sec. 6):
  G0-1  per architecture and C1 target: median P (beta = 1) over 1,000 accepted generator draws for f in {1/2, 1/3};
        f fixed by the larger median; pass iff median P >= log 1.3. Decisions on MoE base 1 and NESTOR.
  G0-2  acceptance >= 1e-3 or >= 50 % of swap chains converge (within 800 iterations) at log 1.05; otherwise C2 is re-run
        matching DCLM only, then any design at log 1.10; still infeasible -> G0-1 failure (C1) / G0-3 failure (C2).
  G0-3  median Delta_Q over 1,000 draws (MoE base 1, German, matched on DCLM, code, math) >= log 1.5, else C2 exploratory.
  G0-4  cross-fitted kappa_H / kappa_L at q = 1/8 (second moments) >= 1.25, else the DCLM-tilt block is not run.
  G0-5  |dCE| <= 1e-3 nats and per-window Pearson r >= 0.999 on the same 2.048M val09 tokens, else stop NESTOR adaptation.
  G0-6  split-half reliability of Q_E >= 0.9, else double the probe set once (reported either way).
Operationalisations of points the text leaves open are listed in OPERATIONALISATION and in the record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:                      # PYTHONSAFEPATH=1 in the kernel
    sys.path.insert(0, HERE)

import crossover as CX  # noqa: E402
import eval_windows as EW  # noqa: E402

G0_DRAWS = 1000                       # G0-1 / G0-3 "1,000 accepted generator draws"
G0_DRAW_INDEX = 0                     # G0 distribution runs use seed BASE + 0; registered draws are d >= 1
LOG_G0 = CX.REG_LOG_G0                # G0-1: log 1.3
LOG_C2 = CX.REG_LOG_C2                # G0-3: log 1.5
LOG_C2_EQ = math.log(2.0)             # C2-eq reading needs median Delta_Q >= log 2 (sec. 3)
ACCEPT_MIN = 1e-3                     # G0-2
SWAP_CONV_MIN = 0.5                   # G0-2 (sec. 6; sec. 14.6 used 8 of 32 chains -- sec. 6 governs)
KAPPA_MIN = CX.REG_KAPPA_MIN          # G0-4: 1.25
KAPPA_Q = CX.REG_KAPPA_Q              # G0-4: q = 1/8
XP_DCE = 1e-3                         # G0-5
XP_R = 0.999
XP_TOKENS = 10_004_085                # G0-5: first 10,004,085 val09 tokens (2,443 windows, eval_ckpt fp32)
XP_H100_REF = {"nestor": 4.784113, "moe_b2": 4.635177}   # H100 eval_ckpt mean CE references
REL_MIN = 0.9                         # G0-6
BETA0_MIN, BETA0_CAP = 0.25, 0.5      # G0-7: beta0_b = max(0.25, |D_b| + 2 SE), capped at 0.5

OPERATIONALISATION = [
    "G0 distribution runs: one generator run per (base, design, target, code stream, f, rung) collecting 1,000 accepted "
    "pairs with seed BASE + 0 (5000 C1, 6000 C2, 6500 DCLM-tilt); registered draws use d >= 1 (seed BASE + d, first "
    "accepted pair).",
    "G0-2 rung = first configuration at which at least one f passes; f is then chosen among the passing f values.",
    "G0-2 acceptance = accepted / proposed over all rejection batches; swap convergence = converged / chains over all "
    "swap rounds (None when rejection alone produced 1,000 pairs).",
    "Code stream (the design with the design): computed on SELECTION statistics only -- per MoE-base-1 C1 target "
    "the median over the G0 draws of P evaluated on the selection tables, at the f with the larger such median; the stream "
    "with the larger mean over targets wins (ties -> 'code' = AlgebraicStack); the same stream for every base; it is "
    "also the 'code' corpus of C2 and DCLM-tilt matching.",
    "G0-4: LOW/HIGH-q = ceil(N/8) lowest / highest DCLM selection-usage modules per stratum; RO on estimation "
    "statistics; kappa_X = RO(X) / (k/N).",
    "G0-6: cells = (non-random probe direction S in {E_ALL, probed subsets}, old corpus c) per base and target; halves = "
    "alternate probe windows within each corpus; Spearman-Brown corrected Pearson r of log Q_kl across cells in PINNED "
    "routing; gate = minimum over the base's targets.",
    "G0-5: per H100-trained base, |local - H100| eval_ckpt fp32 mean CE on the first 10,004,085 val09 "
    "tokens <= 1e-3 (local from results/continual.jsonl key A|<exp>|val09, or the RESULT line of a log); plus per-window "
    "r >= 0.999 for every local_<base>* / h100_<base>* pair in the xplat directory. No registered action for MoE base 2.",
    "G0-7: per MoE base, the registered C1 draws in margin__probe.json (g0_margin.py; d = 1..5 per passing "
    "target, f/tol of the decision-base G0-1 cell); x_d = I_probe,d - P_d; P-bar_b = mean P_d; D_b = mean x_d / P-bar_b; "
    "SE = sd(x_d) / sqrt(n) / P-bar_b; beta0_b = max(0.25, |D_b| + 2 SE), capped at 0.5 ('margin-limited' if capped). "
    "The gate recomputes every P_d and subset from the tables and rejects a mismatching file.",
]


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _q(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    qs = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(x.size), "median": float(qs[2]), "mean": float(x.mean()), "q10": float(qs[0]), "q25": float(qs[1]),
            "q75": float(qs[3]), "q90": float(qs[4])}


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum()) / den if den > 0 else float("nan")


def _sb(r: float) -> float:
    return 2 * r / (1 + r) if math.isfinite(r) and r > -1 else float("nan")


# ----------------------------------------------------------------------------- generator runs
def gen_run(t, design, *, f, target, code, match, tol, n_draws, k) -> dict:
    tilt = CX.reg_tilt(t.sel, design, target=target, code=code)
    seed = CX.REG_SEED_BASE[design] + G0_DRAW_INDEX
    r = CX.reg_pairs(t.sel, tilt, k, f, n_draws, seed, match, tol)
    conv = r["swap_convergence"]
    ok = (r["acceptance"] >= ACCEPT_MIN) or (conv is not None and conv >= SWAP_CONV_MIN)
    pred = CX.reg_predicted(t.est, r["A"], r["B"], design, target=target, code=code) if r["n"] else np.zeros(0)
    pred_sel = CX.reg_predicted(t.sel, r["A"], r["B"], design, target=target, code=code) if r["n"] else np.zeros(0)
    out = {"f": f, "seed": seed, "tol": tol, "matched": list(match), "g0_2_pass": bool(ok and r["n"] > 0),
           "n_pairs": r["n"], "complete": r["complete"], "acceptance": r["acceptance"], "proposals": r["proposals"],
           "batches": r["batches"], "swap_rounds": r["swap_rounds"], "swap_chains": r["swap_chains"],
           "swap_converged": r["swap_converged"], "swap_convergence": conv, "from_swap": r["from_swap"],
           "predicted": _q(pred), "predicted_selection": _q(pred_sel)}
    out["median"] = out["predicted"].get("median")
    out["median_selection"] = out["predicted_selection"].get("median")
    return out


def ladder(t, design, rungs, *, target, code, n_draws, k) -> dict:
    tried = []
    for ri, rung in enumerate(rungs):
        per_f = {}
        for f in CX.REG_POOLS:
            per_f[f"{f:.6f}"] = gen_run(t, design, f=f, target=target, code=code, match=rung["match"], tol=rung["tol"],
                                        n_draws=n_draws, k=k)
        ok = [kf for kf, v in per_f.items() if v["g0_2_pass"] and v["median"] is not None]
        tried.append({"rung": ri, "name": rung["name"], "tol": rung["tol"], "matched": rung["match"], "per_f": per_f})
        if ok:
            best = max(ok, key=lambda kf: per_f[kf]["median"])        # ties keep f = 1/2 (listed first)
            v = per_f[best]
            return {"feasible": True, "rung": ri, "rung_name": rung["name"], "tol": rung["tol"], "matched": rung["match"],
                    "f": v["f"], "f_label": "1/2" if abs(v["f"] - 0.5) < 1e-9 else "1/3", "median": v["median"],
                    "n_pairs": v["n_pairs"], "complete": v["complete"], "tried": tried}
    return {"feasible": False, "rung": None, "median": None, "f": None, "tried": tried}


def rungs_for(design, *, target, code, match_target="german"):
    if design == "c1":
        m = CX.reg_matched("c1", target=target)
        return [{"name": "registered (log 1.05)", "match": m, "tol": CX.REG_TOL},
                {"name": "relaxed tolerance (log 1.10)", "match": m, "tol": CX.REG_TOL_RELAXED}]
    if design == "c2":
        return [{"name": "registered: DCLM, code, math (log 1.05)", "match": CX.reg_matched("c2", code=code), "tol": CX.REG_TOL},
                {"name": "DCLM-only matching (log 1.05)", "match": ["dclm"], "tol": CX.REG_TOL},
                {"name": "DCLM-only matching, relaxed tolerance (log 1.10)", "match": ["dclm"], "tol": CX.REG_TOL_RELAXED}]
    if design == "dclm_tilt":
        m = CX.reg_matched("dclm_tilt", target=match_target, code=code)
        return [{"name": "registered (log 1.05)", "match": m, "tol": CX.REG_TOL},
                {"name": "relaxed tolerance (log 1.10)", "match": m, "tol": CX.REG_TOL_RELAXED}]
    raise ValueError(design)


# ----------------------------------------------------------------------------- G0-5 / G0-6
def g0_5(spec: dict | None, inputs: dict) -> dict:
    if not spec:
        return {"status": "missing", "pass": None, "note": "no g0_5 inputs in the config"}
    if "local" in spec and "h100" in spec:
        for p in (spec["local"], spec["h100"]):
            if not os.path.exists(p):
                return {"status": "missing", "pass": None, "note": f"missing file {p}"}
            inputs[p] = sha256_file(p)
        a, b = EW.load_result(spec["local"]), EW.load_result(spec["h100"])
        pr = EW.paired(a, b, check=True)
        blk = int(a["meta"].get("set", {}).get("block", 4096))
        ok = abs(pr["diff"]) <= XP_DCE and pr["corr"] >= XP_R
        return {"status": "computed", "pass": bool(ok), "abs_dce": abs(pr["diff"]), "r": pr["corr"],
                "n_windows": pr["n_windows"], "tokens": pr["n_windows"] * blk, "ce_local": pr["ce_a"], "ce_h100": pr["ce_b"],
                "registered_tokens": 2_048_000, "tokens_match_registration": pr["n_windows"] * blk == 2_048_000,
                "set_local": a["meta"].get("set"), "set_h100": b["meta"].get("set")}
    if "provisional" in spec:
        pv = spec["provisional"]
        d = abs(float(pv["local_ce"]) - float(pv["h100_ce"]))
        return {"status": "provisional (mean CE only; per-window r not available)",
                "pass": None, "provisional_pass": bool(d <= XP_DCE), "abs_dce": d, "r": None, "inputs": pv}
    return {"status": "missing", "pass": None, "note": "g0_5 needs local+h100 npz or provisional"}


def _probe_cells(rep: dict, z, mode: str) -> list[tuple[str, str, float, float]]:
    eps = float(rep["eps"])
    corpus = np.asarray(z["corpus"]).astype(str)
    cells = []
    for lab in rep["subsets"]:
        if "~rand" in lab:
            continue
        kp, km = f"{mode}|{lab}|{eps:.6g}|+1|kl", f"{mode}|{lab}|{eps:.6g}|-1|kl"
        if kp not in z.files or km not in z.files:
            continue
        qw = (np.asarray(z[kp]) + np.asarray(z[km])) / (2 * eps * eps)
        for c in sorted(set(corpus)):
            idx = np.flatnonzero(corpus == c)
            if len(idx) < 2:
                continue
            cells.append((lab, c, float(qw[idx[0::2]].mean()), float(qw[idx[1::2]].mean())))
    return cells


def reliability_from_cells(cells) -> dict:
    a = np.array([x[2] for x in cells])
    b = np.array([x[3] for x in cells])
    out = {"cells": len(cells)}
    if len(cells) < 3:
        out.update(reliability=None, note="fewer than 3 cells with >= 2 windows per corpus")
        return out
    pos = (a > 0) & (b > 0)
    r_log = _corr(np.log(a[pos]), np.log(b[pos])) if pos.sum() >= 3 else float("nan")
    r_raw = _corr(a, b)
    out.update(r_half_log=r_log, reliability=_sb(r_log), r_half_raw=r_raw, reliability_raw=_sb(r_raw),
               nonpositive_cells=int((~pos).sum()))
    return out


def reliability_from_se(rep: dict, mode: str) -> dict:
    """APPROXIMATION when no per-window file exists: classical split-half reliability from cell means and SEs
    (log scale, delta method): r_half = var_true / (var_true + 2 se_log^2), var_true = var(log v) - mean(se_log^2)."""

    v, s = [], []
    for lab, ent in rep["subsets"].items():
        if "~rand" in lab or mode not in ent:
            continue
        for c, qf in ent[mode]["by_corpus"].items():
            q = qf["Q_kl"]
            if q["value"] > 0 and q["se"] == q["se"] and q["n"] >= 2:
                v.append(math.log(q["value"]))
                s.append(q["se"] / q["value"])
    if len(v) < 3:
        return {"cells": len(v), "reliability": None, "note": "fewer than 3 usable cells"}
    v, s = np.array(v), np.array(s)
    var_true = max(float(v.var(ddof=1)) - float((s ** 2).mean()), 0.0)
    r_half = var_true / (var_true + 2 * float((s ** 2).mean())) if var_true + float((s ** 2).mean()) > 0 else float("nan")
    return {"cells": int(len(v)), "r_half_log": r_half, "reliability": _sb(r_half), "approximate": True}


def g0_6(bases: dict, which: list[str], inputs: dict) -> dict:
    out = {}
    for bn in which:
        d = bases[bn]["dir"]
        files = sorted(f for f in os.listdir(d) if f.startswith("probe__") and f.endswith(".json")) if os.path.isdir(d) else []
        if not files:
            out[bn] = {"status": "missing", "pass": None, "note": "no probe__<target>.json"}
            continue
        per_t = {}
        for fn in files:
            p = os.path.join(d, fn)
            inputs[p] = sha256_file(p)
            rep = json.load(open(p, encoding="utf-8"))
            modes = rep.get("routing_modes") or [rep.get("primary_routing", "pinned")]
            mode = "pinned" if "pinned" in modes else rep.get("primary_routing", modes[0])   # pinned primary
            wf = os.path.join(d, rep.get("windows_file") or fn.replace(".json", "__windows.npz"))
            if os.path.exists(wf):
                inputs[wf] = sha256_file(wf)
                with np.load(wf, allow_pickle=False) as z:
                    r = reliability_from_cells(_probe_cells(rep, z, mode))
                r["source"] = "per-window probe values"
                # E_ALL-only variant (reported)
                with np.load(wf, allow_pickle=False) as z:
                    cells = [c for c in _probe_cells(rep, z, mode) if c[0] == "E_ALL"]
                r["e_all_only"] = reliability_from_cells(cells)
            else:
                r = reliability_from_se(rep, mode)
                r["source"] = "APPROXIMATE (means and SEs; no per-window probe file -- run g0_probe v3)"
            r["routing_mode"] = mode
            if mode != "pinned":
                r["note"] = "no pinned-routing probe values in this file (the design makes pinned primary)"
            per_t[rep.get("target", fn)] = r
        vals = [v["reliability"] for v in per_t.values() if v.get("reliability") is not None and math.isfinite(v["reliability"])]
        rel = min(vals) if vals else None
        out[bn] = {"status": "computed" if rel is not None else "not computable", "reliability": rel,
                   "pass": (rel >= REL_MIN) if rel is not None else None, "per_target": per_t,
                   "approximate": any(v.get("approximate") for v in per_t.values())}
    return out


def _local_ce_jsonl(path: str, key: str) -> dict | None:
    recs = [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]
    hit = [r for r in recs if r.get("key") == key]
    return hit[-1] if hit else None


def _local_ce_result_log(path: str) -> dict | None:
    raw = open(path, "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
    out = None
    for line in txt.splitlines():
        i = line.find("RESULT ")
        if i >= 0:
            try:
                out = json.loads(line[i + 7:].strip())
            except json.JSONDecodeError:
                continue
    return out


def _same_path(a, b) -> bool:
    if not a or not b:
        return True
    na = os.path.normcase(os.path.normpath(str(a)))
    nb = os.path.normcase(os.path.normpath(str(b)))
    return na == nb or na.endswith(nb) or nb.endswith(na)


def _xplat_pairs(xdir: str, base: str) -> list[tuple[str, str]]:
    """(local, h100) per-window npz pairs: xdir/local_<base>*[.npz | dir/*.npz] <-> xdir/h100_<same suffix>."""
    pairs = []
    if not xdir or not os.path.isdir(xdir):
        return pairs
    for name in sorted(os.listdir(xdir)):
        if not name.startswith(f"local_{base}"):
            continue
        a, b = os.path.join(xdir, name), os.path.join(xdir, "h100_" + name[len("local_"):])
        if os.path.isdir(a) and os.path.isdir(b):
            pairs += [(os.path.join(a, f), os.path.join(b, f)) for f in sorted(os.listdir(a))
                      if f.endswith(".npz") and os.path.exists(os.path.join(b, f))]
        elif a.endswith(".npz") and os.path.exists(b):
            pairs.append((a, b))
    return pairs


def g0_5_v4(spec: dict | None, bases: dict, ckpt_by_base: dict, inputs: dict, deviations: list) -> dict:
    """every H100-trained base; |dCE| on the identical first 10,004,085 val09 tokens (eval_ckpt fp32
    mean CE) <= 1e-3; per-window Pearson r >= 0.999 added whenever per-window outputs exist on both platforms."""

    spec = spec or {}
    out = {}
    for bn, bs in (spec.get("bases") or {}).items():
        h100 = float(bs.get("h100_ce", XP_H100_REF.get(bn, float("nan"))))
        loc, src = None, None
        if bs.get("local_jsonl") and os.path.exists(bs["local_jsonl"]):
            inputs[bs["local_jsonl"]] = sha256_file(bs["local_jsonl"])
            loc = _local_ce_jsonl(bs["local_jsonl"], bs.get("jsonl_key") or f"A|{bs.get('experiment', '')}|val09")
            src = f"{bs['local_jsonl']} key {bs.get('jsonl_key')}"
        elif bs.get("local_result_log") and os.path.exists(bs["local_result_log"]):
            inputs[bs["local_result_log"]] = sha256_file(bs["local_result_log"])
            loc = _local_ce_result_log(bs["local_result_log"])
            src = f"{bs['local_result_log']} (last RESULT line)"
        row = {"h100_ce": h100, "local_source": src, "local_record": loc, "threshold": XP_DCE, "r_threshold": XP_R,
               "tokens_registered": XP_TOKENS}
        if not loc or "ce" not in loc:
            row.update(status="pending (local mean CE not available)", pass_=None)
            out[bn] = row
            continue
        d = abs(float(loc["ce"]) - h100)
        row.update(local_ce=float(loc["ce"]), abs_dce=d, mean_ce_pass=bool(d <= XP_DCE), tokens=loc.get("tokens"))
        if loc.get("tokens") is not None and int(loc["tokens"]) != XP_TOKENS:
            deviations.append(f"G0-5 {bn}: local mean CE over {loc['tokens']} tokens (registered {XP_TOKENS})")
        known = [c for c in (ckpt_by_base.get(bn) or []) if c and c != "None"]
        if loc.get("ckpt") and known and not any(_same_path(loc["ckpt"], c) for c in known):
            deviations.append(f"G0-5 {bn}: local CE checkpoint {loc['ckpt']} differs from the G0 routing checkpoint {known}")
            row["checkpoint_mismatch"] = True
        pw = []
        for a, b in _xplat_pairs(spec.get("per_window_dir"), bn):
            inputs[a], inputs[b] = sha256_file(a), sha256_file(b)
            pr = EW.paired(EW.load_result(a), EW.load_result(b), check=True)
            pw.append({"local": a, "h100": b, "r": pr["corr"], "n_windows": pr["n_windows"], "diff": pr["diff"],
                       "pass": bool(pr["corr"] >= XP_R)})
        row["per_window"] = pw
        ok = row["mean_ce_pass"] and all(x["pass"] for x in pw)
        row.update(status="computed" + (" (mean CE + per-window r)" if pw else " (mean CE only; no per-window pair)"), pass_=bool(ok))
        out[bn] = row
    for bn, row in out.items():
        row["pass"] = row.pop("pass_")
        arch = bases.get(bn, {}).get("arch")
        if row["pass"] is None:
            row["action"] = "pending: no adaptation on this base until the local check exists"
        elif row["pass"]:
            row["action"] = "platform check passed: local adaptation on this base allowed"
        elif arch == "nestor":
            row["action"] = "stop NESTOR adaptation (C1-N and C3-NESTOR not run)"
        else:
            row["action"] = ("FAIL: no registered action for an MoE base (the design extends G0-5 without one) -- "
                             "PI decision; C1/C2 need both MoE bases")
    return out


def g0_7(cfg: dict, tables: dict, c1: dict, code: str, inputs: dict, deviations: list) -> dict:
    """per MoE base, D_b = mean_d (I_probe,d - P_d) / P-bar_b from pinned norm-matched random-direction
    quadratic forms on code and math (g0_margin.py); beta0_b = max(0.25, |D_b| + 2 SE) capped at 0.5."""

    out = {}
    db = cfg["decision_base"].get("moe")
    for bn, b in cfg["bases"].items():
        if b["arch"] != "moe" or bn not in tables:
            continue
        f = os.path.join(b["dir"], "margin__probe.json")
        if not os.path.exists(f):
            out[bn] = {"status": "missing", "pass": None, "note": "no margin__probe.json (run g0_margin.py plan + run)"}
            continue
        inputs[f] = sha256_file(f)
        rep = json.load(open(f, encoding="utf-8"))
        problems = []
        if rep.get("routing_mode") != "pinned":
            problems.append(f"routing mode {rep.get('routing_mode')} (registered: pinned)")
        if rep.get("code_stream") != code:
            problems.append(f"code stream {rep.get('code_stream')} != gate choice {code}")
        rows = []
        t = tables[bn]
        for dr in rep.get("draws", []):
            cell = c1.get(f"{db}|{dr['target']}|{code}") or {}
            if not cell.get("feasible"):
                problems.append(f"{dr['target']}: no feasible decision-base G0-1 cell")
                continue
            e = CX.reg_draw(t, "c1", int(dr["d"]), f=cell["f"], target=dr["target"], code=code, tol=cell["tol"])
            if e.get("status") != "ok" or not math.isclose(e["predicted"]["P"], float(dr["P"]), rel_tol=1e-9, abs_tol=1e-12):
                problems.append(f"{dr['target']} d{dr['d']}: P_d {dr['P']} != registered {e.get('predicted')}")
                continue
            rows.append((dr["target"], int(dr["d"]), float(dr["I_probe"]), float(dr["P"])))
        if problems:
            deviations.append(f"G0-7 {bn}: margin file inconsistent with the registered draws: {problems[:3]}")
        if len(rows) < 2:
            out[bn] = {"status": "not computable", "pass": None, "problems": problems, "n_draws": len(rows)}
            continue
        x = np.array([r[2] - r[3] for r in rows])
        pbar = float(np.mean([r[3] for r in rows]))
        D = float(x.mean()) / pbar
        se = float(x.std(ddof=1)) / math.sqrt(len(x)) / pbar
        raw = abs(D) + 2 * se
        beta0 = min(BETA0_CAP, max(BETA0_MIN, raw))
        out[bn] = {"status": "computed" if not problems else "computed (with problems)", "n_draws": len(rows),
                   "P_bar": pbar, "D": D, "SE": se, "abs_D_plus_2SE": raw, "beta0": beta0,
                   "margin_limited": bool(raw > BETA0_CAP), "problems": problems,
                   "draws": [{"target": r[0], "d": r[1], "I_probe": r[2], "P": r[3]} for r in rows],
                   "action": (f"beta0_{bn} = {beta0:.4f} fixed in the gate record" +
                              ("; C1 labelled 'margin-limited' in advance (cap 0.5 binds)" if raw > BETA0_CAP else ""))}
    return out


# ----------------------------------------------------------------------------- main computation
def _resolve(p: str | None, root: str) -> str | None:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def load_config(path: str) -> dict:
    cfg = json.load(open(path, encoding="utf-8-sig"))  # tolerate a BOM (Windows PowerShell utf8)
    root = os.path.dirname(os.path.abspath(path))
    cfg.setdefault("c1_targets", {"moe": ["german", "books"], "nestor": ["german"]})
    cfg.setdefault("code_streams", ["code", "pycode"])
    cfg.setdefault("decision_base", {"moe": "moe_b1", "nestor": "nestor"})
    cfg.setdefault("c2", {"base": "moe_b1", "target": "german"})
    cfg.setdefault("dclm_tilt", {"base": "moe_b1", "match_target": "german"})
    cfg.setdefault("layout", "auto")
    cfg.setdefault("half_seed", 0)
    for b in cfg["bases"].values():
        b["dir"] = _resolve(b["dir"], root)
        b.setdefault("family", {"moe": "moe", "nestor": "priv"}[b["arch"]])
    g5 = cfg.get("g0_5") or {}
    for kk in ("local", "h100", "per_window_dir"):
        if kk in g5:
            g5[kk] = _resolve(g5[kk], root)
    for bs in (g5.get("bases") or {}).values():
        for kk in ("local_jsonl", "local_result_log"):
            if bs.get(kk):
                bs[kk] = _resolve(bs[kk], root)
    return cfg


def compute(cfg: dict, n_draws: int = G0_DRAWS, log=print) -> dict:
    t0 = time.time()
    inputs: dict[str, str] = {}
    deviations: list[str] = []
    tables, layout, kk, ckpts = {}, {}, {}, {}
    for bn, b in cfg["bases"].items():
        if not os.path.isdir(b["dir"]):
            log(f"[gate] base {bn}: directory {b['dir']} missing -> skipped")
            continue
        t = CX.reg_load_dir(b["dir"], b["family"], layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
        for c, s in t.sources.items():
            for role in ("selection", "estimation"):
                p = s[role]["file"]
                if p not in inputs:
                    inputs[p] = sha256_file(p)
        k = CX.reg_k(t)
        if b.get("k") is not None and int(b["k"]) != k:
            raise SystemExit(f"{bn}: config k={b['k']} but registered k = q N = {k}")
        tables[bn], kk[bn] = t, k
        layout[bn] = {c: s["layout"] for c, s in t.sources.items()}
        ckpts[bn] = sorted({str(s[role].get("checkpoint")) for s in t.sources.values() for role in ("selection", "estimation")})
        if len(ckpts[bn]) > 1:     # g0_probe resumes from existing routing files; a glob checkpoint may have moved
            deviations.append(f"{bn}: routing files come from {len(ckpts[bn])} different checkpoints {ckpts[bn]} -- "
                              f"the base is mixed; re-run g0_probe with an explicit checkpoint path in a clean directory")
        if any(v == "crossfit" for v in layout[bn].values()):
            deviations.append(f"{bn}: selection/estimation statistics from seeded halves of one routing stream per corpus "
                              f"(crossfit layout), not the registered selection half / evaluation windows (sec. 4.3)")
        for c, s in t.sources.items():
            nsel = s["selection"]["windows"]
            if s["layout"] == "registered" and nsel != 256:
                deviations.append(f"{bn}/{c}: {nsel} selection windows (registered 256)")
            if s["layout"] == "registered" and s["estimation"]["windows"] != 500:
                deviations.append(f"{bn}/{c}: {s['estimation']['windows']} estimation windows (registered 500)")
        dsel = (t.sources.get("dclm") or {}).get("selection", {})
        dset = dsel.get("set") or (t.sources.get("dclm") or {}).get("set") or {}
        if dset and "val10" not in str(dset.get("path", "")):
            deviations.append(f"{bn}: DCLM selection statistics not from dclm_val10 ({dset.get('path')})")
        log(f"[gate] {bn}: family {t.family}, L={t.L}, N={t.N}, k={k}, excluded {t.excluded}, corpora {sorted(t.sel)}, "
            f"layout {sorted(set(layout[bn].values()))}")
    rec: dict = {"gates": {}, "decisions": {}, "layout": layout, "k": kk, "checkpoints": ckpts}

    # ---------------- G0-1 (C1) for every base x target x code stream
    c1 = {}
    for bn, t in tables.items():
        arch = cfg["bases"][bn]["arch"]
        for tgt in cfg["c1_targets"][arch]:
            for cs in cfg["code_streams"]:
                need = {tgt, "dclm", cs, "math"}
                key = f"{bn}|{tgt}|{cs}"
                if not need <= set(t.sel):
                    c1[key] = {"feasible": False, "missing": sorted(need - set(t.sel)), "median": None}
                    continue
                c1[key] = ladder(t, "c1", rungs_for("c1", target=tgt, code=cs), target=tgt, code=cs, n_draws=n_draws, k=kk[bn])
                log(f"[gate] G0-1 {key}: median P {_fmt(c1[key]['median'])} at f={c1[key]['f']} "
                    f"(rung {c1[key]['rung']}, feasible {c1[key]['feasible']})")
    # code stream choice on the MoE decision base
    mb = cfg["decision_base"].get("moe")
    nb = cfg["decision_base"].get("nestor")
    code_scores = {}

    def _sel_median(cell):          # selection statistics only (f by the selection median as well)
        if not cell.get("feasible"):
            return -math.inf
        pf = cell["tried"][cell["rung"]]["per_f"]
        vals = [v["median_selection"] for v in pf.values() if v["g0_2_pass"] and v["median_selection"] is not None]
        return max(vals) if vals else -math.inf

    for cs in cfg["code_streams"]:
        if mb in tables and all(f"{mb}|{tgt}|{cs}" in c1 and "missing" not in c1[f"{mb}|{tgt}|{cs}"] for tgt in cfg["c1_targets"]["moe"]):
            code_scores[cs] = float(np.mean([_sel_median(c1[f"{mb}|{tgt}|{cs}"]) for tgt in cfg["c1_targets"]["moe"]]))
    if code_scores:
        code = max(cfg["code_streams"], key=lambda cs: (code_scores.get(cs, -math.inf), cs == "code"))
        code_note = ("chosen on the MoE decision base from SELECTION statistics only (mean over targets of the median "
                     "selection-table P), same stream for every base")
    else:
        avail = [cs for cs in cfg["code_streams"] if any(cs in t.sel for t in tables.values())]
        code = "code" if "code" in avail or not avail else avail[0]
        code_note = "MoE decision base not measured -> registered primary stream 'code' (AlgebraicStack)"
        deviations.append("code stream not chosen by the design (MoE base 1 missing)")
    rec["decisions"]["code_stream"] = {"chosen": code, "scores": code_scores, "note": code_note}

    def arch_status(base_name, arch):
        if base_name not in tables:
            return None, {}
        st = {}
        for tgt in cfg["c1_targets"][arch]:
            c = c1.get(f"{base_name}|{tgt}|{code}")
            st[tgt] = bool(c and c["feasible"] and c["median"] is not None and c["median"] >= LOG_G0) if c and "missing" not in c else None
        vals = [v for v in st.values() if v is not None]
        return (any(vals) if vals else None), st

    moe_ok, moe_t = arch_status(mb, "moe")
    nes_ok, nes_t = arch_status(nb, "nestor")
    actions = []
    if moe_ok is False:              # 'MoE fails, NESTOR passes -> C1-N confirmatory' replaced by PIVOT
        actions.append("STOP/PIVOT: re-scope to theory + C3 + D4 + C4-style descriptive analyses; C1, C1-N and C2 are not run as "
                       "confirmatory" + (" (NESTOR passing does not promote C1-N)" if nes_ok else ""))
    else:
        if nes_ok is False:
            actions.append("NESTOR fails: C1-N dropped")
        if moe_ok:
            fails = [tg for tg, v in moe_t.items() if v is False]
            passes = [tg for tg, v in moe_t.items() if v]
            for tg in fails:
                actions.append(f"MoE target {tg} fails: its cells are replaced by {passes[0]} cells (draws per base unchanged)")
            if not fails:
                actions.append("MoE: C1 proceeds with all targets")
        if nes_ok:
            actions.append("NESTOR: C1-N proceeds as a secondary, base-specific test")
    if moe_ok is None:
        actions.append("MoE decision base not measured: G0-1 incomplete")
    if nes_ok is None:
        actions.append("NESTOR decision base not measured: G0-1 incomplete")
    rec["gates"]["G0-1"] = {"threshold": LOG_G0, "code_stream": code, "cells": c1, "moe_pass": moe_ok, "moe_targets": moe_t,
                            "nestor_pass": nes_ok, "nestor_targets": nes_t, "actions": actions}
    rec["gates"]["G0-7"] = g0_7(cfg, tables, c1, code, inputs, deviations) if moe_ok else {}

    # ---------------- G0-3 (C2) and DCLM-tilt generator
    c2cfg = cfg["c2"]
    c2b = c2cfg["base"]
    if c2b in tables and {c2cfg["target"], "dclm", code, "math"} <= set(tables[c2b].sel):
        c2 = ladder(tables[c2b], "c2", rungs_for("c2", target=c2cfg["target"], code=code), target=c2cfg["target"], code=code,
                    n_draws=n_draws, k=kk[c2b])
        if not c2["feasible"]:
            c2_act = "C2 infeasible at every G0-2 rung: treated as a G0-3 failure -> C2 exploratory"
            c2_pass = False
        else:
            c2_pass = c2["median"] >= LOG_C2
            c2_act = ("C2 eligible as confirmatory (pending G2)" if c2_pass else "C2 is exploratory (median Delta_Q < log 1.5)")
            if c2["rung"] >= 1:
                c2_act += "; matched on DCLM only -> C2 on code and math (sec. 9 item 9) dropped"
            if c2["tol"] > CX.REG_TOL:
                c2_act += "; tolerance relaxed to log 1.10 (recorded)"
        rec["gates"]["G0-3"] = {"base": c2b, "target": c2cfg["target"], "threshold": LOG_C2, "result": c2, "pass": c2_pass,
                                "median_dQ_ge_log2": bool(c2["feasible"] and c2["median"] >= LOG_C2_EQ),
                                "action": c2_act}
        log(f"[gate] G0-3 {c2b}: median dQ {_fmt(c2['median'])} at f={c2['f']} (rung {c2['rung']}) -> {c2_act}")
    else:
        rec["gates"]["G0-3"] = {"status": "missing", "pass": None, "note": f"base {c2b} or corpora missing"}
    dt = cfg["dclm_tilt"]
    dtb = dt["base"]
    mt = dt.get("match_target", "german")
    dtr = None
    if dtb in tables and {"dclm", mt, code, "math"} <= set(tables[dtb].sel):
        dtr = ladder(tables[dtb], "dclm_tilt", rungs_for("dclm_tilt", target=None, code=code, match_target=mt), target=mt,
                     code=code, n_draws=n_draws, k=kk[dtb])
        log(f"[gate] DCLM-tilt generator {dtb}: median P_P {_fmt(dtr['median'])} at f={dtr['f']} (rung {dtr['rung']})")

    # ---------------- G0-2 summary
    g02 = []
    for key, c in c1.items():
        if "missing" in c:
            continue
        g02.append({"design": "C1", "cell": key, "feasible": c["feasible"], "rung": c["rung"], "tol": c.get("tol"),
                    "matched": c.get("matched"), "per_f": {kf: {x: v[x] for x in ("acceptance", "swap_convergence", "swap_chains",
                                                                                   "n_pairs", "g0_2_pass")}
                                                           for kf, v in (c["tried"][c["rung"]]["per_f"].items() if c["feasible"] else c["tried"][-1]["per_f"].items())}})
    if rec["gates"]["G0-3"].get("result"):
        c = rec["gates"]["G0-3"]["result"]
        g02.append({"design": "C2", "cell": f"{c2b}|{c2cfg['target']}", "feasible": c["feasible"], "rung": c["rung"],
                    "tol": c.get("tol"), "matched": c.get("matched"),
                    "per_f": {kf: {x: v[x] for x in ("acceptance", "swap_convergence", "swap_chains", "n_pairs", "g0_2_pass")}
                              for kf, v in (c["tried"][c["rung"]]["per_f"].items() if c["feasible"] else c["tried"][-1]["per_f"].items())}})
    if dtr is not None:
        g02.append({"design": "DCLM-tilt", "cell": f"{dtb}|dclm", "feasible": dtr["feasible"], "rung": dtr["rung"],
                    "tol": dtr.get("tol"), "matched": dtr.get("matched"),
                    "per_f": {kf: {x: v[x] for x in ("acceptance", "swap_convergence", "swap_chains", "n_pairs", "g0_2_pass")}
                              for kf, v in (dtr["tried"][dtr["rung"]]["per_f"].items() if dtr["feasible"] else dtr["tried"][-1]["per_f"].items())}})
    g02_act = []
    for row in g02:
        if not row["feasible"]:
            g02_act.append(f"{row['design']} {row['cell']}: infeasible at every rung -> " +
                           {"C1": "G0-1 failure for this cell", "C2": "G0-3 failure (C2 exploratory)",
                            "DCLM-tilt": "DCLM-tilt block not run"}[row["design"]])
        elif row["rung"]:
            g02_act.append(f"{row['design']} {row['cell']}: rung {row['rung']} (tol {row['tol']:.4f}, matched {row['matched']}) recorded")
    rec["gates"]["G0-2"] = {"accept_min": ACCEPT_MIN, "swap_convergence_min": SWAP_CONV_MIN, "rows": g02,
                            "actions": g02_act or ["all designs feasible at the registered configuration"]}

    # ---------------- G0-4
    g04 = {}
    for bn, t in tables.items():
        if "dclm" in t.sel:
            kap = CX.reg_kappa(t, "dclm", KAPPA_Q)
            g04[bn] = {x: kap[x] for x in ("q", "k", "N", "L", "ro_low", "ro_high", "ro_rand_expected", "kappa_L", "kappa_H", "ratio")}
    if dtb in g04:
        ratio = g04[dtb]["ratio"]
        k_ok = ratio >= KAPPA_MIN
        feas = bool(dtr and dtr["feasible"])
        act = ("DCLM-tilt secondary block runs" if k_ok and feas else
               "DCLM-tilt secondary block not run (kappa values reported)" if not k_ok else
               "kappa passes but the DCLM-tilt generator is infeasible (G0-2) -> block not run")
        rec["gates"]["G0-4"] = {"base": dtb, "threshold": KAPPA_MIN, "ratio": ratio, "pass": bool(k_ok), "per_base": g04,
                                "generator": dtr, "action": act}
    else:
        rec["gates"]["G0-4"] = {"status": "missing", "pass": None, "per_base": g04, "note": f"base {dtb} not measured"}

    # ---------------- G0-5, G0-6
    spec5 = cfg.get("g0_5") or {}
    if "bases" in spec5:                                   # the design (registered from 2026-11-01)
        g5 = {"criterion": "the design", "per_base": g0_5_v4(spec5, cfg["bases"], ckpts, inputs, deviations)}
        pb = g5["per_base"]
        g5["pass"] = (None if not pb or any(v["pass"] is None for v in pb.values()) else all(v["pass"] for v in pb.values()))
        g5["action"] = "; ".join(f"{bn}: {v['action']}" for bn, v in pb.items()) or "no H100-trained base configured"
        g5["status"] = "computed" if pb and all(v["pass"] is not None for v in pb.values()) else "pending"
    else:                                                  # pre-the design-v4 formats (two per-window files / provisional)
        g5 = g0_5(spec5, inputs)
        g5["criterion"] = "pre-the design-v4 (superseded)"
        deviations.append("G0-5 evaluated with the pre-the design-v4 input format")
        g5["action"] = ("NESTOR adaptation allowed" if g5.get("pass") else
                        "stop NESTOR adaptation (C1-N and C3-NESTOR not run)" if g5.get("pass") is False else
                        "pending: NESTOR adaptation must wait for the platform check")
    rec["gates"]["G0-5"] = g5
    g6 = g0_6(cfg["bases"], cfg.get("g0_6", {}).get("bases") or [b for b in cfg["bases"] if b in tables], inputs)
    for bn, v in g6.items():
        v["action"] = ("forecast inputs OK (reported)" if v.get("pass") else
                       "double the probe set once (probe_windows x 2) and recompute; reliability reported either way"
                       if v.get("pass") is False else "not computable from the available files")
        if v.get("approximate"):
            deviations.append(f"G0-6 {bn}: approximate reliability (no per-window probe file)")
    rec["gates"]["G0-6"] = g6
    if g5.get("status", "").startswith("provisional"):
        deviations.append("G0-5 provisional (mean CE only)")

    # ---------------- Appendix B rows
    def c1row(base_name, tgts):
        parts = []
        for tg in tgts:
            c = c1.get(f"{base_name}|{tg}|{code}")
            parts.append(f"{tg}: " + ("n/a" if not c or "missing" in c else
                                      f"median P {_fmt(c['median'])} at f = {c['f_label']}" if c["feasible"] else "infeasible"))
        return "; ".join(parts) + f" (code stream {code})"

    g3 = rec["gates"]["G0-3"]
    g4 = rec["gates"]["G0-4"]
    rows = [
        {"row": "G0-1 median P (MoE base 1, German / books); chosen f", "value": c1row(mb, cfg["c1_targets"]["moe"]) if mb in tables else "not measured"},
        {"row": "G0-1 median P (NESTOR, German); chosen f", "value": c1row(nb, cfg["c1_targets"]["nestor"]) if nb in tables else "not measured"},
        {"row": "G0-2 acceptance / swap convergence; tolerance used", "value": "; ".join(
            f"{r['design']} {r['cell']}: " + ", ".join(f"f={float(kf):.3f} acc {_fmt(v['acceptance'], 4)} conv {_fmt(v['swap_convergence'], 2)}"
                                                    for kf, v in r["per_f"].items()) + f", tol {_fmt(r['tol'], 4)}" for r in g02) or "n/a"},
        {"row": "G0-3 median Delta_Q (MoE base 1, German); C2 status",
         "value": (f"{_fmt(g3['result']['median'])} at f = {g3['result'].get('f_label', 'n/a')}; {g3['action']}" if g3.get("result") else "not measured")},
        {"row": "G0-4 kappa_H/kappa_L (q = 1/8); DCLM-tilt status",
         "value": (f"{_fmt(g4['ratio'])}; {g4['action']}" if "ratio" in g4 else "not measured")},
        {"row": "G0-5 |dCE| (first 10,004,085 val09 tokens), per-window r; NESTOR and MoE base 2 status",
         "value": ("; ".join(f"{bn}: |dCE| {_fmt(v.get('abs_dce'), 6)}, r " +
                             (", ".join(_fmt(x['r'], 5) for x in v.get('per_window', [])) or "n/a") + f" -> {v['action']}"
                             for bn, v in g5.get("per_base", {}).items())
                   if "per_base" in g5 else
                   (f"|dCE| {_fmt(g5.get('abs_dce'), 5)}, r {_fmt(g5.get('r'), 5)}; {g5['action']}" if g5.get("status") != "missing" else "not measured"))},
        {"row": "G0-6 Q_E split-half reliability",
         "value": "; ".join(f"{bn}: {_fmt(v.get('reliability'))}{' (approx.)' if v.get('approximate') else ''}" for bn, v in g6.items()) or "n/a"},
        {"row": "G0-7 D_b, SE, beta0_b per MoE base; margin-limited flag",
         "value": "; ".join(f"{bn}: " + (f"D {_fmt(v['D'], 4)}, SE {_fmt(v['SE'], 4)}, n {v['n_draws']}, beta0 {_fmt(v['beta0'], 4)}"
                                         f"{' (margin-limited)' if v['margin_limited'] else ''}" if "D" in v else v.get("status", "n/a"))
                            for bn, v in rec["gates"].get("G0-7", {}).items()) or "not computed (MoE failed G0-1 or no MoE base)"},
    ]
    rec["appendix_b"] = rows
    rec["deviations"] = deviations
    rec["operationalisation"] = OPERATIONALISATION
    rec["inputs"] = inputs
    rec["n_draws"] = n_draws
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G0 gate record (prereg v1.0 sec. 6)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: <dir of first base>/../gate_record.json")
    ap.add_argument("--draws", type=int, default=G0_DRAWS, help="accepted draws per G0 cell (registered: 1000)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rec = compute(cfg, a.draws)
    me = os.path.abspath(__file__)
    rec["script"] = {"path": me, "sha256": sha256_file(me)}
    rec["crossover"] = {"path": os.path.abspath(CX.__file__), "sha256": sha256_file(CX.__file__)}
    rec["config"] = {"path": os.path.abspath(a.config), "sha256": sha256_file(a.config), "resolved": cfg}
    rec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if a.draws != G0_DRAWS:
        rec["deviations"].append(f"{a.draws} draws per cell (registered {G0_DRAWS})")
    out = a.out or os.path.join(os.path.dirname(next(iter(cfg["bases"].values()))["dir"]), "gate_record.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("=" * 100)
    for r in rec["appendix_b"]:
        print(f"{r['row']}\n    {r['value']}")
    print("-" * 100)
    for g in ("G0-1", "G0-2"):
        for x in rec["gates"][g].get("actions", []):
            print(f"{g} ACTION: {x}")
    for g in ("G0-3", "G0-4", "G0-5"):
        print(f"{g} ACTION: {rec['gates'][g].get('action', rec['gates'][g].get('note', 'n/a'))}")
    for bn, v in rec["gates"].get("G0-7", {}).items():
        print(f"G0-7 ACTION ({bn}): {v.get('action', v.get('status'))}")
    for bn, v in rec["gates"]["G0-6"].items():
        print(f"G0-6 ACTION ({bn}): {v['action']}")
    for d in rec["deviations"]:
        print(f"DEVIATION: {d}")
    print(f"record: {out}  ({rec['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
