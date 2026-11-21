"""analyze_confirm_ext.py -- blocks of the confirmatory analysis script (decision d; hash-list entry 2.17).

Called by analyze_confirm.run_all(ctx) as run_ext(ctx, H), H = the analyze_confirm module: its file access (Ctx, which
records every file read with its SHA-256), statistics core and constants are reused, nothing is re-implemented.
Torch-free; deterministic (the only random numbers are the C4 bootstrap, seed H.SEED_BOOT); reads only registered
result files. A quantity whose input files are not planned or not present is reported with the reason, never guessed.

Blocks (preregistration v1.2 / v1.3 section):
  C3            v1.2 sec. 10.1: endpoint revert decomposition (phi_g, iota, shares; seed-then-domain aggregation with the
                pooled within-domain t-interval), single-group revert deltas, WD-only, freeze cross-check, random
                partial-revert control, NESTOR-MoE comparison rule and the routing-health label.
  C3_per_layer  v1.3 sec. 4.8: Delta_l = F - F(revert l), Delta_l / F, additivity gap, domain means and ranges.
  C4            v1.2 sec. 10.2: per C1 draw, the per-window slope b_d of Delta F_w on Delta log RO_w (theta_0 routing of
                the DCLM evaluation windows), beta_C4,d = b_d / F-bar_d; t-interval and two-way bootstrap.
  C5_french     v1.2 secs. 10.3 and 12: hash protocol check and the metrics of the hashed French forecasts.
  C2_eq_rand    v1.2 sec. 8.2 item 7: b_Muon - b_Muon^rand (its alignment component; theory verification v2 P-4) and
                the per-expert gradient SNR (G2-5 record).
  matched_gain  v1.2 sec. 11: F(G*) ratios, I_G (C1), I2_G (C2), FULL across architectures, Pareto points.
  F_rel         v1.2 l. 327 and sec. 9 item 13: F_rel = F - F_DCLM-CONT (same base); C1 and C2 statistics in F_rel.
  bpb           v1.2 sec. 9 item 13: bits-per-byte versions (bytes from the eval files, else from the data bins).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np

FORMAT_EXT = "analysis_confirm_ext/1"
LN2 = math.log(2.0)

# v1.2 sec. 10.1 two-group pairs (g, g'): F_only g = F(revert g') (only g changed), F_only g' = F(revert g)
C3_PAIR = {
    "moe": {"g": "experts", "revert_g": "revert:moe", "g2": "rest", "revert_g2": "revert:kind:attn+embed_head+kind:norm"},
    "nestor": {"g": "column_private", "revert_g": "revert:attn+priv", "g2": "shared",
               "revert_g2": "revert:shared+pub+embed_head"},
}
# freeze cross-check (v1.2 sec. 10.1): the frozen group per architecture (its revert = forgetting with only the
# complement changed)
FREEZE_GROUP = {"moe": ("experts", "revert:moe"), "nestor": ("column_private", "revert:attn+priv")}

POINTERS = {
    "routing_entanglement": {
        "prereg": "v1.3 sec. 4.9",
        "script": "phase6/run_entanglement.py (stored version c6e2e322-d995-4f6b-9a1e-8e290dcc2842); predictions E1-E7 "
                  "hashed before the run: results/entanglement/predictions_E1_E7.md, logs/entanglement_predictions.log",
        "results": ["entanglement_results.json (stored version "
                    "42dbd3a7-61c7-4adb-9929-7579d86673e9)"],
        "note": "entanglement_note.md (version 15b283b4-11af-47ea-9b6b-901abe65c97d)"},
    "gap_trajectory": {
        "prereg": "v1.3 sec. 4.10", "script": "results/gap_trajectory/gap_analysis.py",
        "results": ["results/gap_trajectory/gap_trajectory.json (version 4e616dad-663c-4d1c-b058-acd2557b85d3)"],
        "note": "results/gap_trajectory/gap_note.md (version 7f4a9295-b8d8-4f85-8229-7322803b8ddc)",
        "figure": "results/gap_trajectory/gap_trajectory.png (version 2a325ea8-ffe6-42a1-b147-778eb0933b06)"},
}


# ============================================================================= small helpers
def _arch(base: str | None) -> str:
    return "nestor" if base == "nestor" else "moe"


def _arg(args: list, flag: str) -> str | None:
    args = [str(a) for a in args]
    if flag in args:
        i = args.index(flag)
        return args[i + 1] if i + 1 < len(args) else None
    return None


def _stat(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    return {"F": float(x.mean()), "SE": float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan, "W": n}


def _mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def _ratio(a, b):
    return (a / b) if (a is not None and b not in (None, 0.0)) else None


def _queue_jobs(ctx) -> dict:
    qp = ctx.p(ctx.plan.get("queue") or "confirm/queue_confirm.json")
    if not qp or not os.path.exists(qp):
        return {}
    return {j["id"]: j for j in (ctx.json(qp).get("jobs") or [])}


def _job_files(ctx, job: dict, set_name: str = "dclm") -> dict:
    """{variant: <out>/<variant with ':' -> '-'>__<set>.npz} of a standalone eval_windows job (eval_windows.py l. 856)."""
    args = [str(a) for a in (job.get("args") or job.get("argv") or [])]
    out, vs = _arg(args, "--out"), _arg(args, "--variants")
    sets = [args[i + 1].split("=", 1)[0] for i, a in enumerate(args[:-1]) if a == "--set"]
    if not out or not vs or set_name not in sets:
        return {}
    od = ctx.p(out)
    return {v: os.path.join(od, f"{v.replace(':', '-')}__{set_name}.npz") for v in vs.split(",") if v}


def _pooled(vals_by_dom: dict, H, level: float = 0.95) -> dict:
    """v1.2 sec. 10.1 aggregation: mean over seeds within a domain, then the equal-weight mean over domains; t-interval
    from the pooled within-domain seed variance, SE = sqrt(s2_pooled * sum_d 1/n_d) / D, df = sum_d (n_d - 1)."""
    doms = sorted(d for d, v in vals_by_dom.items() if v)
    if not doms:
        return {"n_domains": 0, "mean": None}
    means = {d: float(np.mean(vals_by_dom[d])) for d in doms}
    m = float(np.mean(list(means.values())))
    ss = sum(float(((np.asarray(vals_by_dom[d]) - means[d]) ** 2).sum()) for d in doms)
    df = sum(len(vals_by_dom[d]) - 1 for d in doms)
    out = {"n_domains": len(doms), "per_domain": means, "n_per_domain": {d: len(vals_by_dom[d]) for d in doms},
           "mean": m, "df": df, "level": level,
           "range": [min(means.values()), max(means.values())]}
    if df > 0:
        s2 = ss / df
        se = math.sqrt(s2 * sum(1.0 / len(vals_by_dom[d]) for d in doms)) / len(doms)
        h = H.tq(df, 0.5 + level / 2) * se
        out.update(se=se, lo=m - h, hi=m + h, method="pooled within-domain seed variance")
    else:
        out.update(se=None, lo=None, hi=None, method="points only (one seed per domain)")
    return out


# ============================================================================= C3 units (confirm FULL runs + G1 FULL runs)
def _c3_units(ctx, H) -> tuple[list[dict], list[str]]:
    units, notes = [], []
    for r in ctx.plan.get("runs", []):
        if r.get("role") != "full":
            continue
        rid = r["id"]
        u = {"rid": rid, "source": "confirm", "base": r["base"], "arch": r.get("arch") or _arch(r["base"]),
             "domain": r.get("target"), "seed": int(r["seed"]) if r.get("seed") is not None else None,
             "F_file": ctx.eval_file(rid, "ckpt", "dclm"), "rev": {}, "layer": {}, "wd": None}
        for ev in r.get("evals") or []:
            if "dclm" not in (ev.get("sets") or []):
                continue
            vs = ev.get("variants", "ckpt")
            for v in (vs.split(",") if isinstance(vs, str) else vs):
                f = ctx.eval_file(rid, v, "dclm")
                if ev.get("name") == "revert_layer":
                    u["layer"][v] = f
                elif v == "wdonly":
                    u["wd"] = f
                elif v.startswith("revert:"):
                    u["rev"][v] = f
        ok, why = ctx.usable(rid)
        u["usable"], u["why"] = ok, why
        units.append(u)
    jobs = _queue_jobs(ctx)
    g1 = H._gplan(ctx, "g1")
    g1runs, g1state = ({r["id"]: r for r in g1[0].get("runs", [])}, g1[1]) if g1 else ({}, {})
    for jid, job in sorted(jobs.items()):
        if not jid.startswith("c3_revert_"):
            continue
        rid = jid[len("c3_revert_"):]
        gr = g1runs.get(rid)
        if gr is None:
            notes.append(f"{jid}: run {rid} not in the G1 plan")
            continue
        root = (g1state.get(rid) or {}).get("root")
        files = _job_files(ctx, job)
        u = {"rid": rid, "source": "G1", "base": gr["base"], "arch": gr.get("arch") or _arch(gr["base"]),
             "domain": gr.get("target"), "seed": int(gr["seed"]) if gr.get("seed") is not None else None,
             "F_file": os.path.join(ctx.p(root), gr.get("eval_dir", "eval_main"), "ckpt__dclm.npz") if root else None,
             "rev": {v: f for v, f in files.items() if v.startswith("revert:")}, "wd": files.get("wdonly"),
             "layer": _job_files(ctx, jobs[f"c3_layer_revert_{rid}"]) if f"c3_layer_revert_{rid}" in jobs else {}}
        why = []
        if (g1state.get(rid) or {}).get("status", "pending") != "done":
            why.append(f"G1 run status {(g1state.get(rid) or {}).get('status', 'pending')}")
        for j in [jid] + ([f"c3_layer_revert_{rid}"] if f"c3_layer_revert_{rid}" in jobs else []):
            if ctx.status(j) != "done":
                why.append(f"{j}: status {ctx.status(j)}")
        u["usable"], u["why"] = (not why), ("; ".join(why) or None)
        units.append(u)
    return units, notes


def _paired_delta(ctx, bf: str, pf: dict, path: str | None) -> dict | None:
    """F(revert) and the paired Delta = F - F(revert) (per-window mean and SE) against the full run's windows."""
    pr = ctx.pair(bf, path) if path else None
    if pr is None:
        return None
    if not np.array_equal(pr["index"], pf["index"]):
        raise ValueError(f"{path}: windows differ from the full run's evaluation")
    d_full, d_rev = pf["cr"] - pf["cb"], pr["cr"] - pr["cb"]
    dl = _stat(d_full - d_rev)
    return {"F_revert": float(d_rev.mean()), "Delta": dl["F"], "Delta_SE": dl["SE"]}


def _c3_run(ctx, u: dict) -> dict:
    bf = ctx.base_file(u["base"], "dclm")
    pf = ctx.pair(bf, u["F_file"]) if u.get("F_file") else None
    rec = {k: u[k] for k in ("rid", "source", "base", "arch", "domain", "seed")}
    if pf is None:
        rec["status"] = "full-run DCLM evaluation missing"
        return rec
    F = _stat(pf["cr"] - pf["cb"])
    rec["F"], rec["F_SE"] = F["F"], F["SE"]
    rec["reverts"] = {}
    for v, f in sorted(u["rev"].items()):
        x = _paired_delta(ctx, bf, pf, f)
        rec["reverts"][v] = None if x is None else dict(x, Delta_over_F=_ratio(x["Delta"], F["F"]))
    pr = C3_PAIR[u["arch"]]
    a, b = rec["reverts"].get(pr["revert_g2"]), rec["reverts"].get(pr["revert_g"])
    if a and b:
        f_only_g, f_only_g2 = a["F_revert"], b["F_revert"]
        phi_g = 0.5 * ((F["F"] - f_only_g2) + f_only_g)
        rec["decomposition"] = {
            "g": pr["g"], "g2": pr["g2"], "F_only_g": f_only_g, "F_only_g2": f_only_g2,
            "phi_g": phi_g, "phi_g2": F["F"] - phi_g, "iota": F["F"] - f_only_g - f_only_g2,
            "share_g": _ratio(phi_g, F["F"]), "share_g2": _ratio(F["F"] - phi_g, F["F"]),
            "iota_over_F": _ratio(F["F"] - f_only_g - f_only_g2, F["F"])}
    else:
        rec["decomposition"] = None
        rec["status"] = f"pair reverts missing ({pr['revert_g']}, {pr['revert_g2']})"
    if u.get("wd"):
        pw = ctx.pair(bf, u["wd"])
        if pw is not None:
            fw = float((pw["cr"] - pw["cb"]).mean())
            rec["WD"] = {"F_WD": fw, "F_WD_over_F": _ratio(fw, F["F"])}
    return rec


def block_c3(ctx, H) -> dict:
    units, notes = _c3_units(ctx, H)
    out = {"prereg": "v1.2 sec. 10.1 (C3 endpoint revert decomposition; MoE base 2 points only)",
           "statement": "The endpoint decomposition is not a path decomposition (v1.2 sec. 10.1).",
           "pairs": C3_PAIR, "units_planned": len(units), "notes": notes, "dropped": [], "per_run": []}
    recs = []
    for u in units:
        if not u["usable"]:
            out["dropped"].append({"rid": u["rid"], "reason": u["why"]})
            continue
        r = _c3_run(ctx, u)
        out["per_run"].append(r)
        if r.get("decomposition"):
            recs.append(r)
    per_base = {}
    for base in sorted({r["base"] for r in out["per_run"]}):
        rr = [r for r in recs if r["base"] == base]
        agg = {}
        for key in ("share_g", "share_g2", "iota_over_F"):
            byd = defaultdict(list)
            for r in rr:
                if r["decomposition"].get(key) is not None:
                    byd[r["domain"]].append(r["decomposition"][key])
            agg[key] = _pooled(byd, H)
        singles = {}
        for v in sorted({v for r in out["per_run"] if r["base"] == base for v in (r.get("reverts") or {})}):
            byd = defaultdict(list)
            for r in out["per_run"]:
                x = (r.get("reverts") or {}).get(v) if r["base"] == base else None
                if x and x.get("Delta_over_F") is not None:
                    byd[r["domain"]].append(x["Delta_over_F"])
            singles[v] = _pooled(byd, H)
        wd = defaultdict(list)
        for r in out["per_run"]:
            if r["base"] == base and (r.get("WD") or {}).get("F_WD_over_F") is not None:
                wd[r["domain"]].append(r["WD"]["F_WD_over_F"])
        arch = _arch(base)
        per_base[base] = {"arch": arch, "g": C3_PAIR[arch]["g"], "g2": C3_PAIR[arch]["g2"], "runs": len(rr),
                          "domains": sorted({r["domain"] for r in rr}), "shares": agg,
                          "single_revert_Delta_over_F": singles,
                          "WD_share": {"F_WD_over_F": _pooled(wd, H), "note": "WD-only evaluations (descriptive)"}}
    out["per_base"] = per_base
    out["freeze_cross_check"] = _c3_freeze(ctx, units, out["per_run"])
    rand = [v for u in units for v in u["rev"] if "rand" in v.split(":", 1)[1]]
    out["random_partial_revert"] = ({"status": "evaluated", "variants": sorted(set(rand))} if rand else
                                    {"status": "not computable: the plan registers no random partial-revert evaluation "
                                               "(eval_windows revert grammar has no random-subset group)",
                                     "prereg": "v1.2 sec. 10.1"})
    out["nestor_moe_comparison"] = {
        "rule": "MoE-versus-NESTOR comparisons only at matched base loss (MoE checkpoint with val09 CE ~ 4.784), with "
                "first-order-free statistics under pinned routing (v1.2 sec. 10.1)",
        "made": False, "reason": "no MoE checkpoint at matched base loss is registered in the plan or FREEZE.json"}
    g1p = ctx.p("g1/gate_record_G1.json")
    g9 = ((ctx.json(g1p).get("gates") or {}).get("G0-9L") or {}) if os.path.exists(g1p) else {}
    applies = g9.get("outcome") not in (None, "pass", "not_evaluable")
    out["NESTOR_label"] = {"G0_9L": {k: g9.get(k) for k in ("outcome", "reason", "action")},
                           "label": "exploratory (routing-health label applies)" if applies else
                                    "base-specific secondary (label unchanged)"}
    return out


def _c3_freeze(ctx, units: list[dict], per_run: list[dict]) -> dict:
    out = {"prereg": "v1.2 sec. 10.1 freeze cross-check (Han et al. 2026: revert and freeze attribution can differ)"}
    byrid = {r["rid"]: r for r in per_run}
    for r in ctx.plan.get("runs", []):
        if r.get("role") != "freeze":
            continue
        arch = r.get("arch") or _arch(r["base"])
        grp, rv = FREEZE_GROUP[arch]
        ok, why = ctx.usable(r["id"])
        rec = {"run": r["id"], "base": r["base"], "frozen_group": grp, "domain": r.get("target"), "seed": r.get("seed")}
        if not ok:
            rec["status"] = why
            out[arch] = rec
            continue
        Ff = ctx.F(r["id"], "dclm", "raw")
        comp = [byrid[u["rid"]] for u in units if u["usable"] and u["base"] == r["base"] and u["domain"] == r.get("target")
                and u["rid"] in byrid and (byrid[u["rid"]].get("reverts") or {}).get(rv)]
        rec["F_freeze"] = None if Ff is None else Ff["F"]
        if Ff is None or not comp:
            rec["status"] = "freeze-run evaluation or the FULL runs' revert evaluations missing"
            out[arch] = rec
            continue
        F_full = [c["F"] for c in comp]
        F_only = [c["reverts"][rv]["F_revert"] for c in comp]
        rec.update(full_runs=[c["rid"] for c in comp], F_FULL_mean=float(np.mean(F_full)),
                   F_only_complement_mean=float(np.mean(F_only)),
                   revert_attribution=float(np.mean([a - b for a, b in zip(F_full, F_only)])),
                   freeze_attribution=float(np.mean(F_full)) - Ff["F"])
        rec["freeze_minus_revert_attribution"] = rec["freeze_attribution"] - rec["revert_attribution"]
        rec["note"] = ("revert attribution = mean over the FULL runs of F - F(revert frozen group) = F - F_only(complement); "
                       "freeze attribution = mean F_FULL - F_freeze; their difference = mean F_only(complement) - F_freeze")
        out[arch] = rec
    return out


# ============================================================================= C3 per layer (v1.3 sec. 4.8)
def _layer_unit(v: str) -> tuple[str, str]:
    """'revert:moe@b0l3' -> ('moe', 'b0l3'); 'revert:attn@b3+priv@b3' -> ('attn+priv', 'b3')."""
    parts = v.split(":", 1)[1].split("+")
    gs, ts = zip(*(p.split("@", 1) for p in parts))
    if len(set(ts)) != 1:
        raise ValueError(f"{v}: parts with different scopes")
    return "+".join(gs), ts[0]


def _unit_key(t: str) -> tuple:
    import re
    m = re.match(r"^b(\d+)(?:l(\d+))?$", t)
    return (int(m.group(1)), int(m.group(2)) if m.group(2) else -1) if m else (1 << 30, t)


def block_c3_layer(ctx, H) -> dict:
    units, _ = _c3_units(ctx, H)
    out = {"prereg": "v1.3 sec. 4.8 (descriptive; one seed per domain, no CI)", "per_run": [], "dropped": []}
    for u in units:
        if not u["layer"]:
            continue
        if not u["usable"]:
            out["dropped"].append({"rid": u["rid"], "reason": u["why"]})
            continue
        bf = ctx.base_file(u["base"], "dclm")
        pf = ctx.pair(bf, u["F_file"]) if u.get("F_file") else None
        rec = {k: u[k] for k in ("rid", "source", "base", "domain", "seed")}
        if pf is None:
            rec["status"] = "full-run DCLM evaluation missing"
            out["per_run"].append(rec)
            continue
        F = float((pf["cr"] - pf["cb"]).mean())
        groups, layers = set(), {}
        for v, f in u["layer"].items():
            g, t = _layer_unit(v)
            groups.add(g)
            x = _paired_delta(ctx, bf, pf, f)
            layers[t] = None if x is None else {"variant": v, "Delta": x["Delta"], "Delta_SE": x["Delta_SE"],
                                                "Delta_over_F": _ratio(x["Delta"], F)}
        if len(groups) != 1:
            raise ValueError(f"{u['rid']}: per-layer reverts of different groups {sorted(groups)}")
        grp = groups.pop()
        allv = f"revert:{grp}"
        a = _paired_delta(ctx, bf, pf, u["rev"].get(allv)) if u["rev"].get(allv) else None
        ok = [x for x in layers.values() if x is not None]
        s = float(sum(x["Delta"] for x in ok)) if len(ok) == len(layers) else None
        rec.update(F=F, group=grp, all_group_revert=allv, layers=dict(sorted(layers.items(), key=lambda kv: _unit_key(kv[0]))),
                   sum_Delta=s, Delta_all=None if a is None else a["Delta"],
                   additivity_gap=None if (s is None or a is None) else s - a["Delta"],
                   sum_over_all=None if (s is None or a is None) else _ratio(s, a["Delta"]))
        out["per_run"].append(rec)
    per_base = {}
    for base in sorted({r["base"] for r in out["per_run"] if r.get("layers")}):
        rr = [r for r in out["per_run"] if r["base"] == base and r.get("layers")]
        toks = sorted({t for r in rr for t in r["layers"]}, key=_unit_key)
        rows = {}
        for t in toks:
            d = [r["layers"][t]["Delta"] for r in rr if r["layers"].get(t)]
            q = [r["layers"][t]["Delta_over_F"] for r in rr if r["layers"].get(t) and r["layers"][t]["Delta_over_F"] is not None]
            rows[t] = {"n_domains": len(d), "Delta_mean": _mean(d), "Delta_range": [min(d), max(d)] if d else None,
                       "Delta_over_F_mean": _mean(q), "Delta_over_F_range": [min(q), max(q)] if q else None}
        gaps = [r["additivity_gap"] for r in rr if r.get("additivity_gap") is not None]
        sums = [r["sum_over_all"] for r in rr if r.get("sum_over_all") is not None]
        per_base[base] = {"group": rr[0]["group"], "domains": sorted(r["domain"] for r in rr), "per_unit": rows,
                          "additivity_gap": {"mean": _mean(gaps), "range": [min(gaps), max(gaps)] if gaps else None},
                          "sum_over_all_group_revert": {"mean": _mean(sums), "range": [min(sums), max(sums)] if sums else None}}
    out["per_base"] = per_base
    return out


# ============================================================================= C4 (v1.2 sec. 10.2)
def _dclm_routing(ctx, base: str, entry: dict, need: dict) -> tuple[dict | None, str]:
    """theta_0 routing of the DCLM evaluation windows: the manifest's G0 directory, then the root's g0/<base>;
    routing__dclm_eval.npz (registered layout) before routing__dclm.npz; the file must hold every needed window
    (grid index with the same token hash). Returns ({'file', 'rows': index -> row, 'routing', 'block'}, note)."""
    import eval_windows as EW
    dirs = []
    for c in (ctx.code, "math"):
        f = (((entry or {}).get("sources") or {}).get(c) or {}).get("selection", {}).get("file")
        if f:
            dirs.append(os.path.dirname(ctx.p(f)))
            break
    dirs.append(ctx.p(f"g0/{base}"))
    tried = []
    for d in dict.fromkeys(dirs):
        for fn in ("routing__dclm_eval.npz", "routing__dclm.npz"):
            f = os.path.join(d, fn)
            if not os.path.exists(f):
                continue
            r = EW.load_result(f)
            rows = {int(i): k for k, i in enumerate(np.asarray(r["index"]))}
            hs = np.asarray(r["hash"]).astype(np.uint64)
            ok = all(i in rows and hs[rows[i]] == h for i, h in need.items())
            tried.append(f"{ctx.rel(f)}: {'covers' if ok else 'does not cover'} the evaluation windows")
            if ok:
                ctx._read(f)
                return {"file": f, "rows": rows, "routing": r["routing"],
                        "block": int((r.get("meta") or {}).get("set", {}).get("block", 4096))}, "; ".join(tried)
    return None, "; ".join(tried) or "no DCLM routing file in " + ", ".join(dirs)


def _ro_w(s2: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-window RO_w(S) = sum_{(l, m) in S} s2_w[l, m] / sum_{l, m} s2_w[l, m] (crossover.reg_ro per window);
    s2: [L, W, N] squared-gate sums of each window, S: [L, k] module indices."""
    num = np.stack([s2[l][:, np.asarray(S[l], dtype=np.int64)].sum(1) for l in range(s2.shape[0])]).sum(0)
    return num / s2.sum(axis=(0, 2))


def _c4_boot(X: np.ndarray, Y: np.ndarray, M: np.ndarray, H, stream: int) -> np.ndarray:
    """Two-way bootstrap over draws and windows (v1.2 sec. 10.2): each replicate resamples the windows (one resample
    shared by all draws, the windows being the same DCLM grid) and the draws, recomputes b_d and F-bar_d on the
    resampled windows and averages beta_C4 over the resampled draws. X = Delta R, Y = Delta F, M = (d_A + d_B) / 2."""
    D, W = X.shape
    rng = np.random.default_rng([int(H.SEED_BOOT), int(stream)])
    out = np.empty(int(H.N_BOOT))
    chunk = 500
    for s in range(0, int(H.N_BOOT), chunk):
        nb = min(chunk, int(H.N_BOOT) - s)
        wi = rng.integers(0, W, size=(nb, W))
        di = rng.integers(0, D, size=(nb, D))
        C = np.zeros((nb, W))
        np.add.at(C, (np.repeat(np.arange(nb), W), wi.ravel()), 1.0)
        Sx, Sy, Sm = C @ X.T, C @ Y.T, C @ M.T
        Sxx, Sxy = C @ (X * X).T, C @ (X * Y).T
        var = Sxx - Sx * Sx / W
        cov = Sxy - Sx * Sy / W
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = (cov / var) / (Sm / W)
        out[s:s + nb] = np.take_along_axis(beta, di, axis=1).mean(1)
    return out


def block_c4(ctx, H) -> dict:
    import crossover as CX
    draws, dropped, e1 = ctx.screen(ctx.draws_of("c1"))
    out = {"prereg": "v1.2 sec. 10.2 (C4 per-window difference-in-differences within DCLM); v1.3 sec. 4.8 (figure)",
           "dropped": list(dropped), "E1": e1, "per_draw": [], "routing_files": {}}
    rows = []
    for dr in draws:
        entry = ctx.entry(dr)
        wa, wb = ctx.windows(dr["A"], "dclm", "raw"), ctx.windows(dr["B"], "dclm", "raw")
        if entry is None or wa is None or wb is None:
            out["dropped"].append({"base": dr["base"], "d": dr["d"], "reason": "manifest entry or DCLM evaluation missing"})
            continue
        if not np.array_equal(wa["index"], wb["index"]):
            raise ValueError(f"C4 {dr['base']} d{dr['d']}: A and B DCLM windows differ")
        b0 = ctx.load(ctx.base_file(dr["base"], "dclm"))
        pos = {int(i): k for k, i in enumerate(np.asarray(b0["index"]))}
        need = {int(i): np.uint64(np.asarray(b0["hash"]).astype(np.uint64)[pos[int(i)]]) for i in wa["index"]}
        key = (dr["base"], ctx.rel(entry.get("sources", {}).get(ctx.code, {}).get("selection", {}).get("file", "")))
        if key not in out["routing_files"]:
            rt, note = _dclm_routing(ctx, dr["base"], entry, need)
            out["routing_files"][key] = (rt, note)
        rt, note = out["routing_files"][key]
        if rt is None:
            out["dropped"].append({"base": dr["base"], "d": dr["d"], "reason": f"theta_0 DCLM routing unavailable: {note}"})
            continue
        fam = entry.get("family", "moe")
        strata, _ = CX.reg_strata(rt["routing"], fam, rt["block"])
        t = ctx.tables(entry.get("sources"), fam, [ctx.code, "math"])
        if t is not None and list(t.strata) != list(strata):
            raise ValueError(f"C4 {dr['base']} d{dr['d']}: DCLM routing strata differ from the manifest's tables")
        ri = np.array([rt["rows"][int(i)] for i in wa["index"]])
        s2 = np.stack([np.asarray(rt["routing"][lk], dtype=np.float64)[ri, 2, :] for lk in strata])
        roA, roB = _ro_w(s2, np.asarray(entry["A"])), _ro_w(s2, np.asarray(entry["B"]))
        if np.any(roA <= 0) or np.any(roB <= 0):
            out["dropped"].append({"base": dr["base"], "d": dr["d"], "reason": "RO_w = 0 in some window (log undefined)"})
            continue
        dR = np.log(roB) - np.log(roA)
        dF = wb["d"] - wa["d"]
        vr = float(np.var(dR, ddof=1))
        cv = float(np.cov(dF, dR, ddof=1)[0, 1])
        Fbar = 0.5 * (float(wa["d"].mean()) + float(wb["d"].mean()))
        bd = cv / vr if vr > 0 else None
        corr = float(np.corrcoef(dF, dR)[0, 1]) if vr > 0 and float(np.var(dF)) > 0 else None
        rec = {"base": dr["base"], "d": dr["d"], "W": int(dR.size), "b": bd, "F_bar": Fbar,
               "beta_C4": (bd / Fbar) if (bd is not None and Fbar != 0) else None, "corr": corr,
               "mean_Delta_R": float(dR.mean()), "sd_Delta_R": math.sqrt(vr)}
        out["per_draw"].append(rec)
        if rec["beta_C4"] is not None:
            rows.append((rec, dR, dF, 0.5 * (wa["d"] + wb["d"]), wa["index"]))
    out["routing_files"] = {f"{k[0]}": (None if v[0] is None else ctx.rel(v[0]["file"])) for k, v in out["routing_files"].items()}

    def summary(sel: list, stream: int) -> dict:
        b = [r[0]["beta_C4"] for r in sel]
        s = {"n": len(b), "mean_beta_C4": _mean(b), "t95": H.t_interval(b, 0.95),
             "mean_within_draw_corr": _mean([r[0]["corr"] for r in sel])}
        if len(sel) >= 2:
            same = all(np.array_equal(sel[0][4], r[4]) for r in sel)
            if same:
                bs = _c4_boot(np.stack([r[1] for r in sel]), np.stack([r[2] for r in sel]), np.stack([r[3] for r in sel]), H, stream)
                s["bootstrap"] = {"B": int(H.N_BOOT), "seed": [int(H.SEED_BOOT), stream], "ci95": H.pct_ci(bs, 0.95),
                                  "method": "two-way: windows (shared resample) x draws, percentile"}
            else:
                s["bootstrap"] = {"status": "draws evaluated on different windows: two-way bootstrap not defined"}
        return s

    out["overall"] = summary(rows, 0)
    out["per_base"] = {b: summary([r for r in rows if r[0]["base"] == b], k + 1)
                       for k, b in enumerate(sorted({r[0]["base"] for r in rows}))}
    out["note"] = ("RO_w(S) uses all tokens of window w at theta_0 (squared effective gates, the registered statistic); "
                   "Delta R_w = log RO_w(B) - log RO_w(A); b_d = cov_w / var_w (ddof 1).")
    return out


# ============================================================================= C5 French (v1.2 secs. 10.3, 12)
def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[o] = np.arange(len(a), dtype=np.float64)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    return float((dx * dy).sum() / den) if den > 0 else None


def _metrics(pts: list[tuple]) -> dict:
    """pts = (log F-hat, log F, benchmark log F): R2 against the training-fold mean, Spearman, median |log ratio|."""
    if len(pts) < 2:
        return {"n": len(pts)}
    lf, lF, lb = (np.array(v) for v in zip(*pts))
    sse, sst = float(((lF - lf) ** 2).sum()), float(((lF - lb) ** 2).sum())
    return {"n": len(pts), "R2_oos": (1.0 - sse / sst) if sst > 0 else None,
            "spearman": _corr(_rank(lf), _rank(lF)) if len(pts) >= 3 else None,
            "median_abs_log_ratio": float(np.median(np.abs(lf - lF)))}


def block_c5_french(ctx, H) -> dict:
    out = {"prereg": "v1.2 secs. 10.3 and 12 (descriptive; prospective French forecasts)"}
    fp = ctx.p("confirm/c5_french_forecasts.json")
    if not os.path.exists(fp):
        out["status"] = "confirm/c5_french_forecasts.json missing"
        return out
    doc = ctx.json(fp)
    import forecasts as FCM
    verifies = FCM.content_sha(doc) == doc.get("sha256")
    lg = ctx.p("logs/c5_forecast_hashes.log")
    posted = None
    if os.path.exists(lg):
        ctx._read(lg)
        for line in open(lg, encoding="utf-8"):
            p = line.split()
            if len(p) >= 2 and p[1] == doc.get("sha256"):
                try:
                    posted = time.mktime(time.strptime(p[0], "%Y-%m-%dT%H:%M:%S"))
                except ValueError:
                    posted = None
                break
    french = list(doc.get("french_runs") or [])
    planned = sorted(r["id"] for r in ctx.plan.get("runs", []) if r.get("block") == "C5" and r.get("role") in ("c5_full", "c5_pair"))
    starts = [float((ctx.state.get(r) or {}).get("start")) for r in french if (ctx.state.get(r) or {}).get("start") is not None]
    first = min(starts) if starts else None
    reasons = []
    if not verifies:
        reasons.append("the file's content no longer matches its sha256 (a forecast was changed after hashing)")
    if posted is None:
        reasons.append("the hash is not in logs/c5_forecast_hashes.log")
    elif first is not None and not posted < first:
        reasons.append("the hash was posted after the first French run started")
    if sorted(french) != planned:
        reasons.append(f"forecast file lists {sorted(french)}, the plan {planned}")
    out["hash_protocol"] = {"sha256": doc.get("sha256"), "verifies": verifies, "posted_epoch": posted,
                            "posted": None if posted is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(posted)),
                            "first_french_start": None if first is None else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first)),
                            "prospective": not reasons, "reasons": reasons}
    code = ctx.code
    meas, miss = {}, []
    for rid in french:
        ok, why = ctx.usable(rid)
        if not ok:
            miss.append(f"{rid}: {why}")
            continue
        for c in ("dclm", "code", "math"):
            f = ctx.F(rid, code if c == "code" else c, "raw")
            if f is not None:
                meas[(rid, c)] = f["F"]
    bench = defaultdict(list)
    for o in doc.get("fit_runs") or []:
        if o.get("F") is not None and o["F"] > 0:
            bench[(o.get("arch"), o.get("corpus"))].append(math.log(o["F"]))
    arch_of = {rid: (ctx.runs.get(rid) or {}).get("arch") or _arch((ctx.runs.get(rid) or {}).get("base")) for rid in french}
    names = sorted({k for row in doc.get("forecasts") or [] for k in (row.get("forecasts") or {})})
    per = {}
    for nm in names:
        pts, by_c, na = [], defaultdict(list), 0
        for row in doc.get("forecasts") or []:
            fh = (row.get("forecasts") or {}).get(nm)
            F = meas.get((row["run"], row["corpus"]))
            bk = bench.get((arch_of.get(row["run"]), row["corpus"]))
            if fh is None or F is None or not (fh > 0 and F > 0) or not bk:
                na += 1
                continue
            p = (math.log(fh), math.log(F), float(np.mean(bk)))
            pts.append(p)
            by_c[row["corpus"]].append(p)
        per[nm] = {"pooled": _metrics(pts), "per_corpus": {c: _metrics(v) for c, v in sorted(by_c.items())}, "not_scored": na}
    pairs = defaultdict(dict)
    for rid in french:
        r = ctx.runs.get(rid) or {}
        if r.get("role") == "c5_pair" and r.get("side") in ("A", "B"):
            pairs[int(r["d"])][r["side"]] = rid
    fc = {(row["run"], row["corpus"]): row.get("forecasts") or {} for row in doc.get("forecasts") or []}
    within = {}
    for nm in names:
        I, Ih = [], []
        for d, ab in sorted(pairs.items()):
            if set(ab) != {"A", "B"}:
                continue
            v = {(s, c): (meas.get((ab[s], c)), (fc.get((ab[s], c)) or {}).get(nm)) for s in ("A", "B") for c in ("code", "math")}
            if all(x[0] and x[0] > 0 and x[1] and x[1] > 0 for x in v.values()):
                lg_ = lambda s, c, k: math.log(v[(s, c)][k])  # noqa: E731
                I.append((lg_("B", "code", 0) - lg_("A", "code", 0)) - (lg_("B", "math", 0) - lg_("A", "math", 0)))
                Ih.append((lg_("B", "code", 1) - lg_("A", "code", 1)) - (lg_("B", "math", 1) - lg_("A", "math", 1)))
        w = {"n": len(I), "I": I, "I_hat": Ih}
        if len(I) >= 3:
            sst = float(((np.array(I) - np.mean(I)) ** 2).sum())
            w["R2_within"] = 1.0 - float(((np.array(I) - np.array(Ih)) ** 2).sum()) / sst if sst > 0 else None
        else:
            w["R2_within"] = None
            w["note"] = "fewer than 3 French crossover pairs: R2_within not defined (contrasts listed)"
        within[nm] = w
    out.update(forecasters_in_file=names, forecasters_registered=8,
               forecasters_not_in_file="the hashed file holds the forecasters forecasts.py c5-french computes; the "
                                       "other registered forecasters are not forecast for French",
               runs_not_usable=miss, metrics=per, R2_within=within,
               label="prospective" if not reasons else "non-prospective",
               benchmark="training-fold mean of log F per (architecture, corpus) over the file's fit runs")
    return out


# ============================================================================= C2-eq b_Muon - b_Muon^rand and SNR
def _c2_stop(ctx) -> int:
    """P-16: the stop step of the C2 optimizer contrast -- the smallest stop_after of the C2 runs (Muon 8; the SGD runs stop
    after step 24 since MF-P1 fix A), 0 if any C2 run runs its whole schedule."""
    s = {int(r.get("stop_after") or 0) for r in ctx.plan.get("runs", []) if r.get("role") in ("c2_sgd", "c2_muon")}
    return min(s) if s and 0 not in s else 0


def block_c2eq_rand(ctx, H) -> dict:
    out = {"prereg": "v1.2 sec. 8.2 item 7 (exploratory); 'its alignment component' = b_Muon - b_Muon^rand "
                     "(theory verification v2 P-4: learned minus norm-matched random)"}
    cells = defaultdict(dict)
    for r in ctx.plan.get("runs", []):
        if r.get("role") == "c2_muon":
            cells[(r["base"], int(r["d"]))][r["side"]] = r
    have = [r["id"] for c in cells.values() for r in c.values() if ctx.haar(r["id"])]
    if not have:
        out["status"] = ("not computable from registered files: no antithetic random-direction (randdir) evaluation is "
                         "planned for the C2 Muon runs")
    else:
        recs = []
        for (b, d), rr in sorted(cells.items()):
            if set(rr) != {"H", "L"}:
                continue
            ok = all(ctx.usable(rr[s]["id"])[0] for s in ("H", "L"))
            dq = rr["H"].get("Delta_Q")
            v = {s: (ctx.F(rr[s]["id"], "dclm", "raw"), ctx.F(rr[s]["id"], "dclm", "rand")) for s in ("H", "L")}
            if not ok or dq in (None, 0) or any(x is None for p in v.values() for x in p):
                recs.append({"base": b, "d": d, "status": "run not usable or evaluation missing"})
                continue
            if any(x["F"] <= 0 for p in v.values() for x in p):
                recs.append({"base": b, "d": d, "status": "F <= 0: log elasticity not defined"})
                continue
            bm = (math.log(v["H"][0]["F"]) - math.log(v["L"][0]["F"])) / float(dq)
            br = (math.log(v["H"][1]["F"]) - math.log(v["L"][1]["F"])) / float(dq)
            recs.append({"base": b, "d": d, "Delta_Q": float(dq), "b_Muon": bm, "b_Muon_rand": br, "alignment": bm - br})
        al = [r["alignment"] for r in recs if "alignment" in r]
        s_stop = _c2_stop(ctx)
        out.update(step=s_stop or 96, per_draw=recs, mean_alignment=H.t_interval(al, 0.95),
                   note=(f"both elasticities at the final checkpoint, which is the step-{s_stop} state (the C2 runs "
                         f"stopped after step {s_stop}, P-16), where the randdir pair is evaluated") if s_stop else
                   "both elasticities at the final checkpoint (step 96), where the randdir pair is evaluated")
    g2p = ctx.p("g2/gate_record_G2.json")
    if os.path.exists(g2p):
        g5 = (ctx.json(g2p).get("gates") or {}).get("G2-5") or {}
        out["per_expert_gradient_SNR"] = {"source": "g2/gate_record_G2.json gate G2-5 (g2_gate.py)",
                                          "status": g5.get("status"), "values": g5.get("values")}
    else:
        out["per_expert_gradient_SNR"] = {"status": "g2/gate_record_G2.json missing"}
    return out


# ============================================================================= matched gain (v1.2 sec. 11)
def _curve(ctx, rid: str, base: str, corpora: list[str], target: str, full: bool) -> dict:
    """{step: {'G': target gain, 'F': {set: F}}}: 64-window interim heads (full=False; theta_0 = the run's in-process
    step000 head when present, else the base file; step 96 = the head windows of the final evaluation) or full sets
    (full=True: interim/stepNNN__<set>__full.npz at the interim steps, paired with the base file as in the C2 block)."""
    run = ctx.runs.get(rid) or {}
    root = ctx.root_of(rid)
    final = int(run.get("steps", 96))
    steps = sorted({int(s) for s in (run.get("interim_full") or {}).get("steps", [])} if full else
                   {int(s) for s in (run.get("interim_steps") or []) if int(s) > 0})
    sets = list(dict.fromkeys(list(corpora) + [target]))
    head = {}
    pts = {}
    for s in steps + [final]:
        row = {}
        for sn in sets:
            if s == final:
                w = ctx.windows(rid, sn, "raw")
                if w is None:
                    row = None
                    break
                d = w["d"]
                if not full:
                    if sn not in head:
                        row = None
                        break
                    m = np.isin(w["index"], head[sn])
                    if int(m.sum()) != len(head[sn]):
                        raise ValueError(f"{rid}/{sn}: interim head windows not in the final evaluation")
                    d = d[m]
            else:
                p = os.path.join(root, "interim", f"step{s:03d}__{sn}{'__full' if full else ''}.npz") if root else None
                b0 = os.path.join(root, "interim", f"step000__{sn}.npz") if root else None
                if not full and b0 and os.path.exists(b0):
                    pp = ctx.pair(b0, p)
                else:
                    pp = ctx.pair(ctx.base_file(base, sn), p, subset=not full)
                if pp is None:
                    row = None
                    break
                if not full:
                    head.setdefault(sn, pp["index"])
                d = pp["cr"] - pp["cb"]
            row[sn] = float(np.mean(d))
        if row is None:
            continue
        pts[s] = {"G": -row[target], "F": {c: row[c] for c in corpora}}
    return pts


def _interp(pts: dict, c: str, Gs: float) -> float | None:
    """log F(G*) by linear interpolation in log G between adjacent evaluation steps (v1.2 sec. 11)."""
    seq = [(math.log(v["G"]), math.log(v["F"][c])) for s, v in sorted(pts.items()) if v["G"] > 0 and v["F"][c] > 0]
    if Gs is None or Gs <= 0 or len(seq) < 1:
        return None
    x = math.log(Gs)
    for x0, y0 in seq:
        if abs(x0 - x) < 1e-15:
            return y0
    for (x0, y0), (x1, y1) in zip(seq[:-1], seq[1:]):
        if min(x0, x1) <= x <= max(x0, x1) and x1 != x0:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def block_matched_gain(ctx, H) -> dict:
    out = {"prereg": "v1.2 sec. 11 (secondary; the confirmatory C2 test stays at s*)", "pareto": []}
    code = ctx.code

    def pareto(rid, role, base, pts, full):
        for s, v in sorted(pts.items()):
            out["pareto"].append({"run": rid, "role": role, "base": base, "step": s, "windows": "full" if full else "head64",
                                  "G": v["G"], "F": v["F"]})

    # C1: A vs B
    draws, dropped, _ = ctx.screen(ctx.draws_of("c1"))
    recs = []
    for dr in draws:
        tgt = (ctx.runs.get(dr["A"]) or {}).get("target")
        c = {s: _curve(ctx, dr[s], dr["base"], [code, "math"], tgt, False) for s in ("A", "B")}
        for s in ("A", "B"):
            pareto(dr[s], "c1", dr["base"], c[s], False)
        fin = {s: c[s].get(max(c[s])) if c[s] else None for s in ("A", "B")}
        if not all(fin.values()):
            recs.append({"base": dr["base"], "d": dr["d"], "status": "evaluation points missing"})
            continue
        Gs = min(fin["A"]["G"], fin["B"]["G"])
        lf = {(s, k): _interp(c[s], k, Gs) for s in ("A", "B") for k in (code, "math")}
        rec = {"base": dr["base"], "d": dr["d"], "G_star": Gs, "G_final": {s: fin[s]["G"] for s in ("A", "B")}}
        if any(v is None for v in lf.values()):
            rec["status"] = "G* not bracketed by the evaluation points of one run (or F <= 0)"
        else:
            rec["F_ratio_B_over_A"] = {k: math.exp(lf[("B", k)] - lf[("A", k)]) for k in (code, "math")}
            rec["I_G"] = (lf[("B", code)] - lf[("A", code)]) - (lf[("B", "math")] - lf[("A", "math")])
        recs.append(rec)
    ig = [r["I_G"] for r in recs if "I_G" in r]
    out["C1"] = {"per_draw": recs, "dropped": dropped, "mean_I_G": H.t_interval(ig, 0.95),
                 "per_base": {b: H.t_interval([r["I_G"] for r in recs if r.get("base") == b and "I_G" in r], 0.95)
                              for b in sorted({r["base"] for r in recs})},
                 "windows": "64-window interim heads at steps 8/24/48 and the same windows of the final evaluation"}
    # C2: H vs L within an arm (P-16: deferred in the core analysis; not computable once the runs stopped early)
    s_stop = _c2_stop(ctx)
    if getattr(ctx, "mode", "full") == "core":
        out["C2"] = {"status": "deferred to the full analysis (P-16 core: C1 matched gain only)"}
    elif s_stop:
        out["C2"] = {"status": f"not computable: matched gain beyond step {s_stop} needs the C2 runs after step "
                               f"{s_stop}; they stopped after step {s_stop} (P-16)"}
    else:
        cells = defaultdict(dict)
        for r in ctx.plan.get("runs", []):
            if r.get("role") in ("c2_sgd", "c2_muon"):
                cells[(r["base"], int(r["d"]))][(r["arm"], r["side"])] = r
        c2 = []
        for (b, d), rr in sorted(cells.items()):
            rec = {"base": b, "d": d}
            r_arm = {}
            for arm in ("sgd", "muon"):
                if not all((arm, s) in rr and ctx.usable(rr[(arm, s)]["id"])[0] for s in ("H", "L")):
                    continue
                cv = {s: _curve(ctx, rr[(arm, s)]["id"], b, ["dclm"], rr[(arm, s)].get("target", "german"), True) for s in ("H", "L")}
                for s in ("H", "L"):
                    pareto(rr[(arm, s)]["id"], f"c2_{arm}", b, cv[s], True)
                if not all(cv.values()):
                    continue
                Gs = min(cv["H"][max(cv["H"])]["G"], cv["L"][max(cv["L"])]["G"])
                h, l_ = _interp(cv["H"], "dclm", Gs), _interp(cv["L"], "dclm", Gs)
                rec[f"G_star_{arm}"] = Gs
                if h is not None and l_ is not None:
                    r_arm[arm] = h - l_
                    rec[f"F_ratio_H_over_L_{arm}"] = math.exp(h - l_)
            if set(r_arm) == {"sgd", "muon"}:
                rec["r_sgd_G"], rec["r_muon_G"] = r_arm["sgd"], r_arm["muon"]
                rec["I2_G"] = r_arm["sgd"] - r_arm["muon"]
            else:
                rec["status"] = "arm runs not usable, evaluation points missing or G* not bracketed"
            c2.append(rec)
        out["C2"] = {"per_draw": c2, "mean_I2_G": H.t_interval([r["I2_G"] for r in c2 if "I2_G" in r], 0.95),
                     "windows": "full sets at the interim full steps and the final evaluation"}
    if getattr(ctx, "mode", "full") == "core":     # P-16: the FULL runs are C3 runs (after the core)
        out["FULL_across_architectures"] = {"status": "deferred to the full analysis (P-16: FULL runs belong to C3)"}
    else:
        full = defaultdict(dict)
        for r in ctx.plan.get("runs", []):
            if r.get("role") == "full" and ctx.usable(r["id"])[0]:
                full[(r.get("target"), int(r["seed"]))][r["base"]] = r
        fr = []
        for (dom, s), bb in sorted(full.items()):
            tset = code if dom == "code" else dom
            cv = {b: _curve(ctx, r["id"], b, ["dclm"], tset, False) for b, r in bb.items()}
            for b, r in bb.items():
                pareto(r["id"], "full", b, cv[b], False)
            for mb in ("moe_b1", "moe_b2"):
                if "nestor" not in cv or mb not in cv or not cv["nestor"] or not cv[mb]:
                    continue
                Gs = min(cv["nestor"][max(cv["nestor"])]["G"], cv[mb][max(cv[mb])]["G"])
                a, b_ = _interp(cv["nestor"], "dclm", Gs), _interp(cv[mb], "dclm", Gs)
                fr.append({"domain": dom, "seed": s, "pair": f"nestor/{mb}", "G_star": Gs,
                           "F_ratio": None if (a is None or b_ is None) else math.exp(a - b_)})
        out["FULL_across_architectures"] = {"pairs": fr, "label": "descriptive: raw MoE-versus-NESTOR comparison (v1.2 sec. "
                                            "10.1 allows only descriptive raw comparisons without a matched-base-loss checkpoint)"}
    out["pareto_note"] = "forgetting-gain Pareto points (rows of 'pareto'); the figure is drawn from these rows"
    return out


# ============================================================================= F_rel (v1.2 l. 327; sec. 9 item 13)
def _f_cont(ctx, H) -> dict:
    """{base: {'run', 'source', 'F': {set: {'F', 'SE'}}}} of the same-base DCLM-CONT run (confirm plan, else G1)."""
    out = {}
    for r in ctx.plan.get("runs", []):
        if r.get("role") == "dclm_cont" and ctx.usable(r["id"])[0]:
            F = {}
            for ev in r.get("evals") or []:
                for sn in ev.get("sets") or []:
                    f = ctx.F(r["id"], sn, "raw")
                    if f is not None:
                        F[sn] = f
            out[r["base"]] = {"run": r["id"], "source": "confirm", "F": F}
    g1 = H._gplan(ctx, "g1")
    if g1:
        gp, st = g1
        for r in gp.get("runs", []):
            if r.get("role") != "dclm_cont" or r["base"] in out or (st.get(r["id"]) or {}).get("status") != "done":
                continue
            root = ctx.p((st.get(r["id"]) or {}).get("root"))
            F = {}
            for sn in r.get("eval_sets") or []:
                pp = ctx.pair(ctx.base_file(r["base"], sn), os.path.join(root, r.get("eval_dir", "eval_main"), f"ckpt__{sn}.npz"))
                if pp is not None:
                    F[sn] = _stat(pp["cr"] - pp["cb"])
            out[r["base"]] = {"run": r["id"], "source": "G1", "F": F}
    return out


def block_frel(ctx, H) -> dict:
    fc = _f_cont(ctx, H)
    out = {"prereg": "v1.2 l. 327 (F_rel = F - F_DCLM-CONT) and sec. 9 item 13 (secondary)",
           "F_cont": {b: {"run": v["run"], "source": v["source"], "F": {s: x["F"] for s, x in v["F"].items()}} for b, v in fc.items()}}

    def rel(rid, sn, kind="raw"):
        b = (ctx.runs.get(rid) or {}).get("base")
        f, c = ctx.F(rid, sn, kind), ((fc.get(b) or {}).get("F") or {}).get(sn)
        if f is None or c is None:
            return None
        return {"F": f["F"] - c["F"], "SE": math.sqrt(f["SE"] ** 2 + c["SE"] ** 2)}

    out["FULL"] = [{"run": r["id"], "base": r["base"], "domain": r.get("target"), "seed": r.get("seed"),
                    "F_dclm": (ctx.F(r["id"], "dclm", "raw") or {}).get("F"), "F_rel_dclm": (rel(r["id"], "dclm") or {}).get("F")}
                   for r in ctx.plan.get("runs", []) if r.get("role") == "full" and ctx.usable(r["id"])[0]]
    draws, dropped, _ = ctx.screen(ctx.draws_of("c1"))
    beta0 = {"moe_b1": ctx.params.get("beta0_1"), "moe_b2": ctx.params.get("beta0_2")}
    X = ctx.cross(draws, "raw", (ctx.code, "math"), Fget=lambda rid, c: rel(rid, c))
    J = [i - float(beta0[b]) * p for i, p, b in zip(X["I"], X["Pv"], X["bases"]) if beta0.get(b) is not None]
    out["C1"] = {"statistic": X["statistic"], "n": X["n"], "dropped": X["dropped"] + dropped,
                 "per_draw": [{"base": b, "d": d, "I_rel": i, "P": p} for b, d, i, p in zip(X["bases"], X["ds"], X["I"], X["Pv"])],
                 "mean_J": H.t_interval(J, 0.95), "signflip": H.signflip_p(J, "greater") if J else None,
                 "beta0": beta0}
    s_stop = _c2_stop(ctx)
    if s_stop:                                 # P-16: no 96-step C2 runs
        out["C2"] = {"status": f"not computable: F_rel at step 96 needs the 96-step C2 runs; they stopped after "
                               f"step {s_stop} (P-16)"}
    else:
        # C2 at the final checkpoint (DCLM-CONT is evaluated at step 96 only)
        cells = defaultdict(dict)
        for r in ctx.plan.get("runs", []):
            if r.get("role") in ("c2_sgd", "c2_muon"):
                cells[(r["base"], int(r["d"]))][(r["arm"], r["side"])] = r
        recs = []
        for (b, d), rr in sorted(cells.items()):
            if len(rr) != 4 or not all(ctx.usable(x["id"])[0] for x in rr.values()):
                continue
            v = {k: rel(x["id"], "dclm") for k, x in rr.items()}
            dq = rr[("sgd", "H")].get("Delta_Q")
            if any(x is None for x in v.values()) or dq is None:
                continue
            rec = {"base": b, "d": d, "Delta_Q": float(dq), "F_rel": {f"{a}|{s}": x["F"] for (a, s), x in v.items()}}
            if all(x["F"] > 0 for x in v.values()):
                r_ = {a: math.log(v[(a, "H")]["F"]) - math.log(v[(a, "L")]["F"]) for a in ("sgd", "muon")}
                rec["I2_rel"] = r_["sgd"] - r_["muon"]
                rec["J2_rel"] = rec["I2_rel"] - H.BETA_REF * float(dq)
            recs.append(rec)
        j2 = [r["J2_rel"] for r in recs if "J2_rel" in r]
        out["C2"] = {"step": 96, "per_draw": recs, "mean_J2": H.t_interval(j2, 0.95),
                     "signflip": H.signflip_p(j2, "greater") if j2 else None,
                     "note": "at step 96 (DCLM-CONT has no interim evaluation, so F_rel at s* < 96 is not defined); log "
                             "statistic only where all four F_rel > 0"}
    return out


# ============================================================================= bits per byte (v1.2 sec. 9 item 13)
def _token_reader(root: str | None = None):
    """The authors' open_token_bin (nestor/data/token_documents.py; numpy only), loaded from its file: importing it as
    nestor.data.token_documents would run nestor/data/__init__.py, which imports the torch datasets. The package is
    looked up as eval_windows.nestor_env puts it on the path: an importable nestor, else <root>/repo/nestor (the
    analysis root, then the directory of eval_windows.py)."""
    import importlib.util
    import eval_windows as EW
    cands = []
    spec = importlib.util.find_spec("nestor")                    # finds the package without executing it
    if spec is not None and spec.submodule_search_locations:
        cands.append(list(spec.submodule_search_locations)[0])
    for r in (root, os.path.dirname(os.path.abspath(EW.__file__))):
        if r:
            cands.append(os.path.join(r, "repo", "nestor"))
    for c in cands:
        f = os.path.join(c, "data", "token_documents.py")
        if os.path.isfile(f):
            s2 = importlib.util.spec_from_file_location("_nestor_token_documents", f)
            m = importlib.util.module_from_spec(s2)
            s2.loader.exec_module(m)
            return m.open_token_bin
    raise ImportError("nestor package not found (importable, or <root>/repo/nestor)")


def _bytes_from_bin(ctx, res: dict):
    """n_bytes per window from the registered data bin (eval_windows --bytes rule: UTF-8 bytes of the target tokens
    arr[1:], tiktoken cl100k_base, special ids 0), after checking every window's token hash; None + reason otherwise."""
    st = (res.get("meta") or {}).get("set") or {}
    path, block = st.get("path"), int(st.get("block", 4096))
    if not path or "starts" not in res:
        return None, "the theta_0 file records no data path or window starts"
    try:
        import eval_windows as EW
        bl = EW.token_byte_lengths(200000)
    except Exception as e:                                       # noqa: BLE001 - recorded
        return None, f"token byte lengths unavailable ({type(e).__name__})"
    if bl is None:
        return None, "tiktoken unavailable"
    try:
        toks, _ = _token_reader(getattr(ctx, "root", None))(path)
    except Exception as e:                                       # noqa: BLE001 - recorded
        return None, f"data bin {path} not readable ({type(e).__name__}: {e})"
    nb = np.zeros(len(res["starts"]), dtype=np.int64)
    for i, s in enumerate(np.asarray(res["starts"], dtype=np.int64)):
        arr = np.asarray(toks[int(s): int(s) + block], dtype=np.int64)
        if np.uint64(EW.window_hash(arr)) != np.uint64(np.asarray(res["hash"]).astype(np.uint64)[i]):
            return None, f"{path}: window {i} token hash differs from the evaluation file"
        nb[i] = int(bl[np.clip(arr[1:], 0, len(bl) - 1)].sum())
    return nb, f"computed from {path} (tiktoken cl100k_base; hashes verified)"


def block_bpb(ctx, H) -> dict:
    out = {"prereg": "v1.2 sec. 9 item 13 (secondary)",
           "identity": "With F_bpb(c) = sum_w (CE_sum_run - CE_sum_0) / (ln 2 * sum_w n_bytes) (eval_windows "
                       "diff_bits_per_byte) and equal-token windows, F_bpb(c) = kappa_c * F(c) with one factor per corpus "
                       "and window set; every C1 / C2 statistic compares one corpus on identical windows, so its log and "
                       "normalized-difference forms are unchanged in bits per byte. Only F levels change."}
    sets = sorted({ctx.code, "math", "dclm"})
    kap = {}
    for base in sorted(ctx.base_files):
        for sn in sets:
            bp = ctx.base_file(base, sn)
            if not bp or not os.path.exists(bp):
                continue
            import eval_windows as EW
            res = EW.load_result(bp)
            ctx._read(bp)
            if "n_bytes" in res:
                nb, src = np.asarray(res["n_bytes"], dtype=np.int64), "n_bytes of the theta_0 evaluation file"
            else:
                nb, src = _bytes_from_bin(ctx, res)
            if nb is None:
                kap[f"{base}|{sn}"] = {"status": f"not computed: {src}"}
                continue
            nt = np.asarray(res["n_tok"], dtype=np.int64)
            kap[f"{base}|{sn}"] = {"kappa": float(nt.sum()) / float(nb.sum()) / LN2, "bytes_per_token": float(nb.sum()) / float(nt.sum()),
                                   "windows": int(nb.size), "source": src}
    out["kappa"] = kap
    ok = {k: v["kappa"] for k, v in kap.items() if "kappa" in v}
    if not ok:
        out["status"] = "not computed: no byte counts (see kappa entries)"
        return out
    # numerical check of the identity on C1: I in bits per byte equals I in nats per token
    draws, _, _ = ctx.screen(ctx.draws_of("c1"))

    def fb(rid, c):
        b = (ctx.runs.get(rid) or {}).get("base")
        f, k = ctx.F(rid, c, "raw"), ok.get(f"{b}|{c}")
        return None if (f is None or k is None) else {"F": f["F"] * k, "SE": f["SE"] * k}

    Xb = ctx.cross(draws, "raw", (ctx.code, "math"), Fget=fb)
    Xn = ctx.cross(draws, "raw", (ctx.code, "math"))
    diffs = [abs(a - b) for a, b in zip(Xb["I"], Xn["I"])] if Xb["n"] == Xn["n"] else None
    out["C1_check"] = {"n": Xb["n"], "statistic": Xb["statistic"], "max_abs_I_bpb_minus_I": max(diffs) if diffs else None,
                       "F_bpb_levels": [{"base": r["base"], "d": r["d"], "F": r["F"]} for r in Xb["draws"]]}
    return out


# ============================================================================= driver + markdown
BLOCKS = (("C3", block_c3), ("C3_per_layer", block_c3_layer), ("C4", block_c4), ("C5_french", block_c5_french),
          ("C2_eq_rand", block_c2eq_rand), ("matched_gain", block_matched_gain), ("F_rel", block_frel), ("bpb", block_bpb))


def run_ext(ctx, H) -> dict:
    out = {}
    for name, fn in BLOCKS:
        try:
            out[name] = fn(ctx, H)
        except Exception as e:                                   # recorded, never hidden
            out[name] = {"status": f"failed: {type(e).__name__}: {e}"}
            ctx.warnings.append(f"block {name} failed: {type(e).__name__}: {e}")
    return out


def not_computable(blocks: dict) -> list[str]:
    out = []
    for name, _ in BLOCKS:
        b = blocks.get(name) or {}
        if str(b.get("status", "")).startswith(("not computable", "not computed", "failed")):
            out.append(f"{name}: {b['status']}")
    rp = ((blocks.get("C3") or {}).get("random_partial_revert") or {}).get("status", "")
    if rp.startswith("not computable"):
        out.append(f"C3 random partial-revert control: {rp}")
    for name, sub in (("matched_gain", "C2"), ("F_rel", "C2")):          # P-16: C2 analyses needing steps > 8
        st_ = str(((blocks.get(name) or {}).get(sub) or {}).get("status", ""))
        if st_.startswith("not computable"):
            out.append(f"{name} {sub}: {st_}")
    return out


def _f(x, n=4):
    return "-" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))


def _iv(v):
    return "-" if not v else "[" + ", ".join(_f(x) for x in v) + "]"


def _ci(d):
    if not d:
        return "-"
    if d.get("lo") is None:
        return f"{_f(d.get('mean'))} (points)"
    return f"{_f(d.get('mean'))} [{_f(d.get('lo'))}, {_f(d.get('hi'))}]"


def markdown(doc: dict) -> list[str]:
    B = doc.get("blocks") or {}
    L = []
    if "C3" in B and not str(B["C3"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        c3 = B.get("C3") or {}
        L += ["## C3 endpoint revert decomposition (v1.2 sec. 10.1)", "", c3.get("statement", ""), ""]
        if c3.get("per_base"):
            L += ["| base | g | share_g | share_g' | iota / F | runs |", "|---|---|---|---|---|---|"]
            for b, v in c3["per_base"].items():
                s = v["shares"]
                L.append(f"| {b} | {v['g']} | {_ci(s['share_g'])} | {_ci(s['share_g2'])} | {_ci(s['iota_over_F'])} | {v['runs']} |")
            L.append("")
        for a, v in (c3.get("freeze_cross_check") or {}).items():
            if isinstance(v, dict) and "run" in v:
                L.append(f"- freeze {a}: F_freeze {_f(v.get('F_freeze'))}; freeze attribution {_f(v.get('freeze_attribution'))} vs "
                         f"revert attribution {_f(v.get('revert_attribution'))} ({v.get('status') or 'ok'})")
        L += [f"- random partial revert: {(c3.get('random_partial_revert') or {}).get('status')}",
              f"- NESTOR: {(c3.get('NESTOR_label') or {}).get('label')}; MoE-NESTOR comparison: "
              f"{(c3.get('nestor_moe_comparison') or {}).get('reason')}", ""]
    if "C3_per_layer" in B and not str(B["C3_per_layer"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        pl = B.get("C3_per_layer") or {}
        L += ["## C3 per layer (v1.3 sec. 4.8)", ""]
        for b, v in (pl.get("per_base") or {}).items():
            L.append(f"- {b} ({v['group']}, domains {', '.join(map(str, v['domains']))}): additivity gap mean "
                     f"{_f(v['additivity_gap']['mean'])} range {_iv(v['additivity_gap']['range'])}; sum / all-group revert "
                     f"{_f(v['sum_over_all_group_revert']['mean'])} range {_iv(v['sum_over_all_group_revert']['range'])}")
    if "C4" in B and not str(B["C4"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        c4 = B.get("C4") or {}
        L += ["", "## C4 per-window difference-in-differences (v1.2 sec. 10.2)", ""]
        if c4.get("overall"):
            o = c4["overall"]
            L.append(f"Overall: n = {o['n']}, mean beta_C4 = {_f(o['mean_beta_C4'])}, t95 {_ci(o['t95'])}, two-way bootstrap "
                     f"{_iv((o.get('bootstrap') or {}).get('ci95'))}; mean within-draw corr {_f(o['mean_within_draw_corr'])}")
            for b, s in (c4.get("per_base") or {}).items():
                L.append(f"- {b}: n = {s['n']}, mean {_f(s['mean_beta_C4'])}, t95 {_ci(s['t95'])}, two-way bootstrap "
                         f"{_iv((s.get('bootstrap') or {}).get('ci95'))}")
        else:
            L.append(f"status: {c4.get('status')}")
    if "C5_french" in B and not str(B["C5_french"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        c5 = B.get("C5_french") or {}
        L += ["", "## C5 French (v1.2 secs. 10.3, 12)", "", f"Label: **{c5.get('label', c5.get('status'))}**"
              + (f" -- {'; '.join(c5['hash_protocol']['reasons'])}" if (c5.get("hash_protocol") or {}).get("reasons") else ""), ""]
        for nm, m in (c5.get("metrics") or {}).items():
            p = m["pooled"]
            L.append(f"- {nm}: n = {p.get('n')}, R2_oos {_f(p.get('R2_oos'))}, Spearman {_f(p.get('spearman'))}, median "
                     f"|log ratio| {_f(p.get('median_abs_log_ratio'))}")
    if "C2_eq_rand" in B and not str(B["C2_eq_rand"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        ce = B.get("C2_eq_rand") or {}
        L += ["", "## C2-eq alignment component and per-expert SNR (v1.2 sec. 8.2 item 7)", "",
              f"b_Muon - b_Muon^rand: {ce.get('status') or _ci(ce.get('mean_alignment'))}",
              f"Per-expert gradient SNR (G2-5): {json.dumps(((ce.get('per_expert_gradient_SNR') or {}).get('values') or {}).get('summary'))}", ""]
    if "matched_gain" in B and not str(B["matched_gain"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        mg = B.get("matched_gain") or {}
        L += ["## Matched gain (v1.2 sec. 11)", "",
              f"- C1 mean I_G: {_ci((mg.get('C1') or {}).get('mean_I_G'))}",
              f"- C2 mean I2_G: {_ci((mg.get('C2') or {}).get('mean_I2_G'))}",
              f"- FULL across architectures: {len((mg.get('FULL_across_architectures') or {}).get('pairs') or [])} pairs (descriptive)",
              f"- Pareto points: {len(mg.get('pareto') or [])}", ""]
    if "F_rel" in B and not str(B["F_rel"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        fr = B.get("F_rel") or {}
        L += ["## F_rel versions (v1.2 sec. 9 item 13)", "",
              f"- C1 ({(fr.get('C1') or {}).get('statistic')}): mean J_rel {_ci((fr.get('C1') or {}).get('mean_J'))}, sign-flip p "
              f"{_f(((fr.get('C1') or {}).get('signflip') or {}).get('p'))}",
              f"- C2 (step 96): mean J2_rel {_ci((fr.get('C2') or {}).get('mean_J2'))}", ""]
    if "bpb" in B and not str(B["bpb"].get("status", "")).startswith("deferred"):  # P-16: deferred blocks
        bp = B.get("bpb") or {}
        L += ["## Bits per byte (v1.2 sec. 9 item 13)", "", bp.get("identity", ""), "",
              f"Status: {bp.get('status') or 'computed'}; C1 check max |I_bpb - I| = {_f((bp.get('C1_check') or {}).get('max_abs_I_bpb_minus_I'))}", ""]
    L += ["## Analysed elsewhere", ""]
    for k, v in POINTERS.items():
        L.append(f"- {k} ({v['prereg']}): {v['script']}; results {', '.join(v['results'])}; note {v['note']}"
                 + (f"; figure {v['figure']}" if v.get("figure") else ""))
    nc = doc.get("not_computable_from_registered_files") or []
    L += ["", "## Not computable from registered files", ""] + ([f"- {x}" for x in nc] or ["- none"]) + [""]
    return L
