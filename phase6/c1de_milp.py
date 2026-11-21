"""C1-DE exact pair generator: exploratory German C1 block (prereg v1.3 version 5; the spec is pending).

Draws matched A/B subset pairs for the German target INSIDE THE REGISTERED POOLS with an exact binary integer program
(scipy.optimize.milp = HiGHS) instead of the registered swap search, which fails the G0-2 rule on the German cells
(G0-1: German infeasible). Theory verification v4 (Table R1) shows that exact pairs exist at f = 1/2 within log 1.05 on
both MoE bases.

Kept from the registration (frozen crossover.py, sha256 d3fcf6c2...):
  tables    crossover.reg_load_dir on the G0 re-run directories, with the layout and half seed of the G0 record;
  tilt      crossover.reg_tilt(sel, 'c1', code) = log s2_code - log s2_math on SELECTION statistics, with the C1 code
            stream of the G0 record (decisions.code_stream.chosen);
  pools     per stratum the npool = round(f N) lowest-tilt experts (side A) and highest-tilt experts (side B), stable
            argsort (== crossover.reg_propose); f = 1/2;
  k         crossover.reg_k = round(N / 4) experts per stratum on each side, exactly;
  matching  crossover.reg_matched('c1', target='german') = [german, dclm]: |log RO_c(A) - log RO_c(B)| <= REG_TOL
            (log 1.05) on selection statistics; every pair is re-checked with the frozen crossover.reg_match_mask;
  record    the draw-record fields of crossover.reg_draw (the confirm plan reads them as it reads books draws), plus
            'selection' (selection-half P and match residuals) and the MILP details under 'generator'.
Generator rules (prereg v1.3 version 5, sec. 4.12; equal to the reference c1de_milp_dryrun.py):
  objective  maximise sum w_i x_i, w ~ N(0, 1) i.i.d. from numpy default_rng(5600 + d), one weight per membership
             variable, A pool before B pool, layer-major (each pool in the registered stable-argsort order);
             scipy.optimize.milp (HiGHS), mip_rel_gap 0, time limit 600 s per solve; solver version recorded;
  tolerance  the MILP uses tau = log 1.05 - 1e-6; if the frozen reg_match_mask (log 1.05, selection half) rejects the
             solution, the problem is solved again with tau lowered by 1e-4, at most three times (status mask_failed);
  duplicates a pair identical to an earlier accepted draw of the same base is discarded and the weights are redrawn
             from the same generator, at most 20 weight vectors per draw (status duplicates); no rule uses P_d;
  failure    a solver failure (time limit, infeasible) ends the draw (status milp_failed); a draw that is not ok is not
             run, and this is recorded.
Only selection-half statistics enter a draw (tilt, pools, matching, objective). predicted.P in each record is the
registered estimation-half P (crossover.reg_predicted on est), computed after the draw exactly as in books records;
selection.P is the same formula on the selection half.

  python c1de_milp.py --g0-record g0/gate_record.json [--bases moe_b1,moe_b2] [--draws 3] [--out-dir results/c1de]
Writes <out-dir>/c1de_<base>_german.json (+ .sha256, crossover.write_manifest format) per base and
<out-dir>/c1de_record.json (summary). Exit codes: 0 every draw ok; 4 at least one draw not ok (recorded); 2 inputs
missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import crossover as CX  # noqa: E402

SEED_BASE = 5600                  # draw d: objective seed 5600 + d (2026-11-14)
TARGET = "german"
F = 0.5                           # registered pool fraction (crossover.REG_POOLS[0])
N_DRAWS = 3
BASES = ("moe_b1", "moe_b2")
MARGIN = 1e-6                     # MILP tolerance = REG_TOL - 1e-6 (sec. 4.12)
TAU_STEP = 1e-4                   # tolerance decrement after a mask rejection
MAX_RESOLVES = 3                  # re-solves after a mask rejection (4 solves at most per weight vector)
MAX_WEIGHTS = 20                  # weight vectors per draw (duplicate rule)
TIME_LIMIT = 600.0                # seconds per solve (HiGHS)
BLOCK = "C1-DE"


def _sha(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pools(tilt: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Registered pools (== crossover.reg_propose): per stratum the npool lowest-tilt (A) and highest-tilt (B) experts."""
    L, N = tilt.shape
    npool = int(round(f * N))
    order = np.argsort(tilt, axis=1, kind="stable")
    return order[:, :npool], order[:, N - npool:]


def build_milp(sel: dict, tilt: np.ndarray, k: int, f: float, corpora: list[str], tau: float):
    """Pools, cardinality rows (exactly k per stratum and pool; rows interleaved A_l, B_l as in the reference) and the
    linear form of the matching W_B - e^tau W_A <= 0 and W_A - e^tau W_B <= 0 per matched corpus, W_c(S) = RO_c(S)
    on selection statistics (rows scaled by 1 / (mean share of the two pools))."""
    lo, hi = pools(tilt, f)
    L, npool = lo.shape
    if npool < k:
        raise ValueError(f"pool of {npool} experts is smaller than k = {k}")
    nA = L * npool
    Aeq = np.zeros((2 * L, 2 * nA))
    for l in range(L):
        Aeq[2 * l, l * npool:(l + 1) * npool] = 1.0
        Aeq[2 * l + 1, nA + l * npool:nA + (l + 1) * npool] = 1.0
    et = math.exp(tau)
    rows = []
    for c in corpora:
        s2 = np.asarray(sel[c], dtype=float)
        w = s2 / s2.sum()
        wa, wb = np.take_along_axis(w, lo, 1).ravel(), np.take_along_axis(w, hi, 1).ravel()
        s = 1.0 / max(wa.mean() + wb.mean(), 1e-300)
        rows.append(np.concatenate([-et * wa, wb]) * s)
        rows.append(np.concatenate([wa, -et * wb]) * s)
    return lo, hi, Aeq, np.array(rows)


def objective(seed: int, n: int) -> np.ndarray:
    """First weight vector of a draw: w ~ N(0, 1) i.i.d., numpy default_rng(seed) (later vectors continue the stream)."""
    return np.random.default_rng(int(seed)).standard_normal(int(n))


def solve(sel: dict, tilt: np.ndarray, k: int, f: float, corpora: list[str], tau: float, w: np.ndarray,
          time_limit: float = TIME_LIMIT) -> dict:
    """One MILP: maximise w.x over the pool-subset pairs matched within tau. Status 'optimal' (A, B as [L, k] arrays,
    sorted per stratum) or 'failed' (time limit, infeasible or another solver status)."""
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp

    lo, hi, Aeq, Ain = build_milp(sel, tilt, k, f, corpora, tau)
    L, npool = lo.shape
    n = Aeq.shape[1]
    w = np.asarray(w, dtype=float)
    cons = [LinearConstraint(Aeq, float(k), float(k)), LinearConstraint(Ain, -np.inf, 0.0)]
    t0 = time.time()
    res = milp(-w, integrality=np.ones(n), bounds=Bounds(0, 1), constraints=cons,
               options={"time_limit": float(time_limit), "mip_rel_gap": 0.0, "disp": False})
    sec = time.time() - t0
    # R6: HiGHS reports status 1 / 'Time limit reached'; such a solve is not reproducible on another machine
    tl = int(res.status) == 1 or "time limit" in str(res.message).lower() or sec >= 0.999 * float(time_limit)
    out = {"milp_status": int(res.status), "message": str(res.message), "sec": round(sec, 4), "time_limit_hit": bool(tl),
           "n_binaries": int(n), "npool": int(npool), "tau": float(tau), "scipy": scipy.__version__,
           "numpy": np.__version__}
    if res.status != 0 or res.x is None:
        out["status"] = "failed"
        return out
    x = np.round(res.x).astype(int)
    a, b = x[:L * npool].reshape(L, npool), x[L * npool:].reshape(L, npool)
    if not (np.all(a.sum(1) == k) and np.all(b.sum(1) == k)):
        out["status"] = "failed"
        out["message"] += " | rounded memberships do not have k per stratum"
        return out
    out.update(status="optimal", objective=float(-res.fun), mip_gap=float(getattr(res, "mip_gap", float("nan"))),
               A=np.sort(np.stack([lo[l][a[l] == 1] for l in range(L)]), axis=1),
               B=np.sort(np.stack([hi[l][b[l] == 1] for l in range(L)]), axis=1))
    return out


def checks(sel: dict, tilt: np.ndarray, A: np.ndarray, B: np.ndarray, k: int, f: float, corpora: list[str],
           tol: float) -> dict:
    """Acceptance checks of a pair: frozen reg_match_mask at tol, exactly k distinct experts per stratum and side,
    every member in its registered pool, A and B disjoint."""
    lo, hi = pools(tilt, f)
    L = tilt.shape[0]
    card = all(len(set(A[l].tolist())) == k == len(A[l]) and len(set(B[l].tolist())) == k == len(B[l]) for l in range(L))
    in_pools = all(set(A[l].tolist()) <= set(lo[l].tolist()) and set(B[l].tolist()) <= set(hi[l].tolist())
                   for l in range(L))
    disjoint = all(not (set(A[l].tolist()) & set(B[l].tolist())) for l in range(L))
    mask = bool(CX.reg_match_mask(sel, A[None], B[None], corpora, tol)[0])
    return {"reg_match_mask": mask, "k_per_stratum": bool(card), "in_registered_pools": bool(in_pools),
            "disjoint": bool(disjoint), "ok": bool(mask and card and in_pools and disjoint)}


def draw(t: CX.RegTables, d: int, *, code: str, earlier: list | None = None, target: str = TARGET, f: float = F,
         k: int | None = None, tol: float = CX.REG_TOL, seed_base: int = SEED_BASE, margin: float = MARGIN,
         tau_step: float = TAU_STEP, max_resolves: int = MAX_RESOLVES, max_weights: int = MAX_WEIGHTS,
         time_limit: float = TIME_LIMIT) -> dict:
    """Draw d of the C1-DE generator on one base (sec. 4.12), in the draw-record format of crossover.reg_draw.
    earlier: (A, B) arrays of the accepted earlier draws of the same base (duplicate rule)."""
    k = CX.reg_k(t) if k is None else int(k)
    seed = int(seed_base) + int(d)
    tilt = CX.reg_tilt(t.sel, "c1", code=code)
    corpora = CX.reg_matched("c1", target=target, code=code)
    lo, _ = pools(tilt, f)
    n = 2 * lo.shape[0] * lo.shape[1]
    entry = {"design": "c1", "block": BLOCK, "d": int(d), "seed": seed, "f": f, "k": k, "q": k / t.N, "N": t.N,
             "L": t.L, "family": t.family, "strata": t.strata, "excluded_strata": t.excluded, "target": target,
             "code": code, "tol": tol, "matched_corpora": corpora}
    gen = {"kind": "milp_exact", "solver": "scipy.optimize.milp (HiGHS)", "seed_rule": f"{seed_base} + d",
           "objective": "maximise w.x, w ~ N(0,1) i.i.d. from numpy default_rng(seed) (redraws continue the stream); "
                        "x = memberships [A pools of strata 0..L-1, B pools of strata 0..L-1], registered pool order",
           "mip_rel_gap": 0.0, "time_limit_s": float(time_limit), "tau_0": tol - margin, "tau_step": tau_step,
           "max_resolves": int(max_resolves), "max_weights": int(max_weights), "statistics": "selection half only",
           "attempts": []}
    entry["generator"] = gen
    rng = np.random.default_rng(seed)
    earlier = list(earlier or [])
    for wi in range(int(max_weights)):
        w = rng.standard_normal(n)
        tau = tol - margin
        ok = False
        for _ in range(int(max_resolves) + 1):
            r = solve(t.sel, tilt, k, f, corpora, tau, w, time_limit)
            att = {kk: v for kk, v in r.items() if kk not in ("A", "B")}
            att["weights"] = wi
            gen["attempts"].append(att)
            gen["time_limit_hit"] = bool(gen.get("time_limit_hit") or att.get("time_limit_hit"))
            if r["status"] != "optimal":
                entry["status"] = "milp_failed"
                return entry
            A, B = r["A"], r["B"]
            chk = checks(t.sel, tilt, A, B, k, f, corpora, tol)
            att["checks"] = chk
            if chk["ok"]:
                ok = True
                break
            tau -= tau_step
        if not ok:
            entry["status"] = "mask_failed"
            return entry
        if any(np.array_equal(A, ea) and np.array_equal(B, eb) for ea, eb in earlier):
            att["duplicate"] = True
            continue
        err = CX.reg_match_errors(t.sel, A[None], B[None], corpora)[:, 0]
        p_est = float(CX.reg_predicted(t.est, A[None], B[None], "c1", target=target, code=code)[0])
        p_sel = float(CX.reg_predicted(t.sel, A[None], B[None], "c1", target=target, code=code)[0])
        ka, kb = t.subset_keys(A), t.subset_keys(B)
        entry.update(status="ok", A=A.tolist(), B=B.tolist(),
                     matched_values={c: {"log_RO_A": float(np.log(CX.reg_ro(t.sel[c], A))),
                                         "log_RO_B": float(np.log(CX.reg_ro(t.sel[c], B))), "diff": float(e)}
                                     for c, e in zip(corpora, err)},
                     predicted={"P": p_est}, subsets={"A": ka, "B": kb}, spec_A=CX.spec_for(ka),
                     spec_B=CX.spec_for(kb), sources=t.sources,
                     selection={"P": p_sel, "match_residuals": {c: float(e) for c, e in zip(corpora, err)},
                                "max_abs_residual": float(np.max(np.abs(err))), "tol": tol, "checks": chk})
        return entry
    entry["status"] = "duplicates"
    return entry


def overlaps(entries: list[dict], k: int, L: int) -> list[dict]:
    """Pairwise share of modules two accepted draws have in common, A and B separately (sec. 4.12 'Dependence')."""
    ok = [e for e in entries if e.get("status") == "ok"]
    out = []
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            sa = sum(len(set(ok[i]["A"][l]) & set(ok[j]["A"][l])) for l in range(L)) / (k * L)
            sb = sum(len(set(ok[i]["B"][l]) & set(ok[j]["B"][l])) for l in range(L)) / (k * L)
            out.append({"d": [ok[i]["d"], ok[j]["d"]], "A": sa, "B": sb})
    return out


def load_g0(g0_record: str) -> tuple[dict, dict]:
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    res = (rec.get("config") or {}).get("resolved") or {}
    return rec, res


def build(g0_record: str, bases: list[str], n_draws: int, out_dir: str, log=print, seed_base: int = SEED_BASE,
          time_limit: float = TIME_LIMIT) -> dict:
    rec, res = load_g0(g0_record)
    code = ((rec.get("decisions") or {}).get("code_stream") or {}).get("chosen")
    if not code:
        raise SystemExit(f"{g0_record}: no decisions.code_stream.chosen")
    lay, hs = res.get("layout", "auto"), int(res.get("half_seed", 0))
    os.makedirs(out_dir, exist_ok=True)
    doc = {"kind": "C1-DE exact pair draws (exploratory German C1; prereg v1.3 sec. 4.12)",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "inputs": {"g0_record": {"path": g0_record, "sha256": _sha(g0_record)}, "layout": lay, "half_seed": hs,
                      "code_stream": code, "crossover_sha256": _sha(os.path.join(HERE, "crossover.py")),
                      "eval_windows_sha256": _sha(os.path.join(HERE, "eval_windows.py")),
                      "c1de_milp_sha256": _sha(os.path.abspath(__file__))},
           "constants": {"target": TARGET, "f": F, "tol": CX.REG_TOL, "q": CX.REG_Q, "seed_base": int(seed_base),
                         "tau_0": CX.REG_TOL - MARGIN, "tau_step": TAU_STEP, "max_resolves": MAX_RESOLVES,
                         "max_weights": MAX_WEIGHTS, "time_limit_s": float(time_limit), "draws_per_base": int(n_draws)},
           "registered_seeds": int(seed_base) == SEED_BASE, "bases": {}, "draws": [], "manifests": {}}
    ok_all = True
    for base in bases:
        b = (res.get("bases") or {}).get(base) or {}
        gdir = b.get("dir") if isinstance(b, dict) else b
        if not gdir or not os.path.isdir(gdir):
            raise SystemExit(f"{base}: G0 directory {gdir} missing")
        fam = "priv" if base == "nestor" else "moe"
        t = CX.reg_load_dir(gdir, fam, layout=lay, half_seed=hs)
        if any(c not in t.sel for c in (TARGET, code, "math", "dclm")):
            raise SystemExit(f"{base}: tables lack {TARGET}/{code}/math/dclm")
        k = CX.reg_k(t)
        ents, earlier = [], []
        for d in range(1, int(n_draws) + 1):
            e = draw(t, d, code=code, earlier=earlier, seed_base=seed_base, time_limit=time_limit)
            e["base"] = base
            if e["status"] == "ok":
                earlier.append((np.array(e["A"]), np.array(e["B"])))
            ents.append(e)
            sel = e.get("selection") or {}
            att = e["generator"]["attempts"]
            row = {"base": base, "d": e["d"], "seed": e["seed"], "status": e["status"], "P_selection": sel.get("P"),
                   "P_estimation": (e.get("predicted") or {}).get("P"), "match_residuals": sel.get("match_residuals"),
                   "max_abs_residual": sel.get("max_abs_residual"), "checks": sel.get("checks"),
                   "solves": len(att), "weight_vectors": 1 + max([a.get("weights", 0) for a in att] or [0]),
                   "solve_sec": round(sum(a.get("sec", 0.0) for a in att), 4), "tau_final": (att[-1] if att else {}).get("tau"),
                   "time_limit_hit": any(a.get("time_limit_hit") for a in att),
                   "scipy": (att[-1] if att else {}).get("scipy")}
            doc["draws"].append(row)
            ok_all = ok_all and e["status"] == "ok"
            if log:
                log(f"[c1de] {base} d{d} seed {e['seed']}: {e['status']}"
                    + (f" | P_sel {sel['P']:.4f} | residuals " + ", ".join(f"{c} {v:+.5f}" for c, v in sel["match_residuals"].items())
                       if e["status"] == "ok" else ""))
        doc["bases"][base] = {"dir": gdir, "family": fam, "L": t.L, "N": t.N, "k": k, "npool": int(round(F * t.N)),
                              "n_ok": sum(e["status"] == "ok" for e in ents), "overlaps": overlaps(ents, k, t.L)}
        p = os.path.join(out_dir, f"c1de_{base}_{TARGET}.json")
        h = CX.write_manifest(p, ents, extra={"prereg": "v1.3 sec. 4.12 (C1-DE, exploratory)", "block": BLOCK,
                                              "base": base, "target": TARGET, "code_stream": code, "f": F,
                                              "tol": CX.REG_TOL, "seed_base": int(seed_base),
                                              "registered_seeds": int(seed_base) == SEED_BASE,
                                              "generator": "c1de_milp.py (exact MILP, HiGHS)"})
        doc["manifests"][base] = {"path": p, "sha256": h}
    doc["all_ok"] = bool(ok_all)
    # R6: draws with a solve that hit the time limit are not reproducible on another machine -> flagged
    doc["time_limit_hits"] = [f"{r['base']}:d{r['d']}" for r in doc["draws"] if r.get("time_limit_hit")]
    txt = json.dumps(doc, sort_keys=True, indent=1, default=float)
    with open(os.path.join(out_dir, "c1de_record.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C1-DE exact MILP pair generator (exploratory German C1, prereg v1.3 sec. 4.12)")
    ap.add_argument("--g0-record", required=True)
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--out-dir", default=os.path.join("results", "c1de"))
    ap.add_argument("--seed-base", type=int, default=SEED_BASE, help="5600 (registered); other values for dry runs only")
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    a = ap.parse_args(argv)
    if not os.path.isfile(a.g0_record):
        print(f"[c1de] {a.g0_record} missing")
        return 2
    try:
        doc = build(a.g0_record, [b for b in a.bases.split(",") if b], a.draws, a.out_dir, seed_base=a.seed_base,
                    time_limit=a.time_limit)
    except SystemExit as e:
        print(f"[c1de] {e}")
        return 2
    print(f"[c1de] {sum(r['status'] == 'ok' for r in doc['draws'])}/{len(doc['draws'])} draws ok -> "
          f"{os.path.join(a.out_dir, 'c1de_record.json')}")
    return 0 if doc["all_ok"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
