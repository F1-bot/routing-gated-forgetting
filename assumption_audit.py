"""Assumption audit at theta_0 (secondary, descriptive; about 1 GPU-h for MoE bases 1 and 2).

Pinned directional probes per MODULE GROUP x OLD CORPUS on one MoE base. The direction is the base's restricted first
Muon step on the BOOKS target, computed exactly as the registered G0 probes (g0_probe.target_gradient, bf16, 16 random
books training windows, seed 12345; g0_probe.muon_direction, Nesterov 0.95) and restricted to each group
(g0_probe.restrict). Routing is pinned to theta_0 (g0_probe.directional_probe, pin_routing=True: theta_0's selections AND
gate values are replayed), so eps -> CE is smooth and the quadratic forms are those of the fixed-routing model.
Old corpora: the first n ESTIMATION windows (g0_margin.estimation_windows) of DCLM, the G0 code stream and math.

Groups (unit keys of the registered expert family, crossover.RegTables):
  layer:<l>            all experts of stratum l                                  (L groups)
  pair:<l>+<l+1>       union of adjacent strata                                   (B5 across layers)
  c1:d<d>:<A|B>        the registered C1 books subsets of this base (crossover.reg_draw, seeds 5000 + d)
  c1:d<d>:<A|B>:h<1|2> a seeded half-split of each subset (per stratum, rng 9700 + 2 d + side)  (B5 within, B6)
  E_ALL                all experts (eps and eps/2: Richardson check of the finite-difference error)
Statistics (per group G and corpus c; windows are common to all groups, so differences are paired by window):
  Q_kl(G,c) = mean_w [KL(+eps) + KL(-eps)] / (2 eps^2);  Q_ce = mean_w [dCE(+eps) + dCE(-eps)] / (2 eps^2);
  Lin(G,c) = mean_w [dCE(+eps) - dCE(-eps)] / (2 eps);   RO_c(G) from the estimation tables at theta_0.
  B5  separability:  iota = Q(G1 u G2) - Q(G1) - Q(G2), share = iota / Q(G1 u G2), log form log Q(G1 u G2) -
      log(Q(G1) + Q(G2)), for every adjacent-layer pair and every subset = h1 u h2.
  B6  within-subset homogeneity: log nu(h1) - log nu(h2) with nu = Q / RO_c (0 under homogeneity), and the CV over
      strata of nu(layer:l).
  A0  first-order share at displacement scale eta: |eta Lin| / (|eta Lin| + eta^2 |Q_ce|) for eta = lr_muon x {1, 8, 96}
      (one step, 8 and 96 first steps), and the crossover scale eta* = |Lin| / |Q_ce|.
Memory (host fact, WDDM pages silently): every direction is kept on the HOST and g0_probe.Perturber moves one at a
time to the device; memguard.reserve before the probes, memguard.check_peak after the first window.

  python assumption_audit.py run      --g0-record g0/gate_record.json --base moe_b1 --out audit/moe_b1 [--windows 16]
                                      [--draws 2] [--checkpoint PATH] [--experiment moe_100m] [--device cuda]
  python assumption_audit.py estimate --g0-record g0/gate_record.json --base moe_b1 [--windows 16] [--draws 2]
  python assumption_audit.py analyze  --out audit/moe_b1          (torch-free: statistics from audit_windows.npz)
Outputs in --out: audit_groups.json (+ .sha256), audit_windows.npz, audit_probe.json.
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

GROUPS, WINDOWS, PROBE = "audit_groups.json", "audit_windows.npz", "audit_probe.json"
GRAD_WINDOWS, GRAD_SEED, EPS0, REGIME_WINDOWS = 16, 12345, 0.0025, 8      # g0_probe / g0_margin G0 defaults
HALF_SEED_BASE = 9700
LR_MUON = 0.0025
ETA_STEPS = (1, 8, 96)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- groups (torch-free)
def base_cell(g0_record: str, base: str, target: str = "books") -> dict:
    import dose_draws as DD                                    # same record parsing (dir fallback, code stream, cell)

    return DD.base_inputs(g0_record, base, target)


def groups_for(t: CX.RegTables, *, cell: dict, code: str, draws: int, target: str = "books") -> list[dict]:
    """The audit's module groups as [L, N] membership masks with unit keys."""
    L, N = t.L, t.N
    out = []

    def add(name, mask, kind, **kw):
        out.append({"name": name, "kind": kind, "mask": mask.astype(bool), **kw})

    for l in range(L):
        m = np.zeros((L, N), bool); m[l] = True
        add(f"layer:{l}", m, "layer", layer=l)
    for l in range(L - 1):
        m = np.zeros((L, N), bool); m[l] = m[l + 1] = True
        add(f"pair:{l}+{l + 1}", m, "pair", parts=[f"layer:{l}", f"layer:{l + 1}"])
    if cell.get("feasible") and draws > 0:
        for d in range(1, draws + 1):
            e = CX.reg_draw(t, "c1", d, f=float(cell["f"]), target=target, code=code, tol=float(cell["tol"]))
            if e.get("status") != "ok":
                continue
            for si, side in enumerate(("A", "B")):
                S = np.asarray(e[side])                        # [L, k] module indices
                m = np.zeros((L, N), bool)
                m[np.arange(L)[:, None], S] = True
                nm = f"c1:d{d}:{side}"
                rng = np.random.default_rng(HALF_SEED_BASE + 2 * d + si)
                h1 = np.zeros((L, N), bool)
                for l in range(L):
                    perm = rng.permutation(S.shape[1])
                    h1[l, S[l, perm[: S.shape[1] // 2]]] = True
                h2 = m & ~h1
                if S.shape[1] >= 2:                           # halves need >= 2 experts per stratum
                    add(nm, m, "subset", d=d, side=side, seed=e["seed"], P=e["predicted"]["P"], parts=[nm + ":h1", nm + ":h2"])
                    add(nm + ":h1", h1, "half", of=nm)
                    add(nm + ":h2", h2, "half", of=nm)
                else:
                    add(nm, m, "subset", d=d, side=side, seed=e["seed"], P=e["predicted"]["P"])
    add("E_ALL", np.ones((L, N), bool), "all")
    for g in out:
        g["keys"] = [t.keys[l][int(m)] for l in range(L) for m in np.flatnonzero(g["mask"][l])]   # mask -> unit keys
        g["n_units"] = int(g["mask"].sum())
    return out


def ro(t: CX.RegTables, mask: np.ndarray, corpus: str) -> float:
    s2 = t.est[corpus]
    return float(s2[mask].sum() / s2.sum())


def estimate(n_groups: int, windows: int, *, corpora: int = 3, tok_per_s: float = 27000.0, block: int = 4096) -> dict:
    fwd = corpora * windows * (1 + 2 * n_groups + 2)              # base record + (+/-) per group + E_ALL at eps/2
    eps_fwd = 2 * 4 * REGIME_WINDOWS
    grad_fwd = 3 * GRAD_WINDOWS
    tok = (fwd + eps_fwd + grad_fwd) * block
    return {"groups": n_groups, "windows_per_corpus": windows, "forwards": fwd + eps_fwd + grad_fwd, "tokens": tok,
            "gpu_hours": round(tok / tok_per_s / 3600 + 0.05, 3), "assumption": f"{tok_per_s:.0f} tok/s fp32 pinned forwards"
            " + 3 min model load / gradient / transfers"}


# ----------------------------------------------------------------------------- statistics (torch-free)
def _ms(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"value": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None, "n": int(len(x))}


def analyze(out_dir: str) -> dict:
    gdoc = json.load(open(os.path.join(out_dir, GROUPS), encoding="utf-8"))
    z = np.load(os.path.join(out_dir, WINDOWS))
    eps = float(z["eps"])
    corpus = z["corpus"].astype(str)
    names = [g["name"] for g in gdoc["groups"]]
    ro_tab = gdoc["ro"]

    def arr(name, sg, fld, e=eps):
        return np.asarray(z[f"{name}|{e:.6g}|{sg}|{fld}"], dtype=np.float64)

    per: dict = {}
    qw: dict = {}
    for nm in names:
        per[nm] = {}
        for c in sorted(set(corpus)):
            msk = corpus == c
            kl = (arr(nm, "+1", "kl") + arr(nm, "-1", "kl"))[msk] / (2 * eps * eps)
            qce = (arr(nm, "+1", "dce") + arr(nm, "-1", "dce"))[msk] / (2 * eps * eps)
            lin = (arr(nm, "+1", "dce") - arr(nm, "-1", "dce"))[msk] / (2 * eps)
            qw[(nm, c)] = kl
            row = {"Q_kl": _ms(kl), "Q_ce": _ms(qce), "Lin": _ms(lin), "RO": ro_tab[nm][c]}
            qv, lv = abs(row["Q_ce"]["value"]), abs(row["Lin"]["value"])
            row["A0"] = {f"eta_{s}": (lv * LR_MUON * s) / (lv * LR_MUON * s + qv * (LR_MUON * s) ** 2)
                         if (lv + qv) > 0 else None for s in ETA_STEPS}
            row["A0"]["eta_star"] = lv / qv if qv > 0 else None
            row["nu"] = row["Q_kl"]["value"] / row["RO"] if row["RO"] > 0 else None
            per[nm][c] = row
    b5, b6 = [], []
    for g in gdoc["groups"]:
        if g.get("parts"):
            p1, p2 = g["parts"]
            for c in sorted(set(corpus)):
                iw = qw[(g["name"], c)] - qw[(p1, c)] - qw[(p2, c)]
                qu = per[g["name"]][c]["Q_kl"]["value"]
                q12 = per[p1][c]["Q_kl"]["value"] + per[p2][c]["Q_kl"]["value"]
                b5.append({"group": g["name"], "kind": g["kind"], "corpus": c, "iota": _ms(iw),
                           "share": (float(iw.mean()) / qu) if qu > 0 else None,
                           "log_form": (math.log(qu) - math.log(q12)) if qu > 0 and q12 > 0 else None})
                if g["kind"] == "subset":
                    n1, n2 = per[p1][c]["nu"], per[p2][c]["nu"]
                    b6.append({"subset": g["name"], "corpus": c, "log_nu_h1_minus_h2":
                               (math.log(n1) - math.log(n2)) if n1 and n2 and n1 > 0 and n2 > 0 else None})
    cv = {}
    for c in sorted(set(corpus)):
        nus = [per[f"layer:{l}"][c]["nu"] for l in range(gdoc["L"]) if per[f"layer:{l}"][c]["nu"]]
        cv[c] = float(np.std(nus, ddof=1) / np.mean(nus)) if len(nus) > 1 else None
    rich = {}
    if "E_ALL" in names and f"E_ALL|{eps / 2:.6g}|+1|kl" in z.files:
        for c in sorted(set(corpus)):
            msk = corpus == c
            q1 = ((arr("E_ALL", "+1", "kl") + arr("E_ALL", "-1", "kl"))[msk] / (2 * eps * eps)).mean()
            q2 = ((arr("E_ALL", "+1", "kl", eps / 2) + arr("E_ALL", "-1", "kl", eps / 2))[msk] / (2 * (eps / 2) ** 2)).mean()
            rich[c] = {"Q_eps": float(q1), "Q_eps_half": float(q2), "rel_diff": float(abs(q1 - q2) / abs(q2)) if q2 else None}
    rep = {"purpose": "assumption audit at theta_0 (B5 separability, B6 within-subset homogeneity, A0)",
           "base": gdoc["base"], "eps": eps, "routing_mode": "pinned", "corpora": sorted(set(corpus)),
           "groups": {g["name"]: {k: g[k] for k in ("kind", "n_units") if k in g} for g in gdoc["groups"]},
           "per_group": per, "B5": b5, "B6": b6, "nu_cv_over_layers": cv, "richardson": rich,
           "windows_per_corpus": {c: int((corpus == c).sum()) for c in sorted(set(corpus))},
           "groups_sha256": gdoc.get("sha256"), "label": "secondary, descriptive (no test)"}
    return rep


# ----------------------------------------------------------------------------- GPU run
def run(g0_record: str, base: str, out_dir: str, *, windows: int = 16, draws: int = 2, device: str | None = None,
        checkpoint: str | None = None, experiment: str | None = None, grad_path: str | None = None, model=None,
        register: str | None = None, log=print) -> dict:
    import torch

    import eval_windows as EW
    import g0_margin as GM
    import g0_probe as G0
    import memguard as MG

    t0 = time.time()
    bi = base_cell(g0_record, base)
    code = bi["code"]
    t = CX.reg_load_dir(bi["dir"], bi["family"], layout=bi["layout"], half_seed=bi["half_seed"])
    groups = groups_for(t, cell=bi["cell"], code=code, draws=draws)
    corpora = ["dclm", code, "math"]
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    meta = EW.load_result(os.path.join(bi["dir"], f"routing__{code}.npz"))["meta"]
    experiment = experiment or meta.get("experiment")
    checkpoint = checkpoint or meta.get("checkpoint") or ((rec.get("checkpoints") or {}).get(base) or [None])[0]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    gdoc = {"base": base, "L": t.L, "N": t.N, "strata": t.strata, "code_stream": code, "g0_dir": bi["dir"],
            "cell": {k: bi["cell"].get(k) for k in ("f", "tol", "median", "rung")}, "draws": draws,
            "half_seed_base": HALF_SEED_BASE,
            "groups": [{k: v for k, v in g.items() if k != "mask"} for g in groups],
            "ro": {g["name"]: {("code" if c == code else c): ro(t, g["mask"], c) for c in corpora} for g in groups}}
    gtxt = json.dumps(gdoc, sort_keys=True, indent=1, default=float)
    gdoc["sha256"] = hashlib.sha256(gtxt.encode()).hexdigest()
    with open(os.path.join(out_dir, GROUPS), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(gdoc, sort_keys=True, indent=1, default=float))
    if model is None:
        EW.register_extra(register)
        model, _ = EW.load_model(experiment, checkpoint, device)
    half_seed = int(bi["half_seed"])
    win = []
    for c in corpora:
        ws = GM.estimation_windows(bi["dir"], c, windows, half_seed=half_seed)
        win += [(ws, i) for i in range(len(ws.index))]
    corpus_of = np.array([("code" if ws.name == code else ws.name) for ws, _ in win]).astype(str)
    rows, params = G0.unit_rows(model), G0.expert_params(model)
    src = grad_path or EW.load_result(os.path.join(bi["dir"], "routing__books.npz"))["meta"]["set"]["path"]
    gws = EW.WindowSet("books_grad", src, int(meta["set"]["block"]), GRAD_WINDOWS, "random", GRAD_SEED)
    grads = G0.target_gradient(model, gws, params, GRAD_WINDOWS, device=device, bf16=True)
    u_e = G0.muon_direction(params, grads, momentum=0.95, nesterov=True)
    u_e = MG.host_copy(u_e)                                   # every direction on the host (patch; WDDM)
    del grads
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    ec = G0.choose_eps(model, win[:REGIME_WINDOWS], u_e, device=device, eps0=EPS0, log=None, pin_routing=True)
    eps = float(ec["eps"])
    plan = []
    for g in groups:
        u_g = MG.host_copy(G0.restrict(u_e, rows, g["keys"]))
        plan += [(g["name"], u_g, eps, +1), (g["name"], u_g, eps, -1)]
        if g["name"] == "E_ALL":
            plan += [(g["name"], u_g, eps / 2, +1), (g["name"], u_g, eps / 2, -1)]
    biggest = max(MG.tensor_bytes(d.values()) for _, d, _, _ in plan)
    MG.reserve(device, 2 * biggest, "assumption_audit: one direction + theta_0 copies of its tensors")
    res = G0.directional_probe(model, win[:1], plan, device=device, log=None, pin_routing=True, count_flips=False)
    MG.check_peak(device, "assumption_audit after the first window")
    MG.check(device, "assumption_audit after the first window")
    if len(win) > 1:
        rest = G0.directional_probe(model, win[1:], plan, device=device, log=log, pin_routing=True, count_flips=False)
        for k in res:
            for fld in ("kl", "dce", "flip"):
                res[k][fld] = np.concatenate([res[k][fld], rest[k][fld]])
    arrs = {"corpus": corpus_of, "window_index": np.array([int(ws.index[i]) for ws, i in win], dtype=np.int64),
            "window_hash": np.array([int(ws.hashes[i]) for ws, i in win], dtype=np.uint64), "eps": np.array(eps)}
    for k, v in res.items():
        for fld in ("kl", "dce"):
            arrs[f"{k}|{fld}"] = np.asarray(v[fld], dtype=np.float64)
    tmp = os.path.join(out_dir, WINDOWS + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, os.path.join(out_dir, WINDOWS))
    rep = analyze(out_dir)
    rep.update(experiment=experiment, checkpoint=checkpoint, device=str(device), eps_check=ec,
               grad={"path": src, "windows": GRAD_WINDOWS, "seed": GRAD_SEED, "bf16": True},
               inputs={"g0_record": {"path": os.path.abspath(g0_record), "sha256": _sha(g0_record)},
                       "routing_files": {s[r]["file"]: _sha(s[r]["file"]) for s in t.sources.values()
                                         for r in ("selection", "estimation")}},
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, PROBE), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rep, indent=1, default=float))
    if log:
        log(f"[audit] {base}: {len(groups)} groups x {len(corpora)} corpora x {windows} windows, eps {eps:g}, "
            f"{rep['seconds']:.0f}s -> {out_dir}")
    return rep


# ============================================================================= prereg v1.3 sec. 4.3: realised directions
REAL_GROUPS, REAL_WINDOWS, REAL_PROBE = "audit_realized_groups.json", "audit_realized_windows.npz", "audit_realized.json"
N_BOOT, BOOT_SEED = 2000, 20261113
B5_CUTS = ((0.1, "approximately separable"), (0.3, "moderately non-separable"), (float("inf"), "non-separable"))
CV_HOMOG, A0_R, A0_SHARE = 0.25, 0.2, 0.25


def realized_groups(t: CX.RegTables, specs: dict) -> list[dict]:
    """sec. 4.3 groups: the routed experts of subset S (the draw-1 C1 books subsets A_1, B_1) within each stratum."""
    import forecasts as FC                                   # torch-free spec -> routing-table keys

    keys_all = [k for row in t.keys for k in row]
    out = []
    for side in sorted(specs):
        ks = set(FC.spec_keys(specs[side], keys_all))
        for l in range(t.L):
            m = np.zeros((t.L, t.N), bool)
            for j, k in enumerate(t.keys[l]):
                m[l, j] = k in ks
            if m.any():
                out.append({"name": f"{side}:l{l}", "side": side, "layer": l, "stratum": t.strata[l], "mask": m,
                            "keys": [t.keys[l][j] for j in np.flatnonzero(m[l])], "n_units": int(m.sum())})
    return out


def group_contrasts(t: CX.RegTables, groups: list[dict], code: str) -> dict:
    """The group's code-math usage contrast as its share of P: RO_code(g)/RO_code(S) - RO_math(g)/RO_math(S)
    (estimation tables; sums to 0 over the groups of a subset)."""
    out = {}
    for side in sorted({g["side"] for g in groups}):
        gs = [g for g in groups if g["side"] == side]
        S = np.logical_or.reduce([g["mask"] for g in gs])
        rc, rm = ro(t, S, code), ro(t, S, "math")
        for g in gs:
            out[g["name"]] = ro(t, g["mask"], code) / rc - ro(t, g["mask"], "math") / rm
    return out


def _registered_eps(g0_dir: str) -> tuple[float, str]:
    p = os.path.join(g0_dir, "margin__probe.json")
    if os.path.exists(p):
        e = json.load(open(p, encoding="utf-8")).get("eps")
        e = json.loads(e) if isinstance(e, str) else e
        if isinstance(e, dict) and e.get("books") is not None:
            return float(e["books"]), f"G0-7 registered step ({p}: books)"
    return EPS0, f"G0-7 step not found in {g0_dir}; g0_probe eps0 = {EPS0}"


def run_realized(g0_record: str, base: str, out_dir: str, *, runs: list[str], state: str | None = None,
                 queue: str | None = None, specs: dict | None = None, final_states: dict | None = None,
                 windows: int = 24, device: str | None = None, checkpoint: str | None = None,
                 experiment: str | None = None, model=None, register: str | None = None, log=print) -> dict:
    """Pinned probes along the realised displacement h_g = delta_g of the draw-1 C1 books runs (sec. 4.3)."""
    import torch

    import eval_windows as EW
    import g0_margin as GM
    import g0_probe as G0

    t0 = time.time()
    bi = base_cell(g0_record, base)
    code = bi["code"]
    t = CX.reg_load_dir(bi["dir"], bi["family"], layout=bi["layout"], half_seed=bi["half_seed"])
    st = json.load(open(state, encoding="utf-8")) if state and os.path.exists(state) else {}
    side_of = {}
    for rid in runs:
        sd = rid.rsplit("_", 1)[-1]
        if sd not in ("A", "B"):
            raise SystemExit(f"run id {rid} does not end in _A / _B")
        side_of[rid] = sd
    if specs is None:
        if not queue or not os.path.exists(queue):
            raise SystemExit("--queue (the confirm queue with the runs' subset specs) is required")
        qj = {j["id"]: j for j in json.load(open(queue, encoding="utf-8-sig"))["jobs"]}
        specs = {side_of[r]: qj[r]["spec"] for r in runs}
    groups = realized_groups(t, specs)
    contrast = group_contrasts(t, groups, code)
    eps, eps_src = _registered_eps(bi["dir"])
    corpora = ["dclm", code, "math"]
    meta = EW.load_result(os.path.join(bi["dir"], f"routing__{code}.npz"))["meta"]
    experiment = experiment or meta.get("experiment")
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    checkpoint = checkpoint or meta.get("checkpoint") or ((rec.get("checkpoints") or {}).get(base) or [None])[0]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    if model is None:
        EW.register_extra(register)
        model, _ = EW.load_model(experiment, checkpoint, device)
    rows = G0.unit_rows(model)
    theta0 = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
    finals = {}
    for rid, sd in side_of.items():
        if final_states and sd in final_states:
            finals[sd] = final_states[sd]
            continue
        fck = (st.get(rid) or {}).get("final")
        if not fck or not os.path.exists(fck):
            raise SystemExit(f"final checkpoint of {rid} not found in {state} (the audit runs after the draw-1 C1 runs)")
        m1, _ = EW.load_model(experiment, fck, "cpu")
        finals[sd] = {n: p.detach().to("cpu", copy=True) for n, p in m1.named_parameters()}
        del m1
    plan, gmeta = [], []
    for g in groups:
        fin, d = finals[g["side"]], {}
        for k in g["keys"]:
            for name, r in rows.get(k, []):
                if name not in d:
                    d[name] = torch.zeros_like(theta0[name])
                if r is None:
                    d[name] = (fin[name] - theta0[name]).to(d[name].dtype)
                else:
                    d[name][r] = (fin[name][r] - theta0[name][r]).to(d[name].dtype)
        rho = math.sqrt(G0.sq_norm(d))
        plan += [(g["name"], d, eps, +1), (g["name"], d, eps, -1)]
        gmeta.append({k: v for k, v in g.items() if k != "mask"} | {"rho": rho, "contrast": contrast[g["name"]],
                                                                     "params": sorted(d)})
    del theta0, finals
    win = []
    for c in corpora:
        ws = GM.estimation_windows(bi["dir"], c, windows, half_seed=int(bi["half_seed"]))
        win += [(ws, i) for i in range(len(ws.index))]
    corpus_of = np.array([("code" if ws.name == code else ws.name) for ws, _ in win]).astype(str)
    res = G0.directional_probe(model, win, plan, device=device, log=log, pin_routing=True, count_flips=False)
    arrs = {"corpus": corpus_of, "window_index": np.array([int(ws.index[i]) for ws, i in win], dtype=np.int64),
            "window_hash": np.array([int(ws.hashes[i]) for ws, i in win], dtype=np.uint64), "eps": np.array(eps)}
    for k, v in res.items():
        arrs[f"{k}|dce"] = np.asarray(v["dce"], dtype=np.float64)
    tmp = os.path.join(out_dir, REAL_WINDOWS + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, os.path.join(out_dir, REAL_WINDOWS))
    gdoc = {"base": base, "runs": runs, "specs": specs, "eps": eps, "eps_source": eps_src, "code_stream": code,
            "corpora": ["dclm", "code", "math"], "windows_per_corpus": windows, "groups": gmeta,
            "contrast_definition": "RO_code(g)/RO_code(S) - RO_math(g)/RO_math(S) on the estimation tables"}
    with open(os.path.join(out_dir, REAL_GROUPS), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(gdoc, sort_keys=True, indent=1, default=float))
    rep = analyze_realized(out_dir)
    rep.update(experiment=experiment, checkpoint=checkpoint, device=str(device), seconds=round(time.time() - t0, 1),
               inputs={"g0_record": {"path": os.path.abspath(g0_record), "sha256": _sha(g0_record)},
                       "state": {"path": state, "sha256": _sha(state) if state and os.path.exists(state) else None},
                       "queue": {"path": queue, "sha256": _sha(queue) if queue and os.path.exists(queue) else None}})
    with open(os.path.join(out_dir, REAL_PROBE), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rep, indent=1, default=float))
    if log:
        log(f"[audit] {base} realised: {len(groups)} groups x 3 corpora x {windows} windows, eps {eps:g} -> {out_dir}")
    return rep


def _twoway(Y: np.ndarray, M: np.ndarray, iters: int = 1000) -> tuple:
    """Additive fit log Q = mu + a_g + b_c on the valid cells M (backfitting; equal to the row / column means for a
    balanced M), effects centred with cell weights: (mu, a, b, residuals on M)."""
    G, C = Y.shape
    a, b = np.zeros(G), np.zeros(C)
    mu = float(Y[M].mean())
    for _ in range(iters):
        a_new = np.array([(Y[g, M[g]] - mu - b[M[g]]).mean() if M[g].any() else 0.0 for g in range(G)])
        b_new = np.array([(Y[M[:, c], c] - mu - a_new[M[:, c]]).mean() if M[:, c].any() else 0.0 for c in range(C)])
        done = np.allclose(a_new, a, rtol=0, atol=1e-14) and np.allclose(b_new, b, rtol=0, atol=1e-14)
        a, b = a_new, b_new
        if done:
            break
    w = M.sum()
    abar, bbar = float((a[:, None] * M).sum() / w), float((b[None, :] * M).sum() / w)
    a, b, mu = a - abar, b - bbar, mu + abar + bbar
    E = np.where(M, Y - mu - a[:, None] - b[None, :], 0.0)
    return mu, a, b, E


def _b5_share(Qf: np.ndarray, Q1: np.ndarray, Q2: np.ndarray) -> dict:
    M = Qf > 0
    Y = np.where(M, np.log(np.where(M, Qf, 1.0)), 0.0)
    mu, a, b, E = _twoway(Y, M)
    G, C = Qf.shape
    ssg = float(sum((a[g] ** 2) * M[g].sum() for g in range(G)))
    ssc = float(sum((b[c] ** 2) * M[:, c].sum() for c in range(C)))
    sse = float((E ** 2).sum())
    hv = M & (Q1 > 0) & (Q2 > 0)
    noise = float(((np.log(np.where(hv, Q1, 1.0)) - np.log(np.where(hv, Q2, 1.0)))[hv] ** 2 / 4.0).mean()) if hv.any() else 0.0
    sse_adj = max(sse - (G - 1) * (C - 1) * noise, 0.0)
    den = ssg + ssc + sse_adj
    return {"share": sse_adj / den if den > 0 else None, "SS_g": ssg, "SS_c": ssc, "SS_e": sse, "SS_e_adj": sse_adj,
            "noise_var_logQ": noise, "a": a, "b": b, "mu": mu, "excluded_cells": int((~M).sum())}


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    def rk(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v), dtype=np.float64)
        return r
    rx, ry = rk(np.asarray(x)), rk(np.asarray(y))
    dx, dy = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    return float((dx * dy).sum() / den) if den > 0 else None


def _label_b5(s):
    return None if s is None else next(lab for cut, lab in B5_CUTS if s < cut)


def analyze_realized(out_dir: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """B5 / B6 / A0 of prereg v1.3 sec. 4.3 from the per-window probe values (torch-free)."""
    gdoc = json.load(open(os.path.join(out_dir, REAL_GROUPS), encoding="utf-8"))
    z = np.load(os.path.join(out_dir, REAL_WINDOWS))
    eps = float(z["eps"])
    cor = z["corpus"].astype(str)
    corpora = [c for c in ("dclm", "code", "math") if (cor == c).any()]
    names = [g["name"] for g in gdoc["groups"]]
    key = lambda n, s: f"{n}|{eps:.6g}|{s:+d}|dce"
    W = {c: np.flatnonzero(cor == c) for c in corpora}
    qw = {n: (z[key(n, 1)] + z[key(n, -1)]) / (2 * eps * eps) for n in names}
    gw = {n: (z[key(n, 1)] - z[key(n, -1)]) / (2 * eps) for n in names}

    def cells(sel: dict) -> tuple:
        Qf = np.array([[qw[n][sel[c]].mean() for c in corpora] for n in names])
        h = {c: (sel[c][: len(sel[c]) // 2], sel[c][len(sel[c]) // 2:]) for c in corpora}
        Q1 = np.array([[qw[n][h[c][0]].mean() for c in corpora] for n in names])
        Q2 = np.array([[qw[n][h[c][1]].mean() for c in corpora] for n in names])
        return Qf, Q1, Q2

    Qf, Q1, Q2 = cells(W)
    Gm = np.array([[gw[n][W[c]].mean() for c in corpora] for n in names])
    b5 = _b5_share(Qf, Q1, Q2)
    sides = sorted({g["side"] for g in gdoc["groups"]})
    side_idx = {s: [i for i, g in enumerate(gdoc["groups"]) if g["side"] == s] for s in sides}
    cvec = np.array([g["contrast"] for g in gdoc["groups"]])

    def b6_of(a: np.ndarray) -> dict:
        out = {}
        for s, ix in side_idx.items():
            ea = np.exp(a[ix])
            out[s] = {"cv": float(ea.std(ddof=1) / ea.mean()) if len(ix) > 1 and ea.mean() > 0 else None,
                      "spearman": _spearman(a[ix], cvec[ix]),
                      "jensen_gap": float(math.log(ea.mean()) - a[ix].mean()) if ea.mean() > 0 else None}
        return out

    b6 = b6_of(b5["a"])
    rng = np.random.default_rng(seed)
    bs_share, bs_b6 = [], {s: {"cv": [], "spearman": []} for s in sides}
    for _ in range(n_boot):
        sel = {c: W[c][rng.integers(0, len(W[c]), len(W[c]))] for c in corpora}
        q = _b5_share(*cells(sel))
        if q["share"] is not None:
            bs_share.append(q["share"])
        for s, v in b6_of(q["a"]).items():
            for k in ("cv", "spearman"):
                if v[k] is not None:
                    bs_b6[s][k].append(v[k])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 1 else None
    b5_rep = {"interaction_share": b5["share"], "ci95": ci(bs_share), "label": _label_b5(b5["share"]),
              "SS": {"groups": b5["SS_g"], "corpora": b5["SS_c"], "interaction": b5["SS_e"],
                     "interaction_minus_noise": b5["SS_e_adj"]}, "noise_var_logQ": b5["noise_var_logQ"],
              "cells_Q_nonpositive": b5["excluded_cells"], "cells": int(Qf.size), "bootstrap": len(bs_share)}
    b6_rep = {}
    for s, v in b6.items():
        rci = ci(bs_b6[s]["spearman"])
        if v["cv"] is None:
            lab = None
        elif v["cv"] < CV_HOMOG:
            lab = "homogeneous"
        elif rci is not None and rci[0] <= 0.0 <= rci[1]:
            lab = "heterogeneous, uncorrelated"
        else:
            lab = "heterogeneous, correlated"
        b6_rep[s] = dict(v, cv_ci95=ci(bs_b6[s]["cv"]), spearman_ci95=rci, label=lab)
    a0 = {}
    for j, c in enumerate(corpora):
        ok = Qf[:, j] > 0
        r = np.abs(Gm[ok, j]) / Qf[ok, j]
        share = float((r >= A0_R).mean()) if r.size else None
        a0[c] = {"median_r": float(np.median(r)) if r.size else None, "share_r_ge_0.2": share, "n": int(r.size),
                 "label": None if share is None else ("A0 approximately holds" if share <= A0_SHARE else "A0 violated")}
    return {"prereg": "v1.3 sec. 4.3 (secondary; labels reported with C1; no decision depends on them)",
            "base": gdoc["base"], "eps": eps, "eps_source": gdoc.get("eps_source"), "corpora": corpora,
            "groups": names, "Q": Qf.tolist(), "gamma": Gm.tolist(), "rho": [g["rho"] for g in gdoc["groups"]],
            "B5": b5_rep, "B6": b6_rep, "A0": a0, "bootstrap": {"n": n_boot, "seed": seed, "unit": "windows within corpus"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="assumption audit at theta_0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "estimate"):
        s = sub.add_parser(name)
        s.add_argument("--g0-record", required=True)
        s.add_argument("--base", required=True)
        s.add_argument("--windows", type=int, default=16)
        s.add_argument("--draws", type=int, default=2)
        if name == "run":
            s.add_argument("--out", required=True)
            s.add_argument("--device", default=None)
            s.add_argument("--checkpoint", default=None)
            s.add_argument("--experiment", default=None)
            s.add_argument("--grad-path", default=None)
            s.add_argument("--register", default=None, help=argparse.SUPPRESS)
            s.add_argument("--directions", choices=("first-step", "realized"), default="first-step",
                           help="realized: prereg v1.3 sec. 4.3 (displacements of the draw-1 C1 books runs, P-10)")
            s.add_argument("--runs", default="", help="realized: the two C1 runs <..._d1_A>,<..._d1_B>")
            s.add_argument("--state", default=None, help="realized: queue state with the runs' final checkpoints")
            s.add_argument("--queue", default=None, help="realized: queue file with the runs' subset specs")
        else:
            s.add_argument("--tok-per-s", type=float, default=27000.0)
    s = sub.add_parser("analyze")
    s.add_argument("--out", required=True)
    s.add_argument("--mode", choices=("first-step", "realized"), default="first-step")
    a = ap.parse_args(argv)
    if a.cmd == "analyze" and a.mode == "realized":
        rep = analyze_realized(a.out)
        print(json.dumps({"B5": rep["B5"], "B6": rep["B6"], "A0": rep["A0"]}, indent=1, default=float))
        return 0
    if a.cmd == "analyze":
        rep = analyze(a.out)
        print(json.dumps({"B5": rep["B5"][:4], "B6": rep["B6"][:4], "nu_cv_over_layers": rep["nu_cv_over_layers"]}, indent=1))
        return 0
    if a.cmd == "estimate":
        bi = base_cell(a.g0_record, a.base)
        t = CX.reg_load_dir(bi["dir"], bi["family"], layout=bi["layout"], half_seed=bi["half_seed"])
        n = len(groups_for(t, cell=bi["cell"], code=bi["code"], draws=a.draws))
        print(json.dumps(estimate(n, a.windows, tok_per_s=a.tok_per_s), indent=1))
        return 0
    if a.directions == "realized":
        run_realized(a.g0_record, a.base, a.out, runs=[r for r in a.runs.split(",") if r], state=a.state, queue=a.queue,
                     windows=a.windows, device=a.device, checkpoint=a.checkpoint, experiment=a.experiment,
                     register=a.register)
        return 0
    run(a.g0_record, a.base, a.out, windows=a.windows, draws=a.draws, device=a.device, checkpoint=a.checkpoint,
        experiment=a.experiment, grad_path=a.grad_path, register=a.register)
    return 0


if __name__ == "__main__":
    sys.exit(main())
