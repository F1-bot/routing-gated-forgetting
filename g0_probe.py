"""G0 pre-adaptation gate: run on the GPU on the trained checkpoints.

  python g0_probe.py estimate --config g0_config_moe.json          # GPU-time estimate, no model load
  python g0_probe.py streams  --config g0_config_moe.json          # preflight: windows per stream (exit 2 if short)
  python g0_probe.py run      --config g0_config_moe.json          # routing -> contrasts -> probes
  python g0_probe.py run      --config g0_config_nestor.json
  python g0_probe.py xplatform --a local/ckpt__dclm.npz --b h100/ckpt__dclm.npz --out g0/xplatform.json
  python g0_probe.py decide --summary g0/moe/g0_summary.json --summary g0/nestor/g0_summary.json \
                            [--xplatform g0/xplatform.json] --out g0/g0_decision.json   # phase-3 heuristic only
  REGISTERED gate values (prereg v1.0 sec. 6): python g0_gate.py --config g0/g0_gate_config.json

Steps of `run` (resumable: finished pieces are skipped on restart):
 1. routing statistics + per-window CE of the base model on every stream (DCLM split halves, code, math,
    German, books, French), eval_windows format with per-window effective-gate summaries;
 2. feasible contrasts (crossover.py): C1 crossover pairs per target (A = low code / high math, matched on
    target and DCLM usage), C2 target-usage tilts, DCLM LOW/HIGH tilts; best feasible predicted |I|;
 3. probes at theta_0 along the optimizer-normalized first step on each target (prereg sec. 5.3):
    u = per-matrix Muon transform (muon_scale x Newton-Schulz-5) of the mean target gradient of the expert
    matrices (routers frozen, training-mode forward with the authors' loss);
      Q_E    = (KL+ + KL-) / (2 eps^2) along u_E on old-data positions (eps halved from 0.0025 until
               KL(2eps)/KL(eps) is in [3.5, 4.5]),
      Lin_S  = (dCE+ - dCE-) / (2 eps),  Quad_S = (dCE+ + dCE-) / (2 eps^2) along u_S (subset rows),
      Sens_S = the KL quadratic form along a norm-matched random direction r_S (same singular values as
               u_S per matrix, random singular vectors) -- the random-direction sensitivity probe,
    and the first-step statistics the SGD calibration needs (sum ||g||^2, sum ||u||^2 per subset);
    per-window probe values go to probe__<target>__windows.npz (G0-6 split-half reliability, g0_gate.py).
Streams may carry "span": [f0, f1] (+ "span_align": "doc") to take windows only from a fraction of the file:
the registered layout (prereg sec. 4.3) is <c> = selection half [0, 0.5], <c>_eval = evaluation half [0.5, 1];
g0_gate.py reads routing__<c>.npz as selection and routing__<c>_eval.npz as estimation statistics.
 4. g0_summary.json: best |I| per target, draws, probes, the G0 decision inputs.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERT_KINDS = ("moe", "pub", "priv")


# ============================================================================= unit -> parameter rows
def unit_rows(model) -> dict[str, list[tuple[str, int | None]]]:
    import subset_adapt as SA

    out: dict[str, list] = {}
    for e in SA.build_catalog(model):
        if e.sliced:
            for r, u in enumerate(e.units):
                out.setdefault(u.key, []).append((e.name, r))
        else:
            out.setdefault(e.units[0].key, []).append((e.name, None))
    return out


def expert_params(model, kinds=EXPERT_KINDS) -> dict[str, Any]:
    import subset_adapt as SA

    return {e.name: e.param for e in SA.build_catalog(model)
            if {u.kind for u in e.units} <= set(kinds) and e.optimizer == "muon"}


# ============================================================================= first-step direction
def target_gradient(model, wset, params: dict, n_windows: int, *, device: str, bf16: bool = True) -> dict:
    """Mean gradient of the training loss over n target windows w.r.t. `params` (routers frozen)."""

    import torch
    from nestor.configs.losses import CausalLMLossConfig
    from nestor.train.frozen_routers import hold_routing_in_eval_mode
    from nestor.train.losses import Loss

    flags = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    loss_fn = Loss.from_config(CausalLMLossConfig())
    model.train()
    hold_routing_in_eval_mode(model)
    n = min(n_windows, len(wset.index))
    try:
        for i in range(n):
            x = torch.as_tensor(wset.tokens(i)).view(1, -1).to(device)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
                out = model(x)
                loss = loss_fn(out.pred, x, out.aux).total / n
            loss.backward()
        grads = {k: p.grad.detach().float().clone() for k, p in params.items() if p.grad is not None}
    finally:
        for p in model.parameters():
            p.grad = None
            p.requires_grad_(flags[id(p)])
        model.eval()
    return grads


def muon_direction(params: dict, grads: dict, *, momentum: float = 0.95, nesterov: bool = True,
                   ns_steps: int = 5) -> dict:
    """Unit-lr direction of the authors' FIRST Muon step from zero momentum, per matrix / expert slice.

    The optimizer steps theta -= lr_group * NS5(d) with d = g + mu * buf, buf = g on the first step
    (Nesterov), and lr_group = lr_muon * muon_scale (reference muP; checked in test_phase3). So
    u = muon_scale * NS5(d) and theta_1 - theta_0 = -lr_muon * u (weight decay aside). NS5 runs in
    bf16, hence it is NOT scale-invariant in practice: orthogonalizing g instead of (1 + mu) g changes
    the bf16 rounding and gave 5-10% relative deviations on tiny low-rank gradients; this function
    feeds NS5 exactly the optimizer's d (bit-identical foreach arithmetic).
    """

    import torch
    from nestor.scaling.blocks import get_scaling
    from nestor.scaling.muon import zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_batched

    names = list(grads)
    gs = [grads[n] for n in names]
    ds = torch._foreach_add(gs, gs, alpha=momentum) if nesterov else gs
    out = {}
    for name, d in zip(names, ds):
        s = get_scaling(params[name])
        scale = float(s.muon_scale) if s is not None else 1.0
        o = zeropower_via_newtonschulz5_batched(d, ns_steps) if d.ndim == 3 else zeropower_via_newtonschulz5(d, ns_steps)
        out[name] = (o.float() * scale)
    return out


def restrict(direction: dict, rows: dict[str, list], keys: list[str]) -> dict:
    """The direction restricted to the parameter rows of the given units (zero elsewhere)."""

    import torch

    keep: dict[str, Any] = {}
    for k in keys:
        for name, r in rows.get(k, []):
            if name not in direction:
                continue
            if r is None:
                keep[name] = None
            else:
                keep.setdefault(name, set())
                if keep[name] is not None:
                    keep[name].add(r)
    out = {}
    for name, rs in keep.items():
        d = direction[name]
        if rs is None:
            out[name] = d.clone()
        else:
            m = torch.zeros(d.shape[0], dtype=torch.bool, device=d.device)
            m[sorted(rs)] = True
            out[name] = d * m.view(-1, *([1] * (d.ndim - 1))).to(d.dtype)
    return out



def _host(d: dict) -> dict:
    """Exact copy of a direction dict in host memory (patch 2026-11-01; see run, probes)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in d.items()}


def _dev(d: dict, device) -> dict:
    """The direction dict on `device` (no copy when it is already there)."""
    return {k: v.to(device) for k, v in d.items()}

def sq_norm(d: dict) -> float:
    return float(sum(float(v.double().pow(2).sum()) for v in d.values()))


def random_like_direction(direction: dict, seed: int) -> dict:
    """Same singular values as `direction` per matrix / slice, Haar-random singular vectors."""

    import torch
    import weight_controls as WC

    out = {}
    for name, d in direction.items():
        g = WC._gen_for(name, seed)
        if d.ndim == 3:
            r = torch.zeros_like(d)
            for i in range(d.shape[0]):
                if bool((d[i] != 0).any()):
                    r[i] = WC.sv_matched_random(d[i], g)
            out[name] = r
        elif d.ndim == 2:
            out[name] = WC.sv_matched_random(d, g)
        else:
            v = torch.randn(d.shape, generator=g, dtype=torch.float64)
            out[name] = (v * (d.double().norm() / v.norm())).to(d.dtype).to(d.device)
    return out


# ============================================================================= directional probes
def _kl_ce(logits0, logits1, x, chunk: int = 512):
    """Mean KL(p0 || p1) and mean CE of p1 over the window's prediction positions.

    Computed in float64 from the (fp32) logits: in fp32 the per-position log-sum-exp rounding (~ulp of
    |lse| ~ 5e-7) does not cancel in sum p0 (lp0 - lp1) and swamps KL ~ eps^2 Q at small eps (it
    produced negative KL quadratic forms on the tiny CPU model).
    """

    import torch

    l0 = logits0[0, :-1]; l1 = logits1[0, :-1]; tgt = x[0, 1:]
    kl, ce = 0.0, 0.0
    for a in range(0, l0.shape[0], chunk):
        lp0 = torch.log_softmax(l0[a:a + chunk].double(), -1)
        lp1 = torch.log_softmax(l1[a:a + chunk].double(), -1)
        kl += float((lp0.exp() * (lp0 - lp1)).sum())
        ce += float(-lp1.gather(1, tgt[a:a + chunk, None]).sum())
    n = l0.shape[0]
    return kl / n, ce / n


class Perturber:
    """Sets theta = theta_0 + c * direction in place for the named parameters, and restores theta_0."""

    def __init__(self, model):
        self.params = dict(model.named_parameters())
        self.theta0: dict = {}

    def set(self, direction: dict, c: float):
        import torch

        with torch.no_grad():
            for name, d in direction.items():
                p = self.params[name]
                if name not in self.theta0:
                    self.theta0[name] = p.detach().clone()
                p.copy_(self.theta0[name] + c * d.to(p.device, p.dtype))

    def restore(self):
        import torch

        with torch.no_grad():
            for name, t in self.theta0.items():
                self.params[name].copy_(t)


def directional_probe(model, windows: list, plan: list[tuple[str, dict, float, int]], *, device: str,
                      log=print, pin_routing: bool = False, count_flips: bool = True) -> dict:
    """For every window: base logits once, then KL and dCE for each (label, direction, eps, sign).

    pin_routing=False (free): the perturbed model routes itself, as in real adaptation. Top-k routing
        is piecewise constant, so tokens whose top-k set flips contribute O(eps) (not O(eps^2)) to KL
        and dCE; with count_flips the result carries the per-window fraction of flipped routing
        decisions ('flip', all router families pooled; eval_windows.decision_flips).
    pin_routing=True (pinned): every perturbed forward replays the routing recorded in the window's
        base forward (top-k indices AND gate values; eval_windows.RoutingPin), so eps -> CE is smooth
        and the quadratic forms are those of the fixed-routing model.
    """

    import torch

    import eval_windows as EW

    model.eval()
    pert = Perturber(model)
    res = {f"{lab}|{eps:.6g}|{sg:+d}": {"kl": [], "dce": [], "flip": []} for lab, _, eps, sg in plan}
    pin = EW.RoutingPin().install(model) if (pin_routing or count_flips) else None
    t0 = time.time()
    try:
        with torch.no_grad():
            for wi, (ws, i) in enumerate(windows):
                x = torch.as_tensor(ws.tokens(i)).view(1, -1).to(device)
                if pin is not None:
                    pin.record()
                logits0 = model(x).pred
                base_dec = pin.decisions() if pin is not None else None
                if pin is not None:
                    pin.mode = None
                _, ce0 = _kl_ce(logits0, logits0, x)
                for lab, d, eps, sg in plan:
                    pert.set(d, sg * eps)
                    flip = float("nan")
                    if pin_routing:
                        pin.buf = base_dec
                        pin.replay()
                        logits1 = model(x).pred
                        pin.done()
                    elif pin is not None:
                        pin.record()
                        logits1 = model(x).pred
                        fl = EW.decision_flips(base_dec, pin.decisions())
                        pin.mode = None
                        tot = sum(t for _, t in fl.values())
                        flip = sum(f for f, _ in fl.values()) / tot if tot else 0.0
                    else:
                        logits1 = model(x).pred
                    kl, ce = _kl_ce(logits0, logits1, x)
                    pert.restore()
                    r = res[f"{lab}|{eps:.6g}|{sg:+d}"]
                    r["kl"].append(kl); r["dce"].append(ce - ce0); r["flip"].append(flip)
                if log and (wi + 1) % 8 == 0:
                    log(f"[g0] probe {wi + 1}/{len(windows)} windows, {time.time() - t0:.0f}s")
    finally:
        if pin is not None:
            pin.uninstall()
    return {k: {"kl": np.array(v["kl"]), "dce": np.array(v["dce"]), "flip": np.array(v["flip"])} for k, v in res.items()}


def _mean_se(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"value": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan"), "n": int(len(a))}


def quad_forms(res: dict, lab: str, eps: float) -> dict:
    kp, km = res[f"{lab}|{eps:.6g}|+1"], res[f"{lab}|{eps:.6g}|-1"]
    out = {"Q_kl": _mean_se((kp["kl"] + km["kl"]) / (2 * eps * eps)),
           "Lin": _mean_se((kp["dce"] - km["dce"]) / (2 * eps)),
           "Quad": _mean_se((kp["dce"] + km["dce"]) / (2 * eps * eps))}
    fl = np.concatenate([kp.get("flip", np.array([])), km.get("flip", np.array([]))])
    if fl.size and np.isfinite(fl).all():
        out["flip_frac"] = float(fl.mean())
    return out


def richardson(res: dict, lab: str, eps: float) -> dict:
    """Central-difference truncation check: Lin_R = (4 Lin(eps/2) - Lin(eps)) / 3 cancels the O(eps^2) term."""

    a, h = quad_forms(res, lab, eps), quad_forms(res, lab, eps / 2)
    lin_r = (4 * h["Lin"]["value"] - a["Lin"]["value"]) / 3
    return {"lin_eps": a["Lin"]["value"], "lin_half_eps": h["Lin"]["value"], "lin_richardson": lin_r,
            "lin_truncation_rel": abs(a["Lin"]["value"] - lin_r) / abs(lin_r) if lin_r else float("nan"),
            "q_kl_half_eps": h["Q_kl"]["value"]}


def choose_eps(model, windows, u, *, device, eps0=0.0025, lo=3.5, hi=4.5, max_halvings=8, log=print,
               pin_routing: bool = False) -> dict:
    """Halve eps until KL(2 eps)/KL(eps) lies in [lo, hi] (quadratic-regime check, prereg sec. 5.3).

    The trail records the flipped-routing fraction of each free-routing step: under top-k flips
    KL(eps) ~ a eps + Q eps^2, the ratio tends to 2 as eps -> 0, and halving cannot pass the check; the
    pinned-routing check (pin_routing=True) tests the smooth part only.
    """

    eps, trail = eps0, []
    for _ in range(max_halvings + 1):
        r = directional_probe(model, windows, [("u", u, eps, +1), ("u", u, 2 * eps, +1)], device=device, log=None,
                              pin_routing=pin_routing)
        k1, k2 = r[f"u|{eps:.6g}|+1"]["kl"].mean(), r[f"u|{2 * eps:.6g}|+1"]["kl"].mean()
        ratio = float(k2 / k1) if k1 > 0 else float("nan")
        f1 = r[f"u|{eps:.6g}|+1"]["flip"]
        trail.append({"eps": eps, "kl_eps": float(k1), "kl_2eps": float(k2), "ratio": ratio,
                      "flip_frac_eps": float(f1.mean()) if np.isfinite(f1).all() else None})
        if lo <= ratio <= hi:
            return {"eps": eps, "ok": True, "pin_routing": pin_routing, "trail": trail}
        eps /= 2
    return {"eps": trail[-1]["eps"], "ok": False, "pin_routing": pin_routing, "trail": trail}


# ============================================================================= orchestration
def resolve_path(p: str) -> str:
    """'glob:<pattern>' -> newest match (the base checkpoints: outputs/<exp>_b05/*/checkpoints/model_final.pt)."""

    if not isinstance(p, str) or not p.startswith("glob:"):
        return p
    import glob as _glob

    hits = sorted(_glob.glob(p[5:], recursive=True), key=os.path.getmtime)
    if not hits:
        raise FileNotFoundError(f"no match for {p}")
    return os.path.abspath(hits[-1])


def load_config(path: str) -> dict:
    cfg = json.load(open(path, encoding="utf-8-sig"))         # tolerate a BOM (Windows PowerShell utf8)
    cfg.setdefault("block", 4096)
    cfg.setdefault("device", None)
    cfg.setdefault("eval_tok_per_s", 9600.0)
    if cfg.get("checkpoint"):
        cfg["checkpoint_spec"] = cfg["checkpoint"]
        try:
            cfg["checkpoint"] = resolve_path(cfg["checkpoint"])
        except FileNotFoundError:
            pass                                       # `estimate` works without the checkpoint
    return cfg


def estimate(cfg: dict) -> dict:
    """GPU-time estimate from token counts and the measured fp32 eval throughput (default 9.6k tok/s)."""

    B = cfg["block"]
    tps = float(cfg["eval_tok_per_s"])
    routing_tok = sum(int(s.get("tokens", 0)) for s in cfg["streams"].values())
    pr = cfg.get("probe", {})
    probe_windows = int(pr.get("probe_windows", 16)) * len(pr.get("probe_corpora", []))
    modes = 2 if pr.get("pin_routing", "both") == "both" else 1
    rand = len(pr.get("random_seeds", [0]))
    c1_draws = int(cfg.get("crossover", {}).get("draws", 3)) if cfg.get("crossover") else 0
    c2 = cfg.get("c2") or {}
    regime_w = int(pr.get("regime_windows", 8))
    probe_tok, per_target = 0, {}
    for tgt in pr.get("targets", []):
        dirs = 1 + 2 * c1_draws + (2 * int(c2.get("draws", 3)) if c2.get("target") == tgt else 0)
        fwd = modes * (1 + dirs * 2 * (1 + rand) + (2 if pr.get("richardson", True) else 0))   # base + (+/-) x dirs
        tok = (probe_windows * fwd + modes * 2 * 4 * regime_w) * B + int(pr.get("grad_windows", 16)) * B * 3
        per_target[tgt] = {"directions": dirs, "forwards_per_window": fwd, "tokens": tok}
        probe_tok += tok
    h = (routing_tok + probe_tok) / tps / 3600
    return {"routing_tokens": routing_tok, "probe_tokens": probe_tok, "per_target": per_target, "gpu_hours": round(h, 4),
            "assumption": f"fp32 eval {tps:.0f} tok/s for every forward; backward ~2 forwards; eps check ~4 halvings"}


def _stream_sets(cfg):
    import eval_windows as EW

    out = {}
    for name, s in cfg["streams"].items():
        out[name] = EW.WindowSet(name, s["path"], cfg["block"], -(-int(s["tokens"]) // cfg["block"]),
                                 s.get("select", "even"), int(s.get("seed", 0)),
                                 span=s.get("span"), span_align=s.get("span_align"))
    return out


def check_streams(cfg: dict) -> dict:
    """Preflight without a model: the windows every stream will really get (file size, span, eligible grid).
    select_windows silently caps n at the eligible windows, so a test bin whose half is too small for the
    registered 256 selection / 500 evaluation windows (prereg sec. 4.3) shows up here as 'short'."""

    for p in (HERE, os.path.join(HERE, "repo")):
        if p not in sys.path:
            sys.path.insert(0, p)
    rows, short = {}, []
    for name, s in cfg["streams"].items():
        want = -(-int(s["tokens"]) // int(cfg["block"]))
        ws = _stream_sets({"streams": {name: s}, "block": cfg["block"]})[name]
        got = int(len(ws.index))
        rows[name] = {"path": ws.path, "file_tokens": ws.file_tokens, "span": ws.span, "span_tokens": ws.span_tokens,
                      "requested_windows": want, "selected_windows": got,
                      "first_window": int(ws.index[0]) if got else None, "last_window": int(ws.index[-1]) if got else None}
        if got < want:
            short.append(name)
    return {"streams": rows, "short": short, "ok": not short}


def run(cfg: dict, *, model=None, log=print) -> dict:
    import torch

    import crossover as CX
    import eval_windows as EW

    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("checkpoint"):
        cfg["checkpoint"] = resolve_path(cfg["checkpoint"])
    if model is None:
        if cfg.get("grad_checkpoint"):
            os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "1"
        EW.register_extra(cfg.get("register"))
        model, _ = EW.load_model(cfg["experiment"], cfg["checkpoint"], device)
    t_start = time.time()
    sets = _stream_sets(cfg)
    files = {}
    for name, ws in sets.items():                        # 1. routing statistics (+ base per-window CE)
        f = os.path.join(out_dir, f"routing__{name}.npz")
        if not os.path.exists(f):
            res = EW.evaluate(model, ws, device=device, routing=True, log=log)
            EW.save_result(f, res, {"variant": "base", "set": ws.describe(), "experiment": cfg.get("experiment"),
                                    "checkpoint": cfg.get("checkpoint"), "precision": "fp32"})
        files[name] = f
    fam, k = cfg["family"], int(cfg["k"])
    summary: dict[str, Any] = {"arch": cfg.get("arch"), "family": fam, "k": k, "contrasts": {}, "probes": {}}
    cx = cfg.get("crossover")
    contrast_files = {}
    if cx:                                               # 2. contrasts
        tab = CX.load_tables(files, fam, cfg.get("measure", "g2"), int(cfg.get("half_seed", 0)))
        jobs = []
        for tgt in cx.get("targets", []):
            jobs.append((f"C1_{tgt}", {"type": "crossover", "low": cx["low"], "high": cx["high"]},
                         [tgt] + list(cx.get("match_extra", ["dclm"])), cx))
        if cfg.get("c2"):
            c2 = cfg["c2"]
            jobs.append((f"C2_{c2['target']}", {"type": "tilt", "corpus": c2["target"]}, c2["match"], c2))
        if cfg.get("dclm_tilt"):
            dt = cfg["dclm_tilt"]
            jobs.append(("DCLM_tilt", {"type": "tilt", "corpus": "dclm"}, dt["match"], dt))
        for name, contrast, match, opts in jobs:
            f = os.path.join(out_dir, f"contrast__{name}.json")
            if not os.path.exists(f):
                rep = CX.draw_pairs(tab, k=int(opts.get("k", k)), contrast=contrast, match=match,
                                    tol=float(opts.get("tol", 0.05)), n_draws=int(opts.get("draws", 3)),
                                    seed=int(opts.get("seed", 0)), lam=float(opts.get("lam", 1.5)),
                                    min_contrast=float(opts.get("min_contrast", math.log(1.3) if contrast["type"] == "crossover" else 0.1)),
                                    max_jaccard=float(opts.get("max_jaccard", 0.5)))
                json.dump(rep, open(f, "w", encoding="utf-8"), indent=1)
            rep = json.load(open(f, encoding="utf-8"))
            contrast_files[name] = f
            summary["contrasts"][name] = {"draws_found": rep["n_draws_found"], "attempts": rep["attempts"],
                                          "best_estimation_contrast": rep["best_estimation_contrast"],
                                          "draw_estimation_contrasts": [d["estimation_half"]["contrast"] for d in rep["draws"]],
                                          "tol": rep["tol"], "file": f}
            scan = opts.get("tol_scan") or []
            if scan:                                     # feasibility vs matching tolerance (no model needed)
                fs = os.path.join(out_dir, f"contrast__{name}__tolscan.json")
                if not os.path.exists(fs):
                    rows = {}
                    for tol in scan:
                        r = CX.draw_pairs(tab, k=int(opts.get("k", k)), contrast=contrast, match=match, tol=float(tol),
                                          n_draws=int(opts.get("draws", 3)), seed=int(opts.get("seed", 0)),
                                          lam=float(opts.get("lam", 1.5)),
                                          min_contrast=float(opts.get("min_contrast", math.log(1.3) if contrast["type"] == "crossover" else 0.1)),
                                          max_jaccard=float(opts.get("max_jaccard", 0.5)), best_search=20)
                        rows[str(tol)] = {"draws_found": r["n_draws_found"], "best_estimation_contrast": r["best_estimation_contrast"]}
                    json.dump(rows, open(fs, "w", encoding="utf-8"), indent=1)
                summary["contrasts"][name]["tol_scan"] = json.load(open(fs, encoding="utf-8"))
    pr = cfg.get("probe")
    if pr:                                               # 3. probes
        rows = unit_rows(model)
        params = expert_params(model)
        old = []
        for c in pr.get("probe_corpora", []):
            ws = sets[c]
            half = EW.split_halves(len(ws.index), seed=int(cfg.get("half_seed", 0)))
            idx = [i for i in range(len(ws.index)) if half[i] == 1][: int(pr.get("probe_windows", 16))]
            old += [(ws, i) for i in idx]
        lr_m = float(pr.get("lr_muon", 0.0025))
        for tgt in pr.get("targets", []):
            f = os.path.join(out_dir, f"probe__{tgt}.json")
            if os.path.exists(f):
                summary["probes"][tgt] = json.load(open(f, encoding="utf-8"))
                continue
            gpath = (pr.get("grad_paths") or {}).get(tgt, cfg["streams"][tgt]["path"])   # target TRAIN data when given
            gws = EW.WindowSet(tgt + "_grad", gpath, cfg["block"], int(pr.get("grad_windows", 16)),
                               "random", int(pr.get("grad_seed", 12345)))
            grads = target_gradient(model, gws, params, int(pr.get("grad_windows", 16)), device=device,
                                    bf16=bool(pr.get("bf16_grad", True)))
            u_e = muon_direction(params, grads, momentum=float(pr.get("momentum", 0.95)),
                                 nesterov=bool(pr.get("nesterov", True)))
            modes = {"both": ("pinned", "free"), "free": ("free",), "pinned": ("pinned",)}[pr.get("pin_routing", "both")]
            primary = modes[0]
            eps_check = {m: choose_eps(model, old[: int(pr.get("regime_windows", 8))], u_e, device=device,
                                       eps0=float(pr.get("eps0", lr_m)), pin_routing=(m == "pinned"), log=None)
                         for m in modes}
            eps_info = eps_check[primary]
            eps = eps_info["eps"]
            # patch 2026-11-01 (performance only, values unchanged): every direction except E_ALL is computed on
            # the device exactly as before and then COPIED to host memory; Perturber.set moves one direction at a
            # time to the device. Keeping ~40 expert-sized directions on a 16 GB GPU made Windows/WDDM silently page
            # VRAM to system memory (no OOM; grouped GEMMs ~200x slower).
            dirs = {"E_ALL": (u_e, grads)}
            for name in (f"C1_{tgt}", f"C2_{tgt}"):      # C2 draws feed the SGD-arm lr calibration
                if name in contrast_files:
                    rep = json.load(open(contrast_files[name], encoding="utf-8"))
                    for di, d in enumerate(rep["draws"]):
                        for side in ("A", "B"):
                            dirs[f"{name}_d{di}_{side}"] = (_host(restrict(u_e, rows, d[side])), _host(restrict(grads, rows, d[side])))
            plan = [("E_ALL", u_e, eps / 2, +1), ("E_ALL", u_e, eps / 2, -1)] if pr.get("richardson", True) else []
            for lab, (u, _) in dirs.items():
                plan += [(lab, u, eps, +1), (lab, u, eps, -1)]
                for sd in pr.get("random_seeds", [0]):
                    r = _host(random_like_direction(_dev(u, device), int(sd)))
                    plan += [(f"{lab}~rand{sd}", r, eps, +1), (f"{lab}~rand{sd}", r, eps, -1)]
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            res_by = {m: directional_probe(model, old, plan, device=device, log=log, pin_routing=(m == "pinned"))
                      for m in modes}
            corpora = sorted({ws.name for ws, _ in old})
            corpus_of = np.array([ws.name for ws, _ in old])
            # per-window values (G0-6 split-half reliability needs them; the JSON keeps means/SEs only)
            wfile = os.path.join(out_dir, f"probe__{tgt}__windows.npz")
            warr = {"corpus": corpus_of.astype(str), "window_index": np.array([int(ws.index[i]) for ws, i in old], dtype=np.int64),
                    "window_hash": np.array([int(ws.hashes[i]) for ws, i in old], dtype=np.uint64),
                    "eps": np.array(float(eps)), "primary_routing": np.array(primary)}
            for m, res in res_by.items():
                for key, v in res.items():
                    for fld in ("kl", "dce", "flip"):
                        warr[f"{m}|{key}|{fld}"] = np.asarray(v[fld], dtype=np.float64)
            np.savez_compressed(wfile + ".tmp.npz", **warr)
            os.replace(wfile + ".tmp.npz", wfile)
            rep_t = {"target": tgt, "eps": eps, "eps_check": eps_check, "primary_routing": primary,
                     "routing_modes": list(modes), "lr_muon": lr_m, "probe_windows": len(old),
                     "windows_file": os.path.basename(wfile),
                     "corpora": corpora, "subsets": {}}
            for lab, (u, g) in dirs.items():
                entry = {"grad_sq": sq_norm(_dev(g, device)), "udir_sq": sq_norm(_dev(u, device))}
                for m, res in res_by.items():
                    em = {"all": quad_forms(res, lab, eps), "by_corpus": {}}
                    for c in corpora:
                        msk = corpus_of == c
                        sub = {kk: {f: vv[f][msk] for f in vv} for kk, vv in res.items()}
                        em["by_corpus"][c] = quad_forms(sub, lab, eps)
                    for sd in pr.get("random_seeds", [0]):
                        em[f"rand{sd}"] = quad_forms(res, f"{lab}~rand{sd}", eps)
                    if lab == "E_ALL" and pr.get("richardson", True):
                        em["richardson"] = richardson(res, lab, eps)
                    entry[m] = em
                entry["all"], entry["by_corpus"] = entry[primary]["all"], entry[primary]["by_corpus"]
                rep_t["subsets"][lab] = entry
            rep_t["Q_E"] = rep_t["subsets"]["E_ALL"]["all"]["Q_kl"]
            json.dump(rep_t, open(f, "w", encoding="utf-8"), indent=1)
            summary["probes"][tgt] = rep_t
    best = [v["best_estimation_contrast"] for kk, v in summary["contrasts"].items() if kk.startswith("C1_")]
    summary["best_C1_abs_I"] = max((abs(b) for b in best if b is not None), default=0.0)
    summary["seconds"] = round(time.time() - t_start, 1)
    summary["routing_files"] = files
    json.dump(summary, open(os.path.join(out_dir, "g0_summary.json"), "w", encoding="utf-8"), indent=1)
    return summary


def xplatform(a: str, b: str, tol: float = 1e-3) -> dict:
    import eval_windows as EW

    r = EW.paired(EW.load_result(a), EW.load_result(b))
    r["tolerance"] = tol
    r["pass"] = bool(abs(r["diff"]) <= tol)
    r["rule"] = "stop Nestor adaptation if the trained-checkpoint cross-platform |dCE| > 1e-3 nats"
    return r


def decide(summaries: list[dict], xp: dict | None = None) -> dict:
    import crossover as CX

    best = {s.get("arch") or f"arch{i}": s.get("best_C1_abs_I", 0.0) for i, s in enumerate(summaries)}
    out = CX.stop_rule(best)
    out["superseded_by"] = ("g0_gate.py: the registered gates (prereg v1.0 sec. 6) use 1,000 registered generator draws, "
                            "G0-5 also needs per-window r >= 0.999; this phase-3 heuristic is NOT a gate decision")
    if xp is not None:
        out["nestor_platform_check"] = {"pass": xp["pass"], "abs_dce": abs(xp["diff"]), "corr": xp.get("corr")}
        out["nestor_adaptation"] = "GO" if xp["pass"] else "STOP"
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="G0 pre-adaptation gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate"); e.add_argument("--config", required=True)
    pf = sub.add_parser("streams", help="preflight: windows per stream (no model)"); pf.add_argument("--config", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", required=True)
    x = sub.add_parser("xplatform"); x.add_argument("--a", required=True); x.add_argument("--b", required=True)
    x.add_argument("--tol", type=float, default=1e-3); x.add_argument("--out", default=None)
    d = sub.add_parser("decide"); d.add_argument("--summary", action="append", required=True)
    d.add_argument("--xplatform", default=None); d.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "estimate":
        print(json.dumps(estimate(load_config(a.config)), indent=1))
        return 0
    if a.cmd == "streams":
        chk = check_streams(load_config(a.config))
        print(json.dumps(chk, indent=1))
        return 0 if chk["ok"] else 2
    if a.cmd == "run":
        cfg = load_config(a.config)
        print(json.dumps({"estimate": estimate(cfg)}), flush=True)
        s = run(cfg)
        print(json.dumps({k: s[k] for k in ("arch", "best_C1_abs_I", "seconds")}), flush=True)
        return 0
    if a.cmd == "xplatform":
        res = xplatform(a.a, a.b, a.tol)
    else:
        res = decide([json.load(open(f, encoding="utf-8")) for f in a.summary],
                     json.load(open(a.xplatform, encoding="utf-8")) if a.xplatform else None)
    txt = json.dumps(res, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
