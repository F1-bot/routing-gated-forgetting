"""theta0 probe units of the prospective forecasts (prereg v1.3 sec. 5.3). GPU at stage C (queue_forecast_inputs.json from
forecasts.py units); CPU-tested on the tiny models (test_forecast_probe.py).

  python forecast_probe.py run --units results/forecast/forecast_units.json --base moe_b1 --out results/forecast/probes \
                               [--device cuda] [--only 3,4] [--fp32-grad]

Per unit (base, subset S, target T, optimizer), on PROBE_WINDOWS = 8 windows of every old corpus c (the first 8 windows of
its registered evaluation set), routing pinned to theta0 (eval_windows.RoutingPin replay of selections and gates):
  u_S    first-step direction at the unit Muon rate (theta_1 - theta_0 = -lr_muon u, weight decay and warm-up aside) of the
         Muon-group matrices of the unit's trainable set: mean training-loss gradient on 8 windows of the unit's training
         data (g0_probe.target_gradient: routers frozen, training-mode forward, the authors' loss; G0 'random' windows,
         seed 12345), transformed as the optimizer's first step: Muon units muon_scale NS5((1 + mu) g)
         (g0_probe.muon_direction); SGD units (C2 SGD arm, G2 SGD calibration) (lr_sgd / lr_muon)(1 + mu) g (Nesterov);
         restricted to the rows of the unit's trainable units (g0_probe.restrict). AdamW-group parameters are not part of
         u_S (G0 probe convention).
  eps    one per gradient group (training data, optimizer, rate): g0_probe.choose_eps along the group direction on the
         routed experts, pinned, eps0 = 0.0025, on the first 8 DCLM windows.
  QE_c   (KL+ + KL-) / (2 eps^2) along u_S;  gamma_c = (dCE+ - dCE-) / (2 eps) along u_S;
  RND_c  (dCE+ + dCE-) / (2 eps^2) along h = g0_probe.random_like_direction(u_S, 9000 + unit index) (per matrix / expert
         slice the singular values of u_S, Haar singular vectors; ||h|| = ||u_S||) = the registered
         (L(theta0 + h) + L(theta0 - h) - 2 L(theta0)) / 2 in the local quadratic regime;
  SVC_c  sum over the expert matrices m of S of (1/N_c) sum_t ||x_t,m dW_m||^2: x_t,m the theta0 input rows of m (tokens
         routed to the expert; the down projection's input is the expert's hidden activation), N_c all tokens of the 8
         windows, dW_m = u_S on m (attention and other non-expert matrices are not part of SVC);
  TF     elements: trainable parameter elements of the unit's spec (subset_adapt catalogue; routers frozen as under
         NESTOR_FREEZE_ROUTERS=1) and those of the FULL recipe.
Output: <out>/probe_u<index>.json (bound to the unit by unit_sha256 = forecast_units.json probe_sha256) and
<out>/probe_u<index>__windows.npz (per-window KL / dCE of the four legs). Finished units are skipped on restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

LR_REF = 0.0025          # registered Muon rate (prereg v1.2 sec. 4.4): the unit of the probe directions and eps0
GRAD_SEED = 12345        # g0_probe target-gradient windows ('random')
PROBE_WINDOWS = 8
GRAD_WINDOWS = 8
EXPERT_KINDS = ("moe", "pub", "priv")
_EXPERT_MOD = re.compile(r"\.(shared_experts\.\d+|private_experts\.\d+\.\d+|public_experts\.\d+)$")


def _sha_file(p: str | None) -> str | None:
    if not p or not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def eval_set(name: str, d: dict, block: int):
    """The registered evaluation set `d` (path, tokens, select, span, span_align) as an eval_windows.WindowSet."""
    import eval_windows as EW

    n = int(d.get("tokens", 2_048_000)) // int(block)
    return EW.WindowSet(name, d["path"], int(block), n, d.get("select", "even"), int(d.get("seed", 0)),
                        span=d.get("span"), span_align=d.get("span_align"))


# ============================================================================= trainable set and TF counts
def trainable_units(model, spec: dict | None) -> tuple[set, list]:
    """The units trained under `spec` exactly as subset_adapt.apply_subset selects them with NESTOR_FREEZE_ROUTERS=1
    (live routers and construction-frozen tensors excluded), without touching the model."""
    import subset_adapt as SA

    entries = SA.build_catalog(model)
    geo = SA.model_geometry(model)
    all_u = [u for e in entries for u in e.units]
    if spec is None:
        tr = set(all_u)
    else:
        flt = spec.get("select", [])
        sel = {u for u in all_u if any(SA.unit_matches(u, f, geo) for f in flt)}
        tr = sel if spec.get("mode", "train") == "train" else set(all_u) - sel
    routers = {u for e in entries if not e.construction_frozen for u in e.units if u.kind == "router"}
    frozen = {u for e in entries if e.construction_frozen for u in e.units}
    return tr - routers - frozen, entries


def element_count(entries: list, tr: set) -> int:
    n = 0
    for e in entries:
        if e.construction_frozen:
            continue
        if e.sliced:
            n += int(e.param[0].numel()) * sum(1 for u in e.units if u in tr)
        elif e.units[0] in tr:
            n += int(e.param.numel())
    return n


def muon_params(entries: list) -> dict:
    """Muon-group parameters (every Muon matrix / bank that some recipe can train; routers excluded)."""
    return {e.name: e.param for e in entries if e.optimizer == "muon" and not e.construction_frozen
            and not all(u.kind == "router" for u in e.units)}


# ============================================================================= SVC: theta0 inputs of the expert matrices
class SVCCapture:
    """Accumulates, for every gradient-group direction and every expert matrix row (bank slice), sum_t ||x_t dW||^2 over
    the theta0 input rows x_t of that matrix (tokens routed to the expert) in the forwards run while `active`."""

    def __init__(self, model, groups: dict, device):
        import torch

        self.torch, self.device, self.groups = torch, device, groups
        self.slot: dict[int, tuple[str, int | None, bool]] = {}
        for name, p in model.named_parameters():
            if p.ndim == 3:
                for e in range(p.shape[0]):
                    self.slot[p[e].data_ptr()] = (name, e, True)
            elif p.ndim == 2:
                self.slot[p.data_ptr()] = (name, None, False)
        self.acc = {g: defaultdict(float) for g in groups}
        self.active, self.in_grouped, self.hits = False, False, 0
        self.handles, self.patched = [], []
        for name, mod in model.named_modules():
            if _EXPERT_MOD.search(name):
                self.handles.append(mod.register_forward_pre_hook(self._pre_hook))
        import nestor.backbones.grouped_mm as GM

        orig = GM.grouped_expert_rows
        cap = self

        def wrapped(x, experts, expert_ids, *a, **kw):
            cap.in_grouped = True
            try:
                out = orig(x, experts, expert_ids, *a, **kw)
            finally:
                cap.in_grouped = False
            if cap.active:
                ids = expert_ids.reshape(x.shape[0], -1)
                for e in torch.unique(ids).tolist():
                    cap.capture(experts[int(e)], x[(ids == int(e)).any(-1)])
            return out

        for mname, mod in list(sys.modules.items()):
            if mname.startswith("nestor") and getattr(mod, "grouped_expert_rows", None) is orig:
                setattr(mod, "grouped_expert_rows", wrapped)
                self.patched.append((mod, orig))

    def _pre_hook(self, mod, args):
        if self.active and not self.in_grouped and args:
            self.capture(mod, args[0].reshape(-1, args[0].shape[-1]))

    @staticmethod
    def _parts(expert) -> list:
        layers = expert.layers
        if getattr(expert, "gated", False) or hasattr(expert, "gate"):
            return [("x", layers[0]), ("x", expert.gate), ("h", layers[1])]
        return [("x", layers[0]), ("h", layers[-1])]

    @staticmethod
    def _hidden(expert, X):
        if getattr(expert, "gated", False) or hasattr(expert, "gate"):
            return expert._activation_fn(expert.gate(X)) * expert.layers[0](X)
        h = X
        for lay in expert.layers[:-1]:
            h = expert._activation_fn(lay(h))
        return h

    def capture(self, expert, X):
        torch = self.torch
        if X.numel() == 0:
            return
        with torch.no_grad():
            X = X.float()
            H = self._hidden(expert, X).float()
            for kind, lin in self._parts(expert):
                s = self.slot.get(lin.weight.data_ptr())
                if s is None:
                    continue
                name, row, stacked = s
                inp = X if kind == "x" else H
                for g, d in self.groups.items():
                    dw = d.get(name)
                    if dw is None:
                        continue
                    dW = (dw[row] if row is not None else dw).to(inp.device, torch.float32)
                    if not bool((dW != 0).any()):
                        continue
                    y = inp @ dW if stacked else inp @ dW.t()
                    self.acc[g][(name, row)] += float(y.double().pow(2).sum())
                    self.hits += 1

    def close(self):
        for h in self.handles:
            h.remove()
        for mod, orig in self.patched:
            setattr(mod, "grouped_expert_rows", orig)
        self.handles, self.patched = [], []


def svc_pass(model, windows: list, groups: dict, device) -> tuple[dict, int]:
    """theta0 forward over `windows` with SVCCapture: {group: {(param, row): sum_t ||x dW||^2}}, number of tokens."""
    import torch

    cap = SVCCapture(model, groups, device)
    n_tok = 0
    model.eval()
    try:
        cap.active = True
        with torch.no_grad():
            for ws, i in windows:
                x = torch.as_tensor(ws.tokens(i)).view(1, -1).to(device)
                model(x)
                n_tok += int(x.numel())
    finally:
        cap.active = False
        cap.close()
    return {g: dict(v) for g, v in cap.acc.items()}, n_tok


# ============================================================================= probes
def group_key(u: dict) -> str:
    o = u["optimizer"]
    return f"{u['data']}|{o['kind']}|{float(o['lr']):.9g}"


def group_direction(params: dict, grads: dict, opt: dict) -> dict:
    """Unit-Muon-rate direction of the optimizer's first step for every Muon-group parameter (device tensors)."""
    import g0_probe as G0

    mu, nest = float(opt.get("momentum", 0.95)), bool(opt.get("nesterov", True))
    if opt["kind"] == "muon":
        return G0.muon_direction(params, grads, momentum=mu, nesterov=nest)
    if opt["kind"] == "sgd":
        f = float(opt["lr"]) / LR_REF * ((1.0 + mu) if nest else 1.0)
        return {k: g.float() * f for k, g in grads.items()}
    raise ValueError(f"unknown optimizer kind {opt['kind']!r}")


def probe_base(model, U: dict, base: str, out: str, *, device: str, only: set | None = None, bf16_grad: bool = True,
               log=print) -> dict:
    """Every unit of `base` in the units document U (forecast_units.json); writes one file per unit."""
    import torch

    import eval_windows as EW
    import g0_probe as G0

    os.makedirs(out, exist_ok=True)
    binfo = U["bases"][base]
    block = int(U.get("block", 4096))
    units = [u for u in U["units"] if u["base"] == base and (only is None or u["index"] in only)]
    todo = []
    for u in units:
        f = os.path.join(out, f"probe_u{u['index']:03d}.json")
        if os.path.exists(f) and json.load(open(f, encoding="utf-8")).get("unit_sha256") == u["probe_sha256"]:
            continue
        todo.append(u)
    if not todo:
        return {"base": base, "done": 0, "skipped": len(units)}
    t0 = time.time()
    sets = {c: eval_set(c, binfo["sets"][c], block) for c in ("dclm", "code", "math")}
    for c, ws in sets.items():
        if len(ws.index) < PROBE_WINDOWS:
            raise SystemExit(f"{base}: evaluation set {c} has {len(ws.index)} < {PROBE_WINDOWS} windows")
    win = {c: [(sets[c], i) for i in range(PROBE_WINDOWS)] for c in sets}
    rows = G0.unit_rows(model)
    _, entries = trainable_units(model, None)
    params = muon_params(entries)
    full_tr, _ = trainable_units(model, None)
    n_full = element_count(entries, full_tr)
    expert_keys = [u_.key for e in entries for u_ in e.units if u_.kind in EXPERT_KINDS]
    groups, grads_cache, eps_info = {}, {}, {}
    for u in todo:                                     # gradient groups: (training data, optimizer, rate)
        gk = group_key(u)
        if gk in groups:
            continue
        if u["data"] not in grads_cache:
            gws = EW.WindowSet(f"{u['target']}_grad", u["data"], block, GRAD_WINDOWS, "random", GRAD_SEED)
            grads_cache = {u["data"]: G0.target_gradient(model, gws, params, GRAD_WINDOWS, device=device, bf16=bf16_grad)}
        d = group_direction(params, grads_cache[u["data"]], u["optimizer"])
        groups[gk] = G0._host(d)
        del d
        e_dir = G0.restrict(groups[gk], rows, expert_keys)
        eps_info[gk] = G0.choose_eps(model, win["dclm"], G0._dev(e_dir, device), device=device, eps0=LR_REF,
                                     pin_routing=True, log=None)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    grads_cache.clear()
    svc_acc, svc_ntok = {}, {}
    for c in sets:                                     # one theta0 capture pass per corpus serves every unit
        svc_acc[c], svc_ntok[c] = svc_pass(model, win[c], groups, device)
    code = {n: _sha_file(os.path.join(HERE, n)) for n in ("forecast_probe.py", "g0_probe.py", "eval_windows.py",
                                                         "subset_adapt.py", "weight_controls.py")}
    done = 0
    for u in todo:
        tu = time.time()
        gk = group_key(u)
        tr, _ = trainable_units(model, u["spec"])
        keys = sorted(x.key for x in tr)
        uS = G0.restrict(groups[gk], rows, keys)
        h = G0._host(G0.random_like_direction(G0._dev(uS, device), int(u["haar_seed"])))
        eps = float(eps_info[gk]["eps"])
        old = [c for c in u["old_corpora"] if c in sets]
        windows = [w for c in old for w in win[c]]
        corpus_of = np.array([c for c in old for _ in win[c]])
        plan = [("u", uS, eps, +1), ("u", uS, eps, -1), ("h", h, eps, +1), ("h", h, eps, -1)]
        res = G0.directional_probe(model, windows, plan, device=device, log=None, pin_routing=True, count_flips=False)
        by = {}
        trows = {(n, r) for k in keys for n, r in rows.get(k, [])}
        for c in old:
            m = corpus_of == c
            sub = {k: {f: v[f][m] for f in v} for k, v in res.items()}
            qu, qh = G0.quad_forms(sub, "u", eps), G0.quad_forms(sub, "h", eps)
            svc = sum(v for (n, r), v in svc_acc[c][gk].items() if (n, r) in trows) / max(svc_ntok[c], 1)
            by[c] = {"QE": qu["Q_kl"]["value"], "QE_se": qu["Q_kl"]["se"], "gamma": qu["Lin"]["value"],
                     "gamma_se": qu["Lin"]["se"], "quad_u": qu["Quad"]["value"], "RND": qh["Quad"]["value"],
                     "RND_se": qh["Quad"]["se"], "RND_Q_kl": qh["Q_kl"]["value"], "SVC": float(svc),
                     "windows": int(m.sum())}
        wfile = os.path.join(out, f"probe_u{u['index']:03d}__windows.npz")
        warr = {"corpus": corpus_of.astype(str), "window_index": np.array([int(ws.index[i]) for ws, i in windows]),
                "window_hash": np.array([int(ws.hashes[i]) for ws, i in windows], dtype=np.uint64), "eps": np.array(eps)}
        for k, v in res.items():
            for fld in ("kl", "dce"):
                warr[f"{k}|{fld}"] = np.asarray(v[fld], dtype=np.float64)
        np.savez_compressed(wfile + ".tmp.npz", **warr)
        os.replace(wfile + ".tmp.npz", wfile)
        doc = {"unit": u["index"], "unit_sha256": u["probe_sha256"], "base": base, "mock": False, "by_corpus": by,
               "elements": {"trainable": element_count(entries, tr), "full_trainable": n_full},
               "udir_sq": G0.sq_norm(uS), "hdir_sq": G0.sq_norm(h), "eps": eps, "eps_check": eps_info[gk],
               "group": gk, "routing": "pinned (theta0 selections and gates)", "lr_ref": LR_REF,
               "windows_file": os.path.basename(wfile), "svc_tokens": svc_ntok, "device": str(device),
               "checkpoint": binfo.get("checkpoint"), "code_sha256": code, "seconds": round(time.time() - tu, 1)}
        tmp = os.path.join(out, f"probe_u{u['index']:03d}.json.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, os.path.join(out, f"probe_u{u['index']:03d}.json"))
        done += 1
        if log:
            log(f"[forecast_probe] {base} unit {u['index']} ({u['subset']}, {u['target']}, {u['optimizer']['kind']}): "
                f"{doc['seconds']} s")
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"base": base, "done": done, "skipped": len(units) - len(todo), "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="theta0 probe units of the prospective forecasts (prereg v1.3 sec. 5.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--units", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--device", default=None)
    r.add_argument("--only", default="")
    r.add_argument("--fp32-grad", action="store_true")
    r.add_argument("--checkpoint", default=None, help="override (tests); default: the queue variable of the base")
    r.add_argument("--experiment", default=None)
    r.add_argument("--register", default=None, help="extra experiment registry (tests: tiny_experiments.py)")
    a = ap.parse_args(argv)
    import torch

    import eval_windows as EW
    import g0_probe as G0

    U = json.load(open(a.units, encoding="utf-8"))
    b = U["bases"][a.base]
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    EW.register_extra(a.register)
    ck = G0.resolve_path(a.checkpoint or b.get("checkpoint") or "")
    if not ck or not os.path.exists(ck):
        raise SystemExit(f"checkpoint of {a.base} not found: {ck!r}")
    model, _ = EW.load_model(a.experiment or b["experiment"], ck, device)
    only = {int(x) for x in a.only.split(",") if x.strip()} or None
    res = probe_base(model, U, a.base, a.out, device=device, only=only, bf16_grad=not a.fp32_grad)
    print(f"[forecast_probe] {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
