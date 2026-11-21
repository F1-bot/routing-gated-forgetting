"""Checks at theta_0 on subset R1 of the S1 A/A pair (MoE base 1, German) -- one GPU job in the G1 queue.

  python theta0_checks.py run moe_100m <ckpt> --draw g1/specs/aa_moe_german_a1.json --subset A \
      --data data/domain/fw2de_train.bin --windows 64 --data-seed 7201 --out g1/theta0_checks.json

g = mean training-loss gradient (authors' loss, routers frozen, bf16 autocast as in training; g0_probe.target_gradient) over
`windows` windows of the target training stream (64 windows x 4096 tokens = one optimizer step of 262,144 tokens), w.r.t. the
Muon expert matrices. The trainable matrices m of R1 are the expert slices (rows of stacked banks) of R1's units.

1. SGD scale c (prereg v1.1 G2-2): "c equates the mean first-step update Frobenius norm of SGD and Muon on R1 at theta_0".
   First step from zero momentum with Nesterov (buf = g, direction g + mu buf = (1 + mu) g):
       Muon  ||dtheta_m|| = lr_muon * ||u_m||_F,  u_m = muon_scale_m * NS5((1 + mu) g_m)    (g0_probe.muon_direction)
       SGD   ||dtheta_m|| = lr_sgd * (1 + mu) * ||g_m||_F
   Both arms share the warm-up/cosine shape, so equal step-1 norms <=> equal peak lrs ratio:
       c = lr_muon * mean_m ||u_m||_F / ((1 + mu) * mean_m ||g_m||_F);   grid = [c/3, c, 3c] (sgd.grid, read by the G2 queue).
   (adapt_extras.sgd_lr_grid matches TOTAL norms over [steady-state, first-step]; it is not the registered rule.)
2. NS-5 scale check (sec. 4.9 item 10): rho(s) = ||NS5(s D) - NS5(D)||_F / ||NS5(D)||_F per matrix for s in {0.25, 0.5, 2},
   D = (1 + mu) g_m (the optimizer's NS5 input, bf16 inside NS5 as in the optimizer); median and max over R1's matrices.
3. D_Muon unit test at step 1 (sec. 4.9 item 10): the authors' MuonAdamW (lr_muon, weight decay 0) stepped once with grads g
   moves R1's matrices by -lr_muon * u_m: relative error ||dtheta_m / (-lr_muon) - u_m|| / ||u_m|| per matrix (pass: max <= 1e-3).
Missing gradients (an expert slice that received no token) count as exact zeros (finding eb4b1968).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

NS5_SCALES = (0.25, 0.5, 2.0)
D_MUON_TOL = 1e-3


def r1_matrices(model, unit_keys: list[str], params: dict) -> list[tuple[str, int | None]]:
    import g0_probe as G0

    rows = G0.unit_rows(model)
    out = []
    for k in unit_keys:
        for name, r in rows.get(k, []):
            if name in params:
                out.append((name, r))
    return sorted(set(out), key=lambda x: (x[0], -1 if x[1] is None else x[1]))


def _slice(t, r):
    return t if r is None else t[r]


def zero_filled(grads: dict, params: dict) -> tuple[dict, list[str]]:
    """Every requested parameter gets a gradient tensor: a parameter without one (no token reached it) is an exact zero."""

    import torch

    out, missing = {}, []
    for name, p in params.items():
        g = grads.get(name)
        if g is None:
            missing.append(name)
            g = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        out[name] = g
    return out, missing


def ns5(d):
    from nestor.scaling.muon import zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_batched

    return zeropower_via_newtonschulz5_batched(d, 5) if d.ndim == 3 else zeropower_via_newtonschulz5(d, 5)


def checks(model, wset, unit_keys: list[str], *, device: str, windows: int, lr_muon: float, momentum: float,
           bf16: bool = True, d_muon: bool = True) -> dict:
    import torch

    import g0_probe as G0

    params_all = G0.expert_params(model)
    mats = r1_matrices(model, unit_keys, params_all)
    if not mats:
        raise ValueError("R1 has no Muon expert matrices in this model")
    names = sorted({n for n, _ in mats})
    params = {n: params_all[n] for n in names}
    import memguard as MG

    g_raw = G0.target_gradient(model, wset, params, windows, device=device, bf16=bf16)
    grads, missing = zero_filled(g_raw, params)
    del g_raw
    MG.check(device, "theta0_checks: R1 gradients")
    # directions one parameter at a time on the device, then to the HOST (host fact: no direction sets on the GPU)
    u = {}
    for n in names:
        u[n] = G0.muon_direction({n: params[n]}, {n: grads[n]}, momentum=momentum, nesterov=True)[n].to("cpu")
    # 1. SGD scale c
    u_fro = [float(_slice(u[n], r).double().norm()) for n, r in mats]           # host tensors
    g_fro = [float(_slice(grads[n], r).double().norm()) for n, r in mats]
    mean_u, mean_g = sum(u_fro) / len(mats), sum(g_fro) / len(mats)
    c = lr_muon * mean_u / ((1 + momentum) * mean_g) if mean_g > 0 else float("nan")
    tot_u = math.sqrt(sum(x * x for x in u_fro))
    tot_g = math.sqrt(sum(x * x for x in g_fro))
    sgd = {"c": c, "grid": [c / 3, c, 3 * c], "lr_muon": lr_muon, "momentum": momentum, "n_matrices": len(mats),
           "mean_u_fro": mean_u, "mean_g_fro": mean_g,
           "mean_first_step_norm_muon": lr_muon * mean_u, "mean_first_step_norm_sgd_at_c": c * (1 + momentum) * mean_g,
           "c_total_norm_matching": lr_muon * tot_u / ((1 + momentum) * tot_g) if tot_g > 0 else float("nan"),
           "zero_gradient_matrices": int(sum(1 for x in g_fro if x == 0.0)),
           "rule": "c = lr_muon mean_m ||u_m||_F / ((1+mu) mean_m ||g_m||_F) over R1's expert matrices; grid [c/3, c, 3c]"}
    # 2. NS-5 scale check on the optimizer's NS5 input D = (1 + mu) g
    rho: dict[str, list[float]] = {f"{s:g}": [] for s in NS5_SCALES}
    for n in names:
        D = grads[n] * (1 + momentum)
        base = ns5(D).float()
        rows_n = [r for nn, r in mats if nn == n]
        for s in NS5_SCALES:
            o = ns5(D * s).float()
            for r in rows_n:
                b = _slice(base, r)
                den = float(b.double().norm())
                if den > 0:
                    rho[f"{s:g}"].append(float((_slice(o, r) - b).double().norm()) / den)
    ns = {}
    for s, v in rho.items():
        v = sorted(v)
        ns[s] = {"n": len(v), "median": v[len(v) // 2] if v else None, "max": v[-1] if v else None}
    out = {"sgd": sgd, "ns5_scale_check": {"scales": list(NS5_SCALES), "per_scale": ns,
                                          "definition": "||NS5(s D) - NS5(D)||_F / ||NS5(D)||_F per R1 matrix"},
           "missing_gradients": missing, "windows": windows, "units": len(unit_keys)}
    # 3. D_Muon unit test: one step of the authors' optimizer
    if d_muon:
        from nestor.configs.optim import MuonAdamWConfig
        from nestor.train.optim import build_optimizer

        opt = build_optimizer(MuonAdamWConfig(lr_muon=lr_muon, lr_adamw=7.5e-5, weight_decay=0.0, betas=(0.9, 0.95)), model)
        before = {n: params[n].detach().to("cpu", copy=True) for n in names}
        for p in model.parameters():
            p.grad = None
        MG.reserve(device, MG.tensor_bytes(params[n] for n in names), "theta0_checks: D_Muon optimizer state")
        for n in names:
            params[n].grad = grads[n].to(params[n].dtype)
        del grads
        opt.step()
        MG.check(device, "theta0_checks: D_Muon step")
        errs = []
        for n, r in mats:
            step = (_slice(params[n].detach().to("cpu"), r).float() - _slice(before[n], r).float()) / (-lr_muon)
            ur = _slice(u[n], r).float()
            den = float(ur.double().norm())
            if den > 0:
                errs.append(float((step - ur).double().norm()) / den)
        errs.sort()
        with torch.no_grad():
            for n in names:
                params[n].copy_(before[n].to(params[n].device))
                params[n].grad = None
        del opt
        out["d_muon_unit_test"] = {"n": len(errs), "median_rel_err": errs[len(errs) // 2] if errs else None,
                                   "max_rel_err": errs[-1] if errs else None, "tol": D_MUON_TOL,
                                   "pass": bool(errs) and errs[-1] <= D_MUON_TOL,
                                   "definition": "theta_1 - theta_0 = -lr_muon u_m (weight decay 0), per R1 matrix"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="theta_0 checks on R1: SGD scale c, NS-5 scale check, D_Muon unit test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment"); r.add_argument("checkpoint")
    r.add_argument("--draw", required=True, help="A/A draw JSON (crossover.reg_aa_draw entry)")
    r.add_argument("--subset", default="A"); r.add_argument("--data", required=True)
    r.add_argument("--windows", type=int, default=64); r.add_argument("--data-seed", type=int, default=0)
    r.add_argument("--block", type=int, default=4096); r.add_argument("--lr-muon", type=float, default=0.0025)
    r.add_argument("--momentum", type=float, default=0.95); r.add_argument("--device", default=None)
    r.add_argument("--no-bf16", action="store_true"); r.add_argument("--register", default=None, help=argparse.SUPPRESS)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch

    import eval_windows as EW

    t0 = time.time()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    EW.register_extra(a.register)
    draw = json.load(open(a.draw, encoding="utf-8"))
    keys = draw["subsets"][a.subset]
    model, _ = EW.load_model(a.experiment, a.checkpoint, device)
    ws = EW.WindowSet("theta0", a.data, a.block, a.windows, "random", a.data_seed)
    res = checks(model, ws, keys, device=device, windows=a.windows, lr_muon=a.lr_muon, momentum=a.momentum,
                 bf16=not a.no_bf16)
    res.update(experiment=a.experiment, checkpoint=EW.file_fingerprint(a.checkpoint), data=ws.describe(),
               draw={"file": os.path.abspath(a.draw), "subset": a.subset, "seed": draw.get("seed"), "units": len(keys)},
               device=device, seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    os.replace(a.out + ".tmp", a.out)
    print(json.dumps({"sgd_c": res["sgd"]["c"], "grid": res["sgd"]["grid"], "ns5": res["ns5_scale_check"]["per_scale"],
                      "d_muon": res.get("d_muon_unit_test"), "seconds": res["seconds"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
