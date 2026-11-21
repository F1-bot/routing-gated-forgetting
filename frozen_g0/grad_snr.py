"""Half-batch per-expert gradient signal-to-noise ratio (measured at G2).

A batch of 2m windows is split into two fixed halves (seeded permutation) with mean training-loss gradients g1, g2
(g0_probe.target_gradient: authors' loss, routers frozen; bf16 autocast as in training). Per expert unit e (subset_adapt
unit keys; all rows of every matrix of that unit):
    gbar_e   = (g1_e + g2_e) / 2                    the full-batch mean gradient
    noise_e  = ||g1_e - g2_e||^2 / 4                unbiased for tr Var(gbar_e)   (Var(g1 - g2) = 4 Var(gbar))
    signal_e = ||gbar_e||^2 - noise_e               unbiased for ||E g_e||^2 (can be negative when noise dominates)
    snr_e    = signal_e / noise_e                   at the estimation batch of 2m windows
    snr_train_e = snr_e * B_train / (2m)            at the training batch (noise of a mean scales as 1 / batch)
The summary reports medians/quartiles per unit kind and the share of units with snr_train < 1.

  python grad_snr.py run <experiment> <checkpoint> --data data/domain/fw2de_train.bin --windows 64 \
      [--train-batch-windows 64] [--kinds moe] [--seed 0] [--device cuda] --out g2/grad_snr_<tag>.json
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

import numpy as np  # noqa: E402


def unit_sq_norms(grads: dict, rows: dict[str, list], units: list[str] | None = None) -> dict[str, float]:
    """sum over the unit's (parameter, row) entries of ||g[param][row]||^2 (row None = the whole tensor)."""

    import torch

    cache: dict[str, torch.Tensor] = {}
    out = {}
    for u in (units if units is not None else sorted(rows)):
        tot = 0.0
        for name, r in rows.get(u, []):
            g = grads.get(name)
            if g is None:
                continue
            if r is None:
                tot += float(g.double().pow(2).sum())
            else:
                if name not in cache:
                    cache[name] = g.double().pow(2).reshape(g.shape[0], -1).sum(1)
                tot += float(cache[name][r])
        out[u] = tot
    return out


def half_batch_snr(model, wset, params: dict, rows: dict, *, device: str, n_windows: int | None = None, seed: int = 0,
                   train_batch_windows: int | None = None, kinds=("moe", "pub", "priv"), bf16: bool = True) -> dict:
    import eval_windows as EW
    import g0_probe as G0

    n = len(wset.index) if n_windows is None else min(int(n_windows), len(wset.index))
    m = n // 2
    if m < 1:
        raise ValueError("need at least 2 windows")
    perm = np.random.default_rng(seed).permutation(n)
    halves = [np.sort(perm[:m]), np.sort(perm[m:2 * m])]
    gs = []
    for h in halves:
        ws = EW.WindowSet(wset.name + f"_h{len(gs)}", wset.path, wset.block, m, indices=wset.index[h])
        g = G0.target_gradient(model, ws, params, m, device=device, bf16=bf16)
        gs.append({k: v.detach().double().cpu() for k, v in g.items()})
    units = sorted(u for u in rows if u.split(":")[0] in kinds)
    gbar = {k: (gs[0][k] + gs[1][k]) / 2 for k in gs[0]}
    diff = {k: gs[0][k] - gs[1][k] for k in gs[0]}
    sq_bar, sq_diff = unit_sq_norms(gbar, rows, units), unit_sq_norms(diff, rows, units)
    scale = (train_batch_windows or 2 * m) / (2 * m)
    per = {}
    for u in units:
        noise = sq_diff[u] / 4
        signal = sq_bar[u] - noise
        snr = signal / noise if noise > 0 else (math.inf if signal > 0 else float("nan"))
        per[u] = {"gbar_sq": sq_bar[u], "noise": noise, "signal": signal, "snr": snr, "snr_train": snr * scale}
    summ = {}
    for kind in sorted({u.split(":")[0] for u in units}):
        v = np.array([per[u]["snr_train"] for u in units if u.startswith(kind + ":")], dtype=np.float64)
        f = v[np.isfinite(v)]
        summ[kind] = {"units": int(len(v)), "median_snr_train": float(np.median(f)) if f.size else None,
                      "q25": float(np.quantile(f, 0.25)) if f.size else None, "q75": float(np.quantile(f, 0.75)) if f.size else None,
                      "share_snr_train_below_1": float((f < 1).mean()) if f.size else None}
    return {"windows": 2 * m, "half_windows": m, "seed": seed, "train_batch_windows": train_batch_windows or 2 * m,
            "halves": [h.tolist() for h in halves], "per_unit": per, "summary": summ}


def main(argv=None) -> int:
    import torch

    import eval_windows as EW
    import g0_probe as G0

    ap = argparse.ArgumentParser(description="half-batch per-expert gradient SNR (G2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment"); r.add_argument("checkpoint")
    r.add_argument("--data", required=True); r.add_argument("--windows", type=int, default=64)
    r.add_argument("--select", default="random", choices=["first", "even", "random"]); r.add_argument("--data-seed", type=int, default=7)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--train-batch-windows", type=int, default=64)
    r.add_argument("--kinds", default="moe,pub,priv"); r.add_argument("--block", type=int, default=4096)
    r.add_argument("--device", default=None); r.add_argument("--out", required=True)
    r.add_argument("--register", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    t0 = time.time()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    EW.register_extra(a.register)
    model, _ = EW.load_model(a.experiment, a.checkpoint, device)
    ws = EW.WindowSet("snr", a.data, a.block, a.windows, a.select, a.data_seed)
    res = half_batch_snr(model, ws, G0.expert_params(model), G0.unit_rows(model), device=device, seed=a.seed,
                         train_batch_windows=a.train_batch_windows, kinds=tuple(k for k in a.kinds.split(",") if k))
    res.update(experiment=a.experiment, checkpoint=EW.file_fingerprint(a.checkpoint), data=ws.describe(),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"summary": res["summary"], "seconds": res["seconds"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
