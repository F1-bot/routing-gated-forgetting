"""MOCK G0 gate record on SIMULATED tables (produced before any real G0 statistic is read).

Builds a complete simulated G0 tree in the layout the chain produces with the phase-3 g0_probe (one routing stream
per corpus, cross-fit halves; probe JSONs without per-window arrays), the design-v4 G0-5 inputs and the G0-7 margin probes,
then runs the real gate pipeline:  g0_gate.py -> g0_margin.py plan -> (simulated probes) -> g0_gate.py.
Every number in the output is SIMULATED (power_sim_v1 worlds); nothing is measured.

  python mock_g0.py --out g0_mock [--draws 1000] [--seed 20261101]
Needs power_sim_v1.py next to this file or in $POWER_SIM_V1. Torch-free.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import eval_windows as EW  # noqa: E402
import g0_gate as GG  # noqa: E402
import g0_margin as GM  # noqa: E402

CORP = ("dclm", "code", "math", "german", "books")


def _ps():
    p = os.environ.get("POWER_SIM_V1", os.path.join(HERE, "power_sim_v1.py"))
    spec = importlib.util.spec_from_file_location("power_sim_v1", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _routing(path, per_window: list[dict], set_path: str, block=4096, columns=None, ckpt="MOCK.pt", exp="MOCK"):
    n = len(per_window)
    rt = {}
    for lk in per_window[0]:
        arr = np.zeros((n, 3, per_window[0][lk].shape[0]))
        for i, tb in enumerate(per_window):
            arr[i, 0] = 1.0
            arr[i, 1] = np.sqrt(tb[lk]) * block
            arr[i, 2] = tb[lk] * block
        rt[lk] = arr
    for b, (kc, c) in (columns or {}).items():
        col = np.zeros((n, 3, c))
        col[:, 0] = kc * block / c
        rt[f"column:b{b}"] = col
    res = {"index": np.arange(n), "starts": np.arange(n) * block, "hash": np.arange(n, dtype=np.uint64) + 11,
           "ce_sum": np.full(n, 4.0 * (block - 1)), "n_tok": np.full(n, block - 1.0), "routing": rt}
    EW.save_result(path, res, {"variant": "base", "set": {"name": os.path.basename(path), "path": set_path, "block": block},
                               "checkpoint": ckpt, "experiment": exp, "MOCK": True})


def build_base(PS, d, arch, lam, seed, rng, n_win=16):
    w = PS.make_world(np.random.default_rng(seed), arch, lam, 0.4, 0.5)
    sel = {c: w["w_sel"][..., PS.CI[c]] for c in CORP}
    est = {c: w["w_est"][..., PS.CI[c]] for c in CORP}
    for tb, noise in ((sel, 0.15), (est, 0.15)):                    # simulated extra streams
        fr = tb["german"] * np.exp(noise * rng.standard_normal(tb["german"].shape))
        tb["french"] = fr / fr.sum(1, keepdims=True)
        py = tb["code"] * np.exp(0.3 * rng.standard_normal(tb["code"].shape))
        tb["pycode"] = py / py.sum(1, keepdims=True)
    L, N = w["L"], w["N"]
    if arch == "moe":
        lks, cols = [f"moe:b0:l{l}" for l in range(L)], None
    else:
        lks = [f"priv:b{l}:l0" for l in range(L)]
        cols = {l: ((8 if l == L - 1 else 3), 8) for l in range(L)}
    half = EW.split_halves(2 * n_win, seed=0)
    os.makedirs(d, exist_ok=True)
    for c in sel:
        wins = [{lk: (sel if h == 0 else est)[c][l] for l, lk in enumerate(lks)} for h in half]
        sp = "data/dclm/dclm_val10.bin" if c == "dclm" else f"data/streams/{c}_route.bin"
        _routing(os.path.join(d, f"routing__{c}.npz"), wins, sp, columns=cols, ckpt=f"MOCK/{os.path.basename(d)}.pt")
    # phase-3 probe JSON (means/SEs only, pinned + free), German target, 8 windows per old corpus
    labs = ["E_ALL"] + [f"C1_german_d{i}_{s}" for i in range(3) for s in "AB"] + [f"C2_german_d{i}_{s}" for i in range(3) for s in "AB"]
    subsets = {}
    for j, lab in enumerate(labs):
        ent = {}
        for mode in ("pinned", "free"):
            bc = {}
            for c, lev in (("dclm", 0.0), ("code", 0.6), ("math", 1.1)):
                q = np.exp(0.5 * j / len(labs) + lev - (0.0 if lab == "E_ALL" else 1.5) + 0.25 * rng.standard_normal(8))
                bc[c] = {"Q_kl": {"value": float(q.mean()), "se": float(q.std(ddof=1) / math.sqrt(8)), "n": 8}}
            ent[mode] = {"all": bc["dclm"], "by_corpus": bc}
        subsets[lab] = ent
    json.dump({"target": "german", "eps": 0.0025, "primary_routing": "pinned", "routing_modes": ["pinned", "free"],
               "probe_windows": 24, "corpora": ["code", "dclm", "math"], "subsets": subsets, "MOCK": True},
              open(os.path.join(d, "probe__german.json"), "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="g0_mock")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args(argv)
    PS = _ps()
    rng = np.random.default_rng(a.seed)
    root = os.path.abspath(a.out)
    g0 = os.path.join(root, "g0")
    build_base(PS, os.path.join(g0, "moe_b1"), "moe", 0.5, a.seed + 1, rng)
    build_base(PS, os.path.join(g0, "moe_b2"), "moe", 0.5, a.seed + 2, rng)
    build_base(PS, os.path.join(g0, "nestor"), "nestor", 0.5, a.seed + 3, rng)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    os.makedirs(os.path.join(g0, "xplat"), exist_ok=True)
    with open(os.path.join(root, "results", "continual.jsonl"), "w") as fh:
        fh.write(json.dumps({"key": "A|nestor_100m|val09", "stage": "pretrain_eval", "model": "nestor_100m", "set": "val09",
                             "ce": 4.784113 + 3e-4, "tokens": 10004085, "ckpt": "MOCK/nestor.pt", "MOCK": True}) + "\n")
    with open(os.path.join(root, "logs", "g0_xplat_local_moe_b2_mean.log"), "w") as fh:
        fh.write("MOCK\nRESULT " + json.dumps({"ce": 4.635177 - 6e-4, "tokens": 10004085, "ckpt": "MOCK/moe_b2.pt"}) + "\n")
    ce = 4.78 + 0.3 * rng.standard_normal(2443)
    for side, c in (("local", ce), ("h100", ce + 2e-4 * rng.standard_normal(2443))):
        n = len(c)
        EW.save_result(os.path.join(g0, "xplat", f"{side}_nestor.npz"),
                       {"index": np.arange(n), "starts": np.arange(n) * 4096, "hash": np.arange(n, dtype=np.uint64) + 5,
                        "n_tok": np.full(n, 4095.0), "ce_sum": c * 4095, "routing": {}}, {"set": {"block": 4096}, "MOCK": True})
    cfg = {"_comment": "MOCK gate config on SIMULATED tables -- not a G0 measurement",
           "bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe", "base": 1, "k": 6},
                     "moe_b2": {"dir": "moe_b2", "arch": "moe", "base": 2, "k": 6},
                     "nestor": {"dir": "nestor", "arch": "nestor", "base": 1, "k": 16}},
           "decision_base": {"moe": "moe_b1", "nestor": "nestor"},
           "c1_targets": {"moe": ["german", "books"], "nestor": ["german"]}, "code_streams": ["code", "pycode"],
           "c2": {"base": "moe_b1", "target": "german"}, "dclm_tilt": {"base": "moe_b1", "match_target": "german"},
           "g0_5": {"bases": {"nestor": {"h100_ce": 4.784113, "local_jsonl": "../results/continual.jsonl",
                                         "jsonl_key": "A|nestor_100m|val09"},
                              "moe_b2": {"h100_ce": 4.635177, "local_result_log": "../logs/g0_xplat_local_moe_b2_mean.log"}},
                    "per_window_dir": "xplat"},
           "layout": "auto", "half_seed": 0}
    cp = os.path.join(g0, "g0_gate_config.json")
    json.dump(cfg, open(cp, "w"), indent=1)
    rec_p = os.path.join(g0, "gate_record.json")
    GG.main(["--config", cp, "--out", rec_p, "--draws", str(a.draws)])
    rec = json.load(open(rec_p))
    for b in ("moe_b1", "moe_b2"):
        if not rec["gates"]["G0-1"].get("moe_pass"):
            break
        d = os.path.join(g0, b)
        GM.plan(d, rec_p, draws=5)
        man, mh = GM._load_manifest(d)
        draws = []
        pool = 0.05 * rng.standard_normal()                       # simulated base-level pool effect
        for e in man["entries"]:
            if e.get("status") != "ok":
                continue
            P = e["predicted"]["P"]
            draws.append({"target": e["target"], "d": e["d"], "seed": e["seed"], "P": P,
                          "I_probe": 1.02 * P + pool + 0.08 * rng.standard_normal()})
        json.dump({"MOCK": True, "routing_mode": "pinned", "code_stream": man["code_stream"], "manifest_sha256": mh,
                   "draws": draws}, open(os.path.join(d, GM.PROBE), "w"), indent=1)
    GG.main(["--config", cp, "--out", rec_p, "--draws", str(a.draws)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
