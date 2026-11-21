"""G0-7 margin-calibration probes (review v2 sec. 2.1).

For every MoE base: the registered C1 draws (crossover.reg_draw: seed 5000 + d, f, tolerance and code stream from the
G0 gate record) and, for each draw side S in {A, B}, the KL quadratic form along a PINNED norm-matched random direction
r_S = g0_probe.random_like_direction(u_S) (u_S = the cell target's first Muon step restricted to S; same per-matrix
singular values, Haar-random singular vectors) on the code and math ESTIMATION windows of the G0 routing files:
    Q_c(S) = mean over windows of corpus c of (KL(+eps) + KL(-eps)) / (2 eps^2)
    I_probe,d = [log Q_code(B) - log Q_code(A)] - [log Q_math(B) - log Q_math(A)]
g0_gate.py turns (I_probe,d, P_d) into D_b = mean_d (I_probe,d - P_d) / P-bar_b, SE and beta0_b (gate G0-7).

  python g0_margin.py plan     --g0-dir g0/moe_b1 --gate-record g0/gate_record.json [--draws 5]   # no model
  python g0_margin.py estimate --g0-dir g0/moe_b1 [--probe-windows 24] [--tok-per-s 9600]
  python g0_margin.py run      --g0-dir g0/moe_b1 [--probe-windows 24] [--device cuda]          # GPU
Outputs in --g0-dir: margin__manifest.json (+ .sha256), margin__windows.npz, margin__probe.json.
Model and checkpoint are read from the G0 routing files' metadata (the checkpoint G0 used), not from a glob.
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
if HERE not in sys.path:                      # PYTHONSAFEPATH=1 in the kernel
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import crossover as CX  # noqa: E402

MANIFEST, WINDOWS, PROBE = "margin__manifest.json", "margin__windows.npz", "margin__probe.json"
GRAD_WINDOWS, GRAD_SEED, EPS0, REGIME_WINDOWS = 16, 12345, 0.0025, 8      # g0_probe G0 defaults


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _cells_from_record(rec: dict, base: str) -> dict:
    cfg = rec["config"]["resolved"]
    arch = cfg["bases"][base]["arch"]
    if arch != "moe":
        raise SystemExit(f"G0-7 is defined per MoE base; {base} is {arch}")
    code = rec["decisions"]["code_stream"]["chosen"]
    db = cfg["decision_base"]["moe"]
    out = {}
    for tgt in cfg["c1_targets"]["moe"]:
        c = rec["gates"]["G0-1"]["cells"].get(f"{db}|{tgt}|{code}")
        ok = rec["gates"]["G0-1"]["moe_targets"].get(tgt)
        if c and c.get("feasible") and ok:
            out[tgt] = {"f": c["f"], "tol": c["tol"], "decision_cell": f"{db}|{tgt}|{code}", "median_P": c["median"]}
    return {"arch": arch, "code": code, "targets": out, "half_seed": cfg.get("half_seed", 0), "layout": cfg.get("layout", "auto")}


def plan(g0_dir: str, gate_record: str, *, base: str | None = None, draws: int = 5) -> dict:
    """Registered C1 draws d = 1..draws per passing target (a failing target's draws go to the other target, as in the
    G0-1 action 'cells replaced by the other target, draws per base unchanged'); hashed manifest."""

    rec = json.load(open(gate_record, encoding="utf-8"))
    base = base or os.path.basename(os.path.normpath(g0_dir))
    cells = _cells_from_record(rec, base)
    tg = sorted(cells["targets"])
    if not tg:
        raise SystemExit("no MoE C1 target passed G0-1: G0-7 is not needed (PIVOT)")
    n_total = draws * len(rec["config"]["resolved"]["c1_targets"]["moe"])
    per = {t: n_total // len(tg) for t in tg}
    t = CX.reg_load_dir(g0_dir, "moe", layout=cells["layout"], half_seed=int(cells["half_seed"]))
    entries = []
    for tgt in tg:
        c = cells["targets"][tgt]
        for d in range(1, per[tgt] + 1):
            e = CX.reg_draw(t, "c1", d, f=c["f"], target=tgt, code=cells["code"], tol=c["tol"])
            e["decision_cell"] = c["decision_cell"]
            entries.append(e)
    path = os.path.join(g0_dir, MANIFEST)
    h = CX.write_manifest(path, entries, extra={"purpose": "G0-7 margin calibration", "base": base,
                                                "code_stream": cells["code"], "gate_record_sha256": _sha(gate_record),
                                                "draws_per_target": per})
    bad = [f"{e['target']}|d{e['d']}" for e in entries if e.get("status") != "ok"]
    return {"manifest": path, "sha256": h, "draws": len(entries), "infeasible": bad, "per_target": per, "code": cells["code"]}


def _load_manifest(g0_dir: str) -> tuple[dict, str]:
    path = os.path.join(g0_dir, MANIFEST)
    txt = open(path, encoding="utf-8").read()
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    rec_h = open(path + ".sha256", encoding="utf-8").read().split()[0]
    if h != rec_h:
        raise SystemExit(f"{path}: content sha256 {h} != recorded {rec_h} (manifest changed after hashing)")
    return json.loads(txt), h


def estimation_windows(g0_dir: str, corpus: str, n: int, *, half_seed: int = 0, check: bool = True):
    """The first n ESTIMATION windows of a corpus (registered layout: routing__<c>_eval.npz; crossfit layout: half 1
    of routing__<c>.npz) as an eval_windows.WindowSet over the original bin; hashes verified against the G0 file."""

    import eval_windows as EW

    f_est = os.path.join(g0_dir, f"routing__{corpus}_eval.npz")
    if os.path.exists(f_est):
        res = EW.load_result(f_est)
        pos = np.arange(len(res["index"]))
    else:
        res = EW.load_result(os.path.join(g0_dir, f"routing__{corpus}.npz"))
        pos = np.flatnonzero(EW.split_halves(len(res["index"]), seed=half_seed) == 1)
    pos = pos[:n]
    st = res["meta"]["set"]
    ws = EW.WindowSet(corpus, st["path"], int(st["block"]), len(pos), indices=res["index"][pos])
    if check and not np.array_equal(ws.hashes, res["hash"][pos].astype(np.uint64)):
        raise SystemExit(f"{corpus}: window tokens differ from the G0 routing file (data file changed?)")
    return ws


def estimate(g0_dir: str, *, probe_windows: int = 24, tok_per_s: float = 9600.0, block: int = 4096) -> dict:
    man, _ = _load_manifest(g0_dir)
    ents = [e for e in man["entries"] if e.get("status") == "ok"]
    tg = sorted({e["target"] for e in ents})
    n_dir = 2 * len(ents)                                         # A and B per draw
    win = 2 * probe_windows                                       # code + math
    fwd = win * (len(tg) + 2 * n_dir)                             # base forward per target batch + (+/-) per direction
    eps_fwd = len(tg) * 2 * 4 * REGIME_WINDOWS                    # ~4 halvings x 2 forwards
    grad_fwd = len(tg) * GRAD_WINDOWS * 3                         # backward ~ 2 forwards
    tok = (fwd + eps_fwd + grad_fwd) * block
    return {"draws": len(ents), "directions": n_dir, "probe_windows_per_corpus": probe_windows, "forwards": fwd + eps_fwd + grad_fwd,
            "tokens": tok, "gpu_hours": round(tok / tok_per_s / 3600, 3), "assumption": f"{tok_per_s:.0f} tok/s fp32 forwards"}


def run(g0_dir: str, *, probe_windows: int = 24, device: str | None = None, model=None, experiment: str | None = None,
        checkpoint: str | None = None, grad_paths: dict | None = None, register: str | None = None, log=print) -> dict:
    import torch

    import eval_windows as EW
    import g0_probe as G0

    t0 = time.time()
    man, mh = _load_manifest(g0_dir)
    ents = [e for e in man["entries"] if e.get("status") == "ok"]
    code = man["code_stream"]
    meta = EW.load_result(os.path.join(g0_dir, f"routing__{code}.npz"))["meta"]
    experiment = experiment or meta.get("experiment")
    checkpoint = checkpoint or meta.get("checkpoint")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        EW.register_extra(register)
        model, _ = EW.load_model(experiment, checkpoint, device)
    half_seed = int(next(iter(ents))["sources"][code].get("half_seed", 0)) if ents else 0
    windows = []
    for c in (code, "math"):
        ws = estimation_windows(g0_dir, c, probe_windows, half_seed=half_seed)
        windows += [(ws, i) for i in range(len(ws.index))]
    corpus_of = np.array([("code" if ws.name == code else "math") for ws, _ in windows])
    rows, params = G0.unit_rows(model), G0.expert_params(model)
    out_arr = {"corpus": corpus_of.astype(str), "window_index": np.array([int(ws.index[i]) for ws, i in windows], dtype=np.int64),
               "window_hash": np.array([int(ws.hashes[i]) for ws, i in windows], dtype=np.uint64)}
    eps_by, draws = {}, []
    for tgt in sorted({e["target"] for e in ents}):
        tsrc = (grad_paths or {}).get(tgt) or EW.load_result(os.path.join(g0_dir, f"routing__{tgt}.npz"))["meta"]["set"]["path"]
        gws = EW.WindowSet(tgt + "_grad", tsrc, int(meta["set"]["block"]), GRAD_WINDOWS, "random", GRAD_SEED)
        grads = G0.target_gradient(model, gws, params, GRAD_WINDOWS, device=device, bf16=True)
        # muon_scale lives on the Parameter objects (get_scaling), so NS5 + scaling run on the real params' device;
        # the directions are then kept on the CPU (Perturber moves one at a time to the model's device)
        u_e = G0.muon_direction(params, grads, momentum=0.95, nesterov=True)
        u_e = {k: v.detach().cpu() for k, v in u_e.items()}
        del grads
        ec = G0.choose_eps(model, windows[:REGIME_WINDOWS], u_e, device=device, eps0=EPS0, log=None, pin_routing=True)
        eps = ec["eps"]
        eps_by[tgt] = ec
        plan, labs = [], []
        for e in [x for x in ents if x["target"] == tgt]:
            for side in ("A", "B"):
                u_s = G0.restrict(u_e, rows, e["subsets"][side])
                r_s = G0.random_like_direction(u_s, seed=1000 * int(e["d"]) + (0 if side == "A" else 1))
                lab = f"{tgt}|d{e['d']}|{side}"
                plan += [(lab, r_s, eps, +1), (lab, r_s, eps, -1)]
                labs.append(lab)
        res = G0.directional_probe(model, windows, plan, device=device, log=log, pin_routing=True, count_flips=False)
        for key, v in res.items():
            for fld in ("kl", "dce"):
                out_arr[f"{key}|{fld}"] = np.asarray(v[fld], dtype=np.float64)
        for e in [x for x in ents if x["target"] == tgt]:
            q = {}
            for c in ("code", "math"):
                msk = corpus_of == c
                q[c] = {}
                for side in ("A", "B"):
                    k = f"{tgt}|d{e['d']}|{side}|{eps:.6g}"
                    qw = (res[k + "|+1"]["kl"][msk] + res[k + "|-1"]["kl"][msk]) / (2 * eps * eps)
                    q[c][side] = {"value": float(qw.mean()), "se": float(qw.std(ddof=1) / math.sqrt(len(qw))) if len(qw) > 1 else None,
                                  "n": int(len(qw))}
            ip = (math.log(q["code"]["B"]["value"]) - math.log(q["code"]["A"]["value"])) - \
                 (math.log(q["math"]["B"]["value"]) - math.log(q["math"]["A"]["value"]))
            draws.append({"target": tgt, "d": int(e["d"]), "seed": e["seed"], "P": e["predicted"]["P"], "Q": q, "I_probe": ip})
        if log:
            log(f"[g0-7] {tgt}: {len(labs)} directions probed, eps {eps:g}, {time.time() - t0:.0f}s")
    out_arr["eps_json"] = np.array(json.dumps({k: v["eps"] for k, v in eps_by.items()}))
    np.savez_compressed(os.path.join(g0_dir, WINDOWS + ".tmp.npz"), **out_arr)
    os.replace(os.path.join(g0_dir, WINDOWS + ".tmp.npz"), os.path.join(g0_dir, WINDOWS))
    rep = {"purpose": "G0-7 margin calibration", "base": man.get("base"), "manifest_sha256": mh,
           "code_stream": code, "experiment": experiment, "checkpoint": checkpoint, "routing_mode": "pinned",
           "direction": "norm-matched random (g0_probe.random_like_direction of the restricted Muon step)",
           "probe_windows_per_corpus": probe_windows, "eps": {k: v["eps"] for k, v in eps_by.items()},
           "eps_check": eps_by, "draws": draws, "windows_file": WINDOWS, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(g0_dir, PROBE), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=float)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G0-7 margin-calibration probes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--g0-dir", required=True); p.add_argument("--gate-record", required=True)
    p.add_argument("--base", default=None); p.add_argument("--draws", type=int, default=5)
    e = sub.add_parser("estimate"); e.add_argument("--g0-dir", required=True); e.add_argument("--probe-windows", type=int, default=24)
    e.add_argument("--tok-per-s", type=float, default=9600.0)
    r = sub.add_parser("run"); r.add_argument("--g0-dir", required=True); r.add_argument("--probe-windows", type=int, default=24)
    r.add_argument("--device", default=None); r.add_argument("--experiment", default=None); r.add_argument("--checkpoint", default=None)
    r.add_argument("--grad-path", action="append", default=[], help="target=path.bin (default: the G0 routing stream of the target)")
    r.add_argument("--register", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        out = plan(a.g0_dir, a.gate_record, base=a.base, draws=a.draws)
    elif a.cmd == "estimate":
        out = estimate(a.g0_dir, probe_windows=a.probe_windows, tok_per_s=a.tok_per_s)
    else:
        gp = dict(x.split("=", 1) for x in a.grad_path)
        rep = run(a.g0_dir, probe_windows=a.probe_windows, device=a.device, experiment=a.experiment, checkpoint=a.checkpoint,
                  grad_paths=gp, register=a.register)
        out = {k: rep[k] for k in ("base", "manifest_sha256", "eps", "seconds")}
        out["I_probe"] = {f"{d['target']}|d{d['d']}": round(d["I_probe"], 4) for d in rep["draws"]}
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
