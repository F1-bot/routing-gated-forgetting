r"""Real-configuration check of the WD-only control (post-freeze amendment A-1). Read-only on a finished run:

    python phase6\wd_only_check.py --root . --run c1_moe_b1_books_d2_A --device cuda --out <file.json>

Builds the decay-only state with weight_controls.wd_only_state (the phase6\ copy next to this script, i.e. the staged
version, before chain6b.ps1 -Reinstall) from the run's run_env.json, subset_manifest.json, train log and frozen
queue job, on --device, and checks: the schedule (max_steps, warm-up, executed) and its sources; the manifest
fingerprint and the train-log learning rates (inside wd_only_state); every changed element of every optimizer tensor
equals theta_0 * prod_t (1 - lr_g,t * wd_g) of its group, with lr_g,t taken from the TRAINING's own train log (step 1 uses
the warm-up factor 0; step k+1 the LR logged after step k), to 1e-5 relative; every other element and every other
tensor equals theta_0 exactly; the decayed elements are exactly the manifest's trainable ones. Nothing of the run is written (the run's output paths are redirected to a temp dir).
Prints the time and the peak VRAM. Exit 0 iff every check passes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", default="c1_moe_b1_books_d2_A")
    ap.add_argument("--attempt", default="m2", help="attempt directory under outputs/confirm/<run>")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)

    def norm(p):
        return os.path.normcase(os.path.abspath(p or "."))

    # The script's own directory (sys.path[0] when run as a script) must not be a search path: every other module comes
    # from the root as at run time; only weight_controls is taken explicitly from the file next to this script.
    sys.path[:] = [p for p in sys.path if norm(p) != norm(HERE)]
    for p in (os.path.join(root, "repo"), root):
        if norm(p) not in {norm(x) for x in sys.path}:
            sys.path.insert(0, p)
    import hashlib
    import importlib.util

    import torch

    wc_path = os.path.join(HERE, "weight_controls.py")
    spec = importlib.util.spec_from_file_location("weight_controls", wc_path)
    WC = importlib.util.module_from_spec(spec)
    sys.modules["weight_controls"] = WC
    spec.loader.exec_module(WC)
    import eval_windows as EW
    import memguard as MG

    out = {"weight_controls": WC.__file__, "weight_controls_sha256": hashlib.sha256(open(wc_path, "rb").read()).hexdigest(),
           "eval_windows": EW.__file__, "memguard": MG.__file__, "device": a.device, "run": a.run}
    run = os.path.join(root, "outputs", "confirm", a.run, a.attempt)
    run_env = json.load(open(os.path.join(run, "run_env.json"), encoding="utf-8"))
    man_p = os.path.join(run, "subset_manifest.json")
    man = json.load(open(man_p, encoding="utf-8"))
    tmp = tempfile.mkdtemp(prefix="wd_only_check_")
    run_env_v = dict(run_env, NESTOR_OUTPUT_ROOT=os.path.join(tmp, "out"), NESTOR_TRAIN_LOG_DIR=os.path.join(tmp, "tl"))
    q = json.load(open(os.path.join(root, json.load(open(os.path.join(root, "FREEZE.json"), encoding="utf-8-sig"))["queue"]["path"]),
                       encoding="utf-8"))
    job = next(j for j in q["jobs"] if j["id"] == a.run)
    try:
        WC.wd_only_schedule(run_env, None)
        out["missing_steps_raises"] = False
    except ValueError as e:
        out["missing_steps_raises"] = str(e)[:200]
    out["schedule"] = WC.wd_only_schedule(run_env, man_p)
    os.chdir(root)
    EW.nestor_env()
    t0 = time.time()
    bm, _ = EW.load_model(job["experiment"], os.path.join(root, run_env["DCLM_INIT_CHECKPOINT"]), "cpu")
    base = MG.host_copy(bm.state_dict())
    del bm
    out["load_theta0_s"] = round(time.time() - t0, 1)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rep = {}
    st = WC.wd_only_state(job["experiment"], base, man_p, run_env_v, device=a.device, report=rep)
    out["report"] = rep
    if a.device == "cuda":
        out["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    log = [json.loads(x) for x in open(os.path.join(run, "trainlog", "train_steps.jsonl"), encoding="utf-8").read().splitlines() if x.strip()]
    groups = man["optimizer_groups"]
    init = sorted(g["initial_lr"] for g in groups)
    executed = rep["executed"]
    prods = []
    for gi, g in enumerate(sorted(groups, key=lambda g: g["initial_lr"])):
        pr = 1.0
        for r in log[:executed - 1]:
            pr *= 1.0 - float(r["lrs"][gi]) * float(g["weight_decay"])
        prods.append(pr)
    out["analytic_products"] = prods
    order = man["optimizer_param_order"]
    names, k = [], 0
    for g in groups:
        gi = init.index(g["initial_lr"])
        for n in order[k:k + g["n_params"]]:
            names.append((n if isinstance(n, str) else (n.get("name") or n.get("param")), gi))
        k += g["n_params"]
    trainable = {n for n, _ in names}
    max_rel, n_changed, n_same = 0.0, 0, 0
    n_zero = sum(int((base[n] == 0).sum()) for n, _ in names)     # exact zeros of theta_0 stay 0 under decay
    tr_el = (man.get("elements") or {}).get("trainable") if isinstance(man.get("elements"), dict) else None
    for n, gi in names:
        x0, x1 = base[n].double(), st[n].double()
        ch = x0 != x1
        n_changed += int(ch.sum())
        n_same += int((~ch).sum())
        if ch.any():
            max_rel = max(max_rel, float((((x1[ch] / x0[ch]) - prods[gi]).abs() / prods[gi]).max()))
    others = [n for n in st if n not in trainable]
    differ = [n for n in others if n in base and not torch.equal(base[n], st[n])]
    out.update(optimizer_tensors=len(trainable), changed_elements=n_changed, unchanged_elements_of_optimizer_tensors=n_same,
               manifest_trainable_elements=tr_el, theta0_zeros_in_optimizer_tensors=n_zero,
               max_rel_err_vs_analytic=max_rel, other_tensors=len(others), other_tensors_differing=differ[:5],
               keys_missing=[n for n in base if n not in st][:5])
    ok = (bool(out["missing_steps_raises"]) and out["schedule"]["max_steps"] == int(job["steps"]) and out["schedule"]["warmup"] == int(job["warmup"])
          and rep.get("fingerprint") == "reproduced" and str(rep.get("train_log_lrs", "")).startswith("equal") and n_changed > 0
          and (tr_el is None or tr_el - n_zero <= n_changed <= tr_el)        # exactly the manifest's trainable elements decayed
          and max_rel < 1e-5 and not differ and not out["keys_missing"])
    out["ok"] = ok
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[wd_only_check] {a.run} on {a.device}: max_steps {rep['max_steps']} warm-up {rep['warmup']} executed {executed}; "
          f"fingerprint {rep.get('fingerprint')}; train-log LRs {rep.get('train_log_lrs')}; {n_changed} elements decayed (manifest trainable {tr_el}), "
          f"max rel. err. vs theta_0*prod(1-lr*wd) {max_rel:.2e}; other tensors differing {len(differ)}; "
          f"{rep['seconds']} s" + (f"; peak VRAM {out['peak_vram_gib']} GiB" if 'peak_vram_gib' in out else "") + f" -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
