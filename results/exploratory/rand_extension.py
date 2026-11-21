"""Exploratory (post-freeze) extension of the C1 random-direction control to all 20 C1 draws.
Uses the registered analysis code (analyze_confirm.Ctx.cross, _beta_set, _diff_ci) unchanged; only for the 28 runs of
results/exploratory/queue_randx.json (no registered randdir evaluation) it supplies the exploratory seed and the
eval_randx file paths. Registered outputs are read only; nothing registered is rewritten."""
import json, math, os, sys
ROOT = r"."
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
import analyze_confirm as AC

def build(verify=True):
    ctx = AC.Ctx(ROOT, "FREEZE.json", verify=verify, mode="core")
    QX = json.load(open(os.path.join(ROOT, "results", "exploratory", "queue_randx.json"), encoding="utf-8"))
    seed_of, out_of = {}, {}
    for j in QX["jobs"]:
        rid = j["id"][len("randx_"):]
        a = j["args"]
        seed_of[rid] = a[a.index("--variants") + 1].split(",")[0].split(":", 1)[1]
        out_of[rid] = os.path.join(ROOT, a[a.index("--out") + 1])
    oh, oe = ctx.haar, ctx.eval_file
    def haar(rid):
        h = oh(rid)
        return h if h else seed_of.get(rid)
    def eval_file(rid, variant, set_name):
        f = oe(rid, variant, set_name)
        if f is None and rid in seed_of and variant.startswith("randdir"):
            return os.path.join(out_of[rid], f"{variant.replace(':', '-')}__{set_name}.npz")
        return f
    ctx.haar, ctx.eval_file = haar, eval_file
    return ctx, oh, seed_of

def complete(ctx, rid):
    h = ctx.haar(rid)
    return bool(h) and all(ctx.eval_file(rid, v.format(h=h), c) and os.path.exists(ctx.eval_file(rid, v.format(h=h), c))
                           for v in ("randdir:{h}", "randdir-:{h}") for c in (ctx.code, "math"))

def run(out_path=None, verify=True):
    ctx, orig_haar, seed_of = build(verify)
    corp = (ctx.code, "math")
    draws, dropped, e1 = ctx.screen(ctx.draws_of("c1"))
    reg = [d for d in draws if orig_haar(d["A"]) and orig_haar(d["B"])]
    allr = [d for d in draws if complete(ctx, d["A"]) and complete(ctx, d["B"])]
    res = {"label": "post-freeze exploratory; seeds fixed before running (queue_randx.json)",
           "n_registered_control_draws": len(reg), "n_draws_with_rand": len(allr),
           "draws": [[d["base"], d["d"]] for d in allr]}
    X6 = ctx.cross(reg, "rand", corp)
    res["check_registered_six"] = {"I": X6["I"], "statistic": X6["statistic"]}
    Xr, Xs, Xw = (ctx.cross(allr, k, corp) for k in ("rand", "sym", "raw"))
    res["statistic"] = {"rand": Xr["statistic"], "sym": Xs["statistic"], "raw": Xw["statistic"]}
    res["per_draw"] = [{"base": b, "d": d, "I_rand": i} for b, d, i in zip(Xr["bases"], Xr["ds"], Xr["I"])]
    res["beta_rand"] = AC._beta_set(Xr["I"], Xr["Pv"], Xr["bases"], "rand")
    res["I_sym_minus_I_rand"] = AC._diff_ci(Xs, Xr)
    res["I_raw_minus_I_rand"] = AC._diff_ci(Xw, Xr)
    res["mean_I"] = {"rand": float(np.mean(Xr["I"])), "sym": float(np.mean(Xs["I"])), "raw": float(np.mean(Xw["I"]))}
    res["share_rand_over_sym"] = res["mean_I"]["rand"] / res["mean_I"]["sym"]
    res["share_rand_over_raw"] = res["mean_I"]["rand"] / res["mean_I"]["raw"]
    pb = {}
    for b in sorted(set(Xr["bases"])):
        sel = [d for d in allr if d["base"] == b]
        r_, s_, w_ = (ctx.cross(sel, k, corp) for k in ("rand", "sym", "raw"))
        pb[b] = {"n": len(sel), "mean_I_rand": float(np.mean(r_["I"])), "mean_I_sym": float(np.mean(s_["I"])),
                 "share_rand_over_sym": float(np.mean(r_["I"]) / np.mean(s_["I"])),
                 "I_sym_minus_I_rand": AC._diff_ci(s_, r_)}
    res["per_base"] = pb
    if out_path:
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=float)
    return res

if __name__ == "__main__":
    out = os.path.join(ROOT, "results", "exploratory", "rand_extension.json")
    r = run(out)
    print(json.dumps({k: r[k] for k in ("n_draws_with_rand", "mean_I", "share_rand_over_sym", "I_sym_minus_I_rand")}, default=float)[:1500])
