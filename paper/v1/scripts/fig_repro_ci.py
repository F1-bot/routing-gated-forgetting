"""Percent-of-MoE intervals for fig_repro panel (b): stage-B forgetting relative to the MoE model.

data/stageB_window_ci.json holds window-bootstrap intervals for forgetting and for absolute differences to MoE, but
not for the relative difference. This script recomputes that bootstrap exactly as it was made (provenance of
stageB_window_ci.json version ad7ee6eb: same window files, seed 20261114, same draw order, 4000 resamples, DCLM
windows resampled jointly across corpora and models), asserts that it reproduces every [point, lo, hi] of the record
to 1e-12, and then adds 100 * (F_a - F_MoE) / F_MoE with 95% percentile intervals from the same paired resamples,
per corpus and for the four-corpus mean (ratio of means, as the record's mean_forgetting_pct).
Single runs: the intervals cover window sampling only.
Input:  <STAGEB_WINDOWS, default stageB_windows>/<exp>/{base,adapt_<corpus>}/ckpt__{dclm09,<corpus>}.npz
Output: figures/fig_repro_values.json (FIG_OUT overrides the folder), written via a temp file
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402

import numpy as np  # noqa: E402

import figstyle as fs  # noqa: E402

SBW = os.environ.get("STAGEB_WINDOWS", r"stageB_windows")
EXPS = ["dense_100m", "moe_100m", "nestor_100m"]
COR = ["pg19", "owm", "algstack", "fw2de"]
B, SEED = 4000, 20261114


def load(e, tag, s):
    z = np.load(os.path.join(SBW, e, tag, f"ckpt__{s}.npz"))
    return z["ce_sum"].astype(np.float64), z["n_tok"].astype(np.float64), z["hash"]


def ce(cs, nt, idx=None):
    if idx is None:
        return cs.sum() / nt.sum()
    return cs[idx].sum(1) / nt[idx].sum(1)


def ci(bs):
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


D, href = {}, {}
for e in EXPS:
    for c in COR:
        for kind, tag in (("before", "base"), ("after", f"adapt_{c}")):
            for s in ("dclm09", c):
                cs, nt, h = load(e, tag, s)
                D[(e, kind, c, s)] = (cs, nt)
                href.setdefault(s, h)
                assert np.array_equal(href[s], h), (e, tag, s)   # identical windows across models

rng = np.random.default_rng(SEED)
idx_d = rng.integers(0, 500, size=(B, 500))
idx_t = {c: rng.integers(0, 500, size=(B, 500)) for c in COR}

res = {}
for e in EXPS:
    F, G = {}, {}
    for c in COR:
        a, b = D[(e, "after", c, "dclm09")], D[(e, "before", c, "dclm09")]
        F[c] = (ce(*a) - ce(*b), ce(*a, idx_d) - ce(*b, idx_d))
        a, b = D[(e, "before", c, c)], D[(e, "after", c, c)]
        G[c] = (ce(*a) - ce(*b), ce(*a, idx_t[c]) - ce(*b, idx_t[c]))
    fm = (np.mean([F[c][0] for c in COR]), np.mean([F[c][1] for c in COR], axis=0))
    gm = (np.mean([G[c][0] for c in COR]), np.mean([G[c][1] for c in COR], axis=0))
    res[e] = (F, G, fm, gm)

# 1) the recomputation must reproduce the record exactly
rec = fs.load("stageB_window_ci.json")
checked = 0
def same(x, y):
    global checked
    assert len(x) == len(y) and all(abs(p - q) <= 1e-12 for p, q in zip(x, y)), (x, y)
    checked += len(x)
for e in EXPS:
    F, G, fm, gm = res[e]
    pm = rec["per_model"][e]
    same([float(fm[0])] + ci(fm[1]), pm["mean_forgetting"])
    same([float(gm[0])] + ci(gm[1]), pm["mean_gain"])
    for c in COR:
        same([float(F[c][0])] + ci(F[c][1]), pm["forgetting"][c])
        same([float(G[c][0])] + ci(G[c][1]), pm["gain"][c])
m = res["moe_100m"]
for a in ("dense_100m", "nestor_100m"):
    dv = rec["differences_vs_moe"][a]
    same([float(res[a][2][0] - m[2][0])] + ci(res[a][2][1] - m[2][1]), dv["mean_forgetting"])
    for c in COR:
        same([float(res[a][0][c][0] - m[0][c][0])] + ci(res[a][0][c][1] - m[0][c][1]), dv["forgetting"][c])
    assert abs(100 * (res[a][2][0] - m[2][0]) / m[2][0] - dv["mean_forgetting_pct"]) <= 1e-9

# 2) percent of MoE, same paired resamples
def pct(num, den):
    return [float(100 * (num[0] - den[0]) / den[0])] + ci(100 * (num[1] - den[1]) / den[1])
out = {
    "note": "stage-B forgetting relative to MoE, 100*(F_a - F_MoE)/F_MoE, [point, lo95, hi95]; window bootstrap "
            "(4000 paired resamples, seed 20261114) recomputed from the window files; single runs, window sampling only",
    "source": "stageB_window_ci.json (recomputation verified: %d values equal within 1e-12); windows: " % 0 + SBW,
    "corpora": {"pg19": "books (PG19)", "owm": "mathematics (OpenWebMath)", "algstack": "code (AlgebraicStack)",
                "fw2de": "German (FineWeb2)"},
    "pct_of_moe": {a: {"mean": pct(res[a][2], m[2]), **{c: pct(res[a][0][c], m[0][c]) for c in COR}}
                   for a in ("dense_100m", "nestor_100m")},
}
pt = rec["paper_table6_5B"]["mean_forgetting"]
out["paper_table6_pct_of_moe"] = {
    "dense": 100 * (pt["dense"] - pt["moe"]) / pt["moe"], "nestor": 100 * (pt["nestor"] - pt["moe"]) / pt["moe"],
    "source": "data/stageB_window_ci.json paper_table6_5B.mean_forgetting (Chu et al. 2026, Table 6: 500M-scale NESTOR "
              "and MoE, 1B dense, adapted for 0.5B tokens at 32k context; point values, no intervals)"}
out["source"] = out["source"].replace("verified: 0 values", f"verified: {checked} values")
fs.FIGDIR.mkdir(exist_ok=True)
dst = fs.FIGDIR / "fig_repro_values.json"
tmp = fs.FIGDIR / "fig_repro_values.tmp.json"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
os.replace(tmp, dst)
print(f"fig_repro_values written; {checked} record values reproduced")
