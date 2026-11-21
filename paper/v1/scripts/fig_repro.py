"""Figure: reproduction of the NESTOR paper's domain adaptation (stage B, Table 6 analogue).

(a) Forgetting, the increase of cross-entropy on 500 windows of 4,096 tokens of DCLM shard 09, after 96 steps of
262,144 tokens at 4k context on each target corpus, for the dense model, MoE base 1 and NESTOR (routers frozen
for the routed models). (b) The same forgetting relative to MoE, 100 * (F - F_MoE) / F_MoE, per corpus and for the
four-corpus mean, next to the mean values of the paper's Table 6 (500M-scale NESTOR and MoE, 1B dense, adapted for
0.5B tokens at 32k context), drawn in a separate facet with open markers because those runs are not peers of ours.
Every cell is a single run; all intervals are 95% window-bootstrap intervals (window sampling only).
Inputs: data/stageB_window_ci.json; fig_repro_values.json from fig_repro_ci.py (data/ is searched first, then figures/)
Output: figures/fig_repro.pdf, figures/fig_repro.png (FIG_OUT overrides the folder)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
C = fs.COLOR
R = fs.load("stageB_window_ci.json")
_v = [p for p in (fs.DATADIR / "fig_repro_values.json", fs.ROOT / "figures" / "fig_repro_values.json") if p.exists()]
V = json.load(open(_v[0], encoding="utf-8"))
PM, PCT, PAPER = R["per_model"], V["pct_of_moe"], V["paper_table6_pct_of_moe"]
for a in ("dense_100m", "nestor_100m"):    # the two files must describe the same runs
    assert abs(PCT[a]["mean"][0] - R["differences_vs_moe"][a]["mean_forgetting_pct"]) < 1e-9
pt = R["paper_table6_5B"]["mean_forgetting"]
assert abs(PAPER["nestor"] - 100 * (pt["nestor"] - pt["moe"]) / pt["moe"]) < 1e-9

CORP = [("pg19", "books"), ("owm", "math"), ("algstack", "code"), ("fw2de", "German")]
MODELS = [("dense_100m", "dense", "dense"), ("moe_100m", "MoE", "moe"), ("nestor_100m", "NESTOR", "nestor")]

fig = fs.figure(2.7)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.04)
ax = fig.add_subplot(gs[0])
gsb = gs[1].subgridspec(2, 1, height_ratios=[5.0, 1.45], hspace=0.10)
bx = fig.add_subplot(gsb[0])
px = fig.add_subplot(gsb[1], sharex=bx)

# (a) absolute forgetting, single runs with window intervals
xg = [0.0, 1.0, 2.0, 3.0, 4.25]
for j, (key, _, ck) in enumerate(MODELS):
    dx = (j - 1) * 0.24
    for g, (c, _) in enumerate(CORP + [("mean", "")]):
        p, lo, hi = PM[key]["mean_forgetting"] if c == "mean" else PM[key]["forgetting"][c]
        x = xg[g] + dx
        ax.plot([x, x], [0, p], color=C[ck], lw=0.8, alpha=0.55, zorder=1, solid_capstyle="butt")
        ax.plot([x, x], [lo, hi], color=fs.INK, lw=0.8, zorder=2)
        ax.plot([x], [p], ls="none", marker="o", ms=4.8, mfc=C[ck], mec="white", mew=0.5, zorder=3)
ax.axvline(3.625, color=fs.LIGHT, lw=0.8, zorder=0)
ax.set_xticks(xg)
ax.set_xticklabels([lab for _, lab in CORP] + ["mean"])
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.45, 4.7)
ax.set_ylim(0, 0.285)
ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
ax.set_ylabel("Forgetting on DCLM (nats)")
ax.set_title("Dense forgets most everywhere")
ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.8, mfc=C[ck], mec="white", mew=0.5, label=lab)
                   for _, lab, ck in MODELS], loc="upper left", ncol=3, columnspacing=1.0, handletextpad=0.2,
          borderaxespad=0.2)

# (b) relative to MoE: this study (forest) and the paper's Table 6 means (separate facet, open markers)
ROWS = [(c, lab.split("\n")[0]) for c, lab in CORP] + [("mean", "mean of four")]
VALUE_LABELS = []
ys = [4.0, 3.0, 2.0, 1.0, -0.25]
for (key, lab, ck), dy in ((MODELS[0], 0.16), (MODELS[2], -0.16)):
    for (c, _), y in zip(ROWS, ys):
        p, lo, hi = PCT[key][c]
        bx.plot([lo, hi], [y + dy] * 2, color=C[ck], lw=1.1, zorder=2, solid_capstyle="butt")
        bx.plot([p], [y + dy], ls="none", marker="o", ms=4.8, mfc=C[ck], mec="white", mew=0.5, zorder=3)
    p, lo, hi = PCT[key]["mean"]
    VALUE_LABELS.append(bx.text(hi + 2.0, ys[-1] + dy, fs.signed(p, 1) + "%", ha="left", va="center",
                                fontsize=fs.SIZE_SECOND))   # beside the point, like every mean label
for axx in (bx, px):
    axx.axvline(0, color=C["moe"], lw=1.0, zorder=1)
bx.axhline(0.38, color=fs.LIGHT, lw=0.8, zorder=0)
bx.set_yticks(ys)
bx.set_yticklabels([lab for _, lab in ROWS[:-1]] + ["mean"])
bx.tick_params(axis="y", length=0)
bx.set_ylim(-0.75, 4.55)
bx.tick_params(axis="x", labelbottom=False)
bx.text(0.8, 4.47, "MoE", color=C["moe"], ha="left", va="top", fontsize=fs.SIZE_SECOND)
bx.set_title("MoE forgets least on average")
for key, ck, dy in (("dense", "dense", 0.16), ("nestor", "nestor", -0.16)):
    p = PAPER[key]
    px.plot([p], [dy], ls="none", marker="D", ms=4.6, mfc="white", mec=C[ck], mew=1.2, zorder=3)
    px.text(p + (2.4 if p > 0 else -2.4), dy, fs.signed(p, 1) + "%", ha="left" if p > 0 else "right",
            va="center", fontsize=fs.SIZE_SECOND)
px.set_yticks([0])
px.set_yticklabels(["preprint\nmean"])
px.tick_params(axis="y", length=0)
px.set_ylim(-0.55, 0.55)
px.set_xlim(-27, 72)
px.set_xticks([-20, 0, 20, 40, 60])
px.set_xlabel("Forgetting relative to MoE (%)")
fs.add_letters(fig, [ax, bx], ["a", "b"], align=True)
_ab = bx.get_window_extent()
for _t in VALUE_LABELS:                              # the mean labels must stay inside panel (b)
    _tb = _t.get_window_extent()
    if not (_ab.x0 <= _tb.x0 and _tb.x1 <= _ab.x1 and _ab.y0 <= _tb.y0 and _tb.y1 <= _ab.y1):
        raise SystemExit(f"value label {_t.get_text()} leaves panel (b)")
figalign.finalize(fig)
figcheck.check(fig, "fig_repro")
fs.save(fig, "fig_repro")
print("fig_repro written")
