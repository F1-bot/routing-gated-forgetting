import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Figure: measured against predicted crossover per MoE base (central C1 result).

Now: the G0-7 probe crossovers at theta_0 (pinned, norm-matched random directions on the
registered books draws; data/gate_record_G0.json) and the
two reference lines. After the analysis lock: the confirmatory draws, read from
data/c1_results.json if that file exists (format: {"moe_b1": [{"d":1,"P":..,"I":..,"I_sym":..}, ...], ...}),
and the C2 draws at step 8 from data/c2_results.json (format: {"moe_b1": [{"d":1,"DQ":..,"I2":..,"J2":..}, ...], ...})
as a fourth panel. Both files are written by scripts/fill_core.py; CORE_DATA_DIR overrides their folder (tests).
Output: figures/fig_crossover.pdf, figures/fig_crossover.png (FIG_OUT overrides the folder)
"""
import os.path  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
g0 = fs.load("gate_record_G0.json")["gates"]["G0-7"]
probes = {"moe_b1": g0["moe_b1"]["draws"], "moe_b2": g0["moe_b2"]["draws"]}  # base 3 is deferred (P-16 as revised)
import json  # noqa: E402
from pathlib import Path  # noqa: E402
RES = Path(os.environ.get("CORE_DATA_DIR", str(fs.DATADIR)))
def _res(name):
    p = RES / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None
results = _res("c1_results.json")
results2 = _res("c2_results.json")
PANELS = [("moe_b1", "MoE base 1 (seed 42)"), ("moe_b2", "MoE base 2 (seed 43)")]
if results2:  # four panels: shorter titles so that the panel letters stay clear of them
    PANELS = [("moe_b1", "MoE base 1 (seed 42)"), ("moe_b2", "MoE base 2 (seed 43)")]
BLUE = fs.COLOR["moe"]

fig = fs.figure(2.6)
gs = fig.add_gridspec(1, 3 if results2 else 2, wspace=0.14 if results2 else 0.05, width_ratios=[1, 1, 1.15] if results2 else None)
axes = []
for j, (key, title) in enumerate(PANELS):
    ax = fig.add_subplot(gs[j], sharey=axes[0] if axes else None)
    axes.append(ax)
    xs = [0.0, 1.4]
    ax.plot(xs, xs, color=fs.INK, lw=0.9, zorder=1)
    ax.plot(xs, [0.25 * x for x in xs], color=fs.INK, lw=0.9, ls=":", zorder=1)
    ax.axhline(0, color=fs.GREY, lw=0.5, zorder=0)
    P = [d["P"] for d in probes[key]]; Ip = [d["I_probe"] for d in probes[key]]
    ax.plot(P, Ip, ls="none", marker="x", ms=4.0, mew=0.9, color=fs.GREY, zorder=2)
    if results and key in results:
        rows = results[key]
        ax.plot([r["P"] for r in rows], [r["I"] for r in rows], ls="none", marker="o", ms=4.6, mfc=BLUE, mec=BLUE, zorder=3)
        if all("I_sym" in r for r in rows):
            ax.plot([r["P"] for r in rows], [r["I_sym"] for r in rows], ls="none", marker="o", ms=4.6, mfc="white", mec=BLUE, zorder=3)
    ax.set_xlim(0, 1.4)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(-0.4, 1.8)
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_title(title, x=0.1)
    ax.set_xlabel("Predicted contrast $P_d$")
    if j == 0:
        ax.set_ylabel("Crossover $I_d$")
    else:
        for t in ax.get_yticklabels():
            t.set_visible(False)
if results2:
    ax = fig.add_subplot(gs[2])
    axes.append(ax)
    dq = [r["DQ"] for k in ("moe_b1", "moe_b2") for r in results2.get(k, [])]
    xmax = 1.1 * max(dq + [1.0])
    ax.plot([0, xmax], [0, 0.25 * xmax], color=fs.INK, lw=0.9, ls=":", zorder=1)
    ax.axhline(0, color=fs.GREY, lw=0.5, zorder=0)
    for k, mk in (("moe_b1", "D"), ("moe_b2", "s")):   # C2 glyphs differ from the C1 circles
        rows = results2.get(k, [])
        ax.plot([r["DQ"] for r in rows], [r["I2"] for r in rows], ls="none", marker=mk, ms=4.4, mfc=BLUE, mec=BLUE, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_title("C2: SGD $-$ Muon", x=0.1)
    ax.set_xlabel("Target contrast $\\Delta_{Q,d}$")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["right"].set_visible(True)    # the spine goes with the ticks
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("Interaction $I2_d$")
handles = [Line2D([], [], color=fs.INK, lw=0.9, label="prediction $I=P$"),
           Line2D([], [], color=fs.INK, lw=0.9, ls=":", label="margin, $0.25\\times$ contrast" if results2 else "margin $I=0.25\\,P$"),
           Line2D([], [], ls="none", marker="x", ms=4.0, mew=0.9, color=fs.GREY, label="probes at pretrained weights"),
           Line2D([], [], ls="none", marker="o", ms=4.6, mfc=BLUE, mec=BLUE, label="adapted, raw" if results2 else "adaptation runs, raw"),
           Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white", mec=BLUE, label="adapted, symmetrized" if results2 else "adaptation runs, symmetrized")]
if results2:
    handles += [Line2D([], [], ls="none", marker="D", ms=4.2, mfc=BLUE, mec=BLUE, label="C2, base 1"),
                Line2D([], [], ls="none", marker="s", ms=4.4, mfc=BLUE, mec=BLUE, label="C2, base 2")]
fig.legend(handles=handles, loc="outside lower center", ncol=4 if results2 else 3, columnspacing=1.0 if results2 else 1.6, handletextpad=0.5)
fs.add_letters(fig, axes, ["a", "b", "c", "d"][:len(axes)], align=True)
figalign.finalize(fig)
figcheck.check(fig, "fig_crossover")
fs.save(fig, "fig_crossover")
print("fig_crossover written", "(with results)" if results else "(placeholder)")
