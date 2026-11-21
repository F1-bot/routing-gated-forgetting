import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Figure: routing entanglement predicts on which crossover targets the registered search succeeds.

Input: data/entanglement_results.json (entanglement.py on the re-run G0 tables;
point values and 95% percentile bootstrap CIs over selection windows) and the G0
generator outcomes it stores in agreement.rows (registered tolerance log 1.05).
Output: figures/fig_entanglement.pdf, figures/fig_entanglement.png
"""
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
E = fs.load("entanglement_results.json")
cells, agree = E["cells"], E["agreement"]["rows"]
thr = E["thresholds"]
C = fs.COLOR
MOE2 = to_hex([0.5 + 0.5 * v for v in to_rgb(C["moe"])])   # MoE base 2: the MoE blue at a 50% tint (and dashed)
BASES = [("moe_b1", "MoE base 1", C["moe"], "-", +0.24), ("moe_b2", "MoE base 2", MOE2, fs.DASH_SEED43, 0.0),
         ("nestor", "NESTOR", C["nestor"], "-", -0.24)]
ROWS = [("c1|books|code", "C1 books, code"), ("c1|books|pycode", "C1 books, Python code"),
        ("c1|german|code", "C1 German, code"), ("c1|german|pycode", "C1 German, Python code"),
        ("c1|french|code", "C1 French, code"), ("c1|french|pycode", "C1 French, Python code"),
        ("c2_dclm_only|german", "C2 German, DCLM matched"), ("c2_full|german", "C2 German, all matched"),
        ("dclm_tilt|german", "DCLM tilt (German)")]
RUNG = {"c2_dclm_only": 1}            # G0-3 ladder: DCLM-only matching is its rung 1 (log 1.05)


def outcome(key):
    """G0 generator outcome at f = 1/2 and tolerance log 1.05, or None if G0 did not run this cell."""
    design = key.split("|")[1]
    rows = [r for r in agree if r["cell"] == key and abs(r["f"] - 0.5) < 1e-3 and r["rung"] == RUNG.get(design, 0)]
    return rows[0]["g0_2_pass"] if rows else None


def marker_kw(col, res):
    if res is None:
        return dict(marker="D", ms=4.0, mfc="white", mec=col, mew=1.0)
    return dict(marker="o", ms=4.6, mfc=col if res else "white", mec=col, mew=1.0)


fig = fs.figure(3.55)
gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0])
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# (a) forest of loads
ny = len(ROWS)
for i, (rk, label) in enumerate(ROWS):
    y0 = ny - 1 - i - (0.55 if i >= 6 else 0.0)
    for base, blabel, col, ls, dy in BASES:
        key = f"{base}|{rk}|f=0.500" if rk.startswith("c1|") else f"{base}|{rk}|f=0.500"
        c = cells[key]
        x = c["point"]["xi_LP"]; lo, hi = c["xi_LP_ci95"]
        ax.plot([lo, hi], [y0 + dy] * 2, color=col, ls=ls, lw=1.1, zorder=2)
        ax.plot([x], [y0 + dy], ls="none", zorder=3, **marker_kw(col, outcome(key)))
yt = [ny - 1 - i - (0.55 if i >= 6 else 0.0) for i in range(ny)]
ax.set_yticks(yt)
ax.set_yticklabels([lab for _, lab in ROWS])
ax.tick_params(axis="y", length=0)
ax.axvline(thr["operating"], color=fs.INK, ls=":", lw=0.9, zorder=1)
ax.axvline(thr["existence"], color=fs.GREY, ls="--", lw=0.9, zorder=1)
ax.set_xlim(-0.03, 1.32)
ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax.set_ylim(min(yt) - 0.55, ny - 0.35)
ax.set_xlabel("Load $\\xi_{\\mathrm{LP}}$ at $f=1/2$ (95% bootstrap CI)")
ax.set_title("Books: low load in every base")

# (b) mechanism: loads against the correlation of target and DCLM routing shares (C1 cells)
for base, blabel, col, ls, dy in BASES:
    for tgt in ["books", "german", "french"]:
        for code, dx in [("code", -0.012), ("pycode", +0.012)]:
            key = f"{base}|c1|{tgt}|{code}|f=0.500"
            c = cells[key]
            x = c["point"]["corr_TD_share"] + dx; y = c["point"]["xi_LP"]; lo, hi = c["xi_LP_ci95"]
            bx.plot([x, x], [lo, hi], color=col, ls=ls, lw=1.0, zorder=2)
            bx.plot([x], [y], ls="none", zorder=3, **marker_kw(col, outcome(key)))
bx.axhline(thr["operating"], color=fs.INK, ls=":", lw=0.9, zorder=1)
bx.axhline(thr["existence"], color=fs.GREY, ls="--", lw=0.9, zorder=1)
bx.text(0.80, 0.335, "books", fontsize=fs.SIZE_SECOND, ha="center", va="bottom")
bx.text(-0.02, 0.88, "German,\nFrench", fontsize=fs.SIZE_SECOND, ha="center", va="top", color=fs.INK)
bx.text(0.615, 0.86, "German,\nFrench", fontsize=fs.SIZE_SECOND, ha="center", va="bottom", color=fs.INK)
bx.set_xlim(-0.25, 1.05)
bx.set_xticks([0.0, 0.5, 1.0])
bx.set_yticks([0.4, 0.6, 0.8, 1.0])
bx.set_ylim(0.32, 1.10)
bx.set_xlabel("Correlation of target and\nDCLM routing shares")
bx.set_ylabel("Load $\\xi_{\\mathrm{LP}}$ at $f=1/2$")
bx.set_title("Low load where routing\ntracks DCLM")

handles = [Line2D([], [], color=C["moe"], ls="-", lw=1.1, label="MoE base 1"),
           Line2D([], [], color=MOE2, ls=fs.DASH_SEED43, lw=1.1, label="MoE base 2"),
           Line2D([], [], color=C["nestor"], ls="-", lw=1.1, label="NESTOR"),
           Line2D([], [], ls="none", marker="o", ms=4.6, mfc=fs.INK, mec=fs.INK, label="generator found pairs"),
           Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white", mec=fs.INK, label="generator failed"),
           Line2D([], [], ls="none", marker="D", ms=4.0, mfc="white", mec=fs.INK, label="cell not run"),
           Line2D([], [], color=fs.INK, ls=":", lw=0.9, label="search threshold, $\\xi_{\\mathrm{LP}}=0.67$"),
           Line2D([], [], color=fs.GREY, ls="--", lw=0.9, label="exact-match limit, $\\xi_{\\mathrm{LP}}=1$")]
fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.4, handletextpad=0.5, handlelength=1.8)
fs.add_letters(fig, [ax, bx], ["a", "b"], align=True)
figalign.finalize(fig)
figcheck.check(fig, "fig_entanglement")
fs.save(fig, "fig_entanglement")
print("fig_entanglement written")
