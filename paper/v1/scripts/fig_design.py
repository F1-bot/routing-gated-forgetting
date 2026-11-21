import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Figure 1: the two registered crossover designs (schematic; no data).

Notation follows Section 4 of the paper: A_d/B_d, H_d/L_d, F_c, P_d, Delta_Q, J_d, J2_d. Top: one layer of experts with
the two pools and one draw. Bottom: a 2x2 crossover table per design whose cells give the sign with which the log
forgetting of each run enters I_d (C1) or I2_d (C2), so each table reads as a difference of differences.
Output: figures/fig_design.pdf, figures/fig_design.png
"""
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
BLUE = fs.COLOR["moe"]
EDGE = fs.COLOR["dense"]                   # palette grey #7F7F7F: outline of unselected experts (white fill)
SQ, GAP = 0.33, 0.055                      # expert square and gap (units; 1 unit = 1 unit in x and y)
CELL = to_rgba(fs.GREY, 0.12)              # pale grey sign cells (palette grey at 12%)
HEAD = to_rgba(BLUE, 0.12)                 # pale blue column headers (MoE blue at 12%)
MINUS = "\N{MINUS SIGN}"
Y_AX = 5.72                                # ordering axis of the top part
CW, CH = 2.2, 0.58                         # table cell width and height
Y_HEAD, Y_ROW1, Y_ROW2, Y_TEST = 4.68, 3.92, 3.21, 2.50
X_COLS = (2.2, 7.8)                        # table columns, same in both panels, each under its pool


def experts(ax, y, chosen_a, chosen_b, pool, n=24, x0=0.42):
    """One routed layer: n experts sorted by a score; the first/last `pool` form the two pools."""
    for i in range(n):
        x = x0 + i * (SQ + GAP)
        sel = i in chosen_a or i in chosen_b
        ax.add_patch(Rectangle((x, y), SQ, SQ, fc=BLUE if sel else "white", ec=BLUE if sel else EDGE, lw=0.7))
    xl = x0 - 0.05
    xr = x0 + n * (SQ + GAP) - GAP + 0.05
    xa = x0 + pool * (SQ + GAP) - GAP / 2 + 0.03
    xb = x0 + (n - pool) * (SQ + GAP) - GAP / 2 - 0.03
    return xl, xa, xb, xr


def bracket(ax, x1, x2, y, text, up=True, size=fs.SIZE_TICK):
    h = 0.14 if up else -0.14
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=fs.INK, lw=0.7)
    ax.text((x1 + x2) / 2, y + (0.08 if up else -0.08), text, ha="center", va="bottom" if up else "top", fontsize=size)


def box(ax, x, y, w, h, text, ec=BLUE, size=fs.SIZE_SECOND):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, p, q, color=fs.INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color, shrinkA=0, shrinkB=0))


def rbox(ax, xc, yc, w, h, fc, ec="none", lw=0.8):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=fc, ec=ec, lw=lw))


def crossover_table(ax, xcols, heads, mid, rows, signs, test):
    """2x2 sign table: columns = the two subsets (under their pools), rows = measured streams or optimizer arms.

    Each cell shows the sign with which the log forgetting of that run enters the crossover statistic."""
    xm = (xcols[0] + xcols[1]) / 2
    for xc, hd in zip(xcols, heads):
        arrow(ax, (xc, Y_AX - 0.08), (xc, Y_HEAD + CH / 2), color=BLUE)
        rbox(ax, xc, Y_HEAD, CW, CH, HEAD, ec=BLUE)
        figalign.centered_text(ax, xc, Y_HEAD, hd, fontsize=fs.SIZE_SECOND)
    ax.text(xm, Y_HEAD, mid, ha="center", va="center", fontsize=fs.SIZE_SECOND)
    for y, lab, pair in zip((Y_ROW1, Y_ROW2), rows, signs):
        ax.text(xm, y, lab, ha="center", va="center", fontsize=fs.SIZE_SECOND)
        for xc, s in zip(xcols, pair):
            rbox(ax, xc, y, CW, CH, CELL)
            figalign.centered_text(ax, xc, y, s, fontsize=fs.SIZE_BASE, color=fs.INK)
    ax.text(xm, Y_TEST, test, ha="center", va="center", fontsize=fs.SIZE_SECOND)


fig = fs.figure(1.95)
gs = fig.add_gridspec(1, 2, wspace=0.04)
axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axa, axb):
    ax.set_xlim(0, 10)
    ax.set_ylim(2.14, 7.78)
    ax.set_aspect("equal")
    ax.set_axis_off()

# (a) C1
ax = axa
ax.set_title("C1: cross-corpus crossover", x=0.08)
y_row = 6.55
xl, xa, xb, xr = experts(ax, y_row, {0, 2, 3, 6, 8, 11}, {12, 15, 17, 18, 21, 23}, pool=12)
bracket(ax, xl, xa, y_row + SQ + 0.22, "A-pool")
bracket(ax, xb, xr, y_row + SQ + 0.22, "B-pool")
ax.text(xa - 0.1, y_row - 0.12, "$A_d$", ha="right", va="top", fontsize=fs.SIZE_SECOND, color=BLUE)
ax.text(xr, y_row - 0.12, "$B_d$", ha="right", va="top", fontsize=fs.SIZE_SECOND, color=BLUE)
arrow(ax, (xl, 5.72), (xr, 5.72))
ax.text((xl + xr) / 2, 5.6, "ordered by code$-$math tilt", ha="center", va="top", fontsize=fs.SIZE_TICK)
crossover_table(ax, X_COLS, ("adapt $A_d$", "adapt $B_d$"), "on books",
                ("$F_{\\mathrm{code}}$", "$F_{\\mathrm{math}}$"), ((MINUS, "+"), ("+", MINUS)), "test: mean $J_d>0$")

# (b) C2
ax = axb
ax.set_title("C2: optimizer crossover", x=0.08)
xl, xa, xb, xr = experts(ax, y_row, {0, 1, 3, 4, 6, 7}, {16, 17, 18, 20, 22, 23}, pool=8)
bracket(ax, xl, xa, y_row + SQ + 0.22, "L-pool")
bracket(ax, xb, xr, y_row + SQ + 0.22, "H-pool")
ax.text(xa - 0.1, y_row - 0.12, "$L_d$", ha="right", va="top", fontsize=fs.SIZE_SECOND, color=BLUE)
ax.text(xr, y_row - 0.12, "$H_d$", ha="right", va="top", fontsize=fs.SIZE_SECOND, color=BLUE)
arrow(ax, (xl, 5.72), (xr, 5.72))
ax.text((xl + xr) / 2, 5.6, "German routing mass", ha="center", va="top", fontsize=fs.SIZE_TICK)
crossover_table(ax, X_COLS, ("adapt $L_d$", "adapt $H_d$"), "on German",
                ("$F_{\\mathrm{DCLM}}$, SGD", "$F_{\\mathrm{DCLM}}$, Muon"), ((MINUS, "+"), ("+", MINUS)),
                "test: mean $J2_d>0$")

fs.add_letters(fig, [axa, axb], ["a", "b"], align=True)
figalign.finalize(fig)
figcheck.check(fig, "fig_design")
fs.save(fig, "fig_design")
print("fig_design written")
