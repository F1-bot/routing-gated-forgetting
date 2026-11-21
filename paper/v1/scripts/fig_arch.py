"""Figure: the three architectures and the adaptation protocol (schematic; numbers from fig_arch_facts.json).

(a) dense block, (b) MoE layer with its frozen token router and the routing record that turns per-token expert
choices into routing masses, (c) one NESTOR layer (block 2): frozen column router, column attention and
column-private experts with their own frozen router, the shared expert pool and the two-way gate, (d) the adaptation
protocol common to all runs. Terms follow Figure 1 and Sections 3-4 (A_d, B_d, H_d, L_d, routing mass W_c(S), frozen
routers). Residual and depth-read connections are not drawn; layer counts, widths, parameter counts, the K_c schedule
and the protocol numbers are in the caption. Every number the figure prints or counts out (24 squares, 3 chosen,
8 columns, 4 active, 8 private and 8 shared experts, 1 and 2 chosen) is read from figures/fig_arch_facts.json
(fig_arch_facts.py) through F(); the keys used are printed at the end. The bars of the routing record are
illustrative (no data). The layout check of figcheck.py runs before saving.
Output: figures/fig_arch.pdf, figures/fig_arch.png (FIG_OUT overrides the folder)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402

from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

fs.apply()
_f = [p for p in (fs.DATADIR / "fig_arch_facts.json", fs.ROOT / "figures" / "fig_arch_facts.json") if p.exists()]
FACTS = json.load(open(_f[0], encoding="utf-8"))["facts"]
USED = set()


def F(key):
    USED.add(key)
    return FACTS[key]["value"]


C = fs.COLOR
TINT = {k: to_rgba(v, 0.12) for k, v in C.items()}   # pale fills: the architecture colours at 12% opacity
OFF = to_rgba(fs.GREY, 0.55)                          # outlines of parts a token does not use
SNOW = "\N{SNOWFLAKE}"
W, H = fs.TEXT_WIDTH_MM, 90.0                         # drawing units: millimetres at print size
fig = fs.figure(H / 25.4)
fig.set_layout_engine("none")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_axis_off()
S2, S3 = fs.SIZE_SECOND, fs.SIZE_TICK


def T(x, y, s, size=S2, ha="center", va="center", color=fs.INK, **kw):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color, **kw)


def CT(x, y, s, size=S2, color=fs.INK, correction=(0.0, 0.0), **kw):
    """Text inside a shape: its ink box is centred on (x, y), moved by an optical correction in points."""
    return figalign.centered_text(ax, x, y, s, fontsize=size, color=color, correction=correction, **kw)


def box(x, y, w, h, s=None, ec=fs.INK, fc="white", lw=0.8, size=S2, r=0.9, ty=None, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw))
    if s:
        CT(x + w / 2, ty if ty is not None else y + h / 2, s, size=size, linespacing=1.2, **kw)


def chip(xc, yc, s, w):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - 2.0), w, 4.0, boxstyle="round,pad=0,rounding_size=1.0",
                                ec="none", fc=fs.LIGHT))
    CT(xc, yc, s, size=S3)


def arrow(p, q, color=fs.INK, lw=0.7, ls="-", ms=6.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0))


def grid(x0, y0, ncol, nrow, s, g, filled, col, ec_empty=fs.GREY, lw=0.5):
    """Expert squares, row-major from the top-left; returns the centres."""
    cen = []
    for k in range(ncol * nrow):
        i, j = k % ncol, k // ncol
        x, y = x0 + i * (s + g), y0 + (nrow - 1 - j) * (s + g)
        on = k in filled
        ax.add_patch(Rectangle((x, y), s, s, fc=col if on else "white", ec=col if on else ec_empty, lw=lw))
        cen.append((x + s / 2, y + s / 2))
    return cen


_, _lp_h, _lp_d = figalign.T2P.get_text_width_height_descent("lp", FontProperties(size=fs.SIZE_BASE), False)
TITLE_ASCENT = (_lp_h - _lp_d) * 25.4 / 72         # mm from the top of a title's layout box to its baseline


def head(x, y, letter, title):
    """Panel letter and title on one baseline: the baseline the title has when the top of its box is at y."""
    yb = y - TITLE_ASCENT
    t = T(x + 4.2, yb, title, size=fs.SIZE_BASE, ha="left", va="baseline")
    T(x, yb, letter, size=fs.SIZE_LETTER, ha="left", va="baseline", fontweight="bold")._optical_title = t


Y_TOP = H - 1.5                                   # panel letters and titles
SIGMA_CORR = (0.0, 0.0)                           # optical corrections (points) on top of ink centring: none for
G_CORR = (0.0, -0.25)                             # the sum sign; g 0.25 pt lower, towards its bowl (judged at 4x)
Y_IN, Y_ATT0, Y_ATT1 = 28.0, 33.0, 40.5           # shared levels of the three top panels (mm)
Y_OUT = 75.0

# ---------------- (a) dense ----------------
xa = 13.5
head(0.0, Y_TOP, "a", "Dense")
T(xa, Y_IN - 0.8, "token state", size=S3, va="top")
arrow((xa, Y_IN), (xa, Y_ATT0))
box(xa - 10.5, Y_ATT0, 21.0, Y_ATT1 - Y_ATT0, "attention", ec=C["dense"], fc=TINT["dense"])
arrow((xa, Y_ATT1), (xa, 44.0))
box(xa - 10.5, 44.0, 21.0, 21.5, "SwiGLU MLP", ec=C["dense"], fc=TINT["dense"])
arrow((xa, 65.5), (xa, Y_OUT))
T(xa, Y_OUT + 0.8, "next layer", size=S3, va="bottom")

# ---------------- (b) MoE ----------------
xb0, xb = 30.5, 45.0
E, K = F("moe.experts_per_layer"), F("moe.experts_per_token")
head(xb0, Y_TOP, "b", "MoE")
T(xb, Y_IN - 0.8, "token state", size=S3, va="top")
arrow((xb, Y_IN), (xb, Y_ATT0))
box(xb - 14.0, Y_ATT0, 28.0, Y_ATT1 - Y_ATT0, "attention", ec=C["moe"], fc=TINT["moe"])
arrow((xb, Y_ATT1), (xb, 44.0))
box(xb - 14.0, 44.0, 28.0, 6.0, f"{SNOW} router: {K} of {E}", ec=fs.INK, size=S3)
sq, gp, ncol = 2.75, 0.55, 8
assert E % ncol == 0
gx0 = xb - (ncol * sq + (ncol - 1) * gp) / 2
chosen = set([2, 13, 21, 6, 17, 10, 0, 23][:K])  # which squares are filled is illustrative; their number is K
cen = grid(gx0, 54.0, ncol, E // ncol, sq, gp, chosen, C["moe"])
for k in sorted(chosen):
    arrow((xb, 50.0), (cen[k][0], cen[k][1] - sq / 2), color=C["moe"], lw=0.6, ms=5)
sig = (xb, 68.6)
ax.add_patch(Circle(sig, 2.2, fc="white", ec=C["moe"], lw=0.8))
CT(*sig, "\N{GREEK CAPITAL LETTER SIGMA}", size=S2, correction=SIGMA_CORR)
for k in sorted(chosen):
    arrow((cen[k][0], cen[k][1] + sq / 2), (sig[0] + (cen[k][0] - sig[0]) * 0.2, sig[1] - 1.9), color=C["moe"],
          lw=0.6, ms=5)
arrow((xb, 70.8), (xb, Y_OUT))
T(xb, Y_OUT + 0.8, "next layer", size=S3, va="bottom")
# routing record (illustrative bars; no data)
xr0, xr1 = 63.5, 91.0
xrm = (xr0 + xr1) / 2
T(xrm, 81.0, "routing record at $\\theta_0$", size=S2, va="top")
heights = [3.5, 6.0, 2.0, 8.5, 4.0, 1.5, 5.0, 7.0, 2.5, 3.0, 9.5, 4.5, 2.0, 6.5, 3.0, 5.5, 1.8, 7.5, 4.0, 2.8, 6.0,
           3.5, 8.0, 2.2]
assert len(heights) == E
in_S = {1, 4, 8, 13, 17, 20}
pitch = (xr1 - xr0) / E
for m in range(E):
    ax.add_patch(Rectangle((xr0 + m * pitch + 0.12, 58.0), pitch - 0.24, heights[m] * 1.0,
                           fc=C["moe"] if m in in_S else OFF, ec="none"))
ax.plot([xr0, xr1], [58.0, 58.0], color=fs.INK, lw=0.6)
T(xrm, 73.2, "$w_c(m)$ per expert", size=S3, va="center")
T(xrm, 56.4, "subset $S$", size=S3, va="top", color=C["moe"])
arrow((xrm, 52.6), (xrm, 49.6), lw=0.6, ms=4.5)
T(xrm, 48.6, "routing mass $W_c(S)$", size=S3, va="top")
arrow((xb + 14.0, 47.0), (xr0 - 0.6, 61.0), color=fs.INK, lw=0.6, ls=(0, (2.5, 1.5)), ms=5)

# ---------------- (c) NESTOR ----------------
xc0 = 95.5
Cc = F("nestor.columns")
blk = 2                                           # the drawn layer belongs to block 2
k_here = F("nestor.columns_per_token")[blk - 1]
kq_here = F("nestor.private_top_k")[blk - 1]
head(xc0, Y_TOP, "c", "NESTOR")
lw_, lg = 3.7, 0.8
lx0 = 110.5
lanes = [lx0 + i * (lw_ + lg) for i in range(Cc)]
xend = lanes[-1] + lw_
xmid = (lanes[0] + xend) / 2
sel = [0, 2, 5, 6, 3, 7, 1, 4][:k_here]           # which columns are active is illustrative; their number is K_c
T(xmid, Y_IN - 0.8, "token state", size=S3, va="top")
arrow((xmid, Y_IN), (xmid, 31.0))
box(lanes[0], 31.0, xend - lanes[0], 6.0, f"{SNOW} column router", ec=fs.INK, size=S3)
nq = F("nestor.private_experts_per_column")
LT = 67.0                                         # lane top
for i, x in enumerate(lanes):
    on = i in sel
    ax.add_patch(FancyBboxPatch((x, 40.5), lw_, LT - 40.5, boxstyle="round,pad=0,rounding_size=0.8",
                                ec=C["nestor"] if on else OFF, fc=TINT["nestor"] if on else "white", lw=0.7))
    ax.add_patch(Rectangle((x + 0.45, 42.5), lw_ - 0.9, 6.0, fc=C["nestor"] if on else fs.LIGHT, ec="none"))
    ax.add_patch(Rectangle((x + 0.45, 50.5), lw_ - 0.9, 1.5, fc="white", ec=fs.INK if on else OFF, lw=0.5))
    grid(x + (lw_ - (2 * 1.3 + 0.3)) / 2, 54.5, 2, nq // 2, 1.3, 0.3, set([3, 6][:kq_here]) if on else set(),
         C["nestor"], ec_empty=C["nestor"] if on else OFF, lw=0.45)
    if on:
        arrow((x + lw_ / 2, 37.0), (x + lw_ / 2, 40.5), lw=0.6, ms=4.5)
gate = (xmid, 72.0)
for i in sel:
    arrow((lanes[i] + lw_ / 2, LT), (gate[0] + (lanes[i] + lw_ / 2 - gate[0]) * 0.28, gate[1] - 1.9),
          color=C["nestor"], lw=0.6, ms=4.5)
ns, nk = F("nestor.shared_experts"), F("nestor.shared_top_k")
bx0 = 149.6
sh_top = 54.5 + 4 * 2.2 + 3 * 0.45
grid(bx0, 54.5, 2, ns // 2, 2.2, 0.45, set([1, 4, 6][:nk]), C["nestor"], ec_empty=C["nestor"], lw=0.5)
arrow((bx0 + 2.4, sh_top + 0.4), (gate[0] + 1.6, gate[1] - 1.4), color=C["nestor"], lw=0.6, ms=4.5)
T(bx0 + 2.4, 52.8, f"{ns} shared\nexperts", size=S3, va="top", linespacing=1.15)
ax.add_patch(Circle(gate, 2.0, fc="white", ec=C["nestor"], lw=0.8))
CT(*gate, "$g$", size=S2, correction=G_CORR)
T(gate[0] + 3.4, gate[1] + 0.6, "two-way gate", size=S3, ha="left", va="bottom")
arrow((gate[0], gate[1] + 2.0), (gate[0], Y_OUT + 2.5))
T(gate[0] - 1.2, Y_OUT + 1.9, "next layer", size=S3, ha="right", va="center")
for y, s in ((45.5, "column\nattention"), (51.2, f"{SNOW} router"), (57.5, f"{nq} private\nexperts")):
    T(lanes[0] - 2.0, y, s, size=S3, ha="right", linespacing=1.12)
    ax.plot([lanes[0] - 1.7, lanes[0] + 0.3], [y, y], color=fs.GREY, lw=0.45)

# ---------------- (d) protocol ----------------
yd0, yd1 = 1.5, 14.5
head(0.0, 21.0, "d", "Adaptation protocol")
B = [(0.5, 24.0, "pretrained\nbase $\\theta_0$", None),
     (29.5, 34.0, "routing record\npicks $S$", None),
     (68.5, 38.0, "adapt $S$ on target", [f"{SNOW} routers frozen"]),
     (111.5, 47.2, "forgetting $F$", ["code", "math", "DCLM"])]
for x, w, s, chips in B:
    # boxes with chips centre their label in the free band between the chip tops and the box top
    box(x, yd0, w, yd1 - yd0, s, ec=fs.INK, size=S2, ty=(yd0 + yd1) / 2 if not chips else (yd0 + 3.7 + 2.0 + yd1) / 2)
    if chips:
        cw = (w - 4.0 - 2.0 * (len(chips) - 1)) / len(chips)
        for j, c in enumerate(chips):
            chip(x + 2.0 + cw / 2 + j * (cw + 2.0), yd0 + 3.7, c, cw)
for (x, w, _, _), (x2, _, _, _) in zip(B[:-1], B[1:]):
    arrow((x + w, (yd0 + yd1) / 2), (x2, (yd0 + yd1) / 2))

figalign.finalize(fig)
figcheck.check(fig, "fig_arch")
fs.save(fig, "fig_arch")
print("fig_arch written; facts used:", ",".join(sorted(USED)))
