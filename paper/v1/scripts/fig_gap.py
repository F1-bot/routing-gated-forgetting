"""Figure: NESTOR - MoE training-loss gap in the 0.5B-token reproduction.

Input: data/gap_trajectory.json (gap_analysis.py output; smooth curves and bands are
display only, all statistics in the text come from the raw logged points).
Output: figures/fig_gap.pdf, figures/fig_gap.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import math

import numpy as np

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
g = fs.load("gap_trajectory.json")
ser, gaps, sch = g["series"], g["gaps"], g["schedule"]
tok = np.array(ser["tokens_B"])
C = fs.COLOR

fig = fs.figure(4.15)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# (a) training loss
runs = [("nestor", "NESTOR", C["nestor"], "-"), ("dense", "Dense", C["dense"], "-"),
        ("moe42", "MoE, seed 42", C["moe"], "-"), ("moe43", "MoE, seed 43", C["moe"], fs.DASH_SEED43)]
for key, label, col, ls in runs:
    raw = np.array(ser["loss"][key]["raw"]); sm = np.array(ser["loss"][key]["smooth"])
    ax1.plot(tok, raw, ls="none", marker="o", ms=1.4, color=col, alpha=0.35, mec="none", zorder=1)
    ax1.plot(tok, sm, color=col, ls=ls, lw=1.3, label=label, zorder=3)
lo, hi = 4.5, 6.8
ax1.set_ylim(lo, hi)
ax1.set_xlim(0, 0.512)
ax1.set_ylabel("Training loss (nats/token)")
ax1.axvspan(tok[95], 0.5, color=fs.LIGHT, zorder=0, lw=0)
ax1.legend(loc="upper right", ncol=2, columnspacing=1.2)
ax1.set_title("Training loss during pretraining")
plt_hide = [t.set_visible(False) for t in ax1.get_xticklabels()]

# top axis: learning rate relative to its peak (schedule shared by all four runs)
Tm, W = sch["this_reproduction"]["T_max"], sch["this_reproduction"]["warmup_steps"]
peak = g["runs"]["moe42"]["peak_lr_max_fitted"]; eta = sch["this_reproduction"]["eta_min"]
def lr_rel(n):
    return n / W if n < W else (eta + (peak - eta) * (1 + math.cos(math.pi * (n - W) / Tm)) / 2) / peak
tps = sch["this_reproduction"]["tokens_per_step"]
top = ax1.secondary_xaxis("top")
ticks_B = [0.1, 0.2, 0.3, 0.4, 0.5]
top.set_xticks(ticks_B)
top.set_xticklabels([f"{lr_rel(b * 1e9 / tps):.2f}" for b in ticks_B])
top.set_xlabel("Learning rate relative to peak", fontsize=fs.SIZE_SECOND)
top.tick_params(labelsize=fs.SIZE_TICK)

# (b) gaps
series = [("N_minus_M42", "NESTOR $-$ MoE, seed 42", C["nestor"], "-"),
          ("N_minus_M43", "NESTOR $-$ MoE, seed 43", C["nestor"], fs.DASH_SEED43),
          ("N_minus_D", "NESTOR $-$ dense", C["nestor"], ":"),
          ("M43_minus_M42", "MoE seed 43 $-$ seed 42", C["moe"], "-")]
for key, label, col, ls in series:
    s = gaps[key]["series"]
    ax2.fill_between(tok, s["lo95"], s["hi95"], color=col, alpha=0.13, lw=0, zorder=1)
    ax2.plot(tok, s["smooth"], color=col, ls=ls, lw=1.3, label=label, zorder=3)
    ax2.plot([0.5], [gaps[key]["heldout_final"]["val09"]], marker="D", ms=4.2, mfc="white", mec=col, mew=1.0, ls="none", zorder=4, clip_on=False)
ax2.plot([], [], marker="D", ms=4.2, mfc="white", mec=fs.INK, mew=1.0, ls="none", label="held-out gap (val09)")
ax2.axhline(0, color=fs.INK, lw=0.6, zorder=2)
ax2.axvspan(tok[95], 0.5, color=fs.LIGHT, zorder=0, lw=0)
lh = gaps["N_minus_Mavg"]["last_half"]
ch, (cl, cu) = lh["change_over_last_half_nats"], lh["change_over_last_half_ci95"]
ax2.text(tok[95] + 0.004, 0.292, "last half", fontsize=fs.SIZE_SECOND, color=fs.INK, ha="left", va="top")
# the change over the last half (ch, cl, cu) is stated in the caption, not drawn
ax2.set_ylim(-0.33, 0.30)
ax2.set_xlim(0, 0.512)
ax2.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_ylabel("Training-loss gap (nats)")
ax2.set_xlabel("Training tokens (billions)")
ax2.legend(loc="lower left", ncol=2, columnspacing=1.5, handlelength=2.2)
ax2.set_title("Training-loss gaps between runs")
fs.add_letters(fig, [ax1, ax2], ["a", "b"])
figalign.finalize(fig)
figcheck.check(fig, "fig_gap")
fs.save(fig, "fig_gap")
print("fig_gap written")
