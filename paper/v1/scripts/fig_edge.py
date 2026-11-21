"""Figure: cost of adaptation on one 16 GB consumer GPU (descriptive; outside the registered EDGE block).

(a) GPU board energy of the 96-step full adaptations of gate G1 (25.2M tokens each): the seven MoE runs without
activation checkpointing and the two NESTOR runs, which fit only with it; the selection repeats edge_pre_freeze.py
and must reproduce its n and medians. (b) 1 Hz board-power traces of one run of each (fig_edge_values.json from
fig_edge_extract.py). (c) Context extension of stage C (48 steps of 1,048,576 tokens): energy of the completed
dense and MoE runs at 8,192 and 16,384 tokens, energy of the attempts that ran out of memory, and NESTOR, for which
every attempt ran out of memory. Board power only (no CPU, RAM or PSU losses).
Inputs: data/edge_pre_freeze.json, figures/fig_edge_values.json (data/ is searched first)
Output: figures/fig_edge.pdf, figures/fig_edge.png (FIG_OUT overrides the folder)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

fs.apply()
C = fs.COLOR
E = fs.load("edge_pre_freeze.json")
_v = [p for p in (fs.DATADIR / "fig_edge_values.json", fs.ROOT / "figures" / "fig_edge_values.json") if p.exists()]
V = json.load(open(_v[0], encoding="utf-8"))["runs"]
HATCH = "//////"

# ---- (a) the full 96-step adaptations, selected exactly as edge_pre_freeze.py does ----
seg = [s for s in E["segments"] if s["source"] in ("G1", "G2")]
has_c = {s["job"] for s in seg if s["segment"] == "m2c"}
full = [s for s in seg if any(k in s["job"] for k in ("_full_", "_eall_", "_dclmcont_"))
        and s["segment"] in ("m2", "m2c") and not (s["segment"] == "m2" and s["job"] in has_c)]
moe = sorted(s["energy_wh"] for s in full if s["arch"] == "moe_100m")
nes = sorted(s["energy_wh"] for s in full if s["arch"] == "nestor_100m" and s["segment"] == "m2c")
ref = E["adaptation_96_steps"]
assert len(moe) == ref["moe_full_runs_no_ckpt"]["n"] == 7 and len(nes) == ref["nestor_full_runs_ckpt"]["n"] == 2
med_m, med_n = float(np.median(moe)), float(np.median(nes))
assert abs(med_m - ref["moe_full_runs_no_ckpt"]["energy_wh"][1]) < 1e-9
assert abs(med_n - ref["nestor_full_runs_ckpt"]["energy_wh"][1]) < 1e-9
ratio = med_n / med_m

# ---- (c) context extension rows ----
sc = [r for r in E["stageC"] if r.get("kind") == "train"]
def cell(arch, seq):
    rows = [r for r in sc if r["arch"] == arch and int(r["seq"]) == seq]
    done = [r["energy_wh"] for r in rows if r["outcome"] == "done"]
    oom = [r["energy_wh"] for r in rows if r["outcome"] == "OOM"]
    assert len(done) <= 1 and all(r["outcome"] in ("done", "OOM") for r in rows)
    return (done[0] if done else None), float(sum(oom)), len(oom)

fig = fs.figure(2.6)
gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.80, 1.40])
gsb = gs[1].subgridspec(2, 1, hspace=0.12)
ax = fig.add_subplot(gs[0])
b1 = fig.add_subplot(gsb[0])
b2 = fig.add_subplot(gsb[1], sharex=b1, sharey=b1)
cx = fig.add_subplot(gs[2])

# (a) raw points (deterministic swarm) with a median tick per architecture
def swarm(vals, scale_wh, step=0.13):
    placed = []
    for v in vals:
        for k in range(0, 12):
            dx = ((k + 1) // 2) * step * (1 if k % 2 else -1)
            if all(abs(dx - px) >= step * 0.99 or abs(v - pv) >= scale_wh for px, pv in placed):
                placed.append((dx, v))
                break
    return [p[0] for p in placed]
for x0, vals, med, col in ((0.0, moe, med_m, C["moe"]), (1.0, nes, med_n, C["nestor"])):
    ax.plot([x0 - 0.46, x0 + 0.46], [med, med], color=fs.INK, lw=1.3, solid_capstyle="butt", zorder=2)
    ax.plot([x0 + dx for dx in swarm(vals, 7.0)], vals, ls="none", marker="o", ms=4.6, mfc=col, mec="white",
            mew=0.5, zorder=3)
ax.text(0.52, med_m, f"median\n{med_m:.1f} Wh", ha="left", va="center", fontsize=fs.SIZE_SECOND, linespacing=1.15)
ax.text(0.48, med_n, f"median\n{med_n:.1f} Wh", ha="right", va="center", fontsize=fs.SIZE_SECOND, linespacing=1.15)
ax.text(1.0, med_n - 16, "with activation\ncheckpointing", ha="center", va="top", fontsize=fs.SIZE_TICK,
        linespacing=1.15)
ax.set_xlim(-0.62, 1.62)
ax.set_ylim(0, 172)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"MoE\n{len(moe)} runs", f"NESTOR\n{len(nes)} runs"])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Board energy per run (Wh)")
ax.set_title("Energy per adaptation run")

# (b) board-power traces, shared axes
for bx, key, label in ((b1, "moe", "MoE"), (b2, "nestor", "NESTOR")):
    r = V[key]
    t, p = np.asarray(r["t_min"]), np.asarray(r["power_w"])
    bx.plot(t, p, color=C[key], lw=0.45, zorder=2)
    bx.plot([0, r["minutes"]], [r["mean_w"]] * 2, color=fs.INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    if key == "moe":   # one label per drawn element: the trace (name, duration) and the dashed mean line
        bx.text(r["minutes"] + 2.5, 246, f"{label}, {r['minutes']:.1f} min", ha="left", va="center",
                fontsize=fs.SIZE_SECOND)
        bx.text(r["minutes"] + 2.5, r["mean_w"] - 12, f"mean {r['mean_w']:.0f} W", ha="left", va="center",
                fontsize=fs.SIZE_TICK)
    else:
        bx.text(r["minutes"], 250, f"{label}, {r['minutes']:.1f} min", ha="right", va="center",
                fontsize=fs.SIZE_SECOND)
        bx.text(r["minutes"], 178, f"mean {r['mean_w']:.0f} W", ha="right", va="center", fontsize=fs.SIZE_TICK)
    bx.set_ylim(0, 270)
    bx.set_yticks([0, 100, 200])
b1.set_xlim(0, 82)
b1.set_xticks([0, 20, 40, 60, 80])
b1.tick_params(axis="x", labelbottom=False)
b2.set_xlabel("Minutes since the run started")
b2.set_ylabel("Board power (W)")
b2.yaxis.set_label_coords(-0.2, 1.06)
b1.set_title("Board power during one run")

# (c) context extension
xs = {("dense_100m", 8192): 0.0, ("dense_100m", 16384): 1.0, ("moe_100m", 8192): 2.6, ("moe_100m", 16384): 3.6,
      ("nestor_100m", 8192): 5.2, ("nestor_100m", 16384): 6.2}
colk = {"dense_100m": "dense", "moe_100m": "moe", "nestor_100m": "nestor"}
for (arch, seq), x in xs.items():
    done, oom, n_oom = cell(arch, seq)
    col = C[colk[arch]]
    base = 0.0
    if done is not None:
        cx.bar(x, done, width=0.78, color=col, edgecolor=col, lw=0.6, zorder=2)
        base = done
    if n_oom:
        cx.bar(x, oom, bottom=base, width=0.78, facecolor="white", edgecolor=col, hatch=HATCH, lw=0.6, zorder=2)
    if done is None:
        cx.text(x, oom + 8, "exceeds 16 GB", rotation=90, ha="center", va="bottom", fontsize=fs.SIZE_SECOND)
cx.set_xticks(list(xs.values()))
cx.set_xticklabels(["8k", "16k"] * 3)
cx.tick_params(axis="x", length=0)
for xm, name in ((0.5, "dense"), (3.1, "MoE"), (5.7, "NESTOR")):
    cx.text(xm, -0.10, name, transform=cx.get_xaxis_transform(), ha="center", va="top", fontsize=fs.SIZE_SECOND)
cx.set_xlim(-0.6, 6.8)
cx.set_ylim(0, 285)
cx.set_ylabel("Board energy per run (Wh)")
cx.set_title("Context extension within 16 GB")
cx.legend(handles=[Patch(facecolor=fs.INK, edgecolor=fs.INK, lw=0.6, label="completed run"),
                   Patch(facecolor="white", edgecolor=fs.INK, hatch=HATCH, lw=0.6, label="out-of-memory attempts")],
          loc="upper left", fontsize=fs.SIZE_TICK, handlelength=1.4, handleheight=1.0, borderaxespad=0.2)

fs.add_letters(fig, [ax, b1, cx], ["a", "b", "c"], align=True)
# The three panels are too narrow for their titles when a title starts at the left spine (the house default),
# so in this figure each title starts 6 pt after its panel letter. add_letters has frozen the layout, so moving
# the titles cannot move the axes or the letters.
_r = fig.canvas.get_renderer()
_letters = {t.get_text(): t for t in fig.texts if t.get_text() in ("a", "b", "c")}
for _a, _l in ((ax, "a"), (b1, "b"), (cx, "c")):
    _t = [t for t in (_a.title, _a._left_title, _a._right_title) if t.get_text()][0]
    _dx = _letters[_l].get_window_extent(_r).x1 + 6.0 * fig.dpi / 72.0 - _t.get_window_extent(_r).x0
    _t.set_x(_t.get_position()[0] + _dx / _a.get_window_extent(_r).width)
figalign.finalize(fig)
figcheck.check(fig, "fig_edge")
fs.save(fig, "fig_edge")
print(f"fig_edge written (energy ratio {ratio:.3f}, medians {med_m:.2f} / {med_n:.2f} Wh)")
