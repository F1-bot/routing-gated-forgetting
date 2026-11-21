"""Shared figure style for paper v1 (every figure imports this module).

Rules: one font family and one size ladder for all figures; fixed colours per
architecture (dense grey, MoE blue, NESTOR orange; Okabe-Ito, colour-blind safe);
figures are drawn at the template's text width and saved WITHOUT bbox cropping,
so a figure included at its natural size prints with exactly these font sizes.
The rcParams mirror apply_figure_style(sizes=(9, 8, 7)) of the figure-style
checklist, except savefig.bbox, which stays 'standard' for the reason above.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGDIR = Path(os.environ.get("FIG_OUT", str(ROOT / "figures")))  # FIG_OUT: test output folder (default figures/)
DATADIR = ROOT / "data"

TEXT_WIDTH_MM = 159.2                  # acnsart: A4 paper, 1 in side margins (geometry hmargin=1in)
TEXT_WIDTH = TEXT_WIDTH_MM / 25.4      # inches
FONT = "DejaVu Sans"                   # bundled with matplotlib, identical on every machine
SIZE_BASE, SIZE_SECOND, SIZE_TICK = 9, 8, 7   # titles/axis labels/series labels; legends/annotations; ticks
SIZE_LETTER = 10                       # panel letters only

COLOR = {"dense": "#7F7F7F", "moe": "#0072B2", "nestor": "#E69F00"}
INK = "#222222"
GREY = "#888888"                       # reference lines, thresholds, secondary text
LIGHT = "#E6E6E6"                      # shaded regions
DASH_SEED43 = (0, (4, 2))              # MoE base 2 (seed 43) is dashed wherever a line is drawn
LOCKED: list = []                      # figure files that save() could not replace (see save)


def apply() -> None:
    """Set the shared rcParams. Call once before creating any figure."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "mathtext.fontset": "dejavusans",
        "font.size": SIZE_BASE,
        "axes.titlesize": SIZE_BASE,
        "axes.labelsize": SIZE_BASE,
        "legend.fontsize": SIZE_SECOND,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "axes.titlelocation": "left",
        "axes.titleweight": "normal",
        "axes.titlepad": 5.0,
        "axes.labelpad": 3.0,
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    })


def figure(height_in: float, width_in: float = TEXT_WIDTH, **kw):
    """A constrained-layout figure at the text width (default)."""
    return plt.figure(figsize=(width_in, height_in), layout="constrained", **kw)


def add_letters(fig, axes, letters, align: bool = False) -> None:
    """Bold panel letters, top-left, outside each axes box, on the baseline of the first line of the panel title.

    align=True puts all letters on one line, the first-line baseline of the highest title (one row of panels).
    Without any titled panel the letters keep their tops level with the top of the axes' tight box. The layout is
    drawn twice (constrained layout converges in two passes with fixed-aspect axes) and then frozen, so saving
    cannot move the axes away from the letters. Each letter records the title it is aligned to (_optical_title),
    which figcheck.py's optical check uses."""
    import figalign
    fig.canvas.draw()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    pos = []
    for ax in axes:
        bb = ax.get_tightbbox(r)
        x0 = min(bb.x0, ax.get_window_extent(r).x0)
        titles = [t for t in (ax.title, ax._left_title, ax._right_title) if t.get_text()]
        if titles:
            ref = max(titles, key=lambda t: figalign.baselines(t, r)[0])
            pos.append((x0 / W, figalign.baselines(ref, r)[0] / H, ref, bb.y1 / H))
        else:
            pos.append((x0 / W, None, None, bb.y1 / H))
    titled = [p for p in pos if p[2] is not None]
    if titled:
        row = max(titled, key=lambda p: p[1])
        for (x0, yb, ref, _), letter in zip(pos, letters):
            y, rr = (row[1], row[2]) if (align or ref is None) else (yb, ref)
            t = fig.text(max(x0, 0.0), y, letter, fontsize=SIZE_LETTER, fontweight="bold", ha="left", va="baseline")
            t._optical_title = rr
    else:
        ytop = max(p[3] for p in pos)
        for (x0, _, _, y1), letter in zip(pos, letters):
            fig.text(max(x0, 0.0), min(ytop if align else y1, 1.0), letter, fontsize=SIZE_LETTER,
                     fontweight="bold", ha="left", va="top")
    fig.set_layout_engine("none")


def signed(x: float, nd: int = 3) -> str:
    """Format with an explicit sign and a true minus sign (U+2212)."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "\N{MINUS SIGN}")


def load(name: str):
    with open(DATADIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, name: str) -> None:
    """Write figures/<name>.pdf (vector) and figures/<name>.png (300 dpi) via temp files.

    A target that another program holds open is left untouched; the new file is then kept next to it as
    <name>.new.<ext> and the name is recorded in LOCKED (and printed)."""
    FIGDIR.mkdir(exist_ok=True)
    meta_pdf = {"Creator": "paper v1 scripts", "Producer": "matplotlib", "CreationDate": None}
    for ext, meta in (("pdf", meta_pdf), ("png", {"Software": "matplotlib"})):
        final = FIGDIR / f"{name}.{ext}"
        tmp = FIGDIR / f"{name}.tmp.{ext}"
        fig.savefig(tmp, format=ext, metadata=meta)
        try:
            os.replace(tmp, final)
        except PermissionError:          # target open in another program (a viewer or a running build)
            alt = FIGDIR / f"{name}.new.{ext}"
            os.replace(tmp, alt)
            LOCKED.append(final.name)
            print(f"LOCKED {final.name}: the new file is kept next to it as {alt.name}")
