"""Automated layout check for the paper v1 figures (called by every figure script after fig.canvas.draw()).

Collects every visible, non-empty Text artist of the figure (fig.findobj(matplotlib.text.Text): panel letters,
titles, axis labels, tick labels, legend texts, annotations), takes its window extent in display pixels at the
figure's dpi, and stops the script with exit status 3 if
  (1) two text boxes intersect after each box is padded by PAD_PX = 1 pixel,
  (2) a text box crosses the figure boundary, or
  (3) a text is set below MIN_PT = 7 pt at the final printed width, i.e. font size x (printed width / figure
      width); the figures are drawn at the text width and included at \linewidth, so the factor is 1
      (PRINT_WIDTH_IN overrides the printed width).
Texts that matplotlib does not draw are skipped: tick labels outside the view limits, labels switched off, and the
tick labels of axes whose axis is turned off. Mathtext sub- and superscripts are part of their Text artist and are
judged by that artist's font size.
On success it prints 'layout check <name>: overlaps 0, outside 0, below 7 pt 0'.
It then runs the optical check (optical(), measurements in figalign.py) and prints 'optical check <name>: ...' with
the largest deviation of ink centre from shape centre over in-shape texts (x and y), of legend handles from the
centre line of their labels, and of panel letters from the baseline of their titles, in points.
"""
from __future__ import annotations

import os

import matplotlib.text as mtext
from matplotlib.transforms import Bbox

PAD_PX = 1.0
MIN_PT = 7.0


def _drawn_tick_labels(ax):
    out = set()
    for axis in (ax.xaxis, ax.yaxis):
        if not axis.get_visible():
            continue
        lo, hi = sorted(axis.get_view_interval())
        tol = 1e-9 * max(1.0, abs(hi - lo))
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            loc = tick.get_loc()
            if loc is None or not (lo - tol <= loc <= hi + tol):
                continue
            for lab in (tick.label1, tick.label2):
                if lab.get_visible() and lab.get_text().strip():
                    out.add(lab)
    return out


def check(fig, name: str, print_width_in: float | None = None) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    all_ticks, drawn_ticks = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {tick.label1, tick.label2}
        if ax.get_visible() and ax.axison:
            drawn_ticks |= _drawn_tick_labels(ax)
    texts = []
    for t in fig.findobj(mtext.Text):
        if not t.get_text().strip() or not t.get_visible():
            continue
        if t in all_ticks and t not in drawn_ticks:
            continue
        a = t.axes
        if a is not None and not a.get_visible():
            continue
        texts.append((t, t.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    pw = print_width_in or float(os.environ.get("PRINT_WIDTH_IN", fig.get_figwidth()))
    scale = pw / fig.get_figwidth()
    pad = [Bbox.from_extents(b.x0 - PAD_PX, b.y0 - PAD_PX, b.x1 + PAD_PX, b.y1 + PAD_PX) for _, b in texts]
    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            gi, gj = getattr(texts[i][0], "_optical", None), getattr(texts[j][0], "_optical", None)
            if gi and gj and gi["group"] == gj["group"]:
                # lines of one centred text block (figalign.centered_text draws a multi-line string line by line):
                # their layout boxes overlap as inside one multi-line Text; their ink must not
                import figalign
                a, b = figalign.ink_bbox(texts[i][0], r), figalign.ink_bbox(texts[j][0], r)
                if a is not None and b is not None and a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
                    overlaps.append((texts[i][0].get_text()[:40], texts[j][0].get_text()[:40]))
                continue
            p, q = pad[i], pad[j]
            if p.x0 < q.x1 and q.x0 < p.x1 and p.y0 < q.y1 and q.y0 < p.y1:
                overlaps.append((texts[i][0].get_text()[:40], texts[j][0].get_text()[:40]))
    outside = [t.get_text()[:40] for t, b in texts if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H]
    small = [(t.get_text()[:40], round(t.get_fontsize() * scale, 2)) for t, _ in texts
             if t.get_fontsize() * scale < MIN_PT - 1e-9]
    res = {"texts": len(texts), "overlaps": len(overlaps), "outside": len(outside), "below_7pt": len(small)}
    print(f"layout check {name}: overlaps {len(overlaps)}, outside {len(outside)}, below 7 pt {len(small)} "
          f"({len(texts)} texts)")
    if overlaps or outside or small:
        for o in overlaps:
            print("  overlap:", o)
        for o in outside:
            print("  outside the figure:", o)
        for o in small:
            print("  below 7 pt:", o)
        raise SystemExit(3)
    res["optical"] = optical(fig, name, r)
    return res


def optical(fig, name: str, renderer=None) -> dict:
    """Optical alignment report (figalign.measure), in points: max |ink centre - shape centre| over in-shape texts
    (x and y), max |handle centre - label centre line| over legend entries, and max |letter baseline - title
    baseline| over panel letters. Report-only for figures that do not use figalign; a figure marked by
    figalign.finalize stops the script with exit status 3 if any deviation exceeds figalign.TARGET_PT (0.25 pt).
    OPTICAL_REPORT=<file.json> also writes the full report there (one entry per figure name)."""
    import json

    import figalign
    rep = figalign.measure(fig, renderer)
    s = figalign.summary(rep)
    print(f"optical check {name}: in-shape max |dx| {s['in_shape_dx']:.2f} pt, |dy| {s['in_shape_dy']:.2f} pt "
          f"({s['n'][0]} texts); legend handles max |dy| {s['legend_dy']:.2f} pt ({s['n'][1]}); "
          f"letters max |dy| {s['letters_dy']:.2f} pt ({s['n'][2]})")
    path = os.environ.get("OPTICAL_REPORT")
    if path:
        allrep = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        allrep[name] = {"summary": s, "report": rep, "aligned": bool(getattr(fig, "_optical_aligned", False))}
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(allrep, fh, indent=1, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    if getattr(fig, "_optical_aligned", False):
        tol = figalign.TARGET_PT
        bad = ([e for e in rep["in_shape"] if max(abs(e["dev"][0]), abs(e["dev"][1])) > tol]
               + [e for e in rep["legend"] if abs(e["dev"]) > tol] + [e for e in rep["letters"] if abs(e["dev"]) > tol])
        if bad:
            for e in bad:
                print(f"  optical deviation above {tol} pt:", e)
            raise SystemExit(3)
    return {"summary": s, "report": rep}
