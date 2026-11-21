"""Optical alignment for the paper v1 figures (used by the figure scripts, figstyle.add_letters and figcheck.py).

matplotlib's ha/va='center' centres the layout box of a text (ascender and descender space, side bearings, and in
the PNG backend metrics rounded to whole pixels), not the ink of its glyphs, so a symbol such as 'g' (descender) or
the sum sign (no descender) sits visibly off the centre of the circle or box that holds it.

ink_extents(s, prop, ismath)
    Ink box of a string in points relative to its origin (baseline, start of the advance), from the unhinted glyph
    outlines of matplotlib.textpath.TextToPath (the outlines the vector PDF embeds), with the text's own font
    properties and mathtext parsing.
centered_text(ax, x, y, s, correction=(dx, dy), **text_kw)
    Draws s with ha='left', va='baseline', whose origin lands exactly on the anchor in every backend whatever the
    font metrics or hinting, plus a ScaledTranslation in points, so that the centre of the ink box falls on (x, y)
    (data coordinates unless transform= is given) moved by the optical correction (points). Multi-line strings are
    drawn line by line, each line centred on its own ink, with the line pitch matplotlib would use. In the PNG
    backend these texts are drawn as their filled glyph outlines (matplotlib's text-as-path route), because that
    backend hints glyphs, rounds text positions to whole pixels and truncates mathtext glyphs to whole pixels; the
    PDF keeps them as text.
finalize(fig)
    After the layout is frozen (after figstyle.add_letters): moves each legend handle vertically onto the optical
    centre line of its label, the cap-height centre above the baseline (descenders do not move it), compensates the
    vector-backend mathtext shift (below), and marks the figure as aligned, which makes figcheck.py enforce the
    tolerance TARGET_PT.
Vector mathtext shift. The PDF, PS and SVG backends lay out mathtext at 72 dpi and round the box height up to whole
    points (matplotlib's to_vector), so they draw a mathtext string ceil(h) - h points (0 to 1 pt) above the baseline
    that the layout, the PNG backend and TextToPath use. finalize() measures that shift for every single-line
    mathtext text from the parser those backends use and lowers the text by it in vector output only.
measure(fig)
    The deviations that figcheck.py reports, from where matplotlib actually draws each text (renderer.draw_text is
    recorded and the outlines are placed exactly as matplotlib's text-as-path renderer places them):
    in-shape texts |ink centre - shape centre| (registered targets of centered_text, otherwise the smallest patch
    that contains the text's ink and no other text), legend handles |handle centre - label centre line|, and
    panel letters |letter baseline - baseline of the first line of the panel title|.
"""
from __future__ import annotations

import types

from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.textpath import TextToPath
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation

T2P = TextToPath()
_MEASURING = [False]                     # True while _records() replays a text's drawing for measurement
TARGET_PT = 0.25                          # tolerance of the optical check, points on each axis
SHAPES = (Rectangle, FancyBboxPatch, Circle, Ellipse)
LETTER_PT = 10                            # figstyle.SIZE_LETTER (kept here to avoid importing figstyle)


def ink_extents(s, prop, ismath=False):
    """(x0, y0, x1, y1) of the glyph ink in points relative to the origin of s; None for blank strings."""
    if not s or not s.strip():
        return None
    verts, codes = T2P.get_text_path(prop, s, ismath=ismath)
    if len(verts) == 0:
        return None
    e = Path(verts, codes).get_extents()
    k = prop.get_size_in_points() / T2P.FONT_SCALE
    return e.x0 * k, e.y0 * k, e.x1 * k, e.y1 * k


def _metrics(s, prop, ismath):
    if not s.strip():
        return 0.0, 0.0, 0.0
    return T2P.get_text_width_height_descent(s, prop, ismath)


def centered_text(ax, x, y, s, *, correction=(0.0, 0.0), transform=None, linespacing=1.2, **kw):
    """Text whose ink box is centred on (x, y), moved by correction (points). Returns the Text or a list of lines."""
    fig = ax.figure
    base = ax.transData if transform is None else transform
    for k in ("ha", "va", "horizontalalignment", "verticalalignment", "multialignment"):
        kw.pop(k, None)
    lines = s.split("\n")
    texts = [ax.text(x, y, line, ha="left", va="baseline", transform=base, **kw) for line in lines]
    prop = texts[0].get_fontproperties()
    parsed = [t._preprocess_math(line) for t, line in zip(texts, lines)]
    inks = [ink_extents(c, prop, m) for c, m in parsed]
    _, lp_h, lp_d = _metrics("lp", prop, False)          # line pitch as in matplotlib.text.Text._get_layout
    min_dy = (lp_h - lp_d) * linespacing
    bases, yb, prev_d = [], 0.0, 0.0
    for i, (c, m) in enumerate(parsed):
        _, h, d = _metrics(c, prop, m)
        h, d = max(h, lp_h), max(d, lp_d)
        if i:
            yb -= prev_d + max(min_dy, (h - d) * linespacing)
        bases.append(yb)
        prev_d = d
    spans = [(e[1] + b, e[3] + b) for e, b in zip(inks, bases) if e]
    cy = (min(a for a, _ in spans) + max(b for _, b in spans)) / 2 if spans else 0.0
    corr = (float(correction[0]), float(correction[1]))
    for t, e, b in zip(texts, inks, bases):
        cx = (e[0] + e[2]) / 2 if e else 0.0
        t.set_transform(base + ScaledTranslation((corr[0] - cx) / 72.0, (corr[1] + b - cy) / 72.0,
                                                 fig.dpi_scale_trans))
        t._optical = {"target": (x, y), "base": base, "correction": corr, "group": id(texts[0])}
        t.draw = types.MethodType(_draw_aligned, t)
    return texts[0] if len(texts) == 1 else texts


def _records(t, renderer):
    """The renderer.draw_text calls that drawing t makes: (x, y, s, prop, angle, ismath), y as passed."""
    rec = []

    def draw_text(gc, x, y, s, prop, angle, ismath=False, mtext=None):
        rec.append((x, y, s, prop.copy(), angle, ismath))

    renderer.draw_text = draw_text
    _MEASURING[0] = True
    try:
        t.draw(renderer)
    finally:
        _MEASURING[0] = False
        del renderer.draw_text
    return rec


def ink_bbox(t, renderer):
    """Ink of a drawn text in display pixels (Bbox), or None."""
    H = renderer.get_canvas_width_height()[1]
    flip = renderer.flipy()
    boxes = []
    for x, y, s, prop, angle, ismath in _records(t, renderer):
        if ismath == "TeX" or not s.strip():
            continue
        verts, codes = T2P.get_text_path(prop, s, ismath=bool(ismath))
        if len(verts) == 0:
            continue
        k = renderer.points_to_pixels(prop.get_size_in_points()) / T2P.FONT_SCALE
        tr = Affine2D().scale(k).rotate_deg(angle).translate(x, H - y if flip else y)
        boxes.append(Path(verts, codes).transformed(tr).get_extents())
    return Bbox.union(boxes) if boxes else None


def baselines(t, renderer):
    """Display y of the baseline of each drawn line of t, in drawing order (top line first)."""
    H = renderer.get_canvas_width_height()[1]
    flip = renderer.flipy()
    return [H - y if flip else y for x, y, *_ in _records(t, renderer)]


def label_centre(t, renderer):
    """Optical centre line of a text label in display y: its first baseline plus half the cap height."""
    b = baselines(t, renderer)
    if not b:
        return None
    e = ink_extents("H", t.get_fontproperties(), False)
    return b[0] + renderer.points_to_pixels((e[1] + e[3]) / 2)


def _legends(fig):
    out = list(fig.legends)
    out += [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return out


def _legend_pairs(leg):
    """(DrawingArea of the handle, label Text) for every legend entry."""
    pairs = []
    for hp in leg.findobj(lambda a: isinstance(a, HPacker), include_self=False):
        ch = hp.get_children()
        da = [c for c in ch if isinstance(c, DrawingArea)]
        ta = [c for c in ch if isinstance(c, TextArea)]
        if len(ch) == 2 and len(da) == 1 and len(ta) == 1:
            pairs.append((da[0], ta[0]._text))
    return pairs


def _handle_centre(da, renderer):
    ys = []
    for a in da.get_children():
        if not a.get_visible():
            continue
        if isinstance(a, Line2D):
            ys.append(float(a.get_transform().transform(a.get_xydata())[:, 1].mean()))
        else:
            b = a.get_window_extent(renderer)
            ys.append((b.y0 + b.y1) / 2)
    return sum(ys) / len(ys) if ys else None


def _is_raster(renderer):
    from matplotlib.backends.backend_agg import RendererAgg
    return isinstance(getattr(renderer, "_renderer", renderer), RendererAgg) or isinstance(renderer, RendererAgg)


def vector_math_shift(s, prop):
    """Points by which the vector backends draw mathtext s above the position TextToPath and the layout assume."""
    vt = T2P.mathtext_parser.parse(s, 72, prop)                    # the parser RendererPdf/PS/SVG use
    p100 = prop.copy()
    p100.set_size(T2P.FONT_SCALE)
    v100 = T2P.mathtext_parser.parse(s, 72, p100)                  # what TextToPath (and ink_extents) use
    if not vt.glyphs or not v100.glyphs:
        return 0.0
    # glyph tuples end with (ox, oy); oy is the baseline position in the vector output (upwards)
    return float(vt.glyphs[0][-1] - v100.glyphs[0][-1] * prop.get_size_in_points() / T2P.FONT_SCALE)


def _draw_aligned(self, renderer):
    """Raster: centred texts as glyph outlines. Vector: mathtext lowered by its vector shift."""
    if _is_raster(renderer):
        if getattr(self, "_optical", None) and not _MEASURING[0] and not self.get_path_effects():
            from matplotlib import patheffects
            self.set_path_effects([patheffects.Normal()])
            try:
                return Text.draw(self, renderer)
            finally:
                self.set_path_effects([])
        return Text.draw(self, renderer)
    if not getattr(self, "_math_shift", 0.0):
        return Text.draw(self, renderer)
    base = self.get_transform()
    self.set_transform(base + ScaledTranslation(0.0, -self._math_shift / 72.0, self.figure.dpi_scale_trans))
    try:
        return Text.draw(self, renderer)
    finally:
        self.set_transform(base)


def compensate_vector_mathtext(fig):
    """Lower every single-line mathtext text by its vector shift in PDF/PS/SVG output; returns {text: shift_pt}."""
    out = {}
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s.strip() or "\n" in s or getattr(t, "_math_shift", None) is not None:
            continue
        clean, ismath = t._preprocess_math(s)
        if ismath is not True:
            continue
        sh = vector_math_shift(clean, t.get_fontproperties())
        if abs(sh) > 1e-6:
            t._math_shift = sh
            t.draw = types.MethodType(_draw_aligned, t)
            out[s] = round(sh, 3)
    return out


def finalize(fig):
    """Align every legend handle with the centre line of its label, compensate the vector mathtext shift, and mark
    the figure as optically aligned."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    shifts = []
    for leg in _legends(fig):
        for da, t in _legend_pairs(leg):
            hc, lc = _handle_centre(da, r), label_centre(t, r)
            if hc is None or lc is None:
                continue
            dy = (lc - hc) / fig.dpi                          # inches
            for a in da.get_children():                   # artist-level transform (a Patch adds its own)
                a.set_transform(Artist.get_transform(a) + ScaledTranslation(0.0, dy, fig.dpi_scale_trans))
            shifts.append((t.get_text(), round(dy * 72.0, 3)))
    fig._math_shifts = compensate_vector_mathtext(fig)
    fig._optical_aligned = True
    fig._legend_shifts = shifts
    return shifts


def _is_letter(t):
    return (len(t.get_text().strip()) == 1 and t.get_fontsize() == LETTER_PT
            and str(t.get_fontweight()) in ("bold", "700"))


def measure(fig, renderer=None):
    """Optical deviations in points: {'in_shape': [...], 'legend': [...], 'letters': [...]}."""
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    cands = [t for ax in fig.axes for t in ax.texts] + list(fig.texts)
    cands = [t for t in cands if t.get_visible() and t.get_text().strip()]
    inks = {t: ink_bbox(t, renderer) for t in cands}
    out = {"in_shape": [], "legend": [], "letters": []}
    groups = {}
    for t in cands:
        if getattr(t, "_optical", None):
            groups.setdefault(t._optical["group"], []).append(t)
    for ts in groups.values():
        bbs = [inks[t] for t in ts if inks[t] is not None]
        if not bbs:
            continue
        b, o = Bbox.union(bbs), ts[0]._optical
        tx, ty = o["base"].transform(o["target"])
        rx, ry = ((b.x0 + b.x1) / 2 - tx) * k, ((b.y0 + b.y1) / 2 - ty) * k
        out["in_shape"].append({"text": "\n".join(t.get_text() for t in ts), "kind": "centered_text",
                                "raw": [round(rx, 3), round(ry, 3)], "correction": list(o["correction"]),
                                "dev": [round(rx - o["correction"][0], 3), round(ry - o["correction"][1], 3)]})
    shapes = [(p, p.get_window_extent(renderer)) for ax in fig.axes for p in ax.patches
              if isinstance(p, SHAPES) and p.get_visible() and p is not ax.patch]
    placed = [(t, inks[t]) for t in cands if inks[t] is not None]
    for t, ib in placed:
        if (getattr(t, "_optical", None) or t.get_rotation() != 0 or _is_letter(t)
                or t.get_horizontalalignment() != "center"):
            continue                                          # only texts meant to be centred in a shape
        cont = [(p, b) for p, b in shapes if b.x0 - 0.5 <= ib.x0 and ib.x1 <= b.x1 + 0.5
                and b.y0 - 0.5 <= ib.y0 and ib.y1 <= b.y1 + 0.5]
        if not cont:
            continue
        p, b = min(cont, key=lambda pb: pb[1].width * pb[1].height)
        if any(u is not t and b.x0 <= (ub.x0 + ub.x1) / 2 <= b.x1 and b.y0 <= (ub.y0 + ub.y1) / 2 <= b.y1
               for u, ub in placed):
            continue                                          # a shape holding several texts is a container
        d = [round(((ib.x0 + ib.x1) - (b.x0 + b.x1)) / 2 * k, 3), round(((ib.y0 + ib.y1) - (b.y0 + b.y1)) / 2 * k, 3)]
        out["in_shape"].append({"text": t.get_text(), "kind": "auto:" + type(p).__name__, "raw": d,
                                "correction": [0.0, 0.0], "dev": d})
    for leg in _legends(fig):
        for da, t in _legend_pairs(leg):
            hc, lc = _handle_centre(da, renderer), label_centre(t, renderer)
            if hc is not None and lc is not None:
                out["legend"].append({"text": t.get_text(), "dev": round((hc - lc) * k, 3)})
    titles = [t for ax in fig.axes for t in (ax.title, ax._left_title, ax._right_title)
              if t.get_visible() and t.get_text().strip()]
    titles += [t for t in cands if t.get_fontsize() == LETTER_PT - 1 and not _is_letter(t)
               and t.get_horizontalalignment() == "left" and str(t.get_fontweight()) in ("normal", "400")]
    tb = {t: baselines(t, renderer) for t in titles}
    for L in [t for t in cands if _is_letter(t)]:
        bl = baselines(L, renderer)
        if not bl:
            continue
        ref = getattr(L, "_optical_title", None)
        if ref is None:
            lb = L.get_window_extent(renderer)
            near = [(t.get_window_extent(renderer).x0 - lb.x1, t) for t in titles
                    if tb[t] and t.get_window_extent(renderer).x0 >= lb.x1 - 2 and abs(tb[t][0] - bl[0]) * k < 15]
            ref = min(near, key=lambda n: n[0])[1] if near else None
        if ref is not None and tb.get(ref, baselines(ref, renderer)):
            rb = tb.get(ref) or baselines(ref, renderer)
            out["letters"].append({"letter": L.get_text(), "title": ref.get_text().split("\n")[0][:40],
                                   "dev": round((bl[0] - rb[0]) * k, 3)})
    return out


def summary(rep):
    """Max |deviation| per category (points): in-shape x, in-shape y, legend y, letters y."""
    m = lambda xs: round(max([abs(v) for v in xs], default=0.0), 3)
    return {"in_shape_dx": m([e["dev"][0] for e in rep["in_shape"]]),
            "in_shape_dy": m([e["dev"][1] for e in rep["in_shape"]]),
            "legend_dy": m([e["dev"] for e in rep["legend"]]),
            "letters_dy": m([e["dev"] for e in rep["letters"]]),
            "n": [len(rep["in_shape"]), len(rep["legend"]), len(rep["letters"])]}
