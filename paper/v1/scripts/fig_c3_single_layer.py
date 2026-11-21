import os
import sys

sys.dont_write_bytecode = True                                  # never write __pycache__ into the paper tree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Figure fig_c3_layers: C3 single-layer (per-block) reverts, one full-adaptation run per domain.

Registered definition. Pre-registration v1.3 sec. 4.8, "C3 per-layer": for one FULL run per domain (seed 101;
4 domains), MoE base 1: 12 single-layer reverts (the routed experts of layer l back to theta_0); NESTOR: 6 single-block
reverts of the column-private group (blocks 0-5); each revert evaluated on DCLM with 2.048M tokens. "Reported.
Delta_l = F - F(revert l) and Delta_l / F per layer, the sum over layers against the all-expert (or all-column-private)
revert (additivity gap), and domain means with the domain range. With one seed per domain there is no CI; the figure is
descriptive." Never cut (v1.3 sec. 7, P-4); reported regardless (v1.3 sec. 9.2). MoE-versus-NESTOR comparisons are made
only at matched base loss (v1.2 sec. 10.1; blocks.C3.nestor_moe_comparison), so the two architectures get separate
panels with independent axes. If the routing-health label applies (v1.2 sec. 10.1, G0-9), C3-NESTOR is reported as
exploratory: the NESTOR panel title then says so (blocks.C3.NESTOR_label.label).

Panels. (a) MoE base 1 and (c) NESTOR: Delta_l / F per unit, mean over domains (marker) and range over domains (bar),
from blocks.C3_per_layer.per_base.<base>.per_unit. Units are numbered from 1 as in the paper: MoE units b<i>l<j> are layer
1-12 in block-then-layer order; NESTOR units b0-b5 (pre-registration blocks 0-5) are block 1-6, the numbering of Fig. 2
(sections/04_design.tex l. 23: blocks 1 to 7, the 7th the dense tail). (b, d) per domain, the sum of the single-unit deltas over the
delta of the whole-group revert, sum_over_all = sum_Delta / Delta_all of blocks.C3_per_layer.per_run (1: the single-unit
deltas add up to the whole-group delta); the additivity gap sum_Delta - Delta_all is printed in the text.

Input: ONLY the full-mode output of the registered analysis (analysis_confirm.json written by
`analyze_confirm.py run --mode full`; blocks.C3_per_layer from analyze_confirm_ext.block_c3_layer and the label of
blocks.C3). Every plotted number is read from that file; nothing is recomputed. The script checks the file (embedded
content hash, .sha256 sidecar, format, mode) and the block's internal consistency (per-unit means, ranges and domain
counts against the per-run values; sum_over_all against sum_Delta / Delta_all) and stops with exit status 4 when a
required key is missing or a value is not a finite number. Values that the analysis reports as null (a unit evaluation
missing in a domain, a ratio that is not defined) are not drawn and are listed on stdout; nothing is defaulted.

Real and mock renders: as fig_c4_per_window.py. Unmarked only from the registered output with a matching .sha256
sidecar, freeze_verification.status 'ok' and the frozen hashes -> figures/fig_c3_layers.pdf and .png; any other input
is a mock: stamped MOCK in a free corner, files fig_c3_layers_MOCK.pdf/.png, --out-dir required and never the
paper's figures folder; a verified analysis at another path is refused.

  python fig_c3_single_layer.py                                 # real output, after the full analysis
  python fig_c3_single_layer.py --input <mock analysis_confirm.json> --out-dir <folder>

Exit status: 0 written; 2 usage; 3 layout check (figcheck.py) or no free corner for the MOCK stamp; 4 required key
missing or invalid value; 5 input integrity or real/mock gate; 6 a target file was held open (new render kept as
<name>.new.pdf/.png, see figstyle.save).
"""
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_SCRIPTS = os.environ.get("PAPER_SCRIPTS", r"paper\v1\scripts")
if not os.path.exists(os.path.join(HERE, "figstyle.py")):
    if not os.path.exists(os.path.join(PAPER_SCRIPTS, "figstyle.py")):
        raise SystemExit(f"figstyle.py not found next to this script or in {PAPER_SCRIPTS} (set PAPER_SCRIPTS)")
    sys.path.insert(0, PAPER_SCRIPTS)

from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import figalign  # noqa: E402
import figcheck  # noqa: E402
import figstyle as fs  # noqa: E402

CONFIG = {
    "registered_output": r"results\analysis\analysis_confirm.json",
    "format": "analysis_confirm/1",
    "mode": "full",
    "freeze_record_sha256": "4f88f9dd29b0f1ed91643491715429846ac125b416b282c07d2a5b1caf2b6c1c",   # FREEZE.json sha256
    "prereg_sha256": "b01bf1dc824567926e4007d35006f32fe76939e14565ad8ca7b0a6f2dddebe3d",        # v1.3 final text
    "name": "fig_c3_layers",
    # base, label, colour key, reverted group (analysis name), unit word, number of units, unit pattern, numbering
    "bases": [("moe_b1", "MoE base 1", "moe", "moe", "layer", 12, r"^b(\d+)l(\d+)$", "ordinal"),
              ("nestor", "NESTOR", "nestor", "attn+priv", "block", 6, r"^b(\d+)$", "block_from_1")],
    "domains": {"books": "books", "code": "code", "german": "German", "math": "mathematics"},   # fill_full CORPUS names
    "height_in": 3.2,
    "rel_tol": 1e-9,
}
MOCK_COLOR = "#D55E00"                                                 # Okabe-Ito vermillion: the stamp only


# ============================================================================= input
def fail(code: int, msg: str):
    print(f"[fig_c3_single_layer] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def req(obj, path: str, at: str | None = None):
    """Value at a dotted key path below the object at path `at`; exit 4 if any key is missing (never a default)."""
    cur, done = obj, ([at] if at else [])
    for part in path.split("."):
        done.append(part)
        if isinstance(cur, dict):
            if part not in cur:
                fail(4, f"required key '{'.'.join(done)}' missing in the analysis output")
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            fail(4, f"required key '{'.'.join(done)}' missing in the analysis output")
    return cur


def num(x, where: str, allow_null: bool = False):
    """A finite float; null only where the analysis itself reports null (allow_null); exit 4 otherwise."""
    if x is None:
        if allow_null:
            return None
        fail(4, f"{where} is null")
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        fail(4, f"{where} is not a finite number: {x!r}")
    return float(x)


def place_mock_stamp(fig, clearance_pt: float = 4.0):
    """Stamp MOCK into the first figure corner (top-right, bottom-right, top-left, bottom-left) whose stamp box keeps
    clearance_pt from every text, axes area and legend; exit 3 if no corner is free. Call after the layout is frozen."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    obst = [t.get_window_extent(r) for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
    obst += [ax.get_window_extent(r) for ax in fig.axes] + [lg.get_window_extent(r) for lg in fig.legends]
    pad = clearance_pt * fig.dpi / 72.0
    for x, y, ha, va in ((0.997, 0.997, "right", "top"), (0.997, 0.003, "right", "bottom"),
                         (0.003, 0.997, "left", "top"), (0.003, 0.003, "left", "bottom")):
        t = fig.text(x, y, "MOCK", ha=ha, va=va, fontsize=10, fontweight="bold", color=MOCK_COLOR)
        b = t.get_window_extent(r)
        if not any(b.x0 - pad < q.x1 and q.x0 < b.x1 + pad and b.y0 - pad < q.y1 and q.y0 < b.y1 + pad for q in obst):
            return f"{va}-{ha}"
        t.remove()
    fail(3, "no free figure corner for the MOCK stamp")


def close(a: float, b: float) -> bool:
    return abs(a - b) <= CONFIG["rel_tol"] * max(1.0, abs(a), abs(b))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_input(path: str) -> tuple[dict, dict]:
    if not os.path.isfile(path):
        fail(5, f"input file not found: {path}")
    raw = open(path, "rb").read()
    doc = json.loads(raw.decode("utf-8"))
    info = {"path": os.path.abspath(path), "file_sha256": sha256_bytes(raw), "bytes": len(raw)}
    body = {k: v for k, v in doc.items() if k not in ("created_utc", "sha256")}
    emb = sha256_bytes(json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False).encode("utf-8"))
    if emb != req(doc, "sha256"):
        fail(5, "embedded content sha256 does not match the file content (not an unmodified analysis output)")
    side = path + ".sha256"
    info["sidecar"] = None
    if os.path.exists(side):
        tok = open(side, encoding="ascii").read().split()
        if not tok or tok[0] != info["file_sha256"]:
            fail(5, f"{side} does not match the file's SHA-256")
        info["sidecar"] = "match"
    if req(doc, "format") != CONFIG["format"]:
        fail(5, f"format {doc['format']!r}, expected {CONFIG['format']!r}")
    if req(doc, "mode") != CONFIG["mode"]:
        fail(5, f"mode {doc['mode']!r}: this figure reads only the full-mode output (analyze_confirm.py run --mode full)")
    info["created_utc"] = req(doc, "created_utc")
    info["verification"] = req(doc, "freeze_verification.status")
    return doc, info


def gate(info: dict, doc: dict) -> bool:
    """True: real render. False: mock render. Exit 5 when neither is allowed."""
    reg = os.path.normcase(os.path.realpath(CONFIG["registered_output"]))
    here = os.path.normcase(os.path.realpath(info["path"]))
    verified = info["verification"] == "ok"
    if here == reg:
        probs = []
        if not verified:
            probs.append(f"freeze_verification.status is {info['verification']!r}, not 'ok'")
        if info["sidecar"] != "match":
            probs.append("the .sha256 sidecar is missing")
        if req(doc, "freeze.sha256_record") != CONFIG["freeze_record_sha256"]:
            probs.append("freeze.sha256_record differs from the frozen record")
        if req(doc, "freeze.prereg_ref.sha256") != CONFIG["prereg_sha256"]:
            probs.append("freeze.prereg_ref.sha256 differs from the frozen pre-registration")
        if probs:
            fail(5, "registered output fails the real-render checks: " + "; ".join(probs))
        return True
    if verified:
        fail(5, "a verified full analysis must be rendered from its registered location "
                f"({CONFIG['registered_output']}); this copy would be stamped MOCK")
    return False


# ============================================================================= C3 per-layer block
def read_c3_layer(doc: dict) -> dict:
    pl = req(doc, "blocks.C3_per_layer")
    if not isinstance(pl, dict):
        fail(4, "blocks.C3_per_layer is not an object")
    if "status" in pl:
        fail(4, f"blocks.C3_per_layer has status {pl['status']!r}; no per-layer results to draw")
    W0 = "blocks.C3_per_layer"
    per_base, per_run, dropped = req(pl, "per_base", W0), req(pl, "per_run", W0), req(pl, "dropped", W0)
    label = req(doc, "blocks.C3.NESTOR_label.label")
    if not isinstance(label, str):
        fail(4, f"blocks.C3.NESTOR_label.label is not a string: {label!r}")
    names = [b[0] for b in CONFIG["bases"]]
    extra = sorted(set(per_base) - set(names))
    if extra:
        fail(4, f"blocks.C3_per_layer.per_base has bases outside the registered design: {extra}")
    notes = [f"dropped run {req(x, 'rid', f'{W0}.dropped.{i}')}: {req(x, 'reason', f'{W0}.dropped.{i}')}"
             for i, x in enumerate(dropped)]
    for i, r in enumerate(per_run):
        wr = f"{W0}.per_run.{i}"
        req(r, "rid", wr), req(r, "base", wr)
        if "layers" not in r:
            notes.append(f"run {req(r, 'rid', wr)} ({req(r, 'base', wr)}, {req(r, 'domain', wr)}): "
                         f"{req(r, 'status', wr)} (not in the figure)")
    out = {}
    for base, blabel, ckey, group, unit, n_units, pat, numbering in CONFIG["bases"]:
        w = f"blocks.C3_per_layer.per_base.{base}"
        if base not in per_base:
            fail(4, f"required key '{w}' missing (registered base without a usable run); dropped: {json.dumps(dropped)}")
        pb = per_base[base]
        if req(pb, "group", w) != group:
            fail(4, f"{w}.group is {pb['group']!r}, the registered reverted group is {group!r}")
        domains = req(pb, "domains", w)
        for dmn in domains:
            if dmn not in CONFIG["domains"]:
                fail(4, f"{w}.domains holds {dmn!r}, not a registered C3 domain {sorted(CONFIG['domains'])}")
        pu = req(pb, "per_unit", w)
        toks = list(pu)
        rx = re.compile(pat)
        bad = [t for t in toks if not rx.match(t)]
        if bad or len(toks) != n_units:
            fail(4, f"{w}.per_unit has units {toks}; the registered design has {n_units} {unit} units matching {pat}")
        toks.sort(key=lambda t: tuple(int(g) for g in rx.match(t).groups()))
        runs = [r for r in per_run if r["base"] == base and "layers" in r]
        for r in runs:
            for key in ("rid", "domain", "layers", "sum_Delta", "Delta_all", "sum_over_all", "all_group_revert"):
                req(r, key, f"{W0}.per_run ({r.get('rid')})")
        if sorted(r["domain"] for r in runs) != sorted(domains):
            fail(4, f"{w}.domains {domains} differ from the domains of its per-run records")
        units = []
        for k, t in enumerate(toks):
            y = pu[t]
            wu = f"{w}.per_unit.{t}"
            # units are numbered from 1 as in the paper: MoE layers 1-12 in block-then-layer order; NESTOR blocks as in
            # Fig. 2 (sections/04_design.tex l. 23, fig_arch.py: "blocks 1 to 7"), so pre-registration block b is block b+1
            name = str(k + 1) if numbering == "ordinal" else str(int(rx.match(t).group(1)) + 1)
            uname = f"{blabel} {unit} {name} ({t})"
            nd = req(y, "n_domains", wu)
            vals = []
            for r in runs:
                lay = r["layers"]
                if t not in lay:
                    fail(4, f"required key '{W0}.per_run ({r['rid']}).layers.{t}' missing in the analysis output")
                if lay[t] is None:
                    notes.append(f"{uname}: no revert evaluation in {r['domain']} (run {r['rid']})")
                    continue
                wl = f"{W0}.per_run ({r['rid']}).layers.{t}"
                q = num(req(lay[t], "Delta_over_F", wl), f"{wl}.Delta_over_F", allow_null=True)
                if q is None:
                    notes.append(f"{uname}: Delta_over_F null in {r['domain']} (F = 0)")
                else:
                    vals.append(q)
            if nd != sum(1 for r in runs if r["layers"][t] is not None):
                fail(4, f"{wu}.n_domains = {nd} does not match the per-run records")
            m = num(req(y, "Delta_over_F_mean", wu), f"{wu}.Delta_over_F_mean", allow_null=not vals)
            rg = req(y, "Delta_over_F_range", wu)
            if vals:
                lo, hi = num(req(y, "Delta_over_F_range.0", wu), f"{wu}.Delta_over_F_range[0]"), \
                    num(req(y, "Delta_over_F_range.1", wu), f"{wu}.Delta_over_F_range[1]")
                if not (close(m, sum(vals) / len(vals)) and close(lo, min(vals)) and close(hi, max(vals))):
                    fail(4, f"{wu}: mean or range of Delta_over_F differ from the per-run values")
            else:
                if m is not None or rg is not None:
                    fail(4, f"{wu}: Delta_over_F mean/range present without per-run values")
                lo = hi = None
                notes.append(f"{uname}: no domain with Delta_over_F (unit not drawn)")
            units.append({"tok": t, "name": name, "mean": m, "lo": lo, "hi": hi, "n": len(vals),
                          "full": len(vals) == len(domains)})
        ratios = []
        for r in runs:
            wr = f"{W0}.per_run ({r['rid']})"
            s = num(r["sum_Delta"], f"{wr}.sum_Delta", allow_null=True)
            a_ = num(r["Delta_all"], f"{wr}.Delta_all", allow_null=True)
            q = num(r["sum_over_all"], f"{wr}.sum_over_all", allow_null=True)
            if r["all_group_revert"] != f"revert:{group}":
                fail(4, f"{wr}.all_group_revert is {r['all_group_revert']!r}, expected 'revert:{group}'")
            if q is not None and (s is None or a_ is None or not close(q, s / a_)):
                fail(4, f"{wr}.sum_over_all is not sum_Delta / Delta_all")
            if q is None:
                notes.append(f"{blabel} {r['domain']}: sum over units / whole-group revert not defined (sum_Delta "
                             f"{s!r}, Delta_all {a_!r}): not drawn")
            ratios.append({"domain": r["domain"], "ratio": q})
        ratios.sort(key=lambda x: list(CONFIG["domains"]).index(x["domain"]))
        out[base] = {"label": blabel, "ckey": ckey, "unit": unit, "units": units, "ratios": ratios,
                     "domains": [x["domain"] for x in ratios]}
    out["_notes"], out["_label"], out["_dropped"] = notes, label, dropped
    return out


# ============================================================================= figure
class HandlerMeanRange(HandlerBase):
    """Legend glyph of the plotted mean-and-range marks, from Line2D artists only (figalign.finalize measures the
    centre of Line2D handles; a LineCollection in a legend box has a null extent)."""

    def __init__(self, ms: float, lw: float, cap: float, color: str):
        super().__init__()
        self.ms, self.lw, self.cap, self.color = ms, lw, cap, color

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x, y, h = 0.5 * width - xdescent, 0.5 * height - ydescent, 0.6 * height
        bar = Line2D([x, x], [y - h, y + h], color=self.color, lw=self.lw, solid_capstyle="butt")
        caps = Line2D([x, x], [y - h, y + h], ls="none", marker="_", ms=2 * self.cap, mew=self.lw, color=self.color)
        mark = Line2D([x], [y], ls="none", marker="o", ms=self.ms, mfc=self.color, mec=self.color, mew=1.0)
        for art in (bar, caps, mark):
            art.set_transform(trans)
        return [bar, caps, mark]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C3 single-layer figure (prereg v1.3 sec. 4.8)")
    ap.add_argument("--input", default=CONFIG["registered_output"],
                    help="full-mode analysis_confirm.json (default: the registered output)")
    ap.add_argument("--out-dir", default=None, help="output folder (default figstyle.FIGDIR; required for a mock)")
    a = ap.parse_args(argv)
    doc, info = load_input(a.input)
    real = gate(info, doc)
    paper_figs = os.path.normcase(os.path.realpath(str(fs.ROOT / "figures")))
    if real:
        out_dir, name = (Path(a.out_dir) if a.out_dir else fs.FIGDIR), CONFIG["name"]
    else:
        if not a.out_dir:
            fail(2, "a mock input needs an explicit --out-dir (mock renders never go to the paper's figures folder)")
        od = os.path.normcase(os.path.realpath(a.out_dir))
        if od == paper_figs or od.startswith(paper_figs + os.sep):
            fail(5, f"refusing to write a mock render into the paper's figures folder {fs.ROOT / 'figures'}")
        out_dir, name = Path(a.out_dir), CONFIG["name"] + "_MOCK"
    D = read_c3_layer(doc)
    exploratory = D["_label"].startswith("exploratory")

    fs.apply()
    MS_MEAN, LW_RANGE, CAP, MS_DOM = 4.0, 1.0, 2.0, 4.0
    fig = fs.figure(CONFIG["height_in"])
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.1])
    axes, any_partial = [], False
    for row, (base, *_rest) in enumerate(CONFIG["bases"]):
        B = D[base]
        col = fs.COLOR[B["ckey"]]
        ax, rx = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        axes += [ax, rx]
        # (a, c) Delta_l / F per unit: mean over domains and range over domains
        xs = list(range(len(B["units"])))
        ys = [0.0]
        for x, u in zip(xs, B["units"]):
            if u["mean"] is None:
                continue
            # bar half-lengths clamped at 0: a float mean may sit one ulp outside its own range (errorbar raises)
            yerr = [[max(0.0, u["mean"] - u["lo"])], [max(0.0, u["hi"] - u["mean"])]]
            ax.errorbar([x], [u["mean"]], yerr=yerr, fmt="o", ms=MS_MEAN,
                        mfc=col if u["full"] else "white", mec=col, mew=1.0, ecolor=col, elinewidth=LW_RANGE,
                        capsize=CAP, capthick=LW_RANGE, zorder=3)
            any_partial |= not u["full"]
            ys += [u["lo"], u["hi"]]
        lo, hi = min(ys), max(ys)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(0.0, color=fs.GREY, lw=0.5, zorder=0)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([u["name"] for u in B["units"]])
        ax.set_xlabel(B["unit"].capitalize())
        ax.set_ylabel(r"$\Delta_\ell/F$")
        ttl = f"{B['label']}: share of $F$ removed per {B['unit']}"
        if base == "nestor" and exploratory:
            ttl += " (exploratory)"
        ax.set_title(ttl)
        # (b, d) per domain: sum of the single-unit deltas over the whole-group delta
        n = len(B["ratios"])
        yy = [n - 1 - i for i in range(n)]
        xr = [1.0]
        for y, r in zip(yy, B["ratios"]):
            if r["ratio"] is not None:
                rx.plot([r["ratio"]], [y], ls="none", marker="D", ms=MS_DOM, mfc=col, mec=col, zorder=3)
                xr.append(r["ratio"])
        rx.axvline(1.0, color=fs.INK, ls=":", lw=0.9, zorder=1)
        lo, hi = min(xr), max(xr)
        span = max(hi - lo, 0.1)
        rx.set_xlim(lo - 0.25 * span, hi + 0.25 * span)
        rx.set_ylim(-0.6, n - 0.4)
        rx.set_yticks(yy)
        rx.set_yticklabels([CONFIG["domains"][r["domain"]] for r in B["ratios"]])
        rx.tick_params(axis="y", length=0)
        rx.set_xlabel(r"$\Sigma_\ell\,\Delta_\ell\,/\,\Delta_{\mathrm{all}}$")
        rx.set_title("Sum / group revert")

    mean_h = Line2D([], [], color=fs.INK)                              # placeholder drawn by HandlerMeanRange
    handles = [mean_h]
    labels = ["mean and range over domains"]
    if any_partial:
        handles.append(Line2D([], [], ls="none", marker="o", ms=MS_MEAN, mfc="white", mec=fs.INK, mew=1.0))
        labels.append("mean over fewer domains")
    handles += [Line2D([], [], ls="none", marker="D", ms=MS_DOM, mfc=fs.INK, mec=fs.INK),
                Line2D([], [], color=fs.INK, ls=":", lw=0.9)]
    labels += ["one run per domain", r"$\Sigma_\ell\,\Delta_\ell=\Delta_{\mathrm{all}}$"]
    fig.legend(handles=handles, labels=labels, loc="outside lower center", ncol=len(handles) if len(handles) <= 3 else 2,
               columnspacing=1.6,
               handletextpad=0.5, handlelength=1.8,
               handler_map={mean_h: HandlerMeanRange(MS_MEAN, LW_RANGE, CAP, fs.INK)})
    fs.add_letters(fig, axes, ["a", "b", "c", "d"])
    corner = None if real else place_mock_stamp(fig)
    shifts = figalign.finalize(fig)
    if not all(math.isfinite(dy) for _, dy in shifts):                 # figalign/figcheck do not catch a NaN centre
        fail(3, f"legend handle alignment undefined: {shifts}")
    figcheck.check(fig, name)
    fs.FIGDIR = Path(out_dir)
    fs.save(fig, name)
    if fs.LOCKED:                                                    # figstyle.save kept <name>.new.<ext> beside them
        fail(6, f"{', '.join(fs.LOCKED)} held open by another program: the new render is next to it as "
                f"{name}.new.<ext>; close the viewer and run again")

    slots = {"figure": name, "real": real, "input": info, "nestor_label": D["_label"], "exploratory_nestor": exploratory,
             "runs_per_base": {b[0]: len(D[b[0]]["domains"]) for b in CONFIG["bases"]},
             "domains_per_base": {b[0]: D[b[0]]["domains"] for b in CONFIG["bases"]},
             "units_per_base": {b[0]: len(D[b[0]]["units"]) for b in CONFIG["bases"]},
             "units_with_fewer_domains": [f"{D[b[0]]['label']} {D[b[0]]['unit']} {u['name']}" for b in CONFIG["bases"]
                                          for u in D[b[0]]["units"] if not u["full"]],
             "dropped": D["_dropped"], "not_drawn": D["_notes"]}
    for msg in D["_notes"]:
        print(f"[fig_c3_single_layer] not drawn: {msg}")
    print("[fig_c3_single_layer] caption slots " + json.dumps(slots, ensure_ascii=False))
    print(f"[fig_c3_single_layer] {name} written to {out_dir}" + ("" if real else f"  (MOCK input: stamped MOCK, {corner})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
