import os
import sys

sys.dont_write_bytecode = True                                  # never write __pycache__ into the paper tree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Figure fig_c4_windows: C4, the per-window difference-in-differences within DCLM, per C1 draw.

Registered definition. Pre-registration v1.3 sec. 4.8: "C4. The per-window results of v1.2 sec. 10.2 become a
main-text figure. No new computation." v1.2 sec. 10.2: for C1 draw d and DCLM window w,
Delta F_w = F_w(B_d) - F_w(A_d) and Delta R_w = log RO_w(B_d) - log RO_w(A_d) at theta_0;
b_d = cov_w(Delta F_w, Delta R_w) / var_w(Delta R_w); beta_C4,d = b_d / F-bar_d with
F-bar_d = (F_DCLM(A_d) + F_DCLM(B_d)) / 2. Estimand: the mean of beta_C4,d over draws (equal weights), overall and
per base. Intervals: a t-interval over draws and a two-way bootstrap over draws and windows (B = 10,000). Also
reported: the within-draw correlation and its mean over draws. v1.3 sec. 9.2: reported regardless.

Input: ONLY the full-mode output of the registered analysis (analysis_confirm.json written by
`analyze_confirm.py run --mode full`; block blocks.C4 from analyze_confirm_ext.block_c4). Every plotted number is
read from that file; nothing is recomputed. The script checks the file (embedded content hash, .sha256 sidecar, format,
mode) and the internal consistency of the block (beta_C4 = b / F_bar per draw; n and the means of the summaries match
the drawn draws) and stops with exit status 4 when a required key is missing or a value is not a finite number.
Values that the analysis itself reports as null (a draw without beta_C4 or correlation, an interval that is not
defined) are not drawn and are listed on stdout; nothing is defaulted or invented.

Real and mock renders. An unmarked figure is written only from the registered output (CONFIG registered_output)
with a matching .sha256 sidecar, freeze_verification.status 'ok' and the frozen freeze-record and pre-registration
hashes; it goes to figures/fig_c4_windows.pdf and .png (figstyle.FIGDIR; FIG_OUT or --out-dir override).
Any other input file is a mock: the figure is stamped MOCK in a free corner, the files are named
fig_c4_windows_MOCK.pdf/.png, --out-dir is required and must not be the paper's figures folder. A verified analysis
at any other path is refused (render the real figure from the registered location).

  python fig_c4_per_window.py                                   # real output, after the full analysis
  python fig_c4_per_window.py --input <mock analysis_confirm.json> --out-dir <folder>

Exit status: 0 written; 2 usage; 3 layout check (figcheck.py) or no free corner for the MOCK stamp; 4 required key
missing or invalid value; 5 input integrity or real/mock gate; 6 a target file was held open (new render kept as
<name>.new.pdf/.png, see figstyle.save).
"""
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_SCRIPTS = os.environ.get("PAPER_SCRIPTS", r"paper\v1\scripts")
if not os.path.exists(os.path.join(HERE, "figstyle.py")):
    if not os.path.exists(os.path.join(PAPER_SCRIPTS, "figstyle.py")):
        raise SystemExit(f"figstyle.py not found next to this script or in {PAPER_SCRIPTS} (set PAPER_SCRIPTS)")
    sys.path.insert(0, PAPER_SCRIPTS)

from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.legend_handler import HandlerLine2D  # noqa: E402
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
    "name": "fig_c4_windows",
    "bases": [("moe_b1", "MoE base 1"), ("moe_b2", "MoE base 2")],     # the C1 bases (role c1 of the frozen plan)
    "height_in": 50.0 / 25.4,                                          # about 50 mm in the main text
    "rel_tol": 1e-9,
}
MOCK_COLOR = "#D55E00"                                                 # Okabe-Ito vermillion: the stamp only


# ============================================================================= input
def fail(code: int, msg: str):
    print(f"[fig_c4_per_window] ERROR: {msg}", file=sys.stderr)
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


# ============================================================================= C4 block
def read_c4(doc: dict) -> dict:
    c4 = req(doc, "blocks.C4")
    if not isinstance(c4, dict):
        fail(4, "blocks.C4 is not an object")
    if "status" in c4:
        fail(4, f"blocks.C4 has status {c4['status']!r}; no per-draw results to draw")
    per_draw, per_base, overall = req(c4, "per_draw", "blocks.C4"), req(c4, "per_base", "blocks.C4"), \
        req(c4, "overall", "blocks.C4")
    dropped, e1 = req(c4, "dropped", "blocks.C4"), req(c4, "E1", "blocks.C4")
    names = [b for b, _ in CONFIG["bases"]]
    extra = sorted(set(per_base) - set(names))
    if extra:
        fail(4, f"blocks.C4.per_base has bases outside the registered C1 bases: {extra}")
    for b in names:
        if b not in per_base:
            fail(4, f"required key 'blocks.C4.per_base.{b}' missing (registered C1 base without usable draws); "
                    f"dropped: {json.dumps(dropped)}")
    draws, notes = {b: [] for b in names}, []
    for k, r in enumerate(per_draw):
        w = f"blocks.C4.per_draw.{k}"
        base = req(r, "base", w)
        if base not in draws:
            fail(4, f"{w}.base {base!r} is not a registered C1 base")
        d = req(r, "d", w)
        W = req(r, "W", w)
        bd = num(req(r, "b", w), f"{w}.b", allow_null=True)
        fbar = num(req(r, "F_bar", w), f"{w}.F_bar")
        beta = num(req(r, "beta_C4", w), f"{w}.beta_C4", allow_null=True)
        corr = num(req(r, "corr", w), f"{w}.corr", allow_null=True)
        if not isinstance(d, int) or not isinstance(W, int):
            fail(4, f"{w}: d and W must be integers ({d!r}, {W!r})")
        if beta is not None and (bd is None or not close(beta, bd / fbar)):
            fail(4, f"{w}: beta_C4 {beta!r} is not b / F_bar ({bd!r} / {fbar!r})")
        if beta is None:
            notes.append(f"{base} d{d}: beta_C4 null (b {bd!r}, F_bar {fbar!r}): not drawn, not in the means")
        if corr is None:
            notes.append(f"{base} d{d}: within-draw correlation null: not drawn in (b)")
        draws[base].append({"d": d, "W": W, "beta": beta, "corr": corr})
    for b in names:
        draws[b].sort(key=lambda r: r["d"])
        if len({r["d"] for r in draws[b]}) != len(draws[b]):
            fail(4, f"blocks.C4.per_draw: draw indices of {b} repeat")

    def summary(s: dict, where: str, rows: list) -> dict:
        n = req(s, "n", where)
        used = [r for r in rows if r["beta"] is not None]
        if n != len(used):
            fail(4, f"{where}.n = {n} but {len(used)} draws carry beta_C4")
        if n == 0:
            fail(4, f"{where}: no draw with beta_C4 (n = 0)")
        out = {"n": n, "mean": num(req(s, "mean_beta_C4", where), f"{where}.mean_beta_C4"),
               "corr_mean": num(req(s, "mean_within_draw_corr", where), f"{where}.mean_within_draw_corr",
                                allow_null=True),
               "t95": None, "boot": None, "boot_status": None, "B": None}
        if not close(out["mean"], sum(r["beta"] for r in used) / n):
            fail(4, f"{where}.mean_beta_C4 is not the mean of the drawn beta_C4,d")
        cs = [r["corr"] for r in used if r["corr"] is not None]
        if (out["corr_mean"] is None) != (not cs) or (cs and not close(out["corr_mean"], sum(cs) / len(cs))):
            fail(4, f"{where}.mean_within_draw_corr is not the mean of the drawn correlations")
        t = req(s, "t95", where)
        if n >= 2:
            wt = f"{where}.t95"
            lo, hi = num(req(t, "lo", wt), f"{wt}.lo"), num(req(t, "hi", wt), f"{wt}.hi")
            if not close(num(req(t, "mean", wt), f"{wt}.mean"), out["mean"]) or not lo <= out["mean"] <= hi:
                fail(4, f"{where}.t95 does not hold the mean")
            out["t95"] = (lo, hi)
            bs = req(s, "bootstrap", where)
            wb = f"{where}.bootstrap"
            if "status" in bs:
                out["boot_status"] = str(bs["status"])
                notes.append(f"{where}: two-way bootstrap not drawn ({bs['status']})")
            else:
                ci = req(bs, "ci95", wb)
                out["B"] = req(bs, "B", wb)
                if ci is None:
                    out["boot_status"] = "ci95 null"
                    notes.append(f"{where}: two-way bootstrap ci95 null, not drawn")
                else:
                    out["boot"] = (num(req(bs, "ci95.0", wb), f"{wb}.ci95[0]"),
                                   num(req(bs, "ci95.1", wb), f"{wb}.ci95[1]"))
        else:
            notes.append(f"{where}: n = {n}, no interval is defined")
        return out

    sums = {b: summary(per_base[b], f"blocks.C4.per_base.{b}", draws[b]) for b in names}
    sums["overall"] = summary(overall, "blocks.C4.overall", [r for b in names for r in draws[b]])
    Ws = sorted({r["W"] for b in names for r in draws[b]})
    return {"draws": draws, "sums": sums, "notes": notes, "dropped": dropped, "E1": e1, "W": Ws}


# ============================================================================= figure
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C4 per-window figure (prereg v1.3 sec. 4.8; v1.2 sec. 10.2)")
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
    D = read_c4(doc)

    fs.apply()
    C = fs.COLOR
    MOE2 = to_hex([0.5 + 0.5 * v for v in to_rgb(C["moe"])])        # MoE base 2: 50% tint, dashed (as Fig. 5)
    STY = {"moe_b1": (C["moe"], "-"), "moe_b2": (MOE2, fs.DASH_SEED43), "overall": (fs.INK, "-")}
    GROUPS = CONFIG["bases"] + [("overall", "overall")]
    X_STRIP, HALF, X_MEAN, X_BOOT = -0.17, 0.11, 0.13, 0.23           # offsets from the group centre
    MS_DRAW, MS_MEAN, LW_T, LW_B = 3.4, 4.4, 2.0, 0.9
    # matplotlib scales a dash pattern by the line width; the thick t-interval of base 2 gets the absolute 4/2 pt
    # dashes that the thin seed-43 lines of the other figures show (DASH_SEED43 at about 1 pt)
    DASH_T = (0, tuple(v / LW_T for v in fs.DASH_SEED43[1]))

    fig = fs.figure(CONFIG["height_in"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def strip_x(g: int, n: int, k: int) -> float:
        return g + X_STRIP + (0.0 if n == 1 else -HALF + 2 * HALF * k / (n - 1))

    # (a) beta_C4,d per draw; mean over draws with the t-interval and the two-way bootstrap interval
    ys = [0.0]
    for g, (key, _) in enumerate(GROUPS):
        col, ls = STY[key]
        s = D["sums"][key]
        xm = g + (X_MEAN if key != "overall" else 0.0)
        if key != "overall":
            rows = D["draws"][key]
            for k, r in enumerate(rows):
                if r["beta"] is not None:
                    ax.plot([strip_x(g, len(rows), k)], [r["beta"]], ls="none", marker="o", ms=MS_DRAW, mfc="white",
                            mec=col, mew=0.8, zorder=3)
                    ys.append(r["beta"])
        if s["t95"]:
            ax.plot([xm, xm], list(s["t95"]), color=col, ls=(DASH_T if ls != "-" else "-"), lw=LW_T,
                    solid_capstyle="butt", dash_capstyle="butt", zorder=2)
            ys += list(s["t95"])
        if s["boot"]:
            xb = xm + (X_BOOT - X_MEAN)
            ax.plot([xb, xb], list(s["boot"]), color=col, ls=ls, lw=LW_B, zorder=2)
            for yb in s["boot"]:
                ax.plot([xb - 0.035, xb + 0.035], [yb, yb], color=col, lw=LW_B, zorder=2)
            ys += list(s["boot"])
        ax.plot([xm], [s["mean"]], ls="none", marker="o", ms=MS_MEAN, mfc=col, mec=col, zorder=4)
        ys.append(s["mean"])
    lo, hi = min(ys), max(ys)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0.0, color=fs.GREY, lw=0.5, zorder=0)
    ax.set_xlim(-0.45, len(GROUPS) - 1 + 0.42)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([lab for _, lab in GROUPS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$\beta_{\mathrm{C4},d}=b_d/\bar{F}_d$")
    ax.set_title(r"Slope of $\Delta F_w$ on $\Delta R_w$, per draw")

    # (b) within-draw correlation per draw and its mean over draws
    ys = [0.0]
    for g, (key, _) in enumerate(GROUPS):
        col, _ls = STY[key]
        s = D["sums"][key]
        xm = g + (X_MEAN if key != "overall" else 0.0)
        if key != "overall":
            rows = D["draws"][key]
            for k, r in enumerate(rows):
                if r["beta"] is not None and r["corr"] is not None:
                    bx.plot([strip_x(g, len(rows), k)], [r["corr"]], ls="none", marker="o", ms=MS_DRAW, mfc="white",
                            mec=col, mew=0.8, zorder=3)
                    ys.append(r["corr"])
        if s["corr_mean"] is not None:
            bx.plot([xm], [s["corr_mean"]], ls="none", marker="o", ms=MS_MEAN, mfc=col, mec=col, zorder=4)
            ys.append(s["corr_mean"])
    lo, hi = min(ys), max(ys)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    bx.set_ylim(max(lo - pad, -1.0), min(hi + pad, 1.0))
    bx.axhline(0.0, color=fs.GREY, lw=0.5, zorder=0)
    bx.set_xlim(-0.45, len(GROUPS) - 1 + 0.42)
    bx.set_xticks(range(len(GROUPS)))
    bx.set_xticklabels([lab for _, lab in GROUPS])
    bx.tick_params(axis="x", length=0)
    bx.set_ylabel("Correlation")
    bx.set_title(r"Correlation of $\Delta F_w$ and $\Delta R_w$")

    handles = [Line2D([], [], ls="none", marker="o", ms=MS_DRAW, mfc="white", mec=fs.INK, mew=0.8, label="single draw"),
               Line2D([], [], ls="none", marker="o", ms=MS_MEAN, mfc=fs.INK, mec=fs.INK, label="mean over draws"),
               Line2D([], [], color=fs.INK, lw=LW_T, solid_capstyle="butt", label="t-interval over draws, 95%"),
               Line2D([], [], color=fs.INK, lw=LW_B, marker="|", ms=4.0, mew=LW_B, label="two-way bootstrap, 95%")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.6, handletextpad=0.5,
               handlelength=1.8, handler_map={handles[3]: HandlerLine2D(numpoints=2)})
    fs.add_letters(fig, [ax, bx], ["a", "b"], align=True)
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

    slots = {"figure": name, "real": real, "input": info, "n_draws": D["sums"]["overall"]["n"],
             "n_draws_per_base": {b: D["sums"][b]["n"] for b, _ in CONFIG["bases"]}, "W_windows": D["W"],
             "B_bootstrap": sorted({D["sums"][k]["B"] for k in D["sums"] if D["sums"][k]["B"] is not None}),
             "dropped": D["dropped"], "E1": D["E1"], "not_drawn": D["notes"]}
    for msg in D["notes"]:
        print(f"[fig_c4_per_window] not drawn: {msg}")
    print("[fig_c4_per_window] caption slots " + json.dumps(slots, ensure_ascii=False))
    print(f"[fig_c4_per_window] {name} written to {out_dir}" + ("" if real else f"  (MOCK input: stamped MOCK, {corner})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
