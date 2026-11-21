"""fill_author.py -- outcome-conditional drafts of the five author slots (paper v1.5.13)

Status: AI-assisted draft for the author's review. The author reviews, revises where necessary and approves these
statements before submission (AI declaration in the Declarations).

Called by scripts/fill_core.py after the core fills, with its Filler object and its module; it never edits a .tex file and
reads nothing but the core analysis output that fill_core.py has already loaded and checked. For each author slot
(fill_core CONFIG["author_ids"]) it picks one variant of scripts/author_slot_variants.json from the registered decision
labels (decisions.C1 and decisions.C2: label, reasons, three-statistic label, interpretation flags) and fills its
<<placeholders>> with numbers from the same output, through the field paths and formatters of fill_core.py.
It computes no number itself. DISC.code and CONCL.code carry no number: they follow the registered C1 label alone
(confirmed, confirmed in the pooled test only, refuted, inconclusive), so that the Discussion and the Conclusion use only
registered quantities that the Results report.
"""
import json
import re
from pathlib import Path

STATUS = "AI-assisted draft for the author's review"
VARIANTS = Path(__file__).with_name("author_slot_variants.json")


# ----------------------------------------------------------------------------- outcome classes
def classify_c1(fc, d):
    dec = fc.F("C1.decision", d) or {}; lab = dec.get("label")
    if not lab:
        return None
    reasons = [str(r) for r in (dec.get("reasons") or [])]; interp = [str(x) for x in (dec.get("interpretation") or [])]
    t3 = str((dec.get("three_statistic_table") or {}).get("label") or "")
    raw = ((fc.F("C1.estimators", d) or {}).get("raw") or {}).get("overall") or {}
    quals = []
    if lab.startswith("confirmed in the pooled"):
        main = edge = "C1-D"
    elif lab.startswith("confirmed"):
        main = edge = "C1-B" if any("routing-gated sensitivity" in x for x in interp) else "C1-A"
        if "margin-limited" in lab:
            quals.append("C1-C")
    elif lab.startswith("refuted"):
        main = "C1-F" if t3.startswith("learned-update law refuted") else ("C1-G" if "does not localize" in t3 else "C1-R")
        edge = main
        if "reversed" in lab:
            quals.append("C1-H"); edge = "C1-H"
    else:                                                  # inconclusive: the registered reason decides the variant
        t = fc.F("C1.test", d) or {}; holm = fc.F("holm", d) or {}
        adj = (holm.get("adjusted") or {}).get("C1"); alpha = holm.get("FWER", 0.05)
        if "no usable draws" in reasons:
            main = "C1-K.none"
        elif any(r.startswith("E1:") for r in reasons):
            main = "C1-K.E1"
        elif t3.startswith("first-order"):
            main = "C1-I"
        elif any(r.startswith("would-be refutation in a margin-limited base") for r in reasons):
            main = "C1-K.ml"
        elif fc.isnum(t.get("mean")) and t["mean"] > 0 and fc.isnum(adj) and adj > alpha:
            main = "C1-J"
        else:
            main = "C1-K.ub"
        edge = "C1-inc" if fc.isnum(raw.get("beta_hat")) and main not in ("C1-K.none", "C1-K.E1") else "C1-inc0"
    if lab.startswith("confirmed") and any("decay-driven" in x for x in interp):
        quals.append("C1-E")
    if any("not confirmatory" in r for r in reasons):
        quals.append("C1-X")
    return {"label": lab, "main": main, "quals": quals, "edge": edge}


def classify_c2(fc, d):
    dec = fc.F("C2.decision", d) or {}; lab = dec.get("label")
    if not lab or lab == "not run":
        return None
    reasons = [str(r) for r in (dec.get("reasons") or [])]
    has_de = fc.isnum((fc.F("C2.Delta_e", d) or {}).get("beta_hat"))
    if lab.startswith("exploratory"):
        base = lab.split(": ", 1)[1] if ": " in lab else lab
        v = ("C2-E" if base == "confirmed" else "C2-F") if has_de else "C2-F0"
        return {"label": lab, "disc": v, "edge": v}
    if lab == "confirmed":
        return {"label": lab, "disc": "C2-A", "edge": "C2-A"}
    if lab == "refuted":
        return {"label": lab, "disc": "C2-B", "edge": "C2-B"}
    if lab == "uninformative":
        return {"label": lab, "disc": "C2-D", "edge": "C2-D"}
    if lab == "inconclusive":
        if "no draws" in reasons:
            return {"label": lab, "disc": "C2-C.none", "edge": "C2-C0"}
        if any(r.startswith("E1:") for r in reasons):
            return {"label": lab, "disc": "C2-C.E1", "edge": "C2-C0"}
        return {"label": lab, "disc": "C2-C", "edge": "C2-C" if has_de else "C2-C0"}
    return {"label": lab, "disc": "C2-F" if has_de else "C2-F0", "edge": "C2-F" if has_de else "C2-F0"}


def code_class(c1):
    """Variant of DISC.code and CONCL.code from the registered C1 label class alone (no number)."""
    if not c1:
        return None
    m = c1["main"]
    if m in ("C1-A", "C1-B"):
        return "CODE.conf"
    if m == "C1-D":
        return "CODE.pooled"
    if m in ("C1-F", "C1-G", "C1-R"):
        return "CODE.ref"
    return "CODE.inc"                                      # C1-I, C1-J, C1-K.*


# ----------------------------------------------------------------------------- placeholders
def _join(names):
    return names[0] if len(names) == 1 else (", ".join(names[:-1]) + " and " + names[-1] if names else "n/a")


def render(text, vals):
    def sub(m):
        if m.group(1) not in vals:
            raise SystemExit(f"[fill_author] placeholder <<{m.group(1)}>> has no value")
        return vals[m.group(1)]
    return re.sub(r"<<(\w+)>>", sub, text)


def values(fc, d, c1, c2, S):
    num, interval, pfmt, isnum = fc.num, fc.interval, fc.pfmt, fc.isnum
    def mnum(x, nd=3):                                    # inside $...$: a plain minus sign
        return f"{float(x):.{nd}f}" if isnum(x) else "n/a"
    est = fc.F("C1.estimators", d) or {}
    def e(k):
        return (est.get(k) or {}).get("overall") or {}
    holm = (fc.F("holm", d) or {}).get("adjusted") or {}
    bases = fc.CONFIG["bases"]
    v = {"beta_ref": num(fc.BETA_REF, 2)}
    for k, nm in (("raw", "b"), ("sym", "bsym"), ("rand", "brand")):
        v[nm], v[nm + "_ci"] = mnum(e(k).get("beta_hat")), interval(e(k).get("lo"), e(k).get("hi"))
    smr = fc.F("C1.sym_minus_rand", d) or {}
    v["smr"], v["smr_ci"] = num(smr.get("mean")), interval(smr.get("lo"), smr.get("hi"))
    v["wd"] = num(fc.F("C1.wd_share", d), 2)
    v["wd3"] = num(fc.F("C1.wd_share", d), 3)                      # C1-E since v1.6.1 (final pass M2)
    t = fc.F("C1.test", d) or {}
    v["mj"], v["p_holm"] = mnum(t.get("mean")), pfmt(holm.get("C1"))
    pb = fc.F("C1.per_base", d) or {}
    v["nonpos_bases"] = _join([bases[b] for b in bases if isinstance(pb.get(b), dict) and isnum(pb[b].get("mean_J")) and pb[b]["mean_J"] <= 0])
    ml = [bases.get(b, b) for b, x in (fc.F("C1.margin_limited", d) or {}).items() if x]
    v["ml_bases"], v["ml_verb"] = _join(ml), ("were" if len(ml) > 1 else "was")
    e1 = fc.F("C1.E1", d) or {}
    v["e1_div"] = str(e1["diverged"]) if isnum(e1.get("diverged")) else "n/a"
    v["e1_runs"] = str(e1["runs"]) if isnum(e1.get("runs")) else "n/a"
    s, D2 = fc.F("C2.s_star", d), fc.F("C2.D", d)
    v["s"], v["D2"] = (str(s) if isnum(s) else "n/a"), (str(D2) if isnum(D2) else "n/a")
    de = fc.F("C2.Delta_e", d) or {}
    v["de"], v["de_ci"] = mnum(de.get("beta_hat")), interval(de.get("lo"), de.get("hi"))
    v["de_ub"] = num((fc.F("C2.decision", d) or {}).get("upper_bound_Delta_e"))
    t2 = fc.F("C2.test", d) or {}
    v["mj2"], v["p2_holm"] = mnum(t2.get("mean")), pfmt(holm.get("C2"))
    v["manip"] = mnum((fc.F("C2.manip", d) or {}).get("mean_D_SGD_over_DQ_mean"), 2)
    sc = fc.F("C1.second_stream", d) or {}; be = (sc.get("beta") or {}).get("overall") or {}
    v["code2"] = S["DISC.code"]["second_stream"] if isnum(be.get("beta_hat")) else ""   # pointer to the Results, no number
    v["x_code"] = S["DISC.code"]["exploratory_suffix"] if c1 and "C1-X" in c1["quals"] else ""
    sfx = S["EDGE.C1"]["suffixes"]
    v["ml_suffix"] = render(sfx["ml_suffix"], v) if c1 and "C1-C" in c1["quals"] else ""
    v["x_suffix"] = sfx["x_suffix"] if c1 and "C1-X" in c1["quals"] else ""
    return v


# ----------------------------------------------------------------------------- slots
def fill_author(f, fc):
    d = f.doc
    S = json.load(open(VARIANTS, encoding="utf-8"))["slots"]
    c1, c2 = classify_c1(fc, d), classify_c2(fc, d)
    cv = code_class(c1)
    v = values(fc, d, c1, c2, S)
    chosen = {}

    def put(slot, vid, text, fields):
        text = render(text, v)
        if "n/a" in text:
            f.warn.append(f"author slot {slot} ({vid}): a placeholder has no value in the analysis output")
        f.put(slot, text, fields)
        chosen[slot] = {"variant": vid, "status": STATUS}

    DI = S["DISC.interpretation"]; parts, ids = [], []
    if c1:
        parts.append(DI["C1"][c1["main"]]); ids.append(c1["main"])
        for q in ("C1-C", "C1-E", "C1-H", "C1-X"):
            if q in c1["quals"]:
                parts.append(DI["C1_qualifiers"][q]); ids.append(q)
    if c2:
        parts.append(DI["C2"][c2["disc"]]); ids.append(c2["disc"])
    if parts:
        put("DISC.interpretation", "+".join(ids), " ".join(parts),
            ["decisions.C1", "decisions.C2", "C1.estimators", "C1.secondary.I_sym_minus_I_rand", "C1.secondary.wd_share", "C1.test", "holm.adjusted",
             "C1.mean_J_per_base", "C1.margin_limited", "C1.E1", "C2.s_star", "C2.D", "C2.Delta_e", "C2.test", "C2.manipulation_check"])
    else:
        f.skip("DISC.interpretation", "no decisions.C1 or decisions.C2 in the analysis output")
    if c1:
        put("EDGE.C1", c1["edge"] + "".join("+" + q for q in c1["quals"] if q in ("C1-C", "C1-X")), S["EDGE.C1"]["variants"][c1["edge"]],
            ["decisions.C1", "C1.estimators.raw.overall", "C1.margin_limited"])
        put("DISC.code", cv + ("+X" if v["x_code"] else "") + ("+CODE2" if v["code2"] else ""), S["DISC.code"]["variants"][cv],
            ["decisions.C1"] + (["C1.secondary.second_code_stream.beta.overall"] if v["code2"] else []))
        put("CONCL.code", cv, S["CONCL.code"]["variants"][cv], ["decisions.C1"])
    else:
        for slot in ("EDGE.C1", "DISC.code", "CONCL.code"):
            f.skip(slot, "no decisions.C1 in the analysis output")
    if c2:
        put("EDGE.C2", c2["edge"], S["EDGE.C2"]["variants"][c2["edge"]], ["decisions.C2", "C2.s_star", "C2.Delta_e"])
    else:
        f.skip("EDGE.C2", "C2 not in the analysis output")
    return chosen
