"""Routing entanglement, prereg v1.3 sec. 4.9 (CPU, descriptive; no decision depends on it): wrapper around the
entanglement.py (theory note v4 addendum sec. 1.6-1.7) on the G0 RE-RUN routing tables, with the prediction
protocol of addendum sec. 1.6 enforced ("predictions to hash before the re-run tables are opened").

  python run_entanglement.py hash-predictions [--note theory/v4/theory_note_v4_addendum.md]
         [--out results/entanglement/predictions_E1_E7.md] [--log logs/entanglement_predictions.log]
      writes the block E1-E7 of the note verbatim (UTF-8, LF) and appends '<UTC> <sha256> <file> note=<sha256>' to the
      log. Idempotent for the same text; refuses a different text once a hash is logged (the predictions are final).
  python run_entanglement.py run --g0-config <re-run gate config> [--gate-record <re-run G0 gate record>]
         [--out-dir results/entanglement] [--no-g01] [--n-draws 1000]
      refuses unless the predictions file exists and its sha256 is logged; checks that theory/v4/entanglement.py equals
      the registered copy (inputs_v13/entanglement.py) and that crossover.py equals its g0_code_freeze.json entry; then,
      per base of the config: ent_<base>.json (the statistics: xi_LP, xi_emp, R2_pi, rho2_piD, rho2_piT.D,
      target-DCLM weight correlation, i_k, calls; C2 and DCLM-tilt rows), the cross-corpus correlation matrices of log
      usage (Pearson and Spearman, per stratum and pooled within strata, selection and estimation halves), the G0-1
      computation for every target (german, books, french; code streams code / pycode; f = 1/2, 1/3; registered generator
      and tolerance) with the share of target-matched proposals rejected by the DCLM constraint
      (1 - acceptance(target and DCLM) / acceptance(target)); a check of the predictions E1-E6 (E7 exploratory: not
      evaluated here) and of V1 (calls at the operational load 0.67 against the G0-2 rows of the gate record);
      entanglement_summary.json with every input sha256 and the order T_predictions < T_tables_opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

CORPORA = ("dclm", "german", "books", "code", "pycode", "math", "french")
G01_TARGETS = ("german", "books", "french")
START = "**Predictions to hash before the re-run tables are opened"
TOOL = os.path.join("theory", "v4", "entanglement.py")
TOOL_REGISTERED = os.path.join("inputs_v13", "entanglement.py")
OPERATIONAL = 0.67


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: str | None) -> str | None:
    if not p or not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return _sha(f.read())


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================================= predictions
def predictions_block(note_text: str) -> str:
    """The E1-E7 block of the addendum: the START line through the E7 bullet (verbatim, LF line ends)."""
    lines = note_text.replace("\r\n", "\n").split("\n")
    i = next((k for k, ln in enumerate(lines) if ln.startswith(START)), None)
    if i is None:
        raise SystemExit("prediction block not found in the note")
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("- **E7")), None)
    if j is None:
        raise SystemExit("E7 not found after the prediction header")
    block = lines[i: j + 1]
    for e in range(1, 8):
        if not any(ln.startswith(f"- **E{e}") for ln in block):
            raise SystemExit(f"E{e} missing from the prediction block")
    return "\n".join(block) + "\n"


def logged_hashes(log: str) -> list[dict]:
    out = []
    if os.path.exists(log):
        for ln in open(log, encoding="utf-8").read().splitlines():
            p = ln.split()
            if len(p) >= 3:
                out.append({"utc": p[0], "sha256": p[1], "file": p[2], "rest": " ".join(p[3:])})
    return out


def cmd_hash(note: str, out: str, log: str) -> dict:
    text = predictions_block(open(note, encoding="utf-8").read())
    b = text.encode("utf-8")
    h = _sha(b)
    prev = [e for e in logged_hashes(log) if os.path.normcase(e["file"]) == os.path.normcase(out)]
    if prev and prev[0]["sha256"] != h:
        raise SystemExit(f"predictions already hashed at {prev[0]['utc']} ({prev[0]['sha256']}); a different text is not "
                         f"accepted (the predictions are final)")
    if os.path.exists(out) and _sha_file(out) != h:
        raise SystemExit(f"{out} exists with a different text")
    if prev:
        return {"sha256": h, "utc": prev[0]["utc"], "file": out, "new": False}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(b)
    os.makedirs(os.path.dirname(os.path.abspath(log)) or ".", exist_ok=True)
    utc = _utc()
    with open(log, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{utc} {h} {out} note={_sha_file(note)}\n")
    return {"sha256": h, "utc": utc, "file": out, "new": True}


# ============================================================================= correlations of log usage
def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=np.float64)
    r[o] = np.arange(len(a), dtype=np.float64)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _corr(X: np.ndarray) -> list:
    """Pearson correlation matrix of the columns of X (no LAPACK)."""
    Xc = X - X.mean(0, keepdims=True)
    sd = np.sqrt((Xc * Xc).sum(0))
    C = (Xc.T @ Xc) / np.outer(sd, sd)
    return [[float(v) for v in row] for row in C]


def _tool_module():
    """The entanglement.py (TOOL) as a module (its log_usage floor is the registered one)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("entanglement_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def correlations(t) -> dict:
    EL = _tool_module()
    out = {}
    for half, tab in (("selection", t.sel), ("estimation", t.est)):
        cs = [c for c in CORPORA if c in tab]
        x = {c: EL.log_usage(np.asarray(tab[c], dtype=np.float64)) for c in cs}
        per = []
        for l in range(t.L):
            X = np.stack([x[c][l] for c in cs], 1)
            R = np.stack([_rank(x[c][l]) for c in cs], 1)
            per.append({"stratum": t.strata[l], "pearson": _corr(X), "spearman": _corr(R)})
        Xp = np.stack([(x[c] - x[c].mean(1, keepdims=True)).ravel() for c in cs], 1)
        Rp = np.stack([np.concatenate([_rank(x[c][l]) - (len(x[c][l]) - 1) / 2 for l in range(t.L)]) for c in cs], 1)
        out[half] = {"corpora": cs, "pooled_within_strata": {"pearson": _corr(Xp), "spearman": _corr(Rp)},
                     "per_stratum": per}
    return out


# ============================================================================= G0-1 computation for every target
def g01_rows(t, base: str, k: int, n_draws: int) -> list[dict]:
    import crossover as CX
    import g0_gate as GG

    rows = []
    for target in G01_TARGETS:
        for code in ("code", "pycode"):
            if target not in t.sel or code not in t.sel or "dclm" not in t.sel or "math" not in t.sel:
                continue
            for f in CX.REG_POOLS:
                full = GG.gen_run(t, "c1", f=f, target=target, code=code, match=CX.reg_matched("c1", target=target),
                                  tol=CX.REG_TOL, n_draws=n_draws, k=k)
                tonly = GG.gen_run(t, "c1", f=f, target=target, code=code, match=[target], tol=CX.REG_TOL,
                                   n_draws=n_draws, k=k)
                a_full, a_t = full.get("acceptance"), tonly.get("acceptance")
                rej = (1.0 - a_full / a_t) if (a_full is not None and a_t) else None
                rows.append({"cell": f"{base}|{target}|{code}", "f": f, "registered_target": target in ("german", "books"),
                             "n_pairs": full["n_pairs"], "g0_2_pass": full["g0_2_pass"], "median_P": full["median"],
                             "acceptance": a_full, "swap_convergence": full["swap_convergence"],
                             "acceptance_target_only": a_t, "dclm_rejection_share": rej})
    return rows


# ============================================================================= prediction check (E1-E6) and V1
def _cells(rows: list[dict]) -> dict:
    return {(r["cell"], round(r["f"], 6)): r for r in rows}


def check_predictions(ent: dict, record: dict | None) -> dict:
    rows = [r for b in ent.values() for r in b.get("rows", [])]
    other = [r for b in ent.values() for r in b.get("other_designs", [])]
    C = _cells(rows)
    half, third = 0.5, round(1 / 3, 6)
    xi = lambda base, tgt, code, f: (C.get((f"{base}|{tgt}|{code}", round(f, 6))) or {}).get("xi_LP")
    res = []
    books = [(b, c, xi(b, "books", c, half)) for b in ("moe_b1", "moe_b2") for c in ("code", "pycode")]
    ok1 = [v for *_, v in books if v is not None]
    res.append({"id": "E1", "values": books, "result": None if not ok1 else all(v < 0.67 for v in ok1),
                "statement": "xi_LP(f=1/2) of every books cell < 0.67 (point 0.55)"})
    g = {(b, c): xi(b, "german", c, half) for b in ("moe_b1", "moe_b2", "nestor") for c in ("code", "pycode")}
    cond = []
    for (b, c), v in g.items():
        if v is None:
            continue
        cond.append(0.72 <= v <= 1.01 if b == "moe_b1" else v >= 0.82 if b == "moe_b2" else v >= 0.90)
    order = []
    for c in ("code", "pycode"):
        bk = [xi(b, "books", c, half) for b in ("moe_b1", "moe_b2")]
        vals = [max([v for v in bk if v is not None], default=None), g.get(("moe_b1", c)), g.get(("moe_b2", c)),
                g.get(("nestor", c))]
        if all(v is not None for v in vals):
            order.append(vals[0] < vals[1] < vals[2] <= vals[3])
    res.append({"id": "E2", "values": {f"{b}|{c}": v for (b, c), v in g.items()}, "ordering_ok": order,
                "result": None if not cond else (all(cond) and all(order)),
                "statement": "German xi_LP in [0.72, 1.01] b1, >= 0.82 b2, >= 0.90 NESTOR; books < German b1 < b2 <= NESTOR"})
    c1_third = [(r["cell"], r["xi_LP"]) for r in rows if abs(r["f"] - 1 / 3) < 1e-6]
    res.append({"id": "E3", "values": c1_third, "result": None if not c1_third else all(v > 0.9 for _, v in c1_third),
                "statement": "xi_LP(f=1/3) > 0.9 for every C1 cell"})
    e4 = []
    for b in ("moe_b1", "moe_b2", "nestor"):
        for c in ("code", "pycode"):
            rg, rb = C.get((f"{b}|german|{c}", half)), C.get((f"{b}|books|{c}", half))
            if rg and rb:
                e4.append({"cell": f"{b}|{c}", "rho2_piT_D_german": rg.get("rho2_piT_D"), "rho2_piT_D_books": rb.get("rho2_piT_D"),
                           "corr_TD_german": rg.get("corr_TD_share"), "corr_TD_books": rb.get("corr_TD_share"),
                           "ok": (rg.get("rho2_piT_D") or 0) > (rb.get("rho2_piT_D") or 0) and
                                 (rb.get("corr_TD_share") or 0) > (rg.get("corr_TD_share") or 0)})
    res.append({"id": "E4", "values": e4, "result": None if not e4 else all(x["ok"] for x in e4),
                "statement": "rho2_piT.D German > books on every base and code stream; corr(books, DCLM) > corr(German, DCLM)"})
    c2 = [(r["cell"], r["f"], r["xi_LP"]) for r in other if r["cell"].endswith("|c2|dclm_only")]
    dt = [(r["cell"], r["f"], r["xi_LP"]) for r in other if r["cell"].endswith("|dclm_tilt")]
    res.append({"id": "E5", "values": {"c2_dclm_only": c2, "dclm_tilt": dt},
                "result": None if not (c2 and dt) else (all(v < 0.4 for _, f, v in c2 if abs(f - 0.5) < 1e-6) and
                                                       all(v > 0.8 for _, f, v in dt if abs(f - 0.5) < 1e-6)),
                "statement": "C2 DCLM-only xi_LP < 0.4; DCLM tilt xi_LP > 0.8 (f = 1/2; indicative)"})
    ik = []
    for tgt in ("german", "books"):
        for c in ("code", "pycode"):
            for f in (half, 1 / 3):
                rn = C.get((f"nestor|{tgt}|{c}", round(f, 6)))
                rm = [C.get((f"{b}|{tgt}|{c}", round(f, 6))) for b in ("moe_b1", "moe_b2")]
                rm = [r for r in rm if r]
                if rn and rm:
                    ik.append({"cell": f"{tgt}|{c}|f={f:.3f}", "i_k_nestor": rn.get("i_k"),
                               "i_k_moe": [r.get("i_k") for r in rm],
                               "ratio": rn.get("i_k") / max(r.get("i_k") for r in rm) if rn.get("i_k") is not None else None})
    res.append({"id": "E6", "values": ik, "result": None if not ik else all((x["ratio"] or 9) < 1.0 for x in ik),
                "statement": "compensation intensity i_k of NESTOR well below the MoE's (reported: ratio < 1)"})
    res.append({"id": "E7", "values": None, "result": None,
                "statement": "exploratory (residual-tilt pools): not evaluated by this wrapper (residual_tilt tool)"})
    v1 = None
    if record:
        g2 = ((record.get("gates") or {}).get("G0-2") or {}).get("rows") or []
        comp = []
        for r in g2:
            if r.get("design") != "C1":
                continue
            for fk, v in (r.get("per_f") or {}).items():
                e = C.get((r["cell"], round(float(fk), 6)))
                if e is None:
                    comp.append({"cell": r["cell"], "f": float(fk), "recorded_pass": v.get("g0_2_pass"), "xi_LP": None,
                                 "agrees": None})
                    continue
                pred = e["xi_LP"] < OPERATIONAL
                comp.append({"cell": r["cell"], "f": float(fk), "recorded_pass": bool(v.get("g0_2_pass")),
                             "xi_LP": e["xi_LP"], "agrees": pred == bool(v.get("g0_2_pass"))})
        n_ok = sum(1 for x in comp if x["agrees"])
        v1 = {"rule": f"predicted G0-2 pass iff xi_LP < {OPERATIONAL}", "rows": comp, "agree": n_ok,
              "compared": sum(1 for x in comp if x["agrees"] is not None)}
    return {"predictions": res, "V1": v1}


# ============================================================================= run
def cmd_run(cfg_path: str, record_path: str | None, out_dir: str, pred: str, log: str, g01: bool, n_draws: int) -> dict:
    import crossover as CX
    import g0_gate as GG

    ph = _sha_file(pred)
    hs = [e for e in logged_hashes(log) if e["sha256"] == ph]
    if ph is None or not hs:
        raise SystemExit(f"predictions {pred} are not hashed in {log}: run 'run_entanglement.py hash-predictions' before "
                         f"the re-run tables are opened")
    t_pred = hs[0]["utc"]
    tool_sha, reg_sha = _sha_file(TOOL), _sha_file(TOOL_REGISTERED)
    if tool_sha is None or tool_sha != reg_sha:
        raise SystemExit(f"{TOOL} ({tool_sha}) differs from the registered copy {TOOL_REGISTERED} ({reg_sha})")
    freeze = json.load(open("g0_code_freeze.json", encoding="utf-8"))
    want_cx = (freeze.get("files") or freeze).get("crossover.py")
    if _sha_file("crossover.py") != want_cx:
        raise SystemExit("crossover.py differs from its g0_code_freeze.json entry")
    t_open = _utc()
    os.makedirs(out_dir, exist_ok=True)
    cfg = GG.load_config(cfg_path)
    ent, corr, g01r, inputs = {}, {}, {}, []
    for base, bb in cfg["bases"].items():
        fam = bb["family"]
        outp = os.path.join(out_dir, f"ent_{base}.json")
        argv = [sys.executable, TOOL, "--g0-dir", bb["dir"], "--family", fam, "--targets", "german,books",
                "--codes", "code,pycode", "--repo", HERE, "--out", outp]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=7200)
        with open(os.path.join(out_dir, f"ent_{base}.log"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(p.stdout + p.stderr)
        if p.returncode != 0:
            raise SystemExit(f"entanglement.py failed on {base}: {p.stderr[-2000:]}")
        doc = json.load(open(outp, encoding="utf-8"))
        for r in doc["rows"]:                               # cells keyed by the base name of the config
            r["cell"] = base + "|" + r["cell"].split("|", 1)[1]
        for r in doc["other_designs"]:
            r["cell"] = base + "|" + r["cell"].split("|", 1)[1]
        ent[base] = doc
        t = CX.reg_load_dir(bb["dir"], fam, layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
        corr[base] = correlations(t)
        if g01:
            g01r[base] = g01_rows(t, base, int(bb.get("k", 6 if fam == "moe" else 16)), n_draws)
        for c, s in sorted(t.sources.items()):
            for role in ("selection", "estimation"):
                f = (s.get(role) or {}).get("file")
                if f:
                    inputs.append({"path": f, "sha256": _sha_file(f), "role": f"routing table {base}/{c}/{role}"})
    record = json.load(open(record_path, encoding="utf-8")) if record_path and os.path.exists(record_path) else None
    chk = check_predictions(ent, record)
    summ = {"prereg": "v1.3 sec. 4.9 (descriptive; main-text figure; no decision depends on it)",
            "predictions": {"file": pred, "sha256": ph, "hashed_utc": t_pred}, "tables_opened_utc": t_open,
            "order_ok": t_pred < t_open, "tool": {"path": TOOL, "sha256": tool_sha}, "crossover_sha256": want_cx,
            "config": {"path": cfg_path, "sha256": _sha_file(cfg_path)},
            "gate_record": {"path": record_path, "sha256": _sha_file(record_path)} if record_path else None,
            "entanglement": {b: {"rows": d["rows"], "other_designs": d["other_designs"], "L": d["L"], "N": d["N"]}
                             for b, d in ent.items()},
            "correlations": corr, "g01": g01r if g01 else None, "check": chk, "inputs": inputs, "created_utc": _utc()}
    with open(os.path.join(out_dir, "entanglement_summary.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summ, fh, indent=1, default=float)
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="routing entanglement (prereg v1.3 sec. 4.9) with the prediction protocol")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash-predictions")
    h.add_argument("--note", default=os.path.join("theory", "v4", "theory_note_v4_addendum.md"))
    h.add_argument("--out", default=os.path.join("results", "entanglement", "predictions_E1_E7.md"))
    h.add_argument("--log", default=os.path.join("logs", "entanglement_predictions.log"))
    r = sub.add_parser("run")
    r.add_argument("--g0-config", required=True)
    r.add_argument("--gate-record", default=None)
    r.add_argument("--out-dir", default=os.path.join("results", "entanglement"))
    r.add_argument("--predictions", default=os.path.join("results", "entanglement", "predictions_E1_E7.md"))
    r.add_argument("--log", default=os.path.join("logs", "entanglement_predictions.log"))
    r.add_argument("--no-g01", action="store_true")
    r.add_argument("--n-draws", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == "hash-predictions":
        res = cmd_hash(a.note, a.out, a.log)
        print(f"[entanglement] predictions {res['file']} sha256 {res['sha256']} hashed {res['utc']}"
              f"{'' if res['new'] else ' (already logged)'}")
        return 0
    s = cmd_run(a.g0_config, a.gate_record, a.out_dir, a.predictions, a.log, not a.no_g01, a.n_draws)
    for p in s["check"]["predictions"]:
        print(f"[entanglement] {p['id']}: {p['result']}  ({p['statement']})")
    if s["check"]["V1"]:
        print(f"[entanglement] V1: {s['check']['V1']['agree']}/{s['check']['V1']['compared']} G0-2 rows reproduced")
    print(f"[entanglement] order predictions {s['predictions']['hashed_utc']} < tables {s['tables_opened_utc']}: {s['order_ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
