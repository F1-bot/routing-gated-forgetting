"""Canonical files of the prospective per-run forecasts (preregistration v1.3, Section 5). Standard library only.

Two files are written before the freeze and their SHA-256 values enter freeze/freeze_record.json:
  forecasts_v13.csv        one row per (run_id, step, eval_corpus, forecaster); format below
  forecasts_v13_meta.json  canonical JSON: fit runs, scalars, inputs with SHA-256, code SHA-256, target run list, csv SHA-256

CSV format (byte-exact):
  * UTF-8 without BOM, ASCII content only; line ending LF; the file ends with LF; no blank lines; no quoting;
  * header line exactly:  run_id,block,base,target,step,eval_corpus,forecaster,quantity,value
  * run_id matches [A-Za-z0-9_.-]+ ; block in BLOCKS ; base in BASES ; target in TARGETS ; step a decimal integer without
    sign or leading zeros ; eval_corpus in EVAL_CORPORA ; forecaster in FORECASTERS ; quantity = F (forgetting in nats:
    mean CE of the adapted model minus mean CE of theta0 on the full evaluation set of eval_corpus) ;
  * value: '%.6e' of a finite float (e.g. 1.234567e-02, -3.000000e-04), or NA when the forecaster is undefined for that run
    (the reason is listed in meta.na_rules) ;
  * rows sorted ascending by (run_id, int(step), eval_corpus, forecaster) as Python compares str/int; keys unique;
  * every target run has every (required step, old eval corpus, forecaster) combination; meta.targets gives the steps; the old eval
    corpora of a run are EVAL_CORPORA without the run's own target (C3 FULL runs on code or math); all rows of a run share block,
    base and target.
Meta JSON format: json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + LF, UTF-8.
Floats in meta are rounded to 9 significant digits before serialisation.

    python forecast_io.py verify forecasts_v13.csv forecasts_v13_meta.json     # exit 0 and prints both SHA-256 if valid
    python forecast_io.py hash forecasts_v13.csv forecasts_v13_meta.json
    python forecast_io.py selftest <empty_dir>                                 # writes a synthetic pair and checks the rules

Format forecasts_v13/3 (prereg v1.3 version 5) adds block DE (C1-DE German draws, prereg sec. 4.12); nothing else changed.
"""
import hashlib
import json
import math
import os
import re
import sys

HEADER = ["run_id", "block", "base", "target", "step", "eval_corpus", "forecaster", "quantity", "value"]
BLOCKS = ["B3", "C1", "C2", "C3", "DE", "DR", "M12B", "NB"]    # third base; C1; C2; C3 FULL runs (decision P-7); C1-DE German draws
                                                               # (prereg v1.3 version 5, sec. 4.12); dose-response; M12-b; NESTOR-books
                                                               # (prereg v1.3 sec. 5.1)
BASES = ["moe_b1", "moe_b2", "moe_b3", "nestor"]
TARGETS = ["books", "code", "german", "math"]                 # adaptation targets of the registered forecast set (C3: the paper's four domains)
EVAL_CORPORA = ["code", "dclm", "math"]                         # code = the registered stream (G0-8: AlgebraicStack 'code')
FORECASTERS = ["KSD", "M1", "M2", "QE", "RND", "RO", "SVC", "TF"]  # prereg v1.2 sec. 10.3 items 8, 3, 4, 2, 7, 1, 5, 6
FORMAT_ID = "forecasts_v13/3"                                  # /3: block DE added (prereg v1.3 version 5)
_RUN = re.compile(r"^[A-Za-z0-9_.-]+$")
_STEP = re.compile(r"^(0|[1-9][0-9]*)$")
_VAL = re.compile(r"^-?[0-9]\.[0-9]{6}e[+-][0-9]{2,3}$")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def fmt_value(x):
    if x is None:
        return "NA"
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("forecast values must be finite (use None for NA)")
    s = "%.6e" % x
    return "0.000000e+00" if s == "-0.000000e+00" else s


def _sort_key(r):
    return (r["run_id"], int(r["step"]), r["eval_corpus"], r["forecaster"])


def _round9(o):
    if isinstance(o, float):
        if not math.isfinite(o):
            raise ValueError("non-finite float in meta")
        return float("%.8e" % o)
    if isinstance(o, dict):
        return {str(k): _round9(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round9(v) for v in o]
    return o


def canonical_json_bytes(obj):
    return (json.dumps(_round9(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def canonical_csv_bytes(rows):
    """rows: iterable of dicts with the HEADER keys; 'value' a float or None; 'step' an int. Returns the canonical bytes."""
    out = []
    for r in rows:
        rec = dict(r)
        rec["step"] = str(int(rec["step"]))
        rec["quantity"] = rec.get("quantity", "F")
        rec["value"] = fmt_value(rec["value"]) if not isinstance(rec["value"], str) else rec["value"]
        for k in HEADER:
            if "," in str(rec[k]) or "\n" in str(rec[k]) or "\r" in str(rec[k]) or '"' in str(rec[k]):
                raise ValueError(f"illegal character in field {k}: {rec[k]!r}")
        out.append(rec)
    out.sort(key=_sort_key)
    lines = [",".join(HEADER)] + [",".join(str(rec[k]) for k in HEADER) for rec in out]
    return ("\n".join(lines) + "\n").encode("ascii")


def write_forecasts(rows, meta, csv_path, meta_path):
    """Write both files canonically; meta['csv'] is (re)filled with file name, SHA-256, row count and columns."""
    b = canonical_csv_bytes(rows)
    with open(csv_path, "wb") as f:
        f.write(b)
    meta = dict(meta)
    meta["format"] = FORMAT_ID
    meta["csv"] = {"file": os.path.basename(csv_path), "sha256": sha256_bytes(b), "n_rows": b.count(b"\n") - 1, "columns": HEADER}
    mb = canonical_json_bytes(meta)
    with open(meta_path, "wb") as f:
        f.write(mb)
    return {"csv_sha256": sha256_bytes(b), "meta_sha256": sha256_bytes(mb)}


def parse_csv_bytes(b):
    errs = []
    if b.startswith(b"\xef\xbb\xbf"):
        errs.append("BOM present")
    if b"\r" in b:
        errs.append("CR found (line endings must be LF)")
    if not b.endswith(b"\n"):
        errs.append("file does not end with LF")
    try:
        t = b.decode("ascii")
    except UnicodeDecodeError:
        return None, errs + ["non-ASCII content"]
    lines = t.split("\n")[:-1] if t.endswith("\n") else t.split("\n")
    if not lines or lines[0] != ",".join(HEADER):
        return None, errs + ["header mismatch"]
    rows = []
    for i, ln in enumerate(lines[1:], start=2):
        if ln == "":
            errs.append(f"line {i}: blank line")
            continue
        f = ln.split(",")
        if len(f) != len(HEADER):
            errs.append(f"line {i}: {len(f)} fields")
            continue
        r = dict(zip(HEADER, f))
        if not _RUN.match(r["run_id"]):
            errs.append(f"line {i}: run_id")
        for k, allowed in (("block", BLOCKS), ("base", BASES), ("target", TARGETS), ("eval_corpus", EVAL_CORPORA), ("forecaster", FORECASTERS)):
            if r[k] not in allowed:
                errs.append(f"line {i}: {k} {r[k]!r} not allowed")
        if not _STEP.match(r["step"]):
            errs.append(f"line {i}: step")
        if r["quantity"] != "F":
            errs.append(f"line {i}: quantity")
        if r["value"] != "NA" and not _VAL.match(r["value"]):
            errs.append(f"line {i}: value format {r['value']!r}")
        rows.append(r)
    return rows, errs


def verify(csv_path, meta_path):
    """Return (ok, errors, {'csv_sha256', 'meta_sha256'})."""
    with open(csv_path, "rb") as f:
        cb = f.read()
    with open(meta_path, "rb") as f:
        mb = f.read()
    rows, errs = parse_csv_bytes(cb)
    try:
        meta = json.loads(mb.decode("utf-8"))
    except Exception as e:                                      # noqa: BLE001 - any parse failure is a verification failure
        return False, errs + [f"meta not JSON: {e}"], {"csv_sha256": sha256_bytes(cb), "meta_sha256": sha256_bytes(mb)}
    if canonical_json_bytes(meta) != mb:
        errs.append("meta is not in canonical form")
    if meta.get("format") != FORMAT_ID:
        errs.append("meta.format")
    c = meta.get("csv", {})
    if c.get("sha256") != sha256_bytes(cb):
        errs.append("meta.csv.sha256 does not match the csv file")
    if rows is not None:
        if c.get("n_rows") != len(rows):
            errs.append("meta.csv.n_rows")
        keys = [_sort_key(r) for r in rows if _STEP.match(r["step"])]
        if keys != sorted(keys):
            errs.append("rows not sorted")
        if len(set(keys)) != len(keys):
            errs.append("duplicate keys")
        try:
            canon = canonical_csv_bytes(rows)
            if canon != cb:
                errs.append("csv is not in canonical form")
        except Exception as e:                                  # noqa: BLE001
            errs.append(f"csv not canonicalisable: {e}")
        targets = meta.get("targets", {})
        have, ident = {}, {}
        for r in rows:
            have.setdefault(r["run_id"], set()).add((int(r["step"]), r["eval_corpus"], r["forecaster"]))
            ident.setdefault(r["run_id"], set()).add((r["block"], r["base"], r["target"]))
        for rid, ids in ident.items():
            if len(ids) != 1:
                errs.append(f"{rid}: rows disagree on block/base/target")
        if set(have) != set(targets):
            errs.append(f"run set differs from meta.targets ({len(set(have) ^ set(targets))} ids)")
        for rid, spec in targets.items():
            tgt = next(iter(ident[rid]))[2] if rid in ident and len(ident[rid]) == 1 else None
            need = {(int(s), e, fc) for s in spec.get("steps", []) for e in EVAL_CORPORA if e != tgt for fc in FORECASTERS}
            if rid in have and have[rid] != need:
                errs.append(f"{rid}: {len(need - have[rid])} missing, {len(have[rid] - need)} unexpected combinations")
        na = meta.get("na_rules", {})
        for r in rows:
            if r["value"] == "NA" and r["forecaster"] not in na:
                errs.append(f"NA for {r['forecaster']} without an na_rule")
                break
    return (not errs), errs, {"csv_sha256": sha256_bytes(cb), "meta_sha256": sha256_bytes(mb)}


def _selftest(d):
    os.makedirs(d, exist_ok=True)
    rows = []
    targets = {"c1_moe_b1_d1_A": {"steps": [96]}, "c2_moe_b1_d1_sgd_H": {"steps": [8, 96]}, "c1de_moe_b2_d1_B": {"steps": [96]},
               "c3_moe_b2_full_code_s101": {"steps": [96]}}                 # the C3 code run stays last (own-target test below)
    kind = {"c1_moe_b1_d1_A": ("C1", "moe_b1", "books"), "c2_moe_b1_d1_sgd_H": ("C2", "moe_b1", "german"),
            "c3_moe_b2_full_code_s101": ("C3", "moe_b2", "code"), "c1de_moe_b2_d1_B": ("DE", "moe_b2", "german")}
    for rid, spec in targets.items():
        blk, bse, tgt = kind[rid]
        for s in spec["steps"]:
            for e in EVAL_CORPORA:
                if e == tgt:
                    continue                                                   # the run's own target is not an old corpus
                for k, fc in enumerate(FORECASTERS):
                    rows.append(dict(run_id=rid, block=blk, base=bse, target=tgt, step=s, eval_corpus=e, forecaster=fc,
                                     value=None if fc == "TF" and e == "dclm" and s == 8 else 0.01 * (k + 1) + 0.001 * s))
    meta = dict(prereg="preregistration v1.3", targets=targets, na_rules={"TF": "selftest"},
                fit=dict(runs=["g1_moe_b1_aa2_R1"], scalars={"moe|RO": {"a": 0.123456789123, "n": 1}}))
    cp, mp = os.path.join(d, "forecasts_v13.csv"), os.path.join(d, "forecasts_v13_meta.json")
    h = write_forecasts(rows, meta, cp, mp)
    ok, errs, _ = verify(cp, mp)
    assert ok, errs
    b = open(cp, "rb").read()
    i = b.index(b"\n") + 1
    j = b.index(b"\n", i)
    line = b[i:j]
    k = line.rindex(b"e") - 1                                                  # last mantissa digit of the first data row
    tampered = line[:k] + (b"1" if line[k:k + 1] != b"1" else b"2") + line[k + 1:]
    assert tampered != line
    open(cp, "wb").write(b[:i] + tampered + b[j:])                             # one changed digit
    ok2, errs2, _ = verify(cp, mp)
    assert not ok2 and any("sha256" in e for e in errs2), errs2
    open(cp, "wb").write(b.replace(b"\n", b"\r\n"))                            # CRLF
    ok3, errs3, _ = verify(cp, mp)
    assert not ok3 and any("CR" in e for e in errs3), errs3
    open(cp, "wb").write(b)
    ok4, _, h4 = verify(cp, mp)
    assert ok4 and h4["csv_sha256"] == h["csv_sha256"] and h4["meta_sha256"] == h["meta_sha256"]
    extra = dict(rows[-1]); extra.update(eval_corpus="code")                   # C3 code run: a row on its own target corpus
    write_forecasts(rows + [extra], meta, cp, mp)
    ok5, errs5, _ = verify(cp, mp)
    assert not ok5 and any("unexpected" in e for e in errs5), errs5
    write_forecasts(rows, meta, cp, mp)
    rows2 = list(reversed(rows))                                               # order of input rows does not matter
    assert canonical_csv_bytes(rows2) == b
    return h


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) == 3 and a[0] in ("verify", "hash"):
        if a[0] == "hash":
            for p in a[1:]:
                print(f"{sha256_file(p)}  {p}")
            return 0
        ok, errs, h = verify(a[1], a[2])
        for e in errs:
            print("ERROR", e)
        print(f"{h['csv_sha256']}  {a[1]}\n{h['meta_sha256']}  {a[2]}")
        print("RESULT:", "VALID" if ok else "INVALID")
        return 0 if ok else 1
    if len(a) == 2 and a[0] == "selftest":
        h = _selftest(a[1])
        print("selftest passed;", h)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
