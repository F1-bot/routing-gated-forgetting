import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Checks for paper v1.5.13 (run from anywhere; exit code 1 if a hard check fails).
All checks read main.tex and the files it reaches by \\input. 1. writing rules (banned words and constructions), 2. citation keys = bibliography keys,
3. \resultTBD inventory, 4. every printed number is in data/claims_ledger.json, and curated data entries
reproduce from data/*.json at the printed rounding, 5. generated tables regenerate byte-identically,
6. every included figure exists, 7. no list environment or \\item (lists are written as prose),
8. core fills (scripts/fill_core.py), if present: every filled ID is a slot in the text and the fills file matches its manifest."""
import json, re, subprocess, tempfile, filecmp  # noqa: E402
from pathlib import Path  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_ledger as BL  # noqa: E402

BANNED = [r"\bdelv\w*", r"\bcrucial(ly)?\b", r"\bnotabl[ey]\b", r"\bpivotal\b", r"\blandscapes?\b", r"\brealms?\b", r"\btapestr\w*",
          r"\bunderscor\w*", r"\bshowcas\w*", r"\bleverag\w*", r"\bseamless\w*", r"\bintricate\w*", r"a testament to", r"\bunlock\w*",
          r"\bharness\w*", r"\bnavigat\w*", r"in conclusion", r"\bnot just\b", r"\bnot only\b", r"\bit is not\b", r"\bit's not\b", r"\bisn't\b"]
REVIEW = [r"\brather than\b", r"---", "\u2014"]

def files():
    return BL.included_files()

def resolve(obj, path):
    for part in re.findall(r'\["[^"]*"\]|\[\d+\]|[^.\[\]]+', path):
        if part.startswith('["'):
            obj = obj[part[2:-2]]
        elif part.startswith("["):
            obj = obj[int(part[1:-1])]
        else:
            obj = obj[part]
    return obj

def main():
    hard = []
    texts = {f: f.read_text(encoding="utf-8") for f in files()}
    vis = {f: BL.strip_comments(s) for f, s in texts.items()}
    # 1 writing rules
    hits = [(f.relative_to(ROOT).as_posix(), m.group(0)) for f, s in vis.items() for pat in BANNED for m in re.finditer(pat, s, re.I)]
    review = [(f.relative_to(ROOT).as_posix(), m.group(0)) for f, s in vis.items() for pat in REVIEW for m in re.finditer(pat, s, re.I)]
    print(f"[1] banned words/constructions: {len(hits)}", hits[:20])
    print(f"    for review (rather than / em dashes): {len(review)}", review[:20])
    if hits: hard.append("writing rules")
    # 2 citations
    cited = set()
    for s in vis.values():
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", s):
            cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    bib = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", (ROOT / "refs.bib").read_text(encoding="utf-8")))
    print(f"[2] cited keys {len(cited)}, bib entries {len(bib)}; cited-not-in-bib {sorted(cited - bib)}; bib-not-cited {sorted(bib - cited)}")
    if cited - bib or bib - cited: hard.append("citations")
    # 3 placeholders
    tbd = sorted({m.group(1) for s in vis.values() for m in re.finditer(r"\\resultTBD\{([^}]*)\}", s)})
    print(f"[3] \\resultTBD IDs ({len(tbd)}):", ", ".join(tbd))
    # 4 numbers
    led = json.loads((ROOT / "data" / "claims_ledger.json").read_text(encoding="utf-8"))
    toks = BL.tex_tokens()
    missing = sorted(set(toks) - set(led))
    print(f"[4] printed numbers {len(toks)}; not in ledger: {missing}")
    if missing: hard.append("ledger coverage")
    checked = failed = 0; fails = []
    cache = {}
    pairs = [(t, e) for t, e in led.items()] + [(t, be) for t, e in led.items() for be in e.get("by_file", {}).values()]
    for t, e in pairs:
        if e.get("kind") != "manual" or not e.get("source", "").startswith("data/"):
            continue
        fld = e["field"]
        if any(x in fld for x in (" x ", "*", "(", ";", "derived")):
            continue
        src = ROOT / e["source"].split(" ")[0]
        obj = cache.setdefault(src, json.loads(src.read_text(encoding="utf-8")))
        try:
            v = resolve(obj, fld.split(" ")[0])
        except Exception as ex:  # noqa: BLE001
            failed += 1; fails.append((t, fld, f"path error {ex}")); continue
        n = t.lstrip("+-").replace(",", "")
        ok = False
        if isinstance(v, (int, float)):
            if "e" in n:
                ok = f"{abs(v):.1e}" == f"{float(n):.1e}"
            else:
                dec = len(n.split(".")[1]) if "." in n else 0
                ok = round(abs(v), dec) == float(n) or (dec == 0 and abs(v) == float(n))
        checked += 1
        if not ok:
            failed += 1; fails.append((t, fld, v))
    print(f"    curated data entries machine-checked: {checked}; mismatches: {failed}", fails[:10])
    if failed: hard.append("ledger values")
    # 5 generated tables
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, TABLES_OUT=td)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py")], env=env, check=True, capture_output=True)
        diff = [p.name for p in Path(td).glob("*.tex") if not filecmp.cmp(p, ROOT / "tables" / p.name, shallow=False)]
        print(f"[5] generated tables regenerated: {sorted(p.name for p in Path(td).glob('*.tex'))}; differing: {diff}")
        if diff: hard.append("tables")
    # 6 figures
    figs = sorted({m.group(1) for s in vis.values() for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", s)})
    absent = [f for f in figs if not (ROOT / f).exists()]
    print(f"[6] figures {figs}; missing: {absent}")
    if absent: hard.append("figures")
    # 7 lists
    lists = [(f.relative_to(ROOT).as_posix(), s[:m.start()].count("\n") + 1, m.group(0)) for f, s in vis.items()
             for m in re.finditer(r"\\begin\{(?:itemize|enumerate|description|list|compactitem|inparaenum)\}|\\item\b", s)]
    inline = [(f.relative_to(ROOT).as_posix(), s[:m.start()].count("\n") + 1, m.group(0)) for f, s in vis.items()
              for m in re.finditer(r"(?<![\w~])\((?:[1-9]|i{1,3}|iv|vi?|[a-h])\)(?=\s)", s)]
    print(f"[7] list environments or \\item: {len(lists)}", lists[:20])
    print(f"    inline enumerations such as (1) or (a), for review: {len(inline)}", inline[:20])
    if lists: hard.append("lists")
    # 8 core fills: absent before the analysis lock
    fp, mp = ROOT / "fills" / "core_fills.tex", ROOT / "fills" / "fill_manifest.json"
    if not fp.exists():
        print("[8] core fills: none; every core slot prints as pending")
    else:
        ftxt = fp.read_text(encoding="utf-8"); ids = re.findall(r"\\setfill\{([^}]*)\}", ftxt)
        man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        sh = re.search(r"sha256 ([0-9a-f]{64})", ftxt)
        slots = set(tbd) | {m.group(1) for s in vis.values() for m in re.finditer(r"\\filledor\{([^}]*)\}", s)}
        unknown = [i for i in ids if i not in slots]
        bad = (not sh) or sh.group(1) != man.get("sha256") or sorted(ids) != sorted(man.get("fills", {}))
        print(f"[8] core fills: {len(ids)} IDs from sha256 {sh.group(1)[:12] if sh else None}; not a slot in the text: {unknown}; manifest mismatch: {bad}")
        if unknown or bad: hard.append("fills")
    ffp, fmp = ROOT / "fills" / "full_fills.tex", ROOT / "fills" / "full_manifest.json"   # remaining blocks (scripts/fill_full.py)
    if ffp.exists():
        ftxt = ffp.read_text(encoding="utf-8"); ids = re.findall(r"\\setfill\{([^}]*)\}", ftxt)
        man = json.loads(fmp.read_text(encoding="utf-8")) if fmp.exists() else {}
        sh = re.search(r"sha256 ([0-9a-f]{64})", ftxt)
        slots = set(tbd) | {m.group(1) for s in vis.values() for m in re.finditer(r"\\filledor\{([^}]*)\}", s)}
        unknown = [i for i in ids if i not in slots]
        bad = (not sh) or sh.group(1) != man.get("sha256") or sorted(ids) != sorted(man.get("fills", {}))
        print(f"[8] full fills: {len(ids)} IDs from sha256 {sh.group(1)[:12] if sh else None}; not a slot in the text: {unknown}; manifest mismatch: {bad}")
        if unknown or bad: hard.append("full fills")
    print("RESULT:", "PASS" if not hard else "FAIL: " + ", ".join(hard))
    return 1 if hard else 0

if __name__ == "__main__":
    sys.exit(main())
