import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
"""Build data/claims_ledger.json: every numeric token printed in the TeX sources, with where it appears and the
source in which it was found. Run in the environment that holds the source documents (paths in SOURCES).
Matching: data/*.json numeric leaves at the printed rounding (tokens with >= 3 significant digits), then the
source documents as standalone numbers. Tokens without a match must be covered by MANUAL entries.
Context snippets are kept only for matches in the shipped data files (data/*.json); the internal source documents are
not quoted (paper v1.5.13)."""
import json, re, glob, collections  # noqa: E402
from pathlib import Path  # noqa: E402
ROOT = Path(__file__).resolve().parent.parent
SOURCES = json.loads((ROOT / "scripts" / "ledger_sources.json").read_text(encoding="utf-8"))
MANUAL = json.loads((ROOT / "scripts" / "ledger_manual.json").read_text(encoding="utf-8"))

def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.splitlines())

def visible(s):
    s = strip_comments(s)
    s = re.sub(r"\\(label|ref|eqref|cite[a-z]*|input|includegraphics|begin|end|url|nolinkurl|setlength)\*?(\[[^\]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"\{\\tabcolsep\}\{[^}]*\}|p\{[\d.]+cm\}|\\addlinespace\[[^\]]*\]|\\definecolor\{[^}]*\}\{[^}]*\}\{[^}]*\}", " ", s)
    s = re.sub(r"(?<=\d)--(?=[$\d])", " to ", s)  # TeX ranges such as 0.48--0.63
    s = re.sub(r"\\texttt\{[0-9a-f]{7,40}\}", " ", s)  # commit identifiers are not numbers
    s = re.sub(r"\\sha\{[0-9a-f]{7,64}\}", " ", s)  # SHA-256 values (macro \sha) are not numbers
    s = s.replace("$-$", "-").replace("$+$", "+").replace("{,}", "")
    s = re.sub(r"(\d)\\times10\^\{(-?\d+)\}", r"\1e\2", s)
    s = re.sub(r"\$(\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}\$", r"\1e\2", s)
    return s

TOKEN = re.compile(r"(?<![A-Za-z_\\\d.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e-?\d+)?(?![\d])")

def included_files():
    """main.tex and every file reached from it by \\input, in reading order (unused tables are not scanned)."""
    seen, order = set(), []
    def walk(f):
        if f in seen or not f.exists():
            return
        seen.add(f); order.append(f)
        for m in re.finditer(r"\\input\{([^}]+)\}", strip_comments(f.read_text(encoding="utf-8"))):
            name = m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"
            walk(ROOT / name)
    walk(ROOT / "main.tex")
    ff = ROOT / "fills" / "full_fills.tex"  # fills of the remaining blocks (scripts/fill_full.py): scanned for the ledger
    if ff.exists() and ff not in seen:
        seen.add(ff); order.append(ff)
    return order

def tex_tokens():
    out = collections.defaultdict(set)
    files = included_files()
    for f in files:
        s = visible(f.read_text(encoding="utf-8"))
        for m in TOKEN.finditer(s):
            out[m.group(0)].add(f.relative_to(ROOT).as_posix())
    return out

def norm(t):
    return t.lstrip("+-").replace(",", "")

def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, o

def sig_digits(t):
    d = norm(t).split("e")[0].replace(".", "").lstrip("0")
    return len(d)

def data_match(t, data):
    n = norm(t)
    if "e" in n:
        target = float(n); dec = None
    else:
        dec = len(n.split(".")[1]) if "." in n else 0
        target = float(n)
    for code, name, leafs in data:
        for path, v in leafs:
            for vv in (v, abs(v)):
                if dec is None:
                    if target and abs(vv - target) / abs(target) < 0.05 and f"{vv:.1e}" == f"{target:.1e}":
                        return code, name, path, v
                elif dec == 0:
                    if isinstance(vv, int) and vv == int(target):
                        return code, name, path, v
                elif round(vv, dec) == target:
                    return code, name, path, v
    return None

def doc_text(p):
    s = Path(p).read_text(encoding="utf-8", errors="replace")
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u00a0", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)_(?=\d{3})", "", s)
    return s

def doc_match(t, docs):
    n = norm(t)
    pat = re.compile(r"(?<![\d.])" + re.escape(n) + r"(?![\d])")
    alts = [pat]
    if "e" in n:
        mant, ex = n.split("e")
        sign, digs = ("-", ex[1:]) if ex.startswith("-") else ("", ex)
        alts.append(re.compile(re.escape(mant) + r"\s*(?:e|E|\u00d7\s*10\^?|x\s*10\^?)\s*" + re.escape(sign) + r"0*" + re.escape(digs) + r"(?!\d)"))
    for code, desc, loc, text in docs:
        for pt in alts:
            for m in pt.finditer(text):
                if spurious(text, m.start(), m.end()):
                    continue
                a, b = max(0, m.start() - 60), min(len(text), m.end() + 60)
                return code, " ".join(text[a:b].split())
    return None

def spurious(text, i, j):
    """True for an occurrence that is part of an identifier, a link, a date, a version or a heading number.
    Unit suffixes (1.739B, 4k, 16 GB) are kept."""
    k = text.rfind("{{artifact:", 0, i)
    if k >= 0 and text.find("}}", k) >= j:
        return True
    left, right = text[max(0, i - 12):i], text[j:j + 12]
    if left[-1:].isalpha() or left[-1:] in "_/#" or right[:1] in "_/":
        return True
    a, b = i, j
    while a > 0 and text[a - 1].isalnum():
        a -= 1
    while b < len(text) and text[b].isalnum():
        b += 1
    word = text[a:b]
    if len(word) >= 6 and re.fullmatch(r"[0-9a-f]*[a-f][0-9a-f]*", word):
        return True  # hash fragment such as e9dcf52b
    if re.search(r"[0-9a-f]*[a-f][0-9a-f]*-$", left) or re.match(r"-[0-9a-f]*[a-f][0-9a-f]*(?:-|\b)", right):
        return True  # a group of a UUID such as b0567a3b-305b-43a4
    if re.search(r"\d{4}-$", left) or re.search(r"\d{4}-\d{2}-$", left) or re.match(r"-\d{2}-\d{2}", right):
        return True  # dates
    if re.search(r"(?:\bv|version |§|#{2,6} )$", left) or re.search(r"\bv\d+\.$", left):
        return True  # versions and heading numbers
    return False

def main():
    toks = tex_tokens()
    data = []
    for code, name in SOURCES["data"]:
        o = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        data.append((code, name, list(leaves(o))))
    docs = [(d["code"], d["desc"], d["location"], doc_text(d["path"])) for d in SOURCES["docs"]]
    docs += [(code + "-text", name, "data/" + name, doc_text(ROOT / "data" / name)) for code, name in SOURCES["data"]]
    GEN = SOURCES["generated_tables"]
    ledger = {}
    for t in sorted(toks, key=lambda x: (norm(x).replace(".", "").isdigit(), x)):
        files = sorted(toks[t])
        entry = {"files": files}
        # MANUAL keys "token@file" give a file-specific source where the same number means something else
        byf = {f: MANUAL[f"{t}@{f}"] for f in files if f"{t}@{f}" in MANUAL}
        rest = [f for f in files if f not in byf]
        if byf:
            entry["by_file"] = {f: dict(v, kind="manual") for f, v in byf.items()}
        if not rest:
            entry["kind"] = "manual-by-file"
        elif t in MANUAL:
            entry.update(MANUAL[t]); entry["kind"] = "manual"
        elif all(f in GEN for f in rest):
            entry.update({"kind": "generated", "source": "; ".join(sorted({GEN[f] for f in rest}))})
        else:
            docm = doc_match(t, docs)
            if docm:
                entry.update({"kind": "document", "document": docm[0]})
                if docm[0].endswith("-text"):  # context only from the shipped data files; internal documents are not quoted
                    entry["context"] = docm[1]
            else:
                n_ = norm(t); dec_ = len(n_.split(".")[1]) if ("." in n_ and "e" not in n_) else 0
                dm = data_match(t, data) if ("e" in n_ or sig_digits(t) >= 3 or (dec_ >= 3 and sig_digits(t) >= 2)) else None
                if dm:
                    entry.update({"kind": "data-automatic", "source": f"data/{dm[1]}", "field": dm[2], "value": dm[3]})
                else:
                    entry["kind"] = "UNMATCHED"
        ledger[t] = entry
    out = ROOT / "data" / "claims_ledger.json"
    tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(ledger, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out)
    kinds = collections.Counter(e["kind"] for e in ledger.values())
    print("tokens:", len(ledger), dict(kinds))
    print("UNMATCHED:", [t for t, e in ledger.items() if e["kind"] == "UNMATCHED"])
    print("data-automatic (text tokens):", {t: e["files"] for t, e in ledger.items() if e["kind"] == "data-automatic"})

if __name__ == "__main__":
    main()
