"""Explicit checkpoint pins for chain6b (explicit checkpoint paths under outputs).

Torch-free, stdlib only. The pins file (ckpt_pins.json) maps a name to {"path", "size", "sha256"}; run_continual.py and
stageB_windows.py read it through NESTOR_CKPT_PINS (experiment names dense_100m / moe_100m / nestor_100m), the
confirm builder through the base names moe_b1 / moe_b2 / moe_b3 / nestor. A pending entry has "path": null.

  python pins_check.py make   --out ckpt_pins.json NAME=PATH [NAME=PATH ...]   size + sha256 of every file
  python pins_check.py pin    --pins ckpt_pins.json NAME PATH                  add or replace one entry (e.g. moe_b3)
  python pins_check.py verify --pins ckpt_pins.json [--full] [--require a,b] [--g0-record g0/gate_record.json]
  python pins_check.py pin-config --pins ckpt_pins.json --name moe_b3 --config g0_moe_b3.json

verify exit codes: 0 every non-pending pin exists with the recorded size (and sha256 with --full) and every G0-record
checkpoint equals its pin; 1 otherwise; 2 a required pin is pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
G0_BASE_PIN = {"moe_b1": "moe_b1", "moe_b2": "moe_b2", "moe_b3": "moe_b3", "nestor": "nestor"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str, root: str) -> str:
    p = os.path.abspath(path)
    try:
        r = os.path.relpath(p, root)
    except ValueError:            # another drive
        return p.replace("\\", "/")
    return (p if r.startswith("..") else r).replace("\\", "/")


def entry(path: str | None, root: str) -> dict:
    if not path:
        return {"path": None, "size": None, "sha256": None, "status": "pending"}
    p = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.isfile(p):
        raise SystemExit(f"checkpoint not found: {p}")
    st = os.stat(p)
    t0 = time.time()
    h = sha256_file(p)
    return {"path": _rel(p, root), "size": st.st_size, "sha256": h,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
            "hashed": time.strftime("%Y-%m-%dT%H:%M:%S"), "hash_seconds": round(time.time() - t0, 1)}


def load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8-sig"))


def save(doc: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, path)


def resolve(doc: dict, name: str, root: str) -> str | None:
    e = (doc.get("pins") or {}).get(name)
    if not e or not e.get("path"):
        return None
    return os.path.abspath(e["path"] if os.path.isabs(e["path"]) else os.path.join(root, e["path"]))


def verify(doc: dict, root: str, *, full: bool = False, require: list[str] | None = None,
           g0_record: str | None = None) -> dict:
    rows, ok, pending_required = [], True, []
    for name, e in sorted((doc.get("pins") or {}).items()):
        if not e.get("path"):
            rows.append({"name": name, "status": "pending"})
            if require and name in require:
                pending_required.append(name)
            continue
        p = resolve(doc, name, root)
        r = {"name": name, "path": e["path"]}
        if not os.path.isfile(p):
            r["status"] = "missing"
        elif os.path.getsize(p) != e.get("size"):
            r["status"] = f"size {os.path.getsize(p)} != pinned {e.get('size')}"
        elif full and sha256_file(p) != e.get("sha256"):
            r["status"] = "sha256 differs"
        else:
            r["status"] = "ok" + (" (sha256)" if full else " (size)")
        ok &= r["status"].startswith("ok")
        rows.append(r)
    g0rows = []
    if g0_record:
        rec = load(g0_record)
        for base, cks in (rec.get("checkpoints") or {}).items():
            pin = G0_BASE_PIN.get(base)
            want = resolve(doc, pin, root) if pin else None
            used = [os.path.abspath(c if os.path.isabs(c) else os.path.join(root, c)) for c in (cks or []) if c and c != "None"]
            same = bool(want) and len(used) == 1 and os.path.normcase(used[0]) == os.path.normcase(want)
            g0rows.append({"base": base, "pin": pin, "g0_checkpoints": cks, "matches_pin": same})
            ok &= same
    return {"ok": ok and not pending_required, "pending_required": pending_required, "pins": rows, "g0_record": g0rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="checkpoint pins (ckpt_pins.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--out", required=True)
    mk.add_argument("--root", default=HERE)
    mk.add_argument("items", nargs="+", help="NAME=PATH (PATH empty or 'pending' -> pending entry)")
    pn = sub.add_parser("pin")
    pn.add_argument("--pins", required=True)
    pn.add_argument("--root", default=HERE)
    pn.add_argument("name")
    pn.add_argument("path")
    vf = sub.add_parser("verify")
    vf.add_argument("--pins", required=True)
    vf.add_argument("--root", default=HERE)
    vf.add_argument("--full", action="store_true")
    vf.add_argument("--require", default="")
    vf.add_argument("--g0-record", default=None)
    vf.add_argument("--out", default=None)
    pc = sub.add_parser("pin-config")
    pc.add_argument("--pins", required=True)
    pc.add_argument("--root", default=HERE)
    pc.add_argument("--name", required=True)
    pc.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.cmd == "make":
        doc = {"_comment": "explicit checkpoint pins; written by pins_check.py make; paths relative to the "
                           "project root", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "pins": {}}
        for it in a.items:
            name, _, path = it.partition("=")
            path = None if path.strip().lower() in ("", "pending", "none") else path
            doc["pins"][name] = entry(path, root)
            print(f"[pins] {name}: {doc['pins'][name].get('path')} {doc['pins'][name].get('sha256')}", flush=True)
        save(doc, a.out)
        return 0
    if a.cmd == "pin":
        doc = load(a.pins)
        doc.setdefault("pins", {})[a.name] = entry(a.path, root)
        doc.setdefault("amendments", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "name": a.name,
                                                 "path": doc["pins"][a.name]["path"]})
        save(doc, a.pins)
        print(f"[pins] {a.name}: {doc['pins'][a.name]['path']} sha256 {doc['pins'][a.name]['sha256']}")
        return 0
    if a.cmd == "verify":
        req = [x for x in a.require.split(",") if x]
        r = verify(load(a.pins), root, full=a.full, require=req, g0_record=a.g0_record)
        for row in r["pins"]:
            print(f"[pins] {row['name']}: {row['status']}")
        for row in r["g0_record"]:
            print(f"[pins] G0 record {row['base']}: matches pin {row['pin']}: {row['matches_pin']}")
        if a.out:
            save(r, a.out)
        if r["pending_required"]:
            print(f"[pins] required pins pending: {r['pending_required']}")
            return 2
        return 0 if r["ok"] else 1
    if a.cmd == "pin-config":
        doc = load(a.pins)
        p = resolve(doc, a.name, root)
        if not p:
            print(f"[pins] {a.name} is pending: pin it first (pins_check.py pin)")
            return 2
        cfg = load(a.config)
        cfg["checkpoint"] = _rel(p, root)
        cfg.setdefault("_pin_edits", []).append(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}: checkpoint pinned from "
                                                f"{os.path.basename(a.pins)}[{a.name}] sha256 {doc['pins'][a.name]['sha256']}")
        save(cfg, a.config)
        print(f"[pins] {a.config}: checkpoint = {cfg['checkpoint']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
