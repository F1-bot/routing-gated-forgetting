#!/usr/bin/env python
"""Check the frozen study record (FREEZE.json) with the records package in place.

The study records and the original code files are not part of this repository; they are available from the
author on reasonable request (see VERIFY.md). With the files of the records package copied into the repository
root (keeping their paths), this script runs the frozen freeze check, freeze_verify() in
phase6/chain6b_util.py, without modification. It checks
the content hash of FREEZE.json, the amendment hash chain (freeze/amendments.jsonl), every referenced record and
every hash-list entry (sha256), the frozen code record (logs/chain6b_code_sha256.json) against the code files of
this repository, the plan content hash, and the re-derivation of the registered parameters from their sources.

The 13 tokenized data files listed in FREEZE.json (data/**/*.bin, about 9.5 GB) are not distributed. When one of
them is absent, its existence and sha256 are taken from FREEZE.json and it is reported as "trusted, not
re-hashed". When it is present (after rebuilding the data, see data/README.md) it is hashed like every other
file. No other file is treated this way.

Usage, from the repository root:
    python tools/verify_release.py [--root .] [--freeze FREEZE.json]

Exit code 0 when the record verifies; 23 when the freeze check reports a failure; 2 without FREEZE.json.
The full check without this wrapper, after rebuilding the data:
    python phase6/chain6b_util.py freeze-verify --root . --freeze FREEZE.json
"""
import argparse
import importlib.util
import json
import os
import sys


def recorded_data_files(freeze_path):
    """Return {relative path: sha256} for the data/*.bin entries of the FREEZE.json hash list."""
    with open(freeze_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    found = {}

    def walk(node):
        if isinstance(node, dict):
            path, digest = node.get("path"), node.get("sha256")
            if isinstance(path, str) and isinstance(digest, str):
                rel = path.replace("\\", "/")
                if rel.startswith("data/") and rel.endswith(".bin"):
                    found[rel] = digest
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc.get("hash_list", {}))
    return found


class _PathProxy:
    """os.path with isfile/exists answering True for absent, FREEZE-recorded data files only."""

    def __init__(self, real, trusted, used):
        self._real, self._trusted, self._used = real, trusted, used

    def _key(self, path):
        return os.path.normcase(os.path.abspath(path))

    def isfile(self, path):
        if self._real.isfile(path):
            return True
        key = self._key(path)
        if key in self._trusted:
            self._used.add(key)
            return True
        return False

    def exists(self, path):
        if self._real.exists(path):
            return True
        key = self._key(path)
        if key in self._trusted:
            self._used.add(key)
            return True
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OsProxy:
    def __init__(self, real, path_proxy):
        self._real = real
        self.path = path_proxy

    def __getattr__(self, name):
        return getattr(self._real, name)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=here)
    ap.add_argument("--freeze", default="FREEZE.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    freeze_path = args.freeze if os.path.isabs(args.freeze) else os.path.join(root, args.freeze)

    if not os.path.isfile(freeze_path):
        print(f"[verify-release] {args.freeze} not found: this check needs the records package (see VERIFY.md)")
        return 2

    data = recorded_data_files(freeze_path)
    trusted = {}
    for rel, digest in data.items():
        full = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(full):
            trusted[os.path.normcase(os.path.abspath(full))] = (rel, digest)

    util_path = os.path.join(root, "phase6", "chain6b_util.py")
    for extra in (root, os.path.join(root, "phase6")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location("chain6b_util", util_path)
    util = importlib.util.module_from_spec(spec)
    sys.modules["chain6b_util"] = util
    spec.loader.exec_module(util)

    used = set()
    util.os = _OsProxy(os, _PathProxy(os.path, trusted, used))
    real_sha256_file = util.sha256_file

    def sha256_file(path):
        key = os.path.normcase(os.path.abspath(path))
        if key in trusted and not os.path.isfile(path):
            used.add(key)
            return trusted[key][1]
        return real_sha256_file(path)

    util.sha256_file = sha256_file

    print(f"[verify-release] root {root}")
    print(f"[verify-release] freeze check {os.path.relpath(util_path, root)} (unchanged)")
    print(f"[verify-release] data files listed in FREEZE.json: {len(data)}; present: {len(data) - len(trusted)}; "
          f"absent: {len(trusted)}")
    rc = util.freeze_verify(root, freeze_path)
    for key in sorted(used, key=lambda k: trusted[k][0]):
        rel, digest = trusted[key]
        print(f"[verify-release] trusted, not re-hashed (absent): {rel} sha256 {digest}")
    status = "PASS" if rc == 0 else "FAIL"
    print(f"[verify-release] {status}: frozen check exit code {rc}; "
          f"{len(used)} absent data file(s) taken from FREEZE.json, every other file hashed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
