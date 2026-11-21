#!/usr/bin/env python
"""Fetch the upstream NESTOR code at the commit used by this study and create the import shim.

1. Clones https://github.com/Ma-Lab-Berkeley/Nestor.git into repo/ and checks out commit
   c797821bb2b933523d863b8cbb019f3473197ac5 (requires git).
2. Creates the package shim/nestor/ (empty __init__.py files) and copies the upstream file
   repo/nestor/data/token_documents.py to shim/nestor/data/ and to modal/ (imported by run_exp.py,
   tok_files.py and the cloud scripts).

No upstream file is modified. Existing files are never overwritten.

Usage, from the repository root:
    python tools/setup_upstream.py [--root .] [--repo-url URL]
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys

URL = "https://github.com/Ma-Lab-Berkeley/Nestor.git"
COMMIT = "c797821bb2b933523d863b8cbb019f3473197ac5"


def run(cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=here)
    ap.add_argument("--repo-url", default=URL)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    repo = os.path.join(root, "repo")

    if not os.path.isdir(os.path.join(repo, ".git")):
        if os.path.exists(repo) and os.listdir(repo):
            sys.exit(f"{repo} exists and is not a git clone; move it away first")
        run(["git", "clone", args.repo_url, repo])
    run(["git", "-c", "advice.detachedHead=false", "checkout", COMMIT], cwd=repo)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != COMMIT:
        sys.exit(f"repo/ is at {head}, expected {COMMIT}")

    src = os.path.join(repo, "nestor", "data", "token_documents.py")
    if not os.path.isfile(src):
        sys.exit(f"missing upstream file {src}")
    shim_data = os.path.join(root, "shim", "nestor", "data")
    os.makedirs(shim_data, exist_ok=True)
    for init in (os.path.join(root, "shim", "nestor", "__init__.py"), os.path.join(shim_data, "__init__.py")):
        if not os.path.exists(init):
            open(init, "w").close()
    for dst in (os.path.join(shim_data, "token_documents.py"), os.path.join(root, "modal", "token_documents.py")):
        if os.path.exists(dst):
            if not filecmp.cmp(src, dst, shallow=False):
                sys.exit(f"{dst} exists and differs from the upstream file; left unchanged")
            continue
        shutil.copyfile(src, dst)
        print("copied", os.path.relpath(dst, root))
    print(f"upstream NESTOR {COMMIT} in repo/; shim/ ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
