r"""audit_shim.py -- post-freeze amendment A-3 (decision: phase6/amendment_A3_post_freeze.md; staged copy and README in
confirm/rerun/). Runs the registered assumption-audit job of the frozen confirm queue (audit_moe_b1 / audit_moe_b2,
confirm/queue_confirm.json positions 83 and 84) after restoring the one function that the frozen assumption_audit.py
calls and the frozen dose_draws.py (prereg v1.3 version) no longer defines: dose_draws.base_inputs, verbatim from the
pre-v1.3 copy phase6/pre_v13/dose_draws.py (lines 113-127).

  <python> confirm/rerun/audit_shim.py <the registered argv of the job after "assumption_audit.py">

Before the audit starts (a failed check exits 3; nothing is written):
  1. the working directory is the project root (the registered argv is relative to it);
  2. assumption_audit.py, dose_draws.py and phase6/pre_v13/dose_draws.py have the sha256 values reviewed for A-3;
  3. the source of base_inputs in phase6/pre_v13/dose_draws.py (Python ast segment) equals BASE_INPUTS_SRC below,
     character for character, and its only free names are json and os;
  4. the frozen root dose_draws.py is imported; only if it has no attribute base_inputs, BASE_INPUTS_SRC is executed
     in that module's namespace (its own json and os imports), compiled under the pre-v1.3 file name and line numbers;
  5. assumption_audit.py runs as __main__ (runpy) with sys.argv = ["assumption_audit.py"] + the arguments given here
     (the registered argv) and sys.path[0] = the project root, as for "python assumption_audit.py ..."; its exit
     code is this process's exit code.
No file is edited: the restored function exists only in this process. Stdlib only; torch is imported by the audit.
"""
from __future__ import annotations

import __future__ as _future
import ast
import builtins
import hashlib
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <root>/confirm/rerun/ -> <root>
AUDIT = os.path.join(ROOT, "assumption_audit.py")
DOSE = os.path.join(ROOT, "dose_draws.py")
PRE_V13 = os.path.join(ROOT, "phase6", "pre_v13", "dose_draws.py")
SHA256 = {                                                     # reviewed for A-3 (confirm/rerun/README.md)
    AUDIT: "39adede34e6922f0d5f2afeff2dc4437d9a92602d26b4f764d8126e1c97de6d8",     # frozen, 2026-11-13
    DOSE: "bbb3c5c03dfb9cdf80f7e08740eabc7554a1fbbcbdc4525f101b34f20562aaa2",      # frozen (prereg v1.3), 2026-11-15
    PRE_V13: "e7377f11d098e85ce59c5ddfaacf5f8f00e7f4e3d1c6e810ce9474b8817250fc",  # pre-v1.3 copy, 2026-11-13
}
FIRST_LINE = 113                                               # base_inputs in phase6/pre_v13/dose_draws.py: lines 113-127
BASE_INPUTS_SRC = \
r'''def base_inputs(g0_record: str, base: str, target: str = "books") -> dict:
    """G0 directory, layout, code stream and the books-cell f of one base from a g0_gate record."""
    rec = json.load(open(g0_record, encoding="utf-8-sig"))
    code = ((rec.get("decisions") or {}).get("code_stream") or {}).get("chosen") or "code"
    cell = ((rec["gates"].get("G0-1") or {}).get("cells") or {}).get(f"{base}|{target}|{code}") or {}
    bases = ((rec.get("config") or {}).get("resolved") or {}).get("bases") or (rec.get("config") or {}).get("bases") or {}
    b = bases.get(base) or {}
    d = b.get("dir")
    if d and not os.path.isabs(d):
        d = os.path.join(os.path.dirname(os.path.abspath(g0_record)), d)
    if not d or not os.path.isdir(d):                 # records store absolute dirs; fall back to <record dir>/<base>
        d = os.path.join(os.path.dirname(os.path.abspath(g0_record)), base)
    cfg = (rec.get("config") or {}).get("resolved") or rec.get("config") or {}
    return {"dir": d, "code": code, "cell": cell, "layout": cfg.get("layout", "auto"), "half_seed": int(cfg.get("half_seed", 0)),
            "family": b.get("family", "moe")}'''


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(msg: str) -> None:
    print(f"[audit-shim] REFUSED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def _free_names(fn: ast.FunctionDef) -> set[str]:
    loads = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    stores = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    args = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    return loads - stores - args - set(dir(builtins))


def check() -> dict:
    """Steps 1-3 (read-only). Returns the facts printed in the job log."""
    if os.path.normcase(os.path.abspath(os.getcwd())) != os.path.normcase(ROOT):
        _fail(f"working directory {os.getcwd()} is not the project root {ROOT}")
    for p, want in SHA256.items():
        if not os.path.isfile(p):
            _fail(f"{p} missing")
        got = _sha(p)
        if got != want:
            _fail(f"{os.path.relpath(p, ROOT)} sha256 {got} is not the reviewed {want}")
    text = open(PRE_V13, encoding="utf-8").read()
    defs = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "base_inputs"]
    if len(defs) != 1:
        _fail(f"phase6/pre_v13/dose_draws.py defines base_inputs {len(defs)} times")
    seg = ast.get_source_segment(text, defs[0])
    if seg != BASE_INPUTS_SRC:
        _fail("the source of base_inputs in phase6/pre_v13/dose_draws.py differs from the restored text")
    if defs[0].lineno != FIRST_LINE:
        _fail(f"base_inputs starts at line {defs[0].lineno} of phase6/pre_v13/dose_draws.py, not {FIRST_LINE}")
    free = _free_names(defs[0])
    if not free <= {"json", "os"}:
        _fail(f"base_inputs uses names outside json/os: {sorted(free)}")
    return {"text_sha256": hashlib.sha256(BASE_INPUTS_SRC.encode("utf-8")).hexdigest(), "lines": f"{defs[0].lineno}-{defs[0].end_lineno}",
            "free_names": sorted(free)}


def _root_first() -> None:
    """sys.path[0] = the project root, as for "python assumption_audit.py" (run as a script, sys.path[0] is this
    file's folder confirm/rerun; imported, the root is put first)."""
    here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    first = os.path.normcase(os.path.abspath(sys.path[0] or os.curdir)) if sys.path else None
    if first == here:
        sys.path[0] = ROOT
    elif first != os.path.normcase(ROOT):
        sys.path.insert(0, ROOT)


def restore() -> dict:
    """Steps 1-4: dose_draws.base_inputs in this process (only if the frozen module lacks it)."""
    facts = check()
    _root_first()
    import dose_draws as DD
    if os.path.normcase(os.path.abspath(DD.__file__)) != os.path.normcase(DOSE):
        _fail(f"dose_draws was imported from {DD.__file__}, not {DOSE}")
    if hasattr(DD, "base_inputs"):
        facts["action"] = "present in the frozen module: not restored"
        return facts
    for name in facts["free_names"]:
        if DD.__dict__.get(name) is not sys.modules.get(name):
            _fail(f"dose_draws.{name} is not the module {name}")
    tree = ast.parse(BASE_INPUTS_SRC)
    ast.increment_lineno(tree, FIRST_LINE - 1)                 # tracebacks name the pre-v1.3 file and its lines
    code = compile(tree, PRE_V13, "exec", flags=_future.annotations.compiler_flag, dont_inherit=True)
    exec(code, DD.__dict__)                                    # the def statement, run in the frozen module's namespace
    if not callable(getattr(DD, "base_inputs", None)) or DD.base_inputs.__code__.co_filename != PRE_V13:
        _fail("restoring dose_draws.base_inputs failed")
    facts["action"] = "restored (absent from the frozen module)"
    return facts


def main(argv: list[str]) -> None:
    facts = restore()
    sys.argv = ["assumption_audit.py"] + list(argv)
    print(f"[audit-shim] A-3: dose_draws.base_inputs {facts['action']}; source phase6/pre_v13/dose_draws.py lines "
          f"{facts['lines']} (text sha256 {facts['text_sha256']}); sha256 verified: assumption_audit.py {SHA256[AUDIT][:12]}, "
          f"dose_draws.py {SHA256[DOSE][:12]}, phase6/pre_v13/dose_draws.py {SHA256[PRE_V13][:12]}; running "
          f"{' '.join(sys.argv)}", flush=True)
    runpy.run_path(AUDIT, run_name="__main__")                 # ends with sys.exit(main()) of assumption_audit.py


if __name__ == "__main__":
    main(sys.argv[1:])
