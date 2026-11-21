"""Torch-free helpers of chain6b.ps1 (stdlib only; no numpy, so no conda Library\\bin PATH prefix is needed).

  python chain6b_util.py install    --root R [--reinstall] [--dry-run]   phase-5/phase-6 code -> root + sha256 code record
  python chain6b_util.py codecheck  --root R                             drift of the code set against the install record
  python chain6b_util.py check-stage B|C --root R                        completeness of results/continual.jsonl
  python chain6b_util.py freeze-write  --root R --plan P --queue Q --prereg F [--g0 F --g1 F --g2 F]
                                       [--forecasts-csv F --forecasts-meta F] [--nestor-books F] [--dose F] [--b3 F]
                                       [--decisions F[,F]] [--envelope-extension GPU_H --envelope-decision TEXT]
                                       [--approved-by NAME] [--out FREEZE.json]
                                       records the 30 gate-determined parameters of g1g2_spec.json freeze.parameters
                                       (value + source); freeze-verify re-derives each from its source (spec check 3)
                                       [--waive ID=DECISION_FILE ...] [--mock-waive-missing (MOCK only)]
                                       and hashes every file of g1g2_spec.json freeze.hash_list (revision 3; resolution:
                                       DEFAULT_HASH_MAP, optional freeze_hash_map.json) with one UTC timestamp -> FREEZE.json
                                       = freeze/freeze_record.json (+ .sha256, logs/freeze.log); spec checks 1 and 2
  python chain6b_util.py freeze-check  (freeze-write arguments; --plan/--queue/--prereg optional)   dry run, writes nothing
  python chain6b_util.py env-snapshot --root R      logs/env/pip_freeze.txt + conda_list_explicit.txt (hash list item 10)
  python chain6b_util.py freeze-verify --root R [--freeze FREEZE.json] [--print-queue]
  python chain6b_util.py mark  --root R --step S [--info TEXT]           step marker results/chain6b/<S>.done
  python chain6b_util.py done  --root R --step S                         exit 0 if the marker exists, else 1
  python chain6b_util.py status --root R

Exit codes: 0 ok; 1 failure/not done; 20 stage incomplete; 21 code drift; 22 frozen G0 file does not match the G0
code freeze (install refused); 23 FREEZE.json does not verify; 25 freeze STOP (spec check 1/2 or authors' commit);
26 freeze not performed (hash-list entries missing or unresolved); env-snapshot 3 = pip freeze written, no conda list.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime

CODE_RECORD = os.path.join("logs", "chain6b_code_sha256.json")
MARKERS = os.path.join("results", "chain6b")
BACKUP = "frozen_g0"

# (source, destination, kind). kind: replace_frozen = replaces a file of the G0 code freeze (the frozen version is
# backed up to frozen_g0/ once, after checking it against g0_code_freeze.json); replace = replaces a non-frozen root
# file (backed up once); new = new file (overwritten on reinstall); seed = copied only if the destination is missing
# (files the operator edits in place, e.g. the base-3 checkpoint pin).
INSTALL = [
    ("phase6/adapt_extras.py", "adapt_extras.py", "replace_frozen"),   # P-16: stop_after (C2 stopped at step 8)
    ("phase5/eval_windows.py", "eval_windows.py", "replace_frozen"),
    ("phase5/grad_snr.py", "grad_snr.py", "replace_frozen"),
    ("phase6/run_continual.py", "run_continual.py", "replace_frozen"),
    ("phase6/stageB_windows.py", "stageB_windows.py", "replace"),
    ("phase6/g1g2_plan.py", "g1g2_plan.py", "replace"),
    ("phase6/g2_gate.py", "g2_gate.py", "replace"),
    ("phase6/g1_gate.py", "g1_gate.py", "replace"),
    ("phase6/mock_g1g2.py", "mock_g1g2.py", "replace"),
    ("phase6/test_gates.py", "test_gates.py", "replace"),
    ("phase6/g1g2_config_MOCK.json", "g1g2_config_MOCK.json", "replace"),
    ("phase6/g1g2_config.json", "g1g2_config.json", "replace"),
    ("phase6/chain6b.ps1", "chain6b.ps1", "new"),
    ("phase6/g0_b3.ps1", "g0_b3.ps1", "new"),
    ("phase6/chain6b_util.py", "chain6b_util.py", "new"),
    ("phase6/g0_rerun_compare.py", "g0_rerun_compare.py", "new"),
    ("phase6/pins_check.py", "pins_check.py", "new"),
    ("phase6/dose_draws.py", "dose_draws.py", "new"),
    ("phase6/nestor_books_feasibility.py", "nestor_books_feasibility.py", "new"),
    ("phase6/assumption_audit.py", "assumption_audit.py", "new"),
    ("phase6/forecasts.py", "forecasts.py", "new"),
    ("phase6/forecast_probe.py", "forecast_probe.py", "new"),            # prereg v1.3 sec. 5.3 theta0 probes (GPU)
    ("phase6/forecast_io.py", "forecast_io.py", "new"),                  # sec. 5.5 format forecasts_v13/3 (block DE)
    ("phase6/budget_v13.py", "budget_v13.py", "new"),                    # final (cut order with C1-DE, 2026-11-14)
    ("phase6/g1g2_spec.json", "g1g2_spec.json", "replace"),              # spec 1.3.0 revision 3 (hash list + C1-DE)
    ("phase6/c1de_milp.py", "c1de_milp.py", "new"),                      # sec. 4.12 C1-DE exact MILP pair generator
    ("phase6/test_c1de_milp.py", "test_c1de_milp.py", "new"),
    ("inputs_v13/decisions_v13_points.md", "decisions_v13_points.md", "new"),   # hash list item 1 (P-1 to P-13)
    ("phase6/decisions_v13_round2.md", "decisions_v13_round2.md", "new"),       # decisions (a)-(d), 2026-11-14
    ("phase6/decisions_v13_round3.md", "decisions_v13_round3.md", "new"),       # decisions (a)-(g), 2026-11-14
    ("phase6/gate_rules_v13.py", "gate_rules_v13.py", "new"),
    ("phase6/weight_controls.py", "weight_controls.py", "replace_frozen"),   # scoped reverts (sec. 4.8): amendment
    ("phase6/hooks_after_B.ps1", "hooks_after_B.ps1", "replace"),
    ("phase6/run_entanglement.py", "run_entanglement.py", "new"),        # sec. 4.9 (CPU; prediction protocol)
    ("phase6/test_c3_scope.py", "test_c3_scope.py", "new"),
    ("phase6/test_forecast_probe.py", "test_forecast_probe.py", "new"),
    ("phase6/test_audit_realized.py", "test_audit_realized.py", "new"),
    ("phase6/confirm_plan.py", "confirm_plan.py", "new"),
    ("phase6/mock_chain6b.py", "mock_chain6b.py", "new"),
    ("phase6/test_phase6.py", "test_phase6.py", "new"),
    ("phase6/analyze_confirm.py", "analyze_confirm.py", "new"),          # 2.17: confirmatory analysis (prereg v1.3 v6)
    ("phase6/analyze_confirm_ext.py", "analyze_confirm_ext.py", "new"),  # 2.17: blocks (C3, C4, C5 French, sec. 11, ...)
    ("phase6/mock_confirm.py", "mock_confirm.py", "new"),                # its known-answer confirm-output simulator (tests)
    ("phase6/test_analyze_confirm.py", "test_analyze_confirm.py", "new"),
    ("phase6/ckpt_pins.json", "ckpt_pins.json", "seed"),
    ("phase6/g0_moe_b3.json", "g0_moe_b3.json", "seed"),
    ("phase6/g0_gate_config_b3.json", "g0/g0_gate_config_b3.json", "seed"),
    ("phase6/gate_record_G0_registered.json", "g0/gate_record_registered.json", "seed"),
    ("phase6/g0_probe_v2_orig.py", "g0_probe_v2_orig.py", "seed"),     # file of the G0 code freeze (swept)
]
# the code set whose hashes are recorded and checked for drift (configs only among the JSON files)
CODE_GLOBS_ROOT = ["*.py", "*.ps1", "*.csv"]
CONFIG_GLOBS_ROOT = ["g0_*.json", "g1g2_*.json", "ckpt_pins.json", "confirm_config*.json"]
CODE_DIRS = [("hooks", ["*.ps1"]), ("phase5", ["*"]), ("phase6", ["*"]), ("g0", ["g0_gate_config*.json", "gate_record_registered.json"])]
STAGE_B_MODELS = ["dense_100m", "moe_100m", "nestor_100m"]
STAGE_B_CORPORA = ["pg19", "owm", "algstack", "fw2de"]
STAGE_C_ITEMS = [("dense_100m", "equal_tokens"), ("moe_100m", "equal_tokens"), ("nestor_100m", "equal_tokens"),
                 ("nestor_100m", "equal_flops_new")]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=str)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical(obj) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def utc_epoch(value) -> float | None:
    """Epoch seconds (UTC) of a timestamp: every time comparison of the freeze is made on this one scale (
    final-text check, time zone). Numbers are epoch seconds (run_queue's start/end = time.time, zone-free);
    a string ending in 'Z' or carrying a UTC offset is converted from that zone; a naive string (now, time.strftime,
    the plan's 'created', run and queue logs) is this host's LOCAL wall-clock time and is converted explicitly
    (time.mktime) -- never read as UTC (on a UTC+2 host that would move a start 2 h later)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if len(s) > 10 and s[10] == " ":
        s = s[:10] + "T" + s[11:]
    if s[-1:] in ("Z", "z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        return time.mktime(d.timetuple()) + d.microsecond / 1e6
    return d.timestamp()


def utc_iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ----------------------------------------------------------------------------- code set
def code_set(root: str) -> dict[str, str]:
    out = {}
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        if os.path.isfile(p) and (any(fnmatch.fnmatch(f, g) for g in CODE_GLOBS_ROOT + CONFIG_GLOBS_ROOT)) \
                and not f.endswith(".tmp"):
            out[f] = sha256_file(p)
    for d, globs in CODE_DIRS:
        dp = os.path.join(root, d)
        if not os.path.isdir(dp):
            continue
        for f in sorted(os.listdir(dp)):
            p = os.path.join(dp, f)
            if os.path.isfile(p) and any(fnmatch.fnmatch(f, g) for g in globs) and not f.endswith((".tmp", ".pyc")):
                out[f"{d}/{f}"] = sha256_file(p)
    return out


def install(root: str, *, reinstall: bool = False, dry: bool = False) -> int:
    rec_path = os.path.join(root, CODE_RECORD)
    if os.path.exists(rec_path) and not reinstall:
        print(f"[install] already installed ({rec_path}); use --reinstall to re-record")
        return 0
    freeze_p = os.path.join(root, "g0_code_freeze.json")
    frozen = (json.load(open(freeze_p, encoding="utf-8-sig")).get("files") or {}) if os.path.exists(freeze_p) else {}
    rows, problems = [], []
    for src, dst, kind in INSTALL:
        sp, dp = os.path.join(root, src), os.path.join(root, dst)
        if not os.path.isfile(sp):
            problems.append(f"source missing: {src}")
            continue
        row = {"src": src, "dst": dst, "kind": kind, "sha256": sha256_file(sp)}
        if kind == "seed" and os.path.exists(dp):
            row["action"] = "kept (seed: destination exists)"
            row["dst_sha256"] = sha256_file(dp)
            rows.append(row)
            continue
        if kind in ("replace_frozen", "replace") and os.path.exists(dp):
            bp = os.path.join(root, BACKUP, os.path.basename(dst))
            if not os.path.exists(bp):
                cur = sha256_file(dp)
                if kind == "replace_frozen":
                    want = frozen.get(os.path.basename(dst))
                    want = want.get("sha256") if isinstance(want, dict) else want
                    if want and cur != want:
                        print(f"[install] REFUSED: {dst} is not the frozen G0 version ({cur[:12]} vs {want[:12]})")
                        return 22
                    row["frozen_sha256_checked"] = bool(want)
                if not dry:
                    os.makedirs(os.path.dirname(bp), exist_ok=True)
                    shutil.copy2(dp, bp)
                row["backup"] = f"{BACKUP}/{os.path.basename(dst)}"
                row["backup_sha256"] = cur
            else:
                row["backup"] = f"{BACKUP}/{os.path.basename(dst)} (existing)"
        if not dry:
            os.makedirs(os.path.dirname(dp) or root, exist_ok=True)
            tmp = dp + ".tmp"
            shutil.copy2(sp, tmp)
            os.replace(tmp, dp)
        row["action"] = "copied" if not dry else "would copy"
        rows.append(row)
    if problems:
        for p in problems:
            print("[install]", p)
        return 1
    commit = None
    head = os.path.join(root, "repo", ".git", "HEAD")
    try:
        ref = open(head).read().strip()
        if ref.startswith("ref:"):
            refp = os.path.join(root, "repo", ".git", *ref[5:].strip().split("/"))
            commit = open(refp).read().strip() if os.path.exists(refp) else ref
        else:
            commit = ref
    except OSError:
        pass
    rec = {"kind": "chain6b code record", "created": now(), "reinstall": reinstall, "repo_commit": commit,
           "install": rows, "files": code_set(root) if not dry else {}}
    rec["sha256"] = hashlib.sha256(canonical({k: v for k, v in rec.items() if k != "created"}).encode()).hexdigest()
    if dry:
        print(canonical(rows))
        return 0
    if os.path.exists(rec_path):
        arch = rec_path.replace(".json", f".{time.strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(rec_path, arch)
        rec["previous_record"] = os.path.relpath(arch, root)
    write_json(rec_path, rec)
    print(f"[install] {len(rows)} files; code record {CODE_RECORD}: {len(rec['files'])} files, sha256 {rec['sha256']}")
    return 0


def codecheck(root: str) -> int:
    rec_path = os.path.join(root, CODE_RECORD)
    if not os.path.exists(rec_path):
        print("[codecheck] no install record")
        return 1
    # hooks\*.ps1 are recorded but not drift-checked: hooks may be added while the chain runs (requirement);
    # chain6b.ps1 logs the sha256 of every hook it executes (logs\chain6b.log)
    want = {k: v for k, v in json.load(open(rec_path, encoding="utf-8"))["files"].items() if not k.startswith("hooks/")}
    cur = {k: v for k, v in code_set(root).items() if not k.startswith("hooks/")}
    changed = sorted(k for k in set(want) & set(cur) if want[k] != cur[k])
    missing = sorted(set(want) - set(cur))
    added = sorted(set(cur) - set(want))
    if changed or missing or added:
        print(f"[codecheck] DRIFT: changed {changed}, missing {missing}, added {added}")
        print("[codecheck] code = registration: re-install deliberately (chain6b.ps1 -Reinstall) and record the amendment")
        return 21
    print(f"[codecheck] {len(cur)} files match the install record")
    return 0


def check_stage(root: str, stage: str) -> int:
    res = os.path.join(root, "results", "continual.jsonl")
    keys = set()
    if os.path.exists(res):
        for line in open(res, encoding="utf-8"):
            if line.strip():
                keys.add(json.loads(line)["key"])
    if stage == "B":
        need = [f"B|{e}|before|{t}" for e in STAGE_B_MODELS for t in STAGE_B_CORPORA + ["dclm"]]
        need += [f"B|{e}|after|{c}|{s}" for c in STAGE_B_CORPORA for e in STAGE_B_MODELS for s in ("target", "dclm")]
    elif stage == "C":
        need = [f"C|{e}|{S}|{st}" for S in (8192, 16384) for e, st in STAGE_C_ITEMS]
    else:
        raise SystemExit(f"unknown stage {stage}")
    miss = [k for k in need if k not in keys]
    print(f"[check-stage {stage}] {len(need) - len(miss)}/{len(need)} results present" + (f"; missing {miss}" if miss else ""))
    return 0 if not miss else 20


# ----------------------------------------------------------------------------- freeze record
def _fileref(root: str, p: str | None) -> dict | None:
    if not p:
        return None
    ap = p if os.path.isabs(p) else os.path.join(root, p)
    if not os.path.isfile(ap):
        raise SystemExit(f"freeze input missing: {p}")
    return {"path": os.path.relpath(ap, root).replace("\\", "/"), "sha256": sha256_file(ap)}


# Gate-determined parameters of the freeze (g1g2_spec.json 1.3.0 freeze.parameters, prereg v1.3 sec. 8), in spec order.
FREEZE_PARAMETERS = [
    "holm_family", "m", "C1_confirmatory", "C1 targets per base", "n_d", "underpowered_flag", "q", "steps", "f",
    "tau (if used)", "code_stream", "beta0_1", "beta0_2", "margin_limited per MoE base", "C2_confirmatory", "C2_draws",
    "s_star", "sgd_lr", "C2 matching set (DCLM-only after G0-3)", "NESTORB_run", "NESTORB_label", "C3N_label",
    "moe_b3 status and beta0_3", "dose-response f_low and accepted draws", "forecast file SHA-256 (csv, meta)",
    "planned budget total", "contingency", "cuts applied", "envelope extension (decision P-6; 0 if none)",
    "decisions O-1 to O-9 and P-1 to P-13"]
# parameter -> key of plan_confirm.json freeze_inputs (confirm_plan.py copies them from the G0/G1/G2 records)
_FROM_PLAN = [("holm_family", "holm_family"), ("m", "m"), ("C1_confirmatory", "C1_confirmatory"),
              ("C1 targets per base", "C1_targets_per_base"), ("n_d", "n_d"), ("underpowered_flag", "underpowered_flag"),
              ("q", "q"), ("steps", "steps"), ("f", "f"), ("tau (if used)", "tau"), ("code_stream", "code_stream"),
              ("beta0_1", "beta0_1"), ("beta0_2", "beta0_2"), ("margin_limited per MoE base", "margin_limited"),
              ("C2_confirmatory", "C2_confirmatory"), ("C2_draws", "C2_draws"), ("s_star", "s_star"), ("sgd_lr", "sgd_lr"),
              ("C2 matching set (DCLM-only after G0-3)", "C2_matching_set"), ("NESTORB_label", "NESTORB_label"),
              ("C3N_label", "C3N_label")]


def _jload(root: str, rel: str):
    return json.load(open(os.path.join(root, rel), encoding="utf-8-sig"))


def _forecast_io(root: str):
    import importlib.util
    p = os.path.join(root, "forecast_io.py")
    if not os.path.isfile(p):
        raise SystemExit(f"freeze waits: {p} missing (install step)")
    spec = importlib.util.spec_from_file_location("forecast_io_freeze", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REC_DEFAULT = {"g0": "g0/gate_record.json", "g1": "g1/gate_record_G1.json", "g2": "g2/gate_record_G2.json"}
_G1_SAME = ("steps", "q", "C1_confirmatory", "NESTORB_signal", "NESTORB_label", "C3N_label")


def _gate_record_check(root: str, refs: dict, fi: dict) -> dict:
    """R5 (review v3 rec. 5): every plan-derived freeze parameter against the gate records themselves -- the
    G2 record's freeze block, the G1 decision where it holds the same key, and the G0 record (code stream; beta0 and
    margin-limited from G0-7 when G0-7R was not needed) -- not only against the plan's copy (freeze_inputs).
    Returns the record paths; raises SystemExit('freeze waits: ...') on any difference."""
    paths = {k: refs.get(k) or v for k, v in _REC_DEFAULT.items()}
    miss = [p for p in paths.values() if not os.path.isfile(os.path.join(root, p))]
    if miss:
        raise SystemExit(f"freeze waits: gate record(s) {', '.join(miss)} missing (freeze parameters are checked against them)")
    g0, g1, g2 = (_jload(root, paths[k]) for k in ("g0", "g1", "g2"))
    fz2, d1, bad = g2.get("freeze") or {}, g1.get("decision") or {}, []
    if g2.get("outcome") != "COMPLETE":
        bad.append(f"{paths['g2']}#outcome = {g2.get('outcome')!r}, not COMPLETE")
    for key in [k for _, k in _FROM_PLAN] + ["NESTORB_signal"]:
        if key not in fz2:
            bad.append(f"{key}: not in {paths['g2']}#freeze")
            continue
        if canonical(fi.get(key)) != canonical(fz2[key]):
            bad.append(f"{key}: plan {fi.get(key)!r} != {paths['g2']}#freeze {fz2[key]!r}")
        if key in _G1_SAME and key in d1 and canonical(d1[key]) != canonical(fz2[key]):
            bad.append(f"{key}: {paths['g2']}#freeze {fz2[key]!r} != {paths['g1']}#decision {d1[key]!r}")
    cs = (g0.get("decisions") or {}).get("code_stream")
    cs = cs.get("chosen") if isinstance(cs, dict) else cs
    if cs is not None and cs != fz2.get("code_stream"):
        bad.append(f"code_stream: {paths['g2']}#freeze {fz2.get('code_stream')!r} != {paths['g0']}#decisions.code_stream {cs!r}")
    if fz2.get("g0_7r") in (None, "not_needed"):
        g07 = (g0.get("gates") or {}).get("G0-7") or {}
        ml = fz2.get("margin_limited") or {}
        for i, b in ((1, "moe_b1"), (2, "moe_b2")):
            v = g07.get(b) or {}
            if "beta0" in v and canonical(v["beta0"]) != canonical(fz2.get(f"beta0_{i}")):
                bad.append(f"beta0_{i}: {paths['g2']}#freeze {fz2.get(f'beta0_{i}')!r} != {paths['g0']}#gates.G0-7.{b}.beta0 {v['beta0']!r}")
            if b in ml and "margin_limited" in v and canonical(v["margin_limited"]) != canonical(ml[b]):
                bad.append(f"margin_limited.{b}: {paths['g2']}#freeze {ml[b]!r} != {paths['g0']}#gates.G0-7.{b}.margin_limited "
                           f"{v['margin_limited']!r}")
    if bad:
        raise SystemExit("freeze waits: freeze parameters differ from the gate records (R5): " + "; ".join(bad))
    return paths


def freeze_parameters(root: str, refs: dict, envelope_extension: float, envelope_decision: str | None) -> dict:
    """The 30 gate-determined parameters, each {'value', 'source'}. refs: plan, nestor_books, dose, b3 (or None),
    forecasts_csv, forecasts_meta, decisions (list), all relative to root. Raises SystemExit('freeze waits: ...') when an
    input is missing or invalid (spec freeze.on_failure)."""
    out: dict = {}

    def put(name, value, source):
        out[name] = {"value": value, "source": source}

    plan = _jload(root, refs["plan"])
    fi = plan.get("freeze_inputs") or {}
    for name, key in _FROM_PLAN:
        if key not in fi:
            raise SystemExit(f"freeze waits: {refs['plan']} freeze_inputs has no '{key}' (parameter '{name}')")
    recs = _gate_record_check(root, refs, fi)                    # R5: against the G0/G1/G2 records directly
    for name, key in _FROM_PLAN:
        put(name, fi[key], f"{refs['plan']}#freeze_inputs.{key} = {recs['g2']}#freeze.{key}")
    nb = _jload(root, refs["nestor_books"])
    if not isinstance(nb.get("feasible"), bool):
        raise SystemExit(f"freeze waits: {refs['nestor_books']} has no feasibility decision")
    put("NESTORB_run", bool(fi.get("NESTORB_signal")) and nb["feasible"],
        f"{refs['plan']}#freeze_inputs.NESTORB_signal AND {refs['nestor_books']}#feasible")
    if refs.get("b3"):
        rec = _jload(root, refs["b3"])
        gates = rec.get("gates") or {}
        code = fi.get("code_stream")
        miss = [n for n, ok in (("G0-1 books cell", f"moe_b3|books|{code}" in ((gates.get("G0-1") or {}).get("cells") or {})),
                                ("G0-5", "moe_b3" in ((gates.get("G0-5") or {}).get("per_base") or {})),
                                ("G0-7", "moe_b3" in (gates.get("G0-7") or {}))) if not ok]
        if miss:
            raise SystemExit(f"freeze waits: {refs['b3']} is incomplete (missing {', '.join(miss)}); complete it or remove "
                             "it so that the third base is recorded as pending")
        blk = (plan.get("blocks") or {}).get("_b3")
        put("moe_b3 status and beta0_3", {"status": "complete", "beta0_3": (gates["G0-7"]["moe_b3"] or {}).get("beta0"),
                                          "planned": blk is not None, "label": (blk or {}).get("label")},
            f"{refs['b3']}#gates.G0-7.moe_b3.beta0; {refs['plan']}#blocks._b3")
    else:
        put("moe_b3 status and beta0_3", {"status": "pending", "beta0_3": None, "planned": False, "label": None},
            "g0/gate_record_b3.json absent: its runs wait for a dated addendum hashed before they start (prereg v1.3 sec. 8)")
    dose = _jload(root, refs["dose"])
    acc = {b: [{"d": e.get("d"), "seed": e.get("seed"), "data_seed": e.get("data_seed"),
                "P": (e.get("predicted") or {}).get("P")} for e in (v.get("accepted") or [])]
           for b, v in sorted((dose.get("bases") or {}).items())}
    put("dose-response f_low and accepted draws", {"f_low": dose.get("f_low"), "accepted": acc},
        f"{refs['dose']}#f_low, bases.*.accepted")
    FIO = _forecast_io(root)
    ok, errs, h = FIO.verify(os.path.join(root, refs["forecasts_csv"]), os.path.join(root, refs["forecasts_meta"]))
    if not ok:
        raise SystemExit(f"freeze waits: forecasts invalid ({'; '.join(errs[:3])})")
    put("forecast file SHA-256 (csv, meta)", {"csv": h["csv_sha256"], "meta": h["meta_sha256"]},
        f"forecast_io.py verify {refs['forecasts_csv']} {refs['forecasts_meta']}")
    for k in ("estimate_total_gpu_h", "estimate_with_contingency_gpu_h"):
        if not isinstance(plan.get(k), (int, float)):
            raise SystemExit(f"freeze waits: {refs['plan']} has no {k} (budget check, budget_v13)")
    put("planned budget total", {"total_gpu_h": plan["estimate_total_gpu_h"],
                                 "with_contingency_gpu_h": plan["estimate_with_contingency_gpu_h"],
                                 "spent_gpu_h": plan.get("estimate_spent_gpu_h"), "envelope_gpu_h": plan.get("envelope_gpu_h")},
        f"{refs['plan']}#estimate_total_gpu_h, estimate_with_contingency_gpu_h, estimate_spent_gpu_h, envelope_gpu_h")
    put("contingency", plan.get("contingency_after_cuts"), f"{refs['plan']}#contingency_after_cuts")
    put("cuts applied", {"applied": plan.get("cut_order_applied") or [], "restored": plan.get("cut_order_restored") or []},
        f"{refs['plan']}#cut_order_applied, cut_order_restored")
    ext = float(envelope_extension or 0.0)
    if plan.get("envelope_decision_required") and ext <= 0.0:
        raise SystemExit("freeze waits for the envelope decision (decision P-6): "
                         f"{plan['envelope_decision_required']} (pass --envelope-extension GPU-h --envelope-decision TEXT)")
    if ext > 0.0 and not envelope_decision:
        raise SystemExit("--envelope-extension needs --envelope-decision (where the P-6 decision is logged)")
    put("envelope extension (decision P-6; 0 if none)", {"gpu_h": ext, "decision": envelope_decision or None},
        "decision P-6 (freeze-write --envelope-extension / --envelope-decision)")
    dec = []
    for d in refs["decisions"]:
        if not os.path.isfile(os.path.join(root, d)):
            raise SystemExit(f"freeze waits: decision record {d} missing")
        dec.append(_fileref(root, d))
    put("decisions O-1 to O-9 and P-1 to P-13", dec, "decision records (path, sha256)")
    return {k: out[k] for k in FREEZE_PARAMETERS}


def _spec_parameter_check(root: str) -> None:
    sp = os.path.join(root, "g1g2_spec.json")
    if os.path.isfile(sp):
        names = (_jload(root, "g1g2_spec.json").get("freeze") or {}).get("parameters")
        if names is not None and list(names) != FREEZE_PARAMETERS:
            raise SystemExit("g1g2_spec.json freeze.parameters differ from chain6b_util.FREEZE_PARAMETERS; update the "
                             "freeze writer before freezing")


def _same_path(root: str, p: str, q: str) -> bool:
    f = lambda x: os.path.normcase(os.path.abspath(x if os.path.isabs(x) else os.path.join(root, x)))  # noqa: E731
    return bool(p) and bool(q) and f(p) == f(q)


# ----------------------------------------------------------------------------- freeze hash list (spec 1.3.0 revision 3)
POWER_SIM_V1_SHA256 = "d9c21f3b033e023b3e3f39fca21fee9094348b03a751a048ac1932b14817bf4e"   # freeze check 2 (since v1.0)
HASH_MAP_FILE = "freeze_hash_map.json"       # optional override: entry id or text -> resolution (not code-checked)
FREEZE_RECORD = "freeze/freeze_record.json"   # spec freeze step 3-4 (FREEZE.json is the same file at the chain's path)
FREEZE_LOG = os.path.join("logs", "freeze.log")
ENV_DIR = "logs/env"
BLOCKING = ("missing", "unresolved", "mismatch")
_GAP = "results/gap_trajectory/gap_analysis.py"
# How chain6b resolves each entry of g1g2_spec.json freeze.hash_list on this project's layout (entries not listed here:
# a plain root-relative path, optionally followed by '(...)', is that file; '<base>' is a wildcard). Keys:
#   files            root-relative paths; 'a|b' = the first that exists
#   globs            patterns, each must match >= 1 file
#   arg              a freeze-write argument (prereg, decisions, queue, nestor_books, forecasts_csv, forecasts_meta)
#   plan_manifests   keys of plan_confirm.json blocks._manifests ('X*' = prefix); optional_plan: absence allowed
#   queue_bins       every data bin named by the confirm queue (training, evaluation, interim, replay)
#   commit           required prefix of the authors' repository HEAD (repo/.git)
#   git_patched      tracked files modified in repo/ (git status; NESTOR_GIT or git on PATH)
#   optional         absence allowed (status optional_absent, recorded);  note / hint: recorded / printed
#   expect_sha256    {files spec: sha256} pins a verified copy; a resolved file with other content -> mismatch (STOP)
DEFAULT_HASH_MAP = {
    "the frozen preregistration version (Appendix B filled)": {"arg": "prereg"},
    "the PI decision record decisions_v13_points.md (P-1 to P-13)": {
        "arg": "decisions", "require": ["decisions_v13_points.md", "decisions_v13_round2.md", "decisions_v13_round3.md"],
        "note": "prereg v1.3 version 7 hash-list row 1 names decisions_v13_points.md, decisions_v13_round2.md and "
                "decisions_v13_round3.md; the spec revision-3 text names only the first (spec file not revised)"},
    "g1/g1_gate_config.json (with the App. C.6 ranges)": {
        "files": ["g1/g1_gate_config.json"],
        "note": "the App. C.6 ranges were never transcribed: G0-9L is not evaluated (decision (g); prereg v1.3 "
                "version 7 hash-list row 5); the G1 gate config is hashed as it stands"},
    "the confirmatory analysis script(s)": {
        "files": ["analyze_confirm.py", "analyze_confirm_ext.py", "test_analyze_confirm.py", "mock_confirm.py"],
        "note": "analyze_confirm.py implements prereg v1.3 version 6 (C1, C2, secs. 4.1-4.6, 4.11, 4.12, 5.7) and runs after "
                "the confirm queue; analyze_confirm_ext.py (called by it) implements C3 with the freeze cross-check and "
                "C3 per layer, C4, C5 French, the C2-eq alignment component and SNR, matched gain and the F_rel / "
                "bits-per-byte versions (v1.2 secs. 8.2, 9, 10, 11, 12; v1.3 sec. 4.8); its known-answer tests and the "
                "confirm-output simulator they use are hashed with it"},
    "the scripts of the blocks added by prereg v1.3 sec. 4 (assumption audit, dose-response, routing entanglement, gap "
    "trajectory, selpin evaluation)": {
        "files": ["assumption_audit.py", "dose_draws.py", "run_entanglement.py", "theory/v4/entanglement.py",
                  "inputs_v13/entanglement.py", _GAP, "eval_windows.py", "nestor_books_feasibility.py", "c1de_milp.py"],
        "note": "selpin evaluation = eval_windows.py variants; run_entanglement.py runs theory/v4/entanglement.py and checks "
                "it against inputs_v13/entanglement.py; the NESTOR-books (sec. 4.4) and C1-DE (sec. 4.12) generators are "
                "included"},
    "forecast_io.py and the forecaster script (prereg v1.3 sec. 5)": {
        "files": ["forecast_io.py", "forecasts.py", "forecast_probe.py"]},
    "power_sim_v11.py": {"files": ["power_sim_v11.py|prereg_v13/power_sim_v11.py"]},
    "power_v13.py": {"files": ["power_v13.py|prereg_v13/power_v13.py"]},
    "commit id c797821 of the authors' repository": {"commit": "c797821"},
    "run_exp.py and every locally patched file of the authors' repository (install lines of subset_adapt/adapt_extras; "
    "any change for finding eb4b1968)": {
        "files": ["run_exp.py", "subset_adapt.py", "adapt_extras.py", "grad_snr.py"], "git_patched": True,
        "note": "subset_adapt.py / adapt_extras.py hold the install lines; grad_snr.py holds the change for finding "
                "eb4b1968; modified tracked files of repo/ from git status"},
    "the queue files of every post-freeze block": {"arg": "queue"},
    "g0/gate_record.json": {
        "files": ["g0/gate_record.json", "g0/gate_record_registered.json"],
        "note": "after the G0 re-run g0/gate_record.json is the re-run record; the registered record "
                "is g0/gate_record_registered.json (both hashed)"},
    "gate_record_MOCK.json and the G1/G2 mock records": {
        "files": [".stage6/g0_mock/g0/gate_record.json|g0_mock/g0/gate_record.json",
                  "g1_gate_record_MOCK.json|g1/gate_record_MOCK.json", "g2_gate_record_MOCK.json|g2/gate_record_MOCK.json"],
        "expect_sha256": {
            ".stage6/g0_mock/g0/gate_record.json|g0_mock/g0/gate_record.json": "aacc760c88036da47a26afb2f913eb3b260632f3e6d130af374a8151e29deeb5",
            "g1_gate_record_MOCK.json|g1/gate_record_MOCK.json": "4cb2b27cad339165a5ced3e1b7bdc300b281c879b2866eee1daf86f8d7649b11",
            "g2_gate_record_MOCK.json|g2/gate_record_MOCK.json": "179f68f98cdd0b8d5a17a136e0ef983a09684fd88c71347595ce99cc0c8ebae8"},
        "note": "decision (c) of 2026-11-14: the G0 mock record is the copy the gate tests actually used -- test_gates.py "
                "reads g0_mock/g0/gate_record.json; the staging regression (logs/phase6_regress_test_gates.log, 2026-11-13 "
                "05:51) read .stage6/g0_mock/g0/gate_record.json (created 05:50:58, unchanged since), and the replica copy "
                "of the 86/86 regression parses to the same JSON after its path rewrite. The G1/G2 mock records equal their "
                "stored versions (c1b7a78b, 9ecdd0c5). The phase-4 G0 mock record gate_record_MOCK.json (stored version "
                "003f7883, sha256 7b5a0f1d...) is not read by the current gate tests and is not hashed"},
    "g0_rerun/gate_record.json (memo v7 item 1.4)": {
        "files": ["g0_rerun/gate_record.json|g0/gate_record.json"],
        "note": "the G0 re-run of chain6a/chain6b wrote g0/gate_record.json"},
    "g0/moe_b3 gate record": {"files": ["g0/gate_record_b3.json"], "optional": True,
                              "note": "pending allowed (freeze precondition 6: dated addendum before its runs)"},
    "the NESTOR-books feasibility record": {"arg": "nestor_books"},
    "forecasts_v13.csv": {"arg": "forecasts_csv"},
    "forecasts_v13_meta.json": {"arg": "forecasts_meta"},
    "C1 books draws d = 1-10 per MoE base": {"plan_manifests": ["C1:moe_b1:books", "C1:moe_b2:books"]},
    "C2 draws d = 1-8": {"plan_manifests": ["C2*"], "optional": True,
                         "note": "absent when G2 gives no C2 draws or no stable SGD lr (O-5)"},
    "A/A manifests (MoE a = 1-3, NESTOR a = 1)": {"globs": ["g1/specs/aa_*.json"]},
    "third-base books draws d = 1-5": {"plan_manifests": ["b3"], "optional": True,
                                       "note": "absent while the third base is pending (dated addendum)"},
    "dose-response draws (2 per MoE base)": {"plan_manifests": ["dose"], "optional": True,
                                             "note": "absent if cut by the P-5 order"},
    "NESTOR-books draws d = 1-3 (if run)": {"plan_manifests": ["nestor_books"], "optional": True,
                                            "note": "not run: the block was dropped at G1-3 (NESTORB_signal false, P-2); "
                                                    "prereg v1.3 version 7 hash-list row 8 marks the draws as not run"},
    "C1-DE German draws d = 1-3 per MoE base (MILP manifests; prereg v1.3 sec. 4.12)": {
        "globs": ["results/c1de/c1de_moe_b*_german.json", "results/c1de/c1de_record.json"],
        "plan_manifests": ["C1-DE*"], "optional_plan": True,
        "note": "generator output of stage C (c1de_milp.py) and, unless the block is cut or not run, the plan's manifests"},
    "M12-b German runs on A_1-A_3": {"plan_manifests": ["C1:moe_b1:books"], "optional": True,
                                     "note": "M12-b reuses A_1-A_3 of the C1 books manifest of MoE b1 (no own manifest)"},
    "C5 draws (DCLM-tilt block not run: G0-4)": {"plan_manifests": ["C5"], "optional": True},
    "every evaluation, selection, training and replay bin used after the freeze (sha256 and size)": {"queue_bins": True},
    "the sha256 bin manifest of the data rebuild (memo v7 item 1.3)": {"files": ["data/data_manifest.json"]},
    "pip freeze of the CUDA venv P\\.venv\\nestor": {
        "files": [ENV_DIR + "/pip_freeze.txt"],
        "hint": "run: .venv\\nestor\\Scripts\\python.exe chain6b_util.py env-snapshot --root ."},
    "conda list --explicit of the conda environment that supplies numpy": {
        "files": [ENV_DIR + "/conda_list_explicit.txt"],
        "hint": "run chain6b_util.py env-snapshot (conda via CONDA_EXE / PATH / the numpy env, else the env's conda-meta "
                "records), or write logs/env/conda_list_explicit.txt with: conda list --explicit --md5 -p <numpy env>"},
    "the driver scripts of the confirmatory blocks": {
        "files": ["chain6b.ps1", "chain6b_util.py", "confirm_plan.py", "run_queue.py", "ckpt_pins.json"],
        "note": "hooks are not hashed: the operator adds and renames hooks while the chain runs; chain6b.ps1 logs the sha256 of "
                "every hook it executes (logs/chain6b.log)"},
}
# The frozen text (prereg v1.3 version 7, sec. 8 hash list) differs from the spec revision-3 hash_list in three rows;
# the spec file is not revised: the difference is recorded here and copied into every hash-list record (FREEZE.json).
PREREG_V7_HASH_ROWS = [
    {"row": 1, "spec_revision_3": "the PI decision record decisions_v13_points.md (P-1 to P-13)",
     "prereg_v1.3_version_7": "decisions_v13_points.md, decisions_v13_round2.md and decisions_v13_round3.md",
     "resolution": "--decisions lists the three files; each is hashed; a missing one blocks the freeze"},
    {"row": 5, "spec_revision_3": "g1/g1_gate_config.json (with the App. C.6 ranges)",
     "prereg_v1.3_version_7": "the App. C.6 ranges were never transcribed; G0-9L is not evaluated (decision (g))",
     "resolution": "g1/g1_gate_config.json hashed as it stands; no range check"},
    {"row": 8, "spec_revision_3": "NESTOR-books draws d = 1-3 (if run)",
     "prereg_v1.3_version_7": "NESTOR-books draws not run (block dropped at G1-3)",
     "resolution": "no NESTOR-books manifest is required (optional entry; status optional_absent)"}]
_PATHLIKE = re.compile(r"^([A-Za-z0-9_.\-/<>*]+\.(?:py|json|csv|ps1|md|txt))(?:\s+\(.*\))?$")


def _abs(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, *p.replace("\\", "/").split("/"))


def _rel(root: str, p: str) -> str:
    ap = os.path.abspath(_abs(root, p))
    try:
        r = os.path.relpath(ap, root)
    except ValueError:                                           # another drive
        return ap.replace("\\", "/")
    return (ap if r.startswith("..") else r).replace("\\", "/")


def repo_commit(root: str) -> str | None:
    head = os.path.join(root, "repo", ".git", "HEAD")
    try:
        ref = open(head).read().strip()
    except OSError:
        return None
    if ref.startswith("ref:"):
        refp = os.path.join(root, "repo", ".git", *ref[5:].strip().split("/"))
        if os.path.exists(refp):
            return open(refp).read().strip()
        packed = os.path.join(root, "repo", ".git", "packed-refs")
        if os.path.exists(packed):
            for line in open(packed):
                if line.strip().endswith(ref[5:].strip()):
                    return line.split()[0]
        return None
    return ref


def _index_patched(root: str) -> list[str] | None:
    """git-free check (git's own rule): tracked files of repo/ whose working-tree content differs from .git/index (stat
    data first unless racily clean (mtime not older than the index), else the blob sha1 of the raw and of the
    CRLF-normalised content, for core.autocrlf); index versions 2-3.
    Staged-but-uncommitted changes are not covered (index vs HEAD needs the object store)."""
    import struct
    repo = os.path.join(root, "repo")
    try:
        b = open(os.path.join(repo, ".git", "index"), "rb").read()
    except OSError:
        return None
    if b[:4] != b"DIRC":
        return None
    ver, n = struct.unpack(">II", b[4:12])
    if ver not in (2, 3):
        return None
    blob = lambda d: hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()   # noqa: E731
    idx_mtime = int(os.stat(os.path.join(repo, ".git", "index")).st_mtime)
    off, out = 12, []
    for _ in range(n):
        f = struct.unpack(">10I", b[off:off + 40])
        mtime_s, mode, size = f[2], f[6], f[9]
        sha1 = b[off + 40:off + 60].hex()
        flags = struct.unpack(">H", b[off + 60:off + 62])[0]
        p = off + 62 + (2 if (ver >= 3 and flags & 0x4000) else 0)
        end = b.index(b"\0", p)
        name = b[p:end].decode("utf-8", "surrogateescape")
        off += (end - off + 8) // 8 * 8                          # 1-8 NUL bytes pad each entry to a multiple of 8
        if (mode >> 12) not in (0b1000, 0b1010):                 # regular files and symlinks (gitlinks skipped)
            continue
        fp = os.path.join(repo, *name.split("/"))
        if not os.path.isfile(fp):
            out.append(f"repo/{name}")
            continue
        st = os.stat(fp)
        if st.st_size == size and int(st.st_mtime) == mtime_s and mtime_s < idx_mtime:
            continue                                             # clean by stat (not racy: older than the index)
        data = open(fp, "rb").read()
        if sha1 not in (blob(data), blob(data.replace(b"\r\n", b"\n"))):
            out.append(f"repo/{name}")
    return sorted(out)


def _git_patched(root: str) -> tuple[list[str] | None, str]:
    """Tracked files of repo/ that differ (root-relative) and the method: git status --porcelain (NESTOR_GIT or git on
    PATH), else the git-free index check; (None, reason) if neither works."""
    import subprocess
    repo = os.path.join(root, "repo")
    if not os.path.isdir(os.path.join(repo, ".git")):
        return None, "repo/.git missing"
    git = os.environ.get("NESTOR_GIT") or shutil.which("git")
    if git:
        try:
            r = subprocess.run([git, "--no-optional-locks", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                               input="", capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            r = None
        if r is not None and r.returncode == 0:
            out = []
            for line in r.stdout.splitlines():
                if len(line) > 3:
                    p = line[3:].strip().strip('"')
                    if " -> " in p:
                        p = p.split(" -> ")[-1]
                    out.append("repo/" + p.replace("\\", "/"))
            return sorted(set(out)), "git status --porcelain --untracked-files=no"
    pf = _index_patched(root)
    if pf is None:
        return None, "git not runnable and .git/index not readable (version 2-3)"
    return pf, "working tree vs .git/index (git not runnable here; staged changes not covered)"


def queue_bins(queue_doc: dict) -> list[str]:
    """Every data bin named by a run_queue queue: DCLM_DATA_PATH, eval/interim set specs 'name=path[:tokens]', args."""
    out: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and ".bin" in o:
            for part in o.split(","):
                part = part.split("=", 1)[-1].strip()
                part = re.sub(r":\d+$", "", part)
                if part.endswith(".bin"):
                    out.add(part.replace("\\", "/"))
    walk(queue_doc)
    return sorted(out)


def _hash_map(root: str) -> tuple[dict, dict | None]:
    hm = dict(DEFAULT_HASH_MAP)
    p = os.path.join(root, HASH_MAP_FILE)
    if not os.path.isfile(p):
        return hm, None
    over = json.load(open(p, encoding="utf-8-sig"))
    hm.update(over.get("entries") or {})
    return hm, {"path": HASH_MAP_FILE, "sha256": sha256_file(p)}


def _resolve(root: str, how: dict, ctx: dict) -> dict:
    files, missing, notes, value, status = [], [], [], None, "ok"

    def have(p: str) -> bool:
        return os.path.isfile(_abs(root, p))
    expect = how.get("expect_sha256") or {}
    for spec_ in how.get("files", []):
        hit = next((x for x in spec_.split("|") if have(x)), None)
        (files.append(hit) if hit else missing.append(spec_))
        if hit and spec_ in expect:                  # a verified copy is pinned: any other content is a STOP
            got = sha256_file(_abs(root, hit))
            if got != expect[spec_]:
                status = "mismatch"
                notes.append(f"{hit}: sha256 {got[:16]}... is not the verified copy {expect[spec_][:16]}...")
            else:
                notes.append(f"{hit}: sha256 equals the verified copy ({expect[spec_][:16]}...)")
    for g in how.get("globs", []):
        import glob as _glob
        ms = sorted(_rel(root, m) for m in _glob.glob(_abs(root, g)) if os.path.isfile(m))
        (files.extend(ms) if ms else missing.append(g))
    if "arg" in how:
        v = ctx["args"].get(how["arg"])
        vals = [x for x in (v if isinstance(v, list) else [v]) if x]
        if not vals:
            missing.append(f"--{how['arg'].replace('_', '-')} (not given)")
        for x in vals:
            (files.append(_rel(root, x)) if have(x) else missing.append(_rel(root, x)))
        names = {os.path.basename(str(x)) for x in vals}
        for req in how.get("require") or []:                      # prereg v1.3 version 7 row 1: all three decision records
            if req not in names:
                missing.append(f"{req} (prereg v1.3 version 7 hash-list row 1; pass it in --{how['arg'].replace('_', '-')})")
    if "plan_manifests" in how:
        mans = ctx["manifests"]
        for key in how["plan_manifests"]:
            ks = sorted(k for k in mans if (k.startswith(key[:-1]) if key.endswith("*") else k == key))
            if not ks:
                if not how.get("optional_plan"):
                    missing.append(f"plan manifest {key} (blocks._manifests of {ctx['plan_path']})")
                else:
                    notes.append(f"plan manifest {key}: not in the plan (block cut or not run)")
                continue
            for k in ks:
                p = mans[k]["path"]
                (files.append(_rel(root, p)) if have(p) else missing.append(_rel(root, p)))
    if how.get("queue_bins"):
        bins = ctx["bins"]
        if not bins:
            missing.append(f"no data bin named by the queue {ctx['queue_path']}")
        for b in bins:
            (files.append(_rel(root, b)) if have(b) else missing.append(_rel(root, b)))
    if "commit" in how:
        c = repo_commit(root)
        value = {"repo_commit": c, "required_prefix": how["commit"]}
        if c is None:
            status = "unresolved"
            notes.append("repo/.git HEAD not readable")
        elif not c.startswith(how["commit"]):
            status = "mismatch"
            notes.append(f"repo HEAD {c} is not commit {how['commit']}")
    if how.get("git_patched"):
        pf, method = _git_patched(root)
        if pf is None:
            if status == "ok":
                status = "unresolved"
            notes.append(f"patched files of repo/ not determinable ({method}; set NESTOR_GIT to git.exe, or list them in "
                         "freeze_hash_map.json)")
        else:
            value = dict(value or {}, patched_files=pf, method=method)
            notes.append(f"repo/: {len(pf)} locally modified tracked file(s) ({method})")
            for p in pf:
                (files.append(p) if have(p) else missing.append(f"{p} (deleted in the working tree)"))
    if missing and status == "ok":
        status = "missing"
    if status == "missing" and how.get("optional") and not files:
        status = "optional_absent"
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return {"status": status, "files": uniq, "missing": missing, "notes": notes, "value": value}


def freeze_hash_list(root: str, args: dict, plan_path: str | None, queue_path: str | None, waivers: dict,
                     mock_waive: bool, utc: str) -> dict:
    """Resolve and hash g1g2_spec.json freeze.hash_list (spec 1.3.0 revision 3); spec checks 1 and 2."""
    sp = os.path.join(root, "g1g2_spec.json")
    if not os.path.isfile(sp):
        raise SystemExit("freeze waits: g1g2_spec.json missing (install step)")
    spec = json.load(open(sp, encoding="utf-8-sig"))
    cats = (spec.get("freeze") or {}).get("hash_list")
    if not cats:
        raise SystemExit("freeze waits: g1g2_spec.json has no freeze.hash_list")
    plan = json.load(open(_abs(root, plan_path), encoding="utf-8")) if plan_path and os.path.isfile(_abs(root, plan_path)) else {}
    qd = json.load(open(_abs(root, queue_path), encoding="utf-8")) if queue_path and os.path.isfile(_abs(root, queue_path)) else {}
    ctx = {"args": args, "manifests": (plan.get("blocks") or {}).get("_manifests") or {}, "plan_path": plan_path,
           "bins": queue_bins(qd), "queue_path": queue_path}
    hm, override = _hash_map(root)
    entries, hashed = [], {}
    for ci, cat in enumerate(cats, 1):
        for ei, text in enumerate(cat.get("paths") or [], 1):
            eid = f"{ci}.{ei}"
            how = hm.get(eid) or hm.get(text)
            if how is None:
                m = _PATHLIKE.match(text.strip())
                if m:
                    p = m.group(1)
                    how = {"globs": [re.sub(r"<[^>]+>", "*", p)]} if ("<" in p or "*" in p) else {"files": [p]}
            if how is None:
                r = {"status": "unresolved", "files": [], "missing": [], "notes": ["no resolution rule (add it to "
                     f"{HASH_MAP_FILE})"], "value": None}
            else:
                r = _resolve(root, how, ctx)
            e = {"id": eid, "category": cat.get("category"), "entry": text, "status": r["status"], "value": r["value"],
                 "missing": r["missing"], "notes": r["notes"] + ([how["note"]] if how and how.get("note") else []),
                 "hint": (how or {}).get("hint")}
            e["files"] = []
            for f in r["files"]:
                ap = _abs(root, f)
                if f not in hashed:
                    hashed[f] = sha256_file(ap)
                e["files"].append({"path": f, "sha256": hashed[f], "size": os.path.getsize(ap)})
            if e["status"] in BLOCKING and e["status"] != "mismatch":
                w = waivers.get(eid) or waivers.get(text)
                if w is not None:
                    e["status"], e["waiver"] = "waived", w
                elif mock_waive:
                    e["status"], e["waiver"] = "waived", {"mock": True, "note": "--mock-waive-missing (approved-by MOCK)"}
            entries.append(e)
    # check 1: a file in both g0_code_freeze.json and the hash list has the frozen sha256 unless a logged amendment
    fzp = os.path.join(root, "g0_code_freeze.json")
    frozen = (json.load(open(fzp, encoding="utf-8-sig")).get("files") or {}) if os.path.isfile(fzp) else {}
    rec_p = os.path.join(root, CODE_RECORD)
    installed = {r_["dst"].replace("\\", "/") for r_ in (json.load(open(rec_p, encoding="utf-8")).get("install") or [])
                 if r_.get("kind") == "replace_frozen"} if os.path.isfile(rec_p) else set()
    check1, stops = [], []
    for k in sorted(frozen):
        want = frozen[k].get("sha256") if isinstance(frozen[k], dict) else frozen[k]
        if k not in hashed:
            continue
        row = {"path": k, "frozen_sha256": want, "sha256": hashed[k]}
        if hashed[k] == want:
            row["status"] = "same"
        else:
            bp = os.path.join(root, BACKUP, os.path.basename(k))
            if k in installed and os.path.isfile(bp) and sha256_file(bp) == want:
                row.update(status="amended", frozen_copy=f"{BACKUP}/{os.path.basename(k)}",
                           amendment="replace_frozen install recorded in " + CODE_RECORD.replace("\\", "/"))
            elif f"g0freeze:{k}" in waivers:
                row.update(status="amended (decision)", decision=waivers[f"g0freeze:{k}"])
            else:
                row["status"] = "MISMATCH"
                stops.append(f"check 1: {k} differs from g0_code_freeze.json ({hashed[k][:12]} vs {want[:12]}) without a "
                             f"logged amendment (STOP; the operator decides; --waive g0freeze:{k}=<decision file>)")
        check1.append(row)
    ps = hashed.get("power_sim_v1.py")
    check2 = {"path": "power_sim_v1.py", "sha256": ps, "required": POWER_SIM_V1_SHA256, "ok": ps == POWER_SIM_V1_SHA256}
    if not check2["ok"]:
        stops.append(f"check 2: power_sim_v1.py sha256 {ps} != {POWER_SIM_V1_SHA256}")
    for e in entries:
        if e["status"] == "mismatch":
            stops.append(f"{e['id']} {e['entry'][:60]}: {'; '.join(e['notes'])}")
    blocking = [e for e in entries if e["status"] in ("missing", "unresolved")]
    for e in blocking:                                           # frozen drivers: the recorded hash names the version
        for m_ in e["missing"]:
            fz_ = frozen.get(m_)
            if fz_:
                e["notes"].append(f"{m_}: g0_code_freeze.json records sha256 {(fz_.get('sha256') if isinstance(fz_, dict) else fz_)[:16]}... "
                                  "(restore that version)")
    rec = {"spec": {"path": "g1g2_spec.json", "sha256": sha256_file(sp), "version": spec.get("version"),
                    "revision": spec.get("revision")},
           "map_override": override, "hashed_utc": utc, "n_files": len(hashed), "entries": entries,
           "prereg_v7_hash_list_rows": PREREG_V7_HASH_ROWS,
           "checks": {"g0_code_freeze": check1, "power_sim_v1": check2}}
    return {"record": rec, "blocking": blocking, "stops": stops}


def record_path(out: str) -> str:
    """The record copy of a freeze file: freeze/freeze_record.json for the chain's FREEZE.json (spec freeze steps 3-4);
    freeze/<stem>_record.json for any other name (tests write several records into one root)."""
    b = os.path.basename(out)
    if b.endswith("_record.json"):                               # the record itself was named
        return out.replace("\\", "/")
    return FREEZE_RECORD if b == "FREEZE.json" else f"freeze/{os.path.splitext(b)[0]}_record.json"


def _parse_waivers(root: str, items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--waive {it!r}: expected ID=DECISION_FILE (ID = entry id like 11.1, entry text, or "
                             "g0freeze:<path>)")
        k, ref = it.split("=", 1)
        if not os.path.isfile(_abs(root, ref)):
            raise SystemExit(f"--waive {k}: the logged decision {ref} does not exist")
        out[k.strip()] = {"decision": {"path": _rel(root, ref), "sha256": sha256_file(_abs(root, ref))}}
    return out


def _print_hash_list(hl: dict, tag: str) -> None:
    for e in hl["record"]["entries"]:
        if e["status"] != "ok":
            print(f"[{tag}] {e['id']:>5} {e['status']:<15} {e['entry'][:90]}")
            for m_ in e["missing"]:
                print(f"[{tag}]        missing: {m_}")
            for n_ in e["notes"]:
                print(f"[{tag}]        note: {n_}")
            if e.get("hint") and e["status"] in BLOCKING:
                print(f"[{tag}]        hint: {e['hint']}")
    for s in hl["stops"]:
        print(f"[{tag}] STOP {s}")
    n_ok = sum(e["status"] == "ok" for e in hl["record"]["entries"])
    print(f"[{tag}] hash list: {len(hl['record']['entries'])} entries ({n_ok} ok, {len(hl['blocking'])} blocking, "
          f"{len(hl['stops'])} stops), {hl['record']['n_files']} files")


def _hl_args(root: str, a) -> dict:
    return {"prereg": getattr(a, "prereg", None), "queue": getattr(a, "queue", None),
            "decisions": [d for d in (getattr(a, "decisions", "") or "").split(",") if d],
            "nestor_books": getattr(a, "nestor_books", None), "forecasts_csv": getattr(a, "forecasts_csv", None),
            "forecasts_meta": getattr(a, "forecasts_meta", None)}


POST_FREEZE_RULE = ("spec freeze check 4 (second half): no post-freeze run started earlier -- every job of the frozen "
                    "queue with a start in its queue state must have started at or after the freeze timestamp; both "
                    "sides are compared as UTC epoch seconds (utc_epoch; naive strings = this host's local time)")


def _state_of_plan(root: str, plan_path: str | None) -> str:
    try:
        plan = json.load(open(_abs(root, plan_path), encoding="utf-8-sig")) if plan_path else {}
    except (OSError, ValueError):
        plan = {}
    return (plan or {}).get("state") or "results/queue_state_confirm.json"


def post_freeze_start_check(root: str, queue_path: str | None, state_path: str | None, freeze_utc: str) -> list[str]:
    """POST_FREEZE_RULE. A post-freeze run = a job of the frozen queue; its start = the queue state's 'start'
    (run_queue: epoch seconds; a string is converted by utc_epoch). One line per violation (empty = pass)."""
    t_f = utc_epoch(freeze_utc)
    if t_f is None:
        return [f"check 4: the freeze timestamp {freeze_utc!r} is not a timestamp"]
    qp = _abs(root, queue_path) if queue_path else None
    jobs = []
    if qp and os.path.isfile(qp):
        try:                                       # the queue file itself is checked by the hash list, not here
            qd = json.load(open(qp, encoding="utf-8-sig"))
        except (OSError, ValueError):
            qd = None
        jl = qd.get("jobs") if isinstance(qd, dict) else None
        jobs = [j.get("id") for j in (jl if isinstance(jl, list) else []) if isinstance(j, dict)]
    sp = _abs(root, state_path) if state_path else None
    st, bad = {}, []
    if jobs and sp and os.path.isfile(sp):
        try:
            st = json.load(open(sp, encoding="utf-8-sig"))
        except (OSError, ValueError) as e:
            return [f"check 4: queue state {state_path} unreadable ({type(e).__name__}): the starts cannot be checked"]
    for jid in jobs:
        rec = st.get(jid) if isinstance(st, dict) else None
        if not isinstance(rec, dict) or rec.get("start") is None:
            continue
        t = utc_epoch(rec["start"])
        if t is None:
            bad.append(f"check 4: post-freeze job {jid}: start {rec['start']!r} is not a timestamp")
        elif t < t_f:
            bad.append(f"check 4: post-freeze job {jid} started {utc_iso(t)}, {(t_f - t) / 60:.1f} min before the freeze "
                       f"{utc_iso(t_f)} (both UTC)")
    return bad


def freeze_check(root: str, a) -> int:
    """Dry run of the hash list (nothing written): what still blocks the freeze."""
    if getattr(a, "mock_waive_missing", False) and getattr(a, "approved_by", None) != "MOCK":
        raise SystemExit("--mock-waive-missing is only for mock runs (--approved-by MOCK)")
    hl = freeze_hash_list(root, _hl_args(root, a), a.plan, a.queue, _parse_waivers(root, a.waive),
                          bool(getattr(a, "mock_waive_missing", False)), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _print_hash_list(hl, "freeze-check")
    c4 = post_freeze_start_check(root, a.queue, _state_of_plan(root, a.plan), hl["record"]["hashed_utc"])
    for x in c4:
        print(f"[freeze-check] STOP {x}")
    prm_ok = True                                                # spec check 3 (R5): parameters vs. the gate records
    try:
        plan = _jload(root, a.plan)
        recs = {k: getattr(a, k, None) or v for k, v in _REC_DEFAULT.items()}
        _gate_record_check(root, recs, plan.get("freeze_inputs") or {})
        print("[freeze-check] parameters: the plan's freeze_inputs equal the G0/G1/G2 gate records")
    except (SystemExit, OSError, ValueError) as e:
        prm_ok = False
        print(f"[freeze-check] parameters: {e}")
    return 25 if hl["stops"] or c4 else (26 if hl["blocking"] or not prm_ok else 0)


def env_snapshot(root: str) -> int:
    """logs/env/pip_freeze.txt (this interpreter) and logs/env/conda_list_explicit.txt (the env that supplies numpy)."""
    import importlib.util
    import subprocess
    d = os.path.join(root, *ENV_DIR.split("/"))
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    r = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"[env-snapshot] pip freeze failed: {r.stderr.strip()[:300]}")
        return 1
    with open(os.path.join(d, "pip_freeze.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# pip freeze --all of {sys.executable} (prefix {sys.prefix}, base {sys.base_prefix}); {stamp}\n")
        fh.write(r.stdout)
    sp_ = importlib.util.find_spec("numpy")
    prefix = None
    if sp_ and sp_.origin:
        p = os.path.dirname(os.path.abspath(sp_.origin))          # <prefix>/Lib/site-packages/numpy
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(p)))
    cands = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    if prefix:
        envs = os.path.dirname(prefix)
        base = os.path.dirname(envs) if os.path.basename(envs).lower() == "envs" else prefix
        cands += [os.path.join(base, "Scripts", "conda.exe"), os.path.join(base, "condabin", "conda.bat")]
    conda = next((c for c in cands if c and os.path.isfile(c)), None)
    txt = how = None
    if conda and prefix:
        r = subprocess.run([conda, "list", "--explicit", "--md5", "-p", prefix], input="", capture_output=True, text=True,
                           timeout=900)
        if r.returncode == 0:
            txt, how = r.stdout, f"conda list --explicit --md5 -p {prefix} ({conda})"
    cm = os.path.join(prefix, "conda-meta") if prefix else ""
    if txt is None and cm and os.path.isdir(cm):                 # what conda list reads: the env's package records
        recs = [json.load(open(os.path.join(cm, f), encoding="utf-8")) for f in sorted(os.listdir(cm)) if f.endswith(".json")]
        subdir = next((r_.get("subdir") for r_ in recs if r_.get("subdir") not in (None, "noarch")), "win-64")
        lines = [f"# platform: {subdir}", "@EXPLICIT"] + sorted(f"{r_['url']}#{r_['md5']}" for r_ in recs if r_.get("url"))
        txt = "\n".join(lines) + "\n"
        how = (f"conda-meta package records of {prefix} (conda executable not found; the content of conda list --explicit "
               "--md5)")
    if txt is None:
        print(f"[env-snapshot] pip freeze written; no conda and no conda-meta for the numpy env {prefix}: write "
              f"{ENV_DIR}/conda_list_explicit.txt with 'conda list --explicit --md5 -p <env>'")
        return 3
    with open(os.path.join(d, "conda_list_explicit.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {how}; {stamp}\n")
        fh.write(txt)
    print(f"[env-snapshot] {ENV_DIR}/pip_freeze.txt and {ENV_DIR}/conda_list_explicit.txt written ({how})")
    return 0


def freeze_write(root: str, a) -> int:
    plan = _jload(root, a.plan)
    if not _same_path(root, plan.get("queue", ""), a.queue):
        raise SystemExit(f"the plan names queue {plan.get('queue')}, not {a.queue}")
    _spec_parameter_check(root)
    b3 = getattr(a, "b3", None)
    refs = {"plan": a.plan, "nestor_books": a.nestor_books, "dose": a.dose, "b3": b3 if b3 and os.path.isfile(
        os.path.join(root, b3)) else None, "forecasts_csv": a.forecasts_csv, "forecasts_meta": a.forecasts_meta,
        "decisions": [d for d in (a.decisions or "").split(",") if d],
        "g0": getattr(a, "g0", None) or _REC_DEFAULT["g0"], "g1": getattr(a, "g1", None) or _REC_DEFAULT["g1"],
        "g2": getattr(a, "g2", None) or _REC_DEFAULT["g2"]}
    for k in ("nestor_books", "dose", "forecasts_csv", "forecasts_meta"):
        if not refs[k] or not os.path.isfile(os.path.join(root, refs[k])):
            raise SystemExit(f"freeze waits: {k} file {refs[k]} missing")
    if not refs["decisions"]:
        raise SystemExit("freeze waits: no decision record (--decisions decisions_v13_points.md[,...])")
    prm = freeze_parameters(root, refs, getattr(a, "envelope_extension", 0.0), getattr(a, "envelope_decision", None))
    if getattr(a, "mock_waive_missing", False) and a.approved_by != "MOCK":
        raise SystemExit("--mock-waive-missing is only for mock runs (--approved-by MOCK)")
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())          # one UTC timestamp for every hash (spec step 3)
    hl = freeze_hash_list(root, _hl_args(root, a), a.plan, a.queue, _parse_waivers(root, getattr(a, "waive", None)),
                          bool(getattr(a, "mock_waive_missing", False)), utc)
    state_rel = _state_of_plan(root, a.plan)
    c4 = post_freeze_start_check(root, a.queue, state_rel, utc)
    for x in c4:
        print(f"[freeze] STOP {x}")
    hl["stops"] = list(hl["stops"]) + c4
    if hl["stops"] or hl["blocking"]:
        _print_hash_list(hl, "freeze")
        if hl["stops"]:
            print("[freeze] STOP: spec check failed (g1g2_spec.json freeze.on_failure: log a pre-freeze amendment; the operator "
                  "decides); nothing written")
            return 25
        print("[freeze] freeze not performed: hash-list entries missing or unresolved (on_failure.missing_file: until "
              "the file exists or a logged decision removes the entry, --waive ID=<decision file>); nothing written")
        return 26
    recs = [("g0", a.g0), ("g1", a.g1), ("g2", a.g2), ("nestor_books", refs["nestor_books"]), ("dose", refs["dose"]),
            ("b3", refs["b3"])]
    doc = {"kind": "freeze record (prereg v1.3 sec. 8; g1g2_spec.json freeze)", "created": now(),
           "approved_by": a.approved_by, "notes": a.notes,
           "prereg": _fileref(root, a.prereg), "plan": _fileref(root, a.plan), "plan_content_sha256": plan.get("sha256"),
           "queue": _fileref(root, a.queue),
           "forecasts": {"csv": _fileref(root, refs["forecasts_csv"]), "meta": _fileref(root, refs["forecasts_meta"])},
           "code_record": _fileref(root, CODE_RECORD),
           "records": {k: _fileref(root, v) for k, v in recs if v},
           "parameter_refs": refs, "parameter_order": FREEZE_PARAMETERS, "parameters": prm,
           "hashed_utc": utc, "hash_list": hl["record"],
           "post_freeze_starts": {"rule": POST_FREEZE_RULE, "state": state_rel, "violations": c4}}
    doc["sha256"] = hashlib.sha256(canonical({k: v for k, v in doc.items() if k != "created"}).encode()).hexdigest()
    out = os.path.join(root, a.out)
    rec_rel = record_path(a.out)
    rec_p = os.path.join(root, *rec_rel.split("/"))
    for p_ in (out, rec_p):
        if os.path.exists(p_) and not a.force:
            raise SystemExit(f"{p_} exists (a freeze is final; --force only for mock runs)")
    write_json(out, doc)
    if os.path.normcase(os.path.abspath(out)) != os.path.normcase(os.path.abspath(rec_p)):
        os.makedirs(os.path.dirname(rec_p), exist_ok=True)
        shutil.copyfile(out, rec_p + ".tmp")
        os.replace(rec_p + ".tmp", rec_p)
    h = sha256_file(rec_p)
    with open(rec_p + ".sha256", "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{h}  {os.path.basename(rec_p)}\n")
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    with open(os.path.join(root, FREEZE_LOG), "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{utc} {rec_rel} sha256 {h}; {a.out} content sha256 {doc['sha256']}; "
                 f"{hl['record']['n_files']} files hashed; approved_by {a.approved_by}\n")
    waived = [e["id"] for e in hl["record"]["entries"] if e["status"] == "waived"]
    print(f"[freeze] {a.out} sha256 {doc['sha256']}; {len(prm)} parameters; hash list {hl['record']['n_files']} files "
          f"(spec revision {hl['record']['spec'].get('revision')}; waived {waived or 'none'}); {rec_rel} sha256 "
          f"{h[:16]}...; forecasts csv {prm['forecast file SHA-256 (csv, meta)']['value']['csv'][:16]}...")
    return 0


# ----------------------------------------------------------------------------- post-freeze amendments
AMEND_LOG = "freeze/amendments.jsonl"                  # append-only, hash-chained; FREEZE.json is never rewritten
AMEND_CODE_SNAPSHOT = "freeze/code_record_at_freeze.json"
ABSENT = "absent"


def _np(p: str) -> str:
    return str(p).replace("\\", "/").lower()


def _freeze_doc(root: str, path: str = "FREEZE.json") -> dict:
    doc = json.load(open(os.path.join(root, path), encoding="utf-8-sig"))
    body = {k: v for k, v in doc.items() if k not in ("created", "sha256")}
    if hashlib.sha256(canonical(body).encode()).hexdigest() != doc.get("sha256"):
        raise SystemExit(f"{path}: content hash does not verify")
    return doc


def amend_records(root: str, freeze_sha: str | None) -> tuple[list[dict], list[str]]:
    """The append-only amendment log: one JSON object per line; prev_sha256 = sha256 of the previous line (of the first
    line: the FREEZE.json content sha256), so an edited, reordered or removed line breaks the chain."""
    p = os.path.join(root, *AMEND_LOG.split("/"))
    if not os.path.isfile(p):
        return [], []
    recs, probs, prev = [], [], freeze_sha
    for i, line in enumerate(open(p, encoding="utf-8").read().split("\n")):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            probs.append(f"amendment log line {i + 1}: not JSON")
            break
        if r.get("prev_sha256") != prev:
            probs.append(f"amendment log line {i + 1} ({r.get('id')}): hash chain broken (edited, reordered or removed line)")
        miss = [k for k in ("id", "utc", "path", "old_sha256", "new_sha256", "reason", "approved_by", "decision",
                            "decision_sha256") if not r.get(k)]
        if miss:
            probs.append(f"amendment log line {i + 1}: fields missing {miss}")
        prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
        recs.append(r)
    return recs, probs


def amend_chain(recs: list[dict], path: str, frozen: str, current: str) -> list[dict] | None:
    """The amendment records of `path` that lead from its frozen sha256 to the current one (None if they do not)."""
    chain, h = [], frozen
    for r in recs:
        if _np(r["path"]) == _np(path) and r["old_sha256"] == h:
            chain.append(r)
            h = r["new_sha256"]
    return chain if chain and h == current else None


def _code_record_amended(root: str, ref: dict, recs: list[dict]) -> tuple[list[str] | None, str | None]:
    """A code record changed by a post-freeze -Reinstall: every file that differs from the frozen record (kept as
    freeze/code_record_at_freeze.json by `amend`, checked against FREEZE.json) must carry an amendment chain."""
    snap = os.path.join(root, *AMEND_CODE_SNAPSHOT.split("/"))
    if not os.path.isfile(snap):
        return None, "no snapshot of the frozen code record (written by chain6b_util.py amend)"
    if sha256_file(snap) != ref.get("sha256"):
        return None, f"{AMEND_CODE_SNAPSHOT} is not the frozen code record"
    old = {_np(k): v for k, v in (json.load(open(snap, encoding="utf-8-sig")).get("files") or {}).items()}
    new = {_np(k): v for k, v in (json.load(open(os.path.join(root, ref["path"]), encoding="utf-8-sig")).get("files")
                                  or {}).items()}
    diff = [p for p in sorted(set(old) | set(new)) if old.get(p) != new.get(p)]
    unexpl = [p for p in diff if not amend_chain(recs, p, old.get(p, ABSENT), new.get(p, ABSENT))]
    if unexpl:
        return None, f"{len(unexpl)} file(s) differ from the frozen code record without an amendment, e.g. {unexpl[:3]}"
    return diff, None


def amend(root: str, a) -> int:
    """Record a post-freeze amendment (append-only, hash-chained) BEFORE the -Reinstall that installs it: every file of
    the code record that differs from the frozen one after staging (phase6\\ sources) and every INSTALL target that
    -Reinstall will rewrite, with old/new sha256, reason, approval, the logged decision (and its sha256), UTC time."""
    doc = _freeze_doc(root)
    if not (a.reason and a.approved_by):
        raise SystemExit("amend: --reason and --approved-by are required")
    dec = os.path.join(root, a.decision)
    if not os.path.isfile(dec):
        raise SystemExit(f"amend: the logged decision {a.decision} does not exist")
    recs, probs = amend_records(root, doc["sha256"])
    if probs:
        raise SystemExit("amend: the amendment log does not verify: " + "; ".join(probs))
    if any(r.get("amendment") == a.id or r.get("id") == a.id for r in recs):     # records are <id>.1, <id>.2, ...
        raise SystemExit(f"amend: id {a.id} is already recorded (the log is append-only)")
    cref = doc.get("code_record") or {}
    snap = os.path.join(root, *AMEND_CODE_SNAPSHOT.split("/"))
    if os.path.isfile(snap):
        if sha256_file(snap) != cref.get("sha256"):
            raise SystemExit(f"amend: {AMEND_CODE_SNAPSHOT} is not the frozen code record")
        src_rec = snap
    else:
        src_rec = os.path.join(root, cref.get("path", CODE_RECORD))
        if sha256_file(src_rec) != cref.get("sha256"):
            raise SystemExit("amend: the code record is no longer the frozen one and no snapshot exists "
                             "(record the amendment after staging and BEFORE chain6b.ps1 -Reinstall)")
    frozen = {_np(k): (k, v) for k, v in (json.load(open(src_rec, encoding="utf-8-sig")).get("files") or {}).items()}
    cur = {_np(k): (k, v) for k, v in code_set(root).items()}
    changes: dict[str, tuple] = {}
    for n in sorted(set(frozen) | set(cur)):
        o, w = frozen.get(n, (None, ABSENT))[1], cur.get(n, (None, ABSENT))[1]
        if o != w:
            changes[n] = ((cur.get(n) or frozen.get(n))[0], o, w, "staged")
    for src, dst, kind in INSTALL:
        sp, dp = os.path.join(root, src), os.path.join(root, dst)
        if not os.path.isfile(sp) or (kind == "seed" and os.path.exists(dp)):
            continue
        s_new, s_cur = sha256_file(sp), (sha256_file(dp) if os.path.isfile(dp) else ABSENT)
        if s_new != s_cur:
            n = _np(dst)
            changes[n] = (dst, frozen.get(n, (dst, s_cur))[1], s_new, f"installed by -Reinstall from {src}")
    rows = []
    for n, (path, old, new, why) in sorted(changes.items()):
        prior = [r for r in recs if _np(r["path"]) == n]
        start = prior[-1]["new_sha256"] if prior else old
        if prior and not amend_chain(recs, path, old, start):
            raise SystemExit(f"amend: the earlier amendments of {path} do not start at its frozen sha256")
        if new != start:
            rows.append({"path": path, "old_sha256": start, "new_sha256": new, "change": why})
    if not rows:
        print("[amend] nothing differs from the frozen code record or the amended state: nothing recorded")
        return 0
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dsha = sha256_file(dec)
    for r in rows:
        print(f"[amend] {a.id}: {r['path']}  {r['old_sha256'][:12]} -> {r['new_sha256'][:12]}  ({r['change']})")
    if a.dry_run:
        print(f"[amend] dry run: {len(rows)} record(s) not written")
        return 0
    if not os.path.isfile(snap):
        os.makedirs(os.path.dirname(snap), exist_ok=True)
        shutil.copyfile(src_rec, snap + ".tmp")
        os.replace(snap + ".tmp", snap)
        with open(snap + ".sha256", "w", encoding="ascii", newline="\n") as fh:
            fh.write(f"{sha256_file(snap)}  {os.path.basename(snap)}\n")
    lp = os.path.join(root, *AMEND_LOG.split("/"))
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    prev = hashlib.sha256(open(lp, encoding="utf-8").read().rstrip("\n").split("\n")[-1].encode("utf-8")).hexdigest() \
        if recs else doc["sha256"]
    with open(lp, "a", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows):
            rec = {"id": f"{a.id}.{i + 1}", "amendment": a.id, "utc": utc, **r, "reason": a.reason, "approved_by": a.approved_by,
                   "decision": a.decision.replace("\\", "/"), "decision_sha256": dsha, "freeze_sha256": doc["sha256"],
                   "prev_sha256": prev}
            line = json.dumps(rec, sort_keys=True)
            fh.write(line + "\n")
            prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
    print(f"[amend] {a.id}: {len(rows)} record(s) appended to {AMEND_LOG}; snapshot {AMEND_CODE_SNAPSHOT}")
    return 0


def retry_prep(root: str, queue: str) -> int:
    """chain6b.ps1 -RetryFailed (E2: an infrastructure failure is re-run with the same seeds): before run_queue
    --retry-failed, every failed train job of the queue whose attempt root still exists is moved aside to
    <root>_failed<epoch> (never deleted), so the job trains again from scratch instead of reusing the finished
    attempt (run_queue keeps a root that holds model_final.pt). The queue state is not edited."""
    q = json.load(open(_abs(root, queue), encoding="utf-8-sig"))
    sp = _abs(root, q.get("state", "results/queue_state.json"))
    st = json.load(open(sp, encoding="utf-8")) if os.path.isfile(sp) else {}
    kinds = {j["id"]: j.get("kind") for j in q.get("jobs", [])}
    moved = []
    for jid, rec in st.items():
        if not isinstance(rec, dict) or rec.get("status") != "failed" or kinds.get(jid) != "train":
            continue
        r = rec.get("root")
        r = _abs(root, r) if r else None
        if r and os.path.isdir(r):
            dst = f"{r}_failed{int(time.time())}"
            os.replace(r, dst)
            moved.append((jid, r, dst))
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    with open(os.path.join(root, "logs", "retry_prep.log"), "a", encoding="utf-8", newline="\n") as fh:
        for jid, r, dst in moved:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {jid}: {r} -> {dst}\n")
    failed = [j for j, v in st.items() if isinstance(v, dict) and v.get("status") == "failed"]
    print(f"[retry-prep] failed jobs {failed or 'none'}; attempt roots moved aside: "
          + ("; ".join(f"{j}: {os.path.basename(d)}" for j, _, d in moved) or "none"))
    return 0


def freeze_verify(root: str, path: str, print_queue: bool = False) -> int:
    fp = os.path.join(root, path)
    if not os.path.exists(fp):
        print(f"[freeze-verify] {path} missing")
        return 23
    doc = json.load(open(fp, encoding="utf-8-sig"))
    bad = []
    recs, aprobs = amend_records(root, doc.get("sha256"))
    bad += aprobs
    amended: dict[str, list[dict]] = {}
    body = {k: v for k, v in doc.items() if k not in ("created", "sha256")}
    if hashlib.sha256(canonical(body).encode()).hexdigest() != doc.get("sha256"):
        bad.append("FREEZE.json content hash")
    fc = doc.get("forecasts") or {}
    refs = [("prereg", doc.get("prereg")), ("plan", doc.get("plan")), ("queue", doc.get("queue")),
            ("forecasts csv", fc.get("csv")), ("forecasts meta", fc.get("meta")), ("code_record", doc.get("code_record"))]
    refs += [(f"record {k}", v) for k, v in (doc.get("records") or {}).items()]
    for name, ref in refs:
        if ref is None:
            bad.append(f"{name}: not in the freeze record")
            continue
        p = os.path.join(root, ref["path"])
        if not os.path.isfile(p):
            bad.append(f"{name}: {ref['path']} missing")
        elif sha256_file(p) != ref["sha256"]:
            diff, why = _code_record_amended(root, ref, recs) if name == "code_record" else (None, None)
            if diff is not None:
                amended[f"code record ({ref['path']})"] = [{"files": diff}]
            else:
                bad.append(f"{name}: {ref['path']} changed since the freeze" + (f" ({why})" if why else ""))
    hl = doc.get("hash_list")
    if not hl:
        bad.append("hash list: not in the freeze record (writer older than spec revision 3)")
    else:
        for e in hl.get("entries") or []:
            for f in e.get("files") or []:
                p = _abs(root, f["path"])
                if not os.path.isfile(p):
                    bad.append(f"hash list {e['id']}: {f['path']} missing")
                elif sha256_file(p) != f["sha256"]:
                    ch = amend_chain(recs, f["path"], f["sha256"], sha256_file(p))
                    if ch:
                        amended[f["path"]] = ch
                    else:
                        bad.append(f"hash list {e['id']}: {f['path']} changed since the freeze")
        if doc.get("hashed_utc"):                              # spec check 4: one zone (UTC epoch seconds)
            bad += post_freeze_start_check(root, (doc.get("queue") or {}).get("path"),
                                           _state_of_plan(root, (doc.get("plan") or {}).get("path")), doc["hashed_utc"])
        c2 = ((hl.get("checks") or {}).get("power_sim_v1") or {})
        if c2.get("sha256") != POWER_SIM_V1_SHA256:
            bad.append("check 2: power_sim_v1.py sha256 in the record is not d9c21f3b...")
    rec_rel = record_path(path)
    rec_p = os.path.join(root, *rec_rel.split("/"))
    if not os.path.isfile(rec_p):
        bad.append(f"{rec_rel} missing")
    else:
        h = sha256_file(rec_p)
        if h != sha256_file(fp):
            bad.append(f"{rec_rel} differs from {path}")
        side = rec_p + ".sha256"
        if not os.path.isfile(side) or open(side, encoding="ascii").read().split()[:1] != [h]:
            bad.append(f"{rec_rel}.sha256 does not match the record")
    if doc.get("code_record"):
        rc = codecheck(root)
        if rc != 0:
            bad.append("code drift since the install record")
    if doc.get("plan"):
        plan = json.load(open(os.path.join(root, doc["plan"]["path"]), encoding="utf-8"))
        if plan.get("sha256") != doc.get("plan_content_sha256"):
            bad.append("plan content hash")
    prm = doc.get("parameters") or {}
    if set(prm) != set(FREEZE_PARAMETERS) or doc.get("parameter_order") != FREEZE_PARAMETERS:
        bad.append("parameters: not the 30 names of g1g2_spec.json freeze.parameters")
    elif not bad:                                                  # spec check 3: re-derive every parameter from its source
        env = (prm.get("envelope extension (decision P-6; 0 if none)") or {}).get("value") or {}
        try:
            again = freeze_parameters(root, doc.get("parameter_refs") or {}, env.get("gpu_h", 0.0), env.get("decision"))
        except SystemExit as e:
            bad.append(f"parameters: re-derivation failed ({e})")
        else:
            diff = [k for k in FREEZE_PARAMETERS if canonical(again[k]) != canonical(prm[k])]
            if diff:
                bad.append("parameter mismatch vs. the source records: " + "; ".join(diff))
    if bad:
        for b in bad:
            print("[freeze-verify] FAIL", b)
        return 23
    if print_queue:
        print(doc["queue"]["path"])
    else:
        for k, ch in amended.items():
            if "files" in ch[0]:
                print(f"[freeze-verify] amended after the freeze: {k}: {len(ch[0]['files'])} file(s), each with an amendment record")
            else:
                print(f"[freeze-verify] amended after the freeze: {k} {ch[0]['old_sha256'][:12]} -> {ch[-1]['new_sha256'][:12]} ("
                      + ", ".join(f"{r['id']} {r['utc']} approved by {r['approved_by']}; {r['decision']}" for r in ch) + ")")
        print(f"[freeze-verify] {path} verifies (sha256 {doc['sha256']}; {len(prm)} parameters re-derived; "
              f"{(hl or {}).get('n_files')} hash-list files re-hashed"
              + (f"; {len(amended)} amended item(s), {len(recs)} record(s) in {AMEND_LOG}" if amended or recs else "")
              + f"); queue {doc['queue']['path']}")
    return 0


# ----------------------------------------------------------------------------- markers
def mark(root: str, step: str, info: str | None) -> int:
    rec_p = os.path.join(root, CODE_RECORD)
    rec_sha = json.load(open(rec_p, encoding="utf-8")).get("sha256") if os.path.exists(rec_p) else None
    write_json(os.path.join(root, MARKERS, f"{step}.done"), {"step": step, "time": now(), "code_record_sha256": rec_sha,
                                                            "info": info})
    return 0


def is_done(root: str, step: str) -> int:
    return 0 if os.path.exists(os.path.join(root, MARKERS, f"{step}.done")) else 1


def status(root: str) -> int:
    d = os.path.join(root, MARKERS)
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".done"):
            m = json.load(open(os.path.join(d, f), encoding="utf-8"))
            print(f"{m['step']:<16} {m['time']}  {m.get('info') or ''}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="chain6b helpers (torch-free)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("install", "codecheck", "check-stage", "freeze-write", "freeze-check", "freeze-verify", "env-snapshot",
                 "mark", "done", "status", "amend", "retry-prep"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
        if name == "install":
            s.add_argument("--reinstall", action="store_true")
            s.add_argument("--dry-run", action="store_true")
        if name == "check-stage":
            s.add_argument("stage", choices=["B", "C"])
        if name in ("freeze-write", "freeze-check"):
            for k in ("plan", "queue", "prereg"):
                s.add_argument(f"--{k}", required=(name == "freeze-write"), default=None)
            for k in ("g0", "g1", "g2"):
                s.add_argument(f"--{k}", default=None)
            s.add_argument("--forecasts-csv", default=os.path.join("freeze", "forecasts_v13.csv"))
            s.add_argument("--forecasts-meta", default=os.path.join("freeze", "forecasts_v13_meta.json"))
            s.add_argument("--nestor-books", default=os.path.join("results", "nestor_books_feasibility.json"))
            s.add_argument("--dose", default=os.path.join("results", "dose", "dose_record.json"))
            s.add_argument("--b3", default=os.path.join("g0", "gate_record_b3.json"), help="used if present, else pending")
            s.add_argument("--decisions", default="decisions_v13_points.md", help="comma-separated decision records")
            s.add_argument("--envelope-extension", type=float, default=0.0, help="GPU-h (decision P-6; 0 if none)")
            s.add_argument("--envelope-decision", default=None, help="where the P-6 decision is logged")
            s.add_argument("--approved-by", default=None)
            s.add_argument("--notes", default=None)
            s.add_argument("--out", default="FREEZE.json")
            s.add_argument("--force", action="store_true")
            s.add_argument("--waive", action="append", default=[], metavar="ID=DECISION_FILE",
                           help="hash-list entry (id like 11.1, or its text, or g0freeze:<path>) removed/accepted by a "
                                "logged decision (the file is hashed into the record)")
            s.add_argument("--mock-waive-missing", action="store_true", help="mock runs only (--approved-by MOCK)")
        if name == "freeze-verify":
            s.add_argument("--freeze", default="FREEZE.json")
            s.add_argument("--print-queue", action="store_true")
        if name == "amend":
            s.add_argument("--id", required=True, help="amendment id, e.g. A-1")
            s.add_argument("--reason", required=True)
            s.add_argument("--approved-by", required=True)
            s.add_argument("--decision", required=True, help="the logged decision (root-relative path)")
            s.add_argument("--dry-run", action="store_true")
        if name == "retry-prep":
            s.add_argument("--queue", required=True)
        if name in ("mark", "done"):
            s.add_argument("--step", required=True)
        if name == "mark":
            s.add_argument("--info", default=None)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.cmd == "install":
        return install(root, reinstall=a.reinstall, dry=a.dry_run)
    if a.cmd == "codecheck":
        return codecheck(root)
    if a.cmd == "check-stage":
        return check_stage(root, a.stage)
    if a.cmd == "freeze-write":
        return freeze_write(root, a)
    if a.cmd == "freeze-check":
        return freeze_check(root, a)
    if a.cmd == "env-snapshot":
        return env_snapshot(root)
    if a.cmd == "freeze-verify":
        return freeze_verify(root, a.freeze, a.print_queue)
    if a.cmd == "amend":
        return amend(root, a)
    if a.cmd == "retry-prep":
        return retry_prep(root, a.queue)
    if a.cmd == "mark":
        return mark(root, a.step, a.info)
    if a.cmd == "done":
        return is_done(root, a.step)
    return status(root)


if __name__ == "__main__":
    sys.exit(main())
