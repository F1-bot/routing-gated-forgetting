"""a4_acceptance.py -- acceptance of the two C3 freeze-mode runs re-run under amendment A-4 (read-only).

Rule (confirm/rerun_a4/amendment_A4_post_freeze.md, section "Acceptance"; the named lists are in
confirm/rerun_a4/expected_unchanged.json): a re-run of c3_freeze_moe_b1_german_s103 (position 154) or
c3_freeze_nestor_german_s103 (position 155) is accepted only if its bitwise audit has frozen_mismatch [],
buffer_mismatch [], no optimizer-state violation and trainable_unchanged exactly equal to the job's named list, and if it
is the same job as the first attempt. Otherwise the run's values are withheld (not reported) and the case is recorded as
a deviation.

Reads only:
  results/queue_state_confirm.json                     the re-run's final state (status, audit_ok, root, audit, start)
  freeze/amendments.jsonl                              the A-4 record (the re-run must start after it)
  confirm/rerun_a4/expected_unchanged.json             the named lists, the first attempt's sha256 values and counts
  confirm/rerun_a4/evidence/first_attempt/<job>/       the first attempt's audit.json, subset_manifest.json, run_env.json
  <root>/audit.json, subset_manifest.json, run_env.json    the re-run's audit record, manifest and environment
never an evaluation, interim, train-log or CE file. Writes only results/a4_acceptance.json, with exactly the keys
accepted, checks, rule and utc.

  python confirm/rerun_a4/a4_acceptance.py [--root R]
  python confirm/rerun_a4/a4_acceptance.py --state FILE --expected FILE --evidence DIR --amendments FILE --out FILE
                                                                                (tests: other inputs and output)

The default output is written only from the default inputs, only when both jobs have a final state (done, failed or
skipped), never over an existing file and never to a path listed in FREEZE.json.
Exit 0 written; 2 refused (usage, inputs, a named list pending or not a list, a first-attempt copy missing or not the
recorded one, output exists or is listed in FREEZE.json); 3 refused (a job has not ended).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

JOBS = ((154, "c3_freeze_moe_b1_german_s103"), (155, "c3_freeze_nestor_german_s103"))
FINAL = ("done", "failed", "skipped")
AMENDMENT = "A-4"
COPIES = (("audit.json", "audit_sha256"), ("subset_manifest.json", "subset_manifest_sha256"),
          ("run_env.json", "run_env_sha256"))
RULE = ("confirm/rerun_a4/amendment_A4_post_freeze.md, section Acceptance: \"A re-run of position 154 or 155 is accepted "
        "only if its bitwise audit has frozen_mismatch [], buffer_mismatch [], no optimizer-state violation and "
        "trainable_unchanged exactly equal to the job's named list (confirm/rerun_a4/expected_unchanged.json). Otherwise "
        "its values are withheld and the case is recorded as a deviation.\" Applied as: the job ended 'done' with "
        "audit_ok true in results/queue_state_confirm.json, its run started after the A-4 record in "
        "freeze/amendments.jsonl and its final checkpoint exists; <root>/audit.json (the state's audit) has "
        "frozen_mismatch [], buffer_mismatch [], optimizer_state.violations [], allow_unchanged true, ok true and "
        "trainable_unchanged equal (as a list) to the named list; it has the first attempt's counts (frozen tensors, "
        "frozen rows, buffers, optimizer state checked and rows) and the same set of trainable tensors; "
        "subset_manifest.json and run_env.json equal (as JSON) the first attempt's (same spec, recipe, data and seed). "
        "Only state, amendment, audit, manifest and environment files are read, no evaluation, interim, train-log or "
        "CE file.")


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


def rel(root: str, p: str) -> str:
    try:
        r = os.path.relpath(os.path.abspath(p), root)
    except ValueError:                       # another drive
        return os.path.abspath(p).replace("\\", "/")
    return (os.path.abspath(p) if r.startswith("..") else r).replace("\\", "/")


def load_json(p: str):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def freeze_paths(root: str) -> set:
    """Every "path" value anywhere in FREEZE.json (hash list, records, plan, queue ...), normalised."""
    fp = os.path.join(root, "FREEZE.json")
    out = set()
    if not os.path.isfile(fp):
        return out

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "path" and isinstance(v, str):
                    out.add(norm(os.path.join(root, v)))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(load_json(fp))
    return out


def utc_epoch(s):
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def amendment_time(path: str):
    """UTC time (epoch seconds) of the last record of amendment A-4 in freeze/amendments.jsonl, or None."""
    if not os.path.isfile(path):
        return None, None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("amendment") == AMENDMENT:
                    last = r
    if last is None:
        return None, None
    return utc_epoch(last.get("utc")), {k: last.get(k) for k in ("id", "utc", "path", "new_sha256")}


def at_root(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)


def diff_keys(a, b) -> list:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return ["(not both JSON objects)"]
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def check_job(root: str, pos: int, jid: str, spec: dict, state: dict, evidence: str, t_a4, a4_rec) -> tuple[bool, dict]:
    why = []
    exp = spec["expected_trainable_unchanged"]
    fa = spec["first_attempt"]
    s = state.get(jid) or {}
    c = {"position": pos, "expected_trainable_unchanged": exp,
         "state": {"status": s.get("status"), "audit_ok": s.get("audit_ok"), "error": s.get("error"), "hours": s.get("hours"),
                   "start": s.get("start"), "root": rel(root, s["root"]) if s.get("root") else None,
                   "audit": rel(root, s["audit"]) if s.get("audit") else None},
         "amendment_record": a4_rec}
    if s.get("status") != "done":
        why.append(f"status {s.get('status')!r} in the queue state ('done' required)")
    if s.get("audit_ok") is not True:
        why.append(f"audit_ok {s.get('audit_ok')!r} in the queue state (true required)")
    if t_a4 is None:
        why.append(f"no {AMENDMENT} record in freeze/amendments.jsonl")
    elif not isinstance(s.get("start"), (int, float)) or s["start"] <= t_a4:
        why.append(f"the run did not start after the {AMENDMENT} record ({a4_rec.get('utc') if a4_rec else None})")
    if not s.get("final") or not os.path.isfile(at_root(root, s["final"])):
        why.append("the state's final checkpoint is missing")
    if not s.get("root"):
        why.append("the state records no root")
        c["reasons"] = why
        return False, c
    rdir = at_root(root, s["root"])
    if norm(rdir) == norm(os.path.join(root, fa["root"])) and s.get("start") is not None and fa.get("start") == s.get("start"):
        why.append("the state still holds the first attempt")
    ap = os.path.join(rdir, "audit.json")
    if s.get("audit") and norm(at_root(root, s["audit"])) != norm(ap):
        why.append("the state's audit is not <root>/audit.json")
    c["rerun_sha256"] = {f: (sha256_file(os.path.join(rdir, f)) if os.path.isfile(os.path.join(rdir, f)) else None)
                         for f, _ in COPIES}
    c["first_attempt_sha256"] = {f: fa[k] for f, k in COPIES}
    if not c["rerun_sha256"]["audit.json"]:
        why.append("<root>/audit.json missing")
        c["reasons"] = why
        return False, c
    au = load_json(ap)
    fa_au = load_json(os.path.join(evidence, jid, "audit.json"))
    opt = au.get("optimizer_state") or {}
    fa_opt = fa_au.get("optimizer_state") or {}
    c["audit"] = {k: au.get(k) for k in ("frozen_tensors_checked", "frozen_rows_checked", "frozen_mismatch", "trainable_unchanged",
                                         "buffers_checked", "buffer_mismatch", "negative_control", "allow_unchanged", "ok")}
    c["audit"]["optimizer_state"] = opt
    c["audit"]["trainable_tensors"] = len(au.get("trainable_changed_fraction") or {})
    if au.get("frozen_mismatch") != []:
        why.append(f"frozen_mismatch {au.get('frozen_mismatch')!r}")
    if au.get("buffer_mismatch") != []:
        why.append(f"buffer_mismatch {au.get('buffer_mismatch')!r}")
    if opt.get("violations") != []:
        why.append(f"optimizer-state violations {opt.get('violations')!r}")
    tu = au.get("trainable_unchanged")
    if tu != exp:
        extra = sorted(set(tu or []) - set(exp))
        missing = sorted(set(exp) - set(tu or []))
        why.append(f"trainable_unchanged {tu!r} is not the named list (extra {extra}, missing {missing})")
    if au.get("allow_unchanged") is not True:
        why.append(f"allow_unchanged {au.get('allow_unchanged')!r} (the A-4 audit runs with --allow-unchanged)")
    if au.get("ok") is not True:
        why.append(f"ok {au.get('ok')!r}")
    for k in ("frozen_tensors_checked", "frozen_rows_checked", "buffers_checked", "negative_control"):
        if au.get(k) != fa_au.get(k):
            why.append(f"{k} {au.get(k)!r} differs from the first attempt ({fa_au.get(k)!r})")
    for k in ("checked", "state_rows_checked"):
        if opt.get(k) != fa_opt.get(k):
            why.append(f"optimizer_state.{k} {opt.get(k)!r} differs from the first attempt ({fa_opt.get(k)!r})")
    ka, kb = set(au.get("trainable_changed_fraction") or {}), set(fa_au.get("trainable_changed_fraction") or {})
    if ka != kb:
        why.append(f"the set of trainable tensors differs from the first attempt ({len(ka - kb)} added, {len(kb - ka)} missing)")
    for f in ("subset_manifest.json", "run_env.json"):
        p = os.path.join(rdir, f)
        if not os.path.isfile(p):
            why.append(f"<root>/{f} missing")
            continue
        dk = diff_keys(load_json(p), load_json(os.path.join(evidence, jid, f)))
        c[f"{f}_equal_to_first_attempt"] = not dk
        if dk:
            why.append(f"<root>/{f} differs from the first attempt in {dk[:8]}")
    c["reasons"] = why
    return not why, c


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="A-4 acceptance of the re-runs of positions 154 and 155")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(here)))
    ap.add_argument("--state", default=None, help="default <root>/results/queue_state_confirm.json")
    ap.add_argument("--expected", default=None, help="default <root>/confirm/rerun_a4/expected_unchanged.json")
    ap.add_argument("--evidence", default=None, help="default <root>/confirm/rerun_a4/evidence/first_attempt")
    ap.add_argument("--amendments", default=None, help="default <root>/freeze/amendments.jsonl")
    ap.add_argument("--out", default=None, help="default <root>/results/a4_acceptance.json")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    dflt = {"state": os.path.join(root, "results", "queue_state_confirm.json"),
            "expected": os.path.join(root, "confirm", "rerun_a4", "expected_unchanged.json"),
            "evidence": os.path.join(root, "confirm", "rerun_a4", "evidence", "first_attempt"),
            "amendments": os.path.join(root, "freeze", "amendments.jsonl"),
            "out": os.path.join(root, "results", "a4_acceptance.json")}
    p = {k: (os.path.abspath(getattr(a, k)) if getattr(a, k) else v) for k, v in dflt.items()}
    real = norm(p["out"]) == norm(dflt["out"])

    def refuse(code: int, msg: str) -> int:
        print(f"[a4_acceptance] refused: {msg}; nothing written")
        return code
    if real and any(norm(p[k]) != norm(dflt[k]) for k in ("state", "expected", "evidence", "amendments")):
        return refuse(2, "the default output takes only the default inputs")
    if os.path.exists(p["out"]):
        return refuse(2, f"{rel(root, p['out'])} exists (never overwritten; move it aside first)")
    if norm(p["out"]) in freeze_paths(root):
        return refuse(2, f"{rel(root, p['out'])} is listed in FREEZE.json")
    for k in ("state", "expected"):
        if not os.path.isfile(p[k]):
            return refuse(2, f"{p[k]} missing")
    jobs = (load_json(p["expected"]).get("jobs") or {})
    for pos, jid in JOBS:
        spec = jobs.get(jid)
        if not isinstance(spec, dict) or spec.get("position") != pos:
            return refuse(2, f"expected_unchanged.json has no entry for {jid} at position {pos}")
        lst = spec.get("expected_trainable_unchanged")
        if not isinstance(lst, list) or not lst or not all(isinstance(x, str) and x for x in lst):
            return refuse(2, f"the named list of {jid} is {lst!r} (pending or not a list of parameter names)")
        fa = spec.get("first_attempt") or {}
        for f, key in COPIES:
            cp = os.path.join(p["evidence"], jid, f)
            if not os.path.isfile(cp):
                return refuse(2, f"first-attempt copy {rel(root, cp)} missing")
            if sha256_file(cp) != fa.get(key):
                return refuse(2, f"{rel(root, cp)} is not the recorded first attempt ({key})")
        if load_json(os.path.join(p["evidence"], jid, "audit.json")).get("trainable_unchanged") != lst:
            return refuse(2, f"the named list of {jid} is not its first attempt's trainable_unchanged")
    state = load_json(p["state"])
    if real:
        open_ = [f"{jid} ({(state.get(jid) or {}).get('status', 'pending')})" for _, jid in JOBS
                 if (state.get(jid) or {}).get("status") not in FINAL]
        if open_:
            return refuse(3, "not ended yet: " + ", ".join(open_))
    t_a4, a4_rec = amendment_time(p["amendments"])
    accepted, checks = {}, {}
    for pos, jid in JOBS:
        accepted[jid], checks[jid] = check_job(root, pos, jid, jobs[jid], state, p["evidence"], t_a4, a4_rec)
        r = checks[jid]["reasons"]
        print(f"[a4_acceptance] {pos} {jid}: {'accepted' if accepted[jid] else 'NOT accepted (values withheld)'}"
              + (f" ({'; '.join(r)})" if r else ""))
    doc = {"accepted": accepted, "checks": checks, "rule": RULE,
           "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(p["out"]), exist_ok=True)
    tmp = p["out"] + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, p["out"])
    print(f"[a4_acceptance] -> {rel(root, p['out'])} sha256 {sha256_file(p['out'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
