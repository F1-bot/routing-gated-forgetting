"""a1_wd_only_checks.py -- read-only check of the [wd_only] lines of the three C3 FULL jobs (positions 131, 133, 140).

Item (prereg_v13/deviations_post_freeze.md, entry D-1, addendum 1, item 6): "Three C3 FULL jobs run WD-only without a
subset manifest: c3_full_moe_b1_books_s101, c3_full_nestor_books_s101 and c3_full_nestor_math_s101 (two on NESTOR). On
this path the fingerprint check is skipped and every parameter decays. It has not yet been checked on real files; the
train-log comparison still applies." The item stays open until these [wd_only] lines are recorded (entry A-2, "Open").

This file is separate from results/a2_acceptance.json (the three A-2 evaluations, read by the paper fill), and the
full analysis does not depend on it (run_full_analysis.ps1 reports it as a warning only).

Reads only logs and queue files, never an evaluation value:
  logs/queue_confirm/<job>_eval_wd.log           the WD-only evaluation of the train job (run_queue: <id>_eval_<name>)
  confirm/queue_confirm.json                     the frozen queue job: id at the position, steps, warm-up, interim stop
  results/queue_state_confirm.json               the job's final state and its root
  <root>/trainlog/train_steps.jsonl              the training log the evaluation compared its learning rates with
Writes only results/a1_wd_only_checks.json = {"pass": {job: true|false}, "checks": {...}, "rule": "...", "utc": "..."}.

  python confirm/post_queue/a1_wd_only_checks.py [--root R]
  python confirm/post_queue/a1_wd_only_checks.py --logs DIR --state FILE --out FILE    (tests: other inputs and output)

Safety as a2_acceptance.py: the default output only from the default logs and state, only when all three jobs have a
final state (done, failed or skipped), never over an existing file, never to a path listed in FREEZE.json.
Exit 0 written; 2 refused (usage, inputs, output exists or is listed in FREEZE.json); 3 refused (a job has not ended).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_acceptance as A2  # noqa: E402  (line format, hashing and path helpers; that file is unchanged)

JOBS = ((131, "c3_full_nestor_books_s101"), (133, "c3_full_nestor_math_s101"), (140, "c3_full_moe_b1_books_s101"))
SOURCE = "frozen queue job (queue_confirm.json)"
EVAL_NAME = "wd"
RULE = ("prereg_v13/deviations_post_freeze.md, entry D-1, addendum 1, item 6: \"Three C3 FULL jobs run WD-only without a "
        "subset manifest: c3_full_moe_b1_books_s101, c3_full_nestor_books_s101 and c3_full_nestor_math_s101 (two on "
        "NESTOR). On this path the fingerprint check is skipped and every parameter decays. It has not yet been checked on "
        "real files; the train-log comparison still applies.\" Entry A-2, Rules item 4: \"Device: if a [wd_only] line of "
        "positions 131, 133, 137, 140, 142 or 147 reports cpu, the extra queue hours are recorded here as a note.\" Applied "
        "as: a job passes only if it ended 'done' in results/queue_state_confirm.json and "
        "logs/queue_confirm/<job>_eval_wd.log holds exactly one [wd_only] line for the job's experiment whose max_steps, "
        "warm-up and executed equal the frozen queue job's steps, warm-up and executed steps (steps, or the interim "
        "stop_after when the job sets one), whose sources include \"frozen queue job (queue_confirm.json)\" and whose "
        "train-log LRs read \"equal at all <executed> steps\"; the training log of the job root recorded in the queue state "
        "must hold steps 1 to <executed>, each with its learning rates. The fingerprint and the device are recorded; a "
        "fingerprint other than 'not checked' and the device cpu are notes, not failures. Only logs and queue files are "
        "read, no evaluation value.")


def train_log_check(root: str, run_root: str, executed: int):
    tl = os.path.join(run_root, "trainlog", "train_steps.jsonl")
    info = {"path": A2.rel(root, tl), "exists": os.path.isfile(tl)}
    if not info["exists"]:
        return info, ["training log trainlog/train_steps.jsonl missing in the job root"], None
    recs = [json.loads(x) for x in open(tl, encoding="utf-8").read().splitlines() if x.strip()]
    steps = [r.get("step") for r in recs]
    with_lrs = [r for r in recs if isinstance(r.get("lrs"), list) and r["lrs"]]
    max_lr = max((abs(float(v)) for r in with_lrs for v in r["lrs"]), default=None)
    info.update(sha256=A2.sha256_file(tl), records=len(recs), first_step=steps[0] if steps else None,
                last_step=steps[-1] if steps else None, records_with_lrs=len(with_lrs), max_lr=max_lr)
    why = []
    if steps != list(range(1, executed + 1)) or len(with_lrs) != executed:
        why.append(f"training log holds {len(recs)} records (steps {info['first_step']}..{info['last_step']}), "
                   f"{len(with_lrs)} with learning rates (expected steps 1..{executed}, each with learning rates)")
    return info, why, max_lr


def check_job(root: str, pos: int, jid: str, queue: dict, state: dict, logs: str) -> tuple[bool, dict]:
    why, notes = [], []
    c = {"position": pos}
    jobs = queue.get("jobs", [])
    qj = jobs[pos - 1] if 0 < pos <= len(jobs) else None
    if not qj or qj.get("id") != jid:
        why.append(f"confirm/queue_confirm.json position {pos} is {qj.get('id') if qj else None!r}, not {jid}")
        c.update(reasons=why, notes=notes)
        return False, c
    steps = qj.get("steps")
    warm = qj.get("warmup", 10)                                   # run_queue: str(j.get("warmup", 10))
    stop = (qj.get("interim") or {}).get("stop_after")
    executed = int(stop) if stop else (int(steps) if steps is not None else None)
    c["frozen_queue"] = {"path": "confirm/queue_confirm.json", "sha256": A2.sha256_file(os.path.join(root, "confirm", "queue_confirm.json")),
                         "experiment": qj.get("experiment"), "steps": steps, "warmup": warm, "stop_after": stop,
                         "executed": executed}
    evs = [e for e in qj.get("evals", []) if e.get("name") == EVAL_NAME]
    if not (evs and any(v.split(":")[0] == "wdonly" for v in evs[0].get("variants", "").split(","))):
        why.append(f"the frozen queue job has no '{EVAL_NAME}' evaluation with the wdonly variant")
    rec = state.get(jid) or {}
    c["job_status"] = rec.get("status")
    c["job_root"] = A2.rel(root, rec["root"]) if rec.get("root") else None
    if rec.get("status") != "done":
        why.append(f"job state {rec.get('status')!r} in the confirm queue state (no value exists for a job that did not "
                   f"end 'done')")
    lp = os.path.join(logs, f"{jid}_eval_{EVAL_NAME}.log")
    c["eval_log"] = {"path": A2.rel(root, lp), "exists": os.path.isfile(lp)}
    line = None
    if not c["eval_log"]["exists"]:
        why.append("WD-only evaluation log missing (no [wd_only] line)")
        c["wd_only_lines"] = 0
    else:
        c["eval_log"].update(sha256=A2.sha256_file(lp), bytes=os.path.getsize(lp))
        lines = [x.rstrip("\r\n") for x in open(lp, encoding="utf-8", errors="replace") if "[wd_only]" in x]
        c["wd_only_lines"] = len(lines)
        if not lines:
            why.append("the WD-only evaluation log has no [wd_only] line")
        elif len(lines) > 1:
            why.append(f"the WD-only evaluation log has {len(lines)} [wd_only] lines (expected one)")
        else:
            line = lines[0]
    c["wd_only_line"] = line
    m = A2.LINE.search(line) if line else None
    if line and not m:
        why.append("the [wd_only] line is not in the A-1/A-2 format (weight_controls.py lines 523-525)")
    g = m.groupdict() if m else {}
    c["steps_found"] = ({"max_steps": A2.as_int(g["max_steps"]), "warmup": A2.as_int(g["warmup"]),
                         "executed": A2.as_int(g["executed"])} if m else None)
    c["source"] = g.get("sources")
    c["fingerprint"] = g.get("fingerprint")
    c["train_log_lrs"] = g.get("train_log_lrs")
    c["device"] = g.get("device")
    c["seconds"] = float(g["seconds"]) if m and g["seconds"].replace(".", "", 1).isdigit() else None
    if m:
        if qj.get("experiment") and g["experiment"] != qj.get("experiment"):
            why.append(f"the line names experiment {g['experiment']}, the frozen queue job {qj.get('experiment')}")
        want = {"max_steps": steps, "warmup": warm, "executed": executed}
        if c["steps_found"] != want:
            why.append(f"the line reports {g['max_steps']}/{g['warmup']}/{g['executed']}, the frozen queue job "
                       f"{steps}/{warm}/{executed} (max_steps/warm-up/executed)")
        if SOURCE not in g["sources"].split(", "):
            why.append(f"the line's sources {g['sources']!r} do not include {SOURCE!r}")
        if executed is not None and g["train_log_lrs"] != f"equal at all {executed} steps":
            why.append(f"the line reads train-log LRs {g['train_log_lrs']!r} (expected 'equal at all {executed} steps')")
        if g["fingerprint"] != "not checked":
            notes.append(f"fingerprint {g['fingerprint']!r}: the FULL path has no subset manifest, 'not checked' expected")
        if g["device"].strip().lower() == "cpu":
            notes.append("device cpu: the extra queue hours are to be recorded as a note (entry A-2, Rules item 4)")
    max_lr = None
    if rec.get("root") and executed:
        c["train_log"], w, max_lr = train_log_check(root, rec["root"], executed)
        why += w
    elif rec.get("status") == "done":
        why.append("the queue state records no job root (training log unknown)")
    c["lr_max_abs_diff"] = {
        "value": None,
        "bound": A2.REL_TOL * max_lr / (1.0 - A2.REL_TOL) if max_lr is not None else None,
        "basis": ("not logged: weight_controls.wd_only_state (lines 512-521) compares the decay-only learning rates "
                  "with the training log in memory and raises unless |a - b| <= 1e-9 max(|a|, |b|) at every step"),
    }
    c["reasons"], c["notes"] = why, notes
    return not why, c


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="[wd_only] checks of the C3 FULL jobs at positions 131, 133 and 140")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(here)))
    ap.add_argument("--logs", default=None, help="default <root>/logs/queue_confirm")
    ap.add_argument("--state", default=None, help="default <root>/results/queue_state_confirm.json")
    ap.add_argument("--out", default=None, help="default <root>/results/a1_wd_only_checks.json")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    d_logs = os.path.join(root, "logs", "queue_confirm")
    d_state = os.path.join(root, "results", "queue_state_confirm.json")
    d_out = os.path.join(root, "results", "a1_wd_only_checks.json")
    logs, state_p, out = (os.path.abspath(x) if x else d for x, d in ((a.logs, d_logs), (a.state, d_state), (a.out, d_out)))
    real = A2.norm(out) == A2.norm(d_out)

    def refuse(code: int, msg: str) -> int:
        print(f"[a1_wd_only_checks] refused: {msg}; nothing written")
        return code
    if real and (A2.norm(logs) != A2.norm(d_logs) or A2.norm(state_p) != A2.norm(d_state)):
        return refuse(2, "the default output takes only the default logs and state")
    if os.path.exists(out):
        return refuse(2, f"{A2.rel(root, out)} exists (never overwritten; move it aside first)")
    if A2.norm(out) in A2.freeze_paths(root):
        return refuse(2, f"{A2.rel(root, out)} is listed in FREEZE.json")
    for p in (os.path.join(root, "confirm", "queue_confirm.json"), state_p):
        if not os.path.isfile(p):
            return refuse(2, f"{p} missing")
    queue = A2.load_json(os.path.join(root, "confirm", "queue_confirm.json"))
    state = A2.load_json(state_p)
    if real:
        open_ = [f"{jid} ({(state.get(jid) or {}).get('status', 'pending')})" for _, jid in JOBS
                 if (state.get(jid) or {}).get("status") not in A2.FINAL]
        if open_:
            return refuse(3, "not ended yet: " + ", ".join(open_))
    passed, checks = {}, {}
    for pos, jid in JOBS:
        passed[jid], checks[jid] = check_job(root, pos, jid, queue, state, logs)
        r, n = checks[jid]["reasons"], checks[jid]["notes"]
        print(f"[a1_wd_only_checks] {pos} {jid}: {'pass' if passed[jid] else 'NOT passed'}"
              + (f" ({'; '.join(r)})" if r else "") + (f" [notes: {'; '.join(n)}]" if n else ""))
    doc = {"pass": passed, "checks": checks, "rule": RULE,
           "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"[a1_wd_only_checks] -> {A2.rel(root, out)} sha256 {A2.sha256_file(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
