"""a2_acceptance.py -- acceptance of the three weight-decay-only (WD-only) values of amendment A-2 (read-only).

Rule (prereg_v13/deviations_post_freeze.md, entry A-2, "Rules" item 5; MF-A2-2; README_A2 step 9): each
[wd_only] line of positions 137, 142 and 147 must report 96/10/96 from the training queue job and "train-log LRs equal
at all 96 steps". If a line reads "train-log LRs not checked", or is missing, that value is not reported and the case is
recorded as a deviation.

Reads only logs and queue files, never an evaluation value:
  logs/queue_confirm/<job>.log                   the evaluation job's log (the [wd_only] line, weight_controls.py 523-525)
  confirm/queue_confirm.json                     the job at each position and its --manifest argument (the G1 run folder)
  results/queue_state_confirm.json               the job's final state
  g1/queue_g1.json, results/queue_state_g1.json  the training queue job (steps, warm-up) and the run folder it recorded
  <run folder>/trainlog/train_steps.jsonl        the training log the evaluation compared its learning rates with
Writes only results/a2_acceptance.json, with exactly the keys accepted, checks, rule and utc.

  python confirm/post_queue/a2_acceptance.py [--root R]
  python confirm/post_queue/a2_acceptance.py --logs DIR --state FILE --out FILE       (tests: other inputs and output)

The default output is written only from the default logs and state, only when all three jobs have a final state
(done, failed or skipped), never over an existing file and never to a path listed in FREEZE.json.
Exit 0 written; 2 refused (usage, inputs, output exists or is listed in FREEZE.json); 3 refused (a job has not ended).

The learning-rate difference: weight_controls.wd_only_state (lines 512-521) compares the decay-only learning rates
with the training log in memory and raises unless |a - b| <= 1e-9 max(|a|, |b|) for every value; the log records only
the outcome. checks.<job>.lr_max_abs_diff therefore gives value null and the bound implied by that tolerance and the
largest learning rate in the training log.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys

JOBS = ((137, "c3_revert_g1_nestor_full_german_s101"),
        (142, "c3_revert_g1_moe_b1_full_math_s101"),
        (147, "c3_revert_g1_moe_b1_full_german_s101"))
STEPS, WARMUP = 96, 10
SOURCE = "training queue job (g1/queue_g1.json)"
LRS_OK = f"equal at all {STEPS} steps"
TRAIN_QUEUE = "g1/queue_g1.json"
FINAL = ("done", "failed", "skipped")
REL_TOL = 1e-9                     # weight_controls.py lines 517-518
# sha256 of the training queue, its state and the three training logs as recorded at the A-2 install
# (check_a2_inputs.ps1, MF-A2-1); compared for information only (that script is the registered check)
A2_RECORDED = {
    "g1/queue_g1.json": "1d2cae60f3f78ac5c6253c0f9307a3283076f6cdee3a0e157b95c3e2fb974b12",
    "results/queue_state_g1.json": "d25d8bb7ed5a4972b0d15a08e79e8d921473ac485f98eb06993328d467cd7ba1",
    "outputs/g1/g1_nestor_full_german_s101/m2c/trainlog/train_steps.jsonl":
        "fef745fc7ef6f12c79a0abe1babfd605c680d7d593eef83ed0f025bc694d5631",
    "outputs/g1/g1_moe_b1_full_math_s101/m2/trainlog/train_steps.jsonl":
        "a4622097ee1ce08f73dd861eb01b8a6e52e2c1ef9ac9dac3d3a2b084a142421f",
    "outputs/g1/g1_moe_b1_full_german_s101/m2/trainlog/train_steps.jsonl":
        "ac989556d02a8cfe29cefa5bf393842059922748e92adf98ab1fbfa604ae8cf7",
}
LINE = re.compile(r"\[wd_only\] (?P<experiment>\S+): max_steps (?P<max_steps>\S+), warm-up (?P<warmup>\S+), "
                  r"executed (?P<executed>\S+) \(sources: (?P<sources>.*)\); fingerprint (?P<fingerprint>.*); "
                  r"train-log LRs (?P<train_log_lrs>.*); (?P<device>[^;]*), (?P<seconds>\S+) s\s*$")
RULE = ("prereg_v13/deviations_post_freeze.md, entry A-2, Rules item 5: \"Acceptance (MF-A2-2; README step 9): each "
        "[wd_only] line of 137, 142 and 147 must report 96/10/96 from the training queue job and \"train-log LRs equal "
        "at all 96 steps\". If a line reads \"train-log LRs not checked\", or is missing, that value is not reported and "
        "the case is recorded as a deviation.\" Applied as: a value is accepted only if its job ended 'done' in "
        "results/queue_state_confirm.json and the job log logs/queue_confirm/<job>.log holds exactly one [wd_only] line "
        "reading max_steps 96, warm-up 10, executed 96, sources exactly \"training queue job (g1/queue_g1.json)\" and "
        "train-log LRs \"equal at all 96 steps\"; that queue job (recorded in results/queue_state_g1.json with this run "
        "folder as its root and status done) lists steps 96 and warm-up 10; and the run's training log holds steps 1 to "
        "96, each with its learning rates. Only logs and queue files are read, no evaluation value.")


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


def arg_after(args: list, flag: str):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def as_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def training_job(root: str, run_dir: str):
    """The g1 queue job whose state records run_dir as its root (weight_controls._training_queue_job, A-2)."""
    qp = os.path.join(root, TRAIN_QUEUE.replace("/", os.sep))
    info = {"path": TRAIN_QUEUE, "exists": os.path.isfile(qp)}
    if not info["exists"]:
        return None, info, ["training queue file g1/queue_g1.json missing"]
    q = load_json(qp)
    info["sha256"] = sha256_file(qp)
    sp = os.path.join(root, q.get("state", "").replace("/", os.sep)) if q.get("state") else None
    info["state_path"] = rel(root, sp) if sp else None
    st = load_json(sp) if sp and os.path.isfile(sp) else {}
    info["state_sha256"] = sha256_file(sp) if sp and os.path.isfile(sp) else None
    hits = [j for j in q.get("jobs", []) if isinstance(j, dict) and norm((st.get(j.get("id")) or {}).get("root") or "?")
            == norm(run_dir)]
    why = []
    if len(hits) != 1:
        why.append(f"{len(hits)} jobs of g1/queue_g1.json record the run folder {rel(root, run_dir)} as their root "
                   f"(expected 1)")
        return None, info, why
    j = hits[0]
    rec = st.get(j["id"]) or {}
    info.update(job=j["id"], steps=j.get("steps"), warmup=j.get("warmup"), state_status=rec.get("status"),
                state_root=rec.get("root"), dclm_run_suffix=(j.get("env") or {}).get("DCLM_RUN_SUFFIX"),
                dclm_seed=(j.get("env") or {}).get("DCLM_SEED"))
    if rec.get("status") != "done":
        why.append(f"training job {j['id']} has state status {rec.get('status')!r} (expected 'done')")
    if info["dclm_run_suffix"] != j["id"]:
        why.append(f"training job {j['id']}: DCLM_RUN_SUFFIX {info['dclm_run_suffix']!r} differs from the job id")
    if j.get("steps") != STEPS or j.get("warmup") != WARMUP:
        why.append(f"training job {j['id']} lists steps {j.get('steps')} and warm-up {j.get('warmup')} "
                   f"(expected {STEPS} and {WARMUP})")
    return j, info, why


def train_log(root: str, run_dir: str):
    tl = os.path.join(run_dir, "trainlog", "train_steps.jsonl")
    info = {"path": rel(root, tl), "exists": os.path.isfile(tl)}
    if not info["exists"]:
        return info, ["training log trainlog/train_steps.jsonl missing"], None
    recs = [json.loads(x) for x in open(tl, encoding="utf-8").read().splitlines() if x.strip()]
    steps = [r.get("step") for r in recs]
    with_lrs = [r for r in recs if isinstance(r.get("lrs"), list) and r["lrs"]]
    max_lr = max((abs(float(v)) for r in with_lrs for v in r["lrs"]), default=None)
    info.update(sha256=sha256_file(tl), records=len(recs), first_step=steps[0] if steps else None,
                last_step=steps[-1] if steps else None, records_with_lrs=len(with_lrs), max_lr=max_lr)
    why = []
    if steps != list(range(1, STEPS + 1)) or len(with_lrs) != STEPS:
        why.append(f"training log holds {len(recs)} records (steps {info['first_step']}..{info['last_step']}), "
                   f"{len(with_lrs)} with learning rates (expected steps 1..{STEPS}, each with learning rates)")
    return info, why, max_lr


def check_job(root: str, pos: int, jid: str, queue: dict, state: dict, logs: str) -> tuple[bool, dict]:
    why = []
    c = {"position": pos}
    jobs = queue.get("jobs", [])
    qj = jobs[pos - 1] if 0 < pos <= len(jobs) else None
    if not qj or qj.get("id") != jid:
        why.append(f"confirm/queue_confirm.json position {pos} is {qj.get('id') if qj else None!r}, not {jid}")
        c["reasons"] = why
        return False, c
    args = [str(a) for a in qj.get("args", [])]
    manifest = arg_after(args, "--manifest")
    variants = (arg_after(args, "--variants") or "").split(",")
    c["experiment"] = args[0] if args else None
    if "wdonly" not in variants:
        why.append("the queue job has no wdonly variant")
    run_dir = os.path.dirname(os.path.abspath(manifest)) if manifest else None
    c["run_folder"] = rel(root, run_dir) if run_dir else None
    rec = state.get(jid) or {}
    c["job_status"] = rec.get("status")
    if rec.get("status") != "done":
        why.append(f"job state {rec.get('status')!r} in the confirm queue state (no value is reported for a job "
                   f"that did not end 'done')")
    # the evaluation log
    lp = os.path.join(logs, f"{jid}.log")
    c["eval_log"] = {"path": rel(root, lp), "exists": os.path.isfile(lp)}
    line = None
    if not c["eval_log"]["exists"]:
        why.append("evaluation log missing (no [wd_only] line)")
        c["wd_only_lines"] = 0
    else:
        c["eval_log"].update(sha256=sha256_file(lp), bytes=os.path.getsize(lp))
        lines = [x.rstrip("\r\n") for x in open(lp, encoding="utf-8", errors="replace") if "[wd_only]" in x]
        c["wd_only_lines"] = len(lines)
        if not lines:
            why.append("the evaluation log has no [wd_only] line")
        elif len(lines) > 1:
            why.append(f"the evaluation log has {len(lines)} [wd_only] lines (expected one)")
        else:
            line = lines[0]
    c["wd_only_line"] = line
    m = LINE.search(line) if line else None
    if line and not m:
        why.append("the [wd_only] line is not in the A-2 format (weight_controls.py lines 523-525)")
    g = m.groupdict() if m else {}
    c["steps_found"] = ({"max_steps": as_int(g["max_steps"]), "warmup": as_int(g["warmup"]),
                         "executed": as_int(g["executed"])} if m else None)
    c["source"] = g.get("sources")
    c["fingerprint"] = g.get("fingerprint")
    c["train_log_lrs"] = g.get("train_log_lrs")
    c["device"] = g.get("device")
    c["seconds"] = float(g["seconds"]) if m and re.fullmatch(r"[0-9.]+", g["seconds"]) else None
    if m:
        if c["experiment"] and g["experiment"] != c["experiment"]:
            why.append(f"the line names experiment {g['experiment']}, the queue job {c['experiment']}")
        if c["steps_found"] != {"max_steps": STEPS, "warmup": WARMUP, "executed": STEPS}:
            why.append(f"the line reports {g['max_steps']}/{g['warmup']}/{g['executed']} "
                       f"(expected {STEPS}/{WARMUP}/{STEPS})")
        if g["sources"] != SOURCE:
            why.append(f"the line's source is {g['sources']!r} (expected {SOURCE!r})")
        if g["train_log_lrs"] != LRS_OK:
            why.append(f"the line reads train-log LRs {g['train_log_lrs']!r} (expected {LRS_OK!r})")
    # the training queue job and the training log
    max_lr = None
    if run_dir:
        _, c["training_queue"], w = training_job(root, run_dir)
        why += w
        c["train_log"], w, max_lr = train_log(root, run_dir)
        why += w
    else:
        why.append("the queue job has no --manifest argument (run folder unknown)")
    bound = REL_TOL * max_lr / (1.0 - REL_TOL) if max_lr is not None else None
    c["lr_max_abs_diff"] = {
        "value": None,
        "bound": bound,
        "basis": ("not logged: weight_controls.wd_only_state (lines 512-521) compares the decay-only learning rates "
                  "with the training log in memory and raises unless |a - b| <= 1e-9 max(|a|, |b|) at every step; "
                  "with a <= max_lr of the training log, |a - b| <= 1e-9 max_lr / (1 - 1e-9) when the line reads "
                  "'equal at all 96 steps'"),
    }
    recorded = {}
    for k, v in A2_RECORDED.items():
        p = os.path.join(root, k.replace("/", os.sep))
        if k.startswith("g1/") or k.startswith("results/") or (run_dir and norm(p).startswith(norm(run_dir))):
            recorded[k] = (sha256_file(p) == v) if os.path.isfile(p) else None
    c["a2_recorded_inputs_unchanged"] = recorded
    c["reasons"] = why
    return not why, c


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="A-2 acceptance of the WD-only values of positions 137, 142 and 147")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(here)))
    ap.add_argument("--logs", default=None, help="default <root>/logs/queue_confirm")
    ap.add_argument("--state", default=None, help="default <root>/results/queue_state_confirm.json")
    ap.add_argument("--out", default=None, help="default <root>/results/a2_acceptance.json")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    d_logs = os.path.join(root, "logs", "queue_confirm")
    d_state = os.path.join(root, "results", "queue_state_confirm.json")
    d_out = os.path.join(root, "results", "a2_acceptance.json")
    logs, state_p, out = (os.path.abspath(x) if x else d for x, d in ((a.logs, d_logs), (a.state, d_state), (a.out, d_out)))
    real = norm(out) == norm(d_out)

    def refuse(code: int, msg: str) -> int:
        print(f"[a2_acceptance] refused: {msg}; nothing written")
        return code
    if real and (norm(logs) != norm(d_logs) or norm(state_p) != norm(d_state)):
        return refuse(2, "the default output takes only the default logs and state")
    if os.path.exists(out):
        return refuse(2, f"{rel(root, out)} exists (never overwritten; move it aside first)")
    if norm(out) in freeze_paths(root):
        return refuse(2, f"{rel(root, out)} is listed in FREEZE.json")
    for p in (os.path.join(root, "confirm", "queue_confirm.json"), state_p):
        if not os.path.isfile(p):
            return refuse(2, f"{p} missing")
    queue = load_json(os.path.join(root, "confirm", "queue_confirm.json"))
    state = load_json(state_p)
    if real:
        open_ = [f"{jid} ({(state.get(jid) or {}).get('status', 'pending')})" for _, jid in JOBS
                 if (state.get(jid) or {}).get("status") not in FINAL]
        if open_:
            return refuse(3, "not ended yet: " + ", ".join(open_))
    accepted, checks = {}, {}
    for pos, jid in JOBS:
        accepted[jid], checks[jid] = check_job(root, pos, jid, queue, state, logs)
        r = checks[jid]["reasons"]
        print(f"[a2_acceptance] {pos} {jid}: {'accepted' if accepted[jid] else 'NOT accepted'}"
              + (f" ({'; '.join(r)})" if r else ""))
    doc = {"accepted": accepted, "checks": checks, "rule": RULE,
           "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, out)
    print(f"[a2_acceptance] -> {rel(root, out)} sha256 {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
