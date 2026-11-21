"""Gated, resumable run queue for the local GPU (Windows; single process; no multiprocessing pipes).

  python run_queue.py queue.json [--dry-run] [--only id1,id2] [--retry-failed]
  (PowerShell: .\\run_queue.ps1 -Queue queue.json [-DryRun])

Every job runs as a child process whose stdout/stderr go to a log FILE (no pipes); the queue state
(results/queue_state.json by default) is rewritten atomically after every job, so a restart skips
finished jobs. A file named STOP next to the state file stops the queue before the next job.

Queue file:
{ "vars": {"python": ".venv/nestor/Scripts/python.exe", "run_exp": "run_exp.py" (default: next to this file),
           "base_moe": "outputs/.../model_final.pt"},
  "env": {...},                                  # global env overrides for every job
  "state": "results/queue_state.json", "logs": "logs/queue",
  "jobs": [ {"id": ..., "kind": "cmd" | "train" | "eval" | "audit" | "gate", "requires": [ids],
             "requires_fail": [gate ids], "est_h": 0.4, ...} ] }
"file_vars": {"name": "file.json#key.0"} become {name} placeholders once the file exists (e.g. the SGD lr grid).
Strings may use {var} placeholders from "vars" (and {out_root} inside a train job); a var "glob:<pattern>"
resolves to the newest match; {root_<job>} / {final_<job>} name a finished train job's run root / final
checkpoint (roots depend on the OOM fallback); "spec_file": "file.json#draws.0.spec_A" picks a sub-object.

kind=train: {"experiment", "steps", "warmup", "tokens_per_step", "seq", "micro", "ckpt": bool,
             "out_root", "env": {...}, "spec": {...} | "spec_file", "interim": {...}, "audit": true,
             "expect_audit_fail": false, "evals": [{"name", "sets": ["dclm=path:tokens", ...],
             "variants": "ckpt", "routing": false, "base": "{base_moe}"}]}
   runs run_exp.py with NESTOR_OUTPUT_ROOT=<out_root>/m<micro>[c] and the run's env; OOM falls back to
   activation checkpointing, then half the micro-batch (like run_continual.train); writes run_env.json;
   then the bitwise audit (subset_adapt.py verify; required to pass unless expect_audit_fail, the GPU
   negative control, which requires it to FAIL) and the listed per-window evals (eval_windows.py run).
kind=gate: {"gate": "g0" | "signal" | "audit" | "negctrl" | "json" | "python", "params": {...}}
   g0      params.decision: g0_decision.json -> pass iff decision == GO
   signal  G1: params.full / params.subset = [{"before": npz, "after": npz}], min_full 0.01,
           min_subset 0.005, min_z 10 -> pass iff every FULL F >= min_full and every subset F >= min_subset
           and F >= min_z * SE (paired per-window SE)
   audit   params.jobs: train job ids whose audits must all be ok
   negctrl params.job: the negative-control train job whose audit must have FAILED
   json    params.file, params.key (dotted), params.equals
   python  params.func "module:function" called with (state, params) -> {"pass": bool, ...}
"""
from __future__ import annotations

import glob
import importlib
import json
import os
import subprocess
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))


def _fmt(obj, vars_: dict):
    if isinstance(obj, str):
        try:
            return obj.format(**vars_)
        except (KeyError, IndexError, ValueError):
            return obj
    if isinstance(obj, list):
        return [_fmt(x, vars_) for x in obj]
    if isinstance(obj, dict):
        return {k: _fmt(v, vars_) for k, v in obj.items()}
    return obj


class Queue:
    def __init__(self, path: str, dry_run: bool = False):
        self.path = os.path.abspath(path)
        self.q = json.load(open(path, encoding="utf-8"))
        self.vars = dict(self.q.get("vars", {}))
        self.vars.setdefault("python", sys.executable)
        self.globbed = {}
        for k, v in list(self.vars.items()):              # "glob:outputs/moe_100m_b05/*/checkpoints/model_final.pt"
            if isinstance(v, str) and v.startswith("glob:"):
                hits = sorted(glob.glob(v[5:], recursive=True), key=os.path.getmtime)
                self.vars[k] = os.path.abspath(hits[-1]) if hits else v
                self.globbed[k] = self.vars[k] if hits else None
        self.state_path = os.path.abspath(self.q.get("state", "results/queue_state.json"))
        self.logs = os.path.abspath(self.q.get("logs", "logs/queue"))
        self.dry = dry_run
        self.genv = {k: str(v) for k, v in _fmt(self.q.get("env", {}), self.vars).items()}   # global env, {var}s filled
        self.state = json.load(open(self.state_path, encoding="utf-8")) if os.path.exists(self.state_path) else {}
        os.makedirs(self.logs, exist_ok=True)
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        ids = [j["id"] for j in self.q["jobs"]]
        if len(ids) != len(set(ids)):
            raise SystemExit("duplicate job ids in queue")

    # ------------------------------------------------------------------ bookkeeping
    def save(self):
        if self.dry:
            return
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, self.state_path)

    def log(self, msg: str):
        line = f"[queue] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if not self.dry:
            with open(os.path.join(self.logs, "queue.log"), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def run_cmd(self, argv: list[str], env: dict, log_name: str) -> int:
        lf = os.path.join(self.logs, f"{log_name}.log")
        if self.dry:
            self.log(f"DRY {' '.join(argv)}")
            return 0
        full = dict(os.environ, **self.genv)
        full.update({k: str(v) for k, v in env.items()})
        with open(lf, "w", encoding="utf-8") as fh:
            fh.write(f"# {' '.join(argv)}\n# env overrides: {json.dumps(env)}\n")
            fh.flush()
            p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=full, cwd=os.getcwd())
        return p.returncode

    def vars_now(self) -> dict:
        """vars + {root_<job>} / {final_<job>} of finished train jobs (their roots depend on OOM fallbacks)."""

        v = dict(self.vars)
        for jid, st in self.state.items():
            key = jid.replace("-", "_")
            for f in ("root", "final"):
                if st.get(f):
                    v[f"{f}_{key}"] = st[f]
        for name, ref in self.q.get("file_vars", {}).items():     # {"sgd_lr0": "g2/sgd_grid.json#grid.0"}
            path, _, ptr = _fmt(ref, v).partition("#")
            if os.path.exists(path):
                try:
                    val = json.load(open(path, encoding="utf-8"))
                    for k in (ptr.split(".") if ptr else []):
                        val = val[int(k)] if isinstance(val, list) else val[k]
                    v[name] = val if isinstance(val, str) else json.dumps(val)
                except (KeyError, IndexError, ValueError):
                    pass
        return v

    def ready(self, job: dict) -> tuple[bool, str]:
        for r in job.get("requires", []):
            st = self.state.get(r, {})
            if st.get("status") not in ("done", "pass"):
                return False, f"requirement {r} is {st.get('status', 'pending')}"
        for r in job.get("requires_fail", []):
            st = self.state.get(r, {})
            if st.get("status") != "fail":
                return False, f"requirement fail({r}) not met ({st.get('status', 'pending')})"
        return True, ""

    # ------------------------------------------------------------------ job kinds
    def job_cmd(self, job):
        rc = self.run_cmd(_fmt(job["argv"], self.vars_now()), _fmt(job.get("env", {}), self.vars_now()), job["id"])
        return {"status": "done" if rc == 0 else "failed", "rc": rc}

    def job_eval(self, job):
        argv = [self.vars["python"], os.path.join(HERE, "eval_windows.py"), "run"] + _fmt(job["args"], self.vars_now())
        rc = self.run_cmd(argv, _fmt(job.get("env", {}), self.vars_now()), job["id"])
        return {"status": "done" if rc == 0 else "failed", "rc": rc}

    def job_audit(self, job):
        a = _fmt(job, self.vars_now())
        out = a.get("out") or os.path.join(self.logs, f"{job['id']}_audit.json")
        argv = [self.vars["python"], os.path.join(HERE, "subset_adapt.py"), "verify", "--init", a["init"],
                "--final", a["final"], "--manifest", a["manifest"], "--out", out]
        rc = self.run_cmd(argv, {}, job["id"])
        return {"status": "done" if rc == 0 else "failed", "rc": rc, "audit": out, "audit_ok": rc == 0}

    def job_train(self, job):
        j = _fmt(job, dict(self.vars_now(), out_root=job.get("out_root", "")))
        seq = int(j.get("seq", 4096))
        tps = int(j.get("tokens_per_step", 262144))
        micro0, ckpt0 = int(j.get("micro", 1)), bool(j.get("ckpt", False))
        attempts = [(micro0, ckpt0)] + ([(micro0, True)] if not ckpt0 else []) + ([(micro0 // 2, True)] if micro0 > 1 else [])
        spec = j.get("spec")
        res: dict[str, Any] = {"attempts": []}
        if spec is None and j.get("spec_file"):          # "contrast__C1_german.json#draws.0.spec_A"
            path, _, ptr = j["spec_file"].partition("#")
            if os.path.exists(path):
                spec = json.load(open(path, encoding="utf-8"))
                for k in (ptr.split(".") if ptr else []):
                    spec = spec[int(k)] if isinstance(spec, list) else spec[k]
            elif self.dry:
                spec = {"mode": "train", "select": [], "_placeholder": j["spec_file"]}
            else:
                res.update(status="failed", error=f"spec_file {path} not found")
                return res
        for micro, ck in attempts:
            accum = tps // (micro * seq)
            if accum < 1 or accum * micro * seq != tps:
                res.update(status="failed", error=f"tokens_per_step {tps} is not a multiple of micro {micro} x seq {seq}")
                return res
            root = os.path.abspath(os.path.join(j["out_root"], f"m{micro}{'c' if ck else ''}"))
            if not self.dry and os.path.isdir(root) and not glob.glob(os.path.join(root, "**", "model_final.pt"), recursive=True):
                stale = root + f"_stale{int(time.time())}"         # crashed earlier attempt: keep it, start clean
                os.replace(root, stale)                              # (appended jsonl logs would mix two attempts)
                self.log(f"{job['id']}: moved unfinished root {root} -> {stale}")
            env = dict(self.genv)
            env.update({k: str(v) for k, v in j.get("env", {}).items()})
            env.update({"NESTOR_OUTPUT_ROOT": root, "DCLM_BLOCK_SIZE": str(seq), "DCLM_CKPT_FREQ": "100000",
                        "DCLM_LOG_FREQ": env.get("DCLM_LOG_FREQ", "5"),
                        "NESTOR_ACTIVATION_CHECKPOINT": "1" if ck else "0",
                        "NESTOR_ACTIVATION_CHECKPOINT_SPLIT": "1" if ck else "0",
                        "NESTOR_ACTIVATION_CHECKPOINT_LEAN": "1" if ck else "0",
                        "NESTOR_TRAIN_LOG_DIR": os.path.join(root, "trainlog")})
            if spec is not None:
                env["NESTOR_SUBSET_SPEC"] = json.dumps(spec)
            if j.get("interim"):
                env["NESTOR_INTERIM_EVAL"] = json.dumps(dict(j["interim"], out=os.path.join(root, "interim")))
            if j.get("negative_control"):
                env["NESTOR_SUBSET_NEGCTRL"] = j["negative_control"]
                j.setdefault("expect_audit_fail", True)
            if not self.dry:
                os.makedirs(root, exist_ok=True)
                json.dump({k: v for k, v in env.items()}, open(os.path.join(root, "run_env.json"), "w", encoding="utf-8"), indent=1)
            argv = [self.vars["python"], self.vars.get("run_exp") or os.path.join(HERE, "run_exp.py"), j["experiment"], "--steps", str(j["steps"]),
                    "--warmup", str(j.get("warmup", 10)), "--micro", str(micro), "--accum", str(accum)]
            name = f"{job['id']}_m{micro}{'c' if ck else ''}"
            t0 = time.time()
            rc = self.run_cmd(argv, env, name)
            res["attempts"].append({"micro": micro, "ckpt": ck, "rc": rc, "minutes": round((time.time() - t0) / 60, 2)})
            if self.dry:
                res.update(status="done", root=root)
                break
            finals = sorted(glob.glob(os.path.join(root, "**", "model_final.pt"), recursive=True))
            if rc == 0 and finals:
                res.update(status="done", root=root, final=finals[-1])
                break
            txt = open(os.path.join(self.logs, f"{name}.log"), encoding="utf-8", errors="replace").read()
            if "OutOfMemoryError" in txt:
                self.log(f"{job['id']}: OOM at micro={micro} ckpt={ck}; falling back")
                os.replace(root, root + f"_oom{int(time.time())}")       # keep the failed attempt; never delete
                continue
            res.update(status="failed", root=root)
            return res
        else:
            res["status"] = "failed"
            return res
        if self.dry:
            return res
        init = j.get("env", {}).get("DCLM_INIT_CHECKPOINT") or self.genv.get("DCLM_INIT_CHECKPOINT")
        if j.get("audit", spec is not None) and spec is not None:
            if not init:
                res.update(status="failed", error="audit needs DCLM_INIT_CHECKPOINT in the job or queue env")
                return res
            out = os.path.join(res["root"], "audit.json")
            argv = [self.vars["python"], os.path.join(HERE, "subset_adapt.py"), "verify", "--init", init, "--final",
                    res["final"], "--manifest", os.path.join(res["root"], "subset_manifest.json"), "--out", out]
            if j.get("allow_unchanged"):
                argv.append("--allow-unchanged")
            rc = self.run_cmd(argv, {}, f"{job['id']}_audit")
            res["audit"], res["audit_ok"] = out, rc == 0
            if j.get("expect_audit_fail"):
                if rc == 0:
                    res["status"] = "failed"
                    res["error"] = "negative control: the audit PASSED but had to fail"
                    return res
            elif rc != 0:
                res["status"] = "failed"
                res["error"] = "bitwise audit failed"
                return res
        for ev in j.get("evals", []):
            argv = [self.vars["python"], os.path.join(HERE, "eval_windows.py"), "run", j["experiment"], res["final"],
                    "--out", os.path.join(res["root"], f"eval_{ev.get('name', 'main')}"), "--variants", ev.get("variants", "ckpt")]
            for s in ev["sets"]:
                argv += ["--set", s]
            for s in ev.get("spans", []):          # registered evaluation grids: "name=0.5,1[,doc]" (prereg sec. 4.3)
                argv += ["--span", s]
            argv += ["--block", str(seq)]
            for flag in ("routing", "bytes"):
                if ev.get(flag):
                    argv.append(f"--{flag}")
            for opt in ("base", "manifest", "select", "seed"):
                if ev.get(opt) is not None:
                    argv += [f"--{opt}", str(ev[opt])]
            if any(v.split(":")[0] == "wdonly" for v in ev.get("variants", "").split(",")):
                argv += ["--manifest", os.path.join(res["root"], "subset_manifest.json"),
                         "--run-env", os.path.join(res["root"], "run_env.json")]
            rc = self.run_cmd(argv, {"DCLM_BLOCK_SIZE": str(seq)}, f"{job['id']}_eval_{ev.get('name', 'main')}")
            if rc != 0:
                res["status"] = "failed"
                res["error"] = f"eval {ev.get('name', 'main')} failed"
                return res
        return res

    def job_gate(self, job):
        if self.dry:
            return {"status": "pass", "detail": "dry run"}
        try:
            return self._gate(job)
        except (OSError, KeyError, IndexError, ValueError, TypeError) as e:     # e.g. a decision file never written
            return {"status": "fail", "detail": {"error": f"{type(e).__name__}: {e}"}}

    def _gate(self, job):
        g, prm = job["gate"], _fmt(job.get("params", {}), self.vars_now())
        if g == "g0":
            d = json.load(open(prm["decision"], encoding="utf-8"))
            ok = d.get("decision") == "GO"
            return {"status": "pass" if ok else "fail", "detail": d}
        if g == "signal":
            import eval_windows as EW

            rows, ok = [], True
            for kind, thr in (("full", float(prm.get("min_full", 0.01))), ("subset", float(prm.get("min_subset", 0.005)))):
                for pair in prm.get(kind, []):
                    r = EW.paired(EW.load_result(pair["before"]), EW.load_result(pair["after"]))
                    z_ok = kind == "full" or r["diff"] >= float(prm.get("min_z", 10)) * r["se"]
                    good = r["diff"] >= thr and z_ok
                    ok &= good
                    rows.append({"kind": kind, **pair, "F": r["diff"], "se": r["se"], "pass": good})
            return {"status": "pass" if ok else "fail", "detail": rows,
                    "decision": "GO" if ok else "lengthen to 192 steps or q = 1/2 (pre-declared)"}
        if g == "audit":
            bad = [j for j in prm["jobs"] if not self.state.get(j, {}).get("audit_ok")]
            return {"status": "pass" if not bad else "fail", "detail": {"not_ok": bad}}
        if g == "negctrl":
            st = self.state.get(prm["job"], {})
            ok = st.get("status") == "done" and st.get("audit_ok") is False
            return {"status": "pass" if ok else "fail", "detail": st}
        if g == "json":
            v = json.load(open(prm["file"], encoding="utf-8"))
            for k in prm["key"].split("."):
                v = v[k]
            return {"status": "pass" if v == prm["equals"] else "fail", "detail": {"value": v}}
        if g == "python":
            mod, fn = prm["func"].split(":")
            out = getattr(importlib.import_module(mod), fn)(self.state, prm)
            return {"status": "pass" if out.get("pass") else "fail", "detail": out}
        raise SystemExit(f"unknown gate {g}")

    # ------------------------------------------------------------------ main loop
    def run(self, only: set | None = None, retry_failed: bool = False) -> int:
        est = sum(float(j.get("est_h", 0)) for j in self.q["jobs"] if (not only or j["id"] in only))
        self.log(f"queue {self.path}: {len(self.q['jobs'])} jobs, estimated {est:.2f} GPU-h{' (DRY RUN)' if self.dry else ''}")
        for k, v in self.globbed.items():
            self.log(f"var {k} = {v if v else 'NO MATCH for ' + self.q['vars'][k]}")
        stop_file = os.path.join(os.path.dirname(self.state_path), "STOP")
        for job in self.q["jobs"]:
            jid = job["id"]
            if only and jid not in only:
                continue
            prev = self.state.get(jid, {}).get("status")
            if prev in ("done", "pass", "fail") or (prev == "failed" and not retry_failed) or prev == "skipped":
                continue
            if os.path.exists(stop_file):
                self.log(f"STOP file found ({stop_file}); stopping before {jid}")
                return 2
            ok, why = self.ready(job)
            if not ok:
                dead = any(self.state.get(r, {}).get("status") in ("fail", "failed", "skipped") for r in job.get("requires", []))
                dead |= any(self.state.get(r, {}).get("status") in ("pass", "done", "skipped") for r in job.get("requires_fail", []))
                if dead:                         # a requirement can no longer be met: skip for good
                    self.state[jid] = {"status": "skipped", "reason": why}
                    self.save()
                    self.log(f"skip {jid}: {why}")
                continue
            self.log(f"start {jid} ({job['kind']}, est {job.get('est_h', 0)} h)")
            t0 = time.time()
            fn = {"cmd": self.job_cmd, "train": self.job_train, "eval": self.job_eval, "audit": self.job_audit,
                  "gate": self.job_gate}[job["kind"]]
            res = fn(job)
            res.update(start=t0, end=time.time(), hours=round((time.time() - t0) / 3600, 4))
            self.state[jid] = res                  # in memory also in a dry run (dependents get listed)
            self.save()                            # no-op in a dry run
            self.log(f"end {jid}: {res['status']}" + (f" ({res.get('error')})" if res.get("error") else ""))
        return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="gated run queue")
    ap.add_argument("queue")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args(argv)
    q = Queue(a.queue, dry_run=a.dry_run)
    return q.run(set(a.only.split(",")) if a.only else None, a.retry_failed)


if __name__ == "__main__":
    sys.exit(main())
