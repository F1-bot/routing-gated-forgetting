"""G1 / G2 run manifests per g1g2_spec.json v1.3.0 'runs' (preregistration v1.3 sec. 6) as run_queue.py queues.

  python g1g2_plan.py g1  --g0-record g0/gate_record.json --config g1g2_config.json --out-dir g1
  python g1g2_plan.py g1r --g0-record g0/gate_record.json --config g1g2_config.json --out-dir g1r   (G1-1 lengthen, O-2)
  python g1g2_plan.py g2  --g0-record g0/gate_record.json --g1-record g1/gate_record_G1.json --config g1g2_config.json --out-dir g2

Runs (spec 1.3.0 'runs'; run ids = the specification's proposed ids):
  G1   MoE base 1: FULL math and German x seeds 101/102; E-ALL German x 101/102; A/A pair a = 1 (German; generator seed
       4001, data seed 7201; R1 = G2-2 SGD calibration subset and Muon reference) and pair a = 2 (books; 4002 / 7202; moved
       into S1 per P-1 so that G1-2 judges the C1 target); DCLM-CONT (seed 101, held-out shard 03/0/0); NESTOR FULL German x
       101/102 (101 has the H100 duplicate); NESTOR books A/A subset a = 1 (4101 / 7251; G1-3); the audit negative control.
  g1r  after a G1-1 lengthen: the 4 MoE FULL runs, both S1 A/A pairs and the NESTOR A/A subset at 192 steps (O-2).
  G2   books pair a = 3 (4003 / 7203) in the G1-2 configuration; pair a = 2 repeated only if G1-2 changed q or steps; 3 SGD
       calibration runs on R1 of pair a = 1 at c/3, c, 3c (WD 0, no clipping) with full DCLM and German evaluations at
       steps 0, 8, 24, 48 (step 0 = in-process theta_0 for G2-3); per-expert gradient SNR.
Interim sets are the registered 64-window heads of the evaluation grids (prereg sec. 4.3; 'head': 64 of the full grid).

Each call writes <out-dir>/plan_<stage>.json (canonical JSON whose sha256 covers every field except "sha256"/"created"),
<out-dir>/queue_<stage>.json (run_queue.py format; resumable by job id), <out-dir>/<stage>_gate_config.json (the gate inputs
named in the hash list: pipeline-test result, routing-health file and App. C.6 ranges, G0-7R record) and specs/*.json.
Inputs are read, never written: the G0 gate record, the registered G0 probe configs, the G0 routing tables. Torch-free.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

TOKENS_PER_STEP = 262144
SEQ = 4096
EVAL_TOKENS = 2048000          # 500 windows x 4096 (sec. 5, W = 500)
PIN_PASSES = 5                 # joint pinned evaluation: 1 theta_0 recording + 4 replays per window (eval_windows r3)

DEFAULT_CONFIG = {
    "_comment": "g1g2_plan.py configuration. Paths are relative to the workspace root (cwd of run_queue).",
    "python": ".venv/nestor/Scripts/python.exe",
    "run_exp": "run_exp.py",
    "experiments": {"moe_b1": "moe_100m", "moe_b2": "moe_100m", "nestor": "nestor_100m"},
    "g0_probe_configs": {"moe_b1": "g0_moe_b1.json", "nestor": "g0_nestor.json", "moe_b2": "g0_moe_b2.json",
                         "moe_b3": "g0_moe_b3.json"},
    "train_data": {"german": "data/domain/fw2de_train.bin", "books": "data/domain/pg19_train.bin",
                   "math": "data/domain/owm_train.bin", "french": "data/domain/fr_train.bin",
                   "code": "data/domain/algstack_train.bin"},
    "dclm_cont": {"path": "data/dclm/dclm_cont_heldout.bin", "pretrain_path": "data/dclm/dclm_train.bin",
                  "note": "DCLM-CONT trains on held-out DCLM training tokens of global shard 03 (used by no "
                          "other step). Bin written by the data rebuild: tok_files.py driver --global-shards 3 --target-tokens "
                          "30000000 -> 29,998,792 tokens, sha256 7000c98f... (data/data_manifest.json)."},
    "recipe": {"lr_muon": 0.0025, "lr_adamw": 7.5e-05, "weight_decay": 0.1, "max_grad_norm": 1.0, "warmup_96": 10,
               "micro": 2, "momentum": 0.95, "steps": 96},
    "seeds": {"aa_generator": {"moe": 4000, "nestor": 4100}, "aa_data": {"moe": 7200, "nestor": 7250},
              "full_moe": [101, 102], "eall_moe": [101, 102], "full_nestor": [101, 102], "dclm_cont": 101,
              "h100_duplicate_seed": 101},                  # FULL seeds 101/102 for every architecture
    "aa_pairs": {"s1": [1, 2], "g2": [3], "targets": {"1": "german", "2": "books", "3": "books"}},
    "nestor_aa": {"pair": 1, "target": "books"},
    "interim_steps": [0, 8, 24, 48],
    "interim_head_windows": 64,
    "sgd_full_steps": [0, 8, 24, 48],
    "negctrl_steps": 8,
    "throughput": {"train_tok_s": {"moe": 34000.0, "nestor": 10200.0}, "eval_tok_s": {"moe": 27000.0, "nestor": 8500.0},
                   "compile_min": {"moe": 3.0, "nestor": 8.5}, "model_load_min": 0.5, "audit_min": 1.0},
    "snr": {"windows": 64, "train_batch_windows": 64, "kinds": "moe"},
    "theta0": {"windows": 64},
    "h100_duplicate_file": "g1/h100_nestor_full_s101.json",
    "gate_config": {
        "spec": "g1g2_spec.json",
        "lookup": "signflip_lookup_v1.csv",
        "pipeline_tests": {"junit": "logs/tests_gpu_pipeline.xml", "log": "logs/tests_gpu_pipeline.log"},
        "code_record": "logs/after_B_code_sha256.json",
        "routing_health": "g0/nestor/routing_health.json",
        "appC6_ranges": None,
        "f_or_stream_changed": False,
        "g07r_record": "g0r/g07r_record.json",
    },
}


# ============================================================================= helpers
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=float)


def sha256_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: str | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        user = json.load(open(path, encoding="utf-8-sig"))

        def merge(a, b):
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    merge(a[k], v)
                else:
                    a[k] = v
        merge(cfg, user)
    return cfg


def g0_decisions(rec: dict) -> dict:
    """The G0 facts G1/G2 depend on, read from the real gate record (g0_gate.py format)."""

    g = rec["gates"]
    g01, g03, g05, g07 = g["G0-1"], g.get("G0-3", {}), g.get("G0-5", {}), g.get("G0-7", {})
    code = (rec.get("decisions", {}).get("code_stream") or {}).get("chosen") or g01.get("code_stream") or "code"
    other = "pycode" if code == "code" else "code"
    cell = g01["cells"].get(f"moe_b1|german|{code}", {})
    # per-target G0-1 feasibility (MoE decision base); the C1 cell actually run on base 1 is the first feasible
    # target in registered order (German cells are replaced by books cells when German is infeasible; draws unchanged)
    mt = g01.get("moe_targets") or {}
    reg_targets = list(mt) if mt else ["german", "books"]
    feas = [t for t in reg_targets if mt.get(t)] if mt else None
    # spec 1.3.0 / prereg v1.3 sec. 3.1: C1 is books-only after G0-1; P-bar of G2-1 and f are those of the MoE base-1 BOOKS
    # cell at the chosen code stream (G0 record: median 0.989 at f = 1/2, stream 'code'), whatever other cells exist
    c1t = "books"
    c1cell = g01["cells"].get(f"moe_b1|{c1t}|{code}", {})
    pb2 = (g05.get("per_base") or {})
    res3 = g03.get("result") or {}
    return {
        "pivot": not bool(g01.get("moe_pass")),
        "nestor_g0_1": bool(g01.get("nestor_pass")),
        "nestor_g0_5": bool((pb2.get("nestor") or {}).get("pass")),
        "moe_b2_g0_5": bool((pb2.get("moe_b2") or {}).get("pass")),
        "code_stream": code, "code_secondary": other,
        "f": c1cell.get("f"), "f_label": c1cell.get("f_label"),
        "P_bar_moe_b1_german": cell.get("median"),
        "P_bar_c1_moe_b1": c1cell.get("median"), "P_bar_source": f"moe_b1|{c1t}|{code}" if c1cell else None,
        "c1_target": c1t, "books_cell_feasible": bool(c1cell.get("feasible", c1cell.get("median") is not None)),
        "c1_registered_targets": reg_targets, "c1_feasible_targets": feas,
        "moe_targets": mt or None, "nestor_targets": g01.get("nestor_targets"),
        "g0_3_pass": bool(g03.get("pass")),
        "c2_matching_feasible": any(r.get("design") == "C2" and r.get("feasible") for r in (g.get("G0-2") or {}).get("rows", [])),
        "c2_matched": res3.get("matched"),
        "beta0": {b: (g07.get(b) or {}).get("beta0") for b in ("moe_b1", "moe_b2")},
        "margin_limited": {b: (g07.get(b) or {}).get("margin_limited") for b in ("moe_b1", "moe_b2")},
        "k": rec.get("k", {}),
        "checkpoints": {b: (v[0] if isinstance(v, list) else v) for b, v in (rec.get("checkpoints") or {}).items()},
        "bases": ((rec.get("config") or {}).get("resolved") or {}).get("bases", {}),
    }


def eval_sets(probe_cfg: dict) -> dict:
    """Registered evaluation grids = the '<c>_eval' streams of the G0 probe config (prereg sec. 4.3)."""

    out = {}
    for name, s in probe_cfg.get("streams", {}).items():
        if not name.endswith("_eval"):
            continue
        c = name[:-5]
        out[c] = {"path": s["path"], "tokens": int(s.get("tokens", EVAL_TOKENS)), "select": s.get("select", "even")}
        if s.get("span"):
            out[c]["span"] = [float(s["span"][0]), float(s["span"][1])]
        if s.get("span_align"):
            out[c]["span_align"] = s["span_align"]
    return out


def set_args(sets: dict, names: list[str]) -> tuple[list[str], list[str], str]:
    """eval_windows CLI pieces: ["name=path:tokens"], ["name=f0,f1[,doc]"], select mode (one per call)."""

    sel = {sets[n].get("select", "even") for n in names}
    if len(sel) != 1:
        raise ValueError(f"sets {names} mix selection modes {sel}")
    a = [f"{n}={sets[n]['path']}:{sets[n]['tokens']}" for n in names]
    sp = []
    for n in names:
        s = sets[n]
        if s.get("span"):
            sp.append(f"{n}={s['span'][0]:g},{s['span'][1]:g}" + (",doc" if s.get("span_align") == "doc" else ""))
    return a, sp, sel.pop()


def interim_cfg(sets: dict, names: list[str], steps: list[int], head: int = 64, full_steps: list[int] | None = None,
                full_sets: list[str] | None = None) -> dict:
    """adapt_extras NESTOR_INTERIM_EVAL config: each set is the registered 64-window HEAD of its full evaluation grid
    ('tokens' = the full grid, 'head' = 64; prereg sec. 4.3). With full_steps (and NESTOR_INTERIM_FULL=1 in the job env) the
    full grid of full_sets is evaluated at those steps and the head file is sliced from it."""
    cfg = {"steps": sorted(int(s) for s in steps), "block": SEQ, "routing": False,
            "sets": [dict({"name": n, "path": sets[n]["path"], "tokens": sets[n]["tokens"], "head": int(head),
                           "select": sets[n].get("select", "even")},
                          **({"span": sets[n]["span"]} if sets[n].get("span") else {}),
                          **({"span_align": sets[n]["span_align"]} if sets[n].get("span_align") else {}))
                     for n in names]}
    if full_steps:
        cfg["full_steps"] = sorted(int(s) for s in full_steps)
        cfg["full_sets"] = list(full_sets or names)
    return cfg


# ============================================================================= cost model
def est_train_h(arch: str, steps: int, cfg: dict, *, interim_evals: int = 0, evals: int = 0, audit: bool = False,
                interim_full_evals: int = 0) -> dict:
    """interim_evals = number of 64-window head evaluations (64/500 of a full grid each); interim_full_evals = full-grid
    interim evaluations (NESTOR_INTERIM_FULL)."""
    tp = cfg["throughput"]
    train = steps * TOKENS_PER_STEP / tp["train_tok_s"][arch] / 3600
    comp = tp["compile_min"][arch] / 60
    ev_one = EVAL_TOKENS / tp["eval_tok_s"][arch] / 3600
    head = int(cfg.get("interim_head_windows", 64)) * SEQ / EVAL_TOKENS
    interim = interim_evals * head * ev_one + interim_full_evals * ev_one
    ev = evals * ev_one + (tp["model_load_min"] / 60 if evals else 0)
    aud = tp["audit_min"] / 60 if audit else 0.0
    tot = train + comp + interim + ev + aud
    return {"train_h": round(train, 4), "compile_h": round(comp, 4), "interim_eval_h": round(interim, 4),
            "eval_h": round(ev, 4), "audit_h": round(aud, 4), "total_h": round(tot, 4)}


def est_eval_h(arch: str, n_sets: int, cfg: dict) -> float:
    tp = cfg["throughput"]
    return round(n_sets * EVAL_TOKENS / tp["eval_tok_s"][arch] / 3600 + tp["model_load_min"] / 60, 4)


# ============================================================================= A/A draws
def aa_draw(base_dir: str, family: str, a: int, target: str, seed_base: int, k: int, out_dir: str) -> dict:
    import crossover as CX

    t = CX.reg_load_dir(base_dir, family)
    e = CX.reg_aa_draw(t, a, target=target, seed_base=seed_base, k=k)
    if e.get("status") != "ok":
        raise SystemExit(f"A/A draw a={a} ({family}, {target}, seed {seed_base + a}, k={k}) infeasible: {e.get('generator')}")
    e["N"], e["L"], e["family"], e["q"] = t.N, t.L, family, k / t.N
    os.makedirs(os.path.join(out_dir, "specs"), exist_ok=True)
    p = os.path.join(out_dir, "specs", f"aa_{family}_{target}_a{a}.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(canonical(e))
    e["_file"] = p
    return e


def all_expert_spec(base_dir: str, family: str) -> dict:
    import crossover as CX

    t = CX.reg_load_dir(base_dir, family)
    # every unit of every stratum. (phase 5 passed a boolean mask to subset_keys, which takes per-stratum INDEX
    # arrays: int(True) = 1 made the E-ALL spec train expert 1 of every layer only; fixed in phase 6)
    keys = [k for row in t.keys for k in row]
    return CX.spec_for(keys)


# ============================================================================= queue builders
class Builder:
    def __init__(self, stage: str, cfg: dict, g0rec_path: str, out_dir: str):
        self.stage, self.cfg, self.out = stage, cfg, out_dir
        self.g0_path = os.path.abspath(g0rec_path)
        self.g0rec = json.load(open(g0rec_path, encoding="utf-8"))
        self.g0 = g0_decisions(self.g0rec)
        self.jobs: list[dict] = []
        self.runs: list[dict] = []
        self.vars = {"python": cfg["python"], "run_exp": cfg["run_exp"]}
        for b, ck in self.g0["checkpoints"].items():
            if ck:
                self.vars[f"ckpt_{b}"] = ck                         # explicit paths (no newest-mtime glob)
        self.probe = {}
        for b, p in cfg["g0_probe_configs"].items():
            if os.path.exists(p):
                self.probe[b] = json.load(open(p, encoding="utf-8-sig"))
        if "moe_b1" not in self.probe:
            raise SystemExit(f"registered G0 probe config for moe_b1 not found: {cfg['g0_probe_configs'].get('moe_b1')}")
        self.sets = {b: eval_sets(c) for b, c in self.probe.items()}
        self.estimates: list[dict] = []
        self.warnings: list[str] = []

    # ---------------------------------------------------------------- job helpers
    def base_dir(self, base: str) -> str:
        d = self.g0["bases"].get(base, {}).get("dir")
        if not d:
            raise SystemExit(f"gate record has no G0 dir for base {base}")
        if not os.path.isabs(d):
            d = os.path.join(os.path.dirname(self.g0_path), d)
        if not os.path.isdir(d):                     # records keep absolute dirs; fall back to <record dir>/<base>
            alt = os.path.join(os.path.dirname(self.g0_path), base)
            if os.path.isdir(alt):
                d = alt
        return d

    def arch(self, base: str) -> str:
        return "nestor" if base == "nestor" else "moe"

    def family(self, base: str) -> str:
        return "priv" if base == "nestor" else "moe"

    def base_eval(self, base: str, names: list[str], jid: str) -> dict:
        sets = self.sets[base if base in self.sets else "moe_b1"]
        a, sp, sel = set_args(sets, names)
        d = os.path.join(self.out, "base", base)
        args = [self.cfg["experiments"][base], f"{{ckpt_{base}}}"]
        for s in a:
            args += ["--set", s]
        for s in sp:
            args += ["--span", s]
        args += ["--select", sel, "--block", str(SEQ), "--out", d]
        h = est_eval_h(self.arch(base), len(names), self.cfg)
        self.jobs.append({"id": jid, "kind": "eval", "args": args, "est_h": h})
        self.estimates.append({"id": jid, "kind": "eval", "base": base, "sets": names, "total_h": h})
        return {n: os.path.join(d, f"ckpt__{n}.npz") for n in names}

    def train(self, rid: str, *, base: str, role: str, data: str, seed: int, steps: int, spec: dict | None,
              eval_sets_: list[str], reflect_sets: list[str] | None = None, interim_sets: list[str] | None = None,
              env_extra: dict | None = None, negctrl: str | None = None, meta: dict | None = None,
              pin_sets: list[str] | None = None, full_steps: list[int] | None = None,
              full_sets: list[str] | None = None) -> dict:
        r = self.cfg["recipe"]
        arch = self.arch(base)
        sets = self.sets[base if base in self.sets else "moe_b1"]
        warm = int(round(r["warmup_96"] * steps / 96))
        env = {"DCLM_DATA_PATH": data, "DCLM_INIT_CHECKPOINT": f"{{ckpt_{base}}}", "DCLM_LR_MUON": str(r["lr_muon"]),
               "DCLM_LR_ADAMW": str(r["lr_adamw"]), "DCLM_WEIGHT_DECAY": str(r["weight_decay"]),
               "NESTOR_FREEZE_ROUTERS": "1", "DCLM_RUN_SUFFIX": rid, "DCLM_SEED": str(seed)}
        env.update(env_extra or {})
        job = {"id": rid, "kind": "train", "experiment": self.cfg["experiments"][base], "steps": steps, "warmup": warm,
               "seq": SEQ, "tokens_per_step": TOKENS_PER_STEP, "micro": int(r["micro"]), "ckpt": False,
               "out_root": os.path.join("outputs", self.stage, rid), "env": env}
        if spec is not None:
            job["spec"] = spec
        if negctrl:
            job["negative_control"] = negctrl
        n_interim = n_interim_full = 0
        if interim_sets:
            job["interim"] = interim_cfg(sets, interim_sets, self.cfg["interim_steps"], int(self.cfg["interim_head_windows"]),
                                         full_steps, full_sets)
            n_interim = len(interim_sets) * len(self.cfg["interim_steps"])
            if full_steps:
                job["env"]["NESTOR_INTERIM_FULL"] = "1"
                n_interim_full = len(full_sets or interim_sets) * len(full_steps)
                n_interim -= len(set(full_sets or interim_sets) & set(interim_sets)) * len(set(full_steps) & set(self.cfg["interim_steps"]))
        evs, n_ev = [], 0
        if eval_sets_:
            a, sp, sel = set_args(sets, eval_sets_)
            evs.append({"name": "main", "sets": a, "spans": sp, "select": sel, "variants": "ckpt"})
            n_ev += len(eval_sets_)
        if reflect_sets:
            a, sp, sel = set_args(sets, reflect_sets)
            evs.append({"name": "reflect", "sets": a, "spans": sp, "select": sel, "variants": "reflect",
                        "base": f"{{ckpt_{base}}}"})
            n_ev += len(reflect_sets)
        if pin_sets:                 # theory verification r3: pinned / selection-pinned replay at theta_0 +- delta
            a, sp, sel = set_args(sets, pin_sets)
            evs.append({"name": "pin", "sets": a, "spans": sp, "select": sel, "base": f"{{ckpt_{base}}}",
                        "variants": "pinned,selpin,reflect+pinned,reflect+selpin"})
            n_ev += PIN_PASSES * len(pin_sets)
        if evs:
            job["evals"] = evs
        est = est_train_h(arch, steps, self.cfg, interim_evals=n_interim, evals=n_ev, audit=spec is not None,
                          interim_full_evals=n_interim_full)
        job["est_h"] = est["total_h"]
        self.jobs.append(job)
        run = {"id": rid, "role": role, "base": base, "arch": arch, "steps": steps, "warmup": warm, "data": data,
               "seed": seed, "eval_dir": "eval_main", "eval_sets": eval_sets_, "reflect_sets": reflect_sets or [],
               "interim_sets": interim_sets or [], "interim_steps": self.cfg["interim_steps"] if interim_sets else [],
               "interim_head_windows": int(self.cfg["interim_head_windows"]) if interim_sets else None,
               "full_steps": sorted(full_steps) if full_steps else [], "full_sets": list(full_sets or interim_sets or []) if full_steps else [],
               "pin_sets": pin_sets or [],
               "subset_spec": spec is not None, "negative_control": negctrl, "env": env, **(meta or {})}
        self.runs.append(run)
        self.estimates.append({"id": rid, "kind": "train", "base": base, "role": role, **est})
        return run

    def cmd(self, jid: str, argv: list[str], est_h: float, out: str | None = None, requires: list[str] | None = None):
        job = {"id": jid, "kind": "cmd", "argv": argv, "est_h": est_h}
        if requires:
            job["requires"] = requires
        self.jobs.append(job)
        self.estimates.append({"id": jid, "kind": "cmd", "total_h": est_h, "out": out})

    # ---------------------------------------------------------------- output
    def write(self, extra: dict) -> dict:
        os.makedirs(self.out, exist_ok=True)
        state = os.path.join("results", f"queue_state_{self.stage}.json")
        queue = {"_comment": f"{self.stage.upper()} queue built by g1g2_plan.py from {self.g0_path}; resumable by job id",
                 "vars": self.vars, "env": {"WANDB_MODE": "disabled", "DCLM_NUM_WORKERS": "0", "PYTHONUTF8": "1",
                         "NESTOR_VRAM_GUARD": "1", "NESTOR_VRAM_BUDGET_FRAC": "0.90"},
                 "state": state, "logs": os.path.join("logs", f"queue_{self.stage}"), "jobs": self.jobs}
        if extra.get("file_vars"):
            queue["file_vars"] = extra.pop("file_vars")
        qtxt = canonical(queue)
        qpath = os.path.join(self.out, f"queue_{self.stage}.json")
        with open(qpath, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(qtxt)
        gcfg = dict(self.cfg.get("gate_config") or {})
        gcfg.update({"rules_version": "1.3.0", "stage": self.stage})
        gpath = os.path.join(self.out, f"{self.stage}_gate_config.json")
        with open(gpath, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(canonical(gcfg))
        extra.setdefault("gate_config", gcfg)
        extra.setdefault("gate_config_file", gpath.replace("\\", "/"))
        tot = round(sum(e["total_h"] for e in self.estimates), 3)
        plan = {"kind": self.stage, "prereg": "v1.3 (spec g1g2 1.3.0)",
                "g0_record": {"path": self.g0_path, "sha256": file_sha256(self.g0_path),
                              "content_sha256": self.g0rec.get("sha256")},
                "g0": self.g0, "config": self.cfg, "sets": self.sets, "runs": self.runs, "queue": qpath.replace("\\", "/"),
                "queue_sha256": sha256_text(qtxt), "state": state.replace("\\", "/"), "estimates": self.estimates,
                "estimate_total_gpu_h": tot, "warnings": self.warnings, **extra}
        body = canonical(plan)
        plan["sha256"] = sha256_text(body)
        plan["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        ppath = os.path.join(self.out, f"plan_{self.stage}.json")
        with open(ppath, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(canonical(plan))
        with open(ppath + ".sha256", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"{plan['sha256']}  content of {os.path.basename(ppath)} without sha256/created\n")
        return plan


def plan_hash(plan: dict) -> str:
    body = {k: v for k, v in plan.items() if k not in ("sha256", "created")}
    return sha256_text(canonical(body))


# ============================================================================= stages
def _fwd(p: str) -> str:
    return p.replace("\\", "/")


def _aa_runs(B: "Builder", stage: str, a: int, target: str, k: int, steps: int, out_dir: str, *, suffix: str = "",
             interim: bool = True) -> dict:
    """One MoE base-1 A/A pair (generator seed 4000 + a, data seed 7200 + a; R1 = subset A, R2 = subset B of the draw)."""
    cfg, sd = B.cfg, B.cfg["seeds"]
    code, code2 = B.g0["code_stream"], B.g0["code_secondary"]
    e = aa_draw(B.base_dir("moe_b1"), "moe", a, target, int(sd["aa_generator"]["moe"]), k, out_dir)
    dseed = int(sd["aa_data"]["moe"]) + a
    runs = {}
    for sub, side in (("R1", "A"), ("R2", "B")):
        rid = f"{stage}_moe_b1_aa{a}_{sub}{suffix}"
        B.train(rid, base="moe_b1", role="aa", data=cfg["train_data"][target], seed=dseed, steps=steps, spec=e[f"spec_{side}"],
                eval_sets_=["dclm", target, code, code2, "math"], reflect_sets=["dclm", code, "math"],
                pin_sets=["dclm", code, "math"], interim_sets=[target] if interim else None,
                meta={"pair": a, "subset": sub, "draw_subset": side, "target": target, "k": k, "generator_seed": e["seed"],
                      "data_seed": dseed, "draw_file": _fwd(e["_file"])})
        runs[sub] = rid
    return {"pair": a, "base": "moe_b1", "target": target, "generator_seed": e["seed"], "k": k, "q": e["q"], "N": e["N"],
            "data_seed": dseed, "draw_file": _fwd(e["_file"]), "runs": runs, "steps": steps, "stage": stage}


def build_g1(cfg: dict, g0rec: str, out_dir: str, *, stage: str = "g1", steps: int | None = None, repeat: bool = False,
             reuse_base_from: str | None = None) -> dict:
    """G1 (S1) runs of spec 1.3.0; repeat=True builds the g1r set of decision O-2 (FULL MoE runs, both S1 A/A pairs and the
    NESTOR A/A subset at 192 steps, paired with the theta_0 evaluations of G1)."""
    B = Builder(stage, cfg, g0rec, out_dir)
    g0, sd, r = B.g0, cfg["seeds"], cfg["recipe"]
    steps = int(steps or r["steps"])
    code, code2 = g0["code_stream"], g0["code_secondary"]
    k_moe = int(g0["k"].get("moe_b1", 6))
    base_sets = ["dclm", code, code2, "math", "german", "books"]
    nestor_ok = g0["nestor_g0_5"] and "nestor" in B.probe
    if reuse_base_from:          # the 192-step repeat pairs with the theta_0 evaluations of G1 (same windows, same base)
        base_files = {b: {n: os.path.join(reuse_base_from, "base", b, f"ckpt__{n}.npz") for n in base_sets}
                      for b in (["moe_b1"] + (["nestor"] if nestor_ok else []))}
    else:
        base_files = {"moe_b1": B.base_eval("moe_b1", base_sets, f"{stage}_base_moe_b1")}
        if nestor_ok:
            base_files["nestor"] = B.base_eval("nestor", base_sets, f"{stage}_base_nestor")
    extra: dict = {"steps": steps, "repeat": repeat, "spec_runs": "g1g2_spec.json 1.3.0 runs.G1" + (" (O-2 repeat)" if repeat else "")}
    dclm_same = os.path.normpath(cfg["dclm_cont"]["path"]) == os.path.normpath(cfg["dclm_cont"]["pretrain_path"])
    if dclm_same:
        B.warnings.append("DCLM-CONT trains on the pretraining shard (config dclm_cont.path): not the held-out shard 03/0/0 "
                          "registered in the design; point it at the held-out bin")
    tg = {int(a): t for a, t in cfg["aa_pairs"]["targets"].items()}
    pairs = [_aa_runs(B, stage, int(a), tg[int(a)], k_moe, steps, out_dir) for a in cfg["aa_pairs"]["s1"]]
    extra["aa_pairs"] = pairs
    p1 = next(p for p in pairs if p["pair"] == 1)
    extra["r1"] = {"run": p1["runs"]["R1"], "draw_file": p1["draw_file"], "subset": "A", "data_seed": p1["data_seed"]}
    if not repeat:
        # theta_0 checks on R1 of pair a = 1: NS-5 scale check, D_Muon unit test (G1-NS5), SGD scale c (G2-2)
        t0 = _fwd(os.path.join(out_dir, "theta0_checks.json"))
        B.cmd(f"{stage}_theta0_moe_b1", ["{python}", "theta0_checks.py", "run", cfg["experiments"]["moe_b1"], "{ckpt_moe_b1}",
                                         "--draw", p1["draw_file"], "--subset", "A", "--data", cfg["train_data"]["german"],
                                         "--windows", str(cfg["theta0"]["windows"]), "--data-seed", str(p1["data_seed"]),
                                         "--lr-muon", str(r["lr_muon"]), "--momentum", str(r["momentum"]), "--out", t0], 0.05, out=t0)
        extra["theta0_checks"] = t0
        # G1-4 negative control: decay masking disabled on a mixed bank -> the bitwise audit must FAIL
        d1 = json.load(open(p1["draw_file"], encoding="utf-8"))
        B.train(f"{stage}_negctrl", base="moe_b1", role="negctrl", data=cfg["train_data"]["german"], seed=p1["data_seed"],
                steps=int(cfg["negctrl_steps"]), spec=d1["spec_A"], eval_sets_=[], negctrl="unmasked_decay")
    for tgt in ("math", "german"):
        for s in sd["full_moe"]:
            B.train(f"{stage}_moe_b1_full_{tgt}_s{s}", base="moe_b1", role="full", data=cfg["train_data"][tgt], seed=int(s),
                    steps=steps, spec=None, eval_sets_=["dclm", code, "math"] + (["german"] if tgt == "german" else []),
                    meta={"target": tgt, "adapt_seed": int(s)})
    if not repeat:
        eall = all_expert_spec(B.base_dir("moe_b1"), "moe")
        for s in sd["eall_moe"]:
            B.train(f"{stage}_moe_b1_eall_german_s{s}", base="moe_b1", role="eall", data=cfg["train_data"]["german"],
                    seed=int(s), steps=steps, spec=eall, eval_sets_=["dclm", code, "math", "german"],
                    meta={"target": "german", "adapt_seed": int(s)})
        B.train(f"{stage}_moe_b1_dclmcont_s{sd['dclm_cont']}", base="moe_b1", role="dclm_cont", data=cfg["dclm_cont"]["path"],
                seed=int(sd["dclm_cont"]), steps=steps, spec=None, eval_sets_=["dclm", code, "math", "german"],
                meta={"target": "dclm", "held_out_tokens": not dclm_same, "adapt_seed": int(sd["dclm_cont"])})
    # NESTOR: listed unconditionally in runs.G1; only a G0-5 failure stops NESTOR adaptation (G0-5 action)
    extra["nestor"] = {"run": bool(nestor_ok), "g0_1": g0["nestor_g0_1"], "g0_5": g0["nestor_g0_5"], "probe_config": "nestor" in B.probe}
    if nestor_ok:
        if not repeat:
            for s in sd["full_nestor"]:
                B.train(f"{stage}_nestor_full_german_s{s}", base="nestor", role="full", data=cfg["train_data"]["german"],
                        seed=int(s), steps=steps, spec=None, eval_sets_=["dclm", "german"],
                        meta={"target": "german", "adapt_seed": int(s), "h100_duplicate": int(s) == int(sd["h100_duplicate_seed"])})
        k_n = int(g0["k"].get("nestor", 16))
        an, tn = int(cfg["nestor_aa"]["pair"]), cfg["nestor_aa"]["target"]
        en = aa_draw(B.base_dir("nestor"), "priv", an, tn, int(sd["aa_generator"]["nestor"]), k_n, out_dir)
        nid = f"{stage}_nestor_aa{an}_R1"
        B.train(nid, base="nestor", role="aa_nestor", data=cfg["train_data"][tn], seed=int(sd["aa_data"]["nestor"]) + an,
                steps=steps, spec=en["spec_A"], eval_sets_=["dclm", tn, code, code2, "math"], reflect_sets=[code, "math"],
                interim_sets=[tn], meta={"pair": an, "subset": "R1", "draw_subset": "A", "target": tn, "k": k_n,
                                         "generator_seed": en["seed"], "data_seed": int(sd["aa_data"]["nestor"]) + an,
                                         "draw_file": _fwd(en["_file"])})
        extra["nestor_aa"] = {"pair": an, "run": nid, "target": tn, "generator_seed": en["seed"],
                              "data_seed": int(sd["aa_data"]["nestor"]) + an, "draw_file": _fwd(en["_file"])}
    else:
        B.warnings.append("NESTOR runs omitted: G0-5 NESTOR status or probe config missing")
    extra["base_eval_files"] = {b: {kk: _fwd(v) for kk, v in f.items()} for b, f in base_files.items()}
    extra["h100_duplicate_file"] = cfg["h100_duplicate_file"]
    return B.write(extra)


def build_g2(cfg: dict, g0rec: str, g1rec_path: str, out_dir: str) -> dict:
    B = Builder("g2", cfg, g0rec, out_dir)
    g0, sd, r = B.g0, cfg["seeds"], cfg["recipe"]
    g1 = json.load(open(g1rec_path, encoding="utf-8"))
    d = g1.get("decision", {})
    if d.get("outcome") != "PROCEED":
        raise SystemExit(f"G1 outcome {d.get('outcome')!r}: no G2 (spec order: G2 only after G1 PROCEED)")
    g1plan = json.load(open(g1["inputs"]["plan"]["path"], encoding="utf-8"))
    rp = (g1["inputs"].get("repeat_plan") or {}).get("path")
    s1plan = json.load(open(rp, encoding="utf-8")) if (d.get("g1_1_lengthened") and rp) else g1plan
    steps = int(d["steps"])
    q = float(d["q"]) if d.get("q") is not None else None             # None under C1 PIVOT (C2 gates only)
    s1pairs = {int(p["pair"]): p for p in s1plan["aa_pairs"]}
    N = int(s1pairs[1].get("N", 24))
    k = int(round(q * N)) if q is not None else None
    code = g0["code_stream"]
    extra: dict = {"g1_record": {"path": os.path.abspath(g1rec_path), "sha256": file_sha256(g1rec_path),
                                 "content_sha256": g1.get("sha256")},
                   "g1_plan": {"path": g1["inputs"]["plan"]["path"], "sha256": g1plan.get("sha256")},
                   "s1_plan": {"path": rp if s1plan is not g1plan else g1["inputs"]["plan"]["path"], "sha256": s1plan.get("sha256")},
                   "q": q, "k": k, "steps": steps, "c1_status_g1": d.get("c1"),
                   "spec_runs": "g1g2_spec.json 1.3.0 runs.G2"}
    base_files = {b: dict(f) for b, f in g1plan["base_eval_files"].items()}
    pairs = []
    if not d.get("pivot"):
        tg = {int(a): t for a, t in cfg["aa_pairs"]["targets"].items()}
        for a in cfg["aa_pairs"]["g2"]:
            pairs.append(_aa_runs(B, "g2", int(a), tg[int(a)], k, steps, out_dir))
        if d.get("repeat_pair2_in_g2"):
            # G1-2 changed q or steps: pair a = 2 repeated in the G1-2 configuration (same seeds; k from the new q)
            pairs.append(_aa_runs(B, "g2", 2, tg[2], k, steps, out_dir, suffix="_cfg"))
    else:
        B.warnings.append("C1 PIVOT at G1: books A/A pairs (G2-4, G2-1) not run; G2 covers the C2 gates only")
    extra["aa_pairs"] = pairs
    # SGD calibration on R1 of pair a = 1 (C2 recipe: WD 0, no clipping, Nesterov 0.95; lr c/3, c, 3c),
    # full DCLM and German evaluations at steps 0, 8, 24, 48 (step 0 = in-process theta_0, paired with the later steps)
    r1 = s1plan["r1"]
    r1_spec = json.load(open(r1["draw_file"], encoding="utf-8"))[f"spec_{r1['subset']}"]
    theta0 = g1plan.get("theta0_checks") or _fwd(os.path.join(os.path.dirname(g1["inputs"]["plan"]["path"]), "theta0_checks.json"))
    sgd_runs = []
    for i in range(3):
        rid = f"g2_sgd_lr{i + 1}"
        B.train(rid, base="moe_b1", role="sgd", data=cfg["train_data"]["german"], seed=int(r1["data_seed"]), steps=steps,
                spec=r1_spec, eval_sets_=[], interim_sets=["dclm", "german"], full_steps=list(cfg["sgd_full_steps"]),
                full_sets=["dclm", "german"],
                env_extra={"NESTOR_SUBSET_OPTIMIZER": "sgd", "NESTOR_SGD_LR": f"{{sgd_lr{i}}}",
                           "NESTOR_SGD_MOMENTUM": str(r["momentum"]), "NESTOR_SGD_NESTEROV": "1", "NESTOR_SGD_WD": "0",
                           "DCLM_WEIGHT_DECAY": "0", "NESTOR_MAX_GRAD_NORM": "none"},
                meta={"lr_index": i, "lr_factor": [1 / 3, 1.0, 3.0][i], "target": "german", "subset": "R1 of pair a = 1"})
        sgd_runs.append(rid)
    extra["file_vars"] = {f"sgd_lr{i}": f"{theta0}#sgd.grid.{i}" for i in range(3)}
    extra["sgd"] = {"runs": sgd_runs, "theta0_checks": theta0, "muon_r1_run": r1["run"], "grid_rule": "{c/3, c, 3c}",
                    "full_steps": list(cfg["sgd_full_steps"])}
    snr_out = _fwd(os.path.join(out_dir, "grad_snr_moe_b1.json"))
    B.cmd("g2_snr_moe_b1", ["{python}", "grad_snr.py", "run", cfg["experiments"]["moe_b1"], "{ckpt_moe_b1}",
                            "--data", cfg["train_data"]["german"], "--windows", str(cfg["snr"]["windows"]),
                            "--select", "random", "--data-seed", str(int(r1["data_seed"])), "--seed", "0",
                            "--train-batch-windows", str(cfg["snr"]["train_batch_windows"]), "--kinds", cfg["snr"]["kinds"],
                            "--out", snr_out], 0.03, out=snr_out)
    extra["snr"] = {"out": snr_out}
    extra["g07r_record"] = (cfg.get("gate_config") or {}).get("g07r_record")
    extra["base_eval_files"] = {b: {kk: _fwd(vv) for kk, vv in f.items()} for b, f in base_files.items()}
    return B.write(extra)


# ============================================================================= CLI
def check_paths(plan: dict) -> list[str]:
    """Every file a queue will need: pinned checkpoints, training streams, evaluation bins (fail before any GPU work)."""

    need = set()
    q = json.load(open(plan["queue"], encoding="utf-8"))
    need.update(v for k, v in q["vars"].items() if k.startswith("ckpt_"))
    for r in plan["runs"]:
        need.add(r["data"])
    for sets in plan["sets"].values():
        for s in sets.values():
            need.add(s["path"])
    return sorted(p for p in need if p and not os.path.exists(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G1/G2 run manifests (run_queue queues + hashed plans)")
    ap.add_argument("stage", choices=["g1", "g1r", "g2"])
    ap.add_argument("--g0-record", required=True)
    ap.add_argument("--g1-record", default=None)
    ap.add_argument("--g1-dir", default="g1", help="g1r: directory of the G1 plan whose theta_0 evaluations are reused")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--no-check", action="store_true", help="skip the input-file existence check (mock plans)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.print_config:
        print(canonical(cfg))
        return 0
    if a.stage == "g1":
        plan = build_g1(cfg, a.g0_record, a.out_dir)
    elif a.stage == "g1r":
        plan = build_g1(cfg, a.g0_record, a.out_dir, stage="g1r", steps=192, repeat=True, reuse_base_from=a.g1_dir)
    else:
        if not a.g1_record:
            ap.error("g2 needs --g1-record")
        plan = build_g2(cfg, a.g0_record, a.g1_record, a.out_dir)
    rows = [f"{e['id']:<40} {e['kind']:<6} {e['total_h']:.3f} h" for e in plan["estimates"]]
    print("\n".join(rows))
    print(f"PLAN {plan['kind']}: {len(plan['runs'])} train runs, {len(plan['estimates'])} jobs, "
          f"estimate {plan['estimate_total_gpu_h']:.2f} GPU-h, sha256 {plan['sha256']}")
    for w in plan["warnings"]:
        print("WARNING", w)
    if not a.no_check:
        missing = check_paths(plan)
        if missing:
            print(f"MISSING INPUTS ({len(missing)}): " + "; ".join(missing[:20]))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
