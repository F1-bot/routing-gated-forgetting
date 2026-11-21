"""Confirmatory + secondary queue builder (the design items 8-9; prereg v1.2 sec. 4.6-4.8 as amended by the
G0 outcome and the design). Torch-free (numpy). One run_queue queue + one hashed plan for every block after the freeze.

  python confirm_plan.py build --g0-record g0/gate_record.json --g1-record g1/gate_record_G1.json \
         --g2-record g2/gate_record_G2.json [--config g1g2_config.json] [--confirm-config confirm_config.json] \
         [--pins ckpt_pins.json] [--b3-record g0/gate_record_b3.json] [--nestor-books results/nestor_books_feasibility.json] \
         [--throughput results/throughput_measured.json] [--states results/queue_state_g1.json,...] --out-dir confirm
  python confirm_plan.py throughput --plans g1/plan_g1.json,g2/plan_g2.json --states results/queue_state_g1.json,... \
         --out results/throughput_measured.json
Outputs: <out>/plan_confirm.json (+ .sha256), <out>/queue_confirm.json, <out>/specs/*.json (hashed draw manifests),
<out>/estimates_confirm.csv. The plan is deterministic (canonical JSON, sha256 without the creation time): rebuilding
from the same inputs reproduces it, which the freeze record (FREEZE.json, chain6b_util.py freeze-write) relies on.

Blocks (status label; registered source):
  C1      confirmatory  MoE bases 1 and 2, the C1 targets of the G2 freeze block (books only after G0: German infeasible,
                        its cells replaced by books cells, draws per base unchanged = c1_draws_per_target), generator 5000+d,
                        data 7000+d (base 1) / 7020+d (base 2; MF-3); evaluations DCLM, target, code (both
                        streams), math; interim (steps 0/8/24/48, 64-window
                        heads); reflected theta0-delta on code, math, DCLM; pinned + selection-pinned replay at theta0 +- delta
                        on code and math; antithetic random directions (Haar 9500 + run index) on code,
                        math, DCLM and WD-only on code, math for the base-1 draws d <= 6 (prereg: "n_d <= 3", 12 runs);
                        run order per (base, target) cell randomized with seed 9000 + cell index (prereg sec. 4.8).
  C2      confirmatory-conditional (G2 freeze c2_status): 2 bases x 4 draws (generator 6000+d: base 1 d=1..4, base 2
                        d=5..8; 4-draw exploratory block d=1,2 / 5,6), arms {SGD (G2-2 lr), Muon} x {H = B, L = A}, WD 0,
                        no clipping, data 8000+d; full DCLM + target grids at steps 8/24/48 (NESTOR_INTERIM_FULL), final 96.
  C3      estimation    FULL x 4 domains (books, math, code, German): NESTOR seeds 101/102, MoE base 1 seeds 101/102,
                        MoE base 2 seed 101 (G1 FULL runs reused); DCLM-CONT (NESTOR, MoE base 2; held-out shard 03);
                        reverts on DCLM (MoE 3, NESTOR 6 groups) for every FULL run incl. the G1 ones; WD-only 3 per arch;
                        training-time freeze run per architecture (German, seed 103). Random partial-revert control: NOT
                        built (no tooling; flagged).
  C5      estimation    French on base 1: 8-step E-ALL drive -> hashed forecasts (forecasts.py c5-french, gate) -> FULL
                        French (seed 101) + 2 crossover pairs (generator 6800+d, data 8800+d).
  dose    secondary     2 reduced-tilt books draws per MoE base 1/2 (dose_draws.py; generator 5100+d, data seeds as recorded:
                        7100+d base 1, 7150+d base 2; accepted on the selection-half P_d, R2).
  M12b    secondary     the base-1 books-cell A_d (d=1..3) trained on German (data 7500+d); antithetic random directions
                        with the SAME Haar seed as the C1 books A_d run.
  b3      secondary     C1-books replication on the third base (n_d = 3: d=1..3, 6 runs; data 7040+d); needs
                        g0/gate_record_b3.json.
  nestor_books exploratory  3 draws x 2 NESTOR runs if results/nestor_books_feasibility.json is feasible; first-order-free
                        statistics (reflected, pinned/selpin, antithetic random).
  audit   secondary     assumption audit at theta_0 on MoE bases 1 and 2 (assumption_audit.py; eval-only).
  SEC     secondary     DCLM-tilt infeasible at G0-4 and D4 (embedding-row training) not implemented -> no runs (flagged).
  C1-DE   exploratory   German C1 draws by exact MILP (prereg v1.3 sec. 4.12; c1de_milp.py at stage C, manifests
                        results/c1de, re-verified here on the plan's tables): d = 1-3 per MoE base, objective seed 5600+d,
                        data 7600+d (base 1) / 7620+d (base 2), 2 subset runs per draw on German with the third-base
                        evaluation set (antithetic random
                        and WD-only on side A); its 12 theta0 probe units are costed in the block; cut after dose-response
                        and before the second code stream (2026-11-14). It never replaces books in C1.
Cuts (P-5): pass 1 (confirm/pass1) decides the cuts with the restoration step; pass 2 rebuilds without the cut blocks (and
without the second code stream if cut), so the queue, the manifests and the forecast units hold only what runs.
 (2026-11-14): no two draws treated as independent share a data seed (check_data_seeds, recorded as
plan.data_seeds); queue order C1, C2, C3, C5, third base, assumption audit, then the cuttable blocks, each behind an
overrun guard (run_queue python gate confirm_plan:overrun_guard): measured spent + remaining planned hours against the
envelope; above it the registered cut order is applied and a block in the cut set is skipped with a log line.
Training jobs (2026-11-14): every job starts at its architecture's measured configuration (confirm config
"train"): MoE micro 2 x accum 32 without activation checkpointing (G1, stage B), NESTOR micro 4 x accum 16 with it (stage
B; in G1 NESTOR at micro 2 without checkpointing ran out of memory). run_queue.py's fallback ladder (checkpointing on,
then half the micro-batch) stays as the safety net. Training is priced with the measured step time of that configuration
plus the g1g2 compile and model-load minutes (Cost; a --throughput file still takes precedence).
"""
from __future__ import annotations

import argparse
import copy
import csv
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

import crossover as CX  # noqa: E402
import g1g2_plan as GP  # noqa: E402

TPS = GP.TOKENS_PER_STEP
SEQ = GP.SEQ
EVAL_TOK = GP.EVAL_TOKENS
HEAD_TOK = 64 * SEQ                                    # interim sets: the first 64 windows of each evaluation grid
PIN_VARIANTS = "pinned,selpin,reflect+pinned,reflect+selpin"
PIN_PASSES = GP.PIN_PASSES                             # 1 theta_0 recording + 4 replays per window
# P-16 (2026-11-15, core-first schedule): each C1 / third-base run evaluates pinned and reflect+pinned with the run
# (core); selpin and reflect+selpin run later as eval jobs of the 'selpin' block on the run's final checkpoint
CORE_PIN_VARIANTS = "pinned,reflect+pinned"
SELPIN_VARIANTS = "selpin,reflect+selpin"

CONFIRM_DEFAULTS = {
    "blocks": {"C1": True, "C2": True, "C3": True, "C5": True, "dose": True, "M12b": True, "b3": True,
               "nestor_books": True, "audit": True, "SEC": True, "C1-DE": True},
    "second_code_stream": True,
    "c1": {"registered_cells": [["moe_b1", "german"], ["moe_b1", "books"], ["moe_b2", "german"], ["moe_b2", "books"]],   # cell index -> order seed 9000 + index
           "order_seed_base": 9000, "controls_base": "moe_b1", "controls_max_d": 6},
    "seeds": {"c1_data": 7000, "m12b_data": 7500, "c2_data": 8000, "c5_data": 8800, "dose_data": 7100, "c1de_data": 7600, "c1de_objective": 5600, "haar": 9500,
              "full": [101, 102], "full_b2": [101], "freeze": 103, "c5_full": 101, "dclm_cont": 101,
              # MF-3 (decision (d); prereg v1.3 version 7 sec. 3.1 'Seeds'): base-specific data seeds of C1
              # (7000 + d / 7020 + d), the third base (7040 + d) and C1-DE (7600 + d / 7620 + d), so that no two draws share data
              "base_offset": {"moe_b1": 0, "moe_b2": 20, "moe_b3": 40}},
    "c2": {"draws": {"moe_b1": [1, 2, 3, 4], "moe_b2": [5, 6, 7, 8]},
           "exploratory_draws": {"moe_b1": [1, 2], "moe_b2": [5, 6]}, "target": "german", "full_steps": [8, 24, 48],
           # P-16: the runs keep the registered 96-step schedule and stop right after the step-8 full evaluation (s* = 8);
           # MF-P1 fix (A) (2026-11-15 ~04:00): the SGD runs continue to step 24 (interim heads 0/8/24) and stop there
           "stop_after": 8, "stop_after_sgd": 24},
    "c3": {"domains": ["books", "math", "code", "german"], "wd_only_domains": ["german", "math", "books"],
           "freeze_target": "german", "per_layer": True},
    "c5": {"target": "french", "pairs": [1, 2], "drive_steps": 8},
    "m12b": {"draws": [1, 2, 3], "target": "german"},
    "b3": {"draws": [1, 2, 3]},
    "nestor_books": {"draws": [1, 2, 3]},
    "c1de": {"bases": ["moe_b1", "moe_b2"], "draws": [1, 2, 3], "target": "german", "f": 0.5,
             "probe_unit": {"directions": 2, "corpora": 3, "windows": 8, "grad_s": 3.0}},   # budget_v13 PROBE_UNIT_MOE_H
    "audit": {"bases": ["moe_b1", "moe_b2"], "windows": 24, "probe_s_per_direction_corpus": 24.66},
    # (2026-11-14): start configuration of every training job and its measured step time (s per optimizer
    # step of 262,144 tokens: median of the per-run medians of the logged train sec_per_step; 96-step runs, no OOM)
    "train": {"moe": {"micro": 2, "ckpt": False, "s_per_step": 8.27,
                      "measured": "G1 logs/queue_g1/g1_moe_b1_*_m2.log (11 runs, 7.79-9.17) + stage B "
                                  "logs/cont_moe_100m_adapt_*_m2.log (4 runs, 8.23-8.80)"},
              "nestor": {"micro": 4, "ckpt": True, "s_per_step": 25.64,
                         "measured": "stage B logs/cont_nestor_100m_adapt_*_m4c.log (4 runs, 24.93-26.25); G1 at micro 2: "
                                     "OOM without checkpointing, 43.98 s/step with it (g1_nestor_full_german_s101)"}},
    "budget": {"envelope_h": 120.0, "contingency": 0.15,
               "cut_order": ["SEC", "contingency 15% -> 10%", "M12b", "nestor_books", "dose", "C1-DE", "second code stream"]},
    # P-16 revised (2026-11-15 ~03:00): the core is C1 on both MoE bases (with its registered controls), C2 (stopped
    # after step 8) and the theta_0 evaluations C1/C2 need; then the core marker, the third base and the assumption audit
    # (never cut; analysed by the full analysis), then the remaining blocks in the registered order under the overrun rule
    "order": ["C1:moe_b1", "C1:moe_b2", "C2", "CORE", "b3", "audit", "selpin", "C3", "C5", "DCLM-CONT", "SEC", "dose",
              "M12b", "nestor_books", "C1-DE"],
    "core": {"blocks": ["C1:moe_b1", "C1:moe_b2", "C2"], "marker": "results/confirm/CORE_DONE.json"},
    "selpin_block": True,
    "overrun_guard": True,                               # R10: run_queue gate before each cuttable block (overrun_guard)
    "revert_groups": {"moe": ["moe", "kind:attn+embed_head+kind:norm", "embed_head"],
                      "nestor": ["shared+pub+embed_head", "attn+priv", "attn", "priv", "pub", "embed_head"]},
}


def c3_revert_tokens(strata: list[str], per_block: bool) -> list[str]:
    """Scope tokens of the C3 per-layer reverts (prereg v1.3 sec. 4.8) from routed strata '<kind>:b<block>:l<layer>'.

    per_block=False: one token 'b<b>l<l>' per stratum (MoE single-layer reverts, weight_controls scope b<i>l<j>);
    per_block=True: one token 'b<b>' per block (NESTOR single-block reverts of the column-private group).
    """
    pairs = sorted({(int(k.split(":")[1][1:]), int(k.split(":")[2][1:])) for k in strata})
    if per_block:
        return [f"b{b}" for b in sorted({b for b, _ in pairs})]
    return [f"b{b}l{l}" for b, l in pairs]


def c3_layer_variant(group: str, tok: str) -> str:
    """eval_windows variant of one C3 per-layer / per-block revert: every '+'-part of the group restricted to the scope
    token, e.g. ('attn+priv', 'b3') -> 'revert:attn@b3+priv@b3' (weight_controls.revert_state reverts the union of the
    parts). 'revert:attn+priv@b3' would be WRONG: eval_windows splits on '+' first, which leaves 'attn' unscoped."""
    return "revert:" + "+".join(f"{g}@{tok}" for g in group.split("+"))


def merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a


def _rel(p: str) -> str:
    return p.replace("\\", "/")


def read_manifest_verified(path: str, expect_sha: str | None = None) -> dict | None:
    """A crossover.write_manifest file whose sha256 equals its .sha256 sidecar (and the recorded value, if given)."""
    if not path or not os.path.exists(path):
        return None
    h = GP.file_sha256(path)
    side = path + ".sha256"
    if os.path.exists(side) and open(side, encoding="utf-8").read().split()[0] != h:
        return None
    if expect_sha and expect_sha != h:
        return None
    return json.load(open(path, encoding="utf-8"))


# ============================================================================= data seeds (MF-3) and the overrun guard (R10)
DRAW_ROLES = {"c1": "C1", "c1_b3": "third base", "c1de": "C1-DE", "dose": "dose", "m12b": "M12-b", "c2_sgd": "C2",
              "c2_muon": "C2", "c5_pair": "C5"}


def data_seed_units(runs: list[dict]) -> dict:
    """Data seed -> the draw units (block, base, d) using it, over the blocks whose draws are treated as independent (C1
    per base and pooled, third base, C1-DE, dose-response, M12-b, C2, C5 pairs). The runs of one draw (sides A and B, the
    four C2 arms) share its seed by design; FULL, freeze, DCLM-CONT and drive runs are not draws."""
    units: dict = {}
    for r in runs:
        blk = DRAW_ROLES.get(r.get("role"))
        if blk is None or r.get("d") is None or r.get("seed") is None:
            continue
        units.setdefault(int(r["seed"]), set()).add((blk, str(r.get("base")), int(r["d"])))
    return {s: sorted(u) for s, u in sorted(units.items())}


def check_data_seeds(runs: list[dict]) -> dict:
    """MF-3 / decision (d): no two draws treated as independent share a data seed (SystemExit otherwise)."""
    units = data_seed_units(runs)
    bad = {s: u for s, u in units.items() if len(u) > 1}
    if bad:
        raise SystemExit("data seed shared by draws treated as independent (MF-3): " + "; ".join(
            f"{s}: " + ", ".join(f"{b} {base} d{d}" for b, base, d in u) for s, u in bad.items()))
    return {"rule": "one data seed per draw unit (block, base, d) over " + ", ".join(sorted(set(DRAW_ROLES.values()))) +
                    " (MF-3; decision (d))",
            "draw_units": {str(s): list(u[0]) for s, u in units.items()}}


def overrun_guard(state: dict, params: dict) -> dict:
    """R10 (registered overrun rule, prereg v1.3 version 7 sec. 7): run_queue python gate before each cuttable
    block. Measured spent hours (the -SpentGpuH hours of the plan + the hours of every finished job of this queue) plus
    the planned hours of every job not yet finished are compared with the envelope (+ the P-6 extension recorded in
    FREEZE.json). Above it, the registered cut order is applied to the cuttable blocks that still have jobs; the gate
    fails -- run_queue then skips the block's jobs and logs it -- only if its own block is in that cut set."""
    plan = json.load(open(params["plan"], encoding="utf-8"))
    q = json.load(open(params["queue"], encoding="utf-8"))
    env_h = float(params["envelope_h"])
    fz = params.get("freeze") or "FREEZE.json"
    if os.path.exists(fz):
        try:
            prm = json.load(open(fz, encoding="utf-8")).get("parameters") or {}
            env_h += float(((prm.get("envelope extension (decision P-6; 0 if none)") or {}).get("value") or {}).get("gpu_h") or 0.0)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    fin = ("done", "pass", "fail", "failed", "skipped")
    stat = {j["id"]: (state.get(j["id"]) or {}) for j in q["jobs"]}
    spent = float(params.get("spent_before_h") or 0.0) + sum(float(s.get("hours") or 0.0) for s in stat.values()
                                                             if s.get("status") in fin)
    remaining = {j["id"]: float(j.get("est_h") or 0.0) for j in q["jobs"] if stat[j["id"]].get("status") not in fin}
    blocks = {k: set(v) for k, v in (plan.get("blocks") or {}).items() if not k.startswith("_")}

    def rem(b: str) -> float:
        return sum(h for jid, h in remaining.items() if jid in blocks.get(b, ()))

    projected = spent + sum(remaining.values())
    cut, cur = [], projected
    for b in params["cut_order"]:
        if cur <= env_h:
            break
        if rem(b) > 0:
            cut.append(b)
            cur -= rem(b)
    me = params["block"]
    ok = me not in cut
    print(f"[overrun-guard] {me}: spent {spent:.2f} + remaining {sum(remaining.values()):.2f} = {projected:.2f} GPU-h, "
          f"envelope {env_h:.1f}" + (f"; cut order -> {cut}" if cut else "") + ("; run" if ok else "; SKIP (registered overrun rule)"),
          flush=True)
    return {"pass": ok, "block": me, "spent_gpu_h": round(spent, 4), "remaining_planned_gpu_h": round(sum(remaining.values()), 4),
            "projected_gpu_h": round(projected, 4), "envelope_gpu_h": env_h, "cut_set": cut, "after_cuts_gpu_h": round(cur, 4)}


def core_marker(state: dict, params: dict) -> dict:
    """P-16 (2026-11-15): run_queue python gate right after the last core job. Writes params['out']
    (results/confirm/CORE_DONE.json): UTC time, plan and queue sha256, and for every core job its status, hours and the
    sha256 of its queue-state entry (canonical JSON). Never overwrites an existing marker. The core analysis
    (analyze_confirm.py run --mode core) refuses to run without it. Always passes: the remaining blocks continue."""
    import hashlib

    def _sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None

    out = params["out"]
    if os.path.exists(out):
        print(f"[core-marker] {out} exists: kept (never overwritten)", flush=True)
        return {"pass": True, "marker": out, "kept_existing": True}
    rows = []
    for jid in params["core_jobs"]:
        s = state.get(jid) or {}
        blob = json.dumps(s, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        rows.append({"id": jid, "status": s.get("status", "missing"), "hours": s.get("hours"),
                     "state_sha256": hashlib.sha256(blob).hexdigest()})
    ok = {"done", "pass"}
    doc = {"what": "P-16 core marker: every core job of the confirm queue has run (in queue order)",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "plan": {"path": params.get("plan"), "sha256": _sha(params.get("plan") or "")},
           "queue": {"path": params.get("queue"), "sha256": _sha(params.get("queue") or "")},
           "core_blocks": params.get("core_blocks"), "n_jobs": len(rows),
           "n_done": sum(1 for r in rows if r["status"] in ok), "not_done": [r["id"] for r in rows if r["status"] not in ok],
           "jobs": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, out)
    print(f"[core-marker] {out}: {doc['n_done']}/{doc['n_jobs']} core jobs done; not done: {doc['not_done']}", flush=True)
    return {"pass": True, "marker": out, "n_done": doc["n_done"], "not_done": doc["not_done"]}


# ============================================================================= cost model
class Cost:
    """Per-job GPU-h. Training: measured per-architecture throughput (confirm_plan.py throughput) when given; else the
    measured step time of the architecture's start configuration (confirm config "train") plus the g1g2
    compile and model-load minutes; else the g1g2 rates (MoE 34k / NESTOR 10.2k tok/s + compile). Evaluation:
    measured when given, else the fp32 rates in the g1g2 config (27k / 8.5k tok/s)."""

    def __init__(self, cfg: dict, measured: dict | None = None, train: dict | None = None):
        self.tp = cfg["throughput"]
        self.meas = measured or {}
        self.step_s = {a: float(v["s_per_step"]) for a, v in (train or {}).items() if (v or {}).get("s_per_step")}

    def train(self, arch: str, steps: int) -> float:
        m = (self.meas.get(arch) or {}).get("train_h_per_step")
        if m:
            return steps * m + (self.meas[arch].get("fixed_h") or 0.0)
        if arch in self.step_s:
            return steps * self.step_s[arch] / 3600 + self.tp["compile_min"][arch] / 60 + self.tp["model_load_min"] / 60
        return steps * TPS / self.tp["train_tok_s"][arch] / 3600 + self.tp["compile_min"][arch] / 60 + self.tp["model_load_min"] / 60

    def eval_tokens(self, arch: str, tokens: int) -> float:
        m = (self.meas.get(arch) or {}).get("eval_h_per_token")
        return tokens * (m if m else 1.0 / self.tp["eval_tok_s"][arch] / 3600)

    def source(self) -> str:
        if self.meas:
            return "measured (G1/G2 queue states)"
        if self.step_s:
            return ("measured step times of the start configuration (confirm config 'train': G1 queue + stage-B logs) + "
                    "g1g2 compile/model load; evaluation: throughput (g1g2 config)")
        return "throughput (g1g2 config)"


def n_passes(variants: str) -> int:
    vs = [v for v in variants.split(",") if v]
    joint = [v for v in vs if v in PIN_VARIANTS.split(",")]
    rest = [v for v in vs if v not in joint]
    return len(rest) + ((1 + len(joint)) if joint else 0)    # one theta_0 recording + one replay per pin variant


# ============================================================================= builder
class ConfirmBuilder(GP.Builder):
    def __init__(self, cfg: dict, ccfg: dict, g0rec: str, out_dir: str, *, pins: dict | None, cost: Cost):
        super().__init__("confirm", cfg, g0rec, out_dir)
        self.cc, self.cost, self.blocks, self.run_index = ccfg, cost, {}, 0
        self.pins = pins or {}
        for b in ("moe_b1", "moe_b2", "moe_b3", "nestor"):
            e = (self.pins.get("pins") or {}).get(b) or {}
            if e.get("path") and f"ckpt_{b}" not in self.vars:
                self.vars[f"ckpt_{b}"] = _rel(e["path"])
        self.base_files: dict[str, dict] = {}
        self.tables: dict[str, CX.RegTables] = {}
        self.deferred_selpin: list[tuple[str, str, list[str]]] = []
        os.makedirs(os.path.join(out_dir, "specs"), exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def table(self, base: str, g0_dir: str | None = None) -> CX.RegTables:
        if base not in self.tables:
            d = g0_dir or self.base_dir(base)
            fam = "priv" if base == "nestor" else "moe"
            lay = ((self.g0rec.get("config") or {}).get("resolved") or {}).get("layout", "auto")
            hs = int(((self.g0rec.get("config") or {}).get("resolved") or {}).get("half_seed", 0))
            self.tables[base] = CX.reg_load_dir(d, fam, layout=lay, half_seed=hs)
        return self.tables[base]

    def manifest(self, name: str, entries: list[dict], extra: dict) -> dict:
        p = os.path.join(self.out, "specs", f"{name}.json")
        h = CX.write_manifest(p, entries, extra=extra)
        return {"path": _rel(p), "sha256": h}

    def subset_meta(self, t: CX.RegTables, S: np.ndarray, code: str) -> dict:
        S = np.asarray(S)
        m = np.zeros((t.L, t.N), bool)
        m[np.arange(t.L)[:, None], S] = True
        ro = {lab: float(t.est[c][m].sum() / t.est[c].sum()) for lab, c in (("dclm", "dclm"), ("code", code), ("math", "math"))
              if c in t.est}
        return {"n_units": int(m.sum()), "n_units_family": int(t.L * t.N), "ro": ro}

    def need_base_eval(self, base: str, names: list[str]) -> dict:
        """theta_0 evaluation files for the sets a base's runs use (one eval job per base, union of sets)."""
        have = self.base_files.setdefault(base, {})
        self._want = getattr(self, "_want", {})
        self._want.setdefault(base, [])
        for n in names:
            if n not in have and n not in self._want[base]:
                self._want[base].append(n)
        return have

    def flush_base_evals(self, reuse: dict) -> None:
        for base, names in sorted(getattr(self, "_want", {}).items()):
            todo = [n for n in names if n not in (reuse.get(base) or {})]
            for n, p in (reuse.get(base) or {}).items():
                self.base_files.setdefault(base, {})[n] = p
            if not todo:
                continue
            sets = self.sets[base if base in self.sets else "moe_b1"]
            groups: dict[str, list[str]] = {}
            for n in todo:
                groups.setdefault(sets[n].get("select", "even"), []).append(n)
            for sel, ns in sorted(groups.items()):
                jid = f"confirm_base_{base}" + ("" if len(groups) == 1 else f"_{sel}")
                files = self.base_eval(base, ns, jid)
                self.estimates[-1]["total_h"] = round(self.cost.eval_tokens(self.arch(base), EVAL_TOK * len(ns))
                                                      + self.cfg["throughput"]["model_load_min"] / 60, 4)
                self.jobs[-1]["est_h"] = self.estimates[-1]["total_h"]
                self.estimates[-1]["block"] = "base_eval"
                self.base_files[base].update(files)

    def defer_selpin(self, run: dict, names: list[str]) -> None:
        """P-16: selpin and reflect+selpin of a C1 / third-base run are evaluated later (block 'selpin', after the core) on
        the run's final checkpoint, by an eval job that writes <run root>/eval_selpin (the analysis finds it through the
        run record's 'selpin' eval)."""
        if not self.cc.get("selpin_block"):
            return
        run["evals"].append({"name": "selpin", "sets": list(names), "variants": SELPIN_VARIANTS,
                             "deferred_job": f"selpin_{run['id']}"})
        self.deferred_selpin.append((run["id"], run["base"], list(names)))

    def flush_selpin(self) -> None:
        for rid, base, names in self.deferred_selpin:
            self.eval_job(f"selpin_{rid}", "selpin", base, f"{{final_{rid}}}", names, SELPIN_VARIANTS,
                          f"{{root_{rid}}}/eval_selpin")
            self.jobs[-1]["requires"] = [rid]
        self.deferred_selpin = []

    def trun(self, rid: str, *, block: str, base: str, role: str, data: str, seed: int, steps: int, spec: dict | None,
             evals: list[dict], interim_sets: list[str] | None = None, interim_full: dict | None = None,
             env_extra: dict | None = None, meta: dict | None = None, interim_steps: list[int] | None = None,
             stop_after: int | None = None) -> dict:
        r = self.cfg["recipe"]
        arch = self.arch(base)
        sets = self.sets[base if base in self.sets else "moe_b1"]
        warm = int(round(r["warmup_96"] * steps / 96))
        st = self.cc["train"][arch]                          # measured start configuration (fallbacks: run_queue)
        micro, ckpt = int(st["micro"]), bool(st["ckpt"])
        if TPS % (micro * SEQ):
            raise SystemExit(f"{rid}: tokens_per_step {TPS} is not a multiple of micro {micro} x seq {SEQ}")
        self.run_index += 1
        env = {"DCLM_DATA_PATH": data, "DCLM_INIT_CHECKPOINT": f"{{ckpt_{base}}}", "DCLM_LR_MUON": str(r["lr_muon"]),
               "DCLM_LR_ADAMW": str(r["lr_adamw"]), "DCLM_WEIGHT_DECAY": str(r["weight_decay"]),
               "NESTOR_FREEZE_ROUTERS": "1", "DCLM_RUN_SUFFIX": rid, "DCLM_SEED": str(seed)}
        env.update(env_extra or {})
        job = {"id": rid, "kind": "train", "experiment": self.cfg["experiments"].get(base, "moe_100m"), "steps": steps,
               "warmup": warm, "seq": SEQ, "tokens_per_step": TPS, "micro": micro, "ckpt": ckpt,
               "out_root": _rel(os.path.join("outputs", "confirm", rid)), "env": env}
        if spec is not None:
            job["spec"] = spec
        h_int = 0.0
        isteps = list(self.cfg["interim_steps"] if interim_steps is None else interim_steps)
        if interim_sets:
            job["interim"] = GP.interim_cfg(sets, interim_sets, isteps)
            h_int += self.cost.eval_tokens(arch, HEAD_TOK * len(interim_sets) * len(isteps))
            if stop_after:                                 # P-16: read by adapt_extras (NESTOR_INTERIM_EVAL)
                job["interim"]["stop_after"] = int(stop_after)
            if interim_full:
                job["interim"]["full_steps"] = list(interim_full["steps"])
                job["interim"]["full_sets"] = list(interim_full["sets"])
                h_int += self.cost.eval_tokens(arch, EVAL_TOK * len(interim_full["sets"]) * len(interim_full["steps"]))
        evs, h_ev, used = [], 0.0, set(interim_sets or [])
        for ev in evals:
            a, sp, sel = GP.set_args(sets, ev["sets"])
            e = {"name": ev["name"], "sets": a, "spans": sp, "select": sel, "variants": ev["variants"]}
            if ev.get("base", ev["variants"] != "ckpt"):
                e["base"] = f"{{ckpt_{base}}}"
            evs.append(e)
            used |= set(ev["sets"])
            h_ev += self.cost.eval_tokens(arch, EVAL_TOK * len(ev["sets"]) * n_passes(ev["variants"])) \
                + self.cfg["throughput"]["model_load_min"] / 60
        if evs:
            job["evals"] = evs
        h_tr = self.cost.train(arch, int(stop_after) if stop_after else steps)   # P-16: optimizer steps actually run
        h_aud = (self.cfg["throughput"]["audit_min"] / 60) if spec is not None else 0.0
        est = round(h_tr + h_int + h_ev + h_aud, 4)
        job["est_h"] = est
        self.jobs.append(job)
        self.need_base_eval(base, sorted(used))
        run = {"id": rid, "block": block, "role": role, "base": base, "arch": arch, "steps": steps, "warmup": warm,
               "data": data, "seed": seed, "run_index": self.run_index, "env": env, "subset_spec": spec is not None,
               "evals": [{"name": ev["name"], "sets": ev["sets"], "variants": ev["variants"]} for ev in evals],
               "interim_sets": interim_sets or [], "interim_steps": isteps if interim_sets else [],
               "interim_full": interim_full, "stop_after": int(stop_after) if stop_after else None, **(meta or {})}
        self.runs.append(run)
        self.estimates.append({"id": rid, "kind": "train", "block": block, "base": base, "role": role,
                               "train_h": round(h_tr, 4), "interim_eval_h": round(h_int, 4), "eval_h": round(h_ev, 4),
                               "audit_h": round(h_aud, 4), "total_h": est,
                               "steps_priced": int(stop_after) if stop_after else int(steps)})
        self.blocks.setdefault(block, []).append(rid)
        return run

    def ecmd(self, jid: str, block: str, argv: list[str], est_h: float, requires: list[str] | None = None):
        self.cmd(jid, argv, round(est_h, 4), requires=requires)
        self.estimates[-1]["block"] = block
        self.blocks.setdefault(block, []).append(jid)

    def eval_job(self, jid: str, block: str, base: str, ckpt: str, names: list[str], variants: str, out: str,
                 extra_args: list[str] | None = None):
        sets = self.sets[base if base in self.sets else "moe_b1"]
        a, sp, sel = GP.set_args(sets, names)
        args = [self.cfg["experiments"].get(base, "moe_100m"), ckpt]
        for s in a:
            args += ["--set", s]
        for s in sp:
            args += ["--span", s]
        args += ["--select", sel, "--block", str(SEQ), "--variants", variants, "--base", f"{{ckpt_{base}}}", "--out", _rel(out)]
        args += list(extra_args or [])
        h = round(self.cost.eval_tokens(self.arch(base), EVAL_TOK * len(names) * n_passes(variants))
                  + self.cfg["throughput"]["model_load_min"] / 60, 4)
        self.jobs.append({"id": jid, "kind": "eval", "args": args, "est_h": h})
        self.estimates.append({"id": jid, "kind": "eval", "block": block, "base": base, "sets": names, "total_h": h})
        self.blocks.setdefault(block, []).append(jid)

    def shuffled(self, items: list, seed: int) -> list:
        rng = np.random.default_rng(seed)
        return [items[i] for i in rng.permutation(len(items))]


# ============================================================================= G1/G2 inputs
def g2_freeze(g2rec: dict, g1rec: dict) -> dict:
    f = dict(g2rec.get("freeze") or {})
    d1 = g1rec.get("decision") or {}
    f.setdefault("steps", d1.get("steps"))
    f.setdefault("q", d1.get("q"))
    f["g2_outcome"] = g2rec.get("outcome")
    return f


def load_states(paths: list[str]) -> dict:
    st = {}
    for p in paths:
        if p and os.path.exists(p):
            st.update(json.load(open(p, encoding="utf-8")))
    return st


# ============================================================================= blocks
def block_c1(B: ConfirmBuilder, fz: dict, steps: int, cells: dict, code: str, code2: str | None) -> None:
    # prereg v1.3 sec. 3.1 / spec 1.3.0: C1 is books-only; draws d = 1..2 n_d of the G0 record's books sequence (P-13)
    targets = ["books"]
    idx = list(fz.get("c1_draw_indices") or [])
    if not fz.get("C1_confirmatory") or not idx:
        B.warnings.append(f"C1 not built: C1_confirmatory {fz.get('C1_confirmatory')}, draws {idx} (G2 freeze; PIVOT or G2-4 "
                          f"failure). Blocks that reuse C1 runs (M12-b, dose-response P = 1 level, assumption audit) need an operator "
                          f"decision")
        return
    per_t = len(idx)
    sd = B.cc["seeds"]
    reg_cells = [tuple(c) for c in B.cc["c1"]["registered_cells"]]
    for base in ("moe_b1", "moe_b2"):
        t = B.table(base)
        for tgt in targets:
            cell = cells.get(f"{base}|{tgt}|{code}") or {}
            if not cell.get("feasible"):
                B.warnings.append(f"C1 {base}|{tgt}: G0-1 cell not feasible -> no runs")
                continue
            dcell = cells.get(f"moe_b1|{tgt}|{code}") or {}
            if (cell.get("f"), cell.get("tol")) != (dcell.get("f"), dcell.get("tol")):
                B.warnings.append(f"C1 {base}|{tgt}: f/tol {cell.get('f')}/{cell.get('tol')} differ from the decision-base "
                                  f"cell (G0-7 drew with {dcell.get('f')}/{dcell.get('tol')})")
            draws = [CX.reg_draw(t, "c1", d, f=float(cell["f"]), target=tgt, code=code, tol=float(cell["tol"]))
                     for d in idx]
            man = B.manifest(f"c1_{base}_{tgt}", draws, {"prereg": "v1.3 sec. 3.1 (C1 books-only; P-13)", "base": base, "target": tgt,
                                                         "code_stream": code, "f": cell["f"], "tol": cell["tol"]})
            B.blocks.setdefault("_manifests", {})[f"C1:{base}:{tgt}"] = man
            jobs = []
            for e in draws:
                if e.get("status") != "ok":
                    B.warnings.append(f"C1 {base}|{tgt} d{e['d']}: draw infeasible")
                    continue
                for side in ("A", "B"):
                    jobs.append((e, side))
            ci = reg_cells.index((base, tgt)) + 1 if (base, tgt) in reg_cells else 90 + len(reg_cells)
            order = B.shuffled(jobs, int(B.cc["c1"]["order_seed_base"]) + ci)
            B.blocks.setdefault("_order", {})[f"C1:{base}:{tgt}"] = {"seed": int(B.cc["c1"]["order_seed_base"]) + ci,
                                                                     "runs": [f"d{e['d']}{s}" for e, s in order]}
            for e, side in order:
                rid = f"c1_{base}_{tgt}_d{e['d']}_{side}"
                main = ["dclm", tgt, code] + ([code2] if code2 and B.cc["second_code_stream"] else []) + ["math"]
                evals = [{"name": "main", "sets": main, "variants": "ckpt"},
                         {"name": "reflect", "sets": [code, "math", "dclm"], "variants": "reflect"},
                         {"name": "pin", "sets": [code, "math"], "variants": CORE_PIN_VARIANTS if B.cc.get("selpin_block") else PIN_VARIANTS}]
                haar = None
                if base == B.cc["c1"]["controls_base"] and int(e["d"]) <= int(B.cc["c1"]["controls_max_d"]):
                    haar = int(sd["haar"]) + B.run_index + 1
                    evals += [{"name": "rand", "sets": [code, "math", "dclm"], "variants": f"randdir:{haar},randdir-:{haar}"},
                              {"name": "wd", "sets": [code, "math"], "variants": "wdonly"}]
                run_ = B.trun(rid, block=f"C1:{base}", base=base, role="c1", data=B.cfg["train_data"][tgt],
                       seed=int(sd["c1_data"]) + int(sd["base_offset"][base]) + int(e["d"]), steps=steps, spec=e[f"spec_{side}"], evals=evals,
                       interim_sets=["dclm", tgt, code, "math"],
                       meta={"target": tgt, "d": int(e["d"]), "side": side, "generator_seed": e["seed"],
                             "P": e["predicted"]["P"], "haar_seed": haar, "cell_index": ci,
                             "subset": B.subset_meta(t, e[side], code), "draw_manifest": man["path"]})
                B.defer_selpin(run_, [code, "math"])


def block_c2(B: ConfirmBuilder, fz: dict, steps: int, g0: dict, code: str) -> None:
    status = str(fz.get("c2_status") or "")
    lr, s_star, n_draws = fz.get("sgd_lr"), fz.get("s_star"), fz.get("C2_draws")
    if not n_draws or lr is None:
        B.warnings.append(f"C2 not built: C2_draws {n_draws}, SGD lr {lr} (c2_status '{status}'; O-5: no stable lr -> not run)")
        return
    confirmatory = bool(fz.get("C2_confirmatory"))
    draws = B.cc["c2"]["draws"] if int(n_draws) == 8 else B.cc["c2"]["exploratory_draws"]
    if s_star is None:
        B.warnings.append("C2: s* undefined (G2-3 'none' or G2-2 failed): 4-draw exploratory block, all steps descriptive (O-6)")
    res = (g0.get("_g03") or {})
    if not res.get("feasible"):
        B.warnings.append("C2 not built: G0-3 generator infeasible")
        return
    tgt = B.cc["c2"]["target"]
    # P-16 (2026-11-15): truncated at s* = 8 on the registered 96-step schedule (steps, warmup, cosine, data order and
    # seeds unchanged); adapt_extras ends the run right after the step-8 full evaluation, model_final.pt = the step-8 state
    stop = int(B.cc["c2"].get("stop_after") or 0)
    if stop and (s_star is None or int(s_star) != stop):
        B.warnings.append(f"P-16: C2 not truncated (s* = {s_star}; the registered truncation step is {stop})")
        stop = 0
    stop_sgd = int(B.cc["c2"].get("stop_after_sgd") or stop) if stop else 0    # MF-P1 (A): SGD to step 24, Muon to 8
    fsteps = [s for s in B.cc["c2"]["full_steps"] if not stop or s <= stop]     # full evaluation at s* = 8 (both arms)
    dclm_only = list(res.get("matched") or []) == ["dclm"]
    sd = B.cc["seeds"]
    for base in ("moe_b1", "moe_b2"):
        t = B.table(base)
        ents = [CX.reg_draw(t, "c2", d, f=float(res["f"]), target=tgt, code=code, tol=float(res["tol"]), dclm_only=dclm_only)
                for d in draws[base]]
        man = B.manifest(f"c2_{base}", ents, {"prereg": "v1.2 sec. 4.5 (C2)", "base": base, "target": tgt,
                                              "matched": res.get("matched"), "c2_status": status})
        B.blocks.setdefault("_manifests", {})[f"C2:{base}"] = man
        for e in ents:
            if e.get("status") != "ok":
                B.warnings.append(f"C2 {base} d{e['d']}: draw infeasible")
                continue
            for arm in ("sgd", "muon"):
                for side, sub in (("H", "B"), ("L", "A")):
                    env = {"DCLM_WEIGHT_DECAY": "0", "NESTOR_MAX_GRAD_NORM": "none", "NESTOR_INTERIM_FULL": "1"}
                    if arm == "sgd":
                        env.update({"NESTOR_SUBSET_OPTIMIZER": "sgd", "NESTOR_SGD_LR": repr(float(lr)),
                                    "NESTOR_SGD_MOMENTUM": str(B.cfg["recipe"]["momentum"]), "NESTOR_SGD_NESTEROV": "1",
                                    "NESTOR_SGD_WD": "0"})
                    evals = [{"name": "main", "sets": ["dclm", tgt], "variants": "ckpt"}]
                    haar = None
                    if arm == "muon":
                        # v1.2 sec. 8.2 item 7 (decision, 2026-11-14): antithetic randdir pair on DCLM at
                        # the final checkpoint -> b_Muon - b_Muon^rand at step 96; Haar seed as for the C1 controls
                        haar = int(sd["haar"]) + B.run_index + 1
                        evals.append({"name": "rand", "sets": ["dclm"], "variants": f"randdir:{haar},randdir-:{haar}"})
                    st_arm = stop_sgd if arm == "sgd" else stop
                    isteps = [s for s in B.cfg["interim_steps"] if not st_arm or s <= st_arm]
                    B.trun(f"c2_{base}_d{e['d']}_{arm}_{side}", block="C2", base=base, role=f"c2_{arm}",
                           data=B.cfg["train_data"][tgt], seed=int(sd["c2_data"]) + int(e["d"]), steps=steps,
                           spec=e[f"spec_{sub}"], evals=evals,
                           interim_sets=["dclm", tgt], interim_full={"steps": fsteps, "sets": ["dclm", tgt]},
                           interim_steps=isteps, stop_after=st_arm or None,
                           env_extra=env, meta={"target": tgt, "d": int(e["d"]), "arm": arm, "side": side,
                                                "generator_seed": e["seed"], "Delta_Q": e["predicted"]["Delta_Q"],
                                                "s_star": s_star, "c2_status": status, "haar_seed": haar,
                                                "subset": B.subset_meta(t, e[sub], code), "draw_manifest": man["path"]})


def block_c3(B: ConfirmBuilder, steps: int, code: str, g1plan: dict | None, states: dict) -> None:
    sd, c3 = B.cc["seeds"], B.cc["c3"]
    have = {}
    for r in (g1plan or {}).get("runs", []):
        if r.get("role") in ("full", "dclm_cont"):
            have[(r["base"], r.get("target"), int(r["seed"]) if r.get("seed") is not None else None, r["role"])] = r["id"]
    rg = B.cc["revert_groups"]
    per_layer = {}
    if c3.get("per_layer", True):                 # prereg v1.3 sec. 4.8 / P-4: counted as C3, never cut
        # routed strata '<kind>:b<block>:l<layer>' (the same (block, layer) as subset_adapt units):
        # MoE-100M = 4 blocks x 3 layers -> 12 single-LAYER reverts, one per stratum (moe@b<b>l<l>);
        # NESTOR-100M = blocks 0-5 (11 private strata) + dense tail -> 6 single-BLOCK reverts of the
        # column-private group, prereg v1.3 sec. 4.8 = column attention + private experts (v1.2 sec. 10.1; the
        # group-level revert 'attn+priv') -> 'revert:attn@b<b>+priv@b<b>' (was 'priv@b<b>', which reverted
        # the private experts only)
        for pb, grp, per_block, want in (("moe_b1", "moe", False, 12), ("nestor", "attn+priv", True, 6)):
            try:
                toks = c3_revert_tokens(B.table(pb).strata, per_block)
            except Exception as e:                                        # noqa: BLE001 - recorded, not fatal
                B.warnings.append(f"C3 per-layer: strata of {pb} unavailable ({e}) -> no per-layer reverts")
                continue
            if len(toks) != want:
                B.warnings.append(f"C3 per-layer: {pb} has {len(toks)} routed {'blocks' if per_block else 'layers'} in its "
                                  f"routing tables; prereg v1.3 sec. 4.8 registers {want}")
            per_layer[pb] = {"group": grp, "scope": "block" if per_block else "block+layer", "units": toks,
                             "variants": ",".join(c3_layer_variant(grp, t) for t in toks)}
        B.blocks["_c3_per_layer"] = {k: {"group": v["group"], "scope": v["scope"], "units": v["units"]} for k, v in per_layer.items()}
    plan_full = []
    for base, seeds in (("nestor", sd["full"]), ("moe_b1", sd["full"]), ("moe_b2", sd["full_b2"])):
        for dom in c3["domains"]:
            for s in seeds:
                plan_full.append((base, dom, int(s)))
    for base, dom, s in plan_full:
        rid_g1 = have.get((base, dom, s, "full"))
        arch = B.arch(base)
        rev = ",".join(f"revert:{g}" for g in rg["nestor" if arch == "nestor" else "moe"])
        wd = dom in c3["wd_only_domains"] and s == int(sd["full"][0]) and base in ("nestor", "moe_b1")
        lrev = per_layer[base]["variants"] if (base in per_layer and s == int(sd["full"][0])) else None
        if rid_g1:
            st1 = states.get(rid_g1) or {}
            fin, root = st1.get("final"), st1.get("root")
            if not fin:
                B.warnings.append(f"C3: G1 FULL run {rid_g1} has no final checkpoint in the G1 queue state -> its reverts "
                                  f"are not planned")
                continue
            xa = (["--manifest", _rel(os.path.join(root, "subset_manifest.json")), "--run-env",
                   _rel(os.path.join(root, "run_env.json"))] if (wd and root) else [])
            B.eval_job(f"c3_revert_{rid_g1}", "C3", base, _rel(fin), ["dclm"], rev + (",wdonly" if xa else ""),
                       os.path.join("outputs", "confirm", "c3_reverts", rid_g1), extra_args=xa)
            if lrev:
                B.eval_job(f"c3_layer_revert_{rid_g1}", "C3", base, _rel(fin), ["dclm"], lrev,
                           os.path.join("outputs", "confirm", "c3_layer_reverts", rid_g1))
            continue
        names = ["dclm", code, "math"] + ([dom] if dom not in ("code", "math") else [])
        evals = [{"name": "main", "sets": names, "variants": "ckpt"},
                 {"name": "revert", "sets": ["dclm"], "variants": rev}]
        if wd:
            evals.append({"name": "wd", "sets": ["dclm"], "variants": "wdonly"})
        if lrev:
            evals.append({"name": "revert_layer", "sets": ["dclm"], "variants": lrev})
        B.trun(f"c3_full_{base}_{dom}_s{s}", block="C3", base=base, role="full", data=B.cfg["train_data"][dom], seed=s,
               steps=steps, spec=None, evals=evals, interim_sets=["dclm", dom if dom != "code" else code],
               meta={"target": dom, "reverts": rg["nestor" if arch == "nestor" else "moe"], "wd_only": wd,
                     "per_layer_reverts": (per_layer.get(base) or {}).get("units") if lrev else None})
    for base in ("nestor", "moe_b2"):
        if have.get((base, "dclm", int(sd["dclm_cont"]), "dclm_cont")):
            continue
        B.trun(f"c3_dclmcont_{base}_s{sd['dclm_cont']}", block="DCLM-CONT", base=base, role="dclm_cont",
               data=B.cfg["dclm_cont"]["path"], seed=int(sd["dclm_cont"]), steps=steps, spec=None,
               evals=[{"name": "main", "sets": ["dclm", code, "math"], "variants": "ckpt"}],
               meta={"target": "dclm", "held_out_tokens": B.cfg["dclm_cont"]["path"] != B.cfg["dclm_cont"]["pretrain_path"]})
    for base in ("moe_b1", "nestor"):
        if base == "nestor":
            t = B.table("nestor")
            blocks = sorted({int(lk.split(":")[1][1:]) for lk in t.strata})
            spec = {"mode": "freeze", "select": [{"kind": "column", "block": blocks}]}
        else:
            spec = {"mode": "freeze", "select": [{"kind": "moe"}]}
        tgt = c3["freeze_target"]
        B.trun(f"c3_freeze_{base}_{tgt}_s{sd['freeze']}", block="C3", base=base, role="freeze", data=B.cfg["train_data"][tgt],
               seed=int(sd["freeze"]), steps=steps, spec=spec,
               evals=[{"name": "main", "sets": ["dclm", code, "math", tgt], "variants": "ckpt"}],
               interim_sets=["dclm", tgt], meta={"target": tgt, "frozen": spec["select"]})
    B.warnings.append("C3: the random partial-revert control (prereg sec. 4.7, C3 additions) is NOT planned: no "
                      "parameter-count- and displacement-matched random revert exists in weight_controls/eval_windows")


def block_c5(B: ConfirmBuilder, steps: int, code: str, forecasts_plan_path: str) -> None:
    t = B.table("moe_b1")
    tgt = B.cc["c5"]["target"]
    sd = B.cc["seeds"]
    if not {tgt, "dclm", code, "math"} <= set(t.sel):
        B.warnings.append(f"C5 not built: base-1 G0 tables lack {sorted({tgt, 'dclm', code, 'math'} - set(t.sel))}")
        return
    import g0_gate as GG

    cell = GG.ladder(t, "c1", GG.rungs_for("c1", target=tgt, code=code), target=tgt, code=code, n_draws=GG.G0_DRAWS,
                     k=CX.reg_k(t))
    B.blocks.setdefault("_c5_cell", {"feasible": cell["feasible"], "f": cell.get("f"), "tol": cell.get("tol"),
                                     "median": cell.get("median"), "rung": cell.get("rung")})
    drive = f"c5_drive_moe_b1_{tgt}_s{sd['c5_full']}"
    eall = {"mode": "train", "select": [{"kind": "moe"}]}
    B.trun(drive, block="C5", base="moe_b1", role="drive", data=B.cfg["train_data"][tgt], seed=int(sd["c5_full"]),
           steps=int(B.cc["c5"]["drive_steps"]), spec=eall,
           evals=[{"name": "main", "sets": ["dclm", code, "math", tgt], "variants": "ckpt"}],
           meta={"target": tgt, "purpose": "k-step drive (prereg sec. 12 item 1)"})
    fc_out = _rel(os.path.join(B.out, "c5_french_forecasts.json"))
    B.ecmd("c5_forecasts", "C5", ["{python}", "forecasts.py", "c5-french", "--plan", _rel(forecasts_plan_path),
                                  "--drive", drive, "--states", "results/queue_state_confirm.json", "--out", fc_out],
           0.01, requires=[drive])
    B.jobs.append({"id": "c5_hash_gate", "kind": "gate", "gate": "python", "requires": ["c5_forecasts"],
                   "params": {"func": "forecasts:c5_hash_gate", "file": fc_out, "log": "logs/c5_forecast_hashes.log"},
                   "est_h": 0.0})
    B.blocks.setdefault("C5", []).append("c5_hash_gate")
    runs = [(f"c5_full_moe_b1_{tgt}_s{sd['c5_full']}", None, None)]
    if cell["feasible"]:
        ents = [CX.reg_draw(t, "c5", d, f=float(cell["f"]), target=tgt, code=code, tol=float(cell["tol"]))
                for d in B.cc["c5"]["pairs"]]
        man = B.manifest("c5_moe_b1", ents, {"prereg": "v1.2 sec. 12 (C5 French)", "target": tgt, "cell": B.blocks["_c5_cell"]})
        B.blocks.setdefault("_manifests", {})["C5"] = man
        for e in ents:
            if e.get("status") == "ok":
                runs += [(f"c5_pair_moe_b1_{tgt}_d{e['d']}_{s}", e, s) for s in ("A", "B")]
    else:
        B.warnings.append(f"C5: the French crossover cell is infeasible at the G0-1 rungs -> FULL French only")
    for rid, e, side in runs:
        B.trun(rid, block="C5", base="moe_b1", role="c5_full" if e is None else "c5_pair", data=B.cfg["train_data"][tgt],
               seed=int(sd["c5_full"]) if e is None else int(sd["c5_data"]) + int(e["d"]), steps=steps,
               spec=None if e is None else e[f"spec_{side}"],
               evals=[{"name": "main", "sets": ["dclm", tgt, code, "math"], "variants": "ckpt"}],
               interim_sets=["dclm", tgt, code, "math"],
               meta={"target": tgt, **({} if e is None else {"d": int(e["d"]), "side": side, "generator_seed": e["seed"],
                                                             "P": e["predicted"]["P"], "subset": B.subset_meta(t, e[side], code)})})
        B.jobs[-1]["requires"] = ["c5_hash_gate"]


def block_dose(B: ConfirmBuilder, steps: int, code: str, dose_record: str | None) -> None:
    """Dose-response (prereg v1.3 sec. 4.2, P-9): the draws fixed at stage C by dose_draws.py (hashed manifest); 2 draws x 2
    subset runs per MoE base whose dose level is run; code and math full sets on the raw and reflected models (DCLM and
    books in the main pass for the forecast scoring and the gain). Data seeds as recorded (7100 + d, proposed)."""
    if not dose_record or not os.path.exists(dose_record):
        B.warnings.append("dose-response: no stage-C dose record (dose_draws.py) -> not planned")
        return
    rec = json.load(open(dose_record, encoding="utf-8"))
    man = rec.get("manifest") or {}
    if not man.get("path") or not os.path.exists(man["path"]):
        B.warnings.append("dose-response: dose record has no draw manifest -> not planned")
        return
    doc = read_manifest_verified(man["path"], man.get("sha256"))
    if doc is None:
        B.warnings.append("dose-response: draw manifest does not verify (sha256 vs sidecar / record) -> not planned")
        return
    ents = doc["entries"]
    B.blocks.setdefault("_manifests", {})["dose"] = {"path": _rel(man["path"]), "sha256": GP.file_sha256(man["path"]),
                                                    "f_low": rec.get("f_low")}
    for base, v in sorted((rec.get("bases") or {}).items()):
        if not v.get("run"):
            B.warnings.append(f"dose-response {base}: level not run ({v.get('reason')})")
            continue
        t = B.table(base)
        for e in [x for x in ents if x.get("base") == base]:
            for side in ("A", "B"):
                B.trun(f"dose_{base}_books_d{e['d']}_{side}", block="dose", base=base, role="dose",
                       data=B.cfg["train_data"]["books"], seed=int(e["data_seed"]), steps=steps, spec=e[f"spec_{side}"],
                       evals=[{"name": "main", "sets": ["dclm", "books", code, "math"], "variants": "ckpt"},
                              {"name": "reflect", "sets": [code, "math"], "variants": "reflect"}],
                       meta={"target": "books", "d": int(e["d"]), "side": side, "f_low": rec.get("f_low"), "generator_seed": e["seed"],
                             "P": e["predicted"]["P"], "overlap_per_layer": e.get("overlap_per_layer"),
                             "subset": B.subset_meta(t, e[side], code), "draw_manifest": _rel(man["path"])})


def block_m12b(B: ConfirmBuilder, steps: int, code: str) -> None:
    sd, m = B.cc["seeds"], B.cc["m12b"]
    c1runs = {(r["base"], r.get("target"), r.get("d"), r.get("side")): r for r in B.runs if r.get("role") == "c1"}
    t = B.table("moe_b1")
    for d in m["draws"]:
        r1 = c1runs.get(("moe_b1", "books", int(d), "A"))
        if r1 is None:
            B.warnings.append(f"M12-b d{d}: C1 books run moe_b1 d{d} A not in the plan -> skipped")
            continue
        if r1.get("haar_seed") is None:
            B.warnings.append(f"M12-b d{d}: the C1 books A run has no Haar draw (controls) -> no common random directions")
        spec = next(j for j in B.jobs if j["id"] == r1["id"])["spec"]
        haar = r1.get("haar_seed")
        evals = [{"name": "main", "sets": ["dclm", code, "math", m["target"]], "variants": "ckpt"}]
        if haar is not None:
            evals.append({"name": "rand", "sets": ["dclm", code], "variants": f"randdir:{haar},randdir-:{haar}"})
        B.trun(f"m12b_moe_b1_books_d{d}_A_on_{m['target']}", block="M12b", base="moe_b1", role="m12b",
               data=B.cfg["train_data"][m["target"]], seed=int(sd["m12b_data"]) + int(d), steps=steps, spec=spec,
               evals=evals, interim_sets=[m["target"]],
               meta={"target": m["target"], "d": int(d), "side": "A", "c1_partner": r1["id"], "haar_seed": haar,
                     "subset": r1.get("subset"),
                     "statistic": "IT_d = [log F_DCLM - log F_code](German) - [log F_DCLM - log F_code](books); "
                                  "learned minus random with common Haar draws"})


def block_b3(B: ConfirmBuilder, steps: int, code: str, b3rec_path: str | None) -> None:
    if not b3rec_path or not os.path.exists(b3rec_path):
        B.warnings.append("third base: g0/gate_record_b3.json missing (run g0_b3.ps1) -> C1-books replication not planned")
        return
    if "ckpt_moe_b3" not in B.vars:
        B.warnings.append("third base: moe_b3 checkpoint not pinned in ckpt_pins.json -> not planned")
        return
    rec = json.load(open(b3rec_path, encoding="utf-8-sig"))
    cell = rec["gates"]["G0-1"]["cells"].get(f"moe_b3|books|{code}") or {}
    if not cell.get("feasible") or (cell.get("median") or 0) < math.log(1.3):
        B.warnings.append(f"third base: books cell not feasible at G0-1 ({cell.get('median')}) -> not planned")
        return
    g5 = ((rec["gates"].get("G0-5") or {}).get("per_base") or {}).get("moe_b3") or {}
    label = "secondary (not in the Holm family)"
    if g5.get("pass") is not True:                              # P-11: a G0-5 failure gives an exploratory label
        label = "exploratory (G0-5 not passed: " + str(g5.get("status")) + "; P-11)"
        B.warnings.append(f"third base: G0-5 not passed ({g5.get('status')}) -> runs planned with the exploratory label (P-11)")
    d3 = os.path.join(os.path.dirname(os.path.abspath(b3rec_path)), "moe_b3")
    t = B.table("moe_b3", g0_dir=d3)
    sd = B.cc["seeds"]
    ents = [CX.reg_draw(t, "c1", d, f=float(cell["f"]), target="books", code=code, tol=float(cell["tol"]))
            for d in B.cc["b3"]["draws"]]
    man = B.manifest("c1_moe_b3_books", ents, {"prereg": "the design (third base; secondary)", "f": cell["f"]})
    B.blocks.setdefault("_manifests", {})["b3"] = man
    B.blocks["_b3"] = {"beta0": ((rec["gates"].get("G0-7") or {}).get("moe_b3") or {}).get("beta0"),
                       "median_P": cell.get("median"), "g0_5": g5.get("status"), "label": label}
    for e in ents:
        if e.get("status") != "ok":
            B.warnings.append(f"third base d{e['d']}: infeasible")
            continue
        for side in ("A", "B"):
            evals = [{"name": "main", "sets": ["dclm", "books", code, "math"], "variants": "ckpt"},
                     {"name": "reflect", "sets": [code, "math", "dclm"], "variants": "reflect"},
                     {"name": "pin", "sets": [code, "math"], "variants": CORE_PIN_VARIANTS if B.cc.get("selpin_block") else PIN_VARIANTS}]
            haar = None
            if side == "A":                       # prereg v1.3 sec. 4.1: full C1 evaluation set (antithetic random, WD-only)
                haar = int(sd["haar"]) + B.run_index + 1
                evals += [{"name": "rand", "sets": [code, "math", "dclm"], "variants": f"randdir:{haar},randdir-:{haar}"},
                          {"name": "wd", "sets": [code, "math"], "variants": "wdonly"}]
            run_ = B.trun(f"b3_c1_moe_b3_books_d{e['d']}_{side}", block="b3", base="moe_b3", role="c1_b3",
                   data=B.cfg["train_data"]["books"], seed=int(sd["c1_data"]) + int(sd["base_offset"]["moe_b3"]) + int(e["d"]), steps=steps,
                   spec=e[f"spec_{side}"], evals=evals,
                   interim_sets=["dclm", "books", code, "math"],
                   meta={"target": "books", "d": int(e["d"]), "side": side, "generator_seed": e["seed"], "haar_seed": haar,
                         "label": label, "P": e["predicted"]["P"], "subset": B.subset_meta(t, e[side], code),
                         "draw_manifest": man["path"]})
            B.defer_selpin(run_, [code, "math"])


def block_nestor_books(B: ConfirmBuilder, steps: int, code: str, feas_path: str | None, fz: dict | None = None) -> None:
    """NESTOR-books (prereg v1.3 sec. 4.4): preconditions feasibility (stage C) AND G1-3 (P-2); 3 draws x 2 runs; code/math
    full sets, interim, antithetic random directions (3 corpora), pinned AND selpin at theta_0 +- delta (P-12), reflected."""
    fz = fz or {}
    if fz.get("NESTORB_signal") is not True:
        B.warnings.append(f"NESTOR books: G1-3 NESTORB_signal = {fz.get('NESTORB_signal')} -> not run, budget released (P-2)")
        return
    if not feas_path or not os.path.exists(feas_path):
        B.warnings.append("NESTOR books: no feasibility file (nestor_books_feasibility.py) -> not planned")
        return
    fe = json.load(open(feas_path, encoding="utf-8"))
    if not fe.get("feasible"):
        B.warnings.append("NESTOR books: infeasible at G0-1 -> no exploratory block")
        return
    c = fe["cells"][fe["code_stream"]]
    t = B.table("nestor")
    sd = B.cc["seeds"]
    ents = [CX.reg_draw(t, "c1", d, f=float(c["f"]), target="books", code=code, tol=float(c["tol"]))
            for d in B.cc["nestor_books"]["draws"]]
    dm = (fe.get("draws_manifest") or {})
    doc = read_manifest_verified(dm.get("path"), dm.get("sha256")) if dm else None
    if dm and (doc is None or [x.get("subsets") for x in doc["entries"]] != [x.get("subsets") for x in ents]):
        B.warnings.append("NESTOR books: re-drawn subsets differ from the stage-C draw manifest (or it does not verify) -> not planned")
        return
    man = B.manifest("c1_nestor_books", ents, {"prereg": "v1.3 sec. 4.4 (exploratory; P-2, P-12)", "feasibility": feas_path,
                                               "label": fz.get("NESTORB_label")})
    B.blocks.setdefault("_manifests", {})["nestor_books"] = man
    for e in ents:
        if e.get("status") != "ok":
            continue
        for side in ("A", "B"):
            haar = int(sd["haar"]) + B.run_index + 1
            B.trun(f"nb_nestor_books_d{e['d']}_{side}", block="nestor_books", base="nestor", role="c1_nestor",
                   data=B.cfg["train_data"]["books"], seed=int(sd["c1_data"]) + int(e["d"]), steps=steps,
                   spec=e[f"spec_{side}"],
                   evals=[{"name": "main", "sets": ["dclm", "books", code, "math"], "variants": "ckpt"},
                          {"name": "reflect", "sets": [code, "math", "dclm"], "variants": "reflect"},
                          {"name": "pin", "sets": [code, "math"], "variants": PIN_VARIANTS},
                          {"name": "rand", "sets": [code, "math", "dclm"], "variants": f"randdir:{haar},randdir-:{haar}"}],
                   interim_sets=["dclm", "books", code, "math"],
                   meta={"target": "books", "d": int(e["d"]), "side": side, "generator_seed": e["seed"], "haar_seed": haar,
                         "label": fz.get("NESTORB_label"), "statistic": "I_sym,pin primary; live I and I_sym alongside",
                         "P": e["predicted"]["P"], "subset": B.subset_meta(t, e[side], code), "draw_manifest": man["path"]})


def block_audit(B: ConfirmBuilder, g0rec_path: str) -> None:
    """Assumption audit (prereg v1.3 sec. 4.3; P-10): directions are the REALISED displacements of the C1 books runs of the
    draw-1 subsets A_1 and B_1, restricted to the routed experts of each layer (24 groups per MoE base); probes on DCLM, code
    and math at 24 windows. The job therefore requires those two runs (run_queue 'requires')."""
    a = B.cc["audit"]
    probe_s = B.cc["audit"].get("probe_s_per_direction_corpus", 24.66)        # G0-7 measured rate (budget_v13 PROBE_S)
    for base in a["bases"]:
        ids = [f"c1_{base}_books_d1_{s}" for s in ("A", "B")]
        have = {r["id"] for r in B.runs}
        if not all(i in have for i in ids):
            B.warnings.append(f"audit {base}: C1 draw-1 runs {ids} not in the plan -> audit not planned (P-10)")
            continue
        n_groups = 2 * B.table(base).L
        est = n_groups * 3 * probe_s / 3600.0
        out = _rel(os.path.join("audit", base))
        B.ecmd(f"audit_{base}", "audit", ["{python}", "assumption_audit.py", "run", "--g0-record", _rel(g0rec_path),
                                         "--base", base, "--out", out, "--windows", str(a["windows"]), "--directions", "realized",
                                         "--runs", ",".join(ids), "--state", _rel(os.path.join("results", "queue_state_confirm.json")),
                                         "--queue", _rel(os.path.join(B.out, "queue_confirm.json")),
                                         "--checkpoint", f"{{ckpt_{base}}}"], est, requires=ids)

def block_c1de(B: ConfirmBuilder, steps: int, code: str, c1de_dir: str | None, fz: dict) -> None:
    """C1-DE (prereg v1.3 sec. 4.12; exploratory): the stage-C MILP draws (c1de_milp.py -> results/c1de), re-verified on the
    plan's tables, 2 subset runs per draw on German (data 7600 + d), the third-base evaluation set; never replaces books."""
    import c1de_milp as DE

    cc, sd = B.cc["c1de"], B.cc["seeds"]
    q = float(fz.get("q") or CX.REG_Q)
    if abs(q - CX.REG_Q) > 1e-12:
        B.warnings.append(f"C1-DE: G1-2 set q = {q} (k equals the pool size) -> not run, recorded (prereg v1.3 sec. 4.12)")
        return
    if not c1de_dir or not os.path.isdir(c1de_dir):
        B.warnings.append(f"C1-DE: no stage-C generator output in {c1de_dir} (c1de_milp.py) -> not planned")
        return
    tgt, f = cc["target"], float(cc["f"])
    n_runs = 0
    for base in cc["bases"]:
        p = os.path.join(c1de_dir, f"c1de_{base}_{tgt}.json")
        doc = read_manifest_verified(p)
        if doc is None:
            B.warnings.append(f"C1-DE {base}: manifest {p} missing or not matching its .sha256 -> not planned")
            continue
        t = B.table(base)
        tilt = CX.reg_tilt(t.sel, "c1", code=code)
        corpora = CX.reg_matched("c1", target=tgt, code=code)
        k = CX.reg_k(t)
        ents, runs = [], []
        for e in doc["entries"]:
            d = int(e["d"])
            if d not in [int(x) for x in cc["draws"]]:
                continue
            ents.append(e)
            if e.get("status") != "ok":
                B.warnings.append(f"C1-DE {base} d{d}: draw {e.get('status')} -> not run (recorded)")
                continue
            bad = []
            if int(e.get("seed", -1)) != int(sd["c1de_objective"]) + d:
                bad.append(f"objective seed {e.get('seed')}")
            if (e.get("code"), e.get("target"), float(e.get("f", -1)), float(e.get("tol", -1))) != (code, tgt, f, CX.REG_TOL):
                bad.append("code stream / target / f / tol")
            A, Bs = np.array(e["A"]), np.array(e["B"])
            chk = DE.checks(t.sel, tilt, A, Bs, k, f, corpora, CX.REG_TOL)
            if not chk["ok"]:
                bad.append(f"checks on the plan tables {chk}")
            p_est = float(CX.reg_predicted(t.est, A[None], Bs[None], "c1", target=tgt, code=code)[0])
            if abs(p_est - float(e["predicted"]["P"])) > 1e-9:
                bad.append(f"P {e['predicted']['P']} != {p_est} on the plan tables")
            if bad:
                B.warnings.append(f"C1-DE {base} d{d}: draw rejected by the plan ({'; '.join(bad)}) -> not run")
                continue
            runs.append(e)
        man = B.manifest(f"c1de_{base}_{tgt}", ents, {"prereg": "v1.3 sec. 4.12 (C1-DE, exploratory)", "base": base,
                                                     "target": tgt, "code_stream": code, "f": f, "tol": CX.REG_TOL,
                                                     "source": {"path": _rel(p), "sha256": GP.file_sha256(p)}})
        B.blocks.setdefault("_manifests", {})[f"C1-DE:{base}"] = man
        for e in runs:
            for side in ("A", "B"):
                evals = [{"name": "main", "sets": ["dclm", tgt, code, "math"], "variants": "ckpt"},
                         {"name": "reflect", "sets": [code, "math", "dclm"], "variants": "reflect"},
                         {"name": "pin", "sets": [code, "math"], "variants": PIN_VARIANTS}]
                haar = None
                if side == "A":                        # as for the third base: antithetic random and WD-only on side A
                    haar = int(sd["haar"]) + B.run_index + 1
                    evals += [{"name": "rand", "sets": [code, "math", "dclm"], "variants": f"randdir:{haar},randdir-:{haar}"},
                              {"name": "wd", "sets": [code, "math"], "variants": "wdonly"}]
                B.trun(f"c1de_{base}_d{e['d']}_{side}", block="C1-DE", base=base, role="c1de",
                       data=B.cfg["train_data"][tgt], seed=int(sd["c1de_data"]) + int(sd["base_offset"][base]) + int(e["d"]), steps=steps,
                       spec=e[f"spec_{side}"], evals=evals, interim_sets=["dclm", tgt, code, "math"],
                       meta={"target": tgt, "d": int(e["d"]), "side": side, "generator_seed": e["seed"], "haar_seed": haar,
                             "label": "exploratory (prereg v1.3 sec. 4.12)", "P": e["predicted"]["P"],
                             "P_selection": (e.get("selection") or {}).get("P"), "subset": B.subset_meta(t, e[side], code),
                             "draw_manifest": man["path"]})
                n_runs += 1
    if n_runs:                                         # budget_v13: the probe units of the C1-DE runs sit in the block
        pu = cc["probe_unit"]
        unit_h = (pu["directions"] * pu["corpora"] * float(B.cc["audit"].get("probe_s_per_direction_corpus", 24.66)) * pu["windows"] / 24
                  + pu["grad_s"]) / 3600.0
        B.estimates.append({"id": "c1de_probe_units", "kind": "probe", "block": "C1-DE", "base": "moe_b1,moe_b2",
                            "role": "c1de_probe", "train_h": 0.0, "interim_eval_h": 0.0, "eval_h": round(n_runs * unit_h, 4),
                            "audit_h": 0.0, "total_h": round(n_runs * unit_h, 4)})


# ============================================================================= build
def build(a) -> dict:
    """P-5 in two passes: pass 1 (<out>/pass1) applies the cut order with the restoration step; pass 2 (<out>) rebuilds
    the plan without the cut blocks (and without the second code stream if it is cut), so the queue, the manifests and the
    forecast units contain only what runs. Without cuts both passes are identical (deterministic build)."""
    ccfg = merge(copy.deepcopy(CONFIRM_DEFAULTS), json.load(open(a.confirm_config, encoding="utf-8-sig")) if a.confirm_config else {})
    a1 = copy.copy(a)
    a1.out_dir = os.path.join(a.out_dir, "pass1")
    p1 = _build(a1, copy.deepcopy(ccfg))
    dec = dict(p1["cut_decision"])
    ccfg2, removed = copy.deepcopy(ccfg), []
    for step in dec["cuts"]:
        if step == "second code stream":
            ccfg2["second_code_stream"] = False
            removed.append(step)
        elif ccfg2["blocks"].get(step):
            ccfg2["blocks"][step] = False
            removed.append(step)
    dec.update(removed=removed, pass1_plan={"path": _rel(os.path.join(a1.out_dir, "plan_confirm.json")), "sha256": p1["sha256"]})
    return _build(a, ccfg2, decision=dec)


def _build(a, ccfg: dict, decision: dict | None = None) -> dict:
    cfg = GP.load_config(a.config)
    g1rec = json.load(open(a.g1_record, encoding="utf-8"))
    g2rec = json.load(open(a.g2_record, encoding="utf-8"))
    pins = json.load(open(a.pins, encoding="utf-8-sig")) if a.pins and os.path.exists(a.pins) else None
    meas = json.load(open(a.throughput, encoding="utf-8")) if a.throughput and os.path.exists(a.throughput) else None
    cost = Cost(cfg, (meas or {}).get("per_arch"), ccfg.get("train"))
    cfg["experiments"].setdefault("moe_b3", "moe_100m")        # third base: same experiment
    B = ConfirmBuilder(cfg, ccfg, a.g0_record, a.out_dir, pins=pins, cost=cost)
    fz = g2_freeze(g2rec, g1rec)
    if fz.get("g2_outcome") != "COMPLETE":
        raise SystemExit(f"G2 outcome {fz.get('g2_outcome')!r}: no confirmatory plan before G2 is COMPLETE")
    steps = int(fz.get("steps") or cfg["recipe"]["steps"])
    code, code2 = B.g0["code_stream"], B.g0["code_secondary"]
    cells = B.g0rec["gates"]["G0-1"]["cells"]
    B.g0["_g03"] = (B.g0rec["gates"].get("G0-3") or {}).get("result") or {}
    states = load_states([p for p in (a.states or "").split(",") if p])
    g1plan_p = (g1rec.get("inputs") or {}).get("plan", {}).get("path")
    g1plan = json.load(open(g1plan_p, encoding="utf-8")) if g1plan_p and os.path.exists(g1plan_p) else None
    g2plan_p = (g2rec.get("inputs") or {}).get("plan", {}).get("path")
    g2plan = json.load(open(g2plan_p, encoding="utf-8")) if g2plan_p and os.path.exists(g2plan_p) else None
    reuse = {}
    for pl in (g1plan, g2plan):
        for b, files in ((pl or {}).get("base_eval_files") or {}).items():
            for n, p in files.items():
                reuse.setdefault(b, {})[n] = p
    bl = ccfg["blocks"]
    plan_path = os.path.join(a.out_dir, "plan_confirm.json")
    if bl["C1"]:
        block_c1(B, fz, steps, cells, code, code2)
    if bl["C2"]:
        block_c2(B, fz, steps, B.g0, code)
    if bl["C3"]:
        block_c3(B, steps, code, g1plan, states)
    if bl["C5"]:
        block_c5(B, steps, code, plan_path)
    if bl["dose"]:
        block_dose(B, steps, code, getattr(a, "dose_record", None))
    if bl["M12b"]:
        block_m12b(B, steps, code)
    if bl["b3"]:
        block_b3(B, steps, code, a.b3_record)
    if bl["audit"]:
        block_audit(B, a.g0_record)
    if bl["nestor_books"]:
        block_nestor_books(B, steps, code, a.nestor_books, fz)
    if bl.get("C1-DE"):
        block_c1de(B, steps, code, getattr(a, "c1de_dir", None), fz)
    if bl["SEC"]:
        B.warnings.append("SEC: no runs (DCLM-tilt infeasible at G0-4; D4 embedding-row training has no subset_adapt "
                          "support) -- first item of the cut order")
    B.flush_selpin()                                     # P-16: the deferred selpin eval jobs (block 'selpin')
    # theta_0 evaluation jobs first, then the blocks in the design order
    nblocks = {k: v for k, v in B.blocks.items() if not k.startswith("_")}
    train_jobs = {j["id"]: j for j in B.jobs}
    B.jobs = []
    B.flush_base_evals(reuse)
    base_jobs = list(B.jobs)
    ordered = []
    for key in ccfg["order"]:
        for bk, ids in nblocks.items():
            if bk == key:
                ordered += [train_jobs[i] for i in ids if i in train_jobs]
    ordered += [j for j in train_jobs.values() if j not in ordered]
    # P-16 revised (2026-11-15 ~03:00): only the theta_0 evaluations of the core blocks' bases come first; any other
    # base evaluation (e.g. the third base) is placed right before the first job that uses its base, i.e. after the marker
    core_cfg = ccfg.get("core") or {}
    base_of = {e["id"]: e.get("base") for e in B.estimates if e.get("id")}
    blk_of = {i: bk for bk, ids in nblocks.items() for i in ids}
    core_bases = {base_of.get(j["id"]) for j in ordered if blk_of.get(j["id"]) in (core_cfg.get("blocks") or [])}
    if core_cfg.get("blocks"):
        later = [j for j in base_jobs if base_of.get(j["id"]) not in core_bases]
        base_jobs = [j for j in base_jobs if j not in later]
        for bj in later:
            k = next((n for n, j in enumerate(ordered) if base_of.get(j["id"]) == base_of.get(bj["id"])), len(ordered))
            ordered.insert(k, bj)
    B.jobs = base_jobs + ordered
    # R10: an overrun guard (run_queue python gate) before each cuttable block of the registered cut order
    if ccfg.get("overrun_guard", True):
        block_of = {i: bk for bk, ids in nblocks.items() for i in ids}
        cut_blocks = [s for s in ccfg["budget"]["cut_order"] if s in nblocks]
        guarded, jobs2 = set(), []
        for j in B.jobs:
            bk = block_of.get(j["id"])
            if bk in cut_blocks:
                gid = f"overrun_guard_{bk}"
                if bk not in guarded:
                    jobs2.append({"id": gid, "kind": "gate", "gate": "python", "est_h": 0.0,
                                  "params": {"func": "confirm_plan:overrun_guard", "block": bk, "plan": _rel(plan_path),
                                             "queue": _rel(os.path.join(a.out_dir, "queue_confirm.json")),
                                             "envelope_h": float(ccfg["budget"]["envelope_h"]),
                                             "spent_before_h": float(a.spent_h or 0.0), "cut_order": cut_blocks,
                                             "freeze": "FREEZE.json"}})
                    guarded.add(bk)
                j["requires"] = list(j.get("requires") or []) + [gid]
            jobs2.append(j)
        B.jobs = jobs2
    # P-16 (2026-11-15): the core marker, a run_queue python gate right after the last core job. It has no 'requires'
    # (run_queue runs the queue in order), so it runs whatever the core jobs' outcome; the theta_0 evaluations count as core
    core = ccfg.get("core") or {}
    core_ids = []
    if core.get("blocks"):
        block_of = {i: bk for bk, ids in nblocks.items() for i in ids}
        core_ids = [j["id"] for j in base_jobs] + [j["id"] for j in B.jobs if block_of.get(j["id"]) in core["blocks"]]
        # base_jobs now holds only the theta_0 evaluations of the core bases (P-16 revised)
        last = max((k for k, j in enumerate(B.jobs) if j["id"] in core_ids), default=-1)
        B.jobs.insert(last + 1, {"id": "core_done", "kind": "gate", "gate": "python", "est_h": 0.0,
                                 "params": {"func": "confirm_plan:core_marker", "out": core["marker"],
                                            "core_blocks": list(core["blocks"]), "core_jobs": core_ids,
                                            "plan": _rel(plan_path), "queue": _rel(os.path.join(a.out_dir, "queue_confirm.json"))}})
    # budget by block
    tot_by = {}
    for e in B.estimates:
        tot_by[e.get("block", "base_eval")] = round(tot_by.get(e.get("block", "base_eval"), 0.0) + e["total_h"], 4)
    total = round(sum(tot_by.values()), 3)
    p16 = None
    if core.get("blocks"):
        cset = set(core_ids)
        core_h = round(sum(e["total_h"] for e in B.estimates if e.get("id") in cset), 4)
        p16 = {"rule": "P-16 revised (2026-11-15 ~03:00): core first -- C1 on both MoE bases with its registered "
                       "controls, C2 truncated at s* = 8 and the theta_0 evaluations C1/C2 need; then the core marker, the "
                       "third base and the assumption audit (never cut; full analysis), and the remaining blocks in the "
                       "registered order under the overrun rule",
               "core_blocks": list(core["blocks"]) + ["base_eval"], "marker": core["marker"], "core_jobs": len(core_ids),
               "core_base_evals": [j["id"] for j in base_jobs],
               "after_marker_first": [b for b in ("b3", "audit") if b in nblocks],
               "core_planned_gpu_h": core_h, "remaining_planned_gpu_h": round(total - core_h, 4),
               "c2_stop_after": int(ccfg["c2"].get("stop_after") or 0) or None,
               "c2_stop_after_sgd": int(ccfg["c2"].get("stop_after_sgd") or ccfg["c2"].get("stop_after") or 0) or None,
               "selpin_block": bool(ccfg.get("selpin_block")),
               "order": list(ccfg["order"])}
    bud = ccfg["budget"]
    spent = float(a.spent_h or 0.0)
    # P-8: contingency on the PLANNED hours only (the spent G0 hours carry none); the v1.2 convention is reported alongside
    with_c = round(spent + total * (1 + bud["contingency"]), 3)
    with_c_v12 = round((spent + total) * (1 + bud["contingency"]), 3)
    code2_h = round(sum(1 for r in B.runs if r.get("role") == "c1") * B.cost.eval_tokens("moe", EVAL_TOK), 4) if code2 else 0.0
    items = dict(tot_by)
    items["second code stream"] = code2_h

    # P-5: the cut order, then the restoration step (re-admit cut items in reverse order while the plan still fits);
    # pass 2 (decision given) keeps the decision of pass 1 and rebuilt without the cut blocks
    keep, cont, cuts = {k: v for k, v in items.items() if k != "second code stream"}, float(bud["contingency"]), []
    if decision is None:
        keep_code2 = True
        cur = with_c
        for step in bud["cut_order"]:
            if cur <= bud["envelope_h"]:
                break
            if step.startswith("contingency"):
                cont = 0.10
            elif step == "second code stream" and keep_code2 and code2_h:
                keep_code2 = False
                keep["C1:moe_b1"] = keep.get("C1:moe_b1", 0.0) - code2_h            # its passes sit inside the C1 jobs
            elif step in keep:
                keep.pop(step)
            else:
                continue
            cuts.append(step)
            cur = round(spent + sum(keep.values()) * (1 + cont), 3)
        restored = []
        if cur <= bud["envelope_h"]:
            for step in reversed(list(cuts)):
                if step.startswith("contingency"):
                    if round(spent + sum(keep.values()) * (1 + float(bud["contingency"])), 3) <= bud["envelope_h"]:
                        cont = float(bud["contingency"])
                        restored.append(step)
                elif step == "second code stream":
                    if round(spent + (sum(keep.values()) + code2_h) * (1 + cont), 3) <= bud["envelope_h"]:
                        keep["C1:moe_b1"] += code2_h
                        keep_code2 = True
                        restored.append(step)
                elif round(spent + (sum(keep.values()) + tot_by[step]) * (1 + cont), 3) <= bud["envelope_h"]:
                    keep[step] = tot_by[step]
                    restored.append(step)
            cuts = [s for s in cuts if s not in restored]
            cur = round(spent + sum(keep.values()) * (1 + cont), 3)
    else:
        cuts, restored, cont = list(decision["cuts"]), list(decision["restored"]), float(decision["cont"])
        cur = round(spent + total * (1 + cont), 3)
        for s in decision.get("removed") or []:
            B.warnings.append(f"cut by the P-5 order: {s} removed from the queue (pass 1: {decision['pass1_plan']['path']})")
    envelope_decision = None if cur <= bud["envelope_h"] else (
        f"cut order exhausted at {cur} GPU-h > {bud['envelope_h']}: the envelope decision is required (P-6: no H100 "
        f"fallback; local GPU time extended and reported; all adaptation runs on one platform); the freeze waits")
    if envelope_decision:
        B.warnings.append(envelope_decision)
    extra = {"freeze_inputs": fz, "steps": steps, "code_stream": code, "code_secondary": code2, "p16": p16,
             "confirm_config": ccfg, "cost_source": cost.source(), "throughput_measured": meas,
             "blocks": {k: v for k, v in B.blocks.items()}, "block_gpu_h": tot_by, "estimate_spent_gpu_h": spent,
             "estimate_with_contingency_gpu_h": with_c, "estimate_with_contingency_v12_convention_gpu_h": with_c_v12,
             "envelope_gpu_h": bud["envelope_h"], "cut_order_applied": cuts, "cut_order_restored": restored,
             "cut_rule": "order with the restoration step (P-5); contingency on planned hours (P-8)",
             "after_cuts_gpu_h": cur, "contingency_after_cuts": cont, "envelope_decision_required": envelope_decision,
             "cut_decision": decision if decision is not None else {"cuts": cuts, "restored": restored, "cont": cont,
                                                                     "cur": cur, "with_c": with_c, "block_gpu_h": tot_by},
             "cut_blocks_removed": (decision or {}).get("removed", []),
             "second_code_stream_gpu_h": code2_h, "base_eval_files": {b: {k: _rel(v) for k, v in f.items()} for b, f in B.base_files.items()},
             "g1_record": {"path": os.path.abspath(a.g1_record), "sha256": GP.file_sha256(a.g1_record),
                           "content_sha256": g1rec.get("sha256")},
             "g2_record": {"path": os.path.abspath(a.g2_record), "sha256": GP.file_sha256(a.g2_record),
                           "content_sha256": g2rec.get("sha256")},
             "status_labels": {"C1": ("confirmatory" if fz.get("C1_confirmatory") else "not run (PIVOT / G2-4: C1 not confirmatory)"),
                               "C2": ("confirmatory" if fz.get("C2_confirmatory") else
                                      ("exploratory" if fz.get("C2_draws") else "not run (O-5)")),
                               "C3": "estimation (per-layer reverts included, never cut: P-4)", "C5": "estimation (prospective French)",
                               "dose": "secondary", "M12b": "secondary", "b3": (B.blocks.get("_b3") or {}).get("label", "secondary"),
                               "audit": "secondary (descriptive; after the C1 draw-1 runs, P-10)",
                               "nestor_books": f"exploratory ({fz.get('NESTORB_label')})", "SEC": "secondary (no runs)",
                               "C1-DE": "exploratory (prereg v1.3 sec. 4.12; outside the Holm family)"}}
    extra["data_seeds"] = check_data_seeds(B.runs)                 # MF-3: raises if two draws share data
    plan = B.write(extra)
    with open(os.path.join(a.out_dir, "estimates_confirm.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "kind", "block", "base", "role", "train_h", "interim_eval_h", "eval_h",
                                           "audit_h", "total_h"], extrasaction="ignore")
        w.writeheader()
        for e in B.estimates:
            w.writerow(e)
    return plan


# ============================================================================= measured throughput
def throughput(plans: list[str], states: list[str]) -> dict:
    st = load_states(states)
    rows = []
    for pp in plans:
        pl = json.load(open(pp, encoding="utf-8"))
        q = json.load(open(pl["queue"], encoding="utf-8"))
        jobs = {j["id"]: j for j in q["jobs"]}
        for r in pl["runs"]:
            s = st.get(r["id"]) or {}
            if s.get("status") != "done" or not s.get("attempts"):
                continue
            j = jobs[r["id"]]
            tr_h = s["attempts"][-1]["minutes"] / 60
            post_h = max(0.0, float(s.get("hours", 0)) - sum(x["minutes"] for x in s["attempts"]) / 60)
            passes = sum(len(ev["sets"]) * n_passes(ev.get("variants", "ckpt")) for ev in j.get("evals", []))
            rows.append({"id": r["id"], "arch": r["arch"], "steps": r["steps"], "train_h": tr_h, "post_h": post_h,
                         "eval_passes": passes})
    per = {}
    for arch in sorted({r["arch"] for r in rows}):
        rr = [r for r in rows if r["arch"] == arch]
        tps = [r["train_h"] / r["steps"] for r in rr]
        ev = [r["post_h"] / r["eval_passes"] / EVAL_TOK for r in rr if r["eval_passes"]]
        per[arch] = {"n_runs": len(rr), "train_h_per_step": float(np.median(tps)), "fixed_h": 0.0,
                     "eval_h_per_token": float(np.median(ev)) if ev else None,
                     "train_h_96": float(np.median(tps)) * 96, "eval_pass_min": float(np.median(ev)) * EVAL_TOK * 60 if ev else None}
    return {"kind": "measured throughput (G1/G2 queue states)", "rows": rows, "per_arch": per,
            "note": "train_h includes compile and model load of the successful attempt; eval time per pass = (job hours - "
                    "training attempts) / evaluation passes (includes the bitwise audit of subset runs)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="confirmatory + secondary queue builder (prereg v1.3 blocks)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--g0-record", required=True)
    b.add_argument("--g1-record", required=True)
    b.add_argument("--g2-record", required=True)
    b.add_argument("--config", default="g1g2_config.json")
    b.add_argument("--confirm-config", default=None)
    b.add_argument("--pins", default="ckpt_pins.json")
    b.add_argument("--b3-record", default="g0/gate_record_b3.json")
    b.add_argument("--nestor-books", default="results/nestor_books_feasibility.json")
    b.add_argument("--dose-record", default="results/dose/dose_record.json")
    b.add_argument("--c1de-dir", default="results/c1de", help="C1-DE generator output (c1de_milp.py, stage C)")
    b.add_argument("--throughput", default=None)
    b.add_argument("--states", default="results/queue_state_g1.json,results/queue_state_g1r.json,results/queue_state_g2.json")
    b.add_argument("--spent-h", type=float, default=None, help="GPU-h already used (envelope check)")
    b.add_argument("--out-dir", default="confirm")
    b.add_argument("--no-check", action="store_true")
    t = sub.add_parser("throughput")
    t.add_argument("--plans", required=True)
    t.add_argument("--states", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "throughput":
        r = throughput([p for p in a.plans.split(",") if p], [p for p in a.states.split(",") if p])
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(GP.canonical(r) + "\n")
        print(json.dumps(r["per_arch"], indent=1))
        return 0
    plan = build(a)
    for k, v in sorted(plan["block_gpu_h"].items()):
        print(f"[confirm] block {k:<14} {v:8.2f} GPU-h  ({len([e for e in plan['estimates'] if e.get('block') == k])} jobs)")
    print(f"[confirm] {len(plan['runs'])} train runs, {len(plan['estimates'])} jobs, {plan['estimate_total_gpu_h']:.2f} GPU-h "
          f"({plan['cost_source']}); with spent + contingency {plan['estimate_with_contingency_gpu_h']:.2f} of "
          f"{plan['envelope_gpu_h']:.0f}; cuts {plan['cut_order_applied']}; sha256 {plan['sha256']}")
    for w in plan["warnings"]:
        print("WARNING", w)
    if not a.no_check:
        missing = GP.check_paths(plan)
        if missing:
            print(f"MISSING INPUTS ({len(missing)}): " + "; ".join(missing[:20]))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
