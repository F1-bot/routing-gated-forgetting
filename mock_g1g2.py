"""SIMULATED G1/G2 outputs for gate tests and the mock gate records (no model, no GPU; every number is synthetic).

Writes, at the paths a g1g2_plan.py plan (spec 1.3.0 runs) names (cwd = the workspace the plan was built in): the run_queue
state file, each run's root with eval_main/ckpt__<set>.npz, eval_reflect/reflect__<set>.npz, eval_pin/*.npz,
interim/stepNNN__<set>.npz (64-window heads) and interim/stepNNN__<set>__full.npz (500 windows; SGD runs) and
trainlog/train_steps.jsonl; the theta_0 evaluation files, theta0_checks.json, the GPU pipeline-test junit file (G1-0),
optionally a routing-health file (G0-9L), the H100 duplicate and the grad_snr output. Per-window CE of a run =
CE_0(w) + F + noise with exact sample moments, so F and SE are known by construction.

  python mock_g1g2.py all --g0-record g0_mock/g0/gate_record.json --config g1g2_config_MOCK.json --lookup signflip_lookup_v1.csv --root mock_g1g2
writes mock_g1g2/g1/gate_record_G1_MOCK.json and mock_g1g2/g2/gate_record_G2_MOCK.json (baseline scenario).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import zlib  # noqa: E402

import numpy as np  # noqa: E402

W = 500
HEAD = 64
N_TOK = 4095

BASE_SCENARIO = {
    "seed": 0,
    "pipeline": {"tests": 6, "failures": 0, "errors": 0, "skipped": 0},
    "full_moe_F_dclm": 0.020, "full_sd": 0.03,
    # pair a = 1 (German, S1): R2 scaled against R1
    "aa_F": {"code": 0.012, "math": 0.010}, "aa_sd": 0.02, "aa_B_scale": {"code": 1.12, "math": 0.93},
    "eall_F": 0.015, "dclm_cont_F": 0.004,
    "nestor_full_F_dclm": {"101": 0.018, "102": 0.016},
    "nestor_aa_F": {"code": 0.011, "math": 0.009},
    "negctrl_audit_fails": True, "audit_fail": [], "diverge": [], "missing": [],
    "aa_gain": {"8": 0.08, "24": 0.20, "48": 0.30},
    "h100_F_dclm": 0.0185,
    # books pairs a = 2 (S1, and its G2 repeat '_cfg') and a = 3 (G2)
    "books_F": {"code": 0.011, "math": 0.010}, "books_sd": 0.02,
    "books_B_scale": {"2": {"code": 0.90, "math": 1.08}, "3": {"code": 1.06, "math": 0.97}},
    "books_F_by_pair": {},                       # optional override {"2": {"code": ..., "math": ...}}
    "sgd": [{"gain": {"8": 0.03, "24": 0.06, "48": 0.09}, "dclm_F": {"8": 0.007, "24": 0.012, "48": 0.02}},
            {"gain": {"8": 0.05, "24": 0.10, "48": 0.15}, "dclm_F": {"8": 0.009, "24": 0.016, "48": 0.026}},
            {"gain": {"8": 0.08, "24": 0.16, "48": 0.22}, "dclm_F": {"8": 0.012, "24": 0.02, "48": 0.03}}],
    "sgd_interim_sd": 0.02, "sgd_full_sd": 0.02, "sgd_clipped": [], "sgd_no_loss_drop": [], "sgd_diverge": [],
    "theta0_c": 0.05,
    "pin_shift": {"pinned": -0.0010, "selpin": -0.0004}, "reflect_F": 0.005,
    "routing_health": None,                      # optional {"routers": {name: {diag: value}}}
}


def write_eval(path: str, index: np.ndarray, hashes: np.ndarray, ce_mean: np.ndarray, meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path + ".tmp.npz", index=index.astype(np.int64), hash=hashes.astype(np.uint64),
                        ce_sum=(ce_mean * N_TOK).astype(np.float64), n_tok=np.full(len(index), N_TOK, dtype=np.int64),
                        meta_json=np.array(json.dumps(meta or {"MOCK": True})))
    os.replace(path + ".tmp.npz", path)


class World:
    def __init__(self, scenario: dict):
        self.s = scenario
        self.rng = np.random.default_rng(int(scenario.get("seed", 0)))
        self.grid: dict[str, tuple] = {}

    def base_ce(self, set_name: str):
        if set_name not in self.grid:
            r = np.random.default_rng(zlib.crc32(set_name.encode("utf-8")))
            idx = np.sort(r.choice(100000, size=W, replace=False))
            hs = r.integers(1, 2 ** 63, size=W, dtype=np.int64).astype(np.uint64)
            ce = 3.0 + 0.5 * r.standard_normal(W)
            self.grid[set_name] = (idx, hs, ce)
        return self.grid[set_name]

    def run_ce(self, set_name: str, F: float, sd: float, n: int = W) -> np.ndarray:
        idx, hs, ce = self.base_ce(set_name)
        e = self.rng.standard_normal(n)
        e = (e - e.mean()) / e.std(ddof=1)               # exact sample moments: F and SE known by construction
        return ce[:n] + F + sd * e


def _train_log(path: str, steps: int, warm: int, *, diverge: bool = False, no_drop: bool = False, clipped: bool = False,
               max_norm=1.0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in range(1, steps + 1):
            loss = 3.5 - 0.6 * (1 - np.exp(-s / 20))
            if diverge and s > warm + 5:
                loss = 3.5 + 0.1 * (s - warm)
            if no_drop:
                loss = 3.5 + 0.001 * s
            fh.write(json.dumps({"step": s, "grad_norm": 0.8 if not clipped else 2.0, "max_norm": max_norm,
                                 "clipped": bool(clipped), "lrs": [0.0025], "methods": ["muon"], "loss": loss + 0.01,
                                 "ce": loss, "micro_batches": 32}) + "\n")


def write_pin(w: World, scen: dict, root: str, run: dict) -> None:
    for set_name in run.get("pin_sets", []):
        idx, hs, ce0 = w.base_ce(set_name)
        z = np.load(os.path.join(root, "eval_main", f"ckpt__{set_name}.npz"))
        f_plus = float((z["ce_sum"] / z["n_tok"]).mean() - ce0.mean())
        for v, side in (("pinned", f_plus), ("selpin", f_plus), ("reflect+pinned", scen["reflect_F"]),
                        ("reflect+selpin", scen["reflect_F"])):
            shift = scen["pin_shift"]["selpin" if v.endswith("selpin") else "pinned"]
            write_eval(os.path.join(root, "eval_pin", f"{v}__{set_name}.npz"), idx, hs, w.run_ce(set_name, side + shift, 0.02))


def _books_F(scen: dict, pair, key: str, sub: str) -> float:
    base = (scen.get("books_F_by_pair") or {}).get(str(pair), scen["books_F"]).get(key, 0.008)
    pa = str(pair).replace("_cfg", "")
    return base * (scen["books_B_scale"].get(pa, {}).get(key, 1.0) if sub == "R2" else 1.0)


def _interim(w: World, scen: dict, root: str, run: dict, F_of_step) -> None:
    """64-window heads at the interim steps; F_of_step(set, step) -> F (theta_0 at step 0)."""
    for set_name in run.get("interim_sets", []):
        idx, hs, ce0 = w.base_ce(set_name)
        for step in run["interim_steps"]:
            ce = ce0[:HEAD].copy() if step == 0 else w.run_ce(set_name, F_of_step(set_name, step), 0.02, HEAD)
            write_eval(os.path.join(root, "interim", f"step{int(step):03d}__{set_name}.npz"), idx[:HEAD], hs[:HEAD], ce)


def _write_run(w: World, scen: dict, plan: dict, run: dict, state: dict, stage_dir: str) -> None:
    rid, role = run["id"], run["role"]
    code = plan["g0"]["code_stream"]
    root = os.path.abspath(os.path.join("outputs", stage_dir, rid, "m2"))
    os.makedirs(root, exist_ok=True)
    st = {"status": "done", "root": root, "final": os.path.join(root, "model_final.pt")}
    if role == "negctrl":
        if scen.get("negctrl_audit_fails", True):
            st["audit_ok"] = False
        else:
            st.update(status="failed", audit_ok=True, error="negative control: the audit PASSED but had to fail")
        state[rid] = st
        return
    if run.get("subset_spec"):
        st["audit_ok"] = rid not in scen.get("audit_fail", [])
    if role == "sgd":
        i = int(run["lr_index"])
        sc = scen["sgd"][i]
        _train_log(os.path.join(root, "trainlog", "train_steps.jsonl"), run["steps"], run["warmup"],
                   diverge=i in scen.get("sgd_diverge", []), no_drop=i in scen.get("sgd_no_loss_drop", []),
                   clipped=i in scen.get("sgd_clipped", []), max_norm=1.0 if i in scen.get("sgd_clipped", []) else None)
        for set_name in ("dclm", "german"):
            idx, hs, ce0 = w.base_ce(set_name)
            for step in sorted(set(run["interim_steps"]) | set(run.get("full_steps", []))):
                F = 0.0 if step == 0 else (-sc["gain"][str(step)] if set_name == "german" else sc["dclm_F"][str(step)])
                full = ce0.copy() if step == 0 else w.run_ce(set_name, F, scen["sgd_full_sd"], W)
                if step in run.get("full_steps", []):
                    write_eval(os.path.join(root, "interim", f"step{int(step):03d}__{set_name}__full.npz"), idx, hs, full)
                head = ce0[:HEAD].copy() if step == 0 else (full[:HEAD] if step in run.get("full_steps", [])
                                                           else w.run_ce(set_name, F, scen["sgd_interim_sd"], HEAD))
                write_eval(os.path.join(root, "interim", f"step{int(step):03d}__{set_name}.npz"), idx[:HEAD], hs[:HEAD], head)
        state[rid] = st
        return
    _train_log(os.path.join(root, "trainlog", "train_steps.jsonl"), run["steps"], run["warmup"],
               diverge=rid in scen.get("diverge", []), max_norm=1.0)
    for set_name in run.get("eval_sets", []):
        key = "code" if set_name == code else set_name
        if role == "full" and run["base"] == "moe_b1":
            F, sd = (scen["full_moe_F_dclm"] if set_name == "dclm" else 0.01), scen["full_sd"]
        elif role == "full" and run["base"] == "nestor":
            F, sd = (scen["nestor_full_F_dclm"][str(run["seed"])] if set_name == "dclm" else -0.2), scen["full_sd"]
        elif role == "aa" and run.get("target") == "german":
            F = scen["aa_F"].get(key, 0.008) * (scen["aa_B_scale"].get(key, 1.0) if run.get("subset") == "R2" else 1.0)
            if set_name == "german":
                F = -scen["aa_gain"]["48"]
            sd = scen["aa_sd"]
        elif role == "aa":
            pair = str(run.get("pair")) + ("_cfg" if rid.endswith("_cfg") and str(run.get("pair")) + "_cfg" in
                                           (scen.get("books_F_by_pair") or {}) else "")
            F = -scen["aa_gain"]["48"] if set_name == "books" else _books_F(scen, pair, key, run.get("subset"))
            sd = scen["books_sd"]
        elif role == "aa_nestor":
            F, sd = (-0.15 if set_name == "books" else scen["nestor_aa_F"].get(key, 0.008)), scen["aa_sd"]
        elif role == "eall":
            F, sd = scen["eall_F"], scen["full_sd"]
        elif role == "dclm_cont":
            F, sd = scen["dclm_cont_F"], scen["full_sd"]
        else:
            F, sd = 0.01, 0.02
        idx, hs, _ = w.base_ce(set_name)
        write_eval(os.path.join(root, "eval_main", f"ckpt__{set_name}.npz"), idx, hs, w.run_ce(set_name, F, sd))
    for set_name in run.get("reflect_sets", []):
        idx, hs, _ = w.base_ce(set_name)
        write_eval(os.path.join(root, "eval_reflect", f"reflect__{set_name}.npz"), idx, hs, w.run_ce(set_name, scen["reflect_F"], 0.02))
    write_pin(w, scen, root, run)
    _interim(w, scen, root, run, lambda s, step: -scen["aa_gain"][str(step)])
    state[rid] = st


def _base_files(w: World, plan: dict, overwrite: bool) -> None:
    for base, files in plan["base_eval_files"].items():
        for set_name, p in files.items():
            if overwrite or not os.path.exists(p):
                idx, hs, ce = w.base_ce(set_name)
                write_eval(p, idx, hs, ce)


def simulate_g1(plan: dict, scen: dict, world: World | None = None) -> World:
    w = world or World(scen)
    state = {}
    _base_files(w, plan, overwrite=plan["kind"] == "g1")
    for run in plan["runs"]:
        if run["id"] not in scen.get("missing", []):
            _write_run(w, scen, plan, run, state, plan["kind"])
    os.makedirs(os.path.dirname(os.path.abspath(plan["state"])), exist_ok=True)
    json.dump(state, open(plan["state"], "w", encoding="utf-8"), indent=1)
    if plan["kind"] == "g1":
        c = float(scen["theta0_c"])
        t0 = plan.get("theta0_checks") or os.path.join(os.path.dirname(plan["queue"]), "theta0_checks.json")
        json.dump({"MOCK": True, "sgd": {"c": c, "grid": [c / 3, c, 3 * c], "rule": "MOCK {c/3, c, 3c}"},
                   "ns5_scale_check": {"per_scale": {"0.25": {"median": 0.01, "max": 0.03}, "0.5": {"median": 0.006, "max": 0.02},
                                                     "2": {"median": 0.004, "max": 0.01}}},
                   "d_muon_unit_test": {"max_rel_err": 2e-6, "pass": True}}, open(t0, "w", encoding="utf-8"), indent=1)
        gc = plan.get("gate_config") or {}
        pt = (gc.get("pipeline_tests") or {}).get("junit")
        if pt and scen.get("pipeline") is not None:
            p = scen["pipeline"]
            os.makedirs(os.path.dirname(os.path.abspath(pt)), exist_ok=True)
            open(pt, "w", encoding="utf-8").write(
                f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="{p["tests"]}" '
                f'failures="{p["failures"]}" errors="{p["errors"]}" skipped="{p["skipped"]}"></testsuite></testsuites>')
        rh = gc.get("routing_health")
        if rh and scen.get("routing_health") is not None:
            os.makedirs(os.path.dirname(os.path.abspath(rh)), exist_ok=True)
            json.dump(scen["routing_health"], open(rh, "w", encoding="utf-8"))
        if scen.get("h100_F_dclm") is not None and plan.get("h100_duplicate_file"):
            hf = plan["h100_duplicate_file"]
            os.makedirs(os.path.dirname(os.path.abspath(hf)), exist_ok=True)
            json.dump({"F_dclm": scen["h100_F_dclm"], "MOCK": True}, open(hf, "w", encoding="utf-8"))
    return w


def simulate_g2(plan: dict, scen: dict, world: World) -> World:
    w = world
    state = {}
    _base_files(w, plan, overwrite=False)
    for run in plan["runs"]:
        if run["id"] not in scen.get("missing", []):
            _write_run(w, scen, plan, run, state, "g2")
    os.makedirs(os.path.dirname(os.path.abspath(plan["state"])), exist_ok=True)
    json.dump(state, open(plan["state"], "w", encoding="utf-8"), indent=1)
    snr = plan["snr"]["out"]
    os.makedirs(os.path.dirname(os.path.abspath(snr)), exist_ok=True)
    json.dump({"MOCK": True, "windows": 64, "train_batch_windows": 64,
               "summary": {"moe": {"units": 288, "median_snr_train": 0.8, "q25": 0.3, "q75": 2.1, "share_snr_train_below_1": 0.55}},
               "per_unit": {}}, open(snr, "w", encoding="utf-8"), indent=1)
    return w


def main(argv=None) -> int:
    import g1_gate
    import g1g2_plan
    import g2_gate
    import gate_common as GC
    import gate_rules_v13 as GR

    ap = argparse.ArgumentParser(description="SIMULATED G1/G2 outputs and mock gate records (spec 1.3.0)")
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--g0-record", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    g0 = os.path.abspath(a.g0_record)
    cfgp = os.path.abspath(a.config)
    lookup = os.path.abspath(a.lookup)
    os.makedirs(a.root, exist_ok=True)
    cfg = g1g2_plan.load_config(cfgp)
    cfg["g0_probe_configs"] = {k: (v if os.path.isabs(v) else os.path.join(os.path.dirname(cfgp), v))
                               for k, v in cfg["g0_probe_configs"].items()}
    os.chdir(a.root)
    scen = copy.deepcopy(BASE_SCENARIO)
    p1 = g1g2_plan.build_g1(cfg, g0, "g1")
    world = simulate_g1(p1, scen)
    r1 = g1_gate.gate(p1, None, "g1/plan_g1.json")
    r1["MOCK"] = "SIMULATED outputs (mock_g1g2.py BASE_SCENARIO) on the MOCK G0 record -- not a measurement"
    h1 = GC.finalize_record(r1, "g1/gate_record_G1_MOCK.json")
    p2 = g1g2_plan.build_g2(cfg, g0, "g1/gate_record_G1_MOCK.json", "g2")
    simulate_g2(p2, scen, world)
    r2 = g2_gate.gate(p2, json.load(open("g1/gate_record_G1_MOCK.json", encoding="utf-8")), GR.Lookup(lookup),
                      plan_path="g2/plan_g2.json", g1_path="g1/gate_record_G1_MOCK.json", lookup_path=lookup)
    r2["MOCK"] = r1["MOCK"]
    h2 = GC.finalize_record(r2, "g2/gate_record_G2_MOCK.json")
    print(json.dumps({"g1": r1["decision"], "g1_sha256": h1, "g2_freeze": r2["freeze"], "g2_outcome": r2["outcome"],
                      "g2_sha256": h2}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
