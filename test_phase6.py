"""CPU tests of the phase-6 tooling. Run in the installed layout (chain6b install, or a staging copy):

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 python -m pytest test_phase6.py -q -rA
(Python 3.12 on the host: PYTHONPATH must contain sitecustom; TEMP/TMP = .tmp.)
Torch-free part: E-ALL spec, G0 decisions of the REAL registered record, G1/G2 gates in a real-G0-shaped simulated world
(German infeasible), re-run comparison, pins, install/drift/freeze, dose generator, NESTOR-books feasibility, confirm
plan, forecasts, run_continual/stageB_windows pins, PowerShell parse. Torch part (tiny models, CPU or
NESTOR_TEST_DEVICE=cuda): g0_probe patch value equivalence with a fixed gradient; assumption audit.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import chain6b_util as CU  # noqa: E402
import crossover as CX  # noqa: E402
import g0_rerun_compare as GRC  # noqa: E402
import g1g2_plan as GP  # noqa: E402
import pins_check as PC  # noqa: E402

REG_RECORD = os.path.join(HERE, "phase6", "gate_record_G0_registered.json")
if not os.path.exists(REG_RECORD):
    REG_RECORD = os.path.join(HERE, "g0", "gate_record_registered.json")
MOCK_CFG = os.path.join(HERE, "g1g2_config_MOCK.json")
LOOKUP = os.path.join(HERE, "signflip_lookup_v1.csv")
PS = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _tables(L=4, N=8, seed=0, corpora=("dclm", "code", "math", "books", "german")):
    """Synthetic RegTables with router-popularity correlation (code vs math anti-correlated in half the modules)."""
    rng = np.random.default_rng(seed)
    pop = rng.gamma(2.0, 1.0, (L, N))
    sel, est = {}, {}
    for i, c in enumerate(corpora):
        tilt = np.exp(0.8 * rng.standard_normal((L, N)) * (1 if c in ("code", "books") else -0.5))
        base = pop * tilt
        sel[c] = base * np.exp(0.05 * rng.standard_normal((L, N)))
        est[c] = base * np.exp(0.05 * rng.standard_normal((L, N)))
    keys = [[f"moe:b{l}:l0:e{m}" for m in range(N)] for l in range(L)]
    return CX.RegTables("moe", [f"moe:b{l}" for l in range(L)], keys, sel, est, {})


# ============================================================================= G0 decisions / gates
def test_eall_spec_trains_every_expert(monkeypatch):
    t = _tables()
    monkeypatch.setattr(CX, "reg_load_dir", lambda d, fam, **kw: t)
    spec = GP.all_expert_spec("unused", "moe")
    assert spec["mode"] == "train" and len(spec["select"]) == t.L
    for f in spec["select"]:
        assert f["expert"] == list(range(t.N))           # phase 5 gave [1, 1, ..., 1] (one expert per layer)


def test_g0_decisions_of_the_real_registered_record():
    d = GP.g0_decisions(json.load(open(REG_RECORD, encoding="utf-8")))
    assert d["pivot"] is False and d["nestor_g0_1"] is False and d["nestor_g0_5"] is True
    assert d["c1_feasible_targets"] == ["books"] and d["c1_registered_targets"] == ["german", "books"]
    assert d["P_bar_source"] == "moe_b1|books|code" and math.isclose(d["P_bar_c1_moe_b1"], 0.9890344727164835, rel_tol=1e-12)
    assert d["f"] == 0.5 and d["P_bar_moe_b1_german"] is None and d["code_stream"] == "code"
    assert d["beta0"] == {"moe_b1": 0.25, "moe_b2": 0.25}


@pytest.fixture(scope="session")
def worlds(tmp_path_factory):
    """mock_g0 tree (all targets feasible) and a copy whose gate record carries the REAL G0 decisions."""
    base = tmp_path_factory.mktemp("g0w")
    # a torch-free child: power_sim_v1 calls numpy LAPACK, which aborts a process that has imported torch (host clash)
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    p = subprocess.run([sys.executable, os.path.join(HERE, "mock_g0.py"), "--out", str(base / "A"), "--draws", "200"],
                       cwd=HERE, env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    shutil.copytree(base / "A", base / "B")
    rp = base / "B" / "g0" / "gate_record.json"
    txt = open(rp, encoding="utf-8").read().replace(json.dumps(str(base / "A"))[1:-1], json.dumps(str(base / "B"))[1:-1])
    R = json.loads(txt)
    for k, c in R["gates"]["G0-1"]["cells"].items():
        if "|german|" in k:
            c.update(feasible=False, rung=None, f=None, median=None, tol=None, matched=None)
    R["gates"]["G0-1"].update(moe_targets={"german": False, "books": True}, nestor_targets={"german": False}, nestor_pass=False)
    json.dump(R, open(rp, "w", encoding="utf-8"), indent=1)
    return {"A": str(base / "A" / "g0" / "gate_record.json"), "B": str(rp)}


def _cfg():
    cfg = GP.load_config(MOCK_CFG)
    cfg["g0_probe_configs"] = {k: (v if os.path.isabs(v) else os.path.join(HERE, v)) for k, v in cfg["g0_probe_configs"].items()}
    return cfg


def _g1g2(rec, tmp, monkeypatch, mod=None):
    import g1_gate as G1
    import g2_gate as G2
    import gate_common as GC
    import gate_rules_v13 as GR
    import mock_g1g2 as M

    os.makedirs(tmp, exist_ok=True)
    monkeypatch.chdir(tmp)
    cfg = _cfg()
    scen = copy.deepcopy(M.BASE_SCENARIO)
    for k, v in (mod or {}).items():
        scen[k] = v
    p1 = GP.build_g1(cfg, rec, "g1")
    w = M.simulate_g1(p1, scen)
    r1 = G1.gate(p1, None, "g1/plan_g1.json")
    GC.finalize_record(r1, "g1/gate_record_G1.json")
    p2 = GP.build_g2(cfg, rec, "g1/gate_record_G1.json", "g2")
    M.simulate_g2(p2, scen, w)
    r2 = G2.gate(p2, json.load(open("g1/gate_record_G1.json", encoding="utf-8")), GR.Lookup(LOOKUP),
                 plan_path="g2/plan_g2.json", g1_path="g1/gate_record_G1.json", lookup_path=LOOKUP)
    GC.finalize_record(r2, "g2/gate_record_G2.json")
    return p1, r1, p2, r2


def test_real_shaped_world_books_targets_and_pbar_from_books(worlds, tmp_path, monkeypatch):
    p1, r1, p2, r2 = _g1g2(worlds["B"], tmp_path / "w", monkeypatch)
    assert len([r for r in p1["runs"] if r["base"] == "nestor"]) == 3        # NESTOR G1 runs although C1-N fails G0-1
    assert r1["decision"]["outcome"] == "PROCEED" and r1["decision"]["c1"] == "FEASIBLE"
    f = r2["freeze"]
    assert r2["outcome"] == "COMPLETE" and f["C1_targets_per_base"] == {"moe_b1": ["books"], "moe_b2": ["books"]}
    assert f["c1_draws_per_base"] == 2 * f["n_d"] and f["c1_draw_indices"] == list(range(1, 2 * f["n_d"] + 1))  # P-13
    assert r2["gates"]["G2-4"]["outcome"] == "pass" and r2["gates"]["G2-4"]["action"] == "books cells kept"
    assert f["P_bar_source"] == f"moe_b1|books|{f['code_stream']}"
    cell = json.load(open(worlds["B"], encoding="utf-8"))["gates"]["G0-1"]["cells"][f["P_bar_source"]]
    assert math.isclose(r2["gates"]["G2-1"]["values"]["Pbar"], cell["median"], rel_tol=1e-12)
    assert math.isclose(f["P_bar"], cell["median"], rel_tol=1e-12)


def test_real_shaped_world_books_failure_pivots_c1(worlds, tmp_path, monkeypatch):
    _, r1, _, r2 = _g1g2(worlds["B"], tmp_path / "w", monkeypatch, {"books_F": {"code": 0.001, "math": 0.001}})
    assert r1["gates"]["G1-2"]["outcome"] == "PIVOT" and r1["decision"]["c1"] == "PIVOT"   # no replacement target left (O-4)
    f = r2["freeze"]
    assert f["C1_confirmatory"] is False and f["holm_family"] == [] and f["m"] == 0 and f["pivot"] is True
    assert f["n_d"] is None and f["c1_draws_per_base"] is None
    assert r2["gates"]["G2-4"]["outcome"] == "not_applicable"
    assert r1["decision"]["NESTORB_label"] == "exploratory" and r2["outcome"] == "COMPLETE"   # the other blocks proceed


def test_baseline_world_uses_the_registered_books_targets(worlds, tmp_path, monkeypatch):
    # v1.3 registers books as the C1 target of both MoE bases (German infeasible at the real G0-1, decision O-4): a world
    # in which German would be feasible does not reopen the German cells
    _, r1, _, r2 = _g1g2(worlds["A"], tmp_path / "w", monkeypatch)
    assert r1["decision"]["outcome"] == "PROCEED"
    assert r2["freeze"]["C1_targets_per_base"] == {"moe_b1": ["books"], "moe_b2": ["books"]}
    assert r2["freeze"]["P_bar_source"].startswith("moe_b1|books|")


# ============================================================================= re-run comparison, pins
def test_rerun_compare_decisions_not_numbers(tmp_path):
    reg = json.load(open(REG_RECORD, encoding="utf-8"))
    r = GRC.compare(reg, copy.deepcopy(reg))
    assert r["decisions_equal"] and r["n_decisions"] > 40
    num = copy.deepcopy(reg)
    num["gates"]["G0-1"]["cells"]["moe_b1|books|code"]["median"] += 0.01
    num["gates"]["G0-7"]["moe_b1"]["D"] -= 0.02
    r = GRC.compare(reg, num)
    assert r["decisions_equal"] and r["numbers"]["G0-1|moe_b1|books|code|median"]["abs_diff"] > 0.009
    dec = copy.deepcopy(reg)
    dec["gates"]["G0-1"]["cells"]["moe_b2|books|code"]["f"] = 1 / 3
    dec["gates"]["G0-7"]["moe_b2"]["beta0"] = 0.3
    r = GRC.compare(reg, dec)
    assert not r["decisions_equal"] and {d["decision"] for d in r["differences"]} == {"G0-1|moe_b2|books|code|f", "G0-7|moe_b2|beta0"}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    json.dump(reg, open(a, "w")), json.dump(dec, open(b, "w"))
    assert GRC.main(["--registered", str(a), "--rerun", str(a)]) == 0 and GRC.main(["--registered", str(a), "--rerun", str(b)]) == 1


def test_pins_make_verify_full_pending_and_g0_crosscheck(tmp_path):
    root = tmp_path
    (root / "ck").mkdir()
    for n in ("a.pt", "b.pt"):
        (root / "ck" / n).write_bytes(os.urandom(4096))
    pins = root / "ckpt_pins.json"
    assert PC.main(["make", "--root", str(root), "--out", str(pins), "moe_b1=ck/a.pt", "nestor=ck/b.pt", "moe_b3=pending"]) == 0
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins), "--full"]) == 0
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins), "--require", "moe_b3"]) == 2
    rec = {"checkpoints": {"moe_b1": ["ck/a.pt"], "nestor": ["ck/b.pt"]}}
    json.dump(rec, open(root / "rec.json", "w"))
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins), "--g0-record", str(root / "rec.json")]) == 0
    json.dump({"checkpoints": {"moe_b1": ["ck/b.pt"]}}, open(root / "rec2.json", "w"))
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins), "--g0-record", str(root / "rec2.json")]) == 1
    data = bytearray((root / "ck" / "a.pt").read_bytes()); data[0] ^= 1
    (root / "ck" / "a.pt").write_bytes(bytes(data))                 # same size, different content
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins)]) == 0
    assert PC.main(["verify", "--root", str(root), "--pins", str(pins), "--full"]) == 1
    cfg = root / "g0_b3.json"
    json.dump({"checkpoint": "PIN_ME:x"}, open(cfg, "w"))
    assert PC.main(["pin-config", "--root", str(root), "--pins", str(pins), "--name", "moe_b3", "--config", str(cfg)]) == 2
    (root / "ck" / "c.pt").write_bytes(b"x" * 10)
    assert PC.main(["pin", "--root", str(root), "--pins", str(pins), "moe_b3", "ck/c.pt"]) == 0
    assert PC.main(["pin-config", "--root", str(root), "--pins", str(pins), "--name", "moe_b3", "--config", str(cfg)]) == 0
    assert json.load(open(cfg))["checkpoint"] == "ck/c.pt"


# ============================================================================= install, drift, stages, freeze
def _fake_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    frozen = {}
    for src, dst, kind in CU.INSTALL:
        sp = root / src
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(f"# new {src}\n")
        if kind in ("replace_frozen", "replace"):
            dp = root / dst
            dp.parent.mkdir(parents=True, exist_ok=True)
            dp.write_text(f"# frozen {dst}\n")
            if kind == "replace_frozen":
                frozen[os.path.basename(dst)] = _sha(dp)
    json.dump({"files": frozen}, open(root / "g0_code_freeze.json", "w"))
    return root


def test_install_backs_up_frozen_files_records_hashes_and_detects_drift(tmp_path):
    root = _fake_root(tmp_path)
    assert CU.install(str(root)) == 0
    rec = json.load(open(root / CU.CODE_RECORD))
    assert (root / "frozen_g0" / "run_continual.py").read_text() == "# frozen run_continual.py\n"
    assert (root / "run_continual.py").read_text() == "# new phase6/run_continual.py\n"
    assert all(r["action"] in ("copied", "kept (seed: destination exists)") for r in rec["install"])
    assert CU.codecheck(str(root)) == 0
    (root / "hooks").mkdir(exist_ok=True)
    (root / "hooks" / "after_stageB.ps1").write_text("exit 0\n")              # hooks may be added while running
    assert CU.codecheck(str(root)) == 0
    (root / "g2_gate.py").write_text("# edited\n")
    assert CU.codecheck(str(root)) == 21
    assert CU.install(str(root), reinstall=True) == 0 and CU.codecheck(str(root)) == 0
    assert json.load(open(root / CU.CODE_RECORD)).get("previous_record")


def test_install_refuses_a_non_frozen_g0_file(tmp_path):
    root = _fake_root(tmp_path)
    (root / "eval_windows.py").write_text("# someone edited the frozen file\n")
    assert CU.install(str(root)) == 22


def test_stage_checks_and_freeze_record(tmp_path):
    root = tmp_path
    (root / "results").mkdir()
    keys = [f"B|{e}|before|{t}" for e in CU.STAGE_B_MODELS for t in CU.STAGE_B_CORPORA + ["dclm"]]
    keys += [f"B|{e}|after|{c}|{s}" for c in CU.STAGE_B_CORPORA for e in CU.STAGE_B_MODELS for s in ("target", "dclm")]
    with open(root / "results" / "continual.jsonl", "w") as fh:
        for k in keys[:-1]:
            fh.write(json.dumps({"key": k}) + "\n")
    assert CU.check_stage(str(root), "B") == 20
    with open(root / "results" / "continual.jsonl", "a") as fh:
        fh.write(json.dumps({"key": keys[-1]}) + "\n")
    assert CU.check_stage(str(root), "B") == 0 and CU.check_stage(str(root), "C") == 20
    plan = _freeze_world(root)
    spec = json.load(open(os.path.join(HERE, "g1g2_spec.json"), encoding="utf-8"))
    assert spec["freeze"]["parameters"] == CU.FREEZE_PARAMETERS and len(CU.FREEZE_PARAMETERS) == 30
    shutil.copy(os.path.join(HERE, "g1g2_spec.json"), root / "g1g2_spec.json")      # freeze-write checks the names
    shutil.copy(os.path.join(HERE, "power_sim_v1.py"), root / "power_sim_v1.py")    # freeze check 2
    json.dump({"files": CU.code_set(str(root))}, open(root / CU.CODE_RECORD, "w"))  # the install record of this root
    # this test is about the 30 parameters: the hash-list entries a tmp root cannot have are waived (MOCK only)
    a = type("A", (), dict(plan="confirm/plan_confirm.json", queue="confirm/queue_confirm.json", prereg="prereg.md",
                           g0=None, g1=None, g2=None, approved_by="MOCK", notes=None, out="FREEZE.json", force=False,
                           waive=[], mock_waive_missing=True,
                           forecasts_csv="freeze/forecasts_v13.csv", forecasts_meta="freeze/forecasts_v13_meta.json",
                           nestor_books="results/nestor_books_feasibility.json", dose="results/dose/dose_record.json",
                           b3="g0/gate_record_b3.json", decisions="decisions_v13_points.md", envelope_extension=0.0,
                           envelope_decision=None))()
    assert CU.freeze_write(str(root), a) == 0
    doc = json.load(open(root / "FREEZE.json", encoding="utf-8"))
    prm = doc["parameters"]
    assert doc["parameter_order"] == CU.FREEZE_PARAMETERS and sorted(prm) == sorted(CU.FREEZE_PARAMETERS)
    assert all(set(v) == {"value", "source"} for v in prm.values())
    assert prm["n_d"]["value"] == 3 and prm["holm_family"]["value"] == ["C1"] and prm["tau (if used)"]["value"] is None
    assert prm["NESTORB_run"]["value"] is False                                 # signal, but the feasibility record says no
    assert prm["moe_b3 status and beta0_3"]["value"]["status"] == "pending"
    assert prm["dose-response f_low and accepted draws"]["value"] == {
        "f_low": 2 / 3, "accepted": {"moe_b1": [{"d": 14, "seed": 5114, "data_seed": 7114, "P": 0.38}]}}
    _, _, h = CU._forecast_io(str(root)).verify(str(root / "freeze" / "forecasts_v13.csv"),
                                                str(root / "freeze" / "forecasts_v13_meta.json"))
    assert prm["forecast file SHA-256 (csv, meta)"]["value"] == {"csv": h["csv_sha256"], "meta": h["meta_sha256"]}
    assert prm["envelope extension (decision P-6; 0 if none)"]["value"] == {"gpu_h": 0.0, "decision": None}
    assert prm["decisions O-1 to O-9 and P-1 to P-13"]["value"][0]["path"] == "decisions_v13_points.md"
    assert CU.freeze_verify(str(root), "FREEZE.json") == 0
    with pytest.raises(SystemExit):
        CU.freeze_write(str(root), a)                                          # a freeze is final
    d2 = json.loads(json.dumps(doc))                                            # an edited parameter, content hash redone
    d2["parameters"]["n_d"]["value"] = 5
    d2["sha256"] = hashlib.sha256(CU.canonical({k: v for k, v in d2.items() if k not in ("created", "sha256")})
                                  .encode()).hexdigest()
    json.dump(d2, open(root / "FREEZE_edit.json", "w", encoding="utf-8"))
    assert CU.freeze_verify(str(root), "FREEZE_edit.json") == 23               # spec check 3: re-derived from the source
    dose_b = (root / "results" / "dose" / "dose_record.json").read_bytes()
    (root / "results" / "dose" / "dose_record.json").write_bytes(dose_b.replace(b"5114", b"5115"))
    assert CU.freeze_verify(str(root), "FREEZE.json") == 23                    # a source record changed after the freeze
    (root / "results" / "dose" / "dose_record.json").write_bytes(dose_b)
    assert CU.freeze_verify(str(root), "FREEZE.json") == 0
    json.dump(dict(plan, envelope_decision_required="cut order exhausted: 131.2 GPU-h > 120"),
              open(root / "confirm" / "plan_confirm.json", "w"))
    a.out = "FREEZE_env.json"
    with pytest.raises(SystemExit, match="envelope decision"):
        CU.freeze_write(str(root), a)                                          # P-6: the freeze waits for the operator
    a.envelope_extension = 12.0
    with pytest.raises(SystemExit, match="envelope-decision"):
        CU.freeze_write(str(root), a)
    a.envelope_decision = "logs/decisions.log, P-6"
    assert CU.freeze_write(str(root), a) == 0 and CU.freeze_verify(str(root), "FREEZE_env.json") == 0
    (root / "g0").mkdir(exist_ok=True)
    b3 = {"gates": {"G0-1": {"cells": {"moe_b3|books|pycode": {"feasible": True}}}, "G0-5": {"per_base": {}}, "G0-7": {}}}
    json.dump(b3, open(root / "g0" / "gate_record_b3.json", "w"))
    a.out = "FREEZE_b3.json"
    with pytest.raises(SystemExit, match="incomplete"):
        CU.freeze_write(str(root), a)                                          # books cell without G0-5 / G0-7
    b3["gates"]["G0-5"]["per_base"]["moe_b3"] = {"pass": True}
    b3["gates"]["G0-7"]["moe_b3"] = {"beta0": 0.5}
    json.dump(b3, open(root / "g0" / "gate_record_b3.json", "w"))
    assert CU.freeze_write(str(root), a) == 0 and CU.freeze_verify(str(root), "FREEZE_b3.json") == 0
    d3 = json.load(open(root / "FREEZE_b3.json", encoding="utf-8"))
    assert d3["parameters"]["moe_b3 status and beta0_3"]["value"] == {"status": "complete", "beta0_3": 0.5,
                                                                       "planned": False, "label": None}
    assert "b3" in d3["records"]
    (root / "confirm" / "queue_confirm.json").write_text('{"jobs": [1]}')
    assert CU.freeze_verify(str(root), "FREEZE.json") == 23


def _freeze_world(root):
    """Minimal freeze inputs: a plan with freeze_inputs and budget fields, NESTOR-books and dose records, a decision record
    and a valid forecast pair (forecast_io selftest files)."""
    (root / "confirm").mkdir()
    (root / "confirm" / "queue_confirm.json").write_text('{"jobs": []}')
    fi = {k: None for _, k in CU._FROM_PLAN}
    fi.update(holm_family=["C1"], m=1, C1_confirmatory=True, C1_targets_per_base={"moe_b1": ["books"]}, n_d=3,
              underpowered_flag=False, q=0.25, steps=96, f=1 / 3, code_stream="pycode", beta0_1=0.25, beta0_2=0.25,
              margin_limited={"moe_b1": False}, C2_confirmatory=False, C2_draws=4, s_star=24, sgd_lr=1 / 60,
              C2_matching_set=["dclm"], NESTORB_signal=True, NESTORB_label="exploratory", C3N_label="estimation")
    plan = {"queue": "confirm/queue_confirm.json", "runs": [], "sha256": "abc", "freeze_inputs": fi,
            "estimate_total_gpu_h": 72.4, "estimate_with_contingency_gpu_h": 86.0, "estimate_spent_gpu_h": 2.75,
            "envelope_gpu_h": 120.0, "contingency_after_cuts": 0.15, "cut_order_applied": [], "cut_order_restored": [],
            "envelope_decision_required": None}
    json.dump(plan, open(root / "confirm" / "plan_confirm.json", "w"))
    for d_ in ("g0", "g1", "g2"):                                            # R5: the gate records themselves
        (root / d_).mkdir(exist_ok=True)
    json.dump({"outcome": "COMPLETE", "freeze": dict(fi, g0_7r="not_needed")}, open(root / "g2" / "gate_record_G2.json", "w"))
    json.dump({"decision": {k: fi[k] for k in ("steps", "q", "C1_confirmatory", "NESTORB_signal", "NESTORB_label", "C3N_label")}},
              open(root / "g1" / "gate_record_G1.json", "w"))
    json.dump({"decisions": {"code_stream": {"chosen": fi["code_stream"]}},
               "gates": {"G0-7": {b: {"beta0": 0.25, "margin_limited": False} for b in ("moe_b1", "moe_b2")}}},
              open(root / "g0" / "gate_record.json", "w"))
    (root / "prereg.md").write_text("prereg v1.3\n")
    (root / "logs").mkdir()
    json.dump({"files": {}}, open(root / CU.CODE_RECORD, "w"))
    (root / "results" / "dose").mkdir(parents=True)
    json.dump({"feasible": False, "decision": "infeasible"}, open(root / "results" / "nestor_books_feasibility.json", "w"))
    json.dump({"f_low": 2 / 3, "bases": {"moe_b1": {"accepted": [{"d": 14, "seed": 5114, "data_seed": 7114,
                                                                  "predicted": {"P": 0.38}}]}}},
              open(root / "results" / "dose" / "dose_record.json", "w"))
    (root / "decisions_v13_points.md").write_text("P-1 to P-14 (test copy)\n")
    shutil.copy(os.path.join(HERE, "forecast_io.py"), root / "forecast_io.py")
    CU._forecast_io(str(root))._selftest(str(root / "freeze"))
    return plan


def _git_index(repo, files: dict[str, bytes]) -> None:
    """A git index (version 2) for the given worktree files, written by hand (blob sha1 of the LF content)."""
    import struct
    ents = []
    for name in sorted(files):
        fp = os.path.join(repo, *name.split("/"))
        st = os.stat(fp)
        blob = files[name]
        sha1 = hashlib.sha1(b"blob %d\0" % len(blob) + blob).digest()
        nb = name.encode()
        e = struct.pack(">10I", int(st.st_ctime), 0, int(st.st_mtime), 0, 0, 0, 0o100644, 0, 0, st.st_size)
        e += sha1 + struct.pack(">H", len(nb)) + nb
        e += b"\0" * (8 - (len(e) % 8))
        ents.append(e)
    body = b"DIRC" + struct.pack(">II", 2, len(ents)) + b"".join(ents)
    os.makedirs(os.path.join(repo, ".git"), exist_ok=True)
    open(os.path.join(repo, ".git", "index"), "wb").write(body + hashlib.sha1(body).digest())


def test_repo_patched_files_without_git(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_bytes(b"x = 1\n")
    (repo / "src" / "b.py").write_bytes(b"y = 2\r\nz = 3\r\n")                    # autocrlf checkout of "y = 2\nz = 3\n"
    (repo / "run.py").write_bytes(b"print(1)\n")
    _git_index(str(repo), {"src/a.py": b"x = 1\n", "src/b.py": b"y = 2\nz = 3\n", "run.py": b"print(1)\n"})
    monkeypatch.setenv("NESTOR_GIT", str(tmp_path / "no_git.exe"))
    monkeypatch.setattr(CU.shutil, "which", lambda *_: None)
    assert CU._index_patched(str(tmp_path)) == []
    pf, method = CU._git_patched(str(tmp_path))
    assert pf == [] and "index" in method
    os.utime(repo / "src" / "a.py", (1, 1))                                     # touched only: content check says equal
    assert CU._index_patched(str(tmp_path)) == []
    (repo / "src" / "a.py").write_bytes(b"x = 2\n")
    (repo / "run.py").unlink()
    assert CU._index_patched(str(tmp_path)) == ["repo/run.py", "repo/src/a.py"]


def _hl_root(tmp_path):
    """A freeze world with a small revision-3-style hash list (spec copy with its own freeze.hash_list)."""
    root = tmp_path / "hl"
    root.mkdir()
    plan = _freeze_world(root)
    spec = json.load(open(os.path.join(HERE, "g1g2_spec.json"), encoding="utf-8"))
    spec["freeze"]["hash_list"] = [
        {"category": "code", "paths": ["a.py", "power_sim_v1.py", "alt.py", "the things", "g0_code_freeze.json"]},
        {"category": "data", "paths": ["every evaluation, selection, training and replay bin used after the freeze "
                                       "(sha256 and size)"]},
        {"category": "records", "paths": ["g0/<base>/margin__probe.json", "g0/moe_b3 gate record",
                                          "the queue files of every post-freeze block", "forecasts_v13.csv"]}]
    json.dump(spec, open(root / "g1g2_spec.json", "w", encoding="utf-8"))
    json.dump({"entries": {"alt.py": {"files": ["alt.py|sub/alt.py"]}, "the things": {"globs": ["things/*.txt"]}}},
              open(root / CU.HASH_MAP_FILE, "w"))
    (root / "a.py").write_text("A = 1\n")
    (root / "sub").mkdir()
    (root / "sub" / "alt.py").write_text("B = 1\n")
    shutil.copy(os.path.join(HERE, "power_sim_v1.py"), root / "power_sim_v1.py")
    json.dump({"files": {"a.py": _sha(root / "a.py")}}, open(root / "g0_code_freeze.json", "w"))
    (root / "g0" / "moe_b1").mkdir(parents=True)
    (root / "g0" / "moe_b1" / "margin__probe.json").write_text("{}")
    (root / "data").mkdir()
    (root / "data" / "x.bin").write_bytes(b"\x01\x02")
    q = {"jobs": [{"id": "r1", "env": {"DCLM_DATA_PATH": "data/x.bin"},
                   "evals": [{"sets": ["dclm=data/y.bin:2048000", "code=data/x.bin:2048000"]}]}]}
    json.dump(q, open(root / "confirm" / "queue_confirm.json", "w"))
    json.dump({"files": CU.code_set(str(root))}, open(root / CU.CODE_RECORD, "w"))  # the install record of this root
    a = type("A", (), dict(plan="confirm/plan_confirm.json", queue="confirm/queue_confirm.json", prereg="prereg.md",
                           g0=None, g1=None, g2=None, approved_by="operator", notes=None, out="FREEZE.json", force=False,
                           forecasts_csv="freeze/forecasts_v13.csv", forecasts_meta="freeze/forecasts_v13_meta.json",
                           nestor_books="results/nestor_books_feasibility.json", dose="results/dose/dose_record.json",
                           b3="g0/gate_record_b3.json", decisions="decisions_v13_points.md", envelope_extension=0.0,
                           envelope_decision=None, waive=[], mock_waive_missing=False))()
    return root, a


def test_freeze_hash_list_revision3_resolution_checks_and_verify(tmp_path):
    root, a = _hl_root(tmp_path)
    r = str(root)
    assert CU.freeze_check(r, a) == 26                                          # things/*.txt and data/y.bin missing
    hl = CU.freeze_hash_list(r, CU._hl_args(r, a), a.plan, a.queue, {}, False, "T")
    st = {e["entry"]: e["status"] for e in hl["record"]["entries"]}
    assert st["alt.py"] == "ok" and st["the things"] == "missing" and st["g0/moe_b3 gate record"] == "optional_absent"
    assert st["g0/<base>/margin__probe.json"] == "ok" and st["a.py"] == "ok" and st["forecasts_v13.csv"] == "ok"
    assert [s for t, s in st.items() if t.startswith("every evaluation")] == ["missing"]
    alt = next(e for e in hl["record"]["entries"] if e["entry"] == "alt.py")
    assert [f["path"] for f in alt["files"]] == ["sub/alt.py"]
    assert CU.freeze_write(r, a) == 26 and not (root / "FREEZE.json").exists()  # nothing written
    (root / "things").mkdir()
    (root / "things" / "t.txt").write_text("t")
    (root / "data" / "y.bin").write_bytes(b"\x03")
    assert CU.freeze_check(r, a) == 0
    assert CU.freeze_write(r, a) == 0
    doc = json.load(open(root / "FREEZE.json", encoding="utf-8"))
    hl = doc["hash_list"]
    assert hl["hashed_utc"] == doc["hashed_utc"] and hl["spec"]["sha256"] == _sha(root / "g1g2_spec.json")
    assert hl["map_override"]["sha256"] == _sha(root / CU.HASH_MAP_FILE)
    bins = next(e for e in hl["entries"] if e["entry"].startswith("every evaluation"))
    assert [(f["path"], f["size"]) for f in bins["files"]] == [("data/x.bin", 2), ("data/y.bin", 1)]
    assert hl["checks"]["power_sim_v1"]["ok"] and hl["checks"]["g0_code_freeze"] == [
        {"path": "a.py", "frozen_sha256": _sha(root / "a.py"), "sha256": _sha(root / "a.py"), "status": "same"}]
    rec = root / "freeze" / "freeze_record.json"
    assert rec.read_bytes() == (root / "FREEZE.json").read_bytes()
    assert (root / "freeze" / "freeze_record.json.sha256").read_text().split()[0] == _sha(rec)
    assert _sha(rec) in (root / "logs" / "freeze.log").read_text()
    assert CU.freeze_verify(r, "FREEZE.json") == 0 and CU.freeze_verify(r, "freeze/freeze_record.json") == 0
    (root / "data" / "y.bin").write_bytes(b"\x04")
    assert CU.freeze_verify(r, "FREEZE.json") == 23                             # a data bin changed after the freeze
    (root / "data" / "y.bin").write_bytes(b"\x03")
    assert CU.freeze_verify(r, "FREEZE.json") == 0
    # a missing entry: only a logged decision removes it (the decision file is hashed into the record)
    (root / "things" / "t.txt").unlink()
    a.out = "FREEZE_w.json"
    assert CU.freeze_write(r, a) == 26
    a.waive = ["1.4=decisions/waiver.md"]
    with pytest.raises(SystemExit, match="does not exist"):
        CU.freeze_write(r, a)
    (root / "decisions").mkdir()
    (root / "decisions" / "waiver.md").write_text("2026-11-14: entry 1.4 removed\n")
    assert CU.freeze_write(r, a) == 0 and CU.freeze_verify(r, "FREEZE_w.json") == 0
    w = next(e for e in json.load(open(root / "FREEZE_w.json"))["hash_list"]["entries"] if e["id"] == "1.4")
    assert w["status"] == "waived" and w["waiver"]["decision"]["sha256"] == _sha(root / "decisions" / "waiver.md")
    assert (root / "freeze" / "FREEZE_w_record.json").exists()
    # check 1: a G0-frozen file changed -> STOP unless the replace_frozen install (frozen copy) logs the amendment
    a.waive, a.out = [], "FREEZE_c1.json"
    (root / "things" / "t.txt").write_text("t")
    orig = (root / "a.py").read_bytes()
    (root / "a.py").write_text("A = 2\n")
    assert CU.freeze_write(r, a) == 25
    (root / "frozen_g0").mkdir()
    (root / "frozen_g0" / "a.py").write_bytes(orig)
    json.dump({"files": CU.code_set(r), "install": [{"src": "phase6/a.py", "dst": "a.py", "kind": "replace_frozen"}]},
              open(root / CU.CODE_RECORD, "w"))
    assert CU.freeze_write(r, a) == 0
    c1 = json.load(open(root / "FREEZE_c1.json"))["hash_list"]["checks"]["g0_code_freeze"][0]
    assert c1["status"] == "amended" and c1["frozen_copy"] == "frozen_g0/a.py"
    # check 2: power_sim_v1.py must be d9c21f3b...
    a.out = "FREEZE_c2.json"
    (root / "power_sim_v1.py").write_text("# edited\n")
    assert CU.freeze_write(r, a) == 25
    shutil.copy(os.path.join(HERE, "power_sim_v1.py"), root / "power_sim_v1.py")
    # an entry without a rule is unresolved; the MOCK waiver needs --approved-by MOCK
    json.dump({"entries": {"alt.py": {"files": ["alt.py|sub/alt.py"]}}}, open(root / CU.HASH_MAP_FILE, "w"))
    hl = CU.freeze_hash_list(r, CU._hl_args(r, a), a.plan, a.queue, {}, False, "T")
    assert [e["status"] for e in hl["record"]["entries"] if e["entry"] == "the things"] == ["unresolved"]
    a.mock_waive_missing = True
    with pytest.raises(SystemExit, match="MOCK"):
        CU.freeze_write(r, a)


def test_env_snapshot_writes_pip_freeze_and_the_conda_list(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDA_EXE", str(tmp_path / "no_conda.exe"))
    monkeypatch.setattr(CU.shutil, "which", lambda *_: None)                  # no conda: the env's conda-meta records
    rc = CU.env_snapshot(str(tmp_path))
    pf = (tmp_path / "logs" / "env" / "pip_freeze.txt").read_text(encoding="utf-8")
    assert "pip freeze --all of" in pf and "==" in pf
    cl = tmp_path / "logs" / "env" / "conda_list_explicit.txt"
    if rc == 0:
        t = cl.read_text(encoding="utf-8")
        assert "@EXPLICIT" in t and "conda-meta" in t.splitlines()[0]
        assert any(("/numpy-" in line) and ("#" in line) for line in t.splitlines())
    else:
        assert rc == 3 and not cl.exists()                                     # interpreter outside a conda env


def test_default_hash_map_covers_the_registered_spec_list():
    spec = json.load(open(os.path.join(HERE, "g1g2_spec.json"), encoding="utf-8"))
    assert spec["revision"] == 3
    texts = [t for c in spec["freeze"]["hash_list"] for t in c["paths"]]
    rule = [t for t in texts if t not in CU.DEFAULT_HASH_MAP and not CU._PATHLIKE.match(t)]
    assert rule == [], rule                                                     # every entry has a resolution rule
    assert not [k for k in CU.DEFAULT_HASH_MAP if k not in texts]               # no stale rule


# ============================================================================= dose, NESTOR books, confirm, forecasts
def test_hash_map_pins_verified_copies_and_maps_the_analysis_script(tmp_path):
    """Decision (c): expect_sha256 pins a verified copy (other content -> mismatch, a STOP). Decision (d): entry 2.17
    resolves to analyze_confirm.py with its known-answer tests and simulator, all three in the INSTALL list."""
    r = str(tmp_path)
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    h = CU.sha256_file(str(tmp_path / "x.json"))
    how = {"files": ["y.json|x.json"], "expect_sha256": {"y.json|x.json": h}}
    out = CU._resolve(r, how, {})
    assert out["status"] == "ok" and out["files"] == ["x.json"]
    (tmp_path / "x.json").write_text('{"a": 1}', encoding="utf-8")
    assert CU._resolve(r, how, {})["status"] == "mismatch"
    e = CU.DEFAULT_HASH_MAP["the confirmatory analysis script(s)"]
    assert e["files"] == ["analyze_confirm.py", "analyze_confirm_ext.py", "test_analyze_confirm.py", "mock_confirm.py"]
    for f in e["files"]:
        assert os.path.isfile(os.path.join(HERE, f)), f
    assert set(e["files"]) <= {dst for _, dst, _ in CU.INSTALL}             # + analyze_confirm_ext.py
    m = CU.DEFAULT_HASH_MAP["gate_record_MOCK.json and the G1/G2 mock records"]
    assert set(m["expect_sha256"]) == set(m["files"]) and all(len(v) == 64 for v in m["expect_sha256"].values())


def test_dose_v13_f_low_rule_and_draws_are_the_registered_generator():
    import dose_draws as DD

    assert DD.choose_f_low({2 / 3: 0.30, 3 / 4: 0.25, 5 / 6: 0.17}) == 2 / 3          # median closest to 0.5
    assert DD.choose_f_low({2 / 3: 0.75, 3 / 4: 0.25, 5 / 6: None}) == 2 / 3          # tie -> the smaller f
    assert DD.choose_f_low({2 / 3: None, 3 / 4: float("nan")}) is None
    t = _tables(L=6, N=12, seed=3)
    k, code, f = 3, "code", 2 / 3
    res = DD.draws_for_base(t, k, code, f, log=None, base="moe_b1")
    ex, acc = res["examined"], res["accepted"]
    assert [r["d"] for r in ex] == list(range(1, len(ex) + 1)) and all(r["seed"] == 5100 + r["d"] for r in ex)
    assert res["run"] == (len(acc) == DD.N_ACCEPT) and (res["run"] or (len(ex) == DD.MAX_SEEDS and res["reason"]))
    inband = [r["d"] for r in ex if r["status"] == "ok" and r["P_selection"] is not None
              and DD.ACCEPT[0] <= r["P_selection"] <= DD.ACCEPT[1]]
    assert [e["d"] for e in acc] == inband[:DD.N_ACCEPT]                # R2: the first draws in band on the SELECTION half
    for e in acc:
        assert e["seed"] == 5100 + e["d"] and e["data_seed"] == 7100 + e["d"] and e["base"] == "moe_b1"
        A, B = np.asarray(e["A"])[None], np.asarray(e["B"])[None]
        assert e["selection_P"] == float(CX.reg_predicted(t.sel, A, B, "c1", target="books", code=code)[0])
        assert e["predicted"]["P"] == float(CX.reg_predicted(t.est, A, B, "c1", target="books", code=code)[0])  # regressor
        ref = CX.reg_draw(t, "c1", 100 + e["d"], f=f, target="books", code=code, k=k, tol=CX.REG_TOL)
        as_int = lambda S: [sorted(int(x) for x in s) for s in S]                      # noqa: E731
        assert as_int(ref["A"]) == as_int(e["A"]) and as_int(ref["B"]) == as_int(e["B"])   # registered generator, larger f


def test_nestor_books_feasibility_is_the_gate_ladder_and_leaves_the_config(worlds, tmp_path):
    import g0_gate as GG
    import nestor_books_feasibility as NB

    gcfg = os.path.join(os.path.dirname(worlds["B"]), "g0_gate_config.json")
    before = _sha(gcfg)
    r = NB.assess(gcfg, worlds["B"], n_draws=100, log=None)
    assert _sha(gcfg) == before
    cfg = GG.load_config(gcfg)
    b = cfg["bases"]["nestor"]
    t = CX.reg_load_dir(b["dir"], b["family"], layout=cfg["layout"], half_seed=int(cfg["half_seed"]))
    code = r["code_stream"]
    direct = GG.ladder(t, "c1", GG.rungs_for("c1", target="books", code=code), target="books", code=code, n_draws=100,
                       k=CX.reg_k(t))
    assert r["cells"][code]["median"] == direct["median"] and r["cells"][code]["f"] == direct["f"]
    assert r["feasible"] == bool(direct["feasible"] and direct["median"] >= GG.LOG_G0)
    if r["feasible"]:
        assert [e["seed"] for e in r["draws"]] == [5001, 5002, 5003] and r["draws"][0]["data_seed"] == 7001


def _confirm(worlds, tmp, monkeypatch, extra=()):
    import confirm_plan as CP

    _, _, _, r2 = _g1g2(worlds["B"], tmp, monkeypatch)
    a = CP.argparse.Namespace(g0_record=worlds["B"], g1_record="g1/gate_record_G1.json", g2_record="g2/gate_record_G2.json",
                              config=None, confirm_config=None, pins=None, b3_record="none.json",
                              nestor_books="none.json", throughput=None,
                              states="results/queue_state_g1.json,results/queue_state_g2.json", spent_h=None,
                              out_dir="confirm", no_check=True)
    for k, v in extra:
        setattr(a, k, v)
    cfg_tmp = os.path.join(str(tmp), "cfg.json")
    c = _cfg()
    json.dump(c, open(cfg_tmp, "w"))
    a.config = cfg_tmp
    return CP.build(a), r2, a


def test_dose_acceptance_on_the_selection_half_and_base_specific_seeds(monkeypatch):
    """R2 (prereg v1.3 version 7 sec. 4.2): the first two draws whose SELECTION-half P_d is in [0.35, 0.65] are accepted;
    the estimation-half P_d stays the regressor. Data seeds 7100 + d (MoE base 1) and 7150 + d (MoE base 2)."""
    import types
    import dose_draws as DD

    P_est = {1: 0.50, 2: 0.90, 3: 0.40, 4: 0.55}                       # d = 1 in band only on the estimation half
    P_sel = {1: 0.20, 2: 0.50, 3: 0.60, 4: 0.45}

    def fake_draw(t, design, dd, **kw):
        d = dd - DD.DOSE_SEED_OFFSET
        return {"status": "ok", "d": dd, "seed": 5000 + dd, "A": [[d]], "B": [[d + 1]], "predicted": {"P": P_est[d]}}

    monkeypatch.setattr(DD.CX, "reg_draw", fake_draw)
    monkeypatch.setattr(DD.CX, "reg_predicted", lambda tab, A, B, design, **kw: np.array([P_sel[int(np.asarray(A)[0][0][0])]]))
    monkeypatch.setattr(DD, "overlap_per_layer", lambda A, B: [0.0])
    for base, off in (("moe_b1", 0), ("moe_b2", 50)):
        res = DD.draws_for_base(types.SimpleNamespace(sel="SEL"), 1, "code", 5 / 6, log=None, base=base)
        acc = res["accepted"]
        assert res["run"] and [e["d"] for e in acc] == [2, 3]
        assert [e["selection_P"] for e in acc] == [0.50, 0.60] and [e["predicted"]["P"] for e in acc] == [0.90, 0.40]
        assert [e["data_seed"] for e in acc] == [7100 + off + 2, 7100 + off + 3]
        assert [r["accepted"] for r in res["examined"]] == [False, True, True]


def test_overrun_guard_applies_the_registered_cut_order(tmp_path, monkeypatch):
    """R10: measured spent + remaining planned hours against the envelope; above it the registered cut order
    picks the blocks to skip (M12-b before dose-response), the P-6 extension in FREEZE.json widens the envelope."""
    import confirm_plan as CP

    monkeypatch.chdir(tmp_path)
    blocks = {"C1:moe_b1": ["a"], "dose": ["d1", "d2"], "M12b": ["m1"], "C1-DE": ["e1"]}
    jobs = [{"id": "a", "est_h": 10.0}, {"id": "overrun_guard_dose", "est_h": 0.0}, {"id": "d1", "est_h": 5.0},
            {"id": "d2", "est_h": 5.0}, {"id": "overrun_guard_M12b", "est_h": 0.0}, {"id": "m1", "est_h": 4.0},
            {"id": "overrun_guard_C1-DE", "est_h": 0.0}, {"id": "e1", "est_h": 6.0}]
    json.dump({"blocks": blocks}, open("plan.json", "w"))
    json.dump({"jobs": jobs}, open("queue.json", "w"))
    state = {"a": {"status": "done", "hours": 12.0}}
    base = {"plan": "plan.json", "queue": "queue.json", "envelope_h": 120.0, "freeze": "none.json",
            "cut_order": ["M12b", "nestor_books", "dose", "C1-DE"]}
    assert all(CP.overrun_guard(state, dict(base, block=b, spent_before_h=80.0))["pass"] for b in ("dose", "M12b", "C1-DE"))
    r = {b: CP.overrun_guard(state, dict(base, block=b, spent_before_h=95.0)) for b in ("dose", "M12b", "C1-DE")}
    assert r["dose"]["projected_gpu_h"] == 127.0 and r["dose"]["cut_set"] == ["M12b", "dose"]
    assert [r[b]["pass"] for b in ("dose", "M12b", "C1-DE")] == [False, False, True]
    json.dump({"parameters": {"envelope extension (decision P-6; 0 if none)": {"value": {"gpu_h": 10.0}}}}, open("FREEZE.json", "w"))
    assert CP.overrun_guard(state, dict(base, block="dose", spent_before_h=95.0, freeze="FREEZE.json"))["pass"] is True


def test_freeze_parameters_are_checked_against_the_gate_records(tmp_path):
    """R5: the plan-derived freeze parameters equal the G2 freeze block, the G1 decision and the G0 record;
    any difference makes freeze-write / freeze-verify wait (spec check 3)."""
    root = tmp_path
    _freeze_world(root)
    refs = {"plan": "confirm/plan_confirm.json", "nestor_books": "results/nestor_books_feasibility.json",
            "dose": "results/dose/dose_record.json", "b3": None, "forecasts_csv": "freeze/forecasts_v13.csv",
            "forecasts_meta": "freeze/forecasts_v13_meta.json", "decisions": ["decisions_v13_points.md"]}
    prm = CU.freeze_parameters(str(root), refs, 0.0, None)
    assert prm["n_d"]["source"].endswith("g2/gate_record_G2.json#freeze.n_d")
    for rel, keys, val in (("g2/gate_record_G2.json", ("freeze", "n_d"), 5), ("g1/gate_record_G1.json", ("decision", "steps"), 192),
                           ("g0/gate_record.json", ("gates", "G0-7", "moe_b1", "beta0"), 0.3),
                           ("g0/gate_record.json", ("decisions", "code_stream", "chosen"), "code")):
        p = root / rel
        orig = p.read_text()
        d = json.loads(orig)
        x = d
        for kk in keys[:-1]:
            x = x[kk]
        x[keys[-1]] = val
        p.write_text(json.dumps(d))
        with pytest.raises(SystemExit, match="gate records"):
            CU.freeze_parameters(str(root), refs, 0.0, None)
        p.write_text(orig)
    assert CU.freeze_parameters(str(root), refs, 0.0, None) == prm


def test_confirm_plan_structure_seeds_order_and_determinism(worlds, tmp_path, monkeypatch):
    plan, r2, args = _confirm(worlds, tmp_path / "c", monkeypatch)
    runs = plan["runs"]
    q = json.load(open(plan["queue"], encoding="utf-8"))
    ids = [j["id"] for j in q["jobs"]]
    assert len(ids) == len(set(ids))
    c1 = [r for r in runs if r["role"] == "c1"]
    fi = plan["freeze_inputs"]                                   # v1.3: C1 draws per base from the plan (G2 n_d, P-5 cuts)
    per_base = 2 * fi["c1_draws_per_base"]
    assert len(c1) == 2 * per_base and {r["target"] for r in c1} == {"books"}
    assert sorted({r["d"] for r in c1}) == fi["c1_draw_indices"]
    assert all(r["seed"] == 7000 + {"moe_b1": 0, "moe_b2": 20}[r["base"]] + r["d"] for r in c1)   # MF-3
    assert plan["blocks"]["_order"]["C1:moe_b1:books"]["seed"] == 9002 and plan["blocks"]["_order"]["C1:moe_b2:books"]["seed"] == 9004
    ctrl = [r for r in c1 if r["haar_seed"] is not None]
    assert ctrl and all(r["base"] == "moe_b1" and r["d"] <= 6 and r["haar_seed"] == 9500 + r["run_index"] for r in ctrl)
    j = next(x for x in q["jobs"] if x["id"] == ctrl[0]["id"])
    names = [e["name"] for e in j["evals"]]
    assert names == ["main", "reflect", "pin", "rand", "wd"] and j["evals"][2]["variants"] == "pinned,reflect+pinned"   # P-16 core
    m12 = [r for r in runs if r["role"] == "m12b"]
    part = {r["id"]: r for r in c1}
    assert len(m12) == 3 and all(r["haar_seed"] == part[r["c1_partner"]]["haar_seed"] for r in m12)
    assert all(r["seed"] == 7500 + r["d"] and r["target"] == "german" for r in m12)
    # (decision 1, v1.2 sec. 8.2 item 7): antithetic randdir pair on DCLM for every C2 Muon run, none for SGD
    c2m, c2s = [r for r in runs if r["role"] == "c2_muon"], [r for r in runs if r["role"] == "c2_sgd"]
    jobs_by = {x["id"]: x for x in q["jobs"]}
    est = {e["id"]: e for e in plan["estimates"]}
    assert c2m and len(c2m) == len(c2s)
    assert all(r["haar_seed"] == 9500 + r["run_index"] for r in c2m) and all(r.get("haar_seed") is None for r in c2s)
    for r in c2m:
        ev = jobs_by[r["id"]]["evals"]
        assert [e["name"] for e in ev] == ["main", "rand"] and ev[1]["base"] == f"{{ckpt_{r['base']}}}"
        assert ev[1]["variants"] == f"randdir:{r['haar_seed']},randdir-:{r['haar_seed']}" and len(ev[1]["sets"]) == 1
        assert abs(est[r["id"]]["eval_h"] - 2 * est[r["id"].replace("_muon_", "_sgd_")]["eval_h"]) <= 2e-4
    assert all([e["name"] for e in jobs_by[r["id"]]["evals"]] == ["main"] for r in c2s)
    hs = [r["haar_seed"] for r in c1 + c2m if r.get("haar_seed") is not None]
    assert len(hs) == len(set(hs))
    fr = {r["base"]: r for r in runs if r["role"] == "freeze"}
    assert set(fr) == {"moe_b1", "nestor"}
    c5 = [x for x in q["jobs"] if x["id"].startswith(("c5_full", "c5_pair"))]
    assert c5 and all(x.get("requires") == ["c5_hash_gate"] for x in c5)
    assert ids.index("c5_hash_gate") < ids.index(c5[0]["id"]) and ids.index("c5_forecasts") < ids.index("c5_hash_gate")
    first_c1 = min(ids.index(r["id"]) for r in c1)
    assert all(ids.index(x) < first_c1 for x in ids if x.startswith("confirm_base_"))
    assert not [r for r in runs if r["block"] == "SEC"] and any("partial-revert" in w for w in plan["warnings"])
    assert all(e["total_h"] > 0 for e in plan["estimates"] if e["kind"] == "train")
    import confirm_plan as CP

    plan2 = CP.build(args)                                                   # same inputs -> same plan and queue
    assert plan2["sha256"] == plan["sha256"] and plan2["queue_sha256"] == plan["queue_sha256"]
    # , MF-3: one data seed per draw unit, recorded in the plan; a shared seed stops the build
    ds = plan["data_seeds"]["draw_units"]
    units = {}
    for r in runs:
        if r["role"] in CP.DRAW_ROLES:
            units.setdefault(int(r["seed"]), set()).add((CP.DRAW_ROLES[r["role"]], r["base"], int(r["d"])))
    assert units and all(len(u) == 1 for u in units.values()) and sorted(int(s) for s in ds) == sorted(units)
    with pytest.raises(SystemExit, match="MF-3"):
        CP.check_data_seeds([{"role": "c1", "base": "moe_b1", "d": 1, "seed": 7001},
                             {"role": "c1", "base": "moe_b2", "d": 1, "seed": 7001}])
    # , R10: never-cut blocks first; an overrun guard in front of every cuttable block, required by its jobs
    blocks = {k: v for k, v in plan["blocks"].items() if not k.startswith("_")}
    pos = {b: min(ids.index(i) for i in v if i in ids) for b, v in blocks.items() if any(i in ids for i in v)}
    cuttable = [b for b in CP.CONFIRM_DEFAULTS["budget"]["cut_order"] if b in pos]
    assert cuttable
    for b in cuttable:
        assert all(pos[e] < pos[b] for e in ("C1:moe_b1", "C1:moe_b2", "C2", "C3", "C5", "b3", "audit") if e in pos)
        g = f"overrun_guard_{b}"
        assert g in ids and ids.index(g) < pos[b] and jobs_by[g]["params"]["func"] == "confirm_plan:overrun_guard"
        assert all(g in (jobs_by[i].get("requires") or []) for i in blocks[b] if i in jobs_by)
    assert sorted(i for i in ids if i.startswith("overrun_guard_")) == sorted(f"overrun_guard_{b}" for b in cuttable)
    # P-16 (2026-11-15): core first, the core marker, selpin deferred, C2 stopped after step 8 on the 96-step schedule
    p16 = plan["p16"]
    core_b = [b for b in p16["core_blocks"] if b != "base_eval"]
    assert core_b == ["C1:moe_b1", "C1:moe_b2", "C2"]                     # P-16 revised (2026-11-15 ~03:00)
    gi = ids.index("core_done")
    gate = jobs_by["core_done"]
    assert gate["kind"] == "gate" and gate["params"]["func"] == "confirm_plan:core_marker" and not gate.get("requires")
    assert gate["params"]["out"] == "results/confirm/CORE_DONE.json"
    core_ids = [i for b in core_b for i in blocks.get(b, []) if i in jobs_by]
    assert core_ids and max(ids.index(i) for i in core_ids) == gi - 1
    core_bases = {r["base"] for r in runs if r.get("block") in core_b}
    base_ids = [x for x in ids if x.startswith("confirm_base_")]
    core_be = [x for x in base_ids if est[x]["base"] in core_bases]
    assert core_be and p16["core_base_evals"] == core_be and all(ids.index(x) < ids.index(core_ids[0]) for x in core_be)
    assert set(gate["params"]["core_jobs"]) == set(core_ids) | set(core_be)
    for x in base_ids:                         # other theta_0 evaluations: after the marker, right before their base's first job
        if x not in core_be:
            first = min(k for k, i in enumerate(ids) if i in est and i != x and est[i]["base"] == est[x]["base"])
            assert gi < ids.index(x) < first
    assert all(ids.index(i) > gi for b, v in blocks.items() if b not in core_b for i in v if i in ids)
    moved = [b for b in ("b3", "audit") if b in pos]
    assert p16["after_marker_first"] == [b for b in ("b3", "audit") if b in blocks]
    nxt = [b for b in pos if b not in core_b + moved]
    assert all(gi < pos[m] < min(pos[b] for b in nxt) for m in moved) if nxt else True
    assert abs(p16["core_planned_gpu_h"] - sum(est[i]["total_h"] for i in set(core_ids) | set(core_be))) < 2e-3
    for r in c1:
        sj = jobs_by[f"selpin_{r['id']}"]
        assert sj["kind"] == "eval" and sj["requires"] == [r["id"]] and ids.index(sj["id"]) > gi
        a_ = sj["args"]
        assert a_[1] == f"{{final_{r['id']}}}" and a_[a_.index("--variants") + 1] == "selpin,reflect+selpin"
        assert a_[a_.index("--out") + 1] == f"{{root_{r['id']}}}/eval_selpin"
        ev = next(e for e in r["evals"] if e["name"] == "selpin")
        assert ev["variants"] == "selpin,reflect+selpin" and ev["deferred_job"] == sj["id"]
    if fi.get("s_star") == 8:                                  # truncated at s* = 8
        for r in c2m + c2s:                     # MF-P1 fix (A): Muon stops after step 8, SGD after step 24 (heads 0/8/24)
            jb = jobs_by[r["id"]]
            st = 24 if r in c2s else 8
            assert jb["steps"] == 96 and jb["warmup"] == r["warmup"] and jb["interim"]["stop_after"] == st
            assert max(jb["interim"]["steps"]) == st and jb["interim"]["full_steps"] == [8]
            assert est[r["id"]]["steps_priced"] == st                   # training priced at the steps actually run
        assert p16["c2_stop_after"] == 8 and p16["c2_stop_after_sgd"] == 24
        assert all(est[r["id"]]["steps_priced"] == 96 for r in c1)
    else:
        assert any(w.startswith("P-16: C2 not truncated") for w in plan["warnings"])
    tot = round(sum(plan["block_gpu_h"].values()), 3)
    assert abs(p16["core_planned_gpu_h"] + p16["remaining_planned_gpu_h"] - tot) < 2e-3


def test_train_start_configuration_and_measured_step_cost(worlds, tmp_path, monkeypatch):
    """ (2026-11-14): NESTOR jobs start at the stage-B configuration (micro 4 x accum 16 with activation
    checkpointing), MoE jobs at their G1/stage-B configuration (micro 2 x accum 32 without); training is priced with the
    measured step times of these configurations plus the g1g2 compile and model-load minutes."""
    import confirm_plan as CP

    plan, _, args = _confirm(worlds, tmp_path / "c", monkeypatch)
    tr = CP.CONFIRM_DEFAULTS["train"]
    assert plan["confirm_config"]["train"] == tr
    assert {a: (v["micro"], v["ckpt"], v["s_per_step"]) for a, v in tr.items()} == {"moe": (2, False, 8.27),
                                                                                   "nestor": (4, True, 25.64)}
    q = json.load(open(plan["queue"], encoding="utf-8"))
    runs = {r["id"]: r for r in plan["runs"]}
    trains = [j for j in q["jobs"] if j["kind"] == "train"]
    assert trains and all(j["id"] in runs for j in trains)
    assert {runs[j["id"]]["arch"] for j in trains} == {"moe", "nestor"}
    for j in trains:
        a = runs[j["id"]]["arch"]
        assert (j["micro"], j["ckpt"]) == (tr[a]["micro"], tr[a]["ckpt"])
        assert j["tokens_per_step"] == 262144 == j["micro"] * {"moe": 32, "nestor": 16}[a] * j["seq"]
    tp = json.load(open(args.config, encoding="utf-8"))["throughput"]
    est = {e["id"]: e for e in plan["estimates"]}
    for j in trains:
        r = runs[j["id"]]
        want = r["steps"] * tr[r["arch"]]["s_per_step"] / 3600 + tp["compile_min"][r["arch"]] / 60 + tp["model_load_min"] / 60
        assert est[j["id"]]["train_h"] == round(want, 4)
    assert plan["cost_source"].startswith("measured step times of the start configuration")
    old = CP.Cost(json.load(open(args.config, encoding="utf-8")))          # without the table: the rates (older plans)
    assert old.train("moe", 96) == 96 * 262144 / tp["train_tok_s"]["moe"] / 3600 + tp["compile_min"]["moe"] / 60 \
        + tp["model_load_min"] / 60


def test_fallback_ladder_below_the_nestor_start(tmp_path, monkeypatch):
    """run_queue.py keeps its OOM ladder as the safety net below the new NESTOR start: micro 4 + checkpointing
    (accum 16) -> micro 2 + checkpointing (accum 32), 262,144 tokens per step in both; the OOM attempt is kept."""
    import run_queue as RQ

    monkeypatch.chdir(tmp_path)
    job = {"id": "nb", "kind": "train", "experiment": "nestor_100m", "steps": 96, "warmup": 10, "seq": 4096,
           "tokens_per_step": 262144, "micro": 4, "ckpt": True, "out_root": os.path.join("outputs", "nb"), "env": {}}
    with open("q.json", "w", encoding="utf-8") as fh:
        json.dump({"vars": {"python": sys.executable}, "logs": "logs", "state": "qs.json", "jobs": [job]}, fh)
    Q = RQ.Queue("q.json")
    calls = []

    def fake_run(argv, env, name):
        calls.append((name, argv[argv.index("--micro") + 1], argv[argv.index("--accum") + 1],
                      env["NESTOR_ACTIVATION_CHECKPOINT"]))
        with open(os.path.join(Q.logs, f"{name}.log"), "w", encoding="utf-8") as fh:
            if name.endswith("_m4c"):
                fh.write("torch.OutOfMemoryError: CUDA out of memory\n")
                return 1
            fh.write("done\n")
        os.makedirs(os.path.join(env["NESTOR_OUTPUT_ROOT"], "checkpoints"))
        open(os.path.join(env["NESTOR_OUTPUT_ROOT"], "checkpoints", "model_final.pt"), "wb").close()
        return 0

    monkeypatch.setattr(Q, "run_cmd", fake_run)
    res = Q.job_train(job)
    assert res["status"] == "done" and os.path.basename(res["root"]) == "m2c"
    assert [(x["micro"], x["ckpt"], x["rc"]) for x in res["attempts"]] == [(4, True, 1), (2, True, 0)]
    assert calls == [("nb_m4c", "4", "16", "1"), ("nb_m2c", "2", "32", "1")]
    kept = sorted(os.listdir(os.path.join("outputs", "nb")))
    assert len(kept) == 2 and kept[0] == "m2c" and kept[1].startswith("m4c_oom")


def test_forecasts_v13_units_make_verify_and_c5_gate(worlds, tmp_path, monkeypatch):
    import forecast_io as FIO
    import forecasts as FC

    _confirm(worlds, tmp_path / "f", monkeypatch)
    g0cfg = os.path.join(os.path.dirname(worlds["B"]), "g0_gate_config.json")
    U = FC.cmd_units("confirm/plan_confirm.json", ["g1/plan_g1.json", "g2/plan_g2.json"], [g0cfg],
                     os.path.join("results", "forecast"))
    assert U["units"] and U["drives"] and os.path.exists(os.path.join("results", "forecast", "queue_forecast_inputs.json"))
    q = json.load(open(os.path.join("results", "forecast", "queue_forecast_inputs.json"), encoding="utf-8"))
    assert q["state"] == "results/queue_state_forecast.json" and q["jobs"]
    # every drive starts at the plan's start configuration of its architecture, priced with its step time;
    # the template (possibly a G1 job at micro 2 without checkpointing) gives only env and experiment
    tr = json.load(open(os.path.join("confirm", "plan_confirm.json"), encoding="utf-8"))["confirm_config"]["train"]
    assert {d["arch"] for d in U["drives"]} == {"moe", "nestor"}
    for d in U["drives"]:
        j, s = d["job"], tr[d["arch"]]
        assert (j["micro"], j["ckpt"]) == (s["micro"], s["ckpt"])
        trs = FC.DRIVE_STEPS * s["s_per_step"] + FC._COMPILE_S[d["arch"]]
        assert j["est_h"] == round((trs + 2 * len(FC.CORPORA) * FC._INT_TOK / FC._TP[f"eval_{d['arch']}"]) / 3600.0, 4)
    g1t = {"experiment": "nestor_100m", "micro": 2, "ckpt": False, "seq": 4096, "tokens_per_step": 262144, "env": {}}
    sets = {FC.set_name(c, "code"): {"path": "x.bin"} for c in FC.CORPORA}
    dj = FC.drive_job("dn", "nestor", "nestor", "german", "muon", "d.bin", g1t, sets, "code", None, train_cfg=tr)
    assert (dj["micro"], dj["ckpt"]) == (4, True)
    dj0 = FC.drive_job("dn", "nestor", "nestor", "german", "muon", "d.bin", g1t, sets, "code", None)
    assert (dj0["micro"], dj0["ckpt"]) == (2, False)
    up = os.path.join("results", "forecast", "forecast_units.json")
    FC.cmd_mock_inputs(up, os.path.join("results", "queue_state_forecast.json"))
    st = ["results/queue_state_g1.json", "results/queue_state_g2.json", "results/queue_state_forecast.json"]
    spec = os.path.join(HERE, "g1g2_spec.json")
    with pytest.raises(SystemExit, match="MOCK"):
        FC.cmd_make(up, None, st, "freeze", spec_path=spec)                    # mock inputs never give registered files
    FC.cmd_make(up, None, st, "freeze", allow_mock=True, spec_path=spec)
    cp, mp = os.path.join("freeze", "forecasts_v13.csv"), os.path.join("freeze", "forecasts_v13_meta.json")
    ok, errs, _ = FIO.verify(cp, mp)
    assert ok, errs
    b = open(cp, "rb").read()
    open(cp, "wb").write(b.replace(b"\n", b"\r\n", 1))
    assert not FIO.verify(cp, mp)[0]
    assert FC.c5_hash_gate({}, {"file": "missing.json"})["pass"] is False


def test_c1de_block_plan_units_and_cut(worlds, tmp_path, monkeypatch):
    """C1-DE (prereg v1.3 sec. 4.12): generator manifests -> plan runs (seeds, evaluations, re-verification) -> forecast
    units (block DE) -> the cut order removes the block from the queue (two-pass build)."""
    import c1de_milp as DE
    import confirm_plan as CP
    import forecasts as FC

    c1de_dir = str(tmp_path / "c1de")
    doc = DE.build(worlds["B"], ["moe_b1", "moe_b2"], 3, c1de_dir, log=None)
    ok = [r for r in doc["draws"] if r["status"] == "ok"]
    assert ok and doc["registered_seeds"]
    plan, _, a = _confirm(worlds, tmp_path / "p", monkeypatch, extra=(("c1de_dir", c1de_dir),))
    de = [r for r in plan["runs"] if r["role"] == "c1de"]
    assert len(de) == 2 * len(ok) and {r["block"] for r in de} == {"C1-DE"}
    man = {b: json.load(open(os.path.join(c1de_dir, f"c1de_{b}_german.json"), encoding="utf-8")) for b in ("moe_b1", "moe_b2")}
    jobs = {j["id"]: j for j in json.load(open(plan["queue"], encoding="utf-8"))["jobs"]}
    for r in de:
        e = next(x for x in man[r["base"]]["entries"] if x["d"] == r["d"])
        assert r["id"] == f"c1de_{r['base']}_d{r['d']}_{r['side']}" and r["target"] == "german"
        off = {"moe_b1": 0, "moe_b2": 20}[r["base"]]                                     # MF-3
        assert r["seed"] == 7600 + off + r["d"] and r["generator_seed"] == 5600 + r["d"]
        assert r["P"] == e["predicted"]["P"] and r["P_selection"] == e["selection"]["P"]
        assert r["label"].startswith("exploratory")
        names = [ev["name"] for ev in jobs[r["id"]]["evals"]]
        assert names == (["main", "reflect", "pin", "rand", "wd"] if r["side"] == "A" else ["main", "reflect", "pin"])
        assert r["evals"][0]["sets"][:2] == ["dclm", "german"] and r["interim_sets"][:2] == ["dclm", "german"]
        assert jobs[r["id"]]["spec"] == e[f"spec_{r['side']}"] and jobs[r["id"]]["env"]["DCLM_SEED"] == str(7600 + off + r["d"])
    mans = plan["blocks"]["_manifests"]
    assert all(os.path.exists(mans[f"C1-DE:{b}"]["path"]) for b in ("moe_b1", "moe_b2"))
    pu = next(e for e in plan["estimates"] if e["id"] == "c1de_probe_units")
    assert pu["block"] == "C1-DE" and pu["total_h"] == round(len(de) * (2 * 3 * 24.66 * 8 / 24 + 3.0) / 3600, 4)
    assert plan["block_gpu_h"]["C1-DE"] > pu["total_h"] and plan["status_labels"]["C1-DE"].startswith("exploratory")
    # forecast units: one DE unit per run, outside the registered 108 + 11
    g0cfg = os.path.join(os.path.dirname(worlds["B"]), "g0_gate_config.json")
    U = FC.cmd_units("confirm/plan_confirm.json", ["g1/plan_g1.json", "g2/plan_g2.json"], [g0cfg],
                     os.path.join("results", "forecast"))
    assert len([u for u in U["units"] if "DE" in u["blocks"]]) == len(de)
    assert U["counts"]["moe"]["target_by_block"]["DE"] == len(de)
    assert U["_est_h"]["probes_c1de_in_plan_block"] == round(len(de) * (2 * 3 * 8 / 24 * FC._PROBE_S + 3) / 3600, 4)
    assert not [d for d in U["counts"]["deviations"] if d["block"] == "DE"] or len(de) != 12
    # a manifest that does not match its .sha256 is not planned; a consistent one with a wrong seed is rejected
    p = os.path.join(c1de_dir, "c1de_moe_b2_german.json")
    b = open(p, "rb").read()
    open(p, "wb").write(b.replace(b'"seed": 5601', b'"seed": 5611', 1))
    plan2 = CP.build(a)
    assert not [r for r in plan2["runs"] if r["role"] == "c1de" and r["base"] == "moe_b2"]
    assert any("C1-DE moe_b2: manifest" in w for w in plan2["warnings"])
    m2 = json.loads(b)
    m2["entries"][0]["seed"] = 5611
    CX.write_manifest(p, m2["entries"], extra={k: v for k, v in m2.items() if k != "entries"})
    plan3 = CP.build(a)
    assert any("C1-DE moe_b2 d1: draw rejected by the plan" in w for w in plan3["warnings"])
    assert len([r for r in plan3["runs"] if r["role"] == "c1de" and r["base"] == "moe_b2"]) == 2 * (len(
        [r for r in ok if r["base"] == "moe_b2"]) - 1)
    open(p, "wb").write(b)
    open(p + ".sha256", "w").write(f"{hashlib.sha256(b).hexdigest()}  {os.path.basename(p)}\n")
    # cuts (P-5): an envelope that forces the order down to C1-DE; pass 2 rebuilds without M12-b and C1-DE
    tot = plan["block_gpu_h"]
    total = sum(tot.values())
    env = round((total - tot["M12b"] - tot["C1-DE"]) * 1.10, 3) + 0.002
    cc = tmp_path / "confirm_cut.json"
    json.dump({"budget": {"envelope_h": env}}, open(cc, "w"))
    a.confirm_config = str(cc)
    pc = CP.build(a)
    assert pc["cut_order_applied"] == ["contingency 15% -> 10%", "M12b", "C1-DE"]
    assert pc["cut_blocks_removed"] == ["M12b", "C1-DE"] and pc["envelope_decision_required"] is None
    assert not [r for r in pc["runs"] if r["role"] in ("c1de", "m12b")]
    ids = [j["id"] for j in json.load(open(pc["queue"], encoding="utf-8"))["jobs"]]
    assert not [i for i in ids if i.startswith(("c1de_", "m12b_"))] and "C1-DE" not in pc["block_gpu_h"]
    p1 = json.load(open(os.path.join("confirm", "pass1", "plan_confirm.json"), encoding="utf-8"))
    assert [r for r in p1["runs"] if r["role"] == "c1de"] and pc["cut_decision"]["pass1_plan"]["sha256"] == p1["sha256"]
    assert any("cut by the P-5 order: C1-DE" in w for w in pc["warnings"])
    assert pc["after_cuts_gpu_h"] <= env


# ============================================================================= pins in run_continual / stageB_windows
def _child(code, env):
    e = dict(os.environ, **env)
    e.pop("PYTHONSAFEPATH", None)
    return subprocess.run([sys.executable, "-c", code], cwd=HERE, env=e, capture_output=True, text=True, timeout=300)


def test_run_continual_and_stageB_windows_read_the_pins(tmp_path):
    (tmp_path / "x.pt").write_bytes(b"ck")
    pins = tmp_path / "pins.json"
    json.dump({"pins": {"moe_100m": {"path": str(tmp_path / "x.pt")}, "dense_100m": {"path": str(tmp_path / "missing.pt")}}},
              open(pins, "w"))
    code = ("import run_continual as R, stageB_windows as S; print('RC', R.pretrained('moe_100m')); "
            "print('SB', S.base_ckpt('moe_100m')); print('NONE', R.pretrained('nestor_100m'))")
    p = _child(code, {"NESTOR_CKPT_PINS": str(pins), "PYTHONPATH": HERE})
    assert p.returncode == 0, p.stderr[-800:]
    out = dict(l.split(" ", 1) for l in p.stdout.splitlines() if l.split(" ", 1)[0] in ("RC", "SB", "NONE"))
    assert os.path.normcase(out["RC"]) == os.path.normcase(str(tmp_path / "x.pt")) == os.path.normcase(out["SB"])
    assert out["NONE"] == "None"
    p = _child("import run_continual as R; R.pretrained('dense_100m')", {"NESTOR_CKPT_PINS": str(pins), "PYTHONPATH": HERE})
    assert p.returncode != 0 and "pinned checkpoint for dense_100m missing" in (p.stderr + p.stdout)


@pytest.mark.skipif(not os.path.exists(PS), reason="Windows PowerShell 5.1 not found")
@pytest.mark.parametrize("name", ["chain6b.ps1", "g0_b3.ps1"])
def test_powershell_scripts_are_ascii_and_parse(name):
    path = os.path.join(HERE, name) if os.path.exists(os.path.join(HERE, name)) else os.path.join(HERE, "phase6", name)
    raw = open(path, "rb").read()
    assert all(b < 128 for b in raw) and b"\r\n" in raw
    cmd = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile('%s', [ref]$null, [ref]$e); "
           "if ($e) { $e | ForEach-Object { $_.Message }; exit 1 } else { exit 0 }" % path)
    p = subprocess.run([PS, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr


# ============================================================================= torch: g0_probe patch equivalence, audit
def _torch():
    return pytest.importorskip("torch")                  # imported lazily: the torch-free tests above run first


def _tiny_moe():
    import test_subset_adapt as T
    from nestor.train.frozen_routers import freeze_routers

    m = T.build(T.moe_backbone(), seed=0)
    freeze_routers(m)
    m.eval()
    return m.to(T.TEST_DEVICE), T.TEST_DEVICE


def _write_bin(path, n, seed, lo=0, hi=255):
    a = np.random.default_rng(seed).integers(lo, hi, n, dtype=np.uint32)
    a.tofile(str(path))
    json.dump({"token_dtype": "uint32", "num_tokens": n, "eot_token": 255}, open(f"{path}.meta.json", "w"))
    return str(path)


def _fixed_gradient(model, wset, params, n, *, device, bf16=True):
    torch = _torch()
    out = {}
    for i, (name, p) in enumerate(sorted(params.items())):
        g = torch.Generator().manual_seed(1000 + i)
        out[name] = (1e-3 * torch.randn(p.shape, generator=g, dtype=torch.float32)).to(p.device, p.dtype)
    return out


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_g0_probe_patch_is_value_equivalent_with_a_fixed_gradient(tmp_path, monkeypatch):
    """patch 2026-11-01 (_host/_dev: directions other than E_ALL on the host, moved one at a time): with the bf16
    target gradient replaced by a fixed gradient, the original g0_probe v2 and the patched file give BITWISE identical
    probe outputs (JSON values and per-window arrays). Device: CPU (NESTOR_TEST_DEVICE=cuda runs it on the GPU)."""
    _torch()
    import g0_probe as NEW

    orig = os.path.join(HERE, "g0_probe_v2_orig.py")
    if not os.path.exists(orig):
        orig = os.path.join(HERE, "phase6", "g0_probe_v2_orig.py")
    assert _sha(orig).startswith("db99a6db")
    OLD = _load_module("g0_probe_v2_orig_test", orig)
    import test_subset_adapt as T

    ranges = {"dclm": (0, 255), "code": (0, 80), "math": (80, 160), "german": (160, 255), "books": (40, 200)}
    streams = {n: {"path": _write_bin(tmp_path / f"{n}.bin", 64 * 24, i, lo, hi), "tokens": 64 * 24, "select": "even"}
               for i, (n, (lo, hi)) in enumerate(ranges.items())}
    outs = {}
    for tag, mod in (("old", OLD), ("new", NEW)):
        monkeypatch.setattr(mod, "target_gradient", _fixed_gradient)
        cfg = {"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": str(tmp_path / tag), "block": 64,
               "device": T.TEST_DEVICE, "streams": streams,
               "crossover": {"low": "code", "high": "math", "targets": ["german"], "match_extra": ["dclm"], "tol": 0.5,
                             "draws": 2, "min_contrast": 0.0, "max_jaccard": 1.0},
               "c2": {"target": "german", "match": ["dclm"], "draws": 1, "tol": 0.5, "min_contrast": 0.0, "max_jaccard": 1.0},
               "probe": {"targets": ["german"], "grad_windows": 2, "probe_corpora": ["dclm", "code"], "probe_windows": 3,
                         "random_seeds": [0, 1], "regime_windows": 2, "bf16_grad": False, "pin_routing": "both"}}
        model, _ = _tiny_moe()
        outs[tag] = mod.run(cfg, model=model, log=None)
    po, pn = outs["old"]["probes"]["german"], outs["new"]["probes"]["german"]
    assert set(po["subsets"]) == set(pn["subsets"]) and len(po["subsets"]) >= 5

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("seconds", "windows_file", "time", "created")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    assert strip(po) == strip(pn)                                        # every probe value bitwise equal
    wo = np.load(tmp_path / "old" / po["windows_file"])
    wn = np.load(tmp_path / "new" / pn["windows_file"])
    assert set(wo.files) == set(wn.files)
    for k in wo.files:
        assert np.array_equal(wo[k], wn[k], equal_nan=wo[k].dtype.kind == "f"), k    # pinned 'flip' arrays are NaN by design


def test_assumption_audit_on_a_tiny_moe(tmp_path, monkeypatch):
    _torch()
    import assumption_audit as AA
    import eval_windows as EW
    import g0_gate as GG
    import g0_probe as G0
    import test_subset_adapt as T

    streams = {}
    for i, n in enumerate(("dclm", "code", "math", "books")):
        p = _write_bin(tmp_path / f"{n}.bin", 64 * 48, i)
        streams[n] = {"path": p, "tokens": 64 * 8, "select": "even", "span": [0, 0.5]}
        streams[n + "_eval"] = {"path": p, "tokens": 64 * 10, "select": "even", "span": [0.5, 1]}
    d = tmp_path / "g0" / "moe_b1"
    model, dev = _tiny_moe()
    G0.run({"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": str(d), "block": 64, "device": dev,
            "streams": streams}, model=model, log=None)
    gcfg = tmp_path / "g0" / "gate.json"
    json.dump({"bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, "code_streams": ["code"],
               "c1_targets": {"moe": ["books"], "nestor": []}}, open(gcfg, "w"))
    rec = GG.compute(GG.load_config(str(gcfg)), n_draws=30, log=lambda *a: None)
    rec["config"] = {"resolved": GG.load_config(str(gcfg))}
    rp = tmp_path / "g0" / "gate_record.json"
    json.dump(rec, open(rp, "w"), default=float)
    t = CX.reg_load_dir(str(d), "moe")
    cell = {"feasible": True, "f": 0.5, "tol": 5.0, "median": 0.5, "rung": 0}           # force the C1 subset groups
    monkeypatch.setattr(AA, "base_cell", lambda g, b, target="books": {"dir": str(d), "code": "code", "cell": cell,
                                                                         "layout": "auto", "half_seed": 0, "family": "moe"})
    rep = AA.run(str(rp), "moe_b1", str(tmp_path / "audit"), windows=2, draws=1, device=dev, model=_tiny_moe()[0],
                 experiment="tiny_moe", grad_path=streams["books"]["path"], log=None)
    g = json.load(open(tmp_path / "audit" / AA.GROUPS))
    names = [x["name"] for x in g["groups"]]
    assert [n for n in names if n.startswith("layer:")] == [f"layer:{l}" for l in range(t.L)]
    assert "E_ALL" in names and any(n.startswith("c1:d1:") for n in names)
    for x in g["groups"]:
        assert x["n_units"] == len(x["keys"]) == len(set(x["keys"]))
    assert rep["routing_mode"] == "pinned" and set(rep["corpora"]) == {"dclm", "code", "math"}
    assert len(rep["B5"]) == 3 * len([x for x in g["groups"] if x.get("parts")]) and rep["richardson"]
    # per-group Q equals a direct pinned probe with the same direction and windows
    import g0_margin as GM

    win = []
    for c in ("dclm", "code", "math"):
        ws = GM.estimation_windows(str(d), c, 2)
        win += [(ws, i) for i in range(len(ws.index))]
    m2 = _tiny_moe()[0]
    params, rows = G0.expert_params(m2), G0.unit_rows(m2)
    gws = EW.WindowSet("books_grad", streams["books"]["path"], 64, AA.GRAD_WINDOWS, "random", AA.GRAD_SEED)
    u = G0.muon_direction(params, G0.target_gradient(m2, gws, params, AA.GRAD_WINDOWS, device=dev, bf16=True))
    grp = next(x for x in g["groups"] if x["name"] == "layer:0")
    ug = G0.restrict({k: v.detach().cpu() for k, v in u.items()}, rows, grp["keys"])
    eps = rep["eps"]
    res = G0.directional_probe(m2, win, [("x", ug, eps, +1), ("x", ug, eps, -1)], device=dev, log=None, pin_routing=True,
                               count_flips=False)
    q = (res[f"x|{eps:.6g}|+1"]["kl"][:2] + res[f"x|{eps:.6g}|-1"]["kl"][:2]) / (2 * eps * eps)
    assert math.isclose(float(q.mean()), rep["per_group"]["layer:0"]["dclm"]["Q_kl"]["value"], rel_tol=1e-9, abs_tol=1e-15)


def test_freeze_check4_compares_run_starts_and_the_freeze_in_one_zone(tmp_path):
    """ (final-text check, time zone): spec freeze check 4, 'no post-freeze run started earlier'.
    The freeze is UTC ('Z'); run_queue starts are epoch seconds; plan 'created', logs and now are naive LOCAL time.
    A run started 1 h before the freeze in local time must fail on any host zone; read as UTC on this UTC+2 host it
    would be 1 h AFTER the freeze and pass."""
    import calendar
    import time
    fz = "2026-11-15T03:54:42Z"
    t_f = calendar.timegm(time.strptime(fz, "%Y-%m-%dT%H:%M:%SZ"))
    loc = lambda t: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))   # noqa: E731
    assert CU.utc_epoch(fz) == t_f and CU.utc_epoch(float(t_f - 3600)) == t_f - 3600
    assert CU.utc_epoch(loc(t_f - 3600)) == t_f - 3600                     # naive = this host's local wall clock
    assert CU.utc_epoch(loc(t_f - 3600).replace("T", " ")) == t_f - 3600   # run_queue / log format
    assert CU.utc_epoch("2026-11-15T05:54:42+02:00") == t_f and CU.utc_epoch("no time") is None
    off = calendar.timegm(time.localtime(t_f)) - t_f                         # this host's UTC offset at the freeze
    assert calendar.timegm(time.strptime(loc(t_f - 3600), "%Y-%m-%dT%H:%M:%S")) - t_f == off - 3600   # the naive-as-UTC error
    root, a = _hl_root(tmp_path)
    r = str(root)
    (root / "things").mkdir()
    (root / "things" / "t.txt").write_text("t")
    (root / "data" / "y.bin").write_bytes(b"\x03")
    q = {"jobs": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}, {"id": "e"}]}
    json.dump(q, open(root / "q4.json", "w"))
    json.dump({"a": {"start": loc(t_f - 3600)}, "b": {"start": t_f - 3600.0}, "c": {"start": loc(t_f + 3600)},
               "d": {"status": "pending"}, "e": {"start": t_f + 1.0}, "x": {"start": 0.0}}, open(root / "s4.json", "w"))
    bad = CU.post_freeze_start_check(r, "q4.json", "s4.json", fz)
    assert len(bad) == 2 and "job a started 2026-11-15T02:54:42Z, 60.0 min before" in bad[0] and "job b started" in bad[1]
    (root / "s4b.json").write_text("{not json")
    assert CU.post_freeze_start_check(r, "q4.json", "s4b.json", fz)[0].startswith("check 4: queue state s4b.json unreadable")
    (root / "q4b.json").write_text("3")                                     # not a queue: nothing to check here
    assert CU.post_freeze_start_check(r, "q4b.json", "s4.json", fz) == []
    # end to end: freeze-check / freeze-write stop (25, nothing written) while a frozen-queue job started earlier
    plan = json.load(open(root / a.plan, encoding="utf-8"))
    sp = root / (plan.get("state") or "results/queue_state_confirm.json")
    sp.parent.mkdir(parents=True, exist_ok=True)
    jid = json.load(open(root / a.queue, encoding="utf-8"))["jobs"][0]["id"]
    now = time.time()
    json.dump({jid: {"start": loc(now - 3600)}}, open(sp, "w"))           # started 1 h ago, local time
    assert CU.freeze_check(r, a) == 25
    assert CU.freeze_write(r, a) == 25 and not (root / "FREEZE.json").exists()
    sp.unlink()
    assert CU.freeze_write(r, a) == 0                                      # nothing started: the freeze is written
    doc = json.load(open(root / "FREEZE.json", encoding="utf-8"))
    assert doc["post_freeze_starts"]["violations"] == [] and doc["post_freeze_starts"]["state"] == os.path.relpath(sp, root).replace("\\", "/")
    assert CU.freeze_verify(r, "FREEZE.json") == 0
    t_fr = CU.utc_epoch(doc["hashed_utc"])
    json.dump({jid: {"start": loc(t_fr - 3600)}}, open(sp, "w"))          # 1 h before the freeze, naive local
    assert CU.freeze_verify(r, "FREEZE.json") == 23
    json.dump({jid: {"start": t_fr - 3600.0}}, open(sp, "w"))              # the same as epoch seconds
    assert CU.freeze_verify(r, "FREEZE.json") == 23
    json.dump({jid: {"start": loc(t_fr + 3600)}}, open(sp, "w"))          # 1 h after the freeze: passes
    assert CU.freeze_verify(r, "FREEZE.json") == 0


def _amend_args(**kw):
    d = dict(id="A-1", reason="WD-only control: run's step count (round 9)", approved_by="PI test", decision="decisions/amend_A1.md",
             dry_run=False)
    d.update(kw)
    return type("A", (), d)


def test_post_freeze_amendment_is_append_only_and_accepted_by_freeze_verify(tmp_path, monkeypatch, capsys):
    """A hash-listed root file changed after the freeze verifies only through an append-only, hash-chained
    amendment record (old/new sha256, reason, approval, decision, UTC) written BEFORE the -Reinstall; FREEZE.json and
    freeze/freeze_record.json are never rewritten; the code record diff after the reinstall must be fully amended."""
    root, a = _hl_root(tmp_path)
    r = str(root)
    (root / "things").mkdir()
    (root / "things" / "t.txt").write_text("t")
    (root / "data" / "y.bin").write_bytes(b"\x03")
    (root / "logs").mkdir(exist_ok=True)
    json.dump({"files": CU.code_set(r)}, open(root / CU.CODE_RECORD, "w"))
    assert CU.freeze_write(r, a) == 0
    fz_bytes, rec_bytes = (root / "FREEZE.json").read_bytes(), (root / "freeze" / "freeze_record.json").read_bytes()
    assert CU.freeze_verify(r, "FREEZE.json") == 0
    # stage a new a.py as an INSTALL source, record the amendment, then "reinstall" (copy + new code record)
    monkeypatch.setattr(CU, "INSTALL", [("phase6/a.py", "a.py", "replace")])
    (root / "phase6").mkdir(exist_ok=True)
    (root / "phase6" / "a.py").write_text("A = 2\n")
    with pytest.raises(SystemExit, match="does not exist"):
        CU.amend(r, _amend_args())
    (root / "decisions").mkdir(exist_ok=True)
    (root / "decisions" / "amend_A1.md").write_text("decision A-1\n")
    assert CU.amend(r, _amend_args(dry_run=True)) == 0 and not (root / "freeze" / "amendments.jsonl").exists()
    assert CU.amend(r, _amend_args()) == 0
    recs, probs = CU.amend_records(r, json.load(open(root / "FREEZE.json"))["sha256"])
    assert not probs and {x["path"] for x in recs} >= {"a.py"} and all(x["approved_by"] == "PI test" for x in recs)
    a_rec = next(x for x in recs if x["path"] == "a.py")
    assert a_rec["old_sha256"] == _sha(root / "a.py") and a_rec["new_sha256"] == _sha(root / "phase6" / "a.py")
    with pytest.raises(SystemExit, match="already recorded"):
        CU.amend(r, _amend_args())
    assert CU.freeze_verify(r, "FREEZE.json") == 23                       # amended, but not yet installed
    shutil.copyfile(root / "phase6" / "a.py", root / "a.py")
    json.dump({"files": CU.code_set(r)}, open(root / CU.CODE_RECORD, "w"))   # what -Reinstall records
    capsys.readouterr()
    assert CU.freeze_verify(r, "FREEZE.json") == 0
    out = capsys.readouterr().out
    assert "amended after the freeze: a.py" in out and "approved by PI test" in out and "each with an amendment record" in out
    assert (root / "FREEZE.json").read_bytes() == fz_bytes and (root / "freeze" / "freeze_record.json").read_bytes() == rec_bytes
    # an edited log line breaks the chain; an unamended change fails
    lp = root / "freeze" / "amendments.jsonl"
    good = lp.read_text()
    lp.write_text(good.replace("round 9", "round 8", 1))
    assert CU.freeze_verify(r, "FREEZE.json") == 23
    lp.write_text(good)
    assert CU.freeze_verify(r, "FREEZE.json") == 0
    (root / "a.py").write_text("A = 3\n")
    assert CU.freeze_verify(r, "FREEZE.json") == 23


def test_retry_prep_moves_only_failed_train_attempts_aside(tmp_path):
    """ (E2): chain6b -RetryFailed moves the attempt root of every failed train job aside (never deleted) so
    run_queue --retry-failed trains it again from scratch; done jobs and the queue state are not touched."""
    r = str(tmp_path)
    for jid in ("t1", "t2"):
        d = tmp_path / "outputs" / jid / "m2" / "run" / "checkpoints"
        d.mkdir(parents=True)
        (d / "model_final.pt").write_bytes(b"x")
    q = {"state": "results/qs.json", "jobs": [{"id": "t1", "kind": "train"}, {"id": "t2", "kind": "train"}, {"id": "e1", "kind": "eval"}]}
    (tmp_path / "q.json").write_text(json.dumps(q))
    (tmp_path / "results").mkdir()
    st = {"t1": {"status": "failed", "error": "eval wd failed", "root": str(tmp_path / "outputs" / "t1" / "m2")},
          "t2": {"status": "done", "root": str(tmp_path / "outputs" / "t2" / "m2")}, "e1": {"status": "failed"}}
    (tmp_path / "results" / "qs.json").write_text(json.dumps(st))
    before = (tmp_path / "results" / "qs.json").read_bytes()
    assert CU.retry_prep(r, "q.json") == 0
    moved = [p.name for p in (tmp_path / "outputs" / "t1").iterdir()]
    assert len(moved) == 1 and moved[0].startswith("m2_failed") and (tmp_path / "outputs" / "t1" / moved[0] / "run").is_dir()
    assert (tmp_path / "outputs" / "t2" / "m2").is_dir()
    assert (tmp_path / "results" / "qs.json").read_bytes() == before
    assert "t1:" in (tmp_path / "logs" / "retry_prep.log").read_text()
    assert CU.retry_prep(r, "q.json") == 0                                   # nothing left to move

