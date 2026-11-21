"""Torch-free tests of the G1/G2 manifests and gates against g1g2_spec.json 1.3.0 (prereg v1.3 sec. 6), on SIMULATED outputs.

  python -m pytest test_gates.py -q -rA        (no torch is imported; the registered spec file must sit next to this file)

1. the registered specification file: identity recorded (no pinned hash), constants equal to gate_rules_v13;
2. every one of the specification's test vectors reproduced by gate_rules_v13 (rule, values, sets);
3. differential fuzz: on random inputs exactly one rule of the SPEC's own expressions holds and it is the implemented one;
4. lookup interpolation identical to the registered power_sim_v1.lookup_power;
5. manifests (spec 1.3.0 runs: pair a = 2 in S1, NESTOR books A/A, O-2 repeat, pair-2 repeat, SGD full steps, heads);
6. end-to-end gate records on simulated outputs for every branch the run outputs can reach; the hook script.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.modules.setdefault("torch", None)          # any accidental torch import fails loudly

import g1_gate as G1  # noqa: E402
import g1g2_plan as P  # noqa: E402
import g2_gate as G2  # noqa: E402
import gate_common as GC  # noqa: E402
import gate_rules_v13 as GR  # noqa: E402
import mock_g1g2 as M  # noqa: E402

G0REC = os.path.join(HERE, "g0_mock", "g0", "gate_record.json")
MOCK_CFG = os.path.join(HERE, "g1g2_config_MOCK.json")
LOOKUP = os.path.join(HERE, "signflip_lookup_v1.csv")
SPEC = os.path.join(HERE, "g1g2_spec.json")


def _spec() -> dict:
    return json.load(open(SPEC, encoding="utf-8"))


def merge(base: dict, mod: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (mod or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def mock_cfg(cfg_mod: dict | None = None) -> dict:
    cfg = P.load_config(MOCK_CFG)
    cfg["g0_probe_configs"] = {k: (v if os.path.isabs(v) else os.path.join(HERE, v)) for k, v in cfg["g0_probe_configs"].items()}
    cfg["gate_config"]["spec"] = SPEC
    cfg["gate_config"]["lookup"] = LOOKUP
    return merge(cfg, cfg_mod)


def g1_world(tmp_path, monkeypatch, mod=None, cfg_mod=None):
    os.makedirs(tmp_path, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    cfg = mock_cfg(cfg_mod)
    scen = merge(M.BASE_SCENARIO, mod)
    p1 = P.build_g1(cfg, G0REC, "g1")
    w = M.simulate_g1(p1, scen)
    return cfg, scen, p1, w


def g1_record(p1, rep=None):
    r1 = G1.gate(p1, rep, "g1/plan_g1.json", "g1r/plan_g1r.json" if rep else "")
    GC.finalize_record(r1, "g1/gate_record_G1.json")
    return r1


def g2_world(tmp_path, monkeypatch, mod=None, cfg_mod=None, g07r=None):
    cfg, scen, p1, w = g1_world(tmp_path, monkeypatch, mod, cfg_mod)
    r1 = g1_record(p1)
    assert r1["decision"]["outcome"] == "PROCEED", r1["decision"]
    if g07r is not None:
        os.makedirs("g0r", exist_ok=True)
        json.dump(g07r, open("g0r/g07r_record.json", "w", encoding="utf-8"))
    p2 = P.build_g2(cfg, G0REC, "g1/gate_record_G1.json", "g2")
    M.simulate_g2(p2, scen, w)
    r2 = G2.gate(p2, json.load(open("g1/gate_record_G1.json", encoding="utf-8")), GR.Lookup(LOOKUP),
                 plan_path="g2/plan_g2.json", g1_path="g1/gate_record_G1.json", lookup_path=LOOKUP)
    return r1, r2, p1, p2


# ============================================================================= 1-4 specification
def test_spec_file_identity_is_recorded_and_constants_match():
    ident = GR.spec_identity(SPEC)
    assert ident["version"] == GR.SPEC_VERSION and ident["n_test_vectors"] == 42 and len(ident["sha256"]) == 64
    c = _spec()["constants"]
    for k, v in c.items():
        if hasattr(GR, k) and not k.endswith("_DEFINITION"):
            got = getattr(GR, k)
            got = list(got) if isinstance(got, tuple) else got
            assert got == v, k
    assert GR.UCL80_FACTOR == 1.7275872940849268
    assert {g["id"] for g in _spec()["gates"] if g["kind"] in ("decision", "labelling", "procedure")} == set(GR.RULES)


VECTORS = _spec()["test_vectors"] if os.path.exists(SPEC) else []


@pytest.mark.parametrize("i", range(len(VECTORS)), ids=[f"{v['gate']}:{v['name']}" for v in VECTORS])
def test_spec_test_vector_reproduced(i):
    v = VECTORS[i]
    res = GR.run_vector(v, GR.Lookup(LOOKUP))
    assert GR.check_vector(v, res) == [], (v, res)


def test_reproduce_all_records_the_file_it_read():
    r = GR.reproduce_all(SPEC, LOOKUP)
    assert r["reproduced"] and r["n"] == 42 and r["failed"] == []
    assert r["spec"]["sha256"] == GC.file_sha256(SPEC)


FUNCS = dict(min=min, max=max, sum=sum, len=len, all=all, any=any, abs=abs, int=int, sqrt=math.sqrt, log=math.log, zip=zip)


def spec_rules_true(gspec: dict, variables: dict, lk: GR.Lookup) -> tuple[list, dict]:
    ns = {"__builtins__": {}, **FUNCS, **_spec()["constants"], **variables, "lookup_power": lk.power}
    for name, expr in (gspec.get("expressions") or {}).items():
        try:
            ns[name] = eval(expr, ns)                                   # noqa: S307 - the team's registered spec text
        except Exception:                                               # noqa: BLE001
            ns[name] = None
    return [r["id"] for r in gspec["rules"] if eval(r["when"], ns)], ns  # noqa: S307


def _rand_inputs(gate: str, rng: random.Random) -> dict:
    U = rng.uniform
    if gate == "G1-0":
        return dict(tests_passed=rng.randint(0, 4), tests_failed=rng.choice([0, 0, 1, 2]), tests_errors=rng.choice([0, 0, 1]))
    if gate == "G1-4":
        return dict(negctrl_audit=rng.choice(["ok", "fail", "fail"]),
                    subset_audits={f"r{i}": rng.choice(["ok", "ok", "ok", "fail"]) for i in range(rng.randint(1, 4))})
    if gate == "G1-1":
        return dict(F_full=[U(0, 0.02) for _ in range(4)], attempt=rng.choice([1, 2]))
    if gate == "G1-2":
        return dict(F={k: U(-0.002, 0.02) for k in GR.G12_KEYS}, SE={k: U(1e-4, 2e-3) for k in GR.G12_KEYS})
    if gate == "G1-3":
        return dict(FN={k: U(0, 0.02) for k in GR.G13_KEYS}, SEN={k: U(1e-4, 2e-3) for k in GR.G13_KEYS})
    if gate == "G0-9L":
        a = rng.randint(1, 30)
        return dict(routers_assessed=a, routers_outside=rng.randint(0, a), ranges_available=rng.random() < 0.8)
    if gate == "G0-7R":
        return dict(q_changed=rng.random() < 0.3, targets_changed=rng.random() < 0.1, f_or_stream_changed=rng.random() < 0.1)
    if gate == "G2-2":
        c = U(0.01, 1.0)
        return dict(c=c, gain24_muon=U(-0.02, 0.2), runs=[{"lr": c * f, "e1_divergent": rng.random() < 0.2, "loss_step1": 2.0,
                                                           "loss_step24": U(1.7, 2.2), "gain24": U(0, 0.1),
                                                           "clip_fraction": rng.choice([0.0, 0.0, 0.02])} for f in GR.SGD_GRID_FACTORS])
    if gate == "G2-3":
        return dict(Fs={s: U(0, 0.02) for s in GR.S_STAR_CANDIDATES}, SEs={s: U(1e-4, 2e-3) for s in GR.S_STAR_CANDIDATES},
                    G22_pass=rng.random() < 0.8, g03_pass=rng.random() < 0.8, c2_matching_feasible=rng.random() < 0.8)
    if gate == "G2-4":
        return dict(FB={k: U(0, 0.02) for k in GR.G24_KEYS}, SEB={k: U(1e-4, 1.5e-3) for k in GR.G24_KEYS}, pivot=rng.random() < 0.2)
    if gate == "G2-1":
        return dict(I_AA=[None if rng.random() < 0.07 else rng.gauss(0, 0.3) for _ in range(3)], beta0=[U(0.25, 0.5), U(0.25, 0.5)],
                    Pbar=U(0.3, 1.1), C1_confirmatory=rng.random() < 0.85, C2_confirmatory=rng.random() < 0.5)
    raise KeyError(gate)


@pytest.mark.parametrize("gate", sorted(GR.RULES))
def test_rules_equal_the_spec_expressions_on_random_inputs(gate):
    gspec = next(g for g in _spec()["gates"] if g["id"] == gate)
    lk, rng = GR.Lookup(LOOKUP), random.Random(20261113)
    seen = set()
    for _ in range(1500):
        x = _rand_inputs(gate, rng)
        true, ns = spec_rules_true(gspec, x, lk)
        assert len(true) == 1, (gate, x, true)                         # exclusive and exhaustive (spec claim)
        res = GR.RULES[gate](**x, **({"lookup": lk} if gate == "G2-1" else {}))
        assert res["rule"] == true[0], (gate, x, res["rule"], true)
        for k in ("r", "Fbar", "share_outside", "s_star", "sgd_lr", "sigma_I", "delta", "p12", "p20", "m", "alpha"):
            if k in res["values"] and ns.get(k) is not None:
                assert GR._close(res["values"][k], ns[k], 1e-12, 1e-15), (gate, k, res["values"][k], ns[k])
        seen.add(res["rule"])
    assert len(seen) >= 2, (gate, seen)


def test_lookup_equals_registered_power_sim_v1():
    ps = pytest.importorskip("power_sim_v1")
    lk = GR.Lookup(LOOKUP)
    rows = [{"D": int(r["D"]), "dist": r["dist"], "delta": float(r["delta"]), "power_sf_025": float(r["power_sf_025"]),
             "power_sf_05": float(r["power_sf_05"])} for r in lk.rows]
    for D in (12, 20):
        for al in (0.025, 0.05):
            for d in np.linspace(-0.5, 3.0, 141):
                assert lk.power(D, float(d), al) == ps.lookup_power(rows, D, float(d), al), (D, al, d)
    with pytest.raises(ValueError):
        lk.power(12, 1.0, 0.01)


def test_rule_domain_errors_are_raised_not_decided():
    with pytest.raises(ValueError):
        GR.g1_2({"R1|CODE": 0.01}, {"R1|CODE": 0.001})
    with pytest.raises(ValueError):
        GR.g1_2({k: 0.01 for k in GR.G12_KEYS}, {k: 0.0 for k in GR.G12_KEYS})       # SE_positive
    with pytest.raises(ValueError):
        GR.g2_2(0.3, [{"lr": 0.9, "e1_divergent": False, "loss_step1": 2, "loss_step24": 1, "gain24": 1, "clip_fraction": 0}] * 3, 0.1)
    assert GR.i_aa({"R1|CODE": 0.01, "R1|math": -0.001, "R2|CODE": 0.01, "R2|math": 0.01}) is None
    assert GR.g1_5(0.02, 0.018, 0.018)["values"]["ratio"] == pytest.approx(0.002)           # numerator alone (O-3)


# ============================================================================= primitives (gate_common, unchanged)
def _ev(idx, hs, ce):
    return {"index": np.asarray(idx), "hash": np.asarray(hs, dtype=np.uint64), "ce_sum": np.asarray(ce, float) * 10,
            "n_tok": np.full(len(idx), 10)}


def test_forgetting_is_the_registered_window_mean_and_pairs_by_hash():
    idx, hs = np.arange(5), np.arange(1, 6)
    b = _ev(idx, hs, [1, 2, 3, 4, 5])
    r = _ev(idx[::-1], hs[::-1], np.array([1.5, 2.1, 3.3, 4.0, 5.1])[::-1])
    f = GC.forgetting(b, r)
    d = np.array([0.5, 0.1, 0.3, 0.0, 0.1])
    assert math.isclose(f["F"], d.mean(), rel_tol=1e-12) and math.isclose(f["SE"], d.std(ddof=1) / math.sqrt(5), rel_tol=1e-12)
    with pytest.raises(ValueError, match="different token content"):
        GC.forgetting(b, _ev(idx, [1, 2, 3, 4, 99], [1, 2, 3, 4, 5]))
    assert math.isclose(GC.gain(b, r)["G"], -f["F"])


def _recs(losses, max_norm=None, clipped=False):
    return [{"step": i + 1, "loss": l, "ce": l, "max_norm": max_norm, "clipped": clipped} for i, l in enumerate(losses)]


def test_e1_divergence_rules():
    assert GC.e1_check(_recs([3.0 - 0.01 * i for i in range(30)]), warmup=5)["diverged"] is False
    assert GC.e1_check(_recs([3.0, 2.9, float("nan"), 2.8]), warmup=1)["diverged"]
    rise = GC.e1_check(_recs([3.0] * 6 + [3.0 + 0.4 * i for i in range(1, 10)] + [2.9]), warmup=5)
    assert rise["diverged"] and any("EMA" in r for r in rise["reasons"])
    assert GC.e1_check(_recs([3.0, 2.8, 2.9, 3.0]), warmup=1)["diverged"]
    assert GC.e1_check([{"step": 1, "grad_norm": 1.0}], warmup=1)["status"] == "no_loss_logged"


def test_record_hash_verifies_and_detects_tampering(tmp_path):
    p = str(tmp_path / "rec.json")
    h = GC.finalize_record({"a": 1, "b": [1.5, None]}, p)
    assert GC.verify_record(p) and json.load(open(p))["sha256"] == h
    rec = json.load(open(p))
    rec["a"] = 2
    json.dump(rec, open(p, "w"))
    assert not GC.verify_record(p)


def test_fsym_decomposition_identity_and_one_sided_flag():
    idx, hs = np.arange(6), np.arange(1, 7)
    rng = np.random.default_rng(3)
    ce0 = 3 + rng.standard_normal(6)
    res = {v: _ev(idx, hs, ce0 + F + 0.001 * rng.standard_normal(6)) for v, F in
           (("ckpt", 0.03), ("reflect", 0.01), ("selpin", 0.025), ("reflect+selpin", 0.008), ("pinned", 0.02), ("reflect+pinned", 0.006))}
    d = GC.fsym_decomposition(_ev(idx, hs, ce0), res)
    assert d["complete"]
    total = d["pin"]["F_sym"] + d["G_gate_pathway"]["value"] + d["Phi_flips"]["value"]
    assert math.isclose(total, d["live"]["F_sym"], rel_tol=1e-12, abs_tol=1e-15)


# ============================================================================= 5 manifests (spec 1.3.0 runs)
def test_g1_plan_runs_seeds_heads_and_deterministic_hash(tmp_path, monkeypatch):
    cfg, _, p1, _ = g1_world(tmp_path, monkeypatch)
    runs = {r["id"]: r for r in p1["runs"]}
    want = {"g1_moe_b1_full_math_s101", "g1_moe_b1_full_math_s102", "g1_moe_b1_full_german_s101", "g1_moe_b1_full_german_s102",
            "g1_moe_b1_eall_german_s101", "g1_moe_b1_eall_german_s102", "g1_moe_b1_aa1_R1", "g1_moe_b1_aa1_R2", "g1_moe_b1_aa2_R1",
            "g1_moe_b1_aa2_R2", "g1_moe_b1_dclmcont_s101", "g1_nestor_full_german_s101", "g1_nestor_full_german_s102",
            "g1_nestor_aa1_R1", "g1_negctrl"}
    assert set(runs) == want                                            # spec runs.G1 (ids as proposed)
    a1, a2, na = runs["g1_moe_b1_aa1_R1"], runs["g1_moe_b1_aa2_R2"], runs["g1_nestor_aa1_R1"]
    assert (a1["target"], a1["generator_seed"], a1["seed"]) == ("german", 4001, 7201)
    assert (a2["target"], a2["generator_seed"], a2["seed"]) == ("books", 4002, 7202)          # P-1: pair a = 2 in S1
    assert (na["target"], na["generator_seed"], na["seed"]) == ("books", 4101, 7251)          # G1-3 reads books (v1.3)
    assert runs["g1_nestor_full_german_s101"]["h100_duplicate"] and not runs["g1_nestor_full_german_s102"]["h100_duplicate"]
    assert {r["seed"] for r in p1["runs"] if r["role"] == "full"} == {101, 102}
    q = json.load(open(p1["queue"], encoding="utf-8"))
    job = next(j for j in q["jobs"] if j["id"] == "g1_moe_b1_aa2_R1")
    assert all(s["head"] == 64 and s["tokens"] == 2048000 for s in job["interim"]["sets"])     # registered 64-window head
    assert "books" in p1["base_eval_files"]["moe_b1"] and "books" in p1["base_eval_files"]["nestor"]
    assert os.path.exists(p1["gate_config_file"]) and p1["gate_config"]["spec"] == SPEC
    h = p1["sha256"]
    p1b = P.build_g1(cfg, G0REC, "g1")
    assert p1b["sha256"] == h and P.plan_hash(p1b) == h


def test_g1r_repeats_full_both_pairs_and_nestor_aa_at_192_steps(tmp_path, monkeypatch):
    cfg, _, p1, _ = g1_world(tmp_path, monkeypatch)
    pr = P.build_g1(cfg, G0REC, "g1r", stage="g1r", steps=192, repeat=True, reuse_base_from="g1")
    ids = {r["id"] for r in pr["runs"]}
    assert ids == {"g1r_moe_b1_full_math_s101", "g1r_moe_b1_full_math_s102", "g1r_moe_b1_full_german_s101",
                   "g1r_moe_b1_full_german_s102", "g1r_moe_b1_aa1_R1", "g1r_moe_b1_aa1_R2", "g1r_moe_b1_aa2_R1",
                   "g1r_moe_b1_aa2_R2", "g1r_nestor_aa1_R1"}                   # decision O-2
    assert all(r["steps"] == 192 for r in pr["runs"]) and not any(j["kind"] == "eval" for j in json.load(open(pr["queue"]))["jobs"])
    d1 = {p["pair"]: p["draw_file"] for p in p1["aa_pairs"]}
    for p in pr["aa_pairs"]:
        assert json.load(open(p["draw_file"])) == json.load(open(d1[p["pair"]]))            # same registered draws


def test_g2_plan_pair3_sgd_full_steps_and_pair2_repeat_only_when_g12_changes(tmp_path, monkeypatch):
    cfg, scen, p1, w = g1_world(tmp_path, monkeypatch)
    g1_record(p1)
    p2 = P.build_g2(cfg, G0REC, "g1/gate_record_G1.json", "g2")
    ids = {r["id"] for r in p2["runs"]}
    assert ids == {"g2_moe_b1_aa3_R1", "g2_moe_b1_aa3_R2", "g2_sgd_lr1", "g2_sgd_lr2", "g2_sgd_lr3"}
    a3 = next(r for r in p2["runs"] if r["id"] == "g2_moe_b1_aa3_R1")
    assert (a3["target"], a3["generator_seed"], a3["seed"]) == ("books", 4003, 7203)
    q = json.load(open(p2["queue"], encoding="utf-8"))
    s1 = next(j for j in q["jobs"] if j["id"] == "g2_sgd_lr1")
    assert s1["env"]["NESTOR_INTERIM_FULL"] == "1" and s1["interim"]["full_steps"] == [0, 8, 24, 48]
    assert s1["interim"]["full_sets"] == ["dclm", "german"] and s1["env"]["NESTOR_SGD_WD"] == "0"
    assert q["file_vars"]["sgd_lr0"].endswith("#sgd.grid.0")
    # G1-2 lengthen -> pair a = 2 repeated in the G1-2 configuration
    cfg2, _, p1b, _ = g1_world(tmp_path / "b", monkeypatch, {"books_F": {"code": 0.0045, "math": 0.0045}, "books_sd": 0.004})
    r1b = g1_record(p1b)
    assert r1b["decision"]["g1_2_rule"] == "lengthen" and r1b["decision"]["repeat_pair2_in_g2"]
    p2b = P.build_g2(cfg2, G0REC, "g1/gate_record_G1.json", "g2")
    assert {"g2_moe_b1_aa2_R1_cfg", "g2_moe_b1_aa2_R2_cfg"} <= {r["id"] for r in p2b["runs"]}
    assert all(r["steps"] == 192 for r in p2b["runs"] if r["role"] in ("aa", "sgd"))


# ============================================================================= 6 end-to-end gate records
def test_g1_baseline_proceeds_with_registered_configuration(tmp_path, monkeypatch):
    _, _, p1, _ = g1_world(tmp_path, monkeypatch)
    r = g1_record(p1)
    d, g = r["decision"], r["gates"]
    assert d["outcome"] == "PROCEED" and d["q"] == 0.25 and d["steps"] == 96 and d["C1_confirmatory"] and not d["pivot"]
    assert [g[k]["rule"] for k in ("G1-0", "G1-4", "G1-1", "G1-2", "G1-3", "G0-7R")] == ["pass"] * 5 + ["not_needed"]
    assert g["G0-9L"]["rule"] == "not_evaluable" and d["NESTORB_label"] == "exploratory"       # App. C.6 ranges missing
    assert g["G1-2"]["record"]["pair"]["target"] == "books" and set(g["G1-2"]["record"]["F"]) == set(GR.G12_KEYS)
    assert g["G1-5"]["values"]["ratio"] == pytest.approx(abs(0.0185 - 0.018) / abs(0.018 - 0.016), rel=1e-6)
    assert r["spec"]["reproduced"] and r["spec"]["sha256"] == GC.file_sha256(SPEC) and not r["problems"]
    assert g["G1-1"]["values"]["Fbar"] == pytest.approx(0.020, rel=1e-9)


@pytest.mark.parametrize("mod,gate,rule", [
    ({"pipeline": {"tests": 6, "failures": 1, "errors": 0, "skipped": 0}}, "G1-0", "stop"),
    ({"negctrl_audit_fails": False}, "G1-4", "audit_broken"),
    ({"audit_fail": ["g1_moe_b1_aa2_R1"]}, "G1-4", "subset_failed"),
])
def test_g1_blocking_gates_hold_and_stop_downstream_decisions(tmp_path, monkeypatch, mod, gate, rule):
    _, _, p1, _ = g1_world(tmp_path, monkeypatch, mod)
    r = g1_record(p1)
    assert r["gates"][gate]["rule"] == rule and r["gates"][gate]["sets"]["g1_blocked"]
    assert r["decision"]["outcome"] == "HOLD" and r["gates"]["G1-1"]["status"] == "not_evaluated"
    assert G1.EXIT[r["decision"]["outcome"]] == 5


@pytest.mark.parametrize("kind", ["diverge", "missing", "no_pipeline"])
def test_g1_holds_on_e1_missing_runs_and_missing_audit_result(tmp_path, monkeypatch, kind):
    mod = {"diverge": {"diverge": ["g1_moe_b1_aa1_R2"]}, "missing": {"missing": ["g1_moe_b1_full_math_s102"]},
           "no_pipeline": {"pipeline": None}}[kind]
    _, _, p1, _ = g1_world(tmp_path, monkeypatch, mod)
    r = g1_record(p1)
    assert r["decision"]["outcome"] == "HOLD", r["decision"]
    if kind == "no_pipeline":
        assert r["gates"]["G1-0"]["status"] == "pending"


def test_g1_1_lengthen_repeat_then_pass_or_stop(tmp_path, monkeypatch):
    cfg, scen, p1, w = g1_world(tmp_path, monkeypatch, {"full_moe_F_dclm": 0.006})
    r = g1_record(p1)
    assert r["gates"]["G1-1"]["rule"] == "lengthen" and r["decision"]["outcome"] == "REPEAT_192"
    assert r["gates"]["G1-2"]["status"] == "not_evaluated"                          # O-2: judged on the 192-step pair
    pr = P.build_g1(cfg, G0REC, "g1r", stage="g1r", steps=192, repeat=True, reuse_base_from="g1")
    M.simulate_g1(pr, merge(scen, {"full_moe_F_dclm": 0.012}), w)
    r2 = g1_record(p1, pr)
    d = r2["decision"]
    assert r2["gates"]["G1-1"]["rule"] == "pass" and d["attempt"] == 2 and d["outcome"] == "PROCEED" and d["steps"] == 192
    assert r2["gates"]["G1-2"]["record"]["F"] and r2["gates"]["G1-1"]["first"]["rule"] == "lengthen"
    M.simulate_g1(pr, merge(scen, {"full_moe_F_dclm": 0.004}), w)
    r3 = g1_record(p1, pr)
    assert r3["gates"]["G1-1"]["rule"] == "stop" and r3["decision"]["outcome"] == "STOP" and G1.EXIT["STOP"] == 4


@pytest.mark.parametrize("F,sd,rule,q,steps", [
    (0.011, 0.02, "pass", 0.25, 96), (0.0045, 0.004, "lengthen", 0.25, 192), (0.0018, 0.002, "widen", 0.5, 192),
    (0.0008, 0.001, "pivot", None, 96)])
def test_g1_2_ladder_on_the_books_pair(tmp_path, monkeypatch, F, sd, rule, q, steps):
    _, _, p1, _ = g1_world(tmp_path, monkeypatch, {"books_F": {"code": F, "math": F}, "books_sd": sd,
                                                   "books_B_scale": {"2": {"code": 1.0, "math": 1.0}}})
    r = g1_record(p1)
    d = r["decision"]
    assert d["g1_2_rule"] == rule and d["q"] == q and d["steps"] == steps
    assert d["pivot"] == (rule == "pivot") and d["C1_confirmatory"] == (rule != "pivot")
    assert d["g0_7r"] == ("recompute" if rule == "widen" else "not_needed")
    assert d["repeat_pair2_in_g2"] == (rule in ("lengthen", "widen"))


def test_g1_3_fail_and_g0_9l_label_with_ranges(tmp_path, monkeypatch):
    rh = {"routers": {f"r{i}": {"entropy": 0.5 if i < 6 else 2.0} for i in range(10)}}
    _, _, p1, _ = g1_world(tmp_path, monkeypatch, {"nestor_aa_F": {"code": 0.003, "math": 0.009}, "routing_health": rh},
                           {"gate_config": {"appC6_ranges": {"source": "MOCK table", "ranges": {"entropy": [1.0, 3.0]}}}})
    r = g1_record(p1)
    assert r["gates"]["G1-3"]["rule"] == "fail" and r["decision"]["NESTORB_signal"] is False
    assert r["gates"]["G0-9L"]["rule"] == "label" and r["gates"]["G0-9L"]["values"]["share_outside"] == pytest.approx(0.6)
    assert r["decision"]["NESTORB_label"] == "exploratory, routing-health label" and r["decision"]["C3N_label"] == "exploratory"


def test_g2_baseline_record_and_freeze_parameters(tmp_path, monkeypatch):
    r1, r2, p1, p2 = g2_world(tmp_path, monkeypatch)
    g, f = r2["gates"], r2["freeze"]
    assert r2["outcome"] == "COMPLETE", r2["hold_reasons"]
    assert g["G2-2"]["rule"] == "pass" and f["sgd_lr"] == pytest.approx(0.05 / 3)          # smallest feasible lr
    assert g["G2-3"]["rule"] == "exploratory_late" and f["s_star"] == 24 and f["C2_draws"] == 4 and not f["C2_confirmatory"]
    assert g["G2-3"]["record"]["theta0_reference"].startswith("in-process")
    assert g["G2-4"]["rule"] == "pass" and f["C1_confirmatory"] and f["m"] == 1 and f["holm_family"] == ["C1"]
    v = g["G2-1"]["values"]
    assert v["m"] == 1 and v["alpha"] == 0.05 and f["n_d"] == g["G2-1"]["sets"]["n_d"]
    pairs = g["G2-1"]["record"]["pairs"]
    assert [pairs[a]["target"] for a in (1, 2, 3)] == ["german", "books", "books"]
    I = [pairs[a]["I_AA"] for a in (1, 2, 3)]
    assert v["sigma_I"] == pytest.approx(math.sqrt(sum(x * x for x in I) / 3), rel=1e-12)
    assert v["delta"] == pytest.approx((0.75 - 0.25) * f["P_bar"] / (GR.UCL80_FACTOR * v["sigma_I"]), rel=1e-12)
    assert f["c1_draw_indices"] == list(range(1, 2 * f["n_d"] + 1))                         # P-13
    assert r2["spec"]["reproduced"]


def test_g2_sstar_8_makes_c2_confirmatory_and_m_2(tmp_path, monkeypatch):
    sgd = copy.deepcopy(M.BASE_SCENARIO["sgd"])
    sgd[0]["dclm_F"] = {"8": 0.008, "24": 0.012, "48": 0.02}
    _, r2, _, _ = g2_world(tmp_path, monkeypatch, {"sgd": sgd, "sgd_full_sd": 0.01})
    f = r2["freeze"]
    assert r2["gates"]["G2-3"]["rule"] == "confirmatory" and f["s_star"] == 8 and f["C2_draws"] == 8
    assert f["m"] == 2 and f["holm_family"] == ["C1", "C2"] and r2["gates"]["G2-1"]["values"]["alpha"] == 0.025


def test_g2_2_failure_applies_o5(tmp_path, monkeypatch):
    sgd = copy.deepcopy(M.BASE_SCENARIO["sgd"])
    for s in sgd:
        s["gain"]["24"] = 0.01                                     # < 25 % of Muon's 0.20 at every lr
    _, r2, _, _ = g2_world(tmp_path, monkeypatch, {"sgd": sgd})
    f = r2["freeze"]
    assert r2["gates"]["G2-2"]["rule"] == "fail" and r2["gates"]["G2-3"]["rule"] == "not_evaluated"
    assert f["C2_confirmatory"] is False and f["C2_draws"] == 4 and f["sgd_lr"] == pytest.approx(0.05 / 3)
    _, r2b, _, _ = g2_world(tmp_path / "b", monkeypatch, {"sgd": sgd, "sgd_no_loss_drop": [0, 1, 2]})
    assert r2b["freeze"]["C2_draws"] == 0 and r2b["freeze"]["sgd_lr"] is None and "not run" in r2b["freeze"]["c2_status"]


def test_g2_4_failure_makes_c1_exploratory_and_g2_1_not_applicable(tmp_path, monkeypatch):
    _, r2, _, _ = g2_world(tmp_path, monkeypatch, {"books_F_by_pair": {"3": {"code": 0.011, "math": 0.0035}}})
    f = r2["freeze"]
    assert r2["gates"]["G2-4"]["rule"] == "fail" and f["C1_confirmatory"] is False and f["pivot"]
    assert r2["gates"]["G2-1"]["rule"] == "not_applicable" and f["n_d"] is None and f["m"] == 0 and f["holm_family"] == []


def test_g2_1_noisy_aa_gives_five_draws_and_flag(tmp_path, monkeypatch):
    _, r2, _, _ = g2_world(tmp_path, monkeypatch, {"aa_B_scale": {"code": 2.2, "math": 0.45}})
    g = r2["gates"]["G2-1"]
    assert g["rule"] in ("large", "flagged") and r2["freeze"]["n_d"] == 5 and r2["freeze"]["c1_draw_indices"] == list(range(1, 11))


def test_g2_uses_the_repeated_pair2_and_g07r_record(tmp_path, monkeypatch):
    mod = {"books_F": {"code": 0.0018, "math": 0.0018}, "books_sd": 0.002, "books_B_scale": {"2": {"code": 1.0, "math": 1.0}},
           "books_F_by_pair": {"2_cfg": {"code": 0.011, "math": 0.010}, "3": {"code": 0.011, "math": 0.010}}}
    cfg, scen, p1, w = g1_world(tmp_path, monkeypatch, mod)
    r1 = g1_record(p1)
    assert r1["decision"]["g1_2_rule"] == "widen" and r1["decision"]["g0_7r"] == "recompute"
    p2 = P.build_g2(cfg, G0REC, "g1/gate_record_G1.json", "g2")
    M.simulate_g2(p2, scen, w)
    g1rec = json.load(open("g1/gate_record_G1.json"))
    r2 = G2.gate(p2, g1rec, GR.Lookup(LOOKUP), plan_path="g2/plan_g2.json", g1_path="g1/gate_record_G1.json")
    assert r2["outcome"] == "HOLD" and any("G0-7R" in h for h in r2["hold_reasons"])           # recompute record missing
    os.makedirs("g0r", exist_ok=True)
    json.dump({"beta0": {"moe_b1": 0.3, "moe_b2": 0.27}, "margin_limited": {"moe_b1": False, "moe_b2": False}, "P_bar": 1.2},
              open("g0r/g07r_record.json", "w"))
    r2 = G2.gate(p2, g1rec, GR.Lookup(LOOKUP), plan_path="g2/plan_g2.json", g1_path="g1/gate_record_G1.json")
    assert r2["outcome"] == "COMPLETE", r2["hold_reasons"]
    assert r2["gates"]["G2-4"]["record"]["pair2_source"].startswith("G2 repeat") and r2["gates"]["G2-4"]["rule"] == "pass"
    assert r2["freeze"]["beta0_1"] == 0.3 and r2["freeze"]["P_bar"] == 1.2 and r2["freeze"]["q"] == 0.5


def test_g2_refuses_without_g1_proceed(tmp_path, monkeypatch):
    cfg, _, p1, _ = g1_world(tmp_path, monkeypatch, {"negctrl_audit_fails": False})
    g1_record(p1)
    with pytest.raises(SystemExit):
        P.build_g2(cfg, G0REC, "g1/gate_record_G1.json", "g2")


def test_real_g0_record_decisions_feed_the_v13_gates():
    cands = [os.path.join(HERE, "g0", "gate_record_registered.json"), os.path.join(HERE, "phase6", "gate_record_G0_registered.json"),
             os.path.join(HERE, "gate_record_G0_registered.json")]
    rec = next((c for c in cands if os.path.exists(c)), None)
    if rec is None:
        pytest.skip("registered G0 record copy not present")
    d = P.g0_decisions(json.load(open(rec, encoding="utf-8")))
    assert d["code_stream"] == "code" and d["f"] == 0.5 and d["P_bar_source"] == "moe_b1|books|code"
    assert d["P_bar_c1_moe_b1"] == pytest.approx(0.9890344727164835, abs=1e-15)
    assert d["c2_matching_feasible"] and d["g0_3_pass"] and d["beta0"] == {"moe_b1": 0.25, "moe_b2": 0.25}


def test_hook_script_order_names_and_parses():
    p = os.path.join(HERE, "hooks_after_B.ps1")
    s = open(p, encoding="utf-8-sig").read()
    order = [s.index(k) for k in ("logs\\after_B_code_sha256.json", "Invoke-Torch 'GPU pipeline audit'", "Build-Plan 'g1'",
                                  "Invoke-TorchFree 'spec 1.3.0 test vectors'", "Invoke-TorchFree 'g1_gate.py'", "Build-Plan 'g2'",
                                  "Invoke-TorchFree 'g2_gate.py'")]
    assert order == sorted(order)
    assert "tests_gpu_pipeline.xml" in s and "gate_record_G1.json" in s and "gate_record_G2.json" in s
    assert "g1_gate_record" not in s and "'--g1-dir', 'g1'" in s
    ps = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    if os.path.exists(ps):
        cmd = ("$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('" + p +
               "', [ref]$null, [ref]$e); if ($e.Count) { $e | % { $_.Message }; exit 1 } else { 'PARSE_OK' }")
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
        assert "PARSE_OK" in r.stdout, r.stdout + r.stderr
