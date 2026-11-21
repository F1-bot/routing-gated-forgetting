"""Phase-4 tests: registered subset generator (crossover.py v3, prereg v1.0 sec. 4.5) and g0_gate.py (sec. 6).

This process must NOT import torch: power_sim_v1.make_world uses numpy BLAS, and the first numpy BLAS call after
`import torch` aborts the process on this platform (OMP Error #15). Run:
    python -m pytest test_phase4.py -q
power_sim_v1.py is taken from $POWER_SIM_V1 or from this directory.
"""
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crossover as CX       # noqa: E402
import eval_windows as EW    # noqa: E402
import dedup_check as DD    # noqa: E402
import g0_gate as GG         # noqa: E402
import g0_margin as GM       # noqa: E402

PS_PATH = os.environ.get("POWER_SIM_V1", os.path.join(HERE, "power_sim_v1.py"))
SMOKE = os.environ.get("G0_SMOKE_DIR", os.path.join(HERE, "g0_smoke", "nestor"))


@pytest.fixture(scope="module")
def PS():
    spec = importlib.util.spec_from_file_location("power_sim_v1", PS_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def world(PS, arch="moe", lam=0.5, seed=1, rho=0.4):
    return PS.make_world(np.random.default_rng(seed), arch, lam, rho, 0.5)


def tabs(PS, w, key="w_sel"):
    return {c: w[key][..., PS.CI[c]] for c in PS.CORPORA}


def test_process_has_no_torch():
    assert "torch" not in sys.modules


# ----------------------------------------------------------------------------- equivalence with power_sim_v1
@pytest.mark.parametrize("arch", ["moe", "nestor"])
def test_propose_and_weighted_identical_to_power_sim(PS, arch):
    w = world(PS, arch)
    tilt = PS.tilt_cm(w)
    k = int(round(0.25 * w["N"]))
    for f in CX.REG_POOLS:
        A1, B1 = PS.propose(np.random.default_rng(7), w, tilt, k, f, 512)
        A2, B2 = CX.reg_propose(np.random.default_rng(7), tilt, k, f, 512)
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2)
    for tau in CX.REG_TAUS:
        A1, B1 = PS.propose_weighted(np.random.default_rng(8), w, tilt, 2 * k, tau, 64)
        A2, B2 = CX.reg_propose_weighted(np.random.default_rng(8), tilt, 2 * k, tau, 64)
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2)
    A1, B1 = PS.propose_random(np.random.default_rng(9), w, k, 128)
    A2, B2 = CX.reg_propose_random(np.random.default_rng(9), w["L"], w["N"], k, 128)
    assert np.array_equal(A1, A2) and np.array_equal(B1, B2)


@pytest.mark.parametrize("arch,lam,design", [("moe", 1.0, "c1"), ("moe", 0.75, "c2"), ("nestor", 1.0, "c1")])
def test_swap_search_identical_to_power_sim(PS, arch, lam, design):
    """Same RNG stream, same chains, same accepted swaps, same converged subsets (registered RO == power_sim RO
    when every layer's statistics are normalised to the same total, as in power_sim worlds)."""
    w = world(PS, arch, lam, seed=11)
    sel = tabs(PS, w)
    k = int(round(0.25 * w["N"]))
    if design == "c1":
        tilt, corp = PS.tilt_cm(w), ("german", "dclm")
    else:
        tilt, corp = np.log(sel["german"]), ("dclm", "code", "math")
    some_work = False
    for f in CX.REG_POOLS:
        A1, B1, fr1 = PS.swap_search(np.random.default_rng(21), w, tilt, k, f, 64, corp)
        A2, B2, fr2, n2 = CX.reg_swap_search(np.random.default_rng(21), sel, tilt, k, f, 64, list(corp))
        assert np.array_equal(A1, A2) and np.array_equal(B1, B2) and fr1 == fr2 and n2 == A2.shape[0]
        some_work |= 0 < n2
        if n2:
            err = CX.reg_match_errors(sel, A2, B2, list(corp))
            assert (np.abs(err) <= CX.REG_TOL + 1e-12).all()
    assert some_work


def test_draw_pairs_identical_to_power_sim_c1_with_swap_fallback(PS):
    w = world(PS, "moe", 1.0, seed=5)
    sel = tabs(PS, w)
    k = 6
    for f, mb, need in ((0.5, 200, 300), (1 / 3, 1, 700)):          # second case forces the swap fallback
        A1, B1, acc1, nb1 = PS.draw_pairs(np.random.default_rng(3), w, "c1", "german", k, need, f_pool=f, M=512, max_batches=mb)
        r = CX.reg_pairs(sel, CX.reg_tilt(sel, "c1"), k, f, need, None, ["german", "dclm"], max_batches=mb,
                         rng=np.random.default_rng(3))
        assert A1 is not None and np.array_equal(A1, r["A"]) and np.array_equal(B1, r["B"])
        assert math.isclose(acc1, r["acceptance"], rel_tol=1e-12) and nb1 == r["batches"]
        P1, P2 = PS.pred_contrast(w, A1, B1), CX.reg_predicted(tabs(PS, w, "w_est"), r["A"], r["B"], "c1")
        assert np.allclose(P1, P2, rtol=1e-12, atol=1e-12)
        if mb == 1:
            assert r["swap_chains"] > 0 and r["from_swap"] > 0


def test_draw_pairs_identical_to_power_sim_c2_and_aa(PS):
    w = world(PS, "moe", 0.5, seed=6)
    sel = tabs(PS, w)
    A1, B1, acc1, _ = PS.draw_pairs(np.random.default_rng(4), w, "c2", "german", 6, 100, f_pool=0.5)
    r = CX.reg_pairs(sel, CX.reg_tilt(sel, "c2", target="german"), 6, 0.5, 100, None, CX.reg_matched("c2"),
                     max_batches=40, swap=False, rng=np.random.default_rng(4))
    assert np.array_equal(A1, r["A"]) and np.array_equal(B1, r["B"]) and math.isclose(acc1, r["acceptance"], rel_tol=1e-12)
    dq = CX.reg_predicted(tabs(PS, w, "w_est"), r["A"], r["B"], "c2", target="german")
    We = w["w_est"][..., PS.CI["german"]]
    assert np.allclose(dq, np.log(PS.ro(We, B1)) - np.log(PS.ro(We, A1)), rtol=1e-12)
    A1, B1, acc1, _ = PS.draw_pairs(np.random.default_rng(5), w, "aa", "german", 6, 50)
    r = CX.reg_pairs(sel, None, 6, None, 50, None, ["german", "dclm"], max_batches=40, aa=True, rng=np.random.default_rng(5))
    assert np.array_equal(A1, r["A"]) and np.array_equal(B1, r["B"])
    for a, b in zip(r["A"], r["B"]):                                   # disjoint per layer
        assert all(not set(x) & set(y) for x, y in zip(a.tolist(), b.tolist()))


# ----------------------------------------------------------------------------- registered semantics
def test_registered_ro_is_global_sum_ratio():
    rng = np.random.default_rng(0)
    s2 = rng.lognormal(size=(3, 8)) * np.array([[1.0], [5.0], [0.2]])   # layers with very different totals
    S = np.array([[0, 1], [2, 3], [4, 5]])
    manual = sum(s2[l, S[l]].sum() for l in range(3)) / s2.sum()
    assert math.isclose(float(CX.reg_ro(s2, S)), manual, rel_tol=1e-14)
    layer_mean = np.mean([s2[l, S[l]].sum() / s2[l].sum() for l in range(3)])
    assert not math.isclose(manual, layer_mean, rel_tol=1e-3)          # the registered definition is NOT the layer mean
    sn = s2 / s2.sum(1, keepdims=True)
    assert math.isclose(float(CX.reg_ro(sn, S)), float(np.mean([sn[l, S[l]].sum() for l in range(3)])), rel_tol=1e-14)


def test_accepted_pairs_satisfy_every_registered_constraint(PS):
    w = world(PS, "nestor", 0.4, seed=12)      # at lam 0.75 DCLM-tilt f=1/3 reaches 58/64 only (G0-2 territory)
    sel = tabs(PS, w)
    k, N = 16, w["N"]
    for design, tilt, corp in (("c1", CX.reg_tilt(sel, "c1"), CX.reg_matched("c1", target="german")),
                               ("c2", CX.reg_tilt(sel, "c2", target="german"), CX.reg_matched("c2")),
                               ("dclm_tilt", CX.reg_tilt(sel, "dclm_tilt"), CX.reg_matched("dclm_tilt", target="german"))):
        for f in CX.REG_POOLS:
            r = CX.reg_pairs(sel, tilt, k, f, 64, 1, corp)
            assert r["complete"]
            npool = int(round(f * N))
            order = np.argsort(tilt, axis=1, kind="stable")
            for A, B in zip(r["A"], r["B"]):
                for l in range(w["L"]):
                    assert len(set(A[l])) == k and len(set(B[l])) == k
                    assert set(A[l]) <= set(order[l, :npool]) and set(B[l]) <= set(order[l, N - npool:])
                    assert not set(A[l]) & set(B[l])
            err = CX.reg_match_errors(sel, r["A"], r["B"], corp)
            assert (np.abs(err) <= CX.REG_TOL + 1e-12).all()
            pred = CX.reg_predicted(tabs(PS, w, "w_sel"), r["A"], r["B"], design, target="german")
            assert np.median(pred) > 0                                  # B = high pool, A = low pool by construction


def _table_obj(PS, w, est_key="w_est"):
    keys = [[f"moe:b0:l{l}:e{i}" for i in range(w["N"])] for l in range(w["L"])]
    return CX.RegTables("moe", [f"moe:b0:l{l}" for l in range(w["L"])], keys, tabs(PS, w), tabs(PS, w, est_key), {"synthetic": True})


def test_seed_rule_determinism_and_manifest(PS, tmp_path):
    t = _table_obj(PS, world(PS, "moe", 0.75, seed=13))
    e1 = CX.reg_draw(t, "c1", 1, f=0.5, target="german")
    e1b = CX.reg_draw(t, "c1", 1, f=0.5, target="german")
    e2 = CX.reg_draw(t, "c1", 2, f=0.5, target="german")
    assert e1["seed"] == 5001 and e2["seed"] == 5002 and e1["A"] == e1b["A"] and e1["A"] != e2["A"]
    assert e1["status"] == "ok" and set(e1["matched_values"]) == {"german", "dclm"} and "P" in e1["predicted"]
    assert all(abs(v["diff"]) <= CX.REG_TOL + 1e-12 for v in e1["matched_values"].values())
    r = CX.reg_pairs(t.sel, CX.reg_tilt(t.sel, "c1"), 6, 0.5, 1, 5001, ["german", "dclm"])
    assert r["A"][0].tolist() == e1["A"]                                # draw d = first accepted pair of seed 5000 + d
    assert CX.reg_draw(t, "c2", 1, f=0.5, target="german")["seed"] == 6001
    assert CX.reg_draw(t, "dclm_tilt", 1, f=0.5, target="german")["seed"] == 6501
    assert CX.reg_draw(t, "c5", 1, f=0.5, target="german")["seed"] == 6801
    sp = e1["spec_A"]["select"]
    assert sum(len(s["expert"]) for s in sp) == 6 * t.L and all(len(s["expert"]) == 6 for s in sp)
    p = str(tmp_path / "m.json")
    h1 = CX.write_manifest(p, [e1, e2])
    import hashlib
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h1 == open(p + ".sha256").read().split()[0]
    assert CX.write_manifest(str(tmp_path / "m2.json"), [e1b, e2]) == h1
    aa = CX.reg_aa_draw(t, 1, target="german", seed_base=4000)
    assert aa["status"] == "ok" and not set(aa["subsets"]["A"]) & set(aa["subsets"]["B"])


def test_kappa_definition():
    sel = {"dclm": np.tile(np.arange(1.0, 9.0), (2, 1))}                # modules 0..7, usage increasing
    est = {"dclm": np.tile(np.array([4, 1, 1, 1, 1, 1, 1, 2.0]), (2, 1))}
    t = CX.RegTables("moe", ["a", "b"], [[f"e{i}" for i in range(8)]] * 2, sel, est, {})
    r = CX.reg_kappa(t, "dclm", 1 / 8)
    assert r["k"] == 1 and r["low"] == [[0], [0]] and r["high"] == [[7], [7]]
    assert math.isclose(r["ro_low"], 8 / 24) and math.isclose(r["ro_high"], 4 / 24) and math.isclose(r["ratio"], 0.5)
    assert math.isclose(r["kappa_L"], (8 / 24) / (1 / 8)) and math.isclose(r["ro_rand_expected"], 1 / 8)


# ----------------------------------------------------------------------------- synthetic g0 directories
def _write_routing(path, tables_by_window: list[dict], block=64, set_path="x.bin", columns=None, checkpoint="ckpt_a.pt"):
    """routing npz whose per-window sums of g^2 reproduce the given per-token tables exactly."""
    n = len(tables_by_window)
    rt = {}
    for lk in tables_by_window[0]:
        arr = np.zeros((n, 3, tables_by_window[0][lk].shape[0]))
        for i, tb in enumerate(tables_by_window):
            arr[i, 2] = tb[lk] * block
            arr[i, 0] = 1.0
            arr[i, 1] = np.sqrt(tb[lk]) * block
        rt[lk] = arr
    for b, (kc, c) in (columns or {}).items():
        col = np.zeros((n, 3, c))
        col[:, 0] = kc * block / c
        rt[f"column:b{b}"] = col
    res = {"index": np.arange(n), "starts": np.arange(n) * block, "hash": np.arange(n, dtype=np.uint64) + 1,
           "ce_sum": np.ones(n) * block, "n_tok": np.full(n, block - 1), "routing": rt}
    EW.save_result(path, res, {"variant": "base", "set": {"name": os.path.basename(path), "path": set_path, "block": block},
                               "checkpoint": checkpoint})


def make_g0_dir(d, w_sel: dict, w_est: dict, *, family="moe", layout="registered", n_win=4, half_seed=0, dense_last=False):
    os.makedirs(d, exist_ok=True)
    L, N = next(iter(w_sel.values())).shape
    if family == "moe":
        lks = [f"moe:b0:l{l}" for l in range(L)]
        cols = None
    else:
        lks = [f"priv:b{l}:l0" for l in range(L)]
        cols = {l: ((8 if (dense_last and l == L - 1) else 3), 8) for l in range(L)}
    for c in w_sel:
        sp = "data/dclm/dclm_val10.bin" if c == "dclm" else f"{c}.bin"
        if layout == "registered":
            _write_routing(os.path.join(d, f"routing__{c}.npz"), [{lk: w_sel[c][l] for l, lk in enumerate(lks)}] * n_win,
                           set_path=sp, columns=cols)
            _write_routing(os.path.join(d, f"routing__{c}_eval.npz"), [{lk: w_est[c][l] for l, lk in enumerate(lks)}] * n_win,
                           set_path=sp.replace("val10", "val09"), columns=cols)
        else:
            half = EW.split_halves(2 * n_win, seed=half_seed)
            wins = [{lk: (w_sel if h == 0 else w_est)[c][l] for l, lk in enumerate(lks)} for h in half]
            _write_routing(os.path.join(d, f"routing__{c}.npz"), wins, set_path=sp, columns=cols)


@pytest.mark.parametrize("layout", ["registered", "crossfit"])
def test_reg_load_dir_layouts_and_dense_tail(PS, tmp_path, layout):
    w = world(PS, "nestor", 0.5, seed=14)
    sel, est = tabs(PS, w), tabs(PS, w, "w_est")
    d = str(tmp_path / layout)
    make_g0_dir(d, sel, est, family="priv", layout=layout, dense_last=True)
    t = CX.reg_load_dir(d, "priv")
    assert t.L == w["L"] - 1 and t.excluded == [f"priv:b{w['L'] - 1}:l0"] and t.N == 64 and CX.reg_k(t) == 16
    for c in sel:
        assert np.allclose(t.sel[c], sel[c][:-1], rtol=1e-12) and np.allclose(t.est[c], est[c][:-1], rtol=1e-12)
        assert t.sources[c]["layout"] == layout


def _gate_cfg(tmp_path, bases, **kw):
    cfg = {"bases": bases, "decision_base": {"moe": "moe_b1", "nestor": "nestor"},
           "c1_targets": {"moe": ["german", "books"], "nestor": ["german"]}, "code_streams": ["code", "pycode"],
           "c2": {"base": "moe_b1", "target": "german"}, "dclm_tilt": {"base": "moe_b1", "match_target": "german"},
           "layout": "auto", "half_seed": 0}
    cfg.update(kw)
    p = tmp_path / "gate_cfg.json"
    p.write_text(json.dumps(cfg))
    return GG.load_config(str(p))


def _with_pycode(sel, est, power=1.5):
    sel, est = dict(sel), dict(est)
    for tb in (sel, est):
        x = tb["code"] ** power
        tb["pycode"] = x / x.sum(1, keepdims=True)
    return sel, est


def test_g0_gate_decisions_on_synthetic_bases(PS, tmp_path):
    strong = world(PS, "moe", 1.0, seed=1)
    weak = world(PS, "nestor", 0.05, seed=4)
    s1, e1 = _with_pycode(tabs(PS, strong), tabs(PS, strong, "w_est"))
    s2, e2 = _with_pycode(tabs(PS, weak), tabs(PS, weak, "w_est"))
    make_g0_dir(str(tmp_path / "moe_b1"), s1, e1, family="moe")
    make_g0_dir(str(tmp_path / "nestor"), s2, e2, family="priv", dense_last=True)
    cfg = _gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe", "base": 1},
                               "nestor": {"dir": "nestor", "arch": "nestor", "base": 1}})
    t0 = time.time()
    rec = GG.compute(cfg, n_draws=200, log=lambda *a: None)
    assert time.time() - t0 < 120
    g1 = rec["gates"]["G0-1"]
    assert g1["moe_pass"] is True and g1["nestor_pass"] is False
    assert any("C1-N dropped" in a for a in g1["actions"]) and any("C1 proceeds" in a for a in g1["actions"])
    # f = argmax median P among feasible pool fractions; median recomputed independently
    t = CX.reg_load_dir(str(tmp_path / "moe_b1"), "moe")
    cell = g1["cells"]["moe_b1|german|" + g1["code_stream"]]
    meds = {}
    for f in CX.REG_POOLS:
        r = CX.reg_pairs(t.sel, CX.reg_tilt(t.sel, "c1", code=g1["code_stream"]), 6, f, 200, 5000, ["german", "dclm"])
        meds[f] = float(np.median(CX.reg_predicted(t.est, r["A"], r["B"], "c1", code=g1["code_stream"])))
    assert math.isclose(cell["f"], max(meds, key=meds.get)) and math.isclose(cell["median"], max(meds.values()), rel_tol=1e-12)
    # code stream: pycode has the stronger code tilt -> larger P -> chosen
    assert rec["decisions"]["code_stream"]["chosen"] == "pycode"
    sc = rec["decisions"]["code_stream"]["scores"]
    assert sc["pycode"] > sc["code"]
    # G0-4 equals reg_kappa on the decision base
    assert math.isclose(rec["gates"]["G0-4"]["ratio"], CX.reg_kappa(t, "dclm", 1 / 8)["ratio"], rel_tol=1e-12)
    assert rec["layout"]["moe_b1"]["german"] == "registered" and not any("crossfit" in d for d in rec["deviations"])
    rows = [r["row"] for r in rec["appendix_b"]]
    assert len(rows) == 8 and rows[0].startswith("G0-1 median P (MoE base 1") and rows[-1].startswith("G0-7")


def test_g0_gate_both_architectures_fail_stop_pivot(PS, tmp_path):
    w1, w2 = world(PS, "moe", 0.05, seed=2), world(PS, "nestor", 0.05, seed=4)
    make_g0_dir(str(tmp_path / "moe_b1"), tabs(PS, w1), tabs(PS, w1, "w_est"))
    make_g0_dir(str(tmp_path / "nestor"), tabs(PS, w2), tabs(PS, w2, "w_est"), family="priv")
    cfg = _gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}, "nestor": {"dir": "nestor", "arch": "nestor"}},
                    code_streams=["code"])
    rec = GG.compute(cfg, n_draws=100, log=lambda *a: None)
    acts = rec["gates"]["G0-1"]["actions"]
    assert any(a.startswith("STOP/PIVOT") for a in acts) and not any("C1-N dropped" in a for a in acts)


def test_g0_gate_moe_fails_nestor_passes_and_target_replacement(PS, tmp_path):
    w1, w2 = world(PS, "moe", 0.05, seed=2), world(PS, "nestor", 1.0, seed=3)
    make_g0_dir(str(tmp_path / "moe_b1"), tabs(PS, w1), tabs(PS, w1, "w_est"))
    make_g0_dir(str(tmp_path / "nestor"), tabs(PS, w2), tabs(PS, w2, "w_est"), family="priv")
    cfg = _gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}, "nestor": {"dir": "nestor", "arch": "nestor"}},
                    code_streams=["code"])
    rec = GG.compute(cfg, n_draws=100, log=lambda *a: None)
    acts = rec["gates"]["G0-1"]["actions"]                      # PIVOT, no C1-N promotion
    assert any(a.startswith("STOP/PIVOT") and "does not promote C1-N" in a for a in acts)
    assert not any("C1 replaced by C1-N" in a for a in acts) and rec["gates"]["G0-7"] == {}
    # one MoE target failing while the other passes -> replacement action (books made uninformative: matched corpus
    # tied to the code/math tilt so that matching forces tiny contrasts)
    w = world(PS, "moe", 1.0, seed=1)
    sel, est = tabs(PS, w), tabs(PS, w, "w_est")
    z = np.log(sel["code"]) - np.log(sel["math"])
    for tb in (sel, est):
        x = np.exp(4.0 * z)
        tb["books"] = x / x.sum(1, keepdims=True)
    d = tmp_path / "rep"
    make_g0_dir(str(d / "moe_b1"), sel, est)
    cfg = _gate_cfg(d, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, code_streams=["code"])
    rec = GG.compute(cfg, n_draws=100, log=lambda *a: None)
    mt = rec["gates"]["G0-1"]["moe_targets"]
    assert mt["german"] is True and mt["books"] is False
    assert any("target books fails: its cells are replaced by german" in a for a in rec["gates"]["G0-1"]["actions"])


def test_g0_2_c2_falls_back_to_dclm_only_matching(PS, tmp_path):
    rng = np.random.default_rng(0)
    L, N = 12, 24
    g = rng.lognormal(0, 1.0, (L, N))
    sel = {"german": g, "code": g ** 3, "math": g ** -3, "dclm": rng.lognormal(0, 0.05, (L, N)),
           "books": rng.lognormal(0, 0.5, (L, N))}
    sel = {c: v / v.sum(1, keepdims=True) for c, v in sel.items()}
    make_g0_dir(str(tmp_path / "moe_b1"), sel, sel)
    cfg = _gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, code_streams=["code"])
    rec = GG.compute(cfg, n_draws=100, log=lambda *a: None)
    c2 = rec["gates"]["G0-3"]["result"]
    assert c2["feasible"] and c2["rung"] == 1 and c2["matched"] == ["dclm"]
    assert c2["tried"][0]["per_f"]["0.500000"]["acceptance"] < 1e-3
    assert "DCLM only" in rec["gates"]["G0-3"]["action"]
    row = [r for r in rec["gates"]["G0-2"]["rows"] if r["design"] == "C2"][0]
    assert row["rung"] == 1


def test_g0_5_and_g0_6_from_per_window_files(tmp_path):
    n, blk = 500, 4096
    rng = np.random.default_rng(1)
    ce = 4.8 + 0.3 * rng.standard_normal(n)
    base = {"index": np.arange(n), "starts": np.arange(n) * blk, "hash": np.arange(n, dtype=np.uint64) + 7,
            "n_tok": np.full(n, blk - 1.0), "routing": {}}
    EW.save_result(str(tmp_path / "loc.npz"), dict(base, ce_sum=ce * (blk - 1)), {"set": {"block": blk}})
    EW.save_result(str(tmp_path / "h1.npz"), dict(base, ce_sum=(ce + 2e-4 * rng.standard_normal(n)) * (blk - 1)), {"set": {"block": blk}})
    EW.save_result(str(tmp_path / "h2.npz"), dict(base, ce_sum=(ce + 2e-3) * (blk - 1)), {"set": {"block": blk}})
    ok = GG.g0_5({"local": str(tmp_path / "loc.npz"), "h100": str(tmp_path / "h1.npz")}, {})
    bad = GG.g0_5({"local": str(tmp_path / "loc.npz"), "h100": str(tmp_path / "h2.npz")}, {})
    assert ok["pass"] is True and ok["tokens"] == 2_048_000 and ok["r"] > 0.999
    assert bad["pass"] is False and abs(bad["abs_dce"] - 2e-3) < 1e-9
    pv = GG.g0_5({"provisional": {"local_ce": 4.7841, "h100_ce": 4.7845}}, {})
    assert pv["pass"] is None and pv["provisional_pass"] is True
    # G0-6: probe JSON + per-window file; strong between-cell spread -> reliable; heavy window noise -> not
    for noise, expect in ((0.02, True), (3.0, False)):
        d = tmp_path / f"probe_{noise}"
        d.mkdir()
        eps = 0.0025
        labs = ["E_ALL"] + [f"C1_german_d{i}_{s}" for i in range(4) for s in "AB"]
        corp = np.repeat(["code", "dclm", "math"], 16)
        arr = {"corpus": corp, "eps": np.array(eps), "primary_routing": np.array("pinned")}
        subsets = {}
        for j, lab in enumerate(labs):
            level = np.exp(0.8 * j + np.array([{"code": 0.0, "dclm": 0.5, "math": 1.0}[c] for c in corp]))
            q = level * np.exp(noise * np.random.default_rng(j).standard_normal(len(corp)))
            kl = q * eps * eps                                       # (kl+ + kl-) / (2 eps^2) = q
            for sg in ("+1", "-1"):
                arr[f"pinned|{lab}|{eps:.6g}|{sg}|kl"] = kl
            subsets[lab] = {}
        np.savez(str(d / "probe__german__windows.npz"), **arr)
        (d / "probe__german.json").write_text(json.dumps({"target": "german", "eps": eps, "primary_routing": "pinned",
                                                          "windows_file": "probe__german__windows.npz", "subsets": subsets}))
        g6 = GG.g0_6({"b": {"dir": str(d)}}, ["b"], {})["b"]
        assert g6["status"] == "computed" and g6["per_target"]["german"]["cells"] == 27
        assert (g6["reliability"] >= 0.9) is expect and g6["pass"] is expect


def test_g0_gate_cli_writes_record_with_hashes(PS, tmp_path):
    w = world(PS, "moe", 1.0, seed=1)
    make_g0_dir(str(tmp_path / "g0" / "moe_b1"), tabs(PS, w), tabs(PS, w, "w_est"), layout="crossfit", n_win=4)
    p = tmp_path / "g0" / "cfg.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe"}},
                                                "code_streams": ["code"]}).encode())    # BOM tolerated
    out = tmp_path / "g0" / "gate_record.json"
    assert GG.main(["--config", str(p), "--out", str(out), "--draws", "50"]) == 0
    rec = json.load(open(out))
    assert len(rec["script"]["sha256"]) == 64 and len(rec["crossover"]["sha256"]) == 64
    assert any("crossfit" in d for d in rec["deviations"]) and any("50 draws" in d for d in rec["deviations"])
    assert all(len(h) == 64 for h in rec["inputs"].values()) and len(rec["inputs"]) == 5
    r2 = GG.compute(GG.load_config(str(p)), n_draws=50, log=lambda *a: None)          # deterministic
    assert r2["gates"]["G0-1"]["cells"] == rec["gates"]["G0-1"]["cells"]


@pytest.mark.skipif(not os.path.isdir(SMOKE), reason="real-checkpoint smoke files not staged")
def test_g0_gate_on_real_nestor_smoke_outputs(tmp_path):
    t = CX.reg_load_dir(SMOKE, "priv")
    assert t.L == 11 and t.N == 64 and t.excluded == ["priv:b6:l0", "priv:b6:l1"] and CX.reg_k(t) == 16
    cfg = _gate_cfg(tmp_path, {"nestor": {"dir": SMOKE, "arch": "nestor", "k": 16}}, code_streams=["code", "pycode"])
    t0 = time.time()
    rec = GG.compute(cfg, n_draws=1000, log=lambda *a: None)
    assert time.time() - t0 < 300
    assert set(rec["gates"]) == {"G0-1", "G0-2", "G0-3", "G0-4", "G0-5", "G0-6", "G0-7"}
    assert rec["gates"]["G0-3"].get("status") == "missing" and rec["gates"]["G0-5"]["status"] == "missing"


def test_g0_gate_flags_mixed_checkpoints_in_one_base(PS, tmp_path):
    """g0_probe resumes from existing routing files and resolves 'glob:' checkpoints to the newest match, so a base
    directory can silently mix checkpoints; the gate records the checkpoints and raises a deviation."""
    w = world(PS, "moe", 1.0, seed=1)
    d = str(tmp_path / "moe_b1")
    make_g0_dir(d, tabs(PS, w), tabs(PS, w, "w_est"))
    rec_ok = GG.compute(_gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, code_streams=["code"]),
                        n_draws=50, log=lambda *a: None)
    assert rec_ok["checkpoints"]["moe_b1"] == ["ckpt_a.pt"] and not any("different checkpoints" in x for x in rec_ok["deviations"])
    L, lks = w["L"], [f"moe:b0:l{l}" for l in range(w["L"])]
    _write_routing(os.path.join(d, "routing__math_eval.npz"), [{lk: tabs(PS, w, "w_est")["math"][l] for l, lk in enumerate(lks)}] * 4,
                   checkpoint="ckpt_b.pt")
    rec = GG.compute(_gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, code_streams=["code"]),
                     n_draws=50, log=lambda *a: None)
    assert rec["checkpoints"]["moe_b1"] == ["ckpt_a.pt", "ckpt_b.pt"]
    assert any("2 different checkpoints" in x for x in rec["deviations"])



# ----------------------------------------------------------------------------- the design additions
def test_code_stream_rule_uses_selection_statistics_only(PS, tmp_path):
    """the choice may not depend on estimation-window statistics."""
    w = world(PS, "moe", 1.0, seed=1)
    sel, est = tabs(PS, w), tabs(PS, w, "w_est")
    z = np.log(sel["code"]) - np.log(sel["math"])
    x = sel["code"] * np.exp(1.0 * z)
    sel["pycode"] = x / x.sum(1, keepdims=True)                 # stronger code-math tilt in SELECTION statistics
    est["pycode"] = sel["math"].copy()                          # ...but no contrast at all on the estimation windows
    make_g0_dir(str(tmp_path / "moe_b1"), sel, est)
    rec = GG.compute(_gate_cfg(tmp_path, {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}), n_draws=100, log=lambda *a: None)
    cells = rec["gates"]["G0-1"]["cells"]
    assert cells["moe_b1|german|code"]["median"] > cells["moe_b1|german|pycode"]["median"]   # estimation favours code
    assert rec["decisions"]["code_stream"]["chosen"] == "pycode"                              # selection decides
    assert "SELECTION statistics only" in rec["decisions"]["code_stream"]["note"]


def _write_npz_ce(path, ce, blk=4096, seed=7):
    n = len(ce)
    EW.save_result(path, {"index": np.arange(n), "starts": np.arange(n) * blk, "hash": np.arange(n, dtype=np.uint64) + seed,
                          "n_tok": np.full(n, blk - 1.0), "ce_sum": np.asarray(ce) * (blk - 1), "routing": {}},
                   {"set": {"block": blk}})


def test_g0_5_mean_ce_sources_and_per_window_pairs(tmp_path):
    (tmp_path / "results").mkdir(); (tmp_path / "logs").mkdir(); (tmp_path / "g0" / "xplat").mkdir(parents=True)
    jl = tmp_path / "results" / "continual.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in [
        {"key": "A|nestor_100m|val10", "stage": "pretrain_eval", "model": "nestor_100m", "set": "val10", "ce": 4.9, "tokens": 5000000},
        {"key": "A|nestor_100m|val09", "stage": "pretrain_eval", "model": "nestor_100m", "set": "val09", "ce": 4.7846,
         "tokens": 10004085, "ckpt": "outputs/nestor_100m_b05/run_h100/checkpoints/model_final.pt"}]) + "\n")
    lg = tmp_path / "logs" / "g0_xplat_local_moe_b2_mean.log"
    lg.write_text("loading...\nRESULT " + json.dumps({"ce": 4.6372, "tokens": 10004085}) + "\n")
    rng = np.random.default_rng(3)
    ce = 4.6 + 0.3 * rng.standard_normal(500)
    _write_npz_ce(str(tmp_path / "g0" / "xplat" / "local_nestor.npz"), ce)
    _write_npz_ce(str(tmp_path / "g0" / "xplat" / "h100_nestor.npz"), ce + 1e-4 * rng.standard_normal(500))
    spec = {"bases": {"nestor": {"h100_ce": 4.784113, "local_jsonl": str(jl), "jsonl_key": "A|nestor_100m|val09"},
                      "moe_b2": {"h100_ce": 4.635177, "local_result_log": str(lg)}},
            "per_window_dir": str(tmp_path / "g0" / "xplat")}
    dev = []
    ck = {"nestor": ["G:/x/outputs/nestor_100m_b05/run_h100/checkpoints/model_final.pt"], "moe_b2": ["None"]}
    out = GG.g0_5_v4(spec, {"nestor": {"arch": "nestor"}, "moe_b2": {"arch": "moe"}}, ck, {}, dev)
    nz, mb = out["nestor"], out["moe_b2"]
    assert nz["pass"] is True and abs(nz["abs_dce"] - abs(4.7846 - 4.784113)) < 1e-12 and nz["per_window"][0]["r"] > 0.999
    assert mb["pass"] is False and abs(mb["abs_dce"] - abs(4.6372 - 4.635177)) < 1e-12 and mb["per_window"] == []
    assert "PI decision" in mb["action"] and dev == []
    # wrong token count and a different checkpoint are recorded as deviations; a missing local value is 'pending'
    lg.write_text("RESULT " + json.dumps({"ce": 4.6352, "tokens": 9999999, "ckpt": "outputs/moe_100m_b05s43/local/model_final.pt"}) + "\n")
    dev = []
    out = GG.g0_5_v4(spec, {"moe_b2": {"arch": "moe"}}, {"moe_b2": ["outputs/moe_100m_b05s43/h100/model_final.pt"]}, {}, dev)
    assert out["moe_b2"]["pass"] is True and any("9999999" in d for d in dev) and any("differs from the G0 routing" in d for d in dev)
    spec["bases"]["nestor"]["jsonl_key"] = "A|nestor_100m|nope"
    out = GG.g0_5_v4(spec, {"nestor": {"arch": "nestor"}}, {}, {}, [])
    assert out["nestor"]["pass"] is None and out["nestor"]["status"].startswith("pending")


def _margin_setup(PS, tmp_path, slope=1.0, noise=0.02, seed=1):
    w = world(PS, "moe", 1.0, seed=1)
    d = tmp_path / "g0"
    make_g0_dir(str(d / "moe_b1"), tabs(PS, w), tabs(PS, w, "w_est"))
    cfgp = d / "gate.json"
    cfgp.write_text(json.dumps({"bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, "code_streams": ["code"]}))
    assert GG.main(["--config", str(cfgp), "--out", str(d / "gate_record.json"), "--draws", "100"]) == 0
    pl = GM.plan(str(d / "moe_b1"), str(d / "gate_record.json"), draws=3)
    man, mh = GM._load_manifest(str(d / "moe_b1"))
    rng = np.random.default_rng(seed)
    draws = [{"target": e["target"], "d": e["d"], "seed": e["seed"], "P": e["predicted"]["P"],
              "I_probe": slope * e["predicted"]["P"] + noise * rng.standard_normal()} for e in man["entries"] if e["status"] == "ok"]
    (d / "moe_b1" / "margin__probe.json").write_text(json.dumps({"routing_mode": "pinned", "code_stream": "code",
                                                                  "manifest_sha256": mh, "draws": draws}))
    return d, cfgp, pl, draws


def test_g0_7_margin_from_registered_draws(PS, tmp_path):
    d, cfgp, pl, draws = _margin_setup(PS, tmp_path, slope=1.1, noise=0.02)
    assert pl["draws"] == 6 and pl["per_target"] == {"books": 3, "german": 3} and not pl["infeasible"]
    rec = GG.compute(GG.load_config(str(cfgp)), n_draws=100, log=lambda *a: None)
    g7 = rec["gates"]["G0-7"]["moe_b1"]
    x = np.array([r["I_probe"] - r["P"] for r in draws])
    pbar = np.mean([r["P"] for r in draws])
    D, se = x.mean() / pbar, x.std(ddof=1) / math.sqrt(len(x)) / pbar
    assert g7["n_draws"] == 6 and math.isclose(g7["D"], D, rel_tol=1e-12) and math.isclose(g7["SE"], se, rel_tol=1e-12)
    assert math.isclose(g7["beta0"], min(0.5, max(0.25, abs(D) + 2 * se)), rel_tol=1e-12) and g7["problems"] == []
    assert any(r["row"].startswith("G0-7") for r in rec["appendix_b"])


def test_g0_7_cap_flag_and_tampered_draws(PS, tmp_path):
    d, cfgp, _, _ = _margin_setup(PS, tmp_path, slope=2.0, noise=0.0)
    rec = GG.compute(GG.load_config(str(cfgp)), n_draws=100, log=lambda *a: None)
    g7 = rec["gates"]["G0-7"]["moe_b1"]
    assert g7["beta0"] == 0.5 and g7["margin_limited"] is True and math.isclose(g7["D"], 1.0, rel_tol=1e-9)
    p = d / "moe_b1" / "margin__probe.json"
    rep_ = json.loads(p.read_text())
    rep_["draws"][0]["P"] += 0.01                               # not the registered draw any more
    p.write_text(json.dumps(rep_))
    rec = GG.compute(GG.load_config(str(cfgp)), n_draws=100, log=lambda *a: None)
    assert rec["gates"]["G0-7"]["moe_b1"]["n_draws"] == 5 and any("G0-7 moe_b1" in x for x in rec["deviations"])


def _bin(path, toks, eot=100257):
    np.asarray(toks, dtype=np.uint32).tofile(path)
    json.dump({"token_dtype": "uint32", "eot_token": eot}, open(str(path) + ".meta.json", "w"))
    return str(path)


def test_dedup_check_exact_and_ngram_rates(tmp_path):
    rng = np.random.default_rng(0)
    E = 100257
    docs = [rng.integers(0, 50000, 40).tolist() for _ in range(10)]
    ref_docs = [docs[0], docs[3], rng.integers(0, 50000, 40).tolist(), docs[5][:20] + rng.integers(50000, 60000, 20).tolist()]
    q = _bin(tmp_path / "q.bin", [t for dd in docs for t in dd + [E]])
    r = _bin(tmp_path / "r.bin", [t for dd in ref_docs for t in dd + [E]])
    res = DD.compare(q, {"ref": r}, n=13, log=None)["refs"]["ref"]
    assert res["doc_exact_count"] == 2 and res["doc_ids"] == [0, 3] and math.isclose(res["doc_exact_rate"], 0.2)
    per_doc = 40 - 13 + 1
    shared = 2 * per_doc + (20 - 13 + 1)                        # two full docs + the 20-token shared prefix of doc 5
    assert math.isclose(res["ngram_token_rate"], shared / (10 * per_doc), rel_tol=1e-12)
    assert math.isclose(res["doc_any_ngram_rate"], 0.3) and math.isclose(res["doc_half_ngram_rate"], 0.2)
    # n-grams are taken within documents only: q2 has 18 within-document 13-grams, all in the reference; q3 has no
    # document of >= 13 tokens, so no n-grams at all (none spans the end-of-text token)
    q2 = _bin(tmp_path / "q2.bin", docs[1][:10] + [E] + docs[1][10:] + [E])
    r2 = _bin(tmp_path / "r2.bin", docs[1] + [E])
    assert DD.compare(q2, {"r": r2}, n=13, log=None)["refs"]["r"]["ngram_token_rate"] == 1.0
    q3 = _bin(tmp_path / "q3.bin", docs[1][:8] + [E] + docs[1][8:16] + [E])
    assert DD.compare(q3, {"r": r2}, n=13, log=None)["query_ngrams"] == 0
