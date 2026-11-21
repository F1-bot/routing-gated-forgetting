"""assumption_audit.py realised-direction mode (prereg v1.3 sec. 4.3, decision P-10).

Torch-free: B5 / B6 / A0 statistics on constructed per-window probe values (separable Q -> interaction share 0; a pure
interaction -> large share; equal module factors -> homogeneous; Q <= 0 cells counted and excluded; A0 from |gamma|/Q).
Tiny MoE (torch): the realised run reproduces a direct pinned probe along delta_g = theta_final - theta0 on the group's
rows, rho_g = ||delta_g||, 2 x L groups.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import assumption_audit as AA


def _write(tmp, Q, gam, eps=0.01, n_win=24, contrast=None, noise=0.0, seed=0):
    """Per-window dce arrays such that (dce+ + dce-)/(2 eps^2) = Q[g, c] (+ noise) and (dce+ - dce-)/(2 eps) = gamma."""
    G, C = Q.shape
    names = [f"{'A' if g < G // 2 else 'B'}:l{g % (G // 2)}" for g in range(G)]
    corpora = ["dclm", "code", "math"][:C]
    cor = np.repeat(np.array(corpora), n_win)
    rng = np.random.default_rng(seed)
    arrs = {"corpus": cor.astype(str), "eps": np.array(eps)}
    for g, n in enumerate(names):
        q = np.repeat(Q[g], n_win) * (1.0 + noise * rng.standard_normal(C * n_win))
        gm = np.repeat(gam[g], n_win)
        s, d = q * 2 * eps * eps, gm * 2 * eps                  # dce+ + dce-, dce+ - dce-
        arrs[f"{n}|{eps:.6g}|+1|dce"] = (s + d) / 2
        arrs[f"{n}|{eps:.6g}|-1|dce"] = (s - d) / 2
    np.savez(os.path.join(tmp, AA.REAL_WINDOWS), **arrs)
    cvec = contrast if contrast is not None else np.linspace(-1, 1, G)
    groups = [{"name": n, "side": n[0], "layer": int(n.split(":l")[1]), "rho": 1.0, "contrast": float(cvec[i])}
              for i, n in enumerate(names)]
    json.dump({"base": "moe_b1", "groups": groups, "eps": eps}, open(os.path.join(tmp, AA.REAL_GROUPS), "w"))


def test_separable_Q_has_zero_interaction_and_A0_holds(tmp_path):
    a = np.log(np.linspace(1.0, 3.0, 24))
    b = np.log(np.array([1.0, 2.0, 0.5]))
    Q = np.exp(a[:, None] + b[None, :])
    _write(str(tmp_path), Q, 0.05 * Q)
    rep = AA.analyze_realized(str(tmp_path), n_boot=50)
    assert rep["B5"]["interaction_share"] == pytest.approx(0.0, abs=1e-12)
    assert rep["B5"]["label"] == "approximately separable" and rep["B5"]["cells_Q_nonpositive"] == 0
    for c in ("dclm", "code", "math"):
        assert rep["A0"][c]["median_r"] == pytest.approx(0.05) and rep["A0"][c]["label"] == "A0 approximately holds"


def test_pure_interaction_is_non_separable(tmp_path):
    rng = np.random.default_rng(3)
    Q = np.exp(rng.normal(0.0, 1.0, (24, 3)) - rng.normal(0.0, 1.0, (24, 3)).mean(1, keepdims=True) * 0)
    Y = np.log(Q)
    Y = Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()      # remove both main effects exactly
    _write(str(tmp_path), np.exp(Y), np.zeros_like(Y))
    rep = AA.analyze_realized(str(tmp_path), n_boot=50)
    assert rep["B5"]["interaction_share"] == pytest.approx(1.0, abs=1e-9) and rep["B5"]["label"] == "non-separable"


def test_homogeneous_module_factors_and_nonpositive_cells(tmp_path):
    Q = np.ones((24, 3))
    Q[5, 1] = -0.1                                          # one Q <= 0 cell
    _write(str(tmp_path), Q, 0.5 * np.abs(Q))
    rep = AA.analyze_realized(str(tmp_path), n_boot=20)
    assert rep["B5"]["cells_Q_nonpositive"] == 1
    for s in ("A", "B"):
        assert rep["B6"][s]["cv"] == pytest.approx(0.0, abs=1e-12) and rep["B6"][s]["label"] == "homogeneous"
        assert rep["B6"][s]["jensen_gap"] == pytest.approx(0.0, abs=1e-12)
    assert rep["A0"]["dclm"]["share_r_ge_0.2"] == 1.0 and rep["A0"]["dclm"]["label"] == "A0 violated"
    assert rep["A0"]["code"]["n"] == 23


def test_heterogeneous_correlated_factor(tmp_path):
    c = np.linspace(-1, 1, 24)
    a = 1.5 * np.concatenate([c[:12], c[12:]])
    Q = np.exp(a[:, None] + np.zeros((1, 3)))
    _write(str(tmp_path), Q, np.zeros_like(Q), contrast=c, noise=0.01)
    rep = AA.analyze_realized(str(tmp_path), n_boot=200)
    for s in ("A", "B"):
        assert rep["B6"][s]["cv"] > 0.25 and rep["B6"][s]["spearman"] > 0.9
        assert rep["B6"][s]["label"] == "heterogeneous, correlated"


def test_realized_run_on_a_tiny_moe(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import crossover as CX
    import eval_windows as EW
    import g0_gate as GG
    import g0_margin as GM
    import g0_probe as G0
    import test_phase6 as T6

    streams = {}
    for i, n in enumerate(("dclm", "code", "math", "books")):
        p = T6._write_bin(tmp_path / f"{n}.bin", 64 * 48, i)
        streams[n] = {"path": p, "tokens": 64 * 8, "select": "even", "span": [0, 0.5]}
        streams[n + "_eval"] = {"path": p, "tokens": 64 * 10, "select": "even", "span": [0.5, 1]}
    d = tmp_path / "g0" / "moe_b1"
    model, dev = T6._tiny_moe()
    G0.run({"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": str(d), "block": 64, "device": dev,
            "streams": streams}, model=model, log=None)
    gcfg = tmp_path / "g0" / "gate.json"
    json.dump({"bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, "code_streams": ["code"],
               "c1_targets": {"moe": ["books"], "nestor": []}}, open(gcfg, "w"))
    rec = GG.compute(GG.load_config(str(gcfg)), n_draws=30, log=lambda *a: None)
    rp = tmp_path / "g0" / "gate_record.json"
    json.dump(rec, open(rp, "w"), default=float)
    t = CX.reg_load_dir(str(d), "moe")
    monkeypatch.setattr(AA, "base_cell", lambda g, b, target="books": {"dir": str(d), "code": "code", "cell": {},
                                                                         "layout": "auto", "half_seed": 0, "family": "moe"})
    specs = {"A": CX.spec_for([t.keys[l][0] for l in range(t.L)]), "B": CX.spec_for([t.keys[l][1] for l in range(t.L)])}
    m0 = T6._tiny_moe()[0]
    rows = G0.unit_rows(m0)
    th0 = {n: p.detach().clone() for n, p in m0.named_parameters()}
    g = torch.Generator().manual_seed(7)
    finals = {}
    for si, side in enumerate(("A", "B")):
        fin = {n: v.clone() for n, v in th0.items()}
        for l in range(t.L):
            for name, r in rows[t.keys[l][si]]:
                v = fin[name][r] if r is not None else fin[name]
                v += 0.05 * th0[name].std().clamp_min(1e-3) * torch.randn(v.shape, generator=g)
        finals[side] = fin
    out = tmp_path / "audit"
    rep = AA.run_realized(str(rp), "moe_b1", str(out), runs=["c1_moe_b1_books_d1_A", "c1_moe_b1_books_d1_B"], specs=specs,
                          final_states=finals, windows=4, device=dev, model=T6._tiny_moe()[0], experiment="tiny_moe", log=None)
    gd = json.load(open(out / AA.REAL_GROUPS))
    assert [x["name"] for x in gd["groups"]] == [f"{s}:l{l}" for s in ("A", "B") for l in range(t.L)]
    assert gd["eps"] == AA.EPS0 and "not found" in gd["eps_source"]
    assert set(rep) >= {"B5", "B6", "A0", "Q", "gamma"} and rep["B5"]["cells"] == 2 * t.L * 3
    # direct reference for group A:l0
    gA = gd["groups"][0]
    dA = {}
    for k in gA["keys"]:
        for name, r in rows[k]:
            dA.setdefault(name, torch.zeros_like(th0[name]))
            if r is None:
                dA[name] = finals["A"][name] - th0[name]
            else:
                dA[name][r] = finals["A"][name][r] - th0[name][r]
    assert gA["rho"] == pytest.approx(math.sqrt(G0.sq_norm(dA)), rel=1e-9)
    win = []
    for c in ("dclm", "code", "math"):
        ws = GM.estimation_windows(str(d), c, 4)
        win += [(ws, i) for i in range(len(ws.index))]
    m2 = T6._tiny_moe()[0]
    eps = gd["eps"]
    res = G0.directional_probe(m2, win, [("x", dA, eps, +1), ("x", dA, eps, -1)], device=dev, log=None, pin_routing=True,
                               count_flips=False)
    q = (res[f"x|{eps:.6g}|+1"]["dce"][:4] + res[f"x|{eps:.6g}|-1"]["dce"][:4]) / (2 * eps * eps)
    gm = (res[f"x|{eps:.6g}|+1"]["dce"][:4] - res[f"x|{eps:.6g}|-1"]["dce"][:4]) / (2 * eps)
    assert rep["Q"][0][0] == pytest.approx(float(q.mean()), rel=1e-9, abs=1e-15)
    assert rep["gamma"][0][0] == pytest.approx(float(gm.mean()), rel=1e-9, abs=1e-15)
