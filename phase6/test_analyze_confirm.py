"""Tests of analyze_confirm.py (prereg v1.3 version 6): known answers by exact enumeration and the mock world end to end.

  python -m pytest test_analyze_confirm.py --noconftest -q
  NESTOR_MOCK_ROOT=<root built by chain6b.ps1 -Mock>  python -m pytest test_analyze_confirm.py --noconftest -q
      adds the end-to-end tests: a copy of that root gets SIMULATED confirm-queue outputs (mock_confirm.py) and the
      analysis is compared with the generating values.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_confirm as AC  # noqa: E402


def brute(x, two_sided=False):
    x = np.asarray(x, dtype=float)
    T = x.mean()
    n = 0
    for s in itertools.product((1, -1), repeat=len(x)):
        m = float(np.mean(np.array(s) * x))
        n += (abs(m) >= abs(T) - 1e-12) if two_sided else (m >= T - 1e-12)
    return n / 2 ** len(x)


# ============================================================================= exact sign-flip test
@pytest.mark.parametrize("D", [1, 2, 3, 5, 8, 12])
def test_signflip_equals_brute_force_enumeration(D):
    rng = np.random.default_rng(D)
    for rep in range(12):
        x = rng.normal(0.1, 0.3, D)
        if rep % 3 == 0:
            x = np.round(x, 1)                      # ties
        assert AC.signflip_p(x)["p"] == brute(x)
        assert AC.signflip_p(x, "two-sided")["p"] == brute(x, True)


def test_signflip_known_answers_and_registered_bounds():
    x = np.arange(1, 13) / 10.0
    assert AC.signflip_p(x)["p"] == 1 / 4096                       # D = 12: 4,096 vectors, minimum p
    y = x.copy()
    y[0] = -y[0]
    assert AC.signflip_p(y)["p"] == 2 / 4096                       # one negative J, the smallest in magnitude
    # C2: 8 draws, v1.2 sec. 8.2 item 3: p >= 2^k / 256 with k negative J2, attained when the negatives are small
    # against every positive (then only flips of the negatives reach T); larger negatives give larger p
    for k, bound in ((1, 0.0078), (2, 0.0156), (3, 0.0313), (4, 0.0625)):
        w = np.concatenate([-0.001 * np.arange(1, k + 1), np.arange(1, 9 - k, dtype=float)])
        assert AC.signflip_p(w)["p"] == 2 ** k / 256 and abs(2 ** k / 256 - bound) < 1e-4
        v = np.concatenate([-np.arange(1, k + 1) / 10.0, np.arange(k + 1, 9) / 10.0])
        assert AC.signflip_p(v)["p"] >= 2 ** k / 256 and AC.signflip_p(v)["p"] == brute(v)
    r = AC.signflip_p(np.arange(1, 21) / 10.0)                     # D = 20: 1,048,576 vectors
    assert r["vectors"] == 1048576 and r["p"] == 2 ** -20
    assert AC.signflip_p([0.3, 0.2, 0.1])["p"] == 1 / 8            # third base / NESTOR-books minimum
    assert AC.signflip_p([0.3, 0.2, 0.1, 0.4, 0.5, 0.6])["p"] == 1 / 64   # C1-DE: all six positive -> p <= 0.025


# ============================================================================= test-inversion CI
def _grid_ci(I, P, lo=-2.0, hi=3.0, n=50001):
    SI, SP = AC.signsums(I), AC.signsums(P)
    g = np.linspace(lo, hi, n)
    acc = np.array([np.mean(np.abs(SI - b * SP) >= abs(SI[0] - b * SP[0]) - 1e-12) > 0.05 for b in g])
    return g[acc].min(), g[acc].max(), (hi - lo) / (n - 1)


@pytest.mark.parametrize("seed,D", [(1, 8), (2, 10), (3, 12)])
def test_test_inversion_matches_a_dense_grid(seed, D):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.4, 1.0, D)
    I = 0.6 * P + rng.normal(0, 0.08, D)
    ci = AC.ci_invert(I, P)
    glo, ghi, step = _grid_ci(I, P)
    assert ci["bounded"] and ci["convex"]
    assert abs(ci["lo"] - glo) <= step and abs(ci["hi"] - ghi) <= step
    assert ci["lo"] <= ci["beta_hat"] <= ci["hi"]
    assert abs(ci["beta_hat"] - float((I * P).sum() / (P * P).sum())) < 1e-15


def test_test_inversion_is_unbounded_below_six_draws():
    rng = np.random.default_rng(0)
    for D in (3, 4, 5):
        P = rng.uniform(0.4, 1.0, D)
        assert AC.ci_invert(0.5 * P, P)["bounded"] is False         # min two-sided p = 2 / 2^D > 0.05
    P = rng.uniform(0.4, 1.0, 6)
    assert AC.ci_invert(0.5 * P + rng.normal(0, 0.05, 6), P)["bounded"] is True


# ============================================================================= Holm, HC3, Welch, t
def test_holm_family_is_fixed_at_the_freeze(monkeypatch):
    """MF-1 / decision (b): family [C1, C2] with no C2 test statistic: C1 enters with raw p 0.04, C2 with
    p = 1, Holm m = 2 -> adjusted 0.08 and C1 is not 'confirmed' (the shrinking family gave 0.04, 'confirmed')."""
    import types
    import analyze_confirm_ext as X

    est = {"overall": {"lo": 0.3, "hi": 0.9}}
    c1 = {"status": None, "estimators": {k: est for k in ("raw", "pin", "sym", "rand")}, "margin_limited": {},
          "mean_J_per_base": {"moe_b1": {"mean_J": 0.3}, "moe_b2": {"mean_J": 0.2}}, "confirmatory": True,
          "E1": {"block_inconclusive": False, "diverged": 0, "runs": 40}, "test": {"p": 0.04}}
    assert AC.decide_c1(c1, 0.04)["label"] == "confirmed"                 # the block itself is confirmable at 0.04
    c2 = {"status": "no usable draws"}
    monkeypatch.setattr(AC, "block_c1", lambda ctx: c1)
    monkeypatch.setattr(AC, "block_c2", lambda ctx: c2)
    for fn in ("block_b3", "block_c1de", "block_nestor_books", "block_selpin", "block_dose", "block_m12b"):
        monkeypatch.setattr(AC, fn, lambda *a, **k: {})
    for fn in ("block_audit", "block_forecasts", "block_budget", "block_edge"):
        monkeypatch.setattr(AC, fn, lambda ctx: {})
    monkeypatch.setattr(X, "run_ext", lambda ctx, mod: {})
    monkeypatch.setattr(X, "not_computable", lambda blocks: [])
    ctx = types.SimpleNamespace(params={"holm_family": ["C1", "C2"], "m": 2}, warnings=[], p=lambda x: os.path.join("nowhere", x),
                                json=lambda x: {}, rel=lambda x: x, freeze_path="FREEZE.json", freeze={}, verify={}, e1={},
                                missing=[], reads={})
    doc = AC.run_all(ctx)
    assert doc["holm"]["p"] == {"C1": 0.04, "C2": 1.0} and doc["holm"]["m"] == 2
    assert doc["holm"]["adjusted"]["C1"] == pytest.approx(0.08, abs=1e-15) and doc["holm"]["entered_with_p_1"] == ["C2"]
    assert doc["decisions"]["C1"]["label"] != "confirmed" and doc["decisions"]["C1"]["label"] == "inconclusive"


def test_e1_divergent_draw_is_dropped_and_the_25_percent_rule(monkeypatch):
    """MF-2 / decision (c): a draw with a divergent run is dropped (no replacement) and counted; more than 25 %
    divergent runs make the block's confirmatory test inconclusive."""
    class S:
        def __init__(self, div):
            self.div = set(div)

        def usable(self, rid):
            return (False, "E1 divergence: loss spike") if rid in self.div else (True, None)

        def e1_of(self, rid):
            return {"diverged": rid in self.div}

    draws = [{"base": "moe_b1", "d": d, "A": f"a{d}", "B": f"b{d}"} for d in range(1, 5)]
    ok, dropped, e1 = AC.Ctx.screen(S({"a2"}), draws)
    assert [x["d"] for x in ok] == [1, 3, 4] and [x["d"] for x in dropped] == [2]
    assert e1["runs"] == 8 and e1["diverged"] == 1 and e1["block_inconclusive"] is False      # 1/8 = 12.5 %
    ok, dropped, e1 = AC.Ctx.screen(S({"a2", "b3"}), draws)
    assert e1["diverged"] == 2 and e1["block_inconclusive"] is False                          # 2/8 = 25 %: not more
    ok, dropped, e1 = AC.Ctx.screen(S({"a1", "b2", "a3"}), draws)
    assert [x["d"] for x in ok] == [4] and e1["diverged"] == 3 and e1["block_inconclusive"] is True
    est = {"overall": {"lo": 0.3, "hi": 0.9}}
    c1 = {"status": None, "estimators": {k: est for k in ("raw", "pin", "sym", "rand")}, "margin_limited": {},
          "mean_J_per_base": {"moe_b1": {"mean_J": 0.3}, "moe_b2": {"mean_J": 0.2}}, "confirmatory": True, "E1": e1}
    d = AC.decide_c1(c1, 0.001)
    assert d["label"] == "inconclusive" and any("E1" in r for r in d["reasons"])


def test_c1de_label_is_descriptive():
    """MF-6 / decision (e): 'sign-consistent with C1' / 'sign-inconsistent' / 'mixed'; never a test label."""
    pd_ = lambda js: {"per_draw": [{"J": j} for j in js]}
    assert AC._label_c1de(pd_([0.1, 0.2, 0.05, 0.3, 0.01, 0.2])) == "sign-consistent with C1"
    assert AC._label_c1de(pd_([-0.1, -0.2, -0.05, -0.3, -0.01, -0.2])) == "sign-inconsistent"
    assert AC._label_c1de(pd_([0.1, -0.2, 0.05, 0.3, 0.01, 0.2])) == "mixed"
    assert AC._label_c1de({"status": "no usable draws"}) == "no usable draws"
    assert set(AC.C1DE_LABELS) == {"sign-consistent with C1", "sign-inconsistent", "mixed"}
    assert "nominal under independence" in AC.C1DE_NOMINAL


def _p_two_sided_matrix(I, P, b):
    """Independent two-sided exact sign-flip p of I - b P (explicit sign matrix; the script's tie tolerance)."""
    D = len(I)
    S = ((np.arange(2 ** D)[:, None] >> np.arange(D)[None, :]) & 1).astype(np.int8) * -2 + 1
    tol = 1e-12 * max(1.0, float(np.abs(I).sum()) + abs(b) * float(np.abs(P).sum()))
    return np.count_nonzero(np.abs(S @ (I - b * P)) >= abs(float((I - b * P).sum())) - tol) / 2 ** D


def test_ci_invert_d20_known_answer_and_timing():
    """R7: D = 20 (the registered C1 at n_d = 5): ci_invert equals an independent bisection of the explicit
    2^20-vector two-sided test (known answer to 1e-8), p crosses alpha at both bounds, and the inversion takes < 60 s."""
    import time
    P = np.array([0.922419, 0.491453, 0.616457, 0.507843, 0.805047, 0.915892, 0.905809, 0.405991, 0.886487, 0.570178,
                  0.481974, 0.881835, 0.652812, 0.582998, 0.401036, 0.783955, 0.463116, 0.807674, 0.509697, 0.8607])
    I = np.array([0.309802, 0.119218, 0.508933, -0.171843, 0.662391, 0.649211, 0.517054, 0.327097, 0.672947, 0.442501,
                  0.312992, 0.65528, 0.428083, 0.255235, 0.13123, 0.693427, 0.346968, 0.31453, -0.358657, 0.753999])
    t0 = time.time()
    r = AC.ci_invert(I, P)
    dt = time.time() - t0
    assert r["D"] == 20 and r["bounded"] and r["convex"] and dt < 60.0
    assert r["lo"] == pytest.approx(0.3891253412, abs=1e-8) and r["hi"] == pytest.approx(0.7155664205, abs=1e-8)
    for b, inside in ((r["lo"] + 1e-7, True), (r["lo"] - 1e-7, False), (r["hi"] - 1e-7, True), (r["hi"] + 1e-7, False)):
        assert bool(_p_two_sided_matrix(I, P, b) > AC.ALPHA) is inside


def test_c1de_manifest_time_limit_hits_are_listed(tmp_path, monkeypatch):
    """R6: a C1-DE draw whose MILP solve hit the time limit is listed (not reproducible on another machine)."""
    import types
    man = {"base": "moe_b2", "entries": [
        {"d": 1, "seed": 5601, "status": "ok", "A": [[0]], "B": [[1]], "generator": {"attempts": [{"message": "Optimal"}]}},
        {"d": 2, "seed": 5602, "status": "milp_failed",
         "generator": {"time_limit_hit": True, "attempts": [{"message": "Time limit reached", "time_limit_hit": True}]}}]}
    mp = tmp_path / "c1de_moe_b2_german.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    monkeypatch.setattr(AC, "_small_block", lambda *a, **k: {"status": "no usable draws"})
    ctx = types.SimpleNamespace(params={}, p=lambda x: x, json=lambda x: json.load(open(x, encoding="utf-8")),
                                plan={"blocks": {"_manifests": {"C1-DE:moe_b2:german": {"path": str(mp), "sha256": "x"}}},
                                      "runs": []})
    out = AC.block_c1de(ctx, {})
    m = out["manifests"]["C1-DE:moe_b2:german"]
    assert [r["time_limit_hit"] for r in m["draws"]] == [False, True] and m["time_limit_hits"] == [2]
    assert out["time_limit_draws"] == ["moe_b2:german:d2"]


def test_holm_step_down():
    assert AC.holm({"C1": 0.03, "C2": 0.01}) == {"C2": 0.02, "C1": 0.03}
    assert AC.holm({"C1": 0.02, "C2": 0.04}) == {"C1": 0.04, "C2": 0.04}
    assert AC.holm({"C1": 0.2}) == {"C1": 0.2}


def test_ols_hc3_equals_the_explicit_formula():
    rng = np.random.default_rng(5)
    P = np.concatenate([rng.normal(0, 0.1, 2), rng.uniform(0.35, 0.65, 4), rng.uniform(0.6, 1.0, 12)])
    y = 0.6 * P + rng.normal(0, 0.05, P.size)
    X = np.column_stack([P, P ** 3])
    r = AC.ols_hc3(y, X, ["b1", "b3"])
    Xi = np.linalg.inv(X.T @ X)
    b = Xi @ X.T @ y
    e = y - X @ b
    V = np.zeros((2, 2))
    for i in range(P.size):
        h = X[i] @ Xi @ X[i]
        V += np.outer(X[i], X[i]) * (e[i] / (1 - h)) ** 2
    V = Xi @ V @ Xi
    assert abs(r["coef"]["b1"]["b"] - b[0]) < 1e-12 and abs(r["coef"]["b3"]["b"] - b[1]) < 1e-12
    assert abs(r["coef"]["b1"]["se_hc3"] - math.sqrt(V[0, 0])) < 1e-12
    from scipy import stats
    t = b[1] / math.sqrt(V[1, 1])
    assert abs(r["coef"]["b3"]["p_two_sided"] - 2 * stats.t.sf(abs(t), P.size - 2)) < 1e-12


def test_welch_and_t_interval():
    from scipy import stats
    w = AC.welch_diff(0.7, 0.05, 3, 0.6, 0.02, 12)
    df = (0.05 ** 2 + 0.02 ** 2) ** 2 / (0.05 ** 4 / 2 + 0.02 ** 4 / 11)
    assert abs(w["df"] - df) < 1e-12
    assert abs(w["hi"] - (0.1 + stats.t.ppf(0.975, df) * math.sqrt(0.05 ** 2 + 0.02 ** 2))) < 1e-12
    x = [0.1, 0.2, 0.15]
    ti = AC.t_interval(x)
    assert ti["df"] == 2 and abs(ti["hi"] - (0.15 + stats.t.ppf(0.975, 2) * np.std(x, ddof=1) / math.sqrt(3))) < 1e-12


# ============================================================================= decision rules
def _c1(p, mj, ub, ml=None, sym=None, rand=None, e1=False):
    est = {k: {"overall": {"beta_hat": u - 0.1, "lo": u - 0.2, "hi": u}} for k, u in zip(("raw", "pin", "sym"), ub)}
    est["sym"]["overall"] = sym or est["sym"]["overall"]
    est["rand"] = {"overall": rand or {"beta_hat": 0.0, "lo": -1.0, "hi": 1.0}}
    return {"estimators": est, "mean_J_per_base": {"moe_b1": {"mean_J": mj[0], "n": 6}, "moe_b2": {"mean_J": mj[1], "n": 6}},
            "margin_limited": ml or {"moe_b1": False, "moe_b2": False}, "E1": {"block_inconclusive": e1, "diverged": 0, "runs": 24},
            "ood": {}, "secondary": {}, "confirmatory": True}


def test_decide_c1_labels():
    D = AC.decide_c1
    assert D(_c1(0.01, (0.1, 0.2), (0.6, 0.6, 0.6)), 0.01)["label"] == "confirmed"
    assert D(_c1(0.01, (0.1, 0.2), (0.6, 0.6, 0.6), ml={"moe_b1": True}), 0.01)["label"] == "confirmed (margin-limited)"
    assert D(_c1(0.01, (0.1, -0.2), (0.6, 0.6, 0.6)), 0.01)["label"].startswith("confirmed in the pooled analysis only")
    assert D(_c1(0.4, (0.1, 0.2), (0.2, 0.2, 0.2)), 0.4)["label"] == "refuted"
    assert D(_c1(0.9, (-0.1, -0.2), (-0.1, -0.05, -0.2)), 0.9)["label"] == "refuted, reversed"
    r = D(_c1(0.4, (0.1, 0.2), (0.2, 0.3, 0.2)), 0.4)
    assert r["label"] == "inconclusive" and any("beta-hat_pin" in x for x in r["reasons"])
    r = D(_c1(0.4, (0.1, 0.2), (0.2, 0.2, 0.2), ml={"moe_b2": True}), 0.4)
    assert r["label"] == "inconclusive" and any("margin-limited" in x for x in r["reasons"])
    assert D(_c1(0.01, (0.1, 0.2), (0.6, 0.6, 0.6), e1=True), 0.01)["label"] == "inconclusive"
    # three-statistic table (upper bound of beta-hat below 0.25)
    t = D(_c1(0.4, (0.1, 0.2), (0.2, 0.3, 0.2), sym={"beta_hat": 0.9, "lo": 0.5, "hi": 1.2},
              rand={"beta_hat": 0.8, "lo": 0.4, "hi": 1.1}), 0.4)
    assert t["three_statistic_table"]["label"].startswith("first-order (transfer) regime")
    t = D(_c1(0.4, (0.1, 0.2), (0.2, 0.2, 0.2), sym={"beta_hat": 0.1, "lo": 0.0, "hi": 0.2},
              rand={"beta_hat": 0.8, "lo": 0.4, "hi": 1.1}), 0.4)
    assert t["three_statistic_table"]["label"] == "learned-update law refuted; routing localization holds"
    t = D(_c1(0.4, (0.1, 0.2), (0.2, 0.2, 0.2), sym={"beta_hat": 0.1, "lo": 0.0, "hi": 0.2},
              rand={"beta_hat": 0.1, "lo": 0.0, "hi": 0.2}), 0.4)
    assert t["three_statistic_table"]["label"].startswith("refuted: routing mass does not localize")


def test_decide_c2_labels():
    base = {"D": 8, "confirmatory": True, "test": {"p": 0.004}, "mean_J2_per_base": {"moe_b1": 0.1, "moe_b2": 0.2},
            "Delta_e": {"beta_hat": 0.6, "lo": 0.3, "hi": 0.9}, "E1": {"block_inconclusive": False},
            "manipulation_check": {"failed": False}}
    assert AC.decide_c2(base, 0.008)["label"] == "confirmed"
    r = copy.deepcopy(base)
    r["Delta_e"]["hi"] = 0.2
    assert AC.decide_c2(r, 0.3)["label"] == "refuted"
    assert AC.decide_c2(base, 0.3)["label"] == "inconclusive"
    r = copy.deepcopy(base)
    r["manipulation_check"]["failed"] = True
    assert AC.decide_c2(r, 0.3)["label"] == "uninformative"
    r = copy.deepcopy(base)
    r["confirmatory"] = False
    assert AC.decide_c2(r, None)["label"] == "exploratory: confirmed"


# ============================================================================= EDGE parsers
def test_edge_power_log_run_log_and_energy(tmp_path):
    from datetime import datetime, timedelta
    p = tmp_path / "gpu_power_20261113_235717.csv"
    rows, s0 = [], datetime(2026, 11, 13, 23, 57, 17, 500000)
    for k in range(40):                    # samples at :17.5 + k s; 100 W up to :26.5, 200 W from :27.5
        x = s0 + timedelta(seconds=k)
        rows.append(f"{x:%Y/%m/%d %H:%M:%S}.{x.microsecond // 1000:03d}, {100.0 if k < 10 else 200.0:.2f} W, 60 %, "
                    f"7156 MiB, 56, 2700 MHz")
    rows.append("2026/11/14 00:")                                     # partial last line (logger running)
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    pw = AC.read_power([str(p)])
    assert pw["rows"] == 40 and pw["unparsed_rows"] == 1 and pw["layout"]["power_W"] == 1 and pw["layout"]["memory_MiB"] == 3
    t0 = AC._sec("2026-11-13 23:57:30", "%Y-%m-%d %H:%M:%S")
    e = AC.window_energy(pw, t0, t0 + 20)                              # [:30, :50] lies in the 200 W part
    assert e["ok"] and abs(e["energy_J"] - 200.0 * 20) < 1e-9 and e["samples"] == 20
    e2 = AC.window_energy(pw, AC._sec("2026-11-13 23:57:26", "%Y-%m-%d %H:%M:%S"), AC._sec("2026-11-13 23:57:28", "%Y-%m-%d %H:%M:%S"))
    # [26, 28]: edge 26 -> 100 W; 26.5 -> 100; 27.5 -> 200; edge 28 -> 200: 0.5*100 + 1*150 + 0.5*200 = 300 J
    assert abs(e2["energy_J"] - 300.0) < 1e-9
    gap = [r for i, r in enumerate(rows[:-1]) if not (20 <= i < 27)]
    p.write_text("\n".join(gap) + "\n", encoding="utf-8")
    e3 = AC.window_energy(AC.read_power([str(p)]), t0 - 5, t0 + 20)
    assert not e3["ok"] and "apart" in e3["reason"]
    log = tmp_path / "run_m2.log"
    log.write_text("\n".join([
        "# .venv/nestor/Scripts/python.exe run_exp.py moe_100m --steps 96",
        "[pretrain] moe_100m_x: d=480 L=12 ctx=4096 | 96 steps x 262,144 tok/step = 0.03B tokens | data=x",
        "2026-11-14 01:13:11 - nestor.moe_100m_x_train - INFO - Run directory: X",
        "2026-11-14 01:16:58 - nestor.moe_100m_x_train - INFO - Step 5 | train/loss: 8.4 | train/sec_per_step: 7.8438 | "
        "train/mem_peak_alloc_gib: 7.6382 | train/mem_peak_reserved_gib: 8.1387 | train/tflops_per_gpu: 28.7",
        "2026-11-14 01:29:12 - nestor.moe_100m_x_train - INFO - Step 95 | train/loss: 7.99 | train/sec_per_step: 7.9688 | "
        "train/mem_peak_alloc_gib: 7.7 | train/mem_peak_reserved_gib: 8.2",
        "2026-11-14 01:29:23 - nestor.moe_100m_x_train - INFO - Training complete!"]) + "\n", encoding="utf-8")
    L = AC.parse_run_log(str(log))
    assert L["header"] == {"steps": 96, "tok_per_step": 262144}
    assert L["t_end"] - L["t_start"] == 16 * 60 + 12 and len(L["steps"]) == 2
    assert max(s["train/mem_peak_reserved_gib"] for s in L["steps"]) == 8.2


def test_edge_run_log_is_the_attempt_of_the_state_root(tmp_path):
    """NESTOR starts at m4c; after an OOM fallback the state root is .../m2c and EDGE reads <run>_m2c.log, not
    the failed attempt's <run>_m4c.log (the last-named one)."""
    for n in ("r_m4c.log", "r_m2c.log", "r_m2.log"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    lg = str(tmp_path)
    for att in ("m2c", "m4c", "m2"):
        assert AC.run_log_path(lg, "r", os.path.join("outputs", "confirm", "r", att)) == os.path.join(lg, f"r_{att}.log")
    assert AC.run_log_path(lg, "r", None) == os.path.join(lg, "r_m4c.log")
    assert AC.run_log_path(lg, "q", None) is None


def test_analysis_is_torch_free():
    code = ("import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch blocked by the test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            "sys.path.insert(0, %r)\n"
            "import analyze_confirm as A, analyze_confirm_ext, gate_common, crossover, assumption_audit, forecasts, chain6b_util\n"
            "rc = A.selftest()\n"
            "assert 'torch' not in sys.modules\n"
            "sys.exit(rc)\n") % HERE
    p = subprocess.run([sys.executable, "-c", code], cwd=HERE, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


# ============================================================================= end to end on the mock world
def _mock_copy(tmp_path_factory, name):
    src = os.environ.get("NESTOR_MOCK_ROOT")
    if not src or not os.path.isfile(os.path.join(src, "FREEZE.json")):
        pytest.skip("NESTOR_MOCK_ROOT (a root built by chain6b.ps1 -Mock) not set")
    dst = tmp_path_factory.mktemp(name) / "r"          # short: Windows MAX_PATH (longest output 113 chars)
    shutil.copytree(src, dst)
    for f in ANALYSIS_FILES:
        shutil.copy2(os.path.join(HERE, f), dst / f)
    return str(dst)


ANALYSIS_FILES = ("analyze_confirm.py", "analyze_confirm_ext.py", "mock_confirm.py")
# the whole analysis runs with torch imports blocked (torch-free check on every end-to-end run)
_NOTORCH = ("import sys, runpy\n"
            "class _NoTorch:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch blocked by the test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _NoTorch())\n"
            "sys.argv = ['analyze_confirm.py'] + sys.argv[1:]\n"
            "runpy.run_path('analyze_confirm.py', run_name='__main__')\n")


def _drift_only_analysis_files(out: str) -> bool:
    """FREEZE verification of a root installed before these files: code drift in the analysis files is the only
    accepted reason (nothing missing; changed / added only analyze_confirm*.py and mock_confirm.py)."""
    import re
    m = re.search(r"DRIFT: changed \[(.*?)\], missing \[(.*?)\], added \[(.*?)\]", out)
    if not m:
        return False
    lists = [[x.strip().strip("'\"") for x in g.split(",") if x.strip()] for g in m.groups()]
    return not lists[1] and set(lists[0] + lists[2]) <= set(ANALYSIS_FILES) and bool(lists[0] + lists[2])


def _sim(root, scen=None):
    argv = [sys.executable, "mock_confirm.py", "simulate", "--root", root]
    if scen:
        sp = os.path.join(root, "scenario.json")
        json.dump(scen, open(sp, "w", encoding="utf-8"))
        argv += ["--scenario", sp]
    p = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return json.load(open(os.path.join(root, "results", "analysis_mock", "truth.json"), encoding="utf-8"))


def _analyze(root, verify: bool = True):
    argv = [sys.executable, "-c", _NOTORCH, "run", "--root", root]
    p = subprocess.run(argv + ([] if verify else ["--no-verify"]), cwd=root, capture_output=True, text=True, timeout=3600)
    verified = verify and p.returncode == 0
    if verify and p.returncode == 23:   # a root installed before the analysis files: code drift is the only accepted reason
        assert _drift_only_analysis_files(p.stdout), p.stdout[-2000:]
        p = subprocess.run(argv + ["--no-verify"], cwd=root, capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    jp = os.path.join(root, "results", "analysis", "analysis_confirm.json")
    doc = json.load(open(jp, encoding="utf-8"))
    assert open(jp + ".sha256", encoding="ascii").read().split()[0] == AC.sha256_file(jp)
    body = {k: v for k, v in doc.items() if k not in ("created_utc", "sha256")}
    assert doc["sha256"] == __import__("hashlib").sha256(AC.canonical(body).encode("utf-8")).hexdigest()
    return doc, verified


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = _mock_copy(tmp_path_factory, "acd")
    truth = _sim(root)
    doc, verified = _analyze(root)
    return root, truth, doc, verified


def test_end_to_end_c1_and_estimators_equal_the_generating_values(e2e):
    root, T, A, _ = e2e
    c1 = A["C1"]
    tr = lambda role, b, d: T["draws"][f"{role}|{b}|{d}"]  # noqa: E731
    assert c1["statistic"] == "log" and c1["D"] == len([k for k in T["draws"] if k.startswith("c1|")])
    for r in c1["per_draw"]:
        t = tr("c1", r["base"], r["d"])
        assert abs(r["I"] - t["I"]["raw"]) < 1e-9 and r["P"] == t["P"]
    for k in ("sym", "pin", "sym_pin", "selpin", "sym_selpin", "rand", "wd"):
        for rec in A["C1"]["statistics"][k]["draws"]:
            assert abs(rec["I_log"] - tr("c1", rec["base"], rec["d"])["I"][k]) < 1e-9
    b0 = {"moe_b1": A["parameters"]["beta0_1"], "moe_b2": A["parameters"]["beta0_2"]}
    J = [tr("c1", r["base"], r["d"])["I"]["raw"] - b0[r["base"]] * r["P"] for r in c1["per_draw"]]
    assert c1["test"]["p"] == brute(J)
    I = np.array([tr("c1", r["base"], r["d"])["I"]["raw"] for r in c1["per_draw"]])
    P = np.array([r["P"] for r in c1["per_draw"]])
    assert abs(c1["estimators"]["raw"]["overall"]["beta_hat"] - float((I * P).sum() / (P * P).sum())) < 1e-9
    if list(A["holm"]["p"]) == ["C1"]:
        assert A["holm"]["adjusted"]["C1"] == c1["test"]["p"]
    assert A["decisions"]["C1"]["label"] == "confirmed"
    for b, cell in c1["ood"].items():                               # I_k / I_96 = 1 by construction
        for k, v in cell["interim_ratio"].items():
            assert abs(v["median"] - 1.0) < 1e-9
        assert cell["out_of_domain"] is False
    s2 = c1["secondary"]["second_code_stream"]
    for x in s2["per_draw"]:
        assert abs(x["I"] - tr("c1", x["base"], x["d"])["I"]["code2"]) < 1e-9


def test_end_to_end_c2_small_blocks_m12b_dose_audit(e2e):
    root, T, A, _ = e2e
    c2 = A["C2"]
    s = str(c2["s_star"]) if c2["s_star"] in (8, 24, 48) else "96"
    for r in c2["per_draw"]:
        t = T["c2"][f"{r['base']}|{r['d']}"]
        assert abs(r["I2"] - (t["r"][f"sgd|{s}"] - t["r"][f"muon|{s}"])) < 1e-9
    assert c2["test"]["p"] == brute([r["J2"] for r in c2["per_draw"]])
    for m in c2["manipulation_check"]["per_draw"]:
        assert abs(m["D_sgd_step8"] - T["c2"][f"{m['base']}|{m['d']}"]["D_log_ratio"]["sgd"]) < 1e-9
    B = A["blocks"]
    for name, role, kind in (("C1_DE", "c1de", "raw"), ("NESTOR_books", "c1_nestor", "sym_pin")):
        b = B[name]
        for r in b["per_draw"]:
            assert abs(r["I"] - T["draws"][f"{role}|{r['base']}|{r['d']}"]["I"][kind]) < 1e-9
        assert b["signflip"]["p"] == brute([r["J"] for r in b["per_draw"]])
        if name == "C1_DE":                         # MF-6 / decision (e): descriptive sign labels
            J = [r["J"] for r in b["per_draw"]]
            want = ("sign-consistent with C1" if all(j > 0 for j in J) else "sign-inconsistent" if all(j < 0 for j in J)
                    else "mixed")
            assert b["label"] == want and "nominal under independence" in b["nominal"]
        else:
            assert b["label"] == ("consistent with C1" if all(r["J"] > 0 for r in b["per_draw"]) else b["label"])
    assert B["NESTOR_books"]["estimators"]["raw"]["overall"]["bounded"] is False      # D = 3: no beta is rejected
    for r in B["M12b"]["per_draw"]:
        assert abs(r["Delta"] - T["m12b"][str(r["d"])]["Delta"]) < 1e-9 and r["common_haar_draws"]
    pc = B["dose"]["pipeline_asymmetry_check"]
    for u in pc["pairs"]:
        assert abs(u["I"] - pc["G2_record_I_AA"][u["pair"][2:]]) < 1e-9          # the A/A level equals the G2 record
    assert B["dose"]["label"] in ("linear dose-response", "non-linear dose-response", "no dose-response")
    for base in ("moe_b1", "moe_b2"):
        assert B["audit"][base]["B5"]["label"] == "approximately separable"


def test_end_to_end_edge_known_energies_and_exclusions(e2e):
    root, T, A, _ = e2e
    ed = A["blocks"]["EDGE"]
    cr = [r for r in ed["runs"] if r["stage"] == "confirm"]
    assert len(cr) == T["edge"]["runs"] - 1
    assert all(abs(r["energy_Wh"] - T["edge"]["run_energy_Wh"]) < 1e-9 and r["wall_s"] == T["edge"]["run_wall_s"] for r in cr)
    ex = [x for x in ed["excluded"] if x["stage"] == "confirm"]
    assert [x["run"] for x in ex] == [T["edge"]["gap_run"]] and "apart" in ex[0]["reason"]
    for b, rec in ed["pre_update_check"]["per_base"].items():
        assert abs(rec["wall_s_per_unit"] - T["edge"]["probe_wall_s"] / rec["units"]) < 1e-9
        assert abs(rec["energy_Wh_per_unit"] - T["edge"]["probe_energy_Wh"] / rec["units"]) < 1e-9


@pytest.fixture(scope="module")
def e2e_stress(tmp_path_factory):
    root = _mock_copy(tmp_path_factory, "acs")
    plan = json.load(open(os.path.join(root, "confirm", "plan_confirm.json"), encoding="utf-8"))
    c1 = sorted(r["id"] for r in plan["runs"] if r.get("role") == "c1" and r["base"] == "moe_b2")
    de = sorted(r["id"] for r in plan["runs"] if r.get("role") == "c1de")
    # antithetic randdir evaluations on the C2 Muon runs (b_Muon^rand) and a French forecast changed after
    # hashing; the plan edit changes its hash, so this root is analysed without FREEZE verification
    for r in plan["runs"]:
        if r.get("role") == "c2_muon":
            r["evals"] = list(r.get("evals") or []) + [{"name": "rand", "sets": ["dclm"], "variants": "randdir:9701,randdir-:9701"}]
    json.dump(plan, open(os.path.join(root, "confirm", "plan_confirm.json"), "w", encoding="utf-8"), indent=1)
    scen = {"f": {"raw": [0.0006, 0.0005]}, "diverge": [c1[0]], "not_done": [de[0]], "c5": {"tamper": True}}
    truth = _sim(root, scen)
    doc, _ = _analyze(root, verify=False)
    return root, truth, doc, scen, plan


def test_end_to_end_unresolved_values_use_the_normalized_difference(e2e_stress):
    root, T, A, scen, plan = e2e_stress
    c1 = A["C1"]
    assert c1["statistic"] == "normalized_difference"
    runs = {r["id"]: r for r in plan["runs"]}
    dropped = {(d["base"], d["d"]) for d in c1["draws_dropped"]}
    div = runs[scen["diverge"][0]]
    assert (div["base"], div["d"]) in dropped and c1["D"] == len([k for k in T["draws"] if k.startswith("c1|")]) - 1
    assert c1["E1"]["diverged"] == 1 and c1["E1"]["block_inconclusive"] is False
    fc, fm = scen["f"]["raw"]
    keep = [(r["base"], r["d"]) for r in c1["per_draw"]]
    for b in {k[0] for k in keep}:          # F-bar over the cell's runs of the test, P_nd from RO (v1.2 sec. 8.1 item 2)
        ds = [d for bb, d in keep if bb == b]
        Fb = np.mean([fc for _ in ds] + [fc * math.exp(T["draws"][f"c1|{b}|{d}"]["I"]["raw"]) for d in ds])
        rows = [r for r in plan["runs"] if r.get("role") == "c1" and r["base"] == b and r["d"] in ds]
        ro = {(r["d"], r["side"]): r["subset"]["ro"] for r in rows}
        rb = {c: np.mean([v[c] for v in ro.values()]) for c in ("code", "math")}
        for r in c1["per_draw"]:
            if r["base"] != b:
                continue
            d = r["d"]
            Ind = (fc * math.exp(T["draws"][f"c1|{b}|{d}"]["I"]["raw"]) - fc) / Fb
            Pnd = (ro[(d, "B")]["code"] - ro[(d, "A")]["code"]) / rb["code"] - (ro[(d, "B")]["math"] - ro[(d, "A")]["math"]) / rb["math"]
            assert abs(r["I"] - Ind) < 1e-9 and abs(r["P"] - Pnd) < 1e-9
    assert c1["test"]["p"] == brute([r["J"] for r in c1["per_draw"]])
    de = A["blocks"]["C1_DE"]
    nd = runs[scen["not_done"][0]]
    assert any((x["base"], x["d"]) == (nd["base"], nd["d"]) for x in de["draws_dropped"]) and len(de["per_draw"]) == 5



# ============================================================================= (analyze_confirm_ext.py)
def _pooled_ref(by_dom: dict):
    """Independent v1.2 sec. 10.1 aggregation: seeds within a domain, then domains; pooled within-domain t-interval."""
    from scipy import stats
    doms = sorted(d for d, v in by_dom.items() if v)
    m = float(np.mean([np.mean(by_dom[d]) for d in doms]))
    df = sum(len(by_dom[d]) - 1 for d in doms)
    if df == 0:
        return m, None, None, 0
    s2 = sum(float(((np.asarray(by_dom[d]) - np.mean(by_dom[d])) ** 2).sum()) for d in doms) / df
    se = math.sqrt(s2 * sum(1.0 / len(by_dom[d]) for d in doms)) / len(doms)
    h = float(stats.t.ppf(0.975, df)) * se
    return m, m - h, m + h, df


def _pair_a(r, a):
    if r["arch"] == "moe":
        return a["revert:moe"], a["revert:kind:attn+embed_head+kind:norm"]
    return a["revert:attn+priv"], a["revert:shared+pub+embed_head"]


def test_end_to_end_c3_decomposition_pooled_interval_and_freeze_check(e2e):
    root, T, A, _ = e2e
    c3 = A["blocks"]["C3"]
    assert not c3["dropped"] and len(c3["per_run"]) == len(T["c3"]["runs"]) == 20      # 14 confirm + 6 G1 FULL runs
    by, n_wd = {}, 0
    for r in c3["per_run"]:
        t = T["c3"]["runs"][r["rid"]]
        ag, ag2 = _pair_a(r, t["a"])                 # F_only g = (1 - a_g') F, F_only g' = (1 - a_g) F
        share, iota = 0.5 * (1.0 + ag - ag2), ag + ag2 - 1.0
        d = r["decomposition"]
        assert abs(r["F"] - t["F"]) < 1e-12 and abs(d["share_g"] - share) < 1e-9 and abs(d["iota_over_F"] - iota) < 1e-9
        assert abs(d["share_g"] + d["share_g2"] - 1.0) < 1e-12 and abs(d["phi_g"] + d["phi_g2"] - r["F"]) < 1e-15
        for v, x in r["reverts"].items():            # single-group revert deltas (descriptive)
            assert abs(x["Delta_over_F"] - t["a"][v]) < 1e-9
        if "WD" in r:
            n_wd += 1
            assert abs(r["WD"]["F_WD_over_F"] - T["scenario"]["c3"]["wd_share"]) < 1e-9
        by.setdefault(r["base"], {}).setdefault(r["domain"], []).append(share)
    assert n_wd >= 2
    for base, v in c3["per_base"].items():
        m, lo, hi, df = _pooled_ref(by[base])
        s = v["shares"]["share_g"]
        assert abs(s["mean"] - m) < 1e-12 and s["df"] == df and len(s["per_domain"]) == 4
        if base == "moe_b2":
            assert df == 0 and s["lo"] is None             # one seed per domain: points only
        else:
            assert df == 4 and abs(s["lo"] - lo) < 1e-12 and abs(s["hi"] - hi) < 1e-12
    fz = c3["freeze_cross_check"]
    for arch, t in T["c3"]["freeze"].items():
        r = fz[arch]
        assert abs(r["F_freeze"] - t["F_freeze"]) < 1e-12 and sorted(r["full_runs"]) == sorted(t["comparators"])
        assert abs(r["revert_attribution"] - (t["F_full_mean"] - t["F_only_mean"])) < 1e-12
        assert abs(r["freeze_attribution"] - (t["F_full_mean"] - t["F_freeze"])) < 1e-12
    assert c3["random_partial_revert"]["status"].startswith("not computable")
    assert "not a path decomposition" in c3["statement"] and c3["nestor_moe_comparison"]["made"] is False


def test_end_to_end_c3_per_layer_deltas_and_additivity_gap(e2e):
    root, T, A, _ = e2e
    pl = A["blocks"]["C3_per_layer"]
    assert len(pl["per_run"]) == 8 and not pl["dropped"]                # 4 domains x (MoE base 1, NESTOR), seed 101
    gap = T["scenario"]["c3"]["layer_gap"]
    for r in pl["per_run"]:
        t = T["c3"]["runs"][r["rid"]]
        assert int(r["seed"]) == 101 and len(r["layers"]) == len(t["layers"])
        for tok, x in r["layers"].items():
            assert abs(x["Delta"] - (t["F"] - t["layers"][x["variant"]])) < 1e-12
        a_all = t["a"][r["all_group_revert"]]
        assert abs(r["Delta_all"] - a_all * t["F"]) < 1e-12
        assert abs(r["additivity_gap"] - gap * a_all * t["F"]) < 1e-12 and abs(r["sum_over_all"] - (1.0 + gap)) < 1e-9
    for base, v in pl["per_base"].items():
        rows = [r for r in pl["per_run"] if r["base"] == base]
        assert len(v["domains"]) == 4
        for tok, u in v["per_unit"].items():
            d = [r["layers"][tok]["Delta"] for r in rows]
            assert abs(u["Delta_mean"] - float(np.mean(d))) < 1e-12 and u["Delta_range"] == [min(d), max(d)]


def test_end_to_end_c4_slopes_intervals_and_routing_files(e2e):
    root, T, A, _ = e2e
    c4 = A["blocks"]["C4"]
    assert not c4["dropped"] and len(c4["per_draw"]) == len([k for k in T["draws"] if k.startswith("c1|")])
    tb = []
    for r in c4["per_draw"]:
        t = T["c4"][f"{r['base']}|{r['d']}"]
        assert abs(r["beta_C4"] - t["beta_C4"]) < 1e-9 and abs(r["F_bar"] - t["F_bar"]) < 1e-12
        tb.append(t["beta_C4"])
    o, ti = c4["overall"], AC.t_interval(tb, 0.95)
    assert abs(o["mean_beta_C4"] - float(np.mean(tb))) < 1e-9 and abs(o["t95"]["lo"] - ti["lo"]) < 1e-9
    lo, hi = o["bootstrap"]["ci95"]
    assert lo < o["mean_beta_C4"] < hi and o["bootstrap"]["B"] == AC.N_BOOT
    assert c4["routing_files"] == {b: f"g0/{b}/routing__dclm_eval.npz" for b in ("moe_b1", "moe_b2")}


def test_c4_two_way_bootstrap_replicates_equal_the_explicit_formula():
    import types
    import analyze_confirm_ext as X
    H = types.SimpleNamespace(SEED_BOOT=AC.SEED_BOOT, N_BOOT=40)
    rng = np.random.default_rng(5)
    D, W = 4, 30
    Xm = rng.normal(size=(D, W))
    Y = 0.3 * Xm + 0.1 * rng.normal(size=(D, W))
    M = 0.5 + 0.05 * rng.normal(size=(D, W))
    got = X._c4_boot(Xm, Y, M, H, 3)
    assert np.array_equal(got, X._c4_boot(Xm, Y, M, H, 3))                   # deterministic
    r2 = np.random.default_rng([AC.SEED_BOOT, 3])
    wi, di = r2.integers(0, W, size=(40, W)), r2.integers(0, D, size=(40, D))
    for b in range(40):
        beta = []
        for j in range(D):
            x, y, m = Xm[j, wi[b]], Y[j, wi[b]], M[j, wi[b]]
            beta.append(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1) / m.mean())
        assert abs(got[b] - float(np.mean(np.array(beta)[di[b]]))) < 1e-10


def test_end_to_end_c5_french_prospective_metrics(e2e):
    from scipy import stats
    root, T, A, _ = e2e
    c5 = A["blocks"]["C5_french"]
    hp = c5["hash_protocol"]
    assert c5["label"] == "prospective" and hp["verifies"] and hp["prospective"] and hp["posted_epoch"] is not None
    F = T["c5"]["F"]
    bench = {c: float(np.mean([math.log(o["F"]) for o in T["c5"]["fit"] if o["corpus"] == c])) for c in ("dclm", "code", "math")}
    for nm in ("ro", "trainable_fraction", "k_step"):
        pts = [(math.log(r["forecasts"][nm]), math.log(F[r["run"]][r["corpus"]]), bench[r["corpus"]])
               for r in T["c5"]["forecasts"] if r["forecasts"][nm] is not None]
        lf, lF, lb = (np.array(v) for v in zip(*pts))
        m = c5["metrics"][nm]["pooled"]
        assert m["n"] == len(pts) == (14 if nm == "k_step" else 15)
        assert abs(m["R2_oos"] - (1.0 - ((lF - lf) ** 2).sum() / ((lF - lb) ** 2).sum())) < 1e-9
        assert abs(m["spearman"] - float(stats.spearmanr(lf, lF).correlation)) < 1e-9
        assert abs(m["median_abs_log_ratio"] - float(np.median(np.abs(lf - lF)))) < 1e-12
    w = c5["R2_within"]["ro"]
    assert w["n"] == 2 and w["R2_within"] is None                             # 2 French pairs: contrasts only
    fc = {(r["run"], r["corpus"]): r["forecasts"]["ro"] for r in T["c5"]["forecasts"]}
    pairs = {}
    for rid in T["c5"]["french"]:
        if "_pair_" in rid:
            pairs.setdefault(rid.rsplit("_", 1)[0], {})[rid.rsplit("_", 1)[1]] = rid
    I = sorted((math.log(F[ab["B"]]["code"] / F[ab["A"]]["code"]) - math.log(F[ab["B"]]["math"] / F[ab["A"]]["math"])) for ab in pairs.values())
    assert np.allclose(sorted(w["I"]), I, atol=1e-12)


def test_end_to_end_matched_gain_known_answers(e2e):
    root, T, A, _ = e2e
    mg, g = A["blocks"]["matched_gain"], T["scenario"]["c1_gain_B"]
    assert len(mg["C1"]["per_draw"]) == len([k for k in T["draws"] if k.startswith("c1|")])
    for r in mg["C1"]["per_draw"]:
        I = T["draws"][f"c1|{r['base']}|{r['d']}"]["I"]["raw"]
        ck = [k for k in r["F_ratio_B_over_A"] if k != "math"][0]
        assert abs(r["I_G"] - I) < 1e-9 and abs(r["G_final"]["B"] / r["G_final"]["A"] - g) < 1e-9
        assert abs(r["F_ratio_B_over_A"][ck] - math.exp(I) / g) < 1e-9 and abs(r["F_ratio_B_over_A"]["math"] - 1.0 / g) < 1e-9
    c2 = [r for r in mg["C2"]["per_draw"] if "I2_G" in r]
    assert len(c2) == len(T["c2"])
    for r in c2:
        t = T["c2"][f"{r['base']}|{r['d']}"]["r"]
        assert abs(r["I2_G"] - (t["sgd|96"] - t["muon|96"])) < 1e-9
    runs = {(t["base"], t["domain"], int(t["seed"])): t for t in T["c3"]["runs"].values() if t["source"] == "confirm"}
    pairs = mg["FULL_across_architectures"]["pairs"]
    exp = [(dom, s, mb) for (b, dom, s) in runs if b == "nestor" for mb in ("moe_b1", "moe_b2") if (mb, dom, s) in runs]
    assert len(pairs) == len(exp) == 7
    for p in pairs:
        tn, tm = runs[("nestor", p["domain"], p["seed"])], runs[(p["pair"].split("/")[1], p["domain"], p["seed"])]
        assert abs(p["F_ratio"] - (tn["F"] / tn["G"]) / (tm["F"] / tm["G"])) < 1e-9
    assert mg["pareto"] and {"run", "step", "G", "F"} <= set(mg["pareto"][0])


def test_end_to_end_frel_and_bits_per_byte(e2e):
    root, T, A, _ = e2e
    fr, code = A["blocks"]["F_rel"], A["C1"]["code_stream"]
    for b, t in T["dclm_cont"].items():
        assert fr["F_cont"][b]["source"] == "confirm" and all(abs(fr["F_cont"][b]["F"][s] - f) < 1e-12 for s, f in t["F"].items())
    g16 = json.load(open(os.path.join(root, "g1", "gate_record_G1.json"), encoding="utf-8"))["gates"]["G1-6"]["values"]["F_cont"]
    assert fr["F_cont"]["moe_b1"]["source"] == "G1" and abs(fr["F_cont"]["moe_b1"]["F"]["dclm"] - g16) < 1e-9
    assert all(abs(r["F_rel_dclm"] - (r["F_dclm"] - fr["F_cont"][r["base"]]["F"]["dclm"])) < 1e-15 for r in fr["FULL"])
    fc, fm = T["scenario"]["f"]["raw"]
    assert fr["C1"]["statistic"] == "log" and fr["C1"]["n"] == len(fr["C1"]["per_draw"])
    for row in fr["C1"]["per_draw"]:
        c = fr["F_cont"][row["base"]]["F"][code]
        I = T["draws"][f"c1|{row['base']}|{row['d']}"]["I"]["raw"]
        assert abs(row["I_rel"] - math.log((fc * math.exp(I) - c) / (fc - c))) < 1e-9     # math terms cancel (F_A = F_B)
    J = [row["I_rel"] - float(fr["C1"]["beta0"][row["base"]]) * row["P"] for row in fr["C1"]["per_draw"]]
    assert fr["C1"]["signflip"]["p"] == brute(J)
    f96 = T["scenario"]["c2"]["f"]["96"]
    for r in fr["C2"]["per_draw"]:
        t, c = T["c2"][f"{r['base']}|{r['d']}"]["r"], fr["F_cont"][r["base"]]["F"]["dclm"]
        lr = {a: math.log((f96 * math.exp(t[f"{a}|96"] / 2) - c) / (f96 * math.exp(-t[f"{a}|96"] / 2) - c)) for a in ("sgd", "muon")}
        assert abs(r["I2_rel"] - (lr["sgd"] - lr["muon"])) < 1e-9
    bp = A["blocks"]["bpb"]
    assert all(abs(bp["kappa"][k]["kappa"] - v) < 1e-12 for k, v in T["bpb"].items() if k in bp["kappa"])
    assert len([k for k in bp["kappa"] if "kappa" in bp["kappa"][k]]) >= 6 and bp["C1_check"]["max_abs_I_bpb_minus_I"] < 1e-12


def test_end_to_end_c2eq_snr_pointers_and_not_computable_list(e2e):
    root, T, A, _ = e2e
    ce = A["blocks"]["C2_eq_rand"]
    nc = A["not_computable_from_registered_files"]
    if T["c2_rand"]:      # a root built by the plan builder: every C2 Muon run evaluates the randdir pair
        assert "status" not in ce and ce["step"] == 96 and len(ce["per_draw"]) == len(T["c2_rand"])
        assert not any(x.startswith("C2_eq_rand") for x in nc)
    else:                 # an older chain-built root (no randdir pair planned)
        assert ce["status"].startswith("not computable")
        assert any(x.startswith("C2_eq_rand") for x in nc)
    g25 = json.load(open(os.path.join(root, "g2", "gate_record_G2.json"), encoding="utf-8"))["gates"]["G2-5"]["values"]
    assert ce["per_expert_gradient_SNR"]["values"] == g25
    assert any(x.startswith("C3 random partial-revert") for x in nc)
    assert set(A["analysed_elsewhere"]) == {"routing_entanglement", "gap_trajectory"} and "not_in_this_script" not in A
    md = open(os.path.join(root, "results", "analysis", "analysis_confirm.md"), encoding="utf-8").read()
    for h in ("## C3 endpoint revert decomposition", "## C3 per layer", "## C4 per-window", "## C5 French", "## Matched gain",
              "## F_rel versions", "## Bits per byte", "## Analysed elsewhere", "## Not computable from registered files"):
        assert h in md


def test_end_to_end_c2eq_alignment_and_tampered_french_forecasts(e2e_stress):
    root, T, A, scen, plan = e2e_stress
    ce = A["blocks"]["C2_eq_rand"]
    assert "status" not in ce and ce["step"] == 96 and len(ce["per_draw"]) == len(T["c2_rand"])
    for r in ce["per_draw"]:
        t = T["c2_rand"][f"{r['base']}|{r['d']}"]
        assert abs(r["b_Muon"] - t["b_muon"]) < 1e-9 and abs(r["b_Muon_rand"] - t["b_rand"]) < 1e-9
        assert abs(r["alignment"] - (t["b_muon"] - t["b_rand"])) < 1e-9
    c5 = A["blocks"]["C5_french"]
    assert c5["label"] == "non-prospective" and not c5["hash_protocol"]["verifies"]
    assert any("changed after hashing" in x for x in c5["hash_protocol"]["reasons"])



def test_bpb_bytes_from_the_data_bin_verify_every_window_hash(tmp_path, monkeypatch):
    """Bits per byte without n_bytes in the eval files: UTF-8 bytes of the target tokens arr[1:] from the registered bin
    (eval_windows --bytes rule), refused if a window's token hash differs from the evaluation file."""
    import analyze_confirm_ext as X
    import eval_windows as EW
    block = 4096
    toks = (np.arange(3 * block, dtype=np.int64) * 7919 % 500).astype(np.uint32)
    path = tmp_path / "x.bin"
    toks.tofile(path)
    bl = (np.arange(500) % 7 + 1).astype(np.int32)
    monkeypatch.setattr(EW, "token_byte_lengths", lambda n: bl)
    monkeypatch.setattr(X, "_token_reader", lambda *a: (lambda p: (np.memmap(p, dtype=np.uint32, mode="r"), 0)))
    starts = np.array([0, block, 2 * block], dtype=np.int64)
    hs = np.array([EW.window_hash(np.asarray(toks[s:s + block], dtype=np.int64)) for s in starts], dtype=np.uint64)
    res = {"meta": {"set": {"path": str(path), "block": block}}, "starts": starts, "hash": hs}
    nb, src = X._bytes_from_bin(None, res)
    assert list(nb) == [int(bl[np.asarray(toks[s + 1:s + block], dtype=np.int64)].sum()) for s in starts], src
    res["hash"] = hs.copy()
    res["hash"][1] += np.uint64(1)
    nb, src = X._bytes_from_bin(None, res)
    assert nb is None and "token hash differs" in src
    monkeypatch.setattr(EW, "token_byte_lengths", lambda n: None)
    assert X._bytes_from_bin(None, dict(res, hash=hs))[1] == "tiktoken unavailable"


def test_matched_gain_interpolation_in_log_gain():
    import analyze_confirm_ext as X
    pts = {8: {"G": 0.01, "F": {"c": 0.001}}, 24: {"G": 0.02, "F": {"c": 0.002}}, 96: {"G": 0.04, "F": {"c": 0.003}}}
    want = math.log(0.002) + (math.log(0.003) - math.log(0.002)) * (math.log(0.03) - math.log(0.02)) / (math.log(0.04) - math.log(0.02))
    assert abs(X._interp(pts, "c", 0.03) - want) < 1e-12
    assert abs(X._interp(pts, "c", 0.02) - math.log(0.002)) < 1e-15        # an evaluation point
    assert X._interp(pts, "c", 0.005) is None and X._interp(pts, "c", 0.05) is None   # not bracketed: no extrapolation
    assert X._interp({8: {"G": 0.01, "F": {"c": -1e-4}}, 96: {"G": 0.04, "F": {"c": 0.003}}}, "c", 0.02) is None  # F <= 0
