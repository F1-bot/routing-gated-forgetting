"""Unit tests of c1de_milp.py (exact MILP pair generator, exploratory German C1 block C1-DE; prereg v1.3 sec. 4.12).
CPU, torch-free. Synthetic registered tables (crossover.RegTables); the frozen crossover.py is the reference for pools,
tilt, matching, match mask and predicted contrasts. REF_VECTORS are the draws of the reference
implementation (c1de_milp_dryrun.py, run on the same synthetic table with seeds 5600 + d)."""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c1de_milp as C1  # noqa: E402
import crossover as CX  # noqa: E402

CODE = "code"
CORPORA = ["german", "dclm"]
CONFIRM_FIELDS = ("status", "d", "seed", "predicted", "spec_A", "spec_B", "A", "B", "subsets", "matched_values",
                  "design", "f", "k", "tol", "matched_corpora", "target", "code", "generator")
REF_VECTORS = json.loads('{"1":{"A":[[1,4,11],[2,4,10],[0,7,10],[3,5,6]],"B":[[0,3,8],[0,3,11],[1,4,8],[7,10,11]]},"2":{"A":[[4,9,11],[6,8,10],[0,2,10],[0,3,6]],"B":[[0,2,3],[1,3,9],[4,8,11],[1,2,11]]},"3":{"A":[[6,7,9],[4,6,7],[2,6,7],[0,3,9]],"B":[[0,2,8],[1,9,11],[1,5,8],[1,7,11]]}}')


def _tables(L=4, N=12, seed=0, corpora=("dclm", "code", "math", "books", "german")):
    """Synthetic RegTables with router-popularity correlation (code vs math anti-correlated)."""
    rng = np.random.default_rng(seed)
    pop = rng.gamma(2.0, 1.0, (L, N))
    sel, est = {}, {}
    for c in corpora:
        tilt = np.exp(0.8 * rng.standard_normal((L, N)) * (1 if c in ("code", "books") else -0.5))
        base = pop * tilt
        sel[c] = base * np.exp(0.05 * rng.standard_normal((L, N)))
        est[c] = base * np.exp(0.05 * rng.standard_normal((L, N)))
    keys = [[f"moe:b{l}:l0:e{m}" for m in range(N)] for l in range(L)]
    return CX.RegTables("moe", [f"moe:b{l}" for l in range(L)], keys, sel, est, {})


def _brute(t, k, f, tau, w):
    """All pool-subset pairs with exactly k per stratum: the number matched within tau and the argmax of w.x."""
    tilt = CX.reg_tilt(t.sel, "c1", code=CODE)
    lo, hi = C1.pools(tilt, f)
    L, npool = lo.shape
    per_side = list(itertools.combinations(range(npool), k))
    best, n_ok = None, 0
    for pa in itertools.product(per_side, repeat=L):
        A = np.stack([lo[l][list(pa[l])] for l in range(L)])
        for pb in itertools.product(per_side, repeat=L):
            B = np.stack([hi[l][list(pb[l])] for l in range(L)])
            err = CX.reg_match_errors(t.sel, A[None], B[None], CORPORA)[:, 0]
            if np.all(np.abs(err) <= tau):
                n_ok += 1
                x = np.zeros(2 * L * npool)
                for l in range(L):
                    x[l * npool + np.array(pa[l])] = 1
                    x[L * npool + l * npool + np.array(pb[l])] = 1
                v = float(w @ x)
                if best is None or v > best[0]:
                    best = (v, np.sort(A, 1), np.sort(B, 1))
    return n_ok, best


def _sequence(t, n=3):
    out, earlier = [], []
    for d in range(1, n + 1):
        e = C1.draw(t, d, code=CODE, earlier=earlier)
        if e["status"] == "ok":
            earlier.append((np.array(e["A"]), np.array(e["B"])))
        out.append(e)
    return out


def test_draws_equal_the_reference_implementation():
    es = _sequence(_tables(), 3)
    for e in es:
        assert e["status"] == "ok" and e["seed"] == 5600 + e["d"]
        assert e["A"] == REF_VECTORS[str(e["d"])]["A"] and e["B"] == REF_VECTORS[str(e["d"])]["B"]


def test_draw_meets_the_registered_constraints_and_the_books_record_format():
    t = _tables()
    k = CX.reg_k(t)
    e = C1.draw(t, 1, code=CODE)
    assert e["status"] == "ok" and e["block"] == "C1-DE" and e["target"] == "german"
    assert e["matched_corpora"] == CX.reg_matched("c1", target="german") == CORPORA
    A, B = np.array(e["A"]), np.array(e["B"])
    lo, hi = C1.pools(CX.reg_tilt(t.sel, "c1", code=CODE), C1.F)
    for l in range(t.L):
        assert len(A[l]) == len(set(A[l])) == k and len(B[l]) == len(set(B[l])) == k   # exactly k per stratum
        assert set(A[l]) <= set(lo[l]) and set(B[l]) <= set(hi[l]) and not set(A[l]) & set(B[l])
    assert bool(CX.reg_match_mask(t.sel, A[None], B[None], CORPORA, CX.REG_TOL)[0])  # frozen mask at REG_TOL
    err = CX.reg_match_errors(t.sel, A[None], B[None], CORPORA)[:, 0]
    assert [e["matched_values"][c]["diff"] for c in CORPORA] == err.tolist()
    assert e["selection"]["max_abs_residual"] <= CX.REG_TOL - C1.MARGIN + 1e-9
    assert e["selection"]["P"] == float(CX.reg_predicted(t.sel, A[None], B[None], "c1", code=CODE)[0])
    assert e["predicted"]["P"] == float(CX.reg_predicted(t.est, A[None], B[None], "c1", code=CODE)[0])
    assert all(f in e for f in CONFIRM_FIELDS) and set(e["predicted"]) == {"P"}      # fields block_c1 reads
    assert e["spec_A"] == CX.spec_for(t.subset_keys(A)) and e["spec_B"] == CX.spec_for(t.subset_keys(B))
    a0 = e["generator"]["attempts"][0]
    assert a0["milp_status"] == 0 and a0["tau"] == CX.REG_TOL - 1e-6 and e["generator"]["time_limit_s"] == 600.0


@pytest.mark.parametrize("tseed", [1, 2, 3])
def test_milp_optimum_equals_brute_force(tseed):
    t = _tables(L=2, N=8, seed=tseed)
    k = CX.reg_k(t)
    assert k == 2
    tau = CX.REG_TOL - C1.MARGIN
    w = C1.objective(5601, 2 * t.L * 4)
    n_ok, best = _brute(t, k, C1.F, tau, w)
    r = C1.solve(t.sel, CX.reg_tilt(t.sel, "c1", code=CODE), k, C1.F, CORPORA, tau, w)
    if n_ok == 0:
        assert r["status"] == "failed" and r["milp_status"] == 2
    else:
        assert r["status"] == "optimal" and r["objective"] == pytest.approx(best[0], abs=1e-6)
        assert np.array_equal(r["A"], best[1]) and np.array_equal(r["B"], best[2])


def test_time_limit_hit_is_recorded_and_the_draw_is_not_run(monkeypatch):
    """R6: a solve stopped by the 600 s limit (HiGHS status 1) is flagged on the attempt and the draw; a normal
    solve is not."""
    import scipy.optimize as so

    t = _tables(L=2, N=8, seed=0)
    k, tau, w = CX.reg_k(t), CX.REG_TOL - C1.MARGIN, C1.objective(5601, 2 * t.L * 4)
    r = C1.solve(t.sel, CX.reg_tilt(t.sel, "c1", code=CODE), k, C1.F, CORPORA, tau, w)
    assert r["time_limit_hit"] is False

    class _Res:
        status, message, x, fun = 1, "Time limit reached. (HiGHS Status 13: model_status is Time limit reached)", None, None

    monkeypatch.setattr(so, "milp", lambda *a, **kw: _Res())
    r = C1.solve(t.sel, CX.reg_tilt(t.sel, "c1", code=CODE), k, C1.F, CORPORA, tau, w)
    assert r["status"] == "failed" and r["time_limit_hit"] is True
    e = C1.draw(_tables(), 1, code=CODE)
    assert e["status"] == "milp_failed" and e["generator"]["time_limit_hit"] is True
    assert e["generator"]["attempts"][0]["time_limit_hit"] is True


def test_no_matched_pair_is_recorded_and_not_run():
    t = _tables(L=2, N=8, seed=1)
    assert _brute(t, 2, C1.F, 1e-6, C1.objective(5601, 16))[0] == 0
    e = C1.draw(t, 1, code=CODE, tol=2e-6)                   # MILP tolerance 1e-6: no pair exists
    assert e["status"] == "milp_failed" and "A" not in e and e["generator"]["attempts"][0]["milp_status"] == 2


def test_duplicate_rule_redraws_the_weights_from_the_same_generator():
    t = _tables()
    e1 = C1.draw(t, 1, code=CODE)
    dup = C1.draw(t, 1, code=CODE, earlier=[(np.array(e1["A"]), np.array(e1["B"]))])
    att = dup["generator"]["attempts"]
    assert dup["status"] == "ok" and (dup["A"], dup["B"]) != (e1["A"], e1["B"])
    assert att[0].get("duplicate") and att[0]["weights"] == 0 and att[-1]["weights"] == 1
    assert dup["seed"] == e1["seed"]                         # same objective seed, second weight vector
    every = C1.draw(t, 1, code=CODE, earlier=[(np.array(e1["A"]), np.array(e1["B"]))], max_weights=1)
    assert every["status"] == "duplicates"


def test_mask_rejection_lowers_tau_by_1e4_at_most_three_times(monkeypatch):
    t = _tables()
    calls = []
    real = C1.CX.reg_match_mask

    def reject_first(sel, A, B, corpora, tol=CX.REG_TOL):
        calls.append(1)
        return np.zeros(1, bool) if len(calls) == 1 else real(sel, A, B, corpora, tol)

    monkeypatch.setattr(C1.CX, "reg_match_mask", reject_first)
    e = C1.draw(t, 1, code=CODE)
    taus = [a["tau"] for a in e["generator"]["attempts"]]
    assert e["status"] == "ok" and taus == pytest.approx([CX.REG_TOL - 1e-6, CX.REG_TOL - 1e-6 - 1e-4], abs=1e-15)
    monkeypatch.setattr(C1.CX, "reg_match_mask", lambda *a, **k: np.zeros(1, bool))
    e2 = C1.draw(t, 1, code=CODE)
    assert e2["status"] == "mask_failed" and len(e2["generator"]["attempts"]) == 4


def test_draws_are_deterministic_and_differ():
    t = _tables()
    es = _sequence(t, 3)
    again = _sequence(t, 3)
    assert [e["seed"] for e in es] == [5601, 5602, 5603]
    assert [(e["A"], e["B"]) for e in again] == [(e["A"], e["B"]) for e in es]
    assert len({json.dumps([e["A"], e["B"]]) for e in es}) == 3      # the duplicate rule keeps them distinct


def test_generation_uses_selection_statistics_only():
    t = _tables()
    e = C1.draw(t, 1, code=CODE)
    t2 = copy.deepcopy(t)
    rng = np.random.default_rng(7)
    t2.est = {c: v * np.exp(rng.normal(0.0, 0.5, v.shape)) for c, v in t.est.items()}
    e2 = C1.draw(t2, 1, code=CODE)
    assert e2["A"] == e["A"] and e2["B"] == e["B"] and e2["selection"]["P"] == e["selection"]["P"]
    assert e2["predicted"]["P"] != e["predicted"]["P"]              # only the reported estimation-half P moves


def test_overlaps_and_manifest_round_trip_in_the_confirm_plan_format(tmp_path):
    t = _tables()
    ents = _sequence(t, 3)
    ov = C1.overlaps(ents, CX.reg_k(t), t.L)
    assert [o["d"] for o in ov] == [[1, 2], [1, 3], [2, 3]] and all(0.0 <= o["A"] <= 1.0 and 0.0 <= o["B"] <= 1.0 for o in ov)
    p = str(tmp_path / "c1de_moe_b1_german.json")
    h = CX.write_manifest(p, ents, extra={"prereg": "test", "block": "C1-DE"})
    doc = json.load(open(p, encoding="utf-8"))
    assert doc["prereg"] == "test" and doc["block"] == "C1-DE" and len(doc["entries"]) == 3
    assert [e["A"] for e in doc["entries"]] == [e["A"] for e in ents]
    assert open(p + ".sha256", encoding="utf-8").read().split()[0] == h


def test_cli_missing_g0_record_exits_2(tmp_path):
    assert C1.main(["--g0-record", str(tmp_path / "none.json"), "--out-dir", str(tmp_path / "out")]) == 2
