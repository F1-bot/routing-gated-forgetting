"""Tests of r8_eval_equivalence.py (comparison rule and child environment; no model, no torch)."""
from __future__ import annotations

import os

import numpy as np
import pytest

import r8_eval_equivalence as R8


def _res(ce, n=4096, shift=0.0, hash_off=0):
    k = len(ce)
    return {"index": np.arange(k), "starts": np.arange(k) * 4096, "hash": np.arange(k, dtype=np.uint64) + hash_off,
            "n_tok": np.full(k, n), "ce_sum": (np.asarray(ce, dtype=np.float64) + shift) * n}


def test_identical_results_pass_bitwise():
    a = _res([4.6, 4.7, 4.5])
    c = R8.compare(a, _res([4.6, 4.7, 4.5]))
    assert c["ok"] and c["bitwise_equal_windows"] == 3 and c["max_abs_window_diff"] == 0.0


@pytest.mark.parametrize("shift, ok", [(0.9e-6, True), (2e-6, False)])
def test_tolerance_is_1e6_nats_per_token(shift, ok):
    c = R8.compare(_res([4.6, 4.7]), _res([4.6, 4.7], shift=shift))
    assert c["ok"] is ok and c["tol"] == 1e-6


def test_window_grid_mismatch_fails_before_any_ce_comparison():
    c = R8.compare(_res([4.6, 4.7]), _res([4.6, 4.7], hash_off=1))
    assert not c["ok"] and c["reason"] == "window grid differs" and c["grid_equal"]["hash"] is False


def test_child_env_orders_the_frozen_dir_first_and_strips_the_libbin_entry(monkeypatch, tmp_path):
    lib = str(tmp_path / "Library" / "bin")
    monkeypatch.setenv("PATH", os.pathsep.join([lib, r"C:\\Windows"]))
    monkeypatch.setenv("PYTHONPATH", "X")
    env = R8.child_env([str(tmp_path / "frozen_g0"), str(tmp_path)], [lib], str(tmp_path))
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "frozen_g0"), str(tmp_path), "X"]
    assert lib not in env["PATH"].split(os.pathsep) and env["PYTHONDONTWRITEBYTECODE"] == "1"
