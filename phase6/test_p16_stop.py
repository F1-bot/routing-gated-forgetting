"""P-16 (2026-11-15): C2 runs stop after step 8 on the registered 96-step schedule (adapt_extras 'stop_after').

Tiny MoE on the CPU through the real pipeline child (test_phase3._run_child), warmup 10 as registered:
(A) the 96-step configuration, uninterrupted; (B) the same configuration with NESTOR_INTERIM_EVAL stop_after = 8;
(C) an 8-step configuration (what P-16 rules out: warmup and cosine follow max_steps).
B's steps 1-8 have A's learning rates, batch order and training losses; B's step-8 evaluation equals A's bitwise; B ends
after step 8 and its model_final.pt is the step-8 state; C's learning rates differ."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import test_phase3 as T3  # noqa: I001  (first: CPU env flags, sys.path, tiny builders, pipeline child)
import eval_windows as EW
import gate_common as GC
import torch
from test_phase3 import pretrained  # noqa: F401  (session fixture: tiny pretrained checkpoints)

HERE = os.path.dirname(os.path.abspath(T3.__file__))
WARMUP_96 = 10


def _run(pre, tmp, tag, steps, warmup, stop=None):
    root, idir, ldir = str(tmp / tag / "adapt"), str(tmp / tag / "interim"), str(tmp / tag / "trainlog")
    interim = {"steps": [8], "out": idir, "block": 64, "routing": False,
               "sets": [{"name": "old", "path": pre["data"], "tokens": 640}]}
    if stop:
        interim["stop_after"] = stop
    log, _ = T3._run_child({"family": "moe"}, data=pre["dom"], steps=steps, root=root, suffix="adapt", init=pre["init"],
                           spec=T3.T.MOE_SPEC,
                           extra={"NESTOR_INTERIM_EVAL": json.dumps(interim), "NESTOR_TRAIN_LOG_DIR": ldir,
                                  "NESTOR_TRAIN_BATCH_SHA": "1", "DCLM_WARMUP_STEPS": str(warmup)})
    return {"log": log, "recs": GC.read_train_steps(os.path.join(ldir, "train_steps.jsonl")), "root": root, "idir": idir,
            "ldir": ldir, "pre": pre}


@pytest.fixture(scope="module")
def runs(pretrained, tmp_path_factory):
    pre = pretrained("moe")
    tmp = tmp_path_factory.mktemp("p16")
    return {"A": _run(pre, tmp, "A", 96, WARMUP_96), "B": _run(pre, tmp, "B", 96, WARMUP_96, stop=8),
            "C": _run(pre, tmp, "C", 8, max(1, round(WARMUP_96 * 8 / 96)))}


def test_stopped_run_keeps_the_96_step_learning_rates_and_batch_order(runs):
    A, B = runs["A"], runs["B"]
    assert [r["step"] for r in A["recs"]] == list(range(1, 97))
    assert [r["step"] for r in B["recs"]] == list(range(1, 9))
    for a, b in zip(A["recs"][:8], B["recs"]):
        assert a["lrs"] == b["lrs"] and a["batch_sha"] == b["batch_sha"] and a["micro_batches"] == b["micro_batches"]
        assert a["loss"] == b["loss"] and a["ce"] == b["ce"]   # same state and batches: the same loss, bitwise (CPU)
    assert len({r["batch_sha"] for r in A["recs"][:8]}) > 1      # the digest distinguishes the steps' batches
    stop = [json.loads(x) for x in open(os.path.join(B["ldir"], "stop.jsonl"), encoding="utf-8")]
    assert len(stop) == 1 and stop[0]["stop_after"] == 8 and stop[0]["stopped_at_step"] == 8
    assert stop[0]["registered_max_steps"] == 96
    assert not os.path.exists(os.path.join(A["ldir"], "stop.jsonl"))


def test_stopped_run_ends_with_the_step_8_state(runs, tmp_path):
    A, B = runs["A"], runs["B"]
    a8 = EW.load_result(os.path.join(A["idir"], "step008__old.npz"))
    b8 = EW.load_result(os.path.join(B["idir"], "step008__old.npz"))
    assert np.array_equal(a8["hash"], b8["hash"]) and np.array_equal(a8["ce_sum"], b8["ce_sum"])
    fin = T3.T._final_ckpt(B["root"], "adapt", "moe")
    ck = torch.load(fin, map_location="cpu", weights_only=False)
    assert int(ck["step"]) == 8
    out = str(tmp_path / "final_eval")
    argv = [sys.executable, os.path.join(HERE, "eval_windows.py"), "run", "tiny_moe", fin, "--set",
            f"old={B['pre']['data']}:640", "--block", "64", "--out", out, "--device", T3.DEV,
            "--register", os.path.join(HERE, "tiny_experiments.py")]
    p = subprocess.run(argv, capture_output=True, text=True, env=T3._child_env(T3.TINY_ENV), cwd=HERE, timeout=1800)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    f = EW.load_result(os.path.join(out, "ckpt__old.npz"))
    assert np.array_equal(f["hash"], b8["hash"])
    assert np.allclose(f["ce_sum"], b8["ce_sum"], rtol=T3.XDEV_RTOL, atol=1e-6)   # final checkpoint = step-8 state


def test_an_8_step_configuration_would_change_the_schedule(runs):
    A, C = runs["A"], runs["C"]
    assert [r["step"] for r in C["recs"]] == list(range(1, 9))
    assert any(a["lrs"] != c["lrs"] for a, c in zip(A["recs"][:8], C["recs"]))
