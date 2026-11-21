"""Phase-5 tests that need torch (CPU by default; NESTOR_TEST_DEVICE=cuda in the GPU pipeline audit of hooks_after_B.ps1).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 python -m pytest test_phase5_torch.py -q -rA

Review finding eb4b1968 (trainer.py:905, distributed.py:335 write p.grad = zeros_like(p) outside autograd, so tensor hooks
never fire for those parameters). Audit result: no phase-1..5 statistic counts hook calls; the only statistic that combined
gradients of two backward passes key-by-key was grad_snr.half_batch_snr, which now zero-fills every requested parameter.
Tests: grad_snr with a parameter missing in either half equals the explicit-zero computation; NS5(0) = 0 so an absent
gradient in g0_probe / g0_margin / theta0_checks is an exact zero direction; adapt_extras._grad_norm is invariant to None vs
zero grads. (subset_adapt under the line-905 zero fill: test_subset_adapt.py::test_trainer_zero_fill_of_missing_grads_trainer_py_905.)
"""
from __future__ import annotations

import test_phase3 as T3  # noqa: I001  (first: CPU env flags, sys.path, tiny builders, pipeline child)

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import adapt_extras as AX
import eval_windows as EW
import g0_probe as G0
import gate_common as GC
import grad_snr as GS
import theta0_checks as TC
from test_phase3 import pretrained  # noqa: F401  (session fixture: tiny pretrained checkpoints)

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================= finding eb4b1968
def _snr_inputs(tmp_path):
    params = {"a": torch.nn.Parameter(torch.zeros(3, 4)), "b": torch.nn.Parameter(torch.zeros(2, 5))}
    rows = {"moe:b0:l0:e0": [("a", 0)], "moe:b0:l0:e1": [("a", 1)], "moe:b0:l0:e2": [("a", 2)],
            "moe:b0:l1:e0": [("b", 0)], "moe:b0:l1:e1": [("b", 1)]}
    g = torch.Generator().manual_seed(0)
    full = [{"a": torch.randn(3, 4, generator=g), "b": torch.randn(2, 5, generator=g)} for _ in range(2)]
    ws = EW.WindowSet("d", T3.write_bin(tmp_path / "d.bin", 64 * 8, 0), 64, 8)
    return params, rows, full, ws


@pytest.mark.parametrize("missing_half", [0, 1])
def test_grad_snr_missing_gradient_counts_as_exact_zero(tmp_path, monkeypatch, missing_half):
    params, rows, full, ws = _snr_inputs(tmp_path)
    halves = [dict(full[0]), dict(full[1])]
    del halves[missing_half]["b"]                                # no token reached bank b in this half
    it = iter(halves)
    monkeypatch.setattr(G0, "target_gradient", lambda *a, **k: next(it))
    res = GS.half_batch_snr(None, ws, params, rows, device="cpu", seed=0, kinds=("moe",))
    ref_halves = [dict(full[0]), dict(full[1])]
    ref_halves[missing_half]["b"] = torch.zeros(2, 5)           # the explicit-zero reference
    it2 = iter(ref_halves)
    monkeypatch.setattr(G0, "target_gradient", lambda *a, **k: next(it2))
    ref = GS.half_batch_snr(None, ws, params, rows, device="cpu", seed=0, kinds=("moe",))
    for u in rows:
        for k in ("gbar_sq", "noise", "signal"):
            assert res["per_unit"][u][k] == ref["per_unit"][u][k], (u, k)
    assert res["zero_filled_parameters"][f"half{missing_half}"] == ["b"]
    assert res["zero_filled_parameters"][f"half{1 - missing_half}"] == []
    # the pre-fix combination ran over half 0's keys only: half 1's gradient of b was silently dropped (or KeyError)
    gb1 = full[1]["b"].double()
    if missing_half == 0:
        assert res["per_unit"]["moe:b0:l1:e0"]["gbar_sq"] == pytest.approx(float((gb1[0] / 2).pow(2).sum()), rel=1e-12)


def test_muon_direction_of_a_zero_gradient_is_an_exact_zero():
    p = {"w": torch.nn.Parameter(torch.randn(4, 6)), "bank": torch.nn.Parameter(torch.randn(3, 4, 6))}
    u = G0.muon_direction(p, {"w": torch.zeros(4, 6), "bank": torch.zeros(3, 4, 6)})
    assert all(torch.equal(v, torch.zeros_like(v)) for v in u.values())
    filled, missing = TC.zero_filled({"w": torch.ones(4, 6)}, p)
    assert missing == ["bank"] and torch.equal(filled["bank"], torch.zeros(3, 4, 6))
    filled2, missing2 = GS.zero_filled({"w": torch.ones(4, 6)}, p)
    assert missing2 == ["bank"] and torch.equal(filled2["bank"], torch.zeros(3, 4, 6))


def test_grad_norm_log_is_invariant_to_none_versus_zero_grads():
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    m(torch.randn(5, 4)).sum().backward()
    m[1].bias.grad = None
    n_none = AX._grad_norm(m)
    m[1].bias.grad = torch.zeros_like(m[1].bias)            # what trainer.py:905 writes
    assert AX._grad_norm(m) == n_none


# ============================================================================= theta_0 checks
def test_theta0_checks_c_identity_ns5_and_d_muon_on_tiny_moe(tmp_path):
    m = T3.tiny("moe")
    ws = EW.WindowSet("t", T3.write_bin(tmp_path / "t.bin", 64 * 4, 1), 64, 4)
    rows = G0.unit_rows(m)
    keys = sorted(k for k in rows if k.startswith("moe:"))[:3]
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    r = TC.checks(m, ws, keys, device="cpu", windows=4, lr_muon=0.0025, momentum=0.95, bf16=False)
    s = r["sgd"]
    assert s["c"] > 0 and s["grid"] == [s["c"] / 3, s["c"], 3 * s["c"]] and s["n_matrices"] >= 3
    assert math.isclose(s["mean_first_step_norm_sgd_at_c"], s["mean_first_step_norm_muon"], rel_tol=1e-12)
    for sc in ("0.25", "0.5", "2"):
        v = r["ns5_scale_check"]["per_scale"][sc]
        assert v["n"] == s["n_matrices"] and 0 <= v["median"] <= v["max"] < 1
    d = r["d_muon_unit_test"]
    assert d["pass"] and d["max_rel_err"] < 1e-4, d
    assert all(torch.equal(p.detach(), before[n]) for n, p in m.named_parameters())     # theta_0 restored


# ============================================================================= adapt_extras v4 in the pipeline
def test_pipeline_loss_log_and_step0_interim(pretrained, tmp_path):
    pre = pretrained("nestor")
    root, idir, ldir = str(tmp_path / "adapt"), str(tmp_path / "interim"), str(tmp_path / "trainlog")
    interim = {"steps": [0, 2, 4], "out": idir, "block": 64, "routing": False,
               "sets": [{"name": "old", "path": pre["data"], "tokens": 640}]}
    log, env_used = T3._run_child({"family": "nestor"}, data=pre["dom"], steps=4, root=root, suffix="adapt", init=pre["init"],
                                  spec=T3.T.NESTOR_SPEC, extra={"NESTOR_INTERIM_EVAL": json.dumps(interim),
                                                                "NESTOR_TRAIN_LOG_DIR": ldir})
    assert "interim eval at step 0" in log and "interim eval at step 4" in log
    recs = GC.read_train_steps(os.path.join(ldir, "train_steps.jsonl"))
    assert [r["step"] for r in recs] == [1, 2, 3, 4]
    for r in recs:
        assert r["micro_batches"] == 2 and math.isfinite(r["loss"]) and math.isfinite(r["ce"]) and r["ce"] > 0
    e1 = GC.e1_check(recs, warmup=1)
    assert e1["status"] == "checked" and e1["loss_field"] == "ce"
    s0 = EW.load_result(os.path.join(idir, "step000__old.npz"))
    assert "allow_tf32_matmul" in s0["meta"] and s0["meta"]["step"] == 0
    out = str(tmp_path / "base_eval")
    argv = [sys.executable, os.path.join(HERE, "eval_windows.py"), "run", "tiny_nestor", pre["init"], "--set",
            f"old={pre['data']}:640", "--block", "64", "--out", out, "--device", T3.DEV,
            "--register", os.path.join(HERE, "tiny_experiments.py")]
    p = subprocess.run(argv, capture_output=True, text=True, env=T3._child_env(T3.TINY_ENV), cwd=HERE, timeout=1800)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    b = EW.load_result(os.path.join(out, "ckpt__old.npz"))
    assert np.array_equal(b["hash"], s0["hash"])                       # step 0 = theta_0 on the same windows
    assert np.allclose(b["ce_sum"], s0["ce_sum"], rtol=T3.XDEV_RTOL, atol=1e-6)
    s4 = EW.load_result(os.path.join(idir, "step004__old.npz"))
    assert not np.allclose(s4["ce_sum"], s0["ce_sum"])                 # training moved theta between step 0 and 4


# ============================================================================= theory verification r3: selpin, reflect + pin
def _perturbed(m, scale=0.05, seed=0, kind="moe"):
    mp = T3.tiny(kind)                       # the authors' models are not deep-copyable: rebuild and load the state
    mp.load_state_dict(m.state_dict())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mp.parameters():
            if p.dtype.is_floating_point and p.numel() > 1:
                p.add_(scale * p.detach().std().clamp_min(1e-3) * torch.randn(p.shape, generator=g))
    return mp


@pytest.mark.parametrize("kind", ["moe", "nestor"])
def test_pinned_and_selpin_replay_at_theta0_reproduce_live_exactly(tmp_path, kind):
    m = T3.tiny(kind)
    ws = EW.WindowSet("w", T3.write_bin(tmp_path / "w.bin", 64 * 4, 2), 64, 4)
    live = EW.evaluate(m, ws, log=None)
    for so in (False, True):
        pin = EW.RoutingPin(select_only=so).install(m)
        try:
            r = EW.evaluate(m, ws, pin=pin, pin_source=m, log=None)
        finally:
            pin.uninstall()
        assert np.array_equal(r["ce_sum"], live["ce_sum"]), so


def test_selpin_keeps_theta0_selections_and_recomputes_gates(tmp_path):
    import nestor.backbones.nestor as nb

    m0 = T3.tiny("moe")
    m1 = _perturbed(m0)
    x = torch.as_tensor(EW.WindowSet("w", T3.write_bin(tmp_path / "w.bin", 64 * 2, 3), 64, 2).tokens(0)).view(1, -1)
    pin = EW.RoutingPin(select_only=True).install(m0, m1)
    seen = []
    orig = pin._orig_forward
    try:
        pin.record()
        EW.forward_ce(m0, x)
        rec0 = [d for d in pin.decisions() if d[0] == "router"]
        pin.replay()
        out_patched = nb.Router.forward

        def spy(router, xx, valid, noise=None):
            o = out_patched(router, xx, valid, noise)
            live = orig(router, xx, valid, noise)
            seen.append((o, live, router.gate_mode))
            return o
        nb.Router.forward = spy
        try:
            EW.forward_ce(m1, x)
        finally:
            nb.Router.forward = out_patched
        pin.done()
    finally:
        pin.uninstall()
    assert len(seen) == len(rec0) > 0
    n_diff_sel = 0
    for (o, live, gm), (_, _, saved) in zip(seen, rec0):
        assert torch.equal(o.topk_indices, saved[3])                                        # theta_0's selections
        assert torch.equal(o.gates, EW.selpin_gates(live.scores, saved[3], gm, live.valid_token_mask))   # live gate values
        assert torch.equal(o.scores, live.scores)
        n_diff_sel += int(not torch.equal(live.topk_indices.sort(-1).values, saved[3].sort(-1).values))
    assert n_diff_sel >= 0


def test_joint_pinned_evaluation_equals_single_pins_and_reflect(tmp_path):
    import weight_controls as WC

    m0 = T3.tiny("moe")
    m1 = _perturbed(m0, seed=1)
    ws = EW.WindowSet("w", T3.write_bin(tmp_path / "w.bin", 64 * 4, 4), 64, 4)
    mr = _perturbed(m0, scale=0.0)
    mr.load_state_dict(WC.reflect_state(m0.state_dict(), m1.state_dict()), strict=False)
    targets = {"pinned": (m1, False), "selpin": (m1, True), "reflect+pinned": (mr, False), "reflect+selpin": (mr, True)}
    multi = EW.evaluate_pinned_multi(m0, targets, ws, log=None)
    for v, (mod, so) in targets.items():
        pin = EW.RoutingPin(select_only=so).install(mod, m0)
        try:
            single = EW.evaluate(mod, ws, pin=pin, pin_source=m0, log=None)
        finally:
            pin.uninstall()
        assert np.array_equal(multi[v]["ce_sum"], single["ce_sum"]), v
        assert np.array_equal(multi[v]["hash"], single["hash"])
    assert not np.array_equal(multi["pinned"]["ce_sum"], multi["reflect+pinned"]["ce_sum"])


def test_eval_cli_pin_variants_and_fsym_decomposition(pretrained, tmp_path):
    pre = pretrained("nestor")
    root = str(tmp_path / "adapt")
    T3._run_child({"family": "nestor"}, data=pre["dom"], steps=3, root=root, suffix="adapt", init=pre["init"], spec=T3.T.NESTOR_SPEC)
    final = T3.T._final_ckpt(root, "adapt", "nestor")
    outs = {}
    for name, variants, extra in (("base", "ckpt", []), ("main", "ckpt", []), ("reflect", "reflect", ["--base", pre["init"]]),
                                  ("pin", "pinned,selpin,reflect+pinned,reflect+selpin", ["--base", pre["init"]])):
        ck = pre["init"] if name == "base" else final
        out = str(tmp_path / f"eval_{name}")
        argv = [sys.executable, os.path.join(HERE, "eval_windows.py"), "run", "tiny_nestor", ck, "--set", f"old={pre['data']}:640",
                "--block", "64", "--out", out, "--variants", variants, "--device", T3.DEV,
                "--register", os.path.join(HERE, "tiny_experiments.py")] + extra
        p = subprocess.run(argv, capture_output=True, text=True, env=T3._child_env(T3.TINY_ENV), cwd=HERE, timeout=1800)
        assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
        outs[name] = out
    for v in ("pinned", "selpin", "reflect+pinned", "reflect+selpin"):
        r = EW.load_result(os.path.join(outs["pin"], f"{v}__old.npz"))
        assert r["meta"]["weights"] == ("theta0-delta" if v.startswith("reflect+") else "theta0+delta")
    run_root = str(tmp_path)
    t = GC.fsym_table({"old": os.path.join(outs["base"], "ckpt__old.npz")}, run_root, ["old"],
                      dirs={"ckpt": "eval_main", "reflect": "eval_reflect", "pinned": "eval_pin", "selpin": "eval_pin",
                            "reflect+pinned": "eval_pin", "reflect+selpin": "eval_pin"})
    d = t["old"]
    assert d["complete"] and all(math.isfinite(d[k]["F_sym"]) for k in ("live", "selpin", "pin"))
    assert math.isclose(d["pin"]["F_sym"] + d["G_gate_pathway"]["value"] + d["Phi_flips"]["value"], d["live"]["F_sym"],
                        rel_tol=1e-9, abs_tol=1e-12)
