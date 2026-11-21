"""CPU tests for the phase-3 tooling on tiny models built with the authors' code.

Modules: eval_windows (per-window CE, routing summaries, pinned routing, flips), weight_controls
(random direction, revert, WD-only, displacement), crossover (cross-fitted RO, constrained pairs),
adapt_extras (SGD arm, interim evals, train log, clip override), g0_probe (first-step direction,
directional probes, end-to-end G0 run), subset_adapt hardening (resume guard, unchanged-trainable and
optimizer-state audit, negative control), run_queue (gates, skip, resume, STOP, OOM fallback) and
prep_streams (document-level disjoint sampling, tokenization).

Run (CPU only, <= 6 threads):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 python -m pytest test_phase3.py -q -rA
Child-process tests run the authors' TrainPipeline for tiny registered experiments (tiny_experiments.py)
with subset_adapt.install() + adapt_extras.install(), as run_exp.py does after the patch note.
"""
from __future__ import annotations

import test_subset_adapt as T  # noqa: I001  (first: CPU env flags, sys.path, threads, tiny builders)

import dataclasses
import glob
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

import adapt_extras as AX
import crossover as CX
import eval_windows as EW
import g0_probe as G0
import prep_streams as PS
import run_queue as RQ
import subset_adapt as SA
import tiny_experiments  # noqa: F401  (registers tiny_nestor / tiny_moe in this process)
import weight_controls as WC
import nestor.backbones.nestor as NB
from nestor.configs.optim import MuonAdamWConfig
from nestor.configs.sched import CosineAnnealingConfig
from nestor.scaling.blocks import get_scaling
from nestor.train.frozen_routers import freeze_routers
from nestor.train.optim import build_optimizer, build_scheduler

HERE = T.HERE
P = T.P
EOT = 255
NESTOR_K, NESTOR_KQ, PUB_K, MOE_K = (2, 3, 4), (2, 1, 1), 2, 2       # tiny geometries (test_subset_adapt)
CLEAN_ENV = (SA.SPEC_ENV, SA.VERIFY_ENV, SA.NEGCTRL_ENV, SA.ALLOW_CHANGE_ENV, SA.MANIFEST_ENV, "NESTOR_OUTPUT_ROOT",
             "NESTOR_SUBSET_OPTIMIZER", "NESTOR_SGD_LR", "NESTOR_SGD_WD", "NESTOR_SGD_MOMENTUM", "NESTOR_INTERIM_EVAL",
             "NESTOR_TRAIN_LOG_DIR", "NESTOR_MAX_GRAD_NORM", "DCLM_INIT_CHECKPOINT")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    SA.reset()
    SA.register_optimizer_hooks()
    for k in CLEAN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NESTOR_FREEZE_ROUTERS", "1")
    yield
    SA.reset()


def write_bin(path, n_tokens: int, seed: int, lo: int = 0, hi: int = EOT, eot: int = EOT, doc_len: int | None = None) -> str:
    a = np.random.default_rng(seed).integers(lo, hi, n_tokens, dtype=np.uint32)
    if doc_len:
        a[doc_len - 1::doc_len] = eot
    a.tofile(str(path))
    with open(f"{path}.meta.json", "w") as fh:
        json.dump({"token_dtype": "uint32", "num_tokens": n_tokens, "eot_token": eot}, fh)
    return str(path)


def tiny(kind: str, seed: int = 0):
    m = T.build(T.nestor_backbone() if kind == "nestor" else T.moe_backbone(), seed=seed)
    freeze_routers(m)
    m.eval()
    return m


def perturb_(model, pred, scale: float = 0.05, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if pred(n):
                p.add_(scale * torch.randn(p.shape, generator=g))


def last_layer_prefix(model) -> str:
    blocks = model.backbone.blocks
    return f"backbone.blocks.{len(blocks) - 1}.layers.{len(blocks[-1].layers) - 1}."


# ============================================================================ eval_windows
def test_window_selection_halves_heads_and_hashes(tmp_path):
    assert EW.select_windows(64 * 10, 64, 4, "first").tolist() == [0, 1, 2, 3]
    ev = EW.select_windows(64 * 10, 64, 4, "even")
    assert ev[0] == 0 and ev[-1] == 9 and len(ev) == 4 and (np.diff(ev) > 0).all()
    r1, r2 = EW.select_windows(6400, 64, 7, "random", 3), EW.select_windows(6400, 64, 7, "random", 3)
    assert np.array_equal(r1, r2) and len(set(r1.tolist())) == 7 and (np.diff(r1) > 0).all() and r1.max() < 100
    assert not np.array_equal(r1, EW.select_windows(6400, 64, 7, "random", 4))
    assert len(EW.select_windows(640, 64, 50)) == 10
    with pytest.raises(ValueError):
        EW.select_windows(10, 64, 1)
    h = EW.split_halves(11, seed=2)
    assert set(h.tolist()) == {0, 1} and abs(int((h == 0).sum()) - int((h == 1).sum())) <= 1
    assert np.array_equal(h, EW.split_halves(11, seed=2)) and not np.array_equal(h, EW.split_halves(11, seed=3))
    path = write_bin(tmp_path / "a.bin", 64 * 12, 0)
    ws = EW.WindowSet("a", path, 64, 12)
    hd = ws.head(5)
    assert np.array_equal(hd.index, ws.index[:5]) and np.array_equal(hd.hashes, ws.hashes[:5])
    assert len(set(ws.hashes.tolist())) == 12
    assert np.array_equal(ws.tokens(3), np.fromfile(path, dtype=np.uint32)[192:256].astype(np.int64))


def test_window_spans_and_document_aligned_split(tmp_path):
    """prereg sec. 4.3: selection half / evaluation half of a test bin (token midpoint; PG-19: book boundary)."""
    assert EW.select_windows(64 * 10, 64, 3, "first", span=(0, 320)).tolist() == [0, 1, 2]
    assert EW.select_windows(64 * 10, 64, 3, "even", span=(320, 640)).tolist() == [5, 7, 9]
    assert EW.select_windows(64 * 10, 64, 100, "even", span=(100, 400)).tolist() == [2, 3, 4, 5]
    assert np.array_equal(EW.select_windows(640, 64, 4, "even"), EW.select_windows(640, 64, 4, "even", span=(0, 640)))
    with pytest.raises(ValueError):
        EW.select_windows(640, 64, 1, span=(10, 60))
    path = write_bin(tmp_path / "b.bin", 64 * 20, 1, doc_len=100)            # EOT at 99, 199, ...
    toks = np.fromfile(path, dtype=np.uint32)
    assert EW.split_point(toks, 0.5) == 640 and EW.split_point(toks, 0.0) == 0 and EW.split_point(toks, 1.0) == 1280
    sp = EW.split_point(toks, 0.5, EOT, "doc")
    assert sp == 700 and toks[sp - 1] == EOT
    sel = EW.WindowSet("s", path, 64, 4, "even", span=(0, 0.5), span_align="doc")
    ev = EW.WindowSet("e", path, 64, 4, "even", span=(0.5, 1), span_align="doc")
    assert sel.span_tokens == [0, 700] and ev.span_tokens == [700, 1280]
    assert sel.index.tolist() == [0, 3, 6, 9] and ev.index.tolist() == [11, 14, 16, 19]
    assert (sel.starts + 64 <= 700).all() and (ev.starts >= 700).all()
    assert ev.describe()["span_tokens"] == [700, 1280] and "span" not in EW.WindowSet("a", path, 64, 2).describe()
    tm = EW.WindowSet("m", path, 64, 4, "even", span=(0.5, 1))               # token midpoint (no alignment)
    assert tm.span_tokens == [640, 1280] and tm.index.min() == 10


@pytest.mark.parametrize("kind", ["nestor", "moe"])
def test_per_window_ce_and_routing_summary_invariants(tmp_path, kind):
    path = write_bin(tmp_path / "a.bin", 64 * 6, 0)
    ws = EW.WindowSet("a", path, 64, 6)
    m = tiny(kind)
    res = EW.evaluate(m, ws, routing=True, log=None)
    S = 64
    for i in range(len(ws.index)):
        x = torch.as_tensor(ws.tokens(i)).view(1, -1)
        with torch.no_grad():
            lg = m(x).pred[0, :-1].float()
        ce = float(torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum"))
        assert abs(res["ce_sum"][i] - ce) <= 1e-5 * abs(ce) and res["n_tok"][i] == S - 1
    rt = res["routing"]
    assert rt, "no routing summaries"
    for key, arr in rt.items():
        cnt, g, g2 = arr[:, 0], arr[:, 1], arr[:, 2]
        assert (g >= 0).all() and (g2 <= g + 1e-6).all() and (g <= cnt + 1e-6).all(), key
        fam, b = key.split(":")[0], int(key.split(":")[1][1:])
        if fam == "column":
            assert np.allclose(cnt.sum(-1), NESTOR_K[b] * S) and np.allclose(g.sum(-1), S, atol=1e-3), key
        elif fam == "priv":
            assert np.allclose(cnt.sum(-1), NESTOR_K[b] * NESTOR_KQ[b] * S), key
        elif fam == "pub":
            assert np.allclose(cnt.sum(-1), PUB_K * S), key
        elif fam == "moe":
            assert np.allclose(cnt.sum(-1), MOE_K * S) and np.allclose(g.sum(-1), S, atol=1e-3), key
    fams = {k.split(":")[0] for k in rt}
    assert fams == ({"column", "pub", "priv"} if kind == "nestor" else {"moe"}), fams


def test_private_effective_gate_is_column_times_alpha_times_switch(tmp_path):
    path = write_bin(tmp_path / "a.bin", 64 * 2, 0)
    ws = EW.WindowSet("a", path, 64, 1)
    m = tiny("nestor")
    res = EW.evaluate(m, ws, routing=True, log=None)
    cap = {}
    pool = m.backbone.blocks[0].layers[0].expert_pool
    h = pool.path_gate.register_forward_hook(lambda mod, i, o: cap.__setitem__("pg", o.detach()))
    x = torch.as_tensor(ws.tokens(0)).view(1, -1)
    with torch.no_grad():
        out = m(x)
    h.remove()
    alpha = torch.softmax(cap["pg"].float(), -1)[..., 1]
    colg = out.aux["blocks"][0]["group_router"]["gates"].float()
    man = torch.cat([((pc["gates"].float() * alpha.unsqueeze(-1) * colg[..., c:c + 1]) ** 2).sum(dim=(0, 1))
                     for c, pc in enumerate(out.aux["blocks"][0]["layers"][0]["expert_pool_private"])]).numpy()
    assert np.allclose(man, res["routing"]["priv:b0:l0"][0, 2], rtol=1e-6, atol=1e-9)


def test_paired_statistics_save_load_and_forget_table(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    a = {"index": np.arange(n), "starts": np.arange(n) * 64, "hash": rng.integers(0, 2 ** 62, n).astype(np.uint64),
         "ce_sum": rng.uniform(300, 320, n), "n_tok": np.full(n, 63), "n_bytes": np.full(n, 250)}
    b = dict(a, ce_sum=a["ce_sum"] + 63 * 0.02 + rng.normal(0, 0.05, n))
    r = EW.paired(a, b)
    manual = b["ce_sum"].sum() / b["n_tok"].sum() - a["ce_sum"].sum() / a["n_tok"].sum()
    d = b["ce_sum"] / 63 - a["ce_sum"] / 63
    assert abs(r["diff"] - manual) < 1e-12 and abs(r["se"] - d.std(ddof=1) / math.sqrt(n)) < 1e-12
    assert abs(r["z"] - r["diff"] / r["se"]) < 1e-9 and r["n_windows"] == n
    assert abs(r["diff_bits_per_byte"] - (b["ce_sum"].sum() - a["ce_sum"].sum()) / math.log(2) / a["n_bytes"].sum()) < 1e-12
    with pytest.raises(ValueError, match="different token content"):
        EW.paired(a, dict(b, hash=b["hash"] + np.uint64(1)))
    sub = {k: (v[5:] if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    assert EW.paired(a, sub)["n_windows"] == 15
    res = dict(a, routing={"moe:b0:l0": rng.random((n, 3, 6))}, flip_moe=rng.integers(0, 5, (n, 2)))
    f = EW.save_result(str(tmp_path / "x" / "ckpt__old.npz"), res, {"variant": "ckpt", "k": [1, 2]})
    back = EW.load_result(f)
    assert back["meta"] == {"variant": "ckpt", "k": [1, 2]}
    assert all(np.array_equal(back[k], res[k]) for k in ("index", "hash", "ce_sum", "n_tok", "n_bytes", "flip_moe"))
    assert np.array_equal(back["routing"]["moe:b0:l0"], res["routing"]["moe:b0:l0"])
    EW.save_result(str(tmp_path / "before" / "ckpt__old.npz"), a, {})
    EW.save_result(str(tmp_path / "after" / "ckpt__old.npz"), b, {})
    ft = EW.forget_table(str(tmp_path / "before"), str(tmp_path / "after"))
    assert set(ft) == {"old"} and abs(ft["old"]["diff"] - r["diff"]) < 1e-12


@pytest.mark.parametrize("kind", ["nestor", "moe"])
def test_routing_pin_replay_flips_and_guards(tmp_path, kind):
    path = write_bin(tmp_path / "a.bin", 64 * 6, 0)
    ws = EW.WindowSet("a", path, 64, 6)
    base = tiny(kind)
    ref = EW.evaluate(base, ws, log=None)
    with EW.RoutingPin().install(base) as pin:                       # self replay: bitwise, zero flips
        own = EW.evaluate(base, ws, pin=pin, pin_source=base, routing=True, log=None)
    assert np.array_equal(own["ce_sum"], ref["ce_sum"])
    flips = {k: v for k, v in own.items() if k.startswith("flip_")}
    assert flips and all(int(v[:, 0].sum()) == 0 and int(v[:, 1].sum()) > 0 for v in flips.values())
    # experts of the very last routed layer only: routing inputs unchanged -> pinning is a no-op
    tail = tiny(kind)
    pre = last_layer_prefix(tail)
    perturb_(tail, lambda n: n.startswith(pre) and "stacked_" in n)
    free = EW.evaluate(tail, ws, log=None)
    with EW.RoutingPin().install(tail, base) as pin:
        pinned = EW.evaluate(tail, ws, pin=pin, pin_source=base, log=None)
    assert not np.array_equal(free["ce_sum"], ref["ce_sum"]) and np.array_equal(pinned["ce_sum"], free["ce_sum"])
    # upstream perturbation: own routing moves; pinned differs from free and flips are counted
    up = tiny(kind)
    perturb_(up, lambda n: "stacked_" in n or "attention" in n, scale=0.1)
    with EW.RoutingPin().install(up, base) as pin:
        pu = EW.evaluate(up, ws, pin=pin, pin_source=base, routing=True, log=None)
    fu = EW.evaluate(up, ws, log=None)
    assert not np.array_equal(pu["ce_sum"], fu["ce_sum"])
    assert sum(int(v[:, 0].sum()) for k, v in pu.items() if k.startswith("flip_")) > 0
    # guards: other architecture -> call-order mismatch; unreplayed decisions; double install
    other = tiny("moe" if kind == "nestor" else "nestor")
    x = torch.as_tensor(ws.tokens(0)).view(1, -1)
    with EW.RoutingPin().install(base, other) as pin:
        pin.record()
        with torch.no_grad():
            EW.forward_ce(base, x)
        pin.replay()
        with pytest.raises(RuntimeError, match="call order mismatch|more decisions"):
            with torch.no_grad():
                EW.forward_ce(other, x)
        pin.record()
        with torch.no_grad():
            EW.forward_ce(base, x)
        pin.replay()
        with pytest.raises(RuntimeError, match="not replayed"):
            pin.done()
        with pytest.raises(RuntimeError, match="another RoutingPin"):
            EW.RoutingPin().install(base)
    assert "RoutingPin" not in NB.Router.forward.__qualname__


def test_no_numpy_blas_after_torch_import():
    """paired() and the MI estimator run in a process that already loaded torch (numpy BLAS would abort it
    on pip-torch + conda-numpy installs: OMP Error #15)."""

    code = ("import sys; sys.path[:0] = ['.', 'repo']\n"
            "import torch, numpy as np\n"
            "import eval_windows as EW, routing_stats as RS\n"
            "n = 9; a = {'index': np.arange(n), 'hash': np.arange(n, dtype=np.uint64), 'ce_sum': np.arange(n) * 1.0 + 5,\n"
            "     'n_tok': np.full(n, 3)}\n"
            "b = dict(a, ce_sum=a['ce_sum'] * 1.1)\n"
            "r = EW.paired(a, b)\n"
            "mi = RS.mutual_information_bits(np.array([[5, 1], [1, 5]]))\n"
            "print('OK', round(r['corr'], 12), round(mi, 12))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE, timeout=300)
    assert p.returncode == 0 and "OK 1.0" in p.stdout, (p.stdout + p.stderr)[-2000:]
    x = np.arange(9) * 1.0 + 5
    assert abs(EW._corr(x, x * 1.1) - 1.0) < 1e-12 and abs(EW._corr(x, -x) + 1.0) < 1e-12
    pya = np.array([[5, 1], [1, 5]]) / 12
    want = sum(pya[i, j] * math.log2(pya[i, j] / 0.25) for i in range(2) for j in range(2))
    assert abs(T.RS.mutual_information_bits(np.array([[5, 1], [1, 5]])) - want) < 1e-12


# ============================================================================ weight_controls
def _trained_pair(kind: str = "nestor", spec=None, steps: int = 3):
    base = tiny(kind)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    m = T.build(T.nestor_backbone() if kind == "nestor" else T.moe_backbone(), seed=0)
    plan = SA.apply_subset(m, spec or (T.NESTOR_SPEC if kind == "nestor" else T.MOE_SPEC))
    freeze_routers(m)
    opt, sch = T.make_opt(m, steps)
    T.train(m, opt, sch, steps)
    return base, base_sd, m, {k: v.clone() for k, v in m.state_dict().items()}, plan


def test_sv_matched_random_direction_state():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(48, 80) @ torch.diag(torch.linspace(0.1, 3, 80))
    r = WC.sv_matched_random(d, g)
    s0, s1 = torch.linalg.svdvals(d.double()), torch.linalg.svdvals(r.double())
    assert float((s0 - s1).abs().max() / s0.max()) < 1e-6
    assert abs(float((d * r).sum() / d.norm() / r.norm())) < 0.2            # a different direction
    _, base_sd, _, ad_sd, _ = _trained_pair("nestor")
    rep: dict = {}
    rd = WC.random_direction_state(base_sd, ad_sd, seed=3, report=rep)
    assert rep, "no changed tensors"
    worst, slices = 0.0, 0
    for k, a in ad_sd.items():
        if not torch.is_floating_point(a):
            continue
        d0, d1 = a.float() - base_sd[k].float(), rd[k].float() - base_sd[k].float()
        if not bool((d0 != 0).any()):
            assert torch.equal(rd[k], a), k                                   # untouched tensors exact
            continue
        if d0.ndim == 1:
            assert abs(float(d0.norm() - d1.norm())) <= 1e-5 * float(d0.norm()), k
            continue
        pairs = [(d0, d1)] if d0.ndim == 2 else [(d0[i], d1[i]) for i in range(d0.shape[0])]
        for x0, x1 in pairs:
            if not bool((x0 != 0).any()):
                assert bool((x1 == 0).all()), k                              # frozen rows stay exact
                continue
            a0, a1 = torch.linalg.svdvals(x0.double()), torch.linalg.svdvals(x1.double())
            worst = max(worst, float((a0 - a1).abs().max() / a0.max()))
            slices += 1
    assert slices > 10 and worst < 1e-4, (slices, worst)
    rd2, rd3 = WC.random_direction_state(base_sd, ad_sd, seed=3), WC.random_direction_state(base_sd, ad_sd, seed=4)
    assert all(torch.equal(rd[k], rd2[k]) for k in rd) and any(not torch.equal(rd[k], rd3[k]) for k in rd)


@pytest.mark.parametrize("kind", ["nestor", "moe"])
def test_revert_groups_partition_and_exact_reverts(kind):
    groups = ["embed_head", "pub", "priv", "attn", "shared", "moe", "router"]
    _, base_sd, m, ad_sd, _ = _trained_pair(kind)
    part = WC.group_partition(m, groups)
    assert set(k for k in part if k.startswith("_")) == {"_one"}, part      # every element in exactly one group
    names = {n for n, _ in m.named_parameters()}
    assert all(torch.equal(WC.revert_state(m, ad_sd, base_sd, groups)[k], base_sd[k]) for k in names)
    assert all(torch.equal(WC.revert_state(m, ad_sd, base_sd, [])[k], ad_sd[k]) for k in ad_sd)
    grp = "pub" if kind == "nestor" else "moe"
    rv = WC.revert_state(m, ad_sd, base_sd, [grp])
    for e in SA.build_catalog(m):
        flags = [u.kind == grp for u in e.units]
        a, b, r = ad_sd[e.name], base_sd[e.name], rv[e.name]
        if e.sliced:
            assert all(torch.equal(r[i], b[i] if f else a[i]) for i, f in enumerate(flags)), e.name
        else:
            assert torch.equal(r, b if flags[0] else a), e.name
    with pytest.raises(ValueError, match="unknown revert group"):
        WC.revert_state(m, ad_sd, base_sd, ["bogus"])


def test_displacement_by_kind_and_unit():
    m = tiny("nestor")
    ref = {k: v.clone() for k, v in m.state_dict().items()}
    cat = {e.name: e for e in SA.build_catalog(m)}
    bank = next(n for n, e in cat.items() if e.sliced and e.units[1].kind == "priv")
    attn = next(n for n, e in cat.items() if not e.sliced and e.units[0].kind == "attn" and e.param.ndim == 2)
    p = dict(m.named_parameters())
    with torch.no_grad():
        p[bank][1].add_(0.5)
        p[attn].add_(0.25)
    d = WC.displacement(m, ref)
    row = p[bank][1].numel()
    assert math.isclose(d["by_unit_sq"][cat[bank].units[1].key], 0.25 * row, rel_tol=1e-6)
    assert math.isclose(d["by_kind_sq"]["attn"], 0.0625 * p[attn].numel(), rel_tol=1e-6)
    assert math.isclose(d["total_sq"], 0.25 * row + 0.0625 * p[attn].numel(), rel_tol=1e-6)


def test_wd_only_refuses_live_routers():
    with pytest.raises(ValueError, match="frozen-router runs only"):
        WC.wd_only_state("tiny_nestor", {}, {}, {"NESTOR_FREEZE_ROUTERS": "0"})


# ============================================================================ crossover
STRATA = ["moe:b0:l0", "moe:b1:l0"]
A_MOD = 12


def _synth_usage(rng):
    """Per-stratum module usage: code high on modules 0-5, math high on 6-11, german/dclm unrelated."""

    base = np.linspace(0.5, 1.5, A_MOD)
    use = {"code": [], "math": [], "german": [], "dclm": []}
    for _ in STRATA:
        tilt = np.r_[np.full(6, 2.0), np.full(6, 0.5)] * rng.uniform(0.7, 1.3, A_MOD)
        use["code"].append(0.01 * base * tilt)
        use["math"].append(0.01 * base / tilt)
        use["german"].append(0.01 * rng.uniform(0.5, 1.5, A_MOD))
        use["dclm"].append(0.01 * rng.uniform(0.5, 1.5, A_MOD))
    return use


def _synth_res(usage: list, rng, n_win: int = 24, block: int = 64, noise: float = 0.02) -> dict:
    routing = {}
    for s, lk in enumerate(STRATA):
        u = usage[s][None, :] * (1 + noise * rng.standard_normal((n_win, A_MOD)))
        arr = np.zeros((n_win, 3, A_MOD))
        arr[:, 2] = u * block
        arr[:, 1] = 2 * u * block
        arr[:, 0] = 4 * u * block
        routing[lk] = arr
    return {"ce_sum": np.zeros(n_win), "routing": routing, "meta": {"set": {"block": block}}}


def _synth_table(seed: int = 0) -> CX.FamilyTable:
    rng = np.random.default_rng(seed)
    use = _synth_usage(rng)
    t = CX.FamilyTable("moe")
    for c in ("code", "math", "german", "dclm"):
        t.add_corpus(c, _synth_res(use[c], rng), half_seed=0)
    return t


def test_crossfit_tables_ro_and_shares():
    rng = np.random.default_rng(1)
    use = _synth_usage(rng)
    res = _synth_res(use["code"], rng)
    half = EW.split_halves(24, seed=0)
    tab = CX.corpus_table(res, "moe", half, 1)
    sel = half == 1
    manual = res["routing"]["moe:b0:l0"][sel].sum(0)[2] / (sel.sum() * 64)
    assert np.allclose(tab["moe:b0:l0"]["g2"], manual) and tab["moe:b0:l0"]["keys"][3] == "moe:b0:l0:e3"
    t = CX.FamilyTable("moe")
    t.add_corpus("code", res, half_seed=0)
    pick = [np.array([0, 3]), np.array([5])]
    ro1 = t.ro("code", 1, pick)
    assert math.isclose(ro1, t.u["code"][1][0][[0, 3]].sum() + t.u["code"][1][1][[5]].sum(), rel_tol=1e-12)
    full = [np.arange(A_MOD), np.arange(A_MOD)]
    assert math.isclose(t.ro_share("code", 1, full), 1.0) and 0 < t.ro_share("code", 1, pick) < 1
    t.set_param_weights({k: 2 for ks in t.keys for k in ks})
    assert math.isclose(t.ro("code", 1, pick), 2 * ro1, rel_tol=1e-12)
    assert np.allclose(t.u["code"]["all"][0], 0.5 * (t.u["code"][0][0] + t.u["code"][1][0]))


def test_draw_pairs_constraints_crossfit_and_distinctness():
    t = _synth_table(0)
    k, tol, mj = 3, 0.05, 0.5
    rep = CX.draw_pairs(t, k=k, contrast={"type": "crossover", "low": "code", "high": "math"}, match=["german", "dclm"],
                        tol=tol, n_draws=3, seed=0, max_jaccard=mj)
    assert rep["n_draws_found"] == 3, rep["attempts"]
    for d in rep["draws"]:
        sel = d["selection_half"]
        assert sel["contrast"] >= math.log(1.3) and all(abs(v) <= tol + 1e-12 for v in sel["match_err"].values())
        est = d["estimation_half"]
        assert est["contrast"] > 0.5 * sel["contrast"]                       # cross-fitted contrast survives
        for s in range(len(STRATA)):
            a = [x for x in d["A"] if x.startswith(STRATA[s] + ":")]
            b = [x for x in d["B"] if x.startswith(STRATA[s] + ":")]
            assert len(a) == k and len(b) == k and not set(a) & set(b)
        ea = [int(x.split(":e")[1]) for x in d["A"]]
        assert sum(e >= 6 for e in ea) > sum(e < 6 for e in ea)              # A = low code / high math modules
        assert all(j <= mj for j in d["jaccard_to_other_draws"]["A"] + d["jaccard_to_other_draws"]["B"])
        assert d["spec_A"] == CX.spec_for(d["A"])
    assert rep["best_feasible"] is not None
    assert rep["best_feasible"]["selection_contrast"] >= max(d["selection_half"]["contrast"] for d in rep["draws"]) - 1e-9
    tilt = CX.draw_pairs(t, k=k, contrast={"type": "tilt", "corpus": "code"}, match=["dclm"], tol=tol, n_draws=2,
                         seed=1, min_contrast=0.1)
    assert tilt["n_draws_found"] == 2 and all(d["selection_half"]["contrast"] >= 0.1 for d in tilt["draws"])
    with pytest.raises(ValueError, match="2k"):
        CX.draw_pairs(t, k=7, contrast={"type": "tilt", "corpus": "code"}, match=[], n_draws=1)


@pytest.mark.parametrize("kind,keys", [
    ("moe", ["moe:b0:l0:e1", "moe:b1:l1:e4", "moe:b1:l1:e5"]),
    ("nestor", ["priv:b0:l0:c1:e2", "priv:b0:l0:c3:e0", "pub:b1:l1:e0"]),
    ("nestor", ["column:b0:c1", "column:b1:c2"]),
])
def test_spec_for_trains_exactly_the_drawn_units(kind, keys):
    m = T.build(T.nestor_backbone() if kind == "nestor" else T.moe_backbone(), seed=0)
    spec = CX.spec_for(keys)
    plan = SA.apply_subset(m, spec)
    got = {u.key for u in plan.trainable_units}
    if keys[0].startswith("column"):
        want = {(int(k.split(":")[1][1:]), int(k.split(":")[2][1:])) for k in keys}
        assert got and {(u.block, u.column) for u in plan.trainable_units} == want
        assert {u.kind for u in plan.trainable_units} == {"attn", "priv"}
    else:
        assert got == set(keys), (got, keys)


def _eval_dir(d, ces: dict, n: int = 12, seed: int = 0):
    """eval_windows-format results <d>/ckpt__<corpus>.npz with per-window mean CE `ces[corpus]` (+ noise)."""

    rng = np.random.default_rng(seed)
    for c, ce in ces.items():
        res = {"index": np.arange(n), "starts": np.arange(n) * 64, "hash": np.arange(n, dtype=np.uint64) + np.uint64(11),
               "ce_sum": (ce + 1e-4 * rng.standard_normal(n)) * 63.0, "n_tok": np.full(n, 63)}
        EW.save_result(os.path.join(str(d), f"ckpt__{c}.npz"), res, {})
    return str(d)


def test_c1_c2_effect_statistics_and_signflip(tmp_path):
    assert CX.signflip_p([1.0, 2.0, 3.0]) == 1 / 8 and CX.signflip_p([1.0, -1.0]) == 0.75
    assert CX.signflip_p([-1.0, -2.0]) == 1.0 and math.isnan(CX.signflip_p([]))
    base = _eval_dir(tmp_path / "base", {"code": 3.0, "math": 2.0, "dclm": 4.0})
    pairs, want = {}, {}
    for i, (fa_c, fb_c, fa_m, fb_m) in enumerate([(0.02, 0.04, 0.03, 0.015), (0.01, 0.03, 0.02, 0.02), (0.05, 0.04, 0.01, 0.02)]):
        a = _eval_dir(tmp_path / f"A{i}", {"code": 3.0 + fa_c, "math": 2.0 + fa_m}, seed=i + 1)
        b = _eval_dir(tmp_path / f"B{i}", {"code": 3.0 + fb_c, "math": 2.0 + fb_m}, seed=i + 7)
        pairs[f"d{i}"] = (a, b)
        want[f"d{i}"] = (math.log(fb_c) - math.log(fa_c)) - (math.log(fb_m) - math.log(fa_m))
    rep = CX.c1_interaction(base, pairs)
    assert rep["n"] == 3 and all(abs(rep["draws"][d]["I"] - want[d]) < 0.05 for d in want)
    assert abs(rep["mean_I"] - np.mean(list(want.values()))) < 0.05 and rep["p_one_sided_signflip"] == CX.signflip_p(
        [rep["draws"][d]["I"] for d in sorted(want)])
    sgd = {"d0": (_eval_dir(tmp_path / "sA", {"dclm": 4.04}), _eval_dir(tmp_path / "sB", {"dclm": 4.01}))}
    muon = {"d0": (_eval_dir(tmp_path / "mA", {"dclm": 4.02}), _eval_dir(tmp_path / "mB", {"dclm": 4.02}))}
    r2 = CX.c2_interaction(base, sgd, muon)
    assert abs(r2["draws"]["d0"]["I_opt"] - (math.log(0.04 / 0.01) - 0.0)) < 0.05 and r2["n"] == 1
    out = str(tmp_path / "c1.json")
    assert CX.main(["interaction", "--base", base] + sum([["--pair", f"{d}={a},{b}"] for d, (a, b) in pairs.items()], [])
                   + ["--out", out]) == 0
    assert json.load(open(out))["n"] == 3


def test_stop_rule_and_decide():
    assert CX.stop_rule({"moe": 0.1, "nestor": -0.2})["decision"] == "STOP_PIVOT"
    assert CX.stop_rule({"moe": 0.1, "nestor": -0.3})["decision"] == "GO"
    assert CX.stop_rule({"moe": None, "nestor": None})["decision"] == "STOP_PIVOT"
    d = G0.decide([{"arch": "moe", "best_C1_abs_I": 0.4}, {"arch": "nestor", "best_C1_abs_I": 0.1}],
                  {"pass": False, "diff": 2e-3, "corr": 0.99})
    assert d["decision"] == "GO" and d["nestor_adaptation"] == "STOP"


# ============================================================================ adapt_extras
def _sgd_settings(lr=0.05, mu=0.9, wd=0.01, kinds=("moe",)):
    return {"lr": lr, "momentum": mu, "nesterov": True, "weight_decay": wd, "scale": "none", "kinds": kinds}


def test_sgd_arm_is_exact_momentum_sgd_and_leaves_other_groups_alone():
    cfg = MuonAdamWConfig(lr_muon=0.0025, lr_adamw=7.5e-5, weight_decay=0.1, betas=(0.9, 0.95))
    m1, m2 = tiny("moe"), tiny("moe")
    o1 = build_optimizer(cfg, m1)
    o2 = AX.convert_to_sgd(build_optimizer(cfg, m2), m2, _sgd_settings())
    assert type(o2).__name__ == "MuonAdamWSGD"
    sgd = {id(p) for g in o2.param_groups if g["method"] == "sgd" for p in g["params"]}
    p1, p2 = dict(m1.named_parameters()), dict(m2.named_parameters())
    sgd_names = [n for n, p in p2.items() if id(p) in sgd]
    cat = {e.name: {u.kind for u in e.units} for e in SA.build_catalog(m2)}
    assert sgd_names and all(cat[n] == {"moe"} for n in sgd_names)
    name2 = {id(p): n for n, p in p2.items()}
    assert not [name2[id(p)] for g in o2.param_groups if g["method"] == "muon" for p in g["params"]
                if cat[name2[id(p)]] == {"moe"}]                             # every MoE matrix left Muon
    gen = torch.Generator().manual_seed(0)
    theta = {n: p2[n].detach().double().clone() for n in sgd_names}
    bufs: dict = {}
    for _ in range(3):
        for n, p in p1.items():
            if p.requires_grad:
                gr = torch.randn(p.shape, generator=gen)
                p.grad, p2[n].grad = gr.clone(), gr.clone()
        for n in sgd_names:
            gd = p2[n].grad.double()
            bufs[n] = bufs.get(n, torch.zeros_like(gd)) * 0.9 + gd
            theta[n] = theta[n] * (1 - 0.05 * 0.01) - 0.05 * (gd + 0.9 * bufs[n])
        o1.step(); o2.step()
        o1.zero_grad(set_to_none=True); o2.zero_grad(set_to_none=True)
    for n in sgd_names:
        assert torch.allclose(p2[n].detach().double(), theta[n], rtol=1e-5, atol=1e-6), n
    for n in p1:
        if n not in sgd_names:
            assert torch.allclose(p1[n], p2[n], rtol=1e-5, atol=1e-7), n     # Muon/AdamW groups unchanged
    bad = dict(_sgd_settings(), kinds=("pub",))
    with pytest.raises(RuntimeError, match="no trainable Muon matrix"):
        AX.convert_to_sgd(build_optimizer(cfg, tiny("moe")), tiny("moe"), bad)


def test_sgd_arm_keeps_frozen_rows_bit_identical_with_decay():
    m = T.build(T.moe_backbone(), seed=0)
    spec = {"mode": "train", "select": [{"kind": "moe", "block": 0, "layer": 0, "expert": [0, 4]},
                                        {"kind": "moe", "block": 1, "layer": 1, "expert": [2]}]}
    plan = SA.apply_subset(m, spec)
    freeze_routers(m)
    before = T.snapshot(m)
    opt = AX.convert_to_sgd(build_optimizer(MuonAdamWConfig(lr_muon=0.0025, lr_adamw=7.5e-5, weight_decay=0.1,
                                                            betas=(0.9, 0.95)), m), m, _sgd_settings(lr=0.5, wd=0.1))
    sch = build_scheduler(CosineAnnealingConfig(T_max=4, warmup_steps=1, eta_min=1e-5), opt, 4)
    T.train(m, opt, sch, 4)
    viol, unchanged = T.audit(plan, before, m)
    assert viol == [] and unchanged == [], (viol, unchanged[:5])
    for p in m.parameters():
        if getattr(p, "_subset_frozen_rows", None) is not None and p in opt.state:
            buf = opt.state[p].get("momentum_buffer")
            if buf is not None:
                assert bool((buf[p._subset_frozen_rows] == 0).all())


def test_sgd_lr_grid_and_g2_selection_rule():
    probe = {"lr_muon": 0.0025, "subsets": {"S": {"grad_sq": 4.0, "udir_sq": 9.0}}}
    g = AX.sgd_lr_grid(probe, "S", momentum=0.95)
    assert math.isclose(g["lr_first_step_matched"], 0.0075 / (1.95 * 2)) and math.isclose(g["lr_steady_state_matched"], 0.0075 * 0.05 / 2)
    assert math.isclose(g["grid"][1] ** 2, g["grid"][0] * g["grid"][2]) and g["grid"][0] < g["grid"][2]
    assert math.isclose(g["grid"][2] / g["grid"][0], 1 / (1 - 0.95 ** 2))
    n = 8
    idx, hsh = np.arange(n), np.arange(n, dtype=np.uint64) + np.uint64(7)

    def r(ce):
        return {"index": idx, "hash": hsh, "ce_sum": np.full(n, ce * 63.0), "n_tok": np.full(n, 63)}
    base, muon = r(5.0), {8: r(4.6), 24: r(4.0), 48: r(3.8)}
    runs = {0.1: {8: r(4.98), 24: r(4.9)}, 0.3: {8: r(4.8), 24: r(4.3)}, 1.0: {8: r(4.6), 24: r(3.5)},
            3.0: {8: r(5.2), 24: r(3.0)}}
    s = AX.sgd_select(base, muon, runs)
    rows = {x["lr"]: x for x in s["runs"]}
    assert not rows[0.1]["feasible"] and rows[0.3]["feasible"] and rows[1.0]["feasible"] and not rows[3.0]["stable"]
    assert s["G2_pass"] and s["chosen_lr"] == 0.3 and math.isclose(s["muon_gain"], 1.0)
    assert not AX.sgd_select(base, muon, {0.1: runs[0.1], 3.0: runs[3.0]})["G2_pass"]


def test_adapt_extras_cli_outputs(tmp_path):
    probe = _dump(tmp_path / "probe.json", {"lr_muon": 0.0025, "subsets": {"S": {"grad_sq": 4.0, "udir_sq": 9.0}}})
    out = str(tmp_path / "g2" / "grid.json")
    assert AX.main(["sgd-lr", "--probe", probe, "--subset", "S", "--out", out]) == 0
    assert len(json.load(open(out))["grid"]) == 3
    d = tmp_path / "interim"
    base = _eval_dir(tmp_path / "base", {"german": 5.0})
    for s, ce in ((8, 4.7), (24, 4.0)):
        r = EW.load_result(os.path.join(base, "ckpt__german.npz"))
        EW.save_result(str(d / f"step{s:03d}__german.npz"), dict(r, ce_sum=r["ce_sum"] * ce / 5.0), {})
    bad = tmp_path / "bad"
    for s, ce in ((8, 5.3), (24, 4.9)):
        r = EW.load_result(os.path.join(base, "ckpt__german.npz"))
        EW.save_result(str(bad / f"step{s:03d}__german.npz"), dict(r, ce_sum=r["ce_sum"] * ce / 5.0), {})
    argv = ["sgd-select", "--base", os.path.join(base, "ckpt__german.npz"), "--muon", str(d), "--run", f"0.1={bad}",
            "--set", "german"]
    assert AX.main(argv) == 3                                                # G2 fails -> exit 3 without --out
    g2 = str(tmp_path / "g2" / "select.json")
    assert AX.main(argv + ["--out", g2]) == 0 and json.load(open(g2))["G2_pass"] is False


def test_adapt_extras_install_is_inert_without_env(tmp_path):
    code = ("import json, os\n"
            "import test_subset_adapt\n"
            "import adapt_extras as AX\n"
            "from nestor.train.trainer import Trainer\n"
            "from nestor.pipelines.train import TrainPipeline\n"
            "import torch.nn.utils as U\n"
            "import nestor.train.optim as O\n"
            "AX.install()\n"
            "import test_subset_adapt as T\n"
            "m = T.build(T.moe_backbone())\n"
            "o = T.make_opt(m, 2)[0]\n"
            "print(json.dumps({'trainer': getattr(Trainer._update_weights, '_extras_patched', False),\n"
            "  'setup': getattr(TrainPipeline.setup, '_extras_patched', False),\n"
            "  'clip': getattr(U.clip_grad_norm_, '_extras_patched', False), 'opt': type(o).__name__}))\n")
    env = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out == {"trainer": False, "setup": False, "clip": False, "opt": "MuonAdamW"}, out


# ============================================================================ g0_probe
def test_muon_group_lr_is_lr_muon_times_muon_scale():
    for kind in ("nestor", "moe"):
        m = tiny(kind)
        opt, _ = T.make_opt(m, 4)
        n = 0
        for g in opt.param_groups:
            if g["method"] != "muon":
                continue
            for p in g["params"]:
                s = float(get_scaling(p).muon_scale)
                assert abs(g["initial_lr"] / 0.0025 - s) <= 1e-7 * s
                n += 1
        assert n > 10


@pytest.mark.parametrize("kind", ["nestor", "moe"])
def test_muon_direction_is_the_optimizers_first_step(tmp_path, kind):
    m = tiny(kind)
    ws = EW.WindowSet("b", write_bin(tmp_path / "b.bin", 64 * 4, 1), 64, 4)
    params = G0.expert_params(m)
    grads = G0.target_gradient(m, ws, params, 4, device="cpu", bf16=False)
    assert grads and set(grads) <= set(params) and all(bool((g != 0).any()) for g in grads.values())
    u = G0.muon_direction(params, grads)
    opt = build_optimizer(MuonAdamWConfig(lr_muon=0.0025, lr_adamw=7.5e-5, weight_decay=0.0, betas=(0.9, 0.95)), m)
    before = {k: params[k].detach().clone() for k in grads}
    for p in m.parameters():
        p.grad = None
    for k in grads:
        params[k].grad = grads[k].clone()
    opt.step()
    for k in grads:
        p = params[k]
        step = (p.detach() - before[k]) / (-0.0025)
        assert float((step - u[k]).norm() / u[k].norm()) < 1e-4, k


def test_directional_probe_lin_matches_autograd_and_pinned_quadratic_regime(tmp_path):
    m = tiny("nestor")
    wsb = EW.WindowSet("b", write_bin(tmp_path / "b.bin", 64 * 4, 1), 64, 4)
    wsa = EW.WindowSet("a", write_bin(tmp_path / "a.bin", 64 * 6, 0), 64, 6)
    params = G0.expert_params(m)
    u = G0.muon_direction(params, G0.target_gradient(m, wsb, params, 4, device="cpu", bf16=False))
    rows = G0.unit_rows(m)
    tailb = SA.model_geometry(m)["dense_tail_block"]
    u_tail = G0.restrict(u, rows, [k for k in rows if k.startswith((f"priv:b{tailb}:", f"pub:b{tailb}:"))])
    old = [(wsa, i) for i in range(6)]

    def autograd(d, pinned: bool):
        ps = dict(m.named_parameters())
        flags = {n: q.requires_grad for n, q in ps.items()}
        for q in ps.values():
            q.requires_grad_(False)
        for n in d:
            ps[n].requires_grad_(True)
            ps[n].grad = None
        pin = EW.RoutingPin().install(m) if pinned else None
        try:
            for ws, i in old:
                x = torch.as_tensor(ws.tokens(i)).view(1, -1)
                if pin:
                    pin.record()
                    with torch.no_grad():
                        m(x)
                    pin.replay()
                lg = m(x).pred[0, :-1].double()
                if pin:
                    pin.done()
                (torch.nn.functional.cross_entropy(lg, x[0, 1:]) / len(old)).backward()
        finally:
            if pin:
                pin.uninstall()
        val = sum(float((ps[n].grad.double() * d[n].double()).sum()) for n in d if ps[n].grad is not None)
        for n, q in ps.items():
            q.grad = None
            q.requires_grad_(flags[n])
        return val

    eps = 1e-3
    r = G0.directional_probe(m, old, [("t", u_tail, eps, +1), ("t", u_tail, eps, -1)], device="cpu", log=None)
    q = G0.quad_forms(r, "t", eps)
    ag = autograd(u_tail, False)
    assert abs(q["Lin"]["value"] - ag) <= 1e-3 * abs(ag) and q["Q_kl"]["value"] > 0 and q["flip_frac"] == 0.0
    plan = [("E", u, e, s) for e in (eps, eps / 2) for s in (+1, -1)]
    rp = G0.directional_probe(m, old, plan, device="cpu", log=None, pin_routing=True)
    rich = G0.richardson(rp, "E", eps)               # O(eps^2) truncation of the central difference removed
    agp = autograd(u, True)
    assert abs(rich["lin_richardson"] - agp) <= 1e-3 * abs(agp), (rich, agp)
    assert abs(rich["lin_eps"] - agp) > abs(rich["lin_richardson"] - agp)
    assert G0.quad_forms(rp, "E", eps)["Q_kl"]["value"] > 0 and bool((rp["E|0.001|+1"]["kl"] >= 0).all())
    chk = G0.choose_eps(m, old[:4], u, device="cpu", eps0=0.0025, pin_routing=True, log=None)
    assert chk["ok"] and 3.5 <= chk["trail"][-1]["ratio"] <= 4.5
    free = G0.choose_eps(m, old[:4], u, device="cpu", eps0=0.0025, pin_routing=False, log=None, max_halvings=2)
    assert all(t["flip_frac_eps"] is not None for t in free["trail"])


def test_restrict_and_random_like_direction():
    m = tiny("nestor")
    params = G0.expert_params(m)
    d = {k: torch.randn(p.shape) for k, p in params.items()}
    rows = G0.unit_rows(m)
    keys = [k for k in rows if k.startswith("priv:b0:l0:c1:")][:2] + [k for k in rows if k.startswith("pub:b1:l1:")][:1]
    r = G0.restrict(d, rows, keys)
    for name, t in r.items():
        keep = {rr for k in keys for (n, rr) in rows[k] if n == name}
        for i in range(t.shape[0]):
            assert torch.equal(t[i], d[name][i]) if i in keep else bool((t[i] == 0).all())
    rl = G0.random_like_direction(r, seed=5)
    for name, t in r.items():
        for i in range(t.shape[0]):
            if bool((t[i] != 0).any()):
                s0, s1 = torch.linalg.svdvals(t[i].double()), torch.linalg.svdvals(rl[name][i].double())
                assert float((s0 - s1).abs().max() / s0.max()) < 1e-5
            else:
                assert bool((rl[name][i] == 0).all())
    assert math.isclose(G0.sq_norm(r), G0.sq_norm(rl), rel_tol=1e-5)


def test_g0_run_end_to_end_on_tiny_moe(tmp_path):
    ranges = {"dclm": (0, 255), "code": (0, 80), "math": (80, 160), "german": (160, 255), "books": (40, 200)}
    streams = {n: {"path": write_bin(tmp_path / f"{n}.bin", 64 * 24, i, lo, hi), "tokens": 64 * 24, "select": "even"}
               for i, (n, (lo, hi)) in enumerate(ranges.items())}
    cfg = {"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": str(tmp_path / "g0"), "block": 64,
           "device": "cpu", "streams": streams,
           "crossover": {"low": "code", "high": "math", "targets": ["german"], "match_extra": ["dclm"], "tol": 0.5,
                         "draws": 2, "min_contrast": 0.0, "max_jaccard": 1.0},
           "c2": {"target": "german", "match": ["dclm"], "draws": 1, "tol": 0.5, "min_contrast": 0.0, "max_jaccard": 1.0},
           "probe": {"targets": ["german"], "grad_windows": 2, "probe_corpora": ["dclm", "code"], "probe_windows": 2,
                     "random_seeds": [0], "regime_windows": 2, "bf16_grad": False, "pin_routing": "both"}}
    s = G0.run(cfg, model=tiny("moe"), log=None)
    assert set(s["contrasts"]) == {"C1_german", "C2_german"} and s["contrasts"]["C1_german"]["draws_found"] >= 1
    pr = s["probes"]["german"]
    assert pr["primary_routing"] == "pinned" and set(pr["eps_check"]) == {"pinned", "free"}
    labs = set(pr["subsets"])
    assert "E_ALL" in labs and any(x.startswith("C1_german_d0_") for x in labs) and "C2_german_d0_A" in labs
    e = pr["subsets"]["E_ALL"]
    assert {"pinned", "free", "all", "by_corpus", "grad_sq", "udir_sq"} <= set(e)
    assert set(e["by_corpus"]) == {"code", "dclm"} and "rand0" in e["pinned"] and "flip_frac" in e["free"]["all"]
    assert e["all"] == e["pinned"]["all"] and pr["Q_E"] == e["pinned"]["all"]["Q_kl"]
    assert "lin_richardson" in e["pinned"]["richardson"] and "lin_truncation_rel" in e["free"]["richardson"]
    assert os.path.exists(os.path.join(cfg["out"], "g0_summary.json"))
    grid = AX.sgd_lr_grid(json.load(open(os.path.join(cfg["out"], "probe__german.json"))), "C2_german_d0_A")
    assert 0 < grid["grid"][0] < grid["grid"][2]
    # per-window probe values (G0-6 input) reproduce the JSON means exactly; g0_gate computes G0-6 from them
    import g0_gate as GG
    wz = np.load(os.path.join(cfg["out"], pr["windows_file"]))
    eps = pr["eps"]
    q = (wz[f"pinned|E_ALL|{eps:.6g}|+1|kl"] + wz[f"pinned|E_ALL|{eps:.6g}|-1|kl"]) / (2 * eps * eps)
    assert math.isclose(float(q.mean()), pr["subsets"]["E_ALL"]["pinned"]["all"]["Q_kl"]["value"], rel_tol=1e-12)
    lin = (wz[f"free|E_ALL|{eps:.6g}|+1|dce"] - wz[f"free|E_ALL|{eps:.6g}|-1|dce"]) / (2 * eps)
    assert math.isclose(float(lin.mean()), pr["subsets"]["E_ALL"]["free"]["all"]["Lin"]["value"], rel_tol=1e-12)
    assert sorted(set(wz["corpus"].tolist())) == ["code", "dclm"] and len(wz["corpus"]) == 4 and str(wz["primary_routing"]) == "pinned"
    g6 = GG.g0_6({"b": {"dir": cfg["out"]}}, ["b"], {})["b"]
    assert g6["status"] == "computed" and g6["per_target"]["german"]["source"] == "per-window probe values"
    assert g6["per_target"]["german"]["cells"] == 2 * len([x for x in pr["subsets"] if "~rand" not in x])
    s2 = G0.run(cfg, model=tiny("moe"), log=None)                          # resumable: reuses every file
    assert s2["probes"]["german"]["subsets"].keys() == pr["subsets"].keys() and s2["best_C1_abs_I"] == s["best_C1_abs_I"]
    est = G0.estimate(G0.load_config(_dump(tmp_path / "cfg.json", cfg)))
    assert est["probe_tokens"] > 0 and est["routing_tokens"] == 5 * 64 * 24 and est["gpu_hours"] >= 0
    f = s["routing_files"]["dclm"]
    xp = G0.xplatform(f, f)
    assert xp["pass"] and xp["diff"] == 0.0


def test_g0_registered_layout_feeds_the_gate(tmp_path):
    """Registered streams (sec. 4.3): <c> = selection half, <c>_eval = evaluation half of the same test bin;
    g0_gate then reads selection statistics from <c> and predicted contrasts from <c>_eval (no crossfit deviation)."""
    import g0_gate as GG
    ranges = {"dclm": (0, 255), "code": (0, 80), "math": (80, 160), "german": (160, 255), "books": (40, 200)}
    streams = {}
    for i, (n, (lo, hi)) in enumerate(ranges.items()):
        p = write_bin(tmp_path / f"{n}.bin", 64 * 48, i, lo, hi)
        streams[n] = {"path": p, "tokens": 64 * 8, "select": "even", "span": [0, 0.5]}
        streams[n + "_eval"] = {"path": p, "tokens": 64 * 10, "select": "even", "span": [0.5, 1]}
    cfg = {"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": str(tmp_path / "g0" / "moe_b1"),
           "block": 64, "device": "cpu", "streams": streams}
    chk = G0.check_streams(cfg)
    assert chk["ok"] and chk["streams"]["code_eval"]["span_tokens"] == [64 * 24, 64 * 48]
    bad = G0.check_streams(dict(cfg, streams={"code": dict(streams["code"], tokens=64 * 30)}))
    assert bad["short"] == ["code"] and bad["streams"]["code"]["selected_windows"] == 24
    G0.run(cfg, model=tiny("moe"), log=None)
    for n in ranges:
        a = EW.load_result(os.path.join(cfg["out"], f"routing__{n}.npz"))
        b = EW.load_result(os.path.join(cfg["out"], f"routing__{n}_eval.npz"))
        assert len(a["index"]) == 8 and len(b["index"]) == 10
        assert int(a["index"].max()) * 64 + 64 <= 64 * 24 <= int(b["index"].min()) * 64
        assert a["meta"]["set"]["span_tokens"] == [0, 64 * 24] and b["meta"]["set"]["span_tokens"] == [64 * 24, 64 * 48]
    gcfg = _dump(tmp_path / "g0" / "gate.json", {"bases": {"moe_b1": {"dir": "moe_b1", "arch": "moe"}}, "code_streams": ["code"]})
    rec = GG.compute(GG.load_config(gcfg), n_draws=50, log=lambda *a: None)
    assert set(rec["layout"]["moe_b1"].values()) == {"registered"}
    assert not any("crossfit" in d for d in rec["deviations"])
    t = CX.reg_load_dir(cfg["out"], "moe")
    r1 = EW.load_result(os.path.join(cfg["out"], "routing__code_eval.npz"))
    assert np.allclose(t.est["code"], CX.reg_s2(r1, t.strata), rtol=0, atol=0)
    assert rec["gates"]["G0-4"]["per_base"]["moe_b1"]["k"] == max(1, math.ceil(t.N / 8))


def test_g0_margin_probes_on_tiny_moe(tmp_path):
    """G0-7 probe run (g0_margin.run) on a crossfit-layout G0 directory, as produced by the phase-3 g0_probe."""
    import g0_margin as GM
    ranges = {"dclm": (0, 255), "code": (0, 80), "math": (80, 160), "german": (160, 255), "books": (40, 200)}
    streams = {n: {"path": write_bin(tmp_path / f"{n}.bin", 64 * 24, i, lo, hi), "tokens": 64 * 8, "select": "even"}
               for i, (n, (lo, hi)) in enumerate(ranges.items())}
    d = str(tmp_path / "g0" / "moe_b1")
    G0.run({"experiment": "tiny_moe", "arch": "moe", "family": "moe", "k": 1, "out": d, "block": 64, "device": "cpu",
            "streams": streams}, model=tiny("moe"), log=None)
    t = CX.reg_load_dir(d, "moe")
    assert set(t.sources["code"]) >= {"layout", "selection", "estimation"} and t.sources["code"]["layout"] == "crossfit"
    A, B = t.subset_keys(np.zeros((t.L, 1), dtype=int)), t.subset_keys(np.ones((t.L, 1), dtype=int))
    ent = {"design": "c1", "d": 1, "seed": 5001, "status": "ok", "target": "german", "predicted": {"P": 0.3},
           "subsets": {"A": A, "B": B}, "sources": t.sources}
    CX.write_manifest(os.path.join(d, GM.MANIFEST), [ent], extra={"code_stream": "code", "base": "moe_b1"})
    est = GM.estimate(d, probe_windows=2)
    assert est["draws"] == 1 and est["directions"] == 2 and est["gpu_hours"] > 0
    rep_ = GM.run(d, probe_windows=2, device="cpu", model=tiny("moe"), log=None)
    assert rep_["routing_mode"] == "pinned" and len(rep_["draws"]) == 1
    z = np.load(os.path.join(d, GM.WINDOWS))
    assert sorted(set(z["corpus"].tolist())) == ["code", "math"] and len(z["corpus"]) == 4
    eps = rep_["eps"]["german"]
    dr = rep_["draws"][0]
    q = {}
    for c in ("code", "math"):
        m = z["corpus"] == c
        for s in ("A", "B"):
            k = f"german|d1|{s}|{eps:.6g}"
            q[(c, s)] = float(((z[k + "|+1|kl"][m] + z[k + "|-1|kl"][m]) / (2 * eps * eps)).mean())
            assert math.isclose(q[(c, s)], dr["Q"][c][s]["value"], rel_tol=1e-12) and q[(c, s)] > 0
    ip = (math.log(q[("code", "B")]) - math.log(q[("code", "A")])) - (math.log(q[("math", "B")]) - math.log(q[("math", "A")]))
    assert math.isclose(ip, dr["I_probe"], rel_tol=1e-12, abs_tol=1e-12)
    half = EW.split_halves(len(EW.load_result(os.path.join(d, "routing__code.npz"))["index"]), seed=0)
    idx = EW.load_result(os.path.join(d, "routing__code.npz"))["index"][np.flatnonzero(half == 1)[:2]]
    assert np.array_equal(z["window_index"][z["corpus"] == "code"], idx)          # estimation half, first 2 windows
    with open(os.path.join(d, GM.MANIFEST), "a", encoding="utf-8") as fh:        # a changed manifest is refused
        fh.write(" ")
    with pytest.raises(SystemExit):
        GM.run(d, probe_windows=2, device="cpu", model=tiny("moe"), log=None)


# ============================================================================ the design additions
def test_reflect_and_antithetic_random_direction_states():
    m = tiny("moe")
    base = {k: v.detach().clone() for k, v in m.state_dict().items()}
    adapted = {k: v.clone() for k, v in base.items()}
    g = torch.Generator().manual_seed(0)
    bank = next(k for k, v in base.items() if v.ndim == 3 and torch.is_floating_point(v))
    adapted[bank][1] += 0.05 * torch.randn(adapted[bank][1].shape, generator=g)        # one expert slice trained
    mat = next(k for k, v in base.items() if v.ndim == 2 and torch.is_floating_point(v) and k != bank)
    adapted[mat] += 0.05 * torch.randn(adapted[mat].shape, generator=g)
    ref = WC.reflect_state(base, adapted)
    for k in base:
        if k in (bank, mat):
            continue
        assert torch.equal(ref[k], base[k]), k                                          # frozen tensors bitwise
    assert torch.equal(ref[bank][0], base[bank][0]) and torch.equal(ref[bank][2:], base[bank][2:])  # frozen rows bitwise
    assert torch.allclose(ref[bank][1], 2 * base[bank][1] - adapted[bank][1], atol=1e-6)
    assert torch.allclose(ref[mat], 2 * base[mat] - adapted[mat], atol=1e-6)
    back = WC.reflect_state(base, ref)
    assert torch.allclose(back[mat], adapted[mat], atol=1e-6) and torch.equal(back[bank][0], base[bank][0])
    rp = WC.random_direction_state(base, adapted, seed=3)
    rm = WC.random_direction_state(base, adapted, seed=3, sign=-1)
    for k in (bank, mat):
        assert torch.allclose(rp[k] + rm[k], 2 * base[k], atol=1e-6)                   # antithetic pair
        assert not torch.allclose(rp[k], adapted[k], atol=1e-4)
    assert math.isclose(float((rp[mat] - base[mat]).norm()), float((adapted[mat] - base[mat]).norm()), rel_tol=1e-4)
    assert math.isclose(float((rm[mat] - base[mat]).norm()), float((adapted[mat] - base[mat]).norm()), rel_tol=1e-4)
    assert torch.equal(rm[bank][0], base[bank][0])
    with pytest.raises(ValueError):
        WC.random_direction_state(base, adapted, seed=3, sign=2)


def test_dose_loss_weight_scales_gradients_exactly_and_is_inert_unset(monkeypatch):
    from nestor.configs.losses import CausalLMLossConfig
    from nestor.train.losses import CausalLMLoss, Loss

    monkeypatch.delenv("NESTOR_LOSS_WEIGHT", raising=False)
    assert AX.loss_weight() is None and AX.patch_loss_weight() is None
    m = tiny("moe")
    x = torch.as_tensor(np.random.default_rng(0).integers(0, 255, (1, 64))).long()
    params = G0.expert_params(m)

    def grads():
        for p in m.parameters():
            p.grad = None
        out = m(x)
        lo = Loss.from_config(CausalLMLossConfig())(out.pred, x, out.aux)
        lo.total.backward()
        return lo, {k: p.grad.detach().clone() for k, p in params.items() if p.grad is not None}

    from nestor.train.frozen_routers import hold_routing_in_eval_mode
    m.train()
    hold_routing_in_eval_mode(m)                       # deterministic routing in the training forward (as target_gradient)
    l0, g0 = grads()
    monkeypatch.setattr(CausalLMLoss, "forward", CausalLMLoss.forward)                  # restored after the test
    monkeypatch.setenv("NESTOR_LOSS_WEIGHT", "0.37")
    assert AX.patch_loss_weight() is not None and getattr(CausalLMLoss.forward, "_dose_c") == 0.37
    l1, g1 = grads()
    assert math.isclose(float(l1.total), 0.37 * float(l0.total), rel_tol=1e-6)
    assert float(l1.components["cross_entropy"]) == float(l0.components["cross_entropy"])
    # exact up to fp32 rounding: relative Frobenius error (elementwise rtol fails on near-zero entries)
    assert g0 and all(float((g1[k] - 0.37 * g0[k]).norm()) <= 1e-5 * float((0.37 * g0[k]).norm()) for k in g0)
    AX.patch_loss_weight()                                                               # idempotent: wraps the original once
    l2, _ = grads()
    assert math.isclose(float(l2.total), 0.37 * float(l0.total), rel_tol=1e-6)
    monkeypatch.setenv("NESTOR_LOSS_WEIGHT", "-1")
    with pytest.raises(ValueError):
        AX.loss_weight()


def test_half_batch_gradient_snr_matches_definition(tmp_path):
    import grad_snr as GS
    m = tiny("moe")
    path = write_bin(tmp_path / "t.bin", 64 * 16, 3, 0, 120)
    ws = EW.WindowSet("t", path, 64, 8, "first")
    params, rows = G0.expert_params(m), G0.unit_rows(m)
    res = GS.half_batch_snr(m, ws, params, rows, device="cpu", seed=1, train_batch_windows=32, kinds=("moe",))
    assert res["windows"] == 8 and res["half_windows"] == 4 and sorted(sum(res["halves"], [])) == list(range(8))
    hs = [EW.WindowSet("h", path, 64, 4, indices=ws.index[np.array(h)]) for h in res["halves"]]
    g1, g2 = (G0.target_gradient(m, h, params, 4, device="cpu", bf16=True) for h in hs)
    units = [u for u in res["per_unit"]][:3]
    for u in units:
        gb = GS.unit_sq_norms({k: (g1[k].double() + g2[k].double()) / 2 for k in g1}, rows, [u])[u]
        nz = GS.unit_sq_norms({k: g1[k].double() - g2[k].double() for k in g1}, rows, [u])[u] / 4
        r = res["per_unit"][u]
        assert math.isclose(r["noise"], nz, rel_tol=1e-9) and math.isclose(r["signal"], gb - nz, rel_tol=1e-9, abs_tol=1e-15)
        assert math.isclose(r["snr_train"], r["snr"] * 32 / 8, rel_tol=1e-12)
    s = res["summary"]["moe"]
    assert s["units"] == len([u for u in rows if u.startswith("moe:")]) and 0 <= s["share_snr_train_below_1"] <= 1
    dup = EW.WindowSet("d", path, 64, 4, indices=np.zeros(4, dtype=np.int64))            # identical halves: zero noise
    rd = GS.half_batch_snr(m, dup, params, rows, device="cpu", seed=0, kinds=("moe",))
    assert all(v["noise"] == 0 for v in rd["per_unit"].values())
    assert all(v["snr"] == math.inf or math.isnan(v["snr"]) for v in rd["per_unit"].values())


def test_g0_config_glob_checkpoint_and_estimate(tmp_path):
    for i, sub in enumerate(("a", "b")):
        (tmp_path / "outputs" / sub / "checkpoints").mkdir(parents=True)
        f = tmp_path / "outputs" / sub / "checkpoints" / "model_final.pt"
        f.write_bytes(b"x")
        os.utime(f, (2e9 + i, 2e9 + i))
    cfg = {"checkpoint": "glob:" + str(tmp_path / "outputs" / "*" / "checkpoints" / "model_final.pt"), "block": 4096,
           "streams": {"dclm": {"path": "x", "tokens": 4096 * 10}}, "crossover": {"draws": 3, "targets": ["german"]},
           "c2": {"target": "german", "draws": 3},
           "probe": {"targets": ["german", "french"], "probe_windows": 8, "probe_corpora": ["dclm", "code", "math"],
                     "random_seeds": [0], "grad_windows": 16, "pin_routing": "both"}}
    c = G0.load_config(_dump(tmp_path / "cfg.json", cfg))
    assert c["checkpoint"] == os.path.abspath(tmp_path / "outputs" / "b" / "checkpoints" / "model_final.pt")
    est = G0.estimate(c)
    assert est["per_target"]["german"]["directions"] == 13 and est["per_target"]["french"]["directions"] == 7
    assert est["per_target"]["german"]["forwards_per_window"] == 2 * (1 + 13 * 4 + 2)
    with pytest.raises(FileNotFoundError):
        G0.resolve_path("glob:" + str(tmp_path / "nothing" / "*.pt"))


def _dump(path, obj) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return str(path)


# ============================================================================ subset_adapt hardening
def test_guard_resume_rules(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("NESTOR_OUTPUT_ROOT", str(root))
    spec, other = T.MOE_SPEC, {"mode": "train", "select": [{"kind": "moe", "block": 0, "layer": 0, "expert": [1]}]}
    SA.guard_resume(spec)                                                    # fresh root
    _dump(SA._manifest_path(), {"fingerprint": SA.spec_fingerprint(spec)})
    SA.guard_resume(spec)                                                    # same spec + env: resume allowed
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        SA.guard_resume(other)
    monkeypatch.setenv("DCLM_LR_MUON", "0.005")
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        SA.guard_resume(spec)                                                # same spec, other lr
    monkeypatch.delenv("DCLM_LR_MUON")
    with pytest.raises(RuntimeError, match="NESTOR_SUBSET_SPEC is unset"):
        SA.guard_resume(None)
    monkeypatch.setenv(SA.ALLOW_CHANGE_ENV, "1")
    SA.guard_resume(other)                                                   # no checkpoint yet: allowed
    os.remove(SA._manifest_path())
    (root / "x").mkdir()
    (root / "x" / "model_final.pt").write_bytes(b"0")
    with pytest.raises(RuntimeError, match="no subset manifest"):
        SA.guard_resume(spec)                                                # checkpoints, no manifest
    monkeypatch.delenv(SA.ALLOW_CHANGE_ENV)
    with pytest.raises(RuntimeError, match="no subset manifest"):
        SA.guard_resume(spec)
    SA.guard_resume(None)                                                    # a FULL run (no spec) is not guarded


def test_verify_fails_on_unchanged_trainable_and_optimizer_state(tmp_path):
    m = T.build(T.moe_backbone(), seed=0)
    man_path = str(tmp_path / "subset_manifest.json")
    plan = SA.apply_subset(m, T.MOE_SPEC, write_manifest=man_path)
    freeze_routers(m)
    init = str(tmp_path / "init.pt")
    torch.save({"model": {k: v.clone() for k, v in m.state_dict().items()}}, init)
    opt, sch = T.make_opt(m, 3)
    T.train(m, opt, sch, 3)
    final = str(tmp_path / "final.pt")
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict()}, final)
    man = json.load(open(man_path))
    assert man["optimizer_class"] == "MuonAdamW" and len(man["optimizer_param_order"]) == sum(len(g["params"]) for g in opt.param_groups)
    ok = SA.verify_checkpoints(init, final, man_path)
    assert ok["ok"] and ok["optimizer_state"]["checked"] and ok["optimizer_state"]["state_rows_checked"] > 0, ok
    # a trainable tensor left unchanged -> audit fails (unless allow_unchanged)
    ck = torch.load(final, weights_only=False)
    tr = next(r["name"] for r in plan.manifest["params"] if r["status"] == "trainable")
    ck["model"][tr] = torch.load(init, weights_only=False)["model"][tr].clone()
    f2 = str(tmp_path / "final_unchanged.pt")
    torch.save(ck, f2)
    r2 = SA.verify_checkpoints(init, f2, man_path)
    assert not r2["ok"] and tr in r2["trainable_unchanged"]
    assert SA.verify_checkpoints(init, f2, man_path, allow_unchanged=True)["ok"]
    # non-zero optimizer state on frozen rows of a partial bank -> audit fails
    ck = torch.load(final, weights_only=False)
    pos = {n: i for i, n in enumerate(man["optimizer_param_order"])}
    part = next(r for r in plan.manifest["params"] if r["status"] == "partial" and r["name"] in pos)
    st = ck["optimizer"]["state"][pos[part["name"]]]
    fr = next(i for i in range(st["momentum_buffer"].shape[0]) if i not in set(part["trainable_rows"]))
    st["momentum_buffer"][fr] += 1e-3
    f3 = str(tmp_path / "final_tampered.pt")
    torch.save(ck, f3)
    r3 = SA.verify_checkpoints(init, f3, man_path)
    assert not r3["ok"] and any("frozen rows" in v for v in r3["optimizer_state"]["violations"])
    # a frozen tensor held by the optimizer -> violation
    frozen = next(r["name"] for r in plan.manifest["params"] if r["status"] == "frozen")
    man_bad = dict(man, optimizer_param_order=man["optimizer_param_order"] + [frozen])
    r4 = SA.verify_checkpoints(init, final, man_bad)
    assert any("frozen tensor is held" in v for v in r4["optimizer_state"]["violations"])


def test_negative_control_env_leaks_decay_in_process(monkeypatch):
    for negctrl in ("", "unmasked_decay"):
        SA.reset()
        SA.register_optimizer_hooks()
        if negctrl:
            monkeypatch.setenv(SA.NEGCTRL_ENV, negctrl)
        m = T.build(T.moe_backbone(), seed=0)
        plan = SA.apply_subset(m, T.MOE_SPEC)
        freeze_routers(m)
        before = T.snapshot(m)
        opt, sch = T.make_opt(m, 3)
        T.train(m, opt, sch, 3)
        viol, _ = T.audit(plan, before, m)
        assert (len(viol) > 0) == bool(negctrl), viol[:4]
        assert plan.manifest["negative_control"] == (negctrl or None)


# ============================================================================ child pipeline runs
DEV = T.TEST_DEVICE                       # "cuda" in the GPU pipeline audit (hooks_after_B.ps1), else "cpu"
# interim evaluations run inside the training process, which enables TF32 on CUDA (nestor/utils/seed.py:28-31);
# the eval CLI runs plain fp32 -> cross-process CE comparisons need a TF32-sized tolerance on the GPU
XDEV_RTOL = 1e-6 if DEV == "cpu" else 2e-3


def _child_env(over: dict) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", **({"CUDA_VISIBLE_DEVICES": ""} if DEV == "cpu" else {}))
    for k in CLEAN_ENV + ("NESTOR_FREEZE_ROUTERS",):
        env.pop(k, None)
    env.update({k: str(v) for k, v in over.items()})
    return env


def _pipeline_child(args: dict) -> None:
    assert P.DATA_PATH == pathlib.Path(args["data"]) and P.BLOCK_SIZE == 64
    SA.install()
    AX.install()
    if args.get("zero_loss"):                       # exactly-zero gradients through the authors' Loss
        import nestor.train.losses as L

        def zero_wrap(cls):                         # Loss.from_config returns subclasses with their own forward
            f = cls.__dict__.get("forward")
            if f is None:
                return
            def fwd(self, *a, _f=f, **k):
                out = _f(self, *a, **k)
                return L.LossOutput(total=out.total * 0.0, components=out.components)
            cls.forward = fwd

        seen, stack = set(), [L.Loss]
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(c.__subclasses__())
        for c in seen:
            zero_wrap(c)
        print(f"[test] zero loss installed on {len(seen)} loss classes", flush=True)
    from nestor.configs.distributed import DistributedConfig
    from nestor.experiments.registry import get_experiment
    from nestor.pipelines.train import TrainPipeline

    exp = get_experiment(f"tiny_{args['family']}")()
    exp.jobs = [dataclasses.replace(j, distributed=DistributedConfig(enabled=False)) for j in exp.jobs]
    for job in exp.jobs:
        TrainPipeline.from_config(job_config=job, experiment_config=exp).run()


def _run_child(args: dict, *, data: str, steps: int, root: str, suffix: str, init: str | None = None,
               spec: dict | None = None, extra: dict | None = None) -> tuple[str, dict]:
    over = {"DCLM_DATA_PATH": data, "DCLM_BLOCK_SIZE": "64", "DCLM_GLOBAL_BATCH": "2", "DCLM_GRAD_ACCUM": "2",
            "DCLM_MAX_STEPS": str(steps), "DCLM_WARMUP_STEPS": "1", "DCLM_CKPT_FREQ": "100000", "DCLM_LOG_FREQ": "1",
            "NESTOR_OUTPUT_ROOT": root, "DCLM_RUN_SUFFIX": suffix}
    if init:
        over.update({"DCLM_INIT_CHECKPOINT": init, "DCLM_LR_MUON": "0.0025", "DCLM_LR_ADAMW": "7.5e-05",
                     "NESTOR_FREEZE_ROUTERS": "1"})
    if spec is not None:
        over[SA.SPEC_ENV] = json.dumps(spec)
    over.update(extra or {})
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--pipeline-child", json.dumps(dict(args, data=data))],
                       capture_output=True, text=True, env=_child_env(over), cwd=HERE, timeout=1800)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-4000:]
    return log, over


@pytest.fixture(scope="session")
def pretrained(tmp_path_factory):
    cache: dict = {}

    def get(family: str) -> dict:
        if family not in cache:
            d = tmp_path_factory.mktemp(f"pre_{family}")
            data, dom = str(d / "pre.bin"), str(d / "dom.bin")
            T._write_bin(data, 64 * 64, 0)
            T._write_bin(dom, 64 * 64, 1, 500, 1500)
            root = str(d / "out")
            _run_child({"family": family}, data=data, steps=3, root=root, suffix="pre")
            cache[family] = {"init": T._final_ckpt(root, "pre", family), "data": data, "dom": dom}
        return cache[family]

    return get


TINY_ENV = {"DCLM_BLOCK_SIZE": "64", "DCLM_GLOBAL_BATCH": "2", "DCLM_GRAD_ACCUM": "2"}


def _tiny_build_env(monkeypatch):
    """Building a tiny experiment in THIS process: pretrain.BLOCK_SIZE is read at import (4096 here), so set
    it like the child's env did (the authors' budget assert checks batch x accum x block = tokens/step)."""

    monkeypatch.setattr(P, "BLOCK_SIZE", 64)
    for k, v in TINY_ENV.items():
        monkeypatch.setenv(k, v)


def test_pipeline_interim_evals_train_log_no_clip_and_eval_cli(pretrained, tmp_path, monkeypatch):
    pre = pretrained("nestor")
    root, idir, ldir = str(tmp_path / "adapt"), str(tmp_path / "interim"), str(tmp_path / "trainlog")
    interim = {"steps": [2, 4], "out": idir, "block": 64, "routing": True, "full_steps": [4], "full_sets": ["oldgrid"],
               "sets": [{"name": "old", "path": pre["data"], "tokens": 640},
                        {"name": "oldgrid", "path": pre["data"], "tokens": 640, "select": "even", "span": [0.5, 1.0], "head": 3}]}
    log, env_used = _run_child({"family": "nestor"}, data=pre["dom"], steps=4, root=root, suffix="adapt", init=pre["init"],
                               spec=T.NESTOR_SPEC, extra={"NESTOR_INTERIM_EVAL": json.dumps(interim), "NESTOR_INTERIM_FULL": "1",
                                                          "NESTOR_TRAIN_LOG_DIR": ldir, "NESTOR_MAX_GRAD_NORM": "none"})
    assert "max_grad_norm overridden: None" in log and "interim eval at step 2" in log and "interim eval at step 4" in log
    final = T._final_ckpt(root, "adapt", "nestor")
    man = os.path.join(root, "subset_manifest.json")
    audit = SA.verify_checkpoints(pre["init"], final, man)
    assert audit["ok"] and audit["optimizer_state"]["checked"], audit
    recs = [json.loads(x) for x in open(os.path.join(ldir, "train_steps.jsonl"))]
    assert [r["step"] for r in recs] == [1, 2, 3, 4]
    assert all(r["max_norm"] is None and r["clipped"] is False and math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0
               for r in recs)
    disp = [json.loads(x) for x in open(os.path.join(idir, "displacement.jsonl"))]
    assert [d["step"] for d in disp] == [2, 4] and all(d["total_sq"] > 0 for d in disp)
    _tiny_build_env(monkeypatch)
    model, _ = EW.load_model("tiny_nestor", final, "cpu")
    init_sd = torch.load(pre["init"], map_location="cpu", weights_only=False)["model"]
    post = WC.displacement(model, init_sd)
    assert math.isclose(post["total_sq"], disp[-1]["total_sq"], rel_tol=1e-5)
    # the eval CLI on the run: every variant, routing summaries, wd-only from the run env
    out = str(tmp_path / "eval")
    run_env = _dump(tmp_path / "run_env.json", env_used)
    argv = [sys.executable, os.path.join(HERE, "eval_windows.py"), "run", "tiny_nestor", final,
            "--set", f"old={pre['data']}:640", "--set", f"dom={pre['dom']}:640", "--block", "64", "--out", out,
            "--variants", "ckpt,base,pinned,randdir:1,randdir-:1,reflect,revert:pub+priv,wdonly", "--base", pre["init"], "--manifest", man,
            "--run-env", run_env, "--routing", "--device", DEV, "--register", os.path.join(HERE, "tiny_experiments.py")]
    p = subprocess.run(argv, capture_output=True, text=True, env=_child_env(TINY_ENV), cwd=HERE, timeout=1800)
    assert p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    R = {f[:-4]: EW.load_result(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npz")}
    assert {f"{v}__{s}" for v in ("ckpt", "base", "pinned", "randdir-1", "revert-pub+priv", "wdonly")
            for s in ("old", "dom")} <= set(R)
    s4, s2 = EW.load_result(os.path.join(idir, "step004__old.npz")), EW.load_result(os.path.join(idir, "step002__old.npz"))
    assert np.array_equal(s4["hash"], R["ckpt__old"]["hash"][:len(s4["hash"])])
    assert np.allclose(s4["ce_sum"], R["ckpt__old"]["ce_sum"][:len(s4["ce_sum"])], rtol=XDEV_RTOL, atol=1e-6)
    assert not np.allclose(s2["ce_sum"], s4["ce_sum"])
    assert any(k.startswith("flip_") for k in R["pinned__old"]) and R["pinned__old"]["routing"]
    for v in ("base", "randdir-1", "randdir--1", "reflect", "revert-pub+priv", "wdonly"):
        assert not np.array_equal(R[f"{v}__old"]["ce_sum"], R["ckpt__old"]["ce_sum"]), v
    assert not np.array_equal(R["randdir-1__old"]["ce_sum"], R["randdir--1__old"]["ce_sum"])
    # registered interim protocol: the interim set is the head of the evaluation grid used by the eval CLI (--span)
    out2 = str(tmp_path / "eval_grid")
    argv2 = [sys.executable, os.path.join(HERE, "eval_windows.py"), "run", "tiny_nestor", final, "--set",
             f"oldgrid={pre['data']}:640", "--select", "even", "--span", "oldgrid=0.5,1", "--block", "64", "--out", out2,
             "--device", DEV, "--register", os.path.join(HERE, "tiny_experiments.py")]
    p2 = subprocess.run(argv2, capture_output=True, text=True, env=_child_env(TINY_ENV), cwd=HERE, timeout=1800)
    assert p2.returncode == 0, (p2.stdout + p2.stderr)[-4000:]
    G = EW.load_result(os.path.join(out2, "ckpt__oldgrid.npz"))
    i4 = EW.load_result(os.path.join(idir, "step004__oldgrid.npz"))
    assert G["meta"]["set"]["span"] == [0.5, 1.0] and len(i4["index"]) == 3
    assert np.array_equal(i4["index"], G["index"][:3]) and np.array_equal(i4["hash"], G["hash"][:3])
    assert np.allclose(i4["ce_sum"], G["ce_sum"][:3], rtol=XDEV_RTOL, atol=1e-6)
    ntok = len(np.fromfile(pre["data"], dtype=np.uint32))
    assert int(G["index"].min()) * 64 >= ntok // 2
    # NESTOR_INTERIM_FULL: full grid at the full steps, head sliced from the same evaluation
    F4 = EW.load_result(os.path.join(idir, "step004__oldgrid__full.npz"))
    assert np.array_equal(F4["index"], G["index"]) and np.allclose(F4["ce_sum"], G["ce_sum"], rtol=XDEV_RTOL, atol=1e-6)
    assert np.array_equal(F4["index"][:3], i4["index"]) and np.array_equal(F4["ce_sum"][:3], i4["ce_sum"])
    assert not os.path.exists(os.path.join(idir, "step002__oldgrid__full.npz"))
    assert not os.path.exists(os.path.join(idir, "step004__old__full.npz"))
    q = subprocess.run([sys.executable, os.path.join(HERE, "eval_windows.py"), "forget", out, out, "--before-variant", "base",
                        "--after-variant", "ckpt"], capture_output=True, text=True, env=_child_env({}), cwd=HERE, timeout=600)
    ft = json.loads(q.stdout)
    assert set(ft) == {"old", "dom"}
    assert math.isclose(ft["old"]["diff"], EW.paired(R["base__old"], R["ckpt__old"])["diff"], rel_tol=1e-12)


def test_pipeline_sgd_arm_and_negative_control(pretrained, tmp_path):
    pre = pretrained("nestor")
    r1, ldir = str(tmp_path / "sgd"), str(tmp_path / "sgdlog")
    log, _ = _run_child({"family": "nestor"}, data=pre["dom"], steps=3, root=r1, suffix="sgd", init=pre["init"],
                        spec=T.NESTOR_SPEC, extra={"NESTOR_SUBSET_OPTIMIZER": "sgd", "NESTOR_SGD_LR": "0.05",
                                                   "NESTOR_SGD_WD": "0.1", "NESTOR_TRAIN_LOG_DIR": ldir})
    assert "[adapt_extras] SGD arm:" in log
    man = json.load(open(os.path.join(r1, "subset_manifest.json")))
    sg = [g for g in man["optimizer_groups"] if g["method"] == "sgd"]
    assert man["optimizer_class"] == "MuonAdamWSGD" and sg and all(math.isclose(g["initial_lr"], 0.05) for g in sg)
    assert all("sgd" in json.loads(x)["methods"] for x in open(os.path.join(ldir, "train_steps.jsonl")))
    a1 = SA.verify_checkpoints(pre["init"], T._final_ckpt(r1, "sgd", "nestor"), os.path.join(r1, "subset_manifest.json"))
    assert a1["ok"], a1
    r2 = str(tmp_path / "neg")
    _run_child({"family": "nestor"}, data=pre["dom"], steps=3, root=r2, suffix="neg", init=pre["init"],
               spec=T.NESTOR_SPEC, extra={SA.NEGCTRL_ENV: "unmasked_decay"})
    a2 = SA.verify_checkpoints(pre["init"], T._final_ckpt(r2, "neg", "nestor"), os.path.join(r2, "subset_manifest.json"))
    assert not a2["ok"] and a2["frozen_mismatch"] and a2["negative_control"] == "unmasked_decay"


def test_wd_only_state_reproduces_a_zero_gradient_run_bitwise(pretrained, tmp_path, monkeypatch):
    pre = pretrained("nestor")
    _tiny_build_env(monkeypatch)
    root = str(tmp_path / "zl")
    log, env_used = _run_child({"family": "nestor", "zero_loss": True}, data=pre["dom"], steps=3, root=root, suffix="zl",
                               init=pre["init"], spec=T.NESTOR_SPEC)
    assert "[test] zero loss installed on" in log
    final = torch.load(T._final_ckpt(root, "zl", "nestor"), map_location="cpu", weights_only=False)["model"]
    base = torch.load(pre["init"], map_location="cpu", weights_only=False)["model"]

    def builder():
        from nestor.experiments.registry import get_experiment
        from nestor.models import Model

        job = get_experiment("tiny_nestor")().jobs[0]
        return Model.from_config(job.model), job

    st = WC.wd_only_state("tiny_nestor", base, os.path.join(root, "subset_manifest.json"), env_used, model_builder=builder)
    names = sorted(set(st) & set(final))
    assert len(names) == len(final), sorted(set(final) - set(st))[:5]
    mism = [n for n in names if not torch.equal(st[n], final[n])]
    assert not mism, mism[:5]
    assert any(not torch.equal(st[n], base[n]) for n in names)              # decay did act


# ============================================================================ run_queue
STUB_RUN_EXP = r'''
import json, os, sys
a = sys.argv[1:]
kv = dict(zip(a[1::2], a[2::2]))
root = os.environ["NESTOR_OUTPUT_ROOT"]
if os.environ.get("STUB_OOM_MICRO") == kv["--micro"] and os.environ.get("NESTOR_ACTIVATION_CHECKPOINT") == "0":
    print("torch.OutOfMemoryError: CUDA out of memory (stub)")
    sys.exit(1)
os.makedirs(os.path.join(root, "run"), exist_ok=True)
open(os.path.join(root, "run", "model_final.pt"), "wb").write(b"stub")
if os.environ.get("NESTOR_SUBSET_SPEC"):
    json.dump({"spec": json.loads(os.environ["NESTOR_SUBSET_SPEC"]), "negative_control": os.environ.get("NESTOR_SUBSET_NEGCTRL")},
              open(os.path.join(root, "subset_manifest.json"), "w"))
json.dump({"argv": a, "env": {k: os.environ.get(k) for k in ("NESTOR_INTERIM_EVAL", "NESTOR_TRAIN_LOG_DIR",
           "NESTOR_ACTIVATION_CHECKPOINT", "DCLM_BLOCK_SIZE", "MY_VAR", "DCLM_INIT_CHECKPOINT")}},
          open(os.path.join(root, "stub_args.json"), "w"))
'''
STUB_VERIFY = r'''
import json, sys
a = [x for x in sys.argv[1:] if x != "--allow-unchanged"]
kv = dict(zip(a[1::2], a[2::2]))
man = json.load(open(kv["--manifest"]))
ok = not man.get("negative_control")
json.dump({"ok": ok}, open(kv["--out"], "w"))
sys.exit(0 if ok else 1)
'''
STUB_EVAL = r'''
import json, os, sys
open(os.environ["STUB_EVAL_LOG"], "a").write(json.dumps(sys.argv[1:]) + "\n")
'''


def _write_cmd(marker: str, tag: str) -> list:
    return [sys.executable, "-c", f"open(r'{marker}', 'a').write('{tag}\\n')"]


def test_run_queue_gates_skip_resume_retry_and_stop(tmp_path, monkeypatch):
    marker = str(tmp_path / "ran.txt")
    _dump(tmp_path / "g.json", {"x": {"decision": "GO"}})
    (tmp_path / "qgate.py").write_text("def ok(state, params):\n    return {'pass': state['a']['status'] == params['want']}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    jobs = [{"id": "a", "kind": "cmd", "argv": _write_cmd(marker, "a"), "est_h": 0.5},
            {"id": "go", "kind": "gate", "gate": "json", "requires": ["a"],
             "params": {"file": str(tmp_path / "g.json"), "key": "x.decision", "equals": "GO"}},
            {"id": "b", "kind": "cmd", "argv": _write_cmd(marker, "b"), "requires": ["go"]},
            {"id": "stop", "kind": "gate", "gate": "json", "requires": ["a"],
             "params": {"file": str(tmp_path / "g.json"), "key": "x.decision", "equals": "STOP"}},
            {"id": "c", "kind": "cmd", "argv": _write_cmd(marker, "c"), "requires": ["stop"]},
            {"id": "d", "kind": "cmd", "argv": _write_cmd(marker, "d"), "requires_fail": ["stop"]},
            {"id": "e", "kind": "cmd", "argv": [sys.executable, "-c", "import sys; sys.exit(3)"]},
            {"id": "f", "kind": "cmd", "argv": _write_cmd(marker, "f"), "requires": ["e"]},
            {"id": "py", "kind": "gate", "gate": "python", "params": {"func": "qgate:ok", "want": "done"}}]
    q = {"state": str(tmp_path / "res" / "state.json"), "logs": str(tmp_path / "logs"), "jobs": jobs}
    qf = _dump(tmp_path / "q.json", q)
    assert RQ.Queue(qf, dry_run=True).run() == 0
    assert not os.path.exists(marker) and not os.path.exists(q["state"])
    assert RQ.Queue(qf).run() == 0
    st = json.load(open(q["state"]))
    want = {"a": "done", "go": "pass", "b": "done", "stop": "fail", "c": "skipped", "d": "done", "e": "failed",
            "f": "skipped", "py": "pass"}
    assert {k: st[k]["status"] for k in want} == want and st["e"]["rc"] == 3
    assert open(marker).read().split() == ["a", "b", "d"]
    assert RQ.Queue(qf).run() == 0 and open(marker).read().split() == ["a", "b", "d"]     # resume: nothing reruns
    RQ.Queue(qf).run(retry_failed=True)
    assert json.load(open(q["state"]))["e"]["status"] == "failed" and open(marker).read().split() == ["a", "b", "d"]
    q2 = dict(q, state=str(tmp_path / "res2" / "state.json"), jobs=[{"id": "z", "kind": "cmd", "argv": _write_cmd(marker, "z")}])
    qf2 = _dump(tmp_path / "q2.json", q2)
    os.makedirs(os.path.dirname(q2["state"]), exist_ok=True)
    stop = os.path.join(os.path.dirname(q2["state"]), "STOP")
    open(stop, "w").close()
    assert RQ.Queue(qf2).run() == 2 and "z" not in open(marker).read().split()
    os.remove(stop)
    assert RQ.Queue(qf2).run() == 0 and open(marker).read().split()[-1] == "z"
    with pytest.raises(SystemExit, match="duplicate"):
        RQ.Queue(_dump(tmp_path / "q3.json", dict(q, jobs=[jobs[0], jobs[0]])))
    # file_vars: a value written by one job feeds a later job; a gate on a missing file fails (no crash)
    gj, out = str(tmp_path / "grid.json"), str(tmp_path / "fv.txt")
    jobs4 = [{"id": "mk", "kind": "cmd", "argv": [sys.executable, "-c", f"import json; json.dump({{'grid': [0.1, 0.3]}}, open(r'{gj}', 'w'))"]},
             {"id": "use", "kind": "cmd", "requires": ["mk"], "argv": [sys.executable, "-c", f"open(r'{out}', 'w').write('{{lr1}}')"]},
             {"id": "gm", "kind": "gate", "gate": "json", "params": {"file": str(tmp_path / "never.json"), "key": "a", "equals": 1}},
             {"id": "after", "kind": "cmd", "requires": ["gm"], "argv": [sys.executable, "-c", "print(1)"]}]
    q4 = dict(q, state=str(tmp_path / "res4" / "state.json"), jobs=jobs4, file_vars={"lr1": gj + "#grid.1"})
    assert RQ.Queue(_dump(tmp_path / "q4.json", q4)).run() == 0
    s4 = json.load(open(q4["state"]))
    assert open(out).read() == "0.3" and s4["gm"]["status"] == "fail" and "FileNotFoundError" in s4["gm"]["detail"]["error"]
    assert s4["after"]["status"] == "skipped"


def test_run_queue_train_jobs_oom_fallback_audits_and_evals(tmp_path, monkeypatch):
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "run_exp.py").write_text(STUB_RUN_EXP)
    (stub / "subset_adapt.py").write_text(STUB_VERIFY)
    (stub / "eval_windows.py").write_text(STUB_EVAL)
    monkeypatch.setattr(RQ, "HERE", str(stub))
    elog = str(tmp_path / "evals.jsonl")
    monkeypatch.setenv("STUB_EVAL_LOG", elog)
    init = str(tmp_path / "init.pt")
    open(init, "wb").write(b"init")
    spec = {"mode": "train", "select": [{"kind": "moe", "block": 0, "layer": 0, "expert": [1]}]}
    stale = tmp_path / "runs" / "t2" / "m1"
    stale.mkdir(parents=True)
    (stale / "junk.txt").write_text("crashed attempt")
    for i, sub in enumerate(("old", "new")):                    # glob: var resolves to the newest match
        (tmp_path / "ck" / sub).mkdir(parents=True)
        (tmp_path / "ck" / sub / "model_final.pt").write_bytes(b"x")
        os.utime(tmp_path / "ck" / sub / "model_final.pt", (1e9 + i, 1e9 + i))
    contrast = {"draws": [{"spec_A": {"mode": "train", "select": [{"kind": "moe", "block": 0, "layer": 0, "expert": [0]}]}},
                          {"spec_A": {"mode": "train", "select": [{"kind": "moe", "block": 0, "layer": 0, "expert": [7]}]}}]}
    cfile = _dump(tmp_path / "contrast.json", contrast)
    used = str(tmp_path / "used.txt")
    jobs = [{"id": "t1", "kind": "train", "experiment": "x", "steps": 4, "seq": 64, "tokens_per_step": 256, "micro": 2,
             "out_root": str(tmp_path / "runs" / "t1"), "spec": spec, "interim": {"steps": [2], "sets": []},
             "env": {"STUB_OOM_MICRO": "2"},
             "evals": [{"name": "main", "sets": ["old=a.bin:640"], "variants": "ckpt,wdonly", "base": "{init}", "routing": True}]},
            {"id": "t2", "kind": "train", "experiment": "x", "steps": 1, "seq": 64, "tokens_per_step": 64, "micro": 1,
             "out_root": str(tmp_path / "runs" / "t2"), "spec": spec},
            {"id": "neg", "kind": "train", "experiment": "x", "steps": 1, "seq": 64, "tokens_per_step": 64, "micro": 1,
             "out_root": str(tmp_path / "runs" / "neg"), "spec": spec, "negative_control": "unmasked_decay"},
            {"id": "bad", "kind": "train", "experiment": "x", "steps": 1, "seq": 64, "tokens_per_step": 100, "micro": 1,
             "out_root": str(tmp_path / "runs" / "bad")},
            {"id": "t3", "kind": "train", "experiment": "x", "steps": 1, "seq": 64, "tokens_per_step": 64, "micro": 1,
             "out_root": str(tmp_path / "runs" / "t3"), "spec_file": cfile + "#draws.1.spec_A"},
            {"id": "use", "kind": "cmd", "requires": ["t1"],
             "argv": [sys.executable, "-c", f"open(r'{used}', 'w').write(r'{{final_t1}}|{{ckpt}}')"]},
            {"id": "ga", "kind": "gate", "gate": "audit", "params": {"jobs": ["t1", "t2"]}},
            {"id": "gn", "kind": "gate", "gate": "negctrl", "params": {"job": "neg"}},
            {"id": "gbad", "kind": "gate", "gate": "audit", "params": {"jobs": ["t1", "neg"]}}]
    q = {"vars": {"python": sys.executable, "init": init, "ckpt": "glob:" + str(tmp_path / "ck" / "*" / "model_final.pt")},
         "env": {"DCLM_INIT_CHECKPOINT": "{init}", "MY_VAR": "1"},
         "state": str(tmp_path / "res" / "state.json"), "logs": str(tmp_path / "logs"), "jobs": jobs}
    assert RQ.Queue(_dump(tmp_path / "q.json", q)).run() == 0
    st = json.load(open(q["state"]))
    t1 = st["t1"]
    assert t1["status"] == "done" and [(a["micro"], a["ckpt"], a["rc"]) for a in t1["attempts"]] == [(2, False, 1), (2, True, 0)]
    assert t1["root"].endswith("m2c") and t1["audit_ok"] is True
    assert glob.glob(str(tmp_path / "runs" / "t1" / "m2_oom*"))                       # failed attempt kept
    args = json.load(open(os.path.join(t1["root"], "stub_args.json")))
    assert args["argv"][:1] == ["x"] and args["argv"][args["argv"].index("--accum") + 1] == "2"
    assert args["env"]["NESTOR_ACTIVATION_CHECKPOINT"] == "1" and args["env"]["MY_VAR"] == "1"
    assert json.loads(args["env"]["NESTOR_INTERIM_EVAL"])["out"] == os.path.join(t1["root"], "interim")
    renv = json.load(open(os.path.join(t1["root"], "run_env.json")))
    assert renv["DCLM_INIT_CHECKPOINT"] == init and json.loads(renv["NESTOR_SUBSET_SPEC"]) == spec
    ev = [json.loads(x) for x in open(elog)]
    assert len(ev) == 1 and "--routing" in ev[0] and ev[0][ev[0].index("--base") + 1] == init
    assert ev[0][ev[0].index("--run-env") + 1] == os.path.join(t1["root"], "run_env.json") and "--manifest" in ev[0]
    assert st["t2"]["status"] == "done" and glob.glob(str(tmp_path / "runs" / "t2" / "m1_stale*" / "junk.txt"))
    assert st["neg"]["status"] == "done" and st["neg"]["audit_ok"] is False
    assert st["bad"]["status"] == "failed" and "not a multiple" in st["bad"]["error"]
    assert st["ga"]["status"] == "pass" and st["gn"]["status"] == "pass" and st["gbad"]["status"] == "fail"
    r3 = json.load(open(os.path.join(st["t3"]["root"], "run_env.json")))
    assert json.loads(r3["NESTOR_SUBSET_SPEC"]) == contrast["draws"][1]["spec_A"]
    assert open(used).read() == f"{t1['final']}|{os.path.abspath(tmp_path / 'ck' / 'new' / 'model_final.pt')}"
    q["env"].pop("DCLM_INIT_CHECKPOINT")
    q2 = dict(q, state=str(tmp_path / "res2" / "state.json"), jobs=[dict(jobs[1], id="noinit", out_root=str(tmp_path / "runs" / "ni"))])
    RQ.Queue(_dump(tmp_path / "q2.json", q2)).run()
    s2 = json.load(open(q2["state"]))["noinit"]
    assert s2["status"] == "failed" and "DCLM_INIT_CHECKPOINT" in s2["error"]


def test_run_queue_powershell_wrapper_dry_run(tmp_path):
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        pytest.skip("no PowerShell")
    q = {"state": str(tmp_path / "res" / "state.json"), "logs": str(tmp_path / "logs"),
         "jobs": [{"id": "a", "kind": "cmd", "argv": [sys.executable, "-c", "print(1)"], "est_h": 0.25}]}
    qf = _dump(tmp_path / "q.json", q)
    p = subprocess.run([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "run_queue.ps1"),
                        "-Queue", qf, "-DryRun", "-Python", sys.executable], capture_output=True, text=True,
                       cwd=str(tmp_path), timeout=300)
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    assert "DRY RUN" in p.stdout and "estimated 0.25 GPU-h" in p.stdout and not os.path.exists(q["state"])


# ============================================================================ prep_streams
def test_prep_streams_tokenize_and_disjoint_document_sampling(tmp_path):
    import zstandard

    docs = [f"document {i} " + "x" * (5 + (i * 7) % 23) for i in range(60)]
    raw = "\n".join(json.dumps({"text": t}) for t in docs).encode()
    (tmp_path / "a.jsonl.zst").write_bytes(zstandard.ZstdCompressor().compress(raw))
    src = str(tmp_path / "src.bin")
    info = PS.tokenize([str(tmp_path / "a.jsonl.zst")], src, fake_encoder=True)
    from nestor.data.token_documents import document_spans, open_token_bin

    toks, eot = open_token_bin(src)
    assert info["docs"] == 60 and int(len(toks)) == sum(len(d.encode()) + 1 for d in docs) == info["tokens"]
    spans = document_spans(toks, eot)
    src_docs = [bytes(np.asarray(toks[s:e], dtype=np.uint32).tobytes()) for s, e in spans]
    out = str(tmp_path / "out")
    man = PS.sample_parts(src, [("p_eval", 300), ("p_route", 400)], out, seed=1)
    got = {}
    for nm in ("p_eval", "p_route"):
        t2, e2 = open_token_bin(os.path.join(out, nm + ".bin"))
        got[nm] = [bytes(np.asarray(t2[s:e], dtype=np.uint32).tobytes()) for s, e in document_spans(t2, e2)]
        meta = json.load(open(os.path.join(out, nm + ".bin.meta.json")))
        assert meta["num_tokens"] == len(t2) == man["parts"][nm]["tokens"] and int(t2[-1]) == e2
        assert not bool(((np.asarray(t2[1:]) == e2) & (np.asarray(t2[:-1]) == e2)).any())     # no doubled eot
        assert all(d in src_docs for d in got[nm]) and man["parts"][nm]["sha256"] == PS.sha256_file(os.path.join(out, nm + ".bin"))
    assert not set(got["p_eval"]) & set(got["p_route"])
    assert got["p_eval"] != src_docs[:len(got["p_eval"])]                    # shuffled, not file order
    man2 = PS.sample_parts(src, [("p_eval", 300), ("p_route", 400)], str(tmp_path / "out2"), seed=1)
    assert all(man2["parts"][k]["sha256"] == man["parts"][k]["sha256"] for k in man["parts"])
    man3 = PS.sample_parts(src, [("p_eval", 300)], str(tmp_path / "out3"), seed=2)
    assert man3["parts"]["p_eval"]["sha256"] != man["parts"]["p_eval"]["sha256"]
    PS.main(["sample", "--in", src, "--out-dir", str(tmp_path / "cli"), "--part", "a=100", "--seed", "1"])
    assert os.path.exists(tmp_path / "cli" / "a.manifest.json")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pipeline-child":
        _pipeline_child(json.loads(sys.argv[2]))
    else:
        sys.exit(pytest.main([__file__, "-q", "-rA"]))
