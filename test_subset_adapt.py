"""CPU tests for subset_adapt.py and routing_stats.py on tiny models built with the authors' code.

Run (CPU only, <= 6 threads):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 python -m pytest test_subset_adapt.py -q -rA
Tiny models come from the authors' builders (nestor.experiments.pretrain._backbone, the paper's
Nestor read / hybrid pool and the MoE baseline geometry, reference muP, stacked expert banks) with
d=64, vocab 256; the optimizer is the authors' MuonAdamW built by nestor.train.optim.build_optimizer
with the adaptation recipe (Muon 0.0025, AdamW 7.5e-5, weight decay 0.1, betas (0.9, 0.95)),
cosine schedule with warmup, gradient accumulation and clip 1.0 -- the step of Trainer._update_weights.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys

# NESTOR_TEST_DEVICE=cuda: GPU pipeline audit (hooks_after_B.ps1) -- the train-path children then see the GPU, because
# the authors' MoE dispatch calls Triton even for CPU tensors on the host. Default: CPU only.
TEST_DEVICE = os.environ.get("NESTOR_TEST_DEVICE", "cpu").strip() or "cpu"
if TEST_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
FLAGS = {  # the build-time part of run_exp.SPEED_STACK that runs on CPU (layout-defining ones first)
    "NESTOR_STACKED_EXPERT_BANKS": "1", "NESTOR_DENSE_NO_ROUTER": "1", "NESTOR_BATCHED_ROUTERS": "1",
    "NESTOR_COLUMN_SLOT_UPDATES": "1", "NESTOR_DENSE_TAIL_ATTN": "1", "NESTOR_GROUPED_MM_FUSED_GATE": "1",
    "NESTOR_MOE_GROUPED_MM": "1", "NESTOR_FUSED_HEAD_CE": "1", "NESTOR_CE_CHUNK_ROWS": "1024",
    "NESTOR_BF16": "1", "NESTOR_COMPILE": "0", "NESTOR_DDP": "0", "NESTOR_DISABLE_FSDP": "1",
    "NESTOR_MUON_SHARD": "0", "WANDB_MODE": "disabled", "DCLM_NUM_WORKERS": "0",
}
for _k, _v in FLAGS.items():
    os.environ.setdefault(_k, _v)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(6)

import nestor.experiments.pretrain as P  # noqa: E402
import routing_stats as RS  # noqa: E402
import subset_adapt as SA  # noqa: E402
from nestor.configs.adapters import TextAdapterConfig  # noqa: E402
from nestor.configs.losses import CausalLMLossConfig  # noqa: E402
from nestor.configs.optim import MuonAdamWConfig  # noqa: E402
from nestor.configs.reference_mup import ReferenceMupConfig  # noqa: E402
from nestor.configs.sched import CosineAnnealingConfig  # noqa: E402
from nestor.heads import LanguageModelingHeadConfig  # noqa: E402
from nestor.models import Model, SequenceModelConfig  # noqa: E402
from nestor.scaling.muon import MuonAdamW, _orthogonalize_updates, zeropower_via_newtonschulz5_batched  # noqa: E402
from nestor.train.frozen_routers import freeze_routers, hold_routing_in_eval_mode  # noqa: E402
from nestor.train.losses import Loss  # noqa: E402
from nestor.train.optim import build_optimizer, build_scheduler  # noqa: E402

V = 256
TINY = P.Scale("tiny", hidden_size=64, num_layers=4, head_dim=8, mlp_hidden=256, max_steps=8, warmup=1,
               tokens_per_step=256)
RECIPE = dict(lr_muon=0.0025, lr_adamw=7.5e-5, weight_decay=0.1)   # run_continual.py stage B


def nestor_backbone():
    """Nestor geometry in miniature: 3 blocks of 4 columns, k = (2, 3, 4) (dense tail), the
    hybrid pool (4 public top-2 + 4 private per column, switch gates) and the column depth read."""
    return P._backbone(TINY, groups=(4, 4, 4), top_k=(2, 3, 4), heads=(2, 2, 1), layers=(1, 2, 1),
                       bank_experts=1, bank_top_k=1, bank_width=TINY.mlp_hidden, public_experts=4,
                       public_top_k=2, private_experts=4, private_top_k=(2, 1, 1), **P.NESTOR_READ)


def moe_backbone(experts: int = 6, top_k: int = 2, width: int = 96):
    """The MoE baseline geometry in miniature: dense attention, Kimi depth read, a top-k bank."""
    return P._backbone(TINY, groups=(1, 1), top_k=(1, 1), heads=(4, 4), layers=(1, 2), bank_experts=experts,
                       bank_top_k=top_k, bank_width=width, block_residual_mode="attn_res_kimi")


def build(backbone, seed: int = 0):
    torch.manual_seed(seed)
    cfg = SequenceModelConfig(
        adapter_config=TextAdapterConfig(tokenizer="cl100k_base", embed_dim=64, max_length=64, vocab_size=V),
        backbone_config=backbone, head_config=LanguageModelingHeadConfig(vocab_size=V),
        reference_mup=ReferenceMupConfig(matrix_optimizer="muon", num_layers=sum(backbone.block_num_layers),
                                         **P.REFERENCE_MUP))
    return Model.from_config(cfg)


def make_opt(model, steps: int, *, lr_muon=RECIPE["lr_muon"], lr_adamw=RECIPE["lr_adamw"],
             weight_decay=RECIPE["weight_decay"], warmup: int = 1):
    opt = build_optimizer(MuonAdamWConfig(lr_muon=lr_muon, lr_adamw=lr_adamw, weight_decay=weight_decay,
                                          betas=(0.9, 0.95)), model)
    sch = build_scheduler(CosineAnnealingConfig(T_max=steps, warmup_steps=warmup, eta_min=1e-5), opt, steps)
    return opt, sch


def train(model, opt, sch, steps: int, *, accum: int = 2, seed: int = 0, batch: int = 2, seq: int = 32,
          frozen_routers: bool = True, step_model=None):
    """The optimizer step of Trainer._train_step/_update_weights (single process, no AMP on CPU)."""
    loss_fn = Loss.from_config(CausalLMLossConfig())
    g = torch.Generator().manual_seed(seed)
    fwd = step_model if step_model is not None else model
    model.train()
    if frozen_routers:
        hold_routing_in_eval_mode(model)
    losses = []
    for _ in range(steps):
        for _ in range(accum):
            x = torch.randint(0, V, (batch, seq), generator=g)
            out = fwd(x)
            loss = loss_fn(out.pred, x, out.aux).total / accum
            loss.backward()
            losses.append(float(loss.detach()) * accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        sch.step()
    return losses


def snapshot(model):
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    state.update({f"buffer:{n}": b.detach().clone() for n, b in model.named_buffers() if n.endswith("choice_bias")})
    return state


def bits_equal(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(SA._bits(a), SA._bits(b))


def audit(plan, before, model):
    """(frozen_violations, trainable_unchanged) from a plan's manifest and a pre-training snapshot."""
    params = dict(model.named_parameters())
    viol, unchanged = [], []
    for rec in plan.manifest["params"]:
        n = rec["name"]
        a, b = before[n], params[n].detach()
        if rec["status"] in ("frozen", "frozen_at_construction"):
            if not bits_equal(a, b):
                viol.append(n)
        elif rec["status"] == "partial":
            tr = set(rec["trainable_rows"])
            for i in range(a.shape[0]):
                same = bits_equal(a[i], b[i])
                if i in tr and same:
                    unchanged.append(f"{n}[{i}]")
                if i not in tr and not same:
                    viol.append(f"{n}[{i}]")
        elif bits_equal(a, b):
            unchanged.append(n)
    for n, buf in model.named_buffers():
        if n.endswith("choice_bias") and not bits_equal(before[f"buffer:{n}"], buf):
            viol.append(f"buffer:{n}")
    return viol, unchanged


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    SA.reset()
    SA.register_optimizer_hooks()
    monkeypatch.setenv("NESTOR_FREEZE_ROUTERS", "1")
    monkeypatch.delenv(SA.VERIFY_ENV, raising=False)
    yield
    SA.reset()


def prepare(model, spec, *, routers_frozen=True):
    plan = SA.apply_subset(model, spec)
    if routers_frozen:
        freeze_routers(model)          # what TrainPipeline._build_model does after Model.from_config
    return plan


NESTOR_SPEC = {"mode": "train", "select": [
    {"kind": "column", "block": 0, "column": [1, 3]},            # 2 of 4 columns of block 0
    {"kind": "column", "block": 1, "column": 2},                 # 1 column of block 1 (both layers)
    {"kind": "pub", "block": 1, "layer": 1, "expert": [0, 2]},   # 2 of 4 public experts in one layer
    {"kind": "priv", "block": -1, "column": 0, "expert": [1]},    # single private expert in the dense tail
    {"kind": ["embed", "path_gate"]},
]}
MOE_SPEC = {"mode": "train", "select": [
    {"kind": "moe", "block": 0, "layer": 0, "expert": [0, 4]},
    {"kind": "moe", "block": 1, "layer": 1, "expert": [5]},
    {"kind": "attn", "block": 1, "layer": 0},
    {"kind": "norm", "role": "final_norm"},
]}


# ============================================================================ catalog / spec
@pytest.mark.parametrize("kind", ["nestor", "moe", "dense"])
def test_catalog_partitions_every_element(kind):
    bb = {"nestor": nestor_backbone(), "moe": moe_backbone(), "dense": moe_backbone(1, 1, 256)}[kind]
    model = build(bb)
    entries = SA.build_catalog(model)
    total = sum(p.numel() for p in model.parameters())
    covered = sum((e.param[0].numel() * len(e.units)) if e.sliced else e.param.numel() for e in entries)
    assert covered == total
    assert len(entries) == len(list(model.parameters()))
    assert all(len(e.units) == e.param.shape[0] for e in entries if e.sliced)
    model.backbone.register_parameter("mystery", torch.nn.Parameter(torch.zeros(3)))
    with pytest.raises(ValueError, match="unclassified"):
        SA.build_catalog(model)


def test_spec_resolution_maps_columns_to_bank_rows():
    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    recs = {r["name"]: r for r in plan.manifest["params"]}
    q = 4
    assert recs["backbone.blocks.0.layers.0.expert_pool.stacked_private.w_up"]["trainable_rows"] == \
        [1 * q + e for e in range(q)] + [3 * q + e for e in range(q)]
    assert recs["backbone.blocks.1.layers.1.expert_pool.stacked_private.w_down"]["trainable_rows"] == list(range(8, 12))
    assert recs["backbone.blocks.1.layers.1.expert_pool.stacked_public.w_gate"]["trainable_rows"] == [0, 2]
    assert recs["backbone.blocks.2.layers.0.expert_pool.stacked_private.w_up"]["trainable_rows"] == [1]
    assert recs["backbone.blocks.0.layers.0.attention_groups.1.q_proj.weight"]["status"] == "trainable"
    assert recs["backbone.blocks.0.layers.0.attention_groups.0.q_proj.weight"]["status"] == "frozen"
    assert recs["adapter.embed.weight"]["status"] == "trainable"
    assert recs["head.projection.layers.0.weight"]["status"] == "frozen"
    assert all(r["status"] == "frozen" for n, r in recs.items() if ".group_router." in n or "_router" in n)
    assert plan.manifest["geometry"]["dense_tail_block"] == 2


def test_matched_random_spec_matches_size_and_applies():
    model = build(nestor_backbone())
    target = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("adapter.embed", "head.")))
    spec = SA.matched_random_spec(model, "column", target, seed=3, mode="freeze")
    largest = max(e.param[0].numel() * 8 for e in SA.build_catalog(model) if e.sliced)   # generous bound
    assert abs(spec["_provenance"]["achieved_elements"] - target) < largest
    plan = SA.apply_subset(build(nestor_backbone()), spec)
    frozen_cols = plan.manifest["elements"]["by_kind"]
    frozen = frozen_cols["attn"].get("frozen", 0) + frozen_cols["priv"].get("frozen", 0)
    assert frozen == spec["_provenance"]["achieved_elements"]
    assert all(f["block"] != 2 for f in spec["select"])          # dense tail excluded by default
    assert SA.matched_random_spec(model, "column", target, seed=3) == spec   # deterministic


def test_spec_guards(monkeypatch):
    with pytest.raises(ValueError, match="match no unit"):
        SA.apply_subset(build(nestor_backbone()), {"mode": "train", "select": [{"kind": "pub", "block": 7}]})
    with pytest.raises(ValueError, match="NESTOR_FREEZE_ROUTERS=1 freezes"):
        SA.apply_subset(build(nestor_backbone()), {"mode": "train", "select": [{"kind": "router"}]})
    monkeypatch.setenv("NESTOR_FREEZE_ROUTERS", "0")
    with pytest.raises(ValueError, match="set NESTOR_FREEZE_ROUTERS=1"):
        SA.apply_subset(build(nestor_backbone()), {"mode": "train", "select": [{"kind": "embed"}]})
    with pytest.raises(ValueError, match="unknown filter keys"):
        SA.load_spec('{"mode": "train", "select": [{"kind": "embed", "colum": 1}]}')
    # dense tail macro selects only the last block; routers stay frozen under the env flag
    monkeypatch.setenv("NESTOR_FREEZE_ROUTERS", "1")
    plan = SA.apply_subset(build(nestor_backbone()), {"mode": "train", "select": [{"kind": "*", "dense_tail": True}]})
    assert all(k.split(":")[1] == "b2" for k in plan.manifest["units_trainable"])
    assert not any(k.startswith("router") for k in plan.manifest["units_trainable"])


# ============================================================================ exactness (i) + (ii)
@pytest.mark.parametrize("family,spec", [("nestor", NESTOR_SPEC), ("moe", MOE_SPEC)])
def test_frozen_bit_identical_and_selected_change(family, spec):
    model = build(nestor_backbone() if family == "nestor" else moe_backbone())
    plan = prepare(model, spec)
    before = snapshot(model)
    opt, sch = make_opt(model, steps=6)
    held = {id(p) for g in opt.param_groups for p in g["params"]}
    frozen_names = [r["name"] for r in plan.manifest["params"] if r["status"].startswith("frozen")]
    named = dict(model.named_parameters())
    assert not any(id(named[n]) in held for n in frozen_names), "frozen tensors must not reach the optimizer"
    losses = train(model, opt, sch, steps=6)
    assert all(math.isfinite(x) for x in losses)
    viol, unchanged = audit(plan, before, model)
    assert viol == [], f"frozen parameters/rows changed: {viol[:5]}"
    assert unchanged == [], f"trainable parameters/rows did not change: {unchanged[:5]}"
    # optimizer state of the frozen rows is exactly zero, and the hooks did run
    for p in SA._PARTIAL.values():
        st = opt.state.get(p, {})
        for k in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            if k in st:
                assert bool((st[k][p._subset_frozen_rows] == 0).all())
    assert SA.STATS["pre_steps"] == 6 and SA.STATS["masked_decay_groups"] > 0
    assert all(g["weight_decay"] in (0.0, RECIPE["weight_decay"]) for g in opt.param_groups)
    assert any(g["weight_decay"] == RECIPE["weight_decay"] for g in opt.param_groups), "wd must be restored"


def test_mode_freeze_shared_groups_h1_style():
    """H1 arm: freeze every shared component (embeddings, head, norms, depth reads, path gates,
    public experts, the dense tail; routers by the env flag) and train only the routed columns."""
    spec = {"mode": "freeze", "select": [{"kind": ["embed", "head", "norm", "depth_read", "path_gate", "pub"]},
                                        {"kind": "*", "dense_tail": True}]}
    model = build(nestor_backbone())
    plan = prepare(model, spec)
    trainable_kinds = {k.split(":")[0] for k in plan.manifest["units_trainable"]}
    assert trainable_kinds == {"attn", "priv"}
    before = snapshot(model)
    opt, sch = make_opt(model, steps=5)
    train(model, opt, sch, steps=5)
    viol, unchanged = audit(plan, before, model)
    assert viol == [] and unchanged == []


# ============================================================================ pitfalls (negative controls)
def test_negative_control_weight_decay_leaks_without_optimizer_hooks():
    """Gradient masks alone are NOT enough: decoupled weight decay multiplies the whole bank."""
    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    SA.remove_optimizer_hooks()                    # keep only the gradient masks
    try:
        before = snapshot(model)
        opt, sch = make_opt(model, steps=4)
        train(model, opt, sch, steps=4)
        viol, _ = audit(plan, before, model)
        assert any("stacked_" in v for v in viol), "expected weight decay to move frozen bank rows"
        assert not any("stacked_" not in v for v in viol), "whole frozen tensors are safe (not in the optimizer)"
    finally:
        SA.register_optimizer_hooks()


def _bank_opt(w, b, *, wd, lr_m=0.02, lr_a=1e-3):
    return MuonAdamW([{"params": [w], "method": "muon", "lr": lr_m, "weight_decay": wd, "momentum": 0.95,
                       "nesterov": True, "ns_steps": 5},
                      {"params": [b], "method": "adamw", "lr": lr_a, "betas": (0.9, 0.95), "eps": 1e-8,
                       "weight_decay": wd}], lr=lr_a)


def test_negative_control_loaded_momentum_leaks_unless_zeroed():
    """Zero gradients do not stop a NON-zero momentum (e.g. an optimizer state loaded from a
    checkpoint) from moving a frozen slice; the pre-step hook zeroes it."""
    frozen = [False, True, False, True, True]
    for hooks_on in (False, True):
        SA.reset()
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(5, 12, 7))
        b = torch.nn.Parameter(torch.randn(5, 7))
        SA.mark_partial(w, frozen, "w")
        SA.mark_partial(b, frozen, "b")
        opt = _bank_opt(w, b, wd=0.0)
        opt.state[w]["momentum_buffer"] = torch.randn_like(w)
        opt.state[b].update(step=3, exp_avg=torch.randn_like(b), exp_avg_sq=torch.rand_like(b))
        w0, b0 = w.detach().clone(), b.detach().clone()
        if not hooks_on:
            SA.remove_optimizer_hooks()
        try:
            for _ in range(3):
                (w.sum() * 0 + (w ** 2).sum() + (b ** 3).sum()).backward()
                opt.step(); opt.zero_grad()
        finally:
            SA.register_optimizer_hooks()
        fr = torch.tensor(frozen)
        same = bits_equal(w0[fr], w.detach()[fr]) and bits_equal(b0[fr], b.detach()[fr])
        assert same == hooks_on


def test_trainer_zero_fill_of_missing_grads_trainer_py_905():
    """The authors' Trainer._update_weights with _fill_missing_grads=True (its FSDP2 branch):
    trainer.py:905 assigns ``p.grad = torch.zeros_like(p)`` to EVERY parameter whose grad is None,
    bypassing autograd and therefore the gradient hooks -- the frozen tensors (requires_grad=False)
    and, from step 2 on, a row-masked bank whose grad we drop to None ("no token reached the bank").
    Frozen tensors stay bit-identical (MuonAdamW does not hold them), the frozen rows of the
    zero-filled bank stay bit-identical (masked decay + zeroed state; momentum is the only update
    source of its trainable rows then), and every trainable tensor/row still changes."""
    import types

    from nestor.train.trainer import Trainer

    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    before = snapshot(model)
    opt, sch = make_opt(model, steps=5)
    stub = types.SimpleNamespace(model=model, optimizer=opt, scheduler=sch, scaler=None,
                                 config=types.SimpleNamespace(max_grad_norm=1.0), _fill_missing_grads=True,
                                 _needs_detached_updates=False, global_step=0, logger=None)
    stub._update_ema = lambda: None
    seen = []
    handle = opt.register_step_pre_hook(
        lambda o, a, k: seen.append((sum(p.grad is None for p in model.parameters()),
                                     sum(p.grad is not None and not p.requires_grad for p in model.parameters()))))
    loss_fn = Loss.from_config(CausalLMLossConfig())
    g = torch.Generator().manual_seed(0)
    model.train()
    hold_routing_in_eval_mode(model)
    bank = next(p for p in SA._PARTIAL.values() if p._subset_name.endswith("stacked_private.w_up"))
    none_before = []
    try:
        for step in range(5):
            x = torch.randint(0, V, (2, 32), generator=g)
            out = model(x)
            loss_fn(out.pred, x, out.aux).total.backward()
            if step >= 1:
                bank.grad = None
            none_before.append(sum(p.grad is None for p in model.parameters()))
            Trainer._update_weights(stub, [])      # the authors' code, lines 895-958
    finally:
        handle.remove()
    n_frozen = sum(not p.requires_grad for p in model.parameters())
    # step 1: the frozen tensors have no grad; line 905 zero-fills them. optimizer.zero_grad() only
    # clears the parameters the optimizer holds, so from step 2 on the frozen tensors carry a stale
    # ZERO grad and only the dropped bank is None before line 905 runs again.
    assert none_before == [n_frozen] + [1] * 4, (none_before, n_frozen)
    assert all(none == 0 and filled == n_frozen for none, filled in seen), seen   # line 905 ran
    assert all(float(p.grad.abs().sum()) == 0.0 for p in model.parameters() if not p.requires_grad)
    viol, unchanged = audit(plan, before, model)
    assert viol == [], viol[:5]
    assert unchanged == [], unchanged[:5]


def test_partial_bank_step_matches_unmasked_step_bitwise():
    """With identical gradients on the trainable rows, a row-masked bank follows the authors'
    unmasked MuonAdamW trajectory BIT-FOR-BIT on those rows (per-slice Newton-Schulz, same decoupled
    decay), while its frozen rows stay bit-identical."""
    torch.manual_seed(1)
    frozen = torch.tensor([True, False, False, True, False, True])
    wA = torch.nn.Parameter(torch.randn(6, 16, 9)); bA = torch.nn.Parameter(torch.randn(6, 9))
    wB = torch.nn.Parameter(wA.detach().clone()); bB = torch.nn.Parameter(bA.detach().clone())
    SA.mark_partial(wB, frozen.tolist(), "wB")
    SA.mark_partial(bB, frozen.tolist(), "bB")
    optA, optB = _bank_opt(wA, bA, wd=0.1), _bank_opt(wB, bB, wd=0.1)
    w0, b0 = wB.detach().clone(), bB.detach().clone()
    g = torch.Generator().manual_seed(2)
    for step in range(6):
        gw, gb = torch.randn(wA.shape, generator=g), torch.randn(bA.shape, generator=g)
        wA.grad, bA.grad = gw.clone(), gb.clone()
        wB.grad, bB.grad = gw.clone(), gb.clone()   # the guard must zero the frozen rows itself
        for o in (optA, optB):
            for grp in o.param_groups:
                grp["lr"] *= 0.9                   # a moving schedule
        optA.step(); optB.step()
    tr = ~frozen
    assert bits_equal(wA.detach()[tr], wB.detach()[tr])
    assert bits_equal(bA.detach()[tr], bB.detach()[tr])
    assert bits_equal(w0[frozen], wB.detach()[frozen]) and bits_equal(b0[frozen], bB.detach()[frozen])
    assert not bits_equal(w0[tr], wB.detach()[tr])


def test_muon_orthogonalizes_stacked_banks_per_slice():
    """muon.py: a stacked bank [E, m, n] joins its shape bucket as E independent slices
    (per-slice Frobenius normalization, batched matmuls) -- zero slices give exact zeros and the
    other slices are bit-identical to orthogonalizing them alone."""
    torch.manual_seed(3)
    d = torch.randn(7, 24, 40)
    d[[0, 4]] = 0
    other = torch.randn(24, 40)
    up = _orthogonalize_updates([d, other], 5)
    assert bool((up[0][[0, 4]] == 0).all())
    keep = [1, 2, 3, 5, 6]
    assert bits_equal(up[0][keep], zeropower_via_newtonschulz5_batched(d[keep], 5))


# ============================================================================ robustness of the hooks
def test_exact_under_activation_checkpointing(monkeypatch):
    for k in ("NESTOR_ACTIVATION_CHECKPOINT", "NESTOR_ACTIVATION_CHECKPOINT_SPLIT", "NESTOR_ACTIVATION_CHECKPOINT_LEAN"):
        monkeypatch.setenv(k, "1")
    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    before = snapshot(model)
    opt, sch = make_opt(model, steps=4)
    train(model, opt, sch, steps=4)
    viol, unchanged = audit(plan, before, model)
    assert viol == [] and unchanged == []


def test_exact_under_torch_compile_aot_eager(monkeypatch):
    """AOTAutograd (the compile path minus inductor codegen, which needs a C++ toolchain): the
    gradient hooks still fire (the VERIFY guard finds no nonzero frozen-row gradient)."""
    monkeypatch.setenv(SA.VERIFY_ENV, "1")
    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    before = snapshot(model)
    opt, sch = make_opt(model, steps=3)
    compiled = torch.compile(model, backend="aot_eager")
    train(model, opt, sch, steps=3, step_model=compiled)
    viol, unchanged = audit(plan, before, model)
    assert viol == [] and unchanged == []
    assert SA.STATS["guard_nonzero_grad_rows"] == 0 and SA.STATS["verified_tensors"] > 0


@pytest.mark.skipif(os.environ.get("NESTOR_TEST_INDUCTOR") != "1",
                    reason="needs a C++ toolchain (dot-source msvc_env.ps1, set NESTOR_TEST_INDUCTOR=1)")
def test_exact_under_torch_compile_inductor(monkeypatch):
    """Same as the aot_eager test with the inductor backend the GPU runs use (CPU codegen here)."""
    monkeypatch.setenv(SA.VERIFY_ENV, "1")
    model = build(nestor_backbone())
    plan = prepare(model, NESTOR_SPEC)
    before = snapshot(model)
    opt, sch = make_opt(model, steps=3)
    compiled = torch.compile(model)
    train(model, opt, sch, steps=3, step_model=compiled)
    viol, unchanged = audit(plan, before, model)
    assert viol == [] and unchanged == []
    assert SA.STATS["guard_nonzero_grad_rows"] == 0 and SA.STATS["verified_tensors"] > 0


def test_verify_mode_and_offline_checkpoint_audit(tmp_path, monkeypatch):
    monkeypatch.setenv(SA.VERIFY_ENV, "1")
    model = build(moe_backbone())
    plan = prepare(model, MOE_SPEC)
    plan.manifest["env"]["NESTOR_FREEZE_ROUTERS"] = "1"
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps(plan.manifest))
    torch.save({"model": model.state_dict(), "step": 0}, tmp_path / "init.pt")
    opt, sch = make_opt(model, steps=4)
    train(model, opt, sch, steps=4)
    assert SA.STATS["guard_nonzero_grad_rows"] == 0 and SA.STATS["verified_tensors"] >= 4
    torch.save({"model": model.state_dict(), "step": 4}, tmp_path / "final.pt")
    res = SA.verify_checkpoints(str(tmp_path / "init.pt"), str(tmp_path / "final.pt"), str(man))
    assert res["ok"] and res["frozen_rows_checked"] > 0 and not res["trainable_unchanged"]
    tampered = model.state_dict()
    tampered["backbone.blocks.0.layers.0.stacked_moe_bank.w_up"][1, 0, 0] += 1e-6   # a frozen row
    torch.save({"model": tampered}, tmp_path / "bad.pt")
    bad = SA.verify_checkpoints(str(tmp_path / "init.pt"), str(tmp_path / "bad.pt"), str(man))
    assert not bad["ok"] and any("rows [1]" in m for m in bad["frozen_mismatch"])


def test_install_is_a_noop_without_env(monkeypatch):
    monkeypatch.delenv(SA.SPEC_ENV, raising=False)
    from nestor.models import base as mb
    orig = mb.Model.from_config
    assert SA.install() is False and mb.Model.from_config is orig


# ============================================================================ end-to-end pipeline
def _write_bin(path, n_tokens, seed, lo=0, hi=1000):
    rng = np.random.default_rng(seed)
    rng.integers(lo, hi, n_tokens, dtype=np.uint32).tofile(path)
    with open(f"{path}.meta.json", "w") as fh:
        json.dump({"token_dtype": "uint32", "num_tokens": n_tokens, "eot_token": 100257}, fh)


def _pipeline_child(args: dict) -> None:
    """Run the authors' TrainPipeline on CPU for a tiny registered experiment (child process;
    its DCLM_* / NESTOR_* environment was set by the parent BEFORE nestor was imported, as
    run_continual.py does for run_exp.py)."""
    assert P.DATA_PATH == __import__("pathlib").Path(args["data"]) and P.BLOCK_SIZE == 64
    SA.install()                       # what run_exp.py does right after compat.install()
    from nestor.configs.distributed import DistributedConfig
    from nestor.experiments.registry import get_experiment, register_experiment
    from nestor.pipelines.train import TrainPipeline
    P.BATCH_SHAPE[(args["family"], "tiny")] = (2, 2)

    def tiny_exp():
        bb = nestor_backbone() if args["family"] == "nestor" else moe_backbone()
        return P._experiment(args["family"], TINY, bb)
    tiny_exp.__name__ = f"tiny_{args['family']}"
    try:
        register_experiment(tiny_exp)
    except ValueError:
        pass
    exp = get_experiment(f"tiny_{args['family']}")()
    exp.jobs = [dataclasses.replace(j, distributed=DistributedConfig(enabled=False)) for j in exp.jobs]
    for job in exp.jobs:
        TrainPipeline.from_config(job_config=job, experiment_config=exp).run()


def _run_child(args: dict) -> str:
    env = dict(os.environ, OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", **({"CUDA_VISIBLE_DEVICES": ""} if TEST_DEVICE == "cpu" else {}))
    for k in (SA.SPEC_ENV, "NESTOR_FREEZE_ROUTERS", "DCLM_INIT_CHECKPOINT", SA.VERIFY_ENV, SA.MANIFEST_ENV):
        env.pop(k, None)
    env.update({"DCLM_DATA_PATH": args["data"], "DCLM_BLOCK_SIZE": "64", "DCLM_GLOBAL_BATCH": "2",
                "DCLM_GRAD_ACCUM": "2", "DCLM_MAX_STEPS": str(args["steps"]), "DCLM_WARMUP_STEPS": "1",
                "DCLM_CKPT_FREQ": "100000", "DCLM_LOG_FREQ": "1", "NESTOR_OUTPUT_ROOT": args["root"],
                "DCLM_RUN_SUFFIX": args["suffix"]})
    if args.get("init"):   # run_continual.py stage-B recipe
        env.update({"DCLM_INIT_CHECKPOINT": args["init"], "DCLM_LR_MUON": "0.0025",
                    "DCLM_LR_ADAMW": "7.5e-05", "NESTOR_FREEZE_ROUTERS": "1"})
    if args.get("spec"):
        env[SA.SPEC_ENV] = args["spec"]
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--pipeline-child", json.dumps(args)],
                       capture_output=True, text=True, env=env, cwd=HERE, timeout=1200)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-3000:]
    return log


def _final_ckpt(root, suffix, family):
    import glob
    hits = sorted(glob.glob(os.path.join(root, "**", "model_final.pt"), recursive=True))
    assert len(hits) == 1, f"expected one model_final.pt under {root}, got {hits}"
    return hits[0]


@pytest.mark.parametrize("family,spec", [("nestor", NESTOR_SPEC), ("moe", MOE_SPEC)])
def test_pipeline_end_to_end_init_checkpoint_subset_run(tmp_path, family, spec):
    """run_exp-style: pretrain 3 steps, then adapt from DCLM_INIT_CHECKPOINT with
    NESTOR_FREEZE_ROUTERS=1 and NESTOR_SUBSET_SPEC through the authors' TrainPipeline; audit the
    two checkpoints with the manifest the hook wrote."""
    data, data2 = str(tmp_path / "pre.bin"), str(tmp_path / "dom.bin")
    _write_bin(data, 64 * 64, 0)
    _write_bin(data2, 64 * 64, 1, 500, 1500)
    root = str(tmp_path / "out")
    _run_child({"family": family, "data": data, "steps": 3, "root": root, "suffix": "pre"})
    init = _final_ckpt(root, "pre", family)
    root2 = str(tmp_path / "out_adapt")
    log = _run_child({"family": family, "data": data2, "steps": 4, "root": root2, "suffix": "adapt",
                      "init": init, "spec": json.dumps(spec)})
    assert "[subset] installed" in log and "[subset] mode=train" in log
    final = _final_ckpt(root2, "adapt", family)
    res = SA.verify_checkpoints(init, final, os.path.join(root2, "subset_manifest.json"))
    assert res["ok"], res
    assert res["frozen_tensors_checked"] > 0 and res["frozen_rows_checked"] > 0 and res["buffers_checked"] > 0
    assert res["trainable_unchanged"] == [], res["trainable_unchanged"][:5]


# ============================================================================ routing statistics (iii)
def test_mutual_information_estimator_analytic():
    assert RS.mutual_information_bits(np.array([[50, 50], [50, 50]])) == pytest.approx(0.0, abs=1e-12)
    assert RS.mutual_information_bits(np.array([[100, 0], [0, 100]])) == pytest.approx(1.0, abs=1e-12)
    t = np.array([[30.0, 10.0], [10.0, 50.0]])
    p = t / t.sum()
    ref = sum(p[i, j] * math.log2(p[i, j] / (p[i].sum() * p[:, j].sum())) for i in range(2) for j in range(2))
    assert RS.mutual_information_bits(t) == pytest.approx(ref, rel=1e-12)
    five = np.eye(5) * 1000          # five equally sized corpora, deterministic routing: H = log2 5
    assert RS.mutual_information_bits(five) == pytest.approx(math.log2(5), abs=1e-12)


def _fake_aux(topk, A):
    B, S, k = topk.shape
    gates = torch.zeros(B, S, A).scatter_(-1, topk, torch.full((B, S, k), 1.0 / k))
    rec = {"scores": gates, "probs": gates, "gates": gates, "topk_indices": topk, "active_mask": gates > 0,
           "valid_token_mask": torch.ones(B, S, dtype=torch.bool)}
    return {"blocks": [{"layers": [{"expert_router": rec}], "valid_token_mask": rec["valid_token_mask"]}]}


def test_random_router_has_near_zero_mi_and_sums_are_exact():
    """A router whose choice ignores the corpus: MI is at the bias level and inside the
    window-permutation null; counts sum to k x tokens; renormalized gate mass sums to tokens."""
    A, k, S = 8, 2, 256
    g = torch.Generator().manual_seed(0)
    acc = RS.RoutingAccumulator(["c0", "c1", "c2"])
    for lab in range(3):
        for _ in range(40):
            topk = torch.rand(1, S, A, generator=g).topk(k, dim=-1).indices
            acc.add(lab, _fake_aux(topk, A))
    res = acc.summarize(null_perms=200, seed=1)
    r = res["routers"]["b0.l0.expert"]
    n_tok = 40 * S
    for lab in ("c0", "c1", "c2"):
        assert sum(r["counts"][lab]) == k * n_tok and r["tokens"][lab] == n_tok
        assert sum(r["gate_mass"][lab]) == pytest.approx(n_tok, rel=1e-9)
        assert sum(r["usage_freq"][lab]) == pytest.approx(k, rel=1e-12)
    assert r["mi_bits"] < 5 * r["mi_bias_bits"]
    assert r["mi_bits"] <= r["mi_null_p95_bits"] * 1.5
    # a corpus-determined router (choice = label) carries log2(3) bits
    acc2 = RS.RoutingAccumulator(["c0", "c1", "c2"])
    for lab in range(3):
        acc2.add(lab, _fake_aux(torch.tensor([[[2 * lab, 2 * lab + 1]] * S]), A))
    assert acc2.summarize()["routers"]["b0.l0.expert"]["mi_bits"] == pytest.approx(math.log2(3), abs=1e-9)


def _acc_on_model(model, corpora, save=False):
    acc = RS.RoutingAccumulator(list(corpora), save_assignments=save)
    hs = RS.install_path_gate_hooks(model, acc)
    model.eval()
    with torch.no_grad():
        for li, wins in enumerate(corpora.values()):
            for w in wins:
                acc.add(li, model(w.view(1, -1)).aux)
    for h in hs:
        h.remove()
    return acc


def test_routing_invariants_on_tiny_nestor_and_moe():
    g = torch.Generator().manual_seed(4)
    corpora = {"A": [torch.randint(0, V, (64,), generator=g) for _ in range(12)],
               "B": [torch.randint(0, V, (64,), generator=g) for _ in range(12)]}
    res = _acc_on_model(build(nestor_backbone()), corpora).summarize(null_perms=50)
    R = res["routers"]
    n_tok = 12 * 64
    for b, (k, C) in enumerate([(2, 4), (3, 4), (4, 4)]):
        col = R[f"b{b}.column"]
        for lab in "AB":
            assert sum(col["counts"][lab]) == k * n_tok and col["tokens"][lab] == n_tok
            assert sum(col["gate_mass"][lab]) == pytest.approx(n_tok, rel=1e-6)   # renorm gates
    for key, r in R.items():
        if r["family"] == "private":
            col = R[f"b{r['block']}.column"]
            for lab in "AB":
                # every token is routed inside each of its k columns: member counts == column counts
                assert r["column_member_counts"][lab] == pytest.approx(col["counts"][lab])
                assert sum(r["counts"][lab]) == r["k"] * sum(col["counts"][lab])
                assert sum(r["gate_mass"][lab]) < sum(r["column_member_counts"][lab])  # switch gates < 1
                assert 0 < sum(r["eff_mass"][lab]) < sum(r["gate_mass"][lab])
            assert r["mi_pair_bits"] == pytest.approx(r["mi_column_bits"] + r["mi_expert_given_column_bits"])
        if r["family"] == "public":
            for lab in "AB":
                assert sum(r["counts"][lab]) == 2 * n_tok
                assert sum(r["gate_mass"][lab]) == pytest.approx(n_tok, rel=1e-6)
        if r["family"] == "path_gate":
            assert 0 < r["alpha_public_mean"]["A"] < 1
    # same-distribution corpora: every MI is at the noise level of its window-permutation null
    for key, r in R.items():
        if "mi_bits" in r and r["family"] != "path_gate" and r["k"] < r["choices"]:
            assert r["mi_bits"] <= max(2 * r["mi_null_p95_bits"], 5 * r["mi_bias_bits"]), key
    moe = _acc_on_model(build(moe_backbone()), corpora).summarize()
    e = moe["routers"]["b0.l0.expert"]
    assert sum(e["counts"]["A"]) == 2 * n_tok and moe["summary"]["expert"]["n"] == 3


def test_routing_detects_corpus_dependence_and_assignment_changes():
    g = torch.Generator().manual_seed(5)
    disjoint = {"lo": [torch.randint(0, V // 2, (64,), generator=g) for _ in range(16)],
                "hi": [torch.randint(V // 2, V, (64,), generator=g) for _ in range(16)]}
    m1, m2 = build(nestor_backbone(), seed=0), build(nestor_backbone(), seed=7)
    acc = _acc_on_model(m1, disjoint, save=True)
    res = acc.summarize(null_perms=50)
    pub = [r for r in res["routers"].values() if r["family"] == "public"]
    assert any(r["mi_bits"] > r["mi_null_p95_bits"] for r in pub)
    a1 = acc.assignments()
    a1b = _acc_on_model(m1, disjoint, save=True).assignments()
    a2 = _acc_on_model(m2, disjoint, save=True).assignments()
    assert max(RS.compare_assignments(a1, a1b).values()) == 0.0
    assert np.mean(list(RS.compare_assignments(a1, a2).values())) > 0.2
    assert a1["b0.column|lo"].dtype == np.uint8 and a1["b0.l0.private|lo"].dtype == np.uint16   # 4 cols, 4x4 pairs
    assert a1["b0.column|lo"].shape == (16 * 64,)
    big = RS.RoutingAccumulator(["x"], save_assignments=True)
    big.add(0, _fake_aux(torch.rand(1, 8, 80).topk(2, dim=-1).indices, 80))
    with pytest.raises(NotImplementedError):
        big.assignments()


def test_overlap_and_select_round_trip():
    """routing_stats.select_spec -> subset_adapt spec; overlap = routing mass through it."""
    g = torch.Generator().manual_seed(6)
    corpora = {"old": [torch.randint(0, V, (64,), generator=g) for _ in range(8)]}
    model = build(moe_backbone())
    stats = _acc_on_model(model, corpora).summarize()
    spec = RS.select_spec(stats, "old", "expert", k=2, order="least")
    plan = SA.apply_subset(build(moe_backbone()), spec)
    ov = RS.routing_mass_through(stats, "old", set(plan.manifest["units_trainable"]))
    for key, v in ov["per_router"].items():
        if v["family"] == "expert":
            r = stats["routers"][key]
            least2 = sorted(r["counts"]["old"])[:2]
            assert v["choices_trainable"] == 2
            assert v["fraction"] == pytest.approx(sum(least2) / sum(r["counts"]["old"]))
    most = RS.routing_mass_through(stats, "old", set(SA.apply_subset(
        build(moe_backbone()), RS.select_spec(stats, "old", "expert", 2, "most")).manifest["units_trainable"]))
    assert most["mean_by_family"]["expert"] > ov["mean_by_family"]["expert"]


def test_select_ratio_orders_contrast_target_and_old_usage():
    """ratio_high picks the experts most enriched in the target relative to the old corpus,
    independently of their absolute usage; ratio_low the most depleted."""
    old = [100, 100, 10, 50, 40, 100]
    tgt = [50, 300, 30, 10, 200, 10]       # (tgt/600)/(old/400) = 1/3, 2, 2, 2/15, 10/3, 1/15
    stats = {"routers": {"b0.l0.expert": {"family": "expert", "block": 0, "layer": 0, "choices": 6, "k": 2,
                                          "counts": {"old": old, "tgt": tgt}}}}
    hi = RS.select_spec(stats, "tgt", "expert", 2, "ratio_high", ref_corpus="old")
    lo = RS.select_spec(stats, "tgt", "expert", 2, "ratio_low", ref_corpus="old")
    assert hi["select"] == [{"kind": "moe", "block": 0, "layer": 0, "expert": [1, 4]}]
    assert lo["select"] == [{"kind": "moe", "block": 0, "layer": 0, "expert": [3, 5]}]
    with pytest.raises(ValueError):
        RS.select_spec(stats, "tgt", "expert", 2, "ratio_high")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--pipeline-child":
        _pipeline_child(json.loads(sys.argv[2]))
    else:
        sys.exit(pytest.main([__file__, "-q", "-rA"]))
