"""C3 per-layer / per-block revert scopes (prereg v1.3 sec. 4.8; weight_controls '<group>@<scope>').

The strata lists are the routed strata of the real G0 routing tables (g0/moe_b1, g0/nestor; probed 2026-11-13):
MoE-100M = 4 blocks x 3 layers; NESTOR-100M = 11 private strata over blocks 0-5 (block 6 = dense tail, excluded).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import confirm_plan as CP          # noqa: E402
import subset_adapt as SA          # noqa: E402
import weight_controls as WC       # noqa: E402

MOE_STRATA = [f"moe:b{b}:l{l}" for b in range(4) for l in range(3)]
NESTOR_STRATA = ["priv:b0:l0", "priv:b1:l0", "priv:b2:l0", "priv:b2:l1", "priv:b3:l0", "priv:b3:l1",
                 "priv:b4:l0", "priv:b4:l1", "priv:b5:l0", "priv:b5:l1", "priv:b5:l2"]
NESTOR_TAIL = 6


def test_tokens_real_geometry():
    moe = CP.c3_revert_tokens(MOE_STRATA, per_block=False)
    assert moe == [f"b{b}l{l}" for b in range(4) for l in range(3)] and len(moe) == 12
    assert CP.c3_revert_tokens(NESTOR_STRATA, per_block=True) == [f"b{b}" for b in range(6)]


@pytest.mark.parametrize("spec,expect", [
    ("b3", [(("block", 3),)]), ("3", [(("block", 3),)]), ("l2", [(("layer", 2),)]),
    ("b1l2", [(("block", 1), ("layer", 2))]), ("0/2", [(("block", 0),), (("block", 2),)]),
    ("b1l2/b3l0", [(("block", 1), ("layer", 2)), (("block", 3), ("layer", 0))]),
])
def test_scope_grammar(spec, expect):
    assert WC._scope(spec) == expect


@pytest.mark.parametrize("bad", ["", "x", "bl2", "b1/", "l", "b1lx", "1l"])
def test_scope_rejects_malformed(bad):
    with pytest.raises(ValueError):
        WC._scope(bad)


def _moe_units():
    return [SA.Unit("moe", b, l, expert=e) for b in range(4) for l in range(3) for e in range(2)] + [SA.Unit("attn", 0, 0)]


def _nestor_units():
    layers = {}
    for k in NESTOR_STRATA + ["priv:b6:l0", "priv:b6:l1"]:
        _, b, l = k.split(":")
        layers.setdefault(int(b[1:]), []).append(int(l[1:]))
    return [SA.Unit("priv", b, l, column=c, expert=e) for b, ls in layers.items() for l in ls for c in range(2) for e in range(2)]


def test_moe_single_layer_reverts_partition_the_expert_group():
    units = _moe_units()
    allp = WC._group_pred("moe", None)
    sel = [frozenset(i for i, u in enumerate(units) if WC._group_pred(f"moe@{t}", None)(u))
           for t in CP.c3_revert_tokens(MOE_STRATA, per_block=False)]
    assert all(len(s) == 2 for s in sel)                                        # one stratum = its 2 experts
    assert sum(len(s) for s in sel) == len(frozenset().union(*sel))             # disjoint
    assert frozenset().union(*sel) == {i for i, u in enumerate(units) if allp(u)}  # cover the whole group


def test_nestor_single_block_reverts_partition_the_private_group():
    units = _nestor_units()
    allp = WC._group_pred("priv", NESTOR_TAIL)
    sel = [frozenset(i for i, u in enumerate(units) if WC._group_pred(f"priv@{t}", NESTOR_TAIL)(u))
           for t in CP.c3_revert_tokens(NESTOR_STRATA, per_block=True)]
    assert sum(len(s) for s in sel) == len(frozenset().union(*sel))
    assert frozenset().union(*sel) == {i for i, u in enumerate(units) if allp(u)}
    assert not any(units[i].block == NESTOR_TAIL for s in sel for i in s)      # dense tail never reverted


def test_nestor_single_block_reverts_are_the_column_private_group():
    """Prereg v1.3 sec. 4.8 reverts the column-PRIVATE group of one block (column attention + private experts,
    v1.2 sec. 10.1). The plan's variant scopes BOTH parts; together the 6 blocks give exactly the group-level revert
    'attn+priv' and never touch the dense tail."""
    units = _nestor_units() + [SA.Unit("attn", b, None, column=c) for b in range(NESTOR_TAIL + 1) for c in range(2)]
    toks = CP.c3_revert_tokens(NESTOR_STRATA, per_block=True)
    vs = [CP.c3_layer_variant("attn+priv", t) for t in toks]
    assert vs[3] == "revert:attn@b3+priv@b3"
    groups = [v.split(":", 1)[1].split("+") for v in vs]                      # eval_windows: arg.split("+")
    sel = [frozenset(i for i, u in enumerate(units) if any(WC._group_pred(g, NESTOR_TAIL)(u) for g in gs)) for gs in groups]
    whole = {i for i, u in enumerate(units) if any(WC._group_pred(g, NESTOR_TAIL)(u) for g in ("attn", "priv"))}
    assert sum(len(s) for s in sel) == len(frozenset().union(*sel)) and frozenset().union(*sel) == whole
    assert all(any(units[i].kind == "attn" for i in s) and any(units[i].kind == "priv" for i in s) for s in sel)
    assert not any(units[i].block == NESTOR_TAIL for s in sel for i in s)
    assert CP.c3_layer_variant("moe", "b0l2") == "revert:moe@b0l2"


def test_old_scopes_unchanged():
    units = [SA.Unit("moe", b, None, expert=e) for b in range(3) for e in range(2)] + \
            [SA.Unit("priv", b, 0, column=c, expert=0) for b in range(2) for c in range(2)] + [SA.Unit("attn", 1)]
    assert [u.block for u in units if WC._group_pred("moe@b1", None)(u)] == [1, 1]
    assert sorted({u.block for u in units if WC._group_pred("moe@0/2", None)(u)}) == [0, 2]
    assert [(u.block, u.column) for u in units if WC._group_pred("priv@b1", 5)(u)] == [(1, 0), (1, 1)]
    assert len([u for u in units if WC._group_pred("priv@l0", 5)(u)]) == 4
    assert len([u for u in units if WC._group_pred("moe", None)(u)]) == 6
