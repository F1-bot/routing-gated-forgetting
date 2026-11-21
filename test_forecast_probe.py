"""forecast_probe.py (prereg v1.3 sec. 5.3) on the tiny CPU models.

* TF element counts equal subset_adapt.apply_subset (NESTOR_FREEZE_ROUTERS=1) for a subset and for FULL;
* QE / gamma / RND reproduce a direct g0_probe computation (same gradient, direction, eps, pinned legs);
* the Haar direction h is norm-matched (||h|| = ||u_S||); the SGD direction is (lr_sgd / 0.0025)(1 + mu) g on S;
* SVC equals an independent computation from forward pre-hooks on every expert's own projections (up, gate, down),
  on the per-expert path and, where grouped_expert_rows runs on CPU, on the grouped path (wrapper);
* finished units are skipped on restart.
"""
from __future__ import annotations

import test_phase3 as T3  # noqa: I001  (first: CPU env flags, sys.path, tiny builders)

import json
import os
import re

import pytest
import torch

import crossover as CX
import eval_windows as EW
import forecast_probe as FP
import g0_probe as G0
import subset_adapt as SA

BLOCK = 64
_EXP = re.compile(r"^backbone\.blocks\.(\d+)\.layers\.(\d+)\..*?(shared_experts\.(\d+)|private_experts\.(\d+)\.(\d+)|public_experts\.(\d+))$")


def _doc(tmp_path, m, kind):
    paths = {c: str(T3.write_bin(tmp_path / f"{c}.bin", BLOCK * 16, 10 + i)) for i, c in enumerate(("dclm", "code", "math", "tgt"))}
    sets = {c: {"path": paths[c], "tokens": BLOCK * 16, "select": "even"} for c in ("dclm", "code", "math")}
    rows = G0.unit_rows(m)
    fam = "moe" if kind == "moe" else "priv"
    keys = sorted(k for k in rows if k.startswith(fam + ":") and ":e" in k)[:3]
    spec = CX.spec_for(keys)
    muon = {"kind": "muon", "lr": 0.0025, "momentum": 0.95, "nesterov": True}
    sgd = {"kind": "sgd", "lr": 0.0125, "momentum": 0.95, "nesterov": True}

    def mk(i, sp, opt, sub):
        return {"index": i, "base": "b", "arch": kind, "target": "books", "optimizer": opt, "spec": sp, "subset": sub,
                "data": paths["tgt"], "haar_seed": 9000 + i, "old_corpora": ["code", "dclm", "math"], "probe_sha256": f"s{i}"}

    U = {"bases": {"b": {"sets": sets, "checkpoint": None}}, "block": BLOCK,
         "units": [mk(0, spec, muon, "SPEC"), mk(1, None, muon, "FULL"), mk(2, spec, sgd, "SPEC")]}
    return U, spec, paths


def _reference_svc(m, wins, direction, rows):
    """sum over expert projections of ||x dW||^2 / N from pre-hooks on the experts' own up / gate / down layers, with the
    per-expert (sparse) dispatch forced, so the experts' modules run on their routed rows."""
    acc, n_tok, hooks = {"v": 0.0}, 0, []
    saved = _set_grouped(m, False)
    for name, mod in m.named_modules():
        mm = _EXP.match(name)
        if not mm:
            continue
        b, l = int(mm.group(1)), int(mm.group(2))
        if mm.group(4) is not None:
            key = f"moe:b{b}:l{l}:e{mm.group(4)}"
        elif mm.group(5) is not None:
            key = f"priv:b{b}:l{l}:c{mm.group(5)}:e{mm.group(6)}"
        else:
            key = f"pub:b{b}:l{l}:e{mm.group(7)}"
        slots = {pn.rsplit(".", 1)[-1]: (pn, r) for pn, r in rows.get(key, []) if pn in direction}
        for part, lay in (("w_up", mod.layers[0]), ("w_gate", getattr(mod, "gate", None)), ("w_down", mod.layers[-1])):
            if lay is None or part not in slots:
                continue
            pn, r = slots[part]
            dW = direction[pn][r].float()

            def hook(_m, args, dW=dW):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                acc["v"] += float((x @ dW).double().pow(2).sum())
            hooks.append(lay.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for ws, i in wins:
                x = torch.as_tensor(ws.tokens(i)).view(1, -1)
                m(x)
                n_tok += int(x.numel())
    finally:
        for h in hooks:
            h.remove()
        _restore_grouped(saved)
    assert hooks, "no expert module found"
    return acc["v"] / n_tok


def _set_grouped(m, on: bool) -> list:
    """Force (on) or forbid (off) the grouped-GEMM expert dispatch; returns the previous flags for _restore_grouped."""
    saved = []
    for mod in m.modules():
        if hasattr(mod, "_moe_grouped_mm"):
            saved.append((mod, "_moe_grouped_mm", mod._moe_grouped_mm))
            mod._moe_grouped_mm = on
        if hasattr(mod, "grouped_dispatch") and hasattr(mod, "private_experts"):
            saved.append((mod, "grouped_dispatch", mod.grouped_dispatch))
            mod.grouped_dispatch = on
    return saved


def _restore_grouped(saved: list) -> None:
    for mod, attr, v in saved:
        setattr(mod, attr, v)


@pytest.mark.parametrize("kind", ["moe", "nestor"])
def test_probe_units_reproduce_reference(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("NESTOR_FREEZE_ROUTERS", "1")
    torch.manual_seed(0)
    m = T3.tiny(kind)
    U, spec, paths = _doc(tmp_path, m, kind)
    out = str(tmp_path / "probes")
    r = FP.probe_base(m, U, "b", out, device="cpu", bf16_grad=False, log=None)
    assert r["done"] == 3
    P = {i: json.load(open(os.path.join(out, f"probe_u{i:03d}.json"), encoding="utf-8")) for i in range(3)}
    for i, sp in ((0, spec), (1, {"mode": "freeze", "select": []})):     # TF counts = apply_subset
        mm = T3.tiny(kind)
        try:
            plan = SA.apply_subset(mm, sp)
        finally:
            SA.reset()
        assert P[i]["elements"]["trainable"] == plan.manifest["elements"]["trainable"], i
    assert P[1]["elements"]["trainable"] == P[1]["elements"]["full_trainable"] > P[0]["elements"]["trainable"] > 0
    rows = G0.unit_rows(m)                                              # direct reference for unit 0
    _, ents = FP.trainable_units(m, None)
    params = FP.muon_params(ents)
    assert params and not any("router" in n for n in params)
    gws = EW.WindowSet("books_grad", paths["tgt"], BLOCK, 8, "random", 12345)
    g = G0.target_gradient(m, gws, params, 8, device="cpu", bf16=False)
    tr, _ = FP.trainable_units(m, spec)
    keys = sorted(x.key for x in tr)
    uS = G0.restrict(G0.muon_direction(params, g), rows, keys)
    eps = P[0]["eps"]
    sets = {c: FP.eval_set(c, U["bases"]["b"]["sets"][c], BLOCK) for c in ("code", "dclm", "math")}
    wins = [(sets[c], i) for c in ("code", "dclm", "math") for i in range(8)]
    h = G0.random_like_direction(uS, 9000)
    res = G0.directional_probe(m, wins, [("u", uS, eps, 1), ("u", uS, eps, -1), ("h", h, eps, 1), ("h", h, eps, -1)],
                               device="cpu", log=None, pin_routing=True, count_flips=False)
    for k, c in enumerate(("code", "dclm", "math")):
        sub = {kk: {f: v[f][8 * k: 8 * k + 8] for f in v} for kk, v in res.items()}
        qu, qh = G0.quad_forms(sub, "u", eps), G0.quad_forms(sub, "h", eps)
        bc = P[0]["by_corpus"][c]
        assert bc["QE"] == pytest.approx(qu["Q_kl"]["value"], rel=1e-9, abs=1e-15)
        assert bc["gamma"] == pytest.approx(qu["Lin"]["value"], rel=1e-9, abs=1e-15)
        assert bc["RND"] == pytest.approx(qh["Quad"]["value"], rel=1e-9, abs=1e-15)
        assert bc["QE"] > 0 and bc["windows"] == 8
        ref = _reference_svc(m, [(sets[c], i) for i in range(8)], uS, rows)
        assert ref > 0 and bc["SVC"] == pytest.approx(ref, rel=1e-6)
    assert P[0]["hdir_sq"] == pytest.approx(P[0]["udir_sq"], rel=1e-5)
    gS = G0.restrict(g, rows, keys)
    assert P[2]["udir_sq"] == pytest.approx((0.0125 / 0.0025 * 1.95) ** 2 * G0.sq_norm(gS), rel=1e-6)
    assert P[1]["udir_sq"] > P[0]["udir_sq"]                            # FULL: every Muon matrix
    r2 = FP.probe_base(m, U, "b", out, device="cpu", bf16_grad=False, log=None)
    assert r2["done"] == 0 and r2["skipped"] == 3                       # restart skips finished units


@pytest.mark.parametrize("kind", ["moe", "nestor"])
def test_svc_capture_grouped_path(tmp_path, monkeypatch, kind):
    """Per-expert path = projection-hook reference; grouped path = a reference recomputed from the very rows the grouped
    calls dispatched (same forward); first routed layer equal across paths (identical inputs; deeper layers may differ
    by top-k flips from GEMM-tiling noise)."""
    import sys
    from collections import defaultdict

    import nestor.backbones.grouped_mm as GM

    torch.manual_seed(1)
    m = T3.tiny(kind)
    U, spec, paths = _doc(tmp_path, m, kind)
    rows = G0.unit_rows(m)
    _, ents = FP.trainable_units(m, None)
    params = FP.muon_params(ents)
    gws = EW.WindowSet("books_grad", paths["tgt"], BLOCK, 8, "random", 12345)
    u = G0.muon_direction(params, G0.target_gradient(m, gws, params, 8, device="cpu", bf16=False))
    ws = FP.eval_set("dclm", U["bases"]["b"]["sets"]["dclm"], BLOCK)
    wins = [(ws, i) for i in range(4)]
    saved = _set_grouped(m, False)
    try:
        a0, n0 = FP.svc_pass(m, wins, {"g": u}, "cpu")                   # per-expert path (pre-hooks)
    finally:
        _restore_grouped(saved)
    ref = _reference_svc(m, wins, u, rows)
    assert n0 == 4 * BLOCK and sum(a0["g"].values()) / n0 == pytest.approx(ref, rel=1e-6)
    saved = _set_grouped(m, True)
    calls, orig = [], GM.grouped_expert_rows

    def spy(x, experts, expert_ids, *a, **kw):
        calls.append((x.detach().clone(), experts, expert_ids.detach().clone()))
        return orig(x, experts, expert_ids, *a, **kw)

    for name, mod in list(sys.modules.items()):
        if name.startswith("nestor") and getattr(mod, "grouped_expert_rows", None) is orig:
            monkeypatch.setattr(mod, "grouped_expert_rows", spy)
    try:
        a1, n1 = FP.svc_pass(m, wins, {"g": u}, "cpu")
    except Exception as e:                                              # noqa: BLE001
        pytest.skip(f"grouped_expert_rows not available on CPU: {type(e).__name__}: {e}")
    finally:
        _restore_grouped(saved)
    if not calls:
        pytest.skip("this tiny model never takes the grouped path")
    name_of = {id(mod): name for name, mod in m.named_modules()}
    exp_ref = defaultdict(float)
    for x, experts, ids in calls:                                       # reference from the dispatched rows
        ids2 = ids.reshape(x.shape[0], -1)
        for e in torch.unique(ids2).tolist():
            ex = experts[int(e)]
            mm = _EXP.match(name_of[id(ex)])
            b, l = int(mm.group(1)), int(mm.group(2))
            key = (f"moe:b{b}:l{l}:e{mm.group(4)}" if mm.group(4) is not None else
                   f"priv:b{b}:l{l}:c{mm.group(5)}:e{mm.group(6)}" if mm.group(5) is not None else f"pub:b{b}:l{l}:e{mm.group(7)}")
            slots = {pn.rsplit(".", 1)[-1]: (pn, r) for pn, r in rows.get(key, []) if pn in u}
            seen = {}
            hs = [lay.register_forward_pre_hook(lambda _m, args, p=p: seen.__setitem__(p, args[0].float()))
                  for p, lay in (("w_up", ex.layers[0]), ("w_gate", getattr(ex, "gate", None)), ("w_down", ex.layers[-1]))
                  if lay is not None]
            try:
                with torch.no_grad():
                    ex(x[(ids2 == int(e)).any(-1)].float())
            finally:
                for h in hs:
                    h.remove()
            for p, inp in seen.items():
                if p in slots:
                    pn, r = slots[p]
                    exp_ref[(pn, r)] += float((inp @ u[pn][r].float()).double().pow(2).sum())
    assert set(a1["g"]) == {k for k, v in exp_ref.items()}
    for k, v in exp_ref.items():
        assert a1["g"][k] == pytest.approx(v, rel=1e-6, abs=1e-12), k
    first = min((int(re.search(r"blocks\.(\d+)", k[0]).group(1)), int(re.search(r"layers\.(\d+)", k[0]).group(1)))
                for k in a0["g"])
    pat = f"blocks.{first[0]}.layers.{first[1]}."
    for k in a0["g"]:
        if pat in k[0]:
            assert a1["g"][k] == pytest.approx(a0["g"][k], rel=1e-5, abs=1e-12), k
