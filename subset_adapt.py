"""Exact subset-restricted adaptation for the Nestor repo (commit c797821).

Trains only a chosen set of modules -- Nestor columns (their attention projections and their
private experts), public (shared) experts, MoE experts, and the shared groups (embeddings, LM head,
norms, depth-read queries/gains, path gates, routers, the dense last block) -- and leaves every
other parameter BIT-IDENTICAL under the authors' MuonAdamW (Muon + AdamW + decoupled weight decay).

Activation (environment; a strict no-op when NESTOR_SUBSET_SPEC is unset):
    NESTOR_SUBSET_SPEC      path to a JSON spec file, or the JSON text itself
    NESTOR_SUBSET_MANIFEST  where to write the resolved manifest (default:
                            $NESTOR_OUTPUT_ROOT/subset_manifest.json, else ./subset_manifest.json)
    NESTOR_SUBSET_VERIFY=1  snapshot the frozen slices before every optimizer step and assert after
                            it that they are unchanged (debug only: costs memory)
In run_exp.py add ``import subset_adapt; subset_adapt.install()`` right after ``compat.install()``.

Spec (JSON):
    {"mode": "train" | "freeze",
     "select": [{"kind": ..., "block": ..., "layer": ..., "column": ..., "expert": ..., "role": ...}, ...]}
  mode "train": the selected units train and everything else is frozen; mode "freeze": the selected
  units are frozen and everything else trains. A filter matches a unit when every key it names
  matches (a value, a list of values, or "*"); negative block indices count from the last block;
  "dense_tail": true restricts a filter to the last block when that block is dense (k = C > 1).
  Unit kinds (one unit = the parameters of one module; units partition every parameter element):
    attn        attention projections of one column (Nestor) or of the layer (MoE/dense: column 0)
    priv        one private expert of a column (block, layer, column, expert)
    pub         one public (shared) expert of a Nestor expert pool (block, layer, expert)
    moe         one expert of a MoE / dense bank (block, layer, expert)
    router      a router's score + noise networks (role: column | expert | public | private)
    path_gate   the public/private mixing gate of a Nestor expert pool (block, layer)
    norm        RMSNorm gains (role: attn_norm | mlp_norm | final_norm)
    depth_read  depth-attention read query + value gains (role: kimi_attn_read | kimi_mlp_read | final_depth_read)
    embed, head
  Macro kinds: "column" = attn + priv; "*" = any kind.

Exactness mechanism (see protocol_engineering.md for the derivation and the tests):
  1. A parameter all of whose units are frozen gets requires_grad=False BEFORE the optimizer is
     built, so MuonAdamW never holds it (no update, no weight decay, no state) and the clip norm
     ignores it. This is exactly how the authors freeze routers (NESTOR_FREEZE_ROUTERS).
  2. A stacked expert bank [E, ...] with frozen and trainable experts stays trainable and gets
     (a) a gradient hook that writes exact zeros into the frozen rows (so they also add nothing to
     the clip norm), and (b) an optimizer pre-step hook that re-zeroes those rows of the gradient
     (guard), zeroes the same rows of momentum_buffer / exp_avg / exp_avg_sq, and applies the
     decoupled weight decay itself -- factor 1.0 on frozen rows, the group's (1 - lr*wd) on
     everything else the step will touch -- while the group's weight_decay is 0 for that step
     (restored by the post-step hook). Muon orthogonalizes each [m, n] slice of a stacked bank
     independently (zeropower_via_newtonschulz5_batched normalizes per slice), so a zero direction
     gives an exactly zero update on a frozen row and the trainable rows are unaffected.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

import torch
from torch import Tensor, nn

SPEC_ENV = "NESTOR_SUBSET_SPEC"
MANIFEST_ENV = "NESTOR_SUBSET_MANIFEST"
VERIFY_ENV = "NESTOR_SUBSET_VERIFY"
NEGCTRL_ENV = "NESTOR_SUBSET_NEGCTRL"          # "unmasked_decay": deliberately leak decay (audit must FAIL)
ALLOW_CHANGE_ENV = "NESTOR_SUBSET_ALLOW_SPEC_CHANGE"
FINGERPRINT_ENV = ("NESTOR_FREEZE_ROUTERS", "NESTOR_SUBSET_OPTIMIZER", "NESTOR_SGD_LR", "NESTOR_SGD_MOMENTUM",
                   "NESTOR_SGD_NESTEROV", "NESTOR_SGD_WD", "NESTOR_SGD_KINDS", "NESTOR_SGD_SCALE", "DCLM_LR_MUON",
                   "DCLM_LR_ADAMW", "DCLM_WEIGHT_DECAY", "DCLM_MAX_STEPS", "DCLM_WARMUP_STEPS", "DCLM_INIT_CHECKPOINT",
                   "DCLM_DATA_PATH", "DCLM_SEED", "NESTOR_MAX_GRAD_NORM", NEGCTRL_ENV, "NESTOR_STACKED_EXPERT_BANKS")
KINDS = ("attn", "priv", "pub", "moe", "router", "path_gate", "norm", "depth_read", "embed", "head")
MACROS = {"column": ("attn", "priv")}
_FILTER_KEYS = {"kind", "block", "layer", "column", "expert", "role", "dense_tail"}

_BL = re.compile(r"^backbone\.blocks\.(\d+)\.(?:layers\.(\d+)\.)?")
_ATTN = re.compile(r"\.attention_groups\.(\d+)\.")
_PRIV_UNSTACKED = re.compile(r"\.expert_pool\.private_experts\.(\d+)\.(\d+)\.")
_PUB_UNSTACKED = re.compile(r"\.expert_pool\.public_experts\.(\d+)\.")
_MOE_UNSTACKED = re.compile(r"\.shared_experts\.(\d+)\.")
_PRIVATE_ROUTER = re.compile(r"\.private_routers\.(\d+)$")


# ----------------------------------------------------------------------------- catalog
@dataclasses.dataclass(frozen=True)
class Unit:
    """One module-level unit; the units of a model partition all of its parameter elements."""

    kind: str
    block: int | None = None
    layer: int | None = None
    column: int | None = None
    expert: int | None = None
    role: str | None = None

    @property
    def key(self) -> str:
        parts = [self.kind]
        for tag, v in (("b", self.block), ("l", self.layer), ("c", self.column), ("e", self.expert)):
            if v is not None:
                parts.append(f"{tag}{v}")
        if self.role is not None:
            parts.append(self.role)
        return ":".join(parts)


@dataclasses.dataclass
class Entry:
    """One parameter tensor and the unit(s) that own it (one per dim-0 row when ``sliced``)."""

    name: str
    param: nn.Parameter
    units: list[Unit]
    sliced: bool
    optimizer: str | None
    construction_frozen: bool


def unwrap(model: nn.Module) -> nn.Module:
    """Strip torch.compile (``_orig_mod``) and DDP (``.module``) wrappers."""

    from torch.nn.parallel import DistributedDataParallel

    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif isinstance(model, DistributedDataParallel):
            model = model.module
        else:
            return model


def _scaling_optimizer(p: Tensor) -> str | None:
    try:
        from nestor.scaling.blocks import get_scaling
    except Exception:  # pragma: no cover
        return None
    s = get_scaling(p)
    return None if s is None else s.optimizer


def _bank_specs(modules: dict[str, nn.Module]) -> dict[str, tuple]:
    """Stacked-bank module name -> (kind, rows-per-column Q or None) for every expert bank."""

    specs: dict[str, tuple] = {}
    for name, m in modules.items():
        if hasattr(m, "num_private_experts") and hasattr(m, "private_experts"):  # ExpertPoolMLP
            specs[f"{name}.stacked_private"] = ("priv", int(m.num_groups), int(m.num_private_experts))
            specs[f"{name}.stacked_public"] = ("pub", None, None)
        if hasattr(m, "num_shared_experts") and hasattr(m, "num_group_specific_experts"):  # routed layer
            if getattr(m, "expert_pool", None) is None:
                if int(m.num_group_specific_experts) != 0 or int(getattr(m, "num_dense_shared_experts", 0)) != 0:
                    raise NotImplementedError(
                        f"{name}: banks with group-specific or dense-shared experts are not supported")
                specs[f"{name}.stacked_moe_bank"] = ("moe", int(m.num_shared_experts), None)
    return specs


def build_catalog(model: nn.Module) -> list[Entry]:
    """Assign every parameter element of ``model`` to exactly one :class:`Unit`.

    Raises:
        ValueError: on a parameter it cannot classify (a new layout must fail loudly, not be
            silently assigned to the wrong unit).
    """

    base = unwrap(model)
    modules = dict(base.named_modules())
    router_prefixes = sorted(
        (n for n, m in modules.items() if hasattr(m, "score_mlp") and hasattr(m, "choice_bias")),
        key=len, reverse=True)
    banks = _bank_specs(modules)
    entries: list[Entry] = []
    for name, p in base.named_parameters():
        prefix, _, leaf = name.rpartition(".")
        owner = modules.get(prefix)
        m = _BL.match(name)
        b = int(m.group(1)) if m else None
        lay = int(m.group(2)) if (m and m.group(2) is not None) else None
        units: list[Unit]
        sliced = False
        rp = next((r for r in router_prefixes if name.startswith(r + ".")), None)
        if rp is not None:
            pm = _PRIVATE_ROUTER.search(rp)
            if pm:
                units = [Unit("router", b, lay, column=int(pm.group(1)), role="private")]
            else:
                tail = rp.rsplit(".", 1)[-1]
                role = {"group_router": "column", "expert_router": "expert",
                        "public_router": "public"}.get(tail, tail)
                units = [Unit("router", b, lay, role=role)]
        elif prefix in banks:
            kind, g_or_s, q = banks[prefix]
            rows = p.shape[0]
            if kind == "priv":
                if rows != g_or_s * q:
                    raise ValueError(f"{name}: {rows} rows != G*Q = {g_or_s}*{q}")
                units = [Unit("priv", b, lay, column=i // q, expert=i % q) for i in range(rows)]
            elif kind == "pub":
                units = [Unit("pub", b, lay, expert=i) for i in range(rows)]
            else:
                if rows != g_or_s:
                    raise ValueError(f"{name}: {rows} rows != shared experts {g_or_s}")
                units = [Unit("moe", b, lay, expert=i) for i in range(rows)]
            sliced = True
        elif (pm := _PRIV_UNSTACKED.search(name)) is not None:
            units = [Unit("priv", b, lay, column=int(pm.group(1)), expert=int(pm.group(2)))]
        elif (pm := _PUB_UNSTACKED.search(name)) is not None:
            units = [Unit("pub", b, lay, expert=int(pm.group(1)))]
        elif (pm := _MOE_UNSTACKED.search(name)) is not None and ".expert_pool." not in name:
            units = [Unit("moe", b, lay, expert=int(pm.group(1)))]
        elif (pm := _ATTN.search(name)) is not None:
            units = [Unit("attn", b, lay, column=int(pm.group(1)))]
        elif ".expert_pool.path_gate." in name:
            units = [Unit("path_gate", b, lay)]
        elif owner is not None and type(owner).__name__ == "DepthAttentionRead":
            units = [Unit("depth_read", b, lay, role=prefix.rsplit(".", 1)[-1])]
        elif owner is not None and type(owner).__name__.endswith("Norm"):
            units = [Unit("norm", b, lay, role=prefix.rsplit(".", 1)[-1])]
        elif name.startswith("adapter.embed."):
            units = [Unit("embed")]
        elif name.startswith("head."):
            units = [Unit("head")]
        else:
            raise ValueError(f"subset_adapt: unclassified parameter {name!r} "
                             f"(owner {type(owner).__name__ if owner is not None else None})")
        entries.append(Entry(name, p, units, sliced, _scaling_optimizer(p), not p.requires_grad))
    return entries


def model_geometry(model: nn.Module) -> dict[str, Any]:
    """Blocks, columns and top-k per block, and the dense-tail block (last block with k = C > 1)."""

    base = unwrap(model)
    cfg = getattr(getattr(base, "backbone", None), "config", None)
    groups = list(getattr(cfg, "num_groups", []) or [])
    topk = list(getattr(cfg, "num_groups_per_token", []) or [])
    n = len(groups)
    dense_tail = n - 1 if n and groups[-1] > 1 and topk[-1] == groups[-1] else None
    return {"num_blocks": n, "num_groups": groups, "num_groups_per_token": topk, "dense_tail_block": dense_tail}


# ----------------------------------------------------------------------------- spec
def load_spec(text_or_path: str) -> dict[str, Any]:
    """Parse a spec given as JSON text or as a path to a JSON file."""

    s = text_or_path.strip()
    spec = json.loads(s) if s.startswith("{") else json.loads(open(s, encoding="utf-8").read())
    if spec.get("mode", "train") not in ("train", "freeze"):
        raise ValueError(f"spec mode must be 'train' or 'freeze', got {spec.get('mode')!r}")
    if not isinstance(spec.get("select", []), list):
        raise ValueError("spec 'select' must be a list of filters")
    for f in spec.get("select", []):
        bad = set(f) - _FILTER_KEYS
        if bad:
            raise ValueError(f"unknown filter keys {sorted(bad)} (allowed: {sorted(_FILTER_KEYS)})")
    return spec


def _as_set(v: Any) -> set | None:
    if v == "*" or v is None:
        return None
    return set(v) if isinstance(v, (list, tuple, set)) else {v}


def unit_matches(u: Unit, f: dict[str, Any], geo: dict[str, Any]) -> bool:
    """Whether unit ``u`` satisfies every key of filter ``f``."""

    kinds = _as_set(f.get("kind", "*"))
    if kinds is not None:
        expanded = set()
        for k in kinds:
            expanded.update(MACROS.get(k, (k,)))
        unknown = expanded - set(KINDS)
        if unknown:
            raise ValueError(f"unknown unit kind(s) {sorted(unknown)}")
        if u.kind not in expanded:
            return False
    nb = geo.get("num_blocks") or 0
    for key in ("block", "layer", "column", "expert", "role"):
        want = _as_set(f.get(key, "*"))
        if want is None:
            continue
        if key == "block" and nb:
            want = {w + nb if isinstance(w, int) and w < 0 else w for w in want}
        if getattr(u, key) not in want:
            return False
    if f.get("dense_tail"):
        if geo.get("dense_tail_block") is None or u.block != geo["dense_tail_block"]:
            return False
    return True


# ----------------------------------------------------------------------------- plan
@dataclasses.dataclass
class Plan:
    spec: dict[str, Any]
    entries: list[Entry]
    trainable_units: set[Unit]
    manifest: dict[str, Any]


def _row_shape(p: Tensor) -> tuple[int, ...]:
    return (p.shape[0],) + (1,) * (p.ndim - 1)


def _frozen_mask(p: Tensor, device: torch.device) -> Tensor:
    """Boolean [E, 1, ..., 1] mask of the frozen rows of a partially trainable parameter."""

    cache = p._subset_mask_cache
    m = cache.get(device)
    if m is None:
        m = p._subset_frozen_rows.to(device).view(_row_shape(p))
        cache[device] = m
    return m


def _mask_grad_hook(p: nn.Parameter):
    def hook(grad: Tensor) -> Tensor:
        return grad.masked_fill(_frozen_mask(p, grad.device), 0)
    return hook


_PARTIAL: dict[int, nn.Parameter] = {}   # id -> partially trainable parameter (live registry)
_NAMES: dict[int, str] = {}               # id -> parameter name of every catalogued parameter
_ORDER_DONE: set[int] = set()             # optimizers whose parameter order was written to the manifest
_MANIFEST_STATE: dict[str, Any] = {"path": None, "fingerprint": None}


def mark_partial(p: nn.Parameter, frozen_rows: Iterable[bool], name: str = "?", *, grad_hook: bool = True) -> None:
    """Make the dim-0 rows ``frozen_rows`` of ``p`` exactly frozen (gradient hook + optimizer hooks).

    The registry holds a strong reference, so a registered parameter's ``id`` is never reused.
    """

    p._subset_frozen_rows = torch.as_tensor(list(frozen_rows), dtype=torch.bool)
    if p._subset_frozen_rows.numel() != p.shape[0]:
        raise ValueError(f"{name}: {p._subset_frozen_rows.numel()} row flags for {p.shape[0]} rows")
    p._subset_mask_cache = {}
    p._subset_name = name
    if grad_hook and not getattr(p, "_subset_hooked", False):
        p.register_hook(_mask_grad_hook(p))
        p._subset_hooked = True
    _PARTIAL[id(p)] = p


def reset() -> None:
    """Forget every registered parameter and counter (tests; one process, several models)."""

    _PARTIAL.clear()
    _PENDING.clear()
    _SNAP.clear()
    STATS.clear()
    _NAMES.clear()
    _ORDER_DONE.clear()
    _MANIFEST_STATE.update(path=None, fingerprint=None)


def apply_subset(model: nn.Module, spec: dict[str, Any], *, write_manifest: str | None = None) -> Plan:
    """Freeze everything the spec does not train; register row masks on mixed stacked banks.

    Must run before the optimizer is built (and before any DDP wrap); safe before ``.to(device)``.
    """

    if os.environ.get("NESTOR_BF16_MASTER") == "1":
        raise NotImplementedError("subset_adapt does not support NESTOR_BF16_MASTER=1")
    entries = build_catalog(model)
    geo = model_geometry(model)
    filters = spec.get("select", [])
    all_units = [u for e in entries for u in e.units]
    selected = {u for u in all_units if any(unit_matches(u, f, geo) for f in filters)}
    unmatched = [i for i, f in enumerate(filters) if not any(unit_matches(u, f, geo) for u in all_units)]
    if unmatched and not spec.get("allow_unmatched", False):
        raise ValueError(f"spec filters {unmatched} match no unit (typo?): {[filters[i] for i in unmatched]}")
    mode = spec.get("mode", "train")
    trainable = selected if mode == "train" else (set(all_units) - selected)

    # Routers: NESTOR_FREEZE_ROUTERS=1 (the paper's adaptation recipe) freezes every router later in
    # TrainPipeline._build_model, holds routing in eval mode and stops the choice_bias updates.
    # Without it a frozen router would still have its choice_bias nudged and route with noise, so
    # freezing any live router requires the flag.
    freeze_routers_env = os.environ.get("NESTOR_FREEZE_ROUTERS", "").strip() == "1"
    base = unwrap(model)
    lfb_routers = any(getattr(m, "loss_free_balancing", False) and hasattr(m, "choice_bias")
                      for m in base.modules())
    live_router_units = {u for e in entries if not e.construction_frozen for u in e.units if u.kind == "router"}
    if freeze_routers_env:
        names_router = any("router" in (_as_set(f.get("kind", "*")) or set()) for f in filters)
        if mode == "train" and names_router and (selected & live_router_units):
            raise ValueError("spec trains router units but NESTOR_FREEZE_ROUTERS=1 freezes every router")
        trainable = trainable - live_router_units   # wildcard matches leave routers frozen
    elif (live_router_units - trainable) and lfb_routers:
        raise ValueError(
            "spec freezes router units of a loss-free-balancing model: set NESTOR_FREEZE_ROUTERS=1, "
            "otherwise the trainer still nudges every choice_bias and routes with noise")

    params_out = []
    n_frozen_t = n_partial_t = 0
    el = Counter()
    for e in entries:
        p = e.param
        flags = [u in trainable for u in e.units]
        if e.construction_frozen:
            status = "frozen_at_construction"
        elif all(flags):
            status = "trainable"
        elif not any(flags):
            status = "frozen"
            p.requires_grad_(False)
            n_frozen_t += 1
        else:
            status = "partial"
            if not e.sliced:
                raise AssertionError(e.name)
            if e.optimizer == "muon" and p.ndim != 3:
                raise NotImplementedError(f"{e.name}: row masks on a 2-D Muon matrix couple the "
                                          "orthogonalized update of the trainable rows to the mask")
            mark_partial(p, [not f for f in flags], e.name)
            n_partial_t += 1
        per_row = p[0].numel() if e.sliced else p.numel()
        for u, f in zip(e.units, flags):
            el[(u.kind, "trainable" if (f and not e.construction_frozen) else "frozen")] += per_row
        rec = {"name": e.name, "shape": list(p.shape), "status": status, "optimizer": e.optimizer}
        if e.sliced:
            rec["units"] = [u.key for u in e.units]
            rec["trainable_rows"] = [i for i, f in enumerate(flags) if f]
        else:
            rec["unit"] = e.units[0].key
        params_out.append(rec)
    tot_tr = sum(v for (k, s), v in el.items() if s == "trainable")
    tot = sum(el.values())
    by_kind = defaultdict(dict)
    for (k, s), v in el.items():
        by_kind[k][s] = v
    for e in entries:
        _NAMES[id(e.param)] = e.name
    manifest = {
        "spec": spec, "geometry": geo, "fingerprint": _MANIFEST_STATE.get("fingerprint"),
        "negative_control": os.environ.get(NEGCTRL_ENV) or None,
        "env": {k: os.environ.get(k) for k in ("NESTOR_FREEZE_ROUTERS", "NESTOR_STACKED_EXPERT_BANKS",
                                                "NESTOR_DENSE_NO_ROUTER", "DCLM_WEIGHT_DECAY")},
        "units_total": len(all_units), "units_selected": len(selected),
        "units_trainable": sorted(u.key for u in trainable),
        "elements": {"trainable": tot_tr, "total": tot, "by_kind": dict(by_kind)},
        "tensors": {"frozen_by_subset": n_frozen_t, "partial": n_partial_t, "count": len(entries)},
        "params": params_out,
    }
    path = write_manifest
    if path:
        _write_json(path, manifest)
        _MANIFEST_STATE["path"] = path
    print(f"[subset] mode={mode}: {len(trainable & set(all_units))}/{len(all_units)} units trainable; "
          f"{tot_tr:,}/{tot:,} elements ({100.0 * tot_tr / max(tot, 1):.2f}%) trainable; "
          f"{n_frozen_t} tensors frozen, {n_partial_t} row-masked banks"
          + (f"; manifest {path}" if path else ""), flush=True)
    return Plan(spec, entries, trainable, manifest)


# ----------------------------------------------------------------------------- optimizer hooks
_PENDING: dict[int, list] = {}
_SNAP: dict[int, list] = {}
_HOOK_HANDLES: list = []
STATS = Counter()   # counters for tests / logs


def _supported(opt: torch.optim.Optimizer) -> bool:
    try:
        from nestor.scaling.muon import MuonAdamW
    except Exception:  # pragma: no cover
        MuonAdamW = ()  # type: ignore
    return isinstance(opt, (MuonAdamW, torch.optim.AdamW))


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _record_optimizer(opt: torch.optim.Optimizer) -> None:
    """Write the optimizer's flattened parameter order (= state_dict indices) into the manifest."""

    _ORDER_DONE.add(id(opt))
    path = _MANIFEST_STATE.get("path")
    if not path or not os.path.exists(path):
        return
    man = json.load(open(path, encoding="utf-8"))
    man["optimizer_class"] = type(opt).__name__
    man["optimizer_param_order"] = [_NAMES.get(id(p), "?") for g in opt.param_groups for p in g["params"]]
    man["optimizer_groups"] = [{"method": g.get("method"), "initial_lr": g.get("initial_lr", g.get("lr")),
                                "weight_decay": g.get("weight_decay"), "n_params": len(g["params"])} for g in opt.param_groups]
    _write_json(path, man)


def _pre_step(opt: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
    if id(opt) not in _ORDER_DONE and _NAMES:
        _record_optimizer(opt)
    if not _PARTIAL:
        return
    negctrl = os.environ.get(NEGCTRL_ENV, "").strip()
    restore = []
    verify = os.environ.get(VERIFY_ENV) == "1"
    snaps = []
    for group in opt.param_groups:
        parts = [p for p in group["params"] if id(p) in _PARTIAL]
        if not parts:
            continue
        if not _supported(opt):
            raise NotImplementedError(f"subset_adapt: row-masked banks need MuonAdamW or AdamW, got {type(opt).__name__}")
        if getattr(opt, "_shard", False):
            raise NotImplementedError("subset_adapt: NESTOR_MUON_SHARD=1 under DDP is not supported")
        for p in parts:
            fr = _frozen_mask(p, p.device)
            if verify:
                snaps.append((p, p.detach()[p._subset_frozen_rows.to(p.device)].clone()))
            if p.grad is not None:
                if verify:
                    STATS["guard_nonzero_grad_rows"] += int((p.grad.masked_fill(~fr, 0) != 0).any())
                p.grad.masked_fill_(fr, 0)
            state = opt.state.get(p)
            if state:
                for k in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                    t = state.get(k)
                    if torch.is_tensor(t) and t.shape == p.shape:
                        t.masked_fill_(fr, 0)
        wd = float(group.get("weight_decay", 0.0) or 0.0)
        if negctrl == "unmasked_decay":
            STATS["negctrl_unmasked_decay_groups"] += 1
            continue                      # the authors' whole-tensor decay then also hits frozen rows
        if wd != 0.0:
            c = 1.0 - group["lr"] * wd
            whole = [p for p in group["params"] if p.grad is not None and id(p) not in _PARTIAL]
            if whole:
                torch._foreach_mul_(whole, c)
            for p in parts:
                if p.grad is None:
                    continue
                fac = torch.where(_frozen_mask(p, p.device),
                                  torch.ones((), dtype=torch.float32, device=p.device),
                                  torch.full((), c, dtype=torch.float32, device=p.device))
                p.detach().mul_(fac)
            restore.append((group, group["weight_decay"]))
            group["weight_decay"] = 0.0
            STATS["masked_decay_groups"] += 1
    STATS["pre_steps"] += 1
    _PENDING[id(opt)] = restore
    if verify:
        _SNAP[id(opt)] = snaps


def _post_step(opt: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
    for group, wd in _PENDING.pop(id(opt), []):
        group["weight_decay"] = wd
    for p, before in _SNAP.pop(id(opt), []):
        after = p.detach()[p._subset_frozen_rows.to(p.device)]
        if not torch.equal(_bits(after), _bits(before)):
            raise RuntimeError(f"subset_adapt VERIFY: frozen rows of {p._subset_name} changed")
        STATS["verified_tensors"] += 1


def register_optimizer_hooks() -> None:
    """Install the global optimizer pre/post-step hooks (idempotent)."""

    if _HOOK_HANDLES:
        return
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    _HOOK_HANDLES.append(register_optimizer_step_pre_hook(_pre_step))
    _HOOK_HANDLES.append(register_optimizer_step_post_hook(_post_step))


def remove_optimizer_hooks() -> None:
    while _HOOK_HANDLES:
        _HOOK_HANDLES.pop().remove()
    _PARTIAL.clear()


# ----------------------------------------------------------------------------- pipeline hook
_INSTALLED = {"done": False}


def _manifest_path() -> str:
    if os.environ.get(MANIFEST_ENV):
        return os.environ[MANIFEST_ENV]
    root = os.environ.get("NESTOR_OUTPUT_ROOT")
    return os.path.join(root, "subset_manifest.json") if root else os.path.abspath("subset_manifest.json")


def install() -> bool:
    """Activate subset adaptation for this process from NESTOR_SUBSET_SPEC (no-op if unset).

    Wraps ``nestor.models.base.Model.from_config`` (the call ``TrainPipeline._build_model`` makes)
    so every model built in this process gets the plan applied right after construction -- before
    ``.to(device)``, NESTOR_FREEZE_ROUTERS, any wrapper, torch.compile and the optimizer -- and
    registers the optimizer hooks. Returns whether anything was installed.
    """

    if _INSTALLED["done"]:
        return True
    raw = os.environ.get(SPEC_ENV, "").strip()
    spec = load_spec(raw) if raw else None
    guard_resume(spec)                     # refuses a spec change inside one NESTOR_OUTPUT_ROOT
    if spec is None:
        return False
    from nestor.models import base as _mb

    original = _mb.Model.from_config

    def from_config_with_subset(config):  # noqa: ANN001, ANN202
        model = original(config)
        apply_subset(model, spec, write_manifest=_manifest_path())
        return model

    _mb.Model.from_config = staticmethod(from_config_with_subset)
    register_optimizer_hooks()
    _INSTALLED["done"] = True
    print(f"[subset] installed (spec from {SPEC_ENV}: mode={spec.get('mode', 'train')}, "
          f"{len(spec.get('select', []))} filters)", flush=True)
    return True


# ----------------------------------------------------------------------------- offline audit
def _bits(t: Tensor) -> Tensor:
    t = t.detach().contiguous().reshape(-1)
    return t.view(torch.uint8) if t.numel() else t


def spec_fingerprint(spec: dict[str, Any] | None) -> str:
    """sha256 over the spec and every env setting that changes what a run trains."""

    import hashlib

    blob = json.dumps({"spec": spec, "env": {k: os.environ.get(k) for k in FINGERPRINT_ENV}}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _has_checkpoints(root: str) -> bool:
    import glob

    return bool(glob.glob(os.path.join(root, "**", "model_*.pt"), recursive=True))


def guard_resume(spec: dict[str, Any] | None) -> None:
    """Forbid resuming (or reusing) a NESTOR_OUTPUT_ROOT under a different spec / training setting.

    The authors' pipeline auto-resumes the latest checkpoint under the output root with its optimizer
    state, so a relaunch with another spec would silently mix two conditions. Rules:
    manifest present and fingerprint differs -> refuse; spec removed but a manifest exists -> refuse;
    spec given, no manifest, but checkpoints exist -> refuse. NESTOR_SUBSET_ALLOW_SPEC_CHANGE=1 only
    lifts the rule when the root holds no checkpoint yet.
    """

    root = os.environ.get("NESTOR_OUTPUT_ROOT")
    fp = spec_fingerprint(spec) if spec is not None else None
    _MANIFEST_STATE["fingerprint"] = fp
    if not root:
        return
    man_path = _manifest_path()
    allow = os.environ.get(ALLOW_CHANGE_ENV) == "1" and not _has_checkpoints(root)
    if os.path.exists(man_path):
        old = json.load(open(man_path, encoding="utf-8")).get("fingerprint")
        if spec is None and not allow:
            raise RuntimeError(f"subset_adapt: {man_path} exists but NESTOR_SUBSET_SPEC is unset -- refusing to "
                               "resume a subset run without its spec (use a fresh NESTOR_OUTPUT_ROOT)")
        if spec is not None and old != fp and not allow:
            raise RuntimeError(f"subset_adapt: spec/training fingerprint changed for {root} ({old} -> {fp}); "
                               "refusing to resume or overwrite (use a fresh NESTOR_OUTPUT_ROOT)")
    elif spec is not None and _has_checkpoints(root) and not allow:
        raise RuntimeError(f"subset_adapt: {root} already holds checkpoints but no subset manifest -- refusing "
                           "to start a subset run that would auto-resume them")


def _optimizer_state_check(manifest: dict[str, Any], ckpt: dict) -> dict[str, Any]:
    """Frozen rows of row-masked banks must have exactly zero optimizer state; frozen tensors no state."""

    out = {"checked": False, "state_rows_checked": 0, "violations": []}
    order = manifest.get("optimizer_param_order")
    opt = ckpt.get("optimizer") if isinstance(ckpt, dict) else None
    if not order or not opt:
        out["reason"] = "no optimizer_param_order in manifest" if not order else "no optimizer state in checkpoint"
        return out
    pos = {n: i for i, n in enumerate(order)}
    state = opt.get("state", {})
    for rec in manifest["params"]:
        n = rec["name"]
        if rec["status"] in ("frozen", "frozen_at_construction"):
            if n in pos:
                out["violations"].append(f"{n}: frozen tensor is held by the optimizer")
            continue
        if rec["status"] != "partial" or n not in pos:
            continue
        st = state.get(pos[n], state.get(str(pos[n]), {}))
        tr = set(rec["trainable_rows"])
        for k in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
            t = st.get(k) if isinstance(st, dict) else None
            if not torch.is_tensor(t):
                continue
            fr = [i for i in range(t.shape[0]) if i not in tr]
            out["state_rows_checked"] += len(fr)
            if fr and bool((t[fr] != 0).any()):
                out["violations"].append(f"{n}.{k}: non-zero state on frozen rows")
    out["checked"] = True
    return out


def verify_checkpoints(init_ckpt: str, final_ckpt: str, manifest: dict[str, Any] | str,
                       *, allow_unchanged: bool = False) -> dict[str, Any]:
    """Bitwise audit of a finished run.

    Compares the ``model`` state dicts of the init checkpoint and of the run's final checkpoint
    using the run's manifest. ``ok`` requires: every frozen tensor and frozen bank row bit-identical,
    every router ``choice_bias`` bit-identical when routers were frozen, NO trainable tensor or row
    left unchanged (unless allow_unchanged), and -- when the manifest recorded the optimizer's
    parameter order -- exactly zero optimizer state on frozen rows and no state for frozen tensors.
    """

    if isinstance(manifest, str):
        manifest = json.load(open(manifest, encoding="utf-8"))
    a = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    b = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    sa, sb = a.get("model", a), b.get("model", b)
    res = {"frozen_tensors_checked": 0, "frozen_rows_checked": 0, "frozen_mismatch": [],
           "trainable_unchanged": [], "trainable_changed_fraction": {}, "buffers_checked": 0,
           "buffer_mismatch": []}
    for rec in manifest["params"]:
        n = rec["name"]
        if n not in sa or n not in sb:
            res["frozen_mismatch"].append(f"{n}: missing")
            continue
        x, y = sa[n], sb[n]
        if rec["status"] in ("frozen", "frozen_at_construction"):
            res["frozen_tensors_checked"] += 1
            if x.dtype != y.dtype or not torch.equal(_bits(x), _bits(y)):
                res["frozen_mismatch"].append(n)
        elif rec["status"] == "partial":
            tr = set(rec["trainable_rows"])
            fr = [i for i in range(x.shape[0]) if i not in tr]
            res["frozen_rows_checked"] += len(fr)
            if not torch.equal(_bits(x[fr]), _bits(y[fr])):
                bad = [i for i in fr if not torch.equal(_bits(x[i]), _bits(y[i]))]
                res["frozen_mismatch"].append(f"{n} rows {bad}")
            idx = sorted(tr)
            res["trainable_changed_fraction"][n] = float((x[idx] != y[idx]).float().mean())
            unchanged = [i for i in idx if torch.equal(_bits(x[i]), _bits(y[i]))]
            if unchanged:
                res["trainable_unchanged"].append(f"{n} rows {unchanged}")
        else:
            frac = float((x != y).float().mean())
            res["trainable_changed_fraction"][n] = frac
            if frac == 0.0:
                res["trainable_unchanged"].append(n)
    if manifest.get("env", {}).get("NESTOR_FREEZE_ROUTERS") == "1":
        for k in sa:
            if k.endswith("choice_bias"):
                res["buffers_checked"] += 1
                if not torch.equal(_bits(sa[k]), _bits(sb[k])):
                    res["buffer_mismatch"].append(k)
    res["optimizer_state"] = _optimizer_state_check(manifest, b)
    res["negative_control"] = manifest.get("negative_control")
    res["allow_unchanged"] = allow_unchanged
    res["ok"] = (not res["frozen_mismatch"] and not res["buffer_mismatch"]
                 and (allow_unchanged or not res["trainable_unchanged"])
                 and not res["optimizer_state"]["violations"])
    return res


def catalog_summary(model: nn.Module) -> dict[str, Any]:
    """Unit counts and element counts per kind (for designing subsets)."""

    entries = build_catalog(model)
    seen: set[Unit] = set()
    units = Counter()
    elems = Counter()
    for e in entries:
        per_row = e.param[0].numel() if e.sliced else e.param.numel()
        for u in e.units:
            if u not in seen:
                seen.add(u)
                units[u.kind] += 1
            elems[u.kind] += per_row
    return {"units": dict(units), "elements": dict(elems), "geometry": model_geometry(model)}


def matched_random_spec(model: nn.Module, kind: str, target_elements: float, *, seed: int = 0,
                        mode: str = "freeze", exclude_dense_tail: bool = True,
                        blocks: Iterable[int] | None = None) -> dict[str, Any]:
    """A spec selecting a seeded random set of ``kind`` atoms whose size best matches a target.

    Atoms: a whole Nestor column (block, column) -- attention + private experts over the block's
    layers -- for ``kind="column"``, else single units of that kind. Atoms are visited in a seeded
    random order and kept when they bring the running total closer to ``target_elements``
    (size-matched controls, e.g. "freeze as many private parameters as the embeddings + head").
    """

    import random

    entries = build_catalog(model)
    geo = model_geometry(model)
    allowed = None if blocks is None else set(blocks)
    size: dict[tuple, int] = defaultdict(int)
    for e in entries:
        per_row = e.param[0].numel() if e.sliced else e.param.numel()
        for u in e.units:
            if kind == "column":
                if u.kind not in ("attn", "priv"):
                    continue
                atom = (u.block, u.column)
            elif u.kind == kind:
                atom = (u.block, u.layer, u.column, u.expert)
            else:
                continue
            if exclude_dense_tail and geo["dense_tail_block"] is not None and u.block == geo["dense_tail_block"]:
                continue
            if allowed is not None and u.block not in allowed:
                continue
            size[atom] += per_row
    atoms = sorted(size)
    random.Random(seed).shuffle(atoms)
    chosen, total = [], 0
    for a in atoms:
        if abs(total + size[a] - target_elements) < abs(total - target_elements):
            chosen.append(a)
            total += size[a]
    filters = []
    for a in sorted(chosen):
        if kind == "column":
            filters.append({"kind": "column", "block": a[0], "column": a[1]})
        else:
            f = {"kind": kind}
            for key, v in zip(("block", "layer", "column", "expert"), a):
                if v is not None:
                    f[key] = v
            filters.append(f)
    return {"mode": mode, "select": filters,
            "_provenance": {"matched_random": kind, "target_elements": int(target_elements),
                            "achieved_elements": int(total), "atoms": len(chosen), "seed": seed}}


def _cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="bitwise audit of a finished run")
    v.add_argument("--init", required=True)
    v.add_argument("--final", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--allow-unchanged", action="store_true", help="do not fail on unchanged trainable tensors/rows")
    v.add_argument("--out", default=None, help="write the full audit JSON here")
    r = sub.add_parser("resolve", help="resolve a spec on a registered experiment (meta device)")
    r.add_argument("experiment")
    r.add_argument("--spec", default=None, help="JSON text or path; omit for the unit catalog only")
    r.add_argument("--out", default=None)
    mt = sub.add_parser("match", help="seeded random spec of a kind, size-matched to a target")
    mt.add_argument("experiment")
    mt.add_argument("--kind", required=True, help="column | priv | pub | moe | attn")
    mt.add_argument("--target", type=float, required=True, help="target number of parameter elements")
    mt.add_argument("--seed", type=int, default=0)
    mt.add_argument("--mode", default="freeze")
    mt.add_argument("--include-dense-tail", action="store_true")
    mt.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        res = verify_checkpoints(args.init, args.final, args.manifest, allow_unchanged=args.allow_unchanged)
        if args.out:
            _write_json(args.out, res)
        print(json.dumps({k: v for k, v in res.items() if k != "trainable_changed_fraction"}, indent=1))
        return 0 if res["ok"] else 1
    os.environ.setdefault("NESTOR_STACKED_EXPERT_BANKS", "1")
    os.environ.setdefault("NESTOR_DENSE_NO_ROUTER", "1")
    import nestor.experiments.pretrain  # noqa: F401
    from nestor.experiments.registry import get_experiment
    from nestor.models import Model

    exp = get_experiment(args.experiment)()
    with torch.device("meta"):
        model = Model.from_config(exp.jobs[0].model)
    if args.cmd == "match":
        spec = matched_random_spec(model, args.kind, args.target, seed=args.seed, mode=args.mode,
                                   exclude_dense_tail=not args.include_dense_tail)
        txt = json.dumps(spec, indent=1)
        if args.out:
            open(args.out, "w", encoding="utf-8").write(txt)
        print(txt)
        return 0
    out: dict[str, Any] = {"catalog": catalog_summary(model)}
    if args.spec:
        plan = apply_subset(model, load_spec(args.spec), write_manifest=args.out)
        out["elements"] = plan.manifest["elements"]
        out["units_trainable"] = len(plan.manifest["units_trainable"])
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
