"""Per-window evaluation of Nestor / MoE checkpoints, paired by window index across runs.

For every evaluated window it records the summed next-token cross-entropy, the target-token count,
optionally the UTF-8 byte count of the target tokens (bits per byte), and a per-window routing
summary: for every routed module the number of tokens selecting it, the sum of its effective gate
and the sum of the squared effective gate. Effective gate = the weight with which the module's output
enters the token's update:
    MoE expert (b,l,e)            g = expert-router gate (renormalized top-k)
    Nestor column (b,c)           g = column-router gate w_c
    Nestor public expert (b,l,e)  g = alpha_public(path gate) * public gate
    Nestor private (b,l,c,e)      g = w_c * alpha_private(path gate) * private (switch) gate
The private pairs are ordered c*Q + e, the row order of stacked_private banks.

Windows are the DclmTokenDataset grid (non-overlapping block-token windows, no EOT remap, the operator's
held-out protocol). Every window carries its grid index, token offset and a 64-bit content hash, so two
runs can be paired window by window and the pairing is verified, not assumed.

Pinned routing: `RoutingPin` records every top-k routing decision (Router.forward and the batched
private-router path) of one model on a window and replays them in a second model on the same window
-- the "adapted weights with the base model's routing" counterfactual.

CLI (GPU or CPU):
  python eval_windows.py run <experiment> <ckpt.pt> --set dclm=data/dclm/dclm_val.bin:2048000 \
        --set code=data/streams/code_eval.bin:2048000 --out evals/<run> [--routing] [--bytes]
        [--select first|even|random] [--seed 0] [--variants ckpt,pinned,selpin,reflect+pinned,reflect+selpin,randdir:0,wdonly,revert:embed_head]
        [--base <base_ckpt.pt>] [--manifest <subset_manifest.json>] [--run-env <run_env.json>]
  python eval_windows.py compare <a.npz> <b.npz>            # paired difference b - a
  python eval_windows.py forget <before_dir> <after_dir>    # per set: F = CE_after - CE_before (+ SE)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ("column", "moe", "pub", "priv")


# ============================================================================= windows
def grid_size(num_tokens: int, block: int) -> int:
    return int(num_tokens) // int(block)


def select_windows(num_tokens: int, block: int, n: int, mode: str = "first", seed: int = 0,
                   span: tuple[int, int] | None = None) -> np.ndarray:
    """Grid indices of the windows to evaluate (sorted, unique).

    span=(t0, t1): only grid windows lying entirely inside tokens [t0, t1) are eligible (e.g. the
    selection / evaluation halves of a test bin, prereg sec. 4.3); the grid stays aligned to token 0, so
    'first', 'even' and 'random' then refer to the eligible windows. span=None keeps the v1 behaviour."""

    total = grid_size(num_tokens, block)
    if total == 0:
        raise ValueError(f"{num_tokens} tokens < one block of {block}")
    lo, hi = 0, total
    if span is not None:
        t0, t1 = int(span[0]), min(int(span[1]), int(num_tokens))
        lo, hi = -(-t0 // int(block)), min(t1 // int(block), total)
        if hi <= lo:
            raise ValueError(f"span {span} holds no full window of {block} tokens")
    m = hi - lo
    n = min(int(n), m)
    if mode == "first":            # eval_ckpt.py / run_continual.py protocol
        idx = np.arange(n)
    elif mode == "even":           # routing_stats.py protocol
        idx = np.unique(np.linspace(0, m - 1, n).round().astype(np.int64))
    elif mode == "random":
        idx = np.sort(np.random.default_rng(seed).choice(m, size=n, replace=False))
    else:
        raise ValueError(mode)
    return (idx + lo).astype(np.int64)


def split_point(toks: np.ndarray, frac: float, eot: int | None = None, align: str | None = None,
                chunk: int = 1 << 22) -> int:
    """Token index splitting a bin at fraction `frac` (0.5 = the token midpoint, prereg sec. 4.3).

    align='doc': moved forward to the next document boundary (the position just after an end-of-text
    token), so no document straddles the split -- PG-19, whose split unit is the book."""

    n = int(len(toks))
    if frac <= 0:
        return 0
    if frac >= 1:
        return n
    p = int(n * float(frac)) if float(frac) != 0.5 else n // 2
    if align in (None, "", "token"):
        return p
    if align != "doc" or eot is None:
        raise ValueError(f"unknown split alignment {align!r} (eot={eot})")
    if p > 0 and int(toks[p - 1]) == int(eot):
        return p
    s = p
    while s < n:
        e = min(n, s + chunk)
        hit = np.flatnonzero(np.asarray(toks[s:e]) == eot)
        if hit.size:
            return int(s + hit[0] + 1)
        s = e
    return n


def split_halves(n: int, seed: int = 0) -> np.ndarray:
    """Balanced seeded assignment of n windows to halves 0/1 (cross-fitting)."""

    perm = np.random.default_rng(seed).permutation(n)
    half = np.empty(n, dtype=np.int8)
    half[perm] = np.arange(n) % 2
    return half


def window_hash(tokens: np.ndarray) -> int:
    return int.from_bytes(hashlib.blake2b(np.ascontiguousarray(tokens, dtype=np.int64).tobytes(),
                                          digest_size=8).digest(), "little")


def token_byte_lengths(vocab_size: int, encoding: str = "cl100k_base") -> np.ndarray | None:
    """UTF-8 byte length of every token id (special / unknown ids -> 0); None without tiktoken."""

    try:
        import tiktoken
    except Exception:
        return None
    enc = tiktoken.get_encoding(encoding)
    out = np.zeros(vocab_size, dtype=np.int32)
    for i in range(vocab_size):
        try:
            out[i] = len(enc.decode_single_token_bytes(i))
        except Exception:
            out[i] = 0
    if enc.eot_token < vocab_size:
        out[enc.eot_token] = 0
    return out


# ============================================================================= routing summaries
def _alpha_from_logits(logits):
    import torch

    return torch.softmax(logits.float(), dim=-1)


class RoutingSummary:
    """Per-forward effective-gate statistics, keyed by family-layer ('moe:b0:l1', 'priv:b2:l0', ...)."""

    def __init__(self, model):
        import re

        self.path: dict[tuple[int, int], Any] = {}
        self.handles = []
        pat = re.compile(r"^backbone\.blocks\.(\d+)\.layers\.(\d+)\.expert_pool\.path_gate$")
        for name, mod in model.named_modules():
            m = pat.match(name)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                self.handles.append(mod.register_forward_hook(
                    lambda _m, _i, out, key=key: self.path.__setitem__(key, out.detach())))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def _stats(gates, active, valid):
        v = valid.unsqueeze(-1).to(gates.dtype)
        g = gates.float() * v
        return (active.float() * v).sum(dim=(0, 1)), g.sum(dim=(0, 1)), (g * g).sum(dim=(0, 1))

    def collect(self, aux: dict) -> dict[str, tuple]:
        """{family-layer key: (counts[A], sum_g[A], sum_g2[A])} for one forward (device tensors)."""

        import torch

        out: dict[str, tuple] = {}
        for b, blk in enumerate(aux.get("blocks", []) or []):
            if not isinstance(blk, dict):
                continue
            valid_b = blk.get("valid_token_mask")
            gr = blk.get("group_router")
            colg = None
            if isinstance(gr, dict) and gr["gates"].shape[-1] > 1:
                colg = gr["gates"].float()
                out[f"column:b{b}"] = self._stats(gr["gates"], gr["active_mask"], gr["valid_token_mask"])
            for l, lay in enumerate(blk.get("layers", []) or []):
                if not isinstance(lay, dict):
                    continue
                er = lay.get("expert_router")
                pool = "expert_pool_private" in lay
                alpha = self.path.get((b, l))
                alpha = _alpha_from_logits(alpha) if alpha is not None else None
                if isinstance(er, dict):
                    g = er["gates"].float()
                    if pool and alpha is not None:
                        g = g * alpha[..., 0:1].reshape(*g.shape[:-1], 1)
                    out[f"{'pub' if pool else 'moe'}:b{b}:l{l}"] = self._stats(g, er["active_mask"], er["valid_token_mask"])
                if pool:
                    cnt, s1, s2 = [], [], []
                    for c, pc in enumerate(lay["expert_pool_private"]):
                        g = pc["gates"].float()
                        if alpha is not None:
                            g = g * alpha[..., 1:2].reshape(*g.shape[:-1], 1)
                        if colg is not None:
                            g = g * colg[..., c:c + 1]
                        st = self._stats(g, pc["active_mask"], pc["valid_token_mask"])
                        cnt.append(st[0]); s1.append(st[1]); s2.append(st[2])
                    out[f"priv:b{b}:l{l}"] = (torch.cat(cnt), torch.cat(s1), torch.cat(s2))
        self.path.clear()
        return out


def module_keys(layer_key: str, n: int, experts_per_column: int | None = None) -> list[str]:
    """Unit keys (subset_adapt format) of the choices of one family-layer summary."""

    fam, rest = layer_key.split(":", 1)
    if fam == "column":
        return [f"column:{rest}:c{c}" for c in range(n)]
    if fam == "priv":
        q = experts_per_column
        return [f"priv:{rest}:c{a // q}:e{a % q}" for a in range(n)]
    return [f"{fam}:{rest}:e{a}" for a in range(n)]


# ============================================================================= pinned routing
class RoutingPin:
    """Record the routing of one forward and replay it in another model (same window).

    Patches nestor.backbones.nestor.Router.forward and the batched private-router function that
    nestor.backbones.expert_pools calls. Each recorded decision carries the router's module name; a
    replay checks that the replaying module has the same name and the outputs the same shapes.
    """

    def __init__(self, pin_path_gate: bool = False, select_only: bool = False):
        """select_only=True ('selpin', theory verification): replay theta_0's top-k SELECTIONS only; gate
        values are recomputed from the replaying model's own (live) router scores with the router's gate rule.
        Default (False, 'pinned'): selections AND gate values are theta_0's."""
        self.select_only = select_only
        self.mode: str | None = None
        self.buf: list = []
        self.pos = 0
        self.pin_path_gate = pin_path_gate
        self.names: dict[int, str] = {}
        self._installed = False
        self._hooks = []

    def install(self, *models):
        import nestor.backbones.expert_pools as ep
        import nestor.backbones.nestor as nb

        for mdl in models:
            for name, mod in mdl.named_modules():
                self.names[id(mod)] = name.replace("_orig_mod.", "")
                if self.pin_path_gate and name.endswith("expert_pool.path_gate"):
                    self._hooks.append(mod.register_forward_hook(self._path_hook))
        if self._installed:
            return self
        other = getattr(nb.Router.forward, "_routing_pin", None)
        if other is not None and other is not self:
            raise RuntimeError("pinned routing: another RoutingPin is installed; uninstall it first")
        pin = self
        self._orig_forward = nb.Router.forward
        self._orig_batched = ep.route_private_batched

        def forward(router, x, valid_token_mask, noise=None):
            if pin.mode is None:
                return pin._orig_forward(router, x, valid_token_mask, noise)
            name = pin.names.get(id(router), "?")
            if pin.mode == "record":
                out = pin._orig_forward(router, x, valid_token_mask, noise)
                pin.buf.append(("router", name, tuple(t.detach().clone() for t in out)))
                return out
            kind, rname, saved = pin._next("router", name)
            if pin.select_only:
                live = pin._orig_forward(router, x, valid_token_mask, noise)
                return selpin_router_output(router.gate_mode, live, nb.RouterOutput(*saved))
            return nb.RouterOutput(*saved)

        def batched(routers, x, valid_token_mask, group_active_mask, training, noise=None):
            if pin.mode is None:
                return pin._orig_batched(routers, x, valid_token_mask, group_active_mask, training, noise)
            name = pin.names.get(id(routers[0]), "?")
            if pin.mode == "record":
                outs, gates, choices = pin._orig_batched(routers, x, valid_token_mask, group_active_mask, training, noise)
                pin.buf.append(("batched", name, ([tuple(t.detach().clone() for t in o) for o in outs],
                                                   gates.detach().clone(), choices.detach().clone())))
                return outs, gates, choices
            _, _, (outs, gates, choices) = pin._next("batched", name)
            if pin.select_only:
                live_outs, _, _ = pin._orig_batched(routers, x, valid_token_mask, group_active_mask, training, noise)
                return selpin_batched(routers[0].gate_mode, live_outs, [nb.RouterOutput(*o) for o in outs], choices)
            return [nb.RouterOutput(*o) for o in outs], gates, choices

        forward._routing_pin = self
        batched._routing_pin = self
        nb.Router.forward = forward
        ep.route_private_batched = batched
        self._installed = True
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.uninstall()
        return False

    def _path_hook(self, mod, inp, out):
        if self.mode is None:
            return None
        name = self.names.get(id(mod), "?")
        if self.mode == "record":
            self.buf.append(("path", name, out.detach().clone()))
            return None
        _, _, saved = self._next("path", name)
        return saved

    def _next(self, kind: str, name: str):
        if self.pos >= len(self.buf):
            raise RuntimeError(f"pinned routing: replay asked for more decisions than recorded ({kind} {name})")
        rec = self.buf[self.pos]
        self.pos += 1
        if rec[0] != kind or rec[1] != name:
            raise RuntimeError(f"pinned routing: call order mismatch: recorded {rec[0]} {rec[1]}, replaying {kind} {name}")
        return rec

    def record(self):
        self.mode, self.buf, self.pos = "record", [], 0

    def replay(self):
        self.mode, self.pos = "replay", 0

    def done(self):
        if self.mode == "replay" and self.pos != len(self.buf):
            raise RuntimeError(f"pinned routing: {len(self.buf) - self.pos} recorded decisions were not replayed")
        self.mode = None

    def decisions(self) -> list:
        return self.buf

    def uninstall(self):
        import nestor.backbones.expert_pools as ep
        import nestor.backbones.nestor as nb

        if self._installed:
            nb.Router.forward = self._orig_forward
            ep.route_private_batched = self._orig_batched
            self._installed = False
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.mode = None


def selpin_gates(scores, topk_indices, gate_mode: str, mask):
    """The authors' gate rule (nestor Router.forward / route_private_batched) on given scores and FIXED selections:
    'switch' = full-softmax probability of each selected choice, 'renorm' = softmax over the selected scores;
    scattered to the choice axis and masked."""

    import torch
    import torch.nn.functional as F

    if gate_mode == "switch":
        tg = F.softmax(scores.float(), dim=-1).gather(dim=-1, index=topk_indices).to(scores.dtype)
    else:
        tg = F.softmax(scores.gather(dim=-1, index=topk_indices).float(), dim=-1).to(scores.dtype)
    g = torch.zeros_like(scores).scatter_add_(-1, topk_indices, tg)
    return g * mask.unsqueeze(-1).to(dtype=g.dtype)


def selpin_router_output(gate_mode: str, live, saved):
    """Selection-pinned RouterOutput: theta_0's top-k indices, gate values from the live scores."""

    import nestor.backbones.nestor as nb

    g = selpin_gates(live.scores, saved.topk_indices, gate_mode, live.valid_token_mask)
    return nb.RouterOutput(scores=live.scores, softmax_probs=live.softmax_probs, gates=g, topk_indices=saved.topk_indices,
                           active_mask=g > 0, valid_token_mask=live.valid_token_mask)


def selpin_batched(gate_mode: str, live_outs: list, saved_outs: list, saved_choices):
    """Selection-pinned batched private routing (route_private_batched layout [B, S, G, Q]): theta_0's choices and
    group membership, gate values from the live per-group scores."""

    import torch

    import nestor.backbones.nestor as nb

    scores = torch.stack([o.scores for o in live_outs], dim=2)
    probs = torch.stack([o.softmax_probs for o in live_outs], dim=2)
    member = torch.stack([o.valid_token_mask for o in saved_outs], dim=2)
    g = selpin_gates(scores, saved_choices, gate_mode, member)
    active = g > 0
    outs = [nb.RouterOutput(scores=scores[:, :, i], softmax_probs=probs[:, :, i], gates=g[:, :, i],
                            topk_indices=saved_choices[:, :, i], active_mask=active[:, :, i], valid_token_mask=member[:, :, i])
            for i in range(len(live_outs))]
    return outs, g, saved_choices


def _family_of(router_name: str) -> str:
    if router_name.endswith("group_router"):
        return "column"
    if "public_router" in router_name:
        return "pub"
    if "private_routers" in router_name:
        return "priv"
    return "moe"


def decision_flips(src: list, own: list) -> dict[str, tuple[int, int]]:
    """Top-k set flips of `own` routing relative to `src` (both recorded on the same window).

    Returns {family: (flipped, total)}. Router decisions count valid tokens; batched private decisions
    count (token, column) positions where the column is active (non-zero gate row) in both models.
    Path-gate records are ignored.
    """

    a_list = [d for d in src if d[0] in ("router", "batched")]
    b_list = [d for d in own if d[0] in ("router", "batched")]
    if len(a_list) != len(b_list):
        raise RuntimeError(f"routing flips: {len(a_list)} vs {len(b_list)} router decisions")
    out: dict[str, tuple[int, int]] = {}
    for (k1, n1, p1), (k2, n2, p2) in zip(a_list, b_list):
        if (k1, n1) != (k2, n2):
            raise RuntimeError(f"routing flips: decision mismatch {k1} {n1} vs {k2} {n2}")
        if k1 == "router":
            a, b = p1[3].sort(-1).values, p2[3].sort(-1).values
            keep = p1[5] & p2[5]
        else:
            a, b = p1[2].sort(-1).values, p2[2].sort(-1).values
            keep = (p1[1].sum(-1) > 0) & (p2[1].sum(-1) > 0)
        diff = (a != b).any(-1) & keep
        f0, t0 = out.get(_family_of(n1), (0, 0))
        out[_family_of(n1)] = (f0 + int(diff.sum()), t0 + int(keep.sum()))
    return out


def topk_of(decisions: list) -> list:
    """Top-k index tensors of a recorded decision list (for flip-rate comparisons)."""

    out = []
    for kind, name, payload in decisions:
        if kind == "router":
            out.append((name, payload[3]))
        elif kind == "batched":
            out.append((name, payload[2]))
    return out


# ============================================================================= evaluation core
def forward_ce(model, x, *, bf16: bool = False):
    """(ce_sum over positions 1..S-1, n_targets, aux) for one [1, S] window."""

    import torch

    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
        out = model(x)
    logits = out.pred[:, :-1, :].reshape(-1, out.pred.shape[-1]).float()
    tgt = x[:, 1:].reshape(-1)
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
    return float(ce), int(tgt.numel()), out.aux


class WindowSet:
    """The windows of one evaluation set (file, block, grid indices) with their tokens and hashes."""

    def __init__(self, name: str, path: str, block: int, n_windows: int, mode: str = "first", seed: int = 0,
                 indices: np.ndarray | None = None, span: tuple[float, float] | None = None,
                 span_align: str | None = None):
        """span=(f0, f1): fractions of the file (e.g. (0, 0.5) = selection half, (0.5, 1) = evaluation half,
        split at the token midpoint; span_align='doc' moves split points to document boundaries)."""
        from nestor.data.token_documents import open_token_bin

        self.name, self.path, self.block = name, os.path.abspath(path), int(block)
        self.toks, self.eot = open_token_bin(path)
        self.file_tokens = int(len(self.toks))
        self.span, self.span_align, self.span_tokens = None, span_align, None
        if span is not None:
            f0, f1 = float(span[0]), float(span[1])
            self.span = [f0, f1]
            self.span_tokens = [split_point(self.toks, f0, self.eot, span_align),
                                split_point(self.toks, f1, self.eot, span_align)]
        self.index = (np.asarray(indices, dtype=np.int64) if indices is not None
                      else select_windows(self.file_tokens, block, n_windows, mode, seed, span=self.span_tokens))
        self.mode, self.seed = mode, seed
        self.starts = self.index * self.block
        self._hashes = None

    def tokens(self, i: int) -> np.ndarray:
        s = int(self.starts[i])
        return np.asarray(self.toks[s: s + self.block], dtype=np.int64)

    @property
    def hashes(self) -> np.ndarray:
        if self._hashes is None:
            self._hashes = np.array([window_hash(self.tokens(i)) for i in range(len(self.index))], dtype=np.uint64)
        return self._hashes

    def head(self, n: int) -> "WindowSet":
        """The first n windows (a paired subset, e.g. for interim evaluations)."""

        w = WindowSet.__new__(WindowSet)
        w.__dict__.update(self.__dict__)
        w.index, w.starts = self.index[:n], self.starts[:n]
        w._hashes = None if self._hashes is None else self._hashes[:n]
        return w

    def describe(self) -> dict:
        d = {"name": self.name, "path": self.path, "block": self.block, "file_tokens": self.file_tokens,
             "windows": int(len(self.index)), "select": self.mode, "seed": self.seed}
        if getattr(self, "span", None) is not None:
            d.update(span=self.span, span_tokens=self.span_tokens, span_align=self.span_align)
        return d


def evaluate(model, wset: WindowSet, *, device: str = "cpu", routing: bool = False, byte_len: np.ndarray | None = None,
             bf16: bool = False, pin: RoutingPin | None = None, pin_source=None, log: Callable = print,
             max_windows: int | None = None) -> dict[str, Any]:
    """Per-window CE (and routing summaries) of `model` on `wset`.

    With `pin` and `pin_source`, every window is first run through `pin_source` (recording its routing)
    and then through `model` with that routing replayed (pinned-routing counterfactual). With routing=True
    as well, `model` is run a third time unpinned and the result carries per-window top-k flip counts of
    its own routing relative to `pin_source`'s: res["flip_<family>"] = int64 [n_windows, 2] (flipped,
    total), family in column/pub/priv/moe (see decision_flips). In pinned mode the routing summaries
    are those of the replayed decisions (path gates are the model's own unless pin_path_gate=True).
    """

    import torch

    model.eval()
    n = len(wset.index) if max_windows is None else min(max_windows, len(wset.index))
    ce = np.zeros(n); nt = np.zeros(n, dtype=np.int64)
    nbytes = np.zeros(n, dtype=np.int64) if byte_len is not None else None
    summ = RoutingSummary(model) if routing else None
    rt: dict[str, list] = {}
    flips: dict[str, np.ndarray] = {}
    t0 = time.time()
    try:
        with torch.no_grad():
            for i in range(n):
                arr = wset.tokens(i)
                x = torch.as_tensor(arr).view(1, -1).to(device)
                if pin is not None:
                    pin.record()
                    forward_ce(pin_source, x, bf16=bf16)
                    src_dec = pin.decisions()
                    pin.replay()
                    c, k, aux = forward_ce(model, x, bf16=bf16)
                    pin.done()
                    if routing:
                        pin.record()
                        forward_ce(model, x, bf16=bf16)
                        own_dec = pin.decisions()
                        pin.mode = None
                        for fam, ft in decision_flips(src_dec, own_dec).items():
                            flips.setdefault(fam, np.zeros((n, 2), dtype=np.int64))[i] = ft
                else:
                    c, k, aux = forward_ce(model, x, bf16=bf16)
                ce[i], nt[i] = c, k
                if i == 0:
                    import memguard as _MG

                    _MG.check(device, f"evaluate {wset.name}: first window")
                if nbytes is not None:
                    nbytes[i] = int(byte_len[np.clip(arr[1:], 0, len(byte_len) - 1)].sum())
                if summ is not None:
                    for key, (a, b1, b2) in summ.collect(aux).items():
                        rt.setdefault(key, []).append(torch.stack([a, b1, b2]).double().cpu().numpy())
                if log and (i + 1) % 100 == 0:
                    log(f"[eval_windows] {wset.name}: {i + 1}/{n} windows, {time.time() - t0:.0f}s")
    finally:
        if summ is not None:
            summ.close()
    res = {"index": wset.index[:n], "starts": wset.starts[:n], "hash": wset.hashes[:n], "ce_sum": ce, "n_tok": nt,
           "seconds": time.time() - t0}
    if nbytes is not None:
        res["n_bytes"] = nbytes
    if routing:
        res["routing"] = {k: np.stack(v) for k, v in rt.items()}      # [n_windows, 3, A]
    for fam, arr in flips.items():
        res[f"flip_{fam}"] = arr
    return res


PIN_VARIANTS = ("pinned", "selpin", "reflect+pinned", "reflect+selpin")


def evaluate_pinned_multi(pin_source, targets: dict, wset: WindowSet, *, device: str = "cpu", bf16: bool = False,
                          log: Callable = print, max_windows: int | None = None) -> dict[str, dict]:
    """Several pinned replays per window with ONE recording of theta_0's routing (theory verification).

    targets = {variant: (model, select_only)}; e.g. pinned -> (theta_0 + delta, False), selpin -> (theta_0 + delta, True),
    reflect+pinned -> (theta_0 - delta, False), reflect+selpin -> (theta_0 - delta, True). Per window: one forward of
    `pin_source` (recording) and one replayed forward per target, i.e. 1 + len(targets) forwards instead of
    2 x len(targets). Every replay must consume exactly the recorded decisions. Returns {variant: result}."""

    import torch

    n = len(wset.index) if max_windows is None else min(max_windows, len(wset.index))
    ce = {v: np.zeros(n) for v in targets}
    nt = {v: np.zeros(n, dtype=np.int64) for v in targets}
    rec = RoutingPin()
    mods = [pin_source] + [m for m, _ in targets.values()]
    for m in mods:
        m.eval()
    t0 = time.time()
    try:
        rec.install(*mods)
        with torch.no_grad():
            for i in range(n):
                x = torch.as_tensor(wset.tokens(i)).view(1, -1).to(device)
                rec.record()
                forward_ce(pin_source, x, bf16=bf16)
                buf = rec.decisions()
                for v, (m, so) in targets.items():
                    rec.select_only = so
                    rec.buf, rec.mode, rec.pos = buf, "replay", 0
                    c, k, _ = forward_ce(m, x, bf16=bf16)
                    rec.done()
                    ce[v][i], nt[v][i] = c, k
                rec.select_only = False
                if i == 0:
                    import memguard as _MG

                    _MG.check(device, f"pinned x{len(targets)} {wset.name}: first window")
                if log and (i + 1) % 100 == 0:
                    log(f"[eval_windows] {wset.name} pinned x{len(targets)}: {i + 1}/{n} windows, {time.time() - t0:.0f}s")
    finally:
        rec.uninstall()
    sec = time.time() - t0
    return {v: {"index": wset.index[:n], "starts": wset.starts[:n], "hash": wset.hashes[:n], "ce_sum": ce[v], "n_tok": nt[v],
                "seconds": sec} for v in targets}


def save_result(path: str, res: dict, meta: dict) -> str:
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    for k, v in (res.get("routing") or {}).items():
        arrays[f"rt|{k}"] = v.astype(np.float64)
    arrays["meta_json"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


def load_result(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if not k.startswith("rt|") and k != "meta_json"}
    out["routing"] = {k[3:]: z[k] for k in z.files if k.startswith("rt|")}
    out["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return out


# ============================================================================= paired statistics
def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation without numpy BLAS: with a pip torch (Intel OpenMP) next to a conda numpy whose
    BLAS uses LLVM OpenMP, the first numpy BLAS call after `import torch` aborts the process (OMP: Error #15)."""

    xc, yc = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    return float((xc * yc).sum()) / den if den > 0 else float("nan")


def paired(a: dict, b: dict, *, check: bool = True) -> dict[str, float]:
    """Paired per-window difference b - a of mean CE per token (nats), with a window-level SE."""

    ia, ib = {int(i): k for k, i in enumerate(a["index"])}, {int(i): k for k, i in enumerate(b["index"])}
    common = sorted(set(ia) & set(ib))
    if not common:
        raise ValueError("no common windows")
    ka, kb = np.array([ia[i] for i in common]), np.array([ib[i] for i in common])
    if check and not np.array_equal(a["hash"][ka], b["hash"][kb]):
        bad = int((a["hash"][ka] != b["hash"][kb]).sum())
        raise ValueError(f"{bad} paired windows have different token content (different file?)")
    ma = a["ce_sum"][ka] / a["n_tok"][ka]
    mb = b["ce_sum"][kb] / b["n_tok"][kb]
    d = mb - ma
    n = len(d)
    tok_a, tok_b = a["n_tok"][ka].sum(), b["n_tok"][kb].sum()
    res = {"n_windows": n, "ce_a": float(a["ce_sum"][ka].sum() / tok_a), "ce_b": float(b["ce_sum"][kb].sum() / tok_b),
           "diff": float(b["ce_sum"][kb].sum() / tok_b - a["ce_sum"][ka].sum() / tok_a),
           "diff_window_mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
           "corr": _corr(ma, mb) if n > 2 else float("nan"),
           "max_abs_window_diff": float(np.abs(d).max())}
    res["z"] = res["diff"] / res["se"] if res["se"] and res["se"] > 0 else float("nan")
    if "n_bytes" in a and "n_bytes" in b:
        by = a["n_bytes"][ka].sum()
        res["diff_bits_per_byte"] = float((b["ce_sum"][kb].sum() - a["ce_sum"][ka].sum()) / math.log(2) / by) if by else float("nan")
    return res


def forget_table(before_dir: str, after_dir: str, variant_before: str = "ckpt", variant_after: str = "ckpt") -> dict:
    """{set: paired(before, after)} for every set present in both directories."""

    out = {}
    for f in sorted(os.listdir(before_dir)):
        if not f.startswith(variant_before + "__") or not f.endswith(".npz"):
            continue
        s = f[len(variant_before) + 2:-4]
        g = os.path.join(after_dir, f"{variant_after}__{s}.npz")
        if os.path.exists(g):
            out[s] = paired(load_result(os.path.join(before_dir, f)), load_result(g))
    return out


# ============================================================================= model loading
def nestor_env() -> None:
    """The training-time construction flags (run_exp.SPEED_STACK), compile off, compat shims."""

    for p in (HERE, os.path.join(HERE, "repo")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import run_exp

        flags = dict(run_exp.SPEED_STACK)
    except Exception:
        flags = {"NESTOR_STACKED_EXPERT_BANKS": "1", "NESTOR_DENSE_NO_ROUTER": "1", "NESTOR_BATCHED_ROUTERS": "1",
                 "NESTOR_COLUMN_SLOT_UPDATES": "1", "NESTOR_DENSE_TAIL_ATTN": "1", "NESTOR_FUSED_HEAD_CE": "1"}
    for k, v in flags.items():
        os.environ.setdefault(k, v)
    os.environ["NESTOR_COMPILE"] = "0"
    try:
        import compat

        compat.install()
    except Exception:
        pass


def load_model(experiment: str, checkpoint: str, device: str):
    nestor_env()
    import nestor.experiments.pretrain  # noqa: F401
    from nestor.eval.checkpoint import load_experiment_checkpoint

    model, info = load_experiment_checkpoint(experiment, checkpoint, device=device)
    return model, info


def register_extra(path: str | None) -> None:
    """Import a python file that registers extra experiments (tests: tiny models)."""

    if not path:
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("extra_experiments", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def file_fingerprint(path: str) -> dict:
    st = os.stat(path)
    with open(path, "rb") as fh:
        head = fh.read(1 << 20)
    return {"path": os.path.abspath(path), "size": st.st_size, "head_sha256": hashlib.sha256(head).hexdigest()}


# ============================================================================= CLI
def _parse_set(s: str):
    name, rest = s.split("=", 1)
    path, _, tok = rest.rpartition(":")
    if not path or not tok.isdigit():          # no token budget given (or a Windows drive colon)
        path, tok = rest, "2048000"
    return name, path, int(tok)


def run_cli(args) -> int:
    import torch

    register_extra(args.register)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    import weight_controls as WC

    variants = [v for v in (args.variants or "ckpt").split(",") if v]
    needs_base = any(v.split(":")[0] in ("pinned", "selpin", "reflect+pinned", "reflect+selpin", "randdir", "randdir-", "reflect",
                                         "revert", "wdonly", "base") for v in variants)
    # theory verification: selection-only pin and pinned replay at theta_0 - delta. Any of the new pin variants
    # sends every plain pin variant through ONE recording of theta_0's routing per window (evaluate_pinned_multi)
    joint = [v for v in variants if v in PIN_VARIANTS]
    if not any(v in ("selpin", "reflect+pinned", "reflect+selpin") for v in joint):
        joint = []
    variants = [v for v in variants if v not in joint]
    model, info = load_model(args.experiment, args.checkpoint, device)
    base = None
    if needs_base:
        if not args.base:
            raise SystemExit("variants pinned/selpin/reflect+pinned/reflect+selpin/randdir/randdir-/reflect/revert/wdonly/base "
                             "need --base <pretrained checkpoint>")
        base, _ = load_model(args.experiment, args.base, device)
    import memguard as MG

    MG.check(device, "eval_windows: models loaded")
    byte_len = token_byte_lengths(model.adapter.vocab_size) if args.bytes else None
    spans = {}
    for sp in getattr(args, "span", None) or []:       # name=f0,f1[,doc]
        nm, _, rest = sp.partition("=")
        parts = rest.split(",")
        spans[nm] = ((float(parts[0]), float(parts[1])), parts[2] if len(parts) > 2 else None)
    sets = []
    for s in args.set:
        name, path, tok = _parse_set(s)
        sp = spans.get(name)
        sets.append(WindowSet(name, path, args.block, -(-tok // args.block), args.select, args.seed,
                              span=sp[0] if sp else None, span_align=sp[1] if sp else None))
    run_env = json.load(open(args.run_env, encoding="utf-8")) if args.run_env else {}
    meta0 = {"experiment": args.experiment, "checkpoint": file_fingerprint(args.checkpoint),
             "base": file_fingerprint(args.base) if args.base else None, "step": info.get("step"),
             "precision": "bf16" if args.bf16 else "fp32", "device": device, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    summary = {}
    # variant states are built on the HOST and loaded in place: no extra full-model copies on the device (host fact:
    # WDDM pages oversubscribed VRAM silently); the elementwise state arithmetic is identical on CPU
    adapted_state = MG.host_copy(model.state_dict()) if needs_base else None
    base_state_host = MG.host_copy(base.state_dict()) if base is not None else None
    for v in variants:
        kind, _, arg = v.partition(":")
        pin, src = None, None
        if kind in ("ckpt", "adapted"):
            target = model
        elif kind == "base":
            target = base
        elif kind == "pinned":
            pin, src, target = RoutingPin(pin_path_gate=(arg == "path")).install(model, base), base, model
        else:
            target = model
            base_state = base_state_host
            if kind == "randdir":
                new = WC.random_direction_state(base_state, adapted_state, seed=int(arg or 0))
            elif kind == "randdir-":                     # antithetic partner of randdir:<seed>
                new = WC.random_direction_state(base_state, adapted_state, seed=int(arg or 0), sign=-1)
            elif kind == "reflect":                      # theta_0 - delta
                new = WC.reflect_state(base_state, adapted_state)
            elif kind == "revert":
                new = WC.revert_state(model, adapted_state, base_state, arg.split("+"))  # e.g. revert:embed_head+pub
            elif kind == "wdonly":
                if not (args.manifest and run_env):
                    raise SystemExit("wdonly needs --manifest and --run-env")
                new = WC.wd_only_state(args.experiment, base_state, args.manifest, run_env)
            else:
                raise SystemExit(f"unknown variant {v}")
            target.load_state_dict(new, strict=False)
            del new
            MG.check(device, f"eval_windows: variant {v} loaded")
        for ws in sets:
            res = evaluate(target, ws, device=device, routing=args.routing, byte_len=byte_len, bf16=args.bf16,
                           pin=pin, pin_source=src)
            meta = dict(meta0, variant=v, set=ws.describe())
            out = save_result(os.path.join(args.out, f"{v.replace(':', '-')}__{ws.name}.npz"), res, meta)
            summary[f"{v}|{ws.name}"] = {"ce": float(res["ce_sum"].sum() / res["n_tok"].sum()),
                                        "windows": int(len(res["ce_sum"])), "seconds": round(res["seconds"], 1), "file": out}
            print(f"RESULT {json.dumps({'variant': v, 'set': ws.name, **summary[f'{v}|{ws.name}']})}", flush=True)
        if pin is not None:
            pin.uninstall()
        if kind not in ("ckpt", "adapted", "base", "pinned"):
            model.load_state_dict(adapted_state)        # every weight-space variant starts from the adapted state
    if joint:
        targets = {}
        model_r = None
        if any(v.startswith("reflect+") for v in joint):
            MG.reserve(device, MG.module_bytes(model), "joint pinned evaluation: reflected model copy")
            model_r, _ = load_model(args.experiment, args.checkpoint, device)    # the authors' model is not deep-copyable
            model_r.load_state_dict(WC.reflect_state(base_state_host, adapted_state), strict=False)
            MG.check(device, "joint pinned evaluation: reflected model loaded")
        for v in joint:
            m = model_r if v.startswith("reflect+") else model
            targets[v] = (m, v.endswith("selpin"))
        for ws in sets:
            res_all = evaluate_pinned_multi(base, targets, ws, device=device, bf16=args.bf16)
            for v, res in res_all.items():
                meta = dict(meta0, variant=v, set=ws.describe(), weights="theta0-delta" if v.startswith("reflect+") else "theta0+delta",
                            pin="selections" if v.endswith("selpin") else "selections+gates", pin_source="base (theta_0)")
                out = save_result(os.path.join(args.out, f"{v}__{ws.name}.npz"), res, meta)
                summary[f"{v}|{ws.name}"] = {"ce": float(res["ce_sum"].sum() / res["n_tok"].sum()),
                                            "windows": int(len(res["ce_sum"])), "seconds": round(res["seconds"], 1), "file": out}
                print(f"RESULT {json.dumps({'variant': v, 'set': ws.name, **summary[f'{v}|{ws.name}']})}", flush=True)
        del model_r
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta0, "results": summary}, fh, indent=1)
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment"); r.add_argument("checkpoint")
    r.add_argument("--set", action="append", required=True, help="name=path.bin[:tokens]")
    r.add_argument("--out", required=True)
    r.add_argument("--block", type=int, default=4096)
    r.add_argument("--select", default="first", choices=["first", "even", "random"])
    r.add_argument("--span", action="append", default=[],
                   help="name=f0,f1[,doc]: windows of set <name> only inside that fraction of the file "
                        "(prereg sec. 4.3: 0,0.5 = selection half, 0.5,1 = evaluation half; doc = split at a document boundary)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--routing", action="store_true")
    r.add_argument("--bytes", action="store_true")
    r.add_argument("--bf16", action="store_true")
    r.add_argument("--device", default=None)
    r.add_argument("--variants", default="ckpt",
                   help="comma list: ckpt, base, pinned[:path], selpin (theta_0 selections, live gate values), "
                        "reflect+pinned, reflect+selpin (pinned replays at theta_0 - delta), reflect (theta_0 - delta), randdir:<seed>, "
                        "randdir-:<seed> (antithetic partner), revert:<g1+g2>, wdonly")
    r.add_argument("--base", default=None)
    r.add_argument("--manifest", default=None)
    r.add_argument("--run-env", default=None)
    r.add_argument("--register", default=None, help=argparse.SUPPRESS)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    f = sub.add_parser("forget"); f.add_argument("before"); f.add_argument("after")
    f.add_argument("--before-variant", default="ckpt"); f.add_argument("--after-variant", default="ckpt")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run_cli(args)
    if args.cmd == "compare":
        print(json.dumps(paired(load_result(args.a), load_result(args.b)), indent=1))
        return 0
    print(json.dumps(forget_table(args.before, args.after, args.before_variant, args.after_variant), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
