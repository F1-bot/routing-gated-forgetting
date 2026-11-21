"""Routing statistics of a Nestor / MoE checkpoint over a set of corpora (CPU or GPU).

For every routed level of the model -- Nestor column routers (one per block), Nestor public
(shared) expert routers and private (column) expert routers (one per layer), the Nestor path gate,
and the MoE bank routers -- it accumulates, per corpus:
    counts[a]     tokens that selected choice a (each selection one draw; sums to k * tokens)
    gate_mass[a]  sum over tokens of the router gate of choice a (sums to tokens for renormalized
                  routers; the Nestor private routers use switch gates, which sum to < 1 per token)
    eff_mass[a]   Nestor experts only: gate x path weight (x column gate for private experts), the
                  weight with which the expert's output enters the token's update
and reports usage frequencies, entropies, and the mutual information between the routing decision
and the corpus label in bits, "counting each selection as one draw" (the definition of Table 9 of
arXiv 2610.31818); for the private experts also I(pair; corpus), I(column; corpus) and the
conditional I(expert; corpus | column) = I(pair) - I(column). Plug-in estimates come with the
first-order iid bias (A-1)(Y-1)/(2 N ln 2) and a window-level permutation null (labels shuffled
across windows, which respects the dependence of tokens within a window).

Windows: non-overlapping ``block``-token windows of each .bin (the DclmTokenDataset grid of the
held-out evaluation, no EOT remap), evenly spaced over the file, ``tokens_per_corpus // block``
of them per corpus (equal token counts per corpus). Models are rebuilt with the training-time
construction flags (run_exp.SPEED_STACK when importable) and evaluated in eval mode (no routing
noise), fp32 by default.

CLI:
  python routing_stats.py collect <experiment> <ckpt.pt> --corpus dclm=data/dclm/dclm_val.bin \
         --corpus pg19=... --tokens-per-corpus 1000000 --out stats.json [--block 4096] [--bf16]
         [--null-perms 50] [--save-assignments assign.npz] [--device cuda]
  python routing_stats.py compare before.npz after.npz          # share of assignments changed
  python routing_stats.py overlap stats.json --corpus dclm --manifest subset_manifest.json
  python routing_stats.py select stats.json --corpus dclm --family moe --k 4 --order least \
         [--seed 0] [--out spec.json]                            # emit a subset_adapt spec
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- information
def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def mutual_information_bits(table: np.ndarray) -> float:
    """Plug-in I(Y; A) in bits from a count table ``[num_labels, num_choices]``."""

    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return 0.0
    pya = t / n
    py = pya.sum(1, keepdims=True)
    pa = pya.sum(0, keepdims=True)
    nz = pya > 0
    # broadcast outer product, not `py @ pa`: no numpy BLAS once torch is loaded (OMP: Error #15 on
    # pip-torch + conda-numpy installs aborts the process at the first numpy BLAS call)
    return float((pya[nz] * np.log2(pya[nz] / (py * pa)[nz])).sum())


def mi_bias_bits(table: np.ndarray) -> float:
    """First-order (iid) bias of the plug-in MI: (A_eff - 1)(Y_eff - 1) / (2 N ln 2)."""

    t = np.asarray(table, dtype=np.float64)
    n = t.sum()
    if n <= 0:
        return 0.0
    a = int((t.sum(0) > 0).sum())
    y = int((t.sum(1) > 0).sum())
    return (a - 1) * (y - 1) / (2.0 * n * LN2)


def permutation_null(win_counts: np.ndarray, win_labels: np.ndarray, num_labels: int,
                     perms: int, seed: int = 0) -> tuple[float, float]:
    """Mean and 95th percentile of the MI with window labels shuffled (window-level null)."""

    if perms <= 0 or len(win_labels) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(perms):
        lab = rng.permutation(win_labels)
        tab = np.zeros((num_labels, win_counts.shape[1]))
        np.add.at(tab, lab, win_counts)
        vals.append(mutual_information_bits(tab))
    return float(np.mean(vals)), float(np.quantile(vals, 0.95))


# ----------------------------------------------------------------------------- aux parsing
def router_records(aux: dict[str, Any]) -> list[tuple[str, str, int, int | None, Any]]:
    """(key, family, block, layer, payload) for every routed decision in a forward's aux.

    family: "column" (Nestor group router of a block with > 1 column), "expert" (MoE / dense bank
    router), "public" (Nestor pool public router), "private" (payload = list of per-column aux).
    """

    out = []
    blocks = aux.get("blocks", [])
    for b, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            continue
        gr = blk.get("group_router")
        if isinstance(gr, dict) and gr["gates"].shape[-1] > 1:
            out.append((f"b{b}.column", "column", b, None, gr))
        for l, lay in enumerate(blk.get("layers", []) or []):
            if not isinstance(lay, dict):
                continue
            pool = "expert_pool_private" in lay
            er = lay.get("expert_router")
            if isinstance(er, dict):
                fam = "public" if pool else "expert"
                out.append((f"b{b}.l{l}.{fam}", fam, b, l, er))
            if pool:
                out.append((f"b{b}.l{l}.private", "private", b, l, lay["expert_pool_private"]))
    return out


def _counts(topk, valid, num_choices):
    import torch

    oh = torch.nn.functional.one_hot(topk.long(), num_classes=num_choices).sum(-2)  # [B,S,A]
    return (oh * valid.unsqueeze(-1)).sum(dim=(0, 1))


def _bits_of(topk, valid, num_choices, offset=0):
    """Per-token bitmask (int64) of the selected choices (0 for masked tokens)."""

    import torch

    w = (torch.ones((), dtype=torch.int64, device=topk.device) << (topk.long() + offset))
    return (w.sum(-1) * valid.long()).reshape(-1)


# ----------------------------------------------------------------------------- accumulator
class RoutingAccumulator:
    """Accumulates per-corpus routing tables from model aux dicts (one forward at a time)."""

    def __init__(self, labels: list[str], save_assignments: bool = False):
        self.labels = labels
        self.L = len(labels)
        self.tab: dict[str, dict[str, Any]] = {}
        self.win: dict[str, list] = defaultdict(list)        # key -> per-window count vectors
        self.win_labels: list[int] = []
        self.path: dict[tuple[int, int], Any] = {}             # (b, l) -> latest path-gate logits
        self.save = save_assignments
        self.assign: dict[tuple[str, int], list] = defaultdict(list)

    def _entry(self, key, family, b, l, A, k, extra=None):
        if key not in self.tab:
            self.tab[key] = {"family": family, "block": b, "layer": l, "choices": A, "k": k,
                             "counts": np.zeros((self.L, A)), "gate_mass": np.zeros((self.L, A)),
                             "eff_mass": np.zeros((self.L, A)), "tokens": np.zeros(self.L),
                             "draw_tokens": np.zeros(self.L), **(extra or {})}
        return self.tab[key]

    def add(self, label: int, aux: dict[str, Any]) -> None:
        import torch

        recs = router_records(aux)
        col_gates = {b: p["gates"].float() for _, fam, b, _, p in recs if fam == "column"}
        self.win_labels.append(label)
        for key, fam, b, l, p in recs:
            if fam in ("column", "expert", "public"):
                gates, topk, valid = p["gates"].float(), p["topk_indices"], p["valid_token_mask"]
                A, k = gates.shape[-1], topk.shape[-1]
                e = self._entry(key, fam, b, l, A, k)
                c = _counts(topk, valid, A).double().cpu().numpy()
                e["counts"][label] += c
                e["gate_mass"][label] += (gates * valid.unsqueeze(-1)).sum(dim=(0, 1)).double().cpu().numpy()
                e["tokens"][label] += int(valid.sum())
                e["draw_tokens"][label] += int(valid.sum())
                if fam == "public" and (b, l) in self.path:
                    alpha = torch.softmax(self.path[(b, l)].float(), dim=-1)[..., 0].reshape(valid.shape)
                    e["eff_mass"][label] += (gates * (alpha * valid).unsqueeze(-1)).sum(dim=(0, 1)).double().cpu().numpy()
                self.win[key].append(c)
                if self.save:
                    self.assign[(key, label)].append(_bits_of(topk, valid, A).cpu().numpy())
            else:  # private: list over columns
                G = len(p)
                Q, kq = p[0]["gates"].shape[-1], p[0]["topk_indices"].shape[-1]
                e = self._entry(key, fam, b, l, G * Q, kq, {"columns": G, "experts": Q})
                cnt = np.zeros(G * Q)
                gm = np.zeros(G * Q)
                em = np.zeros(G * Q)
                draws = 0
                alpha = None
                if (b, l) in self.path:
                    alpha = torch.softmax(self.path[(b, l)].float(), dim=-1)[..., 1]
                bits = None
                for c_idx, pc in enumerate(p):
                    gates, topk, valid = pc["gates"].float(), pc["topk_indices"], pc["valid_token_mask"]
                    sl = slice(c_idx * Q, (c_idx + 1) * Q)
                    cnt[sl] = _counts(topk, valid, Q).double().cpu().numpy()
                    gm[sl] = (gates * valid.unsqueeze(-1)).sum(dim=(0, 1)).double().cpu().numpy()
                    draws += int(valid.sum())
                    if alpha is not None and b in col_gates:
                        wcol = col_gates[b][..., c_idx]
                        wt = alpha.reshape(valid.shape) * wcol * valid
                        em[sl] = (gates * wt.unsqueeze(-1)).sum(dim=(0, 1)).double().cpu().numpy()
                    if self.save:
                        bc = _bits_of(topk, valid, Q, offset=c_idx * Q)
                        bits = bc if bits is None else bits | bc
                e["counts"][label] += cnt
                e["gate_mass"][label] += gm
                e["eff_mass"][label] += em
                e["draw_tokens"][label] += draws            # (token, member column) pairs
                bvalid = aux["blocks"][b].get("valid_token_mask")
                e["tokens"][label] += int(bvalid.sum()) if bvalid is not None else 0
                self.win[key].append(cnt)
                if self.save and bits is not None:
                    self.assign[(key, label)].append(bits.cpu().numpy())
        for (b, l), logits in self.path.items():
            key = f"b{b}.l{l}.path_gate"
            alpha = torch.softmax(logits.float(), dim=-1).reshape(-1, 2)
            e = self._entry(key, "path_gate", b, l, 2, 1)
            e["gate_mass"][label] += alpha.sum(0).double().cpu().numpy()
            e["tokens"][label] += alpha.shape[0]
        self.path.clear()

    # -------------------------------------------------------------------------
    def summarize(self, null_perms: int = 0, seed: int = 0) -> dict[str, Any]:
        routers: dict[str, Any] = {}
        win_labels = np.asarray(self.win_labels)
        for key, e in self.tab.items():
            fam = e["family"]
            r: dict[str, Any] = {"family": fam, "block": e["block"], "layer": e["layer"],
                                 "choices": e["choices"], "k": e["k"]}
            if fam == "path_gate":
                r["alpha_public_mean"] = {lab: float(e["gate_mass"][i, 0] / max(e["tokens"][i], 1))
                                          for i, lab in enumerate(self.labels)}
                r["tokens"] = {lab: int(e["tokens"][i]) for i, lab in enumerate(self.labels)}
                routers[key] = r
                continue
            counts = e["counts"]
            r["counts"] = {lab: counts[i].tolist() for i, lab in enumerate(self.labels)}
            r["gate_mass"] = {lab: e["gate_mass"][i].tolist() for i, lab in enumerate(self.labels)}
            if fam in ("public", "private"):
                r["eff_mass"] = {lab: e["eff_mass"][i].tolist() for i, lab in enumerate(self.labels)}
            tok = e["tokens"]
            r["tokens"] = {lab: int(tok[i]) for i, lab in enumerate(self.labels)}
            if fam == "private":
                r["member_draw_tokens"] = {lab: int(e["draw_tokens"][i]) for i, lab in enumerate(self.labels)}
            # fraction of tokens selecting each choice: sums to k (column / expert / public) and to
            # K_c * k_q for the private (column, expert) pairs
            r["usage_freq"] = {lab: (counts[i] / max(tok[i], 1)).tolist() for i, lab in enumerate(self.labels)}
            r["entropy_bits"] = {lab: entropy_bits(counts[i]) for i, lab in enumerate(self.labels)}
            r["effective_choices_frac"] = {lab: float(2 ** entropy_bits(counts[i]) / e["choices"])
                                           for i, lab in enumerate(self.labels)}
            r["mi_bits"] = mutual_information_bits(counts)
            r["mi_bias_bits"] = float(mi_bias_bits(counts))
            if null_perms:
                wc = np.stack(self.win[key])
                r["mi_null_mean_bits"], r["mi_null_p95_bits"] = permutation_null(wc, win_labels, self.L, null_perms, seed)
            if fam == "private":
                G, Q = e["columns"], e["experts"]
                r["columns"], r["experts"] = G, Q
                col = counts.reshape(self.L, G, Q).sum(-1)
                r["mi_pair_bits"] = r["mi_bits"]
                r["mi_column_bits"] = mutual_information_bits(col)
                r["mi_expert_given_column_bits"] = r["mi_bits"] - r["mi_column_bits"]
                r["column_member_counts"] = {lab: (col[i] / e["k"]).tolist() for i, lab in enumerate(self.labels)}
                if null_perms:
                    wc = np.stack(self.win[key])
                    wcol = wc.reshape(len(wc), G, Q).sum(-1)
                    nul = []
                    rng = np.random.default_rng(seed + 1)
                    for _ in range(null_perms):
                        lab = rng.permutation(win_labels)
                        tp = np.zeros((self.L, G * Q)); np.add.at(tp, lab, wc)
                        tc = np.zeros((self.L, G)); np.add.at(tc, lab, wcol)
                        nul.append(mutual_information_bits(tp) - mutual_information_bits(tc))
                    r["mi_expert_given_column_null_mean_bits"] = float(np.mean(nul))
                    r["mi_expert_given_column_null_p95_bits"] = float(np.quantile(nul, 0.95))
            routers[key] = r
        return {"routers": routers, "summary": table9_summary(routers)}

    def assignments(self) -> dict[str, np.ndarray]:
        """Per-token selected-set bitmasks, in the narrowest unsigned dtype that holds the choices."""

        out = {}
        for (key, lab), v in self.assign.items():
            a = self.tab[key]["choices"]
            if a > 64:
                raise NotImplementedError(f"{key}: {a} choices do not fit a 64-bit selected-set mask")
            dt = np.uint8 if a <= 8 else np.uint16 if a <= 16 else np.uint32 if a <= 32 else np.uint64
            out[f"{key}|{self.labels[lab]}"] = np.concatenate(v).astype(np.int64).view(np.uint64).astype(dt)
        return out


def table9_summary(routers: dict[str, Any]) -> dict[str, Any]:
    """Mean (min-max) MI over the routed layers per level, as in Table 9 of the paper."""

    groups = {"column": [], "public": [], "private_given_column": [], "private_pair": [], "expert": []}
    for r in routers.values():
        fam = r["family"]
        if fam == "column":
            if r["k"] < r["choices"]:          # routed blocks only (the dense tail has k = C)
                groups["column"].append(r["mi_bits"])
        elif fam == "public":
            groups["public"].append(r["mi_bits"])
        elif fam == "private":
            groups["private_given_column"].append(r["mi_expert_given_column_bits"])
            groups["private_pair"].append(r["mi_pair_bits"])
        elif fam == "expert":
            groups["expert"].append(r["mi_bits"])
    return {k: ({"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
                if v else None) for k, v in groups.items()}


# ----------------------------------------------------------------------------- model driving
def install_path_gate_hooks(model, acc: RoutingAccumulator) -> list:
    """Forward hooks capturing every Nestor pool path-gate's logits ([B, S, 2])."""

    import re

    pat = re.compile(r"^backbone\.blocks\.(\d+)\.layers\.(\d+)\.expert_pool\.path_gate$")
    handles = []
    for name, mod in model.named_modules():
        m = pat.match(name)
        if m:
            b, l = int(m.group(1)), int(m.group(2))

            def hook(_mod, _inp, out, b=b, l=l):
                acc.path[(b, l)] = out.detach()
            handles.append(mod.register_forward_hook(hook))
    return handles


def window_starts(num_tokens: int, block: int, n_windows: int) -> list[int]:
    """Evenly spaced windows on the non-overlapping block grid of the file."""

    total = num_tokens // block
    if total == 0:
        raise ValueError(f"file has {num_tokens} tokens < one block of {block}")
    n = min(n_windows, total)
    idx = np.unique(np.linspace(0, total - 1, n).round().astype(np.int64))
    return [int(i) * block for i in idx]


def collect(model, corpora: dict[str, str], *, tokens_per_corpus: int, block: int, device: str,
            bf16: bool = False, null_perms: int = 0, save_assignments: bool = False,
            seed: int = 0, log=print) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run ``model`` (eval mode) over equal-token windows of every corpus and summarize."""

    import torch

    from nestor.data.token_documents import open_token_bin

    labels = list(corpora)
    acc = RoutingAccumulator(labels, save_assignments=save_assignments)
    handles = install_path_gate_hooks(model, acc)
    model.eval()
    info = {}
    t0 = time.time()
    try:
        with torch.no_grad():
            for li, (lab, path) in enumerate(corpora.items()):
                toks, _eot = open_token_bin(path)
                starts = window_starts(len(toks), block, max(1, tokens_per_corpus // block))
                for s in starts:
                    x = torch.as_tensor(np.asarray(toks[s: s + block], dtype=np.int64)).view(1, -1).to(device)
                    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
                        out = model(x)
                    acc.add(li, out.aux)
                info[lab] = {"path": os.path.abspath(path), "windows": len(starts), "tokens": len(starts) * block,
                             "file_tokens": int(len(toks))}
                log(f"[routing] {lab}: {len(starts)} windows x {block} in {time.time() - t0:.1f}s")
    finally:
        for h in handles:
            h.remove()
    res = acc.summarize(null_perms=null_perms, seed=seed)
    res["corpora"] = info
    res["settings"] = {"block": block, "tokens_per_corpus": tokens_per_corpus, "precision": "bf16" if bf16 else "fp32",
                       "window_selection": "evenly spaced non-overlapping block grid, no EOT remap",
                       "null_perms": null_perms, "mi_definition": "each selection one draw (paper Table 9)"}
    return res, (acc.assignments() if save_assignments else {})


def load_model(experiment: str, checkpoint: str, device: str):
    """Rebuild ``experiment`` with the training-time construction flags and load ``checkpoint``."""

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "repo"))
    try:
        import run_exp  # the driver: its SPEED_STACK fixes the parameter set of the runs

        flags = dict(run_exp.SPEED_STACK)
    except Exception:
        flags = {"NESTOR_STACKED_EXPERT_BANKS": "1", "NESTOR_DENSE_NO_ROUTER": "1", "NESTOR_BATCHED_ROUTERS": "1"}
    for k, v in flags.items():
        os.environ.setdefault(k, v)
    os.environ["NESTOR_COMPILE"] = "0"
    try:
        import compat

        compat.install()
    except Exception:
        pass
    import nestor.experiments.pretrain  # noqa: F401
    from nestor.eval.checkpoint import load_experiment_checkpoint

    model, info = load_experiment_checkpoint(experiment, checkpoint, device=device)
    if info["missing"]:
        missing_params = [k for k in info["missing"] if not k.endswith(("inv_freq", "_cached_causal_mask"))]
        if missing_params:
            raise SystemExit(f"missing keys: {missing_params[:5]}")
    return model, info


# ----------------------------------------------------------------------------- post-hoc tools
def compare_assignments(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of tokens whose selected set differs between two runs on the same tokens."""

    out = {}
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            raise ValueError(f"{k}: token counts differ ({x.shape} vs {y.shape})")
        live = (x != 0) | (y != 0)
        out[k] = float(((x != y) & live).sum() / max(int(live.sum()), 1))
    return out


def unit_keys_for(router_key: str, r: dict[str, Any], choice: int, layers_in_block: dict[int, list[int]]) -> list[str]:
    """subset_adapt unit keys owned by one routing choice."""

    b, l, fam = r["block"], r["layer"], r["family"]
    if fam == "column":
        keys = []
        for ll in layers_in_block.get(b, []):
            keys.append(f"attn:b{b}:l{ll}:c{choice}")
            keys.append(f"priv:b{b}:l{ll}:c{choice}:*")
        return keys
    if fam == "public":
        return [f"pub:b{b}:l{l}:e{choice}"]
    if fam == "expert":
        return [f"moe:b{b}:l{l}:e{choice}"]
    if fam == "private":
        q = r["experts"]
        return [f"priv:b{b}:l{l}:c{choice // q}:e{choice % q}"]
    return []


def routing_mass_through(stats: dict[str, Any], corpus: str, trainable_units: set[str],
                         mass: str = "counts") -> dict[str, Any]:
    """Fraction of a corpus' routing mass that passes through trainable units, per router.

    ``mass`` is "counts" (selections), "gate_mass" or "eff_mass" (Nestor experts only).
    """

    routers = stats["routers"]
    layers_in_block: dict[int, list[int]] = defaultdict(list)
    for r in routers.values():
        if r["layer"] is not None and r["layer"] not in layers_in_block[r["block"]]:
            layers_in_block[r["block"]].append(r["layer"])
    priv_prefixes = {u.rsplit(":e", 1)[0] for u in trainable_units if u.startswith("priv:")}
    out = {}
    for key, r in routers.items():
        if r["family"] == "path_gate" or mass not in r:
            continue
        v = np.asarray(r[mass][corpus], dtype=np.float64)
        sel = np.zeros_like(v, dtype=bool)
        for a in range(len(v)):
            keys = unit_keys_for(key, r, a, layers_in_block)
            if r["family"] == "column":
                hit = any(k in trainable_units for k in keys if k.startswith("attn")) or \
                      any(k.rsplit(":*", 1)[0] in priv_prefixes for k in keys if k.endswith(":*"))
            else:
                hit = any(k in trainable_units for k in keys)
            sel[a] = hit
        tot = v.sum()
        out[key] = {"family": r["family"], "fraction": float(v[sel].sum() / tot) if tot > 0 else 0.0,
                    "choices_trainable": int(sel.sum())}
    fams = defaultdict(list)
    for v in out.values():
        fams[v["family"]].append(v["fraction"])
    return {"per_router": out, "mean_by_family": {f: float(np.mean(x)) for f, x in fams.items()}}


def select_spec(stats: dict[str, Any], corpus: str, family: str, k: int, order: str,
                seed: int = 0, mass: str = "counts", extra: list | None = None,
                ref_corpus: str | None = None) -> dict[str, Any]:
    """A subset_adapt spec training, per router of ``family``, k choices picked by ``order``:
    "least" / "most" used by ``corpus``, "random", or -- with ``ref_corpus`` -- "ratio_high" /
    "ratio_low": the largest / smallest usage ratio corpus : ref_corpus (e.g. target : DCLM, the
    contrastive selection that separates "few old tokens" from "few target tokens")."""

    rng = np.random.default_rng(seed)
    filters = []
    for key, r in sorted(stats["routers"].items()):
        if r["family"] != family:
            continue
        if family == "column" and r["k"] >= r["choices"]:
            continue
        v = np.asarray(r[mass][corpus], dtype=np.float64)
        if order == "least":
            pick = np.argsort(v, kind="stable")[:k]
        elif order == "most":
            pick = np.argsort(-v, kind="stable")[:k]
        elif order == "random":
            pick = rng.choice(len(v), size=k, replace=False)
        elif order in ("ratio_high", "ratio_low"):
            if ref_corpus is None:
                raise ValueError("ratio orders need ref_corpus")
            ref = np.asarray(r[mass][ref_corpus], dtype=np.float64)
            eps = 1e-9 * max(v.sum(), ref.sum(), 1.0)
            ratio = (v / max(v.sum(), eps) + eps) / (ref / max(ref.sum(), eps) + eps)
            pick = np.argsort(-ratio if order == "ratio_high" else ratio, kind="stable")[:k]
        else:
            raise ValueError(order)
        pick = sorted(int(a) for a in pick)
        b, l = r["block"], r["layer"]
        if family == "column":
            filters.append({"kind": "column", "block": b, "column": pick})
        elif family == "public":
            filters.append({"kind": "pub", "block": b, "layer": l, "expert": pick})
        elif family == "expert":
            filters.append({"kind": "moe", "block": b, "layer": l, "expert": pick})
        elif family == "private":
            q = r["experts"]
            by_col = defaultdict(list)
            for a in pick:
                by_col[a // q].append(a % q)
            for c, es in sorted(by_col.items()):
                filters.append({"kind": "priv", "block": b, "layer": l, "column": c, "expert": es})
    return {"mode": "train", "select": filters + list(extra or []),
            "_provenance": {"stats_corpus": corpus, "ref_corpus": ref_corpus, "family": family, "k": k,
                            "order": order, "seed": seed, "mass": mass}}


# ----------------------------------------------------------------------------- CLI
def _cli(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Routing statistics of Nestor / MoE checkpoints")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("experiment"); c.add_argument("checkpoint")
    c.add_argument("--corpus", action="append", required=True, help="name=path.bin (repeatable)")
    c.add_argument("--tokens-per-corpus", type=int, default=1_000_000)
    c.add_argument("--block", type=int, default=4096)
    c.add_argument("--device", default=None)
    c.add_argument("--bf16", action="store_true")
    c.add_argument("--null-perms", type=int, default=50)
    c.add_argument("--save-assignments", default=None)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare"); m.add_argument("a"); m.add_argument("b")
    o = sub.add_parser("overlap"); o.add_argument("stats"); o.add_argument("--corpus", required=True)
    o.add_argument("--manifest", required=True); o.add_argument("--mass", default="counts")
    s = sub.add_parser("select"); s.add_argument("stats"); s.add_argument("--corpus", required=True)
    s.add_argument("--family", required=True, choices=["column", "public", "private", "expert"])
    s.add_argument("--k", type=int, required=True); s.add_argument("--order", default="least")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--mass", default="counts")
    s.add_argument("--ref-corpus", default=None, help="for --order ratio_high / ratio_low")
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, info = load_model(args.experiment, args.checkpoint, device)
        corpora = dict(x.split("=", 1) for x in args.corpus)
        res, assign = collect(model, corpora, tokens_per_corpus=args.tokens_per_corpus, block=args.block,
                              device=device, bf16=args.bf16, null_perms=args.null_perms,
                              save_assignments=bool(args.save_assignments))
        res.update({"experiment": args.experiment, "checkpoint": os.path.abspath(args.checkpoint),
                    "step": info.get("step")})
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        if args.save_assignments:
            np.savez_compressed(args.save_assignments, **assign)
        print(json.dumps(res["summary"], indent=1))
        return 0
    if args.cmd == "compare":
        a, b = dict(np.load(args.a)), dict(np.load(args.b))
        print(json.dumps(compare_assignments(a, b), indent=1))
        return 0
    stats = json.load(open(args.stats, encoding="utf-8"))
    if args.cmd == "overlap":
        man = json.load(open(args.manifest, encoding="utf-8"))
        print(json.dumps(routing_mass_through(stats, args.corpus, set(man["units_trainable"]), args.mass), indent=1))
        return 0
    spec = select_spec(stats, args.corpus, args.family, args.k, args.order, args.seed, args.mass,
                       ref_corpus=args.ref_corpus)
    txt = json.dumps(spec, indent=1)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
