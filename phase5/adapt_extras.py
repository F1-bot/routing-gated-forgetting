"""Training-time extensions for the adaptation runs (env-activated; strict no-op when unset).

Install in run_exp.py right after `subset_adapt.install`:  `import adapt_extras; adapt_extras.install`

  NESTOR_SUBSET_OPTIMIZER=sgd   SGD(-momentum) arm (C2): the expert matrices (unit kinds NESTOR_SGD_KINDS,
                                default moe,pub,priv) leave their Muon groups and are stepped by plain
                                SGD with momentum -- the same momentum convention as the authors' Muon
                                (buf = mu*buf + g; Nesterov direction g + mu*buf), WITHOUT the
                                Newton-Schulz orthogonalization. Everything else keeps its optimizer.
      NESTOR_SGD_LR (required)  absolute peak lr, scheduled like every group (warm-up + cosine)
      NESTOR_SGD_MOMENTUM=0.95  NESTOR_SGD_NESTEROV=1  NESTOR_SGD_WD=0 (decoupled, lr-scaled)
      NESTOR_SGD_SCALE=none|muon  multiply the lr by the matrix's muon_scale sqrt(d_out/d_in)
  NESTOR_MAX_GRAD_NORM=none|<x> override the recipe's clip (1.0); SGD law tests need clipping inactive
  NESTOR_TRAIN_LOG_DIR=<dir>    per-step JSONL: grad norm (pre-clip), max norm, clipped flag, group lrs, and the
                                step's mean training loss ("loss" = LossOutput.total, "ce" = its unweighted
                                cross_entropy component; E1 divergence rule, G2-2)
  NESTOR_INTERIM_EVAL=<json|path>  in-process per-window evals at given optimizer steps, e.g.
      {"steps": [8, 24, 48], "out": "<dir>", "block": 4096, "routing": false,
       "sets": [{"name": "dclm", "path": "data/dclm/dclm_val.bin", "tokens": 262144}, ...]}
      writes <out>/step008__dclm.npz (eval_windows format; windows = the HEAD of the final eval's
      window list, so interim and final evals pair) and <out>/displacement.jsonl (realized
      ||theta_t - theta_0||^2 by unit kind and expert unit at every interim step).
      Sets may carry "select", "span", "span_align" and "head" (registered interim set = first 64 windows of the
      evaluation grid: {"tokens": 2048000, "select": "even", "span": [0.5, 1], "head": 64}).
  NESTOR_INTERIM_FULL=1         (C2 runs) additionally evaluate the FULL grid (the set without "head",
      or "full_tokens" (default 2,048,000) of the same file/selection) at "full_steps" (default [8, 24, 48]) for the
      sets named in "full_sets" (default: all) -> <out>/step008__dclm__full.npz; the head file of that step is sliced
      from the same evaluation.
  NESTOR_LOSS_WEIGHT=<c>        (dose arms) deterministic per-token loss weight: the causal-LM loss of
      every token is multiplied by c (total = c x CE; auxiliary router losses unchanged; the logged cross_entropy
      stays unweighted). No random token masking.

CLI (C2 calibration; CPU):
  python adapt_extras.py sgd-lr --probe g0/moe/probe__german.json --subset C2_german_d0_A [--momentum 0.95] [--out f.json]
  python adapt_extras.py sgd-select --base <base eval npz of target> --muon <interim dir> \
        --run 0.31=<interim dir> --run 0.98=<dir> --run 3.1=<dir> --set german [--min-frac 0.25] [--out g2.json]
  (sgd-select exits 3 when G2 fails unless --out is given; then the file's G2_pass decides, e.g. via a json gate)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SGD_KINDS_DEFAULT = "moe,pub,priv"
_INSTALLED = {"done": False}
_CLIP: dict[str, Any] = {}
_STATE: dict[int, dict] = {}


# ============================================================================= SGD arm
def _muon_adamw_sgd_class():
    from nestor.scaling.muon import MuonAdamW

    class MuonAdamWSGD(MuonAdamW):
        """MuonAdamW plus a 'sgd' group method (momentum SGD, no orthogonalization)."""

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                method = group["method"]
                if method == "muon":
                    self._muon_step(group)
                elif method == "adamw":
                    self._adamw_step(group)
                elif method == "sgd":
                    self._sgd_step(group)
                else:
                    raise ValueError(f"Unknown optimizer method '{method}'.")
            if self._shard:
                self._broadcast_muon_params()
            return loss

        def _sgd_step(self, group):
            params, grads, bufs = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                params.append(p); grads.append(p.grad); bufs.append(st["momentum_buffer"])
            if not params:
                return
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            if mu != 0.0:
                torch._foreach_mul_(bufs, mu)
                torch._foreach_add_(bufs, grads)
                directions = torch._foreach_add(grads, bufs, alpha=mu) if group["nesterov"] else bufs
            else:
                directions = grads
            if wd != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            torch._foreach_add_(params, directions, alpha=-lr)

    return MuonAdamWSGD


def sgd_settings() -> dict:
    lr = os.environ.get("NESTOR_SGD_LR")
    if lr is None:
        raise RuntimeError("NESTOR_SUBSET_OPTIMIZER=sgd needs NESTOR_SGD_LR")
    return {"lr": float(lr), "momentum": float(os.environ.get("NESTOR_SGD_MOMENTUM", "0.95")),
            "nesterov": os.environ.get("NESTOR_SGD_NESTEROV", "1") == "1",
            "weight_decay": float(os.environ.get("NESTOR_SGD_WD", "0")),
            "scale": os.environ.get("NESTOR_SGD_SCALE", "none"),
            "kinds": tuple(k for k in os.environ.get("NESTOR_SGD_KINDS", SGD_KINDS_DEFAULT).split(",") if k)}


def convert_to_sgd(opt, model, settings: dict | None = None):
    """A MuonAdamWSGD equal to `opt` except that the expert matrices are in 'sgd' groups."""

    import subset_adapt as SA
    from nestor.scaling.blocks import get_scaling

    st = settings or sgd_settings()
    kind_of = {}
    for e in SA.build_catalog(model):
        kinds = {u.kind for u in e.units}
        kind_of[id(e.param)] = kinds
    new_groups, sgd_groups = [], {}
    moved = 0
    for g in opt.param_groups:
        keep, mv = [], []
        for p in g["params"]:                    # identity, never tensor ==
            ks = kind_of.get(id(p))
            (mv if (g["method"] == "muon" and ks and ks <= set(st["kinds"])) else keep).append(p)
        if keep:
            new_groups.append({**{k: v for k, v in g.items() if k != "params"}, "params": keep})
        for p in mv:
            scale = 1.0
            if st["scale"] == "muon":
                s = get_scaling(p)
                scale = float(s.muon_scale) if s is not None else 1.0
            key = round(scale, 8)
            if key not in sgd_groups:
                sgd_groups[key] = {"params": [], "method": "sgd", "lr": st["lr"] * scale, "momentum": st["momentum"],
                                   "nesterov": st["nesterov"], "weight_decay": st["weight_decay"]}
            sgd_groups[key]["params"].append(p)
            moved += 1
    if not moved:
        raise RuntimeError(f"NESTOR_SUBSET_OPTIMIZER=sgd: no trainable Muon matrix of kinds {st['kinds']}")
    new_groups += [sgd_groups[k] for k in sorted(sgd_groups)]
    d = opt.defaults
    cls = _muon_adamw_sgd_class()
    new = cls(new_groups, lr=d["lr"], betas=d["betas"], eps=d["eps"], weight_decay=d["weight_decay"],
              momentum=d["momentum"], nesterov=d["nesterov"], ns_steps=d["ns_steps"], shard=False)
    before = {id(p) for g in opt.param_groups for p in g["params"]}
    after = [id(p) for g in new.param_groups for p in g["params"]]
    if sorted(after) != sorted(before) or len(after) != len(set(after)):
        raise AssertionError("convert_to_sgd changed the parameter set")
    print(f"[adapt_extras] SGD arm: {moved} expert tensors -> momentum SGD (lr {st['lr']:g}, mu {st['momentum']}, "
          f"nesterov {st['nesterov']}, wd {st['weight_decay']}, scale {st['scale']}); "
          f"{sum(len(g['params']) for g in new_groups) - moved} tensors keep their optimizer", flush=True)
    return new


def patch_optimizer_builder() -> None:
    """Route nestor.train.optim._build_muon_adamw through convert_to_sgd when the SGD arm is on."""

    import nestor.train.optim as O

    if getattr(O._build_muon_adamw, "_sgd_patched", False):
        return
    orig = O._build_muon_adamw

    def wrapped(config, model):
        opt = orig(config, model)
        if os.environ.get("NESTOR_SUBSET_OPTIMIZER", "muon").strip().lower() == "sgd":
            return convert_to_sgd(opt, model)
        return opt

    wrapped._sgd_patched = True
    O._build_muon_adamw = wrapped


# ============================================================================= trainer hooks
def _interim_cfg() -> dict | None:
    raw = os.environ.get("NESTOR_INTERIM_EVAL", "").strip()
    if not raw:
        return None
    cfg = json.loads(raw) if raw.startswith("{") else json.load(open(raw, encoding="utf-8-sig"))
    cfg["steps"] = sorted(int(s) for s in cfg.get("steps", []))
    if os.environ.get("NESTOR_INTERIM_FULL", "").strip() == "1":
        cfg["full_steps"] = sorted(int(s) for s in cfg.get("full_steps", [8, 24, 48]))
        cfg["steps"] = sorted(set(cfg["steps"]) | set(cfg["full_steps"]))
    else:
        cfg["full_steps"] = []
    cfg.setdefault("block", int(os.environ.get("DCLM_BLOCK_SIZE", "4096")))
    if "out" not in cfg:
        raise ValueError("NESTOR_INTERIM_EVAL needs 'out'")
    return cfg


def _log_dir(cfg) -> str | None:
    d = os.environ.get("NESTOR_TRAIN_LOG_DIR") or (cfg or {}).get("out")
    if d:
        os.makedirs(d, exist_ok=True)
    return d


def _append(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _grad_norm(model) -> float:
    gs = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs])))


def _head_res(res: dict, n: int) -> dict:
    """The first n windows of an eval_windows result (every per-window array and routing summary sliced)."""

    nw = len(res["ce_sum"])
    out = {}
    for k, v in res.items():
        if k == "routing":
            out[k] = {kk: vv[:n] for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == nw:
            out[k] = v[:n]
        else:
            out[k] = v
    return out


def loss_weight() -> float | None:
    raw = os.environ.get("NESTOR_LOSS_WEIGHT", "").strip()
    if not raw:
        return None
    c = float(raw)
    if not (math.isfinite(c) and c > 0):
        raise ValueError(f"NESTOR_LOSS_WEIGHT must be a positive finite number, got {raw!r}")
    return c


def patch_loss_weight(c: float | None = None):
    """Deterministic per-token loss weight c on the causal-LM loss. Returns the original forward
    (None when no weight is requested)."""

    c = loss_weight() if c is None else float(c)
    if c is None:
        return None
    from nestor.train.losses import CausalLMLoss, LossOutput

    orig = getattr(CausalLMLoss.forward, "_dose_orig", CausalLMLoss.forward)

    def forward(self, pred, target, aux=None):
        out = orig(self, pred, target, aux)
        comps = dict(out.components)
        comps["loss_weight"] = out.total.detach().new_tensor(c)
        return LossOutput(total=out.total * c, components=comps)

    forward._dose_orig = orig
    forward._dose_c = c
    CausalLMLoss.forward = forward
    print(f"[adapt_extras] deterministic per-token loss weight c = {c}", flush=True)
    return orig


def _interim_eval(trainer, step: int, cfg: dict, st: dict) -> None:
    import eval_windows as EW
    import weight_controls as WC
    from nestor.train.checkpoint import _unwrap_compiled

    base = _unwrap_compiled(trainer.model)
    if "sets" not in st:
        def _ws(s):  # prereg sec. 4.3: interim set = the first 64 windows ("head") of an evaluation grid
            w = EW.WindowSet(s["name"], s["path"], cfg["block"], -(-int(s.get("tokens", 262144)) // cfg["block"]),
                             s.get("select", "first"), int(s.get("seed", 0)), span=s.get("span"),
                             span_align=s.get("span_align"))
            return w.head(int(s["head"])) if s.get("head") else w

        def _full(s):  # the full evaluation grid behind an interim set
            tok = s.get("tokens", 262144) if s.get("head") else s.get("full_tokens", 2048000)
            return EW.WindowSet(s["name"], s["path"], cfg["block"], -(-int(tok) // cfg["block"]),
                                s.get("select", "first"), int(s.get("seed", 0)), span=s.get("span"),
                                span_align=s.get("span_align"))

        st["sets"] = [_ws(s) for s in cfg.get("sets", [])]
        names = cfg.get("full_sets") or [s["name"] for s in cfg.get("sets", [])]
        st["full"] = {s["name"]: _full(s) for s in cfg.get("sets", []) if s["name"] in names} if cfg.get("full_steps") else {}
    device = next(base.parameters()).device.type
    t0 = time.time()
    base.eval()
    try:
        for ws in st["sets"]:
            fw = st["full"].get(ws.name) if step in cfg.get("full_steps", []) else None
            if fw is not None:
                rf = EW.evaluate(base, fw, device=device, routing=bool(cfg.get("routing", False)), log=None)
                EW.save_result(os.path.join(cfg["out"], f"step{step:03d}__{ws.name}__full.npz"), rf,
                               {"variant": f"step{step}", "set": fw.describe(), "step": step, "precision": "fp32", "full": True, **_precision()})
                n = len(ws.index)
                if np.array_equal(fw.index[:n], ws.index):          # the interim head is the head of the full grid
                    res = _head_res(rf, n)
                else:
                    res = EW.evaluate(base, ws, device=device, routing=bool(cfg.get("routing", False)), log=None)
            else:
                res = EW.evaluate(base, ws, device=device, routing=bool(cfg.get("routing", False)), log=None)
            EW.save_result(os.path.join(cfg["out"], f"step{step:03d}__{ws.name}.npz"), res,
                           {"variant": f"step{step}", "set": ws.describe(), "step": step, "precision": "fp32", **_precision()})
        disp = WC.displacement(base, st["theta0"])
        _append(os.path.join(cfg["out"], "displacement.jsonl"), {"step": step, **disp})
    finally:
        trainer._enter_train_mode()
    print(f"[adapt_extras] interim eval at step {step}: {len(st['sets'])} sets in {time.time() - t0:.1f}s", flush=True)


_LOSS: dict[str, Any] = {}


def _loss_take() -> dict:
    """Mean training loss of the micro-batches accumulated since the last optimizer step (one host sync), then reset."""

    n = _LOSS.get("n", 0)
    out = {"loss": None, "ce": None, "micro_batches": n}
    if n:
        out["loss"] = float(_LOSS["total"]) / n
        if _LOSS.get("ce") is not None:
            out["ce"] = float(_LOSS["ce"]) / n
    _LOSS.clear()
    return out


def patch_train_step() -> None:
    """Accumulate the per-micro-batch training loss (authors' LossOutput.total and its unweighted cross_entropy
    component) for the per-step train log (prereg sec. 13 E1; G2-2 'train loss at step 24 below step 1').
    Detached adds on the device; no effect on autograd or numerics."""

    from nestor.train.trainer import Trainer

    if getattr(Trainer._train_step, "_extras_patched", False):
        return
    orig = Trainer._train_step

    def train_step(self, batch, *a, **k):
        out = orig(self, batch, *a, **k)
        try:
            lo = out[0]
            t = lo.total.detach().float()
            _LOSS["total"] = t if _LOSS.get("total") is None else _LOSS["total"] + t
            ce = (getattr(lo, "components", None) or {}).get("cross_entropy")
            if ce is not None:
                c = ce.detach().float()
                _LOSS["ce"] = c if _LOSS.get("ce") is None else _LOSS["ce"] + c
            _LOSS["n"] = _LOSS.get("n", 0) + 1
        except (AttributeError, IndexError, TypeError):
            pass
        return out

    train_step._extras_patched = True
    Trainer._train_step = train_step


def _precision() -> dict:
    return {"allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def patch_trainer() -> None:
    from nestor.train.trainer import Trainer

    if getattr(Trainer._update_weights, "_extras_patched", False):
        return
    orig = Trainer._update_weights
    cfg = _interim_cfg()
    log_dir = _log_dir(cfg)
    if log_dir:
        patch_train_step()

    def wrapped(self, model_aux_batches):
        from nestor.train.checkpoint import _unwrap_compiled

        st = _STATE.setdefault(id(self), {})
        if cfg and "theta0" not in st:
            base = _unwrap_compiled(self.model)
            st["theta0"] = {n: p.detach().to("cpu", copy=True) for n, p in base.named_parameters() if p.requires_grad}
        losses = _loss_take() if log_dir else None          # this step's micro-batches (before any interim eval)
        if cfg and 0 in cfg["steps"] and not st.get("step0_done"):
            # step 0 = theta_0 evaluated IN the training process (same TF32/matmul settings as the later interim steps):
            # the paired reference of interim gains and F (G2-2, G2-3)
            st["step0_done"] = True
            _interim_eval(self, 0, cfg, st)
        pre_norm = _grad_norm(self.model) if (log_dir and self.config.max_grad_norm is None) else None
        _CLIP.pop("last", None)
        orig(self, model_aux_batches)
        step = self.global_step + 1
        if step in (1, 2):                                    # optimizer state exists after step 1 (host fact: WDDM
            import memguard as MG                             # pages oversubscribed VRAM silently -> raise instead)

            MG.check_peak(next(self.model.parameters()).device, f"training step {step}")
        if log_dir:
            gn, mx = _CLIP.get("last", (pre_norm, self.config.max_grad_norm))
            _append(os.path.join(log_dir, "train_steps.jsonl"),
                    {"step": step, "grad_norm": gn, "max_norm": mx, "clipped": bool(mx is not None and gn is not None and gn > mx),
                     "lrs": sorted({round(float(g["lr"]), 12) for g in self.optimizer.param_groups}),
                     "methods": sorted({g.get("method", "?") for g in self.optimizer.param_groups}), "time": time.time(),
                     **(losses or {})})
        if cfg and step in cfg["steps"]:
            _interim_eval(self, step, cfg, st)
        if log_dir:
            _LOSS.clear()                                     # evaluation forwards never count as training loss

    wrapped._extras_patched = True
    Trainer._update_weights = wrapped


def patch_clip_logging() -> None:
    import torch.nn.utils as U

    if getattr(U.clip_grad_norm_, "_extras_patched", False):
        return
    orig = U.clip_grad_norm_

    def clip(parameters, max_norm, *a, **k):
        tn = orig(parameters, max_norm, *a, **k)
        _CLIP["last"] = (float(tn), float(max_norm))
        return tn

    clip._extras_patched = True
    U.clip_grad_norm_ = clip


def patch_max_grad_norm() -> None:
    raw = os.environ.get("NESTOR_MAX_GRAD_NORM", "").strip()
    if not raw:
        return
    from nestor.pipelines.train import TrainPipeline

    if getattr(TrainPipeline.setup, "_extras_patched", False):
        return
    orig = TrainPipeline.setup
    value = None if raw.lower() in ("none", "off") else float(raw)

    def setup(self):
        self.job_config.max_grad_norm = value
        print(f"[adapt_extras] max_grad_norm overridden: {value}", flush=True)
        return orig(self)

    setup._extras_patched = True
    TrainPipeline.setup = setup


def install() -> bool:
    """Activate the extensions requested by the environment (idempotent)."""

    if _INSTALLED["done"]:
        return True
    patch_optimizer_builder()
    patch_max_grad_norm()
    patch_loss_weight()
    if os.environ.get("NESTOR_INTERIM_EVAL") or os.environ.get("NESTOR_TRAIN_LOG_DIR"):
        patch_clip_logging()
        patch_trainer()
    _INSTALLED["done"] = True
    return True


# ============================================================================= C2 calibration
def sgd_lr_grid(probe: dict, subset: str, momentum: float = 0.95, lr_muon: float | None = None) -> dict:
    """Displacement-matched SGD learning rates from the G0 first-step statistics of a subset.

    probe[subset] carries grad_sq = sum ||g_m||_F^2 of the mean target gradient and udir_sq =
    sum ||u_m||_F^2 of the unit-lr Muon direction (u_m = muon_scale * NS(g_m)). Muon's first-step
    displacement is lr_muon * sqrt(udir_sq). Matching SGD's FIRST Nesterov step (direction (1+mu) g)
    gives lr_first = lr_muon sqrt(udir_sq) / ((1+mu) sqrt(grad_sq)); matching its STEADY-STATE step
    (direction ~ g / (1-mu) for consistent gradients) gives lr_ss = lr_muon sqrt(udir_sq) (1-mu) / sqrt(grad_sq).
    The pre-declared 3-run grid is geometric over [lr_ss, lr_first] (ratio 1/(1-mu^2) = 10.3 at mu=0.95).
    Mean-gradient norms underestimate minibatch-gradient norms, so the grid errs high for small batches.
    """

    s = probe["subsets"][subset]
    lr_m = lr_muon if lr_muon is not None else float(probe.get("lr_muon", 0.0025))
    num = lr_m * math.sqrt(s["udir_sq"])
    den = math.sqrt(max(s["grad_sq"], 1e-300))
    lr_first = num / ((1 + momentum) * den)
    lr_ss = num * (1 - momentum) / den
    grid = [lr_ss, math.sqrt(lr_ss * lr_first), lr_first]
    return {"subset": subset, "lr_muon": lr_m, "momentum": momentum, "muon_first_step_norm": num,
            "grad_norm": den, "lr_first_step_matched": lr_first, "lr_steady_state_matched": lr_ss,
            "grid": grid, "rule": "geometric grid over [steady-state-matched, first-step-matched]"}


def sgd_select(base: dict, muon: dict, runs: dict[float, dict], *, min_frac: float = 0.25,
               max_rise: float = 0.05) -> dict:
    """Pre-declared C2 feasibility rule (gate G2) from paired interim evals on the target stream.

    base: eval_windows result of the base model on the target set; muon: {step: result} of the Muon
    reference run; runs: {lr: {step: result}}. gain(step) = CE_base - CE_run(step) (paired windows).
    A learning rate is STABLE if every interim CE is finite and <= CE_base + max_rise, and FEASIBLE if
    stable and gain(24) >= min_frac * gain_muon(24). Choice: the feasible lr whose gain(24) is closest
    to Muon's in log ratio (matched gain). G2 passes iff some lr is feasible.
    """

    import eval_windows as EW

    step = 24 if 24 in muon else max(muon)
    g_mu = -EW.paired(base, muon[step])["diff"]
    rows = []
    for lr, steps in sorted(runs.items()):
        ces = {s: EW.paired(base, r) for s, r in steps.items()}
        finite = all(math.isfinite(v["ce_b"]) for v in ces.values())
        stable = finite and all(v["diff"] <= max_rise for v in ces.values())
        gain = -ces[step]["diff"] if step in ces else float("nan")
        feas = bool(stable and math.isfinite(gain) and g_mu > 0 and gain >= min_frac * g_mu)
        rows.append({"lr": lr, "stable": stable, "gain": gain, "gain_ratio_to_muon": gain / g_mu if g_mu > 0 else float("nan"),
                     "feasible": feas})
    feas = [r for r in rows if r["feasible"]]
    choice = min(feas, key=lambda r: abs(math.log(max(r["gain"], 1e-12) / g_mu))) if feas else None
    return {"step": step, "muon_gain": g_mu, "min_frac": min_frac, "max_rise": max_rise, "runs": rows,
            "G2_pass": choice is not None, "chosen_lr": None if choice is None else choice["lr"],
            "rule": "feasible = stable and gain(24) >= 0.25 gain_muon(24); choose closest log-gain to Muon"}


def _load_interim(d: str, set_name: str) -> dict[int, dict]:
    import eval_windows as EW

    out = {}
    for f in os.listdir(d):
        if f.startswith("step") and f.endswith(f"__{set_name}.npz"):
            out[int(f[4:7])] = EW.load_result(os.path.join(d, f))
    if not out:
        raise SystemExit(f"no interim evals of set {set_name!r} in {d}")
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="C2 SGD-arm calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sgd-lr"); a.add_argument("--probe", required=True); a.add_argument("--subset", required=True)
    a.add_argument("--momentum", type=float, default=0.95); a.add_argument("--lr-muon", type=float, default=None)
    a.add_argument("--out", default=None)
    b = sub.add_parser("sgd-select"); b.add_argument("--base", required=True); b.add_argument("--muon", required=True)
    b.add_argument("--run", action="append", required=True, help="lr=interim_dir"); b.add_argument("--set", required=True)
    b.add_argument("--min-frac", type=float, default=0.25); b.add_argument("--max-rise", type=float, default=0.05)
    b.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "sgd-lr":
        txt = json.dumps(sgd_lr_grid(json.load(open(args.probe, encoding="utf-8")), args.subset, args.momentum, args.lr_muon), indent=1)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            open(args.out, "w", encoding="utf-8").write(txt)
        print(txt)
        return 0
    import eval_windows as EW

    runs = {float(r.split("=", 1)[0]): _load_interim(r.split("=", 1)[1], args.set) for r in args.run}
    res = sgd_select(EW.load_result(args.base), _load_interim(args.muon, args.set), runs,
                     min_frac=args.min_frac, max_rise=args.max_rise)
    txt = json.dumps(res, indent=1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w", encoding="utf-8").write(txt)
    print(txt)
    return 0 if (res["G2_pass"] or args.out) else 3     # with --out the decision lives in the file (json gate)


if __name__ == "__main__":
    sys.exit(main())
