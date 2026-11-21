"""mock_confirm.py -- SIMULATED outputs of the confirm queue with known generating values (tests of analyze_confirm.py
only; nothing produced here is a measurement).

  python mock_confirm.py simulate --root R [--scenario S.json]

For every run of the confirm plan the files that analyze_confirm.py reads are written under
outputs/confirm/<run>/m<micro>[c] (the run_queue layout, c = activation checkpointing: eval_<name>/<variant>__<set>.npz,
interim/stepNNN__<set>[__full].npz, interim/displacement.jsonl,
trainlog/train_steps.jsonl, subset_manifest.json), with results/queue_state_confirm.json, the realized-direction audit
files of the audit jobs, a 1 Hz power log with its README, the run logs of the confirm queue and the probe batch lines
of the forecast queue (EDGE). Every per-window CE is theta_0 CE + F + noise whose sample moments are exact (the first 64
windows and the rest are normalised separately), so every F -- and therefore every I, J and p -- is known by
construction. The generating values go to results/analysis_mock/truth.json.

Generating model (defaults; --scenario overrides):
  crossover blocks (C1, third base, C1-DE, NESTOR-books, dose): for each statistic kind k and draw d,
    I_k,d = beta_k,b * P_d + e_k,d (e from a per-(role, base, d, kind) seeded normal), realised as
    F_k(A, code) = f_code, F_k(B, code) = f_code * exp(I_k,d), F_k(A, math) = F_k(B, math) = f_math;
    sym kinds through the reflected file (2 F_sym - F_plus), rand through the antithetic pair (both sides = F_rand).
  dose: I_d = b1 P_d + b3 P_d^3 + e_d.   C2: r_arm = e_arm Delta_Q + e, F(H) = f exp(r/2), F(L) = f exp(-r/2).
  M12-b: IT_rand = it_rand, IT_learned = it_rand + delta_d.   DCLM: F(B)/F(A) = RO_DCLM(B)/RO_DCLM(A) * exp(e).
 (analyze_confirm_ext.py):
  C3: FULL F_DCLM = f_full[arch] * dom_factor * exp(seed_sd z); F(revert v) = (1 - a_v) F with a_v = a[arch][v] + a_sd z;
      per-layer Delta_l = w_l a_all F (1 + layer_gap), w_l proportional to l + 1; WD-only F_WD = wd_share F; G1 FULL runs
      keep their measured F (their revert files are written relative to it); freeze F = freeze_c * mean F_only(complement).
  C4: theta_0 DCLM routing of the evaluation windows (g0/<base>/routing__dclm_eval.npz when no covering file exists);
      per C1 draw d_B,w = d_A,w + (F_B - F_A) + b_d (Delta R_w - mean) + eta_w with eta orthogonal to [1, Delta R] and
      b_d = beta_C4,d * F-bar_d, so beta_C4,d is exact.  C1 B runs gain c1_gain_B x the A gain (matched gain).
  C5 French: F = F_c exp(run_sd z); hashed forecasts F-hat = F exp(err_sd z), posted 1 h before the first run.
  DCLM-CONT: F_cont per corpus.  bpb: n_bytes added to the theta_0 files.  C2 randdir (only if planned): r_rand.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import zlib
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

HEAD, N_TOK = 64, 4095
DEFAULT = {
    "seed": 11, "sd_window": 0.02, "noise_I": 0.05,
    "beta": {"raw": 0.62, "sym": 0.66, "pin": 0.58, "sym_pin": 0.6, "selpin": 0.6, "sym_selpin": 0.63, "rand": 0.7, "wd": 0.04},
    "beta_base_offset": {"moe_b2": -0.08, "nestor": -0.1},
    "f": {"raw": [0.012, 0.010], "sym": [0.0115, 0.0098], "pin": [0.011, 0.0095], "sym_pin": [0.0108, 0.0094],
          "selpin": [0.0112, 0.0096], "sym_selpin": [0.011, 0.0095], "rand": [0.009, 0.008], "wd": [0.004, 0.0035]},
    "f_dclm": 0.015, "f_code2": 0.011, "beta_code2": 0.5, "gain": 0.05, "dclm_noise": 0.02,
    "interim_exponent": 0.7,
    "dose": {"b1": 0.6, "b3": 0.0},
    "c2": {"e_sgd": 0.9, "e_muon": 0.3, "noise": 0.05, "f": {"8": 0.006, "24": 0.01, "48": 0.014, "96": 0.02},
           "kappa_disp": {"sgd": 0.8, "muon": 0.05}},
    "m12b": {"it_rand": 0.1, "delta": [0.12, 0.2, 0.15]},
    "diverge": [], "not_done": [],
    # (analyze_confirm_ext.py)
    "c3": {"f_full": {"moe": 0.03, "nestor": 0.04}, "gain_full": {"moe": 0.06, "nestor": 0.05},
           "dom_factor": {"books": 1.0, "code": 1.2, "math": 0.9, "german": 1.1}, "seed_sd": 0.05, "a_sd": 0.03,
           "a": {"moe": {"revert:moe": 0.55, "revert:kind:attn+embed_head+kind:norm": 0.35, "revert:embed_head": 0.1},
                 "nestor": {"revert:shared+pub+embed_head": 0.4, "revert:attn+priv": 0.5, "revert:attn": 0.2,
                            "revert:priv": 0.3, "revert:pub": 0.15, "revert:embed_head": 0.08}},
           "layer_gap": 0.1, "wd_share": 0.3, "freeze_c": 0.9},
    "dclm_cont": {"dclm": -0.004, "code": 0.002, "math": 0.001},
    "c4": {"beta": {"moe_b1": 0.8, "moe_b2": 0.6}, "beta_sd": 0.1, "ro_sd": 0.25, "eta_sd": 0.004},
    "c1_gain_B": 1.3,
    "c5": {"F": {"dclm": 0.012, "code": 0.010, "math": 0.008}, "run_sd": 0.2, "fit_sd": 0.3, "n_fit": 6,
           "err_sd": {"ro": 0.3, "trainable_fraction": 0.5, "k_step": 0.2}, "tamper": False},
    "bytes_per_token": {"dclm": 4.2, "pycode": 3.1, "code": 3.0, "math": 2.6, "books": 4.4, "german": 3.6, "french": 3.8},
    "c2_rand": {"e": 0.1, "f": 0.008, "noise": 0.02},
    "edge": {"t0": "2026-11-20 08:00:00", "run_s": 900, "pause_s": 60, "power_W": 200.0, "idle_W": 30.0,
             "gap_run": "first_c1", "gap_s": 8, "probe_s": 600, "probe_W": 180.0, "MiB": 8000.0},
}


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode("utf-8")))


def _norm(e: np.ndarray) -> np.ndarray:
    return (e - e.mean()) / e.std(ddof=1)


def run_ce(ce0: np.ndarray, F: float, sd: float, rng) -> np.ndarray:
    """theta_0 CE + F + noise with exact moments on the first HEAD windows and on the rest (so F is exact on both)."""
    n = ce0.size
    if n <= HEAD:
        return ce0 + F + sd * _norm(rng.standard_normal(n))
    e = np.concatenate([_norm(rng.standard_normal(HEAD)), _norm(rng.standard_normal(n - HEAD))])
    return ce0 + F + sd * e


def write_eval(path: str, index, hashes, ce, meta=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path + ".tmp.npz", index=np.asarray(index).astype(np.int64), hash=np.asarray(hashes).astype(np.uint64),
                        ce_sum=(np.asarray(ce) * N_TOK).astype(np.float64), n_tok=np.full(len(index), N_TOK, dtype=np.int64),
                        meta_json=np.array(json.dumps(meta or {"MOCK": True})))
    os.replace(path + ".tmp.npz", path)



def _attempt(j: dict) -> str:
    """run_queue.py name of a job's first attempt (its root dir and log suffix): m<micro>, + "c" with checkpointing."""
    return f"m{int(j.get('micro', 2))}{'c' if j.get('ckpt') else ''}"

class Sim:
    def __init__(self, root: str, scen: dict):
        self.root, self.s = os.path.abspath(root), scen
        self.plan = json.load(open(os.path.join(self.root, "confirm", "plan_confirm.json"), encoding="utf-8"))
        qp = self.plan.get("queue") or "confirm/queue_confirm.json"
        self.queue = json.load(open(self.p(qp), encoding="utf-8"))
        self.code = self.plan.get("code_stream") or "code"
        self.code2 = self.plan.get("code_secondary")
        self.runs = {r["id"]: r for r in self.plan["runs"]}
        self.jobs = {j["id"]: j for j in self.queue.get("jobs", [])}
        self.truth: dict = {"scenario": scen, "draws": {}, "c2": {}, "m12b": {}, "edge": {}, "runs": {}}
        self.F: dict = {}                       # (rid, set, variant) -> F
        self._base: dict = {}
        self._rt: dict = {}                     # C4: base -> (strata, s2 [L, W, N]) of the DCLM evaluation windows
        self._c4A: dict = {}                    # C4: (base, d) -> per-window d_A
        self.truth.update(c3={"runs": {}, "freeze": {}}, c4={}, c5={}, dclm_cont={}, bpb={}, c2_rand={})

    def p(self, path: str) -> str:
        path = str(path).replace("/", os.sep).replace("\\", os.sep)
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def base(self, base: str, set_name: str):
        key = (base, set_name)
        if key not in self._base:
            bp = self.p(self.plan["base_eval_files"][base][set_name])
            if not os.path.exists(bp):                        # mock world grid (mock_g1g2.World convention)
                r = np.random.default_rng(zlib.crc32(set_name.encode("utf-8")))
                idx = np.sort(r.choice(100000, size=500, replace=False))
                hs = r.integers(1, 2 ** 63, size=500, dtype=np.int64).astype(np.uint64)
                write_eval(bp, idx, hs, 3.0 + 0.5 * r.standard_normal(500))
            z = np.load(bp)
            self._base[key] = (np.asarray(z["index"]), np.asarray(z["hash"]), np.asarray(z["ce_sum"]) / np.asarray(z["n_tok"]))
        return self._base[key]

    def root_of(self, rid: str) -> str:
        j = self.jobs.get(rid) or {}
        out = j.get("out_root") or f"outputs/confirm/{rid}"
        return self.p(os.path.join(out, _attempt(j)))

    # ------------------------------------------------------------------ crossover draws
    def beta(self, kind: str, base: str) -> float:
        return float(self.s["beta"][kind]) + float(self.s["beta_base_offset"].get(base, 0.0))

    def draw_I(self, role: str, base: str, d, P: float) -> dict:
        out = {}
        for k in self.s["beta"]:
            e = float(_rng(role, base, d, k).normal(0.0, self.s["noise_I"]))
            if role == "dose" and k in ("raw", "sym"):
                out[k] = self.s["dose"]["b1"] * P + self.s["dose"]["b3"] * P ** 3 + e
            else:
                out[k] = self.beta(k, base) * P + e
        return out

    def side_F(self, kind: str, side: str, corpus_role: str, I: float) -> float:
        fc, fm = self.s["f"][kind]
        if corpus_role == "math":
            return fm
        return fc * (math.exp(I) if side == "B" else 1.0)

    def variant_F(self, run: dict, I: dict, variant: str, set_name: str, partner_ro=None) -> float:
        side = run["side"]
        role = "code" if set_name == self.code else "math" if set_name == "math" else None   # by set name, never by label
        if role in ("code", "math"):
            plus = {"ckpt": "raw", "pinned": "pin", "selpin": "selpin", "wdonly": "wd"}
            if variant in plus:
                return self.side_F(plus[variant], side, role, I[plus[variant]])
            minus = {"reflect": ("raw", "sym"), "reflect+pinned": ("pin", "sym_pin"), "reflect+selpin": ("selpin", "sym_selpin")}
            if variant in minus:
                kp, ks = minus[variant]
                return 2 * self.side_F(ks, side, role, I[ks]) - self.side_F(kp, side, role, I[kp])
            if variant.startswith("randdir"):
                return self.side_F("rand", side, role, I["rand"])
        if set_name == "dclm":
            ro = (run.get("subset") or {}).get("ro") or {}
            fA = self.s["f_dclm"]
            if side == "A" or not partner_ro:
                return fA
            e = float(_rng("dclm", run["id"]).normal(0.0, self.s["dclm_noise"]))
            return fA * (ro.get("dclm", 1.0) / partner_ro.get("dclm", 1.0)) * math.exp(e)
        if set_name == self.code2:
            return self.s["f_code2"] * (math.exp(I["code2"]) if side == "B" else 1.0)
        g = self.s["gain"] * (float(self.s.get("c1_gain_B", 1.0)) if (run.get("role") == "c1" and side == "B") else 1.0)
        return -g                                           # target sets (books, german): a gain

    def write_crossover_run(self, run: dict, I: dict, partner: dict | None) -> None:
        root = self.root_of(run["id"])
        sd = self.s["sd_window"]
        pro = ((partner or {}).get("subset") or {}).get("ro")
        for ev in run.get("evals") or []:
            vs = ev.get("variants", "ckpt")
            for v in (vs.split(",") if isinstance(vs, str) else vs):
                for sn in ev.get("sets") or []:
                    idx, hs, ce0 = self.base(run["base"], sn)
                    F = self.variant_F(run, I, v, sn, pro)
                    self.F[(run["id"], sn, v)] = F
                    if run.get("role") == "c1" and sn == "dclm" and v == "ckpt":
                        ce = self._c4_ce(run, partner, F, ce0)
                    else:
                        ce = run_ce(ce0, F, sd, _rng("ce", run["id"], v, sn))
                    write_eval(os.path.join(root, f"eval_{ev.get('name', 'main')}", f"{v.replace(':', '-')}__{sn}.npz"),
                               idx, hs, ce)
        g = self.s["interim_exponent"]
        for sn in run.get("interim_sets") or []:
            idx, hs, ce0 = self.base(run["base"], sn)
            F96 = self.F.get((run["id"], sn, "ckpt"), self.variant_F(run, I, "ckpt", sn, pro))
            for k in run.get("interim_steps") or []:
                ce = ce0[:HEAD].copy() if k == 0 else ce0[:HEAD] + F96 * (k / 96.0) ** g + \
                    sd * _norm(_rng("int", run["id"], sn, k).standard_normal(HEAD))
                write_eval(os.path.join(root, "interim", f"step{int(k):03d}__{sn}.npz"), idx[:HEAD], hs[:HEAD], ce)
        self.common(run)

    def common(self, run: dict, diverge: bool | None = None) -> None:
        import mock_g1g2 as MG
        root = self.root_of(run["id"])
        dv = run["id"] in self.s["diverge"] if diverge is None else diverge
        steps = int(run.get("stop_after") or run.get("steps", 96))     # P-16: a stopped run logs its steps 1..stop only
        MG._train_log(os.path.join(root, "trainlog", "train_steps.jsonl"), steps, int(run.get("warmup", 10)), diverge=dv)
        n = int((run.get("subset") or {}).get("n_units") or 72)
        fam = "priv" if run.get("arch") == "nestor" else "moe"
        units = [f"{fam}:b0:l{i // 6}:e{i % 6}" for i in range(n)]
        man = {"units_selected": n, "units_trainable": units, "units_total": 1017,
               "elements": {"trainable": n * 921600, "total": 373387389}, "MOCK": True}
        os.makedirs(root, exist_ok=True)
        json.dump(man, open(os.path.join(root, "subset_manifest.json"), "w", encoding="utf-8"), indent=1)

    # ------------------------------------------------------------------ blocks
    def crossovers(self) -> None:
        by = {}
        for r in self.plan["runs"]:
            if r.get("role") in ("c1", "c1_b3", "c1de", "c1_nestor", "dose") and r.get("side") in ("A", "B"):
                by.setdefault((r["role"], r["base"], int(r["d"])), {})[r["side"]] = r
        for (role, base, d), sides in sorted(by.items()):
            P = float(sides["A"]["P"])
            I = self.draw_I(role, base, d, P)
            I["code2"] = self.s["beta_code2"] * P
            self.truth["draws"][f"{role}|{base}|{d}"] = {"P": P, "I": I, "A": sides["A"]["id"], "B": sides.get("B", {}).get("id")}
            for s in ("A", "B"):
                if s in sides:
                    self.write_crossover_run(sides[s], I, sides["A"] if s == "B" else None)

    def m12b(self) -> None:
        sd = self.s["sd_window"]
        for r in sorted((r for r in self.plan["runs"] if r.get("role") == "m12b"), key=lambda x: int(x["d"])):
            b = r["c1_partner"]
            d = int(r["d"])
            delta = float(self.s["m12b"]["delta"][(d - 1) % len(self.s["m12b"]["delta"])])
            itr = float(self.s["m12b"]["it_rand"])
            itl = itr + delta
            lb = math.log(self.F[(b, "dclm", "ckpt")]) - math.log(self.F[(b, self.code, "ckpt")])
            hb = [v for v in self.runs[b]["evals"] if v["name"] == "rand"][0]["variants"].split(",")[0]
            lbr = math.log(self.F[(b, "dclm", hb)]) - math.log(self.F[(b, self.code, hb)])
            fc = 0.012
            Fg = {("ckpt", self.code): fc, ("ckpt", "dclm"): fc * math.exp(itl + lb),
                  ("rand", self.code): fc, ("rand", "dclm"): fc * math.exp(itr + lbr)}
            root = self.root_of(r["id"])
            for ev in r.get("evals") or []:
                for v in ev["variants"].split(","):
                    for sn in ev["sets"]:
                        kind = "rand" if v.startswith("randdir") else "ckpt"
                        F = Fg.get((kind, sn), 0.01 if sn != "german" else -self.s["gain"])
                        idx, hs, ce0 = self.base(r["base"], sn)
                        write_eval(os.path.join(root, f"eval_{ev['name']}", f"{v.replace(':', '-')}__{sn}.npz"), idx, hs,
                                   run_ce(ce0, F, sd, _rng("ce", r["id"], v, sn)))
            for sn in r.get("interim_sets") or []:
                idx, hs, ce0 = self.base(r["base"], sn)
                for k in r.get("interim_steps") or []:
                    write_eval(os.path.join(root, "interim", f"step{int(k):03d}__{sn}.npz"), idx[:HEAD], hs[:HEAD],
                               ce0[:HEAD] if k == 0 else ce0[:HEAD] - self.s["gain"] * k / 96)
            self.common(r)
            self.truth["m12b"][str(d)] = {"IT_learned": itl, "IT_rand": itr, "Delta": delta}

    def c2(self) -> None:
        c = self.s["c2"]
        sd = self.s["sd_window"]
        cells = {}
        for r in self.plan["runs"]:
            if r.get("role") in ("c2_sgd", "c2_muon"):
                cells.setdefault((r["base"], int(r["d"])), {})[(r["arm"], r["side"])] = r
        for (base, d), rr in sorted(cells.items()):
            DQ = float(rr[("sgd", "H")]["Delta_Q"])
            tr = {"Delta_Q": DQ, "r": {}}
            stops = {arm: int(rr[(arm, "H")].get("stop_after") or 0) for arm in ("sgd", "muon")}   # P-16 / MF-P1 (A)
            stop = min(stops.values()) if all(stops.values()) else 0       # full evaluations at s* (= the Muon stop)
            fsteps = {stp: f for stp, f in c["f"].items() if not stop or int(stp) <= stop}
            tr["stop_after"] = stop or None
            tr["stop_after_arm"] = {a: (s or None) for a, s in stops.items()}
            for arm in ("sgd", "muon"):
                st_a = stops[arm]
                steps_a = dict(fsteps)
                if st_a and st_a > (stop or 0):                            # the final checkpoint of an arm stopped later
                    steps_a.update({stp: f for stp, f in c["f"].items() if int(stp) == st_a})
                for stp, f in steps_a.items():
                    rr_ = c[f"e_{arm}"] * DQ + float(_rng("c2", base, d, arm, stp).normal(0.0, c["noise"]))
                    tr["r"][f"{arm}|{stp}"] = rr_
                    for side, sg in (("H", 0.5), ("L", -0.5)):
                        run = rr[(arm, side)]
                        root = self.root_of(run["id"])
                        F = f * math.exp(sg * rr_)
                        idx, hs, ce0 = self.base(base, "dclm")
                        ce = run_ce(ce0, F, sd, _rng("c2ce", run["id"], stp))
                        if int(stp) == 96:
                            write_eval(os.path.join(root, "eval_main", "ckpt__dclm.npz"), idx, hs, ce)
                        else:
                            if stp in fsteps:                          # full evaluation (s* = 8) and its head
                                write_eval(os.path.join(root, "interim", f"step{int(stp):03d}__dclm__full.npz"), idx, hs, ce)
                            write_eval(os.path.join(root, "interim", f"step{int(stp):03d}__dclm.npz"), idx[:HEAD], hs[:HEAD], ce[:HEAD])
                            if st_a and int(stp) == st_a:              # the final checkpoint IS the step-`stop` state
                                write_eval(os.path.join(root, "eval_main", "ckpt__dclm.npz"), idx, hs, ce)
            for arm in ("sgd", "muon"):
                for side, sg in (("H", 0.5), ("L", -0.5)):
                    run = rr[(arm, side)]
                    root = self.root_of(run["id"])
                    gi, gh, g0 = self.base(base, "german")
                    write_eval(os.path.join(root, "eval_main", "ckpt__german.npz"), gi, gh,
                               run_ce(g0, -self.s["gain"], sd, _rng("c2g", run["id"])))
                    gfull = {}
                    for stp in fsteps:
                        if int(stp) != 96:
                            gfull[int(stp)] = run_ce(g0, -self.s["gain"] * int(stp) / 96, sd, _rng("c2gi", run["id"], stp))
                            write_eval(os.path.join(root, "interim", f"step{int(stp):03d}__german__full.npz"), gi, gh,
                                       gfull[int(stp)])
                    for k in [int(x) for x in run.get("interim_steps") or []]:  # German heads (C2.sgd_step24_gain)
                        if k == 96 or (stops[arm] and k > stops[arm]):
                            continue
                        ce_h = g0[:HEAD] if k == 0 else gfull[k][:HEAD] if k in gfull else g0[:HEAD] - self.s["gain"] * k / 96
                        write_eval(os.path.join(root, "interim", f"step{k:03d}__german.npz"), gi[:HEAD], gh[:HEAD], ce_h)
                    for k in (0,):
                        di, dh, d0 = self.base(base, "dclm")
                        write_eval(os.path.join(root, "interim", "step000__dclm.npz"), di[:HEAD], dh[:HEAD], d0[:HEAD])
                    lines = []
                    for stp in [x for x in (0, 1, 8, 24, 48, 96) if not stops[arm] or x <= stops[arm]]:
                        D = 0.0 if stp == 0 else 0.1 * stp ** 0.5 * math.exp(sg * c["kappa_disp"][arm] * DQ)
                        lines.append(json.dumps({"step": stp, "total_sq": D * D, "by_kind_sq": {"moe": D * D}, "by_unit_sq": {}}))
                    os.makedirs(os.path.join(root, "interim"), exist_ok=True)
                    open(os.path.join(root, "interim", "displacement.jsonl"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
                    self.common(run)
            mr = {s: rr[("muon", s)] for s in ("H", "L")}
            if any(ev.get("name") == "rand" for ev in mr["H"].get("evals") or []):     # only if randdir is planned
                cr = self.s["c2_rand"]
                r_r = cr["e"] * DQ + float(_rng("c2rand", base, d).normal(0.0, cr["noise"]))
                for side, sg in (("H", 0.5), ("L", -0.5)):
                    run = mr[side]
                    root = self.root_of(run["id"])
                    for ev in run.get("evals") or []:
                        if ev.get("name") != "rand":
                            continue
                        for v in ev["variants"].split(","):
                            for sn in ev["sets"]:
                                idx, hs, ce0 = self.base(base, sn)
                                write_eval(os.path.join(root, "eval_rand", f"{v.replace(':', '-')}__{sn}.npz"), idx, hs,
                                           run_ce(ce0, cr["f"] * math.exp(sg * r_r), sd, _rng("c2r", run["id"], v, sn)))
                fin = stops["muon"] or 96                                 # the Muon runs' final checkpoint step
                key = next(k for k in tr["r"] if k.split("|")[0] == "muon" and int(k.split("|")[1]) == fin)
                tr["rand"] = {"r_rand": r_r, "b_muon": tr["r"][key] / DQ, "b_rand": r_r / DQ, "step": fin}
                self.truth["c2_rand"][f"{base}|{d}"] = tr["rand"]
            tr["D_log_ratio"] = {arm: c["kappa_disp"][arm] * DQ for arm in ("sgd", "muon")}
            self.truth["c2"][f"{base}|{d}"] = tr

    def others(self) -> None:
        for r in self.plan["runs"]:
            role = r.get("role")
            if role in ("c1", "c1_b3", "c1de", "c1_nestor", "dose", "m12b", "c2_sgd", "c2_muon", "freeze"):
                continue
            if role == "full":
                self.write_full(r)
            elif role == "dclm_cont":
                self.write_dclm_cont(r)
            elif r.get("block") == "C5":
                self.write_c5_run(r)
            else:
                self.common(r, diverge=False)

    # ------------------------------------------------------------------ helpers
    def _arch(self, run: dict) -> str:
        return run.get("arch") or ("nestor" if run.get("base") == "nestor" else "moe")

    def _write_interim_heads(self, run: dict) -> None:
        root, sd, g = self.root_of(run["id"]), self.s["sd_window"], self.s["interim_exponent"]
        for sn in run.get("interim_sets") or []:
            idx, hs, ce0 = self.base(run["base"], sn)
            F96 = self.F[(run["id"], sn, "ckpt")]
            for k in run.get("interim_steps") or []:
                ce = ce0[:HEAD].copy() if k == 0 else ce0[:HEAD] + F96 * (k / 96.0) ** g + \
                    sd * _norm(_rng("int", run["id"], sn, k).standard_normal(HEAD))
                write_eval(os.path.join(root, "interim", f"step{int(k):03d}__{sn}.npz"), idx[:HEAD], hs[:HEAD], ce)

    def _shares(self, rid: str, arch: str) -> dict:
        c = self.s["c3"]
        return {v: float(a) + c["a_sd"] * float(_rng("c3a", rid, v).standard_normal()) for v, a in c["a"][arch].items()}

    @staticmethod
    def _layer_group(v: str) -> tuple[str, str]:
        parts = v.split(":", 1)[1].split("+")
        gs, ts = zip(*(p.split("@", 1) for p in parts))
        return "+".join(gs), ts[0]

    def _layer_F(self, F: float, a: dict, variants: list[str]) -> dict:
        """F(revert l) = F - w_l a_all F (1 + layer_gap), w_l proportional to l + 1 over the run's per-layer variants."""
        grp = {self._layer_group(v)[0] for v in variants}
        assert len(grp) == 1, grp
        a_all = a[f"revert:{grp.pop()}"]
        w = np.arange(1, len(variants) + 1, dtype=np.float64)
        w /= w.sum()
        gap = self.s["c3"]["layer_gap"]
        return {v: F - float(w[i]) * a_all * F * (1.0 + gap) for i, v in enumerate(variants)}

    def write_full(self, run: dict) -> None:
        c, arch, rid = self.s["c3"], self._arch(run), run["id"]
        dom = run.get("target")
        F = c["f_full"][arch] * c["dom_factor"].get(dom, 1.0) * math.exp(c["seed_sd"] * float(_rng("c3F", rid).standard_normal()))
        G = c["gain_full"][arch] * c["dom_factor"].get(dom, 1.0)
        tset = self.code if dom == "code" else dom
        a = self._shares(rid, arch)
        root, sd = self.root_of(rid), self.s["sd_window"]
        tr = {"source": "confirm", "base": run["base"], "arch": arch, "domain": dom, "seed": run.get("seed"), "F": F, "G": G,
              "a": a, "layers": {}}
        for ev in run.get("evals") or []:
            vs = ev["variants"].split(",") if isinstance(ev.get("variants"), str) else list(ev.get("variants") or ["ckpt"])
            lay = self._layer_F(F, a, vs) if ev.get("name") == "revert_layer" else {}
            for v in vs:
                for sn in ev.get("sets") or []:
                    idx, hs, ce0 = self.base(run["base"], sn)
                    if sn == "dclm":
                        f = F if v == "ckpt" else F * c["wd_share"] if v == "wdonly" else lay[v] if v in lay else F * (1.0 - a[v])
                    else:
                        f = -G if sn == tset else 0.01
                    self.F[(rid, sn, v)] = f
                    if v in lay:
                        tr["layers"][v] = f
                    write_eval(os.path.join(root, f"eval_{ev.get('name', 'main')}", f"{v.replace(':', '-')}__{sn}.npz"),
                               idx, hs, run_ce(ce0, f, sd, _rng("c3ce", rid, v, sn)))
        self._write_interim_heads(run)
        self.common(run, diverge=False)
        self.truth["c3"]["runs"][rid] = tr

    def _g1(self):
        if not hasattr(self, "_g1c"):
            pp, sp = self.p("g1/plan_g1.json"), self.p("results/queue_state_g1.json")
            self._g1c = (json.load(open(pp, encoding="utf-8")) if os.path.exists(pp) else {"runs": []},
                         json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else {})
        return self._g1c

    def _g1_F(self, rid: str, gr: dict) -> float:
        """Measured F_DCLM of a G1 FULL run (its eval file against the confirm theta_0 file, paired by grid index)."""
        idx, hs, ce0 = self.base(gr["base"], "dclm")
        root = self.p(self._g1()[1][rid]["root"])
        z = np.load(os.path.join(root, gr.get("eval_dir", "eval_main"), "ckpt__dclm.npz"))
        pos = {int(i): k for k, i in enumerate(np.asarray(z["index"]))}
        k = np.array([pos[int(i)] for i in idx])
        assert np.array_equal(np.asarray(z["hash"]).astype(np.uint64)[k], np.asarray(hs).astype(np.uint64))
        return float(np.mean(np.asarray(z["ce_sum"])[k] / np.asarray(z["n_tok"])[k] - ce0))

    def g1_reverts(self) -> None:
        """Revert-only evaluations of the G1 FULL runs (c3_revert_<rid>, c3_layer_revert_<rid>), relative to the
        measured F of each G1 run."""
        g1runs = {r["id"]: r for r in self._g1()[0].get("runs", [])}
        sd, c = self.s["sd_window"], self.s["c3"]
        for jid, job in sorted(self.jobs.items()):
            if not (jid.startswith("c3_revert_") or jid.startswith("c3_layer_revert_")):
                continue
            rid = jid.split("revert_", 1)[1]
            gr = g1runs[rid]
            arch = gr.get("arch") or ("nestor" if gr["base"] == "nestor" else "moe")
            tr = self.truth["c3"]["runs"].setdefault(rid, {"source": "G1", "base": gr["base"], "arch": arch,
                                                          "domain": gr.get("target"), "seed": gr.get("seed"),
                                                          "F": self._g1_F(rid, gr), "a": self._shares(rid, arch),
                                                          "layers": {}})
            args = [str(x) for x in job.get("args") or []]
            out = self.p(args[args.index("--out") + 1])
            vs = args[args.index("--variants") + 1].split(",")
            lay = self._layer_F(tr["F"], tr["a"], vs) if jid.startswith("c3_layer_revert_") else {}
            idx, hs, ce0 = self.base(gr["base"], "dclm")
            for v in vs:
                f = lay[v] if v in lay else tr["F"] * c["wd_share"] if v == "wdonly" else tr["F"] * (1.0 - tr["a"][v])
                if v in lay:
                    tr["layers"][v] = f
                write_eval(os.path.join(out, f"{v.replace(':', '-')}__dclm.npz"), idx, hs, run_ce(ce0, f, sd, _rng("g1rev", rid, v)))

    def freezes(self) -> None:
        c, sd = self.s["c3"], self.s["sd_window"]
        for run in self.plan["runs"]:
            if run.get("role") != "freeze":
                continue
            arch, rid = self._arch(run), run["id"]
            rv = "revert:moe" if arch == "moe" else "revert:attn+priv"
            comps = [(k, t) for k, t in self.truth["c3"]["runs"].items() if t["base"] == run["base"] and t["domain"] == run.get("target")]
            F_only = [t["F"] * (1.0 - t["a"][rv]) for _, t in comps]
            Ff = c["freeze_c"] * float(np.mean(F_only)) if F_only else 0.01
            self.truth["c3"]["freeze"][arch] = {"run": rid, "F_freeze": Ff, "comparators": [k for k, _ in comps],
                                                "F_only_mean": float(np.mean(F_only)) if F_only else None,
                                                "F_full_mean": float(np.mean([t["F"] for _, t in comps])) if comps else None}
            root = self.root_of(rid)
            for ev in run.get("evals") or []:
                for sn in ev.get("sets") or []:
                    idx, hs, ce0 = self.base(run["base"], sn)
                    f = Ff if sn == "dclm" else -c["gain_full"][arch] if sn == run.get("target") else 0.01
                    self.F[(rid, sn, "ckpt")] = f
                    write_eval(os.path.join(root, f"eval_{ev.get('name', 'main')}", f"ckpt__{sn}.npz"), idx, hs,
                               run_ce(ce0, f, sd, _rng("frz", rid, sn)))
            self._write_interim_heads(run)
            self.common(run, diverge=False)

    def write_dclm_cont(self, run: dict) -> None:
        root, sd = self.root_of(run["id"]), self.s["sd_window"]
        Fc = {}
        for ev in run.get("evals") or []:
            for sn in ev.get("sets") or []:
                idx, hs, ce0 = self.base(run["base"], sn)
                f = float(self.s["dclm_cont"].get("code" if sn == self.code else sn, 0.0))
                Fc[sn] = f
                self.F[(run["id"], sn, "ckpt")] = f
                write_eval(os.path.join(root, f"eval_{ev.get('name', 'main')}", f"ckpt__{sn}.npz"), idx, hs,
                           run_ce(ce0, f, sd, _rng("cont", run["id"], sn)))
        self.common(run, diverge=False)
        self.truth["dclm_cont"][run["base"]] = {"run": run["id"], "F": Fc}

    # ------------------------------------------------------------------ C5 French
    def write_c5_run(self, run: dict) -> None:
        c5, root, sd = self.s["c5"], self.root_of(run["id"]), self.s["sd_window"]
        Ft = {}
        for ev in run.get("evals") or []:
            for sn in ev.get("sets") or []:
                idx, hs, ce0 = self.base(run["base"], sn)
                corp = "code" if sn == self.code else sn
                if corp in c5["F"]:
                    f = c5["F"][corp] * math.exp(c5["run_sd"] * float(_rng("c5F", run["id"], corp).standard_normal()))
                    Ft[corp] = f
                else:
                    f = -self.s["gain"]
                self.F[(run["id"], sn, "ckpt")] = f
                write_eval(os.path.join(root, f"eval_{ev.get('name', 'main')}", f"ckpt__{sn}.npz"), idx, hs,
                           run_ce(ce0, f, sd, _rng("c5ce", run["id"], sn)))
        self._write_interim_heads(run)
        self.common(run, diverge=False)
        self.truth["c5"].setdefault("F", {})[run["id"]] = Ft

    def c5_forecasts(self) -> None:
        """confirm/c5_french_forecasts.json (forecasts.py c5-french layout, content sha256) with known errors, its hash
        posted to logs/c5_forecast_hashes.log one hour before the first queue job starts."""
        import forecasts as FCM
        c5 = self.s["c5"]
        french = [r["id"] for r in self.plan["runs"] if r.get("block") == "C5" and r.get("role") in ("c5_full", "c5_pair")]
        if not french:
            return
        Ft = self.truth["c5"]["F"]
        table = []
        for rid in french:
            for corp in ("dclm", "code", "math"):
                fc = {nm: Ft[rid][corp] * math.exp(sd_ * float(_rng("c5err", nm, rid, corp).standard_normal()))
                      for nm, sd_ in c5["err_sd"].items()}
                table.append({"run": rid, "corpus": corp, "forecasts": fc})
        table[[i for i, t in enumerate(table) if t["corpus"] == "math"][1]]["forecasts"]["k_step"] = None
        fit = [{"run": f"mock_fit_{i}", "arch": "moe", "corpus": corp,
                "F": c5["F"][corp] * math.exp(c5["fit_sd"] * float(_rng("c5fit", i, corp).standard_normal()))}
               for corp in ("dclm", "code", "math") for i in range(int(c5["n_fit"]))]
        doc = {"kind": "C5 French forecasts (prereg v1.2 sec. 12) -- MOCK", "protocol": "MOCK (mock_confirm.py)",
               "drive": {"run": next((r["id"] for r in self.plan["runs"] if r.get("block") == "C5" and r.get("role") == "drive"), None)},
               "fit_runs": fit, "scalars": {}, "forecasts": table, "french_runs": french,
               "plan_content_sha256": self.plan.get("sha256"), "code_sha256": "MOCK"}
        doc["sha256"] = FCM.content_sha(doc)
        t0 = self._t0() - timedelta(hours=1)
        doc["created"] = t0.strftime("%Y-%m-%dT%H:%M:%S")
        out = self.p("confirm/c5_french_forecasts.json")
        if c5.get("tamper"):                                         # a forecast changed after hashing
            doc["forecasts"][0]["forecasts"]["ro"] *= 1.5
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(FCM.canonical(doc) + "\n")
        os.makedirs(self.p("logs"), exist_ok=True)
        with open(self.p("logs/c5_forecast_hashes.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{doc['created']} {doc['sha256']} {os.path.abspath(out)}\n")
        self.truth["c5"].update(forecasts=table, fit=fit, french=french, tampered=bool(c5.get("tamper")))

    # ------------------------------------------------------------------ C4 routing and per-window DCLM
    def _entry(self, run: dict) -> dict:
        m = json.load(open(self.p(run["draw_manifest"]), encoding="utf-8"))
        return next(e for e in m["entries"] if int(e["d"]) == int(run["d"]))

    def _c4_routing(self, base: str, entry: dict):
        """(strata, s2 [L, W, N] in theta_0-file window order) of the DCLM evaluation windows: an existing covering
        routing file (the analysis's search order), else a new g0/<base>/routing__dclm_eval.npz."""
        if base in self._rt:
            return self._rt[base]
        import crossover as CX
        import eval_windows as EW
        idx, hs, ce0 = self.base(base, "dclm")
        fam = entry.get("family", "moe")
        sel = entry["sources"][self.code]["selection"]["file"]
        dirs = [os.path.dirname(self.p(sel)), self.p(f"g0/{base}")]
        for d in dict.fromkeys(dirs):
            for fn in ("routing__dclm_eval.npz", "routing__dclm.npz"):
                f = os.path.join(d, fn)
                if os.path.exists(f):
                    r = EW.load_result(f)
                    pos = {int(i): k for k, i in enumerate(np.asarray(r["index"]))}
                    hh = np.asarray(r["hash"]).astype(np.uint64)
                    if all(int(i) in pos and hh[pos[int(i)]] == np.uint64(h) for i, h in zip(idx, np.asarray(hs).astype(np.uint64))):
                        st, _ = CX.reg_strata(r["routing"], fam, int(r["meta"].get("set", {}).get("block", 4096)))
                        k = np.array([pos[int(i)] for i in idx])
                        self._rt[base] = (st, np.stack([np.asarray(r["routing"][lk], dtype=np.float64)[k, 2, :] for lk in st]))
                        return self._rt[base]
        r0 = EW.load_result(self.p(sel))
        st, _ = CX.reg_strata(r0["routing"], fam, int(r0["meta"].get("set", {}).get("block", 4096)))
        N = int(r0["routing"][st[0]].shape[-1])
        W, L = len(idx), len(st)
        rng = _rng("c4rt", base)
        lvl = rng.gamma(2.0, 1.0, size=(L, 1, N))
        s2 = lvl * np.exp(self.s["c4"]["ro_sd"] * rng.standard_normal((L, W, N)))
        arr = {"index": np.asarray(idx).astype(np.int64), "hash": np.asarray(hs).astype(np.uint64),
               "ce_sum": (np.asarray(ce0) * N_TOK).astype(np.float64), "n_tok": np.full(W, N_TOK, dtype=np.int64),
               "meta_json": np.array(json.dumps({"MOCK": True, "set": {"name": "dclm_eval", "block": 4096, "windows": W}}))}
        for l, lk in enumerate(st):
            a = np.zeros((W, 3, N))
            a[:, 0, :] = 4096.0 * 6 / N
            a[:, 1, :] = np.sqrt(s2[l]) * 4096.0
            a[:, 2, :] = s2[l] * 4096.0
            arr[f"rt|{lk}"] = a
        out = self.p(f"g0/{base}/routing__dclm_eval.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out + ".tmp.npz", **arr)
        os.replace(out + ".tmp.npz", out)
        self._rt[base] = (st, s2 * 4096.0)
        return self._rt[base]

    def _c4_ce(self, run: dict, partner: dict | None, F: float, ce0: np.ndarray) -> np.ndarray:
        """C1 DCLM per-window CE with an exact per-draw C4 slope (see the module docstring)."""
        key, sd = (run["base"], int(run["d"])), self.s["sd_window"]
        if run["side"] == "A" or partner is None:
            dA = run_ce(ce0, F, sd, _rng("ce", run["id"], "ckpt", "dclm")) - ce0
            self._c4A[key] = dA
            return ce0 + dA
        entry = self._entry(run)
        st, s2 = self._c4_routing(run["base"], entry)

        def ro(S):
            S = np.asarray(S, dtype=np.int64)
            return np.stack([s2[l][:, S[l]].sum(1) for l in range(s2.shape[0])]).sum(0) / s2.sum(axis=(0, 2))

        dR = np.log(ro(entry["B"])) - np.log(ro(entry["A"]))
        x = dR - dR.mean()
        dA = self._c4A[key]
        FA = self.F[(partner["id"], "dclm", "ckpt")]
        c4 = self.s["c4"]
        beta = float(c4["beta"].get(run["base"], 0.7)) + c4["beta_sd"] * float(_rng("c4b", *key).standard_normal())
        b = beta * 0.5 * (FA + F)
        e = _rng("c4e", *key).standard_normal(x.size)
        Xm = np.stack([np.ones_like(x), x], 1)
        e = e - Xm @ np.linalg.lstsq(Xm, e, rcond=None)[0]
        e = c4["eta_sd"] * e / e.std(ddof=1)
        dB = dA + (F - FA) + b * x + e
        self.truth["c4"][f"{run['base']}|{run['d']}"] = {"beta_C4": beta, "b": b, "F_bar": 0.5 * (FA + F)}
        return ce0 + dB

    # ------------------------------------------------------------------ bytes per window
    def add_bytes(self) -> None:
        """n_bytes (eval_windows --bytes field) added to the theta_0 files under the root; kappa = sum n_tok / sum n_bytes / ln 2."""
        bpt = self.s["bytes_per_token"]
        for base, sets in sorted((self.plan.get("base_eval_files") or {}).items()):
            for sn, rel in sorted(sets.items()):
                fp = self.p(rel)
                if os.path.isabs(rel) or not os.path.exists(fp) or sn not in bpt:
                    continue
                z = np.load(fp)
                arr = {k: z[k] for k in z.files}
                z.close()
                if "n_bytes" not in arr:
                    r = _rng("bytes", base, sn)
                    nt = np.asarray(arr["n_tok"], dtype=np.int64)
                    arr["n_bytes"] = np.maximum(1, np.round(nt * bpt[sn] * (1.0 + 0.1 * r.standard_normal(nt.size)))).astype(np.int64)
                    np.savez_compressed(fp + ".tmp.npz", **arr)
                    os.replace(fp + ".tmp.npz", fp)
                self.truth["bpb"][f"{base}|{sn}"] = float(np.asarray(arr["n_tok"]).sum()) / float(arr["n_bytes"].sum()) / math.log(2.0)

    def audit(self) -> None:
        for j in self.queue.get("jobs", []):
            a = j.get("argv") or []
            if j.get("kind") != "cmd" or "assumption_audit.py" not in " ".join(a):
                continue
            base, out = a[a.index("--base") + 1], self.p(a[a.index("--out") + 1])
            os.makedirs(out, exist_ok=True)
            eps, nw = 0.01, 24
            corpora = ["dclm", "code", "math"]
            G = 24
            rng = _rng("audit", base)
            ag = rng.normal(0, 0.5, G)
            bc = np.array([0.0, 0.3, -0.2])
            groups = [{"name": f"g{g:02d}", "side": "A" if g < 12 else "B", "contrast": float(rng.normal()), "rho": 1.0}
                      for g in range(G)]
            arr = {"eps": np.array(eps), "corpus": np.array([c for c in corpora for _ in range(nw)])}
            for gi, g in enumerate(groups):
                Q = np.concatenate([np.full(nw, math.exp(ag[gi] + bc[ci])) for ci in range(3)])
                gam = 0.05 * Q
                arr[f"{g['name']}|{eps:.6g}|+1|dce"] = eps * eps * Q + eps * gam
                arr[f"{g['name']}|{eps:.6g}|-1|dce"] = eps * eps * Q - eps * gam
            np.savez_compressed(os.path.join(out, "audit_realized_windows.npz"), **arr)
            json.dump({"base": base, "groups": groups, "eps_source": "MOCK", "sha256": "MOCK"},
                      open(os.path.join(out, "audit_realized_groups.json"), "w", encoding="utf-8"), indent=1)
            json.dump({"MOCK": True}, open(os.path.join(out, "audit_realized.json"), "w", encoding="utf-8"))
            self.truth.setdefault("audit", {})[base] = {"B5": "approximately separable", "A0_r": 0.05}

    # ------------------------------------------------------------------ queue state and EDGE logs
    def _t0(self) -> datetime:
        """Start of the mock confirm queue (local wall clock): the scenario's edge.t0, but never earlier than one hour
        after this simulation (else the next day at the same clock time), so the mock runs start after the world's
        freeze on any build date; freeze-verify checks it (spec check 4). Unchanged while edge.t0 is ahead."""
        t0 = datetime.strptime(self.s["edge"]["t0"], "%Y-%m-%d %H:%M:%S")
        now = datetime.now().replace(microsecond=0)
        if t0 < now + timedelta(hours=1):
            t0 = datetime.combine(now.date() + timedelta(days=1), t0.time())
        return t0

    def state_and_logs(self) -> None:
        e = self.s["edge"]
        t0 = self._t0()
        state, windows = {}, []
        t = t0 + timedelta(seconds=60)
        logs = self.p(self.queue.get("logs") or "logs/queue_confirm")
        os.makedirs(logs, exist_ok=True)
        gap_run = e.get("gap_run")
        if gap_run == "first_c1":
            gap_run = next((r["id"] for r in self.plan["runs"] if r.get("role") == "c1"), None)
        for j in self.queue.get("jobs", []):
            rid = j["id"]
            st = "done" if rid not in self.s["not_done"] else "failed"
            rec = {"status": st, "start": time.mktime(t.timetuple()), "end": time.mktime(t.timetuple()) + 1.0,
                   "hours": float(j.get("est_h") or 0.0), "mock": True}
            if j.get("kind") == "train":
                ts, te = t, t + timedelta(seconds=e["run_s"])
                rec.update(root=self.root_of(rid), audit_ok=True, start=time.mktime(ts.timetuple()), end=time.mktime(te.timetuple()))
                windows.append((rid, ts, te))
                self._run_log(os.path.join(logs, f"{rid}_{_attempt(j)}.log"), rid, ts, te, j)
                t = te + timedelta(seconds=e["pause_s"])
            state[rid] = rec
        os.makedirs(self.p("results"), exist_ok=True)
        json.dump(state, open(self.p(self.plan.get("state") or "results/queue_state_confirm.json"), "w", encoding="utf-8"), indent=1)
        # P-16: the core marker, written by the queue's real gate function after the last core job
        gate = next((j for j in self.queue.get("jobs", []) if j.get("id") == "core_done"), None)
        if gate and self.s.get("core_marker", True):
            import confirm_plan as CPm
            prm = dict(gate["params"])
            prm.update(out=self.p(prm["out"]), plan=self.p(prm["plan"]), queue=self.p(prm["queue"]))
            CPm.core_marker({k: v for k, v in state.items() if k in set(prm["core_jobs"])}, prm)
        # forecast probe batches (stage C): queue.log start / end lines inside the power log
        fq = self.p("results/forecast/queue_forecast_inputs.json")
        probes = []
        if os.path.exists(fq):
            q = json.load(open(fq, encoding="utf-8"))
            flog = self.p(q.get("logs") or "logs/forecast")
            os.makedirs(flog, exist_ok=True)
            with open(os.path.join(flog, "queue.log"), "a", encoding="utf-8") as fh:
                for j in q.get("jobs", []):
                    if j["id"].startswith("forecast_probe_"):
                        ts, te = t, t + timedelta(seconds=e["probe_s"])
                        fh.write(f"[queue] {ts:%Y-%m-%d %H:%M:%S} start {j['id']} (cmd, est 0.1 h)\n")
                        fh.write(f"[queue] {te:%Y-%m-%d %H:%M:%S} end {j['id']}: done\n")
                        probes.append((j["id"], ts, te))
                        t = te + timedelta(seconds=e["pause_s"])
        # power log: 1 Hz samples at x.5 s; run windows at power_W (from 2 s before to 2 s after), probes at probe_W
        pdir = self.p("logs/power")
        os.makedirs(pdir, exist_ok=True)
        tend = t + timedelta(seconds=60)
        n = int((tend - t0).total_seconds())
        tt = np.array([t0 + timedelta(seconds=k + 0.5) for k in range(n)])
        sec = np.arange(n) + 0.5
        W = np.full(n, e["idle_W"])
        MiB = np.full(n, 600.0)
        for rid, ts, te in windows:
            a, b = (ts - t0).total_seconds() - 2, (te - t0).total_seconds() + 2
            m = (sec >= a) & (sec <= b)
            W[m], MiB[m] = e["power_W"], e["MiB"]
        for jid, ts, te in probes:
            a, b = (ts - t0).total_seconds() - 2, (te - t0).total_seconds() + 2
            m = (sec >= a) & (sec <= b)
            W[m] = e["probe_W"]
        keep = np.ones(n, dtype=bool)
        if gap_run:
            ts = next(w[1] for w in windows if w[0] == gap_run)
            a = (ts - t0).total_seconds() + 100
            keep &= ~((sec > a) & (sec < a + e["gap_s"]))
        parts = [np.flatnonzero(keep)]
        if gap_run and e.get("power_split"):        # P-16 (4): the logger died in the gap and restarted in a new file
            parts = [np.flatnonzero(keep & (sec <= a)), np.flatnonzero(keep & (sec >= a + e["gap_s"]))]
        files = []
        for ix in parts:
            if not ix.size:
                continue
            x0 = tt[ix[0]]
            fn = os.path.join(pdir, f"gpu_power_{x0:%Y%m%d_%H%M%S}.csv")
            files.append(os.path.basename(fn))
            with open(fn, "w", encoding="utf-8") as fh:
                for k in ix:
                    x = tt[k]
                    fh.write(f"{x:%Y/%m/%d %H:%M:%S}.{int(x.microsecond / 1000):03d}, {W[k]:.2f} W, 60 %, {int(MiB[k])} MiB, 56, 2700 MHz\n")
        open(os.path.join(pdir, "README_power_log.txt"), "w", encoding="utf-8").write(
            "MOCK power log (mock_confirm.py). Columns: timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB], "
            "temperature.gpu, clocks.sm [MHz]; 1 Hz; no header; local time.\n")
        self.truth["edge"] = {"run_energy_Wh": e["power_W"] * e["run_s"] / 3600.0, "run_wall_s": e["run_s"], "gap_run": gap_run,
                              "power_files": files,
                              "probe_wall_s": e["probe_s"], "probe_energy_Wh": e["probe_W"] * e["probe_s"] / 3600.0,
                              "runs": len(windows)}

    def _run_log(self, path: str, rid: str, ts: datetime, te: datetime, job: dict) -> None:
        steps, tok = int(job.get("steps", 96)), int(job.get("tokens_per_step", 262144))
        name = f"nestor.mock_{rid}_train"
        L = [f"# MOCK run log of {rid}", f"[pretrain] mock_{rid}: d=480 L=12 ctx=4096 | {steps} steps x {tok:,} tok/step = 0.03B tokens"]
        L.append(f"{ts:%Y-%m-%d %H:%M:%S} - {name} - INFO - Run directory: MOCK")
        span = (te - ts).total_seconds() - 20
        for s in range(5, steps + 1, 5):
            x = ts + timedelta(seconds=10 + span * s / steps)
            L.append(f"{x:%Y-%m-%d %H:%M:%S} - {name} - INFO - Step {s} | train/loss: 3.0 | train/sec_per_step: {span / steps:.4f} | "
                     f"train/mem_peak_alloc_gib: 7.6382 | train/mem_peak_reserved_gib: 8.1387")
        L.append(f"{te:%Y-%m-%d %H:%M:%S} - {name} - INFO - Training complete!")
        open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")

    def simulate(self) -> dict:
        self.crossovers()
        self.m12b()
        self.c2()
        self.others()
        self.g1_reverts()
        self.freezes()
        self.c5_forecasts()
        self.audit()
        self.state_and_logs()
        self.add_bytes()
        out = self.p("results/analysis_mock/truth.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(self.truth, open(out, "w", encoding="utf-8"), indent=1, default=float)
        return {"runs": len(self.plan["runs"]), "truth": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SIMULATED confirm-queue outputs with known generating values")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--root", required=True)
    s.add_argument("--scenario", default=None)
    a = ap.parse_args(argv)
    scen = copy.deepcopy(DEFAULT)
    if a.scenario:
        upd = json.load(open(a.scenario, encoding="utf-8"))
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(scen.get(k), dict):
                scen[k].update(v)
            else:
                scen[k] = v
    r = Sim(a.root, scen).simulate()
    print(f"[mock_confirm] SIMULATED confirm outputs for {r['runs']} runs; truth -> {r['truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
