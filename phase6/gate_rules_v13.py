"""Decision rules of the G1/G2 specification g1g2_spec.json v1.3.0 (preregistration v1.3 sec. 6) as pure functions.

Every decision, labelling and procedure gate of the specification is one function of EXACTLY the specification's
variables. Each returns
    {"gate": <id>, "rule": <rule id of the spec>, "outcome": <spec outcome>, "action": <spec action text>,
     "sets": {<what the rule sets>}, "values": {<the spec's computed expressions>}}
The reported gates (G1-5, G1-6, G1-NS5, G2-5) return the same shape with rule None. g1_gate.py and g2_gate.py compute the
variables from run outputs and call these functions; test_gates.py runs the specification's 42 test vectors through them
(run_vector / check_vector) and checks them against the specification's own rule expressions on random inputs.

Inputs outside a gate's domain (wrong keys or lengths, SE <= 0, lr grid not c/3, c, 3c) raise ValueError: the calling
gate script records the gate as not evaluable and HOLDs instead of deciding on malformed input.

Standard library + numpy (lookup interpolation only); torch-free.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os

import numpy as np

SPEC_VERSION = "1.3.0"          # the gate-table version these rules implement; the spec FILE is read and hashed at run time

# constants of g1g2_spec.json 1.3.0 ("constants"); test_gates.py asserts equality with the specification file
F_FULL_MIN = 0.01
F_SUB_MIN = 0.005
Z_MIN = 10.0
R_PASS = 1.0
R_LONG = 0.5
R_WIDE = 0.25
STEPS_BASE = 96
STEPS_LONG = 192
Q_BASE = 0.25
Q_WIDE = 0.5
N_AA_PAIRS = 3
UCL80_FACTOR = 1.7275872940849268          # sqrt(3 / chi2.ppf(0.20, 3)); the exact constant, not 1.7276
BETA_PLAN = 0.75
POWER_TARGET = 0.8
D_SMALL = 12
D_LARGE = 20
ND_SMALL = 3
ND_LARGE = 5
ALPHA_FAMILY = 0.05
SGD_GRID_FACTORS = (0.3333333333333333, 1.0, 3.0)
GAIN_FRAC_MIN = 0.25
CLIP_FRACTION_REQUIRED = 0.0
S_STAR_CANDIDATES = (8, 24, 48)
C2_CONFIRMATORY_S_STAR = 8
C2_DRAWS_CONFIRMATORY = 8
C2_DRAWS_EXPLORATORY = 4
ROUTING_HEALTH_MAX_SHARE_OUTSIDE = 0.5
LOOKUP_COLUMN = {0.025: "power_sf_025", 0.05: "power_sf_05"}

G12_KEYS = ("R1|CODE", "R1|math", "R2|CODE", "R2|math")
G13_KEYS = ("CODE", "math")
G24_KEYS = tuple(f"{a}|{s}|{c}" for a in (2, 3) for s in ("R1", "R2") for c in ("CODE", "math"))

ACTIONS = {
    ("G1-0", "pass"): "G1 adaptation runs may start",
    ("G1-0", "stop"): "No G1 run starts. Fix the cause; a code change is logged as a pre-freeze amendment and re-hashed; "
                      "re-run the audit (E3)",
    ("G1-4", "pass"): "none",
    ("G1-4", "audit_broken"): "STOP and fix the audit; re-run the negative control; re-audit every subset run checked with "
                              "the faulty audit (E3)",
    ("G1-4", "subset_failed"): "Fix the code; re-run every affected run with the same seeds (E3, not an exclusion); G1 "
                               "decisions wait",
    ("G1-1", "pass"): "steps unchanged",
    ("G1-1", "lengthen"): "All adaptation runs lengthen to 192 steps (warm-up and cosine stretched). The 4 MoE FULL runs are "
                          "repeated at 192 steps (every S1 A/A run, pairs a = 1 and 2, and the NESTOR A/A subset too: "
                          "decision O-2) and G1-1 is re-evaluated with attempt = 2",
    ("G1-1", "stop"): "The study stops before any confirmatory run; the result is reported",
    ("G1-2", "pass"): "q = 1/4; steps unchanged",
    ("G1-2", "lengthen"): "q = 1/4 and 192 steps",
    ("G1-2", "widen"): "q = 1/2 (weighted generator) and 192 steps; G0-7R recomputes the draws, G0-1's median P and G0-7 at "
                       "q = 1/2",
    ("G1-2", "pivot"): "C1 infeasible -> PIVOT as for G0-1: C1 is not confirmatory (C1-N was dropped at G0-1); NESTOR-books "
                       "may still run as an exploratory block if its feasibility check and G1-3 pass; C2 keeps its own gates "
                       "(G0-3, G2); C3, C4, D4, the secondary blocks and the theory proceed (decision O-4)",
    ("G1-3", "pass"): "NESTOR-books may run as an exploratory block if its G0-1-style feasibility check passes (label from "
                      "G0-9L)",
    ("G1-3", "fail"): "NESTOR-books is not run and its budget released",
    ("G0-9L", "not_evaluable"): "Recorded as not evaluable; labels unchanged; the operator decides before the freeze",
    ("G0-9L", "label"): "C3-NESTOR is reported as exploratory and NESTOR-books carries the routing-health label; "
                        "confirmatory claims are unaffected",
    ("G0-9L", "no_label"): "labels unchanged (C3-NESTOR estimation; NESTOR-books exploratory)",
    ("G0-7R", "recompute"): "Re-draw the registered C1 books draws d = 1-10 per MoE base (crossover.py registered-draw, seeds "
                            "5000 + d) in the new configuration; run g0_margin.py plan and run; recompute G0-7 (D_b, SE, "
                            "beta0_b, margin-limited) with g0_gate.py and, if q changed, G0-1's median P of MoE base 1 / books "
                            "(P-bar of G2-1). Old and new values and the reason are recorded",
    ("G0-7R", "not_needed"): "G0-7 values of the G0 record stand",
    ("G2-2", "pass"): "selected SGD learning rate = the SMALLEST feasible value (small-step regime; not the matched-gain "
                      "choice: R-2)",
    ("G2-2", "fail"): "C2 is exploratory (not in the Holm family); what runs follows decision O-5",
    ("G2-3", "not_evaluated"): "C2 status from G2-2 (exploratory; O-5)",
    ("G2-3", "confirmatory"): "C2 is confirmatory with 8 draws (4 per MoE base), evaluated at s* = 8",
    ("G2-3", "exploratory_gate"): "C2 exploratory because G0-3 or G0-2 failed; 4-draw exploratory block at s* = 8 (O-5)",
    ("G2-3", "exploratory_late"): "C2 runs as the 4-draw exploratory block (2 draws per base) analysed at s*; the Holm family "
                                  "is the confirmatory set fixed at the freeze",
    ("G2-3", "none"): "no step resolves DCLM forgetting; 4-draw exploratory block with s* undefined (O-6)",
    ("G2-4", "not_applicable"): "C1 is not confirmatory (PIVOT at G1-2)",
    ("G2-4", "pass"): "books cells kept",
    ("G2-4", "fail"): "books cannot be replaced (German infeasible at G0-1): C1 is not confirmatory (PIVOT); C2, NESTOR-books "
                      "and the other blocks proceed by their own rules",
    ("G2-1", "not_applicable"): "C1 is not run as a confirmatory test",
    ("G2-1", "undefined"): "sigma_I undefined (O-7): n_d = 5, flagged",
    ("G2-1", "small"): "3 draws per cell (D = 12)",
    ("G2-1", "large"): "5 draws per cell (D = 20)",
    ("G2-1", "flagged"): "5 draws per cell; C1 flagged 'underpowered at beta = 0.75' before the freeze",
}


def _res(gate: str, rule: str | None, outcome: str | None, sets: dict | None = None, values: dict | None = None) -> dict:
    return {"gate": gate, "rule": rule, "outcome": outcome, "action": ACTIONS.get((gate, rule)) if rule else None,
            "sets": dict(sets or {}), "values": dict(values or {})}


def _need_keys(name: str, d: dict, keys: tuple) -> None:
    if set(d) != set(keys):
        raise ValueError(f"{name}: keys {sorted(d)} differ from the domain {list(keys)}")


def _need_positive(name: str, d: dict) -> None:
    bad = [k for k, v in d.items() if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0)]
    if bad:
        raise ValueError(f"{name}: SE must be finite and > 0 (domain SE_positive); offending keys {bad}")


def _finite(name: str, d: dict) -> None:
    bad = [k for k, v in d.items() if not (isinstance(v, (int, float)) and math.isfinite(v))]
    if bad:
        raise ValueError(f"{name}: non-finite values at {bad}")


# ============================================================================= lookup (power_sim_v1.lookup_power)
class Lookup:
    """signflip_lookup_v1.csv with power_sim_v1.lookup_power's interpolation: rows of this D and dist, sorted by
    (delta, power); running maximum of power over delta; numpy.interp (linear, clamped to the end values)."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        with open(path, "rb") as fh:
            self.sha256 = hashlib.sha256(fh.read()).hexdigest()
        with open(path, encoding="utf-8", newline="") as fh:
            self.rows = [dict(r) for r in csv.DictReader(fh)]
        self._cache: dict = {}

    def column(self, alpha: float) -> str:
        for a, col in LOOKUP_COLUMN.items():
            if abs(float(alpha) - a) < 1e-12:
                return col
        raise ValueError(f"no sign-flip power column for alpha = {alpha} (registered: 0.025, 0.05)")

    def curve(self, D: int, alpha: float, dist: str = "normal"):
        key = (int(D), self.column(alpha), dist)
        if key not in self._cache:
            pts = sorted((float(r["delta"]), float(r[key[1]])) for r in self.rows if int(r["D"]) == int(D) and r["dist"] == dist)
            if not pts:
                raise ValueError(f"lookup table has no rows for D = {D}, dist = {dist}")
            xs = np.array([a for a, _ in pts], dtype=np.float64)
            ys = np.maximum.accumulate(np.array([b for _, b in pts], dtype=np.float64))
            self._cache[key] = (xs, ys)
        return self._cache[key]

    def power(self, D: int, delta: float, alpha: float, dist: str = "normal") -> float:
        xs, ys = self.curve(D, alpha, dist)
        return float(np.interp(float(delta), xs, ys))


# ============================================================================= G1
def g1_0(tests_passed: int, tests_failed: int, tests_errors: int) -> dict:
    v = {"tests_passed": int(tests_passed), "tests_failed": int(tests_failed), "tests_errors": int(tests_errors)}
    if tests_failed == 0 and tests_errors == 0 and tests_passed >= 1:
        return _res("G1-0", "pass", "pass", {}, v)
    return _res("G1-0", "stop", "STOP", {"g1_blocked": True}, v)


def g1_4(negctrl_audit: str, subset_audits: dict) -> dict:
    v = {"negctrl_audit": negctrl_audit, "n_subset_runs": len(subset_audits),
         "subset_failures": sorted(k for k, x in subset_audits.items() if x != "ok")}
    if negctrl_audit != "fail":
        return _res("G1-4", "audit_broken", "STOP", {"g1_blocked": True}, v)
    if any(x != "ok" for x in subset_audits.values()):
        return _res("G1-4", "subset_failed", "E3", {"g1_blocked": True}, v)
    return _res("G1-4", "pass", "pass", {}, v)


def g1_1(F_full: list, attempt: int) -> dict:
    if len(F_full) != 4:
        raise ValueError(f"G1-1: F_full must hold the 4 MoE base-1 FULL values, got {len(F_full)}")
    if attempt not in (1, 2):
        raise ValueError(f"G1-1: attempt must be 1 or 2, got {attempt}")
    F_full = [float(x) for x in F_full]
    if not all(math.isfinite(x) for x in F_full):
        raise ValueError("G1-1: non-finite F")
    Fbar = sum(F_full) / len(F_full)
    v = {"Fbar": Fbar, "F_full": F_full, "attempt": attempt}
    if Fbar >= F_FULL_MIN:
        return _res("G1-1", "pass", "pass", {}, v)
    if attempt == 1:
        return _res("G1-1", "lengthen", "lengthen", {"steps": STEPS_LONG}, v)
    return _res("G1-1", "stop", "STOP", {"stop": True}, v)


def g1_2(F: dict, SE: dict) -> dict:
    _need_keys("G1-2 F", F, G12_KEYS)
    _need_keys("G1-2 SE", SE, G12_KEYS)
    _finite("G1-2 F", F)
    _need_positive("G1-2 SE", SE)
    ratios = {k: min(F[k] / F_SUB_MIN, F[k] / (Z_MIN * SE[k])) for k in F}
    r = min(ratios.values())
    v = {"r": r, "ratios": ratios}
    if r >= R_PASS:
        return _res("G1-2", "pass", "pass", {"q": Q_BASE}, v)
    if R_LONG <= r:
        return _res("G1-2", "lengthen", "lengthen", {"q": Q_BASE, "steps": STEPS_LONG}, v)
    if R_WIDE <= r:
        return _res("G1-2", "widen", "widen", {"q": Q_WIDE, "steps": STEPS_LONG, "recompute_g07": True}, v)
    return _res("G1-2", "pivot", "PIVOT", {"C1_confirmatory": False, "pivot": True}, v)


def g1_3(FN: dict, SEN: dict) -> dict:
    _need_keys("G1-3 FN", FN, G13_KEYS)
    _need_keys("G1-3 SEN", SEN, G13_KEYS)
    _finite("G1-3 FN", FN)
    _need_positive("G1-3 SEN", SEN)
    per = {c: bool(FN[c] >= F_SUB_MIN and FN[c] >= Z_MIN * SEN[c]) for c in FN}
    v = {"per_corpus": per}
    if all(per.values()):
        return _res("G1-3", "pass", "pass", {"NESTORB_signal": True}, v)
    return _res("G1-3", "fail", "fail", {"NESTORB_signal": False}, v)


def g0_9l(routers_assessed: int, routers_outside: int, ranges_available: bool) -> dict:
    if not ranges_available:
        return _res("G0-9L", "not_evaluable", "not_evaluable", {}, {})
    if routers_assessed < 1:
        raise ValueError("G0-9L: routers_assessed must be >= 1 when the App. C.6 ranges are available")
    share = routers_outside / routers_assessed
    v = {"share_outside": share, "routers_assessed": routers_assessed, "routers_outside": routers_outside}
    if share > ROUTING_HEALTH_MAX_SHARE_OUTSIDE:
        return _res("G0-9L", "label", "label", {"NESTORB_label": "exploratory, routing-health label", "C3N_label": "exploratory"}, v)
    return _res("G0-9L", "no_label", "no_label", {}, v)


def g0_7r(q_changed: bool, targets_changed: bool, f_or_stream_changed: bool) -> dict:
    v = {"q_changed": bool(q_changed), "targets_changed": bool(targets_changed), "f_or_stream_changed": bool(f_or_stream_changed)}
    if q_changed or targets_changed or f_or_stream_changed:
        return _res("G0-7R", "recompute", "recompute", {"g07_recomputed": True}, v)
    return _res("G0-7R", "not_needed", "not_needed", {}, v)


def g1_5(F_h100: float | None, F_101: float, F_102: float) -> dict:
    """Reported: |F_h100 - F_101| / |F_101 - F_102| (numerator alone if the denominator is 0)."""
    if F_h100 is None:
        return _res("G1-5", None, "reported", {}, {"F_h100": None, "F_101": F_101, "F_102": F_102, "ratio": "not available"})
    num, den = abs(F_h100 - F_101), abs(F_101 - F_102)
    return _res("G1-5", None, "reported", {}, {"F_h100": F_h100, "F_101": F_101, "F_102": F_102, "numerator": num,
                                               "denominator": den, "ratio": num / den if den != 0 else num,
                                               "ratio_is_numerator_alone": den == 0})


def g1_6(F_full: dict, F_cont: float) -> dict:
    """Reported: F_rel = F - F_cont per MoE base-1 FULL run (DCLM)."""
    return _res("G1-6", None, "reported", {}, {"F_cont": F_cont, "F_rel": {k: v - F_cont for k, v in F_full.items()}})


# ============================================================================= G2
def stable_flags(runs: list) -> list:
    return [bool((not x["e1_divergent"]) and x["loss_step24"] < x["loss_step1"]) for x in runs]


def _check_grid(c: float, runs: list) -> None:
    if len(runs) != 3:
        raise ValueError(f"G2-2: runs must hold 3 learning rates (c/3, c, 3c), got {len(runs)}")
    for x, fac in zip(runs, SGD_GRID_FACTORS):
        if not math.isclose(float(x["lr"]), float(c) * fac, rel_tol=1e-6):
            raise ValueError(f"G2-2: lr {x['lr']} is not {fac:g} x c (c = {c}); order must be c/3, c, 3c")


def g2_2(c: float, runs: list, gain24_muon: float) -> dict:
    if not (isinstance(c, (int, float)) and c > 0):
        raise ValueError("G2-2: c must be > 0")
    _check_grid(c, runs)
    stable = stable_flags(runs)
    feasible = [bool(st and gain24_muon > 0 and x["gain24"] >= GAIN_FRAC_MIN * gain24_muon
                     and x["clip_fraction"] == CLIP_FRACTION_REQUIRED) for x, st in zip(runs, stable)]
    v = {"c": float(c), "grid": [float(x["lr"]) for x in runs], "stable": stable, "feasible": feasible,
         "gain24_muon": gain24_muon,
         "gain_ratio": [(x["gain24"] / gain24_muon) if gain24_muon else None for x in runs]}
    if any(feasible):
        lr = min(x["lr"] for x, ok in zip(runs, feasible) if ok)
        v["sgd_lr"] = lr
        return _res("G2-2", "pass", "pass", {"G22_pass": True, "sgd_lr": lr}, v)
    return _res("G2-2", "fail", "fail", {"G22_pass": False, "C2_confirmatory": False}, v)


def o5_exploratory_lr(runs: list) -> float | None:
    """Decision O-5: after a G2-2 failure the smallest STABLE learning rate is used for the 4-draw exploratory C2 block;
    None (C2 not run, budget released) if no learning rate is stable."""
    st = stable_flags(runs)
    lrs = [x["lr"] for x, ok in zip(runs, st) if ok]
    return min(lrs) if lrs else None


def g2_3(Fs: dict, SEs: dict, G22_pass: bool, g03_pass: bool, c2_matching_feasible: bool) -> dict:
    if not G22_pass:
        return _res("G2-3", "not_evaluated", "not_evaluated", {}, {})
    Fs = {int(k): float(v) for k, v in Fs.items()}
    SEs = {int(k): float(v) for k, v in SEs.items()}
    _need_keys("G2-3 Fs", Fs, S_STAR_CANDIDATES)
    _need_keys("G2-3 SEs", SEs, S_STAR_CANDIDATES)
    _finite("G2-3 Fs", Fs)
    _need_positive("G2-3 SEs", SEs)
    passes = {s: bool(Fs[s] >= F_SUB_MIN and Fs[s] >= Z_MIN * SEs[s]) for s in S_STAR_CANDIDATES}
    s_star = min([s for s in S_STAR_CANDIDATES if passes[s]], default=None)
    v = {"s_star": s_star, "pass_per_step": passes}
    if s_star == C2_CONFIRMATORY_S_STAR and g03_pass and c2_matching_feasible:
        return _res("G2-3", "confirmatory", "confirmatory", {"s_star": 8, "C2_confirmatory": True, "C2_draws": C2_DRAWS_CONFIRMATORY}, v)
    if s_star == C2_CONFIRMATORY_S_STAR:
        return _res("G2-3", "exploratory_gate", "exploratory", {"s_star": 8, "C2_confirmatory": False, "C2_draws": C2_DRAWS_EXPLORATORY}, v)
    if s_star in (24, 48):
        return _res("G2-3", "exploratory_late", "exploratory", {"s_star": s_star, "C2_confirmatory": False, "C2_draws": C2_DRAWS_EXPLORATORY}, v)
    return _res("G2-3", "none", "exploratory", {"s_star": None, "C2_confirmatory": False, "C2_draws": C2_DRAWS_EXPLORATORY}, v)


def g2_4(FB: dict, SEB: dict, pivot: bool) -> dict:
    if pivot:
        return _res("G2-4", "not_applicable", "not_applicable", {}, {})
    _need_keys("G2-4 FB", FB, G24_KEYS)
    _need_keys("G2-4 SEB", SEB, G24_KEYS)
    _finite("G2-4 FB", FB)
    _need_positive("G2-4 SEB", SEB)
    per = {k: bool(FB[k] >= F_SUB_MIN and FB[k] >= Z_MIN * SEB[k]) for k in FB}
    v = {"per_key": per}
    if all(per.values()):
        return _res("G2-4", "pass", "pass", {}, v)
    return _res("G2-4", "fail", "C1_exploratory", {"C1_confirmatory": False, "pivot": True}, v)


def i_aa(F: dict) -> float | None:
    """I_AA of one A/A pair from F keyed 'R1|CODE', 'R1|math', 'R2|CODE', 'R2|math'; None if any F <= 0 (G2-1)."""
    _need_keys("I_AA F", F, G12_KEYS)
    if any(not (F[k] > 0) for k in G12_KEYS):
        return None
    return (math.log(F["R2|CODE"]) - math.log(F["R1|CODE"])) - (math.log(F["R2|math"]) - math.log(F["R1|math"]))


def g2_1(I_AA: list, beta0: list, Pbar: float, C1_confirmatory: bool, C2_confirmatory: bool, lookup: Lookup) -> dict:
    if len(I_AA) != N_AA_PAIRS:
        raise ValueError(f"G2-1: I_AA must hold {N_AA_PAIRS} pairs, got {len(I_AA)}")
    if len(beta0) != 2:
        raise ValueError("G2-1: beta0 must be [beta0_1, beta0_2]")
    m = int(bool(C1_confirmatory)) + int(bool(C2_confirmatory))
    defined = all(x is not None for x in I_AA)
    sigma_I = math.sqrt(sum(x * x for x in I_AA) / len(I_AA)) if defined else None
    delta = (((BETA_PLAN - max(beta0)) * Pbar / (UCL80_FACTOR * sigma_I)) if sigma_I > 0 else 2.5) if defined else None
    alpha = ALPHA_FAMILY / m if m >= 1 else None
    p12 = lookup.power(D_SMALL, delta, alpha) if (defined and C1_confirmatory) else None
    p20 = lookup.power(D_LARGE, delta, alpha) if (defined and C1_confirmatory) else None
    v = {"m": m, "alpha": alpha, "defined": defined, "sigma_I": sigma_I, "UCL80": (UCL80_FACTOR * sigma_I) if defined else None,
         "beta0_max": max(beta0), "Pbar": Pbar, "delta": delta, "p12": p12, "p20": p20}
    if not C1_confirmatory:
        return _res("G2-1", "not_applicable", "not_applicable", {}, v)
    if not defined:
        return _res("G2-1", "undefined", "n_d=5 (flag)", {"n_d": ND_LARGE, "underpowered_flag": "sigma_I undefined"}, v)
    if p12 >= POWER_TARGET:
        return _res("G2-1", "small", "n_d=3", {"n_d": ND_SMALL, "underpowered_flag": False}, v)
    if p20 >= POWER_TARGET:
        return _res("G2-1", "large", "n_d=5", {"n_d": ND_LARGE, "underpowered_flag": False}, v)
    return _res("G2-1", "flagged", "n_d=5 (flag)", {"n_d": ND_LARGE, "underpowered_flag": "underpowered at beta = 0.75"}, v)


# ============================================================================= routing health (G0-9L variables, decision O-8)
def routing_health_counts(diagnostics: dict, ranges: dict) -> dict:
    """diagnostics {router: {diagnostic: value}}; ranges {diagnostic: [lo, hi]} (App. C.6). A router is assessed if at least
    one of its diagnostics has a range, and outside if at least one such diagnostic lies outside [lo, hi] (O-8)."""
    per, assessed, outside = {}, 0, 0
    for router, diags in sorted(diagnostics.items()):
        rows = {d: {"value": x, "range": list(ranges[d]), "outside": not (ranges[d][0] <= x <= ranges[d][1])}
                for d, x in diags.items() if d in ranges and x is not None}
        if rows:
            assessed += 1
            out = any(r["outside"] for r in rows.values())
            outside += int(out)
            per[router] = {"diagnostics": rows, "outside": out}
    return {"routers_assessed": assessed, "routers_outside": outside, "per_router": per}


# ============================================================================= test-vector interface
RULES = {"G1-0": g1_0, "G1-4": g1_4, "G1-1": g1_1, "G1-2": g1_2, "G1-3": g1_3, "G0-9L": g0_9l, "G0-7R": g0_7r,
         "G2-2": g2_2, "G2-3": g2_3, "G2-4": g2_4, "G2-1": g2_1}


def run_vector(vec: dict, lookup: Lookup) -> dict:
    kw = dict(vec["inputs"])
    if vec["gate"] == "G2-1":
        kw["lookup"] = lookup
    return RULES[vec["gate"]](**kw)


def _close(a, b, rel: float, abs_: float) -> bool:
    if isinstance(b, bool) or b is None or isinstance(b, str):
        return a == b
    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
    return a == b


def check_vector(vec: dict, res: dict, rel: float = 1e-12, abs_: float = 1e-15) -> list:
    """Mismatches between a rule result and a test vector's expected rule, values and sets (empty list = reproduced)."""
    exp = vec["expected"]
    bad = []
    if res["rule"] != exp["rule"]:
        bad.append(f"rule {res['rule']!r} != expected {exp['rule']!r}")
    for k, want in (exp.get("values") or {}).items():
        if k not in res["values"]:
            bad.append(f"value {k} missing")
        elif not _close(res["values"][k], want, rel, abs_):
            bad.append(f"value {k} = {res['values'][k]!r} != expected {want!r}")
    for k, want in (exp.get("sets") or {}).items():
        if k not in res["sets"]:
            bad.append(f"sets.{k} missing")
        elif not _close(res["sets"][k], want, rel, abs_):
            bad.append(f"sets.{k} = {res['sets'][k]!r} != expected {want!r}")
    return bad


def spec_identity(spec_path: str) -> dict:
    """Identity of the registered specification file actually read: path, version, revision, sha256, number of vectors."""
    import json

    with open(spec_path, "rb") as fh:
        raw = fh.read()
    spec = json.loads(raw.decode("utf-8"))
    return {"path": os.path.abspath(spec_path), "version": spec.get("version"), "revision": spec.get("revision"),
            "sha256": hashlib.sha256(raw).hexdigest(), "n_test_vectors": len(spec.get("test_vectors", []))}


def reproduce_all(spec_path: str, lookup_path: str) -> dict:
    """Run every test vector of the specification file. Returns {'n', 'failed', 'reproduced', 'spec': spec_identity(...),
    'lookup_sha256', 'rules_version'}; 'reproduced' is True iff the file has vectors and every one is reproduced."""
    import json

    ident = spec_identity(spec_path)
    spec = json.load(open(spec_path, encoding="utf-8"))
    lk = Lookup(lookup_path)
    failed = []
    for i, vec in enumerate(spec.get("test_vectors", [])):
        try:
            bad = check_vector(vec, run_vector(vec, lk))
        except Exception as e:                       # noqa: BLE001 - any exception is a failed vector
            bad = [f"{type(e).__name__}: {e}"]
        if bad:
            failed.append({"index": i, "gate": vec["gate"], "name": vec["name"], "mismatches": bad})
    n = ident["n_test_vectors"]
    return {"n": n, "failed": failed, "reproduced": bool(n) and not failed, "spec": ident, "lookup_sha256": lk.sha256,
            "rules_version": SPEC_VERSION}


if __name__ == "__main__":
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="reproduce the test vectors of the registered g1g2_spec.json with these rules")
    ap.add_argument("--spec", default="g1g2_spec.json")
    ap.add_argument("--lookup", default="signflip_lookup_v1.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = reproduce_all(a.spec, a.lookup)
    s = r["spec"]
    print(f"spec {s['path']}: version {s['version']} revision {s['revision']} sha256 {s['sha256']}")
    print(f"{r['n'] - len(r['failed'])}/{r['n']} test vectors reproduced by gate_rules_v13 (rules for {SPEC_VERSION})")
    if s["version"] != SPEC_VERSION:
        print(f"WARNING: the spec file is version {s['version']}, these rules implement {SPEC_VERSION}")
    for f in r["failed"]:
        print("FAILED", f)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(r, fh, indent=1)
    sys.exit(0 if r["reproduced"] else 1)
