"""Check g1g2_spec.json: structure, constants, expression syntax, test vectors, and that every decision table is
mutually exclusive and exhaustive (exactly one rule fires) on seeded random inputs.

    python g1g2_spec_check.py g1g2_spec.json [--fuzz 2000] [--seed 20261103]

Exit code 0 if every check passes. Needs power_sim_v1.py (registered; lookup_power) and signflip_lookup_v1.csv next to it.
The same test vectors are the reconciliation target for g1_gate.py and g2_gate.py.
"""
import argparse
import ast
import json
import math
import random
import sys

import pandas as pd

import power_sim_v1 as v1

FUNCS = dict(min=min, max=max, sum=sum, len=len, all=all, any=any, abs=abs, int=int, float=float, zip=zip, sqrt=math.sqrt, log=math.log)


def make_env(spec, rows):
    env = dict(FUNCS)
    env.update({k: v for k, v in spec["constants"].items() if not isinstance(v, dict)})
    env["lookup_power"] = lambda D, delta, alpha: v1.lookup_power(rows, int(D), float(delta), float(alpha), dist=spec["constants"]["LOOKUP"]["dist"])
    return env


def evaluate(gate, inputs, base_env):
    # one globals dict (no separate locals): comprehensions and generator expressions inside an expression see only globals
    env = dict(base_env)
    env.update(inputs)
    env["__builtins__"] = {}
    for name, expr in gate.get("expressions", {}).items():
        try:
            env[name] = eval(expr, env)
        except Exception:
            env[name] = None
    fired = []
    for r in gate["rules"]:
        try:
            if eval(r["when"], env):
                fired.append(r["id"])
        except Exception as e:
            fired.append(f"ERROR:{r['id']}:{type(e).__name__}")
    return fired, env


def fuzz_inputs(gid, rnd):
    U = rnd.uniform
    if gid == "G1-0":
        return {"tests_passed": rnd.randint(0, 20), "tests_failed": rnd.randint(0, 2), "tests_errors": rnd.randint(0, 2)}
    if gid == "G1-4":
        return {"negctrl_audit": rnd.choice(["ok", "fail"]), "subset_audits": {f"r{i}": rnd.choice(["ok", "ok", "fail"]) for i in range(rnd.randint(1, 5))}}
    if gid == "G1-1":
        return {"F_full": [U(-0.005, 0.03) for _ in range(4)], "attempt": rnd.choice([1, 2])}
    if gid == "G1-2":
        ks = ["R1|CODE", "R1|math", "R2|CODE", "R2|math"]
        return {"F": {k: U(-0.005, 0.05) for k in ks}, "SE": {k: U(1e-5, 0.005) for k in ks}}
    if gid == "G1-3":
        return {"FN": {k: U(-0.005, 0.03) for k in ("CODE", "math")}, "SEN": {k: U(1e-5, 0.003) for k in ("CODE", "math")}}
    if gid == "G0-9L":
        n = rnd.randint(1, 40)
        return {"routers_assessed": n, "routers_outside": rnd.randint(0, n), "ranges_available": rnd.random() < 0.8}
    if gid == "G0-7R":
        return {"q_changed": rnd.random() < 0.3, "targets_changed": rnd.random() < 0.3, "f_or_stream_changed": rnd.random() < 0.2}
    if gid == "G2-2":
        c = U(0.01, 1.0)
        runs = [{"lr": c * f, "e1_divergent": rnd.random() < 0.2, "loss_step1": 2.0, "loss_step24": U(1.5, 2.3), "gain24": U(-0.01, 0.1),
                 "clip_fraction": rnd.choice([0.0, 0.0, 0.0, 0.05])} for f in (1 / 3, 1.0, 3.0)]
        return {"c": c, "runs": runs, "gain24_muon": U(-0.01, 0.2)}
    if gid == "G2-3":
        return {"Fs": {s: U(-0.002, 0.03) for s in (8, 24, 48)}, "SEs": {s: U(1e-4, 0.003) for s in (8, 24, 48)},
                "G22_pass": rnd.random() < 0.8, "g03_pass": rnd.random() < 0.8, "c2_matching_feasible": rnd.random() < 0.9}
    if gid == "G2-4":
        ks = [f"{a}|{s}|{c}" for a in (2, 3) for s in ("R1", "R2") for c in ("CODE", "math")]
        return {"FB": {k: U(-0.002, 0.03) for k in ks}, "SEB": {k: U(1e-4, 0.002) for k in ks}, "pivot": rnd.random() < 0.2}
    if gid == "G2-1":
        I = [None if rnd.random() < 0.05 else U(-1.0, 1.0) for _ in range(3)]
        return {"I_AA": I, "beta0": [U(0.25, 0.5), U(0.25, 0.5)], "Pbar": U(0.05, 1.5), "C1_confirmatory": rnd.random() < 0.85,
                "C2_confirmatory": rnd.random() < 0.5}
    raise KeyError(gid)


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--fuzz", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    rows = pd.read_csv(spec["constants"]["LOOKUP"]["file"]).to_dict("records")
    env0 = make_env(spec, rows)
    problems = []
    # structure
    ids = [g["id"] for g in spec["gates"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate gate ids")
    for g in spec["gates"]:
        for r in g["requires"]:
            if r not in ids:
                problems.append(f"{g['id']}: unknown requirement {r}")
        for i in g["inputs"]:
            if i not in spec["inputs"]:
                problems.append(f"{g['id']}: unknown input {i}")
        if g["kind"] == "reported" and g["rules"]:
            problems.append(f"{g['id']}: reported gate with rules")
        if g["kind"] != "reported" and len(g["rules"]) < 2:
            problems.append(f"{g['id']}: decision gate with < 2 rules")
        for e in list(g.get("expressions", {}).values()) + [r["when"] for r in g["rules"]]:
            try:
                ast.parse(e, mode="eval")
            except SyntaxError as ex:
                problems.append(f"{g['id']}: syntax error in {e!r}: {ex}")
    # constants
    try:
        from scipy import stats
        ucl = math.sqrt(3.0 / stats.chi2.ppf(0.20, 3))
        if abs(ucl - spec["constants"]["UCL80_FACTOR"]) > 1e-12:
            problems.append(f"UCL80_FACTOR {spec['constants']['UCL80_FACTOR']} != {ucl}")
    except ImportError:
        pass
    # test vectors
    by_id = {g["id"]: g for g in spec["gates"]}
    n_ok = 0
    for t in spec["test_vectors"]:
        g = by_id[t["gate"]]
        inputs = t["inputs"]
        if t["gate"] == "G2-3":
            inputs = dict(inputs, Fs={int(k): v for k, v in inputs["Fs"].items()}, SEs={int(k): v for k, v in inputs["SEs"].items()})
        fired, env = evaluate(g, inputs, env0)
        exp = t["expected"]
        if fired != [exp["rule"]]:
            problems.append(f"vector {t['gate']} '{t['name']}': fired {fired}, expected {exp['rule']}")
            continue
        bad = [k for k, v in exp.get("values", {}).items() if k in env and not close(env[k], v)]
        rule = next(r for r in g["rules"] if r["id"] == exp["rule"])
        bad += [k for k, v in exp.get("sets", {}).items() if k in rule["sets"] and not str(rule["sets"][k]).startswith("<") and rule["sets"][k] != v]
        if bad:
            problems.append(f"vector {t['gate']} '{t['name']}': mismatched values {[(k, env.get(k), exp['values'].get(k, exp.get('sets', {}).get(k))) for k in bad]}")
        else:
            n_ok += 1
    # exclusivity and exhaustiveness
    rnd = random.Random(a.seed)
    fuzz_fail = {}
    for g in spec["gates"]:
        if not g["rules"]:
            continue
        for _ in range(a.fuzz):
            fired, _ = evaluate(g, fuzz_inputs(g["id"], rnd), env0)
            if len(fired) != 1 or fired[0].startswith("ERROR"):
                fuzz_fail.setdefault(g["id"], fired)
    for k, v in fuzz_fail.items():
        problems.append(f"{k}: not exactly one rule fired on a random input ({v})")
    n_dec = sum(1 for g in spec["gates"] if g["rules"])
    print(f"gates {len(ids)} ({n_dec} with decision tables); test vectors {n_ok}/{len(spec['test_vectors'])} reproduced; "
          f"fuzz {a.fuzz} inputs per decision table, seed {a.seed}: {n_dec - len(fuzz_fail)}/{n_dec} tables exclusive and exhaustive")
    for p in problems:
        print("PROBLEM:", p)
    print("RESULT:", "PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
