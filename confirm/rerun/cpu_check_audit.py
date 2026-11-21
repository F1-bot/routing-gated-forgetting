r"""cpu_check_audit.py -- CPU-only check of amendment A-3 before the re-run (read-only on the project; writes only
confirm/rerun/cpu_check_audit.json). torch is blocked (an import raises), no bytecode is written.

  <python> confirm/rerun/cpu_check_audit.py            (working directory: the project root)

For audit_moe_b1 and audit_moe_b2 of the frozen queue it restores dose_draws.base_inputs through audit_shim.restore()
(the shim's own checks), then runs, in the order of assumption_audit.run_realized (lines 352-374), its torch-free part
on the registered argv: base_cell -> crossover.reg_load_dir -> queue state (final checkpoints of the draw-1 C1 runs) ->
subset specs from the frozen queue -> realized_groups -> group_contrasts -> _registered_eps -> routing meta (experiment,
checkpoint); plus the 3 x 24 estimation windows (g0_margin.estimation_windows, as in line 411; the token reader
repo/nestor/data/token_documents.py is loaded by file, see _token_documents) and, for comparison
only, groups_for (the first-step audit's groups, not used by --directions realized). Compared with the frozen plan
(confirm_config.audit, the audit estimate) and prereg v1.3 sec. 4.3 (24 groups per base: A_1, B_1 x layers 1-12;
corpora DCLM, code, math; 24 windows). Exit 0 if every comparison holds, 1 otherwise.
"""
import sys

sys.dont_write_bytecode = True


class _NoTorch:                                                # torch / triton must not be imported by this check
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "triton"):
            raise ImportError(f"cpu_check_audit: import of {name} blocked (CPU-only, torch-free check)")
        return None


sys.meta_path.insert(0, _NoTorch())

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir(ROOT)
OUT = os.path.join(HERE, "cpu_check_audit.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


spec = importlib.util.spec_from_file_location("audit_shim", os.path.join(HERE, "audit_shim.py"))
SHIM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SHIM)
t0 = time.time()
facts = SHIM.restore()                                         # checks 1-3, then the restoration (step 4)

import numpy as np  # noqa: E402

import assumption_audit as AU  # noqa: E402
import crossover as CX  # noqa: E402
import dose_draws as DD  # noqa: E402
import eval_windows as EW  # noqa: E402
import g0_margin as GM  # noqa: E402

def _token_documents() -> None:
    """The run imports nestor.data.token_documents after eval_windows.nestor_env() put <root>/repo on sys.path (model
    load, lines 378-379); the package nestor.data imports torch in its __init__. This torch-free check loads the same
    file repo/nestor/data/token_documents.py (numpy only) under its module name, with empty parent packages."""
    if "nestor.data.token_documents" in sys.modules:
        return
    import types
    base = os.path.join(ROOT, "repo", "nestor")
    for name, path in (("nestor", base), ("nestor.data", os.path.join(base, "data"))):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    sp = importlib.util.spec_from_file_location("nestor.data.token_documents", os.path.join(base, "data", "token_documents.py"))
    td = importlib.util.module_from_spec(sp)
    sys.modules["nestor.data.token_documents"] = td
    sp.loader.exec_module(td)
    rep["token_documents"] = {"file": "repo/nestor/data/token_documents.py", "sha256": sha(os.path.join(base, "data", "token_documents.py")),
                              "loaded_by_path": True}


QUEUE, PLAN, STATE = "confirm/queue_confirm.json", "confirm/plan_confirm.json", "results/queue_state_confirm.json"
q = json.load(open(QUEUE, encoding="utf-8-sig"))
qj = {j["id"]: j for j in q["jobs"]}
plan = json.load(open(PLAN, encoding="utf-8"))
st = json.load(open(STATE, encoding="utf-8"))
ac = (plan.get("confirm_config") or {}).get("audit") or {}
probe_s = float(ac.get("probe_s_per_direction_corpus", 24.66))
rep = {"kind": "A-3 CPU check", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shim": facts,
       "inputs": {QUEUE: sha(QUEUE), PLAN: sha(PLAN), STATE + " (read-only, at the check)": sha(STATE)},
       "plan_audit": {"confirm_config.audit": ac, "blocks.audit": (plan.get("blocks") or {}).get("audit"),
                      "block_gpu_h.audit": (plan.get("block_gpu_h") or {}).get("audit"), "code_stream": plan.get("code_stream")},
       "dose_draws": {"file": os.path.relpath(DD.__file__, ROOT), "base_inputs_file": DD.base_inputs.__code__.co_filename,
                      "base_inputs_first_line": DD.base_inputs.__code__.co_firstlineno},
       "bases": {}, "checks": {}}
ok_all = True


def chk(name, cond, detail=None):
    global ok_all
    rep["checks"][name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
    ok_all &= bool(cond)


for jid in ("audit_moe_b1", "audit_moe_b2"):
    a = qj[jid]["argv"]
    arg = {a[i]: a[i + 1] for i in range(2, len(a) - 1) if a[i].startswith("--")}
    base, g0rec, windows = arg["--base"], arg["--g0-record"], int(arg["--windows"])
    runs = [r for r in arg["--runs"].split(",") if r]
    ckpt = arg["--checkpoint"].format(**q["vars"])
    r = {"job": jid, "argv_after_script": a[2:], "est_h": qj[jid].get("est_h"), "requires": qj[jid].get("requires")}
    bi = AU.base_cell(g0rec, base)                                       # line 352 (DD.base_inputs)
    r["base_inputs"] = {"dir": bi["dir"], "dir_exists": os.path.isdir(bi["dir"]), "code": bi["code"], "layout": bi["layout"],
                        "half_seed": bi["half_seed"], "family": bi["family"],
                        "cell": {k: bi["cell"].get(k) for k in ("feasible", "f", "tol", "rung", "median_P", "P_median", "n_pairs")
                                 if k in bi["cell"]}, "cell_keys": sorted(bi["cell"])}
    code = bi["code"]
    t = CX.reg_load_dir(bi["dir"], bi["family"], layout=bi["layout"], half_seed=bi["half_seed"])   # line 354
    r["tables"] = {"L": t.L, "N": t.N, "strata": [str(s) for s in t.strata][:3] + ["..."], "units": int(sum(len(k) for k in t.keys))}
    side_of = {rid: rid.rsplit("_", 1)[-1] for rid in runs}
    finals = {}
    for rid in runs:
        s_ = st.get(rid) or {}
        fck = s_.get("final")
        finals[rid] = {"status": s_.get("status"), "final": fck, "exists": bool(fck and os.path.exists(fck)),
                       "bytes": os.path.getsize(fck) if fck and os.path.exists(fck) else None}
    r["draw1_runs"] = finals
    specs = {side_of[rid]: qj[rid]["spec"] for rid in runs}             # line 366 (the frozen queue's subset specs)
    groups = AU.realized_groups(t, specs)                                # line 367
    contrast = AU.group_contrasts(t, groups, code)                       # line 368
    eps, eps_src = AU._registered_eps(bi["dir"])                         # line 369
    meta = EW.load_result(os.path.join(bi["dir"], f"routing__{code}.npz"))["meta"]     # line 371
    rec = json.load(open(g0rec, encoding="utf-8-sig"))
    r["groups"] = {"n": len(groups), "per_side": {sd: sum(g["side"] == sd for g in groups) for sd in ("A", "B")},
                   "layers_per_side": {sd: sorted(g["layer"] for g in groups if g["side"] == sd) for sd in ("A", "B")},
                   "units_per_group": {g["name"]: g["n_units"] for g in groups},
                   "contrast_sum_per_side": {sd: round(sum(contrast[g["name"]] for g in groups if g["side"] == sd), 12) for sd in ("A", "B")}}
    r["eps"] = {"value": eps, "source": eps_src}
    r["routing_meta"] = {"experiment": meta.get("experiment"), "checkpoint": meta.get("checkpoint")}
    r["checkpoint_argv"] = {"path": ckpt, "exists": os.path.exists(ckpt), "bytes": os.path.getsize(ckpt) if os.path.exists(ckpt) else None,
                            "g0_record_checkpoints": (rec.get("checkpoints") or {}).get(base)}
    corpora = ["dclm", code, "math"]                                     # line 370
    _token_documents()                                                   # what eval_windows.nestor_env + torch give the run
    win = {}
    for c in corpora:                                                    # line 411
        ws = GM.estimation_windows(bi["dir"], c, windows, half_seed=int(bi["half_seed"]))
        win[c] = {"name": ws.name, "n": len(ws.index), "first_index": [int(x) for x in ws.index[:3]],
                  "hashes_distinct": len({int(h) for h in ws.hashes}) == len(ws.hashes)}
    r["windows"] = win
    g_first = AU.groups_for(t, cell=bi["cell"], code=code, draws=2)
    r["groups_for_first_step_draws2"] = {"n": len(g_first), "kinds": {k: sum(g["kind"] == k for g in g_first) for k in ("layer", "pair", "subset", "half", "all")}}
    n_reg = 2 * t.L
    est = round(n_reg * 3 * probe_s / 3600.0, 4)
    r["registered"] = {"groups_prereg_4_3": 24, "groups_plan_2L": n_reg, "windows_plan": ac.get("windows"), "est_h_plan_formula": est}
    rep["bases"][base] = r
    chk(f"{base}: G0 directory exists", os.path.isdir(bi["dir"]), bi["dir"])
    chk(f"{base}: code stream = plan code_stream", code == plan.get("code_stream"), [code, plan.get("code_stream")])
    chk(f"{base}: books cell feasible with f = 1/2", bool(bi["cell"].get("feasible")) and abs(float(bi["cell"].get("f", -1)) - 0.5) < 1e-12,
        {k: bi["cell"].get(k) for k in ("feasible", "f", "tol")})
    chk(f"{base}: family moe", bi["family"] == "moe", bi["family"])
    chk(f"{base}: groups = 24 (prereg 4.3) = 2 L (plan)", len(groups) == 24 == n_reg, [len(groups), n_reg])
    chk(f"{base}: 12 layers per side", all(sorted(g["layer"] for g in groups if g["side"] == sd) == list(range(t.L)) for sd in ("A", "B")) and t.L == 12, t.L)
    chk(f"{base}: contrasts sum to 0 per subset", all(abs(v) < 1e-9 for v in r["groups"]["contrast_sum_per_side"].values()), r["groups"]["contrast_sum_per_side"])
    chk(f"{base}: windows 24 = plan, 3 corpora x 24", windows == ac.get("windows") == 24 and all(w["n"] == windows for w in win.values()), {c: w["n"] for c, w in win.items()})
    chk(f"{base}: draw-1 C1 runs done with final checkpoints on disk", all(v["status"] == "done" and v["exists"] for v in finals.values()), finals)
    chk(f"{base}: registered checkpoint exists", os.path.exists(ckpt), ckpt)
    chk(f"{base}: queue est_h = plan formula 2L x 3 x probe_s", abs(float(qj[jid]["est_h"]) - est) < 1e-4, [qj[jid]["est_h"], est])
    chk(f"{base}: eps is the registered G0-7 books step", eps_src.startswith("G0-7 registered step"), eps_src)
chk("no audit output yet (audit/ absent)", not os.path.exists(os.path.join(ROOT, "audit")), os.path.exists(os.path.join(ROOT, "audit")))
chk("torch not imported", "torch" not in sys.modules)
rep["seconds"] = round(time.time() - t0, 1)
rep["pass"] = ok_all
with open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
os.replace(OUT + ".tmp", OUT)
for k, v in rep["checks"].items():
    print(("PASS " if v["pass"] else "FAIL ") + k)
for base, r in rep["bases"].items():
    bi = r["base_inputs"]
    print(f"[{base}] dir {bi['dir']} (exists {bi['dir_exists']}); code {bi['code']}; cell {bi['cell']}; layout {bi['layout']}; "
          f"half_seed {bi['half_seed']}; family {bi['family']}; tables L={r['tables']['L']} N={r['tables']['N']}; "
          f"groups {r['groups']['n']} ({r['groups']['per_side']}); eps {r['eps']['value']:g}; windows "
          f"{ {c: w['n'] for c, w in r['windows'].items()} }; est {r['registered']['est_h_plan_formula']} h (queue {r['est_h']})")
print(f"[cpu-check] shim: {facts['action']}; {'PASS' if ok_all else 'FAIL'} in {rep['seconds']} s -> {OUT}")
sys.exit(0 if ok_all else 1)
