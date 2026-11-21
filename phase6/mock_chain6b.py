"""MOCK end-to-end harness of chain6b.ps1 (phase 6): every GPU step replaced by a SIMULATED stand-in, every non-GPU step
real. Nothing produced here is a measurement.

  python mock_chain6b.py setup --root MOCKROOT --from . --g0-world DIR [--registered FILE]
      builds a self-contained mock project root: the project's ROOT code and configs (the frozen G0 versions, as on the
      host before chain6b's install step), phase5\\ and phase6\\, a simulated G0 tree (DIR = a mock_g0.py tree whose
      gate_record.json carries the real G0 decisions), a registered copy of that record (precheck), a dummy checkpoint
      MOCK.pt with its pins (ckpt_pins.json), and hooks\\after_stageC13.ps1 that plays the freeze (FREEZE.json,
      chain6b_util.py freeze-write) after the chain's stage-C inputs step (confirm plan, forecast files).
  python mock_chain6b.py stageB|stageB_windows|stageC --root R      simulated results (results/continual.jsonl keys)
  python mock_chain6b.py g1g2 --root R                              simulated G1/G2 outputs + the REAL gates (mock_g1g2)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODELS = ["dense_100m", "moe_100m", "nestor_100m"]
CORPORA = ["pg19", "owm", "algstack", "fw2de"]


def _append(root: str, recs: list[dict]) -> None:
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    have = set()
    p = os.path.join(root, "results", "continual.jsonl")
    if os.path.exists(p):
        have = {json.loads(l)["key"] for l in open(p, encoding="utf-8") if l.strip()}
    with open(p, "a", encoding="utf-8") as fh:
        for r in recs:
            if r["key"] not in have:
                fh.write(json.dumps(dict(r, MOCK=True, time=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")


def stage_b(root: str) -> int:
    recs = []
    for i, e in enumerate(MODELS):
        for j, t in enumerate(CORPORA + ["dclm"]):
            recs.append({"key": f"B|{e}|before|{t}", "stage": "adapt_before", "model": e, "set": t, "ce": 3.0 + 0.1 * j + 0.01 * i,
                         "tokens": 2000000})
    for c in CORPORA:
        for i, e in enumerate(MODELS):
            recs.append({"key": f"B|{e}|after|{c}|target", "stage": "adapt_after", "model": e, "corpus": c, "set": c,
                         "ce": 2.8 + 0.1 * CORPORA.index(c) + 0.01 * i, "tokens": 2000000, "ckpt": "MOCK.pt"})
            recs.append({"key": f"B|{e}|after|{c}|dclm", "stage": "adapt_after", "model": e, "corpus": c, "set": "dclm",
                         "ce": 3.45 + 0.02 * i, "tokens": 2000000, "ckpt": "MOCK.pt"})
    _append(root, recs)
    print(f"[mock] stage B: {len(recs)} simulated results")
    return 0


def stage_b_windows(root: str) -> int:
    for e in MODELS:
        for tag in ["base"] + [f"adapt_{c}" for c in CORPORA]:
            d = os.path.join(root, "stageB_windows", e, tag)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "DONE"), "w").write(json.dumps({"MOCK": True}))
    print("[mock] stage-B windows: DONE markers")
    return 0


def stage_c(root: str) -> int:
    items = [("dense_100m", "equal_tokens"), ("moe_100m", "equal_tokens"), ("nestor_100m", "equal_tokens"),
             ("nestor_100m", "equal_flops_new")]
    recs = [{"key": f"C|{e}|{S}|{st}", "stage": "extension", "model": e, "S": S, "setting": st, "ce": 3.3, "tokens": 819200}
            for S in (8192, 16384) for e, st in items]
    _append(root, recs)
    print(f"[mock] stage C: {len(recs)} simulated results")
    return 0


def g1g2(root: str) -> int:
    import g1_gate
    import g1g2_plan
    import g2_gate
    import gate_common as GC
    import gate_rules_v13 as GR
    import mock_g1g2 as M

    os.chdir(root)
    cfg = g1g2_plan.load_config("g1g2_config.json")
    cfg["g0_probe_configs"] = {k: os.path.abspath(v) for k, v in cfg["g0_probe_configs"].items()}
    g0 = os.path.abspath(os.path.join("g0", "gate_record.json"))
    scen = copy.deepcopy(M.BASE_SCENARIO)
    p1 = g1g2_plan.build_g1(cfg, g0, "g1")
    world = M.simulate_g1(p1, scen)
    os.makedirs("g1", exist_ok=True)
    open(os.path.join("g1", "h100_nestor_full_s101.json"), "w").write(json.dumps({"F_dclm": scen["h100_F_dclm"], "MOCK": True}))
    r1 = g1_gate.gate(p1, None, "g1/plan_g1.json")
    r1["MOCK"] = "SIMULATED outputs (mock_chain6b.py) -- not a measurement"
    GC.finalize_record(r1, "g1/gate_record_G1.json")
    if (r1.get("decision") or {}).get("outcome") != "PROCEED":
        print(f"[mock] G1 {r1.get('decision')}")
        return 4
    p2 = g1g2_plan.build_g2(cfg, g0, "g1/gate_record_G1.json", "g2")
    M.simulate_g2(p2, scen, world)
    r2 = g2_gate.gate(p2, json.load(open("g1/gate_record_G1.json", encoding="utf-8")),
                      GR.Lookup(os.path.abspath("signflip_lookup_v1.csv")), plan_path="g2/plan_g2.json",
                      g1_path="g1/gate_record_G1.json", lookup_path="signflip_lookup_v1.csv")
    r2["MOCK"] = r1["MOCK"]
    GC.finalize_record(r2, "g2/gate_record_G2.json")
    f = r2["freeze"]
    print(f"[mock] G1 {r1['decision']['outcome']}; G2 {r2['outcome']}; m={f['m']} holm={f['holm_family']} "
          f"n_d={f['n_d']} C1 confirmatory {f['C1_confirmatory']}; C2 {f['c2_status']}")
    return 0 if r2["outcome"] == "COMPLETE" else 5


FREEZE_HOOK = r'''# MOCK hook (mock_chain6b.py setup): plays the freeze after stageC13 (the chain built the confirm plan and the
# verified forecast files) -- FREEZE.json with the 30 gate-determined parameters and the spec revision-3 hash list
# (chain6b_util.py freeze-write; entries the mock root cannot have are waived as MOCK).
$py = $env:MOCK_PYTHON
$saved = $env:PATH
$env:PATH = "$env:MOCK_LIBBIN;$saved"
'MOCK preregistration v1.3 (placeholder text for the mock freeze)' | Out-File -Encoding ascii preregistration_v1.3_MOCK.md
& $py chain6b_util.py freeze-write --root . --plan confirm/plan_confirm.json --queue confirm/queue_confirm.json --prereg preregistration_v1.3_MOCK.md --g0 g0/gate_record.json --g1 g1/gate_record_G1.json --g2 g2/gate_record_G2.json --decisions decisions_v13_points.md,decisions_v13_round2.md --approved-by MOCK --mock-waive-missing --notes 'mock end-to-end run' 2>&1 | Out-File -Append -Encoding utf8 logs\mock_freeze_hook.log
$rc = $LASTEXITCODE
$env:PATH = $saved
exit $rc
'''


def setup(root: str, src: str, g0_world: str, registered: str | None) -> int:
    if os.path.exists(root):                     # rule: never delete in the project tree -> rename the old mock root aside
        os.replace(root, f"{root}.old.{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(root)
    for f in os.listdir(src):
        sp = os.path.join(src, f)
        if os.path.isfile(sp) and f.endswith((".py", ".json", ".csv", ".ps1")) and f != "FREEZE.json":
            shutil.copy2(sp, os.path.join(root, f))
    for sub in ("phase5", "phase6", "inputs_v13", "frozen_g0"):  # inputs_v13: v1.3 install sources; frozen_g0: the
        # G0-frozen originals of the replace_frozen files (the source root holds the installed amendments)
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(root, sub),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(g0_world, os.path.join(root, "g0"))
    rec = os.path.join(root, "g0", "gate_record.json")
    txt = open(rec, encoding="utf-8").read().replace(json.dumps(os.path.abspath(g0_world))[1:-1],
                                                     json.dumps(os.path.join(root, "g0"))[1:-1])
    open(rec, "w", encoding="utf-8").write(txt)
    shutil.copy2(registered or rec, os.path.join(root, "g0", "gate_record_registered.json"))
    import pins_check as PC

    cks = json.load(open(rec, encoding="utf-8")).get("checkpoints") or {}
    path_of = {b: (v[0] if isinstance(v, list) else v) for b, v in cks.items()}   # the checkpoints the G0 record used
    path_of.setdefault("dense", "MOCK/dense.pt")
    for b, rel in path_of.items():
        fp = os.path.join(root, rel)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "wb").write(f"MOCK checkpoint {b}".encode())
    alias = {"dense_100m": "dense", "moe_100m": "moe_b1", "nestor_100m": "nestor", "moe_b1": "moe_b1", "moe_b2": "moe_b2",
             "nestor": "nestor"}
    pins = {"_comment": "MOCK pins", "created": "MOCK", "pins": {n: PC.entry(path_of[b], root) for n, b in alias.items()
                                                                if b in path_of}}
    pins["pins"]["moe_b3"] = PC.entry(None, root)
    PC.save(pins, os.path.join(root, "ckpt_pins.json"))
    os.makedirs(os.path.join(root, "hooks"), exist_ok=True)
    open(os.path.join(root, "hooks", "after_stageC13.ps1"), "wb").write(FREEZE_HOOK.replace("\n", "\r\n").encode("ascii"))
    if os.path.isdir(os.path.join(src, "sitecustom")):              # Python 3.12 temp-dir ACL fix (host fact)
        shutil.copytree(os.path.join(src, "sitecustom"), os.path.join(root, "sitecustom"),
                        ignore=shutil.ignore_patterns("__pycache__"))
    for d in ("logs", "results", "data"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    print(f"[mock] root {root}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MOCK stand-ins for the GPU steps of chain6b.ps1")
    ap.add_argument("cmd", choices=["setup", "stageB", "stageB_windows", "stageC", "g1g2"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--from", dest="src", default=None)
    ap.add_argument("--g0-world", default=None)
    ap.add_argument("--registered", default=None)
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.cmd == "setup":
        return setup(root, a.src, a.g0_world, a.registered)
    return {"stageB": stage_b, "stageB_windows": stage_b_windows, "stageC": stage_c, "g1g2": g1g2}[a.cmd](root)


if __name__ == "__main__":
    sys.exit(main())
