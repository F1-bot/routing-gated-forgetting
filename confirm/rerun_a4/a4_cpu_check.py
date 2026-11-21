"""a4_cpu_check.py -- CPU evidence for amendment A-4 (no GPU; nothing written outside --tmp and --out).

Checks, with the frozen code (repo/nestor/backbones/attention_residual.py, subset_adapt.py) and the tiny models of
test_subset_adapt.py (the authors' builders, d = 64, vocab 256, the adaptation recipe, frozen routers):
  1. DepthAttentionRead: with exactly one source the pseudo-query's gradient is exactly zero (fp32, and bf16 sources under
     CPU autocast); with two or more sources it is not. One AdamW step on a zero-initialised query with a zero gradient
     leaves it bitwise zero, with or without weight decay.
  2. A freeze-mode run of the tiny MoE (spec freeze kind moe, as position 154) and of the tiny Nestor (freeze the columns of
     the non-tail blocks, as position 155) leaves exactly one trainable tensor unchanged: the query of the first read.
  3. subset_adapt.verify_checkpoints on those runs: ok false without allow_unchanged; ok true with allow_unchanged, and
     trainable_unchanged is the same list in both; the CLI (subset_adapt.py verify --out, with and without
     --allow-unchanged) writes that list to its audit JSON and exits 0 / 1.
    python confirm/rerun_a4/a4_cpu_check.py --root <project root> --tmp <scratch folder> --out <json>
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--tmp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
os.makedirs(a.tmp, exist_ok=False)
sys.path.insert(0, os.path.join(ROOT, "repo"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import test_subset_adapt as T  # noqa: E402  (sets the authors' CPU build flags, like the unit tests)
import subset_adapt as SA  # noqa: E402
from nestor.backbones.attention_residual import DepthAttentionRead  # noqa: E402

torch.set_num_threads(1)
RES = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
       "files_sha256": {}, "read_unit": {}, "adamw_step": {}, "tiny_runs": {}}
for f in ("repo/nestor/backbones/attention_residual.py", "subset_adapt.py", "test_subset_adapt.py"):
    RES["files_sha256"][f] = hashlib.sha256(open(os.path.join(ROOT, *f.split("/")), "rb").read()).hexdigest()

# 1. the read itself
for n_src in (1, 2, 3):
    for dt in ("fp32", "bf16_autocast"):
        torch.manual_seed(0)
        read = DepthAttentionRead(16, max_sources=5)
        src = [torch.randn(2, 6, 16, dtype=torch.float32 if dt == "fp32" else torch.bfloat16) for _ in range(n_src)]
        if dt == "fp32":
            out = read(src)
        else:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = read(src)
        (out.float() * torch.randn(out.shape)).sum().backward()
        g, vg = read.query.grad, read.value_gains.grad
        RES["read_unit"][f"{n_src}_source_{dt}"] = {
            "query_grad_nonzero": int(torch.count_nonzero(g)), "query_grad_max_abs": float(g.abs().max()),
            "value_gains_grad_nonzero": int(torch.count_nonzero(vg)), "query_numel": g.numel()}
for wd in (0.0, 0.1):
    q = torch.nn.Parameter(torch.zeros(16))
    opt = torch.optim.AdamW([q], lr=1e-3, betas=(0.9, 0.95), weight_decay=wd)
    q.grad = torch.zeros(16)
    opt.step()
    RES["adamw_step"][f"weight_decay_{wd}"] = {"query_bitwise_zero_after_step": bool(torch.equal(q.detach(), torch.zeros(16)))}

# 2-3. tiny freeze-mode runs and the audit
SPECS = {"moe": (T.moe_backbone, {"mode": "freeze", "select": [{"kind": "moe"}]}),
         "nestor": (T.nestor_backbone, {"mode": "freeze", "select": [{"kind": "column", "block": [0, 1]}]})}
for fam, (bb, spec) in SPECS.items():
    SA.reset()
    SA.register_optimizer_hooks()
    os.environ["NESTOR_FREEZE_ROUTERS"] = "1"
    os.environ.pop(SA.VERIFY_ENV, None)
    model = T.build(bb())
    plan = T.prepare(model, spec)
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before = T.snapshot(model)
    opt, sch = T.make_opt(model, steps=6)
    losses = T.train(model, opt, sch, steps=6)
    viol, unchanged = T.audit(plan, before, model)
    d = os.path.join(a.tmp, fam)
    os.makedirs(d)
    init_p, final_p, man_p = (os.path.join(d, x) for x in ("init.pt", "final.pt", "subset_manifest.json"))
    torch.save({"model": init_sd}, init_p)
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}}, final_p)
    json.dump(plan.manifest, open(man_p, "w", encoding="utf-8"), indent=1, default=str)
    r0 = SA.verify_checkpoints(init_p, final_p, man_p)
    r1 = SA.verify_checkpoints(init_p, final_p, man_p, allow_unchanged=True)
    cli = {}
    for flag in (False, True):
        out = os.path.join(d, f"audit_cli_{'allow' if flag else 'plain'}.json")
        argv = [sys.executable, os.path.join(ROOT, "subset_adapt.py"), "verify", "--init", init_p, "--final", final_p,
                "--manifest", man_p, "--out", out] + (["--allow-unchanged"] if flag else [])
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=a.tmp,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        doc = json.load(open(out, encoding="utf-8")) if os.path.isfile(out) else {}
        cli["allow_unchanged" if flag else "plain"] = {"rc": p.returncode, "ok": doc.get("ok"), "allow_unchanged": doc.get("allow_unchanged"),
                                                       "trainable_unchanged": doc.get("trainable_unchanged")}
    trainable = [r["name"] for r in plan.manifest["params"] if r["status"] == "trainable"]
    RES["tiny_runs"][fam] = {
        "spec": spec, "steps": 6, "losses_finite": all(x == x for x in losses),
        "trainable_tensors": len(trainable),
        "depth_read_trainable": sorted(n for n in trainable if "kimi_" in n or "final_depth_read" in n)[:6],
        "in_process_audit": {"frozen_violations": viol, "trainable_unchanged": unchanged},
        "verify_checkpoints": {"plain": {"ok": r0["ok"], "trainable_unchanged": r0["trainable_unchanged"],
                                         "frozen_mismatch": r0["frozen_mismatch"], "buffer_mismatch": r0["buffer_mismatch"],
                                         "optimizer_violations": r0["optimizer_state"]["violations"]},
                               "allow_unchanged": {"ok": r1["ok"], "allow_unchanged": r1["allow_unchanged"],
                                                   "trainable_unchanged": r1["trainable_unchanged"]}},
        "cli": cli}
    SA.reset()

Q0 = "backbone.blocks.0.layers.0.kimi_attn_read.query"
ru = RES["read_unit"]
checks = {
    "one_source_query_grad_exactly_zero": all(ru[f"1_source_{dt}"]["query_grad_nonzero"] == 0 for dt in ("fp32", "bf16_autocast")),
    "one_source_value_gain_grad_only_first": all(ru[f"1_source_{dt}"]["value_gains_grad_nonzero"] == 1 for dt in ("fp32", "bf16_autocast")),
    "two_or_more_sources_query_grad_nonzero": all(ru[f"{n}_source_{dt}"]["query_grad_nonzero"] > 0 for n in (2, 3) for dt in ("fp32", "bf16_autocast")),
    "adamw_keeps_zero_query": all(v["query_bitwise_zero_after_step"] for v in RES["adamw_step"].values()),
}
for fam, r in RES["tiny_runs"].items():
    checks[f"{fam}_only_first_read_query_unchanged"] = (r["in_process_audit"]["trainable_unchanged"] == [Q0]
                                                        and r["in_process_audit"]["frozen_violations"] == [])
    checks[f"{fam}_verify_plain_fails_on_it"] = (r["verify_checkpoints"]["plain"]["ok"] is False
                                                 and r["verify_checkpoints"]["plain"]["trainable_unchanged"] == [Q0])
    checks[f"{fam}_verify_allow_passes_and_still_lists_it"] = (r["verify_checkpoints"]["allow_unchanged"]["ok"] is True
                                                               and r["verify_checkpoints"]["allow_unchanged"]["trainable_unchanged"] == [Q0])
    checks[f"{fam}_cli_writes_the_list"] = (r["cli"]["plain"]["rc"] == 1 and r["cli"]["allow_unchanged"]["rc"] == 0
                                            and r["cli"]["allow_unchanged"]["trainable_unchanged"] == [Q0]
                                            and r["cli"]["allow_unchanged"]["allow_unchanged"] is True)
RES["checks"] = checks
RES["passed"], RES["total"] = sum(bool(v) for v in checks.values()), len(checks)
RES["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(RES, indent=1) + "\n")
print(f"[a4-cpu-check] {RES['passed']}/{RES['total']} checks pass -> {a.out}")
sys.exit(0 if RES["passed"] == RES["total"] else 1)
