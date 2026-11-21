"""Amendment A-2 (post-freeze): the WD-only control of a run that is not a job of the frozen queue (the C3 revert
evaluations of the pre-freeze G1 FULL runs, queue positions 137 / 142 / 147) takes its steps and warm-up from the queue that
trained it -- the queue whose run_queue state records the run's folder as the job root, status done, same DCLM_SEED.
Real literals: the three G1 runs (run_env.json has no DCLM_MAX_STEPS / DCLM_WARMUP_STEPS; g1/queue_g1.json: train, 96 steps,
warm-up 10, DCLM_SEED 101). Every item of weight_controls.py except wd_only_schedule is byte-identical to the installed A-1
file (29a4e176...); confirm-queue jobs resolve exactly as before."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
WC_PATH = os.path.join(HERE, "weight_controls.py")
_spec = importlib.util.spec_from_file_location("weight_controls_a2", WC_PATH)
WC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WC)

REAL_G1 = {
 "g1_moe_b1_full_german_s101": {
  "attempt": "m2",
  "job": {
   "env": {
    "DCLM_RUN_SUFFIX": "g1_moe_b1_full_german_s101",
    "DCLM_SEED": "101"
   },
   "id": "g1_moe_b1_full_german_s101",
   "kind": "train",
   "steps": 96,
   "warmup": 10
  },
  "run_env": {
   "DCLM_RUN_SUFFIX": "g1_moe_b1_full_german_s101",
   "DCLM_SEED": "101",
   "DCLM_WEIGHT_DECAY": "0.1",
   "NESTOR_FREEZE_ROUTERS": "1"
  }
 },
 "g1_moe_b1_full_math_s101": {
  "attempt": "m2",
  "job": {
   "env": {
    "DCLM_RUN_SUFFIX": "g1_moe_b1_full_math_s101",
    "DCLM_SEED": "101"
   },
   "id": "g1_moe_b1_full_math_s101",
   "kind": "train",
   "steps": 96,
   "warmup": 10
  },
  "run_env": {
   "DCLM_RUN_SUFFIX": "g1_moe_b1_full_math_s101",
   "DCLM_SEED": "101",
   "DCLM_WEIGHT_DECAY": "0.1",
   "NESTOR_FREEZE_ROUTERS": "1"
  }
 },
 "g1_nestor_full_german_s101": {
  "attempt": "m2c",
  "job": {
   "env": {
    "DCLM_RUN_SUFFIX": "g1_nestor_full_german_s101",
    "DCLM_SEED": "101"
   },
   "id": "g1_nestor_full_german_s101",
   "kind": "train",
   "steps": 96,
   "warmup": 10
  },
  "run_env": {
   "DCLM_RUN_SUFFIX": "g1_nestor_full_german_s101",
   "DCLM_SEED": "101",
   "DCLM_WEIGHT_DECAY": "0.1",
   "NESTOR_FREEZE_ROUTERS": "1"
  }
 }
}
A1_SHA256 = "29a4e176bb381ba86096f921ee5121dd2cd800a03701cfa73bd57877ce4af210"
A1_UNCHANGED_ITEMS_SHA256 = {
 "GROUP_DOC": "212454f771c361f0ed5f9f822fe0529c24fbee08c8c8937687c1e402dc400505",
 "WD_STEP_KEYS": "cc2f788106b6428c84afc51bb3c866eccb507a2a5b295174d53954418f308361",
 "_SCOPE_PART": "e17797c25b8993a76f2d75d66aa9856d963249fb4f6449ea7bcf8f118a67944e",
 "_frozen_queue_job": "cf8c4d60aa481bf94cc4ad693a983c0e0defc1c9e6b4d557b030c38f0459ed50",
 "_gen_for": "c61eec33e95351a9a0728d86e0253d128acbc5ac519311c4cbfcc57702fe4a07",
 "_group_pred": "5c7edae34e10ee13f15192af5789849be62869783b0f1b306a82ef2e9f15fce7",
 "_haar_frame": "e95356652592177a032e957d434e6873b237ffecc5ecf7e758dc1a5d656a0077",
 "_interim_stop": "1fed001aee6b4126825d71f8c6e73ed5bcb4e056d8f05877ccaf65d90857e826",
 "_scope": "9f21fd09b56b5504890d46ad4b84f78958d93313a7c9ee8075d229b9f6ffcf12",
 "_train_log": "cda2480227cdf7e5cbbab0027d4b0b5e824626d136a95e69d2de3cf9404672e7",
 "_wd_device": "d19b01dca24a6a179d3dbb929201418e8961f8e267a123b13cf4da7ac1a6801b",
 "displacement": "4dd9d17828ffe06197ae50de107dc153c188e237972e6bdb31b0bf8664447c2e",
 "group_partition": "7e9a626e4d079f9457daaae4995b8db4904a8c2e4475aab7ad7335e42c41b684",
 "random_direction_state": "c5e950627d9da6c6d2663f029bd21bb7632f42f7269da01a32572643e2e0cdc1",
 "reflect_state": "fdda943e1c753f467b9c975ddf9ee510bb7ff00c79fcc1ead1d4082a3e1d6fed",
 "revert_state": "9bf3936a1c443d348df6102c3f0089a2c8dff3653c1681edbfc4c46caa24eb03",
 "sv_matched_random": "df58414fb5e981dd28151d1e2e0bf6be5f26107e254fce88e73099607e483aff",
 "temp_env": "f649c4bf919f5e72ee8994a425bb53c026fbbb28b3d2d3174c8351054e2e9293",
 "wd_only_state": "816383ecfd5d17359d177b427b7290a6dcc6eb48d118c0304e3c7465d7c6ff68"
}
D2A_JOB = {"id": "c1_moe_b1_books_d2_A", "kind": "train", "steps": 96, "warmup": 10, "env": {"DCLM_SEED": "7002"}}
D2A_ENV = {"DCLM_RUN_SUFFIX": "c1_moe_b1_books_d2_A", "DCLM_SEED": "7002", "NESTOR_FREEZE_ROUTERS": "1"}


def _world(tmp_path, rid, *, g1_job=None, g1_rec=None, confirm_jobs=(), extra_queue=None, attempt="m2"):
    root = tmp_path / "root"
    for d in ("confirm", "g1", "g2", "results"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "FREEZE.json").write_text(json.dumps({"queue": {"path": "confirm/queue_confirm.json"}}))
    (root / "confirm" / "queue_confirm.json").write_text(json.dumps({"jobs": list(confirm_jobs), "state": "results\\queue_state_confirm.json"}))
    run = root / "outputs" / "g1" / rid / attempt
    run.mkdir(parents=True)
    if g1_job is not None:
        (root / "g1" / "queue_g1.json").write_text(json.dumps({"jobs": [g1_job], "state": "results\\queue_state_g1.json"}))
        rec = dict({"status": "done", "root": str(run)}, **(g1_rec or {}))
        (root / "results" / "queue_state_g1.json").write_text(json.dumps({rid: rec}))
    if extra_queue is not None:
        job, rec = extra_queue
        (root / "g2" / "queue_g2.json").write_text(json.dumps({"jobs": [job], "state": "results/queue_state_g2.json"}))
        (root / "results" / "queue_state_g2.json").write_text(json.dumps({rid: dict({"status": "done", "root": str(run)}, **(rec or {}))}))
    return str(run / "subset_manifest.json"), root, run


@pytest.mark.parametrize("rid", sorted(REAL_G1))
def test_g1_full_runs_resolve_from_their_training_queue(tmp_path, rid):
    r = REAL_G1[rid]
    assert not any(k in r["run_env"] for k in WC.WD_STEP_KEYS)                # the real run_env.json: the cause (D-1)
    man, _, _ = _world(tmp_path, rid, g1_job=r["job"], attempt=r["attempt"])
    s = WC.wd_only_schedule(r["run_env"], man)
    assert s == {"max_steps": 96, "warmup": 10, "executed": 96, "stop_after": None, "sources": ["training queue job (g1/queue_g1.json)"]}


def test_without_a_training_queue_it_still_raises(tmp_path):
    rid = "g1_moe_b1_full_math_s101"
    man, _, _ = _world(tmp_path, rid)
    with pytest.raises(ValueError, match="no training queue records this run's folder.*refusing to fall back to the pretraining defaults"):
        WC.wd_only_schedule(REAL_G1[rid]["run_env"], man)


@pytest.mark.parametrize("rec, job_env, why", [
    ({"root": "C:/elsewhere/m2"}, None, "does not record this run's folder"),
    ({"status": "failed"}, None, "state status 'failed'"),
    ({}, {"DCLM_SEED": "102"}, "DCLM_SEED '102'"),
    ({}, {}, "DCLM_SEED None"),
])
def test_a_candidate_queue_is_rejected_unless_root_status_and_seed_match(tmp_path, rec, job_env, why):
    rid = "g1_moe_b1_full_german_s101"
    job = dict(REAL_G1[rid]["job"])
    if job_env is not None:
        job["env"] = job_env
    man, _, _ = _world(tmp_path, rid, g1_job=job, g1_rec=rec)
    with pytest.raises(ValueError, match="refusing to fall back") as e:
        WC.wd_only_schedule(REAL_G1[rid]["run_env"], man)
    assert "rejected" in str(e.value) and why in str(e.value) and "g1/queue_g1.json" in str(e.value)


def test_the_frozen_queue_job_has_priority_and_the_training_queue_is_not_consulted(tmp_path):
    rid = "g1_moe_b1_full_math_s101"
    conf = dict(REAL_G1[rid]["job"])
    other = dict(REAL_G1[rid]["job"], steps=48)                                # would disagree if it were consulted
    man, _, _ = _world(tmp_path, rid, g1_job=other, confirm_jobs=[conf])
    s = WC.wd_only_schedule(REAL_G1[rid]["run_env"], man)
    assert s["sources"] == ["frozen queue job (queue_confirm.json)"] and s["max_steps"] == 96


def test_an_explicit_queue_path_disables_the_training_queue_search(tmp_path):
    rid = "g1_moe_b1_full_math_s101"
    man, root, _ = _world(tmp_path, rid, g1_job=REAL_G1[rid]["job"])
    with pytest.raises(ValueError, match="refusing to fall back"):
        WC.wd_only_schedule(REAL_G1[rid]["run_env"], man, queue_path=str(root / "confirm" / "queue_confirm.json"))


def test_several_training_queues_must_agree(tmp_path):
    rid = "g1_nestor_full_german_s101"
    r = REAL_G1[rid]
    man, _, _ = _world(tmp_path / "same", rid, g1_job=r["job"], extra_queue=(dict(r["job"]), None), attempt=r["attempt"])
    s = WC.wd_only_schedule(r["run_env"], man)
    assert s["sources"] == ["training queue job (g1/queue_g1.json, g2/queue_g2.json)"] and (s["max_steps"], s["warmup"]) == (96, 10)
    man, _, _ = _world(tmp_path / "diff", rid, g1_job=r["job"], extra_queue=(dict(r["job"], warmup=20), None), attempt=r["attempt"])
    with pytest.raises(ValueError, match="disagree"):
        WC.wd_only_schedule(r["run_env"], man)


def test_run_env_steps_and_the_training_queue_must_agree(tmp_path):
    rid = "g1_moe_b1_full_math_s101"
    man, _, _ = _world(tmp_path, rid, g1_job=REAL_G1[rid]["job"])
    env = dict(REAL_G1[rid]["run_env"], DCLM_MAX_STEPS="48", DCLM_WARMUP_STEPS="10")
    with pytest.raises(ValueError, match="sources of the step count disagree"):
        WC.wd_only_schedule(env, man)


def test_confirm_queue_jobs_resolve_exactly_as_before(tmp_path):
    root = tmp_path / "root"
    (root / "confirm").mkdir(parents=True)
    (root / "FREEZE.json").write_text(json.dumps({"queue": {"path": "confirm/queue_confirm.json"}}))
    (root / "confirm" / "queue_confirm.json").write_text(json.dumps({"jobs": [D2A_JOB], "state": "results/queue_state_confirm.json"}))
    run = root / "outputs" / "confirm" / "c1_moe_b1_books_d2_A" / "m2"
    run.mkdir(parents=True)
    s = WC.wd_only_schedule(D2A_ENV, str(run / "subset_manifest.json"))
    assert s == {"max_steps": 96, "warmup": 10, "executed": 96, "stop_after": None, "sources": ["frozen queue job (queue_confirm.json)"]}


def test_every_item_except_wd_only_schedule_is_byte_identical_to_the_installed_a1_file():
    src = open(WC_PATH, encoding="utf-8").read()
    seen = {}
    for n in ast.parse(src).body:
        name = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if name in A1_UNCHANGED_ITEMS_SHA256:
            seg = ast.get_source_segment(src, n)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.decorator_list:
                seg = "\n".join("@" + ast.get_source_segment(src, d) for d in n.decorator_list) + "\n" + seg
            seen[name] = hashlib.sha256(seg.encode()).hexdigest()
    assert seen == A1_UNCHANGED_ITEMS_SHA256
    assert {"wd_only_state", "_frozen_queue_job", "_wd_device", "_train_log", "random_direction_state", "reflect_state", "revert_state"} <= set(seen)
