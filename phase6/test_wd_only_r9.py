""" (post-freeze amendment A-1): the WD-only control's schedule comes from the run's own sources.

Real configuration of the failed job c1_moe_b1_books_d2_A (., read-only copies of run_env.json and of
its frozen queue job): run_env.json has no DCLM_MAX_STEPS / DCLM_WARMUP_STEPS (run_queue passes --steps/--warmup to
run_exp.py), so the rebuilt job fell back to the pretraining recipe (19,080 steps). wd_only_schedule must raise without
a source, resolve 96 / 10 from the frozen queue job, refuse disagreeing sources and honour a P-16 stop. The functions
of weight_controls.py other than wd_only_state are byte-identical to the frozen file (R8-type equivalence of the
non-WD variants)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "phase6" else HERE
for _p in (CODE_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
WC_PATH = os.path.join(HERE, "weight_controls.py")
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("weight_controls_r9", WC_PATH)
WC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(WC)

REAL_ENV = {
 "DCLM_RUN_SUFFIX": "c1_moe_b1_books_d2_A",
 "DCLM_SEED": "7002",
 "DCLM_LR_MUON": "0.0025",
 "DCLM_LR_ADAMW": "7.5e-05",
 "DCLM_WEIGHT_DECAY": "0.1",
 "DCLM_INIT_CHECKPOINT": "outputs/moe_100m_b05/20261101_002357__moe_100m_b05_train/checkpoints/model_final.pt",
 "DCLM_DATA_PATH": "data/domain/pg19_train.bin",
 "NESTOR_FREEZE_ROUTERS": "1",
 "NESTOR_INTERIM_EVAL": "{\"block\": 4096, \"routing\": false, \"sets\": [{\"head\": 64, \"name\": \"dclm\", \"path\": \"data/dclm/dclm_val.bin\", \"select\": \"even\", \"tokens\": 2048000}, {\"head\": 64, \"name\": \"books\", \"path\": \"data/domain/pg19_test.bin\", \"select\": \"even\", \"span\": [0.5, 1.0], \"span_align\": \"doc\", \"tokens\": 2048000}, {\"head\": 64, \"name\": \"code\", \"path\": \"data/domain/algstack_test.bin\", \"select\": \"even\", \"span\": [0.5, 1.0], \"tokens\": 2048000}, {\"head\": 64, \"name\": \"math\", \"path\": \"data/domain/owm_test.bin\", \"select\": \"even\", \"span\": [0.5, 1.0], \"tokens\": 2048000}], \"steps\": [0, 8, 24, 48], \"out\": \"outputs\\\\confirm\\\\c1_moe_b1_books_d2_A\\\\m2\\\\interim\"}"
}
REAL_JOB = {'env': {'DCLM_SEED': '7002'},
 'experiment': 'moe_100m',
 'id': 'c1_moe_b1_books_d2_A',
 'kind': 'train',
 'steps': 96,
 'warmup': 10}
SGD_JOB = {'env': {'DCLM_SEED': '8001'},
 'id': 'c2_moe_b1_d1_sgd_H',
 'interim': {'block': 4096,
             'full_sets': ['dclm', 'german'],
             'full_steps': [8],
             'routing': False,
             'sets': [{'head': 64,
                       'name': 'dclm',
                       'path': 'data/dclm/dclm_val.bin',
                       'select': 'even',
                       'tokens': 2048000},
                      {'head': 64,
                       'name': 'german',
                       'path': 'data/domain/fw2de_test.bin',
                       'select': 'even',
                       'span': [0.5, 1.0],
                       'tokens': 2048000}],
             'steps': [0, 8, 24],
             'stop_after': 24},
 'kind': 'train',
 'steps': 96,
 'warmup': 10}
FROZEN_WC_SHA256 = "473ff8c58542f7f398431181cf11afb14b0b8bf0f261591a88120f3fdc6b2835"
UNCHANGED_SOURCE_SHA256 = {
 "GROUP_DOC": "212454f771c361f0ed5f9f822fe0529c24fbee08c8c8937687c1e402dc400505",
 "_SCOPE_PART": "e17797c25b8993a76f2d75d66aa9856d963249fb4f6449ea7bcf8f118a67944e",
 "_gen_for": "c61eec33e95351a9a0728d86e0253d128acbc5ac519311c4cbfcc57702fe4a07",
 "_group_pred": "5c7edae34e10ee13f15192af5789849be62869783b0f1b306a82ef2e9f15fce7",
 "_haar_frame": "e95356652592177a032e957d434e6873b237ffecc5ecf7e758dc1a5d656a0077",
 "_scope": "9f21fd09b56b5504890d46ad4b84f78958d93313a7c9ee8075d229b9f6ffcf12",
 "displacement": "4dd9d17828ffe06197ae50de107dc153c188e237972e6bdb31b0bf8664447c2e",
 "group_partition": "7e9a626e4d079f9457daaae4995b8db4904a8c2e4475aab7ad7335e42c41b684",
 "random_direction_state": "c5e950627d9da6c6d2663f029bd21bb7632f42f7269da01a32572643e2e0cdc1",
 "reflect_state": "fdda943e1c753f467b9c975ddf9ee510bb7ff00c79fcc1ead1d4082a3e1d6fed",
 "revert_state": "9bf3936a1c443d348df6102c3f0089a2c8dff3653c1681edbfc4c46caa24eb03",
 "sv_matched_random": "df58414fb5e981dd28151d1e2e0bf6be5f26107e254fce88e73099607e483aff",
 "temp_env": "f649c4bf919f5e72ee8994a425bb53c026fbbb28b3d2d3174c8351054e2e9293"
}


def _world(tmp_path, jobs, rid="c1_moe_b1_books_d2_A"):
    root = tmp_path / "root"
    (root / "confirm").mkdir(parents=True)
    (root / "FREEZE.json").write_text(json.dumps({"queue": {"path": "confirm/queue_confirm.json"}}))
    (root / "confirm" / "queue_confirm.json").write_text(json.dumps({"jobs": jobs}))
    run = root / "outputs" / "confirm" / rid / "m2"
    run.mkdir(parents=True)
    (run / "subset_manifest.json").write_text("{}")
    return str(run / "subset_manifest.json")


def test_real_config_without_a_step_source_raises():
    assert not any(k in REAL_ENV for k in WC.WD_STEP_KEYS)          # the real run_env.json: the cause of the hang
    with pytest.raises(ValueError, match="refusing to fall back to the pretraining defaults"):
        WC.wd_only_schedule(REAL_ENV, None)


def test_real_config_resolves_96_steps_and_warmup_10_from_the_frozen_queue_job(tmp_path):
    man = _world(tmp_path, [REAL_JOB])
    s = WC.wd_only_schedule(REAL_ENV, man)
    assert (s["max_steps"], s["warmup"], s["executed"], s["stop_after"]) == (96, 10, 96, None)
    assert s["sources"] == ["frozen queue job (queue_confirm.json)"]
    env_ok = dict(REAL_ENV, DCLM_MAX_STEPS="96", DCLM_WARMUP_STEPS="10")       # a run_env.json that records them agrees
    assert WC.wd_only_schedule(env_ok, man)["sources"] == ["frozen queue job (queue_confirm.json)", "run_env"]


def test_disagreeing_or_incomplete_sources_raise(tmp_path):
    man = _world(tmp_path, [REAL_JOB])
    with pytest.raises(ValueError, match="disagree"):
        WC.wd_only_schedule(dict(REAL_ENV, DCLM_MAX_STEPS="19080", DCLM_WARMUP_STEPS="1526"), man)
    with pytest.raises(ValueError, match="disagree"):
        WC.wd_only_schedule(REAL_ENV, man, max_steps=19080, warmup=1526)
    with pytest.raises(ValueError, match="incomplete"):
        WC.wd_only_schedule(REAL_ENV, None, max_steps=96)
    bad_seed = dict(REAL_JOB, env={"DCLM_SEED": "1"})
    with pytest.raises(ValueError, match="DCLM_SEED"):
        WC.wd_only_schedule(REAL_ENV, _world(tmp_path / "s", [bad_seed]))
    other = dict(REAL_JOB, id="another_run")                                   # FREEZE.json found, job absent
    with pytest.raises(ValueError, match="unknown"):
        WC.wd_only_schedule(REAL_ENV, _world(tmp_path / "o", [other]))


def test_a_stopped_run_executes_its_stop_step_on_the_registered_schedule(tmp_path):
    env = {"DCLM_RUN_SUFFIX": SGD_JOB["id"], "DCLM_SEED": SGD_JOB["env"]["DCLM_SEED"], "NESTOR_FREEZE_ROUTERS": "1",
           "NESTOR_INTERIM_EVAL": json.dumps(dict(SGD_JOB["interim"], out="x"))}
    s = WC.wd_only_schedule(env, _world(tmp_path, [SGD_JOB], rid=SGD_JOB["id"]))
    assert (s["max_steps"], s["warmup"], s["executed"], s["stop_after"]) == (96, 10, 24, 24)


def test_functions_other_than_wd_only_state_are_byte_identical_to_the_frozen_file():
    src = open(WC_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    seen = {}
    for n in tree.body:
        name = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if name in UNCHANGED_SOURCE_SHA256:
            seg = ast.get_source_segment(src, n)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.decorator_list:
                seg = "\n".join("@" + ast.get_source_segment(src, d) for d in n.decorator_list) + "\n" + seg
            seen[name] = hashlib.sha256(seg.encode()).hexdigest()
    assert seen == UNCHANGED_SOURCE_SHA256
    for k in ("random_direction_state", "reflect_state", "revert_state", "displacement", "temp_env", "sv_matched_random"):
        assert k in seen
