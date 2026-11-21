"""Tiny registered experiments for CPU tests: `eval_windows.py run tiny_nestor ... --register tiny_experiments.py`.

Same definitions as the training child of test_subset_adapt.py / test_phase3.py (TINY scale, the
authors' P._experiment, batch shape (2, 2)), so a model_final.pt written by a tiny TrainPipeline run
loads through nestor.eval.checkpoint.load_experiment_checkpoint.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "repo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_subset_adapt as T  # noqa: E402  (sets the CPU build flags before nestor is imported)
import nestor.experiments.pretrain as P  # noqa: E402
from nestor.experiments.registry import register_experiment  # noqa: E402


def register() -> None:
    for family in ("nestor", "moe"):
        P.BATCH_SHAPE[(family, "tiny")] = (2, 2)

        def tiny_exp(family=family):
            bb = T.nestor_backbone() if family == "nestor" else T.moe_backbone()
            return P._experiment(family, T.TINY, bb)

        tiny_exp.__name__ = f"tiny_{family}"
        try:
            register_experiment(tiny_exp)
        except ValueError:          # already registered in this process
            pass


register()
