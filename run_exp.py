"""Run a registered Nestor experiment on ONE Windows GPU with the authors' own pipeline.

Differences from `torchrun -m nestor.cli` (all infrastructure, no change to the model/recipe):
  * torch.distributed disabled (no NCCL on Windows; single process) and no DDP/FSDP wrapper;
  * DataLoader workers = 0 (the sandbox forbids the named pipes multiprocessing needs);
  * varlen_attn is_causal keyword shim (compat.py) -- inert here because the varlen paths
    need FlashAttention, which the Windows torch build lacks, so those flags are OFF;
  * speed flags: the paper's stack minus the FlashAttention-only paths, plus the authors' exact
    memory options NESTOR_FUSED_HEAD_CE (chunked head+CE, 1024-row chunks) for a 16 GB card;
  * compat.sdpa_headpad: exact zero-padding of head_dim 30 -> 32 so SDPA uses a fused kernel.
Usage:  python run_exp.py <experiment> [--steps N] [--warmup N] [--micro M] [--accum A]
"""
from __future__ import annotations

import argparse, dataclasses, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))

SPEED_STACK = {
    "NESTOR_DDP": "0", "NESTOR_DISABLE_FSDP": "1", "NESTOR_MUON_SHARD": "0",
    "NESTOR_COMPILE": "1", "NESTOR_BF16": "1", "NESTOR_DENSE_NO_ROUTER": "1",
    "NESTOR_GATHERED_CAUSAL_FLASH": "1",
    "NESTOR_GATHERED_VARLEN": "0", "NESTOR_ATTN_PACKED_VARLEN": "0", "NESTOR_TRITON_PACKED_ATTN": "0",
    "NESTOR_MOE_GROUPED_MM": "1", "NESTOR_CE_CHUNK_ROWS": "1024", "NESTOR_FUSED_HEAD_CE": "1", "NESTOR_COLUMN_SLOT_UPDATES": "1",
    "NESTOR_DENSE_TAIL_ATTN": "1", "NESTOR_GROUPED_MM_FUSED_GATE": "1",
    "NESTOR_STACKED_EXPERT_BANKS": "1", "NESTOR_BATCHED_ROUTERS": "1",
    "NESTOR_TRITON_INDEX_BUILD": "1", "NESTOR_TRITON_EXPERT_DISPATCH": "1",
    "DCLM_NUM_WORKERS": "0", "WANDB_MODE": "disabled", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--micro", type=int, default=None, help="sequences per micro-step")
    ap.add_argument("--accum", type=int, default=None, help="gradient-accumulation micro-steps")
    args = ap.parse_args()
    # The Nestor-100M b05 pretraining run can be supplied by the verified H100 run (same recipe, data
    # sha256, seed); then the queued local copy is skipped. Continued-training jobs are never skipped.
    skip_flag = os.path.join(HERE, "SKIP_LOCAL_NESTOR_PRETRAIN.flag")
    if (args.experiment == "nestor_100m" and os.environ.get("DCLM_RUN_SUFFIX") == "b05"
            and not os.environ.get("DCLM_INIT_CHECKPOINT") and os.path.exists(skip_flag)):
        print(f"[run_exp] nestor_100m b05 pretraining supplied by the H100 run ({skip_flag}); local run skipped", flush=True)
        os.remove(skip_flag)
        return
    for k, v in SPEED_STACK.items():
        os.environ.setdefault(k, v)
    if args.steps is not None:
        os.environ["DCLM_MAX_STEPS"] = str(args.steps)
    if args.warmup is not None:
        os.environ["DCLM_WARMUP_STEPS"] = str(args.warmup)
    if args.micro is not None:
        os.environ["DCLM_GLOBAL_BATCH"] = str(args.micro)
    if args.accum is not None:
        os.environ["DCLM_GRAD_ACCUM"] = str(args.accum)

    import torch
    import compat
    compat.install()
    import subset_adapt; subset_adapt.install()   # 2026-11-13: env-activated (NESTOR_SUBSET_SPEC); no-op when unset
    import adapt_extras; adapt_extras.install()   # 2026-11-13: env-activated; strict no-op when unset
    # fail loudly (OOM) instead of letting the Windows driver spill VRAM into system RAM
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_FRAC", "0.88")))

    from nestor.configs.distributed import DistributedConfig
    from nestor.experiments.registry import get_experiment
    from nestor.pipelines.train import TrainPipeline
    import nestor.experiments.pretrain  # noqa: F401  (registers the 12 experiments)

    exp = get_experiment(args.experiment)()
    exp.jobs = [dataclasses.replace(j, distributed=DistributedConfig(enabled=False)) for j in exp.jobs]
    print("[run_exp] flags:", {k: os.environ[k] for k in sorted(SPEED_STACK)}, flush=True)
    for job in exp.jobs:
        TrainPipeline.from_config(job_config=job, experiment_config=exp).run()


if __name__ == "__main__":
    main()
