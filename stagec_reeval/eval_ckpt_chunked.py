"""Held-out cross-entropy of a checkpoint, CHUNKED-HEAD variant of eval_ckpt.py (stage-C re-evaluation, 2026-11-14).

Identical to eval_ckpt.py except that the unembedding is applied in row chunks of --head-rows tokens
(model.encode -> head.projection per chunk -> fp32 cross-entropy sum), so the full [tokens, vocab] logits
tensor (16384 x 100k fp32 = 6.5 GB) is never materialized. The projection is per token, so the math is the
same up to GEMM tiling round-off. Original docstring follows.

Held-out cross-entropy of a checkpoint -- the protocol of the repo's scripts/eval_heldout.py.

Same model rebuild (nestor.eval.checkpoint.load_experiment_checkpoint), same dataset class
(DclmTokenDataset, non-overlapping windows, no EOT remap), same fp32 forward and shifted token-mean
cross-entropy. Windows/single-GPU differences: DataLoader workers = 0, batch size 1 (fp32 logits
of 8 x 4096 x 100k would not fit in 16 GB), and the build-time flags of training (run_exp.SPEED_STACK)
so the parameter set matches.

  python eval_ckpt.py <experiment> <checkpoint.pt> <val.bin> [--block 4096] [--max-tokens N]
         [--first-per-doc]   # long-context protocol: first `block` tokens of each document >= block
         [--docs 200]
Prints one line: RESULT {json}
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

HERE = os.environ.get("NESTOR_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "repo"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment"); ap.add_argument("checkpoint"); ap.add_argument("val_bin")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=15_000_000)
    ap.add_argument("--first-per-doc", action="store_true")
    ap.add_argument("--docs", type=int, default=200)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast instead of fp32 (faster; ~1e-3 nats)")
    ap.add_argument("--head-rows", type=int, default=1024)
    args = ap.parse_args()
    import run_exp
    for k, v in run_exp.SPEED_STACK.items():
        os.environ.setdefault(k, v)
    os.environ["NESTOR_COMPILE"] = "0"
    os.environ.setdefault("DCLM_BLOCK_SIZE", str(args.block))
    os.environ.setdefault("DCLM_DATA_PATH", args.val_bin)
    import numpy as np, torch
    import compat; compat.install()
    import nestor.experiments.pretrain  # noqa: F401
    from nestor.eval.checkpoint import load_experiment_checkpoint
    model, info = load_experiment_checkpoint(args.experiment, Path(args.checkpoint), device="cuda")
    assert not info["missing"], f"missing keys: {info['missing'][:5]}"
    model.eval()
    if args.first_per_doc:
        from nestor.data.token_documents import open_token_bin
        toks, eot = open_token_bin(Path(args.val_bin))
        ends = np.flatnonzero(np.asarray(toks) == eot)
        starts = np.concatenate([[0], ends[:-1] + 1])
        keep = [(s, e) for s, e in zip(starts, ends) if e - s >= args.block][: args.docs]
        windows = [np.asarray(toks[s: s + args.block], dtype=np.int64) for s, e in keep]
    else:
        from nestor.configs.data import DclmDatasetConfig
        from nestor.data.datasets import DclmTokenDataset
        ds = DclmTokenDataset(DclmDatasetConfig(data_path=Path(args.val_bin), block_size=args.block))
        n = min(len(ds), -(-args.max_tokens // (args.block - 1)))
        windows = [ds[i][0] for i in range(n)]
    total, count, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for w in windows:
            x = torch.as_tensor(w).long().view(1, -1).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                hid = model.encode(x).pred[:, :-1, :]
            tg = x[:, 1:]
            T = hid.shape[1]
            for a in range(0, T, args.head_rows):
                b = min(T, a + args.head_rows)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    lg = model.head.projection(hid[:, a:b, :]) * model.head.logit_multiplier
                lg = lg.reshape(-1, lg.shape[-1]).float()
                total += float(torch.nn.functional.cross_entropy(lg, tg[:, a:b].reshape(-1), reduction="sum"))
                del lg
            count += int(tg.numel())
            if not args.first_per_doc and count >= args.max_tokens:
                break
    res = {"experiment": args.experiment, "checkpoint": str(args.checkpoint), "val_bin": os.path.basename(args.val_bin),
           "block": args.block, "first_per_doc": args.first_per_doc, "windows": len(windows), "tokens": count,
           "ce": total / max(count, 1), "step": info.get("step"), "seconds": round(time.time() - t0, 1),
           "precision": "bf16" if args.bf16 else "fp32", "head_rows": args.head_rows, "variant": "chunked-head"}
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
