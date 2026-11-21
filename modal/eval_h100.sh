#!/bin/bash
# Cross-platform forward check of the TRAINED Nestor-100M checkpoint: the same eval_ckpt.py
# protocol as the local stage A (fp32, 4k windows, val09 10M tokens + val10 5M tokens), here on H100 with the
# FlashAttention varlen paths. Also stages chunk part_00 for harvest.
set -u
WORK=$(pwd); OUT="$WORK/out"; mkdir -p "$OUT"
cd /root/nestor
cp "$WORK/eval_ckpt.py" /root/nestor/eval_ckpt.py
export TIKTOKEN_CACHE_DIR=/root/cache/tiktoken WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NESTOR_GATHERED_VARLEN=1 NESTOR_ATTN_PACKED_VARLEN=1 NESTOR_TRITON_PACKED_ATTN=1
if [ ! -f /root/data/dclm_val10.bin ]; then
  python -c "from huggingface_hub import hf_hub_download as d; d('mlfoundations/dclm-baseline-1.0', 'global-shard_10_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst', repo_type='dataset', local_dir='/root/data/raw')"
  PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 10 \
      --out /root/data/dclm_val10.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val10.txt" 2>&1
  rm -rf /root/data/dclm_val10.bin.parts
fi
sha256sum /root/data/dclm_val10.bin | tee "$OUT/sha256_val10.txt"
CK=/root/export/nestor_100m_b05_model_final_slim.pt
python eval_ckpt.py nestor_100m "$CK" /root/data/dclm_val.bin --block 4096 --max-tokens 10000000 > "$OUT/eval_val09_h100.txt" 2>&1
if ! grep -q RESULT "$OUT/eval_val09_h100.txt"; then
  CK=$(ls /root/outputs/nestor_100m_b05/*/checkpoints/model_final.pt | tail -1); echo "slim failed; using $CK" >> "$OUT/eval_val09_h100.txt"
  python eval_ckpt.py nestor_100m "$CK" /root/data/dclm_val.bin --block 4096 --max-tokens 10000000 >> "$OUT/eval_val09_h100.txt" 2>&1
fi
python eval_ckpt.py nestor_100m "$CK" /root/data/dclm_val10.bin --block 4096 --max-tokens 5000000 > "$OUT/eval_val10_h100.txt" 2>&1
grep -h RESULT "$OUT"/eval_val*_h100.txt
cp /root/export/chunks/part_00 "$OUT/" && sha256sum "$OUT/part_00"
