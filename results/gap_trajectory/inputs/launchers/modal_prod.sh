#!/bin/bash
# Nestor-100M pretraining on 1x H100, identical recipe to the local queue (1907 steps x 262,144 tokens,
# warmup 153, seed 42, dclm_train16.bin verified by sha256), FlashAttention varlen paths on.
set -u
WORK=$(pwd); OUT="$WORK/out"; mkdir -p "$OUT"
T0=$(date +%s); ts(){ echo "[prod $(date +%H:%M:%S) +$(( $(date +%s)-T0 ))s] $*"; }
cd /root/nestor
export TORCHINDUCTOR_CACHE_DIR=/root/cache/inductor TRITON_CACHE_DIR=/root/cache/triton TIKTOKEN_CACHE_DIR=/root/cache/tiktoken
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DCLM_DATA_PATH=/root/data/dclm_train16.bin MEM_FRAC=0.95 DCLM_NUM_WORKERS=2 NESTOR_OUTPUT_ROOT=/root/outputs
export DCLM_LOG_FREQ=10 DCLM_CKPT_FREQ=500 DCLM_RUN_SUFFIX=b05 NESTOR_ACTIVATION_CHECKPOINT=0
export NESTOR_GATHERED_VARLEN=1 NESTOR_ATTN_PACKED_VARLEN=1 NESTOR_TRITON_PACKED_ATTN=1
ts "train"
python run_exp.py nestor_100m --steps 1907 --warmup 153 --micro 8 --accum 8 > "$OUT/pretrain_nestor_100m_b05_h100.log" 2>&1
echo "exit=$?" >> "$OUT/pretrain_nestor_100m_b05_h100.log"
ts "export model-only checkpoint"
python - <<'EOF' > "$OUT/export_manifest.txt" 2>&1
import glob, hashlib, os, torch
f = sorted(glob.glob('/root/outputs/nestor_100m_b05/*/checkpoints/model_final.pt'))[-1]
sd = torch.load(f, map_location='cpu', weights_only=False)
slim = {'model': sd['model'], 'step': sd.get('step'), 'epoch': sd.get('epoch'), 'git_sha': sd.get('git_sha'),
        'optimizer': {}, 'scheduler': {}}
os.makedirs('/root/export/chunks', exist_ok=True)
dst = '/root/export/nestor_100m_b05_model_final_slim.pt'
torch.save(slim, dst)
h = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
n = sum(t.numel() for t in sd['model'].values())
print('source', f); print('step', sd.get('step')); print('params', n); print('bytes', os.path.getsize(dst)); print('sha256', h)
EOF
cd /root/export && split -b 95M -d -a 2 nestor_100m_b05_model_final_slim.pt chunks/part_ && sha256sum chunks/part_* >> "$OUT/export_manifest.txt"
ls -la /root/outputs/nestor_100m_b05/*/checkpoints/ >> "$OUT/export_manifest.txt"
ts "done"
