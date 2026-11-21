#!/bin/bash
# Second MoE-100M base (DCLM_SEED=43: new init AND new RandomSampler data order), same recipe as the b05 runs,
# on 1x H100; exact data verified by sha256; fp32 held-out eval; model-only export split into 95 MB chunks.
set -u
WORK=$(pwd); OUT="$WORK/out"; mkdir -p "$OUT"
T0=$(date +%s); ts(){ echo "[moe2 $(date +%H:%M:%S) +$(( $(date +%s)-T0 ))s] $*"; }
ts start; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
pip install -q --no-cache-dir torch==2.14.1 > "$OUT/pip_torch.txt" 2>&1
pip install -q --no-cache-dir numpy tiktoken jaxtyping zstandard python-dotenv wandb huggingface_hub > "$OUT/pip_rest.txt" 2>&1
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())" | tee "$OUT/torch_version.txt"
mkdir -p /root/nestor/repo /root/nestor/shim/nestor/data /root/nestor/logs /root/cache
tar xzf "$WORK/nestor_repo.tar.gz" -C /root/nestor/repo
cp "$WORK/compat.py" "$WORK/run_exp.py" "$WORK/tok_files.py" "$WORK/eval_ckpt.py" /root/nestor/
cp "$WORK/token_documents.py" /root/nestor/shim/nestor/data/
touch /root/nestor/shim/nestor/__init__.py /root/nestor/shim/nestor/data/__init__.py
cd /root/nestor
export TORCHINDUCTOR_CACHE_DIR=/root/cache/inductor TRITON_CACHE_DIR=/root/cache/triton TIKTOKEN_CACHE_DIR=/root/cache/tiktoken
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
ts "download + tokenize"
python "$WORK/dl_dclm2.py" /root/data/raw > "$OUT/dl.txt" 2>&1; tail -1 "$OUT/dl.txt"
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 1 --out /root/data/dclm_train16.bin --procs 16 --last-file 15 > "$OUT/tok_train.txt" 2>&1
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 9 --out /root/data/dclm_val.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val.txt" 2>&1
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 10 --out /root/data/dclm_val10.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val10.txt" 2>&1
rm -rf /root/data/*.parts
sha256sum /root/data/dclm_train16.bin /root/data/dclm_val.bin /root/data/dclm_val10.bin | tee "$OUT/sha256_modal.txt"
cat > /tmp/expected.txt <<'EOF'
455754d1d0098da83fb3bfc33d96e292520751a1e431d7a037e6f1bf0be9cd16  /root/data/dclm_train16.bin
0644eac43505ad8f3ba78c8dd8580862e7b69dcbf24080dd3e8082692186ed04  /root/data/dclm_val.bin
9bdfce3d6555fb083e7c01d92bdc23ca48454cdce296606179a76de98c0bd14e  /root/data/dclm_val10.bin
EOF
if ! sha256sum -c /tmp/expected.txt > "$OUT/sha256_check.txt" 2>&1; then ts "DATA HASH MISMATCH - abort"; cat "$OUT/sha256_check.txt"; exit 3; fi
ts "train moe_100m seed 43"
export DCLM_DATA_PATH=/root/data/dclm_train16.bin MEM_FRAC=0.95 DCLM_NUM_WORKERS=2 NESTOR_OUTPUT_ROOT=/root/outputs
export DCLM_LOG_FREQ=10 DCLM_CKPT_FREQ=500 DCLM_RUN_SUFFIX=b05s43 DCLM_SEED=43 NESTOR_ACTIVATION_CHECKPOINT=0
python run_exp.py moe_100m --steps 1907 --warmup 153 --micro 8 --accum 8 > "$OUT/pretrain_moe_100m_b05s43_h100.log" 2>&1
echo "exit=$?" >> "$OUT/pretrain_moe_100m_b05s43_h100.log"
CKF=$(ls /root/outputs/moe_100m_b05s43/*/checkpoints/model_final.pt | tail -1)
ts "eval fp32 val09 10M + val10 5M"
python eval_ckpt.py moe_100m "$CKF" /root/data/dclm_val.bin --block 4096 --max-tokens 10000000 > "$OUT/eval_val09_h100.txt" 2>&1
python eval_ckpt.py moe_100m "$CKF" /root/data/dclm_val10.bin --block 4096 --max-tokens 5000000 > "$OUT/eval_val10_h100.txt" 2>&1
grep -h RESULT "$OUT"/eval_val*_h100.txt
ts "export"
python - "$CKF" <<'PYEOF' > "$OUT/export_manifest.txt" 2>&1
import hashlib, os, sys, torch
f = sys.argv[1]
sd = torch.load(f, map_location='cpu', weights_only=False)
slim = {'model': sd['model'], 'step': sd.get('step'), 'epoch': sd.get('epoch'), 'git_sha': sd.get('git_sha'), 'optimizer': {}, 'scheduler': {}}
os.makedirs('/root/export/chunks', exist_ok=True)
dst = '/root/export/moe_100m_b05s43_model_final_slim.pt'
torch.save(slim, dst)
print('source', f); print('step', sd.get('step')); print('params', sum(t.numel() for t in sd['model'].values()))
print('bytes', os.path.getsize(dst)); print('sha256', hashlib.sha256(open(dst, 'rb').read()).hexdigest())
PYEOF
cd /root/export && split -b 95M -d -a 2 moe_100m_b05s43_model_final_slim.pt chunks/part_ && sha256sum chunks/part_* >> "$OUT/export_manifest.txt"
cp /root/export/chunks/part_00 "$OUT/"
ts done
