#!/bin/bash
# Calibration on 1x H100: software, exact data (sha256 vs local), authors' GPU tests,
# platform equivalence (Nestor 12-step E2E; MoE first 20 steps) and production throughput.
set -u
WORK=$(pwd); mkdir -p "$WORK/out"; OUT="$WORK/out"
T0=$(date +%s)
ts(){ echo "[calib $(date +%H:%M:%S) +$(( $(date +%s)-T0 ))s] $*"; }
ts start
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; echo "nproc=$(nproc)"; free -g | head -2; python -V; df -h / | tail -1

# ---------- 1. software (persists in the warm container for later jobs on this handle)
if [ ! -f /root/.env_ok ]; then
  ts "pip torch==2.14.1 (PyPI default wheel)"
  pip install -q --no-cache-dir torch==2.14.1 > "$OUT/pip_torch.txt" 2>&1 || tail -5 "$OUT/pip_torch.txt"
  for idx in cu130 cu129 cu128 cu126; do
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && break
    ts "CUDA unusable with current wheel -> trying index $idx"
    pip install -q --no-cache-dir --force-reinstall torch==2.14.1 --index-url "https://download.pytorch.org/whl/$idx" >> "$OUT/pip_torch.txt" 2>&1
  done
  pip install -q --no-cache-dir numpy tiktoken jaxtyping zstandard python-dotenv wandb huggingface_hub pytest > "$OUT/pip_rest.txt" 2>&1 || tail -5 "$OUT/pip_rest.txt"
  python -c "import torch; assert torch.cuda.is_available()" && touch /root/.env_ok
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name())" | tee "$OUT/torch_version.txt"
ts "software ready"

# ---------- 2. code: the authors' commit c797821 (git archive) + our launcher/shims
mkdir -p /root/nestor/repo /root/nestor/shim/nestor/data /root/nestor/logs /root/cache
tar xzf "$WORK/nestor_repo.tar.gz" -C /root/nestor/repo
cp "$WORK/compat.py" "$WORK/run_exp.py" "$WORK/tok_files.py" "$WORK/parse_logs.py" "$WORK/local_token_hashes.json" /root/nestor/
cp "$WORK/token_documents.py" /root/nestor/shim/nestor/data/
touch /root/nestor/shim/nestor/__init__.py /root/nestor/shim/nestor/data/__init__.py
cp "$WORK/conftest.py" /root/nestor/repo/conftest.py
cd /root/nestor
export TORCHINDUCTOR_CACHE_DIR=/root/cache/inductor TRITON_CACHE_DIR=/root/cache/triton TIKTOKEN_CACHE_DIR=/root/cache/tiktoken
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- 3. authors' GPU tests (background, while data is prepared)
( cd /root/nestor/repo && PYTHONPATH=/root/nestor/repo timeout 1500 python -m pytest -q -p no:cacheprovider \
    tests/test_gathered_causal_flash.py tests/test_attention_dispatch.py tests/test_grouped_mm_fused_gate.py \
    tests/test_capacity_bucketing.py tests/test_fused_head_ce.py tests/test_column_slot_updates.py \
    tests/test_batched_router.py tests/test_attention_head_split.py tests/test_chunked_ce.py \
    > "$OUT/pytest_gpu.txt" 2>&1; echo "pytest exit=$?" >> "$OUT/pytest_gpu.txt" ) &
PYTEST_PID=$!

# ---------- 4. exact data: same files, same encoder, same order; sha256 compared with the local bins
if [ ! -f /root/data/dclm_train16.bin ]; then
  python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && ts "tiktoken cache warm"
  ts "download DCLM shards from HF"
  python "$WORK/dl_dclm.py" /root/data/raw > "$OUT/dl.txt" 2>&1; tail -2 "$OUT/dl.txt"
  ts "tokenize train16 (16 files)"
  PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 1 \
      --out /root/data/dclm_train16.bin --procs 16 --last-file 15 > "$OUT/tok_train.txt" 2>&1; tail -2 "$OUT/tok_train.txt"
  ts "tokenize val (shard 09, 20M target)"
  PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 9 \
      --out /root/data/dclm_val.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val.txt" 2>&1; tail -2 "$OUT/tok_val.txt"
  rm -rf /root/data/dclm_train16.bin.parts /root/data/dclm_val.bin.parts
fi
ts "sha256"
sha256sum /root/data/dclm_train16.bin /root/data/dclm_val.bin | tee "$OUT/sha256_modal.txt"
cp /root/data/*.meta.json "$OUT/" 2>/dev/null

wait_for_step(){ # log step timeout pid -> stop the run once "Step <step> |" is logged
  local log=$1 step=$2 to=$3 pid=$4 t=0
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "Step $step |" "$log"; then sleep 1; kill -TERM "$pid"; break; fi
    sleep 5; t=$((t+5)); if [ $t -ge "$to" ]; then echo "TIMEOUT waiting for step $step" >> "$log"; kill -TERM "$pid"; break; fi
  done
  wait "$pid" 2>/dev/null
}
export DCLM_DATA_PATH=/root/data/dclm_train16.bin MEM_FRAC=0.95 DCLM_NUM_WORKERS=2 DCLM_CKPT_FREQ=100000 NESTOR_OUTPUT_ROOT=/root/outputs_calib
FA=(NESTOR_GATHERED_VARLEN=1 NESTOR_ATTN_PACKED_VARLEN=1 NESTOR_TRITON_PACKED_ATTN=1)

# ---------- 5a. Nestor E2E, identical to the local 12-step test (steps 12, warmup 2, micro 4 x accum 16), FA paths on
ts "Nestor E2E 12 steps"
env "${FA[@]}" DCLM_LOG_FREQ=1 DCLM_RUN_SUFFIX=e2e NESTOR_ACTIVATION_CHECKPOINT=0 \
  timeout 1800 python run_exp.py nestor_100m --steps 12 --warmup 2 --micro 4 --accum 16 > "$OUT/e2e_nestor_h100.log" 2>&1
echo "exit=$?" >> "$OUT/e2e_nestor_h100.log"; rm -rf /root/outputs_calib/nestor_100m_e2e

# ---------- 5b. production throughput: 1907-step schedule, micro 8 x accum 8, no activation checkpointing
ts "Nestor throughput micro 8"
env "${FA[@]}" DCLM_LOG_FREQ=10 DCLM_RUN_SUFFIX=tp8 NESTOR_ACTIVATION_CHECKPOINT=0 \
  python run_exp.py nestor_100m --steps 1907 --warmup 153 --micro 8 --accum 8 > "$OUT/tp_nestor_m8.log" 2>&1 &
wait_for_step "$OUT/tp_nestor_m8.log" 40 1800 $!

# ---------- 5c. MoE platform check: the local queue's exact config (micro 2 x accum 32), first 20 steps
ts "MoE platform check"
env DCLM_LOG_FREQ=10 DCLM_RUN_SUFFIX=pc NESTOR_ACTIVATION_CHECKPOINT=0 \
  python run_exp.py moe_100m --steps 1907 --warmup 153 --micro 2 --accum 32 > "$OUT/pc_moe_m2.log" 2>&1 &
wait_for_step "$OUT/pc_moe_m2.log" 20 1200 $!
rm -rf /root/outputs_calib

# ---------- 6. harvest-speed probe, tests, summary
head -c 67108864 /dev/urandom > "$OUT/harvest_probe.bin"
ts "waiting for pytest"; wait $PYTEST_PID; tail -5 "$OUT/pytest_gpu.txt"
cp /root/nestor/local_token_hashes.json "$WORK/" 2>/dev/null
cd "$WORK" && python /root/nestor/parse_logs.py out
ts "done"
