#!/bin/bash
# Third MoE-100M base on 1x H100: DCLM_SEED=44 (new init AND new RandomSampler data order), suffix b15s44,
# 5,722 optimizer steps x 262,144 tokens (micro 8 x accum 8 x seq 4096) = 1.50B tokens, warm-up 458 steps
# (= round(0.08 x 5,722): the 8 % rule that gave 153 for the template's 1,907 steps). Every other setting as moe2.sh.
# Data: sha256-verified local copies of the bins on the Modal volume (/vol/data).
# Checkpoints: the trainer saves model_<step>.pt every 1,000 steps on local disk; guard_b15s44.py copies each one
# atomically to /vol/outputs and mirrors the train log to /vol/logs. A fresh container restores the newest step
# checkpoint from the volume, and the authors' auto-resume continues the run (optimizer/scheduler/step restored,
# dataloader fast-forwarded). Budget guard: training stops at a checkpoint boundary if the projected end of
# training + eval + export passes STOP_BY (= job start + STOP_AFTER_S, both set at submit).
# After training: fp32 held-out eval (val09 first 10M tokens, val10 5M, block 4096), model-only export split into
# 95 MB chunks on the volume (/vol/export/b15s44) for harvest from a CPU container.
set -u
WORK=$(pwd); OUT="$WORK/out"; mkdir -p "$OUT"
SUF=b15s44; EXP=moe_100m; RUN=${EXP}_${SUF}; STEPS=5722; WARMUP=458
STOP_BY=${STOP_BY:-0}; STOP_AFTER_S=${STOP_AFTER_S:-0}; RESERVE_S=${RESERVE_S:-260}
VOL=/vol; mkdir -p $VOL/logs $VOL/outputs $VOL/export
LEG=pretrain_${RUN}_h100_leg$(date -u +%Y%m%d_%H%M%S)
T0=$(date +%s); [ "$STOP_AFTER_S" != "0" ] && STOP_BY=$(( T0 + STOP_AFTER_S )); ts(){ echo "[moe3 $(date -u +%H:%M:%S) +$(( $(date +%s)-T0 ))s] $*" | tee -a "$VOL/logs/driver_${LEG}.txt"; }
ts "start leg $LEG; STOP_BY=$STOP_BY RESERVE_S=$RESERVE_S"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
echo "nproc $(nproc)"; free -g | head -2; df -h /root | tail -1
mkdir -p /root/nestor/repo /root/nestor/shim/nestor/data /root/nestor/logs /root/cache /root/data
# data copy from the volume runs while pip installs (both are needed before training; neither changes the other)
( for f in dclm_train16.bin dclm_val.bin dclm_val10.bin; do
    [ -f /root/data/$f ] || { cp $VOL/data/$f /root/data/$f.tmp && mv /root/data/$f.tmp /root/data/$f; }
    cp $VOL/data/$f.meta.json /root/data/$f.meta.json
  done
  sha256sum /root/data/dclm_train16.bin /root/data/dclm_val.bin /root/data/dclm_val10.bin > "$OUT/sha256_modal.txt" ) &
DPID=$!
python -c "import torch" 2>/dev/null || {
  pip install -q --no-cache-dir torch==2.14.1 > "$OUT/pip_torch.txt" 2>&1
  pip install -q --no-cache-dir numpy tiktoken jaxtyping zstandard python-dotenv wandb huggingface_hub > "$OUT/pip_rest.txt" 2>&1; }
pip freeze > "$OUT/pip_freeze.txt" 2>&1
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name())" | tee "$OUT/torch_version.txt"
[ -d /root/nestor/repo/nestor ] || tar xzf "$WORK/nestor_repo.tar.gz" -C /root/nestor/repo
cp "$WORK/compat.py" "$WORK/run_exp.py" "$WORK/tok_files.py" "$WORK/eval_ckpt.py" /root/nestor/
cp "$WORK/token_documents.py" /root/nestor/shim/nestor/data/
touch /root/nestor/shim/nestor/__init__.py /root/nestor/shim/nestor/data/__init__.py
cd /root/nestor
export TORCHINDUCTOR_CACHE_DIR=/root/cache/inductor TRITON_CACHE_DIR=/root/cache/triton TIKTOKEN_CACHE_DIR=/root/cache/tiktoken
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
wait $DPID
cat "$OUT/sha256_modal.txt"
cat > /tmp/expected.txt <<'EOF'
455754d1d0098da83fb3bfc33d96e292520751a1e431d7a037e6f1bf0be9cd16  /root/data/dclm_train16.bin
0644eac43505ad8f3ba78c8dd8580862e7b69dcbf24080dd3e8082692186ed04  /root/data/dclm_val.bin
9bdfce3d6555fb083e7c01d92bdc23ca48454cdce296606179a76de98c0bd14e  /root/data/dclm_val10.bin
EOF
if ! sha256sum -c /tmp/expected.txt > "$OUT/sha256_check.txt" 2>&1; then ts "DATA HASH MISMATCH - abort"; cat "$OUT/sha256_check.txt"; exit 3; fi
ts "data verified"
# the sha256 pass read every bin once: drop those pages (and the freshly written site-packages) from the page cache
python - <<'PYEOF'
import os, site
paths = [os.path.join(d, f) for d in ("/root/data", "/vol/data") for f in os.listdir(d)]
for root in site.getsitepackages():
    for dp, _, fs in os.walk(root):
        paths += [os.path.join(dp, f) for f in fs]
n = 0
for p in paths:
    try:
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
    except OSError:
        pass
print("page cache dropped for", n, "files")
PYEOF
# resume: restore the newest step checkpoint from the volume when this container has none; quarantine unreadable ones
python - <<'PYEOF' 2>&1 | tee -a "$OUT/resume.txt"
import glob, os, re, shutil
RUN = "moe_100m_b15s44"
step = lambda p: int(re.search(r"model_(\d{8})\.pt$", p).group(1))
loc = sorted(glob.glob(f"/root/outputs/{RUN}/*/checkpoints/model_[0-9]*.pt"), key=step)
vol = sorted(glob.glob(f"/vol/outputs/{RUN}/*/checkpoints/model_[0-9]*.pt"), key=step)
print("local:", [os.path.relpath(p, "/root/outputs") for p in loc])
print("volume:", [os.path.relpath(p, "/vol/outputs") for p in vol])
if not loc and vol:
    src = vol[-1]; dst = src.replace("/vol/outputs", "/root/outputs", 1)
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print("restored", src, "->", dst); loc = [dst]
if not loc:
    print("fresh start (no checkpoints)")
else:
    import torch
    for p in reversed(loc):
        try:
            sd = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
            assert all(k in sd for k in ("model", "optimizer", "scheduler", "step")), "incomplete checkpoint"
            print("resume from", p, "step", sd["step"]); break
        except Exception as e:  # truncated by a kill mid-save
            print("unreadable, quarantined:", p, repr(e)[:200]); os.replace(p, p + ".corrupt")
PYEOF
ts "train $RUN (DCLM_SEED=44): $STEPS steps x 262,144 tok, warmup $WARMUP, ckpt every 1000 -> volume"
export DCLM_DATA_PATH=/root/data/dclm_train16.bin MEM_FRAC=0.95 DCLM_NUM_WORKERS=2 NESTOR_OUTPUT_ROOT=/root/outputs
export DCLM_LOG_FREQ=10 DCLM_CKPT_FREQ=1000 DCLM_RUN_SUFFIX=$SUF DCLM_SEED=44 NESTOR_ACTIVATION_CHECKPOINT=0
# infrastructure only: at most 4 inductor compile workers (bounds host memory; kernels are the same)
export TORCHINDUCTOR_COMPILE_THREADS=4
LOG="$OUT/pretrain_${RUN}_h100.log"
python run_exp.py $EXP --steps $STEPS --warmup $WARMUP --micro 8 --accum 8 >> "$LOG" 2>&1 &
TPID=$!
python "$WORK/guard_b15s44.py" --log "$LOG" --pid $TPID --steps $STEPS --stop-by $STOP_BY --reserve $RESERVE_S --leg $LEG --dropcache /root/data/dclm_train16.bin > "$OUT/guard_${LEG}.txt" 2>&1 &
GPID=$!
wait $TPID; RC=$?
echo "exit=$RC" >> "$LOG"
wait $GPID
cp "$LOG" "$VOL/logs/${LEG}.log"; cp "$OUT/guard_${LEG}.txt" "$VOL/logs/"
ts "trainer exit $RC"
CKF=$(ls /root/outputs/$RUN/*/checkpoints/model_final.pt 2>/dev/null | tail -1)
if [ "$RC" != "0" ] || [ -z "$CKF" ] || ! grep -q "Training complete" "$LOG"; then
  ts "training did not finish (rc=$RC); resumable step checkpoints are on the volume"
  ls -la $VOL/outputs/$RUN/*/checkpoints/ > "$OUT/vol_ckpts.txt" 2>&1; sync; sleep 20; exit 4
fi
RD=$(dirname "$(dirname "$CKF")"); REL=${CKF#/root/outputs/}
echo "$RD" > "$OUT/run_dir.txt"; cp "$RD/config.json" "$OUT/run_config.json" 2>/dev/null
( mkdir -p "$(dirname "$VOL/outputs/$REL")" && cp "$CKF" "$VOL/outputs/$REL.tmp" && mv "$VOL/outputs/$REL.tmp" "$VOL/outputs/$REL" ) &
CPID=$!
ts "eval fp32 val09 10M + val10 5M (two concurrent processes; same protocol) + export"
python eval_ckpt.py moe_100m "$CKF" /root/data/dclm_val.bin --block 4096 --max-tokens 10000000 > "$OUT/eval_val09_h100.txt" 2>&1 &
E9=$!
python eval_ckpt.py moe_100m "$CKF" /root/data/dclm_val10.bin --block 4096 --max-tokens 5000000 > "$OUT/eval_val10_h100.txt" 2>&1 &
E10=$!
wait $E10
ts "export (val09 eval still running)"
python - "$CKF" <<'PYEOF' > "$OUT/export_manifest.txt" 2>&1
import hashlib, os, sys, torch
f = sys.argv[1]
sd = torch.load(f, map_location='cpu', weights_only=False)
slim = {'model': sd['model'], 'step': sd.get('step'), 'epoch': sd.get('epoch'), 'git_sha': sd.get('git_sha'), 'optimizer': {}, 'scheduler': {}}
os.makedirs('/root/export/chunks', exist_ok=True)
dst = '/root/export/moe_100m_b15s44_model_final_slim.pt'
torch.save(slim, dst)
print('source', f); print('step', sd.get('step')); print('params', sum(t.numel() for t in sd['model'].values()))
print('bytes', os.path.getsize(dst)); print('sha256', hashlib.sha256(open(dst, 'rb').read()).hexdigest())
PYEOF
cd /root/export && split -b 95M -d -a 2 moe_100m_b15s44_model_final_slim.pt chunks/part_ && sha256sum chunks/part_* >> "$OUT/export_manifest.txt"
mkdir -p $VOL/export/b15s44/chunks && cp /root/export/chunks/part_* $VOL/export/b15s44/chunks/ && cp "$OUT/export_manifest.txt" $VOL/export/b15s44/
(cd $VOL/export/b15s44 && sha256sum chunks/part_*) > "$OUT/export_chunks_volume_sha.txt"
cd /root/nestor
wait $E9
grep -h RESULT "$OUT"/eval_val*_h100.txt | tee "$OUT/eval_results_h100.txt"
wait $CPID
ls -la "$RD/checkpoints" $VOL/outputs/$RUN/*/checkpoints $VOL/export/b15s44/chunks > "$OUT/listing.txt" 2>&1
cp "$OUT"/eval_*.txt "$OUT"/export_*.txt "$OUT/run_dir.txt" "$OUT/torch_version.txt" "$OUT/gpu.txt" "$VOL/logs/" 2>/dev/null
sync; sleep 15
ts done
