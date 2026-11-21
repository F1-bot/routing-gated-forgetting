#!/bin/bash
# DCLM token bins for the third MoE-100M base (b15s44). Rebuilt on a Modal CPU container exactly as moe2.sh did
# (same pip set, dl_dclm2.py download, tok_files.py driver commands with the same arguments, sha256 check against
# the calibration record), then copied atomically into the Modal volume mounted at /vol (data/), with the
# <bin>.meta.json sidecars that DclmTokenDataset reads. Nothing is copied if a hash differs.
set -u
WORK=$(pwd); OUT="$WORK/out"; mkdir -p "$OUT"
T0=$(date +%s); ts(){ echo "[data $(date -u +%H:%M:%S) +$(( $(date +%s)-T0 ))s] $*"; }
ts start; echo "nproc $(nproc)"; free -g | head -2; df -h /root /vol | tail -2
pip install -q --no-cache-dir numpy tiktoken jaxtyping zstandard python-dotenv wandb huggingface_hub > "$OUT/pip_rest.txt" 2>&1
pip freeze > "$OUT/pip_freeze_data.txt" 2>&1
mkdir -p /root/nestor/shim/nestor/data /root/nestor/logs /root/cache /root/data
cp "$WORK/tok_files.py" /root/nestor/
cp "$WORK/token_documents.py" /root/nestor/shim/nestor/data/
touch /root/nestor/shim/nestor/__init__.py /root/nestor/shim/nestor/data/__init__.py
cd /root/nestor
export TIKTOKEN_CACHE_DIR=/root/cache/tiktoken
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
ts "download"
python "$WORK/dl_dclm2.py" /root/data/raw > "$OUT/dl.txt" 2>&1; tail -1 "$OUT/dl.txt"
ts "tokenize train16 (16 procs)"
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 1 --out /root/data/dclm_train16.bin --procs 16 --last-file 15 > "$OUT/tok_train.txt" 2>&1
ts "tokenize val09 / val10"
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 9 --out /root/data/dclm_val.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val.txt" 2>&1
PYTHONPATH=/root/nestor/shim python tok_files.py driver --root /root/data/raw --global-shards 10 --out /root/data/dclm_val10.bin --procs 1 --target-tokens 20000000 > "$OUT/tok_val10.txt" 2>&1
rm -rf /root/data/*.parts
tail -1 "$OUT/tok_train.txt"; tail -1 "$OUT/tok_val.txt"; tail -1 "$OUT/tok_val10.txt"
ts "sha256"
sha256sum /root/data/dclm_train16.bin /root/data/dclm_val.bin /root/data/dclm_val10.bin | tee "$OUT/sha256_modal.txt"
cat > /tmp/expected.txt <<'EOF'
455754d1d0098da83fb3bfc33d96e292520751a1e431d7a037e6f1bf0be9cd16  /root/data/dclm_train16.bin
0644eac43505ad8f3ba78c8dd8580862e7b69dcbf24080dd3e8082692186ed04  /root/data/dclm_val.bin
9bdfce3d6555fb083e7c01d92bdc23ca48454cdce296606179a76de98c0bd14e  /root/data/dclm_val10.bin
EOF
if ! sha256sum -c /tmp/expected.txt > "$OUT/sha256_check.txt" 2>&1; then ts "DATA HASH MISMATCH - abort, nothing copied"; cat "$OUT/sha256_check.txt"; exit 3; fi
cat "$OUT/sha256_check.txt"
cp /root/data/*.meta.json "$OUT/"
ts "copy to volume"
mkdir -p /vol/data
for f in dclm_train16.bin dclm_val.bin dclm_val10.bin; do
  cp /root/data/$f.meta.json /vol/data/$f.meta.json.tmp && mv /vol/data/$f.meta.json.tmp /vol/data/$f.meta.json
  cp /root/data/$f /vol/data/$f.tmp && mv /vol/data/$f.tmp /vol/data/$f
done
sync
ts "re-hash volume copies"
(cd /vol/data && sha256sum dclm_train16.bin dclm_val.bin dclm_val10.bin) | tee "$OUT/sha256_volume.txt"
python - <<'PYEOF' | tee "$OUT/token_counts.txt"
import json
for f in ["dclm_train16.bin", "dclm_val.bin", "dclm_val10.bin"]:
    m = json.load(open("/vol/data/" + f + ".meta.json"))
    print(f, "num_tokens", m.get("num_tokens"), "num_docs", m.get("num_docs"), "dtype", m.get("token_dtype"))
PYEOF
ls -la /vol/data | tee "$OUT/vol_data_ls.txt"
sleep 20
ts done
