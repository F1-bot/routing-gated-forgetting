$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path; Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
foreach ($v in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) { Set-Item -Path "env:$v" -Value '12' }
$env:TEMP = "$Root\.tmp"; $env:TMP = "$Root\.tmp"
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONUTF8 = '1'; $env:WANDB_MODE = 'disabled'
$env:NESTOR_PROJECT_ROOT = $Root
$py = "$Root\.venv\nestor\Scripts\python.exe"
$out = "$Root\stagec_reeval\reeval.jsonl"
'[pi] ' + (Get-Date -Format s) + ' stage-C chunked-head re-evaluation start (stagec_reeval\eval_ckpt_chunked.py)' | Out-File -Append -Encoding utf8 logs\queue.log
$env:DCLM_BLOCK_SIZE = '8192'; $env:DCLM_ROPE_BASE = '500000'; $env:DCLM_DATA_PATH = "$Root\data\longctx\dclm_longdoc_ge8192_val.bin"
$t0 = Get-Date
& $py "$Root\stagec_reeval\eval_ckpt_chunked.py" dense_100m "$Root\outputs\continual\dense_100m_ext8192_equal_tokens_m2\dense_100m_ext8192_equal_tokens\20261114_130030__dense_100m_ext8192_equal_tokens_train\checkpoints\model_final.pt" "$Root\data\longctx\dclm_longdoc_ge8192_val.bin" --block 8192 --first-per-doc --docs 200 --head-rows 1024 *> "$Root\logs\stagec_reeval_dense8192.log"
$r = Select-String -Path "$Root\logs\stagec_reeval_dense8192.log" -Pattern '^RESULT ' | Select-Object -Last 1
if ($r) { $r.Line.Substring(7) | Out-File -Append -Encoding utf8 $out; Write-Host 'dense8192 ok' ([int]((Get-Date)-$t0).TotalSeconds) 's' } else { Write-Host 'dense8192 FAILED' }
$env:DCLM_BLOCK_SIZE = '8192'; $env:DCLM_ROPE_BASE = '500000'; $env:DCLM_DATA_PATH = "$Root\data\longctx\dclm_longdoc_ge8192_val.bin"
$t0 = Get-Date
& $py "$Root\stagec_reeval\eval_ckpt_chunked.py" moe_100m "$Root\outputs\continual\moe_100m_ext8192_equal_tokens_m1c\moe_100m_ext8192_equal_tokens\20261114_133417__moe_100m_ext8192_equal_tokens_train\checkpoints\model_final.pt" "$Root\data\longctx\dclm_longdoc_ge8192_val.bin" --block 8192 --first-per-doc --docs 200 --head-rows 1024 *> "$Root\logs\stagec_reeval_moe8192.log"
$r = Select-String -Path "$Root\logs\stagec_reeval_moe8192.log" -Pattern '^RESULT ' | Select-Object -Last 1
if ($r) { $r.Line.Substring(7) | Out-File -Append -Encoding utf8 $out; Write-Host 'moe8192 ok' ([int]((Get-Date)-$t0).TotalSeconds) 's' } else { Write-Host 'moe8192 FAILED' }
$env:DCLM_BLOCK_SIZE = '16384'; $env:DCLM_ROPE_BASE = '1000000'; $env:DCLM_DATA_PATH = "$Root\data\longctx\dclm_longdoc_ge16384_val.bin"
$t0 = Get-Date
& $py "$Root\stagec_reeval\eval_ckpt_chunked.py" dense_100m "$Root\outputs\continual\dense_100m_ext16384_equal_tokens_m1\dense_100m_ext16384_equal_tokens\20261114_144308__dense_100m_ext16384_equal_tokens_train\checkpoints\model_final.pt" "$Root\data\longctx\dclm_longdoc_ge16384_val.bin" --block 16384 --first-per-doc --docs 200 --head-rows 1024 *> "$Root\logs\stagec_reeval_dense16384.log"
$r = Select-String -Path "$Root\logs\stagec_reeval_dense16384.log" -Pattern '^RESULT ' | Select-Object -Last 1
if ($r) { $r.Line.Substring(7) | Out-File -Append -Encoding utf8 $out; Write-Host 'dense16384 ok' ([int]((Get-Date)-$t0).TotalSeconds) 's' } else { Write-Host 'dense16384 FAILED' }
$env:DCLM_BLOCK_SIZE = '16384'; $env:DCLM_ROPE_BASE = '1000000'; $env:DCLM_DATA_PATH = "$Root\data\longctx\dclm_longdoc_ge16384_val.bin"
$t0 = Get-Date
& $py "$Root\stagec_reeval\eval_ckpt_chunked.py" moe_100m "$Root\outputs\continual\moe_100m_ext16384_equal_tokens_m1c\moe_100m_ext16384_equal_tokens\20261114_152632__moe_100m_ext16384_equal_tokens_train\checkpoints\model_final.pt" "$Root\data\longctx\dclm_longdoc_ge16384_val.bin" --block 16384 --first-per-doc --docs 200 --head-rows 1024 *> "$Root\logs\stagec_reeval_moe16384.log"
$r = Select-String -Path "$Root\logs\stagec_reeval_moe16384.log" -Pattern '^RESULT ' | Select-Object -Last 1
if ($r) { $r.Line.Substring(7) | Out-File -Append -Encoding utf8 $out; Write-Host 'moe16384 ok' ([int]((Get-Date)-$t0).TotalSeconds) 's' } else { Write-Host 'moe16384 FAILED' }
'[pi] ' + (Get-Date -Format s) + ' stage-C chunked-head re-evaluation end' | Out-File -Append -Encoding utf8 logs\queue.log
