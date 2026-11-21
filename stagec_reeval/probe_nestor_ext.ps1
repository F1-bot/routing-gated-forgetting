param([string]$MemFrac = '0.93', [int]$StopAfterStep = 3, [int]$MaxMin = 30)
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path; Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if (Test-Path "$Root\msvc_env.ps1") { . "$Root\msvc_env.ps1" }
$env:TRITON_CACHE_DIR = "$Root\tc"; $env:TORCHINDUCTOR_CACHE_DIR = "$Root\ic"; $env:TIKTOKEN_CACHE_DIR = "$Root\data\tiktoken"
foreach ($v in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')) { Set-Item -Path "env:$v" -Value '12' }
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$env:TEMP = "$Root\.tmp"; $env:TMP = "$Root\.tmp"
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONUTF8 = '1'; $env:WANDB_MODE = 'disabled'
$env:NESTOR_CKPT_PINS = "$Root\ckpt_pins.json"
$tag = "mf" + ($MemFrac -replace '0\.', '')
$env:DCLM_BLOCK_SIZE = '8192'; $env:DCLM_ROPE_BASE = '500000'
$env:DCLM_DATA_PATH = "$Root\data\longctx\dclm_longdoc_ge8192_train.bin"
$env:DCLM_INIT_CHECKPOINT = "$Root\outputs\nestor_100m_b05\20261031_213045__nestor_100m_b05_train_h100\checkpoints\model_final.pt"
$env:DCLM_LR_MUON = '0.0025'; $env:DCLM_LR_ADAMW = '7.5e-05'
$env:DCLM_RUN_SUFFIX = "ext8192_probe_$tag"; $env:NESTOR_GATHERED_CAP_BUCKET = '256'
$env:NESTOR_OUTPUT_ROOT = "$Root\outputs\continual_probe\nestor_ext8192_$tag"
$env:MEM_FRAC = $MemFrac; $env:DCLM_LOG_FREQ = '1'; $env:DCLM_CKPT_FREQ = '100000'
$env:NESTOR_ACTIVATION_CHECKPOINT = '1'; $env:NESTOR_ACTIVATION_CHECKPOINT_SPLIT = '1'; $env:NESTOR_ACTIVATION_CHECKPOINT_LEAN = '1'
$log = "$Root\logs\probe_nestor_ext8192_$tag.log"
"[pi] $(Get-Date -Format s) NESTOR stage-C memory probe start (MEM_FRAC $MemFrac, micro 1, ckpt, seq 8192) -> $log" | Out-File -Append -Encoding utf8 logs\queue.log
$py = "$Root\.venv\nestor\Scripts\python.exe"
$p = Start-Process -FilePath $py -ArgumentList @('run_exp.py','nestor_100m','--steps','48','--warmup','4','--micro','1','--accum','128') -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow
$t0 = Get-Date; $why = 'timeout'
while (-not $p.HasExited) {
    Start-Sleep -Seconds 20
    $txt = (Get-Content $log -Raw -ErrorAction SilentlyContinue) + (Get-Content "$log.err" -Raw -ErrorAction SilentlyContinue)
    if ($txt -match 'OutOfMemoryError') { $why = 'OOM'; break }
    if ($txt -match "Step $StopAfterStep \|") { $why = "reached step $StopAfterStep"; break }
    if (((Get-Date) - $t0).TotalMinutes -gt $MaxMin) { break }
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force; Start-Sleep -Seconds 5 }
Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $p.Id } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
"[pi] $(Get-Date -Format s) NESTOR stage-C memory probe end ($why, $([int]((Get-Date)-$t0).TotalMinutes) min)" | Out-File -Append -Encoding utf8 logs\queue.log
Write-Host "probe $tag : $why after $([int]((Get-Date)-$t0).TotalMinutes) min"
