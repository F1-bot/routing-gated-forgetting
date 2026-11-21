$Root = '.'
Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if (Test-Path "$Root\msvc_env.ps1") { . "$Root\msvc_env.ps1" }
$env:TRITON_CACHE_DIR = "$Root\tc"
$env:TORCHINDUCTOR_CACHE_DIR = "$Root\ic"
$env:TIKTOKEN_CACHE_DIR = "$Root\data\tiktoken"
foreach ($v in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')) { Set-Item -Path "env:$v" -Value '12' }
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'
$env:NESTOR_VRAM_GUARD = '1'
$env:NESTOR_VRAM_BUDGET_FRAC = '0.90'
$env:MEM_FRAC = '0.85'
$env:TEMP = "$Root\.tmp"
$env:TMP = "$Root\.tmp"
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:WANDB_MODE = 'disabled'
$env:NESTOR_CKPT_PINS = "$Root\ckpt_pins.json"
$py = "$Root\.venv\nestor\Scripts\python.exe"
New-Item -ItemType Directory -Force "$Root\logs\exploratory_c2step1" | Out-Null
$log = "$Root\logs\exploratory_c2step1\runner.log"
"start $(Get-Date -Format o) pid $PID" | Out-File -FilePath $log -Encoding utf8
& $py "$Root\run_queue.py" "$Root\results\exploratory\queue_c2step1.json" *>> $log
"end $(Get-Date -Format o) rc $LASTEXITCODE" | Out-File -FilePath $log -Encoding utf8 -Append
