# g0_b3.ps1 -- G0 of the third MoE base (the design and sec. 4 item 4). Windows PowerShell 5.1, ASCII.
# Books cell only: registered routing streams + probes (g0_probe.py), the G0-5 cross-platform check (local eval_ckpt
# mean CE on the first 10,004,085 val09 tokens against the Modal H100 value), G0-1 (books cell), G0-7 margin probes
# (g0_margin.py) and the gate record g0\gate_record_b3.json. GPU; run after the Modal training has landed.
# Before the first run:
#   1. python pins_check.py pin --pins ckpt_pins.json moe_b3 outputs\moe_100m_b15s44\<run>\checkpoints\model_final.pt
#   2. python pins_check.py pin-config --pins ckpt_pins.json --name moe_b3 --config g0_moe_b3.json
#   3. add "h100_ce": <Modal H100 mean CE, first 10,004,085 val09 tokens> to g0_5.bases.moe_b3 in g0\g0_gate_config_b3.json
#   (all three change the code/config set: re-install afterwards with chain6b.ps1 -Reinstall, recorded as an amendment)
# Resumable: g0_probe reuses finished routing files; every step writes its own log under logs\.
# Exit codes: 0 done; 2 stream preflight failed; 3 checkpoint not pinned; 4 H100 reference missing; other = step code.
param([string]$Root = '.', [string]$Python = '')
$ErrorActionPreference = 'Continue'
Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if (Test-Path "$Root\msvc_env.ps1") { . "$Root\msvc_env.ps1" }
$env:TRITON_CACHE_DIR = "$Root\tc"; $env:TORCHINDUCTOR_CACHE_DIR = "$Root\ic"; $env:TIKTOKEN_CACHE_DIR = "$Root\data\tiktoken"
foreach ($v in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS')) { Set-Item -Path "env:$v" -Value '12' }
$env:TORCHINDUCTOR_COMPILE_THREADS = '8'
$env:NESTOR_VRAM_GUARD = '1'; $env:NESTOR_VRAM_BUDGET_FRAC = '0.90'; $env:MEM_FRAC = '0.85'
$env:TEMP = "$Root\.tmp"; $env:TMP = "$Root\.tmp"
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONUTF8 = '1'; $env:WANDB_MODE = 'disabled'
if ($Python) { $py = $Python } else { $py = "$Root\.venv\nestor\Scripts\python.exe" }
$log = "$Root\logs\g0_b3.log"
function Write-Log([string]$m) { $l = "[g0_b3] $(Get-Date -Format s) $m"; $l | Out-File -Append -Encoding utf8 $log; Write-Host $l }
function Invoke-Step([string]$name, [string[]]$argv) {
    "##### $(Get-Date -Format s) $($argv -join ' ')" | Out-File -Append -Encoding utf8 "$Root\logs\g0_b3_$name.log"
    & $py @argv 2>&1 | Out-File -Append -Encoding utf8 "$Root\logs\g0_b3_$name.log"
    $rc = $LASTEXITCODE
    Write-Log "$name exit $rc"
    if ($rc -ne 0) { exit $rc }
}
$cfg = Get-Content "$Root\g0_moe_b3.json" -Raw -Encoding UTF8 | ConvertFrom-Json
if ($cfg.checkpoint -like 'PIN_ME*' -or -not (Test-Path (Join-Path $Root $cfg.checkpoint))) {
    Write-Log "checkpoint not pinned in g0_moe_b3.json ($($cfg.checkpoint))"; exit 3
}
$ck = Join-Path $Root $cfg.checkpoint
$gcfg = "$Root\g0\g0_gate_config_b3.json"
$g5 = (Get-Content $gcfg -Raw -Encoding UTF8 | ConvertFrom-Json).g0_5.bases.moe_b3
if ($null -eq $g5.h100_ce) { Write-Log 'h100_ce missing in g0\g0_gate_config_b3.json (g0_5.bases.moe_b3)'; exit 4 }
Write-Log "start: checkpoint $ck; H100 reference $($g5.h100_ce)"
& $py "$Root\g0_probe.py" streams --config "$Root\g0_moe_b3.json" 2>&1 | Out-File -Append -Encoding utf8 "$Root\logs\g0_b3_streams.log"
if ($LASTEXITCODE -ne 0) { Write-Log 'stream preflight failed (logs\g0_b3_streams.log)'; exit 2 }
Invoke-Step 'probe' @("$Root\g0_probe.py", 'run', '--config', "$Root\g0_moe_b3.json")
if (-not (Test-Path "$Root\logs\g0_xplat_local_moe_b3_mean.log")) {
    & $py "$Root\eval_ckpt.py" moe_100m $ck data/dclm/dclm_val.bin --block 4096 --max-tokens 10000000 2>&1 |
        Out-File -Encoding utf8 "$Root\logs\g0_xplat_local_moe_b3_mean.log"
    Write-Log "G0-5 local mean CE exit $LASTEXITCODE"
}
if (-not (Test-Path "$Root\g0\xplat\local_moe_b3")) {
    Invoke-Step 'xplat_windows' @("$Root\eval_windows.py", 'run', 'moe_100m', $ck, '--set', 'dclm09=data/dclm/dclm_val.bin:2048000',
                                  '--select', 'even', '--out', "$Root\g0\xplat\local_moe_b3")
}
Invoke-Step 'gate1' @("$Root\g0_gate.py", '--config', $gcfg, '--out', "$Root\g0\gate_record_b3.json")
Invoke-Step 'margin_plan' @("$Root\g0_margin.py", 'plan', '--g0-dir', "$Root\g0\moe_b3", '--gate-record', "$Root\g0\gate_record_b3.json", '--base', 'moe_b3')
Invoke-Step 'margin_run' @("$Root\g0_margin.py", 'run', '--g0-dir', "$Root\g0\moe_b3", '--checkpoint', $ck, '--experiment', 'moe_100m',
                           '--grad-path', 'books=data/domain/pg19_train.bin')
Invoke-Step 'gate2' @("$Root\g0_gate.py", '--config', $gcfg, '--out', "$Root\g0\gate_record_b3.json")
Write-Log 'done: g0\gate_record_b3.json'
exit 0
