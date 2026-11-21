# Chain v6a (2026-11-13, after the workspace sweep): wait for rebuilt env + data -> stage-A spot check -> G0 re-run (frozen code)
#   -> G0-5 local evals -> G0 gate hook -> decision comparison with the registered record -> [hook after_G0rerun]
#   -> reproduction stage B -> per-window stage-B evals -> stage-B analysis -> [hook after_stageB]   (hooks read when reached)
$P = '.'; Set-Location $P
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'; New-Item -ItemType Directory -Force "$P\.tmp" | Out-Null; $env:TEMP = "$P\.tmp"; $env:TMP = "$P\.tmp"
$env:OMP_NUM_THREADS = '12'; $env:MKL_NUM_THREADS = '12'; $env:TORCHINDUCTOR_COMPILE_THREADS = '8'
$env:NESTOR_VRAM_GUARD = '1'; $env:NESTOR_VRAM_BUDGET_FRAC = '0.90'
$q = 'logs\queue.log'
"[chain6a] $(Get-Date -Format s) started" | Out-File -Append $q
while (-not (Select-String -Path logs\env_check.log -Pattern 'ENV_OK' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep -Seconds 60 }
while (-not (Test-Path data\data_manifest.json)) { Start-Sleep -Seconds 120 }
$cond = '<conda>\envs\nestor\Library\bin'
$py = "$P\.venv\nestor\Scripts\python.exe"
function TorchFree([string[]]$argv) { $s = $env:PATH; $env:PATH = "$cond;$s"; $o = & $py @argv; $env:PATH = $s; return $o }
$ok = TorchFree @('check_manifest.py', 'data\data_manifest.json')
"[chain6a] $(Get-Date -Format s) data manifest: $ok" | Out-File -Append $q
if ($ok -notmatch 'DATA_OK') { "[chain6a] stop: data manifest not clean" | Out-File -Append $q; exit 2 }
. .\msvc_env.ps1
$env:PYTHONSAFEPATH = $null; $env:PYTHONPATH = $P; $env:PYTHONUNBUFFERED = '1'
function Hook([string]$name) { $h = "$P\hooks\$name.ps1"; if (Test-Path $h) { "[chain6a] $(Get-Date -Format s) hook $name" | Out-File -Append $q; & $h } }
& $py eval_ckpt.py moe_100m outputs\moe_100m_b05\20261101_002357__moe_100m_b05_train\checkpoints\model_final.pt data/dclm/dclm_val.bin --block 4096 --max-tokens 10000000 *> logs\spotcheck_stageA_moe_b1.log
$spot = TorchFree @('spotcheck.py', 'logs\spotcheck_stageA_moe_b1.log', '4.628702028932119')
"[chain6a] $(Get-Date -Format s) stage-A spot check: $spot" | Out-File -Append $q
if ($spot -notmatch 'SPOT_OK') { "[chain6a] stop: spot check failed" | Out-File -Append $q; exit 3 }
foreach ($c in @('g0_moe_b1', 'g0_nestor', 'g0_moe_b2')) {
  & $py g0_probe.py run --config "$c.json" *> "logs\rerun_$c.log"
  "[chain6a] $(Get-Date -Format s) $c exited ($LASTEXITCODE)" | Out-File -Append $q
}
& $py eval_ckpt.py moe_100m outputs\moe_100m_b05s43\20261101_001918__moe_100m_b05s43_train_h100\checkpoints\model_final.pt data/dclm/dclm_val.bin --block 4096 --max-tokens 10000000 *> logs\g0_xplat_local_moe_b2_mean.log
& $py eval_windows.py run nestor_100m outputs\nestor_100m_b05\20261031_213045__nestor_100m_b05_train_h100\checkpoints\model_final.pt --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_nestor *> logs\g0_xplat_local_nestor.log
& $py eval_windows.py run moe_100m outputs\moe_100m_b05s43\20261101_001918__moe_100m_b05s43_train_h100\checkpoints\model_final.pt --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_moe_b2 *> logs\g0_xplat_local_moe_b2.log
"[chain6a] $(Get-Date -Format s) G0-5 local evals done; gate hook" | Out-File -Append $q
& .\hooks_after_G0.ps1
"[chain6a] $(Get-Date -Format s) gate hook exit $LASTEXITCODE" | Out-File -Append $q
$cmp = TorchFree @('compare_g0.py', 'g0_original\gate_record.json', 'g0\gate_record.json', 'g0\rerun_comparison.json')
"[chain6a] $(Get-Date -Format s) G0 re-run comparison: $cmp" | Out-File -Append $q
if ($cmp -notmatch '^MATCH') { "[chain6a] stop: G0 decisions differ (see g0\rerun_comparison.json)" | Out-File -Append $q; exit 4 }
Hook 'after_G0rerun'
& $py run_continual.py B *> logs\continual_stdout_B.log
"[chain6a] $(Get-Date -Format s) stage B exited ($LASTEXITCODE)" | Out-File -Append $q
& $py stageB_windows.py *> logs\stageB_windows.log
"[chain6a] $(Get-Date -Format s) stage-B windows exited ($LASTEXITCODE)" | Out-File -Append $q
TorchFree @('analyze_stageB.py') | Out-File -Encoding utf8 logs\analyze_stageB.log
"[chain6a] $(Get-Date -Format s) stage-B analysis exited ($LASTEXITCODE)" | Out-File -Append $q
Hook 'after_stageB'
"[chain6a] $(Get-Date -Format s) chain v6a finished" | Out-File -Append $q
