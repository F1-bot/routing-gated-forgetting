# Chain v5 (replaces v4 after the g0_probe performance patch): G0 (registered layout, pinned checkpoints) -> G0-5 local evals
#   -> [hook after_G0 = wrapper: gate hook (dedup, gate pass 1, G0-7, gate pass 2), then PAUSE]. No stage B/C (user: pause after G0).
$ws = (Get-Location).Path
. .\msvc_env.ps1
$env:PYTHONSAFEPATH = $null
$env:PYTHONPATH = $ws
$env:PYTHONUNBUFFERED = "1"
$py = ".\.venv\nestor\Scripts\python.exe"
"[chain5] $(Get-Date -Format s) chain v5 started (G0 with patched g0_probe; pause after the G0 gate)" | Out-File -Append logs\queue.log
foreach ($c in @("g0_moe_b1", "g0_nestor", "g0_moe_b2")) {
  & $py g0_probe.py run --config "$c.json" *> "logs\$c.log"
  "[chain5] $(Get-Date -Format s) $c exited ($LASTEXITCODE)" | Out-File -Append logs\queue.log
}
$nck = "outputs\nestor_100m_b05\20261031_213045__nestor_100m_b05_train_h100\checkpoints\model_final.pt"
$mck = "outputs\moe_100m_b05s43\20261101_001918__moe_100m_b05s43_train_h100\checkpoints\model_final.pt"
& $py eval_ckpt.py moe_100m $mck data/dclm/dclm_val.bin --block 4096 --max-tokens 10000000 *> logs\g0_xplat_local_moe_b2_mean.log
& $py eval_windows.py run nestor_100m $nck --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_nestor *> logs\g0_xplat_local_nestor.log
& $py eval_windows.py run moe_100m $mck --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_moe_b2 *> logs\g0_xplat_local_moe_b2.log
"[chain5] $(Get-Date -Format s) G0-5 local evals done" | Out-File -Append logs\queue.log
$h = Join-Path $ws "hooks\after_G0.ps1"
if (Test-Path $h) { "[chain5] $(Get-Date -Format s) hook after_G0 start" | Out-File -Append logs\queue.log; & $h }
"[chain5] $(Get-Date -Format s) chain v5 finished" | Out-File -Append logs\queue.log
