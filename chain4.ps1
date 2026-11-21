# Chain v4 (replaces v3 after stage A): G0 in the REGISTERED stream layout (g0_probe v2, pinned checkpoints) ->
#   G0-5 local evals (pinned checkpoints) -> [hook after_G0: hashes, codeparrot dedup, gate pass 1, G0-7 margin probes, gate pass 2]
#   -> stage B -> [hook after_B] -> stage C -> [hook after_C].   Hooks are hooks\<name>.ps1, read only when reached.
$ws = (Get-Location).Path
. .\msvc_env.ps1
$env:PYTHONSAFEPATH = $null
$env:PYTHONPATH = $ws
$py = ".\.venv\nestor\Scripts\python.exe"
"[chain4] $(Get-Date -Format s) chain v4 started (stage A done; G0 registered layout)" | Out-File -Append logs\queue.log
function Invoke-Hook([string]$name) {
  $h = Join-Path $ws "hooks\$name.ps1"
  if (Test-Path $h) {
    "[chain4] $(Get-Date -Format s) hook $name start" | Out-File -Append logs\queue.log
    & $h
    "[chain4] $(Get-Date -Format s) hook $name end ($LASTEXITCODE)" | Out-File -Append logs\queue.log
  }
}
foreach ($c in @("g0_moe_b1", "g0_nestor", "g0_moe_b2")) {
  & $py g0_probe.py run --config "$c.json" *> "logs\$c.log"
  "[chain4] $(Get-Date -Format s) $c exited ($LASTEXITCODE)" | Out-File -Append logs\queue.log
}
$nck = "outputs\nestor_100m_b05\20261031_213045__nestor_100m_b05_train_h100\checkpoints\model_final.pt"
$mck = "outputs\moe_100m_b05s43\20261101_001918__moe_100m_b05s43_train_h100\checkpoints\model_final.pt"
& $py eval_ckpt.py moe_100m $mck data/dclm/dclm_val.bin --block 4096 --max-tokens 10000000 *> logs\g0_xplat_local_moe_b2_mean.log
& $py eval_windows.py run nestor_100m $nck --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_nestor *> logs\g0_xplat_local_nestor.log
& $py eval_windows.py run moe_100m $mck --set "dclm09=data/dclm/dclm_val.bin:2048000" --select even --out g0\xplat\local_moe_b2 *> logs\g0_xplat_local_moe_b2.log
"[chain4] $(Get-Date -Format s) G0-5 local evals done" | Out-File -Append logs\queue.log
Invoke-Hook "after_G0"
& $py run_continual.py B *> logs\continual_stdout_B.log
"[chain4] $(Get-Date -Format s) stage B exited ($LASTEXITCODE)" | Out-File -Append logs\queue.log
Invoke-Hook "after_B"
& $py run_continual.py C *> logs\continual_stdout_C.log
"[chain4] $(Get-Date -Format s) stage C exited ($LASTEXITCODE)" | Out-File -Append logs\queue.log
Invoke-Hook "after_C"
"[chain4] $(Get-Date -Format s) chain v4 finished" | Out-File -Append logs\queue.log
