# hooks\after_G0.ps1 -- operator wrapper (2026-11-01, user request: work up to G0/G0-5, then pause).
# Runs the G0 gate hook (hashes, codeparrot dedup, gate pass 1, G0-7 margin probes, gate pass 2) unchanged,
# then terminates the chain process so that reproduction stage B does NOT start. Resume later with a new chain script.
$hookDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $hookDir 'after_G0_engineer.ps1')
$rc = $LASTEXITCODE
"[chain4] $(Get-Date -Format s) G0 gate hook finished (exit $rc); chain PAUSED by user request before stage B" | Out-File -Append logs\queue.log
"paused after G0 hook at $(Get-Date -Format s); gate hook exit $rc" | Out-File -Encoding utf8 PAUSED_AFTER_G0.flag
[Environment]::Exit(0)
