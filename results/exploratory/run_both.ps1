& powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'results\exploratory\run_randx.ps1'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'results\exploratory\run_c2step1_nowait.ps1'
"done both $(Get-Date -Format o)"
