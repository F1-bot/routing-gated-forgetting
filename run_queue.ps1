# Gated run queue for the local GPU (Windows PowerShell). Sequential, resumable, no multiprocessing pipes.
#   .\run_queue.ps1 -Queue queue_phase3.json            # run (restart = resume; finished jobs are skipped)
#   .\run_queue.ps1 -Queue queue_phase3.json -DryRun    # print every command and the GPU-hour estimate
#   New-Item results\STOP -ItemType File                # stop cleanly before the next job
param(
    [Parameter(Mandatory = $true)][string]$Queue,
    [switch]$DryRun,
    [string]$Python = ".\.venv\nestor\Scripts\python.exe",
    [string]$CondaLib = "",          # e.g. C:\...\conda\envs\nestor\Library\bin (numpy DLLs for a venv python)
    [string]$Only = "",
    [switch]$RetryFailed
)
$ErrorActionPreference = "Stop"
if (Test-Path .\msvc_env.ps1) { . .\msvc_env.ps1 }          # MSVC for torch.compile (the file)
if ($CondaLib -ne "") { $env:PATH = "$CondaLib;" + $env:PATH }
$env:PYTHONUTF8 = "1"
$env:WANDB_MODE = "disabled"
$env:DCLM_NUM_WORKERS = "0"
$argv = @((Join-Path $PSScriptRoot "run_queue.py"), $Queue)     # next to this script, whatever the cwd
if ($DryRun) { $argv += "--dry-run" }
if ($RetryFailed) { $argv += "--retry-failed" }
if ($Only -ne "") { $argv += @("--only", $Only) }
& $Python @argv
exit $LASTEXITCODE
