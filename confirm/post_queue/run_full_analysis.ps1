# run_full_analysis.ps1 -- the full confirmatory analysis, after the confirm queue, the A-3 audit re-run and the A-4 re-run
# of positions 154 and 155 (prereg v1.3 sec. 3.5, P-16: the full analysis runs after the whole queue, with the frozen
# analyze_confirm.py).
# Windows PowerShell 5.1, ASCII only. Never deletes or overwrites a file. Writes only logs\analysis_full.log (UTF-8, no
# BOM, appended) and, through analyze_confirm.py, results\analysis\analysis_confirm.json, .json.sha256 and .md.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File confirm\post_queue\run_full_analysis.ps1
#   ... -CheckOnly        the preconditions only (console output; nothing written, nothing run)
#
# Command (CPU only; Idle priority, inherited by the child; one thread per numeric library; torch imports blocked):
#   <Root>\.venv\nestor\Scripts\python.exe <this folder>\analysis_notorch.py --root <Root> --log <Root>\logs\analysis_full.log
#       -- run --mode full --root <Root>
# in the chain's Python environment (chain6b.ps1 lines 58-74, threads 1) with PATH prefixed by the conda Library\bin.
# analyze_confirm.py verifies FREEZE.json itself (exit 23 if that fails) and writes results\analysis (its default for
# --mode full, analyze_confirm.py line 2500); results\analysis_core (the core analysis) is not touched.
#
# Preconditions (every failure is listed; exit = the smallest code; nothing is run):
#   41  results\chain6b\confirm.done present and results\chain6b\LOCK reads 'released ... (exit 0)'
#   42  nothing runs: no results\STOP or PAUSE, no open job in logs\queue_confirm\queue.log, no queue, audit or analysis
#       python process (when the process list is readable), GPU memory <= -GpuIdleMiB MiB (0: no GPU check)
#   43  confirm\rerun\rerun_result.json present and not from a dry run; both audit states are reported (from it and from
#       results\queue_state_confirm.json); an audit that is not 'done' is a warning, not a refusal
#   44  results\a2_acceptance.json present, with exactly the keys accepted, checks, rule and utc
#   45  results\analysis\analysis_confirm.json absent (the full-analysis output is never overwritten)
#   46  the root, python, analysis_notorch.py or analyze_confirm.py missing
#   47  confirm\rerun_a4\rerun_a4_result.json present and not from a dry run (the A-4 re-run of positions 154 and 155 has
#       ended), and results\a4_acceptance.json present with exactly the keys accepted, checks, rule and utc; a re-run that
#       is not accepted is a warning (its values are withheld, D-3), not a refusal
# Warning only, never a refusal: results\a1_wd_only_checks.json (the [wd_only] lines of the C3 FULL jobs at positions
# 131, 133 and 140, D-1 addendum 1, item 6; confirm\post_queue\a1_wd_only_checks.py) missing, unreadable or not passed.
# Exit: analyze_confirm's code (0 written, 23 FREEZE.json verification failed, 32 firewall, 1 exception) or 41-47.
param(
    [string]$Root = '.',
    [string]$Python = '',
    [string]$CondaLibBin = '<conda>\envs\nestor\Library\bin',
    [int]$GpuIdleMiB = 4096,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Continue'
if (-not (Test-Path -LiteralPath $Root)) { Write-Host "[run_full_analysis] root $Root missing"; exit 46 }
$Root = (Resolve-Path -LiteralPath $Root).Path
(Get-Process -Id $PID).PriorityClass = 'Idle'                  # inherited by the python child
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$LogPath = Join-Path $Root 'logs\analysis_full.log'
$Launcher = Join-Path $PSScriptRoot 'analysis_notorch.py'
if ($Python) { $py = $Python } else { $py = Join-Path $Root '.venv\nestor\Scripts\python.exe' }
$AuditJobs = @('audit_moe_b1', 'audit_moe_b2')

function Log([string]$msg) {
    $line = "[run_full_analysis] $(Get-Date -Format s) $msg"
    Write-Host $line
    if (-not $CheckOnly) {
        New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
        [System.IO.File]::AppendAllText($LogPath, $line + "`r`n", $Utf8)
    }
}
function Sha([string]$p) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower() }
function Read-Json([string]$p) { return (Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json) }
$Fails = New-Object System.Collections.Generic.List[object]
function Fail([int]$code, [string]$text) { $script:Fails.Add([pscustomobject]@{ code = $code; text = $text }) }

Log "##### start (PID $PID): root $Root, python $py, GPU idle limit $GpuIdleMiB MiB, check only $CheckOnly"

# ------------------------------------------------------------------ 41: the chain finished the confirm step, exit 0
$done = Join-Path $Root 'results\chain6b\confirm.done'
$lockP = Join-Path $Root 'results\chain6b\LOCK'
if (-not (Test-Path -LiteralPath $done)) { Fail 41 'results\chain6b\confirm.done missing (the confirm step has not completed)' }
$lockTxt = ''
if (Test-Path -LiteralPath $lockP) { $lockTxt = ([string](Get-Content -LiteralPath $lockP -Raw)).Trim() }
if ($lockTxt -notmatch '^released .*\(exit 0\)$') { Fail 41 "results\chain6b\LOCK reads '$lockTxt' (expected 'released ... (exit 0)')" }

# ------------------------------------------------------------------ 42: nothing runs
foreach ($f in @('results\STOP', 'PAUSE')) { if (Test-Path -LiteralPath (Join-Path $Root $f)) { Fail 42 "$f present" } }
$open = ''
foreach ($l in @(Get-Content -LiteralPath (Join-Path $Root 'logs\queue_confirm\queue.log') -Tail 400 -ErrorAction SilentlyContinue)) {
    if ($l -match '^\[queue\] \S+ \S+ start (\S+) ') { $open = $Matches[1] }      # as watch_and_rerun.ps1
    elseif ($open -and $l -match ('^\[queue\] \S+ \S+ end ' + [regex]::Escape($open) + ':')) { $open = '' }
}
if ($open) { Fail 42 "logs\queue_confirm\queue.log: job $open started and has not ended" }
try {
    $p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop | Where-Object {
            $_.CommandLine -match 'run_queue\.py|run_exp\.py|eval_windows\.py|assumption_audit\.py|audit_shim\.py|subset_adapt\.py|analyze_confirm\.py|analysis_notorch\.py' })
    if ($p.Count -gt 0) { Fail 42 ('queue, audit or analysis process running: ' + (($p | ForEach-Object { "python PID $($_.ProcessId)" }) -join ', ')) }
} catch { Log "note: process list not readable ($($_.Exception.Message)); relying on the chain LOCK, confirm.done, queue.log and the GPU" }
if ($GpuIdleMiB -gt 0) {
    $mib = -1
    $smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
    if ($smi) {
        $o = @(& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 | ForEach-Object { [string]$_ })
        $v = 0
        if ($LASTEXITCODE -eq 0 -and $o.Count -ge 1 -and [int]::TryParse($o[0].Trim(), [ref]$v)) { $mib = $v }
    }
    if ($mib -lt 0) { Log 'note: GPU memory not readable (nvidia-smi); relying on the other checks' }
    elseif ($mib -gt $GpuIdleMiB) { Fail 42 "GPU memory in use $mib MiB > $GpuIdleMiB MiB (a GPU job runs)" }
    else { Log "GPU memory in use $mib MiB (limit $GpuIdleMiB MiB)" }
} else { Log 'GPU check skipped (-GpuIdleMiB 0)' }

# ------------------------------------------------------------------ 43: the A-3 audit re-run has ended
$rr = Join-Path $Root 'confirm\rerun\rerun_result.json'
if (-not (Test-Path -LiteralPath $rr)) { Fail 43 'confirm\rerun\rerun_result.json missing (the A-3 audit re-run has not ended)' }
else {
    try {
        $r = Read-Json $rr
        Log "A-3 re-run: confirm\rerun\rerun_result.json sha256 $(Sha $rr), utc $($r.utc), run_queue exit $($r.run_queue_exit), dry run $($r.dry_run)"
        if ($r.dry_run) { Fail 43 'confirm\rerun\rerun_result.json is from a dry run (nothing ran)' }
        foreach ($j in $AuditJobs) {
            $e = $r.jobs.PSObject.Properties[$j]
            if ($e) {
                $n = @($e.Value.outputs.PSObject.Properties | Where-Object { $_.Value }).Count
                Log "  rerun_result.json: $j status $($e.Value.status), rc $($e.Value.rc), hours $($e.Value.hours), outputs $n/3, shim line $($e.Value.shim_line_in_log)"
            } else { Log "  rerun_result.json: $j not recorded" }
        }
    } catch { Fail 43 "confirm\rerun\rerun_result.json not readable: $($_.Exception.Message)" }
}
try {
    $st = Read-Json (Join-Path $Root 'results\queue_state_confirm.json')
    foreach ($j in $AuditJobs) {
        $e = $st.PSObject.Properties[$j]
        $s = if ($e) { [string]$e.Value.status } else { 'pending' }
        Log "  results\queue_state_confirm.json: $j $s"
        if ($s -ne 'done') { Log "WARNING: $j is '$s' (the audit block of analyze_confirm.py needs status 'done' and the audit files)" }
    }
} catch { Fail 43 "results\queue_state_confirm.json not readable: $($_.Exception.Message)" }

# ------------------------------------------------------------------ 44: the A-2 acceptance record
$aa = Join-Path $Root 'results\a2_acceptance.json'
if (-not (Test-Path -LiteralPath $aa)) { Fail 44 'results\a2_acceptance.json missing (run confirm\post_queue\a2_acceptance.py after positions 137, 142 and 147)' }
else {
    try {
        $a = Read-Json $aa
        $keys = @($a.PSObject.Properties.Name) -join ','
        if ($keys -ne 'accepted,checks,rule,utc') { Fail 44 "results\a2_acceptance.json has the keys $keys (expected accepted,checks,rule,utc)" }
        else { Log ("A-2 acceptance (results\a2_acceptance.json sha256 $(Sha $aa), utc $($a.utc)): " + (($a.accepted.PSObject.Properties | ForEach-Object { "$($_.Name) $($_.Value)" }) -join '; ')) }
    } catch { Fail 44 "results\a2_acceptance.json not readable: $($_.Exception.Message)" }
}

# ------------------------------------------------------------------ 47: the A-4 re-run of positions 154 and 155 has ended
$r4p = Join-Path $Root 'confirm\rerun_a4\rerun_a4_result.json'
$a4p = Join-Path $Root 'results\a4_acceptance.json'
$A4Jobs = @('c3_freeze_moe_b1_german_s103', 'c3_freeze_nestor_german_s103')
if (-not (Test-Path -LiteralPath $r4p)) { Fail 47 'confirm\rerun_a4\rerun_a4_result.json missing (the A-4 re-run of positions 154 and 155 has not ended)' }
else {
    try {
        $r4 = Read-Json $r4p
        Log "A-4 re-run: confirm\rerun_a4\rerun_a4_result.json sha256 $(Sha $r4p), utc $($r4.utc), run_queue exit $($r4.run_queue_exit), dry run $($r4.dry_run), re-run $(@($r4.rerun) -join ',')"
        if ($r4.dry_run) { Fail 47 'confirm\rerun_a4\rerun_a4_result.json is from a dry run (nothing ran)' }
    } catch { Fail 47 "confirm\rerun_a4\rerun_a4_result.json not readable: $($_.Exception.Message)" }
}
if (-not (Test-Path -LiteralPath $a4p)) { Fail 47 'results\a4_acceptance.json missing (confirm\rerun_a4\a4_acceptance.py runs after the A-4 re-run)' }
else {
    try {
        $a4 = Read-Json $a4p
        $keys = @($a4.PSObject.Properties.Name) -join ','
        if ($keys -ne 'accepted,checks,rule,utc') { Fail 47 "results\a4_acceptance.json has the keys $keys (expected accepted,checks,rule,utc)" }
        else {
            Log ("A-4 acceptance (results\a4_acceptance.json sha256 $(Sha $a4p), utc $($a4.utc)): " + (($A4Jobs | ForEach-Object { "$_ $($a4.accepted.$_)" }) -join '; '))
            foreach ($j in $A4Jobs) {
                if ($a4.accepted.$j -ne $true) { Log "WARNING: $j is not accepted under A-4: its values are withheld (not reported; D-3); reasons: $(@($a4.checks.$j.reasons) -join '; ')" }
            }
        }
    } catch { Fail 47 "results\a4_acceptance.json not readable: $($_.Exception.Message)" }
}

# ------------------------------------------------------------------ warning only: the [wd_only] checks of the C3 FULL jobs
$a1 = Join-Path $Root 'results\a1_wd_only_checks.json'
if (-not (Test-Path -LiteralPath $a1)) {
    Log 'WARNING: results\a1_wd_only_checks.json missing (D-1 addendum 1, item 6: run confirm\post_queue\a1_wd_only_checks.py after positions 131, 133 and 140); not a precondition of the analysis'
} else {
    try {
        $b = Read-Json $a1
        Log ("A-1 [wd_only] checks (results\a1_wd_only_checks.json sha256 $(Sha $a1), utc $($b.utc)): " + (($b.pass.PSObject.Properties | ForEach-Object { "$($_.Name) $($_.Value)" }) -join '; '))
        $bad = @($b.pass.PSObject.Properties | Where-Object { -not $_.Value } | ForEach-Object { $_.Name })
        if ($bad.Count -gt 0) { Log "WARNING: [wd_only] check not passed for $($bad -join ', ') (D-1 addendum 1, item 6 stays open); not a precondition of the analysis" }
    } catch { Log "WARNING: results\a1_wd_only_checks.json not readable: $($_.Exception.Message); not a precondition of the analysis" }
}

# ------------------------------------------------------------------ 45: never overwrite the output
$outDir = Join-Path $Root 'results\analysis'
$outJson = Join-Path $outDir 'analysis_confirm.json'
if (Test-Path -LiteralPath $outJson) { Fail 45 "results\analysis\analysis_confirm.json exists (written $((Get-Item -LiteralPath $outJson).LastWriteTime.ToString('s'))); it is never overwritten: move it aside first" }

# ------------------------------------------------------------------ 46: the programs
foreach ($f in @($py, $Launcher, (Join-Path $Root 'analyze_confirm.py'))) { if (-not (Test-Path -LiteralPath $f)) { Fail 46 "$f missing" } }

if ($Fails.Count -gt 0) {
    foreach ($f in $Fails) { Log "precondition $($f.code) not met: $($f.text)" }
    $code = ($Fails | Sort-Object code | Select-Object -First 1).code
    Log "refused (exit $code); nothing was run"
    exit $code
}
if ($CheckOnly) { Log 'all preconditions met (check only: nothing was run)'; exit 0 }

# ------------------------------------------------------------------ environment (chain6b.ps1 lines 58-74, threads 1) and run
$env:PATH = "$CondaLibBin;$env:PATH"
$env:TIKTOKEN_CACHE_DIR = "$Root\data\tiktoken"
foreach ($v in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')) {
    Set-Item -Path "env:$v" -Value '1'
}
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = "$Root\.tmp"
$env:TMP = "$Root\.tmp"
New-Item -ItemType Directory -Force -Path "$Root\.tmp" | Out-Null
$argv = @($Launcher, '--root', $Root, '--log', $LogPath, '--', 'run', '--mode', 'full', '--root', $Root)
Log "command: $py $($argv -join ' ')"
Log "environment: PATH prefix $CondaLibBin; TIKTOKEN_CACHE_DIR $env:TIKTOKEN_CACHE_DIR; OMP/MKL/OPENBLAS/NUMEXPR/RAYON threads 1; PYTHONPATH $env:PYTHONPATH; priority $((Get-Process -Id $PID).PriorityClass)"
$t0 = Get-Date
& $py @argv
$rc = $LASTEXITCODE
$wall = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
foreach ($f in @('analysis_confirm.json', 'analysis_confirm.json.sha256', 'analysis_confirm.md')) {
    $p = Join-Path $outDir $f
    if (Test-Path -LiteralPath $p) { Log "output results\analysis\$f sha256 $(Sha $p) ($((Get-Item -LiteralPath $p).Length) bytes)" }
    else { Log "output results\analysis\$f not written" }
}
$side = Join-Path $outDir 'analysis_confirm.json.sha256'
if ((Test-Path -LiteralPath $outJson) -and (Test-Path -LiteralPath $side)) {
    $rec = ([string](Get-Content -LiteralPath $side -Raw)).Trim().Split(' ')[0].ToLower()
    Log "sidecar analysis_confirm.json.sha256 $(if ($rec -eq (Sha $outJson)) { 'matches' } else { 'DOES NOT match' }) the file"
}
Log "exit $rc (analysis wall $wall s)"
exit $rc
