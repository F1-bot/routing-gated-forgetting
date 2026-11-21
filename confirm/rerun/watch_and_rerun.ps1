# watch_and_rerun.ps1 -- post-freeze amendment A-3 (confirm\rerun\README.md). Waits until the main confirm queue has ended,
# installs A-3 (phase6\amendment_A3_post_freeze.md, chain6b_util.py amend --id A-3, re-install; a check after every step) and
# re-runs audit_moe_b1 and audit_moe_b2 of the frozen queue:
#     run_queue.py confirm\rerun\queue_confirm_audit_rerun.json --only audit_moe_b1,audit_moe_b2 --retry-failed
# in the chain's environment (chain6b.ps1 lines 52-75). Windows PowerShell 5.1, ASCII only. Log: logs\audit_rerun.log (UTF-8,
# no BOM). Until the main queue has ended it writes only that log and confirm\rerun\WATCH.lock. Never deletes a file.
# Exit code: run_queue's (run_queue also exits 0 when a job ends 'failed': the last log line and confirm\rerun\rerun_result.json
# give the job states); 0 if both jobs are already done; 3 still waiting after -MaxPolls; 31 another watcher runs; 40-48 a
# check refused (README.md section 5).
#   powershell -NoProfile -ExecutionPolicy Bypass -File confirm\rerun\watch_and_rerun.ps1 -Root .
param(
    [string]$Root = '.',
    [int]$PollSec = 300,
    [int]$MaxPolls = 0,                    # 0: wait without limit
    [string]$Python = '',                  # default <Root>\.venv\nestor\Scripts\python.exe (the chain's)
    [int]$GpuIdleMiB = 4096,               # wait while the GPU holds more memory than this (0: no GPU check)
    [switch]$ManualA3,                     # do not amend / re-install: wait until A-3 is installed by hand (README section 6)
    [switch]$NoPowerLogger,                # rehearsal only
    [switch]$DryRunQueue                   # rehearsal only: run_queue --dry-run (runs nothing, writes no state)
)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location $Root
$Utf8 = New-Object System.Text.UTF8Encoding $false
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$Rerun = Join-Path $Root 'confirm\rerun'
$LogFile = Join-Path $Root 'logs\audit_rerun.log'
$WatchLock = Join-Path $Rerun 'WATCH.lock'
$ChainLock = Join-Path $Root 'results\chain6b\LOCK'
$ConfirmDone = Join-Path $Root 'results\chain6b\confirm.done'
$StateFile = Join-Path $Root 'results\queue_state_confirm.json'
$QueueLog = Join-Path $Root 'logs\queue_confirm\queue.log'
$Util = Join-Path $Root 'chain6b_util.py'
$AmendLog = Join-Path $Root 'freeze\amendments.jsonl'
$CodeRecord = Join-Path $Root 'logs\chain6b_code_sha256.json'
$Decision = 'phase6/amendment_A3_post_freeze.md'
$Staged = Join-Path $Root 'phase6\amendment_A3_post_freeze.md'
$Jobs = @('audit_moe_b1', 'audit_moe_b2')
$Terminal = @('done', 'pass', 'fail', 'failed', 'skipped')
$Reason = 'A-3: the frozen assumption_audit.py calls dose_draws.base_inputs, which the frozen dose_draws.py does not define; audit_moe_b1 and audit_moe_b2 (queue positions 83 and 84) re-run after the main queue through confirm/rerun/audit_shim.py, which restores that function verbatim from phase6/pre_v13/dose_draws.py; no frozen file is changed'
$Approver = 'operator (delegated by the project owner)'
if (-not $Python) { $Python = Join-Path $Root '.venv\nestor\Scripts\python.exe' }
$Pin = [ordered]@{                        # reviewed for A-3 (README.md section 2)
    'confirm\rerun\audit_shim.py' = '01591bff7f7db9306861a661ecaccc98b3ee4b7902463b6d900905f8ed21d0ff'
    'confirm\rerun\queue_confirm_audit_rerun.json' = '37de9fc326e6aaad905399cc88be5eb56ef3d19a6de09cea68ab98018111cba8'
    'confirm\rerun\amendment_A3_post_freeze.md' = '48b8d72a52acaba9d4884a2660d1b2e8e020a395e0daffb3647caf4b68043a12'
    'confirm\queue_confirm.json' = '47c95918229a8a2ed3eca4adc49d7d80fc89f2ad1cc4ebad0c2935622000269b'
    'assumption_audit.py' = '39adede34e6922f0d5f2afeff2dc4437d9a92602d26b4f764d8126e1c97de6d8'
    'dose_draws.py' = 'bbb3c5c03dfb9cdf80f7e08740eabc7554a1fbbcbdc4525f101b34f20562aaa2'
    'phase6\pre_v13\dose_draws.py' = 'e7377f11d098e85ce59c5ddfaacf5f8f00e7f4e3d1c6e810ce9474b8817250fc'
}
$DecSha = $Pin['confirm\rerun\amendment_A3_post_freeze.md']

function Log([string]$msg) {
    $line = "[watch-A3] $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($LogFile, $line + "`r`n", $Utf8)
}
function Sha([string]$p) { return (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower() }
function Quit([int]$code, [string]$msg) {
    if ($msg) { Log $msg }
    Log "exit $code"
    [System.IO.File]::WriteAllText($WatchLock, "released $(Get-Date -Format s) by $PID (exit $code)`r`n", $Utf8)   # overwritten, never deleted
    exit $code
}
function Run-Py([string]$tag, [string[]]$argv) {
    Log "$tag | $Python $($argv -join ' ')"
    $lines = New-Object System.Collections.Generic.List[string]
    $global:LASTEXITCODE = 0
    & $Python @argv 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $s = $_.Exception.Message } else { $s = [string]$_ }
        $lines.Add($s)
        [System.IO.File]::AppendAllText($LogFile, "    $s`r`n", $Utf8)
    }
    $rc = $LASTEXITCODE
    Log "$tag | exit $rc"
    return [pscustomobject]@{ rc = $rc; out = $lines.ToArray() }
}
function Test-Pins {
    foreach ($rel in $Pin.Keys) {
        $p = Join-Path $Root $rel
        if (-not (Test-Path -LiteralPath $p)) { return "$rel is missing" }
        $h = Sha $p
        if ($h -ne $Pin[$rel]) { return "$rel is $h, not the reviewed $($Pin[$rel])" }
    }
    return ''
}
$script:CimNote = $false
function Get-QueueProcesses {
    try {
        $p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop | Where-Object {
                $_.CommandLine -match 'run_queue\.py|run_exp\.py|eval_windows\.py|assumption_audit\.py|audit_shim\.py|subset_adapt\.py' })
        if ($p.Count -gt 0) { return (($p | ForEach-Object { "python PID $($_.ProcessId)" }) -join ', ') }
    } catch {
        if (-not $script:CimNote) { Log "note: process list not readable ($($_.Exception.Message)); relying on the chain LOCK, confirm.done and queue.log"; $script:CimNote = $true }
    }
    return ''
}
function Get-GpuMiB {
    if ($GpuIdleMiB -le 0) { return -1 }
    $smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
    if (-not $smi) { return -1 }
    $o = @(& $smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1 | ForEach-Object { [string]$_ })
    $v = 0
    if ($LASTEXITCODE -eq 0 -and $o.Count -ge 1 -and [int]::TryParse($o[0].Trim(), [ref]$v)) { return $v }
    return -1
}
$QueueIds = @((Get-Content -LiteralPath (Join-Path $Root 'confirm\queue_confirm.json') -Raw -Encoding UTF8 | ConvertFrom-Json).jobs | ForEach-Object { $_.id })
function Get-Status {
    $why = New-Object System.Collections.Generic.List[string]
    $fatal = ''
    if (Test-Path -LiteralPath (Join-Path $Root 'results\STOP')) { $why.Add('results\STOP present') }
    if (Test-Path -LiteralPath (Join-Path $Root 'PAUSE')) { $why.Add('PAUSE present') }
    $lockTxt = ''
    if (Test-Path -LiteralPath $ChainLock) { $lockTxt = [string](Get-Content -LiteralPath $ChainLock -Raw); $lockTxt = $lockTxt.Trim() }
    if (-not $lockTxt) { $why.Add('results\chain6b\LOCK missing or empty') }
    elseif ($lockTxt -match '^\d+$') {
        if (Get-Process -Id ([int]$lockTxt) -ErrorAction SilentlyContinue) { $why.Add("chain6b (PID $lockTxt) runs") }
        else { $why.Add("results\chain6b\LOCK names PID $lockTxt, which does not run (the chain ended without releasing it)") }
    } elseif ($lockTxt -match '^released .*\(exit (-?\d+)\)') {
        if ($Matches[1] -ne '0') { $why.Add("chain6b exited with code $($Matches[1]) (STOP or a failure; jobs may be left)") }
    } else { $why.Add("results\chain6b\LOCK not understood: $lockTxt") }
    $done = Test-Path -LiteralPath $ConfirmDone
    if (-not $done) { $why.Add('results\chain6b\confirm.done missing (the confirm step has not completed)') }
    $tail = @(Get-Content -LiteralPath $QueueLog -Tail 400 -ErrorAction SilentlyContinue)
    $open = ''
    foreach ($l in $tail) {                               # the last 'start <id>' without a later 'end <id>' = a job in progress
        if ($l -match '^\[queue\] \S+ \S+ start (\S+) ') { $open = $Matches[1] }
        elseif ($open -and $l -match ('^\[queue\] \S+ \S+ end ' + [regex]::Escape($open) + ':')) { $open = '' }
    }
    if ($open) { $why.Add("queue.log: job $open started and has not ended") }
    $pr = Get-QueueProcesses
    if ($pr) { $why.Add("queue process running: $pr") }
    $mib = Get-GpuMiB
    if ($mib -ge 0 -and $mib -gt $GpuIdleMiB) { $why.Add("GPU memory in use $mib MiB > $GpuIdleMiB MiB") }
    $st = $null
    try { $st = Get-Content -LiteralPath $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json } catch { $why.Add("queue state not readable: $($_.Exception.Message)") }
    $missing = @(); $other = @(); $aud = [ordered]@{}
    if ($st) {
        foreach ($id in $QueueIds) {
            $e = $st.PSObject.Properties[$id]
            $s = if ($e) { [string]$e.Value.status } else { '' }
            if ($Terminal -notcontains $s) { $missing += $id }
            elseif ($s -eq 'failed' -and $Jobs -notcontains $id) { $other += $id }
            if ($Jobs -contains $id) { $aud[$id] = $s }
        }
        if ($missing.Count -gt 0) {
            $why.Add("$($missing.Count) of $($QueueIds.Count) jobs without a final state")
            if ($why.Count -eq 1 -and $done) { $fatal = "confirm.done exists and the chain released its lock, but $($missing.Count) jobs have no final state (e.g. $($missing[0]))" }
        }
    }
    return [pscustomobject]@{ ready = ($why.Count -eq 0); why = ($why -join '; '); fatal = $fatal; audit = $aud; otherFailed = $other; gpuMiB = $mib }
}
function Get-A3 {
    if (-not (Test-Path -LiteralPath $AmendLog)) { return @() }
    $r = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
    return @($r | Where-Object { $_.amendment -eq 'A-3' })
}
function Get-CodeRecordFiles([string[]]$out) {
    foreach ($l in $out) { if ($l -match 'amended after the freeze: code record \(.*\): (\d+) file\(s\)') { return [int]$Matches[1] } }
    return 0
}
function Amend-Args([string]$decision, [bool]$dry) {
    $a = @($Util, 'amend', '--root', $Root, '--id', 'A-3', '--reason', $Reason, '--approved-by', $Approver, '--decision', $decision)
    if ($dry) { $a += '--dry-run' }
    return ,$a
}
function Undo-Staging([string]$why) {
    if (Test-Path -LiteralPath $Staged) {
        $bk = Join-Path $Rerun ('rollback_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        New-Item -ItemType Directory -Force -Path $bk | Out-Null
        Move-Item -LiteralPath $Staged -Destination $bk
        Log "staging undone ($why): phase6\amendment_A3_post_freeze.md moved to $bk"
    }
}
function Test-A3Installed {
    $a3 = @(Get-A3)
    if ($a3.Count -eq 0) { return 'absent' }
    $ok = @($a3 | Where-Object { $_.path -eq $Decision -and $_.new_sha256 -eq $DecSha -and $_.decision -eq $Decision -and $_.decision_sha256 -eq $DecSha })
    if ($ok.Count -lt 1) { return 'other' }
    if (-not (Test-Path -LiteralPath $Staged) -or (Sha $Staged) -ne $DecSha) { return 'nofile' }
    $rec = Get-Content -LiteralPath $CodeRecord -Raw -Encoding UTF8 | ConvertFrom-Json
    $f = $rec.files.PSObject.Properties[$Decision]
    if (-not $f -or $f.Value -ne $DecSha) { return 'recorded' }      # amended, not yet re-installed
    return 'installed'
}
function Install-A3 {
    $s0 = Test-A3Installed
    if ($s0 -eq 'other') { Quit 46 "freeze\amendments.jsonl holds A-3 records that are not this decision file ($DecSha)" }
    if ($s0 -eq 'nofile') { Quit 46 'A-3 is recorded but phase6\amendment_A3_post_freeze.md is missing or differs' }
    if ($s0 -eq 'installed') {
        $fv = Run-Py 'freeze-verify (A-3 found installed)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
        if ($fv.rc -ne 0) { Quit 46 'A-3 is installed but freeze-verify fails (README section 5)' }
        Log 'A-3 already installed: record, phase6 file and code record match; freeze-verify passes'
        return
    }
    if ($s0 -eq 'recorded') { Quit 45 'A-3 is recorded but not re-installed (freeze-verify exits 23): README section 6, step 7' }
    $fv0 = Run-Py 'freeze-verify (before A-3)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    if ($fv0.rc -ne 0) { Quit 44 'freeze-verify fails before A-3: nothing written' }
    $n0 = Get-CodeRecordFiles $fv0.out
    $d0 = Run-Py 'amend --dry-run (before staging)' (Amend-Args 'confirm/rerun/amendment_A3_post_freeze.md' $true)
    $rows0 = @($d0.out | Where-Object { $_ -match '^\[amend\] A-3: ' })
    if ($d0.rc -ne 0 -or $rows0.Count -ne 0 -or -not ($d0.out -contains '[amend] nothing differs from the frozen code record or the amended state: nothing recorded')) {
        Quit 41 'the code set differs from the amended state before A-3 is staged (something else would be recorded): nothing written' }
    if (Test-Path -LiteralPath $Staged) {
        if ((Sha $Staged) -ne $DecSha) { Quit 42 'phase6\amendment_A3_post_freeze.md exists with other content: nothing written' }
        Log 'phase6\amendment_A3_post_freeze.md is already staged (same sha256)'
    } else {
        Copy-Item -LiteralPath (Join-Path $Rerun 'amendment_A3_post_freeze.md') -Destination $Staged
        if ((Sha $Staged) -ne $DecSha) { Undo-Staging 'copy check'; Quit 42 'copy check of the staged decision file failed: nothing recorded' }
        Log "staged phase6\amendment_A3_post_freeze.md (sha256 $DecSha)"
    }
    $d1 = Run-Py 'amend --dry-run (staged)' (Amend-Args $Decision $true)
    $rows1 = @($d1.out | Where-Object { $_ -match '^\[amend\] A-3: ' })
    $want = "[amend] A-3: phase6/amendment_A3_post_freeze.md  absent -> $($DecSha.Substring(0, 12))  (staged)"
    if ($d1.rc -ne 0 -or $rows1.Count -ne 1 -or $rows1[0] -cne $want -or -not ($d1.out -ccontains '[amend] dry run: 1 record(s) not written')) {
        Undo-Staging 'dry run'; Quit 42 'the dry run after staging is not exactly the one expected record: nothing recorded' }
    $before = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    $am = Run-Py 'amend --id A-3' (Amend-Args $Decision $false)
    $after = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    $a3 = @(Get-A3)
    if ($am.rc -ne 0 -or $after -ne ($before + 1) -or $a3.Count -ne 1 -or $a3[0].id -ne 'A-3.1' -or $a3[0].path -ne $Decision -or
        $a3[0].old_sha256 -ne 'absent' -or $a3[0].new_sha256 -ne $DecSha -or $a3[0].decision_sha256 -ne $DecSha) {
        Quit 43 "amend did not append exactly the expected record ($before -> $after lines): README section 5" }
    Log "recorded A-3.1 ($($a3[0].utc)): $Decision absent -> $DecSha"
    $fv1 = Run-Py 'freeze-verify (recorded, not re-installed; exit 23 by design)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    if ($fv1.rc -ne 23) { Log "note: freeze-verify exit $($fv1.rc) between amend and re-install (23 expected)" }
    $in = Run-Py 'install --reinstall' @($Util, 'install', '--root', $Root, '--reinstall')
    if ($in.rc -ne 0) { Quit 45 're-install failed: A-3 is recorded and freeze-verify exits 23 until the re-install (README section 6, step 7)' }
    $mk = Run-Py 'mark install' @($Util, 'mark', '--root', $Root, '--step', 'install', '--info', 'installed (reinstall=True; amendment A-3, confirm/rerun/watch_and_rerun.ps1)')
    if ($mk.rc -ne 0) { Quit 45 'mark install failed after the re-install' }
    $fv2 = Run-Py 'freeze-verify (A-3 installed)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    $n2 = Get-CodeRecordFiles $fv2.out
    if ($fv2.rc -ne 0 -or $n2 -ne ($n0 + 1) -or (Test-A3Installed) -ne 'installed') {
        Quit 46 "freeze-verify after the re-install: exit $($fv2.rc), amended code-record files $n2 (expected $($n0 + 1))" }
    Log "A-3 installed: record A-3.1, code record re-installed, freeze-verify passes ($n2 amended code-record files, $n0 before)"
}
function Start-PowerLogger {
    $dir = Join-Path $Root 'logs\power'
    $running = ''
    try {
        $p = @(Get-CimInstance Win32_Process -Filter "Name='nvidia-smi.exe'" -ErrorAction Stop | Where-Object { $_.CommandLine -match 'gpu_power_' })
        if ($p.Count -gt 0) { $running = "PID $($p[0].ProcessId): $($p[0].CommandLine)" }
    } catch {
        $newest = Get-ChildItem -LiteralPath $dir -Filter 'gpu_power_*.csv' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
        $smiProc = @(Get-Process -Name nvidia-smi -ErrorAction SilentlyContinue)
        if ($newest -and $smiProc.Count -gt 0 -and ((Get-Date) - $newest.LastWriteTime).TotalSeconds -lt 120) {
            $running = "nvidia-smi PID $($smiProc[0].Id); $($newest.Name) written $([int]((Get-Date) - $newest.LastWriteTime).TotalSeconds) s ago" }
    }
    if ($running) { Log "power logger running ($running): no new logger, no README note"; return }
    $smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
    if (-not $smi) { Log 'WARNING: nvidia-smi.exe not found: the re-run has no power log'; return }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $csv = Join-Path $dir "gpu_power_$stamp.csv"
    $p = Start-Process -FilePath $smi -ArgumentList @('--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm',
        '--format=csv,noheader', '-lms', '1000', '-f', $csv) -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 5
    if (-not (Test-Path -LiteralPath $csv)) { Log "WARNING: the power logger (PID $($p.Id)) has not created $csv after 5 s" }
    $note = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') (watch_and_rerun.ps1, amendment A-3): logging resumes for the re-run of " +
            "audit_moe_b1 and audit_moe_b2 after the main queue in gpu_power_$stamp.csv (same command and columns; logger PID $($p.Id))."
    $readme = Join-Path $dir 'README_power_log.txt'
    $pre = ''
    if (Test-Path -LiteralPath $readme) {
        $b = [System.IO.File]::ReadAllBytes($readme)
        if ($b.Length -gt 0 -and $b[$b.Length - 1] -ne 10) { $pre = "`r`n" }
    }
    [System.IO.File]::AppendAllText($readme, $pre + $note + "`r`n", [System.Text.Encoding]::ASCII)
    Log "power logger started: PID $($p.Id) -> logs\power\gpu_power_$stamp.csv; note appended to logs\power\README_power_log.txt"
}

# ------------------------------------------------------------------------------------------------ start
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
if (Test-Path -LiteralPath $WatchLock) {
    $old = ([string](Get-Content -LiteralPath $WatchLock -Raw)).Trim()
    if ($old -match '^\d+$' -and $old -ne "$PID" -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "[watch-A3] another watcher (PID $old) runs"; exit 31 }
}
[System.IO.File]::WriteAllText($WatchLock, "$PID`r`n", $Utf8)
Log "##### start (PID $PID): root $Root, python $Python, poll $PollSec s, max polls $MaxPolls, GPU idle limit $GpuIdleMiB MiB, manual A-3 $ManualA3, power logger $(-not $NoPowerLogger), dry-run queue $DryRunQueue"
if (-not (Test-Path -LiteralPath $Python)) { Quit 40 "python $Python not found (nothing written)" }
if (-not (Test-Path -LiteralPath $Util)) { Quit 40 "$Util not found (nothing written)" }
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed: $bad (nothing written)" }
Log "pinned files verified: $(($Pin.Keys | ForEach-Object { "$_ $($Pin[$_].Substring(0, 12))" }) -join ', ')"
# the chain's environment (chain6b.ps1 lines 52-75), set before any python call
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { Log "note: priority not set ($($_.Exception.Message))" }
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
New-Item -ItemType Directory -Force "$Root\.tmp" | Out-Null
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:WANDB_MODE = 'disabled'
$env:NESTOR_CKPT_PINS = "$Root\ckpt_pins.json"

# ------------------------------------------------------------------------------------------------ wait for the end of the main queue
$n = 0; $lastWhy = ''
while ($true) {
    $s = Get-Status
    if ($s.fatal) { Quit 48 $s.fatal }
    if ($s.ready) {
        $confirm = [Math]::Min($PollSec, 60)
        Log "main queue looks finished; confirming in $confirm s"
        Start-Sleep -Seconds $confirm
        $s = Get-Status
        if ($s.ready) { break }
    }
    if ($s.why -ne $lastWhy -or ($n % 12) -eq 0) { Log "waiting: $($s.why)"; $lastWhy = $s.why }
    $n++
    if ($MaxPolls -gt 0 -and $n -ge $MaxPolls) { Quit 3 "still waiting after $n polls" }
    Start-Sleep -Seconds $PollSec
}
$a1 = $s.audit['audit_moe_b1']; $a2 = $s.audit['audit_moe_b2']
Log "main queue finished: $($QueueIds.Count) jobs with a final state; audit_moe_b1 $a1, audit_moe_b2 $a2; other failed jobs: $(if ($s.otherFailed.Count) { $s.otherFailed -join ', ' } else { 'none' }); GPU memory $($s.gpuMiB) MiB"
if ($a1 -eq 'done' -and $a2 -eq 'done') { Quit 0 'both audit jobs are already done: nothing to re-run' }
foreach ($j in $Jobs) { if (@('failed', 'done') -notcontains $s.audit[$j]) { Quit 48 "$j has state '$($s.audit[$j])' (failed or done expected)" } }
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed: $bad (nothing written)" }

# ------------------------------------------------------------------------------------------------ amendment A-3
if ($ManualA3) {
    $k = 0
    while ((Test-A3Installed) -ne 'installed') {
        if (($k % 12) -eq 0) { Log "manual A-3: waiting for the amend and re-install (README section 6); now: $(Test-A3Installed)" }
        $k++
        if ($MaxPolls -gt 0 -and $k -ge $MaxPolls) { Quit 3 "manual A-3: still not installed after $k polls" }
        Start-Sleep -Seconds $PollSec
    }
}
Install-A3

# ------------------------------------------------------------------------------------------------ re-run
$s = Get-Status
if (-not $s.ready) { Quit 48 "the main queue no longer looks finished: $($s.why)" }
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed before the re-run: $bad" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ev = Join-Path $Rerun "evidence\before_rerun_$stamp"
New-Item -ItemType Directory -Force -Path $ev | Out-Null
foreach ($rel in @('logs\queue_confirm\audit_moe_b1.log', 'logs\queue_confirm\audit_moe_b2.log', 'results\queue_state_confirm.json')) {
    $src = Join-Path $Root $rel
    if (-not (Test-Path -LiteralPath $src)) { Log "evidence: $rel not present"; continue }
    $dst = Join-Path $ev (Split-Path $rel -Leaf)
    Copy-Item -LiteralPath $src -Destination $dst
    if ((Sha $src) -ne (Sha $dst)) { Quit 47 "evidence copy of $rel failed" }
    Log "evidence: $rel -> confirm\rerun\evidence\before_rerun_$stamp ($(Sha $dst))"
}
if (-not $NoPowerLogger) { Start-PowerLogger }
$qa = @((Join-Path $Root 'run_queue.py'), (Join-Path $Rerun 'queue_confirm_audit_rerun.json'), '--only', 'audit_moe_b1,audit_moe_b2', '--retry-failed')
if ($DryRunQueue) { $qa += '--dry-run' }
$r = Run-Py 'run_queue (A-3 re-run)' $qa
$res = [ordered]@{ utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); run_queue_exit = $r.rc; dry_run = [bool]$DryRunQueue; jobs = [ordered]@{} }
$st = Get-Content -LiteralPath $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
$summary = @()
foreach ($j in $Jobs) {
    $e = $st.PSObject.Properties[$j]
    $base = $j.Substring(6)
    $outs = [ordered]@{}
    foreach ($f in @('audit_realized_groups.json', 'audit_realized_windows.npz', 'audit_realized.json')) { $outs[$f] = Test-Path -LiteralPath (Join-Path $Root "audit\$base\$f") }
    $jl = Join-Path $Root "logs\queue_confirm\$j.log"
    $shim = (Test-Path -LiteralPath $jl) -and [bool](Select-String -LiteralPath $jl -Pattern '^\[audit-shim\] A-3: dose_draws\.base_inputs restored' -Quiet)
    $res.jobs[$j] = [ordered]@{ status = $(if ($e) { $e.Value.status } else { $null }); rc = $(if ($e) { $e.Value.rc } else { $null }); hours = $(if ($e) { $e.Value.hours } else { $null }); outputs = $outs; shim_line_in_log = $shim }
    $summary += "$j $($res.jobs[$j].status) (outputs $(@($outs.Values | Where-Object { $_ }).Count)/3, shim line $shim)"
}
[System.IO.File]::WriteAllText((Join-Path $Rerun 'rerun_result.json'), ($res | ConvertTo-Json -Depth 6) + "`r`n", $Utf8)
Quit $r.rc "re-run finished: run_queue exit $($r.rc); $($summary -join '; ')$(if ($DryRunQueue) { ' [dry run: nothing ran, the state is not written]' })"
