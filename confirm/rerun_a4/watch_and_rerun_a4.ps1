# watch_and_rerun_a4.ps1 -- post-freeze amendment A-4 (confirm\rerun_a4\README.md). Waits until the A-3 audit re-run has
# ended (confirm\rerun\rerun_result.json written, the A-3 watcher's lock confirm\rerun\WATCH.lock released with exit 0) and
# the main confirm queue is idle, checks the first attempts of queue positions 154 and 155 against
# confirm\rerun_a4\expected_unchanged.json, installs A-4 (phase6\amendment_A4_post_freeze.md, chain6b_util.py amend --id A-4,
# re-install; a check after every step), keeps the first attempts (copies in confirm\rerun_a4\evidence\before_rerun_<time>\,
# each root moved to <root>_a4failed<epoch seconds>) and re-runs the two jobs of the frozen queue:
#     run_queue.py confirm\rerun_a4\queue_confirm_a4_rerun.json --only <the failed ones of the two> --retry-failed
# in the chain's environment (chain6b.ps1 lines 52-75); then confirm\rerun_a4\a4_acceptance.py writes
# results\a4_acceptance.json and this script writes confirm\rerun_a4\rerun_a4_result.json. Windows PowerShell 5.1, ASCII
# only. Log: logs\a4_rerun.log (UTF-8, no BOM). Until A-3 has ended and the queue is idle it writes only that log and
# confirm\rerun_a4\WATCH_A4.lock. Never deletes a file.
# Exit code: run_queue's if it is not 0 (2: results\STOP); 0 re-run done (or nothing to re-run); 3 still waiting after
# -MaxPolls; 31 another A-4 watcher runs; 40-48 a check refused (README.md section 5); 50 the acceptance refused after the
# re-run (the re-run itself ended; README.md section 7).
#   powershell -NoProfile -ExecutionPolicy Bypass -File confirm\rerun_a4\watch_and_rerun_a4.ps1 -Root .
param(
    [string]$Root = '.',
    [int]$PollSec = 300,
    [int]$MaxPolls = 0,                    # 0: wait without limit
    [string]$Python = '',                  # default <Root>\.venv\nestor\Scripts\python.exe (the chain's)
    [int]$GpuIdleMiB = 4096,               # wait while the GPU holds more memory than this (0: no GPU check)
    [int]$ExpectAmendRecords = 16,         # freeze\amendments.jsonl before A-4: A-1 11, A-2 4, A-3 1 record(s)
    [int]$ExpectCodeFiles = 14,            # amended code-record files before A-4 (13 before A-3)
    [switch]$ManualA4,                     # do not amend / re-install: wait until A-4 is installed by hand (README section 6)
    [switch]$NoPowerLogger,                # rehearsal only
    [switch]$DryRunQueue,                  # rehearsal only: run_queue --dry-run (runs nothing, writes no state)
    [switch]$AcceptDryRunA3                # rehearsal only: accept an A-3 result written with -DryRunQueue
)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location $Root
$Utf8 = New-Object System.Text.UTF8Encoding $false
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$A4 = Join-Path $Root 'confirm\rerun_a4'
$LogFile = Join-Path $Root 'logs\a4_rerun.log'
$WatchLock = Join-Path $A4 'WATCH_A4.lock'
$ResultFile = Join-Path $A4 'rerun_a4_result.json'
$AccFile = Join-Path $Root 'results\a4_acceptance.json'
$A3Lock = Join-Path $Root 'confirm\rerun\WATCH.lock'
$A3Result = Join-Path $Root 'confirm\rerun\rerun_result.json'
$ChainLock = Join-Path $Root 'results\chain6b\LOCK'
$ConfirmDone = Join-Path $Root 'results\chain6b\confirm.done'
$StateFile = Join-Path $Root 'results\queue_state_confirm.json'
$QueueLog = Join-Path $Root 'logs\queue_confirm\queue.log'
$Util = Join-Path $Root 'chain6b_util.py'
$AmendLog = Join-Path $Root 'freeze\amendments.jsonl'
$CodeRecord = Join-Path $Root 'logs\chain6b_code_sha256.json'
$Decision = 'phase6/amendment_A4_post_freeze.md'
$Staged = Join-Path $Root 'phase6\amendment_A4_post_freeze.md'
$A3Decision = 'phase6/amendment_A3_post_freeze.md'
$A3Staged = Join-Path $Root 'phase6\amendment_A3_post_freeze.md'
$Jobs = @('c3_freeze_moe_b1_german_s103', 'c3_freeze_nestor_german_s103')
$AuditJobs = @('audit_moe_b1', 'audit_moe_b2')
$Terminal = @('done', 'pass', 'fail', 'failed', 'skipped')
$Reason = 'A-4: the bitwise audit of the C3 freeze-mode runs at queue positions 154 and 155 failed only because the pseudo-query of the first depth-attention read (one source, hence an exactly zero gradient) stayed bitwise unchanged; both runs re-run after the A-3 audit re-run with the same seeds from confirm/rerun_a4/queue_confirm_a4_rerun.json (allow_unchanged) and are accepted only if trainable_unchanged equals the named list; no frozen file is changed'
$Approver = 'operator (delegated by the project owner)'
if (-not $Python) { $Python = Join-Path $Root '.venv\nestor\Scripts\python.exe' }
$Pin = [ordered]@{                        # reviewed for A-4 (README.md section 2)
    'confirm\rerun_a4\queue_confirm_a4_rerun.json' = '4056fa2d56666c2b81170ae7b4d37a9e657a8462d441b6864f925c61ea2934a1'
    'confirm\rerun_a4\amendment_A4_post_freeze.md' = '3d7aae95950a3657442016fde45b99f48c161a4e9f20f9b396924df6836cf036'
    'confirm\rerun_a4\expected_unchanged.json' = '33f22a4d6e7e6029ab812fee773f77b12ff986418989659b9df1f327ea6c1c26'
    'confirm\rerun_a4\a4_acceptance.py' = '09eb601f8b8a2c95e13a57788b990192574d38bb49d125c5e4cbdb33861fe5a9'
    'confirm\queue_confirm.json' = '47c95918229a8a2ed3eca4adc49d7d80fc89f2ad1cc4ebad0c2935622000269b'
    'run_queue.py' = 'f535c41a19de90802c9d104865da5e0794b363dcc24f4f883b05ee358f172890'
    'subset_adapt.py' = '4b86df24add074b3fc3d04c318a837974403cb90e5d6b90cdbec97c2d9fe4e27'
    'confirm\rerun\amendment_A3_post_freeze.md' = '48b8d72a52acaba9d4884a2660d1b2e8e020a395e0daffb3647caf4b68043a12'
}
$DecSha = $Pin['confirm\rerun_a4\amendment_A4_post_freeze.md']
$A3Sha = $Pin['confirm\rerun\amendment_A3_post_freeze.md']

function Log([string]$msg) {
    $line = "[watch-A4] $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $msg"
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
function Read-Json([string]$p) { return (Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json) }
function N([object]$x) { if ($null -eq $x) { return 0 } return @($x).Count }
function Rel([string]$p) { $f = [System.IO.Path]::GetFullPath($p); if ($f.StartsWith($Root + '\', [System.StringComparison]::OrdinalIgnoreCase)) { return $f.Substring($Root.Length + 1).Replace('\', '/') } return $f }
function Test-Pins {
    foreach ($rel in $Pin.Keys) {
        $p = Join-Path $Root $rel
        if (-not (Test-Path -LiteralPath $p)) { return "$rel is missing" }
        $h = Sha $p
        if ($h -ne $Pin[$rel]) { return "$rel is $h, not the reviewed $($Pin[$rel])" }
    }
    return ''
}
function Test-Expected {
    $e = Read-Json (Join-Path $A4 'expected_unchanged.json')
    foreach ($j in $Jobs) {
        $x = $e.jobs.PSObject.Properties[$j]
        if (-not $x) { return "expected_unchanged.json has no entry for $j" }
        $l = $x.Value.expected_trainable_unchanged
        if ($l -is [string] -or (N $l) -lt 1) { return "the named list of $j is '$l' (pending): README section 3" }
        foreach ($f in @(@('audit.json', 'audit_sha256'), @('subset_manifest.json', 'subset_manifest_sha256'), @('run_env.json', 'run_env_sha256'))) {
            $cp = Join-Path $A4 "evidence\first_attempt\$j\$($f[0])"
            if (-not (Test-Path -LiteralPath $cp)) { return "first-attempt copy evidence\first_attempt\$j\$($f[0]) is missing" }
            if ((Sha $cp) -ne $x.Value.first_attempt.($f[1])) { return "first-attempt copy evidence\first_attempt\$j\$($f[0]) is not the recorded one" }
        }
    }
    return ''
}
$script:CimNote = $false
function Get-QueueProcesses {
    try {
        $p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop | Where-Object {
                $_.CommandLine -match 'run_queue\.py|run_exp\.py|eval_windows\.py|assumption_audit\.py|audit_shim\.py|subset_adapt\.py|analyze_confirm\.py|analysis_notorch\.py' })
        if ($p.Count -gt 0) { return (($p | ForEach-Object { "python PID $($_.ProcessId)" }) -join ', ') }
    } catch {
        if (-not $script:CimNote) { Log "note: process list not readable ($($_.Exception.Message)); relying on the A-3 and chain locks, confirm.done, queue.log and the GPU"; $script:CimNote = $true }
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
$QueueIds = @((Read-Json (Join-Path $Root 'confirm\queue_confirm.json')).jobs | ForEach-Object { $_.id })
function Get-Status {
    $why = New-Object System.Collections.Generic.List[string]
    $fatal = ''
    $st = $null
    try { $st = Read-Json $StateFile } catch { $why.Add("queue state not readable: $($_.Exception.Message)") }
    # the A-3 watcher first
    $a3 = ''
    if (Test-Path -LiteralPath $A3Lock) { $a3 = ([string](Get-Content -LiteralPath $A3Lock -Raw)).Trim() }
    if (-not $a3) { $why.Add('confirm\rerun\WATCH.lock missing or empty (the A-3 watcher has not started)') }
    elseif ($a3 -match '^\d+$') {
        if (Get-Process -Id ([int]$a3) -ErrorAction SilentlyContinue) { $why.Add("the A-3 watcher (PID $a3) runs") }
        else { $why.Add("confirm\rerun\WATCH.lock names PID $a3, which does not run (the A-3 watcher ended without releasing it: confirm\rerun\README.md)") }
    } elseif ($a3 -match '^released .*\(exit (-?\d+)\)') {
        if ($Matches[1] -ne '0') { $why.Add("the A-3 watcher ended with exit $($Matches[1]) (confirm\rerun\README.md section 5); waiting for a successful A-3 run") }
        elseif (Test-Path -LiteralPath $A3Result) {
            try {
                $r3 = Read-Json $A3Result
                if ($r3.dry_run -and -not $AcceptDryRunA3) { $why.Add('confirm\rerun\rerun_result.json is from a dry run (nothing ran)') }
            } catch { $why.Add("confirm\rerun\rerun_result.json not readable: $($_.Exception.Message)") }
        } elseif ($st -and @($AuditJobs | Where-Object { [string]$st.PSObject.Properties[$_].Value.status -eq 'done' }).Count -eq $AuditJobs.Count) {
            # the A-3 watcher exits 0 without a result file when both audit jobs were already done
        } else { $why.Add('the A-3 watcher released its lock with exit 0 but confirm\rerun\rerun_result.json is missing') }
    } else { $why.Add("confirm\rerun\WATCH.lock not understood: $a3") }
    # the chain and the queue, as watch_and_rerun.ps1
    if (Test-Path -LiteralPath (Join-Path $Root 'results\STOP')) { $why.Add('results\STOP present') }
    if (Test-Path -LiteralPath (Join-Path $Root 'PAUSE')) { $why.Add('PAUSE present') }
    $lockTxt = ''
    if (Test-Path -LiteralPath $ChainLock) { $lockTxt = ([string](Get-Content -LiteralPath $ChainLock -Raw)).Trim() }
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
    foreach ($l in $tail) {
        if ($l -match '^\[queue\] \S+ \S+ start (\S+) ') { $open = $Matches[1] }
        elseif ($open -and $l -match ('^\[queue\] \S+ \S+ end ' + [regex]::Escape($open) + ':')) { $open = '' }
    }
    if ($open) { $why.Add("queue.log: job $open started and has not ended") }
    $pr = Get-QueueProcesses
    if ($pr) { $why.Add("queue process running: $pr") }
    $mib = Get-GpuMiB
    if ($mib -ge 0 -and $mib -gt $GpuIdleMiB) { $why.Add("GPU memory in use $mib MiB > $GpuIdleMiB MiB") }
    $js = [ordered]@{}
    if ($st) {
        $missing = @()
        foreach ($id in $QueueIds) {
            $e = $st.PSObject.Properties[$id]
            $s = if ($e) { [string]$e.Value.status } else { '' }
            if ($Terminal -notcontains $s) { $missing += $id }
            if ($Jobs -contains $id) { $js[$id] = $(if ($e) { $e.Value } else { $null }) }
        }
        if ($missing.Count -gt 0) {
            $why.Add("$($missing.Count) of $($QueueIds.Count) jobs without a final state")
            if ($done -and $lockTxt -match '^released .*\(exit 0\)' -and -not $open) { $fatal = "confirm.done exists and the chain released its lock, but $($missing.Count) jobs have no final state (e.g. $($missing[0]))" }
        }
    }
    return [pscustomobject]@{ ready = ($why.Count -eq 0); why = ($why -join '; '); fatal = $fatal; jobs = $js; gpuMiB = $mib }
}
function Get-Recs([string]$id) {
    if (-not (Test-Path -LiteralPath $AmendLog)) { return @() }
    $r = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
    return @($r | Where-Object { $_.amendment -eq $id })
}
function Get-CodeRecordFiles([string[]]$out) {
    foreach ($l in $out) { if ($l -match 'amended after the freeze: code record \(.*\): (\d+) file\(s\)') { return [int]$Matches[1] } }
    return 0
}
function Amend-Args([string]$decision, [bool]$dry) {
    $a = @($Util, 'amend', '--root', $Root, '--id', 'A-4', '--reason', $Reason, '--approved-by', $Approver, '--decision', $decision)
    if ($dry) { $a += '--dry-run' }
    return ,$a
}
function Undo-Staging([string]$why) {
    if (Test-Path -LiteralPath $Staged) {
        $bk = Join-Path $A4 ('rollback_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        New-Item -ItemType Directory -Force -Path $bk | Out-Null
        Move-Item -LiteralPath $Staged -Destination $bk
        Log "staging undone ($why): phase6\amendment_A4_post_freeze.md moved to $bk"
    }
}
function Test-Installed([string]$id, [string]$decision, [string]$sha, [string]$staged) {
    $rs = @(Get-Recs $id)
    if ($rs.Count -eq 0) { return 'absent' }
    $ok = @($rs | Where-Object { $_.path -eq $decision -and $_.new_sha256 -eq $sha -and $_.decision -eq $decision -and $_.decision_sha256 -eq $sha })
    if ($ok.Count -lt 1) { return 'other' }
    if (-not (Test-Path -LiteralPath $staged) -or (Sha $staged) -ne $sha) { return 'nofile' }
    $rec = Read-Json $CodeRecord
    $f = $rec.files.PSObject.Properties[$decision]
    if (-not $f -or $f.Value -ne $sha) { return 'recorded' }      # amended, not yet re-installed
    return 'installed'
}
function Test-A4Installed { return (Test-Installed 'A-4' $Decision $DecSha $Staged) }
function Test-FirstAttempt([string]$j, $e) {
    # the state entry and the live root must be the recorded first attempt with exactly the A-4 audit pattern
    $x = $Exp.jobs.PSObject.Properties[$j].Value
    $fa = $x.first_attempt
    if ([string]$e.status -ne 'failed' -or [string]$e.error -ne 'bitwise audit failed') { return "$j has state '$($e.status)' ($($e.error))" }
    $jr = [string]$e.root
    if ([System.IO.Path]::GetFullPath($jr).ToLower() -ne [System.IO.Path]::GetFullPath((Join-Path $Root $fa.root)).ToLower()) { return "$j root $jr is not the recorded $($fa.root)" }
    foreach ($f in @(@('audit.json', 'audit_sha256'), @('subset_manifest.json', 'subset_manifest_sha256'), @('run_env.json', 'run_env_sha256'))) {
        $p = Join-Path $jr $f[0]
        if (-not (Test-Path -LiteralPath $p)) { return "$j $($fa.root)/$($f[0]) is missing" }
        if ((Sha $p) -ne $fa.($f[1])) { return "$j $($fa.root)/$($f[0]) is not the recorded first attempt" }
    }
    $au = Read-Json (Join-Path $jr 'audit.json')
    if ((N $au.frozen_mismatch) -ne 0 -or (N $au.buffer_mismatch) -ne 0 -or (N $au.optimizer_state.violations) -ne 0 -or $au.allow_unchanged -or $au.ok) {
        return "$j first-attempt audit is not the A-4 pattern (frozen, buffer or optimizer-state findings, or allow_unchanged/ok set)" }
    if ((@($au.trainable_unchanged) -join '|') -ne (@($x.expected_trainable_unchanged) -join '|')) { return "$j first-attempt trainable_unchanged is not the named list" }
    $final = [string]$e.final
    if (-not $final -or -not (Test-Path -LiteralPath $final)) { return "$j final checkpoint of the first attempt is missing" }
    $outRoot = Join-Path $Root $x.out_root
    $others = @(Get-ChildItem -LiteralPath $outRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -match '^m\d+c?$' -and $_.FullName.ToLower() -ne [System.IO.Path]::GetFullPath($jr).ToLower() -and
            @(Get-ChildItem -LiteralPath $_.FullName -Recurse -Filter 'model_final.pt' -ErrorAction SilentlyContinue).Count -gt 0 })
    if ($others.Count -gt 0) { return "$j has another attempt root with a final checkpoint: $(($others | ForEach-Object { $_.Name }) -join ', ')" }
    return ''
}
function Install-A4 {
    $s0 = Test-A4Installed
    if ($s0 -eq 'other') { Quit 46 "freeze\amendments.jsonl holds A-4 records that are not this decision file ($DecSha)" }
    if ($s0 -eq 'nofile') { Quit 46 'A-4 is recorded but phase6\amendment_A4_post_freeze.md is missing or differs' }
    if ($s0 -eq 'installed') {
        $fv = Run-Py 'freeze-verify (A-4 found installed)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
        if ($fv.rc -ne 0) { Quit 46 'A-4 is installed but freeze-verify fails (README section 5)' }
        Log 'A-4 already installed: record, phase6 file and code record match; freeze-verify passes'
        return
    }
    if ($s0 -eq 'recorded') { Quit 45 'A-4 is recorded but not re-installed (freeze-verify exits 23): README section 6, step 7' }
    $s3 = Test-Installed 'A-3' $A3Decision $A3Sha $A3Staged
    if ($s3 -ne 'installed') { Quit 44 "A-3 is not installed ($s3): A-4 is installed after A-3" }
    $fv0 = Run-Py 'freeze-verify (before A-4)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    if ($fv0.rc -ne 0) { Quit 44 'freeze-verify fails before A-4: nothing written' }
    $n0 = Get-CodeRecordFiles $fv0.out
    $before = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    if ($before -ne $ExpectAmendRecords -or $n0 -ne $ExpectCodeFiles) {
        Quit 44 "freeze\amendments.jsonl has $before records and the code record $n0 amended files; expected $ExpectAmendRecords and $ExpectCodeFiles after A-3 (other values only after review: -ExpectAmendRecords, -ExpectCodeFiles): nothing written" }
    if (Test-Path -LiteralPath $Staged) {
        if ((Sha $Staged) -ne $DecSha) { Quit 42 'phase6\amendment_A4_post_freeze.md exists with other content: nothing written' }
        Log 'phase6\amendment_A4_post_freeze.md is already staged (same sha256); the staged dry run below must show exactly this file'
    } else {
        $d0 = Run-Py 'amend --dry-run (before staging)' (Amend-Args 'confirm/rerun_a4/amendment_A4_post_freeze.md' $true)
        $rows0 = @($d0.out | Where-Object { $_ -match '^\[amend\] A-4: ' })
        if ($d0.rc -ne 0 -or $rows0.Count -ne 0 -or -not ($d0.out -contains '[amend] nothing differs from the frozen code record or the amended state: nothing recorded')) {
            Quit 41 'the code set differs from the amended state before A-4 is staged (something else would be recorded): nothing written' }
        Copy-Item -LiteralPath (Join-Path $A4 'amendment_A4_post_freeze.md') -Destination $Staged
        if ((Sha $Staged) -ne $DecSha) { Undo-Staging 'copy check'; Quit 42 'copy check of the staged decision file failed: nothing recorded' }
        Log "staged phase6\amendment_A4_post_freeze.md (sha256 $DecSha)"
    }
    $d1 = Run-Py 'amend --dry-run (staged)' (Amend-Args $Decision $true)
    $rows1 = @($d1.out | Where-Object { $_ -match '^\[amend\] A-4: ' })
    $want = "[amend] A-4: phase6/amendment_A4_post_freeze.md  absent -> $($DecSha.Substring(0, 12))  (staged)"
    if ($d1.rc -ne 0 -or $rows1.Count -ne 1 -or $rows1[0] -cne $want -or -not ($d1.out -ccontains '[amend] dry run: 1 record(s) not written')) {
        Undo-Staging 'dry run'; Quit 42 'the dry run after staging is not exactly the one expected record: nothing recorded' }
    $am = Run-Py 'amend --id A-4' (Amend-Args $Decision $false)
    $after = @(Get-Content -LiteralPath $AmendLog -Encoding UTF8 | Where-Object { $_.Trim() }).Count
    $r4 = @(Get-Recs 'A-4')
    if ($am.rc -ne 0 -or $after -ne ($before + 1) -or $r4.Count -ne 1 -or $r4[0].id -ne 'A-4.1' -or $r4[0].path -ne $Decision -or
        $r4[0].old_sha256 -ne 'absent' -or $r4[0].new_sha256 -ne $DecSha -or $r4[0].decision_sha256 -ne $DecSha) {
        Quit 43 "amend did not append exactly the expected record ($before -> $after lines): README section 5" }
    Log "recorded A-4.1 ($($r4[0].utc)): $Decision absent -> $DecSha; freeze\amendments.jsonl $before -> $after records"
    $fv1 = Run-Py 'freeze-verify (recorded, not re-installed; exit 23 by design)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    if ($fv1.rc -ne 23) { Log "note: freeze-verify exit $($fv1.rc) between amend and re-install (23 expected)" }
    $in = Run-Py 'install --reinstall' @($Util, 'install', '--root', $Root, '--reinstall')
    if ($in.rc -ne 0) { Quit 45 're-install failed: A-4 is recorded and freeze-verify exits 23 until the re-install (README section 6, step 7)' }
    $mk = Run-Py 'mark install' @($Util, 'mark', '--root', $Root, '--step', 'install', '--info', 'installed (reinstall=True; amendment A-4, confirm/rerun_a4/watch_and_rerun_a4.ps1)')
    if ($mk.rc -ne 0) { Quit 45 'mark install failed after the re-install' }
    $fv2 = Run-Py 'freeze-verify (A-4 installed)' @($Util, 'freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    $n2 = Get-CodeRecordFiles $fv2.out
    if ($fv2.rc -ne 0 -or $n2 -ne ($n0 + 1) -or (Test-A4Installed) -ne 'installed') {
        Quit 46 "freeze-verify after the re-install: exit $($fv2.rc), amended code-record files $n2 (expected $($n0 + 1))" }
    Log "A-4 installed: record A-4.1, code record re-installed, freeze-verify passes ($n2 amended code-record files, $n0 before)"
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
    $note = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') (watch_and_rerun_a4.ps1, amendment A-4): logging resumes for the re-run of " +
            "c3_freeze_moe_b1_german_s103 and c3_freeze_nestor_german_s103 after the A-3 audit re-run in gpu_power_$stamp.csv (same command and columns; logger PID $($p.Id))."
    $readme = Join-Path $dir 'README_power_log.txt'
    $pre = ''
    if (Test-Path -LiteralPath $readme) {
        $b = [System.IO.File]::ReadAllBytes($readme)
        if ($b.Length -gt 0 -and $b[$b.Length - 1] -ne 10) { $pre = "`r`n" }
    }
    [System.IO.File]::AppendAllText($readme, $pre + $note + "`r`n", [System.Text.Encoding]::ASCII)
    Log "power logger started: PID $($p.Id) -> logs\power\gpu_power_$stamp.csv; note appended to logs\power\README_power_log.txt"
}
function Write-Result([hashtable]$res) {
    if (Test-Path -LiteralPath $ResultFile) {
        $bk = Join-Path $A4 ('rerun_a4_result.previous_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json')
        Move-Item -LiteralPath $ResultFile -Destination $bk
        Log "previous confirm\rerun_a4\rerun_a4_result.json kept as $(Split-Path $bk -Leaf)"
    }
    [System.IO.File]::WriteAllText($ResultFile, ($res | ConvertTo-Json -Depth 8) + "`r`n", $Utf8)
    Log "wrote confirm\rerun_a4\rerun_a4_result.json ($(Sha $ResultFile))"
}
function Run-Acceptance {
    if (Test-Path -LiteralPath $AccFile) { Log "results\a4_acceptance.json exists ($(Sha $AccFile)): not rewritten"; return [ordered]@{ exit = $null; path = 'results/a4_acceptance.json'; sha256 = (Sha $AccFile); note = 'existed before this run' } }
    $ac = Run-Py 'a4_acceptance' @((Join-Path $A4 'a4_acceptance.py'), '--root', $Root)
    $out = [ordered]@{ exit = $ac.rc; path = 'results/a4_acceptance.json'; sha256 = $null; accepted = $null }
    if ($ac.rc -eq 0 -and (Test-Path -LiteralPath $AccFile)) {
        $out.sha256 = Sha $AccFile
        $acc = (Read-Json $AccFile).accepted
        $out.accepted = [ordered]@{}
        foreach ($j in $Jobs) {
            $v = [bool]$acc.$j
            $out.accepted[$j] = $v
            if ($v) { Log "acceptance: $j accepted" } else { Log "WARNING: acceptance: $j NOT accepted: its values are withheld (README section 7)" }
        }
    }
    return $out
}

# ------------------------------------------------------------------------------------------------ start
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'logs') | Out-Null
if (Test-Path -LiteralPath $WatchLock) {
    $old = ([string](Get-Content -LiteralPath $WatchLock -Raw)).Trim()
    if ($old -match '^\d+$' -and $old -ne "$PID" -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "[watch-A4] another A-4 watcher (PID $old) runs"; exit 31 }
}
[System.IO.File]::WriteAllText($WatchLock, "$PID`r`n", $Utf8)
Log "##### start (PID $PID): root $Root, python $Python, poll $PollSec s, max polls $MaxPolls, GPU idle limit $GpuIdleMiB MiB, expected before A-4: $ExpectAmendRecords records / $ExpectCodeFiles amended code files, manual A-4 $ManualA4, power logger $(-not $NoPowerLogger), dry-run queue $DryRunQueue, accept dry-run A-3 $AcceptDryRunA3"
if (-not (Test-Path -LiteralPath $Python)) { Quit 40 "python $Python not found (nothing written)" }
if (-not (Test-Path -LiteralPath $Util)) { Quit 40 "$Util not found (nothing written)" }
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed: $bad (nothing written)" }
Log "pinned files verified: $(($Pin.Keys | ForEach-Object { "$_ $($Pin[$_].Substring(0, 12))" }) -join ', ')"
$bad = Test-Expected
if ($bad) { Quit 40 "named lists: $bad (nothing written)" }
$Exp = Read-Json (Join-Path $A4 'expected_unchanged.json')
Log "named lists and first-attempt copies verified: $(($Jobs | ForEach-Object { "$_ $(@($Exp.jobs.$_.expected_trainable_unchanged) -join ',')" }) -join '; ')"
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

# ------------------------------------------------------------------------------------------------ wait for A-3 and an idle queue
$n = 0; $lastWhy = ''
while ($true) {
    $s = Get-Status
    if ($s.fatal) { Quit 48 $s.fatal }
    if ($s.ready) {
        $confirm = [Math]::Min($PollSec, 60)
        Log "A-3 ended and the queue looks idle; confirming in $confirm s"
        Start-Sleep -Seconds $confirm
        $s = Get-Status
        if ($s.ready) { break }
    }
    if ($s.why -ne $lastWhy -or ($n % 12) -eq 0) { Log "waiting: $($s.why)"; $lastWhy = $s.why }
    $n++
    if ($MaxPolls -gt 0 -and $n -ge $MaxPolls) { Quit 3 "still waiting after $n polls" }
    Start-Sleep -Seconds $PollSec
}
Log "A-3 ended and the main queue finished: $($QueueIds.Count) jobs with a final state; GPU memory $($s.gpuMiB) MiB"
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed: $bad (nothing written)" }
$rerun = @(); $info = [ordered]@{}
foreach ($j in $Jobs) {
    $e = $s.jobs[$j]
    $st_ = [string]$e.status
    if ($st_ -eq 'failed' -and [string]$e.error -eq 'bitwise audit failed') {
        $why = Test-FirstAttempt $j $e
        if ($why) { Quit 47 "first attempt check: $why (nothing written)" }
        $rerun += $j
        $info[$j] = 'first attempt failed the frozen audit with exactly the named list: re-run'
    } elseif ($st_ -eq 'done') {
        $ap = Join-Path ([string]$e.root) 'audit.json'
        $au = $(if (Test-Path -LiteralPath $ap) { Read-Json $ap } else { $null })
        if ($au -and $au.allow_unchanged) { $info[$j] = 'already re-run under A-4' }
        else { Quit 48 "$j is 'done' without an A-4 audit (allow_unchanged not set): not the recorded first attempt (README section 5)" }
    } else { Quit 48 "$j has state '$st_' ($($e.error)): not the A-4 case (README section 5)" }
    Log "${j}: $($info[$j])"
}
if ($rerun.Count -eq 0) {
    if ((Test-A4Installed) -ne 'installed') { Quit 46 'both jobs are done under A-4 but A-4 is not installed' }
    if (Test-Path -LiteralPath $ResultFile) { Quit 0 'both jobs were already re-run under A-4: nothing to do (confirm\rerun_a4\rerun_a4_result.json kept)' }
    $acc = Run-Acceptance
    Write-Result @{ utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); run_queue_exit = $null; dry_run = $false; rerun = @(); note = 'both jobs were already done under A-4 when this watcher ran'; acceptance = $acc }
    Quit $(if ($acc.exit -eq 0 -or $null -eq $acc.exit) { 0 } else { 50 }) 'nothing to re-run'
}

# ------------------------------------------------------------------------------------------------ amendment A-4
if ($ManualA4) {
    $k = 0
    while ((Test-A4Installed) -ne 'installed') {
        if (($k % 12) -eq 0) { Log "manual A-4: waiting for the amend and re-install (README section 6); now: $(Test-A4Installed)" }
        $k++
        if ($MaxPolls -gt 0 -and $k -ge $MaxPolls) { Quit 3 "manual A-4: still not installed after $k polls" }
        Start-Sleep -Seconds $PollSec
    }
}
Install-A4

# ------------------------------------------------------------------------------------------------ keep the first attempts, then re-run
$s = Get-Status
if (-not $s.ready) { Quit 48 "A-3 or the queue no longer look finished: $($s.why)" }
$bad = Test-Pins
if ($bad) { Quit 40 "pinned file check failed before the re-run: $bad" }
foreach ($j in $rerun) { $why = Test-FirstAttempt $j $s.jobs[$j]; if ($why) { Quit 47 "first attempt check before the re-run: $why" } }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$ev = Join-Path $A4 "evidence\before_rerun_$stamp"
New-Item -ItemType Directory -Force -Path $ev | Out-Null
$copies = @(@($StateFile, (Join-Path $ev 'queue_state_confirm.json')), @($QueueLog, (Join-Path $ev 'queue.log')))
foreach ($j in $rerun) {
    $e = $s.jobs[$j]
    $dj = Join-Path $ev $j
    New-Item -ItemType Directory -Force -Path $dj | Out-Null
    foreach ($f in @('audit.json', 'subset_manifest.json', 'run_env.json')) { $copies += ,@((Join-Path ([string]$e.root) $f), (Join-Path $dj $f)) }
    $names = @(@($e.attempts) | ForEach-Object { "$($j)_m$($_.micro)$(if ($_.ckpt) { 'c' } else { '' }).log" }) + @("$($j)_audit.log")
    $names += @(Get-ChildItem -LiteralPath (Join-Path $Root 'logs\queue_confirm') -Filter "$($j)_eval_*.log" -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
    foreach ($nm in ($names | Select-Object -Unique)) { $copies += ,@((Join-Path $Root "logs\queue_confirm\$nm"), (Join-Path $dj $nm)) }
}
foreach ($c in $copies) {
    if (-not (Test-Path -LiteralPath $c[0])) { Log "evidence: $(Rel $c[0]) not present"; continue }
    Copy-Item -LiteralPath $c[0] -Destination $c[1]
    if ((Sha $c[0]) -ne (Sha $c[1])) { Quit 47 "evidence copy of $(Rel $c[0]) failed" }
    Log "evidence: $(Rel $c[0]) -> $(Rel $c[1]) ($(Sha $c[1]))"
}
$moved = [ordered]@{}
$epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
foreach ($j in $rerun) {
    $src = [System.IO.Path]::GetFullPath([string]$s.jobs[$j].root)
    $dst = "$($src)_a4failed$epoch"
    if (Test-Path -LiteralPath $dst) { Quit 47 "$(Rel $dst) exists: nothing moved for $j" }
}
foreach ($j in $rerun) {
    $src = [System.IO.Path]::GetFullPath([string]$s.jobs[$j].root)
    $dst = "$($src)_a4failed$epoch"
    Move-Item -LiteralPath $src -Destination $dst
    if ((Test-Path -LiteralPath $src) -or -not (Test-Path -LiteralPath (Join-Path $dst 'audit.json')) -or
        (Sha (Join-Path $dst 'audit.json')) -ne $Exp.jobs.$j.first_attempt.audit_sha256) { Quit 47 "moving the first attempt's root $(Rel $src) to $(Rel $dst) failed" }
    $moved[$j] = [ordered]@{ from = (Rel $src); to = (Rel $dst) }
    Log "first attempt kept: $(Rel $src) -> $(Rel $dst)"
}
if (-not $NoPowerLogger) { Start-PowerLogger }
$qa = @((Join-Path $Root 'run_queue.py'), (Join-Path $A4 'queue_confirm_a4_rerun.json'), '--only', ($rerun -join ','), '--retry-failed')
if ($DryRunQueue) { $qa += '--dry-run' }
$r = Run-Py 'run_queue (A-4 re-run)' $qa
$res = [ordered]@{ utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); run_queue_exit = $r.rc; dry_run = [bool]$DryRunQueue;
                   rerun = $rerun; evidence = (Rel $ev); moved_roots = $moved; jobs = [ordered]@{}; acceptance = $null }
$st = Read-Json $StateFile
$summary = @()
foreach ($j in $Jobs) {
    $e = $st.PSObject.Properties[$j]
    $v = $(if ($e) { $e.Value } else { $null })
    $res.jobs[$j] = [ordered]@{ action = $info[$j]; status = $(if ($v) { $v.status } else { $null }); audit_ok = $(if ($v) { $v.audit_ok } else { $null });
                                error = $(if ($v) { $v.error } else { $null }); hours = $(if ($v) { $v.hours } else { $null }); root = $(if ($v -and $v.root) { Rel ([string]$v.root) } else { $null }) }
    $summary += "$j $($res.jobs[$j].status)"
}
if ($DryRunQueue) { Log 'dry run: nothing ran and the state was not written; the acceptance is not run'; $res.acceptance = [ordered]@{ exit = $null; note = 'dry run' } }
elseif ($r.rc -ne 0) { Log "run_queue exit $($r.rc): the acceptance is not run (README section 7)"; $res.acceptance = [ordered]@{ exit = $null; note = "run_queue exit $($r.rc)" } }
else { $res.acceptance = Run-Acceptance }
Write-Result $res
$code = $r.rc
if ($code -eq 0 -and -not $DryRunQueue -and $res.acceptance.exit -ne 0) { $code = 50 }
Quit $code "re-run finished: run_queue exit $($r.rc); $($summary -join '; ')$(if ($DryRunQueue) { ' [dry run: nothing ran, the state is not written]' })"
