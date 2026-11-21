# chain6b.ps1 -- post-G0 pipeline of the design (sec. 4 items 5-9). Windows PowerShell 5.1, ASCII only.
# Run (after the GPU audit, the stage-A spot check and the G0 re-run with its gate):
#     powershell -ExecutionPolicy Bypass -File phase6\chain6b.ps1
# Steps (resumable: a finished step leaves results\chain6b\<step>.done and is skipped on the next start):
#   precheck        G0 re-run vs the registered gate record: every decision must be equal
#   install         phase-5 + phase-6 code into the root, sha256 code record logs\chain6b_code_sha256.json
#   pins            pinned checkpoints exist with the recorded size (sha256 with -VerifyPinsFull) and the G0 record
#                   used exactly the pinned base checkpoints (checked at every start)
#   stageB          reproduction stage B (run_continual.py B), then the completeness check (39 results)
#   stageB_windows  per-window re-evaluation of stage B (stageB_windows.py)
#   stageB_analyze  analyze_stageB.py (torch-free)
#   g1g2            hooks_after_B.ps1: GPU pipeline audit, G1 (+ 192-step repeat), G2 and their gates
#   stageC          reproduction stage C (run_continual.py C), completeness check (8 results)
#   stageC13        prereg v1.3 stage-C inputs: NESTOR-books feasibility, dose draws, C1-DE MILP draws (sec. 4.12),
#                   confirm plan, forecast units, forecast-input queue (GPU), forecast files verified, environment
#                   snapshot logs\env (freeze hash list item 10) (exit 24: a non-queue sub-step failed or the
#                   envelope decision P-6 is required; the forecast-input queue returns run_queue's own exit code,
#                   as the confirm step does);
#                   -SpentGpuH H = GPU-h already spent (budget_v13 'spent' tier) for the envelope check, required
#                   outside -Mock
#   freeze          wait for FREEZE.json, then verify every hash it lists (prereg, plan, queue, forecasts, code)
#   confirm         run_queue.py on the frozen confirmatory + secondary queue named in FREEZE.json
# Hook points: hooks\before_<step>.ps1 and hooks\after_<step>.ps1 (optional; looked up when the step starts, so they
#   can be added while the chain runs). A hook ends with "exit 0" (continue), "exit 10" (pause: the chain stops
#   cleanly and resumes at the next start) or any other code (stop; chain exit 30).
# Control: a file PAUSE in the root stops the chain before its next step (exit 0); run_queue's STOP file (next to the
#   queue state) stops a queue before its next job; a second instance refuses to start (results\chain6b\LOCK).
# Code = registration: before every step after install the code set is compared with the install record; a changed,
#   added or missing code/config file stops the chain (exit 21) until it is re-installed on purpose (-Reinstall).
#   hooks\*.ps1 are excluded from that check (they may be added while the chain runs); each executed hook is logged
#   with its sha256 in logs\chain6b.log.
# Exit codes: 0 done or paused; 11 precheck failed or G0 re-run record missing; 12 pins; 20 stage incomplete;
#   21 code drift; 22 install refused; 23 freeze record invalid; 24 stage-C inputs (prereg v1.3) failed or the
#   envelope decision is required (stageC13's forecast-input queue: run_queue's code); 30 stopped by a hook; 31 another instance runs;
#   any other code is the step's own (hooks_after_B: 3/4/5 gate outcomes, 6 audit, 7 plan inputs, 8 plan hash;
#   run_queue: its exit code).
param(
    [string]$Root = '.',
    [switch]$Mock,
    [switch]$Reinstall,
    [switch]$SkipPrecheck,
    [switch]$VerifyPinsFull,
    [int]$FreezePollSec = 300,
    [string]$StopAfter = '',
    [string]$Python = '',
    [string]$CondaLibBin = '<conda>\envs\nestor\Library\bin',
    [double]$SpentGpuH = -1,
    [switch]$RetryFailed
)
$ErrorActionPreference = 'Continue'
Set-Location $Root
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'          # every child process inherits BelowNormal

# ------------------------------------------------------------------------------------------------ environment
if (Test-Path "$Root\msvc_env.ps1") { . "$Root\msvc_env.ps1" }  # MSVC + SDK for torch.compile / triton
$env:TRITON_CACHE_DIR = "$Root\tc"                             # overrides msvc_env (cwd-relative there)
$env:TORCHINDUCTOR_CACHE_DIR = "$Root\ic"
$env:TIKTOKEN_CACHE_DIR = "$Root\data\tiktoken"
foreach ($v in @('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')) {
    Set-Item -Path "env:$v" -Value '12'
}
$env:TORCHINDUCTOR_COMPILE_THREADS = '1'   # 2026-11-13: >1 forces inductor's SubprocPool, which fails on Windows (pass_fds); 1 = torch's Windows default
$env:NESTOR_VRAM_GUARD = '1'
$env:NESTOR_VRAM_BUDGET_FRAC = '0.90'
$env:MEM_FRAC = '0.85'                                         # run_exp.py training memory fraction
$env:TEMP = "$Root\.tmp"
$env:TMP = "$Root\.tmp"
New-Item -ItemType Directory -Force "$Root\.tmp", "$Root\logs", "$Root\results\chain6b", "$Root\hooks" | Out-Null
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$Root;$Root\sitecustom"                    # sitecustom: Python 3.12 temp-dir ACL fix (host fact)
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:WANDB_MODE = 'disabled'
$env:NESTOR_CKPT_PINS = "$Root\ckpt_pins.json"
if ($Python) { $py = $Python } else { $py = "$Root\.venv\nestor\Scripts\python.exe" }
$log = "$Root\logs\chain6b.log"
$lock = "$Root\results\chain6b\LOCK"
$codeRecord = "$Root\logs\chain6b_code_sha256.json"

function Write-Log([string]$msg) {
    $line = "[chain6b] $(Get-Date -Format s) $msg"
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}
function Exit-Chain([int]$code) {
    Write-Log "exit $code"
    "released $(Get-Date -Format s) by $PID (exit $code)" | Out-File -Encoding ascii $lock    # overwrite, never delete (rule)
    exit $code
}
function Get-Tool([string]$name) {                             # root copy after install, phase6 copy before
    if (Test-Path "$Root\$name") { return "$Root\$name" }
    return "$Root\phase6\$name"
}
function Invoke-Py([string]$name, [string[]]$argv, [switch]$TorchFree) {
    $saved = $env:PATH
    if ($TorchFree -and (Test-Path $CondaLibBin)) { $env:PATH = "$CondaLibBin;$saved" }
    $slog = "$Root\logs\chain6b_$name.log"
    "##### $(Get-Date -Format s) $py $($argv -join ' ')" | Out-File -Append -Encoding utf8 $slog
    try {
        & $py @argv 2>&1 | Out-File -Append -Encoding utf8 $slog
        $rc = $LASTEXITCODE
    } finally { $env:PATH = $saved }
    return $rc
}
function Invoke-Util([string]$name, [string[]]$argv) {          # stdlib-only helper: no PATH prefix needed
    return (Invoke-Py $name (@((Get-Tool 'chain6b_util.py')) + $argv))
}
function Test-Done([string]$step) { return (Test-Path "$Root\results\chain6b\$step.done") }
function Invoke-Hook([string]$when, [string]$step) {
    $h = "$Root\hooks\$($when)_$step.ps1"
    if (-not (Test-Path $h)) { return 0 }
    Write-Log "hook $($when)_$step start (sha256 $((Get-FileHash $h -Algorithm SHA256).Hash.ToLower()))"
    $global:LASTEXITCODE = 0
    $out = & $h 2>&1
    $rc = $LASTEXITCODE
    if ($out) { $out | Out-File -Append -Encoding utf8 "$Root\logs\chain6b_hook_$($when)_$step.log" }
    Write-Log "hook $($when)_$step exit $rc"
    return $rc
}
function Resolve-Hook([int]$rc, [string]$where) {
    if ($rc -eq 0) { return }
    if ($rc -eq 10) { Write-Log "paused by hook ($where); start chain6b.ps1 again to resume"; Exit-Chain 0 }
    Write-Log "stopped by hook ($where), code $rc"
    Exit-Chain 30
}
function Start-Step([string]$step) {                           # $true if the step must run
    if (Test-Path "$Root\PAUSE") { Write-Log "PAUSE file present: stopping before $step"; Exit-Chain 0 }
    if ((Test-Path $codeRecord) -and ($step -ne 'install')) {
        $rc = Invoke-Util 'codecheck' @('codecheck', '--root', $Root)
        if ($rc -ne 0) { Write-Log "code drift before $step (logs\chain6b_codecheck.log)"; Exit-Chain 21 }
    }
    if (Test-Done $step) { Write-Log "skip $step (done)"; return $false }
    Resolve-Hook (Invoke-Hook 'before' $step) "before $step"
    Write-Log "=== $step"
    return $true
}
function Complete-Step([string]$step, [string]$info) {
    $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', $step, '--info', $info)
    Write-Log "--- $step done: $info"
    Resolve-Hook (Invoke-Hook 'after' $step) "after $step"
    if ($StopAfter -eq $step) { Write-Log "StopAfter $step"; Exit-Chain 0 }
}
function Stop-OnFail([int]$rc, [string]$what, [int]$code = -1) {
    if ($rc -eq 0) { return }
    if ($code -lt 0) { $code = $rc }
    Write-Log "$what failed (exit $rc)"
    Exit-Chain $code
}

# ------------------------------------------------------------------------------------------------ lock
if (Test-Path $lock) {
    $old = Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($old -and ($old -match '^\d+$') -and ($old -ne "$PID") -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "[chain6b] another chain6b (PID $old) is running"
        exit 31
    }
}
"$PID" | Out-File -Encoding ascii $lock
$mode = 'GPU'
if ($Mock) { $mode = 'MOCK (GPU commands replaced; non-GPU parts real)' }
Write-Log "##### chain6b start (PID $PID, $mode); python $py; OMP/MKL $env:OMP_NUM_THREADS, compile threads $env:TORCHINDUCTOR_COMPILE_THREADS, VRAM guard $env:NESTOR_VRAM_GUARD @ $env:NESTOR_VRAM_BUDGET_FRAC, MEM_FRAC $env:MEM_FRAC, priority $((Get-Process -Id $PID).PriorityClass)"

# ------------------------------------------------------------------------------------------------ precheck
if (-not $SkipPrecheck) {
    if (Start-Step 'precheck') {
        $reg = "$Root\g0\gate_record_registered.json"
        if (-not (Test-Path $reg)) { $reg = "$Root\phase6\gate_record_G0_registered.json" }
        $new = "$Root\g0\gate_record.json"
        if (-not (Test-Path $new)) { Write-Log "G0 re-run gate record $new missing"; Exit-Chain 11 }
        $rc = Invoke-Py 'precheck' @((Get-Tool 'g0_rerun_compare.py'), '--registered', $reg, '--rerun', $new, '--out', "$Root\logs\g0_rerun_compare.json")
        Stop-OnFail $rc 'G0 re-run decision comparison (a differing decision stops the pipeline)' 11
        Complete-Step 'precheck' 'all G0 decisions equal'
    }
}
# ------------------------------------------------------------------------------------------------ install
if ($Reinstall -or -not (Test-Done 'install')) {
    if (Test-Path "$Root\PAUSE") { Write-Log 'PAUSE file present: stopping before install'; Exit-Chain 0 }
    Resolve-Hook (Invoke-Hook 'before' 'install') 'before install'
    Write-Log '=== install'
    $iargs = @('install', '--root', $Root)
    if ($Reinstall) { $iargs += '--reinstall' }
    $rc = Invoke-Util 'install' $iargs
    Stop-OnFail $rc 'install' 22
    $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'install', '--info', "installed (reinstall=$Reinstall)")
    Write-Log '--- install done'
    Resolve-Hook (Invoke-Hook 'after' 'install') 'after install'
    if ($StopAfter -eq 'install') { Exit-Chain 0 }
} else {
    $rc = Invoke-Util 'codecheck' @('codecheck', '--root', $Root)
    if ($rc -ne 0) { Write-Log 'code drift since install (logs\chain6b_codecheck.log)'; Exit-Chain 21 }
}
# ------------------------------------------------------------------------------------------------ pins
$pargs = @((Get-Tool 'pins_check.py'), 'verify', '--pins', "$Root\ckpt_pins.json", '--root', $Root, '--out', "$Root\logs\pins_verify.json")
if (Test-Path "$Root\g0\gate_record.json") { $pargs += @('--g0-record', "$Root\g0\gate_record.json") }
if ($VerifyPinsFull) { $pargs += '--full' }
$rc = Invoke-Py 'pins' $pargs
Stop-OnFail $rc 'checkpoint pins (logs\chain6b_pins.log)' 12
Write-Log 'pins verified'
# ------------------------------------------------------------------------------------------------ stage B
if (Start-Step 'stageB') {
    if ($Mock) { $rc = Invoke-Py 'stageB' @((Get-Tool 'mock_chain6b.py'), 'stageB', '--root', $Root) -TorchFree }
    else { $rc = Invoke-Py 'stageB' @("$Root\run_continual.py", 'B') }
    Stop-OnFail $rc 'reproduction stage B'
    $rc = Invoke-Util 'stageB_check' @('check-stage', 'B', '--root', $Root)
    Stop-OnFail $rc 'stage B completeness (logs\chain6b_stageB_check.log)' 20
    Complete-Step 'stageB' '39/39 stage-B results'
}
if (Start-Step 'stageB_windows') {
    if ($Mock) { $rc = Invoke-Py 'stageB_windows' @((Get-Tool 'mock_chain6b.py'), 'stageB_windows', '--root', $Root) -TorchFree }
    else { $rc = Invoke-Py 'stageB_windows' @("$Root\stageB_windows.py") }
    Stop-OnFail $rc 'stage-B per-window re-evaluation'
    Complete-Step 'stageB_windows' 'per-window re-evaluation done'
}
if (Start-Step 'stageB_analyze') {
    $rc = Invoke-Py 'stageB_analyze' @("$Root\analyze_stageB.py") -TorchFree
    Stop-OnFail $rc 'analyze_stageB.py'
    Complete-Step 'stageB_analyze' 'stage-B tables written'
}
# ------------------------------------------------------------------------------------------------ G1 / G2
if (Start-Step 'g1g2') {
    if ($Mock) { $rc = Invoke-Py 'g1g2' @((Get-Tool 'mock_chain6b.py'), 'g1g2', '--root', $Root, '--python', $py) -TorchFree }
    else {
        $global:LASTEXITCODE = 0
        $null = & "$Root\hooks_after_B.ps1"
        $rc = $LASTEXITCODE
    }
    Stop-OnFail $rc 'G1/G2 (hooks_after_B.ps1; logs\after_B.log; 3/4/5 gate outcome, 6 audit, 7 plan inputs, 8 plan hash)'
    Complete-Step 'g1g2' 'G2 gate COMPLETE'
}
# ------------------------------------------------------------------------------------------------ stage C
if (Start-Step 'stageC') {
    if ($Mock) { $rc = Invoke-Py 'stageC' @((Get-Tool 'mock_chain6b.py'), 'stageC', '--root', $Root) -TorchFree }
    else { $rc = Invoke-Py 'stageC' @("$Root\run_continual.py", 'C') }
    Stop-OnFail $rc 'reproduction stage C'
    $rc = Invoke-Util 'stageC_check' @('check-stage', 'C', '--root', $Root)
    Stop-OnFail $rc 'stage C completeness (logs\chain6b_stageC_check.log)' 20
    Complete-Step 'stageC' '8/8 stage-C results'
}
# ------------------------------------------------------------------------------------------------ stage C inputs, prereg v1.3
# Freeze checklist step 5 (prereg v1.3 sec. 8): NESTOR-books feasibility (sec. 4.4) -> dose-response f_low and draws
# (sec. 4.2) -> C1-DE exact MILP pairs (sec. 4.12; c1de_milp.py -> results\c1de) -> third-base G0 record or 'pending'
# -> confirm plan (budget, cut order P-5 in two passes; an envelope decision P-6 stops here with exit 24) -> forecast
# units -> forecast-input queue (GPU: KSD drive runs + theta0 probes, sec. 5.3; MOCK: synthetic inputs) -> forecast
# files and forecast_io verify (sec. 5.5) -> environment snapshot (logs\env). Sub-steps leave
# results\chain6b\stageC13_<sub>.done. The freeze (chain6b_util.py freeze-check, then freeze-write: FREEZE.json =
# freeze\freeze_record.json with the spec revision-3 hash list) follows.
if (Start-Step 'stageC13') {
    $b3rec = "$Root\g0\gate_record_b3.json"
    $g1rPlan = "$Root\g1r\plan_g1r.json"
    if (-not (Test-Done 'stageC13_nestor_books')) {
        $rc = Invoke-Py 'stageC13' @((Get-Tool 'nestor_books_feasibility.py'), '--gate-config', "$Root\g0\g0_gate_config.json",
            '--g0-record', "$Root\g0\gate_record.json", '--out', "$Root\results\nestor_books_feasibility.json",
            '--draws-out', "$Root\results\nestor_books_draws.json") -TorchFree
        if (($rc -ne 0) -and ($rc -ne 4)) { Stop-OnFail $rc 'NESTOR-books feasibility (prereg v1.3 sec. 4.4)' 24 }
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_nestor_books', '--info', "record complete (exit $rc; 0 feasible, 4 infeasible)")
        Write-Log "stageC13: NESTOR-books feasibility record complete (exit $rc)"
    }
    if (-not (Test-Done 'stageC13_dose')) {
        $rc = Invoke-Py 'stageC13' @((Get-Tool 'dose_draws.py'), '--gate-config', "$Root\g0\g0_gate_config.json",
            '--g0-record', "$Root\g0\gate_record.json", '--out', "$Root\results\dose\dose_record.json",
            '--manifest', "$Root\results\dose\dose_draws.json") -TorchFree
        if (($rc -ne 0) -and ($rc -ne 4)) { Stop-OnFail $rc 'dose-response f_low and draws (prereg v1.3 sec. 4.2)' 24 }
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_dose', '--info', "dose record written (exit $rc; 4 = no dose level)")
        Write-Log "stageC13: dose-response record written (exit $rc)"
    }
    if (-not (Test-Done 'stageC13_c1de')) {
        # prereg v1.3 sec. 4.12: German C1 pairs by exact MILP on the selection half (objective seeds 5600 + d, d = 1-3
        # per MoE base); exit 4 = a draw is not ok (recorded; the confirm plan skips it)
        $rc = Invoke-Py 'stageC13' @((Get-Tool 'c1de_milp.py'), '--g0-record', 'g0\gate_record.json',
            '--out-dir', 'results\c1de') -TorchFree
        if (($rc -ne 0) -and ($rc -ne 4)) { Stop-OnFail $rc 'C1-DE MILP draws (prereg v1.3 sec. 4.12)' 24 }
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_c1de', '--info', "results\c1de written (exit $rc; 4 = a draw not ok)")
        Write-Log "stageC13: C1-DE MILP draws written (exit $rc)"
    }
    if (Test-Path $b3rec) { Write-Log "stageC13: third-base G0 record present ($b3rec)" }
    else { Write-Log 'stageC13: third-base G0 record pending (its runs wait for a dated addendum, prereg v1.3 sec. 5.6)' }
    if (-not (Test-Done 'stageC13_confirm_plan')) {
        $states = @('results\queue_state_g1.json')            # root-relative (cwd = $Root): the plan stays portable
        if (Test-Path "$Root\results\queue_state_g1r.json") { $states += 'results\queue_state_g1r.json' }
        $states += 'results\queue_state_g2.json'
        $cargs = @((Get-Tool 'confirm_plan.py'), 'build', '--g0-record', 'g0\gate_record.json',
            '--g1-record', 'g1\gate_record_G1.json', '--g2-record', 'g2\gate_record_G2.json',
            '--nestor-books', 'results\nestor_books_feasibility.json', '--dose-record', 'results\dose\dose_record.json',
            '--c1de-dir', 'results\c1de', '--states', ($states -join ','), '--out-dir', 'confirm')
        if (Test-Path $b3rec) { $cargs += @('--b3-record', 'g0\gate_record_b3.json') }
        if ($Mock) { $cargs += '--no-check' }                  # MOCK root: no checkpoints or data bins to check
        if ($SpentGpuH -ge 0) {                                # invariant culture: never '2,75' under a uk-UA locale
            $cargs += @('--spent-h', $SpentGpuH.ToString([System.Globalization.CultureInfo]::InvariantCulture))
        } elseif (-not $Mock) {
            Write-Log ('stageC13: -SpentGpuH missing: the envelope check needs the GPU-h already spent (budget_v13 ' +
                "'spent' tier; prereg v1.3 sec. 7); restart with -SpentGpuH <hours>")
            Exit-Chain 24
        }
        $rc = Invoke-Py 'stageC13' $cargs -TorchFree
        Stop-OnFail $rc 'confirm plan build (exit 2: inputs missing; logs\chain6b_stageC13.log)' 24
        $envReq = (Get-Content "$Root\confirm\plan_confirm.json" -Raw -Encoding UTF8 | ConvertFrom-Json).envelope_decision_required
        if ($envReq) {
            Write-Log "stageC13: cut order exhausted -> the envelope decision (P-6) is required: $envReq"
            Exit-Chain 24
        }
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_confirm_plan', '--info', 'confirm\plan_confirm.json built')
        Write-Log 'stageC13: confirm plan built'
    }
    if (-not (Test-Done 'stageC13_forecast_units')) {
        $fp = @('g1\plan_g1.json')
        if (Test-Path $g1rPlan) { $fp += 'g1r\plan_g1r.json' }
        $fp += 'g2\plan_g2.json'
        $gc = @('g0\g0_gate_config.json')
        if (Test-Path $b3rec) { $gc += 'g0\g0_gate_config_b3.json' }
        $rc = Invoke-Py 'stageC13' @((Get-Tool 'forecasts.py'), 'units', '--confirm-plan', 'confirm\plan_confirm.json',
            '--fit-plans', ($fp -join ','), '--g0-config', ($gc -join ','), '--out-dir', 'results\forecast') -TorchFree
        Stop-OnFail $rc 'forecast units (prereg v1.3 sec. 5.1-5.4)' 24
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_forecast_units', '--info', 'results\forecast\forecast_units.json')
        Write-Log 'stageC13: forecast units and input queue written'
    }
    if (-not (Test-Done 'stageC13_forecast_inputs')) {
        if ($Mock) {
            $rc = Invoke-Py 'stageC13' @((Get-Tool 'forecasts.py'), 'mock-inputs', '--units', 'results\forecast\forecast_units.json',
                '--state', 'results\queue_state_forecast.json') -TorchFree
        } else {
            $rc = Invoke-Py 'stageC13' @("$Root\run_queue.py", "$Root\results\forecast\queue_forecast_inputs.json")
        }
        # a GPU queue: run_queue's own exit code is kept (STOP file / job failure), as in the confirm step
        Stop-OnFail $rc 'forecast inputs: KSD drive runs + theta0 probes (results\forecast\queue_forecast_inputs.json)' 24
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_forecast_inputs', '--info', 'drive runs and probes finished')
        Write-Log 'stageC13: forecast inputs finished'
    }
    if (-not (Test-Done 'stageC13_forecasts')) {
        $st = @('results\queue_state_g1.json')
        if (Test-Path "$Root\results\queue_state_g1r.json") { $st += 'results\queue_state_g1r.json' }
        $st += @('results\queue_state_g2.json', 'results\queue_state_forecast.json')
        $margs = @((Get-Tool 'forecasts.py'), 'make', '--units', 'results\forecast\forecast_units.json', '--states', ($st -join ','),
            '--out-dir', 'freeze')
        if ($Mock) { $margs += '--allow-mock' }
        $rc = Invoke-Py 'stageC13' $margs -TorchFree
        Stop-OnFail $rc 'forecast files (prereg v1.3 sec. 5.5)' 24
        $rc = Invoke-Py 'stageC13' @((Get-Tool 'forecast_io.py'), 'verify', 'freeze\forecasts_v13.csv', 'freeze\forecasts_v13_meta.json') -TorchFree
        Stop-OnFail $rc 'forecast_io.py verify (prereg v1.3 sec. 5.5)' 24
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_forecasts', '--info', 'freeze\forecasts_v13.csv + _meta.json verified')
        Write-Log 'stageC13: forecast files written and verified'
    }
    if (-not (Test-Done 'stageC13_env')) {
        # freeze hash list item 10: pip freeze of this venv + conda list --explicit of the numpy env;
        # exit 3 = conda not found (the freeze-check names the missing file)
        $rc = Invoke-Util 'stageC13' @('env-snapshot', '--root', $Root)
        if (($rc -ne 0) -and ($rc -ne 3)) { Stop-OnFail $rc 'environment snapshot (logs\env)' 24 }
        $null = Invoke-Util 'mark' @('mark', '--root', $Root, '--step', 'stageC13_env', '--info', "logs\env written (exit $rc)")
        Write-Log "stageC13: environment snapshot written (exit $rc)"
    }
    Complete-Step 'stageC13' 'prereg v1.3 stage-C inputs complete; the freeze (freeze_record.json, FREEZE.json) follows'
}
# ------------------------------------------------------------------------------------------------ freeze
if (Start-Step 'freeze') {
    $fz = "$Root\FREEZE.json"
    $n = 0
    while (-not (Test-Path $fz)) {
        if (Test-Path "$Root\PAUSE") { Write-Log 'PAUSE file present while waiting for FREEZE.json'; Exit-Chain 0 }
        if (($n % 12) -eq 0) { Write-Log "waiting for FREEZE.json (poll every $FreezePollSec s)" }
        Start-Sleep -Seconds $FreezePollSec
        $n++
    }
    Start-Sleep -Seconds 2
    $rc = Invoke-Util 'freeze_verify' @('freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    Stop-OnFail $rc 'FREEZE.json verification (logs\chain6b_freeze_verify.log)' 23
    Complete-Step 'freeze' 'FREEZE.json verified'
}
# ------------------------------------------------------------------------------------------------ confirmatory + secondary
if (Start-Step 'confirm') {
    $rc = Invoke-Util 'freeze_verify' @('freeze-verify', '--root', $Root, '--freeze', 'FREEZE.json')
    Stop-OnFail $rc 'FREEZE.json re-verification' 23
    $q = (Get-Content "$Root\FREEZE.json" -Raw -Encoding UTF8 | ConvertFrom-Json).queue.path
    $qargs = @("$Root\run_queue.py", "$Root\$q")
    if ($RetryFailed) {                                            # E2: failed jobs re-run with the same seeds
        $rc = Invoke-Util 'retry_prep' @('retry-prep', '--root', $Root, '--queue', $q)
        Stop-OnFail $rc 'retry-prep (failed attempt roots moved aside) before run_queue --retry-failed' 27
        $qargs += '--retry-failed'
    }
    if ($Mock) { $qargs += '--dry-run' }
    $rc = Invoke-Py 'confirm' $qargs
    Stop-OnFail $rc "confirmatory/secondary queue $q (logs\chain6b_confirm.log)"
    Complete-Step 'confirm' "queue $q finished"
}
Write-Log '##### chain6b finished'
Exit-Chain 0
