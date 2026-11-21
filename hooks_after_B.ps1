# hooks_after_B.ps1 -- G1 and G2 after reproduction stage B (g1g2_spec.json 1.3.0; prereg v1.3 sec. 6).
# Install as hooks\after_B.ps1; chain4 runs it from the workspace root when stage B ends. Windows PowerShell 5.1.
# Order: (0) code_sha256 of all code and configs, before any G1 data exist; (1) GPU pipeline audit (tiny models, CUDA;
# abort on any failure); (2) plan g1 from the REAL g0\gate_record.json; (3) G1 queue; (4) g1_gate.py -- exit 3 runs the
# 192-step FULL repeat (plan g1r, queue, g1_gate.py --repeat-plan); (5) plan g2 + G2 queue only if G1 PROCEEDs;
# (6) g2_gate.py. Resumable: re-running the hook rebuilds each plan deterministically (same sha256) and run_queue.py skips
# every finished job by id; a plan whose hash changed while its queue state exists stops the hook.
# Torch-free steps (plans, gates) get the conda Library\bin PATH prefix; torch steps (queues, pytest) must not.
# Records: g1\gate_record_G1.json, g2\gate_record_G2.json (spec hash list). G1-0 reads logs\tests_gpu_pipeline.xml (junit).
# Log: logs\after_B.log (UTF-8). Exit codes: 0 G2 done; 3/4/5 from the gates (see g1_gate.py / g2_gate.py); 6 audit failed;
# 7 plan inputs missing; 8 plan hash drift; 9 gate code does not reproduce the spec 1.3.0 test vectors.
$ErrorActionPreference = 'Continue'
$py   = '.\.venv\nestor\Scripts\python.exe'
$root = (Get-Location).Path
$env:PYTHONPATH = "$root;$env:PYTHONPATH"
$env:PYTHONUTF8 = '1'
$log  = 'logs\after_B.log'
New-Item -ItemType Directory -Force logs, results | Out-Null
$condaLibBin = '<conda>\envs\nestor\Library\bin'
$g0record = 'g0\gate_record.json'
$config   = 'g1g2_config.json'
$lookup   = 'signflip_lookup_v1.csv'

function Write-Log([string]$msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $log }

function Invoke-Py([string[]]$argv, [switch]$TorchFree) {
    $saved = $env:PATH
    if ($TorchFree) { $env:PATH = "$condaLibBin;$saved" }
    try { & $py @argv 2>&1 | Out-File -Append -Encoding utf8 $log; return $LASTEXITCODE }
    finally { $env:PATH = $saved }
}
function Invoke-TorchFree([string]$name, [string[]]$argv) {
    Write-Log "=== $name (torch-free)"
    $rc = Invoke-Py $argv -TorchFree
    Write-Log "--- $name exit $rc"
    return $rc
}
function Invoke-Torch([string]$name, [string[]]$argv) {
    Write-Log "=== $name"
    $rc = Invoke-Py $argv
    Write-Log "--- $name exit $rc"
    return $rc
}
function Get-PlanSha([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    return (Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json).sha256
}
function Build-Plan([string]$stage, [string[]]$argv, [string]$planPath, [string]$statePath) {
    $old = Get-PlanSha $planPath
    $rc = Invoke-TorchFree "plan $stage" $argv
    if ($rc -ne 0) { Write-Log "plan $stage failed (exit $rc): inputs missing"; exit 7 }
    $new = Get-PlanSha $planPath
    if ($old -and $new -ne $old -and (Test-Path $statePath)) {
        Write-Log "plan $stage hash changed ($old -> $new) while $statePath exists: inputs changed under a running stage"
        exit 8
    }
    Write-Log "plan $stage sha256 $new"
}

Write-Log '##### hooks_after_B start'

# 0. code_sha256: every code and config file, before any G1 data exist (the design firewall; prereg sec. 16)
$files = @(Get-ChildItem -File -Path . -Include *.py, *.ps1, *.json, *.csv -Depth 0 -ErrorAction SilentlyContinue) +
         @(Get-ChildItem -File -Path hooks -Include *.ps1 -Recurse -ErrorAction SilentlyContinue)
$hashes = [ordered]@{}
foreach ($f in ($files | Sort-Object FullName)) { $hashes[$f.FullName.Substring($root.Length + 1)] = (Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower() }
$commit = $null
try { $commit = (& git -C repo rev-parse HEAD 2>$null) } catch { }
[ordered]@{ created = (Get-Date -Format s); repo_commit = $commit; files = $hashes } | ConvertTo-Json -Depth 4 |
    Out-File -Encoding utf8 'logs\after_B_code_sha256.json'
Write-Log "code_sha256: $($hashes.Count) files -> logs\after_B_code_sha256.json (repo $commit)"

# 1. GPU pipeline audit: tiny-model train-path tests on CUDA (they cannot run on this host's CPU: Triton dispatch)
$sw = [Diagnostics.Stopwatch]::StartNew()
$env:NESTOR_TEST_DEVICE = 'cuda'
$rc = Invoke-Torch 'GPU pipeline audit' @('-m', 'pytest', '-q', '-rA', '-p', 'no:cacheprovider',
    '--junitxml', 'logs\tests_gpu_pipeline.xml', '--basetemp', "logs\pytest_gpu_audit_$(Get-Date -Format yyyyMMdd_HHmmss)",
    '-o', 'tmp_path_retention_policy=all',
    'test_subset_adapt.py::test_pipeline_end_to_end_init_checkpoint_subset_run',
    'test_phase3.py::test_pipeline_sgd_arm_and_negative_control',
    'test_phase3.py::test_wd_only_state_reproduces_a_zero_gradient_run_bitwise',
    'test_phase3.py::test_pipeline_interim_evals_train_log_no_clip_and_eval_cli',
    'test_phase5_torch.py::test_pipeline_loss_log_and_step0_interim',
    'test_phase5_torch.py::test_eval_cli_pin_variants_and_fsym_decomposition')
Remove-Item Env:\NESTOR_TEST_DEVICE -ErrorAction SilentlyContinue
$sw.Stop()
Write-Log ("GPU pipeline audit: exit $rc in {0:N1} min (budget 10 min)" -f $sw.Elapsed.TotalMinutes)
if ($rc -ne 0) { Write-Log 'GPU pipeline audit FAILED: G1 aborted'; exit 6 }

# 2.-3. G1 plan (torch-free) and queue (torch children; resumable by job id)
Build-Plan 'g1' @('g1g2_plan.py', 'g1', '--g0-record', $g0record, '--config', $config, '--out-dir', 'g1') 'g1\plan_g1.json' 'results\queue_state_g1.json'
$rc = Invoke-Torch 'G1 queue' @('run_queue.py', 'g1\queue_g1.json')
if ($rc -ne 0) { Write-Log "G1 queue stopped (exit $rc)"; exit $rc }

# 3b. the gate rules reproduce every test vector of g1g2_spec.json 1.3.0 (freeze precondition; checked before any gate)
$rc = Invoke-TorchFree 'spec 1.3.0 test vectors' @('gate_rules_v13.py', '--spec', 'g1g2_spec.json', '--lookup', $lookup,
    '--out', 'logs\spec_v13_vectors.json')
if ($rc -ne 0) { Write-Log 'gate rules do not reproduce the spec 1.3.0 test vectors: no gate is evaluated'; exit 9 }

# 4. G1 gate (torch-free); exit 3 = G1-1 lengthen -> 192-step repeat of the FULL runs, both S1 A/A pairs and the NESTOR
#    A/A subset (decision O-2), then the gate again with --repeat-plan
$rc = Invoke-TorchFree 'g1_gate.py' @('g1_gate.py', '--plan', 'g1\plan_g1.json', '--out', 'g1\gate_record_G1.json')
if ($rc -eq 3) {
    Build-Plan 'g1r' @('g1g2_plan.py', 'g1r', '--g0-record', $g0record, '--config', $config, '--out-dir', 'g1r', '--g1-dir', 'g1') 'g1r\plan_g1r.json' 'results\queue_state_g1r.json'
    $q = Invoke-Torch 'G1 repeat queue' @('run_queue.py', 'g1r\queue_g1r.json')
    if ($q -ne 0) { Write-Log "G1 repeat queue stopped (exit $q)"; exit $q }
    $rc = Invoke-TorchFree 'g1_gate.py (repeat)' @('g1_gate.py', '--plan', 'g1\plan_g1.json', '--repeat-plan', 'g1r\plan_g1r.json', '--out', 'g1\gate_record_G1.json')
}
if ($rc -ne 0) { Write-Log "G1 did not PROCEED (exit $rc): no G2"; exit $rc }

# 5. G2 plan (from the hashed G1 record) and queue -- only after G1 PROCEED
Build-Plan 'g2' @('g1g2_plan.py', 'g2', '--g0-record', $g0record, '--g1-record', 'g1\gate_record_G1.json', '--config', $config, '--out-dir', 'g2') 'g2\plan_g2.json' 'results\queue_state_g2.json'
$rc = Invoke-Torch 'G2 queue' @('run_queue.py', 'g2\queue_g2.json')
if ($rc -ne 0) { Write-Log "G2 queue stopped (exit $rc)"; exit $rc }

# 6. G2 gate (torch-free): G2-2 -> G2-3 -> G2-4 -> G2-1 -> G2-5; gate-determined freeze parameters in the record
$rc = Invoke-TorchFree 'g2_gate.py' @('g2_gate.py', '--plan', 'g2\plan_g2.json', '--g1-record', 'g1\gate_record_G1.json', '--lookup', $lookup, '--out', 'g2\gate_record_G2.json')
Write-Log "##### hooks_after_B end (g2 gate exit $rc)"
exit $rc
