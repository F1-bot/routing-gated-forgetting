# hooks_after_G0.ps1 -- G0 gate record + G0-7 margin probes + codeparrot de-duplication
# (5). Windows PowerShell; run from the workspace root AFTER the three G0
# runs (g0\moe_b1, g0\moe_b2, g0\nestor) and the G0-5 local mean-CE evaluations have finished, and after installing
# g0_gate.py, g0_margin.py, dedup_check.py, crossover.py (v3), eval_windows.py, g0_probe.py into the root.
# GPU budget: G0-7 probes ~0.15-0.3 GPU-h per MoE base at 24 windows per corpus (printed by the 'estimate' step).
$ErrorActionPreference = 'Continue'        # PS 5.1 wraps native stderr in ErrorRecords; 'Stop' would abort on warnings
$py   = '.\.venv\nestor\Scripts\python.exe'
$root = (Get-Location).Path
$env:PYTHONPATH = "$root;$env:PYTHONPATH"  # PYTHONSAFEPATH=1 in the operator kernel
$log  = 'logs\after_G0.log'
New-Item -ItemType Directory -Force logs | Out-Null

# 2026-11-01: on this host the venv's conda numpy aborts (0xC06D007F, delay-load failure) at its first BLAS/LAPACK
# call in a process WITHOUT torch unless the conda env's Library\bin is on PATH; torch processes must NOT get that
# prefix (libiomp5md/libomp clash). Steps that never import torch therefore run with the prefix (-NoTorch).
$condaLibBin = '<conda>\envs\nestor\Library\bin'
function Run-Step([string]$name, [string[]]$argv, [switch]$NoTorch) {
    "=== $(Get-Date -Format s) $name" | Out-File -Append -Encoding utf8 $log
    $savedPath = $env:PATH
    if ($NoTorch) { $env:PATH = "$condaLibBin;$savedPath" }
    & $py @argv 2>&1 | Out-File -Append -Encoding utf8 $log
    $env:PATH = $savedPath
    if ($LASTEXITCODE -ne 0) {
        "STEP FAILED: $name (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
        exit $LASTEXITCODE
    }
}

# 0. hash the gate code BEFORE any G0 statistic is read
Get-FileHash g0_gate.py, crossover.py, g0_margin.py, dedup_check.py -Algorithm SHA256 |
    Format-Table -AutoSize | Out-String | Out-File -Append -Encoding utf8 $log

# 1. codeparrot de-duplication against AlgebraicStack test and the DCLM evaluation shards (CPU, minutes)
Run-Step 'dedup' -NoTorch @('dedup_check.py', '--query', 'data\domain\pycode_test.bin',
                   '--ref', 'algstack=data\domain\algstack_test.bin', '--ref', 'dclm09=data\dclm\dclm_val.bin',
                   '--ref', 'dclm10=data\dclm\dclm_val10.bin', '--n', '13', '--out', 'g0\dedup_pycode.json')

# 2. gate pass 1: G0-1..G0-6 (fixes f per cell and the code stream; G0-7 'missing' until step 3)
Run-Step 'gate pass 1' -NoTorch @('g0_gate.py', '--config', 'g0\g0_gate_config.json', '--out', 'g0\gate_record.json')

# 3. G0-7 only if MoE passed G0-1 (otherwise PIVOT and no margin is needed)
$moe = & $py -c "import json; r=json.load(open(r'g0\gate_record.json', encoding='utf-8')); print(r['gates']['G0-1'].get('moe_pass'))"
if ($moe -ne 'True') {
    "G0-1 MoE pass = $moe -> G0-7 skipped" | Out-File -Append -Encoding utf8 $log
    exit 0
}
foreach ($b in @('moe_b1', 'moe_b2')) {
    Run-Step "G0-7 plan $b" -NoTorch @('g0_margin.py', 'plan', '--g0-dir', "g0\$b", '--gate-record', 'g0\gate_record.json', '--draws', '5')
    Run-Step "G0-7 estimate $b" -NoTorch @('g0_margin.py', 'estimate', '--g0-dir', "g0\$b", '--probe-windows', '24')
    Run-Step "G0-7 probes $b"   @('g0_margin.py', 'run', '--g0-dir', "g0\$b", '--probe-windows', '24', '--device', 'cuda')
}

# 4. gate pass 2: adds G0-7 (D_b, SE, beta0_b, margin-limited flag); the record is deterministic given its inputs
Run-Step 'gate pass 2' -NoTorch @('g0_gate.py', '--config', 'g0\g0_gate_config.json', '--out', 'g0\gate_record.json')
