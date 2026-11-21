# Pretraining queue: Dense -> MoE -> Nestor at 100M, 1,907 steps x 262,144 tokens (0.5B tokens each).
# Same recipe, seed, data and data order for all three; only the micro-batch / activation-checkpoint
# memory settings differ (both are exact and do not change the optimisation).
. .\msvc_env.ps1
$py = ".\.venv\nestor\Scripts\python.exe"
$ws = (Get-Location).Path
$common = @{
  DCLM_DATA_PATH = "$ws\data\dclm\dclm_train16.bin"; NESTOR_OUTPUT_ROOT = "$ws\outputs";
  DCLM_LOG_FREQ = "10"; DCLM_CKPT_FREQ = "200"; MEM_FRAC = "0.85"; DCLM_RUN_SUFFIX = "b05"
}
$runs = @(
  @{ exp = "dense_100m";  micro = 4; accum = 16; ckpt = "0" },
  @{ exp = "moe_100m";    micro = 2; accum = 32; ckpt = "0" },
  @{ exp = "nestor_100m"; micro = 4; accum = 16; ckpt = "1" }
)
foreach ($r in $runs) {
  foreach ($k in $common.Keys) { Set-Item -Path "env:$k" -Value $common[$k] }
  $env:NESTOR_ACTIVATION_CHECKPOINT = $r.ckpt; $env:NESTOR_ACTIVATION_CHECKPOINT_SPLIT = $r.ckpt; $env:NESTOR_ACTIVATION_CHECKPOINT_LEAN = $r.ckpt
  $log = "logs\pretrain_$($r.exp)_b05.log"
  "[queue] $(Get-Date -Format s) start $($r.exp) micro=$($r.micro) accum=$($r.accum) ckpt=$($r.ckpt)" | Out-File -Append logs\queue.log
  & $py run_exp.py $r.exp --steps 1907 --warmup 153 --micro $r.micro --accum $r.accum *> $log
  "[queue] $(Get-Date -Format s) end $($r.exp) exit=$LASTEXITCODE" | Out-File -Append logs\queue.log
}
