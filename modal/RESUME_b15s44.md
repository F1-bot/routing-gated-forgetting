# MoE-100M third base (seed 44, b15s44): state and how to finish it

State (2026-11-13 04:03 UTC): the run was stopped at step 1,000 of 5,722 by the in-job budget guard. Its default
deadline assumed 3x sandbox CPU/memory rates; the H100 sandbox was billed at 1x ($4.058/h for H100 + 1 core + 6 GiB).
Resumable checkpoint (optimizer, scheduler and step included): Modal volume `nestor-data`,
`/outputs/moe_100m_b15s44/20261113_031245__moe_100m_b15s44_train/checkpoints/model_00001000.pt` (3,370,941,320 B).
Verified bins are in `/data` on the same volume (sha256 455754d1... / 0644eac4... / 9bdfce3d...).

Modal spend so far: $0.427 (data build, CPU sandbox, 02:00 hour) + $3.294 (H100, 03:00 hour) + about $0.20 (H100,
04:00-04:03, estimated from the measured rate) = about $3.92.

## Finish in one fresh H100 leg (about 10,900 s, about $12.3 at $4.058/h; harvest about $0.03)
repl:
    c = host.compute.create('modal', provider_params={'image': 'python:3.12', 'gpu': 'H100', 'cpu': 1, 'memory': 6144,
                            'volumes': {'/vol': 'nestor-data'}, 'timeout': 11700})
    j = c.submit_job(intent='b15s44 resume from step 1000', inputs=[<staged copies of moe3_b15s44.sh, guard_b15s44.py,
                     nestor_repo.tar.gz, compat.py, run_exp.py, tok_files.py, eval_ckpt.py, token_documents.py>],
                     command='STOP_AFTER_S=<S> RESERVE_S=260 bash moe3_b15s44.sh', run_timeout_s=11500)
with S = leg_budget_usd / 4.058 * 3600 - 45 (for example $12.6 -> S = 11,133).
moe3_b15s44.sh restores the newest step checkpoint from the volume. The authors' auto-resume then restores the
optimizer, scheduler and step, and fast-forwards the dataloader by 8,000 micro-batches. The same RandomSampler
permutation is reused (checked by code inspection), so the leg continues on unseen data.
The guard copies model_<step>.pt every 1,000 steps to the volume and stops at a checkpoint if the projected end
passes T0 + S. After training it runs the fp32 eval (val09 first 10M tokens = 10,004,085 counted, val10 5M, block 4096)
and the slim export (95 MB chunks) to /vol/export/b15s44/.

## Bring it home
Stop the H100 job when it is done. Then start a CPU job (cpu 0.25, memory 1024, the same volume)
and run `bash harvest_b15s44.sh 00 01`, ... (2 chunks per job; 16 chunks expected). Save each chunk separately, because
hpc/ cannot be read from the local environment. Then run
`python reassemble_ckpt.py export_manifest.txt <out>\model_final.pt part_00 ... part_15`
into outputs\moe_100m_b15s44\<run_dir>_h100\checkpoints\model_final.pt and verify the sha256.

Scripts (LF): data_b15s44.sh, moe3_b15s44.sh, guard_b15s44.py, harvest_b15s44.sh (this folder).
Logs: logs\modal_b15s44\.
