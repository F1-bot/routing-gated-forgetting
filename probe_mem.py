"""Peak-memory / micro-step-time probe for the paper's 100M models on one 16 GB GPU.

python probe_mem.py <experiment> <micro> <ckpt:0|1|split|block> <compile:0|1> [n_iters]
Builds the model exactly as the experiment config does (authors' code), reserves optimizer-state
memory (Muon momentum for matrices, AdamW m+v for the rest), caps PyTorch at 92% of VRAM so an
overflow raises OOM instead of spilling to system RAM, and times fwd+bwd micro-steps (bf16 autocast).
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))
exp_name, micro, ckpt, comp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else 6
import run_exp
for k, v in run_exp.SPEED_STACK.items():
    os.environ.setdefault(k, v)
os.environ["NESTOR_COMPILE"] = "1" if comp else "0"
os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "0" if ckpt == "0" else "1"
os.environ["NESTOR_ACTIVATION_CHECKPOINT_SPLIT"] = "1" if ckpt == "split" else "0"
os.environ["NESTOR_ACTIVATION_CHECKPOINT_BLOCK"] = "1" if ckpt == "block" else "0"
os.environ.setdefault("DCLM_DATA_PATH", os.path.join(HERE, "data", "dclm", "dclm_train.bin"))
import torch
import compat; compat.install()
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MEM_FRAC', '0.85')))
from nestor.experiments.registry import get_experiment
import nestor.experiments.pretrain  # noqa
from nestor.models import Model
from nestor.train.losses import Loss

job = get_experiment(exp_name)().jobs[0]
model = Model.from_config(job.model).cuda()
loss_fn = Loss.from_config(job.loss)
n_params = sum(p.numel() for p in model.parameters())
# optimizer-state reservation: 1 buffer for >=2-D non-embedding matrices (Muon), 2 for the rest (AdamW)
opt_bytes = 0
for name, p in model.named_parameters():
    is_muon = p.ndim >= 2 and "embed" not in name and "head" not in name and "lm_" not in name
    opt_bytes += p.numel() * 4 * (1 if is_muon else 2)
opt_reserve = torch.empty(opt_bytes // 4, dtype=torch.float32, device="cuda")
fwd = torch.compile(model) if comp else model
V = 100263
res = {"exp": exp_name, "micro": micro, "ckpt": ckpt, "compile": comp, "params": n_params,
       "opt_state_gb": round(opt_bytes / 2**30, 2)}
try:
    times = []
    for it in range(n_iters):
        x = torch.randint(0, V, (micro, 4097), device="cuda")
        inp, tgt = x[:, :-1].contiguous(), x[:, 1:].contiguous()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(inp)
            loss = loss_fn(out.pred, tgt, out.aux).total
        loss.backward()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    steady = times[2:] if len(times) > 3 else times[1:]
    res.update(ok=True, loss=round(float(loss), 3), first_s=round(times[0], 1),
               micro_step_s=round(sum(steady) / len(steady), 3),
               tok_per_s=round(micro * 4096 / (sum(steady) / len(steady))),
               peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
except torch.OutOfMemoryError as e:
    res.update(ok=False, err="OOM", peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
except Exception as e:
    res.update(ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
print("RESULT " + json.dumps(res), flush=True)
