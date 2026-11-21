"""Where does the memory go? One fwd+bwd of nestor_100m (micro=1, S=4096), allocation history."""
import os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "repo"))
exp_name = sys.argv[1] if len(sys.argv) > 1 else "nestor_100m"
import run_exp
for k, v in run_exp.SPEED_STACK.items():
    os.environ.setdefault(k, v)
os.environ["NESTOR_COMPILE"] = "0"
os.environ.setdefault("DCLM_DATA_PATH", os.path.join(HERE, "data", "dclm", "dclm_train.bin"))
import torch
import compat; compat.install()
from nestor.experiments.registry import get_experiment
import nestor.experiments.pretrain  # noqa
from nestor.models import Model
from nestor.train.losses import Loss
import nestor.backbones.nestor as nb
print("flex available:", getattr(nb, "_FLEX_ATTENTION_AVAILABLE", None))
job = get_experiment(exp_name)().jobs[0]
model = Model.from_config(job.model).cuda()
loss_fn = Loss.from_config(job.loss)
GB = 2**30
print(f"after build: {torch.cuda.memory_allocated()/GB:.2f} GB")
torch.cuda.memory._record_memory_history(max_entries=200000)
x = torch.randint(0, 100263, (1, 4097), device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x[:, :-1])
    print(f"after forward (saved activations live): {torch.cuda.memory_allocated()/GB:.2f} GB, fwd peak {torch.cuda.max_memory_allocated()/GB:.2f} GB")
    loss = loss_fn(out.pred, x[:, 1:], out.aux).total
print(f"after loss: {torch.cuda.memory_allocated()/GB:.2f} GB, peak {torch.cuda.max_memory_allocated()/GB:.2f} GB")
try:
    loss.backward()
    print(f"after backward: {torch.cuda.memory_allocated()/GB:.2f} GB, peak {torch.cuda.max_memory_allocated()/GB:.2f} GB")
except torch.OutOfMemoryError:
    print("backward OOM at", f"{torch.cuda.max_memory_allocated()/GB:.2f} GB")
snap = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)
# replay the trace to find the live set at the peak
live, cur, peak, peak_live = {}, 0, 0, {}
for ev in snap["device_traces"][0]:
    if ev["action"] == "alloc":
        live[ev["addr"]] = ev; cur += ev["size"]
        if cur > peak:
            peak, peak_live = cur, dict(live)
    elif ev["action"] in ("free_completed",):
        e = live.pop(ev["addr"], None)
        if e is not None:
            cur -= e["size"]
def site(ev):
    fr = [f for f in ev.get("frames", []) if "nestor" in f["filename"] and "site-packages" not in f["filename"]]
    f = fr[0] if fr else (ev.get("frames") or [{"filename": "?", "line": 0, "name": "?"}])[0]
    return f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}"
agg = collections.Counter(); cnt = collections.Counter()
for ev in peak_live.values():
    agg[site(ev)] += ev["size"]; cnt[site(ev)] += 1
print(f"traced peak live: {peak/GB:.2f} GB in {len(peak_live)} tensors; top sites:")
for s, b in agg.most_common(18):
    print(f"  {b/GB:6.2f} GB  x{cnt[s]:<5d} {s}")
