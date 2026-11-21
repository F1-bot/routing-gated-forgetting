import os, time
os.environ.setdefault("TRITON_CACHE_DIR", os.path.abspath("tc"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.abspath("ic"))
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__)

def f(x, w):
    y = torch.nn.functional.silu(x @ w) * (x @ w)
    return torch.nn.functional.rms_norm(y, (y.shape[-1],))

x = torch.randn(8192, 480, device="cuda", dtype=torch.bfloat16)
w = torch.randn(480, 1920, device="cuda", dtype=torch.bfloat16)
cf = torch.compile(f)
t0 = time.time(); out = cf(x, w); torch.cuda.synchronize(); print(f"compile+run {time.time()-t0:.1f}s")
ref = f(x, w)
print("max abs diff vs eager:", float((out.float() - ref.float()).abs().max()))
# grouped_mm and varlen attention availability on sm_89
try:
    a = torch.randn(64, 480, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4, 480, 256, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)
    g = torch._grouped_mm(a, b, offs=offs)
    ref = torch.cat([a[i*16:(i+1)*16] @ b[i] for i in range(4)])
    print("_grouped_mm OK, max diff", float((g.float()-ref.float()).abs().max()))
except Exception as e:
    print("_grouped_mm FAILED:", type(e).__name__, str(e)[:200])
try:
    from torch.nn.attention.varlen import varlen_attn
    q = torch.randn(100, 4, 32, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 40, 100], device="cuda", dtype=torch.int32)
    o = varlen_attn(q, q, q, cu, cu, 60, 60, is_causal=True)
    print("varlen_attn OK", tuple(o.shape))
except Exception as e:
    print("varlen_attn FAILED:", type(e).__name__, str(e)[:300])
