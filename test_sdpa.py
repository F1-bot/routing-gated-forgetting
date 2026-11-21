import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
for hd in (30, 32):
    q = torch.randn(1, 16, 4096, hd, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        try:
            with sdpa_kernel([be]):
                o = F.scaled_dot_product_attention(q, q, q, is_causal=True)
                o.sum().backward()
            print(f"head_dim={hd} {be.name:20s} OK   peak extra {(torch.cuda.max_memory_allocated()-base)/2**20:8.1f} MiB")
        except Exception as e:
            print(f"head_dim={hd} {be.name:20s} FAIL {type(e).__name__}: {str(e).splitlines()[0][:110]}")
        q.grad = None
