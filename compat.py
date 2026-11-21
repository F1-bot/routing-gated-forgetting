"""Compatibility shims for running the Nestor repo on torch 2.14.1 / Windows.

1. ``varlen_attn(..., is_causal=True)``: the repo targets an older varlen API; torch 2.14.1
   expresses causality as ``window_size=(-1, 0)``. Translate the keyword, nothing else.
"""
import torch
import torch.nn.attention.varlen as _vl

_orig_varlen_attn = _vl.varlen_attn


def varlen_attn_compat(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *, is_causal=False, **kw):
    if is_causal:
        kw.setdefault("window_size", (-1, 0))
    return _orig_varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, **kw)


import math
import torch.nn.functional as _F

_orig_sdpa = _F.scaled_dot_product_attention


def sdpa_headpad(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    """2. Exact head-dim padding for SDPA.

    The 100M models have head_dim = 480/16 = 30. The fused SDPA kernels available in the Windows
    build (memory-efficient, cuDNN) need head_dim % 8 == 0, so torch silently falls back to the MATH
    kernel, which materialises [B, H, S, S] scores (4 GB per call at S=4096). Zero-padding q/k/v to the
    next multiple of 8 leaves q.k^T unchanged; the original scale is passed explicitly (1/sqrt(E) if
    the caller gave none) and the padded output columns (exact zeros) are sliced off.
    """
    e, ev = query.size(-1), value.size(-1)
    if e % 8 == 0 and ev % 8 == 0:
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, **kw)
    if scale is None:
        scale = 1.0 / math.sqrt(e)
    pe, pv = (-e) % 8, (-ev) % 8
    out = _orig_sdpa(_F.pad(query, (0, pe)), _F.pad(key, (0, pe)), _F.pad(value, (0, pv)),
                     attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    return out[..., :ev]


def install():
    _vl.varlen_attn = varlen_attn_compat
    _F.scaled_dot_product_attention = sdpa_headpad
    torch.nn.functional.scaled_dot_product_attention = sdpa_headpad


def self_test_headpad():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 512, 30, device="cuda", dtype=torch.float32) for _ in range(3))
    mask = torch.rand(2, 1, 512, 512, device="cuda") > 0.3
    mask |= torch.eye(512, dtype=torch.bool, device="cuda")
    errs = []
    for kwargs in ({"is_causal": True}, {"attn_mask": mask}):
        ref = _orig_sdpa(q, k, v, scale=1 / 30, **kwargs)        # fp32 reference (math kernel)
        out = sdpa_headpad(q.bfloat16(), k.bfloat16(), v.bfloat16(), scale=1 / 30, **kwargs).float()
        ref_bf = _orig_sdpa(q.bfloat16(), k.bfloat16(), v.bfloat16(), scale=1 / 30, **kwargs).float()
        errs.append(((out - ref).abs().max().item(), (ref_bf - ref).abs().max().item()))
    return errs


def self_test():
    torch.manual_seed(0)
    H, D = 4, 64
    lens = [37, 128, 5, 300]
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)
    N = int(cu[-1])
    q, k, v = (torch.randn(N, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = varlen_attn_compat(q, k, v, cu, cu, max(lens), max(lens), is_causal=True)
    ref = []
    for a, b in zip(cu[:-1].tolist(), cu[1:].tolist()):
        ref.append(torch.nn.functional.scaled_dot_product_attention(
            q[a:b].transpose(0, 1)[None], k[a:b].transpose(0, 1)[None], v[a:b].transpose(0, 1)[None],
            is_causal=True)[0].transpose(0, 1))
    ref = torch.cat(ref)
    err = (out.float() - ref.float()).abs().max().item()
    noncausal = _orig_varlen_attn(q, k, v, cu, cu, max(lens), max(lens))
    err_nc = (noncausal.float() - ref.float()).abs().max().item()
    return err, err_nc


if __name__ == "__main__":
    for name, (pad_err, bf16_err) in zip(("causal", "bool mask"), self_test_headpad()):
        print(f"head-pad SDPA ({name}): max |padded_bf16 - fp32 ref| = {pad_err:.4g}; "
              f"unpadded bf16 math kernel vs fp32 ref = {bf16_err:.4g}")
