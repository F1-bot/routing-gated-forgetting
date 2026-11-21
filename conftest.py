# Root conftest for running the authors' tests on torch 2.14.1: translate the old
# varlen_attn(is_causal=True) keyword to window_size=(-1, 0). Nothing else is patched.
import torch.nn.attention.varlen as _vl

_orig = _vl.varlen_attn


def _varlen_attn_compat(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *, is_causal=False, **kw):
    if is_causal:
        kw.setdefault("window_size", (-1, 0))
    return _orig(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, **kw)


_vl.varlen_attn = _varlen_attn_compat
