"""Device-memory budget guard (host fact, 2026-11-01).

On this Windows host (WDDM driver) a CUDA allocation beyond physical VRAM does NOT raise an out-of-memory error: the
driver silently pages device memory to system RAM and kernels slow down ~200x (G0 on MoE base 1: 87.5 % of time in
torch._grouped_mm, process RSS 35 GB). Every tool that builds large tensor sets on the GPU therefore checks an explicit
budget BEFORE allocating (`reserve`) and AFTER building (`check`), and raises instead of oversubscribing.

  used   = max(total - free (cudaMemGetInfo, driver view), torch.cuda.memory_reserved (this process's cache))
  budget = NESTOR_VRAM_BUDGET_FRAC (default 0.90) x total physical memory of the device

The error message contains "OutOfMemoryError" so run_queue.py's existing OOM fallback (activation checkpointing, then half
the micro-batch) is triggered by a guarded training process exactly as by a real CUDA OOM. CPU devices always pass.
NESTOR_VRAM_GUARD=0 disables every check (default: enabled).
"""
from __future__ import annotations

import os
from typing import Any, Iterable

DEFAULT_FRAC = 0.90


class DeviceMemoryBudgetError(RuntimeError):
    pass


def enabled() -> bool:
    return os.environ.get("NESTOR_VRAM_GUARD", "1").strip() != "0"


def budget_frac() -> float:
    try:
        f = float(os.environ.get("NESTOR_VRAM_BUDGET_FRAC", DEFAULT_FRAC))
    except ValueError:
        f = DEFAULT_FRAC
    return min(max(f, 0.05), 1.0)


def _cuda_device(device) -> Any | None:
    import torch

    if device is None:
        return None
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def usage(device) -> dict:
    """Driver and allocator view of device memory (bytes); {'cuda': False} on CPU."""

    import torch

    dev = _cuda_device(device)
    if dev is None:
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info(dev)
    reserved = int(torch.cuda.memory_reserved(dev))
    used = max(int(total) - int(free), reserved)
    return {"cuda": True, "device": str(dev), "total": int(total), "free": int(free), "reserved": reserved, "used": used,
            "budget": int(budget_frac() * int(total))}


def _fmt(b: float) -> str:
    return f"{b / 2 ** 30:.2f} GiB"


def check(device, label: str = "", *, frac: float | None = None, empty_cache: bool = True) -> dict:
    """Raise DeviceMemoryBudgetError if the device is already above its budget (post-condition after building).
    Unused cached blocks are released first, so the allocator's cache does not count as demand."""

    import torch

    if not enabled() or _cuda_device(device) is None:
        return {"cuda": False, "checked": False}
    if empty_cache:
        torch.cuda.empty_cache()
    u = usage(device)
    budget = int((frac if frac is not None else budget_frac()) * u["total"])
    if u["used"] > budget:
        raise DeviceMemoryBudgetError(
            f"torch.OutOfMemoryError (device-memory budget guard, {label}): {_fmt(u['used'])} in use of {_fmt(u['total'])} "
            f"(budget {_fmt(budget)}); refusing to continue on an oversubscribed device (WDDM pages silently)")
    return u


def reserve(device, need_bytes: int, label: str = "", *, frac: float | None = None, empty_cache: bool = True) -> dict:
    """Raise BEFORE allocating `need_bytes` more on the device if that would exceed the budget."""

    import torch

    dev = _cuda_device(device)
    if dev is None or not enabled():
        return {"cuda": False, "checked": False}
    if empty_cache:
        torch.cuda.empty_cache()
    u = usage(dev)
    budget = int((frac if frac is not None else budget_frac()) * u["total"])
    if u["used"] + int(need_bytes) > budget:
        raise DeviceMemoryBudgetError(
            f"torch.OutOfMemoryError (device-memory budget guard, {label}): {_fmt(u['used'])} in use + {_fmt(need_bytes)} "
            f"requested > budget {_fmt(budget)} of {_fmt(u['total'])}; keep these tensors on the host instead")
    return dict(u, need=int(need_bytes))


def check_peak(device, label: str = "", *, frac: float | None = None) -> dict:
    """Training processes: raise if this process's PEAK reservation (torch.cuda.max_memory_reserved, which on WDDM can
    exceed physical memory without an error) is above the budget of the device's physical memory."""

    import torch

    dev = _cuda_device(device)
    if dev is None or not enabled():
        return {"cuda": False, "checked": False}
    total = int(torch.cuda.get_device_properties(dev).total_memory)
    peak = int(torch.cuda.max_memory_reserved(dev))
    budget = int((frac if frac is not None else budget_frac()) * total)
    if peak > budget:
        raise DeviceMemoryBudgetError(
            f"torch.OutOfMemoryError (device-memory budget guard, {label}): peak reservation {_fmt(peak)} > budget "
            f"{_fmt(budget)} of {_fmt(total)} physical; on WDDM this pages silently -- retry with activation "
            f"checkpointing or a smaller micro-batch")
    return {"cuda": True, "device": str(dev), "total": total, "peak_reserved": peak, "budget": budget}


def tensor_bytes(tensors: Iterable) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def module_bytes(model) -> int:
    return tensor_bytes(list(model.parameters()) + list(model.buffers()))


def host_copy(state: dict) -> dict:
    """A host (CPU) copy of a state dict: variant states are built on the host and loaded in place (no device copy)."""

    return {k: (v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v) for k, v in state.items()}
