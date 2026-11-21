"""Device-memory budget guard (memguard.py; host fact: WDDM pages oversubscribed VRAM silently).

  python -m pytest test_memguard.py -q      (no GPU needed: the CUDA memory API is simulated)
"""
from __future__ import annotations

import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import memguard as MG  # noqa: E402

GiB = 2 ** 30


@pytest.fixture
def fake_cuda(monkeypatch):
    st = {"free": 10 * GiB, "total": 16 * GiB, "reserved": 2 * GiB, "peak": 3 * GiB, "emptied": 0}
    monkeypatch.setattr(MG, "_cuda_device", lambda d: None if str(d) == "cpu" else "cuda:0")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (st["free"], st["total"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: st["reserved"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d=None: st["peak"])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: st.__setitem__("emptied", st["emptied"] + 1))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d=None: types.SimpleNamespace(total_memory=st["total"]))
    monkeypatch.delenv("NESTOR_VRAM_GUARD", raising=False)
    monkeypatch.delenv("NESTOR_VRAM_BUDGET_FRAC", raising=False)
    return st


def test_check_passes_within_budget_and_raises_when_oversubscribed(fake_cuda):
    u = MG.check("cuda", "t")                                   # used = 16 - 10 = 6 GiB < 14.4 GiB
    assert u["used"] == 6 * GiB and fake_cuda["emptied"] == 1
    fake_cuda["free"] = 1 * GiB                                 # 15 GiB used > 0.9 x 16
    with pytest.raises(MG.DeviceMemoryBudgetError, match="OutOfMemoryError"):
        MG.check("cuda", "t")
    fake_cuda["free"], fake_cuda["reserved"] = 15 * GiB, 15 * GiB     # driver view low, allocator view high
    with pytest.raises(MG.DeviceMemoryBudgetError):
        MG.check("cuda", "t")


def test_reserve_refuses_an_allocation_that_would_exceed_the_budget(fake_cuda):
    MG.reserve("cuda", 8 * GiB, "fits")                         # 6 + 8 = 14 <= 14.4
    with pytest.raises(MG.DeviceMemoryBudgetError, match="keep these tensors on the host"):
        MG.reserve("cuda", 9 * GiB, "too much")


def test_check_peak_for_training_processes(fake_cuda):
    MG.check_peak("cuda", "step 1")
    fake_cuda["peak"] = 17 * GiB                                # WDDM: reservation beyond physical memory, no error
    with pytest.raises(MG.DeviceMemoryBudgetError, match="OutOfMemoryError"):
        MG.check_peak("cuda", "step 1")


def test_budget_fraction_switch_and_cpu(fake_cuda, monkeypatch):
    fake_cuda["free"] = 1 * GiB
    monkeypatch.setenv("NESTOR_VRAM_BUDGET_FRAC", "0.99")
    MG.check("cuda", "t")                                       # 15 GiB <= 15.84 GiB
    monkeypatch.setenv("NESTOR_VRAM_BUDGET_FRAC", "0.5")
    with pytest.raises(MG.DeviceMemoryBudgetError):
        MG.check("cuda", "t")
    monkeypatch.setenv("NESTOR_VRAM_GUARD", "0")
    assert MG.check("cuda", "t")["checked"] is False and MG.reserve("cuda", 99 * GiB)["checked"] is False
    monkeypatch.delenv("NESTOR_VRAM_GUARD")
    assert MG.check("cpu", "t")["checked"] is False


def test_host_copy_and_byte_counts():
    m = torch.nn.Linear(4, 3)
    h = MG.host_copy(m.state_dict())
    assert all(v.device.type == "cpu" for v in h.values()) and h["weight"].data_ptr() != m.weight.data_ptr()
    assert MG.module_bytes(m) == (4 * 3 + 3) * 4 and MG.tensor_bytes([torch.zeros(5, dtype=torch.float64)]) == 40


def test_run_queue_treats_the_guard_error_as_oom():
    """run_queue.py falls back (activation checkpointing, then half micro-batch) when 'OutOfMemoryError' is in the log."""

    src = open(os.path.join(HERE, "run_queue.py"), encoding="utf-8").read()
    assert '"OutOfMemoryError" in txt' in src
    msg = str(MG.DeviceMemoryBudgetError("torch.OutOfMemoryError (device-memory budget guard, x)"))
    assert "OutOfMemoryError" in msg
