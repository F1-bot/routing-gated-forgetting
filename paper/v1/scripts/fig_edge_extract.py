"""Board-power traces for fig_edge panel (b) (descriptive; outside the registered EDGE block).

The two traced runs are the G1 full adaptations on German with data seed 101: g1_moe_b1_full_german_s101 (segment
m2, the MoE run whose energy is the median of the seven MoE full runs) and g1_nestor_full_german_s101 (segment m2c,
the attempt with activation checkpointing that followed the out-of-memory attempt m2). Segment windows are rebuilt
as in edge_pre_freeze.py (stored copy b2b256e4): job start/end from results/queue_state_g1.json and last-write times
of the job logs in logs/queue_g1. The 1 Hz nvidia-smi log has no header and naive LOCAL timestamps (UTC+2 in
November); they are converted with time.mktime exactly as edge_pre_freeze.py does (a naive-as-UTC parse would
shift every window by two hours). Before writing, the integrated energy, mean power and mean utilisation of each
window must reproduce data/edge_pre_freeze.json (segments) to 1e-6 relative; otherwise the script stops.
Input:  <EDGE_ROOT, default .>/logs/power/gpu_power_20261113_235717.csv, results/queue_state_g1.json,
        logs/queue_g1/<job>_*.log; data/edge_pre_freeze.json
Output: figures/fig_edge_values.json (FIG_OUT overrides the folder), via a temp file
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with PYTHONSAFEPATH=1
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figstyle as fs  # noqa: E402

ROOT = os.environ.get("EDGE_ROOT", r".")
POWER = os.path.join(ROOT, "logs", "power", "gpu_power_20261113_235717.csv")
RUNS = [("moe", "g1_moe_b1_full_german_s101", "m2"), ("nestor", "g1_nestor_full_german_s101", "m2c")]

d = pd.read_csv(POWER, header=None, names=["ts", "pw", "util", "mem", "temp", "clk"], skipinitialspace=True,
                on_bad_lines="skip")
d["ts"] = pd.to_datetime(d["ts"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
for c in ("pw", "util", "mem"):
    d[c] = pd.to_numeric(d[c].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
d = d.dropna(subset=["ts", "pw"]).sort_values("ts").reset_index(drop=True)
T = np.array([time.mktime(x.timetuple()) + x.microsecond / 1e6 for x in d["ts"]])   # naive local -> epoch
PW, UT, MEM = d["pw"].values, d["util"].values, d["mem"].values
DT = np.clip(np.diff(T, append=T[-1]), 0, 5.0)                                         # as edge_pre_freeze.py

state = json.load(open(os.path.join(ROOT, "results", "queue_state_g1.json"), encoding="utf-8-sig"))
segs = fs.load("edge_pre_freeze.json")["segments"]


def window(jid, which):
    v, logdir = state[jid], os.path.join(ROOT, "logs", "queue_g1")
    logs = sorted((os.path.getmtime(os.path.join(logdir, f)), f) for f in os.listdir(logdir)
                  if f == jid + ".log" or f.startswith(jid + "_"))
    prev = v["start"]
    for mt, f in logs:
        b = min(mt, v["end"])
        if b <= prev:
            continue
        if (f[len(jid):].strip("_").replace(".log", "") or "job") == which:
            return prev, b
        prev = b
    raise SystemExit(f"segment {which} of {jid} not found")


out = {"note": "1 Hz GPU board power (nvidia-smi) of the two traced runs; t_min = minutes since the segment start; "
               "board power only (no CPU, RAM or PSU losses)", "power_log": POWER, "runs": {}}
for arch, jid, which in RUNS:
    a, b = window(jid, which)
    m = (T >= a) & (T < b)
    e_wh = float((PW[m] * DT[m]).sum() / 3600.0)
    rec = [s for s in segs if s["job"] == jid and s["segment"] == which]
    assert len(rec) == 1, (jid, which)
    rec = rec[0]
    for key, val in (("energy_wh", e_wh), ("mean_w", float(PW[m].mean())), ("mean_util", float(UT[m].mean())),
                     ("wall_h", (b - a) / 3600.0)):
        assert abs(val - rec[key]) <= 1e-6 * max(1.0, abs(rec[key])), (jid, key, val, rec[key])
    out["runs"][arch] = {
        "job": jid, "segment": which, "start_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a)),
        "end_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b)), "minutes": (b - a) / 60.0,
        "energy_wh": e_wh, "mean_w": float(PW[m].mean()), "mean_util": float(UT[m].mean()),
        "peak_mem_mib": float(MEM[m].max()), "n_samples": int(m.sum()),
        "t_min": [round(float(x), 4) for x in (T[m] - a) / 60.0],
        "power_w": [float(x) for x in PW[m]], "util_pct": [float(x) for x in UT[m]]}
    print(f"{jid} {which}: {(b - a) / 60:.2f} min, {e_wh:.3f} Wh, mean {PW[m].mean():.1f} W, "
          f"util {UT[m].mean():.1f}% (matches data/edge_pre_freeze.json)")
fs.FIGDIR.mkdir(exist_ok=True)
tmp, dst = fs.FIGDIR / "fig_edge_values.tmp.json", fs.FIGDIR / "fig_edge_values.json"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
os.replace(tmp, dst)
print("fig_edge_values written")
