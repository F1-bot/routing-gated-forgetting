"""Checkpoint sync + budget guard for the b15s44 H100 run (runs beside the trainer; CPU-light, no torch).

* Every POLL s the train log is copied to <vol>/logs/<leg>.log (atomic: .tmp then os.replace), so progress can be
  read from the volume while the job runs.
* For every 'Model checkpoint saved: <path>' line that names a step checkpoint (model_<8 digits>.pt) the file is
  copied to the same relative path under <vol>/outputs (atomic). The trainer logs that line only after torch.save
  returned, so only complete files are copied. Older local step checkpoints are then deleted (the newest stays
  for a warm resume) and the copied files are dropped from the page cache.
* Budget: after step checkpoint N is on the volume, the finish time is projected as
      t_done = now + (STEPS - N) * sps * 1.005 + 60 + RESERVE
  (sps = median of the last 100 logged train/sec_per_step values; RESERVE = eval + export + close time). If
  t_done > STOP_BY the trainer gets SIGTERM and the run stays resumable from checkpoint N on the volume.
  A hard backstop stops the trainer once STOP_BY - RESERVE has passed. STOP_BY can be replaced while the job runs
  by writing an epoch value into /tmp/guard_stop_by (read every poll).
* Files named with --dropcache are dropped from the page cache every 4 polls (read-once data; no effect on values).
"""
import argparse, glob, os, re, shutil, signal, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--pid", type=int, required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--stop-by", type=float, default=0.0)
ap.add_argument("--reserve", type=float, default=260.0)
ap.add_argument("--vol", default="/vol")
ap.add_argument("--local-root", default="/root/outputs")
ap.add_argument("--leg", required=True)
ap.add_argument("--poll", type=float, default=30.0)
ap.add_argument("--override", default="/tmp/guard_stop_by")
ap.add_argument("--dropcache", nargs="*", default=[])
a = ap.parse_args()

SAVED = re.compile(r"Model checkpoint saved: (\S+)")
SPS = re.compile(r"Step (\d+) \|.*?train/sec_per_step: ([0-9.]+)")
synced = set()
stopped = False
stop_by = a.stop_by


def utc(t=None):
    return time.strftime("%H:%M:%S", time.gmtime(t if t is not None else time.time()))


def say(msg):
    print(f"[guard {utc()}] {msg}", flush=True)


def dropcache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fdatasync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass


def atomic_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie child counts as exited
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def stop(reason):
    global stopped
    if stopped:
        return
    stopped = True
    say("BUDGET_STOP " + reason)
    os.makedirs(os.path.join(a.vol, "logs"), exist_ok=True)
    with open(os.path.join(a.vol, "logs", f"BUDGET_STOP_{a.leg}.txt"), "w") as fh:
        fh.write(reason + "\n")
    try:
        os.kill(a.pid, signal.SIGTERM)
    except OSError:
        pass


say(f"start: steps {a.steps}, stop_by {utc(stop_by) if stop_by else 'off'} UTC, reserve {a.reserve:.0f}s, leg {a.leg}")
it = 0
while True:
    it += 1
    running = alive(a.pid)
    try:
        v = float(open(a.override).read().strip())
        if v != stop_by:
            say(f"stop_by override: {utc(stop_by) if stop_by else 'off'} -> {utc(v)} UTC")
            stop_by = v
    except (OSError, ValueError):
        pass
    try:
        txt = open(a.log, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        txt = ""
    if txt:
        try:
            atomic_copy(a.log, os.path.join(a.vol, "logs", f"{a.leg}.log"))
        except OSError as e:
            say(f"log copy failed: {e}")
    for p in SAVED.findall(txt):
        m = re.search(r"model_(\d{8})\.pt$", p)
        if not m or p in synced:
            continue
        rel = os.path.relpath(p, a.local_root)
        dst = os.path.join(a.vol, "outputs", rel)
        t0 = time.time()
        try:
            atomic_copy(p, dst)
        except OSError as e:
            say(f"checkpoint copy failed for {rel}: {e}")
            continue
        synced.add(p)
        n = int(m.group(1))
        size = os.path.getsize(p)
        dropcache(p); dropcache(dst)
        for old in glob.glob(os.path.join(os.path.dirname(p), "model_[0-9]*.pt")):
            mo = re.search(r"model_(\d{8})\.pt$", old)
            if mo and int(mo.group(1)) < n and os.path.exists(os.path.join(a.vol, "outputs", os.path.relpath(old, a.local_root))):
                os.remove(old)
                say(f"removed older local {os.path.basename(old)} (kept on volume)")
        say(f"synced step {n} -> {rel} ({size / 1e9:.2f} GB in {time.time() - t0:.0f}s)")
        vals = [float(x) for _, x in SPS.findall(txt)][-100:]
        if stop_by > 0 and vals and n < a.steps:
            sps = statistics.median(vals)
            t_done = time.time() + (a.steps - n) * sps * 1.005 + 60 + a.reserve
            say(f"projection at step {n}: {sps:.4f} s/step -> done (incl. eval/export/close) {utc(t_done)} UTC; stop_by {utc(stop_by)} UTC")
            if t_done > stop_by:
                stop(f"step {n}: projected finish {utc(t_done)} > stop_by {utc(stop_by)} UTC")
    if stop_by > 0 and running and time.time() > stop_by - a.reserve:
        stop(f"hard backstop at {utc()} UTC (stop_by - reserve passed)")
    if it % 4 == 0:
        for f in a.dropcache:
            dropcache(f)
    if not running:
        say("trainer exited")
        break
    time.sleep(a.poll)
