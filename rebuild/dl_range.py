"""Parallel byte-range download of the large raw files in rebuild/raw_plan.json (>= --min-size), sha256-verified
against the LFS oid. Same plain https resolve URLs as dl_raw.py (no xet); resumable through a segment map.

  python dl_range.py [--threads 8] [--min-size 1000000000] [--seg-mb 32]

The first `have` bytes of an existing .part file (written sequentially by dl_raw.py) are kept; the rest is fetched in
segments by several connections and written at their offsets. I/O-bound: the only CPU work is the final sha256.
"""
import argparse, hashlib, json, os, sys, threading, time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TEMP"] = os.environ["TMP"] = os.path.join(ROOT, ".tmp")
LOG = os.path.join(ROOT, "logs", "rebuild", "dl_range.log")
_lock = threading.Lock()


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    with _lock:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


def below_normal():
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
    except Exception as e:
        log(f"priority not set: {e}")


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class FileTask:
    def __init__(self, it, seg):
        self.it, self.seg = it, seg
        self.dest, self.part = it["dest"], it["dest"] + ".part"
        self.map_path = self.part + ".map"
        self.size = it["size"]
        self.url = f"https://huggingface.co/datasets/{it['repo']}/resolve/main/{it['path']}"
        self.lock = threading.Lock()
        have = os.path.getsize(self.part) if os.path.exists(self.part) else 0
        done = set()
        if os.path.exists(self.map_path):
            done = set(json.load(open(self.map_path)))
            have = 0  # the segment map is authoritative once it exists
        self.base = have  # contiguous prefix kept from the sequential downloader
        nseg = (self.size + seg - 1) // seg
        self.todo = [i for i in range(nseg) if i not in done and (i + 1) * seg > self.base]
        self.done = done
        self.nseg = nseg

    def prepare(self):
        if not os.path.exists(self.part):
            open(self.part, "wb").close()
        if not os.path.exists(self.map_path):
            # segments fully inside the sequential prefix are done; a partially covered one is simply re-fetched.
            # The map is written BEFORE the file is pre-allocated, so a full-size .part can never be mistaken for data.
            self.done = {i for i in range(self.nseg) if (i + 1) * self.seg <= self.base}
            self.todo = [i for i in range(self.nseg) if i not in self.done]
            json.dump(sorted(self.done), open(self.map_path, "w"))
        if os.path.getsize(self.part) < self.size:
            with open(self.part, "r+b") as fh:  # sparse pre-allocation; the map says what is real
                fh.truncate(self.size)

    def mark(self, i):
        with self.lock:
            self.done.add(i)
            json.dump(sorted(self.done), open(self.map_path, "w"))


def fetch_segment(task, i, session, attempts=8):
    s = i * task.seg
    e = min(task.size, s + task.seg) - 1
    for k in range(attempts):
        try:
            with session.get(task.url, headers={"Range": f"bytes={s}-{e}"}, stream=True, allow_redirects=True, timeout=(30, 120)) as r:
                if r.status_code == 403 and "sandbox" in r.text[:500].lower():
                    log(f"BLOCKED {task.it['path']}: {r.text[:200]!r}")
                    return False
                if r.status_code != 206:
                    raise requests.ConnectionError(f"status {r.status_code}")
                pos = s
                with open(task.part, "r+b") as fh:
                    fh.seek(s)
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        pos += len(chunk)
                if pos != e + 1:
                    raise requests.ConnectionError(f"short segment {pos - s}/{e + 1 - s}")
            task.mark(i)
            return True
        except Exception as ex:
            log(f"retry {k + 1}/{attempts} seg {i} {task.it['path']}: {type(ex).__name__}: {str(ex)[:120]}")
            time.sleep(min(60, 4 * (k + 1)))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-size", type=int, default=1_000_000_000)
    ap.add_argument("--seg-mb", type=int, default=32)
    ap.add_argument("--plan", default=os.path.join(ROOT, "rebuild", "raw_plan.json"))
    a = ap.parse_args()
    below_normal()
    plan = [x for x in json.load(open(a.plan, encoding="utf-8")) if x["size"] >= a.min_size]
    tasks = []
    for it in plan:
        if os.path.exists(it["dest"]) and os.path.getsize(it["dest"]) == it["size"]:
            log(f"already present {it['path']}")
            continue
        t = FileTask(it, a.seg_mb << 20)
        t.prepare()
        tasks.append(t)
    work = [(t, i) for i in range(max((t.nseg for t in tasks), default=0)) for t in tasks if i in t.todo]
    log(f"{len(tasks)} files, {len(work)} segments of {a.seg_mb} MB, threads={a.threads}")
    sess = requests.Session()
    it_work = iter(work)
    wlock = threading.Lock()
    failed = []
    t0 = time.time()

    def worker():
        while True:
            with wlock:
                nxt = next(it_work, None)
            if nxt is None:
                return
            t, i = nxt
            if not fetch_segment(t, i, sess):
                failed.append((t.it["path"], i))

    ths = [threading.Thread(target=worker) for _ in range(a.threads)]
    for th in ths:
        th.start()
    while any(th.is_alive() for th in ths):
        time.sleep(30)
        left = sum(len(t.todo) - len([i for i in t.todo if i in t.done]) for t in tasks)
        log(f"progress: {left} segments left, {time.time() - t0:.0f}s elapsed")
    for th in ths:
        th.join()
    ok = 0
    for t in tasks:
        if len(t.done) != t.nseg:
            log(f"INCOMPLETE {t.it['path']}: {len(t.done)}/{t.nseg} segments")
            continue
        digest = sha256_file(t.part)
        if digest != t.it["sha256"]:
            log(f"SHA MISMATCH {t.it['path']}: {digest[:16]} != {t.it['sha256'][:16]}; discarding")
            os.remove(t.part)
            os.remove(t.map_path)
            continue
        os.replace(t.part, t.dest)
        os.remove(t.map_path)
        log(f"OK {t.it['repo']}:{t.it['path']} {t.size:,} B sha256 verified")
        ok += 1
    log(f"RANGE_DONE ok={ok}/{len(tasks)} failed_segments={len(failed)}")
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
