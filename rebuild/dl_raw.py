"""Download the raw HF dataset files listed in rebuild/raw_plan.json (plain https resolve URLs, no xet),
resumable, sha256-verified against the LFS oid published by the HF tree API.

  python dl_raw.py [--threads 4] [--plan rebuild\\raw_plan.json] [--only SUBSTR]

Writes logs\\rebuild\\dl_raw.log and data\\raw_manifest.json. Never routes around a blocked domain: a proxy/sandbox
refusal is logged as BLOCKED and the item is left undone for the caller to handle (request_network_access).
"""
import argparse, hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TEMP"] = os.environ["TMP"] = os.path.join(ROOT, ".tmp")
LOG = os.path.join(ROOT, "logs", "rebuild", "dl_raw.log")
MANIFEST = os.path.join(ROOT, "data", "raw_manifest.json")
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
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        k32.GetPriorityClass.restype = wintypes.DWORD
        h = k32.GetCurrentProcess()
        ok = k32.SetPriorityClass(h, 0x4000)
        log(f"BelowNormal priority set={bool(ok)} class=0x{k32.GetPriorityClass(h):x}")
    except Exception as e:  # pragma: no cover
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


class Blocked(Exception):
    pass


def fetch(it, session):
    url = f"https://huggingface.co/datasets/{it['repo']}/resolve/main/{it['path']}"
    dest, part = it["dest"], it["dest"] + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    host = None
    with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 120)) as r:
        host = urlparse(r.url).netloc
        if r.status_code in (403, 407) or (r.status_code >= 400 and "sandbox" in r.text[:500].lower()):
            raise Blocked(f"{r.status_code} from {host}: {r.text[:200]!r}")
        if have and r.status_code == 206:
            start = int(r.headers.get("Content-Range", "bytes 0-").split()[1].split("-")[0])
            if start != have:
                have = 0
        elif r.status_code == 200:
            have = 0
        elif r.status_code == 416:
            os.remove(part)
            raise requests.ConnectionError("416, restarting")
        else:
            r.raise_for_status()
        with open(part, "ab" if have else "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return host


def one(it, session, attempts=6):
    dest = it["dest"]
    t0 = time.time()
    if os.path.exists(dest) and os.path.getsize(dest) == it["size"]:
        if sha256_file(dest) == it["sha256"]:
            log(f"SKIP (verified) {it['path']}")
            return {**it, "ok": True, "host": None, "seconds": 0.0, "sha256_actual": it["sha256"]}
        os.remove(dest)
    host = None
    for k in range(attempts):
        try:
            host = fetch(it, session)
            part = dest + ".part"
            size = os.path.getsize(part)
            if size != it["size"]:
                raise requests.ConnectionError(f"incomplete {size}/{it['size']}")
            digest = sha256_file(part)
            if digest != it["sha256"]:
                os.remove(part)
                raise requests.ConnectionError(f"sha256 mismatch {digest[:12]} != {it['sha256'][:12]}")
            os.replace(part, dest)
            dt = time.time() - t0
            log(f"OK {it['repo']}:{it['path']} {it['size']:,} B via {host} in {dt:.0f}s")
            return {**it, "ok": True, "host": host, "seconds": round(dt, 1), "sha256_actual": digest}
        except Blocked as e:
            log(f"BLOCKED {it['path']}: {e}")
            return {**it, "ok": False, "error": f"BLOCKED: {e}"}
        except Exception as e:
            log(f"retry {k + 1}/{attempts} {it['path']}: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(min(60, 5 * (k + 1)))
    return {**it, "ok": False, "error": "attempts exhausted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--plan", default=os.path.join(ROOT, "rebuild", "raw_plan.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--min-size", type=int, default=0, help="only items with size >= this (bytes)")
    ap.add_argument("--max-size", type=int, default=1 << 62, help="only items with size < this (bytes)")
    a = ap.parse_args()
    below_normal()
    plan = json.load(open(a.plan, encoding="utf-8"))
    if a.only:
        plan = [x for x in plan if a.only in x["path"] or a.only in x["repo"]]
    plan = [x for x in plan if a.min_size <= x["size"] < a.max_size]
    log(f"plan: {len(plan)} files, {sum(x['size'] for x in plan) / 1e9:.2f} GB, threads={a.threads}")
    sess = requests.Session()
    sess.headers["User-Agent"] = "python-requests (dataset rebuild)"
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda it: one(it, sess), plan))
    old = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else {}
    for r in res:
        old[r["dest"]] = {k: r.get(k) for k in ("repo", "path", "size", "sha256", "sha256_actual", "ok", "host", "seconds", "error")}
    json.dump(old, open(MANIFEST, "w", encoding="utf-8"), indent=1)
    bad = [r for r in res if not r["ok"]]
    log(f"DL_DONE ok={len(res) - len(bad)} fail={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
