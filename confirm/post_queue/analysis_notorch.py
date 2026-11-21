"""analysis_notorch.py -- runs the frozen analyze_confirm.py with torch imports blocked and its output appended to a log.

  python confirm/post_queue/analysis_notorch.py --root R --log R/logs/analysis_full.log -- run --mode full --root R

Everything after "--" is passed to analyze_confirm.py unchanged. torch is blocked exactly as in test_analyze_confirm.py
(_NOTORCH: a meta-path finder that raises ImportError for torch and torch.*; runpy.run_path of analyze_confirm.py as
__main__), so the analysis cannot touch the GPU. stdout and stderr go to the log (UTF-8, appended, never truncated) and
to the console. The exit code is analyze_confirm's (0 written, 23 FREEZE.json verification failed, 32 firewall);
1 for an exception outside it; 2 for a usage error.
"""
from __future__ import annotations

import datetime as _dt
import os
import runpy
import sys
import time
import traceback


class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked: the full analysis is torch-free (confirm/post_queue/analysis_notorch.py)")
        return None


class _Tee:
    def __init__(self, log, console):
        self.log, self.console = log, console

    def write(self, s):
        self.log.write(s)
        self.log.flush()
        try:
            self.console.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        self.log.flush()
        try:
            self.console.flush()
        except Exception:
            pass


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        print("usage: analysis_notorch.py --root R --log FILE -- <analyze_confirm.py arguments>")
        return 2
    i = argv.index("--")
    own, fwd = argv[:i], argv[i + 1:]
    opts = dict(zip(own[0::2], own[1::2]))
    if set(opts) != {"--root", "--log"} or len(own) != 4:
        print("usage: analysis_notorch.py --root R --log FILE -- <analyze_confirm.py arguments>")
        return 2
    root, logp = os.path.abspath(opts["--root"]), os.path.abspath(opts["--log"])
    script = os.path.join(root, "analyze_confirm.py")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    log = open(logp, "a", encoding="utf-8", newline="\n")
    console = sys.__stdout__
    sys.stdout = sys.stderr = _Tee(log, console)
    t0 = time.time()
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[analysis_notorch] {utc} pid {os.getpid()}: {script} {' '.join(fwd)} (torch blocked; cwd {root}; "
          f"OMP/MKL/OPENBLAS threads {os.environ.get('OMP_NUM_THREADS')}/{os.environ.get('MKL_NUM_THREADS')}/"
          f"{os.environ.get('OPENBLAS_NUM_THREADS')})")
    os.chdir(root)
    sys.path.insert(0, root)
    sys.meta_path.insert(0, _NoTorch())
    sys.argv = ["analyze_confirm.py"] + fwd
    rc = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc()
        rc = 1
    print(f"[analysis_notorch] exit {rc}; torch imported: {'torch' in sys.modules}; wall {time.time() - t0:.1f} s")
    sys.stdout.flush()
    log.close()
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    return rc


if __name__ == "__main__":
    sys.exit(main())
