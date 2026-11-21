"""Summarise trainer logs (Step lines) and data hashes into one JSON."""
import glob, json, os, re, statistics, sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
pat = re.compile(r"Step (\d+) \| train/loss: ([\d.]+).*?sec_per_step: ([\d.]+).*?mem_peak_alloc_gib: ([\d.]+)")
res = {"runs": {}}
for p in sorted(glob.glob(os.path.join(out_dir, "*.log"))):
    txt = open(p, errors="replace").read()
    rows = [(int(a), float(b), float(c), float(d)) for a, b, c, d in pat.findall(txt)]
    sps = [r[2] for r in rows[2:]] or [r[2] for r in rows]
    res["runs"][os.path.basename(p)] = {
        "steps": [r[0] for r in rows], "loss": [r[1] for r in rows],
        "sec_per_step_median_excl_first2": statistics.median(sps) if sps else None,
        "tok_per_s": (262144 / statistics.median(sps)) if sps else None,
        "mem_peak_alloc_gib": max((r[3] for r in rows), default=None),
        "oom": "OutOfMemoryError" in txt, "traceback": "Traceback" in txt,
        "tail": txt[-1500:] if not rows else "",
    }
hl = os.path.join(out_dir, "sha256_modal.txt")
if os.path.exists(hl):
    res["sha256_modal"] = {os.path.basename(l.split()[1]): l.split()[0] for l in open(hl) if l.strip()}
lh = "local_token_hashes.json"
if os.path.exists(lh):
    loc = {os.path.basename(k): v["sha256"] for k, v in json.load(open(lh)).items()}
    res["sha256_match"] = {k: (res.get("sha256_modal", {}).get(k) == v) for k, v in loc.items() if k in res.get("sha256_modal", {})}
json.dump(res, open(os.path.join(out_dir, "calib_results.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("loss", "tok_per_s", "mem_peak_alloc_gib", "oom")}
                  for k, v in res["runs"].items()}, indent=1))
print("sha256_match", res.get("sha256_match"))
