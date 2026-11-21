import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
if isinstance(m, dict):
    items = m.get("files") or m.get("entries") or m.get("bins") or [v for v in m.values() if isinstance(v, dict)]
else:
    items = m
# a file with a reference must match it exactly; a NEW file (expected is None) must have no recorded problems
bad = [e for e in items if isinstance(e, dict) and not (e.get("match") is True or (e.get("expected") is None and not e.get("problems")))]
new = [e.get("path") for e in items if isinstance(e, dict) and e.get("expected") is None]
print(("DATA_OK" if items and not bad else "DATA_BAD") + f" n={len(items)} bad={len(bad)} new_without_reference={new}")
