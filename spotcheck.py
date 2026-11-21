import json, sys
raw = open(sys.argv[1], "rb").read()
t = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
r = json.loads([l for l in t.splitlines() if l.startswith("RESULT ")][-1][7:])
ref = float(sys.argv[2]); d = abs(r["ce"] - ref)
print(("SPOT_OK" if d <= 1e-6 else "SPOT_BAD") + f" ce={r['ce']!r} ref={ref!r} absdiff={d:.3g} tokens={r.get('tokens')}")
