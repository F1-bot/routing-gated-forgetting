"""Reassemble the H100 Nestor-100M model-only checkpoint from harvested 95 MB chunks and verify it.

usage: python reassemble_ckpt.py manifest.txt out_path chunk_path_00 chunk_path_01 ...
The manifest (export_manifest.txt from the production job) holds the full-file sha256 ('sha256 <hex>')
and one 'sha256sum' line per chunk ('<hex>  chunks/part_NN').
"""
import hashlib, os, re, sys

manifest, out_path, *chunks = sys.argv[1:]
txt = open(manifest, encoding="utf-8", errors="replace").read()
full_sha = re.search(r"^sha256 ([0-9a-f]{64})", txt, re.M).group(1)
part_sha = dict((name, h) for h, name in re.findall(r"^([0-9a-f]{64})\s+chunks/(part_\d+)", txt, re.M))
assert len(chunks) == len(part_sha), (len(chunks), len(part_sha))
os.makedirs(os.path.dirname(out_path), exist_ok=True)
h_all = hashlib.sha256()
with open(out_path + ".tmp", "wb") as out:
    for i, p in enumerate(chunks):
        name = f"part_{i:02d}"
        data = open(p, "rb").read()
        hp = hashlib.sha256(data).hexdigest()
        assert hp == part_sha[name], f"chunk {name} sha mismatch"
        h_all.update(data)
        out.write(data)
assert h_all.hexdigest() == full_sha, "full-file sha mismatch"
os.replace(out_path + ".tmp", out_path)
print(f"OK {len(chunks)} chunks -> {out_path} ({os.path.getsize(out_path)} bytes) sha256 {full_sha}")
