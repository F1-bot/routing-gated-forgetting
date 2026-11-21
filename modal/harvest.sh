#!/bin/bash
# Copy one or more 95 MB chunks of the exported model-only checkpoint into out/ for harvest.
set -eu
mkdir -p out
for i in "$@"; do cp "/root/export/chunks/part_$i" out/; sha256sum "out/part_$i"; done
