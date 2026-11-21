#!/bin/bash
# Copy one or more 95 MB chunks of the exported b15s44 model-only checkpoint from the volume into out/ for harvest.
set -eu
mkdir -p out
for i in "$@"; do cp "/vol/export/b15s44/chunks/part_$i" out/; sha256sum "out/part_$i"; done
