#!/usr/bin/env bash
# ISLES 2022 training set (250 cases), Zenodo 10.5281/zenodo.7153326, CC BY 4.0.
# Login-node only (compute nodes have no internet). Resumable (curl -C -).
set -euo pipefail
DEST="${SCRATCH:-/scratch/$USER}/isles2022_download"
mkdir -p "$DEST"
curl -L -C - --retry 5 --max-time 7200 -o "$DEST/ISLES-2022.zip" \
  "https://zenodo.org/api/records/7153326/files/ISLES-2022.zip/content"
echo "size: $(stat -c %s "$DEST/ISLES-2022.zip") (expected 1692717470)"
