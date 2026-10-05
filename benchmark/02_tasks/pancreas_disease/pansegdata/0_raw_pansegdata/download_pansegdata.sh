#!/bin/bash
# Download PanSegData (OSF project kysnj; CC BY-NC 4.0; no login) into this directory. Login-node file transfer only; resumable (-C -).
#   bash download_pansegdata.sh
# Paper: Zhang et al., Med Image Anal 99:103382, doi:10.1016/j.media.2024.103382 (arXiv 2405.12367)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
get() { curl -sL -C - --retry 5 -o "$1" "https://osf.io/download/$2/"; }
get LICENSE.txt c9hkm
get T1-name_mapping.json czn6d
get T2-name_mapping.json 9hrvs
get t2_info_osf.xlsx h6udm
get t1.zip ch8ay      # 4339937429 bytes, 385 venous-phase T1W scans + labels
get t2.zip bre8p      # 1551270735 bytes, 382 T2W scans + labels
[[ "$(stat -c %s t1.zip)" == 4339937429 && "$(stat -c %s t2.zip)" == 1551270735 ]] || { echo "SIZE MISMATCH" >&2; exit 1; }
echo downloaded
