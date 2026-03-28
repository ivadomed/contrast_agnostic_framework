#!/usr/bin/env bash

set -euo pipefail

# Compatibility wrapper for singular launcher name.
# Usage: bash scripts/run_segmenter.sh [slot] [contrast] [version] [batch_size]
exec bash scripts/run_segmenters.sh "$@"
