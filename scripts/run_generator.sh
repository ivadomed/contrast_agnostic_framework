#!/usr/bin/env bash

set -euo pipefail

# Compatibility wrapper for singular launcher name.
# Usage: bash scripts/run_generator.sh [slot] [version] [contrast]
exec bash scripts/run_generators.sh "$@"
