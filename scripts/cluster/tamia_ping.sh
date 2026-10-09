#!/usr/bin/env bash
# Read-only connectivity check for the Vulcan -> TamIA relay.
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 tamia.alliancecan.ca "hostname && echo ok"
