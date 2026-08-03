#!/usr/bin/env bash
# Progress report for the picai-prostate training packs: one line per fold, with the current
# epoch, seconds/epoch, latest pseudo-dice, and how stale the log is.
#
# WHY THIS EXISTS: the obvious file to tail is the pack dir's <fold>.log, and it is the WRONG
# one — it captures only nnU-Net's start-up stdout and then stops being written, because
# nnU-Net switches to its own training_log_<date>.txt inside each fold's output dir. A pack
# .log frozen hours ago looks exactly like a hung job and is not one. This reads the real
# per-epoch logs so "is it progressing?" is one command instead of a re-derivation.
#
# The "log age" column is the actual liveness signal: anything under a minute or two is
# healthy; a genuinely stuck fold shows a large age.
#
# Usage (run ON tamia):
#   bash 04_23_tamia_pack_progress.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
source "${HERE}/../00_utils/env.sh"
[ -f "${ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${ROOT}"

PRED_ROOT="${PREDICTIONS_ROOT}/${MODEL_TYPE}"

TOTAL_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}" PRED_ROOT="${PRED_ROOT}" .venv/bin/python - <<'PY'
import glob, os, re, time

root = os.environ["PRED_ROOT"]
total = int(os.environ["TOTAL_EPOCHS"])
rows = []
for lg in glob.glob(root + "/*/*/*/*/*/fold_*/training_log_*.txt"):
    p = lg.split("/")
    fold, run_id, contrast = p[-2], p[-5], p[-7]
    try:
        txt = open(lg, errors="ignore").read()
    except OSError:
        continue
    ep = re.findall(r"^\s*[\d\-: .]+: Epoch (\d+)\s*$", txt, re.M)
    et = re.findall(r"Epoch time: ([\d.]+) s", txt)
    dc = re.findall(r"Pseudo dice \[np\.float32\(([\d.]+)\)\]", txt)
    method = run_id.split("_", 1)[1].rsplit("_2026", 1)[0]
    rows.append((contrast, method, fold,
                 int(ep[-1]) if ep else 0,
                 float(et[-1]) if et else float("nan"),
                 float(dc[-1]) if dc else float("nan"),
                 (time.time() - os.path.getmtime(lg)) / 60))

rows.sort()
print(f"{'ctr':4s} {'method':40s} {'fold':7s} {'epoch':>11s} {'s/ep':>6s} {'dice':>6s} {'age(min)':>9s}")
# A fold that reached the final epoch stops writing, so a large age is CORRECT there —
# only flag staleness on folds that are still supposed to be advancing.
def is_stale(ep, age):
    return age > 15 and ep < total - 1

for c, m, f, ep, et, dc, age in rows:
    flag = "  <-- STALE" if is_stale(ep, age) else ("  done" if ep >= total - 1 else "")
    print(f"{c:4s} {m:40s} {f:7s} {ep:>5d}/{total:<5d} {et:6.1f} {dc:6.3f} {age:9.1f}{flag}")

done = sum(1 for r in rows if r[3] >= total - 1)
stale = sum(1 for r in rows if is_stale(r[3], r[6]))
print(f"\nfolds with a training log: {len(rows)}   at/near final epoch: {done}   stale(>15min): {stale}")
running = [r for r in rows if r[3] < total - 1 and r[4] == r[4]]
if running:
    eta = max((total - r[3]) * r[4] for r in running) / 3600
    print(f"slowest remaining fold ETA: ~{eta:.1f} h")
elif rows:
    print("all folds reached the final epoch")
PY
