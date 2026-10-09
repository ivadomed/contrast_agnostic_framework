#!/usr/bin/env python
"""
Builds the per-patient cropped/z-scored cache used by train_synth.py and infer_synth.py -- run
ONCE, before any training job starts (parallel training jobs must never race on cache-building).
Also writes the frozen 100/20 train/val split and the 70-patient eval id list to
outputs/data/synth_train_val_split.json so every downstream script reads the exact same split.

Usage (inside a Slurm job, not the login node):
  .venv/bin/python build_cache.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import synth_common as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="only cache 4 train + 2 val + 2 eval patients")
    args = ap.parse_args()

    sc.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train, val = sc.train_val_split()
    ev = sc.eval_patient_ids()
    log.info("train=%d val=%d eval=%d (candidates checked for BIDS-completeness first)",
              len(train), len(val), len(ev))

    if args.smoke:
        train, val, ev = train[:4], val[:2], ev[:2]

    split = {"train": train, "val": val, "eval": ev, "seed": sc.SEED}
    split_path = sc.OUT_DIR / "data" / "synth_train_val_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))
    log.info("wrote %s", split_path)

    all_ids = list(dict.fromkeys(train + val + ev))
    t0 = time.time()
    for i, pid in enumerate(all_ids):
        sc.build_one_cache(pid)
        if (i + 1) % 20 == 0 or (i + 1) == len(all_ids):
            log.info("cached %d/%d (%.0fs elapsed)", i + 1, len(all_ids), time.time() - t0)
    log.info("done: %d patient caches under %s", len(all_ids), sc.CACHE_DIR)


if __name__ == "__main__":
    main()
