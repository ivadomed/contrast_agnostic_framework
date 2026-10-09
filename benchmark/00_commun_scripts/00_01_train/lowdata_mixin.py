#!/usr/bin/env python3
"""
LowDataMixin — train-set subsetting for low-data-regime benchmarks.

ADDITIVE ONLY. This mixin does not modify any existing trainer; it is layered on
top of an existing nnU-Net trainer class via multiple inheritance, e.g.

    class nnUNetTrainerCHAOSBaselineLowData(LowDataMixin, nnUNetTrainerCHAOSBaseline):
        pass

Mechanism
---------
The regime N (number of training subjects to keep, per fold) is read from the
``RUN_ID`` environment variable — which the training driver already exports into
every job (``00_commun_scripts/00_01_train/train_common.sh``). A benchmark RUN_ID
carries a ``_lowdata_n<NN>`` marker, e.g.::

    chaos_t1in_baseline_lowdata_n08_20260625_143000   →  N = 8

so NO change to the driver, env.sh, or any existing trainer is needed. When the
marker is absent (every normal run) the mixin is a no-op and behaviour is
byte-identical to the underlying trainer.

do_split() delegates to ``super().do_split()`` (the underlying trainer's canonical
``(train_keys, val_keys)`` — including CHAOS/BraTS anti-contamination guards), then
keeps the first N of a *fixed-seed* shuffle of that fold's train keys. The seed is
constant, so subsets are NESTED as N grows (N=1 ⊂ N=2 ⊂ N=4 ⊂ …) and fully
reproducible. The validation set is never touched, so every regime is evaluated on
the same per-fold val cases (fair learning curve).
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

# Fixed across folds/methods → nested, reproducible subsets. Matches the project
# split seed (01_create_splits SEED), but any constant would do.
LOWDATA_SEED = 12345

_RUN_ID_RE = re.compile(r"_lowdata_n0*(\d+)")


def parse_lowdata_n(run_id: str | None) -> int | None:
    """Return N from a ``…_lowdata_n<NN>…`` RUN_ID, or None if not a low-data run."""
    if not run_id:
        return None
    m = _RUN_ID_RE.search(run_id)
    return int(m.group(1)) if m else None


class LowDataMixin:
    """Place FIRST in the MRO so this do_split() wraps the underlying one."""

    def do_split(self):
        tr_keys, val_keys = super().do_split()

        n = parse_lowdata_n(os.environ.get("RUN_ID"))
        if n is None:
            return tr_keys, val_keys

        full = sorted(tr_keys)
        rng = random.Random(LOWDATA_SEED)
        shuffled = full.copy()
        rng.shuffle(shuffled)
        subset = sorted(shuffled[:n])

        kept, avail = len(subset), len(full)
        self.print_to_log_file(
            f"[LowData] fold {self.fold}: N={n} → keeping {kept}/{avail} train "
            f"subjects (val unchanged: {len(val_keys)}). subset={subset}"
        )
        if n > avail:
            self.print_to_log_file(
                f"[LowData] WARNING: requested N={n} > available {avail} train "
                f"subjects for fold {self.fold}; using all {avail}."
            )

        # The inner trainer's do_split already wrote split_verification.json with the
        # FULL train set — overwrite it so the on-disk audit reflects what actually
        # trained (the dataloader uses this subset, our return value).
        try:
            audit = Path(self.output_folder) / "split_verification.json"
            audit.write_text(json.dumps(
                {"fold": self.fold, "low_data_N": n,
                 "n_train": len(subset), "n_train_available": avail,
                 "n_val": len(val_keys), "train_keys": subset,
                 "val_keys": sorted(val_keys)}, indent=2))
        except Exception as e:  # never let an audit-write failure break training
            self.print_to_log_file(f"[LowData] could not rewrite split_verification.json: {e}")
        return subset, val_keys
