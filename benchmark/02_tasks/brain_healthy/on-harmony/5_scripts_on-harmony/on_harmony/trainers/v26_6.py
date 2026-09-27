"""
nnUNetTrainerOnHarmonyV26_6 — V26_6 trainer for the ON-Harmony dataset.

Adds ON-Harmony-specific data loading to nnUNetTrainerV26_6:
  - RAM-cached full-volume loader (OnHarmonyVolumeLoader) for fast throughput
  - Anti-contamination split validation (via nnUNetTrainerOnHarmonyBase)

All synthesis logic, transforms, and WandB image logging are inherited from
nnUNetTrainerV26_6.  This class is intentionally thin.
"""
from __future__ import annotations

import os

from on_harmony.dataloader.onharmony import OnHarmonyVolumeLoader, OnHarmonySingleThreadedWrapper
from on_harmony.trainers.base import nnUNetTrainerOnHarmonyBase
from src.nnunet.trainers.fast import BASE_SEED
from src.nnunet.trainers.v26_6_base import nnUNetTrainerV26_6


class nnUNetTrainerOnHarmonyV26_6(nnUNetTrainerOnHarmonyBase, nnUNetTrainerV26_6):
    """
    V26_6 trainer for ON-Harmony.

    MRO: nnUNetTrainerOnHarmonyV26_6
      → nnUNetTrainerOnHarmonyBase  (do_split — anti-contamination guard)
      → nnUNetTrainerV26_6          (train_step, validation_step, transforms, WandB logging)
      → nnUNetTrainerFast            (seed, epochs, aug-latency hook)
      → nnUNetTrainer
    """

    def get_dataloaders(self):
        # Configure V26_6 transforms, patch sizes, and thread pool
        self._setup_v26_training()

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        train_keys = sorted(dataset_tr.identifiers)
        val_keys = sorted(dataset_val.identifiers)

        # Rank-aware seeds: under DDP each rank must sample DIFFERENT volumes so
        # the global batch sees distinct subjects (true data parallelism).  The
        # 100000 * local_rank offset guarantees disjoint RNG streams per rank.
        rank_off = 100000 * self.local_rank
        self._vol_loader_train = OnHarmonyVolumeLoader(
            dataset_tr, train_keys, rng_seed=BASE_SEED + self.fold + rank_off
        )
        self._vol_loader_val = OnHarmonyVolumeLoader(
            dataset_val, val_keys, rng_seed=BASE_SEED + self.fold + 1000 + rank_off
        )

        self.num_val_iterations_per_epoch = min(
            len(val_keys),
            int(os.environ.get("NNUNET_VAL_ITERS_PER_EPOCH", str(len(val_keys)))),
        )

        return (
            OnHarmonySingleThreadedWrapper(self._vol_loader_train),
            OnHarmonySingleThreadedWrapper(self._vol_loader_val),
        )

    def __del__(self) -> None:
        for attr in ("_vol_loader_train", "_vol_loader_val"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try: obj.stop()
                except Exception: pass
