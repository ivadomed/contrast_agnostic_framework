"""
nnUNetTrainerAutoPETBaseline — AutoPET real-data baseline (no synthesis).

Single input channel (CT OR PET, depending on which per-modality Dataset it's trained
against — DATASET_ID selects this in the 04_0X wrapper, not this class), standard
nnUNet augmentation only, EXCEPT for a widened FOV/zoom augmentation range (see
get_training_transforms below) — this dataset's real, measured train/eval FOV mismatch
(user-directed fix, 2026-09-13) needs equal treatment across all 6 methods, and baseline
is the one method that doesn't go through an AugLab JSON config (see
00_utils/generate_fov_widened_configs.py for the AugLab-side half of this fix and the
full numeric justification). The real-data reference against which the domain-
randomization methods (v26_6_2 + AugLab) and the SynthSeg-EM contender are compared. Two
training modalities (like chaos/brats2024-glioma/on-harmony/open-ms/ispy2) — see
00_utils/env.sh / env_pet.sh: each modality is its own nnU-Net Dataset, held-out FDG test
patients scored on BOTH modalities for the cross-contrast axis, and the PSMA cohort
(different institution, different tracer) scored with the SAME checkpoints for the
cross-institution OOD axis (05_predict) — no separate training run for PSMA.

Inherits split validation from nnUNetTrainerAutoPETBase and seed/epochs/WandB hooks from
nnUNetTrainerFast.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

from autopet.trainers.base import nnUNetTrainerAutoPETBase

# Must match generate_fov_widened_configs.py's NEW_SCALING/NEW_P_SCALING exactly — same
# fix, same numbers, applied via the AugLab JSON route there and this stock-nnU-Net-side
# override here, so every one of the 6 methods gets equal FOV-augmentation exposure.
_FOV_SCALING = (0.4, 1.4)
_FOV_P_SCALING = 0.3


class nnUNetTrainerAutoPETBaseline(nnUNetTrainerAutoPETBase):
    """Standard-augmentation baseline (no synthesis) for autopet — widened FOV/zoom
    augmentation only (see module docstring)."""

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        # Build nnU-Net's own stock pipeline unchanged, then widen ONLY the
        # SpatialTransform's zoom range post-hoc (its scaling/p_scaling/
        # p_synchronize_scaling_across_axes are plain settable instance attributes —
        # verified directly against batchgeneratorsv2's SpatialTransform source rather
        # than assumed). Far safer than copy-pasting nnU-Net's ~100-line stock method
        # and risking silent drift from upstream.
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        composed = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes,
            do_dummy_2d_data_aug, use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label,
        )
        found = False
        for t in composed.transforms:
            if isinstance(t, SpatialTransform):
                t.scaling = _FOV_SCALING
                t.p_scaling = _FOV_P_SCALING
                t.p_synchronize_scaling_across_axes = 0
                found = True
                break
        if not found:
            raise RuntimeError(
                "nnUNetTrainerAutoPETBaseline.get_training_transforms: no SpatialTransform "
                "found in nnU-Net's stock pipeline — upstream nnU-Net must have changed its "
                "augmentation internals; this override needs updating, not silently skipped."
            )
        return composed
