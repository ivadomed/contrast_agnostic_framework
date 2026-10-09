"""
nnUNetTrainerV26_6_2 — V26_6_2 synthesis base trainer.  Dataset-agnostic.

Identical to V26_6 in every respect except the synthesis variant: it overrides
the single _synthesize() hook to add a per-label intensity remap on top of the
whole-image K-means synthesis.  Because train_step, validation_step and the
WandB panel all route through that one hook (defined in nnUNetTrainerV26_6),
train and validation can never use different synthesis — and there is nothing
else to keep in sync.
"""
from __future__ import annotations

import numpy as np
import torch

from src.nnunet.trainers.v26_6_base import nnUNetTrainerV26_6
from src.synthesis.v26_6_2_synthesis import synthesize_batch_fast_v2


class nnUNetTrainerV26_6_2(nnUNetTrainerV26_6):
    """V26_6_2 base trainer — whole-image K-means synth + per-label remap."""

    def _synthesize(self, data01: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """Per-label V26_6_2 synthesis → z-scored volume.  `seg` supplies the
        labels for the per-label affine remap."""
        synth_z, _ = synthesize_batch_fast_v2(data01, seg.long())
        return synth_z

    def _viz_prefix_panels(self, c: dict) -> list:
        """V26_6_2 pipeline panels: original | K-means parcellation | whole-image
        synth.  HONEST — only drawn when this sample was actually synthesised; a
        raw step shows the original sample and a black parcellation placeholder,
        never a fabricated synthetic image.  The per-label synth itself is the
        'net input' panel that the base appends (it IS what the network was fed).
        """
        src01 = c.get("src01")
        if src01 is None:                       # full-volume path: no aligned source
            return []
        orig = self._slice2d(self._disp(src01[0, 0].cpu().float().numpy()))

        if not c["synth"]:                      # raw step → never show synth
            black = np.zeros_like(orig)
            return [
                ("original (raw step)",            orig,  "gray", None, None),
                ("parcellation (n/a — raw step)",  black, "gray", 0, 1),
                ("whole-image synth (n/a → orig)", orig,  "gray", None, None),
            ]

        # synth step → recompute the pipeline stages of THIS source (honest: it
        # is a genuine V26_6_2 synthesis of the exact sample that was fed).
        try:
            from src.synthesis.v26_6_2_synthesis import synthesize_debug_v2
            with torch.no_grad():
                dbg = synthesize_debug_v2(src01.to(self.device),
                                          c["gt"].to(self.device).long())
            parcel = dbg["parcellation"].reshape(1, 1, *dbg["parcellation"].shape)
            parcel2d = self._slice2d(parcel[0, 0].detach().cpu().float().numpy())
            whole2d = self._slice2d(np.clip(dbg["synth01_whole"][0, 0].detach().cpu().float().numpy(), 0, 1))
            return [
                ("original (pre-synth)",     orig,                            "gray",  None, None),
                ("K-means parcellation",     np.ma.masked_less(parcel2d, 0),  "tab10", 0, 7),
                ("whole-image synth (V26_6)", whole2d,                        "gray",  None, None),
            ]
        except Exception:
            return [("original (pre-synth)", orig, "gray", None, None)]
