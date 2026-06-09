"""
nnUNetTrainerV26_6_2 — V26_6_2 synthesis base trainer.  Dataset-agnostic.

Extends V26_6 by passing segmentation labels to synthesize_batch_fast_v2,
enabling per-label intensity remap on top of the whole-image K-means synthesis.
"""
from __future__ import annotations

import torch
from contextlib import nullcontext
from torch.amp import autocast

from src.nnunet.trainers.v26_6_base import nnUNetTrainerV26_6
from src.nnunet.transforms.synth_aug import random_crop_pair
from src.synthesis.v26_6_synthesis import gpu_spatial_augment
from src.synthesis.v26_6_2_synthesis import synthesize_batch_fast_v2


class nnUNetTrainerV26_6_2(nnUNetTrainerV26_6):
    """
    V26_6_2 base trainer.

    Identical to V26_6 except train_step passes labels from batch["target"]
    to synthesize_batch_fast_v2 for per-label intensity remap.
    """

    # ── train_step ────────────────────────────────────────────────────────────

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        _tgt = batch["target"]
        target = (_tgt[0] if isinstance(_tgt, (list, tuple)) else _tgt).to(self.device, non_blocking=True)

        if self._prefetch_future is not None:
            data_list, seg_list = self._prefetch_future.result()
        else:
            data_list, seg_list = None, None

        B = data.shape[0]

        flat  = data.reshape(B, -1)
        v_min = flat.min(dim=1).values.view(B, 1, 1, 1, 1)
        v_max = flat.max(dim=1).values.view(B, 1, 1, 1, 1)
        data_01 = ((data - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)

        data_01, target = gpu_spatial_augment(data_01, target)
        synth_z, _      = synthesize_batch_fast_v2(data_01, target)

        patch_list, seg_list_p = [], []
        for i in range(B):
            p, ps = random_crop_pair(synth_z[i:i+1], target[i:i+1], self._patch_size_cfg, n_crops=1)
            patch_list.append(p)
            seg_list_p.append(ps)
        n_patches = B
        patches = torch.cat(patch_list, dim=0)
        patches_cpu = patches.cpu().float()
        patches_seg_cpu = torch.cat(seg_list_p, dim=0).cpu().to(torch.int16)

        def _do_transforms(patchc, segc, n):
            outs = [self._train_transforms(image=patchc[b], segmentation=segc[b]) for b in range(n)]
            return [o["image"] for o in outs], [o["segmentation"] for o in outs]

        self._prefetch_future = self._transform_pool.submit(
            _do_transforms, patches_cpu, patches_seg_cpu, n_patches
        )

        if data_list is None:
            data_list, seg_list = self._prefetch_future.result()
            self._prefetch_future = None

        data_aug = torch.stack(data_list).to(self.device, non_blocking=True)
        if isinstance(seg_list[0], (list, tuple)):
            n_scales = len(seg_list[0])
            seg_aug = [
                torch.stack([seg_list[b][s] for b in range(n_patches)]).to(self.device, non_blocking=True)
                for s in range(n_scales)
            ]
        else:
            seg_aug = torch.stack(seg_list).to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data_aug)
            l = self.loss(output, seg_aug)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ── WandB visualization ───────────────────────────────────────────────────

    def _log_wandb_images(self, epoch: int = 0) -> None:
        """6-panel V26_6_2 image exposing the full synthesis pipeline:

        T1w | K-means parcellation | whole-image synth (V26_6) |
        label-wise synth (V26_6_2) | GT seg | Prediction.
        """
        try:
            import numpy as np
            import wandb
            if wandb.run is None:
                return
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from src.nnunet.transforms.synth_aug import center_crop_pair
            from src.synthesis.v26_6_2_synthesis import synthesize_debug_v2

            def to_np(t): return t.detach().cpu().float().numpy()

            def norm_brain(v):
                b = v[v != 0]
                if len(b) == 0: return np.zeros_like(v)
                lo, hi = np.percentile(b, 2), np.percentile(b, 98)
                out = np.zeros_like(v)
                m = v != 0
                out[m] = np.clip((v[m] - lo) / max(hi - lo, 1e-6), 0, 1)
                return out

            def crop(vol, seg):  # vol/seg: (1,1,D,H,W) → cropped (1,1,d,h,w)
                c, _ = center_crop_pair(vol, seg, self._patch_size_cfg)
                return c

            def make_panel(batch):
                # take a single sample, normalise to [0,1] over the whole volume
                data_gpu = batch["data"][0:1].to(self.device)
                seg_raw  = batch["target"]
                seg_gpu  = (seg_raw[0] if isinstance(seg_raw, (list, tuple)) else seg_raw)[0:1].to(self.device)
                v_min = data_gpu.min()
                v_max = data_gpu.max()
                img01 = ((data_gpu - v_min) / (v_max - v_min + 1e-7)).clamp(0, 1)

                with torch.no_grad():
                    dbg = synthesize_debug_v2(img01, seg_gpu)
                parcel = dbg["parcellation"].reshape(1, 1, *dbg["parcellation"].shape)  # (1,1,D,H,W)

                raw_c    = crop(data_gpu,            seg_gpu)
                parcel_c = crop(parcel,              seg_gpu)
                whole_c  = crop(dbg["synth01_whole"], seg_gpu)
                label_c  = crop(dbg["synth01_label"], seg_gpu)
                sz_c     = crop(dbg["synth_z"],       seg_gpu)
                _, seg_c = center_crop_pair(data_gpu, seg_gpu, self._patch_size_cfg)

                t_out = self._val_transforms(
                    image=sz_c.cpu().float()[0],
                    segmentation=seg_c.cpu().to(torch.int16)[0],
                )
                inp = t_out["image"].unsqueeze(0).to(self.device)
                self.network.eval()
                logits = self.network(inp)
                logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                pred = logits.argmax(1, keepdim=True)
                self.network.train()

                mid = raw_c.shape[-1] // 2
                return {
                    "raw":    norm_brain(to_np(raw_c[0, 0])[:, :, mid]),
                    "parcel": to_np(parcel_c[0, 0])[:, :, mid],
                    "whole":  np.clip(to_np(whole_c[0, 0])[:, :, mid], 0, 1),
                    "label":  np.clip(to_np(label_c[0, 0])[:, :, mid], 0, 1),
                    "gt":     to_np(seg_c[0, 0].float())[:, :, mid],
                    "pred":   to_np(pred[0, 0].float())[:, :, mid],
                }

            log_dict = {}
            for tag, loader in [("train", self.dataloader_train), ("val", self.dataloader_val)]:
                p = make_panel(next(loader))
                fig, axes = plt.subplots(1, 6, figsize=(24, 4))
                # parcellation: mask background (-1) so clusters stand out
                parcel_m = np.ma.masked_less(p["parcel"], 0)
                axes[0].imshow(p["raw"],   cmap="gray");                      axes[0].set_title("T1w")
                axes[1].imshow(parcel_m,   cmap="tab10", vmin=0, vmax=7);     axes[1].set_title("K-means parcellation")
                axes[2].imshow(p["whole"], cmap="gray");                      axes[2].set_title("whole-image synth (V26_6)")
                axes[3].imshow(p["label"], cmap="gray");                      axes[3].set_title("label-wise synth (V26_6_2)")
                axes[4].imshow(p["gt"],    cmap="tab10", vmin=0, vmax=6);     axes[4].set_title("GT seg")
                axes[5].imshow(p["pred"],  cmap="tab10", vmin=0, vmax=6);     axes[5].set_title("Prediction")
                for ax in axes: ax.axis("off")
                plt.suptitle(f"v26_6_2 {tag} fold{self.fold} ep{epoch}", fontsize=10)
                plt.tight_layout()
                log_dict[f"{tag}/panel"] = wandb.Image(fig)
                plt.close(fig)

            # Clamp to wandb's current step so a resumed run (whose internal step is
            # ahead of the resumed epoch) doesn't get a rejected out-of-order log.
            _safe_step = max(epoch, getattr(wandb.run, "step", epoch) or epoch)
            wandb.log(log_dict, step=_safe_step)
        except Exception as e:
            import traceback
            print(f"[WandB] V26_6_2 image log failed: {e}\n{traceback.format_exc()}")
