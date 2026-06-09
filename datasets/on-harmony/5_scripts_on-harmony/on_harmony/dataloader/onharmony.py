"""
Custom nnUNet data loaders for the ON-Harmony benchmark.

OnHarmonyVolumeLoader
---------------------
Preloads all cases into RAM, serves raw full-volume batches via a prefetch
thread.  Used for validation (raw T1w) and image logging.

OnHarmonyBatchPool
------------------
Background thread pool that pre-prepares TRAINING batches:
  worker: sample case → full-vol min-max norm → random crop → CPU synthesis
          (V26_6) → SpatialTransform → pin_memory → queue
Main train_step just does: dequeue → .to(device) → forward/backward.

With 6 workers and ~0.7 s per batch, throughput ≈ 8 batches/s which keeps the
GPU (0.25 s/step) fully saturated.  Epoch time drops from ~220 s to ~40 s.

Key design choices
------------------
- Full-volume min-max normalisation is done BEFORE cropping to preserve the
  correct intensity scale for the generator's internal mask (images > 0.01).
- Synthesis (V26_6) runs on CPU: all its internals use torch.rand which is
  thread-safe.
- The RAM cache numpy arrays are read-only after init → thread-safe.
"""
from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch

from src.nnunet.transforms.synth_aug import random_crop_pair
from src.synthesis.v26_6_synthesis import compute_kmeans_centroids, synthesize_patch_fast


class OnHarmonyVolumeLoader:
    """
    Full-volume data loader backed by a RAM cache.

    Returns raw (unsynthesised) volumes.  Used for validation and image logging.
    """

    def __init__(
        self,
        dataset,
        case_ids: List[str],
        prefetch_queue_size: int = 2,
        rng_seed: int = 12345,
    ) -> None:
        self.dataset = dataset
        self.case_ids = list(case_ids)

        print(f"[OnHarmonyVolumeLoader] Preloading {len(case_ids)} cases into RAM …")
        t0 = perf_counter()
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        for cid in case_ids:
            data, seg, _seg_prev, _props = dataset.load_case(cid)
            self._cache[cid] = {
                "data": np.array(data, dtype=np.float32),
                "seg":  np.array(seg,  dtype=np.int16),
            }

        elapsed = perf_counter() - t0
        total_mb = sum(v["data"].nbytes + v["seg"].nbytes for v in self._cache.values()) / 1e6
        print(
            f"[OnHarmonyVolumeLoader] Loaded {total_mb:.0f} MB "
            f"({len(case_ids)} cases) in {elapsed:.1f}s"
        )

        self._queue = queue.Queue(maxsize=prefetch_queue_size)
        self._stop_event = threading.Event()
        self._rng_seed = rng_seed
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True, name="OnHarmonyPrefetch"
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        rng = np.random.default_rng(self._rng_seed)
        while not self._stop_event.is_set():
            cid = rng.choice(self.case_ids)
            cached = self._cache[cid]

            data_pt = torch.from_numpy(cached["data"]).unsqueeze(0)
            seg_pt = torch.from_numpy(cached["seg"]).unsqueeze(0)

            if torch.cuda.is_available():
                data_pt = data_pt.pin_memory()
                seg_pt = seg_pt.pin_memory()

            self._queue.put(
                {"data": data_pt, "target": seg_pt, "keys": [cid]},
                block=True,
            )

    def generate_train_batch(self) -> Dict:
        return self._queue.get(block=True)

    def __next__(self) -> Dict:
        return self.generate_train_batch()

    def __iter__(self):
        return self

    def stop(self) -> None:
        self._stop_event.set()
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break
        self._prefetch_thread.join(timeout=5.0)


class OnHarmonyBatchPool:
    """
    Background thread pool that pre-prepares synthesised training batches.

    Each worker runs independently:
      1. Sample B cases from the RAM cache
      2. Full-volume min-max normalise each (preserves CSF > 0.01)
      3. Random crop to initial_patch_size
      4. CPU V26_6 synthesis (steps 2-5)
      5. Apply SpatialTransform + DS downsampling
      6. Stack into batch and pin_memory
      7. Put into output queue

    train_step only does: dequeue → .to(device) → forward/backward.

    Parameters
    ----------
    cache              : {cid: {"data": np.ndarray, "seg": np.ndarray}}
    case_ids           : list of case identifiers
    generator          : V26_6SignedAlphaTargetGenerator (stateless, shared)
    hist_module_cpu    : DifferentiableHistogram3D on CPU (shared, no mutable state)
    transforms         : ComposeTransforms (stateless, shared)
    initial_patch_size : (D, H, W) crop dimensions
    n_patches_per_batch: batch size (default 2)
    n_workers          : number of background threads
    queue_size         : max batches pre-computed ahead
    rng_seed           : base seed (each worker gets a different offset)
    """

    def __init__(
        self,
        cache: Dict[str, Dict[str, np.ndarray]],
        case_ids: List[str],
        generator,
        hist_module_cpu,
        transforms,
        initial_patch_size: tuple,
        n_patches_per_batch: int = 2,
        n_workers: int = 8,
        queue_size: int = 16,
        rng_seed: int = 12345,
    ) -> None:
        self._case_ids = list(case_ids)
        self._transforms = transforms
        self._patch_size = initial_patch_size
        self._n_patches = n_patches_per_batch

        self._cache = {}
        print(f"[OnHarmonyBatchPool] Pre-computing K-means centroids for {len(case_ids)} cases …")
        for cid in case_ids:
            data_np = cache[cid]["data"]
            seg_np = cache[cid]["seg"]
            v_min = float(data_np.min())
            v_max = float(data_np.max())
            image_01 = np.clip((data_np - v_min) / (v_max - v_min + 1e-7), 0.0, 1.0).astype(np.float32)
            image_01_t = torch.from_numpy(image_01).unsqueeze(0)
            centroids = compute_kmeans_centroids(image_01_t)
            self._cache[cid] = {
                "image_01":  image_01_t,
                "seg":       torch.from_numpy(seg_np),
                "data":      cache[cid]["data"],
                "centroids": centroids,
            }

        self._queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._n_workers = n_workers

        # Workers are created but NOT started until first __next__ call.
        # Prevents synthesis threads from consuming CPU during torch.compile.
        self._workers: List[threading.Thread] = [
            threading.Thread(
                target=self._worker_loop,
                args=(rng_seed + i * 7919,),
                daemon=True,
                name=f"SynthWorker-{i}",
            )
            for i in range(n_workers)
        ]
        self._workers_started = False
        print(f"[OnHarmonyBatchPool] {n_workers} synthesis workers ready (lazy start)")

    def _worker_loop(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        while not self._stop_event.is_set():
            try:
                batch = self._prepare_batch(rng)
                self._queue.put(batch, block=True)
            except Exception as e:
                import traceback
                print(f"[SynthWorker] Error: {e}\n{traceback.format_exc()}")

    def _prepare_batch(self, rng: np.random.Generator) -> dict:
        """Build one training batch: B synthesised + augmented patches.

        Foreground oversampling (mirrors nnUNet standard DataLoader):
          For each patch, with P_FG probability, the crop is centred on a
          randomly selected foreground voxel to ensure rare tissue classes
          (CSF, deep GM, cerebellum, brainstem) appear frequently enough.
        """
        P_FG = 0.67
        cids = rng.choice(self._case_ids, size=self._n_patches, replace=True)

        data_list: list = []
        seg_list: list = []

        for cid in cids:
            cached = self._cache[str(cid)]
            image_01 = cached["image_01"]
            seg_pt = cached["seg"].unsqueeze(0)
            centroids = cached["centroids"]

            _, C, D, H, W = image_01.shape
            pd, ph, pw = self._patch_size

            if rng.random() < P_FG:
                fg = torch.nonzero(seg_pt[0, 0] > 0)
                if fg.shape[0] > 0:
                    idx = int(rng.integers(0, fg.shape[0]))
                    cd, ch, cw = fg[idx].tolist()
                    d0 = int(np.clip(cd - pd // 2, 0, max(0, D - pd)))
                    h0 = int(np.clip(ch - ph // 2, 0, max(0, H - ph)))
                    w0 = int(np.clip(cw - pw // 2, 0, max(0, W - pw)))
                    d1 = min(d0 + pd, D); d0 = d1 - pd
                    h1 = min(h0 + ph, H); h0 = h1 - ph
                    w1 = min(w0 + pw, W); w0 = w1 - pw
                    patch_01 = image_01[0, :, d0:d1, h0:h1, w0:w1].unsqueeze(0)
                    patch_seg = seg_pt[0, :, d0:d1, h0:h1, w0:w1].unsqueeze(0)
                else:
                    patch_01, patch_seg = random_crop_pair(image_01, seg_pt, self._patch_size, n_crops=1)
            else:
                patch_01, patch_seg = random_crop_pair(image_01, seg_pt, self._patch_size, n_crops=1)

            synth_z, _ = synthesize_patch_fast(patch_01, patch_seg, centroids)

            t = self._transforms(
                image=synth_z.float()[0],
                segmentation=patch_seg.to(torch.int16)[0],
            )
            data_list.append(t["image"])
            seg_list.append(t["segmentation"])

        data_batch = torch.stack(data_list)

        if isinstance(seg_list[0], (list, tuple)):
            n_scales = len(seg_list[0])
            seg_batch = [
                torch.stack([seg_list[b][s] for b in range(len(seg_list))])
                for s in range(n_scales)
            ]
            return {
                "data":   data_batch.pin_memory() if torch.cuda.is_available() else data_batch,
                "target": [s.pin_memory() if torch.cuda.is_available() else s for s in seg_batch],
            }

        seg_batch_t = torch.stack(seg_list)
        return {
            "data":   data_batch.pin_memory() if torch.cuda.is_available() else data_batch,
            "target": seg_batch_t.pin_memory() if torch.cuda.is_available() else seg_batch_t,
        }

    def update_synth_cache(self, synth_cache: dict) -> None:
        self._synth_cache = synth_cache

    def __next__(self) -> dict:
        if not self._workers_started:
            self._workers_started = True
            for t in self._workers:
                t.start()
            print(f"[OnHarmonyBatchPool] {self._n_workers} workers started (lazy)")
        return self._queue.get(block=True)

    def __iter__(self):
        return self

    @property
    def cache(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._cache

    @property
    def case_ids(self) -> List[str]:
        return self._case_ids

    def stop(self) -> None:
        self._stop_event.set()
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break
        for t in self._workers:
            t.join(timeout=5.0)


class OnHarmonySingleThreadedWrapper:
    """Thin wrapper so nnUNet can call next() on an OnHarmonyVolumeLoader."""

    def __init__(self, loader: OnHarmonyVolumeLoader) -> None:
        self._loader = loader

    def __next__(self) -> Dict:
        return self._loader.generate_train_batch()

    def __iter__(self):
        return self

    def stop(self) -> None:
        self._loader.stop()
