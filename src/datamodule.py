from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from monai.apps import DecathlonDataset
from monai.data import DataLoader, CacheDataset
from omegaconf import DictConfig
from torch.utils.data import Subset

from src.data_registry import get_dataset_spec
from src.dataset import get_preprocessing_transforms, normalize_contrast_name


class BraTSDataModule(pl.LightningDataModule):
    """LightningDataModule for BraTS Task01 training/validation.

    This module owns split creation/loading and dataset instantiation for both
    synthesis and segmentation tasks.

    Args:
        cfg: Hydra configuration.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.project_root = Path(__file__).resolve().parents[1]

        self.split: dict[str, Any] | None = None
        self.train_dataset = None
        self.val_dataset = None

    @staticmethod
    def _subject_id_from_sample(sample: dict[str, Any]) -> str:
        image_path = sample["image"]
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]
        image_name = Path(str(image_path)).name
        if image_name.endswith(".nii.gz"):
            return image_name[:-7]
        return Path(image_name).stem

    def _resolve_path(self, path_like: str) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _load_or_create_split(
        self,
        samples: list[dict[str, Any]],
        split_file: Path,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> dict[str, Any]:
        subjects = [self._subject_id_from_sample(sample) for sample in samples]
        unique_subjects = sorted(set(subjects))
        wrote_file = False

        split_file.parent.mkdir(parents=True, exist_ok=True)
        if split_file.exists():
            with split_file.open("r", encoding="utf-8") as handle:
                split = json.load(handle)
        else:
            shuffled = unique_subjects[:]
            rng = random.Random(seed)
            rng.shuffle(shuffled)

            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))

            split = {
                "seed": seed,
                "train_subjects": shuffled[:n_train],
                "val_subjects": shuffled[n_train : n_train + n_val],
                "test_subjects": shuffled[n_train + n_val :],
            }
            with split_file.open("w", encoding="utf-8") as handle:
                json.dump(split, handle, indent=2)
            wrote_file = True

        if isinstance(split, list):
            if not split:
                split = [{"train": [], "val": [], "test": []}]
            split_entry = dict(split[0] or {})
            train_set = set(split_entry.get("train", []))
            val_set = set(split_entry.get("val", []))
            test_set = set(split_entry.get("test", []))
            split_payload: dict[str, Any] = split_entry
        else:
            train_set = set(split.get("train_subjects", []))
            val_set = set(split.get("val_subjects", []))
            test_set = set(split.get("test_subjects", []))
            split_payload = split

        train_indices, val_indices, test_indices = [], [], []
        for idx, subject in enumerate(subjects):
            if subject in train_set:
                train_indices.append(idx)
            elif subject in val_set:
                val_indices.append(idx)
            elif subject in test_set:
                test_indices.append(idx)

        split_payload["train_indices"] = train_indices
        split_payload["val_indices"] = val_indices
        split_payload["test_indices"] = test_indices
        split_payload["dataset_size"] = len(samples)

        if isinstance(split, list):
            split[0] = split_payload
        if isinstance(split, list) or wrote_file:
            with split_file.open("w", encoding="utf-8") as handle:
                json.dump(split, handle, indent=2)

        return split_payload

    def _build_local_task_samples(self, task_dir: Path) -> list[dict[str, str]]:
        images_tr = task_dir / "imagesTr"
        labels_tr = task_dir / "labelsTr"
        if not images_tr.exists() or not labels_tr.exists():
            raise ValueError(f"Expected imagesTr and labelsTr under {task_dir}")

        samples: list[dict[str, str]] = []
        for image_path in sorted(images_tr.glob("*.nii.gz")):
            label_path = labels_tr / image_path.name
            if not label_path.exists():
                continue
            # Extract subject ID from filename (e.g., "100_0000.nii.gz" -> "100")
            subject_id = image_path.stem.split("_")[0]
            samples.append({"image": str(image_path), "label": str(label_path), "subject": subject_id})

        if not samples:
            raise ValueError(f"No paired training samples found under {task_dir}")
        return samples

    def setup(self, stage: str | None = None) -> None:
        """Build train/val subsets and their MONAI datasets.

        Args:
            stage: Lightning setup stage.
        """
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        dataset_name = str(getattr(self.cfg.data, "name", "brats"))
        dataset_spec = get_dataset_spec(dataset_name)

        available_contrasts_cfg = [str(c) for c in getattr(self.cfg.data, "contrasts", [])]
        available_contrasts = available_contrasts_cfg or [str(c) for c in dataset_spec["contrasts"]]
        source_contrast = normalize_contrast_name(self.cfg.data.source_contrast, available_contrasts)
        label_mapping_cfg = getattr(self.cfg.data, "label_mapping", None)
        if label_mapping_cfg is not None:
            label_mapping = dict(label_mapping_cfg)
        else:
            label_mapping = dict(dataset_spec.get("label_mapping") or {})
        patch_size = tuple(self.cfg.data.patch_size)
        data_dir = self._resolve_path(self.cfg.data.data_dir)
        split_file = self._resolve_path(self.cfg.data.split_file)
        task_name = str(getattr(self.cfg.data, "task_name", dataset_spec.get("task_name", "Task01_BrainTumour")))
        train_mode = "train"
        if str(self.cfg.task) == "segmenter" and str(self.cfg.version) == "v16_bigaug":
            train_mode = "train_bigaug"
        if str(self.cfg.version) == "v17_lpci":
            train_mode = "train_lpci"

        task_dir = data_dir / task_name
        if (task_dir / "imagesTr").exists() and (task_dir / "labelsTr").exists():
            full_samples = self._build_local_task_samples(task_dir)
        else:
            train_dataset_full = DecathlonDataset(
                root_dir=str(data_dir),
                task=task_name,
                transform=get_preprocessing_transforms(
                    mode=train_mode,
                    patch_size=patch_size,
                    source_contrast=source_contrast,
                    contrasts=available_contrasts,
                    label_mapping=label_mapping,
                ),
                section="training",
                download=True,
                cache_rate=0.0,
                num_workers=int(self.cfg.data.num_workers),
            )
            full_samples = train_dataset_full.data

        self.split = self._load_or_create_split(
            samples=full_samples,
            split_file=split_file,
            train_ratio=float(self.cfg.data.train_ratio),
            val_ratio=float(self.cfg.data.val_ratio),
            seed=int(self.cfg.seed),
        )

        train_indices = self.split.get("train_indices", [])
        val_indices = self.split.get("val_indices", [])
        if not train_indices:
            raise ValueError(f"No train indices found in split file: {split_file}")

        train_samples = [full_samples[i] for i in train_indices]
        self.train_dataset = CacheDataset(
            data=train_samples,
            transform=get_preprocessing_transforms(
                mode=train_mode,
                patch_size=patch_size,
                source_contrast=source_contrast,
                contrasts=available_contrasts,
                label_mapping=label_mapping,
            ),
            cache_rate=float(self.cfg.data.cache_rate),
            num_workers=int(self.cfg.data.num_workers),
        )

        # Generator training has no validation loop; avoid building val dataset to cut startup/caching overhead.
        if str(self.cfg.task) == "generator":
            self.val_dataset = None
            return

        if not val_indices:
            raise ValueError(f"No validation indices found in split file: {split_file}")

        val_samples = [full_samples[i] for i in val_indices]
        self.val_dataset = CacheDataset(
            data=val_samples,
            transform=get_preprocessing_transforms(
                mode="val",
                patch_size=patch_size,
                source_contrast=source_contrast,
                contrasts=available_contrasts,
                label_mapping=label_mapping,
            ),
            cache_rate=float(self.cfg.data.cache_rate),
            num_workers=int(self.cfg.data.num_workers),
        )

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader.

        Returns:
            DataLoader: Training dataloader with drop_last=True for stable graph shapes.
        """
        explicit_batch_size = getattr(self.cfg.data, "batch_size", None)
        if explicit_batch_size is not None:
            train_batch_size = int(explicit_batch_size)
        elif str(self.cfg.task) == "segmenter":
            train_batch_size = int(self.cfg.data.batch_size_segmenter)
        else:
            train_batch_size = int(self.cfg.data.batch_size_generator)

        loader_kwargs: dict[str, Any] = {
            "batch_size": train_batch_size,
            "shuffle": True,
            "num_workers": int(self.cfg.data.num_workers),
            "pin_memory": bool(self.cfg.data.pin_memory),
            "drop_last": bool(self.cfg.data.drop_last_train),
        }
        if int(self.cfg.data.num_workers) > 0 and bool(self.cfg.data.persistent_workers):
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(self.cfg.data.prefetch_factor)
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader.

        Returns:
            DataLoader: Validation dataloader.
        """
        if self.val_dataset is None:
            return None

        loader_kwargs: dict[str, Any] = {
            "batch_size": int(self.cfg.data.val_batch_size),
            "shuffle": False,
            "num_workers": int(self.cfg.data.num_workers),
            "pin_memory": bool(self.cfg.data.pin_memory),
        }
        if int(self.cfg.data.num_workers) > 0 and bool(self.cfg.data.persistent_workers):
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(self.cfg.data.prefetch_factor)
        return DataLoader(self.val_dataset, **loader_kwargs)
