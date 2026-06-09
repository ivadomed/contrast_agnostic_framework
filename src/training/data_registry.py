from __future__ import annotations

from copy import deepcopy
from typing import Any


DatasetCatalog: dict[str, dict[str, Any]] = {
    "brats2017": {
        "task_name": "Task01_BrainTumour",
        "num_classes": 4,
        "contrasts": ["flair", "t1w", "t1gd", "t2w"],
        # BraTS-specific enhancing tumor remap.
        "label_mapping": {4: 3},
        # nnU-Net dataset IDs can vary by source contrast for BraTS exports.
        "nnunet_id": {
            "flair": "021",
            "t1w": "022",
            "t2w": "023",
            "t1gd": "024",
        },
    },
    "spider_spine": {
        "task_name": "Task102_SpiderSpine",
        "contrasts": ["t1_sag", "t2_sag", "t2_space"],
        "label_mapping": None,
        "nnunet_id": 102,
    },
    "ms_multi_spine": {
        "task_name": "Task104_MSMultiSpine",
        "contrasts": ["t2w", "mp2rage", "stir", "psir"],
        "label_mapping": None,
        "nnunet_id": 104,
    },
    "atlas": {
        "task_name": "Task01_BrainTumour",
        "num_classes": 2,
        "contrasts": ["t1w"],
        "label_mapping": {},
        "nnunet_id": {"t1w": "031"},
    },
    "on_harmony": {
        "task_name": "ON-Harmony",
        "num_classes": 1,  # background only; label-free training
        "contrasts": ["t1w"],
        "label_mapping": {},
        "nnunet_id": {},
    },
}


def get_dataset_spec(name: str) -> dict[str, Any]:
    key = str(name).strip().lower()
    if key not in DatasetCatalog:
        valid = ", ".join(sorted(DatasetCatalog.keys()))
        raise ValueError(f"Unsupported dataset '{name}'. Expected one of: {valid}")
    return deepcopy(DatasetCatalog[key])


def get_dataset_nnunet_id(name: str, contrast: str) -> str:
    spec = get_dataset_spec(name)
    mapping = spec.get("nnunet_id", {})
    if not isinstance(mapping, dict):
        return str(mapping)
    key = str(contrast).strip().lower()
    if key not in mapping:
        valid = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"No nnUNet dataset id configured for dataset='{name}', contrast='{contrast}'. "
            f"Expected one of: {valid}"
        )
    return str(mapping[key])
