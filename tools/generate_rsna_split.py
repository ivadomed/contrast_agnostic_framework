#!/usr/bin/env python3
"""Generate a deterministic RSNA lumbar spine split from filenames or train.csv."""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-tr",
        default="data/rsna-lumbar-spine-degenerative-classification/imagesTr",
        help="Directory containing source images for split generation.",
    )
    parser.add_argument(
        "--train-csv",
        default="/tmp/rsna-lumbar-spine-degenerative-classification/rsna-2024-lumbar-spine-degenerative-classification.zip",
        help="Optional RSNA train.csv path or Kaggle archive used when imagesTr is not populated.",
    )
    parser.add_argument(
        "--split-file",
        default="data/splits/rsna_spine_split.json",
        help="Output split JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction.")
    return parser.parse_args()


def _subject_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def main() -> None:
    args = parse_args()
    images_tr = Path(args.images_tr)
    split_file = Path(args.split_file)

    subjects = sorted({_subject_id(path) for path in images_tr.glob("*.nii.gz")})
    if not subjects:
        train_csv = Path(args.train_csv)
        if not train_csv.exists():
            raise SystemExit(f"No .nii.gz files found under {images_tr} and no train.csv or archive at {train_csv}")
        if train_csv.suffix.lower() == ".zip":
            with zipfile.ZipFile(str(train_csv), "r") as zf:
                with zf.open("train.csv") as handle:
                    reader = csv.DictReader(line.decode("utf-8") for line in handle)
                    subjects = sorted({row["study_id"] for row in reader if row.get("study_id")})
        else:
            with train_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                subjects = sorted({row["study_id"] for row in reader if row.get("study_id")})
        if not subjects:
            raise SystemExit(f"No study IDs found in {train_csv}")

    rng = random.Random(int(args.seed))
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * float(args.train_ratio)))
    n_train = max(1, min(n_train, n_total - 1))

    payload = [{
        "train": shuffled[:n_train],
        "val": shuffled[n_train:],
    }]

    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote split for {n_total} subjects to {split_file}")
    print(f"Train: {len(payload[0]['train'])}, Val: {len(payload[0]['val'])}")


if __name__ == "__main__":
    main()