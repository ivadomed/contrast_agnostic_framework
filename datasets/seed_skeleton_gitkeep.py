#!/usr/bin/env python3
"""Seed the standard 9-slot dataset skeleton with `.gitkeep` placeholders.

Git does not track directories, only files, and most data trees here are hidden
by `.gitignore` (`**/1_BIDS*/*`, `**/2_nnUNet*/*`, `**/8_results*/*`, `*.nii.gz`
...). To publish the *folder structure* on GitHub even where a folder holds only
(ignored) data, we drop an empty `.gitkeep` in each standard folder and then
force-add them (`git add -f`), which bypasses the ignore rules for just that one
placeholder without exposing any data files.

Skeleton matches `create_dataset_structure.py` exactly (the source of truth also
used by `validate_standard_dataset_structure.py`). Run from anywhere:

    python datasets/seed_skeleton_gitkeep.py            # backfill all datasets
    python datasets/seed_skeleton_gitkeep.py chaos ...  # only named datasets

Idempotent: existing dirs/files are left untouched.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --- standard skeleton (keep in sync with create_dataset_structure.py) -------
LEVEL_1 = ["0_raw", "1_BIDS", "2_nnUNet", "3_conf", "4_splits",
           "5_scripts", "6_checkpoints", "7_analysis", "8_results", "9_tests"]
NNUNET_SUBDIRS = ["raw", "preprocessed"]
SCRIPTS_SUBDIRS = ["00_utils", "01_create_splits", "02_nnunet", "03_preprocess",
                   "04_train", "05_predict", "06_evaluate"]
RESULTS_SUBDIRS = ["01_predictions", "02_metrics"]

EXCLUDE = {"00_commun_scripts", "__pycache__"}


def skeleton_dirs(dataset: str):
    """Yield every standard skeleton dir (relative to the dataset dir)."""
    for slot in LEVEL_1:
        yield f"{slot}_{dataset}"
    for sub in NNUNET_SUBDIRS:
        yield os.path.join(f"2_nnUNet_{dataset}", sub)
    for sub in SCRIPTS_SUBDIRS:
        yield os.path.join(f"5_scripts_{dataset}", sub)
    for sub in RESULTS_SUBDIRS:
        yield os.path.join(f"8_results_{dataset}", sub)


def is_dataset(name: str) -> bool:
    path = os.path.join(HERE, name)
    if not os.path.isdir(path) or name in EXCLUDE:
        return False
    # a dataset has at least one standard slot dir already (skips empty
    # placeholder dirs like brats2017/ms-multi-spine/spider-spine); note some
    # datasets legitimately start at 1_BIDS with no 0_raw slot.
    return any(os.path.isdir(os.path.join(path, f"{slot}_{name}"))
               for slot in LEVEL_1)


def seed(dataset: str):
    created_dirs, created_keeps = [], []
    base = os.path.join(HERE, dataset)
    for rel in skeleton_dirs(dataset):
        d = os.path.join(base, rel)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            created_dirs.append(os.path.relpath(d, HERE))
        keep = os.path.join(d, ".gitkeep")
        if not os.path.exists(keep):
            open(keep, "a").close()
            created_keeps.append(os.path.relpath(keep, HERE))
    return created_dirs, created_keeps


def main(argv):
    requested = argv[1:]
    if requested:
        datasets = requested
    else:
        datasets = sorted(n for n in os.listdir(HERE) if is_dataset(n))
    print(f"Seeding skeleton .gitkeep for {len(datasets)} dataset(s): "
          f"{', '.join(datasets)}\n")
    total_dirs = total_keeps = 0
    for ds in datasets:
        if not os.path.isdir(os.path.join(HERE, ds)):
            print(f"  ! skip {ds}: no such directory")
            continue
        dirs, keeps = seed(ds)
        total_dirs += len(dirs)
        total_keeps += len(keeps)
        print(f"  {ds}: +{len(dirs)} dir(s), +{len(keeps)} .gitkeep")
        for d in dirs:
            print(f"      mkdir {d}")
    print(f"\nDone. Created {total_dirs} dir(s), {total_keeps} new .gitkeep.")
    print("Next: git add -f datasets/**/.gitkeep  (bypasses ignore rules)")


if __name__ == "__main__":
    main(sys.argv)
