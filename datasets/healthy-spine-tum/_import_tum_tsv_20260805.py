#!/usr/bin/env python3
"""
One-off import: external TUM healthy-spine TSV results -> our standard
8_results_<ds>/02_metrics/<model>/<train_contrast>/<category>_<run_id>/fold{k}/eval_all.csv
layout (see CLAUDE.md "Standardized output layout").

Source layout (from tsv.zip):
  tsv/{SG_CT,SG_in-phase}/{Base,PALETTE-05+GE,PALETTE-05+Paper,Paper,
       Synthseg10+GE,Synthseg10NoEM+GE}/fold-{0,1,2}/<test-set>-<test-contrast>.tsv

Each TSV row = one subject, columns = "<label>-<metric>" for label in
{vertebra, ivd, spinal_canal}. We take <label>-global_bin_dsc as "dice"
(whole-structure binary Dice, closest analog to how dice is used elsewhere
in this project) and leave hd95 blank (no HD95 was computed in this data;
user said to ignore for now).
"""
import csv
import re
from pathlib import Path

SRC = Path("/tmp/tsv_inspect/tsv")
DST = Path(
    "/project/aip-jcohen/paulh/mri_synthesis_project/"
    "datasets/healthy-spine-tum/8_results_healthy-spine-tum/02_metrics/healthy_spine_tum_model"
)
TIMESTAMP = "20260805_130846"  # date +%Y%m%d_%H%M%S at import time

TRAIN_CONTRAST_MAP = {"SG_CT": "ct", "SG_in-phase": "inphase"}

# method dir -> (category, run_id method slug)
METHOD_MAP = {
    "Base": ("nnUNet", "baseline"),
    "Paper": ("auglab", "auglab_default"),
    "Synthseg10NoEM+GE": ("auglab", "synthseg_noEM"),
    "Synthseg10+GE": ("auglab", "synthseg_EM"),
    "PALETTE-05+Paper": ("auglab", "auglabAug_v26_6_2"),  # OURS
    "PALETTE-05+GE": ("auglab", "palette05_ge"),  # extra arm, kept its own name
}

LABELS = ["vertebra", "ivd", "spinal_canal"]

TEST_FILE_RE = re.compile(r"^(spider|spinegan)-test-(.+)\.tsv$")


def group_name(fname: str) -> str:
    m = TEST_FILE_RE.match(fname)
    assert m, fname
    test_set, contrast = m.groups()
    contrast = contrast.replace("dixon_part-", "dixon_")
    return f"{test_set}_{contrast}"


def main():
    n_files = 0
    n_rows = 0
    for train_dir, train_contrast in TRAIN_CONTRAST_MAP.items():
        for method_dir, (category, method_slug) in METHOD_MAP.items():
            run_id = f"{category}_healthy-spine-tum_{train_contrast}_{method_slug}_{TIMESTAMP}"
            src_method_dir = SRC / train_dir / method_dir
            if not src_method_dir.is_dir():
                raise SystemExit(f"missing source dir: {src_method_dir}")
            for fold_dir in sorted(src_method_dir.glob("fold-*")):
                fold_idx = fold_dir.name.split("-")[1]
                out_dir = DST / train_contrast / run_id / f"fold{fold_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_rows = []
                for tsv_path in sorted(fold_dir.glob("*.tsv")):
                    grp = group_name(tsv_path.name)
                    with open(tsv_path, newline="") as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for row in reader:
                            case = row["subject_name"]
                            for label in LABELS:
                                dice = row.get(f"{label}-global_bin_dsc", "")
                                out_rows.append((grp, case, label, dice, ""))
                    n_files += 1
                out_path = out_dir / "eval_all.csv"
                with open(out_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["group", "case", "label", "dice", "hd95"])
                    w.writerows(out_rows)
                n_rows += len(out_rows)
    print(f"converted {n_files} source tsv files -> {n_rows} eval_all.csv rows")


if __name__ == "__main__":
    main()
