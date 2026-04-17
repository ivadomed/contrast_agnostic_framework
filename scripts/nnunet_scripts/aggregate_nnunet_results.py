"""
aggregate_nnunet_results.py
──────────────────────────
Reads all seg_B evaluation summaries and produces the same three-file format
as evaluate.py (seg_A):

    eval_long.csv     — one row per (model, eval_contrast) pair
    eval_wide.csv     — one row per model, one column per contrast
    eval_summary.md   — markdown table

Summary files scanned (under eval_dir/seg_B_*/):
  eval_summary_{contrast}_fold{N}.json   — new format (contrast explicit)
  eval_summary_fold{N}.json              — legacy format (contrast inferred from run name)

Output paths:
  gen_raw models → results/eval/seg_B_baseline/multiclass/   (pass --family gen_raw)
  gen_19  models → results/eval/v19/seg_B/                   (pass --family gen_19)

Usage (from project root):
    python scripts/aggregate_nnunet_results.py --family gen_raw
    python scripts/aggregate_nnunet_results.py --family gen_19 --output-dir results/eval/v19/seg_B

Called automatically at the end of run_evaluation_nnunet.sh.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

FOREGROUND_LABELS = [1, 2, 3]   # BraTS: 1=NCR, 2=ED, 3=ET
CONTRASTS = ["flair", "t1w", "t1gd", "t2w"]   # canonical order matching seg_A

DATASET_NAME_MAP = {
    "t1w":  "Dataset022_BraTST1w_gen_raw",
    "t2w":  "Dataset023_BraTST2w_gen_raw",
    "flair":"Dataset021_BraTSFlair_gen_raw",
    "t1gd": "Dataset024_BraTST1gd_gen_raw",
}

TRAINER_MAP = {
    "gen_raw":  "nnUNetTrainerBraTSWandb",
    "gen_19":   "nnUNetTrainerBraTSGen19Wandb",
    "bookends": "nnUNetTrainer_Bookends",
}

CONFIG = "3d_fullres"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_run_name(run_name: str) -> tuple[str, str]:
    """Return (family, source_contrast) from a seg_B run name.

    Examples:
        seg_B_gen_raw_t1w  → ("gen_raw", "t1w")
        seg_B_gen_19_t2w   → ("gen_19",  "t2w")
    """
    suffix = run_name.removeprefix("seg_B_")
    for contrast in CONTRASTS:
        if suffix.endswith(f"_{contrast}"):
            family = suffix[: -(len(contrast) + 1)]
            return family, contrast
    raise ValueError(f"Cannot parse run name: {run_name!r}")


def _parse_summary(summary_path: Path) -> dict:
    """Extract per-class and mean foreground Dice from an nnUNet summary.json."""
    with summary_path.open() as f:
        data = json.load(f)

    per_label: dict[int, list[float]] = defaultdict(list)
    for entry in data.get("metric_per_case", []):
        for label_str, metrics in entry.get("metrics", {}).items():
            label = int(label_str)
            if label in FOREGROUND_LABELS:
                v = metrics.get("Dice")
                if v is not None:
                    per_label[label].append(float(v))

    result: dict = {}
    fg_means = []
    for label in FOREGROUND_LABELS:
        vals = per_label.get(label, [])
        mean = sum(vals) / len(vals) if vals else None
        result[f"c{label}"] = mean
        if mean is not None:
            fg_means.append(mean)

    result["mean_fg_dice"] = sum(fg_means) / len(fg_means) if fg_means else None
    return result


def _ckpt_path(nnunet_results: Path, family: str, source_contrast: str, fold: int) -> str:
    ds_name = DATASET_NAME_MAP[source_contrast]
    trainer  = TRAINER_MAP.get(family, "nnUNetTrainerBraTSWandb")
    return str(
        nnunet_results / ds_name / f"{trainer}__nnUNetPlans__{CONFIG}" / f"fold_{fold}" / "checkpoint_best.pth"
    )


def _format_metric(value) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# ── CSV / Markdown writers (identical contract to evaluate.py) ────────────────

def _write_long_csv(path: Path, rows: list[dict]) -> None:
    fields = ["model_id", "family", "source_contrast", "target_contrast",
              "dice", "ckpt_exists", "checkpoint_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_wide_csv(path: Path, rows: list[dict]) -> None:
    per_class_cols = [f"{c}_c{i}" for c in CONTRASTS for i in range(1, 4)]
    fields = [
        "model_id", "family", "source_contrast", "checkpoint_path", "ckpt_exists",
        *CONTRASTS,
        *per_class_cols,
        "in_domain_dice", "ood_mean_dice", "ood_worst_dice",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fields}
            writer.writerow(out)


def _write_summary_markdown(path: Path, rows: list[dict]) -> None:
    per_class_cols = [f"{c}_c{i}" for c in CONTRASTS for i in range(1, 4)]
    all_cols = list(CONTRASTS) + per_class_cols

    col_header = " | ".join(all_cols)
    col_rule   = " | ".join(["---:"] * len(all_cols))
    header = [
        f"| model_id | family | source_contrast | ckpt_exists | {col_header} | in_domain_dice | ood_mean_dice | ood_worst_dice |",
        f"|---|---|---|---:| {col_rule} |---:|---:|---:|",
    ]
    body = []
    for row in rows:
        cells = [
            str(row["model_id"]),
            str(row["family"]),
            str(row["source_contrast"]),
            str(row["ckpt_exists"]),
            *[_format_metric(row.get(c, "")) for c in all_cols],
            _format_metric(row.get("in_domain_dice", "")),
            _format_metric(row.get("ood_mean_dice", "")),
            _format_metric(row.get("ood_worst_dice", "")),
        ]
        body.append("| " + " | ".join(cells) + " |")

    content = ["# Evaluation Summary", ""] + header + body + [""]
    path.write_text("\n".join(content), encoding="utf-8")


# ── Summary scanning ──────────────────────────────────────────────────────────

# Matches:  eval_summary_t1w_fold0.json   → groups: contrast="t1w", fold="0"
_NEW_RE = re.compile(r"eval_summary_([a-z0-9]+)_fold(\d+)\.json$")
# Matches:  eval_summary_fold0.json       → groups: fold="0"
_OLD_RE = re.compile(r"eval_summary_fold(\d+)\.json$")


def _scan_summaries(eval_dir: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """Scan eval_dir for seg_B summary JSONs.

    Returns:
        { run_name: { eval_contrast: { fold: metrics_dict } } }
    """
    runs: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))

    for run_dir in sorted(eval_dir.glob("seg_B_*")):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        try:
            _, source_contrast = _parse_run_name(run_name)
        except ValueError:
            continue

        for path in sorted(run_dir.glob("eval_summary_*.json")):
            # Try new format first (explicit contrast)
            m = _NEW_RE.match(path.name)
            if m:
                eval_contrast = m.group(1)
                fold = int(m.group(2))
                runs[run_name][eval_contrast][fold] = _parse_summary(path)
                continue
            # Fall back to legacy format (contrast == source_contrast)
            m = _OLD_RE.match(path.name)
            if m:
                fold = int(m.group(1))
                # Only add if not already covered by new-format file
                if fold not in runs[run_name][source_contrast]:
                    runs[run_name][source_contrast][fold] = _parse_summary(path)

    return runs


def _avg_folds(fold_map: dict[int, dict], key: str) -> float | None:
    vals = [fold_map[f][key] for f in fold_map if fold_map[f].get(key) is not None]
    return sum(vals) / len(vals) if vals else None


# ── Row builders ──────────────────────────────────────────────────────────────

def build_rows(
    eval_dir: Path,
    nnunet_results: Path,
    family_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Scan eval_dir and return (long_rows, wide_rows).

    family_filter: if set, only include runs whose family matches (gen_raw / gen_19).
    """
    runs = _scan_summaries(eval_dir)

    long_rows: list[dict] = []
    wide_rows: list[dict] = []

    for run_name, contrast_map in sorted(runs.items()):
        try:
            family, source_contrast = _parse_run_name(run_name)
        except ValueError:
            continue

        if family_filter and family != family_filter:
            continue

        fold_for_ckpt = min(next(iter(contrast_map.values())).keys())
        ckpt = _ckpt_path(nnunet_results, family, source_contrast, fold_for_ckpt)
        ckpt_exists = 1 if Path(ckpt).exists() else 0

        # ── Long rows (one per evaluated contrast) ────────────────────────
        for eval_contrast, fold_map in sorted(contrast_map.items()):
            mean_fg = _avg_folds(fold_map, "mean_fg_dice")
            long_rows.append({
                "model_id":        run_name,
                "family":          family,
                "source_contrast": source_contrast,
                "target_contrast": eval_contrast,
                "dice":            mean_fg if mean_fg is not None else "",
                "ckpt_exists":     ckpt_exists,
                "checkpoint_path": ckpt,
            })

        # ── Wide row (all contrasts in one row) ───────────────────────────
        wide: dict = {
            "model_id":        run_name,
            "family":          family,
            "source_contrast": source_contrast,
            "checkpoint_path": ckpt,
            "ckpt_exists":     ckpt_exists,
        }

        in_domain_dice = None
        ood_vals: list[float] = []

        for eval_contrast, fold_map in contrast_map.items():
            mean_fg = _avg_folds(fold_map, "mean_fg_dice")
            wide[eval_contrast] = mean_fg if mean_fg is not None else ""

            for ci in range(1, 4):
                val = _avg_folds(fold_map, f"c{ci}")
                wide[f"{eval_contrast}_c{ci}"] = val if val is not None else ""

            if mean_fg is not None:
                if eval_contrast == source_contrast:
                    in_domain_dice = mean_fg
                else:
                    ood_vals.append(mean_fg)

        wide["in_domain_dice"]  = in_domain_dice if in_domain_dice is not None else ""
        wide["ood_mean_dice"]   = sum(ood_vals) / len(ood_vals) if ood_vals else ""
        wide["ood_worst_dice"]  = min(ood_vals) if ood_vals else ""

        wide_rows.append(wide)

    return long_rows, wide_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default="results/eval",
        help="Directory containing seg_B_* subdirs (default: results/eval)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/eval/seg_B_baseline/multiclass",
        help="Where to write the three output files",
    )
    parser.add_argument(
        "--nnunet-results",
        default="results/nnUNet",
        help="nnUNet_results root for checkpoint path inference",
    )
    parser.add_argument(
        "--family",
        choices=["gen_raw", "gen_19", "bookends"],
        default=None,
        help="Filter to only include this model family (default: include all)",
    )
    args = parser.parse_args()

    eval_dir       = Path(args.eval_dir)
    output_dir     = Path(args.output_dir)
    nnunet_results = Path(args.nnunet_results)

    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows, wide_rows = build_rows(eval_dir, nnunet_results, family_filter=args.family)

    if not long_rows:
        family_note = f" (family={args.family})" if args.family else ""
        print(f"[aggregate] No seg_B evaluation summaries found{family_note} — nothing to write.")
        return

    long_csv   = output_dir / "eval_long.csv"
    wide_csv   = output_dir / "eval_wide.csv"
    summary_md = output_dir / "eval_summary.md"

    _write_long_csv(long_csv, long_rows)
    _write_wide_csv(wide_csv, wide_rows)
    _write_summary_markdown(summary_md, wide_rows)

    print(f"[aggregate] {len(long_rows)} entries from {len(wide_rows)} model(s)")
    print(f"[aggregate] Saved long CSV : {long_csv}")
    print(f"[aggregate] Saved wide CSV : {wide_csv}")
    print(f"[aggregate] Saved markdown : {summary_md}")


if __name__ == "__main__":
    main()
