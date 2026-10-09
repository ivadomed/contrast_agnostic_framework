#!/usr/bin/env python3
"""
Validate the standard dataset directory structure under benchmark/.

Expected layout per dataset:
  benchmark/<dataset>/
    0_raw_<dataset>/    ← non-BIDS raw data  (may coexist with 1_BIDS)
    1_BIDS_<dataset>/   ← BIDS-formatted data (may coexist with 0_raw)
    2_nnUNet_<dataset>/
      raw/
      preprocessed/
    3_conf_<dataset>/
    4_splits_<dataset>/
    5_scripts_<dataset>/
      00_utils/
        env.sh
      01_<name>/
      02_<name>/
      ...
    6_checkpoints_<dataset>/
    7_analysis_<dataset>/
    8_results_<dataset>/
      01_predictions/        ← REQUIRED (per-fold model predictions)
      02_metrics/            ← REQUIRED (per-run Dice/HD95 + aggregated reports)
      03_aggregated_results/ ← optional
    9_tests_<dataset>/

Naming rules:
  - Slot 0: 0_raw_<dataset>   — non-BIDS raw data
  - Slot 1: 1_BIDS_<dataset>  — BIDS-formatted data
  - Every dataset must have at least one of slot 0 or slot 1 (both allowed)
  - Slots 2–9: N_TYPE_<dataset>
  - Script subdirs:  NN_name          (NN = two-digit number)
  - Script files:    NN_NN_name.ext   (both parts two-digit)
  - 00_utils/ is exempt from file naming rules (config/helper dir)
  - *Trainers.py files are exempt (nnU-Net registration shims must keep this
    exact name for recursive_find_python_class discovery — see the project notes)
  - No unnumbered items at dataset root or inside 5_scripts_*/

Cross-dataset shared code lives in benchmark/00_commun_scripts/ — it is NOT a
dataset and is skipped by this validator (see IGNORED_ENTRIES).
"""

import re
import sys
from pathlib import Path

DATASETS_ROOT = Path(__file__).parent

# ── naming patterns ────────────────────────────────────────────────────────────
DATASET_SUBDIR_RE    = re.compile(r"^\d_[A-Za-z][A-Za-z0-9]*_")   # N_TYPE_<dataset>
SCRIPT_SUBDIR_RE     = re.compile(r"^\d{2}_\w+$")                  # NN_name
SCRIPT_FILE_RE       = re.compile(r"^\d{2}_\d{2}_\w+\.\w+$")       # NN_NN_name.ext

# Slots 2–9: required with fixed TYPE prefixes
REQUIRED_SLOTS = {
    "2": {"nnUNet"},
    "3": {"conf"},
    "4": {"splits"},
    "5": {"scripts"},
    "6": {"checkpoints"},
    "7": {"analysis"},
    "8": {"results"},
    "9": {"tests"},
}

# Slot 0 (raw) and slot 1 (BIDS) data-source slots — may coexist (raw kept alongside BIDSified version)
SLOT0_TYPE = "raw"    # 0_raw_<dataset>  — non-BIDS
SLOT1_TYPE = "BIDS"   # 1_BIDS_<dataset> — BIDS

# Entries directly under benchmark/ that are NOT datasets and must be skipped.
IGNORED_ENTRIES = {
    "00_commun_scripts", "__pycache__",
    "01_commun_results",   # cross-dataset comparison tables, not a per-dataset tree
    "03_archive",          # gitignored dump of excluded/superseded datasets
    "_script_backups",     # pre-edit .bak_YYYYMMDD snapshots, relocated out of 5_scripts_*
}

# Non-dir items permitted at the dataset root (documentation only).
ALLOWED_ROOT_FILES = {"README.md"}

REQUIRED_RESULTS_SUBDIRS = {"01_predictions", "02_metrics"}
REQUIRED_NNUNET_SUBDIRS   = {"raw", "preprocessed"}

# ── 06_evaluate/ canonical entry-point convention (settled 2026-09-27) ─────────
# Ideal sequence, skipping roles that don't apply to a given dataset:
#   06_00_evaluate<_name>.py           (shim)
#   06_01_evaluate_run.sh              (suffix may vary: _own_run/_testset/_ispy2_run/eval_mandible_only)
#   06_02_aggregate_from_config.sh      (or _results / _per_organ_from_config for documented exceptions)
#   06_03_significance_from_config.sh
#   06_04_combined_modality_summary.sh (only if the dataset pools >=2 of its own training modalities)
# A dataset with NEITHER a 06_00 shim NOR a 06_01 evaluate_run (no in-repo
# predict/eval pipeline at all, e.g. healthy-spine-tum's externally-imported
# results) legitimately rebases the whole block down by 1. These are advisory
# checks (WARN, not hard-fail) since real exceptions exist and more may show
# up that aren't foreseen here -- see the project notes' "How experiments work".
EVAL_ROLE_PATTERNS = {
    "shim":          re.compile(r"^06_(\d{2})_evaluate(_\w+)?\.py$"),
    "evaluate_run":  re.compile(r"^06_(\d{2})_(evaluate_run|evaluate_own_run|evaluate_testset|evaluate_\w+_run)\b"),
    "aggregate":     re.compile(r"^06_(\d{2})_aggregate(_from_config|_results|_per_organ_from_config)\b"),
    "significance":  re.compile(r"^06_(\d{2})_significance_from_config\b"),
    "combined":      re.compile(r"^06_(\d{2})_combined_modality_summary\b"),
}
# `eval_mandible_only` (pddca's naming) only counts as fulfilling the
# "evaluate_run" role when nothing else in the directory already matches the
# standard patterns above -- some datasets (hanseg) have BOTH a standard
# evaluate_run.sh AND a separate, differently-purposed eval_mandible_only.sh,
# and treating the latter as a second "evaluate_run" there is a false positive.
EVAL_RUN_FALLBACK_PATTERN = re.compile(r"^06_(\d{2})_eval_mandible_only\b")
EVAL_CANONICAL_NUMBER = {"shim": 0, "evaluate_run": 1, "aggregate": 2, "significance": 3, "combined": 4}
EVAL_LEGACY_ARCHIVE_MIN = 90  # 06_9X = intentionally archived legacy script, not canonical


def validate_eval_entry_points(ds_path: Path, ds: str, slot_map: dict) -> list[str]:
    if "5" not in slot_map:
        return []
    ev_dir = ds_path / slot_map["5"] / "06_evaluate"
    if not ev_dir.is_dir():
        return []

    files = sorted(f.name for f in ev_dir.iterdir() if f.is_file())
    found: dict[str, list[int]] = {role: [] for role in EVAL_ROLE_PATTERNS}
    for f in files:
        for role, pat in EVAL_ROLE_PATTERNS.items():
            m = pat.match(f)
            if m:
                num = int(m.group(1))
                if role == "aggregate" and num >= EVAL_LEGACY_ARCHIVE_MIN:
                    continue  # intentionally archived legacy script, not canonical
                found[role].append(num)

    if not found["evaluate_run"]:
        for f in files:
            m = EVAL_RUN_FALLBACK_PATTERN.match(f)
            if m:
                found["evaluate_run"].append(int(m.group(1)))

    warnings: list[str] = []

    for role, nums in found.items():
        if len(nums) > 1:
            warnings.append(f"  [06_evaluate] multiple '{role}' entry points found at "
                             f"{', '.join(f'06_{n:02d}' for n in sorted(nums))} -- expected exactly one")

    offset = 1 if not found["shim"] and not found["evaluate_run"] else 0

    for role in ("aggregate", "significance", "combined"):
        nums = found[role]
        if not nums:
            continue
        expected = EVAL_CANONICAL_NUMBER[role] - offset
        actual = nums[0]
        if actual != expected:
            warnings.append(f"  [06_evaluate] '{role}' entry point at 06_{actual:02d}_*, expected "
                             f"06_{expected:02d}_* by the project's canonical numbering (advisory -- "
                             f"verify before assuming this needs a fix)")

    cfg_dir = ev_dir / "configs"
    cfg_names = [f.name.lower() for f in cfg_dir.iterdir()] if cfg_dir.is_dir() else []
    if any("significance" in c for c in cfg_names) and not found["significance"]:
        warnings.append("  [06_evaluate] a significance config exists under configs/ but no "
                         "06_XX_significance_from_config.sh wrapper was found")
    if any("combined" in c for c in cfg_names) and not found["combined"]:
        warnings.append("  [06_evaluate] a combined-modality config exists under configs/ but no "
                         "06_XX_combined_modality_summary.sh wrapper was found")

    return warnings


def _slot_and_type(name: str):
    """Parse '2_nnUNet_on-harmony' → ('2', 'nnUNet')."""
    parts = name.split("_", 2)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def validate_dataset(ds_path: Path) -> list[str]:
    errors: list[str] = []
    ds = ds_path.name

    # ── collect present slot numbers ──────────────────────────────────────────
    children = {p.name: p for p in ds_path.iterdir() if not p.name.startswith(".")}
    slot_map: dict[str, str] = {}  # slot_num → dir_name

    for name, path in children.items():
        # Top-level documentation is allowed (and encouraged): README.md plus any
        # other *.md doc a dataset wants at its root (e.g. a cluster-handoff note) --
        # docs are self-explanatory by construction, so there's nothing to enforce
        # beyond "it's a .md file", unlike code or data slots.
        if name in ALLOWED_ROOT_FILES or path.suffix == ".md":
            continue
        if not path.is_dir() and not path.is_symlink():
            errors.append(f"  [root] unexpected file (not a dir): {name}")
            continue
        slot, typ = _slot_and_type(name)
        if slot is None or not DATASET_SUBDIR_RE.match(name):
            errors.append(f"  [root] non-standard name (expected N_TYPE_{ds}): {name!r}")
            continue
        if ds not in name:
            errors.append(f"  [root] dir name does not contain dataset name: {name!r}")
        if slot in slot_map:
            errors.append(f"  [root] duplicate slot {slot}: {slot_map[slot]!r} and {name!r}")
        slot_map[slot] = name

    # ── check slot 0/1 (data source) — at least one must be present ─────────
    # Both 0_raw + 1_BIDS may coexist (raw kept alongside BIDSified version).
    has0 = "0" in slot_map
    has1 = "1" in slot_map
    if not has0 and not has1:
        errors.append(f"  [root] missing data source: need either "
                      f"0_raw_{ds}/ (non-BIDS) or 1_BIDS_{ds}/ (BIDS)")
    if has0:
        _, typ = _slot_and_type(slot_map["0"])
        if typ != SLOT0_TYPE:
            errors.append(f"  [root] slot 0: type {typ!r} should be {SLOT0_TYPE!r}")
    if has1:
        _, typ = _slot_and_type(slot_map["1"])
        if typ != SLOT1_TYPE:
            errors.append(f"  [root] slot 1: type {typ!r} should be {SLOT1_TYPE!r}")

    # ── check slots 2–9 ───────────────────────────────────────────────────────
    for slot, allowed_types in REQUIRED_SLOTS.items():
        if slot not in slot_map:
            errors.append(f"  [root] missing slot {slot} (expected one of: "
                          f"{', '.join(f'{slot}_{t}_{ds}' for t in sorted(allowed_types))})")
            continue
        name = slot_map[slot]
        _, typ = _slot_and_type(name)
        if typ not in allowed_types:
            errors.append(f"  [root] slot {slot}: type {typ!r} not in allowed set {allowed_types}")

    # ── slot 2: nnUNet substructure ───────────────────────────────────────────
    if "2" in slot_map:
        nn_path = ds_path / slot_map["2"]
        for sub in REQUIRED_NNUNET_SUBDIRS:
            if not (nn_path / sub).exists():
                errors.append(f"  [2_nnUNet] missing subdir: {sub}/")

    # ── slot 5: scripts substructure ─────────────────────────────────────────
    if "5" in slot_map:
        sc_path = ds_path / slot_map["5"]
        # env.sh must exist
        if not (sc_path / "00_utils" / "env.sh").exists():
            errors.append("  [5_scripts] missing 00_utils/env.sh")
        # every item must be a numbered dir
        for item in sorted(sc_path.iterdir()):
            if item.name.startswith("."):
                continue
            if not item.is_dir():
                errors.append(f"  [5_scripts] unexpected file at scripts root: {item.name!r}")
                continue
            # Python packages (has __init__.py) are exempt from NN_name naming
            if (item / "__init__.py").exists():
                continue
            if not SCRIPT_SUBDIR_RE.match(item.name):
                errors.append(f"  [5_scripts] non-standard subdir name (expected NN_name): {item.name!r}")
                continue
            # 00_utils is a config/helper dir — file naming not enforced there
            if item.name == "00_utils":
                continue
            # files inside each numbered step subdir must follow NN_NN_name.ext --
            # enforced for code only (.py/.sh); a generated data artifact (a JSON
            # lookup table, a .md report) can sit alongside the script that made it
            # without being forced into the same numbering scheme.
            DATA_EXTS = {".json", ".md", ".csv", ".png", ".pdf", ".txt", ".yaml", ".yml"}
            for f in sorted(item.iterdir()):
                if f.name.startswith(".") or f.is_dir():
                    continue
                if ".bak_" in f.name:
                    errors.append(f"  [5_scripts/{item.name}] pre-edit backup left in the live "
                                  f"tree, should live outside 5_scripts_* (see paper's "
                                  f"cvpr_format_latex_archive/ convention): {f.name!r}")
                    continue
                if f.suffix in DATA_EXTS:
                    continue
                # nnU-Net trainer registration shims must keep their exact class-derived
                # name (recursive_find_python_class scans by filename) — see the project notes.
                if f.name.endswith("Trainers.py"):
                    continue
                if not SCRIPT_FILE_RE.match(f.name):
                    errors.append(f"  [5_scripts/{item.name}] non-standard file name "
                                  f"(expected NN_NN_name.ext): {f.name!r}")

    # ── slot 8: results substructure ─────────────────────────────────────────
    if "8" in slot_map:
        res_path = ds_path / slot_map["8"]
        present = {p.name for p in res_path.iterdir() if p.is_dir()} if res_path.exists() else set()
        for sub in REQUIRED_RESULTS_SUBDIRS:
            if sub not in present:
                errors.append(f"  [8_results] missing subdir: {sub}/")

    # ── 06_evaluate/ canonical entry-point numbering (advisory) ──────────────
    errors.extend(validate_eval_entry_points(ds_path, ds, slot_map))

    return errors


def main() -> int:
    # 02_tasks/ holds datasets one level deeper, grouped by task
    # (02_tasks/<task>/<dataset>/...) -- not a dataset itself.
    tasks_root = DATASETS_ROOT / "02_tasks"
    task_datasets = sorted(
        p for task_dir in (tasks_root.iterdir() if tasks_root.is_dir() else [])
        if task_dir.is_dir() and not task_dir.name.startswith(".")
        for p in task_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    datasets = sorted(p for p in DATASETS_ROOT.iterdir()
                      if p.is_dir() and not p.name.startswith(".")
                      and p.name not in IGNORED_ENTRIES
                      and p.name != "02_tasks") + task_datasets

    if not datasets:
        print("No dataset directories found under benchmark/")
        return 1

    total_errors = 0
    for ds_path in datasets:
        errors = validate_dataset(ds_path)
        status = "OK" if not errors else f"FAIL ({len(errors)} issue{'s' if len(errors) != 1 else ''})"
        print(f"{'✓' if not errors else '✗'}  {ds_path.name:<30} {status}")
        for e in errors:
            print(e)
        total_errors += len(errors)

    print()
    if total_errors == 0:
        print("All datasets pass structure validation.")
        return 0
    else:
        print(f"{total_errors} total issue(s) found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
