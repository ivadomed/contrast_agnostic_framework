#!/usr/bin/env python3
"""
Validate the standard dataset directory structure under datasets/.

Expected layout per dataset:
  datasets/<dataset>/
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
      01_results/
      02_nnUNet_results/
      03_aggregated_results/
    9_tests_<dataset>/

Naming rules:
  - Slot 0: 0_raw_<dataset>   — non-BIDS raw data
  - Slot 1: 1_BIDS_<dataset>  — BIDS-formatted data
  - Every dataset must have at least one of slot 0 or slot 1 (both allowed)
  - Slots 2–9: N_TYPE_<dataset>
  - Script subdirs:  NN_name          (NN = two-digit number)
  - Script files:    NN_NN_name.ext   (both parts two-digit)
  - 00_utils/ is exempt from file naming rules (config/helper dir)
  - No unnumbered items at dataset root or inside 5_scripts_*/
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

# Slot 0 and slot 1 are mutually exclusive data-source slots
SLOT0_TYPE = "raw"    # 0_raw_<dataset>  — non-BIDS
SLOT1_TYPE = "BIDS"   # 1_BIDS_<dataset> — BIDS

REQUIRED_RESULTS_SUBDIRS = {"01_results", "02_nnUNet_results", "03_aggregated_results"}
REQUIRED_NNUNET_SUBDIRS   = {"raw", "preprocessed"}


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
            # files inside each numbered step subdir must follow NN_NN_name.ext
            for f in sorted(item.iterdir()):
                if f.name.startswith(".") or f.is_dir():
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

    return errors


def main() -> int:
    datasets = sorted(p for p in DATASETS_ROOT.iterdir()
                      if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__")

    if not datasets:
        print("No dataset directories found under datasets/")
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
