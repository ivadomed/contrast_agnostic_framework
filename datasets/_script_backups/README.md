# Script backups

Pre-edit `.bak_YYYYMMDD` snapshots of dataset pipeline scripts, taken before a
refactor, relocated out of each dataset's `5_scripts_<name>/` tree so that tree
stays structurally clean (see `datasets/validate_standard_dataset_structure.py`)
without losing the snapshot. Mirrors `paper/cvpr_format_latex_archive/`'s role
for the paper's own pre-edit backups.

Each file keeps its original relative path under its dataset, e.g.:

```
datasets/_script_backups/chaos/5_scripts_chaos/06_evaluate/06_33_ladder_summary_t2spir.py.bak_20260803
```

is the pre-refactor version of
`datasets/chaos/5_scripts_chaos/06_evaluate/06_33_ladder_summary_t2spir.py`.

Not tracked in git (matches how these files were never version-controlled at
their original location either) and ignored by the structure validator.
