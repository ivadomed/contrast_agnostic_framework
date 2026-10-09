# pansegdata dataset-specific Python package (pansegdata pancreas segmentation,
# primary training set as of the 2026-09-04 pivot).
# env.sh adds 5_scripts_pansegdata/ to PYTHONPATH so this imports as `pansegdata`:
#   from pansegdata.trainers.baseline import nnUNetTrainerPANSEGDATABaseline
# nnU-Net discovers the concrete trainer classes via the registration shim
# 02_nnunet/PANSEGDATATrainers.py (copied into the installed nnunetv2 package).
#
# Adapted from benchmark/03_archive/ambl/5_scripts_ambl/ambl/ (closest structural analog:
# two-training-modality MRI segmentation with the full 6-method
# suite + causal-ablation ladder already built and run) — with N_EXPECTED_FOLDS
# fixed to 3 (see trainers/base.py) since pansegdata's splits_final.json has exactly
# 3 entries, unlike ambl's legacy 4-entry file with an unused fold 3.
__all__: list[str] = []
