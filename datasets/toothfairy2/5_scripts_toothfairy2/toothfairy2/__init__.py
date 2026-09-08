# toothfairy2 dataset-specific Python package (I-SPY2 breast tumour segmentation,
# primary training set as of the 2026-09-04 pivot).
# env.sh adds 5_scripts_ispy2/ to PYTHONPATH so this imports as `toothfairy2`:
#   from toothfairy2.trainers.baseline import nnUNetTrainerToothFairy2Baseline
# nnU-Net discovers the concrete trainer classes via the registration shim
# 02_nnunet/ToothFairy2Trainers.py (copied into the installed nnunetv2 package).
#
# Adapted from datasets/ambl/5_scripts_ambl/ambl/ (closest structural analog:
# two-training-modality breast-tumour MRI segmentation with the full 6-method
# suite + causal-ablation ladder already built and run) — with N_EXPECTED_FOLDS
# fixed to 3 (see trainers/base.py) since toothfairy2's splits_final.json has exactly
# 3 entries, unlike ambl's legacy 4-entry file with an unused fold 3.
__all__: list[str] = []
