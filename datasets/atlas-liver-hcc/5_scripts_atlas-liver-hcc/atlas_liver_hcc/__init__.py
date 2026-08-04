# atlas-liver-hcc dataset-specific Python package (HCC liver tumour segmentation).
# env.sh adds 5_scripts_atlas-liver-hcc/ to PYTHONPATH so this imports as `atlas_liver_hcc`:
#   from atlas_liver_hcc.trainers.baseline import nnUNetTrainerAtlasLiverHCCBaseline
# nnU-Net discovers the concrete trainer classes via the registration shim
# 02_nnunet/AtlasLiverHCCTrainers.py (copied into the installed nnunetv2 package).
__all__: list[str] = []
