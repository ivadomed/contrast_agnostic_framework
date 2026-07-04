# open-ms dataset-specific Python package (brain MS lesion segmentation).
# env.sh adds 5_scripts_open-ms/ to PYTHONPATH so this imports as `open_ms`:
#   from open_ms.trainers.v26_6_2 import nnUNetTrainerOpenMSV26_6_2
# nnU-Net discovers the concrete trainer classes via the registration shim
# 02_nnunet/OpenMSTrainers.py (copied into the installed nnunetv2 package).
__all__: list[str] = []
