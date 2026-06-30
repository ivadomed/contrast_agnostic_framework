# Registration file for the LOW-DATA-REGIME benchmark — ADDITIVE, separate from
# CHAOSTrainers.py so no existing result-producing file is touched. Like CHAOSTrainers.py,
# nnU-Net's recursive_find_python_class only searches inside the installed nnunetv2
# package, so copy this there after every fresh venv/nnunetv2 install:
#   cp datasets/chaos/5_scripts_chaos/02_nnunet/CHAOSLowDataTrainers.py \
#      .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/
"""
Makes the CHAOS low-data trainer variants discoverable by nnU-Net.

Each class is just LowDataMixin layered on an EXISTING CHAOS trainer (no behaviour
change unless RUN_ID carries a `_lowdata_n<NN>` marker — see lowdata_mixin.py). One
variant per distinct underlying trainer class used by the 6 benchmarked methods:

    baseline                          → nnUNetTrainerCHAOSBaseline
    v26_6_2 (train050_val100)         → nnUNetTrainerCHAOSV26_6_2_p50
    auglab_default / synthseg_EM /    → nnUNetTrainerCHAOSAugLabDefault
        synthseg_noEM                   (these 3 differ only by AUGLAB_PARAMS_GPU_JSON)
    auglabAug_v26_6_2 (t025_v100)     → nnUNetTrainerCHAOSAugLabValSynth

Project root resolved from NNUNET_PROJECT_ROOT (set by the train/predict scripts),
exactly as CHAOSTrainers.py does.
"""
import os
import sys

_root = os.environ.get("NNUNET_PROJECT_ROOT", "")
if _root:
    _scripts = os.path.join(_root, "datasets", "chaos", "5_scripts_chaos")
    _commun = os.path.join(_root, "datasets", "00_commun_scripts", "00_01_train")
    for _p in (_root, _scripts, _commun):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from lowdata_mixin import LowDataMixin
    from chaos.trainers.baseline import nnUNetTrainerCHAOSBaseline
    from chaos.trainers.v26_6_2_p50 import nnUNetTrainerCHAOSV26_6_2_p50
    from chaos.trainers.auglab_default import nnUNetTrainerCHAOSAugLabDefault
    from chaos.trainers.auglab_valsynth import nnUNetTrainerCHAOSAugLabValSynth

    class nnUNetTrainerCHAOSBaselineLowData(LowDataMixin, nnUNetTrainerCHAOSBaseline):
        pass

    class nnUNetTrainerCHAOSV26_6_2_p50LowData(LowDataMixin, nnUNetTrainerCHAOSV26_6_2_p50):
        pass

    class nnUNetTrainerCHAOSAugLabDefaultLowData(LowDataMixin, nnUNetTrainerCHAOSAugLabDefault):
        pass

    class nnUNetTrainerCHAOSAugLabValSynthLowData(LowDataMixin, nnUNetTrainerCHAOSAugLabValSynth):
        pass

except ImportError as e:
    import warnings
    warnings.warn(
        f"[CHAOSLowDataTrainers] Import failed: {e}. "
        "Make sure NNUNET_PROJECT_ROOT is set to the mri_synthesis_project root "
        "and that datasets/chaos/5_scripts_chaos + datasets/00_commun_scripts are importable."
    )
