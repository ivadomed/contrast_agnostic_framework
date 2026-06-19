#!/usr/bin/env bash
# One-shot venv package install, meant to be run inside a salloc allocation
# (heavy pip/dependency-resolution work doesn't belong on the login node).
# Not part of the run_job abstraction — this is a manual bootstrap step for
# rebuilding .venv from scratch, kept here only so the command is logged.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11
source .venv/bin/activate
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
WHEELS="$SCRATCH/nnunet_wheels"

echo "=== base ML stack (wheelhouse, no internet needed) ==="
pip install --no-cache-dir --no-index \
    "torch==2.12.1" torchvision numpy scipy matplotlib pytest hydra-core wandb nibabel kornia

echo "=== monai (pinned 1.5.2) + curated 'all' extras (wheelhouse) ==="
# Plain "monai[all]" backtracks to a non-functional monai==0.1.0: the wheelhouse
# has no `clearml` at all, and pip silently degrades the whole resolution rather
# than erroring. clearml/itk/pyamg/nni are excluded deliberately, not by omission:
#   - clearml: not in wheelhouse; this project tracks experiments via WandB, not
#     ClearML, so there's no reason to chase it from PyPI.
#   - itk: wheelhouse only has cp38/cp39 builds, none for our python 3.11.
#   - pyamg: not in wheelhouse (algebraic multigrid solvers; unused by this
#     project — SimpleITK/nibabel already cover the imaging I/O monai[all] would
#     otherwise need itk for).
#   - nni: Microsoft's AutoML/hyperparameter-search tool, unrelated to this
#     project's stack; its hyperopt==0.1.2 pin isn't in the wheelhouse either
#     (only 0.2.5/0.2.7 are), and chasing an exact ancient pin from PyPI for an
#     unused dependency isn't worth it.
#   - mlflow: another unused experiment tracker (WandB is this project's actual
#     one); its pyarrow dependency hits the wheelhouse's deliberate "dummy"
#     wheel telling you to `module load arrow/x.y.z` before activating the venv
#     instead of pip-installing it — not worth doing for a tracker nothing here
#     uses.
pip install --no-cache-dir --no-index \
    "monai==1.5.2" einops fire "gdown>=4.7.3" h5py huggingface-hub jsonschema lmdb \
    "lpips==0.1.4" "matplotlib>=3.6.3" nibabel ninja nvidia-ml-py "onnx>=1.13.0" \
    openslide-python optuna pandas "pillow!=8.3.0" psutil pydicom pynrrd \
    "pytorch-ignite==0.4.11" pyyaml "scikit-image>=0.14.2" tensorboard tensorboardX \
    torchio torchvision "tqdm>=4.47.0" zarr imagecodecs tifffile \
    "scipy>=1.12.0"

echo "=== auglab's own core deps (wheelhouse, batchgeneratorsv2 from pre-downloaded wheel) ==="
pip install --no-cache-dir --no-deps "${WHEELS}/batchgeneratorsv2-0.3.3-py3-none-any.whl"
pip install --no-cache-dir --no-index torchio

echo "=== remaining nnunetv2 transitive deps (wheelhouse, version-pinned where needed) ==="
pip install --no-cache-dir --no-index \
    "timm<1.0.23" connected-components-3d blosc2 "SimpleITK>=2.2.1" \
    "scikit-image>=0.19.3" scikit-learn pandas graphviz tifffile requests \
    seaborn imagecodecs yacs einops batchgenerators

echo "=== pre-downloaded packages NOT in the wheelhouse (or too old there) ==="
pip install --no-cache-dir --no-deps \
    "${WHEELS}/acvl_utils-0.2.6.tar.gz" \
    "${WHEELS}/dynamic_network_architectures-0.4.4-py3-none-any.whl" \
    "${WHEELS}/nnunetv2-2.7.0.tar.gz"

echo "=== auglab (editable, from its own git repo under sub-workspaces/) ==="
pip install --no-cache-dir --no-index -e sub-workspaces/auglab_workspace/AugLab

echo "=== registering AugLab trainers into nnunetv2 ==="
auglab_add_nnunettrainer -t nnUNetTrainerDAExt --overwrite
auglab_add_nnunettrainer -t nnUNetTrainerTest --overwrite

echo "=== restoring this project's nnunetv2 patches (see CLAUDE.md) ==="
SITE_PKGS="$(python -c 'import nnunetv2, os; print(os.path.dirname(os.path.dirname(nnunetv2.__file__)))')"
cp src/nnunet/patches/nnunet_logger.py "${SITE_PKGS}/nnunetv2/training/logging/nnunet_logger.py"
cp datasets/brats2024-glioma/5_scripts_brats2024-glioma/02_nnunet/BraTS2024GliomaTrainers.py "${SITE_PKGS}/nnunetv2/training/nnUNetTrainer/"
cp datasets/chaos/5_scripts_chaos/02_nnunet/CHAOSTrainers.py "${SITE_PKGS}/nnunetv2/training/nnUNetTrainer/"
cp datasets/on-harmony/5_scripts_on-harmony/02_nnunet/OnHarmonyTrainers.py "${SITE_PKGS}/nnunetv2/training/nnUNetTrainer/"

echo "=== verification ==="
export NNUNET_PROJECT_ROOT="$(pwd)"
python -c "
import importlib.metadata as m
import torch, numpy, scipy, kornia, nibabel, wandb, monai, batchgeneratorsv2, torchio
import auglab, nnunetv2
from nnunetv2.training.nnUNetTrainer.BraTS2024GliomaTrainers import nnUNetTrainerBraTS2024GliomaV26_6_2
from nnunetv2.training.nnUNetTrainer.CHAOSTrainers import nnUNetTrainerCHAOSV26_6_2
from nnunetv2.training.nnUNetTrainer.OnHarmonyTrainers import nnUNetTrainerOnHarmonyV26_6_2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDAExt import nnUNetTrainerDAExt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTest import nnUNetTrainerTest
from nnunetv2.training.logging.nnunet_logger import WandbLogger
print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())
print('nnunetv2', m.version('nnunetv2'))
print('monai', monai.__version__)
print('auglab', m.version('auglab'))
print('ALL IMPORTS OK')
"
echo "=== done ==="
