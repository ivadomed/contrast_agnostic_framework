#!/usr/bin/env bash
# One-shot venv package install, meant to be run inside a salloc allocation
# (heavy pip/dependency-resolution work doesn't belong on the login node).
# Not part of the run_job abstraction — this is a manual bootstrap step for
# rebuilding .venv from scratch, kept here only so the command is logged.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
module load python/3.11
[ -d .venv ] || virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-cache-dir --no-index --upgrade pip
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
WHEELS="$SCRATCH/nnunet_wheels"

echo "=== base ML stack (wheelhouse, no internet needed) ==="
# kornia pinned <0.8: AugLab's GPU transforms do `from kornia.core import Tensor`,
# an alias kornia dropped in 0.8.x (kornia.core.Tensor no longer exists there;
# confirmed by hand on Killarney 2026-07-06 — 0.8.3 breaks the import, 0.7.2 has it).
# triton: NOT a declared torch dependency (`pip show torch` lists none), but
# nnUNet_compile=1 uses torch.compile's inductor backend, which hard-requires it
# ("torch._inductor.exc.TritonMissing") — without it every GPU fold dies at the
# first train_step, Epoch 0, ~15s in. This was the open-ms Killarney blocker
# (2026-07-06): all 24 Vulcan folds crashed the same way, invisible there because
# their sbatch stdout went to a node-local /tmp path wiped at job end (see the six
# open-ms 04_0*_train_*.sh LOG_DIR fix). Confirmed fix: install triton explicitly.
pip install --no-cache-dir --no-index \
    "torch==2.12.1" torchvision numpy scipy matplotlib pytest hydra-core wandb nibabel "kornia==0.7.2" "triton==3.6.0"

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

# Auto-discover every dataset's nnU-Net trainer-registration shim instead of a
# hand-maintained list -- a hardcoded list silently misses any new dataset's
# shim (training then fails with a cryptic "trainer not found", with no link
# back to this root cause). Only benchmark/02_tasks/ (active datasets) is
# scanned -- benchmark/03_archive/ datasets don't need their trainers
# registered for anyone to actually train against.
while IFS= read -r shim; do
    echo "  -> ${shim}"
    cp "${shim}" "${SITE_PKGS}/nnunetv2/training/nnUNetTrainer/"
done < <(find benchmark/02_tasks -path '*/02_nnunet/*Trainers.py' | sort)

echo "=== verification ==="
export NNUNET_PROJECT_ROOT="$(pwd)"
python -c "
import importlib.metadata as m
import torch, numpy, scipy, kornia, nibabel, wandb, monai, batchgeneratorsv2, torchio
import auglab, nnunetv2
import importlib.util, pathlib
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerDAExt import nnUNetTrainerDAExt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTest import nnUNetTrainerTest
from nnunetv2.training.logging.nnunet_logger import WandbLogger
# Every auto-discovered *Trainers.py shim must import cleanly on its own --
# this is what recursive_find_python_class relies on at train time, so a
# broken shim here means training would fail later with no earlier warning.
trainer_dir = pathlib.Path(nnunetv2.__file__).parent / 'training' / 'nnUNetTrainer'
for shim in sorted(trainer_dir.glob('*Trainers.py')):
    spec = importlib.util.spec_from_file_location(shim.stem, shim)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f'  shim OK: {shim.name}')
print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())
print('nnunetv2', m.version('nnunetv2'))
print('monai', monai.__version__)
print('auglab', m.version('auglab'))
print('ALL IMPORTS OK')
"
echo "=== done ==="
