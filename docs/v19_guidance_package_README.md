# V19 Stochastic Semantic Decoupling — Collaborator Package

This document covers everything needed to use V19 guidance map as an online GPU augmentation,
both standalone and inside nnU-Net training.

---

## What's in the package

```
v19_augmentation/
├── src/
│   ├── target_generators.py        ← V19 generator class (and all predecessors)
│   ├── histogram_ops.py            ← DifferentiableHistogram3D
│   └── auglab_adapter.py           ← AugLab / Kornia ImageOnlyTransform adapter
├── nnunetv2_trainer/
│   └── nnUNetTrainerV19Aug.py      ← Drop-in nnU-Net trainer subclass
└── README.md                       ← this file
```

---

## 1. Running things: standalone

### Minimal usage — no nnU-Net, no AugLab

```python
import torch
from src.histogram_ops import DifferentiableHistogram3D
from src.target_generators import V19LabelConditionedTextureGenerator

gen  = V19LabelConditionedTextureGenerator(label_classes=[1, 2, 3])
hist = DifferentiableHistogram3D(num_bins=64, value_range=(0.0, 1.0))

# image:  [B, C, D, H, W], float32, values in [0, 1]
# labels: [B, 1, D, H, W], long, integer class indices (0 = background)
image  = torch.rand(2, 1, 64, 64, 64)
labels = torch.randint(0, 4, (2, 1, 64, 64, 64))

target_hist, guidance_map, _ = gen(image, hist, labels=labels)
# guidance_map: [B, C, D, H, W], same shape as image, values in [0, 1]
```

**Without labels** (label-free mode, base v18_6 synthesis only):

```python
target_hist, guidance_map, _ = gen(image, hist, labels=None)
```

---

## 2. Integrating in an AugLab pipeline

The `auglab_adapter.py` file provides `RandomV19ContrastGPU`, which wraps V19 as
a Kornia `ImageOnlyTransform`-compatible class.

### Where to put it

```
your_project/
  auglab/
    transforms/
      gpu/
        contrast.py     ← paste RandomV19ContrastGPU here (or as a new file)
        base.py
```

Make sure `src/target_generators.py` and `src/histogram_ops.py` are importable
(add the package root to `PYTHONPATH` or install via pip editable mode).

### Usage

```python
from auglab.transforms.gpu.contrast import RandomV19ContrastGPU

transform = RandomV19ContrastGPU(
    label_classes=[1, 2, 3],   # BraTS default; change for your dataset
    num_bins=64,
    p=0.8,
)

# AugLab passes the seg mask in params['seg'] as one-hot [B, C_seg, D, H, W]
output = transform(image_batch, params={"seg": one_hot_seg_batch}, flags={})
```

**Seg mask formats accepted automatically:**

| Shape | Interpretation |
|---|---|
| `[B, C_seg, D, H, W]` with `C_seg > 1` | One-hot — collapsed via `argmax + 1` |
| `[B, 1, D, H, W]` | Integer index — used directly |
| `None` / not in params | Label-free mode (base synthesis only) |

---

## 3. Integrating in nnU-Net

### Step 1 — copy `nnUNetTrainerV19Aug.py`

```
nnunetv2/training/nnUNetTrainer/nnUNetTrainerV19Aug.py
```

This file provides three trainer classes:

| Class | Description |
|---|---|
| `nnUNetTrainerV19Aug` | Generic trainer, `label_classes=None` → BraTS [1,2,3] default |
| `nnUNetTrainerV19AugBraTS` | Explicitly BraTS-labelled |
| `nnUNetTrainerV19AugBinary` | Binary (foreground = label 1) |

### Step 2 — make `src/` importable

The trainer imports from `src.target_generators` and `src.histogram_ops`.
Add the V19 package root to `PYTHONPATH` **before** launching training:

```bash
export PYTHONPATH=/path/to/v19_augmentation:$PYTHONPATH
```

Or install via editable mode:

```bash
pip install -e /path/to/v19_augmentation
```

### Step 3 — run training

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
    -tr nnUNetTrainerV19Aug
```

For BraTS specifically:

```bash
nnUNetv2_train Dataset001_BraTS 3d_fullres 0 \
    -tr nnUNetTrainerV19AugBraTS
```

### Step 4 — tune the augmentation probability

The default is `aug_p = 0.5` (applied to 50% of training batches).
To change it, subclass and override:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV19Aug import nnUNetTrainerV19Aug

class MyV19Trainer(nnUNetTrainerV19Aug):
    aug_p = 0.8
    label_classes = [1, 2, 3, 4, 5]   # your dataset's foreground labels
    num_hist_bins = 32
```

Place this file anywhere inside `nnunetv2/training/nnUNetTrainer/` and use it
with `-tr MyV19Trainer`.

### How it works in the training loop

```
CPU pipeline (batchgenerators):
  spatial augmentation → noise → blur → gamma → mirror → seg transforms
                                    ↓
                              batch arrives at train_step
                                    ↓
GPU (train_step override):
  [if rand() < aug_p]
    1. per-sample min-max normalise image to [0, 1]
    2. extract full-res seg from target[0]
    3. apply V19 → synthesised image in [0, 1]
    4. rescale back to original intensity range
                                    ↓
              standard nnU-Net forward / loss / backward
```

The V19 step runs inside `torch.no_grad()` — no extra VRAM for augmentation
gradients. It adds roughly 2–5 ms per batch on a 3090 at 128³ patch size.

---

## 4. ⚠️ Critical: label_classes for non-BraTS datasets

V19's semantic decoupling loop iterates over `label_classes`. The default is
`[1, 2, 3]` (BraTS NCR / ED / ET). **Always set this to match your dataset's
foreground label indices** or decoupling has no effect:

```python
# Spine segmentation with 7 vertebra classes labelled 1–7
nnUNetTrainerV19Aug.label_classes = [1, 2, 3, 4, 5, 6, 7]

# Prostate segmentation: peripheral zone=1, central gland=2
nnUNetTrainerV19Aug.label_classes = [1, 2]
```

---

## 5. Dependencies

```
torch >= 2.0
kornia >= 0.7
```

No additional dependencies beyond a standard nnU-Net environment.
