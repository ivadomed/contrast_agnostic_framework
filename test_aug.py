import torch
import torch.nn as nn
from src.intensity_ops import _shared_rand, PartialSynthSegAugmentation3D
aug = PartialSynthSegAugmentation3D()
img = torch.rand(2, 1, 32, 32, 32)
lab = torch.randint(0, 4, (2, 1, 32, 32, 32))
y, l = aug(img, lab)
print(y.shape)
