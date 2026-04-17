import torch
from src.intensity_ops import PartialSynthSegAugmentation3D

aug = PartialSynthSegAugmentation3D(sigma=1.0)
img = torch.rand(2, 1, 32, 32, 32)
lab = torch.zeros(2, 1, 32, 32, 32)
lab[:, :, 10:20, 10:20, 10:20] = 1.0

# Verify logic
synth_map = torch.zeros_like(img)
mask_weights = torch.zeros_like(img)
for c in [1]:
    mask = (lab == c).float()
    synth_map += mask * 5.0
    mask_weights += mask

blurred_synth = aug._separable_gaussian_blur(synth_map)
blurred_weights = aug._separable_gaussian_blur(mask_weights)
safe = blurred_weights.clamp_min(1e-5)
blended = blurred_synth / safe
alpha = blurred_weights.clamp(0.0, 1.0)
y = (1.0 - alpha) * img + alpha * blended
print(y.shape)
print("Max alpha:", alpha.max())
print("Min alpha:", alpha.min())
