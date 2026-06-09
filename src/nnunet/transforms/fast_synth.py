"""
Backward-compatibility shim.  The canonical implementation lives in
src/v26_6_synthesis.py — import from there for new code.
"""
from src.synthesis.v26_6_synthesis import (  # noqa: F401
    synthesize_volume_fast,
    synthesize_patch_fast,
    compute_kmeans_centroids,
    gpu_spatial_augment,
    C_CHOICES      as _C_CHOICES,
    BLUR_SIGMAS    as _BLUR_SIGMAS,
    DARK_THRESHOLD as _DARK_THRESHOLD,
)
