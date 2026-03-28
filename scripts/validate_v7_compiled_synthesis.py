from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import MRI_Synthesis_Net
from src.histogram_ops import DifferentiableHistogram3D
from src.lightning_modules import CompiledSynthesisWrapper


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MRI_Synthesis_Net(in_channels=2, out_channels=1).to(device)
    model.eval()
    hist = DifferentiableHistogram3D(num_bins=128, value_range=(0.0, 1.0)).to(device)
    wrapper = CompiledSynthesisWrapper(model, hist, "v7").to(device)

    x = torch.rand(1, 1, 32, 32, 32, device=device, dtype=torch.float32)
    with torch.inference_mode():
        y = wrapper(x, num_bins=128, num_chunks=8, dark_threshold=0.05)

    assert y.shape == x.shape, f"Unexpected output shape: {y.shape}"
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0), "Output outside [0, 1]"

    print("OK: v7 compiled synthesis forward path executed successfully")


if __name__ == "__main__":
    main()
