#!/usr/bin/env python3

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "scripts" / "nnunet_scripts" / "convert_to_nnunet_format.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()