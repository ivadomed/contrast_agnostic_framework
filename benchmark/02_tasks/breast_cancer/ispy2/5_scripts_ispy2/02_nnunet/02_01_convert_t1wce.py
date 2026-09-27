#!/usr/bin/env python3
"""Build the ispy2 nnU-Net raw dataset for the t1wce training modality.

All logic (case list, path resolution, LPS check, label sanitation, dataset.json)
lives in 02_00_convert_lib.py — read that file's docstring first. This wrapper
exists only so each modality has its own numbered entry point, matching the
02_0X convention used by every other dataset in this repo.

Run through:  bash 02_01_convert_t1wce.sh   (run_job / Slurm — several GB of
NIfTI I/O, past the login-node exception).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ispy2_convert_lib", Path(__file__).with_name("02_00_convert_lib.py")
)
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

if __name__ == "__main__":
    lib.convert("t1wce")
