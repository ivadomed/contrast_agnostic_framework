"""Verify syntax of all scripts in the ms-multi-spine dataset."""
import subprocess
import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "5_scripts_ms-multi-spine"

_py_scripts = sorted(p for p in SCRIPTS_DIR.rglob("*.py")  if "__pycache__" not in str(p))
_sh_scripts = sorted(p for p in SCRIPTS_DIR.rglob("*.sh"))


@pytest.mark.parametrize("script", _py_scripts, ids=[p.name for p in _py_scripts])
def test_python_syntax(script):
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"Syntax error in {script.relative_to(SCRIPTS_DIR)}:\n{r.stderr}"


@pytest.mark.parametrize("script", _sh_scripts, ids=[p.name for p in _sh_scripts])
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"Bash syntax error in {script.relative_to(SCRIPTS_DIR)}:\n{r.stderr}"


def test_scripts_dir_has_expected_steps():
    expected = ["00_utils", "01_create_splits", "02_convert_to_nnunet",
                "03_preprocess", "04_train", "05_predict", "06_evaluate"]
    for name in expected:
        assert (SCRIPTS_DIR / name).exists(), f"Missing step dir: {name}"
