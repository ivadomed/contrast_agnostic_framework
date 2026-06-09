"""Verify syntax of all scripts in the on-harmony dataset."""
import subprocess
import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "5_scripts_on-harmony"

_py_scripts  = sorted(p for p in SCRIPTS_DIR.rglob("*.py")  if "__pycache__" not in str(p))
_sh_scripts  = sorted(p for p in SCRIPTS_DIR.rglob("*.sh"))


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
    """Every numbered step dir (01–07) must exist."""
    expected = ["00_utils", "01_create_splits", "02_convert_to_nnunet",
                "03_preprocess", "04_train", "05_predict", "06_evaluate",
                "07_benchmark"]
    for name in expected:
        assert (SCRIPTS_DIR / name).exists(), f"Missing step dir: {name}"


def test_each_step_has_at_least_one_script():
    """Every numbered step dir (except 00_utils and 05_predict) must have ≥1 script."""
    skip = {"00_utils", "05_predict"}
    for step_dir in sorted(SCRIPTS_DIR.iterdir()):
        if not step_dir.is_dir() or step_dir.name in skip:
            continue
        scripts = [f for f in step_dir.iterdir()
                   if f.suffix in (".py", ".sh") and not f.name.startswith(".")]
        assert scripts, f"Step dir has no scripts: {step_dir.name}"
