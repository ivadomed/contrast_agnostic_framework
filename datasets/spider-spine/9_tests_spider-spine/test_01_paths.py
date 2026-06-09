"""Verify all paths for the spider-spine dataset are correct after reorganisation."""
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # datasets/
from conftest import source_env, env_vars  # noqa: E402

DATASET_ROOT = Path(__file__).parent.parent
ENV_SH = DATASET_ROOT / "5_scripts_spider-spine/00_utils/env.sh"

# ── env.sh ───────────────────────────────────────────────────────────────────

def test_env_sh_exists():
    assert ENV_SH.exists(), f"Missing env.sh: {ENV_SH}"

def test_env_sh_sources_without_error():
    r = subprocess.run(["bash", "-c", f'source "{ENV_SH}"'], capture_output=True, text=True)
    assert r.returncode == 0, f"env.sh errored:\n{r.stderr}"

def test_env_sh_exports_required_vars():
    required = ["RAW_ROOT", "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results",
                "SPLITS_DIR", "CHECKPOINTS_DIR", "DATASET_NAME"]
    exported = env_vars(ENV_SH)
    missing = [v for v in required if v not in exported]
    assert not missing, f"env.sh missing exports: {missing}"

# ── raw data ─────────────────────────────────────────────────────────────────

def test_raw_root_exists():
    p = Path(source_env(ENV_SH, "RAW_ROOT"))
    assert p.exists(), f"RAW_ROOT not found: {p}"

def test_raw_root_has_data():
    p = Path(source_env(ENV_SH, "RAW_ROOT"))
    assert list(p.iterdir()), f"RAW_ROOT is empty: {p}"

# ── nnUNet ───────────────────────────────────────────────────────────────────

def test_nnunet_raw_dir_exists():
    p = Path(source_env(ENV_SH, "nnUNet_raw"))
    assert p.exists(), f"nnUNet_raw not found: {p}"

def test_nnunet_preprocessed_dir_exists():
    p = Path(source_env(ENV_SH, "nnUNet_preprocessed"))
    assert p.exists(), f"nnUNet_preprocessed not found: {p}"

@pytest.mark.parametrize("dataset_id", [
    "Dataset102_SpiderSpine",
    "Dataset102_SpiderSpine_t1",
    "Dataset102_SpiderSpine_t2",
    "Dataset122_SpiderSpine_t1",
    "Dataset123_SpiderSpine_t2",
    "Dataset124_SpiderSpine_t2space",
])
def test_nnunet_raw_dataset_present(dataset_id):
    raw = Path(source_env(ENV_SH, "nnUNet_raw"))
    assert (raw / dataset_id).exists(), f"Missing in nnUNet_raw: {dataset_id}"

@pytest.mark.parametrize("dataset_id", [
    "Dataset122_SpiderSpine_t1",
    "Dataset123_SpiderSpine_t2",
])
def test_nnunet_preprocessed_dataset_present(dataset_id):
    pre = Path(source_env(ENV_SH, "nnUNet_preprocessed"))
    assert (pre / dataset_id).exists(), f"Missing in nnUNet_preprocessed: {dataset_id}"

# ── conf / splits / checkpoints ──────────────────────────────────────────────

def test_conf_data_yaml_exists():
    p = DATASET_ROOT / "3_conf_spider-spine" / "data.yaml"
    assert p.exists(), f"Missing: {p}"

def test_results_subdirs_exist():
    results = DATASET_ROOT / "8_results_spider-spine"
    for sub in ("01_results", "02_nnUNet_results", "03_aggregated_results"):
        assert (results / sub).exists(), f"Missing results subdir: {sub}"
