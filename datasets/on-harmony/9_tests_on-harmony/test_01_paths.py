"""Verify all paths for the on-harmony dataset are correct after reorganisation."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # datasets/
from conftest import source_env, env_vars  # noqa: E402

DATASET_ROOT = Path(__file__).parent.parent
ENV_SH = DATASET_ROOT / "5_scripts_on-harmony/00_utils/env.sh"

# ── env.sh ──────────────────────────────────────────────────────────────────

def test_env_sh_exists():
    assert ENV_SH.exists(), f"Missing env.sh: {ENV_SH}"

def test_env_sh_sources_without_error():
    import subprocess
    r = subprocess.run(["bash", "-c", f'source "{ENV_SH}"'], capture_output=True, text=True)
    assert r.returncode == 0, f"env.sh errored:\n{r.stderr}"

def test_env_sh_exports_required_vars():
    required = ["BIDS_ROOT", "nnUNet_raw", "nnUNet_preprocessed",
                "nnUNet_results", "SPLITS_DIR", "CHECKPOINTS_DIR", "DATASET_NAME"]
    exported = env_vars(ENV_SH)
    missing = [v for v in required if v not in exported]
    assert not missing, f"env.sh missing exports: {missing}"

# ── data source ──────────────────────────────────────────────────────────────

def test_bids_root_exists():
    p = Path(source_env(ENV_SH, "BIDS_ROOT"))
    assert p.exists(), f"BIDS_ROOT not found: {p}"

def test_bids_root_has_subjects():
    p = Path(source_env(ENV_SH, "BIDS_ROOT"))
    subjects = list(p.glob("sub-*"))
    assert subjects, f"No sub-* dirs found under BIDS_ROOT: {p}"

def test_bids_derivatives_exist():
    p = Path(source_env(ENV_SH, "BIDS_ROOT")) / "derivatives"
    assert p.exists(), f"derivatives/ missing under BIDS_ROOT: {p}"

# ── nnUNet ───────────────────────────────────────────────────────────────────

def test_nnunet_raw_dir_exists():
    p = Path(source_env(ENV_SH, "nnUNet_raw"))
    assert p.exists(), f"nnUNet_raw not found: {p}"

def test_nnunet_preprocessed_dir_exists():
    p = Path(source_env(ENV_SH, "nnUNet_preprocessed"))
    assert p.exists(), f"nnUNet_preprocessed not found: {p}"

@pytest.mark.parametrize("dataset_id", [
    "Dataset030_OnHarmonyT1w",
    "Dataset031_OnHarmonyT1w31",
])
def test_nnunet_raw_dataset_present(dataset_id):
    raw = Path(source_env(ENV_SH, "nnUNet_raw"))
    assert (raw / dataset_id).exists(), f"Missing in nnUNet_raw: {dataset_id}"

@pytest.mark.parametrize("dataset_id", [
    "Dataset030_OnHarmonyT1w",
    "Dataset031_OnHarmonyT1w31",
])
def test_nnunet_preprocessed_dataset_present(dataset_id):
    pre = Path(source_env(ENV_SH, "nnUNet_preprocessed"))
    assert (pre / dataset_id).exists(), f"Missing in nnUNet_preprocessed: {dataset_id}"

# ── conf / splits / checkpoints ──────────────────────────────────────────────

def test_conf_data_yaml_exists():
    p = DATASET_ROOT / "3_conf_on-harmony" / "data.yaml"
    assert p.exists(), f"Missing: {p}"

def test_splits_file_exists():
    p = DATASET_ROOT / "4_splits_on-harmony" / "on_harmony_split.json"
    assert p.exists(), f"Missing: {p}"

def test_splits_file_is_valid_json():
    import json
    p = DATASET_ROOT / "4_splits_on-harmony" / "on_harmony_split.json"
    assert p.exists()
    with open(p) as f:
        data = json.load(f)
    assert data, "splits file is empty"

def test_checkpoints_dir_exists():
    p = Path(source_env(ENV_SH, "CHECKPOINTS_DIR"))
    assert p.exists(), f"CHECKPOINTS_DIR not found: {p}"

def test_nnunet_results_dir_exists():
    p = Path(source_env(ENV_SH, "nnUNet_results"))
    assert p.exists(), f"nnUNet_results not found: {p}"

def test_nnunet_results_has_runs():
    p = Path(source_env(ENV_SH, "nnUNet_results")) / "runs"
    assert p.exists(), f"nnUNet_results/runs not found: {p}"

def test_results_subdirs_exist():
    results = DATASET_ROOT / "8_results_on-harmony"
    for sub in ("01_results", "02_nnUNet_results", "03_aggregated_results"):
        assert (results / sub).exists(), f"Missing results subdir: {sub}"
