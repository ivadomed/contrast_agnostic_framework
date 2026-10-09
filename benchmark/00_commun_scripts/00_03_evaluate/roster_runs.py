"""Shared reader for the roster pin file written by 00_02_predict/run_all_predict_common.sh
(<predictions_root>/<model_type>/<training_contrast>/roster_run_ids.tsv, columns METHOD<TAB>CATEGORY<TAB>RUN_ID).
Used by write_configs_from_roster.py and ladder_from_roster.py so configs/ladders never hardcode a timestamp."""
from __future__ import annotations
from pathlib import Path

# Canonical headline order (the project notes "7-method suite"): the 6 usual methods + the DualVal val100 mirror of OURS.
HEADLINE_METHODS = [
    "baseline", "synthseg_noEM", "synthseg_EM", "auglab_default", "srcsm",
    "auglabAug_v26_6_2_train050_val000", "auglabAug_v26_6_2_train050_val100",
]
OURS = "auglabAug_v26_6_2_train050_val000"
# Causal-ablation ladder (project convention): rungs 2-5 live in <contrast>/ablations/, rungs 1/6/7 in the headline dir.
LADDER = [
    ("baseline (floor)", "— (no augmentation at all)", "baseline", False),
    ("+kmeans", "+ K-means intensity clustering", "baseline_kmeans", True),
    ("+label_remap", "+ label remap", "baseline_kmeans_label_remap", True),
    ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill", "baseline_kmeans_label_remap_voronoi", True),
    ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)", "v26_6_2_train050_val000", True),
    ("+AugLab (val000)", "+ full AugLab recipe on top", "auglabAug_v26_6_2_train050_val000", False),
    ("+AugLab (val100)", "+ 100%-synth validation", "auglabAug_v26_6_2_train050_val100", False),
]


def pin_path(dataset_root: Path, dataset_name: str, model_type: str, contrast: str) -> Path:
    return Path(dataset_root) / f"8_results_{dataset_name}" / "01_predictions" / model_type / contrast / "roster_run_ids.tsv"


def read_pins(path: Path) -> dict:
    """method -> (category, run_id)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the roster predict launcher first (it writes the pins)")
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            m, c, r = ln.split("\t")
            out[m] = (c, r)
    return out
