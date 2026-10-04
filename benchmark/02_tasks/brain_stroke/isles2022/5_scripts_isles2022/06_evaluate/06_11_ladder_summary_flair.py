"""isles2022 flair-trained causal-ablation ladder (baseline -> +kmeans -> +label_remap -> +voronoi noise fill -> v26_6_2 real fill ->
+AugLab val000 -> +AugLab val100). Thin wrapper over the shared 00_03_evaluate/ladder_from_roster.py (rungs come from the roster
pin file, no timestamps here), which calls ladder_ood_common.run_ladder.
OOD = every non-training test contrast, equal weight per contrast: dwi, adc. NOTE ADC is derived from DWI, so the
adc column is a held-out *rendering* of the DWI signal, not independent information; flair is the
independent contrast. (Open decision for Paul: pool adc into OOD, or report flair-only OOD.)
Usage: .venv/bin/python 06_11_ladder_summary_flair.py"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/brain_stroke/isles2022
sys.path.insert(0, str(DATASET_ROOT.parents[2] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_from_roster import run  # noqa: E402

if __name__ == "__main__":
    run(dataset_root=DATASET_ROOT, model_type="isles2022_model", contrast="flair",
        ood_contrasts=["dwi", "adc"], task_name="isles2022 FLAIR (own-model)")
