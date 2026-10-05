"""pansegdata t2w-trained causal-ablation ladder (baseline -> +kmeans -> +label_remap -> +voronoi noise fill -> v26_6_2 real fill ->
+AugLab val000 -> +AugLab val100). Thin wrapper over the shared 00_03_evaluate/ladder_from_roster.py (rungs come from the roster
pin file, no timestamps here), which calls ladder_ood_common.run_ladder. OOD = the other training contrast (t1wce-trained -> t2w, t2w-trained -> t1wce).
Usage: .venv/bin/python 06_11_ladder_summary_t2w.py"""
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # benchmark/02_tasks/pancreas_disease/pansegdata
sys.path.insert(0, str(DATASET_ROOT.parents[2] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_from_roster import run  # noqa: E402

if __name__ == "__main__":
    run(dataset_root=DATASET_ROOT, model_type="pansegdata_model", contrast="t2w",
        ood_contrasts=["t1wce"], task_name="pansegdata T2W (own-model)")
