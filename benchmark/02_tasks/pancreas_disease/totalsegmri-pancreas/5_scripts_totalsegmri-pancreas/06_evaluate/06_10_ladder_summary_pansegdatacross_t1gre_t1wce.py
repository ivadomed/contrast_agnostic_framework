"""totalsegmri-pancreas cross-dataset causal-ablation ladder: item `t1gre` scored with the pansegdata t1wce-TRAINED models (rungs from the SOURCE roster pins, no timestamps).
An item in the SAME contrast as the source's training contrast (t1wce) is cross-DATASET evidence only (supplementary; never pooled into an OOD bucket); a different contrast is genuine
held-out-contrast evidence. Pool into the source task's own ladder only by TRUE contrast (see the skill add-eval-companion-end-to-end). Usage: .venv/bin/python 06_10_ladder_summary_pansegdatacross_t1gre_t1wce.py"""
import sys
from pathlib import Path
DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parents[2] / "00_commun_scripts" / "00_03_evaluate"))
from ladder_from_roster import run_companion  # noqa: E402
if __name__ == "__main__":
    run_companion(companion_root=DATASET_ROOT, source_root=DATASET_ROOT.parents[1] / "pancreas_disease" / "pansegdata",
                  model_type="pansegdata_model", contrast="t1wce", item="t1gre", task_name="totalsegmri-pancreas t1gre (pansegdata T1WCE-trained)")
