# Label boundary-cue importance — open-ms

This dataset's branch of a **cross-dataset** analysis; the results, method and reviewer-attack
audit are shared and live in one place rather than being duplicated three times:

- Findings + reviewer attacks: `datasets/00_commun_scripts/00_04_analysis/label_cue_importance/FINDINGS.md`
- Grounding/citations:         `datasets/00_commun_scripts/00_04_analysis/label_cue_importance/LITERATURE_REVIEW.md`
- Metric definitions + self-test: `.../label_cue_importance/cue_metrics.py --sanity`

Run this dataset's branch: `bash scripts/run_label_cues_*.sh`
(or all datasets at once on TamIA via `.../label_cue_importance/run_label_cues_pack.sh`, which is
what produced the reported numbers — one job packing all 8 branches over the node's 4 GPUs).

Outputs land in `outputs/data/` (one .npz per subject x label); aggregate with
`.../label_cue_importance/run_aggregate.sh` (dispatches through run_job — do not run it inline).
