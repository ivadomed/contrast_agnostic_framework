#!/usr/bin/env python3
"""
toothfairy2 CBCT causal-ablation ladder — THIN WRAPPER over the shared engine
datasets/00_commun_scripts/00_03_evaluate/ladder_ood_common.py.

Uses run_ladder_cross_dataset(), not run_ladder(): toothfairy2 has ONE training
modality, so there is no second in-house contrast to serve as OOD, and the OOD axis
comes from cross-dataset evaluators instead. That is the same shape atlas-liver-hcc
used, and it is why this file declares `ood_sources` rather than `ood_contrasts`.

Per CLAUDE.md's shared-layer rule, this file declares ONLY the rung list and the
source roots — no CSV loading, no metric math, no plotting. The shared engine exists
precisely because four hand-copied per-dataset ladders drifted apart and let a
transcription error into the paper's causal-ablation table; do not reintroduce a
local copy of that logic here.

RUN_IDs are read from the TamIA packs' own RUN_IDS.env (see
06_04_write_configs.py for why they are not hardcoded): they are timestamped at
launch, and reading them back from the pack is what guarantees the ladder scores the
runs that actually trained.

⚠️ TWO suite packs, for the same reason 06_04_write_configs.py needs both: the
6-method suite was launched as two node-packs, and EACH pack's RUN_IDS.env lists all
six ids regardless of which three it actually trained. The ladder's rung 1 (baseline)
comes from suite pack A; rungs 6/7 (OURS DualVal + its val100 mirror) come from pack
B. Reading rung 1 from pack B yields an id that was never trained, and the rung
silently renders as "—" — which is exactly how a ladder loses its anchor without
anything erroring.

The packs live on TamIA's scratch, but aggregation runs on Vulcan, where they
do not exist. `--from-metrics` recovers the same seven RUN_IDs from the
already-evaluated metrics dirs instead, so the ladder can be rebuilt wherever
the results live -- needed whenever the shared engine changes and every
ladder has to be regenerated. It is a recovery path, not a replacement: it
requires each rung to match EXACTLY one directory and aborts otherwise,
because "silently picked the wrong run id" is precisely the failure the pack
files were introduced to prevent.

Usage:
  bash 06_05_ladder_summary.sh <SUITE_A_PACK> <SUITE_B_PACK> <LADDER_PACK>
  bash 06_05_ladder_summary.sh --from-metrics
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder_cross_dataset  # noqa: E402


def read_env(pack: Path) -> dict[str, str]:
    f = pack / "RUN_IDS.env"
    if not f.exists():
        raise SystemExit(f"no RUN_IDS.env in {pack}")
    out = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def one_run_id(d: Path, pattern: str) -> str:
    """The single run dir under `d` matching `pattern`, minus its category prefix.

    Aborts on zero or several matches rather than picking one: this is the
    recovery path for a missing pack file, and a ladder rung bound to the wrong
    run id renders as a plausible number with nothing erroring."""
    hits = sorted(x.name for x in d.glob(pattern) if x.is_dir())
    if len(hits) != 1:
        raise SystemExit(f"--from-metrics: {len(hits)} matches for {pattern!r} under {d} "
                         f"(need exactly 1): {hits}")
    name = hits[0]
    for prefix in ("nnUNet_", "auglab_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-metrics", action="store_true",
                    help="recover RUN_IDs from the evaluated metrics dirs instead "
                         "of the TamIA packs (which only exist on TamIA)")
    ap.add_argument("--suite-pack-a", type=Path,
                    help="pack that trained baseline (ladder rung 1)")
    ap.add_argument("--suite-pack-b", type=Path,
                    help="pack that trained OURS (ladder rungs 6/7)")
    ap.add_argument("--ladder-pack", type=Path)
    a = ap.parse_args()
    if not a.from_metrics and not (a.suite_pack_a and a.suite_pack_b and a.ladder_pack):
        ap.error("give all three pack dirs, or --from-metrics")

    # Repo-relative default rather than a bare os.environ[...]: METRICS_ROOT points at
    # SCRATCH on TamIA but aggregation runs on Vulcan against the repo copy, and an unset
    # value would otherwise KeyError (or, in the shell wrappers, write to a literal
    # "${METRICS_ROOT}" dir — a bug this repo has already been bitten by).
    metrics_root = Path(os.environ.get(
        "METRICS_ROOT",
        DATASET_ROOT / "8_results_toothfairy2" / "02_metrics")) / "toothfairy2_model" / "cbct"
    ablations_root = metrics_root / "ablations"

    # Cross-dataset OOD evaluators. hanseg is the companion external test set
    # (head-and-neck CT + MR-T1); its metrics root holds one item column per
    # modality. Sources that are not yet predicted are skipped per rung by the
    # shared engine rather than treated as an error, so this can be run while the
    # cross-dataset half is still in flight.
    hanseg_metrics = Path(os.environ.get(
        "HANSEG_METRICS_ROOT",
        str(DATASET_ROOT.parent / "hanseg" / "8_results_hanseg" / "02_metrics"))) \
        / "toothfairy2_model" / "cbct"

    if a.from_metrics:
        abl = metrics_root / "ablations"
        baseline = one_run_id(metrics_root, "*_baseline_*")
        ours = one_run_id(metrics_root, "*_auglabAug_v26_6_2_train050_val000_*")
        l = {"R2_RUN_ID": one_run_id(abl, "*_baseline_kmeans_2*"),
             "R3_RUN_ID": one_run_id(abl, "*_baseline_kmeans_label_remap_2*"),
             "R4_RUN_ID": one_run_id(abl, "*_baseline_kmeans_label_remap_voronoi_*"),
             "R5_RUN_ID": one_run_id(abl, "*_v26_6_2_train050_val100_*")}
    else:
        sa = read_env(a.suite_pack_a)
        sb = read_env(a.suite_pack_b)
        l = read_env(a.ladder_pack)
        baseline = sa["BASELINE_RUN_ID"]  # rung 1 anchor — from pack A, see docstring
        ours = sb["OURS_RUN_ID"]          # rungs 6/7 — from pack B
    if ours.count("_val000_") != 1:
        raise SystemExit(f"OURS RUN_ID {ours!r} must contain '_val000_' exactly once")

    # (label, ingredient, run_key). run_key is prefixed with the metrics subdir for
    # the ladder-only rungs, which 06_01_evaluate_run.sh routes there via
    # METRICS_SUBDIR=ablations; the headline rungs (1, 6, 7) live flat in the parent.
    rungs = [
        ("baseline (floor)", "— (no augmentation at all)",
         baseline),
        ("+kmeans", "+ K-means intensity clustering",
         f"ablations/{l['R2_RUN_ID']}"),
        ("+label_remap", "+ label remap",
         f"ablations/{l['R3_RUN_ID']}"),
        ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill",
         f"ablations/{l['R4_RUN_ID']}"),
        ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)",
         f"ablations/{l['R5_RUN_ID']}"),
        ("+AugLab (val000)", "+ full AugLab recipe on top",
         ours),
        ("+AugLab (val100)", "+ 100%-synth validation",
         ours.replace("_val000_", "_val100_", 1)),
    ]

    run_ladder_cross_dataset(
        task_name="ToothFairy2 CBCT",
        contrast_label="cbct",
        ood_sources=[hanseg_metrics],
        in_domain_source=metrics_root,
        ablations_root=ablations_root,
        rungs=rungs,
    )


if __name__ == "__main__":
    main()
