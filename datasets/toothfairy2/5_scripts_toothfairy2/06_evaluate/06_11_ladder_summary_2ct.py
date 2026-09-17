#!/usr/bin/env python3
"""
toothfairy2 CBCT causal-ablation ladder with TWO independent CT sources — thin wrapper
over the shared engine's run_ladder_cross_dataset(), same as 06_05_ladder_summary.py.

WHY A SECOND LADDER SCRIPT RATHER THAN EDITING 06_05
----------------------------------------------------
06_05 remains the single-CT-source ladder (hanseg only), which is what the existing
results and the paper currently reference. This one adds PDDCA as a second, independent
CT cohort (RTOG 0522 vs HaN-Seg's Ljubljana) so the CT arm is 82 cases instead of 42,
and so the +AugLab rung's CT-specific −7 Dice cost can be checked for replication
outside a single institution. Keeping both means the before/after is inspectable rather
than silently overwritten.

⚠️⚠️ POOLING IS THE WHOLE POINT, AND IT IS NOT OPTIONAL.
The engine's DEFAULT cross-dataset pooling is a flat, case-count-weighted mean over
every evaluator item. With two CT sources and one MR source that would make the OOD
figure ~82 CT cases against 41 MR cases — i.e. adding a dataset would silently re-weight
the ladder toward CT. Since the +AugLab cost is CT-specific (~−7 on CT, ~0 on MR), that
would AMPLIFY the very rung under study with no method having changed. So this passes
`contrast_groups` + `prefixed_sources`, which switches the engine to equal weight per
GROUP (CT pooled as one stratum, MR as the other).

The contrast_groups tree is LOADED FROM THE CONFIG
(configs/toothfairy2_mandible_only_2ct.yaml) rather than restated here — the engine's
own docstring asks for exactly this, so the ladder's weighting can never silently
disagree with the headline table's.

Scoring is MANDIBLE-ONLY throughout (see the union-bug correction, 2026-09-17), so this
reads the `02_metrics_mandible_only` roots.

Usage: bash 06_11_ladder_summary_2ct.sh   (or: python 06_11_ladder_summary_2ct.py)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"))
from ladder_ood_common import run_ladder_cross_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "configs" / "toothfairy2_mandible_only_2ct.yaml"


def one_run_id(d: Path, pattern: str) -> str:
    """The single run dir under `d` matching `pattern`, minus its category prefix.
    Aborts on zero or several matches — a rung bound to the wrong run id renders as a
    plausible number with nothing erroring (the reason 06_05 has the same guard)."""
    hits = sorted(x.name for x in d.glob(pattern) if x.is_dir())
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} matches for {pattern!r} under {d} (need exactly 1): {hits}")
    name = hits[0]
    for prefix in ("nnUNet_", "auglab_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    groups = cfg["contrast_groups"]

    proj = DATASET_ROOT.parent.parent
    def sub(p: str) -> Path:
        return Path(p.replace("${PROJECT_ROOT}", str(proj)))

    # Same roots the headline 2-CT table reads, in the config's own order.
    srcs = [(sub(s["metrics_dir"]), s.get("column_prefix", "")) for s in cfg["sources"]]
    in_domain_source = srcs[0][0]
    ood_sources = [p for p, _ in srcs[1:]]
    # Rung run dirs are DISCOVERED under the normal ablations/ dir (and resolved under
    # each source via the "ablations/<run>" run_key), but OUTPUT goes to a separate
    # ablations_2ct/ — ablations_root is output-only in the engine. Writing both ladders
    # to one dir would silently overwrite the single-CT ladder, which is the baseline
    # this one is meant to be compared against.
    discover_root = in_domain_source / "ablations"
    ablations_root = in_domain_source / "ablations_2ct"

    baseline = one_run_id(in_domain_source, "*_baseline_2*")
    ours = one_run_id(in_domain_source, "*_auglabAug_v26_6_2_train050_val000_*")
    abl = discover_root
    r2 = one_run_id(abl, "*_baseline_kmeans_2*")
    r3 = one_run_id(abl, "*_baseline_kmeans_label_remap_2*")
    r4 = one_run_id(abl, "*_baseline_kmeans_label_remap_voronoi_*")
    r5 = one_run_id(abl, "*_v26_6_2_train050_val100_*")
    if ours.count("_val000_") != 1:
        raise SystemExit(f"OURS run id {ours!r} must contain '_val000_' exactly once")

    rungs = [
        ("baseline (floor)", "— (no augmentation at all)", baseline),
        ("+kmeans", "+ K-means intensity clustering", f"ablations/{r2}"),
        ("+label_remap", "+ label remap", f"ablations/{r3}"),
        ("+voronoi (noise fill)", "+ Voronoi sub-parcellation, noise fill", f"ablations/{r4}"),
        ("v26_6_2 (real fill)", "same partition, real-intensity fill (PALETTE alone)", f"ablations/{r5}"),
        ("+AugLab (val000)", "+ full AugLab recipe on top", ours),
        ("+AugLab (val100)", "+ 100%-synth validation", ours.replace("_val000_", "_val100_", 1)),
    ]

    run_ladder_cross_dataset(
        task_name="ToothFairy2 CBCT (2 CT sources, CT pooled)",
        contrast_label="cbct",
        ood_sources=ood_sources,
        in_domain_source=in_domain_source,
        ablations_root=ablations_root,
        rungs=rungs,
        contrast_groups=groups,
        prefixed_sources=srcs,
    )


if __name__ == "__main__":
    main()
