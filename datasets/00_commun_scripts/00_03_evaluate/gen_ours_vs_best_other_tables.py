#!/usr/bin/env python3
"""
Build the two "Ours vs best other method" headline tables (one per auglabAug_v26_6_2
DualVal mirror: train050_val000 and train050_val100), one row per training set
(dataset x modality), written to datasets/01_commun_results/.

Reuses the project's existing shared tools as-is (no changes to either):
  - significance_from_config.py already computes, per YAML config, a macroΔ (equal
    weight per contrast) + Holm-corrected paired significance test for "Ours vs every
    other run in the config", including an "ALL contrasts combined (in-domain + OOD)"
    section -- exactly the estimand this table needs (see that script's own docstring
    for the full statistical rationale: contrast-stratified sign-flip permutation test).
  - The existing cross_dataset_*.yaml configs (brats t1n/t2w, chaos t1in/t2spir,
    open-ms flair/t1w) already combine each dataset's own held-out contrasts WITH its
    cross-dataset eval-only counterpart (brats-ssa2024 / amos+sliver07 / mslesseg+ms3seg)
    -- exactly the "including cross-dataset contrasts" scope asked for. on-harmony has
    no cross-dataset eval-only counterpart, so its own multi-contrast config is used as-is.

"Best other method" = whichever of {baseline, synthseg_noEM, synthseg_EM, auglab_default,
srcsm} has the smallest macroΔ (Ours − competitor) on the ALL-contrasts dice comparison --
since Ours' own absolute mean is fixed across all 5 comparisons, ranking competitors by
smallest macroΔ is exactly equivalent to ranking them by largest own absolute dice, with
zero extra computation (no separate aggregate_from_config.py run needed). That same
competitor's hd95 row (from a second significance_from_config.py run) fills the hd95
columns, so both metrics in a row are always against the SAME competitor.

Existing cross_dataset_*.yaml configs list "v26_6_2 (alone)" (chaos/open-ms) alongside
the 5 named methods -- EXCLUDED here since the user's comparison set is explicitly the
5 listed methods, not that arm. Both auglabAug_v26_6_2 mirrors (val000/val100) are also
excluded from each other's competitor set: each table's Holm correction spans exactly the
5 named methods, matching the requested comparison scope exactly (confirmed with the user
rather than reusing the existing 7/8-run configs' correction scope as-is).

Usage:  .venv/bin/python gen_ours_vs_best_other_tables.py
Writes: datasets/01_commun_results/ours_vs_best_other_train050_val000.md
        datasets/01_commun_results/ours_vs_best_other_train050_val100.md
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIG_SCRIPT = Path(__file__).resolve().parent / "significance_from_config.py"
OUT_DIR = PROJECT_ROOT / "datasets" / "01_commun_results"

FIVE_METHODS = ["baseline", "synthseg_noEM", "synthseg_EM", "auglab_default", "srcsm"]

# Per training set: base config (cross_dataset where it exists, else the dataset's own
# multi-contrast config), the 5 named competitors' exact run ids (method -> run id),
# and the two Ours DualVal mirrors' run ids. METRICS_ROOT is only needed for the two
# on-harmony configs (their config uses ${METRICS_ROOT}, everything else uses
# ${PROJECT_ROOT} directly already).
TRAINING_SETS = [
    dict(label="brats2024-gli", modality="t1n",
         config="datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/cross_dataset_t1n_01_results.yaml",
         methods={
             "baseline": "brats2024-glioma_t1n_baseline_20260622_044535",
             "synthseg_noEM": "brats2024-glioma_t1n_synthseg_noEM_20260622_044535",
             "synthseg_EM": "brats2024-glioma_t1n_synthseg_EM_20260622_044535",
             "auglab_default": "brats2024-glioma_t1n_auglab_default_20260622_044535",
             "srcsm": "brats2024-glioma_t1n_srcsm_20260709_122015",
         },
         val000="brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303",
         val100="brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540",
         metrics_root=None),
    dict(label="brats2024-gli", modality="t2w",
         config="datasets/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/configs/cross_dataset_t2w_01_results.yaml",
         methods={
             "baseline": "brats2024-glioma_t2w_baseline_20260620_125115",
             "synthseg_noEM": "brats2024-glioma_t2w_synthseg_noEM_20260620_125442",
             "synthseg_EM": "brats2024-glioma_t2w_synthseg_EM_20260620_125354",
             "auglab_default": "brats2024-glioma_t2w_auglab_default_20260620_125306",
             "srcsm": "brats2024-glioma_t2w_srcsm_20260709_122045",
         },
         val000="brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540",
         val100="brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540",
         metrics_root=None),
    dict(label="chaos", modality="t1in",
         config="datasets/chaos/5_scripts_chaos/06_evaluate/configs/cross_dataset_t1in_01_results.yaml",
         methods={
             "baseline": "chaos_t1in_baseline_20260614_153230",
             "synthseg_noEM": "chaos_t1in_synthseg_noEM_train100_val000_20260611_120000",
             "synthseg_EM": "chaos_t1in_synthseg_EM_train100_val000_20260611_120000",
             "auglab_default": "chaos_t1in_auglab_default_20260611_120000",
             "srcsm": "chaos_t1in_srcsm_20260710_011817",
         },
         val000="chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615",
         val100="chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420",
         metrics_root=None),
    dict(label="chaos", modality="t2spir",
         config="datasets/chaos/5_scripts_chaos/06_evaluate/configs/cross_dataset_t2spir_01_results.yaml",
         methods={
             "baseline": "chaos_t2spir_baseline_20260620_111146",
             "synthseg_noEM": "chaos_t2spir_synthseg_noEM_20260620_112515",
             "synthseg_EM": "chaos_t2spir_synthseg_EM_20260620_112357",
             "auglab_default": "chaos_t2spir_auglab_default_20260620_112240",
             "srcsm": "chaos_t2spir_srcsm_20260709_121945",
         },
         val000="chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202",
         val100="chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053",
         metrics_root=None),
    dict(label="open-ms", modality="flair",
         config="datasets/open-ms/5_scripts_open-ms/06_evaluate/configs/cross_dataset_flair_01_results.yaml",
         methods={
             "baseline": "open-ms_flair_baseline_20260706_061243",
             "synthseg_noEM": "open-ms_flair_synthseg_noEM_train100_val000_20260706_061243",
             "synthseg_EM": "open-ms_flair_synthseg_EM_train100_val000_20260706_061243",
             "auglab_default": "open-ms_flair_auglab_default_20260706_061243",
             "srcsm": "open-ms_flair_srcsm_20260709_072043",
         },
         val000="open-ms_flair_auglabAug_v26_6_2_train050_val000_20260710_054455",
         val100="open-ms_flair_auglabAug_v26_6_2_train050_val100_20260716_095413",
         metrics_root=None),
    dict(label="open-ms", modality="t1w",
         config="datasets/open-ms/5_scripts_open-ms/06_evaluate/configs/cross_dataset_t1w_01_results.yaml",
         methods={
             "baseline": "open-ms_t1w_baseline_20260708_083441",
             "synthseg_noEM": "open-ms_t1w_synthseg_noEM_train100_val000_20260708_083541",
             "synthseg_EM": "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611",
             "auglab_default": "open-ms_t1w_auglab_default_20260708_083511",
             "srcsm": "open-ms_t1w_srcsm_20260709_075121",
         },
         val000="open-ms_t1w_auglabAug_v26_6_2_train050_val000_20260710_054525",
         val100="open-ms_t1w_auglabAug_v26_6_2_train050_val100_20260723_194125",
         metrics_root=None),
    dict(label="on-harmony", modality="T1w",
         config="datasets/on-harmony/5_scripts_on-harmony/06_evaluate/configs/on-harmony_T1w.yaml",
         methods={
             "baseline": "on-harmony_T1w_baseline_20260623_192811",
             "synthseg_noEM": "on-harmony_T1w_synthseg_noEM_20260623_192811",
             "synthseg_EM": "on-harmony_T1w_synthseg_EM_20260623_192811",
             "auglab_default": "on-harmony_T1w_auglab_default_20260623_192811",
             "srcsm": "on-harmony_T1w_srcsm_20260709_122115",
         },
         val000="on-harmony_T1w_auglabAug_v26_6_2_train050_val000_20260727_075205",
         val100="on-harmony_T1w_auglabAug_v26_6_2_train050_val100_20260727_075205",
         metrics_root=str(PROJECT_ROOT / "datasets/on-harmony/8_results_on-harmony/02_metrics")),
    dict(label="on-harmony", modality="T2w",
         config="datasets/on-harmony/5_scripts_on-harmony/06_evaluate/configs/on-harmony_T2w.yaml",
         methods={
             "baseline": "on-harmony_T2w_baseline_20260624_191152",
             "synthseg_noEM": "on-harmony_T2w_synthseg_noEM_20260624_191418",
             "synthseg_EM": "on-harmony_T2w_synthseg_EM_20260624_191329",
             "auglab_default": "on-harmony_T2w_auglab_default_20260625_143115",
             "srcsm": "on-harmony_T2w_srcsm_20260709_122145",
         },
         val000="on-harmony_T2w_auglabAug_v26_6_2_train050_val000_20260710_040443",
         val100="on-harmony_T2w_auglabAug_v26_6_2_train050_val100_20260727_075205",
         metrics_root=str(PROJECT_ROOT / "datasets/on-harmony/8_results_on-harmony/02_metrics")),
]


def build_filtered_config(base_config_path, methods, ref_run_id, metrics_root):
    """Load the base YAML, resolve ${...} vars, and keep only the 5 named methods + ref
    in `runs:` (dropping any 'v26_6_2 alone' arm and the OTHER Ours mirror), so the
    Holm correction in significance_from_config.py is scoped to exactly those 5."""
    cfg = yaml.safe_load(base_config_path.read_text())

    def expand(s):
        env = dict(os.environ)
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        if metrics_root:
            env["METRICS_ROOT"] = metrics_root
        return re.sub(r"\$\{(\w+)\}", lambda m: env.get(m.group(1), m.group(0)), s)

    if "sources" not in cfg:
        # Normalize to sources form so an explicit scratch output_dir can be set below --
        # single-source mode has no output_dir override, and would otherwise write into
        # the dataset's REAL metrics_dir (clobbering its actual significance/aggregate
        # output). Confirmed the hard way: this script's early drafts did exactly that.
        cfg["sources"] = [{"metrics_dir": cfg.pop("metrics_dir"), "column_prefix": "", "column_rename": {}}]
    for s in cfg["sources"]:
        s["metrics_dir"] = expand(s["metrics_dir"])

    cfg["runs"] = list(methods.values()) + [ref_run_id]
    return cfg


def run_significance(cfg, ref_run_id, metric, tmpdir):
    # ALWAYS write into tmpdir, never into any dataset's real metrics_dir/output_dir --
    # see build_filtered_config's note. This is the one and only output_dir this script
    # ever uses; the real per-dataset significance/aggregate files are never touched.
    cfg = dict(cfg)
    cfg["output_dir"] = tmpdir
    cfg_path = Path(tmpdir) / f"cfg_{metric}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    proc = subprocess.run(
        [sys.executable, str(SIG_SCRIPT), str(cfg_path), "--ref", ref_run_id, "--metric", metric],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"significance_from_config.py failed (metric={metric}):\n{proc.stderr}")
    return proc.stdout


def parse_all_contrasts_section(stdout):
    """Return {competitor_run_id: (macro_delta, holm_p)} from the
    '## ALL contrasts combined (in-domain + OOD)' section's table."""
    lines = stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## ALL contrasts combined")), None)
    if start is None:
        raise RuntimeError("ALL contrasts combined section not found in significance output")
    out = {}
    for line in lines[start:]:
        if line.startswith("## ") and "ALL contrasts combined" not in line:
            break
        if not line.startswith("| ") or line.startswith("| competitor") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        comp, macro_delta_s, holm_s = cells[0], cells[2], cells[5]
        macro_delta = float(macro_delta_s.replace("+", ""))
        holm_p = float(holm_s.replace("*", ""))
        out[comp] = (macro_delta, holm_p)
    return out


def fmt_p(p):
    return f"**{p:.4f}**" if p < 0.05 else f"{p:.4f}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ref_key, ref_label in (("val000", "train050_val000"), ("val100", "train050_val100")):
        rows = []
        for ts in TRAINING_SETS:
            base_config_path = PROJECT_ROOT / ts["config"]
            ref_run_id = ts[ref_key]
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = build_filtered_config(base_config_path, ts["methods"], ref_run_id, ts["metrics_root"])
                dice_out = run_significance(cfg, ref_run_id, "dice", tmpdir)
                hd95_out = run_significance(cfg, ref_run_id, "hd95", tmpdir)

            dice_table = parse_all_contrasts_section(dice_out)
            hd95_table = parse_all_contrasts_section(hd95_out)

            # Best other method = smallest macroΔ on dice (== highest own absolute dice,
            # since Ours' own mean is fixed across all 5 comparisons).
            best_other_run_id = min(dice_table, key=lambda r: dice_table[r][0])
            best_other_method = next(m for m, rid in ts["methods"].items() if rid == best_other_run_id)

            dice_delta, dice_p = dice_table[best_other_run_id]
            hd95_delta, hd95_p = hd95_table[best_other_run_id]
            rows.append((ts["label"], ts["modality"], best_other_method,
                         dice_delta, dice_p, hd95_delta, hd95_p))

        lines = [
            f"# Ours (auglabAug_v26_6_2, {ref_label}) vs best other method — all contrasts (incl. cross-dataset)",
            "",
            "Per training set, \"best other method\" = whichever of "
            f"{{{', '.join(FIVE_METHODS)}}} has the highest own average Dice over ALL "
            "contrasts (in-domain + cross-contrast + cross-dataset, equal weight per "
            "contrast — matches the summary table's `all` column). Both Dice and HD95 "
            "deltas/significance below are against that SAME method. Δ = Ours − "
            "competitor (Dice pts, higher better; HD95 mm, lower better so a NEGATIVE "
            "ΔHD95 favors Ours). Significance = Holm-corrected one-sided (Ours better) "
            "p from the contrast-stratified sign-flip permutation test in "
            "significance_from_config.py, Holm scope = exactly these 5 methods "
            "(not the full headline config's run list). **Bold** = p < 0.05.",
            "",
            "| dataset | modality | best other method | ΔDice | p (Dice) | ΔHD95 | p (HD95) |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, modality, best_other, dd, dp, hd, hp in rows:
            lines.append(f"| {label} | {modality} | {best_other} | {dd:+.2f} | {fmt_p(dp)} | "
                         f"{hd:+.2f} | {fmt_p(hp)} |")

        out_path = OUT_DIR / f"ours_vs_best_other_{ref_label}.md"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"[written] {out_path}")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
