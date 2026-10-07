# Retired ladder rung-5 / rung-6 val100 runs (archived 2026-10-07)

**What:** the "PALETTE alone" ladder rung 5 (`*_v26_6_2_train050_val100_*`) and rung-6 PV-alone (`*_v26_6_2_pv_train050_val100_*`) runs that
selected `checkpoint_best` on SYNTHETIC validation (ValSynth trainers). They were replaced by the val000 retrains of 2026-10-05/07
(`*_v26_6_2[_pv]_train050_val000_*`, run table `$SCRATCH/rung5_val000/run_ids.txt`), which every ladder and `ladder_pv_branch.yaml` now use.
Also here: 41 on-harmony `*_best.stale_*` metrics dirs left by two overlapping evaluation controllers on 2026-10-06 (the clean
re-evaluation replaced them).

**Layout:** paths mirror `benchmark/02_tasks/`: `<task>/<dataset>/8_results_<dataset>/...`. `MANIFEST.tsv` = original path <TAB> archive path,
one line per moved directory (158 dirs, 44 GB: model + predictions + metrics of 43 runs, plus the 41 stale dirs). Nothing was deleted.
Restore a run: `mv <archive path> <original path>` (same filesystem).

**Deliberately NOT archived** (still read by live configs/analyses, left in place):
chaos_t1in 20260615_213615 (HP-sweep configs), open-ms flair 20260706_061243 / t1w 20260708_083641 (cross_dataset_*_01_results.yaml rows),
brats t1n 20260730_200711 / t2w 20260620_125217 / t2f 20260917_113412 (texture_analysis_lvl_1 intervention scripts), toothfairy2 20260908_013028
(toothfairy2_cbct_ladder.yaml).

**Still-mentioning scripts (historical, not readers of tables/ladders):** the guarded breast 05_17/05_21 val100 predict wrappers, old chaos/amos/
sliver07 predict-all/evaluate-all batch scripts, brats 05_16, `scripts/cluster/{tamia_rung6_pv_runs,vulcan_rung6_pv_breast_cross,fetch_rung6_pv_metrics}.sh`,
`scripts/cluster/rung5_val000/post_run.sh` (its old-run audit comparisons) and pansegdata `_launch/RUN_IDS_vulcan_rungs.tsv`. Re-running any of them
needs the run restored first.
