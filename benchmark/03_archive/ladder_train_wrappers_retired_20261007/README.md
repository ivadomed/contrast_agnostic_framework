# Retired ladder TRAIN wrappers (archived 2026-10-07)

Moved here so they can never be launched by accident (they train the wrong thing). Nothing was deleted; `MANIFEST.tsv` =
original path <TAB> archive path (55 files). Restore: `mv <archive path> <original path>`.

| what | count | why retired | replaced by |
|---|---|---|---|
| ladder rung 4 (+Voronoi, NOISE fill) wrappers using `transform_params_gpu_baseline_kmeans_label_remap_voronoi_spatialDA_train050.json` | 19 | the noise label step (`label_fill_noise`) refilled every remapped label as ONE noise level, erasing the Voronoi cells inside labels, while the real-fill rung 5 keeps them: the fill swap 4->5 also added Voronoi | generated `*_lblvor.sh` wrappers (`scripts/cluster/rung4_lblvor/make_wrappers.py`, config `..._voronoi_lblvor_spatialDA_train050.json`), retrained on Vulcan 2026-10-07 |
| ladder rung 5 "PALETTE alone" / rung-6 PV-alone `*_v26_6_2[_pv]_train050_val100*` and older `v26_6_2` val100 wrappers | 36 | ValSynth trainers: checkpoint_best chosen on SYNTHETIC validation images (most were already guarded by `ALLOW_VAL100_ALONE`) | `*_v26_6_2[_pv]_train050_val000.sh` (`scripts/cluster/rung5_val000/make_wrappers.py`) |

The old rung-4 config itself stays in AugLab (old runs record it) but is marked `_RETIRED_2026-10-07`, and
`RandomV26_6_2NoiseFillContrastGPU` raises on it unless `AUGLAB_ALLOW_LEGACY_NOISEFILL=1` (reproduction only).

Still-mentioning scripts (historical launchers/packs; re-running them needs the wrapper restored): `scripts/cluster/
{tamia_pack_rung6_pv,tamia_pack_rung5_retrain}.sh`, `scripts/cluster/rung5_val000/make_wrappers.py`, the TamIA pack/ladder
launchers `04_14/04_20/04_23/04_25/04_43/04_46/04_60` of toothfairy2/ispy2/pansegdata/brats/on-harmony, chaos
`04_10/04_25/04_26` launch-all scripts, brats `00_utils/t1c_runs.sh`. Generated wrappers name their source in a header comment only.
