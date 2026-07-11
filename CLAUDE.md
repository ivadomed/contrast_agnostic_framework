# Project: MRI Synthesis — Claude Context

## Cluster resource management (Vulcan / Slurm)

**Three machines are in active use** (`run_job` auto-detects the backend on each — see below, so pipeline scripts run unchanged everywhere):

| Machine | Type / backend | Role |
|---|---|---|
| **Vulcan** (AMII / Alliance) | Slurm cluster | **Primary.** All datasets + checkpoints live here; the main place to train/predict/eval. GitHub push works from here (done manually). Can be queue-congested when priority is low. |
| **Killarney** (Alliance) | Slurm cluster (same as Vulcan) | **Overflow GPU.** Same cluster type. Datasets staged there as of 2026-07-10: **chaos** (`1_BIDS_chaos/chaos-abdominal`, `2_nnUNet_chaos`) and **open-ms**. brats2024-glioma / on-harmony / others not yet staged — rsync from Vulcan first if needed there. Used when Vulcan priority runs low and we need more GPU. |
| **romane** (NeuroPoly lab) | `set_slot` workstation | **Limited-use lab box, only 4 GPUs.** The original dev machine; used for smaller/interactive GPU jobs. |

**SSH hosts + repo path per machine** (for rsync/ssh between them):
- **Vulcan**: `ssh vulcan.alliancecan.ca` — repo at `/project/aip-jcohen/paulh/mri_synthesis_project` (also reachable via the symlink `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`). `RUN_JOB_ACCOUNT=aip-jcohen` (the `run_job_slurm.sh` default).
- **Killarney**: `ssh killarney.alliancecan.ca` — repo at `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`. **`RUN_JOB_ACCOUNT=aip-jcohen` — same account name as Vulcan, do NOT override it.** (`datasets/open-ms/KILLARNEY_HANDOFF.md` speculated `def-jcohen` before this was confirmed on real hardware 2026-07-10 via a Slurm "Invalid account" error listing `aip-jcohen` as the only valid AIP account on this cluster — that doc is stale on this point.) See that same file for the still-relevant port notes: venv rebuild, GPU-type parametrisation, trainer shims.

This repo was originally developed on the `set_slot` workstation (romane) — that's why older scripts called `set_slot` directly; everything is now routed through `run_job` so it works on all three. Source of truth for Alliance cluster policy: `python_usage.html`, `running_jobs.html`, `storage_handling.html` in this directory (mirrored wiki pages) — re-read them if anything below looks stale.

**Pipeline scripts are backend-neutral — call `run_job`, not `set_slot`/`sbatch` directly.** `scripts/job_runner/run_job.sh` (sourced transitively by every dataset's `00_utils/env.sh`) exposes one `run_job` interface with two interchangeable backends — `run_job_set_slot.sh` (romane) and `run_job_slurm.sh` (Vulcan **and** Killarney — any machine where `command -v sbatch` succeeds). This means a pipeline script like `04_00_common.sh` runs unchanged on all three machines. See that directory's file headers for the full interface; in short:
```bash
run_job --name <id> --gpus <n> --slot <spec> --log <file> [--wait] -- <command...>
```
On Slurm, each call becomes its own independent `sbatch` job (true fire-and-forget — no `--partition`, GPU type `gpu:l40s` confirmed via `sinfo`, ~16 CPUs/~110G per GPU on an L40S node) and runs with `WANDB_MODE=offline` (compute nodes have no internet) — sync completed runs from the login node with `scripts/job_runner/wandb_sync.sh`. **Don't add new `set_slot` or raw `sbatch` calls to pipeline scripts** — route through `run_job` so the script keeps working on both backends.

Only `04_train/04_00_common.sh` (brats2024-glioma) and `00_utils/00_00_download_and_extract.sh` (amos) have been ported so far, as the canonical complex/trivial examples. The same two shapes repeat across the other ~85 files that still call `set_slot` directly (chaos's port of `04_00_common.sh`, on-harmony's simpler variant, every dataset's `05_predict`/`06_evaluate` scripts, the `contrast_manifold` analysis scripts, `scripts/experiments/run_full_pipeline_v27.sh`) — port them the same way once the brats2024-glioma path is confirmed working on real Vulcan hardware.

**`.venv/` status: rebuilt and working (2026-06-19).** It was originally copied wholesale from the old workstation (`.venv/bin/python` was a dangling symlink to that machine's `/usr/bin/python`) and has since been rebuilt clean: `module load python/3.11` (CVMFS-provided interpreter, not hardcoded), Python 3.11 (was 3.12 on the old box — don't hardcode either; use `.venv/lib/python3.*/site-packages/...` wherever this path shows up). Verified end-to-end with a real `run_job`-submitted GPU job (`torch.cuda.is_available()` → `True` on an L40S). `requirements.txt` is an unpinned wishlist, not a frozen list — installing from it alone does **not** reproduce a working environment; the full, tested recipe is `scripts/job_runner/_setup_venv_oneshot.sh` (run inside `salloc`/`sbatch`, never directly on the login node). Key deviations it encodes, so the next rebuild doesn't have to rediscover them:
- `nnunetv2==2.7.0` (the version this project is built against) isn't in the Alliance wheelhouse at all (only 2.5.1/2.1 are) — pre-download it with `pip download --no-deps` on the login node first, same for its two underversioned transitive deps (`acvl-utils>=0.2.6,<0.3`, `dynamic-network-architectures>=0.4.1,<0.5`) and `batchgeneratorsv2>=0.3.2` (wheelhouse only has 0.3.0).
- Plain `pip install "monai[all]"` silently backtracks to a broken `monai==0.1.0` if any single extra can't be satisfied locally — it does NOT error, it just degrades. Pin `monai==1.5.2` explicitly and install its extras as an enumerated list, excluding `clearml`/`nni`/`mlflow` (three different experiment-tracking/AutoML tools this project doesn't use — WandB is the actual tracker — and each hits an unsatisfiable-from-wheelhouse pin) and `itk`/`pyamg` (no python-3.11 build / absent entirely; unused — SimpleITK+nibabel already cover the imaging I/O monai would otherwise want itk for).

nnunetv2 also needs 4 files restored that exist ONLY as plain pip-installed files inside its own site-packages dir — there is no auto-copy step for these (unlike the `auglab_add_nnunettrainer` mechanism below), so a fresh `pip install nnunetv2` will NOT bring them back on its own (the setup script above already does this; listed here for when it needs doing by hand):
1. `cp src/nnunet/patches/nnunet_logger.py .venv/lib/python3.*/site-packages/nnunetv2/training/logging/nnunet_logger.py` — hand-patched WandB logger (`nnUNet_wandb_run_id` resume + `allow_val_change` for epoch-extension). Without it, WandB resume-by-id and raising `NNUNET_NUM_EPOCHS` on resume both break.
2. `cp datasets/brats2024-glioma/5_scripts_brats2024-glioma/02_nnunet/BraTS2024GliomaTrainers.py .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/`
3. `cp datasets/chaos/5_scripts_chaos/02_nnunet/CHAOSTrainers.py .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/`
4. `cp datasets/on-harmony/5_scripts_on-harmony/02_nnunet/OnHarmonyTrainers.py .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/`

(2–4 are "registration shims" — nnU-Net's `recursive_find_python_class` only searches inside its own `training/nnUNetTrainer/` dir, so each dataset's real trainer classes, which safely live in git under e.g. `datasets/brats2024-glioma/5_scripts_brats2024-glioma/brats2024_glioma/trainers/` already, need a thin shim there that imports them via `NNUNET_PROJECT_ROOT`. Without its shim, **no custom trainer for that dataset is discoverable at all** — training fails with "trainer not found".)

Also re-run for the AugLab-provided trainers (source of truth: the separate git repo at `sub-workspaces/auglab_workspace/AugLab/`, editable-installed): `pip install -e sub-workspaces/auglab_workspace/AugLab`, then `auglab_add_nnunettrainer -t nnUNetTrainerDAExt` and `-t nnUNetTrainerTest`.

**Lost in this rebuild:** `nnUNetTrainerV19SmokeTest.py` (was still used by `scripts/experiments/run_v19_smoke.sh`), `nnUNetTrainerV19GuidanceAug.py`, `nnUNetTrainerBraTSWandb.py`, `nnUNetTrainerBraTSGen19Wandb.py`, and an unreferenced `primus/` subpackage existed only inside the old `.venv` with no backup anywhere. They were flagged before the rebuild but deliberately not backed up (unclear status, no current callers found except the v19 smoke script) — the old `.venv` was deleted before anyone decided they were worth keeping, so they're gone. If `run_v19_smoke.sh` is ever needed again, `nnUNetTrainerV19SmokeTest` will need to be rewritten from scratch.

**Hard rules (shared cluster — violations risk sanctions for the whole group):**
- Everything goes through Slurm (`sbatch`/`salloc`, or `run_job` above). Login-node exception: compilation/checks under ~10 CPU-min and ~4 GB RAM only.
- Always set `--time`, `--mem`/`--mem-per-cpu`, `--cpus-per-task`, `--account` explicitly — defaults are tiny.
- Never set `--partition` (auto-routed). Never poll `squeue`/`sq` in a tight loop. Space out rapid `sbatch` calls or use array jobs (`--array=1-N`).
- Verify, don't guess: `module spider <name>` for versions, `sinfo -o "%G" --Node | sort -u` for GPU types, `diskusage_report` for quota.
- **`salloc <opts> -- <command>` runs `<command>` on the login node, not the allocated compute node** — only `srun` actually dispatches into the allocation. `salloc --gres=gpu:l40s:1 -- nvidia-smi` fails with "couldn't communicate with the NVIDIA driver" (no GPU on the login node); `salloc --gres=gpu:l40s:1 srun nvidia-smi` correctly sees it. This bites interactive testing, not `run_job`/`sbatch` — a submitted batch script already runs on the compute node with correct GPU binding, no `srun` wrapper needed there. Confirmed 2026-06-19 after this cost a debugging detour.

**Storage — what goes where:**

| Filesystem | Use for | Vulcan quota | Notes |
|---|---|---|---|
| `$HOME` | source code, job scripts, small config | 50 GB, no purge, backed up | small I/O only |
| `$PROJECT` (`/project/aip-jcohen/...`) | datasets, checkpoints, anything long-lived & shared | 5–12.5 TB (role-dependent), no purge, backed up | keep fairly static — frequent moves/renames burden tape backup |
| `$SCRATCH` | active job I/O, in-progress checkpoints | 5 TB/user, purged after 60 days, not backed up | transient only — copy anything worth keeping back to `$PROJECT` |
| `$SLURM_TMPDIR` | per-job node-local scratch | node-local disk, job lifetime only | best for many small files (e.g. unpacked NIfTI/DICOM from `0_raw_*`/`1_BIDS_*`) |

- MRI data = many small files. Don't dump raw/BIDS trees of tens of thousands of small files directly into `$SCRATCH`/`$PROJECT` — archive (`tar`) or stage into `$SLURM_TMPDIR` inside the job.
- No parallel writes to one file on shared filesystems without MPI-IO. Never `rm -rf` with `*` directly — move to `toDelete/`, verify, then delete.
- **Persistent venvs go in `$HOME`/`$PROJECT`, never `$SCRATCH`** — the purge policy can silently break a venv living there. For ephemeral per-job envs, build fresh on `$SLURM_TMPDIR` from a frozen `requirements.txt`.
- Long-running training: jobs are capped at ~168h/7 days per submission — use checkpoint + restart, preferably `--array=1-N%1` self-chaining. If self-resubmitting from inside a script, the continue-condition must be a *positive* "more work remains" test, never a stopping-condition that could fail to trigger.

---

## Project scope

This is an **MRI synthesis project** using domain randomization. The synthesizer is trained to generate synthetic data from T1w data, the goal is **domain randomization across all contrasts** (T1w, T2w, FLAIR, GRE, bold, dwi, epi, …) — there is **no single "target contrast"**. Synthetic images are intentional augmentations that explore far beyond any specific real scanner/contrast cluster.

The analysis pipeline (`contrast_manifold`) measures how synthetic images relate to real multi-scanner, multi-contrast MRI datasets via feature extraction → normalization → manifold analysis (PCA, UMAP, PRDC, Vendi).

---

## Python environment

```bash
.venv/bin/python   # always use this, not system python
```
On Vulcan: `module load python/X.Y` (check version with `module spider python`) *before* activating, and keep this venv in `$HOME` or `$PROJECT` — not `$SCRATCH`.

---

## Dataset structure

All datasets live under `datasets/`. We work in a dataset-centric manner. Every dataset follows the same 9-subdir standard:
```
datasets/
  validate_standard_dataset_structure.py   # run to check compliance
  <dataset>/
    0_raw_<dataset>/     # raw data as downloaded (DICOM, non-BIDS NIfTI, etc.)
    1_BIDS_<dataset>/    # BIDSified data (usually derived from 0_raw — both can coexist)
    2_nnUNet_<dataset>/raw/ + preprocessed/     # nnUNet converted data
    3_conf_<dataset>/data.yaml                  # Hydra data config
    4_splits_<dataset>/                         # train/val/test splits
    5_scripts_<dataset>/                        # pipeline scripts (see below)
    6_checkpoints_<dataset>/                    # model weights
    7_analysis_<dataset>/                       # manifold analysis (on-harmony only)
    8_results_<dataset>/                        # evaluation outputs
      01_predictions/  02_metrics/              # REQUIRED (enforced by validate_standard_dataset_structure.py)
      03_aggregated_results/                    # optional
      # on-harmony still uses the legacy 01_results/ + 02_nnUNet_results/ layout (migration pending)
    9_tests_<dataset>/                          # dataset-specific tests
```

Scripts inside `5_scripts_*/` follow a strict numbered convention:
```
00_utils/           # shared helpers; env.sh sets all paths (source this first)
01_create_splits/   01_NN_name.py/.sh
02_nnunet/
03_preprocess/
04_train/           04_00_common.sh = shared bash functions
05_predict/
06_evaluate/
```

---

## How experiments work — ALWAYS use the shared standardized scripts

**This is a hard project rule, not a style preference.** Every train / predict / evaluate / aggregate step goes through the **shared, standardized script layers** below. Do **not** hand-roll one-off `nnUNetv2_train`/`nnUNetv2_predict`/eval commands, and do **not** duplicate logic that already lives in a shared driver. The entire value of this project is that all datasets behave *identically* so the 6-method results are directly comparable across datasets and contrasts — bypassing, forking, or re-implementing these scripts silently breaks that comparability and is treated as a bug. If something is missing, **add it to the shared layer once (for all datasets)**, don't inline it in one place.

### Three script homes (know which is which)

Scripts live in exactly three places — put new code in the right one:
- **`src/`** — PALETTE method source code only (the contrast transform / model). No pipeline glue.
- **`datasets/00_commun_scripts/`** — the shared **dataset-pipeline** layer: train/predict/evaluate/aggregate/significance that operates on the 9-dir datasets (`00_00_utils` libs, `00_01_train`, `00_02_predict`, `00_03_evaluate` incl. `aggregate_from_config.py` + `significance_from_config.py`, `00_04_analysis`). **All eval/aggregate/significance lives here** (canonical); dataset `5_scripts_*` are thin wrappers over it.
- **`scripts/`** — cross-cutting **infra + research tooling** that is *not* dataset-pipeline: `job_runner/` (the `run_job` backbone — sourced by `common_env.sh`, so it sits *below* `00_commun_scripts`; do not move it), venv setup, `wandb_sync`, the generator/segmenter training entry points, `utils/` (SynthSeg runner + BIDS→nnUNet converters), and `experiments/` runners. `scripts/evaluate/run_significance_all.sh` is the one cross-dataset convenience driver that stays here (it calls the canonical `00_03_evaluate/significance_from_config.py`).

### An "experiment" = a fixed 6-method suite trained on ONE modality

"Start our 6 usual experiments on `<dataset>` `<modality>`" means: train these **exact 6 methods** on that single modality (on-harmony→T1w, open-ms→FLAIR, chaos→T1in/T2spir, brats→t1n/t2w). AugLab configs are under `sub-workspaces/auglab_workspace/AugLab/auglab/configs/`:

| # | method | category | AugLab config (or nnUNet) |
|---|---|---|---|
| 1 | **baseline** | nnUNet | — (nnU-Net default aug) |
| 2 | **auglab_default** | auglab | `transform_params_gpu_default01-23.json` |
| 3 | **synthseg_noEM** | auglab | `transform_params_gpu_default01-23_Synthseg.json` |
| 4 | **synthseg_EM** | auglab | `transform_params_gpu_default01-23_Synthseg_EM.json` |
| 5 | **v26_6_2 (ours, alone)** | nnUNet | — (v26_6_2 transform alone; **train050_val100**) |
| 6 | **auglabAug_v26_6_2 (OURS)** | auglab | `transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train025.json` (**train025_val100**) |

The `trainXXX_valYYY` suffix (in the METHOD id / RUN_ID) encodes the synth-augmentation probabilities baked into the config — keep them as above (v26-alone = train050/val100; OURS = train025/val100). Ablation arms (e.g. `..._noisefill`, `..._no_voronoi`) are **extra** arms on top of these 6, never replacements.

**FOLD POLICY — 3 folds only, permanently.** Always train **folds 0 1 2 only**. We used to train 4 folds but **from 2026-07-09 onward we never train fold 3 again** (3-fold ≡ 4-fold conclusions; fold 3 is wasted GPU on a congested cluster). If a dataset already has a 4-fold split, **reuse it and just train the first 3 folds** — do **not** create a new split and do **not** train fold 3. The shared drivers now **default** to 3 folds (not merely cap): `00_01_train/train_common.sh` `TRAIN_FOLDS` and `00_02_predict/predict_common.sh` `PREDICT_FOLDS` both default to `"0 1 2"`, `run_all_train_common.sh` exports `TRAIN_FOLDS="0 1 2"`, and `eval_folds.py` `EVAL_FOLDS` caps at 0-2 — so per-method wrappers no longer need to set folds explicitly. Only override (e.g. `TRAIN_FOLDS="0 1 2 3"`) for a deliberate one-off or to predict a legacy 4-fold model.

**EPOCH POLICY — respect each dataset's usual epoch count.** Don't change these without a deliberate reason — they keep the 6-method suite comparable within and across runs of a dataset: **brats2024-glioma = 2500**, **on-harmony = 2000**, **open-ms = 2000**, **chaos = 200**.

**How to launch all 6 (one command):** a per-modality launcher lists the 6 per-method wrapper paths in a `METHOD_SCRIPTS` array and sources the shared runner `datasets/00_commun_scripts/00_01_train/run_all_train_common.sh` — which caps folds at `0 1 2`, runs them in order, and supports `--start-from <method>` to resume a suite. Canonical example: `datasets/open-ms/5_scripts_open-ms/04_train/04_12_run_all_flair.sh` (→ `bash 04_12_run_all_flair.sh`). Each method is still its own thin `04_XX_train_<modality>_<method>.sh` wrapper (on-harmony T1w = `04_07`…`04_12`). For a **new** dataset/modality: create the 6 per-method wrappers (each ~5 lines: set METHOD/TRAINER/config, source `04_00_common.sh`) + one per-modality run-all launcher. *(The old `04_06_run_all_training.sh` is dead — ignore it.)*

### AugLab is a SEPARATE, config-driven repo (sync it per-machine)

`sub-workspaces/auglab_workspace/AugLab` is its **own git repo** (editable-installed; it shows as untracked `?? sub-workspaces/` in the main repo — its history does **not** ride the main-repo git). AugLab augmentations are selected entirely by **JSON config** (the `transform_params_gpu_*.json` files above); our nnU-Net trainers (the `nnUNetTrainer*AugLab*` classes) basically wrap/call AugLab's trainer/transform machinery. Two consequences:
- To change augmentation behaviour you usually add/edit a **config**, not code.
- When you *do* change AugLab **code** (a new transform, a param), that change must be **synced to each machine independently** — it will NOT arrive via a main-repo `git pull`. (This bit us: a `palette_noisefill.py` fix existed on one machine but not another even though the main-repo training scripts had merged.)

*Aside — external SynthSeg vs "synthseg" augmentation:* the SynthSeg-style **augmentation** methods (`synthseg_EM`/`noEM`) are just AugLab configs (above) — they do **not** need the external SynthSeg model. The external SynthSeg tool (`scripts/utils/run_synthseg_predict.py`, needs a `SynthSeg/` repo + `.h5` model) is used **only** to generate anatomical **parcellation labels** for a dataset that lacks them — a separate concern from training.

### Train on ONE modality → predict / evaluate / aggregate on ALL modalities

Models are contrast-agnostic, so evaluation is **cross-contrast**: a model trained on one modality is tested on **held-out cases across every modality of the dataset** (on-harmony: T1w/T2w/FLAIR/GRE/dwi/bold/epi; open-ms: FLAIR/T1w/T2w; …). Dice + HD95 are computed **per test-contrast, per fold**, then aggregated and significance-tested across the full 6-method table. **This cross-contrast generalization IS the headline result** — never evaluate only on the training modality.

### The shared layer (canonical — reuse, don't reinvent): `datasets/00_commun_scripts/`

- `00_00_utils/` — shared libs: `common_env.sh`, `splits_lib.py`, `orient.py`, `fov.py`, `eval_metrics.py`/`eval_folds.py`/`eval_aggregate.py`, `stat_tests.py`, `nnunet_convert_lib.py`.
- `00_01_train/train_common.sh` — training driver (fold fan-out via `run_job`; honours optional `TRAIN_FOLDS`, default `"0 1 2 3"`).
- `00_02_predict/predict_common.sh` — prediction driver (own- and cross-model modes).
- `00_03_evaluate/` — `evaluate.py` (Dice+HD95), `summarize_fold.py`, `aggregate_results.py`, `significance_from_config.py` (paired Wilcoxon, Holm-Bonferroni).
- `00_04_analysis/` — `texture_advantage.py`, `mechanism_illustration.py`.

### The 3-tier wrapper pattern — follow it for every new method/dataset

```
per-method wrapper (~5 lines)       dataset shim                         shared driver (the real work)
04_0X_train_<method>.sh        →    04_00_common.sh                 →    00_01_train/train_common.sh
  sets METHOD, TRAINER,               sources env.sh, sets                 (fold loop, run_job)
  DATASET_ID, config JSONs            dataset-specific defaults
05_XX_predict_<method>.sh      →    05_01_predict_common.sh         →    00_02_predict/predict_common.sh
  sets METHOD/TRAINER/CATEGORY        sources env.sh, sets PREDICT_*       (predict, run_job)
06_XX_evaluate_*.sh            →    (dataset eval + configs/*.yaml)  →    00_03_evaluate/* (eval/agg/significance)
```

A new **method** is a ~5-line wrapper (set METHOD/TRAINER/CATEGORY/config, source the dataset common). A new **dataset** gets the numbered `5_scripts_*` skeleton whose `*_common.sh` shims source the shared drivers. **If you catch yourself writing predict/eval/aggregate/significance logic inline, stop — it already exists in `00_commun_scripts` (use it), or belongs there (add it once, for all datasets).**

### Standardized output layout (so aggregation/significance/plots stay uniform)

```
8_results_<ds>/01_predictions/<model>/<train_contrast>/<nnUNet|auglab>/<RUN_ID>/fold{k}/<test_contrast>/
8_results_<ds>/02_metrics/<model>/<train_contrast>/<category>_<RUN_ID>/fold{k}/eval_all.csv
   → aggregated: results_summary.md, results_significance.md, results_heatmap_{dice,hd95}.png
```

Config-driven aggregation + significance consume YAML configs (`06_evaluate/configs/*.yaml`) so a whole dataset's 6-method table is summarized and tested with one command — **use these, do not compute ad-hoc tables/p-values.**

---

## Analysis pipelines (paper: the two mechanism "pillars")

The paper explains *why* our method works via two input-space analyses on the **generated augmentation
volumes** (same volumes + same 4 methods — palette, synthseg_em, synthseg_noem, auglab_default — feed both):

- **Pillar 1 — texture preservation.** `datasets/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/`.
  Census / rank-transform |correlation| (contrast+inversion-invariant) + NMI, source vs each generated
  volume, per anatomical ROI (on-harmony: 31 SynthSeg classes). Grounding: that dir's `LITERATURE_REVIEW.md`.
- **Pillar 2 — histogram-manifold coverage.** `.../histogram_coverage_lvl_1/`. Per-(scanner×contrast)-group
  **Coverage** (Naeem 2020) macro-averaged + **Vendi** (Friedman & Dieng 2023), on 31-class regional
  histograms, with PCA + UMAP plots. Grounding: that dir's `GROUNDING_AUDIT.md` + `LITERATURE_REVIEW.md`.
  Metrics computed by `compute_coverage_metrics.py` (fast precomputed-distance bootstrap), plotted by
  `plot_coverage_metrics.py` / `plot_manifold.py`.

**open-ms variants** of both pillars use only the labels open-ms provides — a 2-region **[lesion, overall]**
feature instead of 31 anatomical ROIs (no SynthSeg parcellation; brainmask deliberately unused). See
`datasets/open-ms/7_analysis_open-ms/histogram_coverage_lvl_1/README.md`.

Vendi API: `from vendi_score import vendi; vendi.score_X(X)` (no `model` arg). Metric deps: `prdc`,
`vendi-score`, `umap-learn` (in `requirements.txt`).

**Previous experiment (deprecated — do NOT build on it):** `.../contrast_manifold/` was an earlier
manifold-coverage exploration (feature extraction → PCA/UMAP/PRDC/Vendi with a custom IND/OOD split and
hand-rolled spread/hull/recall@Nx metrics, feature types like `regional_hist_64`/`hog3d_512`/CURIA). Its
numbers were deemed unreliable and it is **superseded by Pillar 2** above. Kept only for history; not
currently relevant.

---

## Common gotchas

- Whenever we run anything, we want to run a script with a simple bash command (which dispatches through `run_job` on this cluster — see "Cluster resource management" above). The script likely already exists in `5_scripts_*/` and if not, it should be added there — the structure should naturally guide you. We want to avoid running Python scripts directly from the command line without a proper script wrapper. Running existing scripts is good for consistency, and creating new scripts in the right place is good for organization and future reproducibility; it also ensures we don't debug the same thing repeatedly.
