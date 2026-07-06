# Killarney handoff — open-ms MS-lesion experiment (continue from Vulcan)

Paste this whole file as the opening prompt to Claude Code on Killarney. It has the full
context needed to (a) finish the port and (b) fix the live crash that is currently
blocking training.

---

## Who you are / the project
You're continuing an **MRI segmentation research project** (`contrast_agnostic_framework`,
branch `paul/v26_6_2`) just ported from the Vulcan cluster to **Killarney**. Read
`CLAUDE.md` at the repo root first — it documents the dataset-centric layout, the
`run_job` job-runner, the venv, and cluster etiquette. Note: CLAUDE.md was written for
**Vulcan**; you are now on **Killarney**, so GPU type / modules / account differ (see
"Killarney port tasks" below).

## The scientific goal (why open-ms exists)
The project compares an **image-driven** augmentation (**v26_6_2 + AugLab**: takes the real
image, K-means→Voronoi parcellation + affine contrast remap — texture-preserving) against a
**label-driven** one (**SynthSeg-EM**: fills each label region with per-region mean + white
noise — texture-blind). On existing datasets the two are only incrementally different because
those tasks are texture-poor. **open-ms** (brain MS lesion segmentation) was created to show
v26's advantage on a **texture/fine-detail-dominated** task: small focal lesions, tested
cross-contrast. Full background: `datasets/00_commun_scripts/00_04_analysis/README.md` and
the memory file `project_v26_vs_synthseg_differentiator`.

## open-ms setup (already done, verified structurally)
- Data: `open_ms_data` (Lesjak 2018, CC-BY), 30 patients, co-registered T1W/T2W/FLAIR (1mm,
  LPS) + binary consensus lesion masks. Pristine in `0_raw_open-ms/`; BIDS in `1_BIDS_*`.
- nnUNet **Dataset070_OpenMS_FLAIR**: train on **FLAIR** (single channel); cross-contrast
  **test on FLAIR / T2W / T1W** (held-out). Patch 128×160×112, batch 2, binary (bg+lesion).
- **Split (patient-level, no leakage):** 22 train-pool (4-fold CV) + 8 held-out test
  stratified by lesion burden. `4_splits_open-ms/{splits_final,test_cases,partition}.json`.
- Trainers (`5_scripts_open-ms/open_ms/trainers/`, thin wrappers mirroring chaos/on-harmony):
  `base` (do_split guard) → `baseline`, `auglab_default`, `auglab_valsynth`, `v26_6_2`.
  Registered via shim `02_nnunet/OpenMSTrainers.py` (must be copied into the venv — see below).
- Pipeline (all under `5_scripts_open-ms/`): `00_utils/00_01_bidsify.py` →
  `01_create_splits/01_01_create_splits.py` → `02_nnunet/02_00_convert.py` →
  `03_preprocess/03_00_preprocess.sh` → `04_train/04_0[1-6]_*.sh` →
  `05_predict/*` → `06_evaluate/06_01_evaluate_run.sh` + `06_02_lesionwise_analysis.py`.
  Full command list + method RUN_IDs: `datasets/open-ms/README.md`.
- **6 methods × 4 folds, 2000 epochs**: baseline, auglab_default, synthseg_EM, synthseg_noEM,
  v26_6_2 (alone), auglabAug_v26_6_2 (**ours**). The `validate_standard_dataset_structure.py`
  passes for open-ms. On a CPU node, `nnUNetTrainerOpenMSBaseline` trains fine (3 epochs,
  checkpoints) with `nnUNet_compile=0`.

## ⚠️ THE BLOCKER — training crashes at Epoch 0 on GPU (UNRESOLVED, fix this first)
On Vulcan, **all 24 folds FAILED at Epoch 0**, ~45s in, exit code **1** (a Python exception,
not OOM/timeout), **0 checkpoints**. It hits **every method including baseline**, so it's in
the common path, NOT the AugLab/synthesis code. The nnUNet `training_log` in each fold dir
stops right after `Epoch 0 / Current learning rate: 0.01` — i.e. it dies in the **first
training iteration**. The real traceback goes to the sbatch stdout, which the pipeline points
at a **node-local `/tmp` path that is wiped after the job** — so you never see it unless you
capture it to shared storage.

What was tried on Vulcan and RULED OUT / still open:
- First (wrong) hypothesis: `RuntimeError: can't start new thread` (blosc2 `.b2nd` reader
  spawns ~1 thread/core × N DA workers → cgroup PID limit). Fix applied: `BLOSC2_NTHREADS=1`
  in `datasets/00_commun_scripts/00_01_train/train_common.sh` + `DA_WORKERS` 12→8 in the
  open-ms train scripts. **This did NOT fix the GPU crash** — that thread error was a
  *login-node* artifact (login nodes are thread-starved and throw it even with 1 worker). Do
  NOT chase the thread error again.
- The CPU-node validation that "passed" used **`nnUNet_compile=0`**, but the real train
  scripts set **`nnUNet_compile=1`**. So the crash is GPU-specific AND likely tied to the
  first GPU forward — prime suspect **`torch.compile`**. BUT chaos/brats train fine with
  `nnUNet_compile=1`, so it may be **config-specific** (open-ms is the project's first
  *binary* 2-class dataset; patch 128×160×112) or a torch/env interaction — unconfirmed.

**Note:** the crash may behave DIFFERENTLY on Killarney — different GPU, CUDA/driver, and a
freshly-rebuilt venv (possibly a different torch/inductor build). It might not reproduce at
all (if it was a Vulcan torch.compile/CUDA quirk), or it might surface a different error.
So don't assume it's still broken — but don't assume it's fixed either: run the captured-log
debug fold below and confirm a GPU fold trains past Epoch 0 before launching all 24.

**Your first job: get the real traceback (or confirm it's gone), then fix.** On Killarney:
1. Rebuild the venv (below) and get one GPU.
2. Run ONE debug fold with the sbatch/`run_job` `--log` pointing at a **shared-filesystem**
   path (NOT /tmp), 2 epochs, so the traceback survives, e.g.:
   `run_job --name dbg --gpus 1 --slot 0 --wait --log $PWD/datasets/open-ms/8_results_open-ms/_dbg.log -- bash -c "... nnUNetv2_train 70 3d_fullres 0 -tr nnUNetTrainerOpenMSV26_6_2 -p nnUNetPlans -num_gpus 1"`
   with `NNUNET_NUM_EPOCHS=2 NNUNET_ITERS_PER_EPOCH=10 nnUNet_wandb_enabled=0` and the AugLab
   JSON env vars from `04_03_train_auglabAug_v26_6_2_train025_val100.sh`.
3. First test `nnUNet_compile=0`: if it trains, `torch.compile` was the cause → set
   `nnUNet_compile=0` in the six `04_0*_train_*.sh` scripts and you're done. If it still
   crashes, READ THE TRACEBACK (now captured) and fix the actual error.
4. Only after a fold is confirmed training past Epoch 0 **on a real GPU**, launch all 6
   methods (see README). Reuse the existing RUN_IDs (fresh dirs, no checkpoints) or generate
   new ones. Lesson learned the hard way: **never validate on CPU/compile-off and assume the
   GPU/compile-on path works.**

## Killarney port tasks (do before training)
1. **Rebuild the venv** — NOT copied from Vulcan (arch/CUDA/modules differ). Recipe:
   `scripts/job_runner/_setup_venv_oneshot.sh` (Vulcan-specific — adapt module versions to
   Killarney; `module spider python`, check the wheelhouse). Key deviations it encodes:
   `nnunetv2==2.7.0` + its underversioned deps must be pre-downloaded; `monai==1.5.2` with
   enumerated extras; `pip install -e sub-workspaces/auglab_workspace/AugLab`.
2. **Install the trainer shim** into the venv (nnU-Net only discovers trainers inside its own
   package dir): `cp datasets/open-ms/5_scripts_open-ms/02_nnunet/OpenMSTrainers.py .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/`
   Also restore the WandB logger patch + the other datasets' shims (see CLAUDE.md list) and
   `auglab_add_nnunettrainer -t nnUNetTrainerDAExt -t nnUNetTrainerTest`.
3. **Adapt the Slurm GPU request — PARAMETRISE the ONE `run_job_slurm.sh`, do NOT fork a
   `run_job_slurm_killarney.sh`.** `run_job` must stay the single backend-neutral interface
   (it auto-selects the slurm backend via `command -v sbatch`); a per-cluster copy would
   duplicate code and not be auto-selected. This is the ONLY thing the different GPU changes —
   the `nnUNetv2_train` commands are GPU-agnostic. The file already env-parametrises
   cpu/mem/account/time (`: "${RUN_JOB_CPUS_PER_GPU:=16}"` etc.); the gres GPU *type* is the
   only hardcoded bit. Do exactly this:
   - Add near the other defaults: `: "${RUN_JOB_GPU_TYPE:=l40s}"`
   - Change the gres line `echo "#SBATCH --gres=gpu:l40s:${gpus}"` to
     `echo "#SBATCH --gres=gpu${RUN_JOB_GPU_TYPE:+:$RUN_JOB_GPU_TYPE}:${gpus}"`
     (the `:+` makes an empty type produce plain `gpu:N`, which some clusters use).
   - On Killarney: `sinfo -o "%G" --Node | sort -u` for the type, then `export
     RUN_JOB_GPU_TYPE=<h100|…|empty>`; also set `RUN_JOB_CPUS_PER_GPU` / `RUN_JOB_MEM_PER_GPU`
     from `sinfo -o "%N %c %m %G"` (Vulcan's 16 / 110G are for its 64-core/4-GPU L40S nodes).
   **Wrong gres string → jobs never schedule.** Nothing else in the Slurm layer changes: no
   `--partition` is set (auto-routed); `--time`/`--account` already parameterised. This
   parametrisation is cluster-neutral and worth keeping (mergeable back for all clusters).
4. **Account**: `sacctmgr show assoc user=$USER` → set `RUN_JOB_ACCOUNT` (likely `def-jcohen`
   on Killarney; `aip-jcohen` is Vulcan's PAICE allocation).
5. If you skipped copying `2_nnUNet_open-ms/`, regenerate it: bidsify → create_splits →
   convert → `03_00_preprocess.sh` (a CPU job — use `run_job --gpus 0`).

## After training: the analysis
Predict cross-contrast + evaluate, then run the key metric:
`06_evaluate/06_02_lesionwise_analysis.py <OURS_RUN_ID> <SYNTHSEG_RUN_ID>` — lesion-wise
detection, size-stratified, per contrast. **Hypothesis:** v26 ≈ synthseg on in-domain FLAIR
bulk Dice, but increasingly superior on **T2W→T1W** and on **small-lesion detection / fewer
catastrophic misses** (the texture-advantage signature). If results are too good or too bad,
be suspicious and inspect predictions.

## Gotchas
- Everything runs through **`run_job`** (backend-neutral; auto-detects sbatch), never raw
  sbatch/set_slot. `salloc <opts> -- cmd` runs on the LOGIN node; use `srun` to reach the
  compute node.
- Data is many small files — stage/preprocess appropriately; keep the venv in $HOME/$PROJECT.
- open-ms is already **LPS** — no reorientation step needed.
