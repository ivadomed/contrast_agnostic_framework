# Project: MRI Synthesis — Claude Context

## Cluster resource management (Vulcan / Slurm)

**Three machines are in active use** (`run_job` auto-detects the backend on each — see below, so pipeline scripts run unchanged everywhere):

| Machine | Type / backend | Role |
|---|---|---|
| **Vulcan** (AMII / Alliance) | Slurm cluster | **Primary.** All datasets + checkpoints live here; the main place to train/predict/eval. GitHub push works from here (done manually). Can be queue-congested when priority is low. |
| **Killarney** (Alliance) | Slurm cluster (same as Vulcan) | **Overflow GPU.** Same cluster type. Datasets staged there as of 2026-07-10: **chaos** (`1_BIDS_chaos/chaos-abdominal`, `2_nnUNet_chaos`) and **open-ms**. brats2024-glioma / on-harmony / others not yet staged — rsync from Vulcan first if needed there. Used when Vulcan priority runs low and we need more GPU. |
| **romane** (NeuroPoly lab) | `set_slot` workstation | **Limited-use lab box, only 4 GPUs.** The original dev machine; used for smaller/interactive GPU jobs. |
| **TamIA** (Alliance) | Slurm cluster (same family as Vulcan/Killarney) | **Newest, biggest-GPU cluster — added 2026-07-25.** Whole-node H100/H200 allocations (see below), much more GPU per node than Vulcan/Killarney's L40S. **Use these GPUs properly — a coworker was already warned by Alliance staff about under-utilizing allocated GPUs on this account, and TamIA's whole-node model makes idle GPUs especially visible.** Currently only reachable via a relay through Vulcan (see below); as of 2026-08-02 five datasets (`brats2024-glioma`, `chaos`, `on-harmony`, `open-ms`, `picai-prostate`) are staged there, and **all data currently lives on `$SCRATCH` only** (nothing in `$PROJECT` yet) — see the TamIA-specific subsection for what that means operationally. |

**SSH hosts + repo path per machine** (for rsync/ssh between them):
- **Vulcan**: `ssh vulcan.alliancecan.ca` — repo at `/project/aip-jcohen/paulh/mri_synthesis_project` (also reachable via the symlink `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`). `RUN_JOB_ACCOUNT=aip-jcohen` (the `run_job_slurm.sh` default).
- **Killarney**: `ssh killarney.alliancecan.ca` — repo at `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`. **`RUN_JOB_ACCOUNT=aip-jcohen` — same account name as Vulcan, do NOT override it.** (`datasets/open-ms/KILLARNEY_HANDOFF.md` speculated `def-jcohen` before this was confirmed on real hardware 2026-07-10 via a Slurm "Invalid account" error listing `aip-jcohen` as the only valid AIP account on this cluster — that doc is stale on this point.) See that same file for the still-relevant port notes: venv rebuild, GPU-type parametrisation, trainer shims.
- **TamIA**: not reachable directly by name from outside Canadian research networks — the login-node firewall drops non-Canadian traffic at the TCP level, and even from Canada the daemon that runs these sessions cannot complete Alliance's mandatory Duo two-factor login on every fresh connection. **Reach TamIA only by relaying through Vulcan**: `ssh vulcan.alliancecan.ca` then `ssh tamia.alliancecan.ca` from there (a ControlMaster relay block is already configured in Vulcan's own `~/.ssh/config` for this). Repo at `/project/aip-jcohen/paulh/mri_synthesis_project`, cloned directly from GitHub (not copied from Vulcan — Vulcan's `.git` is bloated to 25 GB from historical churn; a fresh GitHub clone is ~100 MB and lands on the same commit). `RUN_JOB_ACCOUNT=aip-jcohen` (same as the other two).

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

## TamIA (newest cluster — bigger GPUs, different layout, whole-node allocation)

**Access:** TamIA cannot be reached directly from this environment (login-node firewall + Duo 2FA on every fresh connection). Always relay through Vulcan:
```bash
ssh vulcan.alliancecan.ca
ssh tamia.alliancecan.ca   # from a vulcan login shell, using the relay ControlMaster set up in vulcan's ~/.ssh/config
```
Wrap remote commands in `bash -lc "..."` — a bare `ssh tamia.alliancecan.ca 'cmd'` gets a non-login shell without the CVMFS module PATH. Deeply nested quoting through the relay mangles reliably (`sed`/`awk`/`$(...)` get eaten) — write scripts to a local file and pipe them over (`tar cf - -C /tmp s.sh | ssh tamia.alliancecan.ca "tar xf - -C /tmp && bash /tmp/s.sh"`) rather than inlining complex shell.

**GPUs — bigger than Vulcan/Killarney, whole-node only:**

| Node class | GPUs/node | CPUs | RAM | Use |
|---|---|---|---|---|
| `h100:4` (the bulk of the cluster) | 4× H100 | 48 | 500,000 MB | **Default — use this class** |
| `h200:8` | 8× H200 | 64 | 1,000,000 MB | Bigger node class — do **not** request unless explicitly asked |

Partitions are `gpubase_bynode_b1/b2/b3` — **allocation is whole-node** (no per-GPU partition), so a submitted job gets all 4 (or 8) GPUs on the node whether it uses them or not. `run_job` env overrides for the default h100 class: `RUN_JOB_GPU_TYPE=h100`, `RUN_JOB_CPUS_PER_GPU=12` (48/4), **`RUN_JOB_MEM_PER_GPU=115G`**. ⚠️ That variable is parsed in **GiB**, not MB — `run_job_slurm.sh` does `mem="$(( ${RUN_JOB_MEM_PER_GPU%G} * gpus ))G"`, stripping a trailing `G` and re-appending one. Passing the raw MB figure (`125000`) requests 500000 **G** and sbatch rejects the job with "Memory specification can not be satisfied" (cost a lost submission 2026-08-02). Leave headroom under the node's 500000 MB: 115G × 4 = 460G. As always, these are env-var overrides only — `run_job_slurm.sh` itself is never forked per cluster. Do **not** pass `--partition` explicitly on TamIA (rejected even for partitions `sinfo` lists as valid) — submit without it and let the scheduler route.

**Use the GPUs properly — this is now a hard requirement, not a nice-to-have.** Because allocation is whole-node, a job that only exercises 1 of 4 (or 1 of 8) GPUs wastes the other 3 (or 7) for the whole wall-clock duration — highly visible to Alliance staff, and **a coworker on this account has already been warned about under-utilizing allocated GPUs.** Concretely, for the H100 nodes:
- Node-pack placement is explicit when it needs to be: `run_job_pack_submit.sh` spreads recorded folds round-robin (`i%4`) by default, but **`PACK_GPU_MAP`** (one GPU index per `index.tsv` row) overrides that. Use it whenever folds cost very different amounts — notably **srcsm, which is ~3x slower per epoch than every other method**: round-robin will pair a srcsm fold with a second fold and make it the straggler that holds the whole node *and its whole dependency chain* open. Give srcsm folds a GPU to themselves and double up the cheap folds instead (canonical example: `picai-prostate/.../04_21_tamia_pack_launch_all.sh`).
- Before choosing a layout, **run the sizing probe** — a few epochs of the heaviest and the slowest method side by side on one real node, reporting per-fold VRAM + epoch time (`picai-prostate/.../04_22_tamia_size_probe.sh` is the reusable pattern). CLAUDE.md already required measuring rather than assuming; that script is what does it.
- Prefer packing **multiple folds onto the same node's GPUs** in one job rather than 1 fold = 1 whole node. With 4 GPUs/node, that means either 4 folds in parallel per job, or — since the current fold policy trains only folds 0/1/2 — 1 GPU sits idle unless a second task (a different method, modality, or ablation arm) is packed alongside.
- The bigger H100 memory headroom vs Vulcan's L40S (each H100 has more VRAM) also means **larger batch sizes are affordable** — this changes the training config (not the shared script), so treat it as a per-cluster override the same way GPU type/CPU/mem are already overridden, not a hand-edit of `train_common.sh`.
- Before changing batch size or fold-packing, size actual per-fold GPU-memory and utilization on one real H100 job first (`nvidia-smi` inside a submitted job, not the login node) rather than assuming Vulcan's numbers transfer — H100 vs L40S have different memory/compute ratios.

**Storage — different layout from Vulcan/Killarney, do not hardcode old paths:**

| Filesystem | Path | Quota | Notes |
|---|---|---|---|
| `$HOME` | `/home/p/paulh` (note extra `p/` nesting) | 25 GiB | source/config only, as elsewhere |
| `$PROJECT` | `/project/aip-jcohen/` (`PROJECT` env var itself is empty — use the literal path) | shared group quota; **file COUNT is the binding constraint** (was 483K/500K files group-wide, only ~17K headroom, while space was two-thirds free) | check `# of files` via `diskusage_report`, not just space |
| `$SCRATCH` | `/scratch/p/paulh` (also extra `p/` nesting; unset in non-login shells — export it explicitly) | 1024 GiB, ~1M files, purge-on-inactivity | **for now, all bulk nnUNet data on TamIA lives here** (see below) |

**Current state (as of 2026-07-31): everything bulk is on `$SCRATCH`, nothing in `$PROJECT` yet.** Given the tight `$PROJECT` file-count quota, the repo + venv live in `$PROJECT` but the actual imaging data — now five datasets (`brats2024-glioma`, `chaos`, `on-harmony`, `open-ms`, and `picai-prostate` — which was downloaded and built directly on TamIA, never staged from Vulcan), not just the original minimal `brats2024-glioma` subset — lives on `$SCRATCH` only. **This means it can be purged on inactivity** — treat it as re-copyable, not durable, until/unless it's deliberately promoted to `$PROJECT` (which would need to be weighed against the file-count quota). The transfer script that repopulates it from Vulcan is kept at `/project/aip-jcohen/paulh/copy_to_tamia.sh` on **Vulcan** (not TamIA) as the recovery path — re-run it from Vulcan if the scratch copy is purged. Cluster-specific path overrides live in `scripts/cluster/tamia_env.sh`, sourced *after* the dataset's own `00_utils/env.sh` (same env-var-override pattern as GPU type above — never fork `run_job.sh` itself). One gotcha already hit: `env.sh` exports `nnUNet_results` unconditionally, so a `${nnUNet_results:-...}` fallback in the override file silently no-ops — override those variables outright, not with a `:-` guard.

**Venv note:** venvs are not portable across clusters — TamIA's was rebuilt fresh via the same `_setup_venv_oneshot.sh` recipe used elsewhere, submitted as a job (never on the login node). One TamIA-specific fix needed: Vulcan's venv carries a hand-written `srcsm_auglab_port.pth` with a hardcoded Vulcan path (not in git, not pip-generated) — every other cluster needs its own one-line `.pth` pointing at its own `sub-workspaces` path, or all custom trainer discovery silently fails.

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

`datasets/01_commun_results/` holds **cross-dataset** comparison tables — summaries
that span all 8 training sets at once (e.g. Ours vs. best-other-method significance,
one row per training set) — distinct from any single dataset's own `8_results_<dataset>/`.
Also holds the **task-level heatmap** (`meta_task_heatmap_summary.md` +
`meta_task_heatmap_heatmap_{dice,hd95}.png`, generated by
`scripts/evaluate/run_meta_task_heatmap.sh`): one row per method, one column per
dataset ("task"), + `overall` + significance vs OURS — the single "which method wins
overall" table. See `combined_modality_summary.py` in the shared layer below for the
per-dataset table this rolls up.

---

## How experiments work — ALWAYS use the shared standardized scripts

**This is a hard project rule, not a style preference.** Every train / predict / evaluate / aggregate step goes through the **shared, standardized script layers** below. Do **not** hand-roll one-off `nnUNetv2_train`/`nnUNetv2_predict`/eval commands, and do **not** duplicate logic that already lives in a shared driver. The entire value of this project is that all datasets behave *identically* so the 6-method results are directly comparable across datasets and contrasts — bypassing, forking, or re-implementing these scripts silently breaks that comparability and is treated as a bug. If something is missing, **add it to the shared layer once (for all datasets)**, don't inline it in one place.

### Three script homes (know which is which)

Scripts live in exactly three places — put new code in the right one:
- **`src/`** — PALETTE method source code only (the contrast transform / model). No pipeline glue.
- **`datasets/00_commun_scripts/`** — the shared **dataset-pipeline** layer: train/predict/evaluate/aggregate/significance that operates on the 9-dir datasets (`00_00_utils` libs, `00_01_train`, `00_02_predict`, `00_03_evaluate` incl. `aggregate_from_config.py` + `significance_from_config.py` + `combined_modality_summary.py` + `meta_task_heatmap.py`, `00_04_analysis`). **All eval/aggregate/significance lives here** (canonical); dataset `5_scripts_*` are thin wrappers over it.
- **`scripts/`** — cross-cutting **infra + research tooling** that is *not* dataset-pipeline: `job_runner/` (the `run_job` backbone — sourced by `common_env.sh`, so it sits *below* `00_commun_scripts`; do not move it), venv setup, `wandb_sync`, the generator/segmenter training entry points, `utils/` (SynthSeg runner + BIDS→nnUNet converters), and `experiments/` runners. `scripts/evaluate/run_significance_all.sh` is the one cross-dataset convenience driver that stays here (it calls the canonical `00_03_evaluate/significance_from_config.py`).

### An "experiment" = a fixed 6-method suite trained on ONE modality

"Start our 6 usual experiments on `<dataset>` `<modality>`" means: train these **exact 6 methods** on that single modality (on-harmony→T1w, open-ms→FLAIR, chaos→T1in/T2spir, brats→t1n/t2w, picai-prostate→T2W/ADC). AugLab configs are under `sub-workspaces/auglab_workspace/AugLab/auglab/configs/`:

| # | method | category | AugLab config (or nnUNet) |
|---|---|---|---|
| 1 | **baseline** | nnUNet | — (nnU-Net default aug) |
| 2 | **auglab_default** | auglab | `transform_params_gpu_default01-23.json` |
| 3 | **synthseg_noEM** | auglab | `transform_params_gpu_default01-23_Synthseg.json` |
| 4 | **synthseg_EM** | auglab | `transform_params_gpu_default01-23_Synthseg_EM.json` |
| 5 | **SRCSM** | auglab | `transform_params_gpu_srcsm_semrandconv.json` |
| 6 | **auglabAug_v26_6_2 (OURS)** | auglab | `transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json` (**train050**; DualVal trainer) |

The `trainXXX_valYYY` suffix (in the METHOD id / RUN_ID) encodes the synth-augmentation probabilities baked into the config — OURS is **train050**. The DualVal trainer (see below) hard-links **both** a `val000` and a `val100` checkpoint mirror from one training run, so best-checkpoint eval needs both RUN_IDs (2 rows) — but since `checkpoint_final.pth` is the identical file in both mirrors, a final-checkpoint eval only needs one, collapsing back to exactly 6 methods. Ablation arms (e.g. `..._noisefill`, `..._no_voronoi`) and SRCSM's earlier life as an extra 7th arm on top of "v26_6_2 alone" are legacy naming — SRCSM is now a fixed member of the 6, not an addition to them.

**FOLD POLICY — 3 folds only, permanently.** Always train **folds 0 1 2 only**. We used to train 4 folds but **from 2026-07-09 onward we never train fold 3 again** (3-fold ≡ 4-fold conclusions; fold 3 is wasted GPU on a congested cluster). If a dataset already has a 4-fold split, **reuse it and just train the first 3 folds** — do **not** create a new split and do **not** train fold 3. The shared drivers now **default** to 3 folds (not merely cap): `00_01_train/train_common.sh` `TRAIN_FOLDS` and `00_02_predict/predict_common.sh` `PREDICT_FOLDS` both default to `"0 1 2"`, `run_all_train_common.sh` exports `TRAIN_FOLDS="0 1 2"`, and `eval_folds.py` `EVAL_FOLDS` caps at 0-2 — so per-method wrappers no longer need to set folds explicitly. Only override (e.g. `TRAIN_FOLDS="0 1 2 3"`) for a deliberate one-off or to predict a legacy 4-fold model.

**EPOCH POLICY — respect each dataset's usual epoch count.** Don't change these without a deliberate reason — they keep the 6-method suite comparable within and across runs of a dataset: **brats2024-glioma = 2500**, **on-harmony = 2000**, **open-ms = 2000**, **picai-prostate = 2000**, **chaos = 200**.

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
- `00_03_evaluate/` — `evaluate.py` (Dice+HD95), `summarize_fold.py`, `aggregate_results.py`, `aggregate_from_config.py` (per-modality summary table + heatmap; also the home of `load_run_cases`/`paired`/`macro_perm`-based `significance_column` — see below), `significance_from_config.py` (full paired-significance report: OOD/IND/per-contrast breakdowns), `combined_modality_summary.py` (pools a dataset's 2 training modalities into one table), `meta_task_heatmap.py` (pools all 4 datasets into one table — see "Standardized output layout" below for all three).
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

**Inline "sig. vs ref" column — auto-wired, not opt-in.** Every table `aggregate_from_config.py` produces (headline per-modality tables, checkpoint sweeps, HP sweeps, causal-ablation ladders, cross-dataset tables — anything driven by a `06_evaluate/configs/*.yaml`) gets an extra column after `all`, `sig. vs ref`, **automatically**, whenever the config's `runs:` list contains a run id matching `REF_SUBSTR = "auglabAug_v26_6_2_train050_val000"` (deliberately narrower than `"v26_6_2"` — that also matches the older non-auglab "v26_6_2 alone" method and its HP-tuning variants). No config change needed; override with `sig_ref: <exact run id>` for an ambiguous run list, or suppress with `no_significance: true`. Value = Holm-corrected one-sided ("ref better") macroΔ sign-flip p-value (same test/math as `significance_from_config.py`'s headline block, shared via `stat_tests.py`'s `macro_perm`/`holm`) of the OURS train050_val000 run vs that row, across ALL tested contrasts (equal weight per contrast, matching the `all` column's own estimand) — blank on the ref's own row. Drawn on the heatmap PNG too (masked/uncolored text column). This is on by default — don't hand-roll a second significance pass on top of it.

**Cross-training-modality tables (`combined_contrasts/`).** Every dataset trains each method TWICE, once per training modality (chaos: t1in/t2spir; brats2024-glioma: t1n/t2w; on-harmony: T1w/T2w; open-ms: flair/t1w). `combined_modality_summary.py` pools BOTH into one row per method — fold data tagged by modality so a 3+3 fold pool is one equally-weighted 6-fold mean (reuses `aggregate_from_config.py`'s `cross_fold_class_mean` unchanged) — writing `8_results_<ds>/02_metrics/<model>/combined_contrasts/01_results_{summary.md,heatmap_dice.png,heatmap_hd95.png}`, same inline `sig. vs ref` column as above (config field `ref:`, required). Driven by `<dataset>_combined_01_results.yaml` (lists each modality's `metrics_dir` + a `method_key → run_id` map) via each dataset's `06_XX_combined_modality_summary.sh`.

**Cross-DATASET task-level heatmap (`datasets/01_commun_results/meta_task_heatmap_*`).** One level up again: `meta_task_heatmap.py` pools all 4 datasets' `combined_contrasts` tables into ONE table — rows = the 7-method suite, columns = one per dataset ("task", literally that dataset's own `all` value), + `overall` (equal weight per task) + `sig. vs ref` (same macroΔ sign-flip test, but stratified by TASK instead of contrast — each task's stratum = every held-out case's paired diff pooled across that task's contrasts+modalities — so the test matches the equal-weight-per-task `overall` column). Reuses the `*_combined_01_results.yaml` configs directly (no run ids re-specified a third time). Run via `bash scripts/evaluate/run_meta_task_heatmap.sh` (config: `scripts/evaluate/meta_task_heatmap.yaml`) — this is the single "which method wins overall" answer; re-run it after any headline run changes rather than eyeballing the 8 per-dataset tables.

Within `02_metrics/<model>/<contrast>/`, a non-headline result set gets its **own dedicated subdir** rather than mixing into the flat headline layout — established conventions so far: `checkpoint_comparison/` (checkpoint_best vs checkpoint_final sweep, all 8 training sets) and `ablations/` (the causal-ablation ladder studies — open-ms FLAIR, chaos T1in, chaos T2spir, brats2024-glioma T1n so far). Headline run dirs stay in the parent and are referenced unprefixed by any config that needs them (cross-dataset tables, `checkpoint_comparison/best_vs_final` configs); run dirs exclusive to the subdir's own study move into it and are referenced as `<subdir>/<run_id>`. `06_01_evaluate_run.sh`'s `METRICS_SUBDIR` env var (e.g. `METRICS_SUBDIR=ablations`, chaos + brats2024-glioma so far) routes evaluation output straight into a subdir like this — set it before evaluating a new ablation run rather than evaluating flat and moving the directory by hand afterward.

**The causal-ablation ladder** (one `06_1X_ladder_summary.py`-style script per dataset/train-contrast — `open-ms/.../06_12_ladder_summary.py`, `brats2024-glioma/.../06_13_ladder_summary.py`, `chaos/.../06_33_ladder_summary_t2spir.py` — reusing `00_commun_scripts/00_03_evaluate/significance_from_config.py`'s `load_run_cases`/`resolve_run_dir` for OOD extraction; don't hand-roll a new CSV loader for a new dataset's ladder): a baseline-anchored rung sequence — `baseline → +K-means intensity clustering → +label-remap → +Voronoi sub-parcellation (noise fill) → v26_6_2/PALETTE alone (identical partition, real-intensity fill) → +AugLab (val000) → +AugLab (val100)` — where each rung adds exactly one ingredient on top of the previous, so the OOD Dice/HD95 delta between two adjacent rungs is attributable to that ingredient alone. The rung 4→5 step (noise-fill → real-fill, partition otherwise unchanged) is the key one-variable test of whether texture preservation causally drives Dice: large (~+7 Dice, ~-4 to -5mm HD95) on texture-defined segmentation targets (open-ms MS lesions, brats2024-glioma tumor sub-regions) and near-zero/noise-level on boundary-defined ones (chaos organs, both T1in and T2spir) — texture preservation only pays off when the segmentation target actually depends on texture.

Predicting/evaluating at a **non-default checkpoint** (project default is `checkpoint_best.pth`) is a uniform mechanism across all 4 dataset families: `predict_common.sh`'s `CHECKPOINT` env var and each dataset's `06_01_evaluate_*.sh`'s `CKPT_TAG` env var route non-best predictions/metrics to sibling `fold{k}/<tag>/` / `{category}_{run_id}_<tag>/` paths, never touching the default best-checkpoint data. Conclusion from the 2026-07-31 sweep (see each dataset's `checkpoint_comparison/best_vs_final_ckpt_summary.md`): `checkpoint_best` beats `checkpoint_final` overall, most clearly on open-ms — final-checkpoint results are not a drop-in upgrade.

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
- `06_01_evaluate_run.sh` (chaos, brats2024-glioma) defaults `DATASET_ID` internally to the training set's *primary* contrast. Evaluating the *secondary* modality (e.g. brats t2w, chaos t2spir) without passing `DATASET_ID` explicitly silently scores against the wrong ground truth — no error, just wrong numbers. Always pass it explicitly for the non-primary modality.
- open-ms's evaluate script requires `CATEGORY` (`nnUNet`/`auglab`) explicitly — unlike chaos/brats, it does not auto-detect it. A wrong `CATEGORY` doesn't error either; it silently writes an empty, `_logs`-only metrics dir. Check `eval_all.csv` actually exists before trusting a run finished.
- When writing a one-off/filtered aggregate or significance config (e.g. to compare a subset of methods), always set an explicit `output_dir` pointing at scratch space. Omitting it makes `aggregate_from_config.py`/`significance_from_config.py` fall back to the dataset's real output dir and **silently overwrite the actual headline `*_summary.md`/`*_significance.md` files.**
