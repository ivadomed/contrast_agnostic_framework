# Project: MRI Synthesis — Claude Context

## Working with Doppel and autonomous sessions

Paul runs this project with a coordinating Claude Code session named **Doppel**, plus a rotating set of task-specific Claude Code sessions (one per dataset/task, or spun up ad hoc for audits, paper writing, git-annex work, etc.) that Doppel dispatches and receives reports from. This is a standing, normal way this project operates — not a one-off.

**Sessions live in named tmux windows, each on its OWN dedicated socket (adopted 2026-09-17, corrected same day).** Launch/revive a named session like this:
```bash
systemd-run --user --scope --unit="claude-<name>" -- tmux -L <name> new-session -d -s <name> "claude --resume '<Session Name>'"
```
Attach with `tmux -L <name> attach -t <name>` (detach `Ctrl-b d`), from any of Paul's connections, whenever he wants to look in or take over. Doppel can create/revive these itself via Bash — legitimate, expected, no need to ask permission each time.

**Why both `systemd-run --user --scope` AND a per-session socket (`-L <name>`) are both required, not just one:**
- Root cause of the original session-loss incident: a bare `tmux new-session -d` places the tmux *server* process inside the cgroup of whichever specific login session happened to run the command (`session-N.scope`, visible via `cat /proc/<pid>/cgroup`) — when *that particular* session ends (SSH drop, reconnect, timeout), systemd tears down its scope and kills everything in it, tmux server included, **even if Paul has other sessions open elsewhere**. `Linger=no` (`loginctl show-user paulh`) makes this worse (kills things even faster) but isn't the whole story — the session-scope binding is the deeper issue, and Paul doesn't have permission to run `loginctl enable-linger` on this account anyway.
- `systemd-run --user --scope` moves the process into the persistent `user@<uid>.service/app.slice/` cgroup instead — only torn down if *every* session for the account ends simultaneously (the true linger scenario), not just the one that happened to launch it.
- **But** if you reuse the default tmux socket for a second/third session, subsequent `tmux new-session` calls just ask the *already-running* server (started earlier, possibly still in the wrong cgroup) to open a new window — the client invocation gets wrapped by `systemd-run`, but the actual server process never moves. Verified live 2026-09-17: wrapping `tmux new-session -d -s <name2>` in `systemd-run` while an old default-socket server was already running left `<name2>`'s pane in the *old* session-scope cgroup, not the new one. **Giving each session its own socket (`-L <name>`) forces a fresh dedicated server process per session, so the `systemd-run` wrapper actually takes effect.**

If a named session ever seems to have vanished, check `tmux -L <name> ls` (each is its own socket now, so `tmux ls` alone with no `-L` only shows the default socket) before assuming the work is lost — the conversation history survives via `claude --resume`, only the live process needs relaunching with the command above.

**Separately: session transcripts themselves also expire** — Claude Code stores them under `~/.claude/projects/` and deletes anything older than `cleanupPeriodDays` (default 30 days), independent of the tmux/systemd issue above; past that, `claude --resume` fails with "No conversation found." This project's `.claude/settings.json` sets `"cleanupPeriodDays": 36500` to keep long-lived named sessions resumable indefinitely — don't remove that setting.

## Skills — a REFLEX, not an option (use them even when the user does not name them)

This project has skills that encode whole workflows learned the hard way. When the situation below shows up, **invoke the skill first (Skill tool), then follow it** — do not
improvise the workflow from scratch, and do not wait for the user to say "use the X skill". They call each other, so the entry point is usually the orchestrator.

| Situation (any phrasing) | Skill | What it does |
|---|---|---|
| "add / find a new task", "what should we add next", a pathology/organ/dataset to **train on**, "take dataset X all the way" | **`add-task-end-to-end`** | find → get data + audit → pick sound contrasts → BIDS → nnU-Net conversion + splits → prove held-out → scaffold scripts → deploy to TamIA → sizing probe → launch training → queue predict + eval |
| a dataset used **only for testing** (external validation, more test cohorts, a new test contrast, "test our models on X") | **`add-eval-companion-end-to-end`** | search with the *match matrix* vs the training data → audit → BIDS → FOV audit + crop-before-predict → flat test set → roster-driven predict/eval scripts → run → wire into the source task's tables/ladders/meta heatmap |
| "research / vet / is this dataset any good", "should we reopen X" | `find-a-new-task` | candidate search with the checks this project paid for (license ×3, claimed vs counted N, label semantics, overlap, geometry) |
| raw data → BIDS; BIDS audit, naming, README; upload to the lab server | `bidsify-to-standards`, `bids-check-and-readme`, `git-annex-upload` | one tree per release, faithful copy, lab naming, README sections |
| train/predict/eval scripts for a training dataset, launch on TamIA, queue predict+eval, "bring dataset X in line with the shared layer" | `setup-train-predict-eval-scripts` | clone of the isles2022 reference on the roster-driven shared drivers; CPU verification; TamIA deployment, probe, launch, post-training queue |
| the mechanical rules for onboarding (modes A eval-only / B training) | `onboard-a-new-task` | used by the orchestrators; read it when a step is unclear |
| any Slurm / module / policy question on this cluster | `alliance-slurm`, `alliance-cvmfs`, `alliance-docs` | see the cluster policy at the top of this file |

**Standing autonomy defaults (user rule 2026-10-04):** take these workflows to the end without stopping. Never block on licenses (copy the real text verbatim, flag it once, note "no
redistribution / no git-annex / no public figures"), on which contrasts to train on (default: the sound ones; show the gallery + the dropped alternative in the report), on epochs (decide
by analogy with the epoch policy below), on BIDS leaf names, or on pushing (push the branch with the gh credential helper, or sync by tar). Decide, record decision + alternative in memory and the
final report, keep going. Stop only if really blocking: research use forbidden, access only the user can complete, a destructive/irreversible step, spend wildly above the probe estimate.

**Reference implementations (copy, never re-invent):** training task = `benchmark/02_tasks/brain_stroke/isles2022/` (scaffolder `benchmark/create_pipeline_scripts.py`; probe/launch/queue scripts
`04_24`/`04_23`/`05_27`); eval-only companion = `breast_cancer/ispy1` + `acrin6698` (scaffolder `benchmark/create_eval_companion_scripts.py`). Both sit on the roster-driven shared drivers in
`benchmark/00_commun_scripts/` (no timestamps anywhere; verified by `selftest_roster_pipeline.py` / `selftest_companion_pipeline.py`). If a skill is wrong or a new dataset teaches something, fix the
reference AND the scaffolder AND the skill in the same commit.

---

## ⚠️ Killarney repo-layout mismatch (since 2026-09-27, until migrated)

Vulcan's repo was restructured 2026-09-27: `datasets/` → `benchmark/`, with every dataset moved one
level deeper into `benchmark/02_tasks/<task>/<dataset>/` (see "Dataset structure" below for the
full mapping). **TamIA was migrated 2026-09-27/30 and is now current** (physical `mv` of its
`$SCRATCH` data done, repo pulled up to date — confirmed via direct HEAD comparison). **Killarney
has NOT been migrated** — its own separate clone and scratch-data layout still use the old
`datasets/<name>/` structure, deliberately, so its in-flight jobs aren't disrupted. **Do not
`git pull` this restructuring commit onto Killarney until its own data directories are migrated to
match** — pulling the new scripts onto still-old data would break every path resolution there. Do
the physical `mv` on Killarney's own `$SCRATCH`/repo checkout first (same task-folder mapping as
Vulcan/TamIA), then pull. Until that follow-up happens, treat Killarney as still being on the
pre-restructuring layout for anything you tell it to run.

## Cluster resource management (Vulcan / Slurm)

**`git mv` on a directory silently drops untracked/gitignored files inside it** — verified via a
sandbox test. For any directory move (dataset restructuring, archival, etc.), use plain `mv` then
`git add -A` to let git's content-based rename detection take over; never `git mv` a directory.

**Multiple agents/sessions share one working tree — never run a bare `git commit`.** It commits
everything currently staged, including files another concurrent agent staged. Always
`git add <exact paths>` then commit, even for a single-file change.

**Three machines are in active use** (`run_job` auto-detects the backend on each — see below, so pipeline scripts run unchanged everywhere):

| Machine | Type / backend | Role |
|---|---|---|
| **Vulcan** (AMII / Alliance) | Slurm cluster | **Primary.** All datasets + checkpoints live here; the main place to train/predict/eval. GitHub push works from here (done manually). Can be queue-congested when priority is low. |
| **Killarney** (Alliance) | Slurm cluster (same as Vulcan) | **Overflow GPU.** Same cluster type. Datasets staged there as of 2026-07-10: **chaos** (`1_BIDS_chaos/chaos-abdominal`, `2_nnUNet_chaos`) and **open-ms**. brats2024-glioma / on-harmony / others not yet staged — rsync from Vulcan first if needed there. Used when Vulcan priority runs low and we need more GPU. ⚠️ **Naming drift (2026-08-28):** Vulcan's `1_BIDS_chaos` leaf was renamed `chaos-abdominal` → `abdomen-chaos`, and `1_BIDS_open-ms`'s leaf `open-ms-brain` → `ms-brain-openms`, for git-annex upload prep — Killarney's already-staged copies still use the **old** leaf names until separately re-synced/renamed there. Don't assume the two clusters' paths match until that's done. |
| **romane** (NeuroPoly lab) | `set_slot` workstation | **Limited-use lab box, only 4 GPUs.** The original dev machine; used for smaller/interactive GPU jobs. |
| **TamIA** (Alliance) | Slurm cluster (same family as Vulcan/Killarney) | **Newest, biggest-GPU cluster — added 2026-07-25.** Whole-node H100/H200 allocations (see below), much more GPU per node than Vulcan/Killarney's L40S. **Use these GPUs properly — a coworker was already warned by Alliance staff about under-utilizing allocated GPUs on this account, and TamIA's whole-node model makes idle GPUs especially visible.** Currently only reachable via a relay through Vulcan (see below); as of 2026-07-31 all 4 datasets (`brats2024-glioma`, `chaos`, `on-harmony`, `open-ms`) are staged there, and **all data currently lives on `$SCRATCH` only** (nothing in `$PROJECT` yet) — see the TamIA-specific subsection for what that means operationally. |

**SSH hosts + repo path per machine** (for rsync/ssh between them):
- **Vulcan**: `ssh vulcan.alliancecan.ca` — repo at `/project/aip-jcohen/paulh/mri_synthesis_project` (also reachable via the symlink `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`). `RUN_JOB_ACCOUNT=aip-jcohen` (the `run_job_slurm.sh` default).
- **Killarney**: `ssh killarney.alliancecan.ca` — repo at `/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project`. **`RUN_JOB_ACCOUNT=aip-jcohen` — same account name as Vulcan, do NOT override it.** (`benchmark/02_tasks/brain_ms/open-ms/KILLARNEY_HANDOFF.md` speculated `def-jcohen` before this was confirmed on real hardware 2026-07-10 via a Slurm "Invalid account" error listing `aip-jcohen` as the only valid AIP account on this cluster — that doc is stale on this point.) See that same file for the still-relevant port notes: venv rebuild, GPU-type parametrisation, trainer shims.
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

nnunetv2 also needs files restored that exist ONLY as plain pip-installed files inside its own site-packages dir — there is no auto-copy step for these (unlike the `auglab_add_nnunettrainer` mechanism below), so a fresh `pip install nnunetv2` will NOT bring them back on its own (the setup script above already does this; listed here for when it needs doing by hand):
1. `cp src/nnunet/patches/nnunet_logger.py .venv/lib/python3.*/site-packages/nnunetv2/training/logging/nnunet_logger.py` — hand-patched WandB logger (`nnUNet_wandb_run_id` resume + `allow_val_change` for epoch-extension). Without it, WandB resume-by-id and raising `NNUNET_NUM_EPOCHS` on resume both break.
2. Every dataset's own nnU-Net trainer-registration shim (each `5_scripts_<name>/02_nnunet/*Trainers.py`) — **auto-discovered and copied via `find benchmark/02_tasks -path '*/02_nnunet/*Trainers.py'`** (fixed 2026-09-27: this used to be a hand-maintained list that had already silently drifted out of sync with reality — it named only 3 of the 8 real shims that existed at the time, missing `OpenMSTrainers.py` from the script itself and `CHAOSLowDataTrainers.py`/`ISPY2Trainers.py`/`ToothFairy2Trainers.py`/`TotalsegPelvicTrainers.py` from both the script and this doc entirely). A hardcoded list can't help but silently miss a future dataset's shim — training then fails with a cryptic "trainer not found", with no link back to this root cause. Don't reintroduce a fixed list here; if you need to know the current real roster, just run that `find` command.

(2 is a "registration shim" mechanism — nnU-Net's `recursive_find_python_class` only searches inside its own `training/nnUNetTrainer/` dir, so each dataset's real trainer classes, which safely live in git under e.g. `benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/brats2024_glioma/trainers/` already, need a thin shim there that imports them via `NNUNET_PROJECT_ROOT`. Without its shim, **no custom trainer for that dataset is discoverable at all** — training fails with "trainer not found".)

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

**From inside a Claude Code session already running on Vulcan, skip the outer hop** — just `ssh
tamia.alliancecan.ca` directly (you're already on Vulcan; wrapping it in another
`ssh vulcan.alliancecan.ca` fails with "Permission denied (keyboard-interactive)").

**Before any TamIA `git pull`, check `git status` for local modifications first** — twice now
they've turned out to be stale pre-2026-09-27-restructuring path hop-counts (`../../../../` vs
the current `../../../../../../`), already fixed upstream. Diff each against the current Vulcan
file to confirm, then `git checkout --` to discard before pulling.

**GPUs — bigger than Vulcan/Killarney, whole-node only:**

| Node class | GPUs/node | CPUs | RAM | Use |
|---|---|---|---|---|
| `h100:4` (the bulk of the cluster) | 4× H100 | 48 | 500,000 MB | **Default — use this class** |
| `h200:8` | 8× H200 | 64 | 1,000,000 MB | Bigger node class — do **not** request unless explicitly asked |

Partitions are `gpubase_bynode_b1/b2/b3` — **allocation is whole-node** (no per-GPU partition), so a submitted job gets all 4 (or 8) GPUs on the node whether it uses them or not. `run_job` env overrides for the default h100 class: `RUN_JOB_GPU_TYPE=h100`, `RUN_JOB_CPUS_PER_GPU=12` (48/4), **`RUN_JOB_MEM_PER_GPU=115G`**. ⚠️ That variable is parsed in **GiB**, not MB — `run_job_slurm.sh` does `mem="$(( ${RUN_JOB_MEM_PER_GPU%G} * gpus ))G"`, stripping a trailing `G` and re-appending one. Passing the raw MB figure (`125000`) requests 500000 **G** and sbatch rejects the job with "Memory specification can not be satisfied" (cost a lost submission 2026-08-02). Leave headroom under the node's 500000 MB: 115G × 4 = 460G. As always, these are env-var overrides only — `run_job_slurm.sh` itself is never forked per cluster. Do **not** pass `--partition` explicitly on TamIA for **GPU** jobs (rejected even for partitions `sinfo` lists as valid) — submit without it and let the scheduler route.

**CPU-only `sbatch` submission on TamIA has been inconsistent** — a bare script (only `--cpus-per-task`/`--mem`, no `--gres`) has both failed with `sbatch: error: No partition specified or system default partition` AND succeeded with no `--partition` at all, on different submissions of similar jobs the same day (2026-09-07, duke-breast-mri cross-dataset eval pipeline). Adding an explicit `--partition=cpubase_bynode_b1/b2` did **not** reliably fix it either — one attempt was then rejected with "partition does not exist or job cannot fit in it," and a bare resubmission of the exact same script immediately after succeeded. Root cause undetermined (possibly transient scheduler/queue state, possibly a `--mem` value that doesn't fit some partitions' per-node limit — unconfirmed). **Don't assume either explicit or omitted `--partition` is the fix** — if a CPU-only submission fails on one, retry with the other before assuming something else is wrong; this has resolved it both times so far.

**Use the GPUs properly — this is now a hard requirement, not a nice-to-have.** Because allocation is whole-node, a job that only exercises 1 of 4 (or 1 of 8) GPUs wastes the other 3 (or 7) for the whole wall-clock duration — highly visible to Alliance staff, and **a coworker on this account has already been warned about under-utilizing allocated GPUs.** Concretely, for the H100 nodes:
- Node-pack placement is explicit when it needs to be: `run_job_pack_submit.sh` spreads recorded folds round-robin (`i%4`) by default, but **`PACK_GPU_MAP`** (one GPU index per `index.tsv` row) overrides that. Use it whenever folds cost very different amounts — notably **srcsm, which is ~3x slower per epoch than every other method**: round-robin will pair a srcsm fold with a second fold and make it the straggler that holds the whole node *and its whole dependency chain* open. Give srcsm folds a GPU to themselves and double up the cheap folds instead. **Don't assume the 3x figure transfers to a new dataset — measure it first**: a 2026-08 attempt on a since-abandoned dataset (unreliable voxel-level annotations, unrelated to this tooling) found srcsm running no slower than the other methods there, the opposite of brats/on-harmony's pattern.
- **`PACK_DIR` must be qualified per training contrast — never share one `PACK_DIR` across two training-contrast batches.** Recording predict/eval jobs for two contrasts (e.g. t1wce and t2w) into the same `PACK_DIR` produces bare-method-name collisions (`baseline_kmeans` means something different per contrast) that silently overwrite each other's cmd files before submission — the **last-recorded contrast wins, the first is silently lost, no error**. Hit 3 times on this project already. Fix: always use a separate, contrast-qualified `PACK_DIR` per batch, and **grep-verify the recorded cmd files reference the right contrast before submitting** (e.g. `grep -oE '<model>/t[12][a-z]*w?/' cmdfile`) — don't trust the recording step silently.
- Before choosing a layout, **run a sizing probe** — a few epochs of the heaviest and the slowest method side by side on one real node, reporting per-fold VRAM + epoch time, using a throwaway results base so it can't pollute real checkpoints. CLAUDE.md already required measuring rather than assuming; write this probe fresh per dataset rather than assuming a prior one's numbers transfer.
- Prefer packing **multiple folds onto the same node's GPUs** in one job rather than 1 fold = 1 whole node. With 4 GPUs/node, that means either 4 folds in parallel per job, or — since the current fold policy trains only folds 0/1/2 — 1 GPU sits idle unless a second task (a different method, modality, or ablation arm) is packed alongside.
- The bigger H100 memory headroom vs Vulcan's L40S (each H100 has more VRAM) also means **larger batch sizes are affordable** — this changes the training config (not the shared script), so treat it as a per-cluster override the same way GPU type/CPU/mem are already overridden, not a hand-edit of `train_common.sh`.
- Before changing batch size or fold-packing, size actual per-fold GPU-memory and utilization on one real H100 job first (`nvidia-smi` inside a submitted job, not the login node) rather than assuming Vulcan's numbers transfer — H100 vs L40S have different memory/compute ratios.

**Storage — different layout from Vulcan/Killarney, do not hardcode old paths:**

| Filesystem | Path | Quota | Notes |
|---|---|---|---|
| `$HOME` | `/home/p/paulh` (note extra `p/` nesting) | 25 GiB | source/config only, as elsewhere |
| `$PROJECT` | `/project/aip-jcohen/` (`PROJECT` env var itself is empty — use the literal path) | shared group quota; **file COUNT is the binding constraint** (was 483K/500K files group-wide, only ~17K headroom, while space was two-thirds free) | check `# of files` via `diskusage_report`, not just space |
| `$SCRATCH` | `/scratch/p/paulh` (also extra `p/` nesting; unset in non-login shells — export it explicitly) | 1024 GiB, ~1M files, purge-on-inactivity | **for now, all bulk nnUNet data on TamIA lives here** (see below) |

**Current state (as of 2026-07-31): everything bulk is on `$SCRATCH`, nothing in `$PROJECT` yet.** Given the tight `$PROJECT` file-count quota, the repo + venv live in `$PROJECT` but the actual imaging data — now all 4 datasets (`brats2024-glioma`, `chaos`, `on-harmony`, `open-ms`), not just the original minimal `brats2024-glioma` subset — lives on `$SCRATCH` only. **This means it can be purged on inactivity** — treat it as re-copyable, not durable, until/unless it's deliberately promoted to `$PROJECT` (which would need to be weighed against the file-count quota). The transfer script that repopulates it from Vulcan is kept at `/project/aip-jcohen/paulh/copy_to_tamia.sh` on **Vulcan** (not TamIA) as the recovery path — re-run it from Vulcan if the scratch copy is purged. Cluster-specific path overrides live in `scripts/cluster/tamia_env.sh`, sourced *after* the dataset's own `00_utils/env.sh` (same env-var-override pattern as GPU type above — never fork `run_job.sh` itself). One gotcha already hit: `env.sh` exports `nnUNet_results` unconditionally, so a `${nnUNet_results:-...}` fallback in the override file silently no-ops — override those variables outright, not with a `:-` guard.

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

**Restructured 2026-09-27: `datasets/` → `benchmark/`, and every dataset now sits one level
deeper, grouped by task, under `benchmark/02_tasks/<task>/<dataset>/`.** `<task>` follows an
`anatomy_pathology` (or `anatomy_healthy` when the dataset isn't disease-specific) naming
convention. Current roster:

| Task folder | Datasets |
|---|---|
| `breast_cancer` | ispy2 (trains); duke-breast-mri, ispy1, acrin6698 (eval-only) |
| `brain_tumor` | brats2024-glioma |
| `brain_ms` | open-ms |
| `brain_healthy` | on-harmony |
| `brain_stroke` | isles2022 (trains; ISLES'22 acute/subacute ischemic stroke — onboarding in progress, see section below) |
| `mandible_healthy` | toothfairy2, hanseg, pddca |
| `abdomen_healthy` | chaos, amos, sliver07 |
| `pelvis_healthy` | totalseg-pelvic |
| `spine_healthy` | healthy-spine-tum |
| `pancreas_disease` | pansegdata (trains; pancreas MRI T1WCE+T2W, 212 subjects after excluding center MCF — see its section below) |

`benchmark/03_archive/<name>` (17 excluded/superseded datasets) stays flat, untouched by the
task taxonomy — archived datasets don't need it, and archival status is orthogonal to anatomy/
pathology. Shared infra keeps the same numbered-sibling convention it always had, just under the
new top-level name: `benchmark/00_commun_scripts/`, `benchmark/01_commun_results/`,
`benchmark/03_archive/`, plus the loose root tools (`validate_standard_dataset_structure.py`,
`create_dataset_structure.py`, `seed_skeleton_gitkeep.py`, `STANDARDIZATION_CHECKLIST.md`,
`conftest.py`).

**Path-depth gotcha for anyone porting an old command/note that predates this move:** a dataset's
own `5_scripts_*/00_utils/env.sh` still resolves `DATASET_ROOT` purely relatively (unaffected by
nesting depth), but everything that reaches OUT of the dataset tree — `${DATASET_ROOT}/../
00_commun_scripts`, `DATASET_ROOT.parent / "00_commun_scripts"`, `Path(__file__).resolve()
.parents[N]`-style `PROJECT_ROOT` computations, the `../../../../sub-workspaces` hop to AugLab —
needed exactly 2 more levels added everywhere, since active datasets moved 2 levels deeper
(`02_tasks/<task>/`) while archived ones (siblings of `00_commun_scripts`, unchanged) did not. If
you ever hand-write a new script for one of the active datasets, copy the hop-count from another
already-working file in that SAME dataset rather than an archived one.

We work in a dataset-centric manner. Every dataset follows the same 9-subdir standard:
```
benchmark/02_tasks/<task>/<dataset>/        # or benchmark/03_archive/<dataset>/ if archived
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
      # (on-harmony migrated to this modern layout; see 8_results_on-harmony/ on disk)
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

`benchmark/01_commun_results/` holds **cross-dataset** comparison tables — summaries
that span all 8 training sets at once (e.g. Ours vs. best-other-method significance,
one row per training set) — distinct from any single dataset's own `8_results_<dataset>/`.
Also holds the **task-level heatmap** (`meta_task_heatmap_summary.md` +
`meta_task_heatmap_heatmap_{dice,hd95}.png`, generated by
`scripts/evaluate/run_meta_task_heatmap.sh`): one row per method, one column per
dataset ("task"), + `overall` + significance vs OURS — the single "which method wins
overall" table. See `combined_modality_summary.py` in the shared layer below for the
per-dataset table this rolls up.

**⚠️ Task-roster drift risk — verify `scripts/evaluate/meta_task_heatmap.yaml`'s `tasks:` list
directly before citing a task count anywhere.** The paper's own copy
(`paper/scripts/meta_task_heatmap_paper.yaml`) and the project-side one above are **edited
independently** — a task landing in the paper (breast/ispy2, toothfairy2, ...) does not mean
someone also wired it into the project-side "which method wins overall" table, and vice versa.
Don't assume they're in sync; `grep -c '^  - name:' scripts/evaluate/meta_task_heatmap.yaml` costs
nothing and prevents citing a stale count.

### Dataset onboarding — pre-flight checklist (do these BEFORE a download completes, not after)

Three checks this project has paid for skipping (each cost real engineering time to unwind after
the fact — see the AMBL and Atlas-Liver-HCC sections below for the concrete cost of skipping #2/#3):

1. **License, verified independently, not read off one page.** The established method (see
   `benchmark/02_tasks/breast_cancer/ispy2/0_raw_ispy2/README.md` / any archived AMBL README for the template): (a) the
   collection/source page's own license statement, (b) the actual API/metadata license fields on a
   real sample of series/records (not just the page's prose), (c) the LICENSE file bundled inside a
   downloaded file itself, if one exists. All three should agree before you call a license settled.
   Duke-breast-mri (via MAMA-MIA/Synapse) got fully downloaded and processed once with **zero**
   license check — caught only when someone else asked. Do the check before, not after.
2. **Claimed-N vs. directly-counted usable-N.** A collection page's "N patients"/"masks for all
   patients" is marketing copy, not a manifest — count the real annotated population yourself from
   the raw series/object metadata (AMBL: 632 claimed → 99 actually have a lesion SEG; I-SPY2: 719 →
   561 with both a real DCE and a real T2w series). Do this **before** choosing between candidate
   datasets, not after committing engineering effort to one.
3. **Ground-truth label semantics, read from the source data/README, not inherited from a script
   docstring.** A docstring calling a dataset "malignant-only" or "tumor-only" is a claim, not a
   fact — verify it against the actual segmentation object's own per-segment labels (`SegmentLabel`/
   `SegmentDescription` in a DICOM-SEG, or equivalent). This exact failure already happened once this
   project (a "malignant-only external test set" docstring was wrong — the real mask unions tumor
   AND benign segments into one class) and was only caught on a second, independent read of the
   source README.

---

## How experiments work — ALWAYS use the shared standardized scripts

**This is a hard project rule, not a style preference.** Every train / predict / evaluate / aggregate step goes through the **shared, standardized script layers** below. Do **not** hand-roll one-off `nnUNetv2_train`/`nnUNetv2_predict`/eval commands, and do **not** duplicate logic that already lives in a shared driver. The entire value of this project is that all datasets behave *identically* so the 6-method results are directly comparable across datasets and contrasts — bypassing, forking, or re-implementing these scripts silently breaks that comparability and is treated as a bug. If something is missing, **add it to the shared layer once (for all datasets)**, don't inline it in one place.

### Three script homes (know which is which)

Scripts live in exactly three places — put new code in the right one:
- **`src/`** — PALETTE method source code only (the contrast transform / model). No pipeline glue.
- **`benchmark/00_commun_scripts/`** — the shared **dataset-pipeline** layer: train/predict/evaluate/aggregate/significance that operates on the 9-dir datasets (`00_00_utils` libs, `00_01_train`, `00_02_predict`, `00_03_evaluate` incl. `aggregate_from_config.py` + `significance_from_config.py` + `combined_modality_summary.py` + `meta_task_heatmap.py`, `00_04_analysis`). **All eval/aggregate/significance lives here** (canonical); dataset `5_scripts_*` are thin wrappers over it.
- **`scripts/`** — cross-cutting **infra + research tooling** that is *not* dataset-pipeline: `job_runner/` (the `run_job` backbone — sourced by `common_env.sh`, so it sits *below* `00_commun_scripts`; do not move it), venv setup, `wandb_sync`, the generator/segmenter training entry points, `utils/` (SynthSeg runner + BIDS→nnUNet converters), and `experiments/` runners. `scripts/evaluate/run_significance_all.sh` is the one cross-dataset convenience driver that stays here (it calls the canonical `00_03_evaluate/significance_from_config.py`).

### An "experiment" = a fixed 6-method suite trained on ONE modality

"Start our 6 usual experiments on `<dataset>` `<modality>`" means: train these **exact 6 methods** on that single modality (on-harmony→T1w, open-ms→FLAIR, chaos→T1in/T2spir, brats→t1n/t2w). AugLab configs are under `sub-workspaces/auglab_workspace/AugLab/auglab/configs/`:

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

**EPOCH POLICY — respect each dataset's usual epoch count.** Don't change these without a deliberate reason — they keep the 6-method suite comparable within and across runs of a dataset: **brats2024-glioma = 2500**, **on-harmony = 2000**, **open-ms = 2000**, **chaos = 200**.

**How to launch all 6 (one command):** a per-modality launcher lists the 6 per-method wrapper paths in a `METHOD_SCRIPTS` array and sources the shared runner `benchmark/00_commun_scripts/00_01_train/run_all_train_common.sh` — which caps folds at `0 1 2`, runs them in order, and supports `--start-from <method>` to resume a suite. Canonical example: `benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/04_train/04_12_run_all_flair.sh` (→ `bash 04_12_run_all_flair.sh`). Each method is still its own thin `04_XX_train_<modality>_<method>.sh` wrapper (on-harmony T1w = `04_07`…`04_12`). For a **new** dataset/modality: create the 6 per-method wrappers (each ~5 lines: set METHOD/TRAINER/config, source `04_00_common.sh`) + one per-modality run-all launcher. *(The old `04_06_run_all_training.sh` is dead — ignore it.)*

### AugLab is a SEPARATE, config-driven repo (sync it per-machine)

`sub-workspaces/auglab_workspace/AugLab` is its **own git repo** (editable-installed; it shows as untracked `?? sub-workspaces/` in the main repo — its history does **not** ride the main-repo git). AugLab augmentations are selected entirely by **JSON config** (the `transform_params_gpu_*.json` files above); our nnU-Net trainers (the `nnUNetTrainer*AugLab*` classes) basically wrap/call AugLab's trainer/transform machinery. Two consequences:
- To change augmentation behaviour you usually add/edit a **config**, not code.
- When you *do* change AugLab **code** (a new transform, a param), that change must be **synced to each machine independently** — it will NOT arrive via a main-repo `git pull`. (This bit us: a `palette_noisefill.py` fix existed on one machine but not another even though the main-repo training scripts had merged.)

*Aside — external SynthSeg vs "synthseg" augmentation:* the SynthSeg-style **augmentation** methods (`synthseg_EM`/`noEM`) are just AugLab configs (above) — they do **not** need the external SynthSeg model. The external SynthSeg tool (`scripts/utils/run_synthseg_predict.py`, needs a `SynthSeg/` repo + `.h5` model) is used **only** to generate anatomical **parcellation labels** for a dataset that lacks them — a separate concern from training.

### Train on ONE modality → predict / evaluate / aggregate on ALL modalities

Models are contrast-agnostic, so evaluation is **cross-contrast**: a model trained on one modality is tested on **held-out cases across every modality of the dataset** (on-harmony: T1w/T2w/FLAIR/GRE/dwi/bold/epi; open-ms: FLAIR/T1w/T2w; …). Dice + HD95 are computed **per test-contrast, per fold**, then aggregated and significance-tested across the full 6-method table. **This cross-contrast generalization IS the headline result** — never evaluate only on the training modality.

### The shared layer (canonical — reuse, don't reinvent): `benchmark/00_commun_scripts/`

- `00_00_utils/` — shared libs: `common_env.sh`, `splits_lib.py`, `orient.py`, `fov.py`, `eval_metrics.py`/`eval_folds.py`/`eval_aggregate.py`, `stat_tests.py`, `nnunet_convert_lib.py`.
- `00_01_train/train_common.sh` — training driver (fold fan-out via `run_job`; honours optional `TRAIN_FOLDS`, default `"0 1 2 3"`).
- `00_02_predict/predict_common.sh` — prediction driver (own- and cross-model modes).
- `00_03_evaluate/` — `evaluate.py` (Dice+HD95), `summarize_fold.py`, `aggregate_results.py`, `aggregate_from_config.py` (per-modality summary table + heatmap; also the home of `load_run_cases`/`paired`/`macro_perm`-based `significance_column` — see below), `significance_from_config.py` (full paired-significance report: OOD/IND/per-contrast breakdowns), `combined_modality_summary.py` (pools a dataset's 2 training modalities into one table), `meta_task_heatmap.py` (pools all 4 datasets into one table — see "Standardized output layout" below for all three), `ladder_ood_common.py` (the causal-ablation ladder engine, `run_ladder()`/`run_ladder_cross_dataset()` — see "The causal-ablation ladder" below; **OOD-only plots as of the 2026-09-07 rework**, `ladder_series.json` carries a `fill_swap_significance` field per source), `ladder_cross_dataset_plot.py` (added 2026-09-08: overlays several already-computed `ladder_series.json` sources' OOD-pooled curves on one comparison figure per training direction, reusing `ladder_ood_common.py`'s own `_write_per_contrast_png` rather than a bespoke plot — see its docstring for the grouping convention).
- **Roster-driven drivers (added 2026-10-04, first used by isles2022; older datasets still carry their own copies and may migrate):**
  `00_00_utils/roster_lib.sh` (find a method's RUN_ID from the predict wrappers' `METHOD=`/`CATEGORY=` lines + the trained run dirs, with a
  hand-editable pin file `01_predictions/<model>/<contrast>/roster_run_ids.tsv`), `00_02_predict/run_all_predict_common.sh`,
  `00_03_evaluate/evaluate_run_common.sh` (the per-run evaluate body every dataset used to copy-paste; hard-fails on #preds != #GT, wrong CATEGORY,
  missing eval_all.csv rows), `run_all_evaluate_common.sh`, `write_configs_from_roster.py`, `ladder_from_roster.py`. A new dataset's predict/eval
  launchers and configs contain NO timestamps. See skill `setup-train-predict-eval-scripts`.
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

**⚠️ This "always use the shared/common scripts" rule is the single most important thing in this file, worth restating plainly: never hand-roll a train/predict/evaluate/aggregate/significance step, never copy-paste one dataset's logic into another's, and never write a one-off script for something a shared driver already does.** A 2026-09-27 full-project audit (below) found that the project mostly follows this well — no hand-rolled duplicate logic was found anywhere in eval/aggregate/significance/ladder code across any of the 8 tasks — but real, costly drift still crept in through **naming/numbering inconsistency** (the same shared script invoked via a differently-numbered wrapper in every dataset) and through **silent staleness** (a hardcoded path/list in a shared file quietly falling behind reality, e.g. a venv-rebuild trainer-shim list that named only 3 of 8 real files, or a cross-task heatmap script that silently blanked 5 of 8 tasks' columns after the `datasets/`→`benchmark/` restructuring and nobody re-ran it to notice). Two concrete habits that prevent this going forward:
- **New dataset → scaffold it, don't hand-write it.** `benchmark/create_dataset_structure.py` now generates the canonical `06_evaluate/` skeleton (below) for you.
- **After any change to a shared path/list/glob, run `benchmark/validate_standard_dataset_structure.py` (full repo, no args) before considering the change done** — it now enforces both the 9-subdir structure AND the canonical `06_evaluate/` numbering below (advisory warnings, not hard failures, since real per-dataset exceptions exist — but a new warning after your change is a signal to go look, not to ignore).

### Canonical `06_evaluate/` entry-point numbering (settled 2026-09-27)

Every dataset's `06_evaluate/` should — pending any real exception noted in that dataset's own comments — read as this exact sequence, skipping roles a dataset genuinely doesn't need, but never renumbering the roles it does have:

| Number | Role | Notes |
|---|---|---|
| `06_00` | `evaluate<_name>.py` shim | may legitimately be absent (e.g. on-harmony scores inline from its evaluate_run driver) |
| `06_01` | `evaluate_run.sh` (or `_own_run`/`_testset`/`_ispy2_run`/`eval_mandible_only` — suffix may vary, number must not) | the per-run evaluate driver |
| `06_02` | `aggregate_from_config.sh` | canonical config-driven aggregate for the dataset's own standalone headline table; documented exceptions use a purpose-built sibling instead of degrading the report — e.g. amos/sliver07's `aggregate_per_organ_from_config.sh`, which reports a per-organ (liver/kidney/spleen) breakdown `aggregate_from_config.py`'s macro-averaged single-number table can't represent, via its own shared script `00_03_evaluate/aggregate_per_organ_from_config.py` rather than a hand-rolled one-off |
| `06_03` | `significance_from_config.sh` | may legitimately be absent if no significance yaml config exists yet for that dataset — don't invent a config just to fill this slot, but don't leave the wrapper un-added once a config does exist either |
| `06_04` | `combined_modality_summary.sh` | only if the dataset pools ≥2 of its own training/test-contrast sources into one table (this includes an eval-only cross-dataset companion pooling its own upstream models, e.g. duke-breast-mri pooling ispy2's t1wce-trained + t2w-trained models — "combined" here means "this dataset's own multi-source table", not specifically "≥2 training modalities") |
| `06_1X` | `ladder_summary*.py` | causal-ablation ladder rung scripts, one per training contrast |
| `06_9X` | archived legacy script | a superseded hand-rolled script kept only because something still calls it — not a live entry point |

`06_XX_write_configs.py`-style config-generation utilities, one-off backfill/translation-compare scripts, and dataset-specific bespoke analyses (e.g. lesion-wise breakdowns) don't have a canonical number — just don't let them collide with the slots above. `validate_standard_dataset_structure.py` checks this automatically; a WARN from it means go verify, not "ignore, it's just advisory."

### Standardized output layout (so aggregation/significance/plots stay uniform)

```
8_results_<ds>/01_predictions/<model>/<train_contrast>/<nnUNet|auglab>/<RUN_ID>/fold{k}/<test_contrast>/
8_results_<ds>/02_metrics/<model>/<train_contrast>/<category>_<RUN_ID>/fold{k}/eval_all.csv
   → aggregated: results_summary.md, results_significance.md, results_heatmap_{dice,hd95}.png
```

Config-driven aggregation + significance consume YAML configs (`06_evaluate/configs/*.yaml`) so a whole dataset's 6-method table is summarized and tested with one command — **use these, do not compute ad-hoc tables/p-values.**

**Inline "sig. vs ref" column — auto-wired, not opt-in.** Every table `aggregate_from_config.py` produces (headline per-modality tables, checkpoint sweeps, HP sweeps, causal-ablation ladders, cross-dataset tables — anything driven by a `06_evaluate/configs/*.yaml`) gets an extra column after `all`, `sig. vs ref`, **automatically**, whenever the config's `runs:` list contains a run id matching `REF_SUBSTR = "auglabAug_v26_6_2_train050_val000"` (deliberately narrower than `"v26_6_2"` — that also matches the older non-auglab "v26_6_2 alone" method and its HP-tuning variants). No config change needed; override with `sig_ref: <exact run id>` for an ambiguous run list, or suppress with `no_significance: true`. Value = Holm-corrected one-sided ("ref better") macroΔ sign-flip p-value (same test/math as `significance_from_config.py`'s headline block, shared via `stat_tests.py`'s `macro_perm`/`holm`) of the OURS train050_val000 run vs that row, across ALL tested contrasts (equal weight per contrast, matching the `all` column's own estimand) — blank on the ref's own row. Drawn on the heatmap PNG too (masked/uncolored text column). This is on by default — don't hand-roll a second significance pass on top of it.

**Cross-training-modality tables (`combined_contrasts/`).** Every dataset trains each method TWICE, once per training modality (chaos: t1in/t2spir; brats2024-glioma: t1n/t2w; on-harmony: T1w/T2w; open-ms: flair/t1w). `combined_modality_summary.py` pools BOTH into one row per method — fold data tagged by modality so a 3+3 fold pool is one equally-weighted 6-fold mean (reuses `aggregate_from_config.py`'s `cross_fold_class_mean` unchanged) — writing `8_results_<ds>/02_metrics/<model>/combined_contrasts/01_results_{summary.md,heatmap_dice.png,heatmap_hd95.png}`, same inline `sig. vs ref` column as above (config field `ref:`, required). Driven by `<dataset>_combined_01_results.yaml` (lists each modality's `metrics_dir` + a `method_key → run_id` map) via each dataset's `06_XX_combined_modality_summary.sh`.

**Cross-DATASET task-level heatmap (`benchmark/01_commun_results/meta_task_heatmap_*`).** One level up again: `meta_task_heatmap.py` pools all 4 datasets' `combined_contrasts` tables into ONE table — rows = the 7-method suite, columns = one per dataset ("task", literally that dataset's own `all` value), + `overall` (equal weight per task) + `sig. vs ref` (same macroΔ sign-flip test, but stratified by TASK instead of contrast — each task's stratum = every held-out case's paired diff pooled across that task's contrasts+modalities — so the test matches the equal-weight-per-task `overall` column). Reuses the `*_combined_01_results.yaml` configs directly (no run ids re-specified a third time). Run via `bash scripts/evaluate/run_meta_task_heatmap.sh` (config: `scripts/evaluate/meta_task_heatmap.yaml`) — this is the single "which method wins overall" answer; re-run it after any headline run changes rather than eyeballing the 8 per-dataset tables.

Within `02_metrics/<model>/<contrast>/`, a non-headline result set gets its **own dedicated subdir** rather than mixing into the flat headline layout — established conventions so far: `checkpoint_comparison/` (checkpoint_best vs checkpoint_final sweep, all 8 training sets) and `ablations/` (the causal-ablation ladder studies — open-ms FLAIR, chaos T1in, chaos T2spir, brats2024-glioma T1n so far). Headline run dirs stay in the parent and are referenced unprefixed by any config that needs them (cross-dataset tables, `checkpoint_comparison/best_vs_final` configs); run dirs exclusive to the subdir's own study move into it and are referenced as `<subdir>/<run_id>`. `06_01_evaluate_run.sh`'s `METRICS_SUBDIR` env var (e.g. `METRICS_SUBDIR=ablations`, chaos + brats2024-glioma so far) routes evaluation output straight into a subdir like this — set it before evaluating a new ablation run rather than evaluating flat and moving the directory by hand afterward.

**The causal-ablation ladder** (one `06_1X_ladder_summary.py`-style script per dataset/train-contrast — `open-ms/.../06_12_ladder_summary.py`, `brats2024-glioma/.../06_13_ladder_summary.py`, `chaos/.../06_33_ladder_summary_t2spir.py` — reusing `00_commun_scripts/00_03_evaluate/significance_from_config.py`'s `load_run_cases`/`resolve_run_dir` for OOD extraction; don't hand-roll a new CSV loader for a new dataset's ladder): a baseline-anchored rung sequence — `baseline → +K-means intensity clustering → +label-remap → +Voronoi sub-parcellation (noise fill) → v26_6_2/PALETTE alone (identical partition, real-intensity fill) → +AugLab (val000) → +AugLab (val100)` — where each rung adds exactly one ingredient on top of the previous, so the OOD Dice/HD95 delta between two adjacent rungs is attributable to that ingredient alone. The rung 4→5 step (noise-fill → real-fill, partition otherwise unchanged) is the key one-variable test of whether texture preservation causally drives Dice: large (~+7 Dice, ~-4 to -5mm HD95) on texture-defined segmentation targets (open-ms MS lesions, brats2024-glioma tumor sub-regions) and near-zero/noise-level on boundary-defined ones (chaos organs, both T1in and T2spir) — texture preservation only pays off when the segmentation target actually depends on texture.

**⚠️ Cross-dataset ladder gotcha: group by TRUE held-out-contrast identity, not by source dataset.**
When a second dataset's own item happens to be the SAME contrast as training (e.g. a t1wce-trained
model tested on another dataset's own t1wce acquisition), that is cross-dataset generalization ONLY —
it is **not** OOD-contrast evidence and must not be pooled into an "OOD" bucket alongside a genuine
held-out contrast (breast: I-SPY2 t1wce-trained → I-SPY2's own t2w is real OOD; the *same* model →
duke-breast-mri's t1wce is cross-dataset-only, excluded from that bucket; the *t2w-trained* model →
duke-breast-mri's t1wce IS genuine OOD and correctly pools with I-SPY2's own analogous contrast). This
is a smaller-scale version of the Atlas-Liver-HCC in-domain-ambiguity problem below — get the
in-domain/OOD call right per training direction, don't default to "cross-dataset = OOD."
`ladder_cross_dataset_plot.py` (above) takes explicit named groups for exactly this reason — it does
not infer grouping from source dataset on its own.

Predicting/evaluating at a **non-default checkpoint** (project default is `checkpoint_best.pth`) is a uniform mechanism across all 4 dataset families: `predict_common.sh`'s `CHECKPOINT` env var and each dataset's `06_01_evaluate_*.sh`'s `CKPT_TAG` env var route non-best predictions/metrics to sibling `fold{k}/<tag>/` / `{category}_{run_id}_<tag>/` paths, never touching the default best-checkpoint data. Conclusion from the 2026-07-31 sweep (see each dataset's `checkpoint_comparison/best_vs_final_ckpt_summary.md`): `checkpoint_best` beats `checkpoint_final` overall, most clearly on open-ms — final-checkpoint results are not a drop-in upgrade.

---

## Analysis pipelines (paper: texture preservation + the causal ablation ladder)

**Correction 2026-08-28:** this section used to describe "two mechanism pillars." That framing is
retired — the paper explains *why* our method works via texture preservation (below) plus the causal
ablation ladder / dissociation table (see "The causal-ablation ladder" further down this file), not via
a second histogram-coverage analysis. Do not reintroduce "Pillar 2" language without a deliberate
decision to reopen it; see `paper/NARRATIVE.md` §3 for the same correction on the paper-narrative side.

- **Texture preservation.** `benchmark/02_tasks/brain_healthy/on-harmony/7_analysis_on-harmony/texture_analysis_lvl_1/`.
  Census / rank-transform |correlation| (contrast+inversion-invariant) + NMI, source vs each generated
  volume, per anatomical ROI (on-harmony: 31 SynthSeg classes). Grounding: that dir's `LITERATURE_REVIEW.md`.
  Same input-space method feeds the paper's NGF (Normalized Gradient Field) supplementary table.

**Histogram-manifold coverage — explored, permanently out of scope for the paper.**
`.../histogram_coverage_lvl_1/` (Coverage (Naeem 2020) macro-averaged + Vendi (Friedman & Dieng 2023) on
31-class regional histograms) and the open-ms 2-region **[lesion, overall]** variant both have finished,
grounded numbers on disk — but the honest result needs more justification than it's worth for CVPR, so it
will never be written up in the paper. Kept only as a reference pipeline, not a paper contribution;
don't cite its numbers as paper-supporting evidence. The still-earlier `.../contrast_manifold/` pipeline
(feature extraction → PCA/UMAP/PRDC/Vendi with a custom IND/OOD split and hand-rolled spread/hull/recall@Nx
metrics) predates this one and was already deemed unreliable before the histogram-coverage redo — both are
now equally out of scope, kept only for history.

---

## Common gotchas

- Whenever we run anything, we want to run a script with a simple bash command (which dispatches through `run_job` on this cluster — see "Cluster resource management" above). The script likely already exists in `5_scripts_*/` and if not, it should be added there — the structure should naturally guide you. We want to avoid running Python scripts directly from the command line without a proper script wrapper. Running existing scripts is good for consistency, and creating new scripts in the right place is good for organization and future reproducibility; it also ensures we don't debug the same thing repeatedly.
- `06_01_evaluate_run.sh` (chaos, brats2024-glioma) defaults `DATASET_ID` internally to the training set's *primary* contrast. Evaluating the *secondary* modality (e.g. brats t2w, chaos t2spir) without passing `DATASET_ID` explicitly silently scores against the wrong ground truth — no error, just wrong numbers. Always pass it explicitly for the non-primary modality.
- open-ms's evaluate script requires `CATEGORY` (`nnUNet`/`auglab`) explicitly — unlike chaos/brats, it does not auto-detect it. A wrong `CATEGORY` doesn't error either; it silently writes an empty, `_logs`-only metrics dir. Check `eval_all.csv` actually exists before trusting a run finished.
- When writing a one-off/filtered aggregate or significance config (e.g. to compare a subset of methods), always set an explicit `output_dir` pointing at scratch space. Omitting it makes `aggregate_from_config.py`/`significance_from_config.py` fall back to the dataset's real output dir and **silently overwrite the actual headline `*_summary.md`/`*_significance.md` files.**

---

## Cluster operations — hard-won, added 2026-08-02

**Slurm jobs go to TamIA, not Vulcan.** Standing default, not a per-case judgement. Vulcan's queue is
routinely backed up (small CPU jobs sitting behind multi-day jobs indefinitely) while TamIA schedules
the same work in ~1–2 min. Vulcan remains the filesystem home (repo, datasets, checkpoints, GitHub
pushes) — it is not the compute target.

**Remote commands over the TamIA relay MUST carry an explicit `cd`.** `ssh tamia '...'` starts in
`$HOME`, so `tar xzf -` extracts into `/home/p/paulh/benchmark/...`, not the repo. This bit repeatedly
on 2026-08-02 and **silently dropped a code fix**, so a completed multi-GPU run produced results with
one label missing and nobody noticed until the output table was read. Always:
```bash
cat script.sh | ssh tamia.alliancecan.ca 'cat > /tmp/s.sh && bash /tmp/s.sh'   # script starts with: cd /project/aip-jcohen/paulh/mri_synthesis_project
```
Piping a script file is also what avoids the nested-quoting mangling already documented above.

**Vulcan's login-node `/tmp` runs ~98% full** (other users' multi-GB dirs). Session scratchpads get
wiped mid-session and `tar` writes silently truncate to 0 bytes. **Stage intermediate data on
`$SCRATCH`, never `/tmp`.** Our own `/tmp` footprint is ~30 MB — the pressure is not ours to fix.

**No heavy compute inline on a login node, even when it "feels small".** A results aggregation held
~6 Vulcan login cores for 9+ minutes (~54 CPU-min against a ~10 CPU-min allowance) before being
caught. Wrap it in `run_job`. Related trap: **don't pipe a long-running command through `tail`** —
it buffers until exit, so a working job looks hung.

---

## The paper (`paper/`)

- **Source:** `paper/cvpr_format_latex/`, build with `pdflatex → bibtex → pdflatex ×2` (latexmk's
  auto-ordering has failed here). CVPR 2026 limit is **8 pages excluding references**; a
  `\label{endofmaintext}` before the bibliography lets you read the true main-text end page out of
  `main.aux`. Supplementary (`sec/X_suppl.tex`) is compiled in and does not count.
- **Narrative is mechanism-first as of 2026-08-02:** the claim is *"preserving texture matters where
  the target is defined by tissue appearance, not by an anatomical interface"*, with the SOTA numbers
  as supporting breadth. `paper/NARRATIVE.md` has the reframe rationale and — important — three
  citation traps that must not be "simplified" back in (never argue from "CHAOS is solved"; never
  infer texture-dependence from rater disagreement; **never claim MS lesions lack clear boundaries**,
  which the clinical literature contradicts).
- **Status + remaining work:** `paper/PAPER_TODO_20260802.md`.
- `sec/_suppl_tables.tex` is **generated from `benchmark/*/8_results_*`**, not hand-written —
  regenerate, don't edit.

---

## Atlas-Liver-HCC exclusion (2026-09-02)

**The whole atlas-liver-hcc extension — training set + its cross-dataset eval companions
(`lld-mmri-hcc`, `lld-mmri-malignant`, `liverhccseg`) — is excluded from the paper and from all
cross-dataset meta-evaluation/aggregation, permanently, not as a temporary hold.** Moved to
`benchmark/03_archive/{atlas-liver-hcc,lld-mmri-hcc,lld-mmri-malignant,liverhccseg}`. Do not re-add
without a deliberate decision to reopen this — the three reasons below are independent, any one of
them alone would justify the exclusion:

1. **Reproducibility/redistribution blocker.** LLD-MMRI-MedSAM2 (source for `lld-mmri-hcc` and
   `lld-mmri-malignant`) carries a custom Data Use Agreement, not a plain CC BY-NC license despite
   the badge at the bottom of its own README: verbatim, "you agree not to further copy, publish, or
   distribute any part of the LLD-MMRI dataset, except for internal use at a single site within the
   same organization... including annotations and cropped image parts." That rules out ever adding
   these two to the public git-annex upload (unlike on-harmony/open-ms/chaos/amos/sliver07 — see
   `project_git_annex_bids_upload` memory) and makes qualitative figures showing its images a real
   gray area. ATLAS itself (CC BY-NC-SA 4.0) and LiverHccSeg (CC BY 4.0, Zenodo) are both fine —
   this reason is specific to LLD-MMRI.
2. **The causal-ablation ladder's OOD contrast pool turned out to hinge on an ambiguous contrast.**
   Investigation (2026-09-01/02) found ATLAS's own CE-T1w training cohort is fat-suppressed, and
   lld-mmri-hcc's Dixon **out-of-phase** contrast reads as appearance-adjacent to it (partial fat
   cancellation lands close to ATLAS's own fat-suppressed appearance; the baseline, non-contrast-
   agnostic model ranks 1st of 7 methods on outphase — same pattern as the known in-domain
   `ce-art` column, nowhere else). In-phase was checked and ruled clean (recall collapses on
   in-phase, unlike outphase, ruling out the same appearance-similarity explanation). Whether
   outphase counts as in-domain or OOD **flips the ladder's sign**: real-fill vs. noise-fill
   macroΔ is −1.85 (p=0.0060, real-fill worse) with outphase included in the 4-contrast OOD pool,
   +2.21 (p=0.0064, real-fill better) restricted to the two unambiguous contrasts (t2wi, dwi), and
   the ambiguity got *worse*, not better, after adding `lld-mmri-malignant`'s 109 extra
   (non-HCC-but-same-cohort) cases as a robustness check. A causal claim that flips sign depending
   on a defensible contrast-inclusion call cannot be reported as a stable result.
3. **ATLAS's own training cohort is itself an uncontrolled mix of CE-T1w phases** (33 arterial / 10
   portal / 8 delayed / 7 unknown / 2 no-contrast-agent, per its own `patient_info_train.json`),
   which is why the `CE_T1w_all`/`in_domain_group` pooling scheme had to exist at all — a clean
   in-domain/out-of-domain split was never fully achievable for this dataset the way it is for the
   other four headline tasks (each single-source-modality with a genuinely held-out second
   in-house contrast).

**What was updated when this landed:** `scripts/evaluate/meta_task_heatmap.yaml` and
`paper/scripts/meta_task_heatmap_paper.yaml` (task entry removed, both regenerated — paper's
4-task `overall` is now 59.6% Dice / 18.6mm HD95 for Ours val000, was 56.7%/32.8mm as a stale
5-task figure); `paper/scripts/{compute_dissociation_pvalues,make_per_contrast_curves,
make_per_contrast_curves_with_ind}.py` (ATLAS-Liver-HCC panel/row removed, 5-panel figures
rebuilt as 4-panel); `sec/4_experiments.tex`, `sec/X_suppl.tex`, `sec/_suppl_data.tex` (headline
table, dataset roster, causal-ablation dissociation table + its now-retracted `+2.09, p=3.5e-3`
row, val000-vs-val100 table, and every paragraph of Liver-HCC-specific prose — all removed or
renumbered for 4 tasks); PDF rebuilt clean (17 pages, was 18). `scripts/cluster/tamia_env_*` for
the 4 archived datasets were left in place (harmless, orphaned).

---

## Stroke task: ISLES 2022 (`brain_stroke/isles2022`) — onboarded to preprocessed, NOT yet trained (2026-10-04)

Chosen by a `find-a-new-task` pass for "pathology, intra-tissue (texture-defined)" (brain MS + glioma already
covered). Source: Zenodo 10.5281/zenodo.7153326 (Sci Data 9:762). Paul approved use 2026-10-04 **despite** the
zip's LICENSE clause "no redistribution without written agreement of the ISLES'22 team" (badge says CC BY 4.0;
same kind of restriction as LLD-MMRI) → **never upload to git-annex or show its images in public figures without
that written agreement.** BIDS leaf `1_BIDS_isles2022/stroke-brain-isles2022` (name provisional).
- **Contrasts:** train `dwi` (Dataset140) and `flair` (Dataset141); every test case scored on dwi and flair. **ADC (in the release, derived
  from DWI) is deliberately NOT used anywhere** (Paul, 2026-10-04: DWI is the contrast the labels were drawn on, ADC nearly redundant); it stays
  in the BIDS tree only as a faithful copy. All three contrasts of a subject live together in ONE BIDS tree; the two nnU-Net datasets are views of the
  same subjects with identical case ids and one shared split (`02_nnunet/02_04_verify_heldout.sh` proves no test subject appears in any train/val
  fold of either dataset, nor in preprocessing).
- **Usable N = 246 of 250:** 3 empty masks (0150/0151/0170) + isles2022_0007 (FLAIR FOV holds only 53% of the
  lesion). FLAIR is on its OWN native grid in all 250 raw cases and is resampled onto the DWI grid by
  `02_nnunet/02_01_convert.py`; DWI/mask share one LAS grid. Partition: 47 test (volume-stratified) + 199 pool,
  3 folds (`01_create_splits/01_01_create_splits.py`). Case ids `isles2022_<NNNN>` carry no contrast suffix.
- **Built:** BIDS, nnU-Net raw + preprocessed (3d_fullres, patch 80x96x80 @2mm, batch 6), splits, trainer package
  `isles2022/trainers` + `02_nnunet/ISLES2022Trainers.py` shim (installed in Vulcan's venv; discovery verified),
  04_* wrappers (6 methods x 2 contrasts + ladder rungs 2-5; **no rung-6 PV wrappers yet**), `04_23_tamia_pack_all.sh`,
  `scripts/cluster/tamia_env_isles2022.sh`, orientation QC PNG in `9_tests_isles2022/`.
- **Training LAUNCHED 2026-10-04 on TamIA** (2000 epochs): 6 whole-node pack chains, jobs 504872-504884, 60 fold-jobs (10 runs/contrast: 6 headline methods
  [OURS DualVal also yields the val100 mirror] + ladder rungs 2-5; folds 0 1 2). Sizing probe (contended s/epoch at 3 folds/GPU): baseline 16.4,
  auglab_default 19.6, synthseg_noEM 28.8, srcsm 32.5, synthseg_EM 42, DualVal ~85; predicted max GPU wall 18.8 h/pack. RUN_IDs in
  `$SCRATCH/isles2022/_packruns/RUN_IDS.tsv` (TamIA). **Predict + eval are queued with Slurm dependencies** (`05_predict/05_27_tamia_queue_predict_eval.sh`
  → controller job `05_28` → predict packs → `06_evaluate/06_08` eval job; status in `$SCRATCH/isles2022/post_training_status.txt`, `DONE_eval.txt`).
  Manual tail afterwards: fetch metrics to Vulcan → `06_05_write_configs.sh` → `06_07_run_all_aggregation.sh` → `meta_task_heatmap.yaml` entry.
  Whole path as a skill: `setup-train-predict-eval-scripts`. TamIA has unpushed hand-copied files (see memory `project_isles2022_training_launch_20261004`).
---

## Pancreas task: PanSegData (`pancreas_disease/pansegdata`) — training launched on KILLARNEY, NOT yet evaluated (2026-10-04)

Found by a `find-a-new-task` pass for "boundary-defined, easy (joints/bones/muscles → widened)": SPIDER (lumbar spine) was dropped by Paul (spine already covered by healthy-spine-tum — avoid
spine/cord), knee/muscle/cardiac candidates failed on access/size/labels. Source: OSF kysnj (Zhang et al., Med Image Anal 99:103382, doi:10.1016/j.media.2024.103382), MRI part only.
**License CC BY-NC 4.0** (non-commercial; verified on the OSF page, bundled LICENSE.txt and the paper) → needs its own NC tag if ever put in the CC-BY git-annex upload. Cohort = patients referred for pancreatic
cystic lesions / suspected PDAC (not healthy); inter-observer kappa only 0.62-0.64; the source does not say whether lesions are inside the mask (cysts appear inside it). Honest caveat: **not the clean "easy" boundary task asked for.**
- **Contrasts:** `t1wce` (venous-phase contrast-enhanced T1, Dataset150) and `t2w` (Dataset151); each scan has its OWN mask → each test item is scored against its own mask. 385 T1 + 382 T2 scans, 405 subjects, 362 with both.
  T1<->T2 pairing is by original case name (the zips' anonymised ids are independent numberings), supported by a same-site null test (non-MCF clearly the same patient).
- **EXCLUDED: all of center MCF (Mayo Florida, 150 paired subjects) + the 43 single-contrast subjects.** MCF defect (source data): venous-T1 arrays stored rotated 180° in-plane vs their headers (and many also S-I
  reversed); the 66 mismatching label headers (RAI/RAS vs LPS) show the same rotation; even after a CNN-detected 180° fix, 3/3 MCF subjects inspected at full size had the mask over the vertebra/paraspinal muscles. A rescue could
  not be verified. Alternative for Paul: rescue by registration/manual curation (≈41% of the paired data). Evidence trail: `5_scripts_pansegdata/02_nnunet/02_05..02_11`, `0_raw_pansegdata/orientation_fix.tsv` (unused), README of the BIDS leaf.
- **Usable N = 212** (NYU 161, NWU 19, AHN 17, MCA 15): test 42 (proportional by center, volume-stratified) / pool 170 / 3 folds (113-114 train, 56-57 val). All kept scans share source orientation LPI. Case ids `pansegdata_<site><NNNN>`.
  Held-out proof `02_04_verify_heldout.sh`: ALL CHECKS PASSED. Questionable but kept: ahn0011 (T1+T2), ahn0001 T2 — judge by CV Dice outliers.
- **Epochs 2000** (my decision by policy analogy: disease cohort, ≤~300 training cases; not a number Paul gave). Sizing probe on Killarney (2000-epoch single-fold wall, H100 / L40S h): baseline 4.9/12.6, auglab_default 6.2-7.6/14.5,
  srcsm 6.4-6.8/13.8, synthseg_noEM 6.6-7.4/14.7, synthseg_EM 8.0-8.2/17.8, OURS DualVal 16.8-17.1/29-31. Ladder rungs 2-5 were NOT probed (16 h/23.5 h limits are guesses).
- **Compute: ALL on Killarney (Paul, 2026-10-04, exceptional).** Per-GPU `run_job` fold jobs through the shared wrappers (no whole-node packs; the scaffolded TamIA scripts 04_23-25 / 05_26-28 / 06_08 /
  `scripts/cluster/tamia_env_pansegdata.sh` are unused). `04_26_killarney_sizing_probe.sh` + `04_27_probe_report.sh`; `04_28_killarney_launch_all.sh` (H100, per-method time limits) launched 60 fold jobs 2026-10-04 21:38-21:49
  (RUN_IDs `8_results_pansegdata/_launch/RUN_IDS_killarney_launch1.tsv`); `04_29_killarney_status.sh [--problems]` = real progress from `fold_N/training_log_*.txt`. A job that hits its limit resumes by re-running its wrapper with the SAME RUN_ID.
  **Predict + eval are queued:** `05_predict/05_30_killarney_queue_predict_eval.sh` → controller job 5942062 (`afterany` all 60 training jobs) → `05_29` (refuses on incomplete training, pins RUN_IDs, submits predicts via the roster drivers, queues
  `06_evaluate/06_09_killarney_eval_stage.sh`; status in `8_results_pansegdata/post_training/`). Manual tail afterwards: `06_05_write_configs.sh` → `06_07_run_all_aggregation.sh` → `scripts/evaluate/meta_task_heatmap.yaml` entry → rsync data to Vulcan for durability.
- Two probe jobs once died at startup on node kn001 (not reproducible; same code ran fine on rerun): after any launch confirm EVERY fold's log reaches epoch ≥1 (`04_29 --problems`) and resubmit dead folds early.
- Scaffolder fixed this session: it now also rewrites the label name in 06_00/06_01 comments (was leaving `lesion`).

---

## Breast task: I-SPY2 (training) + duke-breast-mri (eval) — AMBL archived (2026-09-13)

**Current roster for the breast task: `ispy2` trains, `duke-breast-mri` is the cross-dataset eval
companion. `ambl` is archived at `benchmark/03_archive/ambl` — do not re-add without a deliberate
decision to reopen this**, matching the Atlas-Liver-HCC precedent above. AMBL was I-SPY2's original
training-set candidate and later its planned external test set (see memory
`project_ambl_breast_onboarding_launch`/`project_ispy2_becomes_training_set` for the full pivot
history) — both roles are now retired. Proximate reason for the final archival: AMBL's own
`0_raw_ambl/README.md` documents that its DICOM-SEG segments a radiologist-delineated **every**
enhancing/suspicious finding, labeled `Tumor` OR `Benign` per segment, and the BIDS conversion
**unions both into one binary lesion class** — a "malignant-only" label repeated in at least one
ladder script's docstring was never independently verified against this and was wrong. I-SPY2
(Functional Tumor Volume mask, neoadjuvant-chemo trial, enrollment requires biopsy-proven invasive
cancer) and duke-breast-mri (its own download script: "expert tumour mask", cancer-treatment
patients) are both clean malignant-only cohorts and are unaffected by this.

**Licenses, verified 2026-09-02/09-13:** I-SPY2 = **CC BY 4.0** (verified 3 ways per the pre-flight
checklist above, see its own README). duke-breast-mri (downloaded via MAMA-MIA/Synapse
`syn60868042`) = **CC BY-NC 4.0** — confirmed via the TCIA wiki collection page and MAMA-MIA's own
GitHub README, which states Duke-Breast-Cancer-MRI is specifically why the combined MAMA-MIA release
is CC BY-NC rather than plain CC BY (the other 3 MAMA-MIA source cohorts are CC BY). This is **not**
a redistribution-blocking custom DUA like the retired LLD-MMRI case — CC BY-NC permits sharing/
adapting derivatives for non-commercial purposes with attribution, it only blocks commercial use — but
it does mean duke-breast-mri needs its own NC tag if it's ever folded into the CC-BY git-annex upload
(`project_git_annex_bids_upload` memory), not blended in as if plain CC BY.

**I-SPY2 training config, for reference:** 1000 epochs (a real deviation from the other datasets'
epoch counts in the table above — not yet confirmed as a deliberate choice vs. a leftover default,
flag if revisited), folds 0/1/2, both training contrasts (t1wce/t2w). I-SPY2's raw DCE acquisitions
split into unilateral-crop and bilateral-FOV cases per site; both FOV variants are generated per
patient rather than dropping one, which is why the two training contrasts' held-out test-set sizes
differ (102 vs. 168 cases) despite sharing the same 84 held-out patients — this has not been
independently checked for double-counting/bias, flag if it becomes load-bearing for a paper claim.

**Unilateral-crop is the standard duke-breast-mri evaluation strategy (Paul's decision, 2026-09-17)
— the ONLY reported one, bilateral is retired.** duke-breast-mri's test images are ~100%
bilateral-width; I-SPY2's own t1wce training pool is only 18% bilateral (82% unilateral-cropped —
a hard acquisition ceiling, not a processing choice, see below). Evaluating on full bilateral
images was a real, substantial confound: every arm's absolute Dice rose once tested on FOV-matched
unilateral crops (`benchmark/02_tasks/breast_cancer/duke-breast-mri/5_scripts_duke-breast-mri/02_nnunet/
02_03_derive_unilateral_crop.py` — axis-0-only lesion-side half crop, deliberately NOT reusing
I-SPY2's own empirically-fit anterior-posterior window since that's calibrated to I-SPY2's own
site-specific VOLSER protocol), and the Ours-vs-auglab_default headline gap that originally
motivated this investigation shrank substantially on two arms and flipped to a near-tie on the
other two. New eval items: `t1wce_uni`/`precontrast_uni` (metrics land under
`8_results_duke-breast-mri/02_metrics/ispy2_model/<contrast>/{t1wce_uni,precontrast_uni}/`, same
`DUKE_ITEM`/`METRICS_SUBDIR` routing convention as `precontrast`). `ispy2_combined_01_results.yaml`'s
`duke_t1wce`/`duke_precontrast` source columns now point at the `_uni` dirs. **The original
bilateral tables/ladders are archived at `benchmark/03_archive/{duke-breast-mri,ispy2}
_bilateral_eval_20260917/`** — fully out of the active `benchmark/02_tasks/breast_cancer/ispy2/`/`benchmark/02_tasks/breast_cancer/duke-breast-mri/`
trees (same spirit as the AMBL/atlas-liver-hcc archival precedent, so nobody cites the stale
bilateral numbers by accident), not deleted — see each archive's own README for exactly what
moved and why. The underlying raw per-fold `eval_all.csv`/predictions were left in place (not
archived), only the aggregate/summary/ladder outputs moved.

**Two more eval-only breast companions (added 2026-09-30, session Breast_cancer_3, run on Vulcan):**
- **`ispy1`** — I-SPY1/ACRIN 6657 via MAMA-MIA (Synapse; the SAME expert primary-lesion protocol as
  duke's masks), BIDS `onc-breast-ispy1`, CC BY 3.0. 167/171 cases (3 bilateral cancers + 1 implant
  excluded by a rule fixed before predicting). Natively sagittal/unilateral 1.5T. Items `t1wce`,
  `precontrast`. Pathology/size match verified (T3 ≥3 cm, pre-NAC; mask volume median 14.9 vs I-SPY2
  training 16.1 ml).
- **`acrin6698`** — TCIA ACRIN-6698 (the I-SPY2 DWI sub-study), BIDS `onc-breast-acrin6698`, CC BY 4.0.
  **0/385 patient overlap with our ispy2 subjects (verified by ID).** T0 baseline study only (select by
  StudyDesc `_T0`, not by "TrT0" series names, which also occur at T2). Item `dwi` = b800 trace with the
  trial's "Whole Tumor Manual" DWI ROI — a NEW contrast, held-out for BOTH training directions, but drawn on
  DWI/ADC (not FTV semantics), so always its own column. Bilateral axial → lesion-side crop via the shared
  `00_commun_scripts/00_00_utils/unilateral_crop.py`.
- Wired into `ispy2_combined_01_results.yaml` (columns `ispy1_t1wce`, `ispy1_precontrast`,
  `acrin6698_dwi`; combined + project meta heatmap regenerated, backups `*.bak_20261001_pre_ispy1_acrin`).
  **Per Paul (2026-10-01) ispy2's HEADLINE ladders (`06_10`/`06_11` → `ablations/`, read by
  `paper/scripts/compute_dissociation_pvalues.py`) now pool the new cohorts** (ispy1 t1wce only for the
  t2w-trained ladder — same-contrast rule), **pooled by TRUE contrast (Paul, 2026-10-01)**: `run_ladder(
  ood_groups=...)` → `run_ladder_grouped` (shared engine, opt-in): every case of a contrast from every cohort in
  ONE pool ("all pre-contrast together"), OOD = equal weight per contrast, **OOD-only report**, significance
  patient-merged (`_patient_key` collapses I-SPY2's `_uni`/`_bil` FOV variants — 168 t2w cases = 84 patients),
  per-contrast curves in `ladder_<tc>_cross_dataset_per_contrast.png` (no `_per_contrast.png`). Same case-pooled
  `contrast_groups` (lists) in `ispy2_combined_01_results.yaml`. Result (with the A-P-cropped `*_uniap` items,
  final 2026-10-01): real-fill Δ OOD Dice t1wce-trained −0.85 (p=0.066; HD95 −3.2 mm, p=8e-6), t2w-trained
  +0.50 (p=0.017; HD95 −1.8 mm, p=2.4e-4); combined table Ours 41.2 vs auglab_default 40.1 Dice (p=9.5e-9),
  HD95 57.7 vs 57.1 (n.s.). ⚠️ `paper/scripts/make_per_contrast_curves*.py` read in-domain fields the
  OOD-only grouped JSON no longer carries — check before re-running them. Pre-extension ladder outputs:
  `ablations/*.bak_20261001_pre_ispy1_acrin`.
  I-SPY1's own ladders show real-fill significantly NEGATIVE. MAMA-MIA was never formally excluded (the 2026-09-02 caution was only against
  MERGING cohorts for training); never use MAMA-MIA's own ISPY2 subset — it contains 264 ACRIN-6698
  patients. NACT-Pilot (64) was downloaded but deliberately not onboarded.

**A-P crop standard (Paul, 2026-10-01): full-chest breast test sets are cropped A-P too, not only L-R.**
duke-breast-mri's `*_uni` and acrin6698's `dwi` kept the whole thorax (~330-350 mm A-P) while I-SPY2's
unilateral training crops are ~174 mm (breast + a little chest wall). New standard items: duke
`t1wce_uniap`/`precontrast_uniap`, acrin6698 `dwi_uniap` — shared `00_commun_scripts/00_00_utils/derive_ap_crop.py`
+ `unilateral_crop.ap_skin_window` (27 mm air in front of each case's own anterior skin + 147.4 mm behind,
the measured I-SPY2 unilateral-training geometry; extended, never clipped, to contain the lesion; ACRIN skin
found on b0 since fat is dark at b800). ispy1 is natively unilateral sagittal (200 mm A-P) — untouched.
L-R-only outputs archived: `benchmark/03_archive/{duke-breast-mri_lr_only_unicrop_eval_20261001,
acrin6698_lr_only_crop_eval_20261001}/`. Effect: every method's absolute Dice rose and HD95 collapsed
(~100 → ~30-45 mm); **duke's same-contrast fill-swap (+6.75 Dice, the paper's supplementary "largest effect")
was a full-chest-FOV artefact — it is −0.71 after the A-P crop**; `paper/scripts/compute_dissociation_pvalues.py`'s
SUPPLEMENTARY path still points at the archived `t1wce_uni` ladder.

**Standing policy (Paul, 2026-09-17): every task's `combined_contrasts` table should be
cross-dataset, not just cross-modality** — i.e. it should always pool in that task's external
cross-dataset eval companion (the way ispy2's already pools in duke-breast-mri), not only the
task's own training-modality pair. Worth checking other tasks' `*_combined_01_results.yaml`
configs against this standard if revisiting them.

---

## CHAOS cross-dataset FOV: CROP-before-predict is the standard (Paul, 2026-10-03)

Chaos-trained models are tested on amos (CT+MRI) and sliver07 (CT) full-torso volumes. The CHAOS
axial-FOV restriction is now applied at the **prediction level**: each test volume is cropped to the
CHAOS-equivalent slab *before* `nnUNetv2_predict`, so the model only sees that slab. The old way —
mask pred+GT outside the slab at *scoring* time, network sees everything — is **legacy**.
- Driver: `benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh` (env-driven, TamIA;
  geometry in `00_00_utils/fov.py`; margins `chaos/.../06_evaluate/chaos_fov_margins.json`). Metrics land in
  `<dataset>/8_results_<dataset>/02_metrics/chaos_model/<contrast>/fov_crop/`.
- `chaos_combined_01_results.yaml` and `cross_dataset_{t1in,t2spir}_01_results.yaml` read `.../fov_crop`
  for amos + sliver07 (the ONLY three configs that read those two datasets). Chaos's own contrasts
  (t1in/t2spir/t1out/own CT) are untouched by cropping.
- The masked metrics (unprefixed run dirs under `<contrast>/`) are kept as legacy, not read by any
  roll-up; the amos/sliver07 `06_01_evaluate_run.sh` `FOV=1` path still writes there. Pre-switch outputs:
  `*.bak_20261003_pre_fovcrop` next to each regenerated table. Don't re-point configs back without a reason.
- **The chaos ladders are to be CROSS-DATASET too (Paul, 2026-10-03)**: amos + sliver07 (cropped) pooled into
  the OOD pool by TRUE contrast. `06_34`/`06_35` are wired to amos+sliver07 `fov_crop` sources (commit 8f7bcff,
  with a guard that refuses to run unless every rung resolves in every source). The crop-before-predict
  predictions for rungs 2-4 (kmeans/label_remap/voronoi) come from `06_36_fov_crop_ladder_rungs.sh` (roster
  `chaos_fov_crop_ladder_runs.txt`, shared driver in RUNS_FILE mode; reuses the headline crops via CROP_REUSE_DIR).
  **DONE 2026-10-03:** both ladders regenerated cross-dataset with every rung present in every source (t1in
  14:10; t2spir 15:37 via `06_37_run_ladder_summary_t2spir.sh`, with the RETRAINED rung 5
  `chaos_t2spir_v26_6_2_train050_val100_20261003_123018` — rungs 2-5 now all AugLab GPU synth p=0.5 + flip,
  verified from each run's `transform_params_gpu_used_for_training.json`; predictions made after training ended).
  Fill-swap 4→5: t1in +1.43 Dice (p=3e-13), **t2spir +5.11 (p=7.5e-22)** — the old −0.88/−1.14 t2spir value was
  the pre-AugLab rung-5 implementation confound, not a property of the task. The guard stays: the engine silently
  skips a source that doesn't resolve for a rung, so a partial set would compare different OOD pools across rungs.
- Effect (combined table, Dice/HD95): Ours val000 85.0/30.6 → 85.9/22.7; srcsm sig. vs Ours p 0.106 → 0.043.
  Full per-method before/after in `benchmark/01_commun_results/fov_crop_vs_mask_2026-08-01.md`.
- Regenerating on TamIA needs a path mirror (healthy-spine-tum's config hardcodes Vulcan's absolute path;
  rewrite it in the mirror copy or that task's column silently goes blank).

---

## Cleanup notes (read before deleting anything in the repo root)

**Untracked but load-bearing — do NOT `git clean`:**
- `benchmark/01_commun_results/meta_task_heatmap_summary.md` — **the source of the paper's main
  results table.** Untracked. Losing it loses the headline numbers.
- `benchmark/00_commun_scripts/00_04_analysis/label_cue_importance/` — shared boundary-cue analysis
  (its `FINDINGS.md` / `LITERATURE_REVIEW.md` carry the reviewer-rebuttal material; deliberately kept
  out of the paper).
- `benchmark/*/5_scripts_*/05_predict/05_4X_predict_*_kmeans*` — the ablation-ladder rungs that produce
  the paper's central dissociation.
- `paper/**/*.bak_20260802` — intentional pre-edit backups from the 2026-08-02 paper session.

**Genuine cleanup targets (verify, move to `toDelete/`, then delete — never `rm -rf` with a glob):**
- `${METRICS_ROOT}/` — a directory literally named after an **unexpanded shell variable**. It holds 4
  real `*_significance.md` files misfiled by a bug. Check the correct `8_results_*/02_metrics/...`
  paths already contain them before deleting; otherwise move them there first. **Fix the script that
  wrote them** — an unquoted/unset `METRICS_ROOT` will do it again.
- `benchmark/01_commun_results/"ours_vs_best_other_train050_val000 copy.md"` — duplicate (note the
  space in the filename).
- `paper/cvpr_format_latex.stale_bak_1783838883/` (2.9 MB), `CLAUDE.md.bak.*`, `.scratch_analysis/`
  (29 MB of one-off probes).
- `scripts/cluster/{tamia_env_ambl.sh,tamia_env_ambl_ispy2_target.sh,_ambl_run_aggregation.sh}` —
  orphaned now that AMBL is archived (see "Breast task" section above). Same call as the archived
  liver-HCC datasets' `tamia_env_*` files: harmless, leave in place unless doing a dedicated pass.

**Correction (2026-09-27): the ladder-format inconsistency below is stale, already fixed.** This note
used to say the four ablation ladders didn't share a summary format (BraTS T1n/CHAOS T2spir having an
explicit `ablations/ladder_summary.md` rung table, Open-MS FLAIR/CHAOS T1in only having per-contrast
`*_summary.md` requiring hand-averaging) and that this caused a wrong HD95 number in the paper's
central table. That divergence was root-caused to `benchmark/00_commun_scripts/00_03_evaluate/
ladder_ood_common.py`'s extraction (2026-08-03) already having fixed this: `run_ladder()`/
`run_ladder_cross_dataset()` unconditionally write `ablations/ladder_summary.md` with the explicit
rung table for every caller — verified 2026-09-27 that all four datasets' ladder scripts
(`06_1X_ladder_summary*.py`) already call into this shared engine and already have the rung-table
`ladder_summary.md` on disk (open-ms: `flair/ablations/`, `t1w/ablations/`; chaos: `t1in/ablations/`,
`t2spir/ablations/`). No code change was needed — this note itself just hadn't been updated after the
2026-08-03 consolidation.
