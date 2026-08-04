# Cross-Cluster Monitoring Dashboard — mri_synthesis_project

Tracks training / prediction / evaluation completeness across all four
datasets (CHAOS, open-ms, brats2024-glioma, on-harmony) on **both** Alliance
Canada clusters (vulcan, killarney), plus SLURM queue, storage quota and
fairshare health, with an interactive per-case Dice comparison plotter.

Project root (identical path on both clusters, but **not** a shared
filesystem — each cluster has its own copy of the data):

```
/project/aip-jcohen/paulh/mri_synthesis_project
```

All dashboard code lives under `dashboard/bin/` inside that root (uploaded
identically to both clusters). Status JSON and logs live under
`dashboard/` and `dashboard/logs/`.

## Why it's built this way

vulcan and killarney do not share a filesystem, and Alliance Canada enforces
interactive Duo MFA on every SSH login — including cluster-to-cluster
hops — so there is no way to have one process reach directly into the
other cluster's data. The design instead runs a small scanner locally on
**each** cluster's login node, and has vulcan (which hosts the visible
dashboard) pull killarney's results over an already-authenticated SSH
control socket:

```
killarney: scan_cluster.py (local)  ---scp over SSH ControlMaster--->  vulcan
vulcan:    scan_cluster.py (local) --+
                                      \
                                       --> combine_status.py --> status_combined.json --> Streamlit app
```

Because the control socket requires one interactive Duo approval to open
and then stays alive for the `ControlPersist` window configured in
`~/.ssh/config` on vulcan (`ControlPersist 30d`), the automated sync only
works for as long as that socket is alive — see **Killarney socket
refresh** below.

## Components (`dashboard/bin/`)

| File | Runs on | Purpose |
|---|---|---|
| `dashboard_common.py` | both | Shared constants (dataset/contrast maps, run-dir regex, path helpers) |
| `scan_cluster.py` | both | Scans local run directories + `squeue`/`diskusage_report`/`sshare`; writes `status_<cluster>.json` |
| `killarney_scan_loop.sh` | killarney | Long-running detached loop that re-runs `scan_cluster.py` every 15 min (killarney login nodes disable per-user crontab) |
| `dice_loader.py` | vulcan (imported by `app.py`) | Loads `eval_all.csv` per fold/run for the Dice plotter; cached by path+mtime |
| `sync_remote_status.sh` | vulcan | Pulls killarney's `status_killarney.json` via `scp` over the persistent SSH control socket; fails soft |
| `combine_status.py` | vulcan | Merges both clusters' status into `status_combined.json`; computes the completeness matrix and missing-predictions alerts |
| `refresh_all.sh` | vulcan | Orchestrator: scan vulcan locally → sync killarney → combine. Run by cron every 15 min |
| `refresh_killarney_socket.py` | vulcan | Helper that opens an interactive `ssh` to killarney (stdlib `subprocess` only, no third-party deps) so you see the native Duo menu in your terminal and can select the push option and tap Approve on your phone — must be run interactively, e.g. `cd` into the project root first (see below) |
| `app.py` | vulcan | The Streamlit dashboard itself |

## Launching the dashboard

**Important — `vulcan.alliancecan.ca` is DNS round-robin across TWO login
nodes** (`vulcan1` = 129.128.190.50, `vulcan2` = 129.128.190.51). A plain
`ssh vulcan.alliancecan.ca` or `ssh -L ... vulcan.alliancecan.ca` can land on
either one essentially at random per connection. Since Streamlit only binds
to `127.0.0.1` on whichever node it's running on, a tunnel that lands on the
"wrong" node just hangs/refuses — this is the #1 cause of "I can't connect to
the dashboard" even when the app itself is perfectly healthy. **The fix
deployed here: run an identical Streamlit process on BOTH login nodes**, so
whichever one your SSH session round-robins to, port 18765 is already live.
Both processes read the same shared `/project` files, so they always show
identical data — there is no split-brain risk, just two front doors to the
same house.

Each process is bound to `127.0.0.1` only (never exposed on `0.0.0.0`) and
kept alive across SSH disconnects via `setsid`+`nohup`+`disown`.

Port **8765** was already in use by another process on these login nodes, so
the dashboard runs on **18765** instead.

```bash
# Run this once on EACH login node (vulcan1 and vulcan2) — e.g.
#   ssh vulcan.alliancecan.ca   (repeat the ssh until you land on the other node,
#                                 or `ssh vulcan1`/`ssh vulcan2` directly from within
#                                 an existing vulcan session)
cd /project/aip-jcohen/paulh/mri_synthesis_project
mkdir -p dashboard/logs
nohup setsid .venv/bin/streamlit run dashboard/bin/app.py \
    --server.port 18765 --server.address 127.0.0.1 \
    --server.headless true --browser.gatherUsageStats false \
    > dashboard/logs/streamlit.log 2>&1 < /dev/null &
disown
```

Check it's alive (run on both nodes independently — health on one node says
nothing about the other):

```bash
pgrep -af streamlit
curl -s http://127.0.0.1:18765/_stcore/health   # expect: ok
```

Stop it (run on both nodes if you want it fully down):

```bash
pkill -f "streamlit run dashboard/bin/app.py"
```

If it ever needs restarting on a different port (8765 free again, or
18765 taken), just change `--server.port` in the launch command above and
in your tunnel command below to match — **on both nodes**, so they stay
consistent.

## Viewing it in a browser (SSH tunnel)

The login node has no public web access, so forward the port to your
local machine:

```bash
ssh -L 18765:127.0.0.1:18765 vulcan.alliancecan.ca
```

Leave that SSH session open, then browse to:

```
http://127.0.0.1:18765
```

**If this hangs or refuses to connect, just retry the `ssh -L ...` command** —
DNS round-robin means a fresh connection attempt has a good chance of landing
on the other login node, and since Streamlit now runs on both, a retry alone
usually fixes it. If BOTH nodes fail, then it's a real outage — check
`pgrep -af streamlit` and `dashboard/logs/streamlit.log` on each node.

## Dashboard tabs

1. **Overview** — SLURM queue table (pending/running counts) across both clusters. Each
   row also carries a **stage** column splitting queued/running jobs into `train`
   (`fold<N>_...` job names, parsed by `scan_cluster.py`) vs `predict`/`eval`
   (dataset/method-prefixed job names, via `_job_stage()` in `app.py`), plus a
   **Training / Predict+Eval** summary metric so you can tell at a glance which part
   of the pipeline the cluster is spending allocation on. Training rows additionally
   show **progress** (`current_epoch/num_epochs (pct%)`, sourced from the training
   log's last logged epoch) and **eta** (steady-state estimate: remaining epochs ×
   last epoch duration) — both computed in `combine_status.py`'s
   `attach_epoch_progress()` and left blank for predict/eval rows or training rows
   with no epoch data yet (e.g. still queued).
2. **Completeness matrix** — one row per dataset/contrast/category/method group (most recent run per group), with train/predict/eval progress bars; filterable by dataset. Each row now also carries a **flags** column (🛑 stalled / 🧩 orphan / ⏳ queued — see below), plus `train_stalled` and `train_pending` counts alongside `train_done`/`train_in_progress`.
3. **⚠️ Attention needed** (badge shows the live count, e.g. `⚠️ Attention needed (5)`) — combines two *alarm* panels plus one *informational* panel:
   - **Missing predictions** — scanning *every* run (not just the latest per group), so a superseded/older run with a missing prediction still surfaces. A fold whose training job is merely queued (see ⏳ below) is **never** counted here — it hasn't finished training yet, so "predict/eval not done" isn't actionable. Each row also carries a **status** badge (🟢 running / 🟡 queued / 🔵 other) when `combine_status.py`'s `_inflight_job_for()` finds a live squeue job whose name matches that dataset+method+fold's predict/eval submission pattern — so a genuine gap (blank status, needs submitting) is distinguishable from one that's already been submitted and just hasn't finished (badge shown, no action needed). The summary line splits the count into *N need submitting* vs *N already queued/running*. Matching is per-dataset (job-naming conventions differ across chaos/brats2024-glioma/open-ms/on-harmony — see `PREDICT_JOB_PREFIXES`/`EVAL_JOB_PREFIXES` in `combine_status.py`) and best-effort: eval jobs that embed the full run directory name in their job name get an exact match, while predict jobs that only embed method+fold (no run timestamp) get a lower-confidence prefix match — treat the badge as an informational hint, not an authoritative join.
   - **🛑 Stalled-training alert list** — a fold is flagged *stalled* when Slurm's queue still shows its job `RUNNING`, but its `training_log_*.txt` hasn't been appended to in over `STALL_THRESHOLD_SECONDS` (30 min, `dashboard_common.py`). Each row shows current/expected epoch and how long the log has been silent — this is the "job is alive but nnU-Net has hung/crashed silently" case that a plain squeue glance won't catch.
   - **⏳ Queued trainings** (informational, non-alarming) — a fold whose Slurm job is `PENDING` (not yet scheduled to run). Previously, a fold in this state with a stale `checkpoint_latest.pth` left over from before a preemption/requeue was silently misclassified as `train: done`, which made `combine_status.py` flag it as a false-positive "missing prediction" even though training hadn't actually finished. `scan_cluster.py`'s `classify_run()` now checks Slurm's `PENDING` job list *before* falling back to the checkpoint-implies-done heuristic, giving these folds their own `train: "pending"` state — surfaced here as a queue-position notice, not an alert — and excluded from both `missing_predictions` and the Attention-needed badge count. It does **not** add to the tab's numeric badge.
   - The tab is empty/hidden-badge when there's nothing to act on (queued trainings still render even when the badge is hidden, since they aren't part of the count).
4. **Dice plotter** — pick one or more runs from the completeness matrix; three plot modes (box-by-run, box-by-label, per-case scatter) plus summary stats and CSV export.
5. **Results Explorer** — supervisor-ready, prepared comparison views built entirely from the already-scanned run inventory (no re-scanning). See **Results Explorer tab** below for the full breakdown of presets, domain split, and ablation-ladder gating.
6. **Cluster health** — storage quota and fairshare tables per cluster, with staleness badges (🟢 fresh / 🟡 stale / 🔴 load error).

**🧩 Orphan runs**: a run is flagged *orphan* when its `01_predictions/`
directory has checkpoints/predictions but **no `02_metrics/` directory was
ever created** for it — i.e. predict (and possibly train) happened but eval
was never run, or the metrics dir was never even created. These are
discovered by walking the predictions tree directly (not just the metrics
tree), deduplicated against normally-discovered runs, and always show
`eval: missing` for every fold since there's nothing to read.

**⏳ Queued folds**: a fold is flagged *queued* (`train: "pending"`) when
`squeue` shows its Slurm job in the `PENDING` state — waiting for a Slurm
allocation, not yet running. This is distinct from *stalled* (job is
`RUNNING` but its log has gone silent) and from *done* (no active/pending
job, and a `checkpoint_latest.pth`/`checkpoint_final.pth` exists). Before
this distinction existed, a queued fold with a checkpoint left over from a
prior preemption/requeue fell through to the "done" branch by default,
which produced false-positive "missing prediction" alerts for folds that
simply hadn't started training yet in their current queued job. The
Results Explorer's preset-resolution logic (`resolve_presets()` in
`app.py`) treats a queued preset the same way as an in-progress one —
`status: "in_progress"` rather than `"missing"` — so a supervisor-facing
plot never implies a method's data doesn't exist just because its job is
still in the Slurm queue.

A manual **"🔄 Refresh now"** button in the sidebar re-runs `refresh_all.sh`
on demand (in addition to the automated cron refresh below) and clears the
Streamlit cache.

### Known limitation — Dice plotter is vulcan-only for raw per-case data

`dice_loader.py` reads `eval_all.csv` directly off the local filesystem.
Because the Streamlit process runs on vulcan, it can only read *vulcan's*
`eval_all.csv` files. Runs whose `cluster` field is `killarney` are excluded
from the Dice plotter's run picker (they still appear normally in the
Overview/Completeness/Missing-predictions/Health tabs, which only need the
lightweight `status_*.json` summaries, not the raw per-case CSVs). If you
need to plot a killarney-only run's per-case Dice, run the dashboard on
killarney instead, or `scp` that run's `eval_all.csv` to vulcan manually.

## Results Explorer tab

A supervisor-facing tab (`render_results_explorer` in `app.py`) that turns
the same already-scanned run inventory into two prepared, paper-style
comparison views per dataset/contrast — no re-scanning, no live cluster
access, just regex-matching against the completeness matrix already in
`status_combined.json`. Controls: **Dataset**, **Contrast / modality**
(options depend on dataset — CHAOS: t1in/t2spir, BraTS2024-Glioma: t1n/t2w,
ON-Harmony: T1w/T2w, open-ms: flair/t1w), **Train fraction** / **Val
config** (only affect the `{train}/{val}`-templated "Ours + AugLab"
preset — see below), and a **Domain split** radio (Whole data / In-domain
only / Out-of-domain only, via `dice_loader.filter_domain`; out-of-domain
means every evaluated group other than the run's own training contrast,
e.g. CHAOS trained on t1in evaluated on t1out/t2spir/ct).

### Method-comparison presets (`METHOD_PRESETS` in `dashboard_common.py`)

Fixed baseline/method set, resolved per dataset/contrast/train-fraction/
val-config and rendered as box or bar-with-error-bars plots (both plot
kinds now share one consistent preset→color mapping with a visible
legend), CSV export, PNG via the plotly camera icon:

| Preset | Label shown | Notes |
|---|---|---|
| `baseline` | Baseline (no augmentation) | |
| `synthseg_noEM` | SynthSeg (no EM) | original label-generative SynthSeg variant |
| `synthseg_EM` | SynthSeg (EM) | |
| `srcsm` | SRCSM (Thaler et al., IEEE Access 2025) | |
| `auglab_default` | AugLab (default) | |
| `ours_auglab` | Ours + AugLab (deployed) — train`{train}`/val`{val}` | template preset, this project's deployed method; follows the **Train fraction** / **Val config** selectors |
| `ours_auglab_fixed_train050_val100` | Ours + AugLab (deployed) — train050/val100 | fixed reference preset, always shown regardless of the selectors, pinned to the paper's train050/val100 config so it's always available for at-a-glance comparison against whatever config the selectors are exploring |

The removed `ours_alone` ("Ours (PALETTE alone)") preset from an earlier
revision of this tab is gone — standalone PALETTE (no AugLab) is no longer
part of the Method Comparison view, since the deployed method is always
Ours+AugLab. (The Ablation Ladder below still legitimately shows a
standalone "+ Ours (PALETTE)" rung — that's an intentionally different,
unrelated feature; see below.)

The selector-driven `ours_auglab` preset substitutes the selected **Train
fraction** (`025`/`050`/`090`) and **Val config** (`000`/`100`) into both
the matching regex and the display label at render time
(`resolve_preset_regex` / `resolve_preset_label`), defaulting to
`train050_val000` — the config most consistently reported as the deployed/
paper number, though the best config differs by dataset (e.g. CHAOS
standalone PALETTE peaks at `train050_val100`). When the selectors happen
to land on train050/val100, the selector-driven and fixed presets resolve
to the identical run and label; `render_method_comparison` dedupes this
case (both the underlying rows via `resolve_presets()` and the plot's
category ordering via a `dict.fromkeys`-based dedup) so it renders as a
single bar/box, not two overlapping ones. At any other train/val
combination both presets render as two distinct entries. A preset with no
matching run at the selected dataset/contrast/train/val surfaces as an
explicit `st.warning` (missing) or `st.info` (still in progress / not yet
evaluated), with a detail expander table — it never silently disappears
from the plot.

### Ablation-ladder presets

Two ladders, both gated to the currently selected dataset/contrast/train/
val, with resolution against the newest matching run when duplicates exist:

- **Simple** (`ABLATION_LADDER_SIMPLE`): Base → + Ours (PALETTE) → + Ours +
  AugLab (deployed). Present for **all four datasets** — the default/
  fallback ladder. (This "+ Ours (PALETTE)" rung is unrelated to the
  Method Comparison's now-removed `ours_alone` preset — it's a legitimate,
  intentional step in the mechanism ladder and was left untouched.)
- **Full mechanism** (`ABLATION_LADDER_FULL`): an 8-rung ladder (Base →
  +k-means → +k-means+label-remap → +k-means+label-remap+Voronoi, in both
  a plain-nnUNet and an AugLab-wrapped family) showing the augmentation's
  internal mechanism step by step. Gated to
  `ABLATION_LADDER_FULL_DATASETS = {"open-ms"}` — confirmed present only
  for open-ms as of this session (a full regex parse of all real run
  directories across chaos/brats2024-glioma/on-harmony/open-ms found the
  kmeans/label_remap/voronoi method tokens exclusively under open-ms).
  Requesting the full ladder for any other dataset falls back to the
  simple ladder with an explicit `st.warning` explaining why.

### Domain-split rule

`dice_loader.tag_domain` compares each evaluated case's group against the
run's own training contrast (case-insensitive): a match is **in-domain**,
anything else (e.g. CHAOS trained on t1in, evaluated on t1out/t2spir/ct)
is **out-of-domain**. The Results Explorer's domain radio filters both the
method-comparison and ablation-ladder plots through this same rule via
`filter_domain`, so "Whole data" / "In-domain only" / "Out-of-domain only"
mean the same thing in both plots.

### Known limitation (same as the Dice plotter)

Per-case Dice values for a preset/rung whose winning run lives on
killarney can't be read by this vulcan-hosted process (see **Known
limitation — Dice plotter is vulcan-only** above) — those are skipped with
an explicit warning naming the run rather than silently omitted or
errored.

## Open-ms t1w — new contrast (in progress)

`open-ms`'s `DATASET_CONTRASTS` entry now includes `t1w` alongside `flair`
(`DATASET_CONTRASTS["open-ms"] = ["flair", "t1w"]`), reflecting a genuine,
still-in-progress second-contrast experiment on that dataset — not an
abandoned/stale run. As of this deployment, real runs exist on vulcan for
baseline, SynthSeg (no EM/EM), SRCSM, AugLab (default), and the
`auglabAug_v26_6_2` template family at several train/val configs, each
currently at **3 of 4 folds** (fold3 still pending).

`FOLDS_EXPECTED` (`dashboard_common.py`) is kept at its original
project-wide value of **4** — it was deliberately *not* lowered to 3 to
special-case open-ms/t1w, since redefining "complete" globally would
silently mask incomplete folds project-wide. Under this unchanged
standard, open-ms/t1w runs correctly show as in-progress (`n_folds: 4`,
`train_done`/`eval_done: 3`) everywhere in the dashboard — Completeness
matrix, Results Explorer preset availability, and progress bars — until
fold3 finishes.

## Automated refresh (cron / loop)

### vulcan — cron, every 15 minutes

```
*/15 * * * * /usr/bin/flock -n /tmp/paulh_dashboard_refresh.lock /project/aip-jcohen/paulh/mri_synthesis_project/dashboard/bin/refresh_all.sh >> /project/aip-jcohen/paulh/mri_synthesis_project/dashboard/logs/refresh_cron.log 2>&1
```

Already installed (`crontab -l` on vulcan to confirm/edit). The `flock`
guard prevents overlapping runs if a sync ever hangs. Logs go to
`dashboard/logs/refresh_cron.log`.

### killarney — detached loop (crontab is disabled for regular users here)

killarney's login node returns `Permission denied` / `not allowed to use
this program` for `crontab`, so periodic refresh uses a self-looping
detached shell script instead:

```bash
ssh killarney.alliancecan.ca
cd /project/aip-jcohen/paulh/mri_synthesis_project
mkdir -p dashboard/logs
nohup setsid bash dashboard/bin/killarney_scan_loop.sh \
    > dashboard/logs/scan_loop_stdout.log 2>&1 < /dev/null &
disown
```

It re-runs `scan_cluster.py` every 15 minutes (`INTERVAL_SECONDS` env var
to change) and logs to `dashboard/logs/scan_loop.log`.

Check it's alive:

```bash
pgrep -af killarney_scan_loop
tail -f dashboard/logs/scan_loop.log
```

Stop it:

```bash
pkill -f killarney_scan_loop.sh
```

**If the killarney login node reboots or the process otherwise dies**, the
loop does not restart itself — re-run the launch command above. There is
no systemd user-service option on this cluster for regular users, so this
manual relaunch is the accepted tradeoff.

## Killarney socket refresh (~monthly, manual, requires your phone)

vulcan reaches killarney only through an existing SSH `ControlMaster`
socket (`ControlPersist 30d`, configured in vulcan's `~/.ssh/config`).
Alliance Canada requires interactive Duo MFA on every fresh login, so this
socket must be re-opened by hand roughly once a month, or the sync step
will start failing soft (dashboard keeps showing killarney's *last known*
data, with a 🟡 stale badge and a warning banner).

To refresh the socket:

```bash
ssh vulcan.alliancecan.ca
cd /project/aip-jcohen/paulh/mri_synthesis_project
python3 dashboard/bin/refresh_killarney_socket.py
```

This just runs `ssh killarney.alliancecan.ca true` with your terminal's
stdin/stdout/stderr attached directly (stdlib `subprocess`, no third-party
dependency), so you'll see the normal Duo menu and can pick the push option
and tap Approve on your phone as usual — it does **not** bypass MFA, it's
only a thin wrapper with clearer messaging. **You must run it interactively
from a real terminal** (not piped/backgrounded) and from the project root —
if you get `ModuleNotFoundError: No module named 'pexpect'`, you have a
stale copy predating 2026-07-24; redeploy from `dashboard/bin/`. Once the
socket is open, cron-driven syncs will succeed silently for up to 30 days.

You'll know a refresh is needed when:
- The dashboard sidebar shows a killarney health badge of 🟡 or 🔴, or
- `dashboard/sync_meta.json` on vulcan shows `"last_sync_ok": false`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Port XXXX is not available` on Streamlit launch | Pick a free port (`ss -ltn \| grep <port>` to check) and update both the launch command and your SSH tunnel |
| "I can't connect to the dashboard" / tunnel hangs or refuses, even though it worked before | `vulcan.alliancecan.ca` DNS round-robins between `vulcan1` and `vulcan2` — retry `ssh -L 18765:127.0.0.1:18765 vulcan.alliancecan.ca`, it will likely land on the other node. Fixed as of 2026-07-24 by running Streamlit on **both** login nodes (see Launching the dashboard above); confirm both are up with `ssh vulcan1 'pgrep -af streamlit'` and `ssh vulcan2 'pgrep -af streamlit'` before assuming a real outage |
| Killarney badge stuck 🟡/🔴 | SSH control socket to killarney has expired — run `refresh_killarney_socket.py` (see above) |
| `killarney_scan_loop` not producing fresh data | Check `pgrep -af killarney_scan_loop`; if dead, relaunch (see above) — it does not auto-restart |
| Dashboard shows stale data despite fresh JSON | `st.cache_data(ttl=30)` — wait up to 30s or click "🔄 Refresh now" |
| Missing-predictions alert for a run you already fixed | Cache/staleness only — trigger a manual refresh; alerts are recomputed from the latest scan every time |
| Stalled-training alert for a job that's actually progressing normally | Check the fold's `training_log_*.txt` mtime directly — a very slow epoch (large model/data) can occasionally exceed `STALL_THRESHOLD_SECONDS` (30 min) without being truly hung; raise the threshold in `dashboard_common.py` if this becomes a recurring false positive |
| A run has predictions but shows up as 🧩 orphan even though eval ran | Confirm `02_metrics/<...>/foldN/eval_all.csv` actually exists at the expected path for that run's dataset/contrast/group — orphan detection only checks for the metrics *directory*, so a metrics dir that exists but is misplaced/misnamed will still be flagged |
| Results Explorer shows a preset as "missing" that you know ran | Check the **Train fraction** / **Val config** selectors — the `{train}/{val}`-templated "Ours + AugLab (deployed)" preset only matches the currently selected config; switch train/val to find where that preset's run actually landed. The fixed train050/val100 reference preset is unaffected by these selectors. |
| open-ms/t1w always shows 3/4 folds, never reaches 100% | Expected — fold3 training is still in progress for this contrast; see **Open-ms t1w — new contrast** above. `FOLDS_EXPECTED` was deliberately kept at 4 rather than lowered to 3 |
| "Full mechanism" ladder unavailable / falls back to simple | Expected outside open-ms — see `ABLATION_LADDER_FULL_DATASETS` in `dashboard_common.py`; the granular kmeans/label-remap/Voronoi rungs were only run for that dataset |
| Results Explorer plot missing a rung/preset with a "lives on cluster 'killarney'" warning | Same vulcan-only limitation as the Dice plotter (see above) — that run's `eval_all.csv` isn't readable from this process |
| A fold you know is queued (not stalled/missing) shows up under Missing predictions | Shouldn't happen after the `train: "pending"` fix — if it does, the job's Slurm state may have flipped from `PENDING` back to something else (e.g. `RUNNING` with no recent log = stalled) between the scan and your check; trigger a manual refresh and recheck `squeue -j <jobid>` |
