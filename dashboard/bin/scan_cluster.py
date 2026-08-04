#!/usr/bin/env python3
"""
Cluster-local status scanner for the MRI synthesis project.

Runs on a single cluster (vulcan or killarney) against its own local
filesystem + local Slurm queue, and emits one JSON document describing:
  - queue: all jobs currently in squeue for this user on this cluster
  - runs: every training run directory found under EITHER
    datasets/*/8_results_*/02_metrics/*_model/{contrast}/* (the normal case)
    OR datasets/*/8_results_*/01_predictions/*_model/{contrast}/{category}/*
    with no metrics-side counterpart at all (a run that finished training
    but never had predict/eval triggered — see discover_run_names() below;
    prior versions of this scanner only walked the metrics side and were
    blind to this case entirely).
    Each run carries per-fold status (train/predict/eval) plus, for folds
    still training, live epoch progress parsed from nnU-Net's
    training_log_*.txt.
  - storage: diskusage_report parsed
  - fairshare: sshare parsed

No dependencies beyond the Python stdlib, so it can run with the system
python3 on either login node (no venv activation required).

Usage: python3 scan_cluster.py --project-root /path/to/mri_synthesis_project
                                --cluster vulcan
                                --out /path/to/status.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# scan_cluster.py runs standalone on each cluster's login node (no venv), so
# it needs dashboard_common.py to sit next to it in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard_common import (  # noqa: E402
    RUN_DIR_RE, PRED_RUN_DIR_RE, DATASET_CONTRASTS, DATASET_RESULTS_DIRNAME,
    DATASET_MODEL_DIRNAME, FOLDS_EXPECTED, STALL_THRESHOLD_SECONDS,
)

TRAINING_LOG_EPOCH_RE = re.compile(r'Epoch (\d+) *\n')
TRAINING_LOG_EPOCH_TIME_RE = re.compile(r'Epoch time: ([\d.]+) s')
# Tail-read size for training_log_*.txt: epochs log ~14 lines each at ~80
# bytes/line, so 16KB comfortably covers the last several epochs without
# reading a file that can grow past 100K lines over a full training run.
TRAINING_LOG_TAIL_BYTES = 16_384


def run(cmd, timeout=30):
    # Use a login shell: on some clusters (e.g. Killarney) Slurm client
    # binaries and diskusage_report/sshare are only on PATH after profile
    # scripts run (module loads etc.), while a bare non-login shell (as
    # used on e.g. Vulcan too, harmlessly) does not have them. `bash -lc`
    # works uniformly on both.
    full_cmd = ["bash", "-lc", cmd]
    try:
        p = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except Exception as e:
        return f"__ERROR__ {e}"


def get_queue(cluster):
    out = run("squeue -u \"$USER\" -o '%.12i|%j|%.12T|%.12M|%.20V|%R' --sort=V")
    jobs = []
    if out.startswith("__ERROR__"):
        return jobs, out
    lines = out.strip().splitlines()
    for line in lines[1:]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        jobid, name, state, elapsed, submit, reason = parts[:6]
        # parse fold + dataset + contrast + method + timestamp out of job name
        m = re.match(r'^fold(\d+)_(.+?)_([A-Za-z0-9]+)_(.+)$', name)
        parsed = {}
        if m:
            fold, dataset, contrast, rest = m.groups()
            # rest = method[..._TIMESTAMP]; timestamp is last 15 chars YYYYMMDD_HHMMSS
            ts_m = re.search(r'_(\d{8}_\d{6})$', rest)
            if ts_m:
                method = rest[: ts_m.start()]
                timestamp = ts_m.group(1)
            else:
                method = rest
                timestamp = None
            parsed = {"fold": int(fold), "dataset": dataset, "contrast": contrast,
                      "method": method, "timestamp": timestamp}
        jobs.append({
            "job_id": jobid, "name": name, "state": state, "elapsed": elapsed,
            "submit_time": submit, "reason": reason, "cluster": cluster, **parsed
        })
    return jobs, None


def get_storage(cluster):
    out = run("diskusage_report")
    rows = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("Description"):
            continue
        # Description                Space         # of files
        m = re.match(r'^(.+?)\s{2,}([\d.]+[KMGT]?i?B)/\s*([\d.]+[KMGT]?i?B)\s+(\S+)\s*/\s*(\S+)$', line)
        if m:
            desc, used, total, nfiles, nfiles_max = m.groups()
            rows.append({"description": desc.strip(), "used": used, "total": total,
                         "n_files": nfiles, "n_files_max": nfiles_max})
    return rows


def get_fairshare(cluster):
    out = run("sshare -U -u \"$USER\"")
    rows = []
    lines = out.strip().splitlines()
    if len(lines) >= 3:
        header = lines[0].split()
        for line in lines[2:]:
            vals = line.split()
            if len(vals) == len(header):
                rows.append(dict(zip(header, vals)))
    return rows


def get_epoch_progress(fold_dir, now=None):
    """
    Parse live training progress out of nnU-Net's own per-fold artifacts:
      - debug.json's "num_epochs" for the configured target (reliable —
        written once at job start and never changes).
      - training_log_*.txt's last "Epoch N" header plus the most recent
        "Epoch time: X s" line, tail-read (see TRAINING_LOG_TAIL_BYTES) so
        this stays cheap even for a run that's logged thousands of epochs.
    NOTE: debug.json also carries a "current_epoch" field, but it is written
    ONCE at job start and stays "0" for the entire run (confirmed against a
    run that reached epoch 193) — it is not a live progress signal and is
    deliberately NOT used here; the training log is the only field that
    updates as training proceeds.
    Returns None if this fold has no training log yet (not started, or an
    old run predating log-based tracking), else a dict with:
      current_epoch, num_epochs, epoch_seconds (last logged epoch duration),
      log_mtime (iso), seconds_since_log_update.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    num_epochs = None
    debug_json = fold_dir / "debug.json"
    if debug_json.exists():
        try:
            with open(debug_json) as f:
                d = json.load(f)
            ne = d.get("num_epochs")
            if ne is not None:
                num_epochs = int(ne)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass

    log_files = sorted(fold_dir.glob("training_log_*.txt"))
    if not log_files:
        if num_epochs is None:
            return None
        return {"current_epoch": None, "num_epochs": num_epochs, "epoch_seconds": None,
                "log_mtime": None, "seconds_since_log_update": None}
    log_path = log_files[-1]
    try:
        st = log_path.stat()
        size = st.st_size
        with open(log_path, "rb") as f:
            if size > TRAINING_LOG_TAIL_BYTES:
                f.seek(size - TRAINING_LOG_TAIL_BYTES)
            tail = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return {"current_epoch": None, "num_epochs": num_epochs, "epoch_seconds": None,
                "log_mtime": None, "seconds_since_log_update": None}

    epochs = TRAINING_LOG_EPOCH_RE.findall(tail)
    current_epoch = int(epochs[-1]) if epochs else None
    times = TRAINING_LOG_EPOCH_TIME_RE.findall(tail)
    epoch_seconds = float(times[-1]) if times else None
    log_mtime_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    seconds_since = (now - log_mtime_dt).total_seconds()
    return {
        "current_epoch": current_epoch, "num_epochs": num_epochs,
        "epoch_seconds": epoch_seconds, "log_mtime": log_mtime_dt.isoformat(),
        "seconds_since_log_update": seconds_since,
    }


def classify_run(run_dir_path, contrast_predictions_root, category, run_name,
                  active_train_folds=None, expected_folds=FOLDS_EXPECTED, now=None,
                  pending_train_folds=None):
    """
    run_dir_path: .../02_metrics/{model}/{contrast}/{run_name}/, or None if
        this run has no metrics directory at all yet (predictions-only --
        see discover_run_names(); eval is then "missing" for every fold by
        construction, which is exactly the point: it makes such runs visible
        in the completeness matrix / missing_predictions instead of being
        skipped outright as they were before this got fixed).
    contrast_predictions_root: .../01_predictions/{model}/{contrast}/
    active_train_folds: set of fold ints with a currently RUNNING squeue job
        for this exact (dataset, contrast, method, timestamp) -- used to
        disambiguate "done" vs "in_progress"/"stalled" from a
        checkpoint_best.pth alone (nnU-Net has no checkpoint_final.pth
        marker in this pipeline; a checkpoint plus no active job is our
        best completion signal).
    pending_train_folds: set of fold ints with a currently PENDING (queued,
        not yet running -- e.g. preempted/requeued, or waiting on priority)
        squeue job for this same key. A fold in this set is genuinely still
        mid-training even though it isn't RUNNING right now and may already
        have a checkpoint_latest.pth from before it was preempted -- without
        this, such a fold falls through to the has_checkpoint branch below
        and gets misclassified "done", which then produces a false-positive
        "trained but not predicted" entry in missing_predictions (confirmed
        2026-07-17 against two SRCSM runs stuck behind ~1300 pending jobs in
        their partition). "pending" is deliberately its own train_state --
        distinct from "done" (excludes it from missing_predictions) and from
        "in_progress"/"stalled" (those require an active RUNNING job to have
        current epoch-log freshness to compare against).
    Returns dict with per-fold status train/predict/eval plus, when
    train == "in_progress", "stalled", or "pending", an "epoch" sub-dict (see
    get_epoch_progress).
    """
    if active_train_folds is None:
        active_train_folds = set()
    if pending_train_folds is None:
        pending_train_folds = set()
    if now is None:
        now = datetime.now(timezone.utc)
    folds = {}
    # The predictions-side run dir name drops the "{category}_" prefix that the
    # metrics-side dir name carries, e.g. metrics "auglab_chaos_t1in_X_TS" <->
    # predictions "chaos_t1in_X_TS" under 01_predictions/.../auglab/.
    pred_run_name = run_name
    prefix = category + "_"
    if pred_run_name.startswith(prefix):
        pred_run_name = pred_run_name[len(prefix):]

    pred_run_dir = contrast_predictions_root / category / pred_run_name

    for f in range(expected_folds):
        fold_status = {"train": "missing", "predict": "missing", "eval": "missing"}

        # eval: 02_metrics/.../{run}/fold{f}/eval_all.csv (skipped entirely
        # when run_dir_path is None -- a predictions-only orphan run).
        if run_dir_path is not None:
            eval_csv = run_dir_path / f"fold{f}" / "eval_all.csv"
            if eval_csv.exists():
                fold_status["eval"] = "done"

        # training: any fold_{f} dir with a checkpoint. No checkpoint_final.pth
        # marker exists in this pipeline, so "done" = checkpoint present AND no
        # RUNNING queue job currently claims this fold; "in_progress"/"stalled"
        # = a matching RUNNING job exists (split by whether the training log
        # has updated recently).
        train_state = "missing"
        has_checkpoint = False
        fold_dir_found = None
        if pred_run_dir.exists():
            for ds_dir in pred_run_dir.glob("Dataset*"):
                for trainer_dir in ds_dir.glob("*__nnUNetPlans__3d_fullres"):
                    fold_dir = trainer_dir / f"fold_{f}"
                    if (fold_dir / "checkpoint_best.pth").exists() or (fold_dir / "checkpoint_latest.pth").exists():
                        has_checkpoint = True
                        fold_dir_found = fold_dir

        epoch_info = None
        if f in active_train_folds or f in pending_train_folds:
            # Look up the training log regardless of checkpoint presence --
            # a job can be RUNNING for a while before its first checkpoint
            # write, and we still want epoch progress in that window (and
            # for a PENDING/requeued fold, the log from its last RUNNING
            # stretch is exactly what tells us how far along it already is).
            search_dir = fold_dir_found
            if search_dir is None and pred_run_dir.exists():
                for ds_dir in pred_run_dir.glob("Dataset*"):
                    for trainer_dir in ds_dir.glob("*__nnUNetPlans__3d_fullres"):
                        cand = trainer_dir / f"fold_{f}"
                        if cand.exists():
                            search_dir = cand
            if search_dir is not None:
                epoch_info = get_epoch_progress(search_dir, now=now)
        if f in active_train_folds:
            if epoch_info is not None and epoch_info.get("seconds_since_log_update") is not None \
                    and epoch_info["seconds_since_log_update"] > STALL_THRESHOLD_SECONDS:
                train_state = "stalled"
            else:
                train_state = "in_progress"
        elif f in pending_train_folds:
            # Queued (e.g. preempted-and-requeued, or simply waiting on
            # priority) -- genuinely still mid-training, not abandoned and
            # not complete. Deliberately NOT "stalled": nothing is hung, the
            # job just hasn't been scheduled back in yet.
            train_state = "pending"
        elif has_checkpoint:
            train_state = "done"
        fold_status["train"] = train_state
        if epoch_info is not None:
            fold_status["epoch"] = epoch_info

        # predict: pred_run_dir/fold{f}/<item>/*.nii.gz present for at least one item
        predict_state = "missing"
        fold_pred_dir = pred_run_dir / f"fold{f}"
        if fold_pred_dir.exists():
            for item_dir in fold_pred_dir.iterdir():
                if item_dir.is_dir():
                    niftis = list(item_dir.glob("*.nii.gz"))
                    if niftis:
                        predict_state = "done"
                        break
        fold_status["predict"] = predict_state

        folds[f] = fold_status
    return folds


def scan_dataset_contrast(project_root, dataset, contrast, jobs):
    results_dirname = DATASET_RESULTS_DIRNAME[dataset]
    model_dirname = DATASET_MODEL_DIRNAME[dataset]
    metrics_root = project_root / "datasets" / dataset / results_dirname / "02_metrics" / model_dirname / contrast
    predictions_root = project_root / "datasets" / dataset / results_dirname / "01_predictions" / model_dirname / contrast

    # Build (dataset, contrast, method, timestamp) -> {fold} for RUNNING jobs,
    # and separately for PENDING jobs (queued/requeued but not yet running --
    # e.g. preempted, or waiting on priority behind other jobs in the
    # partition). Both are genuinely "still training", just in different
    # squeue states; classify_run() needs them split so it can distinguish
    # in_progress/stalled (RUNNING) from pending (PENDING) from done/missing.
    running_by_key = {}
    pending_by_key = {}
    for j in jobs:
        if "fold" not in j:
            continue
        state = j.get("state")
        key = (j["dataset"], j["contrast"], j["method"], j.get("timestamp"))
        if state == "RUNNING":
            running_by_key.setdefault(key, set()).add(j["fold"])
        elif state == "PENDING":
            pending_by_key.setdefault(key, set()).add(j["fold"])

    now = datetime.now(timezone.utc)
    runs = []
    seen_run_dir_names = set()

    if metrics_root.exists():
        for entry in metrics_root.iterdir():
            if not entry.is_dir():
                continue
            m = RUN_DIR_RE.match(entry.name)
            if not m:
                continue
            gd = m.groupdict()
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
            key = (dataset, contrast, gd["method"], gd["timestamp"])
            active_folds = running_by_key.get(key, set())
            pending_folds = pending_by_key.get(key, set())
            folds = classify_run(entry, predictions_root, gd["category"], entry.name, active_folds,
                                  now=now, pending_train_folds=pending_folds)
            runs.append({
                "dataset": dataset, "contrast": contrast, "category": gd["category"],
                "method": gd["method"], "timestamp": gd["timestamp"], "run_dir_name": entry.name,
                "mtime": mtime, "folds": folds, "orphan": False,
            })
            seen_run_dir_names.add(entry.name)

    # Orphan discovery: walk 01_predictions/{model}/{contrast}/{category}/*
    # directly and pick up any run that finished training/predicting but has
    # NO counterpart at all under 02_metrics -- i.e. predict/eval were never
    # triggered for it. These are exactly the runs a metrics-only scan is
    # blind to, and they are also the runs most likely to silently rot
    # (a training job completes, nobody remembers to launch predict+eval).
    if predictions_root.exists():
        for category_dir in predictions_root.iterdir():
            if not category_dir.is_dir() or category_dir.name not in ("auglab", "nnUNet"):
                continue
            for entry in category_dir.iterdir():
                if not entry.is_dir():
                    continue
                # Reconstruct the metrics-side run_dir_name this would map to,
                # to dedupe against runs already discovered above.
                candidate_run_dir_name = f"{category_dir.name}_{entry.name}"
                if candidate_run_dir_name in seen_run_dir_names:
                    continue
                m = PRED_RUN_DIR_RE.match(entry.name)
                if not m:
                    continue
                gd = m.groupdict()
                mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
                key = (dataset, contrast, gd["method"], gd["timestamp"])
                active_folds = running_by_key.get(key, set())
                pending_folds = pending_by_key.get(key, set())
                folds = classify_run(None, predictions_root, category_dir.name, candidate_run_dir_name,
                                      active_folds, now=now, pending_train_folds=pending_folds)
                # Skip if genuinely nothing found (e.g. an unrelated directory
                # that happens to match the timestamp pattern) -- require at
                # least one fold with train != "missing".
                if not any(fs["train"] != "missing" for fs in folds.values()):
                    continue
                runs.append({
                    "dataset": dataset, "contrast": contrast, "category": category_dir.name,
                    "method": gd["method"], "timestamp": gd["timestamp"],
                    "run_dir_name": candidate_run_dir_name,
                    "mtime": mtime, "folds": folds, "orphan": True,
                })
                seen_run_dir_names.add(candidate_run_dir_name)

    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    jobs, queue_err = get_queue(args.cluster)
    storage = get_storage(args.cluster)
    fairshare = get_fairshare(args.cluster)

    all_runs = []
    for dataset, contrasts in DATASET_CONTRASTS.items():
        for contrast in contrasts:
            all_runs.extend(scan_dataset_contrast(project_root, dataset, contrast, jobs))

    doc = {
        "cluster": args.cluster,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "queue": jobs,
        "queue_error": queue_err,
        "storage": storage,
        "fairshare": fairshare,
        "runs": all_runs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({len(all_runs)} runs, {len(jobs)} queued jobs)")


if __name__ == "__main__":
    main()
