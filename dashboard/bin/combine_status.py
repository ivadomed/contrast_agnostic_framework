#!/usr/bin/env python3
"""
Combine per-cluster status_<cluster>.json documents (each produced locally by
scan_cluster.py on its own cluster) into a single status_combined.json that
the Streamlit dashboard reads.

Adds two derived views on top of the raw per-cluster run lists:
  - completeness_matrix: one row per (dataset, contrast, category, method),
    collapsed across clusters/timestamps to the single most-recently-modified
    run, with train/predict/eval fold counts and a rollup status.
  - missing_predictions: alert list of runs where training finished for a
    fold but prediction and/or evaluation did not (an actionable backlog:
    those predict/eval jobs just need to be submitted).

Usage: python3 combine_status.py --project-root /path/to/mri_synthesis_project
                                  [--out /path/to/status_combined.json]
"""
import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

CLUSTERS = ["vulcan", "killarney"]

# Job-name prefixes used by each dataset's predict/eval submission scripts
# (see datasets/<dataset>/5_scripts_<dataset>/05_predict/05_01_predict_common.sh's
# PREDICT_JOB_PREFIX and each dataset's 06_evaluate/06_01_evaluate_*.sh
# `run_job --name` line). on-harmony has no separate predict job -- its
# 06_01_evaluate_testset.sh fans out one job per fold that predicts AND
# evaluates every test contrast together, named "onheval_...".
PREDICT_JOB_PREFIXES = {
    "chaos": "chaos_predict",
    "brats2024-glioma": "brats_predict",
    "open-ms": "openms_predict",
}
EVAL_JOB_PREFIXES = {
    "chaos": "chaos_eval",
    "brats2024-glioma": "brats_eval",
    "open-ms": "openms_eval",
    "on-harmony": "onheval",
}


def load_cluster_doc(dashboard_dir, cluster):
    path = dashboard_dir / f"status_{cluster}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        doc["_source_path"] = str(path)
        doc["_stale"] = False
        return doc
    except (json.JSONDecodeError, OSError) as e:
        return {"cluster": cluster, "_load_error": str(e), "_stale": True}


def attach_epoch_progress(all_queue, all_runs):
    """
    Join live epoch progress (current_epoch/num_epochs/epoch_seconds, from
    scan_cluster.get_epoch_progress via each run's folds[fold]["epoch"])
    onto queue entries so the Overview tab can show training progress and
    an ETA for jobs that are actually running/pending a train stage,
    without the dashboard needing to re-derive the join itself.

    Join key: (cluster, dataset, contrast, method, timestamp, fold) --
    exactly the tuple scan_cluster.py already uses to disambiguate queue
    jobs from run directories (see scan_dataset_contrast's running_by_key /
    pending_by_key). Only queue entries with a parsed "fold" (i.e. the job
    name matched the fold-prefixed naming convention) can join at all --
    predict/eval jobs and anything with a non-conforming name are left
    untouched (no epoch/eta fields added).

    Adds, in place, onto each matching queue dict:
      current_epoch, num_epochs, epoch_pct (0-100, rounded),
      eta_seconds (None if epoch_seconds or the remaining-epoch count is
      unknown), based on a simple steady-state assumption (remaining
      epochs * last logged epoch duration) -- no smoothing, so it will be
      noisy right after a preemption/resume but self-corrects within a
      few epochs.
    """
    epoch_by_key = {}
    for r in all_runs:
        for fidx, fstat in r["folds"].items():
            epoch = fstat.get("epoch")
            if not epoch:
                continue
            key = (r["cluster"], r["dataset"], r["contrast"], r["method"],
                   r["timestamp"], int(fidx))
            epoch_by_key[key] = epoch

    for j in all_queue:
        if "fold" not in j:
            continue
        key = (j["cluster"], j["dataset"], j["contrast"], j["method"],
               j.get("timestamp"), j["fold"])
        epoch = epoch_by_key.get(key)
        if epoch is None:
            continue
        current_epoch = epoch.get("current_epoch")
        num_epochs = epoch.get("num_epochs")
        epoch_seconds = epoch.get("epoch_seconds")
        j["current_epoch"] = current_epoch
        j["num_epochs"] = num_epochs
        if current_epoch is not None and num_epochs:
            j["epoch_pct"] = round(100 * current_epoch / num_epochs, 1)
        else:
            j["epoch_pct"] = None
        if current_epoch is not None and num_epochs and epoch_seconds is not None \
                and num_epochs > current_epoch:
            j["eta_seconds"] = (num_epochs - current_epoch) * epoch_seconds
        else:
            j["eta_seconds"] = None


def _inflight_job_for(entry, all_queue):
    """
    Best-effort match of a missing_predictions entry ("trained but not
    predicted" / "predicted but not evaluated") against the live squeue
    listing, so the alert list can flag "already queued, no action needed"
    vs. "actually needs submitting".

    Queue jobs are matched by SUBSTRING against the raw Slurm job name,
    using each dataset's own predict/eval job-naming convention (see
    run_job --name calls in each dataset's 05_predict/06_evaluate scripts):
      - chaos/brats2024-glioma/open-ms PREDICT jobs: "<prefix>_<method>_fold<F>"
        -- no run timestamp in the name, so this can only match on
        method+fold, not the specific run (a retried/superseded run at the
        same method+fold would also match -- acceptable for an informational
        "likely already running" hint, not an authoritative join).
      - chaos/brats2024-glioma EVAL jobs: "<prefix>_<run_dir_name>_fold<F>_..."
        -- includes the training run_dir_name, so this matches the specific
        run.
      - open-ms EVAL jobs: "<prefix>_<category>_<run_dir_name>..." -- no
        per-fold job (evaluates all folds in one job), so fold is ignored.
      - on-harmony: no separate predict job -- 06_01_evaluate_testset.sh
        fans out one job per fold that predicts AND evaluates together,
        named "onheval_<run_dir_name[:26]>_f<F>"; matches both issue types.

    Returns the matching queue job dict (cluster + job_id + state), or None.
    """
    dataset = entry["dataset"]
    method = entry["method"]
    fold = entry["fold"]
    run_dir_name = entry.get("run_dir_name", "")
    cluster = entry["cluster"]
    is_predict = entry["issue"] == "trained but not predicted"

    candidates = []
    if dataset == "on-harmony":
        # Single combined predict+eval job per fold, truncated RUN_ID.
        needle = f"onheval_{run_dir_name[:26]}_f{fold}"
        candidates.append(needle)
    elif is_predict:
        prefix = PREDICT_JOB_PREFIXES.get(dataset)
        if prefix:
            candidates.append(f"{prefix}_{method}_fold{fold}")
    else:
        prefix = EVAL_JOB_PREFIXES.get(dataset)
        if prefix and dataset == "open-ms":
            candidates.append(f"{prefix}_")  # category_run_dir_name follows; check run_dir_name below
        elif prefix:
            candidates.append(f"{prefix}_{run_dir_name}_fold{fold}")

    if not candidates:
        return None

    for j in all_queue:
        if j.get("cluster") != cluster:
            continue
        name = j.get("name", "")
        for needle in candidates:
            if needle not in name:
                continue
            # open-ms eval: prefix matched loosely above -- also require the
            # run_dir_name substring since fold isn't in the job name.
            if dataset == "open-ms" and not is_predict and run_dir_name not in name:
                continue
            return {"job_id": j.get("job_id"), "state": j.get("state"), "cluster": cluster}
    return None


def attach_inflight_status(missing_predictions, all_queue):
    """Adds an 'inflight' sub-dict ({job_id, state, cluster} or None) onto
    each missing_predictions entry -- see _inflight_job_for for the matching
    rules. In place."""
    for entry in missing_predictions:
        entry["inflight"] = _inflight_job_for(entry, all_queue)


def fold_rollup(folds):
    """folds: {fold_idx: {"train":..,"predict":..,"eval":..,"epoch":..?}} -> counts."""
    n = len(folds)
    train_done = sum(1 for v in folds.values() if v["train"] == "done")
    train_active = sum(1 for v in folds.values() if v["train"] == "in_progress")
    train_stalled = sum(1 for v in folds.values() if v["train"] == "stalled")
    # "pending" = a queued (PENDING in squeue, e.g. preempted/requeued or
    # waiting on partition priority) job for this fold -- genuinely still
    # training, not abandoned, not complete. See scan_cluster.classify_run().
    train_pending = sum(1 for v in folds.values() if v["train"] == "pending")
    predict_done = sum(1 for v in folds.values() if v["predict"] == "done")
    eval_done = sum(1 for v in folds.values() if v["eval"] == "done")
    return {
        "n_folds": n,
        "train_done": train_done, "train_in_progress": train_active,
        "train_stalled": train_stalled, "train_pending": train_pending,
        "predict_done": predict_done, "eval_done": eval_done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    dashboard_dir = project_root / "dashboard"
    out_path = Path(args.out) if args.out else dashboard_dir / "status_combined.json"

    sync_meta_path = dashboard_dir / "sync_meta.json"
    sync_meta = {}
    if sync_meta_path.exists():
        try:
            sync_meta = json.loads(sync_meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    cluster_docs = {}
    for cluster in CLUSTERS:
        doc = load_cluster_doc(dashboard_dir, cluster)
        if doc is not None:
            cluster_docs[cluster] = doc

    # killarney's doc is a local copy synced from killarney's own cron scan;
    # mark it stale if the last sync attempt failed (vulcan couldn't reach it).
    if "killarney" in cluster_docs and sync_meta.get("last_sync_ok") is False:
        cluster_docs["killarney"]["_stale"] = True
        cluster_docs["killarney"]["_stale_reason"] = "last sync attempt failed; showing last-known data"

    all_runs = []
    all_queue = []
    all_storage = []
    all_fairshare = []
    for cluster, doc in cluster_docs.items():
        if doc.get("_load_error"):
            continue
        for r in doc.get("runs", []):
            r2 = dict(r)
            r2["cluster"] = cluster
            all_runs.append(r2)
        for j in doc.get("queue", []):
            j2 = dict(j)
            j2["cluster"] = cluster
            all_queue.append(j2)
        for s in doc.get("storage", []):
            s2 = dict(s)
            s2["cluster"] = cluster
            all_storage.append(s2)
        for fs in doc.get("fairshare", []):
            fs2 = dict(fs)
            fs2["cluster"] = cluster
            all_fairshare.append(fs2)

    # Join live epoch progress (current_epoch/num_epochs/eta) onto queue
    # entries for the Overview tab's queue table -- see attach_epoch_progress.
    attach_epoch_progress(all_queue, all_runs)

    # completeness matrix: group by (dataset, contrast, category, method),
    # keep only the most-recently-modified run per group (across clusters).
    groups = {}
    for r in all_runs:
        key = (r["dataset"], r["contrast"], r["category"], r["method"])
        if key not in groups or r["mtime"] > groups[key]["mtime"]:
            groups[key] = r

    completeness_matrix = []
    for key, r in sorted(groups.items()):
        dataset, contrast, category, method = key
        rollup = fold_rollup(r["folds"])
        row = {
            "dataset": dataset, "contrast": contrast, "category": category,
            "method": method, "cluster": r["cluster"], "timestamp": r["timestamp"],
            "run_dir_name": r["run_dir_name"], "mtime": r["mtime"],
            "orphan": r.get("orphan", False),
            **rollup,
        }
        completeness_matrix.append(row)

    # missing_predictions scans EVERY run (not just the latest-mtime one kept
    # per group above) — an older/abandoned run for the same method that got
    # superseded by a fresh timestamp should still surface as a gap, since
    # its trained-but-unpredicted checkpoint represents real leftover work
    # (or wasted compute) someone should look at.
    #
    # BUT: if a strictly newer run at the identical (dataset, contrast,
    # category, method) key already completed that same stage for that same
    # fold, the older run's gap was abandoned in favor of a later, successful
    # attempt -- it's stale history, not a standing gap, and flagging it is a
    # false positive (confirmed 2026-07-17: 3 of 9 flagged entries were of
    # this kind, e.g. chaos/t1in/auglab/auglabAug_v26_6_2_train025_val100's
    # 20260621 run was superseded by a fully-completed 20260715 run).
    #
    # "Newer" here is judged by mtime (actual filesystem recency -- the same
    # field the completeness_matrix grouping above uses to pick its canonical
    # run per key), NOT by the timestamp embedded in the run_dir_name: those
    # two can disagree (e.g. brats2024-glioma/t1n/nnUNet/v26_6_2_train090_val000
    # has a completed run whose run-dir timestamp label (20260608) is
    # textually earlier than the abandoned run's label (20260609), yet its
    # mtime is later -- comparing by timestamp label would have missed it).
    runs_by_key = {}
    for r in all_runs:
        runs_by_key.setdefault(
            (r["dataset"], r["contrast"], r["category"], r["method"]), []
        ).append(r)

    def superseded_by_newer(run, fold_idx, stage):
        key = (run["dataset"], run["contrast"], run["category"], run["method"])
        for other in runs_by_key.get(key, ()):
            if other is run or other["mtime"] <= run["mtime"]:
                continue
            other_fold = other["folds"].get(fold_idx)
            if other_fold and other_fold.get(stage) == "done":
                return True
        return False

    missing_predictions = []
    for r in all_runs:
        dataset, contrast, category, method = r["dataset"], r["contrast"], r["category"], r["method"]
        for fidx, fstat in r["folds"].items():
            if fstat["train"] == "done" and fstat["predict"] != "done":
                if superseded_by_newer(r, fidx, "predict"):
                    continue
                missing_predictions.append({
                    "dataset": dataset, "contrast": contrast, "category": category,
                    "method": method, "cluster": r["cluster"], "timestamp": r["timestamp"],
                    "fold": fidx, "issue": "trained but not predicted",
                    "run_dir_name": r["run_dir_name"],
                })
            elif fstat["predict"] == "done" and fstat["eval"] != "done":
                if superseded_by_newer(r, fidx, "eval"):
                    continue
                missing_predictions.append({
                    "dataset": dataset, "contrast": contrast, "category": category,
                    "method": method, "cluster": r["cluster"], "timestamp": r["timestamp"],
                    "fold": fidx, "issue": "predicted but not evaluated",
                    "run_dir_name": r["run_dir_name"],
                })

    # Flag entries whose predict/eval job is already queued/running so the
    # alert list can distinguish "needs submitting" from "already in
    # flight, just wait" -- see attach_inflight_status.
    attach_inflight_status(missing_predictions, all_queue)

    # stalled_trainings: folds whose scanner-side classification is "stalled"
    # (RUNNING per squeue, but training_log_*.txt hasn't advanced in longer
    # than STALL_THRESHOLD_SECONDS -- see scan_cluster.get_epoch_progress).
    # Surfaced separately from missing_predictions since the fix here is
    # "go look at / requeue this job", not "submit a downstream stage".
    stalled_trainings = []
    for r in all_runs:
        dataset, contrast, category, method = r["dataset"], r["contrast"], r["category"], r["method"]
        for fidx, fstat in r["folds"].items():
            if fstat["train"] != "stalled":
                continue
            epoch = fstat.get("epoch") or {}
            stalled_trainings.append({
                "dataset": dataset, "contrast": contrast, "category": category,
                "method": method, "cluster": r["cluster"], "timestamp": r["timestamp"],
                "fold": fidx, "run_dir_name": r["run_dir_name"],
                "current_epoch": epoch.get("current_epoch"),
                "num_epochs": epoch.get("num_epochs"),
                "seconds_since_log_update": epoch.get("seconds_since_log_update"),
            })

    # pending_trainings: folds whose scanner-side classification is "pending"
    # (a queued/requeued squeue job for this fold, not currently RUNNING --
    # e.g. preempted, or waiting on partition priority behind other jobs).
    # Surfaced separately from missing_predictions/stalled_trainings: this is
    # neither a completion gap nor a hang, it's ordinary queueing -- nothing
    # needs to be done, it's just informational ("still training, will
    # resume when scheduled").
    pending_trainings = []
    for r in all_runs:
        dataset, contrast, category, method = r["dataset"], r["contrast"], r["category"], r["method"]
        for fidx, fstat in r["folds"].items():
            if fstat["train"] != "pending":
                continue
            epoch = fstat.get("epoch") or {}
            pending_trainings.append({
                "dataset": dataset, "contrast": contrast, "category": category,
                "method": method, "cluster": r["cluster"], "timestamp": r["timestamp"],
                "fold": fidx, "run_dir_name": r["run_dir_name"],
                "current_epoch": epoch.get("current_epoch"),
                "num_epochs": epoch.get("num_epochs"),
            })

    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clusters": {c: {"scanned_at": d.get("scanned_at"), "stale": d.get("_stale", False),
                          "stale_reason": d.get("_stale_reason"), "load_error": d.get("_load_error")}
                     for c, d in cluster_docs.items()},
        "sync_meta": sync_meta,
        "runs": all_runs,
        "queue": all_queue,
        "storage": all_storage,
        "fairshare": all_fairshare,
        "completeness_matrix": completeness_matrix,
        "missing_predictions": missing_predictions,
        "stalled_trainings": stalled_trainings,
        "pending_trainings": pending_trainings,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"wrote {out_path} ({len(all_runs)} runs, {len(completeness_matrix)} matrix rows, "
          f"{len(missing_predictions)} missing-prediction alerts, "
          f"{len(stalled_trainings)} stalled-training alerts, "
          f"{len(pending_trainings)} pending-training notices)")


if __name__ == "__main__":
    main()
