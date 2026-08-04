#!/usr/bin/env python3
"""
Streamlit dashboard for the mri_synthesis_project: cross-cluster (vulcan +
killarney) training/prediction/evaluation status, queue/storage/allocation
health, and an interactive per-case Dice plotter.

Run on the vulcan login node (see dashboard/README.md for launch + SSH
tunnel instructions):
    /project/aip-jcohen/paulh/mri_synthesis_project/.venv/bin/streamlit run \\
        /project/aip-jcohen/paulh/mri_synthesis_project/dashboard/bin/app.py \\
        --server.port 8765 --server.address 127.0.0.1
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dice_loader import load_multi_run_dice, RemoteCsvUnavailable, filter_domain  # noqa: E402
from dashboard_common import (  # noqa: E402
    STALL_THRESHOLD_SECONDS, METHOD_PRESETS, ABLATION_LADDER_SIMPLE, ABLATION_LADDER_FULL,
    ABLATION_LADDER_FULL_DATASETS, resolve_preset_regex, resolve_preset_label, TRAIN_FRACTION_CHOICES,
    VAL_CONFIG_CHOICES, DEFAULT_TRAIN_FRACTION, DEFAULT_VAL_CONFIG, DATASET_CONTRASTS,
)

PROJECT_ROOT = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
STATUS_COMBINED = DASHBOARD_DIR / "status_combined.json"
THIS_CLUSTER = "vulcan"  # this app is only ever launched on vulcan's login node

st.set_page_config(page_title="MRI Synthesis — Run Status", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_combined_status(_mtime):
    with open(STATUS_COMBINED) as f:
        return json.load(f)


def get_status():
    if not STATUS_COMBINED.exists():
        st.error(
            f"No status file found at {STATUS_COMBINED}. Run "
            f"`bash dashboard/bin/refresh_all.sh` at least once, or wait for cron."
        )
        st.stop()
    mtime = STATUS_COMBINED.stat().st_mtime
    return load_combined_status(mtime), mtime


def age_str(iso_ts):
    if not iso_ts:
        return "never"
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return iso_ts
    delta = datetime.now(timezone.utc) - dt
    secs = delta.total_seconds()
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{secs / 3600:.1f}h ago"
    return f"{secs / 86400:.1f}d ago"


def duration_str(secs):
    """Format a plain (non-relative) duration in seconds, e.g. an ETA."""
    if secs is None or pd.isna(secs) or secs < 0:
        return "?"
    secs = float(secs)
    if secs < 60:
        return f"{int(secs)}s"
    if secs < 3600:
        return f"{int(secs // 60)}m"
    if secs < 86400:
        h = secs / 3600
        return f"{h:.1f}h"
    return f"{secs / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Sidebar: refresh control + cluster health badges
# ---------------------------------------------------------------------------

def render_sidebar(status):
    st.sidebar.title("MRI Synthesis Dashboard")
    st.sidebar.caption(f"Combined status generated {age_str(status['generated_at'])}")

    if st.sidebar.button("🔄 Refresh now", width="stretch"):
        with st.spinner("Running refresh_all.sh (scans vulcan, syncs killarney, recombines)..."):
            result = subprocess.run(
                ["bash", str(DASHBOARD_DIR / "bin" / "refresh_all.sh")],
                capture_output=True, text=True, timeout=120,
            )
        st.sidebar.code(result.stdout[-2000:] or result.stderr[-2000:])
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Cluster health")
    for cluster, meta in status["clusters"].items():
        badge = "🟢" if not meta.get("stale") and not meta.get("load_error") else "🟡"
        if meta.get("load_error"):
            badge = "🔴"
        st.sidebar.markdown(f"{badge} **{cluster}** — scanned {age_str(meta.get('scanned_at'))}")
        if meta.get("stale_reason"):
            st.sidebar.caption(f"⚠️ {meta['stale_reason']}")
        if meta.get("load_error"):
            st.sidebar.caption(f"⚠️ load error: {meta['load_error']}")

    sync_meta = status.get("sync_meta", {})
    if sync_meta.get("last_sync_ok") is False:
        st.sidebar.warning(
            "Last killarney sync failed (SSH control socket likely expired). "
            "Run `python3 dashboard/bin/refresh_killarney_socket.py` on vulcan, "
            "approve the Duo push on your phone, then refresh."
        )


# ---------------------------------------------------------------------------
# Panel: pending / running jobs
# ---------------------------------------------------------------------------

def _job_stage(name):
    """
    Classify a queue job's pipeline stage from its Slurm job name.
    Training jobs are named "fold<N>_<dataset>_..." (see scan_cluster.py's
    fold-prefixed convention); predict/eval jobs use dataset/method-prefixed
    names without that "foldN_" lead-in (e.g. "chaos_eval_...",
    "amos_predict_..."). Falls back to "other" for anything that parses as
    neither -- keeps this cosmetic classification from ever raising on an
    unexpected name.
    """
    if not isinstance(name, str):
        return "other"
    if re.match(r"^fold\d+_", name):
        return "train"
    if "_eval_" in name or name.endswith("_eval") or "evaluate" in name:
        return "eval"
    if "_predict" in name or "predict_" in name:
        return "predict"
    return "other"


def render_queue_panel(status):
    st.header("Queue: pending / running jobs")
    queue = status.get("queue", [])
    if not queue:
        st.info("No jobs currently in either cluster's queue.")
        return
    df = pd.DataFrame(queue)
    if "name" in df.columns:
        df["stage"] = df["name"].apply(_job_stage)
    else:
        df["stage"] = "other"

    if {"current_epoch", "num_epochs"}.issubset(df.columns):
        def _progress(row):
            ce, ne = row.get("current_epoch"), row.get("num_epochs")
            if pd.notna(ce) and pd.notna(ne):
                pct = row.get("epoch_pct")
                pct_str = f" ({pct:.0f}%)" if pd.notna(pct) else ""
                return f"{int(ce)}/{int(ne)}{pct_str}"
            return ""
        df["progress"] = df.apply(_progress, axis=1)
    else:
        df["progress"] = ""
    if "eta_seconds" in df.columns:
        df["eta"] = df["eta_seconds"].apply(lambda s: duration_str(s) if pd.notna(s) else "")
    else:
        df["eta"] = ""

    cols = ["cluster", "job_id", "state", "stage", "dataset", "contrast", "method", "fold",
            "progress", "eta", "elapsed", "submit_time", "reason", "name"]
    cols = [c for c in cols if c in df.columns]
    sort_keys = [c for c in ["cluster", "state", "dataset"] if c in df.columns]
    df = df[cols].sort_values(sort_keys, na_position="last") if sort_keys else df[cols]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queued", len(df))
    c2.metric("Running", int((df["state"] == "RUNNING").sum()) if "state" in df else 0)
    c3.metric("Pending", int((df["state"] == "PENDING").sum()) if "state" in df else 0)
    if "stage" in df.columns:
        c4.metric("Training / Predict+Eval", f'{(df["stage"] == "train").sum()} / '
                                              f'{df["stage"].isin(["predict", "eval"]).sum()}')

    st.caption(
        "**stage** splits queued/running jobs into training (`fold<N>_...` job names) vs "
        "predict/eval (dataset/method-prefixed names) so you can see at a glance which part "
        "of the pipeline the cluster is spending allocation on right now. **progress**/**eta** "
        "come from the training log's last logged epoch (steady-state estimate: remaining "
        "epochs × last epoch duration) and are only populated for training jobs currently mid-run."
    )
    st.dataframe(df, width="stretch", hide_index=True, height=min(400, 40 + 35 * len(df)))


# ---------------------------------------------------------------------------
# Panel: completeness matrix
# ---------------------------------------------------------------------------

def render_completeness_matrix(status):
    st.header("Training → prediction → eval completeness matrix")
    matrix = status.get("completeness_matrix", [])
    if not matrix:
        st.info("No runs found.")
        return
    df = pd.DataFrame(matrix)
    if "train_stalled" not in df.columns:
        df["train_stalled"] = 0
    if "train_pending" not in df.columns:
        df["train_pending"] = 0
    if "orphan" not in df.columns:
        df["orphan"] = False
    n_folds = df["n_folds"].replace(0, pd.NA)
    df["train_pct"] = (df["train_done"] / n_folds * 100).round(0)
    df["predict_pct"] = (df["predict_done"] / n_folds * 100).round(0)
    df["eval_pct"] = (df["eval_done"] / n_folds * 100).round(0)
    df["flags"] = (
        df["train_stalled"].apply(lambda n: "🛑 stalled" if n else "")
        + df["train_pending"].apply(lambda n: (" " if n else "") + ("⏳ queued" if n else ""))
        + df["orphan"].apply(lambda o: (" " if o else "") + ("🧩 orphan (no metrics dir)" if o else ""))
    ).str.strip()

    n_orphans = int(df["orphan"].sum())
    n_stalled_rows = int((df["train_stalled"] > 0).sum())
    n_pending_rows = int((df["train_pending"] > 0).sum())
    if n_orphans or n_stalled_rows or n_pending_rows:
        st.caption(
            f"🧩 {n_orphans} orphan run(s) with predictions but no metrics directory (never evaluated) · "
            f"🛑 {n_stalled_rows} run(s) with at least one stalled training fold — see the "
            f"Attention needed tab for details · "
            f"⏳ {n_pending_rows} run(s) with at least one fold queued (still training, not an alert)."
        )

    datasets = sorted(df["dataset"].unique())
    sel_datasets = st.multiselect("Filter by dataset", datasets, default=datasets)
    df_view = df[df["dataset"].isin(sel_datasets)] if sel_datasets else df

    display_cols = ["dataset", "contrast", "category", "method", "cluster",
                     "train_done", "train_in_progress", "train_stalled", "train_pending", "n_folds",
                     "predict_done", "eval_done", "train_pct", "predict_pct", "eval_pct",
                     "flags", "mtime"]
    st.dataframe(
        df_view[display_cols].sort_values(["dataset", "contrast", "category", "method"]),
        width="stretch", hide_index=True,
        height=min(600, 40 + 35 * len(df_view)),
        column_config={
            "train_pct": st.column_config.ProgressColumn("train %", min_value=0, max_value=100, format="%d%%"),
            "predict_pct": st.column_config.ProgressColumn("predict %", min_value=0, max_value=100, format="%d%%"),
            "eval_pct": st.column_config.ProgressColumn("eval %", min_value=0, max_value=100, format="%d%%"),
        },
    )
    return df_view


# ---------------------------------------------------------------------------
# Panel: missing-predictions alert list
# ---------------------------------------------------------------------------

def render_missing_predictions(status):
    st.header("⚠️ Missing-predictions alert list")
    alerts = status.get("missing_predictions", [])
    if not alerts:
        st.success("No gaps: every trained fold has been predicted and evaluated.")
        return
    df = pd.DataFrame(alerts)

    if "inflight" in df.columns:
        def _inflight_badge(v):
            if not v or (isinstance(v, float) and pd.isna(v)):
                return ""
            state = v.get("state", "")
            if state == "RUNNING":
                return "🟢 running"
            if state == "PENDING":
                return "🟡 queued"
            return f"🔵 {state}" if state else "🔵 in queue"
        df["status"] = df["inflight"].apply(_inflight_badge)
        n_inflight = int((df["status"] != "").sum())
    else:
        df["status"] = ""
        n_inflight = 0

    n_needs_submit = len(df) - n_inflight
    st.warning(
        f"{len(df)} fold(s) have a completed upstream stage but a missing downstream one "
        f"({n_needs_submit} need submitting, {n_inflight} already queued/running)."
    )
    st.caption(
        "**status** shows 🟢 running / 🟡 queued when a matching predict/eval job is already "
        "live in squeue for that dataset+method+fold -- no action needed there, it just hasn't "
        "finished yet. A blank status means nothing is in flight for it: worth submitting."
    )
    display_cols = [c for c in ["issue", "status", "dataset", "contrast", "category", "method",
                                 "cluster", "fold", "run_dir_name"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(["issue", "dataset", "contrast", "method"]),
        width="stretch", hide_index=True,
        height=min(400, 40 + 35 * len(df)),
    )


# ---------------------------------------------------------------------------
# Panel: stalled-trainings alert list
# ---------------------------------------------------------------------------

def render_stalled_trainings(status):
    st.header("🛑 Stalled-training alert list")
    st.caption(
        "A fold is flagged here when Slurm still shows its job RUNNING but "
        "training_log_*.txt hasn't advanced in over "
        f"{STALL_THRESHOLD_SECONDS // 60} minutes — usually a hang or a "
        "crash that didn't release the allocation. Worth an `scontrol show job` / "
        "log tail, and likely a requeue."
    )
    alerts = status.get("stalled_trainings", [])
    if not alerts:
        st.success("No stalled folds detected: every RUNNING job has a recently-updated training log.")
        return
    df = pd.DataFrame(alerts)
    if "seconds_since_log_update" in df.columns:
        df["stalled_for"] = df["seconds_since_log_update"].apply(
            lambda s: f"{s/3600:.1f}h" if pd.notna(s) and s >= 3600 else (f"{int(s//60)}m" if pd.notna(s) else "?")
        )
    st.warning(f"{len(df)} fold(s) appear stalled.")
    display_cols = [c for c in ["dataset", "contrast", "category", "method", "cluster", "fold",
                                 "current_epoch", "num_epochs", "stalled_for", "run_dir_name"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(["dataset", "contrast", "method"]),
        width="stretch", hide_index=True,
        height=min(400, 40 + 35 * len(df)),
    )


# ---------------------------------------------------------------------------
# Panel: pending-trainings notice list (queued, not stalled -- informational)
# ---------------------------------------------------------------------------

def render_pending_trainings(status):
    st.header("⏳ Training in queue (pending)")
    st.caption(
        "A fold is listed here when Slurm shows its job PENDING (queued, e.g. "
        "preempted-and-requeued, or simply waiting on partition priority) rather "
        "than RUNNING or done. Training is genuinely still in progress -- this is "
        "**not** an alert, nothing needs your attention, and predict/eval correctly "
        "wait until training actually finishes."
    )
    alerts = status.get("pending_trainings", [])
    if not alerts:
        st.success("No folds currently queued.")
        return
    df = pd.DataFrame(alerts)
    st.info(f"{len(df)} fold(s) queued and waiting for a Slurm allocation to resume training.")
    if {"current_epoch", "num_epochs"}.issubset(df.columns):
        df["progress"] = df.apply(
            lambda r: f"{r['current_epoch']}/{r['num_epochs']}"
            if pd.notna(r["current_epoch"]) and pd.notna(r["num_epochs"]) else "?",
            axis=1,
        )
    display_cols = [c for c in ["dataset", "contrast", "category", "method", "cluster", "fold",
                                 "progress", "run_dir_name"] if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values(["dataset", "contrast", "method"]),
        width="stretch", hide_index=True,
        height=min(400, 40 + 35 * len(df)),
    )


# ---------------------------------------------------------------------------
# Panel: interactive Dice plotter
# ---------------------------------------------------------------------------

def render_dice_plotter(status, matrix_df):
    st.header("Interactive Dice plotter")
    if matrix_df is None or matrix_df.empty:
        st.info("No runs available to plot.")
        return

    matrix_df = matrix_df.copy()
    matrix_df["run_label"] = (matrix_df["dataset"] + " / " + matrix_df["contrast"] + " / "
                               + matrix_df["category"] + " / " + matrix_df["method"]
                               + "  [" + matrix_df["cluster"] + "]")
    options = matrix_df.sort_values("run_label")["run_label"].tolist()
    default = options[:2] if len(options) >= 2 else options
    selected_labels = st.multiselect("Select runs to compare", options, default=default)

    if not selected_labels:
        st.info("Select at least one run above.")
        return

    specs = []
    for label in selected_labels:
        row = matrix_df[matrix_df["run_label"] == label].iloc[0]
        specs.append({
            "dataset": row["dataset"], "contrast": row["contrast"],
            "category": row["category"], "method": row["method"],
            "run_dir_name": row["run_dir_name"], "cluster": row["cluster"],
        })

    with st.spinner("Loading per-case Dice from eval_all.csv..."):
        combined, skipped = load_multi_run_dice(str(PROJECT_ROOT), specs, THIS_CLUSTER)

    for s in skipped:
        st.warning(f"Skipped **{s['run_label']}**: {s['reason']}")

    if combined.empty:
        st.info("No per-case Dice data available yet for the selected run(s) (eval not run).")
        return

    plot_kind = st.radio("Plot type", ["Box plot by run", "Box plot by label", "Per-case scatter"],
                          horizontal=True)

    import plotly.express as px

    if plot_kind == "Box plot by run":
        fig = px.box(combined, x="run_label", y="dice", points="all",
                     color="run_label", title="Per-case Dice distribution by run")
        fig.update_layout(showlegend=False, xaxis_title="", height=500)
    elif plot_kind == "Box plot by label":
        fig = px.box(combined, x="label", y="dice", color="run_label", points=False,
                     title="Per-case Dice distribution by anatomical label / run")
        fig.update_layout(height=500)
    else:
        fig = px.strip(combined, x="run_label", y="dice", color="label",
                        hover_data=["case", "fold", "hd95"],
                        title="Per-case Dice (each point = one case/label/fold)")
        fig.update_layout(height=500)

    st.plotly_chart(fig, width="stretch")

    with st.expander("Summary statistics"):
        summary = combined.groupby("run_label")["dice"].agg(["mean", "std", "median", "count"]).round(4)
        st.dataframe(summary, width="stretch")

    with st.expander("Raw per-case data"):
        st.dataframe(combined, width="stretch", height=300)
        st.download_button(
            "Download as CSV", combined.to_csv(index=False).encode(),
            file_name="dice_comparison.csv", mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Results Explorer: prepared method-comparison + ablation-ladder plots for
# supervisor-facing presentation. Resolves METHOD_PRESETS / ABLATION_LADDER_*
# (dashboard_common.py) against the already-scanned completeness_matrix --
# no changes to scan_cluster.py/combine_status.py are needed for this tab.
# ---------------------------------------------------------------------------

def resolve_presets(matrix_df, dataset, contrast, presets, train_fraction, val_config):
    """For each preset, find the newest-by-mtime matching completeness_matrix
    row (if any) for this dataset/contrast, and classify its availability.
    Returns a list of dicts: {preset, status, row}. status is one of:
    'found' (has per-case eval data ready to plot), 'in_progress' (actively
    training/stalled per Slurm), 'not_evaluated' (run exists but no eval yet),
    'missing' (no matching run_dir at all for this train/val config).

    Presets that resolve to an identical (category, regex-pattern) after
    {train}/{val} substitution are deduplicated, keeping only the first
    occurrence -- this matters for always-shown fixed presets (e.g. a
    hardcoded train050/val100 comparison point) that can coincide with a
    selector-driven preset when the selectors happen to land on the same
    config, so the plot never shows the exact same run twice."""
    results = []
    seen_dedup_keys = set()
    sub = matrix_df[(matrix_df["dataset"] == dataset) & (matrix_df["contrast"] == contrast)]
    for preset in presets:
        rx = resolve_preset_regex(preset, train_fraction, val_config)
        dedup_key = (preset.get("category"), rx.pattern)
        if dedup_key in seen_dedup_keys:
            continue
        seen_dedup_keys.add(dedup_key)
        candidates = sub[sub["method"].astype(str).apply(lambda m: bool(rx.match(m)))]
        cat = preset.get("category")
        if cat is not None:
            candidates = candidates[candidates["category"] == cat]
        if candidates.empty:
            results.append({"preset": preset, "status": "missing", "row": None})
            continue
        row = candidates.sort_values("mtime").iloc[-1]
        if row.get("eval_done", 0) and row["eval_done"] > 0:
            status = "found"
        elif (row.get("train_in_progress", 0) > 0 or row.get("train_stalled", 0) > 0
              or row.get("train_pending", 0) > 0):
            status = "in_progress"
        else:
            status = "not_evaluated"
        results.append({"preset": preset, "status": status, "row": row})
    return results


def render_preset_availability(resolved, train_fraction, val_config):
    """Small table + explicit callouts for missing/in-progress/not-yet-evaluated
    presets, so gaps in the supervisor-facing plot are never silent."""
    rows = []
    for r in resolved:
        row = r["row"]
        rows.append({
            "preset": resolve_preset_label(r["preset"], train_fraction, val_config),
            "status": r["status"],
            "run_dir_name": row["run_dir_name"] if row is not None else "—",
            "cluster": row["cluster"] if row is not None else "—",
            "orphan": bool(row["orphan"]) if row is not None else False,
        })
    df = pd.DataFrame(rows)
    n_missing = (df["status"] == "missing").sum()
    n_pending = df["status"].isin(["in_progress", "not_evaluated"]).sum()
    if n_missing:
        st.warning(f"{n_missing} preset(s) have no matching run at this train/val config: "
                    + ", ".join(df[df['status'] == 'missing']['preset']))
    if n_pending:
        st.info(f"{n_pending} preset(s) exist but aren't plottable yet (still training or "
                f"not evaluated): " + ", ".join(df[df['status'].isin(['in_progress', 'not_evaluated'])]['preset']))
    with st.expander("Preset resolution detail", expanded=False):
        st.dataframe(df, width="stretch", hide_index=True)


def build_specs_and_labels(resolved, train_fraction, val_config):
    """From resolved presets with status=='found', build the run_specs list
    for load_multi_run_dice plus a run_dir_name -> paper-style preset label
    map (so the plot x-axis shows 'SynthSeg (EM)' etc. rather than the raw
    method token)."""
    specs, label_map = [], {}
    for r in resolved:
        if r["status"] != "found":
            continue
        row = r["row"]
        specs.append({
            "dataset": row["dataset"], "contrast": row["contrast"],
            "category": row["category"], "method": row["method"],
            "run_dir_name": row["run_dir_name"], "cluster": row["cluster"],
        })
        label_map[row["run_dir_name"]] = resolve_preset_label(r["preset"], train_fraction, val_config)
    return specs, label_map


def load_and_label(specs, label_map, domain_choice):
    combined, skipped = load_multi_run_dice(str(PROJECT_ROOT), specs, THIS_CLUSTER)
    for s in skipped:
        st.warning(f"Skipped **{s['run_label']}**: {s['reason']}")
    if combined.empty:
        return combined
    combined = filter_domain(combined, domain_choice)
    combined["preset_label"] = combined["run_dir_name"].map(label_map)
    return combined


def summary_table(df, group_col):
    agg = df.groupby(group_col, observed=True)["dice"].agg(mean="mean", std="std", n="count")
    n_cases = df.groupby(group_col, observed=True)["case"].nunique().rename("n_cases")
    out = agg.join(n_cases).round(4)
    return out


def render_method_comparison(matrix_df, dataset, contrast, train_fraction, val_config, domain_choice):
    st.subheader("Method comparison")
    st.caption(
        "Prepared comparison across the fixed baseline/method set: Baseline, "
        "SynthSeg (no EM / EM), SRCSM, AugLab (default), and Ours + AugLab "
        "(the deployed configuration) -- shown at the selected train/val config "
        "plus a fixed train050/val100 reference point."
    )
    resolved = resolve_presets(matrix_df, dataset, contrast, METHOD_PRESETS, train_fraction, val_config)
    render_preset_availability(resolved, train_fraction, val_config)

    specs, label_map = build_specs_and_labels(resolved, train_fraction, val_config)
    if not specs:
        st.info("No presets are plottable yet for this dataset/contrast/config.")
        return

    with st.spinner("Loading per-case Dice from eval_all.csv..."):
        combined = load_and_label(specs, label_map, domain_choice)
    if combined.empty:
        st.info("No per-case Dice data available for the selected domain filter.")
        return

    all_labels = [resolve_preset_label(p, train_fraction, val_config) for p in METHOD_PRESETS]
    # dict.fromkeys dedups while preserving first-occurrence order -- needed
    # because the fixed train050/val100 reference preset can produce a label
    # identical to the selector-driven "Ours + AugLab" preset when the
    # selectors themselves land on train050/val100 (resolve_presets() already
    # dedupes the underlying data by (category, regex); this just keeps the
    # category ordering in sync so pd.Categorical doesn't see a repeated
    # category value).
    seen_labels = [lbl for lbl in all_labels if lbl in combined["preset_label"].unique()]
    order = list(dict.fromkeys(seen_labels))
    combined["preset_label"] = pd.Categorical(combined["preset_label"], categories=order, ordered=True)

    import plotly.express as px
    plot_kind = st.radio("Plot type", ["Box plot", "Bar (mean ± std)"], horizontal=True, key="mc_plot_kind")
    domain_tag = {"all": "all cases", "in_domain": "in-domain cases only",
                  "out_of_domain": "out-of-domain cases only"}[domain_choice]
    title = f"{dataset} / {contrast} — method comparison ({domain_tag})"

    # Same preset -> color mapping for both plot kinds, so a method keeps its
    # color whether the supervisor is looking at the box or bar view.
    palette = px.colors.qualitative.Plotly
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(order)}

    if plot_kind == "Box plot":
        fig = px.box(combined.sort_values("preset_label"), x="preset_label", y="dice", points="all",
                     color="preset_label", category_orders={"preset_label": order},
                     color_discrete_map=color_map, title=title)
        fig.update_layout(showlegend=True, legend_title_text="Method", xaxis_title="", height=500)
    else:
        summ = summary_table(combined, "preset_label").reindex(order).dropna(how="all").reset_index()
        fig = px.bar(summ, x="preset_label", y="mean", error_y="std", color="preset_label",
                     category_orders={"preset_label": order}, color_discrete_map=color_map, title=title)
        fig.update_layout(showlegend=True, legend_title_text="Method",
                           xaxis_title="", yaxis_title="Mean Dice", height=500)
    st.plotly_chart(fig, width="stretch")
    st.caption("Use the camera icon in the plot toolbar to export a PNG.")

    with st.expander("Summary statistics (mean ± std, n cases/rows)"):
        st.dataframe(summary_table(combined, "preset_label").reindex(order), width="stretch")

    with st.expander("Raw per-case data"):
        st.dataframe(combined, width="stretch", height=300)
        st.download_button(
            "Download as CSV", combined.to_csv(index=False).encode(),
            file_name=f"{dataset}_{contrast}_method_comparison.csv", mime="text/csv",
            key="mc_download",
        )


def render_ablation_ladder(matrix_df, dataset, contrast, train_fraction, val_config, domain_choice):
    st.subheader("Ablation ladder")
    full_available = dataset in ABLATION_LADDER_FULL_DATASETS
    ladder_choice = st.radio(
        "Ladder", ["Simple (Base → Ours → Ours+AugLab)", "Full mechanism (k-means → label remap → Voronoi)"],
        horizontal=True, key="ladder_choice",
    )
    if ladder_choice.startswith("Full") and not full_available:
        st.warning(
            f"The granular mechanism ladder (k-means / label-remap / Voronoi rungs) was only "
            f"run for **open-ms** in this project -- no such runs exist for **{dataset}**. "
            f"Showing the simple ladder instead."
        )
        ladder = ABLATION_LADDER_SIMPLE
    elif ladder_choice.startswith("Full"):
        ladder = ABLATION_LADDER_FULL
    else:
        ladder = ABLATION_LADDER_SIMPLE

    st.caption(
        "Mechanism-ablation story: each rung adds one component of the synthetic-contrast "
        "augmentation (k-means parcellation → label remapping → Voronoi contrast synthesis), "
        "showing which piece drives the cross-contrast generalization gain."
        if ladder is ABLATION_LADDER_FULL else
        "Coarse ablation: unaugmented baseline vs. PALETTE alone vs. the deployed PALETTE+AugLab "
        "configuration -- available for every dataset."
    )

    resolved = resolve_presets(matrix_df, dataset, contrast, ladder, train_fraction, val_config)
    render_preset_availability(resolved, train_fraction, val_config)

    specs, label_map = build_specs_and_labels(resolved, train_fraction, val_config)
    if not specs:
        st.info("No ladder rungs are plottable yet for this dataset/contrast/config.")
        return

    with st.spinner("Loading per-case Dice from eval_all.csv..."):
        combined = load_and_label(specs, label_map, domain_choice)
    if combined.empty:
        st.info("No per-case Dice data available for the selected domain filter.")
        return

    all_labels = [resolve_preset_label(p, train_fraction, val_config) for p in ladder]
    order = [lbl for lbl in all_labels if lbl in combined["preset_label"].unique()]
    combined["preset_label"] = pd.Categorical(combined["preset_label"], categories=order, ordered=True)

    import plotly.express as px
    domain_tag = {"all": "all cases", "in_domain": "in-domain cases only",
                  "out_of_domain": "out-of-domain cases only"}[domain_choice]
    title = f"{dataset} / {contrast} — ablation ladder ({domain_tag})"

    summ = summary_table(combined, "preset_label").reindex(order).dropna(how="all").reset_index()
    fig = px.line(summ, x="preset_label", y="mean", error_y="std", markers=True,
                  category_orders={"preset_label": order}, title=title)
    fig.update_layout(xaxis_title="", yaxis_title="Mean Dice", height=500)
    st.plotly_chart(fig, width="stretch")
    st.caption("Use the camera icon in the plot toolbar to export a PNG.")

    with st.expander("Summary statistics (mean ± std, n cases/rows)"):
        st.dataframe(summary_table(combined, "preset_label").reindex(order), width="stretch")

    with st.expander("Raw per-case data"):
        st.dataframe(combined, width="stretch", height=300)
        st.download_button(
            "Download as CSV", combined.to_csv(index=False).encode(),
            file_name=f"{dataset}_{contrast}_ablation_ladder.csv", mime="text/csv",
            key="ladder_download",
        )


def render_results_explorer(status, matrix_df):
    st.header("📊 Results Explorer")
    st.caption(
        "Prepared, paper-style comparison plots for showing results to your supervisor: "
        "a fixed method-comparison set per dataset x modality, plus ablation-ladder views "
        "with an in-domain / out-of-domain / whole-data toggle. Built entirely from the "
        "already-scanned run inventory -- nothing here re-scans the clusters."
    )
    if matrix_df is None or matrix_df.empty:
        st.info("No runs available yet.")
        return

    datasets = sorted(d for d in DATASET_CONTRASTS if d in matrix_df["dataset"].unique())
    if not datasets:
        st.info("No known datasets found in the completeness matrix.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dataset = st.selectbox("Dataset", datasets, key="re_dataset")
    contrasts = [c for c in DATASET_CONTRASTS[dataset] if c in matrix_df[matrix_df["dataset"] == dataset]["contrast"].unique()]
    # Guard against a stale widget value carried over from a previously
    # selected dataset whose contrast list differs (e.g. open-ms "flair"
    # is not a valid CHAOS contrast) -- Streamlit raises if a keyed
    # widget's session_state value isn't among its current options.
    if contrasts and st.session_state.get("re_contrast") not in contrasts:
        st.session_state["re_contrast"] = contrasts[0]
    with col2:
        contrast = st.selectbox("Contrast / modality", contrasts, key="re_contrast")
    with col3:
        train_fraction = st.selectbox("Train fraction", TRAIN_FRACTION_CHOICES,
                                       index=TRAIN_FRACTION_CHOICES.index(DEFAULT_TRAIN_FRACTION), key="re_train")
    with col4:
        val_config = st.selectbox("Val config", VAL_CONFIG_CHOICES,
                                   index=VAL_CONFIG_CHOICES.index(DEFAULT_VAL_CONFIG), key="re_val")

    domain_choice = st.radio(
        "Domain split", ["all", "in_domain", "out_of_domain"],
        format_func=lambda d: {"all": "Whole data", "in_domain": "In-domain only",
                                "out_of_domain": "Out-of-domain only"}[d],
        horizontal=True, key="re_domain",
        help="In-domain = eval group matches this run's own training contrast. "
             "Out-of-domain = every other evaluated group (e.g. CHAOS trained on t1in, "
             "evaluated on t1out/t2spir/ct).",
    )

    st.divider()
    render_method_comparison(matrix_df, dataset, contrast, train_fraction, val_config, domain_choice)
    st.divider()
    render_ablation_ladder(matrix_df, dataset, contrast, train_fraction, val_config, domain_choice)


# ---------------------------------------------------------------------------
# Panel: cluster / storage / allocation health
# ---------------------------------------------------------------------------

def render_health_widgets(status):
    st.header("Cluster / storage / allocation health")
    storage = status.get("storage", [])
    fairshare = status.get("fairshare", [])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Storage quotas")
        if storage:
            df = pd.DataFrame(storage)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No storage data.")
    with col2:
        st.subheader("Fairshare / priority")
        if fairshare:
            df = pd.DataFrame(fairshare)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No fairshare data.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    status, _ = get_status()
    render_sidebar(status)

    n_missing = len(status.get("missing_predictions", []))
    n_stalled = len(status.get("stalled_trainings", []))
    n_pending = len(status.get("pending_trainings", []))
    attention_label = "Attention needed"
    if n_missing or n_stalled:
        attention_label = f"⚠️ Attention needed ({n_missing + n_stalled})"

    tabs = st.tabs(["Overview", "Completeness matrix", attention_label,
                     "Dice plotter", "Results Explorer", "Cluster health"])

    with tabs[0]:
        render_queue_panel(status)
    with tabs[1]:
        matrix_df = render_completeness_matrix(status)
    with tabs[2]:
        render_missing_predictions(status)
        st.divider()
        render_stalled_trainings(status)
        st.divider()
        render_pending_trainings(status)
    with tabs[3]:
        # completeness matrix must be computed to populate the run selector;
        # recompute quietly here if the user jumps straight to this tab.
        mdf = pd.DataFrame(status.get("completeness_matrix", []))
        render_dice_plotter(status, mdf)
    with tabs[4]:
        mdf2 = pd.DataFrame(status.get("completeness_matrix", []))
        render_results_explorer(status, mdf2)
    with tabs[5]:
        render_health_widgets(status)


if __name__ == "__main__":
    main()
