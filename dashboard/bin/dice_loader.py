"""
On-demand loader for per-case Dice/HD95 metrics, for the dashboard's
interactive plotting panel. Reads eval_all.csv files directly (schema:
group,case,label,dice,hd95 — confirmed identical across chaos/open-ms/
brats2024-glioma/on-harmony) rather than any pre-aggregated summary, so the
plot always reflects exactly what's on disk for the selected run/fold(s).

Designed to run INSIDE the Streamlit process on the vulcan login node.
Runs hosted on vulcan are read directly off the local filesystem. Runs
hosted on another cluster (killarney) are fetched live over SSH -- reusing
the same multiplexed control-socket convention as sync_remote_status.sh
(ControlMaster/ControlPersist configured once via an interactive
`ssh killarney.alliancecan.ca true` MFA login, good for 30 days of
inactivity) -- via a plain `cat` of the remote eval_all.csv, parsed
in-memory. This never opens a fresh SSH session on its own (BatchMode=yes,
short ConnectTimeout), so a dead/expired control socket fails soft: the
caller gets RemoteCsvUnavailable with an actionable message instead of the
page hanging or crashing.
"""
import functools
import io
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd

from dashboard_common import eval_csv_path, FOLDS_EXPECTED

# Hostnames for the SSH-fetch fallback, keyed by the same cluster names used
# throughout status_combined.json / completeness_matrix rows.
REMOTE_CLUSTER_HOSTS = {
    "vulcan": "vulcan.alliancecan.ca",
    "killarney": "killarney.alliancecan.ca",
}

# How long a fetched remote CSV (or a confirmed-absent one) stays cached
# in-process before a re-fetch is attempted. Short enough that a run which
# finishes evaluating mid-session shows up without a full app restart; long
# enough that flipping dataset/contrast/train/val controls repeatedly
# doesn't re-SSH for the same file on every rerun.
_REMOTE_CACHE_TTL_SECONDS = 120
_REMOTE_SSH_TIMEOUT_SECONDS = 15


class RemoteCsvUnavailable(Exception):
    """Raised when the requested run's eval_all.csv lives on a cluster this
    process cannot read directly AND the live SSH fetch fallback also failed
    (e.g. the control-socket to that cluster is stale/expired, the remote
    file doesn't exist, or it came back empty/malformed)."""


_EMPTY_COLUMNS = ["group", "case", "label", "dice", "hd95",
                  "dataset", "contrast", "run_dir_name", "fold"]

_remote_csv_cache = {}  # (host, path_str) -> (fetched_at_monotonic, df_or_None)


def _parse_csv_bytes(raw_bytes):
    """Shared validation for both local and remote CSV bytes/paths -- returns
    None (never raises) for anything empty/truncated/malformed."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
        return None
    expected = {"group", "case", "label", "dice", "hd95"}
    if not expected.issubset(df.columns):
        return None
    return df


@functools.lru_cache(maxsize=256)
def _read_csv_cached(path_str, mtime_ns):
    """mtime_ns busts the cache if the file is rewritten (e.g. a rerun with
    more folds finished since the page was last loaded). Returns None (rather
    than raising) if the file is empty/truncated/malformed -- this can happen
    when a job is killed or still writing eval_all.csv at scan time."""
    try:
        raw = Path(path_str).read_bytes()
    except OSError:
        return None
    return _parse_csv_bytes(raw)


def _fetch_remote_csv(host, path_str):
    """SSH-cat a remote eval_all.csv via the pre-existing multiplexed control
    socket and parse it in-memory. Returns None on ANY failure (dead socket,
    remote file missing, non-zero exit, timeout, malformed content) -- this
    is a soft-fail path by design, mirroring sync_remote_status.sh."""
    cache_key = (host, path_str)
    cached = _remote_csv_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < _REMOTE_CACHE_TTL_SECONDS:
        return cached[1]
    df = None
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
             f"cat {shlex.quote(path_str)}"],
            capture_output=True, timeout=_REMOTE_SSH_TIMEOUT_SECONDS,
        )
        if proc.returncode == 0 and proc.stdout:
            df = _parse_csv_bytes(proc.stdout)
    except (subprocess.TimeoutExpired, OSError):
        df = None
    _remote_csv_cache[cache_key] = (time.monotonic(), df)
    return df


def load_fold_dice(project_root, dataset, contrast, run_dir_name, fold, this_cluster, run_cluster):
    """Load one fold's eval_all.csv as a DataFrame with added dataset/contrast/
    run_dir_name/fold columns. Returns an empty DataFrame if the file doesn't
    exist yet (fold not evaluated), or is empty/malformed (e.g. a job was
    killed mid-write), rather than raising. Raises RemoteCsvUnavailable only
    when run_cluster != this_cluster AND the live SSH fetch fallback failed."""
    path = eval_csv_path(Path(project_root), dataset, contrast, run_dir_name, fold)
    if run_cluster != this_cluster:
        host = REMOTE_CLUSTER_HOSTS.get(run_cluster)
        if host is None:
            raise RemoteCsvUnavailable(
                f"Run '{run_dir_name}' lives on cluster '{run_cluster}', which has no "
                f"configured SSH hostname for a live fetch."
            )
        df = _fetch_remote_csv(host, str(path))
        if df is None:
            raise RemoteCsvUnavailable(
                f"Run '{run_dir_name}' lives on cluster '{run_cluster}'; live SSH fetch "
                f"of its eval_all.csv from '{host}' failed (control socket may be stale -- "
                f"run `ssh {host} true` interactively once to refresh it -- or the file "
                f"isn't evaluated/present yet). Per-case Dice values were skipped for this run."
            )
        df = df.copy()
    else:
        if not path.exists():
            return pd.DataFrame(columns=_EMPTY_COLUMNS)
        mtime_ns = path.stat().st_mtime_ns
        cached = _read_csv_cached(str(path), mtime_ns)
        if cached is None:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)
        df = cached.copy()
    df["dataset"] = dataset
    df["contrast"] = contrast
    df["run_dir_name"] = run_dir_name
    df["fold"] = fold
    return df


def load_run_dice(project_root, dataset, contrast, run_dir_name, this_cluster, run_cluster,
                   folds=None):
    """Concatenate all available folds (default: all FOLDS_EXPECTED) for one
    run into a single per-case DataFrame."""
    if folds is None:
        folds = range(FOLDS_EXPECTED)
    frames = [
        load_fold_dice(project_root, dataset, contrast, run_dir_name, f, this_cluster, run_cluster)
        for f in folds
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["group", "case", "label", "dice", "hd95",
                                      "dataset", "contrast", "run_dir_name", "fold"])
    return pd.concat(frames, ignore_index=True)


def tag_domain(df, contrast):
    """Add a 'domain' column: 'in_domain' when a row's eval `group` matches
    the run's own training contrast (case-insensitive), 'out_of_domain'
    otherwise. This is the single, dataset-agnostic domain-split rule used
    across all 4 datasets -- e.g. for a CHAOS run trained on t1in, groups
    t1out/t2spir/ct are all out-of-domain, with no per-dataset lookup table
    required. No-op (returns df unchanged) if df is empty or lacks 'group'."""
    if df.empty or "group" not in df.columns:
        return df
    df = df.copy()
    contrast_norm = str(contrast).strip().lower()
    df["domain"] = df["group"].astype(str).str.strip().str.lower().apply(
        lambda g: "in_domain" if g == contrast_norm else "out_of_domain"
    )
    return df


def filter_domain(df, domain_choice):
    """domain_choice: 'all' | 'in_domain' | 'out_of_domain'. Requires df to
    already carry a 'domain' column (see tag_domain). Returns df unchanged
    for 'all' or if the column is missing (nothing to filter on)."""
    if domain_choice == "all" or df.empty or "domain" not in df.columns:
        return df
    return df[df["domain"] == domain_choice]


def load_multi_run_dice(project_root, run_specs, this_cluster):
    """run_specs: list of dicts each with dataset/contrast/run_dir_name/cluster
    (as found in completeness_matrix rows). Returns one combined long-format
    DataFrame tagged with a 'run_label' column for plotting, skipping (with a
    warning list returned alongside) any runs on a different cluster than
    this dashboard process. Each row is also tagged with a 'domain' column
    (see tag_domain) so callers can filter to in-domain / out-of-domain /
    all without a second pass over the source CSVs."""
    frames = []
    skipped = []
    for spec in run_specs:
        label = f"{spec['dataset']}/{spec['contrast']}/{spec['category']}/{spec['method']}"
        try:
            df = load_run_dice(
                project_root, spec["dataset"], spec["contrast"], spec["run_dir_name"],
                this_cluster, spec["cluster"],
            )
        except RemoteCsvUnavailable as e:
            skipped.append({"run_label": label, "reason": str(e)})
            continue
        if df.empty:
            continue
        df["run_label"] = label
        df = tag_domain(df, spec["contrast"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["group", "case", "label", "dice", "hd95",
                                      "dataset", "contrast", "run_dir_name", "fold",
                                      "run_label", "domain"]), skipped
    return pd.concat(frames, ignore_index=True), skipped
