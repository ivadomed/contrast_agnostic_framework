"""
Shared dataset-layout constants for the mri_synthesis_project dashboard.
Used by scan_cluster.py (status scanning) and dice_loader.py (per-case Dice
metric loading) so the two never drift out of sync on directory conventions.
"""
import re

RUN_DIR_RE = re.compile(
    r'^(?P<category>auglab|nnUNet)_'
    r'(?P<dataset>.+?)_'
    r'(?P<contrast>[A-Za-z0-9]+)_'
    r'(?P<method>.+)_'
    r'(?P<timestamp>\d{8}_\d{6})$'
)

# Predictions-side run dirs (01_predictions/{model}/{contrast}/{category}/{run})
# drop the "{category}_" prefix that metrics-side dirs carry -- category is
# already given by the parent directory name there. Used only to discover
# "orphan" runs that finished training/predicting but were never eval'd (no
# 02_metrics counterpart exists for them at all, so RUN_DIR_RE never sees
# them by walking the metrics side).
PRED_RUN_DIR_RE = re.compile(
    r'^(?P<dataset>.+?)_'
    r'(?P<contrast>[A-Za-z0-9]+)_'
    r'(?P<method>.+)_'
    r'(?P<timestamp>\d{8}_\d{6})$'
)

# datasets known to always train both contrasts
DATASET_CONTRASTS = {
    "chaos": ["t1in", "t2spir"],
    "open-ms": ["flair", "t1w"],
    "brats2024-glioma": ["t1n", "t2w"],
    "on-harmony": ["T1w", "T2w"],
}

DATASET_RESULTS_DIRNAME = {
    "chaos": "8_results_chaos",
    "open-ms": "8_results_open-ms",
    "brats2024-glioma": "8_results_brats2024-glioma",
    "on-harmony": "8_results_on-harmony",
}

DATASET_MODEL_DIRNAME = {
    "chaos": "chaos_model",
    "open-ms": "open_ms_model",
    "brats2024-glioma": "brats2024_glioma_model",
    "on-harmony": "on_harmony_model",
}

FOLDS_EXPECTED = 4
# nnU-Net folds are trained/evaluated 0-3 (4 folds) for every dataset/contrast
# in this project. Used by scan_cluster.py's classify_run() eval/train/predict
# rollup and dice_loader.py's default fold range for per-case Dice loading.
# Open-MS/t1w is a genuine, still-running experiment (fold3 not yet submitted
# as of this writing) -- it is intentionally left at FOLDS_EXPECTED=4 like
# everything else, so it correctly shows as incomplete (3/4 folds done) in
# the dashboard rather than silently redefining "complete" as 3 folds
# project-wide (which would misclassify every other dataset's real fold3
# data as unnecessary).

# If a fold has an actively-RUNNING squeue job but its training_log_*.txt
# (or checkpoint) hasn't been touched in longer than this, we surface it as
# "stalled" rather than "in_progress" — nnU-Net writes an epoch record every
# 60-90s in this pipeline, so anything past ~30 idle minutes on a job the
# scheduler still considers RUNNING likely means a hang/crash-without-exit
# rather than legitimate training.
STALL_THRESHOLD_SECONDS = 1800


# ---------------------------------------------------------------------------
# Results Explorer: method-comparison presets, ablation-ladder presets, and
# the train-fraction / val-config knob they're parameterized over.
#
# All patterns match against the `method` field parsed out of a run_dir_name
# by RUN_DIR_RE/PRED_RUN_DIR_RE above -- never against the full run_dir_name
# itself, so these presets are agnostic to timestamp/category prefix.
#
# "{train}" / "{val}" placeholders are substituted at resolution time (in
# app.py) with the user-selected train-fraction / val-config strings before
# the regex is compiled, so a single preset definition covers every
# train025/050/090 x val000/100 combination without hardcoding one.
# ---------------------------------------------------------------------------

# Selectable train-fraction / val-config knob for the two "Ours" presets.
# "050"/"000" (train050_val000) is the config most consistently reported as
# the deployed/paper number across datasets in prior analysis -- shipped as
# the default, but left fully adjustable since the best config differs by
# dataset (e.g. CHAOS standalone PALETTE peaks at train050_val100).
TRAIN_FRACTION_CHOICES = ["025", "050", "090"]
VAL_CONFIG_CHOICES = ["000", "100"]
DEFAULT_TRAIN_FRACTION = "050"
DEFAULT_VAL_CONFIG = "000"

# Method-comparison presets: the fixed baseline/method set requested for the
# "existing previous results" prepared-plot view. `category` restricts which
# run_dir_name prefix (auglab_ vs nnUNet_) is eligible -- both are searched
# only when None. `is_template=True` means method_regex contains "{train}"/
# "{val}" placeholders that must be `.format(train=..., val=...)`-substituted
# before compiling.
METHOD_PRESETS = [
    {
        "key": "baseline",
        "label": "Baseline (no augmentation)",
        "category": "nnUNet",
        "method_regex": r"^baseline$",
        "is_template": False,
    },
    {
        "key": "synthseg_noEM",
        "label": "SynthSeg (no EM)",
        "category": "auglab",
        # plain "synthseg_noEM" on t2spir/brats/on-harmony; CHAOS t1in only
        # ever ran the "_train100_val000" variant -- match either.
        "method_regex": r"^synthseg_noEM(?:_train100_val000)?$",
        "is_template": False,
    },
    {
        "key": "synthseg_EM",
        "label": "SynthSeg (EM)",
        "category": "auglab",
        "method_regex": r"^synthseg_EM(?:_train100_val000)?$",
        "is_template": False,
    },
    {
        "key": "srcsm",
        "label": "SRCSM (Thaler et al., IEEE Access 2025)",
        "category": "auglab",
        "method_regex": r"^srcsm$",
        "is_template": False,
    },
    {
        "key": "auglab_default",
        "label": "AugLab (default)",
        "category": "auglab",
        "method_regex": r"^auglab_default$",
        "is_template": False,
    },
    {
        "key": "ours_auglab",
        "label": "Ours + AugLab (deployed) — train{train}/val{val}",
        "category": "auglab",
        "method_regex": r"^auglabAug_v26_6_2_train{train}_val{val}$",
        "is_template": True,
    },
    # Always-shown fixed comparison point, regardless of the train-fraction/
    # val-config selectors above -- requested so train050/val100 is visible
    # right next to whatever the selectors are currently showing. Not a
    # template (no {train}/{val} substitution), so it always resolves to the
    # same concrete run regardless of selector state. resolve_presets()
    # dedupes this against the selector-driven "ours_auglab" preset above
    # when the selectors are ALSO set to train050/val100, so the plot never
    # shows the same bar twice.
    {
        "key": "ours_auglab_fixed_train050_val100",
        "label": "Ours + AugLab (deployed) — train050/val100",
        "category": "auglab",
        "method_regex": r"^auglabAug_v26_6_2_train050_val100$",
        "is_template": False,
    },
]

# Simple ablation ladder: Base -> Ours-alone -> Ours+AugLab-deployed. Every
# rung exists for every one of the 4 datasets, so this is the ladder to show
# when the full mechanism ladder below isn't available.
ABLATION_LADDER_SIMPLE = [
    {"key": "base", "label": "Base", "category": "nnUNet", "method_regex": r"^baseline$", "is_template": False},
    {"key": "ours", "label": "+ Ours (PALETTE)", "category": "nnUNet",
     "method_regex": r"^v26_6_2_train{train}_val{val}$", "is_template": True},
    {"key": "ours_auglab", "label": "+ Ours + AugLab (deployed)", "category": "auglab",
     "method_regex": r"^auglabAug_v26_6_2_train{train}_val{val}$", "is_template": True},
]

# Full mechanism ablation ladder (kmeans -> +label_remap -> +Voronoi), shipped
# in two families (plain-nnUNet and auglab-wrapped) so both the base pipeline
# and the deployed AugLab pipeline can be inspected rung-by-rung. Confirmed
# present ONLY for open-ms as of this session (regex parse of all 112 real
# run directories across chaos/brats2024-glioma/on-harmony/open-ms found the
# kmeans/label_remap/voronoi method tokens exclusively under open-ms) -- the
# app must surface this restriction explicitly rather than silently showing
# an empty plot for the other 3 datasets.
ABLATION_LADDER_FULL_DATASETS = {"open-ms"}

ABLATION_LADDER_FULL = [
    {"key": "base", "label": "Base", "category": "nnUNet",
     "method_regex": r"^baseline$", "is_template": False},
    {"key": "kmeans", "label": "+ k-means", "category": "nnUNet",
     "method_regex": r"^baseline_kmeans_train{train}_val{val}$", "is_template": True},
    {"key": "label_remap", "label": "+ k-means + label remap", "category": "nnUNet",
     "method_regex": r"^baseline_kmeans_label_remap_train{train}_val{val}$", "is_template": True},
    {"key": "voronoi", "label": "+ k-means + label remap + Voronoi", "category": "nnUNet",
     "method_regex": r"^baseline_kmeans_label_remap_voronoi_train{train}_val{val}$", "is_template": True},
    {"key": "base_auglab", "label": "Base (+ AugLab)", "category": "auglab",
     "method_regex": r"^auglab_default$", "is_template": False},
    {"key": "kmeans_auglab", "label": "+ k-means (+ AugLab)", "category": "auglab",
     "method_regex": r"^auglab_kmeans_train{train}_val{val}$", "is_template": True},
    {"key": "label_remap_auglab", "label": "+ k-means + label remap (+ AugLab)", "category": "auglab",
     "method_regex": r"^auglab_kmeans_label_remap_train{train}_val{val}$", "is_template": True},
    {"key": "voronoi_auglab", "label": "+ k-means + label remap + Voronoi (+ AugLab)", "category": "auglab",
     "method_regex": r"^auglab_kmeans_label_remap_voronoi_train{train}_val{val}$", "is_template": True},
]


def resolve_preset_regex(preset, train_fraction=DEFAULT_TRAIN_FRACTION, val_config=DEFAULT_VAL_CONFIG):
    """Substitute {train}/{val} placeholders (if any) and compile."""
    pattern = preset["method_regex"]
    if preset.get("is_template"):
        pattern = pattern.format(train=train_fraction, val=val_config)
    return re.compile(pattern)


def resolve_preset_label(preset, train_fraction=DEFAULT_TRAIN_FRACTION, val_config=DEFAULT_VAL_CONFIG):
    """Substitute {train}/{val} placeholders in a preset's display label (if
    any) with the actual selected config, so the UI never shows a raw
    '{train}/{val}' literal."""
    label = preset["label"]
    if preset.get("is_template"):
        label = label.format(train=train_fraction, val=val_config)
    return label


def metrics_run_dir(project_root, dataset, contrast, run_dir_name):
    """.../02_metrics/{model}/{contrast}/{run_dir_name}/"""
    results_dirname = DATASET_RESULTS_DIRNAME[dataset]
    model_dirname = DATASET_MODEL_DIRNAME[dataset]
    return (project_root / "datasets" / dataset / results_dirname / "02_metrics"
            / model_dirname / contrast / run_dir_name)


def eval_csv_path(project_root, dataset, contrast, run_dir_name, fold):
    return metrics_run_dir(project_root, dataset, contrast, run_dir_name) / f"fold{fold}" / "eval_all.csv"
