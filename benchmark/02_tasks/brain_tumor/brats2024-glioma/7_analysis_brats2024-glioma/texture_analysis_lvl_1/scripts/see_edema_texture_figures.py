#!/usr/bin/env python
"""
"See the phenomenon" figures for the T2w<->T1n / T1n<->T2f real-fill-vs-noise-fill edema story.

Produces 4 PNGs under ../outputs/plots/:
  1. see_edema_texture_gallery.png  - raw+highpass image w/ contours + intensity histograms,
                                       for 3 patients picked to have a large edema and a
                                       t2w(train)->t1n(eval) SNFH delta_dice near the median of
                                       the negative (real-fill-worse) cases.
  2. see_fingerprints.png           - autocorrelation-decay texture fingerprints, edema vs ring,
                                       per contrast, plus 4 train->eval comparison panels.
  3. see_predictions_t2w_model.png  - GT vs noise-fill vs real-fill EDEMA predictions for the
                                       T2w-trained model, evaluated on t1n/t1c/t2f, same 3 patients.
  4. see_predictions_mirror.png     - mirror case (t1n-trained model on FLAIR vs FLAIR-trained
                                       model on T1n). NOTE: the ablation prediction volumes for
                                       these two training directions are NOT preserved on Vulcan
                                       (only fold-level eval_all.csv metrics survive; see report).
                                       This figure therefore shows image+GT context plus the
                                       exact per-patient Dice deltas from those stored CSVs,
                                       rather than predicted-mask overlays.

Read-only w.r.t. the repo except for new files under outputs/{plots,logs}. CPU only. Run via
run_job on Vulcan, never on the login node (see run_see_edema_texture_figures.sh).
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter, center_of_mass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_cross_contrast_ngf import (  # noqa: E402
    CONTRAST_SUFFIX, LABEL_IDS, load_patient, region_masks, METRICS_ROOT,
)
from compute_region_surround_texture import (  # noqa: E402
    highpass, region_parts, acf_vector, ACF_COLS, LAGS, MIN_VOX,
)

DS_ROOT = THIS_DIR.parents[2]
LVL1 = THIS_DIR.parent
DATA_DIR = LVL1 / "outputs" / "data"
PLOTS_DIR = LVL1 / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_ROOT = DS_ROOT / "8_results_brats2024-glioma" / "01_predictions" / "brats2024_glioma_model"

# ── color palette (Okabe-Ito, colorblind-safe) ──────────────────────────────────────────────
C_EDEMA = "#D55E00"   # vermillion - edema region contour
C_RING = "#0072B2"    # blue - surrounding ring contour (dashed)
C_CORE = "#009E73"    # green - tumor core contour (dotted)
C_FP = "#CC79A7"      # pink - false positive fill
C_FN = "#F0A202"      # amber - false negative fill
C_CONTRAST = {"t1n": "#0072B2", "t1c": "#D55E00", "t2w": "#009E73", "t2f": "#CC79A7"}

CONTRASTS_ORDER = ("t1n", "t1c", "t2w", "t2f")
D_OUT_PRIMARY = 5.0
FOLD_DISPLAY = "fold0"


# ───────────────────────── shared helpers ─────────────────────────
def edema_mask3d(pid: str):
    """Loads label only (cheap) to get the SNFH voxel count for patient screening."""
    label_f = DS_ROOT / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset051_BraTS2024GliomaT1n" / "labelsTr" / f"{pid}.nii.gz"
    lab = np.asarray(nib.load(str(label_f)).dataobj).round().astype(np.int64)
    return int((lab == LABEL_IDS["SNFH"]).sum())


def pick_display_patients(train, eval_c, region, n=3):
    """3 patients: large edema (top-half by SNFH voxel count among the eval cohort), delta_dice
    closest to the median of the NEGATIVE (real-fill-worse) cases for (train, eval, region)."""
    df = pd.read_csv(DATA_DIR / "ngf_vs_ladder_per_patient.csv")
    sub = df[(df.train == train) & (df["eval"] == eval_c) & (df.region == region)].copy()
    neg = sub[sub.delta_dice < 0]
    if len(neg) == 0:
        raise RuntimeError(f"no negative delta_dice rows for {train}->{eval_c}/{region}")
    med = neg.delta_dice.median()
    log.info("%s->%s/%s: %d cases, %d negative, median(negative) delta_dice=%.4f",
             train, eval_c, region, len(sub), len(neg), med)
    sizes = {pid: edema_mask3d(pid) for pid in neg.case.unique()}
    neg = neg.assign(edema_vox=neg.case.map(sizes))
    thresh = neg.edema_vox.median()
    large = neg[neg.edema_vox >= thresh].copy()
    if len(large) < n:
        large = neg.copy()
    large["dist_to_med"] = (large.delta_dice - med).abs()
    chosen = large.sort_values("dist_to_med").head(n)
    log.info("chosen patients:\n%s", chosen[["case", "delta_dice", "edema_vox"]].to_string(index=False))
    return chosen["case"].tolist(), med


def bbox_from_mask(mask: np.ndarray, margin=15):
    idx = np.array(np.nonzero(mask))
    lo = np.maximum(idx.min(axis=1) - margin, 0)
    hi = np.minimum(idx.max(axis=1) + margin + 1, np.array(mask.shape))
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def window_pctl(img: np.ndarray, brain: np.ndarray, lo=1, hi=99):
    v = img[brain]
    if v.size == 0:
        return float(img.min()), float(img.max())
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def load_pred(train, category, run, eval_c, fold, pid):
    p = PRED_ROOT / train / category / run / fold / eval_c / f"{pid}.nii.gz"
    if not p.exists():
        return None
    return np.asarray(nib.load(str(p)).dataobj).round().astype(np.int64)


def load_eval_dice(train, rung_dir_rel, eval_c, label, fold=FOLD_DISPLAY):
    csv = METRICS_ROOT / train / rung_dir_rel / fold / "eval_all.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[(df.group == eval_c) & (df.label == label)]
    if row.empty:
        return None
    return row


T2W_RUNS = {  # relative to 01_predictions/brats2024_glioma_model/t2w/  AND  02_metrics/.../t2w/
    "voronoi": ("auglab", "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659"),
    "realfill": ("nnUNet", "brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"),
}


# ───────────────────────── FIGURE 1: texture gallery ─────────────────────────
def make_fig1(patients, train, eval_c, region):
    per_patient_paths = []
    row_h = 3.4  # inches per contrast row

    for pid in patients:
        vols, label = load_patient(pid, "cpu")
        masks = {k: m.numpy() for k, m in region_masks(label, vols["t1n"]).items()}
        brain = masks["healthy"] | masks["whole_tumor"]
        region_m = masks[region]
        parts = region_parts(region_m, brain, masks["healthy"])
        ring_m = parts[("ring", D_OUT_PRIMARY)]
        core_m = masks["tumor_core"]

        z = int(round(center_of_mass(region_m)[2]))
        bbox3 = bbox_from_mask(masks["whole_tumor"], margin=15)
        bbox2 = (bbox3[0], bbox3[1])
        dim0, dim1 = bbox2[0].stop - bbox2[0].start, bbox2[1].stop - bbox2[1].start
        r = max(dim0 / dim1, 0.35)  # width/height ratio of the crop; floor avoids absurdly thin cols

        fig, axes = plt.subplots(
            4, 3, figsize=(row_h * (2 * r + 1.7), row_h * 4),
            gridspec_kw={"width_ratios": [r, r, 1.7]})
        fig.suptitle(f"patient {pid}  (train={train} eval={eval_c} region={region})", fontsize=11, y=1.01)

        for row_i, c in enumerate(CONTRASTS_ORDER):
            raw = vols[c].numpy().astype(np.float64)
            hp = highpass(raw, brain)
            vmin, vmax = window_pctl(raw, brain)

            raw_crop = raw[bbox2][..., z]
            hp_crop = hp[bbox2][..., z]
            reg_crop = region_m[bbox2][..., z]
            ring_crop = ring_m[bbox2][..., z]
            core_crop = core_m[bbox2][..., z]

            ax = axes[row_i, 0]
            ax.imshow(raw_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            if reg_crop.any():
                ax.contour(reg_crop.T.astype(float), levels=[0.5], colors=[C_EDEMA], linewidths=1.4)
            if ring_crop.any():
                ax.contour(ring_crop.T.astype(float), levels=[0.5], colors=[C_RING], linestyles="dashed", linewidths=1.1)
            if core_crop.any():
                ax.contour(core_crop.T.astype(float), levels=[0.5], colors=[C_CORE], linestyles="dotted", linewidths=1.3)
            ax.set_ylabel(c, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title("raw + contours", fontsize=9)

            hp_lo, hp_hi = window_pctl(hp, brain)
            ax = axes[row_i, 1]
            ax.imshow(hp_crop.T, cmap="gray", vmin=hp_lo, vmax=hp_hi, origin="lower")
            if reg_crop.any():
                ax.contour(reg_crop.T.astype(float), levels=[0.5], colors=[C_EDEMA], linewidths=1.4)
            if ring_crop.any():
                ax.contour(ring_crop.T.astype(float), levels=[0.5], colors=[C_RING], linestyles="dashed", linewidths=1.1)
            if core_crop.any():
                ax.contour(core_crop.T.astype(float), levels=[0.5], colors=[C_CORE], linestyles="dotted", linewidths=1.3)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title("highpass + contours", fontsize=9)

            ax = axes[row_i, 2]
            v_reg, v_ring = hp[region_m], hp[ring_m]
            lo, hi = np.percentile(np.concatenate([v_reg, v_ring]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 60)
            ax.hist(v_reg, bins=bins, density=True, alpha=0.55, color=C_EDEMA, label="edema")
            ax.hist(v_ring, bins=bins, density=True, alpha=0.55, color=C_RING, label="ring (d<=5)")
            ax.set_yticks([])
            ax.tick_params(labelsize=7)
            if row_i == 0:
                ax.set_title("highpass value distribution", fontsize=9)
                ax.legend(fontsize=7, loc="upper right", frameon=False)

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.text(0.5, 0.012,
                  "Each row = one contrast, same axial slice through the edema centroid, same tumor-bbox crop. "
                  "Solid vermillion = edema (SNFH); dashed blue = surrounding ring (1-5 vox); dotted green = tumor core. "
                  "Right column: if the edema and ring histograms overlap heavily in a contrast, that contrast's local "
                  "texture cannot distinguish edema from its surroundings there.",
                  ha="center", fontsize=9, wrap=True)
        out_p = PLOTS_DIR / f"see_edema_texture_gallery_{pid}.png"
        fig.savefig(out_p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        per_patient_paths.append(out_p)

    _stack_pngs(per_patient_paths, PLOTS_DIR / "see_edema_texture_gallery.png", cleanup=True)
    log.info("wrote see_edema_texture_gallery.png")


# ───────────────────────── FIGURE 2: fingerprints ─────────────────────────
def load_fingerprint_table():
    frames = [pd.read_csv(DATA_DIR / f"region_surround_texture_shard{i}.csv")
              for i in range(4) if (DATA_DIR / f"region_surround_texture_shard{i}.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def curve_median_iqr(df, contrast, region, part, d_out, variant, axes_group):
    sub = df[(df.contrast == contrast) & (df.region == region) & (df.part == part)
             & (df.variant == variant)]
    if part == "region":
        sub = sub[sub.d_out.isna()]
    else:
        sub = sub[sub.d_out == d_out]
    if sub.empty:
        return None, None, None, 0
    cols = [f"acf_ax{a}_lag{k}" for a in axes_group for k in LAGS]
    # average across axes in the group, per patient, per lag
    per_lag = []
    for k in LAGS:
        lag_cols = [f"acf_ax{a}_lag{k}" for a in axes_group]
        per_lag.append(sub[lag_cols].mean(axis=1))
    mat = np.vstack(per_lag).T  # n_patients x n_lags
    med = np.nanmedian(mat, axis=0)
    q1 = np.nanpercentile(mat, 25, axis=0)
    q3 = np.nanpercentile(mat, 75, axis=0)
    return med, q1, q3, mat.shape[0]


def make_fig2(region="SNFH"):
    df = load_fingerprint_table()
    pairs = [("t2w", "t1n"), ("t2w", "t1c"), ("t1n", "t2f"), ("t2f", "t1n")]

    combo = plt.figure(figsize=(17, 13.5))
    sf = combo.subfigures(2, 1, height_ratios=[7.2, 5.0], hspace=0.09)
    axes_top = sf[0].subplots(2, 4)
    for col, contrast in enumerate(CONTRASTS_ORDER):
        for row_i, (axes_group, label) in enumerate([((0, 1), "in-plane (ax0,ax1)"), ((2,), "axis 2 (S-I)")]):
            ax = axes_top[row_i, col]
            med_r, q1_r, q3_r, n_r = curve_median_iqr(df, contrast, region, "region", np.nan, "highpass", axes_group)
            med_g, q1_g, q3_g, n_g = curve_median_iqr(df, contrast, region, "ring", D_OUT_PRIMARY, "highpass", axes_group)
            x = list(LAGS)
            if med_r is not None:
                ax.plot(x, med_r, "-", color=C_CONTRAST[contrast], lw=2, label=f"edema (n={n_r})")
                ax.fill_between(x, q1_r, q3_r, color=C_CONTRAST[contrast], alpha=0.2)
            if med_g is not None:
                ax.plot(x, med_g, "--", color=C_CONTRAST[contrast], lw=1.6, label=f"ring (n={n_g})")
                ax.fill_between(x, q1_g, q3_g, color=C_CONTRAST[contrast], alpha=0.1)
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_ylim(-0.3, 1.0)
            ax.set_xticks(x)
            ax.legend(fontsize=6.5, loc="upper right", frameon=False)
            if row_i == 0:
                ax.set_title(contrast, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"autocorr.\n{label}", fontsize=8.5)
            if row_i == 1:
                ax.set_xlabel("lag (voxels)", fontsize=8.5)
    sf[0].suptitle("(A) Edema vs. ring texture fingerprints, per contrast (median+IQR over patients)", fontsize=11, y=1.03)
    sf[0].subplots_adjust(top=0.86, bottom=0.14, hspace=0.35, wspace=0.3)

    axes_bot = sf[1].subplots(1, 4)
    for i, (train, ev) in enumerate(pairs):
        ax = axes_bot[i]
        med_train, q1_t, q3_t, n_t = curve_median_iqr(df, train, region, "region", np.nan, "highpass", (0, 1))
        med_eval_r, q1_er, q3_er, n_er = curve_median_iqr(df, ev, region, "region", np.nan, "highpass", (0, 1))
        med_eval_g, q1_eg, q3_eg, n_eg = curve_median_iqr(df, ev, region, "ring", D_OUT_PRIMARY, "highpass", (0, 1))
        x = list(LAGS)
        ax.plot(x, med_train, "-", color="black", lw=2.2, label=f"{train} (train) edema")
        ax.plot(x, med_eval_r, "-", color=C_CONTRAST[ev], lw=1.8, label=f"{ev} (eval) edema")
        ax.plot(x, med_eval_g, "--", color=C_CONTRAST[ev], lw=1.8, label=f"{ev} (eval) ring")
        ax.set_title(f"{train} -> {ev}", fontsize=10)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylim(-0.3, 1.0)
        ax.set_xticks(x)
        ax.set_xlabel("lag (voxels)", fontsize=8.5)
        ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    axes_bot[0].set_ylabel("in-plane autocorr.", fontsize=9)
    sf[1].suptitle("(B) Training-contrast edema fingerprint vs. eval-contrast edema/ring (real-fill bakes in the "
                    "training contrast's edema texture; compare where it lands)", fontsize=10.5, y=1.06)
    sf[1].subplots_adjust(top=0.80, bottom=0.28, wspace=0.3)

    combo.text(0.5, 0.01,
               "Panel A: solid=edema, dashed=ring, per contrast, highpass autocorrelation vs lag (median+IQR, all 70 eval cases). "
               "Panel B: black solid = training-contrast edema fingerprint; colored solid/dashed = eval-contrast edema/ring. "
               "When black sits closer to eval-edema than eval-ring, the training texture cue still points at the right place; "
               "when it sits closer to eval-ring, real-fill's learned texture is misleading there (t2w->t1n case).",
               ha="center", fontsize=9, wrap=True)
    combo.savefig(PLOTS_DIR / "see_fingerprints.png", dpi=160, bbox_inches="tight")
    plt.close(combo)
    log.info("wrote see_fingerprints.png")


# ───────────────────────── FIGURE 3: t2w-model predictions ─────────────────────────
def dice_score(pred_bin, gt_bin):
    inter = int((pred_bin & gt_bin).sum())
    denom = int(pred_bin.sum() + gt_bin.sum())
    return float("nan") if denom == 0 else 2 * inter / denom


def make_fig3(patients, train="t2w", eval_contrasts=("t1n", "t1c", "t2f"), region="SNFH", fold=FOLD_DISPLAY):
    lid = LABEL_IDS[region]
    for pid in patients:
        vols, label_full = load_patient(pid, "cpu")
        label_np = label_full.numpy()
        masks = {k: m.numpy() for k, m in region_masks(label_full, vols["t1n"]).items()}
        bbox3 = bbox_from_mask(masks["whole_tumor"], margin=15)
        bbox2 = (bbox3[0], bbox3[1])
        z = int(round(center_of_mass(masks[region])[2]))
        gt_bin_full = (label_np == lid)

        fig, axes = plt.subplots(len(eval_contrasts), 5, figsize=(16, 3.6 * len(eval_contrasts)))
        if len(eval_contrasts) == 1:
            axes = axes[None, :]

        for row_i, ev in enumerate(eval_contrasts):
            img = vols[ev].numpy().astype(np.float64)
            brain = masks["healthy"] | masks["whole_tumor"]
            vmin, vmax = window_pctl(img, brain)
            img_crop = img[bbox2][..., z]
            gt_crop = gt_bin_full[bbox2][..., z]

            cat_v, run_v = T2W_RUNS["voronoi"]
            cat_r, run_r = T2W_RUNS["realfill"]
            pred_v = load_pred(train, cat_v, run_v, ev, fold, pid)
            pred_r = load_pred(train, cat_r, run_r, ev, fold, pid)

            ax = axes[row_i, 0]
            ax.imshow(img_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_ylabel(f"eval={ev}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title("eval image", fontsize=9)

            ax = axes[row_i, 1]
            ax.imshow(img_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            if gt_crop.any():
                ax.contourf(gt_crop.T.astype(float), levels=[0.5, 1.5], colors=[C_EDEMA], alpha=0.45)
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title("ground truth", fontsize=9)

            for col_i, (name, pred) in enumerate([("noise-fill", pred_v), ("real-fill", pred_r)]):
                ax = axes[row_i, 2 + col_i]
                ax.imshow(img_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                if pred is None:
                    ax.text(0.5, 0.5, "prediction\nmissing", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="red")
                    d = float("nan")
                else:
                    pred_bin_full = (pred == lid)
                    d = dice_score(pred_bin_full, gt_bin_full)
                    pred_crop = pred_bin_full[bbox2][..., z]
                    if pred_crop.any():
                        ax.contourf(pred_crop.T.astype(float), levels=[0.5, 1.5], colors=[C_RING], alpha=0.45)
                ax.set_xticks([]); ax.set_yticks([])
                if row_i == 0:
                    ax.set_title(f"{name} pred", fontsize=9)
                ax.set_xlabel(f"Dice={d:.3f}" if d == d else "Dice=n/a", fontsize=8.5)

            ax = axes[row_i, 4]
            ax.imshow(img_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", alpha=0.6)
            if pred_v is not None and pred_r is not None:
                pv = (pred_v == lid)[bbox2][..., z]
                pr = (pred_r == lid)[bbox2][..., z]
                fp = pr & ~pv   # real-fill-only positive (relative to noise-fill)
                fn = pv & ~pr   # noise-fill-only positive (lost by real-fill)
                rgba = np.zeros((*fp.shape, 4))
                rgba[fp] = matplotlib.colors.to_rgba(C_FP, 0.6)
                rgba[fn] = matplotlib.colors.to_rgba(C_FN, 0.6)
                ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title("real-fill vs noise-fill\n(pink=gained,amber=lost)", fontsize=8.5)

        fig.suptitle(f"T2w-trained model, patient {pid} - edema (SNFH) predictions, {fold}", fontsize=12, y=1.03)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.5, 0.012,
                  "Vermillion fill = ground-truth edema; blue fill = predicted edema. Dice computed on the full 3D volume "
                  "(not just this slice). Last column: pink = voxels real-fill predicts as edema but noise-fill does not "
                  "(gain); amber = the reverse (loss).",
                  ha="center", fontsize=8.5, wrap=True)
        fig.savefig(PLOTS_DIR / f"see_predictions_t2w_model_{pid}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    log.info("wrote see_predictions_t2w_model_<patient>.png (%d patients)", len(patients))

    # merge per-patient figures into ONE file as requested
    _stack_pngs([PLOTS_DIR / f"see_predictions_t2w_model_{pid}.png" for pid in patients],
                PLOTS_DIR / "see_predictions_t2w_model.png",
                cleanup=True)


def _stack_pngs(paths, out_path, cleanup=False):
    imgs = [plt.imread(str(p)) for p in paths if p.exists()]
    if not imgs:
        log.warning("no images to stack into %s", out_path)
        return
    widths = [im.shape[1] for im in imgs]
    max_w = max(widths)
    padded = []
    for im in imgs:
        if im.shape[1] < max_w:
            pad = max_w - im.shape[1]
            im = np.pad(im, ((0, 0), (0, pad), (0, 0)), constant_values=1.0)
        padded.append(im)
    total_h = sum(im.shape[0] for im in padded) + 10 * (len(padded) - 1)
    canvas = np.ones((total_h, max_w, padded[0].shape[2]))
    y = 0
    for im in padded:
        canvas[y:y + im.shape[0], :im.shape[1]] = im
        y += im.shape[0] + 10
    fig = plt.figure(figsize=(max_w / 160, total_h / 160), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    if cleanup:
        for p in paths:
            if p.exists():
                p.unlink()


# ───────────────────────── FIGURE 4: mirror case ─────────────────────────
def make_fig4(patients_a, patients_b, region="SNFH"):
    """Mirror: (train=t1n, eval=t2f) vs (train=t2f, eval=t1n). Prediction volumes for these two
    ablation directions are not preserved on Vulcan (only fold-level metrics CSVs survive), so
    this figure shows image+GT context (real data) plus the exact per-patient Dice deltas taken
    straight from the stored eval_all.csv-derived table (also real data, not a substitute run)."""
    df = pd.read_csv(DATA_DIR / "ngf_vs_ladder_per_patient.csv")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[2.0, 2.0, 3.4])

    directions = [("t1n", "t2f", patients_a), ("t2f", "t1n", patients_b)]
    for row_i, (train, ev, pats) in enumerate(directions):
        lid = LABEL_IDS[region]
        for col_i, pid in enumerate(pats):
            ax = fig.add_subplot(gs[row_i, 2 * col_i:2 * col_i + 2])
            vols, label_full = load_patient(pid, "cpu")
            masks = {k: m.numpy() for k, m in region_masks(label_full, vols["t1n"]).items()}
            bbox3 = bbox_from_mask(masks["whole_tumor"], margin=15)
            bbox2 = (bbox3[0], bbox3[1])
            z = int(round(center_of_mass(masks[region])[2]))
            brain = masks["healthy"] | masks["whole_tumor"]
            img = vols[ev].numpy().astype(np.float64)
            vmin, vmax = window_pctl(img, brain)
            img_crop = img[bbox2][..., z]
            gt_crop = (label_full.numpy() == lid)[bbox2][..., z]
            ring_crop = masks_ring = region_parts(masks[region], brain, masks["healthy"])[("ring", D_OUT_PRIMARY)][bbox2][..., z]
            ax.imshow(img_crop.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            if gt_crop.any():
                ax.contour(gt_crop.T.astype(float), levels=[0.5], colors=[C_EDEMA], linewidths=1.4)
            if ring_crop.any():
                ax.contour(ring_crop.T.astype(float), levels=[0.5], colors=[C_RING], linestyles="dashed", linewidths=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{pid}\ntrain={train} eval={ev}", fontsize=8.5)

    ax_bar = fig.add_subplot(gs[2, :])
    xw = 0.35
    slot_gap = 1.0     # spacing between patient slots within a direction group
    group_gap = 1.5    # extra spacing between the two direction groups
    xticks, xticklabels = [], []
    cursor = 0.0
    group_spans = []
    for i, (train, ev, pats) in enumerate(directions):
        sub = df[(df.train == train) & (df["eval"] == ev) & (df.region == region)]
        pooled_v, pooled_r = sub.dice_voronoi.mean(), sub.dice_realfill.mean()
        span_start = cursor
        for pid in pats:
            row = sub[sub.case == pid]
            if row.empty:
                continue
            v, r = float(row.dice_voronoi.iloc[0]), float(row.dice_realfill.iloc[0])
            base = cursor
            ax_bar.bar(base, v, width=xw, color=C_RING, alpha=0.9,
                       label="noise-fill" if (i == 0 and pid == pats[0]) else None)
            ax_bar.bar(base + xw, r, width=xw, color=C_EDEMA, alpha=0.9,
                       label="real-fill" if (i == 0 and pid == pats[0]) else None)
            xticks.append(base + xw)
            xticklabels.append(pid[-6:])
            cursor += slot_gap
        span_end = cursor - slot_gap + 2 * xw
        group_spans.append((span_start, span_end, f"{train}-train -> {ev}-eval\n(pooled: noise={pooled_v:.2f} real={pooled_r:.2f})"))
        cursor += group_gap
    ax_bar.set_xticks(xticks)
    ax_bar.set_xticklabels(xticklabels, fontsize=7.5, rotation=20)
    for s, e, label in group_spans:
        ax_bar.text((s + e) / 2, -0.14, label, ha="center", va="top", fontsize=8, transform=ax_bar.get_xaxis_transform())
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel(f"edema ({region}) Dice, {FOLD_DISPLAY}", fontsize=9)
    ax_bar.legend(fontsize=8, loc="upper right", frameon=False)
    ax_bar.set_title("Per-patient noise-fill vs real-fill Dice, both mirror directions", fontsize=9.5)

    fig.suptitle("Mirror case: T1n-trained model on FLAIR (gains with real-fill) vs "
                 "FLAIR-trained model on T1n (loses with real-fill)", fontsize=12, y=1.02)
    fig.subplots_adjust(bottom=0.16, hspace=0.55, top=0.93)
    fig.text(0.5, 0.005,
              "Top two rows: eval image + ground-truth edema (vermillion) + surrounding ring (dashed blue), for context only "
              "- prediction-mask overlays are NOT shown here because the t1n- and t2f-trained ablation models' prediction "
              "volumes were not preserved on Vulcan (only per-fold eval_all.csv metrics remain; see the accompanying report). "
              "Bottom panel: real per-patient edema Dice (noise-fill vs real-fill) for exactly these directions, read from "
              "those stored metrics - this is the same underlying data the +26.3 / -4.1 pooled headline numbers come from.",
              ha="center", fontsize=8.5, wrap=True)
    fig.savefig(PLOTS_DIR / "see_predictions_mirror.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote see_predictions_mirror.png")


def main():
    region = "SNFH"
    patients, med = pick_display_patients("t2w", "t1n", region)
    log.info("FIG1/FIG3 patients (t2w->t1n median-negative, large edema): %s (median negative delta=%.4f)", patients, med)

    make_fig1(patients, "t2w", "t1n", region)
    make_fig2(region)
    make_fig3(patients, train="t2w", eval_contrasts=("t1n", "t1c", "t2f"), region=region)

    pats_a, med_a = pick_display_patients("t1n", "t2f", region)
    pats_b, med_b = pick_display_patients("t2f", "t1n", region)
    log.info("FIG4 direction A (t1n->t2f) patients: %s", pats_a)
    log.info("FIG4 direction B (t2f->t1n) patients: %s", pats_b)
    make_fig4(pats_a, pats_b, region)

    log.info("DONE")


if __name__ == "__main__":
    main()
