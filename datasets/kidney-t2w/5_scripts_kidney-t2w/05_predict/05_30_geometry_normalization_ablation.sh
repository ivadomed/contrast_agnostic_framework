#!/bin/bash
#SBATCH --job-name=kidney_t2w_geom_norm_ablation
#SBATCH --account=aip-jcohen
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/kidney-t2w/_packruns/geom_norm_%j.out
#
# KIDNEY-T2W geometry-normalization ablation.
#
# GOAL: isolate the CONTRAST effect. kidney-t2w is a CORONAL acquisition;
# chaos (the training set) is AXIAL. That geometric difference penalizes
# methods UNEQUALLY (srcsm hardest), so it distorts the method ranking and
# contaminates what is supposed to be a cross-contrast measurement. We want
# the geometrically-normalized input that leaves contrast as the only
# remaining domain difference.
#
# MECHANISM (corrected after 3 earlier ablations, see
# ../../ACQUISITION_PLANE_INVESTIGATION.md):
# nnU-Net resamples every case to the TRAINING PLANS SPACING. chaos's plans
# spacing is (5.5, 1.699, 1.699) in nnU-Net-internal (z,y,x) order, i.e. the
# coarse 5.5mm axis is internal-axis0 == NIfTI array axis2 (SimpleITK reverses
# axis order on read).
#   chaos NIfTI      (256,256,35)  zooms (1.89, 1.89, 5.5)
#                    -> coarse axis already at array axis2. No role mismatch.
#   kidney-t2w NIfTI (240, 14,240) zooms (1.458, 5.5, 1.458)
#                    -> coarse axis at array axis1. ROLE MISMATCH.
# Consequence for un-normalized kidney-t2w: nnU-Net DOWNSAMPLES the rich S-I
# axis 1.458->5.5mm (3.8x, 240 real slices -> ~64, destroying real detail)
# while UPSAMPLING the sparse P-A axis 5.5->1.699mm (3.2x, 14 real slices ->
# ~45, inventing detail). That is the dominant damage -- it is a
# resample-target problem, not an interpolation-quality problem (isotropic
# resampling changed nothing: 0.339 -> 0.332) and not a padding/context
# problem (padding changed literally nothing, byte-identical predictions,
# because nnU-Net already pads internally to patch size).
#
# TWO CANDIDATE NORMALIZATIONS, both making the coarse axis land at array
# axis2 the way chaos's does:
#   (A) swap  -- pure axis permutation (array axes 1<->2, + matching affine
#       column swap). No interpolation at all, so ALL real data stays sharp
#       and GT correspondence is exact. Resulting in-plane view is
#       L-R x S-I = CORONAL (still differs from chaos's axial view), and the
#       through-plane axis is the genuinely-sparse P-A (14 slices @5.5mm --
#       same spacing as chaos's through-plane, just fewer slices).
#       Measured on 8 cases, srcsm fold0: 0.339 -> 0.414 mean Dice.
#   (B) axial -- resample into a true AXIAL grid like chaos's: in-plane
#       L-R x P-A @1.7mm, through-plane S-I @5.5mm. This matches chaos's
#       anatomical VIEW as well as its axis roles, but P-A must be
#       interpolated from only 14 real samples, so each axial slice is sharp
#       in L-R and blurry in P-A. Trades view-fidelity for data-fidelity.
#
# This job runs BOTH normalizations (plus the un-normalized ORIG control)
# over ALL 100 cases and 4 methods (fold0), so we can see whether the
# normalization changes the METHOD RANKING -- which is the actual thing that
# matters for isolating contrast. 8-case spot checks were too noisy to decide
# (one case swung 0.35 -> 0.000).
set -uo pipefail
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
SCRATCH="/scratch/p/paulh"
WORK="${SCRATCH}/kidney-t2w/_packruns/geom_norm_${SLURM_JOB_ID:-manual}"
mkdir -p "${WORK}"

export NNUNET_PROJECT_ROOT="${PROJECT_ROOT}"
RAW="${SCRATCH}/kidney-t2w/2_nnUNet_kidney-t2w/raw"
export nnUNet_preprocessed="${SCRATCH}/chaos/2_nnUNet_chaos/preprocessed"
CHAOS_PRED="${SCRATCH}/chaos/8_results_chaos/01_predictions/chaos_model/t1in"

echo "[geom] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name --format=csv,noheader || true

# ── Phase 1: build the two normalized input variants (+ matching GT) ────────
echo "[geom] ==== PHASE 1: build variants (orig / swap / axial) for all cases ===="
"${PROJECT_ROOT}/.venv/bin/python" - "$WORK" "$RAW" <<'PYEOF'
import sys, os, glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk

WORK, RAW = sys.argv[1], sys.argv[2]
img_in, gt_in = f"{RAW}/imagesTs_t2", f"{RAW}/labelsTs_t2"
cases = sorted(os.path.basename(p)[:-len("_0000.nii.gz")]
               for p in glob.glob(f"{img_in}/*_0000.nii.gz"))
print(f"[geom] {len(cases)} cases", flush=True)

for v in ("orig", "swap", "axial"):
    os.makedirs(f"{WORK}/{v}/img", exist_ok=True)
    os.makedirs(f"{WORK}/{v}/gt", exist_ok=True)

AXIAL_SPACING = (1.7, 1.7, 5.5)   # (L-R, P-A, S-I) == chaos-like axial grid

def swap_axes(path, out, is_seg):
    """Pure axis permutation: array axes 1<->2 + matching affine columns.
    Lossless -- no interpolation, exact GT correspondence."""
    im = nib.load(path)
    data = np.swapaxes(im.get_fdata(), 1, 2)
    aff = im.affine.copy()
    aff[:, [1, 2]] = im.affine[:, [2, 1]]
    nib.save(nib.Nifti1Image(data.astype(im.get_data_dtype()), aff), out)

def axial_reslice(path, out, is_seg):
    """Resample into a chaos-like AXIAL grid (coarse S-I at array axis2)."""
    im = sitk.ReadImage(path)
    size = [int(round(im.GetSize()[i] * im.GetSpacing()[i] / AXIAL_SPACING[i]))
            for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(AXIAL_SPACING); r.SetSize(size)
    r.SetOutputDirection(im.GetDirection()); r.SetOutputOrigin(im.GetOrigin())
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_seg else sitk.sitkBSpline)
    r.SetDefaultPixelValue(0)
    sitk.WriteImage(r.Execute(im), out)

for i, c in enumerate(cases):
    src_i, src_g = f"{img_in}/{c}_0000.nii.gz", f"{gt_in}/{c}.nii.gz"
    # orig: symlink (no copy -- these are the untouched pipeline inputs)
    for src, dst in ((src_i, f"{WORK}/orig/img/{c}_0000.nii.gz"),
                     (src_g, f"{WORK}/orig/gt/{c}.nii.gz")):
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    swap_axes(src_i, f"{WORK}/swap/img/{c}_0000.nii.gz", False)
    swap_axes(src_g, f"{WORK}/swap/gt/{c}.nii.gz", True)
    axial_reslice(src_i, f"{WORK}/axial/img/{c}_0000.nii.gz", False)
    axial_reslice(src_g, f"{WORK}/axial/gt/{c}.nii.gz", True)
    if i == 0:
        for v in ("orig", "swap", "axial"):
            im = nib.load(f"{WORK}/{v}/img/{c}_0000.nii.gz")
            print(f"[geom]   {c} {v:5s}: shape={im.shape} zooms="
                  f"{tuple(round(float(z),3) for z in im.header.get_zooms())}", flush=True)
print("[geom] variants built", flush=True)
PYEOF

# ── Phase 2: predict, 4 methods x 3 variants, fold0, bounded 4-way pool ─────
echo "[geom] ==== PHASE 2: predict 4 methods x 3 variants (fold0), 4-way ===="
# method|trainer|category|run_id
METHODS=(
  "baseline|nnUNetTrainerCHAOSBaseline|nnUNet|chaos_t1in_baseline_20260614_153230"
  "synthseg_EM|nnUNetTrainerCHAOSAugLabDefault|auglab|chaos_t1in_synthseg_EM_train100_val000_20260611_120000"
  "srcsm|nnUNetTrainerCHAOSAugLabDefault|auglab|chaos_t1in_srcsm_20260710_011817"
  "OURS_val100|nnUNetTrainerCHAOSAugLabValSynth|auglab|chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"
)
JOBS=()
for m in "${METHODS[@]}"; do
  for v in orig swap axial; do JOBS+=("${m}|${v}"); done
done
N=${#JOBS[@]}
echo "[geom] ${N} predict tasks"

worker() {
  local slot="$1" idx="$1" gpu=$(( $1 % 4 ))
  while [ "${idx}" -lt "${N}" ]; do
    IFS='|' read -r method trainer category run_id variant <<< "${JOBS[$idx]}"
    local out="${WORK}/pred/${variant}/${method}"
    mkdir -p "${out}"
    local t0=$(date +%s)
    echo "[geom] slot${slot}(GPU${gpu}) start ${method}/${variant}"
    nnUNet_results="${CHAOS_PRED}/${category}/${run_id}" \
    CUDA_VISIBLE_DEVICES=${gpu} \
    "${PROJECT_ROOT}/.venv/bin/nnUNetv2_predict" \
        -i "${WORK}/${variant}/img" -o "${out}" \
        -d 60 -c 3d_fullres -tr "${trainer}" -f 0 --disable_tta \
        -chk checkpoint_best.pth -npp 3 -nps 3 \
        > "${WORK}/predict_${method}_${variant}.log" 2>&1
    echo "[geom] slot${slot}(GPU${gpu}) done ${method}/${variant} rc=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + 4 ))
  done
}
pids=(); for s in 0 1 2 3; do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done

# ── Phase 3: verify counts, then evaluate every (method, variant) ───────────
echo "[geom] ==== PHASE 3: verify prediction counts (expect 100 each) ===="
NCASES=$(ls "${WORK}/orig/img" | wc -l)
bad=0
for d in "${WORK}"/pred/*/*; do
  n=$(find "$d" -name '*.nii.gz' | wc -l)
  [ "$n" -eq "$NCASES" ] || { echo "[geom] BAD COUNT ($n/${NCASES}): $d"; bad=1; }
done
echo "[geom] count verification done (bad=${bad})"

echo "[geom] ==== PHASE 4: evaluate (kidney = chaos {2,3} union vs GT 1) ===="
"${PROJECT_ROOT}/.venv/bin/python" - "$WORK" <<'PYEOF'
import sys, os, glob, csv
import numpy as np
import nibabel as nib

WORK = sys.argv[1]
VARIANTS = ("orig", "swap", "axial")
METHODS = ("baseline", "synthseg_EM", "srcsm", "OURS_val100")

rows, table = [], {}
for v in VARIANTS:
    for m in METHODS:
        pdir = f"{WORK}/pred/{v}/{m}"
        if not os.path.isdir(pdir):
            continue
        dices = []
        for p in sorted(glob.glob(f"{pdir}/*.nii.gz")):
            c = os.path.basename(p)[:-len(".nii.gz")]
            gtp = f"{WORK}/{v}/gt/{c}.nii.gz"
            if not os.path.exists(gtp):
                continue
            pred = np.isin(nib.load(p).get_fdata(), [2, 3])
            gt = nib.load(gtp).get_fdata() == 1
            s = pred.sum() + gt.sum()
            d = 2.0 * (pred & gt).sum() / s if s else float("nan")
            dices.append(d)
            rows.append({"variant": v, "method": m, "case": c, "dice": f"{d:.6f}"})
        a = np.array(dices, float)
        table[(v, m)] = (np.nanmean(a), np.nanmedian(a), len(a),
                         int((a < 0.1).sum()))

with open(f"{WORK}/geom_norm_per_case.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["variant", "method", "case", "dice"]); w.writeheader()
    w.writerows(rows)

print("\n[geom] ===== MEAN DICE (fold0, all cases) =====")
hdr = f"{'method':<14}" + "".join(f"{v:>22}" for v in VARIANTS)
print(hdr); print("-" * len(hdr))
for m in METHODS:
    line = f"{m:<14}"
    for v in VARIANTS:
        if (v, m) in table:
            mu, med, n, nf = table[(v, m)]
            line += f"{mu*100:>9.1f} (med{med*100:5.1f})"
        else:
            line += f"{'--':>22}"
    print(line)
print("\n[geom] ===== n_cases with Dice < 0.10 (catastrophic failures) =====")
for m in METHODS:
    print(f"{m:<14}" + "".join(
        f"{table[(v,m)][3]:>22}" if (v, m) in table else f"{'--':>22}"
        for v in VARIANTS))
print(f"\n[geom] per-case CSV -> {WORK}/geom_norm_per_case.csv")

print("\n[geom] ===== METHOD RANKING per variant (by mean dice) =====")
for v in VARIANTS:
    got = [(m, table[(v, m)][0]) for m in METHODS if (v, m) in table]
    got.sort(key=lambda t: -t[1])
    print(f"  {v:6s}: " + "  >  ".join(f"{m}({mu*100:.1f})" for m, mu in got))
PYEOF
echo "[geom] DONE"
