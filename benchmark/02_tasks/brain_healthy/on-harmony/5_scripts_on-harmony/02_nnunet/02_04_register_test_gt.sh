#!/usr/bin/env bash
# Register T1w SynthSeg labels to non-T1w test contrasts via ANTs SyN (antspyx).
#
# Launches 4 CPU ranks in parallel — each rank processes its share of the
# contrast × subject registration jobs, 8 concurrent ANTs workers per rank.
#   4 ranks × 8 jobs each = 32 concurrent ANTs registrations
#
# Output:
#   data/ON-Harmony/derivatives/synthseg_registered/{contrast}/{sub}_{ses}_{contrast}_seg7.nii.gz
#   data/splits/test_registration_qc.csv
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

PY=".venv/bin/python"
TEST_CASES="data/splits/test_cases.json"
JOBS_PER_RANK=8    # parallel ANTs jobs per rank (4 ranks × 8 = 32 concurrent)

[ -f "$TEST_CASES" ] || { echo "ERROR: $TEST_CASES not found"; exit 1; }
echo "[$(date '+%H:%M:%S')] Starting GT registration across all 4 slots ($((JOBS_PER_RANK * 4)) concurrent ANTs jobs)"

# Write the worker script once — PROJECT_ROOT passed as --project-root to avoid
# hardcoding the path inside the heredoc.
cat > /tmp/register_gt_worker.py << 'PYEOF'
import sys, os, json, argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import ants
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

p = argparse.ArgumentParser()
p.add_argument("--rank",          type=int, default=0)
p.add_argument("--world-size",    type=int, default=4)
p.add_argument("--jobs",          type=int, default=8)
p.add_argument("--project-root",  required=True)
args = p.parse_args()

PROJECT_ROOT = Path(args.project_root)
BIDS         = PROJECT_ROOT / "data" / "ON-Harmony"
OUT_BASE     = BIDS / "derivatives" / "synthseg_registered"
TEST_CASES   = PROJECT_ROOT / "data" / "splits" / "test_cases.json"
QC_CSV       = PROJECT_ROOT / "data" / "splits" / "test_registration_qc.csv"

_MAX = 60
_LUT = np.zeros(_MAX + 2, dtype=np.uint8)
for fs_id, cls in {0:0,2:2,3:1,4:3,5:3,7:6,8:6,10:4,11:4,12:4,13:4,14:3,15:3,
                   16:5,17:4,18:4,26:4,28:4,41:2,42:1,43:3,44:3,46:6,47:6,
                   49:4,50:4,51:4,52:4,53:4,54:4,58:4,60:4}.items():
    _LUT[min(fs_id, _MAX + 1)] = cls
def remap(arr): return _LUT[np.clip(arr, 0, _MAX + 1).astype(np.int32)].astype(np.uint8)

def extract_3d(path, label):
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    if arr.ndim == 3: return img
    vol = arr.mean(axis=-1) if label == "bold" else arr[..., 0]
    return nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header)

def find_gre(sub, ses):
    swi = BIDS / sub / ses / "swi"
    if not swi.exists(): return None
    cands = sorted(swi.glob("*echo-1*part-mag*GRE.nii.gz"))
    return cands[0] if cands else None

def nmi_score(f_path, m_path):
    try:
        f = sitk.ReadImage(str(f_path), sitk.sitkFloat32)
        m = sitk.ReadImage(str(m_path), sitk.sitkFloat32)
        mr = sitk.Resample(m, f, sitk.Transform(), sitk.sitkLinear, 0.0, m.GetPixelID())
        af = sitk.GetArrayFromImage(f).ravel()
        am = sitk.GetArrayFromImage(mr).ravel()
        h, _, _ = np.histogram2d(af, am, bins=64)
        h /= h.sum() + 1e-12
        pf, pm = h.sum(1), h.sum(0); eps = 1e-10
        return float((-np.sum(pf*np.log(pf+eps)) - np.sum(pm*np.log(pm+eps))) /
                     (-np.sum(h*np.log(h+eps)) + eps))
    except: return float("nan")

CONTRASTS = [
    ("T2w",           lambda s, e: BIDS/s/e/"anat"/f"{s}_{e}_T2w.nii.gz"),
    ("bold",          lambda s, e: BIDS/s/e/"func"/f"{s}_{e}_task-rest_bold.nii.gz"),
    ("dwi_ap",        lambda s, e: BIDS/s/e/"dwi"/f"{s}_{e}_dir-AP_dwi.nii.gz"),
    ("epi_ap",        lambda s, e: BIDS/s/e/"fmap"/f"{s}_{e}_dir-AP_epi.nii.gz"),
    ("gre_echo1_mag", lambda s, e: find_gre(s, e)),
]

test_cases = json.loads(TEST_CASES.read_text())
all_jobs   = [(tc, lbl, fn) for tc in test_cases for lbl, fn in CONTRASTS]
my_jobs    = [j for i, j in enumerate(all_jobs) if i % args.world_size == args.rank]
print(f"[rank {args.rank}] {len(my_jobs)} registration jobs, {args.jobs} concurrent")

def process(tc, contrast_label, img_fn):
    sub, ses = tc["subject"], tc["session"]
    src = img_fn(sub, ses)
    if src is None or not Path(src).exists():
        return None

    out_dir = OUT_BASE / contrast_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_seg = out_dir / f"{sub}_{ses}_{contrast_label}_seg7.nii.gz"

    if not out_seg.exists():
        print(f"  [rank {args.rank}] Registering {sub} {ses} {contrast_label} ...", flush=True)
        import tempfile
        vol3d = extract_3d(src, contrast_label)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "vol3d.nii.gz"
            nib.save(vol3d, str(tmp_path))
            fixed  = ants.image_read(tc["t1w"],     pixeltype="float")
            moving = ants.image_read(str(tmp_path), pixeltype="float")
            reg = ants.registration(fixed=fixed, moving=moving,
                                    type_of_transform="SyN", verbose=False)
            mask_ants = ants.image_read(tc["mask"], pixeltype="unsigned int")
            reg_mask  = ants.apply_transforms(
                fixed=moving, moving=mask_ants,
                transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor")
            seg7 = remap(reg_mask.numpy().astype(np.int32))
            nib.save(nib.Nifti1Image(seg7, vol3d.affine, vol3d.header), str(out_seg))
    else:
        print(f"  [rank {args.rank}] EXISTS {sub} {ses} {contrast_label}", flush=True)

    nmi_val = nmi_score(tc["t1w"], src)
    flagged = bool(nmi_val < 0.30 or nmi_val != nmi_val)
    tag = "FLAGGED" if flagged else "OK"
    print(f"  [rank {args.rank}] {sub} {ses} {contrast_label}: NMI={nmi_val:.3f} {tag}", flush=True)
    return {"subject": sub, "session": ses, "scanner": tc["scanner"],
            "contrast": contrast_label, "nmi": nmi_val, "flagged": flagged,
            "registered_seg": str(out_seg)}

results = []
with ThreadPoolExecutor(max_workers=args.jobs) as pool:
    futs = {pool.submit(process, *j): j for j in my_jobs}
    for fut in as_completed(futs):
        try:
            r = fut.result()
            if r: results.append(r)
        except Exception as e:
            j = futs[fut]
            print(f"  [rank {args.rank}] ERROR {j[0]['subject']} {j[1]}: {e}", flush=True)

partial = QC_CSV.parent / f"test_registration_qc_rank{args.rank}.csv"
with open(partial, "w") as f:
    for r in sorted(results, key=lambda x: (x["subject"], x["session"], x["contrast"])):
        f.write(f"{r['subject']},{r['session']},{r['scanner']},"
                f"{r['contrast']},{r['nmi']:.4f},{r['flagged']},{r['registered_seg']}\n")
print(f"[rank {args.rank}] Done → {partial}")
PYEOF

# Launch 4 CPU ranks in parallel — each rank becomes its own blocking run_job call
# (background + wait on PIDs so we continue only when all ranks are done before merging).
PIDS=()
for R in 0 1 2 3; do
    run_job --name "reg_gt_rank${R}" --gpus 0 --slot "${R}" --wait \
        --log "/tmp/reg_gt_r${R}.log" -- \
        "$PY" /tmp/register_gt_worker.py \
            --rank "$R" --world-size 4 --jobs "$JOBS_PER_RANK" \
            --project-root "${PROJECT_ROOT}" &
    PIDS+=($!)
done

echo "[$(date '+%H:%M:%S')] All 4 ranks launched. Waiting..."
wait "${PIDS[@]}"

echo "[$(date '+%H:%M:%S')] All ranks done. Merging QC..."
echo "subject,session,scanner,contrast,nmi,flagged,registered_seg" \
    > data/splits/test_registration_qc.csv
for R in 0 1 2 3; do
    [ -f "data/splits/test_registration_qc_rank${R}.csv" ] && \
        cat "data/splits/test_registration_qc_rank${R}.csv" >> data/splits/test_registration_qc.csv
done

N_TOTAL=$(tail -n +2 data/splits/test_registration_qc.csv | wc -l)
N_FLAG=$(grep -c "True" data/splits/test_registration_qc.csv 2>/dev/null || echo 0)
echo "Done: ${N_TOTAL} registrations, ${N_FLAG} flagged. QC → data/splits/test_registration_qc.csv"
