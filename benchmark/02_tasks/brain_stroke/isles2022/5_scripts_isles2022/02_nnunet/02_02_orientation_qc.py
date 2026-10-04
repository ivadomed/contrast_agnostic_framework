"""Signed orientation/alignment QC: per case, axial + sagittal slice through the lesion centroid for the
nnU-Net-space DWI / ADC / FLAIR (FLAIR resampled onto the DWI grid) with the mask contour. aff2axcodes is blind
to 180-degree flips, so this is looked at by eye: anterior should be the same side in every case, FLAIR must
overlay DWI anatomy, lesions bright on DWI/FLAIR and dark on ADC. Output: 9_tests_isles2022/orientation_qc.png"""
import json
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = Path(__file__).resolve().parents[2]
RAW = DS / "2_nnUNet_isles2022" / "raw" / "Dataset140_ISLES2022_DWI"
man = json.loads((DS / "2_nnUNet_isles2022" / "isles2022_convert_manifest.json").read_text())
# pick spread of spacings/sizes: first case of each distinct spacing + a few others
seen, pick = set(), []
for q in sorted(man, key=lambda q: q["case"]):
    k = tuple(q["spacing"])
    if k not in seen: seen.add(k); pick.append(q["case"])
pick = (pick + [q["case"] for q in man if q["case"] not in pick][::40])[:6]
fig, ax = plt.subplots(len(pick) * 2, 3, figsize=(9, 6 * len(pick)))
for r, cid in enumerate(pick):
    q = next(x for x in man if x["case"] == cid)
    split = q["split"]
    d = RAW / ("imagesTr" if split == "train" else "imagesTs_dwi")
    lab = nib.load(str(RAW / ("labelsTr" if split == "train" else "labelsTs_dwi") / f"{cid}.nii.gz"))
    m = np.asanyarray(lab.dataobj) > 0
    vols = {"DWI": np.asanyarray(nib.load(str(d / f"{cid}_0000.nii.gz")).dataobj)}
    if split == "test":
        for k in ("adc", "flair"):
            vols[k.upper()] = np.asanyarray(nib.load(str(RAW / f"imagesTs_{k}" / f"{cid}_0000.nii.gz")).dataobj)
    else:
        f = RAW.parent / "Dataset141_ISLES2022_FLAIR" / "imagesTr" / f"{cid}_0000.nii.gz"
        vols["FLAIR"] = np.asanyarray(nib.load(str(f)).dataobj)
        sub = "sub-strokecase" + cid.split("_")[1]
        adc = DS / "1_BIDS_isles2022" / "stroke-brain-isles2022" / "derivatives" / "adc" / sub / "ses-0001" / "dwi" / f"{sub}_ses-0001_desc-adc_dwi.nii.gz"
        vols["ADC"] = np.asanyarray(nib.load(str(adc)).dataobj)
    c = np.round(np.argwhere(m).mean(0)).astype(int)
    for j, k in enumerate(("DWI", "ADC", "FLAIR")):
        v = vols[k]
        for i, (sl, ms, name) in enumerate(((v[:, :, c[2]], m[:, :, c[2]], "axial"), (v[:, c[1], :], m[:, c[1], :], "sag/cor"))):
            a = ax[r * 2 + i, j]; nz = v[v != 0]; lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99.5)) if nz.size else (0, 1); a.imshow(np.rot90(sl), cmap="gray", vmin=lo, vmax=hi); a.contour(np.rot90(ms), [0.5], colors="r", linewidths=0.6)
            a.set_title(f"{cid} {k} {name}", fontsize=7); a.axis("off")
fig.tight_layout(); out = DS / "9_tests_isles2022" / "orientation_qc.png"; fig.savefig(out, dpi=70); print("wrote", out, pick)
