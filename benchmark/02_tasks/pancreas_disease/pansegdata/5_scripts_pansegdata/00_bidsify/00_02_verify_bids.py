"""Verify the PanSegData BIDS leaf against the source zips: (1) every image is a byte-exact copy (zip CRC32 == CRC32 of the BIDS file),
(2) every label has the same shape and foreground voxel count as the source label and lives on its image's grid, (3) files present == what
participants.tsv claims (paired/t1only/t2only), sidecars exist, (4) no uncompressed .nii anywhere. Prints ALL CHECKS PASSED or the failures."""
import csv, gzip, sys, zipfile, zlib
from pathlib import Path
import numpy as np, nibabel as nib

DS = Path(__file__).resolve().parents[2]
OUT = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
zf = {s: zipfile.ZipFile(DS / "0_raw_pansegdata" / f"{s}.zip") for s in ("t1", "t2")}
rows = list(csv.DictReader(open(OUT / "participants.tsv"), delimiter="\t"))
fail = []
n_img = n_lab = 0
for r in rows:
    sub = r["participant_id"]
    for seq, cid, stem, suf in (("t1", r["source_t1_id"], f"{sub}_acq-venous", "T1w"), ("t2", r["source_t2_id"], sub, "T2w")):
        img = OUT / sub / "anat" / f"{stem}_{suf}.nii.gz"
        lab = OUT / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz"
        if cid == "n/a":
            for p in (img, lab):
                if p.exists(): fail.append(f"{sub}: unexpected {p.name}")
            continue
        for p in (img, img.with_suffix("").with_suffix(".json"), lab, lab.with_suffix("").with_suffix(".json")):
            if not p.exists(): fail.append(f"{sub}: missing {p.name}")
        if not (img.exists() and lab.exists()):
            continue
        if zlib.crc32(img.read_bytes()) != zf[seq].getinfo(f"{seq}/imagesTr/{cid}_0000.nii.gz").CRC:
            fail.append(f"{sub} {seq}: image CRC differs from source")
        n_img += 1
        src = np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(zf[seq].read(f"{seq}/labelsTr/{cid}.nii.gz"))).dataobj)
        L = nib.load(str(lab)); I = nib.load(str(img))
        d = np.asanyarray(L.dataobj)
        if L.shape != I.shape or not np.allclose(L.affine, I.affine, atol=1e-3): fail.append(f"{sub} {seq}: label not on image grid")
        if d.dtype != np.uint8 or set(np.unique(d)) != {0, 1}: fail.append(f"{sub} {seq}: label dtype/values {d.dtype} {np.unique(d)}")
        if src.shape != d.shape or int((src > 0.5).sum()) != int(d.sum()): fail.append(f"{sub} {seq}: label voxels {int(d.sum())} != source {int((src > 0.5).sum())}")
        n_lab += 1
bare = [p for p in OUT.rglob("*.nii")]
if bare: fail.append(f"{len(bare)} uncompressed .nii files")
ndirs = len([p for p in OUT.glob("sub-*") if p.is_dir()]); ndd = len([p for p in (OUT / "derivatives/labels").glob("sub-*") if p.is_dir()])
if not (ndirs == ndd == len(rows)): fail.append(f"subject dirs {ndirs} / label dirs {ndd} / participants rows {len(rows)}")
print(f"subjects {len(rows)}  images checked {n_img}  labels checked {n_lab}")
print("ALL CHECKS PASSED" if not fail else f"FAILED ({len(fail)}):\n" + "\n".join(fail[:30]))
sys.exit(1 if fail else 0)
