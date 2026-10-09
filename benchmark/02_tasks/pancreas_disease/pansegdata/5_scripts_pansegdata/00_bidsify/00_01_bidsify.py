"""PanSegData raw zips (0_raw_pansegdata/{t1,t2}.zip, OSF kysnj) -> lab BIDS leaf 1_BIDS_pansegdata/onc-pancreas-pansegdata.

Source layout (nnU-Net style, one zip per sequence): t1/imagesTr/<SITE>_<NNNN>_0000.nii.gz + t1/labelsTr/<SITE>_<NNNN>.nii.gz (same for t2).
The anonymised ids of T1 and T2 are INDEPENDENT numberings, so the same patient has different ids in the two zips. T1<->T2 pairing uses the
original case names in Meta informations/T{1,2}-name_mapping.json (normalised: lowercase site letters + integer), and only within the same
anonymised site. Pairing evidence (audit 2026-10-04): non-MCF pairs clearly the same patient (matched vs null centroid 8-15 vs 28-38 mm);
MCF pairs match on extent/volume but not centroid (FOV centring) -> treated as pairs. Unpaired scans keep their own subject.

One subject = one BIDS sub- id shared by both contrasts:  sub-<site><NNNN>  (per-site counter: paired by T1 id, then T1-only, then T2-only).
  anat/sub-X_acq-venous_T1w.nii.gz   venous-phase contrast-enhanced T1 (paper: "all T1W images were acquired in the venous phase")
  anat/sub-X_T2w.nii.gz
  derivatives/labels/sub-X/anat/sub-X_acq-venous_label-pancreas_seg.nii.gz   (drawn on T1)
  derivatives/labels/sub-X/anat/sub-X_label-pancreas_seg.nii.gz              (drawn on T2)
Images are byte-copied from the zips (faithful: mixed orientations LPI/LPS/RAS are kept; they are reoriented at nnU-Net conversion).
Labels are re-written as uint8 {0,1}; where the label header disagrees with its image (66 MCF T1 cases: RAI/RAS vs LPS) the IMAGE affine is
written instead - the voxel arrays were verified aligned (gallery overlays + intensity contrast) - and the case is flagged in participants.tsv.
Re-runnable: existing outputs are kept (tmp + os.replace)."""
import csv, gzip, json, os, re, shutil, zipfile, collections
from pathlib import Path
import numpy as np, nibabel as nib

DS = Path(__file__).resolve().parents[2]
RAW = DS / "0_raw_pansegdata"
LEAF = "onc-pancreas-pansegdata"
OUT = DS / "1_BIDS_pansegdata" / LEAF

LICENSE_SRC = RAW / "LICENSE.txt"


def norm(name):
    s = name.replace(".nii.gz", "").lower()
    m = re.match(r"^([a-z]+)[_-]?0*(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (s,)


def inverse_map(seq):
    d = json.load(open(RAW / f"{seq.upper()}-name_mapping.json"))
    return {v.replace("_0000.nii.gz", ""): k for k, v in d.items() if v != "Not included"}


def atomic_write(dst, data):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("tmp_" + dst.name)
    tmp.write_bytes(data) if isinstance(data, bytes) else data(tmp)
    os.replace(tmp, dst)


def load_gz(zf, member):
    return nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(member)))


zf = {s: zipfile.ZipFile(RAW / f"{s}.zip") for s in ("t1", "t2")}
ids = {s: sorted(Path(n).name.replace("_0000.nii.gz", "") for n in zf[s].namelist() if n.endswith("_0000.nii.gz")) for s in zf}
orig = {s: inverse_map(s) for s in zf}
assert len(ids["t1"]) == 385 and len(ids["t2"]) == 382, {s: len(v) for s, v in ids.items()}
site = lambda cid: cid.split("_")[0]

# --- pair by normalised original name, same anonymised site only
key2t2 = collections.defaultdict(list)
for c in ids["t2"]:
    key2t2[(site(c),) + norm(orig["t2"][c])].append(c)
pairs, used = {}, set()
for c in ids["t1"]:
    k = (site(c),) + norm(orig["t1"][c])
    cand = [x for x in key2t2.get(k, []) if x not in used]
    if len(cand) == 1:
        pairs[c] = cand[0]; used.add(cand[0])
    else:
        assert len(cand) == 0, f"ambiguous pairing for {c}: {cand}"
t2_only = [c for c in ids["t2"] if c not in used]
print(f"pairs {len(pairs)}  t1-only {len(ids['t1']) - len(pairs)}  t2-only {len(t2_only)}", flush=True)

subjects = []  # (sub, t1_id, t2_id)
counter = collections.Counter()
def newsub(st):
    counter[st] += 1
    return f"sub-{st.lower()}{counter[st]:04d}"
for c in ids["t1"]:
    if c in pairs: subjects.append((newsub(site(c)), c, pairs[c]))
for c in ids["t1"]:
    if c not in pairs: subjects.append((newsub(site(c)), c, None))
for c in t2_only:
    subjects.append((newsub(site(c)), None, c))
subjects.sort(key=lambda r: r[0])

rows = []
for sub, t1, t2 in subjects:
    fixed = []
    for seq, cid, bids_stem in (("t1", t1, f"{sub}_acq-venous"), ("t2", t2, sub)):
        if cid is None:
            continue
        suffix = "T1w" if seq == "t1" else "T2w"
        img_dst = OUT / sub / "anat" / f"{bids_stem}_{suffix}.nii.gz"
        lab_dst = OUT / "derivatives/labels" / sub / "anat" / f"{bids_stem}_label-pancreas_seg.nii.gz"
        img_member, lab_member = f"{seq}/imagesTr/{cid}_0000.nii.gz", f"{seq}/labelsTr/{cid}.nii.gz"
        if not img_dst.exists():
            atomic_write(img_dst, zf[seq].read(img_member))
        if not lab_dst.exists():
            img, lab = load_gz(zf[seq], img_member), load_gz(zf[seq], lab_member)
            assert img.shape == lab.shape, f"{seq} {cid}: label shape {lab.shape} != image {img.shape}"
            d = np.asanyarray(lab.dataobj)
            u = np.unique(d)
            assert set(np.round(u).astype(int)) <= {0, 1} and np.all(np.minimum(np.abs(u), np.abs(u - 1)) < 1e-3), f"{seq} {cid}: label values {u[:6]}"
            assert (d > 0.5).sum() > 0, f"{seq} {cid}: empty label"
            hdr_fixed = not np.allclose(img.affine, lab.affine, atol=1e-3)
            o = nib.Nifti1Image((d > 0.5).astype(np.uint8), img.affine)
            o.set_qform(img.affine, 1); o.set_sform(img.affine, 1)
            atomic_write(lab_dst, lambda p, o=o: nib.save(o, str(p)))
            side = lab_dst.with_name(lab_dst.name.replace(".nii.gz", ".json"))
            desc = ("Pancreas mask drawn manually in ITK-SNAP by one radiologist per center (senior radiologist QC); binary 0/1. "
                    "Drawn on the " + ("venous-phase T1W" if seq == "t1" else "T2W") + " scan. The source description does not say whether cystic/neoplastic lesions "
                    "are inside the mask (cysts appear inside it on inspection).")
            if hdr_fixed:
                desc += " The source label header disagreed with its image header; the image affine was written (arrays verified aligned)."
            side.write_text(json.dumps({"Description": desc, "SpatialReference": f"{sub}/anat/{bids_stem}_{suffix}.nii.gz",
                                        "Sources": [f"bids::{sub}/anat/{bids_stem}_{suffix}.nii.gz"],
                                        "SourceCaseId": f"{seq}/{cid}"}, indent=2))
        else:
            lab = nib.load(str(lab_dst)); img = nib.load(str(img_dst))
            hdr_fixed = "image affine was written" in lab_dst.with_name(lab_dst.name.replace(".nii.gz", ".json")).read_text()
        if hdr_fixed:
            fixed.append(seq)
        side_i = img_dst.with_name(img_dst.name.replace(".nii.gz", ".json"))
        if not side_i.exists():
            side_i.write_text(json.dumps({"Description": "Venous-phase contrast-enhanced T1-weighted MRI" if seq == "t1" else "T2-weighted MRI",
                                          "SourceCaseId": f"{seq}/{cid}", "OriginalCaseName": orig[seq][cid]}, indent=2))
    rows.append(dict(participant_id=sub, site=(t1 or t2).split("_")[0], source_t1_id=t1 or "n/a", source_t2_id=t2 or "n/a",
                     original_name_t1=orig["t1"][t1] if t1 else "n/a", original_name_t2=orig["t2"][t2] if t2 else "n/a",
                     pairing="paired" if (t1 and t2) else ("t1only" if t1 else "t2only"),
                     label_header_fixed=",".join(fixed) if fixed else "no"))

with open(OUT / "participants.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
(OUT / "participants.json").write_text(json.dumps({
    "participant_id": {"Description": "BIDS subject id assigned by this project: per-site counter. The source anonymised ids of T1 and T2 are independent numberings."},
    "site": {"Description": "Contributing center (anonymised source prefix): NYU (New York University), MCF and MCA (the two Mayo Clinic sites), NWU (Northwestern University), AHN (Allegheny Health Network)"},
    "source_t1_id": {"Description": "Case id in the source t1.zip (n/a when the subject has no T1)"},
    "source_t2_id": {"Description": "Case id in the source t2.zip (n/a when the subject has no T2)"},
    "original_name_t1": {"Description": "Original (pre-anonymisation) case name from T1-name_mapping.json"},
    "original_name_t2": {"Description": "Original (pre-anonymisation) case name from T2-name_mapping.json"},
    "pairing": {"Description": "paired = same original case name in both mappings within the same site (T1 and T2 of one patient, supported by mask extent/volume agreement); t1only / t2only = no counterpart found"},
    "label_header_fixed": {"Description": "Sequences whose source label header disagreed with the image header; the image affine was written for the label (arrays aligned)"}}, indent=2))
(OUT / "dataset_description.json").write_text(json.dumps({
    "Name": "PanSegData -- multi-center pancreas MRI (venous-phase T1W and T2W) with manual pancreas segmentations",
    "BIDSVersion": "1.9.0", "DatasetType": "raw", "License": "CC BY-NC 4.0",
    "Authors": ["Zhang, Z.", "Keles, E.", "Aktas, H. E.", "Bagci, U.", "and the PanSegData investigators"],
    "HowToAcknowledge": "Zhang, Z., Keles, E., Durak, G., Taktak, Y., Susladkar, O., Gorade, V., Jha, D., Ormeci, A. C., Medetalibeyoglu, A., Yao, L., Wang, B., Isler, I. S., Peng, L., Pan, H., Vendrami, C. L., Bourhani, A., Velichko, Y., Gong, B., Spampinato, C., Pyrros, A., ... Bagci, U. (2025). Large-scale multi-center CT and MRI segmentation of pancreas with deep learning. Medical Image Analysis, 99, 103382. https://doi.org/10.1016/j.media.2024.103382",
    "ReferencesAndLinks": ["https://osf.io/kysnj/", "https://doi.org/10.1016/j.media.2024.103382", "https://arxiv.org/abs/2405.12367"]}, indent=2))
(OUT / "derivatives/labels").mkdir(parents=True, exist_ok=True)
(OUT / "derivatives/labels/dataset_description.json").write_text(json.dumps({
    "Name": "PanSegData pancreas segmentations", "BIDSVersion": "1.9.0", "DatasetType": "derivative", "License": "CC BY-NC 4.0",
    "GeneratedBy": [{"Name": "pansegdata 00_01_bidsify.py", "Description": "Re-wrapped the source labelsTr masks as uint8 {0,1} with BIDS names; image affine written where the source label header disagreed"}],
    "SourceDatasets": [{"URL": "https://osf.io/kysnj/"}]}, indent=2))
shutil.copyfile(LICENSE_SRC, OUT / "LICENSE")
(OUT / "code").mkdir(exist_ok=True)
shutil.copyfile(Path(__file__).resolve(), OUT / "code" / "00_01_bidsify.py")
print("subjects", len(rows), "paired", sum(r["pairing"] == "paired" for r in rows), "header-fixed", sum(r["label_header_fixed"] != "no" for r in rows))
