#!/usr/bin/env python3
"""
Download the BASELINE (T0, pre-treatment) DWI arm of TCIA's ACRIN-6698
collection ("ACRIN 6698/I-SPY2 Breast DWI", CC BY 4.0, public NBIA REST API --
no login) for eval-only cross-contrast testing of I-SPY2-trained models.

Why ACRIN-6698: it is the DWI sub-study of the I-SPY2 trial. TCIA ships its 385
patients as a collection DISJOINT from the "ISPY2" collection our training set
came from (TCIA's own ISPY2 page: 719 ISPY2 + 266 ACRIN-6698 = "I-SPY2 Imaging
Cohort 1"); verified directly 2026-09-30: 0/385 ACRIN-6698 patient numbers
appear among ispy2's 560 BIDS subjects. Same trial => same pathology/eligibility
(biopsy-proven invasive cancer, stage II/III, >=2.5 cm, neoadjuvant) as the
training set; restricting to T0 keeps it PRE-treatment, exactly like training
(T1-T3 are on/post-chemo residual disease -- a different "stage", excluded).

Per patient, from the single study whose StudyDesc ends in "_T0":
  - "DWI TRACE: from S<N>: bVals=..."   (all b-values stacked in one series)
  - "ADC: from S<N>: ..."                (used for the mask physics QC only)
  - "DWI MASK: from S<N>: Whole Tumor Manual"  (MR-format multi-slice mask,
                                          same S<N> as the TRACE it was drawn on)
  - "ISPY2: VOLSER: uni-lateral cropped: Analysis Mask" (SEG, the I-SPY2 FTV
                                          mask -- used only to cross-check tumour
                                          side/location of the DWI mask)
When a T0 study carries test-retest pairs ("TrT0:"/"TrT1:" prefixes), the TrT0
(first) acquisition is taken -- never both (would double-count the patient).

Phase 1 (listing) caches every patient's getSeries JSON under <STAGE>/series/.
Phase 2 downloads the chosen series as NBIA zips and unpacks them under
<STAGE>/raw/<pid>/<role>/. Resumable, rate-limited, per-request timeout.

Run on the VULCAN LOGIN NODE (compute nodes have no internet), in background,
as 3 shards (~1 min/patient/process), then merge:
  for i in 0 1 2; do nohup .venv/bin/python .../download_acrin6698.py --shard-index $i --shard-count 3 \
    > $SCRATCH/acrin6698_download/download_shard$i.log 2>&1 & done
  .venv/bin/python .../download_acrin6698.py --merge
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import zipfile
import os
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
STAGE = Path(os.environ["SCRATCH"]) / "acrin6698_download"
SERIES_DIR = STAGE / "series"
RAW = STAGE / "raw"
SELECTION = STAGE / "t0_selection.json"   # merged view; shards write t0_selection_shard<i>.json
TIMEOUT_S = 300
PAUSE_S = 0.3


def http_get(url: str, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"    [retry {attempt}/{retries}] {url[:120]}: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"failed: {url}")


def list_series(pid: str) -> list[dict]:
    f = SERIES_DIR / f"{pid}.json"
    if f.exists() and f.stat().st_size > 2:
        return json.load(open(f))
    data = json.loads(http_get(f"{API}/getSeries?Collection=ACRIN-6698&PatientID={pid}&format=json"))
    f.write_text(json.dumps(data))
    time.sleep(PAUSE_S)
    return data


def pick_t0(series: list[dict]) -> dict | None:
    t0 = [s for s in series if str(s.get("StudyDesc", "")).endswith("_T0")]
    if not t0:
        return None
    def find(pattern: str):
        hits = [s for s in t0 if re.search(pattern, s.get("SeriesDescription", ""))]
        # prefer the non-retest or TrT0 acquisition over TrT1
        hits.sort(key=lambda s: (1 if "TrT1" in s["SeriesDescription"] else 0, s.get("SeriesNumber", 0)))
        return hits
    masks = [s for s in find(r"DWI MASK: from S\d+") if s["Modality"] == "MR"]
    if not masks:
        return {"status": "no_t0_dwi_mask", "study": t0[0]["StudyInstanceUID"]}
    mask = masks[0]
    src = re.search(r"from (S\d+)", mask["SeriesDescription"]).group(1)
    trace = [s for s in find(rf"DWI TRACE: from {src}:")]
    adc = [s for s in find(rf"ADC: from {src}:")]
    volser = [s for s in t0 if s["Modality"] == "SEG" and "Analysis Mask" in s.get("SeriesDescription", "")]
    if not trace:
        return {"status": "no_matching_trace", "study": mask["StudyInstanceUID"], "src": src}
    sel = {"status": "ok", "study": mask["StudyInstanceUID"], "src_series": src,
           "retest_tag": "TrT0" if "TrT0" in mask["SeriesDescription"] else "",
           "n_t0_dwi_masks": len(masks),
           "roles": {"dwi_trace": trace[0], "dwi_mask": mask}}
    if adc:
        sel["roles"]["adc"] = adc[0]
    if volser:
        sel["roles"]["volser_mask"] = volser[0]
    return sel


def download_series(uid: str, dest: Path) -> str:
    done = dest / ".done"
    if done.exists():
        return "cached"
    dest.mkdir(parents=True, exist_ok=True)
    blob = http_get(f"{API}/getImage?SeriesInstanceUID={uid}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest)
    done.write_text(uid)
    time.sleep(PAUSE_S)
    return "ok"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="only merge shard selections into t0_selection.json")
    a = ap.parse_args()
    if a.merge:
        merged = {}
        for f in sorted(STAGE.glob("t0_selection_shard*.json")):
            merged.update(json.load(open(f)))
        SELECTION.write_text(json.dumps(merged, indent=1))
        print(f"merged {len(merged)} patients -> {SELECTION}")
        return
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    pids = sorted(p["PatientId"] for p in json.load(open(STAGE / "patients.json")))
    pids = [p for i, p in enumerate(pids) if i % a.shard_count == a.shard_index]
    shard_sel = STAGE / f"t0_selection_shard{a.shard_index}.json"
    selection = json.load(open(shard_sel)) if shard_sel.exists() else {}
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        try:
            sel = pick_t0(list_series(pid))
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(pids)}] {pid}: listing FAILED {e}", flush=True)
            continue
        if sel is None:
            selection[pid] = {"status": "no_t0_study"}
        elif sel["status"] != "ok":
            selection[pid] = sel
        else:
            status = {}
            for role, s in sel["roles"].items():
                try:
                    status[role] = download_series(s["SeriesInstanceUID"], RAW / pid / role)
                except Exception as e:  # noqa: BLE001
                    status[role] = f"failed: {e}"
            selection[pid] = {**{k: v for k, v in sel.items() if k != "roles"},
                              "series": {r: {"uid": s["SeriesInstanceUID"],
                                             "desc": s["SeriesDescription"],
                                             "manufacturer": s.get("Manufacturer"),
                                             "model": s.get("ManufacturerModelName")}
                                         for r, s in sel["roles"].items()},
                              "download": status}
        shard_sel.write_text(json.dumps(selection, indent=1))
        print(f"[{i}/{len(pids)}] {pid}: {selection[pid].get('status')} "
              f"{selection[pid].get('download', '')} ({time.time() - t0:.0f}s)", flush=True)
    n_ok = sum(1 for v in selection.values() if v.get("status") == "ok"
               and all(str(x) in ("ok", "cached") for x in v["download"].values()))
    print(f"DONE. {n_ok}/{len(pids)} patients with a complete T0 DWI arm.", flush=True)


if __name__ == "__main__":
    main()
