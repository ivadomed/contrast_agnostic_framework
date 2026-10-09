#!/usr/bin/env python3
"""
Download the I-SPY1 (and NACT-Pilot) subsets of MAMA-MIA (Synapse syn60868042):
for every ISPY1_* / NACT_* case with an expert tumour mask, pull the
pre-contrast (_0000) and first post-contrast (_0001) DCE channels + the expert
mask. Same source, same expert-annotation protocol, same channel convention as
benchmark/02_tasks/breast_cancer/duke-breast-mri/0_raw_duke-breast-mri/download_duke.py
(which this mirrors) -- MAMA-MIA's 4 cohorts were segmented under one protocol,
which is what makes I-SPY1's labels directly compatible with Duke's and I-SPY2's.

Case lists (folder + mask Synapse ids) come from the cached MAMA-MIA listing
already staged for the Duke download (/scratch/paulh/duke_download/
{images_folders_all,expert_seg_entities}.json), filtered to ISPY1_/NACT_ and
written to <STAGE>/ispy1_nact_{image_folders,masks}.json.

Run on the VULCAN LOGIN NODE (compute nodes have no internet). Auth comes from
the existing ~/.synapseConfig (synapseclient default) -- no token on the
command line or in any file here. Resumable (skips non-empty files already on
disk), per-call alarm timeout, manifest written after every case:
  nohup .venv/bin/python benchmark/02_tasks/breast_cancer/ispy1/0_raw_ispy1/download_mamamia_ispy1_nact.py \
    > /scratch/paulh/ispy1_download/download.log 2>&1 &
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import synapseclient

STAGE = Path("/scratch/paulh/ispy1_download")
RAW = STAGE / "raw"
MANIFEST_PATH = STAGE / "ispy1_nact_manifest.json"
CHANNELS = ("_0000.nii.gz", "_0001.nii.gz")   # pre-contrast, first post-contrast
PER_CALL_TIMEOUT_S = 180
MAX_RETRIES = 3


class TimeoutError_(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError_("per-file download timed out")


def download_one(syn, synid, dest_path: Path) -> str:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return "cached"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(PER_CALL_TIMEOUT_S)
        try:
            got = Path(syn.get(synid, downloadLocation=str(dest_path.parent),
                               ifcollision="overwrite.local").path)
            if got.name != dest_path.name:
                got.rename(dest_path)
            return "ok" if dest_path.stat().st_size > 0 else "empty_file"
        except TimeoutError_:
            print(f"    [timeout] {synid} -> {dest_path.name} ({attempt}/{MAX_RETRIES})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    [error] {synid} -> {dest_path.name}: {e} ({attempt}/{MAX_RETRIES})", flush=True)
            time.sleep(5 * attempt)
        finally:
            signal.alarm(0)
    return "failed"


def main():
    folders = json.load(open(STAGE / "ispy1_nact_image_folders.json"))   # name -> folder synid
    masks = json.load(open(STAGE / "ispy1_nact_masks.json"))             # name -> file synid

    syn = synapseclient.Synapse()
    syn.login(silent=True)

    manifest = json.load(open(MANIFEST_PATH)) if MANIFEST_PATH.exists() else {}
    pids = sorted(folders)
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        entry = manifest.get(pid, {"channels": {}, "mask": None})
        pdir = RAW / pid
        if not all((pdir / f"{pid}{c}").exists() for c in CHANNELS):
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(PER_CALL_TIMEOUT_S)
            try:
                children = {c["name"]: c["id"] for c in syn.getChildren(folders[pid])}
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(pids)}] {pid}: FAILED to list children: {e}", flush=True)
                manifest[pid] = entry
                json.dump(manifest, open(MANIFEST_PATH, "w"), indent=1)
                continue
            finally:
                signal.alarm(0)
            for c in CHANNELS:
                name = f"{pid}{c}"
                entry["channels"][name] = (download_one(syn, children[name], pdir / name)
                                           if name in children else "absent")
        else:
            for c in CHANNELS:
                entry["channels"][f"{pid}{c}"] = "cached"
        entry["mask"] = download_one(syn, masks[pid], pdir / f"{pid}_mask.nii.gz") if pid in masks else "absent"
        manifest[pid] = entry
        json.dump(manifest, open(MANIFEST_PATH, "w"), indent=1)
        print(f"[{i}/{len(pids)}] {pid}: {entry['channels']} mask={entry['mask']} "
              f"({time.time() - t0:.0f}s)", flush=True)

    n_ok = sum(1 for e in manifest.values() if e.get("mask") in ("ok", "cached")
               and all(v in ("ok", "cached") for v in e["channels"].values())
               and len(e["channels"]) == len(CHANNELS))
    print(f"DONE. {n_ok}/{len(pids)} cases complete (pre + post + mask).", flush=True)


if __name__ == "__main__":
    main()
