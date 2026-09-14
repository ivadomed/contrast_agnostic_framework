#!/usr/bin/env python3
"""
Download the Duke-Breast-Cancer-MRI subset of MAMA-MIA (Synapse syn60868042):
for each of the 291 DUKE_* cases with an expert tumour mask, pull all 5
DCE-MRI channels (_0000.._0004) + the expert mask.

Run on the VULCAN LOGIN NODE (compute nodes have no internet -- see CLAUDE.md).
Per org policy this must never block the login node indefinitely, so:
  - every single Synapse download is wrapped in a SIGINT-safe per-call
    alarm-based timeout (default 180s) -- a hung request is aborted and
    logged as a failure for that one file, not a hang for the whole script.
  - the script is resumable: it skips any file that already exists on disk
    with a non-zero size, so re-running after a kill/timeout picks up where
    it left off instead of re-downloading everything.
  - progress is written incrementally to a JSON manifest after every case
    (not buffered to the end), so a kill at any point still leaves a usable
    partial manifest for 02_01_convert_test_t1wce.py to work from.
Launch with nohup + background (never as a blocking foreground call):
  SYNAPSE_AUTH_TOKEN=... nohup .venv/bin/python \
    datasets/duke-breast-mri/0_raw_duke-breast-mri/download_duke.py \
    > /scratch/paulh/duke_download/download.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import synapseclient

STAGE = Path("/scratch/paulh/duke_download")
RAW = STAGE / "raw"
MANIFEST_PATH = STAGE / "duke_full_manifest.json"
PER_CALL_TIMEOUT_S = 180
MAX_RETRIES = 2


class TimeoutError_(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError_("per-file download timed out")


def timed_get(syn, synid, dest_dir):
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_CALL_TIMEOUT_S)
    try:
        f = syn.get(synid, downloadLocation=str(dest_dir), ifcollision="overwrite.local")
        return f.path
    finally:
        signal.alarm(0)


def download_one(syn, synid, dest_path: Path) -> str:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return "cached"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            got_path = timed_get(syn, synid, dest_path.parent)
            got = Path(got_path)
            if got.name != dest_path.name:
                got.rename(dest_path)
            if dest_path.stat().st_size > 0:
                return "ok"
            return "empty_file"
        except TimeoutError_:
            print(f"    [timeout] {synid} -> {dest_path.name} (attempt {attempt}/{MAX_RETRIES})", flush=True)
        except Exception as e:
            print(f"    [error] {synid} -> {dest_path.name}: {e} (attempt {attempt}/{MAX_RETRIES})", flush=True)
    return "failed"


def main():
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SystemExit("SYNAPSE_AUTH_TOKEN not set in environment")

    duke_folders = json.load(open(STAGE / "duke_image_folders.json"))   # name -> folder synid
    duke_masks = json.load(open(STAGE / "duke_masks.json"))             # name -> file synid

    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.load(open(MANIFEST_PATH))
        print(f"Resuming: {len(manifest)} cases already have manifest entries", flush=True)

    pids = sorted(duke_folders.keys())
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        pdir = RAW / pid
        entry = manifest.get(pid, {"channels": {}, "mask": None})

        # list the 5 channel files for this case (one getChildren call, timed)
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(PER_CALL_TIMEOUT_S)
        try:
            children = list(syn.getChildren(duke_folders[pid]))
        except (TimeoutError_, Exception) as e:
            print(f"[{i}/{len(pids)}] {pid}: FAILED to list folder children: {e}", flush=True)
            manifest[pid] = entry
            json.dump(manifest, open(MANIFEST_PATH, "w"), indent=1)
            continue
        finally:
            signal.alarm(0)

        for c in children:
            name = c["name"]
            if not name.endswith(".nii.gz"):
                continue
            status = download_one(syn, c["id"], pdir / name)
            entry["channels"][name] = status

        mask_status = download_one(syn, duke_masks[pid], pdir / f"{pid}_mask.nii.gz")
        entry["mask"] = mask_status

        manifest[pid] = entry
        json.dump(manifest, open(MANIFEST_PATH, "w"), indent=1)

        elapsed = time.time() - t0
        n_ok_chan = sum(1 for v in entry["channels"].values() if v in ("ok", "cached"))
        print(f"[{i}/{len(pids)}] {pid}: {n_ok_chan}/5 channels, mask={mask_status} "
              f"(elapsed {elapsed:.0f}s)", flush=True)

    n_complete = sum(
        1 for e in manifest.values()
        if e.get("mask") in ("ok", "cached")
        and sum(1 for v in e["channels"].values() if v in ("ok", "cached")) == 5
    )
    print(f"DONE. {n_complete}/{len(pids)} cases fully downloaded (5/5 channels + mask).", flush=True)


if __name__ == "__main__":
    main()
