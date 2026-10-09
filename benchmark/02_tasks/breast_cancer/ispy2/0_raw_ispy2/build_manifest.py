import json, subprocess, time, os, re

def get_token():
    out = subprocess.run(
        ["curl","-s","--max-time","20","-d",
         "username=nbia_guest&password=&client_id=NBIA&grant_type=password",
         "https://services.cancerimagingarchive.net/nbia-api/oauth/token"],
        capture_output=True, text=True).stdout
    return json.loads(out)["access_token"], time.time()

token, token_ts = get_token()

def api_get(path, params, retries=4):
    global token, token_ts
    if time.time() - token_ts > 3000:
        token, token_ts = get_token()
    import urllib.parse
    qs = urllib.parse.urlencode(params)
    url = f"https://services.cancerimagingarchive.net/nbia-api/services/v2/{path}?{qs}"
    for attempt in range(retries):
        out = subprocess.run(["curl","-s","--max-time","30","-H",f"Authorization: Bearer {token}",url],
                              capture_output=True, text=True).stdout
        try:
            return json.loads(out)
        except Exception:
            time.sleep(3)
    return None

seg = json.load(open("/tmp/ispy2_seg_series.json"))
t0_seg = [s for s in seg if s["StudyDesc"] == "ISPY2_MRI_T0"]
print(f"n T0 SEG series: {len(t0_seg)}", flush=True)

DCE_PATTERNS = [
    "ISPY2: VOLSER: uni-lateral cropped: original DCE",
    "ISPY2: VOLSER: bi-lateral: original DCE",
]

T1_EXCLUDE = re.compile(r"\bt1\b|t1w|t1_|_t1|vibe|fl3d|vibrant", re.I)
T2_INCLUDE = re.compile(r"\bt2\b|t2w|t2_|_t2|fse|tirm|stir|spair", re.I)

manifest = []
errors = []
out_path = "/tmp/ispy2_manifest3.json"
for i, seg_s in enumerate(t0_seg):
    pid = seg_s["PatientID"]
    study_uid = seg_s["StudyInstanceUID"]
    series = api_get("getSeries", {"StudyInstanceUID": study_uid})
    if series is None:
        errors.append({"pid": pid, "error": "api_get failed"})
        continue
    dce = [s for s in series if s.get("SeriesDescription") in DCE_PATTERNS]
    t2_candidates = [s for s in series if s.get("Modality")=="MR" and
                      "VOLSER" not in s.get("SeriesDescription","") and
                      "loc" not in s.get("SeriesDescription","").lower() and
                      T2_INCLUDE.search(s.get("SeriesDescription","")) and
                      not T1_EXCLUDE.search(s.get("SeriesDescription",""))]
    best_t2 = max(t2_candidates, key=lambda s: s.get("ImageCount",0)) if t2_candidates else None
    entry = {
        "pid": pid,
        "study_uid": study_uid,
        "seg_uid": seg_s["SeriesInstanceUID"],
        "seg_desc": seg_s["SeriesDescription"],
        "dce_uid": dce[0]["SeriesInstanceUID"] if dce else None,
        "dce_desc": dce[0]["SeriesDescription"] if dce else None,
        "dce_imagecount": dce[0]["ImageCount"] if dce else None,
        "best_t2_uid": best_t2["SeriesInstanceUID"] if best_t2 else None,
        "best_t2_desc": best_t2["SeriesDescription"] if best_t2 else None,
        "best_t2_imagecount": best_t2["ImageCount"] if best_t2 else None,
        "all_t2_candidates": [{"desc": s["SeriesDescription"], "n": s["ImageCount"]} for s in t2_candidates],
        "n_total_series": len(series),
    }
    manifest.append(entry)
    if (i+1) % 50 == 0 or (i+1)==len(t0_seg):
        n_both = sum(1 for e in manifest if e["dce_uid"] and e["best_t2_uid"])
        print(f"[{i+1}/{len(t0_seg)}] {pid}: dce={'Y' if dce else 'N'} t2={'Y' if best_t2 else 'N'}  running_both={n_both}", flush=True)
        json.dump({"manifest": manifest, "errors": errors}, open(out_path, "w"))

json.dump({"manifest": manifest, "errors": errors}, open(out_path, "w"))
print("MANIFEST3_BUILD_DONE", flush=True)
