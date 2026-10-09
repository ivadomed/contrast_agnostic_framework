#!/usr/bin/env python3
"""
Rebuttal analysis (not in the paper): PALETTE alone (the ladder's real-fill rung,
no Auglab) next to every method of the headline comparison, on the SAME estimand as
the per-setting tables (tab:suppl-dice): every test contrast of the setting,
training contrast included, external cohorts pooled by contrast group -- i.e. the
shared combined_modality_summary.py run on each task's own combined config.

How: for each setting, each source metrics dir is mirrored into scratch with
symlinks (all its run dirs + the PALETTE-alone run, which lives under ablations/),
and the config gets one extra run "palette_alone". The shared script is unchanged.
Validation: the headline methods' values must equal those of the unmodified config.

Usage:  .venv/bin/python paper/scripts/compare_palette_alone_all_domains.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("PROJECT_ROOT", str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_dissociation_pvalues as C  # noqa: E402
from make_suppl_tables import run_setting  # noqa: E402

OUT = REPO / "paper" / "generated_results" / "palette_alone_vs_competitors"
DS = {"Glioma": ("brats2024-glioma", "brain_tumor"), "MS": ("open-ms", "brain_ms"),
      "Breast": ("ispy2", "breast_cancer"), "Abdomen": ("chaos", "abdomen_healthy"),
      "Brain": ("on-harmony", "brain_healthy"), "Mandible": ("toothfairy2", "mandible_healthy"),
      "Pelvis": ("totalseg-pelvic", "pelvis_healthy")}
MOD = {"Glioma": {"T1n": "t1n", "T1c": "t1c", "T2w": "t2w", "FLAIR": "t2f"},
       "MS": {"FLAIR": "flair", "T1w": "t1w"}, "Breast": {"T1-CE": "t1wce", "T2w": "t2w"},
       "Abdomen": {"T1in": "t1in", "T2spir": "t2spir"},
       "Brain": {"T1w": "T1w", "T2w": "T2w", "DWI": "dwi_ap"},
       "Mandible": {"CBCT": "cbct"}, "Pelvis": {"CT": "ct", "MRI": "mri"}}
COLS = [("baseline", "Base"), ("synthseg_noEM", "SynthSeg-noEM"), ("synthseg_EM", "SynthSeg-EM"),
        ("srcsm", "SRCSM"), ("auglab_default", "Auglab"), ("palette_alone", "PALETTE alone"),
        ("auglabAug_v26_6_2_train050_val000", "PALETTE-Aug")]


def find_rung(mdir: Path, rid: str) -> Path | None:
    rid = rid.split("/")[-1]
    cands = []
    for pre in ("", "nnUNet_", "auglab_"):
        cands += [mdir / f"{pre}{rid}", mdir / "ablations" / f"{pre}{rid}",
                  mdir.parent / "ablations" / mdir.name / f"{pre}{rid}"]
    return next((p for p in cands if p.is_dir()), None)


def main():
    scratch = Path(tempfile.mkdtemp(prefix="palette_alone_", dir=os.environ.get("SCRATCH", None)))
    rows, problems = [], []
    for lab, _, rel in C.ROWS:
        task, mod = lab.split(" ", 1)
        ds, tdir = DS[task]
        mr = str(REPO / f"benchmark/02_tasks/{tdir}/{ds}/8_results_{ds}/02_metrics")
        cfg_path = next((REPO / f"benchmark/02_tasks/{tdir}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                        .glob("*combined_01_results.yaml"))
        cfg = yaml.safe_load(cfg_path.read_text())
        mname = MOD[task][mod]
        rid = json.load(open(REPO / rel))["run_keys"][4]
        rid_bare = rid.split("/")[-1]
        for pre in ("nnUNet_", "auglab_"):
            rid_bare = rid_bare[len(pre):] if rid_bare.startswith(pre) else rid_bare

        mirrored = copy.deepcopy(cfg)
        m = [x for x in mirrored["modalities"] if x["name"] == mname][0]
        srcs = m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]
        for i, s in enumerate(srcs):
            md = Path(os.path.expandvars(s["metrics_dir"].replace("${METRICS_ROOT}", mr)))
            rung = find_rung(md, rid)
            if rung is None:
                problems.append(f"{lab}: no PALETTE-alone run in {md}")
                continue
            mirror = scratch / f"{ds}_{mname}_src{i}"
            mirror.mkdir(parents=True, exist_ok=True)
            for e in md.iterdir():
                if e.is_dir() and not (mirror / e.name).exists():
                    (mirror / e.name).symlink_to(e)
            link = mirror / rung.name
            if not link.exists():
                link.symlink_to(rung)
            s["metrics_dir"] = str(mirror)
        if "sources" in m:
            m["sources"] = srcs
        else:
            m["metrics_dir"] = srcs[0]["metrics_dir"]
        m["runs"] = dict(m["runs"], palette_alone=rid_bare)

        (scratch / "orig").mkdir(exist_ok=True); (scratch / "mirr").mkdir(exist_ok=True)
        ref = run_setting(cfg, mname, ds, tdir, scratch / "orig")
        new = run_setting(mirrored, mname, ds, tdir, scratch / "mirr")
        for key, _ in COLS:
            if key == "palette_alone" or key not in ref.get("dice", {}):
                continue
            if ref["dice"][key][0] != new["dice"].get(key, (None,))[0]:
                problems.append(f"{lab} {key}: mirrored {new['dice'].get(key)} != original {ref['dice'][key]}")
        rows.append((lab, {k: new["dice"].get(k, (None, None)) for k, _ in COLS}))
        print(lab, {n: new["dice"].get(k, (None,))[0] for k, n in COLS})

    OUT.mkdir(parents=True, exist_ok=True)
    head = "| setting | " + " | ".join(n for _, n in COLS) + " |"
    lines = [head, "|" + "---|" * (len(COLS) + 1)]
    for lab, d in rows:
        lines.append(f"| {lab} | " + " | ".join("--" if d[k][0] is None else f"{d[k][0]:.1f}" for k, _ in COLS) + " |")
    txt = "\n".join(lines) + "\n\nAll test contrasts of each setting (training contrast included), external cohorts pooled by contrast group: the tab:suppl-dice estimand.\n"
    txt += "\nproblems:\n" + ("\n".join(problems) if problems else "none (headline methods reproduce the unmodified config exactly)") + "\n"
    (OUT / "table_all_domains.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
