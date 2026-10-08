#!/usr/bin/env python3
"""
Generates sec/_suppl_tables.tex: per-SETTING (task x training modality) Dice
and HD95 for the six headline methods, with each setting's own significance.

Each setting is the SAME estimand as its task column in tab:meta, restricted to
one training modality: same held-out contrasts, same external cohorts, same
contrast_groups pooling, same patient-level sign-flip test. That is done by
re-running the shared combined_modality_summary.py on each task's own
*_combined_01_results.yaml with `modalities:` cut down to one entry and the
output sent to scratch -- no second implementation of the table or the test.

Why rebuilt (2026-10-02): the previous _suppl_tables.tex had no generator in the
repo and used a different estimand (own-dataset contrasts only), so its rows
silently disagreed with their task columns in tab:meta.

Usage:
  .venv/bin/python make_suppl_tables.py [--scratch DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "paper/cvpr_format_latex/sec/_suppl_tables.tex"
SCRIPT = REPO / "benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py"

# (task label, dataset, task folder, {modality name in config -> display label})
TASKS = [   # tab:tasks order: appearance-defined, then interface-bounded
    ("Glioma",  "brats2024-glioma", "brain_tumor",
     {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "FLAIR"}),
    ("MS",    "open-ms", "brain_ms", {"flair": "FLAIR", "t1w": "T1w"}),
    ("Breast",     "ispy2", "breast_cancer", {"t1wce": "T1-CE", "t2w": "T2w"}),
    ("Abdomen",      "chaos", "abdomen_healthy", {"t1in": "T1in", "t2spir": "T2spir"}),
    ("Brain", "on-harmony", "brain_healthy", {"T1w": "T1w", "T2w": "T2w", "dwi_ap": "DWI"}),
    ("Mandible",   "toothfairy2", "mandible_healthy", {"cbct": "CBCT"}),
    ("Spine",      "healthy-spine-tum", "spine_healthy", {"ct": "CT", "inphase": "Dixon in-phase"}),
    ("Pelvis",     "totalseg-pelvic", "pelvis_healthy", {"ct": "CT", "mri": "MRI"}),
]
METHODS = [("baseline", "nnU-Net"), ("synthseg_noEM", r"\makecell{SynthSeg\\-noEM}"),
           ("synthseg_EM", r"\makecell{SynthSeg\\-EM}"), ("srcsm", "SRCSM"), ("auglab_default", "Auglab"),
           ("auglabAug_v26_6_2_train050_val000", r"\makecell{\textbf{PALETTE-}\\\textbf{Aug}}")]
REF = "auglabAug_v26_6_2_train050_val000"


def parse_summary(md: Path) -> dict:
    """{metric: {method_key: (all_value or None, sig_p or None)}} from the shared
    summary's own markdown -- the numbers the shared layer computed, not ours."""
    out, metric = {}, None
    for line in md.read_text().splitlines():
        if line.startswith("## Dice"):
            metric = "dice"
        elif line.startswith("## HD95"):
            metric = "hd95"
        elif metric and line.startswith("| **"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            key = re.sub(r"\*|\s*\(Ours\)", "", cells[0]).strip()
            val = re.sub(r"\*", "", cells[-2])
            sig = re.sub(r"\*", "", cells[-1])
            f = lambda x: None if x in ("—", "-", "") else float(x)
            out.setdefault(metric, {})[key] = (f(val), f(sig))
    return out


def run_setting(cfg: dict, mod_name: str, ds: str, task: str, scratch: Path) -> dict:
    one = dict(cfg)
    one["modalities"] = [m for m in cfg["modalities"] if m["name"] == mod_name]
    if not one["modalities"]:
        raise SystemExit(f"{ds}: no modality {mod_name!r} in its combined config")
    out_dir = scratch / f"{ds}_{mod_name}"
    one["output_dir"], one["output_prefix"] = str(out_dir), "setting"
    yml = scratch / f"{ds}_{mod_name}.yaml"
    yml.write_text(yaml.safe_dump(one, sort_keys=False))
    env = dict(os.environ, PROJECT_ROOT=str(REPO),
               METRICS_ROOT=str(REPO / f"benchmark/02_tasks/{task}/{ds}/8_results_{ds}/02_metrics"))
    r = subprocess.run([sys.executable, str(SCRIPT), str(yml)], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{ds}/{mod_name} failed:\n{r.stderr[-2000:]}")
    return parse_summary(out_dir / "setting_summary.md")


def cell(v, best, sig, higher):
    if v is None:
        return "---"
    s = f"{v:.1f}"
    if best is not None and abs(v - best) < 1e-9:
        s = rf"\textbf{{{s}}}"
    if sig is not None and sig < 0.05:
        s += r"$^{\ast}$"
    return s


def table(rows, metric, caption, label):
    higher = metric == "dice"
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "c" * len(METHODS) + "}",
             r"\toprule", "Setting & " + " & ".join(l for _, l in METHODS) + r" \\", r"\midrule"]
    for name, res in rows:
        vals = {k: res.get(metric, {}).get(k, (None, None)) for k, _ in METHODS}
        finite = [v for v, _ in vals.values() if v is not None]
        if not finite:
            continue
        best = (max if higher else min)(finite)
        lines.append(name + " & " + " & ".join(
            cell(vals[k][0], best, vals[k][1] if k != REF else None, higher)
            for k, _ in METHODS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", rf"\caption{{{caption}}}",
              rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args()
    scratch = a.scratch or Path(tempfile.mkdtemp(prefix="suppl_tables_"))
    scratch.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, ds, task, mods in TASKS:
        cfg_path = next((REPO / f"benchmark/02_tasks/{task}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                        .glob("*combined_01_results.yaml"))
        cfg = yaml.safe_load(cfg_path.read_text())
        for mod, disp in mods.items():
            res = run_setting(cfg, mod, ds, task, scratch)
            rows.append((f"{label} {disp}", res))
            d = res.get("dice", {})
            print(f"{label:<11}{disp:<9}", "  ".join(
                f"{k[:6]}={d.get(k, (None,))[0]}" for k, _ in METHODS))

    common = (r" Each row is one setting (one training modality of one task), scored on "
              r"every test contrast of its task, the training contrast and external cohorts "
              r"included, exactly as its task column of \cref{tab:meta}, so the rows of a "
              r"task average to that column. "
              r"$^{\ast}$: PALETTE-Aug significantly better than that method within the "
              r"setting (same patient-level sign-flip test, Holm-corrected across "
              r"competitors). Spine has no SRCSM run.")
    tex = ["% GENERATED by paper/scripts/make_suppl_tables.py -- regenerate, do not edit.",
           table(rows, "dice", r"Per-setting Dice (\%, $\uparrow$), best per row in bold."
                 + common, "tab:suppl-dice"),
           table(rows, "hd95", r"Per-setting HD95 (mm, $\downarrow$), best per row in bold."
                 + common, "tab:suppl-hd95")]
    OUT.write_text("\n".join(tex) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
