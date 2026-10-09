#!/usr/bin/env python3
"""Scaffold a NEW TRAINING dataset's train / predict / evaluate scripts by cloning the reference implementation
(default: benchmark/03_archive/isles2022 [ARCHIVED 2026-10-06 as a benchmark task, kept as THE scaffolding reference; its scripts keep the 02_tasks-depth hop counts on purpose -- never "fix" them, only clone them] -- two training contrasts, 6-method suite + OURS val100 mirror + ladder rungs 2-5,
roster-driven predict/eval on top of the shared drivers in 00_commun_scripts) with name / contrast / id substitutions.
Sibling of create_dataset_structure.py (which makes the empty 9-subdir skeleton); run THAT first.

Clones (relative to <dataset>/5_scripts_<dataset>/):
  00_utils/env.sh, env_<contrast2>.sh      (env.sh carries a loud TODO banner: ids, BIDS leaf, header text must be reviewed)
  02_nnunet/<Name>Trainers.py, 02_03_plan_and_preprocess.sh, 02_04_verify_heldout.sh
  <dataset>/ (python package: nnUNetTrainer<Name>{Base,Baseline,AugLabDefault,AugLabValSynth,AugLabDualVal})
  04_train/*   05_predict/*   06_evaluate/* (shell + python, NOT configs/ -- those are GENERATED from the roster after training)
  and repo-level scripts/cluster/tamia_env_<dataset>.sh
NOT cloned (dataset-specific, hand-written): 00_bidsify/, 01_create_splits/, 02_01_convert*, orientation QC, 9_tests.

Substitutions (two-phase via placeholders, so a new contrast named like another cannot collide):
  dataset slug (isles2022), trainer-class Name (ISLES2022), the two training contrasts (dwi, flair; the test items ARE those two contrasts),
  the foreground label (lesion, only the EVAL_LABELS line) and the target description text in trainer docstrings.
Refuses to overwrite existing files unless --force.

  .venv/bin/python benchmark/create_pipeline_scripts.py --task brain_xyz --dataset my-ds --contrasts t1 t2 \
      --dataset-ids 150 151 --labels tumour --target-desc "glioma tumour"
  (identity check: --dataset isles2022 --task brain_stroke --contrasts dwi flair --dataset-ids 140 141 --out-root <tmp>)
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REF_TASK, REF_DS, REF_NAME = "brain_stroke", "isles2022", "ISLES2022"
REF_C = ("dwi", "flair"); REF_IDS = ("140", "141")
REF_LABEL = "lesion"; REF_DESC = "ischemic stroke lesion"
# The reference's epoch comment credits a decision to the user ("Paul, 2026-10-04: ...") and quotes isles2022's cohort size: never copy it to another dataset.
EPOCH_TODO = ("# PROJECT_TODO EPOCH POLICY: decide by analogy with the project's epoch table (brats 2500; on-harmony / open-ms / isles2022 2000; ispy2 1000; chaos 200),\n"
              "# write the number + the real rationale (cohort size, cost) here as YOUR decision (unless the user gave a number), and never attribute it to the user.\n"
              "# Re-time RUN_JOB_TIME_DEFAULT / job limits from a sizing probe on the target cluster before trusting them.\n")
SKIP_DIRS = {"__pycache__", "logs", "configs", "toDelete"}
TODO_BANNER = ("# >>> SCAFFOLDED from the isles2022 reference by benchmark/create_pipeline_scripts.py. REVIEW BEFORE USE: dataset ids, BIDS leaf,\n"
               "# >>> epoch policy, RUN_JOB_* resources and ALL descriptive comments below still describe the reference, not this dataset.\n")


def make_subst(a):
    name = a.name or re.sub(r"[^A-Za-z0-9]", "", a.dataset).upper()
    c1, c2 = a.contrasts
    pkg = a.pkg or a.dataset.replace("-", "_")
    pairs = [
        (f"{REF_TASK}/{REF_DS}", "@@TASKDS@@", f"{a.task}/{a.dataset}"),   # repo-relative dataset paths in comments/scripts
        (f'"{REF_TASK}", "{REF_DS}"', "@@TASKDS2@@", f'"{a.task}", "{a.dataset}"'),   # os.path.join(...) form in the trainer shim
        ("stroke-brain-isles2022", "@@LEAF@@", a.bids_leaf or f"TODO-{a.dataset}"),
        (f"{REF_DS}_model", "@@MT@@", f"{pkg}_model"),               # MODEL_TYPE: python-safe, e.g. open_ms_model
        (f"{REF_DS}.trainers", "@@PKG@@.trainers", f"{pkg}.trainers"),  # python imports
        (f"from {REF_DS} ", "@@FROMPKG@@ ", f"from {pkg} "),  # (reference token, placeholder, new value) -- order matters: longest / most specific first
        (REF_DESC, "@@DESC@@", a.target_desc),
        ("ISLES'22", "@@PAPERNAME@@", "ISLES'22" if a.dataset == REF_DS else a.dataset),
        (REF_NAME, "@@NAME@@", name),
        (REF_DS, "@@DS@@", a.dataset),
        (REF_C[0].upper(), "@@C1U@@", c1.upper()), (REF_C[1].upper(), "@@C2U@@", c2.upper()),
        (REF_C[0], "@@C1@@", c1), (REF_C[1], "@@C2@@", c2),
        (REF_IDS[0], "@@ID1@@", a.dataset_ids[0]), (REF_IDS[1], "@@ID2@@", a.dataset_ids[1]),
    ]
    return pairs


def strip_ref_only(text: str) -> str:
    """Drop '# <<ref-only' ... '# ref-only>>' blocks: commentary that is only true for the reference dataset."""
    return re.sub(r"\n# <<ref-only.*?# ref-only>>", "", text, flags=re.S)


def map_path(rel: str, pairs, pkg: str) -> str:
    """Relative-path mapping; the reference python-package dir (isles2022/) becomes the python-safe pkg name."""
    parts = Path(rel).parts
    head = parts[0]
    new_head = pkg if head == REF_DS else apply(head, pairs)
    return str(Path(new_head, *[apply(x, pairs) for x in parts[1:]]))


def apply(text: str, pairs) -> str:
    for ref, ph, _ in pairs:
        if ref:
            text = text.replace(ref, ph)
    for _, ph, new in pairs:
        text = text.replace(ph, new)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task folder under benchmark/02_tasks/, e.g. brain_stroke")
    ap.add_argument("--dataset", required=True, help="dataset slug (also the dir name), e.g. my-ds")
    ap.add_argument("--name", help="trainer-class infix (default: slug upper, non-alphanumerics removed), e.g. MYDS")
    ap.add_argument("--pkg", help="python package name for the trainers (default: slug with '-' -> '_', like open-ms -> open_ms)")
    ap.add_argument("--bids-leaf", help="BIDS leaf dir name under 1_BIDS_<dataset>/ (lab schema <pathology>-<anatomy>-<study>); default: TODO-<dataset>")
    ap.add_argument("--contrasts", nargs=2, required=True, metavar=("C1", "C2"), help="the two training contrasts (C1 is the env.sh default)")
    ap.add_argument("--dataset-ids", nargs=2, required=True, metavar=("ID1", "ID2"), help="nnU-Net Dataset ids for C1 / C2 (unused: ls benchmark/*/*/2_nnUNet_*/raw)")
    ap.add_argument("--labels", default=REF_LABEL, help="foreground label name(s) in dataset.json (space-separated string)")
    ap.add_argument("--target-desc", default="segmentation target", help="short description replacing 'ischemic stroke lesion' in trainer docstrings")
    ap.add_argument("--reference", default=str(PROJECT / "benchmark/03_archive" / REF_DS))
    ap.add_argument("--out-root", help="default: benchmark/02_tasks/<task>/<dataset>")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ref = Path(a.reference).resolve()
    out = Path(a.out_root).resolve() if a.out_root else PROJECT / "benchmark/02_tasks" / a.task / a.dataset
    pairs = make_subst(a)
    pkg = a.pkg or a.dataset.replace("-", "_")
    rs, os_ = ref / f"5_scripts_{REF_DS}", out / f"5_scripts_{a.dataset}"
    name = [n for r, p, n in pairs if p == "@@NAME@@"][0]

    sel = []  # (src, dst)
    def add_tree(sub: str, patterns=("*.sh", "*.py")):
        for p in sorted((rs / sub).rglob("*")):
            if p.is_dir() or SKIP_DIRS & set(p.relative_to(rs).parts) or not any(p.match(g) for g in patterns):
                continue
            sel.append((p, os_ / map_path(str(p.relative_to(rs)), pairs, pkg)))
    add_tree("00_utils"); add_tree("04_train"); add_tree("05_predict"); add_tree("06_evaluate"); add_tree(REF_DS)
    for f in (f"02_nnunet/{REF_NAME}Trainers.py", "02_nnunet/02_03_plan_and_preprocess.sh", "02_nnunet/02_04_verify_heldout.sh"):
        sel.append((rs / f, os_ / apply(f, pairs)))
    sel.append((PROJECT / f"scripts/cluster/tamia_env_{REF_DS}.sh", (out.parents[3] if a.out_root is None else out.parent) / "scripts/cluster" / f"tamia_env_{a.dataset}.sh"))
    if a.out_root:  # test mode: keep the repo-level file inside the out root too
        sel[-1] = (sel[-1][0], out / "scripts_cluster" / f"tamia_env_{a.dataset}.sh")

    n_new = n_skip = 0
    for src, dst in sel:
        if not src.exists():
            sys.exit(f"reference file missing: {src}")
        text = apply(strip_ref_only(src.read_text()), pairs)
        if src.name == "env.sh" and src.parent.name == "00_utils":
            text = text.replace("#!/usr/bin/env bash\n", "#!/usr/bin/env bash\n" + TODO_BANNER, 1)
        if src.name == "06_01_evaluate_run.sh":
            text = text.replace(f'EVAL_LABELS="{REF_LABEL}"', f'EVAL_LABELS="{a.labels}"')
            text = text.replace(f"scoring the `{REF_LABEL}` label", f"scoring the `{a.labels}` label")
        if src.name == "04_00_common.sh":
            text = re.sub(r"# EPOCH POLICY:.*?(?=NNUNET_NUM_EPOCHS_DEFAULT)", lambda m: EPOCH_TODO, text, count=1, flags=re.S)
        if src.name.startswith("06_00_evaluate"):
            text = text.replace(f"background 0, {REF_LABEL} 1 -> score `--labels {REF_LABEL}`", f"background 0, {a.labels} 1 -> score `--labels {a.labels}`")
        if dst.exists() and not a.force:
            print("EXISTS (skipped):", dst); n_skip += 1; continue
        print(("would write " if a.dry_run else "write "), dst)
        if not a.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(text)
        n_new += 1
    print(f"\n{n_new} files {'planned' if a.dry_run else 'written'}, {n_skip} skipped. Trainer class infix: {name}")
    print("NEXT: review 00_utils/env.sh (TODO banner), write 00_bidsify/01_create_splits/02_01_convert by hand, then the skill's checklist.")


if __name__ == "__main__":
    main()
