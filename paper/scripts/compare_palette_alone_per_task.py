#!/usr/bin/env python3
"""
Rebuttal analysis (not in the paper), PER TASK: PALETTE alone (the ladder's
real-fill rung, no Auglab) next to every method of the headline comparison, on the
task-level estimand of tab:meta -- each task's full combined config (all training
modalities, all test contrasts incl. the training contrast, external cohorts pooled
by contrast group), run through the shared combined_modality_summary.py unchanged.

Each source metrics dir is mirrored into scratch with symlinks (its run dirs + the
PALETTE-alone run of that modality, which lives under ablations/), and every
modality gets one extra run "palette_alone". Validation: the headline methods'
task values must equal those of the unmodified config (= the tab:meta column).

Usage:  .venv/bin/python paper/scripts/compare_palette_alone_per_task.py
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_dissociation_pvalues as C  # noqa: E402
from make_suppl_tables import SCRIPT  # noqa: E402
import re  # noqa: E402


def parse_summary(md: Path) -> dict:
    """{metric: {method_key: (all, sig)}} -- every method row (bolded or not)."""
    out, metric = {}, None
    for line in md.read_text().splitlines():
        if line.startswith("## Dice"):
            metric = "dice"
        elif line.startswith("## HD95"):
            metric = "hd95"
        elif line.startswith("##"):
            metric = None
        elif metric and line.startswith("| ") and not line.startswith("| method") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            key = re.sub(r"\*|\s*\(Ours\)", "", cells[0]).strip()
            f = lambda x: None if x in ("—", "-", "") else float(re.sub(r"\*", "", x))
            try:
                out.setdefault(metric, {})[key] = (f(cells[-2]), f(cells[-1]))
            except ValueError:
                pass
    return out
from compare_palette_alone_all_domains import DS, MOD, COLS, find_rung  # noqa: E402

OUT = REPO / "paper" / "generated_results" / "palette_alone_vs_competitors"

# cross-task test (same design as tab:meta's p column), via the shared meta_task_heatmap module
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "meta_task_heatmap", REPO / "benchmark/00_commun_scripts/00_03_evaluate/meta_task_heatmap.py")
MT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MT)
from stat_tests import holm as _holm, macro_perm_design as _mpd, fmt_p as _fmt_p  # noqa: E402


def meta_modalities(cfg: dict, metrics_root: str) -> tuple:
    """A combined config dict -> (modalities, contrast_groups) in meta_task_heatmap's format
    (same expansion as MT.load_task_modalities, but from an in-memory, mirrored config)."""
    os.environ["METRICS_ROOT"] = metrics_root
    mods = []
    for m in cfg["modalities"]:
        srcs = m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]
        mods.append({"name": m["name"], "runs": m.get("runs", {}),
                     "sources": [{**s, "metrics_dir": Path(os.path.expandvars(str(s["metrics_dir"])))} for s in srcs]})
    return mods, (cfg.get("contrast_groups") if cfg.get("use_contrast_groups", True) else None)


def crosstask(xt: list, ref: str, comp: str, metric: str, higher_better: bool) -> tuple:
    present = [e for e in (MT.task_design_entries(t["modalities"], ref, comp, metric, t["contrast_groups"], 0)
                           for t in xt) if e]
    ents = [(j, leaf, unit, c) for j, es in enumerate(present) for _, leaf, unit, c in es]
    return _mpd(ents, len(present), higher_better) if present else (float("nan"),) * 3


def run_task(cfg: dict, ds: str, tdir: str, out_dir: Path) -> dict:
    one = dict(cfg)
    one["output_dir"], one["output_prefix"] = str(out_dir), "task"
    yml = out_dir.with_suffix(".yaml")
    out_dir.mkdir(parents=True, exist_ok=True)
    yml.write_text(yaml.safe_dump(one, sort_keys=False))
    env = dict(os.environ, PROJECT_ROOT=str(REPO),
               METRICS_ROOT=str(REPO / f"benchmark/02_tasks/{tdir}/{ds}/8_results_{ds}/02_metrics"))
    r = subprocess.run([sys.executable, str(SCRIPT), str(yml)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{ds} failed:\n{r.stderr[-2000:]}")
    return parse_summary(out_dir / "task_summary.md")


def main():
    scratch = Path(tempfile.mkdtemp(prefix="palette_alone_task_", dir=os.environ.get("SCRATCH")))
    # rung-5 run id per (task, modality name) from the ladders
    rung = {}
    for lab, _, rel in C.ROWS:
        task, mod = lab.split(" ", 1)
        rung[(task, MOD[task][mod])] = json.load(open(REPO / rel))["run_keys"][4]
    rows, problems, sig_rows, xt = [], [], [], []
    for task, (ds, tdir) in DS.items():
        n_prob = len(problems)
        mr = str(REPO / f"benchmark/02_tasks/{tdir}/{ds}/8_results_{ds}/02_metrics")
        cfg = yaml.safe_load(next((REPO / f"benchmark/02_tasks/{tdir}/{ds}/5_scripts_{ds}/06_evaluate/configs")
                                  .glob("*combined_01_results.yaml")).read_text())
        mirrored = copy.deepcopy(cfg)
        for mi, m in enumerate(mirrored["modalities"]):
            rid = rung.get((task, m["name"]))
            if rid is None:
                problems.append(f"{task}/{m['name']}: no ladder"); continue
            bare = rid.split("/")[-1]
            for pre in ("nnUNet_", "auglab_"):
                bare = bare[len(pre):] if bare.startswith(pre) else bare
            srcs = m.get("sources") or [{"metrics_dir": m["metrics_dir"]}]
            for si, s in enumerate(srcs):
                md = Path(os.path.expandvars(s["metrics_dir"].replace("${METRICS_ROOT}", mr)))
                r = find_rung(md, rid)
                if r is None:
                    problems.append(f"{task}/{m['name']}: no PALETTE-alone run in {md}")
                    continue
                mirror = scratch / f"{ds}_{m['name']}_src{si}"
                mirror.mkdir(parents=True, exist_ok=True)
                for e in md.iterdir():
                    if e.is_dir() and not (mirror / e.name).exists():
                        (mirror / e.name).symlink_to(e)
                if not (mirror / r.name).exists():
                    (mirror / r.name).symlink_to(r)
                s["metrics_dir"] = str(mirror)
            if "sources" in m:
                m["sources"] = srcs
            else:
                m["metrics_dir"] = srcs[0]["metrics_dir"]
            m["runs"] = dict(m["runs"], palette_alone=bare)
        if len(problems) == n_prob:     # PALETTE alone present for every modality of the task
            mods, groups = meta_modalities(mirrored, mr)
            xt.append({"name": task, "modalities": mods, "contrast_groups": groups})
        ref = run_task(cfg, ds, tdir, scratch / f"orig_{ds}")
        new = run_task(mirrored, ds, tdir, scratch / f"mirr_{ds}")
        asref = copy.deepcopy(mirrored); asref["ref"] = "palette_alone"
        sig = run_task(asref, ds, tdir, scratch / f"sig_{ds}")
        sig_rows.append((task, {k: sig["dice"].get(k, (None, None))[1] for k, _ in COLS},
                         {k: sig.get("hd95", {}).get(k, (None, None))[1] for k, _ in COLS}))
        for key, _ in COLS:
            if key == "palette_alone" or key not in ref.get("dice", {}):
                continue
            if ref["dice"][key][0] != new["dice"].get(key, (None,))[0]:
                problems.append(f"{task} {key}: mirrored {new['dice'].get(key)} != original {ref['dice'][key]}")
        rows.append((task, {k: new["dice"].get(k, (None, None)) for k, _ in COLS},
                     {k: new.get("hd95", {}).get(k, (None, None)) for k, _ in COLS}))
        print(task, {n: new["dice"].get(k, (None,))[0] for k, n in COLS}, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    def tab(idx, title):
        lines = [f"### {title}", "", "| task | " + " | ".join(n for _, n in COLS) + " |",
                 "|" + "---|" * (len(COLS) + 1)]
        for task, dd, hh in rows:
            d = (dd, hh)[idx]
            lines.append(f"| {task} | " + " | ".join("--" if d[k][0] is None else f"{d[k][0]:.1f}" for k, _ in COLS) + " |")
        return lines
    txt = "\n".join(tab(0, "Dice (%)") + [""] + tab(1, "HD95 (mm)"))
    sl = ["", "### p: PALETTE alone better than that method (Holm-corrected one-sided, patient-level sign-flip; ref = PALETTE alone)", "",
          "| task | metric | " + " | ".join(n for k, n in COLS if k != "palette_alone") + " |",
          "|" + "---|" * (len(COLS) + 1)]
    for task, sd, sh in sig_rows:
        for name, d in (("Dice", sd), ("HD95", sh)):
            sl.append(f"| {task} | {name} | " + " | ".join("--" if d[k] is None else f"{d[k]:.2g}" for k, n in COLS if k != "palette_alone") + " |")
    txt += "\n".join(sl)
    txt += ("\n\nTask-level estimand of tab:meta (all training modalities, all test contrasts incl. the "
            "training contrast, external cohorts pooled by contrast group).\n\nproblems:\n"
            + ("\n".join(problems) if problems else "none (headline methods reproduce the unmodified config exactly)") + "\n")
    # cross-task (tasks = units of equal weight, patients as sign-flip units; Spine has no ladder)
    ours, comps = "auglabAug_v26_6_2_train050_val000", [k for k, _ in COLS if k not in ("palette_alone",
                                                         "auglabAug_v26_6_2_train050_val000")]
    xl = ["", f"### Cross-task (tasks: {', '.join(t['name'] for t in xt)}; same test as tab:meta's p column)", ""]
    for metric in ("dice", "hd95"):
        hb = metric == "dice"
        raw = [crosstask(xt, "palette_alone", c, metric, hb)[2] for c in comps]
        xl.append(f"- {metric}: PALETTE alone better than " + ", ".join(
            f"{c} p_holm={_fmt_p(p)}" for c, p in zip(comps, _holm(raw))))
        d, p2, p_aug = crosstask(xt, ours, "palette_alone", metric, hb)
        p_alone = crosstask(xt, ours, "palette_alone", metric, not hb)[2]
        xl.append(f"- {metric}: PALETTE-Aug minus PALETTE alone macroΔ={d:+.4f}; p two-sided={_fmt_p(p2)}, "
                  f"p(PALETTE-Aug better)={_fmt_p(p_aug)}, p(PALETTE alone better)={_fmt_p(p_alone)}")
    txt += "\n".join(xl) + "\n"
    (OUT / "table_per_task.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
