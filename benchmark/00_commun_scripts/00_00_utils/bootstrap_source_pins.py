#!/usr/bin/env python3
"""Create / check the SOURCE roster pin file (roster_run_ids.tsv) for a task whose training predates the roster drivers, so eval-only companions can use the
shared cross drivers. Roster-era sources (isles2022 and later) get it from their 05_28 controller; do NOT use this for them.

It scans the source's metrics dir  <src>/8_results_<src>/02_metrics/<model_type>/<contrast>/  for  <category>_<RUN_ID>/  (headline) and  ablations/<category>_<RUN_ID>/
(ladder rungs), parses RUN_ID = <dataset>_<contrast>_<METHOD>_<YYYYMMDD_HHMMSS> against the canonical METHOD names (roster_runs.py), and lists EVERY candidate per method,
newest marked. Old datasets often have several generations or non-canonical names: unparsed dirs are listed so you can map them with --map METHOD=RUN_ID (or --pin-from-config
<results yaml> to take the headline run ids from the source's own headline config -- the safest source of truth). Prints the plan; writes only with --write.

  bootstrap_source_pins.py --source-root <.../ispy2> --model-type ispy2_model --contrast t1wce [--pin-from-config <yaml>] [--map baseline=<RUN_ID>] [--write]
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_03_evaluate"))
from roster_runs import HEADLINE_METHODS, LADDER, pin_path  # noqa: E402

METHODS = list(dict.fromkeys(HEADLINE_METHODS + [m for _, _, m, _ in LADDER]))


def scan(metrics: Path):
    out = {}  # method -> [(ts, cat, run_id, in_ablations)]
    unparsed = []
    for sub, abl in ((metrics, False), (metrics / "ablations", True)):
        if not sub.is_dir():
            continue
        for d in sorted(sub.iterdir()):
            m = re.match(r"^(nnUNet|auglab)_(.+)$", d.name)
            if not d.is_dir() or not m:
                continue
            cat, rid = m.groups()
            mm = re.match(r"^.+?_(?P<method>" + "|".join(map(re.escape, sorted(METHODS, key=len, reverse=True))) + r")_(?P<ts>\d{8}_\d{6})$", rid)
            if mm:
                out.setdefault(mm["method"], []).append((mm["ts"], cat, rid, abl))
            else:
                unparsed.append(f"{'ablations/' if abl else ''}{d.name}")
    return out, unparsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True); ap.add_argument("--model-type", required=True); ap.add_argument("--contrast", required=True)
    ap.add_argument("--pin-from-config"); ap.add_argument("--map", nargs="*", default=[]); ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    root = Path(a.source_root).resolve(); name = root.name
    metrics = root / f"8_results_{name}" / "02_metrics" / a.model_type / a.contrast
    found, unparsed = scan(metrics)
    chosen = {}
    cfg_ids = set()
    if a.pin_from_config:
        import yaml
        cfg = yaml.safe_load(Path(a.pin_from_config).read_text()); cfg_ids = set(cfg.get("runs") or [])
    for m in METHODS:
        c = sorted(found.get(m, []))
        if cfg_ids:
            hit = [x for x in c if x[2] in cfg_ids]
            if hit: c = hit
        if c:
            chosen[m] = c[-1]
        print(f"{m:40s} " + (", ".join(f"{x[2]}{'*' if x is c[-1] else ''}" for x in sorted(found.get(m, []))) or "-- none found --"))
    for kv in a.map:
        m, rid = kv.split("=", 1)
        cat = "nnUNet" if (metrics / f"nnUNet_{rid}").is_dir() or (metrics / "ablations" / f"nnUNet_{rid}").is_dir() else "auglab"
        chosen[m] = ("manual", cat, rid, False)
    if unparsed:
        print("\nUNPARSED dirs (map by hand with --map METHOD=RUN_ID if they are the headline runs):\n  " + "\n  ".join(unparsed[:30]))
    missing = [m for m in HEADLINE_METHODS if m not in chosen]
    print(f"\npin plan: {len(chosen)} methods; headline methods without a run: {missing or 'none'} (the DualVal val100 mirror = OURS val000 id with _val100_)")
    if a.write:
        p = pin_path(root, name, a.model_type, a.contrast); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(f"{m}\t{c[1]}\t{c[2]}\n" for m, c in chosen.items())); print("wrote", p)


if __name__ == "__main__":
    main()
