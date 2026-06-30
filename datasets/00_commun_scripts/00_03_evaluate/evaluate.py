#!/usr/bin/env python3
"""
Common, method-agnostic segmentation evaluator (shared across all datasets).

Compares a directory of predictions against ground-truth labels and reports
per-case, per-label Dice and HD95. Knows nothing about which model/contrast
produced the predictions — it just matches `<case>.nii.gz` in --pred_dir to the
same name in --gt_dir.

This replaces the per-dataset copies of `06_00_evaluate*.py`; those files now
delegate here (see each dataset's 06_evaluate/ shim). The metric core lives in
00_00_utils/eval_metrics.py.

Label resolution (pick exactly one source):
  --dataset_json <dataset.json>   read foreground labels {name: id} from the
                                  "labels" block; pred_id == gt_id (same space).
      --labels NAME...            optionally restrict to a subset of those names
                                  (e.g. --labels liver for CT GT that only
                                  annotates the liver — others are excluded
                                  entirely, NOT counted as 0.0).
  --label_map <file|json>         cross-label-space mapping for cross-dataset
                                  evaluation. JSON object {name: [pred_id, gt_id]}
                                  given either as an inline string or a path to a
                                  .json file. Use when the prediction label space
                                  differs from the GT label space.

Usage:
  python evaluate.py --pred_dir D --gt_dir D --dataset_json J --out_csv C [--name TAG] [--labels ...] [--workers N]
  python evaluate.py --pred_dir D --gt_dir D --label_map M.json   --out_csv C [--name TAG] [--workers N]
"""
import argparse
import json
import sys
from pathlib import Path

# Make the shared metric core importable (sibling 00_00_utils dir; the leading
# digits in the dir names mean it can't be a normal package import).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_00_utils"))
import eval_metrics  # noqa: E402


def _triples_from_dataset_json(dataset_json: str, labels_filter):
    fg = {name: int(v)
          for name, v in json.loads(Path(dataset_json).read_text())["labels"].items()
          if isinstance(v, int) and int(v) != 0}
    if labels_filter:
        want = set(labels_filter)
        missing = want - set(fg)
        if missing:
            raise SystemExit(f"--labels not in dataset.json foreground labels: {sorted(missing)}")
        fg = {name: lab for name, lab in fg.items() if name in want}
    return [(name, lab, lab) for name, lab in fg.items()]


def _triples_from_label_map(label_map: str):
    p = Path(label_map)
    raw = p.read_text() if p.exists() else label_map
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--label_map is neither a readable file nor valid JSON: {e}")
    triples = []
    for name, ids in mapping.items():
        if not (isinstance(ids, (list, tuple)) and len(ids) == 2):
            raise SystemExit(f"--label_map['{name}'] must be [pred_id, gt_id], got {ids!r}")
        triples.append((name, int(ids[0]), int(ids[1])))
    return triples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="", help="optional tag stored in a 'group' column")
    ap.add_argument("--dataset_json", default=None,
                    help="read foreground labels from this dataset.json (same label space)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="restrict --dataset_json scoring to these label NAMES")
    ap.add_argument("--label_map", default=None,
                    help="cross-space map {name:[pred_id,gt_id]} as JSON file or inline")
    ap.add_argument("--workers", type=int, default=8, help="parallel worker processes")
    args = ap.parse_args()

    if args.label_map:
        triples = _triples_from_label_map(args.label_map)
        note = "cross-label-space: pred-ids remapped to GT-ids per label (see --label_map)"
    elif args.dataset_json:
        triples = _triples_from_dataset_json(args.dataset_json, args.labels)
        note = ""
    else:
        raise SystemExit("Provide either --dataset_json or --label_map to define labels.")

    eval_metrics.run_evaluation(
        pred_dir=args.pred_dir, gt_dir=args.gt_dir, out_csv=args.out_csv,
        name=args.name, triples=triples, workers=args.workers, summary_note=note)


if __name__ == "__main__":
    main()
