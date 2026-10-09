#!/usr/bin/env python3
"""
generate_pack_gpu_map.py -- computes a PACK_GPU_MAP string for
run_job_pack_submit.sh from a recorded pack dir's index.tsv, so a dataset's
tamia_pack launcher doesn't have to hand-derive the placement by counting
rows itself.

Rule (generalizes the hand-derived brats2024-glioma 04_60 Pack C placement,
documented in the project notes' TamIA section): rows whose 'name' column (3rd col
of index.tsv) matches one of --heavy's substrings each get ONE GPU to
themselves, round-robin across as many GPUs as are available (wrapping if
there are more heavy rows than GPUs -- a degenerate case, best effort).
Every other ("light") row round-robins across whichever GPUs remain once
the heavy rows have claimed theirs; if heavy rows already claim every GPU,
light rows fall back to round-robining across all GPUs (sharing with heavy).
This is the same policy as the documented srcsm-gets-its-own-GPU rule: cheap
methods double up so an expensive one doesn't become the straggler holding
the whole node open.

Usage:
    python3 generate_pack_gpu_map.py <PACK_DIR>/index.tsv --gpus 4 --heavy srcsm
    # prints, e.g.: 3 3 3 0 1 2

    PACK_GPU_MAP="$(python3 "${ROOT}/scripts/job_runner/generate_pack_gpu_map.py" \
        "${PACK_DIR_C}/index.tsv" --gpus 4 --heavy srcsm)"

With no --heavy given (or none matching), output is plain round-robin i%gpus
over the row count -- i.e. identical to run_job_pack_submit.sh's own default
when PACK_GPU_MAP is left empty, so this script is also safe to call
unconditionally even for packs that don't need explicit placement.
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("index_tsv", help="path to a recorded PACK_DIR/index.tsv")
    ap.add_argument("--gpus", type=int, required=True, help="GPUs available on the node")
    ap.add_argument(
        "--heavy",
        action="append",
        default=[],
        help="substring of the 'name' column marking a heavy/slow method that "
        "should get a dedicated GPU (repeatable)",
    )
    args = ap.parse_args()

    with open(args.index_tsv) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    if not rows:
        print(f"generate_pack_gpu_map.py: {args.index_tsv} has no rows", file=sys.stderr)
        return 1

    names = [r[2] if len(r) > 2 else "" for r in rows]
    is_heavy = [any(h in n for h in args.heavy) for n in names]
    n_heavy = sum(is_heavy)
    g = args.gpus

    heavy_gpus_used = min(n_heavy, g)
    light_gpu_pool = list(range(heavy_gpus_used, g)) if heavy_gpus_used < g else list(range(g))

    gpu_map = [0] * len(rows)
    hi = li = 0
    for idx, heavy in enumerate(is_heavy):
        if heavy:
            gpu_map[idx] = hi % g
            hi += 1
        else:
            gpu_map[idx] = light_gpu_pool[li % len(light_gpu_pool)]
            li += 1

    print(" ".join(str(x) for x in gpu_map))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
