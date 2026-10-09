# PALETTE-Aug motion video (Remotion)

65 s, 1920×1080 video of the PALETTE-Aug story on real slices from 7 tasks. Every transformed frame is made by
the real transform helpers in `src/synthesis/v26_6_synthesis.py` (via `scripts/extract_assets.py`).
Built and rendered on TamIA in `/scratch/p/paulh/palette_motion` (CPU job, ~2 min); outputs copied to `out/` (gitignored).

1. Sync this folder: `tar cf - --exclude=node_modules --exclude=out --exclude=public/assets . | ssh tamia.alliancecan.ca 'cd /scratch/p/paulh/palette_motion && tar xf -'`
2. TamIA login node, downloads only (compute nodes have no internet): `bash scripts/stage_downloads.sh`
3. Vulcan: `bash scripts/stage_raw_from_vulcan.sh` (raw BraTS / CHAOS / I-SPY2 cases)
4. TamIA: `sbatch scripts/tamia_build_render.sh` → `out/palette_aug.mp4` (+ `palette_aug_web.mp4`, QC stills)

Edit scenes in `src/Main.tsx`; cases, slices and seeds in `scripts/extract_assets.py`. Scene lengths and event timings live in
`src/timeline.json`, read by both `Main.tsx` and `scripts/sound_design.py` (the synthesized soundtrack, 120 BPM, one beat = 15 frames),
so change timings there, not in either file.
Remotion is free for individuals, non-profits and companies of ≤3 people; larger companies need a company licence.
