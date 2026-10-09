#!/bin/bash
#SBATCH --job-name=palette_motion
#SBATCH --account=aip-jcohen
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/scratch/${USER:0:1}/${USER}/palette_motion/logs/%x_%j.out
# Build + render the PALETTE-Aug motion video on TamIA (CPU only).
#   STAGE=all (default): npm install (once), fonts (once), extract assets, QC stills, full render
#   STAGE=stills: skip the full render
set -euo pipefail
export SCRATCH=/scratch/${USER:0:1}/${USER}
W=$SCRATCH/palette_motion
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
STAGE=${STAGE:-all}
cd "$W"
mkdir -p logs out public/fonts
module load StdEnv/2023 python/3.11 nodejs/20.16.0
export npm_config_cache=$SCRATCH/.npm_cache
# Chrome on a compute node: temp/cache/config on node-local disk and a node-local copy of the binary
# (with these, launch takes ~4 s; without, the first build job timed out connecting to the browser)
export TMPDIR=$SLURM_TMPDIR XDG_CACHE_HOME=$SLURM_TMPDIR/cache XDG_CONFIG_HOME=$SLURM_TMPDIR/config
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
cp -r node_modules/.remotion/chrome-headless-shell/linux64/chrome-headless-shell-linux64 "$SLURM_TMPDIR/chrome"
BROWSER="--browser-executable=$SLURM_TMPDIR/chrome/chrome-headless-shell"
echo "[env] node $(node --version) cpus=${SLURM_CPUS_PER_TASK:-?} proxy=${https_proxy:-none}"
"$REPO/.venv/bin/python" -c "import PIL, nibabel, torch; print('[env] python deps ok')"

# compute nodes have no internet: downloads are staged on the login node by scripts/stage_downloads.sh
for f in node_modules/@remotion/cli public/fonts/bricolage.woff2 public/fonts/plexmono.woff2 public/fonts/plexsans.woff2; do
  [ -e "$f" ] || { echo "[error] missing $f - run scripts/stage_downloads.sh on the login node first"; exit 1; }
done

"$REPO/.venv/bin/python" scripts/extract_assets.py --repo "$REPO" --data-root "$W/raw_stage" "$SCRATCH" --out public/assets
"$REPO/.venv/bin/python" scripts/sound_design.py --timeline src/timeline.json --out public/audio/soundtrack.wav

for fr in 100 210 265 350 470 580 660 800 1100 1400 1650 1900 2080; do
  npx remotion still src/index.ts PaletteAug "out/still_$fr.png" --frame=$fr --log=error $BROWSER
done
echo "[stills] done"
[ "$STAGE" = stills ] && exit 0
npx remotion render src/index.ts PaletteAug out/palette_aug.mp4 --concurrency="${SLURM_CPUS_PER_TASK:-8}" --log=info $BROWSER
ls -la out/palette_aug.mp4
# smaller copy for embedding in a web page
npx remotion ffmpeg -y -loglevel error -i out/palette_aug.mp4 -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart out/palette_aug_web.mp4
ls -la out/palette_aug_web.mp4
echo "[render] done"
