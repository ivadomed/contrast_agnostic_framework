#!/bin/bash
#SBATCH --job-name=palette_motion_diag
#SBATCH --account=aip-jcohen
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/p/paulh/palette_motion/logs/%x_%j.out
# Find a Chrome launch configuration that works on a TamIA compute node.
cd /scratch/p/paulh/palette_motion
module load StdEnv/2023 nodejs/20.16.0
export TMPDIR=$SLURM_TMPDIR XDG_CACHE_HOME=$SLURM_TMPDIR/cache XDG_CONFIG_HOME=$SLURM_TMPDIR/config
mkdir -p $XDG_CACHE_HOME $XDG_CONFIG_HOME
echo "[diag] node $(hostname) shm=$(df -h /dev/shm | tail -1)"
CHDIR=node_modules/.remotion/chrome-headless-shell/linux64/chrome-headless-shell-linux64
cp -r $CHDIR $SLURM_TMPDIR/chrome
LOCAL=$SLURM_TMPDIR/chrome/chrome-headless-shell
ldd $LOCAL | grep "not found" && echo "[diag] missing libs"
timeout 20 $LOCAL --no-sandbox --headless --disable-gpu --dump-dom about:blank > /dev/null 2> $SLURM_TMPDIR/raw.err; echo "[diag] raw chrome exit=$? $(tail -3 $SLURM_TMPDIR/raw.err | tr '\n' ' ')"
try(){ local name=$1; shift; local t0=$(date +%s)
  if timeout 150 npx remotion still src/index.ts PaletteAug out/diag_$name.png --frame=520 --log=error "$@" > $SLURM_TMPDIR/$name.log 2>&1; then echo "[diag] OK $name $(( $(date +%s)-t0 ))s"; else echo "[diag] FAIL $name: $(grep -m1 -E 'Error|error' $SLURM_TMPDIR/$name.log | cut -c1-200)"; fi; }
try default
try local --browser-executable=$LOCAL
try local_swiftshader --browser-executable=$LOCAL --gl=swiftshader
try local_angle --browser-executable=$LOCAL --gl=angle
try local_swangle --browser-executable=$LOCAL --gl=swangle
echo "[diag] done"
