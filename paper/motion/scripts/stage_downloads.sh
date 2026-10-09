#!/bin/bash
# Login-node staging (downloads only, no compilation): npm packages, Remotion's headless Chrome, fonts.
# TamIA compute nodes have no internet, so tamia_build_render.sh expects these to exist already.
set -euo pipefail
cd /scratch/${USER:0:1}/${USER}/palette_motion
module load StdEnv/2023 nodejs/20.16.0
export npm_config_cache=/scratch/${USER:0:1}/${USER}/.npm_cache
mkdir -p public/fonts
if [ ! -d node_modules/@remotion/cli ] && [ -f package-lock.json ]; then
  npm ci --no-audit --no-fund --ignore-scripts   # pinned versions (remotion 4.0.532)
elif [ ! -d node_modules/@remotion/cli ]; then
  V=$(npm view remotion version); echo "[npm] remotion $V"
  npm install --no-audit --no-fund --ignore-scripts --save-exact "remotion@$V" "@remotion/cli@$V" react@18.3.1 react-dom@18.3.1
fi
npx remotion browser ensure
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
latin_url(){ curl -fsS -A "$UA" "$1" | awk '/\/\* latin \*\//{f=1} f && /src: url/{match($0,/https:[^)]+/); print substr($0,RSTART,RLENGTH); exit}'; }
get_font(){ [ -s "public/fonts/$1" ] && return 0; local u; u=$(latin_url "$2" || true); [ -z "$u" ] && u=$(latin_url "$3"); curl -fsS -o "public/fonts/$1" "$u"; echo "[font] $1"; }
get_font bricolage.woff2 "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,200..800&display=swap" "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700&display=swap"
get_font plexmono.woff2  "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&display=swap" "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap"
get_font plexsans.woff2  "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@100..700&display=swap" "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400&display=swap"
ls -la public/fonts; du -sh node_modules
