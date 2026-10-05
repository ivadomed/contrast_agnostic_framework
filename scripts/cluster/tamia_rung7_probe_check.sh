#!/usr/bin/env bash
# Inspect the rung-7 PROBE (tamia_pack_rung7_auglab_pv.sh PROBE=1): per recorded fold, did the real training start and
# finish its 2 epochs with no error, and did the run save the PV config? Run ON TamIA after the probe jobs ended.
#   bash scripts/cluster/tamia_rung7_probe_check.sh
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
R="$(cat /scratch/p/paulh/_packruns_rung7_auglab_pv_root_probe.txt)"
bad=0; n=0
for idx in "$R"/*/index.tsv; do
  while IFS=$'\t' read -r cmd log name done_; do
    n=$((n+1)); f="${name#fold}"; fold="${f%%_*}"; clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
    res="$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)"
    fd="$(ls -d "${res}"/Dataset*/*/fold_${fold} 2>/dev/null | head -1)"
    if [ -z "$fd" ]; then echo "FAIL  ${name}: no fold dir under ${res}"; bad=$((bad+1)); continue; fi
    L="$(ls -t "$fd"/training_log_*.txt 2>/dev/null | head -1)"
    ep=$(grep -c "Epoch [0-9]" "$L" 2>/dev/null); ep=${ep:-0}
    err=$(cat "${cmd%.sh}.log" "$L" 2>/dev/null | grep -ci "Traceback\|CUDA out of memory\|OutOfMemory\|Killed\|RuntimeError")
    pv=$(grep -c '"pv_prob": 1.0' "$fd"/transform_params_gpu_used_for_training.json 2>/dev/null)
    t=$(grep "Epoch time" "$L" 2>/dev/null | tail -1 | grep -o "[0-9.]* s")
    s=OK; [ "${ep}" -ge 2 ] || s="FAIL(epochs=${ep})"; [ "${err:-0}" = 0 ] || s="${s},ERRORS=${err}"; [ "${pv:-0}" -ge 1 ] || s="${s},NO_PV_IN_SAVED_CFG"
    [ "$s" = OK ] || bad=$((bad+1))
    printf '%-6s %-78s epochs=%s last_epoch=%s pv_cfg_saved=%s\n' "${s%%,*}" "${name}" "${ep}" "${t:-?}" "${pv:-0}"
    [ "$s" = OK ] || echo "       -> ${s}"
  done < "$idx"
done
echo "checked ${n} folds, problems: ${bad}"; [ "$bad" = 0 ] && echo "PROBE PASSED" || echo "PROBE FAILED"
