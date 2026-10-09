#!/usr/bin/env bash
# run_job backend: set_slot (the original persistent 4-GPU workstation).
# A slot already carries a fixed GPU+CPU bundle, so --gpus/--cpus/--mem/--time
# are accepted (for call-site compatibility with the slurm backend) and
# ignored here. --wait is also a no-op: set_slot hands off to a detached
# systemd unit and returns right away on its own, with or without an
# explicit wait. WANDB_MODE is left untouched (today's online behavior).

run_job() {
    local slot="" log=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) shift 2 ;;
            --gpus) shift 2 ;;
            --cpus) shift 2 ;;
            --mem)  shift 2 ;;
            --time) shift 2 ;;
            --slot) slot="$2"; shift 2 ;;
            --log)  log="$2"; shift 2 ;;
            --wait) shift ;;
            --) shift; break ;;
            *) echo "run_job (set_slot): unknown arg '$1'" >&2; return 1 ;;
        esac
    done
    if [ -z "${slot}" ]; then
        echo "run_job (set_slot): --slot is required for this backend" >&2
        return 1
    fi

    if [ -n "${log}" ]; then
        set_slot ${slot} "$@" > "${log}" 2>&1
    else
        set_slot ${slot} "$@"
    fi
}
