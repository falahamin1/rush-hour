#!/bin/bash
# Resubmit only the (difficulty, puzzle_idx, method) combos that don't yet
# have a finished policy -- e.g. ones that got SLURM_CANCELLED due to time
# limit. Resubmitting the whole array would waste job slots re-running
# instant no-op skips for every combo that's already done; this only
# touches what's actually missing. Safe to run repeatedly.
#
# Each retried combo resumes from its last saved checkpoint (see
# checkpoint_utils.py / train_single.py), not from scratch.
#
# IMPORTANT: run `acompile` first (see submit_arrays.sh for why).
#
# Usage:
#   bash jobs/retry_failed.sh            # default: 50 concurrent per array
#   bash jobs/retry_failed.sh 20         # smaller throttle for a small retry batch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/projects/amfa5003/rush-hour"
THROTTLE="${1:-50}"

if ! module avail anaconda 2>&1 | grep -qi anaconda; then
    echo "ERROR: 'anaconda' module not visible from this shell."
    echo "Run 'acompile' first, then re-run this script from the compute-node shell it gives you."
    exit 1
fi

declare -A EPISODES=( [10]=750 [12]=1200 [15]=2500 )
declare -A WALLTIME=( [10]="12:00:00" [12]="12:00:00" [15]="20:00:00" )

for D in 10 12 15; do
    manifest="${SCRIPT_DIR}/manifest_d${D}.csv"
    if [[ ! -f "$manifest" ]]; then
        echo "[SKIP] no manifest for difficulty ${D}"
        continue
    fi

    missing=()
    line_no=0
    while IFS=',' read -r PUZZLE_IDX METHOD; do
        line_no=$((line_no + 1))
        policy="${WORK_DIR}/policies/d${D}_p${PUZZLE_IDX}_${METHOD}.pth"
        if [[ ! -f "$policy" ]]; then
            missing+=("$line_no")
        fi
    done < "$manifest"

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[OK] difficulty ${D}: all combos already have a saved policy"
        continue
    fi

    array_spec=$(IFS=,; echo "${missing[*]}")
    sbatch \
        --job-name="rh-retry-d${D}" \
        --output="${WORK_DIR}/logs/rh-d${D}-retry-%A_%a.out" \
        --time="${WALLTIME[$D]}" \
        --array="${array_spec}%${THROTTLE}" \
        --export=ALL,DIFFICULTY="${D}",EPISODES="${EPISODES[$D]}",MANIFEST="${manifest}" \
        "${SCRIPT_DIR}/array_job.slurm"

    echo "[SUBMITTED] difficulty ${D}: ${#missing[@]} missing combo(s) -> tasks ${array_spec}"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
