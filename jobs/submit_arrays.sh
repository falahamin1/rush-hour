#!/bin/bash
# Submit the puzzle-scaling sweep as 3 SLURM job arrays (one per difficulty),
# instead of one .slurm file per combo -- this is what makes 200+ combos
# manageable. Regenerate the manifests first if you haven't:
#   python3 jobs/make_manifest.py --n-puzzles 15
#
# IMPORTANT: run `acompile` first. sbatch inherits MODULEPATH from the
# shell you call it from -- on the plain login node `module load anaconda`
# isn't visible, so every task would die in <1s. See array_job.slurm /
# submit_all.sh for the full explanation.
#
# Usage:
#   bash jobs/submit_arrays.sh            # default: 50 concurrent tasks per array
#   bash jobs/submit_arrays.sh 100        # raise the concurrency cap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/projects/amfa5003/rush-hour"
THROTTLE="${1:-50}"

if ! module avail anaconda 2>&1 | grep -qi anaconda; then
    echo "ERROR: 'anaconda' module not visible from this shell."
    echo "Run 'acompile' first, then re-run this script from the compute-node shell it gives you."
    exit 1
fi

mkdir -p "${WORK_DIR}/logs"

declare -A EPISODES=( [10]=750 [12]=1200 [15]=2500 )
declare -A WALLTIME=( [10]="12:00:00" [12]="12:00:00" [15]="20:00:00" )

for D in 10 12 15; do
    manifest="${SCRIPT_DIR}/manifest_d${D}.csv"
    if [[ ! -f "$manifest" ]]; then
        echo "ERROR: ${manifest} not found. Run: python3 jobs/make_manifest.py --n-puzzles <N>"
        exit 1
    fi
    n=$(wc -l < "$manifest")
    if [[ "$n" -eq 0 ]]; then
        echo "[SKIP] difficulty ${D} -- manifest is empty"
        continue
    fi

    sbatch \
        --job-name="rh-sweep-d${D}" \
        --output="${WORK_DIR}/logs/rh-d${D}-%A_%a.out" \
        --time="${WALLTIME[$D]}" \
        --array="1-${n}%${THROTTLE}" \
        --export=ALL,DIFFICULTY="${D}",EPISODES="${EPISODES[$D]}",MANIFEST="${manifest}" \
        "${SCRIPT_DIR}/array_job.slurm"

    echo "[SUBMITTED] difficulty ${D}: ${n} tasks (throttled to ${THROTTLE} concurrent)"
done

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Per-array detail:  squeue -u \$USER -o '%.10i %.20j %.2t %.10M %.6D %R'"
