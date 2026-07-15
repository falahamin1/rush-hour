#!/bin/bash
# Submit every d*_p*_*.slurm job (one per puzzle x method combo). Run from
# the rush-hour directory:
#   bash jobs/submit_all.sh
#
# IMPORTANT: run `acompile` first. sbatch inherits MODULEPATH from the
# shell you call it from — on the plain login node `module load anaconda`
# isn't visible, so every job dies in <1s with "ModuleNotFoundError: torch"
# (conda silently fails to activate and python3 falls back to the system
# interpreter). acompile drops you into a compute-node shell with the full
# module stack; sbatch from there works correctly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/projects/amfa5003/rush-hour"

if ! module avail anaconda 2>&1 | grep -qi anaconda; then
    echo "ERROR: 'anaconda' module not visible from this shell."
    echo "Run 'acompile' first, then re-run this script from the compute-node shell it gives you."
    exit 1
fi

mkdir -p "${WORK_DIR}/logs"

submitted=0
for slurm in "$SCRIPT_DIR"/d*.slurm; do
    name=$(basename "$slurm" .slurm)
    policy="${WORK_DIR}/policies/${name}.pth"
    if [[ -f "$policy" ]]; then
        echo "[SKIP] ${name} — policy already exists"
        continue
    fi
    sbatch "$slurm"
    echo "[SUBMITTED] ${name}"
    submitted=$((submitted + 1))
done

echo ""
echo "Submitted ${submitted} jobs. Monitor with:  squeue -u \$USER"
