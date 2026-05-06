#!/bin/bash
# Submit all 27 jobs. Run from the rush-hour directory:
#   bash jobs/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/projects/amfa5003/rush-hour"

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
