#!/bin/bash
# Submit 27 independent SLURM jobs: 3 difficulties × 3 puzzles × 3 methods.
# Run this from the Rush-hour-git directory:
#   cd /projects/amfa5003/tangram   (wherever your Rush-hour code lives)
#   bash jobs/submit_all.sh
#
# Each job trains one combo and saves its policy to policies/.
# After all jobs finish, run evaluate_policies.py locally to generate plots.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"   # Rush-hour-git root on the cluster

# difficulty → (episodes, wall_time_hh:mm:ss)
declare -A EPISODES=( [10]=750   [12]=1200  [15]=2500 )
declare -A WALLTIME=( [10]="12:00:00" [12]="12:00:00" [15]="20:00:00" )

METHODS=(hrep vrep gnn)
PUZZLES=(0 1 2)
DIFFICULTIES=(10 12 15)

mkdir -p "${WORK_DIR}/logs"

submitted=0
for D in "${DIFFICULTIES[@]}"; do
    ep="${EPISODES[$D]}"
    wt="${WALLTIME[$D]}"
    for P in "${PUZZLES[@]}"; do
        for M in "${METHODS[@]}"; do

            policy="${WORK_DIR}/policies/d${D}_p${P}_${M}.pth"
            if [[ -f "$policy" ]]; then
                echo "[SKIP] d${D} p${P} ${M} — policy already exists"
                continue
            fi

            job_name="rh-d${D}-p${P}-${M}"

            sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=${wt}
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=${job_name}
#SBATCH --output=${WORK_DIR}/logs/${job_name}.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

python3 train_single.py \
    --difficulty ${D} \
    --puzzle-idx ${P} \
    --method     ${M} \
    --episodes   ${ep} \
    --eval-episodes 50
EOF

            echo "[SUBMITTED] d${D} p${P} ${M}  (${ep} ep, wall=${wt})"
            submitted=$((submitted + 1))
        done
    done
done

echo ""
echo "Submitted ${submitted} jobs. Monitor with:  squeue -u \$USER"
echo "When all finish, copy policies/ back and run:  python3 evaluate_policies.py"
