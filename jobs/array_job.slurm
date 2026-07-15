#!/bin/bash
# Generic array task: trains one (difficulty, puzzle_idx, method) combo,
# picked from a manifest CSV by SLURM_ARRAY_TASK_ID. Do not sbatch this
# directly -- submit_arrays.sh sets --array/--time/--job-name/--export
# per difficulty on the sbatch command line (those can't be baked into
# #SBATCH pragmas here since they vary per difficulty).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=amilan
#SBATCH --qos=normal

module purge
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

WORK_DIR="/projects/amfa5003/rush-hour"
cd "$WORK_DIR"
mkdir -p logs

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
IFS=',' read -r PUZZLE_IDX METHOD <<< "$LINE"

echo "[task ${SLURM_ARRAY_TASK_ID}] difficulty=${DIFFICULTY}  puzzle_idx=${PUZZLE_IDX}  method=${METHOD}  episodes=${EPISODES}"

python3 train_single.py \
    --difficulty "${DIFFICULTY}" \
    --puzzle-idx "${PUZZLE_IDX}" \
    --method     "${METHOD}" \
    --episodes   "${EPISODES}" \
    --eval-episodes 50
