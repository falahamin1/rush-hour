#!/bin/bash

# Difficulty 12 — 3 puzzles × 3 methods × 1200 episodes
#
# Time estimate:
#   ~70 min per 500 episodes per method  →  1200 ep ≈ 168 min per combo
#   9 combos total ≈ 1512 min ≈ 25.2 h  →  wall time 36 h  (needs long QoS)
#
# If the job is interrupted, resubmit — already-finished combos are
# skipped automatically (results saved to results/comparison_d12.csv).

#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=rh-compare-d12
#SBATCH --output=rh-compare-d12.%j.out
#SBATCH --qos=long

module purge
module load anaconda

conda activate amfa-custom-env

python3 comparison_hpc.py \
    --difficulty 12 \
    --episodes 1200 \
    --n-puzzles 3 \
    --eval-episodes 50
