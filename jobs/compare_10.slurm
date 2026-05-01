#!/bin/bash

# Difficulty 10 — 3 puzzles × 3 methods × 750 episodes
#
# Time estimate:
#   ~70 min per 500 episodes per method  →  750 ep ≈ 105 min per combo
#   9 combos total ≈ 945 min ≈ 15.75 h  →  wall time 24 h
#
# If the job is interrupted, resubmit — already-finished combos are
# skipped automatically (results saved to results/comparison_d10.csv).

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=rh-compare-d10
#SBATCH --output=rh-compare-d10.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

conda activate amfa-custom-env

python3 comparison_hpc.py \
    --difficulty 10 \
    --episodes 750 \
    --n-puzzles 3 \
    --eval-episodes 50
