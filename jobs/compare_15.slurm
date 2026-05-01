#!/bin/bash

# Difficulty 15 — 3 puzzles × 3 methods × 2500 episodes
#
# Time estimate:
#   ~70 min per 500 episodes per method  →  2500 ep ≈ 350 min per combo
#   9 combos total ≈ 3150 min ≈ 52.5 h  →  wall time 72 h  (needs long QoS)
#
# If the job is interrupted, resubmit — already-finished combos are
# skipped automatically (results saved to results/comparison_d15.csv).

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=rh-compare-d15
#SBATCH --output=rh-compare-d15.%j.out
#SBATCH --qos=long

module purge
module load anaconda

conda activate amfa-custom-env

python3 comparison_hpc.py \
    --difficulty 15 \
    --episodes 2500 \
    --n-puzzles 3 \
    --eval-episodes 50
