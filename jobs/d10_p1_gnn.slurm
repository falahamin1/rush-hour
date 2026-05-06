#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=rh-d10-p1-gnn
#SBATCH --output=/projects/amfa5003/rush-hour/logs/rh-d10-p1-gnn.%j.out
#SBATCH --qos=normal

module purge
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

mkdir -p /projects/amfa5003/rush-hour/logs
cd /projects/amfa5003/rush-hour

python3 train_single.py \
    --difficulty 10 \
    --puzzle-idx 1 \
    --method     gnn \
    --episodes   750 \
    --eval-episodes 50
