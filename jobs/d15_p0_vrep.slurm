#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=rh-d15-p0-vrep
#SBATCH --output=/projects/amfa5003/rush-hour/logs/rh-d15-p0-vrep.%j.out
#SBATCH --qos=normal

module purge
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

mkdir -p /projects/amfa5003/rush-hour/logs
cd /projects/amfa5003/rush-hour

python3 train_single.py \
    --difficulty 15 \
    --puzzle-idx 0 \
    --method     vrep \
    --episodes   2500 \
    --eval-episodes 50
