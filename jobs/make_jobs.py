"""
Generate one .slurm file per (difficulty, puzzle_idx, method) combo.

Each job trains a single combo via train_single.py and exits — this is what
lets all combos run in parallel on the cluster instead of one giant
sequential script. Existing files are left untouched unless --force is
passed, so it's safe to re-run after adding a new method.

Usage:
  python jobs/make_jobs.py                      # fill in any missing combos
  python jobs/make_jobs.py --methods mlp cnn     # only generate these methods
  python jobs/make_jobs.py --force               # overwrite existing files too
"""
import os
import argparse

WORK_DIR = "/projects/amfa5003/rush-hour"

DIFFICULTIES = [10, 12, 15]
N_PUZZLES = 3
ALL_METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']

# episodes / wall-time per difficulty (same for every method)
EPISODES = {10: 750, 12: 1200, 15: 2500}
WALLTIME = {10: '12:00:00', 12: '12:00:00', 15: '20:00:00'}

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time={walltime}
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=rh-d{difficulty}-p{puzzle_idx}-{method}
#SBATCH --output={work_dir}/logs/rh-d{difficulty}-p{puzzle_idx}-{method}.%j.out
#SBATCH --qos=normal

module purge
module load anaconda/2023.09
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

mkdir -p {work_dir}/logs
cd {work_dir}

python3 train_single.py \\
    --difficulty {difficulty} \\
    --puzzle-idx {puzzle_idx} \\
    --method     {method} \\
    --episodes   {episodes} \\
    --eval-episodes 50
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument('--force', action='store_true',
                     help='Overwrite files that already exist')
    args = ap.parse_args()

    jobs_dir = os.path.dirname(os.path.abspath(__file__))
    written, skipped = 0, 0

    for difficulty in DIFFICULTIES:
        for puzzle_idx in range(N_PUZZLES):
            for method in args.methods:
                name = f'd{difficulty}_p{puzzle_idx}_{method}'
                path = os.path.join(jobs_dir, f'{name}.slurm')
                if os.path.exists(path) and not args.force:
                    skipped += 1
                    continue
                content = TEMPLATE.format(
                    walltime=WALLTIME[difficulty],
                    difficulty=difficulty,
                    puzzle_idx=puzzle_idx,
                    method=method,
                    episodes=EPISODES[difficulty],
                    work_dir=WORK_DIR,
                )
                with open(path, 'w') as f:
                    f.write(content)
                written += 1
                print(f'  wrote {name}.slurm')

    print(f'\n{written} job files written, {skipped} already existed and were left alone.')


if __name__ == '__main__':
    main()
