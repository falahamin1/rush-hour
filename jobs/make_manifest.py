"""
Generate one manifest CSV per difficulty for the puzzle-scaling sweep.

Each manifest_d{D}.csv line is "puzzle_idx,method" (no header, 1-indexed by
line number) so a SLURM array task can grab its row directly via:
    sed -n "${SLURM_ARRAY_TASK_ID}p" manifest_d10.csv

Puzzle availability is not a constraint (rush.txt has ~150K-300K puzzles per
difficulty), so n_puzzles is purely a compute-budget choice. Already-trained
combos are NOT filtered out here — train_single.py skips any combo whose
policy already exists, so re-running this after bumping n_puzzles only
trains the newly-added puzzle indices.

Usage:
  python3 jobs/make_manifest.py --n-puzzles 15
  python3 jobs/make_manifest.py --n-puzzles 20 --methods mlp cnn
"""
import os
import argparse

ALL_METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
DIFFICULTIES = [10, 12, 15]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-puzzles', type=int, required=True,
                     help='Number of puzzles per difficulty (puzzle_idx 0..N-1)')
    ap.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    args = ap.parse_args()

    jobs_dir = os.path.dirname(os.path.abspath(__file__))

    for D in DIFFICULTIES:
        path = os.path.join(jobs_dir, f'manifest_d{D}.csv')
        with open(path, 'w') as f:
            for puzzle_idx in range(args.n_puzzles):
                for method in args.methods:
                    f.write(f'{puzzle_idx},{method}\n')
        n_lines = args.n_puzzles * len(args.methods)
        print(f'  wrote {path}  ({n_lines} tasks)')

    total = args.n_puzzles * len(args.methods) * len(DIFFICULTIES)
    print(f'\n{total} total tasks across {len(DIFFICULTIES)} difficulties.')
    print('Submit with: bash jobs/submit_arrays.sh')


if __name__ == '__main__':
    main()
