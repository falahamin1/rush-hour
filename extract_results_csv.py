"""
Read-only extractor: walks existing on-disk artifacts (policies/*.pth +
logs/*.out) and writes results_extracted.csv in the unified cross-benchmark
format. Does not train, does not touch training code, does not submit jobs.

Ground truth for "what ran" is policies/*.pth (one file per completed
(difficulty, puzzle_idx, method) combo, each holding a 'metrics' dict with
solve_rate/mean_reward/std_reward from a 50-episode greedy-only evaluation,
plus board_str/train_episodes/train_seconds). These predate the dual-mode
eval + curve logging added to train_single.py's schema-writer path, so
per-episode step counts, rollout-to-first-solve, and entropy were never
recorded for these runs at training time.

`solved` is read straight from the saved metrics dict (greedy, already
computed at training time -- not re-derived). `steps` (for both eval modes)
is reconstructed by loading each saved best-model policy and running
frozen-weight rollouts via eval_utils.evaluate() -- inference only, no
training, no gradient updates. One row per (combo, eval_mode): greedy uses
the already-known solved outcome for consistency with the trained-time
number; stochastic solved/steps are both freshly computed since no
stochastic evaluation existed before.

Provenance is decided by cross-referencing logs/*.out: a combo is "trusted"
only if some surviving log shows that exact combo's run printing
"Policy saved" (i.e., completing end-to-end under whatever code produced
that log). A combo whose .pth exists but has no such surviving log is
marked "unverified". Nothing is dropped either way.
"""
import csv
import glob
import os
import re

import numpy as np
import torch

from rush_hour_env import MAX_VEHICLES, NUM_ACTIONS, POSE_DIM, GRID_CHANNELS
from DeepSetRL import DeepSetActorCritic
from GraphNNRL import GNNActorCritic
from MLPRL import MLPActorCritic
from CNNRL import CNNActorCritic
import eval_utils

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(REPO_DIR, 'policies')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
PUZZLE_FILE = os.path.join(REPO_DIR, 'rush.txt')
OUT_CSV = os.path.join(REPO_DIR, 'results_extracted.csv')

METHODS = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
DIFFICULTIES = [10, 12, 15]
N_PUZZLES = 15  # puzzle_idx 0..14, per the manifest/array-job design
N_ROLLOUT_EPISODES = 20  # per (combo, eval_mode) -- 225 x 2 x 20 = 9000 rollouts total

H_DIM = eval_utils.H_DIM
V_DIM = eval_utils.V_DIM

POLICY_SAVED_RE = re.compile(
    r"\[train_single\] Policy saved.*?d(\d+)_p(\d+)_(hrep|vrep|gnn|mlp|cnn)\.pth"
)


def scan_log_provenance():
    completed = set()
    for path in glob.glob(os.path.join(LOGS_DIR, '*.out')):
        try:
            with open(path, errors='ignore') as f:
                text = f.read()
        except OSError:
            continue
        for m in POLICY_SAVED_RE.finditer(text):
            completed.add((int(m.group(1)), int(m.group(2)), m.group(3)))
    return completed


def build_optimal_steps_lookup():
    lookup = {}
    with open(PUZZLE_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                moves, board = parts[0], parts[1]
                lookup.setdefault(board, int(moves))
    return lookup


def build_model(method):
    if method == 'hrep':
        return DeepSetActorCritic(input_dim=H_DIM, num_pieces=MAX_VEHICLES, num_actions=NUM_ACTIONS)
    if method == 'vrep':
        return DeepSetActorCritic(input_dim=V_DIM, num_pieces=MAX_VEHICLES, num_actions=NUM_ACTIONS)
    if method == 'gnn':
        return GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=NUM_ACTIONS)
    if method == 'mlp':
        return MLPActorCritic(input_dim=POSE_DIM, num_actions=NUM_ACTIONS)
    return CNNActorCritic(in_channels=GRID_CHANNELS, num_actions=NUM_ACTIONS)


def mean_steps_on_solved(outcomes):
    solved_steps = [o['steps'] for o in outcomes if o['solved']]
    return float(np.mean(solved_steps)) if solved_steps else None


def main():
    optimal_steps_lookup = build_optimal_steps_lookup()
    log_completed = scan_log_provenance()

    rows = []
    missing = []
    n_done = 0
    for difficulty in DIFFICULTIES:
        for puzzle_idx in range(N_PUZZLES):
            for method in METHODS:
                pth_path = os.path.join(POLICY_DIR, f'd{difficulty}_p{puzzle_idx}_{method}.pth')
                if not os.path.exists(pth_path):
                    missing.append((difficulty, puzzle_idx, method))
                    continue
                ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
                board_str = ckpt['board_str']
                metrics = ckpt['metrics']
                combo = (difficulty, puzzle_idx, method)
                provenance = 'trusted' if combo in log_completed else 'unverified'
                instance_id = f'd{difficulty}_p{puzzle_idx}'
                opt_steps = optimal_steps_lookup.get(board_str, difficulty)

                model = build_model(method)
                model.load_state_dict(ckpt['model_state'])

                greedy_outcomes = eval_utils.evaluate(model, method, board_str, mode='greedy',
                                                       episodes=N_ROLLOUT_EPISODES, seed=puzzle_idx)
                stoch_outcomes = eval_utils.evaluate(model, method, board_str, mode='stochastic',
                                                      episodes=N_ROLLOUT_EPISODES, seed=puzzle_idx)

                # greedy `solved` stays the training-time number (already known,
                # deterministic given board+greedy policy); steps reconstructed
                # from the same deterministic rollout for consistency.
                greedy_solved = 1 if metrics['solve_rate'] >= 0.5 else 0
                greedy_steps = mean_steps_on_solved(greedy_outcomes)
                stoch_rate = sum(o['solved'] for o in stoch_outcomes) / len(stoch_outcomes)
                stoch_solved = 1 if stoch_rate >= 0.5 else 0
                stoch_steps = mean_steps_on_solved(stoch_outcomes)

                for mode, solved, steps in (
                    ('greedy', greedy_solved, greedy_steps),
                    ('stochastic', stoch_solved, stoch_steps),
                ):
                    rows.append({
                        'benchmark': 'rush_hour',
                        'tier': f'd{difficulty}',
                        'method': method,
                        'seed': 0,  # see summary: these 225 runs predate --seed
                        'instance_id': instance_id,
                        'eval_mode': mode,
                        'solved': solved,
                        'steps': f'{steps:.1f}' if steps is not None else '',
                        'optimal_steps': opt_steps,
                        'rollouts_to_solve': '',  # not recoverable, see summary
                        'final_entropy': '',      # not recoverable, see summary
                        'provenance': provenance,
                    })
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ...{n_done}/225 policies evaluated")

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'benchmark', 'tier', 'method', 'seed', 'instance_id', 'eval_mode', 'solved',
            'steps', 'optimal_steps', 'rollouts_to_solve', 'final_entropy', 'provenance',
        ])
        writer.writeheader()
        writer.writerows(rows)

    n_trusted = sum(1 for r in rows if r['provenance'] == 'trusted')
    n_unverified = sum(1 for r in rows if r['provenance'] == 'unverified')
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"trusted={n_trusted}  unverified={n_unverified}  missing_combos={len(missing)}")
    if missing:
        print("Missing (difficulty, puzzle_idx, method) combos:", missing)


if __name__ == '__main__':
    main()
