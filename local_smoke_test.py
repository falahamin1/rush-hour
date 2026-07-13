"""
Local smoke test — trains all 5 encoders (hrep, vrep, gnn, mlp, cnn) for a
handful of PPO rollout/update iterations on one easy puzzle, just to confirm
the whole pipeline (training loop, per-method observation dispatch, model
saving, evaluation) runs cleanly end-to-end before submitting the real job
to the cluster via comparison_hpc.py.

Episode counts are deliberately tiny — this is NOT expected to produce good
solve rates, only to catch crashes/shape errors on your machine first.
Each "episode" below is one PPO rollout of 4096 env steps + update, so even
episodes=2 takes a couple of minutes on a laptop CPU.

Usage:
  python local_smoke_test.py
  python local_smoke_test.py --episodes 3 --eval-episodes 10 --max-moves 8
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from comparison_hpc import evaluate
from run_hrep import train_h_rep
from run_vrep import train_v_rep
from run_gnn import train_graph_rep
from run_mlp import train_flat_mlp
from run_cnn import train_cnn_rep

PUZZLE_FILE = os.path.join(os.path.dirname(__file__), 'rush.txt')
DEFAULT_BOARD = 'IBBxooIooLDDJAALooJoKEEMFFKooMGGHHHM'

METHODS = [
    ('hrep', 'H-rep DeepSet', train_h_rep),
    ('vrep', 'V-rep DeepSet', train_v_rep),
    ('gnn',  'Graph NN',      train_graph_rep),
    ('mlp',  'Flat MLP',      train_flat_mlp),
    ('cnn',  'CNN',           train_cnn_rep),
]


def _pick_puzzle(max_moves):
    """Grab an easy puzzle (<= max_moves) from rush.txt, or fall back."""
    try:
        with open(PUZZLE_FILE) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[0]) <= max_moves:
                    return parts[1]
    except Exception:
        pass
    return DEFAULT_BOARD


def run_smoke_test(episodes=2, eval_episodes=10, max_moves=6):
    board = _pick_puzzle(max_moves)
    print(f"Smoke-test puzzle (<= {max_moves} moves): {board}")
    print(f"episodes={episodes}  eval_episodes={eval_episodes}\n")

    results = {}
    for method, label, train_fn in METHODS:
        print(f"{'-'*60}\n{label} ({method})\n{'-'*60}")
        t0 = time.time()
        _, best_model = train_fn(board_str=board, episodes=episodes)
        train_s = time.time() - t0

        metrics = evaluate(best_model, method, board, eval_episodes=eval_episodes)
        results[method] = {**metrics, 'train_seconds': train_s}

        print(f"  train_time={train_s:.1f}s  "
              f"reward={metrics['mean_reward']:+.2f}+/-{metrics['std_reward']:.2f}  "
              f"solve_rate={metrics['solve_rate']*100:.0f}%\n")

    print(f"{'='*60}\nSUMMARY (expect near-random solve rates at these episode counts)\n{'='*60}")
    print(f"{'method':10s} {'reward':>16s} {'solve%':>8s} {'train_s':>9s}")
    for method, _, _ in METHODS:
        r = results[method]
        print(f"{method:10s} {r['mean_reward']:+7.2f}+/-{r['std_reward']:<5.2f}"
              f" {r['solve_rate']*100:7.0f}% {r['train_seconds']:8.1f}s")

    print("\nIf every method above trained and evaluated without errors, "
          "the pipeline is ready to run for real on the cluster:\n"
          "  python comparison_hpc.py --difficulty 10 --episodes 750")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Local smoke test for all 5 Rush Hour encoders')
    ap.add_argument('--episodes', type=int, default=2,
                     help='PPO rollout/update iterations per method (tiny on purpose)')
    ap.add_argument('--eval-episodes', type=int, default=10)
    ap.add_argument('--max-moves', type=int, default=6,
                     help='Pick a puzzle solvable in <= this many moves (easier = faster to sanity-check)')
    args = ap.parse_args()

    run_smoke_test(episodes=args.episodes,
                    eval_episodes=args.eval_episodes,
                    max_moves=args.max_moves)
