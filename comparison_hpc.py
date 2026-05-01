"""
HPC comparison: train hrep / vrep / gnn on 3 puzzles at a chosen difficulty.

Results are saved to results/comparison_d{D}.csv after *every* (puzzle, method)
combo, so a resubmitted job automatically skips what was already done.

Usage:
  python comparison_hpc.py --difficulty 10 --episodes 750
  python comparison_hpc.py --difficulty 12 --episodes 1200
  python comparison_hpc.py --difficulty 15 --episodes 2500
"""

import os
import sys
import csv
import argparse
import time

import matplotlib
matplotlib.use('Agg')          # headless — no display on the cluster
import matplotlib.pyplot as plt

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS, NUM_ACTIONS
from run_hrep import train_h_rep
from run_vrep import train_v_rep
from run_gnn  import train_graph_rep

H_DIM = MAX_CONSTRAINTS * 3   # 12
V_DIM = 4 * 2                 # 8

PUZZLE_FILE = os.path.join(os.path.dirname(__file__), 'rush.txt')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

CSV_FIELDS = [
    'difficulty', 'puzzle_idx', 'board_str', 'method',
    'mean_reward', 'std_reward', 'solve_rate',
    'train_episodes', 'train_seconds',
]


# ── Puzzle loading ───────────────────────────────────────────────────────────

def load_puzzles(difficulty, n=3):
    """
    Return up to n board strings whose required moves == difficulty.
    The file is sorted hardest-first, so we scan it fully (once).
    """
    found = []
    try:
        with open(PUZZLE_FILE) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[0]) == difficulty:
                    found.append(parts[1])
                    if len(found) == n:
                        break
    except Exception as e:
        print(f"Warning: could not read {PUZZLE_FILE}: {e}")
    return found


# ── Action selection ─────────────────────────────────────────────────────────

def _get_action(model, method, obs, mask, device):
    with torch.no_grad():
        if method == 'hrep':
            s = torch.tensor(obs['h_rep'], dtype=torch.float32)
            s = s.view(MAX_VEHICLES, H_DIM).unsqueeze(0).to(device)
            logits, _ = model(s)
        elif method == 'vrep':
            s = torch.tensor(obs['v_rep'], dtype=torch.float32)
            s = s.view(MAX_VEHICLES, V_DIM).unsqueeze(0).to(device)
            logits, _ = model(s)
        else:   # gnn
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(h, adj)
        logits[0][~mask] = -1e10
        return torch.argmax(logits, dim=-1).item()


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, method, board_str, eval_episodes=50, max_steps=200):
    device = next(model.parameters()).device
    model.eval()
    rewards, solves = [], 0
    for _ in range(eval_episodes):
        env = RushHourGym(board_str)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)
            action = _get_action(model, method, obs, mask, device)
            obs, r, done, _, info = env.step(action)
            total += r
            if done:
                if info.get('solved', False):
                    solves += 1
                break
        rewards.append(total)
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward':  float(np.std(rewards)),
        'solve_rate':  solves / eval_episodes,
    }


# ── CSV helpers ──────────────────────────────────────────────────────────────

def load_completed(csv_path):
    """Return set of (puzzle_idx, method) pairs already written to the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['puzzle_idx']), row['method']))
    return done


def append_result(csv_path, row):
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(csv_path, difficulty, out_dir):
    rows = []
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    puzzles = sorted(set(int(r['puzzle_idx']) for r in rows))
    methods = ['hrep', 'vrep', 'gnn']
    labels  = ['H-rep', 'V-rep', 'GNN']
    colors  = ['#3498db', '#2ecc71', '#e74c3c']

    def _get(ridx, m, key):
        match = [r for r in rows if int(r['puzzle_idx']) == ridx and r['method'] == m]
        return float(match[0][key]) if match else 0.0

    # one figure per puzzle
    for pidx in puzzles:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Rush Hour  |  Difficulty {difficulty} moves  |  Puzzle {pidx + 1}',
                     fontsize=13)

        means  = [_get(pidx, m, 'mean_reward') for m in methods]
        stds   = [_get(pidx, m, 'std_reward')  for m in methods]
        solves = [_get(pidx, m, 'solve_rate') * 100 for m in methods]

        ax1.bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax1.set_ylabel('Mean eval reward')
        ax1.set_title('Reward')

        ax2.bar(labels, solves, color=colors, alpha=0.85)
        ax2.set_ylabel('Solve rate (%)')
        ax2.set_ylim(0, 105)
        ax2.set_title('Solve Rate')

        plt.tight_layout()
        path = os.path.join(out_dir, f'd{difficulty}_puzzle{pidx + 1}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  plot → {path}")

    # summary: all puzzles side-by-side, solve rate only
    fig, axes = plt.subplots(1, len(puzzles),
                             figsize=(5 * len(puzzles), 4), sharey=True)
    if len(puzzles) == 1:
        axes = [axes]
    fig.suptitle(f'Solve Rates — Difficulty {difficulty} moves', fontsize=13)
    for ax, pidx in zip(axes, puzzles):
        solves = [_get(pidx, m, 'solve_rate') * 100 for m in methods]
        bars = ax.bar(labels, solves, color=colors, alpha=0.85)
        ax.set_title(f'Puzzle {pidx + 1}')
        ax.set_ylim(0, 105)
        ax.set_ylabel('Solve rate (%)')
        for bar, val in zip(bars, solves):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f'd{difficulty}_summary.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  summary plot → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_comparison(difficulty, episodes, n_puzzles=3, eval_episodes=50):
    csv_path = os.path.join(RESULTS_DIR, f'comparison_d{difficulty}.csv')

    print(f"\n{'='*65}")
    print(f"Difficulty: {difficulty} moves  |  Training: {episodes} ep  "
          f"|  Eval: {eval_episodes} ep  |  Puzzles: {n_puzzles}")
    print(f"{'='*65}")

    print("Loading puzzles from rush.txt (scanning file) ...")
    puzzles = load_puzzles(difficulty, n=n_puzzles)
    if len(puzzles) < n_puzzles:
        print(f"WARNING: only found {len(puzzles)} puzzles at difficulty={difficulty}")
    if not puzzles:
        print("ERROR: no puzzles found — aborting.")
        return

    for i, b in enumerate(puzzles):
        print(f"  Puzzle {i+1}: {b}")

    completed = load_completed(csv_path)
    total_combos = len(puzzles) * 3
    print(f"\nProgress: {len(completed)}/{total_combos} combos already done")

    methods = [
        ('hrep', lambda b: train_h_rep(board_str=b, episodes=episodes)),
        ('vrep', lambda b: train_v_rep(board_str=b, episodes=episodes)),
        ('gnn',  lambda b: train_graph_rep(board_str=b, episodes=episodes)),
    ]

    for pidx, board in enumerate(puzzles):
        for method_name, train_fn in methods:
            if (pidx, method_name) in completed:
                print(f"\n[SKIP] puzzle={pidx+1}  method={method_name} (already in CSV)")
                continue

            print(f"\n{'─'*65}")
            print(f"  puzzle {pidx+1}/{len(puzzles)}  |  method={method_name}  "
                  f"|  episodes={episodes}")
            print(f"{'─'*65}")

            t0 = time.time()
            _, best_model = train_fn(board)
            elapsed = time.time() - t0

            print(f"\nEvaluating {method_name} on puzzle {pidx+1} ...")
            metrics = evaluate(best_model, method_name, board,
                               eval_episodes=eval_episodes)

            row = {
                'difficulty':     difficulty,
                'puzzle_idx':     pidx,
                'board_str':      board,
                'method':         method_name,
                'mean_reward':    f"{metrics['mean_reward']:.4f}",
                'std_reward':     f"{metrics['std_reward']:.4f}",
                'solve_rate':     f"{metrics['solve_rate']:.4f}",
                'train_episodes': episodes,
                'train_seconds':  f"{elapsed:.1f}",
            }
            append_result(csv_path, row)

            print(f"  solve_rate={metrics['solve_rate']*100:.1f}%  "
                  f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}  "
                  f"time={elapsed/60:.1f} min")

    print(f"\n{'='*65}")
    print("All combos done. Generating plots ...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_results(csv_path, difficulty, RESULTS_DIR)
    print(f"\nCSV  → {csv_path}")
    print(f"Plots → {RESULTS_DIR}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Rush Hour HPC comparison')
    ap.add_argument('--difficulty',    type=int, required=True,
                    help='Exact number of moves required (e.g. 10, 12, 15)')
    ap.add_argument('--episodes',      type=int, required=True,
                    help='Training episodes per (puzzle, method) combo')
    ap.add_argument('--n-puzzles',     type=int, default=3)
    ap.add_argument('--eval-episodes', type=int, default=50)
    args = ap.parse_args()

    run_comparison(
        difficulty=args.difficulty,
        episodes=args.episodes,
        n_puzzles=args.n_puzzles,
        eval_episodes=args.eval_episodes,
    )
