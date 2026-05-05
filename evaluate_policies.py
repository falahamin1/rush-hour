"""
Load saved policies from policies/ and generate comparison plots.

Run locally after copying policies/ back from the cluster:
  python3 evaluate_policies.py                    # use saved metrics (fast)
  python3 evaluate_policies.py --reeval           # re-run evaluation episodes

Plots are written to results/.
"""

import os
import sys
import glob
import argparse
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS, NUM_ACTIONS
from DeepSetRL import DeepSetActorCritic
from GraphNNRL import GNNActorCritic

H_DIM = MAX_CONSTRAINTS * 3   # 12
V_DIM = 4 * 2                 # 8

POLICY_DIR  = os.path.join(os.path.dirname(__file__), 'policies')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

METHODS  = ['hrep', 'vrep', 'gnn']
LABELS   = ['H-rep', 'V-rep', 'GNN']
COLORS   = ['#3498db', '#2ecc71', '#e74c3c']


# ── Model loading ────────────────────────────────────────────────────────────

def build_model(method):
    if method == 'hrep':
        return DeepSetActorCritic(input_dim=H_DIM, num_pieces=MAX_VEHICLES,
                                  num_actions=NUM_ACTIONS)
    if method == 'vrep':
        return DeepSetActorCritic(input_dim=V_DIM, num_pieces=MAX_VEHICLES,
                                  num_actions=NUM_ACTIONS)
    return GNNActorCritic(node_dim=3, num_actions=NUM_ACTIONS)


def load_policy(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(data['method'])
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data


# ── Evaluation ───────────────────────────────────────────────────────────────

def _get_action(model, method, obs, mask):
    with torch.no_grad():
        if method == 'hrep':
            s = torch.tensor(obs['h_rep'], dtype=torch.float32)
            s = s.view(MAX_VEHICLES, H_DIM).unsqueeze(0)
            logits, _ = model(s)
        elif method == 'vrep':
            s = torch.tensor(obs['v_rep'], dtype=torch.float32)
            s = s.view(MAX_VEHICLES, V_DIM).unsqueeze(0)
            logits, _ = model(s)
        else:
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0)
            logits, _ = model(h, adj)
        logits[0][~mask] = -1e10
        return torch.argmax(logits, dim=-1).item()


def evaluate(model, method, board_str, eval_episodes=100, max_steps=200):
    rewards, solves = [], 0
    for _ in range(eval_episodes):
        env = RushHourGym(board_str)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
            action = _get_action(model, method, obs, mask)
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


# ── Data collection ──────────────────────────────────────────────────────────

def collect_results(reeval, eval_episodes):
    """
    Returns a nested dict: results[difficulty][puzzle_idx][method] = metrics dict
    """
    pattern = os.path.join(POLICY_DIR, 'd*_p*_*.pth')
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No policy files found in {POLICY_DIR}/")
        print("Copy the policies/ directory from the cluster first.")
        raise SystemExit(1)

    results = {}
    for path in paths:
        model, data = load_policy(path)
        D  = data['difficulty']
        P  = data['puzzle_idx']
        M  = data['method']

        if reeval:
            print(f"  Re-evaluating d{D} p{P} {M} ...")
            metrics = evaluate(model, M, data['board_str'],
                               eval_episodes=eval_episodes)
        else:
            metrics = data['metrics']

        results.setdefault(D, {}).setdefault(P, {})[M] = metrics
        print(f"  d{D} p{P+1} {M:5s}  solve={metrics['solve_rate']*100:.1f}%  "
              f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_difficulty(difficulty, puzzle_data, out_dir):
    puzzles = sorted(puzzle_data.keys())

    # Per-puzzle bar charts (reward + solve rate side by side)
    for pidx in puzzles:
        mdata = puzzle_data[pidx]
        means  = [mdata.get(m, {}).get('mean_reward', 0) for m in METHODS]
        stds   = [mdata.get(m, {}).get('std_reward', 0)  for m in METHODS]
        solves = [mdata.get(m, {}).get('solve_rate', 0) * 100 for m in METHODS]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f'Rush Hour  |  Difficulty {difficulty} moves  |  Puzzle {pidx + 1}',
            fontsize=13
        )

        ax1.bar(LABELS, means, yerr=stds, color=COLORS, capsize=5, alpha=0.85)
        ax1.set_ylabel('Mean eval reward')
        ax1.set_title('Reward')

        ax2.bar(LABELS, solves, color=COLORS, alpha=0.85)
        ax2.set_ylabel('Solve rate (%)')
        ax2.set_ylim(0, 105)
        ax2.set_title('Solve Rate')
        for bar, val in zip(ax2.patches, solves):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(out_dir, f'd{difficulty}_puzzle{pidx + 1}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  → {path}")

    # Summary: all puzzles side by side, solve rate only
    fig, axes = plt.subplots(1, len(puzzles),
                             figsize=(5 * len(puzzles), 4), sharey=True)
    if len(puzzles) == 1:
        axes = [axes]
    fig.suptitle(f'Solve Rates — Difficulty {difficulty} moves', fontsize=13)
    for ax, pidx in zip(axes, puzzles):
        mdata  = puzzle_data[pidx]
        solves = [mdata.get(m, {}).get('solve_rate', 0) * 100 for m in METHODS]
        bars = ax.bar(LABELS, solves, color=COLORS, alpha=0.85)
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
    print(f"  → {path}")


def plot_cross_difficulty(results, out_dir):
    """One combined figure: rows = difficulties, cols = puzzles, bars = methods."""
    difficulties = sorted(results.keys())
    n_puzzles = max(len(v) for v in results.values())

    fig, axes = plt.subplots(len(difficulties), n_puzzles,
                             figsize=(5 * n_puzzles, 4 * len(difficulties)),
                             sharey=True)
    if len(difficulties) == 1:
        axes = [axes]
    if n_puzzles == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle('Solve Rates — All Difficulties & Puzzles', fontsize=14)

    for row, D in enumerate(difficulties):
        for col in range(n_puzzles):
            ax = axes[row][col]
            mdata = results[D].get(col, {})
            solves = [mdata.get(m, {}).get('solve_rate', 0) * 100 for m in METHODS]
            bars = ax.bar(LABELS, solves, color=COLORS, alpha=0.85)
            ax.set_title(f'D={D}  Puzzle {col + 1}')
            ax.set_ylim(0, 105)
            if col == 0:
                ax.set_ylabel('Solve rate (%)')
            for bar, val in zip(bars, solves):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'all_difficulties_summary.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Evaluate Rush Hour saved policies')
    ap.add_argument('--reeval', action='store_true',
                    help='Re-run evaluation episodes instead of using saved metrics')
    ap.add_argument('--eval-episodes', type=int, default=100,
                    help='Episodes per combo when --reeval is set (default 100)')
    args = ap.parse_args()

    print(f"Loading policies from {POLICY_DIR}/")
    if args.reeval:
        print(f"Re-evaluating with {args.eval_episodes} episodes each ...\n")
    else:
        print("Using saved metrics (pass --reeval to re-run evaluation)\n")

    results = collect_results(args.reeval, args.eval_episodes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nGenerating plots → {RESULTS_DIR}/")
    for D, puzzle_data in sorted(results.items()):
        print(f"\nDifficulty {D}:")
        plot_difficulty(D, puzzle_data, RESULTS_DIR)

    if len(results) > 1:
        print("\nCross-difficulty summary:")
        plot_cross_difficulty(results, RESULTS_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
