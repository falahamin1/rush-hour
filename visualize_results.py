"""
Visualize Rush Hour comparison results from results/*.csv

Run:
    python visualize_results.py
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
METHODS     = ['hrep', 'vrep', 'gnn']
LABELS      = {'hrep': 'H-rep', 'vrep': 'V-rep', 'gnn': 'GNN'}
COLORS      = {'hrep': '#3498db', 'vrep': '#2ecc71', 'gnn': '#e74c3c'}
DIFFICULTIES = [10, 12, 15]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    rows = []
    for d in DIFFICULTIES:
        path = os.path.join(RESULTS_DIR, f'comparison_d{d}.csv')
        if not os.path.exists(path):
            continue
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                rows.append({
                    'difficulty':   int(row['difficulty']),
                    'puzzle_idx':   int(row['puzzle_idx']),
                    'method':       row['method'],
                    'mean_reward':  float(row['mean_reward']),
                    'std_reward':   float(row['std_reward']),
                    'solve_rate':   float(row['solve_rate']),
                    'train_seconds': float(row['train_seconds']),
                })
    return rows


def get(rows, difficulty, puzzle_idx, method, key):
    match = [r for r in rows
             if r['difficulty'] == difficulty
             and r['puzzle_idx'] == puzzle_idx
             and r['method'] == method]
    return float(match[0][key]) if match else None


# ── Plot 1: Solve rate — per difficulty, grouped by method ────────────────────

def plot_solve_rates(rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Rush Hour — Solve Rate per Puzzle', fontsize=14, fontweight='bold')

    for ax, diff in zip(axes, DIFFICULTIES):
        puzzles = sorted(set(r['puzzle_idx'] for r in rows if r['difficulty'] == diff))
        if not puzzles:
            ax.set_title(f'Difficulty {diff}')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        n_puzzles = len(puzzles)
        n_methods = len(METHODS)
        width     = 0.25
        x         = np.arange(n_puzzles)

        for i, method in enumerate(METHODS):
            values = [get(rows, diff, p, method, 'solve_rate') for p in puzzles]
            bars = ax.bar(
                x + (i - 1) * width,
                [v * 100 if v is not None else 0 for v in values],
                width, label=LABELS[method],
                color=COLORS[method], alpha=0.85,
            )
            for bar, v in zip(bars, values):
                if v is not None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f'{v*100:.0f}%',
                            ha='center', va='bottom', fontsize=8)

        ax.set_title(f'Difficulty {diff} moves', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Puzzle {p+1}' for p in puzzles])
        ax.set_ylim(0, 115)
        ax.set_ylabel('Solve rate (%)' if diff == 10 else '')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'solve_rates.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Plot 2: Mean reward — per difficulty, grouped by method ───────────────────

def plot_rewards(rows, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Rush Hour — Mean Eval Reward per Puzzle', fontsize=14, fontweight='bold')

    for ax, diff in zip(axes, DIFFICULTIES):
        puzzles = sorted(set(r['puzzle_idx'] for r in rows if r['difficulty'] == diff))
        if not puzzles:
            ax.set_title(f'Difficulty {diff}')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        width = 0.25
        x     = np.arange(len(puzzles))

        for i, method in enumerate(METHODS):
            means = [get(rows, diff, p, method, 'mean_reward') for p in puzzles]
            ax.bar(
                x + (i - 1) * width,
                [v if v is not None else 0 for v in means],
                width, label=LABELS[method],
                color=COLORS[method], alpha=0.85,
            )

        ax.set_title(f'Difficulty {diff} moves', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Puzzle {p+1}' for p in puzzles])
        ax.set_ylabel('Mean reward' if diff == 10 else '')
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, 'rewards.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Plot 3: Solve rate heatmap — method × difficulty (averaged) ───────────────

def plot_heatmap(rows, out_dir):
    data = np.full((len(METHODS), len(DIFFICULTIES)), np.nan)

    for j, diff in enumerate(DIFFICULTIES):
        for i, method in enumerate(METHODS):
            vals = [r['solve_rate'] for r in rows
                    if r['difficulty'] == diff and r['method'] == method]
            if vals:
                data[i, j] = np.mean(vals) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label='Solve rate (%)')

    ax.set_xticks(range(len(DIFFICULTIES)))
    ax.set_xticklabels([f'{d} moves' for d in DIFFICULTIES])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([LABELS[m] for m in METHODS])
    ax.set_title('Average Solve Rate — Method × Difficulty', fontsize=12, fontweight='bold')

    for i in range(len(METHODS)):
        for j in range(len(DIFFICULTIES)):
            v = data[i, j]
            text = f'{v:.0f}%' if not np.isnan(v) else 'N/A'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if (np.isnan(v) or v < 40 or v > 70) else 'black')

    plt.tight_layout()
    path = os.path.join(out_dir, 'heatmap.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Plot 4: Training time per method ─────────────────────────────────────────

def plot_training_time(rows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, method in enumerate(METHODS):
        times = [r['train_seconds'] / 60 for r in rows if r['method'] == method]
        if not times:
            continue
        ax.scatter([i] * len(times), times, color=COLORS[method],
                   s=60, zorder=3, label=LABELS[method])
        ax.hlines(np.mean(times), i - 0.2, i + 0.2,
                  color=COLORS[method], linewidth=2)

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel('Training time (min)')
    ax.set_title('Training Time per (Puzzle × Difficulty) Combo', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'training_time.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = load_all()
    if not rows:
        print('No CSV files found in results/. Run the comparison jobs first.')
        return

    print(f'Loaded {len(rows)} results across {len(set(r["difficulty"] for r in rows))} difficulty levels.\n')

    # quick text summary
    for diff in DIFFICULTIES:
        d_rows = [r for r in rows if r['difficulty'] == diff]
        if not d_rows:
            continue
        print(f'Difficulty {diff}:')
        for method in METHODS:
            m_rows = [r for r in d_rows if r['method'] == method]
            if not m_rows:
                continue
            avg_solve = np.mean([r['solve_rate'] for r in m_rows]) * 100
            avg_reward = np.mean([r['mean_reward'] for r in m_rows])
            print(f'  {LABELS[method]:6s}  solve={avg_solve:.0f}%  reward={avg_reward:+.2f}  '
                  f'({len(m_rows)} puzzles)')
        print()

    print('Generating plots ...')
    plot_solve_rates(rows, RESULTS_DIR)
    plot_rewards(rows, RESULTS_DIR)
    plot_heatmap(rows, RESULTS_DIR)
    plot_training_time(rows, RESULTS_DIR)
    print('\nDone.')


if __name__ == '__main__':
    main()
