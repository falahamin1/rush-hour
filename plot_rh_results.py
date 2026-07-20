"""
Generate presentation figures for Rush Hour HPC results.

Loads all saved policies, averages across however many puzzles were trained
per difficulty, and produces two figures matching the tangram-figure style:

  results/rh_solve_rate_figure.pdf   — solve-rate bar chart (3 difficulty panels)
  results/rh_reward_figure.pdf       — mean-reward bar chart (3 difficulty panels)

Run:
  python3 plot_rh_results.py
"""

import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

POLICY_DIR  = os.path.join(os.path.dirname(__file__), 'policies')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

DIFFICULTIES = [10, 12, 15]
METHODS      = ['hrep', 'vrep', 'gnn', 'mlp', 'cnn']
LABELS       = ['H-rep', 'V-rep', 'GNN', 'Flat MLP', 'CNN']
COLORS       = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']

plt.rcParams.update({
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """
    Returns data[difficulty][method] = list of metric dicts (one per puzzle).
    """
    data = {D: {M: [] for M in METHODS} for D in DIFFICULTIES}
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, '*.pth'))):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        D = ckpt['difficulty']
        M = ckpt['method']
        if D in data and M in data[D]:
            data[D][M].append(ckpt['metrics'])
    return data


def summary(data):
    """
    Returns means[D][M], stds[D][M] for both solve_rate and mean_reward.
    """
    solve_mean, solve_std, reward_mean, reward_std = {}, {}, {}, {}
    for D in DIFFICULTIES:
        solve_mean[D], solve_std[D] = {}, {}
        reward_mean[D], reward_std[D] = {}, {}
        for M in METHODS:
            entries = data[D][M]
            solves  = [e['solve_rate']  for e in entries]
            rewards = [e['mean_reward'] for e in entries]
            solve_mean[D][M]  = np.mean(solves)  * 100 if solves  else 0
            solve_std[D][M]   = np.std(solves)   * 100 if solves  else 0
            reward_mean[D][M] = np.mean(rewards)        if rewards else 0
            reward_std[D][M]  = np.std(rewards)         if rewards else 0
    return solve_mean, solve_std, reward_mean, reward_std


# ── Shared panel builder ──────────────────────────────────────────────────────

def _n_label(D, data):
    """'n=15' if every method has the same puzzle count at this difficulty,
    else 'n=13-15' to flag that the sample sizes differ (e.g. a combo that
    hasn't finished training yet)."""
    counts = sorted(set(len(data[D][M]) for M in METHODS))
    if len(counts) == 1:
        return f'n={counts[0]}'
    return f'n={counts[0]}-{counts[-1]}'


def _bar_panel(ax, D, means, stds, ylabel, ylim, data, fmt_pct=False):
    x     = np.arange(len(METHODS))
    width = 0.55

    bars = ax.bar(
        x, [means[D][M] for M in METHODS],
        width,
        yerr=[stds[D][M] for M in METHODS],
        color=COLORS,
        alpha=0.88,
        capsize=5,
        error_kw=dict(elinewidth=1.4, capthick=1.4),
        zorder=3,
    )

    for bar, M in zip(bars, METHODS):
        val = means[D][M]
        err = stds[D][M]
        label = f'{val:.0f}%' if fmt_pct else f'{val:.2f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + ylim[1] * 0.02,
            label,
            ha='center', va='bottom', fontsize=9, fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Difficulty  {D}  moves  ({_n_label(D, data)})',
                 fontsize=12, fontweight='bold', pad=16)


# ── Figure 1: Solve rate ──────────────────────────────────────────────────────

def plot_solve_rates(solve_mean, solve_std, data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fig.suptitle(
        'Rush Hour — Solve Rate by Representation  (mean ± std across puzzles)',
        fontsize=13, fontweight='bold', y=1.02,
    )

    for ax, D in zip(axes, DIFFICULTIES):
        _bar_panel(ax, D, solve_mean, solve_std, data=data,
                   ylabel='Solve rate (%)' if D == 10 else '',
                   ylim=(-5, 130), fmt_pct=True)
        if D != 10:
            ax.set_ylabel('')

    # shared legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)]
    axes[1].legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, 'rh_solve_rate_figure.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Figure 2: Mean reward ─────────────────────────────────────────────────────

def plot_rewards(reward_mean, reward_std, data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fig.suptitle(
        'Rush Hour — Mean Eval Reward by Representation  (mean ± std across puzzles)',
        fontsize=13, fontweight='bold', y=1.02,
    )

    for ax, D in zip(axes, DIFFICULTIES):
        _bar_panel(ax, D, reward_mean, reward_std, data=data,
                   ylabel='Mean eval reward' if D == 10 else '',
                   ylim=(-6, 14), fmt_pct=False)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)
        if D != 10:
            ax.set_ylabel('')

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)]
    axes[1].legend(handles=handles, loc='upper center',
                   bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, 'rh_reward_figure.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'Loading policies from {POLICY_DIR}/ ...')
    data = load_data()

    total = sum(len(data[D][M]) for D in DIFFICULTIES for M in METHODS)
    if total == 0:
        print('No policy files found. Copy policies/ from the cluster first.')
        return
    print(f'  {total} policy files loaded.\n')

    solve_mean, solve_std, reward_mean, reward_std = summary(data)

    print('Results summary (averaged across puzzles):')
    for D in DIFFICULTIES:
        print(f'  Difficulty {D}:')
        for M, L in zip(METHODS, LABELS):
            print(f'    {L:6s}  solve={solve_mean[D][M]:.0f}% ± {solve_std[D][M]:.0f}%  '
                  f'reward={reward_mean[D][M]:+.2f} ± {reward_std[D][M]:.2f}')
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('Generating figures ...')
    plot_solve_rates(solve_mean, solve_std, data, RESULTS_DIR)
    plot_rewards(reward_mean, reward_std, data, RESULTS_DIR)
    print('Done.')


if __name__ == '__main__':
    main()
