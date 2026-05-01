"""
Train and compare all three PPO methods on the same Rush Hour puzzle:
  1. H-rep  DeepSet
  2. V-rep  DeepSet
  3. Graph  NN (GCN)

Run: python Rush-hour/comparison.py
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS, NUM_ACTIONS
from run_hrep import train_h_rep
from run_vrep import train_v_rep
from run_gnn import train_graph_rep

PUZZLE_FILE = os.path.join(os.path.dirname(__file__), 'rush.txt')
DEFAULT_BOARD = 'IBBxooIooLDDJAALooJoKEEMFFKooMGGHHHM'
H_DIM = MAX_CONSTRAINTS * 3  # 12
V_DIM = 4 * 2               # 8


# ── Puzzle loading ──────────────────────────────────────────────────────────

def _load_puzzle(max_moves=10, index=0):
    """Return the puzzle at position `index` (0-based) in rush.txt that requires <= max_moves moves."""
    try:
        with open(PUZZLE_FILE) as f:
            count = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[0]) <= max_moves:
                    if count == index:
                        print(f"Loaded puzzle #{index} requiring {parts[0]} moves: {parts[1]}")
                        return parts[1]
                    count += 1
    except Exception:
        pass
    return DEFAULT_BOARD


# ── Action selection helpers ────────────────────────────────────────────────

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
        else:  # gnn
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(h, adj)
        logits[0][~mask] = -1e10
        return torch.argmax(logits, dim=-1).item()


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, method, board_str, episodes=10, max_steps=200):
    device = next(model.parameters()).device
    model.eval()
    rewards, solves = [], 0

    for _ in range(episodes):
        env = RushHourGym(board_str)
        obs, _ = env.reset()
        total = 0
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
        'mean_reward': np.mean(rewards),
        'std_reward':  np.std(rewards),
        'solve_rate':  solves / episodes,
    }


# ── Step-by-step replay ─────────────────────────────────────────────────────

def replay(model, method, board_str, tag, max_steps=200):
    """
    Run one greedy rollout and save a PNG frame for every step.
    Frames are written to Rush-hour/replays/<tag>/ as step_000.png … step_NNN.png.
    """
    folder = os.path.join(os.path.dirname(__file__), 'replays', tag)
    os.makedirs(folder, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    env = RushHourGym(board_str)
    obs, _ = env.reset()

    # Save initial board
    env.render(mode='rgb_array',
               save_path=os.path.join(folder, 'step_000.png'))
    print(f'\n[{tag}] Replay frames → {folder}/')
    print(f'  step 000: initial board')

    total = 0
    solved = False

    for step in range(1, max_steps + 1):
        mask = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)
        action = _get_action(model, method, obs, mask, device)

        vi  = action // 2
        dir_label = 'right/down' if action % 2 == 0 else 'left/up'
        vid = env.inner.vehicles[vi].vid if vi < len(env.inner.vehicles) else '?'

        obs, r, done, _, info = env.step(action)
        total += r

        frame_path = os.path.join(folder, f'step_{step:03d}.png')
        env.render(mode='rgb_array', save_path=frame_path)

        print(f'  step {step:03d}: vehicle {vid} {dir_label}'
              f'  reward={r:+.2f}  cumulative={total:+.2f}')

        if done:
            solved = info.get('solved', False)
            status = 'SOLVED' if solved else 'TIMEOUT'
            print(f'  [{tag}] {status} in {step} steps  total_reward={total:.2f}')
            break

    if not solved and step == max_steps:
        print(f'  [{tag}] Not solved within {max_steps} steps  total_reward={total:.2f}')

    return total, solved


# ── Results plot ────────────────────────────────────────────────────────────

def _plot_results(results):
    names  = list(results.keys())
    means  = [results[n]['mean_reward'] for n in names]
    stds   = [results[n]['std_reward']  for n in names]
    solves = [results[n]['solve_rate']  for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Rush Hour — Method Comparison', fontsize=13)

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.bar(names, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax1.set_ylabel('Mean eval reward')
    ax1.set_title('Reward')

    ax2.bar(names, [s * 100 for s in solves], color=colors, alpha=0.85)
    ax2.set_ylabel('Solve rate (%)')
    ax2.set_ylim(0, 105)
    ax2.set_title('Solve Rate')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'rh_comparison.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def run_comparison(episodes=500, eval_episodes=20, max_moves=10, puzzle_index=0):
    board = _load_puzzle(max_moves=max_moves, index=puzzle_index)
    print(f"\nPuzzle: {board}\n{'='*60}")

    print("\n[1/3] Training H-rep DeepSet ...")
    _, best_h = train_h_rep(board_str=board, episodes=episodes)

    print("\n[2/3] Training V-rep DeepSet ...")
    _, best_v = train_v_rep(board_str=board, episodes=episodes)

    print("\n[3/3] Training Graph NN ...")
    _, best_g = train_graph_rep(board_str=board, episodes=episodes)

    device = torch.device("cpu")
    best_h.to(device); best_v.to(device); best_g.to(device)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    results = {}
    configs = [
        ("H-rep DeepSet", best_h, "hrep"),
        ("V-rep DeepSet", best_v, "vrep"),
        ("Graph NN",      best_g, "gnn"),
    ]
    for name, model, method in configs:
        r = evaluate(model, method, board, episodes=eval_episodes)
        results[name] = r
        print(f"  {name:20s}  reward={r['mean_reward']:+.2f}±{r['std_reward']:.2f}"
              f"  solve={r['solve_rate']*100:.0f}%")

    _plot_results(results)

    # ── Step-by-step replay for each method ──────────────────────────────────
    print("\n" + "="*60)
    print("STEP-BY-STEP REPLAYS")
    print("="*60)
    for name, model, method in configs:
        replay(model, method, board, tag=method)

    return results


if __name__ == '__main__':
    start_time = time.time()
    run_comparison(episodes=500, eval_episodes=40, max_moves=10, puzzle_index=1)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
