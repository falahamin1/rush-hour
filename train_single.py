"""
Train a single (difficulty, puzzle_idx, method) combo and save the best policy.

Saved to:  policies/d{difficulty}_p{puzzle_idx}_{method}.pth

Usage:
  python train_single.py --difficulty 10 --puzzle-idx 0 --method hrep --episodes 750
  python train_single.py --difficulty 12 --puzzle-idx 1 --method gnn  --episodes 1200
"""

import os
import sys
import argparse
import time
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
POLICY_DIR  = os.path.join(os.path.dirname(__file__), 'policies')


def load_puzzle_by_index(difficulty, idx):
    found = []
    with open(PUZZLE_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[0]) == difficulty:
                found.append(parts[1])
                if len(found) == idx + 1:
                    return parts[1]
    raise ValueError(
        f"Not enough puzzles at difficulty={difficulty}: "
        f"found {len(found)}, want index {idx}"
    )


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
        else:
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(h, adj)
        logits[0][~mask] = -1e10
        return torch.argmax(logits, dim=-1).item()


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


def main():
    ap = argparse.ArgumentParser(description='Train one Rush Hour combo on HPC')
    ap.add_argument('--difficulty',    type=int, required=True,
                    help='Exact number of moves (10, 12, or 15)')
    ap.add_argument('--puzzle-idx',    type=int, required=True,
                    help='0-based index of the puzzle within that difficulty')
    ap.add_argument('--method',        choices=['hrep', 'vrep', 'gnn'], required=True)
    ap.add_argument('--episodes',      type=int, required=True)
    ap.add_argument('--eval-episodes', type=int, default=50)
    args = ap.parse_args()

    board_str = load_puzzle_by_index(args.difficulty, args.puzzle_idx)

    print(f"[train_single] difficulty={args.difficulty}  puzzle_idx={args.puzzle_idx}  "
          f"method={args.method}  episodes={args.episodes}")
    print(f"[train_single] board={board_str}")

    TRAIN = {
        'hrep': train_h_rep,
        'vrep': train_v_rep,
        'gnn':  train_graph_rep,
    }

    t0 = time.time()
    _, best_model = TRAIN[args.method](board_str=board_str, episodes=args.episodes)
    elapsed = time.time() - t0

    print(f"\n[train_single] Training done in {elapsed/60:.1f} min. Evaluating ...")
    metrics = evaluate(best_model, args.method, board_str,
                       eval_episodes=args.eval_episodes)

    print(f"[train_single] solve_rate={metrics['solve_rate']*100:.1f}%  "
          f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}")

    os.makedirs(POLICY_DIR, exist_ok=True)
    out_path = os.path.join(
        POLICY_DIR,
        f'd{args.difficulty}_p{args.puzzle_idx}_{args.method}.pth'
    )
    torch.save({
        'method':         args.method,
        'difficulty':     args.difficulty,
        'puzzle_idx':     args.puzzle_idx,
        'board_str':      board_str,
        'model_state':    best_model.state_dict(),
        'metrics':        metrics,
        'train_episodes': args.episodes,
        'train_seconds':  elapsed,
    }, out_path)
    print(f"[train_single] Policy saved → {out_path}")


if __name__ == '__main__':
    main()
