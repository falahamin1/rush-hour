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
import random
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS, NUM_ACTIONS
from run_hrep import train_h_rep
from run_vrep import train_v_rep
from run_gnn  import train_graph_rep
from run_mlp  import train_flat_mlp
from run_cnn  import train_cnn_rep
import eval_utils
import results_schema

H_DIM = MAX_CONSTRAINTS * 3   # 12
V_DIM = 4 * 2                 # 8

PUZZLE_FILE  = os.path.join(os.path.dirname(__file__), 'rush.txt')
POLICY_DIR   = os.path.join(os.path.dirname(__file__), 'policies')
CKPT_DIR     = os.path.join(os.path.dirname(__file__), 'checkpoints')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results_schema_out')


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
        elif method == 'gnn':
            h   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(obs['adj'],   dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(h, adj)
        elif method == 'mlp':
            s = torch.tensor(obs['flat_pose'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(s)
        else:   # cnn
            s = torch.tensor(obs['grid_image'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(s)
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


def _reduce_to_instance(instance_id, greedy_outcomes, stoch_outcomes, optimal_steps):
    """Collapse N resets of one fixed puzzle into the unified schema's single
    per-instance record (this benchmark trains -- and therefore evaluates --
    exactly one board per run, so per_instance always has length 1).

    solved_{mode} is a majority vote across resets: for greedy this is moot,
    since the policy and environment are both deterministic here, so every
    greedy reset agrees. For stochastic, resets can disagree, so majority vote
    is the reduction to a single boolean. `steps` reports the greedy rollout's
    step count when greedy solved it (a single well-defined number, since all
    greedy resets are identical); otherwise the median step count among the
    stochastic resets that solved it; otherwise None (undefined on an instance
    neither mode solved).
    """
    g_rate = sum(o['solved'] for o in greedy_outcomes) / len(greedy_outcomes)
    s_rate = sum(o['solved'] for o in stoch_outcomes) / len(stoch_outcomes)
    solved_greedy = g_rate >= 0.5
    solved_stochastic = s_rate >= 0.5
    if solved_greedy:
        steps = next(o['steps'] for o in greedy_outcomes if o['solved'])
    elif solved_stochastic:
        solved_steps = sorted(o['steps'] for o in stoch_outcomes if o['solved'])
        steps = solved_steps[len(solved_steps) // 2]
    else:
        steps = None
    return {
        "instance_id": instance_id,
        "solved_greedy": solved_greedy,
        "solved_stochastic": solved_stochastic,
        "steps": steps,
        "optimal_steps": optimal_steps,
    }


def main():
    ap = argparse.ArgumentParser(description='Train one Rush Hour combo on HPC')
    ap.add_argument('--difficulty',    type=int, required=True,
                    help='Exact number of moves (10, 12, or 15)')
    ap.add_argument('--puzzle-idx',    type=int, required=True,
                    help='0-based index of the puzzle within that difficulty')
    ap.add_argument('--method',        choices=['hrep', 'vrep', 'gnn', 'mlp', 'cnn'], required=True)
    ap.add_argument('--episodes',      type=int, required=True)
    ap.add_argument('--eval-episodes', type=int, default=50,
                    help='Resets used for the final dual-mode evaluation recorded in the result file')
    ap.add_argument('--seed', type=int, default=0,
                    help='Seeds network init, action sampling during training, minibatch shuffle '
                         'order, and stochastic-evaluation sampling')
    ap.add_argument('--results-dir', default=RESULTS_DIR,
                    help='Output directory for the unified cross-benchmark result JSON')
    ap.add_argument('--policy-dir', default=POLICY_DIR,
                    help='Output directory for the .pth policy file')
    ap.add_argument('--checkpoint-dir', default=CKPT_DIR,
                    help='Directory for the resumable PPO checkpoint')
    ap.add_argument('--curve-eval-episodes', type=int, default=10,
                    help='Resets used for the periodic (every-25-episode) dual-mode curve snapshot; '
                         'kept small since this eval runs many times over the course of training')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = os.path.join(
        args.policy_dir,
        f'd{args.difficulty}_p{args.puzzle_idx}_{args.method}.pth'
    )
    if os.path.exists(out_path):
        print(f"[train_single] {out_path} already exists — skipping.")
        return

    board_str = load_puzzle_by_index(args.difficulty, args.puzzle_idx)

    print(f"[train_single] difficulty={args.difficulty}  puzzle_idx={args.puzzle_idx}  "
          f"method={args.method}  episodes={args.episodes}  seed={args.seed}")
    print(f"[train_single] board={board_str}")

    TRAIN = {
        'hrep': train_h_rep,
        'vrep': train_v_rep,
        'gnn':  train_graph_rep,
        'mlp':  train_flat_mlp,
        'cnn':  train_cnn_rep,
    }

    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f'd{args.difficulty}_p{args.puzzle_idx}_{args.method}.pt'
    )

    tier = f"{args.difficulty}_moves"
    instance_id = f"d{args.difficulty}_p{args.puzzle_idx}"
    curve = []

    def _write_snapshot(per_instance, hp):
        config = dict(hp)
        config.update({
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "curve_eval_episodes": args.curve_eval_episodes,
            "difficulty": args.difficulty,
            "puzzle_idx": args.puzzle_idx,
            "board_str": board_str,
            "seed": args.seed,
        })
        result = results_schema.build_result(
            benchmark="rush_hour", tier=tier, encoder=args.method, seed=args.seed,
            config=config, curve=list(curve), per_instance=per_instance,
        )
        return results_schema.write_result(result, args.results_dir)

    def on_checkpoint(ep, entropy_value, model, hp):
        # Runs at the same cadence as the PPO checkpoint save (every 25
        # episodes, plus the final episode). Result file is fully rewritten
        # every time, so a job killed between checkpoints still leaves behind
        # the most recently completed checkpoint's curve + per-instance data.
        greedy = eval_utils.evaluate(model, args.method, board_str, mode='greedy',
                                      episodes=args.curve_eval_episodes, seed=args.seed)
        stoch = eval_utils.evaluate(model, args.method, board_str, mode='stochastic',
                                     episodes=args.curve_eval_episodes, seed=args.seed)
        g_rate = sum(o['solved'] for o in greedy) / len(greedy)
        s_rate = sum(o['solved'] for o in stoch) / len(stoch)
        curve.append({"iter": ep, "entropy": entropy_value,
                       "solve_greedy": g_rate, "solve_stochastic": s_rate})
        per_instance = [_reduce_to_instance(instance_id, greedy, stoch, args.difficulty)]
        _write_snapshot(per_instance, hp)

    t0 = time.time()
    _, best_model, HP = TRAIN[args.method](board_str=board_str, episodes=args.episodes,
                                            checkpoint_path=ckpt_path, on_checkpoint=on_checkpoint)
    elapsed = time.time() - t0

    print(f"\n[train_single] Training done in {elapsed/60:.1f} min. Evaluating ...")
    metrics = evaluate(best_model, args.method, board_str,
                       eval_episodes=args.eval_episodes)

    print(f"[train_single] solve_rate={metrics['solve_rate']*100:.1f}%  "
          f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}")

    os.makedirs(args.policy_dir, exist_ok=True)
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

    # Final dual-mode evaluation for the unified result schema, on the same
    # best_model whose .pth was just saved above (not the final-episode model,
    # which may have drifted from the tracked best).
    final_greedy = eval_utils.evaluate(best_model, args.method, board_str, mode='greedy',
                                        episodes=args.eval_episodes, seed=args.seed)
    final_stoch = eval_utils.evaluate(best_model, args.method, board_str, mode='stochastic',
                                       episodes=args.eval_episodes, seed=args.seed)
    per_instance = [_reduce_to_instance(instance_id, final_greedy, final_stoch, args.difficulty)]
    result_path = _write_snapshot(per_instance, HP)
    print(f"[train_single] Unified result JSON → {result_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == '__main__':
    main()
