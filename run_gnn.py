"""
PPO training using Graph Neural Network (GNN) for Rush Hour.

Graph: 64 nodes (16 vehicles × 4 constraints each).
Node features: 3-dim half-space parameters (a1, a2, b/GRID).
Adjacency: block-diagonal — within each vehicle, constraints that share
a vertex are connected (K_{2,2} for axis-aligned rectangles).
"""
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS, NUM_ACTIONS
from GraphNNRL import GNNActorCritic
from PPOBuffer import PPOBuffer

PUZZLE_FILE = os.path.join(os.path.dirname(__file__), 'rush.txt')
DEFAULT_BOARD = 'IBBxooIooLDDJAALooJoKEEMFFKooMGGHHHM'


def _load_puzzle(max_moves=10):
    """Return the first puzzle in rush.txt that requires <= max_moves moves."""
    try:
        with open(PUZZLE_FILE) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[0]) <= max_moves:
                    print(f"[GNN] Loaded puzzle requiring {parts[0]} moves.")
                    return parts[1]
    except Exception:
        pass
    return DEFAULT_BOARD


def train_graph_rep(board_str=None, episodes=1000):
    if board_str is None:
        board_str = _load_puzzle()

    HP = {
        "lr": 1e-4,
        "clip_eps": 0.2,
        "ppo_epochs": 5,
        "steps_per_rollout": 4096,
        "batch_size": 64,
        "gamma": 0.99,
        "entropy_coef": 0.05,
        "critic_coef": 0.5,
        "max_grad_norm": 0.5,
        "moving_avg_window": 50,
    }

    device = torch.device("cpu")
    print(f"[GNN] device={device}  board={board_str}")

    env = RushHourGym(board_str)
    model = GNNActorCritic(
        node_dim=3,
        hidden_dim=128,
        num_actions=NUM_ACTIONS,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HP["lr"])

    # Two separate buffers — one for h_rep node features, one for adj matrices
    h_buf   = PPOBuffer(size=HP["steps_per_rollout"],
                        state_shape=(MAX_VEHICLES, MAX_CONSTRAINTS, 3))
    adj_buf = PPOBuffer(size=HP["steps_per_rollout"],
                        state_shape=(MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS))

    reward_history = []
    best_moving_avg = -float('inf')
    best_weights = deepcopy(model.state_dict())

    obs, _ = env.reset()
    ep_reward = 0

    for ep in range(episodes):
        # ── rollout ──────────────────────────────────────────────────────────
        for t in range(HP["steps_per_rollout"]):
            h_ts  = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj_ts = torch.tensor(obs['adj'],  dtype=torch.float32).unsqueeze(0).to(device)

            mask_ts = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)

            with torch.no_grad():
                logits, value = model(h_ts, adj_ts)
                logits[0][~mask_ts] = -1e10
                dist   = Categorical(logits=logits)
                action = dist.sample()
                lp     = dist.log_prob(action)

            obs, reward, done, _, _ = env.step(action.item())

            h_buf.store(h_ts.squeeze(0).cpu(), action, reward, value.item(), lp.item())
            adj_buf.store(adj_ts.squeeze(0).cpu(), action, 0, 0, 0)
            ep_reward += reward

            if done:
                h_buf.finish_path(last_val=0)
                reward_history.append(ep_reward)
                obs, _ = env.reset()
                ep_reward = 0
            elif t == HP["steps_per_rollout"] - 1:
                h_nxt   = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
                adj_nxt = torch.tensor(obs['adj'],  dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, last_val = model(h_nxt, adj_nxt)
                h_buf.finish_path(last_val.item())

        # ── best model tracking ───────────────────────────────────────────────
        if len(reward_history) >= HP["moving_avg_window"]:
            avg = np.mean(reward_history[-HP["moving_avg_window"]:])
            if avg > best_moving_avg:
                best_moving_avg = avg
                best_weights = deepcopy(model.state_dict())
                print(f"  *** NEW BEST GNN (avg={best_moving_avg:.2f}) ep={ep} ***")

        # ── PPO update ────────────────────────────────────────────────────────
        data_h   = h_buf.get()
        data_adj = adj_buf.get()
        idx = np.arange(HP["steps_per_rollout"])

        for _ in range(HP["ppo_epochs"]):
            np.random.shuffle(idx)
            for s in range(0, HP["steps_per_rollout"], HP["batch_size"]):
                mb = idx[s:s + HP["batch_size"]]
                mb_h   = data_h['states'][mb].to(device)
                mb_adj = data_adj['states'][mb].to(device)
                mb_a   = data_h['actions'][mb].to(device)
                mb_adv = data_h['advantages'][mb].to(device)
                mb_ret = data_h['returns'][mb].to(device)
                mb_lp  = data_h['log_probs'][mb].to(device)

                logits, values = model(mb_h, mb_adj)
                dist = Categorical(logits=logits)
                new_lp  = dist.log_prob(mb_a)
                entropy = dist.entropy().mean()

                ratio  = torch.exp(new_lp - mb_lp)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1 - HP["clip_eps"], 1 + HP["clip_eps"]) * mb_adv
                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = F.mse_loss(values.squeeze(-1), mb_ret)
                loss   = a_loss + HP["critic_coef"] * c_loss - HP["entropy_coef"] * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP["max_grad_norm"])
                optimizer.step()

        h_buf.clear()
        adj_buf.clear()

        if ep % 100 == 0:
            recent = np.mean(reward_history[-10:]) if reward_history else 0
            print(f"[GNN]   ep={ep:5d}  recent={recent:.2f}  best={best_moving_avg:.2f}")

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'rh_gnn_final.pth'))

    best_model = GNNActorCritic(node_dim=3, hidden_dim=128, num_actions=NUM_ACTIONS)
    best_model.load_state_dict(best_weights)
    return model, best_model


if __name__ == '__main__':
    train_graph_rep(episodes=3000)
