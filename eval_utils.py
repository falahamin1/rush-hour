"""
Shared dual-mode (greedy / stochastic) policy evaluation for Rush Hour.

Used both for the one-off final evaluation in train_single.py (schema output)
and for the periodic curve snapshots taken during training by each
run_*.py. Greedy always takes the highest-probability action (deterministic
given this benchmark's fixed board per run). Stochastic samples from the
policy's own softmax distribution using a dedicated numpy Generator, kept
separate from torch's global RNG so evaluation never perturbs the action-
sampling/network-init stream that --seed (in train_single.py) makes
reproducible.
"""
import torch
import numpy as np

from rush_hour_env import RushHourGym, MAX_VEHICLES, MAX_CONSTRAINTS

H_DIM = MAX_CONSTRAINTS * 3
V_DIM = 4 * 2


def get_action(model, method, obs, mask, device, mode, rng):
    with torch.no_grad():
        if method == 'hrep':
            s = torch.tensor(obs['h_rep'], dtype=torch.float32).view(MAX_VEHICLES, H_DIM).unsqueeze(0).to(device)
            logits, _ = model(s)
        elif method == 'vrep':
            s = torch.tensor(obs['v_rep'], dtype=torch.float32).view(MAX_VEHICLES, V_DIM).unsqueeze(0).to(device)
            logits, _ = model(s)
        elif method == 'gnn':
            h = torch.tensor(obs['h_rep'], dtype=torch.float32).unsqueeze(0).to(device)
            adj = torch.tensor(obs['adj'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(h, adj)
        elif method == 'mlp':
            s = torch.tensor(obs['flat_pose'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(s)
        else:  # cnn
            s = torch.tensor(obs['grid_image'], dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = model(s)
        logits[0][~mask] = -1e10
        if mode == 'greedy':
            return torch.argmax(logits, dim=-1).item()
        if mode != 'stochastic':
            raise ValueError(f"mode must be 'greedy' or 'stochastic', got {mode!r}")
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        probs = probs / probs.sum()  # guard against fp drift after masking
        return int(rng.choice(len(probs), p=probs))


def evaluate(model, method, board_str, mode, episodes, max_steps=200, seed=0):
    """Run `episodes` resets of the same fixed board_str under `mode`.

    Returns a list of {"solved": bool, "steps": int}, one per episode/reset.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(episodes):
        env = RushHourGym(board_str)
        obs, _ = env.reset()
        solved, steps_taken = False, 0
        for _ in range(max_steps):
            mask = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)
            action = get_action(model, method, obs, mask, device, mode, rng)
            obs, _reward, done, _, info = env.step(action)
            steps_taken += 1
            if done:
                solved = bool(info.get('solved', False))
                break
        outcomes.append({"solved": solved, "steps": steps_taken})
    if was_training:
        model.train()
    return outcomes
