"""
Lightweight checkpoint helpers shared by all run_*.py training scripts.

A checkpoint captures everything needed to resume PPO training exactly
where it left off: model + optimizer state, the running best-policy
weights/score, and the reward history used for the moving average.
Writes are atomic (write to a .tmp file, then os.replace) so a job killed
mid-save never leaves a corrupt checkpoint behind.
"""
import os
import torch


def save_checkpoint(path, *, model, optimizer, episode, best_weights,
                     best_moving_avg, reward_history):
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    torch.save({
        'episode':         episode,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_weights':    best_weights,
        'best_moving_avg': best_moving_avg,
        'reward_history':  reward_history,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    """Return the checkpoint dict, or None if there's nothing to resume from."""
    if path is None or not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=False)
