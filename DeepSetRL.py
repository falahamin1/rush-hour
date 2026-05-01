"""
DeepSet actor-critic for Rush Hour.

The set is the collection of vehicles on the board.  Each vehicle is
encoded independently by phi, then the per-vehicle representations are
summed (permutation-invariant aggregation) before the policy/value heads.

Works for both H-rep (input_dim=12: 4 constraints × 3 params) and
V-rep (input_dim=8: 4 corners × 2 coords).
"""
import torch
import torch.nn as nn


class DeepSetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [Batch, MAX_VEHICLES, input_dim]  (already flat per vehicle)
        # Also accepts [Batch, MAX_VEHICLES, MAX_CONSTRAINTS, params] and flattens.
        if x.dim() == 4:
            B, V, C, P = x.shape
            x = x.view(B, V, C * P)

        B, V, D = x.shape
        local_feats = self.phi(x.reshape(-1, D))
        return local_feats.view(B, V, -1)  # [B, V, hidden_dim]


class DeepSetActorCritic(nn.Module):
    def __init__(self, input_dim, num_pieces, num_actions, hidden_dim=128):
        super().__init__()
        self.encoder = DeepSetEncoder(input_dim, hidden_dim)

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        local_feats = self.encoder(x)                # [B, V, H]
        global_sum  = torch.sum(local_feats, dim=1)  # [B, H]
        latent      = self.rho(global_sum)
        return self.actor(latent), self.critic(latent)
