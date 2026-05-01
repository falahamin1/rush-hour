"""
Graph Neural Network actor-critic for Rush Hour.

Graph construction
------------------
Each vehicle contributes MAX_CONSTRAINTS (=4) constraint-nodes.  With
MAX_VEHICLES=16 we get 64 nodes total.  The adjacency matrix is
block-diagonal: edges only exist *within* a vehicle (between its constraint
half-spaces that share a vertex, i.e. the K_{2,2} bipartite structure for
axis-aligned rectangles).

Two GCN layers propagate messages, then global max-pooling aggregates all
64 nodes into a single latent vector fed to actor/critic heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rush_hour_env import MAX_VEHICLES, MAX_CONSTRAINTS

_TOTAL_NODES = MAX_VEHICLES * MAX_CONSTRAINTS  # 64


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x:   [B, N, in_features]
        # adj: [B, N, N]
        return F.relu(self.proj(torch.bmm(adj, x)))


class GNNActorCritic(nn.Module):
    def __init__(self, node_dim=3, hidden_dim=128, num_actions=32):
        super().__init__()
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
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

    def _build_big_adj(self, adj, device):
        """
        adj: [B, MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS]
        Returns block-diagonal big_adj [B, 64, 64] with self-loops added.
        """
        B = adj.shape[0]
        big_adj = torch.zeros(B, _TOTAL_NODES, _TOTAL_NODES, device=device)
        for i in range(MAX_VEHICLES):
            s = i * MAX_CONSTRAINTS
            e = s + MAX_CONSTRAINTS
            big_adj[:, s:e, s:e] = adj[:, i]
        big_adj += torch.eye(_TOTAL_NODES, device=device).unsqueeze(0)
        return big_adj

    def forward(self, h_rep, adj):
        """
        h_rep: [B, MAX_VEHICLES, MAX_CONSTRAINTS, 3]
        adj:   [B, MAX_VEHICLES, MAX_CONSTRAINTS, MAX_CONSTRAINTS]
        """
        B = h_rep.shape[0]
        device = h_rep.device

        x = h_rep.view(B, _TOTAL_NODES, 3)
        big_adj = self._build_big_adj(adj, device)

        h = self.gcn1(x, big_adj)
        h = self.gcn2(h, big_adj)

        pooled = torch.max(h, dim=1)[0]  # [B, hidden_dim]
        latent  = self.rho(pooled)

        return self.actor(latent), self.critic(latent)
