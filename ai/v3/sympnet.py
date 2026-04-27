"""
LASympNet: Linear-Activation SympNet with Guaranteed Symplecticity

Ported from GeometricMachineLearning.jl (JuliaGNI).
Each layer is a symplectic map by construction; composition is symplectic.
"""

import torch
import torch.nn as nn
from typing import Optional


class LinearLayerQ(nn.Module):
    """Linear layer updating only q: (q, p) -> (q + W_q p + b_q, p)."""
    def __init__(self, dim: int):
        super().__init__()
        self.half = dim // 2
        self.W = nn.Linear(self.half, self.half, bias=True)
    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        return torch.cat([q + self.W(p), p], dim=-1)


class LinearLayerP(nn.Module):
    """Linear layer updating only p: (q, p) -> (q, p + W_p q + b_p)."""
    def __init__(self, dim: int):
        super().__init__()
        self.half = dim // 2
        self.W = nn.Linear(self.half, self.half, bias=True)
    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        return torch.cat([q, p + self.W(q)], dim=-1)


class ActivationLayerQ(nn.Module):
    """Activation applied only to q: (q, p) -> (sigma(q), p)."""
    def __init__(self, dim: int, activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        return torch.cat([self.activation(q), p], dim=-1)


class ActivationLayerP(nn.Module):
    """Activation applied only to p: (q, p) -> (q, sigma(p))."""
    def __init__(self, dim: int, activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        return torch.cat([q, self.activation(p)], dim=-1)


class BiasLayer(nn.Module):
    """Translation: (q, p) -> (q + b_q, p + b_p). Symplectic map."""
    def __init__(self, dim: int):
        super().__init__()
        self.half = dim // 2
        self.b_q = nn.Parameter(torch.zeros(self.half))
        self.b_p = nn.Parameter(torch.zeros(self.half))
    def forward(self, x):
        q, p = x.chunk(2, dim=-1)
        return torch.cat([q + self.b_q, p + self.b_p], dim=-1)


class LASympNet(nn.Module):
    """
    LA-SympNet following Jin et al. (2020) and JuliaGNI implementation.

    Hidden block pattern (repeated nhidden times):
        [LinearLayerQ, LinearLayerP] x depth -> BiasLayer -> ActivationLayerQ -> ActivationLayerP
    Output block (no activation):
        [LinearLayerQ, LinearLayerP] x depth -> BiasLayer
    """
    def __init__(self, dim=6, depth=5, nhidden=1, activation=nn.Tanh(),
                 init_upper_linear=True, init_upper_act=True):
        super().__init__()
        depth = min(depth, 5)
        layers = []
        for _ in range(nhidden):
            for j in range(1, depth + 1):
                is_q = (j % 2 == 1) if init_upper_linear else (j % 2 == 0)
                layers.append(LinearLayerQ(dim) if is_q else LinearLayerP(dim))
            layers.append(BiasLayer(dim))
            if init_upper_act:
                layers.append(ActivationLayerQ(dim, activation))
                layers.append(ActivationLayerP(dim, activation))
            else:
                layers.append(ActivationLayerP(dim, activation))
                layers.append(ActivationLayerQ(dim, activation))
        for j in range(1, depth + 1):
            is_q = (j % 2 == 1) if init_upper_linear else (j % 2 == 0)
            layers.append(LinearLayerQ(dim) if is_q else LinearLayerP(dim))
        layers.append(BiasLayer(dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, dt: Optional[float] = None):
        for layer in self.layers:
            x = layer(x)
        return x

    def integrate(self, initial_state, dt: float, n_steps: int):
        trajectory = [initial_state]
        state = initial_state
        for _ in range(n_steps):
            state = self.forward(state, dt)
            trajectory.append(state)
        return torch.stack(trajectory, dim=1)


class GSympNet(nn.Module):
    """
    G-SympNet with GradientLayers (upscaling symplectic maps).
    More expressive than LA-SympNet; uses upscaled intermediate dimensions.
    """
    def __init__(self, dim=6, n_layers=2, upscaling_dim=None, activation=nn.Tanh(), init_upper=True):
        super().__init__()
        self.dim = dim
        self.up_dim = upscaling_dim or 2 * dim
        self.n_layers = n_layers
        layers = []
        is_upper = init_upper
        for _ in range(n_layers):
            layers.append(LinearLayerQ(self.up_dim) if is_upper else LinearLayerP(self.up_dim))
            layers.append(ActivationLayerQ(self.up_dim, activation) if is_upper else ActivationLayerP(self.up_dim, activation))
            is_upper = not is_upper
        self.layers = nn.ModuleList(layers)
        self.project_up = nn.Linear(dim, self.up_dim, bias=False)
        self.project_down = nn.Linear(self.up_dim, dim, bias=False)

    def forward(self, x, dt=None):
        h = self.project_up(x)
        for layer in self.layers:
            h = layer(h)
        return self.project_down(h)

    def integrate(self, initial_state, dt: float, n_steps: int):
        trajectory = [initial_state]
        state = initial_state
        for _ in range(n_steps):
            state = self.forward(state, dt)
            trajectory.append(state)
        return torch.stack(trajectory, dim=1)
