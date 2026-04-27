"""
Riemannian Adam Optimizer on Stiefel Manifold

Ported from GeometricOptimizers.jl (JuliaGNI).

For each weight matrix W treated as a point on Stiefel manifold St(n,N) = {W: W^T W = I},
the optimizer:
    1. Computes Euclidean gradient ∇L via autograd
    2. Projects to Riemannian gradient: ∇R = ∇L - W (∇L^T W)
    3. Applies Adam momentum (first/second moments) on tangent space
    4. Retracts back to manifold via QR decomposition: W_new = qr(W + step).Q

References:
    - Hairer, Lubich, Wanner (2006): Geometric Numerical Integration
    - Absil, Mahony, Sepulchre (2008): Optimization Algorithms on Matrix Manifolds
    - Bendokat et al. (2020): Grassmann manifold optimization
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class StiefelAdam(torch.optim.Optimizer):
    """
    Adam optimizer with Stiefel manifold constraint on specified parameter groups.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default 1e-3)
        betas: (beta1, beta2) coefficients for running averages (default (0.9, 0.999))
        eps: term for numerical stability (default 1e-8)
        stiefel_params: list of parameter names (or nn.Parameter refs) that should
                        be constrained to Stiefel manifold. If None, all 2D parameters
                        with n_rows >= n_cols are treated as Stiefel.
        weight_decay: L2 penalty (applied only to non-Stiefel params, default 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 stiefel_params: Optional[List[str]] = None, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.stiefel_names = set(stiefel_params or [])
        self._init_state()

    def _init_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment

    def _is_stiefel(self, param, name: str) -> bool:
        """Check if parameter should be treated as Stiefel point."""
        if self.stiefel_names and name in self.stiefel_names:
            return True
        # Auto-detect: 2D matrices with n_rows >= n_cols (orthonormal columns)
        if param.dim() == 2:
            rows, cols = param.shape
            return rows >= cols and cols > 1
        return False

    def _riemannian_gradient(self, W: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """
        Project Euclidean gradient to Riemannian gradient on Stiefel manifold.
        Formula: rgrad(W, ∇L) = ∇L - W (∇L^T W)
        """
        # ∇L^T W is [cols, cols]
        gram = grad.T @ W
        return grad - W @ gram

    def _qr_retraction(self, W: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
        Retract W + step back to Stiefel manifold via QR decomposition.
        W_new = qr(W + step).Q  (thin QR, keeping orthonormal columns)
        """
        Y = W + step
        # Thin QR: Y = Q R where Q is [n, k], R is [k, k]
        Q, R = torch.linalg.qr(Y, mode='reduced')
        # Ensure determinantal consistency (flip signs of columns where R diag is negative)
        # This keeps the retraction on the same connected component
        diag_r = torch.diagonal(R, 0)
        signs = torch.sign(diag_r)
        signs[signs == 0] = 1.0
        Q = Q @ torch.diag_embed(signs)
        return Q

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                state['step'] += 1
                step_t = state['step']

                # Weight decay (only for non-Stiefel params)
                if weight_decay != 0 and not self._is_stiefel(p, ''):
                    grad = grad.add(p.data, alpha=weight_decay)

                # Riemannian gradient for Stiefel params
                if self._is_stiefel(p, ''):
                    grad = self._riemannian_gradient(p.data, grad)

                # Adam moment updates (operating on tangent space for Stiefel)
                m = state['m']
                v = state['v']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                # Compute update step
                update = m / denom
                step = -step_size * update

                # Retraction for Stiefel, standard update otherwise
                if self._is_stiefel(p, ''):
                    p.data.copy_(self._qr_retraction(p.data, step))
                else:
                    p.data.add_(step)

        return loss


def apply_stiefel_to_module(module: nn.Module, optimizer: StiefelAdam,
                            pattern: Optional[List[str]] = None):
    """
    Utility to tag specific modules/layers as Stiefel-constrained.
    By default, all nn.Linear weight matrices are treated as Stiefel.
    """
    stiefel_names = []
    for name, param in module.named_parameters():
        if param.dim() == 2:
            if pattern is None or any(p in name for p in pattern):
                stiefel_names.append(name)
    # Rebuild optimizer with named params (requires re-initialization)
    # This is a convenience hint; in practice set stiefel_params at construction
    return stiefel_names
