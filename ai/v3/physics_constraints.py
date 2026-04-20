"""
Physics as Hard Constraint via Augmented Lagrangian

Replaces soft penalty (λ_h=0.05) with equality constraints:
- Energy conservation: H(t) = H(0) (in vacuum)
- Momentum conservation: p(t) = p(0) (no external forces)
- Hydrostatic equilibrium: ∇p = -ρg (atmosphere)

Reference: ai/model_arch.md - Physics as Hard Constraint
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class AugmentedLagrangianLoss(nn.Module):
    """
    Augmented Lagrangian Loss for Physics Constraints
    
    Replaces soft penalty with equality constraints using the Augmented Lagrangian method:
        L = λ·c + (ρ/2)·c²
    
    Where:
    - c is the constraint violation (should be 0)
    - λ is the Lagrange multiplier (learned parameter)
    - ρ is the penalty parameter (fixed, increases during training)
    
    This enforces physics as hard constraints rather than soft penalties.
    """
    
    def __init__(self, initial_lambda: float = 1.0, rho: float = 10.0):
        super().__init__()
        
        # Lagrange multipliers (learned parameters)
        self.lambda_energy = nn.Parameter(torch.tensor(initial_lambda))
        self.lambda_momentum = nn.Parameter(torch.tensor(initial_lambda))
        self.lambda_hydrostatic = nn.Parameter(torch.tensor(initial_lambda))
        
        # Penalty parameter (fixed, can be increased during training)
        self.rho = rho
    
    def energy_constraint(self, H_pred: torch.Tensor, H_initial: torch.Tensor) -> torch.Tensor:
        """
        Energy conservation: H(t) = H(0)
        
        In vacuum (no drag, no control), the Hamiltonian should be conserved.
        This is a fundamental property of Hamiltonian systems.
        
        Constraint: c_E = H(t) - H(0) = 0
        Loss: L_E = λ·c_E + (ρ/2)·c_E²
        
        Args:
            H_pred: (batch,) predicted Hamiltonian at time t
            H_initial: (batch,) initial Hamiltonian at time 0
        
        Returns:
            loss_E: scalar energy constraint loss
        """
        c_E = H_pred - H_initial
        loss_E = self.lambda_energy * c_E + (self.rho / 2) * c_E**2
        return loss_E.mean()
    
    def momentum_constraint(self, p_pred: torch.Tensor, p_initial: torch.Tensor) -> torch.Tensor:
        """
        Momentum conservation: p(t) = p(0) (no external forces)
        
        In the absence of external forces (gravity is conservative, handled by Hamiltonian),
        the total momentum should be conserved.
        
        Constraint: c_p = p(t) - p(0) = 0
        Loss: L_p = λ·c_p + (ρ/2)·c_p²
        
        Args:
            p_pred: (batch, 3) predicted momentum [px, py, pz]
            p_initial: (batch, 3) initial momentum
        
        Returns:
            loss_p: scalar momentum constraint loss
        """
        c_p = p_pred - p_initial
        loss_p = self.lambda_momentum * c_p + (self.rho / 2) * c_p**2
        return loss_p.mean()
    
    def hydrostatic_constraint(self, pressure_grad: torch.Tensor, 
                                density: torch.Tensor, gravity: torch.Tensor) -> torch.Tensor:
        """
        Hydrostatic equilibrium: ∇p = -ρg
        
        In the atmosphere, the pressure gradient should balance gravity:
            ∇p/∂z = -ρ·g
        
        This is a key constraint for atmospheric modeling.
        
        Constraint: c_h = ∇p + ρg = 0
        Loss: L_h = λ·c_h + (ρ/2)·c_h²
        
        Args:
            pressure_grad: (batch, 3) pressure gradient [∂p/∂x, ∂p/∂y, ∂p/∂z]
            density: (batch,) atmospheric density at altitude
            gravity: (batch,) gravitational acceleration
        
        Returns:
            loss_h: scalar hydrostatic constraint loss
        """
        # Compute ρg (force per unit volume due to gravity)
        rho_g = density * gravity
        
        # Constraint: ∇p + ρg = 0
        # We focus on the z-component (vertical direction)
        c_h = pressure_grad[:, 2] + rho_g
        loss_h = self.lambda_hydrostatic * c_h + (self.rho / 2) * c_h**2
        return loss_h.mean()
    
    def forward(self, H_pred: torch.Tensor, H_initial: torch.Tensor, 
                p_pred: torch.Tensor, p_initial: torch.Tensor,
                pressure_grad: torch.Tensor, density: torch.Tensor, 
                gravity: torch.Tensor) -> torch.Tensor:
        """
        Compute total Augmented Lagrangian loss
        
        Args:
            H_pred: (batch,) predicted Hamiltonian
            H_initial: (batch,) initial Hamiltonian
            p_pred: (batch, 3) predicted momentum
            p_initial: (batch, 3) initial momentum
            pressure_grad: (batch, 3) pressure gradient
            density: (batch,) atmospheric density
            gravity: (batch,) gravitational acceleration
        
        Returns:
            total_loss: scalar total constraint loss
        """
        loss_energy = self.energy_constraint(H_pred, H_initial)
        loss_momentum = self.momentum_constraint(p_pred, p_initial)
        loss_hydrostatic = self.hydrostatic_constraint(pressure_grad, density, gravity)
        
        return loss_energy + loss_momentum + loss_hydrostatic
    
    def increase_penalty(self, factor: float = 2.0):
        """
        Increase penalty parameter ρ (standard in Augmented Lagrangian)
        
        As training progresses, increase ρ to enforce constraints more strictly.
        """
        self.rho *= factor


class StructurePreservingLoss(nn.Module):
    """
    Structure-Preserving Loss for Hamiltonian Neural Networks
    
    Enforces geometric structure of Hamiltonian systems:
    - Symplecticity (preserves phase space volume)
    - Time-reversibility
    - Energy conservation (in vacuum)
    """
    
    def __init__(self, energy_weight: float = 1.0, symplectic_weight: float = 0.1):
        super().__init__()
        
        self.energy_weight = energy_weight
        self.symplectic_weight = symplectic_weight
    
    def energy_loss(self, H_pred: torch.Tensor, H_initial: torch.Tensor) -> torch.Tensor:
        """
        Energy conservation loss
        
        For conservative systems, H should be constant.
        """
        return (H_pred - H_initial).abs().mean()
    
    def symplectic_loss(self, J: torch.Tensor, grad_H: torch.Tensor, 
                        state: torch.Tensor) -> torch.Tensor:
        """
        Symplecticity loss
        
        The flow should preserve the symplectic form: J·∇H
        """
        # Compute the symplectic term
        symplectic_term = torch.bmm(J.unsqueeze(0).expand(state.shape[0], -1, -1),
                                    grad_H.unsqueeze(-1))
        
        # This should equal the time derivative of the state
        # Loss encourages the learned dynamics to be symplectic
        return symplectic_term.norm(dim=-1).mean()
    
    def forward(self, H_pred: torch.Tensor, H_initial: torch.Tensor,
                J: torch.Tensor, grad_H: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute total structure-preserving loss
        """
        loss_energy = self.energy_loss(H_pred, H_initial)
        loss_symplectic = self.symplectic_loss(J, grad_H, state)
        
        return self.energy_weight * loss_energy + self.symplectic_weight * loss_symplectic


class PhysicsConstraintValidator:
    """
    Validator for physics constraints during training
    
    Monitors constraint violations and provides feedback.
    """
    
    def __init__(self):
        self.energy_violations = []
        self.momentum_violations = []
        self.hydrostatic_violations = []
    
    def update(self, energy_loss: float, momentum_loss: float, 
               hydrostatic_loss: float):
        """
        Update violation history
        """
        self.energy_violations.append(energy_loss)
        self.momentum_violations.append(momentum_loss)
        self.hydrostatic_violations.append(hydrostatic_loss)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics on constraint violations
        """
        return {
            'energy_mean': sum(self.energy_violations) / len(self.energy_violations),
            'energy_max': max(self.energy_violations),
            'momentum_mean': sum(self.momentum_violations) / len(self.momentum_violations),
            'momentum_max': max(self.momentum_violations),
            'hydrostatic_mean': sum(self.hydrostatic_violations) / len(self.hydrostatic_violations),
            'hydrostatic_max': max(self.hydrostatic_violations),
        }
    
    def reset(self):
        """
        Reset violation history
        """
        self.energy_violations = []
        self.momentum_violations = []
        self.hydrostatic_violations = []
