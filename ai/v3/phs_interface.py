"""
Port-Hamiltonian System (PHS) Shared Interface Specification

This module defines the structural contract for PHS implementations across
JAX (training) and PyTorch (deployment) backends. All implementations must
satisfy the (J, R, H, g) blueprint to ensure mathematical identity.

Structural Contract:
    J(z) ∈ R^(6×6): Skew-symmetric interconnection matrix (J = -J^T)
    R(z) ∈ R^(6×6): Positive Semi-Definite dissipation matrix (R = LL^T)
    H(z) ∈ R^1: Scalar Hamiltonian (total energy)
    g(z) ∈ R^(6×3): Control coupling matrix

Physical Constants (NIST-aligned):
    μ (Earth gravitational parameter): 3.986004418e14 m³/s²
    J₂ (Earth oblateness): 1.08263e-3
    Rₑ (Earth equatorial radius): 6378137.0 m

State Vector z = [q, p] where:
    q ∈ R³: Position (x, y, z) in meters
    p ∈ R³: Momentum (m*v) in kg·m/s

Reference:
    - Port-Hamiltonian Systems: van der Schaft, Maschke
    - Symplectic Integration: Yoshida (1990)
    - Neural Operators: Kovachki et al.
"""

from typing import Tuple, Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np

# NIST Physical Constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter (m³/s²)
J2_EARTH = 1.08263e-3       # Earth oblateness coefficient
R_EARTH = 6378137.0         # Earth equatorial radius (m)


@dataclass
class PHSConfig:
    """Configuration for Port-Hamiltonian System."""
    
    # Network dimensions
    state_dim: int = 6              # Position (3) + Momentum (3)
    control_dim: int = 3            # Thrust (3D)
    hamiltonian_width: int = 128    # H_net hidden width
    dissipation_width: int = 32     # R_net (Cholesky) hidden width
    coupling_width: int = 64        # g_net hidden width
    
    # Physical parameters
    mu: float = MU_EARTH
    j2: float = J2_EARTH
    r_earth: float = R_EARTH
    
    # Numerical tolerances
    skew_symmetry_tolerance: float = 1e-7
    psd_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-7


@runtime_checkable
class PortHamiltonianProtocol(Protocol):
    """
    Protocol defining the PHS structural contract.
    
    All implementations (JAX, PyTorch, ONNX) must satisfy this interface
    to ensure mathematical identity across backends.
    """
    
    def interconnection_matrix(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the skew-symmetric interconnection matrix J(z).
        
        Spec:
            - Shape: (6, 6)
            - Property: J = -J^T (skew-symmetric)
            - Canonical form: [[0, I], [-I, 0]] for orbital mechanics
            
        Args:
            z: State vector [q, p] ∈ R⁶
            
        Returns:
            J(z) ∈ R^(6×6), skew-symmetric
        """
        ...
    
    def dissipation_matrix(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the positive semi-definite dissipation matrix R(z).
        
        Spec:
            - Shape: (6, 6)
            - Property: R ≽ 0 (PSD)
            - Parameterization: R = LL^T via Cholesky MLP
            
        Args:
            z: State vector [q, p] ∈ R⁶
            
        Returns:
            R(z) ∈ R^(6×6), positive semi-definite
        """
        ...
    
    def hamiltonian(self, z: np.ndarray) -> float:
        """
        Compute the scalar Hamiltonian (total energy).
        
        Spec:
            - Shape: scalar
            - Physical: H = T + V (kinetic + gravitational potential)
            - Activation: Tanh for bounded output
            
        Args:
            z: State vector [q, p] ∈ R⁶
            
        Returns:
            H(z) ∈ R, total energy in Joules
        """
        ...
    
    def control_coupling(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the control coupling matrix g(z).
        
        Spec:
            - Shape: (6, 3)
            - Maps 3D thrust to 6D state space
            - Couples "Port" (thrust) to momentum coordinates
            
        Args:
            z: State vector [q, p] ∈ R⁶
            
        Returns:
            g(z) ∈ R^(6×3), control coupling matrix
        """
        ...
    
    def vector_field(self, z: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """
        Compute the PHS vector field (time derivative).
        
        Equation: ż = (J - R)∇H + gu
        
        Args:
            z: State vector [q, p] ∈ R⁶
            u: Control input (thrust) ∈ R³ (optional)
            
        Returns:
            ż ∈ R⁶, time derivative of state
        """
        ...
    
    def gradient_hamiltonian(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Hamiltonian ∇H(z).
        
        This is critical for:
            - Symplectic integration
            - Energy conservation
            - Augmented Lagrangian loss
            
        Spec:
            - Shape: (6,)
            - Must match autodiff across backends (1e-7 tolerance)
            
        Args:
            z: State vector [q, p] ∈ R⁶
            
        Returns:
            ∇H(z) ∈ R⁶, gradient of Hamiltonian
        """
        ...


def canonical_interconnection_matrix() -> np.ndarray:
    """
    Returns the canonical skew-symmetric interconnection matrix.
    
    J = [[0,  I],
         [-I, 0]]
         
    This is the standard structure for Hamiltonian systems with
    position and momentum coordinates.
    
    Returns:
        J ∈ R^(6×6), skew-symmetric
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    
    J = np.block([
        [Z3,  I3],
        [-I3, Z3]
    ])
    
    return J


def validate_skew_symmetry(J: np.ndarray, tol: float = 1e-7) -> bool:
    """
    Validate that matrix J is skew-symmetric.
    
    Check: J + J^T = 0 within tolerance
    
    Args:
        J: Matrix to validate
        tol: Numerical tolerance
        
    Returns:
        True if skew-symmetric, False otherwise
    """
    return np.allclose(J + J.T, 0, atol=tol)


def validate_psd(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Validate that matrix R is positive semi-definite.
    
    Check: All eigenvalues ≥ -tol
    
    Args:
        R: Matrix to validate
        tol: Numerical tolerance (allows small negative eigenvalues)
        
    Returns:
        True if PSD, False otherwise
    """
    eigvals = np.linalg.eigvalsh(R)
    return np.all(eigvals >= -tol)


def validate_jacobian(jax_grad: np.ndarray, torch_grad: np.ndarray, 
                      tol: float = 1e-7) -> Tuple[bool, float]:
    """
    Validate that JAX and PyTorch gradients match.
    
    Args:
        jax_grad: Gradient from JAX
        torch_grad: Gradient from PyTorch
        tol: Numerical tolerance
        
    Returns:
        (is_valid, max_error)
    """
    error = np.max(np.abs(jax_grad - torch_grad))
    is_valid = error < tol
    return is_valid, float(error)


def compute_gravitational_potential(r: np.ndarray, mu: float = MU_EARTH) -> float:
    """
    Compute gravitational potential energy.
    
    V(r) = -μ / |r|
    
    Args:
        r: Position vector [x, y, z] in meters
        mu: Gravitational parameter
        
    Returns:
        V in Joules
    """
    r_norm = np.linalg.norm(r)
    return -mu / r_norm


def compute_kinetic_energy(p: np.ndarray) -> float:
    """
    Compute kinetic energy from momentum.
    
    T(p) = |p|² / (2m)
    
    For orbital mechanics with unit mass: T = |p|² / 2
    
    Args:
        p: Momentum vector [px, py, pz] in kg·m/s
        
    Returns:
        T in Joules
    """
    return 0.5 * np.sum(p**2)


def standard_hamiltonian(z: np.ndarray, mu: float = MU_EARTH) -> float:
    """
    Compute the standard orbital Hamiltonian (analytic).
    
    H(z) = T(p) + V(r) = |p|²/2 - μ/|r|
    
    This serves as the ground truth for neural Hamiltonian training.
    
    Args:
        z: State vector [x, y, z, px, py, pz]
        mu: Gravitational parameter
        
    Returns:
        H in Joules
    """
    q = z[:3]  # Position
    p = z[3:]  # Momentum
    
    T = compute_kinetic_energy(p)
    V = compute_gravitational_potential(q, mu)
    
    return T + V


def gradient_standard_hamiltonian(z: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """
    Compute the analytic gradient of the standard Hamiltonian.
    
    ∇H = [∂H/∂q, ∂H/∂p]
       = [μ·r/|r|³, p]
    
    Args:
        z: State vector [x, y, z, px, py, pz]
        mu: Gravitational parameter
        
    Returns:
        ∇H ∈ R⁶
    """
    q = z[:3]
    p = z[3:]
    
    r_norm = np.linalg.norm(q)
    grad_q = mu * q / (r_norm**3)  # ∂H/∂q = μ·r/|r|³
    grad_p = p  # ∂H/∂p = p
    
    return np.concatenate([grad_q, grad_p])


class PHSAuditSuite:
    """
    Test suite for validating PHS implementations.
    
    Performs the "Audit and Align" checks:
        1. Skew-symmetry of J
        2. PSD property of R
        3. Jacobian matching between JAX and PyTorch
        4. Hamiltonian gradient accuracy
    """
    
    def __init__(self, config: PHSConfig = None):
        self.config = config or PHSConfig()
        self.test_states = self._generate_test_states(10000)
    
    def _generate_test_states(self, n: int) -> np.ndarray:
        """Generate random orbital states for testing."""
        # Random positions in LEO to GEO range (200km to 36000km altitude)
        r_min = R_EARTH + 200e3
        r_max = R_EARTH + 36000e3
        
        states = []
        for _ in range(n):
            # Random position
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(r_min, r_max)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Random momentum (circular orbit approximation)
            v_mag = np.sqrt(MU_EARTH / r)
            vx = v_mag * np.random.uniform(-1, 1)
            vy = v_mag * np.random.uniform(-1, 1)
            vz = v_mag * np.random.uniform(-1, 1)
            
            states.append([x, y, z, vx, vy, vz])
        
        return np.array(states)
    
    def audit_interconnection(self, model: PortHamiltonianProtocol) -> dict:
        """Audit J matrix for skew-symmetry."""
        max_error = 0.0
        for z in self.test_states[:100]:  # Sample 100 states
            J = model.interconnection_matrix(z)
            error = np.max(np.abs(J + J.T))
            max_error = max(max_error, error)
        
        is_valid = max_error < self.config.skew_symmetry_tolerance
        return {
            'valid': is_valid,
            'max_error': float(max_error),
            'tolerance': self.config.skew_symmetry_tolerance,
        }
    
    def audit_dissipation(self, model: PortHamiltonianProtocol) -> dict:
        """Audit R matrix for PSD property."""
        min_eig = 0.0
        for z in self.test_states[:100]:
            R = model.dissipation_matrix(z)
            eigvals = np.linalg.eigvalsh(R)
            min_eig = min(min_eig, np.min(eigvals))
        
        is_valid = min_eig >= -self.config.psd_tolerance
        return {
            'valid': is_valid,
            'min_eigenvalue': float(min_eig),
            'tolerance': self.config.psd_tolerance,
        }
    
    def audit_hamiltonian_gradient(self, model: PortHamiltonianProtocol) -> dict:
        """Audit Hamiltonian gradient against analytic ground truth."""
        max_error = 0.0
        for z in self.test_states[:100]:
            grad_nn = model.gradient_hamiltonian(z)
            grad_analytic = gradient_standard_hamiltonian(z)
            error = np.max(np.abs(grad_nn - grad_analytic))
            max_error = max(max_error, error)
        
        is_valid = max_error < self.config.gradient_tolerance
        return {
            'valid': is_valid,
            'max_error': float(max_error),
            'tolerance': self.config.gradient_tolerance,
        }
    
    def cross_backend_audit(self, jax_model: PortHamiltonianProtocol,
                           torch_model: PortHamiltonianProtocol) -> dict:
        """
        Audit gradient matching between JAX and PyTorch backends.
        
        This is the critical "Model Drift" check.
        """
        max_error = 0.0
        for z in self.test_states[:100]:
            grad_jax = jax_model.gradient_hamiltonian(z)
            grad_torch = torch_model.gradient_hamiltonian(z)
            error = np.max(np.abs(grad_jax - grad_torch))
            max_error = max(max_error, error)
        
        is_valid = max_error < self.config.gradient_tolerance
        return {
            'valid': is_valid,
            'max_error': float(max_error),
            'tolerance': self.config.gradient_tolerance,
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Port-Hamiltonian System (PHS) Interface Specification")
    print("=" * 70)
    
    config = PHSConfig()
    print(f"\nConfiguration:")
    print(f"  State dimension: {config.state_dim}")
    print(f"  Control dimension: {config.control_dim}")
    print(f"  Hamiltonian width: {config.hamiltonian_width}")
    print(f"  Dissipation width: {config.dissipation_width}")
    print(f"  Gradient tolerance: {config.gradient_tolerance}")
    
    print(f"\nPhysical Constants (NIST):")
    print(f"  μ (Earth): {config.mu:.6e} m³/s²")
    print(f"  J₂ (oblateness): {config.j2:.6e}")
    print(f"  Rₑ (radius): {config.r_earth:.1f} m")
    
    print(f"\nCanonical Interconnection Matrix J:")
    J = canonical_interconnection_matrix()
    print(J)
    print(f"  Skew-symmetric: {validate_skew_symmetry(J)}")
    
    print(f"\nAudit Suite:")
    audit = PHSAuditSuite(config)
    print(f"  Test states generated: {len(audit.test_states)}")
    print(f"  Audit methods: interconnection, dissipation, gradient, cross_backend")
    
    print("\n" + "=" * 70)
    print("This specification ensures mathematical identity across:")
    print("  - JAX (H100 training)")
    print("  - PyTorch (ONNX export)")
    print("  - Go-native (M2 Air inference)")
    print("=" * 70)
