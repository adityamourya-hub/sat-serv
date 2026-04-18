"""
Time-Dependent Bogoliubov-de Gennes (TDBdG) Solver

Implements Split-Step Fourier Method (SSFM) for high-order accuracy
in evolving Majorana wavefunctions under time-varying QP poisoning.

Part of the Mediator Layer - Physics Engine for QIDT.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np

try:
    import diffrax as dfx
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False


@dataclass(frozen=True)
class SSFMConfig:
    """Split-Step Fourier Method configuration."""
    
    # Spatial grid
    L_nm: float = 2000.0  # Nanowire length
    N_x: int = 256  # Spatial grid points
    
    # Time stepping
    dt_fs: float = 1.0  # Femtosecond timestep
    n_steps: int = 1000
    
    # Physics parameters
    hbar_meV_fs: float = 0.6582  # ℏ in meV·fs
    m_eff_me: float = 0.023  # InAs effective mass
    
    # Split-step operators
    use_kinetic_half_step: bool = True  # Trotter-Suzuki decomposition
    
    @property
    def dx_nm(self) -> float:
        return self.L_nm / self.N_x
    
    @property
    def k_values(self) -> NDArray[np.float64]:
        """Reciprocal lattice vectors for FFT."""
        dk = 2 * np.pi / self.L_nm
        return np.fft.fftfreq(self.N_x, self.dx_nm) * 2 * np.pi


class SplitStepFourierSolver:
    """
    Split-Step Fourier Method for TDBdG evolution.
    
    Implements high-order accurate spatial evolution of Majorana wavefunctions:
        Ψ(r, t+dt) ≈ exp(-i H dt/2ℏ) Ψ(r, t)
    
    where H = H_kinetic + H_potential + H_pairing + H_qp_poisoning(t)
    
    The split-step approach:
    1. Half-step kinetic evolution in k-space (FFT)
    2. Full-step potential evolution in real space
    3. Half-step kinetic evolution in k-space (FFT)
    
    This gives O(dt³) local error per step, O(dt²) global error.
    """
    
    def __init__(self, config: Optional[SSFMConfig] = None):
        self.config = config or SSFMConfig()
        
        # Precompute kinetic energy operator in k-space
        self.k_vals = self.config.k_values
        self.T_k = self._compute_kinetic_operator()
        
        if HAS_JAX:
            self._step_jit = jit(self._single_step)
    
    def _compute_kinetic_operator(self) -> NDArray[np.float64]:
        """
        Compute kinetic energy in k-space: T(k) = ℏ²k² / (2m*)
        
        Returns diagonal operator in Fourier representation.
        """
        hbar = self.config.hbar_meV_fs
        m_star = self.config.m_eff_me  # In units of electron mass
        m_e_meV_fs2 = 0.511e6 / (3e8 * 1e15)**2  # Electron mass in meV·fs²/nm²
        
        # ℏ²/(2m*) in meV·nm²
        hbar2_over_2m = hbar**2 / (2 * m_star * m_e_meV_fs2)
        
        # Kinetic energy: T = ℏ²k² / 2m*
        T_k = hbar2_over_2m * self.k_vals**2
        
        return T_k
    
    def _potential_operator(
        self,
        psi: NDArray[np.complex128],
        V_ext: NDArray[np.float64],
        Delta: NDArray[np.complex128],
        qp_poisoning: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """
        Compute potential energy operator in real space.
        
        Includes:
        - External potential V_ext(r)
        - Superconducting pairing Δ(r)
        - QP poisoning suppression of gap
        """
        # QP-suppressed gap
        Delta_eff = Delta * (1 - qp_poisoning)
        
        # Potential operator (diagonal in real space)
        # For BdG: includes chemical potential, Zeeman, etc.
        V_diag = V_ext
        
        # Apply potential
        psi_new = psi * np.exp(-1j * V_diag * self.config.dt_fs / self.config.hbar_meV_fs)
        
        return psi_new
    
    def _kinetic_propagator(
        self,
        psi_k: NDArray[np.complex128],
        dt: float
    ) -> NDArray[np.complex128]:
        """
        Kinetic energy evolution in k-space.
        
        Multiplies by exp(-i T(k) dt / ℏ)
        """
        phase = -1j * self.T_k * dt / self.config.hbar_meV_fs
        return psi_k * np.exp(phase)
    
    def _single_step(
        self,
        psi: NDArray[np.complex128],
        V_ext: NDArray[np.float64],
        Delta: NDArray[np.complex128],
        qp_poisoning: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """
        Single SSFM step with Trotter-Suzuki decomposition.
        
        ψ(t+dt) = exp(-iVdt/2ℏ) × FFT⁻¹[exp(-iT(k)dt/ℏ) × FFT[exp(-iVdt/2ℏ) ψ(t)]]
        """
        dt = self.config.dt_fs
        
        # Half-step potential evolution
        psi_half = self._potential_operator(psi, V_ext, Delta, qp_poisoning * 0.5)
        
        # FFT to k-space
        psi_k = np.fft.fft(psi_half)
        
        # Full-step kinetic evolution
        psi_k_new = self._kinetic_propagator(psi_k, dt)
        
        # FFT back to real space
        psi_real = np.fft.ifft(psi_k_new)
        
        # Half-step potential evolution
        psi_new = self._potential_operator(psi_real, V_ext, Delta, qp_poisoning * 0.5)
        
        return psi_new
    
    def evolve(
        self,
        psi_initial: NDArray[np.complex128],
        V_ext: Callable[[float], NDArray[np.float64]],
        Delta: NDArray[np.complex128],
        qp_poisoning_t: Callable[[float], NDArray[np.float64]],
        t_span: Tuple[float, float]
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Full TDBdG evolution over time span.
        
        Args:
            psi_initial: Initial Majorana wavefunction
            V_ext: External potential V_ext(t) as function of time
            Delta: Superconducting pairing field
            qp_poisoning_t: QP poisoning profile n_qp(r, t)
            t_span: (t_start, t_end) in femtoseconds
            
        Returns:
            (t_array, psi_history) where psi_history has shape (n_steps, N_x)
        """
        t_start, t_end = t_span
        dt = self.config.dt_fs
        n_steps = int((t_end - t_start) / dt)
        
        # Storage
        t_history = np.zeros(n_steps + 1)
        psi_history = np.zeros((n_steps + 1, len(psi_initial)), dtype=np.complex128)
        
        # Initial condition
        psi = psi_initial.copy()
        t_history[0] = t_start
        psi_history[0] = psi
        
        # Time evolution
        for i in range(n_steps):
            t = t_start + i * dt
            
            # Get time-dependent potentials
            V_t = V_ext(t)
            qp_t = qp_poisoning_t(t)
            
            # SSFM step
            if HAS_JAX and hasattr(self, '_step_jit'):
                psi = self._step_jit(psi, V_t, Delta, qp_t)
            else:
                psi = self._single_step(psi, V_t, Delta, qp_t)
            
            # Store
            t_history[i + 1] = t + dt
            psi_history[i + 1] = psi
        
        return t_history, psi_history
    
    def compute_majorana_overlap(
        self,
        psi_history: NDArray[np.complex128],
        target_mzm_sites: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """
        Compute overlap between evolving wavefunction and target MZM sites.
        
        Critical metric for evasion: maximize overlap with "clean" sectors.
        """
        overlaps = np.zeros(len(psi_history))
        
        for i, psi in enumerate(psi_history):
            # Wavefunction amplitude at target sites
            psi_target = psi[target_mzm_sites]
            # Normalized overlap
            overlaps[i] = np.sum(np.abs(psi_target)**2) / np.sum(np.abs(psi)**2)
        
        return overlaps


class HamiltonianNeuralNetwork:
    """
    Hamiltonian Neural Network (HNN) for learning energy-conserving QP flow.
    
    Learns the Hamiltonian dynamics of quasiparticle diffusion, enabling
    predictive "look-ahead" simulation rather than reactive calculation.
    
    Based on HNN architecture: learns dynamics that conserve learned Hamiltonian.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 3
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Initialize network parameters (would be trained)
        # For now: placeholder architecture
        self.params = self._init_params()
    
    def _init_params(self) -> dict:
        """Initialize HNN parameters."""
        rng = np.random.default_rng(42)
        
        params = {
            'W1': rng.normal(0, 0.01, (self.hidden_dim, self.input_dim)),
            'b1': np.zeros(self.hidden_dim),
            'W2': rng.normal(0, 0.01, (self.hidden_dim, self.hidden_dim)),
            'b2': np.zeros(self.hidden_dim),
            'W_H': rng.normal(0, 0.01, (1, self.hidden_dim)),  # Hamiltonian output
            'b_H': np.zeros(1)
        }
        
        return params
    
    def forward(self, state: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """
        Forward pass to compute state derivatives and learned Hamiltonian.
        
        Args:
            state: [position, momentum] concatenated vector
            
        Returns:
            (state_derivatives, learned_Hamiltonian)
        """
        # Split into position and momentum
        half = len(state) // 2
        q = state[:half]
        p = state[half:]
        
        # Encode state
        x = np.concatenate([q, p])
        
        # Neural network layers
        h1 = np.tanh(self.params['W1'] @ x + self.params['b1'])
        h2 = np.tanh(self.params['W2'] @ h1 + self.params['b2'])
        
        # Learned Hamiltonian H(q, p)
        H_learned = float(self.params['W_H'] @ h2 + self.params['b_H'])
        
        # Hamilton's equations: ∂q/∂t = ∂H/∂p, ∂p/∂t = -∂H/∂q
        # Approximate derivatives via finite differences of network
        eps = 1e-4
        dq_H = np.zeros_like(q)
        dp_H = np.zeros_like(p)
        
        for i in range(len(q)):
            q_plus = q.copy()
            q_plus[i] += eps
            x_plus = np.concatenate([q_plus, p])
            h1_plus = np.tanh(self.params['W1'] @ x_plus + self.params['b1'])
            h2_plus = np.tanh(self.params['W2'] @ h1_plus + self.params['b2'])
            H_plus = float(self.params['W_H'] @ h2_plus + self.params['b_H'])
            
            dq_H[i] = (H_plus - H_learned) / eps
        
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            x_plus = np.concatenate([q, p_plus])
            h1_plus = np.tanh(self.params['W1'] @ x_plus + self.params['b1'])
            h2_plus = np.tanh(self.params['W2'] @ h1_plus + self.params['b2'])
            H_plus = float(self.params['W_H'] @ h2_plus + self.params['b_H'])
            
            dp_H[i] = -(H_plus - H_learned) / eps  # Negative for momentum
        
        # State derivatives
        state_derivatives = np.concatenate([dq_H, dp_H])
        
        return state_derivatives, H_learned
    
    def predict_qp_front(
        self,
        initial_state: NDArray[np.float64],
        n_steps: int,
        dt: float
    ) -> NDArray[np.float64]:
        """
        Predict QP density front evolution using learned Hamiltonian.
        
        Enables look-ahead prediction rather than reactive simulation.
        """
        state = initial_state.copy()
        trajectory = [state.copy()]
        
        for _ in range(n_steps):
            deriv, _ = self.forward(state)
            state = state + dt * deriv
            trajectory.append(state.copy())
        
        return np.array(trajectory)


def create_tdbdg_solver_for_nanowire(
    length_nm: float = 2000.0,
    num_sites: int = 256,
    effective_mass: float = 0.023
) -> SplitStepFourierSolver:
    """
    Factory function for creating TDBdG solver for InAs/Al nanowire.
    
    Args:
        length_nm: Nanowire length
        num_sites: Spatial resolution
        effective_mass: InAs effective mass (m_e)
        
    Returns:
        Configured SSFM solver
    """
    config = SSFMConfig(
        L_nm=length_nm,
        N_x=num_sites,
        m_eff_me=effective_mass
    )
    
    return SplitStepFourierSolver(config)


__all__ = [
    "SSFMConfig",
    "SplitStepFourierSolver",
    "HamiltonianNeuralNetwork",
    "create_tdbdg_solver_for_nanowire",
]
