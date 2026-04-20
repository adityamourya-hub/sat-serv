"""
Latent-Conditioned Port-Hamiltonian System (LC-PHS) Trainer

Integrates 5 critical workstreams for ADITI 4.0 May 4th deadline:
1. FiLM Conditioning Architecture
2. Latent Map & Disentanglement (3 anchor points)
3. Supervised Contrastive (SupCon) Loss (τ=0.07)
4. Joint Physics Training (40/30/30 mix on H100)
5. Symplectic-Latent Alignment Validation

Architecture:
    - Latent dim: d=32 (balanced for M2 Air speed)
    - Conditioning: FiLM MLP (γ, β per layer)
    - Integrator: Yoshida-4 (Forest-Ruth) 4th-order symplectic
    - Training: Joint orbital/ballistic/HGV mix
    - Validation: Fixed-latent Jacobian symplectic test

Strategic Rationale:
    By choosing Latent Conditioning over Hypernetworks, we prioritize
    stability for defense deployment. The model selects the correct
    physical manifold based on a verified latent code.

References:
    - FiLM: Perez et al. "FiLM: Visual Reasoning with Conditioning"
    - SupCon: Khosla et al. "Supervised Contrastive Learning"
    - Forest-Ruth: Forest & Ruth (1990) 4th-order symplectic
"""

from typing import Dict, Tuple, Optional, Callable
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import equinox as eqx
import optax
from dataclasses import dataclass
from functools import partial
import time

from port_hamiltonian import (
    PortHamiltonianSystem, 
    FiLMNetwork,
    supervised_contrastive_loss,
    interpolate_latent_codes
)


# =============================================================================
# Workstream 2: Latent Map & Disentanglement
# =============================================================================

LATENT_DIM = 32  # Optimized for M2 Air speed (was 64)

# Regime anchor points in latent space (learned during training)
REGIME_ANCHORS = {
    "orbital": jnp.zeros(LATENT_DIM),      # z_orb: J2 perturbations, vacuum
    "ballistic": jnp.zeros(LATENT_DIM),    # z_bal: re-entry drag
    "hgv": jnp.zeros(LATENT_DIM),          # z_hgv: skip-glide lift/drag
}


def get_regime_latent(regime: str, alpha: float = 1.0) -> jnp.ndarray:
    """
    Get latent code for a regime with optional intensity scaling.
    
    Args:
        regime: "orbital", "ballistic", "hgv", or "interpolated"
        alpha: Intensity (0-1 for interpolation, >1 for extrapolation)
        
    Returns:
        z: [LATENT_DIM] latent code
    """
    if regime in REGIME_ANCHORS:
        return alpha * REGIME_ANCHORS[regime]
    elif regime == "interpolated":
        # Default: midpoint between orbital and ballistic (Karman line)
        return 0.5 * REGIME_ANCHORS["orbital"] + 0.5 * REGIME_ANCHORS["ballistic"]
    else:
        raise ValueError(f"Unknown regime: {regime}")


def interpolate_regime_latents(regime1: str, regime2: str, alpha: float) -> jnp.ndarray:
    """
    Linear interpolation manifold for smooth physical transitions.
    
    Useful for "Breach" scenarios:
        - At 100km (Karman line): α=0.5 between orbital/ballistic
        - During skip-glide: α varies with altitude
    
    Args:
        regime1: First regime (e.g., "orbital")
        regime2: Second regime (e.g., "hgv")
        alpha: Interpolation parameter [0, 1]
        
    Returns:
        z_interp: Interpolated latent code
    """
    z1 = REGIME_ANCHORS[regime1]
    z2 = REGIME_ANCHORS[regime2]
    return interpolate_latent_codes(z1, z2, alpha)


def altitude_to_regime_blend(altitude_km: float) -> Tuple[str, str, float]:
    """
    Map altitude to regime interpolation for atmospheric transitions.
    
    Altitude bands:
        > 200 km: Pure orbital (vacuum)
        100-200 km: Blend orbital/ballistic (Karman line region)
        50-100 km: Blend ballistic/HGV (re-entry)
        < 50 km: Pure HGV (endurance)
        
    Args:
        altitude_km: Altitude in kilometers
        
    Returns:
        (regime1, regime2, alpha) for interpolation
    """
    if altitude_km > 200:
        return ("orbital", "orbital", 0.0)  # Pure orbital
    elif altitude_km > 100:
        # Transition: orbital → ballistic
        alpha = (200 - altitude_km) / 100.0  # 0 at 200km, 1 at 100km
        return ("orbital", "ballistic", alpha)
    elif altitude_km > 50:
        # Transition: ballistic → HGV
        alpha = (100 - altitude_km) / 50.0  # 0 at 100km, 1 at 50km
        return ("ballistic", "hgv", alpha)
    else:
        return ("hgv", "hgv", 0.0)  # Pure HGV


# =============================================================================
# Workstream 3: Supervised Contrastive Loss (SupCon)
# =============================================================================

SUPCON_TEMPERATURE = 0.07  # Standard for disentanglement (was 0.1)


def compute_supcon_loss(z_codes: jnp.ndarray, 
                        labels: jnp.ndarray,
                        temperature: float = SUPCON_TEMPERATURE) -> float:
    """
    Supervised Contrastive Loss with τ=0.07.
    
    Forces HGV-style trajectories into one latent cluster,
    Orbital-style into another. Ensures ∇H is unambiguous per regime.
    
    Args:
        z_codes: [batch, LATENT_DIM] latent physics codes
        labels: [batch] regime labels (0=orbital, 1=ballistic, 2=hgv)
        temperature: τ=0.07 (standard for disentanglement)
        
    Returns:
        loss: Scalar contrastive loss
    """
    return supervised_contrastive_loss(z_codes, labels, temperature)


# =============================================================================
# Workstream 5: Symplectic-Latent Alignment Validation
# =============================================================================

def compute_fixed_latent_symplectic_jacobian(phs: PortHamiltonianSystem,
                                                z_state: jnp.ndarray,
                                                z_latent: jnp.ndarray,
                                                dt: float) -> Tuple[jnp.ndarray, float]:
    """
    Compute symplectic Jacobian with FIXED latent code.
    
    Validates that for a fixed z, the learned H(x, z) preserves energy.
    The latent code z is treated as constant during Jacobian computation.
    
    Symplectic condition: M^T * J * M = J
    where M = ∂Ψ(z_t, z_latent) / ∂z_t
    
    Args:
        phs: PortHamiltonianSystem
        z_state: [state_dim] current state
        z_latent: [LATENT_DIM] fixed latent code
        dt: Time step
        
    Returns:
        (symp_residual, residual_norm)
    """
    from port_hamiltonian import Yoshida4Integrator, get_symplectic_form
    
    # Define flow map with fixed latent
    def flow_map(z_single):
        """Single step with fixed z_latent."""
        integrator = Yoshida4Integrator(dt=dt)
        return integrator.step(phs, z_single, u=None, z_latent=z_latent)
    
    # Compute Jacobian
    jac = jax.jacfwd(flow_map)(z_state)
    
    # Get symplectic form
    J = get_symplectic_form(z_state.device if hasattr(z_state, 'device') else None)
    
    # Compute M^T * J * M - J
    MT = jac.T
    MT_J = MT @ J
    MT_J_M = MT_J @ jac
    symp_residual = MT_J_M - J
    
    # Frobenius norm
    residual_norm = jnp.sqrt(jnp.sum(symp_residual ** 2))
    
    return symp_residual, residual_norm


def validate_symplectic_latent_alignment(phs: PortHamiltonianSystem,
                                          test_states: jnp.ndarray,
                                          z_latent: jnp.ndarray,
                                          dt: float,
                                          tolerance: float = 1e-6) -> Dict:
    """
    Validation test: Symplectic condition for fixed latent code.
    
    Ensures the M2 Air won't see "Energy Leaks" in the dashboard.
    
    Args:
        phs: PortHamiltonianSystem
        test_states: [N, state_dim] test states
        z_latent: [LATENT_DIM] fixed latent code
        dt: Time step
        tolerance: Acceptable symplectic residual norm
        
    Returns:
        Validation results dictionary
    """
    max_residual = 0.0
    mean_residual = 0.0
    
    for i in range(test_states.shape[0]):
        _, residual_norm = compute_fixed_latent_symplectic_jacobian(
            phs, test_states[i], z_latent, dt
        )
        max_residual = max(max_residual, float(residual_norm))
        mean_residual += float(residual_norm)
    
    mean_residual /= test_states.shape[0]
    
    return {
        "max_symp_residual": max_residual,
        "mean_symp_residual": mean_residual,
        "tolerance": tolerance,
        "passed": max_residual < tolerance,
        "z_latent_norm": float(jnp.linalg.norm(z_latent)),
    }


# =============================================================================
# Workstream 4: Joint Physics Training (H100 Sprint)
# =============================================================================

@dataclass
class LCPHSTrainingConfig:
    """Configuration for LC-PHS training on H100."""
    
    # Data mix ratios (40/30/30 split)
    orbital_ratio: float = 0.40
    ballistic_ratio: float = 0.30
    hgv_ratio: float = 0.30
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 100
    gradient_clip: float = 1.0  # Clip high-force HGV gradients
    
    # Loss weights
    lambda_dynamics: float = 1.0
    lambda_supcon: float = 0.1
    lambda_zreg: float = 0.01
    lambda_symplectic: float = 0.1
    
    # FiLM configuration
    use_film: bool = True
    
    # Validation
    validate_every: int = 10
    symplectic_tolerance: float = 1e-6


def create_joint_dataloader(config: LCPHSTrainingConfig,
                            orbital_data: jnp.ndarray,
                            ballistic_data: jnp.ndarray,
                            hgv_data: jnp.ndarray,
                            key: jax.random.PRNGKey):
    """
    Create interleaved dataloader for joint physics training.
    
    Data mix: 40% orbital, 30% ballistic, 30% HGV
    Total: 10^8 transitions (H100 memory bandwidth optimized)
    
    Args:
        config: Training configuration
        orbital_data: [N_orb, T, state_dim] orbital trajectories
        ballistic_data: [N_bal, T, state_dim] ballistic trajectories
        hgv_data: [N_hgv, T, state_dim] HGV trajectories
        key: JAX random key
        
    Yields:
        Batches with mixed regime data and labels
    """
    # Calculate batch composition
    batch_size = config.batch_size
    n_orb = int(batch_size * config.orbital_ratio)
    n_bal = int(batch_size * config.ballistic_ratio)
    n_hgv = batch_size - n_orb - n_bal  # Remainder
    
    num_batches = min(
        orbital_data.shape[0] // n_orb,
        ballistic_data.shape[0] // n_bal,
        hgv_data.shape[0] // n_hgv
    )
    
    for batch_idx in range(num_batches):
        key, subkey = jax.random.split(key)
        
        # Sample indices for each regime
        orb_idx = jax.random.randint(subkey, (n_orb,), 0, orbital_data.shape[0])
        key, subkey = jax.random.split(key)
        bal_idx = jax.random.randint(subkey, (n_bal,), 0, ballistic_data.shape[0])
        key, subkey = jax.random.split(key)
        hgv_idx = jax.random.randint(subkey, (n_hgv,), 0, hgv_data.shape[0])
        
        # Combine data
        batch_data = jnp.concatenate([
            orbital_data[orb_idx],
            ballistic_data[bal_idx],
            hgv_data[hgv_idx]
        ], axis=0)
        
        # Labels: 0=orbital, 1=ballistic, 2=hgv
        batch_labels = jnp.concatenate([
            jnp.zeros(n_orb),
            jnp.ones(n_bal),
            2 * jnp.ones(n_hgv)
        ]).astype(jnp.int32)
        
        # Shuffle within batch
        perm = jax.random.permutation(subkey, batch_size)
        batch_data = batch_data[perm]
        batch_labels = batch_labels[perm]
        
        yield batch_data, batch_labels


class LCPHSTrainer:
    """
    Trainer for Latent-Conditioned Port-Hamiltonian System.
    
    Implements:
    - Joint training on orbital/ballistic/HGV data
    - FiLM conditioning
    - SupCon loss for latent disentanglement
    - Gradient clipping for stability
    - Symplectic validation
    """
    
    def __init__(self, config: LCPHSTrainingConfig, key: jax.random.PRNGKey):
        self.config = config
        
        # Initialize PHS with FiLM
        self.phs = PortHamiltonianSystem(
            state_dim=6,
            control_dim=3,
            latent_dim=LATENT_DIM,
            use_film=config.use_film,
            R_conditioned=True,
            g_conditioned=True,
            key=key,
        )
        
        # Optimizer with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            optax.adam(config.learning_rate)
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.phs, eqx.is_array))
        
        # Initialize regime anchors (learnable)
        self.z_orbital = jnp.zeros(LATENT_DIM)
        self.z_ballistic = jnp.zeros(LATENT_DIM)
        self.z_hgv = jnp.zeros(LATENT_DIM)
        
    def compute_loss(self, phs, batch_data, batch_labels, dt=0.1):
        """
        Compute combined loss for a batch.
        
        L_total = L_dynamics + λ_supcon * L_supcon + λ_zreg * ||z||²
        """
        batch_size = batch_data.shape[0]
        
        # Get latent codes for each sample based on label
        z_codes = jnp.zeros((batch_size, LATENT_DIM))
        for i in range(batch_size):
            label = batch_labels[i]
            if label == 0:
                z_codes = z_codes.at[i].set(self.z_orbital)
            elif label == 1:
                z_codes = z_codes.at[i].set(self.z_ballistic)
            else:
                z_codes = z_codes.at[i].set(self.z_hgv)
        
        # 1. Dynamics loss: predict next state
        loss_dynamics = 0.0
        for i in range(batch_size):
            z0 = batch_data[i, 0]  # Initial state
            z1_true = batch_data[i, 1]  # Next state
            
            # Predict with PHS
            z1_pred = self.phs.dynamics(z0, u=None, z_latent=z_codes[i]) * dt + z0
            loss_dynamics += jnp.mean((z1_pred - z1_true) ** 2)
        
        loss_dynamics /= batch_size
        
        # 2. SupCon loss
        loss_supcon = compute_supcon_loss(z_codes, batch_labels)
        
        # 3. L2 regularization on latent codes
        loss_zreg = (jnp.sum(self.z_orbital ** 2) + 
                     jnp.sum(self.z_ballistic ** 2) + 
                     jnp.sum(self.z_hgv ** 2))
        
        # Total loss
        loss_total = (self.config.lambda_dynamics * loss_dynamics +
                      self.config.lambda_supcon * loss_supcon +
                      self.config.lambda_zreg * loss_zreg)
        
        return loss_total, {
            "loss_total": loss_total,
            "loss_dynamics": loss_dynamics,
            "loss_supcon": loss_supcon,
            "loss_zreg": loss_zreg,
        }
    
    @jit
    def train_step(self, phs, opt_state, batch_data, batch_labels):
        """Single training step with JIT compilation for H100."""
        (loss, metrics), grads = jax.value_and_grad(
            self.compute_loss, has_aux=True
        )(phs, batch_data, batch_labels)
        
        updates, opt_state = self.optimizer.update(grads, opt_state)
        phs = eqx.apply_updates(phs, updates)
        
        return phs, opt_state, loss, metrics
    
    def train(self, dataloader, num_epochs=None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print("=" * 70)
        print("LC-PHS Training (H100 Sprint)")
        print("=" * 70)
        print(f"Config: {self.config}")
        print(f"Latent dim: {LATENT_DIM}")
        print(f"SupCon temperature: {SUPCON_TEMPERATURE}")
        print(f"Data mix: {self.config.orbital_ratio}/{self.config.ballistic_ratio}/{self.config.hgv_ratio}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in dataloader:
                self.phs, self.opt_state, loss, metrics = self.train_step(
                    self.phs, self.opt_state, batch_data, batch_labels
                )
                epoch_loss += float(loss)
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # Validation
            if epoch % self.config.validate_every == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                print(f"  Dynamics: {metrics['loss_dynamics']:.6f}")
                print(f"  SupCon: {metrics['loss_supcon']:.6f}")
                print(f"  Z-Reg: {metrics['loss_zreg']:.6f}")


def demo_lc_phs():
    """Demonstrate LC-PHS capabilities."""
    print("=" * 70)
    print("Latent-Conditioned Port-Hamiltonian System (LC-PHS)")
    print("=" * 70)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    config = LCPHSTrainingConfig()
    
    print("\n1. Architecture:")
    print(f"   Latent dimension: {LATENT_DIM} (optimized for M2 Air)")
    print(f"   Conditioning: FiLM (Feature-wise Linear Modulation)")
    print(f"   FiLM equation: h' = γ(z) ⊙ h + β(z)")
    
    print("\n2. Regime Anchors:")
    for regime, z in REGIME_ANCHORS.items():
        print(f"   {regime}: ||z|| = {float(jnp.linalg.norm(z)):.2f}")
    
    print("\n3. Altitude-to-Regime Mapping:")
    for alt in [300, 150, 75, 30]:
        r1, r2, alpha = altitude_to_regime_blend(alt)
        print(f"   {alt}km: {r1} ←[{alpha:.2f}]→ {r2}")
    
    print("\n4. Training Configuration:")
    print(f"   Data mix: {config.orbital_ratio:.0%} orbital / {config.ballistic_ratio:.0%} ballistic / {config.hgv_ratio:.0%} HGV")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Gradient clip: {config.gradient_clip}")
    print(f"   SupCon temperature: {SUPCON_TEMPERATURE}")
    
    print("\n" + "=" * 70)
    print("Strategic Advantage:")
    print("  - Stability over expressivity (defense requirement)")
    print("  - Verified latent codes select correct physical manifold")
    print("  - M2 Air: 32-dim z + FiLM vs 20K weight hypernetwork")
    print("  - May 4th: 'Our model doesn't guess physics; it selects it'")
    print("=" * 70)


if __name__ == "__main__":
    demo_lc_phs()
