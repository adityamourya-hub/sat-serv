#!/usr/bin/env python3
"""
SupCon Anchor Loop for Learning New Latents

Implements the Supervised Contrastive (SupCon) anchor loop for learning
new latent codes when zero-day regimes are discovered via IMM expansion.

When IMM expands from K → K+1 modes, the new mode needs a latent anchor
point in the 32-dim latent space. This loop:
1. Collects trajectory samples from the new regime
2. Computes SupCon embeddings
3. Updates the REGIME_ANCHORS dictionary with the new latent code
4. Ensures the new anchor is well-separated from existing anchors

This enables the Hypernetwork Atlas to generate task-specific DPHNO weights
for the zero-day regime without retraining from scratch.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import equinox as eqx
import optax
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from port_hamiltonian import (
    PortHamiltonianSystem,
    supervised_contrastive_loss,
    interpolate_latent_codes,
)


# =============================================================================
# Configuration
# =============================================================================

LATENT_DIM = 32
SUPCON_TEMPERATURE = 0.07
ANCHOR_LEARNING_RATE = 1e-3
ANCHOR_EPOCHS = 100
ANCHOR_SEPARATION_THRESHOLD = 0.5  # Minimum cosine similarity separation


# =============================================================================
# Anchor Loop State
# =============================================================================

@dataclass
class AnchorLoopState:
    """State for the SupCon anchor learning loop."""
    
    # Current regime anchors (learned)
    anchors: Dict[str, jnp.ndarray]
    
    # New regime data (to learn anchor for)
    new_regime_name: str
    new_regime_trajectories: jnp.ndarray  # [N, T, state_dim]
    new_regime_labels: jnp.ndarray  # [N] (all same label for new regime)
    
    # Encoder for generating latents
    encoder: eqx.Module
    
    # Optimizer state
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    
    # Training metrics
    epoch: int
    loss_history: list


# =============================================================================
# Anchor Learning Functions
# =============================================================================

def initialize_anchor_loop(
    existing_anchors: Dict[str, jnp.ndarray],
    new_regime_name: str,
    new_trajectories: jnp.ndarray,
    encoder: eqx.Module,
) -> AnchorLoopState:
    """
    Initialize the anchor learning loop for a new regime.
    
    Args:
        existing_anchors: Current regime anchors (orbital, ballistic, hgv)
        new_regime_name: Name of the new regime (e.g., "zero_day_hgv")
        new_trajectories: Trajectory samples from the new regime [N, T, state_dim]
        encoder: Encoder network for generating latents
    
    Returns:
        Initialized AnchorLoopState
    """
    # Create labels for new regime (all same label)
    new_labels = jnp.ones(len(new_trajectories))
    
    # Initialize optimizer
    optimizer = optax.adamw(learning_rate=ANCHOR_LEARNING_RATE)
    opt_state = optimizer.init(encoder)
    
    return AnchorLoopState(
        anchors=existing_anchors,
        new_regime_name=new_regime_name,
        new_regime_trajectories=new_trajectories,
        new_regime_labels=new_labels,
        encoder=encoder,
        optimizer=optimizer,
        opt_state=opt_state,
        epoch=0,
        loss_history=[],
    )


@jit
def anchor_learning_step(
    state: AnchorLoopState,
    batch: jnp.ndarray,
    batch_labels: jnp.ndarray,
) -> Tuple[AnchorLoopState, float]:
    """
    Single step of anchor learning.
    
    Args:
        state: Current anchor loop state
        batch: Batch of trajectories [B, T, state_dim]
        batch_labels: Batch labels [B] (all same for new regime)
    
    Returns:
        (updated_state, loss)
    """
    # Encode trajectories to get latents
    z_codes = state.encoder(batch)
    
    # Compute SupCon loss with all anchors
    # Combine new regime latents with existing anchors
    all_latents = jnp.concatenate([z_codes], axis=0)
    all_labels = batch_labels
    
    # Add existing anchors to the batch for separation
    for anchor_name, anchor_z in state.anchors.items():
        all_latents = jnp.concatenate([all_latents, anchor_z[None, :]], axis=0)
        # Use different label for each existing anchor
        anchor_label = jnp.array([-1.0])  # Negative label for existing anchors
        all_labels = jnp.concatenate([all_labels, anchor_label], axis=0)
    
    # Compute SupCon loss
    loss = supervised_contrastive_loss(all_latents, all_labels, SUPCON_TEMPERATURE)
    
    # Compute gradients
    grads = grad(lambda enc: supervised_contrastive_loss(
        enc(batch), batch_labels, SUPCON_TEMPERATURE
    ))(state.encoder)
    
    # Update encoder
    updates, state.opt_state = state.optimizer.update(grads, state.opt_state)
    state.encoder = eqx.apply_updates(state.encoder, updates)
    
    return state, loss


def compute_anchor_separation(
    new_anchor: jnp.ndarray,
    existing_anchors: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
    """
    Compute cosine similarity separation between new anchor and existing anchors.
    
    Args:
        new_anchor: New latent code [LATENT_DIM]
        existing_anchors: Dictionary of existing anchors
    
    Returns:
        Dictionary of separation metrics
    """
    separations = {}
    for name, anchor in existing_anchors.items():
        # Cosine similarity
        dot = jnp.dot(new_anchor, anchor)
        norm_new = jnp.linalg.norm(new_anchor)
        norm_anchor = jnp.linalg.norm(anchor)
        cosine_sim = dot / (norm_new * norm_anchor + 1e-8)
        
        # Separation = 1 - cosine similarity
        separations[name] = float(1.0 - cosine_sim)
    
    return separations


def run_anchor_loop(
    state: AnchorLoopState,
    num_epochs: int = ANCHOR_EPOCHS,
    batch_size: int = 32,
) -> Tuple[Dict[str, jnp.ndarray], AnchorLoopState]:
    """
    Run the full anchor learning loop.
    
    Args:
        state: Initialized anchor loop state
        num_epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        (updated_anchors, final_state)
    """
    print("=" * 70)
    print(f"SUPCON ANCHOR LOOP: Learning latent for {state.new_regime_name}")
    print("=" * 70)
    
    n_samples = len(state.new_regime_trajectories)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle and batch
        indices = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch = state.new_regime_trajectories[batch_idx]
            batch_labels = state.new_regime_labels[batch_idx]
            
            state, loss = anchor_learning_step(state, batch, batch_labels)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        state.loss_history.append(avg_loss)
        state.epoch = epoch + 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    # Compute final anchor as mean of encoded trajectories
    all_latents = state.encoder(state.new_regime_trajectories)
    new_anchor = jnp.mean(all_latents, axis=0)
    
    # Check separation from existing anchors
    separations = compute_anchor_separation(new_anchor, state.anchors)
    print(f"\nAnchor Separation Metrics:")
    for name, sep in separations.items():
        print(f"  {new_anchor[:5]} vs {name}: {sep:.3f}")
    
    # Validate separation
    min_separation = min(separations.values())
    if min_separation < ANCHOR_SEPARATION_THRESHOLD:
        print(f"WARNING: Minimum separation {min_separation:.3f} < threshold {ANCHOR_SEPARATION_THRESHOLD}")
        print("  New anchor may not be well-separated from existing anchors")
    
    # Add new anchor to dictionary
    updated_anchors = state.anchors.copy()
    updated_anchors[state.new_regime_name] = new_anchor
    
    print(f"\nNew anchor '{state.new_regime_name}' added to REGIME_ANCHORS")
    print(f"Anchor norm: {jnp.linalg.norm(new_anchor):.3f}")
    print("=" * 70)
    
    return updated_anchors, state


# =============================================================================
# Integration with IMM Expansion
# =============================================================================

def on_imm_expansion(
    old_anchors: Dict[str, jnp.ndarray],
    new_mode_index: int,
    new_mode_trajectories: jnp.ndarray,
    encoder: eqx.Module,
) -> Dict[str, jnp.ndarray]:
    """
    Callback function triggered when IMM expands modes (K → K+1).
    
    This function should be called immediately after IMM.spawnZeroDayMode
    to learn a latent anchor for the new zero-day regime.
    
    Args:
        old_anchors: Existing regime anchors before expansion
        new_mode_index: Index of the new mode in IMM
        new_mode_trajectories: Trajectory samples from the new regime
        encoder: Encoder network for generating latents
    
    Returns:
        Updated anchors dictionary including new regime anchor
    """
    new_regime_name = f"zero_day_mode_{new_mode_index}"
    
    print(f"\n[IMM EXPANSION] Mode {new_mode_index} detected")
    print(f"  Triggering SupCon anchor loop for '{new_regime_name}'")
    print(f"  Trajectory samples: {len(new_mode_trajectories)}")
    
    # Initialize anchor loop
    state = initialize_anchor_loop(
        existing_anchors=old_anchors,
        new_regime_name=new_regime_name,
        new_trajectories=new_mode_trajectories,
        encoder=encoder,
    )
    
    # Run anchor learning
    updated_anchors, _ = run_anchor_loop(state, num_epochs=ANCHOR_EPOCHS, batch_size=32)
    
    return updated_anchors


# =============================================================================
# Demo
# =============================================================================

def demo_anchor_loop():
    """Demonstrate the SupCon anchor loop."""
    print("=" * 70)
    print("SUPCON ANCHOR LOOP DEMO")
    print("=" * 70)
    
    # Create dummy encoder
    class DummyEncoder(eqx.Module):
        @eqx.filter_jit
        def __call__(self, x):
            # Simple linear projection to latent space
            W = jnp.eye(6, LATENT_DIM)
            return x @ W
    
    encoder = DummyEncoder()
    
    # Existing anchors
    existing_anchors = {
        "orbital": jnp.zeros(LATENT_DIM),
        "ballistic": jnp.ones(LATENT_DIM) * 0.5,
        "hgv": jnp.ones(LATENT_DIM) * -0.5,
    }
    
    # Generate dummy trajectories for new regime
    key = jax.random.PRNGKey(42)
    new_trajectories = jax.random.normal(key, (100, 10, 6))  # 100 trajectories, 10 steps each
    
    # Run anchor loop
    updated_anchors, _ = run_anchor_loop(
        state=initialize_anchor_loop(
            existing_anchors=existing_anchors,
            new_regime_name="zero_day_demo",
            new_trajectories=new_trajectories,
            encoder=encoder,
        ),
        num_epochs=20,  # Shorter for demo
        batch_size=16,
    )
    
    print("\nFinal Anchors:")
    for name, anchor in updated_anchors.items():
        print(f"  {name}: norm={jnp.linalg.norm(anchor):.3f}, mean={jnp.mean(anchor):.3f}")


if __name__ == "__main__":
    demo_anchor_loop()
