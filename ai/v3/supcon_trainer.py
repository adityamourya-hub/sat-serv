"""
Supervised Contrastive (SupCon) Trainer for LC-PHS

Forces latent space to cluster by physical regime (Orbital/Ballistic/HGV)
for mathematically distinct neighborhoods on the energy manifold.

Three-Stage H100 Training Curriculum:
    Stage 1 (Hours 0-4): Latent Warmup
        - Optimize only SupConTrainer
        - Goal: HGV z distant from Orbital z (Silhouette > 0.8)
    
    Stage 2 (Hours 4-48): Joint Manifold Training
        - Combined loss: L_physics + L_con + L_anchor
        - FiLM learns to reshape energy surface per z region
    
    Stage 3 (Hours 48-72): Symplectic Refinement
        - Low learning rate, high Forest-Ruth symplectic weight
        - Lock in 100Hz stability for M2 Air

Architecture:
    - Anchors: [3, 32] for Orbital/Ballistic/HGV regimes
    - Temperature: τ=0.07 (sharp disentanglement)
    - Batch size: 4096 (large negative sample sets)
    - Hardware: jax.pmap across H100 cluster

Regime Labels:
    0: Altitude > 150km (Pure Orbital)
    1: Altitude 80-150km, Velocity ballistic (Ballistic Re-entry)
    2: Altitude 30-80km, Non-ballistic CL (HGV Skip-Glide)

The Sovereign Result:
    Physical Latent Map with HGV threats as distinct "Red Zone" cluster,
    providing immediate, explainable warning before intercept calculation.

References:
    - SupCon: Khosla et al. "Supervised Contrastive Learning"
    - SimCLR: Chen et al. "A Simple Framework for Contrastive Learning"
    - LC-PHS: Latent-Conditioned Port-Hamiltonian Systems
"""

from typing import Tuple, Dict
import jax
import jax.numpy as jnp
from jax import jit, pmap, vmap
import equinox as eqx
import optax
from functools import partial
import time


class SupConTrainer(eqx.Module):
    """
    Supervised Contrastive Trainer for Physical Regime Disentanglement.
    
    Anchors:
        [0]: Orbital - Vacuum, J2 perturbations
        [1]: Ballistic - Atmospheric entry, drag
        [2]: HGV - Skip-glide, lift/drag coupling
        
    All embeddings normalized to unit hypersphere for cosine similarity.
    """
    
    anchors: jnp.ndarray      # Shape: [3, 32] - regime anchor embeddings
    temperature: float       # τ=0.07 for sharp disentanglement
    
    def __init__(self, key: jax.random.PRNGKey, latent_dim: int = 32, temperature: float = 0.07):
        """
        Initialize 3 distinct anchor points in 32D space.
        
        Args:
            key: JAX random key
            latent_dim: Dimension of latent space (default 32)
            temperature: Temperature τ for contrastive loss (default 0.07)
        """
        # Initialize 3 anchor points
        raw_anchors = jax.random.normal(key, (3, latent_dim))
        
        # L2 Normalize to place anchors on unit hypersphere
        self.anchors = raw_anchors / (jnp.linalg.norm(raw_anchors, axis=1, keepdims=True) + 1e-8)
        self.temperature = temperature
    
    def contrastive_loss(self, latent_batch: jnp.ndarray, labels: jnp.ndarray) -> float:
        """
        Supervised Contrastive Loss (SupCon).
        
        Forces same-regime samples together, different regimes apart.
        
        L_con = -Σ_i (1/|P(i)|) Σ_{p∈P(i)} log(exp(z_i·z_p/τ) / Σ_a exp(z_i·z_a/τ))
        
        Args:
            latent_batch: [batch, 32] embeddings from FiLM encoder
            labels: [batch] integer regime IDs (0=orbital, 1=ballistic, 2=hgv)
            
        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = latent_batch.shape[0]
        
        # 1. Normalize batch embeddings to unit hypersphere
        z = latent_batch / (jnp.linalg.norm(latent_batch, axis=1, keepdims=True) + 1e-8)
        
        # 2. Compute cosine similarity matrix: sim[i,j] = z_i · z_j
        # Shape: [batch, batch]
        similarity_matrix = jnp.matmul(z, z.T) / self.temperature
        
        # 3. Create mask for positive pairs (same regime)
        # labels[:, None] == labels[None, :] creates [batch, batch] boolean matrix
        mask_positives = jnp.equal(labels[:, None], labels[None, :]).astype(jnp.float32)
        
        # Remove self-similarity from positives
        mask_positives = mask_positives - jnp.eye(batch_size)
        
        # 4. Numerical stability: subtract max per row
        logits_max = jnp.max(similarity_matrix, axis=1, keepdims=True)
        logits = similarity_matrix - logits_max
        
        # 5. Compute exp(logits), masking out self-similarity
        exp_logits = jnp.exp(logits) * (1.0 - jnp.eye(batch_size))
        
        # Log-sum-exp for denominator
        log_sum_exp = jnp.log(exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        
        # Log probability: logit - log(sum(exp))
        log_prob = logits - log_sum_exp
        
        # 6. Mean log-likelihood over positive pairs
        # For each sample i, average over all positives p
        num_positives = mask_positives.sum(axis=1)  # |P(i)| for each i
        mean_log_prob_pos = (mask_positives * log_prob).sum(axis=1) / (num_positives + 1e-8)
        
        # Return negative mean (we minimize the negative log-likelihood)
        return -jnp.mean(mean_log_prob_pos)
    
    def anchor_loss(self, latent_batch: jnp.ndarray, labels: jnp.ndarray) -> float:
        """
        Anchor Loss: Forces embeddings toward their class anchor.
        
        Minimizes cosine distance to target anchor (maximizes similarity).
        
        L_anchor = 1 - mean(z · anchor_target)
        
        Args:
            latent_batch: [batch, 32] embeddings
            labels: [batch] regime labels
            
        Returns:
            loss: Scalar anchor loss
        """
        # Normalize batch embeddings
        z = latent_batch / (jnp.linalg.norm(latent_batch, axis=1, keepdims=True) + 1e-8)
        
        # Get target anchors for each sample
        target_anchors = self.anchors[labels]  # [batch, 32]
        
        # Cosine similarity (dot product of normalized vectors)
        cosine_sim = jnp.sum(z * target_anchors, axis=1)
        
        # Minimize cosine distance = maximize similarity
        return 1.0 - jnp.mean(cosine_sim)
    
    def total_contrastive_loss(self, 
                                latent_batch: jnp.ndarray, 
                                labels: jnp.ndarray,
                                lambda_anchor: float = 0.5) -> Tuple[float, Dict]:
        """
        Combined contrastive + anchor loss.
        
        Args:
            latent_batch: [batch, 32] embeddings
            labels: [batch] regime labels
            lambda_anchor: Weight for anchor loss (default 0.5)
            
        Returns:
            (loss_total, metrics_dict)
        """
        loss_con = self.contrastive_loss(latent_batch, labels)
        loss_anchor = self.anchor_loss(latent_batch, labels)
        
        loss_total = loss_con + lambda_anchor * loss_anchor
        
        metrics = {
            "loss_contrastive": float(loss_con),
            "loss_anchor": float(loss_anchor),
            "loss_total": float(loss_total),
        }
        
        return loss_total, metrics
    
    def compute_silhouette_score(self, 
                                  latent_batch: jnp.ndarray, 
                                  labels: jnp.ndarray) -> float:
        """
        Compute Silhouette Score for latent clustering quality.
        
        Score range: [-1, 1]
            +1: Perfect clustering (distinct regimes)
            0: Overlapping clusters
            -1: Incorrect clustering
            
        Target for Stage 1: > 0.8
        
        Args:
            latent_batch: [batch, 32] embeddings
            labels: [batch] regime labels
            
        Returns:
            silhouette_score: Mean silhouette coefficient
        """
        z = latent_batch / (jnp.linalg.norm(latent_batch, axis=1, keepdims=True) + 1e-8)
        batch_size = z.shape[0]
        
        # Compute pairwise distances (1 - cosine similarity)
        similarities = jnp.matmul(z, z.T)
        distances = 1.0 - similarities
        
        silhouette_scores = []
        
        for i in range(batch_size):
            label_i = labels[i]
            
            # a(i): Mean distance to same-class samples
            same_class_mask = (labels == label_i).astype(jnp.float32)
            same_class_mask = same_class_mask.at[i].set(0)  # Exclude self
            
            if same_class_mask.sum() > 0:
                a_i = (distances[i] * same_class_mask).sum() / same_class_mask.sum()
            else:
                a_i = 0.0
            
            # b(i): Mean distance to nearest different-class
            b_i = jnp.inf
            for other_label in [0, 1, 2]:
                if other_label != label_i:
                    other_mask = (labels == other_label).astype(jnp.float32)
                    if other_mask.sum() > 0:
                        mean_dist = (distances[i] * other_mask).sum() / other_mask.sum()
                        b_i = jnp.minimum(b_i, mean_dist)
            
            # Silhouette: (b - a) / max(a, b)
            s_i = (b_i - a_i) / (jnp.maximum(a_i, b_i) + 1e-8)
            silhouette_scores.append(s_i)
        
        return jnp.mean(jnp.array(silhouette_scores))


# =============================================================================
# Three-Stage Training Curriculum
# =============================================================================

class SupConTrainingCurriculum:
    """
    Three-stage training curriculum for H100 cluster.
    
    Stage 1: Latent Warmup (Hours 0-4)
    Stage 2: Joint Manifold Training (Hours 4-48)
    Stage 3: Symplectic Refinement (Hours 48-72)
    """
    
    def __init__(self, batch_size: int = 4096, latent_dim: int = 32):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # Stage configurations
        self.stages = {
            1: {
                "name": "Latent Warmup",
                "hours": (0, 4),
                "lr": 1e-3,
                "loss_weights": {"contrastive": 1.0, "anchor": 1.0, "physics": 0.0, "symplectic": 0.0},
                "target_metric": "silhouette > 0.8",
            },
            2: {
                "name": "Joint Manifold Training",
                "hours": (4, 48),
                "lr": 5e-4,
                "loss_weights": {"contrastive": 0.3, "anchor": 0.3, "physics": 1.0, "symplectic": 0.1},
                "target_metric": "dynamics MSE < 1e-4",
            },
            3: {
                "name": "Symplectic Refinement",
                "hours": (48, 72),
                "lr": 1e-4,
                "loss_weights": {"contrastive": 0.1, "anchor": 0.1, "physics": 1.0, "symplectic": 1.0},
                "target_metric": "symp_residual < 1e-6",
            },
        }
    
    def get_stage_config(self, hour: int) -> Dict:
        """Get configuration for current training hour."""
        for stage_num, config in self.stages.items():
            start, end = config["hours"]
            if start <= hour < end:
                return {"stage": stage_num, **config}
        return {"stage": 3, **self.stages[3]}  # Default to stage 3


# =============================================================================
# H100 Cluster Training with jax.pmap
# =============================================================================

def shard_batch_for_h100(batch_data: jnp.ndarray, 
                          batch_labels: jnp.ndarray,
                          num_devices: int = 8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Shard batch across H100 devices for parallel training.
    
    Batch size 4096 → 512 per device on 8-GPU H100 node.
    
    Args:
        batch_data: [4096, ...] full batch
        batch_labels: [4096] full batch labels
        num_devices: Number of H100 GPUs (default 8)
        
    Returns:
        (sharded_data, sharded_labels) for pmap
    """
    batch_size = batch_data.shape[0]
    per_device = batch_size // num_devices
    
    # Reshape to [num_devices, per_device, ...]
    sharded_data = batch_data.reshape((num_devices, per_device, -1))
    sharded_labels = batch_labels.reshape((num_devices, per_device))
    
    return sharded_data, sharded_labels


@partial(pmap, axis_name="devices")
def parallel_contrastive_step(trainer: SupConTrainer,
                               opt_state,
                               latent_batch: jnp.ndarray,
                               labels: jnp.ndarray,
                               lambda_anchor: float = 0.5):
    """
    Parallel contrastive training step across H100 cluster.
    
    Args:
        trainer: SupConTrainer (replicated across devices)
        opt_state: Optimizer state
        latent_batch: [per_device, 32] sharded batch
        labels: [per_device] sharded labels
        lambda_anchor: Anchor loss weight
        
    Returns:
        Updated trainer, opt_state, loss, metrics
    """
    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(
        lambda t: t.total_contrastive_loss(latent_batch, labels, lambda_anchor),
        has_aux=True
    )(trainer)
    
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    
    # Update (optimizer step handled externally)
    return grads, loss, metrics


# =============================================================================
# Regime Assignment from State
# =============================================================================

def assign_regime_from_state(altitude_km: float, velocity: float, lift_coefficient: float = 0.0) -> int:
    """
    Assign regime label based on orbital state.
    
    Labels:
        0: Altitude > 150km (Pure Orbital)
        1: Altitude 80-150km, Velocity ballistic (Ballistic Re-entry)
        2: Altitude 30-80km, Non-ballistic CL (HGV Skip-Glide)
        
    Args:
        altitude_km: Altitude in kilometers
        velocity: Velocity magnitude in km/s
        lift_coefficient: Lift coefficient (if available from radar)
        
    Returns:
        regime: Integer label (0, 1, or 2)
    """
    if altitude_km > 150.0:
        return 0  # Orbital
    elif altitude_km > 80.0:
        # Check if velocity is ballistic (near-Keplerian) or not
        # Ballistic: ~7.5 km/s at 100km, decreasing
        if velocity > 6.0:
            return 0  # Still effectively orbital
        else:
            return 1  # Ballistic re-entry
    elif altitude_km > 30.0:
        # Distinguish ballistic from HGV by lift coefficient
        if lift_coefficient > 0.1:
            return 2  # HGV with active lift
        else:
            return 1  # Ballistic
    else:
        # Below 30km: usually HGV endurance or terminal
        if lift_coefficient > 0.05:
            return 2  # HGV
        else:
            return 1  # Ballistic terminal


# =============================================================================
# Demo
# =============================================================================

def demo_supcon_trainer():
    """Demonstrate SupCon trainer capabilities."""
    print("=" * 70)
    print("Supervised Contrastive (SupCon) Trainer Demo")
    print("=" * 70)
    
    # Initialize
    key = jax.random.PRNGKey(42)
    trainer = SupConTrainer(key, latent_dim=32, temperature=0.07)
    curriculum = SupConTrainingCurriculum(batch_size=4096)
    
    print("\n1. Configuration:")
    print(f"   Latent dimension: 32")
    print(f"   Temperature τ: 0.07")
    print(f"   Batch size: 4096 (H100 cluster)")
    print(f"   Anchors: [Orbital, Ballistic, HGV] → [3, 32]")
    
    print("\n2. Anchor Embeddings (L2 normalized):")
    for i, regime in enumerate(["Orbital", "Ballistic", "HGV"]):
        norm = float(jnp.linalg.norm(trainer.anchors[i]))
        print(f"   {regime:12s}: ||z|| = {norm:.6f}")
    
    # Create synthetic batch
    print("\n3. Synthetic Batch (4096 samples):")
    key, *subkeys = jax.random.split(key, 4)
    
    # Regime distribution: 40% orbital, 30% ballistic, 30% HGV
    n_orb = 1638
    n_bal = 1229
    n_hgv = 1229
    
    # Generate latent embeddings (simulating FiLM encoder output)
    z_orb = jax.random.normal(subkeys[0], (n_orb, 32)) * 0.1 + trainer.anchors[0]
    z_bal = jax.random.normal(subkeys[1], (n_bal, 32)) * 0.1 + trainer.anchors[1]
    z_hgv = jax.random.normal(subkeys[2], (n_hgv, 32)) * 0.1 + trainer.anchors[2]
    
    latent_batch = jnp.concatenate([z_orb, z_bal, z_hgv], axis=0)
    labels = jnp.concatenate([
        jnp.zeros(n_orb),
        jnp.ones(n_bal),
        2 * jnp.ones(n_hgv)
    ]).astype(jnp.int32)
    
    # Shuffle
    perm = jax.random.permutation(key, 4096)
    latent_batch = latent_batch[perm]
    labels = labels[perm]
    
    print(f"   Orbital: {n_orb} samples")
    print(f"   Ballistic: {n_bal} samples")
    print(f"   HGV: {n_hgv} samples")
    
    # Compute losses
    print("\n4. Contrastive Loss Computation:")
    loss, metrics = trainer.total_contrastive_loss(latent_batch, labels, lambda_anchor=0.5)
    print(f"   L_contrastive: {metrics['loss_contrastive']:.6f}")
    print(f"   L_anchor: {metrics['loss_anchor']:.6f}")
    print(f"   L_total: {metrics['loss_total']:.6f}")
    
    # Compute silhouette score
    print("\n5. Clustering Quality:")
    silhouette = trainer.compute_silhouette_score(latent_batch, labels)
    print(f"   Silhouette Score: {float(silhouette):.4f}")
    print(f"   Target (Stage 1): > 0.8")
    print(f"   Status: {'✅ PASS' if silhouette > 0.8 else '❌ FAIL'}")
    
    # Training curriculum
    print("\n6. Three-Stage Curriculum:")
    for stage_num, config in curriculum.stages.items():
        start, end = config["hours"]
        print(f"\n   Stage {stage_num}: {config['name']} (Hours {start}-{end})")
        print(f"      Learning rate: {config['lr']}")
        print(f"      Loss weights: {config['loss_weights']}")
        print(f"      Target: {config['target_metric']}")
    
    # Regime assignment
    print("\n7. Regime Assignment from State:")
    test_cases = [
        (300.0, 7.8, 0.0),   # 300km, orbital velocity
        (120.0, 6.5, 0.0),   # 120km, re-entry velocity
        (60.0, 5.0, 0.2),    # 60km, HGV with lift
        (40.0, 4.0, 0.05),   # 40km, ambiguous
    ]
    for alt, vel, cl in test_cases:
        regime = assign_regime_from_state(alt, vel, cl)
        regime_name = ["Orbital", "Ballistic", "HGV"][regime]
        print(f"   Alt={alt}km, Vel={vel}km/s, CL={cl} → {regime_name}")
    
    print("\n" + "=" * 70)
    print("Sovereign Result:")
    print("  - HGV threats appear as distinct 'Red Zone' cluster")
    print("  - Physical Latent Map for explainable warnings")
    print("  - Pre-intercept regime identification")
    print("=" * 70)


if __name__ == "__main__":
    demo_supcon_trainer()
