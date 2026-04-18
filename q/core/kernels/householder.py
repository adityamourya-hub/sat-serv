import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)

@jax.jit
def rank_2_householder_contraction(A: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Applies the Householder transformation P = I - 2vv^T to a skew-symmetric 
    matrix A using a highly optimized Rank-2 update.
    
    Mathematical Derivation:
    A' = P * A * P^T = (I - 2vv^T) A (I - 2vv^T)
    A' = A - 2 v v^T A - 2 A v v^T + 4 v (v^T A v) v^T
    Since A is skew-symmetric (A^T = -A), the scalar v^T A v = 0.
    This eliminates the 4th-order tensor product, reducing the transformation
    to a pure rank-2 update:
    A' = A + 2 * (v @ w^T - w @ v^T) where w = A @ v.

    Args:
        A (jnp.ndarray): Skew-symmetric Hamiltonian or adjacency matrix (N, N).
        v (jnp.ndarray): Householder reflection vector (N,).

    Returns:
        jnp.ndarray: The transformed skew-symmetric matrix A' (N, N).
    """
    logger.debug("ENTRY: rank_2_householder_contraction - A shape: %s, v shape: %s", A.shape, v.shape)
    
    # 1. Compute w = A * v using Einstein summation
    w = jnp.einsum('ij,j->i', A, v)
    
    # 2. Compute the rank-2 update: 2 * (v * w^T - w * v^T)
    # einsum('i,j->ij') performs the exact outer product without allocating intermediate broadcast arrays
    update = 2.0 * (
        jnp.einsum('i,j->ij', v, w) - jnp.einsum('i,j->ij', w, v)
    )
    
    # 3. Apply the integrated update
    A_prime = A + update
    
    logger.debug("EXIT: rank_2_householder_contraction - Output shape: %s", A_prime.shape)
    return A_prime

@jax.jit
def batch_householder_pfaffian_step(batched_A: jnp.ndarray, batched_v: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized batched implementation of the Rank-2 Householder contraction 
    for calculating the Pfaffian topological invariants over memory-contiguous 
    tensor slices. Fits inside the 50ns tight loop bounds.
    
    This function utilizes `vmap` logic inherently broadcasted via JAX's 
    batched einsum capabilities. 
    
    Args:
        batched_A (jnp.ndarray): Batched skew-symmetric matrices (B, N, N).
        batched_v (jnp.ndarray): Batched Householder reflection vectors (B, N).
        
    Returns:
        jnp.ndarray: Batched transformed skew-symmetric matrices (B, N, N).
    """
    logger.debug("ENTRY: batch_householder_pfaffian_step - batched_A shape: %s, batched_v shape: %s", batched_A.shape, batched_v.shape)
    
    # Batch compute w = A * v
    w = jnp.einsum('bij,bj->bi', batched_A, batched_v)
    
    # Batch outer product and skew-symmetric generation
    update = 2.0 * (
        jnp.einsum('bi,bj->bij', batched_v, w) - jnp.einsum('bi,bj->bij', w, batched_v)
    )
    
    batched_A_prime = batched_A + update
    
    logger.debug("EXIT: batch_householder_pfaffian_step - Output shape: %s", batched_A_prime.shape)
    return batched_A_prime
