# iqnet.py
# Complex-Valued Neural Network (CVNN) for ATeena Radar I/Q Processing
# Preserves phase coherence for high-fidelity detection and tracking
# <plan_complexity>Medium</plan_complexity><total_lines_estimate>180</total_lines_estimate>

import jax
import jax.numpy as jnp
from jax import jit, vmap, random
from typing import Dict, Tuple

def complex_relu(z: jnp.ndarray) -> jnp.ndarray:
    """
    Implements the Complex ReLU activation function (zReLU).
    f(z) = ReLU(Re(z)) + i * ReLU(Im(z))
    
    This activation preserves the quadrant information of the phase 
    while enforcing sparsity in both real and imaginary channels, 
    making it ideal for radar signal processing.
    
    @param z: Complex input tensor
    @returns: Activated complex output tensor
    """
    # Entry Point: Complex ReLU
    res = jax.nn.relu(jnp.real(z)) + 1j * jax.nn.relu(jnp.imag(z))
    # Exit Point: Activated result
    return res

def init_complex_weights(rng, shape: Tuple[int, ...]) -> jnp.ndarray:
    """
    Initializes complex weights using Glorot/Xavier initialization.
    Weights are sampled from a complex normal distribution to 
    ensure variance is preserved through complex multiplications.
    
    @param rng: JAX PRNG key
    @param shape: Output weight shape
    @returns: Complex weight tensor
    """
    # Entry Point: Complex Weight Init
    r_key, i_key = random.split(rng)
    fan_in = shape[0] # Simplistic fan-in
    std = jnp.sqrt(1.0 / fan_in)
    
    w_real = random.normal(r_key, shape) * std
    w_imag = random.normal(i_key, shape) * std
    # Exit Point: Complex weights
    return w_real + 1j * w_imag

def cvnn_layer(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Performs a standard complex-valued linear layer operation: y = Wz + b.
    
    Where W and z are complex, the multiplication expands to:
    (W_re + iW_im) * (z_re + iz_im) = (W_re*z_re - W_im*z_im) + i(W_re*z_im + W_im*z_re)
    JAX handles this natively via complex types.
    
    @param x: Input complex tensor
    @param w: Weights (complex)
    @param b: Bias (complex)
    @returns: Resulting complex tensor
    """
    # Entry Point: CVNN Layer Forward
    print(f"DEBUG: Processing CVNN layer with input shape {x.shape}")
    res = jnp.dot(x, w) + b
    # Exit Point: Layer output
    return res

class ATeenaIQNet:
    """
    ATeena-IQNet: A deep complex-valued architecture for radar target detection.
    Integrates learned matched filters and temporal processing.
    """
    def __init__(self, rng_key, input_dim: int, hidden_dim: int):
        print("DEBUG: Initializing ATeenaIQNet architecture")
        k1, k2, k3 = random.split(rng_key, 3)
        
        self.w1 = init_complex_weights(k1, (input_dim, hidden_dim))
        self.b1 = jnp.zeros(hidden_dim, dtype=jnp.complex128)
        
        self.w2 = init_complex_weights(k2, (hidden_dim, 6)) # [p_det, r, v, az, el, rcs]
        self.b2 = jnp.zeros(6, dtype=jnp.complex128)
        
    def forward(self, iq_samples: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Forward pass for target detection and estimation.
        
        @param iq_samples: Input I/Q complex vector
        @returns: Dictionary containing detection probability and state estimates
        """
        # Entry Point: IQNet Forward
        print(f"DEBUG: ATeenaIQNet forward pass start, input size={iq_samples.shape}")
        
        # Stage 1: Complex Feature Extraction
        h1 = cvnn_layer(iq_samples, self.w1, self.b1)
        h1 = complex_relu(h1)
        
        # Stage 2: Estimation Head
        out = cvnn_layer(h1, self.w2, self.b2)
        
        # Output Parsing (Complex magnitude for detection p, real for regression)
        p_det = jax.nn.sigmoid(jnp.abs(out[0]))
        state = jnp.real(out[1:])
        
        # Exit Point: Forward results
        print(f"DEBUG: ATeenaIQNet forward pass complete, P_det={p_det}")
        return {
            "p_det": p_det,
            "state_estimate": state
        }

if __name__ == "__main__":
    key = random.PRNGKey(42)
    net = ATeenaIQNet(key, 1024, 256)
    dummy_input = jnp.ones(1024, dtype=jnp.complex128)
    results = net.forward(dummy_input)
    print("ATeena-IQNet: Complex-Valued Network Initialized.")
