# frft.py
# JAX-native Discrete Fractional Fourier Transform (FrFT) — Phase 1.2 Refinement
# Implementing Warm-Start & Narrow Search strategy for high-SNR refocusing
# <plan_complexity>Medium</plan_complexity><total_lines_estimate>180</total_lines_estimate>

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

def calculate_pslr(signal: jnp.ndarray) -> float:
    """
    Computes the Peak-to-Sidelobe Ratio (PSLR) of the transform result.
    PSLR = 10 * log10(P_peak / P_sidelobe_max)
    
    This metric is used to determine how well the FrFT has refocused 
    the chirped signal. A high PSLR indicates a successful collapse 
    of the energy into a single range-rate bin.
    
    @param signal: Resulting complex signal in fractional domain
    @returns: PSLR in decibels
    """
    # Entry Point: PSLR Calculation
    mag_sq = jnp.abs(signal)**2
    peak = jnp.max(mag_sq)
    
    # Simple sidelobe max: excludes the peak bin and its immediate neighbors
    peak_idx = jnp.argmax(mag_sq)
    masked = mag_sq.at[jnp.clip(jnp.arange(peak_idx-2, peak_idx+3), 0, mag_sq.shape[0]-1)].set(0.0)
    sidelobe_max = jnp.max(masked) + 1e-12
    
    res = 10 * jnp.log10(peak / sidelobe_max)
    # Exit Point: PSLR value
    return res

def dfrft(signal: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Computes the Discrete Fractional Fourier Transform (DFRFT).
    
    @param signal: Input complex signal (I/Q)
    @param alpha: Fractional order
    @returns: Transformed signal
    """
    N = signal.shape[0]
    phi = alpha * jnp.pi / 2.0
    cot_phi = 1.0 / jnp.tan(phi + 1e-12)
    # Simplified chirp-z transform approximation for Phase 1.2
    shv = jnp.arange(N)
    q_phase = jnp.exp(1j * 0.5 * cot_phi * (shv**2) / N)
    res = q_phase * jnp.fft.fft(q_phase * signal)
    return res

@jit
def get_analytical_alpha(range_rate_kms: float, carrier_freq_hz: float, fs: float) -> float:
    """
    Computes the theoretical rotation order alpha based on the chirp rate 
    implied by the radial velocity (Doppler derivative).
    
    Formula: alpha = arctan(2 * r_dot / (lambda * fs))
    
    @param range_rate_kms: Radial velocity (km/s)
    @param carrier_freq_hz: Radar carrier frequency (Hz)
    @param fs: Sampling frequency (Hz)
    @returns: Analytical rotation order alpha
    """
    c = 299792.458
    lambda_ = c / carrier_freq_hz
    # Scale the range rate into the fractional-order domain
    # This acts as the baseline for our "Warm-Start" search
    return jnp.arctan2(2.0 * range_rate_kms, lambda_ * fs / 1e3) + 1.0

@jit
def get_optimal_alpha_refined(signal: jnp.ndarray, range_rate_kms: float, carrier_freq_hz: float) -> float:
    """
    Executes a "Warm-Start & Narrow Search" to find the optimal rotation angle alpha.
    
    1. Seed alpha using the ANALYTICAL formula based on TLE range-rate.
    2. Search 100 steps over a restricted +/- 0.05 rad window.
    3. Select alpha that maximizes PSLR to refocus the hypersonic chirp.
    
    @param signal: Input radar return (I/Q)
    @param range_rate_kms: Seed range-rate from TLE ephemeris
    @param carrier_freq_hz: Carrier frequency (Hz)
    @returns: Refined optimal rotation angle alpha
    """
    # Entry Point: Narrow Alpha Search
    fs = 100e6 # Standard 100MHz baseline
    seed_alpha = get_analytical_alpha(range_rate_kms, carrier_freq_hz, fs)
    
    # 2. Narrow Search Grid (100 steps over 0.1 rad range)
    # This restricted search ensures we hit the 50ms p99 latency target
    alpha_grid = jnp.linspace(seed_alpha - 0.05, seed_alpha + 0.05, 100)
    
    @vmap
    def evaluate_alpha(a):
        out = dfrft(signal, a)
        return calculate_pslr(out)
    
    pslr_scores = evaluate_alpha(alpha_grid)
    best_idx = jnp.argmax(pslr_scores)
    optimal_alpha = alpha_grid[best_idx]
    
    # Exit Point: Refined Alpha
    return optimal_alpha

@jit
def process_iq_stream(iq_batch: jnp.ndarray, range_rates: jnp.ndarray) -> jnp.ndarray:
    """
    Batched processing of radar returns using the refined FrFT strategy.
    
    @param iq_batch: (Batch, N) I/Q signals
    @param range_rates: (Batch,) Seed range-rates
    @returns: (Batch, N) refocused signals
    """
    # Entry Point: IQ Stream Processing
    print(f"DEBUG: Processing IQ batch of size {iq_batch.shape[0]}")
    
    # In a full vmap, we might do per-channel optimal search
    # For now, we vmap our refined search logic
    carrier_freq = 3.0e9 # 3 GHz
    
    @vmap
    def single_proc(sig, r_rate):
        best_a = get_optimal_alpha_refined(sig, r_rate, carrier_freq)
        return dfrft(sig, best_a)
        
    res = single_proc(iq_batch, range_rates)
    # Exit Point: Processed batch
    return res

if __name__ == "__main__":
    print("ATeena-IQNet: FrFT Refined Search Module Initialized.")
