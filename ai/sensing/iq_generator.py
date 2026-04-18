import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import logging

# Configure logging for the sensing module
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("iq_generator")

/**
 * IQGenerator provides a high-fidelity synthetic data engine for ATeena-IQNet.
 * It simulates the electromagnetic physics of LFM chirps under tactical 
 * SNR conditions (0 dB challenge).
 * 
 * <plan_complexity>High</plan_complexity><total_lines_estimate>180</total_lines_estimate>
 */

@jit
def generate_cpi(key, target_state, params):
    """
    Generates a Coherent Processing Interval (CPI) of raw I/Q data.
    
    This function implements the Radar Range Equation and LFM signal model 
    to produce realistic sensor returns for training and validation.
    
    @param key: JAX PRNGKey
    @param target_state: [range_m, velocity_ms, azimuth_rad, elev_rad, rcs_m2]
    @param params: Dictionary containing radar hardware parameters:
                   fs (float): Sampling frequency
                   pulse_width (float): Pulse duration (T)
                   bandwidth (float): LFM bandwidth (B)
                   fc (float): Carrier frequency
                   peak_power (float): Pt
                   gain (float): G
                   k_noise (float): Noise scaling (k * T_sys * B)
    @returns: jnp.ndarray: Complex (num_pulses, num_samples) array
    """
    logger.debug(f"Generating CPI for target at R={target_state[0]}m")
    
    fs = params['fs']
    T = params['pulse_width']
    B = params['bandwidth']
    K = B / T # Chirp rate (Hz/s)
    c = 299792458.0
    num_pulses = params.get('num_pulses', 128)
    pri = params.get('pri', 1e-3) # 1ms Pulse Repetition Interval
    
    num_samples = int(fs * T)
    t = jnp.linspace(0, T, num_samples)
    
    # 1. Physics Calculations
    r0 = target_state[0]
    v = target_state[1]
    rcs = target_state[4]
    lambda_ = c / params['fc']
    
    # 2. SNR Logic (Sovereign Radar Equation)
    # Pr = (Pt * G^2 * lambda^2 * sigma) / ((4 * pi)^3 * R^4)
    # SNR = Pr / (k * T_sys * B)
    pr_numer = params['peak_power'] * (params['gain']**2) * (lambda_**2) * rcs
    pr_denom = ((4 * jnp.pi)**3) * (r0**4)
    pr = pr_numer / pr_denom
    
    snr_linear = pr / params['k_noise']
    amplitude = jnp.sqrt(snr_linear)
    
    # 3. Pulse Generation Loop (vmap across CPI)
    def single_pulse(pulse_idx):
        # Time delay (tau) and Doppler (fd) evolve per pulse
        r_pulse = r0 + v * (pulse_idx * pri)
        tau = 2 * r_pulse / c
        fd = 2 * v / lambda_
        
        # LFM Chirp with phase shift
        # phase = pi * K * (t - tau)^2 + 2 * pi * fd * t
        phase = jnp.pi * K * (t - tau)**2 + 2 * jnp.pi * fd * t
        return amplitude * jnp.exp(1j * phase)
    
    iq_clean = vmap(single_pulse)(jnp.arange(num_pulses))
    
    # 4. Complex AWGN
    noise_key, _ = random.split(key)
    # Noise power is normalized to 1.0 (since SNR is applied to amplitude)
    noise = (random.normal(noise_key, shape=iq_clean.shape) + 
             1j * random.normal(noise_key, shape=iq_clean.shape)) * 0.7071
             
    logger.debug("CPI generation complete.")
    return iq_clean + noise

@jit
def batch_generate_cpi(key, batch_states, params):
    """
    Vectorized batch generation of CPIs for high-throughput training.
    """
    keys = random.split(key, batch_states.shape[0])
    return vmap(lambda k, s: generate_cpi(k, s, params))(keys, batch_states)

if __name__ == "__main__":
    # Internal validation logic
    key = random.PRNGKey(42)
    params = {
        'fs': 100e6, 'pulse_width': 100e-6, 'bandwidth': 10e6, 
        'fc': 3e9, 'peak_power': 1e6, 'gain': 1000, 'k_noise': 1e-12,
        'num_pulses': 128, 'pri': 1e-3
    }
    state = jnp.array([1000e3, 7500.0, 0, 0, 0.1]) # 1000km, 7.5km/s (LEO)
    
    cpi = generate_cpi(key, state, params)
    print(f"Generated CPI shape: {cpi.shape}")
    print(f"Mean Power: {jnp.mean(jnp.abs(cpi)**2)}")
