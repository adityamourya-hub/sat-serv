"""
ATeena Multi-Object Tracking Radar Simulator
Generates synthetic IQ data for training ATeena-IQNet

Based on: Technical Research Bridge Analysis Gap 1
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Physical constants
C = 3e8  # Speed of light (m/s)
K_B = 1.38e-23  # Boltzmann constant (J/K)

# ATeena radar parameters (S-band phased array)
ATEENA_DEFAULT_PARAMS = {
    'fc_hz': 3e9,              # Carrier frequency (3 GHz, S-band)
    'wavelength_m': 0.1,      # λ = c/fc
    'bandwidth_hz': 100e6,    # 100 MHz chirp bandwidth
    'pulse_width_us': 100,    # 100 μs pulse
    'pri_ms': 1.0,            # 1 ms pulse repetition interval
    'n_pulses': 128,          # Pulses per CPI
    'n_range_bins': 1024,     # Range FFT size
    'n_elements': 64,         # Phased array elements
    'element_spacing_m': 0.05, # λ/2 spacing
    'transmit_power_w': 100e3, # 100 kW peak power
    'transmit_gain_db': 35,   # 35 dBi transmit gain
    'receive_gain_db': 35,    # 35 dBi per element
    'noise_figure_db': 3.0,   # 3 dB receiver noise figure
    'system_temp_k': 290,     # 290K system temperature
}


class ATeenaSimulator:
    """
    High-fidelity simulator for ATeena phased-array radar.
    Generates raw IQ data for training neural detection networks.
    """
    
    def __init__(self, params: Optional[Dict] = None, seed: Optional[int] = None):
        self.params = params or ATEENA_DEFAULT_PARAMS.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        # Precompute constants
        self.wavelength = C / self.params['fc_hz']
        self.range_resolution = C / (2 * self.params['bandwidth_hz'])
        self.pulse_width_sec = self.params['pulse_width_us'] * 1e-6
        self.pri_sec = self.params['pri_ms'] * 1e-3
        
        logger.info(f"ATeena simulator initialized")
        logger.info(f"  Frequency: {self.params['fc_hz']/1e9:.1f} GHz")
        logger.info(f"  Bandwidth: {self.params['bandwidth_hz']/1e6:.0f} MHz")
        logger.info(f"  Range resolution: {self.range_resolution:.1f} m")
    
    def simulate_cpi(self, 
                     targets: List[Dict],
                     noise_figure_db: Optional[float] = None,
                     add_clutter: bool = True,
                     add_jamming: bool = False) -> np.ndarray:
        """
        Simulate one Coherent Processing Interval (CPI).
        
        Args:
            targets: List of target dictionaries with keys:
                - range_km: Distance to target (km)
                - velocity_kms: Radial velocity (km/s, positive = away)
                - azimuth_deg: Azimuth angle (degrees)
                - elevation_deg: Elevation angle (degrees)
                - rcs_m2: Radar cross-section (m²)
            noise_figure_db: Receiver noise figure (dB). If None, use default.
            add_clutter: Add ground clutter returns
            add_jamming: Add jamming interference
            
        Returns:
            iq_data: Complex array [n_pulses, n_range_bins, n_elements]
        """
        n_p = self.params['n_pulses']
        n_r = self.params['n_range_bins']
        n_e = self.params['n_elements']
        
        # Initialize IQ array
        iq = np.zeros((n_p, n_r, n_e), dtype=np.complex128)
        
        # Add target returns
        for target in targets:
            target_iq = self._generate_target_return(target)
            iq += target_iq
        
        # Add ground clutter if requested
        if add_clutter:
            clutter = self._generate_clutter()
            iq += clutter
        
        # Add thermal noise
        noise_fig = noise_figure_db if noise_figure_db is not None else self.params['noise_figure_db']
        noise = self._generate_thermal_noise(noise_fig)
        iq += noise
        
        # Add jamming if requested
        if add_jamming:
            jamming = self._generate_jamming()
            iq += jamming
        
        return iq.astype(np.complex64)
    
    def _generate_target_return(self, target: Dict) -> np.ndarray:
        """Generate IQ return for a single target."""
        n_p = self.params['n_pulses']
        n_r = self.params['n_range_bins']
        n_e = self.params['n_elements']
        
        # Extract target parameters
        range_km = target['range_km']
        velocity_kms = target['velocity_kms']
        azimuth_deg = target.get('azimuth_deg', 0.0)
        elevation_deg = target.get('elevation_deg', 0.0)
        rcs_m2 = target.get('rcs_m2', 1.0)
        
        # Round-trip time delay (Relative to window start)
        # Assuming a range-gated receiver starting at window_start_km
        window_start_km = target.get('window_start_km', range_km - 0.5) 
        tau = 2 * (range_km - window_start_km) * 1000 / C 
        range_bin = int(tau * self.params['bandwidth_hz'])
        
        if range_bin >= n_r or range_bin < 0:
            # Target outside range window
            return np.zeros((n_p, n_r, n_e), dtype=np.complex128)
        
        # Doppler frequency
        fd = 2 * velocity_kms * 1000 / self.wavelength  # Hz
        
        # Radar equation: received power
        # P_r = P_t * G_t * G_r * λ² * σ / ((4π)³ * R⁴ * L)
        Pt = self.params['transmit_power_w']
        Gt = 10 ** (self.params['transmit_gain_db'] / 10)
        Gr = 10 ** (self.params['receive_gain_db'] / 10)
        R = range_km * 1000  # meters
        
        received_power = (Pt * Gt * Gr * self.wavelength**2 * rcs_m2) / \
                        ((4 * np.pi)**3 * R**4)
        
        # Amplitude (voltage)
        amplitude = np.sqrt(received_power)
        
        # Generate steering vector once
        d = self.params['element_spacing_m']
        theta = np.radians(azimuth_deg)
        phi = np.radians(elevation_deg)
        element_indices = np.arange(n_e)
        steering = np.exp(-1j * 2 * np.pi * d * element_indices * np.sin(theta) * np.cos(phi) / self.wavelength)
        
        # Vectorized pulse generation
        t_slow = np.arange(n_p) * self.pri_sec
        doppler_phase = np.exp(1j * 2 * np.pi * fd * t_slow)
        
        # Coherence: One random phase per CPI per target
        target_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        
        # Combined signal: [n_p, n_e]
        pulse_returns = amplitude * doppler_phase[:, None] * target_phase * steering[None, :]
        
        target_iq = np.zeros((n_p, n_r, n_e), dtype=np.complex128)
        target_iq[:, range_bin, :] = pulse_returns
        
        return target_iq
    
    def _generate_thermal_noise(self, noise_figure_db: float) -> np.ndarray:
        """Generate thermal noise (additive white Gaussian noise)."""
        n_p = self.params['n_pulses']
        n_r = self.params['n_range_bins']
        n_e = self.params['n_elements']
        
        # Noise power: N = k_B * T_sys * B
        T_sys = self.params['system_temp_k'] * (10 ** (noise_figure_db / 10))
        bandwidth = self.params['bandwidth_hz']
        noise_power = K_B * T_sys * bandwidth
        
        # Complex AWGN
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), (n_p, n_r, n_e))
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), (n_p, n_r, n_e))
        noise = noise_real + 1j * noise_imag
        
        return noise.astype(np.complex128)
    
    def _generate_clutter(self) -> np.ndarray:
        """Generate ground clutter returns (vectorized)."""
        n_p, n_r, n_e = self.params['n_pulses'], self.params['n_range_bins'], self.params['n_elements']
        clutter = np.zeros((n_p, n_r, n_e), dtype=np.complex128)
        
        # Clutter in first 10% range bins
        c_r = n_r // 10
        clutter_power = 1e-6
        
        # Generate random amplitudes and phases for the clutter block
        # [n_p, c_r, n_e]
        shape = (n_p, c_r, n_e)
        amp = np.sqrt(clutter_power) * np.random.randn(*shape)
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, shape))
        
        # Low Doppler modulation: stationary ground
        low_doppler = np.exp(1j * np.random.uniform(0, 0.05, (n_p, 1, 1)))
        
        clutter[:, :c_r, :] = amp * phase * low_doppler
        return clutter
    
    def _generate_jamming(self) -> np.ndarray:
        """Generate jamming interference (simplified barrage jammer)."""
        n_p = self.params['n_pulses']
        n_r = self.params['n_range_bins']
        n_e = self.params['n_elements']
        
        # Jammer power (strong, broadband)
        jammer_power = 1e-4
        
        jamming_real = np.random.normal(0, np.sqrt(jammer_power/2), (n_p, n_r, n_e))
        jamming_imag = np.random.normal(0, np.sqrt(jammer_power/2), (n_p, n_r, n_e))
        
        return (jamming_real + 1j * jamming_imag).astype(np.complex128)
    
    def add_range_migration(self, iq_data: np.ndarray, velocity_kms: float) -> np.ndarray:
        """
        Add range migration effect for high-velocity targets.
        Target moves between range bins during CPI.
        """
        n_p = iq_data.shape[0]
        range_shift_per_pulse = velocity_kms * 1000 * self.pri_sec / self.range_resolution
        
        migrated = np.zeros_like(iq_data)
        for pulse in range(n_p):
            shift = int(pulse * range_shift_per_pulse)
            migrated[pulse] = np.roll(iq_data[pulse], shift, axis=0)
        
        return migrated
    
    def compute_snr(self, iq_data: np.ndarray, target_range_bin: int, 
                    target_pulse: Optional[int] = None) -> float:
        """
        Compute SNR for a target at given range bin.
        
        Returns:
            snr_db: Signal-to-noise ratio in dB
        """
        # Signal power (target cell)
        if target_pulse is not None:
            signal_power = np.mean(np.abs(iq_data[target_pulse, target_range_bin, :])**2)
        else:
            signal_power = np.mean(np.abs(iq_data[:, target_range_bin, :])**2)
        
        # Noise power (adjacent cells)
        noise_bins = [target_range_bin - 2, target_range_bin + 2]
        noise_bins = [b for b in noise_bins if 0 <= b < iq_data.shape[1]]
        
        if noise_bins:
            noise_powers = [np.mean(np.abs(iq_data[:, b, :])**2) for b in noise_bins]
            noise_power = np.mean(noise_powers)
        else:
            noise_power = 1e-10  # Default small value
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db


def generate_random_scenario(min_targets: int = 1, 
                              max_targets: int = 50,
                              snr_range: Tuple[float, float] = (-10, 30)) -> Tuple[List[Dict], float]:
    """
    Generate a randomized training scenario with domain randomization.
    
    Returns:
        targets: List of target dictionaries
        noise_figure_db: Randomized noise figure
    """
    n_targets = np.random.randint(min_targets, max_targets + 1)
    
    targets = []
    base_range = np.random.uniform(200, 2000) # LEO altitude
    for _ in range(n_targets):
        # Target must be within the ~1.5km range window
        targets.append({
            'range_km': base_range + np.random.uniform(0, 1.2), # relative to window
            'window_start_km': base_range,
            'velocity_kms': np.random.uniform(7.0, 8.0),
            'azimuth_deg': np.random.uniform(0, 360),
            'elevation_deg': np.random.uniform(10, 90),
            'rcs_m2': 10 ** np.random.uniform(-2, 2)
        })
    
    # Randomize noise figure to vary SNR
    noise_figure_db = np.random.uniform(1.0, 6.0)
    
    return targets, noise_figure_db


def create_training_dataset(output_file: str,
                            n_samples: int = 100000,
                            n_pulses: int = 128,
                            n_range_bins: int = 1024,
                            n_elements: int = 64,
                            seed: int = 42):
    """
    Create large synthetic training dataset for ATeena-IQNet.
    
    Args:
        output_file: Path to save dataset (HDF5 format)
        n_samples: Number of CPI samples to generate
        n_pulses: Pulses per CPI
        n_range_bins: Range bins
        n_elements: Array elements
        seed: Random seed for reproducibility
    """
    import h5py
    
    logger.info(f"Creating training dataset: {n_samples} samples")
    
    # Initialize simulator
    params = ATEENA_DEFAULT_PARAMS.copy()
    params['n_pulses'] = n_pulses
    params['n_range_bins'] = n_range_bins
    params['n_elements'] = n_elements
    
    sim = ATeenaSimulator(params, seed=seed)
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        iq_dataset = f.create_dataset('iq_data', 
                                      shape=(n_samples, n_pulses, n_range_bins, n_elements, 2),
                                      dtype=np.float32,
                                      chunks=(1, n_pulses, n_range_bins, n_elements, 2),
                                      compression='gzip')
        
        labels_dataset = f.create_dataset('labels',
                                          shape=(n_samples, 50, 6),  # Max 50 targets, 6 params each
                                          dtype=np.float32,
                                          fillvalue=-1)
        
        n_targets_dataset = f.create_dataset('n_targets',
                                             shape=(n_samples,),
                                             dtype=np.int32)
        
        snr_dataset = f.create_dataset('snr_db',
                                       shape=(n_samples,),
                                       dtype=np.float32)
        
        # Generate samples
        for i in range(n_samples):
            if i % 1000 == 0:
                logger.info(f"Generated {i}/{n_samples} samples")
            
            # Random scenario
            targets, noise_fig = generate_random_scenario()
            
            # Simulate CPI
            iq = sim.simulate_cpi(targets, noise_figure_db=noise_fig, 
                                  add_clutter=True, add_jamming=False)
            
            # Store IQ (split into I and Q channels)
            iq_dataset[i, :, :, :, 0] = iq.real.astype(np.float32)
            iq_dataset[i, :, :, :, 1] = iq.imag.astype(np.float32)
            
            # Store labels (pad to max 50 targets)
            n_t = min(len(targets), 50)
            labels = np.full((50, 6), -1.0, dtype=np.float32)
            for j, tgt in enumerate(targets[:n_t]):
                labels[j] = [
                    tgt['range_km'],
                    tgt['velocity_kms'],
                    tgt['azimuth_deg'],
                    tgt['elevation_deg'],
                    tgt['rcs_m2'],
                    1.0  # Detection flag
                ]
            labels_dataset[i] = labels
            n_targets_dataset[i] = n_t
            
            # Store SNR of first target (for monitoring)
            if targets:
                t = targets[0]
                t_bin = int((t['range_km'] - t['window_start_km']) * 1000 / sim.range_resolution)
                snr = sim.compute_snr(iq, t_bin)
                snr_dataset[i] = snr
            else:
                snr_dataset[i] = -999
    
    logger.info(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    # Test simulator
    print("Testing ATeena Simulator...")
    
    sim = ATeenaSimulator(seed=42)
    
    # Test single target
    targets = [{
        'range_km': 800,
        'velocity_kms': 7.5,
        'azimuth_deg': 45,
        'elevation_deg': 30,
        'rcs_m2': 1.0
    }]
    
    iq = sim.simulate_cpi(targets, noise_figure_db=3.0)
    
    print(f"IQ shape: {iq.shape}")
    print(f"IQ dtype: {iq.dtype}")
    print(f"Signal power: {np.mean(np.abs(iq)**2):.2e}")
    
    snr = sim.compute_snr(iq, int(800 * 1000 / sim.range_resolution))
    print(f"SNR: {snr:.1f} dB")
    
    # Test random scenario
    print("\nTesting random scenario...")
    targets, noise_fig = generate_random_scenario(min_targets=5, max_targets=20)
    print(f"Generated {len(targets)} targets")
    print(f"Noise figure: {noise_fig:.1f} dB")
    
    iq = sim.simulate_cpi(targets, noise_figure_db=noise_fig)
    print(f"IQ shape: {iq.shape}")
    
    print("\n✓ ATeena simulator ready for training data generation")
