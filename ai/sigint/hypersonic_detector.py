#!/usr/bin/env python3
"""
Hypersonic Glide Vehicle (HGV) Detection using Multi-Resolution DMD (mrDMD)

Implements Steve Brunton's spatiotemporal mode analysis to separate Mach 5+
signatures from orbital clutter. Uses S2 cell snapshots as input matrix.

Challenge 37: Detect hypersonic launches via coherent structure isolation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    from pydmd import MrDMD, SpatioTemporalDMD
    PYDMD_AVAILABLE = True
except ImportError:
    PYDMD_AVAILABLE = False
    logging.warning("pydmd not installed, using JAX implementation")

import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


@dataclass
class HypersonicMode:
    """Detected hypersonic spatiotemporal mode"""
    frequency: float           # Hz (temporal frequency)
    growth_rate: float         # Real part of eigenvalue
    spatial_mode: np.ndarray   # Spatial structure
    temporal_mode: np.ndarray  # Temporal evolution
    amplitude: float           # Mode energy
    mach_estimate: float        # Derived Mach number
    confidence: float          # 0-1 classification confidence


class HypersonicDetector:
    """
    Mach 5+ detection using Multi-Resolution DMD.
    
    Separates dynamics into scales:
    - Background modes (ω ≈ 0): Stationary stars, slow debris
    - Hypersonic modes (ω >> 0): Mach 5+ coherent structures
    """
    
    def __init__(self, 
                 mach_threshold: float = 5.0,
                 svd_rank: int = 10,
                 max_level: int = 5):
        """
        Args:
            mach_threshold: Mach number for hypersonic detection
            svd_rank: Rank for low-rank approximation
            max_level: MrDMD decomposition levels
        """
        self.mach_threshold = mach_threshold
        self.svd_rank = svd_rank
        self.max_level = max_level
        
        # Detection thresholds (tuned for HGV signatures)
        self.target_freq_min = 0.1  # Hz (high-frequency cutoff)
        self.target_freq_max = 10.0  # Hz
        
    def detect(self, 
               frames: np.ndarray, 
               dt: float = 0.1,
               gate_location: str = "Siliguri") -> Tuple[bool, List[HypersonicMode]]:
        """
        Detect hypersonic signatures in sensor frame sequence.
        
        Args:
            frames: [time, pixels] or [time, height, width] sensor data
            dt: Time step between frames (seconds)
            gate_location: Which tripwire gate (for logging)
            
        Returns:
            (detected, modes): Boolean and list of hypersonic modes
        """
        # Reshape to 2D snapshot matrix if needed
        if frames.ndim == 3:
            # Video frames: flatten spatial dimensions
            X = frames.reshape(frames.shape[0], -1)
        else:
            X = frames
            
        logger.info(f"Analyzing {X.shape[0]} frames at {gate_location} gate")
        
        # Apply Multi-Resolution DMD
        if PYDMD_AVAILABLE:
            modes = self._detect_pydmd(X, dt)
        else:
            modes = self._detect_jax(X, dt)
            
        # Filter for hypersonic modes
        hypersonic_modes = self._filter_hypersonic_modes(modes)
        
        detected = len(hypersonic_modes) > 0
        if detected:
            logger.warning(f"HYPERSONIC DETECTED at {gate_location}: "
                          f"{len(hypersonic_modes)} modes, "
                          f"Mach {hypersonic_modes[0].mach_estimate:.1f}")
            
        return detected, hypersonic_modes
    
    def _detect_pydmd(self, X: np.ndarray, dt: float) -> List[HypersonicMode]:
        """Use PyDMD's optimized implementation"""
        # Multi-Resolution DMD for scale separation
        mr_dmd = MrDMD(
            svd_rank=self.svd_rank,
            max_level=self.max_level,
            tikhonov_regularization=1e-10
        )
        mr_dmd.fit(X, dt)
        
        modes = []
        
        # Extract modes from all levels
        for level_idx in range(mr_dmd.max_level):
            level_eigs = mr_dmd.eigs[level_idx]
            level_modes = mr_dmd.modes[level_idx]
            
            for i, eig in enumerate(level_eigs):
                # DMD eigenvalue: μ = exp(ω * dt)
                # ω = log(μ) / dt
                omega = np.log(eig) / dt
                freq = np.abs(omega.imag) / (2 * np.pi)  # Hz
                growth = omega.real
                
                mode = HypersonicMode(
                    frequency=freq,
                    growth_rate=growth,
                    spatial_mode=level_modes[:, i],
                    temporal_mode=None,  # Would need reconstruction
                    amplitude=np.abs(level_modes[:, i]).mean(),
                    mach_estimate=self._freq_to_mach(freq),
                    confidence=0.0  # Will compute below
                )
                modes.append(mode)
                
        return modes
    
    def _detect_jax(self, X: np.ndarray, dt: float) -> List[HypersonicMode]:
        """JAX-native DMD for GPU acceleration on M2 Air"""
        # Convert to JAX array
        X_jax = jnp.array(X)
        
        # SVD for low-rank approximation
        U, S, Vt = jnp.linalg.svd(X_jax, full_matrices=False)
        
        # Truncate to rank-r
        r = min(self.svd_rank, len(S))
        Ur = U[:, :r]
        Sr = S[:r]
        Vr = Vt[:r, :].T
        
        # DMD operator: A_tilde = U^T * X' * V * Σ^{-1}
        X_prime = X_jax[1:, :].T  # Shifted snapshot matrix
        
        # Compute DMD modes using JAX
        A_tilde = Ur.T @ X_prime @ Vr @ jnp.diag(1.0 / Sr)
        
        # Eigen-decomposition
        eigenvalues, eigenvectors = jnp.linalg.eig(A_tilde)
        
        # Convert to physical modes
        phi = X_prime @ Vr @ jnp.diag(1.0 / Sr) @ eigenvectors
        
        modes = []
        
        for i in range(r):
            mu = eigenvalues[i]
            omega = jnp.log(mu) / dt
            freq = jnp.abs(omega.imag) / (2 * jnp.pi)
            
            mode = HypersonicMode(
                frequency=float(freq),
                growth_rate=float(omega.real),
                spatial_mode=np.array(phi[:, i]),
                temporal_mode=None,
                amplitude=float(jnp.abs(phi[:, i]).mean()),
                mach_estimate=self._freq_to_mach(float(freq)),
                confidence=0.85
            )
            modes.append(mode)
            
        return modes
    
    def _filter_hypersonic_modes(self, modes: List[HypersonicMode]) -> List[HypersonicMode]:
        """Filter modes for hypersonic frequency range"""
        hypersonic_modes = []
        
        for mode in modes:
            # Frequency-based filtering
            if self.target_freq_min <= mode.frequency <= self.target_freq_max:
                # Mach number validation
                if mode.mach_estimate >= self.mach_threshold:
                    # Confidence score based on energy and stability
                    confidence = self._compute_confidence(mode)
                    mode.confidence = confidence
                    
                    if confidence > 0.7:  # High-confidence threshold
                        hypersonic_modes.append(mode)
                        
        # Sort by confidence
        hypersonic_modes.sort(key=lambda m: m.confidence, reverse=True)
        return hypersonic_modes
    
    def _freq_to_mach(self, freq: float) -> float:
        """
        Convert temporal frequency to estimated Mach number.
        
        Physics: HGV crossing S2 cell (~10km) in T seconds
        Mach ≈ (cell_size / T) / speed_of_sound
        Frequency f = 1/T
        """
        # S2 cell size at level 12 ≈ 10km
        cell_size_m = 10000.0
        speed_of_sound = 340.0  # m/s
        
        # Velocity estimate: v = distance * frequency
        velocity = cell_size_m * freq  # m/s
        mach = velocity / speed_of_sound
        
        return mach
    
    def _compute_confidence(self, mode: HypersonicMode) -> float:
        """Compute detection confidence score"""
        # Factors:
        # 1. Mode energy (amplitude)
        energy_score = min(mode.amplitude / 10.0, 1.0)
        
        # 2. Frequency stability (growth rate near 0)
        stability_score = np.exp(-abs(mode.growth_rate))
        
        # 3. Mach consistency (in expected range)
        mach_score = 1.0 if 5.0 <= mode.mach_estimate <= 25.0 else 0.5
        
        # Weighted combination
        confidence = (0.4 * energy_score + 
                     0.4 * stability_score + 
                     0.2 * mach_score)
        
        return min(confidence, 1.0)
    
    def reconstruct_trajectory(self, 
                            mode: HypersonicMode, 
                            n_frames: int,
                            dt: float) -> np.ndarray:
        """
        Reconstruct predicted trajectory from DMD mode.
        
        Returns: [n_frames, n_pixels] predicted evolution
        """
        # DMD reconstruction: X(t) = Σ b_k * φ_k * exp(ω_k * t)
        times = np.arange(n_frames) * dt
        
        # Temporal evolution
        temporal = np.exp(1j * 2 * np.pi * mode.frequency * times)
        if mode.growth_rate != 0:
            temporal *= np.exp(mode.growth_rate * times)
            
        # Spatial-temporal reconstruction
        reconstruction = np.outer(temporal, mode.spatial_mode)
        
        return np.real(reconstruction)  # Take real part


class SpatiotemporalModeFilter:
    """
    Pre-filter for separating background clutter from hypersonic signals.
    
    Separates scales:
    - Low-frequency: Stars, slow satellites (< Mach 1)
    - Mid-frequency: Fast aircraft, missiles (Mach 1-5)
    - High-frequency: Hypersonic glide vehicles (Mach 5+)
    """
    
    def __init__(self, 
                 low_cutoff: float = 0.01,   # Hz
                 high_cutoff: float = 0.1):  # Hz
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        
    def separate_scales(self, frames: np.ndarray, dt: float) -> dict:
        """
        Separate frames into three frequency bands.
        
        Returns:
            dict with 'background', 'atmospheric', 'hypersonic' bands
        """
        # FFT-based bandpass filtering
        n_frames = frames.shape[0]
        freqs = np.fft.fftfreq(n_frames, dt)
        
        # Transform each pixel time series
        fft_frames = np.fft.fft(frames, axis=0)
        
        # Band masks
        background_mask = np.abs(freqs) < self.low_cutoff
        atmospheric_mask = (np.abs(freqs) >= self.low_cutoff) & \
                          (np.abs(freqs) < self.high_cutoff)
        hypersonic_mask = np.abs(freqs) >= self.high_cutoff
        
        # Reconstruct bands
        bands = {
            'background': np.real(np.fft.ifft(
                fft_frames * background_mask[:, None], axis=0)),
            'atmospheric': np.real(np.fft.ifft(
                fft_frames * atmospheric_mask[:, None], axis=0)),
            'hypersonic': np.real(np.fft.ifft(
                fft_frames * hypersonic_mask[:, None], axis=0))
        }
        
        return bands


def demo_hypersonic_detection():
    """Demonstrate Mach 5+ detection on synthetic data"""
    print("🚀 Hypersonic Launch Detection Demo (Challenge 37)")
    
    # Generate synthetic sensor data
    n_frames = 100
    n_pixels = 50
    dt = 0.1  # 10Hz sampling
    
    # Background: slow satellite (Mach 0.5)
    background = np.sin(2 * np.pi * 0.005 * np.arange(n_frames))[:, None] * \
                 np.random.randn(n_pixels)[None, :]
    
    # Clutter: random debris
    clutter = np.random.randn(n_frames, n_pixels) * 0.1
    
    # Target: hypersonic signal (Mach 10) - high frequency spike
    target_pos = np.linspace(0, n_pixels, n_frames)
    hypersonic = np.zeros((n_frames, n_pixels))
    for t in range(n_frames):
        if int(target_pos[t]) < n_pixels:
            hypersonic[t, int(target_pos[t])] = 5.0  # Strong signal
    
    # Combined frames
    frames = background + clutter + hypersonic
    
    # Detect
    detector = HypersonicDetector(mach_threshold=5.0)
    detected, modes = detector.detect(frames, dt, "Siliguri")
    
    if detected:
        print(f"✅ Detected {len(modes)} hypersonic mode(s)")
        for i, mode in enumerate(modes[:3]):
            print(f"  Mode {i+1}: Mach {mode.mach_estimate:.1f}, "
                  f"Confidence {mode.confidence:.2%}")
    else:
        print("❌ No hypersonic signature detected")
    
    return detected, modes


if __name__ == "__main__":
    demo_hypersonic_detection()
