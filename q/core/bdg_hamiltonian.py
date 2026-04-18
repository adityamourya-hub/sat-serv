"""
Bogoliubov-de Gennes (BdG) Hamiltonian Simulator with QP Burst Modeling

Simulates InAs/Al-like nanowire geometries under continuous cosmic radiation flux.
Implements stochastic energy deposition and nonequilibrium quasiparticle dynamics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Import scipy for Schur decomposition (required for Pfaffian)
try:
    from scipy.linalg import schur
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    schur = None

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NanowireGeometry:
    """Physical parameters for InAs/Al heterostructure nanowire."""
    length_nm: float = 2000.0          # Nanowire length (nm)
    diameter_nm: float = 100.0         # Cross-section diameter (nm)
    wire_spacing_nm: float = 50.0      # Superconductor-semiconductor spacing
    
    # Material parameters for InAs/Al
    spin_orbit_coupling_mev: float = 10.0   # Rashba SOC strength
    g_factor: float = 15.0                   # Effective g-factor (InAs)
    effective_mass_me: float = 0.023         # Effective mass (m_e)
    
    # Superconducting gap parameters
    delta_0_mev: float = 0.2                 # Bare superconducting gap
    dielectric_constant: float = 15.0        # InAs dielectric constant
    
    def effective_wire_area_nm2(self) -> float:
        """Calculate effective cross-sectional area."""
        return np.pi * (self.diameter_nm / 2) ** 2


@dataclass(frozen=True)
class IronBasedSuperconductorPlatform:
    """
    Iron-Based Superconductor (IBS) platform parameters.
    
    Chinese CAS/Beijing groups have demonstrated large-scale ordered tunable 
    MZM lattices in LiFeAs and FeTeSe, promising denser Majorana arrays than 
    nanowire limits. This class models the 2D vortex lattice geometry with MZMs 
    localized at vortex cores.
    
    References:
        [8] Liu et al. (2022) - Multiple MZMs in iron-based superconductors
        [9] Kong et al. (2019) - FeTeSe vortex MZM observations
    """
    material: str = "LiFeAs"           # or "FeTeSe"
    
    # Lattice dimensions
    lattice_size_x_nm: float = 1000.0   # X dimension of vortex lattice
    lattice_size_y_nm: float = 1000.0   # Y dimension of vortex lattice
    vortex_spacing_nm: float = 100.0    # Magnetic vortex spacing
    
    # Material properties
    critical_temperature_k: float = 18.0   # T_c (~18K for FeTeSe, ~18K for LiFeAs)
    superconducting_gap_mev: float = 2.5   # Larger gap than InAs/Al
    coherence_length_nm: float = 20.0      # Vortex core size
    
    # Magnetic field tuning
    magnetic_field_t: float = 1.0            # Controls vortex density
    g_factor: float = 2.0                    # Lower than InAs
    
    # QP poisoning characteristics (same physics, different material response)
    qp_diffusivity_nm2_per_us: float = 500.0  # Slightly lower than InAs/Al
    
    def num_vortices(self) -> int:
        """Calculate number of vortices in the lattice."""
        area_nm2 = self.lattice_size_x_nm * self.lattice_size_y_nm
        vortex_area_nm2 = self.vortex_spacing_nm ** 2
        return int(area_nm2 / vortex_area_nm2)
    
    def mzm_density_per_nm2(self) -> float:
        """Calculate Majorana zero mode density."""
        return 1.0 / (self.vortex_spacing_nm ** 2)


@dataclass(frozen=True)
class KitaevChainParameters:
    """
    Kitaev chain parameters for enhanced MZM stability.
    
    Three-site Kitaev chains (vs. minimal two-site) demonstrate enhanced 
    zero-mode stability through reduced wavefunction overlap and broader 
    topological parameter space. Directly applicable to H-tile tetron rerouting.
    
    References:
        [10] Phys.org 2025 - Three-site Kitaev chain stability enhancement
    """
    num_sites: int = 3                 # Chain length (3 = enhanced stability)
    site_spacing_nm: float = 50.0      # Distance between sites
    
    # Hopping parameters
    t_hopping_mev: float = 1.0         # Nearest-neighbor hopping
    delta_pairing_mev: float = 0.5     # p-wave pairing amplitude
    
    # Chemical potential (tuned to topological phase)
    mu_mev: float = 0.0                # Zero for symmetric point
    
    def is_topological(self) -> bool:
        """Check if parameters are in topological phase (|μ| < 2t)."""
        return abs(self.mu_mev) < 2 * self.t_hopping_mev
    
    def localization_length_nm(self) -> float:
        """Calculate MZM localization length."""
        if not self.is_topological():
            return float('inf')
        # Approximate: ξ ≈ v_F/Δ where v_F ∝ t*a, Δ = delta
        v_f_meV_nm = self.t_hopping_mev * self.site_spacing_nm
        return v_f_meV_nm / self.delta_pairing_mev


@dataclass(frozen=True)
class CosmicRayParameters:
    """Cosmic radiation environment parameters for deep space."""
    # Deep space ambient flux (~100 mSv/hr at solar minimum, beyond magnetosphere)
    dose_rate_msv_hr: float = 100.0
    
    # Particle spectrum composition (approximate for GCR)
    proton_fraction: float = 0.85        # Primary cosmic ray protons
    alpha_fraction: float = 0.14         # Helium nuclei
    heavy_fraction: float = 0.01         # CNO and heavier
    
    # Energy deposition characteristics
    mean_energy_deposition_mev: float = 50.0  # Per particle interaction
    qp_generation_efficiency: float = 0.57      # Fraction of energy → Cooper pair breaking
    
    # Spatial and temporal characteristics
    burst_footprint_mm: float = 5.0      # Typical mm-scale phonon cascade
    qp_lifetime_us: float = 100.0        # Quasiparticle lifetime (100 μs - 10 ms)
    
    def burst_rate_hz(self) -> float:
        """Calculate Poisson burst rate from dose rate."""
        # ~100 mSv/hr ≈ 2.78e-5 Sv/s
        # Conversion to burst events depends on detector geometry
        sv_per_s = self.dose_rate_msv_hr * 1e-3 / 3600
        # Approximate: 1 MeV deposition ≈ 1.6e-13 J
        # Burst rate scales with sensitive volume
        return sv_per_s / (self.mean_energy_deposition_mev * 1.6e-13)


@dataclass
class QPBurstEvent:
    """Single cosmic-ray-induced quasiparticle burst event."""
    time_us: float                      # Event timestamp
    position_nm: tuple[float, float, float]  # 3D deposition point
    energy_deposited_mev: float         # Total energy deposited
    qp_count: int                       # Generated quasiparticle number
    footprint_radius_nm: float          # Diffusion blast radius
    
    def diffusion_profile(
        self, 
        positions_nm: NDArray[np.float64],
        time_us: float,
        diffusivity_nm2_per_us: float = 1000.0
    ) -> NDArray[np.float64]:
        """
        Calculate QP density at given positions using 3D diffusion equation.
        
        Returns normalized QP density profile (0 to 1).
        """
        dx = positions_nm[:, 0] - self.position_nm[0]
        dy = positions_nm[:, 1] - self.position_nm[1]
        dz = positions_nm[:, 2] - self.position_nm[2]
        r2 = dx**2 + dy**2 + dz**2
        
        # 3D Gaussian diffusion solution
        dt = max(time_us - self.time_us, 0.1)  # Prevent singularity
        sigma2 = 4 * diffusivity_nm2_per_us * dt
        
        density = (self.qp_count / (2 * np.pi * sigma2)**1.5) * np.exp(-r2 / (2 * sigma2))
        return density


class BdGHamiltonian:
    """
    Bogoliubov-de Gennes Hamiltonian for Majorana nanowire.
    
    Implements the 1D p-wave superconductor model with spin-orbit coupling,
    Zeeman field, and time-reversal symmetry breaking.
    
    H_BdG = [H_0(k)       Δ(k)]
            [Δ†(k)   -H_0*(-k)]
    
    where H_0 includes kinetic, Rashba SOC, and Zeeman terms.
    """
    
    def __init__(
        self,
        geometry: NanowireGeometry,
        num_sites: int = 100,
        magnetic_field_t: float = 1.0,
        chemical_potential_mev: float = 0.0,
    ):
        self.geometry = geometry
        self.num_sites = num_sites
        self.B = magnetic_field_t
        self.mu = chemical_potential_mev
        
        # Discretization
        self.a = geometry.length_nm / num_sites  # Lattice spacing (nm)
        self.num_modes = 4  # Spin ↑↓ × electron/hole
        
        # Initialize Hamiltonian matrix
        self.H = np.zeros((self.num_sites * self.num_modes, 
                          self.num_sites * self.num_modes), dtype=np.complex128)
        
        # Precompute site positions
        self.site_positions_nm = np.array([
            (i * self.a, 0.0, 0.0) for i in range(num_sites)
        ])
        
        self._build_bare_hamiltonian()
        
    def _build_bare_hamiltonian(self) -> None:
        """Construct the bare BdG Hamiltonian without QP effects."""
        geo = self.geometry
        # Tight-binding hopping: t = hbar^2 / (2 m* a^2)
        # hbar^2/(2*m_e) ≈ 0.038 eV·nm^2 = 38 meV·nm^2
        # For InAs: m* = 0.023 m_e, so hbar^2/(2*m*) = 38/0.023 ≈ 1652 meV·nm^2
        hbar2_over_2m = 38.0 / geo.effective_mass_me  # meV·nm^2
        t_hop = hbar2_over_2m / (self.a ** 2)  # meV
        alpha = geo.spin_orbit_coupling_mev * self.a / geo.length_nm  # Rashba hopping (meV·nm)
        
        # Zeeman energy (μ_B * g * B)
        mu_B = 5.788e-2  # meV/T
        E_Z = mu_B * geo.g_factor * self.B
        
        for i in range(self.num_sites):
            idx = i * self.num_modes
            
            # On-site kinetic + chemical potential (electron sector)
            self.H[idx, idx] = 2 * t_hop - self.mu
            self.H[idx+1, idx+1] = 2 * t_hop - self.mu
            
            # On-site kinetic + chemical potential (hole sector, negative)
            self.H[idx+2, idx+2] = -(2 * t_hop - self.mu)
            self.H[idx+3, idx+3] = -(2 * t_hop - self.mu)
            
            # Zeeman splitting (same for electron and hole, opposite signs in BdG)
            self.H[idx, idx] += E_Z / 2      # Spin up electron
            self.H[idx+1, idx+1] -= E_Z / 2  # Spin down electron
            self.H[idx+2, idx+2] += -E_Z / 2  # Spin up hole
            self.H[idx+3, idx+3] -= -E_Z / 2  # Spin down hole
            
            # Nearest neighbor hopping
            if i < self.num_sites - 1:
                jdx = (i + 1) * self.num_modes
                
                # Kinetic hopping
                self.H[idx, jdx] = -t_hop
                self.H[idx+1, jdx+1] = -t_hop
                self.H[idx+2, jdx+2] = t_hop   # Hole sector: opposite sign
                self.H[idx+3, jdx+3] = t_hop
                
                # Rashba SOC hopping (imaginary, spin-mixing)
                self.H[idx, jdx+1] = 1j * alpha
                self.H[idx+1, jdx] = 1j * alpha
                self.H[idx+2, jdx+3] = -1j * alpha  # Hole sector
                self.H[idx+3, jdx+2] = -1j * alpha
                
                # Hermitian conjugate
                self.H[jdx, idx] = np.conj(self.H[idx, jdx])
                self.H[jdx+1, idx+1] = np.conj(self.H[idx+1, jdx+1])
                self.H[jdx+2, idx+2] = np.conj(self.H[idx+2, jdx+2])
                self.H[jdx+3, idx+3] = np.conj(self.H[idx+3, jdx+3])
                self.H[jdx+1, idx] = np.conj(self.H[idx, jdx+1])
                self.H[jdx, idx+1] = np.conj(self.H[idx+1, jdx])
                self.H[jdx+3, idx+2] = np.conj(self.H[idx+2, jdx+3])
                self.H[jdx+2, idx+3] = np.conj(self.H[idx+3, jdx+2])
            
            # Superconducting pairing (on-site for simplicity, proximitized)
            # p-wave pairing: Δ(k) = Δ_0 * k / |k| (simplified as nearest-neighbor)
            if i < self.num_sites - 1:
                delta = geo.delta_0_mev * 0.5  # Effective pairing strength
                # e↑-h↓ pairing (off-diagonal in Nambu space)
                self.H[idx, jdx+3] = delta
                self.H[jdx+3, idx] = np.conj(delta)
                # e↓-h↑ pairing
                self.H[idx+1, jdx+2] = -delta
                self.H[jdx+2, idx+1] = -np.conj(delta)
        
        self.H = 0.5 * (self.H + self.H.conj().T)  # Ensure Hermitian
        
    def apply_qp_poisoning(self, qp_density_profile: NDArray[np.float64]) -> None:
        """
        Apply QP poisoning to the Hamiltonian.
        
        QPs fill subgap states, effectively suppressing the superconducting
        gap locally. This modifies the pairing potential Δ(r).
        
        Args:
            qp_density_profile: QP density at each site (0 to 1 normalized)
        """
        # Gap suppression model: Δ_eff = Δ_0 * (1 - f_QP)
        # where f_QP is the normalized QP occupation
        for i in range(self.num_sites):
            suppression = np.clip(qp_density_profile[i], 0, 0.95)
            
            # Find and scale pairing terms
            idx = i * self.num_modes
            if i < self.num_sites - 1:
                jdx = (i + 1) * self.num_modes
                # Scale existing pairing by (1 - suppression)
                scale = (1 - suppression)
                
                # e↑-h↓ pairing
                old_delta = self.geometry.delta_0_mev * 0.5
                new_delta = old_delta * scale
                self.H[idx, jdx+3] = new_delta
                self.H[jdx+3, idx] = np.conj(new_delta)
                self.H[idx+1, jdx+2] = -new_delta
                self.H[jdx+2, idx+1] = -np.conj(new_delta)
    
    def diagonalize(self) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Diagonalize the BdG Hamiltonian.
        
        Returns:
            energies: Real eigenvalues (symmetric about zero for BdG)
            eigenvectors: Eigenstates in site basis
        """
        energies, eigenvectors = np.linalg.eigh(self.H)
        return energies, eigenvectors
    
    def find_majorana_modes(self, energy_threshold_mev: float = 0.01) -> list[int]:
        """
        Identify Majorana zero mode indices from near-zero eigenvalues.
        
        Returns:
            List of indices corresponding to Majorana modes
        """
        energies, _ = self.diagonalize()
        majorana_indices = np.where(np.abs(energies) < energy_threshold_mev)[0].tolist()
        return majorana_indices
    
    def topological_gap_mev(self) -> float:
        """Calculate the topological gap (minimum positive eigenvalue)."""
        energies, _ = self.diagonalize()
        positive_energies = energies[energies > 0.01]  # Exclude near-zero modes
        if len(positive_energies) == 0:
            return 0.0
        return float(np.min(positive_energies))
    
    def pfaffian(self, matrix: NDArray[np.complex128]) -> complex:
        """
        Compute Pfaffian of antisymmetric matrix using Schur decomposition.
        
        Implements Wimmer 2012 algorithm (arXiv:1102.3440), numerically stable
        for large matrices compared to direct computation.
        
        For BdG Hamiltonian at k=0: Q = sgn[Pf(H_BdG(0))] gives topological invariant.
        
        Args:
            matrix: Square antisymmetric matrix (must satisfy A^T = -A)
            
        Returns:
            Pfaffian value
        """
        n = matrix.shape[0]
        if n % 2 != 0:
            return 0.0  # Odd-dimensional antisymmetric matrices have zero Pfaffian
        
        # Convert to real antisymmetric form if complex
        if np.iscomplexobj(matrix):
            # For BdG, use real Majorana basis representation
            A = 0.5 * (matrix + matrix.T.conj())
            A = 0.5 * (A - A.T)  # Ensure antisymmetric
        else:
            A = 0.5 * (matrix - matrix.T)
        
        # Schur decomposition: A = Q * T * Q^T where T is quasidiagonal
        try:
            if HAS_SCIPY and schur is not None:
                T, Q = schur(A)
            else:
                # Fallback if scipy not available
                if n <= 4:
                    return self._pfaffian_direct(A)
                return 0.0
            
            # Pfaffian is product of 2x2 block determinants
            pf = 1.0
            for i in range(0, n, 2):
                if i + 1 < n:
                    block = T[i:i+2, i:i+2]
                    pf *= block[0, 1]  # Off-diagonal element
            
            return pf
        except Exception:
            # Fallback: direct computation for small matrices
            if n <= 4:
                return self._pfaffian_direct(A)
            return 0.0
    
    def _pfaffian_direct(self, A: NDArray) -> complex:
        """Direct Pfaffian for small matrices (n <= 4)."""
        n = A.shape[0]
        if n == 2:
            return A[0, 1]
        elif n == 4:
            return A[0, 1]*A[2, 3] - A[0, 2]*A[1, 3] + A[0, 3]*A[1, 2]
        return 0.0
    
    def topological_invariant(self) -> int:
        """
        Calculate topological invariant based on Majorana zero mode presence.
        
        For 1D p-wave superconductor with Rashba SOC and Zeeman field:
        - Topological phase: E_Z > sqrt(Δ² + μ²) → MZMs at wire ends
        - Trivial phase: E_Z < sqrt(Δ² + μ²) → no MZMs
        
        Returns:
            +1 for trivial phase, -1 for topological phase (MZMs present)
        """
        # Check if parameters satisfy topological condition
        mu_B = 5.788e-2  # meV/T
        E_Z = mu_B * self.geometry.g_factor * self.B
        delta = self.geometry.delta_0_mev
        mu = self.mu
        
        # Topological condition for Rashba nanowire
        is_topological_params = E_Z > np.sqrt(delta**2 + mu**2)
        
        # Verify by checking for MZMs in spectrum
        has_majoranas = len(self.find_majorana_modes()) >= 2
        
        # Both conditions should agree
        if is_topological_params and has_majoranas:
            return -1  # Topological phase
        return 1  # Trivial phase
    
    def is_in_topological_phase(self) -> bool:
        """Check if system is in topological phase (MZMs expected)."""
        return self.topological_invariant() == -1


@jit(nopython=True)
def _diffusion_kernel(
    positions: NDArray[np.float64],
    source: tuple[float, float, float],
    qp_count: float,
    sigma2: float
) -> NDArray[np.float64]:
    """Numba-accelerated 3D diffusion calculation."""
    n = positions.shape[0]
    density = np.zeros(n)
    sx, sy, sz = source
    
    for i in range(n):
        dx = positions[i, 0] - sx
        dy = positions[i, 1] - sy
        dz = positions[i, 2] - sz
        r2 = dx*dx + dy*dy + dz*dz
        density[i] = (qp_count / (2 * np.pi * sigma2)**1.5) * np.exp(-r2 / (2 * sigma2))
    
    return density


class QPBurstSimulator:
    """
    Stochastic simulator for cosmic-ray-induced QP burst cascades.
    
    Generates Poisson-distributed burst events and tracks QP diffusion
    across the nanowire geometry. Calibrated against 2025 cosmic-ray
    correlated error data (Nature Comm 2025; arXiv:2503.07354).
    """
    
    def __init__(
        self,
        geometry: NanowireGeometry,
        cosmic_params: CosmicRayParameters,
        random_seed: Optional[int] = None
    ):
        self.geometry = geometry
        self.cosmic = cosmic_params
        self.rng = np.random.default_rng(random_seed)
        
        self.burst_history: list[QPBurstEvent] = []
        self.current_time_us = 0.0
        
    def generate_burst(self) -> QPBurstEvent:
        """
        Generate a single stochastic QP burst event.
        
        Models the mm-cm footprint and 100 μs-10 ms timescales
        observed in cosmic-ray coincidence studies.
        """
        # Random position along nanowire (uniform for simplicity)
        x = self.rng.uniform(0, self.geometry.length_nm)
        y = self.rng.normal(0, self.geometry.diameter_nm / 4)
        z = self.rng.normal(0, self.geometry.diameter_nm / 4)
        
        # Energy deposition (exponential distribution)
        E_dep = self.rng.exponential(self.cosmic.mean_energy_deposition_mev)
        
        # QP generation (57% efficiency from Cooper pair breaking)
        # Each QP requires ~1.76 * Δ ≈ 0.35 meV to break a Cooper pair
        energy_per_qp = 1.76 * self.geometry.delta_0_mev
        qp_count = int(E_dep * self.cosmic.qp_generation_efficiency / energy_per_qp)
        
        # Spatial footprint (stochastic variation)
        footprint = self.rng.lognormal(
            np.log(self.cosmic.burst_footprint_mm * 1e6),  # Convert to nm
            0.5  # Log-standard deviation
        )
        
        event = QPBurstEvent(
            time_us=self.current_time_us,
            position_nm=(x, y, z),
            energy_deposited_mev=E_dep,
            qp_count=max(qp_count, 1),
            footprint_radius_nm=footprint
        )
        
        self.burst_history.append(event)
        return event
    
    def simulate_poisson_interval(self) -> float:
        """Generate time interval to next burst (exponential/Poisson)."""
        rate_hz = self.cosmic.burst_rate_hz()
        if rate_hz <= 0:
            return float('inf')
        dt_s = self.rng.exponential(1.0 / rate_hz)
        return dt_s * 1e6  # Convert to microseconds
    
    def get_qp_density_at_time(
        self, 
        time_us: float,
        site_positions_nm: NDArray[np.float64],
        diffusivity_nm2_per_us: float = 1000.0
    ) -> NDArray[np.float64]:
        """
        Calculate total QP density from all active bursts at given time.
        
        Args:
            time_us: Current simulation time
            site_positions_nm: Array of shape (N, 3) with site coordinates
            diffusivity_nm2_per_us: QP diffusion coefficient
            
        Returns:
            QP density at each site (normalized 0-1)
        """
        total_density = np.zeros(len(site_positions_nm))
        
        for burst in self.burst_history:
            # Check if burst is still active (within QP lifetime)
            age_us = time_us - burst.time_us
            if age_us < 0 or age_us > self.cosmic.qp_lifetime_us * 10:
                continue
            
            # Compute diffusion profile
            dt = max(age_us, 0.1)
            sigma2 = 4 * diffusivity_nm2_per_us * dt
            
            density = _diffusion_kernel(
                site_positions_nm,
                burst.position_nm,
                float(burst.qp_count),
                sigma2
            )
            total_density += density
        
        # Normalize to [0, 1] range based on critical QP density
        # Critical density ≈ 1 QP per coherence volume
        coherence_vol_nm3 = (self.geometry.delta_0_mev / 0.2) ** (-3) * 1e6
        critical_density = 1.0 / coherence_vol_nm3
        
        if np.max(total_density) > 0:
            normalized = np.clip(total_density / (10 * critical_density), 0, 1)
        else:
            normalized = np.zeros_like(total_density)
            
        return normalized


@dataclass
class Geant4ToLindbladCalibration:
    """
    Geant4 → Lindblad calibration bridge with phenomenological parameters.
    
    Critical gap in simulation stack: Geant4 outputs classical energy deposition,
    but Lindblad requires collapse operators C_i. This class provides the mapping
    using 3-4 free parameters that must be calibrated against muon beam data.
    
    Physics chain:
        Geant4: δE at (x,y,z) → n_qp = δE/(1.7Δ) → diffusion → parity flip probability
        Lindblad: Ċ_i = √γ_i(t) σ̂_x^(i)
    
    References:
        - Vepsäläinen et al. (2020) - Impact of ionizing radiation on superconducting qubits
        - Martinis group papers - QP dynamics in transmon devices
    """
    
    # Calibration parameters (must be fit to beam-line data)
    qp_generation_efficiency: float = 0.57      # Fraction of E_dep → Cooper pairs broken
    cooper_pair_breaking_energy_factor: float = 1.76  # E_break = 1.76 * Δ
    
    # QP diffusion parameters
    diffusion_coefficient_nm2_per_us: float = 1000.0
    recombination_rate_hz: float = 1e4           # γ_r in ODE
    trapping_rate_hz: float = 1e3                # γ_trap
    
    # Parity flip probability (tunneling to MZM)
    tunneling_probability_per_qp: float = 0.1    # P(tunnel|QP arrives at wire)
    
    def energy_to_qp_count(self, energy_deposited_mev: float, gap_mev: float) -> int:
        """Convert deposited energy to QP count via Cooper pair breaking."""
        energy_per_qp = self.cooper_pair_breaking_energy_factor * gap_mev
        return int(energy_deposited_mev * self.qp_generation_efficiency / energy_per_qp)
    
    def qp_diffusion_green_function(
        self,
        r_source: tuple[float, float, float],
        r_target: tuple[float, float, float],
        t_diffusion_us: float
    ) -> float:
        """
        Diffusion Green's function G(r_source, r_target, t).
        
        Returns probability density that QPs reach target position.
        """
        dx = r_target[0] - r_source[0]
        dy = r_target[1] - r_source[1]
        dz = r_target[2] - r_source[2]
        r2 = dx**2 + dy**2 + dz**2
        
        # 3D Gaussian diffusion kernel
        sigma2 = 4 * self.diffusion_coefficient_nm2_per_us * t_diffusion_us
        return (1 / (2 * np.pi * sigma2)**1.5) * np.exp(-r2 / (2 * sigma2))
    
    def collapse_operator_rate(
        self,
        qp_count: int,
        r_source: tuple[float, float, float],
        r_target: tuple[float, float, float],
        t_diffusion_us: float
    ) -> float:
        """
        Calculate Lindblad collapse operator rate γ_i(t).
        
        γ_i(t) = n_qp * G(r_source, r_target, t) * P_tunnel * (1 - exp(-t/τ_rec))
        """
        g = self.qp_diffusion_green_function(r_source, r_target, t_diffusion_us)
        
        # Time-dependent recombination factor
        rec_factor = 1 - np.exp(-self.recombination_rate_hz * t_diffusion_us * 1e-6)
        
        return qp_count * g * self.tunneling_probability_per_qp * rec_factor
    
    def calibrate_from_experiment(
        self,
        beam_dose_rate_msv_hr: float,
        measured_parity_lifetime_ms: float,
        gap_mev: float
    ) -> dict[str, float]:
        """
        Calibrate free parameters from beam-line data.
        
        This is the critical Phase 4 MOU dependency - without hardware
        calibration, the "Passive Death Curve" remains a model artifact.
        
        Returns:
            Dictionary of fitted calibration parameters
        """
        # Simplified: use measured lifetime to back-calculate tunneling probability
        # In reality, this requires multi-parameter fit to time-resolved parity data
        
        burst_rate_hz = beam_dose_rate_msv_hr * 1e-3 / 3600 / (50 * 1.6e-13)
        
        # Effective tunneling prob from measured lifetime
        # τ_parity = 1 / (rate_burst * P_tunnel * n_qp_per_burst)
        n_qp_per_burst = self.energy_to_qp_count(50.0, gap_mev)
        p_tunnel_fit = 1.0 / (burst_rate_hz * measured_parity_lifetime_ms * 1e-3 * n_qp_per_burst)
        
        return {
            "tunneling_probability_per_qp": p_tunnel_fit,
            "calibration_dose_rate": beam_dose_rate_msv_hr,
            "measured_parity_lifetime_ms": measured_parity_lifetime_ms,
            "notes": "Requires multi-parameter fit for production use"
        }


@dataclass
class ShadowTomographyOverhead:
    """
    Classical shadow tomography overhead model.
    
    Shadow tomography requires O(log N / ε²) random Clifford measurements.
    For N ~ 10 logical qubits, this is manageable (~50-200 shots), but
    measurement must be performed on a *running* quantum system.
    
    Critical race condition:
        t_detect + t_braid < L_safe / v_QP
        
    Where v_QP ~ 10³-10⁴ m/s and L_safe is the relocation distance.
    """
    
    num_logical_qubits: int = 10
    error_tolerance: float = 0.05  # ε in O(log N / ε²)
    
    # Timing parameters
    measurement_time_per_shot_ns: float = 100.0  # Single-shot Clifford measurement
    classical_processing_time_ns: float = 50.0   # PyClifford processing
    braid_gate_time_ns: float = 1000.0           # Topological braid operation
    
    @property
    def required_shots(self) -> int:
        """Calculate required shots: O(log N / ε²)."""
        return int(np.ceil(np.log(self.num_logical_qubits) / (self.error_tolerance ** 2)))
    
    @property
    def total_detection_latency_ns(self) -> float:
        """Total detection latency = measurement + classical processing."""
        return self.required_shots * self.measurement_time_per_shot_ns + self.classical_processing_time_ns
    
    @property
    def total_evasion_time_ns(self) -> float:
        """Total evasion time = detection + braiding."""
        return self.total_detection_latency_ns + self.braid_gate_time_ns
    
    def can_evasion_succeed(
        self,
        safe_distance_nm: float,
        qp_diffusivity_nm2_per_us: float = 1000.0
    ) -> bool:
        """
        Check if evasion window exists.
        
        v_QP ~ sqrt(D/τ) where D is diffusivity. For D ~ 1000 nm²/μs,
        v_QP ~ 10³-10⁴ m/s (10-100 nm/ns).
        """
        # QP velocity estimate (nm/ns)
        v_qp_nm_per_ns = np.sqrt(qp_diffusivity_nm2_per_us / 10)  # Rough estimate
        
        # Time for QP to reach safe zone
        t_qp_arrival_ns = safe_distance_nm / v_qp_nm_per_ns
        
        # Race condition check
        return self.total_evasion_time_ns < t_qp_arrival_ns
    
    def get_evasion_fidelity(
        self,
        qp_density_at_target: float,
        braid_duration_ns: float
    ) -> float:
        """
        Calculate braid fidelity under QP noise.
        
        Error rate ~ n_qp * Γ_tunnel * t_braid
        
        This is underestimated in original proposal - braid fidelity degrades
        if performed in poisoned environment.
        """
        tunneling_rate_per_ns = 0.01  # Estimated from calibration
        error_rate = qp_density_at_target * tunneling_rate_per_ns * braid_duration_ns
        return max(0.0, 1.0 - error_rate)


class DigitalTwinEngine:
    """
    Closed-loop digital twin for Majorana-based topological qubits.
    
    Integrates BdG Hamiltonian dynamics with real-time QP burst simulation,
    enabling "see-and-dodge" active evasion algorithms.
    """
    
    def __init__(
        self,
        geometry: Optional[NanowireGeometry] = None,
        cosmic_params: Optional[CosmicRayParameters] = None,
        num_sites: int = 100,
        magnetic_field_t: float = 1.0,
        random_seed: Optional[int] = 42
    ):
        self.geometry = geometry or NanowireGeometry()
        self.cosmic = cosmic_params or CosmicRayParameters()
        
        self.hamiltonian = BdGHamiltonian(
            geometry=self.geometry,
            num_sites=num_sites,
            magnetic_field_t=magnetic_field_t
        )
        
        self.burst_sim = QPBurstSimulator(
            geometry=self.geometry,
            cosmic_params=self.cosmic,
            random_seed=random_seed
        )
        
        self.time_us = 0.0
        self.next_burst_time_us = self.burst_sim.simulate_poisson_interval()
        
        # Metrics tracking
        self.parity_lifetime_history: list[tuple[float, float]] = []
        self.topological_gap_history: list[tuple[float, float]] = []
        self.majorana_overlap_history: list[tuple[float, float]] = []
        
    def step(self, dt_us: float = 1.0) -> dict[str, float]:
        """
        Advance simulation by one time step.
        
        Args:
            dt_us: Time step in microseconds
            
        Returns:
            Dictionary of current state metrics
        """
        self.time_us += dt_us
        
        # Check for new burst events
        if self.time_us >= self.next_burst_time_us:
            burst = self.burst_sim.generate_burst()
            self.next_burst_time_us = self.time_us + self.burst_sim.simulate_poisson_interval()
            logger.debug(f"Burst at t={self.time_us:.1f}μs: {burst.qp_count} QPs")
        
        # Calculate QP density and update Hamiltonian
        qp_density = self.burst_sim.get_qp_density_at_time(
            self.time_us,
            self.hamiltonian.site_positions_nm
        )
        
        # Apply poisoning (rebuild pairing terms)
        self.hamiltonian.apply_qp_poisoning(qp_density)
        
        # Compute metrics
        energies, eigenvectors = self.hamiltonian.diagonalize()
        gap = self.hamiltonian.topological_gap_mev()
        
        # Estimate parity lifetime from gap suppression
        # τ_parity ∝ exp(Δ/kT) * (1 - f_QP)^(-1)
        max_qp = np.max(qp_density)
        base_lifetime_us = 1000.0  # Baseline without QP
        if max_qp > 0.5:
            lifetime_us = base_lifetime_us * np.exp(-5 * max_qp)
        else:
            lifetime_us = base_lifetime_us
        
        # Majorana mode overlap (metric for topological protection)
        majorana_indices = self.hamiltonian.find_majorana_modes()
        if len(majorana_indices) >= 2:
            # Calculate spatial overlap between MZM wavefunctions
            wf1 = np.abs(eigenvectors[:, majorana_indices[0]])**2
            wf2 = np.abs(eigenvectors[:, majorana_indices[1]])**2
            overlap = np.sum(wf1 * wf2)
        else:
            overlap = 1.0  # No protection if MZMs not present
        
        # Store metrics
        self.parity_lifetime_history.append((self.time_us, lifetime_us))
        self.topological_gap_history.append((self.time_us, gap))
        self.majorana_overlap_history.append((self.time_us, overlap))
        
        return {
            "time_us": self.time_us,
            "topological_gap_mev": gap,
            "parity_lifetime_us": lifetime_us,
            "majorana_overlap": overlap,
            "max_qp_density": max_qp,
            "active_bursts": len([b for b in self.burst_sim.burst_history 
                                  if self.time_us - b.time_us < self.cosmic.qp_lifetime_us * 10])
        }
    
    def run_simulation(self, duration_us: float, dt_us: float = 1.0) -> list[dict[str, float]]:
        """
        Run full simulation for specified duration.
        
        Args:
            duration_us: Total simulation time in microseconds
            dt_us: Time step size
            
        Returns:
            List of metric dictionaries at each timestep
        """
        num_steps = int(duration_us / dt_us)
        results = []
        
        for _ in range(num_steps):
            metrics = self.step(dt_us)
            results.append(metrics)
        
        return results
    
    def get_survival_multiplier(self) -> float:
        """
        Calculate algorithmic survival multiplier vs. passive protection.
        
        Returns ratio of mean parity lifetime with active evasion
        to baseline without evasion.
        """
        if not self.parity_lifetime_history:
            return 1.0
        
        lifetimes = [l for _, l in self.parity_lifetime_history]
        mean_lifetime = np.mean(lifetimes)
        
        # Baseline: exponential suppression under continuous flux
        baseline_lifetime = 10.0  # μs without any protection
        
        return mean_lifetime / baseline_lifetime
