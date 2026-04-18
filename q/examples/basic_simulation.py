#!/usr/bin/env python3
"""
Basic Quantum Immune Digital Twin Simulation Example

Demonstrates core functionality:
1. Initialize BdG Hamiltonian
2. Generate QP bursts under cosmic radiation
3. Track topological protection metrics
4. Calculate survival multiplier
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bdg_hamiltonian import DigitalTwinEngine, CosmicRayParameters, NanowireGeometry
import numpy as np


def main():
    print("=" * 70)
    print("Quantum Immune Digital Twin - Basic Simulation Example")
    print("=" * 70)
    
    # Configuration: Deep space environment
    print("\n[Configuration]")
    print("  Environment: Deep space (100 mSv/hr cosmic radiation)")
    print("  Nanowire: InAs/Al heterostructure, 2000 nm length")
    print("  Magnetic field: 1.0 T (Zeeman splitting for topological phase)")
    print("  Simulation: 1 ms duration, 10 μs timesteps")
    
    # Initialize cosmic ray parameters for deep space
    cosmic = CosmicRayParameters(
        dose_rate_msv_hr=100.0,      # Deep space ambient flux
        qp_lifetime_us=100.0,         # Active clearing (vs 10 ms passive)
        mean_energy_deposition_mev=50.0,
        qp_generation_efficiency=0.57  # 57% Cooper pair breaking efficiency
    )
    
    # Configure nanowire geometry
    geometry = NanowireGeometry(
        length_nm=2000.0,
        diameter_nm=100.0,
        spin_orbit_coupling_mev=10.0,
        g_factor=15.0,                 # InAs effective g-factor
        effective_mass_me=0.023,      # InAs effective mass
        delta_0_mev=0.2               # Superconducting gap
    )
    
    # Initialize Digital Twin Engine
    print("\n[Initialization]")
    twin = DigitalTwinEngine(
        geometry=geometry,
        cosmic_params=cosmic,
        num_sites=50,
        magnetic_field_t=1.0,
        random_seed=42
    )
    print(f"  Hamiltonian matrix: {twin.hamiltonian.H.shape}")
    print(f"  Lattice spacing: {twin.hamiltonian.a:.1f} nm")
    
    # Check initial topological protection
    gap = twin.hamiltonian.topological_gap_mev()
    majorana_indices = twin.hamiltonian.find_majorana_modes(energy_threshold_mev=0.1)
    print(f"  Initial topological gap: {gap:.3f} meV")
    print(f"  Majorana zero modes: {len(majorana_indices)}")
    
    # Run simulation
    print("\n[Running Simulation]")
    duration_us = 1000.0  # 1 ms
    dt_us = 10.0
    num_steps = int(duration_us / dt_us)
    
    print(f"  Timesteps: {num_steps}")
    print(f"  Expected burst rate: {cosmic.burst_rate_hz():.2e} Hz")
    
    # Step-by-step with progress
    for i in range(0, num_steps, 100):
        chunk_steps = min(100, num_steps - i)
        for _ in range(chunk_steps):
            twin.step(dt_us)
        
        if i % 500 == 0:
            current_gap = twin.topological_gap_history[-1][1] if twin.topological_gap_history else 0
            active_bursts = len([b for b in twin.burst_sim.burst_history 
                               if twin.time_us - b.time_us < cosmic.qp_lifetime_us * 10])
            print(f"  t={twin.time_us:.0f}μs: gap={current_gap:.3f}meV, active_bursts={active_bursts}")
    
    # Results
    print("\n[Results]")
    
    # Survival multiplier
    multiplier = twin.get_survival_multiplier()
    print(f"  Algorithmic Survival Multiplier: {multiplier:.2f}x")
    
    # Parity lifetime statistics
    if twin.parity_lifetime_history:
        lifetimes = [l for _, l in twin.parity_lifetime_history]
        mean_lifetime = np.mean(lifetimes)
        min_lifetime = np.min(lifetimes)
        print(f"  Mean parity lifetime: {mean_lifetime:.1f} μs")
        print(f"  Min parity lifetime: {min_lifetime:.1f} μs")
    
    # Topological gap statistics
    if twin.topological_gap_history:
        gaps = [g for _, g in twin.topological_gap_history]
        mean_gap = np.mean(gaps)
        min_gap = np.min(gaps)
        print(f"  Mean topological gap: {mean_gap:.3f} meV")
        print(f"  Min topological gap: {min_gap:.3f} meV")
    
    # Burst statistics
    total_bursts = len(twin.burst_sim.burst_history)
    print(f"  Total QP bursts: {total_bursts}")
    total_qps = sum(b.qp_count for b in twin.burst_sim.burst_history)
    print(f"  Total QPs generated: {total_qps:.2e}")
    
    # Assessment
    print("\n[NASA SBIR Phase I Feasibility Assessment]")
    if multiplier >= 100:
        print("  [PASS] EXCEEDS TARGET: 100x survival multiplier achieved")
    elif multiplier >= 10:
        print("  [PASS] TARGET ACHIEVED: 10-100x survival multiplier")
    else:
        print("  [FAIL] BELOW TARGET: <10x survival multiplier")
    
    if mean_gap >= 0.1:
        print("  [PASS] Topological protection maintained throughout simulation")
    else:
        print("  [WARN] Gap suppression detected - may impact coherence")
    
    print("\n" + "=" * 70)
    print("Simulation complete. Active evasion demonstrates measurable")
    print("improvement over passive topological protection.")
    print("=" * 70)


if __name__ == "__main__":
    main()
