#!/usr/bin/env python3
"""
Active Evasion Demo: See-and-Dodge Validation

Demonstrates the core Phase 1 IP:
1. Detection of QP burst via calorimeter trigger
2. Fast diffusion prediction (~50-100 ns)
3. Hamiltonian rerouting to safe sector
4. Measured survival multiplier vs passive baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.active_evasion import ActiveEvasionEngine
from core.bdg_hamiltonian import CosmicRayParameters, NanowireGeometry


def main():
    print("=" * 60)
    print("Quantum Immune Digital Twin - Active Evasion Demo")
    print("Phase 1: 'See-and-Dodge' Algorithm Validation")
    print("=" * 60)
    print()
    
    # Configuration: Deep space radiation environment
    cosmic = CosmicRayParameters(
        dose_rate_msv_hr=100.0,  # 100 mSv/hr continuous flux
        qp_lifetime_us=100.0
    )
    
    # Topological parameters: B=1.5T gives E_Z=1.7meV > Δ=0.1meV
    geometry = NanowireGeometry(
        length_nm=2000.0,
        diameter_nm=100.0,
        delta_0_mev=0.1,  # Smaller gap for better topological phase
        g_factor=20.0,   # Higher g-factor
        spin_orbit_coupling_mev=20.0  # Stronger SOC
    )
    magnetic_field_t = 1.5  # Sweet spot for topological phase
    
    print("Configuration:")
    print(f"  Radiation dose rate: {cosmic.dose_rate_msv_hr} mSv/hr (deep space)")
    print(f"  Nanowire length: {geometry.length_nm} nm")
    print(f"  Superconducting gap: {geometry.delta_0_mev} meV")
    print()
    
    # Initialize active evasion engine
    print("Initializing Active Evasion Engine...")
    engine = ActiveEvasionEngine(
        geometry=geometry,
        cosmic_params=cosmic,
        num_sites=100,
        magnetic_field_t=magnetic_field_t,
        random_seed=42
    )
    
    # Detection latency model
    print(f"  Detection latency: {engine.detection_latency_ns:.0f} ns")
    print(f"  Braiding time: {engine.braid_time_ns:.0f} ns")
    print(f"  Total evasion latency: {engine.detection_latency_ns + engine.braid_time_ns:.0f} ns")
    print()
    
    # Run simulation with active evasion
    duration_ms = 5.0
    dt_us = 10.0
    num_steps = int(duration_ms * 1000 / dt_us)
    
    print(f"Running {duration_ms} ms simulation ({num_steps} steps)...")
    print()
    
    results = engine.run_simulation_with_evasion(duration_ms * 1000, dt_us)
    
    # Statistics
    print("Simulation Complete")
    print("-" * 40)
    
    # Burst and evasion statistics
    num_bursts = len(engine.base_twin.burst_sim.burst_history)
    evasion_stats = engine.get_evasion_statistics()
    
    print(f"QP burst events detected: {num_bursts}")
    print(f"Active evasions executed: {evasion_stats['num_evasions']}")
    
    if evasion_stats['num_evasions'] > 0:
        print(f"  Mean evasion fidelity: {evasion_stats['mean_fidelity']:.3f}")
        print(f"  Success rate (>90% fidelity): {evasion_stats['success_rate']*100:.1f}%")
    print()
    
    # Survival multiplier calculation
    active_multiplier = engine.get_active_survival_multiplier()
    passive_multiplier = engine.base_twin.get_survival_multiplier()
    
    print("Survival Multiplier Analysis:")
    print(f"  Passive baseline: {passive_multiplier:.1f}x")
    print(f"  Active evasion:   {active_multiplier:.1f}x")
    print(f"  Algorithmic gain: {active_multiplier/max(passive_multiplier, 1):.1f}x")
    print()
    
    # Final Hamiltonian state
    final_gap = engine.base_twin.topological_gap_history[-1][1]
    is_topological = engine.base_twin.hamiltonian.is_in_topological_phase()
    
    print("Final State:")
    print(f"  Topological gap: {final_gap:.3f} meV")
    print(f"  In topological phase: {'Yes' if is_topological else 'No'}")
    print()
    
    # NASA SBIR Phase I Assessment
    print("=" * 60)
    print("NASA SBIR Phase I Feasibility Assessment")
    print("=" * 60)
    
    target_met = active_multiplier >= 10.0
    gap_maintained = final_gap > 0.01
    
    if target_met and gap_maintained:
        print("Status: [PASS] 10-100x survival multiplier achieved")
        print("        Topological protection maintained under flux")
    elif target_met:
        print("Status: [PARTIAL] Survival target met, gap marginal")
    else:
        print("Status: [FAIL] Survival multiplier below 10x target")
    
    print()
    print(f"Key Metric: {active_multiplier:.1f}x parity lifetime extension")
    print("=" * 60)


if __name__ == "__main__":
    main()
