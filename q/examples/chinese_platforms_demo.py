#!/usr/bin/env python3
"""
Chinese Deeptech Platform Demonstration

Showcases simulation capabilities for:
1. Iron-Based Superconductor (IBS) vortex lattices (CAS/Beijing LiFeAs, FeTeSe)
2. Three-site Kitaev chains (enhanced MZM stability vs two-site)
3. Partnership leverage with Chinese 15th Five-Year Plan priorities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bdg_hamiltonian import (
    DigitalTwinEngine, 
    CosmicRayParameters, 
    NanowireGeometry,
    IronBasedSuperconductorPlatform,
    KitaevChainParameters
)
import numpy as np


def demo_ibs_platforms():
    """Demonstrate IBS vortex lattice simulations."""
    print("\n" + "=" * 70)
    print("Iron-Based Superconductor (IBS) Platform Demo")
    print("Chinese CAS/Beijing: LiFeAs and FeTeSe MZM Lattices")
    print("=" * 70)
    
    for material in ["LiFeAs", "FeTeSe"]:
        print(f"\n[{material} Configuration]")
        
        if material == "LiFeAs":
            ibs = IronBasedSuperconductorPlatform(
                material="LiFeAs",
                lattice_size_x_nm=1000.0,
                lattice_size_y_nm=1000.0,
                vortex_spacing_nm=80.0,       # Tighter spacing = higher MZM density
                critical_temperature_k=18.0,
                superconducting_gap_mev=2.5,  # Larger gap than InAs/Al
                magnetic_field_t=1.5
            )
        else:  # FeTeSe
            ibs = IronBasedSuperconductorPlatform(
                material="FeTeSe",
                lattice_size_x_nm=1000.0,
                lattice_size_y_nm=1000.0,
                vortex_spacing_nm=100.0,
                critical_temperature_k=14.0,
                superconducting_gap_mev=2.0,
                magnetic_field_t=1.0
            )
        
        print(f"  Lattice: {ibs.lattice_size_x_nm}nm x {ibs.lattice_size_y_nm}nm")
        print(f"  Vortex spacing: {ibs.vortex_spacing_nm}nm")
        print(f"  Number of vortices: ~{ibs.num_vortices()}")
        print(f"  MZM density: {ibs.mzm_density_per_nm2():.2e} per nm²")
        print(f"  Superconducting gap: {ibs.superconducting_gap_mev} meV")
        print(f"  Critical temperature: {ibs.critical_temperature_k} K")
        print(f"  QP diffusivity: {ibs.qp_diffusivity_nm2_per_us} nm²/μs")
        
        # Compare to nanowire
        nanowire = NanowireGeometry(delta_0_mev=0.2)
        print(f"  → Gap advantage: {ibs.superconducting_gap_mev/nanowire.delta_0_mev:.1f}x larger than InAs/Al")
        print(f"  → Scalability: {ibs.num_vortices()} MZMs vs 2 (wire ends) per device")


def demo_kitaev_chains():
    """Demonstrate three-site Kitaev chain stability enhancement."""
    print("\n" + "=" * 70)
    print("Three-Site Kitaev Chain Demo")
    print("Enhanced MZM Stability for H-Tile Tetron Rerouting")
    print("=" * 70)
    
    # Compare 2-site vs 3-site Kitaev chains
    for num_sites in [2, 3]:
        print(f"\n[{num_sites}-Site Kitaev Chain]")
        
        kitaev = KitaevChainParameters(
            num_sites=num_sites,
            site_spacing_nm=50.0,
            t_hopping_mev=1.0,
            delta_pairing_mev=0.5,
            mu_mev=0.0
        )
        
        print(f"  Topological phase: {'YES' if kitaev.is_topological() else 'NO'}")
        print(f"  Localization length: {kitaev.localization_length_nm():.1f} nm")
        
        if num_sites == 3:
            # Three-site chains have reduced wavefunction overlap
            overlap_reduction = np.exp(-2 * kitaev.site_spacing_nm / kitaev.localization_length_nm())
            print(f"  Wavefunction overlap: {overlap_reduction:.2e} (exponentially suppressed)")
            print(f"  → Direct application to H-tile tetron rerouting")
            print(f"  → Broader topological parameter space")


def demo_radiation_resilience():
    """Demonstrate QP poisoning under continuous space flux."""
    print("\n" + "=" * 70)
    print("Radiation Resilience Comparison: IBS vs Nanowire")
    print("Deep Space: 100 mSv/hr Continuous Flux")
    print("=" * 70)
    
    cosmic = CosmicRayParameters(
        dose_rate_msv_hr=100.0,
        qp_lifetime_us=100.0,
        mean_energy_deposition_mev=50.0,
        qp_generation_efficiency=0.57
    )
    
    platforms = [
        ("InAs/Al Nanowire", NanowireGeometry(delta_0_mev=0.2), None),
        ("LiFeAs IBS", None, IronBasedSuperconductorPlatform(
            material="LiFeAs",
            superconducting_gap_mev=2.5,
            qp_diffusivity_nm2_per_us=500.0
        )),
    ]
    
    for name, nanowire_geo, ibs_platform in platforms:
        print(f"\n[{name}]")
        
        if nanowire_geo is not None:  # Nanowire
            geo = nanowire_geo
            twin = DigitalTwinEngine(
                geometry=geo,
                cosmic_params=cosmic,
                num_sites=30,
                random_seed=42
            )
            gap_meV = geo.delta_0_mev
        else:  # IBS
            ibs = ibs_platform
            # Use nanowire as base but with IBS parameters
            geo = NanowireGeometry(delta_0_mev=ibs.superconducting_gap_mev)
            twin = DigitalTwinEngine(
                geometry=geo,
                cosmic_params=cosmic,
                num_sites=30,
                random_seed=42
            )
            gap_meV = ibs.superconducting_gap_mev
        
        # Run short simulation
        results = twin.run_simulation(duration_us=500, dt_us=10)
        
        # Metrics
        survival_mult = twin.get_survival_multiplier()
        final_gap = results[-1]["topological_gap_mev"]
        total_bursts = len(twin.burst_sim.burst_history)
        
        print(f"  Bare gap: {gap_meV} meV")
        print(f"  Survival multiplier: {survival_mult:.1f}x")
        print(f"  QP bursts (0.5ms): {total_bursts}")
        print(f"  Gap retention: {final_gap/gap_meV*100:.1f}%")
        
        # Critical insight: Larger gap helps but doesn't prevent poisoning
        print(f"  → Larger gap extends QP thermalization time but continuous flux still saturates")


def partnership_strategy():
    """Outline Chinese partnership strategy."""
    print("\n" + "=" * 70)
    print("China 15th Five-Year Plan Partnership Strategy")
    print("=" * 70)
    
    print("\n[Chinese Strengths]")
    print("  • IBS vortex lattices: Large-scale ordered MZM arrays (CAS/Beijing)")
    print("  • Nanowire theory: Non-Abelian braiding resilience (SUSTech Lu Haizhou)")
    print("  • Three-site Kitaev: Enhanced stability over minimal chains (2025 realization)")
    print("  • Photonic efficiency: 71%+ single-photon sources (USTC Pan group)")
    print("  • National priority: 1000+ qubit systems, space quantum networks")
    
    print("\n[Global Gap - Unsolved Everywhere]")
    print("  • Continuous QP entropy pump under space flux: NO SOLUTION")
    print("  • Passive phonon sinks: SATURATE under 100 mSv/hr")
    print("  • Readout latency: ms-scale << 100 μs poisoning window")
    print("  • Monolithic Majorana-photonic: NOT REALIZED")
    
    print("\n[Asymmetric Moat]")
    print("  Chinese device fabrication + QIDT active evasion = rad-hard deployment")
    print("  • License twin as design tool for partner platforms")
    print("  • MIIT open competition funding for space quantum networks")
    print("  • Joint Sino-Indian proposals under space autonomy priorities")
    
    print("\n[Target Partnerships]")
    partners = [
        ("SUSTech (Lu Haizhou)", "Nanowire theory, Kitaev chain validation"),
        ("CAS/Beijing IBS groups", "LiFeAs/FeTeSe lattice testbeds"),
        ("USTC (Pan group)", "Photonic readout for sub-nW I/O"),
        ("SJTU/TDLI", "MZM conference network, Dali ties"),
    ]
    
    for org, focus in partners:
        print(f"  • {org}: {focus}")
    
    print("\n[Monetization Path]")
    print("  Phase 1: Open-source core attracts CAS/SUSTech outreach")
    print("  Phase 2: SaaS rad-hard modeling for in-space manufacturing ($6B+ market)")
    print("  Phase 3: Autonomy kernel licensing for mega-constellations")


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Quantum Immune Digital Twin")
    print("Chinese Deeptech Platform Integration Demo")
    print("April 2026 - Phase 0 Deliverable")
    print("=" * 70)
    
    demo_ibs_platforms()
    demo_kitaev_chains()
    demo_radiation_resilience()
    partnership_strategy()
    
    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print("\nKey Insight:")
    print("  Chinese platforms (IBS lattices, 3-site Kitaev) excel at device fabrication")
    print("  and stability, but continuous QP poisoning in space remains UNSOLVED.")
    print("  The Quantum Immune Digital Twin provides the ACTIVE EVASION layer")
    print("  that Hamiltonian-matches the entropy pump at 100 mSv/hr.")
    print("\n  → Partnership: Chinese hardware + QIDT software = rad-hard advantage")
    print("=" * 70)


if __name__ == "__main__":
    main()
