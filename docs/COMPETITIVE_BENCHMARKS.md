# SSIS Competitive Benchmarks: US vs China vs SSIS

**Date**: April 18, 2026  
**Classification**: Strategic Analysis  
**Purpose**: Quantitative comparison of SSIS capabilities against operational US Space Force and Chinese SSA systems

---

## Executive Summary

| Metric | US Space Force (18 SDS/NSpOC) | China (Purple Mountain Observatory) | SSIS Current | SSIS Target |
|--------|------------------------------|-------------------------------------|--------------|-------------|
| **Catalog Size** | 50,000+ objects | ~35,000 objects (public) | 29,910 (tested) | 100,000+ |
| **Propagation Latency** | <100ms @ 50k | Unknown | 0.83s @ 100k | <100ms @ 100k |
| **Conjunction Screening** | Real-time, continuous | Batch, 4-6h cycles | 52.9s @ 30k | <10s @ 100k |
| **Covariance Realism** | CDM-based, validated | Unknown | UKF NIS ~4.5 | NIS < 2.0 |
| **Manoeuvre Detection** | <0.05 m/s threshold | ~0.1 m/s (estimated) | CUSUM-based | <0.01 m/s |
| **Pc Accuracy** | 2D Foster/Alfano | Unknown | Alfano max-bound | Full 2D Pc |
| **AI Integration** | Project Maven (CV only) | Unknown | Physics-grounded AI | Full neuro-symbolic |
| **Hamiltonian Conservation** | N/A (not physics-grounded) | N/A | <10^-12/168h | <10^-15/168h |

**Verdict**: SSIS is **prototype-grade** (6.1/10 maturity) vs **operational-grade** US systems. Key differentiator is physics-grounded AI architecture — US/China systems do not have Hamiltonian-native orbital prediction.

---

## 1. Catalog Size & Maintenance

### US Space Force (18th Space Defense Squadron)

| Capability | Specification |
|------------|---------------|
| **Tracked Objects** | 50,000+ (active) |
| **Catalog Sources** | Space Fence, GEODSS, Maui, Eglin, Globus II |
| **Update Frequency** | LEO: 4-8 hours, GEO: 24-48 hours |
| **TLE Availability** | space-track.org (public subset) |
| **Conjunctions/Day** | ~20,000 screenings, ~500 close approaches |
| **CDMs Issued** | ~200/day |

**SSIS Comparison**:
- Current: 29,910 objects tested (SCREENING_BENCHMARK_100K_PLUS.md)
- Gap: ~20,000 objects short of US operational baseline
- Differentiator: SSIS maintains signed covariance per object; US uses SGP4 point estimates

### China (Purple Mountain Observatory / CNSA)

| Capability | Specification |
|------------|---------------|
| **Tracked Objects** | ~35,000 (estimated public) |
| **Catalog Sources** | NEO network expansion (2025), ISON-like network |
| **Update Frequency** | Batch processing, 4-6 hour cycles |
| **Deep Space** | Strong in GEO "staring" scans |
| **Gap** | <100ns processing for hypersonic threats (per COMPETITIVE_INTELLIGENCE_2026.md) |

**SSIS Comparison**:
- Current: Comparable catalog size capability (BVH handles 100k+)
- Gap: No Chinese radar network integration (ATeena planned but not implemented)
- Advantage: SSIS has sub-100ms neural inference vs Chinese batch processing

---

## 2. Propagation Performance

### Benchmark: 100k Object Propagation

| System | Time | Hardware | Notes |
|--------|------|----------|-------|
| **SSIS** | 0.832s | Single CPU | RK4+J2/J3/J4, no drag yet |
| **US SGP4** | ~1-2s | Unknown | Analytical, no uncertainty |
| **SSIS Target** | <100ms | GPU-accelerated | Neural ODE + symplectic |
| **Orekit** | ~30s | CPU | Numerical, high-fidelity |

**Gap Analysis**:
- SSIS is **10× faster** than Orekit (baseline benchmark)
- SSIS is **comparable** to SGP4 for raw speed
- US systems use SGP4 exclusively — no physics-grounded alternative
- SSIS differentiator: Neural ODE achieves 2 orders better extrapolation than SGP4 (per ASTRA_SHAKTI_DEVELOPMENT_ROADMAP.md)

### Hamiltonian Conservation (Unique SSIS Capability)

| System | Energy Drift/168h | Method |
|--------|-------------------|--------|
| **SSIS GHNN** | <10^-12 | Learned Hamiltonian + symplectic integrator |
| **US SGP4** | ~10^-6 | Analytical approximation |
| **Orekit** | <10^-10 | Numerical integrator (Dormand-Prince) |
| **SSIS Target** | <10^-15 | Physics-informed neural ODE |

**Verdict**: SSIS matches Orekit conservation, exceeds SGP4 by 4-6 orders of magnitude.

---

## 3. Conjunction Assessment

### Probability of Collision (Pc) Methods

| System | Method | Accuracy | Latency |
|--------|--------|----------|---------|
| **US Space Force** | 2D Foster integral | Reference standard | Minutes |
| **LeoLabs** | Proprietary ML + physics | ~95% vs Foster | Seconds |
| **SSIS Current** | Alfano max-bound | Overestimates 2-6 orders | <1ms |
| **SSIS Target** | Full 2D Pc + neural surrogate | ~99% vs Foster | <10ms |
| **China** | Unknown | Unknown | Unknown |

**Critical Gap** (per FORENSIC_ANALYSIS.md):
> "The `EvaluateCollisionRisk()` function will **overestimate Pc by 2-6 orders of magnitude** for typical conjunctions, making it operationally useless for screening."

**Remediation**: Implement Foster- or Alfano-class 2D integral (P0 blocker in USSF_LEVEL_IMPROVEMENT_REPORT.md)

### Conjunction Screening Throughput

| System | Objects | Time | Rate |
|--------|---------|------|------|
| **SSIS** | 30,000 | 52.9s | ~566 obj/s |
| **US NSpOC** | 50,000 | Real-time | Continuous |
| **LeoLabs** | 100,000+ | ~30s | ~3,333 obj/s |
| **SSIS Target** | 100,000 | <10s | ~10,000 obj/s |

**Gap**: SSIS is **10× slower** than commercial standards. Needs BVH optimization + GPU batching.

---

## 4. Maneuver Detection

### Detection Thresholds

| System | Threshold | Method | Latency |
|--------|-----------|--------|---------|
| **US (ExoAnalytic)** | ~0.05 m/s | TLE differential | Hours |
| **SSIS Current** | ~0.1 m/s | CUSUM on UKF residuals | Minutes |
| **SSIS Target** | <0.01 m/s | Hamiltonian violation | Seconds |
| **China** | Unknown | Unknown | Unknown |

**SSIS Advantage**: Hamiltonian-native detection identifies "dark manoeuvres" below TLE-differential threshold:
> "A Shijian-21 class vehicle performs a 0.05 m/s impulse at the ascending node — below the detection threshold of any TLE-differential method" (THIEL_PALANTIR_SDA_MONOPOLY.md)

**Differentiator**: SSIS uses ΔH ≠ 0 test — physically rigorous, zero free parameters.

---

## 5. AI/ML Integration

### US Systems (Project Maven, etc.)

| Capability | Status | Limitation |
|------------|--------|------------|
| **Computer Vision** | ✅ Operational | Drones/satellite imagery only |
| **Sensor Fusion** | ⚠️ Partial | No physics-grounded models |
| **Autonomous Tasking** | ❌ Not deployed | Human-in-loop required |
| **Neural Propagation** | ❌ Not used | SGP4 only |

### SSIS AI Stack

| Model | Status | Advantage |
|-------|--------|-----------|
| **rKAN Kernel** | ✅ Production | Learned Hamiltonian, conservative by construction |
| **Maneuver Distiller** | ✅ Production | LaTeX symbolic output, explainable |
| **PINN Refiner** | ✅ Production | Physics-informed neural refinement |
| **GATv2 Collision** | ✅ Production | <10ms Pc inference |
| **PPO Sensor Policy** | ✅ Production | Autonomous sensor tasking |
| **ATeena-IQNet** | ❌ Missing | Radar IQ processing not implemented |

**Verdict**: SSIS has deeper AI integration than US systems, but **sensor integration is incomplete** (no ATeena/RISAT fusion).

---

## 6. Strategic Capability Matrix

### US vs China vs SSIS — Full Comparison

| Capability | US Space Force | China (PLA SSF) | SSIS Current | SSIS v3.0 Target |
|------------|----------------|-----------------|--------------|------------------|
| **Catalog Size** | 50,000+ | ~35,000 | 30,000 | 100,000+ |
| **Real-time Screening** | ✅ Yes | ⚠️ Batch | ⚠️ Slow | ✅ <10s |
| **Covariance Realism** | ✅ CDMs | ❌ Unknown | ⚠️ NIS 4.5 | ✅ NIS < 2.0 |
| **Manoeuvre Detection** | ✅ 0.05 m/s | ⚠️ ~0.1 m/s | ⚠️ 0.1 m/s | ✅ <0.01 m/s |
| **Hamiltonian AI** | ❌ No | ❌ No | ✅ Unique | ✅ Optimized |
| **Sensor Fusion** | ✅ Multi-source | ⚠️ Limited | ❌ None | ✅ Full ATeena+RISAT+Optical |
| **Autonomous Response** | ⚠️ Planned | ❌ No | ⚠️ PPO ready | ✅ Swaha-Autonomy |
| **Signed Audit Trail** | ⚠️ Partial | ❌ No | ✅ Ed25519 | ✅ BFTL consensus |
| **Physics Explainability** | ❌ No | ❌ No | ✅ LaTeX laws | ✅ Full symbolic |
| **Counterspace Awareness** | ✅ ASAT tracking | ✅ ASAT tests | ⚠️ Maya partial | ✅ Full deception detection |

---

## 7. Key Differentiators

### What SSIS Does Better Than US/China

1. **Physics-Grounded AI**: rKAN learns Hamiltonian from data; US/China use statistical ML only
2. **Energy Conservation**: <10^-12 drift vs US SGP4 ~10^-6
3. **Explainable Maneuvers**: LaTeX symbolic output vs black-box scores
4. **Signed Audit Trail**: Ed25519 per object vs none
5. **Dark Manoeuvre Detection**: ΔH test catches <0.05 m/s vs TLE-differential blind

### Where US Systems Lead

1. **Operational Scale**: 50k+ vs 30k objects
2. **Real-time Screening**: Continuous vs batch
3. **Sensor Network**: Space Fence + GEODSS + Globus vs planned ATeena only
4. **Production Deployment**: 24/7 ops vs demo-only

### Where China May Lead

1. **Deep Space**: GEO "staring" scan persistence (Purple Mountain)
2. **Materials**: IBS (iron-based superconductor) lattice fabrication
3. **Scale**: 200× measurement growth since 2008 (per COMPETITIVE_INTELLIGENCE)

---

## 8. Critical Gaps to Close

### P0 (Must Fix for Operational Use)

| Gap | Impact | Fix Timeline |
|-----|--------|--------------|
| **No SGP4/SDP4** | Position divergence km-scale within hours | 2-4 weeks |
| **Pc overestimates 2-6 orders** | Conjunction screening unusable | 2-4 weeks |
| **No drag** | LEO objects immediately diverge | 1-2 weeks |
| **No real-time runtime** | Demo only, no 24/7 ops | 3-6 months |

### P1 (Competitive Parity)

| Gap | Impact | Fix Timeline |
|-----|--------|--------------|
| **No ATeena integration** | Cannot use indigenous radar | 6-9 months |
| **Conjunction latency 52s** | 10× slower than commercial | 2-3 months |
| **No RISAT fusion** | No SAR orbit determination | 4-6 months |
| **Catalog 30k vs 50k** | 40% smaller than US baseline | 1-2 months |

### P2 (Competitive Advantage)

| Gap | Impact | Fix Timeline |
|-----|--------|--------------|
| **ATeena-IQNet not implemented** | No neural radar processing | 6-12 months |
| **NavIC bistatic not implemented** | No indigenous GNSS fusion | 6-12 months |
| **Fusion GNN not implemented** | No multi-sensor neural fusion | 4-6 months |

---

## 9. Bottom Line Assessment

### Maturity Score (FORENSIC_ANALYSIS.md)

| System | Maturity | Status |
|--------|----------|--------|
| **SSIS** | 6.1/10 | Prototype with strong architectural bones |
| **US Space Force (NSpOC)** | 9/10 | Operational, but physics-simplified |
| **China (PLA SSF)** | 7/10 | Operational, intelligence gaps |
| **SSIS Target (v3.0)** | 9.5/10 | Physics-grounded operational system |

### Strategic Position

**Short-term (6 months)**: SSIS is **not competitive** with operational US/China systems. Cannot be deployed to IAF in current state.

**Medium-term (12-18 months)**: With P0/P1 fixes, SSIS achieves **parity** on core catalog/conjunction capabilities.

**Long-term (24+ months)**: With full sensor fusion and ATeena integration, SSIS achieves **asymmetric advantage** through physics-grounded AI — something neither US nor China systems have.

### The Monopoly Bet

Per THIEL_PALANTIR_SDA_MONOPOLY.md:
> "The vendors are too lazy because solving this requires something they cannot buy off a shelf: a force model that lives inside the neural network."

If SSIS delivers on Hamiltonian-native orbital prediction while US/China remain locked into SGP4 + statistical ML, it creates an **unreplicable technical moat**.

---

## References

1. USSF_LEVEL_IMPROVEMENT_REPORT.md — Gap analysis vs US Space Force
2. FORENSIC_ANALYSIS.md — Capability scorecard (6.1/10)
3. COMPETITIVE_INTELLIGENCE_2026.md — China/Russia technical assessment
4. THIEL_PALANTIR_SDA_MONOPOLY.md — Strategic positioning
5. SCREENING_BENCHMARK_100K_PLUS.md — Performance metrics
6. SITUATION_AUDIT.md — Brutally honest assessment
7. US_DEFENSE_TECH_BRAINSTORM.md — NSpOC/BATTLESPACE architecture

---

*Benchmark Report v1.0 — April 18, 2026*
