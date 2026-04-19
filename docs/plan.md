SSIS Revolutionary Rebuild — Sovereign Space Intelligence Substrate
Transform the existing ADITI 4.0 MVP into a generation-defining, physics-native SDA/SSA platform by fixing all foundational issues, unifying the codebase, and layering revolutionary capabilities.

Current State Assessment
Build Status: BROKEN (3 compilation error sites)
Location	Error	Severity
engine/main.go:273,289	PersistentStore vs *catalog.Store type mismatch; wrong arg count for StartGRPCServer	🔴
internal/legacy_engine/conjunction.go:34+	Pointer-to-interface bug (*orbital.Propagator) — 10+ errors	🔴
pkg/common/auditlog.go:76,79	Missing fmt import	🟡
Test Status: 4 packages failing
Package	Failures	Root Cause
orbital	SGP4 altitude 4.8e13 km (should be ~400km); UKF Mahalanobis NaN; Oracle Iridium-Cosmos MOID off	TLE epoch handling / propagator bugs
catalog	Build failure	Missing math import in mht_full_test.go
physics	NRLMSISE-00 returns 0/NaN at 100-800km	GTD7D implementation returning zero densities
collision, spatial	✅ PASS	—
Architecture: 3 disconnected codebases
Go monolith (pkg, engine, cmd) — operational, 9/9 capabilities, but broken build
Axiom-Zero v3 (v3) — JAX rKAN kernel, SDA modules fully implemented (not stubs!), symbolic attribution
Legacy v1 (v1) — SFMN kernel, broken tests, debug spam — candidate for retirement
Key Strengths Already Present
BVH + Morton sweep-prune spatial acceleration in bvh.go
Foster Pc + Alfano + Monte Carlo 10⁶ in collision
Minkowski-sum geometric Pc in minkowski.go
Suborbital hypersonic lane (8-state EKF) in suborbital_lane.go
4D coverage tensor with weather denial in coverage_4d.go
rKAN kernel in pure Go in rkan_kernel.go
Ed25519 signing, mTLS 1.3, RBAC in grpc.go
Gobra formal verification specs in verification
ONNX Pattern-of-Life model in pattern_of_life.go
NRLMSISE-00 atmospheric model (68KB implementation) in physics
Symbolic distillation + LaTeX writer in symbolic
Rebuild Plan — 7 Phases
Phase 0: Foundation Fix (MUST DO FIRST)
Goal: Compilable, testable, all-green baseline

0.1. Fix auditlog.go — add missing "fmt" import
0.2. Fix main.go — correct StartGRPCServer call signature, resolve PersistentStore→Store adapter
0.3. Fix conjunction.go — replace *orbital.Propagator (pointer-to-interface) with orbital.Propagator (interface value)
0.4. Fix mht_full_test.go — add missing "math" import
0.5. Fix propagate_test.go — SGP4 returning absurd altitude; debug TLE epoch parsing in tle.go
0.6. Fix nrlmsise00.go — GTD7D returning 0/NaN; verify integration constants
0.7. Fix oracle_validation_test.go — Iridium-Cosmos MOID/velocity parity
0.8. Fix scale_test.go — UKF Mahalanobis NaN (likely SafeCholesky input guard)
0.9. Wire real Ed25519 signing in rest.go — replace "ED25519_SIG_PLACEHOLDER" (line 52)
0.10. Run go test ./... — all green before proceeding

Phase 1: Repo Unification & Cleanup
Goal: One coherent repo narrative, zero stray files

1.1. Delete stray root files — test_grpc.go, test_sgp4.go, get_sig.go (if present at root)
1.2. Move/delete patch_*.py from root into ai/training/migrations/ or delete if superseded
1.3. Move profiling artifacts (cpu.prof, mem.prof) into artifacts
1.4. Retire v1 — extract any unique kernels (SFMN, maneuver_distiller) into v3, delete the rest
1.5. Retire newai_v1 — fold unique IP into Axiom-Zero v3
1.6. Update top-level README.md — single coherent narrative: one system, one architecture, sovereign physics core
1.7. Update ARCHITECTURE.md — reflect unified architecture with Go ops plane + Axiom-Zero physics substrate
1.8. Add .golangci.yml — enforce lint rules, prevent future regressions

Phase 2: Axiom-Zero → Go Bridge (Sovereign Physics Core)
Goal: JAX-trained physics models running in pure Go at runtime, zero Python dependency

2.1. rKAN ONNX Export Pipeline — ai/training/export_rkan_onnx.py: train rKAN on real TLE (Starlink, COSMOS-2251), export via jax2tf → SavedModel → ONNX
2.2. Delta-rKAN Maneuver Attributor ONNX — export the maneuver residual model for Go consumption
2.3. POL Transformer ONNX — export pattern-of-life sequence model (6-layer, 256-dim from Axiom-Zero training)
2.4. Extend onnx session pool — register new model slots: axiom_rkan_residual, delta_rkan_attributor, pol_transformer
2.5. Real TLE Validation — benchmark rKAN ONNX vs SGP4 on Starlink-30000 + COSMOS-2251 debris; publish RMSE report in validation_reports
2.6. Go-native rKAN kernel upgrade — enhance rkan_kernel.go to support 13-feature manifest + Yoshida-4 integrator in pure Go (fallback when ONNX unavailable)
2.7. Propagator interface unification — single Propagator interface in orbital dispatching to SGP4, Axiom-rKAN-ONNX, or Axiom-rKAN-Go based on config

Phase 3: Revolutionary Capability Upgrades
Goal: Each of the 9 ADITI capabilities upgraded to exceed any known public SSA system

a) Volumetric Scan
3a.1. 4D time-swept scanning — extend volumetric.go with (lat, lon, alt, duration) → probability-of-presence heatmaps using UKF covariance
3a.2. Adaptive multi-resolution spatial hash — Hilbert-aligned Morton index for cache-friendly linear scans in dense regions

b) Neighbourhood Watch
3b.1. Reachability Risk Index (RRI) — per Indian asset, compute 1-impulse reachable set + Lambert-arc screening; flag foreign objects whose reachable set intersects within 48h
3b.2. Pattern-of-Life deviation scoring — integrate POL transformer inference to detect unusual adversary behavior

c) Conjunction Analysis
3c.1. Physics-informed covariance — replace static process noise with Axiom-Zero learned Hamiltonian residuals
3c.2. Diffusion-based UQ — pkg/collision/diffusion_uq.go for tail-risk Pc estimation beyond Gaussian assumptions

d) Collision Detection
3d.1. Minkowski-sum geometric Pc already implemented — validate + benchmark for non-spherical RSOs
3d.2. RSO shape catalogue ingestion — DISCOS/public datasets for tumble-state envelopes

e) Blind Spot Detection
3e.1. Multi-sensor fusion — extend coverage_4d.go with IAF radar locations (Nagpur, Leh, Port Blair), weather denial from IMD GRIB2
3e.2. Sensor-tasking RL optimization — wire rl PPO sensor policy to blind-spot output for optimal slew scheduling

f) Maneuver Detection
3f.1. Symbolic Maneuver Attribution — route high-confidence detections through Delta-rKAN → symbolic distillation → LaTeX explanation ("station-keep +0.12 m/s radial toward RISAT-2B")
3f.2. Maneuver fingerprint database — per-object Hamiltonian residual library for pattern-of-life attribution

g) Characterisation & Cataloguing
3g.1. POL Transformer integration — replace stub with ONNX transformer for object-class inference within 2 revolutions
3g.2. UCT escalation pipeline — uncorrelated track → POL anomaly → sensor tasking → classification

h) Re-entry Analysis
3h.1. Strategic-Asset Debris-Risk Map — fuse impact footprint with installation registry + IMD wind ensembles
3h.2. Tumble-coupled fragmentation model — NASA Breakup Model semi-analytic equivalent for break-up altitude estimation

i) Rapid Processing
3i.1. Batched SGP4 via gonum SIMD — vectorize propagation for 30k+ objects
3i.2. Hypersonic sub-orbital lane hardening — validate 8-state EKF at 1 Hz sustained with atmospheric coupling

Phase 4: Operator Experience Revolution
Goal: From CLI-only to world-class operational UX

4.1. Svelte + CesiumJS offline web GUI — 3D globe, object inspector, Pc dashboard, SITREP panel, blind-spot heatmap
4.2. Go Wails desktop wrapper — native macOS/Linux/Windows app, no Electron
4.3. Bubbletea TUI — rich terminal UI for air-gapped field deployment
4.4. "Dhruva" LLM SITREP — local Phi-3 ONNX generates Hindi/English voice situational reports every 60s
4.5. Training/Simulation mode — scripted scenarios (ASAT, rendezvous, debris cascade, hypersonic intercept) with scoring

Phase 5: Security, Certification & Hardening
Goal: IAF deployment-grade security and formal verification

5.1. SLSA-3 reproducible build — goreleaser + --mod=vendor + --trimpath + in-toto attestations
5.2. Extend Gobra coverage — beyond Foster Pc to SafeCholesky, Mahalanobis gating, ingest validators
5.3. CBMC model-checking for C-FFI boundaries (if CGO path taken)
5.4. Certification pack — docs/CERTIFICATION_PACK/ with HAZOP, FMEA for Pc pipeline, V&V matrix, DO-178C/IEC 61508 mapping
5.5. Quantum-resistant audit log — Merkle tree with ML-DSA signatures
5.6. Supply chain — full dependency license registry, SBOM with in-toto provenance

Phase 6: Moonshot Differentiators
Goal: Features no other SSA system on Earth has

6.1. Symbolic Law Gallery — live demo where Axiom-Zero rediscovers J₂/Kozai-Lidov from TLEs, prints LaTeX on screen
6.2. Counter-ASAT Gaming Mode — adversarial training scenarios with AI-driven adversary using rendezvous/shadowing tactics
6.3. "Reality Probe" Certification — evaluators bring own TLE, get signed 72h prediction manifest, verify later
6.4. Digital Twin Attack Scenario — PLA-ASAT targeting RISAT-2B demo with evasive maneuver recommendation
6.5. Edge deployment — ARM64 static binary for ruggedized boards (Rafale/AWACS), cached catalogue + 1 Hz hypersonic lane
6.6. Open Transparency Pack — publish Gobra proofs, benchmarks, reproducible build recipe in sealed evaluator repo

Implementation Order (within each phase, numbered by priority)
Phase 0 (Foundation)     ████████░░  ~1-2 weeks
Phase 1 (Unification)    ██████░░░░  ~1 week  
Phase 2 (Axiom Bridge)   ████████████ ~3-4 weeks
Phase 3 (9 Capabilities) ████████████████ ~4-6 weeks
Phase 4 (UX)             ████████████ ~3-4 weeks
Phase 5 (Security/Cert)  ████████░░  ~2-3 weeks
Phase 6 (Moonshots)      ██████████  ~2-4 weeks
Total estimated: 16-24 weeks for full revolutionary rebuild.

Critical Files to Create/Modify
New Files
pkg/collision/diffusion_uq.go — Diffusion-based tail-risk Pc
pkg/orbital/reentry_risk.go — Strategic-asset debris-risk map
pkg/hypersonic/ — Dedicated hypersonic sub-orbital module (refactored from suborbital_lane.go)
pkg/catalog/pol_transformer.go — POL transformer ONNX integration
ai/training/export_rkan_onnx.py — ONNX export pipeline
ai/training/export_pol_onnx.py — POL transformer export
cmd/ssa_gui/ — Svelte + CesiumJS + Wails desktop GUI
docs/CERTIFICATION_PACK/ — HAZOP, FMEA, V&V matrix
.golangci.yml — lint config
Key Modifications
main.go — fix build, add Axiom kernel loading
rest.go:52 — wire real Ed25519 signing
grpc.go — already has mTLS+RBAC, needs cleanup
auditlog.go — fix missing import
conjunction.go — fix pointer-to-interface
mht_full_test.go — fix missing import
rkan_kernel.go — upgrade to 13-feature + Yoshida-4
volumetric.go — add 4D time-swept scanning
neighbourhood.go — add RRI reachability
maneuver.go — add symbolic attribution hook
reentry.go — add strategic-asset risk layer
coverage_4d.go — add multi-sensor fusion + weather
nrlmsise00.go — fix GTD7D returning zeros
onnx — register Axiom-Zero model sessions
Success Metrics
Build: go build ./... — zero errors
Tests: go test ./... — all green, ≥80% coverage on new code
Scale: 100k objects, 168h propagation under 30 min CPU
Pc Accuracy: TCA error <15s over 72h; Pc CI width halved on tails
Maneuver F1: >0.95 detection, symbolic attribution within 5s
Hypersonic: 1 Hz sustained track update
Physics Honesty: Energy drift ≤10⁻¹⁴ on Axiom-Zero paths
Security: mTLS 1.3, Ed25519 on every artifact, RBAC enforced, zero placeholder strings
