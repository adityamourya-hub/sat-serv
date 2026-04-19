ADITI 4.0 — Strategic Elevation Plan for SSIS + Axiom-Zero
Context
The repository at idex_iaf/ is a submission for the Indian Air Force ADITI 4.0 PS24 challenge: an indigenous Space Domain Training, Surveillance & Operations tool covering 9 operational capabilities (Volumetric Scan, Neighbourhood Watch, Conjunction, Collision Pc, Blind Spot, Manoeuvre Detection, Characterisation/Cataloguing, Re-entry, Rapid 30k-object Processing) plus 5 non-functionals (IAF-network compatible, open-source/indigenous data, SDLC+certification, scalable/maintainable, perpetual licensing).

Three parallel codebases currently exist:

ssis/ — Go monolith, single-binary ssa_ops_runner. All 9 capabilities implemented end-to-end with classical orbital mechanics (SGP4, UKF, Foster Pc, Jacchia-Roberts, octree, Ed25519-signed artifacts). This is the primary deliverable and is submission-grade.
newai_v1/ — Python research sandbox: Symplectic Flow-Matching Network (SFMN) = rational-KAN Hamiltonian + Yoshida-4 integrator + symplectic adjoint. Novel physics-informed ML for orbital propagation. Not integrated. Broken tests, DEBUG spam, hardcoded paths.
newai_v3/ — axiom-zero v3.0.0 package: cleaner Flax rewrite of v1 (Yoshida-8, Shifted Legendre rKAN, symbolic distillation, DLPack bridge, float64-enforced). Stronger numerics (energy drift 1e-14) but SDA modules are protocol stubs. Not integrated.
The strategic gap: ssis wins the challenge as-is. newai_v1/v3 contain genuine indigenous algorithmic IP (symplectic Hamiltonian discovery, symbolic distillation of perturbations) that — if fused with ssis — becomes the decisive differentiator versus foreign benchmarks (COMSPOC, LeoLabs, ExoAnalytic). Today they operate as parallel universes.

This plan's intended outcome: elevate from "9/9 capabilities ticked" to "years ahead of anything evaluators have seen" by (a) hardening ssis, (b) fusing Axiom-Zero into it as a Sovereign Physics Core, and (c) layering revolutionary operator experience, certification, and moonshot features.

1. Executive Summary
Current maturity: SSIS is at ~TRL 6 for all 9 capabilities (operationally demonstrated in relevant environment) with strong defence-grade scaffolding (air-gap, signed artifacts, Gobra formal verification hooks, 100k-object scale tests, SBOM, Makefile, goreleaser). Axiom-Zero is at TRL 3 (research validation in lab). The delta to "winning standard" is not capability completeness — it is (1) fused presentation as one coherent system, (2) an unmistakably Indian algorithmic moat, (3) an operator UX that turns junior analysts into experts, (4) certification artefacts beyond demos, (5) one or two moonshot demonstrations that anchor the evaluators' memory.

Verdict: Submission-viable today. Submission-winning after the 90-day plan in §7.

2. Repo Analysis
2.1 Strengths
ssis/pkg is cleanly partitioned: orbital/, collision/, catalog/, ml/, octree/, physics/, crypto/, verification/, spatial/, server/, persistence/, rl/.
Foster Pc + Alfano fallback + Monte-Carlo 10⁶ gives three-tier collision assurance; NaN/Inf-guarded SafeCholesky + Mahalanobis.
Hybrid ONNX + classical deterministic fallback in pkg/ml/ — a defence-procurement-friendly pattern (AI augments, never replaces, classical).
Formal-verification hook in pkg/verification/pc_formal.go using Gobra — rare in the SSA domain.
Comprehensive docs/ folder: ADR-002-HYPERION, HARDENING_COMPLIANCE_MATRIX, OPERATOR_MANUAL, SBOM, validation_reports.
Novel research IP in newai_v3/axiom_zero: Yoshida-8 with det(J)=1±1e-15, rKAN rational Padé basis over Shifted Legendre, symbolic distillation to LaTeX laws, zero-copy DLPack JAX↔Torch bridge. Genuinely original for orbital SSA.
ssis is air-gap ready, single static binary, no external calls.
2.2 Critical Gaps
#	Gap	Severity	Location
G1	Axiom-Zero not integrated into ssis — no gRPC/FFI/cgo bridge	🔴 HIGH	repo-wide
G2	patch_foster.py, patch_main.py, patch_main2.py, patch_main3.py (103 LOC total) — Python post-hoc patches at ssis root	🟡 MED	ssis/
G3	test_grpc.go, test_sgp4.go, get_sig.go at ssis root (not under cmd/ or internal/)	🟡 MED	ssis/
G4	scratch/ folder (check_tle.go, inspect_tle.go) in delivery tree	🟡 MED	ssis/scratch/
G5	gRPC server uses "ED25519_SIG_PLACEHOLDER" string — signing not wired for RPC path	🔴 HIGH	ssis/pkg/server/
G6	No TLS on gRPC; no RBAC/mTLS	🔴 HIGH	ssis/pkg/server/
G7	Conjunction screening is octree range + all-pairs within cell, not true k-d or BVH; O(n²) worst-case in dense clusters	� LOW	ssis/pkg/collision/ [RESOLVED: Replaced with BVH + Morton sweep-prune for O(n log n) performance]
G8	No true GUI; operator interacts via CLI + signed JSON. Adequate for demo, weak for "wow" factor	🔴 HIGH	repo-wide
G9	newai_v1 test suite is broken (test_alpha.py imports non-existent classes); debug prints in every hot path	🟡 MED	newai_v1/test/
G10	newai_v3 SDA modules (sda/ — volumetric, neighbourhood, conjunction, maneuver, reentry) are Protocol stubs with pass bodies	🔴 HIGH	newai_v3/axiom_zero/sda/
G11	Benchmarks in newai_v3 use Hénon-Heiles synthetic trajectories — never validated against real TLE/SGP4	🔴 HIGH	newai_v3/benchmarks/
G12	No supply-chain provenance: SBOM exists but no in-toto/SLSA attestation; no signed reproducible build attestation	🟡 MED	ssis/
G13	Pattern-of-Life model is lightweight stub, not a real sequence model	🟡 MED	ssis/pkg/catalog/
G14	No sensor-tasking simulation loop for training mode	🟡 MED	repo-wide
G15	cpu.prof, mem.prof in ssis root — stale profiling artefacts, should be in artifacts/	🟢 LOW	ssis/
2.3 Risks
Strategic narrative risk: Evaluators seeing three folders (ssis, newai_v1, newai_v3) may infer a fractured project. A single coherent story is essential.
Over-promising risk: SOVEREIGN_MANIFESTO.md-style language in newai_v1 ("First Honest AI") reads as marketing. Evaluators prefer measurable claims.
Benchmark-gap risk: Axiom-Zero claims 10⁻¹² energy drift on Hénon-Heiles but no real-TLE comparison. If asked "show me RMSE vs SGP4 on COSMOS-2251 debris", there is no answer today.
Ops-comprehension risk: A CLI demo undersells the system. IAF panels reward operationally legible UX.
3. Revolutionary Capability-by-Capability Enhancements
Format: Current → Revolutionary → Technical approach → KPI

a) Volumetric Scan
Current: Conical ROI scan with octree + horizon masking in pkg/orbital/volumetric.go.
Revolutionary: Adaptive Multi-Resolution Spatial Hash Grid that dynamically refines voxel density around high-interest regions (LAC hotspots, ASAT debris fields). Add 4D volumetric scanning — time-swept scanning where operators define (lat, lon, alt, duration) and the tool returns probability-of-presence heatmaps using UKF covariance Minkowski-summed with ROI voxels.
Approach: Extend octree with HaMi (Hilbert-aligned Morton index) for cache-friendly linear scans; add density adaptation driven by object count/volume ratio; probability field = Gaussian mixture over UKF posteriors.
KPI: <100 ms for 1000 km³ ROI over 24 h sweep with 30k-object catalogue.
b) Neighbourhood Watch
Current: ScreenIndianAssets() in pkg/orbital/neighbourhood.go, 500 km radius, 3-tier threat scoring.
Revolutionary: Active Threat Surface Model — for each Indian asset (RISAT, CARTOSAT, GSAT, EMISAT), continuously compute an adversary-reachability shell using Axiom-Zero's symplectic reachability cone (from newai_v1/rapid_ssa_engine.py:95 but generalized to 60+ directions and delta-v budget stratification). Any foreign object whose reachable set intersects the Indian asset shell within 48 h is flagged with a Reachability Risk Index (RRI).
Approach: Per-asset 1-impulse reachable-set precompute; 3-impulse Lambert-arc screening; integrate CSpOC conjunction-grade heuristics; colour-code by PoL deviation (is this adversary object behaving unusually?).
KPI: All ~35 critical Indian assets screened vs full catalogue every 60 s; false-positive rate < 5 %.
c) Conjunction Analysis
Current: Octree range + pairwise Mahalanobis + Foster 2D Pc + Alfano fallback + MC 10⁶.
Revolutionary: Physics-Informed Uncertainty — replace static process-noise covariance with Axiom-Zero learned Hamiltonian residuals (Delta-rKAN from newai_v1/maneuver_distiller.py) so covariance reflects actually observed perturbation structure (solar flux, drag, unmodeled third-body). Add Diffusion-based Uncertainty Quantification (per newai_v3/docs/ROADMAP.md Frontier) for tail-risk Pc estimation where Gaussian assumptions break.
Approach: Gobra-verified replacement of math.Cholesky with the SafeCholesky variant; add pkg/collision/diffusion_uq.go that wraps ONNX diffusion head; true BVH (bounding volume hierarchy) replacing all-pairs-in-cell (G7).
KPI: TCA prediction error < 15 s over 72 h horizon; Pc 95 % CI width halved on low-Pc tails.
d) Collision Detection
Current: Foster 2D (Gobra-verified) + Alfano + MC 10⁶.
Revolutionary: Minkowski-Sum Geometric Pc for large, non-spherical objects (rocket bodies, deployed solar panels) where spherical-approx Pc systematically under-counts. Precompute tumble-state envelope; Minkowski-sum the two envelopes along relative velocity; volumetric integral of joint PDF over the convex result.
Approach: pkg/collision/minkowski.go using libqhull-backed convex hulls (CGO) or pure-Go implementation; pair with RSO shape catalogue (ingested from DISCOS/public datasets).
KPI: ≥30 % improvement in Pc fidelity for objects with aspect ratio > 3:1.
e) Blind Spot Detection
Current: 180×360 lat/lon grid, satellite footprint marking in pkg/orbital/coverage.go.
Revolutionary: Multi-Sensor Fusion Blind-Spot Tensor — combine IAF ground radars (Nagpur, Leh, Port Blair notional), proposed space-based optical (from IAF ISRO collaboration), and allied sensors into a time-dependent 4D coverage tensor C(lat, lon, alt, t). Integrate weather denial (NWP cloud data from IMD) to compute effective coverage vs nominal. Output: shadow-zone ranking + sensor-tasking optimization (reuses pkg/rl/ PPO sensor policy).
Approach: Extend coverage.go to tensor; load IMD GRIB2 offline cache; formulate as constrained submodular maximization for sensor slew scheduling.
KPI: Shadow-zone prediction accuracy > 92 % over 24 h; RL policy improves catalogue completeness by > 10 %.
f) Manoeuvre Detection
Current: CUSUM + ONNX hybrid in pkg/orbital/maneuver.go.
Revolutionary: Symbolic Manoeuvre Attribution — when a foreign satellite manoeuvres, Axiom-Zero's Delta-rKAN residual learning + symbolic distillation outputs a human-readable LaTeX explanation of the manoeuvre ("station-keep +0.12 m/s radial", "plane-change 2.4° toward longitude 78.96°E", "rendezvous burn toward RISAT-2B"). No foreign system does this.
Approach: Route every high-confidence manoeuvre event into axiom_zero symbolic pipeline; cache learned per-object Hamiltonian residual libraries; maintain "manoeuvre fingerprint" database for pattern-of-life attribution.
KPI: Manoeuvre detection F1 > 0.95; symbolic attribution within 5 s; human-interpretable ≥80 % of events.
g) Characterisation & Cataloguing
Current: MHT-lite + pkg/catalog/mht_full.go, rapid-recatalog, track association.
Revolutionary: Transformer Pattern-of-Life (POL) Sequence Model replacing the stub (G13). Self-supervised on the catalogue's own orbit history to predict "normal" behaviour per object class (LEO-comms, MEO-navigation, GEO-stationary, HEO-Molniya). Uncorrelated-track escalation grade = P(observation | POL) < threshold. Combine with Axiom-Zero orbital-element extraction for class-inference on unknown uncorrelated tracks within 2 revs.
Approach: Train Decoder-only Transformer (6 layer, 256 dim) on state-vector tokens (via newai_v3/axiom_zero/tokeniser); ONNX export; integrate to catalog store.
KPI: UCT classification top-1 accuracy > 78 % within 2 revs; POL anomaly AUC > 0.93.
h) Re-entry Analysis
Current: Jacchia-Roberts + RKF45 + Halton QMC dispersion in pkg/orbital/reentry.go.
Revolutionary: Strategic-Asset Debris-Risk Map — fuse re-entry impact ellipsoid with MoD strategic-installation registry (user-loaded encrypted JSON) and IMD surface-wind ensembles. Output ranked "Threat-to-Asset" list with evacuation-timing advisory. Add tumble-coupled fragmentation model for break-up altitude estimation using public RCS curves.
Approach: pkg/orbital/reentry_risk.go layers installation GIS onto 3-σ footprint; NRLMSISE-00 already present; break-up via NASA-Breakup-Model semi-analytic equivalent.
KPI: Impact-ellipsoid 1-σ radius < 120 km for 72 h prediction; strategic-asset risk triage within 30 s of TIP issue.
i) Rapid Processing (30k+ objects + hypersonics)
Current: Octree + hash grid + ONNX ranker; scale tests at 100k / 168 h.
Revolutionary: Hypersonic Suborbital Adjunct — dedicated sub-orbital threat lane handling Mach 5–25 glide vehicles with 1 Hz update; ensemble of (a) atmospheric drag-coupled integrator, (b) learned trajectory prior (small transformer), (c) measurement-update Kalman. Add GPU SIMD batch-propagation via ONNX-GPU or a CUDA kernel for SGP4 ensembles (10⁵ objects @ <100 ms).
Approach: New pkg/hypersonic/ module; vectorize pkg/orbital/rapid_processing.go; evaluate [go-hep/hbook] + gonum SIMD intrinsics; fallback path stays CPU.
KPI: 30k-object propagation < 200 ms CPU, < 30 ms GPU; hypersonic track update rate 1 Hz sustained.
Non-Functional — Revolutionary Angle
IAF-network compatible: single static Go binary + data-diode-safe JSON/Protobuf IO; zero outbound sockets in air-gap mode.
Indigenous data: ingest ISRO/ISTRAC TLE feeds, NavIC almanac, INSAT weather, IMD GRIB2; ship with example datasets.
SDLC/certification: DO-333 / IEC 61508 mapping, SLSA-3 build attestation, in-toto provenance, Gobra formal proofs for Pc + SafeCholesky, CBMC model-checking for the ingest gate.
Scalable/maintainable: CI on every PR with scale-smoke test at 10k objects; ADRs for every architectural decision (docs/ADR-002-HYPERION.md pattern).
Perpetual licence: MIT core + commercial-safe list; ship licence registry script to enumerate every transitive dep.
4. Cross-Cutting Architectural & Tech-Stack Revolution
4.1 Unified Architecture
graph TB
  subgraph Ingest
    TLE[TLE/OMM Feeds]
    RAD[IAF Radars]
    OPT[Optical Sensors]
    WX[IMD Weather]
  end
  subgraph "Sovereign Physics Core (Go)"
    OCT[Octree + BVH]
    SGP[SGP4/SDP4]
    UKF[UKF + SNC]
    FOSTER[Foster Pc + Alfano + MC + Minkowski]
    JACCH[Jacchia-Roberts Atm]
  end
  subgraph "Axiom-Zero Physics AI (Python → ONNX)"
    SFMN[Yoshida-8 + rKAN]
    DIST[Symbolic Distillation]
    POL[POL Transformer]
    MAN[Manoeuvre Attributor]
    DUQ[Diffusion UQ]
  end
  subgraph "Ops Plane (Go)"
    RPC[gRPC+mTLS+Ed25519]
    RL[RL Sensor Tasker]
    AUD[Signed Audit Log]
    SIT[LLM SITREP - Phi-3 ONNX]
  end
  subgraph "Operator Plane"
    TUI[Rich TUI]
    WEB[Offline Web GUI - CesiumJS + Svelte]
    TRN[Training/Sim Mode]
  end
  Ingest-->OCT
  OCT-->SGP & UKF
  SGP & UKF-->FOSTER
  SFMN-.residuals.->UKF
  DIST-.laws.->MAN
  POL-.anomalies.->FOSTER
  FOSTER-->RPC
  JACCH-->FOSTER
  DUQ-.tails.->FOSTER
  RPC-->TUI & WEB & RL
  RL-->TRN
  AUD<-->RPC
  SIT-->TUI & WEB
4.2 Integration of Axiom-Zero into SSIS
Two paths, pick A + fallback B:

A. ONNX Export: Train rKAN / POL-Transformer / Delta-rKAN in JAX/Flax, export via jax2tf → ONNX. Load in ssis via existing pkg/ml/onnx session infrastructure. Deterministic, air-gap-safe, no Python at runtime.
B. CGO/FFI (stretch): For the pure symplectic integrator, compile JAX via jax2tf.convert(..., enable_xla=True) → XLA AOT → shared object; call from Go via cgo. Higher perf, more risk.
4.3 Tech-Stack Choices (indigenous-friendly, OSS-only)
Core: Go 1.22, gonum, go-satellite, onnxruntime-go, gRPC+Protobuf, Ed25519.
AI Research: JAX, Flax, PyTorch (DLPack bridge via axiom_zero).
Verification: Gobra (Go), CBMC (C), SymPy (symbolic round-trip).
GUI: Svelte + CesiumJS (offline tile cache) for 3D globe; Go Wails for desktop wrap. Avoid Electron.
Sim/Training: Godot 4 (C#) for immersive training mode, since it is MIT, lightweight, and offline-friendly. Bridge to ssis via gRPC.
Observability: OpenTelemetry (offline OTLP file exporter), Grafana+Prometheus (optional classified-net).
4.4 Repo Hygiene (G2–G4, G15)
Delete scratch/, test_grpc.go, test_sgp4.go, get_sig.go, cpu.prof, mem.prof from ssis/ root.
Move all patch_*.py into a single migration tool under ai/training/ if still needed, or drop entirely.
Enforce via .golangci.yml + pre-commit hook.
5. Performance & Scalability Blueprint
Path	Current	Target	Lever
Propagation 30k obj, Δt=60 s, 24 h	~seconds	<200 ms CPU / <30 ms GPU	Batched SGP4 via gonum SIMD + optional ONNX-GPU kernel
Conjunction screen 30k × 30k	O(n²) worst cell	O(n log n)	Replace all-pairs-in-cell (G7) with BVH + Morton-sorted sweep-and-prune
Pc per pair	~ms	<200 µs	Precompute Cholesky once per track; SIMD Foster integrand
UCT classification	N/A	<50 ms	POL transformer ONNX, INT8 quantized
Manoeuvre attribution	N/A	<5 s	Delta-rKAN cached per-object; symbolic distillation on trigger
Reentry footprint 72 h	seconds	<1 s	Vectorized RKF45, Halton QMC already good
Hypersonic update	N/A	1 Hz sustained	Dedicated pipeline, 6-DoF Kalman + drag
Concurrency plan: one goroutine-per-ingest-feed, work-stealing propagator pool (runtime.NumCPU × 2), lock-free ring buffer for tick updates, sync.Map for octree leaf modification with copy-on-write for iteration.

Memory: memory-map catalogue snapshots; zero-copy Protobuf via google.golang.org/protobuf proto.UnmarshalOptions{Merge: true}; object pools for propagation state.

Measurement: promote existing cpu.prof/mem.prof to a CI benchmark suite with regression thresholds; publish against ARM64 (Kavach) + AMD64.

6. Security, Certification & IAF Compliance Overhaul
6.1 Zero-Trust Hardening (closes G5, G6, G12)
mTLS everywhere for gRPC (client certs issued by air-gapped internal CA using Ed25519 + Kyber768 hybrid; post-quantum-ready).
Replace "ED25519_SIG_PLACEHOLDER" with actual signing path in pkg/server; every RPC response signed; every RPC request has replay-nonce.
RBAC: roles = viewer, analyst, operator, commander, auditor; enforced in interceptors.
Secure boot path: verify binary + model manifest against baked-in pubkey before serving.
STIG-alignment matrix: extend HARDENING_COMPLIANCE_MATRIX.md to DISA STIG + CIS for RHEL, indigenous BOSS Linux, and classified-network hardening guidelines.
6.2 Supply Chain
SLSA-3 compliant reproducible build via goreleaser + --mod=vendor + -trimpath + -buildvcs=false with external timestamping.
in-toto attestations per artefact; Cosign/sigstore-equivalent using Ed25519 (since sigstore assumes internet).
Full dependency licence register; auto-gate merges on licence-compatibility failure.
6.3 Certification & Formal Methods
Extend Gobra coverage beyond Foster Pc to: SafeCholesky bounds, Mahalanobis gating, ingest schema validators.
CBMC for C-FFI boundary (if cgo path is taken).
DO-178C DAL-C / IEC 61508 SIL-2 mapping document — show how each capability maps to independence, MC/DC, configuration management artefacts.
Publish docs/CERTIFICATION_PACK/ containing HAZOP lite, FMEA for Pc pipeline, V&V matrix.
7. Prioritized 90-Day Roadmap
Effort scale: XS=0.5 wk, S=1 wk, M=2 wk, L=4 wk. Impact: 🔥🔥🔥 must-have, 🔥🔥 strong differentiator, 🔥 nice-to-have.

Phase 1 — Immediate (Weeks 1–2): Ship-Readiness
#	Task	Effort	Impact	Owner area
1.1	Clean ssis root: delete scratch/, stray test_*.go, get_sig.go, cpu/mem.prof, patch_*.py (G2–G4, G15)	XS	🔥🔥	Repo hygiene
1.2	Wire real Ed25519 signing into gRPC server; remove placeholder (G5)	S	🔥🔥🔥	pkg/server
1.3	Add mTLS to gRPC, RBAC interceptors (G6)	S	🔥🔥🔥	pkg/server
1.4	Fix newai_v1 broken tests; strip DEBUG prints; remove hardcoded paths (G9)	S	🔥	newai_v1/
1.5	Unify into one repo narrative: add top-level README + ARCHITECTURE.md spanning all three folders	XS	🔥🔥🔥	repo root
1.6	Document fallback decision logic (ONNX→classical, Foster→Alfano) with truth table	XS	🔥🔥	docs/
1.7	Validate Axiom-Zero rKAN on real TLE (Starlink, COSMOS-2251 cloud) — produce RMSE vs SGP4 (G11)	M	🔥🔥🔥	newai_v3 bench
Phase 2 — Short-Term (Weeks 3–8): Revolutionary Features
#	Task	Effort	Impact
2.1	Integrate Axiom-Zero into ssis via jax2tf→ONNX — rKAN residual covariance head + Delta-rKAN manoeuvre attributor + POL transformer (G1, G13)	L	🔥🔥🔥
2.2	Replace octree all-pairs-in-cell with BVH + Morton sweep-prune for conjunctions (G7)	M	🔥🔥🔥
2.3	Symbolic Manoeuvre Attribution pipeline with LaTeX output — live demo script	M	🔥🔥🔥
2.4	Minkowski-Sum geometric Pc for non-spherical RSOs	M	🔥🔥
2.5	Offline Web GUI (Svelte + CesiumJS) with 3D globe, object inspector, Pc dashboard, SITREP panel (G8)	L	🔥🔥🔥
2.6	Multi-sensor Blind-Spot 4D tensor + IMD weather overlay	M	🔥🔥
2.7	Strategic-Asset Re-entry Risk Map with MoD installations ingestion	S	🔥🔥
2.8	Hypersonic sub-orbital lane (6-DoF Kalman, 1 Hz)	M	🔥🔥🔥
2.9	Training/Simulation mode: scripted scenarios (ASAT, rendezvous, debris cascade, hypersonic intercept) via Godot 4 frontend	L	🔥🔥🔥
2.10	SLSA-3 reproducible build + in-toto attestations; certification pack skeleton	M	🔥🔥
2.11	Gobra coverage extension beyond Foster	S	🔥🔥
Phase 3 — Long-Term / Post-Award (Months 3–12)
#	Task	Impact
3.1	Diffusion-UQ for tail-risk Pc (from newai_v3 roadmap Frontier)	🔥🔥
3.2	Swarm-sensor fusion: optical + radar + RF (for future IAF SBSS constellation)	🔥🔥🔥
3.3	Edge deployment to airborne / mobile ground stations (ARM64 static binary)	🔥🔥
3.4	Post-quantum crypto migration (ML-KEM + ML-DSA) for audit log	🔥🔥
3.5	Digital-twin fusion with ISRO/ISTRAC live data if security-cleared	🔥🔥🔥
3.6	Federated learning across allied SSA nodes (QUAD partners) with differential privacy	🔥🔥
3.7	VQE-based resonance search on hardware QPU for deep-perturbation discovery (research)	🔥
Effort-vs-Impact Matrix (Phase 1+2)
        HIGH IMPACT
        ^
   🔥🔥🔥 | 1.5,1.6   1.2,1.3     2.1,2.5,2.9
        |              1.7        2.2,2.3,2.8
   🔥🔥  |                        2.4,2.6
        |                         2.7,2.10,2.11
   🔥   | 1.1,1.4
        +------------------------------->
         XS        S         M         L  EFFORT
8. Bonus "Moonshot" Ideas
"Dhruva" Live SITREP — local Phi-3 LLM consumes signed catalogue diff + Pc table every 60 s and issues Hindi/English voice SITREP. Doctrine-aligned, air-gap-safe. engine/pkg/assistant already hosts the LLM — expand to voice-out (Indic TTS via IndicTrans2 / AI4Bharat models).
Symbolic Law Gallery — live demo in which Axiom-Zero rediscovers J₂ and Kozai-Lidov resonance on stage from TLEs, printing the LaTeX equation on screen. Devastating to foreign-competitor narrative.
"Reality Probe" Certification — evaluators bring their own TLE and get signed 72-h prediction manifest; re-verify 72 h later against ground truth. Signed chain-of-custody.
Counter-ASAT Gaming Mode — Godot trainer where IAF analysts play against AI-driven adversary running rendezvous/shadowing tactics; scoring via Neighbourhood-Watch RRI.
"Blind-Spot Stitcher" — opportunistic amateur-radar / optical data ingest (via OK-to-cite sources) with cryptographic provenance, to close coverage gaps during conflict.
Indigenous "Sovereign Signature" — every artefact signed with a chained attestation: Gobra proof ID → build provenance → key ID → ISRO-verified astronaut-time-stamp. No foreign tool has this.
Digital Twin of an Indian Asset under Attack — demo a scripted PLA-ASAT scenario targeting RISAT-2B; show Dhruva issuing an evasive-manoeuvre recommendation with symbolic delta-v justification.
Quantum-Resistant Audit Log — Merkle log with ML-DSA signatures; survives harvest-now-decrypt-later.
Edge Node on a Rafale / AWACS — compile ssa_ops_runner for a ruggedised ARM64 board; cached catalogue + 1 Hz hypersonic lane onboard.
Open Challenge — publish the Gobra proofs, benchmarks (vs SGP4), and reproducible build recipe in a sealed repo to evaluators. Transparency beats marketing.
9. Critical Files to Modify (for Phase 1–2 execution)
ssis/pkg/server/ — mTLS, RBAC, real Ed25519 signing
ssis/pkg/collision/ — BVH sweep-prune, Minkowski, Diffusion-UQ head
ssis/pkg/orbital/ — hypersonic lane, 4D blind-spot tensor, manoeuvre attributor hook
ssis/pkg/catalog/ — POL Transformer ONNX load path, UCT classifier
ssis/pkg/ml/onnx/ — session pool for new Axiom-Zero models
ssis/ai/training/ — export jax2tf→ONNX for rKAN / POL / Delta-rKAN
newai_v3/axiom_zero/sda/ — implement real bodies vs. current protocol stubs (G10); then fold the useful kernels into ssis via ONNX (the SDA layer itself stays in Go)
New: ssis/pkg/hypersonic/, ssis/pkg/collision/minkowski.go, ssis/pkg/collision/bvh.go, ssis/pkg/catalog/pol_transformer.go, ssis/cmd/ssa_gui/ (Wails + Svelte)
10. Verification Plan
Unit: go test ./... including new BVH, Minkowski, POL. Target ≥80 % coverage on new code.
Scale: --scale-test --catalog-size 100000 --scale-hours 168 --hypersonic-rate 1hz must stay <30 min CPU.
Accuracy: newai_v3 bench extended with real-TLE regression — Starlink-30000 + COSMOS-2251 debris cloud — RMSE vs SGP4 published.
Security: govulncheck ./..., gosec, gobra verify, CBMC on ingest, SLSA-3 provenance attest-verify, mTLS negotiation test.
Operator demo: one-command end-to-end — catalogue ingest → volumetric → neighbourhood → conjunction → Pc → manoeuvre symbolic → re-entry risk → signed SITREP → GUI render.
Certification: publish HARDENING_COMPLIANCE_MATRIX.md update with DO-333 / IEC 61508 / STIG trace.
11. Locked Decisions (confirmed by user)
Integration path: jax2tf → ONNX. All Axiom-Zero heads (rKAN residual covariance, Delta-rKAN manoeuvre attributor, POL Transformer) exported from JAX/Flax, loaded by existing ssis/pkg/ml/onnx session pool. No Python at runtime.
UI stack: Both TUI + Web GUI. TUI (bubbletea) for field/air-gap ops; Svelte + CesiumJS + Wails desktop GUI for evaluator demo. Shared gRPC client.
Timeline: 90 days — full Phase 1 + Phase 2. Certification pack, symbolic manoeuvre attribution, hypersonic lane, training/sim mode all in scope.
newai_v1: Retire post-integration. Fold useful kernels into Axiom-Zero v3; delete v1 folder before submission so evaluators see one coherent repo.
Training-mode frontend: Godot 4 (default, unless user overrides later).