

 command-and-control substrate designed for the Indian Air Force. It provides sub-second volumetric surveillance, hypersonic threat detection, and symplectic orbital prediction at planetary scale.

# Capabilities

| Capability | Description | Status |
| :--- | :--- | :--- |
| **(a) Data Fusion** | Real-time multi-sensor fusion and volumetric screening. | 
| **(b) Proximity Alerts** | Neighbourhood Watch (Intelligent intent & proximity alerts). | 
| **(d) Hypersonic Tracking** | High-fidelity 8-state UKF for Glide Vehicle (HGV) tracking. | 
| **(g) Maneuver Detection** | PINN-accelerated learned residual analysis for delta-V discovery. | 
| **(i) Rapid Processing** | Sub-second propagation of 30k+ objects via Symplectic Kernels. | 

# Security Baseline

- **Zero-Trust gRPC**: Mandatory Ed25519 signing for all telemetry responses via logic-level interceptors.
- **mTLS 1.3 Hardening**: Hardware-anchored transport security with mandatory client-cert verification.
- **Indigenous Physics Core**: Axiom-Zero symplectic substrate (Yoshida-4/8) ensuring zero secular energy drift.
- **Air-Gapped Ready**: Zero external dependencies; designed for on-prem Kubernetes / air-gapped deployment.

# Architecture

SSIS implements a **Differentiable Physics-Managed Awareness** plane, unifying high-performance Go-based catalog management with the **Axiom-Zero v3** JAX-native physics core.

### Design

**Layer 1: Go Operations Plane** (High-concurrency orchestration)
- `pkg/catalog`: High-churn optimized catalog management
- `pkg/orbital`: Unified Propagator interface (SGP4 + Axiom-Zero)
- `pkg/spatial`: BVH + Morton sweep-prune for O(n log n) conjunction screening
- `pkg/crypto`: Ed25519 response signing and audit trails
- `pkg/server`: mTLS 1.3 hardened gRPC/REST endpoints

**Layer 2: Axiom-Zero Physics Substrate** (Symplectic neural flows)
- `ai/v3/axiom_zero`: Hamiltonian-native rKAN physics kernel
- `ai/v3/sfmn_kernel.py`: State-space fusion neural network
- `ai/v3/maneuver_distiller.py`: Delta-V attribution via symbolic distillation
- `ai/v3/hamiltonian_discovery.py`: Learned Hamiltonian dynamics
- `ai/v3/physics_geometry_layer.py`: Equinoctial element differentiable geometry

###  Bridge

1. **Modeling (Python/JAX)**: High-order Hamiltonian dynamics learned using rational-KAN (rKAN) models
2. **Synthesis**: Trained weights for Force Potential (∇V) exported to JSON artifacts
3. **Operation (Go)**: Go-native rKAN kernel implements Legendre basis and softplus projections
4. **Integration**: Trajectories evolved using 4th-Order Symplectic Yoshida Integrator in pure Go, ensuring 10⁻¹⁵ energy conservation



### Prerequisites
- Go 1.21+
- SSIS_SIGNING_KEY (64-byte Ed25519 Private Key)

### Deployment
```bash
# Set sovereign signing identity
export SSIS_SIGNING_KEY=$(cat /etc/ssis/signer.key)

# Run the engine
go run ./engine/main.go
```

---
**Confidentiality Notice**: This software contains Sovereign Defense IP. Unauthorized distribution is prohibited.

## Repository Structure
- `engine/`: Primary operational entrypoint (Go binary)
- `pkg/`: Core operational packages (orbital, ml, crypto, catalog, physics, spatial, server)
- `ai/v3/`: **Axiom-Zero v3** — Hamiltonian-native physics substrate (rKAN, SFMN, Maneuver Distiller)
- `ai/training/`: Air-gapped training and export pipeline
- `internal/legacy_engine/`: Reference implementation (deprecated, kept for validation)
- `docs/`: Technical documentation, hardening logs, and validation reports
- `scripts/`: Automation for build, SBOM generation, and physics validation
- `artifacts/`: Production model bundles and build artifacts

## Quick Start
### Build
```bash
make build-static
```

### Scale Validation
```bash
./engine/main.go --scale-test --catalog-size 100000 --scale-hours 168
```

### Physics Benchmark (Axiom-Zero rKAN)
```bash
python3 scripts/rkan_validation.py
```

## Verification
```bash
go test ./...
pytest ai/v3/test/
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system flow.
