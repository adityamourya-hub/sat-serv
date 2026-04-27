# SSIS ADITI 4.0 Submission Readiness Plan

## Current State Assessment

### ✅ What's Working
- Go engine builds, starts, loads TLE catalogue, runs gRPC server
- gRPC proto covers all 5 RPCs: IngestObservation, SubscribeAlerts, QueryCatalog, GetObjectState, OperatorSession
- 3/6 ONNX models verified (collision, navic, maneuver-load-only)
- 51 Go packages: catalog, orbital, collision, spatial, proximity, physics, ML, cryptography, server, hypersonic, etc.
- Dockerfile, docker-compose (TimescaleDB + Plasma), Makefile all exist
- mTLS scaffolding in grpc.go, Ed25519 signing in cryptography/
- REST API handlers for sovereign volume monitoring (api/handlers/)
- Space-Track TLE ingestion (pkg/ingestion/spacetrack/)
- CARA parity cases (docs/validation_reports/)

### ❌ What's Broken / Missing
1. **Go engine loads MockInferrer** — needs `-tags onnx` + libonnxruntime (Docker-only)
2. **No end-to-end demo script** exercising all 9 capabilities
3. **No validation evidence PDF** for judges
4. **No 3D visualization** (demo impact)
5. **No executive summary document**
6. **verification_suite.py** has 1-space indentation throughout (685 lines)
7. **Maneuver ONNX** has fixed-batch reshape bug (attention layer)
8. **3 models still stub**: Axiom-HNODE, ATeena-IQNet, Rapid Ranker

## Execution Plan (Ordered by Submission Impact)

### Phase 1: demo.sh + Smoke Test (Highest Impact)
1. Create `scripts/demo.sh` — starts engine, loads TLE, runs synthetic scenario
2. Create `scripts/smoke_test.go` — exercises all gRPC RPCs programmatically
3. Wire real models in ONNXPool paths to match `pkg/ml/models/`

### Phase 2: Validation Evidence Package (PDF)
4. Create `docs/VALIDATION_EVIDENCE.md` with all quality gates
5. Convert to PDF via pandoc or direct generation

### Phase 3: Executive Summary (1-2 pages)
6. Create `docs/EXECUTIVE_SUMMARY.md` — physics-first, sovereign, indigenous

### Phase 4: 3D Globe Visualization (Demo Impact)
7. Static HTML + Cesium.js showing satellite positions, conjunction corridors, alerts

### Phase 5: Air-Gapped Package
8. Verify Docker build, create tarball packaging script

### Phase 6: Polish & Cleanup
9. Fix maneuver ONNX batch export
10. Document remaining MockInferrer models with clear justification
