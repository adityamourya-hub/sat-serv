# SSIS Hardening Compliance Matrix (PS24 Delivery)

| Control Area | Requirement | Status | Evidence |
|---|---|---|---|
| Binary Integrity | Deterministic build pipeline | Implemented | Makefile build-static, .goreleaser.yml |
| Supply Chain | Dependency inventory (SBOM) | Implemented | scripts/generate_sbom.sh, docs/SBOM_go_modules.json |
| Runtime Isolation | No external network dependency at runtime | Implemented | Single local binary with local models/assets |
| Cryptographic Non-repudiation | Signed operational alerts | Implemented | pkg/crypto/sign.go, signed_alert_example.json |
| Fail-safe AI behavior | Deterministic fallback with provenance | Implemented | pkg/ml/hybrid/orchestrator.go, fallback ratio metrics |
| Explainability | Structured reason payload attached to alerts | Implemented | pkg/ml/evaluation/operational_runner.go |
| Concurrency Safety | Shared mutable state synchronized | Implemented | pkg/orbital/hybrid_transformer.go |
| Data Ingestion Robustness | CSV parse validation and explicit errors | Implemented | pkg/ml/data/io.go |
| Performance Telemetry | Accurate latency percentiles and summary | Implemented | pkg/ml/onnx/metrics.go |
| Catalog Scalability | Incremental update path for spatial index | Implemented | pkg/catalog/catalog.go, pkg/spatial/hashgrid.go |

## Residual Risks
- Full ONNX tensor binding remains model-dependent; enforce RequireONNX for adjudication runs.
- Legacy test files in orbital package require signature updates for changed APIs.
- LLM tactical assistant currently uses deterministic template mode unless local ONNX runtime integration is completed.

## Acceptance Gate (Recommended)
- Build: make build-static
- Verification: make test-focused
- Packaging: make package
- Artifact integrity: verify SHA256SUMS in delivery root
