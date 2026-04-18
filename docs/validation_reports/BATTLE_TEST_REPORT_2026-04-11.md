# SSIS Battle Test Report (Real-Data Execution)

Date: 2026-04-11
Workspace: ssis

## Executive Verdict

Overall status: CONDITIONAL FAIL

Operational throughput goals were met in non-strict mode, but strict fail-fast execution did not pass, signature validation in the generated report is false, covariance realism and Pc calibration targets failed, and persistence historical replay failed.

## Evidence Artifacts

- Real merged TLE feed: artifacts/live_ops/celestrak_merged.tle
- Real merged catalog JSON: artifacts/live_ops/celestrak_merged_catalog.json
- Scale validation report (signed payload emitted): artifacts/live_ops/live_validation_report_signed.json
- Precompiled orbital test binary used for stress/compliance: orbital.test

## Scenario Matrix

| Scenario | Result | Evidence |
|---|---|---|
| 1M-object 24h scale stress (orbital suite) | PASS (synthetic fallback) | `TestScaleStressFullCatalog` passed in 24.49s, peak memory ~1.40 GB |
| ADITI compliance suite | PASS | `TestADITIComplianceSuite` all subtests passed |
| ADITI hardening suite | PASS | `TestADITIHardeningSuite` all subtests passed |
| IAF certification suite | PASS | `TestIAFCertificationSuite` passed |
| Propagation 24h stability check | SKIP | `TestPropagate24hStability` skipped due TLE checksum validation error in embedded test data |
| TLE parse regression | SKIP | `TestParseTLE` skipped due checksum mismatch in embedded sample |
| Volumetric 1M benchmark threshold | FAIL | `BenchmarkVolumetricMillion` reported 251.96ms (>100ms threshold) |
| MOID 1M benchmark threshold | FAIL | `BenchmarkMOIDFullCatalog` reported 5.817s (>5s threshold) |
| Strict operational scale run with live catalog | FAIL | Panic: strict mode fallback blocked (`ml_error` -> `use_classical_cusum`) |
| Non-strict operational scale run with live catalog | PASS with major caveats | Completed and emitted report; multiple target flags false |
| Persistence restart CRUD | PASS | `TestPersistentStore_CRUD` passed |
| Persistence historical replay correctness | FAIL | `TestPersistentStore_HistoricalQuery`: expected V1, got V2 |

## Live Scale KPI Snapshot

Source: artifacts/live_ops/live_validation_report_signed.json

- catalog_size: 50000
- real_catalog_object_count: 813
- observations_per_day: 14478048 (target >= 1,000,000) -> PASS
- maneuver_detection.f1: 1.000 (target > 0.95) -> PASS
- maneuver_detection.p95_latency_seconds: 0.00004485 (target < 300) -> PASS
- conjunction_performance.elapsed_seconds: 8.316 for 10k (target < 60) -> PASS
- conjunction_performance.objects_per_second: 1202.50 (target >= 166.67) -> PASS
- covariance_realism.innovation_consistency: false -> FAIL
- covariance_realism.nees_consistency: false -> FAIL
- pc_calibration.calibration_pass: false -> FAIL
- ukf_acceptance_ratio: 0 -> FAIL (operationally non-acceptable)
- signature_valid: false -> FAIL

## Fallback Diagnostics (Non-Strict Live Run)

- Fallback events observed: 288
- Reason distribution: 288x `deviation_guardrail`
- Highest observed `deviation_ratio`: 1025.087

This indicates the ML path repeatedly diverged from classical guardrails and was forced into fallback mode.

## Blocking Findings

1. Strict mode is not currently deployable for this workload.
	- Fails immediately on first ML fallback event (`ml_error` in maneuver detection).

2. Signature verification is false in emitted validation report.
	- The run emits a signature payload, but `signature_valid` is false.

3. Covariance realism and Pc calibration targets fail in live-scale report.
	- `innovation_consistency=false`, `nees_consistency=false`, `calibration_pass=false`.

4. Historical persistence replay is incorrect.
	- Historical query returned latest value instead of prior state (`V2` vs expected `V1`).

5. Benchmark thresholds are missed for volumetric and MOID stress tests.

## Temporary Build Unblock Changes Applied During Testing

To run blocked tests, minimal compile-hygiene fixes were applied (non-feature changes):

- pkg/catalog/assignment.go
- pkg/catalog/catalog.go
- pkg/catalog/mht_full.go
- pkg/persistence/sqlite.go

These changes removed stale/unused symbols and schema-field mismatches to allow test execution.

