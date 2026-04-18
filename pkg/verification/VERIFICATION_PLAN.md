# Software Verification Plan: FP Robustness (SafeCholesky + Pc Pipeline)

This plan defines the formal verification strategy for floating-point safety closure in the covariance regularization, Mahalanobis, MHT, UKF, and collision-probability paths.

## 1. Verification Objectives

| DO-333 Objective | Target | Method | Evidence |
|---|---|---|---|
| FM.1 | Formalized low-level FP requirements | Contracted functions in Gobra spec harness | `pkg/verification/gobra_specs.gobra` |
| FM.2 | Formalized design behavior | SafeCholesky + Mahalanobis contracts/invariants | `pkg/math/safe_cholesky.go` |
| FM.3 | Design analysis | Gobra theorem checks over contracts | `scripts/verify_gobra_fp.sh` |
| FM.4 | Code-level formal analysis | Gobra with IEEE mode + divergence checks | `docs/validation_reports/gobra_fp_verification.log` |
| FM.5 | Satisfaction of FP safety requirements | No-NaN proofs for Cholesky/Mahalanobis/Pc linkage | `pkg/verification/gobra_specs.gobra`, `pkg/verification/gobra_extracted_fp_spec.v`, `pkg/verification/safe_cholesky_pipeline.v` |
| FM.6 | No unintended functionality in verified envelope | Explicit bounded domains + finite guards | `pkg/math/safe_cholesky.go`, `pkg/math/safe_cholesky_test.go` |

## 2. Scope

- Covariance regularization with `epsilon >= 1e-12`.
- Cholesky factorization under near-singular and degenerate covariance inputs.
- Mahalanobis distance for MHT and UKF gating.
- Monte-Carlo covariance factorization in collision analysis.
- Foster Pc boundedness and finite propagation assumptions.

## 3. Toolchain

1. Gobra (IEEE floating-point mode): contract and runtime-error verification.
2. Coq: higher-order lemma stitching SafeCholesky and Pc finite guarantees.
3. Go test/benchmark: runtime regression and performance envelope checks.

## 4. Verification Command

```bash
scripts/verify_gobra_fp.sh
```

Script flags:

- `--divergence`
- `--float-mode=ieee`

## 5. Test & Performance Acceptance

- `SafeCholesky` must return finite lower-triangular factors or deterministic error.
- Mahalanobis distance must be finite for degenerate covariance test vectors.
- No NaN/Inf propagation across MHT/UKF/Monte-Carlo call paths.
- No material performance regression in UKF and MHT workloads.

## 6. Traceability

The detailed requirement-to-artifact linkage is maintained in:

- `pkg/verification/TRACEABILITY_MATRIX.md`
