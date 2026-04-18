# Traceability Matrix: Floating-Point Robustness Closure

This matrix maps low-level requirements to implementation and machine-checkable evidence for DO-178C Level B and DO-333 FM.1-FM.6 compliance in the Pc/Mahalanobis pipeline.

| LLR ID | Requirement Description | Implementation | Verification Evidence |
|---|---|---|---|
| LLR-FP-01 | Regularize covariance with `C_reg = C + epsilon*I` and `epsilon >= 1e-12` | `pkg/math/safe_cholesky.go` (`SafeCholesky`) | `pkg/verification/gobra_specs.gobra` (`SpecRegularizationPositiveDefinite`) |
| LLR-FP-02 | Cholesky output is lower-triangular with strictly positive diagonal | `pkg/math/safe_cholesky.go` (`factorLowerFinite`) | `pkg/math/safe_cholesky.go` Gobra contracts + loop invariants |
| LLR-FP-03 | No NaN/Inf propagation in Cholesky under near-singular covariance (`cond(C)>1e16`) | `pkg/math/safe_cholesky.go` (`SafeCholesky`) | `pkg/verification/gobra_specs.gobra` (`SpecSafeCholesky`) |
| LLR-FP-04 | No NaN/Inf propagation in Mahalanobis distance | `pkg/math/safe_cholesky.go` (`MahalanobisSquared`, `MahalanobisDistance`) | `pkg/verification/gobra_specs.gobra` (`SpecMahalanobisNoNaN`) |
| LLR-FP-05 | MHT association path uses verified Mahalanobis solver | `pkg/orbital/catalog_mht.go` (`stateMahalanobis`) | Unit behavior through `pkg/math/safe_cholesky_test.go` |
| LLR-FP-06 | UKF innovation gating uses verified Mahalanobis solver | `pkg/orbital/ukf.go` (`ukfUpdateWithGating`) | Unit behavior through `pkg/math/safe_cholesky_test.go` |
| LLR-FP-07 | Monte-Carlo Pc path uses verified SafeCholesky factorization | `pkg/collision/monte_carlo.go` (`factorCov`) | Collision package regression tests |
| LLR-FP-08 | Pc output remains bounded and finite on adaptive Foster paths | `pkg/collision/foster.go` | `pkg/verification/gobra_specs.gobra` (`SpecPcPipelineNoNaN`) |
| LLR-FP-09 | Formal cross-tool lemma for global no-NaN closure across SafeCholesky -> Mahalanobis -> Pc | `pkg/verification/gobra_extracted_fp_spec.v`, `pkg/verification/safe_cholesky_pipeline.v` | Coq theorem `Global_NoNaN_Pc_Mahalanobis_MHT` |
| LLR-FP-10 | Deterministic formal verification invocation with IEEE-754 mode and divergence checks | `scripts/verify_gobra_fp.sh` | `docs/validation_reports/gobra_fp_verification.log` |

## Certification Notes

- DO-178C A-6 objectives addressed: deterministic behavior under numerical edge cases, bounded output domains, and absence of runtime numeric faults in verified paths.
- DO-333 FM.5 addressed by explicit IEEE-754 Gobra run mode and finite-value contracts.
- MIL-STD-498 traceability objective addressed by direct LLR to artifact linkage.
