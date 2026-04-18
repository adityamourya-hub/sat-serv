# SSIS USSF-Level Improvement Report

- Date: 2026-04-11
- Scope: What SSIS still needs to reach an elite USSF/NASA-grade SSA capability set
- Current status: not yet at target

## Executive Summary

SSIS already has the right high-level architecture for an indigenous SSA/SSA-ops stack: cataloguing, conjunction analysis, maneuver detection, signing, replay, and air-gapped deployment support. The current gap is not the presence of features, but the fidelity, calibration, and operational robustness required for a USSF-level result.

The current validation evidence shows three broad problems: physics accuracy is still oversimplified, strict fail-fast operation is not stable, and audit/replay guarantees are not yet reliable enough for mission-grade use. To reach an ideal US-level bar, the program needs to convert the current prototype into a deterministic, calibrated, benchmarked operational system.

## What Must Be Fixed First

1. Replace simplified propagation with a real TLE propagation stack.
   - Implement SGP4/SDP4 for TLE ingestion and propagation to TEME/ECI.
   - Add proper secular and periodic corrections.
   - Apply Earth rotation, drag, and third-body perturbations where applicable.

2. Replace approximate conjunction probability logic with an operational 2D Pc method.
   - Use a Foster- or Alfano-class 2D integral on the conjunction plane.
   - Calibrate hard-body region handling and covariance projection.
   - Validate against known benchmarks and accepted reference cases.

3. Make strict mode truly deployable.
   - Eliminate fallback-trigger panics in strict runs.
   - Ensure ML failures degrade cleanly, or are blocked earlier with deterministic prechecks.
   - Make zero-fallback operation a testable release gate.

4. Repair persistence and replay semantics.
   - Historical queries must return the requested version, not the latest state.
   - Replay must be deterministic and auditable across restarts.
   - Signed artifacts must remain verifiable end to end.

5. Restore covariance realism.
   - Fix innovation consistency and NEES consistency failures.
   - Recalibrate process noise and acceptance thresholds.
   - Recheck Pc outputs after covariance updates.

## Workstreams Needed To Reach Target

### 1. Physics Fidelity

This is the largest gap. The system needs a production-grade orbital and atmospheric model, not a light approximation.

- Implement SGP4/SDP4-based propagation.
- Add drag acceleration using a validated atmosphere model.
- Add Earth rotation to all ground-to-space transforms.
- Add Sun/Moon third-body perturbations and higher-order geopotential terms where needed.
- Replace current re-entry approximations with a real atmospheric density and footprint model.

### 2. Conjunction and Coverage Analysis

The analysis stack must be upgraded so outputs can stand beside mature US tooling.

- Move volumetric scan from static snapshot logic to epoch-marched propagation.
- Use a true 2D Pc formulation.
- Ensure conjunction searches use calibrated covariance handling.
- Add full time-windowed blind-spot analysis over adversary constellations.

### 3. Maneuver and Cataloguing Intelligence

The current maneuver pipeline is directionally correct, but it needs calibration and stronger hypothesis management.

- Improve maneuver detection thresholds and calibration.
- Add fuller multi-hypothesis tracking and assignment logic.
- Validate object characterisation against a richer labeled set.
- Separate detection confidence from operational decision thresholds.

### 4. Performance and Scale

US-level ops require consistent performance at operational catalog sizes.

- Bring volumetric and MOID benchmarks below target thresholds.
- Use batched propagation or hardware acceleration where necessary.
- Ensure the 50k-object use case is a normal case, not a special test path.
- Keep memory growth and latency predictable under sustained load.

### 5. Verification, Signing, and Replay

This is a mission-trust issue, not just a software issue.

- Make signed validation reports consistently verify with a populated public key.
- Preserve deterministic replay across runs and restarts.
- Keep historical persistence queries version-correct.
- Emit machine-checkable evidence for every operational run.

### 6. Certification and Operational Readiness

To be credible in a classified or air-gapped environment, SSIS needs release discipline.

- Add static analysis and security scanning as standard gates.
- Maintain SBOM and artifact provenance.
- Keep the build reproducible and the runtime offline-safe.
- Document deployment, rollback, and incident response procedures.

## Prioritized Roadmap

### P0: Release Blockers

- Implement SGP4/SDP4 propagation.
- Fix drag and Earth rotation in the propagation and sensor models.
- Replace approximate Pc with a real 2D method.
- Eliminate strict-mode fallback failure.
- Fix historical persistence replay.
- Make signed reports validate successfully.

### P1: Operational Credibility

- Add time-marched volumetric scanning.
- Calibrate covariance realism and maneuver confidence.
- Add third-body perturbations and better re-entry physics.
- Stabilize benchmark performance.

### P2: Elite Capability

- Add multi-hypothesis tracking with proper pruning.
- Add GPU or batch propagation for full-catalog scale.
- Expand ML training data and explainability.
- Tighten certification and reproducibility tooling.

## Definition of Done

SSIS reaches the intended target only when all of the following are true:

- Strict mode completes without fallback failures for the operational scenario.
- Signed reports verify end to end.
- Historical replay returns the requested historical state.
- Covariance realism and Pc calibration pass validated thresholds.
- Volumetric and MOID benchmarks meet target performance.
- The physics layer is benchmarked against accepted reference behavior.

## Bottom Line

The project is close in architecture, but not yet at the level of a mature USSF-grade SSA system. The main work is to replace approximations with validated physics, make the strict path robust, and turn the audit trail into a reliable release gate. Once those are done, the system can credibly claim elite-level operational readiness.