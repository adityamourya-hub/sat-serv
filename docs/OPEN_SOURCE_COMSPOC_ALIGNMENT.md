# BOS-4.0 Open Source Alignment Plan (COMSPOC-like Capability)

## Objective
Build a COMSPOC-like, air-gapped SSA stack in BOS-4.0 by selectively integrating proven open-source components while preserving IAF policy controls, deterministic fallback, and signed auditability.

## Recommended Backbone
1. IBM/spacetech-ssa for operational pipeline patterns (ingest -> propagate -> conjunction -> analytics).
2. Orbital Space Ontology Project for domain entities and relationships.
3. NASA CARA risk math and covariance handling for validated collision probability workflows.
4. AstroLibrary-style high-scale screening concepts for fast broadphase conjunction candidate generation.
5. ssa-gym for Orbit Zoo policy training and simulation-based evaluation.
6. Cesium ecosystem for operator-facing visualization.

## Mapping to Current BOS Packages

| External Repo | BOS Target Area | Integration Contract | Fallback Requirement |
|---|---|---|---|
| IBM/spacetech-ssa | pkg/ingest, pkg/orbital, pkg/propagator, cmd/bos4 | Reuse pipeline sequencing and model-assisted correction flow patterns. Keep BOS source adapters and policy gate as authority. | Deterministic Go path remains primary if model runtime unavailable. |
| Orbital-Space-Ontology-Project | pkg/ontology | Align BOS event/entity names with ontology classes: SpaceObject, Event, Conjunction, Maneuver, ReEntry, CoverageGap. | Existing ontology graph remains valid with optional class aliases. |
| ssa-gym | pkg/rl, ai/rapid, ai/rl | Export BOS simulation traces to Gym-like training schema and import policy artifacts through trained model loader. | Existing heuristic ranking + rules path remains active. |
| orbdetpy | pkg/orbital, pkg/fusion, pkg/propagator | Add cross-validation harness against BOS UKF/propagation outputs for OD and covariance transport. | BOS UKF path remains source of truth in runtime. |
| AstroLibrary | pkg/propagator, pkg/spatial, pkg/octree | Adopt algorithmic pattern for high-cardinality broadphase screening and partitioning. | Legacy local screening path remains callable via flag. |
| NASA CARA tools | pkg/collision, pkg/orbital | Validate Pc, covariance frame transformations, and conjunction metrics against CARA outputs. | Existing adaptive Foster + MC remains active if CARA parity data absent. |
| Cesium repos | pkg/server, docs/diagrams, frontend portal | Publish CZML/JSON endpoints and operational layers for coverage gap, RPO, conjunction, and re-entry overlays. | Text + JSON operator outputs remain supported in strict mode. |

## Air-Gap Compliance Pattern

1. Vendor snapshots into internal mirrors (no live runtime network dependencies).
2. Pin versions and hashes in local manifest.
3. Convert all runtime assets to local file paths.
4. Sign artifacts and reports using existing signer flow.
5. Keep strict-mode fallback behavior for every optional model or sidecar path.

## Integration Sequence (Execution Order)

### Phase A: Ontology and Data Contracts
- Normalize BOS ontology labels and relationship edges to the Orbital Space Ontology structure.
- Add schema validation tests for object/event payloads.

### Phase B: Risk and Covariance Validation
- Add CARA parity test suite for Pc and covariance transforms.
- Store parity fixtures in local artifacts.

### Phase C: High-Scale Screening
- Introduce sampled broadphase benchmark profile at 100k, 500k, and 1M synthetic objects.
- Integrate partitioning improvements inspired by AstroLibrary patterns.

### Phase D: Orbit Zoo Reinforcement Learning
- Build BOS-to-Gym scenario export and policy import pipeline.
- Evaluate policy acceptance under existing guardrails and confidence thresholds.

### Phase E: Visualization Layer
- Add Cesium-compatible output endpoint for propagated tracks, alerts, and event overlays.
- Keep rendering optional and decoupled from core runtime.

## Acceptance Gates

1. Air-gap: zero runtime dependency on external URLs.
2. Determinism: benchmark variance within agreed threshold under fixed seed.
3. Safety: strict mode never silently degrades without fallback audit record.
4. Security: all inference and report outputs signed and verifiable.
5. Parity: CARA comparison tolerances met for Pc and covariance realism checks.

## Immediate Next Sprint Deliverables

1. Ontology class alignment matrix and schema validator tests.
2. CARA parity harness for 10 representative conjunction cases.
3. Broadphase screening benchmark at 100k+ with runtime and memory budget report.
4. Orbit Zoo scenario export interface spec and sample policy ingestion test.
