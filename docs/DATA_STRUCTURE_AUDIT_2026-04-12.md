# SSIS Full Data Structure Audit and Upgrade

Date: 2026-04-12
Scope: `ssis` repository (`pkg`, `engine`, `internal`, runtime-facing docs)

## 1) Executive Summary

This audit reviewed runtime-critical data structures across spatial indexing, temporal feature storage, catalog tracking, graph/ranking flows, and server fanout channels.

Upgrades were implemented in production code for two high-frequency hot paths:

1. Spatial hash grid update/delete path upgraded to O(1) slot-based removals.
2. DSP feature-store window queries upgraded from O(n) scans to O(log n + k) binary-sliced windows.

Both upgrades preserve existing APIs and behavior while reducing allocator pressure and worst-case latency under high object counts.

## 2) Full Inventory (By Runtime Domain)

### Spatial / Volumetric
- `pkg/spatial/hashgrid.go`
- `pkg/octree/octree.go`
- `pkg/orbital/volumetric.go`

Primary structures:
- Hash map keyed by voxel cell to object lists (`map[CellKey][]int`)
- Octree for bounded 3D partitioning
- Radius/AABB query pipelines

### Catalog / Association / Tracking
- `pkg/catalog/catalog.go`
- `pkg/catalog/mht_full.go`
- `pkg/orbital/neighbourhood.go`

Primary structures:
- Record map by object ID
- Incremental grid + octree indexing
- Hypothesis state maps and history slices

### Temporal Signal + Fusion
- `pkg/dsp/innovation_feature_store.go`
- `pkg/dsp/measurement_filter.go`
- `pkg/orbital/fusion.go`

Primary structures:
- Bounded innovation record sequence
- Sliding windows for UKF innovation and maneuver signals
- Structured per-measurement telemetry payloads

### Threat Ranking / Rapid Paths
- `pkg/physics/rapid.go`
- `pkg/ml/hybrid/orchestrator.go`
- `pkg/rl/sensor_steering.go`

Primary structures:
- Slices for ranking and filtering
- Confidence-gated blending of classical + ML decisions

### Concurrency / Fanout
- `pkg/server/measurements_hub.go`
- `pkg/ingest/stream.go`

Primary structures:
- Subscriber set map with buffered per-subscriber channels
- Non-blocking fanout with lagging subscriber drop behavior

## 3) Applied Upgrades

### Upgrade A: HashGrid O(1) Slot Indexing

Files:
- `pkg/spatial/hashgrid.go`
- `pkg/spatial/hashgrid_test.go`

Problem:
- Update/Delete previously required linear scan in cell bucket (`removeFromCell`), causing O(bucket_size) churn under dense cells.
- QueryAABB allocated a dedupe map on every call.

Change:
- Added reverse slot index (`map[int]cellSlot`) storing `{cell_key, bucket_index}` per object ID.
- Delete/Upsert now remove in O(1) via swap-delete + slot fix-up.
- QueryAABB removed dedupe map allocation, since each ID is guaranteed to occupy one slot.

Impact:
- Lower update churn and allocator pressure for high-rate object motion/rebuild workloads.
- Stable semantics retained for range/radius queries.

Validation:
- Existing query tests still pass.
- Added overwrite/no-duplicate and delete-removal regression tests.

### Upgrade B: FeatureStore Ordered Inserts + Binary Windowing

Files:
- `pkg/dsp/innovation_feature_store.go`
- `pkg/dsp/innovation_feature_store_test.go`

Problem:
- Window queries scanned full record slice and then sorted result each call.

Change:
- Store now maintains ordered records by `(timestamp, object_id)`.
- Append performs fast tail-append for monotonic input and binary insertion for out-of-order arrivals.
- `Window(start, end)` now uses binary searches to slice candidate interval directly.

Complexity:
- Before: `Window` O(n) scan + O(k log k) sort
- After: `Window` O(log n + k)

Impact:
- Major speedup for repeated temporal windows in orchestrator and analytics paths.
- More predictable runtime at larger retained window sizes.

Validation:
- Added tests for out-of-order append ordering, capacity trimming, and latest sequence behavior.

## 4) Remaining High-Value Upgrade Candidates (Prioritized)

1. Catalog hypothesis/history compaction
- Candidate files: `pkg/catalog/catalog.go`, `pkg/catalog/mht_full.go`
- Upgrade: bounded ring buffers for `trackHistory`, periodic pruning index for stale hypotheses.

2. Measurement fanout backpressure policy
- Candidate file: `pkg/server/measurements_hub.go`
- Upgrade: optional per-subscriber drop counters + health ejection policy for persistently lagging channels.

3. Collision Monte Carlo memory pooling
- Candidate file: `pkg/collision/monte_carlo.go`
- Upgrade: pooled vectors/scratch buffers for worker loops to reduce allocations at large sample counts.

4. Octree API parity cleanup
- Candidate files: `pkg/octree/*`, callers in legacy/internal packages
- Upgrade: unify object interface and query signatures to remove adapter breakages.

## 5) Operational Notes

- Upgrades were implemented API-compatible to avoid downstream breakage.
- No external dependencies were introduced.
- Behavior remains deterministic and suitable for air-gapped execution.
