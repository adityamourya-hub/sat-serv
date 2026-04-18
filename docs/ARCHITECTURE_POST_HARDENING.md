# SSIS Defense-Scale Architecture — Post-Hardening

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SATELLITE CATALOG INGESTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  Catalog JSON File   │
│  (indian_sat.json)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  JSON Unmarshal                                      │
│  → Scan satellite entries                            │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  PARALLEL WORKERS (per-record fail-soft)             │
│  ────────────────────────────────────────────────────│
│  For each satellite entry:                           │
│    ├─ Validate TLE length (≥2 lines) ✅             │
│    │  └─ If fail: log, skip, continue               │
│    │                                                  │
│    ├─ ParseTLE() with strict validation              │
│    │  ├─ Line 1: format + checksum + fields ✅       │
│    │  ├─ Line 2: format + checksum + fields ✅       │
│    │  └─ Range checks: inc, ecc, raan, mm ✅         │
│    │     └─ If fail: log (name, ID, reason), skip   │
│    │                                                  │
│    ├─ SGP4 Propagate @ epoch ✅                      │
│    │  └─ Compute position (X,Y,Z)                    │
│    │                                                  │
│    └─ Octree Insert ✅                               │
│       └─ Track in spatial index                      │
│                                                       │
│  On error: increment failure counter                 │
│  On success: append to tracked[] array               │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  INGEST PHASE SUMMARY                                │
│  ────────────────────────────────────────────────────│
│  ✅ Loaded N satellite records                       │
│  ⚠️  Skipped M invalid records (logged)              │
│  → Ready for conjunction screening                   │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│              CONJUNCTION SCREENING PIPELINE                      │
│  ──────────────────────────────────────────────────────────────  │
│                                                                   │
│  Broadphase (Octree):                                             │
│  • Query radius around each satellite (80 km default)             │
│  • Collect nearby candidates (O(log n) lookup)                    │
│                                                                   │
│  Narrowphase (Parallel Workers):                                  │
│  For each candidate pair (a, b):                                  │
│    ├─ Adaptive TCA search (±10 min window)                        │
│    ├─ Collision risk evaluation (Alfano/QMC)                      │
│    ├─ Generate report (if Pc > threshold)                         │
│    ├─ Sign report with crypto.Signer                              │
│    └─ Ship to alert channel                                       │
│                                                                   │
│  Error handling: Per-worker failures logged, batch continues      │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  AUDITED ALERTS & LOGS                               │
│  ────────────────────────────────────────────────────│
│  Each event carries:                                 │
│    • ObjectID (NORAD Catalog ID)                     │
│    • EventType (e.g., NEIGHBOURHOOD_WATCH_ALERT)    │
│    • Timestamp (Unix epoch)                          │
│    • HMAC Proof (signature under SSIS_SIGNING_KEY)   │
│                                                       │
│  Forwarded to:                                       │
│    ├─ Central audit vault (immutable)                │
│    ├─ Real-time alerting system                      │
│    └─ Compliance log (for defense audits)            │
└──────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                         SECURITY BOUNDARY                                │
└─────────────────────────────────────────────────────────────────────────┘

KEY MANAGEMENT:
  • SSIS_SIGNING_KEY (env var) ← Must be set before binary execution
  • HSM/KMS integration ready (extend NewSignerFromEnv)
  • Audit entry shows who, what, when (not proof payload itself)

DATA INTEGRITY:
  • TLE checksums validated per spec (NASA format)
  • Orbital elements range-checked (physics first-principles)
  • Parse errors logged per-record (name, ID, error reason)
  • Trackable back to source catalog version/epoch

RESILIENCE:
  • Fail-soft ingest: bad sat → skip, next sat processes
  • Fail-soft conjunction: bad pair → skip, next pair processes
  • Grid rebuild optional (budget CPU for 10k+ catalogs)
  • No single-point failures in worker pools

SCALABILITY:
  • Octree spatial index: O(log n) queries
  • Parallel workers: GOMAXPROCS-tuned concurrency
  • Incremental updates: ApplyDeltaWithRebuild(false) for small deltas
  • Catalog store: ready for horizontal sharding


OBSERVABILITY:
  ✅ Console output: [Asset/Threat/Skip] messages with ID
  ✅ Audit entries: ObjectID, EventType, Timestamp, Proof
  ✅ Error logs: satellite name, NORAD ID, parse reason
  ✅ Failure summaries: counts + percentages per phase
  ✅ Binary size: 3.3 MB (no large dependencies)
```

---

## Deployment Model

### Single-Instance (Current — up to ~10k objects per pass)
```bash
# 1. Set signing key
export SSIS_SIGNING_KEY="$(openssl rand -hex 32)"

# 2. Run engine
./ssis-engine < catalog.json > alerts.jsonl

# 3. Consume alerts
tail -f alerts.jsonl | tee audit-vault/catalog-$(date +%s).log
```

### Distributed (Future — 100k+ objects, sub-second latency)
```
Catalog Ingest Service (sharded by NORAD_CAT_ID % N):
  ├─ Worker 1: IDs 0-33k → [Catalog1 {octree, propagator}]
  ├─ Worker 2: IDs 33k-66k → [Catalog2 {octree, propagator}]
  └─ Worker 3: IDs 66k-100k → [Catalog3 {octree, propagator}]

Conjunction Federation:
  ├─ Broadphase: Distributed octree queries (RPC)
  ├─ Narrowphase: Per-shard workers (local)
  └─ Alerting: Unified topic (Kafka/Pub-Sub)
```

---

## Next Major Upgrades

| Phase | Focus | Effort | Impact |
|-------|-------|--------|--------|
| **3.1** | SGP4-VER + EKF covariance | 2 weeks | TRL-5 orbital fidelity |
| **3.2** | RSA-4096 non-repudiation | 1 week | Defense-grade signing |
| **4.0** | Horizontal sharding (3-shard pilot) | 4 weeks | 100k catalog support |
| **5.0** | Real-time ingestion (Kafka), streaming conjunction | 8 weeks | Operational dashboard |

---

**Architecture Validated:** ✅ All critical paths hardened and tested.
