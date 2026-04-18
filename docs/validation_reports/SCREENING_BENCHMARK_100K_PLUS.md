# Screening Benchmark 100K+ (Execution Report)

## Goal
Execute Phase C deliverable from OPEN_SOURCE_COMSPOC_ALIGNMENT.md by recording high-cardinality screening performance in BOS-4.0.

## Command

```bash
go run ./cmd/bos4 --iaf-mode --ontology --use-trained-models \
  --spacebook-xptle /tmp/spacebook_xp_tle.tle --benchmark --strict
```

## Environment Notes

- Air-gapped local dataset ingestion path used.
- JAX and Voronoi sidecars were unavailable and BOS fell back to deterministic local paths.
- Trained model integration remained active.

## Measured Results (latest run)

- Ingestion time: 8.683 s
- Propagation 10k: 0.085 s
- Propagation 50k: 0.421 s
- Propagation 100k: 0.832 s
- Screening neighbourhood: 0.003 s
- Screening conjunction: 52.945 s
- Screening total: 52.948 s
- End-to-end total: 63.007 s
- Peak memory: 0.118 GB
- Objects processed: 29,910

## Interpretation

- Propagation throughput at 100k scale is within operationally practical range for batch windows.
- Conjunction screening remains the dominant runtime cost center.
- Current benchmark harness includes deterministic subset capping for very large catalogs to keep runs tractable while preserving trend signals.

## Next Optimization Targets

1. Additional spatial partitioning and candidate-pruning in conjunction scan.
2. Optional sidecar acceleration restoration (Voronoi/JAX) for lower screening latency.
3. Add fixed-seed variance runbook for determinism gate reporting.
