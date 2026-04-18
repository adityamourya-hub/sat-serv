# Indian SSA DSP Pre-Processing (Phase 2)

## Purpose

This document defines the sovereign-first DSP layer used to clean radar measurements before UKF fusion in SSIS.

Operational assumption for 2026 India:

- NETRA, MOTR, and DRDO sensors usually provide processed measurements (range, range-rate, azimuth, elevation, SNR), not always raw I/Q.
- Future Indian radars can add raw I/Q streams.
- PAC Kolkata space weather is the preferred environmental source for drag modeling.

## Measurement Model

For each sensor update:

$$
z_k = h(x_k) + v_k, \quad v_k \sim \mathcal{N}(0, R_k)
$$

Where:

- $h(x_k)$ is the radar forward model (`RadarMeasurementFromState`).
- $R_k$ is dynamic covariance scaled from innovation statistics.

## DSP Flow Implemented

Package: `pkg/dsp`

Core type:

- `MeasurementFilter`

Processing order in `ProcessMeasurement`:

1. Predict measurement using callback (`RadarMeasurementFromState` in orbital integration).
2. Compute innovation vector.
3. Compute Mahalanobis distance and apply gate (default $\sqrt{\chi^2_{4,0.99}} \approx 3.64$).
4. Update innovation history window.
5. Update recursive bias estimate (EWMA).
6. Correct bias and scale measurement uncertainty dynamically.
7. Return cleaned measurement to UKF.

## Integration in UKF

`pkg/orbital/fusion.go` now calls DSP immediately before radar update inside `FuseWithUKF`.

A callback hook is available in `FusionInput`:

- `OnCleanedMeasurement func(dsp.CleanedMeasurementRecord)`

This can be wired to:

- persistence (`PersistMeasurementRecord`),
- and UI streaming (WebSocket/SSE bridge in server runtime).

## Persistence

`pkg/persistence/store.go` adds:

- `PersistMeasurementRecord(sample dsp.CleanedMeasurementRecord) (SignedRecord, error)`

Records are signed with Ed25519 and stored through ontology-event backend as event type `measurement_record`, preserving provenance and replayability.

## Performance Validation

`pkg/dsp/measurement_filter_test.go` includes benchmark:

- `BenchmarkProcessMeasurement`

Run:

```bash
go test -bench BenchmarkProcessMeasurement -benchmem ./pkg/dsp
```

Target acceptance for Phase 2 operations:

- >= 10k measurements/sec aggregate throughput
- < 5 ms per measurement path

## Indian Sensor Mapping

- MOTR: processed range/range-rate/angles directly enters DSP filter.
- NETRA: optical-angle tracks can use same innovation gating and adaptive covariance strategy.
- DRDO dual-use radars: high-quality Doppler inputs benefit from bias compensation and dynamic R scaling.

## Notes

- Current implementation is pure Go and has no external runtime DSP dependency.
- Raw I/Q processing (pulse compression, FFT Doppler, CA/OS-CFAR) can be layered later in `pkg/dsp` without changing the UKF interface.
