# ADITI 4.0 IRL Operational Evidence (Workspace-Verified)

Date: 2026-04-12
Scope: Evidence captured from executed commands and generated artifacts in this workspace.

## 1) Executed Pipeline Evidence

### Dataset bootstrap
- Script: scripts/download_aditi_datasets.sh
- Latest summary report: data/aditi4/manifests/download_report.md
- Detailed status table: data/aditi4/manifests/download_status.tsv

Latest status (from report):
- Downloaded/updated: 3
- Already present: 10
- Blocked: 4
- Failed: 4

Blocked/failed highlights:
- Space-Track API requires credentials (SPACETRACK_ID / SPACETRACK_PASSWORD).
- TraCSS bulk artifacts require approved direct URLs (TRACSS_EPHEMERIS_URL / TRACSS_CDM_URL).
- Figshare full archive attempt failed for data.zip in this environment.
- SPLID Dropbox challenge payload attempt failed from discovered link.

### Full-start training launcher
- Script: scripts/run_aditi_fullstart.sh
- Training output directory: artifacts/ml_training_aditi_fullstart
- Report: artifacts/ml_training_aditi_fullstart/training_report.json
- Config log: artifacts/ml_training_aditi_fullstart/training_config.json

## 2) Training Run Metrics (from training_report.json)

Run metadata:
- Verdict: conditional
- IAF mode: true
- Strict mode: false
- Duration: 69733 ms
- Signature valid: true

Dataset loads used in this run:
- XP-TLE: 10156 records (data/aditi4/celestrak/gp_starlink.tle)
- Covariance: 296534 records (../comsoc_data)
- Weather: 25030 records (../comsoc_data/prod_2026-04-11.sw.full.txt)
- EOP: 23476 records (../comsoc_data/prod_2026-04-11.eop.full.txt)
- Catalog: 68466 records (data/aditi4/celestrak/satcat.csv)

Model-level metrics:
- Maneuver intent: F1 0.9837606837606837, precision 0.968040370058873, recall 1.0
- Drag tuner: RMSE 0.0003928770253257885
- Orbit zoo RL: reward improvement 0.24275679887560792
- Covariance realism: innovation_consistent false, nees_consistent false

Produced artifacts:
- artifacts/ml_training_aditi_fullstart/models/maneuver_intent.onnx
- artifacts/ml_training_aditi_fullstart/models/covariance_realism.onnx
- artifacts/ml_training_aditi_fullstart/models/drag_tuner.onnx
- artifacts/ml_training_aditi_fullstart/models/orbit_zoo_rl.onnx
- artifacts/ml_training_aditi_fullstart/models/plasma_pinn.jax.json

## 3) Additional Measured Runtime Evidence

### ML latency-oriented test
Command:
- go test -run TestBenchmarkLatency -v ./pkg/ml

Observed output:
- [Latency] 1000 predictions completed (should be <1s for <50ms target)
- Package test completed successfully.

### Orchestrator benchmark sample (single iteration)
Command:
- go test -run '^$' -bench BenchmarkMeasurementOrchestrator_Process -benchtime=1x ./pkg/orbital

Observed output:
- BenchmarkMeasurementOrchestrator_Process-8: 1505921375 ns/op
- Allocations: 120480 B/op, 1560 allocs/op

## 4) Evidence Gaps (Not Demonstrated by Current Artifacts)

The following claims are not directly proven by currently stored artifacts in this workspace:
- Full MLflow experiment lineage evidence (no mlruns directory found in repo paths used here).
- DVC lineage files for this run (.dvc or dvc.yaml not present in this run path set).
- End-to-end radar-to-alert <500 ms proven by dedicated operational benchmark artifact.
- Full SpaceTrack-TimeSeries 9.7 GB dataset pull success in this environment.
- SPLID challenge payload full download success in this environment.
- TraCSS bulk ephemeris/CDM files downloaded and validated in this environment.

## 5) Interpretation for Acceptance Review

What is verified now:
- End-to-end dataset bootstrap plus training execution works.
- Signed training output is generated and verifiable.
- ONNX/JAX model artifacts are emitted and tracked with hashes.
- Maneuver model metric in this run is above 0.92 F1.

What remains to claim "fully validated IRL production" with strict audit posture:
- Complete blocked dataset acquisition with credentials/approved URLs.
- Re-run in strict mode and retain strict-mode artifact report.
- Capture and store explicit e2e latency benchmark artifact tied to acceptance thresholds.
- Export MLflow + DVC lineage files into immutable run bundle.
