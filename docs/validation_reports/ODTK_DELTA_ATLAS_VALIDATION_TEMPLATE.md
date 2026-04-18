# SSIS Orbital Layer Validation Report (ATLAS / Delta / ODTK Benchmark)

## 1. Program Context
- Mission Window:
- Catalog Scope: (objects)
- Observation Volume: (obs/day)
- Data Sources: `assets/indian_satellite_catalogue.json`, Celestrak, Space-Track, TraCSS set
- Build ID / Git SHA:
- Air-Gapped Execution Node:

## 2. Orbit Propagation Fidelity
- Force Model Profile: J2-J6 + drag + SRP + third-body + tides
- Integrator Settings: step, tolerance, process noise mode (SNC / ASNC)
- Runtime Throughput: propagated states/sec
- 3sigma Envelope Stability: pass/fail
- Energy Drift (median, p95):

## 3. UKF Sensor Fusion & OD
- OD Mode: sequential / batch
- Sensors fused: TLE + radar (range, range-rate, az/el)
- Epochs processed:
- Acceptance Ratio (Mahalanobis gate):
- Innovation Consistency (NIS): mean, p95
- Estimation Consistency (NEES): mean, p95
- Covariance Realism Verdict: PASS/FAIL

## 4. Maneuver Detection & Characterization
- Detector: CUSUM + covariance-aware normalization
- Maneuver Detection F1:
- Detection Latency (seconds): p50/p95
- Characterization outputs:
  - Maneuver time
  - Delta-V magnitude (km/s)
  - Delta-V RSW components [R,S,W]
  - Burn type proxy
  - Intent proxy + score
- False-positive drift suppression result:

## 5. Catalog Maintenance & Rapid Re-catalog
- Association Engine: MHT-lite / PDA
- UCT Count:
- Maneuver-triggered hypothesis splits:
- Tracks-to-promotion for new orbit (target < 3):
- Rapid recatalog latency (sec): p50/p95

## 6. Conjunction / Collision Risk Calibration
- Screened Pairs:
- High-fidelity Pc Method: Foster / Chan / Alfano variant
- Pc Calibration:
  - Brier score
  - Reliability error
  - Calibration verdict
- Comparison against ODTK reference cases:

## 7. Scale Test Results
- Catalog size:
- Observations/day:
- Conjunction throughput (10k objects): elapsed seconds
- Maneuver F1 and p95 latency:
- UKF acceptance ratio:
- Covariance realism metrics included: yes/no

## 8. Signed Audit Artifacts
- Signed validation JSON path:
- Signature valid: true/false
- Signature algorithm: HMAC-SHA256
- Replay command:

## 9. Reviewer Verdict (IAF)
- Operational readiness: PASS/CONDITIONAL/FAIL
- Delta parity confidence: HIGH/MEDIUM/LOW
- ODTK parity confidence: HIGH/MEDIUM/LOW
- ATLAS recatalog parity confidence: HIGH/MEDIUM/LOW
- Action items before deployment:
