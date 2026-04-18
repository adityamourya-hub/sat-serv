# Astra-Shakti Technical Development Plan
## 24-Month Implementation Roadmap for Indigenous Space Domain Awareness

**Document**: Technical Execution Plan  
**Version**: 1.0  
**Classification**: RESTRICTED — Internal Development  
**Based On**: @/Users/adi/idex_iaf/docs/US_DEFENSE_TECH_BRAINSTORM.md  

---

## Executive Summary

This plan transforms the Astra-Shakti architecture into executable workstreams. It specifies **exact deliverables**, **technical milestones**, **resource requirements**, and **success criteria** for each of the 5 AI models and 6 architecture layers.

**Total Budget Allocation** (₹25 Crore):
- Personnel (40 engineers): ₹12 Cr
- Hardware (M2/M4 cluster, sensors): ₹6 Cr
- ISRO/IAF integration: ₹4 Cr
- Certification & documentation: ₹3 Cr

---

## Phase 0: Foundation & Architecture Hardening (Months 1-3)

### Objective
Establish the sovereign substrate: air-gapped build pipeline, cryptographic signing, and SSIS engine production readiness.

### Deliverables

| ID | Deliverable | Specification | Owner |
|:---|:---|:---|:---|
| P0-D1 | Reproducible Build System | SLSA-3 compliant, deterministic Go builds with in-toto attestations | Platform Team |
| P0-D2 | HSM Integration | YubiKey/Thales HSM for Ed25519 signing key generation and storage | Crypto Team |
| P0-D3 | SSIS Engine v1.0 | 3.3 MB static binary, all 9 ADITI capabilities functional | Core Team |
| P0-D4 | BFTL Prototype | Byzantine-Fault Tolerant Ledger for 3-node consensus | Mesh Team |
| P0-D5 | ATeena Data Adapter | gRPC ingest for ATeena radar measurement packets | Integration Team |

### Technical Milestones

**Month 1**:
- [ ] Fix all `go build` errors across `pkg/` tree
- [ ] Implement `pkg/ingest/adapters/radar_measurement_adapter.go`
- [ ] HSM key ceremony with IAF security officers
- [ ] Deploy SSIS v1.0 to first ATeena ground station (Nellore)

**Month 2**:
- [ ] BFTL Raft consensus working between Nellore → Bangalore → Leh
- [ ] Signed measurement provenance: every ATeena detection Ed25519 signed
- [ ] Performance: 10k objects tracked with <5s latency

**Month 3**:
- [ ] First Hamiltonian-predicted blind-spot propagation validated
- [ ] Documentation: SDLC Phase 1 complete (requirements, architecture)

### Success Criteria
```
✓ go build ./... zero errors
✓ go test ./... 100% pass (90%+ coverage)
✓ SSIS binary size ≤ 3.5 MB
✓ BFTL 3-node consensus: 99.9% uptime over 72h test
✓ ATeena ingest: 100 Hz sustained (1 second batches)
```

---

## Phase 1: Sensor Integration & Data Pipeline (Months 4-6)

### Objective
Integrate all IAF/ISRO sensors into unified data fabric with quality gates and feature stores.

### Deliverables

| ID | Deliverable | Specification |
|:---|:---|:---|
| P1-D1 | ATeena-IQNet v0.5 | 1D-CNN + LSTM, detects at SNR -3 dB, ONNX export |
| P1-D2 | Measurement Feature Store | PostgreSQL `measurement_records` table with 25+ feature columns |
| P1-D3 | Sensor Registry v2 | JSON-based configuration for all ATeena + NETRA + ISTRAC sites |
| P1-D4 | Live SSE Stream | Server-Sent Events endpoint for real-time measurement broadcast |
| P1-D5 | Replay Pipeline | Generate labeled Parquet datasets from feature store |

### Technical Specifications

**ATeena-IQNet Architecture**:
```python
# ai/sensing/atena_iqnet.py
class ATeenaIQNet(nn.Module):
    def __init__(self):
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=128, stride=4),    # I/Q input
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=64, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.lstm = nn.LSTM(128, 256, num_layers=2, 
                           batch_first=True, dropout=0.2)
        self.detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Detection probability
            nn.Sigmoid()
        )
        self.estimator = nn.Linear(256, 3)  # Range, Doppler, RCS
```

**Training Data Requirements**:
- 10,000 hours of ATeena recordings (ISRO/IAF archives)
- Synthetic weak signals: add Gaussian noise to known satellite tracks
- Labels: Ground truth from optical correlation + ISTRAC ephemeris

**Feature Store Schema**:
```sql
CREATE TABLE measurement_records (
    -- Provenance
    id BIGINT PRIMARY KEY,
    sensor_tx_ns BIGINT,           -- Sensor transmit timestamp
    receive_ns BIGINT,               -- Processing timestamp
    sensor_id VARCHAR(32),           -- 'atena_nellore', 'netra_leh', etc.
    sensor_type VARCHAR(16),         -- 'radar', 'optical', 'sar'
    site_id INTEGER,
    calibration_version VARCHAR(16),
    provenance_hash BYTEA,           -- SHA-256 of raw packet
    
    -- Raw measurement
    range_km FLOAT,
    range_rate_km_s FLOAT,
    azimuth_deg FLOAT,
    elevation_deg FLOAT,
    snr_db FLOAT,
    rcs_m2 FLOAT,
    
    -- UKF innovation
    innovation_vector FLOAT[4],
    mahalanobis_d2 FLOAT,
    gated_flag BOOLEAN,
    covariance_scale_alpha FLOAT,
    
    -- UKF state
    ukf_pre_state FLOAT[6],          -- [x,y,z,vx,vy,vz]
    ukf_post_state FLOAT[6],
    ukf_cov_pre FLOAT[36],           -- 6x6 flattened
    ukf_cov_post FLOAT[36],
    trace_p FLOAT,
    max_eigen_p FLOAT,
    
    -- Metadata
    object_id INTEGER,
    maneuver_probability FLOAT,
    signature BYTEA                  -- Ed25519 signature
);
```

### Milestones

**Month 4**:
- [ ] ATeena-IQNet training: F1 > 0.90 on held-out test set
- [ ] Feature store deployed to PostgreSQL cluster (3-node replication)
- [ ] 1 million measurement records ingested from ATeena archives

**Month 5**:
- [ ] SSE stream: 100 concurrent subscribers, <100ms latency
- [ ] Replay pipeline: Generate training datasets in <30 minutes
- [ ] NETRA optical integration: Manual correlation with ATeena tracks

**Month 6**:
- [ ] ATeena-IQNet ONNX export: inference <5ms on M2
- [ ] First autonomous weak-signal detection (SNR < 0 dB) validated
- [ ] SDLC Phase 2 complete (design specifications)

---

## Phase 2: Multi-Static Fusion & SAR Integration (Months 7-12)

### Objective
Fuse heterogeneous sensors (ATeena + RISAT + NavIC + optical) into unified orbit estimates via GNN.

### Deliverables

| ID | Deliverable | Specification |
|:---|:---|:---|
| P2-D1 | SAR-OrbitNet v0.5 | ResNet18 + ISAR layer, outputs [x,y,z,vx,vy,vz] |
| P2-D2 | NavIC Bistatic Processor | Correlates NavIC signals with ATeena returns |
| P2-D3 | Multi-Static Fusion GNN | GATv2, fuses 4 sensor types, outputs orbit + uncertainty |
| P2-D4 | Ionospheric Tomography | 3D-CNN, electron density cube from NavIC TEC |
| P2-D5 | Measurement Orchestrator | Unifies all sensor streams into UKF pipeline |

### Technical Specifications

**Multi-Static Fusion GNN**:
```python
# ai/fusion/multistatic_gnn.py
class MultiStaticFusionGNN(nn.Module):
    """
    Graph: Nodes = sensors, Edges = simultaneous observations
    """
    def __init__(self, num_sensor_types=4):
        self.sensor_encoder = nn.Embedding(num_sensor_types, 64)
        
        # GATv2: Graph Attention with dynamic edge weights
        self.gat1 = GATv2Conv(64, 128, heads=4, concat=True)
        self.gat2 = GATv2Conv(512, 256, heads=2, concat=False)
        
        # Orbit decoder
        self.orbit_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # State vector
        )
        
        # Uncertainty head (outputs covariance Cholesky factors)
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 21)  # Lower triangular of 6x6 covariance
        )
        
    def forward(self, sensor_features, edge_index, edge_attr):
        # sensor_features: [num_sensors, feature_dim]
        # edge_index: [2, num_edges] — which sensors observe same object
        # edge_attr: [num_edges, 4] — time_diff, angular_sep, gdop, confidence
        
        x = self.sensor_encoder(sensor_features)
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        
        # Global mean pooling across sensor nodes
        x_global = scatter_mean(x, batch_index, dim=0)
        
        orbit_state = self.orbit_mlp(x_global)
        cov_chol = self.uncertainty_mlp(x_global)
        
        return orbit_state, cov_chol
```

**NavIC Bistatic Algorithm**:
```python
# ai/sensing/navic_bistatic.py
def navic_bistatic_detect(navic_ephemeris, atena_iq_data, prn_code):
    """
    Detect space objects using NavIC signals reflected off them.
    
    Steps:
    1. Compute direct path delay from NavIC sat to ATeena
    2. Search reflected path in ATeena IQ data
    3. Neural enhancement of weak correlation peaks
    """
    # Direct path: known from NavIC ephemeris + ATeena position
    direct_delay = compute_direct_path(navic_ephem, atena_position)
    
    # Reflected path: search in IQ data
    correlation = matched_filter(atena_iq_data, prn_code)
    
    # Neural denoising
    denoised_peak = cnn_denoiser(correlation)
    
    # Detect reflected signal
    if denoised_peak > threshold:
        delay_diff = locate_peak(denoised_peak) - direct_delay
        range_to_object = delay_diff * SPEED_OF_LIGHT / 2
        return Detection(range_to_object, confidence=denoised_peak)
    
    return None
```

**Ionospheric Tomography**:
- Input: Slant Total Electron Content (TEC) from 50+ NavIC ground stations
- Output: 3D electron density cube (lat: 0-35°N, lon: 65-95°E, alt: 100-1000 km)
- Resolution: 1° x 1° x 25 km
- Update rate: 10 minutes

### Training Data Requirements

| Model | Data Source | Volume | Labels |
|:---|:---|:---|:---|
| SAR-OrbitNet | RISAT-2B calibration images | 50,000 chips | Ground truth from ISTRAC TLE |
| NavIC Bistatic | Simulated + ATeena recordings | 100,000 events | Optical correlation |
| Fusion GNN | All sensors, 6 months | 10M multi-sensor observations | ISTRAC ephemeris |
| Ionospheric | NavIC ground stations | 1 year continuous | JPL GIM reference |

### Milestones

**Month 7**:
- [ ] RISAT integration: Data pipeline from ISRO to SSIS
- [ ] SAR-OrbitNet training: RMSE < 5 km on orbit estimate

**Month 8**:
- [ ] NavIC bistatic: First detection of LEO object via reflected signal
- [ ] Ionospheric model: 30% improvement in ATeena range accuracy

**Month 9**:
- [ ] Fusion GNN: Beats single-sensor UKF by 40% in covariance realism
- [ ] 3-sensor fusion demo: ATeena + RISAT + optical → single orbit estimate

**Month 10**:
- [ ] Maya Layer v0.1: Detects Pattern-of-Life deviations
- [ ] Reachability Risk Index: Real-time computation for 100 Indian assets

**Month 11**:
- [ ] 4-sensor fusion: Add NavIC to mix
- [ ] Hamiltonian blind-spot prediction: 72-hour horizon validated

**Month 12**:
- [ ] All models ONNX-exported, running on M2 edge nodes
- [ ] SDLC Phase 3 complete (implementation, unit testing)

---

## Phase 3: AI Deployment & Rapid Processing (Months 13-18)

### Objective
Deploy all AI models to operational edge nodes, achieve <1s latency for hypersonic detection.

### Deliverables

| ID | Deliverable | Specification |
|:---|:---|:---|
| P3-D1 | ATeena-IQNet v1.0 | Production: F1 > 0.95, latency <5ms |
| P3-D2 | SAR-OrbitNet v1.0 | Production: RMSE < 2 km, latency <50ms |
| P3-D3 | Fusion GNN v1.0 | Production: <100ms for 4-sensor fusion |
| P3-D4 | PPO Threat Policy | Trained on 50k scenarios, 8-action space |
| P3-D5 | Hypersonic Pipeline | End-to-end <1s: detection → classification → alert |
| P3-D6 | Swaha-Autonomy | "Human-on-the-loop" for <5min threats |

### Technical Specifications

**PPO Threat Policy (Rapid Processing)**:
```python
# ai/rapid/train_policy_ppo.py (enhanced)
class ThreatPolicyPPONet(nn.Module):
    """
    Observation space: 8-dimensional threat state
    Action space: 8 discrete actions
    """
    def __init__(self):
        self.shared = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor = nn.Linear(256, 8)    # Policy logits
        self.critic = nn.Linear(256, 1)    # Value function
        
    def forward(self, obs):
        # obs: [threat_score, confidence, reachability, 
        #       hypersonic_likelihood, time_pressure, 
        #       sensor_quality, asset_priority, weather_impact]
        features = self.shared(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

# Actions:
ACTIONS = [
    'ATeena_priority_track',      # High-rate tracking
    'ATeena_wide_scan',           # Search mode
    'RISAT_sar_retask',           # Repoint SAR
    'CARTOSAT_optical_image',     # Visual confirmation
    'GSAT_comms_relay',           # Data relay
    'NavIC_pseudorange_collect',  # Timing verification
    'sensor_fusion_activate',       # Multi-sensor correlation
    'alert_escalate'              # Human operator
]
```

**Hypersonic Pipeline Latency Budget**:
| Stage | Component | Latency | Platform |
|:---|:---|:---|:---|
| 1 | ATeena detection | 50 μs | Hardware |
| 2 | IQNet inference | 5 ms | M2 Neural Engine |
| 3 | UKF update | 200 μs | M2 CPU |
| 4 | L/D estimation | 100 μs | M2 CPU |
| 5 | PPO policy | 10 ms | M2 Neural Engine |
| 6 | Alert generation | 30 μs | M2 CPU |
| 7 | Signature + SSE | 50 μs | M2 CPU |
| **Total** | | **~16 ms** | **Well under 1s budget** |

### Milestones

**Month 13**:
- [ ] All ONNX models deployed to 3 edge nodes
- [ ] Model versioning and A/B testing framework
- [ ] Model drift detection: alert when accuracy degrades

**Month 14**:
- [ ] PPO training: 50,000 simulated threat scenarios
- [ ] First autonomous sensor retasking decision (human verified)

**Month 15**:
- [ ] Hypersonic detection: <1s validated in live exercise
- [ ] Maya score: 90% accuracy on deception detection

**Month 16**:
- [ ] "Human-on-the-loop" mode: 10 autonomous decisions per day
- [ ] Operator acceptance: >80% positive feedback

**Month 17**:
- [ ] 100k object tracking: Sustained for 7 days
- [ ] Conjunction screening: 10k pairs in <60 seconds

**Month 18**:
- [ ] Full stack stress test: 72-hour continuous operation
- [ ] Failover validated: Node loss → automatic recovery <30s
- [ ] SDLC Phase 4 complete (system testing)

---

## Phase 4: Certification & Operational Hardening (Months 19-24)

### Objective
Achieve IAF certification, operator training, live exercise validation.

### Deliverables

| ID | Deliverable | Specification |
|:---|:---|:---|
| P4-D1 | Gobra Verification | Memory safety proofs for critical modules |
| P4-D2 | HAZOP Analysis | Hazard and operability study, all failure modes documented |
| P4-D3 | FMEA Report | Failure mode effects analysis, mitigations specified |
| P4-D4 | Operator Training | 200 IAF personnel trained, certified |
| P4-D5 | Live Exercise | Simulated ASAT approach detected and tracked |
| P4-D6 | Certification Package | Complete SDLC documentation for IAF approval |

### Certification Requirements

**Security**:
- [ ] No hardcoded credentials (validated via SAST)
- [ ] All communications mTLS 1.3 with mutual auth
- [ ] Ed25519 signing on every measurement, alert, and decision
- [ ] HSM key management with dual-control

**Safety**:
- [ ] Fail-soft: Single component failure doesn't cascade
- [ ] Graceful degradation: Reduced capability vs. total failure
- [ ] Operator override: Any autonomous decision can be vetoed in <1s

**Performance**:
- [ ] 99.9% uptime over 30-day test
- [ ] <1s latency for hypersonic threats (validated over 100 trials)
- [ ] 100k object capacity validated

**Traceability**:
- [ ] Every decision logged with full provenance chain
- [ ] Deterministic replay: Same inputs → same outputs (bit-identical)
- [ ] Audit trail: 7-year retention, tamper-evident

### Milestones

**Month 19**:
- [ ] Gobra proofs: `pkg/orbital/`, `pkg/crypto/`, `pkg/fusion/`
- [ ] HAZOP workshop: All stakeholders, all failure modes

**Month 20**:
- [ ] FMEA report: 200+ failure modes analyzed, mitigations implemented
- [ ] Security audit: External red team engagement

**Month 21**:
- [ ] Training curriculum: 40-hour operator course
- [ ] First training cohort: 50 IAF officers

**Month 22**:
- [ ] Second training cohort: 50 IAF officers
- [ ] Training simulator: Deployed to 5 IAF bases

**Month 23**:
- [ ] Live exercise: Simulated PLA ASAT approach on IRNSS-1I
- [ ] Exercise evaluation: Detection, tracking, manoeuvre prediction

**Month 24**:
- [ ] Final certification: IAF sign-off
- [ ] Handover to IAF operations
- [ ] Maintenance contract: 5-year support with model retraining

---

## Resource Requirements

### Personnel (40 engineers, ₹12 Cr)

| Team | Count | Skills | Duration |
|:---|:---|:---|:---|
| Core Platform | 8 | Go, systems, distributed systems | 24 months |
| AI/ML | 10 | PyTorch, JAX, ONNX, signal processing | 24 months |
| Integration | 6 | Radar, ISRO interfaces, gRPC | 18 months |
| Mesh/Security | 4 | Cryptography, BFT systems, HSM | 18 months |
| DevOps/QA | 4 | Kubernetes, CI/CD, testing | 24 months |
| UI/UX | 3 | Svelte, CesiumJS, Bubbletea | 12 months |
| Documentation | 2 | Technical writing, SDLC | 18 months |
| Management | 3 | Project management, IAF liaison | 24 months |

### Hardware (₹6 Cr)

| Item | Quantity | Unit Cost | Purpose |
|:---|:---|:---|:---|
| Mac Studio M2 Ultra | 20 | ₹3L | Edge nodes (development + production) |
| Mac Studio M4 Max | 10 | ₹4L | Training cluster (JAX/PyTorch) |
| YubiKey 5 HSM | 50 | ₹15K | Cryptographic signing keys |
| Thales Luna 7 HSM | 2 | ₹20L | Master key ceremony |
| Storage (NAS) | 1 | ₹10L | Model artifacts, datasets |
| Networking | - | ₹5L | mTLS infrastructure |
| Test ATeena Unit | 1 | ₹2Cr | Lab testing (IAF provisioned) |

### ISRO/IAF Integration (₹4 Cr)

| Activity | Cost | Description |
|:---|:---|:---|
| RISAT Integration | ₹1.5 Cr | Data pipeline, API development |
| NavIC Raw Access | ₹1 Cr | Ground station data feeds |
| ATeena Integration | ₹1 Cr | Real-time stream development |
| ISTRAC Archive | ₹0.5 Cr | Historical data for training |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|
| ATeena data access delays | Medium | High | Start with synthetic data; IAF/DRDO escalation |
| RISAT integration complexity | Medium | Medium | Phased: synthetic → calibration → operational |
| Model accuracy below target | Low | High | Classical fallback always available; iterative improvement |
| IAF certification delays | Medium | Medium | Early engagement; parallel documentation |
| Key personnel departure | Low | High | Knowledge documentation; cross-training |
| Hardware supply issues | Low | Medium | Alternative vendors; cloud backup for training |

---

## Success Criteria Summary

| Metric | Target | Measurement |
|:---|:---|:---|
| ATeena-IQNet F1 | > 0.95 | Held-out test set |
| SAR-OrbitNet RMSE | < 2 km | vs. ISTRAC ephemeris |
| Fusion GNN improvement | 40% vs. single-sensor | Covariance realism (NIS) |
| End-to-end latency | < 1s | Hypersonic detection pipeline |
| System uptime | 99.9% | 30-day operational test |
| Object capacity | 100k | Sustained tracking |
| Conjunction screening | 10k pairs < 60s | Benchmark test |
| Operator satisfaction | > 80% | Post-training survey |
| Maya detection accuracy | > 90% | Deception validation set |
| Certification | Pass | IAF sign-off |

---

## Appendix A: File Structure

```
ssis/
├── engine/                    # Go runtime
│   ├── main.go
│   ├── pkg/
│   │   ├── orbital/           # Propagators, UKF, rKAN
│   │   ├── crypto/            # Signing, HSM
│   │   ├── fusion/            # Multi-sensor fusion
│   │   ├── dsp/               # Radar signal processing
│   │   ├── ingest/            # Sensor adapters
│   │   │   └── adapters/
│   │   │       └── radar_measurement_adapter.go
│   │   ├── mesh/              # BFTL, consensus
│   │   └── server/            # SSE, gRPC
│   └── conjunction.go
├── ai/                        # Python training
│   ├── sensing/
│   │   ├── atena_iqnet.py
│   │   ├── navic_bistatic.py
│   │   └── quantum_radar.py
│   ├── orbit/
│   │   └── sar_orbitnet.py
│   ├── fusion/
│   │   └── multistatic_gnn.py
│   ├── rapid/
│   │   └── train_policy_ppo.py
│   └── v3/                    # Axiom-Zero
│       ├── sfmn_kernel.py
│       ├── hamiltonian_discovery.py
│       └── maneuver_distiller.py
├── models/                    # ONNX artifacts
│   ├── physics/
│   ├── manoeuvre/
│   ├── collision/
│   └── rapid/
├── ui/                        # Operator interface
│   ├── web/                   # Svelte + CesiumJS
│   └── tui/                   # Bubbletea
└── docs/                      # This plan
```

---

**Document Control**  
**Author**: Technical Lead  
**Reviewers**: AI Lead, Platform Lead, Security Lead  
**Approved By**: Program Director  
**Next Review**: Monthly, or on Phase completion

---

*This plan transforms strategic vision into executable reality. Every line of code, every model, every test is traceable to an IAF operational requirement.*
