# Phase 2 Technical Development Plan
## Multi-Sensor Integration & Production Hardening (Months 5-8)

**Date**: April 17, 2026  
**Phase**: 2 of 4 (Integration Phase)  
**Duration**: Months 5-8 (4 months)  
**Dependencies**: Phase 1 MVP complete (Engine Runtime, Neural ODE, ATeena synthetic training)

---

## Executive Summary

Phase 2 transitions from the foundational MVP to a **production-ready multi-sensor SSA system**. This phase integrates heterogeneous sensor data through a unified fusion architecture and deploys the first operational sensor (ATeena radar) with real-time neural detection.

| Deliverable | Target Month | Success Criteria |
|:---|:---|:---|
| **Fusion GNN Production** | Month 5 | Multi-sensor graph fusion with <50ms inference |
| **ATeena-IQNet Deployment** | Month 6 | P_d > 0.90 at 0 dB SNR on synthetic data |
| **NavIC Ionospheric Module** | Month 7 | STEC calculation with <5 TECU accuracy |
| **Beta System Integration** | Month 8 | 99.5% uptime, 1000 obs/sec sustained throughput |

---

## Technical Architecture: Phase 2

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2 SYSTEM ARCHITECTURE                         │
│                      (Months 5-8: Integration Phase)                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ATeena Radar          Optical Sensors        SAR (RISAT)          NavIC
       │                      │                      │                  │
       ▼                      ▼                      ▼                  ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
│  IQ Samples  │      │   Astrometry │      │  ISAR Chips  │     │ Pseudorange  │
│  ↓ ATeena-   │      │  ↓ Star-     │      │  ↓ SAR-      │     │  ↓ TEC       │
│    IQNet     │      │    Tracker   │      │    OrbitNet  │     │    Calc      │
│  [CNN-LSTM]  │      │  [ResNet]    │      │  [ResNet-18] │     │  [LSTM]      │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘     └──────┬───────┘
       │                      │                      │                  │
       │    Detections        │    Measurements      │  Orbital Elements│  STEC
       ▼                      ▼                      ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-SENSOR FUSION GNN (Gap 4)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Graph Construction:                                                │    │
│  │  • Nodes: Tracked objects with [pos, vel, cov, RCS, brightness]     │    │
│  │  • Edges: Spatial proximity (<100km) + Orbit-family correlation     │    │
│  │                                                                     │    │
│  │  Architecture: GATv2 (already in ai/collision/) + Sensor Encoders   │    │
│  │  • Radar encoder: 8-dim → 128-dim                                   │    │
│  │  • Optical encoder: 6-dim → 128-dim                                 │    │
│  │  • SAR encoder: 10-dim → 128-dim                                    │    │
│  │  • GATv2 layers: 3 layers × 4 heads × 128 hidden                     │    │
│  │                                                                     │    │
│  │  Task Heads:                                                        │    │
│  │  • Conjunction: Pairwise risk score [0,1]                          │    │
│  │  • Maneuver: Classification [none, impulsive, continuous]          │    │
│  │  • Correlation: Catalog matching embedding                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURAL ODE PROPAGATOR (Gap 8) - ENHANCED                 │
│  • Hybrid physics + neural with online learning capability                   │
│  • Maneuver-aware propagation: detect thrust, compensate trajectory        │
│  • Uncertainty quantification: probabilistic state distributions             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENGINE RUNTIME (Gap 6) - PRODUCTION v1.0                  │
│  • gRPC streaming with backpressure handling                               │
│  • 75 UKF workers + 30 collision workers + 20 ONNX workers                  │
│  • Alert subscription service with SHAP explainability                      │
│  • Hot model swapping (zero-downtime ONNX updates)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workstream 1: Multi-Sensor Fusion GNN (Gap 4)

### Technical Specification

**Building on**: `ai/collision/train_collision_surrogate.py` (existing GATv2 foundation)

**Enhancement Scope**:
```python
class MultiSensorFusionGNN(nn.Module):
    """
    Heterogeneous sensor fusion using GATv2 attention.
    Extends existing collision surrogate to multi-task learning.
    """
    
    def __init__(self, node_dim=26, hidden=128, n_heads=4, n_layers=3):
        # Node embedding (existing)
        self.node_embed = nn.Linear(node_dim, hidden)
        
        # NEW: Sensor-specific encoders for heterogeneous data
        self.radar_encoder = SensorEncoder(
            input_dim=8,      # [range, range_rate, az, el, rcs, snr, beam_width, timestamp]
            hidden_dim=64,
            output_dim=hidden
        )
        self.optical_encoder = SensorEncoder(
            input_dim=6,      # [ra, dec, mag, filter, seeing, timestamp]
            hidden_dim=64,
            output_dim=hidden
        )
        self.sar_encoder = SensorEncoder(
            input_dim=10,     # [range, range_rate, cross_range, rotation_rate, resolution, ...]
            hidden_dim=64,
            output_dim=hidden
        )
        
        # GATv2 layers (existing foundation)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden, hidden, heads=n_heads, concat=False, edge_dim=4)
            for _ in range(n_layers)
        ])
        
        # NEW: Multi-task heads
        self.conjunction_head = ConjunctionHead(hidden)  # Pairwise risk
        self.maneuver_head = ManeuverHead(hidden)        # Per-object classification
        self.correlation_head = CorrelationHead(hidden) # Catalog matching
    
    def forward(self, graph, sensor_data):
        # Fuse sensor features into node representations
        h = self.node_embed(graph.node_features)
        
        if sensor_data['radar'] is not None:
            radar_emb = self.radar_encoder(sensor_data['radar'])
            h = h + radar_emb  # Residual addition
            
        if sensor_data['optical'] is not None:
            optical_emb = self.optical_encoder(sensor_data['optical'])
            h = h + optical_emb
            
        if sensor_data['sar'] is not None:
            sar_emb = self.sar_encoder(sensor_data['sar'])
            h = h + sar_emb
        
        # Message passing with edge attributes
        for gat_layer in self.gat_layers:
            h = gat_layer(h, graph.edge_index, graph.edge_attr)
            h = F.layer_norm(h, h.shape)
            h = F.elu(h)
        
        # Task-specific outputs
        return {
            'conjunction': self.conjunction_head(h, graph.edge_index),
            'maneuver': self.maneuver_head(h),
            'correlation': self.correlation_head(h)
        }
```

### Graph Construction Details

**Node Features** (26-dimensional):
```
Position (3):     [x, y, z] ECI coordinates (km)
Velocity (3):     [vx, vy, vz] (km/s)
Covariance (6):   Diagonal of 6x6 covariance matrix
Observables (3):  RCS (m²), visual_mag, last_detection_s
Orbit (4):        One-hot [LEO, MEO, GEO, HEO]
Sensor mask (4):  Boolean [radar_seen, optical_seen, sar_seen, navic_seen]
Temporal (3):     Age since last obs, track confidence, anomaly score
```

**Edge Construction**:
```python
def build_edges(objects, threshold_km=100):
    edges = []
    edge_attrs = []
    
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects[i+1:], start=i+1):
            dist = torch.norm(obj_i.pos - obj_j.pos)
            
            # Proximity edge
            if dist < threshold_km:
                edges.append([i, j])
                edge_attrs.append([
                    dist.item(),                    # Distance
                    torch.dot(obj_i.vel, obj_j.vel).item(),  # Velocity correlation
                    abs(obj_i.orbit_regime - obj_j.orbit_regime),  # Same regime?
                    time_since_joint_observation(obj_i, obj_j)  # Temporal correlation
                ])
            
            # Orbit family edge (same plane)
            elif same_orbit_plane(obj_i, obj_j, tol_deg=2.0):
                edges.append([i, j])
                edge_attrs.append([dist.item(), 0, 1, 0])
    
    return torch.tensor(edges).t(), torch.tensor(edge_attrs)
```

### Training Strategy

**Data Requirements**:
- Historical conjunction events: 10,000+ samples
- Maneuver catalog: 5,000+ confirmed maneuvers
- Uncorrelated tracks: 50,000+ for correlation training

**Multi-Task Loss**:
```python
L_total = α·L_conjunction + β·L_maneuver + γ·L_correlation + δ·L_graph_reg

Where:
- L_conjunction: Focal loss (handle class imbalance, few conjunctions)
- L_maneuver: Cross-entropy with class weights
- L_correlation: Contrastive loss (positive pairs = same object)
- L_graph_reg: L2 regularization on edge weights
```

**Uncertainty Quantification**:
```python
# Evidential learning for conjunction risk
# Instead of probability p, output belief mass [m_0, m_1, m_uncertain]
# Enables "I don't know" predictions when evidence is weak
```

### Implementation Milestones

| Week | Task | Deliverable |
|:---|:---|:---|
| 5.1 | Sensor encoders | Radar/Optical/SAR encoder modules |
| 5.2 | Graph construction | Edge builder with proximity + orbit family |
| 5.3 | Multi-task heads | Conjunction + Maneuver + Correlation heads |
| 5.4 | Training pipeline | Multi-task loss, evidential uncertainty |
| 6.1 | Synthetic training | Training on simulated multi-sensor scenarios |
| 6.2 | Validation | Precision/recall on held-out conjunctions |
| 6.3 | ONNX export | Dynamic batching for Go integration |
| 6.4 | Performance tuning | <50ms inference for 1000-node graphs |

### Success Criteria
- **Conjunction detection**: Precision >0.95, Recall >0.90 at P_c > 1e-4
- **Maneuver detection**: F1 > 0.85 for Δv > 0.1 m/s
- **Correlation accuracy**: >99% for objects with >3 observations
- **Inference latency**: P95 < 50ms for 1000-node graph on single GPU

---

## Workstream 2: ATeena-IQNet Production Deployment (Gap 1)

### Technical Specification

**Building on**: `ai/radar/simulator/ateena_sim.py` (synthetic training data)

**Production Architecture**:
```python
class ATeenaIQNet(nn.Module):
    """
    Production neural radar detector for ATeena phased array.
    Input: Raw IQ samples [batch, pulses, range_bins, elements, 2]
    Output: Detections with [range, doppler, angle, rcs, confidence]
    """
    
    def __init__(self, config):
        # Stage 1: Learned Matched Filter (replaces classical pulse compression)
        self.range_compression = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(1, config.pulse_width_bins)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Stage 2: Beamforming + Doppler processing
        self.beamformer = NeuralBeamformer(
            n_elements=config.n_elements,
            n_beams=32
        )
        self.doppler_fft = LearnedDopplerProcessor(
            n_pulses=config.n_pulses,
            n_doppler=config.n_doppler_bins
        )
        
        # Stage 3: 2D-CNN on Range-Doppler-Beam cube
        self.feature_extractor = ResNet2D(
            in_channels=64,
            depth=34,
            pretrained=False
        )
        
        # Stage 4: Temporal context (across CPIs)
        self.temporal_encoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Stage 5: Detection head with uncertainty
        self.detection_head = EvidentialDetectionHead(
            input_dim=512,  # 256*2 from bidirectional LSTM
            max_detections=50,
            n_param=6  # [range, doppler, az, el, rcs, confidence]
        )
    
    def forward(self, iq_samples, temporal_context=None):
        """
        iq_samples: [batch, pulses, range, elements, 2] (I/Q channels)
        temporal_context: Optional previous CPI embeddings for tracking
        """
        # Reshape for processing: [batch*elements, 2, pulses, range]
        batch_size = iq_samples.shape[0]
        iq = iq_samples.permute(0, 4, 1, 3, 2)  # [B, 2, P, E, R] -> [B, 2, P, R, E]
        
        # Stage 1: Range compression per element
        range_profiles = []
        for e in range(iq.shape[-1]):
            element_iq = iq[..., e]  # [B, 2, P, R]
            compressed = self.range_compression(element_iq)
            range_profiles.append(compressed)
        
        range_profiles = torch.stack(range_profiles, dim=-1)  # [B, C, P, R, E]
        
        # Stage 2: Beamforming
        beams = self.beamformer(range_profiles)  # [B, C, P, R, 32]
        
        # Stage 3: Doppler processing
        rd_maps = self.doppler_fft(beams)  # [B, C, D, R, 32]
        
        # Stage 4: 2D-CNN per beam
        features = []
        for b in range(rd_maps.shape[-1]):
            rd_beam = rd_maps[..., b]  # [B, C, D, R]
            feat = self.feature_extractor(rd_beam)
            features.append(feat)
        
        features = torch.stack(features, dim=1)  # [B, 32, 512]
        features = features.mean(dim=1)  # [B, 512]
        
        # Stage 5: Temporal + Detection
        if temporal_context is not None:
            lstm_out, hidden = self.temporal_encoder(
                torch.stack([features, temporal_context], dim=1)
            )
        else:
            lstm_out, hidden = self.temporal_encoder(features.unsqueeze(1))
        
        detections = self.detection_head(lstm_out[:, -1, :])
        
        return detections, lstm_out[:, -1, :]  # Return detections + context
```

### Domain Randomization for Robustness

**Training Augmentation** (simulate real-world variations):
```python
class RadarAugmentation:
    """Domain randomization for synthetic-to-real transfer."""
    
    def __init__(self):
        self.noise_range = (-20, 10)  # dB SNR variation
        self.clutter_levels = [0, 0.1, 0.5, 1.0]  # Ground clutter
        self.jamming_types = ['none', 'barrage', 'spot', 'swept']
        self.multipath_scenarios = ['none', 'specular', 'diffuse']
        self.platform_jitter = [0, 0.1, 0.5, 1.0]  # Antenna motion (deg)
    
    def __call__(self, iq_clean):
        # Add noise
        snr_db = np.random.uniform(*self.noise_range)
        noise_power = 10 ** (-snr_db / 10)
        iq_noisy = iq_clean + np.random.randn(*iq_clean.shape) * np.sqrt(noise_power)
        
        # Add clutter
        if np.random.random() < 0.5:
            clutter = self.generate_clutter()
            iq_noisy += clutter * np.random.choice(self.clutter_levels)
        
        # Add jamming
        if np.random.random() < 0.2:
            jamming = self.generate_jamming(np.random.choice(self.jamming_types))
            iq_noisy += jamming
        
        # Platform jitter (phase noise)
        jitter_deg = np.random.choice(self.platform_jitter)
        phase_noise = np.exp(1j * np.random.randn(*iq_noisy.shape[:2]) * np.radians(jitter_deg))
        iq_jittered = iq_noisy * phase_noise[..., None, None]
        
        return iq_jittered
```

### Real-Time Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATEENA REAL-TIME PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

  ATeena Hardware
       │
       ▼ Raw IQ Stream [128 pulses × 1024 range × 64 elements × 2]
┌───────────────────────────────────────────────────────────────────────────────┐
│  FPGA Pre-processing (Optional - for bandwidth reduction)                       │
│  • Digital beamforming (32 beams)                                                │
│  • Decimation: 1024 → 256 range bins                                            │
│  • 16-bit → 8-bit quantization                                                  │
└───────────────────────────────────────────────────────────────────────────────┘
       │
       ▼ Reduced IQ [128 × 256 × 32 × 2] = 2 MB/CPI
┌───────────────────────────────────────────────────────────────────────────────┐
│  ATeena-IQNet Inference (ONNX Runtime)                                        │
│  • Input preprocessing: normalization, windowing                              │
│  • Neural inference: ~20ms on T4 GPU                                            │
│  • Output: Up to 50 detections per CPI                                          │
└───────────────────────────────────────────────────────────────────────────────┘
       │
       ▼ Detections [range, doppler, azimuth, elevation, rcs, confidence]
┌───────────────────────────────────────────────────────────────────────────────┐
│  Track Initiation & Maintenance (JPDA-M/NNJPDA)                                │
│  • Association: Gating + likelihood ratio                                       │
│  • Filter: UKF with neural-ODE prediction                                       │
│  • Track quality: M/N logic + trajectory coherence                            │
└───────────────────────────────────────────────────────────────────────────────┘
       │
       ▼ Confirmed Tracks → gRPC → Engine Runtime
```

### Implementation Milestones

| Week | Task | Deliverable |
|:---|:---|:---|
| 5.1-5.2 | Neural architecture | Complete ATeena-IQNet PyTorch implementation |
| 5.3-5.4 | Synthetic training | Train on 100k CPIs with domain randomization |
| 6.1 | ONNX export | Traceable model with dynamic shapes |
| 6.2 | Validation | P_d > 0.90 at 0 dB SNR benchmark |
| 6.3 | FPGA integration spec | Pre-processing requirements for ATeena hardware |
| 6.4 | End-to-end test | Simulated ATeena → IQNet → Tracks → Engine |
| 7.1-7.2 | Real data collection | ISRO/SAC coordination for ATeena access |
| 7.3-7.4 | Transfer learning | Fine-tuning on real IQ data |
| 8.1-8.4 | Production hardening | Latency optimization, error handling, monitoring |

### Success Criteria
- **Detection performance**: P_d > 0.90 at SNR = 0 dB, P_fa < 1e-6 per CPI
- **Localization accuracy**: σ_range < 10m, σ_angle < 0.1° at SNR > 10 dB
- **Throughput**: 10 CPIs/sec per GPU (real-time for 1 Hz update)
- **Latency**: <50ms end-to-end (IQ input → track output)

---

## Workstream 3: NavIC Ionospheric Module (Gap 2 - Reframed)

### Technical Specification

**Revised Scope**: Ionospheric correction + timing anomaly detection (NOT bistatic radar)

**Architecture**:
```python
class NavICProcessor:
    """
    Process NavIC dual-frequency observations for SSA.
    
    Functions:
    1. STEC calculation from L5 + S-band pseudoranges
    2. Ionospheric correction for radar observations
    3. Timing anomaly detection (spoofing/jamming)
    """
    
    def __init__(self):
        self.f_l5 = 1176.45e6  # Hz
        self.f_s = 2492.028e6  # Hz
        self.lstm_anomaly = NavICAnomalyDetector()
    
    def compute_stec(self, p_l5, p_s):
        """
        Compute Slant Total Electron Content.
        
        STEC = (f₁² · f₂²) / (40.3 · (f₁² - f₂²)) · (P₂ - P₁)
        
        Units: TECU (Total Electron Content Units)
        1 TECU = 10^16 electrons/m²
        """
        f1_sq = self.f_l5 ** 2
        f2_sq = self.f_s ** 2
        
        stec = (f1_sq * f2_sq) / (40.3 * (f1_sq - f2_sq)) * (p_s - p_l5)
        
        # Convert to TECU
        stec_tecu = stec / 1e16
        
        return stec_tecu
    
    def correct_range(self, raw_range_km, elevation_deg, stec_tecu):
        """
        Apply ionospheric range correction.
        
        Delay = 40.3 · STEC / f²
        """
        # Mapping function (simplified, use more accurate for production)
        # M(elev) ≈ 1/sin(elev) for slant to vertical conversion
        mapping = 1.0 / np.sin(np.radians(max(elevation_deg, 5)))
        
        # Vertical TEC
        vtec = stec_tecu / mapping
        
        # Range correction (positive = subtract from measured range)
        delay_m = 40.3 * stec_tecu * 1e16 / (self.f_l5 ** 2)
        correction_km = delay_m / 1000
        
        return raw_range_km - correction_km


class NavICAnomalyDetector(nn.Module):
    """
    LSTM-based detection of NavIC timing anomalies.
    Detects: Spoofing, jamming, receiver faults, space weather events.
    """
    
    def __init__(self, n_sats=7, hidden=128):
        super().__init__()
        
        # Per-satellite feature embedding
        self.sat_embed = nn.Linear(4, 32)  # 4 features per sat
        
        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=32 * n_sats,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Multi-head attention for anomaly localization
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden * 2,
            num_heads=4,
            dropout=0.1
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)  # [normal, ionospheric_disturbance, spoofing, jamming]
        )
    
    def forward(self, navic_history):
        """
        navic_history: [batch, time_steps, n_sats, 4]
            4 features: [pseudorange_residual, carrier_phase, cn0, elevation]
        """
        batch_size, seq_len, n_sats, n_feat = navic_history.shape
        
        # Flatten and embed
        x = navic_history.view(batch_size * seq_len * n_sats, n_feat)
        x = F.relu(self.sat_embed(x))
        x = x.view(batch_size, seq_len, n_sats * 32)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)  # [batch, hidden*2]
        
        # Classification
        logits = self.classifier(pooled)
        
        return F.softmax(logits, dim=-1)
```

### Data Sources

| Source | Data | Access |
|:---|:---|:---|
| ISRO/SAC | NavIC RINEX observation files | MoU required |
| IGS | Multi-GNSS (including IRNSS) observation data | Open access |
| CDDIS | NASA archive of GNSS data | Open access |

### Implementation Milestones

| Week | Task | Deliverable |
|:---|:---|:---|
| 6.1 | RINEX parser | Read NavIC L5/S-band observations |
| 6.2 | STEC calculation | Dual-frequency TEC with validation |
| 6.3 | Chapman model | Ionospheric profile fitting |
| 6.4 | Range correction | Integration with radar processing |
| 7.1 | Anomaly dataset | Collect historical anomalies |
| 7.2 | LSTM training | Train anomaly detector |
| 7.3 | Real-time pipeline | NavIC → TEC → Correction → UKF |
| 7.4 | Validation | <5 TECU accuracy vs IGS products |

---

## Workstream 4: Production Hardening & Beta Deployment

### System Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 DEPLOYMENT ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐  │
  │                     KUBERNETES / ON-PREMISE CLUSTER                      │  │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
  │  │   ATeena    │  │   Engine    │  │   Fusion    │  │   Neural    │   │  │
  │  │   IQNet     │  │   Runtime   │  │    GNN      │  │    ODE      │   │  │
  │  │  (GPU Pod)  │  │  (CPU Pod)  │  │  (GPU Pod)  │  │  (GPU Pod)  │   │  │
  │  │   ONNX RT   │  │   gRPC      │  │   ONNX RT   │  │   ONNX RT   │   │  │
  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │  │
  │         │                │                │                │         │  │
  │         └────────────────┴────────────────┴────────────────┘         │  │
  │                              │                                      │  │
  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
  │  │                     REDIS / NATS QUEUE                         │ │  │
  │  │         (Inter-service messaging & state management)           │ │  │
  │  └─────────────────────────────────────────────────────────────────┘ │  │
  │                              │                                      │  │
  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
  │  │                     PROMETHEUS + GRAFANA                       │ │  │
  │  │           (Metrics, alerting, observability)                   │ │  │
  │  └─────────────────────────────────────────────────────────────────┘ │  │
  └─────────────────────────────────────────────────────────────────────────┘  │
                                       │                                       │
                                       ▼                                       │
  ┌─────────────────────────────────────────────────────────────────────────┐  │
  │                     OPERATOR DASHBOARD (React/WebGL)                    │  │
  │  • Real-time 3D orbital visualization (Cesium.js)                      │  │
  │  • Alert management interface                                          │  │
  │  • SHAP attribution explainability                                     │  │
  │  • Manual track correlation tools                                      │  │
  └─────────────────────────────────────────────────────────────────────────┘  │
```

### Monitoring & Observability

**Key Metrics**:
```yaml
# System Health
system_uptime_percent: >99.5
observation_ingestion_rate: 1000  # obs/sec sustained
end_to_end_latency_ms:
  p50: <30
  p95: <50
  p99: <100

# Model Performance
neural_ode_position_error_m: <50  # 7-day prediction
ateena_detection_probability: >0.90  # at 0 dB SNR
ateena_false_alarm_rate: <1e-6  # per CPI
fusion_gnn_inference_ms: <50  # 1000-node graph

# Data Quality
track_update_rate_hz: 1.0
catalog_completeness_percent: >95  # for LEO >10cm
conjunction_screening_period_s: 30
```

**Alerting Thresholds**:
- Critical: End-to-end latency >200ms for 60s
- High: Any model returning NaN/confidence collapse
- Medium: Queue depth >1000 for 5min
- Low: Single sensor dropout >5min

### Implementation Milestones

| Week | Task | Deliverable |
|:---|:---|:---|
| 7.1 | Containerization | Docker images for all services |
| 7.2 | Orchestration | Kubernetes manifests, autoscaling |
| 7.3 | Monitoring | Prometheus metrics, Grafana dashboards |
| 7.4 | Load testing | 2x sustained throughput validation |
| 8.1 | Security audit | Penetration testing, mTLS verification |
| 8.2 | Disaster recovery | Backup/restore procedures |
| 8.3 | Documentation | Runbooks, API docs, operator guides |
| 8.4 | Beta handover | Deploy to IAF staging environment |

---

## Resource Requirements

### Personnel (Phase 2)

| Role | Count | Months | Focus Area |
|:---|:---|:---|:---|
| ML Engineer (GNN) | 1 | 4 | Fusion GNN architecture |
| ML Engineer (Radar) | 1 | 4 | ATeena-IQNet deployment |
| Signal Processing Engineer | 1 | 3 | NavIC integration |
| Go Engineer (Backend) | 2 | 4 | Production hardening |
| DevOps Engineer | 1 | 2 | Kubernetes, monitoring |
| QA/Validation Engineer | 1 | 2 | System testing |

**Total**: 7 engineers × 4 months = 28 person-months

### Infrastructure

| Resource | Quantity | Duration | Cost Estimate |
|:---|:---|:---|:---|
| GPU Nodes (V100/A10) | 4 | 4 months | ₹40L |
| CPU Nodes | 8 | 4 months | ₹10L |
| Storage (SSD) | 50 TB | 4 months | ₹5L |
| Network (10 Gbps) | 1 | 4 months | ₹5L |
| **Total Infrastructure** | | | **₹60L** |

### External Dependencies

| Dependency | Risk Level | Mitigation |
|:---|:---|:---|
| ISRO ATeena data access | Medium | Synthetic-first strategy, MoU in progress |
| NavIC RINEX from SAC | Low | IGS backup data source available |
| RISAT SAR data | High | Phase 3 dependency, not blocking Phase 2 |

---

## Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|:---|:---|:---|:---|
| Synthetic-to-real domain gap (ATeena) | Medium | High | Heavy domain randomization, uncertainty quantification |
| GNN inference latency >50ms | Low | Medium | Model distillation, early exit mechanisms |
| NavIC data quality issues | Low | Low | IGS backup, multi-GNSS fusion |
| GPU availability constraints | Medium | Medium | CPU fallback for inference, quantization |
| Integration complexity delays | Medium | Medium | CI/CD automation, integration testing from Week 1 |

---

## Success Criteria (Phase 2 Exit Gates)

### Technical Gates

| Gate | Criteria | Verification |
|:---|:---|:---|
| **G2.1** | Fusion GNN inference <50ms P95 | Load test with 1000-node graphs |
| **G2.2** | ATeena P_d > 0.90 at 0 dB SNR | ROC analysis on held-out test set |
| **G2.3** | NavIC STEC <5 TECU error | Comparison with IGS final products |
| **G2.4** | System uptime >99.5% | 30-day continuous operation test |
| **G2.5** | 1000 obs/sec sustained throughput | 24-hour load test with production mix |

### Operational Gates

| Gate | Criteria | Verification |
|:---|:---|:---|
| **G2.6** | Security audit passed | Third-party penetration test report |
| **G2.7** | Operator training complete | Training completion certificates |
| **G2.8** | Runbook documentation complete | Document review sign-off |

---

## Phase 2 → Phase 3 Transition

**Phase 3 Preview**: Advanced Capabilities (Months 9-12)
- SAR-OrbitNet integration (pending ISRO MoU)
- BFTL distributed consensus (multi-site deployment)
- Maneuver prediction with neural ODE
- Real-time conjunction probability refinement

**Go/No-Go Decision**: End of Month 8
- **Go**: All G2.1-G2.8 gates passed → Begin Phase 3
- **No-Go**: Address blockers, extend Phase 2

---

## Conclusion

Phase 2 transforms the Phase 1 MVP into a **production-ready multi-sensor SSA system**. The key deliverables are:

1. **Fusion GNN**: Unified multi-sensor processing with explainable AI
2. **ATeena-IQNet**: Real-time neural radar detection with synthetic-to-real transfer
3. **NavIC Module**: Ionospheric correction and anomaly detection
4. **Beta System**: 99.5% uptime, fully monitored, operator-ready

**Expected Outcome**: By Month 8, the system will process 1000 observations/second from multiple heterogeneous sensors, detect conjunctions with >95% precision, and provide the foundation for full operational deployment in Phase 3.
