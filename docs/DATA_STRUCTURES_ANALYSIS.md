# SSIS Data Structures Analysis (Comprehensive)

**Document Date**: April 10, 2026  
**Scope**: Full Go monorepo + Python ML pipeline  
**Purpose**: Complete reference for data types, interfaces, and flow across the satellite conjunction & SSA operations system

---

## Executive Summary

The SSIS codebase uses a **hybrid physics-AI architecture** with these core principles:

1. **Classical foundations**: Orbital mechanics (TLE, propagation, UKF fusion)
2. **ML augmentation**: ONNX models for maneuver detection, collision risk, threat ranking
3. **Confidence-gated fallback**: If ML confidence is low, classical CUSUM/heuristics take over
4. **Type-safe passing**: Structs encode domain concepts (threat classes, maneuver types, assessment confidence)
5. **Cryptographic integrity**: All operational alerts are signed and verifiable

---

## Part 1: Go Core Data Structures

### 1.1 Catalog & Object Tracking (`pkg/catalog/`)

**`ObjectRecord`** — Represents a tracked object (satellite, debris, or asset)
```go
type ObjectRecord struct {
    ID                int            // Unique ID
    COSPARID          string         // Standard international ID (e.g., "1998-067A")
    Name              string         // Common name
    CountryCode       string         // Owner/Operator (ISO-3166)
    Classification    string         // Security/Functional
    OperationalStatus string         // Active/Debris/Zombies
    Pos               octree.Vector3 // Position in ECI (km)
    PosSigmaKm        float64        // 1-sigma uncertainty (km)
    HardBodyRadiusM   float64        // Required for Alfano Pc
    MassKg            float64        // Required for maneuver verification
    LastUpdateUnix    int64          // Timestamp
    Version           int            // Version counter
}
```

**`TrackObservation`** — Single sensor measurement (radar, TLE, or fused)
```go
type TrackObservation struct {
    TrackID       int              // Reference to tracked object
    Pos           octree.Vector3   // Observed position
    SigmaKm       float64          // Single-sensor uncertainty
    RadarPos      *octree.Vector3  // Optional radar-specific measurement
    RadarSigmaKm  float64          // Radar measurement variance
    TLEPos        *octree.Vector3  // Optional TLE-derived position
    TLESigmaKm    float64          // TLE variance
    TimestampUnix int64            // Sensor measurement time
}
```

**`FusedTrackObservation`** — Post-fusion multi-sensor observation
```go
type FusedTrackObservation struct {
    TrackID       int            // Object ID
    Pos           octree.Vector3 // Fused ECI position
    SigmaKm       float64        // Fused covariance (1-sigma)
    SensorQuality float64        // Quality metric [0,1]; blends radar + TLE reliability
    TimestampUnix int64          // Fusion timestamp
}
```

**`TrackAssociationCandidate`** — Candidate match for data association
```go
type TrackAssociationCandidate struct {
    Record            ObjectRecord // Candidate object from catalog
    DistanceKm        float64      // Geometric distance to new observation (km)
    AssociationScore  float64      // Blended score (geometric + ML)
    Confidence        float64      // Overall confidence [0, 1]
    Attention         float64      // ML attention weight for explainability
}
```

**`TrackAssociationInference`** — ML model output for data association
```go
type TrackAssociationInference struct {
    Scores     map[int]float64 // Per-candidate association scores
    Confidence float64         // Model confidence [0, 1]
    Attention  map[int]float64 // Attention weights for SHAP explainability
}
```

**`HybridAssociationEngine`** — Orchestrator for ML + geometric association
```go
type HybridAssociationEngine struct {
    Model              TrackAssociationModel  // ONNX Transformer model
    MinModelConfidence float64                // Confidence gate (default 0.55)
    ModelBlendWeight   float64                // Weight for ML vs. geometric (default 0.35)
    MaxScoreDelta      float64                // Max allowed score deviation (default 0.75)
}
```

**Interface: `TrackAssociationModel`**
```go
type TrackAssociationModel interface {
    Name() string
    Associate(obs FusedTrackObservation, 
              candidates []ObjectRecord) (TrackAssociationInference, error)
}
```

---

### 1.2 Orbital Mechanics (`pkg/orbital/`)

#### **TLE (Two-Line Element Set)**

```go
type TLE struct {
    Name               string  // Satellite name
    SatelliteNumber    int     // NORAD catalog number
    Classification     string  // Security classification (U/C/S)
    InternationalDesig string  // International designation (e.g., 2020-001A)
    EpochYear          int     // Year of epoch
    EpochDay           float64 // Day of year (fractional)
    MeanMotionDeriv1   float64 // First derivative of mean motion (rev/day²)
    MeanMotionDeriv2   float64 // Second derivative (divided by 6)
    BStar              float64 // B* drag term (NORAD propagation model)
    ElementNumber      int     // Element set number
    Inclination        float64 // Orbital inclination (radians)
    RAAN               float64 // Right Ascension of Ascending Node (radians)
    Eccentricity       float64 // Eccentricity [0, 1)
    ArgOfPerigee       float64 // Argument of perigee (radians)
    MeanAnomaly        float64 // Mean anomaly at epoch (radians)
    MeanMotion         float64 // Mean motion (revolutions/day)
    RevAtEpoch         int     // Revolution number at epoch
}
```

**Parser**: `ParseTLE(data string) (*TLE, error)` — Handles 2-line and 3-line formats per NORAD standard

---

#### **Maneuver Detection**

**`ManeuverEvent`** — Detected or inferred maneuver
```go
type ManeuverEvent struct {
    ObjectID         int            // Satellite ID
    TimeIndex        int            // Time step in propagation
    Score            float64        // Detection score [0, 1]
    DeltaV           float64        // Velocity change magnitude (km/s)
    Type             ManeuverType   // enum: none | impulsive | low_thrust | hypersonic_like
    DeltaVRSW        [3]float64     // Delta-V in RSW frame (radial, along-track, cross-track)
    ManeuverTimeSec  float64        // Duration of burn (seconds)
    IntentProxy      bool           // Analyst assessment of intent (e.g., debris avoidance)
    IntentProxyScore float64        // Confidence in intent [0, 1]
}
```

**`ManeuverFrame`** — Minimal temporal feature for ONNX inference
```go
type ManeuverFrame struct {
    Residual float64 // Norm residual from propagation (km)
    DeltaV   float64 // Estimated delta-V (km/s)
}
```

**`ManeuverInference`** — ML model output (ONNX Runtime)
```go
type ManeuverInference struct {
    Probability float64                // P(maneuver = true) [0, 1]
    Type        ManeuverType           // Predicted class: none | impulsive | low_thrust | hypersonic_like
    DeltaV      float64                // Estimated delta-V (km/s)
    Confidence  float64                // Model confidence [0, 1]
    SHAP        []FeatureContribution  // Per-feature contribution (explainability)
    ManeuverDeltaVRSW [3]float64       // Delta-V in RSW (ONNX output)
    ManeuverTime      float64          // Burn duration (ONNX output)
}
```

**`ManeuverCharacterization`** — Post-event characterization output
```go
type ManeuverCharacterization struct {
    ManeuverTimeSec  float64        // Burn start time (seconds since epoch)
    DeltaVECI        [3]float64     // Delta-V in ECI frame (km/s)
    DeltaVRSW        [3]float64     // Delta-V in RSW frame (km/s)
    BurnType         ManeuverType   // Classified type
    IntentProxy      bool           // High-confidence intent assessment
    IntentProxyScore float64        // Intent confidence
    Confidence       float64        // Overall characterization confidence
}
```

**`FeatureContribution`** — SHAP-based explainability
```go
type FeatureContribution struct {
    Feature      string  // Input feature name (e.g., "residual", "deltaV")
    Value        float64 // Actual value from input
    Contribution float64 // SHAP contribution to output
}
```

**`HybridManeuverDetector`** — ML + CUSUM fallback orchestration
```go
type HybridManeuverDetector struct {
    Model               ManeuverInferenceModel // ONNX model interface
    Fallback            *CUSUMDetector         // Classical detector
    WindowSize          int                    // Sequence length for ONNX (default 12)
    MinProbability      float64                // Gate on P(maneuver) (default 0.60)
    MinConfidence       float64                // Gate on model confidence (default 0.50)
    SuppressAfterDetect int                    // Suppress detections N steps after event
}
```

**`CUSUMDetector`** — Classical change-point detector (deterministic fallback)
```go
type CUSUMDetector struct {
    drift     float64 // Drift threshold (default 0.005)
    threshold float64 // Alert threshold (default 0.05)
    posSum    float64 // Positive CUSUM accumulator
    negSum    float64 // Negative CUSUM accumulator
}

func (d *CUSUMDetector) Update(residual float64) (score float64, alertTriggered bool)
func (d *CUSUMDetector) Reset()
```

**Interface: `ManeuverInferenceModel`**
```go
type ManeuverInferenceModel interface {
    Name() string
    Predict(window []ManeuverFrame) (ManeuverInference, error)
}
```

---

#### **Orbital State Fusion (UKF)**

**`UKFConfig`** — Unscented Kalman Filter parameters
```go
type UKFConfig struct {
    Alpha   float64 // Spread of sigma points (1e-3 to 1e-0)
    Beta    float64 // Heavy-tails correction (2.0 for Gaussian)
    Kappa   float64 // Secondary scaling parameter (3 - n_state)
    QCov    [6][6]float64 // Process noise covariance
    RCov    [3][3]float64 // Measurement noise covariance
}
```

**`RadarMeasurement`** — Radar observation
```go
type RadarMeasurement struct {
    SiteID    int
    RangeKm   float64         // Slant range (km)
    AzimuthRd float64         // Azimuth (radians)
    ElevationRd float64       // Elevation (radians)
    RangeRateKmS float64      // Range rate (km/s)
    SigmaRangeKm float64      // Measurement uncertainty (km)
    TimestampUnix int64
}
```

**`TLEMeasurement`** — Propagated TLE position
```go
type TLEMeasurement struct {
    PosECI     [3]float64     // Position in ECI (km)
    VelECI     [3]float64     // Velocity in ECI (km/s)
    SigmaKm    float64        // Propagation uncertainty (km)
    TimestampUnix int64
}
```

**`FusionInput`** — Container for all measurements
```go
type FusionInput struct {
    ObjectID          int
    RadarMeasurements []RadarMeasurement
    TLEMeasurements   []TLEMeasurement
    PriorState        [6]float64 // ECI state vector [x, y, z, vx, vy, vz]
    PriorCovariance   [6][6]float64
}
```

**`FusionResult`** — Post-filter state and uncertainty
```go
type FusionResult struct {
    State          [6]float64     // Fused [x, y, z, vx, vy, vz] ECI
    Covariance     [6][6]float64  // Fused covariance
    Confidence     float64        // Filter health [0, 1]
    UncertaintyAlert *UncertaintyAlert
}
```

**`UncertaintyAlert`** — Raised when uncertainty exceeds threshold
```go
type UncertaintyAlert struct {
    ObjectID            int
    PosUncertaintyKm    float64
    VelUncertaintySigma float64
    AlarmLevel          string // "info" | "warning" | "critical"
    RecommendedAction   string
}
```

---

### 1.3 Physics & Threat Assessment (`pkg/physics/`)

#### **High-Fidelity Collision Assessment (Alfano Pc)**

**`ConjunctionEvent`** — Represents a close approach between two objects
```go
type ConjunctionEvent struct {
    PrimaryID       int             // Target asset ID
    SecondaryID     int             // Potential perturber ID
    TCA             int64           // Time of Closest Approach (Unix)
    RelPosition     octree.Vector3  // Relative position at TCA (km)
    RelVelocity     octree.Vector3  // Relative velocity at TCA (km/s)
    MissDistanceKm  float64         // Geometric miss distance
    CombinedCov     Covariance6x6   // Combined error distribution
}
```

**`Covariance6x6` and `Covariance3x3`** — Matrix representations
```go
type Covariance6x6 [6][6]float64 // Full state covariance
type Covariance3x3 [3][3]float64 // Position or velocity only
```

**`RiskAssessment`** — Calculated collision probability
```go
type RiskAssessment struct {
    AlfanoPc           float64 // Probability using Alfano (series expansion)
    MaxPc              float64 // Maximum possible probability for the geometry
    ConfidenceInterval [2]float64 // [lower, upper] risk bounds
    Method             string  // "alfano_series" | "bnn_monte_carlo"
}
```

#### **RAPID (Real-time Assessment of Projectile Impact Dynamics)**

**`RapidTrack`** — Input feature vector for threat ranking
```go
type RapidTrack struct {
    ID                  int     // Track identifier
    Mach                float64 // Mach number
    AltitudeKm          float64 // Altitude (km)
    BurnoutVelocityKmS  float64 // Burnout velocity (km/s)
    FlightPathAngleRad  float64 // Flight path angle (radians)
    LaunchLat           float64 // Launch latitude (degrees)
    LaunchLon           float64 // Launch longitude (degrees)
    DistanceToAssetKm   float64 // Distance to protected asset (km)
    RadarCrossSection   float64 // RCS (m²)
    ManeuverIntentProxy bool    // Intent signal from maneuver detector
    ManeuverIntentScore float64 // Intent confidence
}
```

**`RapidAssessment`** — Full threat assessment output
```go
type RapidAssessment struct {
    ID              int                       // Track ID
    Class           ThreatClass               // enum: orbital | ballistic | hypersonic | unknown
    Priority        int                       // Priority level [0, 3]
    Reachable       bool                      // Can threat reach asset?
    DynamicScore    float64                   // Composite threat score [0, 1]
    ScoreConfidence float64                   // Confidence in score
    UncertaintyLow  float64                   // Lower bound score
    UncertaintyHigh float64                   // Upper bound score
    Recommendation  EvasiveRecommendation     // Recommended action + confidence
    FeatureHeatmap  []RapidFeatureContribution // Top K feature contributions
    Ballistic       BallisticThreat           // Sub-module outputs
    Reentry         ReentryState              // Sub-module outputs
}
```

**`RapidRankInference`** — LightGBM ranker model output
```go
type RapidRankInference struct {
    Score           float64            // Threat priority score [0, 1]
    Confidence      float64            // Model confidence
    UncertaintyLow  float64            // Calibrated lower percentile
    UncertaintyHigh float64            // Calibrated upper percentile
    FeatureWeights  map[string]float64 // Per-feature importance
}
```

**`EvasiveAction`** — Recommended defensive response
```go
type EvasiveAction string

const (
    ActionMonitorOnly     EvasiveAction = "monitor_only"
    ActionSensorRetask    EvasiveAction = "sensor_retask"
    ActionPrepareEvasive  EvasiveAction = "prepare_evasive"
    ActionEvasiveNow      EvasiveAction = "evasive_maneuver_now"
    ActionOutOfReachWatch EvasiveAction = "out_of_reach_watch"
)
```

**`EvasiveRecommendation`** — Action + reasoning
```go
type EvasiveRecommendation struct {
    Action     EvasiveAction // Recommended action
    Confidence float64       // Confidence [0, 1]
    Reason     string        // Explanation (e.g., "ballistic on collision course within 45 min")
}
```

**`HybridRapidThreatEngine`** — Orchestrates LightGBM ranker + policy model
```go
type HybridRapidThreatEngine struct {
    Ranker                RapidPriorityModel   // LightGBM ONNX model
    Policy                RapidPolicyModel     // Policy enforcement model
    MinRankerConfidence   float64              // ML confidence gate (default 0.55)
    MaxRankerDeviationPct float64              // Max deviation from baseline (default 0.60)
}
```

**Interface: `RapidPriorityModel`**
```go
type RapidPriorityModel interface {
    Name() string
    Rank(track RapidTrack, class ThreatClass, 
         reachable bool, baselinePriority int) (RapidRankInference, error)
}
```

**Interface: `RapidPolicyModel`**
```go
type RapidPolicyModel interface {
    Name() string
    Recommend(track RapidTrack, class ThreatClass, 
              score float64, reachable bool) (RapidPolicyInference, error)
}
```

---

### 1.4 ML Inference & Hybrid Orchestration (`pkg/ml/`)

#### **ONNX Runtime Interface**

**`SessionConfig`** — ONNX inference session configuration
```go
type SessionConfig struct {
    ModelPath         string  // Path to .onnx file
    ModelID           string  // Semantic ID (e.g., "maneuver_detector")
    ModelVersion      string  // Version string (e.g., "1.0.0")
    MaxBatchSize      int     // Max batch size (default 128)
    CacheSize         int     // Inference cache size (default 512)
    EnableLogging     bool    // Enable latency metrics
    IntraOpNumThreads int     // Intra-op parallelism
    InterOpNumThreads int     // Inter-op parallelism
    ExecutionProvider string  // "CPU" or "CUDA"
}
```

**`InferenceRequest`** — Single ONNX inference request
```go
type InferenceRequest struct {
    ID       string                 // Unique request ID
    ModelID  string                 // Model identifier
    Version  string                 // Model version
    Inputs   map[string]interface{} // Input tensors (keyed by name)
    Metadata map[string]string      // Optional metadata
}
```

**`InferenceResult`** — ONNX model output + metadata
```go
type InferenceResult struct {
    RequestID  string                 // Matches request ID
    Outputs    map[string]interface{} // Output tensors
    Confidence float64                // Overall confidence [0, 1]
    LatencyMs  float64                // Inference time (ms)
    Error      error                  // Inference error
    ModelID    string                 // Which model ran
    Version    string                 // Model version
}
```

**`LatencyMetrics`** — Inference performance tracking
```go
type LatencyMetrics struct {
    MinMs   float64
    MaxMs   float64
    MeanMs  float64
    P50Ms   float64
    P95Ms   float64
    P99Ms   float64
    SampleCount int64
}
```

**Interface: `Inferrer`**
```go
type Inferrer interface {
    InferBatch(requests []InferenceRequest) []InferenceResult
    Close() error
    ModelInfo() ModelInfo
}
```

#### **Hybrid Orchestrator**

**`Thresholds`** — Confidence gates for ML acceptance
```go
type Thresholds struct {
    ManeuverProbability   float64 // P(maneuver) gate (default 0.60)
    ManeuverConfidence    float64 // Engine confidence gate (default 0.50)
    CollisionConfidence   float64 // Collision model confidence (default 0.55)
    RapidConfidence       float64 // RAPID ranker confidence (default 0.55)
    MaxCollisionDeviation float64 // Max score deviation (default 0.35)
    MaxRapidDeviationPct  float64 // Max % deviation (default 0.60)
}
```

**`Orchestrator`** — Coordinates classical → ML → fallback pipeline
```go
type Orchestrator struct {
    Engine     ml.InferenceEngine       // ONNX model container
    Thresholds Thresholds               // Confidence gates
}
```

**Decision Tracking Structs** — Explain which subsystem was used
```go
type ManeuverDecision struct {
    TimeIndex int    // Time step
    Source    string // "classical" | "ml" | "hybrid"
    Reason    string // Explanation
}

type CollisionDecision struct {
    Source string // "classical" | "ml"
    Reason string
}

type RapidDecision struct {
    TrackID int
    Source  string // "classical" | "ml"
    Reason  string
}
```

---

### 1.5 Cryptographic Integrity (`pkg/crypto/`)

**`Signer`** — HMAC/RSA signed alert production
```go
type Signer struct {
    // Implementation uses RSA private key or HMAC secret
}
```

**`AuditEntry`** — Signed operational event
```go
type AuditEntry struct {
    EventID       string    // UUID
    TimestampUnix int64     // Event timestamp
    EventType     string    // "maneuver_detected" | "collision_alert" | "threat_ranked"
    Details       string    // JSON payload
    Signature     string    // Hex-encoded signature
    SignatureAlgo string    // "hmac_sha256" | "rsa_sha256"
}
```

**`SignedOperationalAlert`** — Full alert with cryptographic proof
```go
type SignedOperationalAlert struct {
    EventID           string  `json:"event_id"`
    TimestampUnix     int64   `json:"timestamp_unix"`
    AssetID           int     `json:"asset_id"`
    ThreatID          int     `json:"threat_id"`
    ClassicalStage    string  `json:"classical_stage"`      // classical output (e.g., CUSUM)
    AIPath            string  `json:"ai_path"`              // ML path (e.g., ONNX output)
    ModelConfidence   float64 `json:"model_confidence"`
    UncertaintyLower  float64 `json:"uncertainty_lower"`
    UncertaintyUpper  float64 `json:"uncertainty_upper"`
    ExplainabilityTag string  `json:"explainability_tag"`   // SHAP summary
    Signature         string  `json:"signature"`             // Hex-encoded
    SignatureValid    bool    `json:"signature_valid"`
}
```

---

### 1.6 Spatial Indexing (`pkg/spatial/`)

**`CellKey`** — 3D hash grid cell identifier
```go
type CellKey struct {
    X int
    Y int
    Z int
}
```

**`Point`** — Spatial object
```go
type Point struct {
    ID  int            // Object identifier
    Pos octree.Vector3 // 3D position
}
```

**`HashGrid`** — O(1) spatial range queries (no build time)
```go
type HashGrid struct {
    cellSize  float64                  // Grid granularity
    invCell   float64                  // 1.0 / cellSize
    cells     map[CellKey][]int        // Cell → object IDs
    positions map[int]octree.Vector3   // ID → position
}

func (g *HashGrid) QueryAABB(min, max octree.Vector3) []int
func (g *HashGrid) Upsert(p Point)
func (g *HashGrid) Rebuild(points []Point)
```

---

### 1.7 Octree & 3D Spatial (`pkg/octree/`)

**`Vector3`** — 3D floating-point vector (ECI coordinates)
```go
type Vector3 struct {
    X float64
    Y float64
    Z float64
}
```

Used throughout for positions, velocities, covariance centers, etc.

---

### 1.8 Parity-Check Metrics (`pkg/orbital/parity.go`)

**`ParityResult`** — Tracks parity between internal engine and gold-standard reference
```go
type ParityResult struct {
    ObjectID           int
    ReferencePropagator string  // e.g., "python-sgp4 v2.17"
    PositionMSE        float64 // Mean Squared Error in position
    VelocityMSE        float64 // Mean Squared Error in velocity
    L2NormResidual     float64 // Integrated residual norm
    MaxPosErrorKm      float64 // Peak divergence
    TimestampUnix      int64
    PassStatus         bool    // True if within tolerance (Gated)
}
```

---

## Part 2: Python ML Data Structures

### 2.1 Data Generators (`ai/training/generate_data.py`)

#### **RapidThreatDataGenerator**

Generates synthetic threat tracks (orbital debris, ballistic missiles, hypersonic)

**Output**: `pd.DataFrame` with columns:
```python
{
    'Mach': float,
    'AltitudeKm': float,
    'BurnoutVelocityKmS': float,
    'FlightPathAngleRad': float,
    'LaunchLat': float,
    'LaunchLon': float,
    'DistanceToAssetKm': float,
    'RadarCrossSection': float,
    'ThreatClass': int,  # 0=orbital, 1=ballistic, 2=hypersonic
    'IsReachable': int,  # 0 or 1
    'TimeToThreatMin': float,
    'PastManeuverCount': int,
    'LastManeuverAgeHours': float,
    'Priority': int,  # 0=low, 1=medium, 2=high, 3=critical (label)
}
```

#### **ManeuverDetectorDataGenerator**

Generates synthetic time-series scenarios with maneuver injection

**Key methods**:
- `generate_clean_residuals(length, base_noise)` → residual array [length]
- `inject_maneuver(residuals, deltaVs, time_index)` → modifies residuals in-place
- `generate_scenario(num_windows, window_size)` → (X_windows, y_label, y_deltaV)

**Output arrays** (for ONNX):
```python
X: np.ndarray  # [N_windows, window_size=12, features=2]
               # features: [residual (km), deltaV (km/s)]

y_label: np.ndarray  # [N], integer class {0=none, 1=impulsive, 2=low_thrust, 3=hypersonic}
y_deltaV: np.ndarray # [N], float scalar (km/s)
```

#### **CollisionRiskDataGenerator**

Generates conjunction event scenarios with risk assessment

**Output**: Structured data with position uncertainty, relative velocity, approach geometry

---

### 2.2 PyTorch Model Architectures

#### **ManeuverLSTMDetector** (`ai/training/train_maneuver_detector.py`)

Multi-task LSTM + attention model
```python
class ManeuverLSTMDetector(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, 
                 num_heads=4, dropout=0.2):
        # LSTM: [batch, seq_len, input_size] → [batch, seq_len, hidden_size]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, ...)
        
        # Multi-head attention over sequence
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, ...)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Multi-head outputs
        self.prob_head = nn.Linear(32, 1)        # P(maneuver)
        self.type_head = nn.Linear(32, 4)        # Maneuver class logits
        self.deltav_head = nn.Linear(32, 1)      # ΔV magnitude
        self.confidence_head = nn.Linear(32, 1)  # Confidence [0, 1]
    
    def forward(self, x):  # [batch, 12, 2]
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)
        # Dense + outputs
        return {
            'probability': scalar,
            'type': [4 logits],
            'deltaV': scalar,
            'confidence': scalar,
            'attention_weights': attention matrix
        }
```

**Training**: Cross-entropy for classification, L1 for regression, weighted multi-task loss

---

#### **CollisionRiskBNN** (`ai/training/train_collision_risk.py`)

Bayesian Neural Network for risk quantification
```python
class CollisionRiskBNN(nn.Module):
    # Input: relative state (Δr, Δv), HCM (hard-body model radius)
    # Output: collision probability, uncertainty bounds
```

---

#### **RapidThreatEnv** (`ai/rapid/train_policy_ppo.py`)

RL environment for evasive maneuver policy learning
```python
class RapidThreatEnv(gym.Env):
    def __init__(self):
        self.action_space = ...      # Evasive actions
        self.observation_space = ... # Extended RapidTrack features
    
    def step(self, action):
        return obs, reward, done, info
    
    def reset(self):
        return obs
```

---

#### **TCNForecaster** (`ai/coverage/train_tcn_forecaster.py`)

Temporal Convolutional Network for coverage prediction
```python
class TCNBlock(nn.Module):
    # Dilated causal convolution block
    
class TCNForecaster(nn.Module):
    # Stack of TCN blocks for blind-spot forecasting
```

---

#### **Transformer Association** (`ai/tracker/train_transformer_association.py`)

Transformer for data association (DeepSORT-like)
```python
class AssocDataset(Dataset):
    # Yields (observation, candidate_list, label) tuples

class AssocTransformer(nn.Module):
    # Encoder: observation features
    # Encoder: candidate object catalog features
    # Cross-attention: observe↔candidates
    # Output: association scores for each candidate
```

---

### 2.3 PyTorch to ONNX Export

**Standard ONNX export pattern** across all trainers:

```python
import torch
import torch.onnx

model = ManeuverLSTMDetector(...)
model.load_state_dict(torch.load('model.pt'))

dummy_input = torch.randn(1, 12, 2)  # batch_size=1, seq_len=12, features=2
torch.onnx.export(
    model, dummy_input,
    'maneuver_detector.onnx',
    input_names=['input'],
    output_names=['probability', 'type', 'deltaV', 'confidence'],
    opset_version=14,
    do_constant_folding=True,
    verbose=False
)
```

---

### 2.4 LightGBM Models

#### **RapidRanker** (`ai/rapid/train_ranker_lightgbm.py`)

LightGBM ranking model for threat priority
```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_priority, feature_names=feature_cols)
params = {
    'objective': 'rank_xendcg',
    'metric': 'ndcg',
    'num_leaves': 31,
    'learning_rate': 0.05,
}
model = lgb.train(params, train_data, num_boost_round=100,
                  callbacks=[lgb.early_stopping(50), lcb.log_evaluation(100)])

# ONNX export
import onnxmltools
onnx_model = onnxmltools.convert_lightgbm(model)
onnx.save_model(onnx_model, 'rapid_ranker.onnx')
```

**Input features**: RapidTrack fields (Mach, altitude, distance, RCS, etc.)  
**Outputs**: Threat priority score [0, 1], confidence, uncertainty bounds

---

### 2.5 Quantum Radar Sensing (`ai/sensing/quantum_radar.py`)

**`QuantumRadarSim`** — Parameters for characterizing quantum advantage in detection

| Parameter | Type | Description |
|-----------|------|-------------|
| `Kappa` | float | Transmissivity of the medium (attenuation) |
| `NB` | float | Background thermal photons (noise) |
| `NS` | float | Signal photons per pulse |
| `QIAdvantage` | float | Signal-to-Noise Ratio (SNR) gain of QI over classical |
| `SchmidtRank` | int | Entanglement dimension (characterization) |

---

## Part 3: Data Flow & Integration

### 3.1 End-to-End Inference Pipeline

```
[Observation Input]
    ↓
[Catalog/Association]
    ├─→ GeometricScoring (Euclidean distance)
    ├─→ ML Association (Transformer ONNX)
    └─→ HybridAssociationEngine → ObjectRecord
    ↓
[Propagation + Fusion]
    ├─→ TLE Propagator
    ├─→ RadarMeasurements
    └─→ UKF Fusion → FusedTrackObservation
    ↓
[Maneuver Detection]
    ├─→ CUSUM Detector (classical)
    ├─→ ML Maneuver LSTM (ONNX)
    └─→ HybridManeuverDetector → ManeuverEvent[]
    ↓
[Collision Risk Assessment]
    ├─→ Classical geometry
    ├─→ BNN Model (ONNX)
    └─→ Result: prob, uncertainty bounds
    ↓
[Threat Ranking (RAPID)]
    ├─→ Classify: ThreatClass (orbital/ballistic/hypersonic)
    ├─→ Ranker LightGBM (ONNX)
    ├─→ Policy Model (ONNX)
    └─→ HybridRapidThreatEngine → RapidAssessment
    ↓
[Explainability]
    ├─→ SHAP values (maneuver features)
    └─→ Attention weights (Transformer, LSTM)
    ↓
[Cryptographic Signing]
    └─→ SignedOperationalAlert (HMAC/RSA)
```

---

### 3.2 Key Interfaces (Dependency Injection)

```go
// ML backends plugged via interfaces
type ManeuverInferenceModel interface {
    Name() string
    Predict(window []ManeuverFrame) (ManeuverInference, error)
}

type TrackAssociationModel interface {
    Name() string
    Associate(obs FusedTrackObservation, candidates []ObjectRecord) (TrackAssociationInference, error)
}

type RapidPriorityModel interface {
    Name() string
    Rank(track RapidTrack, class ThreatClass, reachable bool, baselinePriority int) (RapidRankInference, error)
}

type RapidPolicyModel interface {
    Name() string
    Recommend(track RapidTrack, class ThreatClass, score float64, reachable bool) (RapidPolicyInference, error)
}

// Orchestrators compose these
type HybridManeuverDetector {
    Model = ManeuverInferenceModel
    Fallback = CUSUMDetector
}

type HybridAssociationEngine {
    Model = TrackAssociationModel
}

type HybridRapidThreatEngine {
    Ranker = RapidPriorityModel
    Policy = RapidPolicyModel
}
```

---

### 3.3 Operational Metrics & Delivery

**`OperationalRunMetrics`** — Captured after scale tests
```go
type OperationalRunMetrics struct {
    TotalRequests          int       // Total inference samples
    TotalAlerts            int       // Triggered alerts
    FallbackCount          int       // Classical fallback invocations
    FallbackRatio          float64   // % fallback to classical
    ThroughputReqPerSecond float64   // Inferences/sec
    LatencyP50Ms           float64   // Median latency
    LatencyP95Ms           float64   // 95th percentile
    LatencyP99Ms           float64   // 99th percentile
    CollisionAccuracy      float64   // AUC or F1
    CollisionFPR           float64   // False-positive rate
    ExplainabilityFidelity SHAPFidelity
    SignedAlertExample     SignedOperationalAlert
}
```

---

## Part 4: Type Hierarchy & Relationships

```
ObjectRecord
    ├── TrackObservation → FusedTrackObservation
    ├── TrackAssociationCandidate
    └── Used in: Catalog, Association

TLE + Propagator
    ├── → [6] state vector ECI
    ├── → RadarMeasurement (simulated)
    └── → TLEMeasurement → UKF → FusionResult

ManeuverEvent
    ├── Input: Residuals (propagation error)
    ├── Via: CUSUM (classical) / LSTM (ML)
    └── Output: ManeuverInference → ManeuverCharacterization

RapidTrack
    ├── Features: Mach, Altitude, RCS, Distance, Intent
    ├── Via: LightGBM Ranker + Policy Model
    └── Output: RapidAssessment → EvasiveRecommendation

SignedOperationalAlert
    └── Composite of all above with cryptographic proof

ConjunctionEvent
    ├── Input: Two ObjectRecords + Propagation
    ├── Via: Alfano Pc (series expansion)
    └── Output: RiskAssessment (AlfanoPc, MaxPc)

ParityResult
    ├── Reference: python-sgp4 v2.17
    ├── Metric: PositionMSE, VelocityMSE, L2NormResidual
    └── Usage: Engine validation & gating
```

---

## Part 5: Constants & Thresholds

### Maneuver Detection
- **Window size**: 12 time steps
- **Min probability**: 0.60 (ML gate)
- **Min confidence**: 0.50 (ML gate)
- **CUSUM drift**: 0.005
- **CUSUM threshold**: 0.05
- **Suppression**: 2 steps post-detection

### Collision Risk
- **Min confidence**: 0.55 (ML gate)
- **Max deviation**: 0.35 (from classical baseline)

### RAPID Threat Ranking
- **Min ranker confidence**: 0.55 (ML gate)
- **Max deviation**: 0.60% (from baseline)
- **Priority levels**: [0] Low, [1] Medium, [2] High, [3] Critical

### Orbital Mechanics
- **Epoch roll**: TLE year < 57 → 2000+epoch; else 1900+epoch
- **Hash grid cell size**: 25 km
- **Fused track sigma**: Blended from radar + TLE

---

## Part 6: Deployment Artifacts

### Generated ONNX Models

| Model | Source | File | Primary Output |
|-------|--------|------|-----------------|
| Maneuver Detector | ManeuverLSTMDetector (PyTorch) | `maneuver_detector.onnx` | probability, type [4], deltaV, confidence |
| Collision Risk | CollisionRiskBNN (PyTorch) | collision_risk.onnx | probability, uncertainty bounds |
| RAPID Ranker | LightGBM | rapid_ranker.onnx | threat priority score [0,1], confidence |

### Storage Locations

After Phase 2 build:
- `ssis/pkg/ml/models/` → Integrated Go package models
- `ssis/ai/models/` → AI team delivery checkpoint
- `ssis/ai/training/outputs/` → Latest training run artifacts

---

## Part 7: Testing & Validation

**High-level snapshot structures** for unit tests:
```go
// Mock implementations for testing
type mockAssociationModel struct{}
func (m *mockAssociationModel) Associate(...) (TrackAssociationInference, error)

type mockRapidRanker struct{}
func (m *mockRapidRanker) Rank(...) (RapidRankInference, error)

type mockRapidPolicy struct{}
func (m *mockRapidPolicy) Recommend(...) (RapidPolicyInference, error)
```

---

## Summary Table

| Domain | Key Struct | Role | Flow |
|--------|-----------|------|------|
| **Tracking** | ObjectRecord, FusedTrackObservation | Represent satellite positions & uncertainty | Catalog → Association → Propagation |
| **Orbital** | TLE, ManeuverEvent, ManeuverInference | Detect & characterize maneuvers | Propagation → CUSUM/LSTM → Hybrid |
| **Collision** | ConjunctionEvent, RiskAssessment, CollisionRiskBNN | Assess proximity risk (Classical + ML) | Relative geometry → Alfano/BNN → Risk |
| **Threat** | RapidTrack, RapidAssessment, RapidRankInference | Prioritize threats & recommend action | Features → LightGBM → Policy → Action |
| **Benchmark** | ParityResult | Validate engine accuracy vs SGP4 ref | Internal State vs python-sgp4 → MSE/L2 |
| **Crypto** | SignedOperationalAlert, AuditEntry | Prove alert authenticity | All outputs → HMAC/RSA → Signed |
| **ML** | ManeuverInferenceModel, TrackAssociationModel, RapidPriorityModel | Pluggable inference backends | ONNX Runtime → (confidence gated) → fallback |

---

**Document Version**: 1.0  
**Last Updated**: April 10, 2026  
**Maintainers**: CodeForge AI (SDE Mode)
