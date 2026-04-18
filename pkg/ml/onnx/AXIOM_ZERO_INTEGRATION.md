# Axiom-Zero Integration Guide

This document describes the integration of Axiom-Zero (JAX/Flax) physics models into SSIS via ONNX export.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Axiom-Zero Integration Stack                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Python/JAX (Research)                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  axiom_zero/                                                        │    │
│  │   ├── core_physics/rkan.py         - Rational KAN implementation   │    │
│  │   ├── core_physics/integrators.py  - Yoshida-4/6/8 integrators     │    │
│  │   └── sda/                                                          │    │
│  │        ├── conjunction.py          - Physics-informed Pc           │    │
│  │        ├── maneuver_detect.py      - Delta-rKAN attributor          │    │
│  │        ├── neighbourhood.py         - RRI threat model               │    │
│  │        ├── reentry.py               - Debris risk map                │    │
│  │        └── volumetric.py           - 4D probability field        │    │
│  │                                                                     │    │
│  │  bridge/export_onnx.py             - jax2tf→ONNX export pipeline    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼ jax2tf → SavedModel → tf2onnx               │
│                              │                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  models/                                                            │    │
│  │   ├── rkan_residual_covariance.onnx                                 │    │
│  │   ├── delta_rkan_attributor.onnx                                   │    │
│  │   └── pol_transformer.onnx                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Go/ONNX Runtime (Operational)                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  pkg/ml/onnx/                                                        │    │
│  │   ├── rkan_covariance_onnx.go      - ONNX wrapper + Inferrer        │    │
│  │   ├── rkan_covariance_stub.go      - Non-ONNX fallback              │    │
│  │   ├── delta_rkan_attributor_onnx.go - Maneuver detection wrapper    │    │
│  │   ├── delta_rkan_attributor_stub.go - Non-ONNX fallback            │    │
│  │   ├── pol_transformer_onnx.go      - POL analysis wrapper           │    │
│  │   ├── pol_transformer_stub.go      - Non-ONNX fallback              │    │
│  │   ├── axiom_zero_integration.go    - Unified engine interface         │    │
│  │   └── axiom_zero_integration_stub.go - Non-ONNX fallback            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  SSIS Operational Core                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │   pkg/collision/         - Physics-informed covariance inflation   │    │
│  │   pkg/orbital/           - Axiom-Zero propagator integration       │    │
│  │   pkg/catalog/           - POL-based UCT classification            │    │
│  │   pkg/server/grpc.go     - Physics engine selection (AXIOM/SGP4)   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Models

### 1. rKAN Residual Covariance Head

**Purpose:** Learned Hamiltonian residuals for physics-informed uncertainty quantification.

**Input:** `(batch, 13)` orbital features
- Position (3): X, Y, Z in ECI km
- Velocity (3): VX, VY, VZ in km/s
- Orbital elements (4): Semi-major axis, eccentricity, inclination, RAAN
- Environmental (3): F10.7 solar flux, Ap geomagnetic index, local time

**Output:** `(batch, 6)` diagonal covariance inflation
- Position covariance: XX, YY, ZZ in km²
- Velocity covariance: VXVX, VYVY, VZVZ in (km/s)²

**Usage:**
```go
engine, _ := onnx.NewAxiomZeroEngine(config)
covOutput, err := engine.GetPhysicsInformedCovariance(state, envFactors)
inflatedCov := engine.CovarianceModel.InflateCovariance(nominalCov, covOutput)
```

### 2. Delta-rKAN Maneuver Attributor

**Purpose:** Detect and attribute orbital maneuvers with symbolic LaTeX explanations.

**Input:** `(batch, timesteps, 6)` state history
- 20 timesteps of [x, y, z, vx, vy, vz]

**Output:**
- `maneuver_prob`: (batch, 1) probability of maneuver
- `delta_v`: (batch, 3) RTN frame components in m/s
- `maneuver_type`: (batch, 4) one-hot [none, station_keep, phasing, plane_change]

**Usage:**
```go
output, err := engine.DetectAndAttributeManeuver(objectID, stateHistory)
if output.ManeuverDetected {
    fmt.Println(output.LaTeXExplanation)
    // Output: "\Delta v_T = +15.3 \text{ m/s} \Rightarrow \Delta a \approx 4.1 \text{ km (raising)}"
}
```

### 3. POL Transformer

**Purpose:** Pattern-of-Life behavior classification for UCT analysis and anomaly detection.

**Architecture:** 6-layer Transformer, 256-dim, 4-head attention

**Input:** `(batch, seq_len, 6)` orbital state sequence
- 50 timesteps of [x, y, z, vx, vy, vz]

**Output:**
- `behavior_class`: (batch, 8) softmax over classes
- `anomaly_score`: (batch, 1) 0=normal, 1=anomalous
- `embedding`: (batch, 256) latent representation

**Classes:**
1. Unknown
2. LEO Communications (Starlink-like)
3. MEO Navigation (GPS-like)
4. GEO Stationary
5. HEO Molniya
6. LEO Imaging (CARTOSAT-like)
7. LEO Science
8. Debris

**Usage:**
```go
output, err := engine.ClassifyBehavior(objectID, stateSequence)
fmt.Printf("Class: %s (confidence: %.2f)\n", 
    output.BehaviorClass.String(), 
    output.ClassConfidence)
fmt.Printf("Anomaly score: %.3f\n", output.AnomalyScore)
```

## Build Configuration

### With ONNX Runtime
```bash
go build -tags onnx ./...
```

### Without ONNX Runtime (Air-Gap)
```bash
go build ./...
# Uses stub implementations that return default values
```

## Exporting Models

From the Python environment:
```bash
cd /Users/adi/idex_iaf/ai/active/v3/axiom_zero/bridge

# Export all models
python export_onnx.py --output-dir /Users/adi/idex_iaf/models

# Export specific models
python export_onnx.py --models rkan_cov delta_rkan --output-dir ./models
```

## Integration with SSIS

### Server Configuration

The gRPC server (`pkg/server/grpc.go`) supports engine selection via metadata:
```go
// Client sets header: x-ssis-physics-engine: AXIOM
physicsEngine := "SGP4"
if vals := md.Get("x-ssis-physics-engine"); len(vals) > 0 {
    physicsEngine = vals[0]
}

if physicsEngine == "AXIOM" && axiomEngine.IsEnabled() {
    // Use Axiom-Zero for propagation
    objects = orbital.ScanGroundRegionWithAxiom(roi, store, axiomEngine, epoch)
} else {
    // Fallback to SGP4
    objects = orbital.ScanGroundRegion(roi, store, epoch)
}
```

### Collision Analysis

Physics-informed covariance in `pkg/collision`:
```go
// Before Pc calculation, inflate covariance with learned residuals
covOutput, _ := axiomEngine.GetPhysicsInformedCovariance(primaryState, env)
primaryCov = axiomEngine.CovarianceModel.InflateCovariance(primaryCov, covOutput)

// Then proceed with Foster/Alfano Pc calculation
pc := collision.ComputeFosterPc(missDistance, combinedCov, hbr)
```

### Health Check

```go
status := axiomEngine.HealthCheck()
// Returns:
// {
//   "enabled": true,
//   "loaded_at": "2026-04-15T17:02:00Z",
//   "model_dir": "/Users/adi/idex_iaf/models",
//   "models": {
//     "covariance": {"id": "axiom_rkan_covariance", "version": "3.0.0", "status": "loaded"},
//     "maneuver":   {"id": "axiom_delta_rkan_attributor", "version": "3.0.0", "status": "loaded"},
//     "behavior":   {"id": "axiom_pol_transformer", "version": "3.0.0", "status": "loaded"}
//   }
// }
```

## Fallback Behavior

When ONNX runtime is unavailable or models fail to load:
1. **Covariance:** Returns nominal inflation (1e-6 km²) with realism score 0
2. **Maneuver:** Returns no detection with stub LaTeX explanation
3. **Behavior:** Returns ClassUnknown with 0 confidence

The system gracefully degrades to classical methods (SGP4, static SNC covariance).

## Verification

### Unit Tests
```bash
go test -tags onnx ./pkg/ml/onnx/...
```

### Integration Test
```bash
# Export models first
python ai/active/v3/axiom_zero/bridge/export_onnx.py

# Run integration test
./ssa_ops_runner --health-check --verbose
```

### Performance Benchmarks
Expected inference latencies on x86_64:
- rKAN Covariance: < 5ms per object
- Delta-rKAN Attributor: < 10ms per object (20 timesteps)
- POL Transformer: < 20ms per object (50 timesteps)

## References

- **FALLBACK_LOGIC.md** - Complete decision matrix for model fallback
- **ADR-002-HYPERION.md** - Architecture Decision Record for physics integration
- **ai/active/v3/axiom_zero/sda/** - Python SDA module implementations
