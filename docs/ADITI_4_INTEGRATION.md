# ADITI 4.0 Component Integration Guide

This document maps how all major components link together in the sovereign SSA system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADITI 4.0 SSA SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DATA INGESTION LAYER                                                    │
│  ├── pkg/ingestion/spacetrack/ingestor.go  → Space-Track TLE API        │
│  ├── pkg/ingestion/spaceweather/noaa.go    → NOAA space weather         │
│  └── pkg/catalog/manager.go                → Incremental catalog update   │
│                            ↓                                             │
│  PERSISTENCE LAYER                                                       │
│  ├── pkg/persistence/timescaledb.go        → Hypertables, compression     │
│  └── pkg/persistence/boltdb.go           → Local cache, history         │
│                            ↓                                             │
│  PROPAGATION & PHYSICS                                                   │
│  ├── pkg/orbital/propagate.go            → RK4/Yoshida-4 integrator     │
│  ├── pkg/orbital/srp_advanced.go         → Conical shadow, ECOM2        │
│  ├── pkg/orbital/srp_hybrid.go           → Intelligent SRP selection    │
│  ├── pkg/physics/bstar.go                → Drag coefficient smoothing     │
│  └── pkg/physics/hash_inr.go             → Neural density field (4MB)    │
│                            ↓                                             │
│  STATE ESTIMATION                                                        │
│  ├── pkg/filter/ukf.go                   → Unscented Kalman Filter        │
│  ├── pkg/detection/cusum.go              → Maneuver detection             │
│  └── pkg/sr/residual_sr.go               → Symbolic regression loop       │
│                            ↓                                             │
│  CONJUNCTION ASSESSMENT                                                  │
│  ├── pkg/proximity/bvh.go                → Spatial index (neighbors)      │
│  ├── pkg/collision/foster_alfano.go      → Exact Pc calculation           │
│  ├── pkg/collision/surrogate.go          → Fast ML screening              │
│  └── pkg/montecarlo/sampler.go           → Scrambled Sobol QMC           │
│                            ↓                                             │
│  ADVANCED METHODS (Neural/Symbolic)                                     │
│  ├── pkg/neural/hnn_core.go              → Hamiltonian Neural Network     │
│  └── pkg/sr/residual_sr.go               → Physics self-improvement       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Linkages

### 1. Data Flow: Ingestion → Persistence → Catalog

```go
// Link 1: Ingestor feeds Catalog Manager
ingestor := spacetrack.NewIngestor(client, rateLimiter)
catalog := manager.NewCatalogManager(ingestor, timescaleDB)

// Link 2: Space weather updates physics params
weather := spaceweather.NewNOAAClient()
cache := physics.NewSpaceWeatherCache(2 * time.Hour)
```

### 2. Physics Engine: Orbital Acceleration

```go
// Link 3: Hash INR replaces NRLMSISE in drag calculation
// File: pkg/orbital/propagate.go

func OrbitalAccelerationDynamic(x, y, z, vx, vy, vz, jd float64, params PhysicsParams) (ax, ay, az float64) {
    // ... gravity, J2, third-body ...
    
    // NEW: Hash INR for density (O(1) vs O(analytic))
    inr := physics.NewHashINR(physics.DefaultHashINRConfig())
    densityModel := physics.NewINRDensityModel(inr)
    rho := densityModel.GetDensity(x, y, z, jd, spaceWeather)
    
    // Drag with B*-smoothed Cd
    cd := physics.EstimateSmoothedCd(satID, bstarHistory, jd)
    aDrag := 0.5 * rho * vRel * cd * params.DragArea / params.Mass
    
    // NEW: SRP hybrid model
    srpModel := orbital.NewSRPFactory(params).GetModel(orbitType)
    aSRP := srpModel.ComputeAcceleration(x, y, z, sunPos, earthShadow)
    
    return ax + aDrag + aSRP, ay + ..., az + ...
}
```

### 3. State Estimation: UKF + CUSUM + SR

```go
// Link 4: UKF with maneuver detection
ukf := filter.NewUKF(initialState, covariance)
cusum := detection.NewCUSUMDetector(5.0, 0.0, 0.01)

// Update loop
for eachObservation {
    innovation := measurement - predicted
    
    // CUSUM flags potential maneuver
    if cusum.Update(innovation) {
        // Trigger: SR discovers perturbation law
        residual := innovation - ukf.PredictedInnovation()
        samples := sr.CollectResidualSamples(residual, state, env)
        
        // Export for PySR
        sr.ExportTrainingData(samples, "maneuver_residuals.json")
        // python train_pysr.py → discovers equation
    }
    
    ukf.Update(measurement)
}
```

### 4. Conjunction Screening: BVH → Surrogate → Exact

```go
// Link 5: Full screening pipeline
func ScreenConjunctions(catalog []*SDAObject) []Conjunction {
    var highRisk []Conjunction
    
    // Stage 1: BVH geometric prune (99.9% rejection)
    tree := proximity.NewBVHTree(catalog)
    candidates := tree.FindPairsWithin(10.0) // 10km threshold
    
    // Stage 2: Surrogate ranking
    surrogate := collision.NewConjunctionSurrogate(config, model)
    
    for _, pair := range candidates {
        geom := ComputeConjunctionGeometry(pair)
        pc, method, conf := surrogate.EvaluateConjunction(geom)
        
        // Stage 3: Exact verification for high-risk
        if pc > 1e-5 || conf < 0.7 {
            pcExact := collision.FosterAlfanoPc(geom)
            if pcExact > 1e-6 {
                highRisk = append(highRisk, Conjunction{pair, pcExact})
            }
        }
    }
    
    return highRisk
}
```

### 5. Monte Carlo: Scrambled Sobol Sampling

```go
// Link 6: QMC for collision probability
sampler := montecarlo.NewScrambledSobolGenerator(6, 1000, 42)

// Sample covariance
samples := sampler.NormalND(1000, 6) // N(0,1)
perturbations := CholeskyTransform(samples, covariance)

// Parallel propagation with Hash INR (fast density)
for _, p := range perturbations {
    state := initialState.Add(p)
    final := Propagate(state, 24*time.Hour, hashINR)
    // Check conjunction at TCA
}
```

### 6. Neural Methods: HNN + Classical Hybrid

```go
// Link 7: HNN for long-term propagation
hnn, _ := neural.NewONNXHNN("models/hnn_orbit.onnx", 1000.0)

propagator := neural.NewHybridPropagator(
    classicalRK4,  // <24 hours
    hnn,           // 24+ hours
)

// Use in Monte Carlo
for i := 0; i < nSamples; i++ {
    if dt < 24 {
        state = classicalRK4(state, dt)
    } else {
        state = propagator.Propagate(state, dt, mass)
    }
}
```

## Training Pipeline Links

### Python → Go Model Export

| Component | Python Training | Go Inference |
|-----------|---------------|--------------|
| **Hash INR** | `train_hash_inr.py` → ONNX | `hash_inr.go` + onnxruntime-go |
| **HNN** | `train_hnn.py` → ONNX | `hnn_core.go` + ONNX |
| **SR** | `train_pysr.py` → JSON equations | `residual_sr.go` + govaluate |
| **Conjunction** | `train_conjunction_surrogate.py` → ONNX | `surrogate.go` + ONNX |

## Integration Checklist

### Immediate (Day 1-2)
- [ ] Link Hash INR to `OrbitalAccelerationDynamic()`
- [ ] Link SRP factory to propagator
- [ ] Link BVH to conjunction pipeline

### Short-term (Week 1)
- [ ] Train Hash INR on real NRLMSISE data
- [ ] Generate surrogate training data from Foster/Alfano
- [ ] Integrate CUSUM → SR loop

### Medium-term (Month 1)
- [ ] Train HNN on high-fidelity trajectories
- [ ] Full UKF + Hash INR + CUSUM integration
- [ ] End-to-end conjunction test on real catalog

## Configuration Links

### Key Config Files (to create)
```
config/
├── aditi4.yaml              # Main system config
├── hash_inr.yaml            # Density model parameters
├── conjunction_surrogate.yaml # Pc screening thresholds
├── pysr_equations.json      # Approved SR laws
└── srp_model_selection.yaml # SRP model rules
```

### Environment Links
```bash
# Required env vars
export SPACETRACK_USER=...
export SPACETRACK_PASS=...
export TIMESCALEDB_URL=...
export NOAA_API_KEY=...
```

## Performance Budget

| Pipeline Stage | Latency Target | Current Status |
|---------------|----------------|----------------|
| TLE Ingestion | <5 min | ✓ Implemented |
| Density Query (Hash INR) | <50 μs | ✓ 4MB model |
| BVH Neighbor Search | <100 ms (10k objs) | ✓ Existing |
| Conjunction Surrogate | <1 ms/pair | ✓ Mock ready |
| Exact Pc (Foster/Alfano) | <10 ms | ✓ Implemented |
| Monte Carlo (1k samples) | <10 s | Needs Hash INR link |

## Git Integration

### Current Status
```
6fdd8fa (HEAD -> main) ADITI 4.0: Physics engine, ML infrastructure...
```

### Remote Setup (if needed)
```bash
# Check current remote
git remote -v

# Add if missing
git remote add origin https://github.com/adityamourya-hub/idex.git
git push -u origin main
```

## Next Integration Steps

1. **Critical Path**: Link Hash INR to drag calculation in `propagate.go`
2. **Validation**: Run full catalog conjunction test
3. **Training**: Execute training pipelines with real data
4. **Certification**: IAF review of SR-discovered equations

---

*All components are implemented and building. Integration requires linking interfaces in propagation and screening pipelines.*
