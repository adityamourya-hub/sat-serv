# Electronic Warfare (EW) Intent Integration for ADITI 4.0

## Overview

This document describes the integration of RF Intelligence (SIGINT) and Optical Relative Motion (Star Tracker SSA) capabilities into the ADITI 4.0 sovereign SSA stack. This integration enables the system to move from simple object tracking to monitoring the Electromagnetic Order of Battle (EOB).

## Architecture

### 1. RF Signal Processing (pkg/sigint/rf_processor.go)

The RF processor provides real-time signal characterization using FFT-based spectral analysis:

**Key Components:**
- `RFProcessor`: Main signal processing engine
- `SignalSample`: IQ samples from SDR
- `RFSignal`: Detected signal characteristics
- `SpectralPeak`: Frequency domain peaks
- `Geolocator`: TDOA/FDOA geolocation

**Signal Characterization:**
```go
// Three primary physics-based primitives
RSSI (Signal Strength)     // dBm - Jammer detection
Doppler Shift (Δf)         // Hz - Geolocation
Modulation Type            // QPSK, 16-QAM, FHSS - Intent inference
```

**Thresholds:**
- `JammerPowerThreshold`: -30.0 dBm
- `BurstThresholdHz`: 1000.0 Hz
- `ModulationConfidence`: 0.7

### 2. Star Tracker SSA (pkg/sigint/star_tracker.go)

Repurposes star trackers as wide-FOV sensors for non-catalog object detection:

**Key Components:**
- `StarTrackerSSA`: Background subtraction and streak detection
- `StarCatalogEntry`: Reference star catalog
- `DetectedObject`: RSO detection results
- `StarFrame`: Captured sensor frames

**Algorithm:**
1. Capture low-exposure frame
2. Subtract expected star field (from pymeeus ephemeris)
3. Identify remaining objects (RSOs)
4. Characterize relative motion (streak detection)

**Thresholds:**
- `StreakLengthPixels`: 10.0
- `StreakVelocityPixels`: 5.0
- `BackgroundThreshold`: 3.0 sigma
- `FrameExposureTime`: 0.1 seconds
- `FrameRate`: 10.0 Hz

### 3. Intent Inference Matrix (pkg/sigint/intent_matrix.go)

Combines EM and optical signatures to infer payload intent:

**Intent Classification Rules:**

| EM Signature | Optical Signature | Payload Intent | Threat Level |
|------------|-------------------|----------------|--------------|
| Wide-band Noise (Jammer) | Nadir-pointing | Jamming: Ground-link disruption | CRITICAL |
| Periodic Pulses (L-Band) | Star-pointing | Comms Relay: Data movement | MEDIUM |
| High Heat (IR Peak) | Offset-pointing | ELINT: Signal intelligence | HIGH |
| No EM Signature | Rapid proximity | Kinetic/Robotic Threat | CRITICAL |
| Low-power + Station-keeping | Stabilized | Co-orbital Inspector | HIGH |

**Fusion Weights:**
- `EMWeight`: 0.5
- `OpticalWeight`: 0.5
- `IntentConfidenceThreshold`: 0.7

### 4. TDOA/FDOA Geolocation

Multi-satellite geolocation for ground-based jammer positioning:

**Physics:**
```
Δd = c · (t₂ - t₁)
```

Where:
- `c`: Speed of light (299,792,458 m/s)
- `t₁`, `t₂`: Arrival times at satellites A and B
- `Δd`: Distance difference defining hyperbola

**Components:**
- `TDOAMeasurement`: Time difference measurements
- `FDOAMeasurement`: Frequency difference measurements
- `Geolocator`: Hyperbola intersection calculation
- `GeolocationResult`: Estimated emitter position

**Resolution:**
- `TDOAResolutionNs`: 10 nanoseconds
- `FDOAResolutionHz`: 100 Hz

## Integration with Threat Classifier

The `pkg/threat/classifier.go` integrates EW awareness through:

### Extended RSOBehavior Struct

```go
type RSOBehavior struct {
    // EM signatures (EW awareness)
    EMSignalType   string  // JAMMER, COMMS_RELAY, ELINT, UNKNOWN
    EMModulation   string  // QPSK, 16-QAM, FHSS, etc.
    EMPowerDBm     float64
    EMBandwidth    float64
    EMIsBurst      bool
    EMDopplerShift float64

    // Optical signatures (Star Tracker SSA)
    AttitudeMode   string  // NADIR, STAR_POINTING, OFFSET
    IsManeuvering  bool
    RelativeMotion bool
    StreakDetected bool
}
```

### Intent-Based Threat Analysis

```go
func (tc *ThreatClassifier) analyzeWithIntentInference(behavior *RSOBehavior) (ThreatType, ThreatLevel, string) {
    // Jamming: Wide-band noise + Nadir-pointing
    if behavior.EMSignalType == "JAMMER" && behavior.AttitudeMode == "NADIR" {
        return ThreatJammer, LevelCritical, "EW-aware: Jamming detected"
    }
    
    // Comms Relay: Periodic pulses + Star-pointing
    if behavior.EMSignalType == "COMMS_RELAY" && behavior.AttitudeMode == "STAR_POINTING" {
        return ThreatNone, LevelLow, "EW-aware: Comms relay activity"
    }
    
    // ELINT: High power + Offset-pointing
    if behavior.EMSignalType == "ELINT" && behavior.AttitudeMode == "OFFSET" {
        return ThreatInspector, LevelHigh, "EW-aware: ELINT activity"
    }
    
    // Kinetic Threat: No EM + Rapid proximity
    if behavior.EMSignalType == "UNKNOWN" && behavior.StreakDetected {
        return ThreatGrapple, LevelCritical, "EW-aware: Kinetic threat"
    }
}
```

## Usage Examples

### RF Signal Processing

```go
// Create RF processor
processor := sigint.NewRFProcessor(10e6) // 10 MHz sample rate

// Process IQ samples
samples := []complex128{...} // From SDR
signals := processor.CharacterizeSignal(samples, 10e6)

// Detect jammers
for _, signal := range signals {
    if signal.IsJamming {
        fmt.Printf("Jammer detected at %.2f MHz, Power: %.1f dBm\n",
            signal.CenterFreq/1e6, signal.PowerDBm)
    }
}
```

### Star Tracker SSA

```go
// Create star tracker SSA
ssa := sigint.NewStarTrackerSSA()

// Load star catalog
ssa.LoadCatalog("hipparcos_catalog.json")

// Process frame
detected := ssa.ProcessFrame(frame)

// Identify non-catalog objects
for _, obj := range detected {
    if !obj.IsCatalog && obj.IsStreak {
        fmt.Printf("RSO detected: Streak length %.1f pixels, Velocity (%.1f, %.1f)\n",
            obj.StreakLength, obj.VelocityX, obj.VelocityY)
    }
}
```

### Intent Inference

```go
// Create intent matrix
matrix := sigint.NewIntentInferenceMatrix()

// Classify intent
emSig := sigint.EMSignature{
    SignalType: "JAMMER",
    PowerDBm:   -25.0,
    Bandwidth:  100e6,
}

opticalSig := sigint.OpticalSignature{
    AttitudeMode: "NADIR",
    IsStabilized: true,
}

classification := matrix.InferIntent(emSig, opticalSig)
fmt.Printf("Intent: %s (Confidence: %.2f)\n",
    classification.Intent, classification.Confidence)
```

### TDOA Geolocation

```go
// Create geolocator
geo := sigint.NewGeolocator()

// Add satellites to network
geo.AddSatellite("SAT-1", [3]float64{x1, y1, z1}, [3]float64{vx1, vy1, vz1})
geo.AddSatellite("SAT-2", [3]float64{x2, y2, z2}, [3]float64{vx2, vy2, vz2})

// Measure TDOA
tdoa := sigint.TDOAMeasurement{
    SatelliteA: "SAT-1",
    SatelliteB: "SAT-2",
    TimeDiffNs: 50.0, // 50 ns difference
}

// Geolocate
results := geo.CalculateTDOAHyperbola(tdoa)
for _, result := range results {
    fmt.Printf("Jammer location: %.4f°, %.4f° (Confidence: %.2f)\n",
        result.Latitude, result.Longitude, result.Confidence)
}
```

## Strategic Payoff

### 1. Dual-Use Hardware
- Star trackers and comms antennas serve as passive sensors
- No specialized SSA hardware required
- Leverages existing satellite capabilities

### 2. Real-Time Attribution
- Distinguish between space weather and adversary jamming
- Automated threat typing from signal signatures
- Instant notification to ground operators

### 3. Sovereign Signal Library
- TimescaleDB Tracklets store adversary EM patterns
- Machine learning on signature evolution
- Predictive threat assessment

### 4. Kill Chain Integration
```
Detection (S2 Filter) → Tracking (HPOP+EKF) →
Characterization (SIGINT+Optical) →
Intent Inference (Matrix) →
Fire Control Handover (FCH) →
Engagement (Interceptor)
```

## File Structure

```
pkg/sigint/
├── rf_processor.go      # FFT-based signal characterization
├── star_tracker.go      # Background subtraction, streak detection
└── intent_matrix.go     # Multi-modal intent inference

pkg/threat/
└── classifier.go        # EW-aware threat classification
```

## Constants Reference

### RF Processing
```go
JammerPowerThreshold     = -30.0  // dBm
BurstThresholdHz         = 1000.0 // Hz
ModulationConfidence     = 0.7
SpeedOfLight             = 299792458.0 // m/s
TDOAResolutionNs         = 10.0
FDOAResolutionHz         = 100.0
```

### Star Tracker
```go
StreakLengthPixels       = 10.0
StreakVelocityPixels     = 5.0
BackgroundThreshold      = 3.0
StarCatalogAccuracy      = 0.1
FrameExposureTime        = 0.1
FrameRate                = 10.0
```

### Intent Inference
```go
IntentConfidenceThreshold = 0.7
EMWeight                  = 0.5
OpticalWeight             = 0.5
```

## Dependencies

- `github.com/cpmech/gosl/dsp`: FFT implementation
- Standard Go `math`, `math/cmplx`: Complex number operations
- Python/MPDAF: Background subtraction (star field)
- Python/JAX: Modulation identification
- pymeeus: Ephemeris calculation for star catalogs

## Notes

- All RF processing is Go-native for real-time performance
- Star tracker processing uses Python/MPDAF for background subtraction
- Intent inference combines both for maximum accuracy
- TDOA/FDOA requires multiple satellites in mesh network
- MessagePack serialization ensures compact tactical data links
