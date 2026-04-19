# SIGINT Electronic Order of Battle (EOB) Integration

## Overview

This document describes the integration of SIGINT capabilities into ADITI 4.0, transforming it from a passive orbital tracker into a **Proactive Electronic Order of Battle (EOB)** system. By mapping specific microwave and decimetric signatures to adversary intent, the system detects electronic precursors to physical maneuvers.

## Key Components

### 1. Enhanced Frequency Configuration (`pkg/physics/frequencies.go`)

**New SIGINT Bands Added:**

| Band Name | Frequency Range | Strategic Use | Threat Level |
|-----------|----------------|---------------|--------------|
| **SEVER_GORIZONT_S** | 4.435-4.750 GHz | Russian Strategic Troposcatter | 4 (High) |
| **TACTICAL_TROPO** | 350 MHz-2 GHz | Battlefield OTH Communications | 3 (Medium) |
| **ORLAN10_TELEMETRY** | 850-930 MHz | UAV Command/Control | 3 (Medium) |
| **BALLISTIC_TELEM_L** | 1.435-1.525 GHz | Missile Flight Test Telemetry | 5 (Critical) |
| **BALLISTIC_TELEM_S** | 2.200-2.390 GHz | Narrow-band Missile Monitoring | 5 (Critical) |
| **PROFORMA_X** | 8-12 GHz | Radar-to-Missile Data Link | 5 (Critical) |
| **PROFORMA_KA** | 26.5-40 GHz | High-bandwidth Data Link | 5 (Critical) |
| **LINK_16** | 969-1206 MHz | NATO Tactical Data Link | 2 (Low-Med) |
| **HAVE_QUICK** | 225-400 MHz | Frequency Hopping Comms | 2 (Low-Med) |

### 2. Enhanced Intent Matrix (`pkg/sigint/intent_matrix.go`)

**New Intent Rules:**

| Rule | EM Condition | Optical Condition | Threat | Action |
|------|--------------|-------------------|--------|--------|
| **STRATEGIC_TROPO** | 4.4 GHz QPSK + "Hollow" signature | Fixed ground station | HIGH | TACTICAL_ALERT |
| **ELECTRONIC_WARFARE** | 850-930 MHz + high power + no ADS-B | UAV-like behavior | CRITICAL | RF_COUNTERMEASURE |
| **KINETIC_THREAT_SIGINT** | PROFORMA X/Ka + narrow beam | Radar tracking + maneuver | CRITICAL | EVASIVE_MANEUVER |
| **DECEPTION_DETECTED** | Clear COMINT voice | No thermal signature | HIGH | EMCON_INITIATE |
| **BALLISTIC_MISSILE** | L/S-band telemetry bandwidth | Ascent thermal signature | CRITICAL | TACTICAL_ALERT |

### 3. SIGID Bridge (`pkg/sigint/sigid_bridge.go`)

**Spectral Template Matching:**
- Hamming distance calculation for O(1) template matching
- Local SQLite database for air-gap resilience
- Default templates: Link 16, HAVE QUICK, Sever/Gorizont, PROFORMA, TETRA, Inmarsat

**Key Features:**
```go
// Match waterfall against known templates
similarity, templateID := bridge.MatchWaterfall(detectedMask, freqHz, bandwidthHz)

// Modulation checksum for PRI validation
matchScore, templateID := bridge.MatchModulationChecksum(freqHz, pri, modulation)
```

### 4. FlightRadar24 Bridge (`pkg/sigint/fr24_bridge.go`)

**PACE Model Implementation:**
- **Primary**: Real-time FR24 API queries
- **Alternate**: Local SQLite cache (15-minute TTL)
- **Contingency**: Stale cache data if API fails
- **Emergency**: Manual aircraft correlation

**Negative Correlation:**
```go
// Check if detected signal correlates with known aircraft
isCivilian, aircraft, err := bridge.CheckNegativeCorrelation(lat, lon, alt, timestamp)
```

### 5. Android RTL-SDR Bridge (`ai/sigint/android_sdr.py`)

**Capabilities:**
- TCP streaming from rtl_tcp on Android devices
- Real-time IQ sample acquisition
- JAX FFT engine for spectrogram processing
- FFT fingerprinting (Shazam-style signal identification)
- GPS spoofing detection (Android GPS vs SGP4 truth comparison)

**Usage:**
```python
# Connect to Android SDR
sdr = connect_android_sdr("192.168.1.101", 1234)

# Scan frequency range
signals = sdr.scan_frequency_range(1575.0e6, 1576.0e6)

# Detect GPS spoofing
result = sdr.detect_gps_spoofing(android_gps, sgp4_truth)
```

## Integration Flow

```
[Android SDR @ Edge] 
    ↓ (TCP streaming)
[Python JAX FFT Engine] → FFT Fingerprinting
    ↓
[SIGID Bridge Match] → Template ID (Link 16, PROFORMA, etc.)
    ↓
[Intent Matrix] → Intent Classification (STRATEGIC_TROPO, KINETIC_THREAT)
    ↓
[FR24 Bridge] → Negative Correlation Check
    ↓
[RL Policy Head] → Action (TACTICAL_ALERT, RF_COUNTERMEASURE)
    ↓
[Neural HUD] → Threat Visualization
```

## Dual-Eye System

**Global Intelligence (FR24):**
- Civilian aircraft correlation
- Airspace deconfliction
- Negative correlation for threat isolation

**Local Tactical Truth (Android SDR):**
- Edge-based signal verification
- Real-time spectrum monitoring
- Jamming/spoofing detection

**Combined Decision Matrix:**

| FR24 Data | Android SDR | Conclusion |
|-----------|-------------|------------|
| Flight AI101 @ 35k ft | Clean L-band | **Green**: Civilian traffic |
| Flight on-track | L1/L2 noise spike | **Yellow**: GNSS interference |
| No aircraft | Wideband X-band burst | **Red**: Rapid threat |

## Demo Scenario: "Handover"

1. **Phase 1**: FR24 Bridge shows "Clean Corridor" over Siliguri
2. **Phase 2**: Simulate jammer that blinds global API
3. **Phase 3**: System switches to Android RTL-SDR nodes to maintain spectrum custody

This demonstrates **sovereign resilience** - the system operates independently of external infrastructure.

## Files Created/Updated

```
pkg/physics/
└── frequencies.go           # (updated) SIGINT EOB bands

pkg/sigint/
├── intent_matrix.go         # (updated) New ELINT rules
├── sigid_bridge.go          # (new) Spectral template matching
└── fr24_bridge.go           # (new) FR24 API + local cache

ai/sigint/
├── sdr_monitor.py          # (existing) GPS spoofing detection
└── android_sdr.py          # (new) Mobile SDR + JAX FFT
```

## Strategic Payoff

1. **Electronic Signature Identification**: Know the "Who" and "How" before the threat materializes
2. **Sovereign Resilience**: Air-gapped operation with local caching
3. **Dual-Eye Verification**: Global intelligence + local tactical truth
4. **Intent-Aware Response**: Actions tailored to specific threat types
5. **Proactive Detection**: Electronic precursors reveal imminent maneuvers

## Usage Example

```go
// Initialize components
fr24 := NewFR24Bridge(apiKey, "/cache/fr24")
sigid := NewSIGIDBridge("/cache/sigid")
intentMatrix := NewIntentInferenceMatrix()

// Process detection
detectedFreq := 4.45e9 // Sever/Gorizont frequency
detectedBandwidth := 315e6

// 1. Check FR24 for civilian correlation
isCivilian, aircraft, _ := fr24.CheckNegativeCorrelation(lat, lon, alt, timestamp)

// 2. Match against SIGID templates
similarity, templateID := sigid.MatchWaterfall(psdMask, detectedFreq, detectedBandwidth)

// 3. Infer intent
emSig := EMSignature{
    SignalType: "ELINT",
    Modulation: "QPSK",
    Bandwidth: detectedBandwidth,
    PowerDBm: -50.0,
}
optSig := OpticalSignature{
    AttitudeMode: "STABILIZED",
    IsStabilized: true,
}
intent := intentMatrix.InferIntent(emSig, optSig)

// 4. Take action based on intent
if intent.Intent == "STRATEGIC_TROPO" && !isCivilian {
    triggerTacticalAlert("High-Value Node Detected")
}
```

---

**Final Status**: System provides complete Electronic Order of Battle for Siliguri Corridor with air-gap resilience and dual-eye verification.
