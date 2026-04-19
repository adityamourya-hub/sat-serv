# Space Domain Awareness (SDA) Integration for ADITI 4.0

## Overview

This document describes the integration of terrestrial RF monitoring (SDR) and signal intelligence (SIGINT) capabilities into the ADITI 4.0 sovereign SSA stack, enabling full-spectrum Space Domain Awareness (SDA) rather than just Space Situational Awareness (SSA).

## Architecture

### 1. GPS Spoofing Detection (pkg/sigint/gps_spoofing.go)

The GPS spoofing detector compares SGP4 "Truth" position with NMEA GPS data to detect if the ground site is being manipulated by an adversary.

**Key Components:**
- `GPSSpoofDetector`: Main detection engine
- `NMEAData`: Parsed GPS NMEA sentences
- `SGP4Truth`: Expected position from propagator
- `GPSComparisonResult`: Comparison metrics
- `GNSSMonitor`: L1/L2/L5 band monitoring

**Detection Logic:**
```go
// Three primary detection vectors
Position Difference  // Haversine distance (meters)
Velocity Difference  // m/s difference
Altitude Difference  // meters difference
Heading Difference   // degrees difference
```

**Thresholds:**
- `GPSPositionThresholdMeters`: 50.0
- `GPSVelocityThresholdMps`: 10.0
- `GPSAltitudeThresholdMeters`: 100.0
- `GPSHeadingThresholdDegrees`: 10.0
- `GPSSpoofingConfidenceThreshold`: 0.7

**Alert Severity:**
- **CRITICAL**: Spoofing probability > 0.9
- **HIGH**: Spoofing probability > 0.8
- **MEDIUM**: Spoofing probability > 0.7

### 2. NTIA Spectrum Chart (pkg/physics/frequencies.go)

Implements regulatory filter for signal classification and threat assessment using the NTIA spectrum allocation chart.

**Frequency Bands:**

| Band | Frequency | Primary Use | Threat Level |
|------|-----------|-------------|--------------|
| GPS L1 | 1575.42 MHz | Navigation | 5 (Critical) |
| GPS L2 | 1227.60 MHz | Navigation (Military) | 5 (Critical) |
| GPS L5 | 1176.45 MHz | Navigation (Civil) | 5 (Critical) |
| Space Ops | 2200-2290 MHz | Satellite Control | 3 (Medium) |
| SATCOM UHF | 240-270 MHz | Military SATCOM | 4 (High) |
| SATCOM SHF | 7.9-8.4 GHz | Military SATCOM | 4 (High) |
| X-Band Radar | 8.5-10.5 GHz | Space Tracking | 3 (Medium) |
| Ka-Band | 26.5-40 GHz | SATCOM/Radar | 3 (Medium) |
| ADS-B | 1090 MHz | Aircraft Transponder | 1 (Low) |
| ISM 2.4 | 2.4-2.5 GHz | Wi-Fi/Bluetooth | 2 (Low-Medium) |
| Jammer X | 8.0-12.0 GHz | Adversary Jamming | 5 (Critical) |
| Jammer Ka | 26.0-40 GHz | Adversary Jamming | 5 (Critical) |

**Signal Classification:**
- **Adversary**: Military bands with unknown license, jamming bands
- **Friendly**: Space operations, SATCOM, ADS-B
- **Restricted**: Military, GPS, space-related

**Utility Functions:**
- `HzToMHz`, `MHzToHz`, `HzToGHz`, `GHzToHz`
- `CalculateWavelength`: λ = c/f
- `dBmToWatts`, `WattsTodBm`
- `CalculateFreeSpacePathLoss`: FSPL calculation
- `CalculateSignalPowerAtReceiver`: Received power estimation

### 3. RTL-SDR GPS Monitoring (ai/sigint/sdr_monitor.py)

Python module for RTL-SDR GPS monitoring and spoofing detection.

**Key Components:**
- `RTLSDRMonitor`: RTL-SDR device interface
- `GPSNMEAParser`: NMEA sentence parser
- `SpoofingDetector`: Integrated detection engine

**GPS Bands Monitored:**
- **L1**: 1575.42 MHz (20.46 MHz bandwidth)
- **L2**: 1227.60 MHz (20.46 MHz bandwidth)
- **L5**: 1176.45 MHz (20.46 MHz bandwidth)

**Detection Method:**
1. Read IQ samples from RTL-SDR
2. Compute FFT and Power Spectral Density (PSD)
3. Detect signal power
4. Compare with baseline
5. Update spoofing score
6. Check against threshold

**Spoofing Indicators:**
- Power anomaly (> 20 dB above baseline)
- Low CN0 (Carrier-to-noise ratio)
- Invalid GPS fix
- High HDOP/VDOP (Dilution of Precision)
- Abnormal velocity (> 1 km/s for ground vehicle)

### 4. ADS-B Correlation (pkg/sigint/adsb_correlation.go)

Performs ADS-B correlation to distinguish between space threats and terrestrial aircraft clutter.

**Key Components:**
- `ADSBCorrelator`: Main correlation engine
- `ADSBReport`: Decoded ADS-B message
- `SpaceDetection`: S2 filter detection
- `CorrelationResult`: Correlation metrics
- `UCTAlert`: Uncorrelated Track alert

**Correlation Logic:**
1. Calculate expected terrestrial position from space detection
2. Find matching ADS-B report in dataset
3. Check temporal proximity (< 5 seconds)
4. Calculate position/velocity differences
5. Compute correlation score
6. Determine if correlated (terrestrial) or uncorrelated (space)

**Thresholds:**
- `AzimuthThresholdDegrees`: 5.0
- `RangeThresholdMeters`: 5000.0
- `AltitudeThresholdMeters`: 1000.0
- `VelocityThresholdMps`: 100.0
- `CorrelationThreshold`: 0.7

**Alert Types:**
- **RAPID**: High-velocity uncorrelated track (> 1 km/s) - ASAT launch possible
- **INVESTIGATE**: Uncorrelated track - potential space object
- **TERRESTRIAL**: Correlated with aircraft - downgraded threat

### 5. BLE Radar (pkg/sigint/ble_radar.go)

Provides physical node security by detecting unauthorized Bluetooth devices near Reticulum mesh nodes.

**Key Components:**
- `BLERadar`: Bluetooth device detection
- `BLEDevice`: Detected device information
- `BLESecurityAlert`: Security alert
- `BLESecurityReport`: Security summary

**Detection Thresholds:**
- `ProximityThresholdMeters`: 20.0
- `AlertThresholdSeconds`: 10.0
- `MaxRSSI`: -30 dBm (very close)
- `MinRSSI`: -90 dBm (far away)

**Proximity Categories:**
- **IMMEDIATE**: < 1 meter
- **NEAR**: 1-5 meters
- **FAR**: 5-10 meters
- **OUT_OF_RANGE**: > 10 meters

**Alert Severity:**
- **CRITICAL**: RSSI > -40 dBm, immediate action required
- **HIGH**: RSSI > -50 dBm, alert operators
- **MEDIUM**: RSSI > -60 dBm, increase monitoring
- **LOW**: RSSI < -60 dBm, continue monitoring

**Auto-Response (Critical Alerts):**
1. Wipe Plasma buffer (shared memory)
2. Air-gap systems (disconnect network)
3. Alert operators

## Integration Flow

### Complete Kill-Chain with RF Layer

```
T+0s:  Go engine detects orbital maneuver (HPOP)
T+5s:  Python brain identifies material (GaAs/Kevlar)
T+15s: RTL-SDR detects wideband noise floor spike in X-band
T+20s: Neural engine compares spike to SIGID Wiki patterns
T+30s: TDOA solver geolocates jammer ground station
T+45s: LXMF notifies field commander with counter-response
```

### Data Flow

```
[RTL-SDR] → [PSD Analysis] → [NTIA Chart] → [Signal Classification]
                                          ↓
[GPS Receiver] → [NMEA Parser] → [GPSSpoofDetector] → [Truth Comparison]
                                          ↓
[ADS-B Feed] → [ADSBCorrelator] → [Negative Correlation] → [UCT Alert]
                                          ↓
[BLE Radar] → [Device Detection] → [Proximity Check] → [Security Alert]
                                          ↓
[Threat Classifier] → [EW-Aware Analysis] → [Fire-Control Handover]
```

## Usage Examples

### GPS Spoofing Detection

```go
// Create truth position (from SGP4)
truth := &sigint.SGP4Truth{
    Timestamp:  time.Now(),
    Latitude:   26.5,
    Longitude:  88.4,
    Altitude:   100.0,
    IsFixed:    true,
}

// Create detector
detector := sigint.NewGPSSpoofDetector(truth)

// Process NMEA data
nmea := sigint.NMEAData{
    Timestamp:  time.Now(),
    Latitude:   26.51,
    Longitude:  88.41,
    Altitude:   150.0,
    Velocity:   10.0,
    Satellites: 8,
    HDOP:       1.5,
    VDOP:       2.0,
    IsValid:    true,
}

result := detector.ProcessNMEAData(nmea)

if result.SpoofingDetected {
    alert := detector.GenerateAlert(result)
    fmt.Printf("GPS SPOOFING DETECTED: %s\n", alert.RecommendedAction)
}
```

### NTIA Spectrum Classification

```go
// Create spectrum chart
chart := physics.NewNTIASpectrumChart()

// Classify signal
freq := 1575.42e6 // GPS L1
power := -25.0     // dBm (high power)

classification := chart.ClassifySignal(freq, power)

fmt.Printf("Band: %s, Use: %s, Threat: %d, Adversary: %v\n",
    classification.BandName,
    classification.PrimaryUse,
    classification.ThreatLevel,
    classification.IsAdversary)
```

### ADS-B Correlation

```go
// Create correlator
correlator := sigint.NewADSBCorrelator()

// Update ADS-B dataset
reports := []sigint.ADSBReport{...}
correlator.UpdateADSBDataset(reports)

// Correlate space detection
detection := &sigint.SpaceDetection{
    ObjectID:    "99999",
    Azimuth:     45.0,
    Elevation:   30.0,
    Range:       100000.0,
    Velocity:    7500.0,
    Confidence:  0.9,
    Timestamp:   time.Now(),
}

result := correlator.CorrelateWithSpaceDetection(detection)

if result.IsCorrelated {
    fmt.Printf("TERRESTRIAL: Aircraft %s\n", result.ADSBCandidate.Callsign)
} else {
    alert := correlator.GenerateUCTAlert(detection)
    fmt.Printf("SPACE THREAT: %s\n", alert.Recommendation)
}
```

### BLE Radar Security

```go
// Create BLE radar
radar := sigint.NewBLERadar()

// Add whitelisted devices
radar.AddToWhitelist("AA:BB:CC:DD:EE:FF", "Operator Phone")

// Start scanning
radar.StartScanning()

// Perform scan
result, _ := radar.Scan()

// Process results
alerts := radar.ProcessScanResult(result)

for _, alert := range alerts {
    if alert.Severity == "CRITICAL" {
        radar.AutoResponse(alert) // Wipe buffer, air-gap systems
    }
}
```

## Strategic Payoff

### 1. Dual-Use Hardware
- RTL-SDR: Low-cost ($20-30) passive monitoring
- Star Trackers: Repurposed as wide-FOV sensors
- Comms Antennas: Passive RF intelligence
- BLE: Existing hardware for physical security

### 2. Real-Time Attribution
- **Space Weather vs Jamming**: Distinguish natural phenomena from adversary actions
- **Terrestrial vs Space**: ADS-B correlation eliminates aircraft clutter
- **Spoofing Detection**: GPS truth comparison detects manipulation

### 3. Sovereign Signal Library
- TimescaleDB Tracklets store adversary EM patterns
- NTIA chart provides regulatory framework
- SIGID Wiki integration for threat typing
- Machine learning on signature evolution

### 4. Physical Security
- BLE radar detects unauthorized devices
- Auto-response: Wipe Plasma buffer, air-gap systems
- Whitelist management for authorized personnel
- Proximity-based threat escalation

## File Structure

```
pkg/sigint/
├── gps_spoofing.go      # GPS spoofing detection
├── adsb_correlation.go  # ADS-B terrestrial correlation
├── ble_radar.go         # Physical node security
├── rf_processor.go      # RF signal processing (existing)
├── star_tracker.go      # Star tracker SSA (existing)
└── intent_matrix.go     # Intent inference (existing)

pkg/physics/
└── frequencies.go       # NTIA spectrum chart

ai/sigint/
└── sdr_monitor.py       # RTL-SDR GPS monitoring
```

## Constants Reference

### GPS Spoofing
```go
GPSPositionThresholdMeters   = 50.0
GPSVelocityThresholdMps       = 10.0
GPSAltitudeThresholdMeters    = 100.0
GPSHeadingThresholdDegrees    = 10.0
GPSSpoofingConfidenceThreshold = 0.7
EarthRadiusMeters = 6371000.0
GPSMaxVelocity    = 1000.0
```

### ADS-B Correlation
```go
AzimuthThresholdDegrees = 5.0
RangeThresholdMeters    = 5000.0
AltitudeThresholdMeters = 1000.0
VelocityThresholdMps    = 100.0
CorrelationThreshold     = 0.7
```

### BLE Radar
```go
ProximityThresholdMeters = 20.0
AlertThresholdSeconds    = 10.0
MaxRSSI                 = -30.0
MinRSSI                 = -90.0
WhitelistTTL            = 24 * time.Hour
```

## Dependencies

- Go: `math`, `time`, `fmt`
- Python: `numpy`, `scipy`, `pyrtlsdr` (for RTL-SDR)
- External: `dump1090` or `rtl_adsb` (for ADS-B)
- External: SIGID Wiki (for threat patterns)

## Notes

- All GPS processing is Go-native for real-time performance
- RTL-SDR uses Python for FFT and PSD analysis
- ADS-B correlation requires local feed (dump1090/rtl_adsb)
- BLE radar requires noble/macOS CoreBluetooth (platform-specific)
- MessagePack serialization for compact tactical data links
- TimescaleDB for signature library and tracklets

## Next Steps

1. **Integrate pyrtlsdr**: Connect to actual RTL-SDR hardware
2. **ADS-B feed setup**: Deploy dump1090 at Siliguri site
3. **BLE radar implementation**: Platform-specific scanning
4. **SIGID Wiki scraping**: Build synthetic signal generator
5. **TDOA solver**: Multi-satellite geolocation
6. **Live testing**: Field deployment at Siliguri Corridor

## Summary

The SDA integration transforms ADITI 4.0 from a simple SSA tracker to a full-spectrum Space Domain Awareness system:

- **GPS Spoofing Detection**: SGP4 truth vs NMEA comparison
- **NTIA Spectrum Chart**: Regulatory signal classification
- **RTL-SDR Monitoring**: L1/L2/L5 band surveillance
- **ADS-B Correlation**: Terrestrial clutter rejection
- **BLE Radar**: Physical node security

This provides the IAF with sovereign capability to detect and attribute both space-based and terrestrial threats in the electromagnetic spectrum.
