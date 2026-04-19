// Package sigint provides GPS spoofing detection and monitoring
// Compares SGP4 "Truth" position with NMEA GPS data to detect ground site manipulation
package sigint

import (
	"fmt"
	"math"
	"time"
)

const (
	// GPS spoofing detection thresholds
	GPSPositionThresholdMeters     = 50.0  // 50 meters position difference
	GPSVelocityThresholdMps        = 10.0  // 10 m/s velocity difference
	GPSAltitudeThresholdMeters     = 100.0 // 100 meters altitude difference
	GPSHeadingThresholdDegrees     = 10.0  // 10 degrees heading difference
	GPSSpoofingConfidenceThreshold = 0.7   // Confidence threshold for spoofing detection

	// GPS constants
	EarthRadiusMeters = 6371000.0 // Average Earth radius
	GPSMaxVelocity    = 1000.0    // Maximum plausible GPS receiver velocity
)

// NMEAData represents parsed NMEA GPS data
type NMEAData struct {
	Timestamp  time.Time
	Latitude   float64 // Degrees
	Longitude  float64 // Degrees
	Altitude   float64 // Meters
	Velocity   float64 // m/s
	Heading    float64 // Degrees
	Satellites int     // Number of satellites
	HDOP       float64 // Horizontal dilution of precision
	VDOP       float64 // Vertical dilution of precision
	FixQuality int     // 0=No fix, 1=GPS, 2=DGPS
	IsValid    bool
}

// SGP4Truth represents the "ground truth" position from SGP4 propagator
type SGP4Truth struct {
	Timestamp time.Time
	Latitude  float64    // Degrees
	Longitude float64    // Degrees
	Altitude  float64    // Meters
	Velocity  [3]float64 // ECI velocity [x, y, z] in m/s
	IsFixed   bool       // True for fixed ground stations
}

// GPSComparisonResult represents the comparison between NMEA and SGP4 data
type GPSComparisonResult struct {
	PositionDifferenceMeters float64
	AltitudeDifferenceMeters float64
	VelocityDifferenceMps    float64
	HeadingDifferenceDegrees float64
	SpoofingProbability      float64
	SpoofingDetected         bool
	Confidence               float64
	AnomalyType              string
	DetectionTime            time.Time
}

// GPSSpoofDetector monitors GPS data for spoofing attempts
type GPSSpoofDetector struct {
	truthPosition     *SGP4Truth
	nmeaHistory       []NMEAData
	anomalyCount      int
	lastAlertTime     time.Time
	alertThreshold    int // Number of anomalies before alert
	positionTolerance float64
	velocityTolerance float64
}

// NewGPSSpoofDetector creates a new GPS spoofing detector
func NewGPSSpoofDetector(truth *SGP4Truth) *GPSSpoofDetector {
	return &GPSSpoofDetector{
		truthPosition:     truth,
		nmeaHistory:       make([]NMEAData, 0),
		anomalyCount:      0,
		lastAlertTime:     time.Time{},
		alertThreshold:    3, // Require 3 consecutive anomalies
		positionTolerance: GPSPositionThresholdMeters,
		velocityTolerance: GPSVelocityThresholdMps,
	}
}

// UpdateTruthPosition updates the SGP4 truth position
func (gsd *GPSSpoofDetector) UpdateTruthPosition(truth *SGP4Truth) {
	gsd.truthPosition = truth
}

// ProcessNMEAData processes incoming NMEA GPS data
func (gsd *GPSSpoofDetector) ProcessNMEAData(nmea NMEAData) *GPSComparisonResult {
	if !nmea.IsValid {
		return &GPSComparisonResult{
			SpoofingDetected: false,
			AnomalyType:      "INVALID_NMEA",
			DetectionTime:    time.Now(),
		}
	}

	// Store in history
	gsd.nmeaHistory = append(gsd.nmeaHistory, nmea)
	if len(gsd.nmeaHistory) > 100 {
		gsd.nmeaHistory = gsd.nmeaHistory[1:]
	}

	// Compare with truth
	result := gsd.compareWithTruth(nmea)

	// Check for spoofing
	if result.SpoofingProbability > GPSSpoofingConfidenceThreshold {
		gsd.anomalyCount++
	} else {
		gsd.anomalyCount = 0
	}

	// Trigger alert if threshold exceeded
	if gsd.anomalyCount >= gsd.alertThreshold {
		result.SpoofingDetected = true
		gsd.lastAlertTime = time.Now()
	}

	return result
}

// compareWithTruth compares NMEA data with SGP4 truth position
func (gsd *GPSSpoofDetector) compareWithTruth(nmea NMEAData) *GPSComparisonResult {
	if gsd.truthPosition == nil {
		return &GPSComparisonResult{
			SpoofingProbability: 0.0,
			SpoofingDetected:    false,
			AnomalyType:         "NO_TRUTH_DATA",
			DetectionTime:       time.Now(),
		}
	}

	result := &GPSComparisonResult{
		DetectionTime: time.Now(),
	}

	// Calculate position difference (Haversine formula)
	result.PositionDifferenceMeters = haversineDistance(
		gsd.truthPosition.Latitude, gsd.truthPosition.Longitude,
		nmea.Latitude, nmea.Longitude,
	)

	// Calculate altitude difference
	result.AltitudeDifferenceMeters = math.Abs(nmea.Altitude - gsd.truthPosition.Altitude)

	// Calculate velocity difference
	truthVelocity := math.Sqrt(
		gsd.truthPosition.Velocity[0]*gsd.truthPosition.Velocity[0] +
			gsd.truthPosition.Velocity[1]*gsd.truthPosition.Velocity[1] +
			gsd.truthPosition.Velocity[2]*gsd.truthPosition.Velocity[2],
	)
	result.VelocityDifferenceMps = math.Abs(nmea.Velocity - truthVelocity)

	// Calculate heading difference
	result.HeadingDifferenceDegrees = math.Abs(nmea.Heading - calculateHeading(gsd.truthPosition))

	// Calculate spoofing probability
	positionScore := math.Min(result.PositionDifferenceMeters/gsd.positionTolerance, 1.0)
	altitudeScore := math.Min(result.AltitudeDifferenceMeters/GPSAltitudeThresholdMeters, 1.0)
	velocityScore := math.Min(result.VelocityDifferenceMps/gsd.velocityTolerance, 1.0)
	headingScore := math.Min(result.HeadingDifferenceDegrees/GPSHeadingThresholdDegrees, 1.0)

	// Weighted average
	result.SpoofingProbability = (positionScore*0.4 + altitudeScore*0.2 + velocityScore*0.3 + headingScore*0.1)

	// Calculate confidence based on NMEA quality
	dopScore := 1.0 - math.Min((nmea.HDOP+nmea.VDOP)/10.0, 1.0)
	satScore := math.Min(float64(nmea.Satellites)/12.0, 1.0)
	result.Confidence = (dopScore + satScore) / 2.0

	// Determine anomaly type
	if result.PositionDifferenceMeters > gsd.positionTolerance*5 {
		result.AnomalyType = "POSITION_JUMP"
	} else if result.VelocityDifferenceMps > gsd.velocityTolerance*3 {
		result.AnomalyType = "VELOCITY_ANOMALY"
	} else if result.AltitudeDifferenceMeters > GPSAltitudeThresholdMeters*3 {
		result.AnomalyType = "ALTITUDE_JUMP"
	} else {
		result.AnomalyType = "MISMATCH"
	}

	return result
}

// haversineDistance calculates the great-circle distance between two points
func haversineDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const R = EarthRadiusMeters

	lat1Rad := lat1 * math.Pi / 180.0
	lat2Rad := lat2 * math.Pi / 180.0
	deltaLat := (lat2 - lat1) * math.Pi / 180.0
	deltaLon := (lon2 - lon1) * math.Pi / 180.0

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}

// calculateHeading calculates heading from ECI velocity
func calculateHeading(truth *SGP4Truth) float64 {
	// Simplified: convert ECI velocity to local heading
	// In production, use full coordinate transformation
	vx := truth.Velocity[0]
	vy := truth.Velocity[1]

	heading := math.Atan2(vy, vx) * 180.0 / math.Pi
	if heading < 0 {
		heading += 360.0
	}
	return heading
}

// ParseNMEA parses a raw NMEA sentence
func ParseNMEA(sentence string) (*NMEAData, error) {
	// Simplified NMEA parser
	// In production, implement full NMEA parsing for GGA, RMC, VTG sentences
	return &NMEAData{
		Timestamp:  time.Now(),
		Latitude:   0.0,
		Longitude:  0.0,
		Altitude:   0.0,
		Velocity:   0.0,
		Heading:    0.0,
		Satellites: 0,
		HDOP:       0.0,
		VDOP:       0.0,
		FixQuality: 0,
		IsValid:    false,
	}, fmt.Errorf("NMEA parsing not implemented")
}

// GPSSpoofAlert represents a spoofing alert
type GPSSpoofAlert struct {
	AlertID           string
	Severity          string // LOW, MEDIUM, HIGH, CRITICAL
	AnomalyType       string
	PositionDiff      float64
	AltitudeDiff      float64
	VelocityDiff      float64
	Confidence        float64
	Timestamp         time.Time
	RecommendedAction string
}

// GenerateAlert generates a spoofing alert
func (gsd *GPSSpoofDetector) GenerateAlert(result *GPSComparisonResult) *GPSSpoofAlert {
	alertID := fmt.Sprintf("GPS-SPOOF-%d", time.Now().Unix())

	severity := "LOW"
	if result.SpoofingProbability > 0.9 {
		severity = "CRITICAL"
	} else if result.SpoofingProbability > 0.8 {
		severity = "HIGH"
	} else if result.SpoofingProbability > 0.7 {
		severity = "MEDIUM"
	}

	action := "Continue monitoring"
	if severity == "CRITICAL" {
		action = "IMMEDIATE: Switch to inertial navigation, air-gap systems"
	} else if severity == "HIGH" {
		action = "Alert operators, verify with secondary GPS"
	} else if severity == "MEDIUM" {
		action = "Increase monitoring frequency"
	}

	return &GPSSpoofAlert{
		AlertID:           alertID,
		Severity:          severity,
		AnomalyType:       result.AnomalyType,
		PositionDiff:      result.PositionDifferenceMeters,
		AltitudeDiff:      result.AltitudeDifferenceMeters,
		VelocityDiff:      result.VelocityDifferenceMps,
		Confidence:        result.Confidence,
		Timestamp:         time.Now(),
		RecommendedAction: action,
	}
}

// GetAnomalyCount returns the current anomaly count
func (gsd *GPSSpoofDetector) GetAnomalyCount() int {
	return gsd.anomalyCount
}

// ResetAlert resets the anomaly counter
func (gsd *GPSSpoofDetector) ResetAlert() {
	gsd.anomalyCount = 0
	gsd.lastAlertTime = time.Time{}
}

// GetNMEAHistory returns the NMEA data history
func (gsd *GPSSpoofDetector) GetNMEAHistory() []NMEAData {
	return gsd.nmeaHistory
}

// SetTolerance sets custom detection thresholds
func (gsd *GPSSpoofDetector) SetTolerance(position, velocity float64) {
	gsd.positionTolerance = position
	gsd.velocityTolerance = velocity
}

// SetAlertThreshold sets the number of anomalies before alert
func (gsd *GPSSpoofDetector) SetAlertThreshold(threshold int) {
	gsd.alertThreshold = threshold
}

// GNSSSignal represents GNSS signal characteristics for L1/L2 monitoring
type GNSSSignal struct {
	Band        string // L1, L2, L5
	FrequencyHz float64
	SignalPower float64 // dBm
	CN0         float64 // Carrier-to-noise density ratio
	IsSpoofed   bool
	SpoofScore  float64
}

// GNSSMonitor monitors GNSS signals for spoofing attempts
type GNSSMonitor struct {
	signals       map[string]*GNSSSignal
	baselinePower map[string]float64
}

// NewGNSSMonitor creates a new GNSS monitor
func NewGNSSMonitor() *GNSSMonitor {
	return &GNSSMonitor{
		signals:       make(map[string]*GNSSSignal),
		baselinePower: make(map[string]float64),
	}
}

// MonitorSignal monitors a GNSS signal for anomalies
func (gm *GNSSMonitor) MonitorSignal(band string, signal *GNSSSignal) {
	gm.signals[band] = signal

	// Check for power anomaly (sudden increase could indicate spoofing)
	if baseline, exists := gm.baselinePower[band]; exists {
		powerDiff := signal.SignalPower - baseline
		if powerDiff > 20.0 { // 20 dB increase
			signal.SpoofScore += 0.5
		}
	} else {
		gm.baselinePower[band] = signal.SignalPower
	}

	// Check CN0 (should be consistent across satellites)
	if signal.CN0 < 35.0 || signal.CN0 > 55.0 {
		signal.SpoofScore += 0.3
	}

	// Check if spoofing detected
	if signal.SpoofScore > 0.7 {
		signal.IsSpoofed = true
	}
}

// IsSpoofingDetected checks if any GNSS band is being spoofed
func (gm *GNSSMonitor) IsSpoofingDetected() bool {
	for _, signal := range gm.signals {
		if signal.IsSpoofed {
			return true
		}
	}
	return false
}

// GetSpoofedBands returns list of spoofed GNSS bands
func (gm *GNSSMonitor) GetSpoofedBands() []string {
	bands := make([]string, 0)
	for band, signal := range gm.signals {
		if signal.IsSpoofed {
			bands = append(bands, band)
		}
	}
	return bands
}
