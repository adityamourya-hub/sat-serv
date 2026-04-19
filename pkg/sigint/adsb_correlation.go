// Package sigint provides ADS-B correlation for terrestrial clutter rejection
// Distinguishes between space threats and terrestrial aircraft
package sigint

import (
	"fmt"
	"math"
	"time"
)

const (
	// ADS-B correlation thresholds
	AzimuthThresholdDegrees = 5.0
	RangeThresholdMeters    = 5000.0
	AltitudeThresholdMeters = 1000.0
	VelocityThresholdMps    = 100.0
	CorrelationThreshold    = 0.7
)

// ADSBReport represents a decoded ADS-B message
type ADSBReport struct {
	ICAO      string  // Hex ICAO address
	Callsign  string  // Flight callsign
	Latitude  float64 // Degrees
	Longitude float64 // Degrees
	Altitude  float64 // Meters
	Velocity  float64 // m/s
	Heading   float64 // Degrees
	Squawk    int     // Transponder code
	Timestamp time.Time
	IsValid   bool
}

// SpaceDetection represents a space-based detection from S2 filter
type SpaceDetection struct {
	ObjectID   string
	Azimuth    float64 // Degrees from observer
	Elevation  float64 // Degrees
	Range      float64 // Meters
	Velocity   float64 // m/s
	Confidence float64
	Timestamp  time.Time
}

// CorrelationResult represents the result of ADSB correlation
type CorrelationResult struct {
	IsCorrelated       bool
	ADSBCandidate      *ADSBReport
	CorrelationScore   float64
	AzimuthDifference  float64
	RangeDifference    float64
	AltitudeDifference float64
	IsTerrestrial      bool
	Recommendation     string
}

// ADSBCorrelator performs ADS-B correlation with space detections
type ADSBCorrelator struct {
	adsbReports        map[string]*ADSBReport
	correlationHistory map[string]int
	maxReportAge       time.Duration
}

// NewADSBCorrelator creates a new ADS-B correlator
func NewADSBCorrelator() *ADSBCorrelator {
	return &ADSBCorrelator{
		adsbReports:        make(map[string]*ADSBReport),
		correlationHistory: make(map[string]int),
		maxReportAge:       30 * time.Second,
	}
}

// UpdateADSBDataset updates the ADS-B dataset from feed
func (ac *ADSBCorrelator) UpdateADSBDataset(reports []ADSBReport) {
	now := time.Now()

	for _, report := range reports {
		// Validate and update
		if report.IsValid {
			ac.adsbReports[report.ICAO] = &report
		}
	}

	// Remove old reports
	for icao, report := range ac.adsbReports {
		if now.Sub(report.Timestamp) > ac.maxReportAge {
			delete(ac.adsbReports, icao)
		}
	}
}

// CorrelateWithSpaceDetection correlates a space detection with ADS-B data
func (ac *ADSBCorrelator) CorrelateWithSpaceDetection(detection *SpaceDetection) *CorrelationResult {
	result := &CorrelationResult{
		IsCorrelated:     false,
		ADSBCandidate:    nil,
		CorrelationScore: 0.0,
		IsTerrestrial:    false,
		Recommendation:   "NO_CORRELATION",
	}

	// Calculate expected terrestrial position from detection
	expectedLat, expectedLon, expectedAlt := ac.calculateExpectedPosition(detection)

	// Find matching ADS-B report
	for _, report := range ac.adsbReports {
		// Check temporal proximity
		timeDiff := math.Abs(detection.Timestamp.Sub(report.Timestamp).Seconds())
		if timeDiff > 5.0 {
			continue
		}

		// Calculate position differences
		altDiff := math.Abs(expectedAlt - report.Altitude)

		// Convert to meters for range
		rangeDiff := ac.calculateRangeDifference(
			expectedLat, expectedLon,
			report.Latitude, report.Longitude,
		)

		// Calculate velocity difference
		velDiff := math.Abs(detection.Velocity - report.Velocity)

		// Calculate correlation score
		azimuthDiff := math.Abs(detection.Azimuth - ac.calculateAzimuth(report))

		score := ac.calculateCorrelationScore(
			azimuthDiff, rangeDiff, altDiff, velDiff,
		)

		// Check if correlation exceeds threshold
		if score > CorrelationThreshold {
			result.IsCorrelated = true
			result.ADSBCandidate = report
			result.CorrelationScore = score
			result.AzimuthDifference = azimuthDiff
			result.RangeDifference = rangeDiff
			result.AltitudeDifference = altDiff
			result.IsTerrestrial = true
			result.Recommendation = fmt.Sprintf("TERRESTRIAL: Aircraft %s (%s)",
				report.Callsign, report.ICAO)
			break
		}
	}

	return result
}

// calculateExpectedPosition calculates expected terrestrial position from space detection
func (ac *ADSBCorrelator) calculateExpectedPosition(detection *SpaceDetection) (lat, lon, alt float64) {
	// Simplified: Assume observer at Siliguri coordinates
	observerLat := 26.5
	observerLon := 88.4

	// Calculate position from azimuth, elevation, range
	// This is a simplified calculation
	// In production, use full spherical trigonometry

	elevationRad := detection.Elevation * math.Pi / 180.0
	azimuthRad := detection.Azimuth * math.Pi / 180.0

	// Calculate horizontal range
	horizontalRange := detection.Range * math.Cos(elevationRad)

	// Calculate lat/lon offset
	latOffset := (horizontalRange / EarthRadiusMeters) * math.Cos(azimuthRad) * 180.0 / math.Pi
	lonOffset := (horizontalRange / EarthRadiusMeters) * math.Sin(azimuthRad) * 180.0 / math.Pi / math.Cos(observerLat*math.Pi/180.0)

	lat = observerLat + latOffset
	lon = observerLon + lonOffset
	alt = detection.Range * math.Sin(elevationRad)

	return lat, lon, alt
}

// calculateRangeDifference calculates range difference between two positions
func (ac *ADSBCorrelator) calculateRangeDifference(lat1, lon1, lat2, lon2 float64) float64 {
	return haversineDistance(lat1, lon1, lat2, lon2)
}

// calculateAzimuth calculates azimuth from observer to aircraft
func (ac *ADSBCorrelator) calculateAzimuth(report *ADSBReport) float64 {
	// Simplified azimuth calculation
	// In production, use full spherical trigonometry
	observerLat := 26.5
	observerLon := 88.4

	deltaLon := report.Longitude - observerLon
	lat1 := observerLat * math.Pi / 180.0
	lat2 := report.Latitude * math.Pi / 180.0
	dlon := deltaLon * math.Pi / 180.0

	x := math.Sin(dlon) * math.Cos(lat2)
	y := math.Cos(lat1)*math.Sin(lat2) - math.Sin(lat1)*math.Cos(lat2)*math.Cos(dlon)

	azimuth := math.Atan2(x, y) * 180.0 / math.Pi
	if azimuth < 0 {
		azimuth += 360.0
	}

	return azimuth
}

// calculateCorrelationScore calculates correlation score
func (ac *ADSBCorrelator) calculateCorrelationScore(azimuthDiff, rangeDiff, altDiff, velDiff float64) float64 {
	// Normalize differences to 0-1 range
	azimuthScore := 1.0 - math.Min(azimuthDiff/AzimuthThresholdDegrees, 1.0)
	rangeScore := 1.0 - math.Min(rangeDiff/RangeThresholdMeters, 1.0)
	altitudeScore := 1.0 - math.Min(altDiff/AltitudeThresholdMeters, 1.0)
	velocityScore := 1.0 - math.Min(velDiff/VelocityThresholdMps, 1.0)

	// Weighted average
	return (azimuthScore*0.3 + rangeScore*0.3 + altitudeScore*0.2 + velocityScore*0.2)
}

// NegativeCorrelationCheck performs negative correlation check
// If space detection correlates with ADS-B, it's likely terrestrial clutter
func (ac *ADSBCorrelator) NegativeCorrelationCheck(detection *SpaceDetection) (bool, *CorrelationResult) {
	result := ac.CorrelateWithSpaceDetection(detection)

	if result.IsCorrelated {
		return true, result
	}

	return false, result
}

// GetADSBDataset returns current ADS-B dataset
func (ac *ADSBCorrelator) GetADSBDataset() map[string]*ADSBReport {
	return ac.adsbReports
}

// GetReportCount returns number of ADS-B reports
func (ac *ADSBCorrelator) GetReportCount() int {
	return len(ac.adsbReports)
}

// ClearOldReports removes reports older than max age
func (ac *ADSBCorrelator) ClearOldReports() int {
	now := time.Now()
	removed := 0

	for icao, report := range ac.adsbReports {
		if now.Sub(report.Timestamp) > ac.maxReportAge {
			delete(ac.adsbReports, icao)
			removed++
		}
	}

	return removed
}

// SetMaxReportAge sets the maximum age for ADS-B reports
func (ac *ADSBCorrelator) SetMaxReportAge(age time.Duration) {
	ac.maxReportAge = age
}

// UCTAlert represents an Uncorrelated Track alert
type UCTAlert struct {
	AlertID        string
	ObjectID       string
	Azimuth        float64
	Elevation      float64
	Range          float64
	Velocity       float64
	Confidence     float64
	Timestamp      time.Time
	IsRapid        bool
	Recommendation string
}

// GenerateUCTAlert generates an Uncorrelated Track alert
func (ac *ADSBCorrelator) GenerateUCTAlert(detection *SpaceDetection) *UCTAlert {
	result := ac.CorrelateWithSpaceDetection(detection)

	alertID := fmt.Sprintf("UCT-%d", time.Now().Unix())

	if !result.IsCorrelated {
		// Uncorrelated - potential space threat
		recommendation := "INVESTIGATE: Uncorrelated track - potential space object"
		isRapid := detection.Velocity > 1000.0 // > 1 km/s is definitely space

		if isRapid {
			recommendation = "RAPID: High-velocity uncorrelated track - ASAT launch possible"
		}

		return &UCTAlert{
			AlertID:        alertID,
			ObjectID:       detection.ObjectID,
			Azimuth:        detection.Azimuth,
			Elevation:      detection.Elevation,
			Range:          detection.Range,
			Velocity:       detection.Velocity,
			Confidence:     detection.Confidence,
			Timestamp:      time.Now(),
			IsRapid:        isRapid,
			Recommendation: recommendation,
		}
	} else {
		// Correlated - terrestrial clutter
		return &UCTAlert{
			AlertID:        alertID,
			ObjectID:       detection.ObjectID,
			Azimuth:        detection.Azimuth,
			Elevation:      detection.Elevation,
			Range:          detection.Range,
			Velocity:       detection.Velocity,
			Confidence:     0.0, // Downgraded confidence
			Timestamp:      time.Now(),
			IsRapid:        false,
			Recommendation: fmt.Sprintf("TERRESTRIAL: Correlated with aircraft %s", result.ADSBCandidate.Callsign),
		}
	}
}

// ADSBFeed represents an ADS-B data feed interface
type ADSBFeed struct {
	source     string
	port       int
	isActive   bool
	lastUpdate time.Time
}

// NewADSBFeed creates a new ADS-B feed
func NewADSBFeed(source string, port int) *ADSBFeed {
	return &ADSBFeed{
		source:     source,
		port:       port,
		isActive:   false,
		lastUpdate: time.Time{},
	}
}

// Start starts the ADS-B feed
func (af *ADSBFeed) Start() error {
	// TODO: Implement connection to dump1090 or rtl_adsb
	// For now, mark as active
	af.isActive = true
	af.lastUpdate = time.Now()
	return nil
}

// Stop stops the ADS-B feed
func (af *ADSBFeed) Stop() {
	af.isActive = false
}

// IsRunning checks if the feed is running
func (af *ADSBFeed) IsRunning() bool {
	return af.isActive
}

// ParseADSBSentence parses an ADS-B message (simplified)
func ParseADSBSentence(sentence string) (*ADSBReport, error) {
	// TODO: Implement full ADS-B message parsing
	// For now, return empty report
	return &ADSBReport{
		Timestamp: time.Now(),
		IsValid:   false,
	}, fmt.Errorf("ADS-B parsing not implemented")
}

// ADSBStatistics represents ADS-B feed statistics
type ADSBStatistics struct {
	TotalReports   int64
	ValidReports   int64
	InvalidReports int64
	AircraftSeen   int64
	LastReportTime time.Time
	FeedUptime     time.Duration
}

// GetStatistics returns feed statistics
func (ac *ADSBCorrelator) GetStatistics() *ADSBStatistics {
	validCount := 0
	for _, report := range ac.adsbReports {
		if report.IsValid {
			validCount++
		}
	}

	return &ADSBStatistics{
		TotalReports:   int64(len(ac.adsbReports)),
		ValidReports:   int64(validCount),
		InvalidReports: int64(len(ac.adsbReports) - validCount),
		AircraftSeen:   int64(len(ac.adsbReports)),
		LastReportTime: time.Now(),
		FeedUptime:     0, // TODO: track actual uptime
	}
}
