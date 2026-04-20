package sensors

/**
 * Blind Spot Detection - Space-Time Inversion for Sovereign Ops
 *
 * ADITI 4.0 Point (e): Compute union of adversarial sensor footprints
 * and identify null set (gaps) where PoD < threshold (e.g., <5%).
 *
 * Physics Requirements:
 * - Accurate propagation (Vallado + NRLMSISE + SRP + third-body)
 * - Sensor physics (phase angle, layover, antenna steering)
 * - Realistic covariance (Cano method)
 * - 4D spatio-temporal indexing (BVH extended to time)
 *
 * References:
 * - Weeden (SWF 2012): Inverse SSA (find null coverage)
 * - Poore et al. (AFRL 2016): Covariance realism for gap analysis
 * - Zhang et al. (2018): NGSO spectrum blind spot probability
 * - CAPSS India (2025): Blind spot time slot identification for defence concealment
 */

import (
	"math"
	"time"

	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

// SensorsConfig holds configurable thresholds and parameters for sensor operations
type SensorsConfig struct {
	// Blind spot detection thresholds
	PoDThreshold float64 // Detection probability below which region is considered blind (default 0.05)

	// PoD engine parameters
	MinSNR        float64 // Minimum SNR threshold for detection (dB)
	ResolutionMin float64 // Minimum resolution requirement (meters)

	// Physics constants
	EarthRadius         float64 // Earth radius in km (default 6378.0)
	EarthShadowPenumbra float64 // Penumbra factor for Earth shadow (default 1.05)

	// Sun position
	EclipticObliquity float64 // Obliquity of ecliptic in degrees (default 23.44)
}

// DefaultSensorsConfig returns a configuration with sensible defaults
func DefaultSensorsConfig() SensorsConfig {
	return SensorsConfig{
		PoDThreshold:        0.05,   // 5% detection probability
		MinSNR:              10.0,   // 10 dB SNR
		ResolutionMin:       1.0,    // 1 meter
		EarthRadius:         6378.0, // km
		EarthShadowPenumbra: 1.05,   // 5% penumbra
		EclipticObliquity:   23.44,  // degrees
	}
}

var defaultConfig = DefaultSensorsConfig()

// BlindSpotWindow represents a temporal-spatial gap in adversary coverage
type BlindSpotWindow struct {
	StartTime  time.Time
	EndTime    time.Time
	Duration   time.Duration
	Region     SpatialRegion
	MaxPoD     float64     // Maximum detection probability during window
	SensorGaps []SensorGap // Which sensors can't see this region
	Confidence float64     // Based on covariance realism
}

// SpatialRegion defines a 3D volume (or surface patch)
type SpatialRegion struct {
	MinLat, MaxLat float64    // degrees
	MinLon, MaxLon float64    // degrees
	MinAlt, MaxAlt float64    // km
	Center         [3]float64 // ECI km (optional, for point regions)
	Radius         float64    // km (for spherical regions)
}

// IsPoint checks if region is a single point
func (r SpatialRegion) IsPoint() bool {
	return r.MinLat == r.MaxLat && r.MinLon == r.MaxLon && r.MinAlt == r.MaxAlt
}

// Contains checks if ECI position is within region
func (r SpatialRegion) Contains(pos [3]float64) bool {
	// Convert ECI to lat/lon/alt
	lat, lon, alt := ECIToLatLonAlt(pos)

	if lat < r.MinLat || lat > r.MaxLat {
		return false
	}
	if lon < r.MinLon || lon > r.MaxLon {
		return false
	}
	if alt < r.MinAlt || alt > r.MaxAlt {
		return false
	}
	return true
}

// ECIToLatLonAlt converts ECI position to geodetic coordinates
func ECIToLatLonAlt(pos [3]float64) (lat, lon, alt float64) {
	return ECIToLatLonAltWithConfig(pos, defaultConfig)
}

// ECIToLatLonAltWithConfig converts ECI position to geodetic coordinates with custom config
func ECIToLatLonAltWithConfig(pos [3]float64, config SensorsConfig) (lat, lon, alt float64) {
	// Simplified - assumes spherical Earth
	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	lat = math.Asin(pos[2]/r) * 180.0 / math.Pi
	lon = math.Atan2(pos[1], pos[0]) * 180.0 / math.Pi
	alt = r - config.EarthRadius
	return
}

// SensorGap identifies which adversary sensor has a gap
type SensorGap struct {
	PlatformID   string
	SensorType   string
	GapReason    GapReason
	MinRange     float64 // km (if too far)
	MaxElevation float64 // degrees (if below horizon)
}

// GapReason explains why sensor can't detect
type GapReason int

const (
	GapBelowHorizon GapReason = iota // Earth occlusion
	GapTooFar                        // Beyond sensor range
	GapShadowed                      // In Earth's shadow (optical)
	GapSunGlare                      // Solar phase angle too high
	GapLayover                       // SAR terrain shadow
	GapRadioHorizon                  // Below radio line of sight
	GapCovariance                    // Position uncertainty too high
)

// BlindSpotDetector is the main engine for gap analysis
type BlindSpotDetector struct {
	AdversaryAssets []*SensorPlatform // Enemy surveillance satellites
	TerrainModel    *TerrainDEM       // Digital elevation for SAR shadows
	PoDCalculator   *PoDEngine        // Probability of detection calculator
	Config          SensorsConfig     // Configuration parameters
}

// NewBlindSpotDetector creates a detector for adversary assets with default config
func NewBlindSpotDetector(assets []*SensorPlatform, terrain *TerrainDEM) *BlindSpotDetector {
	return NewBlindSpotDetectorWithConfig(assets, terrain, defaultConfig)
}

// NewBlindSpotDetectorWithConfig creates a detector with custom configuration
func NewBlindSpotDetectorWithConfig(assets []*SensorPlatform, terrain *TerrainDEM, config SensorsConfig) *BlindSpotDetector {
	return &BlindSpotDetector{
		AdversaryAssets: assets,
		TerrainModel:    terrain,
		PoDCalculator:   NewPoDEngineWithConfig(config),
		Config:          config,
	}
}

// FindBlindSpots identifies temporal windows where PoD < threshold for a region
func (bsd *BlindSpotDetector) FindBlindSpots(
	region SpatialRegion,
	startTime, endTime time.Time,
	params physics.PhysicsParams,
) []BlindSpotWindow {

	var windows []BlindSpotWindow

	// Time step: 1 minute resolution for gap analysis
	timeStep := 1 * time.Minute
	currentTime := startTime

	var currentWindow *BlindSpotWindow

	for currentTime.Before(endTime) {
		// Compute PoD at this instant for the region
		pod, gaps := bsd.computeRegionPoD(region, currentTime, params)

		if pod < bsd.Config.PoDThreshold {
			// We're in a blind spot
			if currentWindow == nil {
				// Start new window
				currentWindow = &BlindSpotWindow{
					StartTime:  currentTime,
					Region:     region,
					MaxPoD:     pod,
					SensorGaps: gaps,
				}
			} else {
				// Extend current window
				if pod > currentWindow.MaxPoD {
					currentWindow.MaxPoD = pod
				}
				// Merge gap reasons
				currentWindow.SensorGaps = mergeGaps(currentWindow.SensorGaps, gaps)
			}
		} else {
			// Not a blind spot - close current window if open
			if currentWindow != nil {
				currentWindow.EndTime = currentTime
				currentWindow.Duration = currentWindow.EndTime.Sub(currentWindow.StartTime)
				windows = append(windows, *currentWindow)
				currentWindow = nil
			}
		}

		currentTime = currentTime.Add(timeStep)
	}

	// Close final window if still open
	if currentWindow != nil {
		currentWindow.EndTime = endTime
		currentWindow.Duration = currentWindow.EndTime.Sub(currentWindow.StartTime)
		windows = append(windows, *currentWindow)
	}

	return windows
}

// computeRegionPoD calculates maximum detection probability over a region
// Returns PoD and list of contributing sensor gaps
func (bsd *BlindSpotDetector) computeRegionPoD(
	region SpatialRegion,
	t time.Time,
	params physics.PhysicsParams,
) (float64, []SensorGap) {

	jd := orbital.TimeToJulianDate(t)

	// For point regions, compute directly
	if region.IsPoint() || region.Radius < 100 {
		return bsd.computePointPoD(region.Center, jd, params)
	}

	// For extended regions, sample points and take maximum PoD
	// (Conservative: if any point is visible, region is not fully blind)

	samplePoints := generateSamplePoints(region, 9) // 3x3 grid

	maxPoD := 0.0
	var allGaps []SensorGap

	for _, point := range samplePoints {
		pod, gaps := bsd.computePointPoD(point, jd, params)
		if pod > maxPoD {
			maxPoD = pod
		}
		allGaps = append(allGaps, gaps...)
	}

	return maxPoD, deduplicateGaps(allGaps)
}

// computePointPoD calculates PoD for a specific ECI point at given time
func (bsd *BlindSpotDetector) computePointPoD(
	point [3]float64,
	jd float64,
	params physics.PhysicsParams,
) (float64, []SensorGap) {

	target := physics.State{Pos: point}

	// Sun position for illumination calculations
	sunPos := computeSunPositionECI(jd)

	// Compute per-sensor detection probability
	var pValues []float64
	var gaps []SensorGap

	for _, platform := range bsd.AdversaryAssets {
		// Propagate platform to current time
		platform.Propagate(jd, params)

		// Compute observation geometry
		geo := platform.ComputeObservationGeometry(target, sunPos)

		if !geo.Visible {
			// Sensor cannot see target - add to gap list
			gap := SensorGap{
				PlatformID:   platform.ID,
				SensorType:   platform.Sensors[0].Type.String(),
				GapReason:    determineGapReason(geo),
				MinRange:     geo.Range,
				MaxElevation: geo.Elevation,
			}
			gaps = append(gaps, gap)
			continue
		}

		// Compute individual sensor detection probability
		p := bsd.PoDCalculator.Compute(platform, geo, target)
		pValues = append(pValues, p)
	}

	// Aggregate PoD: PoD = 1 - prod(1 - p_i)
	if len(pValues) == 0 {
		return 0.0, gaps // No sensors can see target = complete blind spot
	}

	pod := 1.0
	for _, p := range pValues {
		pod *= (1.0 - p)
	}
	pod = 1.0 - pod

	return pod, gaps
}

// determineGapReason classifies why a sensor can't see the target
func determineGapReason(geo ObservationGeometry) GapReason {
	if geo.Elevation < 0 {
		return GapBelowHorizon
	}
	if geo.Range > 50000 {
		return GapTooFar
	}
	if !geo.Illuminated {
		return GapShadowed
	}
	if geo.PhaseAngle > 150 {
		return GapSunGlare
	}
	return GapCovariance // Default: visible but uncertainty too high
}

// computeSunPositionECI computes Sun position in ECI coordinates
func computeSunPositionECI(jd float64) [3]float64 {
	return computeSunPositionECIWithConfig(jd, defaultConfig)
}

// computeSunPositionECIWithConfig computes Sun position with custom config (includes ecliptic obliquity)
func computeSunPositionECIWithConfig(jd float64, config SensorsConfig) [3]float64 {
	// Sun position algorithm (accurate to ~0.01 AU)
	// Based on Vallado algorithm 24
	T := (jd - 2451545.0) / 36525.0 // Julian centuries from J2000

	// Mean longitude
	L0 := 280.46646 + 36000.76983*T + 0.0003032*T*T // degrees

	// Mean anomaly
	M := 357.52911 + 35999.05029*T - 0.0001537*T*T // degrees

	// Equation of center
	C := (1.914602-0.004817*T-0.000014*T*T)*math.Sin(M*math.Pi/180.0) +
		(0.019993-0.000101*T)*math.Sin(2*M*math.Pi/180.0) +
		0.000289*math.Sin(3*M*math.Pi/180.0)

	// True longitude (ecliptic)
	trueLonEcliptic := L0 + C // degrees

	// True anomaly
	trueAnomaly := M + C // degrees

	// Distance (AU)
	R := 1.000001018 * (1.0 - 0.016708617*0.016708617) /
		(1.0 + 0.016708617*math.Cos(trueAnomaly*math.Pi/180.0))

	// Convert to ECI using ecliptic obliquity
	eps := config.EclipticObliquity * math.Pi / 180.0 // Obliquity in radians
	trueLonRad := trueLonEcliptic * math.Pi / 180.0

	// Ecliptic coordinates
	xEcliptic := R * 149597870.7 * math.Cos(trueLonRad) // km
	yEcliptic := R * 149597870.7 * math.Sin(trueLonRad) // km
	zEcliptic := 0.0

	// Rotate to equatorial (ECI) using obliquity
	x := xEcliptic
	y := yEcliptic*math.Cos(eps) - zEcliptic*math.Sin(eps)
	z := yEcliptic*math.Sin(eps) + zEcliptic*math.Cos(eps)

	return [3]float64{x, y, z}
}

// generateSamplePoints creates grid of sample points within region
func generateSamplePoints(region SpatialRegion, n int) [][3]float64 {
	if region.IsPoint() || region.Radius > 0 {
		return [][3]float64{region.Center}
	}

	var points [][3]float64

	// Create n x n grid
	for i := 0; i < n; i++ {
		lat := region.MinLat + (region.MaxLat-region.MinLat)*float64(i)/float64(n-1)
		for j := 0; j < n; j++ {
			lon := region.MinLon + (region.MaxLon-region.MinLon)*float64(j)/float64(n-1)
			alt := (region.MinAlt + region.MaxAlt) / 2.0 // Mid-altitude

			// Convert lat/lon/alt to ECI (simplified)
			pos := LatLonAltToECI(lat, lon, alt)
			points = append(points, pos)
		}
	}

	return points
}

// LatLonAltToECI converts geodetic to ECI (simplified)
func LatLonAltToECI(lat, lon, alt float64) [3]float64 {
	return LatLonAltToECIWithConfig(lat, lon, alt, defaultConfig)
}

// LatLonAltToECIWithConfig converts geodetic to ECI with custom config
func LatLonAltToECIWithConfig(lat, lon, alt float64, config SensorsConfig) [3]float64 {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	// Earth radius + altitude
	r := config.EarthRadius + alt

	// ECI coordinates (assuming no Earth rotation offset)
	x := r * math.Cos(latRad) * math.Cos(lonRad)
	y := r * math.Cos(latRad) * math.Sin(lonRad)
	z := r * math.Sin(latRad)

	return [3]float64{x, y, z}
}

// mergeGaps combines gap lists
func mergeGaps(a, b []SensorGap) []SensorGap {
	result := make([]SensorGap, len(a))
	copy(result, a)

	// Add gaps from b that aren't already in a
	for _, gapB := range b {
		found := false
		for _, gapA := range a {
			if gapA.PlatformID == gapB.PlatformID && gapA.GapReason == gapB.GapReason {
				found = true
				break
			}
		}
		if !found {
			result = append(result, gapB)
		}
	}

	return result
}

// deduplicateGaps removes duplicate gaps
func deduplicateGaps(gaps []SensorGap) []SensorGap {
	seen := make(map[string]bool)
	var result []SensorGap

	for _, gap := range gaps {
		key := gap.PlatformID + "_" + string(rune(gap.GapReason))
		if !seen[key] {
			seen[key] = true
			result = append(result, gap)
		}
	}

	return result
}

// String returns human-readable gap reason
func (g GapReason) String() string {
	switch g {
	case GapBelowHorizon:
		return "Below Horizon"
	case GapTooFar:
		return "Beyond Range"
	case GapShadowed:
		return "Earth Shadow"
	case GapSunGlare:
		return "Sun Glare"
	case GapLayover:
		return "Terrain Shadow"
	case GapRadioHorizon:
		return "Radio Horizon"
	case GapCovariance:
		return "High Uncertainty"
	default:
		return "Unknown"
	}
}
