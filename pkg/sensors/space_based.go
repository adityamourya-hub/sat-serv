package sensors

/**
 * Space-Based SSA Sensor Modeling
 *
 * Models orbital sensor platforms for:
 * - GSSAP (Geosynchronous Space Situational Awareness Program)
 * - SBSS (Space-Based Space Surveillance)
 * - Commercial SSA satellites
 * - Adversarial systems (Andromeda RG-XX and similar)
 *
 * Key Capabilities:
 * - Self-propagation of sensor platform orbits
 * - Observation geometry from space-based platforms
 * - Multi-sensor fusion across distributed assets
 * - Coverage analysis for LEO/GEO/HEO surveillance
 */

import (
	"fmt"
	"math"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

// SensorPlatform represents a space-based SSA asset
type SensorPlatform struct {
	ID      string
	Name    string
	Type    PlatformType
	Nation  string
	Orbit   PlatformOrbit
	Sensors []SensorPayload
	EpochJD float64
	State   physics.State // Current ECI state
}

// PlatformType classification
type PlatformType int

const (
	PlatformGSSAP       PlatformType = iota // US DoD GEO surveillance
	PlatformSBSS                            // Space-Based Space Surveillance
	PlatformCommercial                      // Commercial SSA (NorthStar, etc.)
	PlatformAdversarial                     // Foreign military SSA (RG-XX, etc.)
	PlatformUnknown                         // Unclassified/unknown origin
)

// PlatformOrbit defines the sensor's orbital regime
type PlatformOrbit struct {
	Regime      OrbitRegime
	Altitude    float64 // km (circular approximation)
	Inclination float64 // degrees
	RAAN        float64 // degrees
	ArgPerigee  float64 // degrees
	MeanAnomaly float64 // degrees at epoch
}

// OrbitRegime classification
type OrbitRegime int

const (
	OrbitLEO       OrbitRegime = iota // LEO SSA (SBSS-like, 600-1000km)
	OrbitMEO                          // Medium Earth (rare for SSA)
	OrbitGEO                          // GEO surveillance (GSSAP, 35786km)
	OrbitHEO                          // HEO for high-latitude coverage
	OrbitLibration                    // L1/L2 for deep space surveillance
)

// SensorPayload defines the instruments on the platform
type SensorPayload struct {
	Type        SensorType
	FOV         float64 // Field of view (degrees)
	RangeLimit  float64 // Maximum detection range (km)
	Wavelength  WavelengthBand
	Sensitivity float64 // Magnitude limit or SNR threshold
}

// SensorType
type SensorType int

const (
	SensorOptical SensorType = iota // Visible/near-IR telescope
	SensorIR                        // Thermal infrared
	SensorRadar                     // Space-based radar (rare)
	SensorLidar                     // Laser ranging (experimental)
)

func (s SensorType) String() string {
	switch s {
	case SensorOptical:
		return "Optical"
	case SensorIR:
		return "IR"
	case SensorRadar:
		return "Radar"
	case SensorLidar:
		return "Lidar"
	default:
		return "Unknown"
	}
}

// WavelengthBand
type WavelengthBand int

const (
	BandVisible   WavelengthBand = iota // 400-700nm
	BandNearIR                          // 700-1400nm
	BandThermalIR                       // 3-5μm or 8-12μm
	BandRadarX                          // X-band radar
)

// Predefined Platforms (GSSAP, SBSS, etc.)

// GSSAP platforms - US DoD GEO surveillance satellites
func NewGSSAP1(epochJD float64) *SensorPlatform {
	// GSSAP 1: GEO, ~0.1° inclination drift for surveillance
	return &SensorPlatform{
		ID:     "GSSAP-1",
		Name:   "GSSAP 1 (USA-253)",
		Type:   PlatformGSSAP,
		Nation: "USA",
		Orbit: PlatformOrbit{
			Regime:      OrbitGEO,
			Altitude:    35786,
			Inclination: 0.1, // Near-equatorial for GEO surveillance
			RAAN:        0.0,
			ArgPerigee:  0.0,
			MeanAnomaly: 0.0,
		},
		Sensors: []SensorPayload{
			{
				Type:        SensorOptical,
				FOV:         2.0,   // Narrow FOV for detailed imaging
				RangeLimit:  50000, // GEO to GEO range
				Wavelength:  BandVisible,
				Sensitivity: 15.0, // Visual magnitude limit
			},
			{
				Type:        SensorIR,
				FOV:         2.0,
				RangeLimit:  50000,
				Wavelength:  BandThermalIR,
				Sensitivity: 12.0,
			},
		},
		EpochJD: epochJD,
	}
}

func NewGSSAP2(epochJD float64) *SensorPlatform {
	return &SensorPlatform{
		ID:     "GSSAP-2",
		Name:   "GSSAP 2 (USA-254)",
		Type:   PlatformGSSAP,
		Nation: "USA",
		Orbit: PlatformOrbit{
			Regime:      OrbitGEO,
			Altitude:    35786,
			Inclination: 0.1,
			RAAN:        120.0, // Separated for coverage
			ArgPerigee:  0.0,
			MeanAnomaly: 120.0,
		},
		Sensors: []SensorPayload{
			{
				Type:        SensorOptical,
				FOV:         2.0,
				RangeLimit:  50000,
				Wavelength:  BandVisible,
				Sensitivity: 15.0,
			},
		},
		EpochJD: epochJD,
	}
}

// SBSS - Space-Based Space Surveillance
// LEO platform for detecting dim LEO/GEO objects
func NewSBSS(epochJD float64) *SensorPlatform {
	return &SensorPlatform{
		ID:     "SBSS",
		Name:   "SBSS (USA-216)",
		Type:   PlatformSBSS,
		Nation: "USA",
		Orbit: PlatformOrbit{
			Regime:      OrbitLEO,
			Altitude:    630,  // ~630km sun-synchronous
			Inclination: 98.0, // SSO for consistent lighting
			RAAN:        0.0,
			ArgPerigee:  0.0,
			MeanAnomaly: 0.0,
		},
		Sensors: []SensorPayload{
			{
				Type:        SensorOptical,
				FOV:         60.0,  // Wide FOV for survey
				RangeLimit:  50000, // Can see GEO from LEO
				Wavelength:  BandVisible,
				Sensitivity: 19.0, // Very sensitive (dim objects)
			},
		},
		EpochJD: epochJD,
	}
}

// Commercial SSA - NorthStar, etc.
func NewNorthStarConstellation(epochJD float64) []*SensorPlatform {
	// NorthStar plans 24+ LEO satellites for SSA
	platforms := make([]*SensorPlatform, 3) // Sample first 3

	for i := 0; i < 3; i++ {
		platforms[i] = &SensorPlatform{
			ID:     fmt.Sprintf("NS-SSA-%02d", i+1),
			Name:   fmt.Sprintf("NorthStar SSA-%d", i+1),
			Type:   PlatformCommercial,
			Nation: "CAN/Commercial",
			Orbit: PlatformOrbit{
				Regime:      OrbitLEO,
				Altitude:    550, // Starlink-like altitude
				Inclination: 53.0,
				RAAN:        float64(i) * 120.0, // Distributed RAAN
				ArgPerigee:  0.0,
				MeanAnomaly: float64(i) * 60.0,
			},
			Sensors: []SensorPayload{
				{
					Type:        SensorOptical,
					FOV:         45.0,
					RangeLimit:  40000,
					Wavelength:  BandVisible,
					Sensitivity: 17.0,
				},
				{
					Type:        SensorIR,
					FOV:         30.0,
					RangeLimit:  30000,
					Wavelength:  BandThermalIR,
					Sensitivity: 14.0,
				},
			},
			EpochJD: epochJD,
		}
	}
	return platforms
}

// Adversarial Systems - Andromeda RG-XX and similar
// Russian/Chinese space-based surveillance
func NewAndromedaRGXX(epochJD float64) *SensorPlatform {
	// Hypothetical adversarial GEO surveillance
	return &SensorPlatform{
		ID:     "RG-XX-1",
		Name:   "Andromeda RG-XX ( suspected GEO surv )",
		Type:   PlatformAdversarial,
		Nation: "Unknown/Adversarial",
		Orbit: PlatformOrbit{
			Regime:      OrbitGEO,
			Altitude:    35786,
			Inclination: 0.2,
			RAAN:        180.0,
			ArgPerigee:  0.0,
			MeanAnomaly: 180.0,
		},
		Sensors: []SensorPayload{
			{
				Type:        SensorOptical,
				FOV:         3.0,
				RangeLimit:  50000,
				Wavelength:  BandVisible,
				Sensitivity: 14.0,
			},
		},
		EpochJD: epochJD,
	}
}

// ObservationGeometry computes visibility from sensor platform to target
type ObservationGeometry struct {
	Range       float64 // km
	RangeRate   float64 // km/s
	PhaseAngle  float64 // degrees (sun-target-observer)
	SolarAngle  float64 // degrees (sun-observer-target)
	Elevation   float64 // degrees above platform local horizon
	Azimuth     float64 // degrees from platform velocity vector
	Visible     bool    // Above horizon and in range
	Illuminated bool    // Target in sunlight
}

// ComputeObservationGeometry determines if target is visible from platform
func (p *SensorPlatform) ComputeObservationGeometry(target physics.State, sunPos [3]float64) ObservationGeometry {
	// Platform to target vector
	dx := target.Pos[0] - p.State.Pos[0]
	dy := target.Pos[1] - p.State.Pos[1]
	dz := target.Pos[2] - p.State.Pos[2]

	range_km := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Platform velocity for azimuth reference
	vMag := math.Sqrt(p.State.Vel[0]*p.State.Vel[0] + p.State.Vel[1]*p.State.Vel[1] + p.State.Vel[2]*p.State.Vel[2])

	// Local zenith (radial outward)
	platformRadius := math.Sqrt(p.State.Pos[0]*p.State.Pos[0] + p.State.Pos[1]*p.State.Pos[1] + p.State.Pos[2]*p.State.Pos[2])
	zenith := [3]float64{p.State.Pos[0] / platformRadius, p.State.Pos[1] / platformRadius, p.State.Pos[2] / platformRadius}

	// LOS unit vector
	los := [3]float64{dx / range_km, dy / range_km, dz / range_km}

	// Elevation above local horizon (90° - angle from zenith)
	cosZenithAngle := los[0]*zenith[0] + los[1]*zenith[1] + los[2]*zenith[2]
	elevation := 90.0 - math.Acos(cosZenithAngle)*180.0/math.Pi

	// Azimuth (angle from velocity vector in local horizontal plane)
	// Simplified: project LOS onto horizontal plane
	horizontalLos := [3]float64{
		los[0] - cosZenithAngle*zenith[0],
		los[1] - cosZenithAngle*zenith[1],
		los[2] - cosZenithAngle*zenith[2],
	}
	// Normalize horizontal
	hMag := math.Sqrt(horizontalLos[0]*horizontalLos[0] + horizontalLos[1]*horizontalLos[1] + horizontalLos[2]*horizontalLos[2])
	if hMag > 0 {
		horizontalLos[0] /= hMag
		horizontalLos[1] /= hMag
		horizontalLos[2] /= hMag
	}

	// Velocity unit vector
	vUnit := [3]float64{p.State.Vel[0] / vMag, p.State.Vel[1] / vMag, p.State.Vel[2] / vMag}
	// Project velocity onto horizontal plane
	vDotZenith := vUnit[0]*zenith[0] + vUnit[1]*zenith[1] + vUnit[2]*zenith[2]
	horizontalV := [3]float64{
		vUnit[0] - vDotZenith*zenith[0],
		vUnit[1] - vDotZenith*zenith[1],
		vUnit[2] - vDotZenith*zenith[2],
	}
	hvMag := math.Sqrt(horizontalV[0]*horizontalV[0] + horizontalV[1]*horizontalV[1] + horizontalV[2]*horizontalV[2])
	if hvMag > 0 {
		horizontalV[0] /= hvMag
		horizontalV[1] /= hvMag
		horizontalV[2] /= hvMag
	}

	// Azimuth angle
	azimuth := math.Acos(horizontalLos[0]*horizontalV[0]+horizontalLos[1]*horizontalV[1]+horizontalLos[2]*horizontalV[2]) * 180.0 / math.Pi

	// Range rate
	relVel := [3]float64{
		target.Vel[0] - p.State.Vel[0],
		target.Vel[1] - p.State.Vel[1],
		target.Vel[2] - p.State.Vel[2],
	}
	rangeRate := los[0]*relVel[0] + los[1]*relVel[1] + los[2]*relVel[2]

	// Phase angle (sun-target-observer angle)
	sunToTarget := [3]float64{target.Pos[0] - sunPos[0], target.Pos[1] - sunPos[1], target.Pos[2] - sunPos[2]}
	sunToTargetMag := math.Sqrt(sunToTarget[0]*sunToTarget[0] + sunToTarget[1]*sunToTarget[1] + sunToTarget[2]*sunToTarget[2])
	targetToObserver := [3]float64{-dx, -dy, -dz}

	cosPhase := (sunToTarget[0]*targetToObserver[0] + sunToTarget[1]*targetToObserver[1] + sunToTarget[2]*targetToObserver[2]) / (sunToTargetMag * range_km)
	phaseAngle := math.Acos(cosPhase) * 180.0 / math.Pi

	// Solar angle (sun-observer-target)
	observerToSun := [3]float64{sunPos[0] - p.State.Pos[0], sunPos[1] - p.State.Pos[1], sunPos[2] - p.State.Pos[2]}
	observerToSunMag := math.Sqrt(observerToSun[0]*observerToSun[0] + observerToSun[1]*observerToSun[1] + observerToSun[2]*observerToSun[2])

	cosSolar := (observerToSun[0]*dx + observerToSun[1]*dy + observerToSun[2]*dz) / (observerToSunMag * range_km)
	solarAngle := math.Acos(cosSolar) * 180.0 / math.Pi

	// Illumination check (target in sunlight = not in Earth's shadow)
	illuminated := IsIlluminated(target.Pos, sunPos)

	// Visibility check
	visible := elevation > 0 && range_km < 50000 // Basic horizon + range check

	return ObservationGeometry{
		Range:       range_km,
		RangeRate:   rangeRate,
		PhaseAngle:  phaseAngle,
		SolarAngle:  solarAngle,
		Elevation:   elevation,
		Azimuth:     azimuth,
		Visible:     visible,
		Illuminated: illuminated,
	}
}

// IsIlluminated checks if point is in Earth's shadow
func IsIlluminated(pos [3]float64, sunPos [3]float64) bool {
	// Simple cylindrical shadow model
	// Earth radius ~6378 km, shadow cylinder extends ~1.4M km

	// Vector from sun to point
	dx := pos[0] - sunPos[0]
	dy := pos[1] - sunPos[1]
	dz := pos[2] - sunPos[2]

	// Sun to Earth center vector
	sx := -sunPos[0]
	sy := -sunPos[1]
	sz := -sunPos[2]
	sunDist := math.Sqrt(sx*sx + sy*sy + sz*sz)

	// Project point onto sun-Earth line
	dot := dx*sx + dy*sy + dz*sz
	if dot < 0 {
		// Point is sunward of Earth, definitely illuminated
		return true
	}

	// Distance from point to sun-Earth line
	projFactor := dot / (sunDist * sunDist)
	closestX := sunPos[0] + sx*projFactor
	closestY := sunPos[1] + sy*projFactor
	closestZ := sunPos[2] + sz*projFactor

	distToLine := math.Sqrt((pos[0]-closestX)*(pos[0]-closestX) +
		(pos[1]-closestY)*(pos[1]-closestY) +
		(pos[2]-closestZ)*(pos[2]-closestZ))

	// Earth shadow radius (including penumbra)
	shadowRadius := 6378.0 * 1.05 // ~1% penumbra

	return distToLine > shadowRadius
}

// Propagate updates the sensor platform's state to a new epoch
func (p *SensorPlatform) Propagate(targetJD float64, params physics.PhysicsParams) {
	dt := (targetJD - p.EpochJD) * 86400.0 // seconds

	state := []float64{
		p.State.Pos[0], p.State.Pos[1], p.State.Pos[2],
		p.State.Vel[0], p.State.Vel[1], p.State.Vel[2],
	}

	newState := orbital.PropagateCartesianState(state, dt, p.EpochJD, params)

	p.State.Pos = [3]float64{newState[0], newState[1], newState[2]}
	p.State.Vel = [3]float64{newState[3], newState[4], newState[5]}
	p.EpochJD = targetJD
}

// MultiPlatformFusion aggregates observations from multiple space-based sensors
type MultiPlatformFusion struct {
	Platforms []*SensorPlatform
}

// NewMultiPlatformFusion creates fusion engine
func NewMultiPlatformFusion(platforms []*SensorPlatform) *MultiPlatformFusion {
	return &MultiPlatformFusion{Platforms: platforms}
}

// ObserveTarget attempts observation from all platforms, returns best geometry
func (mpf *MultiPlatformFusion) ObserveTarget(target physics.State, sunPos [3]float64) (bestGeo ObservationGeometry, bestPlatform *SensorPlatform, obsCount int) {
	bestRange := 1e9
	obsCount = 0

	for _, platform := range mpf.Platforms {
		geo := platform.ComputeObservationGeometry(target, sunPos)

		if geo.Visible {
			obsCount++
			if geo.Range < bestRange {
				bestRange = geo.Range
				bestGeo = geo
				bestPlatform = platform
			}
		}
	}

	return bestGeo, bestPlatform, obsCount
}

// CoverageAnalysis provides statistics on sensor coverage
type CoverageAnalysis struct {
	LEOCoverage  float64 // % of LEO objects visible
	GEOCoverage  float64 // % of GEO objects visible
	AverageRange float64 // km
	RevisitTime  float64 // hours (estimated)
	simultaneity int     // max simultaneous observations
}

// AnalyzeCoverage computes coverage statistics
func (mpf *MultiPlatformFusion) AnalyzeCoverage(sampleTargets []physics.State, sunPos [3]float64) CoverageAnalysis {
	visibleCount := 0
	totalRange := 0.0
	maxSimultaneous := 0

	for _, target := range sampleTargets {
		_, _, count := mpf.ObserveTarget(target, sunPos)
		if count > 0 {
			visibleCount++
		}
		if count > maxSimultaneous {
			maxSimultaneous = count
		}
		totalRange += 40000.0 // Approximate average
	}

	avgRange := totalRange / float64(len(sampleTargets))
	coverage := float64(visibleCount) / float64(len(sampleTargets))

	// Estimate revisit based on platform count and orbit periods
	// LEO: ~90 min period, GEO: 24h period
	revisitEstimate := 24.0 / float64(len(mpf.Platforms))
	if revisitEstimate < 1.0 {
		revisitEstimate = 1.0
	}

	return CoverageAnalysis{
		LEOCoverage:  coverage,
		GEOCoverage:  coverage * 0.8, // GEO slightly less due to geometry
		AverageRange: avgRange,
		RevisitTime:  revisitEstimate,
		simultaneity: maxSimultaneous,
	}
}
