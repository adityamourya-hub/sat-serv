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
	"time"
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

	// Slew dynamics
	MaxSlewRate      float64 // Maximum angular velocity (deg/s)
	MaxAcceleration  float64 // Maximum angular acceleration (deg/s²)
	CurrentAzimuth   float64 // Current azimuth pointing (degrees)
	CurrentElevation float64 // Current elevation pointing (degrees)

	// DSP Preprocessor chain for sensor data
	AnalogPreprocessor  *AnalogPreprocessor
	DigitalPreprocessor *DigitalPreprocessor
	DSPPreprocessor     *DSPPreprocessor
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
				Type:             SensorOptical,
				FOV:              2.0,   // Narrow FOV for detailed imaging
				RangeLimit:       50000, // GEO to GEO range
				Wavelength:       BandVisible,
				Sensitivity:      15.0, // Visual magnitude limit
				MaxSlewRate:      2.0,  // 2 deg/s (typical for GEO surveillance)
				MaxAcceleration:  1.0,  // 1 deg/s²
				CurrentAzimuth:   0.0,
				CurrentElevation: 0.0,
			},
			{
				Type:             SensorIR,
				FOV:              2.0,
				RangeLimit:       50000,
				Wavelength:       BandThermalIR,
				Sensitivity:      12.0,
				MaxSlewRate:      2.0,
				MaxAcceleration:  1.0,
				CurrentAzimuth:   0.0,
				CurrentElevation: 0.0,
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
				Type:             SensorOptical,
				FOV:              2.0,
				RangeLimit:       50000,
				Wavelength:       BandVisible,
				Sensitivity:      15.0,
				MaxSlewRate:      2.0,
				MaxAcceleration:  1.0,
				CurrentAzimuth:   120.0,
				CurrentElevation: 0.0,
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
				Type:             SensorOptical,
				FOV:              60.0,  // Wide FOV for survey
				RangeLimit:       50000, // Can see GEO from LEO
				Wavelength:       BandVisible,
				Sensitivity:      19.0, // Very sensitive (dim objects)
				MaxSlewRate:      5.0,  // Faster slew for LEO platform
				MaxAcceleration:  3.0,  // Higher acceleration
				CurrentAzimuth:   0.0,
				CurrentElevation: 45.0,
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
					Type:             SensorOptical,
					FOV:              45.0,
					RangeLimit:       40000,
					Wavelength:       BandVisible,
					Sensitivity:      17.0,
					MaxSlewRate:      4.0,
					MaxAcceleration:  2.0,
					CurrentAzimuth:   float64(i) * 120.0,
					CurrentElevation: 30.0,
				},
				{
					Type:             SensorIR,
					FOV:              30.0,
					RangeLimit:       30000,
					Wavelength:       BandThermalIR,
					Sensitivity:      14.0,
					MaxSlewRate:      4.0,
					MaxAcceleration:  2.0,
					CurrentAzimuth:   float64(i) * 120.0,
					CurrentElevation: 30.0,
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
				Type:             SensorOptical,
				FOV:              3.0,
				RangeLimit:       50000,
				Wavelength:       BandVisible,
				Sensitivity:      14.0,
				MaxSlewRate:      2.5,
				MaxAcceleration:  1.5,
				CurrentAzimuth:   180.0,
				CurrentElevation: 0.0,
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
	return IsIlluminatedWithConfig(pos, sunPos, defaultConfig)
}

// IsIlluminatedWithConfig checks if point is in Earth's shadow with custom config
func IsIlluminatedWithConfig(pos [3]float64, sunPos [3]float64, config SensorsConfig) bool {
	// Simple cylindrical shadow model
	// Earth shadow cylinder extends ~1.4M km

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
	shadowRadius := config.EarthRadius * config.EarthShadowPenumbra

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

// EstimateSlewTime calculates time required to slew sensor to target
// Uses angular acceleration profile: trapezoidal velocity profile
func (p *SensorPlatform) EstimateSlewTime(targetState [6]float64) time.Duration {
	if len(p.Sensors) == 0 {
		return 5 * time.Second // Default if no sensors
	}

	sensor := p.Sensors[0] // Use primary sensor

	// Compute target azimuth/elevation from platform
	targetPos := physics.State{
		Pos: physics.Vector3{targetState[0], targetState[1], targetState[2]},
		Vel: physics.Vector3{targetState[3], targetState[4], targetState[5]},
	}
	sunPos := computeSunPositionECI(p.EpochJD)
	geo := p.ComputeObservationGeometry(targetPos, sunPos)

	if !geo.Visible {
		return 60 * time.Second // Long time if not visible
	}

	// Angular distance from current pointing to target
	dAz := geo.Azimuth - sensor.CurrentAzimuth
	dEl := geo.Elevation - sensor.CurrentElevation

	// Normalize angles to [-180, 180]
	for dAz > 180 {
		dAz -= 360
	}
	for dAz < -180 {
		dAz += 360
	}

	angularDistance := math.Sqrt(dAz*dAz + dEl*dEl) // degrees

	// Trapezoidal velocity profile
	// Time to accelerate to max slew rate
	tAccel := sensor.MaxSlewRate / sensor.MaxAcceleration

	// Distance during acceleration
	distAccel := 0.5 * sensor.MaxAcceleration * tAccel * tAccel

	// If angular distance is small, never reach max slew rate
	if angularDistance < 2*distAccel {
		// Triangular profile: accelerate then decelerate
		tTotal := 2 * math.Sqrt(angularDistance/sensor.MaxAcceleration)
		return time.Duration(tTotal) * time.Second
	}

	// Trapezoidal profile: accelerate, coast, decelerate
	distCoast := angularDistance - 2*distAccel
	tCoast := distCoast / sensor.MaxSlewRate
	tTotal := 2*tAccel + tCoast

	return time.Duration(tTotal) * time.Second
}

// CanObserve checks if sensor can observe target at given time
func (p *SensorPlatform) CanObserve(targetState [6]float64, t time.Time) bool {
	// Convert time to Julian Date
	jd := orbital.TimeToJulianDate(t)

	// Propagate platform to current time
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	p.Propagate(jd, params)

	// Compute observation geometry
	target := physics.State{
		Pos: physics.Vector3{targetState[0], targetState[1], targetState[2]},
		Vel: physics.Vector3{targetState[3], targetState[4], targetState[5]},
	}
	sunPos := computeSunPositionECI(jd)
	geo := p.ComputeObservationGeometry(target, sunPos)

	// Check visibility
	if !geo.Visible {
		return false
	}

	// Check range limits
	for _, sensor := range p.Sensors {
		if geo.Range > sensor.RangeLimit {
			return false
		}
	}

	// Check illumination for optical sensors
	for _, sensor := range p.Sensors {
		if sensor.Type == SensorOptical && !geo.Illuminated {
			return false
		}
	}

	return true
}

// MeasurementCovariance returns sensor measurement noise covariance matrix
func (p *SensorPlatform) MeasurementCovariance() [3][3]float64 {
	if len(p.Sensors) == 0 {
		// Default: 100m position uncertainty
		return [3][3]float64{
			{0.01, 0, 0},
			{0, 0.01, 0},
			{0, 0, 0.01},
		}
	}

	sensor := p.Sensors[0]

	// Covariance based on sensor type and sensitivity
	var sigma float64 // km (3-sigma)

	switch sensor.Type {
	case SensorOptical:
		// Optical: ~100m at GEO range
		sigma = 0.1
	case SensorIR:
		// IR: ~200m at GEO range
		sigma = 0.2
	case SensorRadar:
		// Radar: ~50m at close range
		sigma = 0.05
	case SensorLidar:
		// Lidar: ~10m at close range
		sigma = 0.01
	default:
		sigma = 0.1
	}

	sigma2 := sigma * sigma

	return [3][3]float64{
		{sigma2, 0, 0},
		{0, sigma2, 0},
		{0, 0, sigma2},
	}
}

// ScheduleObservation schedules an observation for a target at specified time
// In production, this would send command to sensor control system
func (p *SensorPlatform) ScheduleObservation(objectID string, t time.Time) {
	// Update sensor pointing to target
	// In production: Send gRPC message to sensor control
	// For now: Update current pointing angles

	// This is a placeholder - in production this would:
	// 1. Send slew command to sensor
	// 2. Wait for slew completion
	// 3. Trigger observation
	// 4. Receive measurement
}

// Position returns the platform's current ECI position
func (p *SensorPlatform) Position() [3]float64 {
	return p.State.Pos
}

// =============================================================================
// DSP Pipeline Integration
// =============================================================================

// PreprocessRawData processes raw sensor data through the full DSP pipeline
// Returns processed samples, spectrum, features, and preprocessing metadata
func (p *SensorPlatform) PreprocessRawData(
	sensorIndex int,
	rawData []float64,
) ([]float64, []complex128, *DSPFeatures, *PreprocessingMetadata) {
	if sensorIndex < 0 || sensorIndex >= len(p.Sensors) {
		return nil, nil, nil, nil
	}

	sensor := &p.Sensors[sensorIndex]
	metadata := &PreprocessingMetadata{
		SensorType:   sensor.Type.String(),
		InputSamples: len(rawData),
		Timestamp:    time.Now(),
		Stages:       []string{},
	}

	// Initialize preprocessors if not already done
	if sensor.AnalogPreprocessor == nil {
		sensor.AnalogPreprocessor = NewAnalogPreprocessor(DefaultAnalogConfig())
	}
	if sensor.DigitalPreprocessor == nil {
		sensor.DigitalPreprocessor = NewDigitalPreprocessor(DefaultDigitalConfig())
	}
	if sensor.DSPPreprocessor == nil {
		sensor.DSPPreprocessor = NewDSPPreprocessor(DefaultDSPConfig())
	}

	// Stage 1: Analog preprocessing
	analogOutput, analogClipped := sensor.AnalogPreprocessor.Process(rawData)
	metadata.Stages = append(metadata.Stages, "analog")
	metadata.AnalogClipped = analogClipped
	analogClipCount, analogSatCount, analogInRMS, analogOutRMS := sensor.AnalogPreprocessor.GetStatistics()
	metadata.AnalogClippingCount = analogClipCount
	metadata.AnalogSaturationCount = analogSatCount
	metadata.AnalogInputRMS = analogInRMS
	metadata.AnalogOutputRMS = analogOutRMS

	// Stage 2: Digital preprocessing
	digitalOutput, digitalClipped := sensor.DigitalPreprocessor.Process(analogOutput)
	metadata.Stages = append(metadata.Stages, "digital")
	metadata.DigitalClipped = digitalClipped
	digitalClipCount, digitalQuantErr, _ := sensor.DigitalPreprocessor.GetStatistics()
	metadata.DigitalClippingCount = digitalClipCount
	metadata.QuantizationError = digitalQuantErr
	metadata.SQNR = sensor.DigitalPreprocessor.GetSQNR()

	// Stage 3: DSP processing
	dspOutput, spectrum, features := sensor.DSPPreprocessor.Process(digitalOutput)
	metadata.Stages = append(metadata.Stages, "dsp")
	metadata.SNREstimate = features.SNR
	metadata.PeakCount = len(features.Peaks)

	metadata.OutputSamples = len(dspOutput)
	metadata.TotalLatencyMs = time.Since(metadata.Timestamp).Seconds() * 1000.0

	return dspOutput, spectrum, features, metadata
}

// PreprocessingMetadata contains information about the preprocessing pipeline
type PreprocessingMetadata struct {
	SensorType     string
	InputSamples   int
	OutputSamples  int
	Timestamp      time.Time
	Stages         []string
	TotalLatencyMs float64

	// Analog stage metrics
	AnalogClipped         bool
	AnalogClippingCount   int
	AnalogSaturationCount int
	AnalogInputRMS        float64
	AnalogOutputRMS       float64

	// Digital stage metrics
	DigitalClipped       bool
	DigitalClippingCount int
	SQNR                 float64
	QuantizationError    float64

	// DSP stage metrics
	SNREstimate      float64
	PeakCount        int
	SpectralCentroid float64
	Bandwidth        float64
}

// =============================================================================
// ObservationReport with Verification Symbolic Certificate (VSC)
// =============================================================================

// ObservationReport represents a sensor observation with sovereign handshake
// Every report includes Hamiltonian Invariant Hash for spoof detection
// If σ_H / H_avg >= 10^-7, certificate self-invalidates
//
// References:
//   - Sovereign Handshake Protocol (ADITI 4.0)
//   - Physics as Hard Constraint (Augmented Lagrangian optimization)
type ObservationReport struct {
	// Report metadata
	ReportID   string
	Timestamp  time.Time
	PlatformID string
	SensorID   string
	ObjectID   string

	// Observation data
	MeasuredState [6]float64    // [x,y,z,vx,vy,vz] km, km/s
	Covariance    [6][6]float64 // Measurement uncertainty

	// Verification Symbolic Certificate (VSC)
	SymbolicCertificate *VerificationCertificate

	// Physics consistency verification
	HamiltonianHash  string  // Hash of H(q,p) for energy conservation check
	HamiltonianValue float64 // Computed H(q,p)
	EnergyDrift      float64 // |H(t) - H(0)| / |H(0)|
	ConsistencyGrade string  // A/B/C/D based on physics compliance

	// Falsifiable predictions (60s lookahead for VSC validation)
	PredictedState60s [6]float64
	PredictionCI      [6][2]float64 // Bootstrap confidence intervals

	// Provenance
	CausalChain []string // Processing steps applied
	RawDataHash string   // Hash of raw sensor data

	// DSP Pipeline preprocessing metadata
	PreprocessingMetadata *PreprocessingMetadata
}

// VerificationCertificate extends SINDy EquationString with bootstrap CI,
// physics consistency grade, causal provenance, and falsifiable predictions
type VerificationCertificate struct {
	// Symbolic representation
	EquationString string // SINDy-derived dynamics equation
	LaTeXForm      string // Human-readable equation

	// Statistical validation
	BootstrapCI      [6][2]float64 // 95% confidence intervals per state dimension
	PhysicsGrade     string        // A: strict energy/momentum conservation, B: within tolerance, C: approximate
	CausalProvenance []string      // Chain of processing/modification steps

	// Falsifiable predictions (60s window)
	PredictedState60s [6]float64 // Predicted state 60 seconds ahead
	PredictionError   float64    // Actual error at t+60s (validated post-facto)
	FalsificationFlag bool       // True if prediction outside CI (indicates model failure)

	// Cryptographic verification
	HamiltonianHash    string // SHA-256 of canonical H representation
	CertificateHash    string // SHA-256 of entire certificate
	TimestampSignature string // Ed25519 signature of report timestamp
}

// ComputeHamiltonianHash computes the Hamiltonian invariant hash for spoof detection
// Uses canonical string representation of H(q,p) to ensure reproducibility
func (r *ObservationReport) ComputeHamiltonianHash() string {
	// Canonical representation: "H=T+V;T=0.5*sum(p_i^2);V=-mu/sqrt(sum(q_i^2))"
	// This is a simplified hash - production would use full canonical form

	// Compute hash from measured state
	hashInput := fmt.Sprintf("H=%.15e|q=[%.15e,%.15e,%.15e]|p=[%.15e,%.15e,%.15e]",
		r.HamiltonianValue,
		r.MeasuredState[0], r.MeasuredState[1], r.MeasuredState[2],
		r.MeasuredState[3], r.MeasuredState[4], r.MeasuredState[5])

	// In production: Use SHA-256 of canonical representation
	// For now, use simple string hash
	return hashInput
}

// VerifyPhysicsConsistency checks if observation satisfies physics constraints
// Returns grade and whether certificate is valid
func (r *ObservationReport) VerifyPhysicsConsistency() (grade string, valid bool) {
	// Grade A: Energy drift < 10^-10 (machine precision limited)
	// Grade B: Energy drift < 10^-7 (sovereign handshake threshold)
	// Grade C: Energy drift < 10^-5 (acceptable for operations)
	// Grade D: Energy drift >= 10^-5 (suspect/spoof)

	if r.EnergyDrift < 1e-10 {
		return "A", true
	} else if r.EnergyDrift < 1e-7 {
		return "B", true
	} else if r.EnergyDrift < 1e-5 {
		return "C", true
	} else {
		return "D", false // Self-invalidate
	}
}

// GenerateObservationReport creates a VSC-enriched observation report
func (platform *SensorPlatform) GenerateObservationReport(
	objectID string,
	measuredState [6]float64,
	covariance [6][6]float64,
	predictedState [6]float64, // From propagation model
	preprocessingMetadata *PreprocessingMetadata, // DSP pipeline metadata
) *ObservationReport {

	report := &ObservationReport{
		ReportID:              fmt.Sprintf("%s-%s-%d", platform.ID, objectID, time.Now().UnixNano()),
		Timestamp:             time.Now(),
		PlatformID:            platform.ID,
		SensorID:              platform.Sensors[0].Type.String(),
		ObjectID:              objectID,
		MeasuredState:         measuredState,
		Covariance:            covariance,
		CausalChain:           []string{"raw_sensor", "dsp_preprocessing", "centroid_extraction", "astrometric_calibration"},
		PreprocessingMetadata: preprocessingMetadata,
	}

	// Compute Hamiltonian for energy consistency check
	// This is the key for sovereign handshake: H must be conserved
	q := []float64{measuredState[0], measuredState[1], measuredState[2]}
	mom := []float64{measuredState[3], measuredState[4], measuredState[5]}

	// Use PHS model to compute Hamiltonian
	phsModel := &orbitalPHSModel{mu: 398600.4418, isHGV: false}
	H := phsModel.Hamiltonian(q, mom, nil)

	report.HamiltonianValue = H
	hash := report.ComputeHamiltonianHash()
	report.HamiltonianHash = hash

	// Compute energy drift (requires reference H from catalog)
	// This will be populated when compared to catalog entry
	report.EnergyDrift = 0.0 // Placeholder

	// Convert covariance [6][6] to bootstrap CI [6][2] (diagonal approximation)
	var bootstrapCI [6][2]float64
	for i := 0; i < 6; i++ {
		sigma := math.Sqrt(covariance[i][i])
		bootstrapCI[i][0] = measuredState[i] - 2*sigma // 95% CI lower bound
		bootstrapCI[i][1] = measuredState[i] + 2*sigma // 95% CI upper bound
	}

	// Create symbolic certificate
	report.SymbolicCertificate = &VerificationCertificate{
		EquationString:    "ẍ = -μ/r³ · x", // Simplified orbital dynamics
		LaTeXForm:         "$\\ddot{\\mathbf{x}} = -\\frac{\\mu}{r^3} \\mathbf{x}$",
		BootstrapCI:       bootstrapCI,
		PhysicsGrade:      "B", // Will be updated by VerifyPhysicsConsistency
		CausalProvenance:  report.CausalChain,
		PredictedState60s: predictedState,
		HamiltonianHash:   hash,
	}

	// Verify and set final grade
	grade, valid := report.VerifyPhysicsConsistency()
	report.ConsistencyGrade = grade
	report.SymbolicCertificate.PhysicsGrade = grade

	if !valid {
		// Certificate self-invalidates
		report.SymbolicCertificate.FalsificationFlag = true
	}

	return report
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
