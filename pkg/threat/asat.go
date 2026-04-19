// Package threat provides ASAT (Anti-Satellite) detection and threat characterization
// Implements IIP algorithm, vulnerability window analysis, and sleeper proximity detection
package threat

import (
	"math"
	"time"
)

const (
	// ASAT intercept thresholds
	InterceptThresholdM   = 500.0 // 500 meters intercept threshold
	StrikeWindowPhaseDeg  = 10.0  // 10 degrees phase angle for vulnerability
	SleeperVelocityMPS    = 0.1   // 10 cm/s velocity change threshold
	ASATMachThreshold     = 7.0   // Mach 7 for DA-ASAT detection
	DAVerticalVelocityKmS = 7.0   // 7 km/s vertical velocity for DA-ASAT

	// ASAT types
	ASATDirectAscent   = "DA_ASAT"
	ASATCoorbital      = "COORBITAL_ASAT"
	ASATRFJammer       = "RF_JAMMER"
	ASATDirectedEnergy = "DEW_PLATFORM"
)

// ASATType categorizes ASAT threat types
type ASATType string

const (
	ASATTypeDirectAscent   ASATType = "DIRECT_ASCENT"
	ASATTypeCoorbital      ASATType = "COORBITAL"
	ASATTypeRFJammer       ASATType = "RF_JAMMER"
	ASATTypeDirectedEnergy ASATType = "DIRECTED_ENERGY"
	ASATTypeUnknown        ASATType = "UNKNOWN"
)

// ASATSignature represents the detection signature of an ASAT threat
type ASATSignature struct {
	Type          ASATType
	MachNumber    float64
	VerticalVel   float64 // km/s
	ThermalSig    float64 // Kelvin
	IsBacklit     bool    // Sun position relative to sensor
	PhaseAngle    float64 // degrees
	Vulnerability float64 // 0-1 score
}

// IIPResult represents Instantaneous Impact Point calculation
type IIPResult struct {
	ImpactLat    float64 // degrees
	ImpactLon    float64 // degrees
	ImpactAlt    float64 // km
	TimeToImpact float64 // seconds
	MissDistance float64 // km
	IsIntercept  bool
	Confidence   float64
}

// ASATDetector manages ASAT threat detection and characterization
type ASATDetector struct {
	// Propagator for orbit prediction
	propagator *OrbitPropagator

	// Asset catalog for vulnerability assessment
	assetCatalog map[int]*SatelliteAsset

	// Sleeper detection
	sleeperBaseline map[int]*SleeperBaseline

	// Active ASAT alerts
	activeAlerts map[int]*ASATAlert
}

// SatelliteAsset represents a protected space asset
type SatelliteAsset struct {
	NoradID     int
	Name        string
	Position    [3]float64 // ECI km
	Velocity    [3]float64 // ECI km/s
	Covariance  [6]float64 // Upper triangular 3x3
	IsProtected bool
	Value       string // "CRITICAL", "HIGH", "MEDIUM"
}

// SleeperBaseline stores baseline state for sleeper detection
type SleeperBaseline struct {
	NoradID          int
	BaselinePos      [3]float64
	BaselineVel      [3]float64
	BaselineTime     time.Time
	ObservationCount int
}

// ASATAlert represents a detected ASAT threat
type ASATAlert struct {
	AlertID             int
	AlertTime           time.Time
	ASATType            ASATType
	Signature           ASATSignature
	IIPResult           IIPResult
	TargetAsset         int
	IsVulnerable        bool
	VulnerabilityWindow time.Time
}

// NewASATDetector creates a new ASAT detection system
func NewASATDetector() *ASATDetector {
	return &ASATDetector{
		propagator:      NewOrbitPropagator(),
		assetCatalog:    make(map[int]*SatelliteAsset),
		sleeperBaseline: make(map[int]*SleeperBaseline),
		activeAlerts:    make(map[int]*ASATAlert),
	}
}

// CheckASATIntercept calculates if a DA-ASAT has an intercept solution
func (ad *ASATDetector) CheckASATIntercept(missilePos, missileVel [3]float64, asset *SatelliteAsset) IIPResult {
	// 1. Calculate relative position and velocity
	relPos := [3]float64{
		asset.Position[0] - missilePos[0],
		asset.Position[1] - missilePos[1],
		asset.Position[2] - missilePos[2],
	}

	relVel := [3]float64{
		asset.Velocity[0] - missileVel[0],
		asset.Velocity[1] - missileVel[1],
		asset.Velocity[2] - missileVel[2],
	}

	// 2. Time-to-Intercept (TTI) approximation
	dist := math.Sqrt(relPos[0]*relPos[0] + relPos[1]*relPos[1] + relPos[2]*relPos[2])
	relVelMag := math.Sqrt(relVel[0]*relVel[0] + relVel[1]*relVel[1] + relVel[2]*relVel[2])

	if relVelMag < 0.001 {
		// No closure rate
		return IIPResult{
			MissDistance: dist,
			IsIntercept:  false,
			Confidence:   0.0,
		}
	}

	tti := dist / relVelMag

	// 3. Predict Intercept Point using propagator
	predictedMissilePos := ad.propagator.Propagate(missilePos, missileVel, tti)
	predictedAssetPos := ad.propagator.Propagate(asset.Position, asset.Velocity, tti)

	// 4. Calculate miss distance at intercept
	missPos := [3]float64{
		predictedMissilePos[0] - predictedAssetPos[0],
		predictedMissilePos[1] - predictedAssetPos[1],
		predictedMissilePos[2] - predictedAssetPos[2],
	}

	missDistance := math.Sqrt(missPos[0]*missPos[0] + missPos[1]*missPos[1] + missPos[2]*missPos[2])

	// 5. Check if intercept is within 500m threshold
	isIntercept := missDistance < (InterceptThresholdM / 1000.0) // Convert to km

	// 6. Calculate confidence based on geometry
	confidence := ad.calculateInterceptConfidence(dist, relVelMag, tti)

	// 7. Calculate impact point (geographic)
	impactLat := math.Asin(predictedAssetPos[2]/6371.0) * 180.0 / math.Pi
	impactLon := math.Atan2(predictedAssetPos[1], predictedAssetPos[0]) * 180.0 / math.Pi
	altitude := math.Sqrt(predictedAssetPos[0]*predictedAssetPos[0]+predictedAssetPos[1]*predictedAssetPos[1]+predictedAssetPos[2]*predictedAssetPos[2]) - 6371.0

	return IIPResult{
		ImpactLat:    impactLat,
		ImpactLon:    impactLon,
		ImpactAlt:    altitude,
		TimeToImpact: tti,
		MissDistance: missDistance,
		IsIntercept:  isIntercept,
		Confidence:   confidence,
	}
}

// calculateInterceptConfidence computes confidence score for intercept prediction
func (ad *ASATDetector) calculateInterceptConfidence(distance, relVel, tti float64) float64 {
	// Higher confidence for shorter distances, higher closure rates, and reasonable TTI

	// Distance score (closer = higher confidence)
	distScore := 1.0 - (distance / 10000.0) // 10km max
	if distScore < 0 {
		distScore = 0
	}

	// Velocity score (faster = higher confidence)
	velScore := math.Min(relVel/10.0, 1.0) // 10 km/s max

	// TTI score (moderate TTI = higher confidence)
	ttiScore := 1.0
	if tti > 600.0 { // > 10 minutes
		ttiScore = 600.0 / tti
	}

	return 0.4*distScore + 0.3*velScore + 0.3*ttiScore
}

// DetectASATSignature identifies ASAT type from observation
func (ad *ASATDetector) DetectASATSignature(obs LaunchObservation) ASATSignature {
	// Calculate vertical velocity
	verticalVel := math.Abs(obs.Velocity[2])

	// Calculate Mach number
	velMag := math.Sqrt(obs.Velocity[0]*obs.Velocity[0] + obs.Velocity[1]*obs.Velocity[1] + obs.Velocity[2]*obs.Velocity[2])
	mach := velMag * 3.6 // Approximate conversion

	// Determine ASAT type
	asatType := ASATTypeUnknown

	if verticalVel > DAVerticalVelocityKmS && mach > ASATMachThreshold {
		asatType = ASATTypeDirectAscent
	} else if obs.Temperature > 400.0 && obs.Intensity > 0.8 {
		asatType = ASATTypeDirectedEnergy
	} else if obs.Temperature > 350.0 && obs.Intensity > 0.6 {
		asatType = ASATTypeRFJammer
	}

	// Calculate Sun position and phase angle
	sunPos := ad.getSunPosition(time.Now())
	toSun := [3]float64{
		sunPos[0] - obs.Position[0],
		sunPos[1] - obs.Position[1],
		sunPos[2] - obs.Position[2],
	}

	// Normalize vectors
	sunDist := math.Sqrt(toSun[0]*toSun[0] + toSun[1]*toSun[1] + toSun[2]*toSun[2])
	toSun = [3]float64{toSun[0] / sunDist, toSun[1] / sunDist, toSun[2] / sunDist}

	// Calculate phase angle (simplified)
	phaseAngle := math.Acos(toSun[2]) * 180.0 / math.Pi

	// Check if backlit (Sun behind sensor)
	isBacklit := phaseAngle < StrikeWindowPhaseDeg

	// Calculate vulnerability score
	vulnerability := 0.0
	if isBacklit {
		vulnerability = 1.0 - (phaseAngle / StrikeWindowPhaseDeg)
	}

	return ASATSignature{
		Type:          asatType,
		MachNumber:    mach,
		VerticalVel:   verticalVel,
		ThermalSig:    obs.Temperature,
		IsBacklit:     isBacklit,
		PhaseAngle:    phaseAngle,
		Vulnerability: vulnerability,
	}
}

// getSunPosition calculates Sun position in ECI coordinates
func (ad *ASATDetector) getSunPosition(t time.Time) [3]float64 {
	// Simplified Sun position calculation
	// In production, use precise ephemeris (e.g., JPL DE440)

	// Days since J2000 epoch
	j2000 := time.Date(2000, 1, 1, 12, 0, 0, 0, time.UTC)
	days := t.Sub(j2000).Hours() / 24.0

	// Mean anomaly (degrees)
	M := 357.529 + 0.98560028*days

	// Mean longitude (degrees)
	L := 280.459 + 0.98564736*days

	// Ecliptic longitude (degrees)
	lambda := L + 1.915*math.Sin(M*math.Pi/180.0) + 0.020*math.Sin(2*M*math.Pi/180.0)

	// Obliquity of ecliptic (degrees)
	epsilon := 23.439 - 0.00000036*days

	// Convert to equatorial coordinates
	lambdaRad := lambda * math.Pi / 180.0
	epsilonRad := epsilon * math.Pi / 180.0

	ra := math.Atan2(math.Cos(epsilonRad)*math.Sin(lambdaRad), math.Cos(lambdaRad))
	dec := math.Asin(math.Sin(epsilonRad) * math.Sin(lambdaRad))

	// Distance to Sun (AU)
	r := 1.00014 - 0.01671*math.Cos(M*math.Pi/180.0) - 0.00014*math.Cos(2*M*math.Pi/180.0)

	// Convert to ECI (km)
	sunDistKm := r * 149597870.7

	sunPos := [3]float64{
		sunDistKm * math.Cos(dec) * math.Cos(ra),
		sunDistKm * math.Cos(dec) * math.Sin(ra),
		sunDistKm * math.Sin(dec),
	}

	return sunPos
}

// CalculateVulnerabilityWindow identifies optimal time for counter-response
func (ad *ASATDetector) CalculateVulnerabilityWindow(advPos, assetPos [3]float64) (time.Time, bool) {
	sunPos := ad.getSunPosition(time.Now())

	// Vector from adversary to Sun
	toSun := [3]float64{
		sunPos[0] - advPos[0],
		sunPos[1] - advPos[1],
		sunPos[2] - advPos[2],
	}

	// Vector from adversary to asset
	toAsset := [3]float64{
		assetPos[0] - advPos[0],
		assetPos[1] - advPos[1],
		assetPos[2] - advPos[2],
	}

	// Normalize vectors
	sunDist := math.Sqrt(toSun[0]*toSun[0] + toSun[1]*toSun[1] + toSun[2]*toSun[2])
	assetDist := math.Sqrt(toAsset[0]*toAsset[0] + toAsset[1]*toAsset[1] + toAsset[2]*toAsset[2])

	toSunNorm := [3]float64{toSun[0] / sunDist, toSun[1] / sunDist, toSun[2] / sunDist}
	toAssetNorm := [3]float64{toAsset[0] / assetDist, toAsset[1] / assetDist, toAsset[2] / assetDist}

	// Calculate phase angle (dot product)
	dotProduct := toSunNorm[0]*toAssetNorm[0] + toSunNorm[1]*toAssetNorm[1] + toSunNorm[2]*toAssetNorm[2]
	phaseAngle := math.Acos(dotProduct) * 180.0 / math.Pi

	// Check if in strike window
	if phaseAngle < StrikeWindowPhaseDeg {
		return time.Now(), true
	}

	// Calculate time to next strike window (simplified - assumes circular orbit)
	// In production, use precise orbital mechanics
	orbitalPeriod := 90.0 * 60.0 // 90 minutes in seconds
	timeToWindow := (StrikeWindowPhaseDeg / 360.0) * orbitalPeriod

	return time.Now().Add(time.Duration(timeToWindow) * time.Second), false
}

// CheckSleeperAnomaly detects sleeper proximity operations using Mahalanobis gate
func (ad *ASATDetector) CheckSleeperAnomaly(noradID int, currentPos, currentVel [3]float64, currentCov [6]float64) (bool, float64) {
	baseline, exists := ad.sleeperBaseline[noradID]
	if !exists {
		// Initialize baseline
		ad.sleeperBaseline[noradID] = &SleeperBaseline{
			NoradID:          noradID,
			BaselinePos:      currentPos,
			BaselineVel:      currentVel,
			BaselineTime:     time.Now(),
			ObservationCount: 1,
		}
		return false, 0.0
	}

	// Calculate velocity change
	velChange := [3]float64{
		currentVel[0] - baseline.BaselineVel[0],
		currentVel[1] - baseline.BaselineVel[1],
		currentVel[2] - baseline.BaselineVel[2],
	}

	velChangeMag := math.Sqrt(velChange[0]*velChange[0] + velChange[1]*velChange[1] + velChange[2]*velChange[2])

	// Convert to m/s
	velChangeMPS := velChangeMag * 1000.0

	// Mahalanobis distance calculation
	// D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
	// Simplified: use velocity change magnitude relative to covariance

	// Extract velocity variance from covariance
	velVar := currentCov[3] + currentCov[4] + currentCov[5] // Diagonal elements for velocity

	mahalanobisDist := velChangeMPS / math.Sqrt(velVar)

	// Check if anomaly (threshold depends on desired false alarm rate)
	// 3-sigma threshold for 99.7% confidence
	anomalyThreshold := 3.0
	isAnomaly := mahalanobisDist > anomalyThreshold

	// Update baseline if not anomaly
	if !isAnomaly {
		baseline.ObservationCount++
		if baseline.ObservationCount > 100 {
			// Update baseline with exponential moving average
			alpha := 0.1
			for i := 0; i < 3; i++ {
				baseline.BaselineVel[i] = alpha*currentVel[i] + (1-alpha)*baseline.BaselineVel[i]
			}
		}
	}

	return isAnomaly, mahalanobisDist
}

// AddAsset adds a protected asset to the catalog
func (ad *ASATDetector) AddAsset(asset *SatelliteAsset) {
	ad.assetCatalog[asset.NoradID] = asset
}

// GenerateFireControlBrief creates high-precision targeting data for counter-response
func (ad *ASATDetector) GenerateFireControlBrief(alert *ASATAlert, asset *SatelliteAsset, advPos [3]float64) *FireControlBrief {
	// Calculate vulnerability window
	vulnWindow, isVulnerable := ad.CalculateVulnerabilityWindow(advPos, asset.Position)

	brief := &FireControlBrief{
		AlertID:             alert.AlertID,
		ASATType:            alert.ASATType,
		TargetAsset:         asset.NoradID,
		AssetPosition:       asset.Position,
		AssetVelocity:       asset.Velocity,
		AssetCovariance:     asset.Covariance,
		InterceptPoint:      [2]float64{alert.IIPResult.ImpactLat, alert.IIPResult.ImpactLon},
		TimeToIntercept:     alert.IIPResult.TimeToImpact,
		MissDistance:        alert.IIPResult.MissDistance,
		IsVulnerable:        isVulnerable,
		VulnerabilityWindow: vulnWindow,
		Confidence:          alert.IIPResult.Confidence,
		RecommendedAction:   ad.determineRecommendedAction(alert, isVulnerable),
	}

	return brief
}

// determineRecommendedAction selects appropriate counter-response
func (ad *ASATDetector) determineRecommendedAction(alert *ASATAlert, isVulnerable bool) string {
	switch alert.ASATType {
	case ASATTypeDirectAscent:
		if alert.IIPResult.IsIntercept {
			return "KINETIC_INTERCEPT"
		}
		return "EVASIVE_MANEUVER"
	case ASATTypeCoorbital:
		if isVulnerable {
			return "EW_JAMMING"
		}
		return "EMCON_INITIATE"
	case ASATTypeRFJammer:
		return "RF_COUNTERMEASURE"
	case ASATTypeDirectedEnergy:
		return "SHIELD_DEPLOY"
	default:
		return "MONITOR"
	}
}

// FireControlBrief provides targeting data for counter-space operations
type FireControlBrief struct {
	AlertID             int
	ASATType            ASATType
	TargetAsset         int
	AssetPosition       [3]float64
	AssetVelocity       [3]float64
	AssetCovariance     [6]float64
	InterceptPoint      [2]float64
	TimeToIntercept     float64
	MissDistance        float64
	IsVulnerable        bool
	VulnerabilityWindow time.Time
	Confidence          float64
	RecommendedAction   string
}

// OrbitPropagator handles orbit prediction using Yoshida-4 symplectic integration
type OrbitPropagator struct {
	// Integration parameters
	StepSize float64 // seconds
	NumSteps int
}

// NewOrbitPropagator creates a new orbit propagator
func NewOrbitPropagator() *OrbitPropagator {
	return &OrbitPropagator{
		StepSize: 1.0, // 1 second step
		NumSteps: 10,  // 10 steps for propagation
	}
}

// Propagate predicts future position using simplified propagation
func (op *OrbitPropagator) Propagate(pos, vel [3]float64, dt float64) [3]float64 {
	// Simplified propagation (linear extrapolation)
	// In production, implement Yoshida-4 symplectic integration

	// Calculate gravitational acceleration
	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	mu := 398600.4418 // Earth gravitational parameter km³/s²

	acc := [3]float64{
		-mu * pos[0] / (r * r * r),
		-mu * pos[1] / (r * r * r),
		-mu * pos[2] / (r * r * r),
	}

	// Simple Euler integration
	newPos := [3]float64{
		pos[0] + vel[0]*dt + 0.5*acc[0]*dt*dt,
		pos[1] + vel[1]*dt + 0.5*acc[1]*dt*dt,
		pos[2] + vel[2]*dt + 0.5*acc[2]*dt*dt,
	}

	return newPos
}
