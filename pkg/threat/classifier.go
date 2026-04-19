// Package threat provides intelligent threat classification for RSOs
// Detects Inspector, Jammer, and Grapple patterns from orbital behavior and material signatures
package threat

import (
	"fmt"
	"math"
	"time"
)

// ThreatType categorizes RSO threat classification
type ThreatType string

const (
	ThreatNone      ThreatType = "NONE"
	ThreatInspector ThreatType = "COORBITAL_INSPECTOR"
	ThreatJammer    ThreatType = "RF_JAMMER"
	ThreatGrapple   ThreatType = "ROBOTIC_GRAPPLE"
	ThreatASAT      ThreatType = "ASAT_WEAPON"
	ThreatUnknown   ThreatType = "UNKNOWN"
)

// ThreatLevel indicates severity
type ThreatLevel int

const (
	LevelNone     ThreatLevel = 0
	LevelLow      ThreatLevel = 1
	LevelElevated ThreatLevel = 2
	LevelHigh     ThreatLevel = 3
	LevelCritical ThreatLevel = 4
)

// RSOBehavior tracks observed behavioral patterns
type RSOBehavior struct {
	NoradID          int
	FirstSeen        time.Time
	LastSeen         time.Time
	ObservationCount int

	// Orbital patterns
	RelativeVelocityHistory []float64 // km/s
	DistanceHistory         []float64 // km
	ApproachRate            float64   // km/day

	// Physical characteristics
	MaterialID         string
	MaterialConfidence float64
	ThermalSignature   float64 // Temperature estimate
	RadarCrossSection  float64 // m²

	// Attitude/activity
	IsStabilized      bool
	RotationRateHz    float64
	OpticalGlintCount int

	// Proximity events
	ClosestApproachKm     float64
	TimeToClosestApproach time.Duration

	// EM signatures (EW awareness)
	EMSignalType   string // JAMMER, COMMS_RELAY, ELINT, UNKNOWN
	EMModulation   string // QPSK, 16-QAM, FHSS, etc.
	EMPowerDBm     float64
	EMBandwidth    float64 // Hz
	EMIsBurst      bool
	EMDopplerShift float64 // Hz

	// Optical signatures (Star Tracker SSA)
	AttitudeMode   string // NADIR, STAR_POINTING, OFFSET, UNKNOWN
	IsManeuvering  bool
	RelativeMotion bool
	StreakDetected bool

	// MGE-derived structural characterization
	ArrayToBusRatio         float64 // MGE flux ratio
	MGEComponents           int     // Number of Gaussian components
	ArchitectureType        string  // COMPACT, STANDARD, HIGH_POWER_EW_SAR, etc.
	IsStructurallyAnomalous bool
}

// ThreatClassifier analyzes RSO behavior for threat patterns
type ThreatClassifier struct {
	// Thresholds for classification
	InspectorDistanceKm    float64 // Within this distance, check for station-keeping
	GrappleClosureRateKms  float64 // Rapid approach threshold
	JammerThermalThreshold float64 // High thermal = active electronics

	// Pattern detection state
	behaviorLog map[int]*RSOBehavior
}

// NewThreatClassifier creates classifier with default thresholds
func NewThreatClassifier() *ThreatClassifier {
	return &ThreatClassifier{
		InspectorDistanceKm:    1.0,   // 1km for station-keeping
		GrappleClosureRateKms:  0.1,   // 100 m/s approach
		JammerThermalThreshold: 350.0, // Active electronics temp
		behaviorLog:            make(map[int]*RSOBehavior),
	}
}

// Analyze performs comprehensive threat assessment
func (tc *ThreatClassifier) Analyze(behavior *RSOBehavior) (ThreatType, ThreatLevel, string) {
	// Check each threat pattern in order of severity

	// 1. Robotic Grapple (most immediate physical threat)
	if tc.isGrapplePattern(behavior) {
		return ThreatGrapple, LevelCritical,
			"Rapid proximity closure with kinematic deployment detected"
	}

	// 2. ASAT Weapon (high energy, directed approach)
	if tc.isASATPattern(behavior) {
		return ThreatASAT, LevelCritical,
			"High-energy approach profile consistent with kinetic ASAT"
	}

	// 3. EW-aware threat classification (intent inference)
	if tc.hasEMData(behavior) || tc.hasOpticalData(behavior) {
		threatType, level, reason := tc.analyzeWithIntentInference(behavior)
		if threatType != ThreatUnknown {
			return threatType, level, reason
		}
	}

	// 4. MGE-based architectural fingerprinting
	if tc.hasMGEData(behavior) {
		threatType, level, reason := tc.analyzeWithMGE(behavior)
		if threatType != ThreatUnknown {
			return threatType, level, reason
		}
	}

	// 5. RF Jammer (electronics-based threat)
	if tc.isJammerPattern(behavior) {
		return ThreatJammer, LevelHigh,
			"Active electronic payload detected via thermal signature"
	}

	// 5. Co-orbital Inspector (intelligence gathering)
	if tc.isInspectorPattern(behavior) {
		return ThreatInspector, LevelHigh,
			"Station-keeping behavior suggests intelligence collection"
	}

	// 6. Standard satellite
	if behavior.MaterialConfidence > 0.7 {
		return ThreatNone, LevelNone,
			"Standard satellite materials and behavior"
	}

	return ThreatUnknown, LevelLow,
		"Insufficient data for classification"
}

// hasEMData checks if EM signature data is available
func (tc *ThreatClassifier) hasEMData(behavior *RSOBehavior) bool {
	return behavior.EMSignalType != "" || behavior.EMPowerDBm != 0
}

// hasOpticalData checks if optical signature data is available
func (tc *ThreatClassifier) hasOpticalData(behavior *RSOBehavior) bool {
	return behavior.AttitudeMode != "" || behavior.StreakDetected
}

// analyzeWithIntentInference uses EM and optical signatures for threat classification
func (tc *ThreatClassifier) analyzeWithIntentInference(behavior *RSOBehavior) (ThreatType, ThreatLevel, string) {
	// Apply intent inference logic (simplified integration of sigint package)

	// Jamming: Wide-band noise + Nadir-pointing
	if behavior.EMSignalType == "JAMMER" && behavior.AttitudeMode == "NADIR" {
		return ThreatJammer, LevelCritical,
			"EW-aware: Jamming detected via EM and optical signature fusion"
	}

	// Comms Relay: Periodic pulses + Star-pointing
	if behavior.EMSignalType == "COMMS_RELAY" && behavior.AttitudeMode == "STAR_POINTING" {
		return ThreatNone, LevelLow,
			"EW-aware: Comms relay activity detected"
	}

	// ELINT: High power + Offset-pointing
	if behavior.EMSignalType == "ELINT" && behavior.AttitudeMode == "OFFSET" {
		return ThreatInspector, LevelHigh,
			"EW-aware: ELINT activity detected via attitude offset"
	}

	// Kinetic/Robotic Threat: No EM signature + Rapid proximity closure
	if behavior.EMSignalType == "UNKNOWN" && behavior.StreakDetected {
		return ThreatGrapple, LevelCritical,
			"EW-aware: Kinetic threat detected via streak without EM signature"
	}

	// Co-orbital Inspector: Low power + Station-keeping
	if behavior.EMSignalType == "COMMS_RELAY" && behavior.EMPowerDBm < -50.0 && behavior.IsStabilized {
		return ThreatInspector, LevelHigh,
			"EW-aware: Co-orbital inspector detected via low-power comms"
	}

	return ThreatUnknown, LevelNone, ""
}

// hasMGEData checks if MGE structural data is available
func (tc *ThreatClassifier) hasMGEData(behavior *RSOBehavior) bool {
	return behavior.ArrayToBusRatio > 0 || behavior.MGEComponents > 0
}

// analyzeWithMGE uses MGE structural data for architectural fingerprinting
func (tc *ThreatClassifier) analyzeWithMGE(behavior *RSOBehavior) (ThreatType, ThreatLevel, string) {
	if !tc.hasMGEData(behavior) {
		return ThreatUnknown, LevelNone, ""
	}

	// Architectural fingerprinting based on array-to-bus ratio
	ratio := behavior.ArrayToBusRatio
	architecture := behavior.ArchitectureType

	// High-power EW/SAR or Jammer detection
	if ratio >= 8.0 {
		// Check if nadir-pointing (ISR pass)
		if behavior.AttitudeMode == "NADIR" && behavior.IsStabilized {
			return ThreatJammer, LevelCritical,
				fmt.Sprintf("MGE: CRITICAL_JAMMER_DETECTED - Ratio: %.2f, Architecture: %s, Nadir ISR pass", ratio, architecture)
		}
		return ThreatJammer, LevelHigh,
			fmt.Sprintf("MGE: High-power EW platform - Ratio: %.2f, Architecture: %s", ratio, architecture)
	}

	// Standard imaging/comms satellite
	if ratio >= 2.0 && ratio < 8.0 {
		if behavior.AttitudeMode == "NADIR" {
			return ThreatInspector, LevelHigh,
				fmt.Sprintf("MGE: ISR imaging satellite - Ratio: %.2f, Architecture: %s", ratio, architecture)
		}
		return ThreatNone, LevelLow,
			fmt.Sprintf("MGE: Standard comms satellite - Ratio: %.2f, Architecture: %s", ratio, architecture)
	}

	// Compact/debris-clearing drone
	if ratio < 1.0 {
		if behavior.IsManeuvering || behavior.RelativeMotion {
			return ThreatGrapple, LevelHigh,
				fmt.Sprintf("MGE: Compact maneuvering object - Possible debris-clearing drone, Ratio: %.2f", ratio)
		}
		return ThreatNone, LevelLow,
			fmt.Sprintf("MGE: Compact satellite/debris - Ratio: %.2f", ratio)
	}

	// Structural anomaly detection
	if behavior.IsStructurallyAnomalous {
		return ThreatInspector, LevelHigh,
			fmt.Sprintf("MGE: Structural anomaly detected - Ratio: %.2f, Components: %d", ratio, behavior.MGEComponents)
	}

	return ThreatUnknown, LevelNone, ""
}

// isInspectorPattern detects station-keeping behavior
func (tc *ThreatClassifier) isInspectorPattern(b *RSOBehavior) bool {
	// Must be within inspection range
	if b.ClosestApproachKm > tc.InspectorDistanceKm {
		return false
	}

	// Must be stabilized (pointing at target)
	if !b.IsStabilized {
		return false
	}

	// Check for station-keeping (near-zero relative velocity)
	if len(b.RelativeVelocityHistory) >= 3 {
		avgVel := 0.0
		for _, v := range b.RelativeVelocityHistory {
			avgVel += math.Abs(v)
		}
		avgVel /= float64(len(b.RelativeVelocityHistory))

		// Station-keeping: < 10 m/s relative velocity
		if avgVel < 0.01 { // km/s
			return true
		}
	}

	// Check for optical payload (high-res imaging capability)
	if b.MaterialID == "OPTICAL_GLASS" && b.IsStabilized {
		return true
	}

	return false
}

// isJammerPattern detects active electronic payloads
func (tc *ThreatClassifier) isJammerPattern(b *RSOBehavior) bool {
	// High thermal signature indicates active electronics
	if b.ThermalSignature < tc.JammerThermalThreshold {
		return false
	}

	// Must have stabilized attitude (pointing antenna)
	if !b.IsStabilized {
		return false
	}

	// Check for RF-related materials
	highPowerMaterials := map[string]bool{
		"GaAs_SOLAR_PANEL": true, // High-efficiency = high power
		"KEVLAR_COMPOSITE": true, // Shielding
	}

	if highPowerMaterials[b.MaterialID] && b.ThermalSignature > 400.0 {
		return true
	}

	// Unusual RCS for power/mass ratio
	if b.RadarCrossSection > 10.0 && b.ThermalSignature > 380.0 {
		return true
	}

	return false
}

// isGrapplePattern detects rapid approach with deployment
func (tc *ThreatClassifier) isGrapplePattern(b *RSOBehavior) bool {
	// Rapid closure rate
	if b.ApproachRate < tc.GrappleClosureRateKms*86400 { // Convert to km/day
		return false
	}

	// Very close approach
	if b.ClosestApproachKm > 0.5 { // Within 500m
		return false
	}

	// Robotic materials
	if b.MaterialID == "KEVLAR_COMPOSITE" || b.MaterialID == "STEALTH_COATING" {
		// Rapid approach + robotic materials = grapple
		return true
	}

	// Check for attitude instability during approach (deployment)
	if b.RotationRateHz > 0.1 && !b.IsStabilized {
		return true
	}

	return false
}

// isASATPattern detects kinetic ASAT approach
func (tc *ThreatClassifier) isASATPattern(b *RSOBehavior) bool {
	// Very high closure rate
	if b.ApproachRate < tc.GrappleClosureRateKms*2*86400 {
		return false
	}

	// Stealth coating (hard to track)
	if b.MaterialID != "STEALTH_COATING" {
		return false
	}

	// Direct approach trajectory (minimal station-keeping)
	if b.IsStabilized {
		return false // ASATs don't stabilize, they impact
	}

	// Check for terminal guidance behavior
	if len(b.DistanceHistory) >= 5 {
		// Calculate acceleration (should increase near target)
		recent := b.DistanceHistory[len(b.DistanceHistory)-5:]
		if len(recent) >= 3 {
			// Distance should be decreasing non-linearly
			d1 := recent[0] - recent[1]
			d2 := recent[1] - recent[2]
			if d2 > d1*1.5 { // Accelerating
				return true
			}
		}
	}

	return false
}

// UpdateBehavior updates tracked behavior for an RSO
func (tc *ThreatClassifier) UpdateBehavior(
	noradID int,
	relativeVel float64,
	distanceKm float64,
	materialID string,
	materialConf float64,
	isStabilized bool,
) *RSOBehavior {
	behavior, exists := tc.behaviorLog[noradID]
	if !exists {
		behavior = &RSOBehavior{
			NoradID:   noradID,
			FirstSeen: time.Now(),
		}
		tc.behaviorLog[noradID] = behavior
	}

	// Update state
	behavior.LastSeen = time.Now()
	behavior.ObservationCount++
	behavior.RelativeVelocityHistory = append(
		behavior.RelativeVelocityHistory,
		relativeVel,
	)
	behavior.DistanceHistory = append(behavior.DistanceHistory, distanceKm)

	// Keep only recent history
	if len(behavior.RelativeVelocityHistory) > 20 {
		behavior.RelativeVelocityHistory = behavior.RelativeVelocityHistory[1:]
	}
	if len(behavior.DistanceHistory) > 20 {
		behavior.DistanceHistory = behavior.DistanceHistory[1:]
	}

	// Update approach rate
	if len(behavior.DistanceHistory) >= 2 {
		behavior.ApproachRate =
			behavior.DistanceHistory[0] - behavior.DistanceHistory[len(behavior.DistanceHistory)-1]
	}

	// Update closest approach
	if distanceKm < behavior.ClosestApproachKm || behavior.ClosestApproachKm == 0 {
		behavior.ClosestApproachKm = distanceKm
	}

	// Update material info
	behavior.MaterialID = materialID
	behavior.MaterialConfidence = materialConf
	behavior.IsStabilized = isStabilized

	return behavior
}

// GenerateBriefRecommendation returns action code based on threat
func (tc *ThreatClassifier) GenerateBriefRecommendation(
	threatType ThreatType,
	level ThreatLevel,
	isIndianAsset bool,
) string {
	// Critical threats to Indian assets
	if isIndianAsset && level >= LevelHigh {
		return "EMCON_B_INITIATE" // Emission control
	}

	// Non-critical but suspicious
	if level >= LevelElevated {
		return "TRACK_MAINTAIN" // Continue tracking
	}

	// Specific threat types
	switch threatType {
	case ThreatInspector:
		return "INTEL_COLLECTION" // Gather more data
	case ThreatJammer:
		return "FREQ_HOP_PREPARE" // Prepare countermeasures
	case ThreatGrapple:
		return "EVASIVE_MANEUVER" // Physical avoidance
	case ThreatASAT:
		return "IMMEDIATE_DEPLOY" // Emergency action
	default:
		return "TRACK_MAINTAIN"
	}
}

// GetStats returns classifier statistics
func (tc *ThreatClassifier) GetStats() map[string]interface{} {
	stats := map[string]interface{}{
		"tracked_objects": len(tc.behaviorLog),
	}

	threatCounts := make(map[ThreatType]int)
	for _, behavior := range tc.behaviorLog {
		threat, _, _ := tc.Analyze(behavior)
		threatCounts[threat]++
	}

	stats["threat_breakdown"] = threatCounts
	return stats
}
