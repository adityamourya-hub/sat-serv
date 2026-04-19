// Package cola provides Fuel Trap logic for non-kinetic orbital denial
// Uses probability of collision manipulation to force adversary maneuvers
package cola

import (
	"math"
)

const (
	ManeuverThreshold = 1e-4 // 1 in 10,000 (Standard industry trigger)
	WarningThreshold  = 1e-5 // Zone where we begin "Information Injection"
	ScareThreshold    = 5e-5 // Zone where we can push them over the edge with uncertainty inflation
)

// FuelTrapMonitor checks if an adversary's P_c is approaching the maneuver trigger
// Returns: (isTrapPossible, probabilityOfCollision, recommendation)
func FuelTrapMonitor(
	advPos, debPos [3]float64,
	advCov, debCov [6]float64, // Upper triangular of 3x3 covariance matrices
	hardBodyRadius float64,
) (bool, float64, string) {
	// 1. Calculate relative position (Miss Distance)
	relPos := [3]float64{
		advPos[0] - debPos[0],
		advPos[1] - debPos[1],
		advPos[2] - debPos[2],
	}
	missDist := math.Sqrt(relPos[0]*relPos[0] + relPos[1]*relPos[1] + relPos[2]*relPos[2])

	// 2. Combine Covariance matrices (S = Cov_adv + Cov_deb)
	// Using upper triangular representation
	S := [6]float64{
		advCov[0] + debCov[0], // xx
		advCov[1] + debCov[1], // xy
		advCov[2] + debCov[2], // xz
		advCov[3] + debCov[3], // yy
		advCov[4] + debCov[4], // yz
		advCov[5] + debCov[5], // zz
	}

	// 3. Calculate combined sigma (simplified 2D approximation)
	// Focus on primary variance for encounter plane
	sigmaPrimary := math.Sqrt(S[0] + S[3]) // xx + yy (radial/in-track)
	sigmaCross := math.Sqrt(S[5])          // zz (cross-track)

	// 4. Calculate Pc using 2D Gaussian PDF (Foster Method approximation)
	// Pc = (HBR^2 / (2 * sigma_x * sigma_y)) * exp(-r^2 / (2 * sigma^2))
	sigmaProduct := sigmaPrimary * sigmaCross
	exponent := -math.Pow(missDist, 2) / (2 * math.Pow(sigmaPrimary, 2))
	pc := (math.Pow(hardBodyRadius, 2) / (2 * sigmaProduct)) * math.Exp(exponent)

	// 5. Tactical Evaluation
	recommendation := "NO_ACTION"

	if pc >= ManeuverThreshold {
		recommendation = "ADVERSARY_LIKELY_MANEUVERING"
		return true, pc, recommendation
	} else if pc >= ScareThreshold {
		// This is the "Fuel Trap" zone.
		// By injecting slightly noisier data into the public mesh,
		// we can push their calculated 'pc' above 1e-4.
		recommendation = "FUEL_TRAP_ACTIVE - UNCERTAINTY_INFLATION_RECOMMENDED"
		return true, pc, recommendation
	} else if pc >= WarningThreshold {
		recommendation = "MONITOR_FOR_TRAP_OPPORTUNITY"
		return false, pc, recommendation
	}

	return false, pc, recommendation
}

// UncertaintyInflation calculates the required covariance inflation
// to push Pc above the maneuver threshold
func UncertaintyInflation(
	currentPc float64,
	missDist float64,
	hardBodyRadius float64,
) float64 {
	// Solve for required sigma to achieve Pc = ManeuverThreshold
	// Pc = (HBR^2 / (2 * sigma^2)) * exp(-r^2 / (2 * sigma^2))
	// This is a transcendental equation, solved iteratively

	targetPc := ManeuverThreshold

	// Binary search for required sigma factor
	low := 1.0
	high := 10.0

	for i := 0; i < 20; i++ {
		mid := (low + high) / 2.0
		sigma := mid

		exponent := -math.Pow(missDist, 2) / (2 * math.Pow(sigma, 2))
		testPc := (math.Pow(hardBodyRadius, 2) / (2 * math.Pow(sigma, 2))) * math.Exp(exponent)

		if testPc > targetPc {
			high = mid
		} else {
			low = mid
		}
	}

	return (low + high) / 2.0
}

// SpoofCovariance generates inflated covariance for strategic data sharing
// Returns modified covariance matrix that will push adversary's Pc over threshold
func SpoofCovariance(
	originalCov [6]float64,
	inflationFactor float64,
) [6]float64 {
	spoofed := [6]float64{}
	for i := range originalCov {
		spoofed[i] = originalCov[i] * inflationFactor
	}
	return spoofed
}

// FuelTrapOpportunity represents a potential fuel trap scenario
type FuelTrapOpportunity struct {
	AdversaryID     int
	DebrisID        int
	MissDistance    float64 // km
	CurrentPc       float64
	RequiredPc      float64
	InflationFactor float64
	TimeToTCA       float64 // seconds
	Recommendation  string
}

// IdentifyFuelTrapOpportunities scans debris catalog for trap opportunities
func IdentifyFuelTrapOpportunities(
	adversary *Satellite,
	debrisCatalog []*Satellite,
	timeWindow float64, // seconds
) []FuelTrapOpportunity {
	opportunities := make([]FuelTrapOpportunity, 0)

	for _, debris := range debrisCatalog {
		// Skip if too far (use S2 filter in production)
		// Calculate miss distance and Pc
		relPos := [3]float64{
			adversary.Position[0] - debris.Position[0],
			adversary.Position[1] - debris.Position[1],
			adversary.Position[2] - debris.Position[2],
		}
		missDist := math.Sqrt(relPos[0]*relPos[0] + relPos[1]*relPos[1] + relPos[2]*relPos[2])

		if missDist > 5.0 { // 5km threshold for efficiency
			continue
		}

		// Check fuel trap condition
		isTrap, pc, recommendation := FuelTrapMonitor(
			adversary.Position,
			debris.Position,
			adversary.Covariance,
			debris.Covariance,
			adversary.HardBody+debris.HardBody,
		)

		if isTrap && pc < ManeuverThreshold {
			// Calculate required inflation
			inflation := UncertaintyInflation(pc, missDist, adversary.HardBody+debris.HardBody)

			opportunity := FuelTrapOpportunity{
				AdversaryID:     adversary.ID,
				DebrisID:        debris.ID,
				MissDistance:    missDist,
				CurrentPc:       pc,
				RequiredPc:      ManeuverThreshold,
				InflationFactor: inflation,
				TimeToTCA:       0, // Calculate from propagation
				Recommendation:  recommendation,
			}

			opportunities = append(opportunities, opportunity)
		}
	}

	return opportunities
}

// GenerateSpoofedTLE generates a TLE with inflated covariance for fuel trap
// This is the "Information Injection" that triggers adversary maneuvers
func GenerateSpoofedTLE(
	original *Satellite,
	inflationFactor float64,
) Satellite {
	spoofed := *original

	// Inflate covariance
	spoofed.Covariance = SpoofCovariance(original.Covariance, inflationFactor)

	// Add slight position noise to simulate "unmodeled perturbation"
	// This makes the inflation appear legitimate
	spoofed.Position[0] += (2.0*0.5 - 1.0) * 0.01 // 10m noise
	spoofed.Position[1] += (2.0*0.5 - 1.0) * 0.01
	spoofed.Position[2] += (2.0*0.5 - 1.0) * 0.01

	return spoofed
}

// EvaluateFuelTrapEffectiveness estimates the fuel cost to adversary
// Returns estimated delta-v (m/s) required for avoidance maneuver
func EvaluateFuelTrapEffectiveness(
	missDistance float64,
	relativeVelocity float64,
	missTime float64,
) float64 {
	// Simple estimation: delta-v = relative_velocity * (required_separation / time)
	// More accurate: use Lambert's problem or HCW equations

	requiredSeparation := 1.0 // km (1000m safety margin)
	currentSeparation := missDistance

	if currentSeparation < requiredSeparation {
		return 0 // Already too close, maneuver not effective
	}

	// Delta-v required to increase miss distance to safety margin
	deltaV := relativeVelocity * ((requiredSeparation - currentSeparation) / missTime)

	return deltaV * 1000.0 // Convert km/s to m/s
}

// FuelTrapReport provides comprehensive analysis for command decisions
type FuelTrapReport struct {
	TotalOpportunities int
	HighValueTargets   []int   // Adversary IDs with best opportunities
	EstimatedFuelCost  float64 // Total delta-v (m/s) to adversary
	RecommendedActions []string
	RiskAssessment     string
}

// GenerateFuelTrapReport creates command-level fuel trap analysis
func GenerateFuelTrapReport(opportunities []FuelTrapOpportunity) FuelTrapReport {
	report := FuelTrapReport{
		TotalOpportunities: len(opportunities),
		HighValueTargets:   make([]int, 0),
		EstimatedFuelCost:  0.0,
		RecommendedActions: make([]string, 0),
		RiskAssessment:     "LOW",
	}

	// Aggregate fuel cost
	for _, opp := range opportunities {
		deltaV := EvaluateFuelTrapEffectiveness(
			opp.MissDistance,
			0.5,   // Assumed relative velocity (km/s)
			600.0, // 10 minutes
		)
		report.EstimatedFuelCost += deltaV

		// Identify high-value targets (multiple opportunities)
		targetCount := 0
		for _, opp2 := range opportunities {
			if opp2.AdversaryID == opp.AdversaryID {
				targetCount++
			}
		}

		if targetCount > 2 {
			report.HighValueTargets = append(report.HighValueTargets, opp.AdversaryID)
		}
	}

	// Risk assessment
	if report.EstimatedFuelCost > 1000.0 { // 1 km/s total
		report.RiskAssessment = "HIGH - Significant fuel exhaustion"
		report.RecommendedActions = append(report.RecommendedActions, "Execute fuel trap sequence")
	} else if report.EstimatedFuelCost > 100.0 {
		report.RiskAssessment = "MEDIUM - Moderate fuel impact"
		report.RecommendedActions = append(report.RecommendedActions, "Consider selective fuel trap")
	} else {
		report.RiskAssessment = "LOW - Minimal impact"
		report.RecommendedActions = append(report.RecommendedActions, "Monitor for future opportunities")
	}

	return report
}
