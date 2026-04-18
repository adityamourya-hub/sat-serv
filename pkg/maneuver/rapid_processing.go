package maneuver

import (
	"fmt"
	"math"
	"ssis/pkg/physics"
)

type ThreatType string

const (
	ThreatBallistic  ThreatType = "ballistic_missile"
	ThreatHypersonic ThreatType = "hypersonic_glide"
	ThreatSubOrbital ThreatType = "sub_orbital_debris"
	ThreatNominal    ThreatType = "nominal_satellite"
)

type RapidThreatReport struct {
	ObjectID       int
	Classification ThreatType
	ImpactLat      float64
	ImpactLon      float64
	TimeOfImpact   int64
	Confidence     float64
}

/**
 * PredictSubOrbitalThreat implements Rapid Processing (Requirement i).
 * Specialized logic for identifying ballistic missiles, hypersonics, and delivery systems.
 * Uses specific energy gating vs circular energy at surface to detect sub-orbital arcs.
 */
func PredictSubOrbitalThreat(state physics.State, timestamp int64) RapidThreatReport {
	fmt.Printf("[TRC] Entering PredictSubOrbitalThreat at t=%d\n", timestamp)
	defer fmt.Printf("[TRC] Exiting PredictSubOrbitalThreat\n")

	// 1. Base orbital energy check
	rmag := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	v2 := state.Vel[0]*state.Vel[0] + state.Vel[1]*state.Vel[1] + state.Vel[2]*state.Vel[2]
	specificEnergy := 0.5*v2 - physics.EARTH_MU_IAU/rmag

	// 2. Sub-orbital gating: Energy significantly less than circular velocity at surface
	circularEnergyAtSurface := -0.5 * physics.EARTH_MU_IAU / physics.EARTH_R_EQUATORIAL
	classification := ThreatNominal
	confidence := 0.9

	if specificEnergy < circularEnergyAtSurface*1.05 {
		classification = ThreatBallistic
		if v2 > 25.0 { // 5km/s^2 proxy for hypersonic regime at low alt
			classification = ThreatHypersonic
		}
	}

	if classification == ThreatNominal {
		return RapidThreatReport{Classification: classification, Confidence: 0.95}
	}

	// 3. Simplified Ballistic Impact Prediction (WGS84 geodetic proxy)
	impactTime := float64(timestamp) + 600.0 // Placeholder 10 min prediction
	
	return RapidThreatReport{
		Classification: classification,
		ImpactLat:      math.Asin(state.Pos[2]/rmag) * (180.0 / math.Pi),
		ImpactLon:      math.Atan2(state.Pos[1], state.Pos[0]) * (180.0 / math.Pi),
		TimeOfImpact:   int64(impactTime),
		Confidence:     confidence,
	}
}
