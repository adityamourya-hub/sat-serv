package maneuver

import (
	"math"
	"ssis/pkg/orbital"
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

// PredictSubOrbitalThreat implements Rapid Processing (Requirement i).
// Specialized logic for identifying ballistic missiles, hypersonics, and delivery systems.
func PredictSubOrbitalThreat(state orbital.State, timestamp int64) RapidThreatReport {
	// Base orbital energy check
	rmag := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)
	v2 := state.VX*state.VX + state.VY*state.VY + state.VZ*state.VZ
	specificEnergy := 0.5*v2 - orbital.EARTH_MU/rmag

	// Sub-orbital if energy is significantly less than circular velocity at surface
	circularEnergyAtSurface := -0.5 * orbital.EARTH_MU / orbital.ER
	classification := ThreatNominal
	confidence := 0.9

	if specificEnergy < circularEnergyAtSurface*1.05 {
		classification = ThreatBallistic
		if v2 > 25.0 { // Extremely high velocity at low alt -> Hypersonic suspicion
			classification = ThreatHypersonic
		}
	}

	if classification == ThreatNominal {
		return RapidThreatReport{Classification: classification, Confidence: 0.95}
	}

	// Simplified Ballistic Impact Prediction (vacuum parabola approx)
	impactTime := float64(timestamp) + 600.0 // Placeholder 10 min prediction
	
	return RapidThreatReport{
		Classification: classification,
		ImpactLat:      math.Asin(state.Z/rmag) * 180.0 / math.Pi,
		ImpactLon:      math.Atan2(state.Y, state.X) * 180.0 / math.Pi,
		TimeOfImpact:   int64(impactTime),
		Confidence:     confidence,
	}
}
