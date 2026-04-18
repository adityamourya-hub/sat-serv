package hypersonic

import (
	"math"
	"ssis/pkg/orbital"
)

// BallisticThreat estimates the instantaneous impact point of a sub-orbital track.
type BallisticThreat struct {
	ImpactLat    float64
	ImpactLon    float64
	ApogeeKm     float64
	TimeOfFlight float64
}

// ComputeBallisticImpact point maps early radar tracks (burnout velocity, flight path angle)
// to a surface impact coordinate for rapid processing.
func ComputeBallisticImpact(launchLat, launchLon, vBurnoutKmS, gammaRad float64) BallisticThreat {
	// Re-use exported constants from orbital
	const RE = orbital.ER
	const MU = orbital.EARTH_MU

	// V^2 / 2 - MU / r = -MU / 2a
	v2 := vBurnoutKmS * vBurnoutKmS
	h := v2/2.0 - (MU / RE)
	if math.Abs(h) < 1e-9 {
		h = -1e-9 // Avoid singularity for parabolic
	}
	a := -MU / (2.0 * h)

	// Specific angular momentum h_vec
	h_ang := RE * vBurnoutKmS * math.Cos(gammaRad)

	// Eccentricity
	e2 := 1.0 - (h_ang*h_ang)/(MU*a)
	if e2 < 0 { e2 = 0 }
	e := math.Sqrt(e2)

	// Apogee
	apogee := a*(1.0+e) - RE

	// Simplified Range Angle (assuming non-rotating Earth for instantaneous estimate)
	divisor := 1.0 + math.Pow(v2/(MU/RE), 2)*math.Pow(math.Sin(gammaRad)*math.Cos(gammaRad), 2)
	rangeAngle := math.Pi - 2.0*math.Asin(1.0/math.Sqrt(divisor))

	// Time of flight approximation
	tof := 2.0 * (math.Pi * math.Sqrt(math.Abs(math.Pow(a, 3)/MU))) * (0.5) // Simplified half-ellipse

	// Impact coordinates (Simplified purely equatorial eastward launch for demo)
	impactLat := launchLat
	impactLon := launchLon + rangeAngle*(180.0/math.Pi)

	return BallisticThreat{
		ImpactLat:    impactLat,
		ImpactLon:    impactLon,
		ApogeeKm:     apogee,
		TimeOfFlight: tof,
	}
}
