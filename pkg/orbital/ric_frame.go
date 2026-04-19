package orbital

import (
	"math"
)

/**
 * RIC Frame: Radial-In-track-Cross-track Coordinate System
 *
 * Essential for residual analysis and calibration against Fortran truth data.
 * Transforms ECI errors into physically meaningful components:
 *
 * - Radial (R): Pointing from Earth's center to satellite
 * - In-track (I): Along velocity direction (circular orbits)
 * - Cross-track (C): Normal to orbital plane (angular momentum)
 *
 * Why RIC matters:
 * - Radial errors → Eccentricity/GM mismatch
 * - In-track errors → Drag/density model issues
 * - Cross-track errors → Coordinate frame rotation errors
 *
 * Reference: Hill Frame / RTN (Radial-Transverse-Normal)
 */

// RICFrame represents the orthonormal basis vectors
type RICFrame struct {
	R [3]float64 // Radial unit vector (points outward)
	I [3]float64 // In-track unit vector (along velocity)
	C [3]float64 // Cross-track unit vector (normal to plane)
}

// StateVector represents position and velocity in ECI
type StateVector struct {
	Position [3]float64 // km
	Velocity [3]float64 // km/s
}

// ComputeRICFrame builds the RIC basis from a reference state
// Reference is typically the "truth" trajectory from JPL ephemeris
func ComputeRICFrame(ref StateVector) RICFrame {
	// Radial: r̂ = r / |r|
	rMag := norm(ref.Position)
	R := divide(ref.Position, rMag)

	// Cross-track: Ĉ = (r × v) / |r × v|
	// This is the angular momentum direction
	h := cross(ref.Position, ref.Velocity)
	hMag := norm(h)
	C := divide(h, hMag)

	// In-track: Î = Ĉ × r̂ (completes right-handed set)
	I := cross(C, R)

	return RICFrame{R: R, I: I, C: C}
}

// TransformToRIC transforms an ECI delta vector to RIC coordinates
// Input: ECI error vector (predicted - truth)
// Output: (radial, in-track, cross-track) components in km
func (ric *RICFrame) TransformToRIC(deltaECI [3]float64) (radial, inTrack, crossTrack float64) {
	radial = dot(deltaECI, ric.R)
	inTrack = dot(deltaECI, ric.I)
	crossTrack = dot(deltaECI, ric.C)
	return
}

// TransformFromRIC converts RIC components back to ECI
// Useful for applying RIC-based corrections
func (ric *RICFrame) TransformFromRIC(radial, inTrack, crossTrack float64) [3]float64 {
	return add(
		scale(ric.R, radial),
		add(
			scale(ric.I, inTrack),
			scale(ric.C, crossTrack),
		),
	)
}

// ResidualAnalysis performs RIC decomposition of ECI errors
// Returns physically interpretable error components
type ResidualAnalysis struct {
	// Position errors (meters)
	RadialErrorM     float64 // Along radial direction
	InTrackErrorM    float64 // Along velocity direction
	CrossTrackErrorM float64 // Normal to orbital plane

	// Magnitude
	TotalErrorM float64 // |error| in meters

	// Diagnostics
	ErrorSignature string // Classification: "DRAG", "J2", "FRAME"
}

// AnalyzeResiduals computes RIC decomposition and diagnoses root cause
func AnalyzeResiduals(predicted, truth StateVector) ResidualAnalysis {
	// Compute ECI error vector
	delta := subtract(predicted.Position, truth.Position)

	// Convert to meters
	deltaM := scale(delta, 1000.0)

	// Build RIC frame from truth (reference)
	ric := ComputeRICFrame(truth)

	// Transform to RIC
	r, i, c := ric.TransformToRIC(deltaM)

	// Classify error signature
	signature := classifyErrorSignature(r, i, c)

	return ResidualAnalysis{
		RadialErrorM:     r,
		InTrackErrorM:    i,
		CrossTrackErrorM: c,
		TotalErrorM:      norm(deltaM),
		ErrorSignature:   signature,
	}
}

// classifyErrorSignature diagnoses the root cause from RIC pattern
func classifyErrorSignature(r, i, c float64) string {
	absR := math.Abs(r)
	absI := math.Abs(i)
	absC := math.Abs(c)

	// Dominant component determines signature
	if absI > absR && absI > absC {
		// Growing in-track → Drag model
		if absI > 50.0 {
			return "DRAG_MODEL"
		}
		return "DRAG_MINOR"
	}

	if absR > absI && absR > absC {
		// Oscillating radial → J2/eccentricity
		if absR > 20.0 {
			return "J2_ECCENTRICITY"
		}
		return "ORBITAL_ELEMENT"
	}

	if absC > absR && absC > absI {
		// Cross-track → Coordinate frame
		if absC > 10.0 {
			return "COORDINATE_FRAME"
		}
		return "CROSS_TRACK_DRIFT"
	}

	return "BALANCED"
}

// ResidualStats aggregates statistics over a trajectory
type ResidualStats struct {
	// Max errors (meters)
	MaxRadialM     float64
	MaxInTrackM    float64
	MaxCrossTrackM float64
	MaxTotalM      float64

	// RMS errors
	RMSRadialM     float64
	RMSInTrackM    float64
	RMSCrossTrackM float64
	RMSTotalM      float64

	// Mean errors
	MeanRadialM     float64
	MeanInTrackM    float64
	MeanCrossTrackM float64

	// Error counts by signature
	SignatureCounts map[string]int

	// Pass/fail
	Passed bool // All errors < 10m requirement
}

// ComputeResidualStats aggregates residual analysis over time series
func ComputeResidualStats(analyses []ResidualAnalysis) ResidualStats {
	stats := ResidualStats{
		SignatureCounts: make(map[string]int),
		Passed:          true,
	}

	if len(analyses) == 0 {
		return stats
	}

	// Accumulate sums for mean/RMS
	sumR, sumI, sumC := 0.0, 0.0, 0.0
	sumR2, sumI2, sumC2 := 0.0, 0.0, 0.0

	for _, a := range analyses {
		// Max tracking
		if math.Abs(a.RadialErrorM) > stats.MaxRadialM {
			stats.MaxRadialM = math.Abs(a.RadialErrorM)
		}
		if math.Abs(a.InTrackErrorM) > stats.MaxInTrackM {
			stats.MaxInTrackM = math.Abs(a.InTrackErrorM)
		}
		if math.Abs(a.CrossTrackErrorM) > stats.MaxCrossTrackM {
			stats.MaxCrossTrackM = math.Abs(a.CrossTrackErrorM)
		}
		if a.TotalErrorM > stats.MaxTotalM {
			stats.MaxTotalM = a.TotalErrorM
		}

		// Sums
		sumR += a.RadialErrorM
		sumI += a.InTrackErrorM
		sumC += a.CrossTrackErrorM

		sumR2 += a.RadialErrorM * a.RadialErrorM
		sumI2 += a.InTrackErrorM * a.InTrackErrorM
		sumC2 += a.CrossTrackErrorM * a.CrossTrackErrorM

		// Signature counting
		stats.SignatureCounts[a.ErrorSignature]++

		// ADITI requirement: < 10m over 24h
		if a.TotalErrorM > 10.0 {
			stats.Passed = false
		}
	}

	n := float64(len(analyses))

	// Means
	stats.MeanRadialM = sumR / n
	stats.MeanInTrackM = sumI / n
	stats.MeanCrossTrackM = sumC / n

	// RMS
	stats.RMSRadialM = math.Sqrt(sumR2 / n)
	stats.RMSInTrackM = math.Sqrt(sumI2 / n)
	stats.RMSCrossTrackM = math.Sqrt(sumC2 / n)
	stats.RMSTotalM = math.Sqrt((sumR2 + sumI2 + sumC2) / n)

	return stats
}

// Note: norm, dot, cross are defined in srp_advanced.go
// These additional helpers are specific to RIC frame operations:

func divide(v [3]float64, scalar float64) [3]float64 {
	return [3]float64{
		v[0] / scalar,
		v[1] / scalar,
		v[2] / scalar,
	}
}

func add(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
	}
}

func subtract(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
	}
}

func scale(v [3]float64, scalar float64) [3]float64 {
	return [3]float64{
		v[0] * scalar,
		v[1] * scalar,
		v[2] * scalar,
	}
}
