package physics

import (
	"math"
	"time"
)

/**
 * Vector3 represents a 3D vector [x, y, z].
 */
type Vector3 [3]float64

/**
 * State represents the orbital state (position and velocity) in ECI coordinates.
 * Hardened for Clean Architecture: Separates data from behavior.
 */
type State struct {
	Pos        Vector3
	Vel        Vector3
	EpochJD    float64       // Julian Date
	E_hal      float64       // Hamiltonian Energy for auditing
	Covariance [6][6]float64 // Position/velocity covariance (km², km²/s²) - added for ADITI 4.0 Pc computation
}

/**
 * SolarIndices represents the F10.7 and magnetic index values used
 * for high-precision atmospheric density modeling.
 */
type SolarIndices struct {
	F107  float64
	F107A float64
	AP    float64
}

/**
 * Observation represents a single state measurement (e.g. from optical or radar).
 * Used for orbit determination and differentiable propagation.
 */
type Observation struct {
	Timestamp time.Time
	Pos       Vector3
	Vel       Vector3
}

/**
 * PhysicsParams represents the environmental and structural parameters of an object.
 *
 * ADITI 4.0: BallisticCoeff enables per-object drag differentiation based on
 * historical TLE analysis and physical object properties.
 */
type PhysicsParams struct {
	Mass           float64
	DragArea       float64
	SRPArea        float64
	Cd             float64
	Cr             float64
	BallisticCoeff float64 // BC = m/(Cd*A) in kg/m^2, derived from TLE B*
}

/**
 * TLE represents a standard Two-Line Element set.
 * Moved to physics for architectural decoupling.
 */
type TLE struct {
	Name               string
	SatelliteNumber    int
	Classification     string
	InternationalDesig string
	CountryCode        string
	EpochYear          int
	EpochDay           float64
	MeanMotionDeriv1   float64
	MeanMotionDeriv2   float64
	BStar              float64
	ElementNumber      int
	Inclination        float64
	RAAN               float64
	Eccentricity       float64
	ArgOfPerigee       float64
	MeanAnomaly        float64
	MeanMotion         float64
	RevAtEpoch         int
	RawL1              string
	RawL2              string
}

/**
 * StateRefiner is an interface for ML-based state correction (for example PINNs).
 */
type StateRefiner interface {
	RefineStateAdaptive(state []float64, covariance [][]float64, dt float64, threshold float64) ([]float64, error)
}

/**
 * HistoryRefiner is an interface for sequence-based corrections (for example Transformers).
 */
type HistoryRefiner interface {
	AddHistory(objectID int, state State)
	Refine(objectID int, state []float64, covariance [][]float64) ([]float64, [][]float64, error)
}

/**
 * StateToVector converts a physical State object into a 6D float slice.
 * This is used for input to numerical integrators, Kalman filters,
 * and neural ODE kernels that expect a flat double-precision array.
 * It maps Pos[0..2] and Vel[0..2] to the first 6 elements.
 */
func StateToVector(s State) []float64 {
	return []float64{s.Pos[0], s.Pos[1], s.Pos[2], s.Vel[0], s.Vel[1], s.Vel[2]}
}

/**
 * VectorToState reconstructs a State object from a 6D float representation.
 * It is the inverse of StateToVector, used for decoding results from
 * integrators and ML models back into the domain-safe State schema.
 * Handles bounds checking to prevent index-out-of-range panics.
 */
func VectorToState(v []float64) State {
	if len(v) < 6 {
		return State{}
	}
	return State{
		Pos: Vector3{v[0], v[1], v[2]},
		Vel: Vector3{v[3], v[4], v[5]},
	}
}

// Normalized Constants for DU/TU conversion
const (
	RE_KM  = 6378.137
	MU_KM  = 398600.4418
	TU_SEC = 806.8109885 // sqrt(RE^3 / MU)
)

/**
 * ToCanonical converts standard km & km/s units to Earth-normalized Canonical Units (DU, TU).
 * Required for Axiom-Zero GHNN manifold stability.
 */
func (s *State) ToCanonical() (q, p Vector3) {
	q = Vector3{
		s.Pos[0] / RE_KM,
		s.Pos[1] / RE_KM,
		s.Pos[2] / RE_KM,
	}
	p = Vector3{
		s.Vel[0] * TU_SEC / RE_KM,
		s.Vel[1] * TU_SEC / RE_KM,
		s.Vel[2] * TU_SEC / RE_KM,
	}
	return q, p
}

/**
 * FromCanonical reconstructs a state from normalized units back to km and km/s.
 */
func FromCanonical(q, p Vector3, epochJD float64) State {
	scaleAcc := RE_KM / (TU_SEC * TU_SEC) // Not used for state, but for ref
	_ = scaleAcc

	return State{
		Pos: Vector3{
			q[0] * RE_KM,
			q[1] * RE_KM,
			q[2] * RE_KM,
		},
		Vel: Vector3{
			p[0] * RE_KM / TU_SEC,
			p[1] * RE_KM / TU_SEC,
			p[2] * RE_KM / TU_SEC,
		},
		EpochJD: epochJD,
	}
}

/**
 * GetKeplerianEnergy calculates the specific energy (0.5v^2 - mu/r).
 */
func (s *State) GetKeplerianEnergy() float64 {
	r := math.Sqrt(s.Pos[0]*s.Pos[0] + s.Pos[1]*s.Pos[1] + s.Pos[2]*s.Pos[2])
	v2 := s.Vel[0]*s.Vel[0] + s.Vel[1]*s.Vel[1] + s.Vel[2]*s.Vel[2]
	return 0.5*v2 - MU_KM/r
}

/**
 * OrbitRegime represents the orbital regime for covariance tuning.
 */
type OrbitRegime int

const (
	RegimeLEO     OrbitRegime = iota // Low Earth Orbit (< 2000 km)
	RegimeMEO                        // Medium Earth Orbit (2000-35786 km)
	RegimeGEO                        // Geostationary Orbit (~35786 km)
	RegimeHEO                        // High Eccentricity Orbit
	RegimeUnknown                    // Unknown/unspecified regime
)

/**
 * ConstructDefaultCovarianceFromTLE creates a default 6x6 covariance matrix from TLE data.
 *
 * Reference: Hejduk 2019, "Covariance Realism in Conjunction Assessment"
 *
 * TLE-derived covariance is CONSERVATIVE - actual errors may be smaller.
 * This enables Pc computation from TLE data where no formal covariance exists.
 *
 * Typical TLE errors (1-sigma):
 *   LEO:  1.0 km position, 0.001 km/s velocity
 *   GEO:  5.0 km position, 0.0005 km/s velocity (worse due to SRP/tesseral effects)
 *
 * @param tle The TLE data
 * @return 6x6 covariance matrix [km², km²/s²]
 */
func ConstructDefaultCovarianceFromTLE(tle TLE) [6][6]float64 {
	// Determine orbit regime from TLE parameters
	regime := classifyOrbitRegime(tle)

	var posSigma, velSigma float64

	switch regime {
	case RegimeLEO:
		// LEO: Better tracking accuracy, higher drag uncertainty
		posSigma = 1.0   // km (1-sigma)
		velSigma = 0.001 // km/s (1-sigma) ~1 m/s
	case RegimeGEO:
		// GEO: Worse tracking, better velocity (drag-free), SRP uncertainty
		posSigma = 5.0    // km (1-sigma)
		velSigma = 0.0005 // km/s (1-sigma) ~0.5 m/s
	case RegimeMEO:
		// MEO: Intermediate accuracy (GPS, Galileo, etc.)
		posSigma = 2.0    // km (1-sigma)
		velSigma = 0.0008 // km/s (1-sigma) ~0.8 m/s
	case RegimeHEO:
		// HEO: Highly eccentric, variable accuracy
		posSigma = 3.0   // km (1-sigma)
		velSigma = 0.001 // km/s (1-sigma) ~1 m/s
	default:
		// Conservative default
		posSigma = 2.0
		velSigma = 0.001
	}

	// Build diagonal covariance (uncorrelated position/velocity errors)
	// State order: [x, y, z, vx, vy, vz]
	var cov [6][6]float64

	// Position variances (diagonal)
	cov[0][0] = posSigma * posSigma
	cov[1][1] = posSigma * posSigma * 0.25 // In-track typically better (0.5 km for LEO)
	cov[2][2] = posSigma * posSigma * 0.25 // Cross-track typically better

	// Velocity variances (diagonal)
	cov[3][3] = velSigma * velSigma
	cov[4][4] = velSigma * velSigma
	cov[5][5] = velSigma * velSigma

	// Add small cross-correlation for realistic covariance structure
	// Position-velocity correlation is small but non-zero for TLE data
	corrCoeff := 0.1
	cov[0][3] = corrCoeff * posSigma * velSigma
	cov[3][0] = cov[0][3]
	cov[1][4] = corrCoeff * posSigma * velSigma * 0.5
	cov[4][1] = cov[1][4]
	cov[2][5] = corrCoeff * posSigma * velSigma * 0.5
	cov[5][2] = cov[2][5]

	return cov
}

/**
 * classifyOrbitRegime determines the orbit regime from TLE parameters.
 */
func classifyOrbitRegime(tle TLE) OrbitRegime {
	// Semi-major axis from mean motion (using standard formula)
	// n = sqrt(μ/a³) => a = (μ/n²)^(1/3)
	// Mean motion is in revs/day, convert to rad/s
	nRadS := tle.MeanMotion * 2.0 * math.Pi / 86400.0
	if nRadS <= 0 {
		return RegimeUnknown
	}

	a := math.Pow(MU_KM/(nRadS*nRadS), 1.0/3.0)
	rPerigee := a * (1.0 - tle.Eccentricity)
	rApogee := a * (1.0 + tle.Eccentricity)
	altPerigee := rPerigee - RE_KM
	altApogee := rApogee - RE_KM

	// Classify by altitude and eccentricity
	if tle.Eccentricity > 0.5 {
		return RegimeHEO
	}

	if altApogee < 2000.0 {
		return RegimeLEO
	}

	if altPerigee > 30000.0 && altApogee < 42000.0 {
		// Near-geosynchronous
		return RegimeGEO
	}

	if altPerigee > 2000.0 && altApogee < 35000.0 {
		return RegimeMEO
	}

	return RegimeUnknown
}
