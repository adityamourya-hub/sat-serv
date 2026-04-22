package rf

/**
 * TDOA/FDOA Localization for Passive RF Characterization
 *
 * RF-based SDA for GEO objects without optical limitations
 *
 * Algorithms:
 * - TDOA (Time Difference of Arrival): Multi-station time correlation
 * - FDOA (Frequency Difference of Arrival): Doppler-based velocity
 * - Particle Filter: Non-linear, non-Gaussian state estimation
 * - ML Estimation: Maximum likelihood for sub-arcsecond resolution
 *
 * References:
 * - imasmitja/TDOA_algorithms_SR: Particle filter & ML estimation
 * - Kratos KnownSpace: Industry RF survey concepts
 * - Differential Tracking Interferometer for GEO
 */

import (
	"math"
	"math/rand/v2"
)

// Constants
const (
	LIGHT_SPEED = 299792.458 // km/s

	// Orbit constants
	GEO_RADIUS  = 42164.0 // km (geostationary orbit radius)
	LEO_ALT_MIN = 200.0   // km (minimum LEO altitude)
	LEO_ALT_MAX = 1200.0  // km (maximum LEO altitude)

	// Grid search parameters
	GRID_STEP_DEG = 5.0   // degrees for coarse grid search
	GRID_STEP_KM  = 100.0 // km for 3D bounding box search
)

// RFStation represents a ground/space-based RF receiver
type RFStation struct {
	ID        string
	Position  [3]float64 // ECI km
	ClockBias float64    // seconds
	FreqBias  float64    // Hz

	// Antenna characteristics
	Gain      float64 // dBi
	Beamwidth float64 // degrees
	Frequency float64 // MHz (center)
	Bandwidth float64 // MHz
}

// RFSignal represents detected satellite transmission
type RFSignal struct {
	Frequency float64 // MHz
	Bandwidth float64 // MHz
	Power     float64 // dBm
	Timestamp float64 // Julian date
	StationID string

	// Raw measurements
	TOA   float64 // Time of Arrival (seconds from epoch)
	FOA   float64 // Frequency of Arrival (Hz offset from center)
	Phase float64 // Signal phase (radians)
	SNR   float64 // dB
}

// TDOAMeasurement represents time difference between two stations
type TDOAMeasurement struct {
	Station1  string
	Station2  string
	DeltaT    float64 // seconds
	Sigma     float64 // measurement noise (seconds)
	Timestamp float64 // Julian date
}

// FDOAMeasurement represents frequency difference
type FDOAMeasurement struct {
	Station1  string
	Station2  string
	DeltaF    float64 // Hz
	Sigma     float64 // Hz
	Timestamp float64
}

// TDOALocalizer performs multi-station RF localization
type TDOALocalizer struct {
	Stations []RFStation
}

// NewTDOALocalizer creates localizer with station network
func NewTDOALocalizer(stations []RFStation) *TDOALocalizer {
	return &TDOALocalizer{Stations: stations}
}

// ComputeTDOA calculates expected time difference for given emitter position
func (t *TDOALocalizer) ComputeTDOA(emitterPos [3]float64, s1, s2 RFStation) float64 {
	// Distance from each station to emitter
	d1 := distance(emitterPos, s1.Position)
	d2 := distance(emitterPos, s2.Position)

	// Time of flight difference
	deltaDistance := d1 - d2                 // km
	deltaTime := deltaDistance / LIGHT_SPEED // seconds

	// Account for clock bias
	clockBias := s1.ClockBias - s2.ClockBias

	return deltaTime + clockBias
}

// ComputeFDOA calculates expected frequency difference (Doppler)
func (t *TDOALocalizer) ComputeFDOA(emitterPos, emitterVel [3]float64, s1, s2 RFStation, freq float64) float64 {
	// Unit vectors from stations to emitter
	los1 := normalize(subtract(emitterPos, s1.Position))
	los2 := normalize(subtract(emitterPos, s2.Position))

	// Range rate (velocity dot LOS)
	rr1 := dot(emitterVel, los1)
	rr2 := dot(emitterVel, los2)

	// Doppler shift (Hz)
	// f_observed = f_emitted * (1 - v_r/c)
	c := LIGHT_SPEED * 1000 // m/s
	f1 := freq * 1e6 * (1.0 - rr1*1000/c)
	f2 := freq * 1e6 * (1.0 - rr2*1000/c)

	// Frequency bias
	bias := s1.FreqBias - s2.FreqBias

	return (f1 - f2) + bias
}

// Particle represents a hypothesis in particle filter
type Particle struct {
	Position [3]float64
	Velocity [3]float64
	Weight   float64
}

// ParticleFilterTDOA performs Bayesian localization
type ParticleFilterTDOA struct {
	Particles []Particle
	Stations  []RFStation
	N         int // Number of particles
}

// NewParticleFilter creates PF for TDOA localization
func NewParticleFilter(stations []RFStation, nParticles int) *ParticleFilterTDOA {
	pf := &ParticleFilterTDOA{
		Stations:  stations,
		N:         nParticles,
		Particles: make([]Particle, nParticles),
	}

	// Initialize particles around GEO ring
	for i := 0; i < nParticles; i++ {
		angle := float64(i) * 2 * math.Pi / float64(nParticles)
		pf.Particles[i] = Particle{
			Position: [3]float64{
				GEO_RADIUS * math.Cos(angle),
				GEO_RADIUS * math.Sin(angle),
				0,
			},
			Velocity: [3]float64{
				-3.07 * math.Sin(angle),
				3.07 * math.Cos(angle),
				0,
			},
			Weight: 1.0 / float64(nParticles),
		}
	}

	return pf
}

// Update performs particle filter update with TDOA measurements
func (pf *ParticleFilterTDOA) Update(tdoas []TDOAMeasurement) {
	// Weight particles based on measurement likelihood
	for i := range pf.Particles {
		likelihood := 1.0

		for _, tdoa := range tdoas {
			// Find stations
			var s1, s2 *RFStation
			for j := range pf.Stations {
				if pf.Stations[j].ID == tdoa.Station1 {
					s1 = &pf.Stations[j]
				}
				if pf.Stations[j].ID == tdoa.Station2 {
					s2 = &pf.Stations[j]
				}
			}

			if s1 == nil || s2 == nil {
				continue
			}

			// Expected TDOA
			expected := pf.computeExpectedTDOA(pf.Particles[i].Position, *s1, *s2)

			// Measurement likelihood (Gaussian)
			residual := tdoa.DeltaT - expected
			likelihood *= math.Exp(-0.5 * (residual * residual) / (tdoa.Sigma * tdoa.Sigma))
		}

		pf.Particles[i].Weight *= likelihood
	}

	// Normalize weights
	var sumWeights float64
	for i := range pf.Particles {
		sumWeights += pf.Particles[i].Weight
	}

	if sumWeights > 0 {
		for i := range pf.Particles {
			pf.Particles[i].Weight /= sumWeights
		}
	}

	// Resample (systematic resampling)
	pf.resample()
}

// computeExpectedTDOA calculates expected measurement
func (pf *ParticleFilterTDOA) computeExpectedTDOA(pos [3]float64, s1, s2 RFStation) float64 {
	d1 := distance(pos, s1.Position)
	d2 := distance(pos, s2.Position)
	return (d1 - d2) / LIGHT_SPEED
}

// resample performs systematic resampling
func (pf *ParticleFilterTDOA) resample() {
	newParticles := make([]Particle, pf.N)

	// Cumulative weights
	cumsum := make([]float64, pf.N)
	cumsum[0] = pf.Particles[0].Weight
	for i := 1; i < pf.N; i++ {
		cumsum[i] = cumsum[i-1] + pf.Particles[i].Weight
	}

	// Systematic resampling
	step := 1.0 / float64(pf.N)
	start := step * 0.5 // Random offset [0, step)

	for i := 0; i < pf.N; i++ {
		u := start + float64(i)*step

		// Find particle
		idx := 0
		for idx < pf.N && cumsum[idx] < u {
			idx++
		}
		if idx >= pf.N {
			idx = pf.N - 1
		}

		newParticles[i] = pf.Particles[idx]
		newParticles[i].Weight = 1.0 / float64(pf.N)

		// Add small noise for diversity
		newParticles[i].Position[0] += (rand.Float64() - 0.5) * 0.1
		newParticles[i].Position[1] += (rand.Float64() - 0.5) * 0.1
		newParticles[i].Position[2] += (rand.Float64() - 0.5) * 0.01
	}

	pf.Particles = newParticles
}

// GetEstimate returns weighted mean position
func (pf *ParticleFilterTDOA) GetEstimate() ([3]float64, float64) {
	var mean [3]float64
	var covariance float64

	for _, p := range pf.Particles {
		mean[0] += p.Weight * p.Position[0]
		mean[1] += p.Weight * p.Position[1]
		mean[2] += p.Weight * p.Position[2]
	}

	// Compute covariance (simplified - trace only)
	for _, p := range pf.Particles {
		dx := p.Position[0] - mean[0]
		dy := p.Position[1] - mean[1]
		dz := p.Position[2] - mean[2]
		covariance += p.Weight * (dx*dx + dy*dy + dz*dz)
	}

	return mean, covariance
}

// MLLocalizer performs Maximum Likelihood estimation
type MLLocalizer struct {
	Stations []RFStation
}

// NewMLLocalizer creates ML estimator
func NewMLLocalizer(stations []RFStation) *MLLocalizer {
	return &MLLocalizer{Stations: stations}
}

// Localize finds ML position estimate using coarse grid + Levenberg-Marquardt refinement
func (ml *MLLocalizer) Localize(tdoas []TDOAMeasurement) [3]float64 {
	// Stage 1: Coarse grid search (GRID_STEP_DEG resolution for speed)
	bestPos := [3]float64{GEO_RADIUS, 0, 0} // GEO default
	bestLikelihood := -1e18

	for lon := 0.0; lon < 360.0; lon += GRID_STEP_DEG {
		for lat := -10.0; lat <= 10.0; lat += GRID_STEP_DEG {
			pos := geoToCartesian(GEO_RADIUS, lon, lat)
			likelihood := ml.computeLikelihood(pos, tdoas)
			if likelihood > bestLikelihood {
				bestLikelihood = likelihood
				bestPos = pos
			}
		}
	}

	// Stage 2: Levenberg-Marquardt refinement (20 iterations, λ=0.01)
	bestPos = ml.levenbergMarquardt(bestPos, tdoas, 20, 0.01)

	return bestPos
}

// geoToCartesian converts GEO radius, lon, lat to ECI
func geoToCartesian(r, lonDeg, latDeg float64) [3]float64 {
	lonRad := lonDeg * math.Pi / 180.0
	latRad := latDeg * math.Pi / 180.0
	return [3]float64{
		r * math.Cos(latRad) * math.Cos(lonRad),
		r * math.Cos(latRad) * math.Sin(lonRad),
		r * math.Sin(latRad),
	}
}

// computeLikelihood calculates negative chi-squared (higher is better)
func (ml *MLLocalizer) computeLikelihood(pos [3]float64, tdoas []TDOAMeasurement) float64 {
	likelihood := 0.0
	for _, tdoa := range tdoas {
		s1, s2 := ml.findStations(tdoa.Station1, tdoa.Station2)
		if s1 == nil || s2 == nil {
			continue
		}
		expected := ml.computeTDOA(pos, *s1, *s2)
		residual := tdoa.DeltaT - expected
		likelihood -= 0.5 * (residual * residual) / (tdoa.Sigma * tdoa.Sigma)
	}
	return likelihood
}

// findStations looks up stations by ID
func (ml *MLLocalizer) findStations(id1, id2 string) (*RFStation, *RFStation) {
	var s1, s2 *RFStation
	for j := range ml.Stations {
		if ml.Stations[j].ID == id1 {
			s1 = &ml.Stations[j]
		}
		if ml.Stations[j].ID == id2 {
			s2 = &ml.Stations[j]
		}
	}
	return s1, s2
}

// levenbergMarquardt performs LM optimization for TDOA refinement
// Solves: argmin_x Σ (measurement_i - model_i(x))² / sigma_i²
func (ml *MLLocalizer) levenbergMarquardt(initial [3]float64, tdoas []TDOAMeasurement, maxIter int, lambda float64) [3]float64 {
	x := initial

	for iter := 0; iter < maxIter; iter++ {
		// Compute residuals and Jacobian
		residuals := make([]float64, 0, len(tdoas))
		jacobian := make([][]float64, 0, len(tdoas)) // J[i][j] = ∂residual_i/∂x_j

		for _, tdoa := range tdoas {
			s1, s2 := ml.findStations(tdoa.Station1, tdoa.Station2)
			if s1 == nil || s2 == nil {
				continue
			}

			expected := ml.computeTDOA(x, *s1, *s2)
			residual := tdoa.DeltaT - expected
			weight := 1.0 / tdoa.Sigma

			// Compute gradient of TDOA w.r.t position (chain rule)
			// dTDOA/dx = (1/c) * ((x-s1.x)/d1 - (x-s2.x)/d2)
			d1 := distance(x, s1.Position)
			d2 := distance(x, s2.Position)

			grad := [3]float64{}
			for i := 0; i < 3; i++ {
				grad[i] = ((x[i] - s1.Position[i]) / d1) - ((x[i] - s2.Position[i]) / d2)
				grad[i] /= LIGHT_SPEED * tdoa.Sigma // Include weighting
			}

			residuals = append(residuals, residual*weight)
			jacobian = append(jacobian, []float64{grad[0], grad[1], grad[2]})
		}

		if len(residuals) == 0 {
			break
		}

		// J^T * J (Hessian approximation)
		jtj := [3][3]float64{}
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				sum := 0.0
				for k := range jacobian {
					sum += jacobian[k][i] * jacobian[k][j]
				}
				jtj[i][j] = sum
				if i == j {
					jtj[i][j] += lambda // Marquardt damping
				}
			}
		}

		// J^T * r (gradient of cost)
		jtr := [3]float64{}
		for i := 0; i < 3; i++ {
			sum := 0.0
			for k := range jacobian {
				sum += jacobian[k][i] * residuals[k]
			}
			jtr[i] = sum
		}

		// Solve (J^T*J + λI) δ = J^T*r using Gaussian elimination
		delta := Solve3x3(jtj, jtr)

		// Update position
		xNew := [3]float64{x[0] + delta[0], x[1] + delta[1], x[2] + delta[2]}

		// Adaptive damping
		oldLikelihood := ml.computeLikelihood(x, tdoas)
		newLikelihood := ml.computeLikelihood(xNew, tdoas)

		if newLikelihood > oldLikelihood {
			x = xNew
			lambda *= 0.7 // Reduce damping on success
		} else {
			lambda *= 2.0 // Increase damping on failure
		}

		// Convergence check
		deltaNorm := math.Sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2])
		if deltaNorm < 1e-6 { // 1mm convergence
			break
		}
	}

	return x
}

// Solve3x3 solves a 3x3 linear system Ax = b using Gaussian elimination
// Exported for use across the rf package
func Solve3x3(A [3][3]float64, b [3]float64) [3]float64 {
	// Augmented matrix
	M := [3][4]float64{
		{A[0][0], A[0][1], A[0][2], b[0]},
		{A[1][0], A[1][1], A[1][2], b[1]},
		{A[2][0], A[2][1], A[2][2], b[2]},
	}

	// Forward elimination
	for i := 0; i < 3; i++ {
		// Partial pivoting
		maxRow := i
		maxVal := math.Abs(M[i][i])
		for k := i + 1; k < 3; k++ {
			if math.Abs(M[k][i]) > maxVal {
				maxVal = math.Abs(M[k][i])
				maxRow = k
			}
		}
		if maxVal < 1e-15 {
			continue // Singular or near-singular
		}
		M[i], M[maxRow] = M[maxRow], M[i]

		// Eliminate
		for k := i + 1; k < 3; k++ {
			factor := M[k][i] / M[i][i]
			for j := i; j < 4; j++ {
				M[k][j] -= factor * M[i][j]
			}
		}
	}

	// Back substitution
	x := [3]float64{}
	for i := 2; i >= 0; i-- {
		sum := M[i][3]
		for j := i + 1; j < 3; j++ {
			sum -= M[i][j] * x[j]
		}
		if math.Abs(M[i][i]) > 1e-15 {
			x[i] = sum / M[i][i]
		}
	}
	return x
}

func (ml *MLLocalizer) computeTDOA(pos [3]float64, s1, s2 RFStation) float64 {
	d1 := distance(pos, s1.Position)
	d2 := distance(pos, s2.Position)
	return (d1 - d2) / LIGHT_SPEED
}

// Utility functions
func distance(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func subtract(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

func dot(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func normalize(v [3]float64) [3]float64 {
	mag := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
	if mag < 1e-10 {
		return [3]float64{0, 0, 0}
	}
	return [3]float64{v[0] / mag, v[1] / mag, v[2] / mag}
}
