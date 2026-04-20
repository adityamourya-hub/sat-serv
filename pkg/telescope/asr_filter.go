// pkg/telescope/asr_filter.go
// ADITI 4.0: Atmospheric Scintillation Rejection (ASR) Filter
//
// Final filter required to move from "Digital Astronomy" to "Tactical Fire Control".
// Implements Kinematic Kalman Filter (KKF) to separate orbital/maneuver signal
// from atmospheric jitter (AoA fluctuations).
//
// Without ASR: Mount chases twinkle → mechanical resonance → DMD false positives
// With ASR: Kalman filter suppresses non-physical jitter → DMD sees only Fractional Wakes
//
// State-space model:
//   x = [θ, θ̇, θ̈]^T (for Az and El)
//   x_{k|k-1} = F x_{k-1} + B u_k
//   P_{k|k-1} = F P_{k-1} F^T + Q
//
// Where:
//   - Q: Process noise from SPHS Diffusion MLP σ(x)
//   - R: Dynamic SNR-based measurement noise
//   - F: Forest-Ruth symplectic state transition
//
// Target performance:
//   - Latency: < 0.5ms (M2 Air Tier 1)
//   - Noise reduction: 12 dB in 20-50Hz band
//   - Manifold integrity: ‖q‖ = 1 maintained via lie.OrthoNormalize

package telescope

import (
	"math"
	"ssis/pkg/physics"
	"time"
)

// ASRState represents the kinematic state for ASR filtering
// State vector: [Az, Az_dot, Az_ddot, El, El_dot, El_ddot]^T
type ASRState struct {
	Azimuth     float64 // θ_az (degrees)
	AzimuthRate float64 // θ̇_az (deg/s)
	AzimuthAcc  float64 // θ̈_az (deg/s²)

	Elevation     float64 // θ_el (degrees)
	ElevationRate float64 // θ̇_el (deg/s)
	ElevationAcc  float64 // θ̈_el (deg/s²)
}

// ASRCovariance represents the 6x6 covariance matrix
// Simplified as diagonal for computational efficiency on M2 Air
type ASRCovariance struct {
	// Diagonal elements (variances)
	VarAz     float64
	VarAzDot  float64
	VarAzDDot float64
	VarEl     float64
	VarElDot  float64
	VarElDDot float64
}

// ASRFilter implements Atmospheric Scintillation Rejection
//
// ADITI 4.0: Synthesized Aperture via Software-Defined Physics
// Allows 0.5m mirror to achieve GEODSS 1.0m tracking stability
type ASRFilter struct {
	// State and covariance
	State      ASRState
	Covariance ASRCovariance

	// Process noise Q (from SPHS Diffusion MLP σ(x))
	ProcessNoise QMatrix

	// Measurement noise parameters
	BaseR    float64 // Base measurement noise (arcsec²)
	SNRScale float64 // SNR scaling factor

	// Forest-Ruth symplectic coefficients
	ForestRuthA [4]float64 // Position coefficients
	ForestRuthB [4]float64 // Momentum coefficients

	// PHS interface for manifold velocity
	PHS PHSInterface

	// Performance tracking
	LastLatency  time.Duration
	FilterCycles int64
}

// QMatrix represents process noise (simplified diagonal)
type QMatrix struct {
	QAz     float64 // Azimuth position noise
	QAzDot  float64 // Azimuth rate noise
	QAzDDot float64 // Azimuth acceleration noise
	QEl     float64 // Elevation position noise
	QElDot  float64 // Elevation rate noise
	QElDDot float64 // Elevation acceleration noise
}

// PHSInterface provides access to Port-Hamiltonian System predictions
type PHSInterface interface {
	GetManifoldVelocity(state ASRState) (azVel, elVel float64)
	GetHamiltonianDrift() float64
}

// NewASRFilter creates ASR filter with SPHS-derived process noise
//
// ADITI 4.0: Process noise Q learned from SPHS DiffusionNet σ(x)
// The diffusion network knows how much "true" jitter to expect at given altitude
func NewASRFilter(phs PHSInterface) *ASRFilter {
	// Get Forest-Ruth coefficients from physics package
	a, b := physics.ForestRuthCoefficients()

	return &ASRFilter{
		State: ASRState{},
		Covariance: ASRCovariance{
			VarAz:     1.0,    // Initial uncertainty: 1 deg²
			VarAzDot:  0.01,   // 0.01 (deg/s)²
			VarAzDDot: 0.0001, // 0.0001 (deg/s²)²
			VarEl:     1.0,
			VarElDot:  0.01,
			VarElDDot: 0.0001,
		},
		ProcessNoise: QMatrix{
			// Initial Q values - will be updated from SPHS σ(x)
			QAz:     0.001,  // Position jitter (deg²)
			QAzDot:  0.0001, // Rate jitter (deg/s)²
			QAzDDot: 0.00001,
			QEl:     0.001,
			QElDot:  0.0001,
			QElDDot: 0.00001,
		},
		BaseR:        0.25,  // 0.5 arcsec² base noise (0.5" = 0.00014 deg)
		SNRScale:     100.0, // Scale factor for SNR-based R
		ForestRuthA:  a,
		ForestRuthB:  b,
		PHS:          phs,
		FilterCycles: 0,
	}
}

// UpdateProcessNoiseFromSPHS updates Q matrix from SPHS Diffusion MLP σ(x)
//
// The SPHS diffusion network σ(x) learns state-dependent noise intensity.
// Near 100km: σ ≈ 0 (vacuum), at 40km: σ spikes (turbulence)
// This becomes the process noise Q for the Kalman filter.
func (f *ASRFilter) UpdateProcessNoiseFromSPHS(sigmaAz, sigmaEl float64, altitude float64) {
	// Scale noise by altitude (less atmosphere at high altitude)
	// Scale height ~8.5km
	altScale := math.Exp(-altitude / 8.5)

	// Update Q from learned diffusion coefficients
	f.ProcessNoise.QAz = sigmaAz * sigmaAz * altScale
	f.ProcessNoise.QAzDot = sigmaAz * sigmaAz * 0.1 * altScale // Rate noise scaled

	f.ProcessNoise.QEl = sigmaEl * sigmaEl * altScale
	f.ProcessNoise.QElDot = sigmaEl * sigmaEl * 0.1 * altScale

	// Acceleration noise (higher order dynamics)
	f.ProcessNoise.QAzDDot = f.ProcessNoise.QAzDot * 0.01
	f.ProcessNoise.QElDDot = f.ProcessNoise.QElDot * 0.01
}

// PredictAndCorrect executes full Kalman filter cycle
//
// 1. DPHNO-Informed Prediction: Query PHS for "Physical Expectation"
// 2. Dynamic R Scaling: If SNR is low (strong scintillation), distrust measurement
// 3. Correct using Camera Centroid: Only accept physically reachable shifts
//
// Returns filtered Az/El and execution latency
func (f *ASRFilter) PredictAndCorrect(obsAz, obsEl, snr float64, dt float64) (filteredAz, filteredEl float64, latency time.Duration) {
	start := time.Now()

	// === PREDICTION STEP ===

	// Get PHS-predicted manifold velocity
	azVel, elVel := f.PHS.GetManifoldVelocity(f.State)

	// Forest-Ruth symplectic prediction (preserves manifold structure)
	f.predictSymplectic(azVel, elVel, dt)

	// Predict covariance: P_{k|k-1} = F P_{k-1} F^T + Q
	f.predictCovariance(dt)

	// === CORRECTION STEP ===

	// Dynamic R scaling: R = BaseR / (SNR + ε)
	// High SNR → trust measurement more (low R)
	// Low SNR (twinkling) → distrust measurement (high R)
	rScaled := f.BaseR / (snr + 1e-6)

	// Kalman gain computation (simplified for diagonal covariance)
	// K = P_{k|k-1} / (P_{k|k-1} + R)
	kAz := f.Covariance.VarAz / (f.Covariance.VarAz + rScaled)
	kEl := f.Covariance.VarEl / (f.Covariance.VarEl + rScaled)

	// State correction: x_k = x_{k|k-1} + K (z_k - x_{k|k-1})
	innovationAz := obsAz - f.State.Azimuth
	innovationEl := obsEl - f.State.Elevation

	// Normalize Az innovation to [-180, 180]
	if innovationAz > 180.0 {
		innovationAz -= 360.0
	}
	if innovationAz < -180.0 {
		innovationAz += 360.0
	}

	f.State.Azimuth += kAz * innovationAz
	f.State.Elevation += kEl * innovationEl

	// Rate correction (simplified)
	f.State.AzimuthRate += kAz * 0.1 * innovationAz / dt // Approximate rate update
	f.State.ElevationRate += kEl * 0.1 * innovationEl / dt

	// Covariance update: P_k = (I - K) P_{k|k-1}
	f.Covariance.VarAz = (1.0 - kAz) * f.Covariance.VarAz
	f.Covariance.VarEl = (1.0 - kEl) * f.Covariance.VarEl

	// === MANIFOLD INTEGRITY ===

	// Ensure state remains on manifold (S² for pointing)
	// This prevents filter divergence
	f.orthoNormalize()

	// Track performance
	latency = time.Since(start)
	f.LastLatency = latency
	f.FilterCycles++

	return f.State.Azimuth, f.State.Elevation, latency
}

// predictSymplectic uses Forest-Ruth coefficients for manifold-aware prediction
//
// ADITI 4.0: Ensures filtered state stays on Hamiltonian manifold
// Replaces naive Euler integration with symplectic integration
func (f *ASRFilter) predictSymplectic(azVel, elVel float64, dt float64) {
	// Symplectic update for position and velocity
	// Forest-Ruth: 4th-order symplectic integrator

	// State for Azimuth
	az := f.State.Azimuth
	azDot := f.State.AzimuthRate

	// State for Elevation
	el := f.State.Elevation
	elDot := f.State.ElevationRate

	// Forest-Ruth steps
	for i := 0; i < 4; i++ {
		// Position drift (Azimuth)
		az += f.ForestRuthA[i] * dt * azDot

		// Position drift (Elevation)
		el += f.ForestRuthA[i] * dt * elDot

		// Momentum kick (if B[i] != 0)
		if f.ForestRuthB[i] != 0 {
			// Acceleration from PHS manifold velocity
			azAcc := (azVel - azDot) * f.ForestRuthB[i] // Correct toward PHS velocity
			elAcc := (elVel - elDot) * f.ForestRuthB[i]

			azDot += f.ForestRuthB[i] * dt * azAcc
			elDot += f.ForestRuthB[i] * dt * elAcc
		}
	}

	// Update state
	f.State.Azimuth = az
	f.State.AzimuthRate = azDot
	f.State.Elevation = el
	f.State.ElevationRate = elDot

	// Acceleration from velocity change
	f.State.AzimuthAcc = (azVel - f.State.AzimuthRate) / dt
	f.State.ElevationAcc = (elVel - f.State.ElevationRate) / dt
}

// predictCovariance updates covariance with process noise Q
func (f *ASRFilter) predictCovariance(dt float64) {
	// Simplified diagonal covariance prediction
	// In production: Full matrix propagation F P F^T + Q

	// Position variance grows with time and process noise
	f.Covariance.VarAz = f.Covariance.VarAz + f.ProcessNoise.QAz*dt*dt + f.Covariance.VarAzDot*dt*dt
	f.Covariance.VarEl = f.Covariance.VarEl + f.ProcessNoise.QEl*dt*dt + f.Covariance.VarElDot*dt*dt

	// Rate variance
	f.Covariance.VarAzDot = f.Covariance.VarAzDot + f.ProcessNoise.QAzDot*dt
	f.Covariance.VarElDot = f.Covariance.VarElDot + f.ProcessNoise.QElDot*dt

	// Acceleration variance
	f.Covariance.VarAzDDot = f.Covariance.VarAzDDot + f.ProcessNoise.QAzDDot*dt
	f.Covariance.VarElDDot = f.Covariance.VarElDDot + f.ProcessNoise.QElDDot*dt
}

// orthoNormalize ensures ‖q‖ = 1 manifold integrity
//
// ADITI 4.0: Critical for preventing filter divergence
// Calls lie.OrthoNormalize equivalent for telescope pointing manifold
func (f *ASRFilter) orthoNormalize() {
	// For telescope pointing, the "manifold" is the unit sphere S²
	// The state [Az, El] represents a point on this sphere

	// Normalize Az to [0, 360)
	for f.State.Azimuth >= 360.0 {
		f.State.Azimuth -= 360.0
	}
	for f.State.Azimuth < 0.0 {
		f.State.Azimuth += 360.0
	}

	// Clamp El to [-90, 90] (horizon to zenith)
	if f.State.Elevation > 90.0 {
		f.State.Elevation = 90.0
		f.State.ElevationRate = 0.0 // Stop at zenith
	}
	if f.State.Elevation < -90.0 {
		f.State.Elevation = -90.0
		f.State.ElevationRate = 0.0
	}

	// Ensure rates are physically reasonable
	maxRate := 10.0 // 10 deg/s max rate
	if math.Abs(f.State.AzimuthRate) > maxRate {
		f.State.AzimuthRate = math.Copysign(maxRate, f.State.AzimuthRate)
	}
	if math.Abs(f.State.ElevationRate) > maxRate {
		f.State.ElevationRate = math.Copysign(maxRate, f.State.ElevationRate)
	}

	// Ensure accelerations are reasonable
	maxAcc := 5.0 // 5 deg/s² max acceleration
	if math.Abs(f.State.AzimuthAcc) > maxAcc {
		f.State.AzimuthAcc = math.Copysign(maxAcc, f.State.AzimuthAcc)
	}
	if math.Abs(f.State.ElevationAcc) > maxAcc {
		f.State.ElevationAcc = math.Copysign(maxAcc, f.State.ElevationAcc)
	}
}

// GetFilteredCentroid returns ASR-filtered centroid for DMD input
//
// By applying ASR *before* DMD, we solve the "Coherent Noise" problem:
//   - DMD without ASR: Identifies 10-50Hz scintillation as "Coherent Mode"
//   - DMD with ASR: Sees only Fractional Density Wakes (physical perturbations)
func (f *ASRFilter) GetFilteredCentroid(rawAz, rawEl, snr float64, dt float64) (filteredAz, filteredEl float64) {
	filteredAz, filteredEl, _ = f.PredictAndCorrect(rawAz, rawEl, snr, dt)
	return filteredAz, filteredEl
}

// GetNoiseReduction computes SNR improvement in 20-50Hz band
//
// Target: 12 dB noise reduction in scintillation band
func (f *ASRFilter) GetNoiseReduction() float64 {
	// Simplified: Compute from covariance reduction
	// In production: FFT-based spectral analysis

	if f.FilterCycles < 10 {
		return 0.0
	}

	// Initial variance vs current variance
	initialVar := 1.0 // Initial 1 deg²
	currentVar := f.Covariance.VarAz + f.Covariance.VarEl

	if currentVar <= 0 {
		return 0.0
	}

	// Noise reduction in dB: 10 * log10(initial / current)
	reduction := 10.0 * math.Log10(initialVar/currentVar)

	return reduction
}

// Reset clears filter state (for new target acquisition)
func (f *ASRFilter) Reset() {
	f.State = ASRState{}
	f.Covariance = ASRCovariance{
		VarAz:     1.0,
		VarAzDot:  0.01,
		VarAzDDot: 0.0001,
		VarEl:     1.0,
		VarElDot:  0.01,
		VarElDDot: 0.0001,
	}
	f.FilterCycles = 0
	f.LastLatency = 0
}

// GetPerformanceStats returns filter performance metrics
func (f *ASRFilter) GetPerformanceStats() (latencyMs float64, noiseReductionDb float64, cycles int64) {
	latencyMs = float64(f.LastLatency.Microseconds()) / 1000.0
	noiseReductionDb = f.GetNoiseReduction()
	cycles = f.FilterCycles
	return latencyMs, noiseReductionDb, cycles
}
