package rf

/**
 * TDOA-PDOA Localization with Particle Swarm Optimization (PSO)
 *
 * Complete implementation of Chen et al. IEEE WCL 2020 (WCL2020-0388)
 *
 * Two Algorithms:
 *   1. TDOA-PDOA-1: PDOA only (with TDOA initialization)
 *   2. TDOA-PDOA-2: Joint TDOA-PDOA cost function
 *
 * Key Features:
 *   - PSO for global optimization (avoids local minima)
 *   - Phase wrapping handling (2π ambiguity)
 *   - Proper weighting coefficients (bm)
 *   - Search space bounds based on TDOA uncertainty
 *
 * References:
 *   - Chen Hui et al., "A Joint TDOA-PDOA Localization Approach
 *     Using Particle Swarm Optimization", IEEE WCL, 2020
 */

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// PSOConfig configures the Particle Swarm Optimizer
type PSOConfig struct {
	SwarmSize      int     // Number of particles (s)
	MaxIterations  int     // Maximum PSO iterations
	InertiaWeight  float64 // w (typically 0.9 -> 0.4)
	CognitiveCoeff float64 // c1 (typically 2.0)
	SocialCoeff    float64 // c2 (typically 2.0)

	// Search space bounds
	DeltaScale float64 // α in paper (typically 0.8)

	// Convergence
	MinVelocity float64 // Stop if all velocities below this
}

// DefaultPSOConfig returns recommended settings from paper
func DefaultPSOConfig() PSOConfig {
	return PSOConfig{
		SwarmSize:      500, // As used in simulations
		MaxIterations:  100,
		InertiaWeight:  0.9, // Decays to 0.4
		CognitiveCoeff: 2.0,
		SocialCoeff:    2.0,
		DeltaScale:     0.8, // α = 0.8 from paper
		MinVelocity:    1e-6,
	}
}

// PSOParticle represents a candidate solution in PSO
type PSOParticle struct {
	Position     [3]float64 // Current position
	Velocity     [3]float64 // Current velocity
	PersonalBest [3]float64 // Personal best (pbest)
	PersonalCost float64    // Personal best cost
}

// PSOLocalizer implements PSO-based TDOA-PDOA localization
type PSOLocalizer struct {
	Stations []RFStation
	Config   PSOConfig
}

// NewPSOLocalizer creates PSO-based localizer
func NewPSOLocalizer(stations []RFStation, config PSOConfig) *PSOLocalizer {
	return &PSOLocalizer{
		Stations: stations,
		Config:   config,
	}
}

// LocalizationMode selects which algorithm to use
type LocalizationMode int

const (
	TDOA_PDOA_1 LocalizationMode = iota // PDOA only (Algorithm 1)
	TDOA_PDOA_2                         // Joint cost (Algorithm 2)
)

// PSOResult contains PSO localization output
type PSOResult struct {
	Position      [3]float64
	Cost          float64
	Iterations    int
	Converged     bool
	SwarmBestCost float64
}

// LocalizeWithPSO performs PSO-based localization
// Implements both Algorithm 1 and Algorithm 2 from paper
func (pso *PSOLocalizer) LocalizeWithPSO(
	measurement TDOAPDOAMeasurement,
	mode LocalizationMode,
) PSOResult {

	// Step 1: TDOA initialization (quasi-Newton)
	initialGuess := pso.tdoaQuasiNewton(measurement)

	// Step 2: Define search space bounds (Eq. 12-13)
	bounds := pso.computeSearchBounds(initialGuess, measurement)

	// Step 3: Initialize swarm around initial guess
	swarm := pso.initializeSwarm(bounds, initialGuess)

	// Step 4: PSO optimization
	globalBest := pso.runPSO(swarm, measurement, mode, bounds)

	return globalBest
}

// computeSearchBounds defines search space (Eq. 12-13)
// δ = α · σt · 1
// Slb = max(p0 - δ, plb)
// Sub = min(p0 + δ, pub)
type SearchBounds struct {
	Lower [3]float64 // Slb
	Upper [3]float64 // Sub
}

func (pso *PSOLocalizer) computeSearchBounds(
	initial [3]float64,
	m TDOAPDOAMeasurement,
) SearchBounds {

	// Estimate TDOA standard deviation from covariance
	sigmaT := math.Sqrt(m.TDOACovariance[0][1]) // Use first off-diagonal
	if sigmaT < 1e-12 {
		sigmaT = 1e-9 // Default 1ns
	}

	// δ = α · σt · c (convert time to distance)
	c := 299792.458                                    // km/s
	delta := pso.Config.DeltaScale * sigmaT * c * 1000 // meters -> km

	// Compute anchor bounds
	plb, pub := pso.computeAnchorBounds()

	var bounds SearchBounds
	for i := 0; i < 3; i++ {
		// Slb = max(p0 - δ, plb)
		bounds.Lower[i] = math.Max(initial[i]-delta, plb[i])
		// Sub = min(p0 + δ, pub)
		bounds.Upper[i] = math.Min(initial[i]+delta, pub[i])
	}

	return bounds
}

// computeAnchorBounds finds bounding box of anchors
func (pso *PSOLocalizer) computeAnchorBounds() (plb, pub [3]float64) {
	for i := 0; i < 3; i++ {
		plb[i] = math.Inf(1)
		pub[i] = math.Inf(-1)
	}

	for _, s := range pso.Stations {
		for i := 0; i < 3; i++ {
			if s.Position[i] < plb[i] {
				plb[i] = s.Position[i]
			}
			if s.Position[i] > pub[i] {
				pub[i] = s.Position[i]
			}
		}
	}

	return
}

// initializeSwarm creates particles in search space
func (pso *PSOLocalizer) initializeSwarm(
	bounds SearchBounds,
	initial [3]float64,
) []PSOParticle {

	swarm := make([]PSOParticle, pso.Config.SwarmSize)

	for i := range swarm {
		// Random position in bounds, centered on initial guess
		for dim := 0; dim < 3; dim++ {
			range_ := bounds.Upper[dim] - bounds.Lower[dim]
			swarm[i].Position[dim] = initial[dim] +
				(rand.Float64()-0.5)*range_

			// Clamp to bounds
			if swarm[i].Position[dim] < bounds.Lower[dim] {
				swarm[i].Position[dim] = bounds.Lower[dim]
			}
			if swarm[i].Position[dim] > bounds.Upper[dim] {
				swarm[i].Position[dim] = bounds.Upper[dim]
			}

			// Random velocity (small fraction of range)
			swarm[i].Velocity[dim] = (rand.Float64() - 0.5) * range_ * 0.1
		}

		// Initialize personal best
		swarm[i].PersonalBest = swarm[i].Position
		swarm[i].PersonalCost = math.Inf(1)
	}

	return swarm
}

// runPSO executes particle swarm optimization
func (pso *PSOLocalizer) runPSO(
	swarm []PSOParticle,
	m TDOAPDOAMeasurement,
	mode LocalizationMode,
	bounds SearchBounds,
) PSOResult {

	// Global best
	var globalBestPosition [3]float64
	globalBestCost := math.Inf(1)

	// Precompute weights bm (Eq. 7)
	weights := pso.computeWeights(m)

	// Iterations
	for iter := 0; iter < pso.Config.MaxIterations; iter++ {
		// Decay inertia weight: 0.9 -> 0.4
		w := pso.Config.InertiaWeight -
			(0.9-0.4)*float64(iter)/float64(pso.Config.MaxIterations)

		maxVelocity := 0.0

		// Update each particle
		for i := range swarm {
			// Evaluate cost
			var cost float64
			switch mode {
			case TDOA_PDOA_1:
				cost = pso.pdoaCostOnly(swarm[i].Position, m, weights)
			case TDOA_PDOA_2:
				cost = pso.jointCost(swarm[i].Position, m, weights)
			}

			// Update personal best
			if cost < swarm[i].PersonalCost {
				swarm[i].PersonalCost = cost
				swarm[i].PersonalBest = swarm[i].Position
			}

			// Update global best
			if cost < globalBestCost {
				globalBestCost = cost
				globalBestPosition = swarm[i].Position
			}

			// Update velocity (PSO equation)
			for dim := 0; dim < 3; dim++ {
				r1 := rand.Float64()
				r2 := rand.Float64()

				cognitive := pso.Config.CognitiveCoeff * r1 *
					(swarm[i].PersonalBest[dim] - swarm[i].Position[dim])
				social := pso.Config.SocialCoeff * r2 *
					(globalBestPosition[dim] - swarm[i].Position[dim])

				newVelocity := w*swarm[i].Velocity[dim] + cognitive + social

				// Clamp velocity
				maxVel := (bounds.Upper[dim] - bounds.Lower[dim]) * 0.1
				if newVelocity > maxVel {
					newVelocity = maxVel
				}
				if newVelocity < -maxVel {
					newVelocity = -maxVel
				}

				swarm[i].Velocity[dim] = newVelocity

				// Track max velocity for convergence
				if abs(newVelocity) > maxVelocity {
					maxVelocity = abs(newVelocity)
				}
			}

			// Update position
			for dim := 0; dim < 3; dim++ {
				swarm[i].Position[dim] += swarm[i].Velocity[dim]

				// Enforce bounds
				if swarm[i].Position[dim] < bounds.Lower[dim] {
					swarm[i].Position[dim] = bounds.Lower[dim]
					swarm[i].Velocity[dim] *= -0.5 // Bounce
				}
				if swarm[i].Position[dim] > bounds.Upper[dim] {
					swarm[i].Position[dim] = bounds.Upper[dim]
					swarm[i].Velocity[dim] *= -0.5 // Bounce
				}
			}
		}

		// Check convergence
		if maxVelocity < pso.Config.MinVelocity {
			return PSOResult{
				Position:      globalBestPosition,
				Cost:          globalBestCost,
				Iterations:    iter + 1,
				Converged:     true,
				SwarmBestCost: globalBestCost,
			}
		}
	}

	return PSOResult{
		Position:      globalBestPosition,
		Cost:          globalBestCost,
		Iterations:    pso.Config.MaxIterations,
		Converged:     false,
		SwarmBestCost: globalBestCost,
	}
}

// computeWeights computes bm coefficients (Eq. 7)
// bm = σt² / σp,m² = σt² / (γp,m² · (2πfm)²)
func (pso *PSOLocalizer) computeWeights(m TDOAPDOAMeasurement) []float64 {
	// Estimate TDOA variance
	sigmaT2 := m.TDOACovariance[0][1]
	if sigmaT2 < 1e-20 {
		sigmaT2 = 1e-18 // Default
	}

	// For single frequency (M=1), weight is 1.0
	// For multi-frequency, compute per frequency
	M := 1 // Assuming single frequency for now
	weights := make([]float64, M)

	for mIdx := 0; mIdx < M; mIdx++ {
		// bm = σt² / σp,m²
		// Assuming σp,m² is given by PDOAVariance
		sigmaP2 := m.PDOAVariance
		if sigmaP2 < 1e-20 {
			sigmaP2 = 0.01 // Default 0.1 rad std
		}

		weights[mIdx] = sigmaT2 / sigmaP2
	}

	return weights
}

// pdoaCostOnly computes PDOA-only cost (Eq. 6)
// cp(p) = (1/M) Σ bm Σ {wrap[ψ̂ij,m - ψij,m(p)]}²
func (pso *PSOLocalizer) pdoaCostOnly(x [3]float64, m TDOAPDOAMeasurement, weights []float64) float64 {
	cost := 0.0
	M := len(weights)
	if M == 0 {
		M = 1
	}

	lambda := m.Wavelength / 1000.0 // km
	if lambda < 1e-10 {
		lambda = 0.0003 // Default 1 GHz
	}

	for mIdx := 0; mIdx < M; mIdx++ {
		bm := weights[mIdx]
		innerSum := 0.0

		for i := 0; i < len(pso.Stations); i++ {
			for j := i + 1; j < len(pso.Stations); j++ {
				// Measured wrapped PDOA
				measuredPsi := wrapAngle(m.PDOA[i][j])

				// Predicted PDOA
				di := distance(x, pso.Stations[i].Position)
				dj := distance(x, pso.Stations[j].Position)
				deltaD := di - dj
				predictedPhi := 2.0 * math.Pi / lambda * deltaD
				predictedPsi := wrapAngle(predictedPhi)

				// Wrapped residual
				residual := wrapAngle(measuredPsi - predictedPsi)
				innerSum += residual * residual
			}
		}

		cost += bm * innerSum / float64(M)
	}

	return cost
}

// jointCost computes joint TDOA-PDOA cost (Eq. 9)
// cj(p) = ct(p) + cp(p)
func (pso *PSOLocalizer) jointCost(x [3]float64, m TDOAPDOAMeasurement, weights []float64) float64 {
	// TDOA cost (Eq. 2-3)
	tdoaCost := 0.0
	c := 299792.458 // km/s

	for i := 0; i < len(pso.Stations); i++ {
		for j := i + 1; j < len(pso.Stations); j++ {
			measuredTau := m.TDOA[i][j]

			di := distance(x, pso.Stations[i].Position)
			dj := distance(x, pso.Stations[j].Position)
			predictedTau := (di - dj) / c

			residual := measuredTau - predictedTau
			sigma2 := m.TDOACovariance[i][j]
			if sigma2 < 1e-20 {
				sigma2 = 1e-12
			}

			tdoaCost += residual * residual / sigma2
		}
	}

	// PDOA cost
	pdoaCost := pso.pdoaCostOnly(x, m, weights)

	return tdoaCost + pdoaCost
}

// wrapAngle wraps angle to [-π, π] (Eq. 5)
// wrap(φ) = mod(φ + π, 2π) - π
func wrapAngle(angle float64) float64 {
	return math.Mod(angle+math.Pi, 2*math.Pi) - math.Pi
}

// tdoaQuasiNewton performs TDOA-only optimization for initialization
// Uses gradient descent (simplified quasi-Newton)
func (pso *PSOLocalizer) tdoaQuasiNewton(m TDOAPDOAMeasurement) [3]float64 {
	// Start from centroid of stations
	centroid := [3]float64{}
	for _, s := range pso.Stations {
		centroid[0] += s.Position[0]
		centroid[1] += s.Position[1]
		centroid[2] += s.Position[2]
	}
	n := float64(len(pso.Stations))
	centroid[0] /= n
	centroid[1] /= n
	centroid[2] /= n

	// Simple gradient descent (5 iterations)
	position := centroid
	c := 299792.458

	for iter := 0; iter < 5; iter++ {
		gradient := [3]float64{}

		for i := 0; i < len(pso.Stations); i++ {
			for j := i + 1; j < len(pso.Stations); j++ {
				measuredTau := m.TDOA[i][j]

				di := distance(position, pso.Stations[i].Position)
				dj := distance(position, pso.Stations[j].Position)
				predictedTau := (di - dj) / c

				diff := measuredTau - predictedTau

				// Gradient contribution
				if di > 1e-10 {
					gradient[0] += diff * (position[0] - pso.Stations[i].Position[0]) / di / c
					gradient[1] += diff * (position[1] - pso.Stations[i].Position[1]) / di / c
					gradient[2] += diff * (position[2] - pso.Stations[i].Position[2]) / di / c
				}
				if dj > 1e-10 {
					gradient[0] -= diff * (position[0] - pso.Stations[j].Position[0]) / dj / c
					gradient[1] -= diff * (position[1] - pso.Stations[j].Position[1]) / dj / c
					gradient[2] -= diff * (position[2] - pso.Stations[j].Position[2]) / dj / c
				}
			}
		}

		// Update with small step
		step := 100.0 // km
		position[0] -= step * gradient[0]
		position[1] -= step * gradient[1]
		position[2] -= step * gradient[2]
	}

	return position
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// BenchmarkPSO compares both algorithms with Monte Carlo simulation
// Simulation parameters from Chen WCL2020-0388:
// - 8 anchors in 2d×2d square, d=2.5m
// - Target at [1, 1] m
// - σt = 2mm, σp = 0.32mm (0.1265 rad)
func BenchmarkPSO() map[string]struct {
	RMSE       float64
	Converged  float64
	Iterations float64
} {
	// Simulation parameters
	d := 2.5 // Anchor spacing (m)
	numAnchors := 8
	targetPos := [3]float64{1.0, 1.0, 0.0} // Target position (m)
	sigmaT := 0.002                        // 2mm time uncertainty
	sigmaP := 0.1265                       // Phase uncertainty (rad)
	numMonteCarlo := 100                   // Number of Monte Carlo runs

	// Create 8 anchors in 2d×2d square
	stations := make([]RFStation, numAnchors)
	for i := 0; i < numAnchors; i++ {
		row := i / 4
		col := i % 4
		stations[i] = RFStation{
			ID:       fmt.Sprintf("A%d", i),
			Position: [3]float64{float64(col) * d, float64(row) * d, 0.0},
		}
	}

	// Initialize PSO localizer
	config := DefaultPSOConfig()
	config.SwarmSize = 100 // Reduced for benchmark speed
	config.MaxIterations = 50
	pso := NewPSOLocalizer(stations, config)

	// Results accumulation
	results := make(map[string]struct {
		RMSE       float64
		Converged  float64
		Iterations float64
	})

	modes := []struct {
		name string
		mode LocalizationMode
	}{
		{"TDOA-PDOA-1", TDOA_PDOA_1},
		{"TDOA-PDOA-2", TDOA_PDOA_2},
	}

	for _, modeInfo := range modes {
		var totalSquaredError, totalIterations float64
		convergedCount := 0

		for mc := 0; mc < numMonteCarlo; mc++ {
			// Generate noisy measurements
			tdoa := make([][]float64, numAnchors)
			pdoa := make([][]float64, numAnchors)
			fdoa := make([][]float64, numAnchors)
			tdoaCov := make([][]float64, numAnchors)

			for i := range tdoa {
				tdoa[i] = make([]float64, numAnchors)
				pdoa[i] = make([]float64, numAnchors)
				fdoa[i] = make([]float64, numAnchors)
				tdoaCov[i] = make([]float64, numAnchors)
			}

			c := 299792458.0 // m/s
			lambda := 0.3    // 1 GHz wavelength (m)

			for i := 0; i < numAnchors; i++ {
				for j := i + 1; j < numAnchors; j++ {
					di := distance(targetPos, stations[i].Position)
					dj := distance(targetPos, stations[j].Position)

					// True TDOA and PDOA
					trueTDOA := (di - dj) / c
					truePDOA := 2.0 * math.Pi / lambda * (di - dj)

					// Add noise
					tdoa[i][j] = trueTDOA + rand.NormFloat64()*sigmaT
					tdoa[j][i] = -tdoa[i][j]
					pdoa[i][j] = truePDOA + rand.NormFloat64()*sigmaP
					pdoa[j][i] = -pdoa[i][j]
					fdoa[i][j] = 0.0
					fdoa[j][i] = -fdoa[i][j]

					tdoaCov[i][j] = sigmaT * sigmaT
					tdoaCov[j][i] = sigmaT * sigmaT
				}
			}

			measurement := TDOAPDOAMeasurement{
				TDOA:             tdoa,
				PDOA:             pdoa,
				FDOA:             fdoa,
				TDOACovariance:   tdoaCov,
				PDOAVariance:     sigmaP * sigmaP,
				FDOAVariance:     0.0,
				CarrierFrequency: 1e9,
				Wavelength:       lambda,
			}

			// Run PSO
			result := pso.LocalizeWithPSO(measurement, modeInfo.mode)

			// Compute error
			error := distance(result.Position, targetPos)
			totalSquaredError += error * error
			totalIterations += float64(result.Iterations)

			if result.Converged {
				convergedCount++
			}
		}

		rmse := math.Sqrt(totalSquaredError / float64(numMonteCarlo))
		convergenceRate := float64(convergedCount) / float64(numMonteCarlo) * 100
		avgIterations := totalIterations / float64(numMonteCarlo)

		results[modeInfo.name] = struct {
			RMSE       float64
			Converged  float64
			Iterations float64
		}{
			RMSE:       rmse,
			Converged:  convergenceRate,
			Iterations: avgIterations,
		}
	}

	return results
}
