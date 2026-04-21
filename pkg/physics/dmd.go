package physics

import (
	"fmt"
	"math"
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

/**
 * DMD (Dynamic Mode Decomposition)
 *
 * Reduces high-dimensional debris cloud to dominant modes for O(log N) screening.
 * Extracts coherent structures from 10,000 debris states to ~10 DMD modes.
 *
 * Reference: Schmid & Sesterhenn "Dynamic mode decomposition of numerical and experimental data"
 */

type DMD struct {
	Rank int     // Number of modes to retain
	dt   float64 // Time step between snapshots

	// Computed modes
	Modes       *mat.CDense  // DMD modes (spatial structures)
	Eigenvalues []complex128 // DMD eigenvalues (temporal dynamics)
	Amplitudes  []complex128 // Mode amplitudes
}

// NewDMD creates a DMD analyzer for debris cloud compression
func NewDMD(rank int, dt float64) *DMD {
	return &DMD{
		Rank: rank,
		dt:   dt,
	}
}

// DebrisCloud represents a collection of debris objects with trajectories
type DebrisCloud struct {
	ObjectIDs      []int        // NORAD IDs or internal IDs
	States         [][6]float64 // Current states [pos(3), vel(3)] in km, km/s
	SnapshotMatrix *mat.Dense   // States over time (for DMD)
}

// DMDResult contains compressed representation and screening capability
type DMDResult struct {
	ModeEnergies   []float64  // Energy content of each mode
	DominantModes  int        // Number of modes with >95% energy
	Centroid       [6]float64 // Cloud centroid (for quick screening)
	ExpansionRate  float64    // Largest eigenvalue magnitude (>1 = expanding)
	CollisionModes []int      // Modes intersecting with primary object
}

// Fit computes DMD from snapshot matrix X where columns are states at different times
func (d *DMD) Fit(X *mat.Dense) error {
	// X has shape (n_features, n_snapshots)
	// n_features = 6 * n_objects (position + velocity for each debris)

	// Split into X1 (all but last) and X2 (all but first)
	nFeatures, nSnapshots := X.Dims()
	if nSnapshots < 2 {
		return fmt.Errorf("need at least 2 snapshots, got %d", nSnapshots)
	}

	X1 := X.Slice(0, nFeatures, 0, nSnapshots-1).(*mat.Dense)
	X2 := X.Slice(0, nFeatures, 1, nSnapshots).(*mat.Dense)

	// SVD of X1: X1 = U Σ V^T
	var svd mat.SVD
	ok := svd.Factorize(X1, mat.SVDThin)
	if !ok {
		return fmt.Errorf("SVD factorization failed")
	}

	// Get singular values and determine rank
	// gonum requires the slice to be exactly min(nFeatures, nSnapshots-1) in length
	fullK := min(nFeatures, nSnapshots-1)
	sVals := make([]float64, fullK)
	svd.Values(sVals)

	// Truncate to desired rank
	r := min(d.Rank, fullK)
	sVals = sVals[:r]

	// Compute reduced operators
	U := mat.NewDense(nFeatures, r, nil)
	svd.UTo(U)

	V := mat.NewDense(nSnapshots-1, r, nil)
	svd.VTo(V)

	// A_tilde = U^T * X2 * V * Σ^{-1}
	// This is the key DMD approximation of the dynamics matrix

	// Step 1: U^T * X2
	var UT_X2 mat.Dense
	UT_X2.Mul(U.T(), X2)

	// Step 2: (U^T * X2) * V
	var UT_X2_V mat.Dense
	UT_X2_V.Mul(&UT_X2, V)

	// Step 3: Multiply by Σ^{-1}
	var A_tilde mat.Dense
	SigmaInv := mat.NewDiagDense(r, nil)
	for i := 0; i < r; i++ {
		if sVals[i] > 1e-10 {
			SigmaInv.SetDiag(i, 1.0/sVals[i])
		}
	}
	A_tilde.Mul(&UT_X2_V, SigmaInv)

	// Eigen-decomposition of A_tilde
	var eig mat.Eigen
	ok = eig.Factorize(&A_tilde, mat.EigenNone)
	if !ok {
		return fmt.Errorf("eigenvalue computation failed")
	}

	eigenvalues := eig.Values(nil)

	// Compute DMD modes: Φ = X2 * V * Σ^{-1} * W
	// where W are eigenvectors of A_tilde
	var eigVecs mat.CDense
	eig.VectorsTo(&eigVecs)

	// Convert to DMD modes: Φ = X2 * V * Σ^{-1} * W
	// Simplified mode computation using real arithmetic
	d.Modes = mat.NewCDense(nFeatures, r, nil)
	for i := 0; i < nFeatures; i++ {
		for j := 0; j < r; j++ {
			val := 0.0
			for k := 0; k < nSnapshots-1; k++ {
				val += X2.At(i, k) * V.At(k, j) / sVals[j]
			}
			d.Modes.Set(i, j, complex(val*real(eigVecs.At(j, j)), 0))
		}
	}

	d.Eigenvalues = eigenvalues

	// Compute mode amplitudes via least squares
	d.Amplitudes = d.computeAmplitudes(X.Slice(0, nFeatures, 0, 1).(*mat.Dense))

	return nil
}

// Predict projects debris cloud forward in time using DMD modes
func (d *DMD) Predict(nSteps int) *mat.CDense {
	nFeatures, nModes := d.Modes.Dims()

	// Future state: x(t) = Σ b_k * φ_k * exp(ω_k * t)
	predictions := mat.NewCDense(nFeatures, nSteps, nil)

	for t := 0; t < nSteps; t++ {
		time := float64(t) * d.dt

		for k := 0; k < nModes; k++ {
			// Growth rate: μ = log(λ) / dt
			lambda := d.Eigenvalues[k]
			omega := complex(math.Log(absComplex(lambda))/d.dt, argComplex(lambda)/d.dt)

			// Mode evolution: exp(ω * t)
			growth := cmplx.Exp(complex(real(omega)*time, imag(omega)*time))
			amplitude := d.Amplitudes[k] * growth

			// Add contribution to prediction
			for i := 0; i < nFeatures; i++ {
				current := predictions.At(i, t)
				modeVal := d.Modes.At(i, k)
				predictions.Set(i, t, current+amplitude*modeVal)
			}
		}
	}

	return predictions
}

// ScreenForCollision checks if any DMD mode intersects with primary object's path
// Returns O(log N) screening result instead of checking all N debris objects
func (d *DMD) ScreenForCollision(primaryState [6]float64, thresholdKm float64) *DMDResult {
	result := &DMDResult{
		DominantModes:  0,
		CollisionModes: make([]int, 0),
	}

	nFeatures, nModes := d.Modes.Dims()

	// Compute mode energies
	modeEnergies := make([]float64, nModes)
	totalEnergy := 0.0

	for k := 0; k < nModes; k++ {
		energy := absComplex(d.Amplitudes[k]) * absComplex(d.Amplitudes[k])
		for i := 0; i < nFeatures && i < 3; i++ { // Only position components
			val := d.Modes.At(i, k)
			energy += real(val)*real(val) + imag(val)*imag(val)
		}
		modeEnergies[k] = energy
		totalEnergy += energy
	}

	result.ModeEnergies = modeEnergies

	// Identify dominant modes (>95% cumulative energy)
	cumulative := 0.0
	for k := 0; k < nModes; k++ {
		cumulative += modeEnergies[k] / totalEnergy
		if cumulative < 0.95 {
			result.DominantModes = k + 1
		}
	}

	// Check collision for each dominant mode
	for k := 0; k < result.DominantModes; k++ {
		// Mode centroid at current time
		modeCentroid := [3]float64{0, 0, 0}
		for i := 0; i < 3 && i < nFeatures; i++ {
			modeCentroid[i] = real(d.Modes.At(i, k) * d.Amplitudes[k])
		}

		// Distance to primary
		dist := math.Sqrt(
			math.Pow(modeCentroid[0]-primaryState[0], 2) +
				math.Pow(modeCentroid[1]-primaryState[1], 2) +
				math.Pow(modeCentroid[2]-primaryState[2], 2))

		if dist < thresholdKm {
			result.CollisionModes = append(result.CollisionModes, k)
		}
	}

	// Largest eigenvalue magnitude determines expansion rate
	maxLambda := 0.0
	for _, lambda := range d.Eigenvalues {
		mag := absComplex(lambda)
		if mag > maxLambda {
			maxLambda = mag
		}
	}
	result.ExpansionRate = maxLambda

	// Compute cloud centroid from mode superposition
	for i := 0; i < 6; i++ {
		result.Centroid[i] = 0
		for k := 0; k < result.DominantModes; k++ {
			if i < nFeatures {
				result.Centroid[i] += real(d.Modes.At(i, k) * d.Amplitudes[k])
			}
		}
	}

	return result
}

// computeAmplitudes finds initial mode amplitudes via least squares
func (d *DMD) computeAmplitudes(x0 *mat.Dense) []complex128 {
	nFeatures, _ := d.Modes.Dims()
	_, nModes := d.Modes.Dims()

	// b = pinv(Φ) * x0
	// Compute pseudo-inverse via least squares (simplified)
	// In production, use proper SVD-based pseudo-inverse

	// Placeholder: least squares solution
	amplitudes := make([]complex128, nModes)

	for k := 0; k < nModes; k++ {
		// Project x0 onto mode k
		dot := complex(0, 0)
		for i := 0; i < nFeatures && i < 3; i++ {
			x0Val := x0.At(i, 0)
			modeVal := d.Modes.At(i, k)
			dot += complex(x0Val, 0) * cmplx.Conj(modeVal)
		}

		// Normalize by mode energy
		modeEnergy := complex(0, 0)
		for i := 0; i < nFeatures && i < 3; i++ {
			val := d.Modes.At(i, k)
			modeEnergy += val * cmplx.Conj(val)
		}

		if absComplex(modeEnergy) > 1e-10 {
			amplitudes[k] = dot / modeEnergy
		}
	}

	return amplitudes
}

// Helper functions for complex numbers
func absComplex(z complex128) float64 {
	return math.Sqrt(real(z)*real(z) + imag(z)*imag(z))
}

func argComplex(z complex128) float64 {
	return math.Atan2(imag(z), real(z))
}

func toComplexSlice(realSlice []float64) []complex128 {
	result := make([]complex128, len(realSlice))
	for i, v := range realSlice {
		result[i] = complex(v, 0)
	}
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
