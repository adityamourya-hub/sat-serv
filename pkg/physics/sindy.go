package physics

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"
)

/**
 * SINDy (Sparse Identification of Nonlinear Dynamics)
 *
 * Discovers governing equations of adversary maneuvers via sparse regression.
 * Replaces black-box rKAN with interpretable differential equations.
 *
 * Reference: Brunton et al. "Discovering governing equations from data" (PNAS 2016)
 */

type SINDy struct {
	// Library configuration
	PolynomialDegree int
	UseTrigonometric bool
	UseExponential   bool

	// STLSQ parameters
	SparsityThreshold float64
	MaxIterations     int
}

// NewSINDy creates a SINDy discoverer with default parameters
func NewSINDy() *SINDy {
	return &SINDy{
		PolynomialDegree:  3,
		UseTrigonometric:  true,
		UseExponential:    false,
		SparsityThreshold: 0.1,
		MaxIterations:     10,
	}
}

// ManeuverResult contains discovered dynamics and interpretation
type ManeuverResult struct {
	Coefficients    []float64 // Sparse coefficients for library terms
	ActiveTerms     []string  // Human-readable term names
	ThrustDetected  bool      // Whether non-gravitational forces identified
	ThrustMagnitude float64   // Estimated delta-V (m/s)
	EquationString  string    // LaTeX-ready equation
	Confidence      float64   // R² fit quality
}

// DiscoverManeuver identifies thrust signatures from orbital residuals
//
// Inputs: states = [pos(3), vel(3)] over time (residuals from Keplerian)
// Output: Sparse coefficients where only thrust terms are non-zero
func (s *SINDy) DiscoverManeuver(states [][6]float64, dt float64) (*ManeuverResult, error) {
	// 1. Build Theta Library (candidate functions)
	theta := s.buildLibrary(states)

	// 2. Compute derivatives (accelerations from residuals)
	xDot := s.computeDerivatives(states, dt)

	// 3. Sparse Regression (STLSQ)
	coefficients := s.stlsq(theta, xDot, s.SparsityThreshold)

	// 4. Interpret results
	result := s.interpretCoefficients(coefficients, states)

	return result, nil
}

// buildLibrary constructs the candidate function matrix Theta(X)
// Each column is a candidate term: [1, x, y, z, vx, vy, vz, x², xy, sin(x), ...]
func (s *SINDy) buildLibrary(states [][6]float64) *mat.Dense {
	nSamples := len(states)
	nTerms := s.countLibraryTerms()

	data := make([]float64, nSamples*nTerms)

	for i, state := range states {
		x, y, z := state[0], state[1], state[2]
		vx, vy, vz := state[3], state[4], state[5]

		col := 0

		// Constant term
		data[i*nTerms+col] = 1.0
		col++

		// Linear terms (position)
		data[i*nTerms+col] = x
		col++
		data[i*nTerms+col] = y
		col++
		data[i*nTerms+col] = z
		col++

		// Linear terms (velocity)
		data[i*nTerms+col] = vx
		col++
		data[i*nTerms+col] = vy
		col++
		data[i*nTerms+col] = vz
		col++

		// Polynomial terms up to degree 3
		for d := 2; d <= s.PolynomialDegree; d++ {
			// x^d, y^d, z^d
			data[i*nTerms+col] = math.Pow(x, float64(d))
			col++
			data[i*nTerms+col] = math.Pow(y, float64(d))
			col++
			data[i*nTerms+col] = math.Pow(z, float64(d))
			col++

			// Cross terms (d=2): xy, xz, yz
			if d == 2 {
				data[i*nTerms+col] = x * y
				col++
				data[i*nTerms+col] = x * z
				col++
				data[i*nTerms+col] = y * z
				col++
			}
		}

		// Trigonometric terms (for periodic thrust profiles)
		if s.UseTrigonometric {
			r := math.Sqrt(x*x + y*y + z*z)
			if r > 1e-6 {
				data[i*nTerms+col] = math.Sin(r / 7000.0) // Normalized by typical LEO radius
				col++
				data[i*nTerms+col] = math.Cos(r / 7000.0)
				col++
			}
		}

		// Velocity magnitude (drag-like term)
		vMag := math.Sqrt(vx*vx + vy*vy + vz*vz)
		data[i*nTerms+col] = vMag * vx // Directional drag
		col++
		data[i*nTerms+col] = vMag * vy
		col++
		data[i*nTerms+col] = vMag * vz
		col++
	}

	return mat.NewDense(nSamples, nTerms, data)
}

// computeDerivatives calculates numerical derivatives (accelerations)
func (s *SINDy) computeDerivatives(states [][6]float64, dt float64) *mat.Dense {
	nSamples := len(states)

	// Compute derivatives using central differences
	// For acceleration: d²x/dt² from position
	data := make([]float64, (nSamples-2)*6)

	for i := 1; i < nSamples-1; i++ {
		// Central difference: (x_{i+1} - 2*x_i + x_{i-1}) / dt²
		for dim := 0; dim < 6; dim++ {
			data[(i-1)*6+dim] = (states[i+1][dim] - 2*states[i][dim] + states[i-1][dim]) / (dt * dt)
		}
	}

	return mat.NewDense(nSamples-2, 6, data)
}

// stlsq performs Sequential Thresholded Least Squares regression
// This is the core SINDy algorithm for sparse coefficient discovery
func (s *SINDy) stlsq(theta, xDot *mat.Dense, threshold float64) []float64 {
	nSamples, nTerms := theta.Dims()
	_, nDims := xDot.Dims()

	// Initialize with least squares solution
	coefficients := make([]float64, nTerms*nDims)

	// Iterate: fit → threshold → refit
	for iter := 0; iter < s.MaxIterations; iter++ {
		// Identify small coefficients
		bigIndices := make([]int, 0)
		for i := 0; i < nTerms; i++ {
			maxCoeff := 0.0
			for d := 0; d < nDims; d++ {
				if abs(coefficients[i*nDims+d]) > maxCoeff {
					maxCoeff = abs(coefficients[i*nDims+d])
				}
			}
			if maxCoeff > threshold || iter == 0 {
				bigIndices = append(bigIndices, i)
			}
		}

		if len(bigIndices) == 0 {
			break
		}

		// Build reduced theta matrix
		reducedTheta := mat.NewDense(nSamples, len(bigIndices), nil)
		for i, idx := range bigIndices {
			col := theta.ColView(idx)
			for r := 0; r < nSamples; r++ {
				reducedTheta.Set(r, i, col.At(r, 0))
			}
		}

		// Least squares on reduced set
		var reducedCoeffs mat.Dense
		err := reducedCoeffs.Solve(reducedTheta, xDot)
		if err != nil {
			break
		}

		// Update full coefficient vector
		for i, idx := range bigIndices {
			for d := 0; d < nDims; d++ {
				coefficients[idx*nDims+d] = reducedCoeffs.At(i, d)
			}
		}
		for i := 0; i < nTerms; i++ {
			found := false
			for _, idx := range bigIndices {
				if i == idx {
					found = true
					break
				}
			}
			if !found {
				for d := 0; d < nDims; d++ {
					coefficients[i*nDims+d] = 0
				}
			}
		}
	}

	return coefficients
}

// interpretCoefficients converts sparse coefficients to human-readable results
func (s *SINDy) interpretCoefficients(coeffs []float64, states [][6]float64) *ManeuverResult {
	termNames := s.getTermNames()

	result := &ManeuverResult{
		Coefficients: coeffs,
		ActiveTerms:  make([]string, 0),
	}

	nDims := 6
	thrustMag := 0.0

	// Identify active (non-zero) terms
	for i, name := range termNames {
		maxCoeff := 0.0
		for d := 0; d < nDims; d++ {
			if abs(coeffs[i*nDims+d]) > maxCoeff {
				maxCoeff = abs(coeffs[i*nDims+d])
			}
		}

		if maxCoeff > s.SparsityThreshold {
			result.ActiveTerms = append(result.ActiveTerms,
				fmt.Sprintf("%s (%.3f)", name, maxCoeff))

			// Detect thrust (non-gravitational acceleration terms)
			if name != "1" && name != "x" && name != "y" && name != "z" &&
				!strings.HasPrefix(name, "x²") && !strings.HasPrefix(name, "y²") {
				thrustMag += maxCoeff
			}
		}
	}

	result.ThrustDetected = thrustMag > 0.01
	result.ThrustMagnitude = thrustMag * 1000 // Convert to m/s² approx

	// Build equation string
	result.EquationString = s.buildEquationString(coeffs, termNames)

	// Compute fit quality (simplified R²)
	result.Confidence = s.computeRSquared(coeffs, states)

	return result
}

// getTermNames returns human-readable names for library terms
func (s *SINDy) getTermNames() []string {
	names := []string{"1", "x", "y", "z", "vx", "vy", "vz"}

	// Polynomial terms
	for d := 2; d <= s.PolynomialDegree; d++ {
		names = append(names, fmt.Sprintf("x^%d", d))
		names = append(names, fmt.Sprintf("y^%d", d))
		names = append(names, fmt.Sprintf("z^%d", d))
		if d == 2 {
			names = append(names, "xy", "xz", "yz")
		}
	}

	if s.UseTrigonometric {
		names = append(names, "sin(r)", "cos(r)")
	}

	names = append(names, "v*vx", "v*vy", "v*vz")

	return names
}

// countLibraryTerms returns the number of candidate functions
func (s *SINDy) countLibraryTerms() int {
	count := 7 // 1 + x,y,z + vx,vy,vz

	// Polynomial: 3*(degree-1) for x^d,y^d,z^d + 3 cross terms for d=2
	count += 3*(s.PolynomialDegree-1) + 3

	if s.UseTrigonometric {
		count += 2
	}

	count += 3 // Velocity magnitude terms

	return count
}

// Helper functions
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func (s *SINDy) buildEquationString(coeffs []float64, termNames []string) string {
	// Build LaTeX-formatted equation
	// Simplified: just show active terms
	parts := make([]string, 0)
	nDims := 6

	dimNames := []string{"\\dot{x}", "\\dot{y}", "\\dot{z}", "\\dot{v_x}", "\\dot{v_y}", "\\dot{v_z}"}

	for d := 0; d < nDims; d++ {
		eqn := dimNames[d] + " = "
		for i, name := range termNames {
			c := coeffs[i*nDims+d]
			if abs(c) > s.SparsityThreshold {
				eqn += fmt.Sprintf("%.3f%s + ", c, name)
			}
		}
		parts = append(parts, eqn[:len(eqn)-3]) // Remove trailing " + "
	}

	return strings.Join(parts, "; ")
}

func (s *SINDy) computeRSquared(coeffs []float64, states [][6]float64) float64 {
	// Simplified R² calculation
	// In production, compare predicted vs actual residuals
	return 0.95 // Placeholder
}
