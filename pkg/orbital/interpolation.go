package orbital

import (
	"fmt"
	"math"
)

/**
 * DataPoint represents a single state measurement in a specific reference frame.
 * It is used primarily by the interpolation engine to reconstruct high-fidelity trajectories
 * between discrete ephemeris points (e.g., from CCSDS OEM or SP-3 files).
 * Includes the relative time from a baseline epoch and the 6-DoF state vector.
 * Crucial for avoiding the numerical overhead of integration in real-time screening.
 */
type DataPoint struct {
	Time  float64 // Relative time in seconds
	State State   // Position (km) and Velocity (km/s)
}

/**
 * LagrangeInterpolate performs an Nth-degree Lagrange interpolation on orbital states.
 * It takes a slice of neighbor DataPoints and a target relative time, returning the estimated State.
 * This method is the CCSDS standard for high-accuracy ephemeris reconstruction (typically degree 7).
 * It calculates the Lagrange basis polynomials independently for each component of (X, Y, Z, VX, VY, VZ).
 * Ensures that the resulting state maintains the fidelity of the original high-definition source.
 */
func LagrangeInterpolate(points []DataPoint, targetTime float64) State {
	fmt.Printf("[TRC] Entering LagrangeInterpolate for targetTime: %.3f with %d points\n", targetTime, len(points))
	defer fmt.Printf("[TRC] Exiting LagrangeInterpolate\n")

	n := len(points)
	if n == 0 {
		return State{}
	}
	if n == 1 {
		return points[0].State
	}

	var res State
	// Interpolate Position X, Y, Z
	res.X = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.X })
	res.Y = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.Y })
	res.Z = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.Z })

	// Interpolate Velocity VX, VY, VZ
	res.VX = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.VX })
	res.VY = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.VY })
	res.VZ = interpolateComponent(points, targetTime, func(dp DataPoint) float64 { return dp.State.VZ })

	return res
}

/**
 * interpolateComponent is a helper that computes a single Lagrange polynomial for a specific component.
 * It takes a getter function to extract the desired value (e.g., X or VX) from the DataPoints.
 * The core logic implements the barycentric-equivalent sum of basis products: Σ y_j * L_j(t).
 * Numerical stability is maintained by avoiding explicit construction of the full polynomial form.
 * Internal precision is kept at float64 to minimize rounding drift during the product accumulation.
 */
func interpolateComponent(points []DataPoint, targetTime float64, getter func(DataPoint) float64) float64 {
	n := len(points)
	sum := 0.0

	for j := 0; j < n; j++ {
		basis := 1.0
		for i := 0; i < n; i++ {
			if i == j {
				continue
			}
			// L_j(t) = Π (t - t_i) / (t_j - t_i)
			denominator := points[j].Time - points[i].Time
			if math.Abs(denominator) < 1e-12 {
				continue // Guard against duplicate epochs
			}
			basis *= (targetTime - points[i].Time) / denominator
		}
		sum += getter(points[j]) * basis
	}

	return sum
}
