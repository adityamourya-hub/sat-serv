package orbital

import (
	"math"

	"ssis/pkg/physics"
)

/**
 * GaussIOD implements the Gauss method for Initial Orbit Determination (IOD)
 * from three angular observations (RA, Dec, Time).
 *
 * This is required for Req (g) "Un-related tracks correlation" - correlating
 * uncorrelated tracks (UCTs) from optical or radar sensors without prior catalog IDs.
 *
 * Reference: Vallado, "Fundamentals of Astrodynamics and Applications", Section 7.6
 *
 * Given three angular observations (α₁, δ₁, t₁), (α₂, δ₂, t₂), (α₃, δ₃, t₃),
 * the Gauss method computes the preliminary orbit (position and velocity at t₂).
 *
 * The fundamental equation is an 8th-degree polynomial in r₂:
 *   r₂⁸ - a·r₂⁶ - b·r₂³ - c = 0
 *
 * Where a, b, c are functions of the observations and time intervals.
 */

// GaussIODResult holds the output of the Gauss IOD computation
type GaussIODResult struct {
	State       physics.State // Computed state at the middle observation time
	SlantRanges [3]float64    // Estimated slant ranges to the object (km)
	Converged   bool          // Whether the iteration converged
	Iterations  int           // Number of iterations performed
	Residual    float64       // Final residual (km)
}

// AngularObservation represents a single optical/radar angular measurement
type AngularObservation struct {
	Time  float64    // Julian Date
	Alpha float64    // Right Ascension (radians)
	Delta float64    // Declination (radians)
	Site  [3]float64 // Observer ECI position (km)
}

/**
 * GaussInitialOrbitDetermination computes a preliminary orbit from three angular observations.
 *
 * @param obs1 First observation (earliest time)
 * @param obs2 Second observation (middle time)
 * @param obs3 Third observation (latest time)
 * @return GaussIODResult with state at the middle observation time
 */
func GaussInitialOrbitDetermination(obs1, obs2, obs3 AngularObservation) GaussIODResult {
	// Time intervals (seconds)
	tau1 := (obs1.Time - obs2.Time) * 86400.0 // Negative
	tau3 := (obs3.Time - obs2.Time) * 86400.0 // Positive
	_ = tau3 - tau1

	// Line-of-sight unit vectors
	L1 := [3]float64{
		math.Cos(obs1.Delta) * math.Cos(obs1.Alpha),
		math.Cos(obs1.Delta) * math.Sin(obs1.Alpha),
		math.Sin(obs1.Delta),
	}
	L2 := [3]float64{
		math.Cos(obs2.Delta) * math.Cos(obs2.Alpha),
		math.Cos(obs2.Delta) * math.Sin(obs2.Alpha),
		math.Sin(obs2.Delta),
	}
	L3 := [3]float64{
		math.Cos(obs3.Delta) * math.Cos(obs3.Alpha),
		math.Cos(obs3.Delta) * math.Sin(obs3.Alpha),
		math.Sin(obs3.Delta),
	}

	// Cross products for Gauss method
	L1crossL2 := iodCross(L1, L2)
	L2crossL3 := iodCross(L2, L3)

	// D matrix components (only need D12, D21, D22, D23, D32 for the computation)
	D12 := iodDot(L1crossL2, obs2.Site)
	D21 := iodDot(L2crossL3, obs1.Site)
	D22 := iodDot(L2crossL3, obs2.Site)
	D23 := iodDot(L2crossL3, obs3.Site)
	D32 := iodDot(iodCross(L3, L1), obs2.Site)

	// D matrix determinant
	D0 := iodDot(L1, L2crossL3)

	if math.Abs(D0) < 1e-12 {
		return GaussIODResult{Converged: false}
	}

	// Initial guess for slant ranges (rho1, rho2, rho3)
	// Assume the object is far (e.g., ~20,000 km for MEO)
	rho2Guess := 20000.0 // km

	// Iterate to find correct slant ranges
	maxIter := 50
	tol := 1e-6 // km

	var rho1, rho2, rho3 float64
	var r2Mag float64
	var converged bool
	var finalResidual float64

	for i := 0; i < maxIter; i++ {
		// Gauss f and g series coefficients (low-order approximation)
		// f ≈ 1 - μ·τ²/(2·r³), g ≈ τ - μ·τ³/(6·r³)
		muOverR3 := physics.EARTH_MU_IAU / (rho2Guess * rho2Guess * rho2Guess)

		f1 := 1.0 - 0.5*muOverR3*tau1*tau1
		f3 := 1.0 - 0.5*muOverR3*tau3*tau3

		g1 := tau1 - muOverR3*tau1*tau1*tau1/6.0
		g3 := tau3 - muOverR3*tau3*tau3*tau3/6.0

		// Compute slant ranges from Gauss equations
		// ρ₂ = (-D₁₂ + f₁·D₂₂/f₃ - g₁·D₃₂/g₃) / D₀
		// ρ₁ = (6·D₂₁ + g₃·ρ₂·D₀/f₁) / (6·f₁)
		// ρ₃ = (6·D₂₃ + g₁·ρ₂·D₀/f₃) / (6·f₃)

		rho2 = (-D12 + f1*D22/f3 - g1*D32/g3) / D0
		rho1 = (6.0*D21 + g3*rho2*D0/f1) / (6.0 * f1)
		rho3 = (6.0*D23 + g1*rho2*D0/f3) / (6.0 * f3)

		if rho2 < 100.0 || rho2 > 500000.0 {
			// Unphysical solution - likely bad geometry
			converged = false
			break
		}

		// Compute position vectors
		r1 := [3]float64{
			obs1.Site[0] + rho1*L1[0],
			obs1.Site[1] + rho1*L1[1],
			obs1.Site[2] + rho1*L1[2],
		}
		r2 := [3]float64{
			obs2.Site[0] + rho2*L2[0],
			obs2.Site[1] + rho2*L2[1],
			obs2.Site[2] + rho2*L2[2],
		}

		// Update r2 magnitude for next iteration
		r2Mag = math.Sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2])
		residual := math.Abs(r2Mag - rho2Guess)

		if residual < tol {
			converged = true
			finalResidual = residual

			// Compute velocity at t2 using Gibbs/Herrick-Gibbs
			v2 := gibbsVelocity(r1, r2, tau1, tau3)

			return GaussIODResult{
				State: physics.State{
					Pos:     [3]float64{r2[0], r2[1], r2[2]},
					Vel:     [3]float64{v2[0], v2[1], v2[2]},
					EpochJD: obs2.Time,
				},
				SlantRanges: [3]float64{rho1, rho2, rho3},
				Converged:   true,
				Iterations:  i + 1,
				Residual:    residual,
			}
		}

		// Update guess for next iteration
		rho2Guess = r2Mag
	}

	return GaussIODResult{
		Converged:  converged,
		Iterations: maxIter,
		Residual:   finalResidual,
	}
}

func gibbsVelocity(r1, r2 [3]float64, tau1, tau3 float64) [3]float64 {
	dtau1 := tau1
	dtau3 := tau3
	dtau32 := dtau3*dtau3 - dtau1*dtau1

	r2Mag := math.Sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2])

	c1 := -dtau3 / (dtau1 * dtau32)
	c2 := (dtau3 + dtau1) / (dtau1 * dtau3)

	mu := physics.EARTH_MU_IAU
	corr := mu / (12.0 * r2Mag * r2Mag * r2Mag)

	return [3]float64{
		c1*r1[0] + c2*r2[0] + corr*r2[0],
		c1*r1[1] + c2*r2[1] + corr*r2[1],
		c1*r1[2] + c2*r2[2] + corr*r2[2],
	}
}

// iodDot computes dot product for IOD computations
func iodDot(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

// iodCross computes cross product for IOD computations
func iodCross(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}
