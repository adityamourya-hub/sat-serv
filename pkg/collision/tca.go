package collision

import (
	"math"

	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

/**
 * TCAFinder implements iterative minimization of the relative distance between two objects.
 * Uses Newton-Raphson with bisection fallback for robust convergence.
 *
 * The TCA (Time of Closest Approach) is found by minimizing:
 *   f(t) = |r₁(t) - r₂(t)|²
 *
 * The derivative is:
 *   f'(t) = 2 * (r₁ - r₂) · (v₁ - v₂)
 *
 * Newton-Raphson iteration:
 *   t_{n+1} = t_n - f'(t_n) / f''(t_n)
 *
 * Where f''(t) ≈ 2 * |v₁ - v₂|² (dominant term at close approach)
 */

// TCAResult holds the computed Time of Closest Approach and associated parameters
type TCAResult struct {
	TCA              float64   // Julian Date of closest approach
	MissDistance     float64   // km, scalar miss distance at TCA
	RelPosition      [3]float64 // km, relative position at TCA (secondary - primary)
	RelVelocity      [3]float64 // km/s, relative velocity at TCA
	Converged        bool      // Whether the iteration converged
	Iterations       int       // Number of iterations performed
	Method           string    // "newton_raphson" or "bisection"
}

// TCAFinderConfig controls the iterative search behavior
type TCAFinderConfig struct {
	MaxIterations    int
	ToleranceKm      float64   // Stop when miss distance changes less than this
	MaxStepSec       float64   // Maximum time step per iteration
	TimeWindowSec    float64   // Search window around initial guess (+/-)
}

// DefaultTCAFinderConfig returns standard configuration for orbital conjunctions
func DefaultTCAFinderConfig() TCAFinderConfig {
	return TCAFinderConfig{
		MaxIterations: 50,
		ToleranceKm:   1e-6,     // 1 mm precision
		MaxStepSec:    3600.0,   // 1 hour max step
		TimeWindowSec: 86400.0,  // ±1 day search window
	}
}

/**
 * FindTCA computes the exact Time of Closest Approach between two objects.
 *
 * @param primary Initial state of primary object at t0
 * @param secondary Initial state of secondary object at t0
 * @param t0 Initial time (Julian Date)
 * @param cfg Search configuration
 * @param params Propagation parameters
 * @return TCAResult with TCA, miss distance, and convergence status
 */
func FindTCA(primary, secondary physics.State, t0 float64, cfg TCAFinderConfig, params physics.PhysicsParams) TCAResult {
	if cfg.MaxIterations <= 0 {
		cfg = DefaultTCAFinderConfig()
	}

	// Initial relative state
	rRel := [3]float64{
		secondary.Pos[0] - primary.Pos[0],
		secondary.Pos[1] - primary.Pos[1],
		secondary.Pos[2] - primary.Pos[2],
	}
	vRel := [3]float64{
		secondary.Vel[0] - primary.Vel[0],
		secondary.Vel[1] - primary.Vel[1],
		secondary.Vel[2] - primary.Vel[2],
	}

	// Initial guess: use projected closest approach based on current geometry
	// For close approaches, this is often very accurate
	rDotV := rRel[0]*vRel[0] + rRel[1]*vRel[1] + rRel[2]*vRel[2]
	vRelMag2 := vRel[0]*vRel[0] + vRel[1]*vRel[1] + vRel[2]*vRel[2]

	var tCurrent float64
	if vRelMag2 > 1e-12 {
		// Linear projection: dr/dt = v_rel, so closest approach when r·v = 0
		tCurrent = -rDotV / vRelMag2
	} else {
		tCurrent = 0
	}

	// Clamp initial guess to search window
	if tCurrent < -cfg.TimeWindowSec {
		tCurrent = -cfg.TimeWindowSec
	}
	if tCurrent > cfg.TimeWindowSec {
		tCurrent = cfg.TimeWindowSec
	}

	// Try Newton-Raphson first
	result := newtonRaphsonTCA(primary, secondary, t0, tCurrent, cfg, params)

	// If Newton-Raphson failed, fall back to bisection
	if !result.Converged {
		result = bisectionTCA(primary, secondary, t0, cfg, params)
	}

	return result
}

/**
 * newtonRaphsonTCA implements Newton-Raphson iteration for TCA finding.
 * Uses derivative of |r_rel|² for rapid convergence near the minimum.
 */
func newtonRaphsonTCA(primary, secondary physics.State, t0, tGuess float64, cfg TCAFinderConfig, params physics.PhysicsParams) TCAResult {
	tCurrent := tGuess
	var prevMissDist float64 = math.Inf(1)

	for i := 0; i < cfg.MaxIterations; i++ {
		// Propagate both objects to current time estimate
		pState := propagateState(primary, t0, t0+tCurrent/86400.0, params)
		sState := propagateState(secondary, t0, t0+tCurrent/86400.0, params)

		// Compute relative state
		rRel, vRel := relativeState(pState, sState)

		// Current miss distance
		missDist := math.Sqrt(rRel[0]*rRel[0] + rRel[1]*rRel[1] + rRel[2]*rRel[2])

		// Check convergence
		if math.Abs(missDist-prevMissDist) < cfg.ToleranceKm && i > 0 {
			return TCAResult{
				TCA:          t0 + tCurrent/86400.0,
				MissDistance: missDist,
				RelPosition:  rRel,
				RelVelocity:  vRel,
				Converged:    true,
				Iterations:   i + 1,
				Method:       "newton_raphson",
			}
		}
		prevMissDist = missDist

		// Newton-Raphson step: minimize f(t) = |r_rel|²
		// f'(t) = 2 * r_rel · v_rel (scaled by time units)
		// f''(t) ≈ 2 * |v_rel|² (dominant term, in km²/s²)
		rDotV := rRel[0]*vRel[0] + rRel[1]*vRel[1] + rRel[2]*vRel[2]
		vRelMag2 := vRel[0]*vRel[0] + vRel[1]*vRel[1] + vRel[2]*vRel[2]

		if vRelMag2 < 1e-12 {
			// Near-zero relative velocity - Newton-Raphson will fail
			// Fall back to bisection
			break
		}

		// Time step: dt = -f'/(f'' * 86400) to convert to seconds, then to days
		// f' = 2*rDotV (km²/s), f'' = 2*vRelMag2 (km²/s²)
		dt := -rDotV / vRelMag2 // seconds

		// Limit step size
		if dt > cfg.MaxStepSec {
			dt = cfg.MaxStepSec
		} else if dt < -cfg.MaxStepSec {
			dt = -cfg.MaxStepSec
		}

		tCurrent += dt

		// Stay within search window
		if tCurrent < -cfg.TimeWindowSec || tCurrent > cfg.TimeWindowSec {
			break
		}
	}

	// Did not converge
	return TCAResult{
		TCA:        t0 + tCurrent/86400.0,
		Converged:  false,
		Iterations: cfg.MaxIterations,
		Method:     "newton_raphson_failed",
	}
}

/**
 * bisectionTCA implements robust bisection search for TCA.
 * Slower than Newton-Raphson but guaranteed to converge within the search window.
 */
func bisectionTCA(primary, secondary physics.State, t0 float64, cfg TCAFinderConfig, params physics.PhysicsParams) TCAResult {
	// Search window: bracket the minimum
	tLeft := -cfg.TimeWindowSec
	tRight := cfg.TimeWindowSec

	// Evaluate at bracket points to find sign change in derivative
	// We want where dr/dt · v_rel = 0 (the minimum)
	fLeft := evaluateTCAFunction(primary, secondary, t0, tLeft, params)
	fRight := evaluateTCAFunction(primary, secondary, t0, tRight, params)

	// Ensure we bracket a minimum (f' changes sign from negative to positive)
	// If not, expand search
	if fLeft > 0 && fRight > 0 {
		// Both positive - minimum might be outside, but we'll find local min
		// Use the side with smaller |f|
		if math.Abs(fLeft) < math.Abs(fRight) {
			tRight = tLeft
		}
	}

	var tMid float64
	for i := 0; i < cfg.MaxIterations; i++ {
		tMid = (tLeft + tRight) / 2.0
		fMid := evaluateTCAFunction(primary, secondary, t0, tMid, params)

		// Narrow bracket based on derivative sign
		// f = 2*r·v. If f < 0, we're approaching; if f > 0, we're receding
		if fMid < 0 {
			// Approaching - move left bracket right
			tLeft = tMid
			fLeft = fMid
		} else {
			// Receding or at minimum - move right bracket left
			tRight = tMid
			fRight = fMid
		}

		// Check convergence
		if math.Abs(tRight-tLeft) < cfg.ToleranceKm*86400.0 {
			// Converged in time - now compute final state
			pState := propagateState(primary, t0, t0+tMid/86400.0, params)
			sState := propagateState(secondary, t0, t0+tMid/86400.0, params)
			rRel, vRel := relativeState(pState, sState)
			missDist := math.Sqrt(rRel[0]*rRel[0] + rRel[1]*rRel[1] + rRel[2]*rRel[2])

			return TCAResult{
				TCA:          t0 + tMid/86400.0,
				MissDistance: missDist,
				RelPosition:  rRel,
				RelVelocity:  vRel,
				Converged:    true,
				Iterations:   i + 1,
				Method:       "bisection",
			}
		}
	}

	// Did not fully converge - return best estimate
	pState := propagateState(primary, t0, t0+tMid/86400.0, params)
	sState := propagateState(secondary, t0, t0+tMid/86400.0, params)
	rRel, vRel := relativeState(pState, sState)
	missDist := math.Sqrt(rRel[0]*rRel[0] + rRel[1]*rRel[1] + rRel[2]*rRel[2])

	return TCAResult{
		TCA:          t0 + tMid/86400.0,
		MissDistance: missDist,
		RelPosition:  rRel,
		RelVelocity:  vRel,
		Converged:    false,
		Iterations:   cfg.MaxIterations,
		Method:       "bisection_max_iter",
	}
}

/**
 * evaluateTCAFunction computes 2 * r_rel · v_rel (the derivative of |r|²).
 * This is zero at the TCA.
 */
func evaluateTCAFunction(primary, secondary physics.State, t0, dtSec float64, params physics.PhysicsParams) float64 {
	pState := propagateState(primary, t0, t0+dtSec/86400.0, params)
	sState := propagateState(secondary, t0, t0+dtSec/86400.0, params)
	rRel, vRel := relativeState(pState, sState)

	// Return 2 * r · v (this is d(|r|²)/dt)
	return 2.0 * (rRel[0]*vRel[0] + rRel[1]*vRel[1] + rRel[2]*vRel[2])
}

/**
 * propagateState propagates a state by dt seconds using the main propagator.
 */
func propagateState(state physics.State, t0, t1 float64, params physics.PhysicsParams) physics.State {
	dt := (t1 - t0) * 86400.0 // Convert days to seconds
	xv := []float64{
		state.Pos[0], state.Pos[1], state.Pos[2],
		state.Vel[0], state.Vel[1], state.Vel[2],
	}

	next := orbital.PropagateCartesianState(xv, dt, t0, params)

	return physics.State{
		Pos: [3]float64{next[0], next[1], next[2]},
		Vel: [3]float64{next[3], next[4], next[5]},
	}
}

/**
 * relativeState computes relative position and velocity (secondary - primary).
 */
func relativeState(primary, secondary physics.State) ([3]float64, [3]float64) {
	rRel := [3]float64{
		secondary.Pos[0] - primary.Pos[0],
		secondary.Pos[1] - primary.Pos[1],
		secondary.Pos[2] - primary.Pos[2],
	}
	vRel := [3]float64{
		secondary.Vel[0] - primary.Vel[0],
		secondary.Vel[1] - primary.Vel[1],
		secondary.Vel[2] - primary.Vel[2],
	}
	return rRel, vRel
}
