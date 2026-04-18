package orbital

import (
	"fmt"
	"math"
	"time"
)

/**
 * AxiomPropagator implements the Sovereign Physics Core utilizing the rKAN manifold.
 * It uses a 4th-Order Symplectic Yoshida Integrator to ensure long-term energy consistency.
 * Supports the momentum-aware 13-feature manifold.
 */
type AxiomPropagator struct {
	Kernel       *RKANKernel
	MU           float64 // Standard gravitational parameter (km^3/s^2)
	InitialState State
	Epoch        time.Time
}

/**
 * NewAxiomPropagator initializes the Sovereign propagator with an rKAN kernel.
 */
func NewAxiomPropagator(kernel *RKANKernel, init State, epoch time.Time) *AxiomPropagator {
	return &AxiomPropagator{
		Kernel:       kernel,
		MU:           398600.4418, // WGS-84
		InitialState: init,
		Epoch:        epoch,
	}
}

// toJulianDate converts a Go time.Time to Julian Date.
func toJulianDate(t time.Time) float64 {
	// Standard formula for JD from Unix timestamp
	return float64(t.Unix())/86400.0 + 2440587.5
}

// getAcceleration computes the total acceleration (Keplerian + AI Perturbation)
// p and jd are required for the advanced Physical Feature Layer (Resonance, Ephemeris).
func (p *AxiomPropagator) getAcceleration(q, v [3]float64, jd float64) [3]float64 {
	rSq := q[0]*q[0] + q[1]*q[1] + q[2]*q[2]
	r := math.Sqrt(rSq) + 1e-12
	
	// 1. Keplerian Central Force
	kepler := [3]float64{
		-p.MU * q[0] / (r * r * r),
		-p.MU * q[1] / (r * r * r),
		-p.MU * q[2] / (r * r * r),
	}
	
	// 2. Axiom-Zero rKAN Perturbation (Learned 13-feature logic)
	perturb := p.Kernel.GetAcceleration(q, v, jd)
	
	return [3]float64{
		kepler[0] + perturb[0],
		kepler[1] + perturb[1],
		kepler[2] + perturb[2],
	}
}

/**
 * Propagate performs symplectic integration from the initial epoch.
 *
 * @param minutesSinceEpoch: Target delta time in minutes.
 * @return: Integrated State.
 */
func (p *AxiomPropagator) Propagate(minutesSinceEpoch float64) (State, error) {
	// Sanity Guard: Numerical Stability Limit (10 years)
	if math.Abs(minutesSinceEpoch) > MaxPropagationDays*1440 {
		return State{}, fmt.Errorf("Axiom propagation time delta (%.2f min) exceeds stability limits (10 years)", minutesSinceEpoch)
	}

	dt := minutesSinceEpoch * 60.0 // seconds
	
	// For very small deltas or zero, return initial state
	if math.Abs(dt) < 1e-6 {
		return p.InitialState, nil
	}

	// Symplectic Stage Coefficients for Yoshida 4th Order
	w1 := 1.3512071919596576
	w0 := -1.7024143839193153
	
	c1 := w1 / 2.0
	c2 := (w1 + w0) / 2.0
	c3 := c2
	c4 := c1
	
	d1 := w1
	d2 := w0
	d3 := w1
	
	// Initialize integrator state
	q := [3]float64{p.InitialState.X, p.InitialState.Y, p.InitialState.Z}
	v := [3]float64{p.InitialState.VX, p.InitialState.VY, p.InitialState.VZ}

	// Time handling for Ephemeris (Sun/Moon features)
	initialJD := toJulianDate(p.Epoch)
	
	// Dynamic stepping
	stepSize := 60.0 // 1 minute steps
	steps := int(math.Abs(dt) / stepSize)
	if steps == 0 {
		steps = 1
		stepSize = dt
	} else {
		stepSize = dt / float64(steps)
	}

	// Integration Loop
	for s := 0; s < steps; s++ {
		currentJD := initialJD + (float64(s)*stepSize / 86400.0)
		
		// Stage 1
		q[0] += c1 * v[0] * stepSize
		q[1] += c1 * v[1] * stepSize
		q[2] += c1 * v[2] * stepSize
		
		acc1 := p.getAcceleration(q, v, currentJD)
		v[0] += d1 * acc1[0] * stepSize
		v[1] += d1 * acc1[1] * stepSize
		v[2] += d1 * acc1[2] * stepSize
		
		// Stage 2
		q[0] += c2 * v[0] * stepSize
		q[1] += c2 * v[1] * stepSize
		q[2] += c2 * v[2] * stepSize
		
		acc2 := p.getAcceleration(q, v, currentJD)
		v[0] += d2 * acc2[0] * stepSize
		v[1] += d2 * acc2[1] * stepSize
		v[2] += d2 * acc2[2] * stepSize
		
		// Stage 3
		q[0] += c3 * v[0] * stepSize
		q[1] += c3 * v[1] * stepSize
		q[2] += c3 * v[2] * stepSize
		
		acc3 := p.getAcceleration(q, v, currentJD)
		v[0] += d3 * acc3[0] * stepSize
		v[1] += d3 * acc3[1] * stepSize
		v[2] += d3 * acc3[2] * stepSize
		
		// Stage 4
		q[0] += c4 * v[0] * stepSize
		q[1] += c4 * v[1] * stepSize
		q[2] += c4 * v[2] * stepSize
	}

	// Compute final Energy (Hamiltonian) for audit
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	vSq := v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
	energy := 0.5*vSq - p.MU/r // Simplified Keplerian energy
	
	return State{
		X: q[0], Y: q[1], Z: q[2],
		VX: v[0], VY: v[1], VZ: v[2],
		E_hal: energy,
	}, nil
}
