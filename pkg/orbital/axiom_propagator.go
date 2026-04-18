package orbital

import (
	"fmt"
	"math"
	"ssis/pkg/physics"
	"time"
)

/**
 * DynamicsKernel defines the interface for learned force models (rKAN, GHNN).
 */
type DynamicsKernel interface {
	GetAcceleration(q, v [3]float64, jd float64) [3]float64
}

/**
 * AxiomPropagator implements the Sovereign Physics Core.
 * It uses a 4th-Order Symplectic Yoshida Integrator to ensure long-term energy consistency.
 */
type AxiomPropagator struct {
	Kernel       DynamicsKernel
	MU           float64 // Standard gravitational parameter (km^3/s^2)
	InitialState physics.State
	Epoch        time.Time
	Params       physics.PhysicsParams // Force model parameters (Drag, SRP, etc.)
}

/**
 * NewAxiomPropagator initializes the Sovereign propagator with a dynamics kernel.
 */
func NewAxiomPropagator(kernel DynamicsKernel, init physics.State, epoch time.Time, params physics.PhysicsParams) *AxiomPropagator {
	return &AxiomPropagator{
		Kernel:       kernel,
		MU:           physics.EARTH_MU_IAU,
		InitialState: init,
		Epoch:        epoch,
		Params:       params,
	}
}

// toJulianDate converts a Go time.Time to Julian Date.
func toJulianDate(t time.Time) float64 {
	// Standard formula for JD from Unix timestamp
	return float64(t.Unix())/86400.0 + 2440587.5
}

// getConservativeAcceleration computes forces from the scalar energy manifold (Kepler + J2 + GHNN).
func (p *AxiomPropagator) getConservativeAcceleration(q, v [3]float64, jd float64) [3]float64 {
	// 1. Keplerian + Zonal (J2-J4) + Tesseral (J2,2) Harmonics
	ax, ay, az := GravityAccelerationGCRF(q[0], q[1], q[2], jd)

	// 2. Axiom-Zero GHNN (Learned Conservative Potential)
	var perturb [3]float64
	if p.Kernel != nil {
		perturb = p.Kernel.GetAcceleration(q, v, jd)
	}

	return [3]float64{ax + perturb[0], ay + perturb[1], az + perturb[2]}
}

// getNonConservativeAcceleration computes dissipative forces (Drag + SRP).
func (p *AxiomPropagator) getNonConservativeAcceleration(q, v [3]float64, jd float64) [3]float64 {
	ax, ay, az := 0.0, 0.0, 0.0

	// 1. Atmospheric Drag
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	alt := r - physics.EARTH_R_EQUATORIAL
	if alt < 1200 {
		rho := GetDensityUnified(q[0], q[1], q[2], jd)
		v2 := v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
		speed := math.Sqrt(v2)
		if speed > 1e-6 {
			dragAcc := 0.5 * rho * v2 * (p.Params.DragArea / p.Params.Mass) * p.Params.Cd
			ax -= dragAcc * v[0] / speed
			ay -= dragAcc * v[1] / speed
			az -= dragAcc * v[2] / speed
		}
	}

	// 2. Solar Radiation Pressure
	if p.Params.SRPArea > 0 && p.Params.Cr > 0 {
		sax, say, saz := SolarRadiationPressure(q[0], q[1], q[2], jd, p.Params.SRPArea, p.Params.Mass, p.Params.Cr)
		ax += sax
		ay += say
		az += saz
	}

	return [3]float64{ax, ay, az}
}

/**
 * Propagate performs symplectic integration from the initial epoch.
 *
 * @param minutesSinceEpoch: Target delta time in minutes.
 * @return: Integrated State.
 */
func (p *AxiomPropagator) Propagate(minutesSinceEpoch float64) (physics.State, error) {
	// Sanity Guard: Numerical Stability Limit (10 years)
	if math.Abs(minutesSinceEpoch) > MaxPropagationDays*1440 {
		return physics.State{}, fmt.Errorf("Axiom propagation time delta (%.2f min) exceeds stability limits (10 years)", minutesSinceEpoch)
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
	q := [3]float64{p.InitialState.Pos[0], p.InitialState.Pos[1], p.InitialState.Pos[2]}
	v := [3]float64{p.InitialState.Vel[0], p.InitialState.Vel[1], p.InitialState.Vel[2]}

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

	// Integration Loop: Hybrid Yoshida-4 Handoff
	for s := 0; s < steps; s++ {
		currentJD := initialJD + (float64(s) * stepSize / 86400.0)

		// Stage 1: Drift -> Kick (C + NC)
		q[0] += c1 * v[0] * stepSize
		q[1] += c1 * v[1] * stepSize
		q[2] += c1 * v[2] * stepSize

		ac1 := p.getConservativeAcceleration(q, v, currentJD)
		anc1 := p.getNonConservativeAcceleration(q, v, currentJD)
		v[0] += d1 * (ac1[0] + anc1[0]) * stepSize
		v[1] += d1 * (ac1[1] + anc1[1]) * stepSize
		v[2] += d1 * (ac1[2] + anc1[2]) * stepSize

		// Stage 2: Drift -> Kick (C + NC)
		q[0] += c2 * v[0] * stepSize
		q[1] += c2 * v[1] * stepSize
		q[2] += c2 * v[2] * stepSize

		ac2 := p.getConservativeAcceleration(q, v, currentJD)
		anc2 := p.getNonConservativeAcceleration(q, v, currentJD)
		v[0] += d2 * (ac2[0] + anc2[0]) * stepSize
		v[1] += d2 * (ac2[1] + anc2[1]) * stepSize
		v[2] += d2 * (ac2[2] + anc2[2]) * stepSize

		// Stage 3: Drift -> Kick (C + NC)
		q[0] += c3 * v[0] * stepSize
		q[1] += c3 * v[1] * stepSize
		q[2] += c3 * v[2] * stepSize

		ac3 := p.getConservativeAcceleration(q, v, currentJD)
		anc3 := p.getNonConservativeAcceleration(q, v, currentJD)
		v[0] += d3 * (ac3[0] + anc3[0]) * stepSize
		v[1] += d3 * (ac3[1] + anc3[1]) * stepSize
		v[2] += d3 * (ac3[2] + anc3[2]) * stepSize

		// Stage 4: Drift Final
		q[0] += c4 * v[0] * stepSize
		q[1] += c4 * v[1] * stepSize
		q[2] += c4 * v[2] * stepSize
	}

	// Compute final Energy (Hamiltonian) for audit
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	vSq := v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
	energy := 0.5*vSq - p.MU/r // Simplified Keplerian energy

	return physics.State{
		Pos:     physics.Vector3{q[0], q[1], q[2]},
		Vel:     physics.Vector3{v[0], v[1], v[2]},
		EpochJD: initialJD + (float64(steps) * stepSize / 86400.0),
		E_hal:   energy,
	}, nil
}
