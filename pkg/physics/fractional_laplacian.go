// pkg/physics/fractional_laplacian.go
// ADITI 4.0: Fractional Laplacian for HGV Wake Memory
//
// Implements Caputo derivative (α=0.75) for non-local drag prediction
// in HGV transitional flow regime (Kn≈0.01–1.0, altitude 30-80km).
//
// Mathematical foundation:
//   D^α[f](t) = 1/Γ(1-α) ∫₀ᵗ (t-τ)^{-α} f'(τ) dτ
//
// This captures the "heavy tail" memory of HGV bow-shock wakes (O(10s) effects)
// that standard Navier-Stokes cannot represent in transitional flow.
//
// Performance: Optimized for M2 Air memory bandwidth with 10-second window.

package physics

import (
	"math"
	"sync"
)

// FractionalLaplacian implements Caputo derivative for HGV wake prediction
type FractionalLaplacian struct {
	// Fractional derivative order (0 < α < 1)
	Alpha float64 // α = 0.75 for HGV transitional flow

	// Memory window (seconds)
	MemoryWindow float64 // 10 seconds

	// History buffer for wake computation
	history []float64 // Past wake values
	times  []float64 // Timestamps
	maxLen int       // Maximum history length

	// Performance optimization
	lookupTable []float64 // Precomputed (t-τ)^{-α} values
	dt           float64   // Time step for lookup table

	mu sync.RWMutex
}

// NewFractionalLaplacian creates a fractional Laplacian with α=0.75
func NewFractionalLaplacian() *FractionalLaplacian {
	memoryWindow := 10.0 // 10 seconds
	dt := 0.1            // 100ms resolution
	maxLen := int(memoryWindow/dt) + 1

	fl := &FractionalLaplacian{
		Alpha:        0.75,
		MemoryWindow: memoryWindow,
		history:      make([]float64, 0, maxLen),
		times:       make([]float64, 0, maxLen),
		maxLen:       maxLen,
		lookupTable:  make([]float64, maxLen),
		dt:           dt,
	}

	// Precompute lookup table for (t-τ)^{-α}
	fl.precomputeLookupTable()

	return fl
}

// precomputeLookupTable precomputes (t-τ)^{-α} for performance
func (fl *FractionalLaplacian) precomputeLookupTable() {
	for i := 0; i < fl.maxLen; i++ {
		t := float64(i) * fl.dt
		if t > 0 {
			fl.lookupTable[i] = math.Pow(t, -fl.Alpha)
		} else {
			fl.lookupTable[i] = 0 // Avoid singularity at t=0
		}
	}
}

// ComputeCaputo computes Caputo derivative of wake function
//
// D^α[f](t) = 1/Γ(1-α) ∫₀ᵗ (t-τ)^{-α} f'(τ) dτ
//
// Parameters:
//   - t: Current time (seconds)
//   - wakeFunc: Function returning wake derivative at time τ
//
// Returns: Fractional derivative value (dimensionless correction factor)
func (fl *FractionalLaplacian) ComputeCaputo(t float64, wakeFunc func(float64) float64) float64 {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	if t < fl.MemoryWindow {
		// Not enough history yet
		return 0.0
	}

	// Integral approximation using trapezoidal rule
	integral := 0.0
	dt := fl.dt

	// Integrate from t-MemoryWindow to t
	tauStart := t - fl.MemoryWindow
	numSteps := int(fl.MemoryWindow / dt)

	for i := 0; i <= numSteps; i++ {
		tau := tauStart + float64(i)*dt

		// Kernel: (t-τ)^{-α} - use lookup table for performance
		tauIndex := int((t - tau) / dt)
		if tauIndex >= len(fl.lookupTable) {
			tauIndex = len(fl.lookupTable) - 1
		}
		kernel := fl.lookupTable[tauIndex]

		// Derivative: f'(τ)
		wakeDerivative := wakeFunc(tau)

		// Trapezoidal rule
		if i == 0 || i == numSteps {
			integral += 0.5 * kernel * wakeDerivative * dt
		} else {
			integral += kernel * wakeDerivative * dt
		}
	}

	// Caputo prefactor: 1/Γ(1-α)
	gamma := math.Gamma(1.0 - fl.Alpha)

	return integral / gamma
}

// UpdateHistory updates the wake history buffer
func (fl *FractionalLaplacian) UpdateHistory(wakeValue, t float64) {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	fl.history = append(fl.history, wakeValue)
	fl.times = append(fl.times, t)

	// Prune old history beyond memory window
	cutoff := t - fl.MemoryWindow
	for len(fl.times) > 0 && fl.times[0] < cutoff {
		fl.history = fl.history[1:]
		fl.times = fl.times[1:]
	}

	// Enforce max length
	if len(fl.history) > fl.maxLen {
		fl.history = fl.history[len(fl.history)-fl.maxLen:]
		fl.times = fl.times[len(fl.times)-fl.maxLen:]
	}
}

// ApplyFractionalCorrection applies fractional Laplacian correction to predicted position
//
// For HGV tracking, this accounts for non-local drag effects from persistent
// plasma wakes that standard models miss.
//
// Parameters:
//   - predictedAz, predictedEl: Predicted pointing angles (degrees)
//   - velocity: HGV velocity (km/s)
//   - altitude: HGV altitude (km)
//   - currentTime: Current simulation time (seconds)
//
// Returns: Corrected Az, El with fractional wake memory
func (fl *FractionalLaplacian) ApplyFractionalCorrection(
	predictedAz, predictedEl float64,
	velocity, altitude, currentTime float64,
) (correctedAz, correctedEl float64) {
	// Wake model derivative function
	wakeDerivative := func(tau float64) float64 {
		// Simplified wake model: bow shock oscillations + vortex shedding
		// Bow shock: O(0.1s) oscillation
		bowShock := 0.3 * math.Sin(2.0*math.Pi*tau/0.1)

		// Vortex shedding: O(1s) oscillation
		vortex := 0.2 * math.Sin(2.0*math.Pi*tau/1.0)

		// Altitude effect: stronger wake at lower alt (denser air)
		altFactor := math.Exp(-(altitude - 30.0) / 20.0)

		// Velocity effect: higher velocity = stronger wake
		velFactor := velocity / 7.0 // Normalize to Mach ~20

		return (bowShock + vortex) * altFactor * velFactor
	}

	// Compute Caputo derivative
	fractionalCorrection := fl.ComputeCaputo(currentTime, wakeDerivative)

	// Update history with current wake state
	currentWake := wakeDerivative(currentTime)
	fl.UpdateHistory(currentWake, currentTime)

	// Apply correction to pointing angles
	// Fractional correction is dimensionless, typically ±0.05 rad (~3°)
	correctionRad := fractionalCorrection * 0.05 // Scale to reasonable angle

	// Convert to degrees
	correctionDeg := correctionRad * DEG_PER_RAD

	// Apply correction (fractional drag causes lag/lead in position)
	correctedAz = predictedAz + correctionDeg
	correctedEl = predictedEl + correctionDeg

	return correctedAz, correctedEl
}

// Reset clears the history buffer (for new target acquisition)
func (fl *FractionalLaplacian) Reset() {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	fl.history = make([]float64, 0, fl.maxLen)
	fl.times = make([]float64, 0, fl.maxLen)
}

// GetMemoryUsage returns current memory buffer size
func (fl *FractionalLaplacian) GetMemoryUsage() int {
	fl.mu.RLock()
	defer fl.mu.RUnlock()
	return len(fl.history)
}
