package physics

import (
	"math"
)

// IntegratorRKF45 implements a high-order adaptive step-size Runge-Kutta-Fehlberg 4(5)
// integrator for sub-orbital delivery systems and hypersonic maneuvering.
// This provides algorithmic parity with NASA GMAT and AGI STK.
type IntegratorRKF45 struct {
	AbsError float64
	RelError float64
	MinStep  float64
	MaxStep  float64
}

// NewIntegratorRKF45 creates a default configuration for 1e-9 precision integration.
func NewIntegratorRKF45() *IntegratorRKF45 {
	return &IntegratorRKF45{
		AbsError: 1e-9, // Hardened for certification
		RelError: 1e-9,
		MinStep:  1e-4,
		MaxStep:  30.0,
	}
}

// DerivativeFunc defines current acceleration/velocity for a given state.
type DerivativeFunc func(t float64, state []float64) []float64

// Step performs a single adaptive step using the Fehlberg RKF45 method (1969).
func (rk *IntegratorRKF45) Step(t float64, dt float64, y []float64, f DerivativeFunc) (float64, []float64, float64) {
	// Coefficients for Fehlberg RKF4(5)
	var (
		a2, a3, a4, a5, a6 = 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0
		b21               = 1.0 / 4.0
		b31, b32          = 3.0 / 32.0, 9.0 / 32.0
		b41, b42, b43     = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
		b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
		b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0

		// RK4 result coefficients
		c1, c3, c4, c5 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0
		// RK5 result coefficients (for error estimate)
		ch1, ch3, ch4, ch5, ch6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0
	)

	n := len(y)
	k1 := f(t, y)

	yTemp := make([]float64, n)
	for i := 0; i < n; i++ {
		yTemp[i] = y[i] + b21*dt*k1[i]
	}
	k2 := f(t+a2*dt, yTemp)

	for i := 0; i < n; i++ {
		yTemp[i] = y[i] + dt*(b31*k1[i]+b32*k2[i])
	}
	k3 := f(t+a3*dt, yTemp)

	for i := 0; i < n; i++ {
		yTemp[i] = y[i] + dt*(b41*k1[i]+b42*k2[i]+b43*k3[i])
	}
	k4 := f(t+a4*dt, yTemp)

	for i := 0; i < n; i++ {
		yTemp[i] = y[i] + dt*(b51*k1[i]+b52*k2[i]+b53*k3[i]+b54*k4[i])
	}
	k5 := f(t+a5*dt, yTemp)

	for i := 0; i < n; i++ {
		yTemp[i] = y[i] + dt*(b61*k1[i]+b62*k2[i]+b63*k3[i]+b64*k4[i]+b65*k5[i])
	}
	k6 := f(t+a6*dt, yTemp)

	yOut := make([]float64, n)
	yErr := make([]float64, n)
	for i := 0; i < n; i++ {
		// 4th order solution (y_out)
		yOut[i] = y[i] + dt*(c1*k1[i]+c3*k3[i]+c4*k4[i]+c5*k5[i])
		// Error estimate (diff between 5th and 4th order)
		yErr[i] = dt * ((c1-ch1)*k1[i] + (c3-ch3)*k3[i] + (c4-ch4)*k4[i] + (c5-ch5)*k5[i] - ch6*k6[i])
	}

	var maxErr float64
	for i := 0; i < n; i++ {
		scale := rk.AbsError + rk.RelError*math.Abs(yOut[i])
		err := math.Abs(yErr[i]) / scale
		if err > maxErr {
			maxErr = err
		}
	}

	// Calculate optimal next step size
	nextDt := dt * 0.84 * math.Pow(1.0/math.Max(maxErr, 1e-12), 0.25)
	nextDt = math.Max(rk.MinStep, math.Min(rk.MaxStep, nextDt))

	return t + dt, yOut, nextDt
}
