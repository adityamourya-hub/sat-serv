package physics

import (
	"math"
)

/**
 * PropagateCartesianState performs numerical integration of orbital states 
 * using a 4th-Order Runge-Kutta (RK4) method.
 * Zero-allocation inner loop.
 */
func PropagateCartesianState(x []float64, dt float64, jd float64, params PhysicsParams) []float64 {
	h := 10.0 // 10 sec step for inner RK4
	if dt < 0 {
		h = -10.0
	}
	if math.Abs(dt) < math.Abs(h) {
		h = dt
	}
	var curr [6]float64
	copy(curr[:], x[:6])
	elapsed := 0.0

	var k1, k2, k3, k4, tmp [6]float64

	for math.Abs(elapsed) < math.Abs(dt) {
		step := h
		if math.Abs(dt-elapsed) < math.Abs(h) {
			step = dt - elapsed
		}

		// k1 = f(curr)
		ax, ay, az := OrbitalAcceleration(curr[0], curr[1], curr[2], curr[3], curr[4], curr[5], jd, params)
		k1 = [6]float64{curr[3], curr[4], curr[5], ax, ay, az}

		// k2 = f(curr + 0.5*step*k1)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k1[i]*step*0.5
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k2 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// k3 = f(curr + 0.5*step*k2)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k2[i]*step*0.5
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k3 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// k4 = f(curr + step*k3)
		for i := 0; i < 6; i++ {
			tmp[i] = curr[i] + k3[i]*step
		}
		ax, ay, az = OrbitalAcceleration(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], jd, params)
		k4 = [6]float64{tmp[3], tmp[4], tmp[5], ax, ay, az}

		// Update: curr += (step/6) * (k1 + 2*k2 + 2*k3 + k4)
		s6 := step / 6.0
		for i := 0; i < 6; i++ {
			curr[i] += s6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
		}
		elapsed += step
	}
	return curr[:]
}
