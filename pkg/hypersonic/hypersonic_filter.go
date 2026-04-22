package hypersonic

import (
	"math"
	"ssis/pkg/physics"
)

// HypersonicState removed (migrated to pkg/physics)

func HypersonicToVector(s physics.HypersonicState) []float64 {
	return []float64{s.Pos[0], s.Pos[1], s.Pos[2], s.Vel[0], s.Vel[1], s.Vel[2], s.Beta, s.LD}
}

func VectorToHypersonicState(v []float64) physics.HypersonicState {
	if len(v) < 8 {
		return physics.HypersonicState{}
	}
	return physics.HypersonicState{
		Pos:  physics.Vector3{v[0], v[1], v[2]},
		Vel:  physics.Vector3{v[3], v[4], v[5]},
		Beta: v[6],
		LD:   v[7],
	}
}

// HypersonicPropagator performs high-fidelity integration of the 8-state vector.
func HypersonicPropagator(s []float64, dt float64, jd float64) []float64 {
	rk := physics.NewIntegratorRKF45()
	// High precision for 1 Hz tracking
	rk.AbsError = 1e-8
	rk.RelError = 1e-8

	f := func(t float64, x []float64) []float64 {
		rx, ry, rz := x[0], x[1], x[2]
		vx, vy, vz := x[3], x[4], x[5]
		beta := x[6]
		ld := x[7]

		// r := math.Sqrt(rx*rx + ry*ry + rz*rz) (migrated to OrbitalAcceleration)
		speed := math.Sqrt(vx*vx + vy*vy + vz*vz)

		// 1. Gravity (J2 + Point Mass) - Use unified orbital acceleration
		ax, ay, az := physics.OrbitalAcceleration(rx, ry, rz, vx, vy, vz, jd, physics.PhysicsParams{})

		// 2. Atmosphere (Consolidated NRLMSISE-00)
		rho := physics.GetDensityUnified(rx, ry, rz, jd)

		// 3. Drag
		if beta <= 0 {
			beta = 0.01 // Protection
		}
		dragMag := 0.5 * rho * speed * speed * 1000.0 / beta
		ax -= dragMag * vx / speed
		ay -= dragMag * vy / speed
		az -= dragMag * vz / speed

		// 4. Lift (Glide model)
		if math.Abs(ld) > 1e-6 {
			h := crossProduct(physics.Vector3{rx, ry, rz}, physics.Vector3{vx, vy, vz})
			lVec := crossProduct(physics.Vector3{vx, vy, vz}, h)
			lMag := math.Sqrt(lVec[0]*lVec[0] + lVec[1]*lVec[1] + lVec[2]*lVec[2])

			if lMag > 1e-9 {
				lDir := physics.Vector3{lVec[0] / lMag, lVec[1] / lMag, lVec[2] / lMag}
				liftMag := dragMag * ld
				ax += liftMag * lDir[0]
				ay += liftMag * lDir[1]
				az += liftMag * lDir[2]
			}
		}

		return []float64{vx, vy, vz, ax, ay, az, 0, 0}
	}

	_, next, _ := rk.Step(0, dt, s, f)
	return next
}

/**
 * ComputeHypersonicJacobian calculates the 8x8 analytical Jacobian (df/dx).
 * State: [x, y, z, vx, vy, vz, beta, ld]
 */
func ComputeHypersonicJacobian(x, y, z, vx, vy, vz, beta, ld, jd float64) [][]float64 {
	a := make([][]float64, 8)
	for i := range a {
		a[i] = make([]float64, 8)
	}

	// 1. Identity for upper right (dr/dt = v)
	a[0][3], a[1][4], a[2][5] = 1.0, 1.0, 1.0

	// 2. Gravity Gradient (da/dr) - J2 Approximation
	r2 := x*x + y*y + z*z
	r := math.Sqrt(r2)
	invR := 1.0 / r
	invR3 := invR * invR * invR
	invR2 := invR * invR
	muR5 := 3.0 * physics.EARTH_MU_IAU * invR3 * invR2

	a[3][0] = muR5*x*x - physics.EARTH_MU_IAU*invR3
	a[3][1] = muR5 * x * y
	a[3][2] = muR5 * x * z
	a[4][0] = muR5 * y * x
	a[4][1] = muR5*y*y - physics.EARTH_MU_IAU*invR3
	a[4][2] = muR5 * y * z
	a[5][0] = muR5 * z * x
	a[5][1] = muR5 * z * y
	a[5][2] = muR5*z*z - physics.EARTH_MU_IAU*invR3

	// J2 Gradient
	j2Fac := (1.5 * physics.J2 * physics.EARTH_MU_IAU * physics.EARTH_R_EQUATORIAL * physics.EARTH_R_EQUATORIAL) * invR3 * invR2
	a[3][0] += j2Fac * (1.0 - 5.0*z*z*invR2)
	a[4][1] += j2Fac * (1.0 - 5.0*z*z*invR2)
	a[5][2] += j2Fac * (3.0 - 5.0*z*z*invR2)

	// 3. Aero Accel Partials
	speed2 := vx*vx + vy*vy + vz*vz
	speed := math.Sqrt(speed2)
	if speed > 0.01 && beta > 0 {
		rho := physics.GetDensityUnified(x, y, z, jd)
		dragConst := 0.5 * rho * 1000.0 / beta

		// da_drag / dv
		a[3][3] = -dragConst * (2*vx*vx/speed + speed)
		a[3][4] = -dragConst * (2 * vx * vy / speed)
		a[3][5] = -dragConst * (2 * vx * vz / speed)
		a[4][3] = -dragConst * (2 * vy * vx / speed)
		a[4][4] = -dragConst * (2*vy*vy/speed + speed)
		a[4][5] = -dragConst * (2 * vy * vz / speed)
		a[5][3] = -dragConst * (2 * vz * vx / speed)
		a[5][4] = -dragConst * (2 * vz * vy / speed)
		a[5][5] = -dragConst * (2*vz*vz/speed + speed)

		// da_drag / dbeta
		adx := -dragConst * speed * vx
		ady := -dragConst * speed * vy
		adz := -dragConst * speed * vz
		a[3][6] = -adx / beta
		a[4][6] = -ady / beta
		a[5][6] = -adz / beta

		// 4. Lift Partials (Simplified for STM)
		if math.Abs(ld) > 1e-6 {
			nx, ny, nz := physics.CalculateLiftVectorECI(x, y, z, vx, vy, vz)
			liftMag := (0.5 * rho * speed2 * 1000.0 / beta) * ld

			// da_lift / d_ld
			a[3][7] = liftMag / ld * nx
			a[4][7] = liftMag / ld * ny
			a[5][7] = liftMag / ld * nz

			// da_lift / dbeta
			a[3][6] -= (liftMag * nx) / beta
			a[4][6] -= (liftMag * ny) / beta
			a[5][6] -= (liftMag * nz) / beta
		}
	}

	return a
}

/**
 * HypersonicPropagatorWithSTM integrates the 8D state and 8x8 STM
 * simultaneously (72 states total).
 * dot(Phi) = A * Phi
 */
func HypersonicPropagatorWithSTM(s []float64, phi []float64, dt float64, jd float64) ([]float64, []float64) {
	if len(phi) != 64 {
		phi = make([]float64, 64)
		for i := 0; i < 8; i++ {
			phi[i*8+i] = 1.0 // Identity
		}
	}

	rk := physics.NewIntegratorRKF45()
	rk.AbsError = 1e-8
	rk.RelError = 1e-8

	// Combined state vector [8 state | 64 STM]
	y0 := make([]float64, 72)
	copy(y0[0:8], s)
	copy(y0[8:72], phi)

	f := func(t float64, y []float64) []float64 {
		x_s := y[0:8]
		Phi := y[8:72]

		// 1. Compute Dynamics (f)
		rx, ry, rz := x_s[0], x_s[1], x_s[2]
		vx, vy, vz := x_s[3], x_s[4], x_s[5]
		beta, ld := x_s[6], x_s[7]

		speed := math.Sqrt(vx*vx + vy*vy + vz*vz)
		ax, ay, az := physics.OrbitalAcceleration(rx, ry, rz, vx, vy, vz, jd+t/86400.0, physics.PhysicsParams{})
		rho := physics.GetDensityUnified(rx, ry, rz, jd+t/86400.0)

		if beta > 0 && speed > 0.01 {
			dragMag := 0.5 * rho * speed * speed * 1000.0 / beta
			ax -= dragMag * vx / speed
			ay -= dragMag * vy / speed
			az -= dragMag * vz / speed

			if math.Abs(ld) > 1e-6 {
				nx, ny, nz := physics.CalculateLiftVectorECI(rx, ry, rz, vx, vy, vz)
				liftMag := dragMag * ld
				ax += liftMag * nx
				ay += liftMag * ny
				az += liftMag * nz
			}
		}

		derivs := make([]float64, 72)
		derivs[0], derivs[1], derivs[2] = vx, vy, vz
		derivs[3], derivs[4], derivs[5] = ax, ay, az
		derivs[6], derivs[7] = 0, 0 // Parameters assumed constant over dt

		// 2. Compute Jacobian (A)
		A := ComputeHypersonicJacobian(rx, ry, rz, vx, vy, vz, beta, ld, jd+t/86400.0)

		// 3. Compute dot(Phi) = A * Phi
		for i := 0; i < 8; i++ {
			for j := 0; j < 8; j++ {
				sum := 0.0
				for k := 0; k < 8; k++ {
					sum += A[i][k] * Phi[k*8+j]
				}
				derivs[8+i*8+j] = sum
			}
		}

		return derivs
	}

	_, yn, _ := rk.Step(0, dt, y0, f)
	return yn[0:8], yn[8:72]
}

func crossProduct(a, b physics.Vector3) physics.Vector3 {
	return physics.Vector3{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}
