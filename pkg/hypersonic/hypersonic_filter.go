package hypersonic

import (
	"math"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

// HypersonicState represents the 8-state vector for maneuvering targets.
// [x, y, z, vx, vy, vz, beta, ld]
// beta: Ballistic Coefficient (kg/m^2)
// ld: Lift-to-Drag Ratio
type HypersonicState struct {
	Pos  octree.Vector3
	Vel  octree.Vector3
	Beta float64
	LD   float64
}

func (s HypersonicState) ToVector() []float64 {
	return []float64{s.Pos.X, s.Pos.Y, s.Pos.Z, s.Vel.X, s.Vel.Y, s.Vel.Z, s.Beta, s.LD}
}

func VectorToHypersonicState(v []float64) HypersonicState {
	if len(v) < 8 {
		return HypersonicState{}
	}
	return HypersonicState{
		Pos:  octree.Vector3{X: v[0], Y: v[1], Z: v[2]},
		Vel:  octree.Vector3{X: v[3], Y: v[4], Z: v[5]},
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

		r := math.Sqrt(rx*rx + ry*ry + rz*rz)
		speed := math.Sqrt(vx*vx + vy*vy + vz*vz)

		// 1. Gravity (J2 + Point Mass) - Use unified orbital acceleration
		ax, ay, az := orbital.OrbitalAcceleration(rx, ry, rz, vx, vy, vz, jd)

		// 2. Atmosphere (Consolidated NRLMSISE-00)
		rho := orbital.GetDensityUnified(rx, ry, rz, jd)

		// 3. Drag
		if beta <= 0 {
			beta = 0.01 // Protection
		}
		dragMag := 0.5 * rho * speed * speed * 1000.0 / beta
		ax -= dragMag * vx / speed
		ay -= dragMag * vy / speed
		az -= dragMag * vz / speed

		// 4. Lift (Glide model)
		if math.Abs(ld) > 1e-6 && alt < 120.0 {
			h := crossProduct(octree.Vector3{X: rx, Y: ry, Z: rz}, octree.Vector3{X: vx, Y: vy, Z: vz})
			lVec := crossProduct(octree.Vector3{X: vx, Y: vy, Z: vz}, h)
			lMag := math.Sqrt(lVec.X*lVec.X + lVec.Y*lVec.Y + lVec.Z*lVec.Z)
			
			if lMag > 1e-9 {
				lDir := octree.Vector3{X: lVec.X / lMag, Y: lVec.Y / lMag, Z: lVec.Z / lMag}
				liftMag := dragMag * ld
				ax += liftMag * lDir.X
				ay += liftMag * lDir.Y
				az += liftMag * lDir.Z
			}
		}

		return []float64{vx, vy, vz, ax, ay, az, 0, 0}
	}

	_, next, _ := rk.Step(0, dt, s, f)
	return next
}

func crossProduct(a, b octree.Vector3) octree.Vector3 {
	return octree.Vector3{
		X: a.Y*b.Z - a.Z*b.Y,
		Y: a.Z*b.X - a.X*b.Z,
		Z: a.X*b.Y - a.Y*b.X,
	}
}
