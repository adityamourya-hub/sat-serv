package fusion

import (
	safemath "ssis/pkg/math"
	"ssis/pkg/orbital"
)

func stateToVector(s orbital.State) []float64 {
	return []float64{s.X, s.Y, s.Z, s.VX, s.VY, s.VZ}
}

func vectorToState(v []float64) orbital.State {
	if len(v) < 6 {
		return orbital.State{}
	}
	return orbital.State{X: v[0], Y: v[1], Z: v[2], VX: v[3], VY: v[4], VZ: v[5]}
}

func normalizeCovariance(p [][]float64, n int) [][]float64 {
	if len(p) == n && len(p[0]) == n {
		return p
	}
	res := safemath.ZeroMatrix(n, n)
	for i := 0; i < n && i < len(p); i++ {
		for j := 0; j < n && j < len(p[i]); j++ {
			res[i][j] = p[i][j]
		}
	}
	return res
}

func enforcePositiveDiagonal(p [][]float64, jitter float64) [][]float64 {
	n := len(p)
	for i := 0; i < n; i++ {
		if p[i][i] < jitter {
			p[i][i] = jitter
		}
	}
	return p
}

func symmetrize(p [][]float64) [][]float64 {
	n := len(p)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := (p[i][j] + p[j][i]) / 2.0
			p[i][j] = avg
			p[j][i] = avg
		}
	}
	return p
}

func outerScaled(v1, v2 []float64, s float64) [][]float64 {
	return safemath.OuterProductScaled(v1, v2, s)
}

func vecAdd(v1, v2 []float64) []float64 {
	return safemath.VectorAdd(v1, v2)
}

func vecSub(v1, v2 []float64) []float64 {
	return safemath.VectorSubtract(v1, v2)
}

func matAdd(a, b [][]float64) [][]float64 {
	return safemath.MatrixAdd(a, b)
}

func matSub(a, b [][]float64) [][]float64 {
	return safemath.MatrixSubtract(a, b)
}

func matMul(a, b [][]float64) [][]float64 {
	return safemath.MatrixMultiply(a, b)
}

func matTranspose(a [][]float64) [][]float64 {
	return safemath.MatrixTranspose(a)
}

func matVecMul(m [][]float64, v []float64) []float64 {
	return safemath.MatrixVectorMultiply(m, v)
}

func matInverse(m [][]float64, jitter float64) ([][]float64, error) {
	return safemath.MatrixInverse(m, jitter)
}

func maxFloat64(a, b float64) float64 {
	return safemath.MaxFloat64(a, b)
}

// MatrixTrace computes the trace of a matrix.
func MatrixTrace(m [][]float64) float64 {
	if len(m) == 0 {
		return 0
	}
	t := 0.0
	n := len(m)
	if len(m[0]) < n {
		n = len(m[0])
	}
	for i := 0; i < n; i++ {
		t += m[i][i]
	}
	return t
}
