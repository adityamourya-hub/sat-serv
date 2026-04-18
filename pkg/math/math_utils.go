package safemath

import (
	"fmt"
	stdmath "math"
)

// MaxFloat64 returns the maximum of two float64 values, ensuring NaN safety.
func MaxFloat64(a, b float64) float64 {
	if stdmath.IsNaN(a) {
		return b
	}
	if stdmath.IsNaN(b) {
		return a
	}
	if a > b {
		return a
	}
	return b
}

// MinFloat64 returns the minimum of two float64 values, ensuring NaN safety.
func MinFloat64(a, b float64) float64 {
	if stdmath.IsNaN(a) {
		return b
	}
	if stdmath.IsNaN(b) {
		return a
	}
	if a < b {
		return a
	}
	return b
}

// Clamp restricts a value to the [min, max] range.
func Clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// IsFinite returns true if v is not NaN or Inf.
func IsFinite(v float64) bool {
	return !stdmath.IsNaN(v) && !stdmath.IsInf(v, 0)
}

// ZeroMatrix initializes an r x c matrix of zeros.
func ZeroMatrix(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := 0; i < r; i++ {
		m[i] = make([]float64, c)
	}
	return m
}

// MatrixTranspose returns the transpose of matrix a.
func MatrixTranspose(a [][]float64) [][]float64 {
	if len(a) == 0 {
		return nil
	}
	r := len(a)
	c := len(a[0])
	t := ZeroMatrix(c, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t[j][i] = a[i][j]
		}
	}
	return t
}

// MatrixAdd computes a + b.
func MatrixAdd(a, b [][]float64) [][]float64 {
	r := len(a)
	c := len(a[0])
	res := ZeroMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res[i][j] = a[i][j] + b[i][j]
		}
	}
	return res
}

// MatrixSubtract computes a - b.
func MatrixSubtract(a, b [][]float64) [][]float64 {
	r := len(a)
	c := len(a[0])
	res := ZeroMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res[i][j] = a[i][j] - b[i][j]
		}
	}
	return res
}

// MatrixMultiply computes A * B.
func MatrixMultiply(a, b [][]float64) [][]float64 {
	r := len(a)
	if r == 0 {
		return nil
	}
	inter := len(a[0])
	c := len(b[0])
	res := ZeroMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := 0.0
			for k := 0; k < inter; k++ {
				sum += a[i][k] * b[k][j]
			}
			res[i][j] = sum
		}
	}
	return res
}

// MatrixVectorMultiply computes M * v.
func MatrixVectorMultiply(m [][]float64, v []float64) []float64 {
	r := len(m)
	if r == 0 {
		return nil
	}
	c := len(m[0])
	res := make([]float64, r)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += m[i][j] * v[j]
		}
		res[i] = sum
	}
	return res
}

// VectorAdd computes v1 + v2.
func VectorAdd(v1, v2 []float64) []float64 {
	n := len(v1)
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[i] = v1[i] + v2[i]
	}
	return res
}

// VectorSubtract computes v1 - v2.
func VectorSubtract(v1, v2 []float64) []float64 {
	n := len(v1)
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		res[i] = v1[i] - v2[i]
	}
	return res
}

// OuterProductScaled computes s * (v1 * v2^T).
func OuterProductScaled(v1, v2 []float64, s float64) [][]float64 {
	r := len(v1)
	c := len(v2)
	res := ZeroMatrix(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res[i][j] = s * v1[i] * v2[j]
		}
	}
	return res
}

// MatrixInverse computes the inverse of a small square matrix using LU decomposition.
// Note: For large matrices or high performance, use gonum.
func MatrixInverse(m [][]float64, jitter float64) ([][]float64, error) {
	n := len(m)
	if n == 0 {
		return nil, fmt.Errorf("empty matrix")
	}
	
	// Create augmented matrix [m | I]
	aug := ZeroMatrix(n, 2*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i][j] = m[i][j]
		}
		aug[i][i+n] = 1.0
	}

	// Gauss-Jordan elimination
	for i := 0; i < n; i++ {
		pivot := i
		maxVal := stdmath.Abs(aug[i][i])
		for k := i + 1; k < n; k++ {
			if v := stdmath.Abs(aug[k][i]); v > maxVal {
				maxVal = v
				pivot = k
			}
		}

		if maxVal < 1e-15 {
			// Try adding jitter if matrix is singular (SNC/UKF regularization)
			if jitter > 0 {
				aug[i][i] += jitter
				maxVal = stdmath.Abs(aug[i][i])
			}
			if maxVal < 1e-15 {
				return nil, fmt.Errorf("matrix is numerically singular")
			}
		}

		aug[i], aug[pivot] = aug[pivot], aug[i]

		div := aug[i][i]
		for j := i; j < 2*n; j++ {
			aug[i][j] /= div
		}

		for k := 0; k < n; k++ {
			if k != i {
				mul := aug[k][i]
				for j := i; j < 2*n; j++ {
					aug[k][j] -= mul * aug[i][j]
				}
			}
		}
	}

	res := ZeroMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			res[i][j] = aug[i][j+n]
		}
	}
	return res, nil
}
