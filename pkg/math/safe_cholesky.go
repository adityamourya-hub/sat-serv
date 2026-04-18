package safemath

import (
	"fmt"
	stdmath "math"
)

const (
	// DefaultRegularizationEpsilon is the certification baseline epsilon used for C_reg = C + epsilon*I.
	DefaultRegularizationEpsilon = 1e-12
	maxRegularizationTries       = 10
)

// CholeskyMetadata captures finite-precision diagnostics for audited pipelines.
type CholeskyMetadata struct {
	RegularizationEpsilon float64 `json:"regularization_epsilon"`
	MinDiagonalBefore     float64 `json:"min_diagonal_before"`
	MinDiagonalAfter      float64 `json:"min_diagonal_after"`
	ConditionProxy        float64 `json:"condition_proxy"`
	Attempts              int     `json:"attempts"`
	WasRegularized        bool    `json:"was_regularized"`
	Finite                bool    `json:"finite"`
}

// SafeCholesky computes a numerically guarded lower-triangular Cholesky factor.
// It guarantees a finite output or returns an explicit error without propagating NaN/Inf.
// @ requires len(cov) > 0
// @ requires epsilon >= 0
// @ ensures err == nil ==> len(result0) == len(cov)
// @ ensures err == nil ==> result1.Finite
// @ ensures err == nil ==> result1.MinDiagonalAfter > 0
func SafeCholesky(cov [][]float64, epsilon float64) ([][]float64, CholeskyMetadata, error) {
	n := len(cov)
	meta := CholeskyMetadata{Finite: false}
	if n == 0 {
		return nil, meta, fmt.Errorf("empty covariance matrix")
	}
	for i := 0; i < n; i++ {
		if len(cov[i]) != n {
			return nil, meta, fmt.Errorf("non-square covariance matrix")
		}
	}

	sym, minDiag, maxDiag, finite := sanitizeSymmetric(cov)
	meta.MinDiagonalBefore = minDiag
	meta.Finite = finite
	if !finite {
		return nil, meta, fmt.Errorf("covariance contains NaN or Inf")
	}

	if epsilon <= 0 || stdmath.IsNaN(epsilon) || stdmath.IsInf(epsilon, 0) {
		epsilon = DefaultRegularizationEpsilon
	}
	scale := stdmath.Max(1.0, stdmath.Abs(maxDiag))
	baseReg := stdmath.Max(epsilon, epsilon*scale)

	bestErr := fmt.Errorf("safe cholesky did not converge")
	//@ decreases maxRegularizationTries
	for attempt := 0; attempt < maxRegularizationTries; attempt++ {
		//@ invariant attempt >= 0 && attempt <= maxRegularizationTries
		reg := baseReg * stdmath.Pow(10.0, float64(attempt))
		cand := addDiagonal(sym, reg)
		l, minAfter, err := factorLowerFinite(cand, reg)
		meta.Attempts = attempt + 1
		meta.RegularizationEpsilon = reg
		meta.MinDiagonalAfter = minAfter
		meta.ConditionProxy = conditionProxyFromDiag(l)
		meta.WasRegularized = reg > 0

		if err == nil {
			meta.Finite = true
			return l, meta, nil
		}
		bestErr = err
	}

	meta.Finite = false
	return nil, meta, fmt.Errorf("safe cholesky failed after %d attempts: %w", maxRegularizationTries, bestErr)
}

// RegularizeCovariance applies the same bounded regularization policy as SafeCholesky and returns C_reg.
func RegularizeCovariance(cov [][]float64, epsilon float64) ([][]float64, CholeskyMetadata, error) {
	_, meta, err := SafeCholesky(cov, epsilon)
	if err != nil {
		return nil, meta, err
	}
	sym, _, _, _ := sanitizeSymmetric(cov)
	return addDiagonal(sym, meta.RegularizationEpsilon), meta, nil
}

// MahalanobisSquared computes d^T C^{-1} d using only finite forward substitutions from SafeCholesky.
// @ requires len(diff) > 0
// @ requires len(cov) == len(diff)
// @ requires epsilon >= 0
// @ ensures err == nil ==> result0 >= 0
// @ ensures err == nil ==> result1.Finite
func MahalanobisSquared(diff []float64, cov [][]float64, epsilon float64) (float64, CholeskyMetadata, error) {
	l, meta, err := SafeCholesky(cov, epsilon)
	if err != nil {
		return 0, meta, err
	}
	if len(diff) != len(l) {
		return 0, meta, fmt.Errorf("dimension mismatch: diff=%d cov=%d", len(diff), len(l))
	}

	y, err := forwardSubstitution(l, diff)
	if err != nil {
		meta.Finite = false
		return 0, meta, err
	}
	q := 0.0
	for i := range y {
		q += y[i] * y[i]
		if !isFinite(q) {
			meta.Finite = false
			return 0, meta, fmt.Errorf("non-finite Mahalanobis accumulator")
		}
	}
	if q < 0 {
		q = 0
	}
	meta.Finite = true
	return q, meta, nil
}

// MahalanobisDistance computes sqrt(d^T C^{-1} d) with finite guards.
// @ requires len(diff) > 0
// @ requires len(cov) == len(diff)
// @ requires epsilon >= 0
// @ ensures err == nil ==> result0 >= 0
// @ ensures err == nil ==> result1.Finite
func MahalanobisDistance(diff []float64, cov [][]float64, epsilon float64) (float64, CholeskyMetadata, error) {
	q, meta, err := MahalanobisSquared(diff, cov, epsilon)
	if err != nil {
		return 0, meta, err
	}
	d := stdmath.Sqrt(q)
	if !isFinite(d) {
		meta.Finite = false
		return 0, meta, fmt.Errorf("non-finite Mahalanobis distance")
	}
	meta.Finite = true
	return d, meta, nil
}

func sanitizeSymmetric(cov [][]float64) ([][]float64, float64, float64, bool) {
	n := len(cov)
	sym := make([][]float64, n)
	minDiag := stdmath.Inf(1)
	maxDiag := 0.0
	for i := 0; i < n; i++ {
		sym[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := 0.5 * (cov[i][j] + cov[j][i])
			if !isFinite(v) {
				return nil, 0, 0, false
			}
			sym[i][j] = v
		}
		d := sym[i][i]
		if d < minDiag {
			minDiag = d
		}
		ad := stdmath.Abs(d)
		if ad > maxDiag {
			maxDiag = ad
		}
	}
	return sym, minDiag, maxDiag, true
}

func addDiagonal(cov [][]float64, reg float64) [][]float64 {
	n := len(cov)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, n)
		copy(out[i], cov[i])
		out[i][i] += reg
	}
	return out
}

func factorLowerFinite(a [][]float64, minDiag float64) ([][]float64, float64, error) {
	n := len(a)
	l := make([][]float64, n)
	for i := 0; i < n; i++ {
		l[i] = make([]float64, n)
	}

	minAfter := stdmath.Inf(1)
	//@ decreases n
	for i := 0; i < n; i++ {
		//@ invariant i >= 0 && i <= n
		//@ decreases i
		for j := 0; j <= i; j++ {
			//@ invariant j >= 0 && j <= i+1
			sum := 0.0
			//@ decreases j
			for k := 0; k < j; k++ {
				//@ invariant k >= 0 && k <= j
				sum += l[i][k] * l[j][k]
				if !isFinite(sum) {
					return nil, 0, fmt.Errorf("non-finite accumulation at (%d,%d)", i, j)
				}
			}

			if i == j {
				v := a[i][i] - sum
				if !isFinite(v) {
					return nil, 0, fmt.Errorf("non-finite diagonal candidate")
				}
				if v <= minDiag {
					v = minDiag
				}
				d := stdmath.Sqrt(v)
				if !isFinite(d) || d <= 0 {
					return nil, 0, fmt.Errorf("invalid diagonal sqrt")
				}
				l[i][j] = d
				if d < minAfter {
					minAfter = d
				}
				continue
			}

			den := l[j][j]
			if den <= 0 || !isFinite(den) {
				return nil, 0, fmt.Errorf("invalid denominator at row %d", j)
			}
			v := (a[i][j] - sum) / den
			if !isFinite(v) {
				return nil, 0, fmt.Errorf("non-finite off-diagonal at (%d,%d)", i, j)
			}
			l[i][j] = v
		}
	}

	if minAfter <= 0 || !isFinite(minAfter) {
		return nil, 0, fmt.Errorf("invalid post-factorization diagonal")
	}
	return l, minAfter, nil
}

func forwardSubstitution(l [][]float64, b []float64) ([]float64, error) {
	n := len(l)
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := b[i]
		for j := 0; j < i; j++ {
			sum -= l[i][j] * x[j]
			if !isFinite(sum) {
				return nil, fmt.Errorf("non-finite forward substitution sum")
			}
		}
		den := l[i][i]
		if den <= 0 || !isFinite(den) {
			return nil, fmt.Errorf("non-positive triangular diagonal")
		}
		x[i] = sum / den
		if !isFinite(x[i]) {
			return nil, fmt.Errorf("non-finite forward substitution result")
		}
	}
	return x, nil
}

func conditionProxyFromDiag(l [][]float64) float64 {
	if len(l) == 0 {
		return 1
	}
	minD := stdmath.Inf(1)
	maxD := 0.0
	for i := 0; i < len(l); i++ {
		d := stdmath.Abs(l[i][i])
		if d < minD {
			minD = d
		}
		if d > maxD {
			maxD = d
		}
	}
	if minD <= 0 || !isFinite(minD) || !isFinite(maxD) {
		return stdmath.Inf(1)
	}
	return maxD / minD
}

func isFinite(v float64) bool {
	return !stdmath.IsNaN(v) && !stdmath.IsInf(v, 0)
}
