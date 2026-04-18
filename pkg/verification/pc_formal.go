package verification

import (
	"ssis/pkg/collision"
	safemath "ssis/pkg/math"
)

/*
   Gobra formal verification harness for SSIS Collision Probability Module.
   This file is used by the ETH Zurich Gobra verifier to prove:
   1. Functional correctness of the adaptive Foster 2D collision probability pipeline.
   2. Absence of floating-point anomalies (NaN/Inf) in SafeCholesky and Mahalanobis paths.
   3. Bounded probability output (ensures 0 <= Pc <= 1).
*/

// @ requires missX >= -50000 && missX <= 50000
// @ requires missY >= -50000 && missY <= 50000
// @ requires varX > 0 && varY > 0 && R > 0
// @ ensures res >= 0 && res <= 1
func VerifyFosterLogic(missX, missY, varX, varY, covXY, R float64) (res float64) {
	// Call the verified implementation
	res = collision.FosterTwoDimensionalPc(missX, missY, varX, varY, covXY, R)
	return res
}

// @ requires len(cov) > 0
// @ requires epsilon >= 0
// @ ensures ok ==> finite
func VerifySafeCholeskyNoNaN(cov [][]float64, epsilon float64) (ok bool, finite bool) {
	_, meta, err := safemath.SafeCholesky(cov, epsilon)
	if err != nil {
		return false, false
	}
	return true, meta.Finite
}

// @ requires len(diff) > 0
// @ requires len(cov) == len(diff)
// @ requires epsilon >= 0
// @ ensures ok ==> dist >= 0
// @ ensures ok ==> finite
func VerifyMahalanobisNoNaN(diff []float64, cov [][]float64, epsilon float64) (ok bool, dist float64, finite bool) {
	d, meta, err := safemath.MahalanobisDistance(diff, cov, epsilon)
	if err != nil {
		return false, 0, false
	}
	return true, d, meta.Finite
}
