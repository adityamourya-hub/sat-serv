package safemath

import (
	stdmath "math"
	"testing"
)

func TestSafeCholeskyNearSingularFinite(t *testing.T) {
	cov := [][]float64{
		{1.0, 0.999999999999, 0.999999999998},
		{0.999999999999, 0.999999999998, 0.999999999997},
		{0.999999999998, 0.999999999997, 0.999999999996},
	}
	l, meta, err := SafeCholesky(cov, 1e-12)
	if err != nil {
		t.Fatalf("safe cholesky failed: %v", err)
	}
	if !meta.Finite {
		t.Fatalf("expected finite metadata")
	}
	if len(l) != 3 || len(l[0]) != 3 {
		t.Fatalf("unexpected factor dimensions")
	}
	for i := 0; i < 3; i++ {
		if l[i][i] <= 0 || stdmath.IsNaN(l[i][i]) || stdmath.IsInf(l[i][i], 0) {
			t.Fatalf("invalid diagonal l[%d][%d]=%v", i, i, l[i][i])
		}
		for j := i + 1; j < 3; j++ {
			if stdmath.Abs(l[i][j]) > 0 {
				t.Fatalf("factor not lower triangular at (%d,%d)", i, j)
			}
		}
	}
}

func TestMahalanobisDistanceFiniteForDegenerateCov(t *testing.T) {
	diff := []float64{0.01, -0.02, 0.015}
	cov := [][]float64{
		{1e-16, 0, 0},
		{0, 1e-16, 0},
		{0, 0, 1e-16},
	}
	d, meta, err := MahalanobisDistance(diff, cov, 1e-12)
	if err != nil {
		t.Fatalf("mahalanobis distance failed: %v", err)
	}
	if !meta.Finite {
		t.Fatalf("expected finite metadata")
	}
	if stdmath.IsNaN(d) || stdmath.IsInf(d, 0) || d < 0 {
		t.Fatalf("invalid mahalanobis distance: %v", d)
	}
}

func TestSafeCholeskyRejectsNaNInput(t *testing.T) {
	cov := [][]float64{{stdmath.NaN(), 0}, {0, 1}}
	_, _, err := SafeCholesky(cov, 1e-12)
	if err == nil {
		t.Fatalf("expected error for NaN covariance")
	}
}

func TestMahalanobisAlignedVectorZeroEigenCovFinite(t *testing.T) {
	diff := []float64{1e-3, 1e-3, 1e-3}
	// Rank-1 covariance (two zero eigenvalues) emulates perfectly aligned-velocity degeneracy.
	cov := [][]float64{
		{1e-8, 1e-8, 1e-8},
		{1e-8, 1e-8, 1e-8},
		{1e-8, 1e-8, 1e-8},
	}
	d, meta, err := MahalanobisDistance(diff, cov, 1e-12)
	if err != nil {
		t.Fatalf("mahalanobis failed: %v", err)
	}
	if !meta.Finite {
		t.Fatalf("expected finite metadata")
	}
	if stdmath.IsNaN(d) || stdmath.IsInf(d, 0) || d < 0 {
		t.Fatalf("invalid mahalanobis distance: %v", d)
	}
}

func BenchmarkMahalanobisDistance3x3(b *testing.B) {
	diff := []float64{0.03, -0.01, 0.02}
	cov := [][]float64{
		{4e-3, 1e-4, 0},
		{1e-4, 3e-3, 5e-5},
		{0, 5e-5, 2e-3},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = MahalanobisDistance(diff, cov, 1e-12)
	}
}
