package collision

import (
	"math"
	"os"
	"testing"
)

func TestAdaptiveFosterPcGaussConvergesToHighOrderReference(t *testing.T) {
	missR := 0.035
	missI := -0.012
	varR := 1.8e-4
	varI := 2.1e-4
	covRI := 1.7e-5
	R := 0.005

	pc, meta := AdaptiveFosterPcWithMetadata(missR, missI, varR, varI, covRI, R, 7.8)
	if meta.Method != "foster_1992_adaptive_gauss_legendre" {
		t.Fatalf("expected adaptive gauss path, got %s", meta.Method)
	}

	xm, ym, lMax, lMin, ok := principalFrame(missR, missI, varR, varI, covRI)
	if !ok {
		t.Fatalf("principal frame failed")
	}
	nodes, weights := gaussLegendreRule(512)
	ref := integratePcPolar(xm, ym, lMax, lMin, R, nodes, weights)
	if math.Abs(pc-ref) > 1e-12 {
		t.Fatalf("adaptive gauss mismatch: got %.16e ref %.16e", pc, ref)
	}
	// Signature is expected when SSIS_SIGNING_KEY is configured;
	// skip check in test environments without signing infrastructure
	if meta.Signature == "" && os.Getenv("SSIS_SIGNING_KEY") != "" {
		t.Fatalf("expected signed metadata for adaptive Pc result when SSIS_SIGNING_KEY is set")
	}
}

func TestAdaptiveFosterPcFallbackAspectRatio(t *testing.T) {
	missR := 0.052
	missI := 0.001
	varR := 2.0e-3
	varI := 1.0e-6
	covRI := 0.0
	R := 0.005

	pc, meta := AdaptiveFosterPcWithMetadata(missR, missI, varR, varI, covRI, R, 10.0)
	if pc < 0 || pc > 1 {
		t.Fatalf("pc out of range: %f", pc)
	}
	if meta.Method != "alfano_2005_single_integral_series" {
		t.Fatalf("expected alfano fallback, got %s", meta.Method)
	}
	if meta.AspectRatio <= 500 {
		t.Fatalf("expected extreme aspect ratio, got %f", meta.AspectRatio)
	}
}

func TestAdaptiveFosterPcFallbackVelocity(t *testing.T) {
	pc, meta := AdaptiveFosterPcWithMetadata(
		0.04,
		-0.01,
		2.4e-4,
		2.2e-4,
		1e-5,
		0.005,
		12.5,
	)
	if pc < 0 || pc > 1 {
		t.Fatalf("pc out of range: %f", pc)
	}
	if meta.Method != "alfano_2005_single_integral_series" {
		t.Fatalf("expected velocity-triggered alfano fallback, got %s", meta.Method)
	}
}

func TestFosterAliasCompatibility(t *testing.T) {
	missR := 0.03
	missI := -0.02
	varR := 1.5e-4
	varI := 2.7e-4
	covRI := -2.0e-5
	R := 0.005

	legacy := FosterTwoDimensionalPc(missR, missI, varR, varI, covRI, R)
	adaptive := AdaptiveFosterPc(missR, missI, varR, varI, covRI, R)
	if math.Abs(legacy-adaptive) > 1e-15 {
		t.Fatalf("deprecated alias diverged: legacy %.16e adaptive %.16e", legacy, adaptive)
	}
}

func BenchmarkAdaptiveFosterPcAlfanoPath(b *testing.B) {
	missR := 0.052
	missI := 0.001
	varR := 2.0e-3
	varI := 1.0e-6
	R := 0.005

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _ = alfanoSeriesPc(missR, missI, varR, varI, R)
	}
}

func BenchmarkAdaptiveFosterPcGaussPath(b *testing.B) {
	missR := 0.035
	missI := -0.012
	varR := 1.8e-4
	varI := 2.1e-4
	covRI := 1.7e-5
	R := 0.005

	xm, ym, lMax, lMin, ok := principalFrame(missR, missI, varR, varI, covRI)
	if !ok {
		b.Fatal("principal frame failed")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _ = adaptiveGaussLegendrePc(xm, ym, lMax, lMin, R)
	}
}
