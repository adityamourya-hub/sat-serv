package collision

import (
	"math"
	"ssis/pkg/octree"
	"testing"
)

func TestConvexHull2D(t *testing.T) {
	points := []Vector2{
		{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1},
	}
	hull := ConvexHull2D(points)
	if len(hull) != 4 {
		t.Fatalf("Expected 4 points in hull, got %d", len(hull))
	}
}

func TestMinkowskiSum2D(t *testing.T) {
	// Two 1x1 squares
	h1 := []Vector2{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	h2 := []Vector2{{0, 0}, {1, 0}, {1, 1}, {0, 1}}

	sum := MinkowskiSum2D(h1, h2)
	// Minkowski sum of two 1x1 squares should be a 2x2 square
	// Vertices: (0,0), (2,0), (2,2), (0,2)
	if len(sum) != 4 {
		t.Fatalf("Expected 4 points in Minkowski sum hull, got %d", len(sum))
	}
	
	foundMax := false
	for _, v := range sum {
		if v.X == 2 && v.Y == 2 {
			foundMax = true
			break
		}
	}
	if !foundMax {
		t.Errorf("Minkowski sum did not result in expected (2,2) vertex")
	}
}

func TestMinkowskiSumGeometricPc(t *testing.T) {
	// Test case: Concentric collision of two sphere-like small cubes
	// Should approximate spherical Pc
	hull1 := EnvelopeCubeSat
	hull2 := EnvelopeCubeSat
	
	vRel := octree.Vector3{X: 0, Y: 7.5, Z: 0}
	
	// Centered collision (no miss distance)
	missR, missI := 0.0, 0.0
	// 10m sigma
	sigmaR, sigmaI, sigmaRI := 0.01*0.01, 0.01*0.01, 0.0
	
	pc, err := MinkowskiSumGeometricPc(missR, missI, sigmaR, sigmaI, sigmaRI, hull1, hull2, vRel)
	if err != nil {
		t.Fatalf("Minkowski Pc failed: %v", err)
	}
	
	if pc <= 0 || pc > 1 {
		t.Errorf("Invalid Pc value: %v", pc)
	}

	// Parity check with spherical Foster
	// CubeSat sum is 0.2m x 0.2m x 0.2m sum -> 0.4m x 0.4m x 0.4m
	// Projected area is 0.4m * 0.4m = 0.16 m^2 = 1.6e-7 km^2
	areaKm2 := 0.4 * 0.4 * 1e-6
	R_equiv := math.Sqrt(areaKm2 / math.Pi)
	pcSpherical := AdaptiveFosterPc(missR, missI, sigmaR, sigmaI, sigmaRI, R_equiv)
	
	diff := math.Abs(pc - pcSpherical)
	if diff > 1e-4 {
		t.Errorf("Significant divergence from spherical parity: Minkowski=%.8f, Spherical=%.8f (diff=%.8f)", pc, pcSpherical, diff)
	}
}

func TestIntegrateGaussianOverPolygon(t *testing.T) {
	// Integrate N(0, 1) over a large square covering several sigmas
	// Should be close to 1.0
	sx, sy := 1.0, 1.0
	mx, my := 0.0, 0.0
	
	largeSquare := []Vector2{
		{-10, -10}, {10, -10}, {10, 10}, {-10, 10},
	}
	
	pc := integrateGaussianOverPolygon(largeSquare, mx, my, sx, sy)
	if math.Abs(pc-1.0) > 1e-4 {
		t.Errorf("Expected integral over large domain to be ~1.0, got %.6f", pc)
	}
}
