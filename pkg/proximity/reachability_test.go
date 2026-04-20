package proximity

import (
	"math"
	"testing"

	"ssis/pkg/physics"

	"gonum.org/v1/gonum/mat"
)

func TestCalculateRRI_SingularMatrixHandling(t *testing.T) {
	// Test with zero delta-V which should cause singular matrix
	asset := physics.State{
		Pos: physics.Vector3{7000, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	threat := physics.State{
		Pos: physics.Vector3{7100, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}

	_, err := CalculateRRI(asset, threat, 0.0, 1000.0)
	if err == nil {
		t.Errorf("Expected error for zero delta-V (singular matrix), got nil")
	}
}

func TestCalculateRRI_InvalidInputs(t *testing.T) {
	asset := physics.State{
		Pos: physics.Vector3{7000, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	threat := physics.State{
		Pos: physics.Vector3{7100, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}

	// Test negative horizon
	_, err := CalculateRRI(asset, threat, 10.0, -100.0)
	if err == nil {
		t.Errorf("Expected error for negative horizon, got nil")
	}

	// Test NaN position
	assetNaN := physics.State{
		Pos: physics.Vector3{math.NaN(), 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	_, err = CalculateRRI(assetNaN, threat, 10.0, 1000.0)
	if err == nil {
		t.Errorf("Expected error for NaN position, got nil")
	}

	// Test Inf position
	assetInf := physics.State{
		Pos: physics.Vector3{math.Inf(1), 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	_, err = CalculateRRI(assetInf, threat, 10.0, 1000.0)
	if err == nil {
		t.Errorf("Expected error for Inf position, got nil")
	}
}

func TestCalculateRRI_BoundaryConditions(t *testing.T) {
	cfg := GetConfig()
	asset := physics.State{
		Pos: physics.Vector3{7000, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	threat := physics.State{
		Pos: physics.Vector3{7100, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}

	// Test horizon exactly at CW threshold
	rriCW, err := CalculateRRI(asset, threat, 10.0, cfg.CWHorizonThresholdSec-1.0)
	if err != nil {
		t.Errorf("Unexpected error at CW threshold-1: %v", err)
	}
	if rriCW == nil {
		t.Error("Expected RRI at CW threshold-1, got nil")
	}

	// Test horizon just above CW threshold
	rriSTM, err := CalculateRRI(asset, threat, 10.0, cfg.CWHorizonThresholdSec+1.0)
	if err != nil {
		t.Errorf("Unexpected error at CW threshold+1: %v", err)
	}
	if rriSTM == nil {
		t.Error("Expected RRI at CW threshold+1, got nil")
	}
}

func TestCalculateRRI_DeterminantEdgeCases(t *testing.T) {
	asset := physics.State{
		Pos: physics.Vector3{7000, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}
	threat := physics.State{
		Pos: physics.Vector3{7100, 0, 0},
		Vel: physics.Vector3{0, 7.5, 0},
	}

	// Test with very small delta-V (near-singular matrix)
	_, err := CalculateRRI(asset, threat, 0.001, 1000.0)
	if err != nil {
		// This is expected for near-singular matrices
		t.Logf("Expected error for near-singular matrix: %v", err)
	}

	// Test with very large delta-V
	rri, err := CalculateRRI(asset, threat, 1000.0, 1000.0)
	if err != nil {
		t.Errorf("Unexpected error for large delta-V: %v", err)
	}
	if rri == nil {
		t.Error("Expected RRI for large delta-V, got nil")
	}
}

func TestRRI_Intersects(t *testing.T) {
	// Create a simple ellipsoid centered at origin
	matrix := mat.NewSymDense(3, []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})
	ellipsoid := &ReachabilityEllipsoid{
		Center: [3]float64{0, 0, 0},
		Matrix: matrix,
	}
	rri := &RRI{
		ObjectID:   1,
		Ellipsoid:  ellipsoid,
		HorizonSec: 1000.0,
	}

	// Test point inside ellipsoid
	if !rri.Intersects([3]float64{0.5, 0.0, 0.0}) {
		t.Error("Expected point inside ellipsoid to intersect")
	}

	// Test point on ellipsoid boundary
	if !rri.Intersects([3]float64{1.0, 0.0, 0.0}) {
		t.Error("Expected point on ellipsoid boundary to intersect")
	}

	// Test point outside ellipsoid
	if rri.Intersects([3]float64{2.0, 0.0, 0.0}) {
		t.Error("Expected point outside ellipsoid to not intersect")
	}
}

func TestDeterminant3x3(t *testing.T) {
	// Test identity matrix
	identity := mat.NewSymDense(3, []float64{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0})
	det := Determinant3x3(identity)
	if math.Abs(det-1.0) > 1e-10 {
		t.Errorf("Expected determinant 1.0 for identity matrix, got %f", det)
	}

	// Test diagonal matrix
	diag := mat.NewSymDense(3, []float64{2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0})
	det = Determinant3x3(diag)
	if math.Abs(det-24.0) > 1e-10 {
		t.Errorf("Expected determinant 24.0 for diagonal matrix, got %f", det)
	}

	// Test singular matrix
	singular := mat.NewSymDense(3, []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0})
	det = Determinant3x3(singular)
	if math.Abs(det) > 1e-10 {
		t.Errorf("Expected determinant near 0 for singular matrix, got %f", det)
	}
}
