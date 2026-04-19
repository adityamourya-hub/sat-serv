package orbital

import (
	"ssis/pkg/physics"
	"testing"
)

func TestParseOEM(t *testing.T) {
	oemData := `CCSDS_OEM_VERS = 2.0
OBJECT_NAME = SATELLITE_A
OBJECT_ID = 12345
REF_FRAME = EME2000
START_TIME = 2024-03-15T00:00:00.000
STOP_TIME = 2024-03-15T00:10:00.000
META_STOP
2024-03-15T00:00:00.000  7000.000  0.000  0.000  0.000  7.500  0.000
2024-03-15T00:01:00.000  6500.000  1000.000  0.000  1.000  7.000  0.000
2024-03-15T00:05:00.000  6000.000  2000.000  0.000  2.000  6.500  0.000
2024-03-15T00:10:00.000  5000.000  3000.000  0.000  3.000  6.000  0.000
`
	oem, err := ParseOEM(oemData)
	if err != nil {
		t.Fatalf("ParseOEM failed: %v", err)
	}

	if oem.ObjectName != "SATELLITE_A" {
		t.Errorf("Expected OBJECT_NAME SATELLITE_A, got %s", oem.ObjectName)
	}
	if len(oem.Data) != 4 {
		t.Errorf("Expected 4 data points, got %d", len(oem.Data))
	}
}

func TestOEMPropagation(t *testing.T) {
	oemData := `CCSDS_OEM_VERS = 2.0
START_TIME = 2024-03-15T00:00:00.000
STOP_TIME = 2024-03-15T00:10:00.000
META_STOP
2024-03-15T00:00:00.000  7000.000  0.000  0.000  0.000  7.500  0.000
2024-03-15T00:02:00.000  6800.000  1500.000  0.000  -1.50   7.200  0.000
2024-03-15T00:04:00.000  6500.000  2800.000  0.000  -2.50   6.800  0.000
2024-03-15T00:06:00.000  6100.000  3900.000  0.000  -3.50   6.000  0.000
2024-03-15T00:08:00.000  5500.000  4800.000  0.000  -4.50   5.100  0.000
2024-03-15T00:10:00.000  4800.000  5500.000  0.000  -5.50   4.200  0.000
`
	oem, _ := ParseOEM(oemData)
	p, err := NewOEMPropagator(oem)
	if err != nil {
		t.Fatalf("NewOEMPropagator failed: %v", err)
	}

	// Test exact point (2 mins = 120s)
	state, err := p.Propagate(2.0)
	if err != nil {
		t.Errorf("Propagation failed: %v", err)
	}
	if state.Pos[0] != 6800.0 {
		t.Errorf("Expected X=6800.0 at exact point, got %f", state.Pos[0])
	}

	// Test interpolation (1 min = 60s, mid-point between 0 and 2)
	stateMid, _ := p.Propagate(1.0)
	if stateMid.Pos[0] <= 6800.0 || stateMid.Pos[0] >= 7000.0 {
		t.Errorf("Mid-point X %.2f outside expected linear range (6800, 7000)", stateMid.Pos[0])
	}

	// Test range error
	_, err = p.Propagate(15.0)
	if err == nil {
		t.Errorf("Expected error for out-of-range propagation, got nil")
	}
}

func TestLagrangeInterpolationAccuracy(t *testing.T) {
	// Simple linear test: y = 2x
	points := []DataPoint{
		{Time: 0, State: physics.State{Pos: physics.Vector3{0, 0, 0}}},
		{Time: 1, State: physics.State{Pos: physics.Vector3{2, 0, 0}}},
		{Time: 2, State: physics.State{Pos: physics.Vector3{4, 0, 0}}},
		{Time: 3, State: physics.State{Pos: physics.Vector3{6, 0, 0}}},
	}

	res := LagrangeInterpolate(points, 1.5)
	if res.Pos[0] != 3.0 {
		t.Errorf("LagrangeInterpolate failed on linear data: expected 3.0, got %f", res.Pos[0])
	}
}
