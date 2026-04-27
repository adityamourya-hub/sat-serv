package physics

import (
	"math"
	"testing"
)

func TestDeltaVResultComputeFuelCost(t *testing.T) {
	result := &DeltaVResult{DeltaVTotal: 100.0} // 100 m/s delta-v
	result.computeFuelCost()
	
	if result.FuelMassKg <= 0 {
		t.Errorf("Fuel mass should be positive, got %f", result.FuelMassKg)
	}
	
	if result.ElectricFuelKg <= 0 {
		t.Errorf("Electric fuel mass should be positive, got %f", result.ElectricFuelKg)
	}
	
	// Electric propulsion should use less fuel due to higher Isp
	if result.ElectricFuelKg >= result.FuelMassKg {
		t.Error("Electric propulsion should use less fuel than chemical")
	}
}

func TestDeltaVResultAssessAlertLevel(t *testing.T) {
	tests := []struct {
		deltaV   float64
		expected string
	}{
		{0.5, "GREEN"},
		{5.0, "YELLOW"},
		{50.0, "RED"},
		{150.0, "CRITICAL"},
	}
	
	for _, test := range tests {
		result := &DeltaVResult{DeltaVTotal: test.deltaV}
		result.assessAlertLevel()
		if result.AlertLevel != test.expected {
			t.Errorf("DeltaV %f: expected %s, got %s", test.deltaV, test.expected, result.AlertLevel)
		}
	}
}

func TestComputeFromStates(t *testing.T) {
	calc := NewDeltaVCalculator()
	calc.SetSpacecraftMass(1000.0)
	
	before := State{
		Pos: [3]float64{0, 0, 0},
		Vel: [3]float64{7.5, 0, 0}, // km/s
	}
	
	after := State{
		Pos: [3]float64{0, 0, 0},
		Vel: [3]float64{7.6, 0, 0}, // +100 m/s
	}
	
	result := calc.ComputeFromStates(before, after)
	
	if math.Abs(result.DeltaVTotal - 100.0) > 0.1 {
		t.Errorf("Expected ~100 m/s delta-v, got %f", result.DeltaVTotal)
	}
	
	if result.AlertLevel != "YELLOW" {
		t.Errorf("Expected YELLOW alert for 100 m/s, got %s", result.AlertLevel)
	}
}

func TestHamiltonianValidator(t *testing.T) {
	validator := NewHamiltonianValidator()
	
	state := State{
		Pos: [3]float64{7000, 0, 0},  // 7000 km altitude
		Vel: [3]float64{0, 7.5, 0},   // ~circular orbit velocity
	}
	
	H := validator.ComputeHamiltonian(state)
	if H >= 0 {
		t.Errorf("Bound orbit should have negative Hamiltonian, got %f", H)
	}
	
	hash, ratio := validator.ComputeInvariantHash(state)
	if hash == "" {
		t.Error("Hash should not be empty")
	}
	if ratio >= 1e-7 {
		t.Errorf("Hamiltonian ratio should be below threshold, got %e", ratio)
	}
	
	valid, reason := validator.ValidateObservation(state)
	if !valid {
		t.Errorf("Valid observation should pass: %s", reason)
	}
}
