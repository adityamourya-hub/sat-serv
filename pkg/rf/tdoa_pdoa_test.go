package rf

/**
 * TDOA-PDOA Localization Tests
 *
 * Validates implementation of Chen et al. IEEE WCL 2020 (WCL2020-0388)
 */

import (
	"math"
	"testing"
)

// TestTDOAPDOALocalization validates joint TDOA-PDOA algorithm
func TestTDOAPDOALocalization(t *testing.T) {
	// Create 4 ground stations in a square configuration
	stations := []RFStation{
		{ID: "S1", Position: [3]float64{0, 0, 0}},      // Origin
		{ID: "S2", Position: [3]float64{1000, 0, 0}},  // East
		{ID: "S3", Position: [3]float64{0, 1000, 0}},  // North
		{ID: "S4", Position: [3]float64{1000, 1000, 0}}, // NE
	}
	
	// Target position (to be estimated)
	target := [3]float64{400, 600, 800} // km
	
	// Generate TDOA measurements
	tdoa := make([][]float64, 4)
	pdoa := make([][]float64, 4)
	tdoaCov := make([][]float64, 4)
	
	for i := range tdoa {
		tdoa[i] = make([]float64, 4)
		pdoa[i] = make([]float64, 4)
		tdoaCov[i] = make([]float64, 4)
	}
	
	c := 299792.458 // km/s
	lambda := 0.3   // 1 GHz -> 0.3m = 0.0003 km
	
	// Compute true TDOA and PDOA
	for i := 0; i < 4; i++ {
		for j := i+1; j < 4; j++ {
			di := distance(target, stations[i].Position)
			dj := distance(target, stations[j].Position)
			
			// TDOA
			tdoa[i][j] = (di - dj) / c
			tdoa[j][i] = -tdoa[i][j]
			
			// PDOA
			pdoa[i][j] = 2.0 * math.Pi / lambda * (di - dj)
			pdoa[j][i] = -pdoa[i][j]
			
			// Measurement uncertainty
			tdoaCov[i][j] = 1e-12 // 1ns uncertainty squared
			tdoaCov[j][i] = 1e-12
		}
	}
	
	measurement := TDOAPDOAMeasurement{
		TDOA:             tdoa,
		PDOA:             pdoa,
		TDOACovariance:   tdoaCov,
		PDOAVariance:     0.01,
		CarrierFrequency: 1e9, // 1 GHz
		Wavelength:       lambda * 1000, // Convert to meters
	}
	
	// Create localizer and estimate position
	localizer := NewTDOAPDOALocalizer(stations)
	result := localizer.Localize(measurement)
	
	// Validate results
	error := distance(result.Position, target)
	t.Logf("Position error: %.3f km", error)
	t.Logf("TDOA residual: %.6f", result.TDOResidual)
	t.Logf("PDOA residual: %.6f", result.PDOResidual)
	t.Logf("Converged: %v in %d iterations", result.Converged, result.Iterations)
	
	// Check convergence
	if !result.Converged {
		t.Error("Algorithm did not converge")
	}
	
	// Position should be reasonably accurate (within 100m for this synthetic test)
	if error > 0.1 {
		t.Errorf("Position error too large: %.3f km (expected < 0.1 km)", error)
	}
}

// TestTDONlyvsTDOAPDOA compares accuracy
func TestTDONlyvsTDOAPDOA(t *testing.T) {
	stations := []RFStation{
		{ID: "S1", Position: [3]float64{0, 0, 0}},
		{ID: "S2", Position: [3]float64{1000, 0, 0}},
		{ID: "S3", Position: [3]float64{0, 1000, 0}},
		{ID: "S4", Position: [3]float64{1000, 1000, 0}},
	}
	
	target := [3]float64{500, 500, 500}
	
	// Generate measurements with noise
	tdoa := make([][]float64, 4)
	pdoa := make([][]float64, 4)
	tdoaCov := make([][]float64, 4)
	
	for i := range tdoa {
		tdoa[i] = make([]float64, 4)
		pdoa[i] = make([]float64, 4)
		tdoaCov[i] = make([]float64, 4)
	}
	
	c := 299792.458
	lambda := 0.0003 // km (1 GHz)
	
	for i := 0; i < 4; i++ {
		for j := i+1; j < 4; j++ {
			di := distance(target, stations[i].Position)
			dj := distance(target, stations[j].Position)
			
			tdoa[i][j] = (di-dj)/c + 1e-9*randNorm() // Add noise
			pdoa[i][j] = 2*math.Pi/lambda*(di-dj) + 0.01*randNorm()
			tdoaCov[i][j] = 1e-12
		}
	}
	
	measurement := TDOAPDOAMeasurement{
		TDOA:             tdoa,
		PDOA:             pdoa,
		TDOACovariance:   tdoaCov,
		PDOAVariance:     0.01,
		CarrierFrequency: 1e9,
		Wavelength:       0.3,
	}
	
	// Test with different alpha/beta weights
	weights := []struct {
		alpha, beta float64
		name       string
	}{
		{1.0, 0.0, "TDOA-only"},
		{0.0, 1.0, "PDOA-only"},
		{0.7, 0.3, "Joint (Chen 2020)"},
		{0.5, 0.5, "Equal weight"},
	}
	
	for _, w := range weights {
		localizer := NewTDOAPDOALocalizer(stations)
		localizer.Alpha = w.alpha
		localizer.Beta = w.beta
		
		result := localizer.Localize(measurement)
		error := distance(result.Position, target)
		
		t.Logf("%s: Error=%.3f km, Residual=%.4f", w.name, error, result.TotalResidual)
	}
}

// BenchmarkTDOAPDOA performance test
func BenchmarkTDOAPDOALocalization(b *testing.B) {
	stations := []RFStation{
		{ID: "S1", Position: [3]float64{0, 0, 0}},
		{ID: "S2", Position: [3]float64{1000, 0, 0}},
		{ID: "S3", Position: [3]float64{0, 1000, 0}},
		{ID: "S4", Position: [3]float64{1000, 1000, 0}},
	}
	
	target := [3]float64{400, 600, 800}
	
	// Pre-generate measurements
	tdoa := make([][]float64, 4)
	pdoa := make([][]float64, 4)
	tdoaCov := make([][]float64, 4)
	
	for i := range tdoa {
		tdoa[i] = make([]float64, 4)
		pdoa[i] = make([]float64, 4)
		tdoaCov[i] = make([]float64, 4)
	}
	
	c := 299792.458
	lambda := 0.0003
	
	for i := 0; i < 4; i++ {
		for j := i+1; j < 4; j++ {
			di := distance(target, stations[i].Position)
			dj := distance(target, stations[j].Position)
			tdoa[i][j] = (di - dj) / c
			pdoa[i][j] = 2 * math.Pi / lambda * (di - dj)
			tdoaCov[i][j] = 1e-12
		}
	}
	
	measurement := TDOAPDOAMeasurement{
		TDOA:           tdoa,
		PDOA:           pdoa,
		TDOACovariance: tdoaCov,
		PDOAVariance:   0.01,
		Wavelength:     0.3,
	}
	
	localizer := NewTDOAPDOALocalizer(stations)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = localizer.Localize(measurement)
	}
}

// Helper for normal distribution (simplified)
func randNorm() float64 {
	return (float64(1) - 0.5) * 2.0 // Simplified - replace with proper rand
}
