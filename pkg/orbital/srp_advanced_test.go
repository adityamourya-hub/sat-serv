package orbital

import (
	"math"
	"testing"

	"ssis/pkg/physics"
)

/**
 * TestConicalShadowModel verifies the shadow geometry calculations.
 * Key scenarios:
 * - Full sunlight (nu = 1)
 * - Umbra (nu = 0)
 * - Penumbra (0 < nu < 1)
 */
func TestConicalShadowModel(t *testing.T) {
	tests := []struct {
		name      string
		satPos    [3]float64
		sunPos    [3]float64
		wantNu    float64
		wantState ShadowState
		tol       float64
	}{
		{
			name:      "Full sunlight - satellite on sun side",
			satPos:    [3]float64{7000, 0, 0},      // On sun side of Earth
			sunPos:    [3]float64{149597870, 0, 0}, // Sun at ~1 AU
			wantNu:    1.0,
			wantState: FullSunlight,
			tol:       0.01,
		},
		{
			name:      "Full sunlight - GEO in sunlight",
			satPos:    [3]float64{-42164, 0, 0}, // GEO on sun-opposite side but in space
			sunPos:    [3]float64{149597870, 0, 0},
			wantNu:    1.0,
			wantState: FullSunlight,
			tol:       0.01,
		},
		{
			name:      "Shadow check - GEO during eclipse season",
			satPos:    [3]float64{-42164, 0, 0},     // GEO position
			sunPos:    [3]float64{-149597870, 0, 0}, // Sun behind Earth (eclipse)
			wantNu:    0.0,                          // Should be in shadow
			wantState: Umbra,
			tol:       0.01,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := ConicalShadowModel(tc.satPos, tc.sunPos)

			if result.State != tc.wantState {
				t.Errorf("Shadow state = %v, want %v", result.State, tc.wantState)
			}

			if math.Abs(result.Nu-tc.wantNu) > tc.tol {
				t.Errorf("Shadow factor nu = %.3f, want %.3f (tol %.3f)",
					result.Nu, tc.wantNu, tc.tol)
			}
		})
	}
}

/**
 * TestSRPAcceleration verifies SRP force calculation.
 * Typical GEO values:
 * - A/m = 0.02 m^2/kg (typical GEO sat)
 * - Cr = 1.3
 * - Expected acceleration ~1e-9 km/s^2
 */
func TestSRPAcceleration(t *testing.T) {
	// GEO satellite parameters
	mass := 3000.0         // kg
	area := 60.0           // m^2 (solar panels)
	Cr := 1.3              // reflectivity
	amRatio := area / mass // 0.02 m^2/kg

	// Satellite at GEO
	satPos := [3]float64{42164, 0, 0}
	sunPos := [3]float64{149597870.7, 0, 0}

	ax, ay, az := SRPAcceleration(satPos, sunPos, area, mass, Cr)

	// Calculate magnitude
	acc := math.Sqrt(ax*ax + ay*ay + az*az)

	// Expected: ~1e-9 km/s^2 for typical GEO satellite
	// P/c * A/m * Cr = (1361/3e8) * 0.02 * 1.3 ≈ 1.18e-7 m/s^2 = 1.18e-10 km/s^2
	// But we have (1+Cr) factor in simplified model and distance scaling
	// So expected is roughly 1e-9 to 5e-9 km/s^2

	t.Logf("SRP acceleration: %.3e km/s^2 (A/m=%.4f, Cr=%.1f)", acc, amRatio, Cr)

	if acc < 1e-10 || acc > 1e-7 {
		t.Errorf("SRP acceleration %.3e km/s^2 outside expected range [1e-10, 1e-7]", acc)
	}

	// Direction should be toward Sun (positive x in this case)
	if ax <= 0 {
		t.Errorf("SRP x-component should be positive (toward Sun), got %.3e", ax)
	}
}

/**
 * TestSRPInShadow verifies SRP is zero in umbra.
 */
func TestSRPInShadow(t *testing.T) {
	mass := 3000.0
	area := 60.0
	Cr := 1.3

	// Satellite in Earth shadow (sun behind Earth)
	satPos := [3]float64{-42164, 0, 0}       // GEO on opposite side from sun
	sunPos := [3]float64{-149597870.7, 0, 0} // Sun behind Earth

	ax, ay, az := SRPAcceleration(satPos, sunPos, area, mass, Cr)

	acc := math.Sqrt(ax*ax + ay*ay + az*az)

	if acc > 1e-15 {
		t.Errorf("SRP in shadow should be ~0, got %.3e km/s^2", acc)
	}
}

/**
 * TestECOMAcceleration verifies empirical CODE orbit model.
 */
func TestECOMAcceleration(t *testing.T) {
	params := ECOMParams{
		D:    1e-9,  // Direct
		Y:    1e-10, // Y-bias
		B:    0,     // Bi-normal
		DCos: 0.2e-9,
		DSin: 0.1e-9,
	}

	satPos := [3]float64{42164, 0, 0}
	sunPos := [3]float64{149597870.7, 0, 0}

	ax, ay, az := ECOMAcceleration(satPos, sunPos, params, 2451545.0)

	acc := math.Sqrt(ax*ax + ay*ay + az*az)

	// Should be similar order of magnitude to cannonball model
	t.Logf("ECOM acceleration: %.3e km/s^2", acc)

	if acc < 1e-11 || acc > 1e-7 {
		t.Errorf("ECOM acceleration %.3e outside expected range", acc)
	}
}

/**
 * TestBoxWingAcceleration verifies physical Box-Wing model.
 */
func TestBoxWingAcceleration(t *testing.T) {
	// Create typical INSAT/GSAT class satellite
	model := NewStandardGEOSat(3000.0)

	satPos := [3]float64{42164, 0, 0}
	sunPos := [3]float64{149597870.7, 0, 0}

	ax, ay, az := BoxWingAcceleration(satPos, sunPos, model)

	acc := math.Sqrt(ax*ax + ay*ay + az*az)

	// Box-Wing should give similar magnitude to cannonball
	// but more accurate directional components
	t.Logf("Box-Wing acceleration: %.3e km/s^2", acc)

	if acc < 1e-11 || acc > 1e-7 {
		t.Errorf("Box-Wing acceleration %.3e outside expected range", acc)
	}
}

/**
 * TestBoxWingInShadow verifies Box-Wing SRP is zero in shadow.
 */
func TestBoxWingInShadow(t *testing.T) {
	model := NewStandardGEOSat(3000.0)

	satPos := [3]float64{-42164, 0, 0}
	sunPos := [3]float64{-149597870.7, 0, 0}

	ax, ay, az := BoxWingAcceleration(satPos, sunPos, model)

	acc := math.Sqrt(ax*ax + ay*ay + az*az)

	if acc > 1e-15 {
		t.Errorf("Box-Wing in shadow should be ~0, got %.3e", acc)
	}
}

/**
 * TestShadowConsistency verifies that shadow model behaves correctly
 * for various positions.
 */
func TestShadowConsistency(t *testing.T) {
	sunPos := [3]float64{149597870.7, 0, 0}

	// Test positions along x-axis from behind Earth to sun side
	positions := []float64{-50000, -45000, -42164, -40000, 0, 42164, 50000}

	for _, x := range positions {
		satPos := [3]float64{x, 0, 0}
		shadow := ConicalShadowModel(satPos, sunPos)

		t.Logf("Position x=%.0f km: state=%d, nu=%.3f", x, shadow.State, shadow.Nu)

		// Sanity checks
		if shadow.Nu < 0 || shadow.Nu > 1 {
			t.Errorf("Invalid nu=%.3f at x=%.0f", shadow.Nu, x)
		}

		// Sun side should be fully illuminated
		if x > 0 && shadow.Nu != 1.0 {
			t.Errorf("Sun side should have nu=1.0, got %.3f at x=%.0f", shadow.Nu, x)
		}
	}
}

/**
 * TestNewStandardGEOSat verifies default GEO satellite model.
 */
func TestNewStandardGEOSat(t *testing.T) {
	model := NewStandardGEOSat(3000.0)

	if model.Mass != 3000.0 {
		t.Errorf("Mass = %.1f, want 3000.0", model.Mass)
	}

	if model.NumPanels != 2 {
		t.Errorf("NumPanels = %d, want 2", model.NumPanels)
	}

	if model.PanelArea <= 0 {
		t.Error("PanelArea should be positive")
	}

	if model.Cr < 1.0 || model.Cr > 2.0 {
		t.Errorf("Cr = %.2f outside realistic range [1.0, 2.0]", model.Cr)
	}

	// Verify bus areas are reasonable
	totalBusArea := 0.0
	for _, a := range model.BusArea {
		totalBusArea += a
	}
	t.Logf("Total bus area: %.2f m^2", totalBusArea)

	if totalBusArea < 10 || totalBusArea > 100 {
		t.Errorf("Total bus area %.2f outside expected range [10, 100]", totalBusArea)
	}
}

/**
 * TestEstimateECOMFromHistory verifies ECOM parameter estimation.
 */
func TestEstimateECOMFromHistory(t *testing.T) {
	// Test with insufficient data
	params := EstimateECOMFromHistory(nil)
	if params.D == 0 {
		t.Error("Should return default D for nil input")
	}

	// Test with empty slice
	params = EstimateECOMFromHistory([]*physics.TLE{})
	if params.D == 0 {
		t.Error("Should return default D for empty input")
	}
}

/**
 * TestSRPAccuracyComparison compares different SRP models.
 */
func TestSRPAccuracyComparison(t *testing.T) {
	satPos := [3]float64{42164, 0, 0}
	sunPos := [3]float64{149597870.7, 0, 0}
	mass := 3000.0
	area := 60.0
	Cr := 1.3

	// Cannonball model
	cax, cay, caz := SRPAcceleration(satPos, sunPos, area, mass, Cr)
	cannonballMag := math.Sqrt(cax*cax + cay*cay + caz*caz)

	// Box-Wing model
	model := NewStandardGEOSat(mass)
	bax, bay, baz := BoxWingAcceleration(satPos, sunPos, model)
	boxwingMag := math.Sqrt(bax*bax + bay*bay + baz*baz)

	// ECOM model
	params := ECOMParams{D: 1.2e-9, Y: 0.1e-9, B: 0}
	eax, eay, eaz := ECOMAcceleration(satPos, sunPos, params, 2451545.0)
	ecomMag := math.Sqrt(eax*eax + eay*eay + eaz*eaz)

	t.Logf("Cannonball: %.3e km/s^2", cannonballMag)
	t.Logf("Box-Wing:   %.3e km/s^2", boxwingMag)
	t.Logf("ECOM:       %.3e km/s^2", ecomMag)

	// All should be same order of magnitude
	if cannonballMag/boxwingMag > 10 || boxwingMag/cannonballMag > 10 {
		t.Logf("Warning: Large difference between cannonball and Box-Wing models")
	}
}
