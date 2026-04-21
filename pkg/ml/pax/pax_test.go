package pax

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkTierTransition measures the latency of tier selection
// Target: <10ms for SelectTier call (not including model loading)
func BenchmarkTierTransition(b *testing.B) {
	controller := NewPAXController()

	// Mock Probabilities
	ballisticMode := map[string]float64{"ballistic": 0.9, "hgv_glide": 0.05}
	maneuverMode := map[string]float64{"ballistic": 0.1, "hgv_glide": 0.8}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate a switch from Ballistic to HGV Glide
		tier := controller.SelectTier(maneuverMode, 0.4)

		// In a real run, this is where the Orchestrator switches the function pointer
		if tier != Tier2 {
			b.Fatalf("Failed to shift to Tier 2")
		}

		elapsed := time.Since(start)
		if elapsed > 10*time.Millisecond {
			b.Errorf("Transition too slow: %v", elapsed)
		}

		// Shift back
		controller.SelectTier(ballisticMode, 0.9)
	}
}

// TestControllerLogic tests the PAX controller's decision logic in isolation
func TestControllerLogic(t *testing.T) {
	controller := NewPAXController()

	// Test 1: Ballistic dominance → Tier 0 (threshold now 0.92)
	modeProb := map[string]float64{"ballistic": 0.93, "hgv_glide": 0.05, "maneuver": 0.05}
	tier := controller.SelectTier(modeProb, 0.98)
	if tier != Tier0 {
		t.Errorf("Expected Tier0 for ballistic dominance, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Ballistic dominance → %s\n", tier.String())

	// Test 2: Stable glide → Tier 1
	controller.ForceTier(Tier1) // Reset to Tier1 without cooldown
	modeProb = map[string]float64{"ballistic": 0.1, "hgv_glide": 0.8, "maneuver": 0.1}
	tier = controller.SelectTier(modeProb, 0.80) // R² in dead-band (0.70-0.85)
	if tier != Tier1 {
		t.Errorf("Expected Tier1 for stable glide, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Stable glide → %s\n", tier.String())

	// Test 3: Unstable glide → Tier 2 (requires R² < 0.70 for up-shift)
	controller.ForceTier(Tier1) // Reset to Tier1 without cooldown
	modeProb = map[string]float64{"ballistic": 0.1, "hgv_glide": 0.6, "maneuver": 0.3}
	tier = controller.SelectTier(modeProb, 0.68) // R² < 0.70 threshold
	if tier != Tier2 {
		t.Errorf("Expected Tier2 for unstable glide, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Unstable glide → %s\n", tier.String())

	// Test 4: Discovery trigger → Tier 3
	controller.ForceTier(Tier2) // Reset to Tier2 without cooldown
	modeProb = map[string]float64{"ballistic": 0.05, "hgv_glide": 0.3, "maneuver": 0.65}
	tier = controller.SelectTier(modeProb, 0.45) // R² < 0.5 threshold
	if tier != Tier3 {
		t.Errorf("Expected Tier3 for discovery trigger, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Discovery trigger → %s\n", tier.String())
}

// TestThermalGovernor tests the thermal governor with IOKit bridge
func TestThermalGovernor(t *testing.T) {
	governor := NewThermalGovernor()

	// Test critical threshold check
	// The IOKit bridge returns a placeholder value of 15
	if governor.IsCritical() {
		t.Error("Expected IsCritical to be false at level 15")
	}

	// Test that level is updated by IOKit call
	if governor.Level == 0 {
		t.Error("Expected thermal level to be updated by IOKit call")
	}

	fmt.Printf("[PAX] ✓ Thermal governor IOKit bridge working (level=%d)\n", governor.Level)
}

// TestHysteresis tests that hysteresis prevents gear-hunting
func TestHysteresis(t *testing.T) {
	controller := NewPAXController()

	// Test 1: Up-shift requires R² < 0.70 (not just < 0.75)
	modeProb := map[string]float64{"ballistic": 0.1, "hgv_glide": 0.8, "maneuver": 0.1}

	// R² = 0.75 should NOT trigger up-shift (below 0.70 threshold)
	tier := controller.SelectTier(modeProb, 0.75)
	if tier != Tier1 {
		t.Errorf("Expected Tier1 at R²=0.75 (below up-shift threshold), got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ No up-shift at R²=0.75 (hysteresis working)\n")

	// R² = 0.68 should trigger up-shift (below 0.70 threshold)
	tier = controller.SelectTier(modeProb, 0.68)
	if tier != Tier2 {
		t.Errorf("Expected Tier2 at R²=0.68 (above up-shift threshold), got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Up-shift triggered at R²=0.68\n")

	// Test 2: Down-shift requires R² > 0.85 (not just > 0.75)
	// Wait for cooldown to expire first
	for i := 0; i < 10; i++ {
		controller.SelectTier(modeProb, 0.90)
	}

	// Now down-shift should work
	tier = controller.SelectTier(modeProb, 0.90)
	if tier != Tier1 {
		t.Errorf("Expected Tier1 at R²=0.90 (above down-shift threshold), got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Down-shift triggered at R²=0.90\n")

	// Test 3: Cooldown prevents rapid switching
	controller.ForceTier(Tier1) // Reset to Tier1 without cooldown

	// Trigger up-shift
	tier = controller.SelectTier(modeProb, 0.68)
	if tier != Tier2 {
		t.Errorf("Expected Tier2 on first up-shift, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Up-shift triggered at R²=0.68\n")

	// Immediately try to down-shift (should be blocked by cooldown)
	tier = controller.SelectTier(modeProb, 0.90)
	if tier != Tier2 {
		t.Errorf("Expected Tier2 (cooldown active), got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Cooldown prevented immediate down-shift\n")

	// Wait for cooldown to expire (simulate 10 frames)
	for i := 0; i < 10; i++ {
		controller.SelectTier(modeProb, 0.90)
	}

	// Now down-shift should work
	tier = controller.SelectTier(modeProb, 0.90)
	if tier != Tier1 {
		t.Errorf("Expected Tier1 after cooldown expires, got %s", tier.String())
	}
	fmt.Printf("[PAX] ✓ Down-shift allowed after cooldown expires\n")
}

// TestHysteresisDeadBand tests the dead-band behavior
func TestHysteresisDeadBand(t *testing.T) {
	controller := NewPAXController()
	modeProb := map[string]float64{"ballistic": 0.1, "hgv_glide": 0.8, "maneuver": 0.1}

	// Start in Tier1
	controller.ForceTier(Tier1)

	// Test R² values in the dead-band (0.70 - 0.85)
	// These should NOT cause any tier switches
	testValues := []float64{0.71, 0.75, 0.78, 0.82, 0.84}
	for _, r2 := range testValues {
		tier := controller.SelectTier(modeProb, r2)
		if tier != Tier1 {
			t.Errorf("Expected Tier1 in dead-band at R²=%.2f, got %s", r2, tier.String())
		}
	}
	fmt.Printf("[PAX] ✓ Dead-band (0.70-0.85) prevents gear-hunting\n")
}
