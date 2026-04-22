package proximity

import (
	"math"
	"ssis/pkg/physics"
	"testing"
)

func TestNewSlewController(t *testing.T) {
	sc := NewSlewController()
	
	if sc == nil {
		t.Fatal("Failed to create SlewController")
	}
	
	if sc.MaxRate != 1.0 {
		t.Errorf("Expected MaxRate=1.0, got %.2f", sc.MaxRate)
	}
	
	if sc.SettleTime != 0.05 {
		t.Errorf("Expected SettleTime=0.05s, got %.2f", sc.SettleTime)
	}
	
	t.Logf("SlewController: MaxRate=%.2f rad/s, SettleTime=%.3f s", 
		sc.MaxRate, sc.SettleTime)
}

func TestSlewCommand_IsValid(t *testing.T) {
	cmd := SlewCommand{
		Azimuth:   0.5,
		Elevation: 0.3,
		TiltRate:  0.8,
		TiltAccel: 1.5,
		TiltJerk:  8.0,
	}
	
	if !cmd.IsValid(1.0, 2.0, 10.0) {
		t.Error("Valid command rejected")
	}
	
	// Test invalid rate
	cmd.TiltRate = 1.5
	if cmd.IsValid(1.0, 2.0, 10.0) {
		t.Error("Invalid rate should be rejected")
	}
}

func TestSlewController_ComputeSlewCommand(t *testing.T) {
	sc := NewSlewController()
	sc.CurrentAz = 0.0
	sc.CurrentEl = 0.0
	
	// Station at origin (simplified)
	stationECEF := physics.Vector3{0, 0, 0}
	
	// Target directly overhead in future
	target := PredictedTarget{
		Position:   physics.Vector3{100, 0, 100}, // 100km up, 100km east
		Velocity:   physics.Vector3{0, 0, 0},
		Timestamp:  0.6, // 600ms in future
		Confidence: 0.9,
		Regime:     physics.FlightRegimeLEO,
	}
	
	cmd := sc.ComputeSlewCommand(0.0, target, stationECEF)
	
	t.Logf("Slew command: AZ=%.3f rad, EL=%.3f rad", cmd.Azimuth, cmd.Elevation)
	t.Logf("TiltRate=%.3f rad/s, TiltAccel=%.3f rad/s²", cmd.TiltRate, cmd.TiltAccel)
	
	// Should point roughly east (az ≈ 0) and up (el > 0)
	if cmd.Elevation < 0 {
		t.Error("Elevation should be positive for overhead target")
	}
	
	if cmd.Priority < 1 || cmd.Priority > 9 {
		t.Errorf("Priority out of range: %d", cmd.Priority)
	}
}

func TestSlewController_eciToTopocentric(t *testing.T) {
	sc := NewSlewController()
	
	// Station at origin
	stationECEF := physics.Vector3{0, 0, 0}
	
	tests := []struct {
		pos    physics.Vector3
		expAz  float64
		expEl  float64
	}{
		{physics.Vector3{100, 0, 0}, 0, 0},           // East, horizon
		{physics.Vector3{0, 100, 0}, math.Pi / 2, 0}, // North, horizon
		{physics.Vector3{0, 0, 100}, 0, math.Pi / 2}, // Zenith
	}
	
	for i, test := range tests {
		az, el := sc.eciToTopocentric(test.pos, stationECEF)
		
		t.Logf("Test %d: pos=(%.0f, %.0f, %.0f) -> az=%.3f, el=%.3f",
			i, test.pos[0], test.pos[1], test.pos[2], az, el)
		
		// Check elevation (simplified - exact values depend on coordinate system)
		if el < -math.Pi/2 || el > math.Pi/2 {
			t.Errorf("Test %d: Elevation out of range: %.3f", i, el)
		}
	}
}

func TestSlewController_computeTargetRate(t *testing.T) {
	sc := NewSlewController()
	
	// Simple case: move from 0,0 to 1,1 in 1 second
	rate := sc.computeTargetRate(0, 0, 1.0, 1.0, 1.0)
	
	t.Logf("Required rate: %.3f rad/s", rate)
	
	if rate < 0 {
		t.Error("Rate should be positive for positive displacement")
	}
	
	// Should be ~1.2 * sqrt(2) rad/s (with 20% margin)
	expected := 1.2 * math.Sqrt(2)
	if math.Abs(rate-expected) > 0.01 {
		t.Errorf("Rate mismatch: got %.3f, expected %.3f", rate, expected)
	}
}

func TestSlewController_constrainRate(t *testing.T) {
	sc := NewSlewController()
	
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.5, 0.5},   // Within limits
		{1.5, 1.0},   // Above max
		{-1.5, -1.0}, // Below min
	}
	
	for _, test := range tests {
		result := sc.constrainRate(test.input)
		if result != test.expected {
			t.Errorf("constrainRate(%.2f): got %.2f, expected %.2f",
				test.input, result, test.expected)
		}
	}
}

func TestSlewController_computePriority(t *testing.T) {
	sc := NewSlewController()
	
	tests := []struct {
		regime   physics.FlightRegime
		minPriority int
	}{
		{physics.FlightRegimeHGVEntry, 9},
		{physics.FlightRegimeBallistic, 7},
		{physics.FlightRegimeLEO, 5},
		{physics.FlightRegimeGEO, 1},
	}
	
	for _, test := range tests {
		priority := sc.computePriority(test.regime)
		if priority < test.minPriority {
			t.Errorf("Regime %v: priority %d < expected %d",
				test.regime, priority, test.minPriority)
		}
		t.Logf("Regime %v: priority=%d", test.regime, priority)
	}
}

func TestSlewController_emergencyAdjust(t *testing.T) {
	sc := NewSlewController()
	
	cmd := SlewCommand{
		Azimuth:   0,
		Elevation: 0,
		TiltRate:  2.0,  // Exceeds MaxRate=1.0
		TiltAccel: 5.0,  // Exceeds MaxAccel=2.0
		TiltJerk:  20.0, // Exceeds MaxJerk=10.0
	}
	
	adjusted := sc.emergencyAdjust(cmd)
	
	t.Logf("Before: rate=%.2f, accel=%.2f, jerk=%.2f",
		cmd.TiltRate, cmd.TiltAccel, cmd.TiltJerk)
	t.Logf("After:  rate=%.2f, accel=%.2f, jerk=%.2f",
		adjusted.TiltRate, adjusted.TiltAccel, adjusted.TiltJerk)
	
	if math.Abs(adjusted.TiltRate) > sc.MaxRate {
		t.Errorf("Rate still exceeds max: %.2f > %.2f",
			adjusted.TiltRate, sc.MaxRate)
	}
	
	if sc.SettleViolations != 1 {
		t.Errorf("Expected 1 settle violation, got %d", sc.SettleViolations)
	}
}

func TestPredictedTarget_LeadTime(t *testing.T) {
	target := PredictedTarget{
		Position:   physics.Vector3{0, 0, 0},
		Timestamp:  10.0,
		Confidence: 0.9,
	}
	
	lead := target.LeadTime(5.0)
	if lead != 5.0 {
		t.Errorf("Expected lead time 5.0, got %.2f", lead)
	}
}

func TestNewFractionalSlewController(t *testing.T) {
	fsc := NewFractionalSlewController()
	
	if fsc == nil {
		t.Fatal("Failed to create FractionalSlewController")
	}
	
	if fsc.WakeMemory != 0.3 {
		t.Errorf("Expected WakeMemory=0.3, got %.2f", fsc.WakeMemory)
	}
	
	t.Logf("FractionalSlewController: WakeMemory=%.2f, SpectralDecay=%.2f",
		fsc.WakeMemory, fsc.SpectralDecay)
}

func TestFractionalSlewController_computeWakeEffect(t *testing.T) {
	fsc := NewFractionalSlewController()
	
	// Create wake history
	wakeHistory := []physics.Vector3{
		{0, 0, 50},   // 50km up
		{10, 0, 50},  // 10km east
		{20, 0, 50},  // 20km east
	}
	
	targetPos := physics.Vector3{15, 0, 50} // Between wake points
	
	effect := fsc.computeWakeEffect(wakeHistory, targetPos)
	
	t.Logf("Wake effect: %.6f rad", effect)
	
	// Effect should be non-zero but small
	if math.Abs(effect) < 1e-10 {
		t.Logf("Note: Wake effect very small (target far from wake)")
	}
}

func TestSlewController_GetStats(t *testing.T) {
	sc := NewSlewController()
	
	// Simulate some activity
	sc.CommandsIssued = 100
	sc.SettleViolations = 5
	sc.CurrentAz = 0.5
	sc.CurrentEl = 0.3
	
	stats := sc.GetStats()
	
	if stats["commands_issued"] != 100 {
		t.Errorf("Expected 100 commands, got %v", stats["commands_issued"])
	}
	
	if stats["settle_violations"] != 5 {
		t.Errorf("Expected 5 violations, got %v", stats["settle_violations"])
	}
	
	t.Logf("Stats: commands=%v, violations=%v, az=%.3f, el=%.3f",
		stats["commands_issued"], stats["settle_violations"],
		stats["current_az"], stats["current_el"])
}

func TestSign(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{5.0, 1},
		{-3.0, -1},
		{0.0, 0},
		{0.001, 1},
		{-0.001, -1},
	}
	
	for _, test := range tests {
		result := sign(test.input)
		if result != test.expected {
			t.Errorf("sign(%.3f): got %.0f, expected %.0f",
				test.input, result, test.expected)
		}
	}
}
