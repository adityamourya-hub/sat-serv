// Package physics provides Sliding Window SINDy for real-time maneuver tracking
// Implements time-varying sparse identification for phase transition detection
package physics

import (
	"fmt"
	"sync"
	"time"
)

// SlidingSINDy performs real-time sparse regression on a sliding window of states
// This enables detection of phase transitions in adversary behavior (e.g., Ketu -> Mars -> Rahu)
type SlidingSINDy struct {
	mu         sync.RWMutex
	WindowSize int     // FIFO buffer size (e.g., 100 states = 100 seconds at 1Hz)
	Stride     int     // Run regression every N updates (e.g., 10 = 10Hz analysis)
	Buffer     [][6]float64 // Circular buffer of states [pos(3), vel(3)]
	Timestamps []float64    // Corresponding timestamps
	Regressor  *SINDy
	
	// State tracking
	LastIntent     string
	LastCoeffs     []float64
	UpdateCount    int
	TransitionLog  []IntentTransition
	
	// Performance tuning
	LastRegressionTime time.Duration
	StrideCounter    int
}

// IntentTransition records a semantic shift in adversary behavior
type IntentTransition struct {
	Timestamp   time.Time
	OldIntent   string
	NewIntent   string
	Coeffs      *ManeuverResult
	Urgency     string
}

// SITREP (Situation Report) for mesh dispatch
type SITREP struct {
	Source      string    `json:"source"`
	Target      string    `json:"target"`
	Transition  string    `json:"transition"`
	Timestamp   string    `json:"timestamp"`
	Coeffs      map[string]float64 `json:"coefficients"`
	Urgency     string    `json:"urgency"`
	Action      string    `json:"recommended_action"`
}

// NewSlidingSINDy creates a real-time maneuver tracker
func NewSlidingSINDy(windowSize, stride int) *SlidingSINDy {
	return &SlidingSINDy{
		WindowSize:    windowSize,
		Stride:        stride,
		Buffer:        make([][6]float64, 0, windowSize),
		Timestamps:    make([]float64, 0, windowSize),
		Regressor:     NewSINDy(),
		LastIntent:    "INITIALIZING",
		TransitionLog: make([]IntentTransition, 0),
		StrideCounter: 0,
	}
}

// Update adds a new state and returns the current tactical intent
// This is the main entry point - call at 100Hz for real-time tracking
func (s *SlidingSINDy) Update(state [6]float64, timestamp float64) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Maintain FIFO Buffer
	s.Buffer = append(s.Buffer, state)
	s.Timestamps = append(s.Timestamps, timestamp)
	
	if len(s.Buffer) > s.WindowSize {
		s.Buffer = s.Buffer[1:]
		s.Timestamps = s.Timestamps[1:]
	}

	// 2. Check if window is full
	if len(s.Buffer) < s.WindowSize {
		return "INITIALIZING"
	}

	// 3. Stride control: Only run regression at specified intervals
	// This maintains 100Hz data collection but reduces CPU load
	s.StrideCounter++
	if s.StrideCounter < s.Stride {
		return s.LastIntent
	}
	s.StrideCounter = 0

	// 4. Run Sparse Regression on current window
	start := time.Now()
	
	// Calculate dt from timestamps
	dt := s.calculateDT()
	
	result, err := s.Regressor.DiscoverManeuver(s.Buffer, dt)
	if err != nil {
		// Regression failed - maintain last known intent
		return s.LastIntent + "_UNCERTAIN"
	}
	
	s.LastRegressionTime = time.Since(start)

	// 5. Map coefficients to tactical intent
	currentIntent := s.mapResultToIntent(result)

	// 6. Detect Semantic Transition
	if currentIntent != s.LastIntent {
		s.handleTransition(s.LastIntent, currentIntent, result)
		s.LastIntent = currentIntent
	}

	s.LastCoeffs = result.Coefficients
	s.UpdateCount++

	return currentIntent
}

// handleTransition processes intent shifts and generates alerts
// This is called when SINDy detects a change in maneuver signature
func (s *SlidingSINDy) handleTransition(old, new string, result *ManeuverResult) {
	// Determine urgency based on transition type
	urgency := "ROUTINE"
	action := "CONTINUE_MONITORING"
	
	switch {
	// Critical: Passive object becomes kinetic threat
	case old == "SATURN" || old == "KETU_PASSIVE":
		if new == "MARS_KINETIC" || new == "MARS" {
			urgency = "CRITICAL"
			action = "INITIATE_ACTIVE_SCAN"
		}
		
	// Critical: Kinetic threat adds electronic warfare
	case old == "MARS_KINETIC" || old == "MARS":
		if new == "RAHU_DECEPTIVE" || new == "RAHU" {
			urgency = "CRITICAL"
			action = "ACTIVATE_ECM_COUNTERMEASURES"
		}
		
	// Warning: Any movement from passive state
	case old == "INITIALIZING":
		if new != "SATURN" && new != "KETU_PASSIVE" {
			urgency = "WARNING"
			action = "INCREASE_CADENCE"
		}
		
	// De-escalation: Threat returns to passive
	case new == "SATURN" || new == "KETU_PASSIVE":
		urgency = "INFO"
		action = "MAINTAIN_CUSTODY"
	}

	// Record transition
	transition := IntentTransition{
		Timestamp: time.Now(),
		OldIntent: old,
		NewIntent: new,
		Coeffs:    result,
		Urgency:   urgency,
	}
	s.TransitionLog = append(s.TransitionLog, transition)

	// Generate SITREP for critical transitions
	if urgency == "CRITICAL" {
		sitrep := s.generateSITREP(transition, action)
		s.dispatchSITREP(sitrep)
	}
}

// mapResultToIntent maps SINDy coefficients to tactical intent (Siddhantic Meta-Header)
func (s *SlidingSINDy) mapResultToIntent(result *ManeuverResult) string {
	if !result.ThrustDetected || result.ThrustMagnitude < 0.001 {
		return "SATURN" // Passive, Keplerian motion
	}

	// Analyze coefficient pattern
	oscillatoryTerms := 0
	constantTerms := 0
	
	for _, term := range result.ActiveTerms {
		if containsTrig(term) {
			oscillatoryTerms++
		}
		if isConstantTerm(term) {
			constantTerms++
		}
	}

	// Classification logic
	// Mars: High constant thrust, minimal oscillation
	if result.ThrustMagnitude > 0.02 && constantTerms > 0 && oscillatoryTerms == 0 {
		return "MARS_KINETIC"
	}
	
	// Rahu: Oscillatory pattern, lower magnitude
	if oscillatoryTerms > 0 && result.ThrustMagnitude < 0.02 {
		return "RAHU_DECEPTIVE"
	}
	
	// Ketu: Stealthy, very low thrust
	if result.ThrustMagnitude < 0.005 {
		return "KETU_STEALTHY"
	}
	
	// Generic kinetic if unclear
	if result.ThrustMagnitude > 0.01 {
		return "MARS"
	}
	
	return "UNKNOWN"
}

// generateSITREP creates a Situation Report for mesh dispatch
func (s *SlidingSINDy) generateSITREP(transition IntentTransition, action string) SITREP {
	coeffs := make(map[string]float64)
	
	// Extract key coefficients for SITREP
	if transition.Coeffs != nil {
		for i, term := range transition.Coeffs.ActiveTerms {
			if i < len(transition.Coeffs.Coefficients) {
				coeffs[term] = transition.Coeffs.Coefficients[i]
			}
		}
	}
	
	// Ensure essential fields exist
	if _, exists := coeffs["CONST"]; !exists {
		coeffs["CONST"] = 0.0
	}
	if _, exists := coeffs["SIN"]; !exists {
		coeffs["SIN"] = 0.0
	}

	return SITREP{
		Source:     fmt.Sprintf("SINDy-W%02d", s.WindowSize),
		Target:     "UCT-ADITI-SSA",
		Transition: fmt.Sprintf("%s -> %s", transition.OldIntent, transition.NewIntent),
		Timestamp:  transition.Timestamp.Format("2006-01-02T15:04:05Z"),
		Coeffs:     coeffs,
		Urgency:    transition.Urgency,
		Action:     action,
	}
}

// dispatchSITREP sends alert to mesh (placeholder for Reticulum integration)
func (s *SlidingSINDy) dispatchSITREP(sitrep SITREP) {
	// In production, this would serialize and send via Reticulum/Sideband
	fmt.Printf("\n🚨 SITREP // SOURCE: %s // ADITI-SSA\n", sitrep.Source)
	fmt.Printf("   TARGET: %s\n", sitrep.Target)
	fmt.Printf("   TRANSITION: %s\n", sitrep.Transition)
	fmt.Printf("   TIMESTAMP: %s\n", sitrep.Timestamp)
	fmt.Printf("   COEFFS: CONST=%.3f SIN=%.3f\n", sitrep.Coeffs["CONST"], sitrep.Coeffs["SIN"])
	fmt.Printf("   URGENCY: %s // %s\n\n", sitrep.Urgency, sitrep.Action)
}

// GetCurrentIntent returns the latest detected intent (thread-safe)
func (s *SlidingSINDy) GetCurrentIntent() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.LastIntent
}

// GetLastCoeffs returns the latest coefficients (thread-safe)
func (s *SlidingSINDy) GetLastCoeffs() []float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.LastCoeffs
}

// GetTransitionLog returns all recorded transitions
func (s *SlidingSINDy) GetTransitionLog() []IntentTransition {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return append([]IntentTransition(nil), s.TransitionLog...)
}

// GetBufferStats returns current buffer and performance statistics
func (s *SlidingSINDy) GetBufferStats() (fill int, lastRegTime time.Duration, updates int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.Buffer), s.LastRegressionTime, s.UpdateCount
}

// GetSITREP generates current SITREP without transition trigger
func (s *SlidingSINDy) GetSITREP() SITREP {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	transition := IntentTransition{
		Timestamp: time.Now(),
		OldIntent: s.LastIntent,
		NewIntent: s.LastIntent,
		Urgency:   "INFO",
	}
	
	return s.generateSITREP(transition, "STATUS_REPORT")
}

// Reset clears the buffer and state (for testing or new target)
func (s *SlidingSINDy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.Buffer = s.Buffer[:0]
	s.Timestamps = s.Timestamps[:0]
	s.LastIntent = "INITIALIZING"
	s.LastCoeffs = nil
	s.UpdateCount = 0
	s.StrideCounter = 0
}

// calculateDT estimates time step from timestamps
func (s *SlidingSINDy) calculateDT() float64 {
	if len(s.Timestamps) < 2 {
		return 1.0 // Default 1 second
	}
	
	// Calculate average dt over window
	totalDT := 0.0
	count := 0
	
	for i := 1; i < len(s.Timestamps); i++ {
		dt := s.Timestamps[i] - s.Timestamps[i-1]
		if dt > 0 {
			totalDT += dt
			count++
		}
	}
	
	if count == 0 {
		return 1.0
	}
	
	return totalDT / float64(count)
}

// Helper functions
func containsTrig(term string) bool {
	return len(term) >= 3 && (term[:3] == "sin" || term[:3] == "cos")
}

func isConstantTerm(term string) bool {
	return term == "1" || term == "const" || term == "CONST"
}

// =============================================================================
// M2 Air Performance Configuration
// =============================================================================

// DefaultM2AirConfig returns optimized settings for Apple M2
func DefaultM2AirConfig() *SlidingSINDy {
	// M2 Air optimizations:
	// - WindowSize: 100 states (100 seconds at 1Hz, 10 seconds at 10Hz)
	// - Stride: 10 (run regression at 10Hz instead of 100Hz to save CPU)
	// - Uses gonum BLAS which leverages AMX (Apple Matrix) co-processor
	return NewSlidingSINDy(100, 10)
}

// HighCadenceConfig for maximum responsiveness (higher CPU usage)
func HighCadenceConfig() *SlidingSINDy {
	// Run regression every 5 updates = 20Hz at 100Hz input
	return NewSlidingSINDy(100, 5)
}

// LowPowerConfig for extended operations (battery conservation)
func LowPowerConfig() *SlidingSINDy {
	// Run regression every 20 updates = 5Hz at 100Hz input
	return NewSlidingSINDy(50, 20)
}

// =============================================================================
// Usage Example
// =============================================================================
//
// swsindy := physics.DefaultM2AirConfig()
//
// for i := 0; i < 1000; i++ {
//     state := [6]float64{x, y, z, vx, vy, vz} // From sensor/tracker
//     intent := swsindy.Update(state, float64(i))
//     
//     if intent == "MARS_KINETIC" {
//         // Trigger evasion
//     }
// }
//
// transitions := swsindy.GetTransitionLog()
// for _, t := range transitions {
//     fmt.Printf("%s: %s -> %s\n", t.Timestamp, t.OldIntent, t.NewIntent)
// }
//
