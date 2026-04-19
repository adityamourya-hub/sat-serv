// Package threat provides Finite State Machine for context-aware threat detection
// Tracks RSO intent evolution through 5 states with HCW RPO pattern detection
package threat

import (
	"fmt"
	"time"

	"ssis/pkg/physics"
)

// ThreatState defines the current tactical standing of an RSO
type ThreatState int

const (
	StatePassive ThreatState = iota
	StateMonitoring
	StateStalking
	StateFuelTrap
	StateTerminalClosure
	StateIntercepted
)

func (s ThreatState) String() string {
	return [...]string{"PASSIVE", "MONITORING", "STALKING", "FUEL_TRAP", "TERMINAL_CLOSURE", "INTERCEPTED"}[s]
}

// FSMInput provides all inputs required for state transition evaluation
type FSMInput struct {
	DistanceKm    float64 // Distance to asset (km)
	DeltaVMps     float64 // Relative velocity (m/s)
	IsManeuvering bool    // EKF maneuver detection
	IsStabilized  bool    // Attitude stability
	MaterialID    string  // Material from spectral analysis
	MaterialConf  float64 // Material confidence
	TimeStepSec   float64 // Time between observations (seconds)

	// HCW-specific inputs for RPO pattern detection
	TargetPos  [3]float64 // Asset position (ECI, km)
	TargetVel  [3]float64 // Asset velocity (ECI, km/s)
	ChaserPos  [3]float64 // RSO position (ECI, km)
	ChaserVel  [3]float64 // RSO velocity (ECI, km/s)
	MeanMotion float64    // Reference orbit mean motion (rad/s)

	// Fuel trap inputs
	DebrisPosition [3]float64 // Debris position for fuel trap (ECI, km)
	DebrisCov      [6]float64 // Debris covariance (upper triangular)
	Probability    float64    // Current probability of collision
}

// ThreatFSM manages state transitions for a single RSO with dwell timer
type ThreatFSM struct {
	SatID          int
	CurrentState   ThreatState
	LastUpdate     time.Time
	StateEntryTime time.Time

	// Dwell timer - minimum time required in a state before transition
	DwellTimer       time.Duration
	ConsecutiveTicks int // Count of consecutive observations in current state

	// HCW observation history for RPO pattern detection
	HCWHistory []physics.HCWState
	MaxHistory int
}

// NewThreatFSM creates FSM with default dwell timer (3 minutes)
func NewThreatFSM(satID int) *ThreatFSM {
	return &ThreatFSM{
		SatID:          satID,
		CurrentState:   StatePassive,
		LastUpdate:     time.Now(),
		StateEntryTime: time.Now(),
		DwellTimer:     3 * time.Minute, // 3-minute minimum dwell
		MaxHistory:     10,              // Keep 10 observations
		HCWHistory:     make([]physics.HCWState, 0),
	}
}

// Update evaluates inputs and triggers state transitions
// Returns: (stateChanged, transitionMessage)
func (fsm *ThreatFSM) Update(inputs FSMInput) (bool, string) {
	oldState := fsm.CurrentState
	now := time.Now()

	// Update HCW history for RPO pattern detection
	fsm.updateHCWHistory(inputs)

	// Increment consecutive ticks if same state
	if now.Sub(fsm.LastUpdate) < 2*time.Minute {
		fsm.ConsecutiveTicks++
	} else {
		fsm.ConsecutiveTicks = 1
	}

	// Check dwell timer requirement
	dwellSatisfied := now.Sub(fsm.StateEntryTime) >= fsm.DwellTimer

	fsm.LastUpdate = now

	// State transition logic
	switch fsm.CurrentState {
	case StatePassive:
		// Transition to MONITORING if within 100km
		if inputs.DistanceKm < 100.0 {
			fsm.transitionTo(StateMonitoring, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Distance: %.1f km)",
				fsm.SatID, oldState, fsm.CurrentState, inputs.DistanceKm)
		}

	case StateMonitoring:
		// Check for HCW RPO pattern (high-confidence stalking detection)
		if len(fsm.HCWHistory) >= 5 {
			isRPO, residual := physics.IsRPOPattern(fsm.HCWHistory, inputs.MeanMotion, inputs.TimeStepSec)
			if isRPO && inputs.DistanceKm < 50.0 {
				fsm.transitionTo(StateStalking, now)
				return true, fmt.Sprintf("SAT %d: HCW ELLIPSE DETECTED. TRANSITION %s -> %s (Residual: %.3f km)",
					fsm.SatID, oldState, fsm.CurrentState, residual)
			}
		}

		// Standard stalking detection: matching orbit with small delta-v
		if dwellSatisfied && inputs.DistanceKm < 20.0 && inputs.DeltaVMps < 2.0 && inputs.IsManeuvering {
			fsm.transitionTo(StateStalking, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Station-keeping detected)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

		// Fuel trap detection: adversary approaching debris conjunction threshold
		// Check if probability of collision is in the "trap zone" (1e-5 to 1e-4)
		if inputs.Probability > 5e-5 && inputs.Probability < 1e-4 {
			fsm.transitionTo(StateFuelTrap, now)
			return true, fmt.Sprintf("SAT %d: FUEL TRAP OPPORTUNITY. TRANSITION %s -> %s (Pc: %.2e)",
				fsm.SatID, oldState, fsm.CurrentState, inputs.Probability)
		}

		// Return to PASSIVE if far away
		if inputs.DistanceKm > 150.0 {
			fsm.transitionTo(StatePassive, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Departed AOI)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

	case StateStalking:
		// Transition to TERMINAL_CLOSURE: rapid approach with stability
		if dwellSatisfied && inputs.DistanceKm < 2.0 && inputs.DeltaVMps > 0.5 && inputs.IsStabilized {
			fsm.transitionTo(StateTerminalClosure, now)
			return true, fmt.Sprintf("SAT %d: CRITICAL. TRANSITION %s -> %s (Rapid closure: %.2f m/s)",
				fsm.SatID, oldState, fsm.CurrentState, inputs.DeltaVMps)
		}

		// Return to MONITORING if backing off
		if inputs.DistanceKm > 50.0 {
			fsm.transitionTo(StateMonitoring, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Backing off)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

	case StateFuelTrap:
		// Exit fuel trap if probability drops below warning threshold
		if inputs.Probability < 5e-5 {
			fsm.transitionTo(StateMonitoring, now)
			return true, fmt.Sprintf("SAT %d: EXIT FUEL TRAP. TRANSITION %s -> %s (Pc: %.2e)",
				fsm.SatID, oldState, fsm.CurrentState, inputs.Probability)
		}

		// If adversary maneuvers (probability increases), they're reacting to trap
		if inputs.Probability > 1e-4 {
			fsm.transitionTo(StateMonitoring, now)
			return true, fmt.Sprintf("SAT %d: ADVERSARY MANEUVERED. TRANSITION %s -> %s (Trap Successful)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

		// Return to PASSIVE if far away
		if inputs.DistanceKm > 150.0 {
			fsm.transitionTo(StatePassive, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Departed AOI)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

	case StateTerminalClosure:
		// Transition to INTERCEPTED at physical contact range
		if inputs.DistanceKm < 0.2 { // 200 meters
			fsm.transitionTo(StateIntercepted, now)
			return true, fmt.Sprintf("SAT %d: EMERGENCY. TRANSITION %s -> %s (INTERCEPT IMMINENT)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

		// Back off to STALKING if distance increases
		if inputs.DistanceKm > 5.0 {
			fsm.transitionTo(StateStalking, now)
			return true, fmt.Sprintf("SAT %d: TRANSITION %s -> %s (Aborted approach)",
				fsm.SatID, oldState, fsm.CurrentState)
		}

	case StateIntercepted:
		// Terminal state - manual intervention required
		// Could add logic for "cleared" state after countermeasures
	}

	return false, ""
}

// transitionTo handles state transition with dwell timer reset
func (fsm *ThreatFSM) transitionTo(newState ThreatState, now time.Time) {
	fsm.CurrentState = newState
	fsm.StateEntryTime = now
	fsm.ConsecutiveTicks = 0

	// Clear HCW history when leaving MONITORING state
	if newState != StateMonitoring {
		fsm.HCWHistory = make([]physics.HCWState, 0)
	}
}

// updateHCWHistory maintains RIC frame observations for RPO pattern detection
func (fsm *ThreatFSM) updateHCWHistory(inputs FSMInput) {
	// Only update in MONITORING state
	if fsm.CurrentState != StateMonitoring {
		return
	}

	// Convert to RIC frame
	hcwState, err := physics.ComputeRICFrame(
		inputs.TargetPos, inputs.TargetVel,
		inputs.ChaserPos, inputs.ChaserVel,
	)
	if err != nil {
		return
	}

	fsm.HCWHistory = append(fsm.HCWHistory, hcwState)

	// Keep history bounded
	if len(fsm.HCWHistory) > fsm.MaxHistory {
		fsm.HCWHistory = fsm.HCWHistory[1:]
	}
}

// ShouldAlert determines if current state requires immediate notification
func (fsm *ThreatFSM) ShouldAlert() bool {
	return fsm.CurrentState >= StateStalking
}

// GetAlertMessage returns human-readable alert for current state
func (fsm *ThreatFSM) GetAlertMessage() string {
	switch fsm.CurrentState {
	case StatePassive:
		return "No threat"
	case StateMonitoring:
		return "Monitoring - object in AOI"
	case StateStalking:
		return "ALERT: Co-orbital stalking detected"
	case StateFuelTrap:
		return "STRATEGIC: Fuel trap opportunity - adversary in debris conjunction zone"
	case StateTerminalClosure:
		return "CRITICAL: Terminal closure - immediate action required"
	case StateIntercepted:
		return "EMERGENCY: Intercept imminent"
	default:
		return "Unknown state"
	}
}

// GetPriority returns mesh priority level for current state
func (fsm *ThreatFSM) GetPriority() string {
	switch fsm.CurrentState {
	case StatePassive:
		return "ROUTINE"
	case StateMonitoring:
		return "HIGH"
	case StateStalking:
		return "CRITICAL"
	case StateFuelTrap:
		return "HIGH" // Strategic priority, not emergency
	case StateTerminalClosure:
		return "CRITICAL"
	case StateIntercepted:
		return "CRITICAL"
	default:
		return "ROUTINE"
	}
}

// GetActionCode returns recommended action based on state
func (fsm *ThreatFSM) GetActionCode() string {
	switch fsm.CurrentState {
	case StatePassive:
		return "TRACK_MAINTAIN"
	case StateMonitoring:
		return "TRACK_MAINTAIN"
	case StateStalking:
		return "EMCON_B_INITIATE" // Emission control
	case StateFuelTrap:
		return "UNCERTAINTY_INFLATION" // Strategic data manipulation
	case StateTerminalClosure:
		return "EVASIVE_MANEUVER"
	case StateIntercepted:
		return "IMMEDIATE_DEPLOY"
	default:
		return "TRACK_MAINTAIN"
	}
}

// FSMManager manages multiple FSM instances for all tracked RSOs
type FSMManager struct {
	fsms map[int]*ThreatFSM
}

// NewFSMManager creates FSM manager
func NewFSMManager() *FSMManager {
	return &FSMManager{
		fsms: make(map[int]*ThreatFSM),
	}
}

// GetOrCreateFSM retrieves existing FSM or creates new one
func (m *FSMManager) GetOrCreateFSM(satID int) *ThreatFSM {
	if fsm, exists := m.fsms[satID]; exists {
		return fsm
	}

	fsm := NewThreatFSM(satID)
	m.fsms[satID] = fsm
	return fsm
}

// ProcessUpdate processes input for a satellite and returns transition info
func (m *FSMManager) ProcessUpdate(satID int, inputs FSMInput) (bool, string) {
	fsm := m.GetOrCreateFSM(satID)
	return fsm.Update(inputs)
}

// GetFSM returns FSM for a satellite (nil if not tracked)
func (m *FSMManager) GetFSM(satID int) *ThreatFSM {
	return m.fsms[satID]
}

// RemoveFSM removes FSM from tracking
func (m *FSMManager) RemoveFSM(satID int) {
	delete(m.fsms, satID)
}

// GetAlertingFSMs returns all FSMs that should trigger alerts
func (m *FSMManager) GetAlertingFSMs() []*ThreatFSM {
	alerting := make([]*ThreatFSM, 0)

	for _, fsm := range m.fsms {
		if fsm.ShouldAlert() {
			alerting = append(alerting, fsm)
		}
	}

	return alerting
}

// GetStats returns manager statistics
func (m *FSMManager) GetStats() map[string]interface{} {
	stats := make(map[string]interface{})

	stateCounts := make(map[ThreatState]int)
	for _, fsm := range m.fsms {
		stateCounts[fsm.CurrentState]++
	}

	stats["total_tracked"] = len(m.fsms)
	stats["state_distribution"] = stateCounts
	stats["alerting_count"] = len(m.GetAlertingFSMs())

	return stats
}
