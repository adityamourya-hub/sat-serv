// Package rl provides Go-Python RL policy bridge for autonomous SSA decision-making
// Implements ZMQ REQ socket to communicate with Python Actor-Critic policy head
package rl

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/pebbe/zmq4"
)

// Action represents tactical command decisions
type Action int

const (
	ActionPassiveTrack     Action = 0 // Standard custody, minimal sensor draw
	ActionActiveScan       Action = 1 // High frame rate, MGE + Spectroscopy
	ActionFollowTarget     Action = 2 // INDI mount lock on specific target
	ActionEmitterGeo       Action = 3 // TDOA geolocation for jammer detection
	ActionTacticalAlert    Action = 4 // Broadcast FCH to mesh
	ActionFuelOptimalEvade Action = 5 // Execute escape maneuver (on-orbit)
)

// ActionNames maps actions to human-readable strings
var ActionNames = map[Action]string{
	ActionPassiveTrack:     "PASSIVE_TRACK",
	ActionActiveScan:       "ACTIVE_SCAN",
	ActionFollowTarget:     "FOLLOW_TARGET",
	ActionEmitterGeo:       "EMITTER_GEO",
	ActionTacticalAlert:    "TACTICAL_ALERT",
	ActionFuelOptimalEvade: "FUEL_OPTIMAL_EVADE",
}

// Intent represents adversary behavioral classification
type Intent int

const (
	IntentPassive           Intent = 0 // Pure Keplerian, minimal Δv
	IntentRPOInspection     Intent = 1 // HCW-locked, circumnavigation
	IntentElectronicWarfare Intent = 2 // Nadir-pointing, thermal spikes
	IntentKineticThreat     Intent = 3 // High-thrust, non-Keplerian
)

// IntentNames maps intents to human-readable strings
var IntentNames = map[Intent]string{
	IntentPassive:           "PASSIVE",
	IntentRPOInspection:     "RPO_INSPECTION",
	IntentElectronicWarfare: "ELECTRONIC_WARFARE",
	IntentKineticThreat:     "KINETIC_THREAT",
}

// StateVector represents the input state to the RL policy
type StateVector struct {
	// Orbital state (normalized)
	Position [3]float64 `json:"position"` // ECI position (km)
	Velocity [3]float64 `json:"velocity"` // ECI velocity (km/s)

	// Intent classification (from neural network)
	Intent     int     `json:"intent"`     // 0-3 intent label
	Confidence float64 `json:"confidence"` // 0.0-1.0 confidence

	// Threat indicators
	PowerEstimate float64 `json:"power_estimate"` // dBm
	RiskScore     float64 `json:"risk_score"`     // 0.0-100.0

	// Physics residuals (Yoshida-4 integrator)
	HCWResidual  float64 `json:"hcw_residual"`   // Hill-Clohessy-Wiltshire residual
	DeltaVRecent float64 `json:"delta_v_recent"` // Recent maneuver magnitude (m/s)

	// Custody state
	TimeSinceLastCustody float64 `json:"time_since_custody"` // seconds
	FramesInCustody      int     `json:"frames_in_custody"`  // consecutive frames
}

// PolicyResponse represents the action recommendation from Python policy head
type PolicyResponse struct {
	Action     int     `json:"action"`      // 0-5 action index
	ActionProb float64 `json:"action_prob"` // Probability of chosen action
	Value      float64 `json:"value"`       // Critic's state value estimate
	Entropy    float64 `json:"entropy"`     // Policy entropy (exploration)
	Timestamp  int64   `json:"timestamp"`   // Unix nanoseconds
}

// CommandBridge manages ZMQ communication with Python RL policy head
type CommandBridge struct {
	endpoint  string
	timeout   time.Duration
	socket    *zmq4.Socket
	connected bool
}

// NewCommandBridge creates a new RL command bridge
func NewCommandBridge(endpoint string) *CommandBridge {
	if endpoint == "" {
		endpoint = "ipc:///tmp/policy_bridge.ipc"
	}

	return &CommandBridge{
		endpoint:  endpoint,
		timeout:   100 * time.Millisecond, // Fast response required
		connected: false,
	}
}

// Connect establishes ZMQ connection to Python policy head
func (cb *CommandBridge) Connect() error {
	socket, err := zmq4.NewSocket(zmq4.REQ)
	if err != nil {
		return fmt.Errorf("failed to create ZMQ REQ socket: %w", err)
	}

	// Set receive timeout to prevent blocking forever
	socket.SetRcvtimeo(cb.timeout)

	if err := socket.Connect(cb.endpoint); err != nil {
		socket.Close()
		return fmt.Errorf("failed to connect to %s: %w", cb.endpoint, err)
	}

	cb.socket = socket
	cb.connected = true
	fmt.Printf("[RL] Connected to Python policy head at %s\n", cb.endpoint)
	return nil
}

// GetAction requests action decision from RL policy head
func (cb *CommandBridge) GetAction(state StateVector) (*PolicyResponse, error) {
	if !cb.connected || cb.socket == nil {
		return nil, fmt.Errorf("command bridge not connected")
	}

	// Serialize state to JSON
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal state: %w", err)
	}

	// Send state via ZMQ
	if _, err := cb.socket.Send(string(stateJSON), 0); err != nil {
		return nil, fmt.Errorf("failed to send state to policy head: %w", err)
	}

	// Receive response
	reply, err := cb.socket.Recv(0)
	if err != nil {
		return nil, fmt.Errorf("failed to receive response from policy head: %w", err)
	}

	// Parse response
	var response PolicyResponse
	if err := json.Unmarshal([]byte(reply), &response); err != nil {
		return nil, fmt.Errorf("failed to parse policy response: %w", err)
	}

	return &response, nil
}

// ExecuteAction performs the tactical action
func (cb *CommandBridge) ExecuteAction(action Action, state StateVector) error {
	actionName := ActionNames[action]

	switch action {
	case ActionPassiveTrack:
		return cb.executePassiveTrack(state)
	case ActionActiveScan:
		return cb.executeActiveScan(state)
	case ActionFollowTarget:
		return cb.executeFollowTarget(state)
	case ActionEmitterGeo:
		return cb.executeEmitterGeo(state)
	case ActionTacticalAlert:
		return cb.executeTacticalAlert(state)
	case ActionFuelOptimalEvade:
		return cb.executeFuelOptimalEvade(state)
	default:
		return fmt.Errorf("unknown action: %s", actionName)
	}
}

func (cb *CommandBridge) executePassiveTrack(state StateVector) error {
	// Minimal sensor draw, standard custody
	fmt.Printf("[RL] PASSIVE_TRACK: Standard custody for NORAD target\n")
	return nil
}

func (cb *CommandBridge) executeActiveScan(state StateVector) error {
	// High frame rate, MGE + Spectroscopy
	fmt.Printf("[RL] ACTIVE_SCAN: High frame rate for structural analysis\n")
	return nil
}

func (cb *CommandBridge) executeFollowTarget(state StateVector) error {
	// INDI mount lock on specific target
	fmt.Printf("[RL] FOLLOW_TARGET: INDI mount tracking engaged\n")
	return nil
}

func (cb *CommandBridge) executeEmitterGeo(state StateVector) error {
	// TDOA geolocation for jammer detection
	fmt.Printf("[RL] EMITTER_GEO: Initiating TDOA geolocation\n")
	return nil
}

func (cb *CommandBridge) executeTacticalAlert(state StateVector) error {
	// Broadcast FCH to mesh
	fmt.Printf("[RL] TACTICAL_ALERT: Broadcasting FCH to Reticulum mesh\n")
	return nil
}

func (cb *CommandBridge) executeFuelOptimalEvade(state StateVector) error {
	// Execute escape maneuver (on-orbit)
	fmt.Printf("[RL] FUEL_OPTIMAL_EVADE: Executing escape maneuver\n")
	return nil
}

// Disconnect closes ZMQ connection
func (cb *CommandBridge) Disconnect() error {
	if cb.socket != nil {
		cb.socket.Close()
		cb.socket = nil
	}
	cb.connected = false
	fmt.Printf("[RL] Disconnected from Python policy head\n")
	return nil
}

// IsConnected returns connection status
func (cb *CommandBridge) IsConnected() bool {
	return cb.connected
}

// TacticalAgent provides high-level RL-driven decision making
type TacticalAgent struct {
	bridge       *CommandBridge
	lastAction   Action
	lastIntent   Intent
	stateHistory []StateVector
}

// NewTacticalAgent creates a new RL tactical agent
func NewTacticalAgent(bridge *CommandBridge) *TacticalAgent {
	return &TacticalAgent{
		bridge:       bridge,
		lastAction:   ActionPassiveTrack,
		lastIntent:   IntentPassive,
		stateHistory: make([]StateVector, 0, 100),
	}
}

// Decide selects and executes the optimal tactical action
func (ta *TacticalAgent) Decide(currentState StateVector) (Action, error) {
	// Store state history
	ta.stateHistory = append(ta.stateHistory, currentState)
	if len(ta.stateHistory) > 100 {
		ta.stateHistory = ta.stateHistory[1:] // Keep last 100
	}

	// Get action from RL policy head
	response, err := ta.bridge.GetAction(currentState)
	if err != nil {
		// Fallback to rule-based decision
		return ta.fallbackDecision(currentState), nil
	}

	action := Action(response.Action)
	ta.lastAction = action
	ta.lastIntent = Intent(currentState.Intent)

	// Execute the action
	if err := ta.bridge.ExecuteAction(action, currentState); err != nil {
		return action, fmt.Errorf("failed to execute action %s: %w", ActionNames[action], err)
	}

	return action, nil
}

// fallbackDecision provides rule-based fallback when RL is unavailable
func (ta *TacticalAgent) fallbackDecision(state StateVector) Action {
	// High-confidence kinetic threat → immediate alert
	if state.Intent == int(IntentKineticThreat) && state.Confidence > 0.8 {
		return ActionTacticalAlert
	}

	// Electronic warfare with power spike → geolocation
	if state.Intent == int(IntentElectronicWarfare) && state.PowerEstimate > -20.0 {
		return ActionEmitterGeo
	}

	// RPO inspection with close approach → active scanning
	if state.Intent == int(IntentRPOInspection) && state.HCWResidual < 0.1 {
		return ActionActiveScan
	}

	// Default: passive tracking
	return ActionPassiveTrack
}

// GetLastAction returns the most recent action taken
func (ta *TacticalAgent) GetLastAction() Action {
	return ta.lastAction
}

// GetLastIntent returns the most recent intent classification
func (ta *TacticalAgent) GetLastIntent() Intent {
	return ta.lastIntent
}

// GetStateHistory returns historical states
func (ta *TacticalAgent) GetStateHistory() []StateVector {
	return ta.stateHistory
}

// Reset clears agent state
func (ta *TacticalAgent) Reset() {
	ta.lastAction = ActionPassiveTrack
	ta.lastIntent = IntentPassive
	ta.stateHistory = make([]StateVector, 0, 100)
}

// CalculateReward computes physics-informed reward for state transition
func (ta *TacticalAgent) CalculateReward(prevState, nextState StateVector, action Action) float64 {
	// R = w1*CustodyTime - w2*sum|Δv| - w3*ThreatMiss

	// Weights
	w1 := 1.0  // Custody reward
	w2 := 0.5  // Fuel penalty
	w3 := 10.0 // Threat miss penalty (heavy)

	// Custody reward (frames in custody)
	custodyReward := float64(nextState.FramesInCustody) * w1

	// Fuel penalty (recent delta-V)
	fuelPenalty := nextState.DeltaVRecent * w2

	// Threat miss penalty
	threatPenalty := 0.0
	if nextState.Intent == int(IntentKineticThreat) && action != ActionTacticalAlert && action != ActionFuelOptimalEvade {
		threatPenalty = w3 // Heavy penalty for missing kinetic threat
	}

	return custodyReward - fuelPenalty - threatPenalty
}
