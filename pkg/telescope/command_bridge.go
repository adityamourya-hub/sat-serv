package telescope

import (
	"encoding/json"
	"fmt"
	"time"

	"ssis/pkg/orbital"
	"ssis/pkg/physics"

	"github.com/pebbe/zmq4"
)

// TelescopeCommand represents a slew order
type TelescopeCommand struct {
	Type      string  `json:"type"`      // "SLEW", "TRACK", "STOP"
	Azimuth   float64 `json:"az"`        // degrees
	Elevation float64 `json:"el"`        // degrees
	AzRate    float64 `json:"vel_az"`    // deg/s (for rate tracking)
	ElRate    float64 `json:"vel_el"`    // deg/s (for rate tracking)
	TargetID  int     `json:"target_id"` // NORAD ID
	Timestamp int64   `json:"ts"`        // Unix ms (for latency calc)
	LeadTime  float64 `json:"lead_ms"`   // Compensate for processing delay
}

// CommandAcknowledgment represents telescope response
type CommandAcknowledgment struct {
	Status    string  `json:"status"` // "ACK", "ERROR", "TIMEOUT"
	Azimuth   float64 `json:"az"`     // Current mount position
	Elevation float64 `json:"el"`     // Current mount position
	Error     string  `json:"error,omitempty"`
	Mode      string  `json:"mode"` // "SIM", "INDI", "ALPACA"
}

// CommandBridge manages ZMQ REQ-REP to Python telescope controller
type CommandBridge struct {
	socket     *zmq4.Socket
	endpoint   string
	timeout    time.Duration
	stationLat float64
	stationLon float64
	stationAlt float64
	simulation bool
}

// NewCommandBridge creates a new telescope command interface
func NewCommandBridge(endpoint string, lat, lon, alt float64) (*CommandBridge, error) {
	socket, err := zmq4.NewSocket(zmq4.REQ)
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ socket: %w", err)
	}

	// Set receive timeout to prevent blocking forever
	socket.SetRcvtimeo(100 * time.Millisecond)

	if err := socket.Connect(endpoint); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to connect to %s: %w", endpoint, err)
	}

	return &CommandBridge{
		socket:     socket,
		endpoint:   endpoint,
		timeout:    100 * time.Millisecond,
		stationLat: lat,
		stationLon: lon,
		stationAlt: alt,
	}, nil
}

// Close releases ZMQ resources
func (cb *CommandBridge) Close() {
	if cb.socket != nil {
		cb.socket.Close()
	}
}

// Slew sends immediate pointing command with optional lead-time compensation
func (cb *CommandBridge) Slew(cmd TelescopeCommand) (*CommandAcknowledgment, error) {
	cmd.Timestamp = time.Now().UnixMilli()

	data, err := json.Marshal(cmd)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal command: %w", err)
	}

	if _, err := cb.socket.Send(string(data), 0); err != nil {
		return nil, fmt.Errorf("failed to send command: %w", err)
	}

	reply, err := cb.socket.Recv(0)
	if err != nil {
		return nil, fmt.Errorf("telescope timeout (mount may be busy or disconnected): %w", err)
	}

	var ack CommandAcknowledgment
	if err := json.Unmarshal([]byte(reply), &ack); err != nil {
		return nil, fmt.Errorf("failed to parse acknowledgment: %w", err)
	}

	if ack.Status != "ACK" {
		return &ack, fmt.Errorf("telescope rejected command: %s", ack.Error)
	}

	return &ack, nil
}

// PredictiveSlew calculates future Az/El for "Slew and Stare" tracking
func (cb *CommandBridge) PredictiveSlew(propagator orbital.Propagator, targetID int, leadTimeSec float64) (*CommandAcknowledgment, error) {
	// Propagate to future time (account for processing + mount lag)
	minutesAhead := leadTimeSec / 60.0
	futureState, err := propagator.Propagate(minutesAhead)
	if err != nil {
		return nil, fmt.Errorf("propagation failed: %w", err)
	}

	// Convert to Az/El
	az, el := physics.ECIToAzEl(
		futureState.Pos[0], futureState.Pos[1], futureState.Pos[2],
		cb.stationLat, cb.stationLon, cb.stationAlt,
		futureState.EpochJD,
	)

	// Check visibility
	if el < 0 {
		return nil, fmt.Errorf("target below horizon (El=%.1f°) in %.1fs", el, leadTimeSec)
	}

	// Calculate tracking rate for smooth handoff
	azRate, elRate := physics.CalculateTrackingRate(futureState, futureState, cb.stationLat, cb.stationLon, cb.stationAlt, futureState.EpochJD)

	cmd := TelescopeCommand{
		Type:      "SLEW",
		Azimuth:   az,
		Elevation: el,
		AzRate:    azRate,
		ElRate:    elRate,
		TargetID:  targetID,
		LeadTime:  leadTimeSec * 1000,
	}

	return cb.Slew(cmd)
}

// Track initiates continuous rate tracking for high-velocity objects
func (cb *CommandBridge) Track(targetID int, azRate, elRate float64) (*CommandAcknowledgment, error) {
	cmd := TelescopeCommand{
		Type:     "TRACK",
		AzRate:   azRate,
		ElRate:   elRate,
		TargetID: targetID,
	}
	return cb.Slew(cmd)
}

// Stop halts all mount motion
func (cb *CommandBridge) Stop() (*CommandAcknowledgment, error) {
	cmd := TelescopeCommand{
		Type: "STOP",
	}
	return cb.Slew(cmd)
}

// GetLatency calculates round-trip communication latency
func (cb *CommandBridge) GetLatency() (time.Duration, error) {
	start := time.Now()

	cmd := TelescopeCommand{
		Type:      "SLEW",
		Azimuth:   0,
		Elevation: 0,
	}

	_, err := cb.Slew(cmd)
	if err != nil {
		return 0, err
	}

	return time.Since(start), nil
}
