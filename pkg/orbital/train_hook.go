// Package orbital provides training data streaming to Python ML pipeline
// Implements Go-side ingestor for neural-physics training
package orbital

import (
	"fmt"
	"math"
	"time"

	"github.com/pebbe/zmq4"
)

// TrainingSocket wraps ZMQ socket for training data streaming
type TrainingSocket struct {
	socket   *zmq4.Socket
	endpoint string
	isOpen   bool
}

// NewTrainingSocket creates a new training data socket
func NewTrainingSocket(endpoint string) (*TrainingSocket, error) {
	socket, err := zmq4.NewSocket(zmq4.PUSH)
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ socket: %w", err)
	}

	// Set high water mark to prevent blocking
	socket.SetSndhwm(1000)

	// Connect to Python training socket
	if err := socket.Connect(endpoint); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to connect to %s: %w", endpoint, err)
	}

	return &TrainingSocket{
		socket:   socket,
		endpoint: endpoint,
		isOpen:   true,
	}, nil
}

// Close closes the training socket
func (ts *TrainingSocket) Close() error {
	if ts.isOpen {
		ts.isOpen = false
		return ts.socket.Close()
	}
	return nil
}

// SendTrainingData pushes propagated Truth States to Python
// Format: ID|X|Y|Z|VX|VY|VZ|timestamp
func (ts *TrainingSocket) SendTrainingData(satID int, state PropagatorState) error {
	if !ts.isOpen {
		return fmt.Errorf("training socket is closed")
	}

	// Format message
	msg := fmt.Sprintf("%d|%f|%f|%f|%f|%f|%f|%d",
		satID,
		state.X,
		state.Y,
		state.Z,
		state.VX,
		state.VY,
		state.VZ,
		time.Now().UnixMilli())

	// Push to the Python Training Socket
	_, err := ts.socket.Send(msg, 0)
	return err
}

// PropagatorState represents orbital state for training
type PropagatorState struct {
	X  float64 // Position X (m)
	Y  float64 // Position Y (m)
	Z  float64 // Position Z (m)
	VX float64 // Velocity X (m/s)
	VY float64 // Velocity Y (m/s)
	VZ float64 // Velocity Z (m/s)
}

// TrainingPropagator interface for training data generation
type TrainingPropagator interface {
	GetLatestState(satID int) PropagatorState
	PropagateState(satID int, dt float64) PropagatorState
}

// TrainingStreamer manages training data streaming from HPOP to Python
type TrainingStreamer struct {
	propagator TrainingPropagator
	socket     *TrainingSocket
	isRunning  bool
	batchSize  int
	sampleRate float64 // Hz
}

// NewTrainingStreamer creates a new training data streamer
func NewTrainingStreamer(propagator TrainingPropagator, endpoint string) (*TrainingStreamer, error) {
	socket, err := NewTrainingSocket(endpoint)
	if err != nil {
		return nil, err
	}

	return &TrainingStreamer{
		propagator: propagator,
		socket:     socket,
		batchSize:  32,
		sampleRate: 10.0, // 10 Hz default
	}, nil
}

// SetBatchSize configures the batch size for training data
func (ts *TrainingStreamer) SetBatchSize(size int) {
	ts.batchSize = size
}

// SetSampleRate configures the sampling rate in Hz
func (ts *TrainingStreamer) SetSampleRate(rate float64) {
	ts.sampleRate = rate
}

// Start begins streaming training data for a satellite
func (ts *TrainingStreamer) Start(satID int) error {
	if ts.isRunning {
		return fmt.Errorf("streamer already running")
	}

	ts.isRunning = true
	interval := time.Duration(1000.0/ts.sampleRate) * time.Millisecond

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for ts.isRunning {
			select {
			case <-ticker.C:
				state := ts.propagator.GetLatestState(satID)
				if err := ts.socket.SendTrainingData(satID, state); err != nil {
					// Log error but continue streaming
					fmt.Printf("[TRAIN_STREAM] Error sending data: %v\n", err)
				}
			}
		}
	}()

	fmt.Printf("[TRAIN_STREAM] Started streaming for sat %d at %.1f Hz\n", satID, ts.sampleRate)
	return nil
}

// Stop halts training data streaming
func (ts *TrainingStreamer) Stop() {
	ts.isRunning = false
	time.Sleep(100 * time.Millisecond) // Allow goroutine to finish
}

// Close shuts down the training streamer
func (ts *TrainingStreamer) Close() error {
	ts.Stop()
	return ts.socket.Close()
}

// StreamTrainingBatch sends a batch of training data points
func (ts *TrainingStreamer) StreamTrainingBatch(satID int, numPoints int) error {
	for i := 0; i < numPoints; i++ {
		// Propagate to get new state
		state := ts.propagator.PropagateState(satID, 1.0/ts.sampleRate)

		if err := ts.socket.SendTrainingData(satID, state); err != nil {
			return fmt.Errorf("failed to send point %d: %w", i, err)
		}

		// Small delay to match sample rate
		time.Sleep(time.Duration(1000.0/ts.sampleRate) * time.Millisecond)
	}

	fmt.Printf("[TRAIN_STREAM] Sent %d training points for sat %d\n", numPoints, satID)
	return nil
}

// HPOPTrainingPropagator wraps existing HPOP for training data access
type HPOPTrainingPropagator struct {
	// Reference to existing HPOP engine
	satellites map[int]*SatelliteState
	mu         float64 // Gravitational parameter
}

// SatelliteState tracks state for a single satellite
type SatelliteState struct {
	Position  [3]float64 // km
	Velocity  [3]float64 // km/s
	Timestamp time.Time
}

// NewHPOPTrainingPropagator creates a new HPOP-based propagator for training
func NewHPOPTrainingPropagator() *HPOPTrainingPropagator {
	return &HPOPTrainingPropagator{
		satellites: make(map[int]*SatelliteState),
		mu:         3.986004418e14, // m³/s²
	}
}

// AddSatellite registers a satellite for propagation
func (hp *HPOPTrainingPropagator) AddSatellite(satID int, pos, vel [3]float64) {
	hp.satellites[satID] = &SatelliteState{
		Position:  pos,
		Velocity:  vel,
		Timestamp: time.Now(),
	}
}

// GetLatestState implements TrainingPropagator interface
func (hp *HPOPTrainingPropagator) GetLatestState(satID int) PropagatorState {
	sat, exists := hp.satellites[satID]
	if !exists {
		return PropagatorState{}
	}

	return PropagatorState{
		X:  sat.Position[0] * 1000, // Convert km to m
		Y:  sat.Position[1] * 1000,
		Z:  sat.Position[2] * 1000,
		VX: sat.Velocity[0] * 1000, // Convert km/s to m/s
		VY: sat.Velocity[1] * 1000,
		VZ: sat.Velocity[2] * 1000,
	}
}

// PropagateState implements TrainingPropagator interface with Keplerian dynamics
func (hp *HPOPTrainingPropagator) PropagateState(satID int, dt float64) PropagatorState {
	sat, exists := hp.satellites[satID]
	if !exists {
		return PropagatorState{}
	}

	// Convert to meters for calculation
	x := sat.Position[0] * 1000
	y := sat.Position[1] * 1000
	z := sat.Position[2] * 1000
	vx := sat.Velocity[0] * 1000
	vy := sat.Velocity[1] * 1000
	vz := sat.Velocity[2] * 1000

	// Calculate gravitational acceleration
	r := math.Sqrt(x*x + y*y + z*z)
	ax := -hp.mu * x / (r * r * r)
	ay := -hp.mu * y / (r * r * r)
	az := -hp.mu * z / (r * r * r)

	// Simple Euler integration (in production, use RK4 or symplectic integrator)
	vx += ax * dt
	vy += ay * dt
	vz += az * dt
	x += vx * dt
	y += vy * dt
	z += vz * dt

	// Update stored state
	sat.Position = [3]float64{x / 1000, y / 1000, z / 1000}
	sat.Velocity = [3]float64{vx / 1000, vy / 1000, vz / 1000}
	sat.Timestamp = time.Now()

	return PropagatorState{
		X: x, Y: y, Z: z,
		VX: vx, VY: vy, VZ: vz,
	}
}

// StreamTrainingData is the main entry point for training data streaming
// Can be called from the main engine to start streaming for a satellite
func StreamTrainingData(engine interface{}, satID int, endpoint string) error {
	// Type assert to get propagator
	// In production, this would integrate with existing engine

	// Create HPOP propagator
	propagator := NewHPOPTrainingPropagator()

	// Add initial state (would come from engine)
	propagator.AddSatellite(satID, [3]float64{6800, 0, 0}, [3]float64{0, 7.5, 0})

	// Create streamer
	streamer, err := NewTrainingStreamer(propagator, endpoint)
	if err != nil {
		return err
	}
	defer streamer.Close()

	// Start streaming
	return streamer.Start(satID)
}
