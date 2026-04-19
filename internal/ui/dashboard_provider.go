// Package ui provides the Decision Dominance Dashboard for ADITI 4.0
//
// This provider acts as a high-speed data bus between the JAX brain (Python)
// and the Go-native UI. It captures state from pkg/physics engine and latents
// from ai/v3 brain, broadcasting via WebSockets for real-time visualization.
//
// Architecture:
//   JAX Brain (Python) → Unix Socket → DashboardProvider → WebSocket → Frontend
//                        [Zero-Copy IPC]                  [100Hz broadcast]
//
// The dashboard visualizes:
//   - Geometric Manifold: 16x1o latent vectors (gravity field around HGV)
//   - Symbolic Proof: SINDy-discovered equations with L/D coefficients
//   - Safety Gradient: Astrojax imagination rollout heatmap
//
// This transforms the AI from a "Black Box" into a "Co-Processor" that the
// IAF commander can verify and trust.
//
// References:
//   - ADITI 4.0 HF-PGFM Architecture
//   - SE(3)-Equivariant Observer (ai/v3/se3_encoder.py)
//   - iDEX May 4th Demo Requirements

package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// CognitiveState represents the "Brain's" current understanding
// This is the bridge between JAX geometric latents and Go-native visualization
type CognitiveState struct {
	// SE(3) Latent Space (from ai/v3/se3_encoder.py)
	Scalars []float64    `json:"scalars"` // 16x0e Invariants (energy, altitude, |v|)
	Vectors [][3]float64 `json:"vectors"` // 16x1o Equivariants (orbital plane, thrust dir)

	// SINDy Discovered Physics (from pkg/physics/sindy.go)
	SindyCoeffs    map[string]float64 `json:"sindy_coeffs"` // Sparse coefficients Ξ
	ActiveTerms    []string           `json:"active_terms"` // Human-readable equations
	ThrustDetected bool               `json:"thrust_detected"`
	ThrustMag      float64            `json:"thrust_magnitude"` // m/s²

	// Model Confidence & Uncertainty
	Confidence   float64 `json:"confidence"`    // R² fit quality
	EpistemicUnc float64 `json:"epistemic_unc"` // Model uncertainty
	AleatoricUnc float64 `json:"aleatoric_unc"` // Sensor noise

	// Intent Classification (from pkg/threat/classifier.go)
	CurrentIntent string  `json:"current_intent"` // Saturn, Mars, Rahu, etc.
	IntentProb    float64 `json:"intent_prob"`    // Probability [0,1]
	LDRatio       float64 `json:"ld_ratio"`       // Lift-to-drag (hypersonic)

	// Temporal Metadata
	Timestamp        int64   `json:"timestamp"`          // Unix nanoseconds
	FrameID          uint64  `json:"frame_id"`           // Monotonic counter
	ProcessingTimeMs float64 `json:"processing_time_ms"` // End-to-end latency
}

// DashboardMetrics contains system performance data
type DashboardMetrics struct {
	// Latency Budget Tracking (100ms total)
	S2LatencyMs        float64 `json:"s2_latency_ms"`         // S2 conjunction detection
	SE3EncodeLatencyMs float64 `json:"se3_encode_latency_ms"` // SE(3) encoding
	SINDyLatencyMs     float64 `json:"sindy_latency_ms"`      // Maneuver detection
	MBPOLatencyMs      float64 `json:"mbpo_latency_ms"`       // Imagination
	TotalLatencyMs     float64 `json:"total_latency_ms"`      // End-to-end

	// Throughput
	ImaginationRateHz float64 `json:"imagination_rate_hz"` // Trajectories/sec
	SE3BatchSize      int     `json:"se3_batch_size"`      // Parallel encodings

	// Thermal & Power (M2 Air monitoring)
	CPUTempCelsius float64 `json:"cpu_temp_celsius"`
	GPUUtilization float64 `json:"gpu_utilization"` // Metal API
	MemoryPressure string  `json:"memory_pressure"` // normal/warning/critical
}

// DashboardProvider manages the real-time telemetry stream
// This is the "Voice" of the AI - making the Black Box transparent
type DashboardProvider struct {
	mu sync.RWMutex

	// WebSocket clients (frontend connections)
	Clients   map[*websocket.Conn]bool
	ClientsMu sync.RWMutex

	// Current state (updated by JAX brain via IPC)
	CurrentState   CognitiveState
	CurrentMetrics DashboardMetrics

	// IPC for JAX brain communication
	UnixSocketPath string
	Listener       net.Listener

	// Broadcast channels
	StateBroadcast  chan []byte
	MetricBroadcast chan []byte

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Upgrader for WebSocket connections
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// In production, validate origin for security
		return true
	},
}

// NewDashboardProvider creates a new dashboard provider
func NewDashboardProvider(socketPath string) *DashboardProvider {
	ctx, cancel := context.WithCancel(context.Background())

	return &DashboardProvider{
		Clients:         make(map[*websocket.Conn]bool),
		UnixSocketPath:  socketPath,
		StateBroadcast:  make(chan []byte, 100),
		MetricBroadcast: make(chan []byte, 100),
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Start initializes the Unix socket listener and begins broadcasting
func (dp *DashboardProvider) Start() error {
	// Start Unix socket listener for JAX brain IPC
	if err := dp.startUnixSocket(); err != nil {
		return fmt.Errorf("failed to start Unix socket: %w", err)
	}

	// Start 100Hz broadcast loop
	dp.wg.Add(1)
	go dp.broadcastLoop()

	// Start heartbeat for latency monitoring
	dp.wg.Add(1)
	go dp.heartbeatLoop()

	log.Printf("[Dashboard] Provider started on %s", dp.UnixSocketPath)
	log.Printf("[Dashboard] WebSocket endpoint: ws://localhost:8080/ws")

	return nil
}

// startUnixSocket creates the Unix domain socket for JAX brain communication
func (dp *DashboardProvider) startUnixSocket() error {
	// Remove existing socket file
	if err := os.Remove(dp.UnixSocketPath); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Create Unix domain socket
	listener, err := net.Listen("unix", dp.UnixSocketPath)
	if err != nil {
		return err
	}

	dp.Listener = listener

	// Accept connections in background
	dp.wg.Add(1)
	go dp.acceptJAXConnections()

	return nil
}

// acceptJAXConnections handles incoming connections from Python/JAX process
func (dp *DashboardProvider) acceptJAXConnections() {
	defer dp.wg.Done()

	for {
		select {
		case <-dp.ctx.Done():
			return
		default:
		}

		conn, err := dp.Listener.Accept()
		if err != nil {
			if dp.ctx.Err() != nil {
				return
			}
			log.Printf("[Dashboard] Accept error: %v", err)
			continue
		}

		dp.wg.Add(1)
		go dp.handleJAXConnection(conn)
	}
}

// handleJAXConnection processes messages from the JAX brain
func (dp *DashboardProvider) handleJAXConnection(conn net.Conn) {
	defer dp.wg.Done()
	defer conn.Close()

	log.Printf("[Dashboard] JAX brain connected via Unix socket")

	buffer := make([]byte, 65536) // 64KB buffer for protobuf messages

	for {
		select {
		case <-dp.ctx.Done():
			return
		default:
		}

		n, err := conn.Read(buffer)
		if err != nil {
			if dp.ctx.Err() != nil {
				return
			}
			log.Printf("[Dashboard] JAX read error: %v", err)
			return
		}

		// Parse JSON message from JAX brain (newline-delimited JSON)
		var state CognitiveState
		if err := json.Unmarshal(buffer[:n], &state); err != nil {
			log.Printf("[Dashboard] JSON unmarshal error: %v", err)
			continue
		}

		// Update state and broadcast
		dp.UpdateBrainState(state)
	}
}

// UpdateBrainState is called when the JAX brain pushes new latents
func (dp *DashboardProvider) UpdateBrainState(state CognitiveState) {
	dp.mu.Lock()
	dp.CurrentState = state
	dp.mu.Unlock()

	// Marshal to JSON for WebSocket broadcast
	msg, err := json.Marshal(state)
	if err != nil {
		log.Printf("[Dashboard] JSON marshal error: %v", err)
		return
	}

	// Non-blocking send to broadcast channel
	select {
	case dp.StateBroadcast <- msg:
	default:
		log.Printf("[Dashboard] State broadcast channel full, dropping frame")
	}
}

// UpdateMetrics updates system performance metrics
func (dp *DashboardProvider) UpdateMetrics(metrics DashboardMetrics) {
	dp.mu.Lock()
	dp.CurrentMetrics = metrics
	dp.mu.Unlock()

	msg, err := json.Marshal(metrics)
	if err != nil {
		log.Printf("[Dashboard] Metrics JSON marshal error: %v", err)
		return
	}

	select {
	case dp.MetricBroadcast <- msg:
	default:
	}
}

// broadcastLoop runs the 100Hz WebSocket broadcast
func (dp *DashboardProvider) broadcastLoop() {
	defer dp.wg.Done()

	ticker := time.NewTicker(10 * time.Millisecond) // 100Hz
	defer ticker.Stop()

	for {
		select {
		case <-dp.ctx.Done():
			return

		case msg := <-dp.StateBroadcast:
			dp.broadcastToClients(msg)

		case msg := <-dp.MetricBroadcast:
			dp.broadcastToClients(msg)

		case <-ticker.C:
			// Heartbeat: send current state even if no updates
			// This ensures 100Hz refresh for smooth UI
			dp.mu.RLock()
			state := dp.CurrentState
			dp.mu.RUnlock()

			msg, err := json.Marshal(state)
			if err == nil {
				dp.broadcastToClients(msg)
			}
		}
	}
}

// broadcastToClients sends message to all connected WebSocket clients
func (dp *DashboardProvider) broadcastToClients(msg []byte) {
	dp.ClientsMu.RLock()
	clients := make([]*websocket.Conn, 0, len(dp.Clients))
	for client := range dp.Clients {
		clients = append(clients, client)
	}
	dp.ClientsMu.RUnlock()

	for _, client := range clients {
		err := client.WriteMessage(websocket.TextMessage, msg)
		if err != nil {
			// Remove failed client
			dp.ClientsMu.Lock()
			delete(dp.Clients, client)
			dp.ClientsMu.Unlock()
			client.Close()
			log.Printf("[Dashboard] Client disconnected, removed from pool")
		}
	}
}

// heartbeatLoop monitors system health and logs performance
func (dp *DashboardProvider) heartbeatLoop() {
	defer dp.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-dp.ctx.Done():
			return
		case <-ticker.C:
			dp.mu.RLock()
			state := dp.CurrentState
			metrics := dp.CurrentMetrics
			dp.mu.RUnlock()

			log.Printf("[Dashboard] Heartbeat | Frame: %d | Latency: %.2fms | Intent: %s | Conf: %.2f",
				state.FrameID,
				metrics.TotalLatencyMs,
				state.CurrentIntent,
				state.Confidence,
			)
		}
	}
}

// HandleWebSocket upgrades HTTP connections to WebSocket for frontend
func (dp *DashboardProvider) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[Dashboard] WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	// Register client
	dp.ClientsMu.Lock()
	dp.Clients[conn] = true
	clientCount := len(dp.Clients)
	dp.ClientsMu.Unlock()

	log.Printf("[Dashboard] WebSocket client connected (total: %d)", clientCount)

	// Send initial state
	dp.mu.RLock()
	initialState := dp.CurrentState
	dp.mu.RUnlock()

	msg, _ := json.Marshal(initialState)
	conn.WriteMessage(websocket.TextMessage, msg)

	// Keep connection alive and handle client messages
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("[Dashboard] WebSocket error: %v", err)
			}
			break
		}
	}

	// Unregister client
	dp.ClientsMu.Lock()
	delete(dp.Clients, conn)
	dp.ClientsMu.Unlock()

	log.Printf("[Dashboard] WebSocket client disconnected (remaining: %d)", len(dp.Clients))
}

// GetCurrentState returns the latest cognitive state (thread-safe)
func (dp *DashboardProvider) GetCurrentState() CognitiveState {
	dp.mu.RLock()
	defer dp.mu.RUnlock()
	return dp.CurrentState
}

// GetCurrentMetrics returns the latest system metrics (thread-safe)
func (dp *DashboardProvider) GetCurrentMetrics() DashboardMetrics {
	dp.mu.RLock()
	defer dp.mu.RUnlock()
	return dp.CurrentMetrics
}

// Stop gracefully shuts down the provider
func (dp *DashboardProvider) Stop() error {
	log.Printf("[Dashboard] Stopping provider...")

	dp.cancel()

	if dp.Listener != nil {
		dp.Listener.Close()
	}

	// Close all WebSocket clients
	dp.ClientsMu.Lock()
	for client := range dp.Clients {
		client.Close()
		delete(dp.Clients, client)
	}
	dp.ClientsMu.Unlock()

	// Wait for goroutines to finish
	dp.wg.Wait()

	log.Printf("[Dashboard] Provider stopped")
	return nil
}
