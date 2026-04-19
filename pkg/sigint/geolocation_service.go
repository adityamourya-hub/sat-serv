// Package sigint provides high-level geolocation service coordinating TDOA across mesh nodes
// Handles detection → timestamp exchange → computation → target brief generation
package sigint

import (
	"fmt"
	"sync"
	"time"
)

// TDOATimestamp represents a pulse arrival timestamp from a satellite/mesh node
type TDOATimestamp struct {
	NodeID      string     // Satellite/node identifier
	PulseID     string     // Unique pulse identifier
	ArrivalTime float64    // GPSDO timestamp (nanoseconds since Unix epoch)
	Frequency   float64    // Detected frequency (Hz)
	SNR         float64    // Signal-to-noise ratio (dB)
	Confidence  float64    // Detection confidence (0.0-1.0)
	Position    [3]float64 // ECEF position at time of arrival (meters)
	Timestamp   time.Time  // Record timestamp
}

// TDOARequest requests TDOA participation from other mesh nodes
type TDOARequest struct {
	PulseID         string        // Unique pulse/event ID
	LeadNode        string        // Node that detected the pulse
	LeadTime        float64       // Arrival time at lead node
	Frequency       float64       // Detected frequency
	SNR             float64       // Signal quality
	RequestTime     time.Time     // When request was sent
	ResponseTimeout time.Duration // How long to wait for responses
}

// TDOAResponse contains timestamp from a responding node
type TDOAResponse struct {
	PulseID     string
	NodeID      string
	ArrivalTime float64
	Position    [3]float64 // Node position at arrival time
	SNR         float64
	Confidence  float64
	Latency     time.Duration // Network latency
}

// MeshGeolocationResult wraps TDOAResult with mesh metadata
type MeshGeolocationResult struct {
	*TDOAResult
	PulseID        string
	LeadNode       string
	ResponderNodes []string
	ComputeTime    time.Duration
	MeshLatency    time.Duration
}

// GeolocationService coordinates TDOA geolocation across the mesh network
type GeolocationService struct {
	nodeID     string
	solver     *TDOASolver
	timestamps map[string][]TDOATimestamp        // PulseID -> timestamps
	requests   map[string]*TDOARequest           // PulseID -> pending requests
	results    map[string]*MeshGeolocationResult // PulseID -> computed results

	// Callbacks
	onGeolocationComplete func(*MeshGeolocationResult)
	onTimestampReceived   func(TDOATimestamp)

	// Synchronization
	mu sync.RWMutex

	// Configuration
	maxWaitTime         time.Duration
	minResponders       int
	confidenceThreshold float64
}

// NewGeolocationService creates a new geolocation service
func NewGeolocationService(nodeID string) *GeolocationService {
	return &GeolocationService{
		nodeID:              nodeID,
		solver:              NewTDOASolver(),
		timestamps:          make(map[string][]TDOATimestamp),
		requests:            make(map[string]*TDOARequest),
		results:             make(map[string]*MeshGeolocationResult),
		maxWaitTime:         5 * time.Second,
		minResponders:       2, // Minimum 2 other nodes for TDOA
		confidenceThreshold: 0.7,
	}
}

// OnJammingDetected handles new jamming detection from local receiver
// Called when local SDR detects a jamming pulse
func (gs *GeolocationService) OnJammingDetected(freq float64, arrivalTime float64, snr float64, position [3]float64) (*TDOARequest, error) {
	// Generate unique pulse ID
	pulseID := fmt.Sprintf("PULSE-%d-%s", int(arrivalTime), gs.nodeID)

	// Record own timestamp
	timestamp := TDOATimestamp{
		NodeID:      gs.nodeID,
		PulseID:     pulseID,
		ArrivalTime: arrivalTime,
		Frequency:   freq,
		SNR:         snr,
		Confidence:  calculateTimestampConfidence(snr),
		Position:    position,
		Timestamp:   time.Now(),
	}

	gs.mu.Lock()
	gs.timestamps[pulseID] = []TDOATimestamp{timestamp}
	gs.mu.Unlock()

	// Create TDOA request for other nodes
	request := &TDOARequest{
		PulseID:         pulseID,
		LeadNode:        gs.nodeID,
		LeadTime:        arrivalTime,
		Frequency:       freq,
		SNR:             snr,
		RequestTime:     time.Now(),
		ResponseTimeout: gs.maxWaitTime,
	}

	gs.mu.Lock()
	gs.requests[pulseID] = request
	gs.mu.Unlock()

	return request, nil
}

// ProcessTDOARequest processes an incoming TDOA request from another node
// Called when receiving a request via LXMF
func (gs *GeolocationService) ProcessTDOARequest(request TDOARequest) (*TDOAResponse, error) {
	// Check if we detected this pulse
	gs.mu.RLock()
	localTimestamps, exists := gs.timestamps[request.PulseID]
	gs.mu.RUnlock()

	if !exists {
		// We didn't detect this pulse
		return nil, fmt.Errorf("pulse %s not detected locally", request.PulseID)
	}

	// Find matching timestamp (should be first one for this pulse)
	if len(localTimestamps) == 0 {
		return nil, fmt.Errorf("no timestamps for pulse %s", request.PulseID)
	}

	ts := localTimestamps[0]

	// Validate frequency match (within tolerance)
	freqDiff := abs(ts.Frequency - request.Frequency)
	if freqDiff > 100000 { // 100 kHz tolerance
		return nil, fmt.Errorf("frequency mismatch: local=%f, request=%f", ts.Frequency, request.Frequency)
	}

	// Create response
	response := &TDOAResponse{
		PulseID:     request.PulseID,
		NodeID:      gs.nodeID,
		ArrivalTime: ts.ArrivalTime,
		Position:    ts.Position,
		SNR:         ts.SNR,
		Confidence:  ts.Confidence,
		Latency:     time.Since(request.RequestTime),
	}

	return response, nil
}

// ProcessTDOAResponse processes a TDOA response from another node
// Called when receiving a response via LXMF
func (gs *GeolocationService) ProcessTDOAResponse(response TDOAResponse) error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	// Check if we have a pending request for this pulse
	request, exists := gs.requests[response.PulseID]
	if !exists {
		return fmt.Errorf("no pending request for pulse %s", response.PulseID)
	}

	// Record timestamp
	timestamp := TDOATimestamp{
		NodeID:      response.NodeID,
		PulseID:     response.PulseID,
		ArrivalTime: response.ArrivalTime,
		Frequency:   request.Frequency,
		SNR:         response.SNR,
		Confidence:  response.Confidence,
		Position:    response.Position,
		Timestamp:   time.Now(),
	}

	gs.timestamps[response.PulseID] = append(gs.timestamps[response.PulseID], timestamp)

	// Notify callback
	if gs.onTimestampReceived != nil {
		gs.onTimestampReceived(timestamp)
	}

	return nil
}

// ComputeGeolocation performs TDOA geolocation as lead node
// Called after collecting enough timestamps
func (gs *GeolocationService) ComputeGeolocation(pulseID string) (*MeshGeolocationResult, error) {
	gs.mu.RLock()
	request, reqExists := gs.requests[pulseID]
	timestamps, tsExists := gs.timestamps[pulseID]
	gs.mu.RUnlock()

	if !reqExists {
		return nil, fmt.Errorf("no request found for pulse %s", pulseID)
	}

	if !tsExists || len(timestamps) < 2 {
		return nil, fmt.Errorf("insufficient timestamps for pulse %s (need at least 2, have %d)", pulseID, len(timestamps))
	}

	startTime := time.Now()

	// Build satellite positions and TDOA measurements
	satellites := make(map[string][3]float64)
	tdoaMeasurements := make(map[string]float64)

	// Lead node is the reference
	leadTimestamp := timestamps[0]
	leadTime := leadTimestamp.ArrivalTime
	satellites[leadTimestamp.NodeID] = leadTimestamp.Position

	// Build TDOA measurements relative to lead
	responderNodes := make([]string, 0)
	for i := 1; i < len(timestamps); i++ {
		ts := timestamps[i]
		satellites[ts.NodeID] = ts.Position
		responderNodes = append(responderNodes, ts.NodeID)

		// Delta T = T_other - T_lead
		deltaT := ts.ArrivalTime - leadTime
		key := fmt.Sprintf("%s-%s", leadTimestamp.NodeID, ts.NodeID)
		tdoaMeasurements[key] = deltaT
	}

	// Initial guess: subsatellite point of lead node
	initialLat, initialLon := gs.solver.SubSatellitePoint(leadTimestamp.Position)
	initialGuess := NewWGS84Ellipsoid().GeodeticToECEF(initialLat, initialLon, 0)

	// Select appropriate solver method
	var result *TDOAResult
	var err error

	numMeasurements := len(tdoaMeasurements)
	if numMeasurements >= 2 && len(timestamps) >= 3 {
		// Use 3-satellite pinpoint solution
		var sat1, sat2, sat3 [3]float64
		var deltaT12, deltaT13 float64

		sat1 = timestamps[0].Position
		sat2 = timestamps[1].Position
		sat3 = timestamps[2].Position
		deltaT12 = timestamps[1].ArrivalTime - timestamps[0].ArrivalTime
		deltaT13 = timestamps[2].ArrivalTime - timestamps[0].ArrivalTime

		result, err = gs.solver.Geolocate3Sat(sat1, sat2, sat3, deltaT12, deltaT13, initialGuess)
	} else if numMeasurements >= 1 {
		// Use 2-satellite hyperbola solution
		sat1 := timestamps[0].Position
		sat2 := timestamps[1].Position
		deltaT := timestamps[1].ArrivalTime - timestamps[0].ArrivalTime

		result, err = gs.solver.Geolocate2Sat(sat1, sat2, deltaT, initialGuess)
	} else {
		return nil, fmt.Errorf("insufficient measurements for geolocation")
	}

	if err != nil {
		return nil, fmt.Errorf("geolocation failed: %w", err)
	}

	computeTime := time.Since(startTime)

	geoResult := &MeshGeolocationResult{
		TDOAResult:     result,
		PulseID:        pulseID,
		LeadNode:       request.LeadNode,
		ResponderNodes: responderNodes,
		ComputeTime:    computeTime,
		MeshLatency:    time.Since(request.RequestTime),
	}

	// Store result
	gs.mu.Lock()
	gs.results[pulseID] = geoResult
	gs.mu.Unlock()

	// Notify callback
	if gs.onGeolocationComplete != nil {
		gs.onGeolocationComplete(geoResult)
	}

	return geoResult, nil
}

// WaitForGeolocation waits for and returns geolocation result
// Blocks until result available or timeout
func (gs *GeolocationService) WaitForGeolocation(pulseID string, timeout time.Duration) (*MeshGeolocationResult, error) {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		gs.mu.RLock()
		result, exists := gs.results[pulseID]
		gs.mu.RUnlock()

		if exists {
			return result, nil
		}

		time.Sleep(100 * time.Millisecond)
	}

	return nil, fmt.Errorf("timeout waiting for geolocation result for pulse %s", pulseID)
}

// GetTimestamps returns all timestamps for a pulse
func (gs *GeolocationService) GetTimestamps(pulseID string) []TDOATimestamp {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	return gs.timestamps[pulseID]
}

// GetResult returns a computed geolocation result
func (gs *GeolocationService) GetResult(pulseID string) (*MeshGeolocationResult, bool) {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	result, exists := gs.results[pulseID]
	return result, exists
}

// SetOnGeolocationComplete sets callback for geolocation completion
func (gs *GeolocationService) SetOnGeolocationComplete(callback func(*MeshGeolocationResult)) {
	gs.onGeolocationComplete = callback
}

// SetOnTimestampReceived sets callback for timestamp receipt
func (gs *GeolocationService) SetOnTimestampReceived(callback func(TDOATimestamp)) {
	gs.onTimestampReceived = callback
}

// SetMaxWaitTime sets maximum time to wait for TDOA responses
func (gs *GeolocationService) SetMaxWaitTime(duration time.Duration) {
	gs.maxWaitTime = duration
}

// SetMinResponders sets minimum number of responder nodes required
func (gs *GeolocationService) SetMinResponders(n int) {
	gs.minResponders = n
}

// ClearPulseData clears all data for a pulse
func (gs *GeolocationService) ClearPulseData(pulseID string) {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	delete(gs.timestamps, pulseID)
	delete(gs.requests, pulseID)
	delete(gs.results, pulseID)
}

// GetPulseCount returns number of tracked pulses
func (gs *GeolocationService) GetPulseCount() int {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	return len(gs.timestamps)
}

// GetServiceStats returns service statistics
func (gs *GeolocationService) GetServiceStats() map[string]interface{} {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	totalPulses := len(gs.timestamps)
	completedPulses := len(gs.results)
	pendingPulses := len(gs.requests) - completedPulses

	return map[string]interface{}{
		"node_id":        gs.nodeID,
		"total_pulses":   totalPulses,
		"completed":      completedPulses,
		"pending":        pendingPulses,
		"max_wait_time":  gs.maxWaitTime,
		"min_responders": gs.minResponders,
	}
}

// calculateTimestampConfidence calculates confidence from SNR
func calculateTimestampConfidence(snr float64) float64 {
	// Higher SNR = higher confidence
	// SNR > 20 dB → confidence ~1.0
	// SNR < 0 dB → confidence ~0.0
	confidence := (snr + 10.0) / 30.0
	if confidence > 1.0 {
		confidence = 1.0
	}
	if confidence < 0.0 {
		confidence = 0.0
	}
	return confidence
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// SiliguriScenario demonstrates a tactical scenario for Siliguri Corridor
func (gs *GeolocationService) SiliguriScenario() (*MeshGeolocationResult, error) {
	// Simulated satellite positions (LEO satellites over Siliguri)
	// Coordinates around 26.7°N, 88.3°E (Siliguri area)

	// Sat-A: Lead node (this node)
	satAPosition := [3]float64{
		-1540.0 * 1000, // x (km to m)
		5500.0 * 1000,  // y
		3200.0 * 1000,  // z
	}

	// Sat-B: Responder node 1
	satBPosition := [3]float64{
		-1200.0 * 1000,
		5800.0 * 1000,
		2800.0 * 1000,
	}

	// Sat-C: Responder node 2
	satCPosition := [3]float64{
		-1800.0 * 1000,
		5200.0 * 1000,
		3500.0 * 1000,
	}

	// Simulated jammer position (ground, near Siliguri)
	// 26.7°N, 88.3°E, 0m altitude
	jammerLat := 26.7
	jammerLon := 88.3
	jammerAlt := 0.0

	jammerECEF := NewWGS84Ellipsoid().GeodeticToECEF(jammerLat, jammerLon, jammerAlt)

	// Calculate arrival times at each satellite
	distA := distance3D(jammerECEF, satAPosition)
	distB := distance3D(jammerECEF, satBPosition)
	distC := distance3D(jammerECEF, satCPosition)

	timeA := distA / GeoSpeedOfLight
	timeB := distB / GeoSpeedOfLight
	timeC := distC / GeoSpeedOfLight

	// Lead node detects pulse
	freq := 1575.42e6 // GPS L1 frequency (jamming)
	snr := 25.0       // dB

	request, err := gs.OnJammingDetected(freq, timeA, snr, satAPosition)
	if err != nil {
		return nil, fmt.Errorf("lead detection failed: %w", err)
	}

	// Simulate responses from other nodes
	responseB := TDOAResponse{
		PulseID:     request.PulseID,
		NodeID:      "SAT-B",
		ArrivalTime: timeB,
		Position:    satBPosition,
		SNR:         22.0,
		Confidence:  0.85,
		Latency:     50 * time.Millisecond,
	}

	responseC := TDOAResponse{
		PulseID:     request.PulseID,
		NodeID:      "SAT-C",
		ArrivalTime: timeC,
		Position:    satCPosition,
		SNR:         20.0,
		Confidence:  0.75,
		Latency:     60 * time.Millisecond,
	}

	// Process responses
	if err := gs.ProcessTDOAResponse(responseB); err != nil {
		return nil, fmt.Errorf("failed to process response B: %w", err)
	}

	if err := gs.ProcessTDOAResponse(responseC); err != nil {
		return nil, fmt.Errorf("failed to process response C: %w", err)
	}

	// Compute geolocation
	result, err := gs.ComputeGeolocation(request.PulseID)
	if err != nil {
		return nil, fmt.Errorf("geolocation computation failed: %w", err)
	}

	return result, nil
}
