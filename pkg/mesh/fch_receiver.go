// Package mesh provides Fire-Control Handover (FCH) receiver for interceptors
// Implements FCH packet reception and state vector extraction for Kalman Filter input
package mesh

import (
	"fmt"
	"time"
)

// FCHReceiver handles incoming Fire-Control Handover packets
// Runs on interceptor side to receive and process engagement data
type FCHReceiver struct {
	receivedHandovers map[string]*FireControlBrief
	lastReceivedTime  time.Time
	statistics        *FCHReceptionStats
}

// FCHReceptionStats tracks reception statistics
type FCHReceptionStats struct {
	TotalReceived     int64
	ValidReceived     int64
	InvalidReceived   int64
	LateArrivals      int64 // Packets received after engagement window
	AverageLatencyMs  float64
	LastReceptionTime time.Time
	PendingHandovers  map[string]*FireControlBrief
}

// NewFCHReceiver creates a new FCH receiver
func NewFCHReceiver() *FCHReceiver {
	return &FCHReceiver{
		receivedHandovers: make(map[string]*FireControlBrief),
		statistics: &FCHReceptionStats{
			PendingHandovers: make(map[string]*FireControlBrief),
		},
	}
}

// ReceiveFireControlBrief parses incoming FCH packet
// This is the main entry point for interceptor reception
func (r *FCHReceiver) ReceiveFireControlBrief(data []byte) (*FireControlBrief, error) {
	// Deserialize FCH
	serializer := NewFCHSerializer()
	brief, err := serializer.Deserialize(data)
	if err != nil {
		r.statistics.InvalidReceived++
		r.statistics.TotalReceived++
		return nil, fmt.Errorf("fch deserialization failed: %w", err)
	}

	// Validate timing
	latency := time.Since(brief.Timestamp)
	latencyMs := float64(latency.Milliseconds())

	// Update statistics
	r.statistics.TotalReceived++
	r.statistics.ValidReceived++
	r.updateLatencyStats(latencyMs)
	r.statistics.LastReceptionTime = time.Now()

	// Store handover
	r.receivedHandovers[brief.HandoverID] = brief
	r.statistics.PendingHandovers[brief.HandoverID] = brief

	// Check if packet is late (> 30 seconds old)
	if latency > 30*time.Second {
		r.statistics.LateArrivals++
		return brief, fmt.Errorf("late arrival: packet is %v old", latency)
	}

	return brief, nil
}

// updateLatencyStats updates rolling average latency
func (r *FCHReceiver) updateLatencyStats(latencyMs float64) {
	n := float64(r.statistics.ValidReceived)
	r.statistics.AverageLatencyMs = (r.statistics.AverageLatencyMs*(n-1) + latencyMs) / n
}

// ExtractStateVector extracts 6-DOF state vector for interceptor guidance
// Returns position and velocity arrays for Kalman Filter input
func (r *FCHReceiver) ExtractStateVector(brief *FireControlBrief) (position, velocity [3]float64, covariance [6][6]float64) {
	return brief.Pos, brief.Vel, brief.GetCovarianceMatrix()
}

// GetTrustWeight returns Kalman Filter trust weight based on covariance
// Lower covariance = higher trust = higher weight
func (r *FCHReceiver) GetTrustWeight(brief *FireControlBrief) float64 {
	covariance := brief.GetCovarianceMatrix()

	// Calculate trace of position covariance (diagonal elements 0,1,2)
	posVariance := covariance[0][0] + covariance[1][1] + covariance[2][2]

	// Calculate trace of velocity covariance (diagonal elements 3,4,5)
	velVariance := covariance[3][3] + covariance[4][4] + covariance[5][5]

	// Combined uncertainty
	totalUncertainty := posVariance + velVariance

	// Trust weight: inverse of uncertainty (normalized)
	// High uncertainty -> low trust weight
	// Low uncertainty -> high trust weight
	if totalUncertainty < 1e-6 {
		return 1.0 // Perfect trust
	}

	weight := 1.0 / (1.0 + totalUncertainty/1e6)
	if weight > 1.0 {
		weight = 1.0
	}
	if weight < 0.01 {
		weight = 0.01 // Minimum trust
	}

	return weight
}

// IsStale checks if the handover is too old for reliable engagement
func (r *FCHReceiver) IsStale(brief *FireControlBrief, maxAge time.Duration) bool {
	age := time.Since(brief.Timestamp)
	return age > maxAge
}

// GetReceivedHandover retrieves a stored handover by ID
func (r *FCHReceiver) GetReceivedHandover(handoverID string) (*FireControlBrief, bool) {
	brief, exists := r.receivedHandovers[handoverID]
	return brief, exists
}

// GetPendingHandovers returns all handovers awaiting engagement
func (r *FCHReceiver) GetPendingHandovers() map[string]*FireControlBrief {
	return r.statistics.PendingHandovers
}

// MarkEngaged removes handover from pending list after engagement
func (r *FCHReceiver) MarkEngaged(handoverID string) {
	delete(r.statistics.PendingHandovers, handoverID)
}

// GetStatistics returns reception statistics
func (r *FCHReceiver) GetStatistics() *FCHReceptionStats {
	return r.statistics
}

// GetSuccessRate returns reception success rate
func (r *FCHReceiver) GetSuccessRate() float64 {
	if r.statistics.TotalReceived == 0 {
		return 0.0
	}
	return float64(r.statistics.ValidReceived) / float64(r.statistics.TotalReceived)
}

// ClearOldHandovers removes handovers older than specified duration
func (r *FCHReceiver) ClearOldHandovers(maxAge time.Duration) int {
	cleared := 0
	cutoff := time.Now().Add(-maxAge)

	for id, brief := range r.receivedHandovers {
		if brief.Timestamp.Before(cutoff) {
			delete(r.receivedHandovers, id)
			delete(r.statistics.PendingHandovers, id)
			cleared++
		}
	}

	return cleared
}

// InterceptorGuidance computes guidance commands from FCH data
// Returns: time-to-intercept, required delta-v, approach angle
type InterceptorGuidance struct {
	TimeToInterceptSec float64
	RequiredDeltaVmps  float64
	ApproachAngleDeg   float64
	ImpactPoint        [3]float64
	Confidence         float64
}

// ComputeGuidance calculates interceptor guidance from FCH
// This is a simplified calculation - real implementation would use Lambert solver
func (r *FCHReceiver) ComputeGuidance(brief *FireControlBrief, interceptorPos [3]float64) *InterceptorGuidance {
	// Calculate relative position
	dx := brief.Pos[0] - interceptorPos[0]
	dy := brief.Pos[1] - interceptorPos[1]
	dz := brief.Pos[2] - interceptorPos[2]

	// Range
	range_m := dx*dx + dy*dy + dz*dz
	range_m = sqrt(range_m)

	// Simple time-to-intercept (assuming constant velocity intercept)
	// Real calculation would use orbital mechanics
	interceptSpeed := 3000.0 // m/s typical interceptor speed
	timeToIntercept := range_m / interceptSpeed

	// Required delta-v (simplified)
	// Accounts for target velocity and closing speed
	targetSpeed := sqrt(brief.Vel[0]*brief.Vel[0] + brief.Vel[1]*brief.Vel[1] + brief.Vel[2]*brief.Vel[2])
	requiredDeltaV := targetSpeed + interceptSpeed

	// Approach angle
	approachAngle := 0.0
	if range_m > 0 {
		// Dot product for angle calculation
		dot := dx*brief.Vel[0] + dy*brief.Vel[1] + dz*brief.Vel[2]
		approachAngle = dot / (range_m * targetSpeed)
		approachAngle = acos(approachAngle) * 180.0 / 3.14159
	}

	// Impact point prediction (simplified)
	impactPoint := [3]float64{
		brief.Pos[0] + brief.Vel[0]*timeToIntercept,
		brief.Pos[1] + brief.Vel[1]*timeToIntercept,
		brief.Pos[2] + brief.Vel[2]*timeToIntercept,
	}

	// Confidence based on covariance
	trust := r.GetTrustWeight(brief)

	return &InterceptorGuidance{
		TimeToInterceptSec: timeToIntercept,
		RequiredDeltaVmps:  requiredDeltaV,
		ApproachAngleDeg:   approachAngle,
		ImpactPoint:        impactPoint,
		Confidence:         trust * brief.Confidence,
	}
}

// sqrt helper function
func sqrt(x float64) float64 {
	if x < 0 {
		return 0
	}
	// Simple Newton-Raphson square root
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// acos helper function (approximation)
func acos(x float64) float64 {
	// Clamp to valid range
	if x > 1.0 {
		x = 1.0
	}
	if x < -1.0 {
		x = -1.0
	}
	// Taylor series approximation around 0
	return 1.57079632679 - x - x*x*x/6.0
}

// FCHReceptionHandler is the high-level interface for interceptor reception
type FCHReceptionHandler struct {
	receiver *FCHReceiver
	callback func(*FireControlBrief, *InterceptorGuidance)
}

// NewFCHReceptionHandler creates a new reception handler
func NewFCHReceptionHandler(callback func(*FireControlBrief, *InterceptorGuidance)) *FCHReceptionHandler {
	return &FCHReceptionHandler{
		receiver: NewFCHReceiver(),
		callback: callback,
	}
}

// ProcessIncomingPacket processes a raw FCH packet
func (h *FCHReceptionHandler) ProcessIncomingPacket(data []byte, interceptorPos [3]float64) error {
	// Receive and parse
	brief, err := h.receiver.ReceiveFireControlBrief(data)
	if err != nil {
		return err
	}

	// Compute guidance
	guidance := h.receiver.ComputeGuidance(brief, interceptorPos)

	// Execute callback
	if h.callback != nil {
		h.callback(brief, guidance)
	}

	return nil
}

// GetReceiverStats returns reception statistics
func (h *FCHReceptionHandler) GetReceiverStats() *FCHReceptionStats {
	return h.receiver.GetStatistics()
}
