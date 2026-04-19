// Package mesh provides Fire-Control Handover (FCH) for tactical data links
// Implements high-precision 6-DOF state vectors with covariance for interceptor guidance
package mesh

import (
	"fmt"
	"time"
)

// SolutionType defines the type of tactical solution
const (
	SolutionKinetic = "KINETIC" // Ballistic interceptor engagement
	SolutionDEW     = "DEW"     // Directed Energy Weapon (laser/microwave)
	SolutionJam     = "JAM"     // Electronic warfare countermeasure
	SolutionCyber   = "CYBER"   // Cyber/electronic intrusion
)

// FireControlBrief represents the high-precision data required for interception
// Designed to be under 512 bytes for single-packet LoRa/Reticulum delivery
type FireControlBrief struct {
	// Metadata
	HandoverID string    `msgpack:"id"`  // Unique handover identifier
	Timestamp  time.Time `msgpack:"ts"`  // UTC timestamp of handover
	TargetID   int       `msgpack:"tid"` // NORAD/Satellite ID

	// State Vector (6-DOF) in ECEF meters and m/s
	// High-precision position and velocity for interceptor guidance
	Pos [3]float64 `msgpack:"p"` // x, y, z position (meters)
	Vel [3]float64 `msgpack:"v"` // vx, vy, vz velocity (m/s)

	// Covariance Matrix (upper triangle of 6x6, 21 elements)
	// Provides uncertainty quantification for interceptor Kalman Filter
	// Layout: [P00, P01, P02, P03, P04, P05, P11, P12, P13, P14, P15, P22, P23, P24, P25, P33, P34, P35, P44, P45, P55]
	// Where Pij represents covariance between state elements i and j
	// State order: [x, y, z, vx, vy, vz]
	Covariance []float64 `msgpack:"cov"`

	// Tactical Metadata
	Confidence   float64 `msgpack:"conf"` // Confidence score (0-1)
	SolutionType string  `msgpack:"type"` // KINETIC, DEW, JAM, CYBER
	ThreatLevel  int     `msgpack:"thr"`  // Threat assessment level (1-5)

	// Optional Extended Data (not included in minimal packet)
	EstimatedMass    float64 `msgpack:"mass,omitempty"`   // Target mass estimate (kg)
	RCS              float64 `msgpack:"rcs,omitempty"`    // Radar cross-section (m²)
	IntentConfidence float64 `msgpack:"intent,omitempty"` // Intent classification confidence
}

// NewFireControlBrief creates a new FCH with generated ID
func NewFireControlBrief(targetID int, solutionType string) *FireControlBrief {
	return &FireControlBrief{
		HandoverID:   generateHandoverID(targetID),
		Timestamp:    time.Now().UTC(),
		TargetID:     targetID,
		SolutionType: solutionType,
		Confidence:   0.0,
		ThreatLevel:  1,
		Covariance:   make([]float64, 21), // Initialize with zeros
	}
}

// generateHandoverID creates a unique handover identifier
func generateHandoverID(targetID int) string {
	return fmt.Sprintf("FCH-%d-%d", targetID, time.Now().Unix())
}

// SetStateVector sets the 6-DOF state vector
func (f *FireControlBrief) SetStateVector(pos, vel [3]float64) {
	f.Pos = pos
	f.Vel = vel
}

// SetCovarianceFromMatrix sets covariance from full 6x6 matrix
// Only stores upper triangle (21 elements) to save space
func (f *FireControlBrief) SetCovarianceFromMatrix(cov6x6 [6][6]float64) {
	f.Covariance = FlattenCovarianceUpperTriangle(cov6x6)
}

// GetCovarianceMatrix reconstructs full 6x6 covariance from upper triangle
func (f *FireControlBrief) GetCovarianceMatrix() [6][6]float64 {
	return UnflattenCovarianceUpperTriangle(f.Covariance)
}

// FlattenCovarianceUpperTriangle converts 6x6 matrix to upper triangle (21 elements)
func FlattenCovarianceUpperTriangle(cov6x6 [6][6]float64) []float64 {
	flattened := make([]float64, 21)
	idx := 0
	for i := 0; i < 6; i++ {
		for j := i; j < 6; j++ {
			flattened[idx] = cov6x6[i][j]
			idx++
		}
	}
	return flattened
}

// UnflattenCovarianceUpperTriangle reconstructs 6x6 matrix from upper triangle
func UnflattenCovarianceUpperTriangle(flattened []float64) [6][6]float64 {
	var cov6x6 [6][6]float64
	if len(flattened) < 21 {
		return cov6x6 // Return zeros if insufficient data
	}

	idx := 0
	for i := 0; i < 6; i++ {
		for j := i; j < 6; j++ {
			cov6x6[i][j] = flattened[idx]
			cov6x6[j][i] = flattened[idx] // Symmetric matrix
			idx++
		}
	}
	return cov6x6
}

// Validate checks if the FCH has valid data
func (f *FireControlBrief) Validate() error {
	if f.HandoverID == "" {
		return fmt.Errorf("handover ID is required")
	}

	if f.TargetID <= 0 {
		return fmt.Errorf("invalid target ID: %d", f.TargetID)
	}

	if f.Timestamp.IsZero() {
		return fmt.Errorf("timestamp is required")
	}

	if len(f.Covariance) != 21 && len(f.Covariance) != 36 {
		return fmt.Errorf("covariance must have 21 (upper triangle) or 36 (full) elements, got %d", len(f.Covariance))
	}

	// Validate solution type
	validSolutions := []string{SolutionKinetic, SolutionDEW, SolutionJam, SolutionCyber}
	valid := false
	for _, sol := range validSolutions {
		if f.SolutionType == sol {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("invalid solution type: %s", f.SolutionType)
	}

	// Validate confidence range
	if f.Confidence < 0 || f.Confidence > 1 {
		return fmt.Errorf("confidence must be between 0 and 1, got %f", f.Confidence)
	}

	return nil
}

// SizeEstimate returns estimated serialized size in bytes
// Used to ensure packet fits within LoRa MTU (< 512 bytes)
func (f *FireControlBrief) SizeEstimate() int {
	// Rough estimation based on typical MessagePack encoding
	size := 0
	size += len(f.HandoverID) + 10   // String overhead
	size += 12                       // Timestamp
	size += 4                        // TargetID
	size += 6 * 8                    // Pos (3 floats) + Vel (3 floats)
	size += len(f.Covariance) * 8    // Covariance floats
	size += 8                        // Confidence
	size += len(f.SolutionType) + 10 // String overhead
	size += 4                        // ThreatLevel
	// Optional fields
	if f.EstimatedMass > 0 {
		size += 8
	}
	if f.RCS > 0 {
		size += 8
	}
	if f.IntentConfidence > 0 {
		size += 8
	}

	return size
}

// IsWithinSizeLimit checks if the FCH will fit in a single LoRa packet
func (f *FireControlBrief) IsWithinSizeLimit(maxBytes int) bool {
	return f.SizeEstimate() < maxBytes
}

// String returns human-readable summary of the FCH
func (f *FireControlBrief) String() string {
	return fmt.Sprintf(
		"FireControlBrief[ID=%s, Target=%d, Type=%s, Conf=%.2f, Pos=(%.1f, %.1f, %.1f), Vel=(%.1f, %.1f, %.1f)]",
		f.HandoverID,
		f.TargetID,
		f.SolutionType,
		f.Confidence,
		f.Pos[0], f.Pos[1], f.Pos[2],
		f.Vel[0], f.Vel[1], f.Vel[2],
	)
}

// FireControlTelemetry tracks FCH transmission statistics
type FireControlTelemetry struct {
	TotalDispatched      int64
	SuccessfulDispatches int64
	FailedDispatches     int64
	AverageLatencyMs     float64
	LastDispatchTime     time.Time
	PendingHandovers     map[string]*FireControlBrief
}

// NewFireControlTelemetry creates telemetry tracker
func NewFireControlTelemetry() *FireControlTelemetry {
	return &FireControlTelemetry{
		PendingHandovers: make(map[string]*FireControlBrief),
	}
}

// RecordDispatch records a dispatch attempt
func (t *FireControlTelemetry) RecordDispatch(success bool, latencyMs float64) {
	t.TotalDispatched++
	if success {
		t.SuccessfulDispatches++
	} else {
		t.FailedDispatches++
	}

	// Update rolling average
	n := float64(t.TotalDispatched)
	t.AverageLatencyMs = (t.AverageLatencyMs*(n-1) + latencyMs) / n
	t.LastDispatchTime = time.Now()
}

// SuccessRate returns the dispatch success rate
func (t *FireControlTelemetry) SuccessRate() float64 {
	if t.TotalDispatched == 0 {
		return 0.0
	}
	return float64(t.SuccessfulDispatches) / float64(t.TotalDispatched)
}

// FireControlPriorityInt defines integer priority levels for handover processing
const (
	FireControlPriorityCritical = 1 // Kinetic threats, immediate engagement required
	FireControlPriorityHigh     = 2 // Active EW or inspection
	FireControlPriorityMedium   = 3 // Suspicious activity
	FireControlPriorityLow      = 4 // Routine tracking
)

// GetPriorityInt returns processing priority as integer based on threat level and solution type
func (f *FireControlBrief) GetPriorityInt() int {
	if f.ThreatLevel >= 4 || f.SolutionType == SolutionKinetic {
		return FireControlPriorityCritical
	}
	if f.ThreatLevel >= 3 || f.SolutionType == SolutionDEW {
		return FireControlPriorityHigh
	}
	if f.ThreatLevel >= 2 || f.SolutionType == SolutionJam {
		return FireControlPriorityMedium
	}
	return FireControlPriorityLow
}

// GetPriority returns processing priority as Priority type based on threat level and solution type
func (f *FireControlBrief) GetPriority() Priority {
	if f.ThreatLevel >= 4 || f.SolutionType == SolutionKinetic {
		return PriorityCritical
	}
	if f.ThreatLevel >= 3 || f.SolutionType == SolutionDEW {
		return PriorityHigh
	}
	return PriorityRoutine
}
