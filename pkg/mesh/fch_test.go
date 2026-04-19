// Package mesh provides tests for Fire-Control Handover (FCH)
package mesh

import (
	"testing"
)

// TestFireControlBriefSerialization tests FCH serialization round-trip
func TestFireControlBriefSerialization(t *testing.T) {
	// Create a test brief
	brief := NewFireControlBrief(99999, SolutionKinetic)
	brief.SetStateVector(
		[3]float64{6800000.0, 50000.0, 50000.0},
		[3]float64{0.0, 7500.0, 0.0},
	)

	// Create test covariance matrix
	var cov6x6 [6][6]float64
	cov6x6[0][0] = 100.0 // Position variance
	cov6x6[1][1] = 100.0
	cov6x6[2][2] = 100.0
	cov6x6[3][3] = 10.0 // Velocity variance
	cov6x6[4][4] = 10.0
	cov6x6[5][5] = 10.0
	brief.SetCovarianceFromMatrix(cov6x6)

	brief.Confidence = 0.95
	brief.ThreatLevel = 5

	// Validate
	if err := brief.Validate(); err != nil {
		t.Errorf("Brief validation failed: %v", err)
	}

	// Serialize
	serializer := NewFCHSerializer()
	data, err := serializer.Serialize(brief)
	if err != nil {
		t.Errorf("Serialization failed: %v", err)
	}

	// Check size limit
	if len(data) > 512 {
		t.Errorf("Serialized size %d exceeds 512 byte limit", len(data))
	}

	t.Logf("Serialized FCH size: %d bytes", len(data))

	// Deserialize
	received, err := serializer.Deserialize(data)
	if err != nil {
		t.Errorf("Deserialization failed: %v", err)
	}

	// Verify fields
	if received.TargetID != brief.TargetID {
		t.Errorf("TargetID mismatch: %d != %d", received.TargetID, brief.TargetID)
	}

	if received.SolutionType != brief.SolutionType {
		t.Errorf("SolutionType mismatch: %s != %s", received.SolutionType, brief.SolutionType)
	}

	// Verify state vector
	for i := 0; i < 3; i++ {
		if received.Pos[i] != brief.Pos[i] {
			t.Errorf("Position[%d] mismatch: %f != %f", i, received.Pos[i], brief.Pos[i])
		}
		if received.Vel[i] != brief.Vel[i] {
			t.Errorf("Velocity[%d] mismatch: %f != %f", i, received.Vel[i], brief.Vel[i])
		}
	}
}

// TestCovarianceFlattening tests covariance matrix flattening
func TestCovarianceFlattening(t *testing.T) {
	// Create test covariance matrix
	var cov6x6 [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			cov6x6[i][j] = float64(i*10 + j)
		}
	}

	// Flatten
	flattened := FlattenCovarianceUpperTriangle(cov6x6)

	// Should have 21 elements (upper triangle of 6x6)
	if len(flattened) != 21 {
		t.Errorf("Flattened length %d != 21", len(flattened))
	}

	// Unflatten
	restored := UnflattenCovarianceUpperTriangle(flattened)

	// Verify symmetry and values
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			expected := float64(i*10 + j)
			if restored[i][j] != expected {
				t.Errorf("Restored[%d][%d] = %f, expected %f", i, j, restored[i][j], expected)
			}
			// Check symmetry
			if restored[i][j] != restored[j][i] {
				t.Errorf("Matrix not symmetric at [%d][%d]", i, j)
			}
		}
	}
}

// TestFireControlBriefValidation tests validation logic
func TestFireControlBriefValidation(t *testing.T) {
	// Test invalid brief (missing fields)
	invalidBrief := &FireControlBrief{
		TargetID:     -1,
		SolutionType: "INVALID",
	}

	if err := invalidBrief.Validate(); err == nil {
		t.Error("Expected validation error for invalid brief")
	}

	// Test valid brief
	validBrief := NewFireControlBrief(12345, SolutionKinetic)
	validBrief.SetStateVector(
		[3]float64{7000000.0, 0.0, 0.0},
		[3]float64{0.0, 7500.0, 0.0},
	)
	var cov [6][6]float64
	validBrief.SetCovarianceFromMatrix(cov)
	validBrief.Confidence = 0.8

	if err := validBrief.Validate(); err != nil {
		t.Errorf("Valid brief failed validation: %v", err)
	}
}

// TestFCHBuilder tests the builder pattern
func TestFCHBuilder(t *testing.T) {
	var cov [6][6]float64
	cov[0][0] = 100.0
	cov[1][1] = 100.0
	cov[2][2] = 100.0

	brief, err := NewFCHBuilder(12345, SolutionDEW).
		WithStateVector([3]float64{6800000, 10000, 10000}, [3]float64{100, 7500, 100}).
		WithCovariance(cov).
		WithConfidence(0.9).
		WithThreatLevel(4).
		WithExtendedData(500.0, 10.0, 0.85).
		BuildAndValidate()

	if err != nil {
		t.Errorf("Builder failed: %v", err)
	}

	if brief.TargetID != 12345 {
		t.Errorf("TargetID mismatch: %d", brief.TargetID)
	}

	if brief.SolutionType != SolutionDEW {
		t.Errorf("SolutionType mismatch: %s", brief.SolutionType)
	}

	if brief.EstimatedMass != 500.0 {
		t.Errorf("EstimatedMass mismatch: %f", brief.EstimatedMass)
	}
}

// TestFCHReceiver tests the FCH receiver
func TestFCHReceiver(t *testing.T) {
	receiver := NewFCHReceiver()

	// Create test brief
	brief := NewFireControlBrief(12345, SolutionKinetic)
	brief.SetStateVector(
		[3]float64{6800000.0, 0.0, 0.0},
		[3]float64{0.0, 7500.0, 0.0},
	)

	// Serialize
	serializer := NewFCHSerializer()
	data, _ := serializer.Serialize(brief)

	// Receive
	received, err := receiver.ReceiveFireControlBrief(data)
	if err != nil {
		t.Errorf("Reception failed: %v", err)
	}

	if received.TargetID != brief.TargetID {
		t.Errorf("Received TargetID mismatch: %d != %d", received.TargetID, brief.TargetID)
	}

	// Extract state vector
	pos, vel, cov := receiver.ExtractStateVector(received)
	if pos[0] != brief.Pos[0] {
		t.Errorf("Extracted position mismatch")
	}
	if vel[1] != brief.Vel[1] {
		t.Errorf("Extracted velocity mismatch")
	}
	if cov[0][0] != brief.Covariance[0] {
		t.Errorf("Extracted covariance mismatch")
	}

	// Check statistics
	stats := receiver.GetStatistics()
	if stats.TotalReceived != 1 {
		t.Errorf("Expected 1 received, got %d", stats.TotalReceived)
	}
	if stats.ValidReceived != 1 {
		t.Errorf("Expected 1 valid, got %d", stats.ValidReceived)
	}
}

// TestTrustWeight tests the trust weight calculation
func TestTrustWeight(t *testing.T) {
	receiver := NewFCHReceiver()

	// Create brief with low covariance (high trust)
	lowCovBrief := NewFireControlBrief(1, SolutionKinetic)
	var lowCov [6][6]float64
	lowCov[0][0] = 10.0
	lowCov[1][1] = 10.0
	lowCov[2][2] = 10.0
	lowCovBrief.SetCovarianceFromMatrix(lowCov)

	lowTrust := receiver.GetTrustWeight(lowCovBrief)
	t.Logf("Low covariance trust weight: %f", lowTrust)

	// Create brief with high covariance (low trust)
	highCovBrief := NewFireControlBrief(2, SolutionKinetic)
	var highCov [6][6]float64
	highCov[0][0] = 10000.0
	highCov[1][1] = 10000.0
	highCov[2][2] = 10000.0
	highCovBrief.SetCovarianceFromMatrix(highCov)

	highTrust := receiver.GetTrustWeight(highCovBrief)
	t.Logf("High covariance trust weight: %f", highTrust)

	// Low covariance should have higher trust
	if lowTrust <= highTrust {
		t.Error("Low covariance should have higher trust weight")
	}
}

// TestFCHPriority tests priority levels
func TestFCHPriority(t *testing.T) {
	// Critical priority test
	criticalBrief := NewFireControlBrief(1, SolutionKinetic)
	criticalBrief.ThreatLevel = 5

	if criticalBrief.GetPriorityInt() != FireControlPriorityCritical {
		t.Errorf("Expected critical priority, got %d", criticalBrief.GetPriorityInt())
	}

	// High priority test
	highBrief := NewFireControlBrief(2, SolutionDEW)
	highBrief.ThreatLevel = 3

	if highBrief.GetPriorityInt() != FireControlPriorityHigh {
		t.Errorf("Expected high priority, got %d", highBrief.GetPriorityInt())
	}

	// Low priority test
	lowBrief := NewFireControlBrief(3, SolutionJam)
	lowBrief.ThreatLevel = 1

	if lowBrief.GetPriorityInt() != FireControlPriorityMedium {
		t.Errorf("Expected medium priority, got %d", lowBrief.GetPriorityInt())
	}
}

// TestPrepareFireControlHandover tests the main preparation function
func TestPrepareFireControlHandover(t *testing.T) {
	var cov [6][6]float64
	cov[0][0] = 100.0
	cov[1][1] = 100.0
	cov[2][2] = 100.0
	cov[3][3] = 10.0
	cov[4][4] = 10.0
	cov[5][5] = 10.0

	brief, err := PrepareFireControlHandover(
		99999,
		[3]float64{6800000.0, 0.0, 0.0},
		[3]float64{0.0, 7500.0, 0.0},
		cov,
		0.95,
		SolutionKinetic,
		5,
	)

	if err != nil {
		t.Errorf("PrepareFireControlHandover failed: %v", err)
	}

	if brief.TargetID != 99999 {
		t.Errorf("TargetID mismatch")
	}

	if brief.SolutionType != SolutionKinetic {
		t.Errorf("SolutionType mismatch")
	}

	if brief.ThreatLevel != 5 {
		t.Errorf("ThreatLevel mismatch")
	}

	// Check size estimate
	size := brief.SizeEstimate()
	t.Logf("Estimated FCH size: %d bytes", size)

	if size > 512 {
		t.Errorf("Size estimate %d exceeds 512 byte limit", size)
	}
}

// TestFCHTelemetry tests telemetry tracking
func TestFCHTelemetry(t *testing.T) {
	telemetry := NewFireControlTelemetry()

	// Record some dispatches
	telemetry.RecordDispatch(true, 50.0)
	telemetry.RecordDispatch(true, 60.0)
	telemetry.RecordDispatch(false, 100.0)

	if telemetry.TotalDispatched != 3 {
		t.Errorf("Expected 3 dispatches, got %d", telemetry.TotalDispatched)
	}

	if telemetry.SuccessfulDispatches != 2 {
		t.Errorf("Expected 2 successful, got %d", telemetry.SuccessfulDispatches)
	}

	if telemetry.FailedDispatches != 1 {
		t.Errorf("Expected 1 failed, got %d", telemetry.FailedDispatches)
	}

	// Check success rate
	successRate := telemetry.SuccessRate()
	expectedRate := 2.0 / 3.0
	if successRate != expectedRate {
		t.Errorf("Success rate %f != expected %f", successRate, expectedRate)
	}
}

// BenchmarkFCHSerialization benchmarks serialization performance
func BenchmarkFCHSerialization(b *testing.B) {
	brief := NewFireControlBrief(99999, SolutionKinetic)
	brief.SetStateVector(
		[3]float64{6800000.0, 50000.0, 50000.0},
		[3]float64{0.0, 7500.0, 0.0},
	)
	var cov [6][6]float64
	cov[0][0] = 100.0
	cov[1][1] = 100.0
	cov[2][2] = 100.0
	brief.SetCovarianceFromMatrix(cov)
	brief.Confidence = 0.95

	serializer := NewFCHSerializer()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := serializer.Serialize(brief)
		_, _ = serializer.Deserialize(data)
	}
}
