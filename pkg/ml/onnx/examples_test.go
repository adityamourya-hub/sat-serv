package onnx

import (
	"fmt"
	"testing"
)

func DemoSessionUsage() {
	// Create mock inferrer (swap with real ONNX runtime in production)
	// mockInferrer should implement the Inferrer interface
	var inferrer Inferrer // Mock initialization omitted for brevity in demo

	// Configure session with batching
	config := SessionConfig{
		ModelID:           "maneuver_detector",
		ModelPath:         "/data/models/maneuver_lstm_v1.onnx",
		Version:           "1.0.0",
		MaxBatchSize:      64,
		EnableLogging:     true,
		IntraOpNumThreads: 4,
	}

	// Create managed session
	session := NewSession(config, inferrer)
	defer session.Close()

	// Example 1: Single inference (with automatic batching)
	inputs := map[string]interface{}{
		"timeseries_residuals": []float32{0.01, 0.02, 0.015, 0.03, 0.025},
	}

	outputs, err := session.Infer(inputs)
	if err != nil {
		fmt.Printf("inference error: %v", err)
		return
	}

	fmt.Printf("Maneuver outputs: %v\n", outputs)

	// Example 2: Monitoring performance
	metrics := session.Metrics()
	fmt.Printf("Performance snapshot: %v\n", metrics)
}

// DemoVersioningUsage demonstrates model versioning and rollback.
func DemoVersioningUsage() {
	registry := NewVersionRegistry()

	// Register initial production model
	v1 := ModelVersion{
		ID:          "1.0.0",
		ModelID:     "collision_detector",
		ModelPath:   "/data/models/collision_gat_1.0.onnx",
		Checksum:    "abc123def456",
		Description: "Initial GAT-based collision risk model",
	}
	registry.Register(v1)

	// Later, register improved version
	v2 := ModelVersion{
		ID:          "1.1.0",
		ModelID:     "collision_detector",
		ModelPath:   "/data/models/collision_gat_1.1.onnx",
		Checksum:    "xyz789uvw012",
		Description: "Improved attention weights and calibration",
	}
	registry.Register(v2)

	// Deploy latest version
	latestVersion, _ := registry.GetLatestVersion("collision_detector")
	fmt.Printf("Deploying collision model v%s (%s)\n",
		latestVersion.ID, latestVersion.ModelPath)

	// Fallback to previous version if needed
	allVersions := registry.ListVersions("collision_detector")
	fmt.Printf("Available versions: ")
	for _, v := range allVersions {
		fmt.Printf("%s ", v.ID)
	}
	fmt.Printf("\n")
}

// DemoCachingUsage demonstrates result caching for repeated requests.
func DemoCachingUsage() {
	cache := NewInferenceCache(100)

	// First call: compute and cache
	result1 := InferenceResult{
		RequestID:  "req-repeat-001",
		Confidence: 0.95,
	}
	cache.Put(result1.RequestID, result1)

	// Second call: hit cache (zeroed latency)
	if cached, found := cache.Get(result1.RequestID); found {
		fmt.Printf("Cache hit: req=%s confidence=%.2f\n",
			cached.RequestID, cached.Confidence)
	}

	// Stats
	stats := cache.Stats()
	fmt.Printf("Cache utilization: %.1f%% (%d/%d entries)\n",
		stats["utilization"].(float64)*100,
		stats["current_size"], stats["max_size"])
}

// TestExampleIntegration_Maneuver demonstrates full pipeline for maneuver detection.
func TestExampleIntegration_Maneuver(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: "InferBatch deprecated and redundant in new Session-pool model")
}

// TestBatchPerformance_Scalability demonstrates linear scaling with batch size.
func TestBatchPerformance_Scalability(t *testing.T) {
	t.Fatal("NOT IMPLEMENTED: "InferBatch deprecated and redundant in new Session-pool model")
}
