package onnx

import (
	"sync"
	"testing"
)

func TestSessionBasic(t *testing.T) {
	inferrer := NewMockInferrer("test_model", "1.0")
	config := SessionConfig{
		ModelID:       "test_model",
		ModelPath:     "/tmp/model.onnx",
		Version:       "1.0",
		MaxBatchSize:  32,
		EnableLogging: true,
	}

	session := NewSession(config, inferrer)
	defer session.Close()

	inputs := map[string]interface{}{"input_0": []float32{1, 2, 3}}

	outputs, err := session.Infer(inputs)
	if err != nil {
		t.Fatalf("inference failed: %v", err)
	}
	
	if outputs == nil {
		t.Fatal("expected non-nil outputs")
	}
}

func TestSessionBatching(t *testing.T) {
	inferrer := NewMockInferrer("batch_test", "1.0")
	inferrer.SetLatency(10.0)

	config := SessionConfig{
		ModelID:       "batch_test",
		ModelPath:     "/tmp/model.onnx",
		Version:       "1.0",
		MaxBatchSize:  64,
		EnableLogging: true,
	}

	session := NewSession(config, inferrer)
	defer session.Close()

	var wg sync.WaitGroup
	numRequests := 32
	results := make([]map[string]interface{}, numRequests)
	errors := make([]error, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			inputs := map[string]interface{}{"input_0": []float32{1, 2, 3}}
			results[idx], errors[idx] = session.Infer(inputs)
		}(i)
	}

	wg.Wait()

	for i := 0; i < numRequests; i++ {
		if errors[i] != nil {
			t.Fatalf("request %d failed: %v", i, errors[i])
		}
		if results[i] == nil {
			t.Fatalf("request %d returned nil results", i)
		}
	}
}

func TestSessionBatchOversized(t *testing.T) {
	inferrer := NewMockInferrer("oversized_test", "1.0")
	config := SessionConfig{
		ModelID:       "oversized_test",
		ModelPath:     "/tmp/model.onnx",
		Version:       "1.0",
		MaxBatchSize:  16,
		EnableLogging: true,
	}

	session := NewSession(config, inferrer)
	defer session.Close()

	// Submit 50 requests when max batch size is 16
	numRequests := 50
	var wg sync.WaitGroup
	errors := make([]error, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			inputs := map[string]interface{}{"input_0": []float32{1, 2, 3}}
			_, errors[idx] = session.Infer(inputs)
		}(i)
	}

	wg.Wait()

	for i, err := range errors {
		if err != nil {
			t.Fatalf("request %d failed: %v", i, err)
		}
	}
}

func TestLatencyMetrics(t *testing.T) {
	metrics := NewLatencyMetrics("perf_test")

	// Record some latencies
	for i := 0; i < 100; i++ {
		latency := 5.0 + float64(i%10) - 5.0
		metrics.RecordLatency(latency)
	}

	if metrics.TotalInferences == 0 {
		t.Fatalf("total inferences should be > 0")
	}
	if metrics.AvgLatencyMs < 0 {
		t.Fatalf("average latency is negative")
	}

	mmap := metrics.GetMetrics()
	if mmap["model_id"] != "perf_test" {
		t.Fatalf("model ID mismatch in metrics")
	}
}

func TestInferenceCache(t *testing.T) {
	cache := NewInferenceCache(10)

	result := InferenceResult{
		RequestID:  "test-req",
		Confidence: 0.95,
	}

	cache.Put("test-req", result)

	cached, found := cache.Get("test-req")
	if !found {
		t.Fatalf("cache miss on freshly cached item")
	}

	if cached.RequestID != result.RequestID {
		t.Fatalf("cached result ID mismatch")
	}
	if cached.Confidence != result.Confidence {
		t.Fatalf("cached result confidence mismatch")
	}

	stats := cache.Stats()
	if stats["current_size"] != 1 {
		t.Fatalf("expected cache size 1, got %v", stats["current_size"])
	}
}

func TestTensorHelpers(t *testing.T) {
	// Test float32 conversion
	f32Data := []float32{1.0, 2.5, 3.7}
	result, err := TensorToFloat32(f32Data)
	if err != nil {
		t.Fatalf("TensorToFloat32 failed: %v", err)
	}
	if len(result) != 3 || result[0] != 1.0 {
		t.Fatalf("TensorToFloat32 conversion error")
	}

	// Test float64 -> float32
	f64Data := []float64{1.0, 2.5, 3.7}
	result, err = TensorToFloat32(f64Data)
	if err != nil {
		t.Fatalf("float64->float32 conversion failed: %v", err)
	}
	if len(result) != 3 || result[0] != 1.0 {
		t.Fatalf("float64->float32 conversion error")
	}

	// Test confidence extraction
	outputs := map[string]interface{}{
		"score": float32(0.87),
	}
	conf, err := ExtractConfidence(outputs, "score")
	if err != nil {
		t.Fatalf("ExtractConfidence failed: %v", err)
	}
	if conf < 0.86 || conf > 0.88 {
		t.Fatalf("confidence extraction error: got %f", conf)
	}
}

func BenchmarkSessionInference(b *testing.B) {
	inferrer := NewMockInferrer("bench", "1.0")
	inferrer.SetLatency(0.1) // minimal latency
	config := SessionConfig{
		ModelID:       "bench",
		ModelPath:     "/tmp/model.onnx",
		Version:       "1.0",
		MaxBatchSize:  128,
		EnableLogging: false,
	}

	session := NewSession(config, inferrer)
	defer session.Close()

	inputs := map[string]interface{}{"input_0": []float32{1, 2, 3}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		session.Infer(inputs)
	}
}

