# ONNX Runtime Go Integration Layer

Unified inference wrapper, batching, model versioning, and latency instrumentation for all SSA ML features.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SSA ML Features                           │
│  (Maneuver, Collision, Rapid-Threat, Blindspot, etc.)       │
└────────────────────┬────────────────────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────────────────────┐
│              ONNX Integration Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Session (runtime.go)                                 │   │
│  │  - Unified inference wrapper                         │   │
│  │  - Batch processing (batching)                       │   │
│  │  - Request distribution                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Cache (cache.go)                                     │   │
│  │  - LRU inference result caching                      │   │
│  │  - Hit/miss tracking                                 │   │
│  │  - Cache statistics                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Version Registry (cache.go)                          │   │
│  │  - Model version tracking                            │   │
│  │  - Checksum verification                             │   │
│  │  - Rollback support                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Metrics (metrics.go)                                 │   │
│  │  - Latency distribution (min/max/p95/p99)           │   │
│  │  - Error tracking                                    │   │
│  │  - Cache hit rate                                    │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │ implements
┌────────────────────▼────────────────────────────────────────┐
│ Inferrer Interface                                            │
│  - InferBatch([]InferenceRequest) []InferenceResult         │
│  - Close()                                                    │
│  - ModelInfo()                                                │
└────────────────────┬────────────────────────────────────────┘
                     │ backend implementations:
        ┌────────────┼────────────┐
        │            │            │
    ┌───▼───┐  ┌────▼────┐  ┌───▼────┐
    │ ONNX  │  │  Mock   │  │ Stub   │
    │Runtime│  │Inferrer │  │(failsafe)
    └───────┘  └─────────┘  └────────┘
```

## Usage

### Basic Session Setup

```go
package main

import (
    "fmt"
    "ssis/engine/pkg/ml/onnx"
)

func main() {
    // Create a mock inferrer (or real ONNX runtime in production)
    inferrer := onnx.NewMockInferrer("maneuver_detector", "1.0")

    // Configure session
    config := onnx.SessionConfig{
        ModelID:        "maneuver_detector",
        ModelPath:      "/path/to/model.onnx",
        ModelVersion:   "1.0",
        MaxBatchSize:   32,
        CacheSize:      512,
        EnableLogging:  true,
    }

    // Create session
    session := onnx.NewSession(config, inferrer)
    defer session.Close()

    // Single inference
    req := onnx.InferenceRequest{
        ID:     "req-001",
        ModelID: "maneuver_detector",
        Inputs: map[string]interface{}{
            "timeseries_residuals": []float32{0.1, 0.2, 0.3},
        },
    }
    result := session.Infer(req)
    fmt.Printf("Result: %+v\n", result)

    // Batch inference
    requests := []onnx.InferenceRequest{...}
    results := session.InferBatch(requests)
    for _, r := range results {
        fmt.Printf("Result %s: confidence=%f, latency=%.2fms\n",
            r.RequestID, r.Confidence, r.LatencyMs)
    }

    // Get metrics
    metrics := session.Metrics()
    fmt.Printf("Metrics: %+v\n", metrics.GetMetrics())
}
```

### Model Versioning

```go
// Create registry
registry := onnx.NewVersionRegistry()

// Register versions
v1 := onnx.ModelVersion{
    ID:        "1.0.0",
    ModelID:   "collision_detector",
    ModelPath: "/data/collision_v1.onnx",
}
checksum, _ := onnx.ComputeChecksum(v1.ModelPath)
v1.Checksum = checksum
registry.Register(v1)

// Get latest version
latest, _ := registry.GetLatestVersion("collision_detector")
fmt.Printf("Latest: %s (deployed at %v)\n", latest.ID, latest.DeployedAt)

// Verify checksum before loading
match, _ := latest.VerifyChecksum("/data/collision_v1.onnx")
if !match {
    panic("model file has changed since registration")
}
```

### Caching and Metrics

```go
// Cache statistics
cacheStats := session.cache.Stats()
fmt.Printf("Cache hit rate: %.2f%%\n", cacheStats["hit_rate"].(float64)*100)

// Latency metrics
metrics := session.Metrics()
fmt.Printf("P95 latency: %.2fms, P99 latency: %.2fms\n",
    metrics.P95LatencyMs, metrics.P99LatencyMs)
```

## Features

### 1. Unified Inference Wrapper
- Single API for all model types: **`Infer()`** and **`InferBatch()`**
- Automatic request queuing and dispatch
- Model version tracking per request

### 2. Batch Processing
- Configurable max batch size
- Large batches are automatically split and processed
- Per-request latency calculation even within batches

### 3. Model Versioning
- Track deployed versions with MD5 checksums
- Get latest version or specific versions
- Verify model files haven't drifted

### 4. Inference Result Caching
- LRU cache for recent results
- Configurable cache size
- Hit/miss rate tracking

### 5. Latency Instrumentation
- Per-request and per-batch latencies
- Circular buffer of recent samples
- Min, max, average, P95, P99 percentiles
- Error rate and cache hit rate tracking

### 6. Mock/Stub Inferrer
- **MockInferrer** for testing and development
- Configurable latency and error simulation
- **Stub implementations** for feature toggling

## Integration with ML Features

### Maneuver Detection

```go
// In pkg/orbital/maneuver.go
type Detector struct {
    session *onnx.Session
}

func (d *Detector) InferManeuver(frames []ManeuverFrame) (result, error) {
    req := onnx.InferenceRequest{
        ID: generateRequestID(),
        Inputs: map[string]interface{}{
            "window": framesToTensor(frames),
        },
    }
    onnxResult := d.session.Infer(req)
    if onnxResult.Error != nil {
        // fallback to CUSUM
    }
    // Parse onnxResult.Outputs...
}
```

### Collision Risk

```go
// In pkg/orbital/collision.go
type GraphRiskScorer struct {
    session *onnx.Session
}

func (s *GraphRiskScorer) ScoreRisk(graph CollisionNeighborhoodGraph) float64 {
    req := onnx.InferenceRequest{
        ID: generateRequestID(),
        Inputs: map[string]interface{}{
            "node_features": graph.NodeFeatures(),
            "adjacency":     graph.AdjacencyMatrix(),
        },
    }
    result := s.session.Infer(req)
    conf, _ := onnx.ExtractConfidence(result.Outputs, "risk_score")
    return conf
}
```

## Build Tags

### Default (no ONNX runtime)
```bash
go build -o ssis ./engine/cmd/main.go
```
Uses stub implementations; models will error gracefully.

### Production (with ONNX runtime)
```bash
go build -tags onnx -o ssis ./engine/cmd/main.go
```
Compiles native ONNX runtime bindings. Requires `onnxruntime-go` and system ONNX Runtime library.

## Performance Characteristics

| Operation | Latency |
|-----------|---------|
| Inference (CPU) | 5–50ms (model dependent) |
| Batch inference (32 items) | ~10ms per item |
| Cache lookup (hit) | <0.1ms |
| Metrics snapshot | <1ms |

## Testing

```bash
# Run all integration tests
go test ./pkg/ml/onnx -v

# Run benchmarks
go test ./pkg/ml/onnx -bench=. -benchmem

# Test specific feature
go test -run TestSessionBatching ./pkg/ml/onnx -v
```

## Files

- **runtime.go**: Core `Session` and `Inferrer` interface
- **metrics.go**: `LatencyMetrics` for telemetry
- **cache.go**: `InferenceCache`, `VersionRegistry`, checksums
- **common.go**: `MockInferrer`, tensor helpers, batch utilities
- **integration_test.go**: Comprehensive test suite
