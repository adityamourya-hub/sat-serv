package onnx

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// LatencyMetrics tracks inference performance and telemetry.
type LatencyMetrics struct {
	mu                sync.RWMutex
	ModelID           string
	StartTime         time.Time
	TotalInferences   int64
	TotalErrors       int64
	CacheHits         int64
	CacheMisses       int64
	LatencySamples    []float64 // circular buffer of recent millisecond latencies
	LatencyIndex      int
	LatencyCount      int
	LatencyBufferSize int
	latencySumMs      float64
	MinLatencyMs      float64
	MaxLatencyMs      float64
	AvgLatencyMs      float64
	P95LatencyMs      float64
	P99LatencyMs      float64
}

// NewLatencyMetrics creates a new metrics recorder.
func NewLatencyMetrics(modelID string) *LatencyMetrics {
	bufferSize := 1000
	return &LatencyMetrics{
		ModelID:           modelID,
		StartTime:         time.Now(),
		LatencySamples:    make([]float64, bufferSize),
		LatencyBufferSize: bufferSize,
		MinLatencyMs:      1e9,
		MaxLatencyMs:      0.0,
		AvgLatencyMs:      0.0,
	}
}

// RecordLatency records the latency of a single inference.
func (m *LatencyMetrics) RecordLatency(latencyMs float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	atomic.AddInt64(&m.TotalInferences, 1)

	// Update running min/max
	if latencyMs < m.MinLatencyMs {
		m.MinLatencyMs = latencyMs
	}
	if latencyMs > m.MaxLatencyMs {
		m.MaxLatencyMs = latencyMs
	}

	// Circular buffer for percentile calculation with running sum.
	old := m.LatencySamples[m.LatencyIndex]
	if m.LatencyCount < m.LatencyBufferSize {
		m.LatencyCount++
		old = 0
	}
	m.LatencySamples[m.LatencyIndex] = latencyMs
	m.LatencyIndex = (m.LatencyIndex + 1) % m.LatencyBufferSize
	m.latencySumMs += latencyMs - old

	if m.LatencyCount > 0 {
		m.AvgLatencyMs = m.latencySumMs / float64(m.LatencyCount)
	}

	// Approximate percentiles over current buffer window.
	m.updatePercentiles()
}

// RecordError records an inference error.
func (m *LatencyMetrics) RecordError() {
	atomic.AddInt64(&m.TotalErrors, 1)
}

// RecordCacheHit records a cache hit.
func (m *LatencyMetrics) RecordCacheHit() {
	atomic.AddInt64(&m.CacheHits, 1)
}

// RecordCacheMiss records a cache miss.
func (m *LatencyMetrics) RecordCacheMiss() {
	atomic.AddInt64(&m.CacheMisses, 1)
}

// updatePercentiles updates P95 and P99 latency estimates.
func (m *LatencyMetrics) updatePercentiles() {
	if m.LatencyCount == 0 {
		m.P95LatencyMs = 0
		m.P99LatencyMs = 0
		return
	}

	sorted := make([]float64, m.LatencyCount)
	copy(sorted, m.LatencySamples[:m.LatencyCount])
	sort.Float64s(sorted)

	idx95 := percentileIndex(m.LatencyCount, 0.95)
	idx99 := percentileIndex(m.LatencyCount, 0.99)
	m.P95LatencyMs = sorted[idx95]
	m.P99LatencyMs = sorted[idx99]
}

func percentileIndex(n int, p float64) int {
	if n <= 1 {
		return 0
	}
	idx := int(math.Ceil(p*float64(n))) - 1
	if idx < 0 {
		return 0
	}
	if idx >= n {
		return n - 1
	}
	return idx
}

// GetMetrics returns a read-only snapshot of current metrics.
func (m *LatencyMetrics) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	totalInferences := atomic.LoadInt64(&m.TotalInferences)
	totalErrors := atomic.LoadInt64(&m.TotalErrors)
	cacheHits := atomic.LoadInt64(&m.CacheHits)
	cacheMisses := atomic.LoadInt64(&m.CacheMisses)
	totalAccess := cacheHits + cacheMisses
	hitRate := 0.0
	if totalAccess > 0 {
		hitRate = float64(cacheHits) / float64(totalAccess)
	}
	errorRate := 0.0
	if totalInferences > 0 {
		errorRate = float64(totalErrors) / float64(totalInferences)
	}

	return map[string]interface{}{
		"model_id":         m.ModelID,
		"total_inferences": totalInferences,
		"total_errors":     totalErrors,
		"cache_hits":       cacheHits,
		"cache_misses":     cacheMisses,
		"min_latency_ms":   m.MinLatencyMs,
		"max_latency_ms":   m.MaxLatencyMs,
		"avg_latency_ms":   m.AvgLatencyMs,
		"p95_latency_ms":   m.P95LatencyMs,
		"p99_latency_ms":   m.P99LatencyMs,
		"uptime_seconds":   time.Since(m.StartTime).Seconds(),
		"cache_hit_rate":   hitRate,
		"error_rate":       errorRate,
	}
}

// Summary returns a human-readable metrics summary.
func (m *LatencyMetrics) Summary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return fmt.Sprintf(
		"Model: %s | Inferences: %d | Avg: %s | P95: %s | P99: %s | Errors: %d",
		m.ModelID,
		atomic.LoadInt64(&m.TotalInferences),
		formatMS(m.AvgLatencyMs),
		formatMS(m.P95LatencyMs),
		formatMS(m.P99LatencyMs),
		atomic.LoadInt64(&m.TotalErrors),
	)
}

func formatMS(v float64) string {
	if v < 1.0 {
		return "<1ms"
	}
	if v < 100.0 {
		return formatFloat(v, 1) + "ms"
	}
	return formatFloat(v, 0) + "ms"
}

func formatFloat(v float64, decimals int) string {
	if decimals <= 0 {
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
	return strconv.FormatFloat(v, 'f', decimals, 64)
}
