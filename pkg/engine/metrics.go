package engine

import (
	"sync"
	"sync/atomic"
	"time"
)

/**
 * EngineMetrics provides high-resolution stochastic telemetry for the SSIS substrate.
 * It tracks arrival rate (lambda), service rate (mu), and utilization (rho)
 * to ensure M/D/c queue stability as mandated by Gap 6 requirements.
 *
 * <plan_complexity>Medium</plan_complexity><total_lines_estimate>120</total_lines_estimate>
 */
type EngineMetrics struct {
	// Atomic counters for zero-allocation hotspots
	Arrivals      uint64
	Completions   uint64
	ShedEvents    uint64
	QueueLength   int64
	QueueDepth    int64
	ActiveWorkers int64

	// Latency tracking for p50/p99 calculations
	latencyMu     sync.RWMutex
	latencies     []float64 // circular buffer of recent latencies
	latencyIndex  int
	latencyCount  int
	maxLatencyBuf int

	// Alert counters by priority
	criticalAlerts uint64
	highAlerts     uint64
	mediumAlerts   uint64
	lowAlerts      uint64

	// Temporal snapshots for rate calculation
	startTime time.Time
}

/**
 * NewEngineMetrics initializes the stochastic monitoring substrate.
 */
func NewEngineMetrics() *EngineMetrics {
	return &EngineMetrics{
		startTime:     time.Now(),
		maxLatencyBuf: 10000, // Keep last 10k measurements
		latencies:     make([]float64, 10000),
	}
}

/**
 * MarkArrival records a new observation entering the ingestion queue.
 */
func (m *EngineMetrics) MarkArrival() {
	// Entry Point: Ingestion Sync
	atomic.AddUint64(&m.Arrivals, 1)
	atomic.AddInt64(&m.QueueLength, 1)
}

/**
 * MarkCompletion records a successfully processed observation (UKF + ODE).
 */
func (m *EngineMetrics) MarkCompletion() {
	// Entry Point: Worker Sync
	atomic.AddUint64(&m.Completions, 1)
	atomic.AddInt64(&m.QueueLength, -1)
}

/**
 * RecordLatency stores a processing latency sample for percentile calculations.
 */
func (m *EngineMetrics) RecordLatency(duration time.Duration) {
	ms := float64(duration.Milliseconds())

	m.latencyMu.Lock()
	defer m.latencyMu.Unlock()

	m.latencies[m.latencyIndex] = ms
	m.latencyIndex = (m.latencyIndex + 1) % m.maxLatencyBuf
	if m.latencyCount < m.maxLatencyBuf {
		m.latencyCount++
	}
}

/**
 * GetLatencyStats returns p50 and p99 latency in milliseconds.
 */
func (m *EngineMetrics) GetLatencyStats() (p50, p99 float64) {
	m.latencyMu.RLock()
	defer m.latencyMu.RUnlock()

	if m.latencyCount == 0 {
		return 0, 0
	}

	// Copy and sort latencies
	sorted := make([]float64, m.latencyCount)
	start := m.latencyIndex - m.latencyCount
	if start < 0 {
		start += m.maxLatencyBuf
	}
	for i := 0; i < m.latencyCount; i++ {
		sorted[i] = m.latencies[(start+i)%m.maxLatencyBuf]
	}

	// Simple bubble sort (sufficient for small N)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	p50Idx := len(sorted) / 2
	p99Idx := len(sorted) * 99 / 100
	if p99Idx >= len(sorted) {
		p99Idx = len(sorted) - 1
	}

	return sorted[p50Idx], sorted[p99Idx]
}

/**
 * MarkShedding records an observation dropped due to backpressure.
 */
func (m *EngineMetrics) MarkShedding() {
	atomic.AddUint64(&m.ShedEvents, 1)
}

/**
 * GetStochasticHealth returns the current utilization (rho) and throughput.
 * Formula: rho = lambda / (c * mu)
 */
func (m *EngineMetrics) GetStochasticHealth(workerCount int) map[string]float64 {
	elapsed := time.Since(m.startTime).Seconds()
	if elapsed == 0 {
		return nil
	}

	lambda := float64(atomic.LoadUint64(&m.Arrivals)) / elapsed
	mu := float64(atomic.LoadUint64(&m.Completions)) / elapsed / float64(workerCount)

	rho := 0.0
	if mu > 0 {
		rho = lambda / (float64(workerCount) * mu)
	}

	return map[string]float64{
		"lambda_hz":    lambda,
		"worker_mu_hz": mu,
		"rho_util":     rho,
		"queue_depth":  float64(atomic.LoadInt64(&m.QueueLength)),
		"shed_rate":    float64(atomic.LoadUint64(&m.ShedEvents)) / elapsed,
	}
}

// MarkBackpressure records a backpressure event.
func (m *EngineMetrics) MarkBackpressure() {
	atomic.AddUint64(&m.ShedEvents, 1)
}

// GetQueueDepth returns the current queue depth.
func (m *EngineMetrics) GetQueueDepth() int64 {
	return atomic.LoadInt64(&m.QueueLength)
}

// IncQueueDepth increments the queue depth counter.
func (m *EngineMetrics) IncQueueDepth() {
	atomic.AddInt64(&m.QueueLength, 1)
}

// RecordServiceTime records service time.
func (m *EngineMetrics) RecordServiceTime(duration time.Duration) {
	m.RecordLatency(duration)
}

// IncProcessed increments processed counter.
func (m *EngineMetrics) IncProcessed() {
	atomic.AddUint64(&m.Completions, 1)
}

// Alert counting methods
func (m *EngineMetrics) IncCriticalAlert() {
	atomic.AddUint64(&m.criticalAlerts, 1)
}

func (m *EngineMetrics) IncHighAlert() {
	atomic.AddUint64(&m.highAlerts, 1)
}

func (m *EngineMetrics) IncMediumAlert() {
	atomic.AddUint64(&m.mediumAlerts, 1)
}

func (m *EngineMetrics) IncLowAlert() {
	atomic.AddUint64(&m.lowAlerts, 1)
}

// GetAlertCounts returns current alert counts.
func (m *EngineMetrics) GetAlertCounts() (critical, high, medium, low uint64) {
	return atomic.LoadUint64(&m.criticalAlerts),
		atomic.LoadUint64(&m.highAlerts),
		atomic.LoadUint64(&m.mediumAlerts),
		atomic.LoadUint64(&m.lowAlerts)
}
