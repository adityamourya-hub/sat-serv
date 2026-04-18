package engine

import (
	"fmt"
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

	// Temporal snapshots for rate calculation
	startTime time.Time
}

/**
 * NewEngineMetrics initializes the stochastic monitoring substrate.
 */
func NewEngineMetrics() *EngineMetrics {
	return &EngineMetrics{
		startTime: time.Now(),
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
 * MarkShedding records an observation dropped due to backpressure.
 */
func (m *EngineMetrics) MarkShedding() {
	fmt.Printf("DEBUG: Engine shedding event at %v\n", time.Now())
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

// RecordServiceTime records service time (placeholder).
func (m *EngineMetrics) RecordServiceTime(duration time.Duration) {
	// TODO: Implement latency histogram
}

// IncProcessed increments processed counter (placeholder).
func (m *EngineMetrics) IncProcessed() {
	// TODO: Implement processed tracking
}
