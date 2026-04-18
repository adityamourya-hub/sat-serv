package onnx

import (
	"fmt"
	"sync"
	"time"
)

/**
 * InferenceRequest represents a single packet submitted to the ONNX substrate.
 * It includes an internal result channel to enable asynchronous batching.
 */
type InferenceRequest struct {
	ID       string
	ModelID  string
	Version  string
	Inputs   map[string]interface{}
	ResultCh chan InferenceResult // Used for async synchronization
}

/**
 * InferenceResult represents the output of the model integration.
 */
type InferenceResult struct {
	RequestID  string
	ModelID    string
	Version    string
	Outputs    map[string]interface{}
	Confidence float64
	LatencyMs  float64
	Error      error
}

/**
 * SessionConfig provides parameters for provider-agile inference.
 */
type SessionConfig struct {
	ModelPath         string
	ModelID           string
	Version           string
	MaxBatchSize      int    // Target: 32 for Phase 1.6
	ExecutionProvider string // "CPU", "CUDA", or "ONEDNN"
	IntraOpNumThreads int
	EnableLogging     bool
}

// DefaultSessionConfig creates a default session configuration.
func DefaultSessionConfig(modelID, modelPath, version string) SessionConfig {
	return SessionConfig{
		ModelPath:         modelPath,
		ModelID:           modelID,
		Version:           version,
		MaxBatchSize:      32,
		ExecutionProvider: "CPU",
		IntraOpNumThreads: 1,
		EnableLogging:     false,
	}
}

/**
 * Session manages a thread-safe, batched ONNX runtime instance.
 * Refactored in Phase 1.6 to use a Temporal Collator with Static Padding.
 *
 * Logic: Every 50ms, the batchProcessor dispatches whatever is in the buffer.
 * If the buffer is < MaxBatchSize, it pads the request to ensure deterministic
 * forward pass latency and memory stability.
 *
 * <plan_complexity>High</plan_complexity><total_lines_estimate>350</total_lines_estimate>
 */
type Session struct {
	config      SessionConfig
	inferrer    Inferrer
	requestCh   chan InferenceRequest
	stopCh      chan struct{}
	wg          sync.WaitGroup
	initialized bool
}

/**
 * Inferrer defines the CGO/Mock boundary for ONNX execution.
 */
type Inferrer interface {
	InferBatch(requests []InferenceRequest) []InferenceResult
	Close() error
}

/**
 * NewSession initializes the managed inference session.
 */
func NewSession(config SessionConfig, inferrer Inferrer) *Session {
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 32 // Production sweet-spot for 50ms p99
	}

	s := &Session{
		config:    config,
		inferrer:  inferrer,
		requestCh: make(chan InferenceRequest, config.MaxBatchSize*2),
		stopCh:    make(chan struct{}),
	}

	s.wg.Add(1)
	go s.batchProcessor()
	s.initialized = true

	return s
}

/**
 * Infer submits a request and waits for the pooled batch to complete.
 * This effectively hides the CGO context-switching cost under the batch overhead.
 */
func (s *Session) Infer(inputs map[string]interface{}) (map[string]interface{}, error) {
	// Entry Point: Async Inference Handoff
	resCh := make(chan InferenceResult, 1)
	req := InferenceRequest{
		ID:       fmt.Sprintf("req-%d", time.Now().UnixNano()),
		Inputs:   inputs,
		ResultCh: resCh,
	}

	select {
	case s.requestCh <- req:
	case <-s.stopCh:
		return nil, fmt.Errorf("session stopped")
	}

	res := <-resCh
	// Exit Point: Result Retrieved
	return res.Outputs, res.Error
}

/**
 * batchProcessor implements the Temporal Collator logic.
 * Enforces the 50ms p99 ceiling and executes Static Padding (32).
 */
func (s *Session) batchProcessor() {
	defer s.wg.Done()
	fmt.Printf("DEBUG: ONNX Batch Processor active for %s (Limit=50ms, Size=%d)\n", s.config.ModelID, s.config.MaxBatchSize)

	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	batch := make([]InferenceRequest, 0, s.config.MaxBatchSize)

	for {
		select {
		case <-s.stopCh:
			return
		case req := <-s.requestCh:
			batch = append(batch, req)
			if len(batch) >= s.config.MaxBatchSize {
				s.dispatchBatch(batch)
				batch = batch[:0]
				ticker.Reset(50 * time.Millisecond) // Reset timer after full batch dispatch
			}
		case <-ticker.C:
			if len(batch) > 0 {
				s.dispatchBatch(batch)
				batch = batch[:0]
			}
		}
	}
}

/**
 * dispatchBatch applies static padding and executes the forward pass.
 */
func (s *Session) dispatchBatch(batch []InferenceRequest) {
	// Entry Point: Batch Dispatch
	fmt.Printf("DEBUG: Dispatching batch of size %d for %s\n", len(batch), s.config.ModelID)

	actualLen := len(batch)
	paddedBatch := batch

	// 1. Static Padding logic (Phase 1.6 Requirement)
	// We pad the batch to exactly MaxBatchSize to prevent dynamic shape re-alloc in ONNX.
	if actualLen < s.config.MaxBatchSize {
		padding := s.config.MaxBatchSize - actualLen
		for i := 0; i < padding; i++ {
			// Pad with zero-data / null requests
			paddedBatch = append(paddedBatch, InferenceRequest{ID: "PADDING"})
		}
	}

	// 2. CGO execution
	results := s.inferrer.InferBatch(paddedBatch)

	// 3. Result Fan-out
	// Only notify the actual request channels, ignore padding slots.
	for i := 0; i < actualLen; i++ {
		batch[i].ResultCh <- results[i]
	}

	// Exit Point: Batch Processed
}

/**
 * Close gracefully shuts down the batch processor.
 */
func (s *Session) Close() error {
	close(s.stopCh)
	s.wg.Wait()
	return s.inferrer.Close()
}

/**
 * Metrics returns runtime statistics for the session.
 */
func (s *Session) Metrics() map[string]interface{} {
	return map[string]interface{}{
		"model_id":           s.config.ModelID,
		"version":            s.config.Version,
		"max_batch_size":     s.config.MaxBatchSize,
		"execution_provider": s.config.ExecutionProvider,
		"initialized":        s.initialized,
	}
}
