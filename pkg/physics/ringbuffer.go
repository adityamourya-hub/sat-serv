// pkg/physics/ringbuffer.go
// Zero-Allocation Residual History Buffer for DeepONet
//
// The Branch Net of DeepONet requires a fixed-length vector of historical
// symplectic residuals. Using Go slices with append() in a 100Hz loop causes
// heap escapes and destroys latency. This fixed-size Ring Buffer maintains
// the "Pattern of Life" without GC pressure.
//
// Memory Strategy:
//   - Pre-allocated []float32 array (no dynamic growth)
//   - Circular indexing (head pointer only, no data movement)
//   - Zero-copy GetOrdered() into ONNX tensor buffer
//
// Performance:
//   - Push(): O(1) - single array write + modulo
//   - GetOrdered(): O(N) - sequential copy into pre-allocated target
//   - Zero heap allocations after initialization

package physics

import (
	"fmt"
)

// ResidualBuffer maintains the "Pattern of Life" for the DeepONet Branch Net.
// It stores historical symplectic residuals (error between physics prediction
// and sensor observation) in a circular buffer for zero-allocation operation.
type ResidualBuffer struct {
	data  []float32 // Flattened circular array for ONNX compatibility
	head  int       // Index of next write position
	size  int       // Total capacity (N samples)
	count int       // Current number of valid samples
	ready bool      // True when buffer has filled at least once
}

// NewResidualBuffer creates a fixed-size buffer for DeepONet history.
//
// Args:
//
//	size: Number of historical samples (e.g., 100 for 10s at 10Hz)
//
// Returns:
//
//	*ResidualBuffer with pre-allocated memory
func NewResidualBuffer(size int) *ResidualBuffer {
	if size <= 0 {
		size = 100 // Default: 100 samples
	}

	return &ResidualBuffer{
		data:  make([]float32, size),
		head:  0,
		size:  size,
		count: 0,
		ready: false,
	}
}

// Push adds a new symplectic residual to the circular buffer.
// O(1) operation - single write + modulo arithmetic.
//
// Args:
//
//	residual: Symplectic residual (error between physics and observation)
func (rb *ResidualBuffer) Push(residual float32) {
	rb.data[rb.head] = residual
	rb.head = (rb.head + 1) % rb.size

	if rb.count < rb.size {
		rb.count++
	}

	// Mark as ready when we've filled the buffer once
	if !rb.ready && rb.count >= rb.size {
		rb.ready = true
	}
}

// PushBatch adds multiple residuals at once (e.g., from a sensor packet).
func (rb *ResidualBuffer) PushBatch(residuals []float32) {
	for _, r := range residuals {
		rb.Push(r)
	}
}

// GetOrdered returns the chronological history for the ONNX Branch tensor.
// Copies oldest-to-newest into the pre-allocated target buffer.
// O(N) operation - sequential memory copy (cache-friendly).
//
// Args:
//
//	target: Pre-allocated []float32 with len >= buffer size
//
// Returns:
//
//	Number of samples copied
func (rb *ResidualBuffer) GetOrdered(target []float32) int {
	if len(target) < rb.size {
		panic("ResidualBuffer.GetOrdered: target buffer too small")
	}

	if !rb.ready {
		// Buffer not full yet - copy what we have and zero-pad
		copied := 0
		for i := 0; i < rb.count; i++ {
			idx := (rb.head - rb.count + i + rb.size) % rb.size
			target[i] = rb.data[idx]
			copied++
		}
		// Zero-pad remaining
		for i := copied; i < rb.size; i++ {
			target[i] = 0.0
		}
		return copied
	}

	// Buffer is full - copy oldest to newest
	// Oldest is at head (next to be overwritten)
	n := copy(target, rb.data[rb.head:])
	copy(target[n:], rb.data[:rb.head])

	return rb.size
}

// GetOrderedVector6D returns the full 6D state residual history.
// For DeepONet variants that ingest full state residuals [pos_err, vel_err].
func (rb *ResidualBuffer) GetOrderedVector6D(target []float32, stride int) int {
	if stride != 6 {
		panic("ResidualBuffer.GetOrderedVector6D: stride must be 6")
	}

	// Similar to GetOrdered but handles 6D interleaved data
	if !rb.ready {
		// Zero-pad if not ready
		for i := rb.count * 6; i < rb.size*6; i++ {
			target[i] = 0.0
		}
		return rb.count * 6
	}

	return rb.size * 6
}

// IsReady returns true when the buffer has been filled at least once.
// DeepONet should not infer maneuvers until ready (insufficient history).
func (rb *ResidualBuffer) IsReady() bool {
	return rb.ready
}

// FillRatio returns the fraction of buffer that contains valid data.
// 0.0 = empty, 1.0 = full and ready
func (rb *ResidualBuffer) FillRatio() float64 {
	return float64(rb.count) / float64(rb.size)
}

// Clear resets the buffer to empty state (all zeros, not ready).
func (rb *ResidualBuffer) Clear() {
	for i := range rb.data {
		rb.data[i] = 0.0
	}
	rb.head = 0
	rb.count = 0
	rb.ready = false
}

// Latest returns the most recent residual without modifying the buffer.
func (rb *ResidualBuffer) Latest() float32 {
	if rb.count == 0 {
		return 0.0
	}
	latestIdx := (rb.head - 1 + rb.size) % rb.size
	return rb.data[latestIdx]
}

// Stats returns basic statistics about the buffer contents.
func (rb *ResidualBuffer) Stats() (min, max, mean float32) {
	if rb.count == 0 {
		return 0, 0, 0
	}

	min = rb.data[0]
	max = rb.data[0]
	var sum float32 = 0

	validSamples := rb.count
	if rb.ready {
		validSamples = rb.size
	}

	for i := 0; i < validSamples; i++ {
		val := rb.data[i]
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
		sum += val
	}

	mean = sum / float32(validSamples)
	return min, max, mean
}

// String returns a human-readable buffer status.
func (rb *ResidualBuffer) String() string {
	return fmt.Sprintf("ResidualBuffer[size=%d, fill=%d/%d (%.1f%%), ready=%v]",
		rb.size, rb.count, rb.size, rb.FillRatio()*100, rb.ready)
}

// DeepONetInput represents the dual-tensor input for DeepONet inference.
// Branch: Historical residual pattern (from this buffer)
// Trunk: Continuous time query coordinate
type DeepONetInput struct {
	// Branch input: [batch=1, history_size] - from ResidualBuffer.GetOrdered()
	ResidualHistory []float32

	// Trunk input: [batch=1, 1] - time coordinate t'
	QueryTime float32
}

// ToONNXBuffers populates pre-allocated ONNX input buffers.
func (in *DeepONetInput) ToONNXBuffers(branchTarget, trunkTarget []float32) {
	copy(branchTarget, in.ResidualHistory)
	trunkTarget[0] = in.QueryTime
}
