// Package tracking provides circular buffers for high-performance temporal data storage
// Used for light-curve windows and state vector history without GC pressure
package tracking

import (
	"sync"
	"time"
)

// CircularBuffer provides fixed-size ring buffer for time-series data
// Zero-allocation after initialization - critical for 45k satellite tracking
type CircularBuffer struct {
	mu sync.RWMutex
	
	// Fixed-size array (pre-allocated)
	data []BufferEntry
	
	// Current position
	head int  // Write position
	tail int  // Read position (for iteration)
	
	// Capacity
	size   int
	count  int  // Current number of elements
	
	// Satellite ID this buffer belongs to
	satelliteID int
}

// BufferEntry stores a single time-series sample
type BufferEntry struct {
	Timestamp     time.Time
	Intensity     float64     // Light-curve magnitude/flux
	StateVector   [6]float64  // [x, y, z, vx, vy, vz]
	BStar         float64     // Ballistic coefficient
	Covariance    [21]float64 // Upper triangle of 6x6 covariance (optional)
}

// NewCircularBuffer creates a pre-allocated ring buffer
func NewCircularBuffer(satelliteID, capacity int) *CircularBuffer {
	return &CircularBuffer{
		data:        make([]BufferEntry, capacity),
		size:        capacity,
		satelliteID: satelliteID,
		head:        0,
		tail:        0,
		count:       0,
	}
}

// Push adds a new entry to the buffer
// O(1) complexity, zero-allocation
func (cb *CircularBuffer) Push(entry BufferEntry) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	// Write at head position
	cb.data[cb.head] = entry
	
	// Advance head
	cb.head = (cb.head + 1) % cb.size
	
	// Update count (cap at size)
	if cb.count < cb.size {
		cb.count++
	} else {
		// Buffer full, advance tail too
		cb.tail = (cb.tail + 1) % cb.size
	}
}

// GetLastN returns the last N entries (most recent first)
// Returns available entries if fewer than N exist
func (cb *CircularBuffer) GetLastN(n int) []BufferEntry {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	if n > cb.count {
		n = cb.count
	}
	
	result := make([]BufferEntry, n)
	
	// Start from head-1 (most recent)
	pos := (cb.head - 1 + cb.size) % cb.size
	
	for i := 0; i < n; i++ {
		result[i] = cb.data[pos]
		pos = (pos - 1 + cb.size) % cb.size
	}
	
	return result
}

// GetWindow returns entries within a time window
func (cb *CircularBuffer) GetWindow(startTime, endTime time.Time) []BufferEntry {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	result := make([]BufferEntry, 0)
	
	// Iterate through buffer
	for i := 0; i < cb.count; i++ {
		idx := (cb.tail + i) % cb.size
		entry := cb.data[idx]
		
		if entry.Timestamp.After(startTime) && entry.Timestamp.Before(endTime) {
			result = append(result, entry)
		}
	}
	
	return result
}

// Latest returns the most recent entry
func (cb *CircularBuffer) Latest() (BufferEntry, bool) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	if cb.count == 0 {
		return BufferEntry{}, false
	}
	
	pos := (cb.head - 1 + cb.size) % cb.size
	return cb.data[pos], true
}

// Full returns true if buffer is at capacity
func (cb *CircularBuffer) Full() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	return cb.count == cb.size
}

// Count returns current number of entries
func (cb *CircularBuffer) Count() int {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	return cb.count
}

// Clear resets the buffer
func (cb *CircularBuffer) Clear() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	
	cb.head = 0
	cb.tail = 0
	cb.count = 0
	
	// Clear data (optional - helps GC)
	for i := range cb.data {
		cb.data[i] = BufferEntry{}
	}
}

// CircularBufferManager manages buffers for all satellites
type CircularBufferManager struct {
	mu      sync.RWMutex
	buffers map[int]*CircularBuffer  // satelliteID -> buffer
	size    int                      // Default buffer size
}

// NewCircularBufferManager creates a manager for satellite light-curve buffers
func NewCircularBufferManager(bufferSize int) *CircularBufferManager {
	return &CircularBufferManager{
		buffers: make(map[int]*CircularBuffer),
		size:    bufferSize,
	}
}

// GetOrCreate returns buffer for satellite (creates if needed)
func (cbm *CircularBufferManager) GetOrCreate(satelliteID int) *CircularBuffer {
	cbm.mu.RLock()
	if buf, exists := cbm.buffers[satelliteID]; exists {
		cbm.mu.RUnlock()
		return buf
	}
	cbm.mu.RUnlock()
	
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	
	// Double-check after acquiring write lock
	if buf, exists := cbm.buffers[satelliteID]; exists {
		return buf
	}
	
	// Create new buffer
	buf := NewCircularBuffer(satelliteID, cbm.size)
	cbm.buffers[satelliteID] = buf
	return buf
}

// Remove deletes a buffer
func (cbm *CircularBufferManager) Remove(satelliteID int) {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	
	delete(cbm.buffers, satelliteID)
}

// GetStats returns statistics for all buffers
func (cbm *CircularBufferManager) GetStats() map[string]int {
	cbm.mu.RLock()
	defer cbm.mu.RUnlock()
	
	totalEntries := 0
	totalCapacity := 0
	
	for _, buf := range cbm.buffers {
		totalEntries += buf.Count()
		totalCapacity += buf.size
	}
	
	return map[string]int{
		"num_buffers":    len(cbm.buffers),
		"total_entries":  totalEntries,
		"total_capacity": totalCapacity,
		"fill_percentage": (totalEntries * 100) / (totalCapacity + 1),
	}
}
