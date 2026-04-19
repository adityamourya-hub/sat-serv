package engine

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"

	"github.com/apache/arrow/go/v14/arrow/plasma"
)

/**
 * Plasma Bridge: Zero-Copy Memory Sharing between Go and Python
 *
 * Implements Apache Arrow Plasma for 100Hz sensor data streaming.
 * Eliminates serialization overhead - Go writes directly to shared memory,
 * Python maps as NumPy arrays without copying.
 *
 * Critical for M2 Air: Unified memory pool allows GPU (Metal) access
 * to the same buffers without PCIe transfer.
 *
 * Reference: Arrow Plasma - Shared Memory Object Store
 */

const (
	DefaultPlasmaSocket = "/tmp/plasma"
	FrameBufferSize     = 10   // Circular buffer (FRAME_0 to FRAME_9)
	MaxPlasmaMemoryMB   = 2048 // 2GB cap for M2 Air
	ObjectIDLength      = 20   // Arrow ObjectID size
)

// PlasmaBridge manages zero-copy data transfer to Python ML pipeline
type PlasmaBridge struct {
	client       *plasma.Client
	bufferPool   *sync.Pool
	currentFrame int                     // Circular buffer index
	frameIDs     [FrameBufferSize][]byte // Pre-allocated ObjectIDs

	// Metrics
	writesTotal  uint64
	writeLatency time.Duration
	lastError    error
}

// FrameMetadata describes sensor frame layout for Python consumer
type FrameMetadata struct {
	Timestamp  int64   `json:"timestamp"` // Unix nanoseconds
	SensorID   string  `json:"sensor_id"` // e.g., "leh_optical_01"
	Width      int32   `json:"width"`     // Pixels
	Height     int32   `json:"height"`
	Channels   int32   `json:"channels"` // 1 for mono, 3 for RGB
	Datatype   string  `json:"datatype"` // "float32", "uint16"
	ExposureMs float32 `json:"exposure_ms"`
	Azimuth    float32 `json:"azimuth"` // Telescope pointing
	Elevation  float32 `json:"elevation"`
}

// NewPlasmaBridge creates a zero-copy bridge to Python analytics
func NewPlasmaBridge(socketPath string) (*PlasmaBridge, error) {
	if socketPath == "" {
		socketPath = DefaultPlasmaSocket
	}

	client, err := plasma.NewClient(socketPath, plasma.WithHugePagesEnabled(false))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to plasma store: %w", err)
	}

	bridge := &PlasmaBridge{
		client:       client,
		currentFrame: 0,
		bufferPool: &sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, 2048*2048*4) // 16MB max frame
			},
		},
	}

	// Pre-allocate circular buffer ObjectIDs
	for i := 0; i < FrameBufferSize; i++ {
		bridge.frameIDs[i] = []byte(fmt.Sprintf("SSIS_SENSOR_FRAME_%02d", i))
	}

	return bridge, nil
}

// WriteSensorFrame writes raw sensor data to shared memory
// This is the CRITICAL PATH - must complete in < 2ms for 100Hz
func (p *PlasmaBridge) WriteSensorFrame(data []float32, meta FrameMetadata) error {
	start := time.Now()

	// Get next ObjectID from circular buffer
	objectID := p.frameIDs[p.currentFrame]
	p.currentFrame = (p.currentFrame + 1) % FrameBufferSize

	id := plasma.NewID(objectID)

	// Calculate size: data + metadata
	dataSize := int64(len(data) * 4) // float32 = 4 bytes
	metaSize := int64(256)           // JSON metadata buffer
	totalSize := dataSize + metaSize

	// 1. Create object in Plasma (allocates shared memory)
	buf, err := p.client.Create(context.Background(), id, totalSize, nil)
	if err != nil {
		p.lastError = err
		return fmt.Errorf("plasma create failed: %w", err)
	}

	// 2. Write data directly to shared memory buffer
	// This is the ONLY copy - from Go heap to shared mmap
	dataBytes := float32SliceToBytes(data)
	copy(buf, dataBytes)

	// 3. Write metadata after data
	metaBytes := encodeMetadata(meta)
	metaOffset := len(dataBytes)
	copy(buf[metaOffset:], metaBytes)

	// 4. Seal object - makes it immutable and visible to Python
	if err := p.client.Seal(context.Background(), id); err != nil {
		p.lastError = err
		return fmt.Errorf("plasma seal failed: %w", err)
	}

	// 5. (Optional) Set expiration for LRU eviction
	// Plasma will auto-evict when memory pressure hits 2GB
	p.client.Release(context.Background(), id)

	p.writesTotal++
	p.writeLatency = time.Since(start)

	return nil
}

// GetCurrentFrameID returns the ObjectID for the most recent frame
// Send this via gRPC to Python (tiny message, not the data!)
func (p *PlasmaBridge) GetCurrentFrameID() []byte {
	// Return previous frame (current is being written)
	prev := (p.currentFrame - 1 + FrameBufferSize) % FrameBufferSize
	return p.frameIDs[prev]
}

// WriteTensor writes high-dimensional tensor for JAX/Metal
// Supports arbitrary shapes for ML pipeline input
func (p *PlasmaBridge) WriteTensor(data []float32, shape []int64, name string) error {
	objectID := []byte(fmt.Sprintf("SSIS_TENSOR_%s_%d", name, time.Now().UnixNano()))
	id := plasma.NewID(objectID)

	size := int64(len(data) * 4)

	buf, err := p.client.Create(context.Background(), id, size, nil)
	if err != nil {
		return err
	}

	// Copy tensor data
	dataBytes := float32SliceToBytes(data)
	copy(buf, dataBytes)

	// Seal and release
	if err := p.client.Seal(context.Background(), id); err != nil {
		return err
	}
	p.client.Release(context.Background(), id)

	return nil
}

// GetMetrics returns bridge performance stats
func (p *PlasmaBridge) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"writes_total":          p.writesTotal,
		"last_write_latency_ms": p.writeLatency.Seconds() * 1000,
		"current_buffer":        p.currentFrame,
		"last_error":            p.lastError,
	}
}

// Close shuts down plasma connection
func (p *PlasmaBridge) Close() error {
	return p.client.Close()
}

// Helper: Convert float32 slice to bytes without allocation
func float32SliceToBytes(data []float32) []byte {
	// Unsafe but fast - reinterprets slice without copy
	// Safe because Plasma will copy this to mmap anyway
	header := *(*unsafe.Pointer)(unsafe.Pointer(&data))
	return unsafe.Slice((*byte)(header), len(data)*4)
}

// Helper: Encode metadata as binary (faster than JSON for 100Hz)
func encodeMetadata(meta FrameMetadata) []byte {
	buf := make([]byte, 256)

	// Binary encoding: fixed offsets for speed
	binary.LittleEndian.PutUint64(buf[0:8], uint64(meta.Timestamp))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(meta.Width))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(meta.Height))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(meta.Channels))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(meta.Azimuth))
	binary.LittleEndian.PutUint32(buf[24:28], math.Float32bits(meta.Elevation))
	binary.LittleEndian.PutUint32(buf[28:32], math.Float32bits(meta.ExposureMs))

	// Sensor ID as null-terminated string
	copy(buf[32:63], meta.SensorID)

	return buf
}

// SensorConsumerBridge interfaces with internal/sensors
// This replaces the old gRPC-based PythonBridge
type SensorConsumerBridge struct {
	plasma   *PlasmaBridge
	streamID string

	// Batch processing
	batchSize   int
	batchBuffer []float32
	batchCount  int
}

// NewSensorConsumerBridge creates optimized sensor-to-ML bridge
func NewSensorConsumerBridge(streamID string) (*SensorConsumerBridge, error) {
	plasma, err := NewPlasmaBridge("")
	if err != nil {
		return nil, err
	}

	return &SensorConsumerBridge{
		plasma:      plasma,
		streamID:    streamID,
		batchSize:   1, // 100Hz = no batching, direct pass-through
		batchBuffer: make([]float32, 0, 2048*2048),
	}, nil
}

// OnSensorFrame processes incoming sensor data
// Call this at 100Hz from internal/sensors/processor.go
func (scb *SensorConsumerBridge) OnSensorFrame(frame []float32, timestamp time.Time, az, el float32) error {
	meta := FrameMetadata{
		Timestamp: timestamp.UnixNano(),
		SensorID:  scb.streamID,
		Width:     2048,
		Height:    2048,
		Channels:  1,
		Datatype:  "float32",
		Azimuth:   az,
		Elevation: el,
	}

	return scb.plasma.WriteSensorFrame(frame, meta)
}

// GetFrameIDForPython returns the ObjectID to send to Python
// Python uses this to mmap the shared memory
func (scb *SensorConsumerBridge) GetFrameIDForPython() []byte {
	return scb.plasma.GetCurrentFrameID()
}

// Metrics exposes performance counters
func (scb *SensorConsumerBridge) Metrics() map[string]interface{} {
	return scb.plasma.GetMetrics()
}
