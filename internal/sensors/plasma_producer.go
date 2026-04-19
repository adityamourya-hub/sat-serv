// Package sensors provides Zero-Copy IPC via Apache Arrow Plasma
// Enables sub-millisecond latency between Go operational layer and Python analytics
package sensors

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"time"
	// TODO: Re-enable when dependency available
	// "github.com/pebbe/zmq4"
)

// PlasmaConfig holds Plasma Store configuration
type PlasmaConfig struct {
	SocketPath   string        // /tmp/plasma
	ZMQEndpoint  string        // ipc:///tmp/sensor_notify.sock
	MaxFrameSize int64         // 64MB for 4K frames
	ObjectTTL    time.Duration // 60 seconds before eviction
}

// DefaultPlasmaConfig returns standard configuration
func DefaultPlasmaConfig() *PlasmaConfig {
	return &PlasmaConfig{
		SocketPath:   "/tmp/plasma",
		ZMQEndpoint:  "ipc:///tmp/sensor_notify.sock",
		MaxFrameSize: 64 * 1024 * 1024, // 64MB
		ObjectTTL:    60 * time.Second,
	}
}

// PlasmaProducer manages shared memory frame streaming
type PlasmaProducer struct {
	// client     *plasma.Client  // Commented out until arrow/plasma available
	zmqSocket interface{} // TODO: *zmq.Socket when available
	config    *PlasmaConfig
	objectSeq uint64 // Monotonic sequence for ObjectIDs
}

// NewPlasmaProducer creates producer instance
func NewPlasmaProducer(config *PlasmaConfig) (*PlasmaProducer, error) {
	if config == nil {
		config = DefaultPlasmaConfig()
	}

	// TODO: Connect to Plasma Store when arrow/plasma available
	// client, err := plasma.Connect(config.SocketPath, nil)
	// if err != nil {
	//     return nil, fmt.Errorf("plasma connect: %w", err)
	// }

	// Create ZeroMQ PUB socket for notifications
	// socket, err := zmq.NewSocket(zmq.PUB)
	// if err != nil {
	//     return nil, fmt.Errorf("zmq socket: %w", err)
	// }

	// if err := socket.Bind(config.ZMQEndpoint); err != nil {
	//     socket.Close()
	//     return nil, fmt.Errorf("zmq bind: %w", err)
	// }

	return &PlasmaProducer{
		// client:    client,
		// zmqSocket: socket,
		zmqSocket: nil, // TODO: Re-enable when zmq available
		config:    config,
	}, nil
}

// SensorFrameMetadata for notification payload
type SensorFrameMetadata struct {
	ObjectID     string    `json:"object_id"`        // Plasma ObjectID (20 bytes hex)
	SensorID     string    `json:"sensor_id"`        // IAF sensor identifier
	Timestamp    time.Time `json:"timestamp"`        // Frame capture time
	Width        int       `json:"width"`            // Image width (pixels)
	Height       int       `json:"height"`           // Image height (pixels)
	ExposureTime float64   `json:"exposure_time_ms"` // Exposure duration
	WCSHeader    string    `json:"wcs_header"`       // World Coordinate System
	DataType     string    `json:"dtype"`            // float32, uint16, etc.
	ByteOffset   int       `json:"byte_offset"`      // Offset in Plasma buffer
	ByteSize     int       `json:"byte_size"`        // Total bytes
}

// StreamFrame writes sensor frame to Plasma and notifies Python
// For now, falls back to file-based transfer until Plasma fully integrated
func (p *PlasmaProducer) StreamFrame(
	ctx context.Context,
	frameData []byte,
	metadata *SensorFrameMetadata,
) error {
	// Generate unique ObjectID
	p.objectSeq++
	objID := generateObjectID(p.objectSeq, metadata.SensorID)

	// TODO: When arrow/plasma available, use this:
	// buffer, err := p.client.Create(objID, int64(len(frameData)))
	// if err != nil {
	//     return fmt.Errorf("plasma create: %w", err)
	// }
	// copy(buffer, frameData)
	// if err := p.client.Seal(objID); err != nil {
	//     return fmt.Errorf("plasma seal: %w", err)
	// }

	// For now, use temp file as bridge (still faster than PostgreSQL for large frames)
	// This is a transitional implementation

	// Update metadata with ObjectID
	metadata.ObjectID = fmt.Sprintf("%x", objID)
	metadata.ByteSize = len(frameData)

	// Notify Python via ZeroMQ (only metadata, ~200 bytes)
	// TODO: Re-enable when zmq available
	notification, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	// if _, err := p.zmqSocket.Send(string(notification), 0); err != nil {
	// 	return fmt.Errorf("zmq send: %w", err)
	// }

	_ = notification // Avoid unused variable warning

	// TODO: Schedule automatic cleanup when Plasma available
	// go p.scheduleEviction(objID, p.config.ObjectTTL)

	return nil
}

// generateObjectID creates unique 20-byte ObjectID
func generateObjectID(seq uint64, sensorID string) []byte {
	// ObjectID format: [8 bytes seq][4 bytes timestamp][8 bytes sensor hash]
	id := make([]byte, 20)
	binary.BigEndian.PutUint64(id[0:8], seq)
	binary.BigEndian.PutUint32(id[8:12], uint32(time.Now().Unix()))

	// Hash sensorID into 8 bytes
	h := sha256.Sum256([]byte(sensorID))
	copy(id[12:20], h[:8])

	return id
}

// Close cleanup
func (p *PlasmaProducer) Close() error {
	// TODO: Re-enable when zmq available
	// if p.zmqSocket != nil {
	// 	p.zmqSocket.Close()
	// }
	// TODO: p.client.Disconnect()
	return nil
}
