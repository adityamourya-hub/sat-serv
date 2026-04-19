// Package mesh provides Fire-Control Handover serialization
// Implements MessagePack encoding for compact binary transmission
package mesh

import (
	"encoding/json"
	"fmt"
	"time"
	// TODO: Re-enable when msgpack dependency available
	// "github.com/vmihailenco/msgpack/v5"
)

// FCHSerializer handles serialization of FireControlBrief to MessagePack
type FCHSerializer struct {
	maxPacketSize int
}

// NewFCHSerializer creates a new FCH serializer
func NewFCHSerializer() *FCHSerializer {
	return &FCHSerializer{
		maxPacketSize: 512, // LoRa MTU limit
	}
}

// SetMaxPacketSize sets the maximum packet size (default 512 bytes)
func (s *FCHSerializer) SetMaxPacketSize(size int) {
	s.maxPacketSize = size
}

// Serialize converts FireControlBrief to MessagePack bytes
func (s *FCHSerializer) Serialize(brief *FireControlBrief) ([]byte, error) {
	// Validate before serialization
	if err := brief.Validate(); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Serialize to MessagePack (or JSON as fallback)
	// TODO: Re-enable msgpack when dependency available
	// data, err := msgpack.Marshal(brief)
	data, err := json.Marshal(brief)
	if err != nil {
		return nil, fmt.Errorf("marshal failed: %w", err)
	}

	// Check size limit
	if len(data) > s.maxPacketSize {
		return nil, fmt.Errorf("serialized size %d bytes exceeds limit %d bytes", len(data), s.maxPacketSize)
	}

	return data, nil
}

// Deserialize converts MessagePack bytes to FireControlBrief
func (s *FCHSerializer) Deserialize(data []byte) (*FireControlBrief, error) {
	var brief FireControlBrief

	// TODO: Re-enable msgpack when dependency available
	// if err := msgpack.Unmarshal(data, &brief); err != nil {
	if err := json.Unmarshal(data, &brief); err != nil {
		return nil, fmt.Errorf("unmarshal failed: %w", err)
	}

	// Validate after deserialization
	if err := brief.Validate(); err != nil {
		return nil, fmt.Errorf("deserialized data validation failed: %w", err)
	}

	return &brief, nil
}

// SerializeCompact creates a minimal FCH packet (without optional fields)
func (s *FCHSerializer) SerializeCompact(brief *FireControlBrief) ([]byte, error) {
	// Create a compact version with only essential fields
	compact := &FireControlBrief{
		HandoverID:   brief.HandoverID,
		Timestamp:    brief.Timestamp,
		TargetID:     brief.TargetID,
		Pos:          brief.Pos,
		Vel:          brief.Vel,
		Covariance:   brief.Covariance,
		Confidence:   brief.Confidence,
		SolutionType: brief.SolutionType,
		ThreatLevel:  brief.ThreatLevel,
		// Omit optional fields: EstimatedMass, RCS, IntentConfidence
	}

	return s.Serialize(compact)
}

// GetSerializedSize returns the size of the serialized data without actually serializing
func (s *FCHSerializer) GetSerializedSize(brief *FireControlBrief) (int, error) {
	data, err := s.Serialize(brief)
	if err != nil {
		return 0, err
	}
	return len(data), nil
}

// FCHBuilder helps construct FireControlBrief from propagator state
type FCHBuilder struct {
	brief *FireControlBrief
}

// NewFCHBuilder creates a new FCH builder
func NewFCHBuilder(targetID int, solutionType string) *FCHBuilder {
	return &FCHBuilder{
		brief: NewFireControlBrief(targetID, solutionType),
	}
}

// WithStateVector sets the 6-DOF state vector
func (b *FCHBuilder) WithStateVector(pos, vel [3]float64) *FCHBuilder {
	b.brief.SetStateVector(pos, vel)
	return b
}

// WithCovariance sets the covariance matrix (6x6)
func (b *FCHBuilder) WithCovariance(cov6x6 [6][6]float64) *FCHBuilder {
	b.brief.SetCovarianceFromMatrix(cov6x6)
	return b
}

// WithConfidence sets the confidence score
func (b *FCHBuilder) WithConfidence(confidence float64) *FCHBuilder {
	b.brief.Confidence = confidence
	return b
}

// WithThreatLevel sets the threat level
func (b *FCHBuilder) WithThreatLevel(level int) *FCHBuilder {
	b.brief.ThreatLevel = level
	return b
}

// WithExtendedData sets optional extended data
func (b *FCHBuilder) WithExtendedData(mass, rcs, intentConf float64) *FCHBuilder {
	b.brief.EstimatedMass = mass
	b.brief.RCS = rcs
	b.brief.IntentConfidence = intentConf
	return b
}

// Build returns the constructed FireControlBrief
func (b *FCHBuilder) Build() *FireControlBrief {
	return b.brief
}

// BuildAndValidate builds and validates the brief
func (b *FCHBuilder) BuildAndValidate() (*FireControlBrief, error) {
	if err := b.brief.Validate(); err != nil {
		return nil, err
	}
	return b.brief, nil
}

// PrepareFireControlHandover creates FCH from HPOP propagator state
// This is the main entry point for converting propagator output to FCH
func PrepareFireControlHandover(
	targetID int,
	position, velocity [3]float64,
	covariance6x6 [6][6]float64,
	confidence float64,
	solutionType string,
	threatLevel int,
) (*FireControlBrief, error) {

	builder := NewFCHBuilder(targetID, solutionType).
		WithStateVector(position, velocity).
		WithCovariance(covariance6x6).
		WithConfidence(confidence).
		WithThreatLevel(threatLevel)

	brief, err := builder.BuildAndValidate()
	if err != nil {
		return nil, fmt.Errorf("failed to prepare fire control handover: %w", err)
	}

	return brief, nil
}

// FCHBatchSerializer handles batch serialization of multiple FCH packets
type FCHBatchSerializer struct {
	serializer *FCHSerializer
}

// NewFCHBatchSerializer creates a batch serializer
func NewFCHBatchSerializer() *FCHBatchSerializer {
	return &FCHBatchSerializer{
		serializer: NewFCHSerializer(),
	}
}

// SerializeBatch serializes multiple FCH packets
func (b *FCHBatchSerializer) SerializeBatch(briefs []*FireControlBrief) ([][]byte, error) {
	var packets [][]byte
	var errors []error

	for i, brief := range briefs {
		data, err := b.serializer.Serialize(brief)
		if err != nil {
			errors = append(errors, fmt.Errorf("brief %d: %w", i, err))
			continue
		}
		packets = append(packets, data)
	}

	if len(errors) > 0 {
		return packets, fmt.Errorf("batch serialization had %d errors", len(errors))
	}

	return packets, nil
}

// FCHMetadata contains transmission metadata
type FCHMetadata struct {
	SerializedSize    int
	TransmissionTime  time.Time
	Priority          int
	DestinationHash   string
	EncryptionEnabled bool
}

// ExtractFCHMetadata extracts metadata from serialized FCH
func ExtractFCHMetadata(data []byte) (*FCHMetadata, error) {
	brief, err := NewFCHSerializer().Deserialize(data)
	if err != nil {
		return nil, err
	}

	return &FCHMetadata{
		SerializedSize:    len(data),
		TransmissionTime:  time.Now(),
		Priority:          brief.GetPriorityInt(),
		EncryptionEnabled: true, // Default assumption
	}, nil
}

// CompressCovariance reduces covariance precision to save space
// Uses 16-bit floats (half precision) for covariance values
func CompressCovariance(covariance []float64) []uint16 {
	compressed := make([]uint16, len(covariance))
	for i, val := range covariance {
		// Simple compression: scale to uint16 range
		// In production, use proper float16 encoding
		scaled := (val + 1e6) / (2e6) * 65535 // Normalize to 0-65535
		if scaled < 0 {
			scaled = 0
		}
		if scaled > 65535 {
			scaled = 65535
		}
		compressed[i] = uint16(scaled)
	}
	return compressed
}

// DecompressCovariance restores covariance from compressed format
func DecompressCovariance(compressed []uint16) []float64 {
	covariance := make([]float64, len(compressed))
	for i, val := range compressed {
		// Reverse the compression
		covariance[i] = (float64(val) / 65535 * 2e6) - 1e6
	}
	return covariance
}
