package cryptography

/**
 * Sovereign Auditor - Tamper-evident attestation of SDA events
 *
 * Provides cryptographic proof of:
 * - Sensor frame integrity (Merkle commitment)
 * - Detection validity (signed attestation)
 * - Non-repudiation (Ed25519 signatures)
 *
 * Modes:
 * - Shadow Sovereign: Software-only (development/demos)
 * - Hardware Sovereign: AUM-1 PUF-based (production/kinetic)
 */

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/binary"
	"math"
	"time"
)

// Auditor provides tamper-evident attestation interface
type Auditor interface {
	// HashIngress builds Merkle root from raw sensor frames
	HashIngress(frames [][]byte) ([32]byte, error)

	// AttestManeuver generates signed proof of detection
	AttestManeuver(detection ManeuverDetection, merkleRoot [32]byte) (Attestation, error)

	// VerifyAttestation checks external attestation
	VerifyAttestation(att Attestation) bool

	// PublicKey returns auditor's verification key (32 bytes)
	PublicKey() [32]byte
}

// ManeuverDetection represents physics engine output
type ManeuverDetection struct {
	ObjectID    string
	Timestamp   time.Time
	ResidualRMS float64    // Mathematical proof (non-ballistic behavior)
	Confidence  float64    // Bayesian score (0-1)
	Threshold   float64    // Detection threshold used
	Position    [3]float64 // ECI km
	Velocity    [3]float64 // ECI km/s
}

// Attestation is the "Proof of Aggression" datagram
type Attestation struct {
	// Header
	Version   uint8    // Schema version (1)
	Timestamp int64    // Unix nanoseconds
	AuditorID [16]byte // Unique auditor identifier

	// Event
	ObjectID  string // NORAD ID
	EventType uint8  // 0x01 = MANEUVER

	// Proof
	MerkleRoot  [32]byte // Commitment to raw frames
	ResidualRMS float64  // Orbit residual proof
	Confidence  float64  // Detection confidence
	Threshold   float64  // Threshold used

	// Context
	Position [3]float64 // km
	Velocity [3]float64 // km/s

	// Signature (Ed25519)
	Signature [64]byte
}

// ShadowAuditor implements software-only auditor (development mode)
// WARNING: Not for production kinetic attestation - keys are extractable
type ShadowAuditor struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
	auditorID  [16]byte
}

// NewShadowAuditor creates software-based auditor
// Uses random seed (AUM-1 uses PUF-derived seed in production)
func NewShadowAuditor() (*ShadowAuditor, error) {
	// Generate keypair
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}

	// Derive auditor ID from public key
	var id [16]byte
	copy(id[:], priv.Public().(ed25519.PublicKey)[:16])

	return &ShadowAuditor{
		privateKey: priv,
		publicKey:  priv.Public().(ed25519.PublicKey),
		auditorID:  id,
	}, nil
}

// HashIngress builds Merkle tree from frames
func (s *ShadowAuditor) HashIngress(frames [][]byte) ([32]byte, error) {
	tree := NewTree()
	for _, f := range frames {
		tree.AddLeaf(f)
	}
	return tree.Root(), nil
}

// AttestManeuver creates signed attestation
func (s *ShadowAuditor) AttestManeuver(
	detection ManeuverDetection,
	merkleRoot [32]byte,
) (Attestation, error) {

	att := Attestation{
		Version:     1,
		Timestamp:   time.Now().UnixNano(),
		AuditorID:   s.auditorID,
		ObjectID:    detection.ObjectID,
		EventType:   0x01, // MANEUVER
		MerkleRoot:  merkleRoot,
		ResidualRMS: detection.ResidualRMS,
		Confidence:  detection.Confidence,
		Threshold:   detection.Threshold,
		Position:    detection.Position,
		Velocity:    detection.Velocity,
	}

	// Serialize for signing
	msg := serializeAttestation(att)

	// Sign
	sig := ed25519.Sign(s.privateKey, msg)
	copy(att.Signature[:], sig)

	return att, nil
}

// VerifyAttestation checks signature validity
func (s *ShadowAuditor) VerifyAttestation(att Attestation) bool {
	msg := serializeAttestation(att)
	sig := att.Signature[:]
	return ed25519.Verify(s.publicKey, msg, sig)
}

// PublicKey returns verifying key
func (s *ShadowAuditor) PublicKey() [32]byte {
	var pk [32]byte
	copy(pk[:], s.publicKey)
	return pk
}

// serializeAttestation creates deterministic encoding
// Excludes Signature field from serialization
func serializeAttestation(a Attestation) []byte {
	buf := make([]byte, 0, 200)

	buf = append(buf, a.Version)
	buf = binary.LittleEndian.AppendUint64(buf, uint64(a.Timestamp))
	buf = append(buf, a.AuditorID[:]...)
	buf = append(buf, []byte(a.ObjectID)...)
	buf = append(buf, a.EventType)
	buf = append(buf, a.MerkleRoot[:]...)
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.ResidualRMS))
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Confidence))
	buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Threshold))

	// Position and velocity
	for i := 0; i < 3; i++ {
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Position[i]))
		buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(a.Velocity[i]))
	}

	return buf
}
