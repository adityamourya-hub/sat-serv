// pkg/telescope/observation_report.go
// Sovereign ObservationReport Implementation
//
// Go-native implementation of the ObservationReport protobuf schema.
// Provides zero-copy serialization for Reticulum Network Stack (100Hz edge path).
// Digitally signed via Ed25519 for chain of custody.

package telescope

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"ssis/pkg/physics"
)

// ObservationReport represents the sovereign payload for tactical SDA
// Protobuf message: ssis.v1.ObservationReport
type ObservationReport struct {
	// === Target Identification ===
	TargetID    string // NORAD ID or internal designation
	TimestampNs uint64 // Unix timestamp in nanoseconds
	StationID   string // Reporting station

	// === Kinetic Truth ===
	ECIPos  [3]float64 // ECI position [km]
	ECIVel  [3]float64 // ECI velocity [km/s]
	EpochJD float64    // Julian Date

	// === Hamiltonian Sovereign Handshake ===
	HamiltonianInvariant  float64 // Mean H over window
	HamiltonianVariance   float64 // Variance σ_H
	HamiltonianHash       []byte  // SHA-256 of invariant
	HamiltonianConfidence float64 // 0-1 confidence

	// === Optical Signature ===
	VisualMagnitude    float32 // Apparent magnitude
	SpectralClass      string  // e.g., "HGV-Type-D"
	MFBDsilhouetteHash []byte  // Structural hash from GPU reconstruction

	// === Intent Logic ===
	ConflictMetric float32 // 0-1: threat level
	ActivePolicy   string  // e.g., "Maneuver-to-Goal-Alpha"
	ThreatLevel    int32   // 0-4: UNSPECIFIED, LOW, MEDIUM, HIGH, CRITICAL

	// === Execution Tier ===
	ExecutionTier  int32   // 0-4: TIER_UNSPECIFIED, T1, T2, T3, T4
	TierConfidence float32 // Confidence in tier selection

	// === Spectroscopy (from SLiPy handoff) ===
	Spectroscopy *SpectroscopyData

	// === Cryptographic Signature ===
	ED25519Signature []byte
	SignerPublicKey  []byte
}

// SpectroscopyData from SLiPy integration (DMD-triggered handoff)
type SpectroscopyData struct {
	IntegrationStartNs uint64
	IntegrationEndNs   uint64
	SpectralBins       []float32
	WavelengthMin      float32 // nm
	WavelengthMax      float32 // nm
	MaterialID         string
	MaterialConfidence float32
	SWIRIntensity      float32
	PlasmaTemperature  float32 // K
	PlasmaDetected     bool
}

// OpticalFeatures contains MFBD/e3nn morphological characterization
type OpticalFeatures struct {
	EstimatedRCS        float32
	SymmetryScore       float32
	AspectRatio         float32
	MaterialClass       string
	RotationPeriod      float32 // Seconds
	TumbleRate          float32 // deg/s
	IsStabilized        bool
	ReflectanceSpectrum []float32
	SpectralResolution  float32 // nm
}

// ReportSerializer handles protobuf encoding/decoding
//
// Zero-copy serialization for 100Hz edge path
// Target: < 500 bytes, < 1ms latency
type ReportSerializer struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
	nodeID     string
}

// NewReportSerializer creates serializer with fresh Ed25519 keys
func NewReportSerializer(nodeID string) (*ReportSerializer, error) {
	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate Ed25519 keys: %w", err)
	}

	return &ReportSerializer{
		privateKey: privateKey,
		publicKey:  privateKey.Public().(ed25519.PublicKey),
		nodeID:     nodeID,
	}, nil
}

// Serialize encodes report to protobuf bytes
//
// Manual serialization for zero-copy performance
// (In production: Use protoc-generated Go code)
func (rs *ReportSerializer) Serialize(report *ObservationReport) ([]byte, error) {
	// Simplified manual serialization
	// In production: Use proper protobuf marshaling
	buf := make([]byte, 0, 512)

	// Target ID length + string
	buf = append(buf, byte(len(report.TargetID)))
	buf = append(buf, []byte(report.TargetID)...)

	// Timestamp (8 bytes, little-endian)
	timestampBytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(timestampBytes, report.TimestampNs)
	buf = append(buf, timestampBytes...)

	// Station ID
	buf = append(buf, byte(len(report.StationID)))
	buf = append(buf, []byte(report.StationID)...)

	// ECI position (3 x 8 bytes)
	for _, v := range report.ECIPos {
		posBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(posBytes, uint64(v*1e6)) // μm precision
		buf = append(buf, posBytes...)
	}

	// ECI velocity (3 x 8 bytes)
	for _, v := range report.ECIVel {
		velBytes := make([]byte, 8)
		binary.LittleEndian.PutUint64(velBytes, uint64(v*1e6))
		buf = append(buf, velBytes...)
	}

	// Hamiltonian hash (32 bytes)
	if len(report.HamiltonianHash) == 32 {
		buf = append(buf, report.HamiltonianHash...)
	}

	// Threat level (1 byte)
	buf = append(buf, byte(report.ThreatLevel))

	// Conflict metric (4 bytes)
	conflictBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(conflictBytes, uint32(report.ConflictMetric*1e6))
	buf = append(buf, conflictBytes...)

	return buf, nil
}

// SignAndSerialize creates signed ObservationReport
//
// Returns: serialized bytes + Ed25519 signature
func (rs *ReportSerializer) SignAndSerialize(report *ObservationReport) ([]byte, []byte, error) {
	// Serialize
	data, err := rs.Serialize(report)
	if err != nil {
		return nil, nil, fmt.Errorf("serialization failed: %w", err)
	}

	// Sign
	signature := ed25519.Sign(rs.privateKey, data)

	// Populate report fields
	report.ED25519Signature = signature
	report.SignerPublicKey = []byte(rs.publicKey)

	return data, signature, nil
}

// VerifySignature checks Ed25519 signature of report
func (rs *ReportSerializer) VerifySignature(data, signature, publicKey []byte) bool {
	if len(publicKey) != ed25519.PublicKeySize {
		return false
	}
	return ed25519.Verify(publicKey, data, signature)
}

// ComputeHamiltonianHash computes a tolerance-binned SHA-256 of Hamiltonian samples.
// Replaces the hardware-dependent float-string anti-pattern with a
// reproducible hash that is robust to O(1e-9) numerical noise.
// If σ_H / |H_avg| >= 1e-7, the caller should invalidate the certificate.
func ComputeHamiltonianHash(hamiltonianSamples []float64) ([]byte, float64, float64) {
	if len(hamiltonianSamples) == 0 {
		return nil, 0, 0
	}

	var sum float64
	for _, h := range hamiltonianSamples {
		sum += h
	}
	mean := sum / float64(len(hamiltonianSamples))

	var variance float64
	for _, h := range hamiltonianSamples {
		diff := h - mean
		variance += diff * diff
	}
	variance /= float64(len(hamiltonianSamples))

	h := sha256.New()
	// Tolerance-binned hash: quantize H_avg to 1e-9 resolution
	// so that compiler/architecture-dependent rounding does not alter the digest.
	bin := int64(math.Floor(mean / 1e-9))
	h.Write([]byte(fmt.Sprintf("H_%d", bin)))
	for i, samp := range hamiltonianSamples {
		binS := int64(math.Floor(samp / 1e-9))
		h.Write([]byte(fmt.Sprintf("|%d:%d", i, binS)))
	}
	hash := h.Sum(nil)

	return hash, mean, variance
}

// ValidateHamiltonian checks if Hamiltonian meets sovereign criteria
//
// Returns: isValid, confidence
// If σ_H / |H_avg| >= 1e-7, observation is physically inconsistent (spoof)
func ValidateHamiltonian(mean, variance float64) (bool, float64) {
	if mean == 0 {
		return false, 0.0
	}

	ratio := variance / math.Abs(mean)

	// Threshold: 1e-7
	if ratio >= 1e-7 {
		return false, 0.0 // Spoof detected
	}

	// Confidence: 1.0 at ratio=0, 0.0 at ratio=1e-7
	confidence := 1.0 - (ratio / 1e-7)
	if confidence < 0 {
		confidence = 0
	}

	return true, confidence
}

// SpectroscopyHandoff manages ZMQ SUB to SLiPy controller
//
// "Hot" handoff when DMD Tripwire detects coherent plasma mode
// Telescope stays in ClosedLoopTrack while spectrometer integrates 2-second window
type SpectroscopyHandoff struct {
	zmqSocket        *ZMQSocket
	isIntegrating    bool
	integrationStart int64
}

// ZMQSocket represents ZMQ connection (placeholder)
type ZMQSocket struct {
	endpoint string
}

// TriggerHandoff initiates spectroscopy integration when DMD fires
//
// Logic:
//  1. DMD detects coherent plasma mode (maneuver/bow shock)
//  2. Send ZMQ SUB trigger to SLiPy spectroscopy controller
//  3. Telescope stays in ClosedLoopTrack
//  4. Wait 2-second integration window
//  5. Receive spectral data, fuse into ObservationReport via e3nn
func (sh *SpectroscopyHandoff) TriggerHandoff(targetID string, dmdConfidence float64) error {
	if sh.isIntegrating {
		return fmt.Errorf("spectroscopy integration already in progress")
	}

	// Mark integration start
	sh.isIntegrating = true
	sh.integrationStart = time.Now().UnixNano()

	// Send ZMQ trigger to SLiPy (placeholder)
	trigger := fmt.Sprintf("SPECTROSCOPY_TRIGGER:%s:%.3f", targetID, dmdConfidence)
	_ = trigger

	// In production: Actual ZMQ SUB publish

	fmt.Printf("[SPECTROSCOPY] Handoff triggered for %s (DMD confidence: %.2f)\n",
		targetID, dmdConfidence)

	return nil
}

// ReceiveData receives spectral data from SLiPy after integration
//
// Returns spectroscopy data for fusion into ObservationReport
func (sh *SpectroscopyHandoff) ReceiveData() (*SpectroscopyData, error) {
	if !sh.isIntegrating {
		return nil, fmt.Errorf("no active integration")
	}

	// Check if 2-second window complete
	elapsed := time.Now().UnixNano() - sh.integrationStart
	if elapsed < 2e9 { // 2 seconds in nanoseconds
		return nil, fmt.Errorf("integration in progress: %.2f/2.0s",
			float64(elapsed)/1e9)
	}

	// In production: Receive actual spectral data from SLiPy via ZMQ

	// Reset state
	sh.isIntegrating = false

	// Placeholder: Simulated spectral data
	return &SpectroscopyData{
		IntegrationStartNs: uint64(sh.integrationStart),
		IntegrationEndNs:   uint64(time.Now().UnixNano()),
		SpectralBins:       make([]float32, 256), // 256 bins
		WavelengthMin:      400.0,                // 400 nm
		WavelengthMax:      1000.0,               // 1000 nm
		MaterialID:         "HGV-Plasma-Sheath",
		MaterialConfidence: 0.87,
		SWIRIntensity:      150.0,
		PlasmaTemperature:  3500.0, // K
		PlasmaDetected:     true,
	}, nil
}

// FusionResult contains fused observation data
type FusionResult struct {
	Report          *ObservationReport
	Spectroscopy    *SpectroscopyData
	FusedConfidence float32
}

// JulianDateFromTime converts Go time.Time to Julian Date
// Simplified JD calculation
func JulianDateFromTime(t time.Time) float64 {
	// Julian Date from Unix timestamp
	// JD = 2440587.5 + unix_seconds / 86400
	return 2440587.5 + float64(t.Unix())/86400.0
}

// =============================================================================
// Sovereign Handshake Protocol
// =============================================================================

// SovereignHandshake represents the physical proof of work between nodes
//
// Every ObservationReport must include Hamiltonian Invariant Hash.
// If σ_H / H_avg >= 10^-7, certificate self-invalidates (spoof detection).
type SovereignHandshake struct {
	// Report hash for integrity verification
	ReportHash []byte

	// Hamiltonian Invariant for physical consistency (from physics package)
	HamiltonianInvariant *physics.HamiltonianInvariant

	// Timestamp for replay protection
	Timestamp int64

	// Node identity
	NodeID string

	// Spoof detection threshold
	SpoofThreshold float64 // Default: 1e-7
}

// NewSovereignHandshake creates a new handshake for report validation
func NewSovereignHandshake(nodeID string, report *ObservationReport) *SovereignHandshake {
	// Compute report hash
	h := sha256.New()
	h.Write([]byte(report.TargetID))
	h.Write([]byte(fmt.Sprintf("%.15e", report.EpochJD)))
	h.Write([]byte(fmt.Sprintf("%.6f,%.6f,%.6f", report.ECIPos[0], report.ECIPos[1], report.ECIPos[2])))

	return &SovereignHandshake{
		ReportHash:     h.Sum(nil),
		Timestamp:      int64(time.Now().Unix()),
		NodeID:         nodeID,
		SpoofThreshold: 1e-7,
	}
}

// Validate performs sovereign validation of the report
//
// Returns: isValid, confidence (0-1)
// Spoof detection - if σ_H / H_avg >= 10^-7, report is invalid
func (sh *SovereignHandshake) Validate(report *ObservationReport) (bool, float64) {
	// 1. Check report hash integrity
	expectedHash := sha256.New()
	expectedHash.Write([]byte(report.TargetID))
	expectedHash.Write([]byte(fmt.Sprintf("%.15e", report.EpochJD)))
	expectedHash.Write([]byte(fmt.Sprintf("%.6f,%.6f,%.6f", report.ECIPos[0], report.ECIPos[1], report.ECIPos[2])))

	if !equalBytes(sh.ReportHash, expectedHash.Sum(nil)) {
		return false, 0.0 // Report tampered
	}

	// 2. Check Hamiltonian invariant if available
	if len(report.HamiltonianHash) > 0 {
		// Validate Hamiltonian consistency
		isValid, confidence := ValidateHamiltonian(report.HamiltonianInvariant, report.HamiltonianVariance)

		// Spoof detection: σ_H / H_avg >= 10^-7
		if report.HamiltonianInvariant != 0 {
			ratio := report.HamiltonianVariance / math.Abs(report.HamiltonianInvariant)
			if ratio >= sh.SpoofThreshold {
				return false, 0.0 // Spoof detected - non-physical ephemeris
			}
		}

		return isValid, confidence
	}

	// No Hamiltonian data - lower confidence
	return true, 0.5
}

// equalBytes compares two byte slices
func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// FuseSpectroscopy combines optical tracking with spectral data
//
// Uses e3nn material ID to fuse MFBD reconstruction with spectroscopy
func FuseSpectroscopy(report *ObservationReport, spec *SpectroscopyData) *FusionResult {
	// Update report with spectroscopy data
	report.Spectroscopy = spec

	// Update spectral class based on material ID
	if spec.PlasmaDetected {
		report.SpectralClass = "HGV-Plasma-" + spec.MaterialID
		report.ThreatLevel = 4 // CRITICAL
	}

	// Compute fused confidence
	fusedConfidence := float32(report.HamiltonianConfidence) * spec.MaterialConfidence

	return &FusionResult{
		Report:          report,
		Spectroscopy:    spec,
		FusedConfidence: float32(fusedConfidence),
	}
}

// ObservationReportBuilder constructs reports from telescope data
type ObservationReportBuilder struct {
	serializer        *ReportSerializer
	stationID         string
	hamiltonianWindow []float64
}

// NewObservationReportBuilder creates builder for station
func NewObservationReportBuilder(nodeID, stationID string) (*ObservationReportBuilder, error) {
	serializer, err := NewReportSerializer(nodeID)
	if err != nil {
		return nil, err
	}

	return &ObservationReportBuilder{
		serializer:        serializer,
		stationID:         stationID,
		hamiltonianWindow: make([]float64, 0, 100),
	}, nil
}

// BuildReport creates ObservationReport from tracking data
//
// Integrates Hamiltonian samples for physical proof of work
func (b *ObservationReportBuilder) BuildReport(
	targetID string,
	state *physics.State,
	visualMag float32,
	conflictMetric float32,
	threatLevel int32,
) (*ObservationReport, error) {
	// Add Hamiltonian sample
	b.hamiltonianWindow = append(b.hamiltonianWindow, state.E_hal)
	if len(b.hamiltonianWindow) > 100 {
		b.hamiltonianWindow = b.hamiltonianWindow[1:]
	}

	// Compute Hamiltonian hash if we have enough samples
	var hash []byte
	var mean, variance float64
	var confidence float64

	if len(b.hamiltonianWindow) >= 10 {
		hash, mean, variance = ComputeHamiltonianHash(b.hamiltonianWindow)
		_, confidence = ValidateHamiltonian(mean, variance)
	}

	// Build epoch JD from timestamp
	epochJD := JulianDateFromTime(time.Now())

	report := &ObservationReport{
		TargetID:              targetID,
		TimestampNs:           uint64(time.Now().UnixNano()),
		StationID:             b.stationID,
		ECIPos:                [3]float64{state.Pos[0], state.Pos[1], state.Pos[2]},
		ECIVel:                [3]float64{state.Vel[0], state.Vel[1], state.Vel[2]},
		EpochJD:               epochJD,
		HamiltonianInvariant:  mean,
		HamiltonianVariance:   variance,
		HamiltonianHash:       hash,
		HamiltonianConfidence: confidence,
		VisualMagnitude:       visualMag,
		ConflictMetric:        conflictMetric,
		ThreatLevel:           threatLevel,
	}

	return report, nil
}

// SignReport creates signed serialized report
func (b *ObservationReportBuilder) SignReport(report *ObservationReport) ([]byte, []byte, error) {
	return b.serializer.SignAndSerialize(report)
}
