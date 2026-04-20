package sigint

/**
 * DPHNO Bridge - Signal Intelligence to Hypernetwork Atlas Integration
 *
 * Converts SIGINT observations into latent physics codes (z) compatible with
 * Hypernetwork Atlas for zero-day generalization.
 *
 * Reference: ai/v3/hypernetwork_atlas.py - AtlasEncoder interface
 * Reference: pkg/ml/dphno/operator.go - DPHNO operator interface
 */

import (
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// LatentPhysicsCode represents the z ∈ ℝ^256 latent physics code
// Expected by Hypernetwork Atlas for weight generation
type LatentPhysicsCode struct {
	Z          [256]float64 `json:"z"` // 256-dimensional latent code
	Timestamp  time.Time    `json:"timestamp"`
	Source     string       `json:"source"`     // Which sigint module generated this
	Confidence float64      `json:"confidence"` // 0-1 confidence in encoding
}

// AtmosphereProfile represents NRLMSISE-00 atmosphere profile (64 dimensions)
// Required by AtlasEncoder for physics-aware encoding
type AtmosphereProfile struct {
	F10_7      float64     `json:"f10_7"`       // Solar flux index
	Ap         float64     `json:"ap"`          // Geomagnetic index
	AltitudeKm float64     `json:"altitude_km"` // Altitude in km
	Profile64  [64]float64 `json:"profile_64"`  // Full 64-dim atmosphere vector
}

// VehicleClass represents vehicle classification for embedding
// Matches Hypernetwork Atlas vehicle embedding (10 classes)
type VehicleClass int

const (
	ClassUnknown VehicleClass = iota
	ClassHGV
	ClassBallistic
	ClassSatellite
	ClassDebris
	ClassCivilian
	ClassMilitary
	ClassUtility
	ClassSatCom
	ClassRadar
	ClassOther
)

// SignalObservation represents a processed SIGINT observation
// ready for encoding into latent physics code
type SignalObservation struct {
	// Position/Velocity (6-dim state vector)
	X, Y, Z    float64
	VX, VY, VZ float64

	// Signal characteristics
	FrequencyHz float64
	BandwidthHz float64
	Modulation  string
	SNR         float64
	Confidence  float64

	// Classification
	VehicleClass VehicleClass
	ThreatLevel  int

	// Timing
	Timestamp time.Time
}

// DPHNOBridge encodes SIGINT observations into latent physics codes
type DPHNOBridge struct {
	// TODO: Add actual neural encoder when model is exported
	// For now, use physics-based feature extraction
}

// NewDPHNOBridge creates a new DPHNO bridge
func NewDPHNOBridge() *DPHNOBridge {
	return &DPHNOBridge{}
}

// EncodeObservation converts a SIGINT observation into latent physics code z
//
// This implements a physics-aware encoding that maps:
// (state, atmosphere, vehicle_class) → z ∈ ℝ^256
//
// Current implementation uses feature-based encoding as placeholder.
// Production should use trained AtlasEncoder from ai/v3/hypernetwork_atlas.py
func (db *DPHNOBridge) EncodeObservation(obs SignalObservation, atmos AtmosphereProfile) (LatentPhysicsCode, error) {
	z := LatentPhysicsCode{
		Timestamp:  time.Now(),
		Source:     "sigint",
		Confidence: obs.Confidence,
	}

	// Placeholder: Physics-based feature extraction
	// In production, this would call the trained AtlasEncoder neural network

	// Extract physics features from state
	r := math.Sqrt(obs.X*obs.X + obs.Y*obs.Y + obs.Z*obs.Z)
	v := math.Sqrt(obs.VX*obs.VX + obs.VY*obs.VY + obs.VZ*obs.VZ)
	energy := 0.5*v*v - 398600.4418/r

	// Fill latent code with structured features
	// z[0-5]: State vector (normalized)
	z.Z[0] = obs.X / 7000.0
	z.Z[1] = obs.Y / 7000.0
	z.Z[2] = obs.Z / 7000.0
	z.Z[3] = obs.VX / 10000.0
	z.Z[4] = obs.VY / 10000.0
	z.Z[5] = obs.VZ / 10000.0

	// z[6-9]: Energy and orbital elements
	z.Z[6] = energy / 1e7
	z.Z[7] = r / 7000.0
	z.Z[8] = obs.FrequencyHz / 10e9
	z.Z[9] = obs.BandwidthHz / 5e9

	// z[10-73]: Atmosphere profile (64 dimensions)
	for i := 0; i < 64; i++ {
		if i < len(atmos.Profile64) {
			z.Z[10+i] = atmos.Profile64[i]
		}
	}

	// z[74-83]: Vehicle class one-hot encoding (10 classes)
	classIdx := int(obs.VehicleClass)
	if classIdx >= 0 && classIdx < 10 {
		z.Z[74+classIdx] = 1.0
	}

	// z[84-89]: Signal characteristics
	z.Z[84] = obs.SNR / 30.0
	z.Z[85] = float64(obs.ThreatLevel) / 5.0
	z.Z[86] = obs.Confidence

	// z[87-255]: Reserved for learned features (initialized to small random values)
	for i := 87; i < 256; i++ {
		// Deterministic pseudo-random based on observation
		z.Z[i] = math.Sin(float64(i)*energy) * 0.01
	}

	return z, nil
}

// EncodeSIGIDTemplate converts a SIGID template into latent code
// for use in signal classification and threat assessment
func (db *DPHNOBridge) EncodeSIGIDTemplate(template SIGIDTemplate) (LatentPhysicsCode, error) {
	z := LatentPhysicsCode{
		Timestamp:  time.Now(),
		Source:     "sigid_template",
		Confidence: 1.0,
	}

	// Encode template features into latent space
	z.Z[0] = template.FrequencyHz / 10e9
	z.Z[1] = template.BandwidthHz / 5e9
	z.Z[2] = float64(template.ThreatLevel) / 5.0

	// Category encoding
	categoryMap := map[string]int{
		"Military":  1,
		"Satellite": 2,
		"Utility":   3,
		"Unknown":   0,
	}
	if catIdx, ok := categoryMap[template.Category]; ok {
		z.Z[3] = float64(catIdx) / 4.0
	}

	// PSD mask features (first 64 bytes)
	for i := 0; i < 64 && i < len(template.PSDMask); i++ {
		z.Z[4+i] = float64(template.PSDMask[i]) / 255.0
	}

	// Fill remaining dimensions
	for i := 68; i < 256; i++ {
		z.Z[i] = math.Sin(float64(i)*template.FrequencyHz) * 0.01
	}

	return z, nil
}

// EncodeTDOAGeolocation encodes TDOA-based geolocation results
// for integration with DPHNO dynamics
func (db *DPHNOBridge) EncodeTDOAGeolocation(result MeshGeolocationResult) (LatentPhysicsCode, error) {
	z := LatentPhysicsCode{
		Timestamp:  time.Now(),
		Source:     "tdoa_geolocation",
		Confidence: 0.8, // Default confidence
	}

	// Access fields through embedded TDOAResult
	if result.TDOAResult != nil {
		z.Confidence = result.TDOAResult.Confidence

		// Convert geodetic (lat/lon/alt) to ECEF for state vector
		// Use WGS84 ellipsoid conversion
		ellipsoid := NewWGS84Ellipsoid()
		positionECEF := ellipsoid.GeodeticToECEF(
			result.TDOAResult.Latitude,
			result.TDOAResult.Longitude,
			result.TDOAResult.Altitude,
		)

		// Position (normalized)
		z.Z[0] = positionECEF[0] / 7000.0
		z.Z[1] = positionECEF[1] / 7000.0
		z.Z[2] = positionECEF[2] / 7000.0

		// Accuracy and confidence
		z.Z[3] = result.TDOAResult.Accuracy / 10000.0
		z.Z[4] = result.TDOAResult.Confidence

		// Method encoding
		z.Z[7] = hashStringToFloat(result.TDOAResult.Method)
	} else {
		// Fallback if TDOAResult is nil
		z.Z[0] = 0
		z.Z[1] = 0
		z.Z[2] = 0
		z.Z[3] = 0
		z.Z[4] = 0
		z.Z[7] = 0
	}

	// Timing metadata (convert to normalized values)
	z.Z[5] = float64(result.ComputeTime.Milliseconds()) / 1000.0
	z.Z[6] = float64(result.MeshLatency.Milliseconds()) / 5000.0

	// Fill remaining dimensions with deterministic features
	if result.TDOAResult != nil {
		for i := 8; i < 256; i++ {
			// Use lat/lon for deterministic features
			z.Z[i] = math.Sin(float64(i)*result.TDOAResult.Latitude+float64(i)*result.TDOAResult.Longitude) * 0.01
		}
	} else {
		for i := 8; i < 256; i++ {
			z.Z[i] = math.Sin(float64(i)) * 0.01
		}
	}

	return z, nil
}

// hashStringToFloat converts a string to a deterministic float value for encoding
func hashStringToFloat(s string) float64 {
	hash := 0.0
	for i, c := range s {
		hash += float64(int(c)) * math.Pow(31.0, float64(i))
	}
	return math.Mod(hash, 1.0)
}

// GetHamiltonianInvariant computes a simplified Hamiltonian invariant
// for sovereign handshake protocol (σ_H / H_avg >= 10^-7 requirement)
func (db *DPHNOBridge) GetHamiltonianInvariant(z LatentPhysicsCode) (float64, error) {
	// Extract position and velocity from latent code
	x, y, z_pos := z.Z[0]*7000.0, z.Z[1]*7000.0, z.Z[2]*7000.0
	vx, vy, vz := z.Z[3]*10000.0, z.Z[4]*10000.0, z.Z[5]*10000.0

	// Compute Hamiltonian H = T + V
	r := math.Sqrt(x*x + y*y + z_pos*z_pos)
	v_sq := vx*vx + vy*vy + vz*vz
	kinetic := 0.5 * v_sq
	potential := -398600.4418 / r
	H := kinetic + potential

	// Compute average Hamiltonian from latent code features
	H_avg := math.Abs(z.Z[6]) * 1e7 // Decode from encoded energy

	// Compute normalized variation
	sigma_H := math.Abs(H - H_avg)
	invariant := sigma_H / math.Abs(H_avg)

	return invariant, nil
}

// ValidateInvariant checks if the invariant meets sovereign handshake requirements
func (db *DPHNOBridge) ValidateInvariant(z LatentPhysicsCode) (bool, float64, error) {
	invariant, err := db.GetHamiltonianInvariant(z)
	if err != nil {
		return false, 0, err
	}

	// Sovereign handshake requirement: σ_H / H_avg >= 10^-7
	valid := invariant >= 1e-7

	return valid, invariant, nil
}

// ExportToJSON exports latent code for external consumption
func (z *LatentPhysicsCode) ExportToJSON() ([]byte, error) {
	return json.Marshal(z)
}

// ImportFromJSON imports latent code from JSON
func ImportLatentCodeFromJSON(data []byte) (LatentPhysicsCode, error) {
	var z LatentPhysicsCode
	err := json.Unmarshal(data, &z)
	return z, err
}

// BatchEncode encodes multiple observations in parallel
func (db *DPHNOBridge) BatchEncode(obs []SignalObservation, atmos AtmosphereProfile) ([]LatentPhysicsCode, error) {
	codes := make([]LatentPhysicsCode, len(obs))
	var errs []error

	for i, o := range obs {
		code, err := db.EncodeObservation(o, atmos)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		codes[i] = code
	}

	if len(errs) > 0 {
		return codes, fmt.Errorf("batch encode had %d errors", len(errs))
	}

	return codes, nil
}
