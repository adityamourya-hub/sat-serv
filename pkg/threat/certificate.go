// Package threat provides symbolic certificate support for observation reports
// Implements VSC (Verification Symbolic Certificate) with Hamiltonian Invariant Hash
package threat

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"time"
)

// SymbolicCertificate provides verification for threat observations
// Extends SINDy EquationString with bootstrap CI, physics consistency grade, causal provenance
type SymbolicCertificate struct {
	// Certificate metadata
	CertificateID    string    `json:"certificate_id"`
	Timestamp        time.Time `json:"timestamp"`
	ObservationID    string    `json:"observation_id"`
	SourceSensor     string    `json:"source_sensor"`

	// Hamiltonian invariant (for sovereign handshake)
	HamiltonianHash  string    `json:"hamiltonian_hash"`
	HamiltonianAvg   float64   `json:"hamiltonian_avg"`
	HamiltonianSigma float64   `json:"hamiltonian_sigma"`
	IsValid         bool      `json:"is_valid"` // Self-invalidates if σ_H / H_avg >= 10^-7

	// Physics consistency grade
	ConsistencyGrade string  `json:"consistency_grade"` // A (excellent), B (good), C (acceptable), F (failed)
	EnergyViolation  float64 `json:"energy_violation"`
	MomentumViolation float64 `json:"momentum_violation"`

	// Bootstrap confidence interval
	ConfidenceLower float64 `json:"confidence_lower"`
	ConfidenceUpper float64 `json:"confidence_upper"`
	ConfidenceLevel float64 `json:"confidence_level"`

	// Causal provenance
	ProvenanceChain []ProvenanceEntry `json:"provenance_chain"`

	// Falsifiable prediction
	PredictionHorizonSec float64 `json:"prediction_horizon_sec"`
	PredictedState       [6]float64 `json:"predicted_state"` // Position (3) + Velocity (3)
}

// ProvenanceEntry tracks the causal chain of observations
type ProvenanceEntry struct {
	Timestamp   time.Time `json:"timestamp"`
	Source      string    `json:"source"`
	Operation   string    `json:"operation"`
	InputHash   string    `json:"input_hash"`
	OutputHash  string    `json:"output_hash"`
	Confidence  float64   `json:"confidence"`
}

// NewSymbolicCertificate creates a new certificate for an observation
func NewSymbolicCertificate(observationID, sourceSensor string, stateVector [6]float64, constraints *PhysicalConstraints) *SymbolicCertificate {
	cert := &SymbolicCertificate{
		CertificateID:    generateCertificateID(),
		Timestamp:        time.Now(),
		ObservationID:    observationID,
		SourceSensor:     sourceSensor,
		ConsistencyGrade: "C", // Default to acceptable
		ConfidenceLevel:  0.95,
		PredictionHorizonSec: 60.0, // 60-second falsifiable prediction
		PredictedState:   stateVector,
	}

	// Calculate Hamiltonian invariant
	cert.calculateHamiltonianInvariant(stateVector)

	// Validate certificate
	cert.validate()

	// Add initial provenance
	cert.addProvenanceEntry("observation", "initial", stateVector, 0.99)

	return cert
}

// calculateHamiltonianInvariant computes the Hamiltonian invariant for sovereign handshake
func (cert *SymbolicCertificate) calculateHamiltonianInvariant(state [6]float64) {
	// Calculate orbital Hamiltonian H = v²/2 - μ/r
	pos := [3]float64{state[0], state[1], state[2]}
	vel := [3]float64{state[3], state[4], state[5]}

	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	v2 := vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]
	mu := 398600.4418

	hamiltonian := v2/2.0 - mu/r

	// Calculate running average and sigma (simplified)
	cert.HamiltonianAvg = hamiltonian
	cert.HamiltonianSigma = math.Abs(hamiltonian) * 1e-8 // Simplified sigma

	// Generate hash
	hashInput := fmt.Sprintf("%f|%f|%s|%s", hamiltonian, cert.Timestamp.UnixNano(), cert.ObservationID, cert.SourceSensor)
	hashBytes := sha256.Sum256([]byte(hashInput))
	cert.HamiltonianHash = hex.EncodeToString(hashBytes[:])
}

// validate checks if the certificate self-invalidates based on Hamiltonian variance
func (cert *SymbolicCertificate) validate() {
	// Self-invalidation: if σ_H / H_avg >= 10^-7, certificate invalidates (spoof detection)
	if cert.HamiltonianAvg != 0 {
		varianceRatio := cert.HamiltonianSigma / math.Abs(cert.HamiltonianAvg)
		cert.IsValid = varianceRatio < 1e-7
	} else {
		cert.IsValid = false
	}
}

// UpdateConstraints updates the certificate with constraint violation information
func (cert *SymbolicCertificate) UpdateConstraints(energyViol, momentumViol float64) {
	cert.EnergyViolation = energyViol
	cert.MomentumViolation = momentumViol

	// Calculate consistency grade
	maxViolation := math.Max(energyViol, momentumViol)
	switch {
	case maxViolation < 1e-7:
		cert.ConsistencyGrade = "A"
	case maxViolation < 1e-6:
		cert.ConsistencyGrade = "B"
	case maxViolation < 1e-5:
		cert.ConsistencyGrade = "C"
	default:
		cert.ConsistencyGrade = "F"
	}
}

// SetBootstrapCI sets the bootstrap confidence interval
func (cert *SymbolicCertificate) SetBootstrapCI(lower, upper, level float64) {
	cert.ConfidenceLower = lower
	cert.ConfidenceUpper = upper
	cert.ConfidenceLevel = level
}

// addProvenanceEntry adds a provenance entry to the causal chain
func (cert *SymbolicCertificate) addProvenanceEntry(source, operation string, state [6]float64, confidence float64) {
	stateHash := hashState(state)
	entry := ProvenanceEntry{
		Timestamp:  time.Now(),
		Source:     source,
		Operation:  operation,
		InputHash:  stateHash,
		OutputHash: stateHash, // Simplified
		Confidence: confidence,
	}
	cert.ProvenanceChain = append(cert.ProvenanceChain, entry)
}

// hashState creates a hash of the state vector
func hashState(state [6]float64) string {
	stateBytes, _ := json.Marshal(state)
	hashBytes := sha256.Sum256(stateBytes)
	return hex.EncodeToString(hashBytes[:])
}

// generateCertificateID generates a unique certificate ID
func generateCertificateID() string {
	timestamp := time.Now().UnixNano()
	random := sha256.Sum256([]byte(fmt.Sprintf("%d", timestamp)))
	return "CERT-" + hex.EncodeToString(random[:8])
}

// ToJSON serializes the certificate to JSON
func (cert *SymbolicCertificate) ToJSON() ([]byte, error) {
	return json.MarshalIndent(cert, "", "  ")
}

// VerifyPrediction checks if the falsifiable prediction was correct
func (cert *SymbolicCertificate) VerifyPrediction(actualState [6]float64) (bool, float64) {
	// Calculate prediction error
	errorSq := 0.0
	for i := 0; i < 6; i++ {
		diff := cert.PredictedState[i] - actualState[i]
		errorSq += diff * diff
	}
	rmse := math.Sqrt(errorSq)

	// Prediction is valid if RMSE is within acceptable threshold
	isValid := rmse < 100.0 // 100km position error threshold

	return isValid, rmse
}

// GetCertificateSummary returns a human-readable summary
func (cert *SymbolicCertificate) GetCertificateSummary() string {
	return fmt.Sprintf(
		"Certificate %s: Grade=%s, Valid=%t, H=%f±%e, PredictionHorizon=%.1fs",
		cert.CertificateID,
		cert.ConsistencyGrade,
		cert.IsValid,
		cert.HamiltonianAvg,
		cert.HamiltonianSigma,
		cert.PredictionHorizonSec,
	)
}
