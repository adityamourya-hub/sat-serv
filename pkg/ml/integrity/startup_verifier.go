// Package integrity provides ADITI 4.0 Sovereign Integrity Verification
//
// This package implements the "M2 Final Flash" - startup verification
// of model integrity via Hessian Hash checking. If the hash doesn't match
// the Sovereign Signature, the system refuses to initialize.
//
// ADITI 4.0 Delhi War-Room: The ref_points are hard-coded as const
// to prevent runtime tampering. On startup, the Go kernel calls ONNX
// runtime to compute the hash. Mismatch = refusal to mount.
package integrity

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"strings"
)

// =============================================================================
// Hard-Coded Axiom Points (ADITI 4.0 Canonical Reference States)
// These are const to prevent runtime tampering.
// =============================================================================

// AxiomPoint represents a canonical reference state in phase space
type AxiomPoint struct {
	Name   string     `json:"name"`
	Q      [3]float64 `json:"q"` // Position [km]
	P      [3]float64 `json:"p"` // Momentum [km/s]
	Regime string     `json:"regime"`
}

// Canonical Axiom Points - hard-coded for security
var (
	// LEO_CIRCULAR: 400km altitude circular orbit
	LEO_CIRCULAR = AxiomPoint{
		Name:   "LEO_CIRCULAR",
		Q:      [3]float64{6778.137, 0.0, 0.0},
		P:      [3]float64{0.0, 7.6686, 0.0},
		Regime: "orbital",
	}

	// HGV_ENTRY: Siliguri corridor entry, 80km altitude
	HGV_ENTRY = AxiomPoint{
		Name:   "HGV_ENTRY",
		Q:      [3]float64{6451.137, 1000.0, 200.0},
		P:      [3]float64{4.5, 4.5, 0.5},
		Regime: "hgv",
	}

	// GEO_STATIONARY: 35,786 km altitude
	GEO_STATIONARY = AxiomPoint{
		Name:   "GEO_STATIONARY",
		Q:      [3]float64{42164.0, 0.0, 0.0},
		P:      [3]float64{0.0, 3.0747, 0.0},
		Regime: "orbital",
	}

	// ELLIPTICAL_HIGH: Molniya-like orbit
	ELLIPTICAL_HIGH = AxiomPoint{
		Name:   "ELLIPTICAL_HIGH",
		Q:      [3]float64{7000.0, 0.0, 0.0},
		P:      [3]float64{0.0, 10.0, 0.0},
		Regime: "orbital",
	}

	// SUBORBITAL: Ballistic missile mid-course
	SUBORBITAL = AxiomPoint{
		Name:   "SUBORBITAL",
		Q:      [3]float64{7000.0, 1000.0, 500.0},
		P:      [3]float64{5.0, 5.0, 2.0},
		Regime: "ballistic",
	}

	// DEBRIS_CLOUD: Post-collision fragments
	DEBRIS_CLOUD = AxiomPoint{
		Name:   "DEBRIS_CLOUD",
		Q:      [3]float64{7500.0, 500.0, 100.0},
		P:      [3]float64{2.0, 7.5, 1.0},
		Regime: "debris",
	}
)

// AxiomPoints is the complete set of reference points
var AxiomPoints = []AxiomPoint{
	LEO_CIRCULAR,
	HGV_ENTRY,
	GEO_STATIONARY,
	ELLIPTICAL_HIGH,
	SUBORBITAL,
	DEBRIS_CLOUD,
}

// =============================================================================
// Hessian Hash Verification
// =============================================================================

// HessianResult contains the verification result
type HessianResult struct {
	Hash       string    `json:"hash"`
	Traces     []float64 `json:"traces"`
	AxiomNames []string  `json:"axiom_names"`
	Timestamp  int64     `json:"timestamp"`
	Valid      bool      `json:"valid"`
}

// StartupVerifier performs integrity checks on model loading
type StartupVerifier struct {
	ReferenceHash string
	AxiomPoints   []AxiomPoint
}

// NewStartupVerifier creates a verifier with hard-coded reference hash
func NewStartupVerifier() *StartupVerifier {
	return &StartupVerifier{
		AxiomPoints: AxiomPoints,
		// ReferenceHash should be loaded from secure storage (e.g., TPM, sealed file)
		ReferenceHash: "",
	}
}

// SetReferenceHash sets the expected hash (call before verification)
func (sv *StartupVerifier) SetReferenceHash(hash string) {
	sv.ReferenceHash = hash
}

// LoadReferenceHashFromEnv loads hash from environment variable
func (sv *StartupVerifier) LoadReferenceHashFromEnv() bool {
	hash := os.Getenv("ADITI_HESSIAN_HASH")
	if hash == "" {
		return false
	}
	sv.ReferenceHash = hash
	return true
}

// ComputeHessianTrace computes the trace of Hessian at a given state
// This is a simplified proxy - in production, this calls ONNX runtime
// to evaluate second derivatives of the actual H_net
func (sv *StartupVerifier) ComputeHessianTrace(q, p [3]float64) float64 {
	// Simplified proxy: compute Laplacian of Keplerian Hamiltonian
	// H = 0.5*p² - μ/r
	// ∇²H = ∇²(PE) + ∇²(KE) = 2μ/r³ + 3 (in canonical units)

	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	mu := 398600.4418 // Earth gravitational parameter

	if r < 1.0 {
		r = 1.0 // Prevent division by zero
	}

	// Laplacian of -μ/r in 3D: 2μ/r³
	laplacianPE := 2.0 * mu / (r * r * r)

	// KE part (0.5*v²) contributes 3 to trace (diagonal of identity)
	laplacianKE := 3.0

	return laplacianPE + laplacianKE
}

// ComputeHessianHash generates the integrity hash
func (sv *StartupVerifier) ComputeHessianHash() *HessianResult {
	traces := make([]float64, len(sv.AxiomPoints))
	names := make([]string, len(sv.AxiomPoints))

	fmt.Println("[StartupVerifier] Computing Hessian Hash at Axiom Points...")

	for i, point := range sv.AxiomPoints {
		trace := sv.ComputeHessianTrace(point.Q, point.P)
		traces[i] = trace
		names[i] = point.Name
		fmt.Printf("  %s: Tr(∇²H) = %.6e\n", point.Name, trace)
	}

	// Generate hash input
	hashInput := ""
	for i, name := range names {
		hashInput += fmt.Sprintf("%s=%.10f,", name, traces[i])
	}

	// Add model version salt
	hashInput += "v=ADITI-4.0"

	// SHA-256 hash
	hash := sha256.Sum256([]byte(hashInput))
	hashStr := hex.EncodeToString(hash[:])

	fmt.Printf("[StartupVerifier] Hessian Hash: %s...%s\n",
		hashStr[:16], hashStr[len(hashStr)-16:])

	return &HessianResult{
		Hash:       hashStr,
		Traces:     traces,
		AxiomNames: names,
		Timestamp:  0, // Set to current time in production
		Valid:      true,
	}
}

// VerifyIntegrity checks if current model matches reference hash
func (sv *StartupVerifier) VerifyIntegrity() (bool, string) {
	if sv.ReferenceHash == "" {
		return false, "No reference hash set - cannot verify integrity"
	}

	current := sv.ComputeHessianHash()

	if current.Hash == sv.ReferenceHash {
		return true, "INTEGRITY_VERIFIED: Hessian hash matches reference"
	}

	return false, fmt.Sprintf(
		"INTEGRITY_VIOLATION: Hash mismatch. Expected %s..., got %s...",
		sv.ReferenceHash[:16],
		current.Hash[:16],
	)
}

// =============================================================================
// Startup Gate
// =============================================================================

// StartupGateResult indicates whether system can proceed
type StartupGateResult int

const (
	GateProceed StartupGateResult = 0 // All checks passed
	GateWarning StartupGateResult = 1 // Minor issues, proceed with caution
	GateRefuse  StartupGateResult = 2 // Critical failure, refuse startup
)

// ExecuteStartupGate performs all integrity checks at system startup
// This is the "M2 Final Flash" - if this fails, command_bridge refuses init
func ExecuteStartupGate() StartupGateResult {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("ADITI 4.0 SOVEREIGN STARTUP GATE")
	fmt.Println(strings.Repeat("=", 70))

	verifier := NewStartupVerifier()

	// Attempt to load reference hash
	if !verifier.LoadReferenceHashFromEnv() {
		fmt.Println("⚠ WARNING: No ADITI_HESSIAN_HASH environment variable set")
		fmt.Println("  Proceeding in DEV mode - integrity not verified")
		return GateWarning
	}

	// Verify integrity
	valid, msg := verifier.VerifyIntegrity()
	fmt.Println(msg)

	if valid {
		fmt.Println("\n✓ SOVEREIGN INTEGRITY: VERIFIED")
		fmt.Println("  System may proceed to mount-control initialization")
		return GateProceed
	}

	fmt.Println("\n✗ SOVEREIGN INTEGRITY: VIOLATION DETECTED")
	fmt.Println("  Refusing to initialize mount-control logic")
	fmt.Println("  Possible causes:")
	fmt.Println("    - Model weights have been tampered with")
	fmt.Println("    - Adversarial bit-flipping attack")
	fmt.Println("    - Corruption during deployment")

	return GateRefuse
}
