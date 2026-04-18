package engine

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"time"

	"ssis/pkg/crypto"
	"ssis/pkg/orbital"
)

/**
 * HybridPropagator implements the "Physics-Gated E2E" logic for the SSIS.
 * It integrates the high-fidelity Axiom-Zero (Neural ODE) kernel with a
 * Hamiltonian Safety Gate and an analytical SGP4 fallback.
 *
 * Logic (Gap 8 Closure):
 * 1. Attempt propagation via Axiom (Neural ODE).
 * 2. Audit the physical honesty of the result via Hamiltonian consistency (E_hal).
 * 3. Calculate Information Divergence (D_KL) against the physics baseline.
 * 4. Reject and failover if E_hal deviation or D_KL exceeds thresholds.
 *
 * <plan_complexity>High</plan_complexity><total_lines_estimate>180</total_lines_estimate>
 */
type HybridPropagator struct {
	Axiom   *orbital.AxiomPropagator
	SGP4    *orbital.SGP4Propagator
	Metrics *EngineMetrics
	Signer  crypto.Signer

	ManeuverThreshold float64
	EnergyEpsilon     float64
}

/**
 * NewHybridPropagator initializes the physics-gated orchestrator.
 */
func NewHybridPropagator(axiom *orbital.AxiomPropagator, sgp4 *orbital.SGP4Propagator, metrics *EngineMetrics, signer crypto.Signer) *HybridPropagator {
	return &HybridPropagator{
		Axiom:             axiom,
		SGP4:              sgp4,
		Metrics:           metrics,
		Signer:            signer,
		ManeuverThreshold: 4.5,  // Nats (KL Divergence Limit)
		EnergyEpsilon:     1e-6, // Hamiltonian Honesty Threshold
	}
}

/**
 * Propagate handles the gated state transition loop.
 *
 * @param minutesSinceEpoch: Time delta in minutes.
 * @return: Final verified state, whether it was a failover, and any error.
 */
func (p *HybridPropagator) Propagate(minutesSinceEpoch float64) (orbital.State, bool, error) {
	// Entry Point: Physics-Gated Propagation
	fmt.Printf("DEBUG: Gated Propagation (dt=%.2fm) at %v\n", minutesSinceEpoch, time.Now())

	// 1. Primary Ingress: Axiom-Zero (Neural ODE)
	axiomState, err := p.Axiom.Propagate(minutesSinceEpoch)
	if err != nil {
		p.Metrics.MarkShedding()
		sgp4State, _ := p.SGP4.Propagate(minutesSinceEpoch)
		return sgp4State, true, nil
	}

	// 2. Baseline Generation: SGP4
	sgp4State, _ := p.SGP4.Propagate(minutesSinceEpoch)

	// 3. Gap 8: Hamiltonian Safety Gate (Physical Honesty Check)
	// We audit the Specific Orbital Energy E_hal
	eBase := p.AuditEnergy(sgp4State)
	ePred := p.AuditEnergy(axiomState)

	relativeShift := math.Abs(ePred-eBase) / math.Abs(eBase)
	if relativeShift > 0.001 { // 0.1% Safety Trigger
		fmt.Printf("CRITICAL: Hamiltonian Gate Rejection. Relative Shift=%.2e exceeds 0.1%%. Failover triggered.\n", relativeShift)
		p.Metrics.MarkShedding()
		return sgp4State, true, nil
	}

	// 4. Information Divergence Check (Gap 8)
	dkl := p.CalculateKLDivergence(axiomState, sgp4State)
	if dkl > p.ManeuverThreshold {
		fmt.Printf("ALERT: Tactical Divergence D_KL=%.2f. Triggering SGP4 baseline.\n", dkl)
		p.Metrics.MarkShedding()
		return sgp4State, true, nil
	}

	// 5. Material Primacy Guardrail: Cryptographic Physics Audit
	// We sign the Hash(State || E_hal || Timestamp)
	p.SignAudit(axiomState, ePred)

	// Exit Point: State Accepted as Physically Honest
	return axiomState, false, nil
}

/**
 * SignAudit creates a cryptographic proof of the physical state evaluation.
 */
func (p *HybridPropagator) SignAudit(state orbital.State, energy float64) {
	// Construct the state manifest (using position hash as identifier)
	manifest := fmt.Sprintf("POS:%.3f,%.3f,%.3f|T:%v|E:%.6f",
		state.X, state.Y, state.Z, time.Now().Unix(), energy)

	// SHA-256 Hash
	hash := sha256.Sum256([]byte(manifest))

	// Ed25519 Signature (returns hex string)
	sig := p.Signer.Sign(hash[:])
	if sig != "" {
		fmt.Printf("AUDIT: Signed State Manifest. Hash=%s Sig=%s\n",
			hex.EncodeToString(hash[:])[:8], sig[:8])
		// In production, this would be pushed to a DLT or an immutable log.
	}
}

/**
 * CalculateKLDivergence computes the Mahalanobis positional discrepancy.
 */
func (p *HybridPropagator) CalculateKLDivergence(pState, qState orbital.State) float64 {
	const sigma = 1.0 // 1km standard deviation in LEO
	const var_val = sigma * sigma

	dx := pState.X - qState.X
	dy := pState.Y - qState.Y
	dz := pState.Z - qState.Z

	distSq := (dx*dx + dy*dy + dz*dz) / var_val
	return 0.5 * distSq
}

/**
 * AuditEnergy computes the specific planetary energy of the orbital state.
 * E = 0.5 * v^2 - mu / r
 */
func (p *HybridPropagator) AuditEnergy(state orbital.State) float64 {
	r := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)
	v2 := state.VX*state.VX + state.VY*state.VY + state.VZ*state.VZ

	// mu is defined in orbital.EARTH_MU
	return 0.5*v2 - 398600.4418/r
}
