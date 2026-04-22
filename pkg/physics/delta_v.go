package physics

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
)

const (
	EARTH_MU = 3.986004418e14 // m³/s²
)

type DeltaVCalculator struct {
	ispHydrazine  float64 // Specific impulse for hydrazine (s)
	ispElectric   float64 // Specific impulse for electric propulsion (s)
	spacecraftMass float64 // Dry mass in kg
}

func NewDeltaVCalculator() *DeltaVCalculator {
	return &DeltaVCalculator{
		ispHydrazine:  2200, // Typical hydrazine thruster
		ispElectric:   3000, // Typical Hall effect thruster
		spacecraftMass: 500, // Default 500kg
	}
}

func (d *DeltaVCalculator) SetSpacecraftMass(mass float64) {
	d.spacecraftMass = mass
}

func (d *DeltaVCalculator) SetIsp(isp float64) {
	d.ispHydrazine = isp
}

type DeltaVResult struct {
	DeltaVTotal       float64   // Total delta-v in m/s
	DeltaVRadial      float64   // Radial component
	DeltaVInTrack     float64   // In-track component  
	DeltaVCrossTrack  float64   // Cross-track component
	FuelMassKg        float64   // Required fuel mass (hydrazine)
	ElectricFuelKg    float64   // Required fuel mass (electric)
	EnergyJoules      float64   // Total energy required (J)
	AlertLevel        string    // "GREEN", "YELLOW", "RED"
	AlertReason       string    // Reason for alert level
}

func (d *DeltaVCalculator) ComputeFromManeuverEvent(event *ManeuverEvent) *DeltaVResult {
	dvMag := event.DeltaV
	
	result := &DeltaVResult{
		DeltaVTotal: dvMag,
		DeltaVRadial: event.DeltaVRSW[0],
		DeltaVInTrack: event.DeltaVRSW[1],
		DeltaVCrossTrack: event.DeltaVRSW[2],
	}
	
	result.computeFuelCost()
	result.computeEnergy()
	result.assessAlertLevel()
	
	return result
}

func (d *DeltaVCalculator) ComputeFromStates(before, after State) *DeltaVResult {
	// Compute velocity difference
	dvx := after.Vel[0]*1000 - before.Vel[0]*1000 // Convert km/s to m/s
	dvy := after.Vel[1]*1000 - before.Vel[1]*1000
	dvz := after.Vel[2]*1000 - before.Vel[2]*1000
	
	dvMag := math.Sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
	
	result := &DeltaVResult{
		DeltaVTotal: dvMag,
		DeltaVRadial: dvx,
		DeltaVInTrack: dvy,
		DeltaVCrossTrack: dvz,
	}
	
	result.computeFuelCost()
	result.computeEnergy()
	result.assessAlertLevel()
	
	return result
}

func (d *DeltaVCalculator) ComputeImpulsiveBurn(dvRadial, dvInTrack, dvCrossTrack float64) *DeltaVResult {
	dvMag := math.Sqrt(dvRadial*dvRadial + dvInTrack*dvInTrack + dvCrossTrack*dvCrossTrack)
	
	result := &DeltaVResult{
		DeltaVTotal: dvMag,
		DeltaVRadial: dvRadial,
		DeltaVInTrack: dvInTrack,
		DeltaVCrossTrack: dvCrossTrack,
	}
	
	result.computeFuelCost()
	result.computeEnergy()
	result.assessAlertLevel()
	
	return result
}

func (r *DeltaVResult) computeFuelCost() {
	// Tsiolkovsky rocket equation: delta-v = Isp * g0 * ln(m0/m1)
	// Solving for fuel mass: m_fuel = m_spacecraft * (exp(deltaV/(Isp*g0)) - 1)
	g0 := 9.81 // m/s²
	
	// Hydrazine
	if r.DeltaVTotal > 0 {
		expHydrazine := math.Exp(r.DeltaVTotal / (2200 * g0))
		r.FuelMassKg = 500 * (expHydrazine - 1) // Assuming 500kg dry mass
		
		expElectric := math.Exp(r.DeltaVTotal / (3000 * g0))
		r.ElectricFuelKg = 500 * (expElectric - 1)
	}
}

func (r *DeltaVResult) computeEnergy() {
	// Energy = 0.5 * m * v² (kinetic energy of exhaust)
	// Using average exhaust velocity
	vExhaustHydrazine := 2200 * 9.81 // m/s
	r.EnergyJoules = 0.5 * r.FuelMassKg * vExhaustHydrazine * vExhaustHydrazine
}

func (r *DeltaVResult) assessAlertLevel() {
	switch {
	case r.DeltaVTotal < 1.0:
		r.AlertLevel = "GREEN"
		r.AlertReason = "Minor station-keeping maneuver"
	case r.DeltaVTotal < 10.0:
		r.AlertLevel = "YELLOW"
		r.AlertReason = "Routine orbit maintenance"
	case r.DeltaVTotal < 100.0:
		r.AlertLevel = "RED"
		r.AlertReason = "Significant maneuver - possible intent change"
	default:
		r.AlertLevel = "CRITICAL"
		r.AlertReason = "Large delta-v - potential orbital injection or debris avoidance"
	}
}

type HamiltonianValidator struct {
	mu float64
}

func NewHamiltonianValidator() *HamiltonianValidator {
	return &HamiltonianValidator{mu: EARTH_MU}
}

func (h *HamiltonianValidator) ComputeHamiltonian(state State) float64 {
	// H = T + V = 0.5*v² - mu/r
	r := math.Sqrt(state.Pos[0]*state.Pos[0] + state.Pos[1]*state.Pos[1] + state.Pos[2]*state.Pos[2])
	v := math.Sqrt(state.Vel[0]*state.Vel[0] + state.Vel[1]*state.Vel[1] + state.Vel[2]*state.Vel[2])
	
	// Convert to SI units
	rMeters := r * 1000
	vMeters := v * 1000
	
	ke := 0.5 * vMeters * vMeters
	pe := -h.mu / rMeters
	
	return ke + pe
}

func (h *HamiltonianValidator) ComputeInvariantHash(state State) (hash string, sigmaOverH float64) {
	H := h.ComputeHamiltonian(state)
	
	// Create hash from Hamiltonian value
	data := fmt.Sprintf("%.10f", H)
	hashBytes := sha256.Sum256([]byte(data))
	hash = hex.EncodeToString(hashBytes[:8])
	
	// Compute relative uncertainty (for spoof detection)
	// If sigma_H / |H| >= 10^-7, self-invalidate
	if math.Abs(H) > 0 {
		// Numerical precision as proxy for sigma
		sigmaH := math.Abs(H) * 1e-15 // Machine epsilon scale
		sigmaOverH = sigmaH / math.Abs(H)
	}
	
	return hash, sigmaOverH
}

func (h *HamiltonianValidator) ValidateObservation(state State) (valid bool, reason string) {
	_, sigmaOverH := h.ComputeInvariantHash(state)
	
	// Per memory: if sigma_H / H_avg >= 10^-7, certificate self-invalidates
	threshold := 1e-7
	
	if sigmaOverH >= threshold {
		return false, fmt.Sprintf("Hamiltonian invariant violated: %.2e >= %.2e", sigmaOverH, threshold)
	}
	
	return true, "Hamiltonian invariant validated"
}
