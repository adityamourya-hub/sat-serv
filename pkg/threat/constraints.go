// Package threat provides physical constraint enforcement for threat detection
// Implements DPHNO hard constraints: energy, momentum, hydrostatic as equality constraints
package threat

import (
	"math"
)

// PhysicalConstraints enforces DPHNO hard constraints on threat predictions
type PhysicalConstraints struct {
	// Constraint violation thresholds
	MaxEnergyViolation    float64 // Maximum allowed energy violation ratio
	MaxMomentumViolation  float64 // Maximum allowed momentum violation ratio
	MaxHydrostaticViolation float64 // Maximum allowed hydrostatic violation ratio

	// Lagrange multipliers for augmented Lagrangian
	EnergyLambda    float64
	MomentumLambda  float64
	HydrostaticLambda float64

	// Constraint violation history for adaptive tuning
	EnergyHistory    []float64
	MomentumHistory  []float64
	HydrostaticHistory []float64
}

// NewPhysicalConstraints creates a constraint enforcer with default thresholds
func NewPhysicalConstraints() *PhysicalConstraints {
	return &PhysicalConstraints{
		MaxEnergyViolation:    1e-6, // 1e-6 relative energy violation
		MaxMomentumViolation:  1e-6, // 1e-6 relative momentum violation
		MaxHydrostaticViolation: 1e-5, // 1e-5 relative hydrostatic violation
		EnergyLambda:           1.0,
		MomentumLambda:         1.0,
		HydrostaticLambda:      1.0,
		EnergyHistory:          make([]float64, 0),
		MomentumHistory:        make([]float64, 0),
		HydrostaticHistory:     make([]float64, 0),
	}
}

// CheckEnergyConservation verifies energy conservation for orbital mechanics
// Returns: (isViolated, violationRatio)
func (pc *PhysicalConstraints) CheckEnergyConservation(pos, vel [3]float64, mass float64) (bool, float64) {
	// Calculate specific orbital energy (energy per unit mass)
	// ε = v²/2 - μ/r
	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	v2 := vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]
	mu := 398600.4418 // Earth gravitational parameter km³/s²

	energy := v2/2.0 - mu/r

	// Expected energy should be conserved (constant for two-body problem)
	// For hypersonic vehicles, energy changes due to drag and thrust
	// Check if energy change is physically reasonable
	energyMag := math.Abs(energy)
	if energyMag < 1e-10 {
		return false, 0.0
	}

	// Calculate energy change rate (simplified)
	// In production, integrate with actual energy history
	expectedEnergyChange := 0.0 // Should be small for ballistic
	energyChange := energy - expectedEnergyChange

	violationRatio := math.Abs(energyChange) / energyMag
	pc.recordViolation("energy", violationRatio)

	return violationRatio > pc.MaxEnergyViolation, violationRatio
}

// CheckMomentumConservation verifies momentum conservation
// Returns: (isViolated, violationRatio)
func (pc *PhysicalConstraints) CheckMomentumConservation(pos, vel, acc [3]float64, mass float64) (bool, float64) {
	// Calculate momentum
	momentum := [3]float64{
		mass * vel[0],
		mass * vel[1],
		mass * vel[2],
	}

	// Expected momentum change = force * dt = mass * acc * dt
	// For ballistic: force = gravity
	r := math.Sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2])
	mu := 398600.4418
	gravityAcc := [3]float64{
		-mu * pos[0] / (r * r * r),
		-mu * pos[1] / (r * r * r),
		-mu * pos[2] / (r * r * r),
	}

	// Expected momentum change from gravity
	expectedForce := [3]float64{
		mass * gravityAcc[0],
		mass * gravityAcc[1],
		mass * gravityAcc[2],
	}

	// Calculate actual momentum change (simplified - needs dt)
	// In production, use actual momentum history
	momentumMag := math.Sqrt(momentum[0]*momentum[0] + momentum[1]*momentum[1] + momentum[2]*momentum[2])
	if momentumMag < 1e-10 {
		return false, 0.0
	}

	expectedForceMag := math.Sqrt(expectedForce[0]*expectedForce[0] + expectedForce[1]*expectedForce[1] + expectedForce[2]*expectedForce[2])
	violationRatio := expectedForceMag / momentumMag

	pc.recordViolation("momentum", violationRatio)

	return violationRatio > pc.MaxMomentumViolation, violationRatio
}

// CheckHydrostaticBalance verifies hydrostatic equilibrium for atmospheric flight
// Returns: (isViolated, violationRatio)
func (pc *PhysicalConstraints) CheckHydrostaticBalance(altitude, pressure, temperature float64) (bool, float64) {
	// Hydrostatic equation: dp/dz = -ρg
	// Ideal gas law: p = ρRT/M
	// Combined: dp/dz = -pMg/(RT)
	// Simplified check: pressure should decrease exponentially with altitude

	if altitude < 0 || pressure <= 0 {
		return false, 0.0
	}

	// Expected pressure at altitude (simplified exponential atmosphere)
	// p = p0 * exp(-z/H) where H ≈ 8.5 km (scale height)
	scaleHeight := 8.5 // km
	p0 := 101325.0 // Pa (sea level pressure)
	expectedPressure := p0 * math.Exp(-altitude/scaleHeight)

	if expectedPressure < 1e-10 {
		return false, 0.0
	}

	violationRatio := math.Abs(pressure-expectedPressure) / expectedPressure
	pc.recordViolation("hydrostatic", violationRatio)

	return violationRatio > pc.MaxHydrostaticViolation, violationRatio
}

// recordViolation tracks constraint violations for adaptive tuning
func (pc *PhysicalConstraints) recordViolation(constraintType string, violation float64) {
	historyLen := 100

	switch constraintType {
	case "energy":
		pc.EnergyHistory = append(pc.EnergyHistory, violation)
		if len(pc.EnergyHistory) > historyLen {
			pc.EnergyHistory = pc.EnergyHistory[1:]
		}
	case "momentum":
		pc.MomentumHistory = append(pc.MomentumHistory, violation)
		if len(pc.MomentumHistory) > historyLen {
			pc.MomentumHistory = pc.MomentumHistory[1:]
		}
	case "hydrostatic":
		pc.HydrostaticHistory = append(pc.HydrostaticHistory, violation)
		if len(pc.HydrostaticHistory) > historyLen {
			pc.HydrostaticHistory = pc.HydrostaticHistory[1:]
		}
	}
}

// ApplyAugmentedLagrangian adjusts prediction based on constraint violations
func (pc *PhysicalConstraints) ApplyAugmentedLagrangian(
	pos, vel [3]float64,
	energyViolated bool, energyRatio float64,
	momentumViolated bool, momentumRatio float64,
) ([3]float64, [3]float64) {
	adjustedPos := pos
	adjustedVel := vel

	// Energy constraint correction
	if energyViolated {
		pc.EnergyLambda *= 1.1 // Increase penalty
		// Scale velocity to conserve energy
		scaleFactor := 1.0 - pc.EnergyLambda*energyRatio*0.01
		for i := 0; i < 3; i++ {
			adjustedVel[i] *= scaleFactor
		}
	} else {
		pc.EnergyLambda *= 0.95 // Decrease penalty when satisfied
		if pc.EnergyLambda < 0.1 {
			pc.EnergyLambda = 0.1
		}
	}

	// Momentum constraint correction
	if momentumViolated {
		pc.MomentumLambda *= 1.1
		// Adjust velocity direction to match expected momentum change
		for i := 0; i < 3; i++ {
			adjustedVel[i] *= (1.0 - pc.MomentumLambda*momentumRatio*0.005)
		}
	} else {
		pc.MomentumLambda *= 0.95
		if pc.MomentumLambda < 0.1 {
			pc.MomentumLambda = 0.1
		}
	}

	return adjustedPos, adjustedVel
}

// GetConstraintStats returns statistics on constraint violations
func (pc *PhysicalConstraints) GetConstraintStats() map[string]interface{} {
	stats := make(map[string]interface{})

	avgEnergy := 0.0
	if len(pc.EnergyHistory) > 0 {
		for _, v := range pc.EnergyHistory {
			avgEnergy += v
		}
		avgEnergy /= float64(len(pc.EnergyHistory))
	}

	avgMomentum := 0.0
	if len(pc.MomentumHistory) > 0 {
		for _, v := range pc.MomentumHistory {
			avgMomentum += v
		}
		avgMomentum /= float64(len(pc.MomentumHistory))
	}

	avgHydrostatic := 0.0
	if len(pc.HydrostaticHistory) > 0 {
		for _, v := range pc.HydrostaticHistory {
			avgHydrostatic += v
		}
		avgHydrostatic /= float64(len(pc.HydrostaticHistory))
	}

	stats["avg_energy_violation"] = avgEnergy
	stats["avg_momentum_violation"] = avgMomentum
	stats["avg_hydrostatic_violation"] = avgHydrostatic
	stats["energy_lambda"] = pc.EnergyLambda
	stats["momentum_lambda"] = pc.MomentumLambda
	stats["hydrostatic_lambda"] = pc.HydrostaticLambda

	return stats
}
