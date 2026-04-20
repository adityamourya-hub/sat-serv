package simulator

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"ssis/pkg/physics"
)

type ScenarioType string

const (
	ScenarioConjunction ScenarioType = "conjunction"
	ScenarioRapidThreat ScenarioType = "rapid_threat"
	ScenarioASAT        ScenarioType = "asat_intercept"
	ScenarioProximity   ScenarioType = "proximity_ops"
	ScenarioReentry     ScenarioType = "reentry_watch"
)

type ThreatClass string

const (
	ThreatUnknown    ThreatClass = "unknown"
	ThreatOrbital    ThreatClass = "orbital"
	ThreatBallistic  ThreatClass = "ballistic"
	ThreatHypersonic ThreatClass = "hypersonic"
)

type SyntheticScenario struct {
	ScenarioID          string       `json:"scenario_id"`
	Type                ScenarioType `json:"type"`
	AssetID             int          `json:"asset_id"`
	ThreatID            int          `json:"threat_id"`
	ThreatClass         ThreatClass  `json:"threat_class"`
	MissDistanceKm      float64      `json:"miss_distance_km"`
	RelativeVelocityKmS float64      `json:"relative_velocity_km_s"`
	TimeToTCASeconds    float64      `json:"time_to_tca_seconds"`
	SigmaRadialKm       float64      `json:"sigma_radial_km"`
	SigmaIntrackKm      float64      `json:"sigma_intrack_km"`
	SigmaCrosstrackKm   float64      `json:"sigma_crosstrack_km"`
	HardbodyRadiusKm    float64      `json:"hardbody_radius_km"`
	DistanceToAssetKm   float64      `json:"distance_to_asset_km"`
	Mach                float64      `json:"mach"`
	AltitudeKm          float64      `json:"altitude_km"`
	BurnoutVelocityKmS  float64      `json:"burnout_velocity_km_s"`
	RadarCrossSection   float64      `json:"radar_cross_section"`
	RecommendedAction   string       `json:"recommended_action"`
	ScenarioRationale   []string     `json:"scenario_rationale,omitempty"`
	DecisionFactors     []string     `json:"decision_factors,omitempty"`
	AdversarialIntent   float64      `json:"adversarial_intent"`
	RiskLabel           float64      `json:"risk_label"`
	PriorityLabel       float64      `json:"priority_label"`
	CollisionAvoided    bool         `json:"collision_avoided"`
	OperatorDifficulty  float64      `json:"operator_difficulty"`

	// ADITI 4.0: HGV-specific fields for glide dynamics and wake memory
	LiftDragRatio      float64 `json:"lift_drag_ratio"`       // L/D ratio (2-4 for HGV)
	BankAngleRad       float64 `json:"bank_angle_rad"`        // Bank angle for maneuver capability
	FlightPathAngleRad float64 `json:"flight_path_angle_rad"` // Gamma (negative for re-entry)
	WakeMemoryFactor   float64 `json:"wake_memory_factor"`    // Fractional Laplacian correction (0-1)
	AtmosphericDensity float64 `json:"atmospheric_density"`   // kg/m^3 at current altitude

	// ADITI 4.0: HGV wake spectroscopy parameters for plume/wake detection
	WakeTemperatureK    float64  `json:"wake_temperature_k"`            // Wake temperature (K)
	WakeIonizationLevel float64  `json:"wake_ionization_level"`         // Ionization fraction (0-1)
	WakeSpectralLines   []string `json:"wake_spectral_lines,omitempty"` // Dominant spectral lines
	PlumeLuminosity     float64  `json:"plume_luminosity"`              // Plume luminosity (W/sr)

	// ADITI 4.0: Sovereign Handshake protocol - Hamiltonian invariants for spoof detection
	HamiltonianInvariant float64 `json:"hamiltonian_invariant"` // Computed Hamiltonian value at observation epoch
	HamiltonianHash      string  `json:"hamiltonian_hash"`      // Hash of invariant for spoof detection
	HamiltonianSigma     float64 `json:"hamiltonian_sigma"`     // σ_H for spoof detection (σ_H / H_avg ≥ 10^-7)
	SpoofDetected        bool    `json:"spoof_detected"`        // True if invariant violates threshold
}

type GeneratorConfig struct {
	Count int
	Seed  int64
}

func DefaultGeneratorConfig() GeneratorConfig {
	return GeneratorConfig{Count: 10000, Seed: 42}
}

func GenerateSyntheticScenarios(cfg GeneratorConfig) []SyntheticScenario {
	if cfg.Count <= 0 {
		cfg.Count = 10000
	}

	rng = rand.New(rand.NewSource(cfg.Seed))
	out := make([]SyntheticScenario, 0, cfg.Count)

	for i := 0; i < cfg.Count; i++ {
		sType := sampleScenarioType(rng)

		missDistance := bounded(rng.NormFloat64()*0.04+0.08, 0.001, 1.2)
		relVel := bounded(rng.NormFloat64()*1.2+2.8, 0.05, 13.0)
		tca := bounded(rng.NormFloat64()*650+1800, 30, 7200)
		sigmaR := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		sigmaI := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		sigmaC := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		hbr := bounded(rng.NormFloat64()*0.003+0.008, 0.002, 0.03)

		distanceToAsset := bounded(rng.NormFloat64()*450+900, 50, 3200)
		mach := bounded(rng.NormFloat64()*2.3+5.2, 0.4, 13.0)
		altitude := bounded(rng.NormFloat64()*35+85, 5, 350)
		burnout := bounded(rng.NormFloat64()*1.1+2.4, 0.1, 7.0)
		rcs := bounded(rng.NormFloat64()*0.8+1.1, 0.05, 6.0)

		class := classifyThreat(mach, altitude, burnout)
		missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs = applyScenarioProfile(
			rng,
			sType,
			missDistance,
			relVel,
			tca,
			distanceToAsset,
			mach,
			altitude,
			burnout,
			rcs,
		)
		class = classifyThreat(mach, altitude, burnout)

		// ADITI 4.0: Generate HGV-specific dynamics for hypersonic threats
		var liftDragRatio, bankAngleRad, flightPathAngleRad, wakeMemoryFactor, atmosphericDensity, wakeTemperatureK, wakeIonizationLevel, plumeLuminosity float64
		if class == ThreatHypersonic && altitude >= 30 && altitude <= 80 {
			liftDragRatio, bankAngleRad, flightPathAngleRad, wakeMemoryFactor, atmosphericDensity, wakeTemperatureK, wakeIonizationLevel, plumeLuminosity = generateHGVDynamics(altitude, mach)
		} else {
			// Default values for non-HGV scenarios
			liftDragRatio = 0
			bankAngleRad = 0
			flightPathAngleRad = 0
			wakeMemoryFactor = 0
			atmosphericDensity = 0
			wakeTemperatureK = 0
			wakeIonizationLevel = 0
			plumeLuminosity = 0
		}

		// Use enhanced classification for HGV scenarios
		if liftDragRatio > 0 {
			class = classifyThreatEnhanced(mach, altitude, burnout, liftDragRatio, flightPathAngleRad)
		}

		risk := syntheticRisk(missDistance, hbr, sigmaR, sigmaI, sigmaC, relVel, tca, sType)

		// ADITI 4.0: Compute Hamiltonian invariants for Sovereign Handshake protocol
		// Estimate position and velocity from altitude, mach, and miss distance
		// This is a simplified model for synthetic scenario generation
		positionKm := [3]float64{
			math.Cos(math.Pi/4) * (physics.EARTH_R_EQUATORIAL + altitude),
			math.Sin(math.Pi/4) * (physics.EARTH_R_EQUATORIAL + altitude),
			altitude * 0.1, // Small inclination
		}
		velocityKmS := [3]float64{
			mach * 0.34 * math.Cos(math.Pi/4), // Mach to km/s (approximate)
			mach * 0.34 * math.Sin(math.Pi/4),
			0,
		}

		// Compute Hamiltonian invariant
		hamiltonianInvariant := computeHamiltonianInvariant(positionKm, velocityKmS, physics.EARTH_MU_IAU)

		// Compute Hamiltonian hash (use current JD as epoch)
		currentJD := 2460000.0 + float64(rng.Intn(365)) // Approximate current JD
		hamiltonianHash := computeHamiltonianHash(hamiltonianInvariant, currentJD)

		// Simulate Hamiltonian sigma (uncertainty in invariant computation)
		// For synthetic scenarios, add small random variation
		hamiltonianSigma := math.Abs(hamiltonianInvariant) * 1e-8 * rng.Float64()

		// Detect spoof (should be false for legitimate synthetic scenarios)
		// Use a nominal H_avg close to the computed H
		H_avg := hamiltonianInvariant * (1.0 + 1e-9*rng.Float64())
		spoofDetected := detectSpoof(hamiltonianInvariant, H_avg)
		priority := syntheticPriority(class, distanceToAsset, mach, rcs)
		action := recommendedAction(class, priority, distanceToAsset)
		rationale, decisionFactors, intent := scenarioExplainability(sType, class, risk, priority, distanceToAsset, tca, altitude)
		difficulty := bounded(0.3+0.4*risk+0.3*(priority/300.0), 0.0, 1.0)

		out = append(out, SyntheticScenario{
			ScenarioID:           fmt.Sprintf("SIM-%06d", i+1),
			Type:                 sType,
			AssetID:              43000 + rng.Intn(6000),
			ThreatID:             60000 + i,
			ThreatClass:          class,
			MissDistanceKm:       missDistance,
			RelativeVelocityKmS:  relVel,
			TimeToTCASeconds:     tca,
			SigmaRadialKm:        sigmaR,
			SigmaIntrackKm:       sigmaI,
			SigmaCrosstrackKm:    sigmaC,
			HardbodyRadiusKm:     hbr,
			DistanceToAssetKm:    distanceToAsset,
			Mach:                 mach,
			AltitudeKm:           altitude,
			BurnoutVelocityKmS:   burnout,
			RadarCrossSection:    rcs,
			RecommendedAction:    action,
			ScenarioRationale:    rationale,
			DecisionFactors:      decisionFactors,
			AdversarialIntent:    intent,
			RiskLabel:            risk,
			PriorityLabel:        priority,
			CollisionAvoided:     risk < 0.15 || action == "evasive_maneuver_now",
			OperatorDifficulty:   difficulty,
			LiftDragRatio:        liftDragRatio,
			BankAngleRad:         bankAngleRad,
			FlightPathAngleRad:   flightPathAngleRad,
			WakeMemoryFactor:     wakeMemoryFactor,
			AtmosphericDensity:   atmosphericDensity,
			WakeTemperatureK:     wakeTemperatureK,
			WakeIonizationLevel:  wakeIonizationLevel,
			WakeSpectralLines:    []string{}, // Empty for now, could be populated with specific lines
			PlumeLuminosity:      plumeLuminosity,
			HamiltonianInvariant: hamiltonianInvariant,
			HamiltonianHash:      hamiltonianHash,
			HamiltonianSigma:     hamiltonianSigma,
			SpoofDetected:        spoofDetected,
		})
	}

	return out
}

func sampleScenarioType(rng *rand.Rand) ScenarioType {
	r := rng.Float64()
	switch {
	case r < 0.32:
		return ScenarioConjunction
	case r < 0.60:
		return ScenarioRapidThreat
	case r < 0.75:
		return ScenarioASAT
	case r < 0.90:
		return ScenarioProximity
	default:
		return ScenarioReentry
	}
}

/**
 * generateHGVDynamics generates HGV-specific parameters for skip-glide trajectories.
 *
 * ADITI 4.0: Models the "Fractional Advantage" - HGV corridor (30-80km) is
 * Transitional Flow regime where standard Navier-Stokes fails. Uses wake memory
 * and fractional Laplacian concepts for realistic HGV behavior.
 *
 * Returns:
 * - liftDragRatio: L/D ratio (2-4 for HGV, varies with flight phase)
 * - bankAngleRad: Bank angle for maneuver capability (0-π/2)
 * - flightPathAngleRad: Gamma (negative for descending, -5° to -30°)
 * - wakeMemoryFactor: Fractional Laplacian correction (0-1, peaks at 55km)
 * - atmosphericDensity: kg/m^3 at current altitude
 * - wakeTemperatureK: Wake temperature (K)
 * - wakeIonizationLevel: Ionization fraction (0-1)
 * - plumeLuminosity: Plume luminosity (W/sr)
 */
func generateHGVDynamics(altitudeKm, mach float64) (liftDragRatio, bankAngleRad, flightPathAngleRad, wakeMemoryFactor, atmosphericDensity, wakeTemperatureK, wakeIonizationLevel, plumeLuminosity float64) {
	// L/D ratio: 2-4 for HGV, varies with altitude (lower at high altitude due to thin air)
	// Peak L/D at ~55km (center of HGV corridor)
	altitudeCenter := 55.0
	altitudeDeviation := math.Abs(altitudeKm-altitudeCenter) / altitudeCenter
	baseLD := 3.0 - 0.5*altitudeDeviation // Base L/D 3.0, decreases at edges
	liftDragRatio = bounded(baseLD+rngFloat()*0.5, 2.0, 4.0)

	// Bank angle: 0-90° (0-π/2 rad), higher for aggressive maneuvers
	// ASAT scenarios use higher bank angles for rapid threat
	bankAngleRad = bounded(rngFloat()*0.5, 0.0, math.Pi/2)

	// Flight path angle: Negative for descending trajectory
	// -5° to -30° (-0.087 to -0.524 rad)
	// Steeper at lower altitudes (closer to impact)
	steepness := (80.0 - altitudeKm) / 50.0 // 0 at 80km, 1 at 30km
	flightPathAngleRad = bounded(-(0.087 + 0.437*steepness*rngFloat()), -0.524, -0.087)

	// Wake memory factor: Fractional Laplacian correction
	// Peaks at 55km (center of HGV corridor), where transitional flow effects are strongest
	// Models non-local wake memory: past trajectory influences current dynamics
	wakeIntensity := math.Exp(-math.Pow((altitudeKm-altitudeCenter)/20.0, 2.0)) // Gaussian peak at 55km
	wakeMemoryFactor = bounded(wakeIntensity*(0.8+0.2*rngFloat()), 0.0, 1.0)

	// Atmospheric density: exponential model with altitude
	// Simple exponential atmosphere: ρ = ρ₀ * exp(-h/H)
	// ρ₀ = 1.225 kg/m³ (sea level), H = 7.5 km (scale height)
	// More accurate: NRLMSISE-00 model (use pkg/physics.GetDensityUnified)
	// For simulator: use simplified exponential for performance
	seaLevelDensity := 1.225 // kg/m³
	scaleHeight := 7.5       // km
	atmosphericDensity = seaLevelDensity * math.Exp(-altitudeKm/scaleHeight)

	// Density correction for HGV corridor (30-80km): transitional flow regime
	// Standard Navier-Stokes fails here, use fractional correction
	if altitudeKm >= 30 && altitudeKm <= 80 {
		// Fractional correction factor (0.8-1.2) based on wake memory
		fractionalCorrection := 1.0 + 0.2*(wakeMemoryFactor-0.5)
		atmosphericDensity *= fractionalCorrection
	}

	// Wake spectroscopy parameters for plume/wake detection
	// Temperature increases with Mach number (shock heating)
	// Base temperature at 30km: ~200K, increases with mach
	baseTemp := 200.0 + (mach-5.0)*100.0
	wakeTemperatureK = bounded(baseTemp+rngFloat()*50.0, 200.0, 3000.0)

	// Ionization level increases with temperature and decreases with altitude
	// Peak ionization at ~50-60km (ionosphere D/E region)
	ionizationBase := math.Exp(-math.Pow((altitudeKm-55.0)/15.0, 2.0)) // Peak at 55km
	ionizationTempFactor := (wakeTemperatureK - 200.0) / 2800.0        // Higher temp = more ionization
	wakeIonizationLevel = bounded(ionizationBase*ionizationTempFactor, 0.0, 1.0)

	// Plume luminosity scales with Mach number and atmospheric density
	// Higher mach = more combustion energy, higher density = more emission
	plumeLuminosity = mach * 1000.0 * atmosphericDensity * (0.8 + 0.4*rngFloat())

	return liftDragRatio, bankAngleRad, flightPathAngleRad, wakeMemoryFactor, atmosphericDensity, wakeTemperatureK, wakeIonizationLevel, plumeLuminosity
}

/**
 * rngFloat is a helper for random float generation in range [0,1)
 * Uses global RNG from GenerateSyntheticScenarios context
 */
func rngFloat() float64 {
	return rng.Float64()
}

/**
 * computeHamiltonianInvariant computes the Hamiltonian (energy) invariant from orbital state.
 *
 * For Keplerian motion, the Hamiltonian H = 0.5*v² - μ/r is conserved.
 * This invariant is used for the Sovereign Handshake protocol to detect spoofing.
 *
 * ADITI 4.0: Hamiltonian invariants enable detection of state manipulation.
 * If σ_H / H_avg ≥ 10^-7, the observation is flagged as potentially spoofed.
 *
 * @param positionKm Position vector [x, y, z] in km
 * @param velocityKmS Velocity vector [vx, vy, vz] in km/s
 * @param mu Gravitational parameter (km³/s²), default EARTH_MU_IAU
 * @return Hamiltonian invariant value
 */
func computeHamiltonianInvariant(positionKm, velocityKmS [3]float64, mu float64) float64 {
	if mu <= 0 {
		mu = physics.EARTH_MU_IAU
	}

	// Compute magnitude
	r := math.Sqrt(positionKm[0]*positionKm[0] + positionKm[1]*positionKm[1] + positionKm[2]*positionKm[2])
	v2 := velocityKmS[0]*velocityKmS[0] + velocityKmS[1]*velocityKmS[1] + velocityKmS[2]*velocityKmS[2]

	if r <= 0 {
		return 0
	}

	// Keplerian Hamiltonian: H = 0.5*v² - μ/r
	H := 0.5*v2 - mu/r

	return H
}

/**
 * computeHamiltonianHash generates a hash of the Hamiltonian invariant for spoof detection.
 *
 * Uses SHA-256 to hash the Hamiltonian value with a timestamp for temporal binding.
 * The hash is hex-encoded for transmission in ObservationReport.
 *
 * @param H Hamiltonian invariant value
 * @param epochJD Julian Date for temporal binding
 * @return Hex-encoded hash string
 */
func computeHamiltonianHash(H float64, epochJD float64) string {
	// Combine Hamiltonian with epoch for temporal binding
	data := fmt.Sprintf("%.15f|%.15f", H, epochJD)

	// Compute SHA-256 hash
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("%x", hash)
}

/**
 * detectSpoof checks if the Hamiltonian invariant violates the spoof detection threshold.
 *
 * Sovereign Handshake protocol: If σ_H / H_avg ≥ 10^-7, the certificate self-invalidates.
 *
 * @param H Current Hamiltonian value
 * @param H_avg Average Hamiltonian value (from historical observations)
 * @return True if spoof detected (invariant violates threshold)
 */
func detectSpoof(H, H_avg float64) bool {
	if H_avg == 0 {
		return false
	}

	sigmaH := math.Abs(H - H_avg)
	ratio := sigmaH / math.Abs(H_avg)

	// Threshold: 10^-7
	return ratio >= 1e-7
}

/**
 * FractionalLaplacianKernel computes the fractional Laplacian for non-local threat propagation.
 *
 * ADITI 4.0: The "Fractional Advantage" - HGV corridor (30-80km) is Transitional Flow regime
 * where standard Navier-Stokes fails. The fractional Laplacian models non-local wake memory:
 * past trajectory influences current dynamics beyond local neighborhood.
 *
 * The fractional Laplacian (-Δ)^α with α ∈ (0,1) is defined via the Riesz potential:
 * (-Δ)^α u(x) = C(α) * P.V. ∫ (u(x) - u(y)) / |x-y|^(n+2α) dy
 *
 * For discrete implementation, we use a spectral approach with power-law scaling:
 * L_α = F^{-1}[|k|^(2α) * F[u]]
 *
 * @param currentRisk Current risk value at position
 * @param historicalRisk Array of historical risk values (wake memory)
 * @param alpha Fractional order (0 < α ≤ 1), typical: 0.5-0.8 for HGV
 * @return Fractional Laplacian correction factor
 */
func FractionalLaplacianKernel(currentRisk float64, historicalRisk []float64, alpha float64) float64 {
	if alpha <= 0 || alpha > 1 {
		alpha = 0.7 // Default for HGV corridor
	}

	if len(historicalRisk) == 0 {
		return 1.0 // No correction without history
	}

	// Spectral approach: power-law scaling of frequency components
	// For 1D case: L_α = F^{-1}[|k|^(2α) * F[risk]]

	// Simplified implementation: weighted sum with power-law decay
	// Higher α = stronger non-local effects (longer memory)
	n := len(historicalRisk)
	sum := 0.0
	weightSum := 0.0

	for i, risk := range historicalRisk {
		// Power-law weight: w(i) = (n-i)^(-α)
		// Recent history has higher weight
		distance := float64(n - i)
		weight := math.Pow(distance, -alpha)

		sum += weight * risk
		weightSum += weight
	}

	if weightSum == 0 {
		return 1.0
	}

	// Fractional correction: ratio of current risk to weighted historical average
	historicalAvg := sum / weightSum
	if historicalAvg == 0 {
		return 1.0
	}

	correction := currentRisk / historicalAvg

	// Clamp correction to reasonable bounds [0.5, 2.0]
	correction = bounded(correction, 0.5, 2.0)

	return correction
}

/**
 * computeNonLocalThreatPropagation applies fractional Laplacian for threat propagation.
 *
 * ADITI 4.0: Uses wake memory to anticipate HGV position based on past trajectory,
 * not just current position. This is the core "Fractional Advantage" over legacy radar.
 *
 * @param currentRisk Current risk level
 * @param historicalRisk Historical risk observations (wake memory)
 * @param alpha Fractional order (default 0.7 for HGV corridor)
 * @return Adjusted risk with non-local propagation
 */
func computeNonLocalThreatPropagation(currentRisk float64, historicalRisk []float64, alpha float64) float64 {
	correction := FractionalLaplacianKernel(currentRisk, historicalRisk, alpha)
	return currentRisk * correction
}

// Global RNG for HGV dynamics generation (initialized in GenerateSyntheticScenarios)
var rng *rand.Rand

func applyScenarioProfile(
	rng *rand.Rand,
	sType ScenarioType,
	missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs float64,
) (float64, float64, float64, float64, float64, float64, float64, float64) {
	switch sType {
	case ScenarioASAT:
		missDistance = bounded(rng.NormFloat64()*0.03+0.05, 0.001, 0.8)
		relVel = bounded(rng.NormFloat64()*1.7+7.5, 1.8, 13.0)
		tca = bounded(rng.NormFloat64()*400+900, 20, 3600)
		distanceToAsset = bounded(rng.NormFloat64()*300+650, 40, 1800)
		mach = bounded(rng.NormFloat64()*1.8+9.5, 3.0, 13.0)
		altitude = bounded(rng.NormFloat64()*50+450, 120, 1100)
		burnout = bounded(rng.NormFloat64()*0.9+4.2, 1.2, 7.0)
		rcs = bounded(rng.NormFloat64()*0.5+0.7, 0.05, 3.0)
	case ScenarioProximity:
		missDistance = bounded(rng.NormFloat64()*0.09+0.35, 0.01, 3.0)
		relVel = bounded(rng.NormFloat64()*0.12+0.22, 0.02, 1.1)
		tca = bounded(rng.NormFloat64()*2600+8400, 600, 86400)
		distanceToAsset = bounded(rng.NormFloat64()*180+280, 10, 1300)
		mach = bounded(rng.NormFloat64()*0.8+2.1, 0.3, 5.5)
		altitude = bounded(rng.NormFloat64()*120+650, 180, 1800)
		burnout = bounded(rng.NormFloat64()*0.5+1.7, 0.2, 4.5)
		rcs = bounded(rng.NormFloat64()*1.1+2.8, 0.2, 8.0)
	case ScenarioReentry:
		missDistance = bounded(rng.NormFloat64()*0.6+4.0, 0.2, 15.0)
		relVel = bounded(rng.NormFloat64()*0.8+3.2, 0.6, 9.0)
		tca = bounded(rng.NormFloat64()*800+2600, 120, 9600)
		distanceToAsset = bounded(rng.NormFloat64()*500+1200, 120, 4200)
		mach = bounded(rng.NormFloat64()*1.1+6.2, 2.5, 12.5)
		altitude = bounded(rng.NormFloat64()*25+90, 15, 250)
		burnout = bounded(rng.NormFloat64()*0.7+2.2, 0.4, 5.5)
		rcs = bounded(rng.NormFloat64()*1.3+3.0, 0.2, 10.0)
	}
	return missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs
}

func scenarioExplainability(
	sType ScenarioType,
	class ThreatClass,
	risk, priority, distanceToAsset, tca, altitude float64,
) ([]string, []string, float64) {
	rationale := []string{fmt.Sprintf("scenario=%s", sType)}
	factors := []string{
		fmt.Sprintf("risk=%.3f", risk),
		fmt.Sprintf("priority=%.1f", priority),
		fmt.Sprintf("distance_km=%.1f", distanceToAsset),
	}
	intent := 0.2

	switch sType {
	case ScenarioASAT:
		rationale = append(rationale, "high closure and short warning windows indicate interceptor profile")
		factors = append(factors, fmt.Sprintf("time_to_tca_sec=%.0f", tca))
		intent = bounded(0.7+0.25*(priority/300.0), 0.0, 1.0)
	case ScenarioProximity:
		rationale = append(rationale, "persistent close approach with low relative velocity indicates inspection or stalking")
		factors = append(factors, fmt.Sprintf("altitude_km=%.1f", altitude))
		intent = bounded(0.35+0.4*(1.0-math.Min(distanceToAsset/1500.0, 1.0)), 0.0, 1.0)
	case ScenarioReentry:
		rationale = append(rationale, "descending trajectory suggests controlled or uncontrolled re-entry window")
		factors = append(factors, fmt.Sprintf("class=%s", class))
		intent = bounded(0.15+0.25*(priority/300.0), 0.0, 1.0)
	default:
		rationale = append(rationale, "standard conjunction and rapid-threat watchlist flow")
		intent = bounded(0.1+0.35*(priority/300.0), 0.0, 1.0)
	}

	return rationale, factors, intent
}

func WriteScenariosJSONL(path string, scenarios []SyntheticScenario) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, s := range scenarios {
		if err := enc.Encode(s); err != nil {
			return err
		}
	}
	return nil
}

func WriteScenariosJSON(path string, scenarios []SyntheticScenario) error {
	payload := struct {
		Scenarios []SyntheticScenario `json:"scenarios"`
	}{Scenarios: scenarios}

	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func bounded(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

/**
 * adaptiveSigma computes uncertainty sigma based on miss distance and scenario type.
 *
 * Problem: Fixed sigma=0.03km causes Gaussian to be near-zero for far-field conjunctions
 * (e.g., reentry scenarios with 15km miss distance).
 *
 * Solution: Scale sigma proportionally to miss distance to maintain meaningful risk
 * discrimination across all scenario types.
 *
 * Sigma scaling rules:
 * - Conjunction: sigma = max(0.03, 0.1 * missDistance) - tight conjunctions
 * - RapidThreat: sigma = max(0.03, 0.15 * missDistance) - moderate uncertainty
 * - ASAT: sigma = max(0.05, 0.2 * missDistance) - high uncertainty interceptor
 * - Proximity: sigma = max(0.05, 0.25 * missDistance) - persistent ops
 * - Reentry: sigma = max(0.05, 0.3 * missDistance) - atmospheric uncertainty
 *
 * This ensures the Gaussian term exp(-0.5*(x/σ)²) remains non-trivial across
 * the full miss distance range (0.001km to 15km).
 */
func adaptiveSigma(missDistance float64, sType ScenarioType) float64 {
	if missDistance <= 0 {
		return 0.03
	}

	var scale float64
	var minSigma float64

	switch sType {
	case ScenarioConjunction:
		scale = 0.1
		minSigma = 0.03
	case ScenarioRapidThreat:
		scale = 0.15
		minSigma = 0.03
	case ScenarioASAT:
		scale = 0.2
		minSigma = 0.05
	case ScenarioProximity:
		scale = 0.25
		minSigma = 0.05
	case ScenarioReentry:
		scale = 0.3
		minSigma = 0.05
	default:
		scale = 0.15
		minSigma = 0.03
	}

	sigma := scale * missDistance
	if sigma < minSigma {
		sigma = minSigma
	}
	return sigma
}

/**
 * adaptiveSigmaFromCovariance computes sigma from position uncertainty covariance.
 * Used in env.go where scenario type may not be available.
 *
 * Sigma = (sigmaR + sigmaI + sigmaC) / 3.0, but scaled to prevent
 * near-zero Gaussian for large miss distances.
 */
func adaptiveSigmaFromCovariance(missDistance, sigmaR, sigmaI, sigmaC float64) float64 {
	sigmaEff := (sigmaR + sigmaI + sigmaC) / 3.0

	// Scale sigma with miss distance to maintain meaningful Gaussian
	// For large miss distances, use proportional scaling
	if missDistance > 1.0 {
		scale := 0.15 + 0.1*math.Log(missDistance) // Logarithmic scaling
		sigmaScaled := scale * missDistance
		if sigmaScaled > sigmaEff {
			sigmaEff = sigmaScaled
		}
	}

	// Minimum sigma to prevent division by zero
	if sigmaEff < 0.03 {
		sigmaEff = 0.03
	}

	return sigmaEff
}

/**
 * classifyThreatEnhanced classifies threat type with HGV-specific indicators.
 *
 * ADITI 4.0: Enhanced classification includes glide dynamics (L/D ratio),
 * flight path angle, and atmospheric density for HGV detection in the
 * transitional flow regime (30-80km).
 *
 * HGV Indicators:
 * - Mach ≥ 5 AND altitude 30-80km (HGV corridor)
 * - L/D ratio 2-4 (glide capability)
 * - Negative flight path angle (descending trajectory)
 * - High atmospheric density variation (wake memory effects)
 */
func classifyThreatEnhanced(mach, altitude, burnout, liftDragRatio, flightPathAngleRad float64) ThreatClass {
	// HGV detection: Mach ≥ 5 AND HGV corridor (30-80km)
	// Additional check for glide capability (L/D ratio 2-4)
	if mach >= 5 && altitude >= 30 && altitude <= 80 {
		// Confirm HGV if L/D ratio indicates glide capability
		if liftDragRatio >= 2.0 && liftDragRatio <= 4.0 {
			return ThreatHypersonic
		}
		// Fallback: Mach + altitude in HGV corridor
		return ThreatHypersonic
	}

	// Legacy hypersonic check (broader range)
	if mach >= 5 && altitude >= 20 && altitude <= 100 {
		return ThreatHypersonic
	}

	// Ballistic: burnout velocity ≥ 1 km/s AND altitude < 300km
	if burnout >= 1.0 && altitude < 300 {
		return ThreatBallistic
	}

	// Orbital: altitude ≥ 120km
	if altitude >= 120 {
		return ThreatOrbital
	}

	return ThreatUnknown
}

// Legacy classifyThreat for backward compatibility
func classifyThreat(mach, altitude, burnout float64) ThreatClass {
	return classifyThreatEnhanced(mach, altitude, burnout, 0, 0)
}

func syntheticRisk(missDistance, hbr, sigmaR, sigmaI, sigmaC, relVel, tca float64, sType ScenarioType) float64 {
	sigmaEff := adaptiveSigma(missDistance, sType)
	gaussian := math.Exp(-0.5*math.Pow(missDistance/math.Max(sigmaEff, 1e-6), 2.0)) * math.Pow(hbr/math.Max(sigmaEff, 1e-6), 2.0)
	velocityBias := 1.0 / (1.0 + relVel*0.015)
	timeBias := 1.0 - math.Exp(-tca/1200.0)
	pc := gaussian * velocityBias * (0.5 + 0.5*timeBias)
	return bounded(pc, 0.0, 1.0)
}

func syntheticPriority(class ThreatClass, distanceToAsset, mach, rcs float64) float64 {
	score := 20.0
	switch class {
	case ThreatHypersonic:
		score = 100
	case ThreatBallistic:
		score = 80
	case ThreatOrbital:
		score = 40
	}

	if distanceToAsset <= 500 {
		score += 50
	} else if distanceToAsset <= 1500 {
		score += 25
	}

	if mach >= 8 {
		score += 25
	} else if mach >= 5 {
		score += 15
	}
	if rcs < 0.5 {
		score += 10
	}
	return bounded(score, 0.0, 300.0)
}

func recommendedAction(class ThreatClass, priority, distanceToAsset float64) string {
	if distanceToAsset > 2000 {
		return "out_of_reach_watch"
	}
	switch {
	case class == ThreatHypersonic && priority >= 140:
		return "evasive_maneuver_now"
	case class == ThreatBallistic && priority >= 120:
		return "prepare_evasive"
	case priority >= 90:
		return "sensor_retask"
	default:
		return "monitor_only"
	}
}
