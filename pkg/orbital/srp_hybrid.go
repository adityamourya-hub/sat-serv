package orbital

import (
	"math"
	"ssis/pkg/physics"
)

/**
 * Hybrid SRP Engine
 *
 * Tier 1: Own Assets (Box-Wing + ECOM1/5-parameter)
 * Tier 2: Unknown/Adversary (ECOM2/9-parameter)
 *
 * Combines physical Box-Wing model (95% of physics) with
 * empirical ECOM residuals (5% unmodeled effects).
 *
 * Reference: Springer "Satellite Orbits" (Montenbruck & Gill)
 * ECOM2: 9-parameter formulation with periodic terms
 */

/**
 * SRPModelType selects the SRP model tier.
 */
type SRPModelType int

const (
	SRPCannonball SRPModelType = iota // Simple sphere
	SRPBoxWing                        // Physical model for known assets
	SRPECOM1                          // 5-param ECOM (D0, Y0, B0, Bc, Bs)
	SRPECOM2                          // 9-param ECOM (full periodic)
	SRPHybrid                         // Box-Wing + ECOM1 residuals
)

/**
 * ECOM2Params holds the full 9-parameter CODE Orbit Model.
 *
 * Frame: Sun-oriented (D, Y, B)
 * - D: Direct (toward Sun)
 * - Y: Along solar panel rotation axis
 * - B: Bi-normal (completes orthogonal set)
 *
 * Full 9-parameter formulation:
 * D(u) = D0 + D2c·cos(2u) + D2s·sin(2u) + D4c·cos(4u) + D4s·sin(4u)
 * Y(u) = Y0
 * B(u) = B0 + Bc·cos(u) + Bs·sin(u)
 */
type ECOM2Params struct {
	// D-direction (Direct, toward Sun)
	D0  float64 // Constant term
	D2c float64 // cos(2u) coefficient
	D2s float64 // sin(2u) coefficient
	D4c float64 // cos(4u) coefficient
	D4s float64 // sin(4u) coefficient

	// Y-direction (solar panel axis)
	Y0 float64 // Constant bias

	// B-direction (bi-normal)
	B0 float64 // Constant
	Bc float64 // cos(u) coefficient
	Bs float64 // sin(u) coefficient
}

/**
 * ECOM1Params holds the reduced 5-parameter model.
 * Used when fewer observations available or for Hybrid model.
 */
type ECOM1Params struct {
	D0 float64
	Y0 float64
	B0 float64
	Bc float64
	Bs float64
}

/**
 * HybridSRPModel combines Box-Wing with ECOM1 residuals.
 */
type HybridSRPModel struct {
	BoxWing   BoxWingModel
	Residuals ECOM1Params // ECOM for unmodeled effects
}

/**
 * SRPEngine provides unified SRP computation with tier selection.
 */
type SRPEngine struct {
	ModelType SRPModelType

	// Model-specific parameters
	BoxWing *BoxWingModel
	ECOM1   *ECOM1Params
	ECOM2   *ECOM2Params
	Hybrid  *HybridSRPModel

	// Shadow model (conical for all tiers)
	UseConicalShadow bool
}

/**
 * NewSRPEngine creates a configured SRP engine.
 */
func NewSRPEngine(modelType SRPModelType) *SRPEngine {
	return &SRPEngine{
		ModelType:        modelType,
		UseConicalShadow: true,
	}
}

/**
 * SetBoxWing configures the engine for Tier 1 (known assets).
 */
func (e *SRPEngine) SetBoxWing(model BoxWingModel) {
	e.BoxWing = &model
	if e.ModelType == SRPBoxWing {
		e.BoxWing = &model
	}
}

/**
 * SetECOM2 configures the engine for Tier 2 (9-parameter adversary tracking).
 */
func (e *SRPEngine) SetECOM2(params ECOM2Params) {
	e.ECOM2 = &params
}

/**
 * SetECOM1 configures the reduced 5-parameter model.
 */
func (e *SRPEngine) SetECOM1(params ECOM1Params) {
	e.ECOM1 = &params
}

/**
 * SetHybrid configures Tier 1 hybrid (Box-Wing + ECOM1 residuals).
 */
func (e *SRPEngine) SetHybrid(boxWing BoxWingModel, residuals ECOM1Params) {
	e.Hybrid = &HybridSRPModel{
		BoxWing:   boxWing,
		Residuals: residuals,
	}
}

/**
 * ComputeAcceleration calculates SRP using the selected model tier.
 *
 * @param satPos: Satellite position ECI (km)
 * @param sunPos: Sun position ECI (km)
 * @param u: Argument of latitude (rad) - for periodic terms
 * @param jd: Julian Date
 * @return: Acceleration (ax, ay, az) km/s^2
 */
func (e *SRPEngine) ComputeAcceleration(satPos, sunPos [3]float64, u float64, jd float64) (ax, ay, az float64) {
	switch e.ModelType {
	case SRPBoxWing:
		if e.BoxWing != nil {
			return e.computeBoxWing(satPos, sunPos)
		}
	case SRPECOM2:
		if e.ECOM2 != nil {
			return e.computeECOM2(satPos, sunPos, u, jd)
		}
	case SRPECOM1:
		if e.ECOM1 != nil {
			return e.computeECOM1(satPos, sunPos, u, jd)
		}
	case SRPHybrid:
		if e.Hybrid != nil {
			return e.computeHybrid(satPos, sunPos, u, jd)
		}
	default:
		// Cannonball fallback
		return SRPAcceleration(satPos, sunPos, 50.0, 3000.0, 1.3)
	}

	// Fallback to simple model
	return SRPAcceleration(satPos, sunPos, 50.0, 3000.0, 1.3)
}

/**
 * computeBoxWing uses physical model for known IAF assets.
 */
func (e *SRPEngine) computeBoxWing(satPos, sunPos [3]float64) (ax, ay, az float64) {
	return BoxWingAcceleration(satPos, sunPos, *e.BoxWing)
}

/**
 * computeECOM2 implements full 9-parameter CODE Orbit Model.
 *
 * D(u) = D0 + D2c·cos(2u) + D2s·sin(2u) + D4c·cos(4u) + D4s·sin(4u)
 * Y(u) = Y0
 * B(u) = B0 + Bc·cos(u) + Bs·sin(u)
 */
func (e *SRPEngine) computeECOM2(satPos, sunPos [3]float64, u float64, jd float64) (ax, ay, az float64) {
	params := e.ECOM2

	// Shadow check
	var nu float64 = 1.0
	if e.UseConicalShadow {
		shadow := ConicalShadowModel(satPos, sunPos)
		nu = shadow.Nu
		if nu <= 0 {
			return 0, 0, 0
		}
	}

	// Build Sun-oriented frame (D, Y, B)
	D, Y, B := buildECOMFrame(satPos, sunPos)

	// Distance scaling factor
	dx := sunPos[0] - satPos[0]
	dy := sunPos[1] - satPos[1]
	dz := sunPos[2] - satPos[2]
	d := math.Sqrt(dx*dx + dy*dy + dz*dz)
	scaleFactor := (AU_KM / d) * (AU_KM / d)

	// Periodic arguments
	cosU := math.Cos(u)
	sinU := math.Sin(u)
	cos2U := math.Cos(2 * u)
	sin2U := math.Sin(2 * u)
	cos4U := math.Cos(4 * u)
	sin4U := math.Sin(4 * u)

	// D-direction with periodic terms
	accelD := params.D0 +
		params.D2c*cos2U + params.D2s*sin2U +
		params.D4c*cos4U + params.D4s*sin4U

	// Y-direction (constant)
	accelY := params.Y0

	// B-direction with periodic terms
	accelB := params.B0 + params.Bc*cosU + params.Bs*sinU

	// Combine in ECI frame
	ax = (accelD*D[0] + accelY*Y[0] + accelB*B[0]) * scaleFactor * nu
	ay = (accelD*D[1] + accelY*Y[1] + accelB*B[1]) * scaleFactor * nu
	az = (accelD*D[2] + accelY*Y[2] + accelB*B[2]) * scaleFactor * nu

	return ax, ay, az
}

/**
 * computeECOM1 implements 5-parameter reduced model.
 *
 * D(u) = D0
 * Y(u) = Y0
 * B(u) = B0 + Bc·cos(u) + Bs·sin(u)
 */
func (e *SRPEngine) computeECOM1(satPos, sunPos [3]float64, u float64, jd float64) (ax, ay, az float64) {
	params := e.ECOM1

	// Shadow check
	var nu float64 = 1.0
	if e.UseConicalShadow {
		shadow := ConicalShadowModel(satPos, sunPos)
		nu = shadow.Nu
		if nu <= 0 {
			return 0, 0, 0
		}
	}

	// Build Sun-oriented frame
	D, Y, B := buildECOMFrame(satPos, sunPos)

	// Distance scaling
	dx := sunPos[0] - satPos[0]
	dy := sunPos[1] - satPos[1]
	dz := sunPos[2] - satPos[2]
	d := math.Sqrt(dx*dx + dy*dy + dz*dz)
	scaleFactor := (AU_KM / d) * (AU_KM / d)

	// Simplified model
	cosU := math.Cos(u)
	sinU := math.Sin(u)

	accelD := params.D0
	accelY := params.Y0
	accelB := params.B0 + params.Bc*cosU + params.Bs*sinU

	ax = (accelD*D[0] + accelY*Y[0] + accelB*B[0]) * scaleFactor * nu
	ay = (accelD*D[1] + accelY*Y[1] + accelB*B[1]) * scaleFactor * nu
	az = (accelD*D[2] + accelY*Y[1] + accelB*B[2]) * scaleFactor * nu

	return ax, ay, az
}

/**
 * computeHybrid combines Box-Wing (95%) + ECOM1 residuals (5%).
 *
 * This is the recommended Tier 1 model for IAF own assets.
 * Box-Wing handles known physics, ECOM1 absorbs:
 * - MLI degradation
 * - Thermal radiation
 * - Unmodeled attitude effects
 */
func (e *SRPEngine) computeHybrid(satPos, sunPos [3]float64, u float64, jd float64) (ax, ay, az float64) {
	if e.Hybrid == nil {
		return 0, 0, 0
	}

	// Box-Wing provides baseline physical acceleration
	bax, bay, baz := BoxWingAcceleration(satPos, sunPos, e.Hybrid.BoxWing)

	// ECOM1 provides residual correction
	eax, eay, eaz := e.computeECOM1Residual(satPos, sunPos, u, jd)

	// Combined: Physical + Empirical
	return bax + eax, bay + eay, baz + eaz
}

/**
 * computeECOM1Residual computes only the residual acceleration.
 * Used by Hybrid model - note: these are typically much smaller
 * than the full ECOM1 model since Box-Wing handles most physics.
 */
func (e *SRPEngine) computeECOM1Residual(satPos, sunPos [3]float64, u float64, jd float64) (ax, ay, az float64) {
	params := e.Hybrid.Residuals

	var nu float64 = 1.0
	if e.UseConicalShadow {
		shadow := ConicalShadowModel(satPos, sunPos)
		nu = shadow.Nu
	}

	D, Y, B := buildECOMFrame(satPos, sunPos)

	dx := sunPos[0] - satPos[0]
	dy := sunPos[1] - satPos[1]
	dz := sunPos[2] - satPos[2]
	d := math.Sqrt(dx*dx + dy*dy + dz*dz)
	scaleFactor := (AU_KM / d) * (AU_KM / d)

	cosU := math.Cos(u)
	sinU := math.Sin(u)

	// Residuals are typically 1-5% of total SRP
	accelD := params.D0
	accelY := params.Y0
	accelB := params.B0 + params.Bc*cosU + params.Bs*sinU

	ax = (accelD*D[0] + accelY*Y[0] + accelB*B[0]) * scaleFactor * nu
	ay = (accelD*D[1] + accelY*Y[1] + accelB*B[1]) * scaleFactor * nu
	az = (accelD*D[2] + accelY*Y[2] + accelB*B[2]) * scaleFactor * nu

	return ax, ay, az
}

/**
 * buildECOMFrame constructs the Sun-oriented reference frame (D, Y, B).
 *
 * D: Unit vector toward Sun
 * Y: Along solar panel rotation axis (approximate)
 * B: D × Y (completes right-handed system)
 */
func buildECOMFrame(satPos, sunPos [3]float64) (D, Y, B [3]float64) {
	// D: toward Sun
	dx := sunPos[0] - satPos[0]
	dy := sunPos[1] - satPos[1]
	dz := sunPos[2] - satPos[2]
	d := math.Sqrt(dx*dx + dy*dy + dz*dz)

	if d < 1e-12 {
		return [3]float64{1, 0, 0}, [3]float64{0, 1, 0}, [3]float64{0, 0, 1}
	}

	D = [3]float64{dx / d, dy / d, dz / d}

	// Y: Approximate solar panel axis
	// For GEO, this is roughly perpendicular to orbital plane
	// Use Earth's pole (simplified) and orthogonalize
	Y = [3]float64{0, 0, 1}

	// Make Y perpendicular to D
	dotDY := D[0]*Y[0] + D[1]*Y[1] + D[2]*Y[2]
	Y[0] -= dotDY * D[0]
	Y[1] -= dotDY * D[1]
	Y[2] -= dotDY * D[2]

	yNorm := math.Sqrt(Y[0]*Y[0] + Y[1]*Y[1] + Y[2]*Y[2])
	if yNorm > 1e-12 {
		Y[0] /= yNorm
		Y[1] /= yNorm
		Y[2] /= yNorm
	} else {
		// Degenerate case - pick arbitrary perpendicular
		if math.Abs(D[2]) < 0.9 {
			Y = [3]float64{-D[1], D[0], 0}
		} else {
			Y = [3]float64{1, 0, 0}
		}
		yNorm = math.Sqrt(Y[0]*Y[0] + Y[1]*Y[1] + Y[2]*Y[2])
		Y[0] /= yNorm
		Y[1] /= yNorm
		Y[2] /= yNorm
	}

	// B = D × Y (completes right-handed system)
	B = [3]float64{
		D[1]*Y[2] - D[2]*Y[1],
		D[2]*Y[0] - D[0]*Y[2],
		D[0]*Y[1] - D[1]*Y[0],
	}

	return D, Y, B
}

/**
 * ComputeArgumentOfLatitude calculates u for ECOM periodic terms.
 *
 * u = argument of perigee + true anomaly
 * For circular orbits, simplifies to u = Ω + ω + θ
 *
 * @param tle: TLE with orbital elements
 * @return: Argument of latitude in radians
 */
func ComputeArgumentOfLatitude(tle *physics.TLE) float64 {
	// u = RAAN + ArgOfPerigee + MeanAnomaly (approximation)
	// More precise would use true anomaly
	raan := tle.RAAN * math.Pi / 180.0
	argPerigee := tle.ArgOfPerigee * math.Pi / 180.0
	meanAnom := tle.MeanAnomaly * math.Pi / 180.0

	return raan + argPerigee + meanAnom
}

/**
 * DefaultECOM2Params returns typical GEO satellite parameters.
 * Used as initial guess before BLS fitting.
 */
func DefaultECOM2Params() ECOM2Params {
	return ECOM2Params{
		// D-direction: ~1 nm/s^2 baseline
		D0: 1.0e-9,

		// Periodic terms: small perturbations
		D2c: 0.2e-9,
		D2s: 0.1e-9,
		D4c: 0.05e-9,
		D4s: 0.02e-9,

		// Y-bias: small
		Y0: 0.1e-9,

		// B-direction: periodic from shape variations
		B0: 0.05e-9,
		Bc: 0.02e-9,
		Bs: 0.01e-9,
	}
}

/**
 * DefaultECOM1Params returns 5-parameter initial guess.
 */
func DefaultECOM1Params() ECOM1Params {
	return ECOM1Params{
		D0: 1.0e-9,
		Y0: 0.1e-9,
		B0: 0.05e-9,
		Bc: 0.02e-9,
		Bs: 0.01e-9,
	}
}
