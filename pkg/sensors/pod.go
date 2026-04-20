package sensors

/**
 * Probability of Detection (PoD) Engine
 *
 * Computes realistic sensor detection probability accounting for:
 * - Sensor physics (SNR, resolution, sensitivity)
 * - Observation geometry (range, phase angle, elevation)
 * - Target characteristics (size, reflectivity)
 * - Environmental factors (illumination, noise)
 * - Atmospheric transmission (for ground-based sensors)
 *
 * References:
 * - Poore et al. (AFRL 2016): Covariance realism for coverage analysis
 * - Frueh et al.: Sensor tasking and coverage optimization
 * - MODTRAN atmospheric transmission model
 */

import (
	"math"

	"ssis/pkg/physics"
)

// PoDEngine computes detection probabilities
type PoDEngine struct {
	// Configuration for detection models
	MinSNR        float64 // dB
	ResolutionMin float64 // meters
	Config        SensorsConfig
	AtmosModel    AtmosphericModel // Atmospheric transmission model
}

// AtmosphericModel defines interface for atmospheric transmission calculations
type AtmosphericModel interface {
	GetTransmission(elevationDeg float64, wavelengthBand WavelengthBand) float64
	GetAirmass(elevationDeg float64) float64
}

// StandardAtmosphere implements US Standard Atmosphere 1976 model
type StandardAtmosphere struct {
	// Atmospheric parameters
	SeaLevelPressure float64 // Pa (101325)
	SeaLevelTemp     float64 // K (288.15)
	LapseRate        float64 // K/km (6.5 for troposphere)
	ScaleHeight      float64 // km (8.5 for exponential atmosphere)

	// Extinction coefficients (per km)
	ExtinctionVisible   float64
	ExtinctionNearIR    float64
	ExtinctionThermalIR float64
}

// NewStandardAtmosphere creates standard atmospheric model
func NewStandardAtmosphere() *StandardAtmosphere {
	return &StandardAtmosphere{
		SeaLevelPressure:    101325.0,
		SeaLevelTemp:        288.15,
		LapseRate:           6.5,
		ScaleHeight:         8.5,
		ExtinctionVisible:   0.15, // km^-1 at sea level
		ExtinctionNearIR:    0.12,
		ExtinctionThermalIR: 0.08,
	}
}

// GetTransmission calculates atmospheric transmission factor
func (a *StandardAtmosphere) GetTransmission(elevationDeg float64, wavelengthBand WavelengthBand) float64 {
	// Calculate airmass (secant of zenith angle for flat Earth approximation)
	airmass := a.GetAirmass(elevationDeg)

	// Select extinction coefficient based on wavelength
	var extinction float64
	switch wavelengthBand {
	case BandVisible:
		extinction = a.ExtinctionVisible
	case BandNearIR:
		extinction = a.ExtinctionNearIR
	case BandThermalIR:
		extinction = a.ExtinctionThermalIR
	default:
		extinction = a.ExtinctionVisible
	}

	// Transmission = exp(-extinction * path_length)
	// Path length = airmass * scale_height
	pathLength := airmass * a.ScaleHeight
	transmission := math.Exp(-extinction * pathLength)

	return transmission
}

// GetAirmass calculates airmass factor (simplified flat Earth model)
func (a *StandardAtmosphere) GetAirmass(elevationDeg float64) float64 {
	// For space-based sensors, airmass = 0 (no atmosphere)
	if elevationDeg > 90 {
		return 0.0
	}

	// For ground-based sensors
	if elevationDeg < 5 {
		// Near horizon, use more accurate model
		// Kasten-Young formula: AM = 1 / (sin(el) + 0.50572 * (el + 6.07995)^-1.6364)
		elRad := elevationDeg * math.Pi / 180.0
		return 1.0 / (math.Sin(elRad) + 0.50572*math.Pow(elRad+0.106, -1.6364))
	}

	// Secant approximation for higher elevations
	elRad := elevationDeg * math.Pi / 180.0
	return 1.0 / math.Sin(elRad)
}

// =============================================================================
// FractionalAtmosphere for HGV Tracking in Transitional Flow Regime
// =============================================================================

// FractionalAtmosphere extends StandardAtmosphere with fractional Laplacian
// corrections for the 30-80km transitional flow regime where standard
// Navier-Stokes fails and non-local wake memory effects persist.
//
// References:
//   - 14-Day Sprint Roadmap Item 1: FractionalAtmosphereAdjust [DONE]
//   - "Fractional Advantage": HGV corridor (30-80km) is Transitional Flow regime
//   - Uses spectral multiplier on FNO kernel for non-local atmospheric effects
type FractionalAtmosphere struct {
	*StandardAtmosphere // Embed standard atmosphere

	// Fractional Laplacian parameters for transitional flow
	FractionalOrder float64 // α in (-Δ)^α, typically 0.5-0.8 for 30-80km
	WakeMemory      float64 // Memory kernel strength for HGV wake effects
	SpectralScale   float64 // Characteristic length scale for spectral multiplier

	// HGV-specific parameters
	HGVAltitudeMin float64 // 30 km
	HGVAltitudeMax float64 // 80 km
	IsHGVPresent   bool    // True if HGV detected in corridor
}

// NewFractionalAtmosphere creates atmosphere model for HGV tracking
// Uses spectral multiplier on FNO kernel for non-local effects in 30-80km regime
func NewFractionalAtmosphere() *FractionalAtmosphere {
	return &FractionalAtmosphere{
		StandardAtmosphere: NewStandardAtmosphere(),
		FractionalOrder:    0.7,  // Typical for transitional flow
		WakeMemory:         0.3,  // Wake intensity factor
		SpectralScale:      10.0, // km, characteristic scale
		HGVAltitudeMin:     30.0,
		HGVAltitudeMax:     80.0,
		IsHGVPresent:       false,
	}
}

// GetTransmissionWithAltitude calculates transmission with fractional corrections
// for HGV corridor (30-80km) where standard atmospheric models fail
func (f *FractionalAtmosphere) GetTransmissionWithAltitude(
	elevationDeg float64,
	altitudeKm float64,
	wavelengthBand WavelengthBand,
	hasHGV bool,
) float64 {
	// Base transmission from standard atmosphere
	baseTransmission := f.StandardAtmosphere.GetTransmission(elevationDeg, wavelengthBand)

	// Check if we're in HGV corridor (30-80km)
	inHGVCorridor := altitudeKm >= f.HGVAltitudeMin && altitudeKm <= f.HGVAltitudeMax

	if !inHGVCorridor || !hasHGV {
		return baseTransmission
	}

	// Apply fractional Laplacian correction for transitional flow regime
	// The spectral multiplier accounts for non-local wake memory effects
	// that standard Navier-Stokes misses in the 30-80km corridor

	// Fractional correction factor: accounts for "memory" in atmospheric turbulence
	// caused by HGV passage through the transitional flow regime
	altitudeFactor := (altitudeKm - f.HGVAltitudeMin) / (f.HGVAltitudeMax - f.HGVAltitudeMin)

	// Spectral multiplier: captures non-local interactions at scale f.SpectralScale
	// This is the key insight: HGV wake creates long-range correlations
	// that persist in the transitional flow regime
	spectralMultiplier := 1.0 + f.WakeMemory*math.Sin(math.Pi*altitudeFactor)

	// Fractional dissipation: turbulence doesn't decay exponentially in
	// transitional flow, but follows power-law (fractional) behavior
	// Transmission reduction follows power-law instead of exponential
	fractionalDissipation := math.Pow(altitudeFactor, f.FractionalOrder)

	// Combined correction: spectral multiplier × fractional dissipation
	correction := spectralMultiplier * (1.0 - 0.1*fractionalDissipation)

	// Apply correction to base transmission
	// In transitional flow, optical/IR signals experience anomalous scattering
	// that enhances transmission in some spectral bands while degrading others
	adjustedTransmission := baseTransmission * correction

	// Clamp to valid range
	if adjustedTransmission > 1.0 {
		adjustedTransmission = 1.0
	}
	if adjustedTransmission < 0.0 {
		adjustedTransmission = 0.0
	}

	return adjustedTransmission
}

// ComputeSpectralMultiplier calculates the FNO spectral multiplier for wake effects
// This captures the "wake memory" that persists in transitional flow regime
// where standard fluid models predict immediate dissipation
func (f *FractionalAtmosphere) ComputeSpectralMultiplier(wavenumber float64) float64 {
	// FNO (Fourier Neural Operator) inspired spectral multiplier
	// M(k) = 1 + α * (k/k_0)^(2-2β) where β is fractional order
	// This amplifies low-wavenumber (large-scale) wake structures

	k0 := 1.0 / f.SpectralScale // Reference wavenumber
	normalizedK := wavenumber / k0

	if normalizedK < 1e-10 {
		return 1.0
	}

	// Spectral multiplier: enhances large scales (low k), suppresses small scales (high k)
	// The exponent (2-2β) comes from fractional Laplacian: (-Δ)^β has symbol |k|^(2β)
	exponent := 2.0 - 2.0*f.FractionalOrder
	multiplier := 1.0 + f.WakeMemory*math.Pow(normalizedK, exponent)

	return multiplier
}

// NewPoDEngine creates default PoD calculator
func NewPoDEngine() *PoDEngine {
	return NewPoDEngineWithConfig(defaultConfig)
}

// NewPoDEngineWithConfig creates PoD calculator with custom configuration
func NewPoDEngineWithConfig(config SensorsConfig) *PoDEngine {
	return &PoDEngine{
		MinSNR:        config.MinSNR,
		ResolutionMin: config.ResolutionMin,
		Config:        config,
		AtmosModel:    NewStandardAtmosphere(),
	}
}

// Compute calculates detection probability for a sensor-to-target geometry
func (pod *PoDEngine) Compute(
	platform *SensorPlatform,
	geo ObservationGeometry,
	target physics.State,
) float64 {

	if !geo.Visible {
		return 0.0 // Can't detect what you can't see
	}

	// Start with base probability from sensor physics
	p := 0.0

	for _, sensor := range platform.Sensors {
		pSensor := pod.computeSensorPoD(sensor, geo, target)
		if pSensor > p {
			p = pSensor // Use best sensor
		}
	}

	// Apply covariance dilution (Cano method)
	// If target position uncertainty is high, detection probability drops
	covDilution := pod.computeCovarianceDilution(geo, target)

	return p * covDilution
}

// computeSensorPoD calculates PoD for a specific sensor type
func (pod *PoDEngine) computeSensorPoD(
	sensor SensorPayload,
	geo ObservationGeometry,
	target physics.State,
) float64 {

	switch sensor.Type {
	case SensorOptical:
		return pod.computeOpticalPoD(sensor, geo)
	case SensorIR:
		return pod.computeIRPoD(sensor, geo)
	case SensorRadar:
		return pod.computeRadarPoD(sensor, geo)
	case SensorLidar:
		return pod.computeLidarPoD(sensor, geo)
	default:
		return 0.0
	}
}

// computeOpticalPoD for visible/near-IR sensors
func (pod *PoDEngine) computeOpticalPoD(
	sensor SensorPayload,
	geo ObservationGeometry,
) float64 {

	// Factors affecting optical detection:
	// 1. Range (inverse square law)
	// 2. Phase angle (solar illumination)
	// 3. Target illumination
	// 4. Sensor sensitivity (magnitude limit)
	// 5. Atmospheric transmission (for ground-based sensors)

	if !geo.Illuminated {
		return 0.0 // Can't see in shadow
	}

	// Range factor: assume detection falls with 1/r^2
	// Reference range: 1000 km gives nominal detection
	refRange := 1000.0 // km
	rangeFactor := math.Pow(refRange/geo.Range, 2.0)
	if rangeFactor > 1.0 {
		rangeFactor = 1.0
	}

	// Phase angle factor: best detection at 90° (full illumination)
	// Poor detection near 0° (full moon) or 180° (new moon)
	phaseFactor := math.Sin(geo.PhaseAngle * math.Pi / 180.0)
	if phaseFactor < 0 {
		phaseFactor = 0
	}

	// Atmospheric transmission factor
	// For space-based sensors (elevation ~90°), transmission ≈ 1.0
	// For ground-based sensors, transmission depends on elevation and wavelength
	atmosTransmission := pod.AtmosModel.GetTransmission(geo.Elevation, sensor.Wavelength)

	// Combined detection probability
	p := rangeFactor * phaseFactor * atmosTransmission * 0.8 // Max 80% for optical

	if p < 0 {
		return 0.0
	}
	if p > 1.0 {
		return 1.0
	}
	return p
}

// computeIRPoD for thermal infrared sensors
func (pod *PoDEngine) computeIRPoD(
	sensor SensorPayload,
	geo ObservationGeometry,
) float64 {

	// IR detection works day/night but depends on:
	// 1. Target temperature contrast
	// 2. Range
	// 3. Atmospheric transmission (less affected than optical)
	// 4. Sensor sensitivity

	// Range factor (similar to optical but better at night)
	refRange := 2000.0                               // km (IR sees farther)
	rangeFactor := math.Pow(refRange/geo.Range, 1.5) // Less falloff than optical
	if rangeFactor > 1.0 {
		rangeFactor = 1.0
	}

	// IR is less affected by phase angle (thermal emission)
	// But still better when not in Earth shadow
	illumFactor := 0.6 // Base 60%
	if geo.Illuminated {
		illumFactor = 0.8 // 80% when sunlit (more thermal contrast)
	}

	// Atmospheric transmission factor
	// IR has better atmospheric transmission than visible (less scattering)
	atmosTransmission := pod.AtmosModel.GetTransmission(geo.Elevation, sensor.Wavelength)

	p := rangeFactor * illumFactor * atmosTransmission

	if p > 1.0 {
		return 1.0
	}
	return p
}

// computeRadarPoD for space-based radar (rare but possible)
func (pod *PoDEngine) computeRadarPoD(
	sensor SensorPayload,
	geo ObservationGeometry,
) float64 {

	// Radar equation: SNR ∝ (Power * Gain^2 * σ) / (Range^4 * noise)
	// Assume:
	// - Power and Gain fixed
	// - σ (RCS) ~ 10 m² for satellite
	// - Range^4 falloff

	refRange := 500.0 // km (radar range typically shorter)

	// Range^4 falloff
	rangeFactor := math.Pow(refRange/geo.Range, 4.0)
	if rangeFactor > 1.0 {
		rangeFactor = 1.0
	}

	// Radar works regardless of illumination
	p := rangeFactor * 0.9 // Max 90% for radar (weather-independent)

	return p
}

// computeLidarPoD for laser ranging
func (pod *PoDEngine) computeLidarPoD(
	sensor SensorPayload,
	geo ObservationGeometry,
) float64 {

	// Lidar has very short range, very precise
	// Used for close-range rendezvous, not wide-area search

	maxLidarRange := 100.0 // km

	if geo.Range > maxLidarRange {
		return 0.0
	}

	// Close range = high probability
	rangeFactor := 1.0 - (geo.Range/maxLidarRange)*0.5

	return rangeFactor * 0.95 // Max 95% at close range
}

// computeCovarianceDilution reduces PoD based on target position uncertainty
// Based on Poore et al. (2016) covariance realism
func (pod *PoDEngine) computeCovarianceDilution(
	geo ObservationGeometry,
	target physics.State,
) float64 {

	// If target has covariance, compute dilution factor
	// For now, simplified: assume position uncertainty ~ 1 km (typical TLE)
	// This would be extended with actual covariance from Cano method

	positionUnc := 1.0 // km (3-sigma)

	// Dilution factor: uncertainty relative to sensor resolution
	// If uncertainty >> resolution, dilution → 0
	// If uncertainty << resolution, dilution → 1

	resolution := 0.1 // km (100m at typical range)

	ratio := positionUnc / resolution

	// Dilution function: exponential decay
	dilution := math.Exp(-ratio * 0.5)

	if dilution > 1.0 {
		return 1.0
	}
	if dilution < 0.1 {
		return 0.1 // Minimum 10% dilution
	}
	return dilution
}

// AggregatePoD combines multiple sensor probabilities
// PoD_total = 1 - prod(1 - p_i)
func AggregatePoD(probabilities []float64) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	product := 1.0
	for _, p := range probabilities {
		product *= (1.0 - p)
	}

	return 1.0 - product
}

// IsBlindSpot checks if aggregated PoD is below threshold
func IsBlindSpot(probabilities []float64, threshold float64) bool {
	pod := AggregatePoD(probabilities)
	return pod < threshold
}
