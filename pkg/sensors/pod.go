package sensors

/**
 * Probability of Detection (PoD) Engine
 *
 * Computes realistic sensor detection probability accounting for:
 * - Sensor physics (SNR, resolution, sensitivity)
 * - Observation geometry (range, phase angle, elevation)
 * - Target characteristics (size, reflectivity)
 * - Environmental factors (illumination, noise)
 *
 * References:
 * - Poore et al. (AFRL 2016): Covariance realism for coverage analysis
 * - Frueh et al.: Sensor tasking and coverage optimization
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
}

// NewPoDEngine creates default PoD calculator
func NewPoDEngine() *PoDEngine {
	return &PoDEngine{
		MinSNR:        10.0,  // 10 dB SNR threshold
		ResolutionMin: 1.0,   // 1 meter resolution requirement
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
	
	// Combined detection probability (simplified model)
	p := rangeFactor * phaseFactor * 0.8 // Max 80% for optical
	
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
	// 3. Atmospheric transmission (neglected for space-based)
	// 4. Sensor sensitivity
	
	// Range factor (similar to optical but better at night)
	refRange := 2000.0 // km (IR sees farther)
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
	
	p := rangeFactor * illumFactor
	
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
