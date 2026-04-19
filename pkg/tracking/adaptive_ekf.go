// Package tracking provides Adaptive Extended Kalman Filter
// Detects non-ballistic events (maneuvers) by monitoring innovation residuals
package tracking

import (
	"math"

	"ssis/pkg/physics"
)

// AdaptiveEKF extends standard EKF with maneuver detection capability
type AdaptiveEKF struct {
	*ExtendedKalmanFilter

	// Adaptive tuning parameters
	BaseProcessNoise    [28]float64 // Baseline Q matrix
	ManeuverScaleFactor float64     // Multiplier during maneuvers
	ManeuverThreshold   float64     // σ threshold for maneuver detection

	// Maneuver detection state
	IsManeuvering        bool
	ManeuverStartTime    float64 // Julian Date
	ConsecutiveAnomalies int     // Count of anomalous updates

	// Innovation history for statistical analysis
	InnovationWindow [10][3]float64 // Last 10 innovations
	InnovationIdx    int
}

// NewAdaptiveEKF creates EKF with adaptive maneuver detection
func NewAdaptiveEKF(
	satelliteID int,
	initialState physics.State,
	initialBstar float64,
	maneuverThreshold float64,
) *AdaptiveEKF {
	baseEKF := NewEKF(satelliteID, initialState, initialBstar)

	// Save baseline process noise
	var baseQ [28]float64
	copy(baseQ[:], baseEKF.P[:])

	return &AdaptiveEKF{
		ExtendedKalmanFilter: baseEKF,
		BaseProcessNoise:     baseQ,
		ManeuverScaleFactor:  10.0,              // Increase Q by 10x during maneuvers
		ManeuverThreshold:    maneuverThreshold, // Typically 3.0-5.0 σ
		IsManeuvering:        false,
		ManeuverStartTime:    0,
		ConsecutiveAnomalies: 0,
	}
}

// UpdateAdaptive performs EKF update with maneuver detection
// Returns: (innovation, mahalanobis, is_maneuver_detected)
func (ekf *AdaptiveEKF) UpdateAdaptive(measurement [3]float64) ([3]float64, float64, bool) {
	// Get predicted measurement
	zHat := [3]float64{ekf.x[0], ekf.x[1], ekf.x[2]}

	// Calculate innovation
	var innovation [3]float64
	for i := 0; i < 3; i++ {
		innovation[i] = measurement[i] - zHat[i]
	}

	// Calculate innovation covariance S
	S := ekf.computeInnovationCovariance()

	// Calculate Mahalanobis distance
	mahalanobis := ekf.computeMahalanobis(innovation, S)

	// Maneuver detection logic
	maneuverDetected := ekf.detectManeuver(mahalanobis)

	// Adaptive process noise adjustment
	if maneuverDetected {
		ekf.IsManeuvering = true
		ekf.ConsecutiveAnomalies++

		// Increase process noise to "follow" the maneuver
		ekf.scaleProcessNoise(ekf.ManeuverScaleFactor)

		// Record maneuver start time
		if ekf.ConsecutiveAnomalies == 3 {
			ekf.ManeuverStartTime = ekf.LastUpdate
		}
	} else {
		// Decay back to baseline if consistent observations
		if ekf.ConsecutiveAnomalies > 0 {
			ekf.ConsecutiveAnomalies--
		}

		if ekf.ConsecutiveAnomalies == 0 && ekf.IsManeuvering {
			ekf.IsManeuvering = false
			ekf.resetProcessNoise()
		}
	}

	// Standard EKF update with (possibly adapted) Q
	K := ekf.computeKalmanGain(S)

	// State update
	for i := 0; i < 7; i++ {
		for j := 0; j < 3; j++ {
			kIdx := i*3 + j
			if kIdx < len(K) {
				ekf.x[i] += K[kIdx] * innovation[j]
			}
		}
	}

	// Covariance update
	ekf.updateCovarianceAdaptive(K, maneuverDetected)

	// Store innovation for history
	ekf.storeInnovation(innovation)

	return innovation, mahalanobis, maneuverDetected
}

// detectManeuver determines if residual indicates non-ballistic behavior
func (ekf *AdaptiveEKF) detectManeuver(mahalanobis float64) bool {
	// Threshold-based detection
	if mahalanobis > ekf.ManeuverThreshold {
		return true
	}

	// Additional check: look for pattern in innovation window
	if ekf.InnovationIdx >= 3 {
		// Check if recent innovations show systematic bias
		var avgInnovation [3]float64
		count := 0

		for i := 0; i < 3; i++ {
			idx := (ekf.InnovationIdx - 1 - i + 10) % 10
			for j := 0; j < 3; j++ {
				avgInnovation[j] += ekf.InnovationWindow[idx][j]
			}
			count++
		}

		for j := 0; j < 3; j++ {
			avgInnovation[j] /= float64(count)
		}

		// If average innovation is significant, likely maneuver
		avgMag := math.Sqrt(
			avgInnovation[0]*avgInnovation[0] +
				avgInnovation[1]*avgInnovation[1] +
				avgInnovation[2]*avgInnovation[2],
		)

		if avgMag > ekf.ManeuverThreshold*0.5 {
			return true
		}
	}

	return false
}

// scaleProcessNoise increases Q during maneuvers
func (ekf *AdaptiveEKF) scaleProcessNoise(scale float64) {
	for i := 0; i < 7; i++ {
		idx := i*7 - i*(i-1)/2 // Diagonal index
		if idx < len(ekf.P) {
			ekf.P[idx] = ekf.BaseProcessNoise[idx] * scale
		}
	}
}

// resetProcessNoise restores Q to baseline
func (ekf *AdaptiveEKF) resetProcessNoise() {
	for i := range ekf.P {
		ekf.P[i] = ekf.BaseProcessNoise[i]
	}
}

// storeInnovation adds to rolling window
func (ekf *AdaptiveEKF) storeInnovation(innovation [3]float64) {
	ekf.InnovationWindow[ekf.InnovationIdx] = innovation
	ekf.InnovationIdx = (ekf.InnovationIdx + 1) % 10
}

// updateCovarianceAdaptive updates P with optional maneuver compensation
func (ekf *AdaptiveEKF) updateCovarianceAdaptive(K [21]float64, isManeuver bool) {
	if isManeuver {
		// During maneuvers, use "fading memory" - reduce confidence in older estimates
		fadingFactor := 0.9
		for i := 0; i < 7; i++ {
			idx := i*7 - i*(i-1)/2
			if idx < len(ekf.P) {
				ekf.P[idx] *= fadingFactor
			}
		}
	} else {
		// Standard update
		ekf.updateCovariance(K)
	}
}

// GetManeuverCharacteristics returns delta-V estimate and duration
func (ekf *AdaptiveEKF) GetManeuverCharacteristics() (deltaV float64, duration float64, active bool) {
	if !ekf.IsManeuvering {
		return 0, 0, false
	}

	// Estimate delta-V from state deviation
	// Compare current velocity with expected from last ballistic state
	// currentVel := [3]float64{ekf.x[3], ekf.x[4], ekf.x[5]}  // Reserved for future use

	// Simplified delta-V calculation (would need ballistic reference)
	// For now, use magnitude of recent innovations as proxy
	var maxInnovation float64
	for i := 0; i < 10; i++ {
		inv := ekf.InnovationWindow[i]
		mag := math.Sqrt(inv[0]*inv[0] + inv[1]*inv[1] + inv[2]*inv[2])
		if mag > maxInnovation {
			maxInnovation = mag
		}
	}

	// Rough delta-V estimate (km/s)
	deltaV = maxInnovation / 1000.0 // Convert m to km

	// Duration in minutes
	if ekf.ManeuverStartTime > 0 {
		duration = (ekf.LastUpdate - ekf.ManeuverStartTime) * 1440.0 // days to minutes
	}

	return deltaV, duration, true
}

// InspectorPatternCheck detects co-orbital inspection behavior
// Returns true if RSO shows station-keeping behavior near target
func (ekf *AdaptiveEKF) InspectorPatternCheck(targetPos [3]float64) bool {
	if !ekf.IsManeuvering {
		return false
	}

	// Calculate relative position
	relativePos := [3]float64{
		ekf.x[0] - targetPos[0],
		ekf.x[1] - targetPos[1],
		ekf.x[2] - targetPos[2],
	}

	// Distance to target
	distance := math.Sqrt(
		relativePos[0]*relativePos[0] +
			relativePos[1]*relativePos[1] +
			relativePos[2]*relativePos[2],
	)

	// Inspector pattern: Within 1km, maintaining relative position
	if distance < 1.0 { // 1km
		// Check if relative velocity is near-zero (station-keeping)
		relativeVel := [3]float64{
			ekf.x[3] - 0, // Would need target velocity
			ekf.x[4] - 0,
			ekf.x[5] - 0,
		}

		relativeVelMag := math.Sqrt(
			relativeVel[0]*relativeVel[0] +
				relativeVel[1]*relativeVel[1] +
				relativeVel[2]*relativeVel[2],
		)

		// Station-keeping: relative velocity < 10 m/s
		if relativeVelMag < 0.01 { // 10 m/s in km/s
			return true
		}
	}

	return false
}

// ThreatAssessment combines maneuver + material analysis
func (ekf *AdaptiveEKF) ThreatAssessment(
	materialID string,
	materialThreatLevel int,
	proximityKm float64,
) (threatLevel int, description string) {
	baseThreat := materialThreatLevel

	// Maneuvering factor
	if ekf.IsManeuvering {
		baseThreat += 3

		// Inspector pattern bonus
		if ekf.ConsecutiveAnomalies > 5 {
			baseThreat += 2
		}
	}

	// Proximity factor
	if proximityKm < 5 {
		baseThreat += 3
	} else if proximityKm < 50 {
		baseThreat += 2
	} else if proximityKm < 100 {
		baseThreat += 1
	}

	threatLevel = min(10, baseThreat)

	// Generate description
	if threatLevel >= 8 {
		description = "CRITICAL: Likely inspector/ASAT - immediate action required"
	} else if threatLevel >= 6 {
		description = "HIGH: Suspicious maneuvering pattern detected"
	} else if threatLevel >= 4 {
		description = "ELEVATED: Monitor for approach maneuvers"
	} else {
		description = "LOW: Standard orbital behavior"
	}

	return threatLevel, description
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
