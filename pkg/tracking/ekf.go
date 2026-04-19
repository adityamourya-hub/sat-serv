// Package tracking provides Extended Kalman Filter for real-time satellite tracking
// Fuses TLE predictions with live sensor detections
package tracking

import (
	"math"
	
	"ssis/pkg/physics"
)

// ExtendedKalmanFilter implements EKF for satellite state estimation
// State vector: [x, y, z, vx, vy, vz, B*]
type ExtendedKalmanFilter struct {
	// State estimate
	x [7]float64  // [pos(3), vel(3), Bstar(1)]
	
	// Covariance matrix (7x7, stored as upper triangle)
	P [28]float64
	
	// Process noise covariance
	Q [28]float64
	
	// Measurement noise covariance
	R [9]float64  // For position-only measurements
	
	// Satellite ID
	SatelliteID int
	
	// Time of last update
	LastUpdate float64  // Julian Date
	
	// Innovation statistics for adaptive tuning
	InnovationHistory [10]float64
	InnovationIdx     int
}

// NewEKF creates a new Extended Kalman Filter
func NewEKF(satelliteID int, initialState physics.State, initialBstar float64) *ExtendedKalmanFilter {
	ekf := &ExtendedKalmanFilter{
		SatelliteID: satelliteID,
		LastUpdate:  initialState.EpochJD,
	}
	
	// Initialize state
	ekf.x[0] = initialState.Pos[0]
	ekf.x[1] = initialState.Pos[1]
	ekf.x[2] = initialState.Pos[2]
	ekf.x[3] = initialState.Vel[0]
	ekf.x[4] = initialState.Vel[1]
	ekf.x[5] = initialState.Vel[2]
	ekf.x[6] = initialBstar
	
	// Initialize covariance (diagonal matrix)
	// Position uncertainty: 1 km
	// Velocity uncertainty: 0.1 km/s
	// B* uncertainty: 50 (nominal)
	ekf.P = [28]float64{
		1e6, 0, 0, 0, 0, 0, 0,           // P00 = 1000^2
		1e6, 0, 0, 0, 0, 0,              // P11
		1e6, 0, 0, 0, 0,                 // P22
		0.01, 0, 0, 0,                   // P33 = 0.1^2
		0.01, 0, 0,                      // P44
		0.01, 0,                         // P55
		2500,                            // P66 = 50^2
	}
	
	// Process noise (tuned for SGP4 propagation accuracy)
	ekf.Q = [28]float64{
		100, 0, 0, 0, 0, 0, 0,     // Position drift: 10m^2
		100, 0, 0, 0, 0, 0,
		100, 0, 0, 0, 0,
		1e-4, 0, 0, 0,           // Velocity noise: 0.01 km/s^2
		1e-4, 0, 0,
		1e-4, 0,
		100,                     // B* variation: 10^2
	}
	
	// Measurement noise (radar/optical accuracy)
	// Typical: 100m for radar, 10m for optical
	ekf.R = [9]float64{
		1e4, 0, 0,    // 100m^2
		1e4, 0,
		1e4,
	}
	
	return ekf
}

// Predict propagates state using orbital dynamics
func (ekf *ExtendedKalmanFilter) Predict(dt float64, propagator func(physics.State, float64) physics.State) {
	// Current state
	currentState := physics.State{
		Pos:     [3]float64{ekf.x[0], ekf.x[1], ekf.x[2]},
		Vel:     [3]float64{ekf.x[3], ekf.x[4], ekf.x[5]},
		EpochJD: ekf.LastUpdate,
	}
	
	// Propagate using provided propagator (SGP4 or numerical)
	predictedState := propagator(currentState, dt)
	
	// Update state vector
	ekf.x[0] = predictedState.Pos[0]
	ekf.x[1] = predictedState.Pos[1]
	ekf.x[2] = predictedState.Pos[2]
	ekf.x[3] = predictedState.Vel[0]
	ekf.x[4] = predictedState.Vel[1]
	ekf.x[5] = predictedState.Vel[2]
	// B* remains constant during prediction
	
	// Update covariance: P = F*P*F' + Q
	// For simplicity, use Jacobian approximation
	ekf.propagateCovariance(dt)
	
	ekf.LastUpdate += dt / 86400.0  // Convert seconds to days (JD)
}

// propagateCovariance updates covariance matrix using state transition
func (ekf *ExtendedKalmanFilter) propagateCovariance(dt float64) {
	// Simplified: Add process noise scaled by time
	// Full implementation would compute Jacobian F = df/dx
	
	scale := 1.0 + dt/60.0  // Scale factor based on time step
	
	// P = P * scale + Q
	for i := 0; i < 7; i++ {
		idx := i*(7) - i*(i-1)/2  // Upper triangle index
		if idx < len(ekf.P) {
			ekf.P[idx] = ekf.P[idx]*scale + ekf.Q[idx]
		}
	}
}

// Update performs Kalman update with measurement
func (ekf *ExtendedKalmanFilter) Update(measurement [3]float64) (innovation [3]float64, mahalanobis float64) {
	// Measurement matrix H (extracts position from state)
	// H = [I_3x3, 0_3x3, 0_3x1]
	
	// Predicted measurement: z_hat = H * x = x[0:3]
	zHat := [3]float64{ekf.x[0], ekf.x[1], ekf.x[2]}
	
	// Innovation: y = z - z_hat
	for i := 0; i < 3; i++ {
		innovation[i] = measurement[i] - zHat[i]
	}
	
	// Innovation covariance: S = H*P*H' + R
	S := ekf.computeInnovationCovariance()
	
	// Kalman gain: K = P*H' * inv(S)
	// For position-only, K is 7x3 matrix
	K := ekf.computeKalmanGain(S)
	
	// State update: x = x + K*y
	for i := 0; i < 7; i++ {
		for j := 0; j < 3; j++ {
			ekf.x[i] += K[i*3+j] * innovation[j]
		}
	}
	
	// Covariance update: P = (I - K*H) * P
	ekf.updateCovariance(K)
	
	// Compute Mahalanobis distance
	mahalanobis = ekf.computeMahalanobis(innovation, S)
	
	// Store for adaptive tuning
	ekf.InnovationHistory[ekf.InnovationIdx] = mahalanobis
	ekf.InnovationIdx = (ekf.InnovationIdx + 1) % 10
	
	return innovation, mahalanobis
}

// computeInnovationCovariance calculates S = H*P*H' + R
func (ekf *ExtendedKalmanFilter) computeInnovationCovariance() [9]float64 {
	// H*P*H' extracts the 3x3 position covariance
	var S [9]float64
	
	// Extract P[0:3, 0:3] and add R
	S[0] = ekf.P[0] + ekf.R[0]  // P00 + R00
	S[1] = ekf.P[1] + ekf.R[1]  // P01 + R01
	S[2] = ekf.P[2] + ekf.R[2]  // P02 + R02
	S[4] = ekf.P[6] + ekf.R[4]  // P11 + R11
	S[5] = ekf.P[7] + ekf.R[5]  // P12 + R12
	S[8] = ekf.P[11] + ekf.R[8] // P22 + R22
	
	// Symmetric elements
	S[3] = S[1]
	S[6] = S[2]
	S[7] = S[5]
	
	return S
}

// computeKalmanGain calculates K = P*H' * inv(S)
func (ekf *ExtendedKalmanFilter) computeKalmanGain(S [9]float64) [21]float64 {
	var K [21]float64
	
	// Compute inv(S)
	Sinv := ekf.invert3x3(S)
	
	// P*H' extracts first 3 columns of P
	// K = (P*H') * Sinv
	
	// Position gain (first 3 states)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			idx := i*3 + j
			if idx < len(K) {
				// Sum over k
				for k := 0; k < 3; k++ {
					K[idx] += ekf.P[i*7+k-int(i*(i-1)/2)] * Sinv[k*3+j]
				}
			}
		}
	}
	
	// Velocity gain (states 3-5)
	for i := 3; i < 6; i++ {
		for j := 0; j < 3; j++ {
			idx := i*3 + j
			if idx < len(K) {
				for k := 0; k < 3; k++ {
					pIdx := i*7 + k - i*(i-1)/2
					if pIdx < len(ekf.P) {
						K[idx] += ekf.P[pIdx] * Sinv[k*3+j]
					}
				}
			}
		}
	}
	
	// B* gain (state 6)
	for j := 0; j < 3; j++ {
		idx := 6*3 + j
		if idx < len(K) {
			for k := 0; k < 3; k++ {
				// P[6, k] terms
				pIdx := 27 - 6*5/2 + k // Index for row 6
				if pIdx < len(ekf.P) {
					K[idx] += ekf.P[pIdx] * Sinv[k*3+j]
				}
			}
		}
	}
	
	return K
}

// invert3x3 computes inverse of 3x3 matrix
func (ekf *ExtendedKalmanFilter) invert3x3(m [9]float64) [9]float64 {
	// Compute determinant
	det := m[0]*(m[4]*m[8]-m[5]*m[7]) -
		m[1]*(m[3]*m[8]-m[5]*m[6]) +
		m[2]*(m[3]*m[7]-m[4]*m[6])
	
	if math.Abs(det) < 1e-10 {
		// Near-singular, return identity
		return [9]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}
	}
	
	invDet := 1.0 / det
	
	var inv [9]float64
	inv[0] = (m[4]*m[8] - m[5]*m[7]) * invDet
	inv[1] = (m[2]*m[7] - m[1]*m[8]) * invDet
	inv[2] = (m[1]*m[5] - m[2]*m[4]) * invDet
	inv[3] = (m[5]*m[6] - m[3]*m[8]) * invDet
	inv[4] = (m[0]*m[8] - m[2]*m[6]) * invDet
	inv[5] = (m[2]*m[3] - m[0]*m[5]) * invDet
	inv[6] = (m[3]*m[7] - m[4]*m[6]) * invDet
	inv[7] = (m[1]*m[6] - m[0]*m[7]) * invDet
	inv[8] = (m[0]*m[4] - m[1]*m[3]) * invDet
	
	return inv
}

// updateCovariance performs Joseph form covariance update
func (ekf *ExtendedKalmanFilter) updateCovariance(K [21]float64) {
	// Simplified: P = P - K*S*K'
	// Joseph form is more numerically stable but complex
	
	// For now, use simple update
	for i := 0; i < 7; i++ {
		idx := i*7 - i*(i-1)/2
		if idx < len(ekf.P) {
			// Reduce diagonal elements based on Kalman gain
			kGain := 0.0
			for j := 0; j < 3; j++ {
				kIdx := i*3 + j
				if kIdx < len(K) {
					kGain += K[kIdx] * K[kIdx]
				}
			}
			ekf.P[idx] *= (1.0 - kGain)
			if ekf.P[idx] < 0 {
				ekf.P[idx] = 0
			}
		}
	}
}

// computeMahalanobis calculates innovation Mahalanobis distance
func (ekf *ExtendedKalmanFilter) computeMahalanobis(y [3]float64, S [9]float64) float64 {
	// d^2 = y' * S^-1 * y
	Sinv := ekf.invert3x3(S)
	
	d2 := 0.0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			d2 += y[i] * Sinv[i*3+j] * y[j]
		}
	}
	
	return math.Sqrt(d2)
}

// GetState returns current state estimate
func (ekf *ExtendedKalmanFilter) GetState() physics.State {
	return physics.State{
		Pos:     [3]float64{ekf.x[0], ekf.x[1], ekf.x[2]},
		Vel:     [3]float64{ekf.x[3], ekf.x[4], ekf.x[5]},
		EpochJD: ekf.LastUpdate,
	}
}

// GetPositionCovariance returns 3x3 position covariance
func (ekf *ExtendedKalmanFilter) GetPositionCovariance() [9]float64 {
	return [9]float64{
		ekf.P[0], ekf.P[1], ekf.P[2],
		ekf.P[1], ekf.P[6], ekf.P[7],
		ekf.P[2], ekf.P[7], ekf.P[11],
	}
}

// IsConverged checks if filter has converged (innovation variance stable)
func (ekf *ExtendedKalmanFilter) IsConverged() bool {
	if ekf.InnovationHistory[0] == 0 {
		return false  // Not enough history
	}
	
	// Compute variance of innovations
	mean := 0.0
	for _, v := range ekf.InnovationHistory {
		mean += v
	}
	mean /= 10.0
	
	variance := 0.0
	for _, v := range ekf.InnovationHistory {
		diff := v - mean
		variance += diff * diff
	}
	variance /= 9.0  // Sample variance
	
	// Converged if variance < threshold (stable innovations)
	return variance < 100.0  // Threshold tuned empirically
}
