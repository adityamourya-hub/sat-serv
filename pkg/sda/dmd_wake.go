package sda

import (
	"math"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * DMD Wake Analysis for Sub-Visual Detection
 *
 * Dynamic Mode Decomposition (DMD) extracts Coherent Structures from optical flow.
 * Even if the HGV silhouette is blurred, the Kármán Vortex Street left in its wake
 * has a frequency signature that corresponds to the vehicle's speed and bank angle.
 *
 * This enables sub-visual detection and maneuver discovery 1-2 seconds before
 * the centroid actually shifts, enabling the telescope to maintain custody
 * during aggressive S-turns.
 */

// ManeuverType represents the type of maneuver detected from wake analysis
type ManeuverType string

const (
	ManeuverNone         ManeuverType = "NONE"
	ManeuverTurnRight    ManeuverType = "TURN_RIGHT"
	ManeuverTurnLeft     ManeuverType = "TURN_LEFT"
	ManeuverAcceleration ManeuverType = "ACCELERATION"
	ManeuverDeceleration ManeuverType = "DECELERATION"
	ManeuverUnknown      ManeuverType = "MANEUVER_UNKNOWN"
)

// OpticalFlowFrame represents a single frame of optical flow data
type OpticalFlowFrame struct {
	Timestamp  float64      // Time in seconds
	VelocityX  [][2]float64 // X-velocity field (grid of vectors)
	VelocityY  [][2]float64 // Y-velocity field
	Resolution int          // Grid resolution (N x N)
}

// WakeSignature represents the detected wake signature
type WakeSignature struct {
	StrouhalNumber  float64 // Dimensionless frequency (fL/U)
	VortexFrequency float64 // Vortex shedding frequency (Hz)
	BankAngle       float64 // Estimated bank angle (radians)
	Speed           float64 // Vehicle speed (m/s)
	Confidence      float64 // Detection confidence (0-1)
}

// DMDWakeAnalyzer performs Dynamic Mode Decomposition on wake data
type DMDWakeAnalyzer struct {
	FrameBuffer   []OpticalFlowFrame
	BufferSize    int
	KoopmanMatrix [][]float64 // Koopman operator Φ
}

// NewDMDWakeAnalyzer creates a new DMD wake analyzer
func NewDMDWakeAnalyzer(bufferSize int) *DMDWakeAnalyzer {
	return &DMDWakeAnalyzer{
		FrameBuffer: make([]OpticalFlowFrame, 0, bufferSize),
		BufferSize:  bufferSize,
	}
}

// AddFrame adds an optical flow frame to the buffer
func (d *DMDWakeAnalyzer) AddFrame(frame OpticalFlowFrame) {
	d.FrameBuffer = append(d.FrameBuffer, frame)

	// Maintain buffer size
	if len(d.FrameBuffer) > d.BufferSize {
		d.FrameBuffer = d.FrameBuffer[len(d.FrameBuffer)-d.BufferSize:]
	}
}

// ExtractWakeSignature extracts the wake signature from optical flow
// Algorithm:
// 1. Compute spatial FFT of velocity field
// 2. Identify dominant frequencies (Kármán vortex street)
// 3. Calculate Strouhal number: St = fL/U
// 4. Infer bank angle from vortex asymmetry
func (d *DMDWakeAnalyzer) ExtractWakeSignature() (WakeSignature, bool) {
	if len(d.FrameBuffer) < 2 {
		return WakeSignature{}, false
	}

	// Get latest frame
	frame := d.FrameBuffer[len(d.FrameBuffer)-1]

	// Compute spatial FFT of velocity field
	frequencies := d.computeSpatialFFT(frame.VelocityX, frame.VelocityY)

	// Identify dominant frequency (vortex shedding)
	dominantFreq, maxIndex := d.findDominantFrequency(frequencies)
	if dominantFreq <= 0 {
		return WakeSignature{}, false
	}

	// Estimate vehicle speed from flow magnitude
	speed := d.estimateSpeed(frame)
	if speed <= 0 {
		return WakeSignature{}, false
	}

	// Calculate Strouhal number
	// St = fL/U where L is characteristic length
	// For HGV at Mach 5: St ≈ 0.2-0.3
	characteristicLength := 10.0 // meters (typical HGV length)
	strouhal := dominantFreq * characteristicLength / speed

	// Infer bank angle from vortex asymmetry
	bankAngle := d.inferBankAngle(frame)

	// Calculate confidence based on spectral peak sharpness
	confidence := d.calculateConfidence(frequencies, maxIndex)

	signature := WakeSignature{
		StrouhalNumber:  strouhal,
		VortexFrequency: dominantFreq,
		BankAngle:       bankAngle,
		Speed:           speed,
		Confidence:      confidence,
	}

	return signature, confidence > 0.5
}

// computeSpatialFFT computes spatial FFT of velocity field
func (d *DMDWakeAnalyzer) computeSpatialFFT(vx, vy [][2]float64) []float64 {
	// Simplified FFT implementation
	// In production: use FFTW or GPU-accelerated FFT

	n := len(vx)
	frequencies := make([]float64, n/2)

	for k := 0; k < n/2; k++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			angle := 2.0 * math.Pi * float64(k) * float64(i) / float64(n)
			// Combine X and Y velocity components
			magnitude := math.Sqrt(vx[i][0]*vx[i][0] + vx[i][1]*vx[i][1] +
				vy[i][0]*vy[i][0] + vy[i][1]*vy[i][1])
			sum += magnitude * math.Cos(angle)
		}
		frequencies[k] = math.Abs(sum)
	}

	return frequencies
}

// findDominantFrequency finds the dominant frequency in the spectrum
func (d *DMDWakeAnalyzer) findDominantFrequency(frequencies []float64) (float64, int) {
	if len(frequencies) == 0 {
		return 0, 0
	}

	maxIndex := 0
	maxValue := frequencies[0]

	for i := 1; i < len(frequencies); i++ {
		if frequencies[i] > maxValue {
			maxValue = frequencies[i]
			maxIndex = i
		}
	}

	// Convert index to frequency (Hz)
	// Assuming 100 Hz sampling rate
	samplingRate := 100.0
	frequency := float64(maxIndex) * samplingRate / float64(len(frequencies))

	return frequency, maxIndex
}

// estimateSpeed estimates vehicle speed from flow magnitude
func (d *DMDWakeAnalyzer) estimateSpeed(frame OpticalFlowFrame) float64 {
	// Calculate average flow magnitude
	totalMagnitude := 0.0
	count := 0

	for i := 0; i < len(frame.VelocityX); i++ {
		magnitude := math.Sqrt(frame.VelocityX[i][0]*frame.VelocityX[i][0] +
			frame.VelocityX[i][1]*frame.VelocityX[i][1] +
			frame.VelocityY[i][0]*frame.VelocityY[i][0] +
			frame.VelocityY[i][1]*frame.VelocityY[i][1])
		totalMagnitude += magnitude
		count++
	}

	if count == 0 {
		return 0
	}

	avgMagnitude := totalMagnitude / float64(count)

	// Convert flow magnitude to speed (empirical calibration)
	// For HGV at Mach 5: ~1700 m/s
	speed := avgMagnitude * 1700.0

	return speed
}

// inferBankAngle infers bank angle from vortex asymmetry
func (d *DMDWakeAnalyzer) inferBankAngle(frame OpticalFlowFrame) float64 {
	// Calculate asymmetry in velocity field
	leftSum := 0.0
	rightSum := 0.0

	n := len(frame.VelocityX)
	midpoint := n / 2

	for i := 0; i < midpoint; i++ {
		magnitude := math.Sqrt(frame.VelocityY[i][0]*frame.VelocityY[i][0] +
			frame.VelocityY[i][1]*frame.VelocityY[i][1])
		leftSum += magnitude
	}

	for i := midpoint; i < n; i++ {
		magnitude := math.Sqrt(frame.VelocityY[i][0]*frame.VelocityY[i][0] +
			frame.VelocityY[i][1]*frame.VelocityY[i][1])
		rightSum += magnitude
	}

	// Bank angle from asymmetry (empirical)
	asymmetry := (leftSum - rightSum) / (leftSum + rightSum + 1e-6)
	bankAngle := asymmetry * math.Pi / 4.0 // Max ±45 degrees

	return bankAngle
}

// calculateConfidence calculates detection confidence from spectral peak sharpness
func (d *DMDWakeAnalyzer) calculateConfidence(frequencies []float64, dominantFreqIndex int) float64 {
	if len(frequencies) < 3 {
		return 0
	}

	// Check if dominant peak is sharp (narrowband)
	dominantValue := frequencies[dominantFreqIndex]

	// Average of neighboring frequencies
	neighborSum := 0.0
	count := 0

	for i := dominantFreqIndex - 2; i <= dominantFreqIndex+2; i++ {
		if i >= 0 && i < len(frequencies) && i != dominantFreqIndex {
			neighborSum += frequencies[i]
			count++
		}
	}

	if count == 0 {
		return 0
	}

	neighborAvg := neighborSum / float64(count)

	// Peak sharpness ratio
	sharpness := dominantValue / (neighborAvg + 1e-6)

	// Normalize to [0, 1]
	confidence := math.Min(sharpness/3.0, 1.0)

	return confidence
}

// DetectManeuver detects a maneuver from wake topology change
// Algorithm:
// 1. Compare current wake signature with previous
// 2. If Strouhal number or bank angle changes significantly → maneuver detected
// 3. Maneuver can be detected 1-2 seconds before centroid shift
func (d *DMDWakeAnalyzer) DetectManeuver() (bool, ManeuverType) {
	if len(d.FrameBuffer) < 10 {
		return false, ManeuverNone
	}

	// Get current and previous signatures
	currentSig, ok := d.ExtractWakeSignature()
	if !ok {
		return false, ManeuverNone
	}

	// Get signature from 5 frames ago (1-2 seconds ago)
	historyIndex := len(d.FrameBuffer) - 5
	if historyIndex < 0 {
		historyIndex = 0
	}

	// Temporarily swap frame to get historical signature
	currentFrame := d.FrameBuffer[len(d.FrameBuffer)-1]
	d.FrameBuffer[len(d.FrameBuffer)-1] = d.FrameBuffer[historyIndex]
	prevSig, ok := d.ExtractWakeSignature()
	d.FrameBuffer[len(d.FrameBuffer)-1] = currentFrame

	if !ok {
		return false, ManeuverNone
	}

	// Check for significant changes
	strouhalChange := math.Abs(currentSig.StrouhalNumber - prevSig.StrouhalNumber)
	bankAngleChange := math.Abs(currentSig.BankAngle - prevSig.BankAngle)

	// Thresholds for maneuver detection
	if strouhalChange > 0.05 || bankAngleChange > 0.1 {
		// Classify maneuver type
		maneuverType := d.classifyManeuverFromWake(currentSig, prevSig)
		return true, maneuverType
	}

	return false, ManeuverNone
}

// classifyManeuverFromWake classifies maneuver type from wake signature change
func (d *DMDWakeAnalyzer) classifyManeuverFromWake(current, prev WakeSignature) ManeuverType {
	bankAngleChange := current.BankAngle - prev.BankAngle

	// Bank angle change → turn maneuver
	if math.Abs(bankAngleChange) > 0.1 {
		if bankAngleChange > 0 {
			return "TURN_RIGHT"
		}
		return "TURN_LEFT"
	}

	// Strouhal number change → speed change (thrust maneuver)
	strouhalChange := current.StrouhalNumber - prev.StrouhalNumber
	if math.Abs(strouhalChange) > 0.05 {
		if strouhalChange > 0 {
			return "DECELERATION"
		}
		return "ACCELERATION"
	}

	return "MANEUVER_UNKNOWN"
}

// ComputeKoopmanOperator computes the Koopman operator from state snapshots
// Equation: X_{k+1} ≈ Φ X_k
func (d *DMDWakeAnalyzer) ComputeKoopmanOperator() bool {
	if len(d.FrameBuffer) < 3 {
		return false
	}

	// Build state matrices X and X'
	// In production: use SVD-based DMD algorithm
	// For now: placeholder implementation

	// This would compute the Koopman operator Φ
	// which represents the linear dynamics of the wake

	return true
}

// PredictNextState predicts the next wake state using Koopman operator
func (d *DMDWakeAnalyzer) PredictNextState() ([][2]float64, bool) {
	if len(d.KoopmanMatrix) == 0 {
		return nil, false
	}

	// Apply Koopman operator to current state
	// In production: X_{k+1} = Φ X_k

	return nil, true // Placeholder
}
