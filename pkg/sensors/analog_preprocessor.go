package sensors

/**
 * Analog Preprocessor
 *
 * Simulates analog front-end signal conditioning for sensor inputs.
 * Implements anti-aliasing filtering, automatic gain control (AGC),
 * noise modeling, and saturation detection.
 *
 * This stage models the physical analog electronics before ADC conversion,
 * critical for realistic sensor simulation and understanding system limitations.
 *
 * References:
 *   - Oppenheim & Schafer: Discrete-Time Signal Processing
 *   - Texas Instruments: Analog Front-End Design Guide
 *   - ADI: Low Noise Amplifier Design
 */

import (
	"math"
)

// =============================================================================
// Analog Preprocessor Configuration
// =============================================================================

// AnalogConfig defines parameters for analog signal conditioning
type AnalogConfig struct {
	// Anti-aliasing filter parameters
	FilterCutoffFreq float64 // Hz, cutoff frequency for anti-aliasing
	FilterOrder      int     // 2nd, 4th, or 8th order Butterworth
	FilterType       string  // "Butterworth", "Chebyshev", "Bessel"

	// AGC parameters
	AGCAttackTime  float64 // seconds, attack time constant
	AGCReleaseTime float64 // seconds, release time constant
	AGCTargetLevel float64 // dBm, target output level
	AGCMaxGain     float64 // dB, maximum gain
	AGCMinGain     float64 // dB, minimum gain

	// Noise parameters
	NoiseFigure float64 // dB, overall noise figure
	Temperature float64 // Kelvin, for thermal noise
	Bandwidth   float64 // Hz, noise bandwidth

	// Saturation parameters
	SaturationLevel   float64 // V, absolute maximum input level
	ClippingThreshold float64 // V, clipping detection threshold
}

// DefaultAnalogConfig returns standard configuration for space-based sensors
func DefaultAnalogConfig() AnalogConfig {
	return AnalogConfig{
		FilterCutoffFreq:  1000.0, // 1 kHz typical
		FilterOrder:       4,      // 4th order Butterworth
		FilterType:        "Butterworth",
		AGCAttackTime:     0.01,   // 10 ms
		AGCReleaseTime:    0.1,    // 100 ms
		AGCTargetLevel:    -10.0,  // -10 dBm
		AGCMaxGain:        60.0,   // 60 dB
		AGCMinGain:        0.0,    // 0 dB
		NoiseFigure:       3.0,    // 3 dB
		Temperature:       300.0,  // 300 K
		Bandwidth:         1000.0, // 1 kHz
		SaturationLevel:   1.0,    // 1 V
		ClippingThreshold: 0.95,   // 0.95 V
	}
}

// =============================================================================
// Analog Preprocessor
// =============================================================================

// AnalogPreprocessor simulates analog front-end signal conditioning
type AnalogPreprocessor struct {
	config AnalogConfig

	// AGC state
	currentGain  float64
	peakDetector float64

	// Filter state (for IIR filter implementation)
	filterState  []float64
	filterCoeffs []float64

	// Statistics
	clippingCount   int
	saturationCount int
	inputRMS        float64
	outputRMS       float64
}

// NewAnalogPreprocessor creates analog preprocessor with specified config
func NewAnalogPreprocessor(config AnalogConfig) *AnalogPreprocessor {
	ap := &AnalogPreprocessor{
		config:       config,
		currentGain:  10.0, // Start with 10 dB gain
		peakDetector: 0.0,
		filterState:  make([]float64, config.FilterOrder),
	}

	// Compute filter coefficients after struct creation
	ap.filterCoeffs = ap.computeFilterCoefficients()

	return ap
}

// Process applies analog signal conditioning to input samples
// Returns processed samples and clipping flag
func (ap *AnalogPreprocessor) Process(samples []float64) ([]float64, bool) {
	clipped := false
	output := make([]float64, len(samples))

	for i, sample := range samples {
		// 1. Check for saturation
		if math.Abs(sample) > ap.config.SaturationLevel {
			ap.saturationCount++
			continue // Drop saturated samples
		}

		// 2. Check for clipping
		if math.Abs(sample) > ap.config.ClippingThreshold {
			ap.clippingCount++
			clipped = true
		}

		// 3. Apply anti-aliasing filter
		filtered := ap.applyAntiAliasingFilter(sample)

		// 4. Apply AGC
		agcOutput := ap.applyAGC(filtered)

		// 5. Add thermal noise
		noisy := ap.addThermalNoise(agcOutput)

		output[i] = noisy
	}

	// Update RMS statistics
	ap.updateRMSStatistics(samples, output)

	return output, clipped
}

// Reset resets the preprocessor state
func (ap *AnalogPreprocessor) Reset() {
	ap.currentGain = 10.0
	ap.peakDetector = 0.0
	ap.filterState = make([]float64, ap.config.FilterOrder)
	ap.clippingCount = 0
	ap.saturationCount = 0
	ap.inputRMS = 0.0
	ap.outputRMS = 0.0
}

// GetConfig returns current configuration
func (ap *AnalogPreprocessor) GetConfig() AnalogConfig {
	return ap.config
}

// GetStatistics returns processing statistics
func (ap *AnalogPreprocessor) GetStatistics() (clipping, saturation int, inputRMS, outputRMS float64) {
	return ap.clippingCount, ap.saturationCount, ap.inputRMS, ap.outputRMS
}

// =============================================================================
// Anti-Aliasing Filter Implementation
// =============================================================================

// computeFilterCoefficients computes Butterworth filter coefficients
// Returns numerator (b) and denominator (a) coefficients for IIR filter
func (ap *AnalogPreprocessor) computeFilterCoefficients() []float64 {
	// Simplified: For production, use proper IIR filter design
	// Here, we implement a simple first-order low-pass
	// H(s) = 1 / (1 + s/ωc)

	// Discretize using bilinear transform
	// For 4th order, would cascade 4 first-order sections
	omegaC := 2.0 * math.Pi * ap.config.FilterCutoffFreq
	T := 1.0 / (2.0 * ap.config.FilterCutoffFreq) // Sampling period

	// Bilinear transform coefficients
	alpha := omegaC * T / 2.0
	b0 := alpha / (1.0 + alpha)
	a1 := (1.0 - alpha) / (1.0 + alpha)

	return []float64{b0, a1} // [b0, a1] for first-order IIR
}

// applyAntiAliasingFilter applies IIR low-pass filter
func (ap *AnalogPreprocessor) applyAntiAliasingFilter(sample float64) float64 {
	if len(ap.filterCoeffs) < 2 {
		return sample
	}

	b0 := ap.filterCoeffs[0]
	a1 := ap.filterCoeffs[1]

	// First-order IIR: y[n] = b0*x[n] - a1*y[n-1]
	output := b0*sample - a1*ap.filterState[0]

	// Update state
	ap.filterState[0] = output

	return output
}

// =============================================================================
// AGC Implementation
// =============================================================================

// applyAGC applies automatic gain control
func (ap *AnalogPreprocessor) applyAGC(sample float64) float64 {
	// Update peak detector
	if math.Abs(sample) > ap.peakDetector {
		ap.peakDetector = ap.peakDetector + (math.Abs(sample)-ap.peakDetector)/ap.config.AGCAttackTime
	} else {
		ap.peakDetector = ap.peakDetector - (ap.peakDetector-math.Abs(sample))/ap.config.AGCReleaseTime
	}

	// Convert target level to linear scale
	targetLinear := math.Pow(10.0, ap.config.AGCTargetLevel/10.0)

	// Compute error
	error := targetLinear - ap.peakDetector

	// Adjust gain (simplified PI controller)
	kp := 0.1 // Proportional gain
	gainAdjust := kp * error

	// Convert dB to linear
	currentGainLinear := math.Pow(10.0, ap.currentGain/10.0)
	currentGainLinear += gainAdjust

	// Clamp gain
	ap.currentGain = 10.0 * math.Log10(currentGainLinear)
	if ap.currentGain > ap.config.AGCMaxGain {
		ap.currentGain = ap.config.AGCMaxGain
	}
	if ap.currentGain < ap.config.AGCMinGain {
		ap.currentGain = ap.config.AGCMinGain
	}

	// Apply gain
	gainLinear := math.Pow(10.0, ap.currentGain/10.0)
	return sample * gainLinear
}

// =============================================================================
// Noise Modeling
// =============================================================================

// addThermalNoise adds thermal (Johnson-Nyquist) noise
// Noise power: P = kTB (Boltzmann constant * Temperature * Bandwidth)
func (ap *AnalogPreprocessor) addThermalNoise(sample float64) float64 {
	// Boltzmann constant in J/K
	k := 1.380649e-23

	// Thermal noise power (Watts)
	noisePower := k * ap.config.Temperature * ap.config.Bandwidth

	// Convert to voltage (assuming 50 ohm impedance)
	noiseVoltage := math.Sqrt(noisePower * 50.0)

	// Apply noise figure
	noiseFigureLinear := math.Pow(10.0, ap.config.NoiseFigure/10.0)
	noiseVoltage *= math.Sqrt(noiseFigureLinear)

	// Add Gaussian noise
	noise := gaussianRandom() * noiseVoltage

	return sample + noise
}

// gaussianRandom generates Gaussian random number with zero mean, unit variance
// Box-Muller transform using simple pseudo-random
func gaussianRandom() float64 {
	// Simple pseudo-random using time-based seed (simplified)
	// In production, use proper random number generator
	seed := float64(12345) // Fixed seed for reproducibility
	seed = math.Mod(seed*1103515245+12345, 2147483648.0)
	u1 := seed / 2147483648.0

	seed = math.Mod(seed*1103515245+12345, 2147483648.0)
	u2 := seed / 2147483648.0

	// Handle edge case
	if u1 <= 1e-10 {
		return 0.0
	}

	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

// =============================================================================
// Statistics
// =============================================================================

// updateRMSStatistics computes RMS of input and output
func (ap *AnalogPreprocessor) updateRMSStatistics(input, output []float64) {
	if len(input) == 0 {
		return
	}

	sumIn := 0.0
	sumOut := 0.0

	for _, v := range input {
		sumIn += v * v
	}
	for _, v := range output {
		sumOut += v * v
	}

	ap.inputRMS = math.Sqrt(sumIn / float64(len(input)))
	ap.outputRMS = math.Sqrt(sumOut / float64(len(output)))
}
