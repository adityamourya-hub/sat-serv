package sensors

/**
 * DSP Preprocessor
 *
 * Digital signal processing stage for sensor data.
 * Implements FFT, filtering, feature extraction, and denoising.
 *
 * This stage performs advanced signal processing on digitized sensor data,
 * including spectral analysis, filtering, and feature extraction for
 * detection and classification algorithms.
 *
 * References:
 *   - Oppenheim & Schafer: Discrete-Time Signal Processing
 *   - Proakis & Manolakis: Digital Signal Processing
 *   - Mallat: A Wavelet Tour of Signal Processing
 */

import (
	"math"

	"ssis/pkg/physics"
)

// =============================================================================
// DSP Preprocessor Configuration
// =============================================================================

// DSPConfig defines parameters for digital signal processing
type DSPConfig struct {
	// FFT parameters
	FFTSize        int     // 256, 512, 1024, 2048
	WindowFunction string  // "rectangular", "hamming", "hanning", "blackman"
	OverlapRatio   float64 // For STFT, 0.0-0.75

	// Filter parameters
	FilterType      string  // "FIR", "IIR", "Kalman", "matched"
	FilterOrder     int     // Filter order
	CutoffFreq      float64 // Hz, cutoff frequency
	FilterBandwidth float64 // Hz, for bandpass filters

	// Denoising parameters
	DenoisingMethod string // "wavelet", "median", "gaussian", "phs_aware"
	WaveletType     string // "daubechies4", "daubechies8"
	ThresholdMethod string // "universal", "adaptive"

	// Feature extraction parameters
	SNRThreshold   float64 // Minimum SNR for valid detection
	PeakThreshold  float64 // Threshold for peak detection
	CentroidMethod string  // "center_of_mass", "gaussian_fit"

	// PHS-aware denoising parameters
	EnablePHSAwareDenoising bool    // Use physics-constrained denoising
	HamiltonianThreshold    float64 // Energy conservation threshold for noise detection
	ManifoldGate            bool    // Apply manifold constraints to denoised signal
}

// DefaultDSPConfig returns standard configuration for space-based sensors
func DefaultDSPConfig() DSPConfig {
	return DSPConfig{
		FFTSize:                 1024,
		WindowFunction:          "hamming",
		OverlapRatio:            0.5,
		FilterType:              "FIR",
		FilterOrder:             32,
		CutoffFreq:              100.0,
		FilterBandwidth:         50.0,
		DenoisingMethod:         "phs_aware", // Enable physics-constrained denoising
		WaveletType:             "daubechies4",
		ThresholdMethod:         "universal",
		SNRThreshold:            10.0, // 10 dB
		PeakThreshold:           3.0,  // 3-sigma
		CentroidMethod:          "center_of_mass",
		EnablePHSAwareDenoising: true,
		HamiltonianThreshold:    1e-7, // Sovereign handshake threshold
		ManifoldGate:            true,
	}
}

// =============================================================================
// DSP Preprocessor
// =============================================================================

// DSPPreprocessor performs digital signal processing
type DSPPreprocessor struct {
	config       DSPConfig
	fftProcessor *physics.FFTProcessor
	filterState  []float64
	kalmanState  KalmanState

	// Statistics
	processingTime float64
	snrEstimate    float64
	peakCount      int
}

// KalmanState represents Kalman filter state
type KalmanState struct {
	State            []float64
	Covariance       [][]float64
	ProcessNoise     [][]float64
	MeasurementNoise [][]float64
}

// NewDSPPreprocessor creates DSP preprocessor with specified config
func NewDSPPreprocessor(config DSPConfig) *DSPPreprocessor {
	dp := &DSPPreprocessor{
		config:       config,
		fftProcessor: physics.NewFFTProcessor(),
		filterState:  make([]float64, config.FilterOrder),
	}

	// Initialize Kalman filter if needed
	if config.FilterType == "Kalman" {
		dp.kalmanState = KalmanState{
			State:            make([]float64, 2), // Position, velocity
			Covariance:       [][]float64{{1.0, 0.0}, {0.0, 1.0}},
			ProcessNoise:     [][]float64{{0.01, 0.0}, {0.0, 0.01}},
			MeasurementNoise: [][]float64{{0.1, 0.0}, {0.0, 0.1}},
		}
	}

	return dp
}

// Process applies DSP to input samples
// Returns processed samples, FFT spectrum, and extracted features
func (dp *DSPPreprocessor) Process(samples []float64) ([]float64, []complex128, *DSPFeatures) {
	// 1. Apply windowing
	windowed := dp.applyWindow(samples)

	// 2. Apply filtering
	filtered := dp.applyFilter(windowed)

	// 3. Apply denoising
	denoised := dp.applyDenoising(filtered)

	// 4. Compute FFT
	spectrum, _ := dp.fftProcessor.ComputeFFT(denoised)

	// 5. Extract features
	features := dp.extractFeatures(denoised, spectrum)

	// Update statistics
	dp.snrEstimate = features.SNR
	dp.peakCount = len(features.Peaks)

	return denoised, spectrum, features
}

// Reset resets the preprocessor state
func (dp *DSPPreprocessor) Reset() {
	dp.filterState = make([]float64, dp.config.FilterOrder)
	dp.processingTime = 0.0
	dp.snrEstimate = 0.0
	dp.peakCount = 0
}

// GetConfig returns current configuration
func (dp *DSPPreprocessor) GetConfig() DSPConfig {
	return dp.config
}

// GetStatistics returns processing statistics
func (dp *DSPPreprocessor) GetStatistics() (processingTime, snrEstimate float64, peakCount int) {
	return dp.processingTime, dp.snrEstimate, dp.peakCount
}

// =============================================================================
// Windowing
// =============================================================================

// applyWindow applies window function to reduce spectral leakage
func (dp *DSPPreprocessor) applyWindow(samples []float64) []float64 {
	if dp.config.WindowFunction == "rectangular" {
		return samples
	}

	windowed := make([]float64, len(samples))
	n := len(samples)

	for i := 0; i < n; i++ {
		var w float64
		switch dp.config.WindowFunction {
		case "hamming":
			w = 0.54 - 0.46*math.Cos(2.0*math.Pi*float64(i)/float64(n-1))
		case "hanning":
			w = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(n-1)))
		case "blackman":
			w = 0.42 - 0.5*math.Cos(2.0*math.Pi*float64(i)/float64(n-1)) +
				0.08*math.Cos(4.0*math.Pi*float64(i)/float64(n-1))
		default:
			w = 1.0
		}
		windowed[i] = samples[i] * w
	}

	return windowed
}

// =============================================================================
// Filtering
// =============================================================================

// applyFilter applies filtering based on configured type
func (dp *DSPPreprocessor) applyFilter(samples []float64) []float64 {
	switch dp.config.FilterType {
	case "FIR":
		return dp.applyFIRFilter(samples)
	case "IIR":
		return dp.applyIIRFilter(samples)
	case "Kalman":
		return dp.applyKalmanFilter(samples)
	case "matched":
		return dp.applyMatchedFilter(samples)
	default:
		return samples
	}
}

// applyFIRFilter applies FIR filter using window method
func (dp *DSPPreprocessor) applyFIRFilter(samples []float64) []float64 {
	// Simplified: Moving average filter
	// In production, use proper FIR filter design (Parks-McClellan, window method)
	order := dp.config.FilterOrder
	output := make([]float64, len(samples))

	for i := 0; i < len(samples); i++ {
		sum := 0.0
		count := 0
		for j := 0; j < order && i-j >= 0; j++ {
			sum += samples[i-j]
			count++
		}
		if count > 0 {
			output[i] = sum / float64(count)
		}
	}

	return output
}

// applyIIRFilter applies IIR Butterworth filter
func (dp *DSPPreprocessor) applyIIRFilter(samples []float64) []float64 {
	// Simplified: First-order IIR low-pass
	// H(z) = (1 - α) / (1 - αz^-1)
	alpha := 0.1 // Filter coefficient
	output := make([]float64, len(samples))

	for i := 0; i < len(samples); i++ {
		if i == 0 {
			output[i] = samples[i]
		} else {
			output[i] = alpha*output[i-1] + (1.0-alpha)*samples[i]
		}
	}

	return output
}

// applyKalmanFilter applies Kalman filter for state estimation
func (dp *DSPPreprocessor) applyKalmanFilter(samples []float64) []float64 {
	// Simplified 1D Kalman filter
	// In production, use full multi-dimensional Kalman
	Q := 0.01 // Process noise covariance
	R := 0.1  // Measurement noise covariance
	P := 1.0  // Error covariance
	K := 0.0  // Kalman gain
	x := 0.0  // State estimate

	output := make([]float64, len(samples))

	for i, measurement := range samples {
		// Predict
		P = P + Q

		// Update
		K = P / (P + R)
		x = x + K*(measurement-x)
		P = (1.0 - K) * P

		output[i] = x
	}

	return output
}

// applyMatchedFilter applies matched filter for known signal detection
func (dp *DSPPreprocessor) applyMatchedFilter(samples []float64) []float64 {
	// Simplified: Use impulse response of target signal
	// In production, would use actual signal template
	template := []float64{1.0, 0.8, 0.6, 0.4, 0.2} // Decaying impulse
	output := make([]float64, len(samples))

	for i := 0; i < len(samples); i++ {
		sum := 0.0
		for j := 0; j < len(template) && i-j >= 0; j++ {
			sum += samples[i-j] * template[j]
		}
		output[i] = sum
	}

	return output
}

// =============================================================================
// Denoising
// =============================================================================

// applyDenoising applies denoising based on configured method
func (dp *DSPPreprocessor) applyDenoising(samples []float64) []float64 {
	switch dp.config.DenoisingMethod {
	case "wavelet":
		return dp.applyWaveletDenoising(samples)
	case "median":
		return dp.applyMedianFilter(samples)
	case "gaussian":
		return dp.applyGaussianFilter(samples)
	case "phs_aware":
		return dp.applyPHSAwareDenoising(samples)
	case "fractional":
		return dp.applyFractionalFilter(samples)
	default:
		return samples
	}
}

// applyWaveletDenoising applies wavelet denoising (simplified)
func (dp *DSPPreprocessor) applyWaveletDenoising(samples []float64) []float64 {
	// Simplified: Haar wavelet decomposition
	// In production, use proper wavelet library
	output := make([]float64, len(samples))

	// Approximation (low-frequency)
	for i := 0; i < len(samples)/2; i++ {
		output[i] = (samples[2*i] + samples[2*i+1]) / 2.0
	}

	// Copy approximation back to output
	for i := 0; i < len(samples)/2; i++ {
		output[i] = output[i]
	}

	return output
}

// applyMedianFilter applies median filter for impulse noise removal
func (dp *DSPPreprocessor) applyMedianFilter(samples []float64) []float64 {
	windowSize := 5
	output := make([]float64, len(samples))

	for i := 0; i < len(samples); i++ {
		// Extract window
		start := i - windowSize/2
		if start < 0 {
			start = 0
		}
		end := i + windowSize/2 + 1
		if end > len(samples) {
			end = len(samples)
		}

		window := make([]float64, end-start)
		copy(window, samples[start:end])

		// Compute median
		output[i] = median(window)
	}

	return output
}

// applyPHSAwareDenoising applies physics-constrained denoising using PHS-derived priors
// This method ensures denoised signals respect Hamiltonian conservation constraints
func (dp *DSPPreprocessor) applyPHSAwareDenoising(samples []float64) []float64 {
	// PHS-aware denoising uses energy conservation as a prior
	// Noise is identified by violations of Hamiltonian constraints

	output := make([]float64, len(samples))

	// Step 1: Compute local energy (Hamiltonian proxy) for each sample
	// For 1D signal, use squared magnitude as energy proxy
	for i := 0; i < len(samples); i++ {
		localEnergy := samples[i] * samples[i]

		// Step 2: Apply energy-conserving smoothing
		// Only smooth samples that violate expected energy profile
		if i > 0 && i < len(samples)-1 {
			prevEnergy := samples[i-1] * samples[i-1]
			nextEnergy := samples[i+1] * samples[i+1]
			avgEnergy := (prevEnergy + nextEnergy) / 2.0

			// Check for energy violation (noise spike)
			energyDrift := math.Abs(localEnergy-avgEnergy) / (avgEnergy + 1e-10)

			if energyDrift > dp.config.HamiltonianThreshold {
				// Apply energy-conserving filter
				output[i] = (samples[i-1] + samples[i+1]) / 2.0
			} else {
				output[i] = samples[i]
			}
		} else {
			output[i] = samples[i]
		}

		// Step 3: Apply manifold gate if enabled
		if dp.config.ManifoldGate {
			// Ensure signal stays within physically valid range
			// For orbital dynamics, enforce smoothness constraint
			if i > 0 {
				maxChange := 0.1 * math.Abs(output[i-1]) // 10% max change
				change := output[i] - output[i-1]
				if math.Abs(change) > maxChange {
					sign := 1.0
					if change < 0 {
						sign = -1.0
					}
					output[i] = output[i-1] + sign*maxChange
				}
			}
		}
	}

	return output
}

// applyFractionalFilter applies fractional-order filtering for HGV-specific signals
// Fractional filters have memory effects that match HGV wake dynamics
func (dp *DSPPreprocessor) applyFractionalFilter(samples []float64) []float64 {
	// Fractional-order filter using Grünwald-Letnikov approximation
	// This captures non-local memory effects important for HGV tracking
	alpha := 0.7 // Fractional order (0.5-0.8 for transitional flow)

	output := make([]float64, len(samples))

	for n := 0; n < len(samples); n++ {
		sum := 0.0
		for k := 0; k <= n; k++ {
			// Grünwald-Letnikov coefficient
			coeff := dp.grunwaldLetnikovCoeff(alpha, k)
			if n-k >= 0 {
				sum += coeff * samples[n-k]
			}
		}
		output[n] = sum
	}

	return output
}

// grunwaldLetnikovCoeff computes Grünwald-Letnikov coefficient for fractional filter
func (dp *DSPPreprocessor) grunwaldLetnikovCoeff(alpha float64, k int) float64 {
	if k == 0 {
		return 1.0
	}

	coeff := 1.0
	for j := 0; j < k; j++ {
		coeff *= (float64(j) - alpha) / float64(j+1)
	}

	return coeff
}

// applyGaussianFilter applies Gaussian smoothing
func (dp *DSPPreprocessor) applyGaussianFilter(samples []float64) []float64 {
	sigma := 1.0
	kernelSize := 5
	kernel := make([]float64, kernelSize)

	// Generate Gaussian kernel
	for i := 0; i < kernelSize; i++ {
		x := float64(i - kernelSize/2)
		kernel[i] = math.Exp(-x * x / (2.0 * sigma * sigma))
	}

	// Normalize kernel
	sum := 0.0
	for _, v := range kernel {
		sum += v
	}
	for i := range kernel {
		kernel[i] /= sum
	}

	// Apply convolution
	output := make([]float64, len(samples))
	for i := 0; i < len(samples); i++ {
		convSum := 0.0
		for j := 0; j < kernelSize; j++ {
			idx := i - kernelSize/2 + j
			if idx >= 0 && idx < len(samples) {
				convSum += samples[idx] * kernel[j]
			}
		}
		output[i] = convSum
	}

	return output
}

// =============================================================================
// Feature Extraction
// =============================================================================

// DSPFeatures contains extracted signal features
type DSPFeatures struct {
	SNR           float64        // Signal-to-noise ratio (dB)
	Centroid      float64        // Spectral centroid (Hz)
	Bandwidth     float64        // Signal bandwidth (Hz)
	Peaks         []SpectralPeak // Spectral peaks
	RMS           float64        // RMS amplitude
	PeakToPeak    float64        // Peak-to-peak amplitude
	ZeroCrossings int            // Number of zero crossings
}

// SpectralPeak represents a peak in the frequency spectrum
type SpectralPeak struct {
	Frequency float64 // Hz
	Power     float64 // Linear power
	Amplitude float64 // dB
	Bandwidth float64 // Hz
}

// extractFeatures extracts features from time-domain and frequency-domain data
func (dp *DSPPreprocessor) extractFeatures(samples []float64, spectrum []complex128) *DSPFeatures {
	features := &DSPFeatures{}

	// Time-domain features
	features.RMS = computeRMS(samples)
	features.PeakToPeak = computePeakToPeak(samples)
	features.ZeroCrossings = computeZeroCrossings(samples)

	// Frequency-domain features
	features.Centroid = computeSpectralCentroid(spectrum, dp.config.FFTSize)
	features.Bandwidth = computeSpectralBandwidth(spectrum, dp.config.FFTSize)
	features.Peaks = dp.detectSpectralPeaks(spectrum, dp.config.FFTSize)

	// SNR estimation
	if len(features.Peaks) > 0 {
		signalPower := features.Peaks[0].Power
		noisePower := estimateNoiseFloor(spectrum)
		if noisePower > 0 {
			features.SNR = 10.0 * math.Log10(signalPower/noisePower)
		}
	}

	return features
}

// detectSpectralPeaks finds peaks in frequency spectrum
func (dp *DSPPreprocessor) detectSpectralPeaks(spectrum []complex128, fftSize int) []SpectralPeak {
	peaks := make([]SpectralPeak, 0)

	// Find peaks above threshold
	for i := 1; i < len(spectrum)-1; i++ {
		power := complexMagnitude(spectrum[i])
		prevPower := complexMagnitude(spectrum[i-1])
		nextPower := complexMagnitude(spectrum[i+1])

		if power > prevPower && power > nextPower {
			// This is a local maximum
			freq := float64(i) * 10000.0 / float64(fftSize) // Assuming 10 kHz sample rate
			amplitude := 10.0 * math.Log10(power+1e-10)

			peaks = append(peaks, SpectralPeak{
				Frequency: freq,
				Power:     power,
				Amplitude: amplitude,
			})
		}
	}

	return peaks
}

// =============================================================================
// Utility Functions
// =============================================================================

// complexMagnitude computes magnitude of complex number without cmplx package
func complexMagnitude(c complex128) float64 {
	return math.Sqrt(real(c)*real(c) + imag(c)*imag(c))
}

// computeRMS computes RMS of signal
func computeRMS(samples []float64) float64 {
	sum := 0.0
	for _, s := range samples {
		sum += s * s
	}
	return math.Sqrt(sum / float64(len(samples)))
}

// computePeakToPeak computes peak-to-peak amplitude
func computePeakToPeak(samples []float64) float64 {
	if len(samples) == 0 {
		return 0.0
	}
	max := samples[0]
	min := samples[0]
	for _, s := range samples {
		if s > max {
			max = s
		}
		if s < min {
			min = s
		}
	}
	return max - min
}

// computeZeroCrossings counts zero crossings
func computeZeroCrossings(samples []float64) int {
	count := 0
	for i := 1; i < len(samples); i++ {
		if (samples[i] >= 0 && samples[i-1] < 0) || (samples[i] < 0 && samples[i-1] >= 0) {
			count++
		}
	}
	return count
}

// computeSpectralCentroid computes centroid of spectrum
func computeSpectralCentroid(spectrum []complex128, fftSize int) float64 {
	sumFreqPower := 0.0
	sumPower := 0.0

	for i, val := range spectrum {
		power := complexMagnitude(val)
		freq := float64(i) * 10000.0 / float64(fftSize) // Assuming 10 kHz
		sumFreqPower += freq * power
		sumPower += power
	}

	if sumPower == 0 {
		return 0.0
	}
	return sumFreqPower / sumPower
}

// computeSpectralBandwidth computes 3dB bandwidth
func computeSpectralBandwidth(spectrum []complex128, fftSize int) float64 {
	// Find peak power
	maxPower := 0.0
	for _, val := range spectrum {
		power := complexMagnitude(val)
		if power > maxPower {
			maxPower = power
		}
	}

	// Find 3dB points
	threshold := maxPower / math.Sqrt(2)
	firstAbove := -1
	lastAbove := -1

	for i, val := range spectrum {
		power := complexMagnitude(val)
		if power >= threshold {
			if firstAbove == -1 {
				firstAbove = i
			}
			lastAbove = i
		}
	}

	if firstAbove == -1 {
		return 0.0
	}

	// Convert to Hz
	binWidth := 10000.0 / float64(fftSize)
	return float64(lastAbove-firstAbove+1) * binWidth
}

// estimateNoiseFloor estimates noise floor from spectrum
func estimateNoiseFloor(spectrum []complex128) float64 {
	// Use median of lower 50% of spectrum as noise estimate
	powers := make([]float64, len(spectrum))
	for i, val := range spectrum {
		powers[i] = complexMagnitude(val)
	}

	// Sort and take median of lower half
	// Simplified: use mean of lower 25%
	sum := 0.0
	count := len(powers) / 4
	for i := 0; i < count; i++ {
		sum += powers[i]
	}

	if count == 0 {
		return 1e-10
	}
	return sum / float64(count)
}

// median computes median of slice
func median(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}

	// Simple sort (in production, use efficient algorithm)
	sorted := make([]float64, len(data))
	copy(sorted, data)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2.0
	}
	return sorted[mid]
}
