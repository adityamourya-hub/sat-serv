package sensors

/**
 * Digital Preprocessor
 *
 * Simulates ADC (Analog-to-Digital Converter) and digital signal conditioning.
 * Implements sampling, quantization, dithering, and clipping detection.
 *
 * This stage models the conversion from continuous analog signals to discrete
 * digital samples, critical for understanding quantization noise, dynamic range,
 * and the limitations of digital sensor systems.
 *
 * References:
 *   - Oppenheim & Schafer: Discrete-Time Signal Processing
 *   - Maxim Integrated: ADC Fundamentals
 *   - Texas Instruments: Understanding Data Converters
 */

import (
	"math"
)

// =============================================================================
// Digital Preprocessor Configuration
// =============================================================================

// DigitalConfig defines parameters for ADC and digital processing
type DigitalConfig struct {
	// Sampling parameters
	SampleRate      float64 // Hz, sampling frequency
	InputSampleRate float64 // Hz, input sample rate for resampling

	// Quantization parameters
	BitDepth       int     // 8, 12, 16, or 24 bits
	FullScaleVolts float64 // V, full-scale input voltage
	DitherType     string  // "none", "TPDF", "gaussian"

	// Resampling parameters
	InterpolationMethod  string // "linear", "cubic", "sinc"
	AntiAliasFilterOrder int    // Filter order for resampling

	// Clipping parameters
	SoftClip      bool    // Use soft clipping instead of hard clipping
	SoftClipRatio float64 // Soft clipping ratio
}

// DefaultDigitalConfig returns standard configuration for space-based sensors
func DefaultDigitalConfig() DigitalConfig {
	return DigitalConfig{
		SampleRate:           10000.0, // 10 kHz
		InputSampleRate:      10000.0, // 10 kHz (no resampling)
		BitDepth:             16,      // 16-bit ADC
		FullScaleVolts:       2.0,     // ±1V range
		DitherType:           "TPDF",  // Triangular PDF dither
		InterpolationMethod:  "linear",
		AntiAliasFilterOrder: 4,
		SoftClip:             false,
		SoftClipRatio:        0.95,
	}
}

// =============================================================================
// Digital Preprocessor
// =============================================================================

// DigitalPreprocessor simulates ADC and digital signal conditioning
type DigitalPreprocessor struct {
	config DigitalConfig

	// Quantization state
	quantizationLevels int
	stepSize           float64
	ditherAccumulator  float64

	// Statistics
	clippingCount     int
	quantizationError float64
	snrEstimate       float64
}

// NewDigitalPreprocessor creates digital preprocessor with specified config
func NewDigitalPreprocessor(config DigitalConfig) *DigitalPreprocessor {
	dp := &DigitalPreprocessor{
		config:            config,
		ditherAccumulator: 0.0,
	}

	// Compute quantization parameters
	dp.quantizationLevels = 1 << uint(config.BitDepth)
	dp.stepSize = (2.0 * config.FullScaleVolts) / float64(dp.quantizationLevels)

	return dp
}

// Process applies ADC simulation to input samples
// Returns quantized samples and clipping flag
func (dp *DigitalPreprocessor) Process(samples []float64) ([]float64, bool) {
	clipped := false
	output := make([]float64, len(samples))

	for i, sample := range samples {
		// 1. Check for clipping
		if math.Abs(sample) > dp.config.FullScaleVolts {
			dp.clippingCount++
			clipped = true

			if dp.config.SoftClip {
				// Apply soft clipping
				sample = dp.softClip(sample)
			} else {
				// Hard clip
				if sample > dp.config.FullScaleVolts {
					sample = dp.config.FullScaleVolts
				} else {
					sample = -dp.config.FullScaleVolts
				}
			}
		}

		// 2. Apply dither (if enabled)
		if dp.config.DitherType != "none" {
			sample = dp.applyDither(sample)
		}

		// 3. Quantize
		quantized := dp.quantize(sample)
		output[i] = quantized

		// Track quantization error
		dp.quantizationError += (sample - quantized) * (sample - quantized)
	}

	// Update SNR estimate
	if len(samples) > 0 {
		dp.quantizationError /= float64(len(samples))
		signalPower := 0.0
		for _, s := range samples {
			signalPower += s * s
		}
		signalPower /= float64(len(samples))
		if signalPower > 0 {
			dp.snrEstimate = 10.0 * math.Log10(signalPower/dp.quantizationError)
		}
	}

	return output, clipped
}

// Reset resets the preprocessor state
func (dp *DigitalPreprocessor) Reset() {
	dp.ditherAccumulator = 0.0
	dp.clippingCount = 0
	dp.quantizationError = 0.0
	dp.snrEstimate = 0.0
}

// GetConfig returns current configuration
func (dp *DigitalPreprocessor) GetConfig() DigitalConfig {
	return dp.config
}

// GetStatistics returns processing statistics
func (dp *DigitalPreprocessor) GetStatistics() (clipping int, quantizationError, snrEstimate float64) {
	return dp.clippingCount, dp.quantizationError, dp.snrEstimate
}

// GetSQNR computes theoretical Signal-to-Quantization Noise Ratio
// SQNR = 6.02N + 1.76 dB where N is bit depth
func (dp *DigitalPreprocessor) GetSQNR() float64 {
	return 6.02*float64(dp.config.BitDepth) + 1.76
}

// =============================================================================
// Resampling
// =============================================================================

// Resample converts input sample rate to output sample rate
func (dp *DigitalPreprocessor) Resample(samples []float64, inputRate, outputRate float64) []float64 {
	if inputRate == outputRate {
		return samples
	}

	// Compute resampling ratio
	ratio := outputRate / inputRate
	newLength := int(math.Ceil(float64(len(samples)) * ratio))
	output := make([]float64, newLength)

	for i := 0; i < newLength; i++ {
		// Compute fractional index in input
		inputIndex := float64(i) / ratio

		// Interpolate
		switch dp.config.InterpolationMethod {
		case "linear":
			output[i] = dp.linearInterpolate(samples, inputIndex)
		case "cubic":
			output[i] = dp.cubicInterpolate(samples, inputIndex)
		default:
			output[i] = dp.linearInterpolate(samples, inputIndex)
		}
	}

	return output
}

// linearInterpolate performs linear interpolation
func (dp *DigitalPreprocessor) linearInterpolate(samples []float64, index float64) float64 {
	i := int(index)
	if i >= len(samples)-1 {
		return samples[len(samples)-1]
	}
	if i < 0 {
		return samples[0]
	}

	frac := index - float64(i)
	return samples[i]*(1.0-frac) + samples[i+1]*frac
}

// cubicInterpolate performs cubic interpolation
func (dp *DigitalPreprocessor) cubicInterpolate(samples []float64, index float64) float64 {
	i := int(index)
	if i < 1 || i >= len(samples)-2 {
		return dp.linearInterpolate(samples, index)
	}

	frac := index - float64(i)

	// Cubic interpolation coefficients
	c0 := samples[i+1] - samples[i]
	c1 := samples[i+2] - samples[i+1] - c0
	c2 := samples[i] - samples[i+1] - c1
	c3 := samples[i+2] - samples[i]

	return c3 + frac*(c2+frac*(c1+frac*c0))
}

// =============================================================================
// Quantization
// =============================================================================

// quantize converts analog value to discrete digital value
func (dp *DigitalPreprocessor) quantize(sample float64) float64 {
	// Convert to ADC code
	code := (sample + dp.config.FullScaleVolts) / dp.stepSize

	// Clamp to valid range
	if code < 0 {
		code = 0
	}
	if code >= float64(dp.quantizationLevels) {
		code = float64(dp.quantizationLevels) - 1
	}

	// Convert back to voltage
	quantized := code*dp.stepSize - dp.config.FullScaleVolts

	return quantized
}

// =============================================================================
// Dithering
// =============================================================================

// applyDither adds dither signal to reduce quantization artifacts
func (dp *DigitalPreprocessor) applyDither(sample float64) float64 {
	var dither float64

	switch dp.config.DitherType {
	case "TPDF":
		// Triangular PDF dither (sum of two uniform)
		u1 := simpleRandom()
		u2 := simpleRandom()
		dither = (u1 + u2 - 1.0) * dp.stepSize / 2.0
	case "gaussian":
		// Gaussian dither
		dither = gaussianRandom() * dp.stepSize / 2.0
	case "none":
		dither = 0.0
	default:
		dither = 0.0
	}

	return sample + dither
}

// =============================================================================
// Clipping
// =============================================================================

// softClip applies soft clipping to prevent hard saturation
func (dp *DigitalPreprocessor) softClip(sample float64) float64 {
	ratio := dp.config.SoftClipRatio
	if math.Abs(sample) <= ratio*dp.config.FullScaleVolts {
		return sample
	}

	// Hyperbolic tangent soft clipping
	sign := 1.0
	if sample < 0 {
		sign = -1.0
	}

	normalized := math.Abs(sample) / (ratio * dp.config.FullScaleVolts)
	softClipped := sign * dp.config.FullScaleVolts * math.Tanh(normalized/ratio)

	return softClipped
}

// =============================================================================
// Utilities
// =============================================================================

// simpleRandom generates simple pseudo-random number [0, 1)
func simpleRandom() float64 {
	// Simple LCG (Linear Congruential Generator)
	// In production, use proper RNG
	seed := float64(54321) // Fixed seed for reproducibility
	seed = math.Mod(seed*1103515245+12345, 2147483648.0)
	return seed / 2147483648.0
}
