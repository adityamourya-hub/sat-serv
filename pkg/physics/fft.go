// Package physics provides Go-native FFT utilities using gosl
// Replaces scipy dependency for Digital Signal Processing
package physics

import (
	"math"
	// "github.com/cpmech/gosl/num"  // TODO: Enable when gosl available
)

// FFTProcessor provides Fast Fourier Transform utilities
// for spectral analysis without Python/scipy dependency
// Uses pure Go DFT implementation (can be upgraded to FFT when gosl available)
type FFTProcessor struct {
	// Cache for twiddle factors to speed up repeated transforms
	twiddleCache map[int][]complex128
}

// NewFFTProcessor creates FFT processor
func NewFFTProcessor() *FFTProcessor {
	return &FFTProcessor{}
}

// ComputeFFT performs forward DFT on real input
// Returns complex spectrum [N/2+1] (positive frequencies only for real input)
// Note: This is O(N^2) DFT, not O(N log N) FFT. Upgrade to gosl when available.
func (f *FFTProcessor) ComputeFFT(input []float64) ([]complex128, error) {
	n := len(input)
	output := make([]complex128, n)

	// Generate or retrieve twiddle factors
	twiddle := f.getTwiddleFactors(n)

	// DFT: X[k] = sum(x[n] * e^(-2πi*k*n/N))
	for k := 0; k < n; k++ {
		sum := complex(0, 0)
		for j := 0; j < n; j++ {
			idx := (k * j) % n
			sum += complex(input[j], 0) * twiddle[idx]
		}
		output[k] = sum
	}

	// Return only positive frequencies for real input
	return output[:n/2+1], nil
}

// getTwiddleFactors generates or retrieves cached twiddle factors
func (f *FFTProcessor) getTwiddleFactors(n int) []complex128 {
	if f.twiddleCache == nil {
		f.twiddleCache = make(map[int][]complex128)
	}

	if twiddle, exists := f.twiddleCache[n]; exists {
		return twiddle
	}

	// Generate twiddle factors: W_N^k = e^(-2πi*k/N)
	twiddle := make([]complex128, n)
	for k := 0; k < n; k++ {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(math.Cos(angle), math.Sin(angle))
	}

	f.twiddleCache[n] = twiddle
	return twiddle
}

// ComputeIFFT performs inverse DFT
func (f *FFTProcessor) ComputeIFFT(spectrum []complex128) ([]float64, error) {
	n := len(spectrum)*2 - 2 // Full length for real output

	// Reconstruct full spectrum (conjugate symmetry for real input)
	fullSpectrum := make([]complex128, n)
	copy(fullSpectrum, spectrum)

	// Add negative frequencies (conjugate of positive)
	for k := 1; k < len(spectrum)-1; k++ {
		fullSpectrum[n-k] = complex(real(spectrum[k]), -imag(spectrum[k]))
	}

	// Inverse DFT: x[n] = (1/N) * sum(X[k] * e^(2πi*k*n/N))
	output := make([]float64, n)
	twiddle := f.getTwiddleFactors(n)

	for j := 0; j < n; j++ {
		sum := complex(0, 0)
		for k := 0; k < n; k++ {
			idx := (k * j) % n
			// Use conjugate of twiddle for inverse
			sum += fullSpectrum[k] * complex(real(twiddle[idx]), -imag(twiddle[idx]))
		}
		output[j] = real(sum) / float64(n)
	}

	return output, nil
}

// ComputePowerSpectrum returns magnitude squared of FFT
func (f *FFTProcessor) ComputePowerSpectrum(input []float64) ([]float64, error) {
	fft, err := f.ComputeFFT(input)
	if err != nil {
		return nil, err
	}

	power := make([]float64, len(fft))
	for i, c := range fft {
		// |z|² = re² + im²
		power[i] = real(c)*real(c) + imag(c)*imag(c)
	}

	return power, nil
}

// FindDominantFrequency identifies peak frequency in spectrum
func (f *FFTProcessor) FindDominantFrequency(powerSpectrum []float64, sampleRate float64) (freq float64, magnitude float64) {
	if len(powerSpectrum) == 0 {
		return 0, 0
	}

	// Find index of maximum power
	maxIdx := 0
	maxPower := powerSpectrum[0]

	// Skip DC component (index 0)
	for i := 1; i < len(powerSpectrum); i++ {
		if powerSpectrum[i] > maxPower {
			maxPower = powerSpectrum[i]
			maxIdx = i
		}
	}

	// Convert index to frequency
	// For N-point FFT with sampleRate: freq = idx * sampleRate / N
	n := 2 * (len(powerSpectrum) - 1) // Original signal length
	freq = float64(maxIdx) * sampleRate / float64(n)
	magnitude = maxPower

	return freq, magnitude
}

// Convolve performs FFT-based convolution
// Much faster than direct convolution for large kernels
func (f *FFTProcessor) Convolve(signal, kernel []float64) []float64 {
	// Implementation using FFT multiplication theorem
	// FFT(convolution) = FFT(signal) * FFT(kernel)

	// Pad to next power of 2
	n := nextPowerOf2(len(signal) + len(kernel) - 1)

	// Zero-pad signal and kernel
	sigPadded := make([]float64, n)
	kerPadded := make([]float64, n)
	copy(sigPadded, signal)
	copy(kerPadded, kernel)

	// FFT both
	sigFFT, _ := f.ComputeFFT(sigPadded)
	kerFFT, _ := f.ComputeFFT(kerPadded)

	// Multiply in frequency domain
	product := make([]complex128, len(sigFFT))
	for i := range product {
		product[i] = sigFFT[i] * kerFFT[i]
	}

	// Inverse FFT
	result, _ := f.ComputeIFFT(product)

	// Return valid convolution result
	validLen := len(signal) + len(kernel) - 1
	return result[:validLen]
}

// CrossCorrelate performs FFT-based cross-correlation
// Used for template matching and signal alignment
func (f *FFTProcessor) CrossCorrelate(signal, template []float64) []float64 {
	// Cross-correlation via FFT:
	// FFT(xcorr) = FFT(signal) * conj(FFT(template))

	n := nextPowerOf2(len(signal) + len(template) - 1)

	sigPadded := make([]float64, n)
	tempPadded := make([]float64, n)
	copy(sigPadded, signal)
	copy(tempPadded, template)

	sigFFT, _ := f.ComputeFFT(sigPadded)
	tempFFT, _ := f.ComputeFFT(tempPadded)

	// Multiply signal FFT with conjugate of template FFT
	product := make([]complex128, len(sigFFT))
	for i := range product {
		// conj(a+bi) = a-bi
		conjTemp := complex(real(tempFFT[i]), -imag(tempFFT[i]))
		product[i] = sigFFT[i] * conjTemp
	}

	result, _ := f.ComputeIFFT(product)

	validLen := len(signal) + len(template) - 1
	return result[:validLen]
}

// nextPowerOf2 returns smallest power of 2 >= n
func nextPowerOf2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// Window applies window function to signal (Hann window)
func (f *FFTProcessor) Window(signal []float64) []float64 {
	n := len(signal)
	windowed := make([]float64, n)

	for i := 0; i < n; i++ {
		// Hann window: w[n] = 0.5 * (1 - cos(2πn/(N-1)))
		hann := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
		windowed[i] = signal[i] * hann
	}

	return windowed
}
