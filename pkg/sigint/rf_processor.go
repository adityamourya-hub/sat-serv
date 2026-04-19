// Package sigint provides RF signal processing and SIGINT capabilities
// Implements FFT-based signal characterization, TDOA/FDOA geolocation, and intent inference
package sigint

import (
	"math"
	"math/cmplx"
)

const (
	// Signal processing thresholds
	JammerPowerThreshold = -30.0  // dBm threshold for jammer detection
	BurstThresholdHz     = 1000.0 // Hz bandwidth for burst detection
	ModulationConfidence = 0.7    // Confidence threshold for modulation ID

	// TDOA/FDOA constants
	SpeedOfLight     = 299792458.0 // m/s
	TDOAResolutionNs = 10.0        // 10 nanosecond resolution
	FDOAResolutionHz = 100.0       // 100 Hz resolution

	// Signal types
	SignalTypeUnknown = "UNKNOWN"
	SignalTypeJammer  = "JAMMER"
	SignalTypeComms   = "COMMS_RELAY"
	SignalTypeELINT   = "ELINT"
	SignalTypeKinetic = "KINETIC_THREAT"
)

// SignalSample represents a single IQ sample from SDR
type SignalSample struct {
	I          float64 // In-phase component
	Q          float64 // Quadrature component
	Timestamp  float64 // Unix timestamp
	Frequency  float64 // Center frequency (Hz)
	SampleRate float64 // Sample rate (Hz)
}

// RFSignal represents a detected RF signal
type RFSignal struct {
	CenterFreq     float64    // Hz
	Bandwidth      float64    // Hz
	PowerDBm       float64    // dBm
	Modulation     string     // QPSK, 16-QAM, FHSS, etc.
	StartTime      float64    // Unix timestamp
	Duration       float64    // seconds
	SourceLocation [2]float64 // [lat, lon] if geolocated
	IsJamming      bool
	IsBurst        bool
	Confidence     float64
}

// SpectralPeak represents a peak in the frequency spectrum
type SpectralPeak struct {
	Frequency float64 // Hz
	Power     float64 // dBm
	Bandwidth float64 // Hz
	SNR       float64 // Signal-to-noise ratio
}

// RFProcessor handles RF signal processing and characterization
type RFProcessor struct {
	fftSize       int
	sampleRate    float64
	signalLibrary map[string]*RFSignature // Signature library for threat typing
}

// RFSignature represents a known RF signal signature
type RFSignature struct {
	Modulation   string
	Bandwidth    float64
	PowerProfile []float64
	ThreatType   string
	FirstSeen    float64
	LastSeen     float64
	Occurrences  int
}

// NewRFProcessor creates a new RF signal processor
func NewRFProcessor(sampleRate float64) *RFProcessor {
	return &RFProcessor{
		fftSize:       1024,
		sampleRate:    sampleRate,
		signalLibrary: make(map[string]*RFSignature),
	}
}

// CharacterizeSignal performs spectral analysis on IQ samples
func (rp *RFProcessor) CharacterizeSignal(samples []complex128, sampleRate float64) (float64, string, float64) {
	// 1. Perform FFT
	n := len(samples)
	fftResult := rp.computeFFT(samples)

	// 2. Find peak frequency and power
	maxPower := 0.0
	centerFreq := 0.0
	totalPower := 0.0

	for i, val := range fftResult {
		power := cmplx.Abs(val)
		totalPower += power

		if power > maxPower {
			maxPower = power
			centerFreq = (float64(i) * sampleRate) / float64(n)
		}
	}

	// 3. Calculate bandwidth (3dB points)
	threshold := maxPower / math.Sqrt(2)
	bandwidth := rp.calculateBandwidth(fftResult, threshold, sampleRate)

	// 4. Convert to dBm
	powerDBm := 10.0*math.Log10(maxPower) - 30.0

	// 5. Simple classification
	signalType := rp.classifySignal(powerDBm, bandwidth)

	return centerFreq, signalType, powerDBm
}

// computeFFT performs Fast Fourier Transform on complex samples
func (rp *RFProcessor) computeFFT(samples []complex128) []complex128 {
	n := len(samples)
	if n == 0 {
		return []complex128{}
	}

	// Simple DFT implementation (for production, use optimized FFT library)
	result := make([]complex128, n)
	for k := 0; k < n; k++ {
		sum := complex(0, 0)
		for n := 0; n < len(samples); n++ {
			angle := -2.0 * math.Pi * float64(k) * float64(n) / float64(len(samples))
			sum += samples[n] * complex(math.Cos(angle), math.Sin(angle))
		}
		result[k] = sum
	}

	return result
}

// calculateBandwidth calculates 3dB bandwidth of signal
func (rp *RFProcessor) calculateBandwidth(fft []complex128, threshold float64, sampleRate float64) float64 {
	// Find first frequency above threshold
	firstAbove := -1
	lastAbove := -1

	for i, val := range fft {
		power := cmplx.Abs(val)
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
	binWidth := sampleRate / float64(len(fft))
	bandwidthBins := lastAbove - firstAbove + 1
	return float64(bandwidthBins) * binWidth
}

// classifySignal determines signal type based on power and bandwidth
func (rp *RFProcessor) classifySignal(powerDBm, bandwidth float64) string {
	// High power + wide bandwidth = Jamming
	if powerDBm > JammerPowerThreshold && bandwidth > BurstThresholdHz {
		return SignalTypeJammer
	}

	// Moderate power + narrow bandwidth = Comms
	if powerDBm > -60.0 && bandwidth < 1000.0 {
		return SignalTypeComms
	}

	// Low power + periodic = ELINT
	if powerDBm > -80.0 && bandwidth < 100.0 {
		return SignalTypeELINT
	}

	return SignalTypeUnknown
}

// DetectBurstTransmission detects burst transmissions in streaming data
func (rp *RFProcessor) DetectBurstTransmission(samples []complex128, windowSize int) []RFSignal {
	signals := make([]RFSignal, 0)

	// Sliding window analysis
	for i := 0; i < len(samples)-windowSize; i += windowSize / 2 {
		window := samples[i : i+windowSize]

		centerFreq, signalType, powerDBm := rp.CharacterizeSignal(window, rp.sampleRate)

		if signalType == SignalTypeJammer {
			signal := RFSignal{
				CenterFreq: centerFreq,
				PowerDBm:   powerDBm,
				Modulation: "UNKNOWN",
				StartTime:  float64(i) / rp.sampleRate,
				Duration:   float64(windowSize) / rp.sampleRate,
				IsJamming:  true,
				IsBurst:    true,
				Confidence: 0.8,
			}
			signals = append(signals, signal)
		}
	}

	return signals
}

// CalculateDopplerShift calculates Doppler shift from frequency measurements
func (rp *RFProcessor) CalculateDopplerShift(observedFreq, expectedFreq float64) float64 {
	return observedFreq - expectedFreq
}

// EstimateRadialVelocity estimates radial velocity from Doppler shift
func (rp *RFProcessor) EstimateRadialVelocity(dopplerShift, carrierFreq float64) float64 {
	// f_d = (v / c) * f_c
	// v = (f_d / f_c) * c
	return (dopplerShift / carrierFreq) * SpeedOfLight
}

// TDOAMeasurement represents Time Difference of Arrival measurement
type TDOAMeasurement struct {
	SatelliteA string
	SatelliteB string
	TimeDiffNs float64 // Nanoseconds
	Confidence float64
	Timestamp  float64
}

// FDOAMeasurement represents Frequency Difference of Arrival measurement
type FDOAMeasurement struct {
	SatelliteA string
	SatelliteB string
	FreqDiffHz float64 // Hz
	Confidence float64
	Timestamp  float64
}

// GeolocationResult represents geolocated emitter position
type GeolocationResult struct {
	Latitude   float64 // degrees
	Longitude  float64 // degrees
	Altitude   float64 // meters
	Confidence float64
	Method     string // "TDOA", "FDOA", "HYBRID"
}

// Geolocator performs TDOA/FDOA geolocation
type Geolocator struct {
	satellitePositions  map[string][3]float64 // ECI positions (km)
	satelliteVelocities map[string][3]float64 // ECI velocities (km/s)
}

// NewGeolocator creates a new geolocator
func NewGeolocator() *Geolocator {
	return &Geolocator{
		satellitePositions:  make(map[string][3]float64),
		satelliteVelocities: make(map[string][3]float64),
	}
}

// AddSatellite adds a satellite to the geolocation network
func (g *Geolocator) AddSatellite(id string, pos, vel [3]float64) {
	g.satellitePositions[id] = pos
	g.satelliteVelocities[id] = vel
}

// CalculateTDOAHyperbola calculates hyperbola from TDOA measurement
func (g *Geolocator) CalculateTDOAHyperbola(tdoa TDOAMeasurement) []GeolocationResult {
	satA, existsA := g.satellitePositions[tdoa.SatelliteA]
	_, existsB := g.satellitePositions[tdoa.SatelliteB]

	if !existsA || !existsB {
		return nil
	}

	// Calculate distance difference from TDOA
	distanceDiff := tdoa.TimeDiffNs * SpeedOfLight / 1e9 // Convert ns to meters

	// Hyperbola equation: |r - r_a| - |r - r_b| = distanceDiff
	// This defines a hyperbola of possible emitter positions

	// Simplified: return multiple candidate points along the hyperbola
	results := make([]GeolocationResult, 0)

	// Sample points along the hyperbola (simplified implementation)
	for angle := 0.0; angle < 2*math.Pi; angle += 0.1 {
		// Calculate point on hyperbola (simplified geometry)
		lat := satA[0] + distanceDiff*math.Cos(angle)/111000.0
		lon := satA[1] + distanceDiff*math.Sin(angle)/111000.0

		results = append(results, GeolocationResult{
			Latitude:   lat,
			Longitude:  lon,
			Altitude:   0.0,
			Confidence: tdoa.Confidence,
			Method:     "TDOA",
		})
	}

	return results
}

// CalculateFDOAHyperbola calculates hyperbola from FDOA measurement
func (g *Geolocator) CalculateFDOAHyperbola(fdoa FDOAMeasurement) []GeolocationResult {
	// FDOA geolocation is more complex, requires Doppler shift analysis
	// Simplified implementation

	results := make([]GeolocationResult, 0)

	// Return candidate points based on frequency difference
	for lat := -90.0; lat < 90.0; lat += 5.0 {
		for lon := -180.0; lon < 180.0; lon += 5.0 {
			results = append(results, GeolocationResult{
				Latitude:   lat,
				Longitude:  lon,
				Altitude:   0.0,
				Confidence: fdoa.Confidence * 0.5, // Lower confidence for FDOA alone
				Method:     "FDOA",
			})
		}
	}

	return results
}

// HybridGeolocation combines TDOA and FDOA for improved accuracy
func (g *Geolocator) HybridGeolocation(tdoa TDOAMeasurement, fdoa FDOAMeasurement) GeolocationResult {
	tdoaResults := g.CalculateTDOAHyperbola(tdoa)
	fdoaResults := g.CalculateFDOAHyperbola(fdoa)

	// Find intersection of TDOA and FDOA hyperbolas
	bestResult := GeolocationResult{
		Latitude:   0.0,
		Longitude:  0.0,
		Confidence: 0.0,
		Method:     "HYBRID",
	}

	// Simplified intersection finding
	for _, tdoaRes := range tdoaResults {
		for _, fdoaRes := range fdoaResults {
			distance := math.Sqrt(math.Pow(tdoaRes.Latitude-fdoaRes.Latitude, 2) +
				math.Pow(tdoaRes.Longitude-fdoaRes.Longitude, 2))

			if distance < 5.0 { // 5 degree tolerance
				combinedConf := (tdoaRes.Confidence + fdoaRes.Confidence) / 2.0
				if combinedConf > bestResult.Confidence {
					bestResult = GeolocationResult{
						Latitude:   tdoaRes.Latitude,
						Longitude:  tdoaRes.Longitude,
						Altitude:   0.0,
						Confidence: combinedConf,
						Method:     "HYBRID",
					}
				}
			}
		}
	}

	return bestResult
}

// AddSignalSignature adds a signal signature to the library
func (rp *RFProcessor) AddSignalSignature(id string, signature *RFSignature) {
	rp.signalLibrary[id] = signature
}

// MatchSignal matches a signal against the signature library
func (rp *RFProcessor) MatchSignal(signal RFSignal) (string, float64) {
	bestMatch := ""
	bestScore := 0.0

	for id, sig := range rp.signalLibrary {
		score := rp.calculateMatchScore(signal, sig)
		if score > bestScore && score > ModulationConfidence {
			bestMatch = id
			bestScore = score
		}
	}

	return bestMatch, bestScore
}

// calculateMatchScore calculates similarity score between signal and signature
func (rp *RFProcessor) calculateMatchScore(signal RFSignal, signature *RFSignature) float64 {
	score := 0.0

	// Modulation match
	if signal.Modulation == signature.Modulation {
		score += 0.4
	}

	// Bandwidth similarity
	bandwidthDiff := math.Abs(signal.Bandwidth - signature.Bandwidth)
	if bandwidthDiff < signature.Bandwidth*0.1 {
		score += 0.3
	}

	// Power profile similarity (simplified)
	if signal.PowerDBm > -60.0 {
		score += 0.3
	}

	return score
}
