// pkg/ml/dmd/tripwire.go
// ADITI 4.0: Dynamic Mode Decomposition (DMD) Hypersonic Tripwire
//
// Identifies coherent structures in plasma wake while FNO handles bulk density.
// Detects Kármán Vortex Street and Bow Shock oscillations to trigger
// Tier 3 AI-IMM reasoning when HGV transitions from glide to maneuver.

package dmd

import (
	"fmt"
	"math"
	"sort"
)

// DMDMode represents a single DMD mode with its properties
type DMDMode struct {
	Eigenvalue complex128 // DMD eigenvalue λ = μ + iω
	Mode       []float64  // Spatial mode (DMD mode vector)
	Energy     float64    // Mode energy ||Mode||²
	Frequency  float64    // Oscillation frequency (Hz)
	GrowthRate float64    // Temporal growth/decay rate
}

// IsHighFrequency returns true if mode frequency > threshold
// Typical HGV bow shock: 0.5-5 Hz
func (m *DMDMode) IsHighFrequency(threshold float64) bool {
	return math.Abs(m.Frequency) > threshold
}

// IsGrowing returns true if mode is unstable (growing)
func (m *DMDMode) IsGrowing() bool {
	return m.GrowthRate > 0
}

// DMDAnalyzer performs Dynamic Mode Decomposition on spectral feeds
type DMDAnalyzer struct {
	// Data matrices
	SnapshotMatrix [][]float64 // Columns = time snapshots, rows = spectral bins

	// DMD parameters
	NumModes     int     // Number of DMD modes to retain
	SVDThreshold float64 // SVD truncation threshold

	// Tripwire thresholds
	ManeuverThreshold float64 // Energy threshold for "maneuver mode"
	HighFreqThreshold float64 // Frequency threshold for high-frequency detection
	BowShockThreshold float64 // Energy threshold for bow shock detection

	// Mode indices (populated after analysis)
	ManeuverIdx int // Index of maneuver mode
	BowShockIdx int // Index of bow shock mode
	GlideIdx    int // Index of stable glide mode

	// History for temporal analysis
	ModeHistory []DMDSnapshot
}

// DMDSnapshot represents DMD state at single time point
type DMDSnapshot struct {
	Timestamp    int64
	Modes        []DMDMode
	DominantMode int
	TotalEnergy  float64
}

// NewDMDAnalyzer creates analyzer for hypersonic detection
func NewDMDAnalyzer() *DMDAnalyzer {
	return &DMDAnalyzer{
		NumModes:          10,   // Retain top 10 modes
		SVDThreshold:      1e-6, // SVD truncation
		ManeuverThreshold: 0.3,  // 30% energy for maneuver detection
		HighFreqThreshold: 0.5,  // 0.5 Hz threshold
		BowShockThreshold: 0.2,  // 20% energy for bow shock
		SnapshotMatrix:    make([][]float64, 0),
		ModeHistory:       make([]DMDSnapshot, 0),
		ManeuverIdx:       -1,
		BowShockIdx:       -1,
		GlideIdx:          -1,
	}
}

// AddSnapshot adds new spectral measurement to data matrix
func (d *DMDAnalyzer) AddSnapshot(spectralData []float64, timestamp int64) {
	// Add as column to snapshot matrix
	// Transpose: Each row is a spectral bin, each column is time
	if len(d.SnapshotMatrix) == 0 {
		// Initialize rows
		d.SnapshotMatrix = make([][]float64, len(spectralData))
		for i := range d.SnapshotMatrix {
			d.SnapshotMatrix[i] = make([]float64, 0)
		}
	}

	// Add data
	for i, val := range spectralData {
		if i < len(d.SnapshotMatrix) {
			d.SnapshotMatrix[i] = append(d.SnapshotMatrix[i], val)
		}
	}

	// Maintain window size (keep last 100 snapshots)
	maxSnapshots := 100
	for i := range d.SnapshotMatrix {
		if len(d.SnapshotMatrix[i]) > maxSnapshots {
			d.SnapshotMatrix[i] = d.SnapshotMatrix[i][1:]
		}
	}
}

// ComputeModes performs DMD analysis on snapshot matrix
// Returns list of DMD modes sorted by energy
func (d *DMDAnalyzer) ComputeModes() []DMDMode {
	if len(d.SnapshotMatrix) == 0 || len(d.SnapshotMatrix[0]) < 2 {
		return nil
	}

	// DMD Algorithm (simplified)
	// 1. Build X1 and X2 matrices (snapshots 1..n-1 and 2..n)
	// 2. Compute SVD of X1
	// 3. Truncate and compute Atilde
	// 4. Eigen-decomposition of Atilde
	// 5. Reconstruct DMD modes

	// For now, return synthetic modes based on spectral content
	return d.computeSyntheticModes()
}

// computeSyntheticModes generates modes from spectral analysis
// In production, this would perform full SVD-DMD
func (d *DMDAnalyzer) computeSyntheticModes() []DMDMode {
	modes := make([]DMDMode, 0)

	if len(d.SnapshotMatrix) == 0 {
		return modes
	}

	// Analyze each spectral bin as a "mode"
	for _, binData := range d.SnapshotMatrix {
		if len(binData) < 2 {
			continue
		}

		// Compute energy
		energy := 0.0
		for _, val := range binData {
			energy += val * val
		}

		// Estimate frequency from zero crossings
		freq := estimateFrequency(binData)

		// Compute growth rate (linear fit to log amplitude)
		growth := estimateGrowthRate(binData)

		mode := DMDMode{
			Eigenvalue: complex(growth, 2*math.Pi*freq),
			Mode:       binData,
			Energy:     energy,
			Frequency:  freq,
			GrowthRate: growth,
		}

		modes = append(modes, mode)
	}

	// Sort by energy descending
	sort.Slice(modes, func(i, j int) bool {
		return modes[i].Energy > modes[j].Energy
	})

	// Truncate to NumModes
	if len(modes) > d.NumModes {
		modes = modes[:d.NumModes]
	}

	// Identify mode types
	d.identifyModeTypes(modes)

	return modes
}

// identifyModeTypes categorizes modes by frequency/energy characteristics
func (d *DMDAnalyzer) identifyModeTypes(modes []DMDMode) {
	for i, mode := range modes {
		// Bow shock: High frequency (>0.5 Hz), moderate energy
		if mode.IsHighFrequency(d.HighFreqThreshold) && mode.Energy > d.BowShockThreshold {
			d.BowShockIdx = i
		}

		// Maneuver: Growing mode with significant energy
		if mode.IsGrowing() && mode.Energy > d.ManeuverThreshold {
			d.ManeuverIdx = i
		}

		// Glide: Low frequency, stable, high energy
		if !mode.IsHighFrequency(0.1) && !mode.IsGrowing() && mode.Energy > 0.4 {
			d.GlideIdx = i
		}
	}
}

// estimateFrequency estimates dominant frequency from time series
func estimateFrequency(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Count zero crossings
	crossings := 0
	for i := 1; i < len(data); i++ {
		if data[i-1]*data[i] < 0 {
			crossings++
		}
	}

	// Frequency = crossings / (2 * duration)
	// Assuming 10 Hz sampling rate (typical for HGV tracking)
	duration := float64(len(data)) / 10.0 // seconds
	if duration > 0 {
		return float64(crossings) / (2.0 * duration)
	}
	return 0.0
}

// estimateGrowthRate estimates exponential growth/decay rate
func estimateGrowthRate(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Fit log(|data|) = a + b*t
	// Growth rate = b
	logData := make([]float64, 0, len(data))
	for _, val := range data {
		if val != 0 {
			logData = append(logData, math.Log(math.Abs(val)))
		}
	}

	if len(logData) < 2 {
		return 0.0
	}

	// Linear regression for slope
	n := float64(len(logData))
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0

	for i, y := range logData {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	// Convert to growth rate (per second, assuming 10 Hz)
	return slope * 10.0
}

// CheckTripwire analyzes spectral feed for hypersonic maneuver detection
//
// TRIGGER: When "Maneuver Mode" (λ_maneuver) dominates spectrum
// This signals target is no longer "gliding" but "shedding energy"
//
// Returns true if Tier 3 AI-IMM reasoning should be triggered
func (d *DMDAnalyzer) CheckTripwire(spectralFeed []float64, timestamp int64) bool {
	// Add to snapshot matrix
	d.AddSnapshot(spectralFeed, timestamp)

	// Need minimum data for DMD
	if len(d.SnapshotMatrix) == 0 || len(d.SnapshotMatrix[0]) < 10 {
		return false
	}

	// Compute DMD modes
	modes := d.ComputeModes()
	if len(modes) == 0 {
		return false
	}

	// Calculate total energy
	totalEnergy := 0.0
	for _, mode := range modes {
		totalEnergy += mode.Energy
	}

	// Check for maneuver mode dominance
	maneuverDetected := false
	if d.ManeuverIdx >= 0 && d.ManeuverIdx < len(modes) {
		maneuverMode := modes[d.ManeuverIdx]
		maneuverEnergyRatio := maneuverMode.Energy / totalEnergy

		// TRIGGER: Maneuver mode > threshold AND growing
		if maneuverEnergyRatio > d.ManeuverThreshold && maneuverMode.IsGrowing() {
			maneuverDetected = true
			fmt.Printf("[DMD_TRIPWIRE] MANEUVER DETECTED: Mode %d, Energy %.2f%%, Growth %.3f/s\n",
				d.ManeuverIdx, maneuverEnergyRatio*100, maneuverMode.GrowthRate)
		}
	}

	// Check for bow shock (high-frequency oscillation)
	bowShockDetected := false
	if d.BowShockIdx >= 0 && d.BowShockIdx < len(modes) {
		bowShockMode := modes[d.BowShockIdx]
		bowShockEnergyRatio := bowShockMode.Energy / totalEnergy

		if bowShockEnergyRatio > d.BowShockThreshold {
			bowShockDetected = true
			fmt.Printf("[DMD_TRIPWIRE] BOW SHOCK: Mode %d, Freq %.2f Hz, Energy %.2f%%\n",
				d.BowShockIdx, bowShockMode.Frequency, bowShockEnergyRatio*100)
		}
	}

	// Store snapshot
	snapshot := DMDSnapshot{
		Timestamp:    timestamp,
		Modes:        modes,
		DominantMode: 0,
		TotalEnergy:  totalEnergy,
	}
	if len(modes) > 0 {
		snapshot.DominantMode = 0 // Highest energy mode
	}

	d.ModeHistory = append(d.ModeHistory, snapshot)
	if len(d.ModeHistory) > 100 {
		d.ModeHistory = d.ModeHistory[1:]
	}

	// TRIPWIRE: Either maneuver or strong bow shock
	return maneuverDetected || bowShockDetected
}

// GetManeuverConfidence returns confidence level (0-1) for maneuver detection
// Based on recent history of mode energies
func (d *DMDAnalyzer) GetManeuverConfidence() float64 {
	if len(d.ModeHistory) < 5 {
		return 0.0
	}

	// Count recent detections
	recentDetections := 0
	checkWindow := 10

	if len(d.ModeHistory) < checkWindow {
		checkWindow = len(d.ModeHistory)
	}

	for i := len(d.ModeHistory) - checkWindow; i < len(d.ModeHistory); i++ {
		snapshot := d.ModeHistory[i]

		if d.ManeuverIdx >= 0 && d.ManeuverIdx < len(snapshot.Modes) {
			mode := snapshot.Modes[d.ManeuverIdx]
			if mode.IsGrowing() && mode.Energy/snapshot.TotalEnergy > d.ManeuverThreshold {
				recentDetections++
			}
		}
	}

	// Confidence = fraction of recent snapshots showing maneuver
	return float64(recentDetections) / float64(checkWindow)
}

// GetModeSignature returns string signature of current dynamics
// "GLIDE", "MANEUVER", "BOW_SHOCK", "TURBULENT"
func (d *DMDAnalyzer) GetModeSignature() string {
	if len(d.ModeHistory) == 0 {
		return "UNKNOWN"
	}

	snapshot := d.ModeHistory[len(d.ModeHistory)-1]

	// Check for maneuver mode
	if d.ManeuverIdx >= 0 && d.ManeuverIdx < len(snapshot.Modes) {
		mode := snapshot.Modes[d.ManeuverIdx]
		if mode.IsGrowing() && mode.Energy/snapshot.TotalEnergy > d.ManeuverThreshold {
			return "MANEUVER"
		}
	}

	// Check for bow shock
	if d.BowShockIdx >= 0 && d.BowShockIdx < len(snapshot.Modes) {
		mode := snapshot.Modes[d.BowShockIdx]
		if mode.IsHighFrequency(d.HighFreqThreshold) {
			return "BOW_SHOCK"
		}
	}

	// Check for stable glide
	if d.GlideIdx >= 0 && d.GlideIdx < len(snapshot.Modes) {
		mode := snapshot.Modes[d.GlideIdx]
		if !mode.IsGrowing() && !mode.IsHighFrequency(0.1) {
			return "GLIDE"
		}
	}

	return "TURBULENT"
}

// HypersonicTripwire provides high-level interface for DMD-based detection
type HypersonicTripwire struct {
	Analyzer            *DMDAnalyzer
	LastTriggerTime     int64
	CooldownMs          int64 // Minimum time between triggers
	ConsecutiveTriggers int   // For confidence accumulation
}

// NewHypersonicTripwire creates tripwire for HGV detection
func NewHypersonicTripwire() *HypersonicTripwire {
	return &HypersonicTripwire{
		Analyzer:            NewDMDAnalyzer(),
		CooldownMs:          1000, // 1 second cooldown
		ConsecutiveTriggers: 0,
	}
}

// Process analyzes spectral feed and returns trigger status
// Returns: (shouldTrigger, confidence, signature)
func (ht *HypersonicTripwire) Process(spectralFeed []float64, timestamp int64) (bool, float64, string) {
	// Check cooldown
	if timestamp-ht.LastTriggerTime < ht.CooldownMs {
		return false, ht.Analyzer.GetManeuverConfidence(), ht.Analyzer.GetModeSignature()
	}

	// Run DMD analysis
	triggered := ht.Analyzer.CheckTripwire(spectralFeed, timestamp)
	confidence := ht.Analyzer.GetManeuverConfidence()
	signature := ht.Analyzer.GetModeSignature()

	if triggered {
		ht.LastTriggerTime = timestamp
		ht.ConsecutiveTriggers++

		// Require 2+ consecutive triggers for high confidence
		if ht.ConsecutiveTriggers >= 2 {
			return true, confidence, signature
		}
	} else {
		ht.ConsecutiveTriggers = 0
	}

	return false, confidence, signature
}
