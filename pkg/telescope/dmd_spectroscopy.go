// pkg/telescope/dmd_spectroscopy.go
// ADITI 4.0: DMD Spectroscopy Trigger for Sensor 1 (Narrow FOV)
//
// Sensor 1: Narrow FOV (high-mag) → DMD Spectroscopy
// Physics: High-res spectra for material ID (SLiPy + e3nn irrep tensors)
// DMD decomposes coherent modes in fractional-smoothed velocity field.
//
// Trigger: Fired when DMD detects coherent plasma mode in spectral feed
// Integration: SLiPy spectroscopy controller via ZMQ

package telescope

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/ml/dmd"
)

// DMDSpectroscopyTrigger manages SLiPy integration for material ID
type DMDSpectroscopyTrigger struct {
	dmdAnalyzer  *dmd.DMDAnalyzer
	spectroscopy *SpectroscopyHandoff

	// Trigger thresholds
	energyThreshold float64 // Minimum DMD mode energy for trigger
	freqMin         float64 // Hz - minimum frequency for plasma detection
	freqMax         float64 // Hz - maximum frequency for plasma detection

	// Integration control
	integrationTime time.Duration // 2-second default
	cooldownTime    time.Duration // Prevent re-trigger

	lastTriggerTime int64
	triggerCount    int64
}

// NewDMDSpectroscopyTrigger creates trigger for narrow-FOV sensor
func NewDMDSpectroscopyTrigger() *DMDSpectroscopyTrigger {
	return &DMDSpectroscopyTrigger{
		dmdAnalyzer:     dmd.NewDMDAnalyzer(),
		spectroscopy:    &SpectroscopyHandoff{},
		energyThreshold: 0.3,             // 30% of total spectral energy
		freqMin:         0.5,             // 0.5 Hz (typical HGV bow shock)
		freqMax:         5.0,             // 5 Hz (upper plasma oscillation)
		integrationTime: 2 * time.Second, // SLiPy integration window
		cooldownTime:    5 * time.Second, // Minimum between triggers
		lastTriggerTime: 0,
		triggerCount:    0,
	}
}

// ProcessSpectralFeed analyzes spectral data and triggers if DMD criteria met
//
// ADITI 4.0: DMD identifies coherent plasma modes that indicate:
//   - HGV re-entry wakes
//   - Maneuver signatures
//   - Material chemistry (via spectral fingerprint)
func (dt *DMDSpectroscopyTrigger) ProcessSpectralFeed(
	spectralData []float64,
	timestamp int64,
	targetID string,
) (*dmd.DMDSnapshot, bool, error) {

	// Check cooldown
	now := time.Now().UnixMilli()
	if now-dt.lastTriggerTime < dt.cooldownTime.Milliseconds() {
		return nil, false, nil // Still in cooldown
	}

	// Run DMD analysis
	dt.dmdAnalyzer.AddSnapshot(spectralData, timestamp)
	modes := dt.dmdAnalyzer.ComputeModes()

	if len(modes) == 0 {
		return nil, false, nil
	}

	// Check for coherent plasma modes
	plasmaMode := dt.findPlasmaMode(modes)

	if plasmaMode != nil {
		// Trigger spectroscopy handoff
		fmt.Printf("[DMD_SPEC] Plasma mode detected: Freq=%.2fHz, Energy=%.2f%%\n",
			plasmaMode.Frequency, plasmaMode.Energy*100)

		// Initiate SLiPy integration
		dt.triggerSpectroscopy(targetID, plasmaMode)

		dt.lastTriggerTime = now
		dt.triggerCount++

		return &dmd.DMDSnapshot{
			Timestamp:    timestamp,
			Modes:        modes,
			DominantMode: 0,
			TotalEnergy:  plasmaMode.Energy,
		}, true, nil
	}

	return nil, false, nil
}

// findPlasmaMode identifies DMD mode matching plasma signature criteria
//
// Criteria:
//   - Frequency in [0.5, 5.0] Hz (HGV bow shock / plasma oscillation)
//   - Energy > 30% of total
//   - Growing or stable (not decaying)
func (dt *DMDSpectroscopyTrigger) findPlasmaMode(modes []dmd.DMDMode) *dmd.DMDMode {
	for _, mode := range modes {
		freq := abs(mode.Frequency)

		// Check frequency range
		if freq < dt.freqMin || freq > dt.freqMax {
			continue
		}

		// Check energy threshold
		if mode.Energy < dt.energyThreshold {
			continue
		}

		// Prefer growing or stable modes
		if mode.GrowthRate < -0.1 {
			continue // Decaying too fast
		}

		return &mode
	}

	return nil
}

// triggerSpectroscopy initiates SLiPy 2-second integration
//
// "Hot" handoff: Telescope stays in ClosedLoopTrack while
// spectrometer integrates. Results fused into ObservationReport.
func (dt *DMDSpectroscopyTrigger) triggerSpectroscopy(targetID string, mode *dmd.DMDMode) error {
	// Compute DMD confidence
	confidence := mode.Energy

	// Send ZMQ trigger to SLiPy
	if err := dt.spectroscopy.TriggerHandoff(targetID, confidence); err != nil {
		return fmt.Errorf("spectroscopy handoff failed: %w", err)
	}

	fmt.Printf("[DMD_SPEC] Triggered SLiPy integration for %s (%.2f sec)\n",
		targetID, dt.integrationTime.Seconds())

	return nil
}

// ReceiveSpectralData waits for SLiPy integration to complete
//
// Call this after integrationTime has elapsed
func (dt *DMDSpectroscopyTrigger) ReceiveSpectralData() (*SpectroscopyData, error) {
	return dt.spectroscopy.ReceiveData()
}

// FuseWithReport combines spectral data into ObservationReport
//
// Uses e3nn material ID to classify detected plasma/material
func (dt *DMDSpectroscopyTrigger) FuseWithReport(
	report *ObservationReport,
	specData *SpectroscopyData,
) error {
	// Set spectroscopy data
	report.Spectroscopy = specData

	// Classify based on plasma detection
	if specData.PlasmaDetected {
		report.SpectralClass = fmt.Sprintf("HGV-Plasma-%s", specData.MaterialID)
		report.ThreatLevel = 4 // CRITICAL

		// Set MFBD silhouette hash if available
		if len(report.MFBDsilhouetteHash) == 0 {
			report.MFBDsilhouetteHash = []byte(fmt.Sprintf("0xPLASMA_%s", specData.MaterialID))
		}
	}

	// Update visual magnitude from SWIR intensity
	if specData.SWIRIntensity > 0 {
		// Approximate magnitude from intensity
		swir := float64(specData.SWIRIntensity)
		report.VisualMagnitude = float32(-2.5*log10(swir/100.0) + 10.0)
	}

	fmt.Printf("[DMD_SPEC] Fused spectroscopy: Material=%s, Temp=%.0fK, Threat=%d\n",
		specData.MaterialID, specData.PlasmaTemperature, report.ThreatLevel)

	return nil
}

// GetTriggerStats returns DMD trigger performance
func (dt *DMDSpectroscopyTrigger) GetTriggerStats() (count int64, lastTrigger int64) {
	return dt.triggerCount, dt.lastTriggerTime
}

// abs returns absolute value of float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// log10 computes base-10 logarithm
func log10(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Use natural log and convert: log10(x) = ln(x) / ln(10)
	const ln10 = 2.302585092994046
	return ln(x) / ln10
}

// ln computes natural logarithm using math package
func ln(x float64) float64 {
	return math.Log(x)
}
