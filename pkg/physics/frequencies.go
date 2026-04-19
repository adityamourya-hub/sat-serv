// Package physics provides frequency spectrum management and NTIA chart integration
// Implements regulatory filter for signal classification and threat assessment
package physics

import (
	"fmt"
	"math"
	"time"
)

// FrequencyBand represents a frequency band with regulatory information
type FrequencyBand struct {
	Name         string
	MinFreqHz    float64
	MaxFreqHz    float64
	CenterFreqHz float64
	BandwidthHz  float64
	PrimaryUse   string
	IsMilitary   bool
	IsRestricted bool
	IsSpace      bool
	IsGPS        bool
	IsSatcom     bool
	LicenseType  string
	ThreatLevel  int // 1-5
}

// NTIASpectrumChart represents the NTIA spectrum allocation chart
type NTIASpectrumChart struct {
	bands map[string]*FrequencyBand
}

// NewNTIASpectrumChart creates a new NTIA spectrum chart
func NewNTIASpectrumChart() *NTIASpectrumChart {
	chart := &NTIASpectrumChart{
		bands: make(map[string]*FrequencyBand),
	}
	chart.initializeBands()
	return chart
}

// initializeBands initializes the spectrum chart with standard bands
func (nsc *NTIASpectrumChart) initializeBands() {
	// GPS Bands
	nsc.addBand(&FrequencyBand{
		Name:         "GPS_L1",
		MinFreqHz:    1575.42e6 - 10.23e6,
		MaxFreqHz:    1575.42e6 + 10.23e6,
		CenterFreqHz: 1575.42e6,
		BandwidthHz:  20.46e6,
		PrimaryUse:   "GPS Navigation",
		IsMilitary:   false,
		IsRestricted: true,
		IsSpace:      true,
		IsGPS:        true,
		LicenseType:  "Federal",
		ThreatLevel:  5, // Critical - GPS spoofing
	})

	nsc.addBand(&FrequencyBand{
		Name:         "GPS_L2",
		MinFreqHz:    1227.60e6 - 10.23e6,
		MaxFreqHz:    1227.60e6 + 10.23e6,
		CenterFreqHz: 1227.60e6,
		BandwidthHz:  20.46e6,
		PrimaryUse:   "GPS Navigation",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		IsGPS:        true,
		LicenseType:  "Federal",
		ThreatLevel:  5,
	})

	nsc.addBand(&FrequencyBand{
		Name:         "GPS_L5",
		MinFreqHz:    1176.45e6 - 10.23e6,
		MaxFreqHz:    1176.45e6 + 10.23e6,
		CenterFreqHz: 1176.45e6,
		BandwidthHz:  20.46e6,
		PrimaryUse:   "GPS Navigation (Civil)",
		IsMilitary:   false,
		IsRestricted: true,
		IsSpace:      true,
		IsGPS:        true,
		LicenseType:  "Federal",
		ThreatLevel:  5,
	})

	// Space Operations Bands
	nsc.addBand(&FrequencyBand{
		Name:         "SPACE_OPS_2200",
		MinFreqHz:    2200.0e6,
		MaxFreqHz:    2290.0e6,
		CenterFreqHz: 2245.0e6,
		BandwidthHz:  90.0e6,
		PrimaryUse:   "Space Operations, Satellite Control",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		IsSatcom:     true,
		LicenseType:  "Federal",
		ThreatLevel:  3, // Medium - Space operations
	})

	nsc.addBand(&FrequencyBand{
		Name:         "SATCOM_UHF",
		MinFreqHz:    240.0e6,
		MaxFreqHz:    270.0e6,
		CenterFreqHz: 255.0e6,
		BandwidthHz:  30.0e6,
		PrimaryUse:   "Military Satellite Communications",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		IsSatcom:     true,
		LicenseType:  "Federal",
		ThreatLevel:  4, // High - Military SATCOM
	})

	nsc.addBand(&FrequencyBand{
		Name:         "SATCOM_SHF",
		MinFreqHz:    7.9e9,
		MaxFreqHz:    8.4e9,
		CenterFreqHz: 8.15e9,
		BandwidthHz:  500.0e6,
		PrimaryUse:   "Military Satellite Communications",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		IsSatcom:     true,
		LicenseType:  "Federal",
		ThreatLevel:  4,
	})

	// X-Band Radar
	nsc.addBand(&FrequencyBand{
		Name:         "X_BAND_RADAR",
		MinFreqHz:    8.5e9,
		MaxFreqHz:    10.5e9,
		CenterFreqHz: 9.5e9,
		BandwidthHz:  2.0e9,
		PrimaryUse:   "Radar, Space Tracking",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		LicenseType:  "Federal",
		ThreatLevel:  3,
	})

	// Ka-Band
	nsc.addBand(&FrequencyBand{
		Name:         "KA_BAND",
		MinFreqHz:    26.5e9,
		MaxFreqHz:    40.0e9,
		CenterFreqHz: 33.25e9,
		BandwidthHz:  13.5e9,
		PrimaryUse:   "Satellite Communications, Radar",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		IsSatcom:     true,
		LicenseType:  "Federal",
		ThreatLevel:  3,
	})

	// Civil Bands (for correlation)
	nsc.addBand(&FrequencyBand{
		Name:         "VHF_AM",
		MinFreqHz:    118.0e6,
		MaxFreqHz:    137.0e6,
		CenterFreqHz: 127.5e6,
		BandwidthHz:  19.0e6,
		PrimaryUse:   "Aircraft AM",
		IsMilitary:   false,
		IsRestricted: false,
		IsSpace:      false,
		LicenseType:  "Civil",
		ThreatLevel:  1, // Low - Civil aviation
	})

	nsc.addBand(&FrequencyBand{
		Name:         "VHF_FM",
		MinFreqHz:    137.0e6,
		MaxFreqHz:    174.0e6,
		CenterFreqHz: 155.5e6,
		BandwidthHz:  37.0e6,
		PrimaryUse:   "Aircraft FM",
		IsMilitary:   false,
		IsRestricted: false,
		IsSpace:      false,
		LicenseType:  "Civil",
		ThreatLevel:  1,
	})

	nsc.addBand(&FrequencyBand{
		Name:         "ADS_B_1090",
		MinFreqHz:    1090.0e6,
		MaxFreqHz:    1090.0e6,
		CenterFreqHz: 1090.0e6,
		BandwidthHz:  1.0e6,
		PrimaryUse:   "ADS-B Transponder",
		IsMilitary:   false,
		IsRestricted: false,
		IsSpace:      false,
		LicenseType:  "Civil",
		ThreatLevel:  1,
	})

	// ISM Bands
	nsc.addBand(&FrequencyBand{
		Name:         "ISM_2_4",
		MinFreqHz:    2.4e9,
		MaxFreqHz:    2.5e9,
		CenterFreqHz: 2.45e9,
		BandwidthHz:  100.0e6,
		PrimaryUse:   "Wi-Fi, Bluetooth",
		IsMilitary:   false,
		IsRestricted: false,
		IsSpace:      false,
		LicenseType:  "Unlicensed",
		ThreatLevel:  2, // Low-Medium - ISM interference
	})

	nsc.addBand(&FrequencyBand{
		Name:         "ISM_5_8",
		MinFreqHz:    5.725e9,
		MaxFreqHz:    5.875e9,
		CenterFreqHz: 5.8e9,
		BandwidthHz:  150.0e6,
		PrimaryUse:   "Wi-Fi",
		IsMilitary:   false,
		IsRestricted: false,
		IsSpace:      false,
		LicenseType:  "Unlicensed",
		ThreatLevel:  2,
	})

	// Jamming Bands (common jammer frequencies)
	nsc.addBand(&FrequencyBand{
		Name:         "JAMMER_X_BAND",
		MinFreqHz:    8.0e9,
		MaxFreqHz:    12.0e9,
		CenterFreqHz: 10.0e9,
		BandwidthHz:  4.0e9,
		PrimaryUse:   "Jamming (Adversary)",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Unknown",
		ThreatLevel:  5, // Critical - Jamming
	})

	nsc.addBand(&FrequencyBand{
		Name:         "JAMMER_KA_BAND",
		MinFreqHz:    26.0e9,
		MaxFreqHz:    40.0e9,
		CenterFreqHz: 33.0e9,
		BandwidthHz:  14.0e9,
		PrimaryUse:   "Jamming (Adversary)",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Unknown",
		ThreatLevel:  5,
	})

	// SIGINT EOB Bands - Adversary Electronic Signatures
	// Sever/Gorizont (Russian Strategic Troposcatter)
	nsc.addBand(&FrequencyBand{
		Name:         "SEVER_GORIZONT_S",
		MinFreqHz:    4.435e9,
		MaxFreqHz:    4.750e9,
		CenterFreqHz: 4.5925e9,
		BandwidthHz:  315.0e6,
		PrimaryUse:   "Russian Strategic Troposcatter Communications",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Adversary",
		ThreatLevel:  4, // High - Strategic command node
	})

	// Tactical Troposcatter (Over-the-horizon battlefield)
	nsc.addBand(&FrequencyBand{
		Name:         "TACTICAL_TROPO",
		MinFreqHz:    350.0e6,
		MaxFreqHz:    2.0e9,
		CenterFreqHz: 1.175e9,
		BandwidthHz:  1.65e9,
		PrimaryUse:   "Tactical Troposcatter Communications",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Adversary",
		ThreatLevel:  3, // Medium - Tactical coordination
	})

	// Orlan-10 UAV Telemetry
	nsc.addBand(&FrequencyBand{
		Name:         "ORLAN10_TELEMETRY",
		MinFreqHz:    850.0e6,
		MaxFreqHz:    930.0e6,
		CenterFreqHz: 890.0e6,
		BandwidthHz:  80.0e6,
		PrimaryUse:   "Orlan-10 UAV Command/Control",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Adversary",
		ThreatLevel:  3, // Medium - UAV coordination
	})

	// Ballistic Missile Telemetry (L-band)
	nsc.addBand(&FrequencyBand{
		Name:         "BALLISTIC_TELEM_L",
		MinFreqHz:    1.435e9,
		MaxFreqHz:    1.525e9,
		CenterFreqHz: 1.48e9,
		BandwidthHz:  90.0e6,
		PrimaryUse:   "Ballistic Missile Flight Test Telemetry",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		LicenseType:  "Adversary",
		ThreatLevel:  5, // Critical - Missile test/launch
	})

	// Ballistic Missile Telemetry (S-band)
	nsc.addBand(&FrequencyBand{
		Name:         "BALLISTIC_TELEM_S",
		MinFreqHz:    2.200e9,
		MaxFreqHz:    2.390e9,
		CenterFreqHz: 2.295e9,
		BandwidthHz:  190.0e6,
		PrimaryUse:   "Ballistic Missile Narrow-band Monitoring",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      true,
		LicenseType:  "Adversary",
		ThreatLevel:  5, // Critical - Missile activity
	})

	// PROFORMA Data Links (X-band)
	nsc.addBand(&FrequencyBand{
		Name:         "PROFORMA_X",
		MinFreqHz:    8.0e9,
		MaxFreqHz:    12.0e9,
		CenterFreqHz: 10.0e9,
		BandwidthHz:  4.0e9,
		PrimaryUse:   "PROFORMA Radar-to-Missile Data Link",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Adversary",
		ThreatLevel:  5, // Critical - Imminent engagement
	})

	// PROFORMA Data Links (Ka-band)
	nsc.addBand(&FrequencyBand{
		Name:         "PROFORMA_KA",
		MinFreqHz:    26.5e9,
		MaxFreqHz:    40.0e9,
		CenterFreqHz: 33.25e9,
		BandwidthHz:  13.5e9,
		PrimaryUse:   "PROFORMA High-bandwidth Data Link",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Adversary",
		ThreatLevel:  5, // Critical - Engagement imminent
	})

	// Link 16 (NATO/Allied - for correlation)
	nsc.addBand(&FrequencyBand{
		Name:         "LINK_16",
		MinFreqHz:    969.0e6,
		MaxFreqHz:    1206.0e6,
		CenterFreqHz: 1087.5e6,
		BandwidthHz:  237.0e6,
		PrimaryUse:   "Link 16 Tactical Data Link",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Allied",
		ThreatLevel:  2, // Low-Medium - Friendly force
	})

	// HAVE QUICK (Frequency hopping)
	nsc.addBand(&FrequencyBand{
		Name:         "HAVE_QUICK",
		MinFreqHz:    225.0e6,
		MaxFreqHz:    399.975e6,
		CenterFreqHz: 312.5e6,
		BandwidthHz:  174.975e6,
		PrimaryUse:   "HAVE QUICK Frequency Hopping",
		IsMilitary:   true,
		IsRestricted: true,
		IsSpace:      false,
		LicenseType:  "Allied",
		ThreatLevel:  2, // Low-Medium - Friendly comms
	})
}

// addBand adds a frequency band to the chart
func (nsc *NTIASpectrumChart) addBand(band *FrequencyBand) {
	nsc.bands[band.Name] = band
}

// FindBandByFrequency finds the band containing a given frequency
func (nsc *NTIASpectrumChart) FindBandByFrequency(freqHz float64) *FrequencyBand {
	for _, band := range nsc.bands {
		if freqHz >= band.MinFreqHz && freqHz <= band.MaxFreqHz {
			return band
		}
	}
	return nil
}

// ClassifySignal classifies a signal based on frequency
func (nsc *NTIASpectrumChart) ClassifySignal(freqHz float64, powerDBm float64) *SignalClassification {
	band := nsc.FindBandByFrequency(freqHz)

	classification := &SignalClassification{
		FrequencyHz:  freqHz,
		PowerDBm:     powerDBm,
		BandName:     "UNKNOWN",
		PrimaryUse:   "UNKNOWN",
		ThreatLevel:  2, // Default medium threat
		IsAdversary:  false,
		IsFriendly:   false,
		IsRestricted: false,
		IsSpace:      false,
	}

	if band != nil {
		classification.BandName = band.Name
		classification.PrimaryUse = band.PrimaryUse
		classification.ThreatLevel = band.ThreatLevel
		classification.IsRestricted = band.IsRestricted
		classification.IsSpace = band.IsSpace

		// Determine if adversarial
		if band.IsMilitary && band.LicenseType == "Unknown" {
			classification.IsAdversary = true
		}
		if band.Name == "JAMMER_X_BAND" || band.Name == "JAMMER_KA_BAND" {
			classification.IsAdversary = true
		}
		if band.IsGPS && powerDBm > -80.0 {
			// High power in GPS band could be spoofing
			classification.IsAdversary = true
		}

		// Determine if friendly
		if band.Name == "SPACE_OPS_2200" || band.Name == "SATCOM_UHF" {
			classification.IsFriendly = true
		}
		if band.Name == "ADS_B_1090" {
			classification.IsFriendly = true
		}
	}

	return classification
}

// SignalClassification represents the classification of a detected signal
type SignalClassification struct {
	FrequencyHz  float64
	PowerDBm     float64
	BandName     string
	PrimaryUse   string
	ThreatLevel  int
	IsAdversary  bool
	IsFriendly   bool
	IsRestricted bool
	IsSpace      bool
	Confidence   float64
}

// FrequencyRange represents a frequency range
type FrequencyRange struct {
	MinFreqHz float64
	MaxFreqHz float64
}

// GetMilitaryBands returns all military frequency bands
func (nsc *NTIASpectrumChart) GetMilitaryBands() []*FrequencyBand {
	bands := make([]*FrequencyBand, 0)
	for _, band := range nsc.bands {
		if band.IsMilitary {
			bands = append(bands, band)
		}
	}
	return bands
}

// GetSpaceBands returns all space-related frequency bands
func (nsc *NTIASpectrumChart) GetSpaceBands() []*FrequencyBand {
	bands := make([]*FrequencyBand, 0)
	for _, band := range nsc.bands {
		if band.IsSpace {
			bands = append(bands, band)
		}
	}
	return bands
}

// GetGPSBands returns all GPS frequency bands
func (nsc *NTIASpectrumChart) GetGPSBands() []*FrequencyBand {
	bands := make([]*FrequencyBand, 0)
	for _, band := range nsc.bands {
		if band.IsGPS {
			bands = append(bands, band)
		}
	}
	return bands
}

// GetSatcomBands returns all satellite communication bands
func (nsc *NTIASpectrumChart) GetSatcomBands() []*FrequencyBand {
	bands := make([]*FrequencyBand, 0)
	for _, band := range nsc.bands {
		if band.IsSatcom {
			bands = append(bands, band)
		}
	}
	return bands
}

// GetBandsByThreatLevel returns bands with threat level >= threshold
func (nsc *NTIASpectrumChart) GetBandsByThreatLevel(minLevel int) []*FrequencyBand {
	bands := make([]*FrequencyBand, 0)
	for _, band := range nsc.bands {
		if band.ThreatLevel >= minLevel {
			bands = append(bands, band)
		}
	}
	return bands
}

// FrequencyUtils provides utility functions for frequency calculations
type FrequencyUtils struct{}

// HzToMHz converts Hz to MHz
func (fu *FrequencyUtils) HzToMHz(hz float64) float64 {
	return hz / 1e6
}

// MHzToHz converts MHz to Hz
func (fu *FrequencyUtils) MHzToHz(mhz float64) float64 {
	return mhz * 1e6
}

// HzToGHz converts Hz to GHz
func (fu *FrequencyUtils) HzToGHz(hz float64) float64 {
	return hz / 1e9
}

// GHzToHz converts GHz to Hz
func (fu *FrequencyUtils) GHzToHz(ghz float64) float64 {
	return ghz * 1e9
}

// CalculateWavelength calculates wavelength from frequency
func (fu *FrequencyUtils) CalculateWavelength(freqHz float64) float64 {
	const speedOfLight = 299792458.0
	return speedOfLight / freqHz
}

// dBmToWatts converts dBm to Watts
func (fu *FrequencyUtils) dBmToWatts(dBm float64) float64 {
	return math.Pow(10, (dBm-30)/10)
}

// WattsTodBm converts Watts to dBm
func (fu *FrequencyUtils) WattsTodBm(watts float64) float64 {
	return 10*math.Log10(watts) + 30
}

// CalculateFreeSpacePathLoss calculates free space path loss
func (fu *FrequencyUtils) CalculateFreeSpacePathLoss(freqHz float64, distanceMeters float64) float64 {
	// FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
	// Simplified: FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.44 (d in km, f in MHz)
	distanceKm := distanceMeters / 1000.0
	freqMHz := freqHz / 1e6

	return 20*math.Log10(distanceKm) + 20*math.Log10(freqMHz) + 32.44
}

// CalculateSignalPowerAtReceiver calculates received power
func (fu *FrequencyUtils) CalculateSignalPowerAtReceiver(transmitPowerWatts, distanceMeters, freqHz float64) float64 {
	pathLoss := fu.CalculateFreeSpacePathLoss(freqHz, distanceMeters)
	receivedPower := transmitPowerWatts / math.Pow(10, pathLoss/10)
	return receivedPower
}

// SpectrumMonitor represents a spectrum monitoring session
type SpectrumMonitor struct {
	chart        *NTIASpectrumChart
	observations map[string]int // Band name -> count
}

// NewSpectrumMonitor creates a new spectrum monitor
func NewSpectrumMonitor() *SpectrumMonitor {
	return &SpectrumMonitor{
		chart:        NewNTIASpectrumChart(),
		observations: make(map[string]int),
	}
}

// ObserveSignal observes a signal and updates statistics
func (sm *SpectrumMonitor) ObserveSignal(freqHz, powerDBm float64) {
	classification := sm.chart.ClassifySignal(freqHz, powerDBm)

	if classification.BandName != "UNKNOWN" {
		sm.observations[classification.BandName]++
	}
}

// GetBandStatistics returns observation statistics by band
func (sm *SpectrumMonitor) GetBandStatistics() map[string]int {
	return sm.observations
}

// GetAdversaryBandCount returns count of adversary band observations
func (sm *SpectrumMonitor) GetAdversaryBandCount() int {
	count := 0
	for bandName, band := range sm.chart.bands {
		if band.IsMilitary && band.LicenseType == "Unknown" {
			count += sm.observations[bandName]
		}
		if band.Name == "JAMMER_X_BAND" || band.Name == "JAMMER_KA_BAND" {
			count += sm.observations[bandName]
		}
	}
	return count
}

// GetGPSAnomalyCount returns count of GPS band observations with high power
func (sm *SpectrumMonitor) GetGPSAnomalyCount() int {
	// In production, track power levels
	return 0
}

// SpectrumAlert represents a spectrum alert
type SpectrumAlert struct {
	AlertID     string
	BandName    string
	AlertType   string
	Severity    string
	Description string
	Timestamp   float64
}

// GenerateAlerts generates alerts based on spectrum observations
func (sm *SpectrumMonitor) GenerateAlerts() []*SpectrumAlert {
	alerts := make([]*SpectrumAlert, 0)

	// Check for adversary band activity
	adversaryCount := sm.GetAdversaryBandCount()
	if adversaryCount > 10 {
		alerts = append(alerts, &SpectrumAlert{
			AlertID:     fmt.Sprintf("SPECTRUM-ALERT-%d", time.Now().Unix()),
			BandName:    "MULTIPLE_ADVERSARY",
			AlertType:   "ADVERSARY_ACTIVITY",
			Severity:    "HIGH",
			Description: fmt.Sprintf("High adversary band activity detected: %d observations", adversaryCount),
			Timestamp:   float64(time.Now().Unix()),
		})
	}

	return alerts
}

// Helper function for timestamp
func timeNow() float64 {
	return float64(time.Now().Unix())
}
