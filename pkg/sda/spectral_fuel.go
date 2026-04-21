package sda

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Spectral-to-Fuel Lookup Table
 *
 * Maps detected spectral lines (Xenon, Hydrazine, Cold Gas) directly to
 * propulsion parameters (Isp, thrust, burn rate) for automatic engine type
 * identification and fuel model updates.
 *
 * This enables "Propulsion Locking" - automatic detection of propulsion
 * type from 1D slitless spectra, eliminating manual classification.
 */

// SpectralLine represents a detected atomic/molecular emission line
type SpectralLine struct {
	WavelengthNm float64 // Wavelength in nanometers
	Intensity    float64 // Relative intensity (0-1)
	Element      string  // Element name (e.g., "Xe", "N2H4", "N2")
}

// PropulsionSignature defines the spectral signature of a propulsion type
type PropulsionSignature struct {
	EngineType    EngineType
	SpectralLines []SpectralLine
	Isp           float32    // Specific impulse (seconds)
	ThrustRange   [2]float32 // Min/max thrust (N)
	BurnRate      float32    // Typical burn rate (m/s²)
	Confidence    float32    // Detection confidence threshold
}

// SpectralToFuelTable maps spectral signatures to propulsion parameters
type SpectralToFuelTable struct {
	signatures map[string]PropulsionSignature // Key: element name
}

// NewSpectralToFuelTable creates a new spectral-to-fuel lookup table
func NewSpectralToFuelTable() *SpectralToFuelTable {
	table := &SpectralToFuelTable{
		signatures: make(map[string]PropulsionSignature),
	}
	table.initializeDefaultSignatures()
	return table
}

// initializeDefaultSignatures populates the table with known propulsion signatures
func (t *SpectralToFuelTable) initializeDefaultSignatures() {
	// Xenon (Ion Propulsion)
	// Characteristic lines: 828.0 nm, 834.7 nm, 881.9 nm
	t.signatures["Xe"] = PropulsionSignature{
		EngineType: EngineElectric,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 828.0, Intensity: 0.9, Element: "Xe"},
			{WavelengthNm: 834.7, Intensity: 0.85, Element: "Xe"},
			{WavelengthNm: 881.9, Intensity: 0.8, Element: "Xe"},
		},
		Isp:         3000.0,                // High Isp for ion propulsion
		ThrustRange: [2]float32{0.05, 0.5}, // Low thrust
		BurnRate:    0.001,                 // Very low burn rate
		Confidence:  0.95,
	}

	// Hydrazine (Chemical Monoprop)
	// Characteristic lines: 336.1 nm (N2), 337.1 nm (NH)
	t.signatures["N2H4"] = PropulsionSignature{
		EngineType: EngineChemical,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 336.1, Intensity: 0.8, Element: "N2"},
			{WavelengthNm: 337.1, Intensity: 0.75, Element: "NH"},
		},
		Isp:         230.0,                // Moderate Isp for monoprop
		ThrustRange: [2]float32{0.5, 5.0}, // Moderate thrust
		BurnRate:    0.05,                 // Moderate burn rate
		Confidence:  0.9,
	}

	// Nitrogen Tetroxide/MMH (Biprop)
	// Characteristic lines: NO bands at 226-230 nm, CN bands at 388 nm
	t.signatures["N2O4"] = PropulsionSignature{
		EngineType: EngineChemical,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 226.0, Intensity: 0.7, Element: "NO"},
			{WavelengthNm: 230.0, Intensity: 0.65, Element: "NO"},
			{WavelengthNm: 388.0, Intensity: 0.6, Element: "CN"},
		},
		Isp:         320.0,                 // High Isp for biprop
		ThrustRange: [2]float32{5.0, 50.0}, // High thrust
		BurnRate:    0.1,                   // High burn rate
		Confidence:  0.85,
	}

	// Nitrogen (Cold Gas)
	// Characteristic lines: 337.1 nm (N2 first positive system)
	t.signatures["N2"] = PropulsionSignature{
		EngineType: EngineColdGas,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 337.1, Intensity: 0.6, Element: "N2"},
		},
		Isp:         70.0,                  // Low Isp for cold gas
		ThrustRange: [2]float32{0.01, 0.1}, // Very low thrust
		BurnRate:    0.005,                 // Very low burn rate
		Confidence:  0.8,
	}

	// Krypton (Alternative Ion Propulsion)
	// Characteristic lines: 760.2 nm, 819.0 nm
	t.signatures["Kr"] = PropulsionSignature{
		EngineType: EngineElectric,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 760.2, Intensity: 0.85, Element: "Kr"},
			{WavelengthNm: 819.0, Intensity: 0.8, Element: "Kr"},
		},
		Isp:         2500.0,                // High Isp (slightly lower than Xenon)
		ThrustRange: [2]float32{0.05, 0.5}, // Low thrust
		BurnRate:    0.001,                 // Very low burn rate
		Confidence:  0.85,
	}
}

// IdentifyPropulsion identifies propulsion type from detected spectral lines
// Returns the best-matching propulsion signature and confidence score
func (t *SpectralToFuelTable) IdentifyPropulsion(detectedLines []SpectralLine) (PropulsionSignature, float32) {
	bestMatch := PropulsionSignature{EngineType: EnginePassive}
	bestScore := float32(0.0)

	// Score each signature against detected lines
	for element, signature := range t.signatures {
		score := t.scoreSignature(detectedLines, signature, element)
		if score > bestScore && score >= signature.Confidence {
			bestMatch = signature
			bestScore = score
		}
	}

	return bestMatch, bestScore
}

// scoreSignature calculates the match score between detected lines and a signature
func (t *SpectralToFuelTable) scoreSignature(detectedLines []SpectralLine, signature PropulsionSignature, element string) float32 {
	if len(detectedLines) == 0 || len(signature.SpectralLines) == 0 {
		return 0.0
	}

	matches := 0
	totalIntensity := float32(0.0)
	targetIntensity := float32(0.0)

	// Check for wavelength matches (within 0.5 nm tolerance)
	for _, detected := range detectedLines {
		for _, reference := range signature.SpectralLines {
			wavelengthDiff := detected.WavelengthNm - reference.WavelengthNm
			if wavelengthDiff < 0 {
				wavelengthDiff = -wavelengthDiff
			}

			// Match if wavelength is within tolerance and element matches
			if wavelengthDiff < 0.5 && detected.Element == reference.Element {
				matches++
				totalIntensity += float32(detected.Intensity)
				targetIntensity += float32(reference.Intensity)
				break
			}
		}
	}

	// Calculate score based on match ratio and intensity correlation
	matchRatio := float32(matches) / float32(len(signature.SpectralLines))
	intensityCorrelation := float32(1.0)
	if targetIntensity > 0 {
		intensityCorrelation = totalIntensity / targetIntensity
		if intensityCorrelation > 1.0 {
			intensityCorrelation = 1.0
		}
	}

	// Combined score (70% match ratio, 30% intensity correlation)
	score := 0.7*matchRatio + 0.3*intensityCorrelation

	return score
}

// UpdateMetadataFromSpectra updates asset metadata based on spectral detection
// This is the "Propulsion Locking" mechanism
func (t *SpectralToFuelTable) UpdateMetadataFromSpectra(targetID uint32, detectedLines []SpectralLine, metadataStore *MetadataStore) bool {
	signature, confidence := t.IdentifyPropulsion(detectedLines)

	if confidence < 0.8 {
		return false // Low confidence, don't update
	}

	// Get current metadata
	meta := metadataStore.Get(targetID)

	// Update propulsion parameters
	meta.EngineType = signature.EngineType
	meta.Isp = signature.Isp
	meta.BurnRate = signature.BurnRate

	// Recalculate fuel budget with new Isp
	// Use current mass estimates
	originalDV := CalculateTsiolkovskyDV(meta.Isp, meta.MassInitial, meta.MassFinal)
	meta.EstimatedDV = originalDV

	// Persist updated metadata
	metadataStore.Set(meta)

	return true
}

// GetIspFromSpectra returns Isp directly from spectral lines (for quick lookup)
func (t *SpectralToFuelTable) GetIspFromSpectra(detectedLines []SpectralLine) float32 {
	signature, confidence := t.IdentifyPropulsion(detectedLines)
	if confidence < 0.8 {
		return 0.0 // No confident match
	}
	return signature.Isp
}

// GetBurnRateFromSpectra returns burn rate directly from spectral lines
func (t *SpectralToFuelTable) GetBurnRateFromSpectra(detectedLines []SpectralLine) float32 {
	signature, confidence := t.IdentifyPropulsion(detectedLines)
	if confidence < 0.8 {
		return 0.0 // No confident match
	}
	return signature.BurnRate
}

// SpectralObservation represents a single spectral observation event
type SpectralObservation struct {
	TargetID      uint32
	Timestamp     int64 // Unix nanoseconds
	DetectedLines []SpectralLine
	Confidence    float32
	EngineType    EngineType
}

// SpectralHistory maintains a history of spectral observations for an object
type SpectralHistory struct {
	observations []SpectralObservation
}

// AddObservation adds a new spectral observation to the history
func (sh *SpectralHistory) AddObservation(obs SpectralObservation) {
	sh.observations = append(sh.observations, obs)

	// Keep only last 100 observations
	if len(sh.observations) > 100 {
		sh.observations = sh.observations[len(sh.observations)-100:]
	}
}

// GetLatestObservation returns the most recent spectral observation
func (sh *SpectralHistory) GetLatestObservation() (SpectralObservation, bool) {
	if len(sh.observations) == 0 {
		return SpectralObservation{}, false
	}
	return sh.observations[len(sh.observations)-1], true
}

// DetectEngineTypeChange detects if the engine type has changed between observations
// This could indicate a propulsion system failure or spoofing attempt
func (sh *SpectralHistory) DetectEngineTypeChange() (bool, EngineType, EngineType) {
	if len(sh.observations) < 2 {
		return false, EnginePassive, EnginePassive
	}

	latest := sh.observations[len(sh.observations)-1]
	previous := sh.observations[len(sh.observations)-2]

	if latest.EngineType != previous.EngineType {
		return true, previous.EngineType, latest.EngineType
	}

	return false, EnginePassive, EnginePassive
}
