package sda

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Material Identification for Area-to-Mass Ratio Calculation
 *
 * Reconstructs bus material from spectral data (Gallium Arsenide solar cells vs.
 * multi-layer insulation) to calculate Area-to-Mass (A/m) ratio with precision.
 * This is the "Golden Key" to solving the drag equation at 400km.
 */

// MaterialType represents the type of material detected
type MaterialType int

const (
	MaterialUnknown      MaterialType = iota
	MaterialGaAsSolar                 // Gallium Arsenide solar cells
	MaterialSiliconSolar              // Silicon solar cells
	MaterialMLI                       // Multi-layer insulation
	MaterialAluminum                  // Aluminum structure
	MaterialCarbonFiber               // Carbon fiber composite
	MaterialTitanium                  // Titanium structure
)

func (mt MaterialType) String() string {
	switch mt {
	case MaterialGaAsSolar:
		return "GaAs_SOLAR"
	case MaterialSiliconSolar:
		return "Si_SOLAR"
	case MaterialMLI:
		return "MLI"
	case MaterialAluminum:
		return "ALUMINUM"
	case MaterialCarbonFiber:
		return "CARBON_FIBER"
	case MaterialTitanium:
		return "TITANIUM"
	default:
		return "UNKNOWN"
	}
}

// MaterialSignature defines the spectral signature of a material
type MaterialSignature struct {
	MaterialType  MaterialType
	SpectralLines []SpectralLine
	DensityKgM3   float32 // Density in kg/m³
	AreaToMass    float32 // Typical A/m ratio (m²/kg)
	Reflectivity  float32 // Albedo (0-1)
	Confidence    float32 // Detection confidence threshold
}

// MaterialIDTable maps spectral signatures to material properties
type MaterialIDTable struct {
	signatures map[string]MaterialSignature
}

// NewMaterialIDTable creates a new material identification table
func NewMaterialIDTable() *MaterialIDTable {
	table := &MaterialIDTable{
		signatures: make(map[string]MaterialSignature),
	}
	table.initializeDefaultSignatures()
	return table
}

// initializeDefaultSignatures populates the table with known material signatures
func (t *MaterialIDTable) initializeDefaultSignatures() {
	// Gallium Arsenide (GaAs) Solar Cells
	// Characteristic lines: 868.2 nm (Ga), 871.2 nm (As)
	t.signatures["GaAs"] = MaterialSignature{
		MaterialType: MaterialGaAsSolar,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 868.2, Intensity: 0.9, Element: "Ga"},
			{WavelengthNm: 871.2, Intensity: 0.85, Element: "As"},
		},
		DensityKgM3:  5320.0, // GaAs density
		AreaToMass:   0.02,   // High A/m for solar arrays
		Reflectivity: 0.3,    // Dark solar cells
		Confidence:   0.9,
	}

	// Silicon Solar Cells
	// Characteristic lines: 251.6 nm (Si), 288.2 nm (Si)
	t.signatures["Si"] = MaterialSignature{
		MaterialType: MaterialSiliconSolar,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 251.6, Intensity: 0.8, Element: "Si"},
			{WavelengthNm: 288.2, Intensity: 0.75, Element: "Si"},
		},
		DensityKgM3:  2330.0, // Silicon density
		AreaToMass:   0.015,  // Moderate A/m
		Reflectivity: 0.25,   // Moderate reflectivity
		Confidence:   0.85,
	}

	// Multi-Layer Insulation (MLI)
	// Characteristic lines: Kapton bands at 320-350 nm
	t.signatures["Kapton"] = MaterialSignature{
		MaterialType: MaterialMLI,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 320.0, Intensity: 0.7, Element: "C-N"},
			{WavelengthNm: 335.0, Intensity: 0.65, Element: "C-N"},
			{WavelengthNm: 350.0, Intensity: 0.6, Element: "C-N"},
		},
		DensityKgM3:  1420.0, // Kapton density
		AreaToMass:   0.05,   // Very high A/m (thin films)
		Reflectivity: 0.8,    // High reflectivity
		Confidence:   0.8,
	}

	// Aluminum Structure
	// Characteristic lines: 396.1 nm (Al), 669.6 nm (Al)
	t.signatures["Al"] = MaterialSignature{
		MaterialType: MaterialAluminum,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 396.1, Intensity: 0.85, Element: "Al"},
			{WavelengthNm: 669.6, Intensity: 0.8, Element: "Al"},
		},
		DensityKgM3:  2700.0, // Aluminum density
		AreaToMass:   0.005,  // Low A/m (structural)
		Reflectivity: 0.9,    // Very high reflectivity
		Confidence:   0.85,
	}

	// Carbon Fiber Composite
	// Characteristic lines: Carbon bands at 388 nm, 422 nm
	t.signatures["C"] = MaterialSignature{
		MaterialType: MaterialCarbonFiber,
		SpectralLines: []SpectralLine{
			{WavelengthNm: 388.0, Intensity: 0.75, Element: "C"},
			{WavelengthNm: 422.0, Intensity: 0.7, Element: "C"},
		},
		DensityKgM3:  1600.0, // Carbon fiber density
		AreaToMass:   0.01,   // Moderate A/m
		Reflectivity: 0.4,    // Moderate reflectivity
		Confidence:   0.75,
	}
}

// IdentifyMaterial identifies material type from detected spectral lines
func (t *MaterialIDTable) IdentifyMaterial(detectedLines []SpectralLine) (MaterialSignature, float32) {
	bestMatch := MaterialSignature{MaterialType: MaterialUnknown}
	bestScore := float32(0.0)

	for element, signature := range t.signatures {
		score := t.scoreMaterial(detectedLines, signature, element)
		if score > bestScore && score >= signature.Confidence {
			bestMatch = signature
			bestScore = score
		}
	}

	return bestMatch, bestScore
}

// scoreMaterial calculates the match score between detected lines and a material signature
func (t *MaterialIDTable) scoreMaterial(detectedLines []SpectralLine, signature MaterialSignature, _element string) float32 {
	if len(detectedLines) == 0 || len(signature.SpectralLines) == 0 {
		return 0.0
	}

	matches := 0
	totalIntensity := float32(0.0)
	targetIntensity := float32(0.0)

	// Check for wavelength matches (within 1.0 nm tolerance for materials)
	for _, detected := range detectedLines {
		for _, reference := range signature.SpectralLines {
			wavelengthDiff := detected.WavelengthNm - reference.WavelengthNm
			if wavelengthDiff < 0 {
				wavelengthDiff = -wavelengthDiff
			}

			// Match if wavelength is within tolerance and element matches
			if wavelengthDiff < 1.0 && detected.Element == reference.Element {
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

	// Combined score (60% match ratio, 40% intensity correlation)
	score := 0.6*matchRatio + 0.4*intensityCorrelation

	return score
}

// CalculateAreaToMass calculates the Area-to-Mass ratio from detected materials
// This is the "Golden Key" for drag equation: F_drag = 0.5 * ρ * v² * C_d * A
func (t *MaterialIDTable) CalculateAreaToMass(detectedLines []SpectralLine) (float32, MaterialType) {
	signature, confidence := t.IdentifyMaterial(detectedLines)

	if confidence < 0.7 {
		// Default to moderate A/m if no confident match
		return 0.01, MaterialUnknown
	}

	return signature.AreaToMass, signature.MaterialType
}

// UpdateMetadataWithMaterial updates asset metadata with material properties
func (t *MaterialIDTable) UpdateMetadataWithMaterial(targetID uint32, detectedLines []SpectralLine, metadataStore *MetadataStore) bool {
	signature, confidence := t.IdentifyMaterial(detectedLines)

	if confidence < 0.7 {
		return false // Low confidence, don't update
	}

	meta := metadataStore.Get(targetID)

	// Store material type in platform class for tracking
	// In production: add dedicated MaterialType field
	if meta.PlatformClass == "Unknown" || meta.PlatformClass == "" {
		meta.PlatformClass = signature.MaterialType.String()
	}

	// Update mass estimate based on material density
	// In production: calculate from observed dimensions
	// For now: use existing mass estimate

	metadataStore.Set(meta)

	return true
}
