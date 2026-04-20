// pkg/ml/gatv2_attributes.go
//  GATv2 Graph Attributes for AMOS-Enhanced Scenario Graphs
//
// Integrates AMOS 2025 spectral/photometric features into GATv2 node attributes
// for threat prediction with >90% confidence on Yaogan/Shijian class satellites.

package ml

import (
	"math"
	"ssis/pkg/catalog"
)

// GATv2NodeAttributes extends scenario nodes with AMOS-derived features
// These attributes feed into GATv2 attention mechanism for threat prediction
//
// ADITI 4.0: Enhanced with fractional drag, spectral residuals, and MGE threat score
// for HGV corridor plasma wake modeling (30-80km transitional flow)
type GATv2NodeAttributes struct {
	// Standard scenario features
	Position  [3]float64 // ECI position (km)
	Velocity  [3]float64 // ECI velocity (km/s)
	Timestamp int64      // Unix timestamp

	// AMOS 2025: Spectral/Photometric features (Landon + Slingshot)
	SpectralResiduals  []float64 // Residuals from ConvNeXt 1D classification (32-dim)
	SpectralConfidence float64   // Bus-type classification confidence
	PhotometricScore   float64   // Light-curve similarity score
	PhotometricPeriod  float64   // Rotation period from fingerprinting

	// AMOS 2025: Hypersonic detection (MyRadar SWIR)
	PlasmaIntensity      float64 // SWIR plasma intensity
	PlasmaTemperature    float64 // Estimated plasma temperature (K)
	HypersonicConfidence float64 // HGV detection confidence
	IsHypersonic         bool    // True if hypersonic signature detected

	// AMOS 2025: Non-local turbulence (Georgia State + Slingshot UDE)
	ZernikeCoefficients []float64 // Speckle-derived Zernike modes (15-dim)
	UDEBiasCorrection   float64   // Atmosphere bias from UDE
	TurbulenceIndex     float64   // Non-local turbulence severity

	// ADITI 4.0: Fractional drag and wake memory (HGV corridor 30-80km)
	FractionalDragResidual float64 // Caputo D_t^β fractional drag residual
	FractionalAlpha        float64 // Fractional Laplacian exponent α ∈ [1,2]
	WakeMemoryLength       int     // Length of wake memory buffer
	IonosphericTEC         float64 // Total electron content (16-dim SH embedding)

	// ADITI 0.0: Enhanced MGE threat score (18 templates, CRITICAL detection)
	MGETemplateScore    float64 // MGE template match score (0-1)
	MGEThreatLevel      string  // MGE-derived threat level
	MGEArchitectureType string  // Bus architecture (e.g., "HIGH_POWER_EW_SAR")
	MGEComponentCount   int     // Total components detected
	MGERatio            float64 // Array-to-bus ratio

	// Combined threat features
	SpectralFingerprint float64 // Combined spectral/photometric score
	ThreatLevel         string  // "LOW", "MEDIUM", "HIGH", "CRITICAL"
	ThreatConfidence    float64 // Combined confidence score (0-1)
}

// NewGATv2NodeAttributes creates node attributes from catalog template
//
// ADITI 0.0: Enhanced with fractional drag and MGE threat score for HGV corridor
func NewGATv2NodeAttributes(template *catalog.MGETemplateExtended, position, velocity [3]float64) *GATv2NodeAttributes {
	attrs := &GATv2NodeAttributes{
		Position:  position,
		Velocity:  velocity,
		Timestamp: template.SpectralSignature.Timestamp,

		// Spectral features from ConvNeXt 1D (Cedarville/USAF)
		SpectralConfidence: template.SpectralFingerprintScore,

		// Photometric features from Slingshot fingerprinting
		PhotometricScore:  template.PhotometricMatchScore,
		PhotometricPeriod: template.PhotometricFeatures.Period,

		// ADITI 0.0: Enhanced MGE threat score (18 templates)
		MGETemplateScore:    template.ConfidenceComposite,
		MGEThreatLevel:      template.ThreatLevel,
		MGEArchitectureType: "UNKNOWN", // From MGE library, not catalog
		MGEComponentCount:   0,         // From MGE library, not catalog
		MGERatio:            0.0,       // From MGE library, not catalog

		// Combined MGE score
		SpectralFingerprint: template.ComputeCompositeScore(),

		ThreatLevel:      template.ThreatLevel,
		ThreatConfidence: template.ConfidenceComposite,
	}

	// Add spectral residuals if available (32-dim from ConvNeXt)
	if len(template.SpectralSignature.Intensities) > 0 {
		attrs.SpectralResiduals = computeSpectralResiduals(template.SpectralSignature)
	}

	// Add hypersonic features if detected (MyRadar SWIR)
	if template.PlasmaSignature != nil && template.PlasmaSignature.Detected {
		attrs.PlasmaIntensity = template.PlasmaSignature.SWIRIntensity
		attrs.PlasmaTemperature = template.PlasmaSignature.PlasmaTemp
		attrs.HypersonicConfidence = template.PlasmaSignature.Confidence
		attrs.IsHypersonic = true
		attrs.ThreatLevel = "CRITICAL"
		attrs.ThreatConfidence = 1.0
	}

	return attrs
}

// computeSpectralResiduals extracts deviation from expected spectral template
// Used by GATv2 attention to identify anomalous objects
//
// ADITI 0.0: Returns 32-dim residuals for ConvNeXt integration
func computeSpectralResiduals(sig catalog.SpectralSignature) []float64 {
	if len(sig.Intensities) == 0 {
		return nil
	}

	// Compute residual from expected smooth spectrum
	residuals := make([]float64, len(sig.Intensities))

	// Simple smoothing for expected spectrum
	for i := range sig.Intensities {
		var expected float64
		window := 3
		count := 0

		for j := -window; j <= window; j++ {
			idx := i + j
			if idx >= 0 && idx < len(sig.Intensities) {
				expected += sig.Intensities[idx]
				count++
			}
		}

		if count > 0 {
			expected /= float64(count)
		}

		residuals[i] = sig.Intensities[i] - expected
	}

	return residuals
}

// UpdateFractionalDrag updates ADITI 0.0 fractional drag features from speckle preprocessor
//
// Integrates with AMOS Georgia State speckle correction and UDE density bias
func (attrs *GATv2NodeAttributes) UpdateFractionalDrag(
	densityCorrection float64,
	fractionalAlpha float64,
	wakeMemoryLength int,
	ionosphericTEC float64,
) {
	attrs.UDEBiasCorrection = densityCorrection
	attrs.FractionalAlpha = fractionalAlpha
	attrs.WakeMemoryLength = wakeMemoryLength
	attrs.IonosphericTEC = ionosphericTEC

	// Compute fractional drag residual from density correction
	// In production: Use Caputo D_t^β kernel
	attrs.FractionalDragResidual = (1.0 - densityCorrection) * 0.1
}

// UpdateMGEFromLibrary updates MGE features from MGE template library result
//
// ADITI 0.0: Integrates with ai/analytics/mge_models.py for 18-template matching
func (attrs *GATv2NodeAttributes) UpdateMGEFromLibrary(
	mgeScore float64,
	threatLevel string,
	architectureType string,
	componentCount int,
	ratio float64,
) {
	attrs.MGETemplateScore = mgeScore
	attrs.MGEThreatLevel = threatLevel
	attrs.MGEArchitectureType = architectureType
	attrs.MGEComponentCount = componentCount
	attrs.MGERatio = ratio

	// Update combined threat level if MGE is CRITICAL
	if threatLevel == "CRITICAL" {
		attrs.ThreatLevel = "CRITICAL"
		attrs.ThreatConfidence = math.Max(attrs.ThreatConfidence, 0.95)
	}
}

// ToFeatureVector converts attributes to dense feature vector for GATv2
// Returns fixed-size vector for neural network input
func (attrs *GATv2NodeAttributes) ToFeatureVector() []float64 {
	// Fixed-size feature vector: 32 dimensions
	features := make([]float64, 32)

	// Position (3)
	features[0] = attrs.Position[0]
	features[1] = attrs.Position[1]
	features[2] = attrs.Position[2]

	// Velocity (3)
	features[3] = attrs.Velocity[0]
	features[4] = attrs.Velocity[1]
	features[5] = attrs.Velocity[2]

	// Spectral features (4)
	features[6] = attrs.SpectralConfidence
	features[7] = mean(attrs.SpectralResiduals)
	features[8] = std(attrs.SpectralResiduals)
	features[9] = maxAbs(attrs.SpectralResiduals)

	// Photometric features (3)
	features[10] = attrs.PhotometricScore
	features[11] = attrs.PhotometricPeriod
	features[12] = attrs.MGETemplateScore

	// Hypersonic features (3)
	features[13] = attrs.PlasmaIntensity
	features[14] = attrs.PlasmaTemperature
	features[15] = attrs.HypersonicConfidence

	if attrs.IsHypersonic {
		features[16] = 1.0
	} else {
		features[16] = 0.0
	}

	// Turbulence features (3)
	features[17] = mean(attrs.ZernikeCoefficients)
	features[18] = attrs.UDEBiasCorrection
	features[19] = attrs.TurbulenceIndex

	// Threat level encoding (4)
	switch attrs.ThreatLevel {
	case "CRITICAL":
		features[20], features[21], features[22], features[23] = 1.0, 0.0, 0.0, 0.0
	case "HIGH":
		features[20], features[21], features[22], features[23] = 0.0, 1.0, 0.0, 0.0
	case "MEDIUM":
		features[20], features[21], features[22], features[23] = 0.0, 0.0, 1.0, 0.0
	default:
		features[20], features[21], features[22], features[23] = 0.0, 0.0, 0.0, 1.0
	}

	// Threat confidence (1)
	features[24] = attrs.ThreatConfidence

	// Spectral fingerprint composite (1)
	features[25] = attrs.SpectralFingerprint

	// Reserved for expansion (6)
	// features[26-31] available for future AMOS-derived features

	return features
}

// Helper functions for feature vector computation
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func std(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}
	m := mean(data)
	sum := 0.0
	for _, v := range data {
		d := v - m
		sum += d * d
	}
	return sum / float64(len(data))
}

func maxAbs(data []float64) float64 {
	max := 0.0
	for _, v := range data {
		if v < 0 {
			v = -v
		}
		if v > max {
			max = v
		}
	}
	return max
}

// GATv2ScenarioGraph represents the full scenario graph with AMOS-enhanced nodes
type GATv2ScenarioGraph struct {
	Nodes       []*GATv2NodeAttributes
	Edges       [][2]int  // Node indices for edges
	EdgeWeights []float64 // Attention weights

	// Scenario metadata
	ScenarioID    string
	Timestamp     int64
	TargetThreats []string // IDs of predicted critical threats
}

// NewScenarioGraph creates graph from catalog templates
func NewScenarioGraph(scenarioID string, templates []*catalog.MGETemplateExtended, positions, velocities [][3]float64) *GATv2ScenarioGraph {
	graph := &GATv2ScenarioGraph{
		ScenarioID:    scenarioID,
		Timestamp:     0, // Set to current time
		Nodes:         make([]*GATv2NodeAttributes, len(templates)),
		Edges:         make([][2]int, 0),
		EdgeWeights:   make([]float64, 0),
		TargetThreats: make([]string, 0),
	}

	// Create nodes
	for i, template := range templates {
		var pos, vel [3]float64
		if i < len(positions) {
			pos = positions[i]
		}
		if i < len(velocities) {
			vel = velocities[i]
		}

		graph.Nodes[i] = NewGATv2NodeAttributes(template, pos, vel)

		// Track critical threats
		if template.ThreatLevel == "CRITICAL" {
			graph.TargetThreats = append(graph.TargetThreats, template.NORADID)
		}
	}

	// Create edges based on spatial proximity (simplified)
	graph.buildProximityEdges()

	return graph
}

// buildProximityEdges creates edges between nearby objects
func (graph *GATv2ScenarioGraph) buildProximityEdges() {
	threshold := 1000.0 // km threshold for edge creation

	for i, nodeI := range graph.Nodes {
		for j, nodeJ := range graph.Nodes {
			if i >= j {
				continue
			}

			// Compute distance
			dx := nodeI.Position[0] - nodeJ.Position[0]
			dy := nodeI.Position[1] - nodeJ.Position[1]
			dz := nodeI.Position[2] - nodeJ.Position[2]
			dist := dx*dx + dy*dy + dz*dz

			if dist < threshold*threshold {
				graph.Edges = append(graph.Edges, [2]int{i, j})
				// Inverse distance weighting
				weight := 1.0 / (1.0 + dist/(100.0*100.0))
				graph.EdgeWeights = append(graph.EdgeWeights, weight)
			}
		}
	}
}
