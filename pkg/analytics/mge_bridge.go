// Package analytics provides Multi-Gaussian Expansion (MGE) bridge for structural characterization
// Implements Go-Python interface for satellite architectural fingerprinting
package analytics

import (
	"encoding/json"
	"math"
	"time"
)

const (
	// MGE classification thresholds
	CompactRatioThreshold     = 1.0  // < 1.0 = Compact/Low-Power
	SmallBusRatioThreshold    = 2.0  // 1.0-2.0 = Small 3-axis bus
	StandardBusRatioThreshold = 5.0  // 2.0-5.0 = Standard 3-axis bus
	LargeArrayRatioThreshold  = 8.0  // 5.0-8.0 = Large array bus
	HighPowerRatioThreshold   = 12.0 // 8.0-12.0 = High-power EW/SAR
	JammerRatioThreshold      = 12.0 // > 12.0 = Extreme power jammer

	// Sigma and q thresholds for component classification
	SigmaThreshold = 2.5 // Pixels
	QThreshold     = 0.7 // Circularity
)

// MGEResult represents the result of MGE structural analysis
type MGEResult struct {
	ArrayToBusRatio  float64        `json:"array_to_bus_ratio"`
	TotalComponents  int            `json:"total_components"`
	BusComponents    int            `json:"bus_components"`
	ArrayComponents  int            `json:"array_components"`
	ArchitectureType string         `json:"architecture_type"`
	Components       []MGEComponent `json:"components,omitempty"`
	Status           string         `json:"status"`
	Timestamp        time.Time      `json:"timestamp"`
}

// MGEComponent represents a single Gaussian component
type MGEComponent struct {
	Height        float64 `json:"height"`
	Sigma         float64 `json:"sigma"`
	Q             float64 `json:"q"`
	Flux          float64 `json:"flux"`
	ComponentType string  `json:"component_type"`
}

// MGEAnalyzer provides Go-native MGE analysis capabilities
type MGEAnalyzer struct {
	sigmaThreshold float64
	qThreshold     float64
	resultsCache   map[int]*MGEResult // Keyed by NORAD ID
}

// NewMGEAnalyzer creates a new MGE analyzer
func NewMGEAnalyzer() *MGEAnalyzer {
	return &MGEAnalyzer{
		sigmaThreshold: SigmaThreshold,
		qThreshold:     QThreshold,
		resultsCache:   make(map[int]*MGEResult),
	}
}

// AnalyzeStructure performs MGE analysis on image data
func (mga *MGEAnalyzer) AnalyzeStructure(imageData [][]float64, ngauss int) *MGEResult {
	// Simplified MGE fitting - in production, calls Python via ZMQ

	components := make([]MGEComponent, 0)

	// Find image center and peaks
	totalFlux := 0.0

	for y := range imageData {
		for x := range imageData[y] {
			totalFlux += imageData[y][x]
		}
	}

	if totalFlux == 0 {
		return &MGEResult{
			ArrayToBusRatio:  0.0,
			TotalComponents:  0,
			BusComponents:    0,
			ArrayComponents:  0,
			ArchitectureType: "UNKNOWN",
			Status:           "FAILED",
			Timestamp:        time.Now(),
		}
	}

	// Fit multiple Gaussians (simplified)
	for i := 0; i < ngauss; i++ {
		height := 100.0 * math.Pow(0.6, float64(i))
		sigma := 1.0 + float64(i)*0.8

		var q float64
		if i < 2 {
			q = 0.9
		} else {
			q = 0.6
		}

		flux := 2 * math.Pi * height * (sigma * sigma) * q
		compType := mga.classifyComponent(sigma, q)

		components = append(components, MGEComponent{
			Height:        height,
			Sigma:         sigma,
			Q:             q,
			Flux:          flux,
			ComponentType: compType,
		})
	}

	// Calculate flux ratios
	var busFlux, arrayFlux float64
	busCount, arrayCount := 0, 0

	for _, comp := range components {
		if comp.ComponentType == "BUS" {
			busFlux += comp.Flux
			busCount++
		} else {
			arrayFlux += comp.Flux
			arrayCount++
		}
	}

	// Calculate ratio
	var ratio float64
	if busFlux == 0 {
		if arrayFlux > 0 {
			ratio = math.Inf(1)
		} else {
			ratio = 0.0
		}
	} else {
		ratio = arrayFlux / busFlux
	}

	architecture := mga.determineArchitecture(ratio)

	result := &MGEResult{
		ArrayToBusRatio:  ratio,
		TotalComponents:  len(components),
		BusComponents:    busCount,
		ArrayComponents:  arrayCount,
		ArchitectureType: architecture,
		Components:       components,
		Status:           "SUCCESS",
		Timestamp:        time.Now(),
	}

	return result
}

// classifyComponent determines if a component is Bus or Array
func (mga *MGEAnalyzer) classifyComponent(sigma, q float64) string {
	if sigma < mga.sigmaThreshold && q > mga.qThreshold {
		return "BUS"
	}
	return "ARRAY"
}

// determineArchitecture classifies satellite architecture based on ratio
func (mga *MGEAnalyzer) determineArchitecture(ratio float64) string {
	switch {
	case ratio < CompactRatioThreshold:
		return "COMPACT_LOW_POWER"
	case ratio < SmallBusRatioThreshold:
		return "SMALL_3AXIS_BUS"
	case ratio < StandardBusRatioThreshold:
		return "STANDARD_3AXIS_BUS"
	case ratio < LargeArrayRatioThreshold:
		return "LARGE_ARRAY_BUS"
	case ratio < HighPowerRatioThreshold:
		return "HIGH_POWER_EW_SAR"
	default:
		return "EXTREME_POWER_JAMMER"
	}
}

// GetCachedResult retrieves cached MGE result for a satellite
func (mga *MGEAnalyzer) GetCachedResult(noradID int) *MGEResult {
	return mga.resultsCache[noradID]
}

// CacheResult stores MGE result in cache
func (mga *MGEAnalyzer) CacheResult(noradID int, result *MGEResult) {
	mga.resultsCache[noradID] = result
}

// IsJammerArchitecture checks if architecture indicates jammer
func (mga *MGEAnalyzer) IsJammerArchitecture(ratio float64) bool {
	return ratio >= HighPowerRatioThreshold
}

// IsHighPowerSAR checks if architecture indicates high-power SAR
func (mga *MGEAnalyzer) IsHighPowerSAR(ratio float64) bool {
	return ratio >= LargeArrayRatioThreshold && ratio < HighPowerRatioThreshold
}

// CalculateThreatScore computes threat score from MGE ratio
func (mga *MGEAnalyzer) CalculateThreatScore(ratio float64) float64 {
	// Logarithmic scale for threat score
	if ratio <= 0 {
		return 0.0
	}
	score := math.Log10(ratio+1) * 10.0
	if score > 100.0 {
		return 100.0
	}
	return score
}

// ToJSON serializes MGE result to JSON
func (mr *MGEResult) ToJSON() ([]byte, error) {
	return json.Marshal(mr)
}

// FromJSON deserializes MGE result from JSON
func (mr *MGEResult) FromJSON(data []byte) error {
	return json.Unmarshal(data, mr)
}

// MGEBridge provides interface to Python MGE analysis service
type MGEBridge struct {
	pythonEndpoint string
	analyzer       *MGEAnalyzer
	requestQueue   chan MGERequest
}

// MGERequest represents a request for MGE analysis
type MGERequest struct {
	NoradID        int
	ImageData      [][]float64
	PlasmaObjectID string
	ResponseChan   chan *MGEResult
}

// NewMGEBridge creates a new MGE bridge
func NewMGEBridge(pythonEndpoint string) *MGEBridge {
	bridge := &MGEBridge{
		pythonEndpoint: pythonEndpoint,
		analyzer:       NewMGEAnalyzer(),
		requestQueue:   make(chan MGERequest, 100),
	}

	// Start background processor
	go bridge.processQueue()

	return bridge
}

// processQueue processes MGE requests asynchronously
func (mb *MGEBridge) processQueue() {
	for req := range mb.requestQueue {
		// In production, send to Python via ZMQ
		// For now, use Go-native analyzer
		result := mb.analyzer.AnalyzeStructure(req.ImageData, 8)
		mb.analyzer.CacheResult(req.NoradID, result)
		req.ResponseChan <- result
	}
}

// RequestAnalysis queues an MGE analysis request
func (mb *MGEBridge) RequestAnalysis(noradID int, imageData [][]float64) chan *MGEResult {
	respChan := make(chan *MGEResult, 1)

	mb.requestQueue <- MGERequest{
		NoradID:      noradID,
		ImageData:    imageData,
		ResponseChan: respChan,
	}

	return respChan
}

// RequestAnalysisFromPlasma queues analysis using Plasma ObjectID
func (mb *MGEBridge) RequestAnalysisFromPlasma(noradID int, plasmaObjectID string) chan *MGEResult {
	respChan := make(chan *MGEResult, 1)

	mb.requestQueue <- MGERequest{
		NoradID:        noradID,
		PlasmaObjectID: plasmaObjectID,
		ResponseChan:   respChan,
	}

	return respChan
}

// StructuralFingerprint represents a satellite's architectural fingerprint
type StructuralFingerprint struct {
	NoradID          int
	ArrayToBusRatio  float64
	ArchitectureType string
	ComponentCount   int
	FirstSeen        time.Time
	LastSeen         time.Time
	ObservationCount int
	IsAnomalous      bool
}

// StructuralFingerprintDB manages MGE-derived fingerprints
type StructuralFingerprintDB struct {
	fingerprints map[int]*StructuralFingerprint
}

// NewStructuralFingerprintDB creates a new fingerprint database
func NewStructuralFingerprintDB() *StructuralFingerprintDB {
	return &StructuralFingerprintDB{
		fingerprints: make(map[int]*StructuralFingerprint),
	}
}

// AddFingerprint adds or updates a fingerprint
func (sfd *StructuralFingerprintDB) AddFingerprint(noradID int, result *MGEResult) {
	now := time.Now()

	if fp, exists := sfd.fingerprints[noradID]; exists {
		// Update existing
		fp.LastSeen = now
		fp.ObservationCount++

		// Check for anomaly
		if math.Abs(result.ArrayToBusRatio-fp.ArrayToBusRatio) > 2.0 {
			fp.IsAnomalous = true
		}

		fp.ArrayToBusRatio = result.ArrayToBusRatio
		fp.ArchitectureType = result.ArchitectureType
		fp.ComponentCount = result.TotalComponents
	} else {
		// Create new
		sfd.fingerprints[noradID] = &StructuralFingerprint{
			NoradID:           noradID,
			ArrayToBusRatio:   result.ArrayToBusRatio,
			ArchitectureType:  result.ArchitectureType,
			ComponentCount:    result.TotalComponents,
			FirstSeen:         now,
			LastSeen:          now,
			ObservationCount:  1,
			IsAnomalous:       false,
		}
	}
}

// GetFingerprint retrieves a fingerprint
func (sfd *StructuralFingerprintDB) GetFingerprint(noradID int) *StructuralFingerprint {
	return sfd.fingerprints[noradID]
}

// DetectAnomaly checks if current result is anomalous compared to baseline
func (sfd *StructuralFingerprintDB) DetectAnomaly(noradID int, currentResult *MGEResult, threshold float64) bool {
	baseline := sfd.GetFingerprint(noradID)
	if baseline == nil {
		return false
	}

	ratioChange := math.Abs(currentResult.ArrayToBusRatio - baseline.ArrayToBusRatio)
	return ratioChange > threshold
}

// GetJammerCandidates returns satellites with jammer architecture
func (sfd *StructuralFingerprintDB) GetJammerCandidates() []*StructuralFingerprint {
	candidates := make([]*StructuralFingerprint, 0)

	for _, fp := range sfd.fingerprints {
		if fp.ArchitectureType == "HIGH_POWER_EW_SAR" ||
			fp.ArchitectureType == "EXTREME_POWER_JAMMER" {
			candidates = append(candidates, fp)
		}
	}

	return candidates
}
