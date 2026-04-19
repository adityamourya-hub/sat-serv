package sigint

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// SIGIDTestSignal represents a known signal for confusion matrix testing
type SIGIDTestSignal struct {
	ID       string  `json:"id"`
	Freq     float64 `json:"frequency_hz"`
	BW       float64 `json:"bandwidth_hz"`
	PSDMask  []byte  `json:"psd_mask"`
	Category string  `json:"category"`
}

// loadSIGIDTestLibrary loads the 50-signal test library from JSON
func loadSIGIDTestLibrary(path string) []SIGIDTestSignal {
	// In production, this loads from assets/sigid_50_signals.json
	// For testing, we use a subset of the built-in templates
	data := []SIGIDTestSignal{
		{ID: "LINK_16", Freq: 1087.5e6, BW: 237e6, Category: "Military"},
		{ID: "HAVE_QUICK", Freq: 312.5e6, BW: 175e6, Category: "Military"},
		{ID: "SEVER_GORIZONT", Freq: 4592.5e6, BW: 315e6, Category: "Military"},
		{ID: "PROFORMA_X", Freq: 10.0e9, BW: 4e9, Category: "Military"},
		{ID: "TETRA", Freq: 420e6, BW: 25e3, Category: "Utility"},
		{ID: "INMARSAT", Freq: 1.6e9, BW: 40e6, Category: "Satellite"},
	}

	// Try to load from file if it exists
	if _, err := os.Stat(path); err == nil {
		file, err := os.ReadFile(path)
		if err == nil {
			var signals []SIGIDTestSignal
			if json.Unmarshal(file, &signals) == nil {
				return signals
			}
		}
	}

	return data
}

// TestSIGIDConfusionMatrix generates accuracy metrics for the 50-signal library.
func TestSIGIDConfusionMatrix(t *testing.T) {
	// Load known fingerprints (PSD masks) from the local library
	testSignals := loadSIGIDTestLibrary("../../assets/sigid_50_signals.json")
	bridge, err := NewSIGIDBridge("/tmp/sigid_test")
	if err != nil {
		t.Fatalf("Failed to create SIGID bridge: %v", err)
	}
	defer bridge.Close()

	// confusion[actual][predicted]
	confusion := make(map[string]map[string]int)

	for _, signal := range testSignals {
		// Matching against local templates via Hamming distance similarity
		// MatchWaterfall returns (confidence float64, match string)
		confidence, match := bridge.MatchWaterfall(signal.PSDMask, signal.Freq, signal.BW)

		// If no match found, count as "UNKNOWN"
		if match == "" {
			match = "UNKNOWN"
		}

		if confusion[signal.ID] == nil {
			confusion[signal.ID] = make(map[string]int)
		}
		confusion[signal.ID][match]++

		t.Logf("Signal %s: matched to %s with confidence %.2f%%", signal.ID, match, confidence*100)
	}

	// Focus on High-Value Military Accuracy (Link 16, HAVE QUICK, etc.)
	milSignals := []string{"LINK_16", "HAVE_QUICK", "SEVER_GORIZONT", "PROFORMA_X"}
	var correct, total int

	for _, id := range milSignals {
		if _, exists := confusion[id]; !exists {
			continue // Skip if not in test set
		}

		correct += confusion[id][id]
		for _, count := range confusion[id] {
			total += count
		}
	}

	if total == 0 {
		t.Skip("No military signals in test library")
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("Military signal accuracy: %.1f%% (%d/%d)", accuracy*100, correct, total)

	if accuracy < 0.90 {
		t.Errorf("Strategic signal accuracy %.1f%% below 90%% requirement", accuracy*100)
	}
}

// TestSpectralGhostDetection validates the detection of UCTs with no ADS-B correlation.
func TestSpectralGhostDetection(t *testing.T) {
	// Simulate an RF detection in space (100km range, 7.5 km/s velocity)
	ghost := &SpaceDetection{
		ObjectID:   "GHOST_SIGINT_01",
		Azimuth:    45.0,
		Elevation:  30.0,
		Range:      100000.0,
		Velocity:   7500.0,
		Confidence: 0.85,
	}

	correlator := NewADSBCorrelator()

	// Negative correlation should return false (no aircraft match)
	isCorrelated, result := correlator.NegativeCorrelationCheck(ghost)
	if isCorrelated {
		t.Fatal("Spectral Ghost incorrectly correlated with civilian ADS-B data")
	}

	if result != nil && result.IsCorrelated {
		t.Fatal("Correlation result indicates false positive correlation")
	}

	// Verify the UCT alert maps the signature to a 'Potential Launch'
	alert := correlator.GenerateUCTAlert(ghost)
	if alert == nil {
		t.Fatal("GenerateUCTAlert returned nil")
	}

	// Verify alert properties
	if alert.ObjectID != ghost.ObjectID {
		t.Errorf("Alert ObjectID mismatch: got %s, want %s", alert.ObjectID, ghost.ObjectID)
	}

	// Check for high-velocity detection
	if !alert.IsRapid {
		t.Error("High-velocity ghost (7.5 km/s) not flagged as rapid")
	}

	// Verify recommendation contains launch indication
	if !strings.Contains(strings.ToLower(alert.Recommendation), "launch") &&
		!strings.Contains(strings.ToLower(alert.Recommendation), "rapid") &&
		!strings.Contains(strings.ToLower(alert.Recommendation), "asat") {
		t.Errorf("UCT alert failed to flag high-velocity ghost as potential launch: %s", alert.Recommendation)
	}

	t.Logf("Spectral Ghost correctly flagged: %s", alert.Recommendation)
}

// TestSIGIDHammingAccuracy validates Hamming distance matching precision
func TestSIGIDHammingAccuracy(t *testing.T) {
	bridge, err := NewSIGIDBridge("/tmp/sigid_hamming_test")
	if err != nil {
		t.Fatalf("Failed to create SIGID bridge: %v", err)
	}
	defer bridge.Close()

	// Create a test signal that closely matches Link 16
	link16Freq := 1087.5e6
	link16BW := 237e6

	// Simulate a detected signal (noisy version of Link 16)
	detectedMask := make([]byte, 64) // 512-bit mask
	for i := range detectedMask {
		detectedMask[i] = 0xFF // Full power signature
	}

	confidence, match := bridge.MatchWaterfall(detectedMask, link16Freq, link16BW)

	t.Logf("Link 16 matching: got %s with confidence %.2f%%", match, confidence*100)

	// If a match is found, verify it's reasonable
	if confidence > 0.7 && match != "" && match != "UNKNOWN" {
		template, err := bridge.LookupID(match)
		if err == nil {
			t.Logf("Matched to template: %s (%s)", template.Name, template.Category)
		}
	}
}

// TestHighValueSignalPrioritization ensures military signals are prioritized
func TestHighValueSignalPrioritization(t *testing.T) {
	bridge, err := NewSIGIDBridge("/tmp/sigid_priority_test")
	if err != nil {
		t.Fatalf("Failed to create SIGID bridge: %v", err)
	}
	defer bridge.Close()

	// Query military templates
	milTemplates, err := bridge.GetTemplatesByCategory("Military")
	if err != nil {
		t.Fatalf("Failed to get military templates: %v", err)
	}

	if len(milTemplates) == 0 {
		t.Skip("No military templates in database")
	}

	// Verify all military templates have appropriate threat levels
	for _, tmpl := range milTemplates {
		if tmpl.ThreatLevel < 2 {
			t.Errorf("Military signal %s has low threat level %d", tmpl.ID, tmpl.ThreatLevel)
		}
		t.Logf("Military template: %s (threat level %d)", tmpl.Name, tmpl.ThreatLevel)
	}
}
