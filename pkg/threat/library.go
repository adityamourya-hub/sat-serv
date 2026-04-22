package threat

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

/**
 * Sovereign Threat Library
 *
 * Transforms SINDy-discovered physics (L/D ratios) into actionable intelligence.
 *
 * Pipeline:
 *   SINDy Output (L/D=2.05) → Library Lookup → Identity (DF-ZF) → RL Action
 *
 * Requirements Satisfied:
 *   - Explainability: "Discovered L/D matches DF-ZF profile"
 *   - Sovereignty: Air-gapped YAML, updatable with classified INTEL
 *   - Adaptability: Auto-UCT capture for new threats
 */

// Aerodynamics defines the physics signature of a threat
type Aerodynamics struct {
	LDRatioMin       float64 `yaml:"ld_ratio_min"`       // Minimum L/D ratio
	LDRatioMax       float64 `yaml:"ld_ratio_max"`       // Maximum L/D ratio
	CDMin            float64 `yaml:"cd_range_min"`       // Drag coefficient range (optional)
	CDMax            float64 `yaml:"cd_range_max"`       // Drag coefficient range (optional)
	CLMin            float64 `yaml:"cl_range_min"`       // Lift coefficient range (optional)
	CLMax            float64 `yaml:"cl_range_max"`       // Lift coefficient range (optional)
	MaxMach          float64 `yaml:"max_mach"`           // Maximum operating Mach
	GlideAltitudeMin float64 `yaml:"glide_altitude_min"` // Min glide altitude (km)
	GlideAltitudeMax float64 `yaml:"glide_altitude_max"` // Max glide altitude (km)
}

// ThreatProfile represents a known threat in the intelligence database
type ThreatProfile struct {
	ID                   string       `yaml:"id"`                    // Unique identifier
	CommonName           string       `yaml:"common_name"`           // Human-readable name
	CountryCode          string       `yaml:"country_code"`          // CHN, RUS, etc.
	Aerodynamics         Aerodynamics `yaml:"aerodynamics"`          // Physics signature
	TacticalProfile      string       `yaml:"tactical_profile"`      // Behavior description
	IntentLabel          string       `yaml:"intent_label"`          // MARS_HYPERSONIC, etc.
	ThreatLevel          int          `yaml:"threat_level"`          // 1-10 scale
	CountermeasurePolicy string       `yaml:"countermeasure_policy"` // RL policy selector
	Notes                string       `yaml:"notes"`                 // Intelligence notes
}

// LibraryConfiguration controls library behavior
type LibraryConfiguration struct {
	AutoUpdateUCT         bool    `yaml:"auto_update_uct"`         // Auto-add new discoveries
	UCTCertaintyThreshold float64 `yaml:"uct_certainty_threshold"` // Min confidence to add
	MaxLibrarySize        int     `yaml:"max_library_size"`        // Size limit
	ValidationRequired    bool    `yaml:"validation_required"`     // Human review
}

// LibraryMetadata tracks library provenance
type LibraryMetadata struct {
	Version        string    `yaml:"version"`
	Classification string    `yaml:"classification"`
	LastUpdated    time.Time `yaml:"last_updated"`
	Authority      string    `yaml:"authority"`
	Distribution   string    `yaml:"distribution"`
}

// ThreatLibrary is the sovereign intelligence database
type ThreatLibrary struct {
	Threats       []ThreatProfile      `yaml:"threats"`
	Configuration LibraryConfiguration `yaml:"configuration"`
	Metadata      LibraryMetadata      `yaml:"metadata"`

	// Runtime state
	path  string // File path for persistence
	dirty bool   // Modified since load
}

// UCTRecord captures an Uncorrelated Track for analysis
type UCTRecord struct {
	ID              string    `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	DiscoveredLD    float64   `json:"discovered_ld"`
	DiscoveredCD    float64   `json:"discovered_cd"`
	DiscoveredCL    float64   `json:"discovered_cl"`
	Altitude        float64   `json:"altitude"`
	Velocity        float64   `json:"velocity"`
	MachNumber      float64   `json:"mach_number"`
	Certainty       float64   `json:"certainty"`        // SINDy confidence
	DiscoveryMethod string    `json:"discovery_method"` // "mrDMD+SINDy"
	Location        string    `json:"location"`         // S2 cell reference
}

// LoadThreatLibrary initializes the intelligence database
func LoadThreatLibrary(path string) (*ThreatLibrary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read threat library: %w", err)
	}

	var lib ThreatLibrary
	if err := yaml.Unmarshal(data, &lib); err != nil {
		return nil, fmt.Errorf("failed to parse threat library: %w", err)
	}

	lib.path = path
	lib.dirty = false

	fmt.Printf("[THREAT_LIBRARY] Loaded %d threat profiles from %s\n",
		len(lib.Threats), path)

	return &lib, nil
}

// Save persists the library to disk
func (tl *ThreatLibrary) Save() error {
	if !tl.dirty {
		return nil // No changes to save
	}

	// Update metadata
	tl.Metadata.LastUpdated = time.Now()

	data, err := yaml.Marshal(tl)
	if err != nil {
		return fmt.Errorf("failed to marshal library: %w", err)
	}

	if err := os.WriteFile(tl.path, data, 0600); err != nil {
		return fmt.Errorf("failed to write library: %w", err)
	}

	tl.dirty = false
	fmt.Printf("[THREAT_LIBRARY] Saved to %s\n", tl.path)
	return nil
}

// Identify performs fuzzy matching against the threat library
// Returns the closest match or nil for UCT (Uncorrelated Track)
func (tl *ThreatLibrary) Identify(
	discoveredLD float64,
	discoveredCD, discoveredCL float64,
	currentAlt, currentVel float64,
	confidence float64,
) (*ThreatMatch, error) {

	// Calculate Mach number
	mach := currentVel / 340.0 // Approximate

	// Score each threat profile
	var bestMatch *ThreatProfile
	bestScore := -1.0

	for i := range tl.Threats {
		threat := &tl.Threats[i]

		// Calculate match score (0-1, higher is better)
		score := tl.calculateMatchScore(
			threat,
			discoveredLD, discoveredCD, discoveredCL,
			currentAlt, mach,
		)

		if score > bestScore {
			bestScore = score
			bestMatch = threat
		}
	}

	// Threshold for positive identification
	// Below 0.7 confidence, classify as UCT
	if bestScore < 0.7 {
		// Record UCT for future analysis
		if tl.Configuration.AutoUpdateUCT {
			tl.recordUCT(UCTRecord{
				ID:              fmt.Sprintf("UCT_%d", time.Now().Unix()),
				Timestamp:       time.Now(),
				DiscoveredLD:    discoveredLD,
				DiscoveredCD:    discoveredCD,
				DiscoveredCL:    discoveredCL,
				Altitude:        currentAlt,
				Velocity:        currentVel,
				MachNumber:      mach,
				Certainty:       confidence,
				DiscoveryMethod: "mrDMD+SINDy",
			})
		}

		return &ThreatMatch{
			Profile:        nil,
			MatchScore:     bestScore,
			Classification: "UCT_UNCORRELATED_TRACK",
			Certainty:      confidence,
		}, nil
	}

	return &ThreatMatch{
		Profile:        bestMatch,
		MatchScore:     bestScore,
		Classification: bestMatch.IntentLabel,
		Certainty:      confidence * bestScore,
	}, nil
}

// ThreatMatch represents the identification result
type ThreatMatch struct {
	Profile        *ThreatProfile // nil for UCT
	MatchScore     float64        // 0-1 similarity score
	Classification string         // Intent label or UCT
	Certainty      float64        // Combined confidence
}

// calculateMatchScore computes similarity between discovery and threat profile
func (tl *ThreatLibrary) calculateMatchScore(
	threat *ThreatProfile,
	ld, cd, cl float64,
	altitude, mach float64,
) float64 {

	scores := make([]float64, 0)

	// 1. L/D ratio match (primary discriminator)
	ldCenter := (threat.Aerodynamics.LDRatioMin + threat.Aerodynamics.LDRatioMax) / 2
	ldRange := threat.Aerodynamics.LDRatioMax - threat.Aerodynamics.LDRatioMin
	ldDiff := abs(ld - ldCenter)
	ldScore := 1.0 - (ldDiff / (ldRange / 2))
	if ldScore < 0 {
		ldScore = 0
	}
	scores = append(scores, ldScore*0.5) // 50% weight

	// 2. Altitude envelope match
	altCenter := (threat.Aerodynamics.GlideAltitudeMin + threat.Aerodynamics.GlideAltitudeMax) / 2
	altRange := threat.Aerodynamics.GlideAltitudeMax - threat.Aerodynamics.GlideAltitudeMin
	altDiff := abs(altitude - altCenter)
	altScore := 1.0 - (altDiff / (altRange / 2))
	if altScore < 0 {
		altScore = 0
	}
	scores = append(scores, altScore*0.3) // 30% weight

	// 3. Mach capability check
	if mach <= threat.Aerodynamics.MaxMach {
		scores = append(scores, 0.2) // 20% weight
	}

	// 4. Optional: Cd/Cl refinement (if available)
	if cd > 0 && cl > 0 {
		calcLD := cl / cd
		calcCenter := (threat.Aerodynamics.LDRatioMin + threat.Aerodynamics.LDRatioMax) / 2
		calcDiff := abs(calcLD - calcCenter)
		calcScore := 1.0 - (calcDiff / (threat.Aerodynamics.LDRatioMax - threat.Aerodynamics.LDRatioMin))
		if calcScore > 0 {
			scores = append(scores, calcScore*0.1) // 10% bonus
		}
	}

	// Sum weighted scores
	total := 0.0
	for _, s := range scores {
		total += s
	}

	return total
}

// recordUCT logs an uncorrelated track for future analysis
func (tl *ThreatLibrary) recordUCT(uct UCTRecord) {
	// In production: Write to database, trigger analyst alert
	fmt.Printf("[THREAT_LIBRARY] UCT Recorded: L/D=%.2f, Alt=%.1fkm, Certainty=%.1f%%\n",
		uct.DiscoveredLD, uct.Altitude, uct.Certainty*100)

	// If certainty high enough and config allows, auto-add
	if uct.Certainty >= tl.Configuration.UCTCertaintyThreshold {
		tl.proposeNewThreat(uct)
	}
}

// proposeNewThreat suggests adding a new profile based on UCT
func (tl *ThreatLibrary) proposeNewThreat(uct UCTRecord) {
	// Create proposed threat (requires human validation)
	proposed := ThreatProfile{
		ID:          fmt.Sprintf("HGV_PROPOSED_%s", uct.ID),
		CommonName:  "Unknown HGV (Auto-Discovered)",
		CountryCode: "UNK",
		Aerodynamics: Aerodynamics{
			LDRatioMin:       uct.DiscoveredLD * 0.9,
			LDRatioMax:       uct.DiscoveredLD * 1.1,
			CDMin:            uct.DiscoveredCD * 0.8,
			CDMax:            uct.DiscoveredCD * 1.2,
			CLMin:            uct.DiscoveredCL * 0.8,
			CLMax:            uct.DiscoveredCL * 1.2,
			GlideAltitudeMin: uct.Altitude - 5,
			GlideAltitudeMax: uct.Altitude + 5,
		},
		IntentLabel: "MARS_UNKNOWN_AUTO",
		ThreatLevel: 8, // Unknown = high caution
		Notes:       fmt.Sprintf("Auto-discovered. Certainty: %.1f%%", uct.Certainty*100),
	}

	if !tl.Configuration.ValidationRequired {
		// Auto-add if validation not required
		tl.Threats = append(tl.Threats, proposed)
		tl.dirty = true
		fmt.Printf("[THREAT_LIBRARY] Auto-added new threat: %s\n", proposed.ID)
	} else {
		// Queue for human validation
		fmt.Printf("[THREAT_LIBRARY] Proposed new threat (awaiting validation): %s\n", proposed.ID)
	}
}

// GetThreatByID retrieves a specific threat profile
func (tl *ThreatLibrary) GetThreatByID(id string) (*ThreatProfile, bool) {
	for i := range tl.Threats {
		if tl.Threats[i].ID == id {
			return &tl.Threats[i], true
		}
	}
	return nil, false
}

// ListThreatsByIntent returns all threats matching an intent label
func (tl *ThreatLibrary) ListThreatsByIntent(intent string) []ThreatProfile {
	results := make([]ThreatProfile, 0)
	for _, t := range tl.Threats {
		if t.IntentLabel == intent {
			results = append(results, t)
		}
	}
	return results
}

// GetAllThreats returns the complete intelligence database
func (tl *ThreatLibrary) GetAllThreats() []ThreatProfile {
	return tl.Threats
}

// GetMetadata returns library provenance information
func (tl *ThreatLibrary) GetMetadata() LibraryMetadata {
	return tl.Metadata
}

// Helper: absolute value for float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
