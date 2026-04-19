package threat

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

/**
 * Prior Store: Learning Intelligence for Bayesian Threat Classification
 *
 * Transforms the Bayesian Resolver from static to adaptive by tracking
 * historical encounter frequencies. Provides "Regional Bias" - knowing
 * that L/D=1.8 in Siliguri is more likely DF-ZF than a prototype based
 * on historical data.
 *
 * Features:
 * - Persistent JSON-backed storage
 * - Thread-safe concurrent access
 * - Dirichlet-inspired prior with Laplace smoothing
 * - Regional encounter tracking
 * - Self-updating intelligence
 *
 * Reference: Dirichlet Distribution for Bayesian Priors
 */

// PriorStore manages historical encounter frequencies for Bayesian priors
type PriorStore struct {
	mu       sync.RWMutex
	FilePath string

	// Core data: threat encounter counts
	Counts map[string]int `json:"counts"`

	// Metadata for intelligence tracking
	Region          string    `json:"region"` // e.g., "Siliguri_Corridor"
	LastUpdated     time.Time `json:"last_updated"`
	TotalEncounters int       `json:"total_encounters"`

	// Regional threat prevalence (learned over time)
	RegionalBias map[string]float64 `json:"regional_bias"`
}

// EncounterRecord represents a single threat sighting
type EncounterRecord struct {
	ThreatID   string    `json:"threat_id"`
	Timestamp  time.Time `json:"timestamp"`
	Location   string    `json:"location"`    // S2 cell reference
	LDMeasured float64   `json:"ld_measured"` // Discovered L/D
	Confidence float64   `json:"confidence"`  // Classification confidence
	Altitude   float64   `json:"altitude"`    // km
	Verified   bool      `json:"verified"`    // Human/ground truth verified
}

// PriorStats provides analytics on the prior distribution
type PriorStats struct {
	TotalEncounters  int
	UniqueThreats    int
	MostCommon       string
	MostCommonCount  int
	Rarest           string
	RarestCount      int
	RegionalDominant string
	Entropy          float64
}

// NewPriorStore creates or loads a persistent prior store
func NewPriorStore(path string, region string) (*PriorStore, error) {
	store := &PriorStore{
		FilePath:        path,
		Counts:          make(map[string]int),
		Region:          region,
		RegionalBias:    make(map[string]float64),
		LastUpdated:     time.Now(),
		TotalEncounters: 0,
	}

	// Try to load existing store
	if _, err := os.Stat(path); err == nil {
		data, err := os.ReadFile(path)
		if err == nil {
			if err := json.Unmarshal(data, store); err == nil {
				fmt.Printf("[PRIOR_STORE] Loaded %d encounters from %s\n",
					store.TotalEncounters, path)
				return store, nil
			}
		}
	}

	// Create new store with default regional priors
	store.initializeRegionalPriors(region)

	if err := store.Save(); err != nil {
		return nil, fmt.Errorf("failed to save new store: %w", err)
	}

	fmt.Printf("[PRIOR_STORE] Created new store for region: %s\n", region)
	return store, nil
}

// initializeRegionalPriors sets sensible defaults based on region
func (ps *PriorStore) initializeRegionalPriors(region string) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	switch region {
	case "Siliguri_Corridor", "Eastern_Front":
		// Higher prior for Chinese HGVs near Chicken's Neck
		ps.Counts["HGV_TYPE_A_REGIONAL"] = 15    // DF-ZF
		ps.Counts["HGV_TYPE_B_EXPERIMENTAL"] = 8 // Starry Sky-2
		ps.Counts["HGV_TYPE_D_TACTICAL"] = 5     // Tactical
		ps.Counts["BM_TYPE_A_SRB"] = 10          // Ballistic baseline
		ps.RegionalBias["HGV_TYPE_A_REGIONAL"] = 0.35

	case "Western_Front", "Rajasthan":
		// Mix of Pakistani and Chinese threats
		ps.Counts["HGV_TYPE_A_REGIONAL"] = 10
		ps.Counts["HGV_TYPE_D_TACTICAL"] = 12
		ps.Counts["BM_TYPE_A_SRB"] = 15
		ps.RegionalBias["HGV_TYPE_D_TACTICAL"] = 0.30

	case "Northern_Front", "Tibetan":
		// Russian/Chinese high-altitude threats
		ps.Counts["HGV_TYPE_C_RUSSIAN"] = 8 // Avangard
		ps.Counts["HGV_TYPE_A_REGIONAL"] = 12
		ps.Counts["HGV_TYPE_E_NEXTGEN"] = 3 // Unknown prototypes
		ps.RegionalBias["HGV_TYPE_C_RUSSIAN"] = 0.25

	default:
		// Uniform priors for unknown regions
		ps.Counts["HGV_TYPE_A_REGIONAL"] = 5
		ps.Counts["BM_TYPE_A_SRB"] = 5
	}

	// Calculate total
	ps.TotalEncounters = 0
	for _, count := range ps.Counts {
		ps.TotalEncounters += count
	}
}

// GetPrior returns P(Threat) using Dirichlet-inspired distribution with Laplace smoothing
// Formula: P(Threat_i) = (count_i + α) / (total + α·K)
// where α=1 for Laplace smoothing, K=number of threat types
func (ps *PriorStore) GetPrior(threatID string) float64 {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	alpha := 1.0 // Laplace smoothing parameter

	// Count unique threat types
	K := len(ps.Counts)
	if K == 0 {
		K = 1
	}

	// Get count for this threat (with smoothing)
	count, exists := ps.Counts[threatID]
	if !exists {
		count = 0
	}

	// Apply regional bias if available
	bias := 1.0
	if regionalBias, exists := ps.RegionalBias[threatID]; exists {
		// Blend empirical count with regional expectation
		// 70% empirical, 30% regional bias
		empiricalPrior := float64(count+1) / float64(ps.TotalEncounters+K)
		bias = 0.7*empiricalPrior + 0.3*regionalBias
	}

	// Dirichlet-smoothed prior
	numerator := float64(count) + alpha*bias
	denominator := float64(ps.TotalEncounters) + alpha*float64(K)

	if denominator == 0 {
		return 1.0 / float64(K) // Uniform if no data
	}

	return numerator / denominator
}

// GetAllPriors returns the complete prior distribution
func (ps *PriorStore) GetAllPriors() map[string]float64 {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	priors := make(map[string]float64)

	// Include all known threats
	for threatID := range ps.Counts {
		priors[threatID] = ps.getPriorUnlocked(threatID)
	}

	return priors
}

// getPriorUnlocked internal version without lock (caller must hold lock)
func (ps *PriorStore) getPriorUnlocked(threatID string) float64 {
	alpha := 1.0
	K := len(ps.Counts)
	if K == 0 {
		K = 1
	}

	count := ps.Counts[threatID]
	numerator := float64(count) + alpha
	denominator := float64(ps.TotalEncounters) + alpha*float64(K)

	if denominator == 0 {
		return 1.0 / float64(K)
	}

	return numerator / denominator
}

// RecordEncounter records a confirmed threat identification
// This is the learning mechanism - updates priors for future classifications
func (ps *PriorStore) RecordEncounter(record EncounterRecord) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Update counts
	ps.Counts[record.ThreatID]++
	ps.TotalEncounters++
	ps.LastUpdated = time.Now()

	// Update regional bias (exponential moving average)
	// New bias = 0.9 * old bias + 0.1 * new encounter
	oldBias := ps.RegionalBias[record.ThreatID]
	ps.RegionalBias[record.ThreatID] = 0.9*oldBias + 0.1*(1.0/float64(len(ps.Counts)))

	// Persist to disk
	if err := ps.saveUnlocked(); err != nil {
		return fmt.Errorf("failed to save encounter: %w", err)
	}

	fmt.Printf("[PRIOR_STORE] Recorded encounter: %s at %s (total: %d)\n",
		record.ThreatID, record.Location, ps.TotalEncounters)

	return nil
}

// RecordUCT records an uncorrelated track for potential future classification
// If UCT is repeatedly seen, it may become a new threat type
func (ps *PriorStore) RecordUCT(uctID string, ldMeasured, altitude float64) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Track UCT separately
	uctKey := "UCT_" + uctID
	ps.Counts[uctKey]++

	// If UCT seen >5 times with similar L/D, consider it emerging threat
	if ps.Counts[uctKey] >= 5 {
		fmt.Printf("[PRIOR_STORE] Emerging threat pattern detected: %s (L/D=%.2f, n=%d)\n",
			uctID, ldMeasured, ps.Counts[uctKey])
	}

	ps.saveUnlocked()
}

// Save persists the store to disk
func (ps *PriorStore) Save() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	return ps.saveUnlocked()
}

// saveUnlocked internal version (caller must hold lock)
func (ps *PriorStore) saveUnlocked() error {
	data, err := json.MarshalIndent(ps, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(ps.FilePath, data, 0644)
}

// GetStats returns analytics on the prior distribution
func (ps *PriorStore) GetStats() PriorStats {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	stats := PriorStats{
		TotalEncounters: ps.TotalEncounters,
		UniqueThreats:   len(ps.Counts),
	}

	if len(ps.Counts) == 0 {
		return stats
	}

	// Find most and least common
	maxCount := -1
	minCount := 999999

	for threatID, count := range ps.Counts {
		if count > maxCount {
			maxCount = count
			stats.MostCommon = threatID
			stats.MostCommonCount = count
		}
		if count < minCount {
			minCount = count
			stats.Rarest = threatID
			stats.RarestCount = count
		}
	}

	// Calculate entropy of distribution
	stats.Entropy = ps.calculateEntropyUnlocked()

	// Regional dominant
	maxBias := 0.0
	for threatID, bias := range ps.RegionalBias {
		if bias > maxBias {
			maxBias = bias
			stats.RegionalDominant = threatID
		}
	}

	return stats
}

// calculateEntropyUnlocked calculates Shannon entropy of prior distribution
func (ps *PriorStore) calculateEntropyUnlocked() float64 {
	entropy := 0.0

	for threatID := range ps.Counts {
		p := ps.getPriorUnlocked(threatID)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// ThreatPrior represents a threat with its prior probability
type ThreatPrior struct {
	ThreatID string
	Prior    float64
	Count    int
}

// GetTopThreats returns the N most likely threats by prior probability
func (ps *PriorStore) GetTopThreats(n int) []ThreatPrior {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	priors := make([]ThreatPrior, 0, len(ps.Counts))
	for threatID, count := range ps.Counts {
		priors = append(priors, ThreatPrior{
			ThreatID: threatID,
			Prior:    ps.getPriorUnlocked(threatID),
			Count:    count,
		})
	}

	// Sort by prior (descending)
	sort.Slice(priors, func(i, j int) bool {
		return priors[i].Prior > priors[j].Prior
	})

	// Return top N
	if n > len(priors) {
		n = len(priors)
	}

	return priors[:n]
}

// Merge combines another prior store into this one
// Useful for aggregating intelligence across multiple sensors/nodes
func (ps *PriorStore) Merge(other *PriorStore) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	// Merge counts
	for threatID, count := range other.Counts {
		ps.Counts[threatID] += count
		ps.TotalEncounters += count
	}

	// Merge regional bias (weighted average)
	for threatID, bias := range other.RegionalBias {
		if existing, exists := ps.RegionalBias[threatID]; exists {
			ps.RegionalBias[threatID] = (existing + bias) / 2
		} else {
			ps.RegionalBias[threatID] = bias
		}
	}

	ps.LastUpdated = time.Now()

	return ps.saveUnlocked()
}

// Reset clears all encounter history (use with caution)
func (ps *PriorStore) Reset() error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	ps.Counts = make(map[string]int)
	ps.TotalEncounters = 0
	ps.RegionalBias = make(map[string]float64)
	ps.LastUpdated = time.Now()

	return ps.saveUnlocked()
}

// ImportJSON loads prior data from external intelligence feed
func (ps *PriorStore) ImportJSON(jsonData []byte) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	var imported struct {
		Counts       map[string]int     `json:"counts"`
		RegionalBias map[string]float64 `json:"regional_bias"`
	}

	if err := json.Unmarshal(jsonData, &imported); err != nil {
		return fmt.Errorf("failed to import: %w", err)
	}

	// Merge with existing
	for threatID, count := range imported.Counts {
		ps.Counts[threatID] += count
		ps.TotalEncounters += count
	}

	for threatID, bias := range imported.RegionalBias {
		ps.RegionalBias[threatID] = bias
	}

	return ps.saveUnlocked()
}

// ExportJSON exports prior data for sharing with other nodes
func (ps *PriorStore) ExportJSON() ([]byte, error) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	return json.MarshalIndent(ps, "", "  ")
}

// Example Usage:
// store, _ := threat.NewPriorStore("data/siliguri_priors.json", "Siliguri_Corridor")
//
// // Record confirmed encounter
// store.RecordEncounter(threat.EncounterRecord{
//     ThreatID: "HGV_TYPE_A_REGIONAL",
//     Location: "S2:123456789",
//     LDMeasured: 2.05,
// })
//
// // Get prior for Bayesian classification
// prior := store.GetPrior("HGV_TYPE_A_REGIONAL")  // Returns 0.35 if DF-ZF is common
