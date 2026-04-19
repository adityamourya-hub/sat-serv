// Package sigint provides SIGID Wiki integration for signal identification
// Implements spectral template matching for rapid signal identification
package sigint

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
)

// SIGIDTemplate represents a signal template from SIGID Wiki
type SIGIDTemplate struct {
	ID           string    `json:"id"`
	Name         string    `json:"name"`
	Category     string    `json:"category"`     // Military, Satellite, Utility, Unknown
	FrequencyHz  float64   `json:"frequency_hz"` // Typical center frequency
	BandwidthHz  float64   `json:"bandwidth_hz"`
	Modulation   string    `json:"modulation"`   // QPSK, FSK, OFDM, etc.
	PATTERN      string    `json:"pattern"`      // Visual pattern description
	PSDMask      []byte    `json:"psd_mask"`     // 2D bit-mask for spectral matching
	PRI          float64   `json:"pri"`          // Pulse Repetition Interval (ms)
	ThreatLevel  int       `json:"threat_level"` // 1-5
	LastUpdated  time.Time `json:"last_updated"`
}

// SIGIDBridge provides spectral template matching and signal identification
type SIGIDBridge struct {
	db        *sql.DB
	templates map[string]SIGIDTemplate
	cachePath string
}

// NewSIGIDBridge creates a new SIGID bridge with local database
func NewSIGIDBridge(cachePath string) (*SIGIDBridge, error) {
	// Ensure cache directory exists
	if err := os.MkdirAll(cachePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	dbPath := filepath.Join(cachePath, "sigid_templates.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	bridge := &SIGIDBridge{
		db:        db,
		templates: make(map[string]SIGIDTemplate),
		cachePath: cachePath,
	}

	// Initialize database schema
	if err := bridge.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Load default military templates
	if err := bridge.loadDefaultTemplates(); err != nil {
		return nil, fmt.Errorf("failed to load default templates: %w", err)
	}

	return bridge, nil
}

// initSchema creates the database schema
func (sb *SIGIDBridge) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS templates (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL,
		category TEXT NOT NULL,
		frequency_hz REAL,
		bandwidth_hz REAL,
		modulation TEXT,
		pattern TEXT,
		psd_mask BLOB,
		pri REAL,
		threat_level INTEGER,
		last_updated TIMESTAMP
	);

	CREATE INDEX IF NOT EXISTS idx_category ON templates(category);
	CREATE INDEX IF NOT EXISTS idx_frequency ON templates(frequency_hz);
	CREATE INDEX IF NOT EXISTS idx_threat ON templates(threat_level);
	`

	_, err := sb.db.Exec(schema)
	return err
}

// loadDefaultTemplates loads built-in military signal templates
func (sb *SIGIDBridge) loadDefaultTemplates() error {
	// Check if templates already exist
	var count int
	err := sb.db.QueryRow("SELECT COUNT(*) FROM templates").Scan(&count)
	if err != nil {
		return err
	}
	if count > 0 {
		return nil // Templates already loaded
	}

	templates := []SIGIDTemplate{
		{
			ID:          "LINK_16",
			Name:        "Link 16",
			Category:    "Military",
			FrequencyHz: 1087.5e6,
			BandwidthHz: 237e6,
			Modulation:  "QPSK",
			PATTERN:     "Spike Train",
			PRI:         0.0, // Continuous
			ThreatLevel: 2,
			LastUpdated: time.Now(),
		},
		{
			ID:          "HAVE_QUICK",
			Name:        "HAVE QUICK II",
			Category:    "Military",
			FrequencyHz: 312.5e6,
			BandwidthHz: 175e6,
			Modulation:  "FHSS",
			PATTERN:     "Frequency Hopping",
			PRI:         0.0,
			ThreatLevel: 2,
			LastUpdated: time.Now(),
		},
		{
			ID:          "SEVER_GORIZONT",
			Name:        "Sever/Gorizont Troposcatter",
			Category:    "Military",
			FrequencyHz: 4592.5e6,
			BandwidthHz: 315e6,
			Modulation:  "QPSK",
			PATTERN:     "Hollow Tropo",
			PRI:         0.0,
			ThreatLevel: 4,
			LastUpdated: time.Now(),
		},
		{
			ID:          "PROFORMA_X",
			Name:        "PROFORMA X-Band",
			Category:    "Military",
			FrequencyHz: 10.0e9,
			BandwidthHz: 4e9,
			Modulation:  "OFDM",
			PATTERN:     "Wideband Data",
			PRI:         0.25, // 0.25 ms PRI
			ThreatLevel: 5,
			LastUpdated: time.Now(),
		},
		{
			ID:          "TETRA",
			Name:        "TETRA Police/Utility",
			Category:    "Utility",
			FrequencyHz: 420e6,
			BandwidthHz: 25e3,
			Modulation:  "PI/4-DQPSK",
			PATTERN:     "TETRA Bursts",
			PRI:         14.167, // TDMA frame
			ThreatLevel: 3,
			LastUpdated: time.Now(),
		},
		{
			ID:          "INMARSAT",
			Name:        "Inmarsat Satellite",
			Category:    "Satellite",
			FrequencyHz: 1.6e9,
			BandwidthHz: 40e6,
			Modulation:  "QPSK",
			PATTERN:     "Satellite Uplink",
			PRI:         0.0,
			ThreatLevel: 1,
			LastUpdated: time.Now(),
		},
	}

	for _, template := range templates {
		if err := sb.AddTemplate(template); err != nil {
			return err
		}
	}

	return nil
}

// AddTemplate adds a new signal template to the database
func (sb *SIGIDBridge) AddTemplate(template SIGIDTemplate) error {
	psdMaskJSON, _ := json.Marshal(template.PSDMask)

	_, err := sb.db.Exec(
		`INSERT OR REPLACE INTO templates 
		(id, name, category, frequency_hz, bandwidth_hz, modulation, pattern, psd_mask, pri, threat_level, last_updated)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		template.ID, template.Name, template.Category,
		template.FrequencyHz, template.BandwidthHz,
		template.Modulation, template.PATTERN,
		psdMaskJSON, template.PRI, template.ThreatLevel,
		template.LastUpdated,
	)

	if err == nil {
		sb.templates[template.ID] = template
	}

	return err
}

// MatchWaterfall performs spectral template matching using Hamming distance
func (sb *SIGIDBridge) MatchWaterfall(detectedMask []byte, freqHz float64, bandwidthHz float64) (float64, string) {
	bestMatch := 0.0
	bestID := "UNKNOWN"

	// Query templates in frequency range
	rows, err := sb.db.Query(
		`SELECT id, psd_mask, pattern FROM templates 
		WHERE frequency_hz BETWEEN ? AND ?`,
		freqHz-bandwidthHz, freqHz+bandwidthHz,
	)
	if err != nil {
		return 0.0, "UNKNOWN"
	}
	defer rows.Close()

	for rows.Next() {
		var id, pattern string
		var psdMaskJSON []byte
		if err := rows.Scan(&id, &psdMaskJSON, &pattern); err != nil {
			continue
		}

		// Decode PSD mask
		var templateMask []byte
		if err := json.Unmarshal(psdMaskJSON, &templateMask); err != nil {
			// If no mask stored, use pattern matching
			continue
		}

		// Calculate Hamming distance
		similarity := sb.calculateSimilarity(detectedMask, templateMask)

		if similarity > bestMatch && similarity > 0.70 {
			bestMatch = similarity
			bestID = id
		}
	}

	return bestMatch, bestID
}

// calculateSimilarity computes Hamming distance between two masks
func (sb *SIGIDBridge) calculateSimilarity(maskA, maskB []byte) float64 {
	if len(maskA) != len(maskB) {
		// Pad shorter mask
		maxLen := int(math.Max(float64(len(maskA)), float64(len(maskB))))
		if len(maskA) < maxLen {
			padded := make([]byte, maxLen)
			copy(padded, maskA)
			maskA = padded
		}
		if len(maskB) < maxLen {
			padded := make([]byte, maxLen)
			copy(padded, maskB)
			maskB = padded
		}
	}

	// Count matching bits
	matches := 0
	for i := range maskA {
		// XOR to find differences, then count zero bits
		diff := maskA[i] ^ maskB[i]
		matches += 8 - countBits(diff)
	}

	totalBits := len(maskA) * 8
	return float64(matches) / float64(totalBits)
}

// countBits counts the number of set bits in a byte
func countBits(b byte) int {
	count := 0
	for i := 0; i < 8; i++ {
		if b&(1<<i) != 0 {
			count++
		}
	}
	return count
}

// LookupID retrieves full template details by ID
func (sb *SIGIDBridge) LookupID(id string) (*SIGIDTemplate, error) {
	if template, ok := sb.templates[id]; ok {
		return &template, nil
	}

	// Query from database
	var template SIGIDTemplate
	var psdMaskJSON []byte

	err := sb.db.QueryRow(
		`SELECT id, name, category, frequency_hz, bandwidth_hz, 
		modulation, pattern, psd_mask, pri, threat_level, last_updated
		FROM templates WHERE id = ?`,
		id,
	).Scan(
		&template.ID, &template.Name, &template.Category,
		&template.FrequencyHz, &template.BandwidthHz,
		&template.Modulation, &template.PATTERN,
		&psdMaskJSON, &template.PRI, &template.ThreatLevel,
		&template.LastUpdated,
	)

	if err != nil {
		return nil, err
	}

	json.Unmarshal(psdMaskJSON, &template.PSDMask)
	sb.templates[id] = template

	return &template, nil
}

// MatchModulationChecksum validates pulse parameters against known templates
func (sb *SIGIDBridge) MatchModulationChecksum(freqHz float64, pri float64, modulation string) (float64, string) {
	// Query templates by frequency and modulation
	rows, err := sb.db.Query(
		`SELECT id, pri, threat_level FROM templates 
		WHERE frequency_hz BETWEEN ? AND ? AND modulation = ?`,
		freqHz*0.95, freqHz*1.05, modulation,
	)
	if err != nil {
		return 0.0, "UNKNOWN"
	}
	defer rows.Close()

	bestMatch := 0.0
	bestID := "UNKNOWN"

	for rows.Next() {
		var id string
			var templatePRI float64
		var threatLevel int
		if err := rows.Scan(&id, &templatePRI, &threatLevel); err != nil {
			continue
		}

		// PRI matching tolerance: +/- 5%
		if templatePRI > 0 {
			priDiff := math.Abs(pri-templatePRI) / templatePRI
			if priDiff < 0.05 {
				// Good PRI match
				matchScore := 1.0 - priDiff
				if matchScore > bestMatch {
					bestMatch = matchScore
					bestID = id
				}
			}
		}
	}

	return bestMatch, bestID
}

// GetTemplatesByCategory retrieves templates by category
func (sb *SIGIDBridge) GetTemplatesByCategory(category string) ([]SIGIDTemplate, error) {
	rows, err := sb.db.Query(
		`SELECT id, name, category, frequency_hz, bandwidth_hz, 
		modulation, pattern, pri, threat_level, last_updated
		FROM templates WHERE category = ?`,
		category,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	templates := []SIGIDTemplate{}
	for rows.Next() {
		var template SIGIDTemplate
		err := rows.Scan(
			&template.ID, &template.Name, &template.Category,
			&template.FrequencyHz, &template.BandwidthHz,
			&template.Modulation, &template.PATTERN,
			&template.PRI, &template.ThreatLevel,
			&template.LastUpdated,
		)
		if err != nil {
			continue
		}
		templates = append(templates, template)
	}

	return templates, nil
}

// Close closes the database connection
func (sb *SIGIDBridge) Close() error {
	return sb.db.Close()
}
