package catalog

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

/**
 * ITU Filing Database Support
 *
 * Implements ITU-R S.1645 compliant orbital slot management for:
 * - Frequency coordination and interference analysis
 * - Orbital slot registration and verification
 * - Notification of new satellite networks (API)
 * - Coordination arc calculations
 *
 * Reference: ITU Radio Regulations, Article 9 and 11
 * - S.1645: Methodologies for calculating epfd (equivalent power flux-density)
 */

// ITUFilingStatus represents the status of an ITU filing
type ITUFilingStatus string

const (
	ITUStatusPlanned        ITUFilingStatus = "Planned"
	ITUStatusNotified       ITUFilingStatus = "Notified"       // API received
	ITUStatusCoordinated    ITUFilingStatus = "Coordinated"    // Successfully coordinated
	ITUStatusBroughtIntoUse ITUFilingStatus = "BroughtIntoUse" // BIU achieved
	ITUStatusSuspended      ITUFilingStatus = "Suspended"
	ITUStatusCancelled      ITUFilingStatus = "Cancelled"
)

// OrbitalSlot represents a geostationary orbital position
type OrbitalSlot struct {
	Longitude      float64 // Degrees East (-180 to 180)
	Latitude       float64 // Always ~0 for GEO, but included for inclined GEO
	AltitudeKm     float64 // GEO altitude ~35786 km
	HalfArcDeg     float64 // Coordination arc half-width (typically ±0.1° for GSO)
	FrequencyBands []FrequencyBand
	ITUReference   string // ITU filing reference (e.g., "IND12345")
}

// FrequencyBand defines spectrum allocation
type FrequencyBand struct {
	LowerMHz float64
	UpperMHz float64
	Service  string // FSS, BSS, MSS, etc.
	Region   string // Region 1, 2, or 3
}

// ITUFiling represents a complete ITU satellite network filing
type ITUFiling struct {
	ID              string
	ITUReference    string // API/Arc/Coord number
	Administration  string // Country code (e.g., "IND" for India)
	SatelliteName   string
	Status          ITUFilingStatus
	FilingDate      time.Time
	BringingIntoUse *time.Time // BIU deadline
	OrbitalSlots    []OrbitalSlot
	FrequencyBands  []FrequencyBand
	NetworkType     string // GSO, non-GSO, etc.
	BeamPatterns    []BeamPattern
	EPFDMasks       []EPFDMask
	CoordinationArc CoordinationArc
}

// BeamPattern describes antenna coverage
type BeamPattern struct {
	BeamID       string
	BeamName     string
	SatelliteID  string
	ServiceArea  GeoPolygon
	EIRPdBW      float64
	FrequencyMHz float64
}

// GeoPolygon represents a geographic region
type GeoPolygon struct {
	Vertices [][2]float64 // [lon, lat] pairs
}

// EPFDMask defines equivalent power flux-density limits
type EPFDMask struct {
	FrequencyBand FrequencyBand
	EPFDLimits    []EPFDPoint // dB(W/m²) vs angle
}

// EPFDPoint is a single EPFD constraint
type EPFDPoint struct {
	AngleDeg float64 // Off-axis angle
	EPFDdBW  float64 // EPFD limit
}

// CoordinationArc defines the arc for coordination calculations
type CoordinationArc struct {
	CenterLongitude float64
	MinLongitude    float64
	MaxLongitude    float64
	HalfWidthDeg    float64
}

// ITUFilingDatabase manages ITU filings for orbital slot coordination
type ITUFilingDatabase struct {
	mu      sync.RWMutex
	filings map[string]*ITUFiling // By ID
	byAdmin map[string][]string   // Administration -> filing IDs
	bySlot  map[float64][]string  // Longitude -> filing IDs (for GEO)
}

// NewITUFilingDatabase creates a new ITU filing database
func NewITUFilingDatabase() *ITUFilingDatabase {
	return &ITUFilingDatabase{
		filings: make(map[string]*ITUFiling),
		byAdmin: make(map[string][]string),
		bySlot:  make(map[float64][]string),
	}
}

// AddFiling adds or updates an ITU filing
func (db *ITUFilingDatabase) AddFiling(filing *ITUFiling) error {
	if filing.ID == "" {
		return fmt.Errorf("filing ID required")
	}
	if filing.ITUReference == "" {
		return fmt.Errorf("ITU reference required")
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Remove old slot mappings if updating
	if existing, ok := db.filings[filing.ID]; ok {
		for _, slot := range existing.OrbitalSlots {
			db.removeFromSlotIndex(filing.ID, slot.Longitude)
		}
	}

	// Store filing
	db.filings[filing.ID] = filing

	// Index by administration
	adminKey := filing.Administration
	db.byAdmin[adminKey] = append(db.byAdmin[adminKey], filing.ID)

	// Index by orbital slots (for GEO)
	for _, slot := range filing.OrbitalSlots {
		if slot.Latitude == 0 && math.Abs(slot.AltitudeKm-35786) < 500 {
			// GEO slot - index by longitude
			db.bySlot[slot.Longitude] = append(db.bySlot[slot.Longitude], filing.ID)
		}
	}

	return nil
}

// removeFromSlotIndex removes a filing from the slot index
func (db *ITUFilingDatabase) removeFromSlotIndex(filingID string, longitude float64) {
	slots := db.bySlot[longitude]
	for i, id := range slots {
		if id == filingID {
			db.bySlot[longitude] = append(slots[:i], slots[i+1:]...)
			break
		}
	}
}

// GetFiling retrieves a filing by ID
func (db *ITUFilingDatabase) GetFiling(id string) (*ITUFiling, bool) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	filing, ok := db.filings[id]
	return filing, ok
}

// GetFilingsByAdministration returns all filings for an administration
func (db *ITUFilingDatabase) GetFilingsByAdministration(admin string) []*ITUFiling {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var result []*ITUFiling
	ids := db.byAdmin[admin]
	for _, id := range ids {
		if filing, ok := db.filings[id]; ok {
			result = append(result, filing)
		}
	}
	return result
}

// GetFilingsBySlot returns filings for a specific GEO longitude
func (db *ITUFilingDatabase) GetFilingsBySlot(longitude float64, toleranceDeg float64) []*ITUFiling {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var result []*ITUFiling

	// Check slots within tolerance
	for slotLon, ids := range db.bySlot {
		if math.Abs(slotLon-longitude) <= toleranceDeg {
			for _, id := range ids {
				if filing, ok := db.filings[id]; ok {
					result = append(result, filing)
				}
			}
		}
	}

	return result
}

// FindInterferences identifies potential coordination issues
func (db *ITUFilingDatabase) FindInterferences(filing *ITUFiling) []InterferenceAnalysis {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var interferences []InterferenceAnalysis

	for _, slot := range filing.OrbitalSlots {
		// Find nearby filings (±2° for GEO coordination arc)
		nearby := db.GetFilingsBySlot(slot.Longitude, 2.0)

		for _, other := range nearby {
			if other.ID == filing.ID {
				continue // Skip self
			}

			// Check frequency overlap
			overlap := db.checkFrequencyOverlap(filing, other)
			if len(overlap) > 0 {
				analysis := InterferenceAnalysis{
					Filing1:              filing.ID,
					Filing2:              other.ID,
					SeparationDeg:        math.Abs(slot.Longitude - other.OrbitalSlots[0].Longitude),
					OverlappingFreq:      overlap,
					RequiresCoordination: true,
				}
				interferences = append(interferences, analysis)
			}
		}
	}

	return interferences
}

// InterferenceAnalysis describes a potential coordination conflict
type InterferenceAnalysis struct {
	Filing1              string
	Filing2              string
	SeparationDeg        float64
	OverlappingFreq      []FrequencyOverlap
	RequiresCoordination bool
	EPFDCompliance       bool
}

// FrequencyOverlap describes overlapping spectrum
type FrequencyOverlap struct {
	Band1    FrequencyBand
	Band2    FrequencyBand
	Lower    float64
	Upper    float64
	WidthMHz float64
}

// checkFrequencyOverlap identifies overlapping frequency bands
func (db *ITUFilingDatabase) checkFrequencyOverlap(f1, f2 *ITUFiling) []FrequencyOverlap {
	var overlaps []FrequencyOverlap

	for _, b1 := range f1.FrequencyBands {
		for _, b2 := range f2.FrequencyBands {
			// Check if bands overlap
			lower := math.Max(b1.LowerMHz, b2.LowerMHz)
			upper := math.Min(b1.UpperMHz, b2.UpperMHz)

			if upper > lower {
				overlap := FrequencyOverlap{
					Band1:    b1,
					Band2:    b2,
					Lower:    lower,
					Upper:    upper,
					WidthMHz: upper - lower,
				}
				overlaps = append(overlaps, overlap)
			}
		}
	}

	return overlaps
}

// CalculateCoordinationArc computes the coordination arc per ITU-R S.1645
func CalculateCoordinationArc(longitude float64, frequencyGHz float64) CoordinationArc {
	// Arc width depends on frequency (higher freq = narrower beam = smaller arc)
	// S.1645 formula: θ ≈ 1.15° for 4 GHz, ~0.4° for 20 GHz
	baseHalfWidth := 0.5 // degrees
	if frequencyGHz > 10 {
		baseHalfWidth = 0.2
	}

	return CoordinationArc{
		CenterLongitude: longitude,
		MinLongitude:    longitude - baseHalfWidth,
		MaxLongitude:    longitude + baseHalfWidth,
		HalfWidthDeg:    baseHalfWidth,
	}
}

// CheckEPFDCompliance verifies EPFD mask compliance
func CheckEPFDCompliance(angleDeg, epfdDBW float64, mask EPFDMask) bool {
	// Linear interpolation of mask
	for i := 0; i < len(mask.EPFDLimits)-1; i++ {
		p1 := mask.EPFDLimits[i]
		p2 := mask.EPFDLimits[i+1]

		if angleDeg >= p1.AngleDeg && angleDeg <= p2.AngleDeg {
			// Linear interpolation
			t := (angleDeg - p1.AngleDeg) / (p2.AngleDeg - p1.AngleDeg)
			limit := p1.EPFDdBW + t*(p2.EPFDdBW-p1.EPFDdBW)
			return epfdDBW <= limit
		}
	}

	return false
}

// MarshalJSON serializes database to JSON
func (db *ITUFilingDatabase) MarshalJSON() ([]byte, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	return json.Marshal(struct {
		Filings []ITUFiling `json:"filings"`
		Count   int         `json:"count"`
	}{
		Filings: db.getAllFilings(),
		Count:   len(db.filings),
	})
}

// getAllFilings returns all filings (must hold read lock)
func (db *ITUFilingDatabase) getAllFilings() []ITUFiling {
	result := make([]ITUFiling, 0, len(db.filings))
	for _, filing := range db.filings {
		result = append(result, *filing)
	}
	return result
}

// SlotConflictDetector identifies slot conflicts for mission planning
type SlotConflictDetector struct {
	MinimumSeparationDeg float64 // Typically 0.1° for GSO
}

// NewSlotConflictDetector creates a detector with default separation
func NewSlotConflictDetector() *SlotConflictDetector {
	return &SlotConflictDetector{
		MinimumSeparationDeg: 0.1, // Standard ITU coordination arc
	}
}

// DetectConflicts finds slots that violate separation requirements
func (d *SlotConflictDetector) DetectConflicts(slots []OrbitalSlot) []SlotConflict {
	var conflicts []SlotConflict

	for i := 0; i < len(slots); i++ {
		for j := i + 1; j < len(slots); j++ {
			s1, s2 := slots[i], slots[j]

			// Check frequency overlap
			if !d.hasFrequencyOverlap(s1.FrequencyBands, s2.FrequencyBands) {
				continue
			}

			// Check spatial separation
			separation := math.Abs(s1.Longitude - s2.Longitude)
			if separation < d.MinimumSeparationDeg {
				conflict := SlotConflict{
					Slot1:         s1,
					Slot2:         s2,
					SeparationDeg: separation,
					RequiredDeg:   d.MinimumSeparationDeg,
					Type:          "Spatial",
				}
				conflicts = append(conflicts, conflict)
			}
		}
	}

	return conflicts
}

// SlotConflict describes a slot coordination issue
type SlotConflict struct {
	Slot1         OrbitalSlot
	Slot2         OrbitalSlot
	SeparationDeg float64
	RequiredDeg   float64
	Type          string // "Spatial" or "Frequency"
}

// hasFrequencyOverlap checks if two slots have overlapping frequencies
func (d *SlotConflictDetector) hasFrequencyOverlap(b1, b2 []FrequencyBand) bool {
	for _, fb1 := range b1 {
		for _, fb2 := range b2 {
			if fb1.UpperMHz > fb2.LowerMHz && fb1.LowerMHz < fb2.UpperMHz {
				return true
			}
		}
	}
	return false
}

// GetAvailableSlots suggests available orbital slots given constraints
func (db *ITUFilingDatabase) GetAvailableSlots(
	longitudeRange [2]float64,
	frequencyBand FrequencyBand,
	minSeparationDeg float64,
) []OrbitalSlot {
	db.mu.RLock()
	defer db.mu.RUnlock()

	var available []OrbitalSlot

	// Scan the range at coarse resolution
	for lon := longitudeRange[0]; lon <= longitudeRange[1]; lon += 0.5 {
		// Check if slot is occupied
		occupied := false
		nearby := db.GetFilingsBySlot(lon, minSeparationDeg)
		for _, filing := range nearby {
			for _, slot := range filing.OrbitalSlots {
				for _, fb := range slot.FrequencyBands {
					if frequencyBand.UpperMHz > fb.LowerMHz && frequencyBand.LowerMHz < fb.UpperMHz {
						occupied = true
						break
					}
				}
				if occupied {
					break
				}
			}
			if occupied {
				break
			}
		}

		if !occupied {
			available = append(available, OrbitalSlot{
				Longitude:      lon,
				Latitude:       0,
				AltitudeKm:     35786,
				FrequencyBands: []FrequencyBand{frequencyBand},
			})
		}
	}

	return available
}

// VerifyFilingCompliance checks if a filing meets ITU requirements
func VerifyFilingCompliance(filing *ITUFiling) []ComplianceIssue {
	var issues []ComplianceIssue

	// Check required fields
	if filing.ITUReference == "" {
		issues = append(issues, ComplianceIssue{
			Severity: "Error",
			Code:     "MISSING_ITU_REF",
			Message:  "ITU reference number required",
		})
	}

	if filing.Administration == "" {
		issues = append(issues, ComplianceIssue{
			Severity: "Error",
			Code:     "MISSING_ADMIN",
			Message:  "Administration country code required",
		})
	}

	if len(filing.OrbitalSlots) == 0 {
		issues = append(issues, ComplianceIssue{
			Severity: "Error",
			Code:     "NO_SLOTS",
			Message:  "At least one orbital slot required",
		})
	}

	if len(filing.FrequencyBands) == 0 {
		issues = append(issues, ComplianceIssue{
			Severity: "Error",
			Code:     "NO_FREQS",
			Message:  "At least one frequency band required",
		})
	}

	// Check slot validity
	for i, slot := range filing.OrbitalSlots {
		if slot.Longitude < -180 || slot.Longitude > 180 {
			issues = append(issues, ComplianceIssue{
				Severity: "Error",
				Code:     "INVALID_LONGITUDE",
				Message:  fmt.Sprintf("Slot %d: longitude out of range", i),
				Field:    fmt.Sprintf("OrbitalSlots[%d].Longitude", i),
			})
		}

		if math.Abs(slot.AltitudeKm-35786) > 500 && filing.NetworkType == "GSO" {
			issues = append(issues, ComplianceIssue{
				Severity: "Warning",
				Code:     "NON_GEO_ALTITUDE",
				Message:  fmt.Sprintf("Slot %d: altitude not GEO standard", i),
			})
		}
	}

	return issues
}

// ComplianceIssue describes a filing compliance problem
type ComplianceIssue struct {
	Severity string // Error, Warning, Info
	Code     string
	Message  string
	Field    string
}
