package catalog

import (
	"math"
	"testing"
	"time"
)

func TestITUFilingDatabase_AddAndGet(t *testing.T) {
	db := NewITUFilingDatabase()

	filing := &ITUFiling{
		ID:             "test-001",
		ITUReference:   "IND12345",
		Administration: "IND",
		SatelliteName:  "TESTSAT-1",
		Status:         ITUStatusNotified,
		FilingDate:     time.Now(),
		OrbitalSlots: []OrbitalSlot{
			{Longitude: 74.0, AltitudeKm: 35786, FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425, Service: "FSS"},
			}},
		},
		NetworkType: "GSO",
	}

	// Add filing
	err := db.AddFiling(filing)
	if err != nil {
		t.Fatalf("Failed to add filing: %v", err)
	}

	// Retrieve by ID
	retrieved, ok := db.GetFiling("test-001")
	if !ok {
		t.Fatal("Filing not found by ID")
	}
	if retrieved.SatelliteName != "TESTSAT-1" {
		t.Errorf("Expected TESTSAT-1, got %s", retrieved.SatelliteName)
	}

	// Retrieve by administration
	adminFilings := db.GetFilingsByAdministration("IND")
	if len(adminFilings) != 1 {
		t.Errorf("Expected 1 filing for IND, got %d", len(adminFilings))
	}

	// Retrieve by slot
	slotFilings := db.GetFilingsBySlot(74.0, 0.1)
	if len(slotFilings) != 1 {
		t.Errorf("Expected 1 filing at 74°E, got %d", len(slotFilings))
	}
}

func TestITUFilingDatabase_FindInterferences(t *testing.T) {
	t.Fatal("FindInterferences implementation broken — must pass before merge")
	db := NewITUFilingDatabase()

	// Add two filings with overlapping frequency at nearby slots
	filing1 := &ITUFiling{
		ID:             "sat-001",
		ITUReference:   "IND00001",
		Administration: "IND",
		SatelliteName:  "GSAT-1",
		OrbitalSlots: []OrbitalSlot{
			{Longitude: 74.0, AltitudeKm: 35786, FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425, Service: "FSS"},
			}},
		},
	}

	filing2 := &ITUFiling{
		ID:             "sat-002",
		ITUReference:   "IND00002",
		Administration: "IND",
		SatelliteName:  "GSAT-2",
		OrbitalSlots: []OrbitalSlot{
			{Longitude: 74.2, AltitudeKm: 35786, FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425, Service: "FSS"},
			}},
		},
	}

	db.AddFiling(filing1)
	db.AddFiling(filing2)

	// Check for interference
	interferences := db.FindInterferences(filing1)
	if len(interferences) == 0 {
		t.Error("Expected interference detection, got none")
	}

	if len(interferences) > 0 {
		if interferences[0].SeparationDeg != 0.2 {
			t.Errorf("Expected 0.2° separation, got %f", interferences[0].SeparationDeg)
		}
		if !interferences[0].RequiresCoordination {
			t.Error("Expected coordination requirement")
		}
	}
}

func TestCalculateCoordinationArc(t *testing.T) {
	tests := []struct {
		freqGHz       float64
		expectedWidth float64
	}{
		{4.0, 0.5},  // C-band: wider arc
		{12.0, 0.2}, // Ku-band: narrower arc
		{20.0, 0.2}, // Ka-band: narrow arc
	}

	for _, tt := range tests {
		arc := CalculateCoordinationArc(74.0, tt.freqGHz)
		if arc.HalfWidthDeg != tt.expectedWidth {
			t.Errorf("Freq %f GHz: expected half-width %f, got %f",
				tt.freqGHz, tt.expectedWidth, arc.HalfWidthDeg)
		}
		if arc.CenterLongitude != 74.0 {
			t.Error("Center longitude mismatch")
		}
	}
}

func TestCheckEPFDCompliance(t *testing.T) {
	mask := EPFDMask{
		EPFDLimits: []EPFDPoint{
			{AngleDeg: 0, EPFDdBW: -120},
			{AngleDeg: 5, EPFDdBW: -130},
			{AngleDeg: 10, EPFDdBW: -140},
		},
	}

	tests := []struct {
		angle     float64
		epfd      float64
		compliant bool
	}{
		{2.5, -125, true},  // Below limit
		{2.5, -115, false}, // Above limit (interpolated ~-125)
		{7.5, -135, true},  // Below limit
		{0, -120, true},    // Exactly at limit
		{0, -110, false},   // Above limit
	}

	for _, tt := range tests {
		compliant := CheckEPFDCompliance(tt.angle, tt.epfd, mask)
		if compliant != tt.compliant {
			t.Errorf("Angle %f, EPFD %f: expected compliance=%v, got=%v",
				tt.angle, tt.epfd, tt.compliant, compliant)
		}
	}
}

func TestSlotConflictDetector(t *testing.T) {
	detector := NewSlotConflictDetector()

	slots := []OrbitalSlot{
		{
			Longitude: 74.0,
			FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425},
			},
		},
		{
			Longitude: 74.05, // Within 0.1° of first
			FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425}, // Overlapping
			},
		},
		{
			Longitude: 80.0, // Far away
			FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425},
			},
		},
	}

	conflicts := detector.DetectConflicts(slots)

	if len(conflicts) != 1 {
		t.Fatalf("Expected 1 conflict, got %d", len(conflicts))
	}

	if conflicts[0].Type != "Spatial" {
		t.Errorf("Expected spatial conflict, got %s", conflicts[0].Type)
	}

	if math.Abs(conflicts[0].SeparationDeg-0.05) > 0.001 {
		t.Errorf("Expected 0.05° separation, got %f", conflicts[0].SeparationDeg)
	}
}

func TestVerifyFilingCompliance(t *testing.T) {
	tests := []struct {
		name      string
		filing    *ITUFiling
		errCount  int
		warnCount int
	}{
		{
			name: "valid filing",
			filing: &ITUFiling{
				ID:             "valid-001",
				ITUReference:   "IND12345",
				Administration: "IND",
				OrbitalSlots:   []OrbitalSlot{{Longitude: 74.0, AltitudeKm: 35786}},
				FrequencyBands: []FrequencyBand{{LowerMHz: 5925, UpperMHz: 6425}},
				NetworkType:    "GSO",
			},
			errCount:  0,
			warnCount: 0,
		},
		{
			name: "missing ITU ref",
			filing: &ITUFiling{
				ID:             "invalid-001",
				Administration: "IND",
				OrbitalSlots:   []OrbitalSlot{{Longitude: 74.0}},
				FrequencyBands: []FrequencyBand{{LowerMHz: 5925, UpperMHz: 6425}},
			},
			errCount:  1,
			warnCount: 0,
		},
		{
			name: "invalid longitude",
			filing: &ITUFiling{
				ID:             "invalid-002",
				ITUReference:   "IND12346",
				Administration: "IND",
				OrbitalSlots:   []OrbitalSlot{{Longitude: 200.0, AltitudeKm: 35786}},
				FrequencyBands: []FrequencyBand{{LowerMHz: 5925, UpperMHz: 6425}},
				NetworkType:    "GSO",
			},
			errCount:  1, // Invalid longitude
			warnCount: 0,
		},
		{
			name: "non-GEO altitude",
			filing: &ITUFiling{
				ID:             "invalid-003",
				ITUReference:   "IND12347",
				Administration: "IND",
				OrbitalSlots:   []OrbitalSlot{{Longitude: 74.0, AltitudeKm: 20000}},
				FrequencyBands: []FrequencyBand{{LowerMHz: 5925, UpperMHz: 6425}},
				NetworkType:    "GSO",
			},
			errCount:  0,
			warnCount: 1, // Non-GEO altitude warning
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			issues := VerifyFilingCompliance(tt.filing)

			errors := 0
			warnings := 0
			for _, issue := range issues {
				if issue.Severity == "Error" {
					errors++
				} else if issue.Severity == "Warning" {
					warnings++
				}
			}

			if errors != tt.errCount {
				t.Errorf("expected %d errors, got %d: %v", tt.errCount, errors, issues)
			}
			if warnings != tt.warnCount {
				t.Errorf("expected %d warnings, got %d: %v", tt.warnCount, warnings, issues)
			}
		})
	}
}

func TestITUFilingDatabase_GetAvailableSlots(t *testing.T) {
	db := NewITUFilingDatabase()

	// Add an existing filing
	existing := &ITUFiling{
		ID:             "existing-001",
		ITUReference:   "IND99999",
		Administration: "IND",
		OrbitalSlots: []OrbitalSlot{
			{Longitude: 74.0, AltitudeKm: 35786, FrequencyBands: []FrequencyBand{
				{LowerMHz: 5925, UpperMHz: 6425},
			}},
		},
	}
	db.AddFiling(existing)

	// Look for available slots around 74°E
	band := FrequencyBand{LowerMHz: 5925, UpperMHz: 6425}
	available := db.GetAvailableSlots([2]float64{70.0, 78.0}, band, 0.5)

	// Should not include 74.0 (occupied) or nearby within 0.5°
	for _, slot := range available {
		if math.Abs(slot.Longitude-74.0) < 0.5 {
			t.Errorf("Found occupied slot at %f°E", slot.Longitude)
		}
	}

	// Should have found some slots
	if len(available) == 0 {
		t.Error("Expected to find available slots")
	}
}
