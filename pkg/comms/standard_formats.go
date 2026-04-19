// Package comms provides standard tactical messaging formats
// Implements SALUTE, SITREP, and 9-Line Medevac formats
package comms

import (
	"fmt"
	"time"
)

// SALUTEReport represents a Spot Report (SALUTE)
// Size, Activity, Location, Unit, Time, Equipment
type SALUTEReport struct {
	ReportID     string
	DateTime     time.Time
	ReportingUnit string
	Size         string // S1 (Squad), S2 (Platoon), S3 (Company), etc.
	Activity     string // What the enemy is doing
	Location     string // Grid coordinates or description
	Unit         string // Enemy unit identification
	Time         string // When observed
	Equipment    string // Weapons, vehicles, etc.
	Priority     int    // 1=Low, 2=Medium, 3=High, 4=Critical
}

// NewSALUTE creates a new SALUTE report
func NewSALUTE(reportingUnit string) *SALUTEReport {
	return &SALUTEReport{
		ReportID:      fmt.Sprintf("SALUTE-%d", time.Now().Unix()),
		DateTime:      time.Now(),
		ReportingUnit: reportingUnit,
		Priority:      2,
	}
}

// Format formats the SALUTE report for transmission
func (sr *SALUTEReport) Format() string {
	return fmt.Sprintf(
		"SALUTE REPORT %s\n"+
			"SIZE: %s\n"+
			"ACTIVITY: %s\n"+
			"LOCATION: %s\n"+
			"UNIT: %s\n"+
			"TIME: %s\n"+
			"EQUIPMENT: %s\n"+
			"REPORTING UNIT: %s\n"+
			"DATETIME: %s",
		sr.ReportID,
		sr.Size,
		sr.Activity,
		sr.Location,
		sr.Unit,
		sr.Time,
		sr.Equipment,
		sr.ReportingUnit,
		sr.DateTime.Format(time.RFC3339),
	)
}

// FormatBrief formats a brief SALUTE report
func (sr *SALUTEReport) FormatBrief() string {
	return fmt.Sprintf(
		"SALUTE: %s/%s/%s/%s/%s/%s/%s",
		sr.Size,
		sr.Activity,
		sr.Location,
		sr.Unit,
		sr.Time,
		sr.Equipment,
		sr.ReportingUnit,
	)
}

// SITREP represents a Situation Report
type SITREP struct {
	ReportID     string
	DateTime     time.Time
	ReportingUnit string
	Line1        string // Enemy situation
	Line2        string // Friendly situation
	Line3        string // Logistics status
	Line4        string // Personnel status
	Line5        string // Weather/terrain
	Line6        string // Recommendations
	Line7        string // Next report time
	Priority     int
}

// NewSITREP creates a new SITREP
func NewSITREP(reportingUnit string) *SITREP {
	return &SITREP{
		ReportID:      fmt.Sprintf("SITREP-%d", time.Now().Unix()),
		DateTime:      time.Now(),
		ReportingUnit: reportingUnit,
		Priority:      2,
	}
}

// Format formats the SITREP for transmission
func (sr *SITREP) Format() string {
	return fmt.Sprintf(
		"SITREP %s\n"+
			"LINE 1 (ENEMY): %s\n"+
			"LINE 2 (FRIENDLY): %s\n"+
			"LINE 3 (LOGISTICS): %s\n"+
			"LINE 4 (PERSONNEL): %s\n"+
			"LINE 5 (WEATHER/TERRAIN): %s\n"+
			"LINE 6 (RECOMMENDATIONS): %s\n"+
			"LINE 7 (NEXT REPORT): %s\n"+
			"REPORTING UNIT: %s\n"+
			"DATETIME: %s",
		sr.ReportID,
		sr.Line1,
		sr.Line2,
		sr.Line3,
		sr.Line4,
		sr.Line5,
		sr.Line6,
		sr.Line7,
		sr.ReportingUnit,
		sr.DateTime.Format(time.RFC3339),
	)
}

// FormatBrief formats a brief SITREP
func (sr *SITREP) FormatBrief() string {
	return fmt.Sprintf(
		"SITREP: L1=%s L2=%s L3=%s L4=%s",
		sr.Line1,
		sr.Line2,
		sr.Line3,
		sr.Line4,
	)
}

// MedevacLine represents a 9-Line Medevac request
type MedevacLine struct {
	Line1 string // Location
	Line2 string // Radio frequency, callsign
	Line3 string // Patients precedence
	Line4 string // Special equipment
	Line5 string // Patients type
	Line6 string // Security
	Line7 string // Marking method
	Line8 string // Patient nationality
	Line9 string // CBRN contamination
}

// NewMedevacLine creates a new 9-Line Medevac request
func NewMedevacLine() *MedevacLine {
	return &MedevacLine{
		Line1: "", // Location
		Line2: "", // Radio frequency, callsign
		Line3: "", // Patients precedence
		Line4: "", // Special equipment
		Line5: "", // Patients type
		Line6: "", // Security
		Line7: "", // Marking method
		Line8: "", // Patient nationality
		Line9: "", // CBRN contamination
	}
}

// Format formats the 9-Line Medevac for transmission
func (ml *MedevacLine) Format() string {
	return fmt.Sprintf(
		"MEDEVAC REQUEST\n"+
			"LINE 1 (LOCATION): %s\n"+
			"LINE 2 (RADIO/FREQ/CALLSIGN): %s\n"+
			"LINE 3 (PATIENTS PRECEDENCE): %s\n"+
			"LINE 4 (SPECIAL EQUIPMENT): %s\n"+
			"LINE 5 (PATIENTS TYPE): %s\n"+
			"LINE 6 (SECURITY): %s\n"+
			"LINE 7 (MARKING METHOD): %s\n"+
			"LINE 8 (PATIENT NATIONALITY): %s\n"+
			"LINE 9 (CBRN CONTAMINATION): %s",
		ml.Line1,
		ml.Line2,
		ml.Line3,
		ml.Line4,
		ml.Line5,
		ml.Line6,
		ml.Line7,
		ml.Line8,
		ml.Line9,
	)
}

// FormatBrief formats a brief 9-Line Medevac
func (ml *MedevacLine) FormatBrief() string {
	return fmt.Sprintf(
		"MEDEVAC: L1=%s L2=%s L3=%s",
		ml.Line1,
		ml.Line2,
		ml.Line3,
	)
}

// Validate validates the 9-Line Medevac
func (ml *MedevacLine) Validate() error {
	if ml.Line1 == "" {
		return fmt.Errorf("Line 1 (Location) is required")
	}
	if ml.Line2 == "" {
		return fmt.Errorf("Line 2 (Radio/Freq/Callsign) is required")
	}
	if ml.Line3 == "" {
		return fmt.Errorf("Line 3 (Patients Precedence) is required")
	}
	return nil
}

// Precedence levels for Medevac
const (
	PrecedenceUrgent    = "URGENT"
	PrecedencePriority  = "PRIORITY"
	PrecedenceRoutine   = "ROUTINE"
	PrecedenceTactical = "TACTICAL IMMEDIATE"
)

// Special equipment codes for Medevac
const (
	EquipNone       = "NONE"
	EquipVentilator = "VENTILATOR"
	EquipBlood      = "BLOOD"
	EquipLitter     = "LITTER"
	EquipSuction    = "SUCTION"
)

// Security codes for Medevac
const (
	SecurityNoEnemy      = "NO ENEMY"
	SecurityEnemyPresent = "ENEMY IN AREA"
	SecurityEnemyApproach = "ENEMY APPROACHING"
	SecurityEnemyFiring   = "ENEMY FIRING IN AREA"
)

// Patient types for Medevac
const (
	PatientLitter = "LITTER"
	PatientAmbulatory = "AMBULATORY"
)

// SpotReport represents a generic spot report
type SpotReport struct {
	ReportID     string
	DateTime     time.Time
	Location     string
	Observation  string
	ReportingUnit string
	Confidence   int // 1-100
	Priority     int // 1-4
}

// NewSpotReport creates a new spot report
func NewSpotReport(location, observation, reportingUnit string) *SpotReport {
	return &SpotReport{
		ReportID:      fmt.Sprintf("SPOT-%d", time.Now().Unix()),
		DateTime:      time.Now(),
		Location:     location,
		Observation:   observation,
		ReportingUnit: reportingUnit,
		Confidence:    75,
		Priority:      2,
	}
}

// Format formats the spot report
func (sr *SpotReport) Format() string {
	return fmt.Sprintf(
		"SPOT REPORT %s\n"+
			"LOCATION: %s\n"+
			"OBSERVATION: %s\n"+
			"CONFIDENCE: %d%%\n"+
			"PRIORITY: %d\n"+
			"REPORTING UNIT: %s\n"+
			"DATETIME: %s",
		sr.ReportID,
		sr.Location,
		sr.Observation,
		sr.Confidence,
		sr.Priority,
		sr.ReportingUnit,
		sr.DateTime.Format(time.RFC3339),
	)
}

// IntelligenceReport represents an intelligence report
type IntelligenceReport struct {
	ReportID     string
	DateTime     time.Time
	Source       string
	Reliability  string // A1, B2, C3, etc.
	Information  string
	Assessment   string
	ReportingUnit string
}

// NewIntelligenceReport creates a new intelligence report
func NewIntelligenceReport(source, reportingUnit string) *IntelligenceReport {
	return &IntelligenceReport{
		ReportID:      fmt.Sprintf("INT-%d", time.Now().Unix()),
		DateTime:      time.Now(),
		Source:        source,
		Reliability:   "B2",
		ReportingUnit: reportingUnit,
	}
}

// Format formats the intelligence report
func (ir *IntelligenceReport) Format() string {
	return fmt.Sprintf(
		"INTELLIGENCE REPORT %s\n"+
			"SOURCE: %s\n"+
			"RELIABILITY: %s\n"+
			"INFORMATION: %s\n"+
			"ASSESSMENT: %s\n"+
			"REPORTING UNIT: %s\n"+
			"DATETIME: %s",
		ir.ReportID,
		ir.Source,
		ir.Reliability,
		ir.Information,
		ir.Assessment,
		ir.ReportingUnit,
		ir.DateTime.Format(time.RFC3339),
	)
}

// LogisticsReport represents a logistics status report
type LogisticsReport struct {
	ReportID       string
	DateTime       time.Time
	ReportingUnit  string
	FuelStatus     string // Percentage or description
	AmmoStatus     string // Percentage or description
	FoodStatus     string // Days of supply
	WaterStatus    string // Days of supply
	EquipmentStatus string
	MedicalStatus  string
	Priority       int
}

// NewLogisticsReport creates a new logistics report
func NewLogisticsReport(reportingUnit string) *LogisticsReport {
	return &LogisticsReport{
		ReportID:      fmt.Sprintf("LOG-%d", time.Now().Unix()),
		DateTime:      time.Now(),
		ReportingUnit: reportingUnit,
		Priority:      2,
	}
}

// Format formats the logistics report
func (lr *LogisticsReport) Format() string {
	return fmt.Sprintf(
		"LOGISTICS REPORT %s\n"+
			"FUEL: %s\n"+
			"AMMUNITION: %s\n"+
			"FOOD: %s\n"+
			"WATER: %s\n"+
			"EQUIPMENT: %s\n"+
			"MEDICAL: %s\n"+
			"REPORTING UNIT: %s\n"+
			"DATETIME: %s",
		lr.ReportID,
		lr.FuelStatus,
		lr.AmmoStatus,
		lr.FoodStatus,
		lr.WaterStatus,
		lr.EquipmentStatus,
		lr.MedicalStatus,
		lr.ReportingUnit,
		lr.DateTime.Format(time.RFC3339),
	)
}

// ReportManager manages tactical reports
type ReportManager struct {
	saluteReports  []*SALUTEReport
	sitrepReports  []*SITREP
	medevacLines   []*MedevacLine
	spotReports    []*SpotReport
	intelReports   []*IntelligenceReport
	logisticsReports []*LogisticsReport
}

// NewReportManager creates a new report manager
func NewReportManager() *ReportManager {
	return &ReportManager{
		saluteReports:   make([]*SALUTEReport, 0),
		sitrepReports:   make([]*SITREP, 0),
		medevacLines:    make([]*MedevacLine, 0),
		spotReports:     make([]*SpotReport, 0),
		intelReports:    make([]*IntelligenceReport, 0),
		logisticsReports: make([]*LogisticsReport, 0),
	}
}

// AddSALUTE adds a SALUTE report
func (rm *ReportManager) AddSALUTE(report *SALUTEReport) {
	rm.saluteReports = append(rm.saluteReports, report)
}

// AddSITREP adds a SITREP
func (rm *ReportManager) AddSITREP(report *SITREP) {
	rm.sitrepReports = append(rm.sitrepReports, report)
}

// AddMedevac adds a 9-Line Medevac
func (rm *ReportManager) AddMedevac(report *MedevacLine) {
	rm.medevacLines = append(rm.medevacLines, report)
}

// AddSpot adds a spot report
func (rm *ReportManager) AddSpot(report *SpotReport) {
	rm.spotReports = append(rm.spotReports, report)
}

// AddIntel adds an intelligence report
func (rm *ReportManager) AddIntel(report *IntelligenceReport) {
	rm.intelReports = append(rm.intelReports, report)
}

// AddLogistics adds a logistics report
func (rm *ReportManager) AddLogistics(report *LogisticsReport) {
	rm.logisticsReports = append(rm.logisticsReports, report)
}

// GetSALUTEByID retrieves a SALUTE report by ID
func (rm *ReportManager) GetSALUTEByID(id string) *SALUTEReport {
	for _, report := range rm.saluteReports {
		if report.ReportID == id {
			return report
		}
	}
	return nil
}

// GetSITREPByID retrieves a SITREP by ID
func (rm *ReportManager) GetSITREPByID(id string) *SITREP {
	for _, report := range rm.sitrepReports {
		if report.ReportID == id {
			return report
		}
	}
	return nil
}

// GetRecentReports returns reports within a time window
func (rm *ReportManager) GetRecentReports(duration time.Duration) []*SALUTEReport {
	cutoff := time.Now().Add(-duration)
	filtered := make([]*SALUTEReport, 0)
	
	for _, report := range rm.saluteReports {
		if report.DateTime.After(cutoff) {
			filtered = append(filtered, report)
		}
	}
	
	return filtered
}

// GetHighPriorityReports returns reports with high priority
func (rm *ReportManager) GetHighPriorityReports(minPriority int) []*SALUTEReport {
	filtered := make([]*SALUTEReport, 0)
	
	for _, report := range rm.saluteReports {
		if report.Priority >= minPriority {
			filtered = append(filtered, report)
		}
	}
	
	return filtered
}

// GetReportCount returns the total number of reports
func (rm *ReportManager) GetReportCount() int {
	return len(rm.saluteReports) +
		len(rm.sitrepReports) +
		len(rm.medevacLines) +
		len(rm.spotReports) +
		len(rm.intelReports) +
		len(rm.logisticsReports)
}

// ClearReports clears all reports
func (rm *ReportManager) ClearReports() {
	rm.saluteReports = make([]*SALUTEReport, 0)
	rm.sitrepReports = make([]*SITREP, 0)
	rm.medevacLines = make([]*MedevacLine, 0)
	rm.spotReports = make([]*SpotReport, 0)
	rm.intelReports = make([]*IntelligenceReport, 0)
	rm.logisticsReports = make([]*LogisticsReport, 0)
}
