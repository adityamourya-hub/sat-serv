// Package comms provides PACE (Primary, Alternate, Contingency, Emergency) planning
// Implements redundancy for resilient communications in air-gapped defense systems
package comms

import (
	"fmt"
	"time"
)

const (
	// PACE level constants
	PACEPrimary     = "PRIMARY"
	PACEAlternate   = "ALTERNATE"
	PACEContingency = "CONTINGENCY"
	PACEEmergency   = "EMERGENCY"
)

// PACELevel represents a PACE communication level
type PACELevel string

// PACEMethod represents a communication method
type PACEMethod struct {
	Name        string
	Type        string // MESH, RADIO, SATELLITE, CELLULAR, PHYSICAL
	Description string
	IsActive    bool
	LastTest    time.Time
	SuccessRate float64 // 0.0-1.0
	Latency     time.Duration
	Bandwidth   string  // e.g., "9600 bps"
	Range       float64 // meters
	Reliability float64 // 0.0-1.0
}

// PACEPlan represents a PACE communication plan
type PACEPlan struct {
	PlanID       string
	Primary      *PACEMethod
	Alternate    *PACEMethod
	Contingency  *PACEMethod
	Emergency    *PACEMethod
	CurrentLevel PACELevel
	AutoFailover bool
}

// NewPACEPlan creates a new PACE plan
func NewPACEPlan(planID string) *PACEPlan {
	return &PACEPlan{
		PlanID:       planID,
		CurrentLevel: PACEPrimary,
		AutoFailover: true,
	}
}

// SetPrimary sets the primary communication method
func (pp *PACEPlan) SetPrimary(method *PACEMethod) {
	pp.Primary = method
}

// SetAlternate sets the alternate communication method
func (pp *PACEPlan) SetAlternate(method *PACEMethod) {
	pp.Alternate = method
}

// SetContingency sets the contingency communication method
func (pp *PACEPlan) SetContingency(method *PACEMethod) {
	pp.Contingency = method
}

// SetEmergency sets the emergency communication method
func (pp *PACEPlan) SetEmergency(method *PACEMethod) {
	pp.Emergency = method
}

// GetCurrentMethod returns the current active method
func (pp *PACEPlan) GetCurrentMethod() *PACEMethod {
	switch pp.CurrentLevel {
	case PACEPrimary:
		return pp.Primary
	case PACEAlternate:
		return pp.Alternate
	case PACEContingency:
		return pp.Contingency
	case PACEEmergency:
		return pp.Emergency
	default:
		return pp.Primary
	}
}

// TestMethod tests a communication method
func (pp *PACEPlan) TestMethod(level PACELevel) (bool, error) {
	var method *PACEMethod

	switch level {
	case PACEPrimary:
		method = pp.Primary
	case PACEAlternate:
		method = pp.Alternate
	case PACEContingency:
		method = pp.Contingency
	case PACEEmergency:
		method = pp.Emergency
	default:
		return false, fmt.Errorf("invalid PACE level: %s", level)
	}

	if method == nil {
		return false, fmt.Errorf("method not configured for level %s", level)
	}

	// Simulate test (in production, actual connectivity test)
	method.LastTest = time.Now()

	// Update success rate based on test result
	// For now, assume success
	method.SuccessRate = 0.95

	return true, nil
}

// TestAllMethods tests all PACE methods
func (pp *PACEPlan) TestAllMethods() map[PACELevel]bool {
	results := make(map[PACELevel]bool)

	levels := []PACELevel{PACEPrimary, PACEAlternate, PACEContingency, PACEEmergency}
	for _, level := range levels {
		success, _ := pp.TestMethod(level)
		results[level] = success
	}

	return results
}

// Failover performs automatic failover to the next available level
func (pp *PACEPlan) Failover() (PACELevel, *PACEMethod, error) {
	if !pp.AutoFailover {
		return pp.CurrentLevel, nil, fmt.Errorf("auto-failover disabled")
	}

	// Try next level
	var nextLevel PACELevel
	switch pp.CurrentLevel {
	case PACEPrimary:
		nextLevel = PACEAlternate
	case PACEAlternate:
		nextLevel = PACEContingency
	case PACEContingency:
		nextLevel = PACEEmergency
	case PACEEmergency:
		return PACEEmergency, pp.Emergency, fmt.Errorf("no fallback available from emergency")
	}

	// Test next level
	success, err := pp.TestMethod(nextLevel)
	if err != nil {
		return pp.CurrentLevel, nil, err
	}

	if success {
		pp.CurrentLevel = nextLevel
		return nextLevel, pp.GetCurrentMethod(), nil
	}

	// Recursively try next level
	return pp.Failover()
}

// ManualFailover performs manual failover to a specific level
func (pp *PACEPlan) ManualFailover(level PACELevel) error {
	var method *PACEMethod

	switch level {
	case PACEPrimary:
		method = pp.Primary
	case PACEAlternate:
		method = pp.Alternate
	case PACEContingency:
		method = pp.Contingency
	case PACEEmergency:
		method = pp.Emergency
	default:
		return fmt.Errorf("invalid PACE level: %s", level)
	}

	if method == nil {
		return fmt.Errorf("method not configured for level %s", level)
	}

	success, err := pp.TestMethod(level)
	if err != nil {
		return err
	}

	if !success {
		return fmt.Errorf("method test failed for level %s", level)
	}

	pp.CurrentLevel = level
	return nil
}

// GetStatus returns the current PACE status
func (pp *PACEPlan) GetStatus() *PACEStatus {
	return &PACEStatus{
		PlanID:       pp.PlanID,
		CurrentLevel: pp.CurrentLevel,
		Primary:      pp.Primary != nil && pp.Primary.IsActive,
		Alternate:    pp.Alternate != nil && pp.Alternate.IsActive,
		Contingency:  pp.Contingency != nil && pp.Contingency.IsActive,
		Emergency:    pp.Emergency != nil && pp.Emergency.IsActive,
		AutoFailover: pp.AutoFailover,
	}
}

// PACEStatus represents the status of a PACE plan
type PACEStatus struct {
	PlanID       string
	CurrentLevel PACELevel
	Primary      bool
	Alternate    bool
	Contingency  bool
	Emergency    bool
	AutoFailover bool
}

// CreateDefaultADITI4PACE creates a default PACE plan for ADITI 4.0
func CreateDefaultADITI4PACE() *PACEPlan {
	plan := NewPACEPlan("ADITI4-COMMS")

	// Primary: Reticulum Mesh (LXMF)
	plan.SetPrimary(&PACEMethod{
		Name:        "Reticulum Mesh LXMF",
		Type:        "MESH",
		Description: "Encrypted mesh network with LXMF",
		IsActive:    true,
		SuccessRate: 0.95,
		Latency:     100 * time.Millisecond,
		Bandwidth:   "9600 bps",
		Range:       10000.0, // 10 km
		Reliability: 0.9,
	})

	// Alternate: Handheld Radio (Hand-keyed)
	plan.SetAlternate(&PACEMethod{
		Name:        "Handheld VHF Radio",
		Type:        "RADIO",
		Description: "VHF handheld radio with manual encryption",
		IsActive:    true,
		SuccessRate: 0.85,
		Latency:     500 * time.Millisecond,
		Bandwidth:   "1200 bps",
		Range:       5000.0, // 5 km
		Reliability: 0.8,
	})

	// Contingency: Cellphone / Satellite link
	plan.SetContingency(&PACEMethod{
		Name:        "Satellite Phone",
		Type:        "SATELLITE",
		Description: "Iridium satellite phone",
		IsActive:    true,
		SuccessRate: 0.75,
		Latency:     2 * time.Second,
		Bandwidth:   "2400 bps",
		Range:       0.0, // Global
		Reliability: 0.7,
	})

	// Emergency: Whistle / Signal Flare / Courier
	plan.SetEmergency(&PACEMethod{
		Name:        "Physical Signal",
		Type:        "PHYSICAL",
		Description: "Whistle, signal flare, or courier",
		IsActive:    true,
		SuccessRate: 0.5,
		Latency:     30 * time.Minute, // Courier time
		Bandwidth:   "N/A",
		Range:       1000.0, // Visual/audible range
		Reliability: 0.4,
	})

	return plan
}

// CreateSpaceDSAPACE creates a PACE plan for Space Domain Awareness
func CreateSpaceDSAPACE() *PACEPlan {
	plan := NewPACEPlan("SDA-COMMS")

	// Primary: Reticulum Mesh (LXMF)
	plan.SetPrimary(&PACEMethod{
		Name:        "Reticulum Mesh LXMF",
		Type:        "MESH",
		Description: "Encrypted mesh network with LXMF",
		IsActive:    true,
		SuccessRate: 0.95,
		Latency:     100 * time.Millisecond,
		Bandwidth:   "9600 bps",
		Range:       10000.0,
		Reliability: 0.9,
	})

	// Alternate: HF Radio
	plan.SetAlternate(&PACEMethod{
		Name:        "HF Radio with ALE",
		Type:        "RADIO",
		Description: "High Frequency radio with Automatic Link Establishment",
		IsActive:    true,
		SuccessRate: 0.85,
		Latency:     1 * time.Second,
		Bandwidth:   "2400 bps",
		Range:       500000.0, // 500 km
		Reliability: 0.8,
	})

	// Contingency: Satellite Uplink
	plan.SetContingency(&PACEMethod{
		Name:        "Satellite Uplink",
		Type:        "SATELLITE",
		Description: "Commercial satellite uplink",
		IsActive:    true,
		SuccessRate: 0.75,
		Latency:     5 * time.Second,
		Bandwidth:   "9600 bps",
		Range:       0.0, // Global
		Reliability: 0.7,
	})

	// Emergency: Morse Code / Signal Lamp
	plan.SetEmergency(&PACEMethod{
		Name:        "Morse Code / Signal Lamp",
		Type:        "PHYSICAL",
		Description: "Visual Morse code or signal lamp",
		IsActive:    true,
		SuccessRate: 0.4,
		Latency:     10 * time.Minute,
		Bandwidth:   "N/A",
		Range:       50000.0, // 50 km visual
		Reliability: 0.3,
	})

	return plan
}

// PACEManager manages multiple PACE plans
type PACEManager struct {
	plans map[string]*PACEPlan
}

// NewPACEManager creates a new PACE manager
func NewPACEManager() *PACEManager {
	return &PACEManager{
		plans: make(map[string]*PACEPlan),
	}
}

// AddPlan adds a PACE plan
func (pm *PACEManager) AddPlan(plan *PACEPlan) {
	pm.plans[plan.PlanID] = plan
}

// GetPlan retrieves a PACE plan
func (pm *PACEManager) GetPlan(planID string) (*PACEPlan, error) {
	plan, exists := pm.plans[planID]
	if !exists {
		return nil, fmt.Errorf("plan not found: %s", planID)
	}
	return plan, nil
}

// TestAllPlans tests all PACE plans
func (pm *PACEManager) TestAllPlans() map[string]bool {
	results := make(map[string]bool)

	for planID, plan := range pm.plans {
		testResults := plan.TestAllMethods()

		// Plan is healthy if at least primary and alternate work
		results[planID] = testResults[PACEPrimary] && testResults[PACEAlternate]
	}

	return results
}

// GetPlanCount returns the number of PACE plans
func (pm *PACEManager) GetPlanCount() int {
	return len(pm.plans)
}

// GetAllStatus returns status for all plans
func (pm *PACEManager) GetAllStatus() map[string]*PACEStatus {
	status := make(map[string]*PACEStatus)

	for planID, plan := range pm.plans {
		status[planID] = plan.GetStatus()
	}

	return status
}

// CommunicationAudit performs an audit of all communication methods
func (pm *PACEManager) CommunicationAudit() *CommAudit {
	audit := &CommAudit{
		Timestamp: time.Now(),
		Plans:     make([]string, 0),
		Healthy:   true,
	}

	for planID := range pm.plans {
		audit.Plans = append(audit.Plans, planID)
	}

	testResults := pm.TestAllPlans()
	for planID, healthy := range testResults {
		if !healthy {
			audit.Healthy = false
			audit.Issues = append(audit.Issues, fmt.Sprintf("Plan %s failed health check", planID))
		}
	}

	return audit
}

// CommAudit represents a communication audit result
type CommAudit struct {
	Timestamp time.Time
	Plans     []string
	Healthy   bool
	Issues    []string
}

// RedundancyCalculator calculates communication redundancy
type RedundancyCalculator struct {
	primarySuccessRate     float64
	alternateSuccessRate   float64
	contingencySuccessRate float64
	emergencySuccessRate   float64
}

// NewRedundancyCalculator creates a new redundancy calculator
func NewRedundancyCalculator() *RedundancyCalculator {
	return &RedundancyCalculator{
		primarySuccessRate:     0.95,
		alternateSuccessRate:   0.85,
		contingencySuccessRate: 0.75,
		emergencySuccessRate:   0.5,
	}
}

// CalculateOverallReliability calculates overall system reliability
func (rc *RedundancyCalculator) CalculateOverallReliability() float64 {
	// Probability of all methods failing
	allFailProbability := (1 - rc.primarySuccessRate) *
		(1 - rc.alternateSuccessRate) *
		(1 - rc.contingencySuccessRate) *
		(1 - rc.emergencySuccessRate)

	// Overall reliability is 1 minus probability of all failing
	return 1.0 - allFailProbability
}

// CalculateExpectedLatency calculates expected latency with failover
func (rc *RedundancyCalculator) CalculateExpectedLatency(primary, alternate, contingency, emergency time.Duration) time.Duration {
	// Weighted average based on success rates
	totalSuccessRate := rc.primarySuccessRate + rc.alternateSuccessRate + rc.contingencySuccessRate + rc.emergencySuccessRate

	// Convert durations to seconds as float64
	primarySec := float64(primary.Seconds())
	alternateSec := float64(alternate.Seconds())
	contingencySec := float64(contingency.Seconds())
	emergencySec := float64(emergency.Seconds())

	expectedLatencySec := (primarySec*rc.primarySuccessRate +
		alternateSec*rc.alternateSuccessRate +
		contingencySec*rc.contingencySuccessRate +
		emergencySec*rc.emergencySuccessRate) / totalSuccessRate

	return time.Duration(expectedLatencySec * float64(time.Second))
}

// SetSuccessRates sets the success rates for each level
func (rc *RedundancyCalculator) SetSuccessRates(primary, alternate, contingency, emergency float64) {
	rc.primarySuccessRate = primary
	rc.alternateSuccessRate = alternate
	rc.contingencySuccessRate = contingency
	rc.emergencySuccessRate = emergency
}
