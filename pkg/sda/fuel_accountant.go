package sda

import (
	"math"
	"time"

	"ssis/pkg/catalog"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Fuel Depletion Modeling (FDM)
 *
 * Implements dynamic attrition tracking by treating every observed maneuver
 * as an expenditure from a finite propulsive bank. This enables the system to
 * identify when an adversary is "empty" and reclassify interceptors as debris.
 */

// FuelAccountant manages fuel accounting for space objects
// Sits between SINDy maneuver detector and Catalog Store
type FuelAccountant struct {
	MetadataStore           *MetadataStore
	Catalog                 *catalog.Store
	EfficiencyFactor        float32              // ACS overhead penalty (default 1.05 = 5%)
	StationKeepingThreshold float32              // Minimum fuel for station-keeping (m/s)
	SpectralTable           *SpectralToFuelTable // Spectral-to-fuel lookup table
}

// NewFuelAccountant creates a new fuel accountant
func NewFuelAccountant(metadataStore *MetadataStore, catalog *catalog.Store) *FuelAccountant {
	return &FuelAccountant{
		MetadataStore:           metadataStore,
		Catalog:                 catalog,
		EfficiencyFactor:        1.05, // 5% ACS overhead
		StationKeepingThreshold: 10.0, // 10 m/s minimum for station-keeping
		SpectralTable:           NewSpectralToFuelTable(),
	}
}

// ProcessManeuver processes a detected maneuver and updates fuel budget
// Algorithm:
// 1. Calculate burn magnitude
// 2. Load metadata
// 3. Apply efficiency penalty
// 4. Update remaining DV
// 5. Apply zero-fuel lockdown if depleted
func (a *FuelAccountant) ProcessManeuver(targetID uint32, dvVector [3]float64) {
	// 1. Calculate the magnitude of the burn
	dvMag := math.Sqrt(dvVector[0]*dvVector[0] + dvVector[1]*dvVector[1] + dvVector[2]*dvVector[2])

	// 2. Load metadata
	meta := a.MetadataStore.Get(targetID)
	if meta.EngineType == EnginePassive {
		return // No fuel to track for passive objects
	}

	// 3. Update budget with efficiency penalty
	dvSpent := float32(dvMag) * a.EfficiencyFactor
	meta.EstimatedDV -= dvSpent
	meta.LastManifold = time.Now().UnixNano()

	// 4. Update effective mass (accounting for Isp decay)
	a.updateEffectiveMass(&meta, dvSpent)

	// 5. Zero-fuel lockdown
	if meta.EstimatedDV <= 0 {
		meta.EstimatedDV = 0
		// Downgrade threat status
		meta.PlatformClass = "DEBRIS_PASSIVE"
	}

	// 6. Check for End-of-Life (EOL) status
	a.checkEOLStatus(&meta)

	// 7. Persist to metadata store
	a.MetadataStore.Set(meta)
}

// updateEffectiveMass updates the effective remaining mass accounting for Isp decay
// As mass decreases, the same fuel provides slightly higher Δv (Tsiolkovsky equation)
func (a *FuelAccountant) updateEffectiveMass(meta *AssetMetadata, dvSpent float32) {
	// Calculate fuel mass consumed
	// From Tsiolkovsky: Δv = Isp * g0 * ln(m_initial / m_final)
	// Rearranged: m_final = m_initial * exp(-Δv / (Isp * g0))

	g0 := float32(9.80665) // m/s²

	// Calculate new final mass after this burn
	newMassFinal := meta.MassFinal * float32(math.Exp(-float64(dvSpent)/(float64(meta.Isp)*float64(g0))))

	if newMassFinal < meta.MassInitial*0.1 {
		// Don't allow mass to drop below 10% of initial (dry mass + structure)
		newMassFinal = meta.MassInitial * 0.1
	}

	meta.MassFinal = newMassFinal
}

// checkEOLStatus checks if the object is End-of-Life (EOL)
// EOL is triggered when fuel is below station-keeping threshold
func (a *FuelAccountant) checkEOLStatus(meta *AssetMetadata) {
	if meta.EstimatedDV < a.StationKeepingThreshold {
		// Mark as EOL - cannot perform station-keeping
		meta.PlatformClass = "EOL_DEBRIS"
	}
}

// GetFuelStatus returns detailed fuel status including EOL warning
func (a *FuelAccountant) GetFuelStatus(targetID uint32) FuelStatus {
	meta := a.MetadataStore.Get(targetID)

	originalDV := CalculateTsiolkovskyDV(meta.Isp, meta.MassInitial, meta.MassFinal)
	percentage := float32(100.0)
	if originalDV > 0 {
		percentage = meta.EstimatedDV / originalDV * 100.0
	}

	return FuelStatus{
		RemainingDV:  meta.EstimatedDV,
		OriginalDV:   originalDV,
		Percentage:   percentage,
		IsEOL:        meta.PlatformClass == "EOL_DEBRIS" || meta.PlatformClass == "DEBRIS_PASSIVE",
		EngineType:   meta.EngineType,
		LastManeuver: meta.LastManifold,
	}
}

// FuelStatus represents the fuel status of a space object
type FuelStatus struct {
	RemainingDV  float32 // Remaining fuel budget (m/s)
	OriginalDV   float32 // Original fuel budget (m/s)
	Percentage   float32 // Percentage remaining
	IsEOL        bool    // End-of-Life status
	EngineType   EngineType
	LastManeuver int64 // Unix timestamp of last maneuver
}

// PredictReplacementWindow predicts when an adversary will need replacement
// Based on current fuel burn rate and remaining budget
func (a *FuelAccountant) PredictReplacementWindow(targetID uint32, observationWindow time.Duration) (time.Time, bool) {
	meta := a.MetadataStore.Get(targetID)

	if meta.EngineType == EnginePassive || meta.EstimatedDV <= 0 {
		return time.Time{}, false // No prediction for passive or depleted objects
	}

	// Calculate burn rate over observation window
	// In production: query maneuver history from database
	// For now: use average burn rate from metadata
	burnRate := meta.BurnRate // m/s²

	if burnRate <= 0 {
		return time.Time{}, false
	}

	// Calculate time to depletion
	timeToDepletion := float64(meta.EstimatedDV) / float64(burnRate)

	// Predict replacement window (add 30% buffer)
	replacementTime := time.Now().Add(time.Duration(timeToDepletion*1.3) * time.Second)

	return replacementTime, true
}

// GetFleetFuelStatus returns aggregate fuel status for a fleet (e.g., all Inspectors)
func (a *FuelAccountant) GetFleetFuelStatus(platformClass string) FleetFuelStatus {
	records := a.Catalog.Snapshot()

	var totalFuel float32
	var depletedCount int
	var totalCount int

	for _, obj := range records {
		meta := a.MetadataStore.Get(uint32(obj.ID))

		if meta.PlatformClass == platformClass {
			totalFuel += meta.EstimatedDV
			totalCount++

			if meta.EstimatedDV < a.StationKeepingThreshold {
				depletedCount++
			}
		}
	}

	averageFuel := float32(0.0)
	if totalCount > 0 {
		averageFuel = totalFuel / float32(totalCount)
	}

	return FleetFuelStatus{
		PlatformClass: platformClass,
		TotalFuel:     totalFuel,
		AverageFuel:   averageFuel,
		DepletedCount: depletedCount,
		TotalCount:    totalCount,
		DepletionRate: float32(depletedCount) / float32(totalCount),
	}
}

// FleetFuelStatus represents aggregate fuel status for a fleet
type FleetFuelStatus struct {
	PlatformClass string
	TotalFuel     float32 // Total remaining fuel (m/s)
	AverageFuel   float32 // Average fuel per asset (m/s)
	DepletedCount int     // Number of depleted assets
	TotalCount    int     // Total number of assets
	DepletionRate float32 // Fraction of depleted assets
}

// ProcessSINDyManeuver processes a SINDy-detected maneuver
// This is the main entry point for the attrition pipeline
func (a *FuelAccountant) ProcessSINDyManeuver(targetID uint32, thrustMagnitude float64, thrustDirection [3]float64) {
	// Calculate DV vector from thrust magnitude and direction
	dvVector := [3]float64{
		thrustDirection[0] * thrustMagnitude,
		thrustDirection[1] * thrustMagnitude,
		thrustDirection[2] * thrustMagnitude,
	}

	a.ProcessManeuver(targetID, dvVector)
}

// ProcessSpectralObservation processes a spectral observation and updates metadata
// This is the "Propulsion Locking" mechanism - automatic engine type detection
func (a *FuelAccountant) ProcessSpectralObservation(targetID uint32, detectedLines []SpectralLine) bool {
	if a.SpectralTable == nil {
		return false
	}

	// Update metadata from spectral data
	updated := a.SpectralTable.UpdateMetadataFromSpectra(targetID, detectedLines, a.MetadataStore)
	return updated
}

// GenerateFuelAwareThreatMap generates the fuel-aware threat map for sovereign stack
func (a *FuelAccountant) GenerateFuelAwareThreatMap(primaryID int, rriSolver *RRISolver) []FuelAwareThreatEntry {
	threatList := rriSolver.GenerateThreatList(primaryID, a.Catalog)

	var fuelAwareEntries []FuelAwareThreatEntry

	for _, threat := range threatList {
		fuelStatus := a.GetFuelStatus(uint32(threat.ObjectID))

		entry := FuelAwareThreatEntry{
			ThreatEntry:       threat,
			FuelStatus:        fuelStatus,
			ManeuversObserved: a.CountManeuvers(uint32(threat.ObjectID)),
		}

		fuelAwareEntries = append(fuelAwareEntries, entry)
	}

	return fuelAwareEntries
}

// CountManeuvers counts maneuvers observed for a target (placeholder)
// In production: query maneuver history from database
func (a *FuelAccountant) CountManeuvers(targetID uint32) int {
	meta := a.MetadataStore.Get(targetID)

	// Simple heuristic: count based on fuel expended vs original
	originalDV := CalculateTsiolkovskyDV(meta.Isp, meta.MassInitial, meta.MassInitial-(meta.MassInitial-meta.MassFinal))
	if originalDV <= 0 {
		return 0
	}

	// Estimate maneuvers based on fuel expenditure
	// Each maneuver typically costs 10-50 m/s
	estimatedManeuvers := int((originalDV - meta.EstimatedDV) / 25.0)

	return estimatedManeuvers
}

// FuelAwareThreatEntry represents a threat entry with fuel awareness
type FuelAwareThreatEntry struct {
	ThreatEntry
	FuelStatus
	ManeuversObserved int
}
