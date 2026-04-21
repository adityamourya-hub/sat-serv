package sda

import (
	"math"
	"time"

	"ssis/pkg/catalog"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Reachability Risk Index (RRI) Solver
 *
 * Implements platform-aware risk calculation that transforms
 * geometric "could they?" into strategic "can they?" analysis.
 *
 * RRI = Δv_intercept / Δv_available
 * - RRI < 0.1: High Threat (trivial fuel expenditure)
 * - RRI > 1.0: Zero Threat (physically impossible)
 */

// RRISolver calculates the Reachability Risk Index
type RRISolver struct {
	MetadataStore *MetadataStore
	Watch         *NeighborhoodWatch
	ReserveFactor float32 // Reserve factor for station-keeping fuel (default 0.8)
}

// NewRRISolver creates a new RRI solver
func NewRRISolver(metadataStore *MetadataStore, watch *NeighborhoodWatch) *RRISolver {
	return &RRISolver{
		MetadataStore: metadataStore,
		Watch:         watch,
		ReserveFactor: 0.8, // 20% reserve for station-keeping
	}
}

// CalculateRisk calculates the Reachability Risk Index for an adversary
// RRI = Δv_intercept / (Δv_available * ReserveFactor)
func (s *RRISolver) CalculateRisk(adversaryID uint32, primaryID int, horizon time.Duration) float32 {
	// 1. Metadata Lookup
	meta := s.MetadataStore.Get(adversaryID)
	if meta.EngineType == EnginePassive {
		return float32(math.Inf(1)) // Ignore passive objects (infinite RRI = zero threat)
	}

	// 2. Get available fuel budget
	availableDV := meta.EstimatedDV * s.ReserveFactor
	if availableDV <= 0 {
		return float32(math.Inf(1)) // No fuel available
	}

	// 3. Solve for intercept ΔV over multiple horizons
	// Use the HCW-based reachability calculation
	minDV, _, _ := s.Watch.ComputeReachabilitySet(primaryID, int(adversaryID), float64(horizon.Seconds()))

	// 4. Calculate RRI
	rri := float32(minDV) / availableDV

	return rri
}

// CalculateRiskMultiHorizon calculates RRI across multiple time horizons
// This provides a more comprehensive threat assessment
func (s *RRISolver) CalculateRiskMultiHorizon(adversaryID uint32, primaryID int) map[time.Duration]float32 {
	horizons := []time.Duration{
		1 * time.Hour,
		3 * time.Hour,
		6 * time.Hour,
		12 * time.Hour,
		24 * time.Hour,
	}

	results := make(map[time.Duration]float32)
	for _, horizon := range horizons {
		results[horizon] = s.CalculateRisk(adversaryID, primaryID, horizon)
	}

	return results
}

// GetThreatLevel converts RRI to threat level
func (s *RRISolver) GetThreatLevel(rri float32) ThreatLevel {
	if math.IsInf(float64(rri), 1) || rri > 1.0 {
		return ThreatNominal // Physically impossible
	}

	if rri < 0.1 {
		return ThreatCritical // Trivial fuel expenditure
	}

	if rri < 0.3 {
		return ThreatHigh
	}

	if rri < 0.5 {
		return ThreatMedium
	}

	if rri < 1.0 {
		return ThreatLow
	}

	return ThreatNominal
}

// SolveLambertWithMetadata solves the Relative Lambert problem with metadata awareness
// This is the optimized solver that queries platform metadata before propagation
func (s *RRISolver) SolveLambertWithMetadata(adversaryID uint32, primaryState [6]float64, horizon time.Duration) (deltaV float32, feasible bool) {
	// 1. Metadata Lookup
	meta := s.MetadataStore.Get(adversaryID)
	if meta.EngineType == EnginePassive {
		return 0, false // Passive objects cannot maneuver
	}

	// 2. Get adversary current state (in production, from catalog)
	// For now, use primary state as placeholder
	adversaryState := primaryState

	// 3. Solve Lambert
	// Use the simplified Lambert solver from threat_eval.go
	// In production: Use Izzo's algorithm for full elliptical solution
	dt := float64(horizon.Seconds())

	// Extract positions
	r1 := [3]float64{adversaryState[0], adversaryState[1], adversaryState[2]}
	r2 := [3]float64{primaryState[0], primaryState[1], primaryState[2]}

	// Minimum energy transfer (Hohmann-like approximation)
	deltaVVect := [3]float64{
		(r2[0] - r1[0]) / dt,
		(r2[1] - r1[1]) / dt,
		(r2[2] - r1[2]) / dt,
	}

	// Add gravity compensation
	mu := 398600.4418 // km³/s²
	r := math.Sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2])
	gravityComp := mu / (r * r * r)
	deltaVVect[0] += gravityComp * r1[0] * dt / 2.0
	deltaVVect[1] += gravityComp * r1[1] * dt / 2.0
	deltaVVect[2] += gravityComp * r1[2] * dt / 2.0

	// Calculate magnitude
	deltaVMag := math.Sqrt(deltaVVect[0]*deltaVVect[0] + deltaVVect[1]*deltaVVect[1] + deltaVVect[2]*deltaVVect[2])

	// Convert to m/s
	deltaVMeters := deltaVMag * 1000.0

	// Check feasibility against available fuel
	availableDV := meta.EstimatedDV * s.ReserveFactor
	feasible = float32(deltaVMeters) < availableDV

	return float32(deltaVMeters), feasible
}

// UpdateFuelOnManeuver decrements fuel when SINDy detects a maneuver
// This is called during H100 Distillation to track fuel in real-time
func (s *RRISolver) UpdateFuelOnManeuver(adversaryID uint32, deltaV float32) {
	s.MetadataStore.DecrementFuel(adversaryID, deltaV)
}

// GetFuelStatus returns the current fuel status for an object
func (s *RRISolver) GetFuelStatus(adversaryID uint32) (currentDV float32, percentage float32) {
	meta := s.MetadataStore.Get(adversaryID)

	if meta.EstimatedDV <= 0 {
		return 0, 0
	}

	// Calculate original DV from Tsiolkovsky equation
	originalDV := CalculateTsiolkovskyDV(meta.Isp, meta.MassInitial, meta.MassFinal)

	if originalDV <= 0 {
		return meta.EstimatedDV, 100.0
	}

	percentage = meta.EstimatedDV / originalDV * 100.0
	return meta.EstimatedDV, percentage
}

// GenerateThreatList generates a prioritized threat list based on RRI
// This is the tactical outcome for the IAF War-Room
func (s *RRISolver) GenerateThreatList(primaryID int, catalog *catalog.Store) []ThreatEntry {
	records := catalog.Snapshot()
	var threats []ThreatEntry

	for _, obj := range records {
		if obj.CountryCode == "IN" {
			continue // Skip Indian assets
		}

		adversaryID := uint32(obj.ID)

		// Calculate RRI at 12-hour horizon
		rri := s.CalculateRisk(adversaryID, primaryID, 12*time.Hour)

		// Skip if not a threat
		if math.IsInf(float64(rri), 1) || rri > 1.0 {
			continue
		}

		// Get metadata
		meta := s.MetadataStore.Get(adversaryID)

		// Create threat entry
		entry := ThreatEntry{
			ObjectID:      obj.ID,
			ObjectName:    obj.Name,
			CountryCode:   obj.CountryCode,
			PlatformClass: meta.PlatformClass,
			EngineType:    meta.EngineType,
			RRI:           rri,
			ThreatLevel:   s.GetThreatLevel(rri),
			FuelRemaining: meta.EstimatedDV,
		}

		threats = append(threats, entry)
	}

	// Sort by RRI (ascending - lower RRI = higher threat)
	// Simple bubble sort for now (in production: use proper sorting)
	for i := 0; i < len(threats); i++ {
		for j := i + 1; j < len(threats); j++ {
			if threats[j].RRI < threats[i].RRI {
				threats[i], threats[j] = threats[j], threats[i]
			}
		}
	}

	return threats
}

// ThreatEntry represents a single entry in the prioritized threat list
type ThreatEntry struct {
	ObjectID      int
	ObjectName    string
	CountryCode   string
	PlatformClass string
	EngineType    EngineType
	RRI           float32
	ThreatLevel   ThreatLevel
	FuelRemaining float32
}
