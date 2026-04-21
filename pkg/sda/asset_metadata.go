package sda

import (
	"math"
	"time"

	"ssis/pkg/catalog"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Asset Metadata and Propulsion Plane
 *
 * Implements the Fuel-Capacity Metadata Table that transforms
 * Reachability Risk Index (RRI) from geometric "could they?" to
 * strategic "can they?" analysis.
 */

// EngineType represents the propulsion system type
type EngineType uint8

const (
	EnginePassive  EngineType = 0 // No propulsion (debris, dead satellites)
	EngineChemical EngineType = 1 // Chemical monoprop/biprop
	EngineElectric EngineType = 2 // Electric/Ion propulsion
	EngineColdGas  EngineType = 3 // Cold gas thrusters
)

func (et EngineType) String() string {
	switch et {
	case EnginePassive:
		return "PASSIVE"
	case EngineChemical:
		return "CHEMICAL"
	case EngineElectric:
		return "ELECTRIC"
	case EngineColdGas:
		return "COLD_GAS"
	default:
		return "UNKNOWN"
	}
}

// AssetMetadata represents the propulsion plane for a space object
type AssetMetadata struct {
	TargetID      uint32
	PlatformClass string // e.g., "Shijian-Inspector", "Starlink", "GSAT"
	EngineType    EngineType
	EstimatedDV   float32 // Total budget in m/s
	BurnRate      float32 // Typical maneuver acceleration (m/s²)
	LastManifold  int64   // Last confirmed ΔV event (UnixNano)
	Isp           float32 // Specific impulse (seconds)
	MassInitial   float32 // Initial mass (kg)
	MassFinal     float32 // Final mass (kg) - for fuel tracking

	// ADITI 4.0: Hamiltonian Provenance Hash for anti-spoof verification
	// Every catalog entry carries a physics-based cryptographic seal
	HamiltonianHash      string  // SHA-256 hash of energy state
	HashTimestamp        int64   // Unix timestamp when hash was computed
	EnergyDriftSigma     float64 // σ_H / H_avg at ingestion
	LastValidationStatus string  // "VALID", "DEGRADED", "SPOOF_DETECTED"
}

// MetadataStore manages the propulsion metadata for all catalog objects
type MetadataStore struct {
	metadata map[uint32]AssetMetadata
}

// NewMetadataStore creates a new metadata store
func NewMetadataStore() *MetadataStore {
	return &MetadataStore{
		metadata: make(map[uint32]AssetMetadata),
	}
}

// Get retrieves metadata for a target ID
func (ms *MetadataStore) Get(targetID uint32) AssetMetadata {
	meta, ok := ms.metadata[targetID]
	if !ok {
		// Return default passive metadata if not found
		return AssetMetadata{
			TargetID:    targetID,
			EngineType:  EnginePassive,
			EstimatedDV: 0,
			BurnRate:    0,
		}
	}
	return meta
}

// Set updates metadata for a target ID
func (ms *MetadataStore) Set(meta AssetMetadata) {
	ms.metadata[meta.TargetID] = meta
}

// DecrementFuel reduces the EstimatedDV after a detected maneuver
// This is called when SINDy detects a ΔV event
func (ms *MetadataStore) DecrementFuel(targetID uint32, deltaV float32) {
	meta := ms.Get(targetID)
	if meta.EstimatedDV > 0 {
		meta.EstimatedDV -= deltaV
		meta.LastManifold = time.Now().UnixNano()
		ms.Set(meta)
	}
}

// UpdateHamiltonianHash updates the Hamiltonian provenance hash for a target
// This is called by the ingestion pipeline after validation
func (ms *MetadataStore) UpdateHamiltonianHash(targetID uint32, hash string, driftSigma float64, status string) {
	meta := ms.Get(targetID)
	meta.HamiltonianHash = hash
	meta.HashTimestamp = time.Now().UnixNano()
	meta.EnergyDriftSigma = driftSigma
	meta.LastValidationStatus = status
	ms.Set(meta)
}

// GetValidationStatus returns the current Hamiltonian validation status
func (ms *MetadataStore) GetValidationStatus(targetID uint32) (hash string, driftSigma float64, status string, timestamp int64) {
	meta := ms.Get(targetID)
	return meta.HamiltonianHash, meta.EnergyDriftSigma, meta.LastValidationStatus, meta.HashTimestamp
}

// EstimateFromConvNeXt estimates propulsion metadata from ConvNeXt classification
// In production, this queries the ML model for platform classification
func (ms *MetadataStore) EstimateFromConvNeXt(obj catalog.SDAObject, platformClass string) AssetMetadata {
	meta := AssetMetadata{
		TargetID:      uint32(obj.ID),
		PlatformClass: platformClass,
		LastManifold:  time.Now().UnixNano(),
	}

	// Estimate based on asset type and platform class
	switch obj.AssetType {
	case "PAYLOAD":
		// Active satellites have propulsion
		switch platformClass {
		case "Inspector", "RPO":
			meta.EngineType = EngineChemical
			meta.Isp = 300.0         // Typical monoprop
			meta.BurnRate = 0.05     // m/s²
			meta.EstimatedDV = 500.0 // m/s
		case "HeavyBus":
			meta.EngineType = EngineChemical
			meta.Isp = 320.0          // Biprop
			meta.BurnRate = 0.1       // m/s²
			meta.EstimatedDV = 1000.0 // m/s
		default:
			meta.EngineType = EngineChemical
			meta.Isp = 300.0
			meta.BurnRate = 0.05
			meta.EstimatedDV = 300.0 // m/s
		}

	case "ROCKET_BODY":
		// Rocket bodies are typically passive
		meta.EngineType = EnginePassive
		meta.EstimatedDV = 0

	case "DEBRIS":
		// Debris is passive
		meta.EngineType = EnginePassive
		meta.EstimatedDV = 0

	default:
		// Conservative estimate for unknown types
		meta.EngineType = EnginePassive
		meta.EstimatedDV = 0
	}

	// Estimate mass from asset type
	switch obj.AssetType {
	case "PAYLOAD":
		meta.MassInitial = 1000.0 // kg
		meta.MassFinal = 800.0    // kg (200kg fuel)
	case "ROCKET_BODY":
		meta.MassInitial = 2000.0 // kg
		meta.MassFinal = 2000.0   // kg (no fuel)
	case "DEBRIS":
		meta.MassInitial = 100.0 // kg
		meta.MassFinal = 100.0   // kg
	default:
		meta.MassInitial = 500.0 // kg
		meta.MassFinal = 500.0   // kg
	}

	return meta
}

// CalculateTsiolkovskyDV calculates ΔV using Tsiolkovsky Rocket Equation
// Δv = Isp * g0 * ln(m_initial / m_final)
func CalculateTsiolkovskyDV(isp, massInitial, massFinal float32) float32 {
	g0 := float32(9.80665) // m/s²
	if massFinal <= 0 || massInitial <= massFinal {
		return 0
	}
	dv := isp * g0 * float32(math.Log(float64(massInitial)/float64(massFinal)))
	return dv
}

// EstimateFuelFromConvNeXt estimates fuel capacity based on ConvNeXt classification
// This provides the metadata for the RRI solver
func EstimateFuelFromConvNeXt(obj catalog.SDAObject) AssetMetadata {
	store := NewMetadataStore()

	// Simulate ConvNeXt classification based on object characteristics
	platformClass := inferPlatformClass(obj)

	return store.EstimateFromConvNeXt(obj, platformClass)
}

// inferPlatformClass infers platform class from object characteristics
// In production, this queries the ConvNeXt model
func inferPlatformClass(obj catalog.SDAObject) string {
	// Simple heuristic based on RCS and mass
	if obj.RCS > 10.0 {
		return "HeavyBus"
	} else if obj.RCS > 1.0 {
		if obj.CountryCode != "IN" {
			return "Inspector" // Foreign satellites with RCS > 1m² are likely inspectors
		}
		return "StandardBus"
	} else if obj.RCS > 0.1 {
		return "SmallSat"
	}

	return "Unknown"
}

// InitializeMetadataFromCatalog populates metadata from existing catalog
func (ms *MetadataStore) InitializeMetadataFromCatalog(cat *catalog.Store) {
	records := cat.Snapshot()
	for _, obj := range records {
		meta := EstimateFuelFromConvNeXt(obj)
		ms.Set(meta)
	}
}
