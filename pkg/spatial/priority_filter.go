// Package spatial provides spatial filtering and tactical pressure calculation
// Implements O(1) proximity checks for context-aware sensor mesh
// Uses basic spherical geometry without external S2 library dependency
package spatial

import (
	"fmt"
	"math"
)

// GeoPoint represents a geodetic point in spherical coordinates
type GeoPoint struct {
	Lat float64 // Latitude in degrees
	Lon float64 // Longitude in degrees
}

// HVA represents a High-Value Asset requiring protection
type HVA struct {
	ID         int        // Satellite ID
	Name       string     // Asset name (e.g., "GSAT-7", "RISAT-1")
	Priority   float64    // Strategic priority: 1.0 (Low) to 10.0 (Strategic)
	CurrentPos GeoPoint   // Current geodetic position
	ECIPos     [3]float64 // ECI coordinates (km)
	OrbitAlt   float64    // Orbital altitude (km)
	Type       string     // Asset type: COMM, IMAGING, NAVIGATION, WEATHER
}

// UCT represents an Uncorrelated Track (unknown object)
type UCT struct {
	ID          int        // Track ID
	Pos         GeoPoint   // Geodetic position
	ECIPos      [3]float64 // ECI coordinates (km)
	Uncertainty float64    // Measurement uncertainty (km)
	LastObsTime float64    // Last observation time (Unix seconds)
	Confidence  float64    // Detection confidence (0.0-1.0)
}

// TacticalPressureResult contains tactical pressure calculation results
type TacticalPressureResult struct {
	TotalPressure    float64           // Overall tactical pressure (Ψ)
	AssetPressures   map[int]float64   // Pressure per HVA
	UCTContributions map[int]float64   // Contribution per UCT
	ThreatLevel      string            // LOW, MEDIUM, HIGH
	ProtectionAlerts []ProtectionAlert // Alerts for UCTs in protection shell
}

// ProtectionAlert represents a proximity alert for an HVA
type ProtectionAlert struct {
	HVAID      int     // HVA ID
	HVAName    string  // HVA name
	UCTID      int     // UCT ID
	Distance   float64 // Distance from HVA (km)
	Pressure   float64 // Local pressure contribution
	AlertLevel string  // WATCH, WARNING, CRITICAL
}

// PriorityFilter manages spatial filtering and tactical pressure calculation
type PriorityFilter struct {
	protectionRadiusKm float64 // Protection shell radius (default 50km)
	earthRadiusKm      float64 // Earth radius (default 6371km)
}

// NewPriorityFilter creates a new priority filter
func NewPriorityFilter() *PriorityFilter {
	return &PriorityFilter{
		protectionRadiusKm: 50.0, // Default 50km protection shell
		earthRadiusKm:      6371.0,
	}
}

// SetProtectionRadius sets the protection shell radius
func (pf *PriorityFilter) SetProtectionRadius(radiusKm float64) {
	pf.protectionRadiusKm = radiusKm
}

// CalculateTacticalPressure computes weighted curiosity spike (Ψ)
// Ψ = Σ(P_i · σ_j / (1 + d_ij)) for all HVA-UCT pairs in protection shell
func (pf *PriorityFilter) CalculateTacticalPressure(ucts []UCT, hvas []HVA) TacticalPressureResult {
	result := TacticalPressureResult{
		AssetPressures:   make(map[int]float64),
		UCTContributions: make(map[int]float64),
		ProtectionAlerts: make([]ProtectionAlert, 0),
	}

	var totalPressure float64 = 0.0

	// Check each HVA against each UCT
	for _, hva := range hvas {
		var assetPressure float64 = 0.0

		for _, uct := range ucts {
			// Calculate spherical distance using Haversine formula
			distanceKm := pf.sphericalDistance(hva.CurrentPos, uct.Pos)

			// Check if within protection shell
			if distanceKm <= pf.protectionRadiusKm {
				// Distance-based decay: closer = higher pressure
				proximityFactor := 1.0 / (1.0 + distanceKm/pf.protectionRadiusKm)

				// Tactical pressure: Priority × Proximity × Uncertainty
				pressure := hva.Priority * proximityFactor * uct.Uncertainty

				assetPressure += pressure
				totalPressure += pressure

				// Track contributions
				result.AssetPressures[hva.ID] = assetPressure
				result.UCTContributions[uct.ID] += pressure

				// Generate protection alert
				alert := pf.generateAlert(hva, uct, distanceKm, pressure)
				result.ProtectionAlerts = append(result.ProtectionAlerts, alert)
			}
		}
	}

	result.TotalPressure = totalPressure
	result.ThreatLevel = pf.classifyThreatLevel(totalPressure)

	return result
}

// sphericalDistance calculates great-circle distance between two geodetic points using Haversine formula
func (pf *PriorityFilter) sphericalDistance(p1, p2 GeoPoint) float64 {
	lat1 := p1.Lat * math.Pi / 180.0
	lat2 := p2.Lat * math.Pi / 180.0
	deltaLat := lat2 - lat1
	deltaLon := (p2.Lon - p1.Lon) * math.Pi / 180.0

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return pf.earthRadiusKm * c
}

// generateAlert creates a protection alert for an HVA-UCT pair
func (pf *PriorityFilter) generateAlert(hva HVA, uct UCT, distanceKm float64, pressure float64) ProtectionAlert {
	alertLevel := "WATCH"

	if distanceKm < 10.0 {
		alertLevel = "CRITICAL"
	} else if distanceKm < 25.0 {
		alertLevel = "WARNING"
	}

	return ProtectionAlert{
		HVAID:      hva.ID,
		HVAName:    hva.Name,
		UCTID:      uct.ID,
		Distance:   distanceKm,
		Pressure:   pressure,
		AlertLevel: alertLevel,
	}
}

// classifyThreatLevel converts pressure to threat level
func (pf *PriorityFilter) classifyThreatLevel(pressure float64) string {
	if pressure < 5.0 {
		return "LOW"
	} else if pressure < 20.0 {
		return "MEDIUM"
	}
	return "HIGH"
}

// GetHVAsInRegion returns HVAs within a geographic region
func (pf *PriorityFilter) GetHVAsInRegion(hvas []HVA, center GeoPoint, radiusKm float64) []HVA {
	inRegion := make([]HVA, 0)

	for _, hva := range hvas {
		distance := pf.sphericalDistance(center, hva.CurrentPos)
		if distance <= radiusKm {
			inRegion = append(inRegion, hva)
		}
	}

	return inRegion
}

// FilterUCTsByDistance filters UCTs by distance from a reference point
func (pf *PriorityFilter) FilterUCTsByDistance(ucts []UCT, refPoint GeoPoint, maxDistanceKm float64) []UCT {
	filtered := make([]UCT, 0)

	for _, uct := range ucts {
		distance := pf.sphericalDistance(refPoint, uct.Pos)
		if distance <= maxDistanceKm {
			filtered = append(filtered, uct)
		}
	}

	return filtered
}

// CreateDefaultIndianAssets returns default Indian space assets with priorities
func CreateDefaultIndianAssets() []HVA {
	return []HVA{
		{
			ID:       1001,
			Name:     "GSAT-7",
			Priority: 10.0, // Strategic military comms
			Type:     "COMM",
			OrbitAlt: 36000.0, // GEO
		},
		{
			ID:       1002,
			Name:     "RISAT-1",
			Priority: 9.0, // Strategic imaging
			Type:     "IMAGING",
			OrbitAlt: 536.0, // LEO
		},
		{
			ID:       1003,
			Name:     "INSAT-3D",
			Priority: 8.0, // Weather monitoring
			Type:     "WEATHER",
			OrbitAlt: 36000.0, // GEO
		},
		{
			ID:       1004,
			Name:     "IRNSS-1A",
			Priority: 9.5, // Navigation
			Type:     "NAVIGATION",
			OrbitAlt: 36000.0, // GEO
		},
		{
			ID:       1005,
			Name:     "CARTOSAT-2",
			Priority: 8.5, // High-res imaging
			Type:     "IMAGING",
			OrbitAlt: 630.0, // LEO
		},
	}
}

// UpdateHVAPositions updates HVA positions from propagator
func (pf *PriorityFilter) UpdateHVAPositions(hvas []HVA, positions map[int][3]float64) {
	for i := range hvas {
		if pos, exists := positions[hvas[i].ID]; exists {
			// Convert ECI (km) to geodetic coordinates
			lat, lon := pf.eciToLatLon(pos[0], pos[1], pos[2])
			hvas[i].CurrentPos = GeoPoint{Lat: lat, Lon: lon}
			hvas[i].ECIPos = pos
		}
	}
}

// eciToLatLon converts ECI coordinates to geodetic latitude/longitude
func (pf *PriorityFilter) eciToLatLon(x, y, z float64) (float64, float64) {
	r := math.Sqrt(x*x + y*y + z*z)
	lat := math.Asin(z/r) * 180.0 / math.Pi
	lon := math.Atan2(y, x) * 180.0 / math.Pi
	return lat, lon
}

// latLonToECI converts geodetic coordinates to ECI
func (pf *PriorityFilter) latLonToECI(lat, lon, alt float64) [3]float64 {
	r := pf.earthRadiusKm + alt
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	x := r * math.Cos(latRad) * math.Cos(lonRad)
	y := r * math.Cos(latRad) * math.Sin(lonRad)
	z := r * math.Sin(latRad)

	return [3]float64{x, y, z}
}

// CreateSiliguriHVAs creates HVAs in the Siliguri Corridor region
func (pf *PriorityFilter) CreateSiliguriHVAs() []HVA {
	// Siliguri coordinates
	lat := 26.5
	lon := 88.4
	alt := 400.0 // LEO

	eci := pf.latLonToECI(lat, lon, alt)

	return []HVA{
		{
			ID:         2001,
			Name:       "SILIGURI-SENSOR-1",
			Priority:   8.0,
			Type:       "SENSOR",
			OrbitAlt:   alt,
			CurrentPos: GeoPoint{Lat: lat, Lon: lon},
			ECIPos:     eci,
		},
	}
}

// PrintPressureSummary prints a tactical pressure summary
func (pf *PriorityFilter) PrintPressureSummary(result TacticalPressureResult) {
	fmt.Printf("\n=== TACTICAL PRESSURE SUMMARY ===\n")
	fmt.Printf("Total Pressure (Ψ): %.2f\n", result.TotalPressure)
	fmt.Printf("Threat Level: %s\n", result.ThreatLevel)
	fmt.Printf("\nAsset Pressures:\n")
	for assetID, pressure := range result.AssetPressures {
		fmt.Printf("  Asset %d: %.2f\n", assetID, pressure)
	}
	fmt.Printf("\nTop UCT Contributors:\n")
	for uctID, contribution := range result.UCTContributions {
		if contribution > 1.0 {
			fmt.Printf("  UCT %d: %.2f\n", uctID, contribution)
		}
	}
	if len(result.ProtectionAlerts) > 0 {
		fmt.Printf("\nProtection Alerts:\n")
		for _, alert := range result.ProtectionAlerts {
			fmt.Printf("  [%s] %s - UCT %d at %.1f km (Pressure: %.2f)\n",
				alert.AlertLevel, alert.HVAName, alert.UCTID, alert.Distance, alert.Pressure)
		}
	}
	fmt.Printf("===============================\n\n")
}
