package proximity

import (
	"fmt"
	"math"
	"ssis/pkg/catalog"
	"ssis/pkg/octree"
	"ssis/pkg/spatial"
	"time"
)

/**
 * ProximityAlert represents a potential intersection or close-approach event 
 * between a protected asset and a resident space object (RSO).
 */
type ProximityAlert struct {
	AssetID      int
	ThreatID     int
	DistanceKm   float64
	ThreatLevel  float64 // 0.0 to 1.0
	Description  string
	Interjecting bool
}

/**
 * ScreenIndianAssets implements the 'Neighbourhood Watch' screening requirement 
 * for the ADITI 4.0 challenge. 
 * It uses the high-performance 4D ST-BVH for O(n log n) spatial-temporal queries, 
 * ensuring that only objects proximal in both space and time are flagged.
 * Entry Point: Logs the current catalog size and threat radius.
 * Exit Point: Logs the number of proximity alerts generated.
 */
func ScreenIndianAssets(store *catalog.Store, threatRadiusKm float64) []ProximityAlert {
	fmt.Printf("DEBUG: Entry ScreenIndianAssets at %v radius=%.2f\n", time.Now(), threatRadiusKm)
	if store == nil {
		fmt.Printf("DEBUG: Exit ScreenIndianAssets: nil store\n")
		return nil
	}
	if threatRadiusKm <= 0 {
		threatRadiusKm = 500.0 // Default 500km screening radius
	}

	snapshot := store.Snapshot()

	// Separate Indian assets from others
	indianAssets := make([]catalog.ObjectRecord, 0)
	others := make([]spatial.BVHObject, 0, len(snapshot))

	for _, rec := range snapshot {
		if rec.CountryCode == "IND" {
			indianAssets = append(indianAssets, rec)
		}
		// All objects go into BVH for spatial queries
		others = append(others, rec)
	}

	if len(others) == 0 || len(indianAssets) == 0 {
		fmt.Printf("DEBUG: Exit ScreenIndianAssets: no objects to screen\n")
		return nil
	}

	// Build ST-BVH for spatial-temporal acceleration - O(n log n)
	bvh := spatial.NewBVH(others)

	alerts := make([]ProximityAlert, 0)
	seenPairs := make(map[[2]int]struct{})

	// For each Indian asset, query ST-BVH for nearby threats within a temporal window
	for _, asset := range indianAssets {
		// Query BVH for objects within threat radius and a 1-second temporal tolerance
		min := octree.Vector3{X: asset.Pos.X - threatRadiusKm, Y: asset.Pos.Y - threatRadiusKm, Z: asset.Pos.Z - threatRadiusKm}
		max := octree.Vector3{X: asset.Pos.X + threatRadiusKm, Y: asset.Pos.Y + threatRadiusKm, Z: asset.Pos.Z + threatRadiusKm}
		
		// Use a tight temporal window around the asset's current timestamp
		tMin := asset.Timestamp - 0.5
		tMax := asset.Timestamp + 0.5
		
		nearby := bvh.QueryRange(min, max, tMin, tMax)

		for _, obj := range nearby {
			threat := obj.(catalog.ObjectRecord)
			if threat.ID == asset.ID || threat.CountryCode == "IND" {
				continue // Skip self and other Indian assets
			}

			// Canonical pair key
			pair := [2]int{asset.ID, threat.ID}
			if pair[0] > pair[1] {
				pair[0], pair[1] = pair[1], pair[0]
			}
			if _, exists := seenPairs[pair]; exists {
				continue
			}
			seenPairs[pair] = struct{}{}

			dx := asset.Pos.X - threat.Pos.X
			dy := asset.Pos.Y - threat.Pos.Y
			dz := asset.Pos.Z - threat.Pos.Z
			dist := math.Sqrt(dx*dx + dy*dy + dz*dz)

			if dist <= threatRadiusKm {
				level, desc, interject := evaluateThreat(asset, threat, dist)
				alerts = append(alerts, ProximityAlert{
					AssetID:      asset.ID,
					ThreatID:     threat.ID,
					DistanceKm:   dist,
					ThreatLevel:  level,
					Description:  desc,
					Interjecting: interject,
				})
			}
		}
	}

	fmt.Printf("DEBUG: Exit ScreenIndianAssets: generated %d alerts\n", len(alerts))
	return alerts
}

/**
 * evaluateThreat performs a heuristic assessment of a proximal encounter.
 * It considers the asset type (Adversary vs. Debris) and distance to calculate a threat level.
 * Entry Point: Logs the encounter IDs and distance.
 * Exit Point: Resulting threat level [0, 1] and description.
 */
func evaluateThreat(asset, threat catalog.ObjectRecord, dist float64) (float64, string, bool) {
	fmt.Printf("DEBUG: Entry evaluateThreat Asset=%d Threat=%d Dist=%.2f\n", asset.ID, threat.ID, dist)
	level := 0.1 // Base level for proximity
	desc := "Proximity detected"
	interject := false

	// If the threat is an ADVERSARY asset type, boost level
	if threat.AssetType == "ADVERSARY_SAT" || threat.AssetType == "SIGINT_SAT" {
		level += 0.4
		desc = "Adversary asset in close proximity"
	}

	// Threat Level based on distance
	if dist < 100.0 {
		level += 0.3
		desc = "Tight proximity alert (<100km)"
	}
	if dist < 20.0 {
		level = 0.95
		desc = "Critical intercept/conjunction imminent"
		interject = true
	}

	resLevel := clamp01(level)
	fmt.Printf("DEBUG: Exit evaluateThreat: level=%.2f desc=%s\n", resLevel, desc)
	return resLevel, desc, interject
}
