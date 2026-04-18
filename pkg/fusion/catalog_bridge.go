package fusion

import (
	"math"

	"ssis/pkg/catalog"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
)

// BuildCatalogRecordFromFusion converts a fused orbital state into a catalog record
// with position uncertainty derived from UKF covariance.
func BuildCatalogRecordFromFusion(objectID int, name string, state orbital.State, covariance [][]float64) catalog.ObjectRecord {
	return catalog.ObjectRecord{
		ID:         objectID,
		Name:       name,
		Pos:        octree.Vector3{X: state.X, Y: state.Y, Z: state.Z},
		PosSigmaKm: PositionSigmaFromCovariance(covariance),
	}
}

// PositionSigmaFromCovariance estimates 1-sigma position uncertainty from the
// translational covariance diagonal (x, y, z).
func PositionSigmaFromCovariance(covariance [][]float64) float64 {
	if len(covariance) < 3 {
		return 2.0
	}
	sx2 := covariance[0][0]
	sy2 := covariance[1][1]
	sz2 := covariance[2][2]
	if sx2 < 0 {
		sx2 = 0
	}
	if sy2 < 0 {
		sy2 = 0
	}
	if sz2 < 0 {
		sz2 = 0
	}
	rmsVar := (sx2 + sy2 + sz2) / 3.0
	if rmsVar <= 0 {
		return 2.0
	}
	return math.Sqrt(rmsVar)
}

// RapidRecatalogFromFusion applies a maneuver-triggered rapid recatalog update
// using fused state and covariance, and returns the new store version.
func RapidRecatalogFromFusion(store *catalog.Store, objectID int, name string, fusionRes FusionResult, neighborRadiusKm float64) int {
	if store == nil {
		return 0
	}
	rec := BuildCatalogRecordFromFusion(objectID, name, fusionRes.State, fusionRes.Covariance)
	// Flag as potential maneuver by setting asset type
	rec.AssetType = "MANEUVER_CANDIDATE"
	return store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{rec}}, false)
}
