package proximity

import (
	"math"
	"ssis/pkg/catalog"
	"ssis/pkg/physics"
	"ssis/pkg/spatial"
)

type CoverageSummary struct {
	TotalAreaCovered float64
	BlindSpotRatio   float64
	ActiveSensors    int
}

/**
 * ComputeBlindSpots identifies areas within a target ROI that are NOT currently covered by
 * selected adversary satellites. It implements Requirement (e) - Blind Spot Detection.
 */
func ComputeBlindSpots(adversaries []catalog.ObjectRecord, roi RegionOfInterest) CoverageSummary {
	// 1. Initialize Coverage Index over the ROI bounds
	index := spatial.NewCoverageIndex(180, 360)

	fovHalfAngleRad := 30.0 * physics.RAD_PER_DEG // Assume 30 deg half-angle sensor default

	for _, sat := range adversaries {
		r := math.Sqrt(sat.Pos.X*sat.Pos.X + sat.Pos.Y*sat.Pos.Y + sat.Pos.Z*sat.Pos.Z)
		alt := r - physics.EARTH_R_EQUATORIAL
		if alt < 0 {
			continue
		}

		// 2. Calculate Footprint Radius on Earth Surface
		// Simplified conical projection: rhoKm = alt * tan(fov)
		rhoKm := alt * math.Tan(fovHalfAngleRad)
		rhoDeg := rhoKm / (physics.EARTH_R_EQUATORIAL * physics.RAD_PER_DEG)

		// 3. Calculate Sub-Satellite Point (SSP) in Lat/Lon
		lat := math.Asin(sat.Pos.Z/r) * (180.0 / math.Pi)
		lon := math.Atan2(sat.Pos.Y, sat.Pos.X) * (180.0 / math.Pi)

		// 4. Mark footprint in the index
		index.MarkCircularCoverage(lat, lon, rhoDeg)
	}

	return CoverageSummary{
		BlindSpotRatio: index.BlindSpotRatio(),
		ActiveSensors:  len(adversaries),
	}
}
