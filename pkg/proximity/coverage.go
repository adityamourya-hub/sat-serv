package proximity

import (
	"math"
	"ssis/pkg/catalog"
	"ssis/pkg/orbital"
	"ssis/pkg/spatial"
)

type CoverageSummary struct {
	TotalAreaCovered float64
	BlindSpotRatio   float64
	ActiveSensors    int
}

// ComputeBlindSpots identifies areas within a target ROI that are NOT currently covered by
// selected adversary satellites. It implements Requirement (e) - Blind Spot Detection.
func ComputeBlindSpots(adversaries []catalog.ObjectRecord, roi RegionOfInterest) CoverageSummary {
	// 1. Initialize Coverage Index over the ROI bounds
	// We use a high-resolution grid for the specified ROI
	index := spatial.NewCoverageIndex(180, 360)

	fovHalfAngleRad := 30.0 * orbital.DE2RA // Assume 30 deg half-angle sensor default

	for _, sat := range adversaries {
		r := math.Sqrt(sat.Pos.X*sat.Pos.X + sat.Pos.Y*sat.Pos.Y + sat.Pos.Z*sat.Pos.Z)
		alt := r - orbital.ER
		if alt < 0 {
			continue
		}

		// Calculate Footprint Radius on Earth Surface
		// Simplified conical projection: rho = alt * tan(fov)
		rhoKm := alt * math.Tan(fovHalfAngleRad)
		rhoDeg := rhoKm / (orbital.ER * orbital.DE2RA)

		// Calculate Sub-Satellite Point (SSP) in Lat/Lon
		lat := math.Asin(sat.Pos.Z/r) * 180.0 / math.Pi
		lon := math.Atan2(sat.Pos.Y, sat.Pos.X) * 180.0 / math.Pi

		// Mark footprint in the index
		index.MarkCircularCoverage(lat, lon, rhoDeg)
	}

	return CoverageSummary{
		BlindSpotRatio: index.BlindSpotRatio(),
		ActiveSensors:  len(adversaries),
	}
}
