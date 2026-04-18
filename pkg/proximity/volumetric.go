package proximity

import (
	"math"
	"time"
	"ssis/pkg/catalog"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
)

type RegionOfInterest struct {
	LatDeg   float64
	LonDeg   float64
	RadiusKm float64
}

// ScanGroundRegion implements Volumetric Scan (ADITI 4.0 Requirement a).
// It identifies all space objects currently passing through the user-defined
// region above a ground site.
func ScanGroundRegion(roi RegionOfInterest, store *catalog.Store, epoch time.Time) []catalog.SDAObject {
	if store == nil {
		return nil
	}

	jd := float64(epoch.Unix())/86400.0 + 2440587.5
	gst := orbital.GreenwichSiderealTime(jd)

	// 1. Convert Ground Site to Unit Vector in ECI
	phi := roi.LatDeg * orbital.DE2RA
	lambda := roi.LonDeg * orbital.DE2RA + gst
	siteUnit := [3]float64{
		math.Cos(phi) * math.Cos(lambda),
		math.Cos(phi) * math.Sin(lambda),
		math.Sin(phi),
	}

	// 2. Define Bounding Box in ECI to query Octree
	// The search volume extends from ground (ER) to 45,000km.
	// We create a box that encapsulates the conical shell.
	maxR := 45000.0
	centerR := (orbital.ER + maxR) / 2.0
	halfDepth := (maxR - orbital.ER) / 2.0
	
	// Exact conical geometry for large ROIs:
	// theta is the geocentric angle. 
	theta := roi.RadiusKm / orbital.ER
	// The maximum lateral spread from the central axis at maxR is maxR * sin(theta) if it's a cone.
	// But actually, we want a bounding box that guarantees covering the cone from ER to maxR.
	sideLength := maxR * math.Tan(theta)
	if theta > math.Pi/4 {
		sideLength = maxR // Cap it to ensure we don't blow up the box
	}

	min := octree.Vector3{
		X: siteUnit[0]*centerR - sideLength - halfDepth,
		Y: siteUnit[1]*centerR - sideLength - halfDepth,
		Z: siteUnit[2]*centerR - sideLength - halfDepth,
	}
	max := octree.Vector3{
		X: siteUnit[0]*centerR + sideLength + halfDepth,
		Y: siteUnit[1]*centerR + sideLength + halfDepth,
		Z: siteUnit[2]*centerR + sideLength + halfDepth,
	}

	// 3. Query SVO for 4D space-time candidates
	// Bound the temporal window drastically to match epoch +/- 1 hour
	epochF := float64(epoch.Unix())
	candidates := store.QueryRange(min, max, epochF-3600, epochF+3600) 
	results := make([]catalog.SDAObject, 0)

	cosTheta := math.Cos(theta)
	for _, rec := range candidates {
		var p [3]float64
		if rec.IsSGP4 {
			pos, _, err := orbital.PropagateSGP4(rec.TLELine1, rec.TLELine2, epoch)
			if err != nil {
				continue
			}
			p = pos
			rec.Pos = octree.Vector3{X: p[0], Y: p[1], Z: p[2]}
		} else {
			p = [3]float64{rec.Pos.X, rec.Pos.Y, rec.Pos.Z}
		}

		rmag := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
		if rmag < orbital.ER || rmag > maxR {
			continue
		}

		// Narrow phase 1: Horizon Mask / Earth Occlusion
		// Vector from site to object: d = p - (siteUnit * ER)
		distSite := [3]float64{p[0] - siteUnit[0]*orbital.ER, p[1] - siteUnit[1]*orbital.ER, p[2] - siteUnit[2]*orbital.ER}
		// LOS check: distSite dot siteUnit > 0
		if distSite[0]*siteUnit[0] + distSite[1]*siteUnit[1] + distSite[2]*siteUnit[2] <= 0 {
			continue // Below horizon
		}

		// Narrow phase 2: Angular distance check (ROI Conical Shell)
		pUnit := [3]float64{p[0] / rmag, p[1] / rmag, p[2] / rmag}
		dot := pUnit[0]*siteUnit[0] + pUnit[1]*siteUnit[1] + pUnit[2]*siteUnit[2]
		
		if dot >= cosTheta {
			results = append(results, rec)
		}
	}

	return results
}

/**
 * ScanGroundRegionWithAxiom implements Volumetric Scan utilizing the Sovereign Physics Core.
 * It identifies space objects passing through a ground ROI using Axiom-Zero (rKAN) propagation.
 *
 * @param roi: Geodetic region of interest.
 * @param store: Catalog state.
 * @param kernel: Pre-loaded Axiom-Zero rKAN manifold.
 * @param epoch: Target time.
 * @return: Signed-ready records within ROI.
 */
func ScanGroundRegionWithAxiom(roi RegionOfInterest, store *catalog.Store, kernel *orbital.RKANKernel, epoch time.Time) []catalog.SDAObject {
	if store == nil || kernel == nil {
		return nil
	}

	jd := float64(epoch.Unix())/86400.0 + 2440587.5
	gst := orbital.GreenwichSiderealTime(jd)

	phi := roi.LatDeg * orbital.DE2RA
	lambda := roi.LonDeg * orbital.DE2RA + gst
	siteUnit := [3]float64{
		math.Cos(phi) * math.Cos(lambda),
		math.Cos(phi) * math.Sin(lambda),
		math.Sin(phi),
	}

	maxR := 45000.0
	centerR := (orbital.ER + maxR) / 2.0
	halfDepth := (maxR - orbital.ER) / 2.0
	theta := roi.RadiusKm / orbital.ER
	sideLength := maxR * math.Tan(theta)
	if theta > math.Pi/4 {
		sideLength = maxR
	}

	min := octree.Vector3{
		X: siteUnit[0]*centerR - sideLength - halfDepth,
		Y: siteUnit[1]*centerR - sideLength - halfDepth,
		Z: siteUnit[2]*centerR - sideLength - halfDepth,
	}
	max := octree.Vector3{
		X: siteUnit[0]*centerR + sideLength + halfDepth,
		Y: siteUnit[1]*centerR + sideLength + halfDepth,
		Z: siteUnit[2]*centerR + sideLength + halfDepth,
	}

	epochF := float64(epoch.Unix())
	candidates := store.QueryRange(min, max, epochF-3600, epochF+3600)
	results := make([]catalog.SDAObject, 0)

	cosTheta := math.Cos(theta)
	for _, rec := range candidates {
		var p [3]float64
		
		// Use Axiom Processor for high-fidelity state recovery
		// We initialize a temporary Propagator for the candidate
		init := orbital.State{X: rec.Pos.X, Y: rec.Pos.Y, Z: rec.Pos.Z}
		// For the demo, we use the timestamp stored in the record
		deltaMins := (epochF - rec.EpochTT) / 60.0
		
		axProp := orbital.NewAxiomPropagator(kernel, init, time.Unix(int64(rec.EpochTT), 0))
		state, err := axProp.Propagate(deltaMins)
		if err != nil {
			continue
		}
		p = [3]float64{state.X, state.Y, state.Z}
		rec.Pos = octree.Vector3{X: p[0], Y: p[1], Z: p[2]}

		rmag := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
		if rmag < orbital.ER || rmag > maxR {
			continue
		}

		distSite := [3]float64{p[0] - siteUnit[0]*orbital.ER, p[1] - siteUnit[1]*orbital.ER, p[2] - siteUnit[2]*orbital.ER}
		if distSite[0]*siteUnit[0]+distSite[1]*siteUnit[1]+distSite[2]*siteUnit[2] <= 0 {
			continue
		}

		pUnit := [3]float64{p[0] / rmag, p[1] / rmag, p[2] / rmag}
		dot := pUnit[0]*siteUnit[0] + pUnit[1]*siteUnit[1] + pUnit[2]*siteUnit[2]

		if dot >= cosTheta {
			results = append(results, rec)
		}
	}

	return results
}
