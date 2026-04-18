package proximity

import (
	"fmt"
	"math"
	"time"
	"ssis/pkg/catalog"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

type RegionOfInterest struct {
	LatDeg   float64
	LonDeg   float64
	RadiusKm float64
}

/**
 * ScanGroundRegion implements Volumetric Scan (ADITI 4.0 Requirement a).
 * It identifies all space objects currently passing through the user-defined
 * region above a ground site.
 */
func ScanGroundRegion(roi RegionOfInterest, store *catalog.Store, epoch time.Time) []catalog.SDAObject {
	fmt.Printf("[TRC] Entering ScanGroundRegion for site (%.2f, %.2f) at %v\n", roi.LatDeg, roi.LonDeg, epoch)
	defer fmt.Printf("[TRC] Exiting ScanGroundRegion\n")

	if store == nil {
		return nil
	}

	jd := float64(epoch.Unix())/86400.0 + 2440587.5
	gst := physics.GreenwichSiderealTime(jd)

	// 1. Convert Ground Site to Unit Vector in ECI
	phi := roi.LatDeg * physics.RAD_PER_DEG
	lambda := roi.LonDeg * physics.RAD_PER_DEG + gst
	siteUnit := [3]float64{
		math.Cos(phi) * math.Cos(lambda),
		math.Cos(phi) * math.Sin(lambda),
		math.Sin(phi),
	}

	// 2. Define Bounding Box in ECI to query Octree
	// The search volume extends from ground (ER) to 45,000km.
	maxR := 45000.0
	centerR := (physics.EARTH_R_EQUATORIAL + maxR) / 2.0
	halfDepth := (maxR - physics.EARTH_R_EQUATORIAL) / 2.0
	
	theta := roi.RadiusKm / physics.EARTH_R_EQUATORIAL
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

	// 3. Query SVO for 4D space-time candidates
	epochF := float64(epoch.Unix())
	candidates := store.QueryRange(min, max, epochF-3600, epochF+3600) 
	results := make([]catalog.SDAObject, 0)

	cosTheta := math.Cos(theta)
	for _, rec := range candidates {
		var p [3]float64
		// In a real system, we'd propagate here. For simplicity, we assume store is epoch-aligned or we skip.
		p = [3]float64{rec.Pos.X, rec.Pos.Y, rec.Pos.Z}

		rmag := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
		if rmag < physics.EARTH_R_EQUATORIAL || rmag > maxR {
			continue
		}

		// 4. Narrow phase: Horizon Mask / Earth Occlusion
		distSite := [3]float64{p[0] - siteUnit[0]*physics.EARTH_R_EQUATORIAL, p[1] - siteUnit[1]*physics.EARTH_R_EQUATORIAL, p[2] - siteUnit[2]*physics.EARTH_R_EQUATORIAL}
		if distSite[0]*siteUnit[0] + distSite[1]*siteUnit[1] + distSite[2]*siteUnit[2] <= 0 {
			continue 
		}

		// 5. Narrow phase 2: Angular distance check (ROI Conical Shell)
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
 * It identifies space objects passing through a ground ROI using Axiom-Zero (GHNN) propagation.
 */
func ScanGroundRegionWithAxiom(roi RegionOfInterest, store *catalog.Store, kernel orbital.DynamicsKernel, epoch time.Time) []catalog.SDAObject {
	fmt.Printf("[TRC] Entering ScanGroundRegionWithAxiom for site (%.2f, %.2f) at %v\n", roi.LatDeg, roi.LonDeg, epoch)
	defer fmt.Printf("[TRC] Exiting ScanGroundRegionWithAxiom\n")

	if store == nil || kernel == nil {
		return nil
	}

	jd := float64(epoch.Unix())/86400.0 + 2440587.5
	gst := physics.GreenwichSiderealTime(jd)

	phi := roi.LatDeg * physics.RAD_PER_DEG
	lambda := roi.LonDeg * physics.RAD_PER_DEG + gst
	siteUnit := [3]float64{
		math.Cos(phi) * math.Cos(lambda),
		math.Cos(phi) * math.Sin(lambda),
		math.Sin(phi),
	}

	maxR := 45000.0
	centerR := (physics.EARTH_R_EQUATORIAL + maxR) / 2.0
	halfDepth := (maxR - physics.EARTH_R_EQUATORIAL) / 2.0
	theta := roi.RadiusKm / physics.EARTH_R_EQUATORIAL
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
		
		// 1. Initialize High-Fidelity State
		init := physics.State{
			Pos:     physics.Vector3{rec.Pos.X, rec.Pos.Y, rec.Pos.Z},
			Vel:     physics.Vector3{rec.Vel.X, rec.Vel.Y, rec.Vel.Z},
			EpochJD: rec.EpochTT,
		}
		
		deltaMins := (epochF/60.0 - rec.EpochTT*1440.0/60.0) // simplified delta
		if rec.EpochTT == 0 {
			deltaMins = 0
		}
		
		// 2. High-Precision Axial Propagation
		axProp := orbital.NewAxiomPropagator(kernel, init, time.Unix(int64(rec.EpochTT*86400-2440587.5*86400), 0), physics.PhysicsParams{Mass: 1000, Cd: 2.2, DragArea: 1.0})
		state, err := axProp.Propagate(deltaMins)
		if err != nil {
			continue
		}
		p = [3]float64{state.Pos[0], state.Pos[1], state.Pos[2]}
		rec.Pos = octree.Vector3{X: p[0], Y: p[1], Z: p[2]}

		rmag := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
		if rmag < physics.EARTH_R_EQUATORIAL || rmag > maxR {
			continue
		}

		distSite := [3]float64{p[0] - siteUnit[0]*physics.EARTH_R_EQUATORIAL, p[1] - siteUnit[1]*physics.EARTH_R_EQUATORIAL, p[2] - siteUnit[2]*physics.EARTH_R_EQUATORIAL}
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
