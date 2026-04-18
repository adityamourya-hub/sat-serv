package orbital_test

import (
	"math"
	"ssis/pkg/catalog"
	"ssis/pkg/hypersonic"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
	"ssis/pkg/proximity"
	"testing"
	"time"
)

func TestADITIHardeningSuite(t *testing.T) {
	// 1. Setup Scaling Catalog
	store := catalog.NewStore(500.0)
	epoch := time.Now()

	// Compute ECI direction of ROI center (lat=0, lon=0) at test epoch
	jd := float64(epoch.Unix())/86400.0 + 2440587.5
	gst := orbital.GreenwichSiderealTime(jd)
	siteX := math.Cos(gst) // lat=0, lon=0+gst
	siteY := math.Sin(gst)

	// Add 100 objects along the ROI ECI direction to test voxel efficiency
	for i := 0; i < 100; i++ {
		r := 7000.0 + float64(i)
		store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: []catalog.ObjectRecord{
			{ID: i, Name: "DB_OBJ", Pos: octree.Vector3{X: siteX * r, Y: siteY * r, Z: 0}, Timestamp: float64(epoch.Unix())},
		}}, true)
	}

	// 2. Test Voxel Scanning (Requirement a - Hardened)
	t.Run("VoxelScanningEfficiency", func(t *testing.T) {
		roi := proximity.RegionOfInterest{LatDeg: 0, LonDeg: 0, RadiusKm: 100}
		// This now used the QueryRange internal logic
		found := proximity.ScanGroundRegion(roi, store, epoch)
		t.Logf("Hardened Volumetric Scan found %d objects in ROI volume", len(found))
		// Should find objects near the X-axis site
		if len(found) == 0 {
			t.Errorf("Voxel scan failed to detect objects in broad bounding box")
		}
	})

	// 3. Test MOID Calculation (Requirement c)
	t.Run("MOIDCalculation", func(t *testing.T) {
		s1 := orbital.State{X: 7000, Y: 0, Z: 0, VX: 0, VY: 7.5, VZ: 0}
		s2 := orbital.State{X: 7000, Y: 0, Z: 0, VX: 0, VY: 0, VZ: 7.5} // Crossing at 90 deg

		moid := proximity.CalculateMOID(s1, s2)
		t.Logf("MOID distance: %.4f km", moid)
		if moid > 1.0 { // Should be exactly 0 at intersection
			t.Errorf("Expected small MOID for intersecting orbits, got %.4f", moid)
		}
	})

	// 4. Test Re-entry Prediction (Requirement h)
	t.Run("ReentryImpact", func(t *testing.T) {
		// Start at 120km, steep descent
		initial := orbital.State{X: 6378.137 + 120, Y: 0, Z: 0, VX: -1.0, VY: 6.5, VZ: 0}
		impact := hypersonic.PredictImpactFootprint(initial, 0.05, 150.0, 15.0)

		t.Logf("Predict Impact at Lat: %.2f, Lon: %.2f", impact.LatDeg, impact.LonDeg)
		if math.Abs(impact.LatDeg) > 90 || math.Abs(impact.LonDeg) > 180 {
			t.Errorf("Invalid impact coordinates: Lat=%.2f, Lon=%.2f", impact.LatDeg, impact.LonDeg)
		}
	})

	// 5. Test RK45 Ballistic Integrator (Requirement i)
	t.Run("RK45Integrator", func(t *testing.T) {
		rk := physics.NewIntegratorRKF45()
		// Simple gravity-only derivative
		mu := 398600.44
		f := func(t float64, y []float64) []float64 {
			r := math.Sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])
			r3 := r * r * r
			return []float64{y[3], y[4], y[5], -mu * y[0] / r3, -mu * y[1] / r3, -mu * y[2] / r3}
		}

		y := []float64{7000, 0, 0, 0, 7.5, 0}
		tOut, yOut, nextDt := rk.Step(0, 10.0, y, f)

		t.Logf("RK45 Step: t=%.1f, nextDt=%.1f", tOut, nextDt)
		if tOut != 10.0 || len(yOut) != 6 {
			t.Errorf("RK45 step failed")
		}
	})
}
