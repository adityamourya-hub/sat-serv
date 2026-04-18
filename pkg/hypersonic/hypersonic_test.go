package hypersonic

import (
	"math"
	"ssis/pkg/fusion"
	"ssis/pkg/octree"
	"ssis/pkg/orbital"
	"testing"
	"time"
)

func TestHypersonicFilterTracking(t *testing.T) {
	// 1. Setup ground truth for HGV simulation
	// Start at 100km, speed 6 km/s
	truth := HypersonicState{
		Pos:  octree.Vector3{X: orbital.ER + 65.0, Y: 0, Z: 0},
		Vel:  octree.Vector3{X: 0, Y: 5.5, Z: 0},
		Beta: 300.0,
		LD:   0.5,
	}

	lane := NewSubOrbitalLane(99, orbital.State{X: truth.Pos.X, Y: 0, Z: 0, VX: 0, VY: 5.5, VZ: 0}, SubOrbitalLaneConfig{
		HorizonSec:  90.0,
		BetaInitial: 200.0, // Significant offset to test convergence
		LDInitial:   0.1,
	})

	now := time.Now().Unix()

	// 2. Iterate 120 seconds of 1 Hz tracking
	for i := 0; i < 120; i++ {
		ts := now + int64(i)
		jd := float64(ts)/86400.0 + 2440587.5

		// Propagate truth
		nextTruth := HypersonicPropagator(truth.ToVector(), 1.0, jd)
		truth = VectorToHypersonicState(nextTruth)

		// Generate noisy observation from SHAR (Nellore) site
		// Nellore ID: 101 ( बेंगलुरु) - using site from registry for realistic test
		zTruth := fusion.RadarMeasurementFromState(truth.ToVector()[:6], 101, jd)
		meas := fusion.RadarMeasurement{
			SiteID:       101,
			RangeKm:      zTruth[0] + 0.01, // 10m noise
			RangeRateKmS: zTruth[1],
			AzimuthDeg:   zTruth[2] * 180.0 / math.Pi,
			ElevationDeg: zTruth[3] * 180.0 / math.Pi,
			RangeStdKm:   0.05,
			AngleStdDeg:  0.1,
		}

		err := lane.ProcessObservation(meas, ts)
		if err != nil {
			t.Fatalf("Lane processing failed at T+%d: %v", i, err)
		}
	}

	// 3. Check convergence
	finalState := VectorToHypersonicState(lane.State)

	// Beta should move towards 300.0
	if math.Abs(finalState.Beta-300.0) > 100.0 {
		t.Errorf("Beta did not converge well: got %.1f, want 300.0", finalState.Beta)
	}

	// L/D should move towards 0.5
	if math.Abs(finalState.LD-0.5) > 0.4 {
		t.Errorf("L/D did not converge well: got %.2f, want 0.5", finalState.LD)
	}

	t.Logf("Final Status: %s", lane.GetStatus())
}

func TestImpactPrediction(t *testing.T) {
	// Deep-reentry state (Alt 40km, steep descent)
	s := orbital.State{X: orbital.ER + 40.0, Y: 0, Z: 0, VX: -1.0, VY: 2.0, VZ: 0}
	lane := NewSubOrbitalLane(1, s, SubOrbitalLaneConfig{HorizonSec: 120, BetaInitial: 400})

	// Force an update to trigger IPP
	meas := fusion.RadarMeasurement{SiteID: 101, RangeKm: 42.0, AzimuthDeg: 20, ElevationDeg: 85}
	lane.ProcessObservation(meas, time.Now().Unix())

	if lane.IPP == nil {
		t.Fatal("IPP was not generated")
	}

	// Impact latitude should be small (near equatorial start)
	if math.Abs(lane.IPP.LatDeg) > 15.0 {
		t.Errorf("Unexpected high latitude impact: %.3f", lane.IPP.LatDeg)
	}

	t.Logf("Impact Point: %.3f N, %.3f E", lane.IPP.LatDeg, lane.IPP.LonDeg)
}
