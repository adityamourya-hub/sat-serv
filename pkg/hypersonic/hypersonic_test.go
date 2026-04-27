package hypersonic

import (
	"math"
	"ssis/pkg/fusion"
	"ssis/pkg/physics"
	"testing"
	"time"
)

func TestHypersonicFilterTracking(t *testing.T) {
	// SKIPPED: Test needs refactoring for HGVState API changes
	t.Fatal("NOT IMPLEMENTED: Test needs refactoring for HGVState API changes")
}

func TestImpactPrediction(t *testing.T) {
	// Deep-reentry state (Alt 40km, steep descent)
	s := physics.State{Pos: physics.Vector3{6378.137 + 40.0, 0, 0}, Vel: physics.Vector3{-1.0, 2.0, 0}}
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
