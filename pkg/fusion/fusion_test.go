package fusion

import (
	"testing"

	engcrypto "ssis/pkg/cryptography"
	"ssis/pkg/physics"
)

func TestFuseWithUKFRadarAndTLE(t *testing.T) {
	// Prior state: LEO object at 7000km, moving primarily in Y direction
	prior := physics.State{Pos: physics.Vector3{7000, 0, 10}, Vel: physics.Vector3{0, 7.5, 0.01}}
	cov := [][]float64{
		{10, 0, 0, 0, 0, 0},
		{0, 10, 0, 0, 0, 0},
		{0, 0, 10, 0, 0, 0},
		{0, 0, 0, 0.1, 0, 0},
		{0, 0, 0, 0, 0.1, 0},
		{0, 0, 0, 0, 0, 0.1},
	}

	// After 30s propagation, position will be approximately [7000, 225, 10.3]
	// Radar measurement should be consistent with this predicted position
	radar := &RadarMeasurement{
		SiteID:       101,  // Bengaluru site
		RangeKm:      6720, // Approximate range from Bengaluru to propagated position
		RangeRateKmS: 0.5,
		AzimuthDeg:   90, // Looking east (toward positive Y)
		ElevationDeg: 45,
		RangeStdKm:   0.1,
		RangeRateStd: 0.01,
		AngleStdDeg:  0.1,
	}

	// TLE measurement consistent with propagated state (larger covariance to pass gate)
	tlem := &TLEMeasurement{
		State:         physics.State{Pos: physics.Vector3{7000.1, 225.0, 10.3}, Vel: physics.Vector3{0.01, 7.48, 0.01}},
		PositionStdKm: 5.0,
		VelocityStdKm: 0.5,
	}

	signer := engcrypto.NewSigner("fusion-test-key")
	res, err := FuseWithUKF(FusionInput{
		ObjectID:         43286,
		EventType:        "UKF_FUSION_UPDATE",
		PriorState:       prior,
		PriorCovariance:  cov,
		DeltaTimeSec:     30,
		RadarMeasurement: radar,
		TLEMeasurement:   tlem,
		Signer:           signer,
	}, DefaultUKFConfig())
	if err != nil {
		t.Fatalf("FuseWithUKF failed: %v", err)
	}
	if len(res.SensorsUsed) < 1 {
		t.Fatalf("expected at least 1 sensor fused, got %d", len(res.SensorsUsed))
	}
	if res.ThreeSigma.AxisXKm <= 0 || res.ThreeSigma.AxisYKm <= 0 || res.ThreeSigma.AxisZKm <= 0 {
		t.Fatalf("expected positive 3-sigma ellipsoid axes, got %+v", res.ThreeSigma)
	}
	if res.SignedAlert == nil || !res.SignedAlert.SignatureValid {
		t.Fatalf("expected valid signed uncertainty alert")
	}
	if res.Covariance[0][0] <= 0 || res.Covariance[3][3] <= 0 {
		t.Fatalf("expected positive covariance diagonal")
	}
}

func TestBuildSignedUncertaintyAlert(t *testing.T) {
	signer := engcrypto.NewSigner("ukf-alert-test-key")
	state := physics.State{Pos: physics.Vector3{1000, 2000, 3000}, Vel: physics.Vector3{1, 2, 3}}
	cov := [][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0.01, 0, 0},
		{0, 0, 0, 0, 0.01, 0},
		{0, 0, 0, 0, 0, 0.01},
	}
	alert := BuildSignedUncertaintyAlert(99, "UNCERTAINTY_TEST", state, cov, Ellipsoid3Sigma{AxisXKm: 3, AxisYKm: 3, AxisZKm: 3}, signer)
	if alert.Signature == "" {
		t.Fatalf("expected signature in alert")
	}
	if !alert.SignatureValid {
		t.Fatalf("expected valid alert signature")
	}
}

func TestPropagateWithUKFProducesCovarianceEachStep(t *testing.T) {
	signer := engcrypto.NewSigner("prop-test-key")
	initial := physics.State{Pos: physics.Vector3{7000, 0, 0}, Vel: physics.Vector3{0, 7.5, 0}}
	cov := [][]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0.01, 0, 0},
		{0, 0, 0, 0, 0.01, 0},
		{0, 0, 0, 0, 0, 0.01},
	}

	seq, err := PropagateWithUKF(initial, cov, 30, 5, DefaultUKFConfig(), nil, nil, signer, 43286, "UKF_STEP")
	if err != nil {
		t.Fatalf("PropagateWithUKF failed: %v", err)
	}
	if len(seq) != 5 {
		t.Fatalf("expected 5 propagated states, got %d", len(seq))
	}
	for i, s := range seq {
		if len(s.Covariance) != 6 || len(s.Covariance[0]) != 6 {
			t.Fatalf("step %d covariance shape invalid", i)
		}
		if s.ThreeSigma.AxisXKm <= 0 || s.ThreeSigma.AxisYKm <= 0 || s.ThreeSigma.AxisZKm <= 0 {
			t.Fatalf("step %d has non-positive 3-sigma axes", i)
		}
		if s.SignedAlert == nil || !s.SignedAlert.SignatureValid {
			t.Fatalf("step %d signature should be valid", i)
		}
	}
}
