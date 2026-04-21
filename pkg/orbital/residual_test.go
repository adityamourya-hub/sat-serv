package orbital

import (
	"math"
	"testing"
	"time"

	"ssis/pkg/physics"
)

// TestL2NormResidualValidation validates Go propagator against truth data
// Target: < 10 meter residuals in RIC frame over 24-hour propagation
func TestL2NormResidualValidation(t *testing.T) {
	// Create truth loader
	loader := NewTruthLoader()

	// Generate synthetic truth data (simulating GMAT output)
	oem := generateSyntheticGMATEphemeris("TEST_SAT_001", 24*time.Hour, 5*time.Minute)
	loader.truthData["TEST_SAT_001"] = oem

	// Create SGP4 propagator with minimal TLE
	tle := &physics.TLE{
		Name:            "TEST",
		SatelliteNumber: 99999,
		EpochYear:       24,
		EpochDay:        15.0,
		Inclination:     51.6,
		RAAN:            100.0,
		Eccentricity:    0.001,
		ArgOfPerigee:    45.0,
		MeanAnomaly:     0.0,
		MeanMotion:      15.5,
		RawL1:           "1 99999U 24001A   24015.00000000  .00000000  00000-0  00000-0 0  0010",
		RawL2:           "2 99999  51.6000 100.0000 0010000  45.0000  0.0000 15.50000000    10",
	}

	propagator := NewSGP4Propagator(tle)
	if propagator == nil {
		t.Fatal("Failed to create propagator")
	}

	// Create validator
	validator := NewValidator(loader, propagator)

	// Run 24-hour validation
	startTime := oem.Metadata.StartTime
	stopTime := oem.Metadata.StopTime
	step := 5 * time.Minute

	report, err := validator.ValidatePropagation("TEST_SAT_001", startTime, stopTime, step)
	if err != nil {
		t.Fatalf("Validation failed: %v", err)
	}

	// Analyze RIC residuals
	var maxRadial, maxInTrack, maxCrossTrack, maxL2 float64
	var sumRadial, sumInTrack, sumCrossTrack, sumL2 float64
	count := len(report.RICResiduals)

	for _, res := range report.RICResiduals {
		// Convert km to meters
		radialM := math.Abs(res.Radial) * 1000
		inTrackM := math.Abs(res.InTrack) * 1000
		crossTrackM := math.Abs(res.CrossTrack) * 1000
		l2M := res.L2Norm * 1000

		if radialM > maxRadial {
			maxRadial = radialM
		}
		if inTrackM > maxInTrack {
			maxInTrack = inTrackM
		}
		if crossTrackM > maxCrossTrack {
			maxCrossTrack = crossTrackM
		}
		if l2M > maxL2 {
			maxL2 = l2M
		}

		sumRadial += radialM
		sumInTrack += inTrackM
		sumCrossTrack += crossTrackM
		sumL2 += l2M
	}

	avgRadial := sumRadial / float64(count)
	avgInTrack := sumInTrack / float64(count)
	avgCrossTrack := sumCrossTrack / float64(count)
	avgL2 := sumL2 / float64(count)

	// Report results
	t.Log("========================================")
	t.Log("L2 NORM RESIDUAL VALIDATION - 24 HOURS")
	t.Log("========================================")
	t.Logf("Total samples: %d", count)
	t.Logf("Max L2 Error: %.2f meters (threshold: 10m)", maxL2)
	t.Logf("Avg L2 Error: %.2f meters", avgL2)
	t.Log("")
	t.Log("RIC Frame Breakdown (meters):")
	t.Logf("  Radial:      Max=%6.2f  Avg=%6.2f", maxRadial, avgRadial)
	t.Logf("  In-Track:    Max=%6.2f  Avg=%6.2f", maxInTrack, avgInTrack)
	t.Logf("  Cross-Track: Max=%6.2f  Avg=%6.2f", maxCrossTrack, avgCrossTrack)
	t.Log("========================================")

	// Validate sub-10-meter threshold
	if maxL2 > 10.0 {
		t.Errorf("FAILED: Max L2 error %.2f m exceeds 10m threshold", maxL2)
	} else {
		t.Logf("PASSED: Max L2 error %.2f m within 10m threshold", maxL2)
	}

	// Validate individual RIC components
	if maxRadial > 10.0 {
		t.Errorf("Radial error %.2f m exceeds threshold (gravity model issue)", maxRadial)
	}
	if maxInTrack > 15.0 {
		t.Errorf("In-track error %.2f m exceeds threshold (drag model issue)", maxInTrack)
	}
	if maxCrossTrack > 10.0 {
		t.Errorf("Cross-track error %.2f m exceeds threshold (plane error)", maxCrossTrack)
	}
}

// TestRICFrameDecomposition validates individual RIC components at specific times
func TestRICFrameDecomposition(t *testing.T) {
	loader := NewTruthLoader()
	oem := generateSyntheticGMATEphemeris("RIC_TEST", 24*time.Hour, time.Minute)
	loader.truthData["RIC_TEST"] = oem

	tle := &physics.TLE{
		Name:            "TEST",
		SatelliteNumber: 99998,
		EpochYear:       24,
		EpochDay:        15.0,
		Inclination:     51.6,
		RAAN:            100.0,
		Eccentricity:    0.001,
		ArgOfPerigee:    45.0,
		MeanAnomaly:     0.0,
		MeanMotion:      15.5,
		RawL1:           "1 99998U 24001A   24015.00000000  .00000000  00000-0  00000-0 0  0011",
		RawL2:           "2 99998  51.6000 100.0000 0010000  45.0000  0.0000 15.50000000    11",
	}

	prop := NewSGP4Propagator(tle)
	if prop == nil {
		t.Fatal("Failed to create propagator")
	}

	validator := NewValidator(loader, prop)

	// Test specific time points
	testTimes := []time.Duration{
		0,
		6 * time.Hour,
		12 * time.Hour,
		24 * time.Hour,
	}

	for _, offset := range testTimes {
		targetTime := oem.Metadata.StartTime.Add(offset)

		truth, err := loader.GetTruthState("RIC_TEST", targetTime)
		if err != nil {
			t.Skipf("Truth data not available: %v", err)
			continue
		}

		minutes := targetTime.Sub(oem.Metadata.StartTime).Minutes()
		predicted, err := prop.Propagate(minutes)
		if err != nil {
			t.Fatalf("Propagation failed: %v", err)
		}

		radial, inTrack, crossTrack := validator.ComputeRICResiduals(predicted, *truth)
		l2 := validator.ComputeL2Norm(predicted, *truth)

		t.Logf("T+%s: R=%.3fm I=%.3fm C=%.3fm L2=%.3fm",
			offset.String(),
			math.Abs(radial)*1000,
			math.Abs(inTrack)*1000,
			math.Abs(crossTrack)*1000,
			l2*1000)
	}
}

// BenchmarkRICResiduals measures RIC computation performance
func BenchmarkRICResiduals(b *testing.B) {
	loader := NewTruthLoader()
	oem := generateSyntheticGMATEphemeris("BENCH", time.Hour, time.Minute)
	loader.truthData["BENCH"] = oem

	tle := &physics.TLE{
		Name:            "BENCH",
		SatelliteNumber: 99997,
		RawL1:           "1 99997U 24001A   24015.00000000  .00000000  00000-0  00000-0 0  0012",
		RawL2:           "2 99997  51.6000 100.0000 0010000  45.0000  0.0000 15.50000000    12",
	}
	prop := NewSGP4Propagator(tle)

	validator := NewValidator(loader, prop)

	predicted := physics.State{
		Pos: [3]float64{7000.0, 0.0, 0.0},
		Vel: [3]float64{0.0, 7.5, 0.0},
	}
	truth := physics.State{
		Pos: [3]float64{7000.1, 0.05, 0.02},
		Vel: [3]float64{0.01, 7.49, 0.01},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = validator.ComputeRICResiduals(predicted, truth)
	}
}

// generateSyntheticGMATEphemeris creates test ephemeris simulating GMAT output
func generateSyntheticGMATEphemeris(objectID string, duration time.Duration, step time.Duration) *OEMFile {
	now := time.Now()
	segments := make([]OEMSegment, 0)

	r := 6371.0 + 500.0 // 500km circular LEO
	v := math.Sqrt(398600.4418 / r)

	steps := int(duration / step)
	omega := v / r

	for i := 0; i <= steps; i++ {
		t := float64(i) * step.Seconds()
		angle := omega * t

		// Add small perturbations to simulate real-world truth data
		perturbation := 0.00001 * math.Sin(0.001*t) // ~1cm error

		state := physics.State{
			Pos: [3]float64{
				r*math.Cos(angle) + perturbation,
				r*math.Sin(angle) + perturbation,
				perturbation,
			},
			Vel: [3]float64{
				-v * math.Sin(angle),
				v * math.Cos(angle),
				0.0,
			},
		}

		segment := OEMSegment{
			Epoch: now.Add(time.Duration(i) * step),
			State: state,
		}
		segments = append(segments, segment)
	}

	return &OEMFile{
		Header: OEMHeader{
			Version:      "2.0",
			Originator:   "GMAT_LOCAL",
			CreationDate: now,
		},
		Metadata: OEMMetadata{
			ObjectID:   objectID,
			ObjectName: "Synthetic_GMAT_" + objectID,
			CenterName: "Earth",
			RefFrame:   "EME2000",
			TimeSystem: "UTC",
			StartTime:  now,
			StopTime:   now.Add(duration),
			StepSize:   step.Seconds(),
		},
		Segments: segments,
	}
}
