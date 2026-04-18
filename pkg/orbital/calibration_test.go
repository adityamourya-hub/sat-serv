package orbital_test
 
 import (
 	"math"
 	"math/rand"
	"ssis/pkg/fusion"
	"ssis/pkg/proximity"
 	"testing"
 )

// quasiMonteCarloPc provides a high-fidelity truth baseline for calibration tests.
func quasiMonteCarloPc(miss, hbr, sigX, sigY, sigZ float64, n int) float64 {
	hits := 0
	hbr2 := hbr * hbr
	for i := 0; i < n; i++ {
		// Box-Muller for 3D Gaussian
		u1 := rand.Float64()
		u2 := rand.Float64()
		u3 := rand.Float64()
		u4 := rand.Float64()
		
		z1 := math.Sqrt(-2*math.Log(math.Max(u1, 1e-12))) * math.Cos(2*math.Pi*u2)
		z2 := math.Sqrt(-2*math.Log(math.Max(u1, 1e-12))) * math.Sin(2*math.Pi*u2)
		z3 := math.Sqrt(-2*math.Log(math.Max(u3, 1e-12))) * math.Cos(2*math.Pi*u4)
		
		dx := miss + z1*sigX
		dy := z2 * sigY
		dz := z3 * sigZ
		
		if dx*dx+dy*dy+dz*dz <= hbr2 {
			hits++
		}
	}
	return float64(hits) / float64(n)
}

func TestProbabilityCalibrationBrierScore(t *testing.T) {
	// 1. Generate synthetic conjunction dataset
	rand.Seed(1337)
	nSamples := 500
	inputs := make([]proximity.CollisionRiskInput, nSamples)
	truePcs := make([]float64, nSamples)
	labels := make([]bool, nSamples)
	preds := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		// Sample realistic collision scenarios
		miss := rand.Float64() * 0.5 + 0.01 // 10m to 510m
		hbr := 0.01                        // 10m
		sig := rand.Float64()*0.05 + 0.005  // 5m to 55m
		
		inputs[i] = proximity.CollisionRiskInput{
			MissDistanceKm:      miss,
			HardbodyRadiusKm:    hbr,
			SigmaRadialKm:       sig,
			SigmaIntrackKm:      sig * 1.5,
			SigmaCrosstrackKm:   sig * 0.8,
			RelativeVelocityKmS: 7.5,
			TimeToTCASeconds:    300,
		}

		// Calculate "Truth" using high-sample QMC
		truePcs[i] = quasiMonteCarloPc(miss, hbr, sig, sig*1.5, sig*0.8, 10000)
		
		// Generate binary label from truth
		if rand.Float64() < truePcs[i] {
			labels[i] = true
		}

		// Initial prediction (uncalibrated)
		res := proximity.EvaluateCollisionRisk(inputs[i])
		preds[i] = res.Probability
	}

	// 2. Training phase: Use first 80% to train calibrator
	trainSize := int(float64(nSamples) * 0.8)
	cal := fusion.NewProbabilityCalibrator(preds[:trainSize], labels[:trainSize])

	// 3. Validation phase: Use remaining 20% to evaluate
	valPreds := preds[trainSize:]
	valLabels := labels[trainSize:]
	calibratedPreds := make([]float64, len(valPreds))
	for i := range valPreds {
		calibratedPreds[i] = cal.Calibrate(valPreds[i])
	}

	metrics := fusion.EvaluatePcCalibration(calibratedPreds, valLabels)

	t.Logf("Calibration Results (N=%d):", metrics.Samples)
	t.Logf("  Brier Score: %.4f (Target < 0.20)", metrics.BrierScore)
	t.Logf("  Reliability Error: %.4f (Target < 0.10)", metrics.ReliabilityError)
	t.Logf("  Calibration Slope (Alpha): %.4f", cal.Alpha)
	t.Logf("  Calibration Offset (Beta): %.4f", cal.Beta)

	if !metrics.CalibrationPass {
		t.Errorf("Calibration failed pass criteria: Brier=%.4f, Rel=%.4f", metrics.BrierScore, metrics.ReliabilityError)
	}
}
