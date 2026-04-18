package maneuver

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"ssis/pkg/physics"
)

// AMOSDetectorConfig controls maneuver detection thresholds inspired by AMOS-style residual logic.
type AMOSDetectorConfig struct {
	CUSUMDrift     float64
	CUSUMThreshold float64
	SigmaGate      float64
	MinDeltaVKmS   float64
	SupportSamples int
}

func DefaultAMOSDetectorConfig() AMOSDetectorConfig {
	return AMOSDetectorConfig{
		CUSUMDrift:     0.005,
		CUSUMThreshold: 0.05,
		SigmaGate:      3.0,
		MinDeltaVKmS:   0.01,
		SupportSamples: 3,
	}
}

/**
 * HeuristicSHAP generates a SHAP-like local attribution for a maneuver window.
 * It identifies the contribution of specific features (residual and delta-v) to the detection.
 */
func HeuristicSHAP(window []physics.ManeuverFrame) []physics.FeatureContribution {
	if len(window) == 0 {
		return nil
	}
	
	totalRes := 0.0
	totalDV := 0.0
	for _, f := range window {
		totalRes += math.Abs(f.Residual)
		totalDV += math.Abs(f.DeltaV)
	}
	
	denom := totalRes + totalDV
	if denom <= 0 {
		denom = 1.0
	}
	
	return []physics.FeatureContribution{
		{Feature: "track_residual", Contribution: totalRes / denom},
		{Feature: "estimated_dv", Contribution: totalDV / denom},
	}
}

// DetectAndCharacterizeManeuverAMOSStyle performs detection + characterization in one pass.
func DetectAndCharacterizeManeuverAMOSStyle(states []physics.State, timesSec []float64, residuals []float64, residualSigma []float64, deltaVs []float64, indianAssets []physics.State, cfg AMOSDetectorConfig) physics.ManeuverCharacterizationOutput {
	fmt.Printf("[TRC] Entering DetectAndCharacterizeManeuverAMOSStyle\n")
	defer fmt.Printf("[TRC] Exiting DetectAndCharacterizeManeuverAMOSStyle\n")

	if cfg.CUSUMDrift <= 0 {
		cfg = DefaultAMOSDetectorConfig()
	}
	if len(states) < 8 || len(timesSec) < len(states) {
		return physics.ManeuverCharacterizationOutput{}
	}
	events := DetectManeuversCovarianceAware(0, residuals, residualSigma, deltaVs, NewCUSUMDetector(cfg.CUSUMDrift, cfg.CUSUMThreshold))
	if len(events) == 0 {
		return physics.ManeuverCharacterizationOutput{}
	}
	idx := events[0].TimeIndex
	if idx < cfg.SupportSamples || idx+cfg.SupportSamples >= len(states) {
		return physics.ManeuverCharacterizationOutput{}
	}

	prior := states[maxInt(0, idx-cfg.SupportSamples):idx]
	post := states[idx+1 : minInt(len(states), idx+1+cfg.SupportSamples)]
	tPrior := timesSec[maxInt(0, idx-cfg.SupportSamples):idx]
	tPost := timesSec[idx+1 : minInt(len(timesSec), idx+1+cfg.SupportSamples)]

	char := CharacterizeManeuverFromTracks(prior, post, tPrior, tPost, indianAssets)
	dvmag := math.Sqrt(char.DeltaVECI[0]*char.DeltaVECI[0] + char.DeltaVECI[1]*char.DeltaVECI[1] + char.DeltaVECI[2]*char.DeltaVECI[2])
	if dvmag < cfg.MinDeltaVKmS {
		return physics.ManeuverCharacterizationOutput{}
	}

	rmsPrior := linearFitResidualRMS(prior)
	rmsPost := linearFitResidualRMS(post)
	conf := char.Confidence
	if rmsPost > rmsPrior*2.5 {
		conf *= 0.85
	}

	return physics.ManeuverCharacterizationOutput{
		Detected:            true,
		DetectionIndex:      idx,
		ManeuverTimeSec:     char.ManeuverTimeSec,
		DeltaVMagnitudeKmS:  dvmag,
		DeltaVRSW:           char.DeltaVRSW,
		DeltaVECI:           char.DeltaVECI,
		BurnType:            char.BurnType,
		IntentProxy:         char.IntentProxy,
		IntentProxyScore:    char.IntentProxyScore,
		CoPlanarAlign:       char.CoPlanarAlign,
		ProximityOps:        char.ProximityOps,
		StealthyManeuver:    char.StealthyManeuver,
		Confidence:          conf,
		PriorFitResidualRMS: rmsPrior,
		PostFitResidualRMS:  rmsPost,
	}
}

// linearFitResidualRMS approximates constrained least-squares fit quality over a short arc.
func linearFitResidualRMS(samples []physics.State) float64 {
	n := len(samples)
	if n < 2 {
		return 0
	}
	xData := make([]float64, n)
	yx := make([]float64, n)
	yy := make([]float64, n)
	yz := make([]float64, n)
	for i := range samples {
		xData[i] = float64(i)
		yx[i] = samples[i].Pos[0]
		yy[i] = samples[i].Pos[1]
		yz[i] = samples[i].Pos[2]
	}
	fit := func(y []float64) float64 {
		X := mat.NewDense(n, 2, nil)
		for i := 0; i < n; i++ {
			X.Set(i, 0, 1.0)
			X.Set(i, 1, xData[i])
		}
		Y := mat.NewDense(n, 1, y)
		var beta mat.Dense
		if err := beta.Solve(X, Y); err != nil {
			return 1e9
		}
		ss := 0.0
		for i := 0; i < n; i++ {
			pred := beta.At(0, 0) + beta.At(1, 0)*xData[i]
			e := y[i] - pred
			ss += e * e
		}
		return math.Sqrt(ss / float64(n))
	}
	rx := fit(yx)
	ry := fit(yy)
	rz := fit(yz)
	return (rx + ry + rz) / 3.0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
