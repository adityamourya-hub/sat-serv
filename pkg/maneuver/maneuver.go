package maneuver

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"time"

	"ssis/pkg/orbital"
)

type ManeuverType string

const (
	ManeuverNone       ManeuverType = "none"
	ManeuverImpulsive  ManeuverType = "impulsive"
	ManeuverLowThrust  ManeuverType = "low_thrust"
	ManeuverHypersonic ManeuverType = "hypersonic_like"
)

type ManeuverEvent struct {
	ObjectID  int
	TimeIndex int
	Score     float64
	DeltaV    float64
	Type      ManeuverType
	// Characterization outputs (optional) appended for post-detection depth.
	DeltaVRSW        [3]float64
	ManeuverTimeSec  float64
	IntentProxy      bool
	IntentProxyScore float64
	CoPlanarAlign    float64
	ProximityOps     bool
	StealthyManeuver bool
}

// ManeuverFrame is a model-ready temporal feature vector at a specific time step.
// The minimal feature set keeps the current API lightweight while enabling ONNX models.
type ManeuverFrame struct {
	Residual float64
	DeltaV   float64
}

type FeatureContribution struct {
	Feature      string
	Value        float64
	Contribution float64
}

type ManeuverInference struct {
	Probability float64
	Type        ManeuverType
	DeltaV      float64
	Confidence  float64
	SHAP        []FeatureContribution
	// Optional ONNX-ready scalar outputs for richer characterization.
	ManeuverDeltaVRSW [3]float64
	ManeuverTime      float64
}

type ManeuverCharacterization struct {
	ManeuverTimeSec  float64
	DeltaVECI        [3]float64
	DeltaVRSW        [3]float64
	BurnType         ManeuverType
	IntentProxy      bool
	IntentProxyScore float64
	CoPlanarAlign    float64
	ProximityOps     bool
	StealthyManeuver bool
	Confidence       float64
}

// ManeuverInferenceModel is implemented by ML backends (for example ONNX Runtime).
// Predict receives a sliding window and returns a cognitive inference payload.
type ManeuverInferenceModel interface {
	Name() string
	Predict(window []ManeuverFrame) (ManeuverInference, error)
}

// HybridManeuverDetector orchestrates model-first detection with CUSUM fallback.
// This preserves deterministic behavior even when the ML model is unavailable or uncertain.
type HybridManeuverDetector struct {
	Model               ManeuverInferenceModel
	Fallback            *CUSUMDetector
	WindowSize          int
	MinProbability      float64
	MinConfidence       float64
	SuppressAfterDetect int
}

func NewHybridManeuverDetector(model ManeuverInferenceModel) *HybridManeuverDetector {
	return &HybridManeuverDetector{
		Model:               model,
		Fallback:            NewCUSUMDetector(0.005, 0.05),
		WindowSize:          12,
		MinProbability:      0.60,
		MinConfidence:       0.50,
		SuppressAfterDetect: 2,
	}
}

type CUSUMDetector struct {
	drift     float64
	threshold float64
	posSum    float64
	negSum    float64
}

func NewCUSUMDetector(drift, threshold float64) *CUSUMDetector {
	if drift <= 0 {
		drift = 0.005
	}
	if threshold <= 0 {
		threshold = 0.05
	}
	return &CUSUMDetector{drift: drift, threshold: threshold}
}

func (d *CUSUMDetector) Update(residual float64) (float64, bool) {
	d.posSum = math.Max(0.0, d.posSum+residual-d.drift)
	d.negSum = math.Max(0.0, d.negSum-residual-d.drift)
	score := math.Max(d.posSum, d.negSum)
	return score, score >= d.threshold
}

func (d *CUSUMDetector) Reset() {
	d.posSum = 0
	d.negSum = 0
}

func DetectManeuvers(objectID int, residuals []float64, deltaVs []float64, detector *CUSUMDetector) []ManeuverEvent {
	if detector == nil {
		detector = NewCUSUMDetector(0.005, 0.05)
	}

	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}

	events := make([]ManeuverEvent, 0)
	for i := 0; i < n; i++ {
		score, alarm := detector.Update(residuals[i])
		if !alarm {
			continue
		}

		events = append(events, ManeuverEvent{
			ObjectID:  objectID,
			TimeIndex: i,
			Score:     score,
			DeltaV:    deltaVs[i],
			Type:      classifyManeuverType(deltaVs[i], residuals[i]),
		})
		detector.Reset()
	}

	return events
}

// DetectManeuversCovarianceAware normalizes residuals by per-step 1-sigma uncertainty
// before passing them to CUSUM, reducing false positives from measurement-noise drift.
func DetectManeuversCovarianceAware(objectID int, residuals []float64, residualSigma []float64, deltaVs []float64, detector *CUSUMDetector) []ManeuverEvent {
	if detector == nil {
		detector = NewCUSUMDetector(0.005, 0.05)
	}
	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}
	if len(residualSigma) < n {
		n = len(residualSigma)
	}
	if n == 0 {
		return nil
	}

	events := make([]ManeuverEvent, 0)
	for i := 0; i < n; i++ {
		sig := residualSigma[i]
		if sig <= 1e-6 {
			sig = 1e-6
		}
		normalizedResidual := residuals[i] / (3.0 * sig)
		if math.Abs(normalizedResidual) < 1.0 && math.Abs(deltaVs[i]) < 0.01 {
			// Suppress low-significance jitter so CUSUM does not drift into false alarms.
			detector.Reset()
			continue
		}
		score, alarm := detector.Update(normalizedResidual)
		if !alarm {
			continue
		}
		deltaV := deltaVs[i]
		if math.Abs(normalizedResidual) < 1.5 && deltaV < 0.015 {
			detector.Reset()
			continue
		}
		events = append(events, ManeuverEvent{
			ObjectID:        objectID,
			TimeIndex:       i,
			Score:           score,
			DeltaV:          deltaV,
			Type:            classifyManeuverType(deltaV, residuals[i]),
			DeltaVRSW:       [3]float64{0, deltaV, 0},
			ManeuverTimeSec: float64(i),
		})
		detector.Reset()
	}
	return events
}

func DetectManeuversHybrid(objectID int, residuals []float64, deltaVs []float64, hybrid *HybridManeuverDetector) []ManeuverEvent {
	if hybrid == nil {
		return DetectManeuvers(objectID, residuals, deltaVs, nil)
	}

	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}
	if n == 0 {
		return nil
	}

	if hybrid.WindowSize <= 0 {
		hybrid.WindowSize = 12
	}
	if hybrid.MinProbability <= 0 {
		hybrid.MinProbability = 0.60
	}
	if hybrid.MinConfidence <= 0 {
		hybrid.MinConfidence = 0.50
	}
	if hybrid.SuppressAfterDetect < 0 {
		hybrid.SuppressAfterDetect = 0
	}
	if hybrid.Fallback == nil {
		hybrid.Fallback = NewCUSUMDetector(0.005, 0.05)
	}

	frames := make([]ManeuverFrame, n)
	for i := 0; i < n; i++ {
		frames[i] = ManeuverFrame{Residual: residuals[i], DeltaV: deltaVs[i]}
	}

	events := make([]ManeuverEvent, 0, n/4+1)
	suppressUntil := -1

	for i := 0; i < n; i++ {
		if i <= suppressUntil {
			_, _ = hybrid.Fallback.Update(residuals[i])
			continue
		}

		usedML := false
		if hybrid.Model != nil {
			start := i - hybrid.WindowSize + 1
			if start < 0 {
				start = 0
			}
			inf, err := hybrid.Model.Predict(frames[start : i+1])
			if err == nil && inf.Probability >= hybrid.MinProbability && inf.Confidence >= hybrid.MinConfidence && inf.Type != ManeuverNone {
				score := inf.Probability * inf.Confidence
				deltaVRSW := inf.ManeuverDeltaVRSW
				if deltaVRSW == [3]float64{} {
					deltaVRSW = [3]float64{0, maxOr(inf.DeltaV, deltaVs[i]), 0}
				}
				events = append(events, ManeuverEvent{
					ObjectID:        objectID,
					TimeIndex:       i,
					Score:           score,
					DeltaV:          maxOr(inf.DeltaV, deltaVs[i]),
					Type:            inf.Type,
					DeltaVRSW:       deltaVRSW,
					ManeuverTimeSec: inf.ManeuverTime,
					IntentProxy:     false,
				})
				// Strategic Hook: Symbolic Attribution (2.3)
				if inf.Confidence >= 0.8 && (inf.Type == ManeuverImpulsive || inf.Type == ManeuverLowThrust) {
					go func(objID int, window []ManeuverFrame) {
						_ = InvokeSymbolicAttribution(objID, window)
					}(objectID, frames[start:i+1])
				}
				hybrid.Fallback.Reset()
				suppressUntil = i + hybrid.SuppressAfterDetect
				usedML = true
			}
		}

		if usedML {
			continue
		}

		score, alarm := hybrid.Fallback.Update(residuals[i])
		if !alarm {
			continue
		}

		deltaV := deltaVs[i]

		events = append(events, ManeuverEvent{
			ObjectID:        objectID,
			TimeIndex:       i,
			Score:           score,
			DeltaV:          deltaV,
			Type:            classifyManeuverType(deltaV, residuals[i]),
			DeltaVRSW:       [3]float64{0, deltaV, 0},
			ManeuverTimeSec: float64(i),
		})
		hybrid.Fallback.Reset()
		suppressUntil = i + hybrid.SuppressAfterDetect
	}

	return events
}

// CharacterizeManeuverFromTracks estimates maneuver time, burn type, and delta-v vector.
// It follows a practical variant of the PCA-style method:
//  1. fit prior and post windows using mean state,
//  2. find closest pair (PCA proxy),
//  3. compute delta-v in ECI and rotate to RSW:
//     DeltaV_RSW = R^T * (v_new - v_prior).
func CharacterizeManeuverFromTracks(
	prior []orbital.State,
	post []orbital.State,
	timesPriorSec []float64,
	timesPostSec []float64,
	indianAssets []orbital.State,
) ManeuverCharacterization {
	if len(prior) == 0 || len(post) == 0 {
		return ManeuverCharacterization{}
	}

	priorFit := meanState(prior)
	postFit := meanState(post)

	idxPrior, idxPost := pcaIndices(prior, post)
	maneuverTime := 0.5 * (safeTimeAt(timesPriorSec, idxPrior) + safeTimeAt(timesPostSec, idxPost))

	// Propagate both states to the maneuver time using small steps for RK4 stability
	baseJd := float64(time.Now().Unix())/86400.0 + 2440587.5
	prop := func(s orbital.State, dt float64) orbital.State {
		if math.Abs(dt) < 1.0 {
			return orbital.VectorToState(orbital.PropagateCartesianState(orbital.StateToVector(s), dt, baseJd))
		}
		const maxStep = 30.0
		n := int(math.Ceil(math.Abs(dt) / maxStep))
		step := dt / float64(n)
		v := orbital.StateToVector(s)
		for i := 0; i < n; i++ {
			v = orbital.PropagateCartesianState(v, step, baseJd)
		}
		return orbital.VectorToState(v)
	}

	priorAtT := prop(prior[idxPrior], maneuverTime-safeTimeAt(timesPriorSec, idxPrior))
	postAtT := prop(post[idxPost], maneuverTime-safeTimeAt(timesPostSec, idxPost))

	dvECI := [3]float64{postAtT.VX - priorAtT.VX, postAtT.VY - priorAtT.VY, postAtT.VZ - priorAtT.VZ}
	dvRSW := eciDeltaVToRSW(dvECI, priorAtT)
	burnType := classifyManeuverType(norm3(dvECI), distance3(postAtT, priorAtT))

	intentScore := alignmentIntentProxy(postFit, indianAssets)
	priorPostGap := distance3(priorFit, postFit)
	
	// Enhanced Intent Proxies
	proxOps := false
	for _, a := range indianAssets {
		if distance3(postFit, a) < 150.0 { // 150km threshold for prox-ops suspicion
			proxOps = true
			break
		}
	}
	
	dvMag := norm3(dvRSW)
	stealthy := false
	if dvMag > 0 && dvMag < 0.05 && priorPostGap > 5.0 {
		stealthy = true // Small Delta-V, significant trajectory shift
	}

	confidence := clamp01(0.50 + 0.30*clamp01(dvMag/0.1) + 0.10*intentScore + 0.10*clamp01(priorPostGap/5.0))

	return ManeuverCharacterization{
		ManeuverTimeSec:  maneuverTime,
		DeltaVECI:        dvECI,
		DeltaVRSW:        dvRSW,
		BurnType:         burnType,
		IntentProxy:      intentScore >= 0.7 || proxOps,
		IntentProxyScore: intentScore,
		CoPlanarAlign:    intentScore,
		ProximityOps:     proxOps,
		StealthyManeuver: stealthy,
		Confidence:       confidence,
	}
}

func pcaIndices(prior []orbital.State, post []orbital.State) (int, int) {
	bestI, bestJ := 0, 0
	bestD := math.MaxFloat64
	for i := range prior {
		for j := range post {
			d := distance3(prior[i], post[j])
			if d < bestD {
				bestD = d
				bestI, bestJ = i, j
			}
		}
	}
	return bestI, bestJ
}

func meanState(states []orbital.State) orbital.State {
	if len(states) == 0 {
		return orbital.State{}
	}
	var s orbital.State
	for _, st := range states {
		s.X += st.X
		s.Y += st.Y
		s.Z += st.Z
		s.VX += st.VX
		s.VY += st.VY
		s.VZ += st.VZ
	}
	n := float64(len(states))
	s.X /= n
	s.Y /= n
	s.Z /= n
	s.VX /= n
	s.VY /= n
	s.VZ /= n
	return s
}

func eciDeltaVToRSW(deltaVECI [3]float64, reference orbital.State) [3]float64 {
	r := [3]float64{reference.X, reference.Y, reference.Z}
	v := [3]float64{reference.VX, reference.VY, reference.VZ}

	rHat := normalize3(r)
	h := cross3(r, v)
	wHat := normalize3(h)
	sHat := cross3(wHat, rHat)

	return [3]float64{
		dot3(deltaVECI, rHat),
		dot3(deltaVECI, sHat),
		dot3(deltaVECI, wHat),
	}
}

func alignmentIntentProxy(subject orbital.State, indianAssets []orbital.State) float64 {
	if len(indianAssets) == 0 {
		return 0
	}
	subjPlane := normalize3(cross3([3]float64{subject.X, subject.Y, subject.Z}, [3]float64{subject.VX, subject.VY, subject.VZ}))
	best := 0.0
	for _, a := range indianAssets {
		assetPlane := normalize3(cross3([3]float64{a.X, a.Y, a.Z}, [3]float64{a.VX, a.VY, a.VZ}))
		align := math.Abs(dot3(subjPlane, assetPlane))
		if align > best {
			best = align
		}
	}
	return clamp01(best)
}

func safeTimeAt(times []float64, idx int) float64 {
	if idx < 0 || idx >= len(times) {
		return float64(idx)
	}
	return times[idx]
}

func distance3(a, b orbital.State) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	dz := a.Z - b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func norm3(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func dot3(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func normalize3(v [3]float64) [3]float64 {
	n := norm3(v)
	if n <= 1e-12 {
		return [3]float64{}
	}
	return [3]float64{v[0] / n, v[1] / n, v[2] / n}
}

func HeuristicSHAP(window []ManeuverFrame) []FeatureContribution {
	if len(window) == 0 {
		return nil
	}

	var sumResidual float64
	var sumDeltaV float64
	for _, f := range window {
		sumResidual += math.Abs(f.Residual)
		sumDeltaV += f.DeltaV
	}

	meanResidual := sumResidual / float64(len(window))
	meanDeltaV := sumDeltaV / float64(len(window))
	den := meanResidual + math.Abs(meanDeltaV)
	if den <= 1e-9 {
		den = 1.0
	}

	return []FeatureContribution{
		{Feature: "residual_mean_abs", Value: meanResidual, Contribution: meanResidual / den},
		{Feature: "delta_v_mean", Value: meanDeltaV, Contribution: math.Abs(meanDeltaV) / den},
	}
}

func classifyManeuverType(deltaV, residual float64) ManeuverType {
	absResidual := math.Abs(residual)
	switch {
	case deltaV >= 0.08:
		return ManeuverImpulsive
	case deltaV >= 0.015:
		return ManeuverLowThrust
	case absResidual >= 0.25:
		return ManeuverHypersonic
	default:
		return ManeuverNone
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func maxOr(a, b float64) float64 {
	if a > 0 {
		return a
	}
	return b
}

// InvokeSymbolicAttribution bridges the Go detection to the Python Axiom-Zero distillers.
func InvokeSymbolicAttribution(objectID int, window []ManeuverFrame) error {
	fmt.Printf("IAF_SDA: Triggering Symbolic Attribution for Asset %d...\n", objectID)
	
	// 1. Prepare Data Shell
	type Payload struct {
		ObjectID int             `json:"object_id"`
		Q        [][3]float64    `json:"q"`
		Res      [][]float64     `json:"res"`
	}

	payload := Payload{
		ObjectID: objectID,
		Q:        make([][3]float64, len(window)),
		Res:      make([][]float64, len(window)),
	}

	for i, f := range window {
		payload.Q[i] = [3]float64{0, 0, float64(i) * 100.0} 
		payload.Res[i] = []float64{f.Residual}
	}

	// 2. Serialize to Temporary File
	tmpFile := fmt.Sprintf("maneuver_res_%d.json", objectID)
	data, _ := json.Marshal(payload)
	_ = os.WriteFile(tmpFile, data, 0644)
	defer os.Remove(tmpFile)

	// 3. Execute Axiom-Zero symbolic pipeline
	// Standardized path for consolidated kernels (Phase 1.1)
	cmd := exec.Command(".venv/bin/python3", "ai/v3/symbolic_attributor.py", 
		"--residuals", tmpFile, 
		"--out", fmt.Sprintf("attribution_%d.tex", objectID))
	
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}
