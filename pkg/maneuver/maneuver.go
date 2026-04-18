package maneuver

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"time"

	"ssis/pkg/physics"
)

/**
 * ManeuverInferenceModel is implemented by ML backends (for example ONNX Runtime).
 * Predict receives a sliding window and returns a cognitive inference payload.
 */
type ManeuverInferenceModel interface {
	Name() string
	Predict(window []physics.ManeuverFrame) (physics.ManeuverInference, error)
}

/**
 * HybridManeuverDetector orchestrates model-first detection with CUSUM fallback.
 * This preserves deterministic behavior even when the ML model is unavailable or uncertain.
 */
type HybridManeuverDetector struct {
	Model               ManeuverInferenceModel
	Fallback            *CUSUMDetector
	WindowSize          int
	MinProbability      float64
	MinConfidence       float64
	SuppressAfterDetect int
}

/**
 * NewHybridManeuverDetector initializes a detector with a specific ML model and default fallback settings.
 */
func NewHybridManeuverDetector(model ManeuverInferenceModel) *HybridManeuverDetector {
	fmt.Printf("[TRC] Entering NewHybridManeuverDetector\n")
	defer fmt.Printf("[TRC] Exiting NewHybridManeuverDetector\n")
	return &HybridManeuverDetector{
		Model:               model,
		Fallback:            NewCUSUMDetector(0.005, 0.05),
		WindowSize:          12,
		MinProbability:      0.60,
		MinConfidence:       0.50,
		SuppressAfterDetect: 2,
	}
}

/**
 * InputRegime specifies the quality level of tracking data for threshold tuning.
 */
type InputRegime int

const (
	RegimeTLE InputRegime = iota // Two-Line Elements: km-level position errors
	RegimeSP                     // Special Perturbations: m-level position errors (radar/optical OD)
	RegimeGP                     // General Perturbations with covariance
)

/**
 * CUSUMDetector implements the Cumulative Sum algorithm for sequential change detection.
 */
type CUSUMDetector struct {
	drift     float64
	threshold float64
	posSum    float64
	negSum    float64
	regime    InputRegime // Added for regime-aware thresholds (ADITI 4.0)
}

/**
 * NewCUSUMDetector creates a statistical change detector with specified drift and threshold parameters.
 *
 * WARNING: For TLE data, use NewCUSUMDetectorForRegime(RegimeTLE) instead.
 * The default thresholds (drift=0.005, threshold=0.05) are appropriate for SP-quality
 * data (σ_position ~ 10m), but will produce excessive false positives with TLE data
 * (σ_position ~ 1km).
 */
func NewCUSUMDetector(drift, threshold float64) *CUSUMDetector {
	if drift <= 0 {
		drift = 0.005
	}
	if threshold <= 0 {
		threshold = 0.05
	}
	return &CUSUMDetector{drift: drift, threshold: threshold, regime: RegimeSP}
}

/**
 * NewCUSUMDetectorForRegime creates a CUSUM detector with regime-appropriate thresholds.
 *
 * ADITI 4.0: Calibrated thresholds for different input data quality levels.
 *
 * TLE Input (σ_pos ~ 1km, σ_vel ~ 1m/s):
 *   - drift = 0.5 (100× less sensitive than SP)
 *   - threshold = 5.0 (100× higher than SP)
 *   - Rationale: TLE residual noise is 100× larger than SP, thresholds must scale accordingly
 *
 * SP Input (σ_pos ~ 10m, σ_vel ~ 0.01m/s):
 *   - drift = 0.005
 *   - threshold = 0.05
 *   - Rationale: Standard values for radar-quality OD
 *
 * Minimum Detectable ΔV:
 *   - TLE: ~3 m/s (limited by TLE noise floor)
 *   - SP:  ~0.03 m/s (enables GEO station-keeping detection at ~0.1 m/burn)
 */
func NewCUSUMDetectorForRegime(regime InputRegime) *CUSUMDetector {
	switch regime {
	case RegimeTLE:
		// TLE: km-level errors require much less sensitive thresholds
		// Without this, false positive rate ~10-50% per orbit
		return &CUSUMDetector{
			drift:     0.5, // 100× less sensitive
			threshold: 5.0, // 100× higher threshold
			regime:    regime,
		}
	case RegimeSP, RegimeGP:
		// SP/GP: m-level errors with covariance - standard thresholds
		return &CUSUMDetector{
			drift:     0.005,
			threshold: 0.05,
			regime:    regime,
		}
	default:
		// Conservative default for unknown data quality
		return &CUSUMDetector{
			drift:     0.05,
			threshold: 0.5,
			regime:    RegimeTLE, // Assume worst case
		}
	}
}

/**
 * Update steps the CUSUM logic with a new residual measurement.
 */
func (d *CUSUMDetector) Update(residual float64) (float64, bool) {
	d.posSum = math.Max(0.0, d.posSum+residual-d.drift)
	d.negSum = math.Max(0.0, d.negSum-residual-d.drift)
	score := math.Max(d.posSum, d.negSum)
	return score, score >= d.threshold
}

/**
 * Reset clears the internal sums to stop sustained firing or restart after a detection.
 */
func (d *CUSUMDetector) Reset() {
	d.posSum = 0
	d.negSum = 0
}

/**
 * DetectManeuvers performs batch CUSUM detection on a sequence of state residuals.
 */
func DetectManeuvers(objectID int, residuals []float64, deltaVs []float64, detector *CUSUMDetector) []physics.ManeuverEvent {
	fmt.Printf("[TRC] Entering DetectManeuvers for object %d\n", objectID)
	defer fmt.Printf("[TRC] Exiting DetectManeuvers\n")

	if detector == nil {
		detector = NewCUSUMDetector(0.005, 0.05)
	}

	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}

	events := make([]physics.ManeuverEvent, 0)
	for i := 0; i < n; i++ {
		score, alarm := detector.Update(residuals[i])
		if !alarm {
			continue
		}

		events = append(events, physics.ManeuverEvent{
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

/**
 * DetectManeuversCovarianceAware normalizes residuals by uncertainty to reduce false-positives in high-noise regimes.
 */
func DetectManeuversCovarianceAware(objectID int, residuals []float64, residualSigma []float64, deltaVs []float64, detector *CUSUMDetector) []physics.ManeuverEvent {
	fmt.Printf("[TRC] Entering DetectManeuversCovarianceAware for object %d\n", objectID)
	defer fmt.Printf("[TRC] Exiting DetectManeuversCovarianceAware\n")

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

	events := make([]physics.ManeuverEvent, 0)
	for i := 0; i < n; i++ {
		sig := residualSigma[i]
		if sig <= 1e-6 {
			sig = 1e-6
		}
		normalizedResidual := residuals[i] / (3.0 * sig)
		if math.Abs(normalizedResidual) < 1.0 && math.Abs(deltaVs[i]) < 0.01 {
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
		events = append(events, physics.ManeuverEvent{
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

/**
 * DetectManeuversHybrid dispatches to either the ML model or the statistical fallback.
 */
func DetectManeuversHybrid(objectID int, residuals []float64, deltaVs []float64, hybrid *HybridManeuverDetector) []physics.ManeuverEvent {
	fmt.Printf("[TRC] Entering DetectManeuversHybrid for object %d\n", objectID)
	defer fmt.Printf("[TRC] Exiting DetectManeuversHybrid\n")

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
	if hybrid.Fallback == nil {
		hybrid.Fallback = NewCUSUMDetector(0.005, 0.05)
	}

	frames := make([]physics.ManeuverFrame, n)
	for i := 0; i < n; i++ {
		frames[i] = physics.ManeuverFrame{Residual: residuals[i], DeltaV: deltaVs[i]}
	}

	events := make([]physics.ManeuverEvent, 0, n/4+1)
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
			if err == nil && inf.Probability >= hybrid.MinProbability && inf.Confidence >= hybrid.MinConfidence && inf.Type != physics.ManeuverNone {
				score := inf.Probability * inf.Confidence
				deltaVRSW := inf.ManeuverDeltaVRSW
				if deltaVRSW == [3]float64{} {
					deltaVRSW = [3]float64{0, maxOr(inf.DeltaV, deltaVs[i]), 0}
				}
				events = append(events, physics.ManeuverEvent{
					ObjectID:        objectID,
					TimeIndex:       i,
					Score:           score,
					DeltaV:          maxOr(inf.DeltaV, deltaVs[i]),
					Type:            inf.Type,
					DeltaVRSW:       deltaVRSW,
					ManeuverTimeSec: inf.ManeuverTime,
				})
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
		events = append(events, physics.ManeuverEvent{
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

/**
 * CharacterizeManeuverFromTracks performs differential state analysis between two tracking windows.
 */
func CharacterizeManeuverFromTracks(
	prior []physics.State,
	post []physics.State,
	timesPriorSec []float64,
	timesPostSec []float64,
	indianAssets []physics.State,
) physics.ManeuverCharacterization {
	fmt.Printf("[TRC] Entering CharacterizeManeuverFromTracks\n")
	defer fmt.Printf("[TRC] Exiting CharacterizeManeuverFromTracks\n")

	if len(prior) == 0 || len(post) == 0 {
		return physics.ManeuverCharacterization{}
	}

	priorFit := meanState(prior)
	postFit := meanState(post)

	idxPrior, idxPost := pcaIndices(prior, post)
	maneuverTime := 0.5 * (safeTimeAt(timesPriorSec, idxPrior) + safeTimeAt(timesPostSec, idxPost))

	baseJd := float64(time.Now().Unix())/86400.0 + 2440587.5
	prop := func(s physics.State, dt float64) physics.State {
		xv := physics.StateToVector(s)
		nextV := physics.PropagateCartesianState(xv, dt, baseJd, physics.PhysicsParams{})
		return physics.VectorToState(nextV)
	}

	priorAtT := prop(prior[idxPrior], maneuverTime-safeTimeAt(timesPriorSec, idxPrior))
	postAtT := prop(post[idxPost], maneuverTime-safeTimeAt(timesPostSec, idxPost))

	dvECI := [3]float64{postAtT.Vel[0] - priorAtT.Vel[0], postAtT.Vel[1] - priorAtT.Vel[1], postAtT.Vel[2] - priorAtT.Vel[2]}
	dvRSW := eciDeltaVToRSW(dvECI, priorAtT)
	burnType := classifyManeuverType(norm3(dvECI), distance3(postAtT, priorAtT))

	intentScore := alignmentIntentProxy(postFit, indianAssets)
	priorPostGap := distance3(priorFit, postFit)

	proxOps := false
	for _, a := range indianAssets {
		if distance3(postFit, a) < 150.0 {
			proxOps = true
			break
		}
	}

	dvMag := norm3(dvRSW)
	stealthy := dvMag > 0 && dvMag < 0.05 && priorPostGap > 5.0
	confidence := clamp01(0.50 + 0.30*clamp01(dvMag/0.1) + 0.10*intentScore + 1.0*clamp01(priorPostGap/5.0))

	return physics.ManeuverCharacterization{
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

/**
 * pcaIndices finds the closest pair of states in two temporal windows as a proxy for the maneuver point.
 */
func pcaIndices(prior []physics.State, post []physics.State) (int, int) {
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

/**
 * meanState calculates the centroid of a state sequence in cartesian space.
 */
func meanState(states []physics.State) physics.State {
	if len(states) == 0 {
		return physics.State{}
	}
	var s physics.State
	for _, st := range states {
		s.Pos[0] += st.Pos[0]
		s.Pos[1] += st.Pos[1]
		s.Pos[2] += st.Pos[2]
		s.Vel[0] += st.Vel[0]
		s.Vel[1] += st.Vel[1]
		s.Vel[2] += st.Vel[2]
	}
	n := float64(len(states))
	s.Pos[0] /= n
	s.Pos[1] /= n
	s.Pos[2] /= n
	s.Vel[0] /= n
	s.Vel[1] /= n
	s.Vel[2] /= n
	return s
}

/**
 * eciDeltaVToRSW rotates a velocity residual from GCRF to the local Radial-South-West frame.
 */
func eciDeltaVToRSW(deltaVECI [3]float64, reference physics.State) [3]float64 {
	r := [3]float64{reference.Pos[0], reference.Pos[1], reference.Pos[2]}
	v := [3]float64{reference.Vel[0], reference.Vel[1], reference.Vel[2]}

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

/**
 * alignmentIntentProxy scores the planar alignment of a target vs defensive assets.
 */
func alignmentIntentProxy(subject physics.State, indianAssets []physics.State) float64 {
	if len(indianAssets) == 0 {
		return 0
	}
	subjPlane := normalize3(cross3([3]float64{subject.Pos[0], subject.Pos[1], subject.Pos[2]}, [3]float64{subject.Vel[0], subject.Vel[1], subject.Vel[2]}))
	best := 0.0
	for _, a := range indianAssets {
		assetPlane := normalize3(cross3([3]float64{a.Pos[0], a.Pos[1], a.Pos[2]}, [3]float64{a.Vel[0], a.Vel[1], a.Vel[2]}))
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

/**
 * distance3 computes the Euclidean distance between two cartesian states.
 */
func distance3(a, b physics.State) float64 {
	dx := a.Pos[0] - b.Pos[0]
	dy := a.Pos[1] - b.Pos[1]
	dz := a.Pos[2] - b.Pos[2]
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

func classifyManeuverType(deltaV, residual float64) physics.ManeuverType {
	absResidual := math.Abs(residual)
	switch {
	case deltaV >= 0.08:
		return physics.ManeuverImpulsive
	case deltaV >= 0.015:
		return physics.ManeuverLowThrust
	case absResidual >= 0.25:
		return physics.ManeuverHypersonic
	default:
		return physics.ManeuverNone
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

/**
 * InvokeSymbolicAttribution bridges the Go detection to the Python Axiom-Zero distillers for force-decomposition.
 */
func InvokeSymbolicAttribution(objectID int, window []physics.ManeuverFrame) error {
	fmt.Printf("[TRC] Entering InvokeSymbolicAttribution for object %d\n", objectID)
	defer fmt.Printf("[TRC] Exiting InvokeSymbolicAttribution\n")

	type Payload struct {
		ObjectID int          `json:"object_id"`
		Q        [][3]float64 `json:"q"`
		Res      [][]float64  `json:"res"`
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

	tmpFile := fmt.Sprintf("maneuver_res_%d.json", objectID)
	data, _ := json.Marshal(payload)
	_ = os.WriteFile(tmpFile, data, 0644)
	defer os.Remove(tmpFile)

	cmd := exec.Command(".venv/bin/python3", "ai/v3/symbolic_attributor.py",
		"--residuals", tmpFile,
		"--out", fmt.Sprintf("attribution_%d.tex", objectID))

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}
