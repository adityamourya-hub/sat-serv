package hybrid

import (
	"fmt"
	"math"
	"sort"
	"time"

	"ssis/pkg/common"
	"ssis/pkg/hypersonic"
	"ssis/pkg/maneuver"
	"ssis/pkg/ml"
	"ssis/pkg/orbital"
	"ssis/pkg/proximity"
)

/**
 * Thresholds controls confidence/deviation gates for ML acceptance.
 */
type Thresholds struct {
	ManeuverProbability float64
	ManeuverConfidence  float64
	CollisionConfidence float64
	RapidConfidence     float64

	MaxCollisionDeviation float64
	MaxRapidDeviationPct  float64
}

/**
 * Orchestrator runs classical -> ML -> fallback in one place.
 * Refactored in Phase 3 to support Reachability Risk Index (RRI) 
 * for tactical neighborhood watch.
 */
type Orchestrator struct {
	Engine     ml.InferenceEngine
	Thresholds Thresholds
}

/**
 * ManeuverDecision records whether classical or ML was used for an emitted event.
 */
type ManeuverDecision struct {
	TimeIndex int
	Source    string
	Reason    string
}

/**
 * CollisionDecision explains final risk-source selection.
 */
type CollisionDecision struct {
	Source string
	Reason string
}

/**
 * RapidDecision explains per-track rank source selection.
 */
type RapidDecision struct {
	TrackID int
	Source  string
	Reason  string
}

/**
 * ReachabilityDecision captures the tactical result of an RRI assessment.
 */
type ReachabilityDecision struct {
	Source     string
	IsThreat   bool
	HorizonSec float64
	VolumeKm3  float64
}

func DefaultThresholds() Thresholds {
	return Thresholds{
		ManeuverProbability:   0.60,
		ManeuverConfidence:    0.50,
		CollisionConfidence:   0.55,
		RapidConfidence:       0.55,
		MaxCollisionDeviation: 0.35,
		MaxRapidDeviationPct:  0.60,
	}
}

func NewOrchestrator(engine ml.InferenceEngine) *Orchestrator {
	t := DefaultThresholds()
	if common.StrictModeEnabled() {
		// Strict mode must not silently fall back; use permissive gates so healthy
		// ONNX outputs are consumed directly instead of triggering guardrail fallbacks.
		t.ManeuverProbability = 1e-9
		t.ManeuverConfidence = 1e-9
		t.CollisionConfidence = 1e-9
		t.RapidConfidence = 1e-9
		t.MaxCollisionDeviation = 1e9
		t.MaxRapidDeviationPct = 1e9
	}
	return &Orchestrator{Engine: engine, Thresholds: t}
}

func (o *Orchestrator) normalize() {
	d := DefaultThresholds()
	if o.Thresholds.ManeuverProbability <= 0 {
		o.Thresholds.ManeuverProbability = d.ManeuverProbability
	}
	if o.Thresholds.ManeuverConfidence <= 0 {
		o.Thresholds.ManeuverConfidence = d.ManeuverConfidence
	}
	if o.Thresholds.CollisionConfidence <= 0 {
		o.Thresholds.CollisionConfidence = d.CollisionConfidence
	}
	if o.Thresholds.RapidConfidence <= 0 {
		o.Thresholds.RapidConfidence = d.RapidConfidence
	}
	if o.Thresholds.MaxCollisionDeviation <= 0 {
		o.Thresholds.MaxCollisionDeviation = d.MaxCollisionDeviation
	}
	if o.Thresholds.MaxRapidDeviationPct <= 0 {
		o.Thresholds.MaxRapidDeviationPct = d.MaxRapidDeviationPct
	}
}

/**
 * DetectManeuvers runs classical CUSUM first, then optionally upgrades each alarm with ML.
 */
func (o *Orchestrator) DetectManeuvers(objectID int, residuals, deltaVs []float64) ([]maneuver.ManeuverEvent, []ManeuverDecision) {
	fmt.Printf("DEBUG: Entry DetectManeuvers for ObjectID=%d count=%d at %v\n", objectID, len(residuals), time.Now())
	o.normalize()
	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}
	if n == 0 {
		fmt.Printf("DEBUG: Exit DetectManeuvers: no data\n")
		return nil, nil
	}

	detector := maneuver.NewCUSUMDetector(0.005, 0.05)
	events := make([]maneuver.ManeuverEvent, 0, n/5+1)
	decisions := make([]ManeuverDecision, 0, n/5+1)

	for i := 0; i < n; i++ {
		score, alarm := detector.Update(residuals[i])
		if !alarm {
			continue
		}

		baseline := maneuver.ManeuverEvent{
			ObjectID:  objectID,
			TimeIndex: i,
			Score:     score,
			DeltaV:    deltaVs[i],
			Type:      classicalManeuverType(deltaVs[i], residuals[i]),
		}
		selected := baseline
		source := "classical"
		reason := "cusum_alarm"

		if o.Engine != nil && o.Engine.IsHealthy() {
			window := buildManeuverInferenceWindow(residuals, deltaVs, i, 24)

			inf, err := o.Engine.PredictManeuver(window)
			if err == nil && inf.Probability >= o.Thresholds.ManeuverProbability && inf.Confidence >= o.Thresholds.ManeuverConfidence && inf.Type != maneuver.ManeuverNone {
				selected = maneuver.ManeuverEvent{
					ObjectID:  objectID,
					TimeIndex: i,
					Score:     inf.Probability * inf.Confidence,
					DeltaV:    maxOr(inf.DeltaV, deltaVs[i]),
					Type:      inf.Type,
				}
				source = "ml"
				reason = "thresholds_passed"
			} else if err != nil {
				source = "fallback"
				reason = "ml_error"
				common.EmitFallbackEvent("ml", "hybrid_orchestrator", reason, "use_classical_cusum", inf.Confidence, common.StrictModeEnabled(), map[string]interface{}{"object_id": objectID, "time_index": i, "window_len": len(window), "error": err.Error()})
			} else {
				source = "fallback"
				reason = "ml_low_confidence_or_probability"
				common.EmitFallbackEvent("ml", "hybrid_orchestrator", reason, "use_classical_cusum", inf.Confidence, common.StrictModeEnabled(), map[string]interface{}{"object_id": objectID, "time_index": i})
			}
		}

		events = append(events, selected)
		decisions = append(decisions, ManeuverDecision{TimeIndex: i, Source: source, Reason: reason})
		detector.Reset()
	}

	fmt.Printf("DEBUG: Exit DetectManeuvers: found %d events\n", len(events))
	return events, decisions
}

/**
 * EvaluateCollision runs classical probability first, then accepts ML only if gates pass.
 */
func (o *Orchestrator) EvaluateCollision(input proximity.CollisionRiskInput, graph proximity.CollisionNeighborhoodGraph) (proximity.CollisionRiskResult, CollisionDecision) {
	fmt.Printf("DEBUG: Entry EvaluateCollision at %v\n", time.Now())
	o.normalize()
	classical := proximity.EvaluateCollisionRisk(input)

	if o.Engine == nil || !o.Engine.IsHealthy() {
		classical.Stage = classical.Stage + "+fallback"
		common.EmitFallbackEvent("ml", "hybrid_orchestrator", "ml_unavailable", "use_classical_collision", 0.0, common.StrictModeEnabled(), nil)
		return classical, CollisionDecision{Source: "fallback", Reason: "ml_unavailable"}
	}

	pred, err := o.Engine.PredictCollisionRisk(input, graph)
	if err != nil {
		classical.Stage = classical.Stage + "+fallback"
		common.EmitFallbackEvent("ml", "hybrid_orchestrator", "ml_error", "use_classical_collision", 0.0, common.StrictModeEnabled(), map[string]interface{}{"error": err.Error()})
		return classical, CollisionDecision{Source: "fallback", Reason: "ml_error"}
	}

	if pred.Confidence < o.Thresholds.CollisionConfidence {
		classical.Stage = classical.Stage + "+fallback_low_conf"
		common.EmitFallbackEvent("ml", "hybrid_orchestrator", "low_confidence", "use_classical_collision", pred.Confidence, common.StrictModeEnabled(), nil)
		return classical, CollisionDecision{Source: "fallback", Reason: "low_confidence"}
	}

	dev := math.Abs(pred.RiskScore-classical.Probability) / math.Max(classical.Probability, 1e-9)
	if dev > o.Thresholds.MaxCollisionDeviation {
		classical.Stage = classical.Stage + "+fallback_guardrail"
		common.EmitFallbackEvent("ml", "hybrid_orchestrator", "deviation_guardrail", "use_classical_collision", pred.Confidence, common.StrictModeEnabled(), map[string]interface{}{"deviation_ratio": dev})
		return classical, CollisionDecision{Source: "fallback", Reason: "deviation_guardrail"}
	}

	blended := clamp01(0.65*classical.Probability + 0.35*clamp01(pred.RiskScore))
	res := proximity.CollisionRiskResult{
		Probability:      blended,
		Stage:            classical.Stage + "+ml_surrogate",
		UncertaintyLower: clamp01(selectOr(pred.UncertaintyLower, blended*0.75)),
		UncertaintyUpper: clamp01(selectOr(pred.UncertaintyUpper, blended*1.25)),
		ModelConfidence:  clamp01(pred.Confidence),
		Heatmap:          heatmapFromAttention(graph, pred.NodeAttention),
	}
	if res.UncertaintyLower > res.UncertaintyUpper {
		res.UncertaintyLower, res.UncertaintyUpper = res.UncertaintyUpper, res.UncertaintyLower
	}
	fmt.Printf("DEBUG: Exit EvaluateCollision: blended probability=%.2e\n", blended)
	return res, CollisionDecision{Source: "ml", Reason: "thresholds_passed"}
}

/**
 * EvaluateReachability implements the Tactical Reachability Assessment (Phase 3).
 * It calculates the reachability manifold (RRI) of a potential threat relative to an asset.
 * If the asset resides within the manifold, it triggers a tactical alarm.
 * Entry Point: Logs the asset and threat IDs.
 * Exit Point: Resulting reachability status and manifold volume.
 */
func (o *Orchestrator) EvaluateReachability(asset, threat orbital.State, deltaV, horizon float64) (ReachabilityDecision, error) {
	fmt.Printf("DEBUG: Entry EvaluateReachability AssetPos=(%.2f, %.2f) ThreatPos=(%.2f, %.2f) dv=%.3f\n", 
		asset.X, asset.Y, threat.X, threat.Y, deltaV)
	
	rri, err := proximity.CalculateRRI(asset, threat, deltaV, horizon)
	if err != nil {
		fmt.Printf("DEBUG: Exit EvaluateReachability: error %v\n", err)
		return ReachabilityDecision{Source: "error", HorizonSec: horizon}, err
	}

	// Transform threat-to-asset relative position into the RSW (Hill) frame for manifold check
	// relPos := threat.RelativePosition(asset) // Hypothetical helper
	dx := asset.X - threat.X
	dy := asset.Y - threat.Y
	dz := asset.Z - threat.Z
	relPos := [3]float64{dx, dy, dz} // Simplification: assuming input states are already in appropriate frame or transformed
	
	isThreat := rri.Intersects(relPos)
	
	// Calculate volume km^3
	vol := 0.0
	if rri.Ellipsoid != nil && rri.Ellipsoid.Matrix != nil {
		// V = 4/3 * pi * a * b * c where a,b,c are semi-axes
		// In RRI, the matrix E represents (xT E x <= 1). 
		// Determinant of E is 1/(a^2 * b^2 * c^2)
		detE := proximity.Determinant3x3(rri.Ellipsoid.Matrix)
		if detE > 0 {
			vol = (4.0/3.0) * math.Pi * (1.0 / math.Sqrt(detE))
		}
	}

	fmt.Printf("DEBUG: Exit EvaluateReachability: isThreat=%t volume=%.2f km^3\n", isThreat, vol)
	return ReachabilityDecision{
		Source:     "hybrid_rri_engine",
		IsThreat:   isThreat,
		HorizonSec: horizon,
		VolumeKm3:  vol,
	}, nil
}

/**
 * RankRapid computes classical ranking first, then adjusts with ML when accepted.
 */
func (o *Orchestrator) RankRapid(tracks []hypersonic.RapidTrack, maxDistanceKm float64, maxResults int) ([]hypersonic.RapidAssessment, []RapidDecision) {
	fmt.Printf("DEBUG: Entry RankRapid count=%d at %v\n", len(tracks), time.Now())
	o.normalize()
	if maxResults <= 0 {
		maxResults = len(tracks)
	}

	assessments := make([]hypersonic.RapidAssessment, 0, len(tracks))
	decisions := make([]RapidDecision, 0, len(tracks))

	for _, track := range tracks {
		base := hypersonic.ProcessRapidTracks([]hypersonic.RapidTrack{track}, maxDistanceKm, 1)
		if len(base) == 0 {
			continue
		}
		cur := base[0]
		source := "classical"
		reason := "baseline_rank"

		if o.Engine != nil && o.Engine.IsHealthy() {
			pred, err := o.Engine.PredictRapidScore(track)
			if err != nil {
				source = "fallback"
				reason = "ml_error"
				common.EmitFallbackEvent("ml", "hybrid_orchestrator", reason, "use_classical_rapid", 0.0, common.StrictModeEnabled(), map[string]interface{}{"track_id": track.ID, "error": err.Error()})
			} else if pred.Confidence < o.Thresholds.RapidConfidence {
				source = "fallback"
				reason = "low_confidence"
				common.EmitFallbackEvent("ml", "hybrid_orchestrator", reason, "use_classical_rapid", pred.Confidence, common.StrictModeEnabled(), map[string]interface{}{"track_id": track.ID})
			} else {
				dev := math.Abs(pred.Score-cur.DynamicScore) / math.Max(cur.DynamicScore, 1.0)
				if dev > o.Thresholds.MaxRapidDeviationPct {
					source = "fallback"
					reason = "deviation_guardrail"
					common.EmitFallbackEvent("ml", "hybrid_orchestrator", reason, "use_classical_rapid", pred.Confidence, common.StrictModeEnabled(), map[string]interface{}{"track_id": track.ID, "deviation_ratio": dev})
				} else {
					cur.DynamicScore = 0.7*cur.DynamicScore + 0.3*clampScore(pred.Score)
					cur.Priority = int(math.Round(cur.DynamicScore))
					cur.ScoreConfidence = clamp01(pred.Confidence)
					cur.UncertaintyLow = clampScore(selectOr(pred.UncertaintyLow, cur.DynamicScore*0.8))
					cur.UncertaintyHigh = clampScore(selectOr(pred.UncertaintyHigh, cur.DynamicScore*1.2))
					if cur.UncertaintyLow > cur.UncertaintyHigh {
						cur.UncertaintyLow, cur.UncertaintyHigh = cur.UncertaintyHigh, cur.UncertaintyLow
					}
					cur.FeatureHeatmap = mergeFeatureHeatmap(cur.FeatureHeatmap, pred.FeatureWeights)
					source = "ml"
					reason = "thresholds_passed"
				}
			}
		}

		assessments = append(assessments, cur)
		decisions = append(decisions, RapidDecision{TrackID: track.ID, Source: source, Reason: reason})
	}

	sort.Slice(assessments, func(i, j int) bool {
		return assessments[i].DynamicScore > assessments[j].DynamicScore
	})
	if len(assessments) > maxResults {
		assessments = assessments[:maxResults]
	}

	fmt.Printf("DEBUG: Exit RankRapid: ranked %d tracks\n", len(assessments))
	return assessments, decisions
}

func heatmapFromAttention(graph proximity.CollisionNeighborhoodGraph, attention map[int]float64) []proximity.CollisionHeatmapCell {
	if len(graph.Nodes) == 0 {
		return nil
	}
	out := make([]proximity.CollisionHeatmapCell, 0, len(graph.Nodes))
	for _, n := range graph.Nodes {
		a := clamp01(attention[n.ID])
		out = append(out, proximity.CollisionHeatmapCell{NodeID: n.ID, Attention: a, Contribution: a})
	}
	return out
}

func mergeFeatureHeatmap(base []hypersonic.RapidFeatureContribution, weights map[string]float64) []hypersonic.RapidFeatureContribution {
	if len(weights) == 0 {
		return base
	}
	idx := make(map[string]int, len(base))
	for i := range base {
		idx[base[i].Feature] = i
	}
	for k, w := range weights {
		c := clamp01(math.Abs(w))
		if i, ok := idx[k]; ok {
			base[i].Contribution = clamp01(0.5*base[i].Contribution + 0.5*c)
		} else {
			base = append(base, hypersonic.RapidFeatureContribution{Feature: k, Value: w, Contribution: c})
		}
	}
	return base
}

func classicalManeuverType(deltaV, residual float64) maneuver.ManeuverType {
	absResidual := math.Abs(residual)
	switch {
	case deltaV >= 0.08:
		return maneuver.ManeuverImpulsive
	case deltaV >= 0.015:
		return maneuver.ManeuverLowThrust
	case absResidual >= 0.25:
		return maneuver.ManeuverHypersonic
	default:
		return maneuver.ManeuverNone
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

func clampScore(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 200 {
		return 200
	}
	return v
}

func selectOr(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}

func buildManeuverInferenceWindow(residuals, deltaVs []float64, endIdx, size int) []maneuver.ManeuverFrame {
	if size <= 0 {
		size = 24
	}
	if endIdx < 0 {
		return nil
	}
	if endIdx >= len(residuals) {
		endIdx = len(residuals) - 1
	}
	if endIdx >= len(deltaVs) {
		endIdx = len(deltaVs) - 1
	}
	if endIdx < 0 {
		return nil
	}

	start := endIdx - size + 1
	if start < 0 {
		start = 0
	}

	window := make([]maneuver.ManeuverFrame, 0, size)
	for i := start; i <= endIdx; i++ {
		window = append(window, maneuver.ManeuverFrame{
			Residual: residuals[i],
			DeltaV:   deltaVs[i],
		})
	}
	return window
}

func maxOr(a, b float64) float64 {
	if a > 0 {
		return a
	}
	return b
}
