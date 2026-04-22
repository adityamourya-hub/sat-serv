package hybrid

import (
	"fmt"
	"math"
	"sort"

	"ssis/pkg/common"
	"ssis/pkg/hypersonic"
	"ssis/pkg/maneuver"
	"ssis/pkg/ml"
	"ssis/pkg/ml/pax"
	"ssis/pkg/physics"
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
 * Refactored to utilize the physics substrate schemas.
 */
type Orchestrator struct {
	Engine     ml.InferenceEngine
	Thresholds Thresholds
	PAX        *pax.Controller // PAX controller for tier selection
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
		t.ManeuverProbability = 1e-9
		t.ManeuverConfidence = 1e-9
		t.CollisionConfidence = 1e-9
		t.RapidConfidence = 1e-9
		t.MaxCollisionDeviation = 1e9
		t.MaxRapidDeviationPct = 1e9
	}
	return &Orchestrator{
		Engine:     engine,
		Thresholds: t,
		PAX:        pax.NewPAXController(), // Initialize PAX controller
	}
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
 * Now integrated with PAX controller for tier-based execution routing.
 */
func (o *Orchestrator) DetectManeuvers(objectID int, residuals, deltaVs []float64) ([]physics.ManeuverEvent, []ManeuverDecision) {
	fmt.Printf("[TRC] Entering Orchestrator.DetectManeuvers for object %d\n", objectID)
	defer fmt.Printf("[TRC] Exiting Orchestrator.DetectManeuvers\n")

	o.normalize()
	n := len(residuals)
	if len(deltaVs) < n {
		n = len(deltaVs)
	}
	if n == 0 {
		return nil, nil
	}

	detector := maneuver.NewCUSUMDetector(0.005, 0.05)
	events := make([]physics.ManeuverEvent, 0, n/5+1)
	decisions := make([]ManeuverDecision, 0, n/5+1)

	// Get current PAX tier
	activeTier := pax.Tier1 // Default to Tier1
	if o.PAX != nil {
		activeTier = o.PAX.GetActiveTier()
		fmt.Printf("[PAX] Active tier: %s\n", activeTier.String())
	}

	for i := 0; i < n; i++ {
		score, alarm := detector.Update(residuals[i])
		if !alarm {
			continue
		}

		baseline := physics.ManeuverEvent{
			ObjectID:  objectID,
			TimeIndex: i,
			Score:     score,
			DeltaV:    deltaVs[i],
			Type:      classicalManeuverType(deltaVs[i], residuals[i]),
		}
		selected := baseline
		source := "classical"
		reason := "cusum_alarm"

		// PAX Tier Routing
		if o.PAX != nil {
			modeProbs := map[string]float64{
				"ballistic": 0.05,
				"hgv_glide": 0.85,
				"maneuver":  0.10,
			}
			sindyR2 := 0.65 // Placeholder - would come from actual SINDy
			selectedTier := o.PAX.SelectTier(modeProbs, sindyR2)

			// Route based on tier
			switch selectedTier {
			case pax.Tier0:
				// Go-native ballistic only (no ML)
				source = "classical_tier0"
				reason = "pax_tier0_ballistic"
			case pax.Tier1:
				// P1 Audit: INT8 ONNX (pruned model) for thermal-throttled scenarios
				// Provides 3x speedup vs FP32, with graceful fallback if unavailable
				if o.Engine != nil && o.Engine.IsHealthy() {
					window := buildManeuverInferenceWindow(residuals, deltaVs, i, 24)
					inf, err := o.Engine.PredictManeuverINT8(window)
					if err == nil && inf.Probability >= o.Thresholds.ManeuverProbability {
						selected = physics.ManeuverEvent{
							ObjectID:  objectID,
							TimeIndex: i,
							Score:     inf.Probability * inf.Confidence,
							DeltaV:    maxOr(inf.DeltaV, deltaVs[i]),
							Type:      inf.Type,
						}
						source = "ml_tier1_int8"
						reason = "pax_tier1_pruned"
					}
				}
			case pax.Tier2:
				// FP32 full DPHNO
				if o.Engine != nil && o.Engine.IsHealthy() {
					window := buildManeuverInferenceWindow(residuals, deltaVs, i, 24)
					inf, err := o.Engine.PredictManeuver(window)
					if err == nil && inf.Probability >= o.Thresholds.ManeuverProbability && inf.Confidence >= o.Thresholds.ManeuverConfidence && inf.Type != physics.ManeuverNone {
						selected = physics.ManeuverEvent{
							ObjectID:  objectID,
							TimeIndex: i,
							Score:     inf.Probability * inf.Confidence,
							DeltaV:    maxOr(inf.DeltaV, deltaVs[i]),
							Type:      inf.Type,
						}
						source = "ml_tier2_full"
						reason = "pax_tier2_dphno"
					}
				}
			case pax.Tier3:
				// JAX discovery (zero-day)
				source = "classical_tier3"
				reason = "pax_t3_jax_discovery"
			}
		} else if o.Engine != nil && o.Engine.IsHealthy() {
			// Fallback to original ML path if PAX not available
			window := buildManeuverInferenceWindow(residuals, deltaVs, i, 24)

			inf, err := o.Engine.PredictManeuver(window)
			if err == nil && inf.Probability >= o.Thresholds.ManeuverProbability && inf.Confidence >= o.Thresholds.ManeuverConfidence && inf.Type != physics.ManeuverNone {
				selected = physics.ManeuverEvent{
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

	return events, decisions
}

/**
 * EvaluateCollision runs classical probability first, then accepts ML only if gates pass.
 */
func (o *Orchestrator) EvaluateCollision(input physics.CollisionRiskInput, graph physics.CollisionNeighborhoodGraph) (proximity.CollisionRiskResult, CollisionDecision) {
	fmt.Printf("[TRC] Entering Orchestrator.EvaluateCollision\n")
	defer fmt.Printf("[TRC] Exiting Orchestrator.EvaluateCollision\n")

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
	return res, CollisionDecision{Source: "ml", Reason: "thresholds_passed"}
}

/**
 * EvaluateReachability implements the Tactical Reachability Assessment (Phase 3).
 */
func (o *Orchestrator) EvaluateReachability(asset, threat physics.State, deltaV, horizon float64) (ReachabilityDecision, error) {
	fmt.Printf("[TRC] Entering Orchestrator.EvaluateReachability AssetPos=(%.2f, %.2f) ThreatPos=(%.2f, %.2f) dv=%.3f\n",
		asset.Pos[0], asset.Pos[1], threat.Pos[0], threat.Pos[1], deltaV)

	rri, err := proximity.CalculateRRI(asset, threat, deltaV, horizon)
	if err != nil {
		fmt.Printf("[ERR] EvaluateReachability: %v\n", err)
		return ReachabilityDecision{Source: "error", HorizonSec: horizon}, err
	}

	dx := asset.Pos[0] - threat.Pos[0]
	dy := asset.Pos[1] - threat.Pos[1]
	dz := asset.Pos[2] - threat.Pos[2]
	relPos := [3]float64{dx, dy, dz}

	isThreat := rri.Intersects(relPos)

	vol := 0.0
	if rri.Ellipsoid != nil && rri.Ellipsoid.Matrix != nil {
		detE := proximity.Determinant3x3(rri.Ellipsoid.Matrix)
		if detE > 0 {
			vol = (4.0 / 3.0) * math.Pi * (1.0 / math.Sqrt(detE))
		}
	}

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
func (o *Orchestrator) RankRapid(tracks []physics.RapidTrack, maxDistanceKm float64, maxResults int) ([]hypersonic.RapidAssessment, []RapidDecision) {
	fmt.Printf("[TRC] Entering Orchestrator.RankRapid\n")
	defer fmt.Printf("[TRC] Exiting Orchestrator.RankRapid\n")

	o.normalize()
	if maxResults <= 0 {
		maxResults = len(tracks)
	}

	assessments := make([]hypersonic.RapidAssessment, 0, len(tracks))
	decisions := make([]RapidDecision, 0, len(tracks))

	for _, track := range tracks {
		base := hypersonic.ProcessRapidTracks([]physics.RapidTrack{track}, maxDistanceKm, 1)
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

	return assessments, decisions
}

func heatmapFromAttention(graph physics.CollisionNeighborhoodGraph, attention map[int]float64) []physics.CollisionHeatmapCell {
	if len(graph.Nodes) == 0 {
		return nil
	}
	out := make([]physics.CollisionHeatmapCell, 0, len(graph.Nodes))
	for _, n := range graph.Nodes {
		a := clamp01(attention[n.ObjectID])
		out = append(out, physics.CollisionHeatmapCell{NodeID: n.ObjectID, Attention: a, Contribution: a})
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

func classicalManeuverType(deltaV, residual float64) physics.ManeuverType {
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

func buildManeuverInferenceWindow(residuals, deltaVs []float64, endIdx, size int) []physics.ManeuverFrame {
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

	window := make([]physics.ManeuverFrame, 0, size)
	for i := start; i <= endIdx; i++ {
		window = append(window, physics.ManeuverFrame{
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
