package explainability

import (
	"fmt"
	"math"
	"sort"
	"ssis/pkg/maneuver"
	"ssis/pkg/physics"
)

/**
 * SHAPFeature holds a SHAP-like local attribution.
 */
type SHAPFeature struct {
	Feature      string  `json:"feature"`
	Value        float64 `json:"value"`
	Contribution float64 `json:"contribution"`
}

/**
 * AttentionPoint captures normalized temporal attention for sequence models.
 */
type AttentionPoint struct {
	Index  int     `json:"index"`
	Weight float64 `json:"weight"`
	Value  float64 `json:"value"`
}

/**
 * UncertaintySummary captures generic uncertainty diagnostics.
 */
type UncertaintySummary struct {
	Mean          float64 `json:"mean"`
	Lower         float64 `json:"lower"`
	Upper         float64 `json:"upper"`
	Width         float64 `json:"width"`
	RelativeWidth float64 `json:"relative_width"`
	Confidence    float64 `json:"confidence"`
}

/**
 * RapidExplanation packages explainability outputs for rapid-threat scoring.
 */
type RapidExplanation struct {
	TopFeatures []SHAPFeature      `json:"top_features"`
	Uncertainty UncertaintySummary `json:"uncertainty"`
}

/**
 * ManeuverExplanation packages explainability outputs for maneuver detection.
 */
type ManeuverExplanation struct {
	SHAP        []physics.FeatureContribution `json:"shap"`
	Attention   []AttentionPoint               `json:"attention"`
	Uncertainty UncertaintySummary             `json:"uncertainty"`
}

/**
 * CollisionExplanation packages explainability outputs for collision-risk inference.
 */
type CollisionExplanation struct {
	Heatmap     []physics.CollisionHeatmapCell `json:"heatmap"`
	Uncertainty UncertaintySummary               `json:"uncertainty"`
}

/**
 * ExplainRapid builds SHAP-like feature attributions and uncertainty summary.
 */
func ExplainRapid(track physics.RapidTrack, inference physics.RapidRankInference) RapidExplanation {
	fmt.Printf("[TRC] Entering ExplainRapid for track %d\n", track.ID)
	defer fmt.Printf("[TRC] Exiting ExplainRapid\n")

	features := map[string]float64{
		"mach":                  track.Mach,
		"altitude_km":           track.AltitudeKm,
		"burnout_velocity_km_s": track.BurnoutVelocityKmS,
		"distance_to_asset_km":  track.DistanceToAssetKm,
		"radar_cross_section":   track.RadarCrossSection,
	}

	weights := inference.FeatureWeights
	if len(weights) == 0 {
		weights = make(map[string]float64, len(features))
		for k, v := range features {
			weights[k] = v
		}
	}

	shap := make([]SHAPFeature, 0, len(weights))
	totalAbs := 0.0
	for k, raw := range weights {
		totalAbs += math.Abs(raw)
		shap = append(shap, SHAPFeature{Feature: k, Value: features[k], Contribution: raw})
	}
	if totalAbs <= 1e-12 {
		totalAbs = 1.0
	}

	for i := range shap {
		shap[i].Contribution = shap[i].Contribution / totalAbs
	}
	sort.Slice(shap, func(i, j int) bool {
		return math.Abs(shap[i].Contribution) > math.Abs(shap[j].Contribution)
	})
	if len(shap) > 5 {
		shap = shap[:5]
	}

	return RapidExplanation{
		TopFeatures: shap,
		Uncertainty: SummarizeUncertainty(
			inference.Score,
			inference.UncertaintyLow,
			inference.UncertaintyHigh,
			inference.Confidence,
		),
	}
}

/**
 * ExplainManeuver builds SHAP, attention over the window, and uncertainty.
 */
func ExplainManeuver(window []physics.ManeuverFrame, inf physics.ManeuverInference) ManeuverExplanation {
	fmt.Printf("[TRC] Entering ExplainManeuver\n")
	defer fmt.Printf("[TRC] Exiting ExplainManeuver\n")

	attention := BuildAttentionFromWindow(window)
	shap := inf.SHAP
	if len(shap) == 0 {
		shap = maneuver.HeuristicSHAP(window)
	}

	margin := math.Max(0.05, 0.5*(1.0-inf.Confidence))
	lower := clamp01(inf.Probability - margin)
	upper := clamp01(inf.Probability + margin)

	return ManeuverExplanation{
		SHAP:      shap,
		Attention: attention,
		Uncertainty: SummarizeUncertainty(
			inf.Probability,
			lower,
			upper,
			inf.Confidence,
		),
	}
}

/**
 * ExplainCollision builds node heatmap and uncertainty bounds.
 */
func ExplainCollision(graph physics.CollisionNeighborhoodGraph, inf physics.CollisionGraphInference) CollisionExplanation {
	fmt.Printf("[TRC] Entering ExplainCollision\n")
	defer fmt.Printf("[TRC] Exiting ExplainCollision\n")

	heatmap := make([]physics.CollisionHeatmapCell, 0, len(graph.Nodes))
	for _, n := range graph.Nodes {
		att := inf.NodeAttention[n.ObjectID]
		heatmap = append(heatmap, physics.CollisionHeatmapCell{
			NodeID:       n.ObjectID,
			Attention:    att,
			Contribution: att,
		})
	}
	sort.Slice(heatmap, func(i, j int) bool {
		return heatmap[i].Attention > heatmap[j].Attention
	})

	return CollisionExplanation{
		Heatmap: heatmap,
		Uncertainty: SummarizeUncertainty(
			inf.RiskScore,
			inf.UncertaintyLower,
			inf.UncertaintyUpper,
			inf.Confidence,
		),
	}
}

/**
 * BuildAttentionFromWindow uses normalized residual magnitudes as attention proxy.
 */
func BuildAttentionFromWindow(window []physics.ManeuverFrame) []AttentionPoint {
	if len(window) == 0 {
		return nil
	}

	points := make([]AttentionPoint, len(window))
	total := 0.0
	for i, f := range window {
		w := math.Abs(f.Residual) + 0.25*math.Abs(f.DeltaV)
		points[i] = AttentionPoint{Index: i, Weight: w, Value: f.Residual}
		total += w
	}

	if total <= 1e-12 {
		uniform := 1.0 / float64(len(points))
		for i := range points {
			points[i].Weight = uniform
		}
		return points
	}

	for i := range points {
		points[i].Weight /= total
	}
	return points
}

/**
 * SummarizeUncertainty computes width diagnostics from prediction bounds.
 */
func SummarizeUncertainty(mean, lower, upper, confidence float64) UncertaintySummary {
	if lower > upper {
		lower, upper = upper, lower
	}
	width := upper - lower
	if width <= 0 && mean != 0 {
		margin := math.Max(0.05, 0.5*(1.0-confidence)) * math.Abs(mean)
		lower = mean - margin
		upper = mean + margin
		width = upper - lower
	}
	den := math.Max(math.Abs(mean), 1e-9)
	return UncertaintySummary{
		Mean:          mean,
		Lower:         lower,
		Upper:         upper,
		Width:         width,
		RelativeWidth: width / den,
		Confidence:    clamp01(confidence),
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
