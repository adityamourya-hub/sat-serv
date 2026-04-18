package hypersonic

import (
	"math"
	"sort"
)

type ThreatClass string

const (
	ThreatUnknown    ThreatClass = "unknown"
	ThreatOrbital    ThreatClass = "orbital"
	ThreatBallistic  ThreatClass = "ballistic"
	ThreatHypersonic ThreatClass = "hypersonic"
)

type RapidTrack struct {
	ID                  int
	Mach                float64
	AltitudeKm          float64
	BurnoutVelocityKmS  float64
	FlightPathAngleRad  float64
	AzimuthRad          float64 // 0=North, pi/2=East
	LaunchLat           float64
	LaunchLon           float64
	DistanceToAssetKm   float64
	RadarCrossSection   float64
	ManeuverIntentProxy bool
	ManeuverIntentScore float64
}

type RapidAssessment struct {
	ID              int
	Class           ThreatClass
	Priority        int
	Reachable       bool
	DynamicScore    float64
	ScoreConfidence float64
	UncertaintyLow  float64
	UncertaintyHigh float64
	Recommendation  EvasiveRecommendation
	FeatureHeatmap  []RapidFeatureContribution
	Ballistic       BallisticThreat
	Reentry         ReentryState
}

type EvasiveAction string

const (
	ActionMonitorOnly     EvasiveAction = "monitor_only"
	ActionSensorRetask    EvasiveAction = "sensor_retask"
	ActionPrepareEvasive  EvasiveAction = "prepare_evasive"
	ActionEvasiveNow      EvasiveAction = "evasive_maneuver_now"
	ActionOutOfReachWatch EvasiveAction = "out_of_reach_watch"
)

type EvasiveRecommendation struct {
	Action     EvasiveAction
	Confidence float64
	Reason     string
}

type RapidFeatureContribution struct {
	Feature      string
	Value        float64
	Contribution float64
}

type RapidRankInference struct {
	Score           float64
	Confidence      float64
	UncertaintyLow  float64
	UncertaintyHigh float64
	FeatureWeights  map[string]float64
}

type RapidPriorityModel interface {
	Name() string
	Rank(track RapidTrack, class ThreatClass, reachable bool, baselinePriority int) (RapidRankInference, error)
}

type RapidPolicyInference struct {
	Action     EvasiveAction
	Confidence float64
	Reason     string
}

type RapidPolicyModel interface {
	Name() string
	Recommend(track RapidTrack, class ThreatClass, score float64, reachable bool) (RapidPolicyInference, error)
}

type HybridRapidThreatEngine struct {
	Ranker                RapidPriorityModel
	Policy                RapidPolicyModel
	MinRankerConfidence   float64
	MaxRankerDeviationPct float64
}

func NewHybridRapidThreatEngine(ranker RapidPriorityModel, policy RapidPolicyModel) *HybridRapidThreatEngine {
	return &HybridRapidThreatEngine{
		Ranker:                ranker,
		Policy:                policy,
		MinRankerConfidence:   0.55,
		MaxRankerDeviationPct: 0.60,
	}
}

func ClassifyRapidTrack(track RapidTrack) ThreatClass {
	if track.Mach >= 5.0 && track.AltitudeKm >= 20.0 && track.AltitudeKm <= 100.0 {
		return ThreatHypersonic
	}
	if track.BurnoutVelocityKmS >= 1.0 && track.AltitudeKm < 300.0 {
		return ThreatBallistic
	}
	if track.AltitudeKm >= 120.0 {
		return ThreatOrbital
	}
	return ThreatUnknown
}

func ReachabilityGate(track RapidTrack, maxDistanceKm float64) bool {
	return track.DistanceToAssetKm <= maxDistanceKm
}

func threatPriority(track RapidTrack, class ThreatClass) int {
	score := 0
	switch class {
	case ThreatHypersonic:
		score += 100
	case ThreatBallistic:
		score += 80
	case ThreatOrbital:
		score += 40
	default:
		score += 20
	}

	if track.DistanceToAssetKm <= 500 {
		score += 50
	} else if track.DistanceToAssetKm <= 1500 {
		score += 25
	}

	if track.Mach >= 8 {
		score += 25
	} else if track.Mach >= 5 {
		score += 15
	}

	if track.RadarCrossSection < 0.5 {
		score += 10
	}

	if track.ManeuverIntentProxy {
		score += 25
	}
	if track.ManeuverIntentScore > 0 {
		score += int(math.Round(10.0 * clamp01(track.ManeuverIntentScore)))
	}

	return score
}

func ProcessRapidTracks(tracks []RapidTrack, maxDistanceKm float64, maxResults int) []RapidAssessment {
	return ProcessRapidTracksHybrid(tracks, maxDistanceKm, maxResults, nil)
}

func ProcessRapidTracksHybrid(tracks []RapidTrack, maxDistanceKm float64, maxResults int, engine *HybridRapidThreatEngine) []RapidAssessment {
	if maxResults <= 0 {
		maxResults = len(tracks)
	}

	type rankedItem struct {
		track     RapidTrack
		class     ThreatClass
		priority  int
		dynScore  float64
		conf      float64
		low       float64
		high      float64
		expl      []RapidFeatureContribution
		recommend EvasiveRecommendation
	}

	items := make([]rankedItem, 0, len(tracks))

	for _, track := range tracks {
		reachable := ReachabilityGate(track, maxDistanceKm)
		if !reachable {
			continue
		}

		class := ClassifyRapidTrack(track)
		baseline := threatPriority(track, class)
		dynScore := float64(baseline)
		conf := 1.0
		low := clampScore(dynScore * 0.8)
		high := clampScore(dynScore * 1.2)
		expl := heuristicRapidHeatmap(track, class)

		if engine != nil {
			if engine.MinRankerConfidence <= 0 {
				engine.MinRankerConfidence = 0.55
			}
			if engine.MaxRankerDeviationPct <= 0 {
				engine.MaxRankerDeviationPct = 0.60
			}

			if engine.Ranker != nil {
				r, err := engine.Ranker.Rank(track, class, reachable, baseline)
				if err == nil {
					candidate := clampScore(r.Score)
					dev := math.Abs(candidate-float64(baseline)) / math.Max(float64(baseline), 1.0)
					if r.Confidence >= engine.MinRankerConfidence && dev <= engine.MaxRankerDeviationPct {
						dynScore = 0.7*float64(baseline) + 0.3*candidate
						conf = clamp01(r.Confidence)
						if r.UncertaintyLow > 0 || r.UncertaintyHigh > 0 {
							low = clampScore(r.UncertaintyLow)
							high = clampScore(r.UncertaintyHigh)
							if low > high {
								low, high = high, low
							}
						}
						expl = combineRapidHeatmap(expl, r.FeatureWeights)
					}
				}
			}
		}

		rec := defaultRecommendation(class, dynScore, reachable)
		if engine != nil && engine.Policy != nil {
			p, err := engine.Policy.Recommend(track, class, dynScore, reachable)
			if err == nil && p.Action != "" {
				rec = EvasiveRecommendation{Action: p.Action, Confidence: clamp01(p.Confidence), Reason: p.Reason}
			}
		}

		items = append(items, rankedItem{
			track:     track,
			class:     class,
			priority:  int(math.Round(dynScore)),
			dynScore:  dynScore,
			conf:      conf,
			low:       low,
			high:      high,
			expl:      expl,
			recommend: rec,
		})
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].dynScore > items[j].dynScore
	})

	out := make([]RapidAssessment, 0, maxResults)
	for i := 0; i < len(items) && len(out) < maxResults; i++ {
		item := items[i]
		a := RapidAssessment{
			ID:              item.track.ID,
			Class:           item.class,
			Priority:        item.priority,
			Reachable:       true,
			DynamicScore:    item.dynScore,
			ScoreConfidence: item.conf,
			UncertaintyLow:  item.low,
			UncertaintyHigh: item.high,
			Recommendation:  item.recommend,
			FeatureHeatmap:  item.expl,
		}

		switch item.class {
		case ThreatBallistic, ThreatHypersonic:
			a.Ballistic = ComputeBallisticImpact(
				item.track.LaunchLat,
				item.track.LaunchLon,
				item.track.BurnoutVelocityKmS,
				item.track.FlightPathAngleRad*(180.0/math.Pi),
				item.track.AzimuthRad*(180.0/math.Pi),
			)
			a.Reentry = SolveAerospace(item.track.Mach, item.track.AltitudeKm)
		case ThreatOrbital:
			a.Reentry = SolveAerospace(item.track.Mach, item.track.AltitudeKm)
		}

		out = append(out, a)
	}

	return out
}

func defaultRecommendation(class ThreatClass, score float64, reachable bool) EvasiveRecommendation {
	if !reachable {
		return EvasiveRecommendation{Action: ActionOutOfReachWatch, Confidence: 0.75, Reason: "outside engagement envelope"}
	}

	switch {
	case class == ThreatHypersonic && score >= 140:
		return EvasiveRecommendation{Action: ActionEvasiveNow, Confidence: 0.88, Reason: "hypersonic high-priority inbound track"}
	case class == ThreatBallistic && score >= 120:
		return EvasiveRecommendation{Action: ActionPrepareEvasive, Confidence: 0.82, Reason: "ballistic trajectory with high threat score"}
	case score >= 90:
		return EvasiveRecommendation{Action: ActionSensorRetask, Confidence: 0.76, Reason: "elevated threat; retask sensor and verify track"}
	default:
		return EvasiveRecommendation{Action: ActionMonitorOnly, Confidence: 0.70, Reason: "monitor under current rules of engagement"}
	}
}

func heuristicRapidHeatmap(track RapidTrack, class ThreatClass) []RapidFeatureContribution {
	base := []RapidFeatureContribution{
		{Feature: "distance_to_asset_km", Value: track.DistanceToAssetKm, Contribution: invScaled(track.DistanceToAssetKm, 2000)},
		{Feature: "mach", Value: track.Mach, Contribution: clamp01(track.Mach / 10.0)},
		{Feature: "altitude_km", Value: track.AltitudeKm, Contribution: clamp01(track.AltitudeKm / 150.0)},
		{Feature: "radar_cross_section", Value: track.RadarCrossSection, Contribution: invScaled(track.RadarCrossSection, 5.0)},
		{Feature: "maneuver_intent_score", Value: track.ManeuverIntentScore, Contribution: clamp01(track.ManeuverIntentScore)},
	}

	classBoost := 0.1
	if class == ThreatHypersonic {
		classBoost = 0.25
	} else if class == ThreatBallistic {
		classBoost = 0.18
	}

	base = append(base, RapidFeatureContribution{Feature: "threat_class_boost", Value: float64(classPriorityWeight(class)), Contribution: classBoost})
	return base
}

func combineRapidHeatmap(base []RapidFeatureContribution, modelWeights map[string]float64) []RapidFeatureContribution {
	if len(modelWeights) == 0 {
		return base
	}

	byName := make(map[string]int, len(base))
	for i := range base {
		byName[base[i].Feature] = i
	}
	for k, w := range modelWeights {
		c := clamp01(math.Abs(w))
		if idx, ok := byName[k]; ok {
			base[idx].Contribution = clamp01(0.5*base[idx].Contribution + 0.5*c)
			continue
		}
		base = append(base, RapidFeatureContribution{Feature: k, Value: 0, Contribution: c})
	}
	return base
}

func classPriorityWeight(class ThreatClass) int {
	switch class {
	case ThreatHypersonic:
		return 100
	case ThreatBallistic:
		return 80
	case ThreatOrbital:
		return 40
	default:
		return 20
	}
}

func invScaled(v, max float64) float64 {
	if max <= 0 {
		return 0
	}
	s := 1.0 - (v / max)
	return clamp01(s)
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
	if v > 300 {
		return 300
	}
	return v
}
