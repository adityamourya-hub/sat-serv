package evaluation

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"ssis/pkg/ml/hybrid"
	"ssis/pkg/physics"
)

// ClassificationMetrics captures standard binary-classification quality metrics.
type ClassificationMetrics struct {
	Accuracy          float64
	FalsePositiveRate float64
	Precision         float64
	Recall            float64
	TP                int
	FP                int
	TN                int
	FN                int
}

// LatencyMetrics captures timing distribution for an operation.
type LatencyMetrics struct {
	Iterations int
	AvgMs      float64
	P50Ms      float64
	P95Ms      float64
	P99Ms      float64
	MinMs      float64
	MaxMs      float64
}

// SHAPFeature is a model/baseline attribution point used for fidelity checks.
type SHAPFeature struct {
	Feature      string
	Contribution float64
}

// SHAPFidelity captures explainability alignment against a classical baseline.
type SHAPFidelity struct {
	OverlapScore    float64
	RankCorrelation float64
	CombinedScore   float64
}

// ManeuverCase is one labeled sample for maneuver detection evaluation.
type ManeuverCase struct {
	ObjectID    int
	Residuals   []float64
	DeltaVs     []float64
	HasManeuver bool
}

// CollisionCase is one labeled sample for conjunction-risk evaluation.
type CollisionCase struct {
	Input       physics.CollisionRiskInput
	Graph       physics.CollisionNeighborhoodGraph
	IsCollision bool
}

// RapidCase is one labeled sample for rapid-threat ranking evaluation.
type RapidCase struct {
	Track        physics.RapidTrack
	IsHighThreat bool
}

func BinaryMetrics(truth, pred []bool) (ClassificationMetrics, error) {
	if len(truth) == 0 {
		return ClassificationMetrics{}, errors.New("truth labels cannot be empty")
	}
	if len(truth) != len(pred) {
		return ClassificationMetrics{}, errors.New("truth and prediction length mismatch")
	}

	m := ClassificationMetrics{}
	for i := range truth {
		switch {
		case truth[i] && pred[i]:
			m.TP++
		case !truth[i] && pred[i]:
			m.FP++
		case !truth[i] && !pred[i]:
			m.TN++
		case truth[i] && !pred[i]:
			m.FN++
		}
	}

	total := float64(len(truth))
	if total == 0 {
		return m, nil
	}
	m.Accuracy = float64(m.TP+m.TN) / total
	m.FalsePositiveRate = frac(m.FP, m.FP+m.TN)
	m.Precision = frac(m.TP, m.TP+m.FP)
	m.Recall = frac(m.TP, m.TP+m.FN)
	return m, nil
}

/**
 * EvaluateManeuverAccuracy computes maneuver detection metrics using the orchestrator.
 * Refactored to utilize the physics substrate schemas (ADITI 4.0).
 */
func EvaluateManeuverAccuracy(o *hybrid.Orchestrator, cases []ManeuverCase) (ClassificationMetrics, error) {
	fmt.Printf("[TRC] Entering EvaluateManeuverAccuracy for %d cases\n", len(cases))
	defer fmt.Printf("[TRC] Exiting EvaluateManeuverAccuracy\n")

	if o == nil {
		return ClassificationMetrics{}, errors.New("orchestrator is nil")
	}
	truth := make([]bool, 0, len(cases))
	pred := make([]bool, 0, len(cases))
	for _, c := range cases {
		events, _ := o.DetectManeuvers(c.ObjectID, c.Residuals, c.DeltaVs)
		truth = append(truth, c.HasManeuver)
		pred = append(pred, len(events) > 0)
	}
	return BinaryMetrics(truth, pred)
}

/**
 * EvaluateCollisionAccuracy computes classification metrics from collision probabilities.
 */
func EvaluateCollisionAccuracy(o *hybrid.Orchestrator, cases []CollisionCase, threshold float64) (ClassificationMetrics, error) {
	fmt.Printf("[TRC] Entering EvaluateCollisionAccuracy for %d cases\n", len(cases))
	defer fmt.Printf("[TRC] Exiting EvaluateCollisionAccuracy\n")

	if o == nil {
		return ClassificationMetrics{}, errors.New("orchestrator is nil")
	}
	if threshold <= 0 {
		threshold = 1e-4
	}
	truth := make([]bool, 0, len(cases))
	pred := make([]bool, 0, len(cases))
	for _, c := range cases {
		res, _ := o.EvaluateCollision(c.Input, c.Graph)
		truth = append(truth, c.IsCollision)
		pred = append(pred, res.Probability >= threshold)
	}
	return BinaryMetrics(truth, pred)
}

/**
 * EvaluateRapidAccuracy computes classification metrics from dynamic scores.
 */
func EvaluateRapidAccuracy(o *hybrid.Orchestrator, cases []RapidCase, maxDistanceKm float64, highScoreThreshold float64) (ClassificationMetrics, error) {
	fmt.Printf("[TRC] Entering EvaluateRapidAccuracy for %d cases\n", len(cases))
	defer fmt.Printf("[TRC] Exiting EvaluateRapidAccuracy\n")

	if o == nil {
		return ClassificationMetrics{}, errors.New("orchestrator is nil")
	}
	if highScoreThreshold <= 0 {
		highScoreThreshold = 100
	}
	truth := make([]bool, 0, len(cases))
	pred := make([]bool, 0, len(cases))
	for _, c := range cases {
		assessments, _ := o.RankRapid([]physics.RapidTrack{c.Track}, maxDistanceKm, 1)
		score := 0.0
		if len(assessments) > 0 {
			score = assessments[0].DynamicScore
		}
		truth = append(truth, c.IsHighThreat)
		pred = append(pred, score >= highScoreThreshold)
	}
	return BinaryMetrics(truth, pred)
}

// MeasureLatency runs fn for iterations and returns timing distribution metrics.
func MeasureLatency(iterations int, fn func()) LatencyMetrics {
	if iterations <= 0 {
		iterations = 1
	}
	d := make([]float64, 0, iterations)
	for i := 0; i < iterations; i++ {
		start := time.Now()
		fn()
		d = append(d, float64(time.Since(start).Microseconds())/1000.0)
	}
	sort.Float64s(d)

	total := 0.0
	for _, v := range d {
		total += v
	}
	return LatencyMetrics{
		Iterations: iterations,
		AvgMs:      total / float64(iterations),
		P50Ms:      percentile(d, 50),
		P95Ms:      percentile(d, 95),
		P99Ms:      percentile(d, 99),
		MinMs:      d[0],
		MaxMs:      d[len(d)-1],
	}
}

// SHAPFidelityAgainstBaseline computes explainability fidelity against classical attributions.
func SHAPFidelityAgainstBaseline(model, baseline []SHAPFeature, topK int) SHAPFidelity {
	if topK <= 0 {
		topK = 5
	}
	bm := topAbsMap(baseline, topK)
	mm := topAbsMap(model, topK)
	if len(bm) == 0 {
		return SHAPFidelity{OverlapScore: 1, RankCorrelation: 1, CombinedScore: 1}
	}

	baseMass := 0.0
	overlap := 0.0
	common := make([]string, 0)
	for f, b := range bm {
		baseMass += b
		if m, ok := mm[f]; ok {
			overlap += math.Min(b, m)
			common = append(common, f)
		}
	}
	overlapScore := 0.0
	if baseMass > 0 {
		overlapScore = overlap / baseMass
	}

	rankCorr := spearman(common, model, baseline)
	combined := clamp01(0.7*overlapScore + 0.3*((rankCorr+1.0)/2.0))
	return SHAPFidelity{OverlapScore: overlapScore, RankCorrelation: rankCorr, CombinedScore: combined}
}

func topAbsMap(features []SHAPFeature, k int) map[string]float64 {
	t := make([]SHAPFeature, len(features))
	copy(t, features)
	sort.Slice(t, func(i, j int) bool {
		return math.Abs(t[i].Contribution) > math.Abs(t[j].Contribution)
	})
	if len(t) > k {
		t = t[:k]
	}
	out := make(map[string]float64, len(t))
	norm := 0.0
	for _, f := range t {
		norm += math.Abs(f.Contribution)
	}
	if norm <= 1e-12 {
		norm = 1.0
	}
	for _, f := range t {
		out[f.Feature] = math.Abs(f.Contribution) / norm
	}
	return out
}

func spearman(common []string, a, b []SHAPFeature) float64 {
	if len(common) < 2 {
		return 1.0
	}
	ra := rankByAbs(a)
	rb := rankByAbs(b)
	n := float64(len(common))
	sumD2 := 0.0
	for _, f := range common {
		d := float64(ra[f] - rb[f])
		sumD2 += d * d
	}
	return 1.0 - (6.0*sumD2)/(n*(n*n-1.0))
}

func rankByAbs(features []SHAPFeature) map[string]int {
	t := make([]SHAPFeature, len(features))
	copy(t, features)
	sort.Slice(t, func(i, j int) bool {
		return math.Abs(t[i].Contribution) > math.Abs(t[j].Contribution)
	})
	r := make(map[string]int, len(t))
	for i, f := range t {
		r[f.Feature] = i + 1
	}
	return r
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	pos := (p / 100.0) * float64(len(sorted)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return sorted[lo]
	}
	fracPart := pos - float64(lo)
	return sorted[lo] + fracPart*(sorted[hi]-sorted[lo])
}

func frac(num, den int) float64 {
	if den <= 0 {
		return 0
	}
	return float64(num) / float64(den)
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
