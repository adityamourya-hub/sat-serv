package fusion

import (
	"math"
	"sort"
	safemath "ssis/pkg/math"

	"gonum.org/v1/gonum/stat"
)

type CovarianceRealismMetrics struct {
	Samples               int     `json:"samples"`
	MeanNIS               float64 `json:"mean_nis"`
	P95NIS                float64 `json:"p95_nis"`
	MeanNEES              float64 `json:"mean_nees"`
	P95NEES               float64 `json:"p95_nees"`
	InnovationConsistency bool    `json:"innovation_consistency"`
	NEESConsistency       bool    `json:"nees_consistency"`
}

type PcCalibrationMetrics struct {
	Samples          int     `json:"samples"`
	BrierScore       float64 `json:"brier_score"`
	ReliabilityError float64 `json:"reliability_error"`
	CalibrationPass  bool    `json:"calibration_pass"`
}

// EvaluateCovarianceRealism computes NIS/NEES style consistency statistics.
func EvaluateCovarianceRealism(innovations [][]float64, innovationCovariances [][][]float64, stateErrors [][]float64, stateCovariances [][][]float64) CovarianceRealismMetrics {
	nis := make([]float64, 0, len(innovations))
	for i := range innovations {
		if i >= len(innovationCovariances) {
			break
		}
		inv, err := safemath.MatrixInverse(innovationCovariances[i], 1e-9)
		if err != nil {
			continue
		}
		nis = append(nis, QuadraticForm(innovations[i], inv)/math.Max(float64(len(innovations[i])), 1))
	}
	nees := make([]float64, 0, len(stateErrors))
	for i := range stateErrors {
		if i >= len(stateCovariances) {
			break
		}
		inv, err := safemath.MatrixInverse(stateCovariances[i], 1e-9)
		if err != nil {
			continue
		}
		nees = append(nees, QuadraticForm(stateErrors[i], inv)/math.Max(float64(len(stateErrors[i])), 1))
	}

	meanNIS, p95NIS := meanP95(nis)
	meanNEES, p95NEES := meanP95(nees)
	return CovarianceRealismMetrics{
		Samples:               minIntCR(len(nis), len(nees)),
		MeanNIS:               meanNIS,
		P95NIS:                p95NIS,
		MeanNEES:              meanNEES,
		P95NEES:               p95NEES,
		InnovationConsistency: meanNIS > 0.25 && meanNIS < 3.5,
		NEESConsistency:       meanNEES > 0.25 && meanNEES < 3.5,
	}
}

// EvaluatePcCalibration computes calibration quality for predicted probabilities.
func EvaluatePcCalibration(predicted []float64, labels []bool) PcCalibrationMetrics {
	n := len(predicted)
	if len(labels) < n {
		n = len(labels)
	}
	if n == 0 {
		return PcCalibrationMetrics{}
	}
	brier := 0.0
	for i := 0; i < n; i++ {
		p := crClamp01(predicted[i])
		y := 0.0
		if labels[i] {
			y = 1.0
		}
		d := p - y
		brier += d * d
	}
	brier /= float64(n)

	const bins = 10
	count := make([]float64, bins)
	sumP := make([]float64, bins)
	sumY := make([]float64, bins)
	for i := 0; i < n; i++ {
		p := crClamp01(predicted[i])
		b := int(p * bins)
		if b >= bins {
			b = bins - 1
		}
		count[b]++
		sumP[b] += p
		if labels[i] {
			sumY[b] += 1
		}
	}
	rel := 0.0
	for b := 0; b < bins; b++ {
		if count[b] == 0 {
			continue
		}
		mp := sumP[b] / count[b]
		my := sumY[b] / count[b]
		rel += (count[b] / float64(n)) * math.Abs(mp-my)
	}

	return PcCalibrationMetrics{
		Samples:          n,
		BrierScore:       brier,
		ReliabilityError: rel,
		CalibrationPass:  brier <= 0.20 && rel <= 0.10,
	}
}

func meanP95(v []float64) (float64, float64) {
	if len(v) == 0 {
		return 0, 0
	}
	m := stat.Mean(v, nil)
	copyV := append([]float64(nil), v...)
	sort.Float64s(copyV)
	idx := int(0.95 * float64(len(copyV)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(copyV) {
		idx = len(copyV) - 1
	}
	return m, copyV[idx]
}

func crClamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func minIntCR(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ProbabilityCalibrator applies Platt scaling for post-hoc probability calibration.
type ProbabilityCalibrator struct {
	Alpha float64 // Sigmoid slope
	Beta  float64 // Sigmoid offset
}

// NewProbabilityCalibrator trains Platt scaling on validation data.
// Returns calibrator that maps logit(p) -> p_calibrated.
func NewProbabilityCalibrator(predictions []float64, labels []bool) *ProbabilityCalibrator {
	n := len(predictions)
	if len(labels) < n {
		n = len(labels)
	}
	if n < 10 {
		return &ProbabilityCalibrator{Alpha: 1.0, Beta: 0.0}
	}

	const eps = 1e-6
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		p := crClamp01(predictions[i])
		if p < eps {
			p = eps
		} else if p > 1.0-eps {
			p = 1.0 - eps
		}
		x[i] = math.Log(p / (1.0 - p))
		if labels[i] {
			y[i] = 1.0
		}
	}

	alpha := 1.0
	beta := 0.0
	lr := 0.02
	for iter := 0; iter < 600; iter++ {
		gA := 0.0
		gB := 0.0
		for i := 0; i < n; i++ {
			z := alpha*x[i] + beta
			s := 1.0 / (1.0 + math.Exp(-z))
			err := s - y[i]
			gA += err * x[i]
			gB += err
		}
		gA /= float64(n)
		gB /= float64(n)
		alpha -= lr * gA
		beta -= lr * gB
		if math.Abs(gA)+math.Abs(gB) < 1e-6 {
			break
		}
	}

	if math.IsNaN(alpha) || math.IsInf(alpha, 0) || math.Abs(alpha) < 1e-6 {
		alpha = 1.0
	}
	if math.IsNaN(beta) || math.IsInf(beta, 0) {
		beta = 0.0
	}
	return &ProbabilityCalibrator{Alpha: alpha, Beta: beta}
}

// Calibrate applies the learned transformation to a raw probability.
func (cal *ProbabilityCalibrator) Calibrate(p float64) float64 {
	p = crClamp01(p)
	if p < 0.001 || p > 0.999 {
		// Avoid numerical issues at extremes
		return p
	}
	logit := math.Log(p / (1 - p))
	transformed := cal.Alpha*logit + cal.Beta
	// Apply sigmoid to transformed logit
	return 1.0 / (1.0 + math.Exp(-transformed))
}
