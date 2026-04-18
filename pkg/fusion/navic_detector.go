package fusion

import (
	"fmt"
	"math"
)

// NavICAnomaly represents a detected GNSS anomaly.
type NavICAnomaly struct {
	Type       string  `json:"type"`       // "normal", "jamming", "spoofing"
	Confidence float64 `json:"confidence"` // [0, 1]
	Score      float64 `json:"score"`      // Raw anomaly score
	Evidence   string  `json:"evidence"`   // Human-readable description
}

// NavICFeatures holds dual-frequency pseudorange residuals for one epoch.
type NavICFeatures struct {
	PRResidualL5  float64 `json:"pr_residual_l5"`  // L5 pseudorange residual (m)
	PRResidualS   float64 `json:"pr_residual_s"`   // S-band pseudorange residual (m)
	CPResidualL5  float64 `json:"cp_residual_l5"`  // L5 carrier-phase residual (cycles)
	CPResidualS   float64 `json:"cp_residual_s"`   // S-band carrier-phase residual (cycles)
	IonoDiff      float64 `json:"iono_diff"`        // Dual-frequency ionospheric delay difference (m)
	CN0L5         float64 `json:"cn0_l5"`           // C/N0 ratio L5 (dB-Hz)
	CN0S          float64 `json:"cn0_s"`            // C/N0 ratio S-band (dB-Hz)
	GDOP          float64 `json:"gdop"`             // Geometric dilution of precision
	ClockDrift    float64 `json:"clock_drift_ns_s"` // Clock drift rate (ns/s)
	Multipath     float64 `json:"multipath_m"`      // Code-minus-carrier multipath estimator (m)
	ElevationRad  float64 `json:"elevation_rad"`    // Satellite elevation angle (rad)
	CrossCorr     float64 `json:"cross_corr"`       // Inter-SV consistency metric
}

// NavICDetector implements a rule-based anomaly detector for NavIC pseudorange residuals.
// This provides the Go-side fast-path; the Python LSTM model handles complex patterns.
type NavICDetector struct {
	// Thresholds
	JammingCN0Drop       float64
	SpoofingDriftRate    float64
	SpoofingIonoDiff     float64
	CrossCorrThreshold   float64
	ResidualSpikeThresh  float64
}

// NewNavICDetector creates a detector with calibrated default thresholds.
func NewNavICDetector() *NavICDetector {
	return &NavICDetector{
		JammingCN0Drop:      10.0, // dB-Hz drop indicates jamming
		SpoofingDriftRate:   0.1,  // m/epoch drift rate indicates spoofing
		SpoofingIonoDiff:    2.0,  // m iono inconsistency indicates spoofing
		CrossCorrThreshold:  0.7,  // Below this indicates anomaly
		ResidualSpikeThresh: 5.0,  // m residual spike threshold
	}
}

// Detect analyzes a sequence of NavIC features and classifies anomalies.
func (d *NavICDetector) Detect(history []NavICFeatures) (NavICAnomaly, error) {
	if len(history) < 3 {
		return NavICAnomaly{Type: "normal", Confidence: 0.5, Score: 0}, nil
	}

	jammingScore := d.computeJammingScore(history)
	spoofingScore := d.computeSpoofingScore(history)

	if jammingScore > spoofingScore && jammingScore > 0.5 {
		return NavICAnomaly{
			Type:       "jamming",
			Confidence: math.Min(jammingScore, 0.99),
			Score:      jammingScore,
			Evidence:   fmt.Sprintf("C/N0 drop detected (score=%.2f); broadband interference pattern", jammingScore),
		}, nil
	}

	if spoofingScore > 0.5 {
		return NavICAnomaly{
			Type:       "spoofing",
			Confidence: math.Min(spoofingScore, 0.99),
			Score:      spoofingScore,
			Evidence:   fmt.Sprintf("Pseudorange drift with iono inconsistency (score=%.2f)", spoofingScore),
		}, nil
	}

	return NavICAnomaly{
		Type:       "normal",
		Confidence: 1.0 - math.Max(jammingScore, spoofingScore),
		Score:      0,
		Evidence:   "No anomaly detected",
	}, nil
}

func (d *NavICDetector) computeJammingScore(history []NavICFeatures) float64 {
	if len(history) < 2 {
		return 0
	}
	score := 0.0
	n := len(history)

	// Check for sudden C/N0 drops
	for i := 1; i < n; i++ {
		dropL5 := history[i-1].CN0L5 - history[i].CN0L5
		dropS := history[i-1].CN0S - history[i].CN0S
		if dropL5 > d.JammingCN0Drop {
			score += dropL5 / (d.JammingCN0Drop * 3)
		}
		if dropS > d.JammingCN0Drop {
			score += dropS / (d.JammingCN0Drop * 3)
		}
	}

	// Check for residual spikes
	for _, f := range history {
		if math.Abs(f.PRResidualL5) > d.ResidualSpikeThresh {
			score += 0.1
		}
		if math.Abs(f.PRResidualS) > d.ResidualSpikeThresh {
			score += 0.1
		}
	}

	// Check cross-correlation degradation
	for _, f := range history {
		if f.CrossCorr < d.CrossCorrThreshold {
			score += 0.15
		}
	}

	return math.Min(score, 1.0)
}

func (d *NavICDetector) computeSpoofingScore(history []NavICFeatures) float64 {
	if len(history) < 3 {
		return 0
	}
	score := 0.0

	// Check for smooth pseudorange drift (spoofing signature)
	drifts := make([]float64, len(history)-1)
	for i := 1; i < len(history); i++ {
		drifts[i-1] = history[i].PRResidualL5 - history[i-1].PRResidualL5
	}
	// Consistent drift direction = spoofing
	positiveDrifts := 0
	negativeDrifts := 0
	for _, dr := range drifts {
		if dr > d.SpoofingDriftRate {
			positiveDrifts++
		} else if dr < -d.SpoofingDriftRate {
			negativeDrifts++
		}
	}
	maxConsistent := math.Max(float64(positiveDrifts), float64(negativeDrifts))
	if maxConsistent > float64(len(drifts))*0.6 {
		score += maxConsistent / float64(len(drifts))
	}

	// Check ionospheric delay inconsistency
	for _, f := range history {
		if math.Abs(f.IonoDiff) > d.SpoofingIonoDiff {
			score += 0.1
		}
	}

	// Check clock drift anomaly
	for _, f := range history {
		if math.Abs(f.ClockDrift) > 0.5 {
			score += 0.1
		}
	}

	// Subtle cross-corr degradation (spoofing is less dramatic than jamming)
	lowCorr := 0
	for _, f := range history {
		if f.CrossCorr < 0.85 && f.CrossCorr > d.CrossCorrThreshold {
			lowCorr++
		}
	}
	if lowCorr > len(history)/2 {
		score += 0.15
	}

	return math.Min(score, 1.0)
}
