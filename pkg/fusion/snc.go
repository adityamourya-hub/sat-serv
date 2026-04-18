package fusion

import (
	"math"
	safemath "ssis/pkg/math"
)

// SNCConfig controls state noise compensation used in covariance propagation.
type SNCConfig struct {
	QcAccelKm2Sec3 float64 // continuous-time acceleration PSD
	TauSec         float64 // Gauss-Markov correlation time
	Adaptive       bool
}

func DefaultSNCConfig() SNCConfig {
	return SNCConfig{QcAccelKm2Sec3: 2.5e-9, TauSec: 1800, Adaptive: false}
}

// BuildSNCCovariance computes discrete process covariance using first-order
// Gauss-Markov shaping (DMC/SNC style) for translational dynamics.
func BuildSNCCovariance(dtSec float64, cfg SNCConfig, innovationNorm float64) [][]float64 {
	if dtSec <= 0 {
		dtSec = 1
	}
	if cfg.TauSec <= 0 {
		cfg.TauSec = 1800
	}
	qc := cfg.QcAccelKm2Sec3
	if qc <= 0 {
		qc = 2.5e-9
	}
	if cfg.Adaptive {
		scale := 1.0 + 0.5*math.Min(math.Max(innovationNorm, 0), 5.0)
		qc *= scale
	}

	beta := math.Exp(-dtSec / cfg.TauSec)
	gm := 1.0 - beta*beta
	if gm < 1e-8 {
		gm = 1e-8
	}
	q11 := qc * gm * dtSec * dtSec * dtSec / 3.0
	q12 := qc * gm * dtSec * dtSec / 2.0
	q22 := qc * gm * dtSec

	q := safemath.ZeroMatrix(6, 6)
	for axis := 0; axis < 3; axis++ {
		ri := axis
		vi := axis + 3
		q[ri][ri] = q11
		q[ri][vi] = q12
		q[vi][ri] = q12
		q[vi][vi] = q22
	}
	return q
}
