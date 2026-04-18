package fusion

import (
	"fmt"
	"math"
	"sort"
	"time"

	engcrypto "ssis/pkg/crypto"
	safemath "ssis/pkg/math"
	"ssis/pkg/orbital"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type MeasurementKind string

const (
	MeasurementRadar MeasurementKind = "radar"
	MeasurementTLE   MeasurementKind = "tle"
)

type ODMeasurement struct {
	TimestampUnix int64
	Kind          MeasurementKind
	Radar         *RadarMeasurement
	TLE           *TLEMeasurement
}

type ODEpoch struct {
	TimestampUnix int64           `json:"timestamp_unix"`
	Kind          MeasurementKind `json:"kind"`
	State         orbital.State   `json:"state"`
	Covariance    [][]float64     `json:"covariance"`
	ThreeSigma    Ellipsoid3Sigma `json:"three_sigma"`
	InnovationNIS float64         `json:"innovation_nis"`
	Mahalanobis   float64         `json:"mahalanobis"`
	Accepted      bool            `json:"accepted"`
}

type ODSummary struct {
	Epochs                  int     `json:"epochs"`
	AcceptedEpochs          int     `json:"accepted_epochs"`
	AcceptanceRatio         float64 `json:"acceptance_ratio"`
	MeanNIS                 float64 `json:"mean_nis"`
	P95NIS                  float64 `json:"p95_nis"`
	MeanMahalanobis         float64 `json:"mean_mahalanobis"`
	CovarianceTraceMean     float64 `json:"covariance_trace_mean"`
	InnovationConsistencyOK bool    `json:"innovation_consistency_ok"`
}

type ODSignedOutput struct {
	ObjectID       int       `json:"object_id"`
	GeneratedAtUTC string    `json:"generated_at_utc"`
	Summary        ODSummary `json:"summary"`
	Signature      string    `json:"signature"`
	SignatureValid bool      `json:"signature_valid"`
	PublicKeyHex   string    `json:"public_key_hex,omitempty"`
}

type ODResult struct {
	Epochs  []ODEpoch      `json:"epochs"`
	Summary ODSummary      `json:"summary"`
	Signed  ODSignedOutput `json:"signed"`
}

// RunSequentialODUKF performs sequential orbit determination using UKF with
// adaptive SNC and innovation gating (Mahalanobis consistency checks).
func RunSequentialODUKF(initial orbital.State, initialCov [][]float64, measurements []ODMeasurement, ukfCfg UKFConfig, sncCfg SNCConfig, pinn *orbital.PINNRefiner, trans *orbital.HybridTransformer, signer *engcrypto.Signer, objectID int) (ODResult, error) {
	if len(measurements) == 0 {
		return ODResult{}, fmt.Errorf("no measurements provided")
	}
	if ukfCfg.Alpha <= 0 {
		ukfCfg = DefaultUKFConfig()
	}

	sort.Slice(measurements, func(i, j int) bool { return measurements[i].TimestampUnix < measurements[j].TimestampUnix })
	x := stateToVector(initial)
	p := normalizeCovariance(initialCov, 6)
	lastTS := measurements[0].TimestampUnix
	if lastTS == 0 {
		lastTS = time.Now().Unix()
	}

	epochs := make([]ODEpoch, 0, len(measurements))
	nisValues := make([]float64, 0, len(measurements))
	mahValues := make([]float64, 0, len(measurements))
	traceValues := make([]float64, 0, len(measurements))
	accepted := 0

	for _, meas := range measurements {
		ts := meas.TimestampUnix
		if ts == 0 {
			ts = lastTS + 60
		}
		dt := float64(ts - lastTS)
		if dt <= 0 {
			dt = 1
		}
		lastTS = ts

		sigma, wm, wc, err := GenerateSigmaPoints(x, p, ukfCfg)
		if err != nil {
			return ODResult{}, fmt.Errorf("sigma points failed: %w", err)
		}
		baseJd := float64(lastTS)/86400.0 + 2440587.5
		predSigma := make([][]float64, len(sigma))
		for i := range sigma {
			predSigma[i] = propagateSigmaPoint(sigma[i], dt, baseJd)
		}
		x, p = UnscentedMeanCov(predSigma, wm, wc)
		if pinn != nil {
			ctx := orbital.DefaultPINNRefineContext()
			ctx.LongArcThresholdSec = math.Max(90.0, dt)
			refined, err := pinn.RefineStateAdaptive(x, p, dt, ctx)
			if err == nil {
				x = refined
			}
		}
		p = matAdd(p, BuildSNCCovariance(dt, sncCfg, 0))
		p = enforcePositiveDiagonal(symmetrize(p), maxFloat64(ukfCfg.MinNumericalJitter, 1e-9))

		epoch := ODEpoch{TimestampUnix: ts, Kind: meas.Kind, State: vectorToState(x), Covariance: p, ThreeSigma: ComputeThreeSigmaEllipsoid(p), Accepted: false}
		traceValues = append(traceValues, MatrixTrace(p))

		var z []float64
		var r [][]float64
		var h func([]float64) []float64
		switch meas.Kind {
		case MeasurementRadar:
			if meas.Radar == nil {
				epochs = append(epochs, epoch)
				continue
			}
			z = RadarMeasurementVector(*meas.Radar)
			r = RadarMeasurementNoise(*meas.Radar)
			siteID := meas.Radar.SiteID
			h = func(s []float64) []float64 { return RadarMeasurementFromState(s, siteID, baseJd) }
		case MeasurementTLE:
			if meas.TLE == nil {
				epochs = append(epochs, epoch)
				continue
			}
			z = TLEMeasurementVector(*meas.TLE)
			r = TLEMeasurementNoise(*meas.TLE)
			h = func(s []float64) []float64 { return []float64{s[0], s[1], s[2], s[3], s[4], s[5]} }
		default:
			epochs = append(epochs, epoch)
			continue
		}

		sx, sp, nis, mahal, ok, err := UKFUpdateWithGating(x, p, z, r, h, ukfCfg)
		if err != nil {
			epoch.Accepted = false
			epochs = append(epochs, epoch)
			continue
		}
		epoch.InnovationNIS = nis
		epoch.Mahalanobis = mahal
		epoch.Accepted = ok
		nisValues = append(nisValues, nis)
		mahValues = append(mahValues, mahal)

		if ok {
			x, p = sx, sp
			if trans != nil {
				trans.AddHistory(objectID, vectorToState(x))
				refinedX, refinedP, err := trans.Refine(objectID, x, p)
				if err == nil {
					x, p = refinedX, refinedP
				}
			}
			epoch.State = vectorToState(x)
			epoch.Covariance = p
			epoch.ThreeSigma = ComputeThreeSigmaEllipsoid(p)
			accepted++
		}
		epochs = append(epochs, epoch)
	}

	summary := buildODSummary(epochs, nisValues, mahValues, traceValues, accepted)
	signed := ODSignedOutput{ObjectID: objectID, GeneratedAtUTC: time.Now().UTC().Format(time.RFC3339), Summary: summary}
	if signer != nil && objectID != 0 {
		ts := time.Now().Unix()
		signed.Signature = signer.GenerateProof(objectID, "OD_UKF_SUMMARY", ts)
		payload := fmt.Sprintf("%d:%s:%d", objectID, "OD_UKF_SUMMARY", ts)
		signed.SignatureValid = signer.Verify([]byte(payload), signed.Signature)
		signed.PublicKeyHex = signer.PublicKeyHex()
	}

	return ODResult{Epochs: epochs, Summary: summary, Signed: signed}, nil
}

// UKFUpdateWithGating performs a gated measurement update.
func UKFUpdateWithGating(x []float64, p [][]float64, z []float64, r [][]float64, h func([]float64) []float64, cfg UKFConfig) ([]float64, [][]float64, float64, float64, bool, error) {
	sigma, wm, wc, err := GenerateSigmaPoints(x, p, cfg)
	if err != nil {
		return nil, nil, 0, 0, false, err
	}

	m := len(z)
	zSigma := make([][]float64, len(sigma))
	for i := range sigma {
		zSigma[i] = h(sigma[i])
		if len(zSigma[i]) != m {
			return nil, nil, 0, 0, false, fmt.Errorf("measurement dimension mismatch")
		}
	}

	zPred := make([]float64, m)
	for i := range zSigma {
		for j := 0; j < m; j++ {
			zPred[j] += wm[i] * zSigma[i][j]
		}
	}

	s := safemath.ZeroMatrix(m, m)
	pxz := safemath.ZeroMatrix(len(x), m)
	for i := range zSigma {
		dz := vecSub(zSigma[i], zPred)
		dx := vecSub(sigma[i], x)
		s = matAdd(s, outerScaled(dz, dz, wc[i]))
		pxz = matAdd(pxz, outerScaled(dx, dz, wc[i]))
	}
	s = matAdd(s, r)
	s = enforcePositiveDiagonal(symmetrize(s), maxFloat64(cfg.MinNumericalJitter, 1e-9))

	sInv, err := matInverse(s, cfg.MinNumericalJitter)
	if err != nil {
		return nil, nil, 0, 0, false, err
	}
	innov := vecSub(z, zPred)
	mahal2, _, err := safemath.MahalanobisSquared(innov, s, maxFloat64(cfg.MinNumericalJitter, safemath.DefaultRegularizationEpsilon))
	if err != nil {
		return nil, nil, 0, 0, false, fmt.Errorf("safe mahalanobis failed: %w", err)
	}
	mahal := math.Sqrt(maxFloat64(mahal2, 0))
	gate := 6.5
	if float64(m) > 4 {
		gate = 7.0
	}
	accepted := mahal <= gate

	nis := mahal2 / math.Max(float64(m), 1)
	if !accepted {
		return x, p, nis, mahal, false, nil
	}

	k := matMul(pxz, sInv)
	xPost := vecAdd(x, matVecMul(k, innov))
	pPost := matSub(p, matMul(matMul(k, s), matTranspose(k)))
	pPost = enforcePositiveDiagonal(symmetrize(pPost), maxFloat64(cfg.MinNumericalJitter, 1e-9))
	return xPost, pPost, nis, mahal, true, nil
}

// GenerateSigmaPoints creates sigma points for UKF using Cholesky decomposition.
func GenerateSigmaPoints(x []float64, pcov [][]float64, cfg UKFConfig) ([][]float64, []float64, []float64, error) {
	n := len(x)
	n_sigma := 2*n + 1
	lambda := cfg.Alpha*cfg.Alpha*(float64(n)+cfg.Kappa) - float64(n)

	wm := make([]float64, n_sigma)
	wc := make([]float64, n_sigma)
	wm[0] = lambda / (float64(n) + lambda)
	wc[0] = wm[0] + (1.0 - cfg.Alpha*cfg.Alpha + cfg.Beta)
	for i := 1; i < n_sigma; i++ {
		wm[i] = 1.0 / (2.0 * (float64(n) + lambda))
		wc[i] = wm[i]
	}

	pSym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			pSym.SetSym(i, j, pcov[i][j])
		}
	}
	var chol mat.Cholesky
	if ok := chol.Factorize(pSym); !ok {
		return nil, nil, nil, fmt.Errorf("covariance matrix is not positive definite")
	}
	var l mat.TriDense
	chol.LTo(&l)

	sigma := make([][]float64, n_sigma)
	sigma[0] = make([]float64, n)
	copy(sigma[0], x)

	scale := math.Sqrt(float64(n) + lambda)
	for i := 0; i < n; i++ {
		sigma[i+1] = make([]float64, n)
		sigma[i+n+1] = make([]float64, n)
		for j := 0; j < n; j++ {
			val := l.At(j, i) * scale
			sigma[i+1][j] = x[j] + val
			sigma[i+n+1][j] = x[j] - val
		}
	}
	return sigma, wm, wc, nil
}

// QuadraticForm computes v^T * A * v.
func QuadraticForm(v []float64, a [][]float64) float64 {
	av := matVecMul(a, v)
	s := 0.0
	for i := range v {
		s += v[i] * av[i]
	}
	return s
}

// UnscentedMeanCov computes weighted mean and covariance from sigma points.
func UnscentedMeanCov(sigma [][]float64, wm, wc []float64) ([]float64, [][]float64) {
	n := len(sigma[0])
	x := make([]float64, n)
	for i := range sigma {
		for j := 0; j < n; j++ {
			x[j] += wm[i] * sigma[i][j]
		}
	}
	p := safemath.ZeroMatrix(n, n)
	for i := range sigma {
		dx := vecSub(sigma[i], x)
		p = matAdd(p, outerScaled(dx, dx, wc[i]))
	}
	return x, p
}

func propagateSigmaPoint(s []float64, dt float64, jd float64) []float64 {
	params := orbital.PhysicsParams{
		Mass:     2500,
		DragArea: 15.0,
		SRPArea:  15.0,
		Cd:       2.2,
		Cr:       1.2,
	}
	return orbital.PropagateCartesianState(s, dt, jd, params)
}

func buildODSummary(epochs []ODEpoch, nis []float64, mahal []float64, traces []float64, accepted int) ODSummary {
	meanNIS := 0.0
	p95NIS := 0.0
	meanMah := 0.0
	traceMean := 0.0
	if len(nis) > 0 {
		meanNIS = stat.Mean(nis, nil)
		nisCopy := append([]float64(nil), nis...)
		sort.Float64s(nisCopy)
		idx := int(0.95 * float64(len(nisCopy)-1))
		if idx < 0 {
			idx = 0
		}
		p95NIS = nisCopy[idx]
	}
	if len(mahal) > 0 {
		meanMah = stat.Mean(mahal, nil)
	}
	if len(traces) > 0 {
		traceMean = stat.Mean(traces, nil)
	}
	consistency := meanNIS > 0.25 && meanNIS < 3.5
	return ODSummary{
		Epochs:                  len(epochs),
		AcceptedEpochs:          accepted,
		AcceptanceRatio:         float64(accepted) / math.Max(float64(len(epochs)), 1),
		MeanNIS:                 meanNIS,
		P95NIS:                  p95NIS,
		MeanMahalanobis:         meanMah,
		CovarianceTraceMean:     traceMean,
		InnovationConsistencyOK: consistency,
	}
}
