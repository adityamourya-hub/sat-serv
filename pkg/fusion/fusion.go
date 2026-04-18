package fusion

import (
	"fmt"
	"math"
	"time"

	engcrypto "ssis/pkg/crypto"
	safemath "ssis/pkg/math"
	"ssis/pkg/physics"
)

type UKFConfig struct {
	Alpha                     float64 `json:"alpha"`
	Beta                      float64 `json:"beta"`
	Kappa                     float64 `json:"kappa"`
	ProcessNoiseAccelKmSec2   float64 `json:"process_noise_accel"`
	SNCGaussMarkovTauSec      float64 `json:"snc_tau_sec"`
	MinNumericalJitter        float64 `json:"min_numerical_jitter"`
}

func DefaultUKFConfig() UKFConfig {
	return UKFConfig{
		Alpha:                   0.001,
		Beta:                    2.0,
		Kappa:                   0.0,
		ProcessNoiseAccelKmSec2: 1e-9,
		SNCGaussMarkovTauSec:    1800,
		MinNumericalJitter:      1e-9,
	}
}

type RadarMeasurement struct {
	SiteID         int     `json:"site_id"`
	TimestampUnix  int64   `json:"timestamp_unix"`
	RangeKm        float64 `json:"range_km"`
	RangeRateKmS   float64 `json:"range_rate_km_s"`
	AzimuthDeg     float64 `json:"azimuth_deg"`
	ElevationDeg   float64 `json:"elevation_deg"`
	RangeStdKm     float64 `json:"range_std_km"`
	RangeRateStd   float64 `json:"range_rate_std"`
	AngleStdDeg    float64 `json:"angle_std_deg"`
}

type TLEMeasurement struct {
	State         physics.State `json:"state"`
	PositionStdKm float64       `json:"position_std_km"`
	VelocityStdKm float64       `json:"velocity_std_km"`
}

type FusionInput struct {
	ObjectID         int              `json:"object_id"`
	EventType        string           `json:"event_type"`
	PriorState       physics.State    `json:"prior_state"`
	PriorCovariance  [][]float64       `json:"prior_covariance"`
	DeltaTimeSec     float64          `json:"delta_time_sec"`
	RadarMeasurement *RadarMeasurement `json:"radar_measurement,omitempty"`
	TLEMeasurement   *TLEMeasurement   `json:"tle_measurement,omitempty"`
	Signer           *engcrypto.Signer `json:"-"`
}

type Ellipsoid3Sigma struct {
	AxisXKm float64 `json:"axis_x_km"`
	AxisYKm float64 `json:"axis_y_km"`
	AxisZKm float64 `json:"axis_z_km"`
}

type FusionResult struct {
	ObjectID    int              `json:"object_id"`
	State       physics.State    `json:"state"`
	Covariance  [][]float64      `json:"covariance"`
	ThreeSigma  Ellipsoid3Sigma  `json:"three_sigma"`
	SensorsUsed []string         `json:"sensors_used"`
	SignedAlert *UncertaintyAlert `json:"signed_alert,omitempty"`
}

type UncertaintyAlert struct {
	ObjectID       int             `json:"object_id"`
	AlertType      string          `json:"alert_type"`
	TimestampUTC   string          `json:"timestamp_utc"`
	State          physics.State   `json:"state"`
	ThreeSigma     Ellipsoid3Sigma `json:"three_sigma"`
	Signature      string          `json:"signature"`
	SignatureValid bool            `json:"signature_valid"`
	PublicKeyHex   string          `json:"public_key_hex"`
}

type RadarSite struct {
	ID     int
	LatDeg float64
	LonDeg float64
	AltKm  float64
}

var radarSiteRegistry = map[int]RadarSite{
	101: {ID: 101, LatDeg: 13.0, LonDeg: 77.0, AltKm: 0.9}, // Bengaluru
	102: {ID: 102, LatDeg: 28.6, LonDeg: 77.2, AltKm: 0.2}, // Delhi
}

func FuseWithUKF(input FusionInput, cfg UKFConfig) (FusionResult, error) {
	fmt.Printf("IAF_SDA: Fusing data for Object %d (dt=%.1fs)\n", input.ObjectID, input.DeltaTimeSec)

	x := physics.StateToVector(input.PriorState)
	p := input.PriorCovariance
	if len(p) != 6 {
		p = safemath.ZeroMatrix(6, 6)
		for i := 0; i < 6; i++ {
			p[i][i] = 1.0
		}
	}

	// 1. Prediction (Using Physics Core Numerical Propagator)
	jd := float64(time.Now().Unix())/86400.0 + 2440587.5
	if input.DeltaTimeSec > 0 {
		x = physics.PropagateCartesianState(x, input.DeltaTimeSec, jd, physics.PhysicsParams{})
		// Propagate covariance with SNC
		q := buildSNCCovariance(input.DeltaTimeSec, cfg)
		p = safemath.MatrixAdd(p, q)
	}

	res := FusionResult{
		ObjectID:    input.ObjectID,
		SensorsUsed: make([]string, 0),
	}

	// 2. Multi-Sensor Update
	if input.RadarMeasurement != nil {
		z := RadarMeasurementVector(*input.RadarMeasurement)
		r := RadarMeasurementNoise(*input.RadarMeasurement)
		siteID := input.RadarMeasurement.SiteID
		h := func(s []float64) []float64 { return RadarMeasurementFromState(s, siteID, jd) }

		xNew, pNew, _, _, ok, err := UKFUpdateWithGating(x, p, z, r, h, cfg)
		if err == nil && ok {
			x, p = xNew, pNew
			res.SensorsUsed = append(res.SensorsUsed, "radar")
		}
	}

	if input.TLEMeasurement != nil {
		z := TLEMeasurementVector(*input.TLEMeasurement)
		r := TLEMeasurementNoise(*input.TLEMeasurement)
		h := func(s []float64) []float64 { return []float64{s[0], s[1], s[2], s[3], s[4], s[5]} }

		xNew, pNew, _, _, ok, err := UKFUpdateWithGating(x, p, z, r, h, cfg)
		if err == nil && ok {
			x, p = xNew, pNew
			res.SensorsUsed = append(res.SensorsUsed, "tle")
		}
	}

	res.State = physics.VectorToState(x)
	res.Covariance = p
	res.ThreeSigma = ComputeThreeSigmaEllipsoid(p)

	if input.Signer != nil {
		res.SignedAlert = BuildSignedUncertaintyAlert(input.ObjectID, input.EventType, res.State, p, res.ThreeSigma, input.Signer)
	}

	return res, nil
}

func PropagateWithUKF(initial physics.State, initialCov [][]float64, stepSec float64, steps int, cfg UKFConfig, pinn physics.StateRefiner, trans physics.HistoryRefiner, signer *engcrypto.Signer, objectID int, eventType string) ([]FusionResult, error) {
	currState := initial
	currCov := initialCov
	results := make([]FusionResult, 0, steps)
	startTime := time.Now()

	for i := 0; i < steps; i++ {
		input := FusionInput{
			ObjectID:        objectID,
			EventType:       eventType,
			PriorState:      currState,
			PriorCovariance: currCov,
			DeltaTimeSec:    stepSec,
			Signer:          signer,
		}
		res, err := FuseWithUKF(input, cfg)
		if err != nil {
			return nil, err
		}
		currState = res.State
		currCov = res.Covariance
		results = append(results, res)
	}
	fmt.Printf("IAF_SDA: UKF propagation complete in %v\n", time.Since(startTime))
	return results, nil
}

func BuildSignedUncertaintyAlert(objectID int, alertType string, state physics.State, cov [][]float64, ts Ellipsoid3Sigma, signer *engcrypto.Signer) *UncertaintyAlert {
	alert := &UncertaintyAlert{
		ObjectID:     objectID,
		AlertType:    alertType,
		TimestampUTC: time.Now().UTC().Format(time.RFC3339),
		State:        state,
		ThreeSigma:   ts,
		PublicKeyHex: signer.PublicKeyHex(),
	}
	tsUnix := time.Now().Unix()
	alert.Signature = signer.GenerateProof(objectID, alertType, tsUnix)
	payload := fmt.Sprintf("%d:%s:%d", objectID, alertType, tsUnix)
	alert.SignatureValid = signer.Verify([]byte(payload), alert.Signature)
	return alert
}

func buildSNCCovariance(dtSec float64, cfg UKFConfig) [][]float64 {
	q := safemath.ZeroMatrix(6, 6)
	if dtSec <= 0 {
		return q
	}
	qc := cfg.ProcessNoiseAccelKmSec2 * cfg.ProcessNoiseAccelKmSec2
	tau := cfg.SNCGaussMarkovTauSec
	if tau <= 0 {
		tau = 1
	}
	gm := 1.0 - math.Exp(-2.0*dtSec/tau)
	if gm < 1e-6 {
		gm = 1e-6
	}
	q11 := qc * gm * dtSec * dtSec * dtSec / 3.0
	q12 := qc * gm * dtSec * dtSec / 2.0
	q22 := qc * gm * dtSec

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

func ComputeThreeSigmaEllipsoid(p [][]float64) Ellipsoid3Sigma {
	return Ellipsoid3Sigma{
		AxisXKm: 3 * math.Sqrt(safemath.MaxFloat64(p[0][0], 0)),
		AxisYKm: 3 * math.Sqrt(safemath.MaxFloat64(p[1][1], 0)),
		AxisZKm: 3 * math.Sqrt(safemath.MaxFloat64(p[2][2], 0)),
	}
}

func RadarMeasurementVector(m RadarMeasurement) []float64 {
	return []float64{m.RangeKm, m.RangeRateKmS, m.AzimuthDeg * physics.RAD_PER_DEG, m.ElevationDeg * physics.RAD_PER_DEG}
}

func RadarMeasurementNoise(m RadarMeasurement) [][]float64 {
	rStd := m.RangeStdKm
	if rStd <= 0 {
		rStd = 0.05
	}
	rrStd := m.RangeRateStd
	if rrStd <= 0 {
		rrStd = 0.005
	}
	aStdRad := m.AngleStdDeg * physics.RAD_PER_DEG
	if aStdRad <= 0 {
		aStdRad = 0.08 * physics.RAD_PER_DEG
	}
	r := safemath.ZeroMatrix(4, 4)
	r[0][0] = rStd * rStd
	r[1][1] = rrStd * rrStd
	r[2][2] = aStdRad * aStdRad
	r[3][3] = aStdRad * aStdRad
	return r
}

func RadarMeasurementFromState(x []float64, siteID int, jd float64) []float64 {
	px, py, pz := x[0], x[1], x[2]
	vx, vy, vz := x[3], x[4], x[5]

	site, ok := radarSiteRegistry[siteID]
	if !ok {
		rng := math.Sqrt(px*px + py*py + pz*pz)
		if rng < 1e-9 {
			rng = 1e-9
		}
		rangeRate := (px*vx + py*vy + pz*vz) / rng
		az := math.Atan2(py, px)
		el := math.Asin(safemath.Clamp(pz/rng, -1, 1))
		return []float64{rng, rangeRate, az, el}
	}

	gst := physics.GreenwichSiderealTime(jd)
	phi := site.LatDeg * physics.RAD_PER_DEG
	lambda := site.LonDeg * physics.RAD_PER_DEG + gst
	cosPhi := math.Cos(phi)
	sinPhi := math.Sin(phi)
	cosLambda := math.Cos(lambda)
	sinLambda := math.Sin(lambda)

	rs := physics.RE_KM + site.AltKm
	sx := rs * cosPhi * cosLambda
	sy := rs * cosPhi * sinLambda
	sz := rs * sinPhi

	rx, ry, rz := px-sx, py-sy, pz-sz
	rvx, rvy, rvz := vx, vy, vz 

	rng := math.Sqrt(rx*rx + ry*ry + rz*rz)
	if rng < 1e-9 {
		rng = 1e-9
	}
	rangeRate := (rx*rvx + ry*rvy + rz*rvz) / rng

	s := (sinPhi*cosLambda)*rx + (sinPhi*sinLambda)*ry - cosPhi*rz
	e := (-sinLambda)*rx + cosLambda*ry
	z := (cosPhi*cosLambda)*rx + (cosPhi*sinLambda)*ry + sinPhi*rz

	az := math.Atan2(e, -s)
	if az < 0 {
		az += 2.0 * math.Pi
	}
	el := math.Asin(safemath.Clamp(z/rng, -1, 1))

	return []float64{rng, rangeRate, az, el}
}

func TLEMeasurementVector(m TLEMeasurement) []float64 {
	s := physics.StateToVector(m.State)
	return []float64{s[0], s[1], s[2], s[3], s[4], s[5]}
}

func TLEMeasurementNoise(m TLEMeasurement) [][]float64 {
	pStd := m.PositionStdKm
	if pStd <= 0 {
		pStd = 1.0
	}
	vStd := m.VelocityStdKm
	if vStd <= 0 {
		vStd = 0.01
	}
	r := safemath.ZeroMatrix(6, 6)
	r[0][0] = pStd * pStd
	r[1][1] = pStd * pStd
	r[2][2] = pStd * pStd
	r[3][3] = vStd * vStd
	r[4][4] = vStd * vStd
	r[5][5] = vStd * vStd
	return r
}
