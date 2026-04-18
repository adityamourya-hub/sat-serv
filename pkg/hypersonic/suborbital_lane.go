package hypersonic

import (
	"fmt"
	"math"
	"sync"

	"ssis/pkg/fusion"
	safemath "ssis/pkg/math"
	"ssis/pkg/orbital"
)

type SubOrbitalLaneConfig struct {
	HorizonSec  float64
	BetaInitial float64
	LDInitial   float64
	MaxHistory  int
}

type SubOrbitalLane struct {
	ObjectID int
	mu       sync.RWMutex
	Config   SubOrbitalLaneConfig

	LastUpdateUnix int64
	State          []float64   // 8-state vector
	Covariance     [][]float64 // 8x8 matrix

	History []HypersonicState
	IPP     *ImpactArea
}

func NewSubOrbitalLane(objectID int, initial orbital.State, config SubOrbitalLaneConfig) *SubOrbitalLane {
	if config.HorizonSec <= 0 {
		config.HorizonSec = 90.0
	}
	if config.BetaInitial <= 0 {
		config.BetaInitial = 250.0 // 250 kg/m^2 typical
	}
	if config.MaxHistory <= 0 {
		config.MaxHistory = 20
	}

	x := make([]float64, 8)
	copy(x, []float64{initial.X, initial.Y, initial.Z, initial.VX, initial.VY, initial.VZ, config.BetaInitial, config.LDInitial})

	p := safemath.ZeroMatrix(8, 8)
	for i := 0; i < 6; i++ {
		p[i][i] = 1.0 // 1 km / 1 km/s initial uncertainty
	}
	p[6][6] = 2500.0 // Beta uncertainty
	p[7][7] = 0.25   // L/D uncertainty (0.5 sigma)

	return &SubOrbitalLane{
		ObjectID:   objectID,
		Config:     config,
		State:      x,
		Covariance: p,
		History:    make([]HypersonicState, 0),
	}
}

// ProcessObservation performs a 1 Hz UKF update for the hypersonic lane.
func (l *SubOrbitalLane) ProcessObservation(meas fusion.RadarMeasurement, tsUnix int64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	ts := tsUnix
	dt := float64(ts - l.LastUpdateUnix)
	if l.LastUpdateUnix == 0 {
		dt = 0
	}
	l.LastUpdateUnix = ts

	// 1. UKF Prediction
	cfg := fusion.DefaultUKFConfig()
	sigma, wm, wc, err := fusion.GenerateSigmaPoints(l.State, l.Covariance, cfg)
	if err != nil {
		return err
	}

	jd := float64(ts)/86400.0 + 2440587.5
	predSigma := make([][]float64, len(sigma))
	for i := range sigma {
		if dt > 0 {
			predSigma[i] = HypersonicPropagator(sigma[i], dt, jd)
		} else {
			predSigma[i] = sigma[i]
		}
	}

	xPred, pPred := fusion.UnscentedMeanCov(predSigma, wm, wc)

	// Add process noise (SNC + Random Walk for aero)
	q := safemath.ZeroMatrix(8, 8)
	// BuildSNCCovariance should be in fusion or maneuver? It's in fusion.
	// Wait, I should probably check if BuildSNCCovariance is exported in fusion.
	// Actually, I saw it in ukf.go (unexported). Let's check.
	// Actually, I moved BuildSNCCovariance to a separate file snc.go (exported).
	qSNC := fusion.BuildSNCCovariance(dt, fusion.DefaultSNCConfig(), 0)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			q[i][j] = qSNC[i][j]
		}
	}
	q[6][6] = 10.0 * dt  // Beta random walk
	q[7][7] = 0.005 * dt // L/D random walk
	pPred = safemath.MatrixAdd(pPred, q)

	// 2. UKF Update
	z := fusion.RadarMeasurementVector(meas)
	r := fusion.RadarMeasurementNoise(meas)
	siteID := meas.SiteID
	h := func(s []float64) []float64 { return fusion.RadarMeasurementFromState(s[:6], siteID, jd) }

	xPost, pPost, _, _, ok, err := fusion.UKFUpdateWithGating(xPred, pPred, z, r, h, cfg)
	if err != nil {
		return err
	}

	if ok {
		l.State = xPost
		l.Covariance = pPost
	} else {
		l.State = xPred
		l.Covariance = pPred
	}

	// 3. Update History
	l.History = append(l.History, VectorToHypersonicState(l.State))
	if len(l.History) > l.Config.MaxHistory {
		l.History = l.History[1:]
	}

	// 4. Update IPP
	ipp := l.predictImpactAtHorizon(l.Config.HorizonSec, jd)
	l.IPP = &ipp

	return nil
}

func (l *SubOrbitalLane) predictImpactAtHorizon(horizon float64, jd float64) ImpactArea {
	y := make([]float64, 8)
	copy(y, l.State)

	// Fast integration (fixed step 1.0s for IPP)
	t := 0.0
	for t < horizon {
		y = HypersonicPropagator(y, 1.0, jd+t/86400.0)
		t += 1.0
		r := math.Sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])
		if r-orbital.ER <= 0 {
			break
		}
	}

	rFinal := math.Sqrt(y[0]*y[0] + y[1]*y[1] + y[2]*y[2])
	if rFinal < 1e-9 {
		rFinal = orbital.ER
	}
	lat := math.Asin(y[2]/rFinal) * 180.0 / math.Pi
	lon := math.Atan2(y[1], y[0]) * 180.0 / math.Pi

	return ImpactArea{
		LatDeg:      lat,
		LonDeg:      lon,
		RadiusKm:    math.Sqrt(fusion.MatrixTrace(l.Covariance[:3])), // Proxy for uncertainty
		MajorAxisKm: 5.0,
		MinorAxisKm: 1.0,
	}
}

func (l *SubOrbitalLane) GetStatus() string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.IPP == nil {
		return "Initializing..."
	}
	return fmt.Sprintf("Target %d: Alt=%.1fkm, Beta=%.1f, L/D=%.2f -> IPP: %.3fN, %.3fE",
		l.ObjectID, (math.Sqrt(l.State[0]*l.State[0]+l.State[1]*l.State[1]+l.State[2]*l.State[2]) - orbital.ER),
		l.State[6], l.State[7], l.IPP.LatDeg, l.IPP.LonDeg)
}
