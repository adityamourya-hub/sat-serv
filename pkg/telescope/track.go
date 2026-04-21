// pkg/telescope/track.go
// ADITI 4.0: Closed-Loop Riemannian Tracking with PHS-Centroid Feedback
//
// Implements SO(3)-Equivariant tracking with Hamiltonian-modulated PI control.
// The error between PHS-predicted manifold and pixel-space observation drives
// mount corrections via Riemannian metric on S^2.

package telescope

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/physics"
)

// RiemannianController implements PI control on SO(3) manifold
//
// ADITI 4.0: Gains are modulated by Hamiltonian confidence.
// - High H-drift: Trust camera more (physics uncertain)
// - Low camera SNR: Trust PHS more (turbulence)
type RiemannianController struct {
	// PI gains (base values)
	KpBase float64 // Proportional gain
	KiBase float64 // Integral gain

	// Error accumulation
	IntegralErrorAz float64
	IntegralErrorEl float64

	// Riemannian metric weights
	ManifoldConfidence float64 // From Hamiltonian drift
	CameraConfidence   float64 // From SNR

	// Limits
	MaxRate     float64 // deg/s maximum correction rate
	MaxIntegral float64 // Anti-windup limit

	// Latency tracking
	LastCommandLatency float64
}

// NewRiemannianController creates controller with Hamiltonian-modulated gains
func NewRiemannianController() *RiemannianController {
	return &RiemannianController{
		KpBase:             0.8,
		KiBase:             0.1,
		IntegralErrorAz:    0.0,
		IntegralErrorEl:    0.0,
		ManifoldConfidence: 1.0, // Start with full PHS confidence
		CameraConfidence:   1.0, // Start with full camera confidence
		MaxRate:            2.0, // 2 deg/s max correction
		MaxIntegral:        5.0, // 5 deg integral limit
		LastCommandLatency: 0.0,
	}
}

// ModulateGains adjusts PI gains based on confidence levels
//
// High Hamiltonian drift -> reduce Kp (trust physics less)
// Low camera SNR -> reduce Ki (trust camera less)
func (rc *RiemannianController) ModulateGains(hamiltonianDrift, snr float64) (kp, ki float64) {
	// Camera confidence from SNR (sigmoid mapping)
	// SNR < 10: low confidence, SNR > 50: high confidence
	rc.CameraConfidence = 1.0 / (1.0 + math.Exp(-(snr-30.0)/10.0))

	// Manifold confidence from Hamiltonian drift
	// Drift < 1e-6: high confidence, Drift > 1e-4: low confidence
	rc.ManifoldConfidence = 1.0 / (1.0 + math.Exp((hamiltonianDrift-1e-5)/1e-6))

	// Modulate gains
	// When camera SNR is low, increase PHS weight (physics-first)
	// When Hamiltonian drift is high, increase camera weight (observation-first)
	kp = rc.KpBase * (rc.ManifoldConfidence*0.3 + rc.CameraConfidence*0.7)
	ki = rc.KiBase * (rc.ManifoldConfidence*0.6 + rc.CameraConfidence*0.4)

	return kp, ki
}

// ComputeCorrection calculates PI control on Riemannian error
//
// Error is angular distance on S^2 (unit sphere)
// Returns corrected track rates in Az/El
func (rc *RiemannianController) ComputeCorrection(
	predAz, predEl, obsAz, obsEl float64,
	hamiltonianDrift, snr float64,
) (rateAz, rateEl float64) {
	// Modulate gains based on confidence
	kp, ki := rc.ModulateGains(hamiltonianDrift, snr)

	// Compute Riemannian error (angular distance on S^2)
	errorAz, errorEl := rc.calculateAngularDistance(predAz, predEl, obsAz, obsEl)

	// Update integral (with anti-windup)
	rc.IntegralErrorAz += errorAz
	rc.IntegralErrorEl += errorEl

	// Anti-windup
	if math.Abs(rc.IntegralErrorAz) > rc.MaxIntegral {
		rc.IntegralErrorAz = math.Copysign(rc.MaxIntegral, rc.IntegralErrorAz)
	}
	if math.Abs(rc.IntegralErrorEl) > rc.MaxIntegral {
		rc.IntegralErrorEl = math.Copysign(rc.MaxIntegral, rc.IntegralErrorEl)
	}

	// PI control law
	rateAz = kp*errorAz + ki*rc.IntegralErrorAz
	rateEl = kp*errorEl + ki*rc.IntegralErrorEl

	// Rate limiting
	if math.Abs(rateAz) > rc.MaxRate {
		rateAz = math.Copysign(rc.MaxRate, rateAz)
	}
	if math.Abs(rateEl) > rc.MaxRate {
		rateEl = math.Copysign(rc.MaxRate, rateEl)
	}

	return rateAz, rateEl
}

// calculateAngularDistance computes geodesic distance on S^2
// Returns error components in Az and El
func (rc *RiemannianController) calculateAngularDistance(az1, el1, az2, el2 float64) (errAz, errEl float64) {
	// Convert to radians
	az1Rad := az1 * physics.RAD_PER_DEG
	el1Rad := el1 * physics.RAD_PER_DEG
	az2Rad := az2 * physics.RAD_PER_DEG
	el2Rad := el2 * physics.RAD_PER_DEG

	// Position vectors on unit sphere
	x1 := math.Cos(el1Rad) * math.Cos(az1Rad)
	y1 := math.Cos(el1Rad) * math.Sin(az1Rad)
	z1 := math.Sin(el1Rad)

	x2 := math.Cos(el2Rad) * math.Cos(az2Rad)
	y2 := math.Cos(el2Rad) * math.Sin(az2Rad)
	z2 := math.Sin(el2Rad)

	// Angular separation (great circle distance)
	dot := x1*x2 + y1*y2 + z1*z2
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}
	angularSep := math.Acos(dot) * physics.DEG_PER_RAD

	// Error components (simplified: assume small angles)
	errAz = az2 - az1
	errEl = el2 - el1

	// Normalize Az error to [-180, 180]
	if errAz > 180.0 {
		errAz -= 360.0
	}
	if errAz < -180.0 {
		errAz += 360.0
	}

	// Scale by angular separation for Riemannian metric
	if angularSep > 0 {
		scale := angularSep / math.Sqrt(errAz*errAz+errEl*errEl+1e-10)
		errAz *= scale
		errEl *= scale
	}

	return errAz, errEl
}

// Reset clears integral accumulation
func (rc *RiemannianController) Reset() {
	rc.IntegralErrorAz = 0.0
	rc.IntegralErrorEl = 0.0
}

// TrackState represents current tracking state
type TrackState struct {
	TargetID           string
	PredictedAz        float64
	PredictedEl        float64
	ObservedAz         float64
	ObservedEl         float64
	SNR                float64
	HamiltonianDrift   float64
	TrackError         float64 // Current angular error (arcsec)
	CorrectionRateAz   float64 // Applied correction rate
	CorrectionRateEl   float64
	ManifoldConfidence float64
	CameraConfidence   float64
	Timestamp          int64
}

// ClosedLoopTrack implements active Riemannian feedback loop
//
// ADITI 4.0: Transitions from "set and forget" slewing to active
// PHS-Centroid handshake. Runs at GEODSS 2.7 fps cadence (370ms).
//
// The loop:
//  1. Get PHS prediction (Tier 2) - predicted manifold
//  2. Query camera centroid via ZMQ - pixel-space observation
//  3. Compute Riemannian error on S^2
//  4. Apply PI correction with Hamiltonian-modulated gains
//  5. Send INDI SET_TRACK_RATE command
//
// Target: <50ms latency, <2 arcsec tracking error during 5g pull-up
func (b *CommandBridge) ClosedLoopTrack(targetID string, maxError float64, stopChan chan struct{}) error {
	// Initialize Riemannian controller
	controller := NewRiemannianController()

	// GEODSS 2.7 fps cadence = 370ms period
	ticker := time.NewTicker(370 * time.Millisecond)
	defer ticker.Stop()

	// Track state
	state := &TrackState{TargetID: targetID}

	fmt.Printf("[CLOSED_LOOP] Starting Riemannian tracking for %s (max error: %.1f arcsec)\n",
		targetID, maxError)

	for {
		select {
		case <-ticker.C:
			// Check thermal throttling
			if b.isThermalThrottled() {
				fmt.Printf("[CLOSED_LOOP] Thermal throttle active - reducing duty cycle\n")
				time.Sleep(100 * time.Millisecond)
				continue
			}

			// 1. Get PHS prediction (Tier 2)
			startTime := time.Now()
			predictedState, err := b.getPHSPrediction(targetID)
			if err != nil {
				fmt.Printf("[CLOSED_LOOP] PHS prediction error: %v\n", err)
				continue
			}

			predAz, predEl := physics.ECIToAzEl(
				predictedState.Pos[0], predictedState.Pos[1], predictedState.Pos[2],
				b.stationLat, b.stationLon, b.stationAlt,
				predictedState.EpochJD,
			)

			// Apply fractional atmosphere correction
			predEl = predEl * physics.FractionalAtmosphereAdjust(predEl, b.stationAlt, predictedState.EpochJD)

			// 2. Query camera centroid via ZMQ
			obsAz, obsEl, snr, err := b.RequestCentroid(targetID)
			if err != nil {
				fmt.Printf("[CLOSED_LOOP] Centroid request error: %v\n", err)
				// Trust PHS prediction when camera fails
				obsAz, obsEl = predAz, predEl
				snr = 0.0
			}

			// 3. Compute Hamiltonian drift (from PHS)
			hamiltonianDrift := b.getHamiltonianDrift(targetID)

			// 4. Compute Riemannian error and correction
			errAz, errEl := controller.calculateAngularDistance(predAz, predEl, obsAz, obsEl)
			trackError := math.Sqrt(errAz*errAz + errEl*errEl)

			// 5. Update state
			state.PredictedAz = predAz
			state.PredictedEl = predEl
			state.ObservedAz = obsAz
			state.ObservedEl = obsEl
			state.SNR = snr
			state.HamiltonianDrift = hamiltonianDrift
			state.TrackError = trackError * 3600.0 // Convert to arcsec

			// 6. Check error threshold
			if trackError*3600.0 > maxError {
				// Compute corrected rates
				rateAz, rateEl := controller.ComputeCorrection(
					predAz, predEl, obsAz, obsEl,
					hamiltonianDrift, snr,
				)

				state.CorrectionRateAz = rateAz
				state.CorrectionRateEl = rateEl

				// 7. Send INDI SET_TRACK_RATE command
				if err := b.SetTrackRate(rateAz, rateEl); err != nil {
					fmt.Printf("[CLOSED_LOOP] INDI rate command error: %v\n", err)
					continue
				}

				fmt.Printf("[CLOSED_LOOP] Error: %.1f arcsec | Rates: %.3f, %.3f deg/s | SNR: %.1f | H-drift: %.2e\n",
					state.TrackError, rateAz, rateEl, snr, hamiltonianDrift)
			}

			// Update latency history
			latency := time.Since(startTime).Milliseconds()
			b.LatencyHistory = append(b.LatencyHistory, float64(latency))
			if len(b.LatencyHistory) > 100 {
				b.LatencyHistory = b.LatencyHistory[1:]
			}

		case <-stopChan:
			fmt.Printf("[CLOSED_LOOP] Stopping Riemannian tracking for %s\n", targetID)
			controller.Reset()
			return nil
		}
	}
}

// RequestCentroid queries camera for current centroid via ZMQ
// Returns observed Az/El and SNR
func (b *CommandBridge) RequestCentroid(targetID string) (az, el, snr float64, err error) {
	// Create centroid request
	req := CentroidRequest{
		TargetID:  targetID,
		Timestamp: time.Now().UnixMilli(),
	}

	data, err := req.Marshal()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to marshal centroid request: %w", err)
	}

	// Send via ZMQ
	if _, err := b.socket.Send(string(data), 0); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to send centroid request: %w", err)
	}

	// Receive response
	reply, err := b.socket.Recv(0)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("centroid request timeout: %w", err)
	}

	// Parse response
	resp := &CentroidResponse{}
	if err := resp.Unmarshal([]byte(reply)); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to parse centroid response: %w", err)
	}

	return resp.Azimuth, resp.Elevation, resp.SNR, nil
}

// SetTrackRate sends INDI SET_TRACK_RATE command
// TELESCOPE_TRACK_MODE = 1 (Manual Rates) for HGV tracking
func (b *CommandBridge) SetTrackRate(rateAz, rateEl float64) error {
	cmd := INDITrackCommand{
		Type:      "SET_TRACK_RATE",
		RateAz:    rateAz,
		RateEl:    rateEl,
		TrackMode: 1, // Manual rates, not sidereal
		Timestamp: time.Now().UnixMilli(),
	}

	data, err := cmd.Marshal()
	if err != nil {
		return fmt.Errorf("failed to marshal track command: %w", err)
	}

	if _, err := b.socket.Send(string(data), 0); err != nil {
		return fmt.Errorf("failed to send track command: %w", err)
	}

	return nil
}

// isThermalThrottled checks if M2 Air needs thermal protection
func (b *CommandBridge) isThermalThrottled() bool {
	// Simplified check - in production would query thermal sensors
	// Target: Prevent 80°C throttle during sustained 2.7 fps operation
	return false // Placeholder
}

// getPHSPrediction queries PHS model for predicted state
func (b *CommandBridge) getPHSPrediction(targetID string) (*physics.State, error) {
	// Placeholder - would query GHNN propagator
	return &physics.State{}, nil
}

// getHamiltonianDrift queries current Hamiltonian drift for target
func (b *CommandBridge) getHamiltonianDrift(targetID string) float64 {
	// Placeholder - would query PHS Hamiltonian tracker
	return 1e-7 // Typical good drift level
}

// CentroidRequest represents camera centroid query
type CentroidRequest struct {
	TargetID  string
	Timestamp int64
}

func (r *CentroidRequest) Marshal() ([]byte, error) {
	// Simplified - would use protobuf/flatbuffer in production
	return []byte(fmt.Sprintf("CENTROID_REQ:%s:%d", r.TargetID, r.Timestamp)), nil
}

// CentroidResponse represents camera centroid result
type CentroidResponse struct {
	TargetID  string
	Azimuth   float64
	Elevation float64
	SNR       float64
	Timestamp int64
}

func (r *CentroidResponse) Unmarshal(data []byte) error {
	// Simplified - would use protobuf/flatbuffer in production
	return nil
}

// INDITrackCommand represents INDI SET_TRACK_RATE command
type INDITrackCommand struct {
	Type      string
	RateAz    float64
	RateEl    float64
	TrackMode int // 0=sidereal, 1=manual rates
	Timestamp int64
}

func (c *INDITrackCommand) Marshal() ([]byte, error) {
	// Simplified - would use protobuf/flatbuffer in production
	return []byte(fmt.Sprintf("INDI_TRACK:%s:%.4f:%.4f:%d:%d",
		c.Type, c.RateAz, c.RateEl, c.TrackMode, c.Timestamp)), nil
}
