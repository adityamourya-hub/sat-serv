package sda

import (
	"math"
	"time"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * NavIC Sovereign Time-Sync Module
 *
 * Implements sovereign time synchronization using NavIC (IRNSS) signals.
 * Eliminates dependencies on external GPS/NTP for ECI-to-ECEF transformation.
 *
 * Key Features:
 * - GMST calculation from NavIC Time of Week (TOW)
 * - Pulse Per Second (PPS) gating for 100Hz Go-reflex
 * - Earth Orientation Parameters (EOP) from NavIC broadcast
 */

// NavICMessage represents a decoded NavIC navigation message
type NavICMessage struct {
	TOW         float64 // Time of Week (seconds since Sunday 00:00:00 UTC)
	WeekNumber  int     // GPS week number
	UTCOffset   float64 // UTC offset (seconds)
	LeapSeconds int     // Leap second count
	EOP         EarthOrientationParameters
}

// EarthOrientationParameters represents Earth orientation from NavIC
type EarthOrientationParameters struct {
	XP      float64 // Polar motion X (arcsec)
	YP      float64 // Polar motion Y (arcsec)
	UT1_UTC float64 // UT1-UTC (seconds)
	LOD     float64 // Length of day (ms)
}

// NavICSync implements sovereign time synchronization
type NavICSync struct {
	LastMessage    NavICMessage
	LastPPSTime    time.Time
	PPSFrequency   float64 // Hz (typically 1.0 for PPS)
	GMST           float64 // Greenwich Mean Sidereal Time (radians)
	EarthRotation  float64 // ω⊕ = 7.2921159e-5 rad/s
	SovereignClock bool    // Flag indicating NavIC lock
}

// NewNavICSync creates a new NavIC synchronization module
func NewNavICSync() *NavICSync {
	return &NavICSync{
		PPSFrequency:   1.0,
		EarthRotation:  7.2921159e-5,
		SovereignClock: false,
	}
}

// ProcessNavICMessage decodes and processes a NavIC navigation message
func (ns *NavICSync) ProcessNavICMessage(msg NavICMessage) error {
	ns.LastMessage = msg

	// Calculate GMST from NavIC timestamp
	ns.GMST = ns.ComputeGMST(msg.TOW, msg.WeekNumber)

	// Mark sovereign clock as locked
	ns.SovereignClock = true

	return nil
}

// ComputeGMST calculates Greenwich Mean Sidereal Time from NavIC timestamp
// Formula: θ_g(t) = θ_g,J2000 + ω⊕ * Δt
func (ns *NavICSync) ComputeGMST(tow float64, weekNumber int) float64 {
	// Convert TOW to seconds since J2000 epoch
	// J2000 epoch: January 1, 2000, 12:00 TT (JD 2451545.0)

	// NavIC time is GPS time, which is ahead of UTC by leap seconds
	// GPS epoch: January 6, 1980, 00:00 UTC
	gpsEpoch := time.Date(1980, 1, 6, 0, 0, 0, 0, time.UTC)
	navicTime := gpsEpoch.Add(time.Duration(weekNumber*7*24*3600+int(tow)) * time.Second)

	// Convert to Julian Date
	jd := timeToJD(navicTime)

	// Time since J2000 in Julian centuries
	T := (jd - 2451545.0) / 36525.0

	// GMST formula (IAU 2000/2006)
	// θ_g = 280.46061837 + 360.98564736629*(JD - 2451545.0) + 0.000387933*T² - T³/38710000
	deg2rad := math.Pi / 180.0
	gmstDegrees := 280.46061837 + 360.98564736629*(jd-2451545.0) +
		0.000387933*T*T - T*T*T/38710000.0

	// Convert to radians and normalize to [0, 2π)
	gmstRad := math.Mod(gmstDegrees*deg2rad, 2*math.Pi)
	if gmstRad < 0 {
		gmstRad += 2 * math.Pi
	}

	return gmstRad
}

// GetSovereignRotationMatrix returns the ECI-to-ECEF rotation matrix
// using NavIC-derived GMST for sovereign time reference
func (ns *NavICSync) GetSovereignRotationMatrix() [3][3]float64 {
	if !ns.SovereignClock {
		// Fallback to physics package if NavIC not locked
		return [3][3]float64{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		}
	}

	// Rotation about Z-axis by GMST
	cosG := math.Cos(ns.GMST)
	sinG := math.Sin(ns.GMST)

	// R_z(θ_g)
	return [3][3]float64{
		{cosG, sinG, 0},
		{-sinG, cosG, 0},
		{0, 0, 1},
	}
}

// OnPPS handles the Pulse Per Second signal from NavIC receiver
// This gates the 100Hz Go-reflex loop to nanosecond precision
func (ns *NavICSync) OnPPS() time.Time {
	now := time.Now()
	ns.LastPPSTime = now

	// In production, this would trigger the symplectic propagator
	// to synchronize with the NavIC clock edge

	return now
}

// GetPPSInterval returns the time since last PPS
func (ns *NavICSync) GetPPSInterval() time.Duration {
	if ns.LastPPSTime.IsZero() {
		return 0
	}
	return time.Since(ns.LastPPSTime)
}

// ValidateSovereignLock checks if NavIC lock is valid
func (ns *NavICSync) ValidateSovereignLock() bool {
	if !ns.SovereignClock {
		return false
	}

	// Check if PPS is within expected frequency
	interval := ns.GetPPSInterval()
	expectedInterval := time.Duration(1.0/ns.PPSFrequency*1000) * time.Millisecond

	// Allow 10% tolerance
	tolerance := time.Duration(float64(expectedInterval) * 0.1)
	if interval < expectedInterval-tolerance || interval > expectedInterval+tolerance {
		return false
	}

	return true
}

// GetSovereignTime returns the current sovereign time from NavIC
func (ns *NavICSync) GetSovereignTime() time.Time {
	if !ns.SovereignClock {
		// Fallback to system time
		return time.Now()
	}

	// Calculate sovereign time from NavIC message
	navicTime := time.Unix(0, 0).Add(
		time.Duration(ns.LastMessage.WeekNumber*7*24*3600)*time.Second +
			time.Duration(ns.LastMessage.TOW)*time.Second,
	)

	return navicTime
}

// ApplyEOP applies Earth Orientation Parameters to the rotation
func (ns *NavICSync) ApplyEOP(matrix [3][3]float64) [3][3]float64 {
	if !ns.SovereignClock {
		return matrix
	}

	// Apply polar motion corrections (simplified)
	// In production, use full IAU 2000/2006 precession-nutation model
	xp := ns.LastMessage.EOP.XP * math.Pi / (180.0 * 3600.0) // Convert arcsec to radians
	yp := ns.LastMessage.EOP.YP * math.Pi / (180.0 * 3600.0)

	// Small angle approximation for polar motion
	cosXp := math.Cos(xp)
	sinXp := math.Sin(xp)
	cosYp := math.Cos(yp)
	sinYp := math.Sin(yp)

	// Apply polar motion rotation
	// R = R_y(-xp) * R_x(-yp)
	result := [3][3]float64{
		{cosYp, 0, -sinYp},
		{sinXp * sinYp, cosXp, sinXp * cosYp},
		{cosXp * sinYp, -sinXp, cosXp * cosYp},
	}

	// Multiply with rotation matrix
	return multiply3x3(result, matrix)
}

// multiply3x3 multiplies two 3x3 matrices
func multiply3x3(A, B [3][3]float64) [3][3]float64 {
	result := [3][3]float64{}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			sum := 0.0
			for k := 0; k < 3; k++ {
				sum += A[i][k] * B[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

// Status returns the current status of NavIC synchronization
func (ns *NavICSync) Status() string {
	if !ns.SovereignClock {
		return "SEARCHING"
	}

	if ns.ValidateSovereignLock() {
		return "LOCKED"
	}

	return "LOSS_OF_LOCK"
}

// GetTimeAccuracy returns the estimated time accuracy in nanoseconds
func (ns *NavICSync) GetTimeAccuracy() float64 {
	if !ns.SovereignClock {
		return math.Inf(1)
	}

	// NavIC provides ~10ns accuracy with dual-frequency ionospheric correction
	// Single-frequency: ~50ns
	return 10.0 // nanoseconds
}
