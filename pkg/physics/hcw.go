// Package physics provides Hill-Clohessy-Wiltshire (HCW) equations
// for relative motion analysis in Proximity Operations (RPO)
package physics

import (
	"math"
)

// HCWState represents position and velocity in the RIC (Radial, In-track, Cross-track) frame
// RIC frame is centered on the reference satellite (asset)
type HCWState struct {
	Pos [3]float64 // x=Radial, y=In-track, z=Cross-track (km)
	Vel [3]float64 // Velocity components (km/s)
}

// PredictRelativePosition calculates where the RSO *should* be according to HCW equations
// given initial state, mean motion n (rad/s), and time t (seconds)
func PredictRelativePosition(initial HCWState, n float64, t float64) [3]float64 {
	x0, y0, z0 := initial.Pos[0], initial.Pos[1], initial.Pos[2]
	vx0, vy0, vz0 := initial.Vel[0], initial.Vel[1], initial.Vel[2]

	sinNt, cosNt := math.Sincos(n * t)

	// Radial (x) - HCW equation
	// x(t) = (4x0 + 2*vy0/n) - (3x0 + 2*vy0/n)*cos(nt) + (vx0/n)*sin(nt)
	x_t := (4*x0 + 2*vy0/n) - (3*x0+2*vy0/n)*cosNt + (vx0/n)*sinNt

	// In-track (y) - HCW equation
	// y(t) = 6*(n*x0 + vy0)*t + y0 - (2*vx0/n) + (3*x0 + 2*vy0/n)*2*sin(nt) + (2*vx0/n)*cos(nt)
	y_t := 6*(n*x0+vy0)*t + y0 - (2*vx0/n) + (3*x0+2*vy0/n)*2*sinNt + (2*vx0/n)*cosNt

	// Cross-track (z) - HCW equation
	// z(t) = z0*cos(nt) + (vz0/n)*sin(nt)
	z_t := z0*cosNt + (vz0/n)*sinNt

	return [3]float64{x_t, y_t, z_t}
}

// CalculateMeanMotion computes mean motion n from semi-major axis a
// n = sqrt(mu / a^3) where mu = 398600.4418 km^3/s^2 (Earth gravitational parameter)
func CalculateMeanMotion(a float64) float64 {
	const mu = 398600.4418 // Earth gravitational parameter (km^3/s^2)
	return math.Sqrt(mu / (a * a * a))
}

// IsRPOPattern checks if the observed track follows an HCW ellipse
// Returns true if average residual is below threshold (high-confidence RPO pattern)
func IsRPOPattern(observations []HCWState, n float64, timeStepSeconds float64) (bool, float64) {
	if len(observations) < 5 {
		return false, 0
	}

	initial := observations[0]
	var totalResidual float64

	for i := 1; i < len(observations); i++ {
		t := timeStepSeconds * float64(i)
		predictedPos := PredictRelativePosition(initial, n, t)
		
		// Calculate Euclidean distance between Predicted and Observed
		dx := observations[i].Pos[0] - predictedPos[0]
		dy := observations[i].Pos[1] - predictedPos[1]
		dz := observations[i].Pos[2] - predictedPos[2]
		
		residual := math.Sqrt(dx*dx + dy*dy + dz*dz)
		totalResidual += residual
	}

	avgResidual := totalResidual / float64(len(observations)-1)
	
	// If the average deviation is < 50 meters (0.05 km), it's a high-confidence RPO pattern
	return avgResidual < 0.05, avgResidual
}

// ComputeRICFrame converts ECI state to RIC (Radial-Intrack-Crosstrack) frame
// relative to a reference satellite
func ComputeRICFrame(
	targetPos, targetVel [3]float64, // Reference satellite state
	chaserPos, chaserVel [3]float64, // RSO state
) (HCWState, error) {
	// Calculate unit vectors for RIC frame
	// Radial (R): Points from Earth center to reference satellite
	rMag := math.Sqrt(targetPos[0]*targetPos[0] + targetPos[1]*targetPos[1] + targetPos[2]*targetPos[2])
	if rMag == 0 {
		return HCWState{}, nil
	}
	
	rUnit := [3]float64{
		targetPos[0] / rMag,
		targetPos[1] / rMag,
		targetPos[2] / rMag,
	}
	
	// Cross-track (C): R x V (normalized)
	h := [3]float64{
		targetPos[1]*targetVel[2] - targetPos[2]*targetVel[1],
		targetPos[2]*targetVel[0] - targetPos[0]*targetVel[2],
		targetPos[0]*targetVel[1] - targetPos[1]*targetVel[0],
	}
	hMag := math.Sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2])
	
	if hMag == 0 {
		return HCWState{}, nil
	}
	
	cUnit := [3]float64{
		h[0] / hMag,
		h[1] / hMag,
		h[2] / hMag,
	}
	
	// In-track (I): C x R
	iUnit := [3]float64{
		cUnit[1]*rUnit[2] - cUnit[2]*rUnit[1],
		cUnit[2]*rUnit[0] - cUnit[0]*rUnit[2],
		cUnit[0]*rUnit[1] - cUnit[1]*rUnit[0],
	}
	
	// Relative position
	relPos := [3]float64{
		chaserPos[0] - targetPos[0],
		chaserPos[1] - targetPos[1],
		chaserPos[2] - targetPos[2],
	}
	
	// Relative velocity
	relVel := [3]float64{
		chaserVel[0] - targetVel[0],
		chaserVel[1] - targetVel[1],
		chaserVel[2] - targetVel[2],
	}
	
	// Project relative position onto RIC frame
	x := relPos[0]*rUnit[0] + relPos[1]*rUnit[1] + relPos[2]*rUnit[2]
	y := relPos[0]*iUnit[0] + relPos[1]*iUnit[1] + relPos[2]*iUnit[2]
	z := relPos[0]*cUnit[0] + relPos[1]*cUnit[1] + relPos[2]*cUnit[2]
	
	// Project relative velocity onto RIC frame
	vx := relVel[0]*rUnit[0] + relVel[1]*rUnit[1] + relVel[2]*rUnit[2]
	vy := relVel[0]*iUnit[0] + relVel[1]*iUnit[1] + relVel[2]*iUnit[2]
	vz := relVel[0]*cUnit[0] + relVel[1]*cUnit[1] + relVel[2]*cUnit[2]
	
	return HCWState{
		Pos: [3]float64{x, y, z},
		Vel: [3]float64{vx, vy, vz},
	}, nil
}

// RPOMetrics provides detailed RPO pattern analysis
type RPOMetrics struct {
	IsRPO          bool     // True if pattern matches HCW ellipse
	ResidualKm     float64  // Average residual in km
	Correlation    float64  // Correlation coefficient (0-1)
	EllipseParams  Ellipse  // Fitted ellipse parameters
}

// Ellipse represents fitted ellipse parameters for RPO
type Ellipse struct {
	SemiMajorAxis  float64  // km
	SemiMinorAxis  float64  // km
	Orientation    float64  // radians
	Center         [3]float64 // km
}

// AnalyzeRPOPattern provides comprehensive RPO pattern analysis
func AnalyzeRPOPattern(observations []HCWState, n float64, timeStepSeconds float64) RPOMetrics {
	if len(observations) < 5 {
		return RPOMetrics{IsRPO: false}
	}

	isRPO, residual := IsRPOPattern(observations, n, timeStepSeconds)
	
	// Calculate correlation coefficient
	correlation := calculateCorrelation(observations, n, timeStepSeconds)
	
	// Fit ellipse to in-track/radial motion
	ellipse := fitEllipse(observations)
	
	return RPOMetrics{
		IsRPO:         isRPO,
		ResidualKm:    residual,
		Correlation:   correlation,
		EllipseParams: ellipse,
	}
}

// calculateCorrelation computes correlation between predicted and observed
func calculateCorrelation(observations []HCWState, n float64, timeStepSeconds float64) float64 {
	if len(observations) < 3 {
		return 0
	}

	initial := observations[0]
	var sumXY, sumX, sumY, sumX2, sumY2 float64
	count := 0

	for i := 1; i < len(observations); i++ {
		t := timeStepSeconds * float64(i)
		predicted := PredictRelativePosition(initial, n, t)
		observed := observations[i].Pos
		
		// Use radial component for correlation
		x := predicted[0]
		y := observed[0]
		
		sumXY += x * y
		sumX += x
		sumY += y
		sumX2 += x * x
		sumY2 += y * y
		count++
	}

	if count == 0 {
		return 0
	}

	numerator := sumXY - (sumX * sumY / float64(count))
	denominator := math.Sqrt((sumX2 - sumX*sumX/float64(count)) * (sumY2 - sumY*sumY/float64(count)))
	
	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

// fitEllipse fits ellipse to radial/in-track motion
func fitEllipse(observations []HCWState) Ellipse {
	if len(observations) < 3 {
		return Ellipse{}
	}

	// Simple ellipse fitting using bounding box
	var minX, maxX, minY, maxY float64 = 1e9, -1e9, 1e9, -1e9
	var sumX, sumY float64

	for _, obs := range observations {
		x := obs.Pos[0] // Radial
		y := obs.Pos[1] // In-track
		
		if x < minX { minX = x }
		if x > maxX { maxX = x }
		if y < minY { minY = y }
		if y > maxY { maxY = y }
		
		sumX += x
		sumY += y
	}

	count := float64(len(observations))
	center := [3]float64{sumX / count, sumY / count, 0}

	return Ellipse{
		SemiMajorAxis: (maxX - minX) / 2,
		SemiMinorAxis: (maxY - minY) / 2,
		Center:        center,
	}
}
