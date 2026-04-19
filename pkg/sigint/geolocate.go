// Package sigint provides TDOA/FDOA geolocation capabilities for emitter positioning
// Implements Newton-Raphson iterative solver for high-precision ground coordinate calculation
package sigint

import (
	"fmt"
	"math"
)

const (
	// Physical constants (duplicates from rf_processor.go for module independence)
	GeoSpeedOfLight    = 299792458.0 // m/s
	GeoSpeedOfLightKmS = 299792.458  // km/s

	// WGS84 Earth Ellipsoid parameters
	WGS84SemiMajorAxis = 6378137.0                                    // a (meters)
	WGS84Flattening    = 1.0 / 298.257223563                          // f
	WGS84SemiMinorAxis = WGS84SemiMajorAxis * (1.0 - WGS84Flattening) // b (meters)

	// Solver parameters
	MaxIterations        = 50
	ConvergenceThreshold = 0.1 // meters
	StepSize             = 0.5 // Newton-Raphson step damping
)

// WGS84Ellipsoid provides WGS84 Earth model calculations
type WGS84Ellipsoid struct {
	A  float64 // Semi-major axis (m)
	B  float64 // Semi-minor axis (m)
	F  float64 // Flattening
	E2 float64 // First eccentricity squared
}

// NewWGS84Ellipsoid creates a WGS84 ellipsoid
func NewWGS84Ellipsoid() *WGS84Ellipsoid {
	a := WGS84SemiMajorAxis
	f := WGS84Flattening
	b := a * (1.0 - f)
	e2 := 1.0 - (b*b)/(a*a)

	return &WGS84Ellipsoid{
		A:  a,
		B:  b,
		F:  f,
		E2: e2,
	}
}

// GeodeticToECEF converts latitude/longitude/altitude to Earth-Centered Earth-Fixed coordinates
// Input: lat/lon in degrees, alt in meters
// Output: [x, y, z] in meters
func (w *WGS84Ellipsoid) GeodeticToECEF(lat, lon, alt float64) [3]float64 {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	sinLon := math.Sin(lonRad)
	cosLon := math.Cos(lonRad)

	// Radius of curvature in the prime vertical
	N := w.A / math.Sqrt(1.0-w.E2*sinLat*sinLat)

	x := (N + alt) * cosLat * cosLon
	y := (N + alt) * cosLat * sinLon
	z := (N*(1.0-w.E2) + alt) * sinLat

	return [3]float64{x, y, z}
}

// ECEFToGeodetic converts ECEF coordinates to latitude/longitude/altitude
// Input: x, y, z in meters
// Output: lat/lon in degrees, alt in meters
func (w *WGS84Ellipsoid) ECEFToGeodetic(x, y, z float64) (lat, lon, alt float64) {
	// Longitude is straightforward
	lon = math.Atan2(y, x) * 180.0 / math.Pi

	// Distance from z-axis
	p := math.Sqrt(x*x + y*y)

	// Initial guess for latitude
	lat = math.Atan2(z, p*(1.0-w.E2))

	var sinLat, cosLat float64

	// Iterate to find correct latitude (Bowring's method)
	for i := 0; i < 10; i++ {
		sinLat = math.Sin(lat)
		cosLat = math.Cos(lat)

		// Radius of curvature in the prime vertical
		N := w.A / math.Sqrt(1.0-w.E2*sinLat*sinLat)

		// New latitude estimate
		newLat := math.Atan2(z+w.E2*N*sinLat, p)

		// Check convergence
		if math.Abs(newLat-lat) < 1e-12 {
			break
		}
		lat = newLat
	}

	// Convert to degrees
	latDeg := lat * 180.0 / math.Pi

	// Calculate altitude
	sinLat = math.Sin(lat)
	cosLat = math.Cos(lat)
	N := w.A / math.Sqrt(1.0-w.E2*sinLat*sinLat)
	alt = p/cosLat - N

	return latDeg, lon, alt
}

// ProjectToSurface projects a 3D point onto the WGS84 ellipsoid surface
func (w *WGS84Ellipsoid) ProjectToSurface(point [3]float64) [3]float64 {
	lat, lon, _ := w.ECEFToGeodetic(point[0], point[1], point[2])
	return w.GeodeticToECEF(lat, lon, 0.0)
}

// TDOASolver performs Time Difference of Arrival geolocation
type TDOASolver struct {
	ellipsoid *WGS84Ellipsoid
}

// NewTDOASolver creates a new TDOA solver
func NewTDOASolver() *TDOASolver {
	return &TDOASolver{
		ellipsoid: NewWGS84Ellipsoid(),
	}
}

// TDOAResult represents the result of TDOA geolocation
type TDOAResult struct {
	Latitude   float64 // degrees
	Longitude  float64 // degrees
	Altitude   float64 // meters
	Accuracy   float64 // CEP in meters
	Confidence float64 // 0.0-1.0
	Iterations int     // Number of iterations to converge
	Residual   float64 // Final residual in meters
	Method     string  // "TDOA_2SAT", "TDOA_3SAT", "TDOA_DOA_HYBRID"
}

// Geolocate2Sat solves TDOA with 2 satellites and Earth surface constraint
// Produces a hyperbola of possible positions; returns the point closest to initial guess
func (t *TDOASolver) Geolocate2Sat(sat1, sat2 [3]float64, deltaT float64, initialGuess [3]float64) (*TDOAResult, error) {
	// Convert deltaT to distance difference
	distanceDiff := deltaT * SpeedOfLight

	// Project initial guess to Earth surface
	guess := t.ellipsoid.ProjectToSurface(initialGuess)

	var finalResidual float64
	iterations := 0

	// Newton-Raphson iteration
	for i := 0; i < MaxIterations; i++ {
		iterations = i + 1

		// Calculate current distances
		d1 := distance3D(guess, sat1)
		d2 := distance3D(guess, sat2)

		// Observed vs calculated range difference
		calculatedDiff := d1 - d2
		residual := distanceDiff - calculatedDiff
		finalResidual = math.Abs(residual)

		// Check convergence
		if finalResidual < ConvergenceThreshold {
			break
		}

		// Calculate gradient (Jacobian)
		// How does range difference change with position?
		grad1 := normalize3D(subtract3D(guess, sat1))
		grad2 := normalize3D(subtract3D(guess, sat2))
		jacobian := subtract3D(grad1, grad2)

		// Update guess along gradient, damped step
		step := multiply3D(jacobian, residual*StepSize)
		newGuess := add3D(guess, step)

		// Re-project onto Earth surface
		guess = t.ellipsoid.ProjectToSurface(newGuess)
	}

	// Convert result to geodetic coordinates
	lat, lon, alt := t.ellipsoid.ECEFToGeodetic(guess[0], guess[1], guess[2])

	// Estimate accuracy based on geometry (GDOP)
	accuracy := t.estimateAccuracy2Sat(sat1, sat2, guess)

	return &TDOAResult{
		Latitude:   lat,
		Longitude:  lon,
		Altitude:   alt,
		Accuracy:   accuracy,
		Confidence: calculateConfidence(finalResidual, accuracy),
		Iterations: iterations,
		Residual:   finalResidual,
		Method:     "TDOA_2SAT",
	}, nil
}

// Geolocate3Sat solves TDOA with 3 satellites for pinpoint geolocation
// Intersection of two hyperbolas yields a single point
func (t *TDOASolver) Geolocate3Sat(sat1, sat2, sat3 [3]float64, deltaT12, deltaT13 float64, initialGuess [3]float64) (*TDOAResult, error) {
	// Convert deltaT to distance differences
	distanceDiff12 := deltaT12 * SpeedOfLight
	distanceDiff13 := deltaT13 * SpeedOfLight

	// Project initial guess to Earth surface
	guess := t.ellipsoid.ProjectToSurface(initialGuess)

	var finalResidual float64
	iterations := 0

	// Newton-Raphson iteration for 2 constraints
	for i := 0; i < MaxIterations; i++ {
		iterations = i + 1

		// Calculate current distances
		d1 := distance3D(guess, sat1)
		d2 := distance3D(guess, sat2)
		d3 := distance3D(guess, sat3)

		// Residuals for both TDOA measurements
		calculatedDiff12 := d1 - d2
		calculatedDiff13 := d1 - d3

		residual12 := distanceDiff12 - calculatedDiff12
		residual13 := distanceDiff13 - calculatedDiff13

		// Combined residual (sum of squares)
		finalResidual = math.Sqrt(residual12*residual12 + residual13*residual13)

		// Check convergence
		if finalResidual < ConvergenceThreshold {
			break
		}

		// Calculate gradients for both constraints
		grad1 := normalize3D(subtract3D(guess, sat1))
		grad2 := normalize3D(subtract3D(guess, sat2))
		grad3 := normalize3D(subtract3D(guess, sat3))

		jacobian12 := subtract3D(grad1, grad2)
		jacobian13 := subtract3D(grad1, grad3)

		// Combined update (weighted average of both constraints)
		step12 := multiply3D(jacobian12, residual12*StepSize)
		step13 := multiply3D(jacobian13, residual13*StepSize)
		combinedStep := multiply3D(add3D(step12, step13), 0.5)

		// Update guess
		newGuess := add3D(guess, combinedStep)

		// Re-project onto Earth surface
		guess = t.ellipsoid.ProjectToSurface(newGuess)
	}

	// Convert result to geodetic coordinates
	lat, lon, alt := t.ellipsoid.ECEFToGeodetic(guess[0], guess[1], guess[2])

	// Estimate accuracy
	accuracy := t.estimateAccuracy3Sat(sat1, sat2, sat3, guess)

	return &TDOAResult{
		Latitude:   lat,
		Longitude:  lon,
		Altitude:   alt,
		Accuracy:   accuracy,
		Confidence: calculateConfidence(finalResidual, accuracy),
		Iterations: iterations,
		Residual:   finalResidual,
		Method:     "TDOA_3SAT",
	}, nil
}

// DOAMeasurement represents Direction of Arrival measurement from a satellite
type DOAMeasurement struct {
	SatelliteID string
	Azimuth     float64 // degrees from satellite (0 = north)
	Elevation   float64 // degrees above horizon (0 = horizon, 90 = zenith)
	Confidence  float64 // 0.0-1.0
}

// Geolocate2SatWithDOA solves TDOA with 2 satellites + Direction of Arrival for pinpoint
func (t *TDOASolver) Geolocate2SatWithDOA(sat1, sat2 [3]float64, deltaT float64,
	doa DOAMeasurement, initialGuess [3]float64) (*TDOAResult, error) {

	// First, solve for hyperbola using TDOA only
	tdoaResult, err := t.Geolocate2Sat(sat1, sat2, deltaT, initialGuess)
	if err != nil {
		return nil, err
	}

	// Convert DOA to ECEF direction vector
	doaDirection := t.doaToECEF(sat1, doa.Azimuth, doa.Elevation)

	// Find point on hyperbola closest to DOA line
	guess := t.ellipsoid.GeodeticToECEF(tdoaResult.Latitude, tdoaResult.Longitude, 0)
	sat1Pos := sat1

	// Refine using DOA constraint
	for i := 0; i < MaxIterations/2; i++ {
		// Vector from satellite to guess
		toGuess := subtract3D(guess, sat1Pos)
		toGuessDir := normalize3D(toGuess)

		// Angle between DOA direction and actual direction
		dotProduct := dot3D(doaDirection, toGuessDir)
		angleError := math.Acos(dotProduct) * 180.0 / math.Pi

		if math.Abs(angleError) < 0.1 {
			break
		}

		// Adjust guess perpendicular to DOA direction
		// Move along the hyperbola to minimize angle error
		perp := subtract3D(toGuessDir, multiply3D(doaDirection, dotProduct))
		if norm3D(perp) > 1e-10 {
			perp = normalize3D(perp)
			guess = add3D(guess, multiply3D(perp, 100.0)) // 100m step
			guess = t.ellipsoid.ProjectToSurface(guess)
		}
	}

	lat, lon, alt := t.ellipsoid.ECEFToGeodetic(guess[0], guess[1], guess[2])

	return &TDOAResult{
		Latitude:   lat,
		Longitude:  lon,
		Altitude:   alt,
		Accuracy:   tdoaResult.Accuracy * 0.5, // DOA improves accuracy
		Confidence: math.Min(1.0, tdoaResult.Confidence+0.2),
		Iterations: tdoaResult.Iterations,
		Residual:   tdoaResult.Residual,
		Method:     "TDOA_DOA_HYBRID",
	}, nil
}

// doaToECEF converts DOA angles to ECEF direction vector
func (t *TDOASolver) doaToECEF(satPos [3]float64, azimuth, elevation float64) [3]float64 {
	// Convert satellite position to local ENU frame
	lat, lon, _ := t.ellipsoid.ECEFToGeodetic(satPos[0], satPos[1], satPos[2])

	// Azimuth and elevation to ENU
	azRad := azimuth * math.Pi / 180.0
	elRad := elevation * math.Pi / 180.0

	east := math.Sin(azRad) * math.Cos(elRad)
	north := math.Cos(azRad) * math.Cos(elRad)
	up := math.Sin(elRad)

	// ENU to ECEF rotation
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0

	sinLat := math.Sin(latRad)
	cosLat := math.Cos(latRad)
	sinLon := math.Sin(lonRad)
	cosLon := math.Cos(lonRad)

	// Rotation matrix ENU → ECEF
	x := -sinLon*east - sinLat*cosLon*north + cosLat*cosLon*up
	y := cosLon*east - sinLat*sinLon*north + cosLat*sinLon*up
	z := cosLat*north + sinLat*up

	return normalize3D([3]float64{x, y, z})
}

// estimateAccuracy2Sat estimates CEP accuracy for 2-satellite TDOA
func (t *TDOASolver) estimateAccuracy2Sat(sat1, sat2, guess [3]float64) float64 {
	// Calculate geometry (baseline distance)
	baseline := distance3D(sat1, sat2)

	// Calculate distance to satellites
	d1 := distance3D(guess, sat1)
	d2 := distance3D(guess, sat2)

	// Estimate GDOP based on geometry
	// For 2 satellites, we can only constrain to hyperbola
	// Accuracy depends on baseline length and geometry
	gdop := (d1 + d2) / baseline

	// Assume 10ns timing accuracy → ~3m range error
	timingError := 3.0 // meters

	// CEP scales with GDOP
	return timingError * gdop
}

// estimateAccuracy3Sat estimates CEP accuracy for 3-satellite TDOA
func (t *TDOASolver) estimateAccuracy3Sat(sat1, sat2, sat3, guess [3]float64) float64 {
	// Calculate baselines
	baseline12 := distance3D(sat1, sat2)
	baseline13 := distance3D(sat1, sat3)

	// Average baseline
	avgBaseline := (baseline12 + baseline13) / 2.0

	// Calculate distances to satellites
	d1 := distance3D(guess, sat1)
	d2 := distance3D(guess, sat2)
	d3 := distance3D(guess, sat3)

	avgDistance := (d1 + d2 + d3) / 3.0

	// GDOP improves significantly with 3 satellites
	gdop := avgDistance / avgBaseline

	// Assume 10ns timing accuracy
	timingError := 3.0 // meters

	return timingError * gdop
}

// calculateConfidence calculates overall confidence score
func calculateConfidence(residual, accuracy float64) float64 {
	// Based on residual and accuracy
	residualScore := math.Exp(-residual / 100.0)  // Good if residual < 100m
	accuracyScore := math.Exp(-accuracy / 1000.0) // Good if accuracy < 1000m

	return 0.5*residualScore + 0.5*accuracyScore
}

// 3D vector math helpers

func distance3D(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func subtract3D(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

func add3D(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
}

func multiply3D(v [3]float64, scalar float64) [3]float64 {
	return [3]float64{v[0] * scalar, v[1] * scalar, v[2] * scalar}
}

func dot3D(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func norm3D(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func normalize3D(v [3]float64) [3]float64 {
	n := norm3D(v)
	if n < 1e-15 {
		return [3]float64{0, 0, 0}
	}
	return [3]float64{v[0] / n, v[1] / n, v[2] / n}
}

// GeolocateEmitter provides a unified interface for TDOA geolocation
// Automatically selects appropriate method based on available data
func (t *TDOASolver) GeolocateEmitter(satellites map[string][3]float64,
	tdoaMeasurements map[string]float64, // key format: "SAT1-SAT2" -> deltaT
	doa *DOAMeasurement,
	initialGuess [3]float64) (*TDOAResult, error) {

	// Count available measurements
	numTDOA := len(tdoaMeasurements)

	if numTDOA == 0 {
		return nil, fmt.Errorf("no TDOA measurements provided")
	}

	if numTDOA >= 2 && len(satellites) >= 3 {
		// Extract 3-satellite solution
		var sat1, sat2, sat3 [3]float64
		var deltaT12, deltaT13 float64
		var sat1Name, sat3Name string

		// Find measurements
		for key, deltaT := range tdoaMeasurements {
			var s1, s2 string
			fmt.Sscanf(key, "%s-%s", &s1, &s2)

			if sat1Name == "" {
				sat1Name = s1
				sat1 = satellites[s1]
				sat2 = satellites[s2]
				deltaT12 = deltaT
			} else if s1 == sat1Name {
				sat3Name = s2
				sat3 = satellites[s2]
				deltaT13 = deltaT
				break
			}
		}

		if sat3Name != "" {
			return t.Geolocate3Sat(sat1, sat2, sat3, deltaT12, deltaT13, initialGuess)
		}
	}

	// Fall back to 2-satellite + DOA if available
	if numTDOA >= 1 && doa != nil {
		for key, deltaT := range tdoaMeasurements {
			var s1, s2 string
			fmt.Sscanf(key, "%s-%s", &s1, &s2)

			if doa.SatelliteID == s1 || doa.SatelliteID == s2 {
				return t.Geolocate2SatWithDOA(satellites[s1], satellites[s2], deltaT, *doa, initialGuess)
			}
		}
	}

	// Fall back to 2-satellite only
	if numTDOA >= 1 {
		for key, deltaT := range tdoaMeasurements {
			var s1, s2 string
			fmt.Sscanf(key, "%s-%s", &s1, &s2)
			return t.Geolocate2Sat(satellites[s1], satellites[s2], deltaT, initialGuess)
		}
	}

	return nil, fmt.Errorf("insufficient measurements for geolocation")
}

// SubSatellitePoint calculates the point on Earth surface directly below a satellite
func (t *TDOASolver) SubSatellitePoint(satPos [3]float64) (lat, lon float64) {
	// Project satellite position to Earth surface
	surfacePoint := t.ellipsoid.ProjectToSurface(satPos)
	lat, lon, _ = t.ellipsoid.ECEFToGeodetic(surfacePoint[0], surfacePoint[1], surfacePoint[2])
	return lat, lon
}

// CalculateHyperbolaPoints generates points along the TDOA hyperbola for visualization
func (t *TDOASolver) CalculateHyperbolaPoints(sat1, sat2 [3]float64, deltaT float64, numPoints int) [][3]float64 {
	distanceDiff := deltaT * SpeedOfLight

	// Find midpoint between satellites
	midpoint := multiply3D(add3D(sat1, sat2), 0.5)

	// Direction vector between satellites
	direction := normalize3D(subtract3D(sat2, sat1))

	// Generate points perpendicular to direction, offset by distanceDiff
	points := make([][3]float64, 0, numPoints)

	for i := 0; i < numPoints; i++ {
		angle := float64(i) * 2.0 * math.Pi / float64(numPoints)

		// Perpendicular vector (simplified - use cross product with arbitrary vector)
		var perp [3]float64
		if math.Abs(direction[2]) < 0.9 {
			perp = normalize3D([3]float64{-direction[1], direction[0], 0})
		} else {
			perp = normalize3D([3]float64{0, -direction[2], direction[1]})
		}

		// Rotate around direction
		cosAngle := math.Cos(angle)
		sinAngle := math.Sin(angle)

		rotated := add3D(
			multiply3D(perp, cosAngle),
			multiply3D(cross3D(direction, perp), sinAngle),
		)

		// Offset by distanceDiff/2 along direction from midpoint
		offset := multiply3D(direction, distanceDiff/2.0)
		point := add3D(midpoint, offset)
		point = add3D(point, multiply3D(rotated, 100000.0)) // 100km radius

		// Project to Earth surface
		point = t.ellipsoid.ProjectToSurface(point)
		points = append(points, point)
	}

	return points
}

func cross3D(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}
