package collision

import (
	"math"
	"ssis/pkg/octree"
)

// RSOEnvelope defines the 3D geometry of a Resident Space Object.
type RSOEnvelope struct {
	Vertices []octree.Vector3
}

// Standard RSO Envelopes for fallback/modelling.
var (
	EnvelopeRocketBody = RSOEnvelope{
		Vertices: []octree.Vector3{
			{X: -2, Y: -2, Z: -5}, {X: 2, Y: -2, Z: -5}, {X: 2, Y: 2, Z: -5}, {X: -2, Y: 2, Z: -5},
			{X: -2, Y: -2, Z: 5}, {X: 2, Y: -2, Z: 5}, {X: 2, Y: 2, Z: 5}, {X: -2, Y: 2, Z: 5},
		},
	}
	EnvelopeStarlink = RSOEnvelope{
		Vertices: []octree.Vector3{
			{X: -1.5, Y: -0.1, Z: -3}, {X: 1.5, Y: -0.1, Z: -3}, {X: 1.5, Y: 0.1, Z: -3}, {X: -1.5, Y: 0.1, Z: -3},
			{X: -1.5, Y: -0.1, Z: 3}, {X: 1.5, Y: -0.1, Z: 3}, {X: 1.5, Y: 0.1, Z: 3}, {X: -1.5, Y: 0.1, Z: 3},
		},
	}
	EnvelopeCubeSat = RSOEnvelope{
		Vertices: []octree.Vector3{
			{X: -0.1, Y: -0.1, Z: -0.1}, {X: 0.1, Y: -0.1, Z: -0.1}, {X: 0.1, Y: 0.1, Z: -0.1}, {X: -0.1, Y: 0.1, Z: -0.1},
			{X: -0.1, Y: -0.1, Z: 0.1}, {X: 0.1, Y: -0.1, Z: 0.1}, {X: 0.1, Y: 0.1, Z: 0.1}, {X: -0.1, Y: 0.1, Z: 0.1},
		},
	}
)

// MinkowskiSumGeometricPc computes the collision probability by integrating the encounter 
// distribution over the projected Minkowski sum of the two object shapes.
func MinkowskiSumGeometricPc(missR, missI, sigmaR, sigmaI, sigmaRI float64, hull1, hull2 RSOEnvelope, vRel octree.Vector3) (float64, error) {
	// 1. Project 3D hulls to 2D encounter plane
	// Scale vertices to km (assuming input in meters)
	h1Conv := make([]octree.Vector3, len(hull1.Vertices))
	for i, v := range hull1.Vertices {
		h1Conv[i] = octree.Vector3{X: v.X * 0.001, Y: v.Y * 0.001, Z: v.Z * 0.001}
	}
	h2Conv := make([]octree.Vector3, len(hull2.Vertices))
	for i, v := range hull2.Vertices {
		h2Conv[i] = octree.Vector3{X: v.X * 0.001, Y: v.Y * 0.001, Z: v.Z * 0.001}
	}

	poly1 := ProjectToEncounterPlane(h1Conv, vRel)
	poly2 := ProjectToEncounterPlane(h2Conv, vRel)

	// 2. Compute 2D Minkowski Sum (Convex)
	// We centered at (0,0) for the shapes, the miss vector is handled by the Gaussian mean.
	minkowskiHull := MinkowskiSum2D(poly1, poly2)

	// 3. Coordinate Transformation to Principal Frame (Diagonal Covariance)
	xm, ym, lambdaMax, lambdaMin, ok := principalFrame(missR, missI, sigmaR, sigmaI, sigmaRI)
	if !ok {
		return 0, nil
	}

	// Transform hull vertices to principal frame
	// principalFrame uses theta = 0.5 * atan2(2*cov, varR - varI)
	theta := 0.5 * math.Atan2(2.0*sigmaRI, sigmaR-sigmaI)
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)

	pfHull := make([]Vector2, len(minkowskiHull))
	for i, v := range minkowskiHull {
		pfHull[i] = Vector2{
			X: cosT*v.X + sinT*v.Y,
			Y: -sinT*v.X + cosT*v.Y,
		}
	}

	// 4. Numerical Integration over the Polygon in Principal Frame
	// P(X, Y) ~ N([xm, ym], [lambdaMax, 0; 0, lambdaMin])
	sigmaX := math.Sqrt(lambdaMax)
	sigmaY := math.Sqrt(lambdaMin)

	return integrateGaussianOverPolygon(pfHull, xm, ym, sigmaX, sigmaY), nil
}

// integrateGaussianOverPolygon uses Green's Theorem to integrate a 2D Gaussian over a polygon.
func integrateGaussianOverPolygon(poly []Vector2, mx, my, sx, sy float64) float64 {
	if len(poly) < 3 {
		return 0
	}

	sum := 0.0
	const sqrt2 = 1.41421356237
	
	// Ensure the polygon is closed
	n := len(poly)
	
	// Use cached Gauss-Legendre rule from foster.go for efficiency and accuracy
	nodes, weights := gaussLegendreRule(48) // 48-point quadrature for stable integration
	
	for i := 0; i < n; i++ {
		p1 := poly[i]
		p2 := poly[(i+1)%n]

		dx := p2.X - p1.X
		dy := p2.Y - p1.Y
		if math.Abs(dx) < 1e-15 && math.Abs(dy) < 1e-15 {
			continue
		}

		edgeSum := 0.0
		for j := range nodes {
			t := 0.5 * (nodes[j] + 1.0)
			x := p1.X + t*dx
			y := p1.Y + t*dy
			
			termX := 0.5 * (1.0 + math.Erf((x-mx)/(sx*sqrt2)))
			termY := (1.0 / (sy * math.Sqrt(2.0*math.Pi))) * math.Exp(-0.5*math.Pow((y-my)/sy, 2))
			
			edgeSum += termX * termY * weights[j]
		}
		
		sum += 0.5 * edgeSum * dy
	}

	return math.Abs(sum)
}
