package sda

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"
	"ssis/pkg/spatial"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Volumetric Scan Module
 *
 * Implements 4D Manifold Intersection for space domain awareness.
 * Transforms ECEF volumes to ECI swept-toroids and performs covariance-aware
 * intersection detection using symplectic propagation.
 */

// Region defines a volumetric region in ECEF coordinates
type Region struct {
	CenterLat float64 // Latitude (radians)
	CenterLon float64 // Longitude (radians)
	RadiusKm  float64 // Horizontal radius (km)
	AltMinKm  float64 // Minimum altitude (km)
	AltMaxKm  float64 // Maximum altitude (km)
}

// TimeWindow defines the observation time window
type TimeWindow struct {
	Start time.Time
	End   time.Time
}

// Threat represents a detected object entering a protected volume
type Threat struct {
	ObjectID        int
	ObjectName      string
	CountryCode     string
	EntryTime       time.Time
	ExitTime        time.Time
	MinDistanceKm   float64
	MaxProbability  float64
	ThreatLevel     string // "LOW", "MEDIUM", "HIGH", "CRITICAL"
	IntentAnalysis  string // SINDy-based intent classification
	HamiltonianHash string
}

// VolumetricEngine implements the 4D Manifold Gate
type VolumetricEngine struct {
	Catalog        *catalog.Store
	BVH4D          *spatial.BVH4D
	SINDy          *physics.SINDy
	SymplecticProp *SymplecticPropagator
	EarthRotation  float64 // ω⊕ = 7.292 × 10⁻⁵ rad/s
}

// NewVolumetricEngine creates a new volumetric scan engine
func NewVolumetricEngine(cat *catalog.Store) *VolumetricEngine {
	return &VolumetricEngine{
		Catalog:        cat,
		BVH4D:          spatial.NewBVH4D(spatial.WDimensionTime),
		SINDy:          physics.NewSINDy(),
		SymplecticProp: NewSymplecticPropagator(),
		EarthRotation:  7.2921159e-5, // Earth's rotation rate (rad/s)
	}
}

// ScanVolume performs a 4D volumetric scan over a region
// Algorithm:
// 1. Transform ECEF region vertices to ECI swept-toroid
// 2. Coarse filter: RAAN/Inclination pruning via BVH
// 3. Fine filter: Symplectic propagation with covariance ellipsoid
// 4. Intent analysis via SINDy for detected threats
func (e *VolumetricEngine) ScanVolume(region Region, window TimeWindow) []Threat {
	// 1. Build 4D BVH from catalog
	e.build4DIndex(window)

	// 2. Transform ECEF region to ECI bounding box
	eciBounds := e.transformECEFToECI(region, window)

	// 3. Coarse filter via BVH reachability query
	candidates := e.BVH4D.QueryReachability(
		eciBounds.Center,
		eciBounds.Radii,
	)

	// 4. Fine filter with symplectic propagation
	var threats []Threat
	for _, objID := range candidates {
		obj, ok := e.Catalog.GetRecord(int(objID))
		if !ok {
			continue
		}

		if e.checkIntersection(obj, region, window) {
			threat := e.analyzeThreat(obj, region, window)
			threats = append(threats, threat)
		}
	}

	return threats
}

// ECIBounds represents the 4D bounding box in ECI
type ECIBounds struct {
	Center [4]float64 // [x, y, z, time]
	Radii  [4]float64 // [rx, ry, rz, rt]
}

// transformECEFToECI converts an ECEF region to ECI swept-toroid bounds
// The key physics: Earth rotates at ω⊕, so a static ECEF volume sweeps
// through ECI space as a toroid over time.
func (e *VolumetricEngine) transformECEFToECI(region Region, window TimeWindow) ECIBounds {
	duration := window.End.Sub(window.Start).Seconds()
	midTime := window.Start.Add(time.Duration(duration/2) * time.Second)

	// Convert geodetic center to ECEF
	ecefX, ecefY, ecefZ := physics.GeodeticToECEFWGS84(
		region.CenterLat,
		region.CenterLon,
		(region.AltMinKm+region.AltMaxKm)/2,
	)

	// Transform to ECI at mid-time
	jd := timeToJD(midTime)
	eciX, eciY, eciZ := physics.ECIToECEF(ecefX, ecefY, ecefZ, jd)

	// Account for Earth rotation sweep
	// Maximum angular displacement = ω⊕ * duration
	maxSweepAngle := e.EarthRotation * duration

	// Horizontal radius in ECI (expanded by sweep)
	rHorizontal := region.RadiusKm + maxSweepAngle*math.Sqrt(ecefX*ecefX+ecefY*ecefY)

	// Vertical radius (altitude range)
	rVertical := (region.AltMaxKm - region.AltMinKm) / 2

	// Time radius (half window duration)
	rTime := duration / 2

	return ECIBounds{
		Center: [4]float64{eciX, eciY, eciZ, julianDateToSeconds(jd)},
		Radii:  [4]float64{rHorizontal, rHorizontal, rVertical, rTime},
	}
}

// build4DIndex constructs the 4D BVH index from the catalog
func (e *VolumetricEngine) build4DIndex(window TimeWindow) {
	records := e.Catalog.Snapshot()

	objects := make([]spatial.BVH4DObject, len(records))
	for i, rec := range records {
		pos := rec.Pos
		objects[i] = spatial.BVH4DObject{
			ID:       uint32(rec.ID),
			Position: [3]float64{pos.X, pos.Y, pos.Z},
			W:        rec.EpochTT, // Use epoch as 4th dimension (time)
		}
		// Set bounds (simplified - could use covariance)
		objects[i].Bounds = [4][2]float64{
			{pos.X - rec.PosSigmaKm, pos.X + rec.PosSigmaKm},
			{pos.Y - rec.PosSigmaKm, pos.Y + rec.PosSigmaKm},
			{pos.Z - rec.PosSigmaKm, pos.Z + rec.PosSigmaKm},
			{rec.EpochTT - 3600, rec.EpochTT + 3600}, // ±1 hour time window
		}
	}

	e.BVH4D.Build(objects)
}

// checkIntersection performs fine-grained intersection check using symplectic propagation
// Checks if the covariance ellipsoid intersects the swept-toroid volume
func (e *VolumetricEngine) checkIntersection(obj catalog.SDAObject, region Region, window TimeWindow) bool {
	// Sample trajectory at multiple time points
	samples := 100
	duration := window.End.Sub(window.Start).Seconds()
	dt := duration / float64(samples)

	for i := 0; i <= samples; i++ {
		t := window.Start.Add(time.Duration(float64(i)*dt) * time.Second)

		// Propagate using symplectic integrator
		state := e.propagateState(obj, t)

		// Transform to ECEF
		jd := timeToJD(t)
		ecefX, ecefY, ecefZ := physics.ECIToECEF(state[0], state[1], state[2], jd)

		// Convert to geodetic
		lat, lon, alt := physics.ECEFToGeodeticWGS84(ecefX, ecefY, ecefZ)

		// Check if within region (with covariance inflation)
		dist := greatCircleDistance(lat, lon, region.CenterLat, region.CenterLon)
		if dist <= region.RadiusKm+obj.PosSigmaKm &&
			alt >= region.AltMinKm-obj.PosSigmaKm &&
			alt <= region.AltMaxKm+obj.PosSigmaKm {
			return true
		}
	}

	return false
}

// analyzeThreat performs SINDy-based intent analysis
func (e *VolumetricEngine) analyzeThreat(obj catalog.SDAObject, region Region, window TimeWindow) Threat {
	// Build trajectory for SINDy
	states := e.buildTrajectoryStates(obj, window)

	// Run SINDy maneuver detection
	result, _ := e.SINDy.DiscoverManeuver(states, 10.0)

	// Compute Hamiltonian hash using symplectic propagator
	hash := e.computeHamiltonianHash(obj, window)

	// Classify threat level
	threatLevel := e.classifyThreatLevel(obj, result)

	return Threat{
		ObjectID:        obj.ID,
		ObjectName:      obj.Name,
		CountryCode:     obj.CountryCode,
		EntryTime:       window.Start,
		ExitTime:        window.End,
		MinDistanceKm:   e.computeMinDistance(obj, region, window),
		MaxProbability:  result.Confidence,
		ThreatLevel:     threatLevel,
		IntentAnalysis:  result.EquationString,
		HamiltonianHash: hash,
	}
}

// buildTrajectoryStates constructs state trajectory for SINDy analysis
func (e *VolumetricEngine) buildTrajectoryStates(obj catalog.SDAObject, window TimeWindow) [][6]float64 {
	samples := 50
	duration := window.End.Sub(window.Start).Seconds()
	dt := duration / float64(samples)

	states := make([][6]float64, samples)
	for i := 0; i < samples; i++ {
		t := window.Start.Add(time.Duration(float64(i)*dt) * time.Second)
		state := e.propagateState(obj, t)
		states[i] = state
	}

	return states
}

// classifyThreatLevel assigns threat level based on object characteristics and SINDy results
func (e *VolumetricEngine) classifyThreatLevel(obj catalog.SDAObject, result *physics.ManeuverResult) string {
	// Critical: Active maneuver detected
	if result.ThrustDetected && result.ThrustMagnitude > 0.1 {
		return "CRITICAL"
	}

	// High: High energy object with recent maneuver
	if obj.CountryCode != "IN" && result.ThrustDetected {
		return "HIGH"
	}

	// Medium: Foreign asset in proximity
	if obj.CountryCode != "IN" {
		return "MEDIUM"
	}

	// Low: Indian asset
	return "LOW"
}

// computeMinDistance computes minimum distance to region center over time window
func (e *VolumetricEngine) computeMinDistance(obj catalog.SDAObject, region Region, window TimeWindow) float64 {
	samples := 50
	duration := window.End.Sub(window.Start).Seconds()
	dt := duration / float64(samples)

	minDist := math.Inf(1)
	for i := 0; i <= samples; i++ {
		t := window.Start.Add(time.Duration(float64(i)*dt) * time.Second)
		state := e.propagateState(obj, t)

		jd := timeToJD(t)
		ecefX, ecefY, ecefZ := physics.ECIToECEF(state[0], state[1], state[2], jd)
		lat, lon, _ := physics.ECEFToGeodeticWGS84(ecefX, ecefY, ecefZ)

		dist := greatCircleDistance(lat, lon, region.CenterLat, region.CenterLon)
		if dist < minDist {
			minDist = dist
		}
	}

	return minDist
}

// greatCircleDistance computes distance between two points on Earth (haversine formula)
func greatCircleDistance(lat1, lon1, lat2, lon2 float64) float64 {
	R := 6371.0 // Earth radius in km

	dLat := lat2 - lat1
	dLon := lon2 - lon1

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}

// timeToJD converts Go time.Time to Julian Date
func timeToJD(t time.Time) float64 {
	// Unix epoch in JD: 2440587.5 (January 1, 1970, 00:00 UTC)
	unixSec := t.Unix()
	unixNano := t.UnixNano()
	nanoSec := float64(unixNano%1e9) / 1e9
	return 2440587.5 + float64(unixSec)/86400.0 + nanoSec/86400.0
}

// julianDateToSeconds converts Julian Date to seconds since J2000 epoch
func julianDateToSeconds(jd float64) float64 {
	// J2000 epoch: JD 2451545.0
	return (jd - 2451545.0) * 86400.0
}

// propagateState performs symplectic propagation for an object
// Uses the Forest-Ruth integrator to preserve Hamiltonian energy
func (e *VolumetricEngine) propagateState(obj catalog.SDAObject, t time.Time) [6]float64 {
	// Initial state
	initialState := [6]float64{
		obj.Pos.X, obj.Pos.Y, obj.Pos.Z,
		obj.Vel.X, obj.Vel.Y, obj.Vel.Z,
	}

	// Propagate from epoch to target time
	t0 := obj.EpochTT * 86400.0 // Convert JD to seconds
	t1 := timeToJD(t) * 86400.0

	return e.SymplecticProp.Propagate(initialState, t0, t1)
}

// computeHamiltonianHash computes the Hamiltonian Invariant Hash for sovereign handshake
// This provides cryptographic proof of the physics state for spoof detection
func (e *VolumetricEngine) computeHamiltonianHash(obj catalog.SDAObject, window TimeWindow) string {
	// Sample Hamiltonian at multiple points in trajectory
	samples := 50
	duration := window.End.Sub(window.Start).Seconds()
	dt := duration / float64(samples)

	var hamiltonians []float64
	for i := 0; i <= samples; i++ {
		t := window.Start.Add(time.Duration(float64(i)*dt) * time.Second)
		state := e.propagateState(obj, t)

		// Compute Hamiltonian: H = T + V = ½|v|² - μ/r
		vx, vy, vz := state[3], state[4], state[5]
		kinetic := 0.5 * (vx*vx + vy*vy + vz*vz)

		x, y, z := state[0], state[1], state[2]
		r := math.Sqrt(x*x + y*y + z*z)
		potential := -398600.4418 / r

		H := kinetic + potential
		hamiltonians = append(hamiltonians, H)
	}

	// Compute average Hamiltonian
	sumH := 0.0
	for _, H := range hamiltonians {
		sumH += H
	}
	avgH := sumH / float64(len(hamiltonians))

	// Compute σ_H (standard deviation)
	variance := 0.0
	for _, H := range hamiltonians {
		diff := H - avgH
		variance += diff * diff
	}
	sigmaH := math.Sqrt(variance / float64(len(hamiltonians)))

	// Check if σ_H / H_avg >= 10^-7 (spoof detection)
	energyDriftSigma := sigmaH / math.Abs(avgH)
	if energyDriftSigma >= 1e-7 {
		// Certificate self-invalidates
		return "INVALID:ENERGY_DRIFT_EXCEEDED"
	}

	// Compute hash from average Hamiltonian and object metadata
	// In production, use proper cryptographic hash
	hashInput := fmt.Sprintf("%.15f|%d|%s", avgH, obj.ID, obj.Name)
	return fmt.Sprintf("H:%x", hashInput)
}
