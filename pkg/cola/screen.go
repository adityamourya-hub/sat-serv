// Package cola provides Collision Avoidance (COLA) and Conjunction Assessment (CA)
// Implements two-pass screening with S2 spatial filtering and Foster method for Pc calculation
package cola

import (
	"math"
	"math/rand/v2"
	"time"

	"ssis/pkg/spatial"
)

const (
	// Collision probability thresholds
	PcThresholdWarning   = 1e-5 // 0.001% - Warning
	PcThresholdCritical  = 1e-3 // 0.1% - Mandatory maneuver threshold
	PcThresholdEmergency = 1e-2 // 1% - Emergency action

	// Time bucketing for S2 screening
	TimeBucketSeconds = 10.0
)

// Conjunction represents a potential collision event
type Conjunction struct {
	PrimaryID         int            // Asset NORAD ID
	SecondaryID       int            // Debris/RSO NORAD ID
	TimeOfClosest     time.Time      // Time of closest approach
	MinimumDistance   float64        // km
	RelativeVelocity  float64        // km/s
	Probability       float64        // Probability of collision (Pc)
	MaxProbability    float64        // Pc with worst-case covariance
	EncounterPlane    EncounterPlane // B-plane parameters
	RecommendedAction string         // Action recommendation
}

// EncounterPlane defines the B-plane (encounter plane) perpendicular to relative velocity
type EncounterPlane struct {
	Bx             float64 // B-plane x coordinate (km)
	By             float64 // B-plane y coordinate (km)
	SigmaBx        float64 // Uncertainty in Bx (km)
	SigmaBy        float64 // Uncertainty in By (km)
	Correlation    float64 // Correlation between Bx and By
	HardBodyRadius float64 // Combined hard-body radius (km)
}

// Satellite represents a satellite/debris for conjunction screening
type Satellite struct {
	ID         int
	Position   [3]float64 // ECI position (km)
	Velocity   [3]float64 // ECI velocity (km/s)
	Covariance [6]float64 // Upper triangular of 3x3 covariance matrix (km^2, km^2/s, etc.)
	HardBody   float64    // Physical radius (km)
	Epoch      time.Time
}

// ConjunctionScreener performs two-pass conjunction screening
type ConjunctionScreener struct {
	s2Index    *spatial.S2CellIndex
	timeWindow time.Duration
	s2Level    int
}

// NewConjunctionScreener creates screener with S2 spatial index
func NewConjunctionScreener(s2Level int, timeWindow time.Duration) *ConjunctionScreener {
	return &ConjunctionScreener{
		s2Index:    spatial.NewS2CellIndex(s2Level),
		timeWindow: timeWindow,
		s2Level:    s2Level,
	}
}

// ScreenConjunctions performs two-pass conjunction screening
// Pass 1: S2 proximity filter (O(N) instead of O(N^2))
// Pass 2: Detailed Pc calculation for candidates
func (cs *ConjunctionScreener) ScreenConjunctions(
	asset *Satellite,
	catalog []*Satellite,
	startTime time.Time,
) []Conjunction {
	// Pass 1: S2 proximity filter
	candidates := cs.s2ProximityFilter(asset, catalog, startTime)

	// Pass 2: Detailed Pc calculation for candidates
	conjunctions := make([]Conjunction, 0)

	for _, candidate := range candidates {
		conj := cs.calculateConjunction(asset, candidate)
		if conj.Probability > PcThresholdWarning {
			conjunctions = append(conjunctions, conj)
		}
	}

	return conjunctions
}

// s2ProximityFilter uses S2 cells for fast proximity screening
func (cs *ConjunctionScreener) s2ProximityFilter(
	asset *Satellite,
	catalog []*Satellite,
	startTime time.Time,
) []*Satellite {
	candidates := make([]*Satellite, 0)

	// Bucket time into intervals
	numBuckets := int(cs.timeWindow.Seconds() / TimeBucketSeconds)

	for t := 0; t < numBuckets; t++ {
		assetTime := startTime.Add(time.Duration(t) * time.Second * TimeBucketSeconds)

		// Propagate asset to this time bucket (simplified - use actual propagator)
		assetPos := cs.propagatePosition(asset, assetTime)
		assetLat, assetLon := eciToLatLon(assetPos[0], assetPos[1], assetPos[2])
		assetCell := spatial.LatLonToS2Cell(assetLat, assetLon, cs.s2Level)

		// Check all catalog objects in same cell
		for _, obj := range catalog {
			objPos := cs.propagatePosition(obj, assetTime)
			objLat, objLon := eciToLatLon(objPos[0], objPos[1], objPos[2])
			objCell := spatial.LatLonToS2Cell(objLat, objLon, cs.s2Level)

			// If same S2 cell, add to candidates
			if assetCell == objCell {
				candidates = append(candidates, obj)
			}
		}
	}

	return candidates
}

// calculateConjunction computes detailed conjunction analysis using Foster method
func (cs *ConjunctionScreener) calculateConjunction(primary, secondary *Satellite) Conjunction {
	// Find time of closest approach (TCA)
	tca, minDist := cs.findTCA(primary, secondary)

	// Calculate relative velocity at TCA
	relVel := cs.calculateRelativeVelocity(primary, secondary, tca)

	// Compute encounter plane (B-plane)
	encounterPlane := cs.computeEncounterPlane(primary, secondary, tca)

	// Calculate probability of collision using Foster method
	pc := cs.fosterMethod(encounterPlane)

	// Determine recommended action
	action := cs.determineAction(pc, minDist, relVel)

	return Conjunction{
		PrimaryID:         primary.ID,
		SecondaryID:       secondary.ID,
		TimeOfClosest:     tca,
		MinimumDistance:   minDist,
		RelativeVelocity:  relVel,
		Probability:       pc,
		MaxProbability:    pc, // Would compute with worst-case covariance
		EncounterPlane:    encounterPlane,
		RecommendedAction: action,
	}
}

// findTCA finds time of closest approach using golden-section search
func (cs *ConjunctionScreener) findTCA(primary, secondary *Satellite) (time.Time, float64) {
	// Simplified: assume linear motion over short time window
	// In production, use numerical optimization with actual propagator

	t0 := primary.Epoch
	t1 := t0.Add(cs.timeWindow)

	minDist := math.MaxFloat64
	tca := t0

	// Coarse search
	for t := t0; t.Before(t1); t = t.Add(time.Minute) {
		dist := cs.distanceAtTime(primary, secondary, t)
		if dist < minDist {
			minDist = dist
			tca = t
		}
	}

	return tca, minDist
}

// distanceAtTime calculates distance between two satellites at given time
func (cs *ConjunctionScreener) distanceAtTime(primary, secondary *Satellite, t time.Time) float64 {
	p1 := cs.propagatePosition(primary, t)
	p2 := cs.propagatePosition(secondary, t)

	dx := p1[0] - p2[0]
	dy := p1[1] - p2[1]
	dz := p1[2] - p2[2]

	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// calculateRelativeVelocity computes relative velocity vector
func (cs *ConjunctionScreener) calculateRelativeVelocity(primary, secondary *Satellite, t time.Time) float64 {
	v1 := cs.propagateVelocity(primary, t)
	v2 := cs.propagateVelocity(secondary, t)

	dvx := v1[0] - v2[0]
	dvy := v1[1] - v2[1]
	dvz := v1[2] - v2[2]

	return math.Sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
}

// computeEncounterPlane calculates B-plane parameters
func (cs *ConjunctionScreener) computeEncounterPlane(primary, secondary *Satellite, t time.Time) EncounterPlane {
	// Get relative position and velocity at TCA
	r1 := cs.propagatePosition(primary, t)
	r2 := cs.propagatePosition(secondary, t)
	v1 := cs.propagateVelocity(primary, t)
	v2 := cs.propagateVelocity(secondary, t)

	// Relative state
	relPos := [3]float64{r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]}
	relVel := [3]float64{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]}

	// Normalize relative velocity to get encounter plane normal
	vMag := math.Sqrt(relVel[0]*relVel[0] + relVel[1]*relVel[1] + relVel[2]*relVel[2])
	kHat := [3]float64{relVel[0] / vMag, relVel[1] / vMag, relVel[2] / vMag}

	// Compute B-plane basis vectors
	// iHat: direction from Earth center to encounter point
	rMag := math.Sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2])
	iHat := [3]float64{r1[0] / rMag, r1[1] / rMag, r1[2] / rMag}

	// jHat: kHat x iHat
	jHat := [3]float64{
		kHat[1]*iHat[2] - kHat[2]*iHat[1],
		kHat[2]*iHat[0] - kHat[0]*iHat[2],
		kHat[0]*iHat[1] - kHat[1]*iHat[0],
	}

	// Project relative position onto B-plane
	bx := relPos[0]*jHat[0] + relPos[1]*jHat[1] + relPos[2]*jHat[2]
	by := relPos[0]*iHat[0] + relPos[1]*iHat[1] + relPos[2]*iHat[2]

	// Combined hard-body radius
	hardBody := primary.HardBody + secondary.HardBody

	// Simplified covariance propagation (in production, use UT)
	sigmaBx := 0.1 // km
	sigmaBy := 0.1 // km
	correlation := 0.0

	return EncounterPlane{
		Bx:             bx,
		By:             by,
		SigmaBx:        sigmaBx,
		SigmaBy:        sigmaBy,
		Correlation:    correlation,
		HardBodyRadius: hardBody,
	}
}

// fosterMethod calculates probability of collision using Foster method
// P_c = (1 / (2*pi*sigma_x*sigma_y)) * integral over hard-body area
func (cs *ConjunctionScreener) fosterMethod(ep EncounterPlane) float64 {
	// For uncorrelated case (correlation = 0)
	// Pc = exp(-R^2 / (2*sigma^2)) where R is hard-body radius
	// and sigma is combined uncertainty

	sigmaBx := ep.SigmaBx
	sigmaBy := ep.SigmaBy
	R := ep.HardBodyRadius

	// Combined uncertainty (assuming circular)
	sigma := math.Sqrt((sigmaBx*sigmaBx + sigmaBy*sigmaBy) / 2.0)

	// Foster method approximation for circular hard body
	p := math.Exp(-(R * R) / (2.0 * sigma * sigma))

	// Scale by 1/(2*pi) for proper normalization
	p = p / (2.0 * math.Pi)

	return p
}

// determineAction recommends action based on Pc and geometry
func (cs *ConjunctionScreener) determineAction(pc, minDist, relVel float64) string {
	if pc > PcThresholdEmergency {
		return "EMERGENCY_EVASIVE"
	}

	if pc > PcThresholdCritical {
		if relVel > 0.5 { // High relative velocity
			return "MANEUVER_IMMEDIATE"
		}
		return "MANEUVER_PLANNED"
	}

	if pc > PcThresholdWarning {
		return "MONITOR_INCREASED"
	}

	return "NO_ACTION"
}

// propagatePosition propagates satellite position to given time
// In production, use actual propagator (SGP4/HPOP)
func (cs *ConjunctionScreener) propagatePosition(sat *Satellite, t time.Time) [3]float64 {
	// Simplified: linear propagation
	dt := t.Sub(sat.Epoch).Seconds()

	return [3]float64{
		sat.Position[0] + sat.Velocity[0]*dt,
		sat.Position[1] + sat.Velocity[1]*dt,
		sat.Position[2] + sat.Velocity[2]*dt,
	}
}

// propagateVelocity propagates satellite velocity to given time
func (cs *ConjunctionScreener) propagateVelocity(sat *Satellite, t time.Time) [3]float64 {
	// Simplified: constant velocity (no perturbations)
	return sat.Velocity
}

// eciToLatLon converts ECI coordinates to latitude/longitude (simplified)
func eciToLatLon(x, y, z float64) (lat, lon float64) {
	r := math.Sqrt(x*x + y*y + z*z)
	lat = math.Asin(z/r) * 180.0 / math.Pi
	lon = math.Atan2(y, x) * 180.0 / math.Pi
	return lat, lon
}

// StrategicDenial provides strategic denial capabilities
type StrategicDenial struct {
	screener *ConjunctionScreener
}

// NewStrategicDenial creates strategic denial module
func NewStrategicDenial(screener *ConjunctionScreener) *StrategicDenial {
	return &StrategicDenial{
		screener: screener,
	}
}

// FuelTrap identifies adversary satellites vulnerable to fuel exhaustion
// by finding debris conjunctions that can be spoofed to high Pc
func (sd *StrategicDenial) FuelTrap(
	adversaryID int,
	adversary *Satellite,
	debrisCatalog []*Satellite,
	startTime time.Time,
) []Conjunction {
	// Find debris near adversary's orbit
	conjunctions := sd.screener.ScreenConjunctions(adversary, debrisCatalog, startTime)

	// Identify conjunctions with Pc just below threshold
	// These can be spoofed to force adversary maneuvers
	vulnerable := make([]Conjunction, 0)

	for _, conj := range conjunctions {
		if conj.Probability > PcThresholdWarning*10 && conj.Probability < PcThresholdCritical*0.9 {
			vulnerable = append(vulnerable, conj)
		}
	}

	return vulnerable
}

// SpoofPc artificially inflates Pc for fuel trap operations
// Returns spoofed conjunction data for transmission
func (sd *StrategicDenial) SpoofPc(conj Conjunction, targetPc float64) Conjunction {
	spoofed := conj
	spoofed.MaxProbability = targetPc

	// Adjust encounter plane to match target Pc
	// This requires inverse Foster method calculation
	spoofed.EncounterPlane.SigmaBx *= 0.5 // Reduce uncertainty to increase Pc
	spoofed.EncounterPlane.SigmaBy *= 0.5

	return spoofed
}

// SelectiveDataSharing generates degraded TLEs for public catalog
// while maintaining precise data for allies via Reticulum mesh
type SelectiveDataSharing struct {
	noiseLevel float64 // Amount of noise to add (km/s in velocity)
}

// NewSelectiveDataSharing creates data sharing module
func NewSelectiveDataSharing(noiseLevel float64) *SelectiveDataSharing {
	return &SelectiveDataSharing{
		noiseLevel: noiseLevel,
	}
}

// GeneratePublicTLE creates degraded TLE for public consumption
func (sds *SelectiveDataSharing) GeneratePublicTLE(sat *Satellite) Satellite {
	public := *sat

	// Add noise to velocity
	for i := 0; i < 3; i++ {
		public.Velocity[i] += (2.0*rand.Float64() - 1.0) * sds.noiseLevel
	}

	// Increase drag coefficient noise (B-star)
	// This makes orbit prediction less accurate

	return public
}

// GenerateAllyTLE provides precise TLE for ally consumption
func (sds *SelectiveDataSharing) GenerateAllyTLE(sat *Satellite) Satellite {
	return *sat // No degradation
}

// OrbitalShellHeatMap provides visualization of orbital shell density
type OrbitalShellHeatMap struct {
	shells map[float64]int // Altitude -> debris count
}

// NewOrbitalShellHeatMap creates heat map
func NewOrbitalShellHeatMap() *OrbitalShellHeatMap {
	return &OrbitalShellHeatMap{
		shells: make(map[float64]int),
	}
}

// AddSatellite adds satellite to appropriate altitude shell
func (ohm *OrbitalShellHeatMap) AddSatellite(sat *Satellite) {
	// Calculate altitude from position
	r := math.Sqrt(sat.Position[0]*sat.Position[0] + sat.Position[1]*sat.Position[1] + sat.Position[2]*sat.Position[2])
	altitude := r - 6371.0 // Earth radius (km)

	// Bucket to 10km shells
	shellAlt := math.Floor(altitude/10.0) * 10.0
	ohm.shells[shellAlt]++
}

// GetDensity returns debris density at given altitude
func (ohm *OrbitalShellHeatMap) GetDensity(altitude float64) int {
	shellAlt := math.Floor(altitude/10.0) * 10.0
	return ohm.shells[shellAlt]
}

// IsDenied determines if orbital shell is denied (high debris density)
func (ohm *OrbitalShellHeatMap) IsDenied(altitude float64) bool {
	density := ohm.GetDensity(altitude)
	return density > 100 // Threshold for "denied" shell
}

// GetDeniedShells returns list of denied altitude shells
func (ohm *OrbitalShellHeatMap) GetDeniedShells() []float64 {
	denied := make([]float64, 0)

	for altitude, density := range ohm.shells {
		if density > 100 {
			denied = append(denied, altitude)
		}
	}

	return denied
}
