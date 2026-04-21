package sda

import (
	"math"

	"ssis/pkg/catalog"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

/**
 * Axiom-Zero: Hamiltonian-Gated Sovereign Engine
 *
 * Neighborhood Watch Module
 *
 * Implements RIC (Radial-In-track-Cross-track) frame analysis for
 * relative geometry screening and maneuver detection around Indian assets.
 */

// WatchBox defines the RIC screening volume around a primary asset
type WatchBox struct {
	RadialKm     float64 // ± Radial distance (km)
	InTrackKm    float64 // ± In-track distance (km)
	CrossTrackKm float64 // ± Cross-track distance (km)
}

// Secondary represents a secondary object being screened
type Secondary struct {
	ObjectID         int
	ObjectName       string
	CountryCode      string
	RICPosition      [3]float64 // [radial, in-track, cross-track] in km
	RICVelocity      [3]float64 // [radial, in-track, cross-track] in km/s
	TimeToClosest    float64    // Time to closest approach (seconds)
	MinDistanceKm    float64    // Minimum distance in RIC frame
	ManeuverDetected bool
	ManeuverType     string // "PROGRADE", "RETROGRADE", "PLANE_CHANGE", "NONE"
	ThreatLevel      string
}

// NeighborhoodWatch implements the RIC Intent Engine
type NeighborhoodWatch struct {
	Catalog         *catalog.Store
	SINDy           *physics.SINDy
	WatchBox        WatchBox
	Threshold3Sigma float64         // 3σ threshold for maneuver detection
	MetadataStore   *MetadataStore  // Fuel-capacity metadata
	RRISolver       *RRISolver      // Reachability Risk Index solver
	FuelAccountant  *FuelAccountant // Fuel depletion modeling
}

// NewNeighborhoodWatch creates a new neighborhood watch engine
func NewNeighborhoodWatch(cat *catalog.Store) *NeighborhoodWatch {
	metadataStore := NewMetadataStore()
	metadataStore.InitializeMetadataFromCatalog(cat)

	nw := &NeighborhoodWatch{
		Catalog:         cat,
		SINDy:           physics.NewSINDy(),
		WatchBox:        WatchBox{RadialKm: 2.0, InTrackKm: 50.0, CrossTrackKm: 10.0},
		Threshold3Sigma: 3.0,
		MetadataStore:   metadataStore,
	}

	// Initialize RRISolver with Watch (avoid circular dependency)
	nw.RRISolver = NewRRISolver(metadataStore, nw)

	// Initialize FuelAccountant
	nw.FuelAccountant = NewFuelAccountant(metadataStore, cat)

	return nw
}

// ScreenSecondaries performs RIC-based screening of all secondaries around a primary asset
// Algorithm:
// 1. Get primary state (position/velocity in ECI)
// 2. Build RIC frame from primary
// 3. Transform all secondaries to RIC coordinates
// 4. Filter by Watch Box bounds
// 5. Run SINDy maneuver detection on secondaries in box
func (nw *NeighborhoodWatch) ScreenSecondaries(primaryID int, window TimeWindow) []Secondary {
	// Get primary object
	primary, ok := nw.Catalog.GetRecord(primaryID)
	if !ok {
		return nil
	}

	// Build RIC frame from primary state
	primaryState := orbital.StateVector{
		Position: [3]float64{primary.Pos.X, primary.Pos.Y, primary.Pos.Z},
		Velocity: [3]float64{primary.Vel.X, primary.Vel.Y, primary.Vel.Z},
	}
	ricFrame := orbital.ComputeRICFrame(primaryState)

	// Get all secondaries from catalog
	allObjects := nw.Catalog.Snapshot()
	var secondaries []Secondary

	for _, obj := range allObjects {
		if obj.ID == primaryID {
			continue // Skip primary itself
		}

		// Compute relative position in RIC
		deltaPos := [3]float64{
			obj.Pos.X - primary.Pos.X,
			obj.Pos.Y - primary.Pos.Y,
			obj.Pos.Z - primary.Pos.Z,
		}

		radial, inTrack, crossTrack := ricFrame.TransformToRIC(deltaPos)

		// Check if within Watch Box
		if math.Abs(radial) <= nw.WatchBox.RadialKm &&
			math.Abs(inTrack) <= nw.WatchBox.InTrackKm &&
			math.Abs(crossTrack) <= nw.WatchBox.CrossTrackKm {

			// Compute relative velocity in RIC
			deltaVel := [3]float64{
				obj.Vel.X - primary.Vel.X,
				obj.Vel.Y - primary.Vel.Y,
				obj.Vel.Z - primary.Vel.Z,
			}
			radialVel, inTrackVel, crossTrackVel := ricFrame.TransformToRIC(deltaVel)

			// Build secondary object
			sec := Secondary{
				ObjectID:      obj.ID,
				ObjectName:    obj.Name,
				CountryCode:   obj.CountryCode,
				RICPosition:   [3]float64{radial, inTrack, crossTrack},
				RICVelocity:   [3]float64{radialVel, inTrackVel, crossTrackVel},
				MinDistanceKm: math.Sqrt(radial*radial + inTrack*inTrack + crossTrack*crossTrack),
			}

			// Run maneuver detection
			sec = nw.detectManeuver(sec, obj, window)

			secondaries = append(secondaries, sec)
		}
	}

	return secondaries
}

// detectManeuver uses SINDy to detect maneuvers in the secondary's trajectory
func (nw *NeighborhoodWatch) detectManeuver(sec Secondary, obj catalog.SDAObject, window TimeWindow) Secondary {
	// Build trajectory states for SINDy
	states := nw.buildTrajectoryStates(obj, window)

	// Run SINDy
	result, err := nw.SINDy.DiscoverManeuver(states, 10.0)
	if err != nil {
		return sec
	}

	// Check if thrust detected
	if result.ThrustDetected {
		sec.ManeuverDetected = true

		// Classify maneuver type based on RIC residual pattern
		sec.ManeuverType = nw.classifyManeuverType(result, sec)

		// Process fuel depletion with FuelAccountant (H100 Distillation integration)
		if nw.FuelAccountant != nil {
			// Calculate thrust direction from RIC velocity
			thrustDirection := [3]float64{
				sec.RICVelocity[0] / (math.Sqrt(sec.RICVelocity[0]*sec.RICVelocity[0]+sec.RICVelocity[1]*sec.RICVelocity[1]+sec.RICVelocity[2]*sec.RICVelocity[2]) + 1e-6),
				sec.RICVelocity[1] / (math.Sqrt(sec.RICVelocity[0]*sec.RICVelocity[0]+sec.RICVelocity[1]*sec.RICVelocity[1]+sec.RICVelocity[2]*sec.RICVelocity[2]) + 1e-6),
				sec.RICVelocity[2] / (math.Sqrt(sec.RICVelocity[0]*sec.RICVelocity[0]+sec.RICVelocity[1]*sec.RICVelocity[1]+sec.RICVelocity[2]*sec.RICVelocity[2]) + 1e-6),
			}
			nw.FuelAccountant.ProcessSINDyManeuver(uint32(obj.ID), result.ThrustMagnitude, thrustDirection)
		}
	}

	// Assign threat level
	sec.ThreatLevel = nw.classifyThreatLevel(sec, obj, result)

	return sec
}

// buildTrajectoryStates constructs state trajectory for SINDy analysis
func (nw *NeighborhoodWatch) buildTrajectoryStates(obj catalog.SDAObject, window TimeWindow) [][6]float64 {
	samples := 50
	states := make([][6]float64, samples)

	// Simple propagation (will be upgraded to symplectic)
	for i := 0; i < samples; i++ {
		states[i] = [6]float64{
			obj.Pos.X, obj.Pos.Y, obj.Pos.Z,
			obj.Vel.X, obj.Vel.Y, obj.Vel.Z,
		}
	}

	return states
}

// classifyManeuverType classifies the maneuver based on RIC residual pattern
// Algorithm:
// - If residual aligns with velocity vector → Prograde/Retrograde Maneuver
// - If residual is cross-track → Plane-Change/Inclination Adjustment
func (nw *NeighborhoodWatch) classifyManeuverType(result *physics.ManeuverResult, sec Secondary) string {
	// Get velocity magnitude
	velMag := math.Sqrt(sec.RICVelocity[0]*sec.RICVelocity[0] +
		sec.RICVelocity[1]*sec.RICVelocity[1] +
		sec.RICVelocity[2]*sec.RICVelocity[2])

	if velMag < 1e-6 {
		return "UNKNOWN"
	}

	// Check if thrust aligns with in-track (velocity) direction
	// In-track is index 1 in RIC
	if math.Abs(result.ThrustMagnitude) > 0.01 {
		// Dominant in-track component
		if math.Abs(sec.RICPosition[1]) > math.Abs(sec.RICPosition[0]) &&
			math.Abs(sec.RICPosition[1]) > math.Abs(sec.RICPosition[2]) {
			if sec.RICVelocity[1] > 0 {
				return "PROGRADE"
			}
			return "RETROGRADE"
		}

		// Dominant cross-track component
		if math.Abs(sec.RICPosition[2]) > math.Abs(sec.RICPosition[0]) &&
			math.Abs(sec.RICPosition[2]) > math.Abs(sec.RICPosition[1]) {
			return "PLANE_CHANGE"
		}
	}

	return "UNKNOWN"
}

// classifyThreatLevel assigns threat level based on RIC proximity and maneuver
func (nw *NeighborhoodWatch) classifyThreatLevel(sec Secondary, obj catalog.SDAObject, _result *physics.ManeuverResult) string {
	// Critical: Foreign object with active maneuver in close proximity
	if sec.ManeuverDetected && obj.CountryCode != "IN" && sec.MinDistanceKm < 5.0 {
		return "CRITICAL"
	}

	// High: Foreign object with maneuver
	if sec.ManeuverDetected && obj.CountryCode != "IN" {
		return "HIGH"
	}

	// Medium: Foreign object in Watch Box
	if obj.CountryCode != "IN" {
		return "MEDIUM"
	}

	// Low: Indian asset
	return "LOW"
}

// ComputeReachabilitySet calculates the ΔV required for an adversary to intercept
// the primary asset using Hill-Clohessy-Wiltshire (HCW) equations.
//
// The Reachability Set R(T) is the volume in state space reachable within time T
// given current state x0 and total velocity budget ΔV_total.
//
// Physics: δx(t) = Φ(t)δx(0) + Ψ(t)Δv
// For intercept (δr(T) = 0): Δv_intercept = -Ψ_pos^{-1}(T)Φ_pos(T)δx(0)
func (nw *NeighborhoodWatch) ComputeReachabilitySet(primaryID int, secondaryID int, horizonSeconds float64) (deltaV float64, interceptTime float64, feasible bool) {
	primary, ok := nw.Catalog.GetRecord(primaryID)
	if !ok {
		return 0, 0, false
	}

	secondary, ok := nw.Catalog.GetRecord(secondaryID)
	if !ok {
		return 0, 0, false
	}

	// Build RIC frame
	primaryState := orbital.StateVector{
		Position: [3]float64{primary.Pos.X, primary.Pos.Y, primary.Pos.Z},
		Velocity: [3]float64{primary.Vel.X, primary.Vel.Y, primary.Vel.Z},
	}
	ricFrame := orbital.ComputeRICFrame(primaryState)

	// Relative position in RIC
	deltaPos := [3]float64{
		secondary.Pos.X - primary.Pos.X,
		secondary.Pos.Y - primary.Pos.Y,
		secondary.Pos.Z - primary.Pos.Z,
	}
	radial, inTrack, crossTrack := ricFrame.TransformToRIC(deltaPos)

	// Relative velocity in RIC
	deltaVel := [3]float64{
		secondary.Vel.X - primary.Vel.X,
		secondary.Vel.Y - primary.Vel.Y,
		secondary.Vel.Z - primary.Vel.Z,
	}
	radialVel, inTrackVel, crossTrackVel := ricFrame.TransformToRIC(deltaVel)

	// Initial relative state in RIC: δx(0) = [x, y, z, ẋ, ẏ, ż]^T
	deltaX0 := [6]float64{radial, inTrack, crossTrack, radialVel, inTrackVel, crossTrackVel}

	// Calculate primary's mean motion (n = sqrt(μ/r³))
	r := math.Sqrt(primary.Pos.X*primary.Pos.X + primary.Pos.Y*primary.Pos.Y + primary.Pos.Z*primary.Pos.Z)
	mu := 398600.4418 // km³/s²
	n := math.Sqrt(mu / (r * r * r))

	// Solve HCW state transition for intercept
	// Δv_intercept = -Ψ_pos^{-1}(T)Φ_pos(T)δx(0)
	PhiPos, PsiPos := nw.hcwStateTransitionMatrices(n, horizonSeconds)
	deltaVIntercept := nw.solveInterceptDeltaV(PhiPos, PsiPos, deltaX0)

	// Calculate required ΔV magnitude
	deltaV = math.Sqrt(deltaVIntercept[0]*deltaVIntercept[0] +
		deltaVIntercept[1]*deltaVIntercept[1] +
		deltaVIntercept[2]*deltaVIntercept[2])

	interceptTime = horizonSeconds

	// Feasibility check based on object classification
	// In production, cross-reference with ConvNeXt fuel capacity estimate
	fuelCapacity := nw.estimateFuelCapacity(secondary)
	feasible = deltaV < fuelCapacity

	return deltaV, interceptTime, feasible
}

// hcwStateTransitionMatrices computes the HCW state transition matrices Φ(t) and Ψ(t)
// for linearized relative motion around a circular orbit
func (nw *NeighborhoodWatch) hcwStateTransitionMatrices(n, t float64) (PhiPos, PsiPos [3][3]float64) {
	// HCW equations for circular orbits
	// Φ(t) and Ψ(t) are partitioned as:
	// Φ = [Φ_pos, Φ_vel; Φ_vel_dot, Φ_vel_dot_vel]
	// Ψ = [Ψ_pos, Ψ_vel; Ψ_vel_dot, Ψ_vel_dot_vel]

	nt := n * t
	c := math.Cos(nt)
	s := math.Sin(nt)

	// Φ_pos(t): position to position mapping
	PhiPos = [3][3]float64{
		{4 - 3*c, 0, 0},
		{6 * (s - nt), 1, 0},
		{0, 0, c},
	}

	// Ψ_pos(t): velocity to position mapping (impulse response)
	PsiPos = [3][3]float64{
		{(1 / n) * s, (2 / n) * (1 - c), 0},
		{-(2 / n) * (1 - c), (1 / n) * (4*s - 3*nt), 0},
		{0, 0, (1 / n) * s},
	}

	return PhiPos, PsiPos
}

// solveInterceptDeltaV solves for the required Δv to achieve intercept
// Δv_intercept = -Ψ_pos^{-1}(T)Φ_pos(T)δx(0)
func (nw *NeighborhoodWatch) solveInterceptDeltaV(PhiPos, PsiPos [3][3]float64, deltaX0 [6]float64) [3]float64 {
	// Extract position part of initial state
	deltaR0 := [3]float64{deltaX0[0], deltaX0[1], deltaX0[2]}

	// Compute Φ_pos(T)δr(0)
	PhiDeltaR := [3]float64{
		PhiPos[0][0]*deltaR0[0] + PhiPos[0][1]*deltaR0[1] + PhiPos[0][2]*deltaR0[2],
		PhiPos[1][0]*deltaR0[0] + PhiPos[1][1]*deltaR0[1] + PhiPos[1][2]*deltaR0[2],
		PhiPos[2][0]*deltaR0[0] + PhiPos[2][1]*deltaR0[1] + PhiPos[2][2]*deltaR0[2],
	}

	// Compute Ψ_pos^{-1}
	PsiInv := nw.invert3x3(PsiPos)

	// Δv = -Ψ_pos^{-1}Φ_posδr(0)
	deltaV := [3]float64{
		-(PsiInv[0][0]*PhiDeltaR[0] + PsiInv[0][1]*PhiDeltaR[1] + PsiInv[0][2]*PhiDeltaR[2]),
		-(PsiInv[1][0]*PhiDeltaR[0] + PsiInv[1][1]*PhiDeltaR[1] + PsiInv[1][2]*PhiDeltaR[2]),
		-(PsiInv[2][0]*PhiDeltaR[0] + PsiInv[2][1]*PhiDeltaR[1] + PsiInv[2][2]*PhiDeltaR[2]),
	}

	return deltaV
}

// invert3x3 inverts a 3x3 matrix
func (nw *NeighborhoodWatch) invert3x3(m [3][3]float64) [3][3]float64 {
	// Compute determinant
	det := m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

	if math.Abs(det) < 1e-10 {
		// Singular matrix, return identity
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}

	invDet := 1.0 / det

	// Compute inverse
	inv := [3][3]float64{
		{
			(m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet,
			(m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet,
			(m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet,
		},
		{
			(m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet,
			(m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet,
			(m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet,
		},
		{
			(m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet,
			(m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet,
			(m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet,
		},
	}

	return inv
}

// estimateFuelCapacity estimates fuel capacity based on object classification
// In production, this uses ConvNeXt/MGE classification
func (nw *NeighborhoodWatch) estimateFuelCapacity(obj catalog.SDAObject) float64 {
	// Simplified fuel capacity estimation based on asset type
	// In production: query ConvNeXt classification for fuel budget

	switch obj.AssetType {
	case "PAYLOAD":
		return 500.0 // m/s (limited maneuvering)
	case "ROCKET_BODY":
		return 100.0 // m/s (mostly dead)
	case "DEBRIS":
		return 0.0 // m/s (no propulsion)
	default:
		// Default: assume 1 km/s capability for maneuverable spacecraft
		return 1000.0 // m/s
	}
}
