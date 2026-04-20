// Package threat provides missile and hypersonic detection logic
// Implements 4D Tripwire for launch detection and IMM filter for hypersonic tracking
package threat

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/mesh"
	"ssis/pkg/spatial"
)

var (
	// Detection thresholds - loaded from ThreatConfig
	cfg = GetConfig()

	MachThreshold          = cfg.MachThreshold
	VerticalVelocityKmS    = cfg.VerticalVelocityKmS
	AltitudeBoostThreshold = cfg.AltitudeBoostThreshold
	PlumeTempMin           = cfg.PlumeTempMin
	PlasmaTempMin          = cfg.PlasmaTempMin

	// S2 cell levels
	LaunchCorridorLevel   = cfg.LaunchCorridorLevel
	RegionalTripwireLevel = cfg.RegionalTripwireLevel

	// IMM filter parameters
	IMMModelSwitchProb = cfg.IMMModelSwitchProb
	IMMUpdateRateHz    = cfg.IMMUpdateRateHz

	// Fractional Laplacian parameters
	FractionalOrder   = cfg.FractionalOrder
	WakeMemoryDepth   = cfg.WakeMemoryDepth
	WakeDecayFactor   = cfg.WakeDecayFactor
	HGVCorridorMinAlt = cfg.HGVCorridorMinAlt
	HGVCorridorMaxAlt = cfg.HGVCorridorMaxAlt
)

// DetectionPhase identifies the current phase of missile/hypersonic flight
type DetectionPhase int

const (
	PhaseUnknown  DetectionPhase = iota
	PhaseBoost                   // IR plume detection (3000-5000K)
	PhaseGlide                   // Plasma sheath (2000K+)
	PhaseTerminal                // Final approach
)

func (p DetectionPhase) String() string {
	return [...]string{"UNKNOWN", "BOOST", "GLIDE", "TERMINAL"}[p]
}

// MissileThreatType identifies the type of ballistic/hypersonic threat
type MissileThreatType int

const (
	MissileThreatUnknown MissileThreatType = iota
	MissileThreatBallistic
	MissileThreatHGV
	MissileThreatCruise
	MissileThreatDecoy
)

func (t MissileThreatType) String() string {
	return [...]string{"UNKNOWN", "BALLISTIC_MISSILE", "HGV", "CRUISE_MISSILE", "DECOY"}[t]
}

// LaunchCorridor defines a geographic region monitored for launches
type LaunchCorridor struct {
	Name        string
	MinLat      float64  // Degrees
	MaxLat      float64  // Degrees
	MinLon      float64  // Degrees
	MaxLon      float64  // Degrees
	MinAltKm    float64  // km
	MaxAltKm    float64  // km
	CellIDs     []uint64 // Pre-computed S2 cell IDs for this corridor
	ThreatLevel string   // HIGH/CRITICAL
}

// LaunchObservation represents a single sensor observation
type LaunchObservation struct {
	Timestamp    time.Time
	Position     [3]float64 // ECI coordinates (km)
	Velocity     [3]float64 // ECI velocity (km/s)
	Temperature  float64    // Kelvin (IR signature)
	Intensity    float64    // Normalized intensity (0-1)
	SourceSensor string     // Sensor identifier
	CellID       uint64     // S2 cell ID
}

// LaunchTripwireEngine monitors launch corridors for high-energy events
type LaunchTripwireEngine struct {
	Corridors    map[int]*LaunchCorridor // Keyed by corridor ID
	ActiveAlerts map[int]*LaunchAlert    // Keyed by alert ID
	NextAlertID  int
	AlertHistory []LaunchAlert
	S2Level      int

	// IMM filter state
	immFilters map[int]*IMMFilter // Keyed by track ID
}

// NewLaunchTripwireEngine creates a new launch detection engine
func NewLaunchTripwireEngine() *LaunchTripwireEngine {
	return &LaunchTripwireEngine{
		Corridors:    make(map[int]*LaunchCorridor),
		ActiveAlerts: make(map[int]*LaunchAlert),
		AlertHistory: make([]LaunchAlert, 0),
		S2Level:      LaunchCorridorLevel,
		immFilters:   make(map[int]*IMMFilter),
	}
}

// AddLaunchCorridor registers a geographic region for monitoring
func (lte *LaunchTripwireEngine) AddLaunchCorridor(id int, corridor *LaunchCorridor) {
	// Pre-compute S2 cell IDs for this corridor
	corridor.CellIDs = generateCorridorCells(corridor, lte.S2Level)
	lte.Corridors[id] = corridor
}

// CheckLaunchTripwire evaluates an observation against launch corridor tripwires
// Returns: (isLaunch, corridorID, confidence)
func (lte *LaunchTripwireEngine) CheckLaunchTripwire(obs LaunchObservation) (bool, int, float64) {
	// 1. Check if observation is in any launch corridor
	corridorID, inCorridor := lte.isInLaunchCorridor(obs.CellID)
	if !inCorridor {
		return false, 0, 0.0
	}

	// 2. Physics Check: Vertical velocity > 1.0 km/s (approx Mach 3)
	verticalVelocity := math.Abs(obs.Velocity[2]) // Z-component in ECI
	if verticalVelocity < VerticalVelocityKmS {
		return false, corridorID, 0.0
	}

	// 3. Altitude Check: Below 100km for boost phase
	altitude := math.Sqrt(obs.Position[0]*obs.Position[0]+obs.Position[1]*obs.Position[1]+obs.Position[2]*obs.Position[2]) - 6371.0
	if altitude > AltitudeBoostThreshold {
		return false, corridorID, 0.0
	}

	// 4. Temperature Check: IR plume signature (3000-5000K)
	if obs.Temperature < PlumeTempMin {
		return false, corridorID, 0.0
	}

	// 5. Calculate confidence based on signature strength
	confidence := lte.calculateLaunchConfidence(obs)

	return true, corridorID, confidence
}

// isInLaunchCorridor checks if an S2 cell ID is within any monitored corridor
func (lte *LaunchTripwireEngine) isInLaunchCorridor(cellID uint64) (int, bool) {
	for id, corridor := range lte.Corridors {
		for _, corridorCell := range corridor.CellIDs {
			if corridorCell == cellID {
				return id, true
			}
		}
	}
	return 0, false
}

// calculateLaunchConfidence computes confidence score for launch detection
func (lte *LaunchTripwireEngine) calculateLaunchConfidence(obs LaunchObservation) float64 {
	confidence := 0.0

	// Temperature contribution (higher is better)
	tempScore := (obs.Temperature - PlumeTempMin) / (5000.0 - PlumeTempMin)
	if tempScore > 0 {
		confidence += 0.4 * tempScore
	}

	// Velocity contribution
	verticalVel := math.Abs(obs.Velocity[2])
	velScore := (verticalVel - VerticalVelocityKmS) / (3.0 - VerticalVelocityKmS) // Up to 3 km/s
	if velScore > 0 {
		confidence += 0.3 * velScore
	}

	// Intensity contribution
	confidence += 0.3 * obs.Intensity

	return math.Min(confidence, 1.0)
}

// LaunchAlert represents a detected launch event
type LaunchAlert struct {
	AlertID        int
	AlertTime      time.Time
	CorridorID     int
	Observation    LaunchObservation
	Confidence     float64
	ThreatType     MissileThreatType
	DetectionPhase DetectionPhase
	CurrentMach    float64
	Azimuth        float64
	Elevation      float64
	IsConfirmed    bool
	TrackID        int

	// Symbolic certificate for sovereign handshake
	Certificate *SymbolicCertificate
}

// GenerateLaunchAlert creates a new launch alert from an observation
func (lte *LaunchTripwireEngine) GenerateLaunchAlert(obs LaunchObservation, corridorID int, confidence float64) *LaunchAlert {
	lte.NextAlertID++

	// Calculate current Mach number
	velocityMag := math.Sqrt(obs.Velocity[0]*obs.Velocity[0] + obs.Velocity[1]*obs.Velocity[1] + obs.Velocity[2]*obs.Velocity[2])
	currentMach := velocityMag * 3.6 // Convert km/s to Mach (approx)

	// Calculate azimuth and elevation (simplified)
	azimuth := math.Atan2(obs.Velocity[1], obs.Velocity[0]) * 180.0 / math.Pi
	elevation := math.Atan2(obs.Velocity[2], math.Sqrt(obs.Velocity[0]*obs.Velocity[0]+obs.Velocity[1]*obs.Velocity[1])) * 180.0 / math.Pi

	// Create state vector for certificate
	stateVector := [6]float64{
		obs.Position[0], obs.Position[1], obs.Position[2],
		obs.Velocity[0], obs.Velocity[1], obs.Velocity[2],
	}

	// Generate symbolic certificate with Hamiltonian invariant
	cert := NewSymbolicCertificate(
		fmt.Sprintf("LAUNCH-%d", lte.NextAlertID),
		obs.SourceSensor,
		stateVector,
		NewPhysicalConstraints(),
	)

	alert := &LaunchAlert{
		AlertID:        lte.NextAlertID,
		AlertTime:      time.Now(),
		CorridorID:     corridorID,
		Observation:    obs,
		Confidence:     confidence,
		ThreatType:     MissileThreatUnknown, // Will be determined by IMM filter
		DetectionPhase: PhaseBoost,           // Default to boost phase
		CurrentMach:    currentMach,
		Azimuth:        azimuth,
		Elevation:      elevation,
		IsConfirmed:    false,
		TrackID:        lte.NextAlertID, // Initial track ID
		Certificate:    cert,
	}

	lte.ActiveAlerts[alert.AlertID] = alert
	lte.AlertHistory = append(lte.AlertHistory, *alert)

	return alert
}

// generateCorridorCells generates S2 cell IDs for a launch corridor
func generateCorridorCells(corridor *LaunchCorridor, level int) []uint64 {
	cells := make([]uint64, 0)

	// Sample grid within corridor
	nSamples := 20
	dLat := (corridor.MaxLat - corridor.MinLat) / float64(nSamples)
	dLon := (corridor.MaxLon - corridor.MinLon) / float64(nSamples)

	for i := 0; i <= nSamples; i++ {
		lat := corridor.MinLat + float64(i)*dLat
		for j := 0; j <= nSamples; j++ {
			lon := corridor.MinLon + float64(j)*dLon
			cellID := spatial.LatLonToS2Cell(lat, lon, level)
			cells = append(cells, cellID)
		}
	}

	return cells
}

// WakeMemoryState stores historical state for fractional Laplacian dynamics
type WakeMemoryState struct {
	Position [3]float64
	Velocity [3]float64
	Time     time.Time
	Altitude float64
}

// IMM (Interacting Multiple Model) Filter for hypersonic tracking
// Handles model switching between ballistic boost and hypersonic glide
type IMMFilter struct {
	TrackID        int
	ModelA         *KalmanFilter // Constant Acceleration (Ballistic)
	ModelB         *KalmanFilter // Constant Turn (Hypersonic)
	ProbModelA     float64       // Probability weight for Model A
	ProbModelB     float64       // Probability weight for Model B
	LastUpdate     time.Time
	StateVector    []float64   // Fused state vector
	Covariance     [][]float64 // Fused covariance
	IsNonBallistic bool        // Flag for non-ballistic threat

	// Fractional Laplacian dynamics for HGV tracking
	WakeHistory []WakeMemoryState // Historical states for wake memory

	// DPHNO hard constraints
	Constraints *PhysicalConstraints
}

// KalmanFilter represents a standard Kalman filter for a specific motion model
type KalmanFilter struct {
	State      []float64
	Covariance [][]float64
	Q          [][]float64 // Process noise
	R          [][]float64 // Measurement noise
}

// NewIMMFilter creates a new IMM filter for hypersonic tracking
func NewIMMFilter(trackID int) *IMMFilter {
	return &IMMFilter{
		TrackID:        trackID,
		ProbModelA:     0.5, // Equal initial probability
		ProbModelB:     0.5,
		LastUpdate:     time.Now(),
		StateVector:    make([]float64, 6), // Position (3) + Velocity (3)
		Covariance:     make([][]float64, 6),
		IsNonBallistic: false,
		WakeHistory:    make([]WakeMemoryState, 0),
		Constraints:    NewPhysicalConstraints(),
		ModelA: &KalmanFilter{
			State:      make([]float64, 6),
			Covariance: make([][]float64, 6),
		},
		ModelB: &KalmanFilter{
			State:      make([]float64, 6),
			Covariance: make([][]float64, 6),
		},
	}
}

// calculateFractionalLaplacian computes fractional Laplacian for non-local dynamics
// Uses Grünwald-Letnikov approximation for fractional derivatives
func calculateFractionalLaplacian(history []WakeMemoryState, alpha float64, decay float64) [3]float64 {
	if len(history) < 2 {
		return [3]float64{0, 0, 0}
	}

	// Grünwald-Letnikov coefficients
	coeffs := make([]float64, len(history))
	coeffs[0] = 1.0
	for n := 1; n < len(history); n++ {
		coeffs[n] = coeffs[n-1] * (1.0 - (alpha+1.0)/float64(n))
	}

	// Apply fractional derivative to velocity history
	fractionalEffect := [3]float64{0, 0, 0}
	weightSum := 0.0

	for i, state := range history {
		weight := coeffs[i] * math.Pow(decay, float64(i))
		for j := 0; j < 3; j++ {
			fractionalEffect[j] += weight * state.Velocity[j]
		}
		weightSum += weight
	}

	if weightSum > 0 {
		for j := 0; j < 3; j++ {
			fractionalEffect[j] /= weightSum
		}
	}

	return fractionalEffect
}

// updateWakeHistory adds current state to wake memory with decay
func (imm *IMMFilter) updateWakeHistory(obs LaunchObservation) {
	altitude := math.Sqrt(obs.Position[0]*obs.Position[0]+obs.Position[1]*obs.Position[1]+obs.Position[2]*obs.Position[2]) - 6371.0

	state := WakeMemoryState{
		Position: obs.Position,
		Velocity: obs.Velocity,
		Time:     time.Now(),
		Altitude: altitude,
	}

	imm.WakeHistory = append(imm.WakeHistory, state)

	// Keep history bounded
	if len(imm.WakeHistory) > WakeMemoryDepth {
		imm.WakeHistory = imm.WakeHistory[1:]
	}
}

// isInHGVCorridor checks if current altitude is in HGV transitional flow regime
func isInHGVCorridor(altitude float64) bool {
	return altitude >= HGVCorridorMinAlt && altitude <= HGVCorridorMaxAlt
}

// UpdateIMMFilter updates the IMM filter with a new observation
func (imm *IMMFilter) UpdateIMMFilter(obs LaunchObservation) {
	// Update wake history for fractional dynamics
	imm.updateWakeHistory(obs)

	// Update both models
	imm.updateModel(imm.ModelA, obs, true)  // Ballistic model
	imm.updateModel(imm.ModelB, obs, false) // Hypersonic model

	// Calculate model probabilities based on residuals
	residualA := imm.calculateResidual(imm.ModelA, obs)
	residualB := imm.calculateResidual(imm.ModelB, obs)

	// Update probabilities (simplified Bayesian update)
	totalResidual := residualA + residualB
	imm.ProbModelA = residualA / totalResidual
	imm.ProbModelB = residualB / totalResidual

	// Fuse state vectors based on probabilities
	for i := 0; i < 6; i++ {
		imm.StateVector[i] = imm.ProbModelA*imm.ModelA.State[i] + imm.ProbModelB*imm.ModelB.State[i]
	}

	// Enforce DPHNO hard constraints on fused state
	if imm.Constraints != nil {
		pos := [3]float64{imm.StateVector[0], imm.StateVector[1], imm.StateVector[2]}
		vel := [3]float64{imm.StateVector[3], imm.StateVector[4], imm.StateVector[5]}

		// Check energy conservation
		energyViolated, energyRatio := imm.Constraints.CheckEnergyConservation(pos, vel, 1000.0)

		// Check momentum conservation (simplified acceleration)
		acc := [3]float64{0, 0, -0.0098} // Gravity
		momentumViolated, momentumRatio := imm.Constraints.CheckMomentumConservation(pos, vel, acc, 1000.0)

		// Apply augmented Lagrangian correction
		adjustedPos, adjustedVel := imm.Constraints.ApplyAugmentedLagrangian(
			pos, vel, energyViolated, energyRatio, momentumViolated, momentumRatio,
		)

		// Update state vector with constrained values
		for i := 0; i < 3; i++ {
			imm.StateVector[i] = adjustedPos[i]
			imm.StateVector[i+3] = adjustedVel[i]
		}
	}

	// Check for non-ballistic threat
	if imm.ProbModelB > IMMModelSwitchProb {
		imm.IsNonBallistic = true
	}

	imm.LastUpdate = time.Now()
}

// updateModel updates a single Kalman filter model
func (imm *IMMFilter) updateModel(kf *KalmanFilter, obs LaunchObservation, isBallistic bool) {
	// Simplified Kalman filter update
	// In production, implement full prediction and update steps

	// Measurement update
	for i := 0; i < 3; i++ {
		kf.State[i] = obs.Position[i]   // Direct position measurement
		kf.State[i+3] = obs.Velocity[i] // Direct velocity measurement
	}

	// Apply model-specific dynamics
	if isBallistic {
		// Constant acceleration model (gravity)
		kf.State[5] -= 0.0098 // Gravity in km/s²
	} else {
		// Hypersonic glide model with fractional Laplacian dynamics
		altitude := math.Sqrt(obs.Position[0]*obs.Position[0]+obs.Position[1]*obs.Position[1]+obs.Position[2]*obs.Position[2]) - 6371.0

		if isInHGVCorridor(altitude) && len(imm.WakeHistory) >= 2 {
			// Use fractional Laplacian for non-local dynamics in transitional flow regime
			fractionalEffect := calculateFractionalLaplacian(imm.WakeHistory, FractionalOrder, WakeDecayFactor)

			// Apply fractional dynamics to velocity
			for i := 0; i < 3; i++ {
				kf.State[i+3] = 0.7*kf.State[i+3] + 0.3*fractionalEffect[i]
			}
		} else {
			// Standard constant turn model for non-HGV corridor or insufficient history
			velMag := math.Sqrt(kf.State[3]*kf.State[3] + kf.State[4]*kf.State[4] + kf.State[5]*kf.State[5])
			if velMag > 0 {
				// Apply slight turn
				kf.State[3] *= 0.999
				kf.State[4] *= 1.001
			}
		}
	}
}

// calculateResidual calculates measurement residual for a model
func (imm *IMMFilter) calculateResidual(kf *KalmanFilter, obs LaunchObservation) float64 {
	residual := 0.0

	// Position residual
	for i := 0; i < 3; i++ {
		diff := kf.State[i] - obs.Position[i]
		residual += diff * diff
	}

	// Velocity residual
	for i := 0; i < 3; i++ {
		diff := kf.State[i+3] - obs.Velocity[i]
		residual += diff * diff
	}

	return math.Sqrt(residual)
}

// PredictImpact calculates predicted impact point and time
func (imm *IMMFilter) PredictImpact() ([2]float64, float64, float64) {
	// Simplified ballistic prediction
	// In production, use full orbital mechanics or atmospheric flight models

	// Current position
	lat := math.Asin(imm.StateVector[2]/6371.0) * 180.0 / math.Pi
	lon := math.Atan2(imm.StateVector[1], imm.StateVector[0]) * 180.0 / math.Pi

	// Current velocity magnitude
	velMag := math.Sqrt(imm.StateVector[3]*imm.StateVector[3] + imm.StateVector[4]*imm.StateVector[4] + imm.StateVector[5]*imm.StateVector[5])

	// Time to impact (simplified)
	altitude := math.Sqrt(imm.StateVector[0]*imm.StateVector[0]+imm.StateVector[1]*imm.StateVector[1]+imm.StateVector[2]*imm.StateVector[2]) - 6371.0
	timeToImpact := altitude / velMag

	// Predicted impact point (current trajectory)
	impactLat := lat + imm.StateVector[4]*timeToImpact/6371.0*180.0/math.Pi
	impactLon := lon + imm.StateVector[3]*timeToImpact/6371.0*180.0/math.Pi

	return [2]float64{impactLat, impactLon}, timeToImpact, imm.ProbModelB
}

// GetThreatClassification returns the threat type based on IMM filter state
func (imm *IMMFilter) GetThreatClassification() MissileThreatType {
	if imm.IsNonBallistic {
		return MissileThreatHGV
	}

	// Check for ballistic characteristics
	velMag := math.Sqrt(imm.StateVector[3]*imm.StateVector[3] + imm.StateVector[4]*imm.StateVector[4] + imm.StateVector[5]*imm.StateVector[5])

	if velMag > 5.0 { // >5 km/s (Mach 15+)
		return MissileThreatBallistic
	}

	return MissileThreatCruise
}

// NewMissileBrief creates a Red-Brief from a launch alert
func NewMissileBrief(alert *LaunchAlert, imm *IMMFilter, radarID string) *mesh.MissileBrief {
	// Calculate launch point from observation
	launchLat := math.Asin(alert.Observation.Position[2]/6371.0) * 180.0 / math.Pi
	launchLon := math.Atan2(alert.Observation.Position[1], alert.Observation.Position[0]) * 180.0 / math.Pi

	// Get prediction from IMM filter
	impactPoint, tti, confidence := imm.PredictImpact()

	// Get threat classification
	threatType := imm.GetThreatClassification().String()

	brief := &mesh.MissileBrief{
		TargetBrief: mesh.TargetBrief{
			MsgID:      generateMissileUUID(),
			Timestamp:  time.Now(),
			Priority:   mesh.PriorityCritical,
			OriginNode: "IAF_SPACE_COMMAND",
			NoradID:    alert.TrackID,
			CommonName: "MISSILE_THREAT_" + threatType,
			ActionCode: "RADAR_CUE_INITIATE",
		},
		BriefType:    "LAUNCH_EVENT",
		LaunchPoint:  [2]float64{launchLat, launchLon},
		Azimuth:      alert.Azimuth,
		CurrentMach:  alert.CurrentMach,
		ThreatType:   threatType,
		ImpactPoint:  impactPoint,
		TimeToImpact: tti,
		Confidence:   confidence,
		Phase:        alert.DetectionPhase.String(),
		Temperature:  alert.Observation.Temperature,
		RadarCue: mesh.RadarCue{
			RadarID:       radarID,
			CueLat:        impactPoint[0],
			CueLon:        impactPoint[1],
			CueAlt:        50.0,                                                // 50km cue altitude
			CueTime:       time.Now().Add(time.Duration(tti-60) * time.Second), // 1 minute before impact
			StareDuration: 300.0,                                               // 5 minutes stare
			BeamWidth:     2.0,                                                 // 2 degrees beam width
		},
	}

	return brief
}

// generateMissileUUID creates a simple UUID for missile message identification
func generateMissileUUID() string {
	return "MSL-" + time.Now().Format("20060102-150405.000")
}

// SDARadar represents a ground-based radar station for SDA fusion
type SDARadar struct {
	ID           string
	Name         string
	Location     [2]float64 // [lat, lon] degrees
	MaxRangeKm   float64
	MinAzimuth   float64
	MaxAzimuth   float64
	MinElevation float64
	MaxElevation float64
	Frequency    string // "X-BAND", "S-BAND"
	Status       string // "ACTIVE", "STANDBY", "MAINTENANCE"
}

// SDAManager manages Space Domain Awareness and ground radar cueing
type SDAManager struct {
	Radars          map[string]*SDARadar
	ActiveCues      map[int]*mesh.RadarCue // Keyed by alert ID
	HandoverHistory []SDAHandover
}

// SDAHandover represents a handoff from space-based to ground-based sensors
type SDAHandover struct {
	AlertID       int
	SpaceSensor   string
	RadarID       string
	HandoverTime  time.Time
	CueLat        float64
	CueLon        float64
	CueAlt        float64
	Acquisition   bool
	TimeToAcquire float64
}

// NewSDAManager creates a new SDA fusion manager
func NewSDAManager() *SDAManager {
	return &SDAManager{
		Radars:          make(map[string]*SDARadar),
		ActiveCues:      make(map[int]*mesh.RadarCue),
		HandoverHistory: make([]SDAHandover, 0),
	}
}

// AddRadar registers a ground radar for SDA fusion
func (sda *SDAManager) AddRadar(radar *SDARadar) {
	sda.Radars[radar.ID] = radar
}

// SelectOptimalRadar selects the best radar for cueing based on geometry
func (sda *SDAManager) SelectOptimalRadar(impactPoint [2]float64, timeToImpact float64) *SDARadar {
	bestRadar := (*SDARadar)(nil)
	bestScore := 0.0

	for _, radar := range sda.Radars {
		if radar.Status != "ACTIVE" {
			continue
		}

		// Calculate geometric score
		score := sda.calculateRadarScore(radar, impactPoint, timeToImpact)
		if score > bestScore {
			bestScore = score
			bestRadar = radar
		}
	}

	return bestRadar
}

// calculateRadarScore computes suitability score for a radar
func (sda *SDAManager) calculateRadarScore(radar *SDARadar, impactPoint [2]float64, timeToImpact float64) float64 {
	// Distance from radar to impact point
	radarLat := radar.Location[0]
	radarLon := radar.Location[1]

	dLat := impactPoint[0] - radarLat
	dLon := impactPoint[1] - radarLon

	// Haversine distance (simplified)
	distance := math.Sqrt(dLat*dLat+dLon*dLon) * 111.0 // Approx km

	// Score decreases with distance
	distanceScore := 1.0 - (distance / radar.MaxRangeKm)
	if distanceScore < 0 {
		distanceScore = 0
	}

	// Score increases with time to impact (allows radar preparation)
	timeScore := math.Min(timeToImpact/600.0, 1.0) // Max score at 10 minutes

	return 0.6*distanceScore + 0.4*timeScore
}

// ExecuteHandoff performs the handoff from space-based to ground-based sensors
func (sda *SDAManager) ExecuteHandoff(alert *LaunchAlert, imm *IMMFilter) *SDAHandover {
	// Get prediction from IMM filter
	impactPoint, timeToImpact, _ := imm.PredictImpact()

	// Select optimal radar
	radar := sda.SelectOptimalRadar(impactPoint, timeToImpact)
	if radar == nil {
		return nil
	}

	// Generate RadarCue
	cue := &mesh.RadarCue{
		RadarID:       radar.ID,
		CueLat:        impactPoint[0],
		CueLon:        impactPoint[1],
		CueAlt:        50.0,
		CueTime:       time.Now().Add(time.Duration(timeToImpact-60) * time.Second),
		StareDuration: 300.0,
		BeamWidth:     2.0,
	}

	sda.ActiveCues[alert.AlertID] = cue

	// Record handover
	handover := SDAHandover{
		AlertID:       alert.AlertID,
		SpaceSensor:   "IAF_SPACE_IR",
		RadarID:       radar.ID,
		HandoverTime:  time.Now(),
		CueLat:        cue.CueLat,
		CueLon:        cue.CueLon,
		CueAlt:        cue.CueAlt,
		Acquisition:   true, // Will be confirmed when radar acquires
		TimeToAcquire: timeToImpact - 60.0,
	}

	sda.HandoverHistory = append(sda.HandoverHistory, handover)

	return &handover
}

// InitializeIndianRadars sets up standard IAF ground radars
func (sda *SDAManager) InitializeIndianRadars() {
	// Arudhra Radar (Long-range surveillance)
	sda.AddRadar(&SDARadar{
		ID:           "ARUDHRA",
		Name:         "Arudhra LRTR",
		Location:     [2]float64{26.5, 91.0}, // Assam region
		MaxRangeKm:   400.0,
		MinAzimuth:   0.0,
		MaxAzimuth:   360.0,
		MinElevation: 0.0,
		MaxElevation: 90.0,
		Frequency:    "S-BAND",
		Status:       "ACTIVE",
	})

	// Ashwini Radar (Medium-range)
	sda.AddRadar(&SDARadar{
		ID:           "ASHWINI",
		Name:         "Ashwini MRTR",
		Location:     [2]float64{28.0, 77.0}, // Delhi region
		MaxRangeKm:   250.0,
		MinAzimuth:   0.0,
		MaxAzimuth:   360.0,
		MinElevation: 0.0,
		MaxElevation: 90.0,
		Frequency:    "X-BAND",
		Status:       "ACTIVE",
	})

	// Rohini Radar (Western sector)
	sda.AddRadar(&SDARadar{
		ID:           "ROHINI",
		Name:         "Rohini MRTR",
		Location:     [2]float64{23.0, 72.5}, // Gujarat region
		MaxRangeKm:   200.0,
		MinAzimuth:   0.0,
		MaxAzimuth:   360.0,
		MinElevation: 0.0,
		MaxElevation: 90.0,
		Frequency:    "X-BAND",
		Status:       "ACTIVE",
	})
}

// GetActiveCues returns all active radar cues
func (sda *SDAManager) GetActiveCues() map[int]*mesh.RadarCue {
	return sda.ActiveCues
}

// UpdateCueStatus updates the status of a radar cue after acquisition attempt
func (sda *SDAManager) UpdateCueStatus(alertID int, acquired bool) {
	if cue, exists := sda.ActiveCues[alertID]; exists {
		// Update handover history
		for i := range sda.HandoverHistory {
			if sda.HandoverHistory[i].AlertID == alertID {
				sda.HandoverHistory[i].Acquisition = acquired
				break
			}
		}

		// Remove from active cues if acquired or expired
		if acquired || time.Now().After(cue.CueTime.Add(time.Duration(cue.StareDuration)*time.Second)) {
			delete(sda.ActiveCues, alertID)
		}
	}
}

// GetHandoverStatistics returns statistics on handoff performance
func (sda *SDAManager) GetHandoverStatistics() map[string]interface{} {
	totalHandoffs := len(sda.HandoverHistory)
	successfulAcquisitions := 0

	for _, handover := range sda.HandoverHistory {
		if handover.Acquisition {
			successfulAcquisitions++
		}
	}

	successRate := 0.0
	if totalHandoffs > 0 {
		successRate = float64(successfulAcquisitions) / float64(totalHandoffs)
	}

	return map[string]interface{}{
		"total_handoffs":          totalHandoffs,
		"successful_acquisitions": successfulAcquisitions,
		"success_rate":            successRate,
		"active_cues":             len(sda.ActiveCues),
	}
}
