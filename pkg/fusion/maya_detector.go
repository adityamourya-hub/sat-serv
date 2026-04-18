package fusion

import (
	"fmt"
	"math"
	"strings"
)

// DeceptionType categorizes the type of counterspace deception detected.
type DeceptionType string

const (
	DeceptionNone             DeceptionType = "none"
	DeceptionSpoofing        DeceptionType = "spoofing"
	DeceptionDecoy           DeceptionType = "decoy"
	DeceptionCamouflage      DeceptionType = "camouflage"
	DeceptionTrajectoryMask  DeceptionType = "trajectory_mask"
	DeceptionCatalogPoison   DeceptionType = "catalog_poison"
)

// DeceptionEvidence holds one check result.
type DeceptionEvidence struct {
	CheckName    string             `json:"check_name"`
	Triggered    bool               `json:"triggered"`
	Severity     float64            `json:"severity"` // [0, 1]
	Description  string             `json:"description"`
	PhysicsBasis string             `json:"physics_basis"`
	Metrics      map[string]float64 `json:"metrics,omitempty"`
}

// MayaAssessment is the aggregate deception assessment for an object.
type MayaAssessment struct {
	ObjectID       int                 `json:"object_id"`
	MayaScore      float64             `json:"maya_score"` // [0, 1]
	DeceptionType  DeceptionType       `json:"deception_type"`
	Evidence       []DeceptionEvidence `json:"evidence"`
	Recommendation string              `json:"recommendation"`
	Confidence     float64             `json:"confidence"`
	Summary        string              `json:"summary"`
}

// ObjectState represents a single orbital state vector.
type ObjectState struct {
	EpochJD          float64 `json:"epoch_jd"`
	X, Y, Z          float64 `json:"x,y,z"`       // km ECI
	VX, VY, VZ       float64 `json:"vx,vy,vz"`    // km/s ECI
	DetectedManeuver bool    `json:"detected_maneuver"`
}

// ManeuverEvent represents a detected maneuver.
type ManeuverEvent struct {
	EpochJD      float64    `json:"epoch_jd"`
	DeltaVMS     float64    `json:"delta_v_ms"`     // m/s
	DirectionECI [3]float64 `json:"direction_eci"`
	DurationS    float64    `json:"duration_s"`
}

// TrackEvent represents a tracking status at an epoch.
type TrackEvent struct {
	EpochJD       float64 `json:"epoch_jd"`
	Tracked       bool    `json:"tracked"`
	NObservations int     `json:"n_observations"`
	ResidualRMS   float64 `json:"residual_rms"`
}

const muEarth = 3.986004418e5 // km³/s²

// MayaDetector performs physics-grounded counterspace deception detection.
type MayaDetector struct {
	Weights map[string]float64
}

// NewMayaDetector creates a detector with default check weights.
func NewMayaDetector() *MayaDetector {
	return &MayaDetector{
		Weights: map[string]float64{
			"hamiltonian":  0.35,
			"rcs_orbit":    0.25,
			"maneuver_pat": 0.25,
			"catalog":      0.15,
		},
	}
}

// Detect runs all deception checks and produces an aggregate assessment.
func (d *MayaDetector) Detect(objectID int, states []ObjectState,
	rcsHistory []float64, catalogType string, altitudeKm float64,
	maneuvers []ManeuverEvent, tracks []TrackEvent) MayaAssessment {

	evidence := make([]DeceptionEvidence, 0, 4)

	evidence = append(evidence, d.checkHamiltonian(states))
	evidence = append(evidence, d.checkRCS(rcsHistory, catalogType, altitudeKm))
	evidence = append(evidence, d.checkManeuverPattern(maneuvers))
	evidence = append(evidence, d.checkCatalog(tracks))

	// Aggregate maya score
	mayaScore := 0.0
	for _, e := range evidence {
		w := d.Weights[e.CheckName]
		if w == 0 {
			w = 0.25
		}
		mayaScore += w * e.Severity
	}
	mayaScore = math.Min(mayaScore, 1.0)

	deceptionType := d.classifyDeception(evidence)
	recommendation := d.recommend(mayaScore)

	// Data completeness → confidence
	completeness := 0.0
	if len(states) >= 5 {
		completeness += 0.25
	}
	if len(rcsHistory) >= 3 {
		completeness += 0.25
	}
	if len(maneuvers) >= 2 {
		completeness += 0.25
	}
	if len(tracks) >= 5 {
		completeness += 0.25
	}

	triggered := make([]string, 0)
	for _, e := range evidence {
		if e.Triggered {
			triggered = append(triggered, e.Description)
		}
	}
	summary := fmt.Sprintf("Object %d: Maya score %.2f (%s). %s",
		objectID, mayaScore, deceptionType,
		strings.Join(triggered, "; "))

	return MayaAssessment{
		ObjectID:       objectID,
		MayaScore:      mayaScore,
		DeceptionType:  deceptionType,
		Evidence:       evidence,
		Recommendation: recommendation,
		Confidence:     completeness,
		Summary:        summary,
	}
}

func (d *MayaDetector) checkHamiltonian(states []ObjectState) DeceptionEvidence {
	if len(states) < 3 {
		return DeceptionEvidence{CheckName: "hamiltonian", Triggered: false, Severity: 0,
			Description: "Insufficient history", PhysicsBasis: "N/A"}
	}

	energies := make([]float64, len(states))
	for i, s := range states {
		r := math.Sqrt(s.X*s.X + s.Y*s.Y + s.Z*s.Z)
		v := math.Sqrt(s.VX*s.VX + s.VY*s.VY + s.VZ*s.VZ)
		energies[i] = 0.5*v*v - muEarth/r
	}

	unexplained := 0
	maxDE := 0.0
	for i := 1; i < len(energies); i++ {
		dE := math.Abs(energies[i] - energies[i-1])
		if dE > 0.1 && !states[i].DetectedManeuver {
			unexplained++
			if dE > maxDE {
				maxDE = dE
			}
		}
	}

	if unexplained > 0 {
		severity := math.Min(maxDE/1.0, 1.0)
		return DeceptionEvidence{
			CheckName: "hamiltonian", Triggered: true, Severity: severity,
			Description: fmt.Sprintf("%d unexplained energy jump(s), max ΔE=%.3f km²/s²", unexplained, maxDE),
			PhysicsBasis: "H = ½v² - μ/r is conserved; unexplained changes indicate hidden maneuver or spoofed data",
			Metrics:      map[string]float64{"n_jumps": float64(unexplained), "max_dE": maxDE},
		}
	}
	return DeceptionEvidence{CheckName: "hamiltonian", Triggered: false, Severity: 0,
		Description: "Energy consistent", PhysicsBasis: "Conservation of energy"}
}

func (d *MayaDetector) checkRCS(rcs []float64, catalogType string, altKm float64) DeceptionEvidence {
	if len(rcs) == 0 {
		return DeceptionEvidence{CheckName: "rcs_orbit", Triggered: false, Severity: 0,
			Description: "No RCS data", PhysicsBasis: "N/A"}
	}

	// Expected RCS ranges
	expected := map[string][2]float64{
		"cubesat":     {0.001, 0.05},
		"smallsat":    {0.05, 0.5},
		"medium_sat":  {0.5, 5.0},
		"large_sat":   {5.0, 50.0},
		"debris":      {0.0001, 1.0},
		"rocket_body": {1.0, 20.0},
	}

	median := sortedMedian(rcs)
	rng, ok := expected[catalogType]
	if !ok {
		rng = [2]float64{0.01, 10.0}
	}

	inRange := median >= rng[0] && median <= rng[1]
	altAnomaly := altKm < 400 && median > 20.0

	severity := 0.0
	parts := make([]string, 0)
	if !inRange {
		severity += 0.4
		parts = append(parts, fmt.Sprintf("RCS %.2fm² outside [%.3f, %.1f] for %s", median, rng[0], rng[1], catalogType))
	}
	if altAnomaly {
		severity += 0.3
		parts = append(parts, fmt.Sprintf("Large RCS at low alt (%.0fkm) — possible decoy", altKm))
	}

	triggered := !inRange || altAnomaly
	desc := "RCS consistent"
	if len(parts) > 0 {
		desc = strings.Join(parts, "; ")
	}

	return DeceptionEvidence{
		CheckName: "rcs_orbit", Triggered: triggered, Severity: math.Min(severity, 1.0),
		Description: desc, PhysicsBasis: "RCS correlates with object size; mismatches indicate decoys",
		Metrics: map[string]float64{"rcs_median": median},
	}
}

func (d *MayaDetector) checkManeuverPattern(maneuvers []ManeuverEvent) DeceptionEvidence {
	if len(maneuvers) < 2 {
		return DeceptionEvidence{CheckName: "maneuver_pat", Triggered: false, Severity: 0,
			Description: "Insufficient maneuver history", PhysicsBasis: "N/A"}
	}

	patterns := make([]string, 0)

	// Rapid small maneuvers (decoy deployment)
	rapidSmall := 0
	for i := 1; i < len(maneuvers); i++ {
		dt := (maneuvers[i].EpochJD - maneuvers[i-1].EpochJD) * 86400
		if dt < 300 && maneuvers[i].DeltaVMS < 0.5 {
			rapidSmall++
		}
	}
	if rapidSmall >= 3 {
		patterns = append(patterns, "rapid_small_maneuvers")
	}

	// Oscillatory thrust
	if len(maneuvers) >= 4 {
		signChanges := 0
		for i := 1; i < len(maneuvers)-1; i++ {
			dot1 := dotProduct(maneuvers[i-1].DirectionECI, maneuvers[i].DirectionECI)
			dot2 := dotProduct(maneuvers[i].DirectionECI, maneuvers[i+1].DirectionECI)
			if dot1*dot2 < 0 {
				signChanges++
			}
		}
		if signChanges >= 2 {
			patterns = append(patterns, "oscillatory_thrust")
		}
	}

	triggered := len(patterns) > 0
	severity := math.Min(float64(len(patterns))*0.35, 1.0)
	desc := "No deception patterns"
	if triggered {
		desc = strings.Join(patterns, "; ")
	}

	return DeceptionEvidence{
		CheckName: "maneuver_pat", Triggered: triggered, Severity: severity,
		Description: desc, PhysicsBasis: "Known adversary tactics: decoy splits, oscillatory masking",
	}
}

func (d *MayaDetector) checkCatalog(tracks []TrackEvent) DeceptionEvidence {
	if len(tracks) < 5 {
		return DeceptionEvidence{CheckName: "catalog", Triggered: false, Severity: 0,
			Description: "Insufficient tracking history", PhysicsBasis: "N/A"}
	}

	appearances := 0
	disappearances := 0
	residualSpikes := 0
	for i := 1; i < len(tracks); i++ {
		if tracks[i].Tracked && !tracks[i-1].Tracked {
			appearances++
		}
		if !tracks[i].Tracked && tracks[i-1].Tracked {
			disappearances++
		}
		if tracks[i].ResidualRMS > 5.0 {
			residualSpikes++
		}
	}

	anomalyCount := appearances + disappearances + residualSpikes
	triggered := anomalyCount >= 2
	severity := math.Min(float64(anomalyCount)*0.2, 1.0)

	parts := make([]string, 0)
	if appearances > 0 {
		parts = append(parts, fmt.Sprintf("%d appearance(s)", appearances))
	}
	if disappearances > 0 {
		parts = append(parts, fmt.Sprintf("%d disappearance(s)", disappearances))
	}
	if residualSpikes > 0 {
		parts = append(parts, fmt.Sprintf("%d residual spike(s)", residualSpikes))
	}
	desc := "Track consistent"
	if len(parts) > 0 {
		desc = strings.Join(parts, "; ")
	}

	return DeceptionEvidence{
		CheckName: "catalog", Triggered: triggered, Severity: severity,
		Description: desc, PhysicsBasis: "Objects don't spontaneously appear/vanish without physics basis",
	}
}

func (d *MayaDetector) classifyDeception(evidence []DeceptionEvidence) DeceptionType {
	triggered := make(map[string]DeceptionEvidence)
	for _, e := range evidence {
		if e.Triggered {
			triggered[e.CheckName] = e
		}
	}
	if len(triggered) == 0 {
		return DeceptionNone
	}
	if _, ok := triggered["rcs_orbit"]; ok {
		if _, ok2 := triggered["maneuver_pat"]; ok2 {
			return DeceptionDecoy
		}
	}
	if _, ok := triggered["hamiltonian"]; ok {
		if _, ok2 := triggered["catalog"]; ok2 {
			return DeceptionSpoofing
		}
	}
	if _, ok := triggered["catalog"]; ok {
		return DeceptionCatalogPoison
	}
	if _, ok := triggered["rcs_orbit"]; ok {
		return DeceptionCamouflage
	}
	return DeceptionSpoofing
}

func (d *MayaDetector) recommend(score float64) string {
	if score > 0.7 {
		return "ESCALATE"
	}
	if score > 0.4 {
		return "CONFIRM_VISUAL"
	}
	if score > 0.2 {
		return "MONITOR"
	}
	return "NOMINAL"
}

// --- helpers ---

func sortedMedian(data []float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	sorted := make([]float64, n)
	copy(sorted, data)
	// Simple insertion sort for small slices
	for i := 1; i < n; i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2.0
}

func dotProduct(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}
