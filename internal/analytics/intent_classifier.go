// Package analytics provides Pattern-of-Life (PoL) based intent classification
// for maneuvering RSOs
package analytics

import (
	"math"
	"time"

	"ssis/pkg/physics"
)

// IntentCategory represents the classified intent of a maneuver
type IntentCategory int

const (
	IntentStationKeeping IntentCategory = iota
	IntentOrbitMaintenance
	IntentRendezvous
	IntentInspection
	IntentCoOrbitalThreat
	IntentDeorbit
	IntentUnknown
)

var intentNames = map[IntentCategory]string{
	IntentStationKeeping:   "STATION_KEEPING",
	IntentOrbitMaintenance: "ORBIT_MAINTENANCE",
	IntentRendezvous:       "RENDEZVOUS",
	IntentInspection:       "INSPECTION",
	IntentCoOrbitalThreat:  "CO_ORBITAL_THREAT",
	IntentDeorbit:          "DEORBIT",
	IntentUnknown:          "UNKNOWN",
}

// ManeuverHistory tracks observed maneuvers for pattern recognition
type ManeuverHistory struct {
	ObjectID         int32
	Events           []physics.ManeuverEvent
	TimeStamps       []time.Time
	DeltaVMagnitudes []float64
	Intervals        []float64 // days between maneuvers
}

// IntentClassifier determines maneuver intent based on pattern analysis
type IntentClassifier struct {
	history map[int32]*ManeuverHistory

	// Thresholds for classification
	rendezvousThreshold     float64 // km miss distance
	inspectionThreshold     float64 // km approach distance
	threatVelocityThreshold float64 // km/s relative velocity
	patternWindow           int     // number of events to analyze
}

// NewIntentClassifier creates intent classifier with default thresholds
func NewIntentClassifier() *IntentClassifier {
	return &IntentClassifier{
		history:                 make(map[int32]*ManeuverHistory),
		rendezvousThreshold:     5.0, // 5km for rendezvous
		inspectionThreshold:     1.0, // 1km for inspection
		threatVelocityThreshold: 0.5, // 500 m/s for co-orbital threat
		patternWindow:           5,
	}
}

// RecordManeuver stores a maneuver event for pattern analysis
func (ic *IntentClassifier) RecordManeuver(objID int32, event physics.ManeuverEvent, t time.Time, currentState physics.State, targetState physics.State) {
	h, exists := ic.history[objID]
	if !exists {
		h = &ManeuverHistory{ObjectID: objID}
		ic.history[objID] = h
	}

	h.Events = append(h.Events, event)
	h.TimeStamps = append(h.TimeStamps, t)
	h.DeltaVMagnitudes = append(h.DeltaVMagnitudes, event.DeltaV)

	// Calculate interval since last maneuver
	if len(h.TimeStamps) > 1 {
		interval := t.Sub(h.TimeStamps[len(h.TimeStamps)-2]).Hours() / 24.0
		h.Intervals = append(h.Intervals, interval)
	}
}

// ClassifyIntent determines intent from maneuver pattern
func (ic *IntentClassifier) ClassifyIntent(objID int32, currentState physics.State, targetState physics.State) IntentCategory {
	h, exists := ic.history[objID]
	if !exists || len(h.Events) == 0 {
		return IntentUnknown
	}

	// Calculate metrics
	missDistance := ic.computeMissDistance(currentState, targetState)
	relVel := ic.computeRelativeVelocity(currentState, targetState)
	pattern := ic.analyzePattern(h)

	// Decision tree for intent classification

	// Co-orbital threat: high relative velocity, close approach
	if missDistance < ic.threatVelocityThreshold*10 && relVel > ic.threatVelocityThreshold {
		if pattern.Regularity > 0.8 && pattern.DeltaVConsistency < 0.3 {
			return IntentCoOrbitalThreat
		}
	}

	// Rendezvous: close approach, matching velocity
	if missDistance < ic.rendezvousThreshold && relVel < 0.1 {
		if len(h.Events) >= 2 {
			return IntentRendezvous
		}
	}

	// Inspection: very close approach, low velocity
	if missDistance < ic.inspectionThreshold && relVel < 0.05 {
		return IntentInspection
	}

	// Station keeping: regular, small maneuvers
	if pattern.Regularity > 0.7 && pattern.AvgDeltaV < 0.5 {
		return IntentStationKeeping
	}

	// Deorbit: large delta-v, increasing drag
	if pattern.AvgDeltaV > 100.0 && currentState.Pos[2] < 200.0 {
		return IntentDeorbit
	}

	// Orbit maintenance: moderate, regular maneuvers
	if pattern.Regularity > 0.5 && pattern.AvgDeltaV < 5.0 {
		return IntentOrbitMaintenance
	}

	return IntentUnknown
}

// ManeuverPattern stores pattern analysis results
type ManeuverPattern struct {
	Regularity        float64 // 0-1, higher = more regular timing
	DeltaVConsistency float64 // coefficient of variation of delta-v
	AvgDeltaV         float64 // average delta-v magnitude
	Frequency         float64 // maneuvers per week
}

func (ic *IntentClassifier) analyzePattern(h *ManeuverHistory) ManeuverPattern {
	pattern := ManeuverPattern{}

	if len(h.Events) == 0 {
		return pattern
	}

	// Average delta-v
	var sumDV float64
	for _, dv := range h.DeltaVMagnitudes {
		sumDV += dv
	}
	pattern.AvgDeltaV = sumDV / float64(len(h.DeltaVMagnitudes))

	// Frequency (maneuvers per week)
	if len(h.TimeStamps) >= 2 {
		totalDays := h.TimeStamps[len(h.TimeStamps)-1].Sub(h.TimeStamps[0]).Hours() / 24.0
		if totalDays > 0 {
			pattern.Frequency = float64(len(h.Events)) / (totalDays / 7.0)
		}
	}

	// Regularity: low variance in intervals = high regularity
	if len(h.Intervals) > 1 {
		mean, stddev := meanStdDev(h.Intervals)
		if mean > 0 {
			cv := stddev / mean
			pattern.Regularity = 1.0 / (1.0 + cv) // Inverse CV, normalized
		}
	}

	// DeltaV consistency: low CV = consistent
	if len(h.DeltaVMagnitudes) > 1 {
		mean, stddev := meanStdDev(h.DeltaVMagnitudes)
		if mean > 0 {
			pattern.DeltaVConsistency = stddev / mean
		}
	}

	return pattern
}

func (ic *IntentClassifier) computeMissDistance(a, b physics.State) float64 {
	dx := a.Pos[0] - b.Pos[0]
	dy := a.Pos[1] - b.Pos[1]
	dz := a.Pos[2] - b.Pos[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func (ic *IntentClassifier) computeRelativeVelocity(a, b physics.State) float64 {
	dvx := a.Vel[0] - b.Vel[0]
	dvy := a.Vel[1] - b.Vel[1]
	dvz := a.Vel[2] - b.Vel[2]
	return math.Sqrt(dvx*dvx + dvy*dvy + dvz*dvz)
}

func meanStdDev(values []float64) (mean, stddev float64) {
	if len(values) == 0 {
		return 0, 0
	}

	var sum float64
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(len(values))

	var variance float64
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values) - 1)
	stddev = math.Sqrt(variance)

	return mean, stddev
}

// IntentReport is the tactical output of intent classification
type IntentReport struct {
	ObjectID       int32
	Intent         IntentCategory
	Confidence     float64
	ThreatLevel    string // "LOW", "MEDIUM", "HIGH", "CRITICAL"
	PatternDetails ManeuverPattern
	Recommendation string
}

// GenerateReport creates a tactical report for an object
func (ic *IntentClassifier) GenerateReport(objID int32, currentState, targetState physics.State) IntentReport {
	intent := ic.ClassifyIntent(objID, currentState, targetState)
	pattern := ManeuverPattern{}

	if h, exists := ic.history[objID]; exists {
		pattern = ic.analyzePattern(h)
	}

	report := IntentReport{
		ObjectID:       objID,
		Intent:         intent,
		PatternDetails: pattern,
	}

	// Calculate confidence and threat level
	switch intent {
	case IntentCoOrbitalThreat:
		report.Confidence = 0.9
		report.ThreatLevel = "CRITICAL"
		report.Recommendation = "IMMEDIATE_TRACKING_PRIORITY"
	case IntentRendezvous:
		report.Confidence = 0.75
		report.ThreatLevel = "HIGH"
		report.Recommendation = "ENHANCED_OBSERVATION"
	case IntentInspection:
		report.Confidence = 0.7
		report.ThreatLevel = "HIGH"
		report.Recommendation = "MAXIMUM_SENSOR_COVERAGE"
	case IntentDeorbit:
		report.Confidence = 0.6
		report.ThreatLevel = "MEDIUM"
		report.Recommendation = "TRACK_REENTRY_CORRIDOR"
	case IntentStationKeeping:
		report.Confidence = 0.8
		report.ThreatLevel = "LOW"
		report.Recommendation = "ROUTINE_MONITORING"
	case IntentOrbitMaintenance:
		report.Confidence = 0.75
		report.ThreatLevel = "LOW"
		report.Recommendation = "ROUTINE_MONITORING"
	default:
		report.Confidence = 0.3
		report.ThreatLevel = "MEDIUM"
		report.Recommendation = "INCREASE_OBSERVATION_FREQUENCY"
	}

	return report
}

func (i IntentCategory) String() string {
	if name, ok := intentNames[i]; ok {
		return name
	}
	return "UNKNOWN"
}
