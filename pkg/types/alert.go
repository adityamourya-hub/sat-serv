// Package types provides shared alert structures for the SSIS engine.
package types

import "time"

// AlertType identifies the semantic nature of an alert.
type AlertType int

const (
	AlertTypeUnknown AlertType = iota
	AlertTypeConjunction
	AlertTypeManeuver
	AlertTypeSensorTasking
	AlertTypeReentry
	AlertTypeCollisionRisk
	AlertTypeAnomaly
)

// AlertPriority defines the tactical urgency.
type AlertPriority int

const (
	AlertPriorityUnspecified AlertPriority = iota
	AlertPriorityLow
	AlertPriorityMedium
	AlertPriorityHigh
	AlertPriorityCritical
)

// SHAPAttribution provides feature-level explainability for ML detections.
type SHAPAttribution struct {
	FeatureScores       map[string]float64 // e.g. "vx_residual": 0.85
	BaseValue           float64            // Model baseline probability
	ExplanationSummary  string             // Human-readable justification
}

// Alert defines a critical event with causal explainability.
type Alert struct {
	AlertID      string
	Type         AlertType
	Priority     AlertPriority
	TargetID     string
	SecondaryID  string // For conjunctions
	Probability  float64
	Message      string
	KLDivergence float64 // ML divergence metric
	Explanation  *SHAPAttribution
	Timestamp    time.Time
	Signature    []byte // Ed25519 signature
}

// Reset clears the alert for reuse from a sync pool.
func (a *Alert) Reset() {
	a.AlertID = ""
	a.Type = AlertTypeUnknown
	a.Priority = AlertPriorityUnspecified
	a.TargetID = ""
	a.SecondaryID = ""
	a.Probability = 0
	a.Message = ""
	a.KLDivergence = 0
	a.Explanation = nil
	a.Timestamp = time.Time{}
	a.Signature = nil
}

// AlertSubscription defines filters for receiving real-time alerts.
type AlertSubscription struct {
	SubscriberID   string
	ObjectIDs      []string // Empty = all objects
	MinProbability float64
	MinPriority    AlertPriority
	AlertTypes     []AlertType // Empty = all types
}
