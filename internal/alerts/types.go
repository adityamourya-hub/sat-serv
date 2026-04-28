// Package alerts provides the sovereign alerting system for the SSIS Command Center.
// Classification: SOVEREIGN_INTEL / AIR_GAPPED_ONLY
package alerts

import (
	"time"
)

// Severity levels for the alerting system.
type Severity string

const (
	SeverityInfo      Severity = "INFO"
	SeverityWarning   Severity = "WARNING"
	SeverityCritical  Severity = "CRITICAL"
	SeverityEmergency Severity = "EMERGENCY"
)

// Category identifies the alert source domain.
type Category string

const (
	CategoryConjunction      Category = "CONJUNCTION_RISK"
	CategoryManeuver         Category = "MANEUVER_DETECTION"
	CategoryReentry          Category = "REENTRY_RISK"
	CategoryBlindSpot        Category = "BLIND_SPOT"
	CategorySensorHealth     Category = "SENSOR_HEALTH"
	CategoryAnomaly          Category = "BEHAVIOURAL_ANOMALY"
	CategoryIntrusion        Category = "SOVEREIGN_INTRUSION"
	CategoryCollisionRisk    Category = "COLLISION_RISK"
)

// Status tracks the alert lifecycle.
type Status string

const (
	StatusOpen         Status = "OPEN"
	StatusAcknowledged Status = "ACKNOWLEDGED"
	StatusResolved     Status = "RESOLVED"
	StatusExpired      Status = "EXPIRED"
	StatusEscalated    Status = "ESCALATED"
)

// Role for RBAC on WebSocket connections.
type Role string

const (
	RoleOperator  Role = "OPERATOR"
	RoleCommander Role = "COMMANDER"
	RoleSystem    Role = "SYSTEM"
)

// EnrichedAlert is the full alert payload sent to the dashboard and stored in PostgreSQL.
type EnrichedAlert struct {
	AlertID   string   `json:"alert_id"`
	CreatedAt time.Time `json:"created_at"`

	// Classification
	Category Category `json:"category"`
	Severity Severity `json:"severity"`
	Status   Status   `json:"status"`

	// Target objects
	PrimaryNoradID      int    `json:"primary_norad_id,omitempty"`
	PrimaryObjectName   string `json:"primary_object_name,omitempty"`
	SecondaryNoradID    int    `json:"secondary_norad_id,omitempty"`
	SecondaryObjectName string `json:"secondary_object_name,omitempty"`

	// Content
	Title             string `json:"title"`
	Message           string `json:"message"`
	RecommendedAction string `json:"recommended_action,omitempty"`

	// Quantitative
	Metrics      map[string]interface{} `json:"metrics,omitempty"`
	KLDivergence float64                `json:"kl_divergence,omitempty"`
	Confidence   float64                `json:"confidence,omitempty"`

	// Geospatial
	Latitude        float64 `json:"latitude,omitempty"`
	Longitude       float64 `json:"longitude,omitempty"`
	AltitudeKm      float64 `json:"altitude_km,omitempty"`
	SovereignVolume string  `json:"sovereign_volume,omitempty"`

	// Rules engine
	TriggeredRule string `json:"triggered_rule,omitempty"`

	// Acknowledgement
	AcknowledgedBy string     `json:"acknowledged_by,omitempty"`
	AcknowledgedAt *time.Time `json:"acknowledged_at,omitempty"`
	AckAction      string     `json:"ack_action,omitempty"`

	// Escalation
	EscalationDeadline *time.Time `json:"escalation_deadline,omitempty"`

	// Signature (hex-encoded Ed25519)
	Signature string `json:"signature,omitempty"`
}

// AuditEntry represents a tamper-evident audit log record.
type AuditEntry struct {
	LoggedAt      time.Time              `json:"logged_at"`
	ActorID       string                 `json:"actor_id"`
	ActorRole     Role                   `json:"actor_role"`
	Action        string                 `json:"action"`
	EntityType    string                 `json:"entity_type"`
	EntityID      string                 `json:"entity_id"`
	Details       map[string]interface{} `json:"details,omitempty"`
	PreviousState interface{}            `json:"previous_state,omitempty"`
	NewState      interface{}            `json:"new_state,omitempty"`
	Signature     string                 `json:"signature,omitempty"`
}

// WSMessage is the WebSocket envelope sent to dashboard clients.
type WSMessage struct {
	Type    string      `json:"type"`    // "ALERT", "ACK", "ESCALATION", "HEARTBEAT"
	Payload interface{} `json:"payload"`
	SentAt  time.Time   `json:"sent_at"`
}

// AckRequest is the body of POST /api/alerts/:id/ack.
type AckRequest struct {
	OperatorID string `json:"operator_id" binding:"required"`
	Action     string `json:"action" binding:"required"` // e.g. "MONITOR", "MANEUVER_ORDERED", "SENSOR_TASKED"
	Notes      string `json:"notes,omitempty"`
}

// SeverityOrder returns numeric priority (higher = more urgent).
func SeverityOrder(s Severity) int {
	switch s {
	case SeverityInfo:
		return 0
	case SeverityWarning:
		return 1
	case SeverityCritical:
		return 2
	case SeverityEmergency:
		return 3
	default:
		return -1
	}
}

// SeverityColour returns the CSS hex colour for each severity level.
func SeverityColour(s Severity) string {
	switch s {
	case SeverityInfo:
		return "#60A5FA"
	case SeverityWarning:
		return "#FACC15"
	case SeverityCritical:
		return "#F87171"
	case SeverityEmergency:
		return "#FF0000"
	default:
		return "#8892A8"
	}
}

// RoleCanView returns true if the role is allowed to see the given severity.
func RoleCanView(role Role, severity Severity) bool {
	switch role {
	case RoleCommander, RoleSystem:
		return true // Commander sees everything
	case RoleOperator:
		return severity == SeverityInfo || severity == SeverityWarning || severity == SeverityCritical
	default:
		return false
	}
}
