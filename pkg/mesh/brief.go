// Package mesh provides Tactical Intelligence Packets for Reticulum mesh transmission
// High-density TargetBrief optimized for low-bandwidth LoRa/RF links
package mesh

import (
	"encoding/json"
	"fmt"
	"time"
	// TODO: Re-enable when dependency available
	// "github.com/vmihailenco/msgpack/v5"
)

// TargetBrief is the tactical intelligence packet for IAF commanders
// Optimized for MessagePack serialization (~450 bytes)
type TargetBrief struct {
	// Header
	MsgID      string    `json:"msg_id" msgpack:"msg_id"` // UUID-V4
	Timestamp  time.Time `json:"ts" msgpack:"ts"`         // ISO-8601
	Priority   Priority  `json:"pri" msgpack:"pri"`       // CRITICAL|HIGH|ROUTINE
	OriginNode string    `json:"origin" msgpack:"origin"` // RNS address hash

	// Asset Identity
	NoradID        int    `json:"norad_id" msgpack:"norad_id"`   // 43864
	CosparID       string `json:"cospar_id" msgpack:"cospar_id"` // "2018-105A"
	CommonName     string `json:"name" msgpack:"name"`           // "GSAT-7A_ADVERSARY_PROXY"
	CountryCode    string `json:"cc" msgpack:"cc"`               // "CHN"
	Classification string `json:"cls" msgpack:"cls"`             // ADVERSARY_ISR

	// Orbital State (6-DOF)
	Epoch          time.Time  `json:"epoch" msgpack:"epoch"` // ISO-8601
	PosECEF        [3]float64 `json:"pos" msgpack:"pos"`     // [x, y, z] km
	VelECEF        [3]float64 `json:"vel" msgpack:"vel"`     // [vx, vy, vz] km/s
	CovarianceDiag [3]float64 `json:"cov" msgpack:"cov"`     // [p11, p22, p33]
	SlopeTrend     SlopeTrend `json:"trend" msgpack:"trend"` // STABLE|MANEUVERING|DECAYING

	// Characterization (from MPDAF/SLiPy)
	MaterialID     string  `json:"mat" msgpack:"mat"`           // "GaAs_SOLAR_PANEL"
	MaterialConf   float64 `json:"mat_conf" msgpack:"mat_conf"` // 0.98
	AttitudeState  string  `json:"att" msgpack:"att"`           // "STABILIZED_NADIR"
	RotationRateHz float64 `json:"rot_hz" msgpack:"rot_hz"`     // 0.0
	ThermalSig     string  `json:"thermal" msgpack:"thermal"`   // "ACTIVE_BUS"

	// Tactical (Siliguri-specific)
	TargetVolume  string    `json:"volume" msgpack:"volume"` // "SILIGURI_LEO_S2_L13"
	AccessStart   time.Time `json:"aos" msgpack:"aos"`       // Acquisition of Signal
	AccessEnd     time.Time `json:"los" msgpack:"los"`       // Loss of Signal
	MaxElevation  float64   `json:"max_el" msgpack:"max_el"` // 62.4 degrees
	ISRCapability string    `json:"isr" msgpack:"isr"`       // "OPTICAL_HIGH_RES"
	ActionCode    string    `json:"act" msgpack:"act"`       // "EMCON_B_INITIATE"

	// Structural Characterization (MGE-derived)
	ArrayToBusRatio    float64 `json:"mge_ratio" msgpack:"mge_ratio"`           // MGE flux ratio
	MGEComponents      int     `json:"mge_components" msgpack:"mge_components"` // Gaussian components
	ArchitectureType   string  `json:"mge_arch" msgpack:"mge_arch"`             // COMPACT, STANDARD, etc.
	TemplateMatch      string  `json:"mge_template" msgpack:"mge_template"`     // Matched platform template
	TemplateConfidence float64 `json:"mge_conf" msgpack:"mge_conf"`             // Template match confidence
}

// Priority levels for TargetBrief
type Priority string

const (
	PriorityCritical Priority = "CRITICAL"
	PriorityHigh     Priority = "HIGH"
	PriorityRoutine  Priority = "ROUTINE"
)

// SlopeTrend indicates orbital behavior
type SlopeTrend string

const (
	SlopeStable      SlopeTrend = "STABLE"
	SlopeManeuvering SlopeTrend = "MANEUVERING"
	SlopeDecaying    SlopeTrend = "DECAYING"
)

// SerializeToMessagePack converts TargetBrief to MessagePack bytes
func (tb *TargetBrief) SerializeToMessagePack() ([]byte, error) {
	// TODO: Re-enable when msgpack dependency available
	// return msgpack.Marshal(tb)
	return nil, nil
}

// DeserializeFromMessagePack creates TargetBrief from MessagePack bytes
func DeserializeFromMessagePack(data []byte) (*TargetBrief, error) {
	// TODO: Re-enable when msgpack dependency available
	// var tb TargetBrief
	// return msgpack.Unmarshal(data, &tb)
	return nil, nil
}

// ToJSON serializes to human-readable JSON (for logging/debug)
func (tb *TargetBrief) ToJSON() ([]byte, error) {
	return json.MarshalIndent(tb, "", "  ")
}

// EstimateSize returns estimated serialized size in bytes
func (tb *TargetBrief) EstimateSize() int {
	// Header: ~50 bytes
	// Identity: ~100 bytes
	// Orbital: ~150 bytes
	// Characterization: ~80 bytes
	// Tactical: ~70 bytes
	return 450
}

// IsHighValue returns true if brief meets high-value criteria
func (tb *TargetBrief) IsHighValue() bool {
	return tb.Priority == PriorityCritical
}

// MissileBrief extends TargetBrief with missile/hypersonic specific fields
// This is the "Red-Brief" for immediate missile threat notification
type MissileBrief struct {
	// Base TargetBrief fields (embedded)
	TargetBrief

	// Missile-specific header
	BriefType string `json:"brief_type" msgpack:"brief_type"` // "LAUNCH_EVENT"

	// Launch kinematics
	LaunchPoint [2]float64 `json:"launch_point" msgpack:"launch_point"` // [lat, lon] degrees
	Azimuth     float64    `json:"azimuth" msgpack:"azimuth"`           // degrees
	CurrentMach float64    `json:"mach" msgpack:"mach"`                 // Mach number
	ThreatType  string     `json:"threat_type" msgpack:"threat_type"`   // "HGV", "BALLISTIC", "CRUISE"

	// Prediction
	ImpactPoint  [2]float64 `json:"impact_point" msgpack:"impact_point"` // [lat, lon] degrees
	TimeToImpact float64    `json:"tti" msgpack:"tti"`                   // seconds
	Confidence   float64    `json:"conf" msgpack:"conf"`                 // 0-1

	// Detection phase
	Phase       string  `json:"phase" msgpack:"phase"` // "BOOST", "GLIDE", "TERMINAL"
	Temperature float64 `json:"temp" msgpack:"temp"`   // Kelvin

	// Radar cueing
	RadarCue RadarCue `json:"radar_cue" msgpack:"radar_cue"`
}

// RadarCue provides specific coordinates for ground radar acquisition
type RadarCue struct {
	RadarID       string    `json:"radar_id" msgpack:"radar_id"`     // "ARUDHRA", "ASHWINI"
	CueLat        float64   `json:"cue_lat" msgpack:"cue_lat"`       // degrees
	CueLon        float64   `json:"cue_lon" msgpack:"cue_lon"`       // degrees
	CueAlt        float64   `json:"cue_alt" msgpack:"cue_alt"`       // km
	CueTime       time.Time `json:"cue_time" msgpack:"cue_time"`     // When to start staring
	StareDuration float64   `json:"stare_dur" msgpack:"stare_dur"`   // seconds
	BeamWidth     float64   `json:"beam_width" msgpack:"beam_width"` // degrees
}

// Validate checks if brief has required fields
func (tb *TargetBrief) Validate() error {
	if tb.MsgID == "" {
		return fmt.Errorf("missing msg_id")
	}
	if tb.NoradID == 0 {
		return fmt.Errorf("missing norad_id")
	}
	if tb.Timestamp.IsZero() {
		return fmt.Errorf("missing timestamp")
	}
	return nil
}

// NewTargetBrief creates a brief from operational data
func NewTargetBrief(
	noradID int,
	pos, vel [3]float64,
	materialID string,
	materialConf float64,
	attitudeState string,
) *TargetBrief {
	return &TargetBrief{
		MsgID:         generateUUID(),
		Timestamp:     time.Now().UTC(),
		NoradID:       noradID,
		PosECEF:       pos,
		VelECEF:       vel,
		MaterialID:    materialID,
		MaterialConf:  materialConf,
		AttitudeState: attitudeState,
	}
}

// generateUUID creates UUID-V4 (placeholder - use proper UUID lib in production)
func generateUUID() string {
	// In production: github.com/google/uuid
	return "uuid-" + time.Now().Format("20060102150405")
}

// Action codes for tactical response
const (
	ActionEMCONB    = "EMCON_B_INITIATE"   // Emission control
	ActionTRACK     = "TRACK_MAINTAIN"     // Continue tracking
	ActionALERT     = "FORCE_PROTECTION"   // Alert units
	ActionINTERDICT = "INTERDICTION_READY" // Prepare countermeasures
)

// Classification types
const (
	ClassAdversaryISR = "ADVERSARY_ISR"
	ClassFriendly     = "FRIENDLY"
	ClassNeutral      = "NEUTRAL"
	ClassUnknown      = "UNKNOWN"
	ClassDebris       = "DEBRIS"
)
