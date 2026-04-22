// Package hud provides Neural HUD integration for Threat Environment visualization
// Displays tactical pressure, threat levels, and protection alerts in real-time
package hud

import (
	"encoding/json"
	"fmt"
	"time"
)

// ThreatLevel represents the current threat classification
type ThreatLevel string

const (
	ThreatLow    ThreatLevel = "LOW"
	ThreatMedium ThreatLevel = "MEDIUM"
	ThreatHigh   ThreatLevel = "HIGH"
)

// TacticalHUDState represents the current state for the Neural HUD
type TacticalHUDState struct {
	Timestamp        time.Time      `json:"timestamp"`
	TacticalPressure float64        `json:"tactical_pressure"` // Ψ
	ThreatLevel      ThreatLevel    `json:"threat_level"`
	UCTCount         int            `json:"uct_count"`
	HVACount         int            `json:"hva_count"`
	ProtectionAlerts []AlertDisplay `json:"protection_alerts"`
	CurrentAction    string         `json:"current_action"`
	CuriosityMode    string         `json:"curiosity_mode"` // EXPLORATION or EXPLOITATION
	ExplorationScore float64        `json:"exploration_score"`
}

// AlertDisplay represents a protection alert for HUD display
type AlertDisplay struct {
	HVAName    string    `json:"hva_name"`
	UCTID      int       `json:"uct_id"`
	Distance   float64   `json:"distance_km"`
	AlertLevel string    `json:"alert_level"`
	Timestamp  time.Time `json:"timestamp"`
}

// HUDColor represents RGB color values for HUD elements
type HUDColor struct {
	R uint8 `json:"r"`
	G uint8 `json:"g"`
	B uint8 `json:"b"`
}

// ThemeColors defines color schemes for different threat levels
type ThemeColors struct {
	Background HUDColor `json:"background"`
	Text       HUDColor `json:"text"`
	Accent     HUDColor `json:"accent"`
}

// GetThemeColors returns color scheme based on threat level
func GetThemeColors(level ThreatLevel) ThemeColors {
	switch level {
	case ThreatLow:
		return ThemeColors{
			Background: HUDColor{R: 10, G: 30, B: 50},    // Dark blue
			Text:       HUDColor{R: 100, G: 200, B: 255}, // Light blue
			Accent:     HUDColor{R: 0, G: 255, B: 128},   // Green
		}
	case ThreatMedium:
		return ThemeColors{
			Background: HUDColor{R: 50, G: 30, B: 10},    // Dark amber
			Text:       HUDColor{R: 255, G: 200, B: 100}, // Light amber
			Accent:     HUDColor{R: 255, G: 200, B: 0},   // Yellow
		}
	case ThreatHigh:
		return ThemeColors{
			Background: HUDColor{R: 50, G: 10, B: 10},    // Dark red
			Text:       HUDColor{R: 255, G: 100, B: 100}, // Light red
			Accent:     HUDColor{R: 255, G: 0, B: 0},     // Bright red
		}
	default:
		return ThemeColors{
			Background: HUDColor{R: 20, G: 20, B: 30},
			Text:       HUDColor{R: 200, G: 200, B: 220},
			Accent:     HUDColor{R: 100, G: 100, B: 255},
		}
	}
}

// TacticalHUD manages the Neural HUD state and rendering
type TacticalHUD struct {
	currentState TacticalHUDState
	lastUpdate   time.Time
	updateRate   time.Duration
}

// NewTacticalHUD creates a new Tactical HUD instance
func NewTacticalHUD() *TacticalHUD {
	return &TacticalHUD{
		currentState: TacticalHUDState{
			Timestamp:        time.Now(),
			TacticalPressure: 0.0,
			ThreatLevel:      ThreatLow,
			UCTCount:         0,
			HVACount:         0,
			ProtectionAlerts: make([]AlertDisplay, 0),
			CurrentAction:    "PASSIVE_TRACK",
			CuriosityMode:    "EXPLOITATION",
			ExplorationScore: 0.0,
		},
		lastUpdate: time.Now(),
		updateRate: 100 * time.Millisecond, // 10Hz update rate
	}
}

// UpdateState updates the HUD state with new tactical information
func (hud *TacticalHUD) UpdateState(
	tacticalPressure float64,
	uctCount int,
	hvaCount int,
	alerts []AlertDisplay,
	currentAction string,
	curiosityMode string,
	explorationScore float64,
) {
	hud.currentState.Timestamp = time.Now()
	hud.currentState.TacticalPressure = tacticalPressure
	hud.currentState.UCTCount = uctCount
	hud.currentState.HVACount = hvaCount
	hud.currentState.ProtectionAlerts = alerts
	hud.currentState.CurrentAction = currentAction
	hud.currentState.CuriosityMode = curiosityMode
	hud.currentState.ExplorationScore = explorationScore

	// Classify threat level based on pressure
	hud.currentState.ThreatLevel = hud.classifyThreatLevel(tacticalPressure)

	hud.lastUpdate = time.Now()
}

// classifyThreatLevel converts pressure to threat level
func (hud *TacticalHUD) classifyThreatLevel(pressure float64) ThreatLevel {
	if pressure < 5.0 {
		return ThreatLow
	} else if pressure < 20.0 {
		return ThreatMedium
	}
	return ThreatHigh
}

// ToJSON exports HUD state to JSON for WebSocket transmission
func (hud *TacticalHUD) ToJSON() ([]byte, error) {
	return json.Marshal(hud.currentState)
}

// GetDisplayText returns formatted text for console display
func (hud *TacticalHUD) GetDisplayText() string {
	status := fmt.Sprintf(`
╔════════════════════════════════════════════════════════════╗
║           TACTICAL HUD                                     ║
╠════════════════════════════════════════════════════════════╣
║ Timestamp: %s                                      ║
║ Threat Level: %-12s  Tactical Pressure: %.2f      ║
║ UCT Count: %-4d              HVA Count: %-4d              ║
║ Current Action: %-15s  Curiosity Mode: %-12s ║
║ Exploration Score: %.3f                                    ║
`,
		hud.currentState.Timestamp.Format("15:04:05.000"),
		hud.currentState.ThreatLevel,
		hud.currentState.TacticalPressure,
		hud.currentState.UCTCount,
		hud.currentState.HVACount,
		hud.currentState.CurrentAction,
		hud.currentState.CuriosityMode,
		hud.currentState.ExplorationScore,
	)

	// Add alerts if any
	if len(hud.currentState.ProtectionAlerts) > 0 {
		status += "║ PROTECTION ALERTS:                                            ║\n"
		for _, alert := range hud.currentState.ProtectionAlerts {
			alertLine := fmt.Sprintf("║   [%-8s] %s - UCT %d at %.1f km            ║\n",
				alert.AlertLevel, alert.HVAName, alert.UCTID, alert.Distance)
			status += alertLine
		}
	}

	status += "╚════════════════════════════════════════════════════════════╝\n"

	return status
}

// GetProgressBar returns a visual progress bar for tactical pressure
func (hud *TacticalHUD) GetProgressBar(width int) string {
	maxPressure := 30.0 // Assume 30 is max reasonable pressure
	ratio := hud.currentState.TacticalPressure / maxPressure
	if ratio > 1.0 {
		ratio = 1.0
	}

	filled := int(ratio * float64(width))
	empty := width - filled

	bar := "[" + string(repeat('█', filled)) + string(repeat('░', empty)) + "]"
	return bar
}

// repeat helper for progress bar
func repeat(char rune, count int) string {
	result := make([]rune, count)
	for i := range result {
		result[i] = char
	}
	return string(result)
}

// WebSocketMessage represents a message sent to the frontend
type WebSocketMessage struct {
	Type    string      `json:"type"` // "STATE_UPDATE", "ALERT", "ACTION"
	Payload interface{} `json:"payload"`
}

// CreateStateUpdateMessage creates a WebSocket message for state updates
func CreateStateUpdateMessage(state TacticalHUDState) WebSocketMessage {
	return WebSocketMessage{
		Type:    "STATE_UPDATE",
		Payload: state,
	}
}

// CreateAlertMessage creates a WebSocket message for alerts
func CreateAlertMessage(alert AlertDisplay) WebSocketMessage {
	return WebSocketMessage{
		Type:    "ALERT",
		Payload: alert,
	}
}

// CreateActionMessage creates a WebSocket message for action updates
func CreateActionMessage(action string, explorationScore float64) WebSocketMessage {
	return WebSocketMessage{
		Type: "ACTION",
		Payload: map[string]interface{}{
			"action":            action,
			"exploration_score": explorationScore,
			"timestamp":         time.Now(),
		},
	}
}

// HUDConfig represents HUD configuration
type HUDConfig struct {
	UpdateRate       time.Duration `json:"update_rate"`
	ShowAlerts       bool          `json:"show_alerts"`
	ShowProgressBar  bool          `json:"show_progress_bar"`
	ConsoleOutput    bool          `json:"console_output"`
	WebSocketEnabled bool          `json:"websocket_enabled"`
	WebSocketPort    int           `json:"websocket_port"`
}

// DefaultHUDConfig returns default HUD configuration
func DefaultHUDConfig() HUDConfig {
	return HUDConfig{
		UpdateRate:       100 * time.Millisecond,
		ShowAlerts:       true,
		ShowProgressBar:  true,
		ConsoleOutput:    true,
		WebSocketEnabled: true,
		WebSocketPort:    8080,
	}
}
