// Package threat provides configurable thresholds and parameters for threat detection
package threat

import (
	"encoding/json"
	"os"
)

// ThreatConfig holds all configurable thresholds and parameters
type ThreatConfig struct {
	// Detection thresholds
	MachThreshold          float64 `json:"mach_threshold"`
	VerticalVelocityKmS    float64 `json:"vertical_velocity_kms"`
	AltitudeBoostThreshold float64 `json:"altitude_boost_threshold"`
	PlumeTempMin           float64 `json:"plume_temp_min"`
	PlasmaTempMin          float64 `json:"plasma_temp_min"`

	// S2 cell levels
	LaunchCorridorLevel   int `json:"launch_corridor_level"`
	RegionalTripwireLevel int `json:"regional_tripwire_level"`

	// IMM filter parameters
	IMMModelSwitchProb float64 `json:"imm_model_switch_prob"`
	IMMUpdateRateHz    float64 `json:"imm_update_rate_hz"`

	// Fractional Laplacian parameters
	FractionalOrder   float64 `json:"fractional_order"`
	WakeMemoryDepth   int     `json:"wake_memory_depth"`
	WakeDecayFactor   float64 `json:"wake_decay_factor"`
	HGVCorridorMinAlt float64 `json:"hgv_corridor_min_alt"`
	HGVCorridorMaxAlt float64 `json:"hgv_corridor_max_alt"`

	// ASAT intercept thresholds
	InterceptThresholdM   float64 `json:"intercept_threshold_m"`
	StrikeWindowPhaseDeg  float64 `json:"strike_window_phase_deg"`
	SleeperVelocityMPS    float64 `json:"sleeper_velocity_mps"`
	ASATMachThreshold     float64 `json:"asat_mach_threshold"`
	DAVerticalVelocityKmS float64 `json:"da_vertical_velocity_kms"`

	// FSM thresholds
	FSMDwellTimerMinutes  int     `json:"fsm_dwell_timer_minutes"`
	FSMMaxHistory         int     `json:"fsm_max_history"`
	FSMDistanceKm         float64 `json:"fsm_distance_km"`
	FSMStalkingDistanceKm float64 `json:"fsm_stalking_distance_km"`
	FSMTerminalDistanceKm float64 `json:"fsm_terminal_distance_km"`

	// Classifier thresholds
	InspectorDistanceKm    float64 `json:"inspector_distance_km"`
	GrappleClosureRateKms  float64 `json:"grapple_closure_rate_kms"`
	JammerThermalThreshold float64 `json:"jammer_thermal_threshold"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *ThreatConfig {
	return &ThreatConfig{
		MachThreshold:          3.0,
		VerticalVelocityKmS:    1.0,
		AltitudeBoostThreshold: 100.0,
		PlumeTempMin:           3000.0,
		PlasmaTempMin:          2000.0,
		LaunchCorridorLevel:    10,
		RegionalTripwireLevel:  12,
		IMMModelSwitchProb:     0.1,
		IMMUpdateRateHz:        100.0,
		FractionalOrder:        1.7,
		WakeMemoryDepth:        10,
		WakeDecayFactor:        0.95,
		HGVCorridorMinAlt:      30.0,
		HGVCorridorMaxAlt:      80.0,
		InterceptThresholdM:    500.0,
		StrikeWindowPhaseDeg:   10.0,
		SleeperVelocityMPS:     0.1,
		ASATMachThreshold:      7.0,
		DAVerticalVelocityKmS:  7.0,
		FSMDwellTimerMinutes:   3,
		FSMMaxHistory:          10,
		FSMDistanceKm:          100.0,
		FSMStalkingDistanceKm:  20.0,
		FSMTerminalDistanceKm:  2.0,
		InspectorDistanceKm:    1.0,
		GrappleClosureRateKms:  0.1,
		JammerThermalThreshold: 350.0,
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(path string) (*ThreatConfig, error) {
	config := DefaultConfig()

	data, err := os.ReadFile(path)
	if err != nil {
		return config, nil // Return defaults if file doesn't exist
	}

	if err := json.Unmarshal(data, config); err != nil {
		return nil, err
	}

	return config, nil
}

// SaveConfig saves configuration to a JSON file
func SaveConfig(config *ThreatConfig, path string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// Global config instance (can be overridden by loading from file)
var globalConfig = DefaultConfig()

// GetConfig returns the current global configuration
func GetConfig() *ThreatConfig {
	return globalConfig
}

// SetConfig sets the global configuration
func SetConfig(config *ThreatConfig) {
	globalConfig = config
}
