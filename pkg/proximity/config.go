package proximity

import (
	"fmt"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// Config holds proximity analysis configuration parameters
type Config struct {
	// Screening parameters
	DefaultThreatRadiusKm   float64 `yaml:"default_threat_radius_km"`
	DefaultFOVHalfAngleDeg  float64 `yaml:"default_fov_half_angle_deg"`
	TemporalToleranceSec    float64 `yaml:"temporal_tolerance_sec"`

	// Reachability parameters
	CWHorizonThresholdSec float64 `yaml:"cw_horizon_threshold_sec"`
	STMHorizonMaxSec      float64 `yaml:"stm_horizon_max_sec"`

	// Coverage parameters
	CoverageIndexResolutionLat int `yaml:"coverage_index_resolution_lat"`
	CoverageIndexResolutionLon int `yaml:"coverage_index_resolution_lon"`

	// Threat evaluation thresholds
	TightProximityKm     float64 `yaml:"tight_proximity_km"`
	CriticalInterceptKm float64 `yaml:"critical_intercept_km"`
}

// DefaultConfig returns the default configuration values
func DefaultConfig() Config {
	return Config{
		DefaultThreatRadiusKm:   500.0,
		DefaultFOVHalfAngleDeg:  30.0,
		TemporalToleranceSec:    0.5,
		CWHorizonThresholdSec:   1800.0,
		STMHorizonMaxSec:        7200.0,
		CoverageIndexResolutionLat: 180,
		CoverageIndexResolutionLon: 360,
		TightProximityKm:       100.0,
		CriticalInterceptKm:    20.0,
	}
}

// LoadConfig loads proximity configuration from the specified YAML file
// Falls back to defaults if the file doesn't exist or is invalid
func LoadConfig(configPath string) (Config, error) {
	cfg := DefaultConfig()

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil // Use defaults if file doesn't exist
		}
		return cfg, fmt.Errorf("failed to read config file: %w", err)
	}

	var fullConfig struct {
		Proximity Config `yaml:"proximity"`
	}

	if err := yaml.Unmarshal(data, &fullConfig); err != nil {
		return cfg, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Override defaults with loaded values
	if fullConfig.Proximity.DefaultThreatRadiusKm > 0 {
		cfg.DefaultThreatRadiusKm = fullConfig.Proximity.DefaultThreatRadiusKm
	}
	if fullConfig.Proximity.DefaultFOVHalfAngleDeg > 0 {
		cfg.DefaultFOVHalfAngleDeg = fullConfig.Proximity.DefaultFOVHalfAngleDeg
	}
	if fullConfig.Proximity.TemporalToleranceSec > 0 {
		cfg.TemporalToleranceSec = fullConfig.Proximity.TemporalToleranceSec
	}
	if fullConfig.Proximity.CWHorizonThresholdSec > 0 {
		cfg.CWHorizonThresholdSec = fullConfig.Proximity.CWHorizonThresholdSec
	}
	if fullConfig.Proximity.STMHorizonMaxSec > 0 {
		cfg.STMHorizonMaxSec = fullConfig.Proximity.STMHorizonMaxSec
	}
	if fullConfig.Proximity.CoverageIndexResolutionLat > 0 {
		cfg.CoverageIndexResolutionLat = fullConfig.Proximity.CoverageIndexResolutionLat
	}
	if fullConfig.Proximity.CoverageIndexResolutionLon > 0 {
		cfg.CoverageIndexResolutionLon = fullConfig.Proximity.CoverageIndexResolutionLon
	}
	if fullConfig.Proximity.TightProximityKm > 0 {
		cfg.TightProximityKm = fullConfig.Proximity.TightProximityKm
	}
	if fullConfig.Proximity.CriticalInterceptKm > 0 {
		cfg.CriticalInterceptKm = fullConfig.Proximity.CriticalInterceptKm
	}

	// Validate loaded configuration
	if err := cfg.Validate(); err != nil {
		return cfg, fmt.Errorf("config validation failed: %w", err)
	}

	return cfg, nil
}

// Validate checks that configuration parameters are within reasonable ranges
func (c *Config) Validate() error {
	if c.DefaultThreatRadiusKm <= 0 || c.DefaultThreatRadiusKm > 100000 {
		return fmt.Errorf("default_threat_radius_km must be between 0 and 100000 km, got %f", c.DefaultThreatRadiusKm)
	}
	if c.DefaultFOVHalfAngleDeg <= 0 || c.DefaultFOVHalfAngleDeg > 90 {
		return fmt.Errorf("default_fov_half_angle_deg must be between 0 and 90 degrees, got %f", c.DefaultFOVHalfAngleDeg)
	}
	if c.TemporalToleranceSec <= 0 || c.TemporalToleranceSec > 3600 {
		return fmt.Errorf("temporal_tolerance_sec must be between 0 and 3600 seconds, got %f", c.TemporalToleranceSec)
	}
	if c.CWHorizonThresholdSec <= 0 || c.CWHorizonThresholdSec > c.STMHorizonMaxSec {
		return fmt.Errorf("cw_horizon_threshold_sec must be positive and less than stm_horizon_max_sec")
	}
	if c.STMHorizonMaxSec <= 0 || c.STMHorizonMaxSec > 86400 {
		return fmt.Errorf("stm_horizon_max_sec must be between 0 and 86400 seconds, got %f", c.STMHorizonMaxSec)
	}
	if c.CoverageIndexResolutionLat <= 0 || c.CoverageIndexResolutionLat > 3600 {
		return fmt.Errorf("coverage_index_resolution_lat must be between 0 and 3600, got %d", c.CoverageIndexResolutionLat)
	}
	if c.CoverageIndexResolutionLon <= 0 || c.CoverageIndexResolutionLon > 7200 {
		return fmt.Errorf("coverage_index_resolution_lon must be between 0 and 7200, got %d", c.CoverageIndexResolutionLon)
	}
	if c.TightProximityKm <= 0 || c.TightProximityKm > c.DefaultThreatRadiusKm {
		return fmt.Errorf("tight_proximity_km must be positive and less than default_threat_radius_km")
	}
	if c.CriticalInterceptKm <= 0 || c.CriticalInterceptKm > c.TightProximityKm {
		return fmt.Errorf("critical_intercept_km must be positive and less than tight_proximity_km")
	}
	return nil
}

var (
	globalConfig Config
	configOnce   sync.Once
	configLoadError error
)

// GetConfig returns the global configuration singleton
// Loads from configs/config.yaml on first call
func GetConfig() Config {
	configOnce.Do(func() {
		globalConfig, configLoadError = LoadConfig("configs/config.yaml")
		if configLoadError != nil {
			// Log error but continue with defaults
			fmt.Printf("[WARN] Failed to load proximity config, using defaults: %v\n", configLoadError)
		}
	})
	return globalConfig
}

// ReloadConfig forces a reload of the configuration from the config file
// Useful for testing or runtime configuration updates
func ReloadConfig(configPath string) error {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		return err
	}
	globalConfig = cfg
	configLoadError = nil
	return nil
}
