// pkg/config/config.go - Viper Configuration System for ADITI 4.0
//
// Air-Gap Config Mandate:
// In an air-gapped tactical environment (like a Hanle node), operators should not
// be passing 15 different --flags for tolerances, ZMQ endpoints, and SQLite paths
// every time they run a command. Viper allows locking down a config.yaml file on
// the ruggedized server, merging with environment variables, and binding directly
// to Cobra flags. This guarantees Configuration Determinism.
//
// Configuration Hierarchy (priority order):
// 1. Command-line flags (highest priority)
// 2. Environment variables (ADITI_* prefix)
// 3. Config file (config.yaml)
// 4. Built-in defaults (lowest priority)

package config

import (
	"fmt"
	"strings"

	"github.com/spf13/viper"
)

// SensorConfig represents runtime configuration for a sensor
// Defined here to avoid import cycles with pkg/boundary
type SensorConfig struct {
	Name        string  `mapstructure:"name"`          // Human-readable name
	Type        string  `mapstructure:"type"`          // optical|radar|rf
	Endpoint    string  `mapstructure:"endpoint"`      // ZMQ endpoint for data stream
	Enabled     bool    `mapstructure:"enabled"`       // Whether sensor is active
	Priority    int     `mapstructure:"priority"`      // Fusion priority (higher = more trusted)
	MaxRangeKm  float64 `mapstructure:"max_range_km"`  // Maximum detection range
	MinRangeKm  float64 `mapstructure:"min_range_km"`  // Minimum detection range
	FieldOfView float64 `mapstructure:"field_of_view"` // Field of view [rad]
}

// Config represents the complete ADITI 4.0 configuration
// Viper unmarshals YAML/JSON directly into this struct
type Config struct {
	// Database configuration for sovereign catalog
	Database struct {
		Path    string `mapstructure:"path"`     // SQLite/PostgreSQL path
		WALMode bool   `mapstructure:"wal_mode"` // Enable WAL for concurrent access
	} `mapstructure:"database"`

	// Boundary configuration for external interfaces
	Boundary struct {
		TelescopeZMQ string `mapstructure:"telescope_zmq"` // ZMQ endpoint for telescope control
		IngestZMQ    string `mapstructure:"ingest_zmq"`    // ZMQ endpoint for track ingestion
	} `mapstructure:"boundary"`

	// Physics configuration for integrators
	Physics struct {
		DefaultRegime string  `mapstructure:"default_regime"` // leo|meo|geo|hgv
		Tolerance     float64 `mapstructure:"tolerance"`      // RKF45 integration tolerance
		MinStep       float64 `mapstructure:"min_step"`       // Minimum step size (s)
		MaxStep       float64 `mapstructure:"max_step"`       // Maximum step size (s)
		Propagator    string  `mapstructure:"propagator"`     // Default propagator name
	} `mapstructure:"physics"`

	// PAX thermal governor configuration
	PAX struct {
		ThermalLimitWatts int `mapstructure:"thermal_limit_watts"` // M2 Air TDP limit
		DefaultTier       int `mapstructure:"default_tier"`        // 1=Go, 2=ONNX, 3=DPHNO, 4=JAX
	} `mapstructure:"pax"`

	// Discovery filter configuration
	Discovery struct {
		EnableSINDy bool     `mapstructure:"enable_sindy"` // Enable latent force discovery
		Threshold   float64  `mapstructure:"threshold"`    // P(discovery) threshold
		ModeList    []string `mapstructure:"mode_list"`    // IMM mode list
	} `mapstructure:"discovery"`

	// Slew control configuration
	Slew struct {
		DefaultTelescope string  `mapstructure:"default_telescope"` // Default telescope ID
		SettleTimeMs     float64 `mapstructure:"settle_time_ms"`    // Target settle time
		MaxRate          float64 `mapstructure:"max_rate"`          // Maximum slew rate (rad/s)
		LeadTimeMs       float64 `mapstructure:"lead_time_ms"`      // Prediction horizon
	} `mapstructure:"slew"`

	// Sensors configuration for sensor registry
	Sensors []SensorConfig `mapstructure:"sensors"`

	// Node configuration for distributed deployment
	Node struct {
		ID        string   `mapstructure:"id"`         // Node identifier
		Role      string   `mapstructure:"role"`       // primary|replica|edge
		Bind      string   `mapstructure:"bind"`       // gRPC bind address
		Peers     []string `mapstructure:"peers"`      // Peer node addresses
		TimeoutMs int      `mapstructure:"timeout_ms"` // gRPC timeout
	} `mapstructure:"node"`
}

// Cfg is the global configuration instance
// Initialized by LoadConfig()
var Cfg Config

// LoadConfig loads configuration from file, environment, and defaults
// cfgFile: explicit config file path (optional, overrides search paths)
func LoadConfig(cfgFile string) error {
	// Set up Viper
	v := viper.New()

	// Set default values (fallback if no config file)
	setDefaults(v)

	// Configure config file search
	if cfgFile != "" {
		v.SetConfigFile(cfgFile)
	} else {
		// Search in multiple locations (air-gapped compliance)
		v.AddConfigPath("/etc/aditi/")
		v.AddConfigPath("$HOME/.aditi/")
		v.AddConfigPath(".")
		v.SetConfigName("config")
		v.SetConfigType("yaml")
	}

	// Environment variable overrides
	// Example: ADITI_PHYSICS_TOLERANCE=1e-12
	v.SetEnvPrefix("ADITI")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Read config file (optional - if not found, use defaults)
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return fmt.Errorf("read config: %w", err)
		}
		// Config file not found is OK - we use defaults
	}

	// Unmarshal into Config struct
	if err := v.Unmarshal(&Cfg); err != nil {
		return fmt.Errorf("unmarshal config: %w", err)
	}

	return nil
}

// setDefaults defines built-in default values
// These are used if no config file or environment variable is set
func setDefaults(v *viper.Viper) {
	// Database defaults
	v.SetDefault("database.path", "./sda.db")
	v.SetDefault("database.wal_mode", true)

	// Boundary defaults
	v.SetDefault("boundary.telescope_zmq", "tcp://localhost:5555")
	v.SetDefault("boundary.ingest_zmq", "tcp://localhost:5556")

	// Physics defaults
	v.SetDefault("physics.default_regime", "leo")
	v.SetDefault("physics.tolerance", 1e-11)
	v.SetDefault("physics.min_step", 0.001)
	v.SetDefault("physics.max_step", 60.0)

	// PAX defaults
	v.SetDefault("pax.thermal_limit_watts", 8) // M2 Air TDP
	v.SetDefault("pax.default_tier", 2)        // Pruned ONNX

	// Discovery defaults
	v.SetDefault("discovery.enable_sindy", true)
	v.SetDefault("discovery.threshold", 0.7)
	v.SetDefault("discovery.mode_list", []string{"ballistic", "entry", "glide", "maneuver"})

	// Slew defaults
	v.SetDefault("slew.default_telescope", "T1")
	v.SetDefault("slew.settle_time_ms", 50.0)
	v.SetDefault("slew.max_rate", 1.0)
	v.SetDefault("slew.lead_time_ms", 500.0)

	// Node defaults
	v.SetDefault("node.id", "sovereign-node-001")
	v.SetDefault("node.role", "primary")
	v.SetDefault("node.bind", "0.0.0.0:50051")
	v.SetDefault("node.peers", []string{})
	v.SetDefault("node.timeout_ms", 100)
}

// GetPhysicsTolerance returns the physics tolerance with regime-specific override
func GetPhysicsTolerance(regime string) float64 {
	// In production, would have regime-specific tolerance profiles
	// For now, use the global tolerance
	return Cfg.Physics.Tolerance
}

// GetDatabasePath returns the database path
func GetDatabasePath() string {
	return Cfg.Database.Path
}

// GetTelescopeZMQ returns the telescope ZMQ endpoint
func GetTelescopeZMQ() string {
	return Cfg.Boundary.TelescopeZMQ
}

// GetIngestZMQ returns the ingest ZMQ endpoint
func GetIngestZMQ() string {
	return Cfg.Boundary.IngestZMQ
}

// GetThermalLimit returns the thermal limit in watts
func GetThermalLimit() int {
	return Cfg.PAX.ThermalLimitWatts
}

// IsWALModeEnabled returns whether WAL mode is enabled
func IsWALModeEnabled() bool {
	return Cfg.Database.WALMode
}
