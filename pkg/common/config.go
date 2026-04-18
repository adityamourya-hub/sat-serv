package common

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

/**
 * Config represents the centralized operational parameter store for SSIS.
 * It enables air-gapped tuning of the engine, physics, and sensing substrates 
 * without requiring binary recompilation.
 * 
 * <plan_complexity>Medium</plan_complexity><total_lines_estimate>120</total_lines_estimate>
 */
type Config struct {
	Engine struct {
		BatchSize     int     `yaml:"batch_size"`
		WorkerCount   int     `yaml:"worker_count"`
		Provider      string  `yaml:"provider"`
		MaxQueueDepth int     `yaml:"max_queue_depth"`
	} `yaml:"engine"`

	Physics struct {
		DKLThreshold    float64 `yaml:"d_kl_threshold"`
		PropagationStep float64 `yaml:"propagation_step"`
		SigmaPosition   float64 `yaml:"sigma_position"`
	} `yaml:"physics"`

	Sensing struct {
		PDetThreshold float64 `yaml:"p_det_threshold"`
		NumPulses     int     `yaml:"num_pulses"`
		Fs            float64 `yaml:"fs"`
	} `yaml:"sensing"`

	Security struct {
		MTLSEnabled   bool   `yaml:"mtls_enabled"`
		JWTExpiration string `yaml:"jwt_expiration"`
		AuditLevel    string `yaml:"audit_level"`
	} `yaml:"security"`
}

/**
 * LoadConfig ingests the YAML parameter store from the specified path.
 * 
 * @param path: Path to the config.yaml file.
 * @return: Pointer to initialized Config struct.
 */
func LoadConfig(path string) (*Config, error) {
	// Entry Point: Config Loading
	fmt.Printf("DEBUG: Loading SSIS Configuration from %s at %v\n", path, time.Now())
	
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	fmt.Printf("DEBUG: Configuration loaded (Provider: %s, D_KL: %.2f)\n", 
		cfg.Engine.Provider, cfg.Physics.DKLThreshold)
	
	// Exit Point: Config Ready
	return &cfg, nil
}

/**
 * GetJWTDuration returns the parsed time.Duration for token expiration.
 */
func (c *Config) GetJWTDuration() time.Duration {
	d, err := time.ParseDuration(c.Security.JWTExpiration)
	if err != nil {
		return 24 * time.Hour // Default fallback
	}
	return d
}
