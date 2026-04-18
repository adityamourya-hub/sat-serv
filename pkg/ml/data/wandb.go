package data

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
)

// WandBClient wraps Weights & Biases logging for dataset versioning.
type WandBClient struct {
	projectName string
	entityName  string
	runName     string
	runID       string
	enabled     bool
}

// NewWandBClient creates a new WandB client for logging.
// If W&B is not available, it gracefully disables logging.
func NewWandBClient(projectName, entityName, runName string) *WandBClient {
	// Check if wandb CLI is available
	_, err := exec.LookPath("wandb")
	enabled := err == nil

	return &WandBClient{
		projectName: projectName,
		entityName:  entityName,
		runName:     runName,
		enabled:     enabled,
		runID:       fmt.Sprintf("%s-%d", runName, os.Getpid()),
	}
}

// LogDatasetVersion logs a dataset version to W&B.
func (w *WandBClient) LogDatasetVersion(version DataVersion, metrics map[string]interface{}) error {
	if !w.enabled {
		return fmt.Errorf("wandb not available; logging disabled")
	}

	// Prepare artifact metadata
	artifactMeta := map[string]interface{}{
		"version":           version.ID,
		"name":              version.Name,
		"purpose":           version.Purpose,
		"model":             version.Model,
		"sample_count":      version.SampleCount,
		"tle_count":         version.TLECount,
		"propagation_steps": version.PropagationSteps,
		"noise_std":         version.NoiseSTD,
		"covariance_scale":  version.CovarianceScale,
		"description":       version.Description,
		"checksum":          version.Checksum,
		"splits":            version.Splits,
	}

	// Log metrics
	if metrics != nil {
		for k, v := range metrics {
			artifactMeta[fmt.Sprintf("metric_%s", k)] = v
		}
	}

	// Create JSON config file for wandb (in /tmp)
	tmpFile := fmt.Sprintf("/tmp/wandb_dataset_%s.json", version.ID)
	data, err := json.MarshalIndent(artifactMeta, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		return err
	}

	// Log to wandb via CLI
	// Note: This is a simplified approach; real implementation would use wandb-go SDK
	cmd := exec.Command("wandb", "artifact", "log-artifact",
		"--type", "dataset",
		"--property", fmt.Sprintf("version=%s", version.ID),
		"--property", fmt.Sprintf("purpose=%s", version.Purpose),
		tmpFile)

	if err := cmd.Run(); err != nil {
		// Not fatal; W&B might not be configured
		return fmt.Errorf("wandb logging failed (non-fatal): %v", err)
	}

	return nil
}

// LogMetrics logs dataset metrics directly.
func (w *WandBClient) LogMetrics(versionID string, metrics map[string]interface{}) error {
	if !w.enabled {
		return nil // silently skip if wandb not available
	}

	// Create a simple JSON log file
	logData := map[string]interface{}{
		"version_id": versionID,
		"metrics":    metrics,
	}

	data, err := json.MarshalIndent(logData, "", "  ")
	if err != nil {
		return err
	}

	tmpFile := fmt.Sprintf("/tmp/wandb_metrics_%s.json", versionID)
	return os.WriteFile(tmpFile, data, 0644)
}

// SummarizeDataset creates a summary suitable for logging.
func SummarizeDataset(generator *SyntheticDataGenerator, samples []Sample) map[string]interface{} {
	metrics := computeMetrics(samples)

	summary := map[string]interface{}{
		"dataset_info": map[string]interface{}{
			"version":              generator.Version().ID,
			"purpose":              generator.config.Purpose,
			"model":                generator.config.Model,
			"total_samples":        len(samples),
			"tle_count":            len(generator.tles),
			"propagation_step_min": generator.config.SampleIntervalMin,
			"propagation_horizon":  generator.config.PropagationSteps,
		},
		"noise_config": map[string]interface{}{
			"position_noise_km":        generator.config.NoiseSTDKm,
			"velocity_noise_km_s":      generator.config.NoiseSTDVel,
			"covariance_scale":         generator.config.CovarianceScale,
			"clock_bias_scale_seconds": generator.config.ClockBiasScale,
			"tracking_error_rate":      generator.config.TrackingErrorRate,
		},
		"metrics": metrics,
	}

	return summary
}
