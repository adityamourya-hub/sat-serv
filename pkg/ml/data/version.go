package data

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// DataVersion represents metadata about a generated dataset.
type DataVersion struct {
	ID               string    `json:"id"`                // unique version ID (hash-based)
	Name             string    `json:"name"`              // human-readable name
	GeneratedAt      time.Time `json:"generated_at"`      // when dataset was created
	Purpose          string    `json:"purpose"`           // model training purpose (e.g., "maneuver_detector")
	Model            string    `json:"model"`             // ML model identifier
	SampleCount      int64     `json:"sample_count"`      // total samples
	TLECount         int       `json:"tle_count"`         // unique TLEs sampled
	PropagationSteps int       `json:"propagation_steps"` // prediction horizon (minutes)
	NoiseSTD         float64   `json:"noise_std"`         // added Gaussian noise (km)
	CovarianceScale  float64   `json:"covariance_scale"`  // measurement covariance scaling
	Description      string    `json:"description"`       // optional notes
	Checksum         string    `json:"checksum"`          // MD5 of dataset
	Splits           SplitInfo `json:"splits"`            // train/val/test split info
}

// SplitInfo describes train/val/test data distribution.
type SplitInfo struct {
	TrainFraction float64 `json:"train_fraction"`
	ValFraction   float64 `json:"val_fraction"`
	TestFraction  float64 `json:"test_fraction"`
	TrainCount    int64   `json:"train_count"`
	ValCount      int64   `json:"val_count"`
	TestCount     int64   `json:"test_count"`
}

// DataConfig configures synthetic dataset generation.
type DataConfig struct {
	// Generation parameters
	TLECount          int     // number of random TLEs to sample
	SamplesPerTLE     int     // propagation steps per TLE
	PropagationSteps  int     // how many minutes ahead to propagate
	SampleIntervalMin float64 // time between samples (minutes)

	// Noise and uncertainty
	NoiseSTDKm        float64 // position measurement noise (km)
	NoiseSTDVel       float64 // velocity measurement noise (km/s)
	CovarianceScale   float64 // measurement covariance scaling factor
	ClockBiasScale    float64 // timing error scaling (seconds)
	TrackingErrorRate float64 // probability of lost track [0, 1]

	// Split configuration
	TrainFraction float64 // default 0.7
	ValFraction   float64 // default 0.15
	TestFraction  float64 // default 0.15

	// Metadata
	Purpose     string // model purpose
	Model       string // model identifier
	Description string // optional notes
	Seed        int64  // RNG seed for reproducibility
}

// DefaultDataConfig returns sensible defaults for ML datasets.
func DefaultDataConfig(purpose, model string) DataConfig {
	return DataConfig{
		TLECount:          500,
		SamplesPerTLE:     20,
		PropagationSteps:  1440, // 24 hours
		SampleIntervalMin: 5.0,

		NoiseSTDKm:        0.05,  // 50 meters
		NoiseSTDVel:       0.001, // 1 m/s
		CovarianceScale:   1.0,
		ClockBiasScale:    1e-6, // 1 microsecond
		TrackingErrorRate: 0.01,

		TrainFraction: 0.7,
		ValFraction:   0.15,
		TestFraction:  0.15,

		Purpose:     purpose,
		Model:       model,
		Description: "",
		Seed:        42,
	}
}

// Sample represents a single training sample (trajectory point with noise).
type Sample struct {
	// Identifiers
	ID            string  `json:"id"`
	TLEID         string  `json:"tle_id"`
	EpochUnix     int64   `json:"epoch_unix"`
	PropagateMins float64 `json:"propagate_mins"`

	// True (noiseless) state
	TrueX, TrueY, TrueZ    float64 `json:"true_x,true_y,true_z"`
	TrueVX, TrueVY, TrueVZ float64 `json:"true_vx,true_vy,true_vz"`

	// Noisy observations
	ObsX, ObsY, ObsZ    float64 `json:"obs_x,obs_y,obs_z"`
	ObsVX, ObsVY, ObsVZ float64 `json:"obs_vx,obs_vy,obs_vz"`

	// Residuals (for maneuver detection training)
	ResidualX, ResidualY, ResidualZ float64 `json:"residual_x,residual_y,residual_z"`

	// Uncertainty (covariance diagonal)
	CovX, CovY, CovZ    float64 `json:"cov_x,cov_y,cov_z"`
	CovVX, CovVY, CovVZ float64 `json:"cov_vx,cov_vy,cov_vz"`

	// Metadata
	Split       string  `json:"split"` // "train", "val", "test"
	Risk        float64 `json:"risk"`  // collision risk proxy
	Confidence  float64 `json:"confidence"`
	ThreatClass string  `json:"threat_class"` // "low", "medium", "high"
}

// VersionRegistry tracks generated dataset versions for reproducibility.
type VersionRegistry struct {
	Versions map[string]DataVersion `json:"versions"`
}

// NewVersionRegistry creates a new version registry.
func NewVersionRegistry() *VersionRegistry {
	return &VersionRegistry{
		Versions: make(map[string]DataVersion),
	}
}

// Register adds a new dataset version to the registry.
func (r *VersionRegistry) Register(version DataVersion) error {
	if version.ID == "" {
		return fmt.Errorf("version ID is required")
	}
	r.Versions[version.ID] = version
	return nil
}

// GetVersion retrieves a specific dataset version by ID.
func (r *VersionRegistry) GetVersion(versionID string) (DataVersion, error) {
	v, exists := r.Versions[versionID]
	if !exists {
		return DataVersion{}, fmt.Errorf("version %s not found", versionID)
	}
	return v, nil
}

// GetLatestByPurpose returns the most recent version for a given purpose.
func (r *VersionRegistry) GetLatestByPurpose(purpose string) (DataVersion, error) {
	var latest DataVersion
	var latestTime time.Time

	for _, v := range r.Versions {
		if v.Purpose == purpose && v.GeneratedAt.After(latestTime) {
			latest = v
			latestTime = v.GeneratedAt
		}
	}

	if latestTime.IsZero() {
		return DataVersion{}, fmt.Errorf("no versions found for purpose %s", purpose)
	}
	return latest, nil
}

// SaveRegistry persists the version registry to JSON.
func (r *VersionRegistry) SaveRegistry(path string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadRegistry loads the version registry from JSON.
func LoadRegistry(path string) (*VersionRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var reg VersionRegistry
	if err := json.Unmarshal(data, &reg); err != nil {
		return nil, err
	}
	return &reg, nil
}

// ComputeChecksum computes MD5 checksum of a dataset file.
func ComputeChecksum(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}

// GenerateVersionID creates a unique version ID from config and timestamp.
func GenerateVersionID(purpose, model string) string {
	now := time.Now()
	seed := fmt.Sprintf("%s-%s-%d", purpose, model, now.UnixNano())
	hash := md5.Sum([]byte(seed))
	return fmt.Sprintf("%s-%d", fmt.Sprintf("%x", hash)[:12], now.UnixNano())
}
