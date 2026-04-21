package spatial

import (
	"fmt"
	"math"
	"os"

	"github.com/goccy/go-yaml"
)

type SensorType string

const (
	SensorOptical SensorType = "optical"
	SensorRadar   SensorType = "radar"
)

type SensorStatus string

const (
	SensorActive      SensorStatus = "active"
	SensorStandby     SensorStatus = "standby"
	SensorMaintenance SensorStatus = "maintenance"
	SensorOffline     SensorStatus = "offline"
)

type Sensor struct {
	ID                   string       `yaml:"id"`
	Type                 SensorType   `yaml:"type"`
	Lat                  float64      `yaml:"lat"`
	Lon                  float64      `yaml:"lon"`
	Alt                  float64      `yaml:"alt_km"`
	MaxRange             float64      `yaml:"max_range_km"`
	MinElevation         float64      `yaml:"min_elevation_deg"`
	CloudBlindThreshold  float64      `yaml:"cloud_blind_threshold,omitempty"`  // For optical
	RainDegradeThreshold float64      `yaml:"rain_degrade_threshold,omitempty"` // For radar
	Status               SensorStatus `yaml:"status"`
}

// IsAvailable returns true if sensor can be tasked
func (s *Sensor) IsAvailable() bool {
	return s.Status == SensorActive || s.Status == SensorStandby
}

type SensorRegistry struct {
	Sensors []Sensor `yaml:"sensors"`
}

func NewDefaultSensorRegistry() *SensorRegistry {
	return &SensorRegistry{
		Sensors: []Sensor{
			{
				ID:                  "leh_optical",
				Type:                SensorOptical,
				Lat:                 34.1526,
				Lon:                 77.5771,
				Alt:                 3.500,
				MaxRange:            36000, // GEO reach
				MinElevation:        15,
				CloudBlindThreshold: 0.40,
				Status:              SensorActive,
			},
			{
				ID:                  "mount_abu_optical",
				Type:                SensorOptical,
				Lat:                 24.5925,
				Lon:                 72.7156,
				Alt:                 1.722,
				MaxRange:            36000,
				MinElevation:        15,
				CloudBlindThreshold: 0.40,
				Status:              SensorActive,
			},
			{
				ID:                   "nellore_radar",
				Type:                 SensorRadar,
				Lat:                  13.4348,
				Lon:                  79.9197,
				Alt:                  0.020,
				MaxRange:             2000, // LEO optimized
				MinElevation:         10,
				RainDegradeThreshold: 5.0, // mm/h
				Status:               SensorActive,
			},
		},
	}
}

// LoadSensorRegistryYAML loads sensor configuration from YAML file
func LoadSensorRegistryYAML(path string) (*SensorRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read sensor config: %w", err)
	}

	var reg SensorRegistry
	if err := yaml.Unmarshal(data, &reg); err != nil {
		return nil, fmt.Errorf("failed to parse sensor config: %w", err)
	}

	// Validate loaded sensors
	for i, s := range reg.Sensors {
		if s.ID == "" {
			return nil, fmt.Errorf("sensor at index %d missing ID", i)
		}
		if s.Type != SensorOptical && s.Type != SensorRadar {
			return nil, fmt.Errorf("sensor %s has invalid type: %s", s.ID, s.Type)
		}
	}

	return &reg, nil
}

// GetAvailableSensors returns only sensors that can be tasked
func (sr *SensorRegistry) GetAvailableSensors() []Sensor {
	available := make([]Sensor, 0)
	for _, s := range sr.Sensors {
		if s.IsAvailable() {
			available = append(available, s)
		}
	}
	return available
}

// Deprecated: LoadSensorRegistry kept for backward compatibility
func LoadSensorRegistry(path string) (*SensorRegistry, error) {
	return LoadSensorRegistryYAML(path)
}

// CanSee checks if a target point (lat, lon, alt_km) is within the sensor's line-of-sight.
func (s *Sensor) CanSee(tLat, tLon, tAlt float64) bool {
	// 1. Geodetic to ECEF (Spherical approximation for SSA broad phase)
	const R = 6371.0

	sX, sY, sZ := llhToEcef(s.Lat, s.Lon, s.Alt+R)
	tX, tY, tZ := llhToEcef(tLat, tLon, tAlt+R)

	// 2. Relative Vector
	dx, dy, dz := tX-sX, tY-sY, tZ-sZ
	dist := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Range Check
	if dist > s.MaxRange {
		return false
	}

	// 3. Elevation Check
	// Normal vector at sensor (approximate)
	nx, ny, nz := sX/(s.Alt+R), sY/(s.Alt+R), sZ/(s.Alt+R)

	// dot = nx*dx + ny*dy + nz*dz
	dot := nx*dx + ny*dy + nz*dz

	// elevation = asin(dot / dist)
	elev := math.Asin(dot/dist) * 180.0 / math.Pi

	return elev >= s.MinElevation
}

func llhToEcef(lat, lon, r float64) (x, y, z float64) {
	latRad := lat * math.Pi / 180.0
	lonRad := lon * math.Pi / 180.0
	x = r * math.Cos(latRad) * math.Cos(lonRad)
	y = r * math.Cos(latRad) * math.Sin(lonRad)
	z = r * math.Sin(latRad)
	return
}
