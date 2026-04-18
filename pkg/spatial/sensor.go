package spatial

import (
	"encoding/json"
	"math"
	"os"
)

type SensorType string

const (
	SensorOptical SensorType = "optical"
	SensorRadar   SensorType = "radar"
)

type Sensor struct {
	ID                    string     `json:"id"`
	Type                  SensorType `json:"type"`
	Lat                   float64    `json:"lat"`
	Lon                   float64    `json:"lon"`
	Alt                   float64    `json:"alt_km"`
	MaxRange              float64    `json:"max_range_km"`
	MinElevation          float64    `json:"min_elevation_deg"`
	CloudBlindThreshold   float64    `json:"cloud_blind_threshold"` // For optical
	RainDegradeThreshold  float64    `json:"rain_degrade_threshold"` // For radar
}

type SensorRegistry struct {
	Sensors []Sensor `json:"sensors"`
}

func NewDefaultSensorRegistry() *SensorRegistry {
	return &SensorRegistry{
		Sensors: []Sensor{
			{
				ID:                   "leh_optical",
				Type:                 SensorOptical,
				Lat:                  34.1526,
				Lon:                  77.5771,
				Alt:                  3.500,
				MaxRange:             36000, // GEO reach
				MinElevation:         15,
				CloudBlindThreshold:  0.40,
			},
			{
				ID:                   "mount_abu_optical",
				Type:                 SensorOptical,
				Lat:                  24.5925,
				Lon:                  72.7156,
				Alt:                  1.722,
				MaxRange:             36000,
				MinElevation:         15,
				CloudBlindThreshold:  0.40,
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
			},
		},
	}
}

func LoadSensorRegistry(path string) (*SensorRegistry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var reg SensorRegistry
	if err := json.Unmarshal(data, &reg); err != nil {
		return nil, err
	}
	return &reg, nil
}

// CanSee checks if a target point (lat, lon, alt_km) is within the sensor's line-of-sight.
func (s *Sensor) CanSee(tLat, tLon, tAlt float64) bool {
	// 1. Geodetic to ECEF (Spherical approximation for SSA broad phase)
	const R = 6371.0
	
	sX, sY, sZ := llhToEcef(s.Lat, s.Lon, s.Alt + R)
	tX, tY, tZ := llhToEcef(tLat, tLon, tAlt + R)

	// 2. Relative Vector
	dx, dy, dz := tX-sX, tY-sY, tZ-sZ
	dist := math.Sqrt(dx*dx + dy*dy + dz*dz)

	// Range Check
	if dist > s.MaxRange {
		return false
	}

	// 3. Elevation Check
	// Normal vector at sensor (approximate)
	nx, ny, nz := sX/ (s.Alt + R), sY/ (s.Alt + R), sZ/ (s.Alt + R)
	
	// dot = nx*dx + ny*dy + nz*dz
	dot := nx*dx + ny*dy + nz*dz
	
	// elevation = asin(dot / dist)
	elev := math.Asin(dot / dist) * 180.0 / math.Pi
	
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
