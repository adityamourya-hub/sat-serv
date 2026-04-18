package spatial

import (
	"math"
	"time"
)

type WeatherData struct {
	CloudCover float64 // 0.0 to 1.0
	RainRate   float64 // mm/h
}

type WeatherProvider interface {
	GetWeatherAt(lat, lon float64, t time.Time) WeatherData
}

type MockIMDProvider struct {
	MonsoonIntensity float64 // 0.0 to 1.0
	WindSpeedKph     float64
	StartTime        time.Time
}

func NewMockIMDProvider() *MockIMDProvider {
	return &MockIMDProvider{
		MonsoonIntensity: 0.6,
		WindSpeedKph:     25.0,
		StartTime:        time.Now(),
	}
}

// GetWeatherAt simulates a time-evolving weather front across India.
func (p *MockIMDProvider) GetWeatherAt(lat, lon float64, t time.Time) WeatherData {
	// Simple simulation: a monsoon front that moves from SW (Kerala) to NE (Ladakh)
	// Base front orientation: lon - lat = constant
	
	elapsedHours := t.Sub(p.StartTime).Hours()
	
	// Phase shift based on time and wind
	// Moving at ~25kph across the peninsula (~3000km)
	// Roughly 0.2 degrees per hour
	frontProgress := elapsedHours * 0.2
	
	// Weather potential: centered around a diagonal band
	// Band: lon - 1.2*lat = value
	potential := math.Sin((lon - 1.2*lat - frontProgress) * 0.1)
	
	// Add some perlin-like noise (sine-wave harmonics)
	noise := 0.2 * math.Sin(lat * 0.5) * math.Cos(lon * 0.5)
	
	val := (potential + 1.0) / 2.0 + noise
	val = math.Max(0, math.Min(1, val * p.MonsoonIntensity))

	rain := 0.0
	if val > 0.7 {
		rain = (val - 0.7) * 50.0 // Heavy rain up to 15mm/h
	}

	return WeatherData{
		CloudCover: val,
		RainRate:   rain,
	}
}
