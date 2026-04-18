package spatial

import (
	"sync"
	"time"
)

const (
	LatRes = 180
	LonRes = 360
	TimeRes = 24
)

var ShellAltitudes = []float64{500, 2000, 20000, 35786, 42000}

type CoverageTensor4D struct {
	// [Time][Shell][Word]
	Grid [TimeRes][5][]uint64
}

func NewCoverageTensor4D() *CoverageTensor4D {
	c := &CoverageTensor4D{}
	wordsPerShell := (LatRes * LonRes + 63) / 64
	for t := 0; t < TimeRes; t++ {
		for s := 0; s < 5; s++ {
			c.Grid[t][s] = make([]uint64, wordsPerShell)
		}
	}
	return c
}

// ComputeBlindSpots updates the tensor based on sensor registry and weather.
func (c *CoverageTensor4D) ComputeBlindSpots(reg *SensorRegistry, weather WeatherProvider, startTime time.Time) {
	var wg sync.WaitGroup

	// Parallelize across TimeSteps
	for t := 0; t < TimeRes; t++ {
		wg.Add(1)
		go func(tStep int) {
			defer wg.Done()
			currentTime := startTime.Add(time.Duration(tStep) * time.Hour)
			
			// For each shell
			for s := 0; s < len(ShellAltitudes); s++ {
				alt := ShellAltitudes[s]
				c.computeShellAtTime(tStep, s, alt, reg, weather, currentTime)
			}
		}(t)
	}
	wg.Wait()
}

func (c *CoverageTensor4D) computeShellAtTime(tStep, sIdx int, alt float64, reg *SensorRegistry, weather WeatherProvider, t time.Time) {
	// Pre-filter available sensors for this time/shell to avoid heavy LOS if weather-blind
	activeSensors := make([]*Sensor, 0)
	for i := range reg.Sensors {
		s := &reg.Sensors[i]
		w := weather.GetWeatherAt(s.Lat, s.Lon, t)
		
		available := true
		if s.Type == SensorOptical && w.CloudCover > s.CloudBlindThreshold {
			available = false
		}
		if s.Type == SensorRadar && w.RainRate > s.RainDegradeThreshold {
			// Optional: degrade range
			// For simplicity in this demo, we keep it active or reduce range
		}
		
		if available {
			activeSensors = append(activeSensors, s)
		}
	}

	// Iterate over grid cells
	for lat := 0; lat < LatRes; lat++ {
		latDeg := float64(lat) - 90.0 + 0.5
		for lon := 0; lon < LonRes; lon++ {
			lonDeg := float64(lon) - 180.0 + 0.5
			
			observable := false
			for _, s := range activeSensors {
				if s.CanSee(latDeg, lonDeg, alt) {
					observable = true
					break
				}
			}
			
			if observable {
				c.mark(tStep, sIdx, lat*LonRes + lon)
			}
		}
	}
}

func (c *CoverageTensor4D) mark(t, s, cellID int) {
	word := cellID / 64
	bit := uint(cellID % 64)
	c.Grid[t][s][word] |= (1 << bit)
}

func (c *CoverageTensor4D) GetBlindSpotRatio(t, s int) float64 {
	covered := 0
	for _, word := range c.Grid[t][s] {
		covered += bitsCount(word)
	}
	total := LatRes * LonRes
	return 1.0 - (float64(covered) / float64(total))
}

// Heatmap returns a slice of 0 (blind) or 1 (observable) for a given time/shell.
func (c *CoverageTensor4D) Heatmap(t, s int) []float64 {
	out := make([]float64, LatRes*LonRes)
	for i := 0; i < LatRes*LonRes; i++ {
		word := i / 64
		bit := uint(i % 64)
		if (c.Grid[t][s][word] & (1 << bit)) != 0 {
			out[i] = 1.0
		}
	}
	return out
}
