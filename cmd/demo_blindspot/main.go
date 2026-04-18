package main

import (
	"fmt"
	"strings"
	"ssis/pkg/spatial"
	"time"
)

func main() {
	fmt.Println("IAF SSA: Multi-Sensor Blind-Spot 4D Analysis Engine")
	fmt.Println("--------------------------------------------------")

	// 1. Initialize Registry and Mock Weather
	reg := spatial.NewDefaultSensorRegistry()
	weather := spatial.NewMockIMDProvider()
	tensor := spatial.NewCoverageTensor4D()

	// 2. Compute 4D Tensor (24 hour forecast)
	start := time.Now()
	startTime := time.Now()
	fmt.Printf("[PROVISIONAL] Starting LOS compute for %d sensors across 5 orbital shells...\n", len(reg.Sensors))
	tensor.ComputeBlindSpots(reg, weather, startTime)
	elapsed := time.Since(start)
	fmt.Printf("[OK] 4D Observability Tensor populated in %v\n\n", elapsed)

	// 3. Display Results for LEO-Near (500km) and GEO (35786km)
	fmt.Println("Blind-Spot Ratios (Network Obsolescence Percentage):")
	fmt.Println("Time (H) | LEO-Near (500km) | GEO (35786km) | Weather Event")
	fmt.Println("---------|-----------------|---------------|---------------")

	for t := 0; t < 24; t += 4 {
		leoRatio := tensor.GetBlindSpotRatio(t, 0)
		geoRatio := tensor.GetBlindSpotRatio(t, 3)
		
		event := "Nominal"
		// Check Mount Abu weather for this time
		w := weather.GetWeatherAt(24.59, 72.71, startTime.Add(time.Duration(t)*time.Hour))
		if w.CloudCover > 0.4 {
			event = "Mt. Abu Cloudy"
		}
		if w.CloudCover > 0.8 {
			event = "Mid-Peninsula Monsoon"
		}

		fmt.Printf(" T+%02dh   |      %5.1f%%      |     %5.1f%%     | %s\n", 
			t, leoRatio*100, geoRatio*100, event)
	}

	fmt.Println("\n[STRATEGIC] Notice: GEO visibility is highly sensitive to Ladakh/Mount Abu optical availability.")
	fmt.Printf("[IMD] Current Simulated Monsoon Intensity: %.1f%%\n", weather.MonsoonIntensity*100)
	
	// ASCII Heatmap snippet (Sample 5x10 grid of India region)
	fmt.Println("\nRegional Blind-Spot Snapshot (India Sector, LEO):")
	heatmap := tensor.Heatmap(0, 0) // T+0, LEO
	
	// India Lat: 8-38, Lon: 68-98
	for lat := 38; lat >= 8; lat -= 3 {
		fmt.Printf("%02dN | ", lat)
		for lon := 68; lon <= 98; lon += 3 {
			idx := (lat+90)*spatial.LonRes + (lon+180)
			if idx < len(heatmap) && heatmap[idx] > 0 {
				fmt.Print(" . ") // Observable
			} else {
				fmt.Print(" B ") // Blind
			}
		}
		fmt.Println()
	}
	fmt.Println("     " + strings.Repeat("---", 11))
	fmt.Println("       68E  74E  80E  86E  92E")
}
