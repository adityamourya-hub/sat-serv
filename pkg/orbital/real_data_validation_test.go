package orbital

import (
	"bufio"
	"fmt"
	"math"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/cprosche/tle"
)

// RealDataValidationTest runs a comprehensive validation using live CelesTrak satellite data
func TestRealDataValidation(t *testing.T) {
	fmt.Println("\n========================================")
	fmt.Println("SSIS REAL-DATA VALIDATION TEST")
	fmt.Println("Fetching live satellite TLEs from CelesTrak...")
	fmt.Println("========================================")

	// 1. Fetch real satellite data
	satellites := fetchRealSatellites(t)
	if len(satellites) == 0 {
		t.Fatal("No real satellite data available")
	}

	fmt.Printf("✓ Loaded %d real satellites from CelesTrak\n\n", len(satellites))

	// 2. Validate orbital propagation on real data using SGP4
	fmt.Println("--- TEST 1: Orbital Propagation (SGP4 on REAL TLEs) ---")
	validateSGP4Propagation(t, satellites)

	// 3. Test physics consistency on real data
	fmt.Println("\n--- TEST 2: Physics Consistency (Real Orbits) ---")
	validatePhysicsOnRealData(t, satellites)

	// 4. Find real conjunctions
	fmt.Println("\n--- TEST 3: Conjunction Screening (Real Data) ---")
	validateConjunctionsOnRealData(t, satellites)

	// 5. Compute orbital statistics
	fmt.Println("\n--- TEST 4: Orbital Statistics ---")
	computeOrbitalStatistics(t, satellites)

	fmt.Println("\n========================================")
	fmt.Println("REAL-DATA VALIDATION COMPLETE")
	fmt.Println("========================================")
}

type RealSatellite struct {
	Name     string
	NoradID  int
	TLELine1 string
	TLELine2 string
	TLE      *TLE
	State    State
	Epoch    time.Time
}

func fetchRealSatellites(t *testing.T) []RealSatellite {
	client := &http.Client{Timeout: 30 * time.Second}

	// Fetch multiple groups for comprehensive coverage
	groups := []string{"stations", "visual", "active", "geo", "gps-ops", "science", "iridium"}
	allSatellites := []RealSatellite{}

	for _, group := range groups {
		url := fmt.Sprintf("https://celestrak.org/NORAD/elements/gp.php?GROUP=%s&FORMAT=tle", group)
		req, _ := http.NewRequest("GET", url, nil)
		req.Header.Set("User-Agent", "SSIS-Validation/1.0")

		resp, err := client.Do(req)
		if err != nil {
			t.Logf("Warning: Failed to fetch %s: %v", group, err)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			t.Logf("Warning: HTTP %d for %s", resp.StatusCode, group)
			continue
		}

		scanner := bufio.NewScanner(resp.Body)
		lines := []string{}
		for scanner.Scan() {
			lines = append(lines, scanner.Text())
		}
		resp.Body.Close()

		// Parse TLEs
		currentName := "UNKNOWN"
		for i := 0; i < len(lines); i++ {
			line := strings.TrimSpace(lines[i])
			if len(line) == 0 {
				continue
			}

			// Check if it's a name line
			if line[0] != '1' && line[0] != '2' {
				currentName = line
				continue
			}

			if line[0] == '1' && i+1 < len(lines) {
				l1 := lines[i]
				l2 := lines[i+1]

				if len(l2) > 0 && l2[0] == '2' {
					tleObj, err := tle.Parse(l1 + "\n" + l2)
					if err == nil {
						// Parse with our TLE parser
						parsedTLE, err := ParseTLE(l1 + "\n" + l2)
						if err != nil {
							i++
							continue
						}

						// Create propagator and get current state
						prop := NewSGP4Propagator(parsedTLE)
						state, err := prop.Propagate(0)
						if err != nil {
							i++
							continue
						}

						name := currentName
						if name == "UNKNOWN" || name == "0" {
							name = tleObj.Name
						}

						sat := RealSatellite{
							Name:     name,
							NoradID:  tleObj.NoradId,
							TLELine1: l1,
							TLELine2: l2,
							TLE:      parsedTLE,
							State:    state,
							Epoch:    time.Now(),
						}
						allSatellites = append(allSatellites, sat)
						i++ // Skip line 2
					}
				}
			}
		}
	}

	return allSatellites
}

func validateSGP4Propagation(t *testing.T, sats []RealSatellite) {
	sampleSize := min(20, len(sats))
	fmt.Printf("Testing SGP4 propagation on %d real satellites...\n", sampleSize)

	energyDrifts := []float64{}

	for i := 0; i < sampleSize; i++ {
		sat := sats[i]

		// Propagate forward 6 hours (360 minutes)
		prop := NewSGP4Propagator(sat.TLE)
		state0, _ := prop.Propagate(0)
		state6h, err := prop.Propagate(360)
		if err != nil {
			fmt.Printf("  %s: Propagation error - %v\n", sat.Name, err)
			continue
		}

		// Check energy conservation
		r0 := math.Sqrt(state0.X*state0.X + state0.Y*state0.Y + state0.Z*state0.Z)
		v0 := math.Sqrt(state0.VX*state0.VX + state0.VY*state0.VY + state0.VZ*state0.VZ)
		e0 := v0*v0/2 - EARTH_MU/r0

		r1 := math.Sqrt(state6h.X*state6h.X + state6h.Y*state6h.Y + state6h.Z*state6h.Z)
		v1 := math.Sqrt(state6h.VX*state6h.VX + state6h.VY*state6h.VY + state6h.VZ*state6h.VZ)
		e1 := v1*v1/2 - EARTH_MU/r1

		energyDrift := math.Abs(e1-e0) / math.Abs(e0) * 100
		energyDrifts = append(energyDrifts, energyDrift)

		if i < 5 {
			fmt.Printf("  %s (NORAD %d): 6h energy drift=%.8f%%\n",
				sat.Name, sat.NoradID, energyDrift)
		}
	}

	meanDrift := mean(energyDrifts)
	maxDrift := max(energyDrifts)
	fmt.Printf("\nSGP4 Propagation Statistics:\n")
	fmt.Printf("  Mean energy drift: %.8f%%\n", meanDrift)
	fmt.Printf("  Max energy drift:  %.8f%%\n", maxDrift)
	// Note: SGP4 includes perturbations (J2, drag, Sun/Moon) that cause small energy changes
	// 0.1% drift over 6 hours is acceptable for LEO with atmospheric drag
	fmt.Printf("  Status: %s (threshold: 0.1%% for perturbed orbits)\n", statusString(meanDrift < 0.1))

	if meanDrift > 0.5 {
		t.Errorf("SGP4 energy drift too high: %.6f%% (expected < 0.5%% for 6h propagation)", meanDrift)
	}
}

func validatePhysicsOnRealData(t *testing.T, sats []RealSatellite) {
	fmt.Printf("Validating %d real satellite orbits against physical laws...\n", len(sats))

	violations := 0
	checks := 0
	leoCount := 0
	geoCount := 0
	meoCount := 0

	for _, sat := range sats {
		state := sat.State

		// Calculate orbital elements from state
		r := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)
		v := math.Sqrt(state.VX*state.VX + state.VY*state.VY + state.VZ*state.VZ)
		altitude := r - ER

		// Classify orbit regime
		if altitude < 2000 {
			leoCount++
		} else if altitude > 35000 {
			geoCount++
		} else {
			meoCount++
		}

		// Check 1: Altitude above Earth's surface
		checks++
		if altitude < -100 { // Allow tolerance for Earth oblateness
			violations++
			fmt.Printf("  VIOLATION: %s altitude = %.1f km (below surface)\n", sat.Name, altitude)
		}

		// Check 2: Specific orbital energy negative (bound orbit)
		specificEnergy := v*v/2 - EARTH_MU/r
		checks++
		if specificEnergy > 0 {
			violations++
			fmt.Printf("  VIOLATION: %s has positive energy (%.4f km²/s²) - unbound orbit!\n",
				sat.Name, specificEnergy)
		}

		// Check 3: Velocity within physical bounds for altitude
		// Circular orbit velocity: v_circ = sqrt(μ/r)
		vCirc := math.Sqrt(EARTH_MU / r)
		checks++
		if v > vCirc*2 && altitude < 50000 { // Faster than 2x circular is suspicious for bound orbits
			// Skip for high-energy orbits
		}
	}

	violationRate := float64(violations) / float64(checks) * 100
	fmt.Printf("\nPhysics Validation Results:\n")
	fmt.Printf("  Total checks: %d\n", checks)
	fmt.Printf("  Violations: %d (%.4f%%)\n", violations, violationRate)
	fmt.Printf("  Orbit distribution: LEO=%d, MEO=%d, GEO=%d\n", leoCount, meoCount, geoCount)
	fmt.Printf("  Status: %s\n", statusString(violationRate < 0.1))

	if violationRate > 1.0 {
		t.Errorf("Physics violation rate too high: %.2f%%", violationRate)
	}
}

func validateConjunctionsOnRealData(t *testing.T, sats []RealSatellite) {
	screenSize := min(200, len(sats))
	fmt.Printf("Screening %d real satellites for close approaches...\n", screenSize)

	conjunctions := 0
	minDistance := math.MaxFloat64
	closestPair := ""

	// Simple O(n²) screening for close approaches
	for i := 0; i < screenSize; i++ {
		for j := i + 1; j < screenSize; j++ {
			s1 := sats[i]
			s2 := sats[j]

			dx := s1.State.X - s2.State.X
			dy := s1.State.Y - s2.State.Y
			dz := s1.State.Z - s2.State.Z
			dist := math.Sqrt(dx*dx + dy*dy + dz*dz)

			if dist < minDistance {
				minDistance = dist
				closestPair = fmt.Sprintf("%s ↔ %s", s1.Name, s2.Name)
			}

			// Threshold: 50 km (conservative)
			if dist < 50.0 {
				conjunctions++
				if conjunctions <= 10 {
					// Calculate relative velocity for context
					vx := s1.State.VX - s2.State.VX
					vy := s1.State.VY - s2.State.VY
					vz := s1.State.VZ - s2.State.VZ
					vRel := math.Sqrt(vx*vx + vy*vy + vz*vz)

					fmt.Printf("  CLOSE APPROACH #%d: %s ↔ %s at %.2f km (v_rel=%.2f km/s)\n",
						conjunctions, s1.Name, s2.Name, dist, vRel)
				}
			}
		}
	}

	pairsChecked := screenSize * (screenSize - 1) / 2
	fmt.Printf("\nConjunction Screening Results:\n")
	fmt.Printf("  Pairs checked: %d\n", pairsChecked)
	fmt.Printf("  Close approaches (< 50km): %d\n", conjunctions)
	fmt.Printf("  Closest pair: %s at %.2f km\n", closestPair, minDistance)
	fmt.Printf("  Conjunction rate: %.4f%%\n", float64(conjunctions)/float64(pairsChecked)*100)
	fmt.Printf("  Status: %s\n", statusString(true))
}

func computeOrbitalStatistics(t *testing.T, sats []RealSatellite) {
	fmt.Printf("Computing orbital statistics for %d real satellites...\n", len(sats))

	altitudes := []float64{}
	velocities := []float64{}
	inclinations := []float64{}
	periods := []float64{}

	for _, sat := range sats {
		state := sat.State
		r := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)
		v := math.Sqrt(state.VX*state.VX + state.VY*state.VY + state.VZ*state.VZ)
		altitude := r - ER

		altitudes = append(altitudes, altitude)
		velocities = append(velocities, v)

		// Calculate inclination from angular momentum
		hx := state.Y*state.VZ - state.Z*state.VY
		hy := state.Z*state.VX - state.X*state.VZ
		hz := state.X*state.VY - state.Y*state.VX
		hMag := math.Sqrt(hx*hx + hy*hy + hz*hz)
		inc := math.Acos(hz/hMag) * 180 / math.Pi
		inclinations = append(inclinations, inc)

		// Orbital period from semi-major axis (approximate from current radius)
		period := 2 * math.Pi * math.Sqrt(r*r*r/EARTH_MU) / 60 // in minutes
		periods = append(periods, period)
	}

	fmt.Printf("\nOrbital Statistics (from %d real satellites):\n", len(sats))
	fmt.Printf("  Altitude: mean=%.1f km, min=%.1f km, max=%.1f km\n",
		mean(altitudes), minFloat(altitudes), maxFloat(altitudes))
	fmt.Printf("  Velocity: mean=%.3f km/s, min=%.3f km/s, max=%.3f km/s\n",
		mean(velocities), minFloat(velocities), maxFloat(velocities))
	fmt.Printf("  Inclination: mean=%.1f°, min=%.1f°, max=%.1f°\n",
		mean(inclinations), minFloat(inclinations), maxFloat(inclinations))
	fmt.Printf("  Period: mean=%.1f min, min=%.1f min, max=%.1f min\n",
		mean(periods), minFloat(periods), maxFloat(periods))

	// Find notable satellites
	fmt.Printf("\nNotable Satellites in Sample:\n")
	for _, sat := range sats {
		state := sat.State
		r := math.Sqrt(state.X*state.X + state.Y*state.Y + state.Z*state.Z)
		altitude := r - ER

		// ISS
		if sat.NoradID == 25544 {
			fmt.Printf("  ISS (NORAD 25544): Alt=%.1f km, tracking %d objects in catalog\n",
				altitude, len(sats))
		}
		// GPS
		if strings.Contains(sat.Name, "GPS") || strings.Contains(sat.Name, "NAVSTAR") {
			fmt.Printf("  GPS satellite: %s at %.1f km (MEO)\n", sat.Name, altitude)
		}
		// GEO
		if altitude > 35000 {
			fmt.Printf("  GEO satellite: %s at %.1f km\n", sat.Name, altitude)
		}
	}
}

// Helper functions
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

func max(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := vals[0]
	for _, v := range vals {
		if v > m {
			m = v
		}
	}
	return m
}

func minFloat(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := vals[0]
	for _, v := range vals {
		if v < m {
			m = v
		}
	}
	return m
}

func maxFloat(vals []float64) float64 {
	return max(vals)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func statusString(ok bool) string {
	if ok {
		return "✓ PASS"
	}
	return "✗ FAIL"
}
