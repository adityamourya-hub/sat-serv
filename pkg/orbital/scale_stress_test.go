package orbital_test

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"runtime"
	"ssis/pkg/catalog"
	"ssis/pkg/maneuver"
	"ssis/pkg/octree"
	"ssis/pkg/physics"
	"ssis/pkg/proximity"
	"strings"
	"testing"
	"time"

	"github.com/cprosche/tle"
)

var (
	fullCatalog  []catalog.ObjectRecord
	indianAssets []catalog.ObjectRecord
	indianROIs   = []proximity.RegionOfInterest{
		{LatDeg: 12.9789, LonDeg: 77.5917, RadiusKm: 500}, // Bangalore
		{LatDeg: 28.6100, LonDeg: 77.2300, RadiusKm: 500}, // Delhi
		{LatDeg: 19.0761, LonDeg: 72.8775, RadiusKm: 500}, // Mumbai
		{LatDeg: 13.0825, LonDeg: 80.2750, RadiusKm: 500}, // Chennai
		{LatDeg: 17.3617, LonDeg: 78.4747, RadiusKm: 500}, // Hyderabad
		{LatDeg: 22.5675, LonDeg: 88.3700, RadiusKm: 500}, // Kolkata
		{LatDeg: 23.0225, LonDeg: 72.5714, RadiusKm: 500}, // Ahmedabad
		{LatDeg: 18.5203, LonDeg: 73.8567, RadiusKm: 500}, // Pune
		{LatDeg: 26.9124, LonDeg: 75.7873, RadiusKm: 500}, // Jaipur
		{LatDeg: 26.8467, LonDeg: 80.9462, RadiusKm: 500}, // Lucknow
	}
)

func init() {
	// Attempt Load, but Benchmarks will call it or fail if empty.
}

func fetchTLEs() (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", "https://celestrak.org/NORAD/elements/gp.php?GROUP=all&FORMAT=tle", nil)
	req.Header.Set("User-Agent", "SSIS-IAF-ADITI/1.0")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("http error: %d", resp.StatusCode)
	}

	var sb strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		sb.WriteString(scanner.Text() + "\n")
	}
	return sb.String(), nil
}

func loadFullCatalog() {
	raw, err := fetchTLEs()
	if err != nil {
		fmt.Printf("Warning: Failed to fetch online TLEs: %v. Using synthetic fallback.\n", err)
		// Multi-object fallback (2 objects)
		raw = "ISS (ZARYA)\n1 25544U 98067A   26101.50000000  .00016717  00000-0  30242-3 0  9999\n2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.50033100421234\n" +
			"CARTOSAT-3\n1 44788U 19081A   26101.50000000  .00000123  00000-0  12345-4 0  9998\n2 44788  97.4567 123.4567 0001234  45.6789 314.3210 15.12345678123456\n"
	}

	lines := strings.Split(raw, "\n")
	baseRecords := make([]catalog.ObjectRecord, 0)

	// Track common name for 3-line format
	currentName := "UNKNOWN"

	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if len(line) == 0 {
			continue
		}

		// Check if it's a name line (not starting with 1 or 2)
		if line[0] != '1' && line[0] != '2' {
			currentName = line
			continue
		}

		if line[0] == '1' && i+1 < len(lines) {
			l1 := lines[i]
			l2 := lines[i+1]
			// Ensure Line 2 starts with '2'
			if len(l2) > 0 && l2[0] == '2' {
				t, err := tle.Parse(l1 + "\n" + l2)
				if err == nil {
					// Success, calculate elements
					n_rad_s := t.MeanMotion * 2.0 * math.Pi / 86400.0
					mu := 398600.4418
					a := math.Pow(mu/(n_rad_s*n_rad_s), 1.0/3.0)

					k := physics.Keplerian{
						A: a, E: t.Eccentricity, I: t.InclinationDegrees * math.Pi / 180.0,
						RAAN: t.RightAscensionDegrees * math.Pi / 180.0, ARGP: t.PerigeeArgumentDegrees * math.Pi / 180.0,
					}
					pos := physics.KeplerToECI(k, t.MeanAnomalyDegrees*math.Pi/180.0, mu)

					name := t.Name
					if name == "" || name == "0" {
						name = currentName
					}

					country := "UNK"
					if strings.Contains(name, "GSAT") || strings.Contains(name, "INSAT") || strings.Contains(name, "RISAT") || strings.Contains(name, "CARTOSAT") {
						country = "IND"
					}

					rec := catalog.ObjectRecord{
						ID: t.NoradId, Name: name, CountryCode: country, AssetType: "SATELLITE",
						Pos: octree.Vector3{X: pos[0], Y: pos[1], Z: pos[2]},
					}
					baseRecords = append(baseRecords, rec)
					if country == "IND" || strings.Contains(name, "ISS") {
						indianAssets = append(indianAssets, rec)
					}
					i++ // Skip Line 2
				}
			}
		}
	}

	// 2. Perturb to 1,000,000 objects
	const TotalTarget = 1000000
	fullCatalog = make([]catalog.ObjectRecord, 0, TotalTarget)
	fullCatalog = append(fullCatalog, baseRecords...)

	if len(baseRecords) == 0 {
		fmt.Println("Warning: No base TLEs loaded. Generating fully synthetic 1M catalog.")
		rng := rand.New(rand.NewSource(0xAD1712026))
		for i := 0; i < TotalTarget; i++ {
			a := 6378.0 + 300.0 + rng.Float64()*35000.0
			k := physics.Keplerian{A: a, E: rng.Float64() * 0.1, I: rng.Float64() * math.Pi, RAAN: rng.Float64() * 2 * math.Pi, ARGP: rng.Float64() * 2 * math.Pi}
			pos := physics.KeplerToECI(k, rng.Float64()*2*math.Pi, 398600.4418)
			fullCatalog = append(fullCatalog, catalog.ObjectRecord{
				ID: 100000 + i, Name: fmt.Sprintf("SYN-%d", i), CountryCode: "SYN", AssetType: "DEBRIS",
				Pos: octree.Vector3{X: pos[0], Y: pos[1], Z: pos[2]},
			})
		}
		// Add some Indian assets for MOID benchmark
		for i := 0; i < 200; i++ {
			rec := fullCatalog[i]
			rec.CountryCode = "IND"
			rec.Name = fmt.Sprintf("IND-ASSET-%d", i)
			indianAssets = append(indianAssets, rec)
		}
		return
	}

	rng := rand.New(rand.NewSource(0xAD1712026))
	idx := 0
	for len(fullCatalog) < TotalTarget {
		base := baseRecords[idx%len(baseRecords)]
		// Convert to Keplerian then perturb
		k := physics.CartesianToKepler(physics.State{Pos: physics.Vector3{base.Pos.X, base.Pos.Y, base.Pos.Z}}, 398600.4418)

		// SMA: ±5 km Gaussian
		k.A += rng.NormFloat64() * 5.0
		// Ecc: ±0.001
		k.E = math.Abs(k.E + rng.NormFloat64()*0.001)
		// Inclination: ±0.05°
		k.I += rng.NormFloat64() * 0.05 * (math.Pi / 180.0)
		// RAAN/ArgP/MA: ±0.1°
		k.RAAN += rng.NormFloat64() * 0.1 * (math.Pi / 180.0)
		k.ARGP += rng.NormFloat64() * 0.1 * (math.Pi / 180.0)

		pos := physics.KeplerToECI(k, rng.Float64()*2*math.Pi, 398600.4418)
		fullCatalog = append(fullCatalog, catalog.ObjectRecord{
			ID:          100000 + len(fullCatalog),
			Name:        fmt.Sprintf("%s-DEB-%d", base.Name, len(fullCatalog)),
			CountryCode: "DEB",
			AssetType:   "DEBRIS",
			Pos:         octree.Vector3{X: pos[0], Y: pos[1], Z: pos[2]},
		})
		idx++
	}
	fmt.Printf("Catalog Scaled: %d objects ready.\n", len(fullCatalog))
}

func BenchmarkVolumetricMillion(b *testing.B) {
	if len(fullCatalog) == 0 {
		loadFullCatalog()
	}

	store := catalog.NewStore(500.0)
	store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: fullCatalog}, true)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, roi := range indianROIs {
			start := time.Now()
			objs := proximity.ScanGroundRegion(roi, store, time.Now())
			_ = objs
			elapsed := time.Since(start)
			if elapsed > 100*time.Millisecond {
				b.Errorf("Volumetric query too slow: %v", elapsed)
			}
		}
	}
}

func BenchmarkMOIDFullCatalog(b *testing.B) {
	if len(fullCatalog) == 0 {
		loadFullCatalog()
	}

	// Brute force is too slow (200 * 1M = 200M MOIDs)
	// We use the Octree to prune to candidates within 500km
	store := catalog.NewStore(500.0)
	store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: fullCatalog}, true)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		conjunctions := 0
		for _, asset := range indianAssets {
			// Broad Phase: Query Octree for objects within 500km
			min := octree.Vector3{X: asset.Pos.X - 500, Y: asset.Pos.Y - 500, Z: asset.Pos.Z - 500}
			max := octree.Vector3{X: asset.Pos.X + 500, Y: asset.Pos.Y + 500, Z: asset.Pos.Z + 500}
			candidates := store.QueryRange(min, max, 0, 1e12)

			for _, cand := range candidates {
				s1 := physics.State{Pos: physics.Vector3{asset.Pos.X, asset.Pos.Y, asset.Pos.Z}}
				s2 := physics.State{Pos: physics.Vector3{cand.Pos.X, cand.Pos.Y, cand.Pos.Z}}
				_ = proximity.CalculateMOID(s1, s2)
				conjunctions++
			}
		}
		elapsed := time.Since(start)
		if elapsed > 5*time.Second {
			b.Errorf("MOID screen too slow: %v (found %d pairs)", elapsed, conjunctions)
		}
	}
}

func TestScaleStressFullCatalog(t *testing.T) {
	if len(fullCatalog) == 0 {
		loadFullCatalog()
	}

	store := catalog.NewStore(1000.0)
	store.ApplyDeltaWithRebuild(catalog.Delta{Upserts: fullCatalog}, true)

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	t.Logf("Initial Memory: %.2f GB", float64(mem.Alloc)/1e9)

	initialGoroutines := runtime.NumGoroutine()

	// 24 Hour Simulation
	for h := 0; h < 24; h++ {
		// Inject 1000 ballistic arcs
		for i := 0; i < 1000; i++ {
			s := physics.State{Pos: physics.Vector3{6378 + 50, 0, 0}, Vel: physics.Vector3{1.0, 3.0, 1.0}}
			_ = maneuver.PredictSubOrbitalThreat(s, 1600000000)
		}

		// Run Volumetric Scans
		for _, roi := range indianROIs {
			_ = proximity.ScanGroundRegion(roi, store, time.Now())
		}

		// Periodic Conjunction Screen
		if h%1 == 0 {
			for _, asset := range indianAssets {
				min := octree.Vector3{X: asset.Pos.X - 200, Y: asset.Pos.Y - 200, Z: asset.Pos.Z - 200}
				max := octree.Vector3{X: asset.Pos.X + 200, Y: asset.Pos.Y + 200, Z: asset.Pos.Z + 200}
				_ = store.QueryRange(min, max, 0, 1e12)
			}
		}
	}

	runtime.ReadMemStats(&mem)
	t.Logf("Peak Memory: %.2f GB", float64(mem.Alloc)/1e9)
	if mem.Alloc > 6*1e9 {
		t.Errorf("Memory exceeds 6GB: %.2f GB", float64(mem.Alloc)/1e9)
	}

	if runtime.NumGoroutine() > initialGoroutines+10 {
		t.Errorf("Goroutine leak detected: %d -> %d", initialGoroutines, runtime.NumGoroutine())
	}
}
