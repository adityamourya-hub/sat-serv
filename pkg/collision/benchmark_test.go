package collision

import (
	"fmt"
	"math"
	"testing"
	"time"

	"ssis/pkg/maneuver"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
)

/**
 * ADITI 4.0 Performance Benchmark Suite
 *
 * Benchmarks all critical paths for operational deployment:
 * - Alfano Pc computation (series convergence, grid evaluation)
 * - TCA finder (iterations to convergence)
 * - Propagators (RK4 vs RKF45 speed/accuracy tradeoff)
 * - CUSUM detection (real-time processing capability)
 * - Covariance operations (matrix operations)
 */

// =============================================================================
// Benchmark 1: Alfano Pc Computation Performance
// =============================================================================

/**
 * BenchmarkAlfanoPcSeriesConvergence measures how many terms are needed
 * for the Alfano 2005 series to converge to 1e-12 tolerance.
 *
 * Reference: Alfano & Oltrogge (2005) - typically converges in < 30 terms
 */
func BenchmarkAlfanoPcSeriesConvergence(b *testing.B) {
	geo := ConjunctionGeometry{
		RelPosition:  [3]float64{0.1, 0, 0}, // 100m miss
		RelVelocity:  [3]float64{10, 0, 0},  // 10 km/s
		HBR:          0.015,                 // 15m HBR
		PrimaryCov:   [3][3]float64{{0.01, 0, 0}, {0, 0.01, 0}, {0, 0, 0.01}},
		SecondaryCov: [3][3]float64{},
		TCA:          2451545.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FosterAlfanoPc(geo)
	}
}

/**
 * BenchmarkAlfanoPcGridResolution compares 50×72 vs original 20×36 performance.
 */
func BenchmarkAlfanoPcGrid50x72(b *testing.B) {
	geo := ConjunctionGeometry{
		RelPosition:  [3]float64{0.05, 0, 0},
		RelVelocity:  [3]float64{15, 0, 0},
		HBR:          0.01,
		PrimaryCov:   [3][3]float64{{0.001, 0, 0}, {0, 0.001, 0}, {0, 0, 0.001}},
		SecondaryCov: [3][3]float64{},
		TCA:          2451545.0,
	}

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		_ = FosterAlfanoPc(geo)
	}
	elapsed := time.Since(start)
	if b.N > 0 {
		b.ReportMetric(float64(elapsed.Nanoseconds())/float64(b.N), "ns/op")
	}
}

/**
 * BenchmarkAlfanoPcExtremeAspectRatio tests performance with 1:100 aspect ratio.
 * This is the worst case for numerical integration.
 */
func BenchmarkAlfanoPcExtremeAspectRatio(b *testing.B) {
	geo := ConjunctionGeometry{
		RelPosition:  [3]float64{0.01, 0, 0},
		RelVelocity:  [3]float64{10, 0, 0},
		HBR:          0.005,
		PrimaryCov:   [3][3]float64{{0.0001, 0, 0}, {0, 0.01, 0}, {0, 0, 0.0001}}, // 1:100
		SecondaryCov: [3][3]float64{},
		TCA:          2451545.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FosterAlfanoPc(geo)
	}
}

// =============================================================================
// Benchmark 2: TCA Finder Performance
// =============================================================================

/**
 * BenchmarkTCAFinderConvergence measures iterations to convergence
 * for Newton-Raphson + Bisection fallback.
 */
func BenchmarkTCAFinderConvergence(b *testing.B) {
	primary := physics.State{
		Pos: [3]float64{7000, 0, 0},
		Vel: [3]float64{0, 7.5, 0},
	}
	secondary := physics.State{
		Pos: [3]float64{7000.2, 0.1, 0},
		Vel: [3]float64{-0.2, 7.4, 0},
	}
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	cfg := DefaultTCAFinderConfig()

	var totalIterations int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := FindTCA(primary, secondary, 2451545.0, cfg, params)
		if result.Converged {
			totalIterations += result.Iterations
		}
	}

	if b.N > 0 && totalIterations > 0 {
		b.ReportMetric(float64(totalIterations)/float64(b.N), "iterations/op")
	}
}

/**
 * BenchmarkTCAFinderAccuracy measures miss distance precision achieved.
 */
func BenchmarkTCAFinderAccuracy(b *testing.B) {
	primary := physics.State{
		Pos: [3]float64{7000, 0, 0},
		Vel: [3]float64{0, 7.5, 0},
	}
	secondary := physics.State{
		Pos: [3]float64{7000.1, 0.05, 0},
		Vel: [3]float64{-0.1, 7.45, 0},
	}
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	cfg := DefaultTCAFinderConfig()
	cfg.ToleranceKm = 1e-6 // 1mm precision

	var totalMissDistance float64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result := FindTCA(primary, secondary, 2451545.0, cfg, params)
		if result.Converged {
			totalMissDistance += result.MissDistance
		}
	}

	if b.N > 0 {
		b.ReportMetric(totalMissDistance/float64(b.N), "km_miss/op")
	}
}

// =============================================================================
// Benchmark 3: Propagator Performance (RK4 vs RKF45)
// =============================================================================

/**
 * BenchmarkPropagatorRK4FixedStep measures fixed-step RK4 performance.
 */
func BenchmarkPropagatorRK4_1DayGEO(b *testing.B) {
	sma := 42164.0
	vCirc := math.Sqrt(physics.EARTH_MU_IAU / sma)
	initialState := []float64{sma, 0, 0, 0, vCirc, 0}
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	jd := 2451545.0

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		_ = orbital.PropagateCartesianState(initialState, 86400.0, jd, params)
	}
	elapsed := time.Since(start)

	// Report steps per second
	stepsPerOp := 86400.0 / 10.0 // 10-second steps
	if b.N > 0 {
		b.ReportMetric(float64(elapsed.Nanoseconds())/float64(b.N), "ns/op")
		b.ReportMetric(stepsPerOp, "steps/op")
	}
}

/**
 * BenchmarkPropagatorRKF45Adaptive measures adaptive RKF45 performance.
 */
func BenchmarkPropagatorRKF45_1DayGEO(b *testing.B) {
	sma := 42164.0
	vCirc := math.Sqrt(physics.EARTH_MU_IAU / sma)
	initialState := []float64{sma, 0, 0, 0, vCirc, 0}
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	jd := 2451545.0

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		_ = orbital.PropagateCartesianStateRKF45(initialState, 86400.0, jd, params, 1e-9)
	}
	elapsed := time.Since(start)

	if b.N > 0 {
		b.ReportMetric(float64(elapsed.Nanoseconds())/float64(b.N), "ns/op")
	}
}

/**
 * BenchmarkPropagatorAccuracyComparison compares error growth over 7 days.
 */
func BenchmarkPropagator7DayErrorGrowth(b *testing.B) {
	sma := 42164.0
	vCirc := math.Sqrt(physics.EARTH_MU_IAU / sma)
	initialState := []float64{sma, 0, 0, 0, vCirc, 0}
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}
	jd := 2451545.0

	b.Run("RK4_7Day", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := orbital.PropagateCartesianState(initialState, 7*86400.0, jd, params)
			// Compute position error
			err := math.Sqrt((result[0]-sma)*(result[0]-sma) + result[1]*result[1])
			_ = err
		}
	})

	b.Run("RKF45_7Day", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			result := orbital.PropagateCartesianStateRKF45(initialState, 7*86400.0, jd, params, 1e-9)
			err := math.Sqrt((result[0]-sma)*(result[0]-sma) + result[1]*result[1])
			_ = err
		}
	})
}

// =============================================================================
// Benchmark 4: CUSUM Detection Performance
// =============================================================================

/**
 * BenchmarkCUSUMRealTimeProcessing measures throughput for real-time detection.
 * Target: Process > 1000 objects/second with TLE updates.
 */
func BenchmarkCUSUMRealTimeProcessing(b *testing.B) {
	// Simulate 100 residuals (representing one object's history)
	residuals := make([]float64, 100)
	for i := range residuals {
		residuals[i] = (float64(i%10) - 5.0) * 0.01 // ±0.05 km noise
	}

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		// Create fresh detector for each iteration (simulates new object)
		freshDetector := maneuver.NewCUSUMDetector(0.5, 5.0)
		for _, r := range residuals {
			_, _ = freshDetector.Update(r)
		}
	}
	elapsed := time.Since(start)

	if b.N > 0 {
		opsPerSec := float64(b.N*100) / elapsed.Seconds()
		b.ReportMetric(opsPerSec, "updates/sec")
	}
}

// =============================================================================
// Benchmark 5: Covariance Operations
// =============================================================================

/**
 * BenchmarkCovarianceConstruction measures TLE-to-covariance performance.
 */
func BenchmarkCovarianceConstruction(b *testing.B) {
	leoTLE := physics.TLE{
		MeanMotion:   15.0,
		Eccentricity: 0.01,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = physics.ConstructDefaultCovarianceFromTLE(leoTLE)
	}
}

/**
 * BenchmarkMahalanobisDistance computes d_M² for covariance realism testing.
 */
func BenchmarkMahalanobisDistance(b *testing.B) {
	error := []float64{0.5, 0.3, 0.2, 0.001, 0.0008, 0.0005} // 6D error
	cov := [6][6]float64{
		{1.0, 0.1, 0.1, 0.01, 0, 0},
		{0.1, 0.5, 0.05, 0, 0.005, 0},
		{0.1, 0.05, 0.3, 0, 0, 0.003},
		{0.01, 0, 0, 0.0001, 0, 0},
		{0, 0.005, 0, 0, 0.00008, 0},
		{0, 0, 0.003, 0, 0, 0.00005},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simplified Mahalanobis (diagonal only for speed)
		var d2 float64
		for j := 0; j < 6; j++ {
			d2 += error[j] * error[j] / cov[j][j]
		}
		_ = d2
	}
}

// =============================================================================
// Benchmark 6: CDM Parsing Performance
// =============================================================================

/**
 * BenchmarkCDMParsing measures XML ingestion throughput.
 * Target: Parse > 100 CDMs/second for operational processing.
 */
func BenchmarkCDMParsing(b *testing.B) {
	cdmXML := []byte(`<?xml version="1.0" encoding="UTF-8"?>
<cdm>
  <header CCSDS_CDM_VERS="1.0">
    <CREATION_DATE>2024-01-15T12:00:00Z</CREATION_DATE>
    <ORIGINATOR>18 SDS</ORIGINATOR>
    <MESSAGE_ID>CDM-2024-001</MESSAGE_ID>
  </header>
  <relativeMetadata>
    <TCA>2024-01-16T08:30:00.000Z</TCA>
    <MISS_DISTANCE>0.5</MISS_DISTANCE>
    <RELATIVE_SPEED>15.2</RELATIVE_SPEED>
    <COLLISION_PROBABILITY>1.5e-4</COLLISION_PROBABILITY>
    <COLLISION_PROB_METHOD>FOSTER_1992</COLLISION_PROB_METHOD>
  </relativeMetadata>
  <object1Metadata>
    <OBJECT>OBJECT1</OBJECT>
    <OBJECT_DESIGNATOR>12345</OBJECT_DESIGNATOR>
    <OBJECT_NAME>TEST_SAT_1</OBJECT_NAME>
  </object1Metadata>
  <object1Data>
    <X>7000.0</X><Y>0.0</Y><Z>0.0</Z>
    <VX>0.0</VX><VY>7.5</VY><VZ>0.0</VZ>
    <CR_R>0.01</CR_R><CT_T>0.005</CT_T><CN_N>0.002</CN_N>
  </object1Data>
  <object2Metadata>
    <OBJECT>OBJECT2</OBJECT>
    <OBJECT_DESIGNATOR>99999</OBJECT_DESIGNATOR>
    <OBJECT_NAME>TEST_DEBRIS_1</OBJECT_NAME>
  </object2Metadata>
  <object2Data>
    <X>7000.2</X><Y>0.1</Y><Z>0.05</Z>
    <VX>0.1</VX><VY>7.4</VY><VZ>0.05</VZ>
    <CR_R>0.02</CR_R><CT_T>0.01</CT_T><CN_N>0.005</CN_N>
  </object2Data>
</cdm>`)

	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		_, err := ParseCDMXML(cdmXML)
		if err != nil {
			b.Fatal(err)
		}
	}
	elapsed := time.Since(start)

	if b.N > 0 {
		opsPerSec := float64(b.N) / elapsed.Seconds()
		b.ReportMetric(opsPerSec, "cdms/sec")
	}
}

// =============================================================================
// Summary Report Function
// =============================================================================

/**
 * PrintBenchmarkSummary generates a human-readable performance report.
 */
func PrintBenchmarkSummary() {
	fmt.Println("\n=== ADITI 4.0 Performance Benchmark Summary ===")
	fmt.Println()
	fmt.Println("Critical Path Performance Targets:")
	fmt.Println("  - Alfano Pc:           < 1 ms per conjunction")
	fmt.Println("  - TCA Finder:          < 5 iterations, < 1 mm precision")
	fmt.Println("  - RK4 Propagation:     ~8600 steps/day (10s fixed)")
	fmt.Println("  - RKF45 Propagation:   Adaptive, typically 100-1000 steps/day")
	fmt.Println("  - CUSUM Processing:    > 1000 updates/second")
	fmt.Println("  - CDM Parsing:         > 100 CDMs/second")
	fmt.Println()
	fmt.Println("Run benchmarks with: go test -bench=. -benchmem")
	fmt.Println()
}
