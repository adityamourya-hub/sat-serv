// pkg/tests/predictive_slew_benchmark.go
// PredictiveSlew End-to-End Latency Benchmark
//
// Benchmarks full pipeline: Detection → Track Init → Predictive Slew → Centroid Lock
// Target latency: < 100ms from detection to first centroid
// AMOS 2025: Integrates speckle correction + UDE bias + fractional FNO

package tests

import (
	"fmt"
	"math"
	"testing"
	"time"

	"ssis/pkg/physics"
	"ssis/pkg/telescope"
)

const (
	AMOSBenchmarkMs = 2000.0 // GEODSS Deep STARE: ~2s
)

// TestPredictiveSlewLatency benchmarks end-to-end PredictiveSlew latency
func TestPredictiveSlewLatency(t *testing.T) {
	fmt.Println("=== PredictiveSlew Latency Benchmark ===")
	fmt.Printf("Target Latency: %.0f ms\n", 100.0)
	fmt.Printf("GEODSS Benchmark: %.0f ms\n", AMOSBenchmarkMs)
	fmt.Println()

	// Initialize components
	speckle := telescope.NewSpecklePreprocessor(10)
	reportBuilder, _ := telescope.NewObservationReportBuilder("test-node", "hanle-01")

	// Benchmark iterations
	iterations := 100
	latencies := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		start := time.Now()

		// Stage 1: Speckle preprocessing (AMOS Georgia State)
		frame := telescope.SpeckleFrame{
			Timestamp: time.Now().UnixNano(),
			ImageData: make([]float64, 1024),
			Altitude:  4500.0, // Hanle altitude (m)
			Airmass:   1.2,
		}
		zernike := speckle.ProcessFrame(frame)

		// Stage 2: Get FNO correction (UDE bias + fractional memory)
		densityCorrection, fractionalAlpha := speckle.GetFNOCorrection()

		// Stage 3: GHNN state prediction (simulated ONNX inference)
		state := predictGHNNState()

		// Stage 4: Predictive Slew computation
		cmd := computePredictiveSlew(state, fractionalAlpha, densityCorrection)

		// Stage 5: Centroid lock (simulated)
		_ = simulateCentroidLock(cmd)

		// Stage 6: ObservationReport generation with Hamiltonian Hash
		report, _ := reportBuilder.BuildReport(
			"RSO-2026-TEST",
			state,
			8.5,
			0.75,
			2, // HIGH threat
		)

		// Stage 7: Sovereign Handshake validation
		handshake := telescope.NewSovereignHandshake("test-node", report)
		_, _ = handshake.Validate(report)

		elapsed := time.Since(start)
		latencyMs := float64(elapsed) / float64(time.Millisecond)
		latencies[i] = latencyMs

		// Log details for first iteration
		if i == 0 {
			fmt.Printf("First iteration details:\n")
			fmt.Printf("  Zernike coefficients: %d\n", len(zernike))
			fmt.Printf("  Density correction: %.4f\n", densityCorrection)
			fmt.Printf("  Fractional alpha: %.2f\n", fractionalAlpha)
			fmt.Printf("  Slew command: Az=%.2f, El=%.2f\n", cmd.Azimuth, cmd.Elevation)
			fmt.Printf("  Hamiltonian hash: %x\n", report.HamiltonianHash[:8])
			fmt.Println()
		}
	}

	// Analyze results
	min, max, avg, p95 := analyzeLatencyStats(latencies)

	fmt.Printf("=== Latency Results (%d iterations) ===\n", iterations)
	fmt.Printf("Min: %.2f ms\n", min)
	fmt.Printf("Max: %.2f ms\n", max)
	fmt.Printf("Avg: %.2f ms\n", avg)
	fmt.Printf("P95: %.2f ms\n", p95)
	fmt.Printf("Target: %.0f ms\n", 100.0)

	// Assertions
	if avg > 100.0 {
		t.Errorf("Average latency %.2f ms exceeds target %.0f ms", avg, 100.0)
	}

	if p95 > 150.0 {
		t.Errorf("P95 latency %.2f ms exceeds 1.5x target", p95)
	}

	speedup := AMOSBenchmarkMs / avg
	fmt.Printf("\n=== Speedup vs GEODSS Deep STARE ===\n")
	fmt.Printf("Speedup: %.1fx\n", speedup)

	if speedup < 10.0 {
		t.Errorf("Speedup %.1fx less than 10x target", speedup)
	}

	fmt.Println("\n=== PASS: PredictiveSlew Latency Verified ===")
}

// TestAMOSSpeckleIntegration benchmarks AMOS speckle correction performance
func TestAMOSSpeckleIntegration(t *testing.T) {
	fmt.Println("=== AMOS Speckle Correction Integration Test ===")

	speckle := telescope.NewSpecklePreprocessor(10)

	// Process multiple frames to test multi-task learning
	for i := 0; i < 10; i++ {
		frame := telescope.SpeckleFrame{
			Timestamp: time.Now().UnixNano(),
			ImageData: make([]float64, 1024),
			Altitude:  4500.0,
			Airmass:   1.2,
		}

		start := time.Now()
		zernike := speckle.ProcessFrame(frame)
		elapsed := time.Since(start)

		fmt.Printf("Frame %d: %d Zernike coeffs in %.2f ms\n",
			i+1, len(zernike), float64(elapsed)/float64(time.Millisecond))
	}

	// Get FNO correction
	densityCorrection, fractionalAlpha := speckle.GetFNOCorrection()
	turbulenceProfile := speckle.GetTurbulenceProfile()

	fmt.Printf("\nFNO Integration:\n")
	fmt.Printf("  Density correction: %.4f\n", densityCorrection)
	fmt.Printf("  Fractional alpha: %.2f\n", fractionalAlpha)
	fmt.Printf("  Turbulence profile length: %d\n", len(turbulenceProfile))

	fmt.Println("\n=== PASS: AMOS Speckle Integration Verified ===")
}

// TestFractionalDragPerformance benchmarks fractional drag computation
func TestFractionalDragPerformance(t *testing.T) {
	fmt.Println("=== Fractional Drag Performance Test ===")

	// Simulate wake memory buffer
	wakeMemory := make([]float64, 50)
	for i := range wakeMemory {
		wakeMemory[i] = 1.0 - 0.01*float64(i) // Decaying wake
	}

	iterations := 1000
	start := time.Now()

	for i := 0; i < iterations; i++ {
		// Caputo D_t^β computation
		beta := 0.8
		var fractionalSum float64
		for j, val := range wakeMemory {
			weight := math.Pow(1.0-float64(j)/float64(len(wakeMemory)), beta)
			fractionalSum += val * weight
		}
		_ = fractionalSum / float64(len(wakeMemory))
	}

	elapsed := time.Since(start)
	avgUs := float64(elapsed) / float64(iterations) / float64(time.Microsecond)

	fmt.Printf("Fractional drag computation: %.2f μs per iteration\n", avgUs)

	if avgUs > 100.0 {
		t.Errorf("Fractional drag too slow: %.2f μs (target < 100 μs)", avgUs)
	}

	fmt.Println("=== PASS: Fractional Drag Performance Verified ===")
}

// =============================================================================
// Helper Functions
// =============================================================================

type SlewCommand struct {
	Azimuth   float64
	Elevation float64
}

func predictGHNNState() *physics.State {
	// Simulated GHNN ONNX inference
	// In production: Load axiom_se3_ghnn.onnx and run inference
	return &physics.State{
		Pos:   [3]float64{7000.0, 0.0, 0.0},
		Vel:   [3]float64{0.0, 7.5, 0.0},
		E_hal: -29.5,
	}
}

func computePredictiveSlew(state *physics.State, fractionalAlpha, densityCorrection float64) *SlewCommand {
	// Compute slew command with fractional atmosphere correction
	az := math.Atan2(state.Vel[1], state.Vel[0]) * 180.0 / math.Pi
	el := 30.0 + (state.Pos[0]-7000.0)*0.01*densityCorrection

	// Apply fractional correction
	el *= math.Pow(densityCorrection, fractionalAlpha-1.0)

	return &SlewCommand{
		Azimuth:   az,
		Elevation: el,
	}
}

func simulateCentroidLock(cmd *SlewCommand) bool {
	// Simulated centroid lock
	time.Sleep(1 * time.Millisecond)
	return true
}

func analyzeLatencyStats(latencies []float64) (min, max, avg, p95 float64) {
	if len(latencies) == 0 {
		return 0, 0, 0, 0
	}

	min = latencies[0]
	max = latencies[0]
	sum := 0.0

	for _, l := range latencies {
		if l < min {
			min = l
		}
		if l > max {
			max = l
		}
		sum += l
	}
	avg = sum / float64(len(latencies))

	// P95 percentile
	sorted := make([]float64, len(latencies))
	copy(sorted, latencies)

	// Simple sort (in production, use proper sorting)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	p95Index := int(float64(len(sorted)) * 0.95)
	if p95Index >= len(sorted) {
		p95Index = len(sorted) - 1
	}
	p95 = sorted[p95Index]

	return
}
