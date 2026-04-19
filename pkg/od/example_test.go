package od

/**
 * OD Integration Example: From TLE to Covariance to Pc
 *
 * This example demonstrates the complete pipeline:
 * 1. Start with TLE (no covariance)
 * 2. Collect sensor observations
 * 3. Run Batch LS OD to generate state + covariance
 * 4. Use covariance for Pc computation in conjunction analysis
 *
 * This addresses the critical gap: TLEs lack covariance,
 * but OD produces it from your own sensor data.
 */

import (
	"fmt"
	"math"
	"time"

	"ssis/pkg/collision"
	"ssis/pkg/physics"
)

// Example_ODtoPc demonstrates the complete TLE → OD → Pc pipeline
func Example_ODtoPc() {
	// Step 1: Start with TLE (no covariance available)
	tle := physics.TLE{
		Name:            "TEST-SAT",
		SatelliteNumber: 12345,
		EpochYear:       24,
		EpochDay:        100.5, // Day 100.5 of 2024
		MeanMotion:      15.0,  // ~LEO
		Inclination:     53.0,
		RAAN:            10.0,
		Eccentricity:    0.01,
		ArgOfPerigee:    20.0,
		MeanAnomaly:     30.0,
	}

	fmt.Println("Step 1: Initial TLE (no covariance)")
	fmt.Printf("  Satellite: %s\n", tle.Name)
	fmt.Printf("  Epoch: %02d/%06.3f\n", tle.EpochYear, tle.EpochDay)
	fmt.Printf("  Mean Motion: %.3f rev/day\n", tle.MeanMotion)
	fmt.Println()

	// Step 2: Simulate sensor observations (your own radar/optical data)
	fmt.Println("Step 2: Collect sensor observations")

	// Create ground station network (NETRA + IAF optical)
	netra := NewNETRARadar()
	iafOptical := NewIAFOptical()

	// Simulate observations over 2 hours
	var observations []Observation
	startTime := time.Now()

	for i := 0; i < 12; i++ { // 12 observations, 10 min intervals
		obsTime := startTime.Add(time.Duration(i*10) * time.Minute)

		// Alternate between radar and optical
		if i%2 == 0 {
			// Radar range observation
			observations = append(observations, Observation{
				Type:     ObsRange,
				Value:    800.0 + float64(i)*0.5, // Synthetic range in km
				Sigma:    0.01,                   // 10m precision
				Time:     obsTime,
				Station:  netra,
				TargetID: tle.Name,
			})
		} else {
			// Optical angle observation
			observations = append(observations, Observation{
				Type:     ObsElevation,
				Value:    45.0 + float64(i)*0.1, // degrees
				Sigma:    0.001,                 // ~3.6 arcsec
				Time:     obsTime,
				Station:  iafOptical,
				TargetID: tle.Name,
			})
		}
	}

	fmt.Printf("  Collected %d observations\n", len(observations))
	fmt.Printf("  Radar: %d (10m precision)\n", len(observations)/2)
	fmt.Printf("  Optical: %d (3.6 arcsec precision)\n", len(observations)/2)
	fmt.Println()

	// Step 3: Run Batch Least Squares OD
	fmt.Println("Step 3: Run Batch Least Squares OD")

	batch := NewObservationBatch(observations)
	estimator := NewBatchLSEstimator()
	params := physics.PhysicsParams{Mass: 1000, Cd: 2.2, Cr: 1.0}

	result, err := estimator.EstimateFromTLE(batch, tle, params)
	if err != nil {
		fmt.Printf("  Note: %v\n", err)
	}

	fmt.Printf("  OD Converged: %v\n", result.Converged)
	fmt.Printf("  Iterations: %d\n", result.Iterations)
	fmt.Printf("  Residual RMS: %.3f km\n", result.ResidualRMS)
	fmt.Printf("  State: pos=[%.1f, %.1f, %.1f] km, vel=[%.3f, %.3f, %.3f] km/s\n",
		result.State.Pos[0], result.State.Pos[1], result.State.Pos[2],
		result.State.Vel[0], result.State.Vel[1], result.State.Vel[2])
	fmt.Println()

	// Step 4: Extract covariance matrix (6x6)
	fmt.Println("Step 4: Extract 6x6 covariance matrix")
	fmt.Println("  Position covariance (3x3, km²):")
	for i := 0; i < 3; i++ {
		fmt.Printf("    [%.4e, %.4e, %.4e]\n",
			result.Covariance[i][0], result.Covariance[i][1], result.Covariance[i][2])
	}

	// Position uncertainty (RSS)
	posUncertainty := math.Sqrt(
		result.Covariance[0][0] +
			result.Covariance[1][1] +
			result.Covariance[2][2])
	fmt.Printf("  Position uncertainty (RSS): %.3f km\n", posUncertainty)
	fmt.Println()

	// Step 5: Use covariance for Pc computation
	fmt.Println("Step 5: Pc computation using OD-generated covariance")

	// Extract 3x3 position covariance from 6x6 state covariance
	var posCov3D [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			posCov3D[i][j] = result.Covariance[i][j]
		}
	}

	// Create collision geometry
	geo := collision.ConjunctionGeometry{
		RelPosition: [3]float64{0.1, 0, 0}, // 100m miss distance
		RelVelocity: [3]float64{10, 0, 0},
		HBR:         0.015, // 15m combined radius
		TCA:         result.EpochJD + 1.0,
		// Primary covariance from OD (position only)
		PrimaryCov: posCov3D,
		// Secondary covariance (another object, using synthetic 3x3)
		SecondaryCov: [3][3]float64{
			{0.01, 0, 0},
			{0, 0.01, 0},
			{0, 0, 0.01},
		},
	}

	// Compute Pc using Alfano method
	pc := collision.FosterAlfanoPc(geo)
	fmt.Printf("  Collision Probability (Pc): %.4e\n", pc)

	if pc > 1e-4 {
		fmt.Println("  Risk Level: HIGH (>1e-4)")
	} else if pc > 1e-6 {
		fmt.Println("  Risk Level: MEDIUM (>1e-6)")
	} else {
		fmt.Println("  Risk Level: LOW (<1e-6)")
	}
	fmt.Println()

	// Summary
	fmt.Println("=== Summary ===")
	fmt.Println("✓ TLE input (no covariance)")
	fmt.Println("✓ Sensor observations (radar + optical)")
	fmt.Println("✓ Batch LS OD → State + 6x6 covariance")
	fmt.Println("✓ Pc computation using Alfano 2005")
	fmt.Println()
	fmt.Println("This pipeline provides indigenous OD capability")
	fmt.Println("for sovereign SSA without relying on external catalog")
	fmt.Println("covariance data (which is rarely shared).")
}
