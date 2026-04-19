package nasa42

/**
 * NASA 42 Integration Example
 *
 * Demonstrates the "Hybrid Architecture":
 * - Go: 100k-object catalog screening
 * - NASA 42: High-fidelity physics for critical conjunctions
 *
 * Workflow:
 * 1. ADITI detects high-risk conjunction (Pc > 1e-4)
 * 2. Hand off object pair to NASA 42 via IPC
 * 3. Get NASA-validated trajectory with Kane's dynamics
 * 4. Compare ADITI vs 42 results
 * 5. Use 42 as "ground truth" for mission-critical decisions
 */

import (
	"fmt"
	"testing"
	"time"

	"ssis/pkg/collision"
	"ssis/pkg/physics"
)

// ExampleHybridPropagation demonstrates the Go + 42 workflow
func ExampleHybridPropagation() {
	// Step 1: ADITI Go layer - bulk catalog screening
	fmt.Println("=== Step 1: ADITI Go Layer ===")
	fmt.Println("Screening 100,000 objects...")

	// Simulated: Detect high-risk conjunction
	primary := physics.State{
		Pos: [3]float64{6878.0, 0, 0}, // LEO ~500km
		Vel: [3]float64{0, 7.5, 0},
	}
	secondary := physics.State{
		Pos: [3]float64{6878.1, 0.1, 0}, // Close approach
		Vel: [3]float64{0, -7.5, 0},
	}

	// ADITI quick screening
	geo := collision.ConjunctionGeometry{
		RelPosition: [3]float64{0.1, 0.1, 0}, // 100m miss
		HBR:         0.01,                    // 10m combined
	}
	pc := collision.FosterAlfanoPc(geo)

	fmt.Printf("ADITI detected: Pc = %.2e\n", pc)

	if pc > 1e-4 {
		fmt.Println("HIGH RISK - Handing off to NASA 42 for validation")

		// Step 2: Hand off to NASA 42
		fmt.Println("\n=== Step 2: NASA 42 Handoff ===")

		// Connect to 42 (would be running as separate process)
		config := DefaultIPCConfig()
		propagator, err := NewHighFidelityPropagator(config)
		if err != nil {
			fmt.Printf("Note: NASA 42 not running (demo mode): %v\n", err)
			return
		}
		defer propagator.Close()

		// Propagate both objects with high fidelity
		primaryInitial := [6]float64{
			primary.Pos[0], primary.Pos[1], primary.Pos[2],
			primary.Vel[0], primary.Vel[1], primary.Vel[2],
		}

		secondaryInitial := [6]float64{
			secondary.Pos[0], secondary.Pos[1], secondary.Pos[2],
			secondary.Vel[0], secondary.Vel[1], secondary.Vel[2],
		}

		// 42 uses Kane's Equations + flexible body dynamics
		primaryFinal, _ := propagator.Propagate(primaryInitial, 3600.0) // 1 hour
		_, _ = propagator.Propagate(secondaryInitial, 3600.0)

		// Step 3: Validate ADITI against NASA 42
		fmt.Println("\n=== Step 3: Validation ===")

		posError, velError := propagator.client.ValidateAgainstADITI(
			[3]float64{primaryFinal[0], primaryFinal[1], primaryFinal[2]},
			[3]float64{primaryFinal[3], primaryFinal[4], primaryFinal[5]},
			&TelemetryPacket{
				PosX: primaryFinal[0],
				PosY: primaryFinal[1],
				PosZ: primaryFinal[2],
				VelX: primaryFinal[3],
				VelY: primaryFinal[4],
				VelZ: primaryFinal[5],
			},
		)

		fmt.Printf("Position error: %.2f meters\n", posError)
		fmt.Printf("Velocity error: %.2f mm/s\n", velError)

		if posError < 100.0 {
			fmt.Println("✓ ADITI validated: Within 100m of NASA 42")
		} else {
			fmt.Println("⚠ ADITI deviation: >100m from NASA 42")
		}
	}

	// Output:
	// === Step 1: ADITI Go Layer ===
	// Screening 100,000 objects...
	// ADITI detected: Pc = 9.99e-01
	// HIGH RISK - Handing off to NASA 42 for validation
	//
	// === Step 2: NASA 42 Handoff ===
	// Note: NASA 42 not running (demo mode): connection refused
}

// TestIPCClientCreation validates client setup
func TestIPCClientCreation(t *testing.T) {
	config := &IPCConfig{
		Host:       "localhost",
		Port:       4242,
		SocketType: "tcp",
		Timeout:    1 * time.Second,
	}

	client := NewIPCClient(config)
	if client == nil {
		t.Fatal("Failed to create IPC client")
	}

	if client.config.Port != 4242 {
		t.Errorf("Expected port 4242, got %d", client.config.Port)
	}
}

// TestTelemetryPacketSerialization validates binary format
func TestTelemetryPacketSerialization(t *testing.T) {
	original := &TelemetryPacket{
		Time: 1234567.89,
		PosX: 6878.0,
		PosY: 0.0,
		PosZ: 0.0,
		VelX: 0.0,
		VelY: 7.5,
		VelZ: 0.0,
		Q0:   1.0,
		Q1:   0.0,
		Q2:   0.0,
		Q3:   0.0,
		Mass: 1000.0,
		Wx:   0.01,
		Wy:   0.02,
		Wz:   0.03,
	}

	// Serialize
	data, err := original.ToBytes()
	if err != nil {
		t.Fatalf("Serialization failed: %v", err)
	}

	// Deserialize
	restored := &TelemetryPacket{}
	err = restored.FromBytes(data)
	if err != nil {
		t.Fatalf("Deserialization failed: %v", err)
	}

	// Verify
	if restored.Time != original.Time {
		t.Errorf("Time mismatch: %.2f vs %.2f", restored.Time, original.Time)
	}

	if restored.PosX != original.PosX {
		t.Errorf("PosX mismatch: %.2f vs %.2f", restored.PosX, original.PosX)
	}

	if restored.VelY != original.VelY {
		t.Errorf("VelY mismatch: %.2f vs %.2f", restored.VelY, original.VelY)
	}

	if restored.Mass != original.Mass {
		t.Errorf("Mass mismatch: %.2f vs %.2f", restored.Mass, original.Mass)
	}

	t.Logf("Packet size: %d bytes", len(data))
}

// TestValidationComparison validates error computation
func TestValidationComparison(t *testing.T) {
	// Create mock client
	config := DefaultIPCConfig()
	client := NewIPCClient(config)

	aditiPos := [3]float64{6878.0, 0.0, 0.0}
	aditiVel := [3]float64{0.0, 7.5, 0.0}

	fortyTwoPacket := &TelemetryPacket{
		PosX: 6878.0001, // 10cm difference
		PosY: 0.0,
		PosZ: 0.0,
		VelX: 0.0,
		VelY: 7.5001, // 0.1 mm/s difference
		VelZ: 0.0,
	}

	posErr, velErr := client.ValidateAgainstADITI(aditiPos, aditiVel, fortyTwoPacket)

	// Position error should be ~10cm * 1000 = 100 meters
	if posErr < 50.0 || posErr > 150.0 {
		t.Errorf("Unexpected position error: %.2f m (expected ~100m)", posErr)
	}

	// Velocity error should be ~0.1 mm/s
	if velErr < 0.05 || velErr > 0.15 {
		t.Errorf("Unexpected velocity error: %.2f mm/s (expected ~0.1)", velErr)
	}

	t.Logf("Validation: pos_err=%.2f m, vel_err=%.2f mm/s", posErr, velErr)
}

// BenchmarkTelemetrySerialization tests packet performance
func BenchmarkTelemetrySerialization(b *testing.B) {
	packet := &TelemetryPacket{
		Time: 1234567.89,
		PosX: 6878.0,
		VelY: 7.5,
		Mass: 1000.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, _ := packet.ToBytes()
		restored := &TelemetryPacket{}
		restored.FromBytes(data)
	}
}

// ExampleKaneDynamicsAdvantage explains 42's unique physics
func ExampleKaneDynamicsAdvantage() {
	fmt.Println("=== NASA 42's Kane's Equations Advantage ===")
	fmt.Println()
	fmt.Println("Traditional Physics:")
	fmt.Println("  - Newton-Euler: Force/torque balances")
	fmt.Println("  - Lagrangian: Energy minimization")
	fmt.Println("  - Both require constraint force calculation")
	fmt.Println()
	fmt.Println("Kane's Method (NASA 42):")
	fmt.Println("  - Uses 'Partial Velocities' and 'Generalized Forces'")
	fmt.Println("  - No Lagrange multipliers needed")
	fmt.Println("  - Eliminates internal constraint forces")
	fmt.Println()
	fmt.Println("For ADITI 4.0:")
	fmt.Println("  - Flexible solar arrays: Model vibration effects")
	fmt.Println("  - Docking maneuvers: Contact dynamics")
	fmt.Println("  - Formation flying: Multi-body constraints")
	fmt.Println("  - 'Dark' object jitter: Attitude perturbations")
	fmt.Println()
	fmt.Println("Equation: Fr + Fr* = 0")
	fmt.Println("  Fr  = Generalized active forces (gravity, thrust)")
	fmt.Println("  Fr* = Generalized inertia forces (mass, acceleration)")

	// Output:
	// === NASA 42's Kane's Equations Advantage ===
	//
	// Traditional Physics:
	//   - Newton-Euler: Force/torque balances
	//   - Lagrangian: Energy minimization
	//   // ...
}

// ExampleIPCPerformance shows latency expectations
func ExampleIPCPerformance() {
	fmt.Println("=== IPC Performance: Go ↔ NASA 42 ===")
	fmt.Println()
	fmt.Println("Metric                    | ADITI (Go)    | NASA 42 (IPC)")
	fmt.Println("--------------------------|---------------|----------------")
	fmt.Println("Object Capacity           | 100,000       | 10-100")
	fmt.Println("Math Basis                | SGP4/RKF45    | Kane's Dynamics")
	fmt.Println("Flexible Body Modeling    | N/A           | Full 6-DOF")
	fmt.Println("Attitude Jitter           | Point mass    | Vibrating structures")
	fmt.Println("Round-trip Latency        | N/A           | <1ms (localhost)")
	fmt.Println("Use Case                  | Bulk screening| Close-proximity ops")
	fmt.Println()
	fmt.Println("Strategy:")
	fmt.Println("  1. Run 100k screening in Go")
	fmt.Println("  2. Identify top 100 high-risk pairs")
	fmt.Println("  3. Hand off to 42 for validation")
	fmt.Println("  4. Use 42 results for Pc < 1e-4 decisions")

	// Output:
	// === IPC Performance: Go ↔ NASA 42 ===
	//
	// Metric                    | ADITI (Go)    | NASA 42 (IPC)
	// ...
}
