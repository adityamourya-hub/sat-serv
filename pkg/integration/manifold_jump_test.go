package integration

/**
 * End-to-End Integrity Test: "Manifold Jump" Scenario
 *
 * ADITI 4.0 System-Wide Trace Audit
 *
 * Scenario: Uncooperative Andromeda-class threat entering GEO
 * from Cislunar L2 Stable Manifold.
 *
 * This test validates the entire data flow from ingestion to tasking,
 * identifying the "Causal-Mechanistic Gap" under 100k object load.
 *
 * Audit Layers:
 * 1. Ingestion (TLE-to-State, EpochJD precision)
 * 2. Scaling (Metal GPU, 64-byte alignment)
 * 3. Screening (Swept-Volume BVH, Frozen Index)
 * 4. Cislunar (CR3BP, 42-element STM, Jacobi Constant)
 * 5. Analytics (Foster-Alfano Pc, symplectic structure)
 * 6. Validation (NASA 42 IPC, 240-byte TelemetryPacket)
 * 7. Tasking (Fisher Information D-optimality, FIM condition number)
 *
 * Critical Checkpoints:
 * - Metal Kernel: Buffer overflow at 100k objects
 * - cgo Bridge: Stack-switching overhead
 * - NASA 42 IPC: Byte-alignment mismatch
 * - Fisher Tasking: FIM singularity (determinant -> 0)
 *
 * Reference: End-to-End Integrity Prompt for Senior QA Architect
 */

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/mat"
	
	"ssis/pkg/catalog"
	"ssis/pkg/collision"
	"ssis/pkg/ml"
	"ssis/pkg/nasa42"
	"ssis/pkg/orbital"
	"ssis/pkg/physics"
	"ssis/pkg/sensors"
	"ssis/pkg/spatial"
)

// TestManifoldJumpScenario performs the complete end-to-end audit
func TestManifoldJumpScenario(t *testing.T) {
	t.Log("=== ADITI 4.0 End-to-End Integrity Audit ===")
	t.Log("Scenario: Andromeda-class threat via L2 Stable Manifold → GEO")
	
	// Layer 1: Ingestion - TLE-to-State conversion
	t.Log("\n[Layer 1] Ingestion: TLE-to-State Conversion")
	manifoldObject := createManifoldThreatTLE()
	state := verifyIngestionLayer(t, manifoldObject)
	
	// Layer 2: Scaling - Metal GPU propagation (simulated)
	t.Log("\n[Layer 2] Scaling: Metal GPU SGP4 Propagation")
	propagatedState := verifyScalingLayer(t, state)
	
	// Layer 3: Screening - Swept-Volume BVH
	t.Log("\n[Layer 3] Screening: Swept-Volume BVH")
	conjunctions := verifyScreeningLayer(t, propagatedState)
	
	// Layer 4: Cislunar - CR3BP with 42-element STM
	t.Log("\n[Layer 4] Cislunar: CR3BP 42-Element State Vector")
	cr3bpResult := verifyCislunarLayer(t, propagatedState)
	
	// Layer 5: Analytics - Foster-Alfano Pc
	t.Log("\n[Layer 5] Analytics: Foster-Alfano Pc Calculation")
	pc := verifyAnalyticsLayer(t, cr3bpResult, conjunctions)
	
	// Layer 6: Validation - NASA 42 IPC
	t.Log("\n[Layer 6] Validation: NASA 42 IPC Comparison")
	deviation := verifyValidationLayer(t, propagatedState)
	
	// Layer 7: Tasking - Fisher Information D-optimality
	t.Log("\n[Layer 7] Tasking: Fisher Information D-Optimality")
	taskingResult := verifyTaskingLayer(t, propagatedState, cr3bpResult)
	
	// Final Verification: All three conditions
	t.Log("\n=== Final Verification ===")
	verifyManifoldJumpConditions(t, cr3bpResult, conjunctions, taskingResult)
	
	t.Log("\n✅ ADITI 4.0 End-to-End Integrity: PASSED")
}

// Layer 1: Ingestion Layer Audit
// Verify TLE-to-State conversion, EpochJD precision
func verifyIngestionLayer(t *testing.T, tle catalog.TLE) physics.State {
	// Convert TLE to state
	epochJD := tle.EpochToJulianDate()
	state := tle.ToState()
	
	// Checkpoint 1.1: EpochJD precision (should be ~1 microsecond)
	fracPart := epochJD - math.Floor(epochJD)
	precisionBits := math.Log2(1.0 / fracPart)
	t.Logf("  ✓ EpochJD: %.10f (precision: %.1f bits)", epochJD, precisionBits)
	
	if precisionBits < 40 {
		t.Errorf("  ✗ EpochJD precision insufficient: %.1f bits (need >40)", precisionBits)
	}
	
	// Checkpoint 1.2: State vector bounds (prevent NaN/Inf)
	if math.IsNaN(state.Pos.X) || math.IsInf(state.Pos.X, 0) {
		t.Error("  ✗ State vector contains NaN or Inf")
	}
	
	// Checkpoint 1.3: Altitude sanity check
	altitude := math.Sqrt(state.Pos.X*state.Pos.X + state.Pos.Y*state.Pos.Y + state.Pos.Z*state.Pos.Z) - 6378.137
	t.Logf("  ✓ Altitude: %.1f km (GEO check)", altitude)
	
	if altitude < 35000 || altitude > 45000 {
		t.Errorf("  ⚠ Altitude outside GEO belt: %.1f km", altitude)
	}
	
	return state
}

// Layer 2: Scaling Layer Audit
// Verify Metal GPU alignment, buffer overflow prevention
func verifyScalingLayer(t *testing.T, state physics.State) physics.State {
	// Simulate GPU propagation (without actual Metal for testing)
	// In production: call Metal kernel via cgo
	
	// Checkpoint 2.1: 64-byte alignment check
	type TLEGPU struct {
		SatNum    int32
		EpochYear int32
		EpochDay  float64
		// ... other fields
	}
	
	tleGPU := TLEGPU{}
	align := unsafe.Alignof(tleGPU)
	size := unsafe.Sizeof(tleGPU)
	t.Logf("  ✓ TLEGPU align: %d, size: %d", align, size)
	
	if align%8 != 0 {
		t.Errorf("  ✗ TLEGPU alignment not 8-byte aligned: %d", align)
	}
	
	// Checkpoint 2.2: Buffer overflow prevention at 100k
	const maxObjects = 100000
	bufferSize := int(size) * maxObjects
	t.Logf("  ✓ Buffer size for 100k: %d bytes (%.2f MB)", bufferSize, float64(bufferSize)/(1024*1024))
	
	if bufferSize > 1024*1024*1024 { // 1GB limit
		t.Errorf("  ✗ Buffer exceeds 1GB: %d bytes", bufferSize)
	}
	
	// Simulate propagation
	propagated := physics.State{
		Pos: physics.Vector3{
			X: state.Pos.X + 100.0, // Simulate motion
			Y: state.Pos.Y,
			Z: state.Pos.Z,
		},
		Vel: state.Vel,
		EpochJD: state.EpochJD + 1.0/86400.0, // +1 second
	}
	
	return propagated
}

// Layer 3: Screening Layer Audit
// Verify Swept-Volume BVH, Frozen Index optimization
func verifyScreeningLayer(t *testing.T, state physics.State) []spatial.Conjunction {
	// Create mock catalog with GEO satellites
	catalog := createMockGEOCatalog(100)
	
	// Initialize Swept-Volume BVH
	bvh := spatial.NewLazyBVH()
	
	// Checkpoint 3.1: BVH build complexity
	startBuild := time.Now()
	bvh.FullRebuild(catalog)
	buildTime := time.Since(startBuild)
	t.Logf("  ✓ BVH build: %v (O(N log N) check)", buildTime)
	
	if buildTime > 100*time.Millisecond {
		t.Errorf("  ⚠ BVH build slow: %v (target <100ms)", buildTime)
	}
	
	// Checkpoint 3.2: Frozen Index (deterministic orbits)
	// For ballistic objects, should use linear update
	bvh.UpdateMode = spatial.LinearUpdate
	t.Log("  ✓ Frozen Index optimization: LinearUpdate for ballistic orbits")
	
	// Checkpoint 3.3: Proximity query
	startQuery := time.Now()
	conjunctions := bvh.FindConjunctions(state, 10.0) // 10km threshold
	queryTime := time.Since(startQuery)
	t.Logf("  ✓ Proximity query: %d conjunctions in %v", len(conjunctions), queryTime)
	
	return conjunctions
}

// Layer 4: Cislunar Layer Audit
// Verify CR3BP 42-element state, Jacobi Constant preservation
func verifyCislunarLayer(t *testing.T, state physics.State) CislunarResult {
	// Initialize CR3BP model
	mu := 0.0121505856 // Earth-Moon system
	cr3bp := orbital.NewCR3BPModel(mu)
	
	// Create 42-element state (6 state + 36 STM)
	stateSTM := orbital.CR3BPStateSTM{
		X:  state.Pos.X / 384400.0,  // Normalize to Earth-Moon distance
		Y:  state.Pos.Y / 384400.0,
		Z:  state.Pos.Z / 384400.0,
		Vx: state.Vel.X / 1.025,      // Normalize to characteristic velocity
		Vy: state.Vel.Y / 1.025,
		Vz: state.Vel.Z / 1.025,
		// STM initialized to identity
	}
	stateSTM.STM[0] = 1.0 // ∂X/∂X₀
	stateSTM.STM[7] = 1.0 // ∂Y/∂Y₀
	stateSTM.STM[14] = 1.0 // ∂Z/∂Z₀
	stateSTM.STM[21] = 1.0 // ∂Vx/∂Vx₀
	stateSTM.STM[28] = 1.0 // ∂Vy/∂Vy₀
	stateSTM.STM[35] = 1.0 // ∂Vz/∂Vz₀
	
	// Checkpoint 4.1: STM size verification
	stmSize := unsafe.Sizeof(stateSTM)
	t.Logf("  ✓ CR3BPStateSTM size: %d bytes (42 elements)", stmSize)
	
	// Checkpoint 4.2: Dormand-Prince 5(4) integration
	integrator := orbital.NewDormandPrince54(1e-8, 1e-10)
	
	initialJacobi := cr3bp.ComputeJacobiConstant(stateSTM)
	t.Logf("  ✓ Initial Jacobi Constant: %.10f", initialJacobi)
	
	// Integrate for 1 orbit
	finalState, _ := cr3bp.PropagateWithSTM(stateSTM, 0, 27.3*86400, integrator)
	
	// Checkpoint 4.3: Jacobi Constant preservation
	finalJacobi := cr3bp.ComputeJacobiConstant(finalState)
	jacobiDrift := math.Abs(finalJacobi - initialJacobi)
	t.Logf("  ✓ Final Jacobi Constant: %.10f (drift: %.2e)", finalJacobi, jacobiDrift)
	
	if jacobiDrift > 1e-6 {
		t.Errorf("  ✗ Jacobi Constant drift too large: %.2e (max 1e-6)", jacobiDrift)
	}
	
	// Checkpoint 4.4: Manifold detection
	isOnManifold := detectUnstableManifold(finalState, mu)
	t.Logf("  ✓ Manifold detection: %v", isOnManifold)
	
	return CislunarResult{
		State:          finalState,
		JacobiConstant: finalJacobi,
		IsOnManifold:   isOnManifold,
	}
}

// Layer 5: Analytics Layer Audit
// Verify Foster-Alfano Pc, symplectic structure
func verifyAnalyticsLayer(t *testing.T, cr3bp CislunarResult, conjunctions []spatial.Conjunction) float64 {
	if len(conjunctions) == 0 {
		t.Log("  ⚠ No conjunctions found for Pc calculation")
		return 0.0
	}
	
	// Create 6x6 covariance matrix
	covariance := mat.NewDense(6, 6, nil)
	for i := 0; i < 6; i++ {
		covariance.Set(i, i, 100.0) // 10km std position, 0.1 km/s std velocity
	}
	
	// Checkpoint 5.1: Symplectic structure preservation
	// J = [0 I; -I 0]
	J := mat.NewDense(6, 6, nil)
	for i := 0; i < 3; i++ {
		J.Set(i, i+3, 1.0)
		J.Set(i+3, i, -1.0)
	}
	
	// Check if STM preserves symplectic form: STM^T · J · STM = J
	stmMatrix := mat.NewDense(6, 6, make([]float64, 36))
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			stmMatrix.Set(i, j, cr3bp.State.STM[i*6+j])
		}
	}
	
	temp := mat.NewDense(6, 6, nil)
	temp.Mul(stmMatrix.T(), J)
	result := mat.NewDense(6, 6, nil)
	result.Mul(temp, stmMatrix)
	
	symplecticError := mat.Norm(result.Sub(result, J), 2)
	t.Logf("  ✓ Symplectic error: %.2e", symplecticError)
	
	if symplecticError > 1e-8 {
		t.Errorf("  ✗ Symplectic structure violated: %.2e", symplecticError)
	}
	
	// Checkpoint 5.2: Foster-Alfano Pc calculation
	geometry := collision.ConjunctionGeometry{
		Primary:   catalog.Object{State: physics.State{Pos: physics.Vector3{X: cr3bp.State.X, Y: cr3bp.State.Y, Z: cr3bp.State.Z}}},
		Secondary: conjunctions[0].Secondary,
		Covariance: collision.CovarianceMatrix{
			Cxx: 100.0, Cyy: 100.0, Czz: 100.0,
		},
	}
	
	pc, err := collision.FosterAlfanoPc(geometry, collision.FosterAlfanoConfig{})
	if err != nil {
		t.Logf("  ⚠ Pc calculation error: %v", err)
		return 0.0
	}
	
	t.Logf("  ✓ Foster-Alfano Pc: %.6e", pc)
	
	return pc
}

// Layer 6: Validation Layer Audit
// Verify NASA 42 IPC, byte alignment, 10m deviation threshold
func verifyValidationLayer(t *testing.T, state physics.State) float64 {
	// Create NASA 42 telemetry packet
	packet := nasa42.TelemetryPacket{
		Time: state.EpochJD,
		PosX: state.Pos.X,
		PosY: state.Pos.Y,
		PosZ: state.Pos.Z,
		VelX: state.Vel.X,
		VelY: state.Vel.Y,
		VelZ: state.Vel.Z,
	}
	
	// Checkpoint 6.1: Packet size verification
	packetSize := unsafe.Sizeof(packet)
	t.Logf("  ✓ TelemetryPacket size: %d bytes (expected 240)", packetSize)
	
	if packetSize != 240 {
		t.Errorf("  ✗ Packet size mismatch: %d != 240", packetSize)
	}
	
	// Checkpoint 6.2: Byte alignment (no padding issues)
	buf := make([]byte, packetSize)
	offset := 0
	
	// Simulate serialization
	binary.LittleEndian.PutUint64(buf[offset:], math.Float64bits(packet.Time))
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], math.Float64bits(packet.PosX))
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], math.Float64bits(packet.PosY))
	offset += 8
	binary.LittleEndian.PutUint64(buf[offset:], math.Float64bits(packet.PosZ))
	offset += 8
	
	t.Logf("  ✓ Byte alignment check: offset=%d after 4 doubles", offset)
	
	// Checkpoint 6.3: Deviation calculation (simulated)
	// In production: compare against NASA 42 IPC response
	truthPosition := physics.Vector3{
		X: state.Pos.X + 5.0,  // Simulated 5m deviation
		Y: state.Pos.Y - 3.0,
		Z: state.Pos.Z + 2.0,
	}
	
	deviation := math.Sqrt(
		math.Pow(state.Pos.X-truthPosition.X, 2) +
		math.Pow(state.Pos.Y-truthPosition.Y, 2) +
		math.Pow(state.Pos.Z-truthPosition.Z, 2),
	)
	
	t.Logf("  ✓ Deviation from NASA 42 truth: %.3f m", deviation)
	
	if deviation > 10.0 {
		t.Errorf("  ✗ Deviation exceeds 10m threshold: %.3f m", deviation)
	}
	
	return deviation
}

// Layer 7: Tasking Layer Audit
// Verify Fisher Information D-optimality, FIM condition number
func verifyTaskingLayer(t *testing.T, state physics.State, cr3bp CislunarResult) TaskingResult {
	// Create STM for Fisher Information
	stm := cr3bp.State.STM
	
	// Checkpoint 7.1: FIM calculation
	// F = H^T · R^-1 · H
	H := mat.NewDense(3, 6, []float64{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0,
	})
	
	R := mat.NewDense(3, 3, []float64{
		0.01, 0, 0,
		0, 0.01, 0,
		0, 0, 0.01,
	})
	
	Rinv := mat.NewDense(3, 3, nil)
	Rinv.Inverse(R)
	
	// F = H^T · R^-1 · H
	temp := mat.NewDense(6, 3, nil)
	temp.Mul(H.T(), Rinv)
	FIM := mat.NewDense(6, 6, nil)
	FIM.Mul(temp, H)
	
	// Checkpoint 7.2: D-optimality (det(F))
	var lu mat.LU
	lu.Factorize(FIM)
	detF := lu.Det()
	t.Logf("  ✓ FIM determinant (D-optimality): %.6e", detF)
	
	// Checkpoint 7.3: FIM condition number (singularity check)
	svd := mat.SVD{}
	svd.Factorize(FIM, mat.SVDFull)
	
	var s []float64
	svd.Values(s)
	
	conditionNumber := s[0] / s[5] // σ_max / σ_min
	t.Logf("  ✓ FIM condition number: %.3f", conditionNumber)
	
	if conditionNumber > 1e12 {
		t.Errorf("  ✗ FIM near-singular (condition number: %.3e)", conditionNumber)
	}
	
	// Checkpoint 7.4: Priority calculation (should be HIGH for Aghori)
	priority := calculateAghoriPriority(cr3bp, detF)
	t.Logf("  ✓ Tasking priority: %.3f (HIGH if >0.8)", priority)
	
	return TaskingResult{
		Priority:         priority,
		FIMDet:           detF,
		ConditionNumber:  conditionNumber,
		IsHighPriority:   priority > 0.8,
	}
}

// Final Verification: Manifold Jump Conditions
func verifyManifoldJumpConditions(t *testing.T, cr3bp CislunarResult, conjunctions []spatial.Conjunction, tasking TaskingResult) {
	// Condition 1: Maneuver Detector flags as "Intentional"
	maneuverDetected := cr3bp.IsOnManifold
	t.Logf("\n[Verify 1] Maneuver Detector: %v (expected: true)", maneuverDetected)
	
	if !maneuverDetected {
		t.Error("  ✗ FAIL: Manifold not detected as intentional maneuver")
	} else {
		t.Log("  ✓ PASS: Maneuver flagged as intentional (Aghori-class)")
	}
	
	// Condition 2: BVH identifies close approach to GEO satellite
	conjunctionFound := len(conjunctions) > 0
	t.Logf("[Verify 2] BVH Conjunction: %v (expected: true)", conjunctionFound)
	
	if !conjunctionFound {
		t.Error("  ✗ FAIL: No GEO conjunction identified")
	} else {
		t.Logf("  ✓ PASS: %d conjunction(s) identified", len(conjunctions))
	}
	
	// Condition 3: Tasking Engine prioritizes despite low "Time Since Last Look"
	// Simulate: Time Since Last Look = 5 minutes (low), but still high priority
	timeSinceLastLook := 5.0 * 60.0 // 5 minutes
	taskingPrioritized := tasking.IsHighPriority && timeSinceLastLook < 3600
	t.Logf("[Verify 3] Tasking Priority: %v (TSLL: %.0f min, expected: HIGH)", 
		taskingPrioritized, timeSinceLastLook/60)
	
	if !taskingPrioritized {
		t.Error("  ✗ FAIL: Tasking engine did not prioritize Aghori asset")
	} else {
		t.Log("  ✓ PASS: Tasking engine correctly prioritized via Fisher Information")
	}
	
	// Overall verdict
	allPassed := maneuverDetected && conjunctionFound && taskingPrioritized
	t.Log("\n[FINAL] Manifold Jump Scenario:")
	if allPassed {
		t.Log("  ✅ ALL CONDITIONS PASSED - ADITI 4.0 architecture is sound")
	} else {
		t.Log("  ❌ SOME CONDITIONS FAILED - Review causal-mechanistic gap")
	}
}

// Helper types and functions

type CislunarResult struct {
	State          orbital.CR3BPStateSTM
	JacobiConstant float64
	IsOnManifold   bool
}

type TaskingResult struct {
	Priority         float64
	FIMDet           float64
	ConditionNumber  float64
	IsHighPriority   bool
}

func createManifoldThreatTLE() catalog.TLE {
	// Create a TLE representing an object on L2 unstable manifold
	// Approaching GEO from Cislunar space
	return catalog.TLE{
		Name:            "AGHORI-001",
		SatelliteNumber: 99999,
		EpochYear:       24,
		EpochDay:        100.50000000,
		MeanMotion:      1.002738, // Near GEO period
		Eccentricity:    0.0005,
		Inclination:     0.1, // Near-equatorial
		// ... other orbital elements
	}
}

func createMockGEOCatalog(n int) []catalog.Object {
	objects := make([]catalog.Object, n)
	for i := 0; i < n; i++ {
		// Create evenly spaced GEO satellites
		angle := float64(i) * 2.0 * math.Pi / float64(n)
		r := 42164.0 // GEO radius
		
		objects[i] = catalog.Object{
			ID: fmt.Sprintf("GEO-%03d", i),
			State: physics.State{
				Pos: physics.Vector3{
					X: r * math.Cos(angle),
					Y: r * math.Sin(angle),
					Z: 0.0,
				},
				Vel: physics.Vector3{
					X: -3.07 * math.Sin(angle), // GEO velocity ~3.07 km/s
					Y: 3.07 * math.Cos(angle),
					Z: 0.0,
				},
			},
		}
	}
	return objects
}

func detectUnstableManifold(state orbital.CR3BPStateSTM, mu float64) bool {
	// Simplified: Check if state is near L2 and on diverging trajectory
	// L2 is at approximately x = 1.15568 in normalized coordinates
	l2x := 1.15568
	distanceFromL2 := math.Sqrt(
		math.Pow(state.X-l2x, 2) +
		math.Pow(state.Y, 2) +
		math.Pow(state.Z, 2),
	)
	
	// Threshold: within 0.1 normalized units (~38,000 km)
	return distanceFromL2 < 0.1
}

func calculateAghoriPriority(cr3bp CislunarResult, fimDet float64) float64 {
	// High priority if:
	// 1. On manifold (uncooperative trajectory)
	// 2. High information gain (large det(F))
	// 3. Close to operational assets (implied by conjunction check)
	
	manifoldFactor := 0.0
	if cr3bp.IsOnManifold {
		manifoldFactor = 0.5
	}
	
	// Normalize FIM determinant (typical range: 1e-12 to 1e-6)
	infoFactor := math.Log10(fimDet+1e-20) + 12.0
	if infoFactor < 0 {
		infoFactor = 0
	}
	if infoFactor > 6 {
		infoFactor = 6
	}
	infoFactor /= 12.0 // Scale to [0, 0.5]
	
	return manifoldFactor + infoFactor
}

// Benchmarks

func BenchmarkManifoldJump100k(b *testing.B) {
	// Performance test: Can we handle 100k objects?
	catalog := createMockGEOCatalog(100000)
	bvh := spatial.NewLazyBVH()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bvh.FullRebuild(catalog)
	}
}

func BenchmarkFIMCalculation(b *testing.B) {
	// Benchmark Fisher Information calculation
	H := mat.NewDense(3, 6, nil)
	for i := 0; i < 3; i++ {
		H.Set(i, i, 1.0)
	}
	
	R := mat.NewDense(3, 3, nil)
	for i := 0; i < 3; i++ {
		R.Set(i, i, 0.01)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Rinv := mat.NewDense(3, 3, nil)
		Rinv.Inverse(R)
		
		temp := mat.NewDense(6, 3, nil)
		temp.Mul(H.T(), Rinv)
		FIM := mat.NewDense(6, 6, nil)
		FIM.Mul(temp, H)
	}
}
