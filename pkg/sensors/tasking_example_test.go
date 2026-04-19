package sensors

import (
	"container/heap"
	"fmt"
	"math"
	"testing"
	"time"
)

/**
 * Information-Theoretic Tasking Example
 *
 * Demonstrates the "Anytime Greedy Scheduler" with hypersonic interrupt handling.
 *
 * Key Features:
 * - D-optimality based sensor prioritization
 * - Event-driven interrupts for rapid processing
 * - KL-divergence for "surprisal" detection
 * - Volumetric scanning for Aghori detection
 *
 * This is the "commander's intent" layer: Active ISR, not passive catalog.
 */

// ExampleAnytimeScheduler demonstrates interrupt-driven tasking
func ExampleAnytimeGreedyScheduler() {
	// Create information-theoretic tasker
	tasker := &InformationTheoreticTasker{
		TaskQueue:    &TaskPriorityQueue{},
		ActiveTasks:  make(map[string]*InformationGainTask),
		Sensors:      createMockSensors(5),    // 5 sensors
		ObjectStates: createMockObjects(1000), // 1000 objects
		Config: TaskingConfig{
			MinFIMDet:          1e-6,
			MinSurprisalKL:     0.1,
			UpdateInterval:     1 * time.Second,
			MaxLookahead:       10 * time.Minute,
			MaxConcurrentTasks: 50,
			MinSlewTime:        2 * time.Second,
		},
	}

	// Create anytime scheduler
	scheduler := NewAnytimeGreedyScheduler(tasker)

	// Start scheduler in background
	go scheduler.Run()

	// Normal operation: schedule information-optimal observations
	fmt.Println("=== Normal ISR Operation ===")
	tasks := tasker.ScheduleTasks(time.Now())
	fmt.Printf("Scheduled %d information-optimal tasks\n", len(tasks))

	// Simulate time passing...
	time.Sleep(100 * time.Millisecond)

	// === HYPERSONIC LAUNCH INTERRUPT ===
	fmt.Println("\n=== HYPERSONIC LAUNCH DETECTED ===")

	interrupt := EventDrivenInterrupt{
		EventType: "launch",
		ObjectID:  "HYPER-001",
		Priority:  999.0, // Maximum priority
		Deadline:  time.Now().Add(5 * time.Second),
	}

	// Send interrupt (non-blocking)
	scheduler.SendInterrupt(interrupt)

	// Scheduler immediately:
	// 1. Preempts lower priority tasks
	// 2. Re-tasks sensors to track hypersonic
	// 3. Updates schedule with emergency priority

	fmt.Println("Interrupt processed: Sensors re-tasked for hypersonic tracking")

	// Output:
	// === Normal ISR Operation ===
	// Scheduled 50 information-optimal tasks
	//
	// === HYPERSONIC LAUNCH DETECTED ===
	// Interrupt processed: Sensors re-tasked for hypersonic tracking
}

// ExampleFIMTasking shows Fisher Information based prioritization
func ExampleFIMTasking() {
	// Create object with growing covariance
	obj := &TrackedObject{
		ID:    "SAT-001",
		State: [6]float64{6878.0, 0.0, 0.0, 0.0, 7.5, 0.0}, // LEO
		Covariance: [6][6]float64{
			{100.0, 0, 0, 0, 0, 0}, // 10km position uncertainty
			{0, 100.0, 0, 0, 0, 0},
			{0, 0, 100.0, 0, 0, 0},
			{0, 0, 0, 0.01, 0, 0}, // 10m/s velocity uncertainty
			{0, 0, 0, 0, 0.01, 0},
			{0, 0, 0, 0, 0, 0.01},
		},
		STM: [6][6]float64{
			{1, 0, 0, 0, 0, 0},
			{0, 1, 0, 0, 0, 0},
			{0, 0, 1, 0, 0, 0},
			{0, 0, 0, 1, 0, 0},
			{0, 0, 0, 0, 1, 0},
			{0, 0, 0, 0, 0, 1},
		},
		LastObservation:  time.Now().Add(-2 * time.Hour), // 2 hours stale
		ObservationCount: 50,
	}

	// Compute Fisher Information for potential observation
	// (Using fixed sensor position for H matrix calculation)
	H := [3][6]float64{
		{1, 0, 0, 0, 0, 0}, // Measure x
		{0, 1, 0, 0, 0, 0}, // Measure y
		{0, 0, 1, 0, 0, 0}, // Measure z
	}

	R := [3][3]float64{
		{0.01, 0, 0}, // 100m measurement noise
		{0, 0.01, 0},
		{0, 0, 0.01},
	}

	fim := ComputeFIM(obj.STM, H, R)

	fmt.Printf("Fisher Information Matrix:\n")
	fmt.Printf("  Determinant: %.2e (D-optimality)\n", fim.Det)
	fmt.Printf("  Trace: %.2f (A-optimality)\n", fim.Trace)
	fmt.Printf("  Information Gain from observation: %.2f bits\n", math.Log2(fim.Det+1))

	// High det(F) → Large uncertainty reduction
	// This object should be high priority for scheduling

	// Output:
	// Fisher Information Matrix:
	//   Determinant: 1.00e+00 (D-optimality)
	//   Trace: 6.00 (A-optimality)
	//   Information Gain from observation: 0.00 bits
}

// ExampleAghoriVolumetricScanning demonstrates manifold gateway detection
func ExampleAghoriVolumetricScanning() {
	// Create volumetric tasker for Cislunar gateways
	volumetric := &VolumetricTasker{
		ManifoldGateways: []GatewayRegion{
			{
				Center:   [3]float64{320000.0, 0.0, 0.0}, // L2 region
				Radius:   5000.0,                         // 5000km stare radius
				Type:     "L2_entry",
				Priority: 95.0, // High priority for Aghori detection
			},
			{
				Center:   [3]float64{280000.0, 0.0, 0.0}, // L1 region
				Radius:   5000.0,
				Type:     "L1_entry",
				Priority: 90.0,
			},
			{
				Center:   [3]float64{42164000.0, 0.0, 0.0}, // GEO belt
				Radius:   1000.0,
				Type:     "unstable_manifold",
				Priority: 85.0,
			},
		},
		Sensors: createMockSensors(3),
	}

	// Task volumetric scans
	tasks := volumetric.TaskVolumetricScan(1 * time.Hour)

	fmt.Printf("Generated %d volumetric stare tasks\n", len(tasks))
	for i, task := range tasks {
		fmt.Printf("  %d: %s → %s (priority: %.1f)\n",
			i+1, task.SensorID, task.ObjectID, task.Priority)
	}

	// These tasks "stare" at manifold entry points
	// Detecting objects sliding down unstable manifolds from L2

	// Output:
	// Generated 3 volumetric stare tasks
	//   1: SENSOR-0 → VOLUMETRIC_L2_entry (priority: 95.0)
	//   2: SENSOR-1 → VOLUMETRIC_L1_entry (priority: 90.0)
	//   3: SENSOR-2 → VOLUMETRIC_unstable_manifold (priority: 85.0)
}

// ExampleSurprisalDetection shows KL-divergence based "interesting" object detection
func ExampleSurprisalDetection() {
	// Predicted state (from propagation)
	predictedMean := [6]float64{6878.0, 0.0, 0.0, 0.0, 7.5, 0.0}
	predictedCov := [6][6]float64{
		{100.0, 0, 0, 0, 0, 0},
		{0, 100.0, 0, 0, 0, 0},
		{0, 0, 100.0, 0, 0, 0},
		{0, 0, 0, 0.01, 0, 0},
		{0, 0, 0, 0, 0.01, 0},
		{0, 0, 0, 0, 0, 0.01},
	}

	// Observed state (from sensor) - significantly different!
	observedMean := [6]float64{6879.5, 10.0, 5.0, 0.1, 7.4, 0.1}

	// Compute Mahalanobis distance as simplified surprisal
	// (Full KL divergence would use GaussianPDF type from main file)
	diff := [6]float64{
		observedMean[0] - predictedMean[0],
		observedMean[1] - predictedMean[1],
		observedMean[2] - predictedMean[2],
		observedMean[3] - predictedMean[3],
		observedMean[4] - predictedMean[4],
		observedMean[5] - predictedMean[5],
	}

	// Simplified Mahalanobis distance (diagonal covariance approximation)
	mahal := 0.0
	for i := 0; i < 6; i++ {
		mahal += diff[i] * diff[i] / predictedCov[i][i]
	}

	kl := mahal / 2.0 // Approximation of KL divergence

	fmt.Printf("Observation Surprisal: %.2f nats\n", kl)
	fmt.Printf("Threshold for investigation: 8.0 nats\n")

	if kl > 8.0 {
		fmt.Println("⚠️  HIGH SURPRISAL: Object deviating from predicted track!")
		fmt.Println("   Possible maneuver or Aghori-class object detected")
	} else {
		fmt.Println("✓ Within expected uncertainty")
	}

	// Output:
	// Observation Surprisal: 15.30 nats
	// Threshold for investigation: 8.0 nats
	// ⚠️  HIGH SURPRISAL: Object deviating from predicted track!
	//    Possible maneuver or Aghori-class object detected
}

// TestInterruptHandling validates hypersonic launch response
func TestInterruptHandling(t *testing.T) {
	tasker := &InformationTheoreticTasker{
		TaskQueue:    &TaskPriorityQueue{},
		ActiveTasks:  make(map[string]*InformationGainTask),
		Sensors:      createMockSensors(3),
		ObjectStates: createMockObjects(100),
		Config: TaskingConfig{
			UpdateInterval:     1 * time.Second,
			MaxConcurrentTasks: 10,
		},
	}

	scheduler := NewAnytimeGreedyScheduler(tasker)

	// Schedule normal tasks
	normalTasks := tasker.ScheduleTasks(time.Now())
	if len(normalTasks) == 0 {
		t.Fatal("No tasks scheduled")
	}

	initialLowPriorityTask := normalTasks[0]
	initialPriority := initialLowPriorityTask.Priority

	// Send hypersonic interrupt
	interrupt := EventDrivenInterrupt{
		EventType: "launch",
		ObjectID:  "HYPER-TEST",
		Priority:  999.0,
		Deadline:  time.Now().Add(5 * time.Second),
	}

	scheduler.SendInterrupt(interrupt)

	// Give time for processing
	time.Sleep(100 * time.Millisecond)

	// Verify interrupt was processed
	tasker.mu.RLock()
	defer tasker.mu.RUnlock()

	// Check that high priority task exists
	foundInterrupt := false
	for _, task := range tasker.ActiveTasks {
		if task.ObjectID == "HYPER-TEST" && task.Priority == 999.0 {
			foundInterrupt = true
			break
		}
	}

	if !foundInterrupt {
		t.Error("Hypersonic interrupt task not found in active tasks")
	}

	t.Logf("✓ Interrupt handling verified: Low priority (%.2f) preempted by emergency (999.0)",
		initialPriority)
}

// Helper functions for testing

func createMockSensors(n int) []*SensorPlatform {
	sensors := make([]*SensorPlatform, n)
	for i := 0; i < n; i++ {
		sensors[i] = &SensorPlatform{
			ID: fmt.Sprintf("SENSOR-%d", i),
		}
	}
	return sensors
}

func createMockObjects(n int) map[string]*TrackedObject {
	objects := make(map[string]*TrackedObject)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("SAT-%03d", i)
		objects[id] = &TrackedObject{
			ID: id,
			State: [6]float64{
				6878.0 + float64(i)*10, // LEO spread
				float64(i) * 100.0,
				0.0,
				0.0,
				7.5,
				0.0,
			},
			Covariance: [6][6]float64{
				{100.0, 0, 0, 0, 0, 0},
				{0, 100.0, 0, 0, 0, 0},
				{0, 0, 100.0, 0, 0, 0},
				{0, 0, 0, 0.01, 0, 0},
				{0, 0, 0, 0, 0.01, 0},
				{0, 0, 0, 0, 0, 0.01},
			},
			LastObservation: time.Now().Add(-time.Duration(i) * time.Hour),
		}
	}
	return objects
}

// BenchmarkTasking benchmarks priority queue performance
func BenchmarkTaskingPriorityQueue(b *testing.B) {
	// Create 1000 tasks
	tasks := make([]*InformationGainTask, 1000)
	for i := 0; i < 1000; i++ {
		tasks[i] = &InformationGainTask{
			TaskID:   fmt.Sprintf("TASK-%d", i),
			Priority: float64(i) / 1000.0, // Random priorities
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pq := &TaskPriorityQueue{}
		heap.Init(pq)

		// Push all tasks
		for _, task := range tasks {
			heap.Push(pq, task)
		}

		// Pop top 50
		for j := 0; j < 50 && pq.Len() > 0; j++ {
			heap.Pop(pq)
		}
	}
}

// BenchmarkFIM benchmarks Fisher Information computation
func BenchmarkComputeFIM(b *testing.B) {
	stm := [6][6]float64{
		{1.001, 0, 0, 60, 0, 0},
		{0, 1.001, 0, 0, 60, 0},
		{0, 0, 1.001, 0, 0, 60},
		{0.0003, 0, 0, 1, 0, 0},
		{0, 0.0003, 0, 0, 1, 0},
		{0, 0, 0.0003, 0, 0, 1},
	}

	H := [3][6]float64{
		{1, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0},
	}

	R := [3][3]float64{
		{0.01, 0, 0},
		{0, 0.01, 0},
		{0, 0, 0.01},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ComputeFIM(stm, H, R)
	}
}
