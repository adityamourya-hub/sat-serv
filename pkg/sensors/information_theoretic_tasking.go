package sensors

/**
 * Information-Theoretic Sensor Tasking Engine (ITTE)
 *
 * ADITI 4.0: Active ISR Platform for 100k Object Catalog
 *
 * Transitions from passive catalog to active intelligence gathering:
 * - Fisher Information Matrix (FIM) based prioritization
 * - D-optimality for covariance minimization
 * - KL-Divergence for "surprisal" detection
 * - Event-driven interrupts for hypersonic launches
 *
 * Mathematical Foundation:
 * - Information Gain: I = H^T R^-1 H (FIM)
 * - Cramér-Rao Lower Bound: P >= F^-1
 * - D-Optimality: max det(F) = min uncertainty volume
 * - KL-Divergence: D_KL(P_obs || P_pred) = surprisal
 *
 * References:
 * - Linares-Laboratory/Deep-RL-SSA: A3C for large-scale tasking
 * - Marsden & West: Geometric mechanics for STM propagation
 */

import (
	"container/heap"
	"math"
	"sync"
	"time"
)

// FisherInformationMatrix represents the 6×6 FIM for orbit determination
type FisherInformationMatrix struct {
	F     [6][6]float64 // Information matrix
	Det   float64       // det(F) for D-optimality
	Trace float64       // tr(F) for A-optimality
}

// ComputeFIM calculates Fisher Information for a potential observation
// F = H^T R^-1 H
// where H is measurement sensitivity from STM
func ComputeFIM(
	stm [6][6]float64, // State Transition Matrix
	observationMatrix [3][6]float64, // H (position measurement)
	measurementCovariance [3][3]float64, // R
) FisherInformationMatrix {

	// Compute H^T
	HT := transpose36(observationMatrix)

	// Compute R^-1
	Rinv := invert3x3(measurementCovariance)

	// Compute H^T R^-1
	var HTRinv [6][3]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				HTRinv[i][j] += HT[i][k] * Rinv[k][j]
			}
		}
	}

	// Compute F = H^T R^-1 H
	var F [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			for k := 0; k < 3; k++ {
				F[i][j] += HTRinv[i][k] * observationMatrix[k][j]
			}
		}
	}

	// Compute determinant (D-optimality criterion)
	det := computeDeterminant6x6(F)

	// Compute trace (A-optimality criterion)
	trace := 0.0
	for i := 0; i < 6; i++ {
		trace += F[i][i]
	}

	return FisherInformationMatrix{
		F:     F,
		Det:   det,
		Trace: trace,
	}
}

// InformationGainTask represents a scheduled observation with predicted information
type InformationGainTask struct {
	TaskID        string
	ObjectID      string
	SensorID      string
	ScheduledTime time.Time

	// Predicted information metrics
	PredictedFIM                FisherInformationMatrix
	ExpectedCovarianceReduction float64 // trace(P_before) - trace(P_after)
	SurprisalKL                 float64 // KL divergence from prediction

	// Tasking priority (higher = more urgent)
	Priority float64

	// For heap ordering
	index int // index in heap
}

// TaskPriorityQueue implements a max-heap for information-based prioritization
type TaskPriorityQueue []*InformationGainTask

func (pq TaskPriorityQueue) Len() int { return len(pq) }

func (pq TaskPriorityQueue) Less(i, j int) bool {
	// Max-heap: higher priority first
	return pq[i].Priority > pq[j].Priority
}

func (pq TaskPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *TaskPriorityQueue) Push(x interface{}) {
	n := len(*pq)
	task := x.(*InformationGainTask)
	task.index = n
	*pq = append(*pq, task)
}

func (pq *TaskPriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	task := old[n-1]
	old[n-1] = nil  // avoid memory leak
	task.index = -1 // for safety
	*pq = old[0 : n-1]
	return task
}

// Peek returns highest priority task without removing
func (pq *TaskPriorityQueue) Peek() *InformationGainTask {
	if len(*pq) == 0 {
		return nil
	}
	return (*pq)[0]
}

// InformationTheoreticTasker implements D-optimality sensor scheduling
type InformationTheoreticTasker struct {
	mu sync.RWMutex

	// Priority queue of pending tasks
	TaskQueue *TaskPriorityQueue

	// Currently executing tasks
	ActiveTasks map[string]*InformationGainTask

	// Sensor pool
	Sensors []*SensorPlatform

	// Object catalog with covariance
	ObjectStates map[string]*TrackedObject

	// Configuration
	Config TaskingConfig
}

// TaskingConfig for information-theoretic optimization
type TaskingConfig struct {
	// FIM thresholds
	MinFIMDet      float64 // Minimum determinant for D-optimality
	MinSurprisalKL float64 // Minimum KL divergence for "interesting" objects

	// Timing
	UpdateInterval time.Duration // How often to recompute priorities
	MaxLookahead   time.Duration // How far ahead to schedule

	// Resource limits
	MaxConcurrentTasks int
	MinSlewTime        time.Duration
}

// TrackedObject represents catalog entry with covariance
type TrackedObject struct {
	ID         string
	State      [6]float64    // [x,y,z,vx,vy,vz]
	Covariance [6][6]float64 // P matrix
	STM        [6][6]float64 // Current state transition matrix

	// Information metrics
	LastObservation      time.Time
	ObservationCount     int
	TotalInformationGain float64 // Accumulated tr(F)

	// Classification
	IsAghori    bool // Uncooperative/unknown
	IsAndromeda bool // Highly maneuverable
	ThreatScore float64
}

// ComputePriority calculates task priority based on information theory
func (t *InformationTheoreticTasker) ComputePriority(
	obj *TrackedObject,
	sensor *SensorPlatform,
	scheduledTime time.Time,
) float64 {

	// Base priority from covariance (higher uncertainty = higher priority)
	covTrace := computeTrace6x6(obj.Covariance)

	// Predicted FIM for this observation
	H := computeObservationMatrix(obj.State, sensor)
	R := sensor.MeasurementCovariance()

	fim := ComputeFIM(obj.STM, H, R)

	// D-optimality: prefer observations that maximize det(F)
	dOptimality := math.Log(math.Abs(fim.Det) + 1e-10)

	// Surprisal factor (Aghori detection)
	surprisal := 0.0
	if obj.IsAghori {
		surprisal = obj.ThreatScore * 100.0 // Boost priority
	}

	// Time since last observation (custody factor)
	timeSinceLast := scheduledTime.Sub(obj.LastObservation).Hours()
	custodyFactor := math.Min(timeSinceLast/24.0, 1.0) // Max at 24 hours

	// Combined priority (weighted sum)
	priority := 0.4*covTrace + 0.3*dOptimality + 0.2*surprisal + 0.1*custodyFactor*100.0

	return priority
}

// ScheduleTasks generates information-optimal observation schedule
func (t *InformationTheoreticTasker) ScheduleTasks(now time.Time) []*InformationGainTask {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Clear old queue
	t.TaskQueue = &TaskPriorityQueue{}
	heap.Init(t.TaskQueue)

	// For each object-sensor pair, compute priority
	for _, obj := range t.ObjectStates {
		for _, sensor := range t.Sensors {
			// Check visibility
			if !sensor.CanObserve(obj.State, now) {
				continue
			}

			// Compute scheduled time (accounting for slew)
			slewTime := sensor.EstimateSlewTime(obj.State)
			scheduledTime := now.Add(slewTime)

			// Compute priority
			priority := t.ComputePriority(obj, sensor, scheduledTime)

			// Only schedule if above threshold
			if priority > 0.1 {
				task := &InformationGainTask{
					TaskID:        generateTaskID(),
					ObjectID:      obj.ID,
					SensorID:      sensor.ID,
					ScheduledTime: scheduledTime,
					Priority:      priority,
				}
				heap.Push(t.TaskQueue, task)
			}
		}
	}

	// Extract top N tasks
	numTasks := math.Min(float64(t.TaskQueue.Len()), float64(t.Config.MaxConcurrentTasks))
	scheduled := make([]*InformationGainTask, 0, int(numTasks))

	for len(scheduled) < int(numTasks) && t.TaskQueue.Len() > 0 {
		task := heap.Pop(t.TaskQueue).(*InformationGainTask)
		scheduled = append(scheduled, task)
		t.ActiveTasks[task.TaskID] = task
	}

	return scheduled
}

// EventDrivenInterrupt for hypersonic/rapid processing
type EventDrivenInterrupt struct {
	EventType string // "launch", "maneuver", "conjunction"
	ObjectID  string
	Priority  float64   // Override priority
	Deadline  time.Time // Must execute by
}

// ProcessInterrupt handles high-priority events
func (t *InformationTheoreticTasker) ProcessInterrupt(event EventDrivenInterrupt) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Preempt lower priority tasks
	for _, task := range t.ActiveTasks {
		if task.Priority < event.Priority {
			// Preempt this task
			delete(t.ActiveTasks, task.TaskID)
			// Re-queue with lower priority
			task.Priority = 0.01 // Minimal priority
			heap.Push(t.TaskQueue, task)
		}
	}

	// Schedule interrupt task immediately
	interruptTask := &InformationGainTask{
		TaskID:        generateTaskID(),
		ObjectID:      event.ObjectID,
		Priority:      event.Priority,
		ScheduledTime: time.Now(), // Immediate
	}

	t.ActiveTasks[interruptTask.TaskID] = interruptTask

	// Trigger sensor re-tasking
	t.triggerRetasking()
}

// triggerRetasking signals sensors to recompute schedules
func (t *InformationTheoreticTasker) triggerRetasking() {
	// In production: Send gRPC messages to sensor control systems
	// For now: Set flag for next scheduling cycle
}

// AnytimeGreedyScheduler for interruptible tasking
type AnytimeGreedyScheduler struct {
	tasker *InformationTheoreticTasker

	// Interrupt channel
	interruptChan chan EventDrivenInterrupt

	// Stop signal
	stopChan chan bool
}

// NewAnytimeGreedyScheduler creates interruptible scheduler
func NewAnytimeGreedyScheduler(tasker *InformationTheoreticTasker) *AnytimeGreedyScheduler {
	return &AnytimeGreedyScheduler{
		tasker:        tasker,
		interruptChan: make(chan EventDrivenInterrupt, 100),
		stopChan:      make(chan bool),
	}
}

// Run starts the anytime scheduler
func (s *AnytimeGreedyScheduler) Run() {
	for {
		select {
		case <-s.stopChan:
			return

		case event := <-s.interruptChan:
			// Process high-priority interrupt
			s.tasker.ProcessInterrupt(event)

		default:
			// Normal scheduling cycle
			tasks := s.tasker.ScheduleTasks(time.Now())

			// Dispatch to sensors (non-blocking)
			for _, task := range tasks {
				go s.dispatchTask(task)
			}

			// Wait for next cycle
			time.Sleep(s.tasker.Config.UpdateInterval)
		}
	}
}

// dispatchTask sends task to sensor (async)
func (s *AnytimeGreedyScheduler) dispatchTask(task *InformationGainTask) {
	// Find sensor
	for _, sensor := range s.tasker.Sensors {
		if sensor.ID == task.SensorID {
			// Schedule observation
			sensor.ScheduleObservation(task.ObjectID, task.ScheduledTime)
			return
		}
	}
}

// SendInterrupt injects high-priority event
func (s *AnytimeGreedyScheduler) SendInterrupt(event EventDrivenInterrupt) {
	select {
	case s.interruptChan <- event:
		// Successfully queued
	default:
		// Channel full - handle immediately in goroutine
		go func() {
			s.tasker.ProcessInterrupt(event)
		}()
	}
}

// Helper functions

func transpose36(m [3][6]float64) [6][3]float64 {
	var result [6][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 6; j++ {
			result[j][i] = m[i][j]
		}
	}
	return result
}

func invert3x3(m [3][3]float64) [3][3]float64 {
	// Compute determinant
	det := m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

	if math.Abs(det) < 1e-10 {
		return [3][3]float64{} // Singular
	}

	invDet := 1.0 / det

	var result [3][3]float64
	result[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * invDet
	result[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * invDet
	result[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * invDet
	result[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * invDet
	result[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * invDet
	result[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * invDet
	result[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * invDet
	result[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * invDet
	result[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * invDet

	return result
}

func computeDeterminant6x6(m [6][6]float64) float64 {
	// For symmetric positive definite, use Cholesky or LU
	// Simplified: Return product of diagonal (for diagonal-dominant)
	det := 1.0
	for i := 0; i < 6; i++ {
		det *= m[i][i]
	}
	return det
}

func computeTrace6x6(m [6][6]float64) float64 {
	trace := 0.0
	for i := 0; i < 6; i++ {
		trace += m[i][i]
	}
	return trace
}

func computeObservationMatrix(state [6]float64, sensor *SensorPlatform) [3][6]float64 {
	// For position-only measurement
	// H = [I_3x3, 0_3x3]
	var H [3][6]float64
	H[0][0] = 1.0
	H[1][1] = 1.0
	H[2][2] = 1.0

	// For range/range-rate, would include velocity terms
	// H = ∂measurement/∂state (Jacobian)

	return H
}

func generateTaskID() string {
	// Simple ID generation
	return time.Now().Format("20060102-150405.000000000")
}

// EstimateSlewTime placeholder (would be in SensorPlatform)
func (s *SensorPlatform) EstimateSlewTime(targetState [6]float64) time.Duration {
	// Simplified: Fixed slew time
	return 5 * time.Second
}

// CanObserve placeholder (would be in SensorPlatform)
func (s *SensorPlatform) CanObserve(targetState [6]float64, t time.Time) bool {
	// Simplified: Always true
	return true
}

// MeasurementCovariance placeholder
func (s *SensorPlatform) MeasurementCovariance() [3][3]float64 {
	// 100m position uncertainty
	return [3][3]float64{
		{0.01, 0, 0},
		{0, 0.01, 0},
		{0, 0, 0.01},
	}
}

// ScheduleObservation placeholder
func (s *SensorPlatform) ScheduleObservation(objectID string, t time.Time) {
	// In production: Send command to sensor
}

// VolumetricTasking for Aghori detection
type VolumetricTasker struct {
	ManifoldGateways []GatewayRegion   // L1/L2 manifold entry points
	BlindSpots       []BlindSpotWindow // Detected coverage gaps
	Sensors          []*SensorPlatform
}

// GatewayRegion represents manifold "stare" point
type GatewayRegion struct {
	Center   [3]float64 // ECI position
	Radius   float64    // Stare radius (km)
	Type     string     // "L1_entry", "L2_entry", "unstable_manifold"
	Priority float64    // Information value
}

// TaskVolumetricScan generates "stare" tasks for gateway regions
func (v *VolumetricTasker) TaskVolumetricScan(duration time.Duration) []*InformationGainTask {
	tasks := make([]*InformationGainTask, 0)

	for _, gateway := range v.ManifoldGateways {
		// Find best sensor for this region
		bestSensor := v.selectBestSensor(gateway)

		if bestSensor != nil {
			task := &InformationGainTask{
				TaskID:        generateTaskID(),
				ObjectID:      "VOLUMETRIC_" + gateway.Type,
				SensorID:      bestSensor.ID,
				ScheduledTime: time.Now(),
				Priority:      gateway.Priority,
			}
			tasks = append(tasks, task)
		}
	}

	return tasks
}

func (v *VolumetricTasker) selectBestSensor(gateway GatewayRegion) *SensorPlatform {
	// Select sensor with best view of gateway
	var best *SensorPlatform
	bestScore := -1.0

	for _, sensor := range v.Sensors {
		// Compute visibility score
		score := 1.0 / (1.0 + distance(sensor.Position(), gateway.Center))

		if score > bestScore {
			bestScore = score
			best = sensor
		}
	}

	return best
}

// Position placeholder
func (s *SensorPlatform) Position() [3]float64 {
	return [3]float64{0, 0, 0}
}

func distance(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}
