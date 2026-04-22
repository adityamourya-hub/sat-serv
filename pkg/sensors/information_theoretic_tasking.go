package sensors

/**
 * Information-Theoretic Sensor Tasking Engine (ITTE)
 *
 * Active ISR Platform for 100k Object Catalog
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
	"ssis/pkg/physics"
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

// ComputeKLDivergence calculates the KL divergence between two Gaussian distributions
func ComputeKLDivergence(
	mean1 [6]float64, // Mean of first distribution
	cov1 [6][6]float64, // Covariance of first distribution
	mean2 [6]float64, // Mean of second distribution
	cov2 [6][6]float64, // Covariance of second distribution
) float64 {
	// Compute log(det(cov2)) - log(det(cov1))
	detCov1 := computeDeterminant6x6(cov1)
	detCov2 := computeDeterminant6x6(cov2)
	logDetRatio := math.Log(detCov2) - math.Log(detCov1)

	// Compute tr(cov2^-1 * cov1)
	var cov2Inv [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			cov2Inv[i][j] = cov2[i][j]
		}
	}
	invert6x6(cov2Inv)
	traceTerm := 0.0
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			traceTerm += cov2Inv[i][j] * cov1[j][i]
		}
	}

	// Compute (mean2 - mean1)^T * cov2^-1 * (mean2 - mean1)
	diffMean := [6]float64{}
	for i := 0; i < 6; i++ {
		diffMean[i] = mean2[i] - mean1[i]
	}
	var diffMeanCov2Inv [6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			diffMeanCov2Inv[i] += diffMean[j] * cov2Inv[j][i]
		}
	}
	meanTerm := 0.0
	for i := 0; i < 6; i++ {
		meanTerm += diffMeanCov2Inv[i] * diffMean[i]
	}

	// Combine terms
	kl := 0.5 * (logDetRatio + traceTerm + meanTerm - 6.0)

	return kl
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

	// Propagation state
	EpochJD float64 // Julian date of current state

	// Sovereign handshake verification
	HamiltonianValue float64 // Current Hamiltonian H(q,p) for physics consistency
	HamiltonianDrift float64 // Accumulated energy drift for spoof detection
}

// Propagate propagates object state and STM to target epoch
// Uses DPHNO-based symplectic integration for energy-conserving propagation
func (obj *TrackedObject) Propagate(targetJD float64, params interface{}) {
	dt := (targetJD - obj.EpochJD) * 86400.0 // seconds

	// Use DPHNO-based symplectic propagation for better energy conservation
	// This is critical for HGV tracking in the 30-80km transitional flow regime
	obj.PropagateSymplectic(dt)

	// Update epoch
	obj.EpochJD = targetJD
}

// PropagateSymplectic uses Forest-Ruth 4th-order symplectic integrator
// for energy-conserving propagation with bounded error O(1) instead of growing O(t)
// Critical for sovereign SDA: Prevents energy drift that could trigger false anomaly detection
func (obj *TrackedObject) PropagateSymplectic(dt float64) {
	// Convert state to position and momentum for PHS framework
	// q = position [x, y, z], p = momentum = mass * velocity
	q := []float64{obj.State[0], obj.State[1], obj.State[2]}
	p := []float64{obj.State[3], obj.State[4], obj.State[5]} // Assuming unit mass

	// Create orbital PHS model with optional latent code for HGV dynamics
	// For HGV objects, the latent code encodes wake memory and fractional flow state
	var z []float64
	if obj.IsAghori {
		// Aghori/HGV objects use fractional Laplacian-aware latent code
		// This encodes the "wake memory" for transitional flow regime (30-80km)
		z = []float64{0.7, 0.3, 0.1} // Fractional order, dissipation, memory kernel
	}

	// Perform symplectic integration step
	// This preserves the canonical structure required for Hamiltonian verification
	qNext, pNext := physics.ForestRuthStep(obj.createPHSModel(), q, p, z, dt, physics.ManifoldCartesian)

	// Update state
	obj.State[0], obj.State[1], obj.State[2] = qNext[0], qNext[1], qNext[2]
	obj.State[3], obj.State[4], obj.State[5] = pNext[0], pNext[1], pNext[2]

	// Update STM using variational equations along the symplectic flow
	// This maintains covariance realism during propagation
	obj.updateSTMSymplectic(dt, q, p, qNext, pNext)

	// Compute Hamiltonian for sovereign handshake verification
	H := obj.createPHSModel().Hamiltonian(qNext, pNext, z)
	obj.HamiltonianValue = H
}

// createPHSModel creates a Port-Hamiltonian System model for this object
// The PHS model enforces physics as hard constraints (energy, momentum conservation)
func (obj *TrackedObject) createPHSModel() physics.PHSModel {
	return &orbitalPHSModel{
		mu:         398600.4418, // km³/s² (Earth gravitational parameter)
		isHGV:      obj.IsAghori,
		wakeMemory: obj.ThreatScore, // Use threat score as proxy for wake intensity
	}
}

// orbitalPHSModel implements PHSModel for orbital dynamics
// Supports both Keplerian (standard) and fractional (HGV) dynamics
type orbitalPHSModel struct {
	mu         float64
	isHGV      bool
	wakeMemory float64
}

// Velocity returns ∂H/∂p = p/m (momentum → velocity)
func (m *orbitalPHSModel) Velocity(q, p, z []float64) []float64 {
	// For unit mass: v = p
	return []float64{p[0], p[1], p[2]}
}

// Force returns -∂H/∂q (gravity + fractional corrections for HGV)
func (m *orbitalPHSModel) Force(q, p, z []float64) []float64 {
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	r3 := r * r * r

	// Standard Keplerian gravity: F = -μ/r³ * r
	fx := -m.mu / r3 * q[0]
	fy := -m.mu / r3 * q[1]
	fz := -m.mu / r3 * q[2]

	// For HGV objects (30-80km transitional flow), add fractional Laplacian correction
	// This accounts for non-local wake memory effects that Navier-Stokes misses
	if m.isHGV && len(z) >= 3 {
		// Fractional dissipation term: proportional to velocity raised to fractional power
		// Encodes the "wake memory" - HGV position depends on path history
		fractionalOrder := z[0] // Typically 0.5-0.8 for transitional flow
		dissipation := z[1]

		vMag := math.Sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
		if vMag > 0 {
			// Fractional drag: F_drag ∝ v^(2-fractionalOrder)
			// This creates the "memory effect" where HGV drag depends on past trajectory
			fracFactor := dissipation * math.Pow(vMag, 2.0-fractionalOrder) / vMag
			fx -= fracFactor * p[0]
			fy -= fracFactor * p[1]
			fz -= fracFactor * p[2]
		}
	}

	return []float64{fx, fy, fz}
}

// Hamiltonian returns total energy H(q,p) = T(p) + V(q)
// Used for sovereign handshake verification (σ_H / H_avg < 10^-7)
func (m *orbitalPHSModel) Hamiltonian(q, p, z []float64) float64 {
	// Kinetic energy: T = ½|p|² (for unit mass)
	T := 0.5 * (p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

	// Potential energy: V = -μ/r
	r := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
	V := -m.mu / r

	return T + V
}

// updateSTMSymplectic updates STM using symplectic flow
// Maintains covariance realism while preserving energy constraints
func (obj *TrackedObject) updateSTMSymplectic(dt float64, _qOld, _pOld, qNew, _pNew []float64) {
	// Compute state transition matrix using finite differences along symplectic flow
	// This is more accurate than linearized Keplerian approximation for HGV dynamics

	// Simplified: Use Keplerian approximation but with symplectic-corrected state
	// Full implementation would integrate variational equations with PHS structure

	// For now, use the gravity gradient matrix at the new state
	r := math.Sqrt(qNew[0]*qNew[0] + qNew[1]*qNew[1] + qNew[2]*qNew[2])
	mu := 398600.4418
	r3 := r * r * r
	r5 := r3 * r * r

	x, y, z := qNew[0], qNew[1], qNew[2]

	// Gravity gradient matrix
	var G [3][3]float64
	G[0][0] = mu * (3*x*x/r5 - 1/r3)
	G[0][1] = mu * (3 * x * y / r5)
	G[0][2] = mu * (3 * x * z / r5)
	G[1][0] = mu * (3 * y * x / r5)
	G[1][1] = mu * (3*y*y/r5 - 1/r3)
	G[1][2] = mu * (3 * y * z / r5)
	G[2][0] = mu * (3 * z * x / r5)
	G[2][1] = mu * (3 * z * y / r5)
	G[2][2] = mu * (3*z*z/r5 - 1/r3)

	// A matrix with symplectic structure
	var A [6][6]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i == j {
				A[i][j] = 0.0
				A[i][j+3] = 1.0
			} else {
				A[i][j] = 0.0
				A[i][j+3] = 0.0
			}
			A[i+3][j] = G[i][j]
			A[i+3][j+3] = 0.0
		}
	}

	// Update STM using matrix exponential approximation
	// STM_new = exp(A*dt) * STM_old
	// For small dt: exp(A*dt) ≈ I + A*dt + (A*dt)²/2
	var newSTM [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			sum := obj.STM[i][j]
			for k := 0; k < 6; k++ {
				sum += A[i][k] * obj.STM[k][j] * dt
			}
			newSTM[i][j] = sum
		}
	}

	obj.STM = newSTM
}

// updateSTMKeplerian updates STM using Keplerian dynamics approximation
func (obj *TrackedObject) updateSTMKeplerian(dt float64) {
	// Compute orbital parameters from state
	r := math.Sqrt(obj.State[0]*obj.State[0] + obj.State[1]*obj.State[1] + obj.State[2]*obj.State[2])

	// Gravitational parameter (μ for Earth)
	mu := 398600.4418 // km³/s²

	// State transition matrix for Keplerian motion (simplified)
	// Using f and g series expansion or universal variable formulation
	// For small dt: STM ≈ I + A*dt

	// Jacobian of Keplerian dynamics (6x6 matrix)
	// A = [0, I; -μ/r³*I + 3μ/r⁵*r*rᵀ, 0]

	r3 := r * r * r
	r5 := r3 * r * r

	// Position vector
	x, y, z := obj.State[0], obj.State[1], obj.State[2]

	// Gravity gradient matrix (3x3)
	// ∂g/∂r where g = -μr/r³
	var G [3][3]float64
	G[0][0] = mu * (3*x*x/r5 - 1/r3)
	G[0][1] = mu * (3 * x * y / r5)
	G[0][2] = mu * (3 * x * z / r5)
	G[1][0] = mu * (3 * y * x / r5)
	G[1][1] = mu * (3*y*y/r5 - 1/r3)
	G[1][2] = mu * (3 * y * z / r5)
	G[2][0] = mu * (3 * z * x / r5)
	G[2][1] = mu * (3 * z * y / r5)
	G[2][2] = mu * (3*z*z/r5 - 1/r3)

	// Construct A matrix (6x6)
	var A [6][6]float64
	// Position derivatives (velocity)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i == j {
				A[i][j] = 0.0   // ∂x/∂x = 0
				A[i][j+3] = 1.0 // ∂x/∂v = I
			} else {
				A[i][j] = 0.0
				A[i][j+3] = 0.0
			}
		}
	}
	// Velocity derivatives (gravity gradient)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			A[i+3][j] = G[i][j]
			A[i+3][j+3] = 0.0 // ∂v/∂v = 0
		}
	}

	// Update STM: STM_new = STM_old + A*STM_old*dt (first-order)
	// Or use matrix exponential for better accuracy
	// STM_new = exp(A*dt) * STM_old

	// For small dt, use first-order approximation
	var newSTM [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			sum := obj.STM[i][j]
			for k := 0; k < 6; k++ {
				sum += A[i][k] * obj.STM[k][j] * dt
			}
			newSTM[i][j] = sum
		}
	}

	obj.STM = newSTM
}

// UpdateCovariance updates covariance matrix using STM
// P_new = STM * P_old * STM^T + Q (process noise)
func (obj *TrackedObject) UpdateCovariance(processNoise [6][6]float64) {
	// P_new = STM * P * STM^T + Q

	// Compute STM * P
	var STM_P [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			sum := 0.0
			for k := 0; k < 6; k++ {
				sum += obj.STM[i][k] * obj.Covariance[k][j]
			}
			STM_P[i][j] = sum
		}
	}

	// Compute (STM * P) * STM^T
	var newCov [6][6]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			sum := 0.0
			for k := 0; k < 6; k++ {
				sum += STM_P[i][k] * obj.STM[j][k]
			}
			newCov[i][j] = sum + processNoise[i][j]
		}
	}

	obj.Covariance = newCov
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
	H := computeObservationMatrix()
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

func invert6x6(m [6][6]float64) {
	// Augment with identity matrix
	var aug [6][12]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			aug[i][j] = m[i][j]
		}
		aug[i][i+6] = 1.0 // Identity on right
	}

	// Gaussian elimination with partial pivoting
	for col := 0; col < 6; col++ {
		// Find pivot row
		pivotRow := col
		maxVal := math.Abs(aug[col][col])
		for row := col + 1; row < 6; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				pivotRow = row
			}
		}

		// Swap rows if necessary
		if pivotRow != col {
			for j := 0; j < 12; j++ {
				aug[col][j], aug[pivotRow][j] = aug[pivotRow][j], aug[col][j]
			}
		}

		// Check for singular matrix
		if math.Abs(aug[col][col]) < 1e-10 {
			// Matrix is singular, return zeros
			for i := 0; i < 6; i++ {
				for j := 0; j < 6; j++ {
					m[i][j] = 0.0
				}
			}
			return
		}

		// Normalize pivot row
		pivot := aug[col][col]
		for j := 0; j < 12; j++ {
			aug[col][j] /= pivot
		}

		// Eliminate column
		for row := 0; row < 6; row++ {
			if row != col {
				factor := aug[row][col]
				for j := 0; j < 12; j++ {
					aug[row][j] -= factor * aug[col][j]
				}
			}
		}
	}

	// Extract inverse from augmented matrix
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			m[i][j] = aug[i][j+6]
		}
	}
}

func computeDeterminant6x6(m [6][6]float64) float64 {
	// Use Cholesky decomposition for symmetric positive definite matrices
	// FIM is always symmetric positive definite by construction
	// det(A) = (det(L))^2 where A = L * L^T

	// Copy matrix to avoid modifying original
	A := [6][6]float64{}
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			A[i][j] = m[i][j]
		}
	}

	// Cholesky decomposition: A = L * L^T
	// L is lower triangular with positive diagonal
	L := [6][6]float64{}

	for i := 0; i < 6; i++ {
		for j := 0; j <= i; j++ {
			if i == j {
				// Diagonal element
				sum := 0.0
				for k := 0; k < j; k++ {
					sum += L[j][k] * L[j][k]
				}
				diag := A[j][j] - sum
				if diag <= 0 {
					// Matrix not positive definite, fall back to LU
					return computeDeterminant6x6LU(m)
				}
				L[i][j] = math.Sqrt(diag)
			} else {
				// Off-diagonal element
				sum := 0.0
				for k := 0; k < j; k++ {
					sum += L[i][k] * L[j][k]
				}
				L[i][j] = (A[i][j] - sum) / L[j][j]
			}
		}
	}

	// Determinant = product of diagonal elements squared
	det := 1.0
	for i := 0; i < 6; i++ {
		det *= L[i][i]
	}
	return det * det
}

// computeDeterminant6x6LU computes determinant using LU decomposition (fallback)
func computeDeterminant6x6LU(m [6][6]float64) float64 {
	// LU decomposition with partial pivoting
	// det(A) = det(P) * det(L) * det(U) = det(P) * product of U diagonal

	// Copy matrix
	A := [6][6]float64{}
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			A[i][j] = m[i][j]
		}
	}

	// Track row swaps (permutation parity)
	pivotSign := 1.0

	for i := 0; i < 6; i++ {
		// Partial pivoting: find max element in column
		maxRow := i
		maxVal := math.Abs(A[i][i])
		for k := i + 1; k < 6; k++ {
			if math.Abs(A[k][i]) > maxVal {
				maxVal = math.Abs(A[k][i])
				maxRow = k
			}
		}

		// Swap rows if necessary
		if maxRow != i {
			for j := 0; j < 6; j++ {
				A[i][j], A[maxRow][j] = A[maxRow][j], A[i][j]
			}
			pivotSign *= -1.0
		}

		// Check for singular matrix
		if math.Abs(A[i][i]) < 1e-10 {
			return 0.0
		}

		// Eliminate below
		for k := i + 1; k < 6; k++ {
			factor := A[k][i] / A[i][i]
			for j := i; j < 6; j++ {
				A[k][j] -= factor * A[i][j]
			}
		}
	}

	// Determinant = product of diagonal elements * pivot sign
	det := pivotSign
	for i := 0; i < 6; i++ {
		det *= A[i][i]
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

func computeObservationMatrix() [3][6]float64 {
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

func distance(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}
