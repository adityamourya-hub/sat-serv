package collision

import (
	"context"
	"fmt"
	"log"
	"math"
	"runtime"
	"time"

	"ssis/pkg/physics"

	"github.com/golang/geo/s2"
)

// NOTE: S2 geometry temporarily disabled due to API compatibility issues
// The collision monitor uses basic distance calculations as fallback
/*
import (
	"github.com/golang/geo/s2"
)
*/

// ConjunctionCandidate represents a potential collision pair
type ConjunctionCandidate struct {
	PrimaryID   int
	SecondaryID int
	DistanceKm  float64
	TCA         time.Time // Time of Closest Approach
	Probability float64   // Collision probability (Monte Carlo)
	S2Distance  float64   // Quick S2-based distance
}

/**
 * Collision Monitor: Executive Loop for 100Hz Avoidance System
 *
 * Three-stage filtering pipeline:
 * 1. Coarse Filter: Basic distance search (< 2ms)
 * 1. Coarse Filter: S2 integer range search (< 2ms)
 * 2. Fine Filter: Euclidean distance check (< 5ms)
 * 3. High-Fidelity: Pc calculation + RL policy (< 20ms)
 *
 * Total loop: < 50ms for sensor-to-action latency
 *
 * DMD Optimization: Uses DMD-reduced modes for debris storm handling,
 * treating mode centroids as virtual candidates (1000 debris → 10 modes)
 */

// Monitor orchestrates the collision avoidance pipeline
type Monitor struct {
	db         CandidateSearcher   // S2 integer index
	propagator *physics.Propagator // High-fidelity physics
	policy     PolicyClient        // RL evasion policy
	threshold  float64             // Pc threshold (default 1e-4)

	// Tactical state
	hvaID         int // High-Value Asset NORAD ID
	lastAlert     time.Time
	alertCooldown time.Duration

	// Performance metrics
	loopCount     uint64
	s2Latency     time.Duration
	propLatency   time.Duration
	policyLatency time.Duration
}

// CandidateSearcher interface for S2 database queries
type CandidateSearcher interface {
	FindPotentialConjunctions(ctx context.Context, point s2.Point, radiusKm float64) ([]ConjunctionCandidate, error)
	FindConjunctionsFast(ctx context.Context, primary physics.OrbitalObject, radiusKm float64) ([]ConjunctionCandidate, error)
}

// PolicyClient interface for RL evasion head
type PolicyClient interface {
	RequestAction(ctx context.Context, req *PolicyRequest) (*PolicyResponse, error)
}

// PolicyRequest sends threat context to RL policy engine
type PolicyRequest struct {
	HvaID     int32
	Threats   []ThreatContext
	Timestamp time.Time
}

// ThreatContext describes a collision threat
type ThreatContext struct {
	ObjectID  int32
	Pc        float64   // Probability of collision
	Urgency   int32     // 1-10 scale
	TCA       time.Time // Time of closest approach
	RangeKm   float64
	RelVelKms float64
}

// PolicyResponse contains evasion maneuver recommendation
type PolicyResponse struct {
	Action     EvasionAction
	Confidence float64
	DeltaV     [3]float64 // m/s
	BurnTime   time.Duration
	Rationale  string
}

// EvasionAction types
type EvasionAction int

const (
	ActionNone    EvasionAction = iota // No action required
	ActionMonitor                      // Increased tracking
	ActionStandby                      // Prepare for maneuver
	ActionExecute                      // Immediate burn
)

// SITREP formats for mesh broadcast (Signals Handbook compliance)
type SITREP struct {
	Timestamp   time.Time `json:"timestamp"`
	HvaID       int32     `json:"hva_id"`
	ThreatCount int       `json:"threat_count"`
	MaxPc       float64   `json:"max_pc"`
	ActionTaken string    `json:"action_taken"`
	OperatorID  string    `json:"operator_id"`
}

// SpotReport for immediate threat escalation
type SpotReport struct {
	Timestamp time.Time `json:"timestamp"`
	HvaID     int32     `json:"hva_id"`
	ThreatID  int32     `json:"threat_id"`
	Pc        float64   `json:"pc"`
	TCA       time.Time `json:"tca"`
	Urgency   string    `json:"urgency"`
	GridRef   string    `json:"grid_ref"` // S2 cell reference
}

// Performance targets for 100Hz loop
const (
	TargetS2Latency     = 2 * time.Millisecond
	TargetPropLatency   = 5 * time.Millisecond
	TargetPolicyLatency = 20 * time.Millisecond
	TargetTotalLatency  = 50 * time.Millisecond

	// Default Pc threshold (1 in 10,000)
	DefaultPcThreshold = 1e-4
)

// NewMonitor creates the collision avoidance executive
func NewMonitor(searcher CandidateSearcher, prop *physics.Propagator, policy PolicyClient) *Monitor {
	return &Monitor{
		db:            searcher,
		propagator:    prop,
		policy:        policy,
		threshold:     DefaultPcThreshold,
		alertCooldown: 5 * time.Second,
	}
}

// SetHVA sets the High-Value Asset to protect
func (m *Monitor) SetHVA(noradID int) {
	m.hvaID = noradID
	log.Printf("[MONITOR] HVA set: NORAD %d", noradID)
}

// Run starts the 100Hz executive loop
func (m *Monitor) Run(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Millisecond) // 100Hz
	defer ticker.Stop()

	log.Printf("[MONITOR] Executive loop started (100Hz)")
	log.Printf("[MONITOR] Pc threshold: %.1e", m.threshold)

	for {
		select {
		case <-ctx.Done():
			log.Printf("[MONITOR] Shutting down")
			return
		case <-ticker.C:
			if m.hvaID == 0 {
				continue // No HVA assigned
			}

			loopStart := time.Now()
			if err := m.evaluate(ctx); err != nil {
				log.Printf("[MONITOR] Evaluation error: %v", err)
			}

			// Monitor latency
			totalLatency := time.Since(loopStart)
			if totalLatency > TargetTotalLatency {
				log.Printf("[MONITOR] WARNING: Loop latency %v exceeds target %v",
					totalLatency, TargetTotalLatency)
			}

			m.loopCount++
		}
	}
}

// evaluate executes the three-stage filtering pipeline
func (m *Monitor) evaluate(ctx context.Context) error {
	// === STAGE 1: Get HVA state ===
	hvaState, err := m.getHVAState()
	if err != nil {
		return fmt.Errorf("hva state: %w", err)
	}

	// === STAGE 2: S2 Coarse Filter (< 2ms) ===
	s2Start := time.Now()

	// 50km search shell for candidate selection
	candidates, err := m.db.FindPotentialConjunctions(ctx,
		s2.PointFromLatLng(s2.LatLngFromDegrees(hvaState.Lat, hvaState.Lng)),
		50.0)
	if err != nil {
		return fmt.Errorf("s2 search: %w", err)
	}

	m.s2Latency = time.Since(s2Start)

	if len(candidates) == 0 {
		return nil // No candidates to evaluate
	}

	// === STAGE 3: Fine Filter + Pc Calculation (< 5ms) ===
	propStart := time.Now()

	var threats []ThreatContext

	// DMD Optimization: If >100 candidates, use mode centroids
	if len(candidates) > 100 {
		// Treat DMD mode centroids as virtual candidates
		// (Implementation would call ai/v3/dmd_debris_cloud.py)
		candidates = m.reduceViaDMD(candidates)
	}

	// High-fidelity Pc calculation for each candidate
	for _, candidate := range candidates {
		// Quick Euclidean pre-filter (cheap)
		if candidate.S2Distance > 10.0 { // 10km coarse threshold
			continue
		}

		// Full Pc calculation (expensive, only on near candidates)
		pc := m.calculatePc(hvaState, candidate)

		if pc > m.threshold {
			threats = append(threats, ThreatContext{
				ObjectID:  int32(candidate.SecondaryID),
				Pc:        pc,
				Urgency:   m.calculateUrgency(pc),
				RangeKm:   candidate.S2Distance,
				RelVelKms: 7.5, // LEO typical
			})
		}
	}

	m.propLatency = time.Since(propStart)

	// === STAGE 4: Policy Decision (< 20ms) ===
	if len(threats) > 0 {
		return m.handleThreats(ctx, threats, hvaState)
	}

	return nil
}

// getHVAState retrieves current HVA orbital state
func (m *Monitor) getHVAState() (*physics.OrbitalObject, error) {
	// In production: query from catalog or propagate from TLE
	// For now: placeholder
	return &physics.OrbitalObject{
		NoradID:    m.hvaID,
		S2CellID:   0,
		AltitudeKm: 408, // ISS altitude
		Timestamp:  time.Now(),
	}, nil
}

// reduceViaDMD applies DMD mode reduction for debris storm
// Converts 1000 debris fragments → 10 dominant modes
func (m *Monitor) reduceViaDMD(candidates []ConjunctionCandidate) []ConjunctionCandidate {
	// Would integrate with ai/v3/dmd_debris_cloud.py
	// For now: simple spatial clustering
	if len(candidates) <= 10 {
		return candidates
	}

	// Return first 10 as "mode centroids" (simplified)
	// Full implementation would use DMD mode extraction
	return candidates[:10]
}

// calculatePc computes Probability of Collision
func (m *Monitor) calculatePc(hva *physics.OrbitalObject, candidate ConjunctionCandidate) float64 {
	// High-fidelity Pc using Foster-Alfano or similar
	// This is the expensive physics calculation

	// Placeholder: simplified geometric Pc
	distance := candidate.S2Distance // km

	// Typical satellite cross-section ~ 10m radius
	// Pc ~ (combined_area) / (4 * pi * distance^2)
	areaHVA := 10.0   // m^2
	areaThreat := 1.0 // m^2 (debris)
	combinedRadius := math.Sqrt((areaHVA + areaThreat) / math.Pi)

	// Convert to probability (simplified 2D)
	if distance < combinedRadius/1000.0 {
		return 1.0 // Certain collision
	}

	// Inverse square law approximation
	pc := (combinedRadius * combinedRadius) / (4 * math.Pi * distance * distance * 1e6)

	return math.Min(pc, 1.0)
}

// calculateUrgency maps Pc to urgency scale (1-10)
func (m *Monitor) calculateUrgency(pc float64) int32 {
	switch {
	case pc > 1e-2: // 1 in 100
		return 10 // Critical
	case pc > 1e-3: // 1 in 1000
		return 8 // High
	case pc > 1e-4: // 1 in 10000
		return 6 // Elevated
	case pc > 1e-5:
		return 4 // Moderate
	default:
		return 2 // Low
	}
}

// handleThreats coordinates evasion response
func (m *Monitor) handleThreats(ctx context.Context, threats []ThreatContext, hva *physics.OrbitalObject) error {
	policyStart := time.Now()

	// Find highest threat
	maxThreat := threats[0]
	for _, t := range threats {
		if t.Pc > maxThreat.Pc {
			maxThreat = t
		}
	}

	log.Printf("[MONITOR] %d threats detected, max Pc: %.2e (urgency: %d)",
		len(threats), maxThreat.Pc, maxThreat.Urgency)

	// Request RL policy decision
	req := &PolicyRequest{
		HvaID:     int32(m.hvaID),
		Threats:   threats,
		Timestamp: time.Now(),
	}

	resp, err := m.policy.RequestAction(ctx, req)
	if err != nil {
		log.Printf("[MONITOR] Policy request failed: %v", err)
		// Fallback: standalone decision
		resp = m.fallbackDecision(maxThreat)
	}

	m.policyLatency = time.Since(policyStart)

	// Execute evasion if required
	if resp.Action == ActionExecute {
		m.executeEvasion(resp, maxThreat)
		m.generateSITREP(threats, resp)
		m.broadcastSpotReport(maxThreat, hva)
	}

	return nil
}

// fallbackDecision provides autonomous decision when RL unavailable
func (m *Monitor) fallbackDecision(threat ThreatContext) *PolicyResponse {
	return &PolicyResponse{
		Action:     ActionExecute,
		Confidence: 0.7,
		DeltaV:     [3]float64{0.5, 0, 0}, // Along-track burn
		BurnTime:   30 * time.Second,
		Rationale:  "Autonomous: Pc exceeds threshold, no RL available",
	}
}

// executeEvasion sends commands to propulsion system
func (m *Monitor) executeEvasion(resp *PolicyResponse, threat ThreatContext) {
	log.Printf("[MONITOR] ⚠️  EVASION EXECUTED")
	log.Printf("[MONITOR]   Action: %v", resp.Action)
	log.Printf("[MONITOR]   ΔV: [%.3f, %.3f, %.3f] m/s",
		resp.DeltaV[0], resp.DeltaV[1], resp.DeltaV[2])
	log.Printf("[MONITOR]   Burn: %v", resp.BurnTime)
	log.Printf("[MONITOR]   Rationale: %s", resp.Rationale)

	// TODO: Send to propulsion controller via Reticulum/Sideband
	m.lastAlert = time.Now()
}

// generateSITREP creates situation report for mesh broadcast
func (m *Monitor) generateSITREP(threats []ThreatContext, resp *PolicyResponse) {
	maxPc := 0.0
	for _, t := range threats {
		if t.Pc > maxPc {
			maxPc = t.Pc
		}
	}

	sitrep := SITREP{
		Timestamp:   time.Now(),
		HvaID:       int32(m.hvaID),
		ThreatCount: len(threats),
		MaxPc:       maxPc,
		ActionTaken: resp.Rationale,
		OperatorID:  runtime.GOARCH, // System identifier
	}

	// Marshal and broadcast over mesh
	log.Printf("[SITREP] Generated: %+v", sitrep)
	// mesh.Broadcast(sitrep)
}

// broadcastSpotReport sends immediate threat alert
func (m *Monitor) broadcastSpotReport(threat ThreatContext, hva *physics.OrbitalObject) {
	// Prevent alert spam
	if time.Since(m.lastAlert) < m.alertCooldown {
		return
	}

	spot := SpotReport{
		Timestamp: time.Now(),
		HvaID:     int32(m.hvaID),
		ThreatID:  threat.ObjectID,
		Pc:        threat.Pc,
		TCA:       time.Now().Add(10 * time.Minute), // Estimated
		Urgency:   fmt.Sprintf("URGENT-%d", threat.Urgency),
		GridRef:   fmt.Sprintf("S2-%d", hva.S2CellID),
	}

	log.Printf("[SPOT] ⚠️  IMMEDIATE THREAT: NORAD-%d Pc=%.2e",
		threat.ObjectID, threat.Pc)
	log.Printf("[SPOT] Broadcast: %+v", spot)

	// mesh.BroadcastPriority(spot)
	m.lastAlert = time.Now()
}

// GetMetrics returns performance statistics
func (m *Monitor) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"loop_count":        m.loopCount,
		"hva_id":            m.hvaID,
		"threshold":         m.threshold,
		"s2_latency_ms":     m.s2Latency.Seconds() * 1000,
		"prop_latency_ms":   m.propLatency.Seconds() * 1000,
		"policy_latency_ms": m.policyLatency.Seconds() * 1000,
		"target_total_ms":   TargetTotalLatency.Seconds() * 1000,
	}
}
