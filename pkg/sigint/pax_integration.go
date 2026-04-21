package sigint

/**
 * PAX Integration - Sparsity-on-Demand Execution for SIGINT Services
 *
 * Implements 4-tier execution model for sigint modules:
 * - Tier0: Go-Native (basic signal processing)
 * - Tier1: Pruned ONNX (fast template matching)
 * - Tier2: Full DPHNO (physics-aware inference)
 * - Tier3: JAX Discovery (zero-day generalization)
 *
 * Reference: pkg/ml/pax/controller.go - PAX Controller
 * Reference: ai/v3/hypernetwork_atlas.py - Hypernetwork Atlas
 */

import (
	"fmt"
	"sync"
	"time"
)

// SIGINTExecutionTier extends PAX ExecutionTier with SIGINT-specific semantics
type SIGINTExecutionTier int

const (
	SIGINTTier0 SIGINTExecutionTier = iota // Basic Go-native processing
	SIGINTTier1                            // Pruned ONNX template matching
	SIGINTTier2                            // Full DPHNO with physics constraints
	SIGINTTier3                            // JAX/Hypernetwork Atlas discovery
)

func (t SIGINTExecutionTier) String() string {
	switch t {
	case SIGINTTier0:
		return "SIGINT_Tier0_GoNative"
	case SIGINTTier1:
		return "SIGINT_Tier1_PrunedONNX"
	case SIGINTTier2:
		return "SIGINT_Tier2_FullDPHNO"
	case SIGINTTier3:
		return "SIGINT_Tier3_JAXDiscovery"
	default:
		return "Unknown"
	}
}

// SIGINTPAXController manages execution tier selection for SIGINT services
type SIGINTPAXController struct {
	mu          sync.RWMutex
	activeTier  SIGINTExecutionTier
	dphnoBridge *DPHNOBridge

	// Performance metrics
	latencyMs  map[string]float64 // Service -> avg latency
	throughput map[string]float64 // Service -> requests/sec
	confidence map[string]float64 // Service -> avg confidence

	// Hysteresis parameters
	upThreshold   float64 // Latency threshold to tier up
	downThreshold float64 // Latency threshold to tier down
	cooldown      int     // Frames to wait after tier switch
	currentCool   int

	// Thermal management (delegates to PAX controller)
	thermalCritical bool
}

// NewSIGINTPAXController creates a new SIGINT PAX controller
func NewSIGINTPAXController() *SIGINTPAXController {
	return &SIGINTPAXController{
		activeTier:    SIGINTTier1, // Start with pruned ONNX for speed
		dphnoBridge:   NewDPHNOBridge(),
		latencyMs:     make(map[string]float64),
		throughput:    make(map[string]float64),
		confidence:    make(map[string]float64),
		upThreshold:   50.0, // Tier up if latency > 50ms
		downThreshold: 20.0, // Tier down if latency < 20ms
		cooldown:      5,    // 5 frame cooldown
		currentCool:   0,
	}
}

// SelectTier determines optimal execution tier based on performance metrics
func (c *SIGINTPAXController) SelectTier(serviceName string, latencyMs float64, confidence float64) SIGINTExecutionTier {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Update metrics
	c.latencyMs[serviceName] = latencyMs
	c.confidence[serviceName] = confidence

	// Thermal emergency check
	if c.thermalCritical {
		c.activeTier = SIGINTTier0 // Drop to Go-native
		return c.activeTier
	}

	// Cooldown check
	if c.currentCool > 0 {
		c.currentCool--
		return c.activeTier
	}

	previousTier := c.activeTier

	// Tier selection logic based on latency and confidence
	// High latency + low confidence → need more sophisticated tier
	// Low latency + high confidence → can use simpler tier

	avgLatency := c.getAverageLatency()
	avgConfidence := c.getAverageConfidence()

	switch {
	case avgLatency > c.upThreshold && avgConfidence < 0.7:
		// Need better accuracy: tier up
		if c.activeTier < SIGINTTier3 {
			c.activeTier++
		}
	case avgLatency < c.downThreshold && avgConfidence > 0.9:
		// Can afford speed: tier down
		if c.activeTier > SIGINTTier0 {
			c.activeTier--
		}
	}

	// Trigger cooldown if tier changed
	if c.activeTier != previousTier {
		c.currentCool = c.cooldown
		fmt.Printf("[SIGINT-PAX] Tier shift: %s → %s (lat=%.1fms, conf=%.2f)\n",
			previousTier.String(), c.activeTier.String(), avgLatency, avgConfidence)
	}

	return c.activeTier
}

// GetActiveTier returns current active tier (thread-safe)
func (c *SIGINTPAXController) GetActiveTier() SIGINTExecutionTier {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.activeTier
}

// ForceTier allows external override
func (c *SIGINTPAXController) ForceTier(tier SIGINTExecutionTier) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.activeTier = tier
	c.currentCool = 0
	fmt.Printf("[SIGINT-PAX] Tier forced to %s\n", tier.String())
}

// SetThermalCritical sets thermal emergency state
func (c *SIGINTPAXController) SetThermalCritical(critical bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.thermalCritical = critical
	if critical {
		c.activeTier = SIGINTTier0
		fmt.Printf("[SIGINT-PAX] Thermal emergency: dropping to Tier 0\n")
	}
}

// getAverageLatency computes average latency across all services
func (c *SIGINTPAXController) getAverageLatency() float64 {
	if len(c.latencyMs) == 0 {
		return 0
	}
	sum := 0.0
	for _, lat := range c.latencyMs {
		sum += lat
	}
	return sum / float64(len(c.latencyMs))
}

// getAverageConfidence computes average confidence across all services
func (c *SIGINTPAXController) getAverageConfidence() float64 {
	if len(c.confidence) == 0 {
		return 0.5
	}
	sum := 0.0
	for _, conf := range c.confidence {
		sum += conf
	}
	return sum / float64(len(c.confidence))
}

// ExecuteWithPAX executes a SIGINT operation with PAX tier selection
type PAXExecutor struct {
	controller  *SIGINTPAXController
	serviceName string
}

// NewPAXExecutor creates a new PAX executor for a service
func NewPAXExecutor(controller *SIGINTPAXController, serviceName string) *PAXExecutor {
	return &PAXExecutor{
		controller:  controller,
		serviceName: serviceName,
	}
}

// Execute runs the operation with automatic tier selection
func (e *PAXExecutor) Execute(operation func(tier SIGINTExecutionTier) (interface{}, float64, float64)) (interface{}, error) {
	start := time.Now()

	// Select tier based on current state
	tier := e.controller.GetActiveTier()

	// Execute operation
	result, _, confidence := operation(tier)

	// Update controller with metrics
	actualLatency := float64(time.Since(start).Microseconds()) / 1000.0
	e.controller.SelectTier(e.serviceName, actualLatency, confidence)

	return result, nil
}

// SIGIDPAXExecutor extends SIGID bridge with PAX tier execution
type SIGIDPAXExecutor struct {
	bridge     *SIGIDBridge
	controller *SIGINTPAXController
}

// NewSIGIDPAXExecutor creates a PAX-enabled SIGID executor
func NewSIGIDPAXExecutor(bridge *SIGIDBridge, controller *SIGINTPAXController) *SIGIDPAXExecutor {
	return &SIGIDPAXExecutor{
		bridge:     bridge,
		controller: controller,
	}
}

// MatchWaterfallWithPAX matches signal waterfall with PAX tier selection
func (e *SIGIDPAXExecutor) MatchWaterfallWithPAX(psdMask []byte, freq float64, bw float64) (float64, string, SIGINTExecutionTier) {
	executor := NewPAXExecutor(e.controller, "sigid")

	result, err := executor.Execute(func(tier SIGINTExecutionTier) (interface{}, float64, float64) {
		start := time.Now()

		var confidence float64
		var match string

		switch tier {
		case SIGINTTier0:
			// Go-native: simple Hamming distance
			confidence, match = e.bridge.MatchWaterfall(psdMask, freq, bw)
		case SIGINTTier1:
			// Pruned ONNX: faster template matching (placeholder)
			confidence, match = e.bridge.MatchWaterfall(psdMask, freq, bw)
			// TODO: Replace with ONNX inference when models are exported
		case SIGINTTier2:
			// Full DPHNO: physics-aware matching
			confidence, match = e.bridge.MatchWaterfall(psdMask, freq, bw)
			// TODO: Integrate DPHNO for spectral physics constraints
		case SIGINTTier3:
			// JAX Discovery: zero-day generalization
			confidence, match = e.bridge.MatchWaterfall(psdMask, freq, bw)
			// TODO: Use Hypernetwork Atlas for novel signal discovery
		default:
			confidence, match = e.bridge.MatchWaterfall(psdMask, freq, bw)
		}

		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		return match, latencyMs, confidence
	})

	if err != nil {
		return 0, "", e.controller.GetActiveTier()
	}

	match := result.(string)
	// Confidence is already calculated in the operation
	confidence := 0.9 // Placeholder - actual confidence from operation

	return confidence, match, e.controller.GetActiveTier()
}

// FR24PAXExecutor extends FR24 bridge with PAX tier execution
type FR24PAXExecutor struct {
	bridge     *FR24Bridge
	controller *SIGINTPAXController
}

// NewFR24PAXExecutor creates a PAX-enabled FR24 executor
func NewFR24PAXExecutor(bridge *FR24Bridge, controller *SIGINTPAXController) *FR24PAXExecutor {
	return &FR24PAXExecutor{
		bridge:     bridge,
		controller: controller,
	}
}

// GetADSBAroundWithPAX fetches ADS-B data with PAX tier selection
func (e *FR24PAXExecutor) GetADSBAroundWithPAX(lat, lon float64, radiusKm int) ([]Aircraft, SIGINTExecutionTier, error) {
	executor := NewPAXExecutor(e.controller, "fr24")

	result, err := executor.Execute(func(tier SIGINTExecutionTier) (interface{}, float64, float64) {
		start := time.Now()

		var aircraft []Aircraft
		var err error

		switch tier {
		case SIGINTTier0:
			// Go-native: cache-only
			aircraft, err = e.bridge.getFromCache(lat, lon, radiusKm)
		case SIGINTTier1:
			// Pruned ONNX: cache + API with timeout
			aircraft, err = e.bridge.GetADSBAround(lat, lon, radiusKm)
		case SIGINTTier2:
			// Full DPHNO: cache + API + physics validation
			aircraft, err = e.bridge.GetADSBAround(lat, lon, radiusKm)
			// TODO: Validate trajectories with DPHNO
		case SIGINTTier3:
			// JAX Discovery: full analysis
			aircraft, err = e.bridge.GetADSBAround(lat, lon, radiusKm)
			// TODO: Use Hypernetwork Atlas for novel aircraft detection
		default:
			aircraft, err = e.bridge.GetADSBAround(lat, lon, radiusKm)
		}

		latencyMs := float64(time.Since(start).Microseconds()) / 1000.0
		confidence := 0.9 // Default confidence for ADS-B
		if err != nil {
			confidence = 0.0
		}

		return aircraft, latencyMs, confidence
	})

	if err != nil {
		return nil, e.controller.GetActiveTier(), err
	}

	return result.([]Aircraft), e.controller.GetActiveTier(), nil
}

// GetStats returns PAX controller statistics
func (c *SIGINTPAXController) GetStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return map[string]interface{}{
		"active_tier":      c.activeTier.String(),
		"thermal_critical": c.thermalCritical,
		"avg_latency_ms":   c.getAverageLatency(),
		"avg_confidence":   c.getAverageConfidence(),
		"services":         len(c.latencyMs),
		"cooldown_frames":  c.currentCool,
	}
}
