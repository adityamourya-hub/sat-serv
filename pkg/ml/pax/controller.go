package pax

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"sync"
	"time"
)

// Policy represents a sequence of modes over a horizon (placeholder for integration with ai_imm)
type Policy struct {
	ModeSequence []int
	Horizon      int
	Probability  float64
}

// PAXConfig holds runtime thresholds loaded from JSON config file
type PAXConfig struct {
	Thermal struct {
		UpshiftThreshold    float64 `json:"upshift_threshold"`
		DownshiftThreshold  float64 `json:"downshift_threshold"`
		CooldownDurationSec float64 `json:"cooldown_duration_sec"`
	} `json:"thermal"`

	TierSwitching struct {
		UpThreshold    float64 `json:"up_threshold"`
		DownThreshold  float64 `json:"down_threshold"`
		CooldownFrames int     `json:"cooldown_frames"`
	} `json:"tier_switching"`

	Conflict struct {
		ConflictThreshold float64 `json:"conflict_threshold"`
		ResolutionTimer   int     `json:"resolution_timer"`
		ConflictSafe      float64 `json:"conflict_safe"`
	} `json:"conflict"`

	ModeProbabilities struct {
		BallisticThreshold  float64 `json:"ballistic_threshold"`
		HGVGlideThreshold   float64 `json:"hgv_glide_threshold"`
		ManeuverThreshold   float64 `json:"maneuver_threshold"`
		ManeuverR2Threshold float64 `json:"maneuver_r2_threshold"`
	} `json:"mode_probabilities"`

	ForestRuth struct {
		Gamma float64 `json:"gamma"`
	} `json:"forest_ruth"`

	HotReload struct {
		Enabled     bool   `json:"enabled"`
		IntervalSec int    `json:"interval_sec"`
		ConfigPath  string `json:"config_path"`
	} `json:"hot_reload"`
}

// ExecutionTier represents the complexity level of the physics engine
type ExecutionTier int

const (
	Tier0 ExecutionTier = iota // Go-Native (Ballistic)
	Tier1                      // INT8 ONNX (Stable Glide)
	Tier2                      // FP32 Lie-PHS (Active Maneuver)
	Tier3                      // JAX Discovery (Zero-Day)
)

func (t ExecutionTier) String() string {
	switch t {
	case Tier0:
		return "Tier0_GoNative"
	case Tier1:
		return "Tier1_PrunedONNX"
	case Tier2:
		return "Tier2_FullDPHNO"
	case Tier3:
		return "Tier3_JAXDiscovery"
	default:
		return "Unknown"
	}
}

// Controller implements the PAX policy layer with thread-safe tier selection
type Controller struct {
	mu         sync.RWMutex
	Governor   *ThermalGovernor
	ActiveTier ExecutionTier
	Config     *PAXConfig // Runtime configuration loaded from TOML
	configPath string     // Path to config file for hot-reload

	// Hysteresis parameters (loaded from config)
	UpThreshold    float64 // R² threshold to shift UP (Tier 1 → Tier 2)
	DownThreshold  float64 // R² threshold to shift DOWN (Tier 2 → Tier 1)
	CooldownFrames int     // Frames to wait after gear shift

	// ADITI 4.0: Thermal Schmidt Triggering
	thermalCooldownStart time.Time // When thermal cooldown began
	cooldownDurationSec  float64   // Cooldown duration in seconds (default 30s)
	currentCool          int

	// ADITI 4.0: Forest-Ruth coefficients (shared across all tiers)
	// Ensures thermal-manifold synchronicity - same integrator coefficients
	// regardless of tier complexity. Only H-net complexity scales.
	ForestRuthGamma float64   // γ = 1/(2-2^(1/3)) = 1.35120719195966
	ForestRuthA     []float64 // Position coefficients
	ForestRuthB     []float64 // Momentum coefficients

	// ADITI 4.0: Conflict Metric for Tier 3 triggering
	ConflictThreshold float64 // JSD threshold for conflict alarm (default 0.5)
	ResolutionTimer   int     // Frames to stay in Tier 3 for resolution (default 50 = 500ms @ 100Hz)
	ConflictSafe      float64 // Safe threshold below which intent is clear (default 0.2)
}

// NewPAXController creates a new PAX controller with default config
func NewPAXController() *Controller {
	return NewPAXControllerWithConfig("configs/pax_config.json")
}

// NewPAXControllerWithConfig creates a new PAX controller loading from specified config path
func NewPAXControllerWithConfig(configPath string) *Controller {
	c := &Controller{
		Governor:             NewThermalGovernor(),
		ActiveTier:           Tier1,
		configPath:           configPath,
		currentCool:          0,
		thermalCooldownStart: time.Time{},
	}

	// Set defaults first
	c.setDefaults()

	// Load config from file (overwrites defaults if successful)
	if err := c.LoadConfig(configPath); err != nil {
		fmt.Printf("[PAX] Failed to load config from %s, using defaults: %v\n", configPath, err)
	}

	return c
}

// LoadConfig loads thresholds from JSON config file
func (c *Controller) LoadConfig(configPath string) error {
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}

	var config PAXConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to decode config: %w", err)
	}

	c.mu.Lock()
	c.Config = &config
	c.configPath = configPath

	// Apply config values
	c.UpThreshold = config.TierSwitching.UpThreshold
	c.DownThreshold = config.TierSwitching.DownThreshold
	c.CooldownFrames = config.TierSwitching.CooldownFrames
	c.cooldownDurationSec = config.Thermal.CooldownDurationSec
	c.ConflictThreshold = config.Conflict.ConflictThreshold
	c.ResolutionTimer = config.Conflict.ResolutionTimer
	c.ConflictSafe = config.Conflict.ConflictSafe

	// Forest-Ruth coefficients
	gamma := config.ForestRuth.Gamma
	c.ForestRuthGamma = gamma
	c.ForestRuthA = []float64{gamma / 2, (1 - gamma) / 2, (1 - gamma) / 2, gamma / 2}
	c.ForestRuthB = []float64{gamma, 1 - 2*gamma, gamma, 0}

	c.mu.Unlock()

	fmt.Printf("[PAX] Config loaded from %s\n", configPath)
	return nil
}

// ReloadConfig hot-reloads configuration from file
func (c *Controller) ReloadConfig() error {
	if c.configPath == "" {
		return fmt.Errorf("no config path set")
	}
	return c.LoadConfig(c.configPath)
}

// setDefaults sets default values when config loading fails
func (c *Controller) setDefaults() {
	// Forest-Ruth (Yoshida 4th-order) coefficients
	gamma := 1.35120719195966 // 1/(2-2^(1/3))
	a := []float64{gamma / 2, (1 - gamma) / 2, (1 - gamma) / 2, gamma / 2}
	b := []float64{gamma, 1 - 2*gamma, gamma, 0}

	c.UpThreshold = 0.70
	c.DownThreshold = 0.85
	c.CooldownFrames = 10
	c.cooldownDurationSec = 30.0
	c.ForestRuthGamma = gamma
	c.ForestRuthA = a
	c.ForestRuthB = b
	c.ConflictThreshold = 0.5
	c.ResolutionTimer = 50
	c.ConflictSafe = 0.2
}

// SelectTier determines the optimal complexity for the next frame
// Thread-safe: uses RWMutex for concurrent access
// Implements hysteresis to prevent gear-hunting
func (c *Controller) SelectTier(probs map[string]float64, sindyR2 float64) ExecutionTier {
	c.mu.Lock()
	defer c.mu.Unlock()

	// 1. Thermal Emergency (Overrides Hysteresis)
	if c.Governor.IsCritical() {
		c.ActiveTier = Tier1
		fmt.Printf("[PAX] Thermal Critical (level=%d). Shedding complexity to Tier 1.\n", c.Governor.Level)
		return Tier1
	}

	// 2. ADITI 4.0: Thermal Schmidt Triggering
	// Prevent gear-hunt oscillation at thermal boundary
	// Up-shift (T1 → T2): Only if Temp < 72°C (hysteresis buffer)
	// Down-shift (T2 → T1): Immediately if Temp > 80°C (safety margin)
	// Cooldown: Lock in Tier 1 for 30s after thermal down-shift

	temp := c.Governor.Temperature()
	upshiftThreshold := 72.0   // °C
	downshiftThreshold := 80.0 // °C

	if c.ActiveTier == Tier1 && temp > upshiftThreshold {
		// Prevent up-shift if too hot
		fmt.Printf("[PAX] Thermal block: Temp=%.1f°C > %.1f°C (upshift blocked)\n", temp, upshiftThreshold)
		return Tier1
	}

	if c.ActiveTier == Tier2 && temp > downshiftThreshold {
		// Immediate down-shift for safety
		c.ActiveTier = Tier1
		c.thermalCooldownStart = time.Now()
		fmt.Printf("[PAX] Thermal down-shift: Temp=%.1f°C > %.1f°C (cooldown 30s)\n", temp, downshiftThreshold)
		return Tier1
	}

	// Check thermal cooldown lock
	if c.thermalCooldownStart != (time.Time{}) {
		cooldownElapsed := time.Since(c.thermalCooldownStart).Seconds()
		if cooldownElapsed < 30.0 {
			// Still in cooldown
			return Tier1
		} else {
			// Cooldown expired
			c.thermalCooldownStart = time.Time{}
		}
	}

	// 3. Cooldown Logic: Prevent rapid switching
	if c.currentCool > 0 {
		c.currentCool--
		return c.ActiveTier
	}

	previousTier := c.ActiveTier

	// 4. Gear Shifting Logic
	if probs["ballistic"] > 0.92 {
		c.ActiveTier = Tier0
	} else if probs["hgv_glide"] > 0.5 {
		// Hysteresis: Use different thresholds for shifting up vs down
		if c.ActiveTier == Tier1 && sindyR2 < c.UpThreshold {
			c.ActiveTier = Tier2
		} else if c.ActiveTier == Tier2 && sindyR2 > c.DownThreshold {
			c.ActiveTier = Tier1
		}
	} else if probs["maneuver"] > 0.4 && sindyR2 < 0.5 {
		c.ActiveTier = Tier3
	}

	// 5. Trigger Cooldown if we shifted gears
	if c.ActiveTier != previousTier {
		c.currentCool = c.CooldownFrames
		fmt.Printf("[PAX] Gear shift: %s → %s (cooldown %d frames)\n", previousTier.String(), c.ActiveTier.String(), c.CooldownFrames)
	}

	return c.ActiveTier
}

// GetForestRuthCoefficients returns the Forest-Ruth coefficients.
// ADITI 4.0: All tiers must share these coefficients to ensure
// thermal-manifold synchronicity. Only H-net complexity scales.
func (c *Controller) GetForestRuthCoefficients() (float64, []float64, []float64) {
	return c.ForestRuthGamma, c.ForestRuthA, c.ForestRuthB
}

// GetActiveTier returns the current active tier (thread-safe)
func (c *Controller) GetActiveTier() ExecutionTier {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.ActiveTier
}

// ForceTier allows external override (e.g., manual command, safety override)
func (c *Controller) ForceTier(tier ExecutionTier) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ActiveTier = tier
	c.currentCool = 0 // Reset cooldown for immediate switching
	fmt.Printf("[PAX] Tier forced to %s\n", tier.String())
}

// EvaluateConflict implements the Conflict Metric for Tier 3 triggering
//
// ADITI 4.0: When GoalMap suggests heading for known target but Shadow Goal suggests
// novel maneuver, the system enters High Epistemic Tension. This triggers Tier 3
// (Full DPHNO Scan) to resolve ambiguity before tracking collapse.
//
// Conflict Metric: C = JSD(π_GoalMap || π_Shadow)
// - If C < τ_safe (0.2): Intent clear, continue Tier 2 (EFE Reasoning)
// - If C > τ_alarm (0.5): Ambiguity too high, trigger Tier 3 (Full DPHNO Scan)
func (c *Controller) EvaluateConflict(goalPolicy, shadowPolicy Policy) float64 {
	// Placeholder: Calculate JSD between policy distributions
	// In production, this would call the EFESolver.CalculateConflictMetric method
	// For now, return a mock conflict value based on policy differences

	// Simple heuristic: conflict based on mode sequence difference
	conflict := 0.0
	if len(goalPolicy.ModeSequence) != len(shadowPolicy.ModeSequence) {
		conflict = 0.8 // High conflict for different horizons
	} else {
		for i := 0; i < len(goalPolicy.ModeSequence); i++ {
			if goalPolicy.ModeSequence[i] != shadowPolicy.ModeSequence[i] {
				conflict += 0.3 // Increment for each mode mismatch
			}
		}
	}

	// Trigger Tier 3 if conflict exceeds threshold
	if conflict > c.ConflictThreshold {
		c.mu.Lock()
		defer c.mu.Unlock()

		// AMBIGUITY DETECTED: Trigger Tier 3 Resolution
		// Overrides thermal governor for 500ms 'High-Fidelity' burst
		c.ActiveTier = Tier3
		c.ResolutionTimer = 50 // frames (500ms @ 100Hz)

		fmt.Printf("[PAX] Conflict Detected (%.4f > %.4f). Escalating to Tier 3 for resolution.\n",
			conflict, c.ConflictThreshold)
	}

	return conflict
}
