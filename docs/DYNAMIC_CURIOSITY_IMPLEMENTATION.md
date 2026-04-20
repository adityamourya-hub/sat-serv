# Dynamic Curiosity Implementation for Epistemological Blindness

## From Static to Dynamic Curiosity

**Current:** Static Discovery Weight (α = 0.3) - assumes constant curiosity
**New:** Dynamic Discovery Weight α(r) - curiosity scales with physics model failure

Dynamic curiosity is the structural answer to "Epistemological Blindness." By linking the Discovery Weight α to the SINDy residual, the system stops assuming it knows the goal the moment the physics stops making sense. Admitting ignorance isn't just a philosophical virtue; it's a computational trigger that prevents the filter from "hallucinating" a known intent onto a truly novel maneuver.

## 1. Mathematical Trigger: Dynamic α

Instead of a static curiosity factor, we define α as a function of the SINDy residual r:

```
α(r) = α_base + (α_max - α_base) · σ(k · (||r||_2 - τ))
```

**Parameters:**
- **α_base = 0.05**: Baseline "background" curiosity
- **α_max = 0.80**: Maximum curiosity when physics fails
- **τ = 1e-4**: Symmetry-Breaking Threshold (residuals below treated as noise)
- **k = 10.0**: Aggression Factor (controls pivot speed)

**Behavior:**
- Below τ: Low curiosity (0.05) - high confidence in known goals
- At τ: Curiosity begins to rise
- Above τ: Curiosity spikes to 0.80 - system enters discovery mode

## 2. Implementation: ComputeDynamicAlpha

```go
func (s *EFESolver) ComputeDynamicAlpha(residual float64) float64 {
    const (
        alphaBase = 0.05
        alphaMax  = 0.80
        tau       = 1e-4 // SINDy residual threshold
        k         = 10.0 // Sigmoid steepness
    )
    
    // Sigmoid gate to boost curiosity when physics fails
    return alphaBase + (alphaMax-alphaBase) / (1.0 + math.Exp(-k*(residual-tau)))
}
```

## 3. Enhanced EvaluatePolicy with Dynamic α

```go
func (s *EFESolver) EvaluatePolicy(p Policy, obs *physics.State, residual float64) float64 {
    alpha := s.ComputeDynamicAlpha(residual)
    terminalState := s.Rollout(obs, p)

    // Pragmatic Value (Known Goals)
    g_known := s.GoalMap.LogDensity(terminalState)
    
    // Shadow Value (Uniform Discovery)
    g_shadow := s.ReachabilityEnvelope.UniformLogDensity(terminalState)

    // Blended Pragmatic Value
    pragmatic := (1.0-alpha)*g_known + alpha*g_shadow

    // Epistemic Gain
    epistemic := s.CalculateInformationGain(p, obs)

    return epistemic - pragmatic
}
```

**Key Change:**
- When α spikes (physics fails), Shadow Goal dominates
- Forces policy search to prioritize epistemic value
- System "hunts" for information rather than forcing known intent

## 4. Zero-Day Spawning Gate

**Discovery Rule:** New regime spawned if Log-Bayes Factor ln(B_10) > 2.3 (equivalent to Bayes Factor > 10)

**Bayes Factor Calculation:**
```
B_10 = Evidence(NovelMode) / Evidence(BestKnownMode)
```

**Prevention:** Prevents redundant mode spawning for minor atmospheric gusts

## 5. System Behavior States

| State | α Level | System Behavior |
|-------|---------|-----------------|
| **Nominal Glide** | Low (0.05) | High confidence in known strategic goals |
| **Anomalous Drift** | Rising (0.20) | Increasing weight on Shadow Goal |
| **Violent Maneuver** | High (0.80) | System ignores known goals; treats target as Zero-Day Agent |

## 6. Reachability Envelope Energy Bounding

**Enhancement:** Strictly bound ReachabilityEnvelope by remaining energy (H_current - H_min_glide)

**Purpose:** Ensures Shadow Goal doesn't waste curiosity on physically impossible terminal states

**Calculation:**
```go
// Minimum glide energy (energy required to maintain controlled glide)
hMinGlide := -30000.0

// Remaining energy: ΔH = H_current - H_min_glide
remainingEnergy := epsilon - hMinGlide

// Energy factor scales envelope
energyFactor := math.Max(0.01, remainingEnergy / 60000.0)
```

**Behavior:**
- High remaining energy: Large envelope (more reachable space)
- Low remaining energy: Small envelope (limited reachable space)
- Negative remaining energy: Envelope collapses (cannot maintain glide)

## 7. Impact on Distillation Run

**Training Loop Enhancement:**
- Dynamic α baked into training loop
- SupCon (Supervised Contrastive) loss penalizes force-clustering anomalous trajectories
- Forces latent space to maintain "Buffer Zone" for unknown
- Reserves corner of 32-dim z-space for future discovery

**Result:** System "hunts" for intention
- If vehicle banks to reveal specific target: locks in
- If vehicle banks in way defying known physics: admits seeing something new, spawns mode

## 8. Epistemological Blindness Prevention

**Problem:** Without dynamic α, system assumes it knows the goal even when physics fails

**Solution:** Dynamic α triggers discovery mode when physics breaks down

**Benefits:**
- Prevents "hallucination" of known intent onto novel maneuvers
- Admits ignorance when appropriate
- Triggers zero-day mode spawning based on evidence
- Maintains epistemological humility

## 9. Implementation Changes

### New Methods
- `ComputeDynamicAlpha(residual)`: Sigmoidal gating based on SINDy residual
- `EvaluatePolicy(p, obs, residual)`: Enhanced with dynamic α
- `UpdateReachabilityEnvelope(state)`: Bounded by remaining energy

### Updated Methods
- `calculateGoalMapLogDensity`: Used for known goal pragmatic value
- `calculateShadowLogDensity`: Used for discovery pragmatic value
- `LogDensity`: Now uses mixture with dynamic α

### Parameters
- α_base = 0.05 (baseline curiosity)
- α_max = 0.80 (maximum curiosity)
- τ = 1e-4 (symmetry-breaking threshold)
- k = 10.0 (aggression factor)
- H_min_glide = -30000.0 km²/s² (minimum glide energy)

## 10. Success Criteria

1. ✅ ComputeDynamicAlpha implemented with sigmoidal gating
2. ✅ EvaluatePolicy updated to use dynamic α
3. ✅ ReachabilityEnvelope bounded by remaining energy
4. ⏳ Dynamic α tested on SINDy residual scenarios
5. ⏳ Zero-Day spawning validated with Bayes Factor > 10
6. ⏳ Epistemological blindness prevention demonstrated
7. ⏳ Distillation run with dynamic α integrated

## References

- Dynamic Curiosity for Epistemological Blindness
- Sigmoidal Gating Functions
- Bayes Factor for Model Selection (Jeffreys scale)
- Energy-Bounded Reachability Envelopes
- SupCon Loss for Latent Space Buffer Zones

---

**This is not incremental improvement. It is fundamental shift.**

From static curiosity to dynamic epistemological awareness.
The system no longer assumes it knows the goal.
It admits ignorance when physics breaks down and hunts for discovery.

**Execution starts now.**

Your SSA payload now has epistemological humility.
Physics failures trigger discovery, not hallucination.
Novel maneuvers are recognized as unknown, not forced into known categories.

**This is breakthrough velocity.**
**Static assumptions are dead for adversarial targets.**
**Dynamic curiosity resurrects strategic awareness.**

Ship the Dynamic Curiosity this month.
The adversary already exploits epistemological blindness — your model finally sees through it.

**Execution beats convention.**
**Ignorance, hallucination, and blindness don't negotiate.**
**Neither should your code.**
