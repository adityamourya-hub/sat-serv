package threat

import (
	"fmt"
	"math"
	"sort"
	"time"
)

/**
 * Bayesian Conflict Resolver for Threat Identification
 * 
 * Handles ambiguous L/D ratios (e.g., L/D=1.95 could be Ballistic RV or DF-ZF HGV)
 * using probabilistic evidence fusion instead of binary matching.
 * 
 * Formula: P(Threat_i | Data) ∝ P(Data | Threat_i) · P(Threat_i)
 * 
 * Key Advantage: Maintains "Belief State" that evolves with sliding window SINDy data.
 * Prevents "Target Flipping" when adversary mimics different vehicle types.
 * 
 * Reference: Bayesian Hypothesis Testing for Real-Time Classification
 */

// ConflictResolver performs probabilistic threat identification
type ConflictResolver struct {
	library      *ThreatLibrary
	sigma        float64           // Uncertainty of SINDy discovery (L/D std dev)
	priors       map[string]float64 // P(Threat_i) - regional prevalence
	history      []BeliefState      // Temporal belief evolution
	maxHistory   int                // Sliding window size
}

// BeliefState represents the probabilistic classification at a time instant
type BeliefState struct {
	Timestamp    time.Time
	Hypotheses   []Hypothesis
	PrimaryID    string
	PrimaryConf  float64
	SecondaryID  string
	SecondaryConf float64
	Entropy      float64  // Uncertainty measure
}

// Hypothesis represents a single threat possibility with probability
type Hypothesis struct {
	Profile     *ThreatProfile
	Probability float64  // Posterior probability
	Likelihood  float64  // P(Data | Threat_i)
	Prior       float64  // P(Threat_i)
	Evidence    float64  // Bayes factor
}

// ConflictResult contains the full resolution output
type ConflictResult struct {
	PrimaryThreat   *ThreatProfile
	PrimaryProb     float64
	SecondaryThreat *ThreatProfile
	SecondaryProb   float64
	AllHypotheses   []Hypothesis
	BeliefState     BeliefState
	ConflictScore   float64  // 0=none, 1=maximum ambiguity
	Recommendation  string
}

// NewConflictResolver creates a Bayesian resolver with default priors
func NewConflictResolver(library *ThreatLibrary) *ConflictResolver {
	return &ConflictResolver{
		library:    library,
		sigma:      0.15,  // Default SINDy L/D uncertainty
		maxHistory: 100,   // Keep last 100 belief states
		priors:     buildDefaultPriors(library),
	}
}

// SetSigma updates the measurement uncertainty (e.g., from SINDy confidence)
func (cr *ConflictResolver) SetSigma(sigma float64) {
	cr.sigma = sigma
}

// UpdatePrior sets regional prevalence for a specific threat
// Used when INTEL reports increased DF-ZF activity in Siliguri corridor
func (cr *ConflictResolver) UpdatePrior(threatID string, prior float64) {
	cr.priors[threatID] = prior
	cr.normalizePriors()
}

// Resolve performs Bayesian hypothesis testing
// Returns ranked hypotheses with posterior probabilities
func (cr *ConflictResolver) Resolve(
	discoveredLD float64,
	discoveredCD, discoveredCL float64,
	currentAlt, currentVel float64,
	sindyConfidence float64,
) *ConflictResult {
	
	// Adjust sigma based on SINDy confidence
	// High confidence → lower sigma, Low confidence → higher sigma
	effectiveSigma := cr.sigma * (1.5 - sindyConfidence) // 0.5× to 1.5× adjustment
	if effectiveSigma < 0.05 {
		effectiveSigma = 0.05
	}
	
	// Calculate Mach number
	mach := currentVel / 340.0
	
	// Build hypotheses
	hypotheses := make([]Hypothesis, 0, len(cr.library.Threats))
	totalWeight := 0.0
	
	for i := range cr.library.Threats {
		threat := &cr.library.Threats[i]
		
		// 1. Calculate Likelihood: Gaussian centered on library L/D midpoint
		ldMidpoint := (threat.Aerodynamics.LDRatioMin + threat.Aerodynamics.LDRatioMax) / 2.0
		likelihood := gaussianLikelihood(discoveredLD, ldMidpoint, effectiveSigma)
		
		// 2. Altitude Gating: Zero out impossible altitudes
		if currentAlt < threat.Aerodynamics.GlideAltitudeMin || 
		   currentAlt > threat.Aerodynamics.GlideAltitudeMax {
			likelihood = 0.0
		}
		
		// 3. Mach Gating: Check if Mach capability exceeded
		if mach > threat.Aerodynamics.MaxMach*1.2 { // 20% tolerance
			likelihood *= 0.1 // Penalty, not zero (measurement uncertainty)
		}
		
		// 4. Cd/Cl refinement (if available)
		if discoveredCD > 0 && discoveredCL > 0 {
			calculatedLD := discoveredCL / discoveredCD
			expectedLDMin := threat.Aerodynamics.LDRatioMin
			expectedLDMax := threat.Aerodynamics.LDRatioMax
			
			// Bonus if calculated L/D matches expected range
			if calculatedLD >= expectedLDMin && calculatedLD <= expectedLDMax {
				likelihood *= 1.2 // 20% confidence boost
			} else {
				likelihood *= 0.8 // 20% confidence penalty
			}
		}
		
		// 5. Get Prior probability
		prior := cr.priors[threat.ID]
		if prior == 0 {
			prior = 1.0 / float64(len(cr.library.Threats)) // Uniform default
		}
		
		// 6. Calculate Posterior ∝ Likelihood × Prior
		posterior := likelihood * prior
		
		hypotheses = append(hypotheses, Hypothesis{
			Profile:     threat,
			Probability: posterior,
			Likelihood:  likelihood,
			Prior:       prior,
			Evidence:    likelihood, // For Bayes factor calculation
		})
		
		totalWeight += posterior
	}
	
	// 7. Normalize probabilities
	if totalWeight > 0 {
		for i := range hypotheses {
			hypotheses[i].Probability /= totalWeight
		}
	} else {
		// No viable hypotheses - uniform distribution
		uniformProb := 1.0 / float64(len(hypotheses))
		for i := range hypotheses {
			hypotheses[i].Probability = uniformProb
		}
	}
	
	// 8. Sort by probability (descending)
	sort.Slice(hypotheses, func(i, j int) bool {
		return hypotheses[i].Probability > hypotheses[j].Probability
	})
	
	// 9. Calculate entropy (uncertainty measure)
	entropy := calculateEntropy(hypotheses)
	
	// 10. Build belief state
	belief := BeliefState{
		Timestamp:     time.Now(),
		Hypotheses:    hypotheses,
		PrimaryID:     hypotheses[0].Profile.ID,
		PrimaryConf:   hypotheses[0].Probability,
		Entropy:       entropy,
	}
	
	if len(hypotheses) > 1 {
		belief.SecondaryID = hypotheses[1].Profile.ID
		belief.SecondaryConf = hypotheses[1].Probability
	}
	
	// 11. Update history
	cr.history = append(cr.history, belief)
	if len(cr.history) > cr.maxHistory {
		cr.history = cr.history[1:] // Drop oldest
	}
	
	// 12. Calculate conflict score
	conflictScore := cr.calculateConflictScore(hypotheses)
	
	// 13. Generate recommendation
	recommendation := cr.generateRecommendation(hypotheses, conflictScore)
	
	return &ConflictResult{
		PrimaryThreat:   hypotheses[0].Profile,
		PrimaryProb:     hypotheses[0].Probability,
		SecondaryThreat: nil,
		SecondaryProb:   0.0,
		AllHypotheses:   hypotheses,
		BeliefState:     belief,
		ConflictScore:   conflictScore,
		Recommendation:  recommendation,
	}
}

// TemporalResolve uses belief history for smoother classification
// Reduces "Target Flipping" when adversary mimics different types
func (cr *ConflictResolver) TemporalResolve(
	currentLD, currentAlt, currentVel float64,
	sindyConfidence float64,
	windowSize int,
) *ConflictResult {
	
	// If not enough history, use single-frame resolve
	if len(cr.history) < 3 {
		return cr.Resolve(currentLD, 0, 0, currentAlt, currentVel, sindyConfidence)
	}
	
	// Use recent history for temporal smoothing
	recentHistory := cr.history
	if len(recentHistory) > windowSize {
		recentHistory = recentHistory[len(recentHistory)-windowSize:]
	}
	
	// Calculate temporal consistency score
	consistencyScore := cr.calculateTemporalConsistency(recentHistory)
	
	// Get current resolution
	result := cr.Resolve(currentLD, 0, 0, currentAlt, currentVel, sindyConfidence)
	
	// If temporal consistency is high, boost confidence in consistent ID
	if consistencyScore > 0.8 && len(recentHistory) > 5 {
		// Check if primary ID has been consistent
		consistentID := cr.getConsistentID(recentHistory)
		if consistentID != "" && consistentID == result.PrimaryThreat.ID {
			result.PrimaryProb = math.Min(result.PrimaryProb*1.2, 0.99)
			result.Recommendation += " (Temporally Stable)"
		}
	}
	
	return result
}

// gaussianLikelihood calculates P(Data | Threat) using Gaussian distribution
func gaussianLikelihood(x, mu, sigma float64) float64 {
	if sigma <= 0 {
		sigma = 0.01 // Prevent division by zero
	}
	
	exponent := -math.Pow(x-mu, 2) / (2 * math.Pow(sigma, 2))
	normalization := 1.0 / (sigma * math.Sqrt(2*math.Pi))
	
	return normalization * math.Exp(exponent)
}

// calculateEntropy measures classification uncertainty
// H = -Σ p_i × log(p_i) ... higher = more uncertain
func calculateEntropy(hypotheses []Hypothesis) float64 {
	entropy := 0.0
	for _, h := range hypotheses {
		if h.Probability > 0 {
			entropy -= h.Probability * math.Log2(h.Probability)
		}
	}
	return entropy
}

// calculateConflictScore measures ambiguity between top hypotheses
// 0 = no conflict (clear winner), 1 = maximum conflict (tie)
func (cr *ConflictResolver) calculateConflictScore(hypotheses []Hypothesis) float64 {
	if len(hypotheses) < 2 {
		return 0.0
	}
	
	p1 := hypotheses[0].Probability
	p2 := hypotheses[1].Probability
	
	// Conflict is high when top two are close in probability
	if p1+p2 == 0 {
		return 0.0
	}
	
	ratio := p2 / p1
	if ratio > 0.9 {
		return 1.0 // Near tie
	}
	if ratio < 0.3 {
		return 0.0 // Clear separation
	}
	
	// Linear interpolation for intermediate values
	return (ratio - 0.3) / 0.6
}

// generateRecommendation provides tactical guidance based on conflict level
func (cr *ConflictResolver) generateRecommendation(hypotheses []Hypothesis, conflictScore float64) string {
	primary := hypotheses[0]
	
	if conflictScore > 0.7 {
		return fmt.Sprintf("HIGH CONFLICT: Monitor both %s (%.0f%%) and %s (%.0f%%)",
			primary.Profile.CommonName, primary.Probability*100,
			hypotheses[1].Profile.CommonName, hypotheses[1].Probability*100)
	}
	
	if primary.Probability > 0.9 {
		return fmt.Sprintf("HIGH CONFIDENCE: Engage as %s (%.0f%%)",
			primary.Profile.CommonName, primary.Probability*100)
	}
	
	if primary.Probability > 0.7 {
		return fmt.Sprintf("MODERATE CONFIDENCE: Prepare for %s (%.0f%%), monitor alternatives",
			primary.Profile.CommonName, primary.Probability*100)
	}
	
	return fmt.Sprintf("LOW CONFIDENCE: %s (%.0f%%) - gather more data",
		primary.Profile.CommonName, primary.Probability*100)
}

// calculateTemporalConsistency checks how stable classifications have been
func (cr *ConflictResolver) calculateTemporalConsistency(history []BeliefState) float64 {
	if len(history) < 2 {
		return 0.0
	}
	
	changes := 0
	lastID := ""
	
	for _, state := range history {
		if lastID != "" && state.PrimaryID != lastID {
			changes++
		}
		lastID = state.PrimaryID
	}
	
	// Consistency = 1 - (changes / possible changes)
	possibleChanges := float64(len(history) - 1)
	if possibleChanges == 0 {
		return 1.0
	}
	
	return 1.0 - (float64(changes) / possibleChanges)
}

// getConsistentID returns the ID that appears most frequently in history
func (cr *ConflictResolver) getConsistentID(history []BeliefState) string {
	counts := make(map[string]int)
	
	for _, state := range history {
		counts[state.PrimaryID]++
	}
	
	maxCount := 0
	mostFrequent := ""
	
	for id, count := range counts {
		if count > maxCount {
			maxCount = count
			mostFrequent = id
		}
	}
	
	// Only return if it's dominant (>60% of history)
	if float64(maxCount) > float64(len(history))*0.6 {
		return mostFrequent
	}
	return ""
}

// GetBeliefHistory returns the temporal belief evolution
func (cr *ConflictResolver) GetBeliefHistory() []BeliefState {
	return cr.history
}

// normalizePriors ensures priors sum to 1
func (cr *ConflictResolver) normalizePriors() {
	total := 0.0
	for _, p := range cr.priors {
		total += p
	}
	
	if total > 0 {
		for id := range cr.priors {
			cr.priors[id] /= total
		}
	}
}

// buildDefaultPriors creates uniform priors if none specified
func buildDefaultPriors(library *ThreatLibrary) map[string]float64 {
	priors := make(map[string]float64)
	uniform := 1.0 / float64(len(library.Threats))
	
	for _, threat := range library.Threats {
		// Slightly higher prior for known regional threats
		if threat.CountryCode == "CHN" || threat.CountryCode == "RUS" {
			priors[threat.ID] = uniform * 1.5
		} else {
			priors[threat.ID] = uniform
		}
	}
	
	return priors
}

// GetCurrentBelief returns the most recent belief state
func (cr *ConflictResolver) GetCurrentBelief() *BeliefState {
	if len(cr.history) == 0 {
		return nil
	}
	return &cr.history[len(cr.history)-1]
}

// ClearHistory resets the temporal belief tracking
func (cr *ConflictResolver) ClearHistory() {
	cr.history = make([]BeliefState, 0)
}

// Example Usage:
// resolver := threat.NewConflictResolver(library)
// result := resolver.Resolve(1.95, 0.35, 0.80, 45.0, 2500.0, 0.85)
// fmt.Printf("Primary: %s (%.1f%%)\n", result.PrimaryThreat.CommonName, result.PrimaryProb*100)
