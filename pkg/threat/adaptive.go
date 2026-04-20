// Package threat provides machine learning components for adaptive threshold updates
// Implements online learning for dynamic threshold adjustment based on environmental conditions
package threat

import (
	"math"
	"sync"
	"time"
)

// AdaptiveLearner provides online learning for threshold adaptation
type AdaptiveLearner struct {
	mu sync.RWMutex

	// Learning parameters
	LearningRate float64 // α for gradient descent
	DecayRate    float64 // For exponential moving average
	BatchSize    int     // Mini-batch size for updates

	// Feature statistics
	FeatureMean  []float64
	FeatureStd   []float64
	FeatureCount int

	// Threshold state
	BaseThresholds    map[string]float64   // Original configured thresholds
	AdaptedThresholds map[string]float64   // Adapted thresholds
	ThresholdHistory  map[string][]float64 // For trend analysis

	// Performance metrics
	TruePositives  int
	FalsePositives int
	TrueNegatives  int
	FalseNegatives int

	// Last update time
	LastUpdate time.Time
}

// NewAdaptiveLearner creates a new adaptive threshold learner
func NewAdaptiveLearner(baseThresholds map[string]float64) *AdaptiveLearner {
	return &AdaptiveLearner{
		LearningRate:      0.01,
		DecayRate:         0.95,
		BatchSize:         100,
		BaseThresholds:    baseThresholds,
		AdaptedThresholds: copyThresholds(baseThresholds),
		ThresholdHistory:  make(map[string][]float64),
		FeatureMean:       make([]float64, 6), // Position (3) + Velocity (3)
		FeatureStd:        make([]float64, 6),
		LastUpdate:        time.Now(),
	}
}

// ObservationFeatures extracts features from a launch observation
type ObservationFeatures struct {
	Altitude    float64
	VelocityMag float64
	Temperature float64
	Intensity   float64
	TimeOfDay   float64 // Normalized 0-1
	DayOfYear   float64 // Normalized 0-1
}

// ExtractFeatures creates feature vector from observation
func ExtractFeatures(obs LaunchObservation) []float64 {
	altitude := math.Sqrt(obs.Position[0]*obs.Position[0]+obs.Position[1]*obs.Position[1]+obs.Position[2]*obs.Position[2]) - 6371.0
	velocityMag := math.Sqrt(obs.Velocity[0]*obs.Velocity[0] + obs.Velocity[1]*obs.Velocity[1] + obs.Velocity[2]*obs.Velocity[2])

	now := obs.Timestamp
	hour := float64(now.Hour()) / 24.0
	dayOfYear := float64(now.YearDay()) / 366.0

	return []float64{
		altitude,
		velocityMag,
		obs.Temperature,
		obs.Intensity,
		hour,
		dayOfYear,
	}
}

// UpdateLearner processes an observation and updates adaptive thresholds
func (al *AdaptiveLearner) UpdateLearner(obs LaunchObservation, isThreat bool, predictedThreat bool) {
	al.mu.Lock()
	defer al.mu.Unlock()

	// Extract features
	features := ExtractFeatures(obs)

	// Update feature statistics
	al.updateFeatureStats(features)

	// Update confusion matrix
	if isThreat && predictedThreat {
		al.TruePositives++
	} else if !isThreat && predictedThreat {
		al.FalsePositives++
	} else if !isThreat && !predictedThreat {
		al.TrueNegatives++
	} else {
		al.FalseNegatives++
	}

	// Adapt thresholds based on performance
	al.adaptThresholds(features, isThreat, predictedThreat)

	al.LastUpdate = time.Now()
}

// updateFeatureStats updates running statistics for features
func (al *AdaptiveLearner) updateFeatureStats(features []float64) {
	al.FeatureCount++

	// Online mean update
	for i := range features {
		delta := features[i] - al.FeatureMean[i]
		al.FeatureMean[i] += delta / float64(al.FeatureCount)

		// Online variance (simplified)
		delta2 := features[i] - al.FeatureMean[i]
		al.FeatureStd[i] = math.Sqrt(al.FeatureStd[i]*al.FeatureStd[i]*(1.0-1.0/float64(al.FeatureCount)) + delta2*delta2/float64(al.FeatureCount))
	}
}

// adaptThresholds adjusts thresholds based on performance metrics
func (al *AdaptiveLearner) adaptThresholds(features []float64, isThreat bool, predictedThreat bool) {
	// Calculate current F1 score
	precision := 0.0
	recall := 0.0

	if al.TruePositives+al.FalsePositives > 0 {
		precision = float64(al.TruePositives) / float64(al.TruePositives+al.FalsePositives)
	}
	if al.TruePositives+al.FalseNegatives > 0 {
		recall = float64(al.TruePositives) / float64(al.TruePositives+al.FalseNegatives)
	}

	f1 := 0.0
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}

	// Adjust thresholds to optimize F1 score
	// If precision is low, increase thresholds (reduce false positives)
	// If recall is low, decrease thresholds (reduce false negatives)

	adjustmentFactor := 1.0

	// Use F1 score for more balanced adaptation
	if f1 < 0.8 {
		// Low F1 - need to balance precision and recall
		if precision < recall {
			adjustmentFactor = 1.0 + al.LearningRate*(0.8-precision)
		} else {
			adjustmentFactor = 1.0 - al.LearningRate*(0.8-recall)
		}
	}

	if precision < 0.8 && recall > 0.7 {
		// Too many false positives - increase thresholds
		adjustmentFactor = 1.0 + al.LearningRate*(0.8-precision)
	} else if recall < 0.7 && precision > 0.8 {
		// Too many false negatives - decrease thresholds
		adjustmentFactor = 1.0 - al.LearningRate*(0.7-recall)
	}

	// Apply adjustment with feature-based modulation
	// High altitude = more conservative (higher thresholds)
	altitude := features[0]
	altitudeModulation := 1.0
	if altitude > 50.0 {
		altitudeModulation = 1.1 // More conservative at high altitude
	} else if altitude < 30.0 {
		altitudeModulation = 0.9 // More aggressive at low altitude
	}

	finalAdjustment := adjustmentFactor * altitudeModulation

	// Update all thresholds
	for thresholdName, baseValue := range al.BaseThresholds {
		newValue := baseValue * finalAdjustment

		// Apply exponential smoothing
		currentValue := al.AdaptedThresholds[thresholdName]
		al.AdaptedThresholds[thresholdName] = al.DecayRate*currentValue + (1.0-al.DecayRate)*newValue

		// Track history
		al.ThresholdHistory[thresholdName] = append(al.ThresholdHistory[thresholdName], al.AdaptedThresholds[thresholdName])
		if len(al.ThresholdHistory[thresholdName]) > 100 {
			al.ThresholdHistory[thresholdName] = al.ThresholdHistory[thresholdName][1:]
		}
	}
}

// GetAdaptedThreshold returns the current adapted value for a threshold
func (al *AdaptiveLearner) GetAdaptedThreshold(thresholdName string) float64 {
	al.mu.RLock()
	defer al.mu.RUnlock()

	if value, exists := al.AdaptedThresholds[thresholdName]; exists {
		return value
	}
	return al.BaseThresholds[thresholdName]
}

// GetPerformanceMetrics returns current classification performance
func (al *AdaptiveLearner) GetPerformanceMetrics() map[string]interface{} {
	al.mu.RLock()
	defer al.mu.RUnlock()

	precision := 0.0
	recall := 0.0
	accuracy := 0.0

	total := al.TruePositives + al.FalsePositives + al.TrueNegatives + al.FalseNegatives
	if total > 0 {
		accuracy = float64(al.TruePositives+al.TrueNegatives) / float64(total)
	}

	if al.TruePositives+al.FalsePositives > 0 {
		precision = float64(al.TruePositives) / float64(al.TruePositives+al.FalsePositives)
	}
	if al.TruePositives+al.FalseNegatives > 0 {
		recall = float64(al.TruePositives) / float64(al.TruePositives+al.FalseNegatives)
	}

	f1 := 0.0
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}

	return map[string]interface{}{
		"true_positives":  al.TruePositives,
		"false_positives": al.FalsePositives,
		"true_negatives":  al.TrueNegatives,
		"false_negatives": al.FalseNegatives,
		"precision":       precision,
		"recall":          recall,
		"f1_score":        f1,
		"accuracy":        accuracy,
		"total_samples":   total,
		"feature_count":   al.FeatureCount,
	}
}

// Reset clears learning history
func (al *AdaptiveLearner) Reset() {
	al.mu.Lock()
	defer al.mu.Unlock()

	al.TruePositives = 0
	al.FalsePositives = 0
	al.TrueNegatives = 0
	al.FalseNegatives = 0
	al.FeatureCount = 0
	al.AdaptedThresholds = copyThresholds(al.BaseThresholds)
	al.ThresholdHistory = make(map[string][]float64)
	al.LastUpdate = time.Now()
}

// copyThresholds creates a deep copy of threshold map
func copyThresholds(original map[string]float64) map[string]float64 {
	copy := make(map[string]float64)
	for k, v := range original {
		copy[k] = v
	}
	return copy
}

// AdaptiveThresholdManager manages adaptive thresholds for the threat package
type AdaptiveThresholdManager struct {
	learner *AdaptiveLearner
}

// NewAdaptiveThresholdManager creates a new adaptive threshold manager
func NewAdaptiveThresholdManager() *AdaptiveThresholdManager {
	// Initialize with base thresholds from config
	baseThresholds := map[string]float64{
		"MachThreshold":          GetConfig().MachThreshold,
		"VerticalVelocityKmS":    GetConfig().VerticalVelocityKmS,
		"AltitudeBoostThreshold": GetConfig().AltitudeBoostThreshold,
		"PlumeTempMin":           GetConfig().PlumeTempMin,
		"FSMDistanceKm":          GetConfig().FSMDistanceKm,
	}

	return &AdaptiveThresholdManager{
		learner: NewAdaptiveLearner(baseThresholds),
	}
}

// ProcessObservation processes an observation through the adaptive learner
func (atm *AdaptiveThresholdManager) ProcessObservation(obs LaunchObservation, isThreat bool, predictedThreat bool) {
	atm.learner.UpdateLearner(obs, isThreat, predictedThreat)
}

// GetThreshold returns the current adaptive value for a threshold
func (atm *AdaptiveThresholdManager) GetThreshold(thresholdName string) float64 {
	return atm.learner.GetAdaptedThreshold(thresholdName)
}

// GetMetrics returns performance metrics
func (atm *AdaptiveThresholdManager) GetMetrics() map[string]interface{} {
	return atm.learner.GetPerformanceMetrics()
}

// Reset resets the adaptive learner
func (atm *AdaptiveThresholdManager) Reset() {
	atm.learner.Reset()
}
