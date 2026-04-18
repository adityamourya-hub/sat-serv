package fusion

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// SensorType identifies the source sensor for an observation.
type SensorType int

const (
	SensorRadar   SensorType = iota // ATeena phased-array radar
	SensorSAR                       // RISAT SAR/ISAR
	SensorOptical                   // CCD optical telescope
	SensorNavIC                     // NavIC GNSS pseudorange
)

func (s SensorType) String() string {
	switch s {
	case SensorRadar:
		return "ATeena"
	case SensorSAR:
		return "RISAT-SAR"
	case SensorOptical:
		return "Optical-CCD"
	case SensorNavIC:
		return "NavIC"
	default:
		return "Unknown"
	}
}

// SensorObservation is a single observation from any sensor type.
type SensorObservation struct {
	SensorType   SensorType         `json:"sensor_type"`
	EpochJD      float64            `json:"epoch_jd"`
	Features     map[string]float64 `json:"features"`
	PositionECI  [3]float64         `json:"position_eci,omitempty"`
	Confidence   float64            `json:"confidence"`
	StationID    string             `json:"station_id,omitempty"`
}

// FusedState is the output of multi-sensor fusion.
type FusedState struct {
	SemiMajorAxis float64    `json:"a_km"`
	Eccentricity  float64    `json:"eccentricity"`
	Inclination   float64    `json:"inclination_rad"`
	RAAN          float64    `json:"raan_rad"`
	ArgPeriapsis  float64    `json:"arg_periapsis_rad"`
	MeanAnomaly   float64    `json:"mean_anomaly_rad"`
	Covariance    [6][6]float64 `json:"covariance"`
	Confidence    float64    `json:"confidence"`
	SensorCount   int        `json:"sensor_count"`
	SensorTypes   []string   `json:"sensor_types"`
}

// WeightedFuser performs weighted averaging fusion of multi-sensor observations.
// This serves as the Go-side classical fusion baseline; the Python GNN provides
// the neural fusion path.
type WeightedFuser struct {
	mu            sync.Mutex
	sensorWeights map[SensorType]float64
}

// NewWeightedFuser creates a fuser with default sensor weights.
func NewWeightedFuser() *WeightedFuser {
	return &WeightedFuser{
		sensorWeights: map[SensorType]float64{
			SensorRadar:   1.0,
			SensorSAR:     0.8,
			SensorOptical: 0.6,
			SensorNavIC:   0.4,
		},
	}
}

// SetWeight updates the fusion weight for a sensor type.
func (f *WeightedFuser) SetWeight(s SensorType, w float64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sensorWeights[s] = w
}

// Fuse combines multiple sensor observations into a single orbital state estimate.
func (f *WeightedFuser) Fuse(observations []SensorObservation) (FusedState, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(observations) == 0 {
		return FusedState{}, fmt.Errorf("no observations to fuse")
	}

	// Sort by epoch for temporal ordering
	sort.Slice(observations, func(i, j int) bool {
		return observations[i].EpochJD < observations[j].EpochJD
	})

	// Weighted feature accumulation
	featureSum := make(map[string]float64)
	weightSum := make(map[string]float64)
	totalWeight := 0.0
	sensorSet := make(map[SensorType]bool)

	for _, obs := range observations {
		w := f.sensorWeights[obs.SensorType] * obs.Confidence
		sensorSet[obs.SensorType] = true
		totalWeight += w
		for k, v := range obs.Features {
			featureSum[k] += v * w
			weightSum[k] += w
		}
	}

	// Compute weighted averages
	fused := make(map[string]float64)
	for k, s := range featureSum {
		if weightSum[k] > 0 {
			fused[k] = s / weightSum[k]
		}
	}

	// Build sensor type list
	types := make([]string, 0, len(sensorSet))
	for st := range sensorSet {
		types = append(types, st.String())
	}

	// Confidence: geometric mean of individual confidences, boosted by sensor diversity
	confProduct := 1.0
	for _, obs := range observations {
		confProduct *= math.Max(obs.Confidence, 0.01)
	}
	diversity := math.Min(float64(len(sensorSet))/4.0, 1.0) // Max 4 sensor types
	confidence := math.Pow(confProduct, 1.0/float64(len(observations))) * (0.7 + 0.3*diversity)

	return FusedState{
		SemiMajorAxis: fused["a_km"],
		Eccentricity:  fused["eccentricity"],
		Inclination:   fused["inclination_rad"],
		RAAN:          fused["raan_rad"],
		ArgPeriapsis:  fused["arg_periapsis_rad"],
		MeanAnomaly:   fused["mean_anomaly_rad"],
		Confidence:    confidence,
		SensorCount:   len(observations),
		SensorTypes:   types,
	}, nil
}
