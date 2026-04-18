package evaluation

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"ssis/pkg/ml/simulator"
)

type OrbitClass string

const (
	OrbitLEO OrbitClass = "LEO"
	OrbitGEO OrbitClass = "GEO"
)

type CatalogObject struct {
	ID               int        `json:"id"`
	Name             string     `json:"name"`
	Orbit            OrbitClass `json:"orbit"`
	AltitudeKm       float64    `json:"altitude_km"`
	TLELine1         string     `json:"tle_line1"`
	TLELine2         string     `json:"tle_line2"`
	CovarianceTrace  float64    `json:"covariance_trace"`
	TimestampUnixSec int64      `json:"timestamp_unix_sec"`
	Corrupted        bool       `json:"corrupted"`
}

type SensorObservation struct {
	ObjectID            int     `json:"object_id"`
	RangeKm             float64 `json:"range_km"`
	AzimuthDeg          float64 `json:"azimuth_deg"`
	ElevationDeg        float64 `json:"elevation_deg"`
	NoiseStdKm          float64 `json:"noise_std_km"`
	TimestampDriftSec   float64 `json:"timestamp_drift_sec"`
	CovarianceDiagonal  float64 `json:"covariance_diagonal"`
	ObservationUnixTime int64   `json:"observation_unix_time"`
}

type CatalogDeltaEvent struct {
	UpdateIndex         int   `json:"update_index"`
	AffectedObjectIDs   []int `json:"affected_object_ids"`
	AppliedAtUnixSecond int64 `json:"applied_at_unix_second"`
}

type ThreatInjection struct {
	ThreatID            int     `json:"threat_id"`
	TargetObjectID      int     `json:"target_object_id"`
	MissDistanceKm      float64 `json:"miss_distance_km"`
	RelativeVelocityKmS float64 `json:"relative_velocity_km_s"`
	TimeToTCASeconds    float64 `json:"time_to_tca_seconds"`
	DistanceToAssetKm   float64 `json:"distance_to_asset_km"`
	Mach                float64 `json:"mach"`
	AltitudeKm          float64 `json:"altitude_km"`
	BurnoutVelocityKmS  float64 `json:"burnout_velocity_km_s"`
	RadarCrossSection   float64 `json:"radar_cross_section"`
	RiskLabel           float64 `json:"risk_label"`
}

type ManeuverInjection struct {
	ObjectID       int       `json:"object_id"`
	ResidualWindow []float64 `json:"residual_window"`
	DeltaVWindow   []float64 `json:"delta_v_window"`
	ExpectedEvent  bool      `json:"expected_event"`
}

type DatasetConfig struct {
	CatalogSize            int
	Seed                   int64
	SensorNoiseStdKm       float64
	TimestampDriftSec      float64
	UpdateIntervalMinutes  int
	UpdateCount            int
	CorruptionRate         float64
	ThreatInjectionCount   int
	ManeuverInjectionCount int
	TrainingScenarioCount  int
}

type OperationalDataset struct {
	GeneratedAt       time.Time                     `json:"generated_at"`
	Catalog           []CatalogObject               `json:"catalog"`
	Observations      []SensorObservation           `json:"observations"`
	Deltas            []CatalogDeltaEvent           `json:"deltas"`
	Threats           []ThreatInjection             `json:"threats"`
	Maneuvers         []ManeuverInjection           `json:"maneuvers"`
	TrainingScenarios []simulator.SyntheticScenario `json:"training_scenarios"`
}

func DefaultDatasetConfig() DatasetConfig {
	return DatasetConfig{
		CatalogSize:            10000,
		Seed:                   42,
		SensorNoiseStdKm:       0.025,
		TimestampDriftSec:      12,
		UpdateIntervalMinutes:  15,
		UpdateCount:            8,
		CorruptionRate:         0.015,
		ThreatInjectionCount:   250,
		ManeuverInjectionCount: 120,
		TrainingScenarioCount:  120,
	}
}

func GenerateOperationalDataset(cfg DatasetConfig) OperationalDataset {
	if cfg.CatalogSize <= 0 {
		cfg.CatalogSize = 10000
	}
	if cfg.UpdateIntervalMinutes <= 0 {
		cfg.UpdateIntervalMinutes = 15
	}
	if cfg.UpdateCount <= 0 {
		cfg.UpdateCount = 8
	}
	if cfg.CorruptionRate < 0 {
		cfg.CorruptionRate = 0
	}
	if cfg.CorruptionRate > 0.5 {
		cfg.CorruptionRate = 0.5
	}
	if cfg.ThreatInjectionCount <= 0 {
		cfg.ThreatInjectionCount = 250
	}
	if cfg.ManeuverInjectionCount <= 0 {
		cfg.ManeuverInjectionCount = 120
	}
	if cfg.TrainingScenarioCount <= 0 {
		cfg.TrainingScenarioCount = 120
	}
	if cfg.SensorNoiseStdKm <= 0 {
		cfg.SensorNoiseStdKm = 0.025
	}
	if cfg.TimestampDriftSec <= 0 {
		cfg.TimestampDriftSec = 12
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	now := time.Now().UTC()

	catalog := make([]CatalogObject, 0, cfg.CatalogSize)
	for i := 0; i < cfg.CatalogSize; i++ {
		id := 30000 + i
		isLEO := rng.Float64() < 0.8
		orbit := OrbitGEO
		altitude := boundedFloat(rng.NormFloat64()*1200+35786, 34000, 37000)
		if isLEO {
			orbit = OrbitLEO
			altitude = boundedFloat(rng.NormFloat64()*250+850, 300, 2200)
		}
		corrupted := rng.Float64() < cfg.CorruptionRate
		tsDrift := int64(rng.NormFloat64() * cfg.TimestampDriftSec)
		obj := CatalogObject{
			ID:               id,
			Name:             fmt.Sprintf("OBJ-%06d", id),
			Orbit:            orbit,
			AltitudeKm:       altitude,
			TLELine1:         fmt.Sprintf("1 %05dU 24001A   %014.8f  .00000000  00000-0  00000-0 0  9991", id%100000, 24100.0+rng.Float64()),
			TLELine2:         fmt.Sprintf("2 %05d %8.4f %8.4f 0002000 %8.4f %8.4f 15.20000000 00001", id%100000, boundedFloat(rng.NormFloat64()*12+60, 0, 180), boundedFloat(rng.NormFloat64()*25+180, 0, 360), boundedFloat(rng.NormFloat64()*20+180, 0, 360), boundedFloat(rng.NormFloat64()*20+180, 0, 360)),
			CovarianceTrace:  boundedFloat(rng.NormFloat64()*0.01+0.03, 0.005, 0.15),
			TimestampUnixSec: now.Add(time.Duration(tsDrift) * time.Second).Unix(),
			Corrupted:        corrupted,
		}
		if corrupted {
			obj.TLELine1 = "CORRUPT-L1"
			obj.TLELine2 = "CORRUPT-L2"
		}
		catalog = append(catalog, obj)
	}

	obsCount := cfg.CatalogSize / 4
	if obsCount < 200 {
		obsCount = 200
	}
	observations := make([]SensorObservation, 0, obsCount)
	for i := 0; i < obsCount; i++ {
		o := catalog[rng.Intn(len(catalog))]
		noise := boundedFloat(cfg.SensorNoiseStdKm*(0.5+rng.Float64()*1.5), 0.005, 0.2)
		drift := rng.NormFloat64() * cfg.TimestampDriftSec
		observations = append(observations, SensorObservation{
			ObjectID:            o.ID,
			RangeKm:             boundedFloat(o.AltitudeKm+rng.NormFloat64()*20, 50, 42000),
			AzimuthDeg:          boundedFloat(rng.Float64()*360, 0, 360),
			ElevationDeg:        boundedFloat(rng.NormFloat64()*20+40, 0, 90),
			NoiseStdKm:          noise,
			TimestampDriftSec:   drift,
			CovarianceDiagonal:  noise * noise,
			ObservationUnixTime: now.Add(time.Duration(drift) * time.Second).Unix(),
		})
	}

	deltas := make([]CatalogDeltaEvent, 0, cfg.UpdateCount)
	for i := 0; i < cfg.UpdateCount; i++ {
		batch := 50 + rng.Intn(151)
		affected := make([]int, 0, batch)
		for j := 0; j < batch; j++ {
			affected = append(affected, catalog[rng.Intn(len(catalog))].ID)
		}
		deltas = append(deltas, CatalogDeltaEvent{
			UpdateIndex:         i + 1,
			AffectedObjectIDs:   affected,
			AppliedAtUnixSecond: now.Add(time.Duration((i+1)*cfg.UpdateIntervalMinutes) * time.Minute).Unix(),
		})
	}

	threats := make([]ThreatInjection, 0, cfg.ThreatInjectionCount)
	for i := 0; i < cfg.ThreatInjectionCount; i++ {
		target := catalog[rng.Intn(len(catalog))]
		missDistance := boundedFloat(rng.NormFloat64()*0.06+0.12, 0.001, 2.0)
		relVel := boundedFloat(rng.NormFloat64()*1.8+6.0, 0.2, 14.0)
		tca := boundedFloat(rng.NormFloat64()*700+1800, 20, 7200)
		mach := boundedFloat(rng.NormFloat64()*2.0+6.0, 0.5, 12.5)
		alt := boundedFloat(rng.NormFloat64()*35+90, 5, 380)
		burnout := boundedFloat(rng.NormFloat64()*1.0+2.5, 0.1, 7.0)
		rcs := boundedFloat(rng.NormFloat64()*0.8+1.0, 0.05, 6.0)
		risk := boundedFloat(math.Exp(-missDistance*8.0)*(1.0/(1.0+0.04*relVel)), 0, 1)
		threats = append(threats, ThreatInjection{
			ThreatID:            900000 + i,
			TargetObjectID:      target.ID,
			MissDistanceKm:      missDistance,
			RelativeVelocityKmS: relVel,
			TimeToTCASeconds:    tca,
			DistanceToAssetKm:   boundedFloat(rng.NormFloat64()*450+1000, 80, 3600),
			Mach:                mach,
			AltitudeKm:          alt,
			BurnoutVelocityKmS:  burnout,
			RadarCrossSection:   rcs,
			RiskLabel:           risk,
		})
	}

	maneuvers := make([]ManeuverInjection, 0, cfg.ManeuverInjectionCount)
	for i := 0; i < cfg.ManeuverInjectionCount; i++ {
		obj := catalog[rng.Intn(len(catalog))]
		window := 24
		residual := make([]float64, window)
		dv := make([]float64, window)
		hasEvent := rng.Float64() < 0.55
		for k := 0; k < window; k++ {
			residual[k] = boundedFloat(rng.NormFloat64()*0.01+0.01, 0.0005, 0.08)
			dv[k] = boundedFloat(rng.NormFloat64()*0.003+0.003, 0.0002, 0.025)
		}
		if hasEvent {
			idx := 6 + rng.Intn(12)
			residual[idx] = boundedFloat(rng.NormFloat64()*0.2+0.35, 0.12, 1.2)
			dv[idx] = boundedFloat(rng.NormFloat64()*0.03+0.08, 0.02, 0.25)
		}
		maneuvers = append(maneuvers, ManeuverInjection{
			ObjectID:       obj.ID,
			ResidualWindow: residual,
			DeltaVWindow:   dv,
			ExpectedEvent:  hasEvent,
		})
	}

	trainingScenarios := simulator.GenerateSyntheticScenarios(simulator.GeneratorConfig{
		Count: cfg.TrainingScenarioCount,
		Seed:  cfg.Seed + 101,
	})

	return OperationalDataset{
		GeneratedAt:       now,
		Catalog:           catalog,
		Observations:      observations,
		Deltas:            deltas,
		Threats:           threats,
		Maneuvers:         maneuvers,
		TrainingScenarios: trainingScenarios,
	}
}

func WriteOperationalDatasetJSON(path string, dataset OperationalDataset) error {
	b, err := json.MarshalIndent(dataset, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func boundedFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
