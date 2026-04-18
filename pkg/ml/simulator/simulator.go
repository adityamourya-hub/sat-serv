package simulator

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type ScenarioType string

const (
	ScenarioConjunction ScenarioType = "conjunction"
	ScenarioRapidThreat ScenarioType = "rapid_threat"
	ScenarioASAT        ScenarioType = "asat_intercept"
	ScenarioProximity   ScenarioType = "proximity_ops"
	ScenarioReentry     ScenarioType = "reentry_watch"
)

type ThreatClass string

const (
	ThreatUnknown    ThreatClass = "unknown"
	ThreatOrbital    ThreatClass = "orbital"
	ThreatBallistic  ThreatClass = "ballistic"
	ThreatHypersonic ThreatClass = "hypersonic"
)

type SyntheticScenario struct {
	ScenarioID          string       `json:"scenario_id"`
	Type                ScenarioType `json:"type"`
	AssetID             int          `json:"asset_id"`
	ThreatID            int          `json:"threat_id"`
	ThreatClass         ThreatClass  `json:"threat_class"`
	MissDistanceKm      float64      `json:"miss_distance_km"`
	RelativeVelocityKmS float64      `json:"relative_velocity_km_s"`
	TimeToTCASeconds    float64      `json:"time_to_tca_seconds"`
	SigmaRadialKm       float64      `json:"sigma_radial_km"`
	SigmaIntrackKm      float64      `json:"sigma_intrack_km"`
	SigmaCrosstrackKm   float64      `json:"sigma_crosstrack_km"`
	HardbodyRadiusKm    float64      `json:"hardbody_radius_km"`
	DistanceToAssetKm   float64      `json:"distance_to_asset_km"`
	Mach                float64      `json:"mach"`
	AltitudeKm          float64      `json:"altitude_km"`
	BurnoutVelocityKmS  float64      `json:"burnout_velocity_km_s"`
	RadarCrossSection   float64      `json:"radar_cross_section"`
	RecommendedAction   string       `json:"recommended_action"`
	ScenarioRationale   []string     `json:"scenario_rationale,omitempty"`
	DecisionFactors     []string     `json:"decision_factors,omitempty"`
	AdversarialIntent   float64      `json:"adversarial_intent"`
	RiskLabel           float64      `json:"risk_label"`
	PriorityLabel       float64      `json:"priority_label"`
	CollisionAvoided    bool         `json:"collision_avoided"`
	OperatorDifficulty  float64      `json:"operator_difficulty"`
}

type GeneratorConfig struct {
	Count int
	Seed  int64
}

func DefaultGeneratorConfig() GeneratorConfig {
	return GeneratorConfig{Count: 10000, Seed: 42}
}

func GenerateSyntheticScenarios(cfg GeneratorConfig) []SyntheticScenario {
	if cfg.Count <= 0 {
		cfg.Count = 10000
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	out := make([]SyntheticScenario, 0, cfg.Count)

	for i := 0; i < cfg.Count; i++ {
		sType := sampleScenarioType(rng)

		missDistance := bounded(rng.NormFloat64()*0.04+0.08, 0.001, 1.2)
		relVel := bounded(rng.NormFloat64()*1.2+2.8, 0.05, 13.0)
		tca := bounded(rng.NormFloat64()*650+1800, 30, 7200)
		sigmaR := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		sigmaI := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		sigmaC := bounded(rng.NormFloat64()*0.008+0.02, 0.001, 0.09)
		hbr := bounded(rng.NormFloat64()*0.003+0.008, 0.002, 0.03)

		distanceToAsset := bounded(rng.NormFloat64()*450+900, 50, 3200)
		mach := bounded(rng.NormFloat64()*2.3+5.2, 0.4, 13.0)
		altitude := bounded(rng.NormFloat64()*35+85, 5, 350)
		burnout := bounded(rng.NormFloat64()*1.1+2.4, 0.1, 7.0)
		rcs := bounded(rng.NormFloat64()*0.8+1.1, 0.05, 6.0)

		class := classifyThreat(mach, altitude, burnout)
		missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs = applyScenarioProfile(
			rng,
			sType,
			missDistance,
			relVel,
			tca,
			distanceToAsset,
			mach,
			altitude,
			burnout,
			rcs,
		)
		class = classifyThreat(mach, altitude, burnout)
		risk := syntheticRisk(missDistance, hbr, sigmaR, sigmaI, sigmaC, relVel, tca)
		priority := syntheticPriority(class, distanceToAsset, mach, rcs)
		action := recommendedAction(class, priority, distanceToAsset)
		rationale, decisionFactors, intent := scenarioExplainability(sType, class, risk, priority, distanceToAsset, tca, altitude)
		difficulty := bounded(0.3+0.4*risk+0.3*(priority/300.0), 0.0, 1.0)

		out = append(out, SyntheticScenario{
			ScenarioID:          fmt.Sprintf("SIM-%06d", i+1),
			Type:                sType,
			AssetID:             43000 + rng.Intn(6000),
			ThreatID:            60000 + i,
			ThreatClass:         class,
			MissDistanceKm:      missDistance,
			RelativeVelocityKmS: relVel,
			TimeToTCASeconds:    tca,
			SigmaRadialKm:       sigmaR,
			SigmaIntrackKm:      sigmaI,
			SigmaCrosstrackKm:   sigmaC,
			HardbodyRadiusKm:    hbr,
			DistanceToAssetKm:   distanceToAsset,
			Mach:                mach,
			AltitudeKm:          altitude,
			BurnoutVelocityKmS:  burnout,
			RadarCrossSection:   rcs,
			RecommendedAction:   action,
			ScenarioRationale:   rationale,
			DecisionFactors:     decisionFactors,
			AdversarialIntent:   intent,
			RiskLabel:           risk,
			PriorityLabel:       priority,
			CollisionAvoided:    risk < 0.15 || action == "evasive_maneuver_now",
			OperatorDifficulty:  difficulty,
		})
	}

	return out
}

func sampleScenarioType(rng *rand.Rand) ScenarioType {
	r := rng.Float64()
	switch {
	case r < 0.32:
		return ScenarioConjunction
	case r < 0.60:
		return ScenarioRapidThreat
	case r < 0.75:
		return ScenarioASAT
	case r < 0.90:
		return ScenarioProximity
	default:
		return ScenarioReentry
	}
}

func applyScenarioProfile(
	rng *rand.Rand,
	sType ScenarioType,
	missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs float64,
) (float64, float64, float64, float64, float64, float64, float64, float64) {
	switch sType {
	case ScenarioASAT:
		missDistance = bounded(rng.NormFloat64()*0.03+0.05, 0.001, 0.8)
		relVel = bounded(rng.NormFloat64()*1.7+7.5, 1.8, 13.0)
		tca = bounded(rng.NormFloat64()*400+900, 20, 3600)
		distanceToAsset = bounded(rng.NormFloat64()*300+650, 40, 1800)
		mach = bounded(rng.NormFloat64()*1.8+9.5, 3.0, 13.0)
		altitude = bounded(rng.NormFloat64()*50+450, 120, 1100)
		burnout = bounded(rng.NormFloat64()*0.9+4.2, 1.2, 7.0)
		rcs = bounded(rng.NormFloat64()*0.5+0.7, 0.05, 3.0)
	case ScenarioProximity:
		missDistance = bounded(rng.NormFloat64()*0.09+0.35, 0.01, 3.0)
		relVel = bounded(rng.NormFloat64()*0.12+0.22, 0.02, 1.1)
		tca = bounded(rng.NormFloat64()*2600+8400, 600, 86400)
		distanceToAsset = bounded(rng.NormFloat64()*180+280, 10, 1300)
		mach = bounded(rng.NormFloat64()*0.8+2.1, 0.3, 5.5)
		altitude = bounded(rng.NormFloat64()*120+650, 180, 1800)
		burnout = bounded(rng.NormFloat64()*0.5+1.7, 0.2, 4.5)
		rcs = bounded(rng.NormFloat64()*1.1+2.8, 0.2, 8.0)
	case ScenarioReentry:
		missDistance = bounded(rng.NormFloat64()*0.6+4.0, 0.2, 15.0)
		relVel = bounded(rng.NormFloat64()*0.8+3.2, 0.6, 9.0)
		tca = bounded(rng.NormFloat64()*800+2600, 120, 9600)
		distanceToAsset = bounded(rng.NormFloat64()*500+1200, 120, 4200)
		mach = bounded(rng.NormFloat64()*1.1+6.2, 2.5, 12.5)
		altitude = bounded(rng.NormFloat64()*25+90, 15, 250)
		burnout = bounded(rng.NormFloat64()*0.7+2.2, 0.4, 5.5)
		rcs = bounded(rng.NormFloat64()*1.3+3.0, 0.2, 10.0)
	}
	return missDistance, relVel, tca, distanceToAsset, mach, altitude, burnout, rcs
}

func scenarioExplainability(
	sType ScenarioType,
	class ThreatClass,
	risk, priority, distanceToAsset, tca, altitude float64,
) ([]string, []string, float64) {
	rationale := []string{fmt.Sprintf("scenario=%s", sType)}
	factors := []string{
		fmt.Sprintf("risk=%.3f", risk),
		fmt.Sprintf("priority=%.1f", priority),
		fmt.Sprintf("distance_km=%.1f", distanceToAsset),
	}
	intent := 0.2

	switch sType {
	case ScenarioASAT:
		rationale = append(rationale, "high closure and short warning windows indicate interceptor profile")
		factors = append(factors, fmt.Sprintf("time_to_tca_sec=%.0f", tca))
		intent = bounded(0.7+0.25*(priority/300.0), 0.0, 1.0)
	case ScenarioProximity:
		rationale = append(rationale, "persistent close approach with low relative velocity indicates inspection or stalking")
		factors = append(factors, fmt.Sprintf("altitude_km=%.1f", altitude))
		intent = bounded(0.35+0.4*(1.0-math.Min(distanceToAsset/1500.0, 1.0)), 0.0, 1.0)
	case ScenarioReentry:
		rationale = append(rationale, "descending trajectory suggests controlled or uncontrolled re-entry window")
		factors = append(factors, fmt.Sprintf("class=%s", class))
		intent = bounded(0.15+0.25*(priority/300.0), 0.0, 1.0)
	default:
		rationale = append(rationale, "standard conjunction and rapid-threat watchlist flow")
		intent = bounded(0.1+0.35*(priority/300.0), 0.0, 1.0)
	}

	return rationale, factors, intent
}

func WriteScenariosJSONL(path string, scenarios []SyntheticScenario) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, s := range scenarios {
		if err := enc.Encode(s); err != nil {
			return err
		}
	}
	return nil
}

func WriteScenariosJSON(path string, scenarios []SyntheticScenario) error {
	payload := struct {
		Scenarios []SyntheticScenario `json:"scenarios"`
	}{Scenarios: scenarios}

	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

func bounded(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func classifyThreat(mach, altitude, burnout float64) ThreatClass {
	if mach >= 5 && altitude >= 20 && altitude <= 100 {
		return ThreatHypersonic
	}
	if burnout >= 1.0 && altitude < 300 {
		return ThreatBallistic
	}
	if altitude >= 120 {
		return ThreatOrbital
	}
	return ThreatUnknown
}

func syntheticRisk(missDistance, hbr, sigmaR, sigmaI, sigmaC, relVel, tca float64) float64 {
	sigmaEff := (sigmaR + sigmaI + sigmaC) / 3.0
	gaussian := math.Exp(-0.5*math.Pow(missDistance/math.Max(sigmaEff, 1e-6), 2.0)) * math.Pow(hbr/math.Max(sigmaEff, 1e-6), 2.0)
	velocityBias := 1.0 / (1.0 + relVel*0.015)
	timeBias := 1.0 - math.Exp(-tca/1200.0)
	pc := gaussian * velocityBias * (0.5 + 0.5*timeBias)
	return bounded(pc, 0.0, 1.0)
}

func syntheticPriority(class ThreatClass, distanceToAsset, mach, rcs float64) float64 {
	score := 20.0
	switch class {
	case ThreatHypersonic:
		score = 100
	case ThreatBallistic:
		score = 80
	case ThreatOrbital:
		score = 40
	}

	if distanceToAsset <= 500 {
		score += 50
	} else if distanceToAsset <= 1500 {
		score += 25
	}

	if mach >= 8 {
		score += 25
	} else if mach >= 5 {
		score += 15
	}
	if rcs < 0.5 {
		score += 10
	}
	return bounded(score, 0.0, 300.0)
}

func recommendedAction(class ThreatClass, priority, distanceToAsset float64) string {
	if distanceToAsset > 2000 {
		return "out_of_reach_watch"
	}
	switch {
	case class == ThreatHypersonic && priority >= 140:
		return "evasive_maneuver_now"
	case class == ThreatBallistic && priority >= 120:
		return "prepare_evasive"
	case priority >= 90:
		return "sensor_retask"
	default:
		return "monitor_only"
	}
}
