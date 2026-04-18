package evaluation

import "fmt"

// ScenarioCategory maps to mandatory ADITI-aligned operational test categories.
type ScenarioCategory string

const (
	CategoryIngestionCatalogRealism     ScenarioCategory = "ingestion_catalog_realism"
	CategoryPropagationSpatialScaling   ScenarioCategory = "propagation_spatial_scaling"
	CategoryConjunctionCollision        ScenarioCategory = "conjunction_collision_pipeline"
	CategoryRapidThreatTriage           ScenarioCategory = "rapid_threat_triage"
	CategoryManeuverCoverage            ScenarioCategory = "maneuver_coverage_analytics"
	CategoryTrainingSimulator           ScenarioCategory = "training_simulator"
	CategoryAIIntegrationExplainability ScenarioCategory = "ai_integration_explainability"
	CategoryTrustAudit                  ScenarioCategory = "trust_audit_layer"
	CategoryScalabilityResilience       ScenarioCategory = "scalability_resilience"
	CategoryEndToEndWatchCycle          ScenarioCategory = "end_to_end_watch_cycle"
)

// OperationalScenario captures an IAF-style mission test plan entry.
type OperationalScenario struct {
	ID              string             `json:"id"`
	MissionContext  string             `json:"mission_context"`
	Description     string             `json:"description"`
	InputSpec       string             `json:"input_spec"`
	ExpectedOutputs string             `json:"expected_outputs"`
	SuccessCriteria []string           `json:"success_criteria"`
	Categories      []ScenarioCategory `json:"categories"`
}

func MandatoryCategories() []ScenarioCategory {
	return []ScenarioCategory{
		CategoryIngestionCatalogRealism,
		CategoryPropagationSpatialScaling,
		CategoryConjunctionCollision,
		CategoryRapidThreatTriage,
		CategoryManeuverCoverage,
		CategoryTrainingSimulator,
		CategoryAIIntegrationExplainability,
		CategoryTrustAudit,
		CategoryScalabilityResilience,
		CategoryEndToEndWatchCycle,
	}
}

// DefaultIAFOperationalScenarios provides 10 realistic mission scenarios.
func DefaultIAFOperationalScenarios() []OperationalScenario {
	return []OperationalScenario{
		{
			ID:              "OPS-01",
			MissionContext:  "LEO Surveillance Watch Floor",
			Description:     "Ingest 10k mixed LEO/GEO catalog with timestamp drift and noisy observations during shift handover.",
			InputSpec:       "10k catalog entries, 2% malformed updates, sensor covariance 0.01-0.08 km, timestamp drift +-12 s",
			ExpectedOutputs: "Versioned ingest commit, failed records quarantined, correlated observations prioritized",
			SuccessCriteria: []string{"Ingest success >= 97%", "All malformed records quarantined", "Correlation recall >= 0.9"},
			Categories:      []ScenarioCategory{CategoryIngestionCatalogRealism, CategoryTrustAudit},
		},
		{
			ID:              "OPS-02",
			MissionContext:  "GEO Asset Protection",
			Description:     "Run J2 propagation and spatial indexing on 50k objects with periodic deltas every 15 minutes.",
			InputSpec:       "50k catalog, LEO/GEO ratio 80/20, update interval 15 min",
			ExpectedOutputs: "Propagation completion under latency budget and stable octree/hash-grid candidate generation",
			SuccessCriteria: []string{"Propagation p95 <= 120 ms per batch", "Candidate miss rate <= 1%"},
			Categories:      []ScenarioCategory{CategoryPropagationSpatialScaling, CategoryScalabilityResilience},
		},
		{
			ID:              "OPS-03",
			MissionContext:  "Simultaneous Conjunction Burst",
			Description:     "Inject multiple concurrent close approaches and evaluate classical + hybrid collision path.",
			InputSpec:       "200 conjunction candidates with mixed covariance and relative velocity profiles",
			ExpectedOutputs: "Signed conjunction alerts with uncertainty bounds and fallback provenance when required",
			SuccessCriteria: []string{"Collision FPR <= 0.12", "All alerts signed", "Fallback activates on low confidence"},
			Categories:      []ScenarioCategory{CategoryConjunctionCollision, CategoryAIIntegrationExplainability, CategoryTrustAudit},
		},
		{
			ID:              "OPS-04",
			MissionContext:  "Rapid Ballistic/Hypersonic Triage",
			Description:     "Process rapid-threat tracks under concurrent load with action recommendation priority stack.",
			InputSpec:       "Mixed Mach 2-12 tracks, range 100-3000 km, 10-50 parallel requests",
			ExpectedOutputs: "Ranked threat queue, recommendation confidence, signed actionable alert",
			SuccessCriteria: []string{"Top-5 recall >= 0.9", "Throughput >= 150 req/s equivalent", "Priority stability >= 0.85"},
			Categories:      []ScenarioCategory{CategoryRapidThreatTriage, CategoryScalabilityResilience},
		},
		{
			ID:              "OPS-05",
			MissionContext:  "Unexpected Maneuver Screening",
			Description:     "Detect abrupt and low-thrust maneuvers with residual spikes and false alarm suppression.",
			InputSpec:       "Time-series residual windows with maneuver injections and noise floor drift",
			ExpectedOutputs: "Maneuver events with AI/classical provenance and explanatory features",
			SuccessCriteria: []string{"Maneuver recall >= 0.85", "False alarms <= 0.1", "Attribution generated for each event"},
			Categories:      []ScenarioCategory{CategoryManeuverCoverage, CategoryAIIntegrationExplainability},
		},
		{
			ID:              "OPS-06",
			MissionContext:  "Blind-Spot Coverage Forecast",
			Description:     "Run blind-spot analysis while ingesting updates and predictive uncertainty estimates.",
			InputSpec:       "Coverage grid updates every 5 min with sparse observation zones",
			ExpectedOutputs: "Forecast blind-spot ratios and retask suggestions",
			SuccessCriteria: []string{"Forecast error <= 15%", "Retask suggestions include confidence bounds"},
			Categories:      []ScenarioCategory{CategoryManeuverCoverage, CategoryAIIntegrationExplainability},
		},
		{
			ID:              "OPS-07",
			MissionContext:  "Operator Training Drill",
			Description:     "Generate and execute 100+ training scenarios per run for conjunction avoidance decisions.",
			InputSpec:       "Synthetic scenario generation count >= 120 per run",
			ExpectedOutputs: "Scenario pack with difficulty labels and recommended actions",
			SuccessCriteria: []string{"Scenario count >= 100", "Difficulty distribution balanced", "Action policy output available"},
			Categories:      []ScenarioCategory{CategoryTrainingSimulator, CategoryEndToEndWatchCycle},
		},
		{
			ID:              "OPS-08",
			MissionContext:  "Trust and Audit Validation",
			Description:     "Verify every emitted alert is cryptographically signed with full classical + AI provenance.",
			InputSpec:       "Mixed conjunction/rapid/maneuver alerts under concurrent run",
			ExpectedOutputs: "Signed alert payload with provenance fields and explanation",
			SuccessCriteria: []string{"Signature verification pass rate 100%", "No alert without provenance"},
			Categories:      []ScenarioCategory{CategoryTrustAudit, CategoryAIIntegrationExplainability},
		},
		{
			ID:              "OPS-09",
			MissionContext:  "Scale and Chaos Exercise",
			Description:     "Run 100k+ catalog stress with 50 concurrent requests while injecting corruption and delays.",
			InputSpec:       "100001 objects, corruption 1-2%, latency injection 10-40 ms",
			ExpectedOutputs: "Graceful degradation, fallback activation, no panic, maintained signed alerts",
			SuccessCriteria: []string{"No crash", "Fallback ratio reported", "p99 within declared budget"},
			Categories:      []ScenarioCategory{CategoryScalabilityResilience, CategoryTrustAudit},
		},
		{
			ID:              "OPS-10",
			MissionContext:  "24-Hour SSA Watch Cycle",
			Description:     "Simulate full watch cycle with periodic updates, concurrent analyst queries, and threat injections.",
			InputSpec:       "Compressed 24h cycle, updates each 15 min, mixed threat and maneuver events",
			ExpectedOutputs: "Operational metrics report, signed alert samples, ADITI-aligned pass/fail matrix",
			SuccessCriteria: []string{"All mandatory categories exercised", "Report generated", "End-to-end pass verdict"},
			Categories:      []ScenarioCategory{CategoryEndToEndWatchCycle, CategoryConjunctionCollision, CategoryRapidThreatTriage, CategoryManeuverCoverage},
		},
	}
}

func ValidateScenarioCoverage(scenarios []OperationalScenario) error {
	covered := map[ScenarioCategory]bool{}
	for _, s := range scenarios {
		for _, c := range s.Categories {
			covered[c] = true
		}
	}
	for _, mandatory := range MandatoryCategories() {
		if !covered[mandatory] {
			return fmt.Errorf("mandatory category missing: %s", mandatory)
		}
	}
	if len(scenarios) < 8 || len(scenarios) > 12 {
		return fmt.Errorf("scenario count must be between 8 and 12, got %d", len(scenarios))
	}
	return nil
}
