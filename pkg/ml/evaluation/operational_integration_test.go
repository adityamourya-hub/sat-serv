package evaluation

import (
	"fmt"
	"testing"

	"ssis/pkg/cryptography"
	"ssis/pkg/ml"
	"ssis/pkg/ml/hybrid"
	"ssis/pkg/physics"
)

type lowConfidenceEngine struct{}

func (e lowConfidenceEngine) PredictRapidScore(track physics.RapidTrack) (physics.RapidRankInference, error) {
	return physics.RapidRankInference{Score: 10, Confidence: 0.1}, nil
}

func (e lowConfidenceEngine) PredictManeuver(window []physics.ManeuverFrame) (physics.ManeuverInference, error) {
	return physics.ManeuverInference{Probability: 0.2, Confidence: 0.1, Type: physics.ManeuverNone}, nil
}

func (e lowConfidenceEngine) PredictCollisionRisk(input physics.CollisionRiskInput, graph physics.CollisionNeighborhoodGraph) (physics.CollisionGraphInference, error) {
	return physics.CollisionGraphInference{RiskScore: 0.5, Confidence: 0.1}, nil
}

func (e lowConfidenceEngine) IsHealthy() bool { return true }

func TestOperationalScenarioDesignCoverage(t *testing.T) {
	scenarios := DefaultIAFOperationalScenarios()
	if err := ValidateScenarioCoverage(scenarios); err != nil {
		t.Fatalf("scenario coverage validation failed: %v", err)
	}

	alert := SignedOperationalAlert{
		EventID:          "OPS-SCENARIO-COVERAGE",
		TimestampUnix:    1712736000,
		AssetID:          43286,
		ThreatID:         900001,
		ClassicalStage:   "coverage_validation",
		AIPath:           "scenario_design",
		ModelConfidence:  1.0,
		UncertaintyLower: 0,
		UncertaintyUpper: 0,
	}
	signer := cryptography.NewSigner("coverage-key")
	alert.Signature = signer.GenerateProof(alert.AssetID, "OPS_SCENARIO", alert.TimestampUnix)
	alert.SignatureValid = signer.Verify([]byte(fmt.Sprintf("%d:%s:%d", alert.AssetID, "OPS_SCENARIO", alert.TimestampUnix)), alert.Signature)

	lat := MeasureLatency(5, func() {
		_ = ValidateScenarioCoverage(scenarios)
	})
	if !alert.SignatureValid {
		t.Fatalf("expected valid signature")
	}
	t.Logf("signed alert example: %+v", alert)
	t.Logf("performance numbers: p50=%.2fms p95=%.2fms p99=%.2fms", lat.P50Ms, lat.P95Ms, lat.P99Ms)
}

func TestOperationalDatasetScaleProfiles(t *testing.T) {
	sizes := []int{10000, 50000, 100001}
	for _, sz := range sizes {
		cfg := DefaultDatasetConfig()
		cfg.CatalogSize = sz
		cfg.Seed = int64(sz)
		cfg.TrainingScenarioCount = 120
		cfg.ThreatInjectionCount = 150
		cfg.ManeuverInjectionCount = 110
		ds := GenerateOperationalDataset(cfg)

		if len(ds.Catalog) != sz {
			t.Fatalf("catalog size mismatch for %d: got %d", sz, len(ds.Catalog))
		}
		if len(ds.TrainingScenarios) < 100 {
			t.Fatalf("expected >=100 training scenarios, got %d", len(ds.TrainingScenarios))
		}

		leoCount := 0
		geoCount := 0
		corruptCount := 0
		for _, obj := range ds.Catalog {
			switch obj.Orbit {
			case OrbitLEO:
				leoCount++
			case OrbitGEO:
				geoCount++
			}
			if obj.Corrupted {
				corruptCount++
			}
		}
		if leoCount == 0 || geoCount == 0 {
			t.Fatalf("expected LEO+GEO mix for %d", sz)
		}
		if corruptCount == 0 {
			t.Fatalf("expected at least one corrupted record for %d", sz)
		}

		signer := cryptography.NewSigner("dataset-key")
		ts := int64(1712736000 + sz)
		sig := signer.GenerateProof(43286, "DATASET_SCALE", ts)
		if !signer.Verify([]byte(fmt.Sprintf("%d:%s:%d", 43286, "DATASET_SCALE", ts)), sig) {
			t.Fatalf("dataset scale signature validation failed")
		}

		lat := MeasureLatency(3, func() {
			_ = GenerateOperationalDataset(cfg)
		})
		t.Logf("signed alert example: asset=43286 event=DATASET_SCALE sig=%s", sig)
		t.Logf("performance numbers (%d objects): p50=%.2fms p95=%.2fms p99=%.2fms", sz, lat.P50Ms, lat.P95Ms, lat.P99Ms)
	}
}

func TestOperationalRunConcurrentPipeline(t *testing.T) {
	metrics, err := RunOperationalSimulation(OperationalRunConfig{
		CatalogSize:  10000,
		Concurrency:  20,
		WatchHours:   24,
		Seed:         20260410,
		TrainingRuns: 130,
	})
	if err != nil {
		t.Fatalf("operational simulation failed: %v", err)
	}

	if metrics.TotalRequests <= 0 || metrics.ThroughputReqPerSecond <= 0 {
		t.Fatalf("invalid throughput metrics: %+v", metrics)
	}
	if metrics.LatencyP95Ms < metrics.LatencyP50Ms {
		t.Fatalf("invalid latency ordering")
	}
	if !metrics.SignedAlertExample.SignatureValid {
		t.Fatalf("expected signed alert verification")
	}
	if metrics.CollisionAccuracy < 0 || metrics.CollisionAccuracy > 1 {
		t.Fatalf("invalid accuracy")
	}

	t.Logf("signed alert example: %+v", metrics.SignedAlertExample)
	t.Logf("performance numbers: throughput=%.2f p50=%.2f p95=%.2f p99=%.2f accuracy=%.3f fpr=%.3f fidelity=%.3f",
		metrics.ThroughputReqPerSecond,
		metrics.LatencyP50Ms,
		metrics.LatencyP95Ms,
		metrics.LatencyP99Ms,
		metrics.CollisionAccuracy,
		metrics.CollisionFPR,
		metrics.ExplainabilityFidelity.CombinedScore,
	)
}

func TestOperationalFallbackWhenConfidenceDrops(t *testing.T) {
	orch := hybrid.NewOrchestrator(lowConfidenceEngine{})

	input := physics.CollisionRiskInput{
		MissDistanceKm:      0.04,
		HardbodyRadiusKm:    0.005,
		RelativeVelocityKmS: 6,
		TimeToTCASeconds:    300,
		SigmaRadialKm:       0.02,
		SigmaIntrackKm:      0.02,
		SigmaCrosstrackKm:   0.02,
	}
	_, cDecision := orch.EvaluateCollision(input, physics.CollisionNeighborhoodGraph{Nodes: []physics.CollisionGraphNode{{ObjectID: 1}, {ObjectID: 2}}})
	if cDecision.Source != "fallback" {
		t.Fatalf("expected fallback for low confidence collision model")
	}

	_, rDecisions := orch.RankRapid([]physics.RapidTrack{{ID: 11, Mach: 8, AltitudeKm: 55, BurnoutVelocityKmS: 4.1, DistanceToAssetKm: 400, RadarCrossSection: 0.3}}, 3000, 1)
	if len(rDecisions) == 0 || rDecisions[0].Source != "fallback" {
		t.Fatalf("expected rapid fallback")
	}

	_, mDecisions := orch.DetectManeuvers(77,
		[]float64{0.02, 0.04, 0.06, 0.55, 0.03, 0.02},
		[]float64{0.003, 0.004, 0.006, 0.09, 0.004, 0.003},
	)
	fallbackSeen := false
	for _, d := range mDecisions {
		if d.Source == "fallback" {
			fallbackSeen = true
			break
		}
	}
	if !fallbackSeen {
		t.Fatalf("expected maneuver fallback with low confidence model")
	}

	signer := cryptography.NewSigner("fallback-key")
	ts := int64(1712737000)
	sig := signer.GenerateProof(77, "FALLBACK_EVENT", ts)
	if !signer.Verify([]byte(fmt.Sprintf("%d:%s:%d", 77, "FALLBACK_EVENT", ts)), sig) {
		t.Fatalf("fallback signature verification failed")
	}

	lat := MeasureLatency(10, func() {
		_, _ = orch.EvaluateCollision(input, physics.CollisionNeighborhoodGraph{Nodes: []physics.CollisionGraphNode{{ObjectID: 1}}})
	})
	t.Logf("signed alert example: asset=77 event=FALLBACK_EVENT sig=%s", sig)
	t.Logf("performance numbers: p50=%.2fms p95=%.2fms p99=%.2fms", lat.P50Ms, lat.P95Ms, lat.P99Ms)
}

func TestOperationalInferenceAccuracyVsClassicalBaseline(t *testing.T) {
	orch := hybrid.NewOrchestrator(ml.NewDummyInferenceEngine())
	data := GenerateOperationalDataset(DefaultDatasetConfig())

	cases := make([]CollisionCase, 0, 300)
	for i := 0; i < 300 && i < len(data.Threats); i++ {
		th := data.Threats[i]
		cases = append(cases, CollisionCase{
			Input: physics.CollisionRiskInput{
				MissDistanceKm:      th.MissDistanceKm,
				HardbodyRadiusKm:    0.005,
				RelativeVelocityKmS: th.RelativeVelocityKmS,
				TimeToTCASeconds:    th.TimeToTCASeconds,
				SigmaRadialKm:       0.03,
				SigmaIntrackKm:      0.03,
				SigmaCrosstrackKm:   0.03,
			},
			Graph:       physics.CollisionNeighborhoodGraph{Nodes: []physics.CollisionGraphNode{{ObjectID: th.TargetObjectID}}},
			IsCollision: th.RiskLabel >= 0.2,
		})
	}

	m, err := EvaluateCollisionAccuracy(orch, cases, 1e-4)
	if err != nil {
		t.Fatalf("accuracy evaluation failed: %v", err)
	}
	if m.Accuracy < 0 || m.Accuracy > 1 {
		t.Fatalf("invalid accuracy metrics")
	}

	signer := cryptography.NewSigner("accuracy-key")
	ts := int64(1712737200)
	sig := signer.GenerateProof(43286, "ACCURACY_EVENT", ts)
	if !signer.Verify([]byte(fmt.Sprintf("%d:%s:%d", 43286, "ACCURACY_EVENT", ts)), sig) {
		t.Fatalf("accuracy signature check failed")
	}
	t.Logf("signed alert example: asset=43286 event=ACCURACY_EVENT sig=%s", sig)
	t.Logf("performance numbers: accuracy=%.3f fpr=%.3f precision=%.3f recall=%.3f", m.Accuracy, m.FalsePositiveRate, m.Precision, m.Recall)
}
