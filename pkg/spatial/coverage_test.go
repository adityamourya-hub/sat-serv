package spatial

import "testing"

type mockBlindSpotModel struct{}

func (m mockBlindSpotModel) Name() string { return "mock_tcn_forecaster" }

func (m mockBlindSpotModel) Forecast(history []BlindSpotSample, horizonHours int) (BlindSpotForecastResult, error) {
	pts := make([]BlindSpotForecastPoint, 0, horizonHours)
	for i := 1; i <= horizonHours; i++ {
		pred := 0.35
		pts = append(pts, BlindSpotForecastPoint{
			HourAhead:         i,
			PredictedRatio:    pred,
			UncertaintyLower:  0.28,
			UncertaintyUpper:  0.44,
			ModelContribution: 1.0,
		})
	}
	return BlindSpotForecastResult{Points: pts, ModelName: "mock_tcn", Confidence: 0.9, UsedFallback: false}, nil
}

func TestCoverageIndexBlindSpotReduction(t *testing.T) {
	idx := NewCoverageIndex(36, 72)
	initial := idx.BlindSpotCount()
	idx.MarkCircularCoverage(0, 0, 20)
	idx.MarkCircularCoverage(25, 25, 20)
	after := idx.BlindSpotCount()

	if after >= initial {
		t.Fatalf("Expected blind spots to reduce after marking coverage")
	}
	if idx.BlindSpotRatio() <= 0 || idx.BlindSpotRatio() >= 1 {
		t.Fatalf("Expected blind spot ratio in (0,1), got %f", idx.BlindSpotRatio())
	}
}

func TestForecastBlindSpotsFallback(t *testing.T) {
	history := []BlindSpotSample{
		{TimestampHour: 100, BlindRatio: 0.45},
		{TimestampHour: 101, BlindRatio: 0.43},
		{TimestampHour: 102, BlindRatio: 0.42},
		{TimestampHour: 103, BlindRatio: 0.41},
	}

	res := ForecastBlindSpots(history, 24, nil)
	if len(res.Points) != 24 {
		t.Fatalf("expected 24 forecast points, got %d", len(res.Points))
	}
	if !res.UsedFallback {
		t.Fatalf("expected fallback forecast path")
	}
	if res.Points[0].PredictedRatio < 0 || res.Points[0].PredictedRatio > 1 {
		t.Fatalf("predicted ratio out of range")
	}
}

func TestForecastBlindSpotsHybrid(t *testing.T) {
	history := []BlindSpotSample{
		{TimestampHour: 200, BlindRatio: 0.52},
		{TimestampHour: 201, BlindRatio: 0.51},
		{TimestampHour: 202, BlindRatio: 0.49},
		{TimestampHour: 203, BlindRatio: 0.50},
		{TimestampHour: 204, BlindRatio: 0.48},
	}

	f := NewHybridBlindSpotForecaster(mockBlindSpotModel{})
	f.ModelBlendWeight = 0.5
	res := ForecastBlindSpots(history, 48, f)

	if len(res.Points) != 48 {
		t.Fatalf("expected 48 points, got %d", len(res.Points))
	}
	if res.UsedFallback {
		t.Fatalf("expected model-hybrid path")
	}
	if res.Points[0].ModelContribution <= 0 {
		t.Fatalf("expected model contribution in hybrid forecast")
	}
}
