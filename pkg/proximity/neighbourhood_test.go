package proximity_test

import (
	"testing"

	"ssis/pkg/catalog"
	"ssis/pkg/proximity"
	"ssis/pkg/types"
)

func TestScreenIndianAssets_NilStore(t *testing.T) {
	alerts := proximity.ScreenIndianAssets(nil, 100.0)
	if alerts != nil {
		t.Errorf("Expected nil alerts for nil store, got %d alerts", len(alerts))
	}
}

func TestScreenIndianAssets_EmptyCatalog(t *testing.T) {
	store := catalog.NewStore(100.0)
	alerts := proximity.ScreenIndianAssets(store, 100.0)
	if alerts != nil {
		t.Errorf("Expected nil alerts for empty catalog, got %d alerts", len(alerts))
	}
}

func TestScreenIndianAssets_NoIndianAssets(t *testing.T) {
	store := catalog.NewStore(100.0)
	// Add non-Indian asset
	obj := catalog.SDAObject{
		ID:          1,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{obj}}
	store.ApplyDeltaWithRebuild(delta, false)

	alerts := proximity.ScreenIndianAssets(store, 100.0)
	if alerts != nil {
		t.Errorf("Expected nil alerts when no Indian assets, got %d alerts", len(alerts))
	}
}

func TestScreenIndianAssets_ZeroRadius(t *testing.T) {
	store := catalog.NewStore(100.0)
	// Add Indian asset
	indianAsset := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	// Add threat
	threat := catalog.SDAObject{
		ID:          2,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7050, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset, threat}}
	store.ApplyDeltaWithRebuild(delta, false)

	// Test with zero radius - should use default from config
	alerts := proximity.ScreenIndianAssets(store, 0.0)
	if alerts == nil {
		t.Error("Expected alerts with zero radius (should use default), got nil")
	}
}

func TestScreenIndianAssets_NegativeRadius(t *testing.T) {
	store := catalog.NewStore(100.0)
	indianAsset := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	threat := catalog.SDAObject{
		ID:          2,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7050, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset, threat}}
	store.ApplyDeltaWithRebuild(delta, false)

	// Test with negative radius - should use default from config
	alerts := proximity.ScreenIndianAssets(store, -100.0)
	if alerts == nil {
		t.Error("Expected alerts with negative radius (should use default), got nil")
	}
}

func TestScreenIndianAssets_DuplicatePairPrevention(t *testing.T) {
	store := catalog.NewStore(100.0)
	indianAsset := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	threat := catalog.SDAObject{
		ID:          2,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7050, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset, threat}}
	store.ApplyDeltaWithRebuild(delta, false)

	alerts := proximity.ScreenIndianAssets(store, 100.0)
	if alerts == nil || len(alerts) != 1 {
		t.Errorf("Expected exactly 1 alert, got %d", len(alerts))
	}
}

func TestEvaluateThreat_ThresholdBoundaries(t *testing.T) {
	// This tests the threat level boundaries using config thresholds
	// We need to use the actual evaluateThreat function or test through ScreenIndianAssets
	// For now, we'll test through ScreenIndianAssets with specific distances

	store := catalog.NewStore(100.0)
	indianAsset := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}

	cfg := proximity.GetConfig()

	// Test at tight proximity boundary (just below threshold)
	threatTight := catalog.SDAObject{
		ID:          2,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7000 + cfg.TightProximityKm - 1.0, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset, threatTight}}
	store.ApplyDeltaWithRebuild(delta, false)

	alerts := proximity.ScreenIndianAssets(store, cfg.DefaultThreatRadiusKm)
	if len(alerts) == 0 {
		t.Error("Expected alert at tight proximity boundary")
	}
	if alerts[0].ThreatLevel < 0.4 {
		t.Errorf("Expected elevated threat level at tight proximity, got %.2f", alerts[0].ThreatLevel)
	}
}

func TestScreenIndianAssets_SkipIndianAssets(t *testing.T) {
	store := catalog.NewStore(100.0)
	indianAsset1 := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	indianAsset2 := catalog.SDAObject{
		ID:          2,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7050, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset1, indianAsset2}}
	store.ApplyDeltaWithRebuild(delta, false)

	alerts := proximity.ScreenIndianAssets(store, 100.0)
	if alerts != nil && len(alerts) > 0 {
		t.Errorf("Expected no alerts when only Indian assets present, got %d alerts", len(alerts))
	}
}

func TestScreenIndianAssets_TemporalWindow(t *testing.T) {
	store := catalog.NewStore(100.0)
	indianAsset := catalog.SDAObject{
		ID:          1,
		CountryCode: "IND",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     1000.0,
	}
	// Threat at same position but different time (outside temporal window)
	threat := catalog.SDAObject{
		ID:          2,
		CountryCode: "USA",
		Pos:         types.Vector3{X: 7000, Y: 0, Z: 0},
		Vel:         types.Vector3{X: 0, Y: 7.5, Z: 0},
		EpochTT:     2000.0, // 1000 seconds later - outside temporal window
	}
	delta := catalog.Delta{Upserts: []catalog.SDAObject{indianAsset, threat}}
	store.ApplyDeltaWithRebuild(delta, false)

	cfg := proximity.GetConfig()
	alerts := proximity.ScreenIndianAssets(store, cfg.DefaultThreatRadiusKm)

	// Should not alert due to temporal window
	if alerts != nil && len(alerts) > 0 {
		t.Errorf("Expected no alerts due to temporal window, got %d alerts", len(alerts))
	}
}
