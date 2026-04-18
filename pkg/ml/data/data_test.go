package data

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestDataVersion_GenerateID(t *testing.T) {
	id1 := GenerateVersionID("maneuver_detector", "lstm_v1")
	id2 := GenerateVersionID("maneuver_detector", "lstm_v1")

	// IDs should be different (different timestamps)
	if id1 == id2 {
		t.Fatalf("expected different IDs for different timestamps")
	}

	// IDs should be well-formed
	if len(id1) < 10 {
		t.Fatalf("ID too short: %s", id1)
	}
}

func TestVersionRegistry_Register(t *testing.T) {
	registry := NewVersionRegistry()

	v := DataVersion{
		ID:    "test-v1",
		Name:  "test dataset v1",
		Model: "test_model",
	}

	if err := registry.Register(v); err != nil {
		t.Fatalf("failed to register: %v", err)
	}

	retrieved, err := registry.GetVersion("test-v1")
	if err != nil {
		t.Fatalf("failed to get version: %v", err)
	}

	if retrieved.ID != v.ID {
		t.Fatalf("version ID mismatch: %s vs %s", retrieved.ID, v.ID)
	}
}

func TestVersionRegistry_Persistence(t *testing.T) {
	tmpDir := t.TempDir()
	registryPath := filepath.Join(tmpDir, "registry.json")

	// Create and save
	registry := NewVersionRegistry()
	v := DataVersion{
		ID:    "persist-v1",
		Name:  "persistent test",
		Model: "test",
	}
	registry.Register(v)

	if err := registry.SaveRegistry(registryPath); err != nil {
		t.Fatalf("failed to save: %v", err)
	}

	// Load and verify
	loaded, err := LoadRegistry(registryPath)
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}

	retrieved, err := loaded.GetVersion("persist-v1")
	if err != nil {
		t.Fatalf("failed to get loaded version: %v", err)
	}

	if retrieved.Name != v.Name {
		t.Fatalf("name mismatch after load: %s vs %s", retrieved.Name, v.Name)
	}
}

func TestSyntheticDataGenerator_Basic(t *testing.T) {
	config := DataConfig{
		TLECount:          10,
		SamplesPerTLE:     5,
		PropagationSteps:  1440,
		SampleIntervalMin: 60.0,
		NoiseSTDKm:        0.05,
		TrainFraction:     0.7,
		ValFraction:       0.15,
		TestFraction:      0.15,
		Purpose:           "test",
		Model:             "test_model",
		Seed:              42,
	}

	gen := NewSyntheticDataGenerator(config)

	// Generate TLEs
	tles, err := gen.GenerateTLEs(config.TLECount)
	if err != nil {
		t.Fatalf("failed to generate TLEs: %v", err)
	}

	if len(tles) != config.TLECount {
		t.Fatalf("expected %d TLEs, got %d", config.TLECount, len(tles))
	}

	// Generate dataset
	samples, err := gen.GenerateDataset()
	if err != nil {
		t.Fatalf("failed to generate dataset: %v", err)
	}

	expectedSampleCount := config.TLECount * config.SamplesPerTLE
	if len(samples) != expectedSampleCount {
		t.Fatalf("expected %d samples, got %d", expectedSampleCount, len(samples))
	}

	// Verify splits
	trainCount, valCount, testCount := 0, 0, 0
	for _, s := range samples {
		switch s.Split {
		case "train":
			trainCount++
		case "val":
			valCount++
		case "test":
			testCount++
		}
	}

	if trainCount == 0 || valCount == 0 || testCount == 0 {
		t.Fatalf("invalid splits: train=%d, val=%d, test=%d", trainCount, valCount, testCount)
	}

	// Verify version metadata
	version := gen.Version()
	if version.ID == "" {
		t.Fatalf("version ID is empty")
	}
	if version.SampleCount != int64(len(samples)) {
		t.Fatalf("sample count mismatch: %d vs %d", version.SampleCount, len(samples))
	}
}

func TestWriter_WriteJSONL(t *testing.T) {
	tmpDir := t.TempDir()
	writer := NewWriter(tmpDir)

	// Generate small dataset
	config := DataConfig{
		TLECount:      2,
		SamplesPerTLE: 3,
		Seed:          42,
		Purpose:       "test",
		Model:         "test",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	// Write JSONL
	filename := "test.jsonl"
	if err := writer.WriteJSONL(samples, filename); err != nil {
		t.Fatalf("failed to write JSONL: %v", err)
	}

	// Verify file exists and has correct line count
	filePath := filepath.Join(tmpDir, filename)
	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("file not created: %v", err)
	}

	// Read back and verify
	readSamples, err := ReadJSONL(filePath)
	if err != nil {
		t.Fatalf("failed to read JSONL: %v", err)
	}

	if len(readSamples) != len(samples) {
		t.Fatalf("sample count mismatch: %d vs %d", len(readSamples), len(samples))
	}
}

func TestWriter_WriteCSV(t *testing.T) {
	tmpDir := t.TempDir()
	writer := NewWriter(tmpDir)

	// Generate small dataset
	config := DataConfig{
		TLECount:      2,
		SamplesPerTLE: 3,
		Seed:          42,
		Purpose:       "test",
		Model:         "test",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	// Write CSV
	filename := "test.csv"
	if err := writer.WriteCSV(samples, filename); err != nil {
		t.Fatalf("failed to write CSV: %v", err)
	}

	// Verify file exists and has correct line count
	filePath := filepath.Join(tmpDir, filename)
	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("file not created: %v", err)
	}

	// Read back and verify
	readSamples, err := ReadCSV(filePath)
	if err != nil {
		t.Fatalf("failed to read CSV: %v", err)
	}

	if len(readSamples) != len(samples) {
		t.Fatalf("sample count mismatch: %d vs %d", len(readSamples), len(samples))
	}
}

func TestWriter_WriteVersion(t *testing.T) {
	tmpDir := t.TempDir()
	writer := NewWriter(tmpDir)

	version := DataVersion{
		ID:    "test-v1",
		Name:  "test dataset",
		Model: "test_model",
	}

	if err := writer.WriteVersion(version, "version.json"); err != nil {
		t.Fatalf("failed to write version: %v", err)
	}

	filePath := filepath.Join(tmpDir, "version.json")
	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("file not created: %v", err)
	}
}

func TestWriter_WriteMetrics(t *testing.T) {
	tmpDir := t.TempDir()
	writer := NewWriter(tmpDir)

	// Generate small dataset
	config := DataConfig{
		TLECount:      2,
		SamplesPerTLE: 3,
		Seed:          42,
		Purpose:       "test",
		Model:         "test",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	if err := writer.WriteMetrics(samples, "metrics.json"); err != nil {
		t.Fatalf("failed to write metrics: %v", err)
	}

	filePath := filepath.Join(tmpDir, "metrics.json")
	if _, err := os.Stat(filePath); err != nil {
		t.Fatalf("file not created: %v", err)
	}
}

func TestComputeMetrics(t *testing.T) {
	// Generate small dataset
	config := DataConfig{
		TLECount:      3,
		SamplesPerTLE: 5,
		Seed:          42,
		Purpose:       "test",
		Model:         "test",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	metrics := computeMetrics(samples)

	// Verify basic metrics are present
	if _, ok := metrics["total_samples"]; !ok {
		t.Fatalf("missing total_samples metric")
	}
	if _, ok := metrics["avg_risk"]; !ok {
		t.Fatalf("missing avg_risk metric")
	}
	if _, ok := metrics["high_threat_pct"]; !ok {
		t.Fatalf("missing high_threat_pct metric")
	}

	// Verify counts
	if metrics["total_samples"] != len(samples) {
		t.Fatalf("sample count mismatch")
	}
}

func TestWandBClient_LogMetrics(t *testing.T) {
	client := NewWandBClient("test-project", "test-entity", "test-run")

	metrics := map[string]interface{}{
		"sample_count": 1000,
		"avg_risk":     0.45,
	}

	// Should not error even if wandb not available
	err := client.LogMetrics("test-v1", metrics)
	if err != nil && client.enabled {
		t.Fatalf("failed to log metrics: %v", err)
	}
}

func TestSummarizeDataset(t *testing.T) {
	config := DataConfig{
		TLECount:      2,
		SamplesPerTLE: 3,
		Seed:          42,
		Purpose:       "test",
		Model:         "test",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	summary := SummarizeDataset(gen, samples)

	if datasetInfo, ok := summary["dataset_info"].(map[string]interface{}); ok {
		if datasetInfo["total_samples"] != len(samples) {
			t.Fatalf("summary sample count mismatch")
		}
	} else {
		t.Fatalf("missing dataset_info in summary")
	}

	if metrics, ok := summary["metrics"].(map[string]interface{}); ok {
		if metrics["total_samples"] != len(samples) {
			t.Fatalf("summary metrics sample count mismatch")
		}
	} else {
		t.Fatalf("missing metrics in summary")
	}
}

func TestFullEndToEnd(t *testing.T) {
	tmpDir := t.TempDir()

	// Create generator
	config := DefaultDataConfig("maneuver_detection", "lstm_attention_v1")
	config.TLECount = 10
	config.SamplesPerTLE = 15
	config.Seed = 42

	gen := NewSyntheticDataGenerator(config)

	// Generate TLEs and dataset
	gen.GenerateTLEs(config.TLECount)
	samples, err := gen.GenerateDataset()
	if err != nil {
		t.Fatalf("failed to generate dataset: %v", err)
	}

	// Write to all formats
	writer := NewWriter(tmpDir)
	version := gen.Version()

	writer.WriteVersion(version, "version.json")
	writer.WriteJSONL(samples, "dataset.jsonl")
	writer.WriteCSV(samples, "dataset.csv")
	writer.WriteMetrics(samples, "metrics.json")

	// Verify all files exist
	files := []string{"version.json", "dataset.jsonl", "dataset.csv", "metrics.json"}
	for _, file := range files {
		path := filepath.Join(tmpDir, file)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("file not created: %s", file)
		}
	}

	// Read back and verify
	readSamples, err := ReadJSONL(filepath.Join(tmpDir, "dataset.jsonl"))
	if err != nil {
		t.Fatalf("failed to read JSONL: %v", err)
	}

	if len(readSamples) != len(samples) {
		t.Fatalf("readback sample count mismatch: %d vs %d", len(readSamples), len(samples))
	}

	fmt.Printf("End-to-end test passed: %d samples generated and persisted\n", len(samples))
}

func BenchmarkGenerateDataset(b *testing.B) {
	config := DataConfig{
		TLECount:      100,
		SamplesPerTLE: 10,
		Seed:          42,
		Purpose:       "bench",
		Model:         "bench",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.GenerateDataset()
	}
}

func BenchmarkWriteJSONL(b *testing.B) {
	tmpDir := b.TempDir()

	config := DataConfig{
		TLECount:      50,
		SamplesPerTLE: 10,
		Seed:          42,
		Purpose:       "bench",
		Model:         "bench",
	}

	gen := NewSyntheticDataGenerator(config)
	gen.GenerateTLEs(config.TLECount)
	samples, _ := gen.GenerateDataset()

	writer := NewWriter(tmpDir)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		writer.WriteJSONL(samples, fmt.Sprintf("bench_%d.jsonl", i))
	}
}
