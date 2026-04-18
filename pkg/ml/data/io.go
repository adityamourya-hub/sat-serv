package data

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Writer handles dataset serialization (JSONL, CSV, etc.).
type Writer struct {
	outputDir string
	version   DataVersion
}

// NewWriter creates a new dataset writer.
func NewWriter(outputDir string) *Writer {
	return &Writer{
		outputDir: outputDir,
	}
}

// WriteJSONL writes samples to a JSONL file (one sample per line).
func (w *Writer) WriteJSONL(samples []Sample, filename string) error {
	filePath := fmt.Sprintf("%s/%s", w.outputDir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	encoder := json.NewEncoder(writer)
	for _, sample := range samples {
		if err := encoder.Encode(sample); err != nil {
			return err
		}
	}

	return nil
}

// WriteJSON writes samples to a JSON file.
func (w *Writer) WriteJSON(samples []Sample, filename string) error {
	filePath := fmt.Sprintf("%s/%s", w.outputDir, filename)
	payload := map[string]interface{}{
		"version": w.version,
		"samples": samples,
	}

	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filePath, data, 0644)
}

// WriteCSV writes samples to a CSV file.
func (w *Writer) WriteCSV(samples []Sample, filename string) error {
	filePath := fmt.Sprintf("%s/%s", w.outputDir, filename)
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	csvWriter := csv.NewWriter(file)
	defer csvWriter.Flush()

	// Write header
	header := []string{
		"id", "tle_id", "epoch_unix", "propagate_mins",
		"true_x", "true_y", "true_z", "true_vx", "true_vy", "true_vz",
		"obs_x", "obs_y", "obs_z", "obs_vx", "obs_vy", "obs_vz",
		"residual_x", "residual_y", "residual_z",
		"cov_x", "cov_y", "cov_z", "cov_vx", "cov_vy", "cov_vz",
		"split", "risk", "confidence", "threat_class",
	}
	if err := csvWriter.Write(header); err != nil {
		return err
	}

	// Write records
	for _, sample := range samples {
		record := []string{
			sample.ID, sample.TLEID, strconv.FormatInt(sample.EpochUnix, 10), fmt.Sprintf("%.2f", sample.PropagateMins),
			fmt.Sprintf("%.6f", sample.TrueX), fmt.Sprintf("%.6f", sample.TrueY), fmt.Sprintf("%.6f", sample.TrueZ),
			fmt.Sprintf("%.6f", sample.TrueVX), fmt.Sprintf("%.6f", sample.TrueVY), fmt.Sprintf("%.6f", sample.TrueVZ),
			fmt.Sprintf("%.6f", sample.ObsX), fmt.Sprintf("%.6f", sample.ObsY), fmt.Sprintf("%.6f", sample.ObsZ),
			fmt.Sprintf("%.6f", sample.ObsVX), fmt.Sprintf("%.6f", sample.ObsVY), fmt.Sprintf("%.6f", sample.ObsVZ),
			fmt.Sprintf("%.6f", sample.ResidualX), fmt.Sprintf("%.6f", sample.ResidualY), fmt.Sprintf("%.6f", sample.ResidualZ),
			fmt.Sprintf("%.6e", sample.CovX), fmt.Sprintf("%.6e", sample.CovY), fmt.Sprintf("%.6e", sample.CovZ),
			fmt.Sprintf("%.6e", sample.CovVX), fmt.Sprintf("%.6e", sample.CovVY), fmt.Sprintf("%.6e", sample.CovVZ),
			sample.Split, fmt.Sprintf("%.6f", sample.Risk), fmt.Sprintf("%.6f", sample.Confidence), sample.ThreatClass,
		}
		if err := csvWriter.Write(record); err != nil {
			return err
		}
	}

	return nil
}

// WriteVersion writes version metadata.
func (w *Writer) WriteVersion(version DataVersion, filename string) error {
	w.version = version
	filePath := fmt.Sprintf("%s/%s", w.outputDir, filename)

	data, err := json.MarshalIndent(version, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filePath, data, 0644)
}

// WriteMetrics writes dataset statistics.
func (w *Writer) WriteMetrics(samples []Sample, filename string) error {
	metrics := computeMetrics(samples)
	filePath := fmt.Sprintf("%s/%s", w.outputDir, filename)

	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filePath, data, 0644)
}

// computeMetrics calculates dataset statistics.
func computeMetrics(samples []Sample) map[string]interface{} {
	if len(samples) == 0 {
		return map[string]interface{}{}
	}

	// Count splits
	trainCount, valCount, testCount := 0, 0, 0
	riskSum, confSum := 0.0, 0.0
	noiseMag, residualMag := 0.0, 0.0
	highThreatCount := 0

	for _, s := range samples {
		switch s.Split {
		case "train":
			trainCount++
		case "val":
			valCount++
		case "test":
			testCount++
		}

		riskSum += s.Risk
		confSum += s.Confidence

		// Noise magnitude (obs - true)
		dx := s.ObsX - s.TrueX
		dy := s.ObsY - s.TrueY
		dz := s.ObsZ - s.TrueZ
		noiseMag += (dx*dx + dy*dy + dz*dz)

		// Residual magnitude
		residualMag += (s.ResidualX*s.ResidualX + s.ResidualY*s.ResidualY + s.ResidualZ*s.ResidualZ)

		if s.ThreatClass == "high" {
			highThreatCount++
		}
	}

	avgRisk := riskSum / float64(len(samples))
	avgConf := confSum / float64(len(samples))
	avgNoise := (noiseMag / float64(len(samples)))
	avgResidual := (residualMag / float64(len(samples)))

	return map[string]interface{}{
		"total_samples":    len(samples),
		"train_count":      trainCount,
		"val_count":        valCount,
		"test_count":       testCount,
		"avg_risk":         avgRisk,
		"avg_confidence":   avgConf,
		"avg_noise_mag_km": avgNoise,
		"avg_residual":     avgResidual,
		"high_threat_pct":  float64(highThreatCount) / float64(len(samples)),
	}
}

// ReadJSONL reads samples from a JSONL file.
func ReadJSONL(filePath string) ([]Sample, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	samples := make([]Sample, 0)

	for scanner.Scan() {
		var sample Sample
		if err := json.Unmarshal(scanner.Bytes(), &sample); err != nil {
			return nil, err
		}
		samples = append(samples, sample)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return samples, nil
}

// ReadCSV reads samples from a CSV file.
func ReadCSV(filePath string) ([]Sample, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	csvReader := csv.NewReader(file)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, err
	}

	samples := make([]Sample, 0, len(records)-1)
	const expectedCSVFields = 29

	// Skip header
	for i := 1; i < len(records); i++ {
		record := records[i]
		if len(record) < expectedCSVFields {
			return nil, fmt.Errorf("csv row %d has %d fields, expected at least %d", i+1, len(record), expectedCSVFields)
		}

		propagateMins, err := parseCSVFloat(record[3], i+1, "propagate_mins")
		if err != nil {
			return nil, err
		}
		trueX, err := parseCSVFloat(record[4], i+1, "true_x")
		if err != nil {
			return nil, err
		}
		trueY, err := parseCSVFloat(record[5], i+1, "true_y")
		if err != nil {
			return nil, err
		}
		trueZ, err := parseCSVFloat(record[6], i+1, "true_z")
		if err != nil {
			return nil, err
		}
		trueVX, err := parseCSVFloat(record[7], i+1, "true_vx")
		if err != nil {
			return nil, err
		}
		trueVY, err := parseCSVFloat(record[8], i+1, "true_vy")
		if err != nil {
			return nil, err
		}
		trueVZ, err := parseCSVFloat(record[9], i+1, "true_vz")
		if err != nil {
			return nil, err
		}
		obsX, err := parseCSVFloat(record[10], i+1, "obs_x")
		if err != nil {
			return nil, err
		}
		obsY, err := parseCSVFloat(record[11], i+1, "obs_y")
		if err != nil {
			return nil, err
		}
		obsZ, err := parseCSVFloat(record[12], i+1, "obs_z")
		if err != nil {
			return nil, err
		}
		obsVX, err := parseCSVFloat(record[13], i+1, "obs_vx")
		if err != nil {
			return nil, err
		}
		obsVY, err := parseCSVFloat(record[14], i+1, "obs_vy")
		if err != nil {
			return nil, err
		}
		obsVZ, err := parseCSVFloat(record[15], i+1, "obs_vz")
		if err != nil {
			return nil, err
		}
		residualX, err := parseCSVFloat(record[16], i+1, "residual_x")
		if err != nil {
			return nil, err
		}
		residualY, err := parseCSVFloat(record[17], i+1, "residual_y")
		if err != nil {
			return nil, err
		}
		residualZ, err := parseCSVFloat(record[18], i+1, "residual_z")
		if err != nil {
			return nil, err
		}
		covX, err := parseCSVFloat(record[19], i+1, "cov_x")
		if err != nil {
			return nil, err
		}
		covY, err := parseCSVFloat(record[20], i+1, "cov_y")
		if err != nil {
			return nil, err
		}
		covZ, err := parseCSVFloat(record[21], i+1, "cov_z")
		if err != nil {
			return nil, err
		}
		covVX, err := parseCSVFloat(record[22], i+1, "cov_vx")
		if err != nil {
			return nil, err
		}
		covVY, err := parseCSVFloat(record[23], i+1, "cov_vy")
		if err != nil {
			return nil, err
		}
		covVZ, err := parseCSVFloat(record[24], i+1, "cov_vz")
		if err != nil {
			return nil, err
		}
		risk, err := parseCSVFloat(record[26], i+1, "risk")
		if err != nil {
			return nil, err
		}
		confidence, err := parseCSVFloat(record[27], i+1, "confidence")
		if err != nil {
			return nil, err
		}

		sample := Sample{
			ID:            record[0],
			TLEID:         record[1],
			PropagateMins: propagateMins,
			TrueX:         trueX,
			TrueY:         trueY,
			TrueZ:         trueZ,
			TrueVX:        trueVX,
			TrueVY:        trueVY,
			TrueVZ:        trueVZ,
			ObsX:          obsX,
			ObsY:          obsY,
			ObsZ:          obsZ,
			ObsVX:         obsVX,
			ObsVY:         obsVY,
			ObsVZ:         obsVZ,
			ResidualX:     residualX,
			ResidualY:     residualY,
			ResidualZ:     residualZ,
			CovX:          covX,
			CovY:          covY,
			CovZ:          covZ,
			CovVX:         covVX,
			CovVY:         covVY,
			CovVZ:         covVZ,
			Split:         record[25],
			Risk:          risk,
			Confidence:    confidence,
			ThreatClass:   record[28],
		}
		samples = append(samples, sample)
	}

	return samples, nil
}

func parseCSVFloat(s string, row int, field string) (float64, error) {
	val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, fmt.Errorf("csv row %d invalid %s value %q: %w", row, field, s, err)
	}
	return val, nil
}

// ReadJSON reads samples and version from a JSON file.
func ReadJSON(filePath string) ([]Sample, DataVersion, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, DataVersion{}, err
	}

	payload := struct {
		Version DataVersion `json:"version"`
		Samples []Sample    `json:"samples"`
	}{}

	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, DataVersion{}, err
	}

	return payload.Samples, payload.Version, nil
}
