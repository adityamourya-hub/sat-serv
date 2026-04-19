// Package sensors provides integration with Python analytics microservices
// Handles UCT (Uncorrelated Track) detection via MPDAF 3D matched filtering
package sensors

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// UCTDetection represents an uncorrelated track from optical sensors
type UCTDetection struct {
	NoradCatID      *int32    `json:"norad_cat_id"`     // Assigned after catalog correlation
	ObjectName      string    `json:"object_name"`      // Temporary designation
	DetectionTime   time.Time `json:"detection_time"`   // ISO8601 timestamp
	RADeg           float64   `json:"ra_deg"`           // Right Ascension (degrees)
	DecDeg          float64   `json:"dec_deg"`          // Declination (degrees)
	VelocityRA      float64   `json:"velocity_ra"`      // Proper motion RA (deg/day)
	VelocityDec     float64   `json:"velocity_dec"`     // Proper motion Dec (deg/day)
	PeakSN          float64   `json:"peak_sn"`          // Peak signal-to-noise
	Confidence      float64   `json:"confidence"`       // Detection confidence (0-1)
	DetectionSource string    `json:"detection_source"` // "MPDAF_3D_STREAK"
	SensorType      string    `json:"sensor_type"`      // "OPTICAL", "IR", etc.
	SensorID        string    `json:"sensor_id"`        // IAF sensor identifier
	StreakLength    float64   `json:"streak_length_px"` // Pixels
	StreakAngle     float64   `json:"streak_angle_deg"` // Degrees (0-180)
	IntegratedFlux  float64   `json:"integrated_flux"`  // ADU or photons
}

// PythonBridge manages Python analytics microservice execution
type PythonBridge struct {
	pythonPath     string
	scriptDir      string
	dbPool         *pgxpool.Pool
	tempDir        string
	plasmaProducer *PlasmaProducer
	usePlasma      bool // Feature flag for Zero-Copy IPC
}

// NewPythonBridge creates bridge instance
func NewPythonBridge(dbPool *pgxpool.Pool) *PythonBridge {
	return &PythonBridge{
		pythonPath: "python3", // Assumes python3 in PATH
		scriptDir:  "./ai/v3", // Relative to project root
		dbPool:     dbPool,
		tempDir:    "/tmp/iaf_uct",
		usePlasma:  false, // Default to PostgreSQL for compatibility
	}
}

// EnablePlasma enables Zero-Copy IPC via Apache Arrow Plasma
func (b *PythonBridge) EnablePlasma() error {
	config := DefaultPlasmaConfig()
	producer, err := NewPlasmaProducer(config)
	if err != nil {
		return fmt.Errorf("failed to initialize Plasma: %w", err)
	}
	b.plasmaProducer = producer
	b.usePlasma = true
	return nil
}

// RunStreakDetection executes MPDAF 3D matched filtering on sensor frames
func (b *PythonBridge) RunStreakDetection(
	ctx context.Context,
	sensorID string,
	startTime, endTime time.Time,
	frameCount int,
) ([]*UCTDetection, error) {

	// Step 1: Export sensor frames from database to FITS format
	framesPath, err := b.exportSensorFrames(ctx, sensorID, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("export sensor frames: %w", err)
	}
	defer os.RemoveAll(framesPath)

	// Step 2: Execute Python streak detection script
	outputPath := filepath.Join(b.tempDir, fmt.Sprintf("uct_%s_%d.json",
		sensorID, time.Now().Unix()))

	cmd := exec.CommandContext(ctx, b.pythonPath,
		"-m", "streak_detection",
		"--input", framesPath,
		"--output", outputPath,
		"--sensor-id", sensorID,
		"--sn-threshold", "5.0",
	)
	cmd.Dir = b.scriptDir

	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("python streak detection failed: %w\nOutput: %s", err, output)
	}

	// Step 3: Parse UCT results
	detections, err := b.parseUCTResults(outputPath)
	if err != nil {
		return nil, fmt.Errorf("parse UCT results: %w", err)
	}

	// Step 4: Store in database
	if err := b.storeUCTDetections(ctx, detections, sensorID); err != nil {
		return nil, fmt.Errorf("store UCT detections: %w", err)
	}

	return detections, nil
}

// exportSensorFrames retrieves frames from PostgreSQL and exports to numpy format
func (b *PythonBridge) exportSensorFrames(
	ctx context.Context,
	sensorID string,
	startTime, endTime time.Time,
) (string, error) {

	query := `
		SELECT 
			timestamp,
			image_data,
			exposure_time,
			wcs_header
		FROM sensor_frames
		WHERE sensor_id = $1
		  AND timestamp BETWEEN $2 AND $3
		ORDER BY timestamp
	`

	rows, err := b.dbPool.Query(ctx, query, sensorID, startTime, endTime)
	if err != nil {
		return "", fmt.Errorf("query sensor frames: %w", err)
	}
	defer rows.Close()

	// Create temp directory
	tempPath := filepath.Join(b.tempDir, fmt.Sprintf("frames_%s_%d",
		sensorID, time.Now().Unix()))
	if err := os.MkdirAll(tempPath, 0755); err != nil {
		return "", err
	}

	// Export frames (implementation would write to numpy/FITS files)
	// This is a simplified placeholder
	frameCount := 0
	for rows.Next() {
		var ts time.Time
		var imageData []byte
		var exposureTime float64
		var wcsHeader *string

		if err := rows.Scan(&ts, &imageData, &exposureTime, &wcsHeader); err != nil {
			continue
		}

		// Write frame to temp file
		framePath := filepath.Join(tempPath, fmt.Sprintf("frame_%04d.npy", frameCount))
		if err := os.WriteFile(framePath, imageData, 0644); err != nil {
			continue
		}
		frameCount++
	}

	return tempPath, rows.Err()
}

// parseUCTResults reads JSON output from Python script
func (b *PythonBridge) parseUCTResults(path string) ([]*UCTDetection, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var detections []*UCTDetection
	if err := json.Unmarshal(data, &detections); err != nil {
		return nil, err
	}

	return detections, nil
}

// storeUCTDetections saves detections to uct_detections table
func (b *PythonBridge) storeUCTDetections(
	ctx context.Context,
	detections []*UCTDetection,
	sensorID string,
) error {

	query := `
		INSERT INTO uct_detections (
			object_name, detection_time, ra_deg, dec_deg,
			velocity_ra, velocity_dec, peak_sn, confidence,
			detection_source, sensor_type, sensor_id,
			streak_length_px, streak_angle_deg, integrated_flux,
			correlation_status, created_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, 'UNCORRELATED', NOW()
		)
		ON CONFLICT (object_name, detection_time) DO NOTHING
	`

	for _, det := range detections {
		det.SensorID = sensorID

		_, err := b.dbPool.Exec(ctx, query,
			det.ObjectName,
			det.DetectionTime,
			det.RADeg,
			det.DecDeg,
			det.VelocityRA,
			det.VelocityDec,
			det.PeakSN,
			det.Confidence,
			det.DetectionSource,
			det.SensorType,
			det.SensorID,
			det.StreakLength,
			det.StreakAngle,
			det.IntegratedFlux,
		)
		if err != nil {
			// Log but continue with other detections
			continue
		}
	}

	return nil
}

// CorrelateUCTWithCatalog attempts to match UCTs with known Space-Track objects
func (b *PythonBridge) CorrelateUCTWithCatalog(ctx context.Context, since time.Time) error {
	query := `
		WITH uct_candidates AS (
			SELECT 
				uct.*,
				cs.norad_cat_id as candidate_norad,
				cs.object_name as candidate_name,
				-- Calculate angular distance
				ST_Distance(
					ST_SetSRID(ST_MakePoint(uct.ra_deg, uct.dec_deg), 4326)::geography,
					cs.position::geography
				) / 1000.0 as distance_km  -- Approximate at LEO altitude
			FROM uct_detections uct
			JOIN current_states cs ON TRUE  -- Cross join for correlation
			WHERE uct.correlation_status = 'UNCORRELATED'
			  AND uct.created_at > $1
			  AND cs.last_updated > NOW() - INTERVAL '5 minutes'
		)
		UPDATE uct_detections uct
		SET 
			norad_cat_id = uc.candidate_norad,
			correlation_status = 'CORRELATED',
			correlation_confidence = 
				CASE 
					WHEN uc.distance_km < 100 THEN 'HIGH'
					WHEN uc.distance_km < 500 THEN 'MEDIUM'
					ELSE 'LOW'
				END,
			correlated_at = NOW()
		FROM uct_candidates uc
		WHERE uct.id = uc.id
		  AND uc.distance_km < 1000  -- 1000km correlation radius
		  AND uc.peak_sn > 5.0
	`

	_, err := b.dbPool.Exec(ctx, query, since)
	return err
}

// GetUncorrelatedUCTs returns UCTs that couldn't be matched to catalog
func (b *PythonBridge) GetUncorrelatedUCTs(
	ctx context.Context,
	since time.Time,
	minConfidence float64,
) ([]*UCTDetection, error) {

	query := `
		SELECT 
			object_name, detection_time, ra_deg, dec_deg,
			velocity_ra, velocity_dec, peak_sn, confidence,
			detection_source, sensor_type, sensor_id
		FROM uct_detections
		WHERE correlation_status = 'UNCORRELATED'
		  AND created_at > $1
		  AND confidence >= $2
		ORDER BY detection_time DESC
	`

	rows, err := b.dbPool.Query(ctx, query, since, minConfidence)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var detections []*UCTDetection
	for rows.Next() {
		var d UCTDetection
		err := rows.Scan(
			&d.ObjectName, &d.DetectionTime, &d.RADeg, &d.DecDeg,
			&d.VelocityRA, &d.VelocityDec, &d.PeakSN, &d.Confidence,
			&d.DetectionSource, &d.SensorType, &d.SensorID,
		)
		if err != nil {
			continue
		}
		detections = append(detections, &d)
	}

	return detections, rows.Err()
}
