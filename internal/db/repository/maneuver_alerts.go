// Package repository provides data access for maneuver detection alerts
package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// OrbitalTrendSample represents a single orbital trend record
type OrbitalTrendSample struct {
	Time          time.Time
	SemiMajorAxis float64
	Inclination   float64
	Eccentricity  float64
	RAAN          float64
	ArgOfPerigee  float64
	MeanAnomaly   float64
	MeanMotion    float64
	BStar         float64
}

// ManeuverAlertRepo provides data access for maneuver detection
type ManeuverAlertRepo struct {
	pool *pgxpool.Pool
}

// NewManeuverAlertRepo creates repository instance
func NewManeuverAlertRepo(pool *pgxpool.Pool) *ManeuverAlertRepo {
	return &ManeuverAlertRepo{pool: pool}
}

// GetOrbitalHistory retrieves last N orbital trend records for a satellite
func (r *ManeuverAlertRepo) GetOrbitalHistory(ctx context.Context, noradID int32, limit int) ([]*OrbitalTrendSample, error) {
	query := `
		SELECT 
			time, semi_major_axis, inclination, eccentricity, 
			raan, arg_of_perigee, mean_anomaly, mean_motion, bstar
		FROM orbital_trends
		WHERE norad_cat_id = $1
		ORDER BY time DESC
		LIMIT $2
	`

	rows, err := r.pool.Query(ctx, query, noradID, limit)
	if err != nil {
		return nil, fmt.Errorf("query orbital history: %w", err)
	}
	defer rows.Close()

	var samples []*OrbitalTrendSample
	for rows.Next() {
		var s OrbitalTrendSample
		err := rows.Scan(
			&s.Time, &s.SemiMajorAxis, &s.Inclination, &s.Eccentricity,
			&s.RAAN, &s.ArgOfPerigee, &s.MeanAnomaly, &s.MeanMotion, &s.BStar,
		)
		if err != nil {
			slog.Error("Failed to scan orbital history", "error", err)
			continue
		}
		samples = append(samples, &s)
	}

	return samples, rows.Err()
}

// InsertManeuverAlert inserts a detected maneuver into threat_vector table
func (r *ManeuverAlertRepo) InsertManeuverAlert(ctx context.Context, noradID int32, objectName, metric string,
	currentValue, mean, stdDev, zScore float64, alertLevel, maneuverType string) error {

	query := `
		INSERT INTO threat_vector (
			threat_type, severity, status, primary_norad_id, primary_object_name,
			raw_message, alert_origin, created_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, NOW()
		)
	`

	// Build JSON with maneuver details
	rawMessage := map[string]interface{}{
		"detection_type":  "MANEUVER",
		"metric":          metric,
		"current_value":   currentValue,
		"historical_mean": mean,
		"std_deviation":   stdDev,
		"z_score":         zScore,
		"maneuver_type":   maneuverType,
		"delta":           currentValue - mean,
	}

	// Map alert level to severity
	severity := "LOW"
	if alertLevel == "YELLOW" {
		severity = "MEDIUM"
	} else if alertLevel == "RED" {
		severity = "HIGH"
	}

	rawJSON, _ := json.Marshal(rawMessage)

	_, err := r.pool.Exec(ctx, query,
		"MANEUVER_DETECTED", // threat_type
		severity,
		"OPEN", // status
		noradID,
		objectName,
		rawJSON,
		"INTERNAL_ANALYTICS", // alert_origin
	)

	if err != nil {
		return fmt.Errorf("insert maneuver alert: %w", err)
	}

	return nil
}

// GetRecentTLEInserts retrieves satellites with recent TLE updates for analysis
func (r *ManeuverAlertRepo) GetRecentTLEInserts(ctx context.Context, since time.Duration, limit int) ([]int32, error) {
	query := `
		SELECT norad_cat_id
		FROM current_states
		WHERE last_updated > NOW() - $1::INTERVAL
		ORDER BY last_updated DESC
		LIMIT $2
	`

	rows, err := r.pool.Query(ctx, query, since, limit)
	if err != nil {
		return nil, fmt.Errorf("query recent TLEs: %w", err)
	}
	defer rows.Close()

	var noradIDs []int32
	for rows.Next() {
		var id int32
		if err := rows.Scan(&id); err == nil {
			noradIDs = append(noradIDs, id)
		}
	}

	return noradIDs, rows.Err()
}

// GetSatelliteName retrieves object name for a NORAD ID
func (r *ManeuverAlertRepo) GetSatelliteName(ctx context.Context, noradID int32) (string, error) {
	var name string
	err := r.pool.QueryRow(ctx,
		"SELECT object_name FROM current_states WHERE norad_cat_id = $1",
		noradID,
	).Scan(&name)
	if err != nil {
		return "", err
	}
	return name, nil
}
