// Package repository provides data access for blind spot detection
package repository

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// AdversarySatellite represents an adversary asset for blind spot analysis
type AdversarySatellite struct {
	NoradCatID  int32
	ObjectName  string
	CountryCode string
	SensorType  string // "OPTICAL", "SAR", "ALL"
	TLELine1    string
	TLELine2    string
}

// BlindSpotRepo provides data access for blind spot detection
type BlindSpotRepo struct {
	pool *pgxpool.Pool
}

// NewBlindSpotRepo creates repository instance
func NewBlindSpotRepo(pool *pgxpool.Pool) *BlindSpotRepo {
	return &BlindSpotRepo{pool: pool}
}

// GetAdversarySatellites retrieves adversary assets with current TLEs
func (r *BlindSpotRepo) GetAdversarySatellites(ctx context.Context, sensorType string) ([]*AdversarySatellite, error) {
	query := `
		SELECT 
			am.norad_cat_id, 
			am.object_name,
			am.country_code,
			am.sensor_capabilities,
			cs.tle_line1,
			cs.tle_line2
		FROM asset_metadata am
		JOIN current_states cs ON am.norad_cat_id = cs.norad_cat_id
		WHERE am.is_adversary = TRUE
		  AND am.is_active = TRUE
		  AND cs.tle_line1 IS NOT NULL
		  AND cs.tle_line2 IS NOT NULL
		  AND ($1 = 'ALL' OR am.sensor_capabilities @> ARRAY[$1])
		ORDER BY am.priority_level DESC, am.norad_cat_id
	`

	rows, err := r.pool.Query(ctx, query, sensorType)
	if err != nil {
		return nil, fmt.Errorf("query adversary satellites: %w", err)
	}
	defer rows.Close()

	return r.scanAdversarySatellites(rows)
}

// GetSatellitesInVicinity uses PostGIS to find potential visibility candidates
// This acts as a coarse filter to reduce computation
func (r *BlindSpotRepo) GetSatellitesInVicinity(ctx context.Context, lon, lat, alt float64, radiusKm float64) ([]int32, error) {
	query := `
		SELECT norad_cat_id
		FROM current_states
		WHERE is_adversary = TRUE
		  AND ST_DWithin(
			position::geography,
			ST_SetSRID(ST_MakePoint($1, $2, $3 * 1000), 4326)::geography,
			$4 * 1000  -- radius in meters
		)
	`

	rows, err := r.pool.Query(ctx, query, lon, lat, alt, radiusKm)
	if err != nil {
		return nil, fmt.Errorf("query satellites in vicinity: %w", err)
	}
	defer rows.Close()

	var ids []int32
	for rows.Next() {
		var id int32
		if err := rows.Scan(&id); err == nil {
			ids = append(ids, id)
		}
	}

	return ids, rows.Err()
}

// StoreBlindSpotAlert stores a detected blind spot opportunity
func (r *BlindSpotRepo) StoreBlindSpotAlert(ctx context.Context, siteName string,
	startTime, endTime time.Time, duration time.Duration, protectedFrom string) error {

	query := `
		INSERT INTO threat_vector (
			threat_type, severity, status, description, 
			raw_message, alert_origin, created_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, NOW()
		)
	`

	description := fmt.Sprintf("Blind spot at %s: %v window protected from %s sensors",
		siteName, duration, protectedFrom)

	rawMessage := map[string]interface{}{
		"site_name":        siteName,
		"start_time":       startTime,
		"end_time":         endTime,
		"duration_minutes": duration.Minutes(),
		"protected_from":   protectedFrom,
		"detection_type":   "BLIND_SPOT",
	}

	rawJSON, _ := json.Marshal(rawMessage)

	_, err := r.pool.Exec(ctx, query,
		"BLIND_SPOT",
		"INFO", // Blind spots are opportunities, not threats
		"ACTIVE",
		description,
		rawJSON,
		"ANALYTICS",
	)

	return err
}

// GetAccessWindowsForSite retrieves historical access windows for analysis
func (r *BlindSpotRepo) GetAccessWindowsForSite(ctx context.Context, siteName string, since time.Time) ([]AccessWindowRecord, error) {
	query := `
		SELECT 
			satellite_id, window_start, window_end, 
			max_elevation, sensor_type
		FROM access_windows
		WHERE site_name = $1
		  AND window_start > $2
		ORDER BY window_start DESC
	`

	rows, err := r.pool.Query(ctx, query, siteName, since)
	if err != nil {
		return nil, fmt.Errorf("query access windows: %w", err)
	}
	defer rows.Close()

	var records []AccessWindowRecord
	for rows.Next() {
		var r AccessWindowRecord
		err := rows.Scan(&r.SatelliteID, &r.StartTime, &r.EndTime, &r.MaxElevation, &r.SensorType)
		if err != nil {
			continue
		}
		records = append(records, r)
	}

	return records, rows.Err()
}

// AccessWindowRecord represents a stored access window
type AccessWindowRecord struct {
	SatelliteID  int32
	StartTime    time.Time
	EndTime      time.Time
	MaxElevation float64
	SensorType   string
}

// StoreAccessWindow stores a calculated access window
func (r *BlindSpotRepo) StoreAccessWindow(ctx context.Context, siteName string,
	satelliteID int32, window AccessWindowRecord) error {

	query := `
		INSERT INTO access_windows (
			site_name, satellite_id, window_start, window_end, 
			max_elevation, sensor_type, calculated_at
		) VALUES ($1, $2, $3, $4, $5, $6, NOW())
		ON CONFLICT (site_name, satellite_id, window_start) 
		DO UPDATE SET 
			window_end = EXCLUDED.window_end,
			max_elevation = EXCLUDED.max_elevation,
			calculated_at = NOW()
	`

	_, err := r.pool.Exec(ctx, query,
		siteName, satelliteID, window.StartTime, window.EndTime,
		window.MaxElevation, window.SensorType,
	)

	return err
}

// scanAdversarySatellites reads satellite data from rows
func (r *BlindSpotRepo) scanAdversarySatellites(rows pgx.Rows) ([]*AdversarySatellite, error) {
	var satellites []*AdversarySatellite

	for rows.Next() {
		var s AdversarySatellite
		err := rows.Scan(
			&s.NoradCatID, &s.ObjectName, &s.CountryCode, &s.SensorType,
			&s.TLELine1, &s.TLELine2,
		)
		if err != nil {
			slog.Error("Failed to scan adversary satellite", "error", err)
			continue
		}
		satellites = append(satellites, &s)
	}

	return satellites, rows.Err()
}
