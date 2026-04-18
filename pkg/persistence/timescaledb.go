package persistence

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"ssis/pkg/catalog"
	"ssis/pkg/physics"
	// PostgreSQL driver - add one of these to go.mod:
	// go get github.com/lib/pq              // pure Go, stable
	// go get github.com/jackc/pgx/v5        // faster, more features
	// _ "github.com/lib/pq"
)

/**
 * TimescaleDBBackend implements high-volume TLE storage using TimescaleDB.
 *
 * Features:
 * - Hypertable partitioning for time-series queries
 * - Deduplication via TLE hash
 * - Continuous aggregate integration for B* smoothing
 * - Batch ingestion for high throughput
 *
 * ADITI 4.0: Designed for 50k+ objects, 1.5 updates/day, 5.5 GB/year growth.
 */
type TimescaleDBBackend struct {
	db *sql.DB
}

/**
 * TimescaleDBConfig holds connection parameters.
 */
type TimescaleDBConfig struct {
	Host     string
	Port     int
	Database string
	User     string
	Password string
	SSLMode  string // prefer, require, disable
}

/**
 * NewTimescaleDBBackend creates a new TimescaleDB persistence backend.
 */
func NewTimescaleDBBackend(cfg TimescaleDBConfig) (*TimescaleDBBackend, error) {
	connString := fmt.Sprintf(
		"host=%s port=%d dbname=%s user=%s password=%s sslmode=%s pool_max_conns=20",
		cfg.Host, cfg.Port, cfg.Database, cfg.User, cfg.Password, cfg.SSLMode,
	)

	db, err := sql.Open("pgx", connString)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Verify connection
	if err := db.PingContext(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &TimescaleDBBackend{db: db}, nil
}

/**
 * Close closes the database connection pool.
 */
func (t *TimescaleDBBackend) Close() error {
	return t.db.Close()
}

/**
 * UpsertObject inserts or updates an object in the active catalog and TLE history.
 * Uses deduplication logic to avoid storing identical TLEs.
 */
func (t *TimescaleDBBackend) UpsertObject(record catalog.SDAObject) error {
	ctx := context.Background()

	// Parse TLE to extract orbital elements
	tle, err := physics.ParseTLE(record.TLELine1 + "\n" + record.TLELine2)
	if err != nil {
		return fmt.Errorf("failed to parse TLE: %w", err)
	}

	// Convert Julian Date epoch to timestamp
	epochTime := julianDateToTime(tle.EpochYear, tle.EpochDay)

	// Insert into TLE history with deduplication
	_, err = t.db.ExecContext(ctx, `
		SELECT insert_tle_deduplicated($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	`,
		tle.SatelliteNumber,
		epochTime,
		tle.Inclination,
		tle.RAAN,
		tle.Eccentricity,
		tle.ArgOfPerigee,
		tle.MeanAnomaly,
		tle.MeanMotion,
		tle.BStar,
		record.TLELine1,
		record.TLELine2,
	)

	if err != nil {
		return fmt.Errorf("failed to insert TLE: %w", err)
	}

	// Update active catalog
	_, err = t.db.ExecContext(ctx, `
		SELECT update_active_catalog($1)
	`, tle.SatelliteNumber)

	if err != nil {
		return fmt.Errorf("failed to update active catalog: %w", err)
	}

	return nil
}

/**
 * BatchInsertTLEs efficiently inserts multiple TLEs in a single transaction.
 * Used by the ingestion loop for high-throughput updates.
 */
func (t *TimescaleDBBackend) BatchInsertTLEs(tles []*physics.TLE, source string) (int, error) {
	ctx := context.Background()

	// Begin transaction
	tx, err := t.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	var inserted int
	for _, tleObj := range tles {
		if tleObj == nil || tleObj.SatelliteNumber == 0 {
			continue
		}

		epochTime := julianDateToTime(tleObj.EpochYear, tleObj.EpochDay)

		var result bool
		err := tx.QueryRowContext(ctx, `
			SELECT insert_tle_deduplicated($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		`,
			tleObj.SatelliteNumber,
			epochTime,
			tleObj.Inclination,
			tleObj.RAAN,
			tleObj.Eccentricity,
			tleObj.ArgOfPerigee,
			tleObj.MeanAnomaly,
			tleObj.MeanMotion,
			tleObj.BStar,
			tleObj.RawL1,
			tleObj.RawL2,
			source,
		).Scan(&result)

		if err == nil && result {
			inserted++
		}
	}

	// Update active catalogs for all inserted TLEs
	for _, tleObj := range tles {
		if tleObj != nil && tleObj.SatelliteNumber != 0 {
			_, _ = tx.ExecContext(ctx, `SELECT update_active_catalog($1)`, tleObj.SatelliteNumber)
		}
	}

	if err := tx.Commit(); err != nil {
		return inserted, err
	}

	return inserted, nil
}

/**
 * DeleteObject marks an object as decayed (soft delete).
 */
func (t *TimescaleDBBackend) DeleteObject(id int) error {
	ctx := context.Background()

	_, err := t.db.ExecContext(ctx, `
		UPDATE active_catalog 
		SET decayed_at = NOW() 
		WHERE norad_id = $1
	`, id)

	return err
}

/**
 * GetLatestSnapshot retrieves the active catalog.
 */
func (t *TimescaleDBBackend) GetLatestSnapshot() ([]catalog.SDAObject, error) {
	ctx := context.Background()

	rows, err := t.db.QueryContext(ctx, `
		SELECT 
			norad_id, 
			EXTRACT(EPOCH FROM epoch) / 86400.0 + 2440587.5 as epoch_tt,
			raw_line1, 
			raw_line2,
			EXTRACT(EPOCH FROM updated_at) as timestamp
		FROM active_catalog 
		WHERE decayed_at IS NULL
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []catalog.SDAObject
	for rows.Next() {
		var r catalog.SDAObject
		err := rows.Scan(&r.ID, &r.EpochTT, &r.TLELine1, &r.TLELine2, &r.Timestamp)
		if err != nil {
			continue
		}
		records = append(records, r)
	}

	return records, rows.Err()
}

/**
 * GetObjectAtTime retrieves the closest TLE for an object at a specific time.
 */
func (t *TimescaleDBBackend) GetObjectAtTime(id int, queryTime time.Time) (catalog.SDAObject, error) {
	ctx := context.Background()

	var r catalog.SDAObject
	err := t.db.QueryRowContext(ctx, `
		SELECT 
			norad_id,
			EXTRACT(EPOCH FROM epoch) / 86400.0 + 2440587.5 as epoch_tt,
			raw_line1,
			raw_line2,
			EXTRACT(EPOCH FROM updated_at) as timestamp
		FROM tle_history
		WHERE norad_id = $1 AND epoch <= $2
		ORDER BY epoch DESC
		LIMIT 1
	`, id, queryTime).Scan(&r.ID, &r.EpochTT, &r.TLELine1, &r.TLELine2, &r.Timestamp)

	return r, err
}

/**
 * QueryObjectsInWindow retrieves TLEs within a time range.
 */
func (t *TimescaleDBBackend) QueryObjectsInWindow(start, end time.Time) ([]catalog.SDAObject, error) {
	ctx := context.Background()

	rows, err := t.db.QueryContext(ctx, `
		SELECT 
			norad_id,
			EXTRACT(EPOCH FROM epoch) / 86400.0 + 2440587.5 as epoch_tt,
			raw_line1,
			raw_line2,
			EXTRACT(EPOCH FROM updated_at) as timestamp
		FROM tle_history
		WHERE epoch BETWEEN $1 AND $2
		ORDER BY norad_id, epoch
	`, start, end)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []catalog.SDAObject
	for rows.Next() {
		var r catalog.SDAObject
		err := rows.Scan(&r.ID, &r.EpochTT, &r.TLELine1, &r.TLELine2, &r.Timestamp)
		if err != nil {
			continue
		}
		records = append(records, r)
	}

	return records, rows.Err()
}

/**
 * GetTLEHistory retrieves TLE history for B* smoothing.
 */
func (t *TimescaleDBBackend) GetTLEHistory(noradID int, days int) ([]physics.TLE, error) {
	ctx := context.Background()

	rows, err := t.db.QueryContext(ctx, `
		SELECT 
			norad_id,
			EXTRACT(YEAR FROM epoch)::integer as epoch_year,
			EXTRACT(DOY FROM epoch) + EXTRACT(HOUR FROM epoch)/24.0 + EXTRACT(MINUTE FROM epoch)/1440.0 as epoch_day,
			mean_motion,
			b_star,
			raw_line1,
			raw_line2
		FROM tle_history
		WHERE norad_id = $1 AND epoch > NOW() - INTERVAL '$2 days'
		ORDER BY epoch ASC
	`, noradID, days)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var history []physics.TLE
	for rows.Next() {
		var tle physics.TLE
		err := rows.Scan(
			&tle.SatelliteNumber,
			&tle.EpochYear,
			&tle.EpochDay,
			&tle.MeanMotion,
			&tle.BStar,
			&tle.RawL1,
			&tle.RawL2,
		)
		if err != nil {
			continue
		}
		history = append(history, tle)
	}

	return history, rows.Err()
}

/**
 * GetSmoothedBStar retrieves pre-computed B* smoothing from continuous aggregate.
 */
func (t *TimescaleDBBackend) GetSmoothedBStar(noradID int) (*physics.BStarSmoothed, error) {
	ctx := context.Background()

	var bs physics.BStarSmoothed
	err := t.db.QueryRowContext(ctx, `
		SELECT 
			norad_id,
			ballistic_coeff,
			b_star_mean,
			b_star_std,
			bucket as last_updated,
			sample_count as num_samples,
			b_star_slope_per_day as trend_slope,
			CASE WHEN sample_count >= 20 THEN 0.9 ELSE sample_count::float / 20.0 END as confidence
		FROM b_star_smoothed_30d
		WHERE norad_id = $1
		ORDER BY bucket DESC
		LIMIT 1
	`, noradID).Scan(
		&bs.SatelliteNumber,
		&bs.BallisticCoeff,
		&bs.BStarMean,
		&bs.BStarStd,
		&bs.LastUpdated,
		&bs.NumSamples,
		&bs.TrendSlope,
		&bs.Confidence,
	)

	if err != nil {
		return nil, err
	}

	return &bs, nil
}

/**
 * LogManeuver records a detected maneuver in the database.
 */
func (t *TimescaleDBBackend) LogManeuver(noradID int, preEpoch, postEpoch time.Time, deltaV, deltaA, confidence float64) error {
	ctx := context.Background()

	_, err := t.db.ExecContext(ctx, `
		INSERT INTO maneuver_log (
			norad_id, pre_maneuver_tle_epoch, post_maneuver_tle_epoch,
			delta_v, delta_a, confidence, maneuver_type
		) VALUES ($1, $2, $3, $4, $5, $6, 'DELTA_V')
	`, noradID, preEpoch, postEpoch, deltaV, deltaA, confidence)

	return err
}

/**
 * InsertSpaceWeather stores NOAA space weather data.
 */
func (t *TimescaleDBBackend) InsertSpaceWeather(sw physics.SpaceWeatherData) error {
	ctx := context.Background()

	_, err := t.db.ExecContext(ctx, `
		INSERT INTO space_weather (timestamp, f107, f107_avg_81d, ap, kp)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT (timestamp) DO UPDATE SET
			f107 = EXCLUDED.f107,
			ap = EXCLUDED.ap,
			kp = EXCLUDED.kp
	`, sw.Timestamp, sw.F107, sw.F107Avg, sw.Ap, sw.Kp)

	return err
}

/**
 * GetLatestSpaceWeather retrieves the most recent space weather indices.
 */
func (t *TimescaleDBBackend) GetLatestSpaceWeather() (physics.SpaceWeatherData, error) {
	ctx := context.Background()

	var sw physics.SpaceWeatherData
	err := t.db.QueryRowContext(ctx, `
		SELECT timestamp, f107, f107_avg_81d, ap, kp
		FROM space_weather
		ORDER BY timestamp DESC
		LIMIT 1
	`).Scan(&sw.Timestamp, &sw.F107, &sw.F107Avg, &sw.Ap, &sw.Kp)

	return sw, err
}

/**
 * PersistHypothesis persists an MHT hypothesis (not used in TimescaleDB).
 */
func (t *TimescaleDBBackend) PersistHypothesis(hyp *catalog.GlobalHypothesis) error {
	// Not implemented for TimescaleDB - use BoltDB for MHT
	return nil
}

/**
 * Helper: Convert Julian Date (year, day) to time.Time
 */
func julianDateToTime(year int, day float64) time.Time {
	// Julian Day for Jan 1 of the epoch year
	var jdYear0 float64
	if year >= 2000 {
		jdYear0 = 2451545.0 + float64(year-2000)*365.25
	} else {
		jdYear0 = 2451545.0 - float64(2000-year)*365.25
	}

	jd := jdYear0 + day - 1.0

	// Convert Julian Date to Unix timestamp
	// JD 2440587.5 = Unix epoch (1970-01-01 00:00:00)
	unixSeconds := (jd - 2440587.5) * 86400.0

	return time.Unix(int64(unixSeconds), 0)
}

/**
 * Compile-time check for interface compliance
 */
var _ PersistenceBackend = (*TimescaleDBBackend)(nil)
