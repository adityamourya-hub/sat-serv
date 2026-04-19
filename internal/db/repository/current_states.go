// Package repository provides data access layer for ISDA database
// All queries use pgx parameterized statements (IAF security requirement)
package repository

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// StateVector represents a satellite's position and velocity
type StateVector struct {
	NoradCatID        int32
	ObjectID          string
	ObjectName        string
	Epoch             time.Time
	MeanMotion        float64
	Eccentricity      float64
	Inclination       float64
	RAAN              float64
	ArgOfPerigee      float64
	MeanAnomaly       float64
	BStar             float64
	Position          [3]float64  // Lon, Lat, Alt (km)
	Velocity          [3]float64  // km/s
	Covariance        [21]float64 // Upper triangle of 6x6 matrix
	Owner             string
	ObjectType        string
	OperationalStatus string
	LastUpdated       time.Time
	TLELine1          string
	TLELine2          string
}

// VolumetricScanResult from 3D spatial query
type VolumetricScanResult struct {
	NoradCatID  int32
	ObjectName  string
	Longitude   float64
	Latitude    float64
	AltitudeKm  float64
	Velocity    [3]float64
	LastUpdated time.Time
}

// CurrentStatesRepo provides data access for current_states table
type CurrentStatesRepo struct {
	pool *pgxpool.Pool
}

// NewCurrentStatesRepo creates repository instance
func NewCurrentStatesRepo(pool *pgxpool.Pool) *CurrentStatesRepo {
	return &CurrentStatesRepo{pool: pool}
}

// GetByNoradID retrieves satellite state by NORAD catalog ID
func (r *CurrentStatesRepo) GetByNoradID(ctx context.Context, noradID int32) (*StateVector, error) {
	query := `
		SELECT 
			norad_cat_id, object_id, object_name, epoch,
			mean_motion, eccentricity, inclination, raan, arg_of_perigee, mean_anomaly, bstar,
			ST_X(position), ST_Y(position), ST_Z(position) / 1000.0,
			velocity_vector, covariance,
			owner, object_type, operational_status, last_updated, tle_line1, tle_line2
		FROM current_states
		WHERE norad_cat_id = $1
	`

	var sv StateVector
	var lon, lat, alt float64
	var velocitySlice []float64
	var covSlice []float64

	err := r.pool.QueryRow(ctx, query, noradID).Scan(
		&sv.NoradCatID, &sv.ObjectID, &sv.ObjectName, &sv.Epoch,
		&sv.MeanMotion, &sv.Eccentricity, &sv.Inclination, &sv.RAAN,
		&sv.ArgOfPerigee, &sv.MeanAnomaly, &sv.BStar,
		&lon, &lat, &alt,
		&velocitySlice, &covSlice,
		&sv.Owner, &sv.ObjectType, &sv.OperationalStatus,
		&sv.LastUpdated, &sv.TLELine1, &sv.TLELine2,
	)
	if err != nil {
		return nil, fmt.Errorf("query current_states: %w", err)
	}

	sv.Position = [3]float64{lon, lat, alt}
	if len(velocitySlice) >= 3 {
		copy(sv.Velocity[:], velocitySlice)
	}
	if len(covSlice) >= 21 {
		copy(sv.Covariance[:], covSlice)
	}

	return &sv, nil
}

// GetAllForPropagation retrieves all states needing propagation (older than threshold)
func (r *CurrentStatesRepo) GetAllForPropagation(ctx context.Context, ageThreshold time.Duration, batchSize int32) ([]*StateVector, error) {
	query := `
		SELECT 
			norad_cat_id, object_id, object_name, epoch,
			mean_motion, eccentricity, inclination, raan, arg_of_perigee, mean_anomaly, bstar,
			ST_X(position), ST_Y(position), ST_Z(position) / 1000.0,
			velocity_vector, covariance,
			owner, object_type, operational_status, last_updated, tle_line1, tle_line2
		FROM current_states
		WHERE last_updated < NOW() - $1::INTERVAL
		ORDER BY last_updated ASC
		LIMIT $2
	`

	rows, err := r.pool.Query(ctx, query, ageThreshold, batchSize)
	if err != nil {
		return nil, fmt.Errorf("query for propagation: %w", err)
	}
	defer rows.Close()

	return r.scanStateVectors(rows)
}

// GetIndianAssetsForPropagation retrieves Indian sovereign assets (Tier 1) needing propagation
// These are processed with highest priority in the three-tier strategy
func (r *CurrentStatesRepo) GetIndianAssetsForPropagation(ctx context.Context, ageThreshold time.Duration) ([]*StateVector, error) {
	query := `
		SELECT 
			cs.norad_cat_id, cs.object_id, cs.object_name, cs.epoch,
			cs.mean_motion, cs.eccentricity, cs.inclination, cs.raan, 
			cs.arg_of_perigee, cs.mean_anomaly, cs.bstar,
			ST_X(cs.position), ST_Y(cs.position), ST_Z(cs.position) / 1000.0,
			cs.velocity_vector, cs.covariance,
			cs.owner, cs.object_type, cs.operational_status, 
			cs.last_updated, cs.tle_line1, cs.tle_line2
		FROM current_states cs
		JOIN asset_metadata am ON cs.norad_cat_id = am.norad_cat_id
		WHERE am.is_indian_asset = TRUE
		  AND cs.last_updated < NOW() - $1::INTERVAL
		ORDER BY am.priority_level ASC, cs.last_updated ASC
		LIMIT 500  -- Cap at 500 Indian assets (current count ~50)
	`

	rows, err := r.pool.Query(ctx, query, ageThreshold)
	if err != nil {
		return nil, fmt.Errorf("query indian assets for propagation: %w", err)
	}
	defer rows.Close()

	return r.scanStateVectors(rows)
}

// UpdateState updates propagated position and velocity
func (r *CurrentStatesRepo) UpdateState(ctx context.Context, sv *StateVector) error {
	query := `
		UPDATE current_states
		SET 
			position = ST_SetSRID(ST_MakePoint($1, $2, $3 * 1000.0), 4326),
			velocity_vector = ARRAY[$4, $5, $6],
			last_updated = NOW()
		WHERE norad_cat_id = $7
	`

	_, err := r.pool.Exec(ctx, query,
		sv.Position[0], sv.Position[1], sv.Position[2], // Lon, Lat, Alt(km->m)
		sv.Velocity[0], sv.Velocity[1], sv.Velocity[2],
		sv.NoradCatID,
	)
	if err != nil {
		return fmt.Errorf("update state for NORAD %d: %w", sv.NoradCatID, err)
	}

	return nil
}

// UpdateTLE updates full TLE and orbital elements (from Space-Track ingestion)
func (r *CurrentStatesRepo) UpdateTLE(ctx context.Context, sv *StateVector) error {
	query := `
		INSERT INTO current_states (
			norad_cat_id, object_id, object_name, epoch,
			mean_motion, eccentricity, inclination, raan, arg_of_perigee, mean_anomaly, bstar,
			position, velocity_vector, covariance,
			owner, object_type, operational_status, last_updated, tle_line1, tle_line2
		) VALUES (
			$1, $2, $3, $4,
			$5, $6, $7, $8, $9, $10, $11,
			ST_SetSRID(ST_MakePoint($12, $13, $14 * 1000.0), 4326),
			ARRAY[$15, $16, $17], $18,
			$19, $20, $21, NOW(), $22, $23
		)
		ON CONFLICT (norad_cat_id) DO UPDATE SET
			object_id = EXCLUDED.object_id,
			object_name = EXCLUDED.object_name,
			epoch = EXCLUDED.epoch,
			mean_motion = EXCLUDED.mean_motion,
			eccentricity = EXCLUDED.eccentricity,
			inclination = EXCLUDED.inclination,
			raan = EXCLUDED.raan,
			arg_of_perigee = EXCLUDED.arg_of_perigee,
			mean_anomaly = EXCLUDED.mean_anomaly,
			bstar = EXCLUDED.bstar,
			position = EXCLUDED.position,
			velocity_vector = EXCLUDED.velocity_vector,
			covariance = EXCLUDED.covariance,
			owner = EXCLUDED.owner,
			object_type = EXCLUDED.object_type,
			operational_status = EXCLUDED.operational_status,
			last_updated = NOW(),
			tle_line1 = EXCLUDED.tle_line1,
			tle_line2 = EXCLUDED.tle_line2
	`

	_, err := r.pool.Exec(ctx, query,
		sv.NoradCatID, sv.ObjectID, sv.ObjectName, sv.Epoch,
		sv.MeanMotion, sv.Eccentricity, sv.Inclination, sv.RAAN,
		sv.ArgOfPerigee, sv.MeanAnomaly, sv.BStar,
		sv.Position[0], sv.Position[1], sv.Position[2], // Lon, Lat, Alt(km)
		sv.Velocity[0], sv.Velocity[1], sv.Velocity[2],
		sv.Covariance[:],
		sv.Owner, sv.ObjectType, sv.OperationalStatus,
		sv.TLELine1, sv.TLELine2,
	)
	if err != nil {
		return fmt.Errorf("upsert TLE for NORAD %d: %w", sv.NoradCatID, err)
	}

	return nil
}

// VolumetricScanBBox queries 3D bounding box using &&& operator
func (r *CurrentStatesRepo) VolumetricScanBBox(
	ctx context.Context,
	minLon, minLat, minAltKm float64,
	maxLon, maxLat, maxAltKm float64,
) ([]VolumetricScanResult, error) {
	query := `
		SELECT 
			norad_cat_id,
			object_name,
			ST_X(position) as longitude,
			ST_Y(position) as latitude,
			ST_Z(position) / 1000.0 as altitude_km,
			velocity_vector,
			last_updated
		FROM current_states
		WHERE position &&& ST_SetSRID(ST_MakeEnvelope(
			$1, $2, $3 * 1000.0,  -- min_lon, min_lat, min_alt (m)
			$4, $5, $6 * 1000.0   -- max_lon, max_lat, max_alt (m)
		), 4326)
		AND last_updated > NOW() - INTERVAL '5 minutes'
		ORDER BY last_updated DESC
	`

	rows, err := r.pool.Query(ctx, query, minLon, minLat, minAltKm, maxLon, maxLat, maxAltKm)
	if err != nil {
		return nil, fmt.Errorf("volumetric scan bbox: %w", err)
	}
	defer rows.Close()

	var results []VolumetricScanResult
	for rows.Next() {
		var r VolumetricScanResult
		var velSlice []float64

		err := rows.Scan(
			&r.NoradCatID, &r.ObjectName,
			&r.Longitude, &r.Latitude, &r.AltitudeKm,
			&velSlice, &r.LastUpdated,
		)
		if err != nil {
			continue // Skip bad rows
		}

		if len(velSlice) >= 3 {
			copy(r.Velocity[:], velSlice)
		}
		results = append(results, r)
	}

	return results, rows.Err()
}

// VolumetricScanCone queries sensor cone (simplified - uses bounding box approximation)
func (r *CurrentStatesRepo) VolumetricScanCone(
	ctx context.Context,
	originLon, originLat, originAltKm float64,
	azimuthDeg, elevationDeg float64,
	fovHDeg, fovVDeg float64,
	rangeKm float64,
) ([]VolumetricScanResult, error) {
	// Convert sensor cone to approximate bounding box
	// Full implementation would use proper cone-geometry intersection

	// Approximate: 1 deg = 111 km at equator
	rangeDeg := rangeKm / 111.0

	minLon := originLon - rangeDeg
	maxLon := originLon + rangeDeg
	minLat := originLat - rangeDeg
	maxLat := originLat + rangeDeg
	minAlt := originAltKm - rangeKm*0.3 // Approximate
	maxAlt := originAltKm + rangeKm*0.3

	return r.VolumetricScanBBox(ctx, minLon, minLat, minAlt, maxLon, maxLat, maxAlt)
}

// Helper function to scan state vectors from rows
func (r *CurrentStatesRepo) scanStateVectors(rows pgx.Rows) ([]*StateVector, error) {
	var states []*StateVector

	for rows.Next() {
		var sv StateVector
		var lon, lat, alt float64
		var velocitySlice []float64
		var covSlice []float64

		err := rows.Scan(
			&sv.NoradCatID, &sv.ObjectID, &sv.ObjectName, &sv.Epoch,
			&sv.MeanMotion, &sv.Eccentricity, &sv.Inclination, &sv.RAAN,
			&sv.ArgOfPerigee, &sv.MeanAnomaly, &sv.BStar,
			&lon, &lat, &alt,
			&velocitySlice, &covSlice,
			&sv.Owner, &sv.ObjectType, &sv.OperationalStatus,
			&sv.LastUpdated, &sv.TLELine1, &sv.TLELine2,
		)
		if err != nil {
			continue // Skip malformed rows
		}

		sv.Position = [3]float64{lon, lat, alt}
		if len(velocitySlice) >= 3 {
			copy(sv.Velocity[:], velocitySlice)
		}
		if len(covSlice) >= 21 {
			copy(sv.Covariance[:], covSlice)
		}

		states = append(states, &sv)
	}

	return states, rows.Err()
}
