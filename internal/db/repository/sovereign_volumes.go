// Package repository provides sovereign volume monitoring for IAF strategic areas
// Tactical areas: Siliguri Corridor (Chicken's Neck) and Galwan Valley
package repository

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// SovereignVolume defines a strategically monitored 3D volume
type SovereignVolume struct {
	Name        string
	Description string
	MinLon      float64
	MaxLon      float64
	MinLat      float64
	MaxLat      float64
	MinAltKm    float64
	MaxAltKm    float64
	AlertLevel  string // LOW, MEDIUM, HIGH, CRITICAL
}

// StrategicVolumes returns predefined IAF strategic monitoring zones
func StrategicVolumes() []SovereignVolume {
	return []SovereignVolume{
		{
			Name:        "Siliguri_Corridor_LEO",
			Description: "Siliguri Corridor (Chicken's Neck) - LEO ISR monitoring zone (Jilin-1, Yaogan swarms)",
			MinLon:      87.8,
			MaxLon:      89.5,
			MinLat:      25.5,
			MaxLat:      27.2,
			MinAltKm:    300,  // LEO floor (300km)
			MaxAltKm:    1200, // LEO ceiling (1200km) - Spy satellite zone
			AlertLevel:  "CRITICAL",
		},
		{
			Name:        "Galwan_Valley",
			Description: "Galwan Valley - LAC flashpoint region",
			MinLon:      76.5,
			MaxLon:      78.5,
			MinLat:      34.0,
			MaxLat:      35.0,
			MinAltKm:    0,
			MaxAltKm:    2000,
			AlertLevel:  "HIGH",
		},
		{
			Name:        "IAF_Test_Range",
			Description: "Integrated Test Range ( Abdul Kalam Island ) - Missile test monitoring",
			MinLon:      86.5,
			MaxLon:      88.0,
			MinLat:      20.5,
			MaxLat:      21.5,
			MinAltKm:    0,
			MaxAltKm:    1000,
			AlertLevel:  "HIGH",
		},
		{
			Name:        "Sriharikota_Range",
			Description: "Satish Dhawan Space Centre - Launch monitoring",
			MinLon:      80.0,
			MaxLon:      81.0,
			MinLat:      13.5,
			MaxLat:      14.5,
			MinAltKm:    0,
			MaxAltKm:    1000,
			AlertLevel:  "MEDIUM",
		},
	}
}

// SovereignVolumeRepo provides sovereign volume monitoring queries
type SovereignVolumeRepo struct {
	pool *pgxpool.Pool
}

// NewSovereignVolumeRepo creates repository instance
func NewSovereignVolumeRepo(pool *pgxpool.Pool) *SovereignVolumeRepo {
	return &SovereignVolumeRepo{pool: pool}
}

// IntrusionResult represents an adversary satellite in sovereign volume
type IntrusionResult struct {
	VolumeName  string
	NoradCatID  int32
	ObjectName  string
	CountryCode string
	Longitude   float64
	Latitude    float64
	AltitudeKm  float64
	Velocity    [3]float64
	DetectedAt  time.Time
	ThreatLevel string // Derived from asset tag
}

// ScanVolumeForAdversaries detects adversary satellites in a sovereign volume
func (r *SovereignVolumeRepo) ScanVolumeForAdversaries(ctx context.Context, volume SovereignVolume) ([]IntrusionResult, error) {
	query := `
		SELECT 
			$1::TEXT as volume_name,
			cs.norad_cat_id,
			cs.object_name,
			am.country_code,
			ST_X(cs.position) as longitude,
			ST_Y(cs.position) as latitude,
			ST_Z(cs.position) / 1000.0 as altitude_km,
			cs.velocity_vector,
			cs.last_updated,
			CASE 
				WHEN am.country_code IN ('US', 'RUS', 'CHN') THEN 'HIGH'
				WHEN am.country_code NOT IN ('IND') THEN 'MEDIUM'
				ELSE 'FRIENDLY'
			END as threat_level
		FROM current_states cs
		JOIN asset_metadata am ON cs.norad_cat_id = am.norad_cat_id
		WHERE cs.position &&& ST_SetSRID(ST_MakeEnvelope(
			$2, $3, $4 * 1000.0,  -- min_lon, min_lat, min_alt (m)
			$5, $6, $7 * 1000.0   -- max_lon, max_lat, max_alt (m)
		), 4326)
		AND am.is_indian_asset = FALSE  -- Only non-Indian assets
		AND am.country_code IS NOT NULL
		AND cs.last_updated > NOW() - INTERVAL '5 minutes'
		ORDER BY threat_level DESC, cs.last_updated DESC
	`

	rows, err := r.pool.Query(ctx, query,
		volume.Name, volume.MinLon, volume.MinLat, volume.MinAltKm,
		volume.MaxLon, volume.MaxLat, volume.MaxAltKm,
	)
	if err != nil {
		return nil, fmt.Errorf("scan volume %s: %w", volume.Name, err)
	}
	defer rows.Close()

	return r.scanIntrusions(rows)
}

// ScanAllVolumes checks all strategic volumes for adversary presence
func (r *SovereignVolumeRepo) ScanAllVolumes(ctx context.Context) (map[string][]IntrusionResult, error) {
	volumes := StrategicVolumes()
	results := make(map[string][]IntrusionResult)

	for _, volume := range volumes {
		intrusions, err := r.ScanVolumeForAdversaries(ctx, volume)
		if err != nil {
			logf("Failed to scan volume %s: %v", volume.Name, err)
			continue
		}
		results[volume.Name] = intrusions
	}

	return results, nil
}

// GetSovereignAssetCoverage returns Indian assets visible in a volume
func (r *SovereignVolumeRepo) GetSovereignAssetCoverage(ctx context.Context, volume SovereignVolume) ([]*StateVector, error) {
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
		WHERE cs.position &&& ST_SetSRID(ST_MakeEnvelope(
			$1, $2, $3 * 1000.0,
			$4, $5, $6 * 1000.0
		), 4326)
		AND am.is_indian_asset = TRUE
		AND cs.last_updated > NOW() - INTERVAL '5 minutes'
		ORDER BY am.priority_level ASC
	`

	rows, err := r.pool.Query(ctx, query,
		volume.MinLon, volume.MinLat, volume.MinAltKm,
		volume.MaxLon, volume.MaxLat, volume.MaxAltKm,
	)
	if err != nil {
		return nil, fmt.Errorf("get sovereign coverage: %w", err)
	}
	defer rows.Close()

	return r.scanStateVectors(rows)
}

// VolumetricScan3D performs precise 3D spatial intersection using PostGIS N-D index
// Uses &&& for fast bounding box filter and ST_3DIntersects for precision
func (r *SovereignVolumeRepo) VolumetricScan3D(ctx context.Context, volume SovereignVolume,
	countries []string, minTimeInVolume time.Duration) ([]IntrusionResult, error) {

	query := `
		WITH siliguri_box AS (
			SELECT ST_SetSRID(
				ST_MakeBox3D(
					ST_MakePoint($1, $2, $3 * 1000), -- MinLon, MinLat, MinAlt(m)
					ST_MakePoint($4, $5, $6 * 1000)  -- MaxLon, MaxLat, MaxAlt(m)
				), 4326
			) AS geom
		)
		SELECT 
			$7::TEXT as volume_name,
			cs.norad_cat_id,
			cs.object_name,
			am.country_code,
			ST_X(cs.position) as longitude,
			ST_Y(cs.position) as latitude,
			ST_Z(cs.position) / 1000.0 as altitude_km,
			cs.velocity_vector,
			cs.last_updated,
			CASE 
				WHEN am.country_code = 'CHN' THEN 'HIGH'
				WHEN am.country_code = 'PAK' THEN 'HIGH'
				WHEN am.is_adversary = TRUE THEN 'MEDIUM'
				ELSE 'LOW'
			END as threat_level
		FROM current_states cs
		JOIN asset_metadata am ON cs.norad_cat_id = am.norad_cat_id,
		siliguri_box v
		WHERE 
			-- N-Dimensional Bounding Box operator (Fast spatial filter)
			cs.position &&& v.geom
			-- Mathematical 3D Intersection (Precise check)
			AND ST_3DIntersects(cs.position, v.geom)
			-- Policy Filter: Only flag Adversary or High-Interest assets
			AND (am.is_adversary = TRUE OR am.country_code = ANY($8))
			-- Only recent propagation data
			AND cs.last_updated > NOW() - INTERVAL '5 minutes'
		ORDER BY threat_level DESC, cs.last_updated DESC
	`

	// Convert country codes to PostgreSQL array
	countriesAny := "{}"
	if len(countries) > 0 {
		countriesAny = "{" + countries[0]
		for _, c := range countries[1:] {
			countriesAny += "," + c
		}
		countriesAny += "}"
	}

	rows, err := r.pool.Query(ctx, query,
		volume.MinLon, volume.MinLat, volume.MinAltKm,
		volume.MaxLon, volume.MaxLat, volume.MaxAltKm,
		volume.Name,
		countriesAny,
	)
	if err != nil {
		return nil, fmt.Errorf("3D volumetric scan: %w", err)
	}
	defer rows.Close()

	return r.scanIntrusions(rows)
}

// ISRSaturationAlert detects when 3+ satellites enter volume within time window
// This indicates Jilin-1 swarm or coordinated ISR activity
func (r *SovereignVolumeRepo) ISRSaturationAlert(ctx context.Context, volume SovereignVolume,
	windowMinutes int, thresholdCount int) (*SaturationAlert, error) {

	query := `
		SELECT 
			cs.norad_cat_id,
			cs.object_name,
			am.country_code,
			cs.last_updated,
			ST_Z(cs.position) / 1000.0 as alt_km
		FROM current_states cs
		JOIN asset_metadata am ON cs.norad_cat_id = am.norad_cat_id
		WHERE cs.position &&& ST_SetSRID(ST_MakeEnvelope(
			$1, $2, $3 * 1000.0,
			$4, $5, $6 * 1000.0
		), 4326)
		AND am.is_adversary = TRUE
		AND cs.last_updated > NOW() - ($7 || ' minutes')::INTERVAL
		ORDER BY cs.last_updated DESC
	`

	rows, err := r.pool.Query(ctx, query,
		volume.MinLon, volume.MinLat, volume.MinAltKm,
		volume.MaxLon, volume.MaxLat, volume.MaxAltKm,
		windowMinutes,
	)
	if err != nil {
		return nil, fmt.Errorf("saturation check: %w", err)
	}
	defer rows.Close()

	var satellites []SatellitePass
	for rows.Next() {
		var s SatellitePass
		err := rows.Scan(&s.NoradCatID, &s.ObjectName, &s.CountryCode, &s.Timestamp, &s.AltitudeKm)
		if err != nil {
			continue
		}
		satellites = append(satellites, s)
	}

	if len(satellites) >= thresholdCount {
		return &SaturationAlert{
			VolumeName:        volume.Name,
			SatelliteCount:    len(satellites),
			Threshold:         thresholdCount,
			TimeWindow:        time.Duration(windowMinutes) * time.Minute,
			Satellites:        satellites,
			AlertLevel:        "CRITICAL",
			RecommendedAction: "EMCON_ORDER", // Emission Control
			DetectedAt:        time.Now(),
		}, nil
	}

	return nil, nil // No saturation
}

// SatellitePass represents a satellite passing through a volume
type SatellitePass struct {
	NoradCatID  int32
	ObjectName  string
	CountryCode string
	Timestamp   time.Time
	AltitudeKm  float64
}

// SaturationAlert represents ISR saturation detection
type SaturationAlert struct {
	VolumeName        string
	SatelliteCount    int
	Threshold         int
	TimeWindow        time.Duration
	Satellites        []SatellitePass
	AlertLevel        string
	RecommendedAction string
	DetectedAt        time.Time
}

// logf helper to avoid circular import
func logf(format string, v ...interface{}) {
	fmt.Printf(format+"\n", v...)
}

// scanIntrusions reads intrusion results from rows
func (r *SovereignVolumeRepo) scanIntrusions(rows pgx.Rows) ([]IntrusionResult, error) {
	var intrusions []IntrusionResult

	for rows.Next() {
		var i IntrusionResult
		var velSlice []float64

		err := rows.Scan(
			&i.VolumeName, &i.NoradCatID, &i.ObjectName, &i.CountryCode,
			&i.Longitude, &i.Latitude, &i.AltitudeKm,
			&velSlice, &i.DetectedAt, &i.ThreatLevel,
		)
		if err != nil {
			continue // Skip malformed rows
		}

		if len(velSlice) >= 3 {
			copy(i.Velocity[:], velSlice)
		}

		intrusions = append(intrusions, i)
	}

	return intrusions, rows.Err()
}

// scanStateVectors reads state vectors from rows
func (r *SovereignVolumeRepo) scanStateVectors(rows pgx.Rows) ([]*StateVector, error) {
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
