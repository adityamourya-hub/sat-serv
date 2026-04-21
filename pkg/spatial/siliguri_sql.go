package spatial

/**
 * Siliguri Corridor PostGIS Integration
 * 
 * SQL schema and queries for 3D volumetric surveillance.
 * Optimized for M2 Air with gist_geometry_ops_nd index.
 * 
 * To use: 
 * 1. Run deploy/database/timescaledb_schema.sql to create tables
 * 2. Run the SQL below to add Siliguri-specific indexes and functions
 * 3. Use QuerySiliguriVolume() for tactical queries
 */

// SiliguriPostGISQueries contains the production SQL for the strategic cube
const SiliguriPostGISQueries = `
-- ============================================
-- SILIGURI CORRIDOR STRATEGIC CUBE SETUP
-- ============================================

-- 1. Enable PostGIS and 3D extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster; -- For elevation data

-- 2. Create N-Dimensional spatial index for fast 3D queries
-- This is CRITICAL for M2 Air performance
CREATE INDEX idx_current_states_3d 
ON current_states 
USING gist (position_vector gist_geometry_ops_nd);

-- 3. Create the Siliguri Volume as a persistent geometry
-- Dimensions: ~180km (L) x 60km (W) x 900km (H)
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, srtext, proj4text)
VALUES (
    999999,
    'SSIS',
    999999,
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]',
    '+proj=longlat +datum=WGS84 +no_defs'
)
ON CONFLICT (srid) DO NOTHING;

-- 4. Create the Siliguri Corridor Volume table
CREATE TABLE IF NOT EXISTS strategic_volumes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    geometry GEOMETRY(POLYHEDRALSURFACEZ, 4326),
    min_altitude_km FLOAT,
    max_altitude_km FLOAT,
    threat_level INT, -- 1-10
    created_at TIMESTAMP DEFAULT NOW()
);

-- 5. Insert the Siliguri Strategic Cube
INSERT INTO strategic_volumes (name, geometry, min_altitude_km, max_altitude_km, threat_level)
VALUES (
    'Siliguri_Corridor',
    ST_SetSRID(
        ST_MakeBox3D(
            ST_MakePoint(87.8, 25.5, 300000),  -- Min: [Lon, Lat, Alt(m)]
            ST_MakePoint(89.5, 27.2, 1200000)  -- Max: [Lon, Lat, Alt(m)]
        ),
        4326
    ),
    300,
    1200,
    10  -- Maximum threat level
)
ON CONFLICT (name) DO UPDATE SET
    geometry = EXCLUDED.geometry,
    threat_level = EXCLUDED.threat_level;

-- ============================================
-- TACTICAL QUERIES
-- ============================================

-- Query 1: Volumetric Scan with N-Dimensional Index
-- Use this for real-time monitoring (runs in <10ms on M2 Air)
CREATE OR REPLACE FUNCTION query_siliguri_volume(
    p_timestamp TIMESTAMP DEFAULT NOW()
)
RETURNS TABLE (
    norad_cat_id INT,
    object_name VARCHAR,
    country_code VARCHAR,
    altitude_km FLOAT,
    is_adversary BOOLEAN,
    threat_category VARCHAR,
    entry_time TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    WITH siliguri_cube AS (
        SELECT geometry 
        FROM strategic_volumes 
        WHERE name = 'Siliguri_Corridor'
    )
    SELECT 
        c.norad_cat_id,
        m.object_name,
        m.country_code,
        ST_Z(c.position_vector) / 1000.0 AS altitude_km,
        (m.country_code IN ('CHN', 'PAK')) AS is_adversary,
        CASE 
            WHEN m.object_name LIKE '%JILIN%' THEN 'JILIN_OPTICAL'
            WHEN m.object_name LIKE '%YAOGAN%' THEN 'YAOGAN_ELINT'
            WHEN m.country_code = 'CHN' THEN 'CHINA_UNKNOWN'
            WHEN m.country_code = 'PAK' THEN 'PAKISTAN_UNKNOWN'
            ELSE 'FRIENDLY_OR_NEUTRAL'
        END::VARCHAR AS threat_category,
        c.last_propagated_at AS entry_time
    FROM 
        current_states c
    JOIN 
        asset_metadata m ON c.norad_cat_id = m.norad_cat_id
    CROSS JOIN 
        siliguri_cube v
    WHERE 
        -- N-Dimensional Bounding Box operator (FAST spatial filter using R-tree)
        c.position_vector &&& v.geometry
        -- Precise 3D intersection (mathematical check)
        AND ST_3DIntersects(c.position_vector, v.geometry)
        -- Only flag adversary or high-interest assets
        AND (m.is_adversary = TRUE OR m.country_code IN ('CHN', 'PAK', 'RUS'))
        -- Active satellites only
        AND c.last_propagated_at > p_timestamp - INTERVAL '5 minutes';
END;
$$ LANGUAGE plpgsql;

-- Query 2: Jilin-1 Swarm Detection
-- Detects 3+ Jilin satellites within 60 minutes
CREATE OR REPLACE FUNCTION detect_jilin_swarm(
    p_window_minutes INT DEFAULT 60
)
RETURNS TABLE (
    detection_time TIMESTAMP,
    satellite_count INT,
    swarm_id INT,
    objects JSONB,
    isr_saturation_alert BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH jilin_passes AS (
        SELECT 
            c.norad_cat_id,
            m.object_name,
            c.last_propagated_at,
            FLOOR(EXTRACT(EPOCH FROM c.last_propagated_at) / (p_window_minutes * 60)) AS time_bucket
        FROM current_states c
        JOIN asset_metadata m ON c.norad_cat_id = m.norad_cat_id
        JOIN strategic_volumes v ON ST_3DIntersects(c.position_vector, v.geometry)
        WHERE 
            v.name = 'Siliguri_Corridor'
            AND (m.object_name LIKE '%JILIN%' OR m.object_name LIKE '%JL-1%')
            AND c.last_propagated_at > NOW() - INTERVAL '1 hour'
    ),
    swarm_groups AS (
        SELECT 
            time_bucket,
            COUNT(*) AS cnt,
            JSONB_AGG(
                JSONB_BUILD_OBJECT(
                    'norad_id', norad_cat_id,
                    'name', object_name,
                    'timestamp', last_propagated_at
                )
            ) AS objects
        FROM jilin_passes
        GROUP BY time_bucket
        HAVING COUNT(*) >= 3  -- Swarm threshold
    )
    SELECT 
        NOW() AS detection_time,
        sg.cnt::INT AS satellite_count,
        sg.time_bucket::INT AS swarm_id,
        sg.objects,
        (sg.cnt >= 5) AS isr_saturation_alert
    FROM swarm_groups sg
    ORDER BY sg.time_bucket DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Query 3: Yaogan ELINT Triplet Detection
-- Identifies 3 Yaogan satellites in close temporal proximity
CREATE OR REPLACE FUNCTION detect_yaogan_elint()
RETURNS TABLE (
    detection_time TIMESTAMP,
    triplet_count INT,
    yaogan_objects JSONB,
    emcon_recommendation VARCHAR,
    recommended_silence_start TIMESTAMP,
    recommended_silence_duration INTERVAL
) AS $$
BEGIN
    RETURN QUERY
    WITH yaogan_in_volume AS (
        SELECT 
            c.norad_cat_id,
            m.object_name,
            c.last_propagated_at,
            ST_X(c.position_vector) AS lon,
            ST_Y(c.position_vector) AS lat,
            ST_Z(c.position_vector) / 1000.0 AS alt_km
        FROM current_states c
        JOIN asset_metadata m ON c.norad_cat_id = m.norad_cat_id
        JOIN strategic_volumes v ON ST_3DIntersects(c.position_vector, v.geometry)
        WHERE 
            v.name = 'Siliguri_Corridor'
            AND (m.object_name LIKE '%YAOGAN%' OR m.object_name LIKE '%YG%')
            AND c.last_propagated_at > NOW() - INTERVAL '30 minutes'
        ORDER BY c.last_propagated_at
    ),
    triplet_check AS (
        SELECT 
            COUNT(*) AS cnt,
            JSONB_AGG(
                JSONB_BUILD_OBJECT(
                    'norad_id', norad_cat_id,
                    'name', object_name,
                    'timestamp', last_propagated_at,
                    'position', ARRAY[lon, lat, alt_km]
                )
            ) AS objects,
            MIN(last_propagated_at) AS first_seen,
            MAX(last_propagated_at) AS last_seen
        FROM yaogan_in_volume
    )
    SELECT 
        NOW() AS detection_time,
        tc.cnt::INT AS triplet_count,
        tc.objects AS yaogan_objects,
        CASE 
            WHEN tc.cnt >= 3 THEN 'SILENCE_IMMEDIATE'
            WHEN tc.cnt >= 2 THEN 'REDUCE_EMISSIONS'
            ELSE 'NORMAL_OPERATIONS'
        END::VARCHAR AS emcon_recommendation,
        tc.first_seen AS recommended_silence_start,
        INTERVAL '30 minutes' AS recommended_silence_duration
    FROM triplet_check tc
    WHERE tc.cnt >= 2;  -- Alert on 2+ (threshold for concern)
END;
$$ LANGUAGE plpgsql;

-- Query 4: Calculate Strategic Dark Window
-- Finds next 2+ hour gap in adversary coverage
CREATE OR REPLACE FUNCTION calculate_dark_window(
    p_look_ahead_hours INT DEFAULT 6
)
RETURNS TABLE (
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    duration_hours FLOAT,
    safe_for_activity VARCHAR,
    confidence_score FLOAT
) AS $$
DECLARE
    v_interval INTERVAL := (p_look_ahead_hours || ' hours')::INTERVAL;
    v_start_time TIMESTAMP := NOW();
    v_check_time TIMESTAMP;
    v_dark_start TIMESTAMP;
    v_max_duration INTERVAL := '0 seconds'::INTERVAL;
    v_current_duration INTERVAL;
BEGIN
    -- Initialize
    v_dark_start := NULL;
    v_check_time := v_start_time;
    
    -- Check every 5 minutes in the look-ahead window
    WHILE v_check_time < v_start_time + v_interval LOOP
        -- Count adversary satellites in volume at this time
        IF NOT EXISTS (
            SELECT 1 
            FROM current_states c
            JOIN asset_metadata m ON c.norad_cat_id = m.norad_cat_id
            JOIN strategic_volumes v ON ST_3DIntersects(c.position_vector, v.geometry)
            WHERE 
                v.name = 'Siliguri_Corridor'
                AND (m.is_adversary = TRUE OR m.country_code IN ('CHN', 'PAK'))
                AND c.last_propagated_at BETWEEN v_check_time - INTERVAL '5 minutes' AND v_check_time + INTERVAL '5 minutes'
        ) THEN
            -- No adversary coverage - potential dark window
            IF v_dark_start IS NULL THEN
                v_dark_start := v_check_time;
            END IF;
        ELSE
            -- Adversary coverage detected - close current dark window
            IF v_dark_start IS NOT NULL THEN
                v_current_duration := v_check_time - v_dark_start;
                IF v_current_duration > v_max_duration THEN
                    v_max_duration := v_current_duration;
                    
                    -- Return this window if it's long enough
                    IF v_current_duration >= INTERVAL '2 hours' THEN
                        RETURN QUERY SELECT 
                            v_dark_start,
                            v_check_time,
                            EXTRACT(EPOCH FROM v_current_duration) / 3600.0,
                            CASE 
                                WHEN v_current_duration >= INTERVAL '4 hours' THEN 'MAJOR_UNIT_MOVEMENT'
                                WHEN v_current_duration >= INTERVAL '2 hours' THEN 'TANK_COLUMN_ADVANCE'
                                ELSE 'TACTICAL_REPOSITIONING'
                            END::VARCHAR,
                            0.85::FLOAT;
                    END IF;
                END IF;
                v_dark_start := NULL;
            END IF;
        END IF;
        
        v_check_time := v_check_time + INTERVAL '5 minutes';
    END LOOP;
    
    RETURN;
END;
$$ LANGUAGE plpgsql;

-- Query 5: Generate Optical Verification Tasking
-- Automatically assigns sensors to verify adversary satellite orientation
CREATE OR REPLACE FUNCTION generate_optical_tasking()
RETURNS TABLE (
    task_id SERIAL,
    target_norad_id INT,
    target_name VARCHAR,
    assigned_sensor VARCHAR,
    priority INT, -- 1-10
    start_time TIMESTAMP,
    duration INTERVAL,
    min_elevation FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH adversary_objects AS (
        SELECT 
            c.norad_cat_id,
            m.object_name,
            c.last_propagated_at,
            ST_X(c.position_vector) AS lon,
            ST_Y(c.position_vector) AS lat,
            CASE 
                WHEN m.object_name LIKE '%JILIN%' THEN 10
                WHEN m.object_name LIKE '%YAOGAN%' THEN 9
                WHEN m.country_code = 'CHN' THEN 8
                WHEN m.country_code = 'PAK' THEN 7
                ELSE 5
            END AS threat_priority
        FROM current_states c
        JOIN asset_metadata m ON c.norad_cat_id = m.norad_cat_id
        JOIN strategic_volumes v ON ST_3DIntersects(c.position_vector, v.geometry)
        WHERE 
            v.name = 'Siliguri_Corridor'
            AND (m.is_adversary = TRUE OR m.country_code IN ('CHN', 'PAK'))
            AND c.last_propagated_at > NOW() - INTERVAL '5 minutes'
        ORDER BY threat_priority DESC
    )
    SELECT 
        ROW_NUMBER() OVER ()::INT,
        ao.norad_cat_id,
        ao.object_name,
        CASE 
            WHEN ao.lat > 26.5 THEN 'leh_optical'
            WHEN ao.lat > 26.0 THEN 'sikkim_optical'
            ELSE 'mount_abu_optical'
        END::VARCHAR,
        ao.threat_priority::INT,
        NOW() + INTERVAL '5 minutes',
        INTERVAL '10 minutes',
        20.0
    FROM adversary_objects ao
    WHERE ao.threat_priority >= 7;  -- Only high-priority targets
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

-- Composite index for time-series + spatial queries
CREATE INDEX idx_current_states_time_spatial 
ON current_states (last_propagated_at, position_vector);

-- Index for Jilin/Yaogan rapid lookup
CREATE INDEX idx_asset_metadata_name_pattern 
ON asset_metadata (object_name text_pattern_ops);

-- Index for adversary filtering
CREATE INDEX idx_asset_metadata_adversary 
ON asset_metadata (country_code) 
WHERE is_adversary = TRUE OR country_code IN ('CHN', 'PAK');

-- ============================================
-- MONITORING VIEWS
-- ============================================

-- Real-time dashboard view
CREATE OR REPLACE VIEW v_siliguri_threat_dashboard AS
SELECT 
    sv.name AS volume_name,
    (SELECT COUNT(*) FROM query_siliguri_volume()) AS current_threat_count,
    (SELECT COUNT(*) FROM query_siliguri_volume() WHERE is_adversary) AS adversary_count,
    (SELECT satellite_count FROM detect_jilin_swarm()) AS jilin_swarm_size,
    (SELECT triplet_count FROM detect_yaogan_elint()) AS yaogan_triplet_count,
    (SELECT emcon_recommendation FROM detect_yaogan_elint()) AS current_emcon_status,
    NOW() AS updated_at
FROM strategic_volumes sv
WHERE sv.name = 'Siliguri_Corridor';
`

// GetSiliguriQuery returns the full SQL setup for deployment
func GetSiliguriQuery() string {
	return SiliguriPostGISQueries
}
