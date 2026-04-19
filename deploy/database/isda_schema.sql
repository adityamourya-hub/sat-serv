-- ISDA Database Schema
-- IAF Space Domain Analytics - PostGIS 3D/4D Spatial Database
-- 
-- Extensions Required:
--   CREATE EXTENSION IF NOT EXISTS postgis;
--   CREATE EXTENSION IF NOT EXISTS timescaledb;
--
-- Coordinate System: WGS84 (EPSG:4326) for PointZ (Lon, Lat, Alt)

-- ============================================================
-- CORE STATE TABLES
-- ============================================================

-- Real-time satellite states with 3D spatial indexing
CREATE TABLE IF NOT EXISTS current_states (
    norad_cat_id INTEGER PRIMARY KEY,
    object_id VARCHAR(16),                    -- International Designator YYYY-XXXCCC
    object_name VARCHAR(64),
    
    -- SGP4 Orbital Elements (TLE-derived)
    epoch TIMESTAMPTZ NOT NULL,
    mean_motion DOUBLE PRECISION,             -- rev/day
    eccentricity DOUBLE PRECISION,
    inclination DOUBLE PRECISION,             -- degrees
    raan DOUBLE PRECISION,                    -- Right Ascension of Ascending Node (deg)
    arg_of_perigee DOUBLE PRECISION,          -- degrees
    mean_anomaly DOUBLE PRECISION,            -- degrees
    bstar DOUBLE PRECISION,                   -- Drag term (1/Earth radii)
    
    -- 3D Position with PostGIS PointZ (Lon, Lat, Alt in meters)
    -- Altitude stored in meters (multiply km by 1000)
    position GEOMETRY(PointZ, 4326),
    
    -- Velocity vector [vx, vy, vz] in km/s
    velocity_vector DOUBLE PRECISION[3],
    
    -- Covariance matrix (6x6 symmetric, stored as 21-element upper triangle)
    -- Order: [xx, xy, xz, xvx, xvy, xvz, yy, yz, yvx, yvy, yvz, zz, zvx, zvy, zvz, vxvx, vxvy, vxvz, vyvy, vyvz, vzvz]
    covariance DOUBLE PRECISION[21],
    
    -- Metadata
    owner VARCHAR(32),                        -- Country/organization code
    object_type VARCHAR(16),                  -- PAYLOAD, DEBRIS, ROCKET_BODY, UNKNOWN
    operational_status VARCHAR(16),           -- OPERATIONAL, NON_OPERATIONAL, DECAYED
    
    -- Timestamps
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    tle_line1 TEXT,                           -- Raw TLE line 1 for provenance
    tle_line2 TEXT                            -- Raw TLE line 2 for provenance
);

-- 3D Bounding Box GIST Index for &&& operator queries
CREATE INDEX IF NOT EXISTS idx_current_states_position_3d 
ON current_states USING GIST (position gist_geometry_ops_nd);

-- Standard B-tree indexes for common queries
CREATE INDEX IF NOT EXISTS idx_current_states_epoch 
ON current_states(epoch DESC);

CREATE INDEX IF NOT EXISTS idx_current_states_owner 
ON current_states(owner);

CREATE INDEX IF NOT EXISTS idx_current_states_object_type 
ON current_states(object_type);

-- ============================================================
-- TIME-SERIES TRENDS (TimescaleDB Hypertable)
-- ============================================================

-- Orbital element history for maneuver detection and ML features
CREATE TABLE IF NOT EXISTS orbital_trends (
    time TIMESTAMPTZ NOT NULL,
    norad_cat_id INTEGER REFERENCES current_states(norad_cat_id) ON DELETE CASCADE,
    
    -- Orbital elements
    semi_major_axis DOUBLE PRECISION,         -- km
    inclination DOUBLE PRECISION,             -- degrees
    eccentricity DOUBLE PRECISION,
    raan DOUBLE PRECISION,                      -- degrees
    arg_of_perigee DOUBLE PRECISION,          -- degrees
    mean_anomaly DOUBLE PRECISION,            -- degrees
    mean_motion DOUBLE PRECISION,             -- rev/day
    
    -- Deltas for change detection
    mean_motion_delta_1h DOUBLE PRECISION,    -- Change vs 1 hour ago
    inclination_delta_1h DOUBLE PRECISION,
    eccentricity_delta_1h DOUBLE PRECISION,
    
    mean_motion_delta_24h DOUBLE PRECISION,   -- Change vs 24 hours ago
    inclination_delta_24h DOUBLE PRECISION,
    
    -- Z-scores for statistical anomaly detection
    zscore_mean_motion DOUBLE PRECISION,
    zscore_inclination DOUBLE PRECISION,
    zscore_eccentricity DOUBLE PRECISION,
    
    -- ML/Behavioral profiling
    behavioral_profile VARCHAR(32),           -- STATION_KEEPING, PHASING, RPO, TUMBLING, SLEEPER, UNKNOWN
    profile_confidence DOUBLE PRECISION,      -- 0.0 to 1.0
    
    -- Position for trend queries
    position GEOMETRY(PointZ, 4326)
);

-- Convert to TimescaleDB hypertable for efficient time-series storage
SELECT create_hypertable('orbital_trends', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for trend queries by satellite
CREATE INDEX IF NOT EXISTS idx_orbital_trends_norad_time 
ON orbital_trends(norad_cat_id, time DESC);

-- Index for behavioral profile queries
CREATE INDEX IF NOT EXISTS idx_orbital_trends_profile 
ON orbital_trends(behavioral_profile, time DESC) 
WHERE behavioral_profile IS NOT NULL;

-- ============================================================
-- ASSET METADATA (Static Reference Data)
-- ============================================================

CREATE TABLE IF NOT EXISTS asset_metadata (
    norad_cat_id INTEGER PRIMARY KEY REFERENCES current_states(norad_cat_id),
    
    -- Identification
    intl_designator VARCHAR(16) NOT NULL,     -- YYYY-XXXCCC format
    object_name VARCHAR(64),
    object_type VARCHAR(16),                  -- PAYLOAD, DEBRIS, ROCKET_BODY
    
    -- Launch/Origin information
    country_code VARCHAR(8),                  -- ISO 3166-1 alpha-3
    launch_date DATE,
    launch_site VARCHAR(32),
    owner VARCHAR(32),                        -- Operating organization
    
    -- Physical characteristics (for HPOP/SRP/drag)
    mass_kg DOUBLE PRECISION,
    drag_area_m2 DOUBLE PRECISION,          -- Cross-sectional area for drag
    srp_area_m2 DOUBLE PRECISION,             -- Area for solar radiation pressure
    reflectivity_coefficient DOUBLE PRECISION, -- CR (typically 1.0-1.5)
    
    -- IAF-specific tags
    is_indian_asset BOOLEAN DEFAULT FALSE,
    priority_level INTEGER DEFAULT 5,         -- 1=Critical, 2=High, 3=Medium, 4=Low, 5=Background
    mission_type VARCHAR(32),                 -- ISR, COMMS, NAV, WEATHER, etc.
    
    -- ITU filing information (for GEO coordination)
    itu_filing_ref VARCHAR(32),
    orbital_slot_deg DOUBLE PRECISION,        -- For GEO satellites
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_asset_metadata_country 
ON asset_metadata(country_code);

CREATE INDEX IF NOT EXISTS idx_asset_metadata_indian 
ON asset_metadata(is_indian_asset) 
WHERE is_indian_asset = TRUE;

CREATE INDEX IF NOT EXISTS idx_asset_metadata_priority 
ON asset_metadata(priority_level);

-- ============================================================
-- ALERTS & THREAT VECTORS
-- ============================================================

-- Conjunction Data Messages (CDM) and threat detections
CREATE TABLE IF NOT EXISTS threat_vector (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Threat classification
    threat_type VARCHAR(16) NOT NULL,         -- CDM, TIP, RPO_DETECTED, SLEEPER, UNCORRELATED
    severity VARCHAR(8) NOT NULL,             -- LOW, MEDIUM, HIGH, CRITICAL
    status VARCHAR(16) DEFAULT 'OPEN',        -- OPEN, ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE
    
    -- Primary asset (Indian or protected asset)
    primary_norad_id INTEGER REFERENCES current_states(norad_cat_id),
    primary_object_name VARCHAR(64),
    
    -- Secondary object (for CDM/RPO)
    secondary_norad_id INTEGER,
    secondary_object_name VARCHAR(64),
    secondary_object_type VARCHAR(16),
    
    -- Conjunction details (for CDM)
    tca TIMESTAMPTZ,                          -- Time of Closest Approach
    miss_distance_km DOUBLE PRECISION,
    relative_speed_kms DOUBLE PRECISION,      -- km/s
    collision_probability DOUBLE PRECISION,   -- 0.0 to 1.0
    
    -- Geometry (conjunction point)
    conjunction_point GEOMETRY(PointZ, 4326),
    
    -- Screening data
    screening_type VARCHAR(16),               -- BOX, PLANE, ALL
    
    -- Alert metadata
    alert_origin VARCHAR(32),                   -- SPACETRACK, INTERNAL, SENSOR_XYZ
    operator_acknowledged_by VARCHAR(32),
    operator_notes TEXT,
    
    -- Raw data storage
    raw_message JSONB,                        -- Original CDM/TIP JSON
    
    -- Resolution tracking
    resolved_at TIMESTAMPTZ,
    resolution_action VARCHAR(32)             -- MANUEVER_AVOIDED, MONITOR_ONLY, FALSE_POSITIVE, etc.
);

-- Indexes for threat monitoring
CREATE INDEX IF NOT EXISTS idx_threat_vector_status 
ON threat_vector(status, severity, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_threat_vector_primary 
ON threat_vector(primary_norad_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_threat_vector_tca 
ON threat_vector(tca) 
WHERE threat_type = 'CDM';

-- ============================================================
-- SENSOR TASKING (Requirement g)
-- ============================================================

CREATE TABLE IF NOT EXISTS sensor_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Task identification
    sensor_id VARCHAR(32) NOT NULL,           -- NETRA, OTCG, etc.
    task_type VARCHAR(16) NOT NULL,           -- UCT_CONFIRM, RPO_TRACK, REENTRY_MONITOR, ROUTINE
    
    -- Target information
    target_norad_id INTEGER,
    target_object_name VARCHAR(64),
    target_predicted_position GEOMETRY(PointZ, 4326),
    
    -- Observation window
    observation_start TIMESTAMPTZ NOT NULL,
    observation_end TIMESTAMPTZ NOT NULL,
    
    -- Computed optimal observation parameters
    optimal_azimuth DOUBLE PRECISION,           -- degrees
    optimal_elevation DOUBLE PRECISION,         -- degrees
    optimal_range_km DOUBLE PRECISION,
    
    -- Scheduling
    priority INTEGER DEFAULT 5,               -- 1-5 scale
    scheduled_at TIMESTAMPTZ,
    
    -- Status tracking
    status VARCHAR(16) DEFAULT 'PENDING',   -- PENDING, SCHEDULED, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
    
    -- Result data (when tracklet received)
    tracklet_received_at TIMESTAMPTZ,
    tracklet_data JSONB,                      -- {azimuth, elevation, range, timestamp, sigma}
    
    -- Fusion result (after EKF update)
    fused_state_id UUID,                      -- Links to orbital_trends entry
    
    -- Operator interface
    requested_by VARCHAR(32),                 -- System or operator name
    approved_by VARCHAR(32),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_sensor_tasks_sensor_status 
ON sensor_tasks(sensor_id, status, observation_start);

CREATE INDEX IF NOT EXISTS idx_sensor_tasks_target 
ON sensor_tasks(target_norad_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_tasks_time 
ON sensor_tasks(observation_start, observation_end);

-- ============================================================
-- INGESTION LOG & AUDIT
-- ============================================================

CREATE TABLE IF NOT EXISTS ingestion_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    
    source VARCHAR(32) NOT NULL,              -- SPACETRACK, SENSOR_XYZ, MANUAL
    source_query TEXT,                        -- URL or query parameters
    
    records_received INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    
    -- Metadata
    query_start_time TIMESTAMPTZ,
    query_end_time TIMESTAMPTZ,
    api_response_time_ms INTEGER,
    
    -- Error tracking
    errors JSONB,                             -- Array of error messages
    
    -- Audit
    processed_by VARCHAR(32)                  -- Hostname or worker ID
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_time 
ON ingestion_log(ingested_at DESC);

-- ============================================================
-- HELPER FUNCTIONS & TRIGGERS
-- ============================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER IF NOT EXISTS trigger_asset_metadata_updated_at
    BEFORE UPDATE ON asset_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER IF NOT EXISTS trigger_threat_vector_updated_at
    BEFORE UPDATE ON threat_vector
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER IF NOT EXISTS trigger_sensor_tasks_updated_at
    BEFORE UPDATE ON sensor_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to compute 3D bounding box from satellite position
CREATE OR REPLACE FUNCTION get_position_bbox(
    center_lon DOUBLE PRECISION,
    center_lat DOUBLE PRECISION,
    center_alt_km DOUBLE PRECISION,
    radius_km DOUBLE PRECISION
) RETURNS GEOMETRY AS $$
DECLARE
    delta_deg DOUBLE PRECISION;
BEGIN
    -- Approximate degree conversion (1 deg lat ~ 111 km)
    delta_deg := radius_km / 111.0;
    
    RETURN ST_SetSRID(ST_MakeEnvelope(
        center_lon - delta_deg,
        center_lat - delta_deg,
        center_lon + delta_deg,
        center_lat + delta_deg,
        (center_alt_km - radius_km) * 1000,  -- Convert to meters
        (center_alt_km + radius_km) * 1000
    ), 4326);
END;
$$ LANGUAGE plpgsql;

-- Function for volumetric scan (3D bounding box query)
CREATE OR REPLACE FUNCTION volumetric_scan_bbox(
    min_lon DOUBLE PRECISION,
    min_lat DOUBLE PRECISION,
    min_alt_km DOUBLE PRECISION,
    max_lon DOUBLE PRECISION,
    max_lat DOUBLE PRECISION,
    max_alt_km DOUBLE PRECISION
) RETURNS TABLE (
    norad_cat_id INTEGER,
    object_name VARCHAR(64),
    longitude DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    altitude_km DOUBLE PRECISION,
    velocity_kms DOUBLE PRECISION[3],
    last_updated TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cs.norad_cat_id,
        cs.object_name,
        ST_X(cs.position) as longitude,
        ST_Y(cs.position) as latitude,
        ST_Z(cs.position) / 1000.0 as altitude_km,  -- Convert meters to km
        cs.velocity_vector,
        cs.last_updated
    FROM current_states cs
    WHERE cs.position &&& ST_SetSRID(ST_MakeEnvelope(
        min_lon, min_lat, min_alt_km * 1000,
        max_lon, max_lat, max_alt_km * 1000
    ), 4326)
    AND cs.last_updated > NOW() - INTERVAL '5 minutes'
    ORDER BY cs.last_updated DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- SENSOR DATA & UCT DETECTION (Module i - Rapid Processing)
-- ============================================================

-- Sensor frames from IAF optical/radar sensors
-- Stores raw image data for 3D matched filtering (MPDAF integration)
CREATE TABLE IF NOT EXISTS sensor_frames (
    id BIGSERIAL PRIMARY KEY,
    sensor_id VARCHAR(32) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    image_data BYTEA,  -- Raw image bytes (numpy/FITS format)
    exposure_time_ms INTEGER,
    width_pixels INTEGER,
    height_pixels INTEGER,
    wcs_header TEXT,  -- World Coordinate System (FITS header style)
    filter_band VARCHAR(16),  -- V, R, I, etc.
    sensor_type VARCHAR(16) DEFAULT 'OPTICAL',  -- OPTICAL, IR, RADAR
    site_location GEOGRAPHY(POINT, 4326),  -- Sensor location
    altitude_m NUMERIC,
    azimuth_deg NUMERIC,
    elevation_deg NUMERIC,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(sensor_id, timestamp)
);

-- Index for efficient time-series queries
CREATE INDEX IF NOT EXISTS idx_sensor_frames_time 
ON sensor_frames(sensor_id, timestamp DESC);

-- Select hypertable for TimescaleDB (if available)
-- SELECT create_hypertable('sensor_frames', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Uncorrelated Track (UCT) detections from 3D matched filtering
-- These are "unknown" objects detected by sensors but not in Space-Track catalog
CREATE TABLE IF NOT EXISTS uct_detections (
    id BIGSERIAL PRIMARY KEY,
    object_name VARCHAR(64) NOT NULL,  -- Temporary designation: UCT_YYYYMMDD_HHMMSS
    norad_cat_id INTEGER REFERENCES asset_metadata(norad_cat_id),  -- NULL until correlated
    detection_time TIMESTAMPTZ NOT NULL,
    ra_deg NUMERIC(10, 6) NOT NULL,  -- Right Ascension (J2000)
    dec_deg NUMERIC(10, 6) NOT NULL,  -- Declination (J2000)
    velocity_ra NUMERIC(10, 6),  -- Proper motion RA (deg/day)
    velocity_dec NUMERIC(10, 6),  -- Proper motion Dec (deg/day)
    peak_sn NUMERIC(6, 2),  -- Peak signal-to-noise ratio
    confidence NUMERIC(4, 3),  -- Detection confidence (0.0 - 1.0)
    detection_source VARCHAR(32) DEFAULT 'MPDAF_3D_STREAK',  -- Algorithm used
    sensor_type VARCHAR(16) DEFAULT 'OPTICAL',
    sensor_id VARCHAR(32) NOT NULL,
    streak_length_px NUMERIC(8, 2),  -- Streak length in pixels
    streak_angle_deg NUMERIC(6, 2),  -- Streak angle (0-180 degrees)
    integrated_flux NUMERIC(12, 2),  -- Total flux (ADU or photons)
    correlation_status VARCHAR(16) DEFAULT 'UNCORRELATED',  -- UNCORRELATED, CORRELATED, FALSE_POSITIVE
    correlation_confidence VARCHAR(8),  -- HIGH, MEDIUM, LOW
    correlated_at TIMESTAMPTZ,
    matched_norad_id INTEGER,  -- If correlated, which catalog object
    position GEOGRAPHY(POINT, 4326),  -- Geographic position for spatial queries
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(object_name, detection_time)
);

-- Indexes for UCT queries
CREATE INDEX IF NOT EXISTS idx_uct_detections_time ON uct_detections(detection_time DESC);
CREATE INDEX IF NOT EXISTS idx_uct_detections_status ON uct_detections(correlation_status, created_at);
CREATE INDEX IF NOT EXISTS idx_uct_detections_position ON uct_detections USING GIST(position);

-- Access windows for blind spot calculation (Module e)
-- Pre-calculated visibility windows for ground sites
CREATE TABLE IF NOT EXISTS access_windows (
    id BIGSERIAL PRIMARY KEY,
    site_name VARCHAR(32) NOT NULL,  -- Siliguri_Corridor, Abdul_Kalam_Test_Range, etc.
    satellite_id INTEGER REFERENCES asset_metadata(norad_cat_id),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    max_elevation NUMERIC(6, 2),  -- Maximum elevation during pass
    sensor_type VARCHAR(16) DEFAULT 'ALL',  -- OPTICAL, SAR, ALL
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(site_name, satellite_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_access_windows_site ON access_windows(site_name, window_start DESC);

-- ============================================================
-- INITIAL DATA (Optional seed data)
-- ============================================================

-- Common object type lookup
CREATE TABLE IF NOT EXISTS object_types (
    code VARCHAR(16) PRIMARY KEY,
    description VARCHAR(64),
    category VARCHAR(16)  -- FUNCTIONAL, NON_FUNCTIONAL, DEBRIS
);

INSERT INTO object_types (code, description, category) VALUES
    ('PAYLOAD', 'Operational Satellite', 'FUNCTIONAL'),
    ('PAYLOAD_DEBRIS', 'Debris from Payload', 'NON_FUNCTIONAL'),
    ('ROCKET_BODY', 'Rocket Stage/Body', 'NON_FUNCTIONAL'),
    ('ROCKET_DEBRIS', 'Debris from Rocket', 'NON_FUNCTIONAL'),
    ('UNKNOWN', 'Unknown Object Type', 'NON_FUNCTIONAL'),
    ('OTHER', 'Other Space Object', 'NON_FUNCTIONAL')
ON CONFLICT (code) DO NOTHING;

-- Priority levels reference
CREATE TABLE IF NOT EXISTS priority_levels (
    level INTEGER PRIMARY KEY,
    name VARCHAR(16),
    description VARCHAR(64)
);

INSERT INTO priority_levels (level, name, description) VALUES
    (1, 'CRITICAL', 'Indian strategic assets requiring immediate protection'),
    (2, 'HIGH', 'Active Indian operational satellites'),
    (3, 'MEDIUM', 'Indian non-operational assets, allied assets'),
    (4, 'LOW', 'Tracked objects of interest'),
    (5, 'BACKGROUND', 'Catalogued debris and routine objects')
ON CONFLICT (level) DO NOTHING;

-- ============================================================
-- COMMENTS FOR DOCUMENTATION (IAF SDLC)
-- ============================================================

COMMENT ON TABLE current_states IS 'Real-time satellite states propagated from TLEs using SGP4. Position stored as PostGIS PointZ (Lon/Lat/Alt-meters). Updated every 60 seconds by propagator worker.';
COMMENT ON TABLE orbital_trends IS 'Time-series orbital element history for maneuver detection and ML behavioral profiling. TimescaleDB hypertable with 1-day chunks.';
COMMENT ON TABLE asset_metadata IS 'Static satellite metadata including physical characteristics for HPOP and IAF priority classifications.';
COMMENT ON TABLE threat_vector IS 'Conjunction and threat alerts from CDM, TIP, and internal detection systems. Links to current_states for primary asset.';
COMMENT ON TABLE sensor_tasks IS 'Automated sensor tasking requests for UCT confirmation, RPO tracking, and routine monitoring. Supports EKF tracklet fusion.';
COMMENT ON COLUMN current_states.covariance IS '6x6 position/velocity covariance matrix stored as upper triangle (21 elements) in km and km/s units';
COMMENT ON COLUMN current_states.position IS 'PostGIS PointZ with altitude in meters (multiply km by 1000). Uses EPSG:4326 (WGS84).';
COMMENT ON TABLE sensor_frames IS 'Raw image data from IAF optical/radar sensors for 3D matched filtering. Stores FITS/numpy arrays for MPDAF processing.';
COMMENT ON TABLE uct_detections IS 'Uncorrelated Track (UCT) detections from 3D matched filtering. Unknown objects detected by sensors but not in Space-Track catalog.';
COMMENT ON TABLE access_windows IS 'Pre-calculated visibility windows for strategic ground sites. Used for blind spot detection (Module e).';
COMMENT ON COLUMN uct_detections.detection_source IS 'Algorithm used: MPDAF_3D_STREAK, BACKGROUND_SUBTRACTION, etc.';
COMMENT ON COLUMN uct_detections.correlation_status IS 'UNCORRELATED: unknown object, CORRELATED: matched to Space-Track, FALSE_POSITIVE: noise/artifact';
