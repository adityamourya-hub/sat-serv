--
--  Space Situational Awareness Database Schema
-- TimescaleDB Implementation for High-Volume TLE History
--
-- Features:
-- - Hypertable partitioning by epoch for time-series optimization
-- - Deduplication via TLE hash
-- - Continuous aggregates for pre-computed B* smoothing
-- - Compression for warm-tier data
--

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

--
-- 1. TLE_HISTORY: Primary storage for all TLE epochs
--    Partitioned by epoch for time-series queries
--
CREATE TABLE tle_history (
    norad_id INTEGER NOT NULL,
    epoch TIMESTAMPTZ NOT NULL,
    
    -- Parsed orbital elements (for indexed queries)
    inclination DOUBLE PRECISION,
    raan DOUBLE PRECISION,
    eccentricity DOUBLE PRECISION,
    arg_perigee DOUBLE PRECISION,
    mean_anomaly DOUBLE PRECISION,
    mean_motion DOUBLE PRECISION,  -- revs/day
    
    -- Drag parameters (critical for B* smoothing)
    b_star DOUBLE PRECISION,
    
    -- Storage and metadata
    raw_line1 TEXT NOT NULL,
    raw_line2 TEXT NOT NULL,
    tle_hash UUID NOT NULL,        -- MD5 hash for deduplication
    
    -- Ingestion tracking
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'spacetrack',  -- spacetrack, isro, etc.
    
    -- Primary key: norad_id + epoch for time-series uniqueness
    PRIMARY KEY (norad_id, epoch)
);

-- Convert to hypertable (partitioned by epoch)
-- Chunk size: 7 days (optimal for 1.5 updates/day × 50k objects)
SELECT create_hypertable('tle_history', 'epoch', chunk_time_interval => INTERVAL '7 days');

--
-- 2. Indices for common query patterns
--

-- Index for B* smoothing queries (last 30 days for specific satellite)
CREATE INDEX idx_tle_history_norad_epoch ON tle_history (norad_id, epoch DESC);

-- Index for recent updates (incremental ingestion)
CREATE INDEX idx_tle_history_epoch_recent ON tle_history (epoch DESC) 
    WHERE epoch > NOW() - INTERVAL '7 days';

-- Index for drag analysis (B* values)
CREATE INDEX idx_tle_history_bstar ON tle_history (b_star, epoch) 
    WHERE b_star IS NOT NULL AND b_star != 0;

-- Unique constraint for deduplication (hash + norad_id + epoch)
-- Note: TLEs with same content but different epochs are valid updates
CREATE UNIQUE INDEX idx_tle_history_dedup ON tle_history (norad_id, tle_hash, epoch);

--
-- 3. ACTIVE_CATALOG: Latest TLE per object (materialized view)
--
CREATE TABLE active_catalog (
    norad_id INTEGER PRIMARY KEY,
    epoch TIMESTAMPTZ NOT NULL,
    
    -- Parsed elements
    inclination DOUBLE PRECISION,
    raan DOUBLE PRECISION,
    eccentricity DOUBLE PRECISION,
    arg_perigee DOUBLE PRECISION,
    mean_anomaly DOUBLE PRECISION,
    mean_motion DOUBLE PRECISION,
    semi_major_axis DOUBLE PRECISION,  -- computed
    
    -- Drag
    b_star DOUBLE PRECISION,
    smoothed_b_star DOUBLE PRECISION,  -- from continuous aggregate
    ballistic_coeff DOUBLE PRECISION,  -- BC = 0.02 / B* (kg/m^2)
    
    -- Raw storage
    raw_line1 TEXT NOT NULL,
    raw_line2 TEXT NOT NULL,
    
    -- Object metadata
    object_name VARCHAR(100),
    country_code VARCHAR(3),
    object_type VARCHAR(20),  -- PAYLOAD, DEBRIS, ROCKET_BODY, etc.
    
    -- Status tracking
    is_maneuvering BOOLEAN DEFAULT FALSE,
    last_maneuver_at TIMESTAMPTZ,
    decayed_at TIMESTAMPTZ,
    
    -- Update tracking
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    tle_count INTEGER DEFAULT 1  -- number of historical TLEs
);

-- Index for active catalog queries
CREATE INDEX idx_active_catalog_country ON active_catalog (country_code) 
    WHERE decayed_at IS NULL;

CREATE INDEX idx_active_catalog_type ON active_catalog (object_type) 
    WHERE decayed_at IS NULL;

--
-- 4. SPACE_WEATHER: NOAA solar/geomagnetic indices
--
CREATE TABLE space_weather (
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Solar indices
    f107 DOUBLE PRECISION,         -- 10.7 cm flux (sfu)
    f107_avg_81d DOUBLE PRECISION, -- 81-day centered average
    
    -- Geomagnetic indices
    ap DOUBLE PRECISION,           -- Planetary amplitude
    kp DOUBLE PRECISION,           -- Planetary K-index
    
    -- Source metadata
    source VARCHAR(20) DEFAULT 'noaa_swpc',
    
    PRIMARY KEY (timestamp)
);

-- Convert to hypertable (smaller chunks for frequent updates)
SELECT create_hypertable('space_weather', 'timestamp', chunk_time_interval => INTERVAL '1 day');

--
-- 5. CONTINUOUS AGGREGATES: Pre-computed B* smoothing
--
-- This automatically maintains rolling 30-day B* statistics per satellite
-- Refreshes every hour, queried instantly by the physics engine
--
CREATE MATERIALIZED VIEW b_star_smoothed_30d
WITH (timescaledb.continuous) AS
SELECT
    norad_id,
    time_bucket(INTERVAL '1 hour', epoch) AS bucket,
    
    -- B* statistics for smoothing
    AVG(b_star) AS b_star_mean,
    STDDEV(b_star) AS b_star_std,
    COUNT(*) AS sample_count,
    MIN(b_star) AS b_star_min,
    MAX(b_star) AS b_star_max,
    
    -- Derived ballistic coefficient
    CASE 
        WHEN AVG(b_star) > 0 THEN 0.02 / AVG(b_star)
        ELSE 50.0  -- nominal
    END AS ballistic_coeff,
    
    -- Trend slope (simple linear regression)
    -- Used for maneuver detection
    REGR_SLOPE(b_star, EXTRACT(EPOCH FROM epoch)) * 86400 AS b_star_slope_per_day
    
FROM tle_history
WHERE epoch > NOW() - INTERVAL '30 days'
    AND b_star IS NOT NULL
    AND b_star != 0
GROUP BY norad_id, time_bucket(INTERVAL '1 hour', epoch)
WITH NO DATA;

-- Refresh policy: Every hour, refresh last 2 hours of data
SELECT add_continuous_aggregate_policy('b_star_smoothed_30d',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

--
-- 6. MANEUVER_LOG: Detected orbital maneuvers
--
CREATE TABLE maneuver_log (
    id SERIAL PRIMARY KEY,
    norad_id INTEGER NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- TLE references
    pre_maneuver_tle_epoch TIMESTAMPTZ,
    post_maneuver_tle_epoch TIMESTAMPTZ,
    
    -- Change metrics
    delta_v DOUBLE PRECISION,        -- km/s (estimated)
    delta_a DOUBLE PRECISION,        -- km (semi-major axis change)
    confidence DOUBLE PRECISION,     -- 0-1 based on B* variance
    
    -- Classification
    maneuver_type VARCHAR(20),       -- DRAG_ANOMALY, DELTA_V, STATION_KEEPING
    reviewed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    
    -- Indices
    CONSTRAINT fk_norad FOREIGN KEY (norad_id) REFERENCES active_catalog(norad_id)
);

CREATE INDEX idx_maneuver_log_norad ON maneuver_log (norad_id, detected_at DESC);
CREATE INDEX idx_maneuver_log_recent ON maneuver_log (detected_at DESC) 
    WHERE reviewed = FALSE;

--
-- 7. COMPRESSION POLICY: Warm tier (1-5 years)
--
-- Enable compression for chunks older than 1 year
-- Achieves ~90% storage reduction
--
ALTER TABLE tle_history SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'norad_id'  -- Keep norad_id uncompressed for queries
);

-- Compress chunks after 1 year
SELECT add_compression_policy('tle_history', INTERVAL '1 year');

--
-- 8. RETENTION POLICY: Cold tier migration (5+ years)
--
-- Automatically drop chunks older than 7 years
-- (Export to parquet/S3 before dropping for cold storage)
--
SELECT add_retention_policy('tle_history', INTERVAL '7 years');

--
-- 9. FUNCTIONS: Deduplication and ingestion helpers
--

-- Function: Deduplicated TLE insertion
CREATE OR REPLACE FUNCTION insert_tle_deduplicated(
    p_norad_id INTEGER,
    p_epoch TIMESTAMPTZ,
    p_inclination DOUBLE PRECISION,
    p_raan DOUBLE PRECISION,
    p_eccentricity DOUBLE PRECISION,
    p_arg_perigee DOUBLE PRECISION,
    p_mean_anomaly DOUBLE PRECISION,
    p_mean_motion DOUBLE PRECISION,
    p_b_star DOUBLE PRECISION,
    p_raw_line1 TEXT,
    p_raw_line2 TEXT,
    p_source VARCHAR(50) DEFAULT 'spacetrack'
) RETURNS BOOLEAN AS $$
DECLARE
    v_hash UUID;
    v_exists BOOLEAN;
BEGIN
    -- Generate hash from raw TLE lines
    v_hash := md5(p_raw_line1 || p_raw_line2)::UUID;
    
    -- Check for exact duplicate (same hash, norad, epoch)
    SELECT EXISTS(
        SELECT 1 FROM tle_history 
        WHERE norad_id = p_norad_id 
        AND tle_hash = v_hash 
        AND epoch = p_epoch
    ) INTO v_exists;
    
    IF v_exists THEN
        RETURN FALSE;  -- Duplicate, skip
    END IF;
    
    -- Insert new TLE
    INSERT INTO tle_history (
        norad_id, epoch, inclination, raan, eccentricity,
        arg_perigee, mean_anomaly, mean_motion, b_star,
        raw_line1, raw_line2, tle_hash, source
    ) VALUES (
        p_norad_id, p_epoch, p_inclination, p_raan, p_eccentricity,
        p_arg_perigee, p_mean_anomaly, p_mean_motion, p_b_star,
        p_raw_line1, p_raw_line2, v_hash, p_source
    );
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function: Update active catalog from latest TLE
CREATE OR REPLACE FUNCTION update_active_catalog(p_norad_id INTEGER)
RETURNS VOID AS $$
DECLARE
    v_latest RECORD;
    v_smoothed RECORD;
    v_bc DOUBLE PRECISION;
BEGIN
    -- Get latest TLE
    SELECT * INTO v_latest
    FROM tle_history
    WHERE norad_id = p_norad_id
    ORDER BY epoch DESC
    LIMIT 1;
    
    IF NOT FOUND THEN
        RETURN;
    END IF;
    
    -- Get smoothed B* from continuous aggregate
    SELECT b_star_mean, ballistic_coeff
    INTO v_smoothed
    FROM b_star_smoothed_30d
    WHERE norad_id = p_norad_id
    ORDER BY bucket DESC
    LIMIT 1;
    
    -- Calculate ballistic coefficient
    IF v_smoothed.ballistic_coeff IS NOT NULL THEN
        v_bc := v_smoothed.ballistic_coeff;
    ELSIF v_latest.b_star IS NOT NULL AND v_latest.b_star != 0 THEN
        v_bc := 0.02 / v_latest.b_star;
    ELSE
        v_bc := 50.0;  -- nominal
    END IF;
    
    -- Upsert active catalog
    INSERT INTO active_catalog (
        norad_id, epoch, inclination, raan, eccentricity,
        arg_perigee, mean_anomaly, mean_motion,
        b_star, smoothed_b_star, ballistic_coeff,
        raw_line1, raw_line2, updated_at, tle_count
    ) VALUES (
        v_latest.norad_id, v_latest.epoch, 
        v_latest.inclination, v_latest.raan, v_latest.eccentricity,
        v_latest.arg_perigee, v_latest.mean_anomaly, v_latest.mean_motion,
        v_latest.b_star, v_smoothed.b_star_mean, v_bc,
        v_latest.raw_line1, v_latest.raw_line2, NOW(), 1
    )
    ON CONFLICT (norad_id) DO UPDATE SET
        epoch = EXCLUDED.epoch,
        inclination = EXCLUDED.inclination,
        raan = EXCLUDED.raan,
        eccentricity = EXCLUDED.eccentricity,
        arg_perigee = EXCLUDED.arg_perigee,
        mean_anomaly = EXCLUDED.mean_anomaly,
        mean_motion = EXCLUDED.mean_motion,
        b_star = EXCLUDED.b_star,
        smoothed_b_star = EXCLUDED.smoothed_b_star,
        ballistic_coeff = EXCLUDED.ballistic_coeff,
        raw_line1 = EXCLUDED.raw_line1,
        raw_line2 = EXCLUDED.raw_line2,
        updated_at = EXCLUDED.updated_at,
        tle_count = active_catalog.tle_count + 1;
    
END;
$$ LANGUAGE plpgsql;

--
-- 10. VIEWS: Common query patterns
--

-- View: Active Indian assets with drag parameters
CREATE VIEW active_indian_assets AS
SELECT 
    norad_id,
    object_name,
    epoch,
    semi_major_axis,
    b_star,
    smoothed_b_star,
    ballistic_coeff,
    is_maneuvering,
    CASE 
        WHEN ballistic_coeff < 20 THEN 'HIGH_DRAG'
        WHEN ballistic_coeff > 100 THEN 'LOW_DRAG'
        ELSE 'MODERATE'
    END AS drag_profile
FROM active_catalog
WHERE country_code = 'IND'
    AND decayed_at IS NULL;

-- View: Recent maneuvers (last 7 days)
CREATE VIEW recent_maneuvers AS
SELECT 
    m.*,
    a.object_name,
    a.country_code
FROM maneuver_log m
JOIN active_catalog a ON m.norad_id = a.norad_id
WHERE m.detected_at > NOW() - INTERVAL '7 days'
ORDER BY m.detected_at DESC;

--
-- INITIAL DATA LOADING
--

-- Insert some sample space weather data (for testing)
INSERT INTO space_weather (timestamp, f107, f107_avg_81d, ap, kp)
VALUES 
    (NOW(), 140.0, 135.0, 15.0, 2.0),
    (NOW() - INTERVAL '1 hour', 138.0, 135.0, 12.0, 1.67);

--
-- COMMENTS
--

COMMENT ON TABLE tle_history IS 'Complete TLE history partitioned by epoch. ~5.5 GB/year growth for 50k objects.';
COMMENT ON TABLE active_catalog IS 'Latest state per object. Materialized from tle_history.';
COMMENT ON TABLE space_weather IS 'NOAA space weather indices for dynamic drag modeling.';
COMMENT ON TABLE maneuver_log IS 'Detected orbital maneuvers with delta-V estimates.';
COMMENT ON MATERIALIZED VIEW b_star_smoothed_30d IS 'Pre-computed B* smoothing: 30-day rolling statistics per satellite. Refreshes hourly.';
