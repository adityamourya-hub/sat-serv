-- Enable the High-Performance Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- CORE TABLES FOR IAF SSA
-- ============================================================

-- Tracklet Hypertable for UCT detections
CREATE TABLE uct_detections (
    observation_time TIMESTAMPTZ NOT NULL,
    tracklet_id UUID NOT NULL,
    norad_id INTEGER,
    s2_cell_id BIGINT,
    position_vector GEOMETRY(PointZ, 4326),
    intensity FLOAT
);

-- Convert to Hypertable partitioned by time
SELECT create_hypertable('uct_detections', 'observation_time');

-- Optimized Indexing
CREATE INDEX idx_s2_cell ON uct_detections (s2_cell_id);
CREATE INDEX idx_spatial_3d ON uct_detections USING GIST (position_vector);

-- Current states (existing implementation)
CREATE TABLE current_states (
    norad_cat_id INTEGER PRIMARY KEY,
    position GEOMETRY(PointZ, 4326) NOT NULL,
    velocity_vector FLOAT[3],
    epoch TIMESTAMPTZ,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_current_states_spatial ON current_states USING GIST (position);

-- Orbital trends hypertable
CREATE TABLE orbital_trends (
    time TIMESTAMPTZ NOT NULL,
    norad_cat_id INTEGER,
    semi_major_axis FLOAT,
    inclination FLOAT,
    eccentricity FLOAT,
    raan FLOAT,
    arg_of_perigee FLOAT,
    mean_anomaly FLOAT,
    mean_motion FLOAT,
    bstar FLOAT
);

SELECT create_hypertable('orbital_trends', 'time', chunk_time_interval => INTERVAL '1 day');
