-- ============================================================
-- SSIS SOVEREIGN ALERTING SYSTEM — DATABASE SCHEMA
-- Classification: SOVEREIGN_INTEL / AIR_GAPPED_ONLY
-- Dependencies: 01_setup.sql (PostGIS, TimescaleDB already enabled)
-- ============================================================

-- Severity Enum
DO $$ BEGIN
    CREATE TYPE alert_severity AS ENUM ('INFO', 'WARNING', 'CRITICAL', 'EMERGENCY');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Alert Status Lifecycle
DO $$ BEGIN
    CREATE TYPE alert_status AS ENUM ('OPEN', 'ACKNOWLEDGED', 'RESOLVED', 'EXPIRED', 'ESCALATED');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Alert Category
DO $$ BEGIN
    CREATE TYPE alert_category AS ENUM (
        'CONJUNCTION_RISK',
        'MANEUVER_DETECTION',
        'REENTRY_RISK',
        'BLIND_SPOT',
        'SENSOR_HEALTH',
        'BEHAVIOURAL_ANOMALY',
        'SOVEREIGN_INTRUSION',
        'COLLISION_RISK'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Operator Role
DO $$ BEGIN
    CREATE TYPE operator_role AS ENUM ('OPERATOR', 'COMMANDER', 'SYSTEM');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- 1. ALERTS TABLE (TimescaleDB Hypertable)
-- Core alert storage — one row per generated alert
-- ============================================================
CREATE TABLE IF NOT EXISTS alerts (
    -- Identity
    alert_id        UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Classification
    category        alert_category NOT NULL,
    severity        alert_severity NOT NULL DEFAULT 'INFO',
    status          alert_status NOT NULL DEFAULT 'OPEN',

    -- Target objects
    primary_norad_id    INTEGER,
    primary_object_name TEXT,
    secondary_norad_id  INTEGER,          -- For conjunctions
    secondary_object_name TEXT,

    -- Alert content
    title           TEXT NOT NULL,         -- Short one-line summary
    message         TEXT NOT NULL,         -- Detailed description
    recommended_action TEXT,               -- What operator should do

    -- Quantitative data (JSON for flexibility)
    metrics         JSONB DEFAULT '{}',    -- e.g. {"pc": 1.2e-4, "miss_km": 0.8, "z_score": 4.5}

    -- ML Explainability
    shap_attribution JSONB,               -- SHAP feature scores
    kl_divergence   DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,

    -- Geospatial (for map display)
    position        GEOMETRY(PointZ, 4326),
    sovereign_volume TEXT,                 -- e.g. "Siliguri_Corridor_LEO"

    -- Rules engine
    triggered_rule  TEXT,                  -- Rule ID that generated this alert
    rule_version    TEXT,                  -- Version of rules config

    -- Acknowledgement
    acknowledged_by TEXT,                  -- Operator ID
    acknowledged_at TIMESTAMPTZ,
    ack_action      TEXT,                  -- Action taken by operator
    resolved_at     TIMESTAMPTZ,

    -- Cryptographic integrity
    signature       BYTEA,                -- Ed25519 signature of alert payload

    -- Auto-escalation
    escalation_deadline TIMESTAMPTZ,      -- When to auto-escalate if unacknowledged
    escalated_to    alert_severity,

    -- Composite primary key for TimescaleDB hypertable (must include partitioning column)
    PRIMARY KEY (alert_id, created_at)
);

-- Convert to hypertable for time-series performance
SELECT create_hypertable('alerts', 'created_at', if_not_exists => TRUE);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX IF NOT EXISTS idx_alerts_category ON alerts (category);
CREATE INDEX IF NOT EXISTS idx_alerts_primary ON alerts (primary_norad_id);
CREATE INDEX IF NOT EXISTS idx_alerts_spatial ON alerts USING GIST (position);
CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts (created_at DESC) WHERE status = 'OPEN';

-- ============================================================
-- 2. ALERT_RULES TABLE
-- Configurable rules loaded from YAML, persisted for audit trail
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id         TEXT PRIMARY KEY,
    category        alert_category NOT NULL,
    severity        alert_severity NOT NULL,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,

    -- Rule definition (YAML-sourced)
    description     TEXT NOT NULL,
    condition_json  JSONB NOT NULL,        -- Serialised rule condition
    
    -- Thresholds
    threshold_value DOUBLE PRECISION,
    threshold_unit  TEXT,                  -- "probability", "km", "sigma", "minutes"
    
    -- Escalation
    escalation_timeout_min INTEGER,        -- Minutes before auto-escalate
    escalation_severity alert_severity,

    -- Metadata
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version         TEXT NOT NULL DEFAULT '1.0.0'
);

-- ============================================================
-- 3. AUDIT_LOG TABLE (TimescaleDB Hypertable)
-- Immutable, signed audit trail for IAF certification
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    log_id          BIGSERIAL,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Who
    actor_id        TEXT NOT NULL,          -- Operator ID or 'SYSTEM'
    actor_role      operator_role NOT NULL DEFAULT 'SYSTEM',

    -- What
    action          TEXT NOT NULL,          -- e.g. 'ALERT_CREATED', 'ALERT_ACKNOWLEDGED', 'RULE_UPDATED'
    entity_type     TEXT NOT NULL,          -- 'alert', 'rule', 'sensor'
    entity_id       TEXT NOT NULL,          -- UUID or rule_id

    -- Details
    details         JSONB DEFAULT '{}',     -- Action-specific metadata
    previous_state  JSONB,                  -- State before change (for auditing)
    new_state       JSONB,                  -- State after change

    -- Integrity
    signature       BYTEA,                  -- Ed25519 signature of (logged_at + actor_id + action + entity_id + details)
    
    PRIMARY KEY (log_id, logged_at)
);

SELECT create_hypertable('audit_log', 'logged_at', if_not_exists => TRUE);

-- Index for compliance queries
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log (actor_id, logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log (entity_type, entity_id, logged_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action, logged_at DESC);

-- ============================================================
-- 4. SENSOR_STATUS TABLE
-- Tracks sensor health for degradation alerting
-- ============================================================
CREATE TABLE IF NOT EXISTS sensor_status (
    sensor_id       TEXT NOT NULL,
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE, DEGRADED, OFFLINE, MAINTENANCE
    health_score    DOUBLE PRECISION,                -- 0.0 to 1.0
    last_observation TIMESTAMPTZ,
    error_rate      DOUBLE PRECISION,
    details         JSONB DEFAULT '{}',
    PRIMARY KEY (sensor_id, reported_at)
);

SELECT create_hypertable('sensor_status', 'reported_at', if_not_exists => TRUE);

-- ============================================================
-- 5. CONTINUOUS AGGREGATES for dashboard performance
-- ============================================================

-- Alert counts by hour (for trend charts)
CREATE MATERIALIZED VIEW IF NOT EXISTS alert_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    category,
    severity,
    COUNT(*) AS alert_count
FROM alerts
GROUP BY bucket, category, severity
WITH NO DATA;

-- Refresh policy: every 5 minutes, cover last 24 hours
SELECT add_continuous_aggregate_policy('alert_hourly_stats',
    start_offset    => INTERVAL '24 hours',
    end_offset      => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists   => TRUE
);

-- ============================================================
-- 6. RETENTION POLICIES
-- ============================================================

-- Keep raw alerts for 1 year
SELECT add_retention_policy('alerts', INTERVAL '365 days', if_not_exists => TRUE);

-- Keep audit log for 5 years (compliance requirement)
SELECT add_retention_policy('audit_log', INTERVAL '1825 days', if_not_exists => TRUE);

-- Keep sensor status for 90 days
SELECT add_retention_policy('sensor_status', INTERVAL '90 days', if_not_exists => TRUE);
