# ADITI 4.0 TimescaleDB Deployment Guide

## Overview

This directory contains the production-grade database schema for the IDEX-IAF SSIS high-volume TLE storage system.

**Capacity Planning** (50k objects, 1.5 updates/day):
- Raw data growth: ~5.5 GB/year
- With indices: ~11 GB/year
- 10-year mission: ~110 GB hot storage
- Compression (warm tier): ~90% reduction = ~11 GB total

## Architecture

### Three-Tier Storage Strategy

1. **Hot Tier (0-12 months)**: Full TimescaleDB hypertable with active indexing
   - Real-time B* smoothing
   - Maneuver detection
   - 2-hour prediction queries

2. **Warm Tier (1-5 years)**: TimescaleDB compression enabled
   - Columnar storage (~90% compression)
   - Long-term trend analysis
   - Slightly higher query latency acceptable

3. **Cold Tier (5+ years)**: Parquet/S3 export
   - Forensic analysis
   - Historical debris studies
   - Export before retention policy drops

## Quick Start

### 1. Install TimescaleDB

```bash
# Docker (development)
docker run -d --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=ssis_password \
  -e POSTGRES_USER=ssis \
  -e POSTGRES_DB=ssis_catalog \
  timescale/timescaledb:latest-pg15

# Production: Use managed TimescaleDB or install on dedicated server
```

### 2. Initialize Schema

```bash
# Run the schema creation
psql -h localhost -U ssis -d ssis_catalog -f timescaledb_schema.sql
```

### 3. Add Go Dependencies

```bash
# Option A: Pure Go driver (stable, recommended)
go get github.com/lib/pq

# Option B: High-performance driver (more features)
go get github.com/jackc/pgx/v5

# Then uncomment the driver import in pkg/persistence/timescaledb.go
```

### 4. Configure Connection

```go
cfg := persistence.TimescaleDBConfig{
    Host:     "localhost",
    Port:     5432,
    Database: "ssis_catalog",
    User:     "ssis",
    Password: "ssis_password",
    SSLMode:  "disable", // use "require" in production
}

backend, err := persistence.NewTimescaleDBBackend(cfg)
```

## Schema Highlights

### Tables

| Table | Purpose | Partitioning |
|-------|---------|--------------|
| `tle_history` | All TLE epochs | Hypertable by `epoch` (7-day chunks) |
| `active_catalog` | Latest TLE per object | Regular table |
| `space_weather` | NOAA F10.7, Ap, Kp | Hypertable by `timestamp` (1-day chunks) |
| `maneuver_log` | Detected maneuvers | Regular table with indices |

### Continuous Aggregates

- **`b_star_smoothed_30d`**: Pre-computed 30-day rolling B* statistics
  - Refreshes every hour
  - Provides instant ballistic coefficient lookups
  - Reduces smoothing computation by ~100x

### Key Features

1. **Deduplication**: `insert_tle_deduplicated()` uses MD5 hash to avoid storing identical TLEs
2. **Automatic Updates**: `update_active_catalog()` maintains latest state per object
3. **Maneuver Detection**: Semi-major axis changes flagged in `maneuver_log`
4. **Compression**: Automatic after 1 year (90% storage reduction)
5. **Retention**: Automatic drop after 7 years (export to cold storage first)

## Performance Tuning

### Hypertable Chunk Size

Current: 7 days for `tle_history`
- 50k objects × 1.5 updates/day × 7 days = ~525k rows/chunk
- Optimal for ~100k-1M rows per chunk

Adjust if needed:
```sql
SELECT set_chunk_time_interval('tle_history', INTERVAL '3 days');
```

### Indexing Strategy

Already optimized in schema:
- `idx_tle_history_norad_epoch`: B* smoothing queries (hot path)
- `idx_tle_history_recent`: Incremental ingestion (last 7 days)
- `idx_tle_history_dedup`: Duplicate prevention (unique constraint)

### Query Examples

```sql
-- Get smoothed B* for satellite (pre-computed, instant)
SELECT ballistic_coeff, b_star_mean, confidence
FROM b_star_smoothed_30d
WHERE norad_id = 25544
ORDER BY bucket DESC
LIMIT 1;

-- Get TLE history for smoothing (uses time-series index)
SELECT * FROM tle_history
WHERE norad_id = 25544
  AND epoch > NOW() - INTERVAL '30 days'
ORDER BY epoch;

-- Active Indian assets with drag profiles
SELECT * FROM active_indian_assets;

-- Recent unreviewed maneuvers
SELECT * FROM recent_maneuvers WHERE reviewed = false;
```

## Backup Strategy

### Hot Tier (Critical)
```bash
# Continuous archiving with pg_basebackup
pg_basebackup -h localhost -U ssis -D /backup/ssis_hot -Ft -z -P
```

### Warm Tier (Compressed)
```bash
# Logical backup of compressed chunks
pg_dump -h localhost -U ssis -t tle_history --exclude-table-data 'tle_history_*' > schema.sql
```

### Cold Tier (Archive)
```bash
# Export to Parquet for S3
psql -c "COPY (SELECT * FROM tle_history WHERE epoch < NOW() - INTERVAL '5 years') TO '/s3/cold/';"
```

## Monitoring

### Key Metrics

```sql
-- Table sizes
SELECT hypertable_name, pg_size_pretty(total_bytes) 
FROM timescaledb_information.hypertable;

-- Compression ratio
SELECT hypertable_name, 
       before_total_bytes / after_total_bytes as compression_ratio
FROM timescaledb_information.compression_settings;

-- Query performance (enable pg_stat_statements)
SELECT query, calls, mean_exec_time 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
```

## Security

### IAF Deployment

1. **Network**: Database in isolated subnet
2. **Access**: mTLS with client certificates
3. **Encryption**: AES-256 at rest
4. **Audit**: All insertions logged with Ed25519 signatures
5. **Air-gap**: Optional proxy ingestor in DMZ

```
[Space-Track] → [DMZ Proxy] → [Data Diode] → [Internal DB]
                     ↓
              [NOAA SWPC]
```

## Troubleshooting

### High Write Latency
- Check chunk count: `SELECT count(*) FROM timescaledb_information.chunks;`
- If >1000 chunks, adjust retention policy
- Enable async commits: `synchronous_commit = off` (risk of 1s data loss)

### Slow B* Queries
- Verify continuous aggregate is refreshing: `SELECT * FROM timescaledb_information.continuous_aggregates;`
- Check if `b_star_smoothed_30d` is populated
- Force refresh: `CALL refresh_continuous_aggregate('b_star_smoothed_30d', now() - interval '30 days', now());`

### Deduplication Issues
- Check hash collisions: `SELECT tle_hash, count(*) FROM tle_history GROUP BY tle_hash HAVING count(*) > 1;`
- Should be rare with MD5 + epoch

## References

- TimescaleDB Docs: https://docs.timescale.com/
- Hypertable Best Practices: https://docs.timescale.com/use-timescale/latest/hypertables/about-hypertables/
- Continuous Aggregates: https://docs.timescale.com/use-timescale/latest/continuous-aggregates/
- Compression: https://docs.timescale.com/use-timescale/latest/compression/
