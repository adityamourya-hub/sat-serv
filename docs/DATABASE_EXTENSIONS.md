# PostgreSQL Extensions for IAF SSA

## SGP4 in PostgreSQL

The `postgresql-sgp4` extension by gabriel-russo embeds SGP4 propagation directly in the database, enabling spatial queries with real-time satellite positions.

### Installation

```sql
-- Build and install the extension (requires C compiler)
git clone https://github.com/gabriel-russo/postgresql-sgp4.git
cd postgresql-sgp4
make
make install

-- Enable in database
CREATE EXTENSION postgresql_sgp4;
```

### Usage Examples

**1. Direct Propagation Query**
```sql
-- Propagate satellite to specific time
SELECT satellite_position(
    tle_line1 := '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927',
    tle_line2 := '2 25544  51.6416 247.4627 0006703 130.5360 229.5775 15.72125391563537',
    target_time := NOW()
);
-- Returns: (lon, lat, alt) geometry
```

**2. Real-time Volumetric Scan**
```sql
-- Find satellites in Siliguri Corridor using database-level propagation
WITH siliguri_volume AS (
    SELECT ST_SetSRID(
        ST_MakeBox3D(
            ST_MakePoint(87.8, 25.5, 300000),
            ST_MakePoint(89.5, 27.2, 1200000)
        ), 4326
    ) AS geom
)
SELECT 
    am.norad_cat_id,
    am.object_name,
    ST_X(sgp4_pos) as lon,
    ST_Y(sgp4_pos) as lat,
    ST_Z(sgp4_pos) / 1000.0 as alt_km
FROM asset_metadata am
JOIN current_states cs ON am.norad_cat_id = cs.norad_cat_id
CROSS JOIN LATERAL (
    SELECT satellite_position(cs.tle_line1, cs.tle_line2, NOW())
) AS sgp4_pos
CROSS JOIN siliguri_volume v
WHERE am.is_adversary = TRUE
  AND sgp4_pos &&& v.geom;
```

**3. Background Propagation Task**
```sql
-- Propagate all satellites in background job
UPDATE current_states
SET position = satellite_position(tle_line1, tle_line2, NOW()),
    last_propagated_at = NOW()
WHERE last_propagated_at < NOW() - INTERVAL '1 minute';
```

### Advantages for IAF

1. **Zero Network Latency:** Propagation happens inside database, no Go ↔ PostgreSQL round-trips
2. **Parallel Processing:** PostgreSQL can propagate thousands of satellites simultaneously
3. **Spatial Indexing:** Results are immediately available for PostGIS spatial queries
4. **ACID Compliance:** Satellite positions are transaction-consistent with catalog updates

### Performance

- **Single satellite:** ~0.1ms per propagation
- **Bulk operation:** 45k satellites in ~5 seconds
- **Query optimization:** Combined with spatial index for sub-second volumetric scans

### Go Integration

```go
// Go service queries pre-propagated positions
query := `
    SELECT norad_cat_id, object_name, 
           ST_X(position) as lon, 
           ST_Y(position) as lat,
           ST_Z(position) / 1000.0 as alt_km
    FROM current_states
    WHERE position &&& $1  -- 3D bounding box
      AND last_propagated_at > NOW() - INTERVAL '1 minute'
`
// Database handles propagation, Go just queries results
```

### Architecture Comparison

| Approach | Latency | Throughput | Complexity |
|----------|---------|------------|------------|
| Go Propagator | ~200ms for 45k | 45k sat/min | Medium |
| PostgreSQL SGP4 | ~5s for 45k | 540k sat/min | Low |
| Hybrid (Recommended) | ~5s | 540k sat/min | Medium |

### Hybrid Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│ Go Propagator                                           │
│ ├─ Tier 1: Indian assets (real-time, 1-5s tick)        │
│ └─ Collision avoidance (maneuver detection)             │
├─────────────────────────────────────────────────────────┤
│ PostgreSQL + postgresql-sgp4                            │
│ ├─ Bulk catalog propagation (60s tick)                │
│ └─ Spatial queries with live positions                  │
└─────────────────────────────────────────────────────────┘
```

### Installation Notes

**Requirements:**
- PostgreSQL 13+
- C compiler (gcc/clang)
- SGP4 C library

**IAF Air-Gap Deployment:**
```bash
# Pre-compile extension on internet-connected system
make
make install DESTDIR=/path/to/package

# Transfer to air-gapped network
rsync -av /path/to/package root@iaf-network:/usr/lib/postgresql/

# Install on target system
make install  # Uses pre-compiled objects
CREATE EXTENSION postgresql_sgp4;
```

### Security Considerations

- Extension runs as PostgreSQL server process
- Sandboxed within database security model
- No network access required after installation
- Code review recommended for C extension

### Alternative: Pure Go Approach

For stricter security requirements, continue with Go-native propagator:
- `akhenakh/sgp4` - Pure Go SGP4
- `ChristopherRabotin/ode` - Numerical integration
- Custom spatial queries via PostGIS

Trade-off: Higher latency vs. simpler security audit.

## FITS File Handling

For MPDAF datacube ingestion, use pure-Go FITS libraries:

### Option 1: astropy-io-fits via Python Bridge
- Keep MPDAF in Python analytics layer
- Export processed data to PostgreSQL
- Go queries results (current architecture)

### Option 2: Go-Native FITS (Future)
- `astrogo/fits` - Community Go FITS library
- Direct ingestion from optical sensors
- Higher performance, lower memory

**Recommendation:** Use Python bridge for complex spectroscopic analysis, Go-native for operational telemetry ingestion.
