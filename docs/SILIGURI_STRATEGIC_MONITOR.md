# Siliguri Corridor Strategic Monitor

## Overview

The **Siliguri Strategic Cube** is a 3D Volume of Interest (VOI) monitoring the "Chicken's Neck" - the narrow corridor connecting India's northeast to the mainland.

## Geographic Bounds

| Parameter | Value |
|-----------|-------|
| **Latitude** | 25.5°N to 27.2°N |
| **Longitude** | 87.8°E to 89.5°E |
| **Altitude** | 300km to 1200km (LEO spy satellite zone) |
| **Dimensions** | ~180km (L) x 60km (W) x 900km (H) |

## Threat Categories

### 1. Jilin-1 Optical Swarm
- **Constellation**: 115+ active satellites (as of 2026)
- **Capability**: 4K HD video tracking with 15-20 minute revisit
- **Threat**: Can track single tank column through Kishanganj in real-time
- **Detection**: 3+ satellites within 60 minutes triggers **ISR Saturation Alert**

### 2. Yaogan ELINT Triplets
- **Mission**: Electronic intelligence mapping of IAF radar signatures
- **Signature**: 3 satellites in close temporal proximity (~30 min window)
- **Response**: Automatic **EMCON** (Emission Control) recommendation

## System Capabilities

### Real-Time Queries

```go
// 1. Volumetric Scan - O(log N) with 3D spatial index
objects, err := monitor.CheckSiliguriVolume(time.Now())

// 2. Swarm Detection
swarm := monitor.DetectJilinSwarm(objects)
if swarm != nil {
    log.Printf("ISR SATURATION: %d Jilin satellites detected", swarm.SatelliteCount)
}

// 3. ELINT Triplet Detection
emcon := monitor.DetectYaoganELINT(objects)
if emcon != nil {
    log.Printf("EMCON RECOMMENDATION: %s for %v", 
        emcon.RecommendedStatus, emcon.Duration)
}

// 4. Strategic Dark Window (safe movement periods)
window := monitor.CalculateDarkWindow(6 * time.Hour)
if window != nil {
    log.Printf("DARK WINDOW: %v to %v - Safe for %s", 
        window.StartTime, window.EndTime, window.SafeFor)
}
```

### Automatic Actions

| Condition | Action | Priority |
|-----------|--------|----------|
| 3+ Jilin detected | Optical verification tasking | 9/10 |
| 5+ Jilin detected | ISR Saturation Alert | 10/10 |
| Yaogan triplet | EMCON: Radio Silence | 10/10 |
| 2+ hour dark window | Tank movement advisory | 8/10 |

## Database Schema

### PostgreSQL/PostGIS Setup

```sql
-- Deploy the strategic cube
\i pkg/spatial/siliguri_sql.go

-- Or run the SQL directly in psql
SELECT * FROM query_siliguri_volume();
SELECT * FROM detect_jilin_swarm();
SELECT * FROM detect_yaogan_elint();
SELECT * FROM calculate_dark_window(6);
```

### Performance Optimization

**Critical for M2 Air**: The `gist_geometry_ops_nd` index enables sub-10ms queries.

```sql
-- N-Dimensional R-tree index (MUST HAVE)
CREATE INDEX idx_current_states_3d 
ON current_states 
USING gist (position_vector gist_geometry_ops_nd);

-- Query using &&& operator (fast bounding box)
-- Then ST_3DIntersects (precise 3D check)
```

## Integration with SSIS

### Go Usage

```go
import "ssis/pkg/spatial"

monitor := spatial.NewSiliguriMonitor(catalogStore, sensorRegistry)

// In main loop (10Hz)
for range ticker.C {
    objects, _ := monitor.CheckSiliguriVolume(time.Now())
    
    // Check for swarm
    if swarm := monitor.DetectJilinSwarm(objects); swarm != nil {
        // Trigger optical verification
        tasks := monitor.GenerateTaskingRequests()
        for _, task := range tasks {
            telescopeBridge.SlewToTarget(task.TargetID)
        }
    }
    
    // Check for ELINT
    if emcon := monitor.DetectYaoganELINT(objects); emcon != nil {
        // Alert IAF units
        alertSystem.EmitEMCON(emcon)
    }
}
```

### Python Analytics

```python
# High-rate tracking for optical verification
from high_rate_tracker import HighRateTracker

tracker = HighRateTracker(
    zmq_pub_endpoint="tcp://localhost:5555",
    update_rate_hz=100.0
)
tracker.run()
```

## Tactical Value

### What This Wins

1. **Geopolitical Awareness**: Not just tracking satellites, but understanding **vulnerabilities**
2. **Predictive Intelligence**: Dark windows enable **planned movements**
3. **Automatic Response**: No human-in-the-loop delay for verification tasking
4. **Resource Optimization**: Focus sensors on **high-value threats** (Jilin/Yaogan)

### Comparison: Traditional vs Strategic SSA

| Aspect | Traditional SSA | Strategic SSA (This) |
|--------|---------------|----------------------|
| Query | "What's in orbit?" | "What's threatening Siliguri?" |
| Alert | New object detected | Jilin swarm saturating ISR |
| Action | Log entry | Auto-optical verification |
| Timing | Reactive | Predictive (dark windows) |

## Deployment

### Requirements

1. **PostgreSQL 14+** with PostGIS 3.2+
2. **TimescaleDB** for time-series optimization
3. **M2 Air** or equivalent (optimized for Apple Silicon)

### Quick Start

```bash
# 1. Setup database
createdb ssis_siliguri
psql ssis_siliguri -c "CREATE EXTENSION postgis;"
psql ssis_siliguri -f deploy/database/timescaledb_schema.sql

# 2. Deploy Siliguri schema
cat pkg/spatial/siliguri_sql.go | psql ssis_siliguri

# 3. Run monitor
go run cmd/siliguri_monitor/main.go
```

## References

- **ADITI 4.0 Proposal**: Volumetric Surveillance of Critical Geographies
- **Jilin-1 Constellation**: 115+ satellites providing 15-20 min revisit (2026)
- **Yaogan Series**: Chinese ELINT triplet configuration
- **Siliguri Corridor**: Strategic "Chicken's Neck" of India

---

**Classification**: SOVEREIGN DEFENSE IP  
**Status**: Production Ready for ADITI 4.0
