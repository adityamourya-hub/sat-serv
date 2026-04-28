# SSIS Alerting System - Local Demo Guide

**Purpose:** Test alerting system locally using single Go binary (36MB)  
**Classification:** SOVEREIGN_INTEL / AIR_GAPPED_ONLY

---

## Prerequisites

- PostgreSQL 16 with TimescaleDB extension (local or via Docker)
- Go 1.26+ (for building the binary)
- Python 3.11+ (for serving static files, optional)

---

## Quick Start (5 Minutes)

### 1. Start PostgreSQL with TimescaleDB

**Option A: Using Docker (Recommended for Demo)**
```bash
docker run -d \
  --name ssis_demo_db \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=ssis_db \
  timescale/timescaledb-ha:pg16
```

**Option B: Local PostgreSQL Installation**
```bash
# Install TimescaleDB extension
# (Follow TimescaleDB installation guide for your OS)
```

### 2. Run Database Migrations

```bash
# Wait for database to be ready
docker exec -it ssis_demo_db pg_isready -U postgres -d ssis_db

# Run migrations
docker exec -i ssis_demo_db psql -U postgres -d ssis_db < init-db/01_setup.sql
docker exec -i ssis_demo_db psql -U postgres -d ssis_db < init-db/02_alerts.sql
```

### 3. Build the Go Binary

```bash
cd /Users/manishmaurya/git-repos/Idex/Project-1-iaf/idex
CGO_ENABLED=1 go build -tags onnx -o ssis_engine ./engine/main.go
```

### 4. Set Environment Variables

```bash
export SSIS_DB_URL="postgres://postgres:changeme@localhost:5432/ssis_db?sslmode=disable"
export SSIS_CONFIG_PATH="configs"
export SSIS_MODEL_DIR="pkg/ml/models"
export SSIS_GRPC_PORT=50051
export SSIS_REST_PORT=8080
```

### 5. Run the Engine

```bash
./ssis_engine --provider=CPU
```

You should see:
```
INFO: Connected to PostgreSQL for alerting system
INFO: Alert evaluator started
INFO: Hardened gRPC server starting on :50051
INFO: REST server starting on :8080
```

### 6. Serve the UI

**Option A: Python Simple HTTP Server**
```bash
cd pkg/hud
python3 -m http.server 8000
```

**Option B: Using the Engine's Static File Serving (if configured)**
```bash
# If the engine serves static files, access directly at http://localhost:8080/
```

### 7. Open the UI

Open browser to: `http://localhost:8000/globe.html`

---

## Testing the Alerting System

### Method 1: Insert Test Alerts Directly into Database

```bash
# Connect to database
docker exec -it ssis_demo_db psql -U postgres -d ssis_db

# Insert a test alert
INSERT INTO alerts (
  alert_id,
  created_at,
  category,
  severity,
  status,
  primary_norad_id,
  secondary_norad_id,
  title,
  message,
  metrics,
  recommended_action,
  escalation_deadline
) VALUES (
  'demo-alert-001',
  NOW(),
  'CONJUNCTION',
  'CRITICAL',
  'OPEN',
  '25544',
  '12345',
  'CONJUNCTION — ISS / DEBRIS',
  'TCA: +4h 23m | Miss: 1.2 km | Pc: 2.3×10⁻⁴',
  '{"pc": 0.00023, "miss_km": 1.2}'::jsonb,
  'Monitor and prepare for potential action',
  NOW() + INTERVAL '1 hour'
);

# Insert another test alert
INSERT INTO alerts (
  alert_id,
  created_at,
  category,
  severity,
  status,
  primary_norad_id,
  title,
  message,
  metrics,
  recommended_action,
  escalation_deadline
) VALUES (
  'demo-alert-002',
  NOW(),
  'MANEUVER',
  'WARNING',
  'OPEN',
  '36564',
  'MANEUVER — GSAT-30',
  'Detected Δv: 0.12 m/s | Type: Station-keeping',
  '{"dv_ms": 0.12}'::jsonb,
  'Review maneuver catalog',
  NOW() + INTERVAL '2 hours'
);
```

### Method 2: Test SSE Endpoint Directly

```bash
# Test SSE connection
curl -N http://localhost:8080/api/alerts/stream
```

You should see heartbeats every 30 seconds:
```
data: {"type":"HEARTBEAT","payload":"ping"}
```

### Method 3: Test REST API

```bash
# List all alerts
curl http://localhost:8080/api/alerts

# List only critical alerts
curl "http://localhost:8080/api/alerts?severity=CRITICAL"

# Acknowledge an alert
curl -X POST http://localhost:8080/api/alerts/demo-alert-001/ack \
  -H "Content-Type: application/json" \
  -d '{"operator_id":"OPERATOR","action":"MONITOR","notes":"Demo acknowledgement"}'
```

### Method 4: Trigger Alerts via Engine (If Running Catalog)

If you have satellite data loaded, alerts will be generated automatically when:
- Conjunction risk exceeds threshold (Pc > 1e-4)
- Maneuver detected (Z-score > 3)
- Re-entry risk (perigee < 100km)

---

## UI Verification

### Check Alert Panel

1. Open `http://localhost:8000/globe.html`
2. Look at the right sidebar under "Active Alerts"
3. You should see the test alerts with:
   - Severity badge (CRITICAL in red, WARNING in amber)
   - Alert title
   - Alert metadata
   - Acknowledge button

### Check Toast Notifications

1. When a new alert is inserted into the database
2. Toast should appear in top-right corner
3. Click × to dismiss manually

### Check Sound

1. Insert an EMERGENCY severity alert
2. You should hear a continuous klaxon sound
3. Acknowledge the alert to stop the sound

### Check Globe Colour Coding

1. Satellites with alerts should be colour-coded:
   - INFO: Blue (#60a5fa)
   - WARNING: Amber (#facc15)
   - CRITICAL: Red (#f87171)
   - EMERGENCY: Bright red (#ff0000) with pulsing

---

## Troubleshooting

### Engine Won't Start

**Error:** "Failed to connect to PostgreSQL"
```bash
# Check database is running
docker ps | grep ssis_demo_db

# Check database is ready
docker exec -it ssis_demo_db pg_isready -U postgres -d ssis_db

# Check connection string
echo $SSIS_DB_URL
```

**Error:** "Failed to initialize alert evaluator"
```bash
# Check config file exists
ls -la configs/alert_rules.yaml

# Check config path
echo $SSIS_CONFIG_PATH
```

### SSE Connection Fails

**Error:** Browser console shows SSE connection error
```bash
# Check engine is serving on port 8080
curl http://localhost:8080/health

# Check if SSE endpoint is accessible
curl -N http://localhost:8080/api/alerts/stream
```

### Alerts Not Appearing in UI

**Check:** Alerts in database
```bash
docker exec -it ssis_demo_db psql -U postgres -d ssis_db -c "SELECT * FROM alerts;"
```

**Check:** SSE connection in browser console
- Open browser DevTools (F12)
- Go to Network tab
- Look for alerts/stream connection
- Check for errors

**Check:** JavaScript console for errors
- Open browser DevTools (F12)
- Go to Console tab
- Look for JavaScript errors

---

## Demo Script (Automated)

Save this as `demo.sh`:

```bash
#!/bin/bash

echo "=== SSIS Alerting Demo ==="

# 1. Start database
echo "Starting PostgreSQL..."
docker run -d --name ssis_demo_db \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=ssis_db \
  timescale/timescaledb-ha:pg16

# Wait for database
echo "Waiting for database..."
sleep 10

# 2. Run migrations
echo "Running migrations..."
docker exec -i ssis_demo_db psql -U postgres -d ssis_db < init-db/01_setup.sql
docker exec -i ssis_demo_db psql -U postgres -d ssis_db < init-db/02_alerts.sql

# 3. Set environment
export SSIS_DB_URL="postgres://postgres:changeme@localhost:5432/ssis_db?sslmode=disable"
export SSIS_CONFIG_PATH="configs"
export SSIS_MODEL_DIR="pkg/ml/models"

# 4. Build engine
echo "Building engine..."
CGO_ENABLED=1 go build -tags onnx -o ssis_engine ./engine/main.go

# 5. Start engine in background
echo "Starting engine..."
./ssis_engine --provider=CPU &
ENGINE_PID=$!

# 6. Start HTTP server for UI
echo "Starting UI server..."
cd pkg/hud
python3 -m http.server 8000 &
UI_PID=$!

# 7. Insert test alerts
echo "Inserting test alerts..."
sleep 5
docker exec -i ssis_demo_db psql -U postgres -d ssis_db << 'EOF'
INSERT INTO alerts (alert_id, created_at, category, severity, status, primary_norad_id, secondary_norad_id, title, message, metrics, recommended_action, escalation_deadline)
VALUES ('demo-alert-001', NOW(), 'CONJUNCTION', 'CRITICAL', 'OPEN', '25544', '12345', 'CONJUNCTION — ISS / DEBRIS', 'TCA: +4h 23m | Miss: 1.2 km | Pc: 2.3×10⁻⁴', '{"pc": 0.00023, "miss_km": 1.2}'::jsonb, 'Monitor and prepare for potential action', NOW() + INTERVAL '1 hour');
EOF

echo ""
echo "=== Demo Ready ==="
echo "Open browser to: http://localhost:8000/globe.html"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $ENGINE_PID $UI_PID; docker rm -f ssis_demo_db" EXIT
wait
```

Run it:
```bash
chmod +x demo.sh
./demo.sh
```

---

## Cleanup

```bash
# Stop engine
pkill ssis_engine

# Stop UI server
pkill -f "python3 -m http.server"

# Remove database container
docker rm -f ssis_demo_db

# Remove binary
rm ssis_engine
```

---

## Performance Notes

- **Binary size:** 36MB (statically linked with ONNX Runtime)
- **Memory usage:** ~200MB (includes ONNX models)
- **Database:** ~50MB initial (grows with alerts)
- **Network:** Localhost only (no external calls)

---

## Demo Checklist

Before presenting the demo:

- [ ] Database running and migrations applied
- [ ] Engine starts without errors
- [ ] Alert evaluator started (check logs)
- [ ] SSE endpoint accessible (curl test)
- [ ] UI loads in browser
- [ ] Test alerts visible in UI
- [ ] Toast notifications appear
- [ ] Sound plays for EMERGENCY alerts
- [ ] Globe satellites colour-coded
- [ ] Acknowledge button works
- [ ] REST API returns alerts
