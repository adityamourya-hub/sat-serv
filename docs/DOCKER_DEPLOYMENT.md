# SSIS Air-Gapped Docker Deployment Guide

**Classification:** SOVEREIGN_INTEL / AIR_GAPPED_ONLY  
**Authority:** IAF SSA Directorate  
**Version:** 1.0  
**Date:** 2026-04-28

---

## Overview

This guide describes how to deploy the SSIS Sovereign Space Intelligence Substrate with the integrated alerting system in an air-gapped environment using Docker Compose.

---

## Prerequisites

### Hardware Requirements
- **CPU:** 8 cores minimum (16 cores recommended)
- **RAM:** 32GB minimum (64GB recommended)
- **Storage:** 100GB SSD minimum
- **Network:** Air-gapped (no external internet access required)

### Software Requirements
- Docker 24.0+
- Docker Compose 2.20+

---

## Architecture

The deployment consists of four containers:

1. **db** - PostgreSQL 16 with TimescaleDB and PostGIS extensions
   - Auto-runs database migrations from `init-db/` directory
   - Includes alert system schema (02_alerts.sql)
   - Persistent data volume

2. **plasma** - Apache Arrow Plasma Store (2GB shared memory)
   - Zero-copy data bridge for analytics

3. **engine** - SSIS Sovereign Engine with Alerting System
   - gRPC server (port 50051)
   - REST API + SSE (port 8080)
   - Alert evaluator goroutine
   - SSE hub for real-time push

4. **analytics** - Python analytics workspace (optional)
   - Jupyter/MPDAF/JAX support
   - Connected to Plasma store

---

## Deployment Steps

### 1. Prepare Configuration Files

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values (use strong passwords in production)
nano .env
```

**Required Environment Variables:**
- `POSTGRES_PASSWORD` - Database password
- `SSIS_DB_URL` - Database connection string (auto-generated from above)
- `SSIS_CONFIG_PATH` - Path to alert rules config (default: /app/configs)
- `SSIS_MODEL_DIR` - Path to ONNX models (default: /app/pkg/ml/models)

### 2. Verify Database Migrations

Ensure the following files exist in `init-db/`:
- `01_setup.sql` - Base schema (catalog, current_states, orbital_trends)
- `02_alerts.sql` - Alerting system schema (alerts, alert_rules, audit_log, sensor_status)

These will auto-run on first container startup.

### 3. Verify Alert Rules Configuration

Ensure `configs/alert_rules.yaml` exists with your alerting rules. The container will mount this at `/app/configs/alert_rules.yaml`.

### 4. Build and Start Containers

```bash
# Build the engine image
docker-compose build engine

# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

Expected output:
```
NAME              STATUS              PORTS
ssis_datalake     Up (healthy)        0.0.0.0:5432->5432/tcp
plasma_bridge     Up                  0.0.0.0:8081->8081/tcp
ssis_engine       Up (healthy)        0.0.0.0:50051->50051/tcp, 0.0.0.0:8080->8080/tcp
ssis_analytics    Up
```

### 5. Verify Alerting System

```bash
# Check engine logs for alert evaluator startup
docker-compose logs engine | grep "Alert evaluator"

# Test health endpoint
curl http://localhost:8080/health

# Test SSE endpoint (should connect and send heartbeats)
curl -N http://localhost:8080/api/alerts/stream
```

---

## Air-Gapped Considerations

### No External Dependencies

The deployment is fully air-gapped:
- **Database:** Local PostgreSQL container
- **Models:** Copied from `pkg/ml/models/` directory
- **Config:** Mounted from `configs/` directory
- **No CDN:** All assets bundled in container image

### Image Transfer for Air-Gapped Deployment

If deploying to an air-gapped system:

```bash
# On build machine:
docker-compose build engine
docker save ssis_engine:latest -o ssis_engine.tar
docker save timescale/timescaledb-ha:pg16 -o timescaledb.tar
docker save apache/arrow-dev:latest-python-3.11 -o arrow.tar
docker save python:3.11-slim -o python.tar

# Transfer tar files to air-gapped system (via secure media)
scp *.tar airgapped-host:/tmp/

# On air-gapped system:
docker load -i /tmp/ssis_engine.tar
docker load -i /tmp/timescaledb.tar
docker load -i /tmp/arrow.tar
docker load -i /tmp/python.tar
docker-compose up -d
```

---

## Alerting System Integration

### Database Schema

The alerting system uses the following tables (created by `02_alerts.sql`):
- `alerts` - Enriched alerts with severity, category, metrics
- `alert_rules` - Rule definitions and metadata
- `audit_log` - Audit trail with Ed25519 signatures
- `sensor_status` - Sensor health monitoring

### Configuration

The alert rules are loaded from `configs/alert_rules.yaml`:
- Hot-reload on SIGHUP signal
- YAML format with thresholds and escalation policies
- Categories: conjunction, maneuver, reentry, blind_spot, sensor_health, anomaly, intrusion, collision

### SSE Endpoint

Real-time alerts are pushed via Server-Sent Events:
- **Endpoint:** `GET /api/alerts/stream`
- **Format:** JSON messages with type field (ALERT, HEARTBEAT)
- **Role-based filtering:** Operator vs Commander visibility

### REST Endpoints

- `GET /api/alerts` - Query alerts with pagination and filtering
- `POST /api/alerts/:id/ack` - Acknowledge alert with audit logging

---

## Monitoring

### Container Health Checks

```bash
# Check all container health
docker-compose ps

# View engine logs
docker-compose logs -f engine

# View database logs
docker-compose logs -f db
```

### Database Metrics

```bash
# Connect to database
docker exec -it ssis_datalake psql -U aditya_admin -d ssis_db

# Check alert counts
SELECT severity, COUNT(*) FROM alerts GROUP BY severity;

# Check recent alerts
SELECT * FROM alerts ORDER BY created_at DESC LIMIT 10;

# Check audit trail
SELECT * FROM audit_log ORDER BY logged_at DESC LIMIT 10;
```

---

## Troubleshooting

### Engine Fails to Start

```bash
# Check database connection
docker exec -it ssis_engine env | grep SSIS_DB_URL

# Check if database is healthy
docker exec -it ssis_datalake pg_isready -U aditya_admin -d ssis_db

# View engine startup logs
docker-compose logs engine
```

### Alert Evaluator Not Starting

```bash
# Check if alert rules file exists
docker exec -it ssis_engine ls -la /app/configs/alert_rules.yaml

# Check evaluator logs
docker-compose logs engine | grep -i alert

# Verify database schema
docker exec -it ssis_datalake psql -U aditya_admin -d ssis_db -c "\dt"
```

### SSE Connection Issues

```bash
# Test SSE endpoint manually
curl -N -H "Accept: text/event-stream" http://localhost:8080/api/alerts/stream

# Check if SSE hub is initialized
docker-compose logs engine | grep "SSE"
```

---

## Security Considerations

### Air-Gapped Validation

- **No external DNS:** Verify no `net.Lookup*` calls
- **No external APIs:** Verify no HTTP calls to external services
- **No CDN:** Verify all assets bundled in image
- **mTLS:** Enable gRPC mTLS for production (set SSIS_GRPC_CERT/KEY)

### Database Security

- **Strong Passwords:** Use secure passwords in .env
- **SSL Mode:** Set to `require` or `verify-full` in production
- **Network Isolation:** Database only accessible from engine container

### Ed25519 Signatures

All alerts and audit entries are signed with Ed25519:
- **Key Management:** Store signing key securely (SSIS_SIGNING_KEY env var)
- **Signature Verification:** Verify signatures on all received alerts
- **Audit Trail:** All actions logged with signatures

---

## Scaling Considerations

### Database Scaling

TimescaleDB hypertables automatically partition by time:
- **Alerts:** 1-year retention with continuous aggregates
- **Audit Log:** 5-year retention for compliance

### SSE Fan-Out

The SSE hub supports multiple concurrent clients:
- **Default:** 100 concurrent connections
- **Scaling:** Deploy multiple engine instances with load balancer for higher concurrency

---

## Backup and Recovery

### Database Backup

```bash
# Backup entire database
docker exec -it ssis_datalake pg_dump -U aditya_admin ssis_db > backup.sql

# Restore from backup
docker exec -i ssis_datalake psql -U aditya_admin ssis_db < backup.sql
```

### Volume Backup

```bash
# Backup PostgreSQL data volume
docker run --rm -v ssis_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

---

## Maintenance

### Rule Hot-Reload

To reload alert rules without restarting the engine:

```bash
# Send SIGHUP to engine container
docker-compose kill -s HUP engine

# Verify reload in logs
docker-compose logs engine | grep -i reload
```

### Container Updates

```bash
# Pull new image
docker-compose pull engine

# Recreate container with new image
docker-compose up -d engine

# Verify health
docker-compose ps
```

---

## Certification Checklist

Before production deployment in air-gapped environment:

- [ ] Database migrations run successfully (01_setup.sql, 02_alerts.sql)
- [ ] Alert rules configuration loaded without errors
- [ ] SSE endpoint connects and sends heartbeats
- [ ] REST API endpoints respond correctly
- [ ] No external network calls detected
- [ ] Ed25519 signatures generated for all alerts
- [ ] Audit trail entries created for all actions
- [ ] Health checks pass for all services
- [ ] Backup procedures tested
- [ ] Security hardening completed (mTLS, strong passwords)
