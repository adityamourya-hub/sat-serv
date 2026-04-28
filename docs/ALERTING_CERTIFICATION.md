# SSIS Sovereign Alerting System — Certification Checklist

**Classification:** SOVEREIGN_INTEL / AIR_GAPPED_ONLY  
**Authority:** IAF SSA Directorate  
**Version:** 1.0  
**Date:** 2026-04-28

---

## 1. Security Model Validation

### 1.1 Air-Gapped Operation
- [ ] Verify no external network calls in any component
- [ ] Check DNS lookups are disabled (no `net.Dial`, `net.Lookup*`)
- [ ] Confirm no external API dependencies (CDN, cloud services)
- [ ] Validate all libraries are Go stdlib or audited Go packages

### 1.2 Cryptographic Integrity
- [ ] Verify Ed25519 signatures on every alert
- [ ] Verify Ed25519 signatures on every audit log entry
- [ ] Confirm signature verification logic in gRPC ingestion
- [ ] Validate signer key management (SSIS_SIGNING_KEY env var)

### 1.3 Authentication & Authorization
- [ ] Verify mTLS on gRPC (certificate validation)
- [ ] Verify role-based filtering on SSE (Operator vs Commander)
- [ ] Confirm JWT claims extraction (when auth is implemented)
- [ ] Validate no anonymous access to critical endpoints

### 1.4 Data Sovereignty
- [ ] Verify all data stored in local PostgreSQL/TimescaleDB
- [ ] Confirm no data leaves air-gapped environment
- [ ] Validate audit trail is tamper-evident
- [ ] Verify export functionality for certification review

---

## 2. Functional Requirements

### 2.1 Alert Generation
- [ ] Verify conjunction alerts are generated (Pc threshold)
- [ ] Verify maneuver alerts are generated (Z-score threshold)
- [ ] Verify re-entry alerts are generated (perigee altitude)
- [ ] Verify blind spot alerts are generated (coverage gap duration)
- [ ] Verify sensor health alerts are generated (degradation threshold)
- [ ] Verify behavioural anomaly alerts are generated (pattern detection)

### 2.2 Alert Enrichment
- [ ] Verify severity assignment based on YAML rules
- [ ] Verify recommended action generation
- [ ] Verify SHAP attribution inclusion
- [ ] Verify KL divergence metric inclusion
- [ ] Verify geospatial coordinates inclusion
- [ ] Verify sovereign volume association

### 2.3 Alert Persistence
- [ ] Verify alerts stored in `alerts` hypertable
- [ ] Verify audit log entries in `audit_log` hypertable
- [ ] Verify rule version tracking
- [ ] Verify signature storage
- [ ] Verify escalation deadline calculation

### 2.4 Alert Distribution
- [ ] Verify SSE connection establishment
- [ ] Verify SSE message format compliance
- [ ] Verify role-based filtering (Operator sees INFO+WARNING+CRITICAL)
- [ ] Verify Commander sees all severities including EMERGENCY
- [ ] Verify heartbeat/ping mechanism
- [ ] Verify automatic reconnection on disconnect

### 2.5 Alert Acknowledgement
- [ ] Verify POST /api/alerts/:id/ack functionality
- [ ] Verify status update to ACKNOWLEDGED
- [ ] Verify audit log entry for acknowledgement
- [ ] Verify operator ID capture
- [ ] Verify action capture
- [ ] Verify notes capture
- [ ] Verify emergency sound stops on acknowledgement

---

## 3. Performance Requirements

### 3.1 Latency
- [ ] Verify alert generation < 100ms from event detection
- [ ] Verify SSE push < 50ms from enrichment
- [ ] Verify database insert < 10ms per alert
- [ ] Verify audit log insert < 5ms per entry

### 3.2 Throughput
- [ ] Verify 100 alerts/sec sustained rate
- [ ] Verify 1000 alerts/sec burst handling
- [ ] Verify SSE fan-out to 100 concurrent clients
- [ ] Verify database query performance with 1M alerts

### 3.3 Scalability
- [ ] Verify hypertable partitioning (time-based)
- [ ] Verify continuous aggregates for dashboard queries
- [ ] Verify retention policies (alerts: 1 year, audit: 5 years)
- [ ] Verify index optimization (status, severity, category, spatial)

---

## 4. Reliability Requirements

### 4.1 Fault Tolerance
- [ ] Verify graceful degradation if PostgreSQL unavailable
- [ ] Verify graceful degradation if SSE hub unavailable
- [ ] Verify rule hot-reload (SIGHUP) without restart
- [ ] Verify evaluator restart on crash

### 4.2 Data Integrity
- [ ] Verify no alert loss during SSE disconnect
- [ ] Verify no alert loss during database downtime
- [ ] Verify no duplicate alerts (deduplication window)
- [ ] Verify audit trail completeness

### 4.3 Error Handling
- [ ] Verify proper error logging for all failures
- [ ] Verify error messages do not leak sensitive data
- [ ] Verify circuit breaker for repeated failures
- [ ] Verify exponential backoff for reconnection

---

## 5. User Interface Requirements

### 5.1 Alert Panel
- [ ] Verify alert list displays correctly
- [ ] Verify sortable columns (severity, time, category)
- [ ] Verify filterable by severity/category/status
- [ ] Verify colour-coded rows
- [ ] Verify acknowledge button functionality
- [ ] Verify scroll performance with 100+ alerts

### 5.2 Toast Notifications
- [ ] Verify toast appears on new alert
- [ ] Verify toast colour matches severity
- [ ] Verify toast auto-dismiss (30s for non-emergency)
- [ ] Verify toast manual close
- [ ] Verify toast positioning (top-right)

### 5.3 Sound Player
- [ ] Verify INFO: no sound
- [ ] Verify WARNING: single chime (800Hz sine)
- [ ] Verify CRITICAL: repeating alarm (3× 600Hz square)
- [ ] Verify EMERGENCY: continuous klaxon (500Hz sawtooth)
- [ ] Verify emergency sound stops on acknowledgement
- [ ] Verify Web Audio API compatibility

### 5.4 Globe Colour Coding
- [ ] Verify satellite colour matches alert severity
- [ ] Verify INFO: blue (#60a5fa)
- [ ] Verify WARNING: amber (#facc15)
- [ ] Verify CRITICAL: red (#f87171)
- [ ] Verify EMERGENCY: red (#ff0000) with pulsing
- [ ] Verify colour resets on acknowledgement

---

## 6. Audit Trail Requirements

### 6.1 Completeness
- [ ] Verify every alert has audit entry (ALERT_CREATED)
- [ ] Verify every acknowledgement has audit entry (ALERT_ACKNOWLEDGED)
- [ ] Verify every rule update has audit entry (RULE_UPDATED)
- [ ] Verify every escalation has audit entry (ALERT_ESCALATED)

### 6.2 Integrity
- [ ] Verify every audit entry is Ed25519 signed
- [ ] Verify signature includes: timestamp, actor, action, entity_id, details
- [ ] Verify signature verification capability
- [ ] Verify tamper detection (signature mismatch)

### 6.3 Export
- [ ] Verify JSON export functionality
- [ ] Verify PDF export functionality (when implemented)
- [ ] Verify date range filtering
- [ ] Verify severity filtering
- [ ] Verify operator filtering

---

## 7. Configuration Management

### 7.1 Rule Configuration
- [ ] Verify YAML syntax validation
- [ ] Verify rule enable/disable functionality
- [ ] Verify threshold modification
- [ ] Verify escalation timeout modification
- [ ] Verify hot-reload without restart

### 7.2 System Configuration
- [ ] Verify SSIS_DB_URL environment variable
- [ ] Verify SSIS_CONFIG_PATH environment variable
- [ ] Verify SSIS_SIGNING_KEY environment variable
- [ ] Verify default values for missing configuration

### 7.3 Sensor Configuration
- [ ] Verify sensor registry loading
- [ ] Verify sensor status monitoring
- [ ] Verify sensor health alert generation

---

## 8. Compliance Requirements

### 8.1 IAF Certification
- [ ] Verify all security requirements met
- [ ] Verify all functional requirements met
- [ ] Verify audit trail export for review
- [ ] Verify documentation completeness
- [ ] Verify source code availability for audit

### 8.2 DRDO Certification
- [ ] Verify ML model explainability (SHAP)
- [ ] Verify physics model validation
- [ ] Verify uncertainty quantification
- [ ] Verify conformal UQ implementation

### 8.3 Sovereignty Requirements
- [ ] Verify no foreign dependencies
- [ ] Verify no cloud services
- [ ] Verify no telemetry leakage
- [ ] Verify data at rest encryption (when implemented)
- [ ] Verify data in transit encryption (mTLS, TLS)

---

## 9. Testing Requirements

### 9.1 Unit Tests
- [ ] Verify rules engine unit tests
- [ ] Verify repository unit tests
- [ ] Verify SSE hub unit tests
- [ ] Verify evaluator unit tests

### 9.2 Integration Tests
- [ ] Verify end-to-end alert flow (engine → evaluator → SSE → dashboard)
- [ ] Verify PostgreSQL integration
- [ ] Verify SSE integration
- [ ] Verify REST API integration

### 9.3 Performance Tests
- [ ] Verify load test (100 alerts/sec)
- [ ] Verify stress test (1000 alerts/sec burst)
- [ ] Verify database query performance
- [ ] Verify SSE fan-out performance

### 9.4 Security Tests
- [ ] Verify penetration testing (no SQL injection)
- [ ] Verify authentication bypass testing
- [ ] Verify authorization bypass testing
- [ ] Verify signature validation testing

---

## 10. Documentation Requirements

### 10.1 Architecture Documentation
- [ ] Verify architecture diagram completeness
- [ ] Verify component description completeness
- [ ] Verify data flow documentation
- [ ] Verify security model documentation

### 10.2 API Documentation
- [ ] Verify REST API endpoint documentation
- [ ] Verify SSE message format documentation
- [ ] Verify request/response examples
- [ ] Verify error code documentation

### 10.3 Operator Documentation
- [ ] Verify user guide for dashboard
- [ ] Verify alert severity guide
- [ ] Verify acknowledgment procedure guide
- [ ] Verify escalation procedure guide

---

## 11. Deployment Requirements

### 11.1 Database Setup
- [ ] Verify init-db/02_alerts.sql execution
- [ ] Verify TimescaleDB extension enabled
- [ ] Verify PostGIS extension enabled
- [ ] Verify hypertable creation
- [ ] Verify index creation
- [ ] Verify retention policy setup

### 11.2 Configuration Setup
- [ ] Verify configs/alert_rules.yaml deployment
- [ ] Verify environment variable setup
- [ ] Verify SSL certificate deployment (mTLS)
- [ ] Verify signing key deployment

### 11.3 Binary Deployment
- [ ] Verify single binary compilation
- [ ] Verify static linking (no CGO dependencies)
- [ ] Verify cross-compilation (if needed)
- [ ] Verify systemd service configuration

---

## 12. Maintenance Requirements

### 12.1 Monitoring
- [ ] Verify alert generation rate monitoring
- [ ] Verify SSE connection monitoring
- [ ] Verify database performance monitoring
- [ ] Verify error rate monitoring

### 12.2 Log Management
- [ ] Verify log rotation
- [ ] Verify log retention (90 days minimum)
- [ ] Verify log aggregation (when implemented)
- [ ] Verify log analysis capability

### 12.3 Backup & Recovery
- [ ] Verify PostgreSQL backup procedure
- [ ] Verify backup retention (30 days minimum)
- [ ] Verify restore procedure
- [ ] Verify disaster recovery procedure

---

## Certification Sign-Off

| Section | Reviewer | Date | Status | Notes |
|---------|----------|------|--------|-------|
| Security Model | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Functional Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Performance Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Reliability Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| User Interface Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Audit Trail Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Configuration Management | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Compliance Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Testing Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Documentation Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Deployment Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |
| Maintenance Requirements | ___________ | ___________ | ☐ Passed ☐ Failed | |

**Final Certification Decision:**
[ ] APPROVED — System meets all certification requirements
[ ] CONDITIONAL — System meets requirements with conditions
[ ] REJECTED — System does not meet requirements

**Authority:** _________________________  
**Date:** _________________________  
**Signature:** _________________________
