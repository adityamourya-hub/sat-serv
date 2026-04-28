# SSIS Sovereign Alerting System — Architecture Document

**Version:** 1.0  
**Classification:** SOVEREIGN_INTEL / AIR_GAPPED_ONLY  
**Authority:** IAF SSA Directorate  

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    SSIS ENGINE (Go Runtime)                       │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ UKF Workers  │  │ ONNX Pool    │  │ Analytics              │  │
│  │ (75 workers) │  │ (Axiom/Rapid)│  │ • ManeuverDetect       │  │
│  └──────┬───────┘  └──────┬───────┘  │ • BlindSpotDetect      │  │
│         │                 │          │ • IntentClassifier      │  │
│         ▼                 ▼          └──────────┬─────────────┘  │
│  ┌──────────────────────────────────────────────▼──────────────┐ │
│  │                  types.EventBus (in-memory pub/sub)          │ │
│  │                  Publish(alert *types.Alert)                 │ │
│  └──────┬────────────────────┬────────────────────┬────────────┘ │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │ gRPC Stream   │  │ Alert Evaluator  │  │ WebSocket Hub     │  │
│  │ (existing)    │  │ (rules engine)   │  │ (NEW: /ws/alerts) │  │
│  └──────────────┘  └────────┬─────────┘  └────────┬──────────┘  │
│                             │                     │              │
│                    ┌────────▼─────────┐           │              │
│                    │ PostgreSQL/       │           │              │
│                    │ TimescaleDB       │           │              │
│                    │ • alerts (hyper)  │           │              │
│                    │ • alert_rules     │           │              │
│                    │ • audit_log       │           │              │
│                    └──────────────────┘           │              │
└──────────────────────────────────────────────────┼──────────────┘
                                                   │
                    ┌──────────────────────────────▼──────────────┐
                    │         COMMAND CENTER DASHBOARD             │
                    │  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
                    │  │ Globe    │ │ Alert    │ │ Toast      │  │
                    │  │ (WebGL)  │ │ Panel    │ │ + Sound    │  │
                    │  │ colour-  │ │ queue    │ │ player     │  │
                    │  │ coded    │ │ sortable │ │            │  │
                    │  │ objects  │ │ filterable│ │            │  │
                    │  └──────────┘ └──────────┘ └────────────┘  │
                    └─────────────────────────────────────────────┘
```

## 2. Components

### 2.1 Alert Evaluator (`internal/alerts/evaluator.go`)
- Subscribes to `types.EventBus`
- Evaluates incoming engine events against configurable YAML rules
- Enriches alerts with severity, colour coding, recommended action
- Persists to PostgreSQL `alerts` table
- Publishes enriched alerts to WebSocket Hub

### 2.2 Rules Engine (`internal/alerts/rules.go`)
- Loads from `configs/alert_rules.yaml`
- Hot-reloadable (SIGHUP) without restart
- Rule types:
  - **Conjunction Risk**: Pc threshold + miss distance
  - **Maneuver Detection**: Z-score threshold per orbital element
  - **Re-entry Risk**: Perigee altitude + decay rate
  - **Blind Spot**: Coverage gap duration threshold
  - **Sensor Health**: Degradation / offline detection
  - **Behavioural Anomaly**: Mega-constellation + adversarial patterns

### 2.3 WebSocket Hub (`internal/alerts/ws_hub.go`)
- Pure Go implementation using `gorilla/websocket`
- Concurrent fan-out to all connected dashboard clients
- Heartbeat/ping to detect stale connections
- Role-based filtering (Operator sees INFO+WARNING, Commander sees all)
- Message format: JSON envelope `{type, severity, alert, timestamp}`

### 2.4 REST Endpoints (wired into existing `gin` router)
- `GET  /api/alerts` — paginated alert history (filterable by type, severity, time)
- `GET  /ws/alerts` — WebSocket upgrade for real-time push
- `POST /api/alerts/:id/ack` — acknowledge + log action
- `GET  /api/alerts/export` — JSON/PDF export for certification

### 2.5 Audit Logger (`internal/alerts/audit.go`)
- Every alert lifecycle event logged: created → acknowledged → resolved
- Operator identity, timestamp, action taken
- Exportable as JSON for IAF certification
- Tamper-evident: Ed25519 signed audit entries

## 3. Alert Severity Model

| Level | Colour | Sound | Auto-escalate | Example |
|-------|--------|-------|---------------|---------|
| INFO | `#60A5FA` (blue) | None | No | New object catalogued |
| WARNING | `#FACC15` (amber) | Single tone | After 30min | Maneuver detected (station-keeping) |
| CRITICAL | `#F87171` (red) | Repeating alarm | After 10min | Conjunction Pc > 1e-4 |
| EMERGENCY | `#FF0000` (red pulse) | Continuous klaxon | Immediate | Co-orbital threat / HGV re-entry |

## 4. Data Flow (Alert Lifecycle)

```
1. Engine detects event (conjunction/maneuver/anomaly)
2. Engine publishes types.Alert to EventBus
3. Alert Evaluator receives, matches against YAML rules
4. Evaluator enriches: severity, colour, recommended_action, SHAP explanation
5. Evaluator persists to PostgreSQL (alerts table)
6. Evaluator logs to audit_log table
7. Evaluator pushes to WebSocket Hub
8. Hub fans-out to all connected dashboard clients
9. Dashboard shows toast + sound + adds to alert queue
10. Operator clicks "Acknowledge" → POST /api/alerts/:id/ack
11. Ack logged to audit_log, alert status updated
```

## 5. Security Model

- **Air-gapped**: Zero external network calls. All processing local.
- **Signed alerts**: Ed25519 signature on every alert (existing `cryptography.Signer`)
- **Signed audit**: Every audit entry signed for tamper evidence
- **mTLS on gRPC**: Existing implementation preserved
- **Role-based WS**: JWT claims checked on WebSocket upgrade
- **No foreign cloud**: PostgreSQL + Go + vanilla JS. No npm/CDN.

## 6. Sovereign Constraints

- All libraries are Go stdlib or well-audited Go packages
- `gorilla/websocket` — pure Go, no CGO, audited by Go community
- `pgx/v5` — already in use for PostgreSQL
- No external API calls. No telemetry leakage. No DNS lookups.
- Exportable audit trail for DRDO/IAF certification review
