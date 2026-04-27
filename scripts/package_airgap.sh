#!/bin/bash
# ============================================================================
# SSIS Air-Gapped Deployment Package Builder
# Produces a single tarball that can be transferred to an offline IAF node
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-v1.0.0}"
DIST_DIR="dist/ssis-${VERSION}-airgap"
TARBALL="dist/ssis-${VERSION}-airgap.tar.gz"

echo "═══════════════════════════════════════════════════"
echo "  SSIS Air-Gapped Package Builder (${VERSION})"
echo "═══════════════════════════════════════════════════"

# Clean
rm -rf "$DIST_DIR" "$TARBALL"
mkdir -p "$DIST_DIR"

# 1. Docker image
echo "[1/6] Building Docker image..."
docker build -f deploy/Dockerfile -t ssis-engine:${VERSION} .

echo "[2/6] Exporting Docker image..."
docker save ssis-engine:${VERSION} | gzip > "${DIST_DIR}/ssis-engine-${VERSION}.tar.gz"

# 2. Model weights (backup)
echo "[3/6] Copying model weights..."
mkdir -p "${DIST_DIR}/models"
cp pkg/ml/models/*.onnx "${DIST_DIR}/models/" 2>/dev/null || true
cp pkg/ml/models/*.pt   "${DIST_DIR}/models/" 2>/dev/null || true

# 3. Configuration
echo "[4/6] Copying configuration..."
mkdir -p "${DIST_DIR}/configs"
cp configs/config.prod.yaml    "${DIST_DIR}/configs/"
cp configs/model_registry.yaml "${DIST_DIR}/configs/"
cp configs/pax_config.json     "${DIST_DIR}/configs/" 2>/dev/null || true
cp configs/sensors.yaml        "${DIST_DIR}/configs/" 2>/dev/null || true
cp configs/threat_library.yaml "${DIST_DIR}/configs/" 2>/dev/null || true

# 4. Database schema
echo "[5/6] Copying database schema..."
mkdir -p "${DIST_DIR}/database"
cp deploy/database/*.sql "${DIST_DIR}/database/" 2>/dev/null || true
cp docker-compose.yml "${DIST_DIR}/"

# 5. Documentation
cp docs/EXECUTIVE_SUMMARY.md     "${DIST_DIR}/"
cp docs/VALIDATION_EVIDENCE.md   "${DIST_DIR}/"

# 6. Deployment instructions
cat > "${DIST_DIR}/DEPLOY.md" << 'DEPLOY_EOF'
# SSIS Air-Gapped Deployment Guide

## Prerequisites
- Docker Engine 24+ (or Podman)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space

## Quick Start

```bash
# 1. Load Docker image
docker load < ssis-engine-*.tar.gz

# 2. Generate signing key (ONCE, then store securely)
openssl rand -hex 32 > /etc/ssis/signing.key

# 3. Run the engine
docker run -d \
    --name ssis-engine \
    -e SSIS_SIGNING_KEY=$(cat /etc/ssis/signing.key) \
    -p 50051:50051 \
    -p 8080:8080 \
    ssis-engine:v1.0.0

# 4. Verify
curl http://localhost:8080/health
```

## With TimescaleDB (Full Stack)
```bash
# Copy docker-compose.yml to deployment node
docker-compose up -d
```

## Security Notes
- Generate unique Ed25519 keys for each deployment node
- Enable mTLS by mounting certs to /app/certs/
- Set `SSIS_AUDIT_LEVEL=VERBOSE` for operational logging
DEPLOY_EOF

# Package
echo "[6/6] Creating tarball..."
cd dist
tar czf "ssis-${VERSION}-airgap.tar.gz" "ssis-${VERSION}-airgap/"
cd ..

SIZE=$(du -sh "$TARBALL" | cut -f1)
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Package ready: $TARBALL ($SIZE)"
echo "  Contents:"
echo "    - Docker image (ssis-engine:${VERSION})"
echo "    - Model weights (ONNX + TorchScript)"
echo "    - Configuration files"
echo "    - Database schema"
echo "    - Deployment instructions"
echo "═══════════════════════════════════════════════════"
