#!/bin/bash
# setup_mtls.sh: Generate Sovereign mTLS artifacts for  gRPC Hardening
# Usage: ./scripts/setup_mtls.sh [output_dir]

OUT_DIR=${1:-"./certs"}
mkdir -p "$OUT_DIR"

echo "Generating Sovereign Certificate Authority..."
openssl genrsa -out "$OUT_DIR/ca.key" 4096
openssl req -new -x509 -sha256 -key "$OUT_DIR/ca.key" -out "$OUT_DIR/ca.crt" -days 3650 \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=IAF/OU=SDA/CN=SSIS-Root-CA"

echo "Generating gRPC Server Certificate..."
openssl genrsa -out "$OUT_DIR/server.key" 2048
openssl req -new -key "$OUT_DIR/server.key" -out "$OUT_DIR/server.csr" \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=IAF/OU=SDA/CN=localhost"
openssl x509 -req -in "$OUT_DIR/server.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial \
    -out "$OUT_DIR/server.crt" -days 365 -sha256

echo "Generating Client Certificates (RBAC Tiers)..."
# Commander
openssl genrsa -out "$OUT_DIR/commander.key" 2048
openssl req -new -key "$OUT_DIR/commander.key" -out "$OUT_DIR/commander.csr" \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=IAF/OU=SDA/CN=iaf-commander"
openssl x509 -req -in "$OUT_DIR/commander.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial \
    -out "$OUT_DIR/commander.crt" -days 365 -sha256

# Operator
openssl genrsa -out "$OUT_DIR/operator.key" 2048
openssl req -new -key "$OUT_DIR/operator.key" -out "$OUT_DIR/operator.csr" \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=IAF/OU=SDA/CN=iaf-operator"
openssl x509 -req -in "$OUT_DIR/operator.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial \
    -out "$OUT_DIR/operator.crt" -days 365 -sha256

# Viewer
openssl genrsa -out "$OUT_DIR/viewer.key" 2048
openssl req -new -key "$OUT_DIR/viewer.key" -out "$OUT_DIR/viewer.csr" \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=IAF/OU=SDA/CN=iaf-viewer"
openssl x509 -req -in "$OUT_DIR/viewer.csr" -CA "$OUT_DIR/ca.crt" -CAkey "$OUT_DIR/ca.key" -CAcreateserial \
    -out "$OUT_DIR/viewer.crt" -days 365 -sha256

echo "Setting permissions..."
chmod 600 "$OUT_DIR"/*.key
chmod 644 "$OUT_DIR"/*.crt

echo "mTLS artifacts generated in $OUT_DIR"
