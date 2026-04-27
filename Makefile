SHELL := /bin/bash

GO ?= go
PKG := ./engine
OUT_DIR := build
BIN := $(OUT_DIR)/ssis_engine
DIST_DIR := dist

GOFLAGS ?= -trimpath
LDFLAGS ?= -s -w -buildid=
CGO_ENABLED ?= 0

.PHONY: all fmt test-focused test-all build build-onnx build-static sbom package clean

all: fmt test-focused build-onnx

fmt:
	$(GO) fmt ./...

test-focused:
	$(GO) test ./pkg/spatial ./pkg/catalog

test-all:
	$(GO) test ./...

verify-pc:
	@echo "Running Gobra formal verification on pkg/collision..."
	gobra -i pkg/collision/collision.go pkg/verification/pc_formal.go

# Build with real ONNX inference (requires libonnxruntime + CGO)
# Without -tags onnx, all models compile as MockInferrer stubs
build-onnx:
	mkdir -p $(OUT_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -tags onnx -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG)

build:
	mkdir -p $(OUT_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -tags onnx -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG)

build-static:
	mkdir -p $(OUT_DIR)
	@if CGO_ENABLED=$(CGO_ENABLED) $(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG); then \
		echo "Built static-compatible binary with CGO_ENABLED=$(CGO_ENABLED)"; \
	else \
		echo "WARNING: static build unavailable (likely ONNX CGO requirement); falling back to reproducible CGO build"; \
		CGO_ENABLED=1 $(GO) build $(GOFLAGS) -tags onnx -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG); \
	fi

sbom:
	./scripts/generate_sbom.sh $(DIST_DIR)/sbom/ssis_go_modules.json

package: build-static sbom
	./scripts/package_delivery.sh $(DIST_DIR)

clean:
	rm -rf $(OUT_DIR) $(DIST_DIR)
