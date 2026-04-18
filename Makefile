SHELL := /bin/bash

GO ?= go
PKG := ./cmd/ssa_ops_runner
OUT_DIR := build
BIN := $(OUT_DIR)/ssa_ops_runner
DIST_DIR := dist

GOFLAGS ?= -trimpath
LDFLAGS ?= -s -w -buildid=
CGO_ENABLED ?= 0

.PHONY: all fmt test-focused test-all build build-static sbom package clean

all: fmt test-focused build-static

fmt:
	$(GO) fmt ./...

test-focused:
	$(GO) test ./pkg/ml/data ./pkg/ml/onnx ./pkg/spatial ./pkg/catalog ./engine/pkg/assistant

test-all:
	$(GO) test ./...

verify-pc:
	@echo "Running Gobra formal verification on pkg/collision..."
	gobra -i pkg/collision/collision.go pkg/verification/pc_formal.go

build:
	mkdir -p $(OUT_DIR)
	CGO_ENABLED=1 $(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG)

build-static:
	mkdir -p $(OUT_DIR)
	@if CGO_ENABLED=$(CGO_ENABLED) $(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG); then \
		echo "Built static-compatible binary with CGO_ENABLED=$(CGO_ENABLED)"; \
	else \
		echo "WARNING: static build unavailable (likely ONNX CGO requirement); falling back to reproducible CGO build"; \
		CGO_ENABLED=1 $(GO) build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BIN) $(PKG); \
	fi

sbom:
	./scripts/generate_sbom.sh $(DIST_DIR)/sbom/ssis_go_modules.json

package: build-static sbom
	./scripts/package_delivery.sh $(DIST_DIR)

clean:
	rm -rf $(OUT_DIR) $(DIST_DIR)
