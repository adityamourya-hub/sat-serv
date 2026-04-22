#!/bin/bash
# scripts/live_maneuver_demo.sh
# Master script for  Symbolic Attribution Demo

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "    - SYMBOLIC MANOEUVRE ATTRIBUTION LIVE PIPELINE     "
echo "================================================================"
echo ""

# 1. Clean up stale attribution files
rm -f attribution_*.tex maneuver_res_*.json mock_attribution.tex maneuver_attribution.tex

# 2. Build the Go demo executable
echo "[1/3] Building Go Detection Engine..."
go build -o bin/demo_symbolic cmd/demo_symbolic/main.go

# 3. Run the live demo
echo "[2/3] Running Live Detection + Symbolic Attribution..."
echo "      (This will trigger the Axiom-Zero Physics Core)"
echo ""
./bin/demo_symbolic

# 4. Display Results
echo ""
echo "[3/3] Final Strategic Intelligence Product (SITREP):"
if [ -f attribution_555.tex ]; then
    echo "----------------------------------------------------------------"
    cat attribution_555.tex
    echo "----------------------------------------------------------------"
    echo "✅ Success: Symbolic laws distilled and exported to LaTeX."
else
    echo "❌ Error: Symbolic attribution failed to generate LaTeX SITREP."
    exit 1
fi

echo ""
echo "Demo finished successfully."
