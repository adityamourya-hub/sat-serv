#!/usr/bin/env python3
"""
Validate ONNX models from model_registry.yaml

ADITI 4.0: CI script that loads each exported ONNX from the registry
and validates it using onnx.checker.check_model.

Usage:
    python scripts/validate_onnx_models.py --registry configs/model_registry.yaml
"""

import argparse
import json
import yaml
from pathlib import Path
import onnx
from onnx import checker, shape_inference


def validate_model(model_path: str) -> dict:
    """Validate a single ONNX model and return validation results"""
    result = {
        "model": model_path,
        "valid": False,
        "errors": [],
        "warnings": [],
    }

    try:
        # Load model
        model = onnx.load(model_path)

        # Check model
        checker.check_model(model)

        # Infer shapes
        model = shape_inference.infer_shapes(model)

        result["valid"] = True
        result["opset_version"] = model.opset_import[0].version
        result["producer_name"] = model.producer_name if model.producer_name else "unknown"
        result["graph_name"] = model.graph.name

    except Exception as e:
        result["errors"].append(str(e))

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX models from registry")
    parser.add_argument("--registry", required=True, help="Path to model_registry.yaml")
    parser.add_argument("--output", default="onnx_validation_report.json", help="Output JSON report")
    args = parser.parse_args()

    # Load registry
    registry_path = Path(args.registry)
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    models = registry["models"]
    results = []

    print(f"Validating {len(models)} models from registry...")

    for model in models:
        model_path = Path(model["onnx_path"])
        if not model_path.exists():
            result = {
                "model": model["name"],
                "valid": False,
                "errors": [f"Model file not found: {model_path}"],
            }
            results.append(result)
            print(f"❌ {model['name']}: File not found")
            continue

        result = validate_model(str(model_path))
        result["name"] = model["name"]
        result["version"] = model["version"]
        results.append(result)

        if result["valid"]:
            print(f"✅ {model['name']}: Valid (opset {result['opset_version']})")
        else:
            print(f"❌ {model['name']}: Invalid - {', '.join(result['errors'])}")

    # Generate summary
    valid_count = sum(1 for r in results if r["valid"])
    print(f"\nSummary: {valid_count}/{len(results)} models valid")

    # Write report
    report = {
        "total_models": len(models),
        "valid_models": valid_count,
        "invalid_models": len(models) - valid_count,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report written to {args.output}")

    # Exit with error if any models are invalid
    if valid_count < len(models):
        exit(1)


if __name__ == "__main__":
    main()
