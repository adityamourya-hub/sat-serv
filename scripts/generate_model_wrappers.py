#!/usr/bin/env python3
"""
Generate Go model wrappers from model_registry.yaml

: Code generation script that reads model_registry.yaml
and generates Go struct wrappers for ONNX inference.

Usage:
 python scripts/generate_model_wrappers.py --registry configs/model_registry.yaml --output pkg/ml/models/generated
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any


def go_type_from_dtype(dtype: str) -> str:
 """Convert ONNX dtype to Go type"""
 dtype_map = {
 "float32": "float32",
 "float16": "float16",
 "int8": "int8",
 "int32": "int32",
 "bool": "bool",
 }
 return dtype_map.get(dtype, "interface{}")


def go_shape_to_comment(shape: List[int]) -> str:
 """Convert shape list to Go comment"""
 return f"[{', '.join(map(str, shape))}]"


def generate_model_struct(model: Dict[str, Any]) -> str:
 """Generate Go struct for a single model"""
 name = model["go_wrapper"]
 onnx_path = model["onnx_path"]
 version = model["version"]
 opset = model["opset"]
 quantization = model["quantization"]
 tier = model["tier"]
 description = model["description"]

 # Generate input fields
 input_fields = []
 for inp in model["input_schema"]:
 input_fields.append(f" // {inp['name']}: {go_shape_to_comment(inp['shape'])} {inp['dtype']}")
 input_fields.append(f" {inp['name']} []float32")

 # Generate output fields
 output_fields = []
 for out in model["output_schema"]:
 output_fields.append(f" // {out['name']}: {go_shape_to_comment(out['shape'])} {out['dtype']}")
 output_fields.append(f" {out['name']} []float32")

 code = f'''// {name} wraps the {model['name']} ONNX model
// {description}
// Version: {version}
// ONNX Path: {onnx_path}
// Opset: {opset}
// Quantization: {quantization}
// PAX Tier: {tier}
type {name} struct {{
 // Model metadata
 Name string
 Version string
 ONNXPath string
 Opset int
 Quantization string
 Tier int
 Description string

 // Input tensors
{chr(10).join(input_fields)}

 // Output tensors
{chr(10).join(output_fields)}

 // ONNX session (loaded at runtime)
 session interface{{}} // *ort.AdvancedSession[float32]
}}

// New{name} creates a new model wrapper with default values
func New{name}() *{name} {{
 return &{name}{{
 Name: "{model['name']}",
 Version: "{version}",
 ONNXPath: "{onnx_path}",
 Opset: {opset},
 Quantization: "{quantization}",
 Tier: {tier},
 Description: "{description}",
 }}
}}

// GetInputShapes returns the expected input tensor shapes
func (m *{name}) GetInputShapes() map[string][]int {{
 return map[string][]int{{
{chr(10).join([f' "{inp["name"]}": {inp["shape"]},' for inp in model["input_schema"]])}
 }}
}}

// GetOutputShapes returns the expected output tensor shapes
func (m *{name}) GetOutputShapes() map[string][]int {{
 return map[string][]int{{
{chr(10).join([f' "{out["name"]}": {out["shape"]},' for out in model["output_schema"]])}
 }}
}}
'''
 return code


def generate_registry_loader(models: List[Dict[str, Any]]) -> str:
 """Generate Go code for loading all models from registry"""
 model_names = [m["go_wrapper"] for m in models]
 
 code = f'''// ModelRegistry holds all registered ML models
type ModelRegistry struct {{
 models map[string]interface{{}}
 mu sync.RWMutex
}}

// NewModelRegistry creates a new model registry
func NewModelRegistry() *ModelRegistry {{
 registry := &ModelRegistry{{
 models: make(map[string]interface{{}}),
 }}
 
 // Initialize all models
{chr(10).join([f' registry.Register("{m["name"]}", New{m["go_wrapper"]}())' for m in models])}
 
 return registry
}}

// Register adds a model to the registry
func (r *ModelRegistry) Register(name string, model interface{{}}) {{
 r.mu.Lock()
 defer r.mu.Unlock()
 r.models[name] = model
}}

// Get retrieves a model by name
func (r *ModelRegistry) Get(name string) (interface{{}}, bool) {{
 r.mu.RLock()
 defer r.mu.RUnlock()
 model, ok := r.models[name]
 return model, ok
}}

// GetByTier returns all models for a specific PAX tier
func (r *ModelRegistry) GetByTier(tier int) []interface{{}} {{
 r.mu.RLock()
 defer r.mu.RUnlock()
 
 var result []interface{{}}
 for _, model := range r.models {{
 if m, ok := model.(interface{{ GetTier() int }}); ok {{
 if m.GetTier() == tier {{
 result = append(result, model)
 }}
 }}
 }}
 return result
}}

// GetTier returns the PAX tier for a model
func (m *{model_names[0]}) GetTier() int {{
 return m.Tier
}}
'''
 return code


def generate_go_file(models: List[Dict[str, Any]], settings: Dict[str, Any]) -> str:
 """Generate complete Go file with all model wrappers"""
 
 header = f'''// Code generated from configs/model_registry.yaml
// DO NOT EDIT - regenerate with: python scripts/generate_model_wrappers.py
//
// : Auto-generated ONNX model wrappers
// Package: {settings.get("go_package", "ssis/pkg/ml/models")}

package models

import (
 "sync"
)

'''

 # Generate all model structs
 model_structs = "\n".join([generate_model_struct(m) for m in models])
 
 # Generate registry loader
 registry_loader = generate_registry_loader(models)
 
 return header + model_structs + "\n" + registry_loader


def main():
 parser = argparse.ArgumentParser(description="Generate Go model wrappers from YAML registry")
 parser.add_argument("--registry", required=True, help="Path to model_registry.yaml")
 parser.add_argument("--output", required=True, help="Output directory for generated Go code")
 args = parser.parse_args()

 # Load registry
 registry_path = Path(args.registry)
 with open(registry_path) as f:
 registry = yaml.safe_load(f)

 models = registry["models"]
 settings = registry.get("settings", {})

 # Generate Go code
 go_code = generate_go_file(models, settings)

 # Write to output file
 output_dir = Path(args.output)
 output_dir.mkdir(parents=True, exist_ok=True)
 
 output_file = output_dir / "models_generated.go"
 with open(output_file, "w") as f:
 f.write(go_code)

 print(f"Generated Go wrappers for {len(models)} models")
 print(f"Output: {output_file}")


if __name__ == "__main__":
 main()
